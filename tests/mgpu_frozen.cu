// mgpu_frozen -- the frozen-state multi-rank harness: everything of interFoam that is NOT the numerical step,
// driven on N virtual ranks (ProcComm loopback: one host thread per rank, one GPU) from a state somebody else
// made, and compared with what OpenFOAM does with the same state. See tools/mgpu_oracle/FROZEN.md.
//
//   mgpu_frozen globals <case> <time>              the hidden globals: deltaN, needReference, pRefPoint, the
//                                                  continuity error, the end-of-run statistics
//   mgpu_frozen fo      <case> <time> <out>        the function objects of <case>/system/controlDict, once, on the
//                                                  state at <time>: <out>/postProcessing
//   mgpu_frozen write   <case> <time> <out> [ascii|binary]
//                                                  the per-rank writer: <out>/processorN/<time>/ (the fields the
//                                                  solver would write, with their processor patch entries)
//   mgpu_frozen procvals <case> <time>             the processor patch values of alpha/U/p_rgh from the halo
//                                                  exchange and the processor patches' weights, against the stored
//                                                  ones OpenFOAM wrote
//
// <case> is a decomposed case: processor0..N-1/{constant/polyMesh, <time>/{alpha.<p>,U,p_rgh,p,phi,...}} and
// system/. N is the number of processor directories; a case with ONE processor directory (np=1) is the same code
// with the communicator absent (ProcComm::current() null), i.e. the single-GPU path.
#include "proc_comm.cuh"
#include "proc_globals.cuh"
#include "case_roots.cuh"
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "geometric_field.cuh"
#include "read_surface_field.cuh"
#include "mrf_read.cuh"
#include "inter_function_objects.cuh"
#include "inter_writer_cpp.cuh"
#include "inter_case_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace brae;

namespace {

std::mutex   outMutex;

std::string fmtG(double v)
{
    char b[64];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}
std::atomic<int> failures{0};

void say(const std::string& s)
{
    std::lock_guard<std::mutex> l(outMutex);
    std::printf("%s\n", s.c_str());
    std::fflush(stdout);
}

void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    say(std::string(ok ? "  ok:   " : "  FAIL: ") + what);
}

// ---------------------------------------------------------------------------------------------------------
// one rank's frozen state
// ---------------------------------------------------------------------------------------------------------

struct RankState
{
    int                  rank = 0, np = 1;
    std::string          caseDir, procDir, timeName, phase = "water", alphaName = "alpha.water";
    CaseRoots            roots;
    PrimitiveMesh        m;
    FvGeometry           g;
    std::vector<FvPatch> patches;
    std::map<std::string, std::vector<label>> zones;
    // cells
    std::vector<scalar>  alpha, prgh, p, ux, uy, uz;
    std::vector<vector>  U;
    // the file's patch values (the state's own, as OpenFOAM stored them), by patch index
    std::vector<std::vector<scalar>> alphaB, prghB, pB;
    std::vector<std::vector<vector>> UB;
    SurfaceScalarField   phi, alphaPhi0;
    // scalar uniform/time
    label                timeIndex = 0;
    scalar               deltaT = 0, deltaT0 = 0;

    int  nCells() const { return static_cast<int>(m.nCells()); }
};

template <typename T>
std::vector<std::vector<T>> patchValues(const FieldData<T>& fd, const std::vector<FvPatch>& patches)
{
    std::vector<std::vector<T>> out(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        out[pi].assign(static_cast<std::size_t>(patches[pi].size), T{});
        for (const auto& b : fd.boundary)
        {
            if (b.name != patches[pi].name) continue;
            if (b.hasValue)
            {
                if (b.valueUniform) std::fill(out[pi].begin(), out[pi].end(), b.uniformValue);
                else if (b.values.size() == out[pi].size()) out[pi] = b.values;
            }
            break;
        }
    }
    return out;
}

std::string procDirOf(const std::string& caseDir, int rank, int np)
{
    return np > 1 || fs::exists(caseDir + "/processor0") ? caseDir + "/processor" + std::to_string(rank) : caseDir;
}

int countRanks(const std::string& caseDir)
{
    const int n = countProcessorDirs(caseDir);
    return n > 0 ? n : 1;
}

void loadRank(RankState& s, const std::string& caseDir, int rank, int np, const std::string& timeName)
{
    s.rank = rank;
    s.np = np;
    s.caseDir = caseDir;
    s.timeName = timeName;
    s.procDir = procDirOf(caseDir, rank, np);
    s.roots = CaseRoots(caseDir);
    s.roots.procDir = s.procDir;
    s.roots.rank = rank;
    s.roots.nProcs = np;
    s.roots.parallel = fs::exists(caseDir + "/processor0");
    const std::string pm = s.procDir + "/constant/polyMesh";
    auto stage = [&](const char* what, auto&& f)
    {
        try { f(); }
        catch (const std::exception& e) { throw std::runtime_error(std::string(what) + " (" + s.procDir + "): " + e.what()); }
    };
    stage("mesh", [&]() { s.m.read(pm, pm, pm); s.g.build(s.m); s.patches = buildPatches(s.m, s.g, /*mirrorACMI=*/false); });
    stage("cellZones", [&]() { s.zones = readCellZones(pm); });
    const std::string td = s.procDir + "/" + timeName;
    FieldData<scalar> a, pr, pp;
    FieldData<vector> u;
    stage("alpha", [&]() { a = readField<scalar>(td + "/" + s.alphaName); });
    stage("p_rgh", [&]() { pr = readField<scalar>(td + "/p_rgh"); });
    stage("p", [&]() { pp = readField<scalar>(td + "/p"); });
    stage("U", [&]() { u = readField<vector>(td + "/U"); });
    s.alpha = a.internalField;
    s.prgh = pr.internalField;
    s.p = pp.internalField;
    s.U = u.internalField;
    s.alphaB = patchValues(a, s.patches);
    s.prghB = patchValues(pr, s.patches);
    s.pB = patchValues(pp, s.patches);
    s.UB = patchValues(u, s.patches);
    for (const vector& v : s.U)
    {
        s.ux.push_back(v.x);
        s.uy.push_back(v.y);
        s.uz.push_back(v.z);
    }
    stage("phi", [&]() { s.phi = readSurfaceField(td + "/phi", s.patches, s.m.nInternalFaces()); });
    if (fs::exists(td + "/alphaPhi0." + s.phase))
        stage("alphaPhi0", [&]() { s.alphaPhi0 = readSurfaceField(td + "/alphaPhi0." + s.phase, s.patches, s.m.nInternalFaces()); });
    const std::string ut = td + "/uniform/time";
    if (fs::exists(ut))
    {
        const FoamDict d = readDict(ut);
        s.timeIndex = static_cast<label>(d.scalarOr("index", 0));
        s.deltaT = d.scalarOr("deltaT", 0);
        s.deltaT0 = d.scalarOr("deltaT0", 0);
    }
}

// Run `body(rank, comm)` on np rank threads (np == 1: on this thread with NO communicator installed -- the
// single-GPU path). The first exception of any rank is reported; a rank's exception that is not a collective
// abort would hang the others, so every body is wrapped in comm->guarded.
template <class Body>
void onRanks(int np, Body body)
{
    if (np == 1)
    {
        ProcComm::setCurrent(nullptr);
        body(0);
        return;
    }
    std::vector<std::unique_ptr<ProcComm>> comms = ProcComm::createLoopback(np);
    std::vector<std::thread> th;
    std::vector<std::string> err(np);
    for (int r = 0; r < np; ++r)
    {
        th.emplace_back([&, r]()
        {
            comms[r]->bindThread();
            try
            {
                comms[r]->guarded([&]() { body(r); });
            }
            catch (const std::exception& e)
            {
                err[r] = e.what();
            }
        });
    }
    for (std::thread& t : th) t.join();
    ProcComm::setCurrent(nullptr);
    for (int r = 0; r < np; ++r)
    {
        if (!err[r].empty())
        {
            ++failures;
            say("  FAIL: rank " + std::to_string(r) + " threw: " + err[r]);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// globals
// ---------------------------------------------------------------------------------------------------------

// The coupled value on a processor face: w*own + (1-w)*neighbour with OpenFOAM's weights
// (processorFvPatch::makeWeights): nbrNf.(nbrCf - nbrCn) / (nf.delta + nbrNf.(nbrCf - nbrCn)),
// delta = (Cf - Cn) - (nbrCf - nbrCn), nbrNf = nbrSf/(|nbrSf| + VSMALL)  -- evaluated by the caller.

struct SerialRef
{
    scalar sumV = 0;
    label  nCells = 0;
};

int doGlobals(const std::string& caseDir, const std::string& timeName)
{
    const int np = countRanks(caseDir);
    say("globals on " + std::to_string(np) + " rank(s)");
    // the expected numbers from every rank's own data, summed here sequentially in rank order
    std::vector<RankState> st(np);
    for (int r = 0; r < np; ++r) loadRank(st[r], caseDir, r, np, timeName);
    double sumV = 0, n = 0;
    for (int r = 0; r < np; ++r)
    {
        double s = 0;
        for (scalar v : st[r].g.V()) s += v;
        sumV += s;
        n += static_cast<double>(st[r].g.V().size());
    }
    const scalar expectDeltaN = scalar(1e-8) / std::cbrt(static_cast<scalar>(sumV / n));
    say("expected deltaN (global average V over " + std::to_string((long)n) + " cells): " + fmtG(expectDeltaN));

    std::vector<scalar> dN(np);
    std::vector<int>    needRef(np), refCell(np);
    std::vector<scalar> refValue(np);
    std::vector<std::array<double, 5>> endStats(np);
    std::vector<scalar> contErr(np);
    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    onRanks(np, [&](int r)
    {
        RankState& s = st[r];
        dN[r] = cpu::interfaceProps::deltaN(s.g.V());
        GeometricField<scalar> prgh = buildField<scalar>(readField<scalar>(s.procDir + "/" + s.timeName + "/p_rgh"),
                                                         s.patches, s.m.nCells());
        const cpu::interFoam::InterFields::PressureReference ref = cpu::interFoam::readPressureReferenceOf(prgh, fvSolution, s.m, s.g);
        needRef[r] = ref.needReference;
        refCell[r] = ref.pRefCell;
        refValue[r] = ref.pRefValue;
        // end-of-run statistics from this rank's cells
        double maxU = 0, aMin = s.alpha.empty() ? 0 : s.alpha[0], aMax = aMin, mass = 0;
        for (std::size_t c = 0; c < s.alpha.size(); ++c)
        {
            maxU = std::fmax(maxU, std::sqrt(s.ux[c]*s.ux[c] + s.uy[c]*s.uy[c] + s.uz[c]*s.uz[c]));
            aMin = std::fmin(aMin, s.alpha[c]);
            aMax = std::fmax(aMax, s.alpha[c]);
            mass += s.alpha[c]*s.g.V()[c];
        }
        double worst = 0;
        pg::reduceEndStats(maxU, worst, aMin, aMax, mass);
        endStats[r] = {maxU, worst, aMin, aMax, mass};
    });
    bool same = true;
    for (int r = 1; r < np; ++r) same = same && dN[r] == dN[0];
    check("deltaN is the same on every rank", same);
    check("deltaN equals 1e-8/cbrt(global average V)", dN[0] == expectDeltaN || (np == 1));
    say("  deltaN " + fmtG(dN[0]) + (np > 1 ? "  (rank-0 local would be " + fmtG(cpu::interfaceProps::deltaN(st[0].g.V())) + ")" : ""));
    int nHolders = 0;
    for (int r = 0; r < np; ++r) nHolders += refCell[r] >= 0;
    bool needSame = true;
    for (int r = 1; r < np; ++r) needSame = needSame && needRef[r] == needRef[0];
    check("needReference agrees on every rank", needSame);
    say("  needReference " + std::to_string(needRef[0]) + ", pRefValue " + fmtG(refValue[0]) + ", reference cell held by "
        + std::to_string(nHolders) + " rank(s)");
    if (needRef[0]) check("the reference cell lives on exactly one rank", nHolders == 1);
    for (int r = 0; r < np; ++r)
        if (refCell[r] >= 0) say("  rank " + std::to_string(r) + " holds reference cell " + std::to_string(refCell[r]));
    bool statsSame = true;
    for (int r = 1; r < np; ++r) statsSame = statsSame && endStats[r] == endStats[0];
    check("the end-of-run statistics are the same on every rank", statsSame);
    say("  maxU " + fmtG(endStats[0][0]) + " alpha [" + fmtG(endStats[0][2]) + ", " + fmtG(endStats[0][3]) + "] mass "
        + fmtG(endStats[0][4]));
    return failures ? 1 : 0;
}


// ---------------------------------------------------------------------------------------------------------
// function objects
// ---------------------------------------------------------------------------------------------------------

void copyDir(const fs::path& from, const fs::path& to)
{
    fs::create_directories(to);
    fs::copy(from, to, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
}

int doFo(const std::string& caseDir, const std::string& timeName, const std::string& out, int timeIndexOverride)
{
    const int np = countRanks(caseDir);
    say("function objects on " + std::to_string(np) + " rank(s), state at " + timeName + ", output " + out + "/postProcessing");
    fs::remove_all(out);
    copyDir(caseDir + "/system", out + "/system");
    // BRAE_FO_CONTROLDICT: the controlDict to take instead of <case>/system/controlDict (frozen_state.sh keeps
    // controlDict.run, the one the OpenFOAM runs themselves used)
    if (const char* alt = std::getenv("BRAE_FO_CONTROLDICT"))
        fs::copy_file(std::string(alt), out + "/system/controlDict", fs::copy_options::overwrite_existing);
    std::vector<RankState> st(np);
    for (int r = 0; r < np; ++r) loadRank(st[r], caseDir, r, np, timeName);
    onRanks(np, [&](int r)
    {
        RankState& s = st[r];
        fo::Source src;
        src.newStep = []() {};
        src.alpha = [&s]() -> const std::vector<scalar>& { return s.alpha; };
        src.p_rgh = [&s]() -> const std::vector<scalar>& { return s.prgh; };
        src.Ux = [&s]() -> const std::vector<scalar>& { return s.ux; };
        src.Uy = [&s]() -> const std::vector<scalar>& { return s.uy; };
        src.Uz = [&s]() -> const std::vector<scalar>& { return s.uz; };
        src.alphaPatch = [&s](std::size_t pi) -> const std::vector<scalar>& { return s.alphaB[pi]; };
        std::shared_ptr<fo::FunctionObjectList> list = fo::createFunctionObjects(
            out, out + "/0", s.m, s.g, s.patches, s.zones, s.alphaName, false, src, r == 0);
        if (!list) throw std::runtime_error("the case has no function object brae runs");
        fo::Step step;
        step.timeIndex = timeIndexOverride >= 0 ? timeIndexOverride : s.timeIndex;
        step.time = static_cast<scalar>(std::strtod(s.timeName.c_str(), nullptr));
        step.deltaT = s.deltaT;
        step.sinceStart = step.time;
        step.runTimeWrite = false;
        const bool stop = list->execute(step);
        if (r == 0) say(std::string("  execute done; runTimeControl asks to end: ") + (stop ? "yes" : "no"));
    });
    return failures ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------
// the writer
// ---------------------------------------------------------------------------------------------------------

// processorFvPatch geometry, as OpenFOAM builds it (processorFvPatch.C makeWeights / delta; the neighbour's Cf, Sf
// and face-cell centres exchanged once): the weights of every processor face of every rank.
struct ProcGeom
{
    // per rank, per patch index: the weights w (own side)
    std::vector<std::vector<std::vector<scalar>>> w;
};

ProcGeom processorWeights(const std::vector<RankState>& st)
{
    ProcGeom pg_;
    pg_.w.resize(st.size());
    for (std::size_t r = 0; r < st.size(); ++r)
    {
        const RankState& s = st[r];
        pg_.w[r].resize(s.patches.size());
        for (std::size_t pi = 0; pi < s.patches.size(); ++pi)
        {
            const FvPatch& p = s.patches[pi];
            if (p.type != "processor" || p.size == 0) continue;
            const RankState& n = st[static_cast<std::size_t>(p.neighbProcNo)];
            // the neighbour's patch to me: the processor patch of rank n whose neighbProcNo is me
            const FvPatch* q = nullptr;
            for (const FvPatch& c : n.patches)
                if (c.type == "processor" && c.neighbProcNo == static_cast<label>(r)) q = &c;
            if (!q || q->size != p.size) throw std::runtime_error("processor patches of ranks do not pair");
            pg_.w[r][pi].resize(static_cast<std::size_t>(p.size));
            for (label i = 0; i < p.size; ++i)
            {
                const std::size_t k = static_cast<std::size_t>(i);
                const vector Cn = s.g.C()[static_cast<std::size_t>(p.faceCells[k])];
                const vector nbrCn = n.g.C()[static_cast<std::size_t>(q->faceCells[k])];
                const vector nbrCf = q->Cf[k];
                const vector nbrSf = n.g.Sf()[static_cast<std::size_t>(q->start + i)];
                const scalar nbrMag = std::sqrt(dot(nbrSf, nbrSf));
                const scalar den = nbrMag + scalar(1e-300);   // VSMALL is 1e-300; a division per component, as the field operator does
                const vector nbrNf{nbrSf.x/den, nbrSf.y/den, nbrSf.z/den};
                const vector delta = (p.Cf[k] - Cn) - (nbrCf - nbrCn);
                const scalar nbrDn = dot(nbrNf, nbrCf - nbrCn);
                pg_.w[r][pi][k] = nbrDn/(dot(p.nf[k], delta) + nbrDn);
            }
        }
    }
    return pg_;
}

void writeSystem(const std::string& caseDir, const std::string& out, const std::string& format)
{
    fs::remove_all(out);
    copyDir(caseDir + "/system", out + "/system");
    if (const char* alt = std::getenv("BRAE_FO_CONTROLDICT"))
        fs::copy_file(std::string(alt), out + "/system/controlDict", fs::copy_options::overwrite_existing);
    if (!format.empty())
    {
        const std::string cd = out + "/system/controlDict";
        std::ifstream in(cd);
        std::stringstream ss;
        ss << in.rdbuf();
        std::string t = ss.str();
        const std::size_t a = t.find("writeFormat");
        if (a != std::string::npos)
        {
            const std::size_t e = t.find(';', a);
            t.replace(a, e - a + 1, "writeFormat " + format + ";");
        }
        std::ofstream o(cd);
        o << t;
    }
}

int doWrite(const std::string& caseDir, const std::string& timeName, const std::string& out, const std::string& format,
            int nWrites, int purge)
{
    const int np = countRanks(caseDir);
    say("per-rank writer on " + std::to_string(np) + " rank(s), state " + timeName + " -> " + out + "/processorN/<time>");
    writeSystem(caseDir, out, format);
    if (purge > 0)
    {
        const std::string cd = out + "/system/controlDict";
        std::ifstream in(cd);
        std::stringstream ss;
        ss << in.rdbuf();
        std::string t = ss.str();
        const std::size_t a = t.find("purgeWrite");
        if (a != std::string::npos)
        {
            const std::size_t e = t.find(';', a);
            t.replace(a, e - a + 1, "purgeWrite " + std::to_string(purge) + ";");
        }
        std::ofstream o(cd);
        o << t;
    }
    std::vector<RankState> st(np);
    for (int r = 0; r < np; ++r) loadRank(st[r], caseDir, r, np, timeName);
    for (int r = 0; r < np; ++r) copyDir(st[r].procDir + "/constant", out + "/processor" + std::to_string(r) + "/constant");
    onRanks(np, [&](int r)
    {
        RankState& s = st[r];
        CaseRoots roots = s.roots;
        roots.dictDir = out;
        roots.procDir = np > 1 || fs::exists(caseDir + "/processor0") ? out + "/processor" + std::to_string(r) : out;
        const std::string startDir = s.procDir + "/" + s.timeName;
        cpu::interFoam::InterWriter writer(roots, startDir, s.patches, s.phase);
        // Time's own two step sizes as the state had them (uniform/time: deltaT0 then deltaT)
        writer.stepTaken(s.deltaT0);
        writer.stepTaken(s.deltaT);
        GeometricField<scalar> alpha = buildField<scalar>(readField<scalar>(startDir + "/" + s.alphaName), s.patches, s.m.nCells());
        GeometricField<vector> U = buildField<vector>(readField<vector>(startDir + "/U"), s.patches, s.m.nCells());
        GeometricField<scalar> prgh = buildField<scalar>(readField<scalar>(startDir + "/p_rgh"), s.patches, s.m.nCells());
        // THE SOLVER'S STATE: the patch objects hold the values the last step left on them. buildField only built them
        // from the file; the frozen state's patch values are the file's, put back here (processor faces included)
        auto holdValues = [&](auto& field, const auto& stored)
        {
            for (std::size_t pi = 0; pi < s.patches.size(); ++pi)
                if (s.patches[pi].size > 0 && stored[pi].size() == static_cast<std::size_t>(s.patches[pi].size))
                    field.boundary[pi]->setStoredValues(stored[pi]);
        };
        holdValues(alpha, s.alphaB);
        holdValues(U, s.UB);
        holdValues(prgh, s.prghB);
        cpu::interFoam::InterWriteState ws;
        ws.alpha1 = &alpha;
        ws.U = &U;
        ws.p_rgh = &prgh;
        ws.p = &s.p;
        ws.pBoundary = &s.pB;
        ws.phi = &s.phi;
        ws.alphaPhi0 = &s.alphaPhi0;
        ws.deltaT = s.deltaT;
        ws.timeIndex = s.timeIndex;
        for (int k = 0; k < nWrites; ++k)
        {
            ws.time = static_cast<scalar>(std::strtod(s.timeName.c_str(), nullptr)) + 1e-6*k;
            ws.timeIndex = s.timeIndex + k;
            writer.write(ws);
            writer.finish();
        }
    });
    return failures ? 1 : 0;
}


// ---------------------------------------------------------------------------------------------------------
// processor patch values from the halo
// ---------------------------------------------------------------------------------------------------------

// A scalar cell field's halo through ProcComm (device exchange on the rank's stream), then w*own + (1-w)*nbr on every
// processor face. `plan` is built once per rank; the exchange is COLLECTIVE.
struct HaloEval
{
    std::unique_ptr<HaloPlan> plan;
    std::vector<int>          procPatch;   // patch index of every processor patch, in plan order
    int                       nC = 0;
    cudaStream_t              stream{};
    double*                   d = nullptr;

    void init(const RankState& s)
    {
        ProcComm* comm = ProcComm::current();
        nC = s.nCells();
        std::vector<ProcessorPatchDesc> descs;
        for (std::size_t pi = 0; pi < s.patches.size(); ++pi)
        {
            const FvPatch& p = s.patches[pi];
            if (p.type != "processor" || p.size == 0) continue;
            ProcessorPatchDesc d_;
            d_.nbrRank = static_cast<int>(p.neighbProcNo);
            for (label c : p.faceCells) d_.faceCells.push_back(static_cast<int>(c));
            descs.push_back(std::move(d_));
            procPatch.push_back(static_cast<int>(pi));
        }
        plan = comm->buildPlanFromPatches(nC, std::move(descs), 4);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaMalloc(&d, sizeof(double)*static_cast<std::size_t>(plan->nExt() + 1));
    }

    // the cell field extended by its halo (collective)
    std::vector<double> extend(const std::vector<scalar>& cells)
    {
        std::vector<double> h(static_cast<std::size_t>(plan->nExt()), 0.0);
        std::copy(cells.begin(), cells.end(), h.begin());
        cudaMemcpyAsync(d, h.data(), sizeof(double)*h.size(), cudaMemcpyHostToDevice, stream);
        ProcComm::current()->exchangeCells(*plan, d, stream);
        cudaMemcpyAsync(h.data(), d, sizeof(double)*h.size(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return h;
    }

    void done()
    {
        ProcComm::current()->barrier();   // loopback: no rank may free staging a peer still reads
        plan.reset();
        cudaFree(d);
        cudaStreamDestroy(stream);
    }
};

int doProcVals(const std::string& caseDir, const std::string& timeName)
{
    const int np = countRanks(caseDir);
    say("processor patch values on " + std::to_string(np) + " rank(s), state " + timeName);
    if (np == 1) { say("  np 1: no processor patches"); return 0; }
    std::vector<RankState> st(np);
    for (int r = 0; r < np; ++r) loadRank(st[r], caseDir, r, np, timeName);
    const ProcGeom geom = processorWeights(st);
    struct Row { double maxDiff = 0, maxRef = 0; long n = 0, bitEq = 0; double ownDiff = 0, nbrDiff = 0; };
    std::vector<std::map<std::string, Row>> rows(np);
    onRanks(np, [&](int r)
    {
        RankState& s = st[r];
        HaloEval he;
        he.init(s);
        struct F { const char* name; const std::vector<scalar>* cells; const std::vector<std::vector<scalar>>* stored; };
        std::vector<std::vector<scalar>> uxB(s.patches.size()), uyB(s.patches.size()), uzB(s.patches.size());
        for (std::size_t pi = 0; pi < s.patches.size(); ++pi)
            for (const vector& v : s.UB[pi]) { uxB[pi].push_back(v.x); uyB[pi].push_back(v.y); uzB[pi].push_back(v.z); }
        const F fields[] = {{"alpha", &s.alpha, &s.alphaB}, {"p_rgh", &s.prgh, &s.prghB}, {"p", &s.p, &s.pB},
                            {"Ux", &s.ux, &uxB}, {"Uy", &s.uy, &uyB}, {"Uz", &s.uz, &uzB}};
        for (const F& f : fields)
        {
            const std::vector<double> ext = he.extend(*f.cells);
            Row& row = rows[r][f.name];
            int face = 0;
            for (int pi : he.procPatch)
            {
                const FvPatch& p = s.patches[static_cast<std::size_t>(pi)];
                for (label i = 0; i < p.size; ++i, ++face)
                {
                    const std::size_t k = static_cast<std::size_t>(i);
                    const double own = (*f.cells)[static_cast<std::size_t>(p.faceCells[k])];
                    const double nbr = ext[static_cast<std::size_t>(he.nC + he.plan->desc().faceToSlot[static_cast<std::size_t>(face)])];
                    const double w = geom.w[r][static_cast<std::size_t>(pi)][k];
                    const double v = w*own + (1.0 - w)*nbr;
                    const double ref = (*f.stored)[static_cast<std::size_t>(pi)][k];
                    row.maxDiff = std::fmax(row.maxDiff, std::fabs(v - ref));
                    row.maxRef = std::fmax(row.maxRef, std::fabs(ref));
                    row.ownDiff = std::fmax(row.ownDiff, std::fabs(own - ref));
                    row.nbrDiff = std::fmax(row.nbrDiff, std::fabs(nbr - ref));
                    ++row.n;
                    row.bitEq += (v == ref);
                }
            }
        }
        he.done();
    });
    for (const char* name : {"alpha", "p_rgh", "p", "Ux", "Uy", "Uz"})
    {
        Row t;
        for (int r = 0; r < np; ++r)
        {
            const Row& x = rows[r][name];
            t.maxDiff = std::fmax(t.maxDiff, x.maxDiff);
            t.maxRef = std::fmax(t.maxRef, x.maxRef);
            t.ownDiff = std::fmax(t.ownDiff, x.ownDiff);
            t.nbrDiff = std::fmax(t.nbrDiff, x.nbrDiff);
            t.n += x.n;
            t.bitEq += x.bitEq;
        }
        char b[256];
        std::snprintf(b, sizeof b, "  %-6s %ld processor faces: w*own+(1-w)*nbr vs stored: max|diff| %.3e (max|stored| %.3e), %ld bit-equal; "
                      "own cell vs stored %.3e, neighbour cell vs stored %.3e",
                      name, t.n, t.maxDiff, t.maxRef, t.bitEq, t.ownDiff, t.nbrDiff);
        say(b);
        check(std::string(name) + " processor face values from the halo and the weights match the stored ones to 1e-9 of max",
              t.maxDiff <= 1e-9*std::max(t.maxRef, 1e-30));
    }
    return failures ? 1 : 0;
}

}   // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc < 4)
        {
            std::fprintf(stderr, "usage: mgpu_frozen globals|fo|write|procvals <case> <time> [out] [format]\n");
            return 2;
        }
        const std::string mode = argv[1], caseDir = argv[2], timeName = argv[3];
        if (mode == "globals") return doGlobals(caseDir, timeName);
        if (mode == "procvals") return doProcVals(caseDir, timeName);
        if (mode == "fo" && argc >= 5) return doFo(caseDir, timeName, argv[4], argc >= 6 ? std::atoi(argv[5]) : -1);
        if (mode == "write" && argc >= 5)
            return doWrite(caseDir, timeName, argv[4], argc >= 6 ? argv[5] : "", argc >= 7 ? std::atoi(argv[6]) : 1,
                           argc >= 8 ? std::atoi(argv[7]) : 0);
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "mgpu_frozen: %s\n", e.what());
        return 1;
    }
}
