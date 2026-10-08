// brae's interFoam across a ROTATING cyclicAMI against REAL OpenFOAM's, on RAS/mixerVesselAMI.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_ami_vs_openfoam.sh stages it and carries the measurements. The rotor's cellZone turns
// under solidBodyMotionSolver, sliding its side of the AMI pair past the stator's; every step moves the
// points, recomputes the AMI and couples the two sides through its weights. THE CONTROL is OpenFOAM's own
// answer at the same instant with the rotor held still (omega 0), the AMI then a fixed, near one-to-one
// map.
//
// THE DEVICE LOOP MUST REFUSE the case, naming the patch.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "cyclic_acmi_cpp.cuh"
#include "cyclic_ami_cpp.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include "inter_solve_log.cuh"
#include "patch_entry_lookup.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

// MEASURED, 100 fixed steps of 2e-4 with every solve pinned (bounds about 30x): alpha 4.4e-13, p_rgh
// 5.4e-14, U 3.5e-12, k 9.5e-14, epsilon 1.2e-13, nut 2.1e-12; the flux across the pair 1.3e-11 of its
// largest; initial residuals 3.2e-13 (p_rgh), 1.4e-13 (U) and 6.7e-16 (k, epsilon) in step one and
// 4.0e-10, 8.6e-11 and 8.3e-14 over the run.
const scalar B_ALPHA = 1e-11;
const scalar B_PRGH = 2e-12;
const scalar B_U = 1e-10;
const scalar B_K = 3e-12;
const scalar B_EPSILON = 3e-12;
const scalar B_NUT = 6e-11;
const scalar B_PHI = 4e-10;

namespace {
int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

struct Diff
{
    scalar linf = 0;
    scalar refMax = 0;
    scalar rel() const { return linf/std::fmax(refMax, scalar(1e-300)); }
};

Diff compare(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.linf = std::fmax(d.linf, std::fabs(a[i] - b[i]));
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    return d;
}

Diff compare(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const vector e{a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z};
        d.linf = std::fmax(d.linf, mag(e));
        d.refMax = std::fmax(d.refMax, mag(b[i]));
    }
    return d;
}

std::vector<vector> readPoints(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    static const std::regex head(R"(\n\s*([0-9]+)\s*\n?\()");
    std::smatch mh;
    std::vector<vector> out;
    if (!std::regex_search(text, mh, head)) return out;
    const std::size_t n = static_cast<std::size_t>(std::atol(mh[1].str().c_str()));
    const char* p = text.c_str() + mh.position(0) + mh.length(0);
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        while (*p && *p != '(')
        {
            ++p;
        }
        if (!*p) break;
        ++p;
        char* end = nullptr;
        vector v;
        v.x = std::strtod(p, &end);
        p = end;
        v.y = std::strtod(p, &end);
        p = end;
        v.z = std::strtod(p, &end);
        p = end;
        out.push_back(v);
    }
    return out;
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/mixerVesselAMI (rotating cyclicAMI) ==\n");
    if (argc < 7)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <stillOfTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string stillDir = argv[6];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> patches = buildPatches(m, g, /*mirrorACMI=*/true);
    // for the device loop, which must refuse: the geometry and patches before any coupling
    const FvGeometry gRaw = g;
    const std::vector<FvPatch> uncoupled = patches;
    cpu::cyclicACMI::Interfaces acmi = cpu::cyclicACMI::setup(m, g, patches, scalar(0));
    attachCyclicCoupling(patches, m, g);
    cpu::cyclicAMIFvPatch::Interfaces ami = cpu::cyclicAMIFvPatch::setup(caseDir + "/constant/polyMesh", m, g, patches);
    const label nC = m.nCells();

    MutableMesh mm;
    mm.m = &m;
    mm.g = &g;
    mm.patches = &patches;
    mm.acmi = &acmi;
    mm.ami = &ami;
    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/false, &fin,
                                     /*endTime=*/scalar(1e30), /*pressureTaps=*/nullptr, &mm);
    check("brae ran the same number of steps", r.steps == nSteps);
    std::printf("  brae's time after %d steps: %.17g\n", (int)r.steps, (double)r.time);
    const bool turbulent = fin.turbulence.on;
    std::printf("  turbulence: %s\n", turbulent ? "kEpsilon" : "laminar");
    if (turbulent)
    {
        check("...and the closure is kEpsilon", fin.turbulence.model == InterRasModel::KEpsilon);
    }
    check("the mesh moves", fin.dynamicMesh != nullptr);

    // THE PATH: one AMI pair, coupled in every field, its faces meeting more than one face each
    check("the mesh has ONE cyclicAMI pair", ami.pairs().size() == 1);
    if (ami.pairs().size() != 1)
    {
        return 1;
    }
    const cpu::cyclicAMIFvPatch::Pair& pair = ami.pairs()[0];
    std::size_t nFieldsCoupled = 0;
    std::size_t nFields = 0;
    for (const label pi : {pair.src, pair.tgt})
    {
        const std::size_t k = static_cast<std::size_t>(pi);
        nFieldsCoupled += fin.alpha1.boundary[k]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.U.boundary[k]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.p_rgh.boundary[k]->coupled() ? 1 : 0;
        nFields += 3;
        if (turbulent)
        {
            nFieldsCoupled += fin.turbulence.k.boundary[k]->coupled() ? 1 : 0;
            nFieldsCoupled += fin.turbulence.epsilon.boundary[k]->coupled() ? 1 : 0;
            nFieldsCoupled += fin.turbulence.nut.boundary[k]->coupled() ? 1 : 0;
            nFields += 3;
        }
    }
    check("...coupled in every field, on both sides", nFieldsCoupled == nFields);
    std::size_t multi = 0;
    for (const auto& a : pair.weights.srcAddress)
    {
        multi += a.size() > 1 ? 1 : 0;
    }
    std::printf("  at the end %zu of %zu source faces meet more than one target face\n", multi,
                pair.weights.srcAddress.size());
    check("...and at the end the AMI is not one-to-one", multi > pair.weights.srcAddress.size()/2);

    // THE MOTION: brae's moved points against the ones OpenFOAM wrote
    {
        const std::vector<vector> ofPts = readPoints(ofDir + "/polyMesh/points");
        check("OpenFOAM wrote the moved points", ofPts.size() == m.points().size());
        scalar dPts = 0;
        for (std::size_t i = 0; i < ofPts.size() && i < m.points().size(); ++i)
        {
            dPts = std::fmax(dPts, mag(m.points()[i] - ofPts[i]));
        }
        std::printf("  points: Linf %.4e\n", (double)dPts);
        check("every point is where OpenFOAM moved it", dPts < scalar(1e-15));
    }

    auto readCells = [&](const std::string& path)
    {
        const FieldData<scalar> fd = readField<scalar>(path);
        std::vector<scalar> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };
    auto readVectorCells = [&](const std::string& path)
    {
        const FieldData<vector> fd = readField<vector>(path);
        std::vector<vector> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };

    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofA = brae::gatecheck::readOfSolves(logPath, fin.alphaName);
    std::printf("  OpenFOAM logged %zu p_rgh solves, brae ran %zu\n", ofP.size(), r.pSolves.size());
    check("it ran as many p_rgh solves as OpenFOAM logged", !ofP.empty() && r.pSolves.size() == ofP.size());
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(1e-11), scalar(1e-8),
                                               scalar(-1), nullptr, true, fin.pSolveFinal.tol);
    // the case runs explicit MULES (MULESCorr no), which logs no alpha solve at all
    check("explicit MULES: no alpha solve logged by either code", !fin.alphaCtl.MULESCorr && ofA.empty()
          && r.alphaSolves.empty());
    {
        // correctPhi: once at the start (initCorrectPhi.H) and after every mesh move. Its matrix is the
        // laplacian across the AMI as it stands, so its counts see the AMI's geometry directly.
        const std::vector<LinearSolveRecord> ofPc = brae::gatecheck::readOfSolves(logPath, "pcorr");
        std::size_t same = 0;
        for (std::size_t q = 0; q < ofPc.size() && q < r.pcorrSolves.size(); ++q)
        {
            same += (ofPc[q].nIterations == r.pcorrSolves[q].nIterations) ? 1 : 0;
        }
        std::printf("  pcorr: OpenFOAM logged %zu solves, brae ran %zu; %zu counts equal\n", ofPc.size(),
                    r.pcorrSolves.size(), same);
        check("every pcorr solve took OpenFOAM's iteration count, one a step and one at the start",
              ofPc.size() == static_cast<std::size_t>(nSteps + 1) && r.pcorrSolves.size() == ofPc.size()
              && same == ofPc.size());
    }
    for (int c = 0; c < 3; ++c)
    {
        static const char* names[3] = {"Ux", "Uy", "Uz"};
        const std::vector<LinearSolveRecord> ofU = brae::gatecheck::readOfSolves(logPath, names[c]);
        std::printf("  OpenFOAM logged %zu %s solves, brae ran %zu\n", ofU.size(), names[c], r.uSolves[c].size());
        check("...and as many momentum-predictor solves", !ofU.empty() && ofU.size() == r.uSolves[c].size());
        failures += brae::gatecheck::compareSolves("host", r.uSolves[c], ofU, nSteps, names[c],
                                                   scalar(1e-11), scalar(2e-9));
    }
    if (turbulent)
    {
        const std::vector<LinearSolveRecord> ofE = brae::gatecheck::readOfSolves(logPath, "epsilon");
        const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
        check("OpenFOAM's log gave epsilon and k solves", !ofE.empty() && ofK.size() == ofE.size());
        failures += brae::gatecheck::compareSolves("host", r.epsilonSolves, ofE, nSteps, "epsilon",
                                                   scalar(1e-12), scalar(3e-12));
        failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                                   scalar(1e-12), scalar(3e-12));
    }

    // BEFORE any fmax: std::fmax drops a NaN -- see tests/device_gate_finite.cuh
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    check("OpenFOAM's fields have one value per cell",
          ofAlpha.size() == static_cast<std::size_t>(nC) && ofPrgh.size() == ofAlpha.size() && ofU.size() == ofAlpha.size());
    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    auto worstAt = [&](const char* name, auto diffOf)
    {
        scalar worst = 0;
        label at = -1;
        for (label c = 0; c < nC; ++c)
        {
            const scalar d = diffOf(c);
            if (d > worst)
            {
                worst = d;
                at = c;
            }
        }
        if (at >= 0)
        {
            std::printf("  worst %-7s cell %d at (%.4g %.4g %.4g): %.4e\n", name, (int)at, (double)g.C()[at].x,
                        (double)g.C()[at].y, (double)g.C()[at].z, (double)worst);
        }
    };
    worstAt("alpha", [&](label c) { return std::fabs(fin.alpha1.internal[c] - ofAlpha[c]); });
    worstAt("p_rgh", [&](label c) { return std::fabs(fin.p_rgh.internal[c] - ofPrgh[c]); });
    worstAt("U", [&](label c)
    {
        const vector e{fin.U.internal[c].x - ofU[c].x, fin.U.internal[c].y - ofU[c].y, fin.U.internal[c].z - ofU[c].z};
        return mag(e);
    });
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B_ALPHA);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B_PRGH);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B_U);
    if (turbulent)
    {
        failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
        failures += brae::gatecheck::nonFinite("brae epsilon", fin.turbulence.epsilon.internal);
        failures += brae::gatecheck::nonFinite("brae nut", fin.turbulence.nut.internal);
        const Diff dK = compare(fin.turbulence.k.internal, readCells(ofDir + "/k"));
        const Diff dE = compare(fin.turbulence.epsilon.internal, readCells(ofDir + "/epsilon"));
        const Diff dN = compare(fin.turbulence.nut.internal, readCells(ofDir + "/nut"));
        std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
        std::printf("  epsilon: relative %.4e   (epsilon up to %.4e)\n", (double)dE.rel(), (double)dE.refMax);
        std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dN.rel(), (double)dN.refMax);
        check("k agrees with OpenFOAM's relatively", dK.rel() < B_K);
        check("epsilon agrees with OpenFOAM's relatively", dE.rel() < B_EPSILON);
        check("nut agrees with OpenFOAM's relatively", dN.rel() < B_NUT);
    }

    // THE FLUX ACROSS THE PAIR, face by face, against the phi OpenFOAM writes (relative to the mesh
    // motion, as interFoam leaves it)
    {
        const FieldData<scalar> ofPhi = readField<scalar>(ofDir + "/phi");
        scalar dPhi = 0;
        scalar phiScale = 0;
        std::size_t nPhiFaces = 0;
        for (const label pi : {pair.src, pair.tgt})
        {
            const FvPatch& q = patches[static_cast<std::size_t>(pi)];
            const PatchFieldData<scalar>* b = findPatchEntry(ofPhi.boundary, q);
            if (!b)
            {
                continue;
            }
            const std::size_t nf = static_cast<std::size_t>(q.size);
            const std::vector<scalar> ofb = b->valueUniform ? std::vector<scalar>(nf, b->uniformValue) : b->values;
            const Diff d = compare(fin.phi.boundary[static_cast<std::size_t>(pi)], ofb);
            std::printf("  phi on %-6s Linf %.4e (|phi| up to %.4e)\n", q.name.c_str(), (double)d.linf, (double)d.refMax);
            dPhi = std::fmax(dPhi, d.linf);
            phiScale = std::fmax(phiScale, d.refMax);
            nPhiFaces += ofb.size();
        }
        check("OpenFOAM wrote phi on both sides of the pair", nPhiFaces == pair.weights.srcAddress.size()
              + pair.weights.tgtAddress.size());
        check("the pair carries flux", phiScale > scalar(1e-10));
        check("the flux across the pair is OpenFOAM's, face by face", dPhi <= B_PHI*phiScale);
    }

    // THE CONTROL, on the oracle
    const Diff dStill = compare(readVectorCells(stillDir + "/U"), ofU);
    std::printf("  CONTROL: OpenFOAM with the rotor held still, U relative %.4e\n", (double)dStill.rel());
    check("turning the rotor moves OpenFOAM's own U far more than brae is from it",
          dStill.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dStill.rel() > scalar(1e-6));

    // THE DEVICE LOOP REFUSES, by name
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device refusal is not exercised)\n");
    }
    else
    {
        bool named = false;
        try
        {
            InterFields dev;
            runInterFoamDevice(caseDir, startDir, m, gRaw, uncoupled, nSteps, false, &dev);
        }
        catch (const std::exception& e)
        {
            named = std::string(e.what()).find("AMI1") != std::string::npos
                 || std::string(e.what()).find("AMI2") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses the case and names the patch", named);
    }

    std::printf("test_inter_ami_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
