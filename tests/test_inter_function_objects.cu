// interFoam's runtime function objects (inter_function_objects.cuh), held to numbers computed here from the
// same arrays: no GPU and no OpenFOAM involved.
//   * volFieldValue (all and cellZone), surfaceFieldValue (a patch), probes, fieldMinMax, exprField with
//     the three expressions a refill study uses, and runTimeControl's minMax stop -- the values written
//     into postProcessing/<object>/<start>/ against the sums the test forms itself;
//   * the cadence: `writeControl adjustableRunTime; writeInterval 0.002` writes at the steps whose time
//     crosses the next multiple and never at the start; `timeStep` writes every step;
//   * the stop: a speed condition that grows with the step is not satisfied before its threshold and is at it;
//   * `enabled false` removes an object; an object type brae does not run is left alone;
//   * what must stop the run at start-up, named: a weighted operation, autowrite, an unknown patch, a bad
//     expression, a minMax mode that is not ported.
#include "inter_function_objects.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef BRAE_TEST_MESH_DIR
#error "BRAE_TEST_MESH_DIR must name a polyMesh directory (the CMake target defines it)"
#endif

using namespace brae;
using namespace brae::fo;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

bool near(scalar a, scalar b, scalar rel = 1e-12)
{
    return std::fabs(a - b) <= rel*std::max(std::fabs(a), std::fabs(b)) + 1e-300;
}

struct Fixture
{
    PrimitiveMesh                                  m;
    FvGeometry                                     g;
    std::vector<FvPatch>                           patches;
    std::vector<scalar>                            alpha, prgh, ux, uy, uz, uzBase;
    std::vector<std::vector<scalar>>               aPatch;
    std::map<std::string, std::vector<label>>      zones;
    int                                            newSteps = 0;
    std::vector<scalar>                            patchCache;

    Fixture()
    {
        const std::string d = BRAE_TEST_MESH_DIR;
        m.read(d, d, d);
        g.build(m);
        patches = buildPatches(m, g);
        const std::size_t n = g.V().size();
        for (std::size_t c = 0; c < n; ++c)
        {
            alpha.push_back(0.2 + 0.03*static_cast<scalar>(c));
            prgh.push_back(100.0 + static_cast<scalar>(c));
            ux.push_back(0.1*static_cast<scalar>(c + 1));
            uy.push_back(-0.05*static_cast<scalar>(c));
            uzBase.push_back(0.02*static_cast<scalar>(c + 1));
            uz.push_back(uzBase.back());
        }
        for (const FvPatch& p : patches)
        {
            std::vector<scalar> v;
            for (std::size_t i = 0; i < p.magSf.size(); ++i)
            {
                v.push_back(0.5 + 0.01*static_cast<scalar>(i));
            }
            aPatch.push_back(v);
        }
        zones["zoneA"] = {0, 1, 2, 7};
    }

    void setStep(label i)
    {
        for (std::size_t c = 0; c < uz.size(); ++c)
        {
            uz[c] = uzBase[c]*static_cast<scalar>(i);
        }
    }

    Services services(const std::string& caseDir)
    {
        Services s;
        s.caseDir = caseDir;
        s.startName = "0";
        s.precision = 10;
        s.mesh = &m;
        s.geom = &g;
        s.patches = &patches;
        s.cellZones = &zones;
        s.src.newStep = [this]() { ++newSteps; };
        s.src.alpha = [this]() -> const std::vector<scalar>& { return alpha; };
        s.src.p_rgh = [this]() -> const std::vector<scalar>& { return prgh; };
        s.src.Ux = [this]() -> const std::vector<scalar>& { return ux; };
        s.src.Uy = [this]() -> const std::vector<scalar>& { return uy; };
        s.src.Uz = [this]() -> const std::vector<scalar>& { return uz; };
        s.src.alphaPatch = [this](std::size_t p) -> const std::vector<scalar>& { return aPatch[p]; };
        return s;
    }
};

// A host implementation of the Source's device contract: it receives compiled expression trees and raw field
// pointers, never a named host vector. This checks batching, recursive expression expansion, cellZone indexing,
// all supported leaves and the untouched host fallback without needing a GPU.
struct FakeDevice
{
    Fixture* fx;
    int batches = 0, requests = 0;
    bool answer = true;
    std::vector<std::string> expressions;
    std::vector<scalar> cx, cy, cz;

    explicit FakeDevice(Fixture& f) : fx(&f)
    {
        for (const vector& c : f.g.C()) { cx.push_back(c.x); cy.push_back(c.y); cz.push_back(c.z); }
    }

    DeviceArrays fields() const
    {
        DeviceArrays a;
        a.alpha = fx->alpha.data(); a.ux = fx->ux.data(); a.uy = fx->uy.data(); a.uz = fx->uz.data();
        a.V = fx->g.V().data(); a.Cx = cx.data(); a.Cy = cy.data(); a.Cz = cz.data();
        a.nCells = static_cast<label>(fx->g.V().size());
        return a;
    }

    scalar eval(const std::vector<DeviceExpr>& ns, int i, const DeviceArrays& f, label c) const
    {
        const DeviceExpr& e = ns[static_cast<std::size_t>(i)];
        if (e.kind == DeviceExpr::Constant) return e.constant;
        if (e.kind == DeviceExpr::Field)
        {
            switch (e.leaf)
            {
                case DeviceExpr::Alpha: return f.alpha[c];
                case DeviceExpr::Ux: return f.ux[c];
                case DeviceExpr::Uy: return f.uy[c];
                case DeviceExpr::Uz: return f.uz[c];
                case DeviceExpr::PosX: return f.Cx[c];
                case DeviceExpr::PosY: return f.Cy[c];
                default: return f.Cz[c];
            }
        }
        if (e.kind == DeviceExpr::Unary) return -eval(ns, e.left, f, c);
        if (e.kind == DeviceExpr::MagSqrU)
        {
            const scalar x = eval(ns, e.left, f, c), y = eval(ns, e.right, f, c), z = eval(ns, e.third, f, c);
            return x*x + y*y + z*z;
        }
        const scalar a = eval(ns, e.left, f, c), b = eval(ns, e.right, f, c);
        return e.op == '+' ? a + b : e.op == '-' ? a - b : e.op == '*' ? a*b : a/b;
    }

    bool reduce(std::vector<DeviceIntegral>& qs)
    {
        ++batches;
        requests += static_cast<int>(qs.size());
        for (DeviceIntegral& q : qs)
        {
            expressions.push_back(q.expression);
            if (!answer) { q.result->clear(); continue; }
            scalar sum = 0;
            for (label i = 0; i < q.n; ++i)
            {
                const label c = q.cells ? q.cells[i] : i;
                sum += q.fields.V[c]*eval(q.nodes, static_cast<int>(q.nodes.size() - 1), q.fields, c);
            }
            q.result->assign(1, sum);
        }
        return answer;
    }

    void attach(Services& s)
    {
        s.src.deviceFields = [this]() { return fields(); };
        s.src.deviceIntegrals = [this](std::vector<DeviceIntegral>& qs) { return reduce(qs); };
    }
};

std::string tmpDir(const char* tag)
{
    const std::string d = (std::filesystem::temp_directory_path() / (std::string("brae_fo_") + tag)).string();
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d + "/system");
    return d;
}

FoamDict dictOf(const std::string& dir, const std::string& body)
{
    const std::string path = dir + "/system/controlDict";
    std::ofstream(path) << "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n" << body;
    return readDict(path);
}

// rows of a .dat file: numbers only, parentheses removed
std::vector<std::vector<scalar>> rows(const std::string& path)
{
    std::vector<std::vector<scalar>> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        for (char& ch : line)
        {
            if (ch == '(' || ch == ')')
            {
                ch = ' ';
            }
        }
        std::istringstream is(line);
        std::vector<scalar> r;
        std::string tok;
        while (is >> tok)
        {
            r.push_back(std::stod(tok));
        }
        out.push_back(r);
    }
    return out;
}

const char* kFunctions = R"FO(
probeDt 0.002;
functions
{
    windowCoverage
    {
        type surfaceFieldValue;  libs (fieldFunctionObjects);
        regionType patch;        name left;
        operation areaAverage;   fields (alpha.water);
        writeFields false;       log false;
        writeControl adjustableRunTime;  writeInterval $probeDt;
    }
    windowMin
    {
        type surfaceFieldValue;  libs (fieldFunctionObjects);
        regionType patch;        name left;
        operation min;           fields (alpha.water);
        writeControl adjustableRunTime;  writeInterval $probeDt;
    }
    liquid_zoneA
    {
        type volFieldValue;  libs (fieldFunctionObjects);
        regionType cellZone; name zoneA;
        operation volIntegrate; fields (alpha.water);
        writeFields false;   log false;
        writeControl adjustableRunTime; writeInterval $probeDt;
    }
    alphaZ
    {
        type exprField; libs (fieldFunctionObjects);
        field alphaZ; action new;
        expression "alpha.water*pos().z()";
        dimensions [0 1 0 0 0 0 0];
        autowrite false;
        writeControl writeTime;
    }
    alphaKE
    {
        type exprField; libs (fieldFunctionObjects);
        field alphaKE; action new;
        expression "505.05*alpha.water*magSqr(U)";
        dimensions [1 -1 -2 0 0 0 0];
        autowrite false;
        writeControl writeTime;
    }
    alphaUz
    {
        type exprField; libs (fieldFunctionObjects);
        field alphaUz; action new;
        expression "alpha.water*U.z()";
        dimensions [0 1 -1 0 0 0 0];
        autowrite false;
        writeControl writeTime;
    }
    liquid_total
    {
        type volFieldValue;  libs (fieldFunctionObjects);
        regionType all;
        operation volIntegrate;  fields (alpha.water alphaZ alphaKE alphaUz);
        writeFields false;   log false;
        writeControl adjustableRunTime;  writeInterval $probeDt;
    }
    probes
    {
        type probes;  libs (sampling);
        fields (alpha.water U p_rgh);
        fixedLocations true;
        writeControl adjustableRunTime;  writeInterval $probeDt;
        probeLocations
        (
            (0.5 0.5 0.5)   // lower left
            (3.5 2.5 0.5)   // inside
        );
    }
    minMaxU
    {
        type fieldMinMax;  libs (fieldFunctionObjects);
        fields (U);  mode magnitude;
        writeControl adjustableRunTime;  writeInterval $probeDt;
        log false;
    }
    speedLive
    {
        type volFieldValue;  libs (fieldFunctionObjects);
        regionType all;
        operation volIntegrate;  fields (alphaUz);
        writeToFile false;  writeFields false;  log false;
        writeControl timeStep;  writeInterval 1;
    }
    pulseStop
    {
        type runTimeControl;  libs (utilityFunctionObjects);
        conditions
        {
            speedReached
            {
                type minMax;  functionObject speedLive;
                fields (volIntegrate(alphaUz));
                mode maximum;  value THRESHOLD;
            }
        }
        satisfiedAction end;
    }
    interface
    {
        type surfaces;  libs (sampling);
        surfaceFormat vtk;  fields (U);
        writeControl adjustableRunTime;  writeInterval 0.05;
        surfaces { alpha05 { type isoSurface; isoField alpha.water; isoValue 0.5; } }
    }
}
)FO";

std::string withThreshold(scalar t)
{
    std::string s = kFunctions;
    const std::size_t p = s.find("THRESHOLD");
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", t);
    s.replace(p, 9, buf);
    return s;
}

std::string replaceOnce(std::string s, const std::string& from, const std::string& to)
{
    const std::size_t p = s.find(from);
    if (p == std::string::npos)
    {
        std::printf("  FAIL: test text `%s` not found\n", from.c_str());
        ++failures;
        return s;
    }
    s.replace(p, from.size(), to);
    return s;
}

// true when reading `body` stops with a message that contains `what`
bool refuses(Fixture& fx, const char* tag, const std::string& body, const std::string& what)
{
    const std::string dir = tmpDir(tag);
    try
    {
        FoamDict d = dictOf(dir, body);
        FunctionObjectList::read(d, fx.services(dir));
    }
    catch (const std::exception& e)
    {
        const std::string msg = e.what();
        const bool ok = msg.find(what) != std::string::npos;
        if (!ok)
        {
            std::printf("        message was: %s\n", msg.c_str());
        }
        return ok;
    }
    return false;
}

}   // namespace

int main()
{
    Fixture fx;
    const std::vector<scalar>& V = fx.g.V();
    const std::size_t n = V.size();

    // the stop threshold: 2.5 times the speed integral at step 1, so the speed (which grows with the step) is
    // satisfied from step 3
    scalar speed1 = 0;
    for (std::size_t c = 0; c < n; ++c)
    {
        speed1 += V[c]*fx.alpha[c]*fx.uzBase[c];
    }

    std::printf("-- the objects of a refill study over twelve steps of 0.0005 --\n");
    const std::string dir = tmpDir("main");
    FoamDict d = dictOf(dir, withThreshold(2.5*speed1));
    std::shared_ptr<FunctionObjectList> list = FunctionObjectList::read(d, fx.services(dir));
    check("the list is built", list != nullptr);
    check("a stop is possible (pulseStop)", list && list->canStop());

    const scalar dt = 0.0005;
    label firstStop = 0;
    for (label i = 1; i <= 12; ++i)
    {
        fx.setStep(i);
        Step s;
        s.timeIndex = i;
        s.time = static_cast<scalar>(i)*dt;
        s.deltaT = dt;
        s.sinceStart = s.time;
        const bool stop = list->execute(s);
        if (stop && firstStop == 0)
        {
            firstStop = i;
        }
        if (i < 3)
        {
            check(("speed below its threshold at step " + std::to_string(i)).c_str(), !stop);
        }
    }
    check("the speed condition is first satisfied at step 3", firstStop == 3);
    check("newStep was called once per step", fx.newSteps == 12);

    // the cadence: 0.002 apart, never at the start -> steps 4, 8, 12
    const std::string pp = dir + "/postProcessing";
    auto total = rows(pp + "/liquid_total/0/volFieldValue.dat");
    check("liquid_total has three rows (steps 4, 8, 12)", total.size() == 3);
    bool timesOk = total.size() == 3 && near(total[0][0], 0.002, 1e-9) && near(total[1][0], 0.004, 1e-9)
                && near(total[2][0], 0.006, 1e-9);
    check("...at t = 0.002, 0.004, 0.006", timesOk);

    // each row against the sums formed here
    bool valuesOk = total.size() == 3;
    for (std::size_t k = 0; k < total.size() && valuesOk; ++k)
    {
        const label step = static_cast<label>(4*(k + 1));
        fx.setStep(step);
        scalar a = 0, az = 0, ake = 0, auz = 0;
        for (std::size_t c = 0; c < n; ++c)
        {
            a   += V[c]*fx.alpha[c];
            az  += V[c]*fx.alpha[c]*fx.g.C()[c].z;
            ake += V[c]*505.05*fx.alpha[c]*(fx.ux[c]*fx.ux[c] + fx.uy[c]*fx.uy[c] + fx.uz[c]*fx.uz[c]);
            auz += V[c]*fx.alpha[c]*fx.uz[c];
        }
        valuesOk = total[k].size() == 5 && near(total[k][1], a, 1e-9) && near(total[k][2], az, 1e-9)
                && near(total[k][3], ake, 1e-9) && near(total[k][4], auz, 1e-9);
        if (!valuesOk)
        {
            std::printf("        row %zu: got %g %g %g %g, want %g %g %g %g\n", k, total[k][1], total[k][2],
                        total[k][3], total[k][4], a, az, ake, auz);
        }
    }
    check("liquid_total's four integrals match the sums formed here, row by row", valuesOk);

    {
        auto z = rows(pp + "/liquid_zoneA/0/volFieldValue.dat");
        scalar want = 0;
        for (label c : fx.zones["zoneA"])
        {
            want += V[static_cast<std::size_t>(c)]*fx.alpha[static_cast<std::size_t>(c)];
        }
        check("the cellZone integral matches", z.size() == 3 && near(z[0][1], want, 1e-9));
    }
    {
        auto w = rows(pp + "/windowCoverage/0/surfaceFieldValue.dat");
        int pi = -1;
        for (std::size_t i = 0; i < fx.patches.size(); ++i)
        {
            if (fx.patches[i].name == "left") pi = static_cast<int>(i);
        }
        scalar num = 0, den = 0, mn = 1e300;
        for (std::size_t i = 0; i < fx.patches[static_cast<std::size_t>(pi)].magSf.size(); ++i)
        {
            const scalar a = fx.patches[static_cast<std::size_t>(pi)].magSf[i];
            num += a*fx.aPatch[static_cast<std::size_t>(pi)][i];
            den += a;
            mn = std::min(mn, fx.aPatch[static_cast<std::size_t>(pi)][i]);
        }
        check("areaAverage on a patch matches", w.size() == 3 && near(w[0][1], num/den, 1e-9));
        auto wm = rows(pp + "/windowMin/0/surfaceFieldValue.dat");
        check("min on a patch matches", wm.size() == 3 && near(wm[0][1], mn, 1e-9));
    }
    {
        // probes: the cell the point is in, by the same search the object ran
        const std::vector<label> cells = locatePoints({vector{0.5, 0.5, 0.5}, vector{3.5, 2.5, 0.5}}, fx.m, fx.g, "t");
        auto pa = rows(pp + "/probes/0/alpha.water");
        auto pu = rows(pp + "/probes/0/U");
        auto pp_ = rows(pp + "/probes/0/p_rgh");
        bool ok = pa.size() == 3 && pa[0].size() == 3 && near(pa[0][1], fx.alpha[static_cast<std::size_t>(cells[0])])
               && near(pa[0][2], fx.alpha[static_cast<std::size_t>(cells[1])]);
        check("probes: alpha.water at the cells holding the points", ok);
        check("probes: U is three components per probe", pu.size() == 3 && pu[0].size() == 1 + 2*3);
        check("probes: p_rgh at the cells", pp_.size() == 3 && near(pp_[0][1], fx.prgh[static_cast<std::size_t>(cells[0])]));
        const vector c0 = fx.g.C()[static_cast<std::size_t>(cells[0])];
        check("probe 0 lies in the cell it found", std::fabs(c0.x - 0.5) <= 0.5 + 1e-9 && std::fabs(c0.y - 0.5) <= 0.5 + 1e-9);
        std::ifstream hf(pp + "/probes/0/alpha.water");
        std::string h1;
        std::getline(hf, h1);
        check("probes: the header names the first probe's point", h1.rfind("# Probe 0 (", 0) == 0);
    }
    {
        std::ifstream mf(pp + "/minMaxU/0/fieldMinMax.dat");
        std::vector<std::string> data;
        for (std::string line; std::getline(mf, line);)
        {
            if (!line.empty() && line[0] != '#') data.push_back(line);
        }
        check("fieldMinMax wrote its three rows", data.size() == 3);
        if (data.size() == 3)
        {
            // step 12 is the last row: |U| min and max over the cells, from the arrays at that step
            fx.setStep(12);
            scalar mn = 1e300, mx = 0;
            for (std::size_t c = 0; c < n; ++c)
            {
                const scalar m = std::sqrt(fx.ux[c]*fx.ux[c] + fx.uy[c]*fx.uy[c] + fx.uz[c]*fx.uz[c]);
                mn = std::min(mn, m);
                mx = std::max(mx, m);
            }
            std::string row = data[2];
            for (char& ch : row) if (ch == '(' || ch == ')') ch = ' ';
            std::istringstream is(row);
            std::vector<std::string> tk;
            for (std::string w; is >> w;) tk.push_back(w);
            check("fieldMinMax's |U| minimum and maximum match",
                  tk.size() >= 9 && near(std::stod(tk[3]), mn, 1e-9) && near(std::stod(tk[8]), mx, 1e-9));
        }
    }
    Fixture stateFx;
    const std::string stateDir = tmpDir("state");
    FoamDict stateDict = dictOf(stateDir, withThreshold(1e30));
    auto stateList = FunctionObjectList::read(stateDict, stateFx.services(stateDir));
    for (label i = 1; i <= 12; ++i)
    {
        stateFx.setStep(i);
        Step s; s.timeIndex = i; s.time = i*0.0005; s.deltaT = 0.0005; s.sinceStart = s.time;
        stateList->execute(s);
    }
    const FunctionObjectState st = stateList->state();
    {
        const auto scalarOf = [&](const std::string& object, const std::string& key) -> const FunctionObjectState::Value&
        {
            return st.results.at(object).at("scalar").at(key);
        };
        check("volume result keys include the operation", scalarOf("liquid_total", "volIntegrate(alpha.water)").width >= 0
              && scalarOf("liquid_zoneA", "volIntegrate(zoneA,alpha.water)").width >= 0);
        check("surface result keys include the operation and patch",
              scalarOf("windowCoverage", "areaAverage(left,alpha.water)").width >= 0
              && scalarOf("windowMin", "min(left,alpha.water)").width >= 0);
        check("the stop's live integral is saved", scalarOf("speedLive", "volIntegrate(alphaUz)").width >= 0);
        check("probe and min/max results are saved", st.results.at("probes").at("scalar").count("average(alpha.water)")
              && st.results.at("minMaxU").at("scalar").count("max(mag(U))"));
        check("runTimeControl condition properties are saved", st.properties.at("__control_pulseStop").size() == 1
              && st.properties.at("__control_pulseStop")[0].find("pulseStop\n{\n    speedReached") != std::string::npos);

        const std::string savedDir = tmpDir("savedState");
        const std::string saved = savedDir + "/uniform/functionObjects";
        std::filesystem::create_directories(saved);
        std::ofstream os(saved + "/functionObjectProperties");
        os << "cdf7e925f5746741c316f5fbcf39ad0dfca90775 { speedLive { scalar { volIntegrate(alphaUz) 123.5; } } "
              "liquid_zoneA { scalar { volIntegrate(zoneA,alpha.water) 7.25; } } }\n";
        os.close();
        FoamDict rd = dictOf(dir, withThreshold(1e30));
        auto restarted = FunctionObjectList::read(rd, fx.services(dir), savedDir);
        check("saved scalar results are available before their next monitor", restarted != nullptr);
    }
    check("an object that is not ported (surfaces) wrote nothing", !std::filesystem::exists(pp + "/interface"));
    check("an exprField wrote no directory (autowrite false)", !std::filesystem::exists(pp + "/alphaUz"));
    check("speedLive wrote no file (writeToFile false)", !std::filesystem::exists(pp + "/speedLive"));

    std::printf("-- a restart names its own directory --\n");
    {
        Fixture fx2;
        const std::string dir2 = tmpDir("restart");
        FoamDict d2 = dictOf(dir2, withThreshold(1e30));
        Services s2 = fx2.services(dir2);
        s2.startName = "0.5";
        auto l2 = FunctionObjectList::read(d2, s2);
        Step s;
        s.timeIndex = 1000;
        s.time = 0.5 + dt;
        s.deltaT = dt;
        s.sinceStart = dt;
        for (label i = 1; i <= 4; ++i)
        {
            s.timeIndex = 1000 + i;
            s.time = 0.5 + static_cast<scalar>(i)*dt;
            s.sinceStart = static_cast<scalar>(i)*dt;
            l2->execute(s);
        }
        check("postProcessing/<object>/0.5/ holds the restarted run", std::filesystem::exists(dir2 + "/postProcessing/liquid_total/0.5/volFieldValue.dat"));
    }

    std::printf("-- enabled false --\n");
    {
        Fixture fx3;
        const std::string dir3 = tmpDir("disabled");
        std::string body = withThreshold(1e30);
        body = replaceOnce(body, "type runTimeControl;", "type runTimeControl; enabled false;");
        FoamDict d3 = dictOf(dir3, body);
        auto l3 = FunctionObjectList::read(d3, fx3.services(dir3));
        check("a disabled runTimeControl cannot stop the run", l3 && !l3->canStop());
    }

    std::printf("-- a probe in no cell is skipped, as OpenFOAM's are --\n");
    {
        Fixture f4;
        const std::string dir4 = tmpDir("skipprobe");
        FoamDict d4 = dictOf(dir4, replaceOnce(withThreshold(1e30), "(3.5 2.5 0.5)", "(30.5 2.5 0.5)"));
        auto l4 = FunctionObjectList::read(d4, f4.services(dir4));
        for (label i = 1; i <= 4; ++i)
        {
            Step s;
            s.timeIndex = i;
            s.time = static_cast<scalar>(i)*dt;
            s.deltaT = dt;
            s.sinceStart = s.time;
            l4->execute(s);
        }
        auto pa = rows(dir4 + "/postProcessing/probes/0/alpha.water");
        check("the file has the time and one column: the probe outside is left out", pa.size() == 1 && pa[0].size() == 2);
        std::ifstream hf(dir4 + "/postProcessing/probes/0/alpha.water");
        std::string h1, h2;
        std::getline(hf, h1);
        std::getline(hf, h2);
        check("...and the remaining probe is numbered 0 again", h1.rfind("# Probe 0 (", 0) == 0 && h2.rfind("# Probe", 0) != 0);
    }

    std::printf("-- what stops the run at start-up --\n");
    {
        Fixture f;
        check("an operation brae does not have names the operations it has",
              refuses(f, "r1", "functions { v { type volFieldValue; regionType all; operation weightedAverage; "
                               "fields (alpha.water); writeControl timeStep; writeInterval 1; } }",
                      "volIntegrate, volAverage"));
        check("autowrite true names itself",
              refuses(f, "r2", replaceOnce(withThreshold(1), "autowrite false;", "autowrite true;"), "autowrite"));
        check("an unknown patch is named",
              refuses(f, "r3", replaceOnce(withThreshold(1), "name left;", "name nowhere;"), "nowhere"));
        check("a bad expression shows the expression",
              refuses(f, "r4", replaceOnce(withThreshold(1), "alpha.water*U.z()", "alpha.water*foo.z()"), "alpha.water*foo.z()"));
        check("a minMax mode that is not ported is named",
              refuses(f, "r5", replaceOnce(withThreshold(1), "mode maximum;", "mode range;"), "range"));
        check("a region type that is not ported is named",
              refuses(f, "r6", replaceOnce(withThreshold(1), "regionType cellZone;", "regionType faceZone;"), "faceZone"));
        check("a zone the mesh does not have is named",
              refuses(f, "r7", replaceOnce(withThreshold(1), "name zoneA;", "name noZone;"), "noZone"));
        check("a run whose every probe lies outside the mesh is refused",
              refuses(f, "r8", replaceOnce(replaceOnce(withThreshold(1), "(3.5 2.5 0.5)", "(30.5 2.5 0.5)"),
                                           "(0.5 0.5 0.5)", "(30.5 2.5 0.5)"), "none of the locations"));
    }

    std::printf("-- expression evaluator --\n");
    {
        Fixture f;
        const std::string dirE = tmpDir("expr");
        const std::string body = std::string("functions { a { type exprField; field a; action new; "
            "expression \"-(alpha.water + 2)*3/4 - sqr(1.5) + mag(U) - magSqr(U) + pos().x()\"; autowrite false; } "
            "tot { type volFieldValue; regionType all; operation volIntegrate; fields (a); "
            "writeControl timeStep; writeInterval 1; } }");
        FoamDict de = dictOf(dirE, body);
        auto le = FunctionObjectList::read(de, f.services(dirE));
        Step s;
        s.timeIndex = 1;
        s.time = 1;
        s.deltaT = 1;
        s.sinceStart = 1;
        le->execute(s);
        scalar want = 0;
        for (std::size_t c = 0; c < n; ++c)
        {
            const scalar u2 = f.ux[c]*f.ux[c] + f.uy[c]*f.uy[c] + f.uz[c]*f.uz[c];
            want += V[c]*(-(f.alpha[c] + 2)*3.0/4.0 - 1.5*1.5 + std::sqrt(u2) - u2 + f.g.C()[c].x);
        }
        auto r = rows(dirE + "/postProcessing/tot/0/volFieldValue.dat");
        check("-(a+2)*3/4 - sqr(1.5) + mag(U) - magSqr(U) + pos().x() integrates to the hand sum",
              r.size() == 1 && near(r[0][1], want, 1e-10));
    }

    std::printf("-- device Source: batched integrals, cellZones and fallback --\n");
    {
        Fixture f;
        f.setStep(2);
        FakeDevice dev(f);
        const std::string dd = tmpDir("device");
        Services svc = f.services(dd);
        dev.attach(svc);
        const std::string body = R"FO(functions {
            az { type exprField; field az; expression "alpha.water*pos().z()"; autowrite false; }
            nested { type exprField; field nested; expression "2*az + 0.25*pos().y()"; autowrite false; }
            au { type exprField; field au; expression "-(alpha.water*U.z() + 0.5)"; autowrite false; }
            ke { type exprField; field ke; expression "505.05*alpha.water*magSqr(U)"; autowrite false; }
            root { type exprField; field root; expression "sqrt(alpha.water)"; autowrite false; }
            all { type volFieldValue; regionType all; operation volIntegrate; fields (nested au ke root);
                  writeControl timeStep; writeInterval 1; }
            zone { type volFieldValue; regionType cellZone; name zoneA; operation volIntegrate;
                   fields (alpha.water nested); writeControl timeStep; writeInterval 1; }
            avg { type volFieldValue; regionType all; operation volAverage; fields (nested);
                  writeControl timeStep; writeInterval 1; }
            other { type probes; fields (alpha.water); fixedLocations true; probeLocations ((0.5 0.5 0.5));
                    writeControl timeStep; writeInterval 1; }
        })FO";
        FoamDict d = dictOf(dd, body);
        auto l = FunctionObjectList::read(d, svc);
        Step s; s.timeIndex = 1; s.time = 1; s.deltaT = 1; s.sinceStart = 1;
        l->execute(s);
        check("all supported due fields arrive in one device batch", dev.batches == 1 && dev.requests == 5);
        auto a = rows(dd + "/postProcessing/all/0/volFieldValue.dat");
        auto z = rows(dd + "/postProcessing/zone/0/volFieldValue.dat");
        const std::vector<scalar>& fV = f.g.V();
        scalar nested = 0, au = 0, ke = 0, root = 0;
        for (std::size_t c = 0; c < fV.size(); ++c)
        {
            const scalar nz = 2*f.alpha[c]*f.g.C()[c].z + 0.25*f.g.C()[c].y;
            const scalar u2 = f.ux[c]*f.ux[c] + f.uy[c]*f.uy[c] + f.uz[c]*f.uz[c];
            nested += fV[c]*nz;
            au += fV[c]*(-(f.alpha[c]*f.uz[c] + 0.5));
            ke += fV[c]*505.05*f.alpha[c]*u2;
            root += fV[c]*std::sqrt(f.alpha[c]);
        }
        check("nested pos, unary minus, constants, U and magSqr(U) match",
              a.size() == 1 && a[0].size() == 5 && near(a[0][1], nested, 1e-9) && near(a[0][2], au, 1e-9)
              && near(a[0][3], ke, 1e-9) && near(a[0][4], root, 1e-9));
        scalar za = 0, zn = 0;
        for (label cc : f.zones["zoneA"])
        {
            const std::size_t c = static_cast<std::size_t>(cc);
            za += fV[c]*f.alpha[c];
            zn += fV[c]*(2*f.alpha[c]*f.g.C()[c].z + 0.25*f.g.C()[c].y);
        }
        check("cellZone uses exactly its indexed cells", z.size() == 1 && z[0].size() == 3
              && near(z[0][1], za) && near(z[0][2], zn));
        check("unsupported sqrt remains a correct host result", a.size() == 1 && near(a[0][4], root, 1e-9));
        auto av = rows(dd + "/postProcessing/avg/0/volFieldValue.dat");
        scalar volume = 0; for (scalar v : fV) volume += v;
        check("volAverage stays on its host operation path", av.size() == 1 && near(av[0][1], nested/volume, 1e-9)
              && dev.requests == 5);

        Fixture g;
        g.setStep(2);
        FakeDevice no(g); no.answer = false;
        const std::string dh = tmpDir("deviceFallback");
        Services sh = g.services(dh); no.attach(sh);
        FoamDict hd = dictOf(dh, body);
        auto hl = FunctionObjectList::read(hd, sh);
        hl->execute(s);
        auto ha = rows(dh + "/postProcessing/all/0/volFieldValue.dat");
        check("declined device batch falls back to every host integral", ha.size() == 1 && ha[0].size() == 5
              && near(ha[0][1], nested, 1e-9) && near(ha[0][2], au, 1e-9)
              && near(ha[0][3], ke, 1e-9) && near(ha[0][4], root, 1e-9));

        Fixture q;
        FakeDevice qd(q); const std::string dq = tmpDir("deviceStop");
        Services sq = q.services(dq); qd.attach(sq);
        FoamDict qdict = dictOf(dq, withThreshold(2.5*speed1));
        auto ql = FunctionObjectList::read(qdict, sq);
        label stopAt = 0;
        for (label i = 1; i <= 3; ++i)
        {
            q.setStep(i); Step qs; qs.timeIndex = i; qs.time = i*dt; qs.deltaT = dt; qs.sinceStart = qs.time;
            if (ql->execute(qs) && stopAt == 0) stopAt = i;
        }
        check("runTimeControl consumes the device value and still stops at step 3", stopAt == 3);
    }

    std::printf(failures ? "FAILED: %d\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
