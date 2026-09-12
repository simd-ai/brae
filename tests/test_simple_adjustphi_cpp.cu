// adjustPhi's two face sets are ONE set -- the host mirror on an inletOutlet outflow.
//
// adjustPhi.C:59 classifies a patch as FIXED outflow by `fixesValue() && !isA<inletOutlet>`, and the
// scale loop applies massCorr to everything not fixed. brae's pEqn_cpp carried the full predicate in
// the SUM loop and `fixesValue()` alone in the SCALE loop, so an inletOutlet outflow was counted
// adjustable and then never scaled: massCorr was computed for one face set and applied to another,
// and global continuity missed by exactly the inletOutlet share. The V2 device mask had the same
// missing half, where it surfaced as deviceAdjustPhi REFUSING a case OpenFOAM solves ("adjustable
// mass outflow 0.000000" on this very fixture).
//
// validation/simpleBoxIO is the discriminating case: the ONLY outflow is an inletOutlet patch and no
// p patch fixes a value, so adjustPhi is live and everything hangs on classifying that one patch.
#include "simpleFoam_cpp.cuh"
#include "foam_field_reader.cuh"
#include "scheme_parse.cuh"
#include <cmath>
#include <cstdio>
#include <string>

using namespace brae;

static int failures = 0;
static void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("  %-58s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

static double relV(const std::vector<vector>& a, const std::vector<vector>& b)
{
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        num += (a[i].x-b[i].x)*(a[i].x-b[i].x) + (a[i].y-b[i].y)*(a[i].y-b[i].y)
             + (a[i].z-b[i].z)*(a[i].z-b[i].z);
        den += b[i].x*b[i].x + b[i].y*b[i].y + b[i].z*b[i].z;
    }
    return den > 0 ? std::sqrt(num/den) : std::sqrt(num);
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::printf("usage: %s <caseDir> <ofConvergedTime> <iters>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], ofT = argv[2];
    const int iters = std::atoi(argv[3]);

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");
    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    cpu::SimpleControl ctl(cd);

    cpu::SimpleFields f = cpu::createFields(caseDir + "/0", simpleDict, m, g, patches);

    // ENGAGEMENT: without an inletOutlet U patch and a live adjustPhi this gate is the plain path.
    bool anyIO = false;
    for (const auto& b : f.U.boundary) if (b->isInletOutlet()) anyIO = true;
    check("the fixture carries an inletOutlet U patch (engagement)", anyIO);
    bool anyFixedP = false;
    for (const auto& b : f.p.boundary) if (b->fixesValue()) anyFixedP = true;
    check("no p patch fixes a value, so adjustPhi is live (engagement)", !anyFixedP);

    const FoamDict tp = readDict(caseDir + "/constant/transportProperties");
    cpu::StepInput in;
    in.nu = tp.scalarOr("nu", 1.5e-05);
    in.relaxU = 0.7; in.relaxP = 0.3;    // the fixture's own relaxationFactors
    // LAMINAR path: nuEff is used as given (StepInput's contract), so it must be filled.
    in.nuEff.assign(static_cast<std::size_t>(nC), in.nu);
    in.nuEffBnd.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        in.nuEffBnd[pi].assign(static_cast<std::size_t>(patches[pi].size), in.nu);
    // Schemes from the case, not assumed: bounded Gauss upwind on div(phi,U), corrected laplacians.
    in.bounded = parseFieldDivScheme(caseDir, "U").bounded;
    DeviceSimpleControls sctl;
    parseFvSchemesControls(caseDir, sctl);
    in.correctedLaplacian = sctl.nonOrth;

    for (int it = 1; it <= iters; ++it)
    {
        (void)cpu::simpleStep(f, ctl, in, m, g, patches);
        if (it == 1)
        {
            // THE SHARP CHECK. After adjustPhi + the pressure correction, the net boundary flux is
            // global continuity itself. The broken scaler left it off by the whole inletOutlet share
            // (measured 5.0e-02 here, the entire outflow) because massCorr never touched the one
            // adjustable patch it was computed for.
            scalar net = 0, massIn = 0;
            for (const auto& bp : f.phi.boundary)
                for (scalar v : bp) { net += v; if (v < 0) massIn -= v; }
            std::printf("     net boundary flux after iter 1: %.6e  (massIn %.6e)\n",
                        (double)net, (double)massIn);
            check("global continuity holds after adjustPhi (|net|/massIn < 1e-10)",
                  massIn > 1e-6 && std::fabs((double)net) / (double)massIn < 1e-10);
            check("...and the case actually flows (massIn > 1e-6, non-vacuous)", massIn > 1e-6);
        }
    }

    // Converged-vs-converged against real OpenFOAM. Measured 3.295e-06, IDENTICAL at 200, 600 and
    // 1200 iterations -- the host mirror converges to a fixed point 3.3e-06 from OpenFOAM's on this
    // inletOutlet+all-Neumann shape, while the V2 device path reaches 4.6e-11 on the same case. That
    // converged offset is an OPEN FINDING (PORT.md), not noise; the bound sits above it and exists to
    // catch the mask/scale-set class of defect, whose broken form fails the continuity check above
    // outright before this line matters.
    const FieldData<vector> uOF = readField<vector>(caseDir + "/" + ofT + "/U");
    const double dU = relV(f.U.internal, uOF.internalField);
    std::printf("     U vs OpenFOAM at its converged %s: %.6e\n", ofT.c_str(), dU);
    check("U matches OpenFOAM's converged state", dU < 1e-5);

    // The simpleFoam mirror's coupled-patch refusal (same placeholder mechanism as the rho arm).
    {
        std::vector<FvPatch> coupled = patches;
        coupled[0].type = "cyclicAMI";
        bool threw = false;
        std::string msg;
        try { (void)cpu::createFields(caseDir + "/0", simpleDict, m, g, coupled); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        check("a coupled patch is refused by the simpleFoam mirror", threw);
        check("...naming the patch", msg.find("cyclicAMI") != std::string::npos
                                  && msg.find(coupled[0].name) != std::string::npos);
    }

    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
