// END-TO-END _cpp kOmegaSST: run a real SST case start to finish on the MIRRORED reference path, with no
// CUDA anywhere, and compare the result against real OpenFOAM.
//
// This is the step that has to pass before any of it is ported to the GPU. A single-iteration residual
// probe is NOT the same evidence: it never exercises the iteration-to-iteration coupling (nut feeding
// nuEff, phi feeding the next convection, the F1 blend moving as k and omega move), which is exactly
// where a port drifts. So this runs the whole case.
//
// THE ORACLE IS REAL OPENFOAM. validation/pitzDailySST/2000 is simpleFoam v2412 output, reproduced
// bit-identically by rerunning OpenFOAM on the checked-in case.
//
// TWO MODES:
//   fixed-point:  <case> 2000 2000 1   -- one iteration from OpenFOAM's own converged state. The printed
//                                        initial residuals are directly comparable to its log.
//   end-to-end:   <case> 0    2000 N   -- from 0/, N iterations, compared against OpenFOAM's converged.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "cell_wall_dist.cuh"
#include "simpleFoam_cpp.cuh"
#include "kOmegaSST_cpp.cuh"
#include "komega_sst_coeffs.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static scalar l2rel(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar n = 0, d = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        n += (a[i] - b[i]) * (a[i] - b[i]);
        d += b[i] * b[i];
    }
    return d > 0 ? std::sqrt(n / d) : std::sqrt(n);
}

static scalar l2relVec(const std::vector<vector>& a, const std::vector<vector>& b)
{
    scalar n = 0, d = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        n += magSqr(a[i] - b[i]);
        d += magSqr(b[i]);
    }
    return d > 0 ? std::sqrt(n / d) : std::sqrt(n);
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::printf("usage: %s <caseDir> <startTime> <refTime> <iters>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], startT = argv[2], refT = argv[3];
    const int iters = std::atoi(argv[4]);
    const scalar nu = 1e-5;

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    // Bisect knobs. The case diverges on this path and each of these isolates one ingredient of it;
    // they only ever turn a feature OFF, so a converging run under a knob names the culprit.
    const bool forceSimple  = std::getenv("SST_SIMPLE")     != nullptr;
    const bool forceUpwind  = std::getenv("SST_UPWIND")     != nullptr;
    const bool noLimited    = std::getenv("SST_NOLIMITED")  != nullptr;
    const bool noBounded    = std::getenv("SST_NOBOUNDED")  != nullptr;
    const bool laminar      = std::getenv("SST_LAMINAR")    != nullptr;
    if (forceSimple) cd.consistent = false;
    cpu::SimpleControl ctl(cd);
    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + startT, simpleDict, m, g, patches);

    GeometricField<scalar> k =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/k"), patches, nC);
    GeometricField<scalar> omega =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/omega"), patches, nC);
    GeometricField<scalar> nut =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/nut"), patches, nC);
    k.evaluateBoundary();
    omega.evaluateBoundary();
    nut.evaluateBoundary();

    // Relaxation from the case. The SST tutorials write `".*" 0.9` rather than naming k and omega, and
    // the dict layer carries OpenFOAM's regex-keyword lookup, so asking for "k" finds it.
    scalar relaxU = 0.9, relaxP = 1.0, relaxK = 0.9, relaxOmega = 0.9;
    if (forceSimple) relaxP = 0.3;   // plain SIMPLE needs pressure relaxation
    if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
    {
        if (const FoamDict* eq = rf->subDict("equations"))
        {
            relaxU     = eq->scalarOr("U", relaxU);
            relaxK     = eq->scalarOr("k", relaxK);
            relaxOmega = eq->scalarOr("omega", relaxOmega);
        }
        if (const FoamDict* fl = rf->subDict("fields"))
        {
            relaxP = fl->scalarOr("p", relaxP);
        }
    }

    cpu::TurbulenceState turb;
    turb.k = &k;
    turb.epsilon = &omega;      // the SST rides the same slot
    turb.nut = &nut;
    turb.sst = true;
    turb.y = cellWallDist(m, g, patches);
    turb.relaxK = relaxK;
    turb.relaxEpsilon = relaxOmega;
    // div(phi,k) and div(phi,omega): `bounded Gauss limitedLinear 1` in this case's fvSchemes.
    turb.boundedTurb = !noBounded;
    turb.limitedLinearTurb = !noLimited;
    turb.turbLimiterCoeff = 1.0;
    turb.tol = 1e-10;
    turb.relTol = 0.0;
    turb.maxIter = 2000;
    {
        // v2412 names it momentumTransport; the older tutorials still ship turbulenceProperties.
        std::string tp = caseDir + "/constant/momentumTransport";
        {
            std::ifstream probe(tp);
            if (!probe.good()) tp = caseDir + "/constant/turbulenceProperties";
        }
        const FoamDict turbProps = readDict(tp);
        if (const FoamDict* ras = turbProps.subDict("RAS")) readKOmegaSSTCoeffs(ras, turb.sstCoeffs);
    }

    cpu::StepInput in;
    in.nu = nu;
    in.turb = laminar ? nullptr : &turb;
    in.relaxU = relaxU;
    in.relaxP = relaxP;
    // pitzDailySST: `bounded Gauss linearUpwind grad(U)` on momentum, `Gauss linear corrected` laplacian.
    in.bounded = true;
    in.linearUpwind = !forceUpwind;
    in.scheme = forceUpwind ? cpu::DivScheme::upwind : cpu::DivScheme::linearUpwind;
    in.correctedLaplacian = true;
    in.tolU = 1e-10;
    in.tolP = 1e-08;
    in.relTolP = 0.0;

    if (laminar)
    {
        in.nuEff.assign(nC, nu);
        in.nuEffBnd.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) in.nuEffBnd[pi].assign(patches[pi].size, nu);
    }

    std::printf("test_simple_sst_cpp: %s  %s -> %s, %d iteration(s)\n",
                caseDir.c_str(), startT.c_str(), refT.c_str(), iters);
    std::printf("  relax: U %.2f  p %.2f  k %.2f  omega %.2f   SIMPLEC %s\n",
                relaxU, relaxP, relaxK, relaxOmega, cd.consistent ? "yes" : "no");

    for (int it = 0; it < iters; ++it)
    {
        const cpu::Residuals res = cpu::simpleStep(f, ctl, in, m, g, patches);
        const bool report = (it == 0) || ((it + 1) % 200 == 0) || (it + 1 == iters);
        if (report)
        {
            scalar omMax = 0, kMax = 0, nutMax = 0;
            for (label c = 0; c < nC; ++c)
            {
                omMax  = std::fmax(omMax,  omega.internal[c]);
                kMax   = std::fmax(kMax,   k.internal[c]);
                nutMax = std::fmax(nutMax, nut.internal[c]);
            }
            std::printf("  it %5d   U %.4e   p %.4e   max: omega %.4e  k %.4e  nut %.4e\n", it + 1,
                        res.count("U") ? res.at("U") : -1.0,
                        res.count("p") ? res.at("p") : -1.0, omMax, kMax, nutMax);
        }
    }

    // Compare against OpenFOAM's converged fields.
    const std::vector<vector> Uref =
        buildField<vector>(readField<vector>(caseDir + "/" + refT + "/U"), patches, nC).internal;
    const std::vector<scalar> pref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/p"), patches, nC).internal;
    const std::vector<scalar> kref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/k"), patches, nC).internal;
    const std::vector<scalar> oref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/omega"), patches, nC).internal;
    const std::vector<scalar> nref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/nut"), patches, nC).internal;

    const scalar eU = l2relVec(f.U.internal, Uref);
    const scalar eP = l2rel(f.p.internal, pref);
    const scalar eK = l2rel(k.internal, kref);
    const scalar eO = l2rel(omega.internal, oref);
    const scalar eN = l2rel(nut.internal, nref);

    std::printf("  vs OpenFOAM %s (L2 rel):  U %.3e  p %.3e  k %.3e  omega %.3e  nut %.3e\n",
                refT.c_str(), eU, eP, eK, eO, eN);

    const char* bound = std::getenv("SST_CPP_TOL");
    const scalar tol = bound ? std::atof(bound) : 0.0;
    if (tol > 0.0)
    {
        const bool ok = (eU < tol) && (eP < tol) && (eK < tol) && (eO < tol) && (eN < tol);
        std::printf("  %s (bound %.1e)\n", ok ? "PASS" : "FAIL", tol);
        return ok ? 0 : 1;
    }
    return 0;
}
