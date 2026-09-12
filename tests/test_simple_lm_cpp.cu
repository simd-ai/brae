// END-TO-END _cpp kOmegaSSTLM: run the T3A transition case start to finish on the MIRRORED reference
// path, with no CUDA anywhere, and compare the result against real OpenFOAM.
//
// This is the step that has to pass before any of it is ported to the GPU. kOmegaSSTLM is kOmegaSST plus
// two transported scalars (ReThetat, gammaInt) and three overrides of the base model, and the coupling
// between them is the whole model: gammaIntEff gates k's production, k and omega feed RT and Rev, those
// feed Fonset, which feeds gammaInt, which becomes the next iteration's gammaIntEff. A single-iteration
// probe never exercises that loop, so this runs the whole case.
//
// THE ORACLE IS REAL OPENFOAM: T3A run by simpleFoam v2412 to its own residualControl.
//
// T3A IS A TRANSITION CASE, which is what makes it worth running: a fully-turbulent kOmegaSST gets the
// flat-plate skin friction wrong in exactly the region the transition model exists to capture, so the
// gate has a built-in discriminator -- the same case with plain kOmegaSST must be measurably further
// from OpenFOAM's kOmegaSSTLM answer than this path is.
//
// TWO MODES:
//   fixed-point:  <case> <T> <T> 1   -- one iteration from OpenFOAM's own converged state.
//   end-to-end:   <case> 0   <T> N   -- from 0/, N iterations, compared against OpenFOAM's converged.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "cell_wall_dist.cuh"
#include "simpleFoam_cpp.cuh"
#include "kOmegaSST_cpp.cuh"
#include "kOmegaSSTLM_cpp.cuh"
#include "komega_sst_coeffs.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

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
    const scalar nu = 1.5e-5;   // T3A's transportProperties

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
    // MUST_READ in OpenFOAM: kOmegaSSTLM has no defaults for these and refuses to construct without them.
    GeometricField<scalar> ReThetat =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/ReThetat"), patches, nC);
    GeometricField<scalar> gammaInt =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/gammaInt"), patches, nC);
    k.evaluateBoundary();
    omega.evaluateBoundary();
    nut.evaluateBoundary();
    ReThetat.evaluateBoundary();
    gammaInt.evaluateBoundary();

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
    // LM off (LM_PLAIN_SST=1) is the DISCRIMINATOR, not a convenience: it runs the identical solver with
    // the transition model removed, so the gate can require the transition path to be measurably closer
    // to OpenFOAM's kOmegaSSTLM answer than a fully-turbulent one.
    if (std::getenv("LM_PLAIN_SST") == nullptr)
    {
        turb.lm       = true;
        turb.ReThetat = &ReThetat;
        turb.gammaInt = &gammaInt;
    }
    turb.y = cellWallDist(m, g, patches);
    turb.relaxK = relaxK;
    turb.relaxEpsilon = relaxOmega;
    // div(phi,k) and div(phi,omega): `bounded Gauss limitedLinear 1` in this case's fvSchemes.
    turb.boundedTurb = !noBounded;
    // T3A asks for `bounded Gauss linearUpwind grad` on every turbulence scalar, which is a different
    // matrix from limitedLinear AND from upwind: linearUpwind's matrix is upwind's plus a deferred
    // gradient correction on the source.
    turb.limitedLinearTurb = false;
    turb.linearUpwindTurb  = !noLimited;
    turb.turbLimiterCoeff = 1.0;
    // THE CASE'S OWN SOLVER SETTINGS, not a tight default. T3A caps every turbulence solve at
    // `maxIter 10` with `relTol 0.1`, and on a resolved mesh that is not a detail: the omega equation
    // there has production ~2x destruction in the second cell layer, so solving it to 1e-10 drives omega
    // to the local production/destruction balance every iteration instead of the smoothed value ten
    // symGaussSeidel sweeps leave. Over-solving a stiff sub-equation is a different iteration, not a
    // more accurate one.
    turb.tol = 1e-08;
    turb.relTol = 0.1;
    turb.maxIter = 10;
    if (const FoamDict* sv = fvSolution.subDict("solvers"))
    {
        if (const FoamDict* so = sv->subDict("omega"))
        {
            turb.tol     = so->scalarOr("tolerance", turb.tol);
            turb.relTol  = so->scalarOr("relTol",    turb.relTol);
            turb.maxIter = so->intOr   ("maxIter",   turb.maxIter);
        }
    }
    if (std::getenv("LM_TIGHT_TURB")) { turb.tol = 1e-10; turb.relTol = 0.0; turb.maxIter = 2000; }
    {
        // v2412 names it momentumTransport; the older tutorials still ship turbulenceProperties.
        std::string tp = caseDir + "/constant/momentumTransport";
        {
            std::ifstream probe(tp);
            if (!probe.good()) tp = caseDir + "/constant/turbulenceProperties";
        }
        const FoamDict turbProps = readDict(tp);
        if (const FoamDict* ras = turbProps.subDict("RAS"))
        {
            readKOmegaSSTCoeffs(ras, turb.sstCoeffs);
            cpu::kOmegaSSTLM::readCoeffs(ras, turb.lmCoeffs);
        }
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
    // THE CASE'S OWN LINEAR-SOLVER SETTINGS. T3A asks for `relTol 0.1` on both U and p and caps U at
    // `maxIter 10`; the CUDA V2 driver reads them and converges here. Solving each outer iteration to
    // 1e-10 instead is not a stricter version of the same iteration -- with SIMPLEC and no pressure
    // relaxation it is a different one, and on this mesh it walks away from the answer.
    in.tolU = 1e-08; in.relTolU = 0.1;
    in.tolP = 1e-06; in.relTolP = 0.1;
    if (const FoamDict* sv = fvSolution.subDict("solvers"))
    {
        if (const FoamDict* su = sv->subDict("U"))
        {
            in.tolU    = su->scalarOr("tolerance", in.tolU);
            in.relTolU = su->scalarOr("relTol",    in.relTolU);
        }
        if (const FoamDict* sp = sv->subDict("p"))
        {
            in.tolP    = sp->scalarOr("tolerance", in.tolP);
            in.relTolP = sp->scalarOr("relTol",    in.relTolP);
        }
    }
    if (std::getenv("LM_TIGHT_LINEAR")) { in.tolU = 1e-10; in.relTolU = 0.0; in.tolP = 1e-10; in.relTolP = 0.0; }

    if (laminar)
    {
        in.nuEff.assign(nC, nu);
        in.nuEffBnd.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) in.nuEffBnd[pi].assign(patches[pi].size, nu);
    }

    std::printf("test_simple_lm_cpp: %s  %s -> %s, %d iteration(s)\n",
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
            scalar giMin = 1e300, giMax = 0, rtMax = 0;
            if (turb.lm)
                for (label c = 0; c < nC; ++c)
                {
                    giMin = std::fmin(giMin, gammaInt.internal[c]);
                    giMax = std::fmax(giMax, gammaInt.internal[c]);
                    rtMax = std::fmax(rtMax, ReThetat.internal[c]);
                }
            if (turb.lm)
                std::printf("  it %5d   gammaInt [%.4f, %.4f]  max ReThetat %.4e\n",
                            it + 1, giMin, giMax, rtMax);
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

    // WHERE. A single L2 number cannot distinguish a field that is uniformly 20% out from one that is
    // exact except for a handful of cells, and those have different causes.
    {
        std::vector<label> idx(nC);
        for (label c = 0; c < nC; ++c) idx[c] = c;
        std::sort(idx.begin(), idx.end(), [&](label a, label b) {
            return magSqr(f.U.internal[a] - Uref[a]) > magSqr(f.U.internal[b] - Uref[b]);
        });
        scalar top10 = 0, total = 0;
        for (label c = 0; c < nC; ++c) total += magSqr(f.U.internal[c] - Uref[c]);
        for (int i = 0; i < 10 && i < nC; ++i) top10 += magSqr(f.U.internal[idx[i]] - Uref[idx[i]]);
        std::printf("  worst U cells (%.1f%% of the squared error is in the top 10):\n",
                    total > 0 ? 100.0 * top10 / total : 0.0);
        const int nShow = std::getenv("LM_SHOW") ? std::atoi(std::getenv("LM_SHOW")) : 5;
        for (int i = 0; i < nShow && i < nC; ++i)
        {
            const label c = idx[i];
            std::printf("    cell %6d  C (%.4e %.4e)  brae U (%9.4e %9.4e) p %9.4e   OF U (%9.4e %9.4e) p %9.4e\n",
                        (int)c, g.C()[c].x, g.C()[c].y,
                        f.U.internal[c].x, f.U.internal[c].y, f.p.internal[c],
                        Uref[c].x, Uref[c].y, pref[c]);
        }
    }

    const char* bound = std::getenv("LM_CPP_TOL");
    const scalar tol = bound ? std::atof(bound) : 0.0;
    if (tol > 0.0)
    {
        const bool ok = (eU < tol) && (eP < tol) && (eK < tol) && (eO < tol) && (eN < tol);
        std::printf("  %s (bound %.1e)\n", ok ? "PASS" : "FAIL", tol);
        return ok ? 0 : 1;
    }
    // NET FLUX THROUGH EACH PATCH. OpenFOAM's constrainHbyA sets HbyA_b = U_b on every patch whose U is
    // not assignable -- slip and noSlip both -- so the flux through `above` and `plate` must be exactly
    // zero. A nonzero one is mass entering the domain through a wall, which the pressure field then has
    // to accommodate.
    {
        std::printf("  net flux per patch (brae):\n");
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            scalar sum = 0, asum = 0;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                sum  += f.phi.boundary[pi][i];
                asum += std::fabs(f.phi.boundary[pi][i]);
            }
            std::printf("    %-14s %-10s n %5d   sum %13.6e   sum|phi| %13.6e\n",
                        patches[pi].name.c_str(), patches[pi].type.c_str(),
                        (int)patches[pi].size, sum, asum);
        }
    }

    // A ROW of cells at roughly constant height, ordered by x: the profile that says whether an error is
    // one column or the whole upstream region.
    if (const char* ys = std::getenv("LM_ROW"))
    {
        const scalar yTarget = std::atof(ys);
        std::vector<label> row;
        for (label c = 0; c < nC; ++c)
            if (std::fabs(g.C()[c].y - yTarget) < 0.06 * yTarget) row.push_back(c);
        std::sort(row.begin(), row.end(),
                  [&](label a, label b) { return g.C()[a].x < g.C()[b].x; });
        std::printf("  row near y = %.4g (%d cells):\n", yTarget, (int)row.size());
        const int stride = std::max<int>(1, (int)row.size() / 24);
        for (std::size_t i = 0; i < row.size(); i += stride)
        {
            const label c = row[i];
            std::printf("    x %.5e   brae Ux %9.4e p %9.4e    OF Ux %9.4e p %9.4e\n",
                        g.C()[c].x, f.U.internal[c].x, f.p.internal[c], Uref[c].x, pref[c]);
        }
    }
    return 0;
}
