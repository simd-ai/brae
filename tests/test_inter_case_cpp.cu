// interFoam's createFields against damBreak's own case, as OpenFOAM's blockMesh and setFields build it.
//
// The case is prepared by tests/interfoam_createfields_vs_openfoam.sh, because damBreak ships 0.orig
// and no mesh -- the fixture is a PROCEDURE, not a directory this repo can check in.
//
// WHAT IS UNDER TEST is the one translation from the case on disk to the fields the solver runs on.
// rhoSimpleFoam's mirror wrote down why it has to be ONE: a private copy in the driver is the defect
// this project keeps finding a level up -- the gate proves the step, the driver feeds it something
// else, and nothing compares the two. buildInterFields is what both call.
//
// THREE ARMS THAT ARE NOT "did the numbers load":
//
//   * THE ALPHA FIELD'S NAME comes from `phases (water air)` in transportProperties, not from a
//     convention. A port that reads "alpha1" finds no file on any shipped case.
//   * phi IS READ WHEN IT IS THERE (createAlphaFluxes.H). On a cold start it is not, and the fallback
//     is linearInterpolate(U) & Sf; on a restart it is, and recomputing it from U discards exactly the
//     continuity error the previous pressure corrector drove down. ONLY THE COLD-START BRANCH IS
//     EXERCISED HERE -- damBreak's 0 directory has no phi, and this gate does not run a step to make
//     one. The read branch is readPhiIfPresent's own, shared with every other brae solver and gated
//     with them; what this arm adds is that interFoam goes through that shared reader at all rather
//     than computing the flux unconditionally.
//   * rho USES THE RAW ALPHA. setFields leaves alpha exactly 0 or 1, so on THIS case the raw and
//     clamped forms agree -- which is precisely why the distinction is gated in
//     tests/test_two_phase_mixture.cu on out-of-range values instead of here.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "inter_case_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

void checkNum(const char* what, scalar got, scalar want, scalar tol = scalar(1e-10))
{
    const bool ok = std::fabs(got - want) <= tol * std::fmax(scalar(1), std::fabs(want));
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s (got %.17g want %.17g)\n", what, (double)got, (double)want);
    if (!ok) ++failures;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== interFoam createFields vs damBreak ==\n");
    if (argc < 3)
    {
        std::printf("  SKIP: usage: test_inter_case_cpp <caseDir> <startDir>\n");
        return 77;
    }
    const std::string caseDir = argv[1], startDir = argv[2];
    if (!std::filesystem::exists(caseDir + "/constant/polyMesh/owner"))
    {
        std::printf("  SKIP: no mesh at %s/constant/polyMesh\n", caseDir.c_str());
        return 77;
    }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    std::printf("  mesh: %ld cells, %ld internal faces, %zu patches\n",
                (long)m.nCells(), (long)m.nInternalFaces(), patches.size());

    const InterFields f = buildInterFields(caseDir, startDir, m, g, patches);

    // ---- 1. the case's own settings ---------------------------------------------------------------
    check   ("the alpha field is named from `phases (water air)`", f.alphaName == "alpha.water");
    checkNum("rho1 (water)", f.mixture.phases.rho1, scalar(1000));
    checkNum("rho2 (air)",   f.mixture.phases.rho2, scalar(1));
    checkNum("sigma",        f.mixture.sigma,       scalar(0.07));
    checkNum("cAlpha, through the regex key \"alpha.water.*\"", f.interface.cAlpha, scalar(1));
    checkNum("nAlphaCorr",      scalar(f.alphaCtl.nAlphaCorr),      scalar(2));
    checkNum("nAlphaSubCycles", scalar(f.alphaCtl.nAlphaSubCycles), scalar(1));
    check   ("MULESCorr yes",   f.alphaCtl.MULESCorr);
    checkNum("nLimiterIter (CMULES reads it with NO default)",
             scalar(f.mulesCtl.nLimiterIter), scalar(5));
    check   ("adjustTimeStep yes", f.timeCtl.base.adjustTimeStep);
    checkNum("maxCo",      f.timeCtl.base.maxCo,     scalar(1));
    checkNum("maxAlphaCo", f.timeCtl.maxAlphaCo,     scalar(1));
    checkNum("maxDeltaT",  f.timeCtl.base.maxDeltaT, scalar(1));
    check   ("div(rhoPhi,U) is `Gauss linearUpwind grad(U)`", f.divRhoPhiU == DivScheme::linearUpwind);
    check   ("div(phi,alpha) is `Gauss vanLeer`",   f.divPhiAlpha   == AlphaFluxScheme::vanLeer);
    check   ("div(phirb,alpha) is `Gauss linear`",  f.divPhirbAlpha == AlphaFluxScheme::linear);
    check   ("ddt(alpha) is Euler",                 f.ddtAlpha      == AlphaDdt::Euler);

    // ---- 2. gravity -------------------------------------------------------------------------------
    checkNum("g.y", f.g.y, scalar(-9.81));
    checkNum("damBreak sets no constant/hRef, so hRef is 0", f.hRef,       scalar(0));
    checkNum("...and ghRef with it",                         f.ghRefValue, scalar(0));

    // ---- 3. the fields ----------------------------------------------------------------------------
    const label nC = m.nCells();
    checkNum("alpha1 has one value per cell", scalar(f.alpha1.internal.size()), scalar(nC));
    checkNum("alpha2 too",                    scalar(f.alpha2.size()),          scalar(nC));

    // setFields paints a column of water: alpha is exactly 0 or 1, and BOTH must be present.
    label nWater = 0, nAir = 0, nOther = 0;
    for (scalar a : f.alpha1.internal)
    {
        if (a == scalar(1))      ++nWater;
        else if (a == scalar(0)) ++nAir;
        else                     ++nOther;
    }
    std::printf("  setFields left %ld water cells, %ld air cells, %ld in between\n",
                (long)nWater, (long)nAir, (long)nOther);
    check("setFields painted a column of water", nWater > 0);
    check("...and the rest is air", nAir > 0);
    check("...with a SHARP interface -- no partially filled cells", nOther == 0);

    // rho follows alpha exactly: 1000 in the water, 1 in the air, nothing between.
    scalar rhoMin = f.rho[0], rhoMax = f.rho[0];
    for (scalar r : f.rho) { rhoMin = std::fmin(rhoMin, r); rhoMax = std::fmax(rhoMax, r); }
    checkNum("rho is 1 where alpha is 0",    rhoMin, scalar(1));
    checkNum("...and 1000 where alpha is 1", rhoMax, scalar(1000));
    for (label c = 0; c < nC; ++c)
        if (f.alpha1.internal[c] == scalar(1) && f.rho[c] != scalar(1000))
        { check("every water cell carries water's density", false); break; }

    // mu = alpha*rho1*nu1 + (1-alpha)*rho2*nu2
    scalar worstMu = 0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar a = f.alpha1.internal[c];
        const scalar want = a*scalar(1000)*scalar(1e-6) + (scalar(1)-a)*scalar(1)*scalar(1.48e-5);
        worstMu = std::fmax(worstMu, std::fabs(f.mu[c] - want));
    }
    checkNum("mu is the alpha-weighted blend", worstMu, scalar(0), scalar(1e-18));

    // gh = (g & C) - ghRef, and p = p_rgh + rho*gh
    scalar worstGh = 0, worstP = 0;
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar wantGh = (f.g.x*C.x + f.g.y*C.y + f.g.z*C.z) - f.ghRefValue;
        worstGh = std::fmax(worstGh, std::fabs(f.gh[c] - wantGh));
        worstP  = std::fmax(worstP,  std::fabs(f.p[c] - (f.p_rgh.internal[c] + f.rho[c]*f.gh[c])));
    }
    checkNum("gh = (g & C) - ghRef",       worstGh, scalar(0), scalar(1e-12));
    checkNum("p  = p_rgh + rho*gh",        worstP,  scalar(0), scalar(1e-12));
    check   ("...and p is NOT p_rgh, because the water column weighs something",
             std::fabs(f.p[0] - f.p_rgh.internal[0]) + std::fabs(f.p[nC-1] - f.p_rgh.internal[nC-1])
             > scalar(1e-6));

    // ---- 4. phi: read on a restart, computed on a cold start --------------------------------------
    check("a cold start has no phi on disk, so it is computed from U",
          !f.phiWasRead && !std::filesystem::exists(startDir + "/phi"));
    checkNum("...one value per internal face",
             scalar(f.phi.internal.size()), scalar(m.nInternalFaces()));
    // U is zero everywhere at t=0 on damBreak, so phi is too -- assert that rather than a magnitude,
    // because a non-zero phi here would mean the fallback read something it should not have.
    scalar maxPhi = 0;
    for (scalar s : f.phi.internal) maxPhi = std::fmax(maxPhi, std::fabs(s));
    checkNum("damBreak starts from rest, so the initial flux is zero", maxPhi, scalar(0), scalar(1e-30));

    // ...and rhoPhi is built from it, so it is zero too, but PRESENT -- UEqn reads it on the first
    // outer iteration before alphaEqn has ever written it.
    checkNum("rhoPhi exists from the start", scalar(f.rhoPhi.internal.size()),
             scalar(m.nInternalFaces()));

    std::printf("test_inter_case_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
