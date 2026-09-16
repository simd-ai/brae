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
#include "foam_field_reader.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "pbicgstab.cuh"
#include <memory>
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

    // ---- 5. ONE ALPHA STEP ON THE REAL CASE -------------------------------------------------------
    // The first time any of this advects anything on a mesh OpenFOAM built. What is asserted is what
    // holds for ANY flux on ANY mesh -- boundedness, exactly -- plus non-vacuity: the interface has to
    // have MOVED, or the bound is satisfied by doing nothing.
    //
    // CONSERVATION IS NOT ASSERTED HERE, and the reason is the fixture, not the code. An exactly
    // conservative arm needs a discretely divergence-free flux; the analytic vortex below is
    // divergence-free as a continuous field but its pointwise face fluxes are not, so alpha's total
    // legitimately drifts by that residual. The measured divergence is printed beside the drift so the
    // two can be compared, and the exact conservation claim stays where the fixture supports it --
    // tests/test_mules_cpp.cu, on a flux built to close.
    {
        // damBreak sets MULESCorr yes, so the very first thing to check is that brae says so.
        {
            GeometricField<scalar> a = buildField<scalar>(
                readField<scalar>(startDir + "/" + f.alphaName), patches, nC);
            a.evaluateBoundary();
            AlphaStepInput in;
            in.phi = &f.phi; in.phiCN = &f.phi;
            in.nAlphaCorr = f.alphaCtl.nAlphaCorr;
            in.MULESCorr  = f.alphaCtl.MULESCorr;
            in.rho1 = f.mixture.phases.rho1; in.rho2 = f.mixture.phases.rho2;
            in.deltaT = f.deltaT;
            SurfaceScalarField aPhi, rPhi;
            bool threw = false;
            try
            {
                alphaEqnStep(a, f.alpha1.internal, in, f.interface, f.mulesCtl,
                             m, g, patches, aPhi, rPhi);
            }
            catch (const std::exception&) { threw = true; }
            check("damBreak's `MULESCorr yes` is refused by name -- the implicit pre-solve is not wired",
                  threw && f.alphaCtl.MULESCorr);
        }

        // ...and now the explicit path, which 30 of the 44 shipped tutorials take.
        // A VORTEX from a stream function that vanishes on the bounding box, so the velocity is
        // tangential at every wall: psi = A sin(pi x/Lx) sin(pi y/Ly), U = (dpsi/dy, -dpsi/dx).
        vector lo = g.C()[0], hi = g.C()[0];
        for (label c = 0; c < nC; ++c)
        {
            lo.x = std::fmin(lo.x, g.C()[c].x); hi.x = std::fmax(hi.x, g.C()[c].x);
            lo.y = std::fmin(lo.y, g.C()[c].y); hi.y = std::fmax(hi.y, g.C()[c].y);
        }
        const scalar Lx = hi.x - lo.x, Ly = hi.y - lo.y, A = scalar(0.05);
        auto Uat = [&](const vector& X)
        {
            const scalar sx = std::sin(scalar(M_PI)*(X.x - lo.x)/Lx);
            const scalar cx = std::cos(scalar(M_PI)*(X.x - lo.x)/Lx);
            const scalar sy = std::sin(scalar(M_PI)*(X.y - lo.y)/Ly);
            const scalar cy = std::cos(scalar(M_PI)*(X.y - lo.y)/Ly);
            return vector{ A*sx*cy*scalar(M_PI)/Ly, -A*cx*sy*scalar(M_PI)/Lx, scalar(0)};
        };

        SurfaceScalarField phi;
        phi.internal.resize(static_cast<std::size_t>(m.nInternalFaces()));
        for (label fi = 0; fi < m.nInternalFaces(); ++fi)
        {
            const vector U = Uat(g.Cf()[fi]);
            const vector& S = g.Sf()[fi];
            phi.internal[fi] = U.x*S.x + U.y*S.y + U.z*S.z;
        }
        phi.boundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            phi.boundary[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(0));

        // ...AND THEN PROJECTED ONTO THE DIVERGENCE-FREE SPACE, because MULES's bound assumes one.
        //
        // This was found by measuring, not by reasoning: the pointwise vortex flux above is
        // divergence-free as a CONTINUOUS field and emphatically not as a discrete one -- |div(phi)|
        // came out at 2.15e+01 -- and alpha left [0,1] by 3.1e-02 within 20 steps. That is not a MULES
        // defect: interFoam's explicit solve passes divU as a zeroField (alphaEqn.H:217), i.e. it
        // ASSUMES the flux closes, and OpenFOAM would go out of bounds on the same input. A boundedness
        // gate fed a divergent flux tests nothing but the fixture.
        //
        // The projection is CorrectPhi's: solve laplacian(1, pcorr) == div(phi) with every patch
        // zeroGradient (the flux is tangential at all of them) and a reference cell to pin the
        // singular system, then subtract the resulting face flux.
        scalar divBefore = 0, worstDiv = 0;
        {
            auto worstOf = [&](const SurfaceScalarField& p)
            {
                const std::vector<scalar> d = fvc::div(p, m, g, patches);
                scalar w = 0;
                for (scalar v : d) w = std::fmax(w, std::fabs(v));
                return w;
            };
            divBefore = worstOf(phi);

            GeometricField<scalar> pcorr;
            pcorr.internal.assign(static_cast<std::size_t>(nC), scalar(0));
            for (const FvPatch& q : patches)
                pcorr.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
            pcorr.evaluateBoundary();

            SurfaceScalarField one;
            one.internal.assign(static_cast<std::size_t>(m.nInternalFaces()), scalar(1));
            one.boundary.resize(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                one.boundary[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(1));

            FvScalarMatrix pe = fvm::laplacian<scalar>(one, pcorr, m, g, patches, /*corrected=*/false);
            const std::vector<scalar> dv = fvc::div(phi, m, g, patches);
            for (label c = 0; c < nC; ++c) pe.source[c] += dv[c]*g.V()[c];
            // an all-zeroGradient laplacian is singular: pin one cell, as fvMatrix::setReference does
            pe.source[0] += pe.diag[0]*scalar(0);
            pe.diag[0]   += pe.diag[0];
            pbicgstab(pe, pcorr.internal, m, patches, scalar(1e-14), scalar(0), 2000);
            pcorr.evaluateBoundary();

            const SurfaceScalarField corr = matrixFlux(pe, pcorr.internal, m, patches);
            for (label fi = 0; fi < m.nInternalFaces(); ++fi) phi.internal[fi] -= corr.internal[fi];
            worstDiv = worstOf(phi);
            std::printf("  vortex flux: worst |div(phi)| %.3e before projection, %.3e after\n",
                        (double)divBefore, (double)worstDiv);
            check("the projection actually made the flux divergence-free",
                  worstDiv < scalar(1e-6)*std::fmax(scalar(1), divBefore));
        }

        GeometricField<scalar> alpha = buildField<scalar>(
            readField<scalar>(startDir + "/" + f.alphaName), patches, nC);
        alpha.evaluateBoundary();

        AlphaStepInput in;
        in.phi = &phi; in.phiCN = &phi;
        in.cAlpha = f.interface.cAlpha;
        in.nAlphaCorr = f.alphaCtl.nAlphaCorr;
        in.rho1 = f.mixture.phases.rho1; in.rho2 = f.mixture.phases.rho2;
        in.alphaScheme  = f.divPhiAlpha;
        in.alpharScheme = f.divPhirbAlpha;
        in.MULESCorr = false;                 // the path this arm exercises
        in.deltaT = scalar(2e-3);

        auto mass = [&](const std::vector<scalar>& a)
        {
            scalar s = 0;
            for (label c = 0; c < nC; ++c) s += a[c]*g.V()[c];
            return s;
        };
        const scalar mass0 = mass(alpha.internal);
        const std::vector<scalar> start = alpha.internal;

        scalar worstExcursion = 0;
        const label nSteps = 20;
        for (label step = 0; step < nSteps; ++step)
        {
            const std::vector<scalar> old = alpha.internal;
            SurfaceScalarField aPhi, rPhi;
            alphaEqnStep(alpha, old, in, f.interface, f.mulesCtl, m, g, patches, aPhi, rPhi);
            for (scalar v : alpha.internal)
                worstExcursion = std::fmax(worstExcursion, std::fmax(-v, v - scalar(1)));
        }

        scalar moved = 0;
        for (label c = 0; c < nC; ++c) moved = std::fmax(moved, std::fabs(alpha.internal[c] - start[c]));
        const scalar drift = std::fabs(mass(alpha.internal) - mass0) / mass0;

        std::printf("  %ld alpha steps on damBreak's mesh, vortex flux:\n", (long)nSteps);
        std::printf("    worst excursion outside [0,1] = %.3e\n", (double)worstExcursion);
        std::printf("    largest change in any cell    = %.4f\n", (double)moved);
        std::printf("    relative mass drift           = %.3e  (worst |div(phi)| = %.3e)\n",
                    (double)drift, (double)worstDiv);

        // THE BOUND IS THE FIXTURE'S OWN RESIDUAL, not a round number. The projection leaves
        // |div(phi)| at some small epsilon; each step can therefore inject at most epsilon*deltaT of
        // alpha into a cell, and nSteps of them at most nSteps*epsilon*deltaT. Measured: the residual
        // divergence is 2.4e-12, which predicts 9.7e-14, and the observed excursion is 9.5e-14. So
        // what is left is the projection's residue and nothing else -- and the bound TIGHTENS on its
        // own if the projection is ever improved, which a hand-picked constant would not.
        const scalar permitted = scalar(2) * worstDiv * in.deltaT * static_cast<scalar>(nSteps);
        std::printf("    the residual divergence permits at most %.3e; observed %.3e\n",
                    (double)permitted, (double)worstExcursion);
        check("alpha stays in [0,1] to the level the projection's residual allows, and no further",
              worstExcursion <= permitted);
        check("...and the interface actually MOVED, so the bound is not satisfied by doing nothing",
              moved > scalar(0.1));
    }

    std::printf("test_inter_case_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
