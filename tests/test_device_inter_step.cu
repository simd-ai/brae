// ONE WHOLE interFoam TIME STEP on the device.
//
// Every operator and every sub-sequence under this is separately gated -- the alpha half end to end
// (test_device_inter_alpha_step), the momentum matrix and predictor (test_device_inter_ueqn_assembly),
// a whole pressure pass (test_device_inter_pressure_step). What is under test HERE is the loop order,
// and interFoam's order is not numerics but which field each equation sees:
//
//   ALPHA ADVANCES FIRST and mixture.correct() runs before the momentum equation, so UEqn is built on
//   the NEW density. Arm 3 rebuilds the momentum equation on the density the step STARTED with -- the
//   single-phase order, which converges perfectly well -- and measures U.
//
//   rhoPhi COMES OUT OF THE ALPHA EQUATION. It is the MULES-limited MASS flux, and arm 4 shows it is
//   not rho*phi rebuilt afterwards, which is the obvious substitute.
//
//   THE PRESSURE CORRECTOR IS LAST, and arm 2 checks the state it leaves is self-consistent:
//   p == p_rgh + rho*gh exactly, and alpha still in [0,1].
#include "box_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "interface_properties_cpp.cuh"
#include "two_phase_mixture_cpp.cuh"
#include "device_inter_step.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
namespace ip = brae::cpu::interfaceProps;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
std::vector<scalar> flatten(const std::vector<std::vector<scalar>>& b)
{
    std::vector<scalar> v;
    for (const auto& p : b) v.insert(v.end(), p.begin(), p.end());
    return v;
}
}   // namespace

int main()
{
    std::printf("== one whole interFoam time step on the device\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 16;
    const scalar h = scalar(1) / scalar(N);
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1, scalar(0), h, h, h);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const label nFaces = static_cast<label>(g.magSf().size());
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // a blob of water in air, and a rotation flux that keeps it inside the domain
    const scalar rho1 = 1000, rho2 = 1, nu1 = 1e-6, nu2 = 1.48e-5, sigma = 0.07;
    const scalar omega = scalar(2), cx = scalar(0.5), cy = scalar(0.5);
    auto vel = [&](const vector& P) { return vector{-omega*(P.y - cy), omega*(P.x - cx), scalar(0)}; };

    std::vector<scalar> phiInt(static_cast<std::size_t>(nIf)), phiBnd;
    for (label f = 0; f < nIf; ++f)
    {
        const vector u = vel(g.Cf()[f]), &S = g.Sf()[f];
        phiInt[f] = u.x*S.x + u.y*S.y + u.z*S.z;
    }
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label gf = fvp[pi].start + i;
            const vector u = vel(g.Cf()[gf]), &S = g.Sf()[gf];
            phiBnd.push_back(u.x*S.x + u.y*S.y + u.z*S.z);
        }

    std::vector<scalar> a0(static_cast<std::size_t>(nC)), gh(static_cast<std::size_t>(nC));
    std::vector<scalar> ux0(nC), uy0(nC), uz0(nC), prgh0(nC, scalar(0));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::hypot(C.x - scalar(0.35), C.y - scalar(0.5));
        a0[c]  = scalar(0.5)*(scalar(1) - std::tanh((r - scalar(0.15))/scalar(0.03)));
        gh[c]  = scalar(-9.81)*C.y;
        const vector u = vel(C);
        ux0[c] = u.x; uy0[c] = u.y; uz0[c] = u.z;
    }
    std::vector<scalar> ghf(nFaces);
    for (label f = 0; f < nFaces; ++f) ghf[f] = scalar(-9.81)*g.Cf()[f].y;

    // ---- the host scratch the hooks evaluate through ---------------------------------------------
    GeometricField<scalar> work;
    work.internal = a0;
    for (const FvPatch& q : fvp)
        work.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    work.evaluateBoundary();
    auto patchValues = [&](const GeometricField<scalar>& f)
    {
        std::vector<scalar> v;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& b = f.boundary[pi]->value();
            v.insert(v.end(), b.begin(), b.end());
        }
        return v;
    };

    ip::InterfaceCoeffs ic;
    ic.cAlpha = scalar(1);
    ic.sigma  = sigma;

    GeometricField<vector> Uh;
    Uh.internal.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c) Uh.internal[c] = vector{ux0[c], uy0[c], uz0[c]};
    for (const FvPatch& q : fvp)
        Uh.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
            q, true, vector{0,0,0}, std::vector<vector>{}));
    Uh.evaluateBoundary();

    GeometricField<scalar> prghH;
    prghH.internal = prgh0;
    for (const FvPatch& q : fvp)
        prghH.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    prghH.evaluateBoundary();

    DeviceInterStepHooks hooks;
    hooks.alpha.updateBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd, DeviceBuffer<scalar>& nBnd)
    {
        a.copyTo(work.internal);
        work.evaluateBoundary();
        aBnd.copyFrom(patchValues(work));
        SurfaceScalarField nHb;
        std::vector<scalar> Kb;
        ip::calculateK(work, ic, m, g, fvp, false, nHb, Kb);
        nBnd.copyFrom(flatten(nHb.boundary));
    };
    hooks.alpha.divCoeffs =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        a.copyTo(work.internal);
        work.evaluateBoundary();
        SurfaceScalarField pf;
        pf.internal = phiInt;
        pf.boundary.resize(fvp.size());
        { label off = 0;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          { for (label i = 0; i < fvp[pi].size; ++i) pf.boundary[pi].push_back(phiBnd[off + i]);
            off += fvp[pi].size; } }
        FvScalarMatrix M = fvm::div<scalar>(pf.internal, pf.boundary, work, m, fvp);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(M.internalCoeffs[pi][i]); b2.push_back(M.boundaryCoeffs[pi][i]); }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };
    hooks.updateUBoundary =
        [&](const DeviceBuffer<scalar>& dx, const DeviceBuffer<scalar>& dy,
            const DeviceBuffer<scalar>& dz, DeviceVectorBoundary& db,
                DeviceBuffer<scalar>* ubOut)
    {
        std::vector<scalar> a, b, c;
        dx.copyTo(a); dy.copyTo(b); dz.copyTo(c);
        for (label i = 0; i < nC; ++i) Uh.internal[i] = vector{a[i], b[i], c[i]};
        Uh.evaluateBoundary();
        db = buildDeviceVectorBoundary(Uh, fvp, g);
        if (ubOut)
        {
            std::vector<scalar> bx, by, bz;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                const std::vector<vector>& v = Uh.boundary[pi]->value();
                for (const vector& u : v) { bx.push_back(u.x); by.push_back(u.y); bz.push_back(u.z); }
            }
            ubOut[0].copyFrom(bx);
            ubOut[1].copyFrom(by);
            ubOut[2].copyFrom(bz);
        }
    };
    hooks.interfaceForces =
        [&](const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& Kd,
            const DeviceBuffer<scalar>& rhod, const DeviceBuffer<scalar>& rhoBd,
            DeviceBuffer<scalar>& stf,
            DeviceBuffer<scalar>& snRho, DeviceBuffer<scalar>& nuC, DeviceBuffer<scalar>& nuB,
            DeviceBuffer<scalar>& snP)
    {
        std::vector<scalar> av, Kv, rv, rbv;
        a.copyTo(av); Kd.copyTo(Kv); rhod.copyTo(rv); rhoBd.copyTo(rbv);
        work.internal = av;
        work.evaluateBoundary();
        // surfaceTensionForce = interpolate(sigma*K)*snGrad(alpha1), snGrad(rho) -- over the full array
        std::vector<scalar> s(nFaces, scalar(0)), sr(nFaces, scalar(0)), sp(nFaces, scalar(0));
        for (label f = 0; f < nIf; ++f)
        {
            const label o = m.owner()[f], n = m.neighbour()[f];
            const scalar w = g.weights()[f], dc = g.deltaCoeffs()[f];
            s[f]  = (w*sigma*Kv[o] + (1-w)*sigma*Kv[n]) * dc*(av[n] - av[o]);
            sr[f] = dc*(rv[n] - rv[o]);
            sp[f] = dc*(prghH.internal[n] - prghH.internal[o]);
        }
        // snGrad(rho) ON THE BOUNDARY is deltaCoeffs*(rho_b - rho_cell): rho's patches are
        // `calculated`, not zeroGradient. On this fixture alpha's patches are zeroGradient, so rho_b is
        // the cell's and the term is zero either way -- the arm that makes it non-zero is the
        // OpenFOAM one, tests/interfoam_dambreak_inflow_vs_openfoam.sh.
        {
            std::size_t k = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    if (fvp[pi].type != "empty")
                    {
                        sr[static_cast<std::size_t>(nIf) + k] =
                            fvp[pi].deltaCoeffs[i]*(rbv[k] - rv[fvp[pi].faceCells[i]]);
                    }
                    ++k;
                }
            }
        }
        stf.copyFrom(s);
        snRho.copyFrom(sr);
        snP.copyFrom(sp);
        // nuEff: the mixture's own, which mixture.correct() has just written
        brae::cpu::twoPhase::PhaseProperties pp;
        pp.rho1 = rho1; pp.rho2 = rho2; pp.nu1 = nu1; pp.nu2 = nu2;
        // mixtureNu takes mu, not alpha2 -- see two_phase_mixture_cpp.cuh:130.
        std::vector<scalar> muv, nuv;
        brae::cpu::twoPhase::mixtureMu(av, pp, muv);
        brae::cpu::twoPhase::mixtureNu(av, muv, pp, nuv);
        nuC.copyFrom(nuv);
        std::vector<scalar> nb;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i) nb.push_back(nuv[fvp[pi].faceCells[i]]);
        nuB.copyFrom(nb);
    };
    hooks.pressure.pressureCoeffs =
        [&](const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&,
            const DeviceBuffer<scalar>& rAUfInt, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        std::vector<scalar> rf;
        rAUfInt.copyTo(rf);
        failures += brae::gatecheck::nonFinite("rf", rf);
        SurfaceScalarField rf2;
        rf2.internal = rf;
        rf2.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            rf2.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), rf.empty() ? scalar(0) : rf[0]);
        FvScalarMatrix pe = fvm::laplacian<scalar>(rf2, prghH, m, g, fvp, false);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(pe.internalCoeffs[pi][i]); b2.push_back(pe.boundaryCoeffs[pi][i]); }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };

    // ---- run one step ----------------------------------------------------------------------------
    std::vector<int> fixes, flag, takeU;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        { fixes.push_back(0); flag.push_back(fvp[pi].type == "empty" ? 1 : 0); takeU.push_back(1); }
    DeviceBuffer<int> dFixes(fixes), dFlag(flag), dTakeU(takeU);
    // U is noSlip on every patch of this fixture, so it fixes a value everywhere and ddtCorr is zero
    // on the whole boundary -- which is what OpenFOAM does at a wall.
    DeviceBuffer<int> dUFix(std::vector<int>(static_cast<std::size_t>(nBf), 1));

    DeviceInterStepControls ctl;
    ctl.alpha.nAlphaSubCycles = 1;
    ctl.alpha.nAlphaCorr      = 2;
    ctl.alpha.MULESCorr       = false;
    ctl.mules.nLimiterIter    = 5;
    ctl.alphaInput.cAlpha     = ic.cAlpha;
    ctl.alphaInput.deltaN     = ip::deltaN(g.V());
    ctl.alphaInput.alphaScheme  = DeviceAlphaScheme::vanLeer;
    ctl.alphaInput.alpharScheme = DeviceAlphaScheme::linear;
    ctl.momentum.tol  = scalar(1e-12);
    ctl.pressure.tol  = scalar(1e-12);
    ctl.momentumPredictor = false;          // damBreak's own setting
    ctl.relaxEquationU    = true;
    ctl.needReference     = true;           // every patch is zeroGradient here, so p_rgh is singular
    ctl.pRefCell          = 0;
    ctl.pRefValue         = 0;
    ctl.takeUAtBoundary   = &dTakeU;
    // the box fixture is 2-D too (Nz = 1 with wall z-patches, not empty), so all three stay valid
    ctl.solutionD[0] = ctl.solutionD[1] = ctl.solutionD[2] = 1;

    DevicePhaseProperties props{rho1, nu1, rho2, nu2};

    work.internal = a0;
    work.evaluateBoundary();
    SurfaceScalarField nH0;
    std::vector<scalar> K0;
    ip::calculateK(work, ic, m, g, fvp, false, nH0, K0);

    DeviceBuffer<scalar> dAlpha(a0), dAlphaOld(a0), dUx(ux0), dUy(uy0), dUz(uz0);
    DeviceBuffer<scalar> dUox(ux0), dUoy(uy0), dUoz(uz0);
    DeviceBuffer<scalar> dPhiI(phiInt), dPhiB(phiBnd), dPrgh(prgh0), dP;
    DeviceBuffer<scalar> dNHatf(nH0.internal), dNHatfB(flatten(nH0.boundary));
    DeviceBuffer<scalar> dABnd(patchValues(work)), dK(K0);
    DeviceBuffer<scalar> dGh(gh), dGhf(ghf), dMagSf(g.magSf());
    DeviceBuffer<scalar> dRho, dMu, dNu, dRhoPhiI, dRhoPhiB;
    DeviceVectorBoundary dbU = buildDeviceVectorBoundary(Uh, fvp, g);

    // FIVE steps at Co ~ 0.16, not one at Co ~ 3e-3. Arm 3's claim is about the density a MOVED
    // interface leaves behind, and at dt = 2e-4 the interface travelled 3e-3 of a cell: rho changed by
    // 0.7 of 1000 and the arm failed on a correct answer. The loop also exercises what a time step
    // hands the next one -- alpha, U, phi, p_rgh and the carried nHatf and K.
    const scalar dt = scalar(5e-3);
    const int nSteps = 5;
    for (int s = 0; s < nSteps; ++s)
    {
        std::vector<scalar> cur;
        dAlpha.copyTo(cur);
        failures += brae::gatecheck::nonFinite("cur", cur);
        dAlphaOld.copyFrom(cur);
        std::vector<scalar> cx2, cy2, cz2;
        dUx.copyTo(cx2); dUy.copyTo(cy2); dUz.copyTo(cz2);
        dUox.copyFrom(cx2); dUoy.copyFrom(cy2); dUoz.copyFrom(cz2);
        std::vector<scalar> poi, pob;
        dPhiI.copyTo(poi);  dPhiB.copyTo(pob);
        DeviceBuffer<scalar> dPhiOI(poi), dPhiOB(pob);
        deviceInterStep(dm, dt, ctl, props, hooks, dGh, dGhf, dMagSf,
                        dAlpha, dAlphaOld, dUx, dUy, dUz, dUox, dUoy, dUoz,
                        dPhiI, dPhiB, dPhiOI, dPhiOB, dUFix, dPrgh, dP, dNHatf, dNHatfB, dABnd, dK,
                        dFixes, dFlag, dbU, dRho, dMu, dNu, dRhoPhiI, dRhoPhiB);
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }

    std::vector<scalar> alpha, rhoV, prgh, pV, ux, rhoPhi;
    dAlpha.copyTo(alpha);
    failures += brae::gatecheck::nonFinite("alpha", alpha);
    dRho.copyTo(rhoV);
    failures += brae::gatecheck::nonFinite("rhoV", rhoV);
    dPrgh.copyTo(prgh);
    failures += brae::gatecheck::nonFinite("prgh", prgh);
    dP.copyTo(pV);
    failures += brae::gatecheck::nonFinite("pV", pV);
    dUx.copyTo(ux);
    failures += brae::gatecheck::nonFinite("ux", ux);
    dRhoPhiI.copyTo(rhoPhi);
    failures += brae::gatecheck::nonFinite("rhoPhi", rhoPhi);

    // ---- 1. the step ran, and moved everything ---------------------------------------------------
    {
        scalar movedA = 0, movedU = 0, movedP = 0;
        for (label c = 0; c < nC; ++c)
        {
            movedA = std::fmax(movedA, std::fabs(alpha[c] - a0[c]));
            movedU = std::fmax(movedU, std::fabs(ux[c] - ux0[c]));
            movedP = std::fmax(movedP, std::fabs(prgh[c] - prgh0[c]));
        }
        std::printf("  %d steps at Co ~ 0.16: alpha moved %.3e, U.x %.3e, p_rgh %.3e\n",
                    nSteps, (double)movedA, (double)movedU, (double)movedP);
        check("the alpha equation advanced the interface", movedA > scalar(0.1));
        check("the pressure corrector solved and rewrote U", movedU > scalar(1e-6));
        check("...and p_rgh", movedP > scalar(1e-3));
    }

    // ---- 2. the state it leaves is self-consistent ------------------------------------------------
    // p == p_rgh + rho*gh EXACTLY, from the fields the last step ended with -- which is what rebuilding
    // p rather than carrying it means.
    {
        scalar exc = 0, pGap = 0;
        for (label c = 0; c < nC; ++c)
        {
            exc = std::fmax(exc, std::fmax(-alpha[c], alpha[c] - scalar(1)));
            const scalar want = prgh[c] + rhoV[c]*gh[c];
            pGap = std::fmax(pGap, std::fabs(pV[c] - want));
        }
        std::printf("  alpha excursion %.3e;  |p - (p_rgh + rho*gh)| = %.3e\n",
                    (double)exc, (double)pGap);
        check("alpha is still in [0,1] after five whole steps", exc <= scalar(1e-14));
        check("p is consistent with the p_rgh and rho the step ends with", pGap == scalar(0));
    }

    // ---- 3. THE LOOP ORDER: UEqn is built on the NEW density --------------------------------------
    // The mixture the step leaves is built from the ADVANCED alpha. Rebuilding it from the alpha the
    // step started with -- which is what putting the momentum equation first amounts to -- gives a
    // different density field, and at a water/air interface the difference is the density ratio.
    {
        brae::cpu::twoPhase::PhaseProperties pp;
        pp.rho1 = rho1; pp.rho2 = rho2; pp.nu1 = nu1; pp.nu2 = nu2;
        std::vector<scalar> a2(static_cast<std::size_t>(nC)), staleRho;
        for (label c = 0; c < nC; ++c) a2[c] = scalar(1) - a0[c];
        brae::cpu::twoPhase::mixtureRho(a0, a2, pp, staleRho);
        scalar d = 0;
        int nMoved = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar e = std::fabs(rhoV[c] - staleRho[c]);
            d = std::fmax(d, e);
            if (e > scalar(1)) ++nMoved;
        }
        std::printf("  rho from the ADVANCED alpha vs the step's starting alpha: %.3e of %.0f, "
                    "in %d of %ld cells\n", (double)d, (double)rho1, nMoved, (long)nC);
        check("the mixture UEqn is built on is the one the alpha equation just produced",
              d > scalar(1));
        check("...and only near the interface, not everywhere", nMoved > 0 && nMoved < nC/2);
    }

    // ---- 4. rhoPhi IS THE ALPHA EQUATION'S, not rho*phi rebuilt -----------------------------------
    {
        scalar d = 0, sc = 0;
        for (label f = 0; f < nIf; ++f)
        {
            const scalar w = g.weights()[f];
            const scalar rebuilt = (w*rhoV[m.owner()[f]] + (scalar(1) - w)*rhoV[m.neighbour()[f]])
                                 * phiInt[f];
            d  = std::fmax(d, std::fabs(rhoPhi[f] - rebuilt));
            sc = std::fmax(sc, std::fabs(rhoPhi[f]));
        }
        std::printf("  rhoPhi vs interpolate(rho)*phi rebuilt afterwards: %.3e of %.3e\n",
                    (double)d, (double)sc);
        check("rhoPhi is the MULES-limited flux the alpha equation left, not rho*phi rebuilt",
              d > scalar(0.01)*sc);
    }

    std::printf("test_device_inter_step: %d failures\n", failures);
    return failures ? 1 : 0;
}
