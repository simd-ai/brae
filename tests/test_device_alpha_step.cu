// A WHOLE alphaEqn.H on the device -- the corrector loop driven here, as a real caller drives it --
// held against the host step and against MULES' own contract.
//
// Every operator this chains is already gated on its own -- the compression flux and the nested alpha
// flux bit-identically (test_device_alpha_flux), the interface normal and curvature to 3.8e-15
// (test_device_interface_properties), the donor flux bit-identically and the limiter to 3.7e-15
// (test_device_mules). What is NOT covered by any of those is the ORDER, and in alphaEqn.H the order
// is physics: phic before phir, phir from the nHatf the PREVIOUS corrector left, alpha2 rebuilt from
// the CURRENT alpha1, and mixture.correct() at the BOTTOM of the corrector. Arm 5 below breaks exactly
// that last one and measures the damage, so the file has a control for the thing it is testing rather
// than for the pieces underneath it.
//
// THE FIXTURE IS A RIGID ROTATION, and that is not decoration. MULES' whole claim is boundedness under
// a DIVERGENCE-FREE flux; feed it one that is not and alpha leaves [0,1] legitimately, which is how an
// earlier fixture in this tree reported a 3.06e-02 excursion that was the fixture's fault and not the
// limiter's. On an orthogonal box a rigid rotation is discretely divergence-free EXACTLY -- u is linear
// so sum_f u(Cf).Sf collapses to omega.(sum_f Cf x Sf), and on a hex every Cf - C is parallel to its own
// Sf -- and arm 0 measures that before anything else runs. It also keeps the blob inside the domain, so
// sum(V*alpha) is conserved and arm 4 can say so.
//
// THE COMPARISON WITH THE HOST IS MEASURED, NOT BIT-EXACT: MULES' five face-to-cell budgets are gathers
// here and a serial face loop there, so they sum in a different order and lambda inherits that. What
// the measurement says is that it barely matters -- 1.1e-16 after one step and 9.99e-16 after forty of
// two correctors each, which is four ULP of a field of order 1. The limiter absorbs the difference
// rather than amplifying it, which is not obvious in advance for a nonlinear limiter and is the reason
// arm 3 prints the number instead of only asserting a bound.
//
// THE BOUNDARY IS EVALUATED ON THE HOST between CORRECTORS, which is the contract
// device_alpha_step.cuh states -- but from the DEVICE's own alpha, never from the host run's. The two
// trajectories never touch; if they did, this file would be measuring a copy. Doing it only between
// STEPS was the first version of this file, and arm 3a below caught it: one corrector agreed with the
// host to 2.2e-16 and two to 5.7e-08, which was corrector 2 reading the boundary corrector 1 started
// from. That is why the nAlphaCorr loop is here and not inside the call.
#include "box_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "alpha_eqn_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include "device_alpha_presolve.cuh"
#include "fvm.cuh"
#include "mules_cpp.cuh"
#include "device_alpha_step.cuh"
#include "device_alpha_flux.cuh"
#include "device_interface_properties.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
namespace ifm   = brae::cpu::interFoam;
namespace ip    = brae::cpu::interfaceProps;
namespace mules = brae::cpu::MULES;

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

// a GeometricField's evaluated patch values, in the mesh's boundary-face order
std::vector<scalar> flatten2(const GeometricField<scalar>& f, const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> v;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& b = f.boundary[pi]->value();
        v.insert(v.end(), b.begin(), b.end());
    }
    return v;
}
}   // namespace

int main()
{
    std::printf("== a whole alphaEqn step: device vs host ==\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 24;
    const scalar h = scalar(1) / scalar(N);
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1, scalar(0), h, h, h);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    // u = omega x (r - centre), omega along z. Zero on the z faces, so the 2-D box stays 2-D.
    const scalar omega = scalar(2), cx = scalar(0.5), cy = scalar(0.5);
    auto vel = [&](const vector& P) { return vector{-omega*(P.y - cy), omega*(P.x - cx), scalar(0)}; };
    SurfaceScalarField phi;
    phi.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const vector u = vel(g.Cf()[f]), &S = g.Sf()[f];
        phi.internal[f] = u.x*S.x + u.y*S.y + u.z*S.z;
    }
    phi.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phi.boundary[pi].resize(static_cast<std::size_t>(fvp[pi].size));
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label gf = fvp[pi].start + i;
            const vector u = vel(g.Cf()[gf]), &S = g.Sf()[gf];
            phi.boundary[pi][i] = u.x*S.x + u.y*S.y + u.z*S.z;
        }
    }

    // ---- 0. THE FIXTURE ITSELF: is the flux discretely divergence-free? ---------------------------
    // fvc::div returns sum(phi)/V, so the scale this must be small against is |phi|/V, not 1.
    {
        std::vector<scalar> d(static_cast<std::size_t>(nC), scalar(0));
        for (label f = 0; f < nIf; ++f) { d[m.owner()[f]] += phi.internal[f]; d[m.neighbour()[f]] -= phi.internal[f]; }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                d[fvp[pi].faceCells[i]] += phi.boundary[pi][i];
        scalar worst = 0, scale = 0;
        for (label c = 0; c < nC; ++c) worst = std::fmax(worst, std::fabs(d[c]/g.V()[c]));
        for (label f = 0; f < nIf; ++f) scale = std::fmax(scale, std::fabs(phi.internal[f]));
        scale /= g.V()[0];
        std::printf("  the flux: worst |div phi| = %.3e against a scale of %.3e\n",
                    (double)worst, (double)scale);
        check("the rotation flux is discretely divergence-free, so boundedness means something",
              worst < scalar(1e-12)*scale);
    }

    // a blob off-centre, so the rotation carries it somewhere
    GeometricField<scalar> aH;
    aH.internal.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::hypot(C.x - scalar(0.35), C.y - scalar(0.5));
        aH.internal[c] = scalar(0.5)*(scalar(1) - std::tanh((r - scalar(0.15))/scalar(0.03)));
    }
    for (const FvPatch& q : fvp)
        aH.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    aH.evaluateBoundary();
    const std::vector<scalar> a0 = aH.internal;

    ip::InterfaceCoeffs ic;
    ic.cAlpha = scalar(1);
    mules::Controls mctl;
    mctl.nLimiterIter = 5;      // damBreak's own value

    const int nAlphaCorr = 2;
    const scalar dt = scalar(0.01);              // Co = 0.24 on this mesh
    const int nSteps = 40;                       // ~0.8 rad, which carries the interface ~4 widths
    const scalar rho1 = scalar(1000), rho2 = scalar(1);

    ifm::AlphaStepInput in;
    in.phi = &phi;
    in.phiCN = &phi;
    in.cAlpha = ic.cAlpha;
    in.nAlphaCorr = nAlphaCorr;
    in.rho1 = rho1;
    in.rho2 = rho2;
    in.deltaT = dt;
    in.MULESCorr = false;
    in.alphaScheme  = ifm::AlphaFluxScheme::vanLeer;
    in.alpharScheme = ifm::AlphaFluxScheme::linear;

    // ---- the HOST trajectory ----------------------------------------------------------------------
    auto runHost = [&](int steps, bool mulesCorr)
    {
        ifm::AlphaStepInput hin = in;
        hin.MULESCorr = mulesCorr;
        hin.tolAlpha = scalar(1e-12);
        hin.relTolAlpha = 0;
        hin.maxIterAlpha = 2000;
        GeometricField<scalar> a;
        a.internal = a0;
        for (const FvPatch& q : fvp)
            a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        a.evaluateBoundary();
        SurfaceScalarField nHatf, alphaPhi, rhoPhi;
        std::vector<scalar> K;
        ip::calculateK(a, ic, m, g, fvp, /*gradLeastSquares=*/false, nHatf, K);
        for (int s = 0; s < steps; ++s)
        {
            const std::vector<scalar> old = a.internal;
            ifm::alphaEqnStep(a, old, hin, ic, mctl, m, g, fvp, alphaPhi, rhoPhi, nHatf, K, nullptr);
        }
        return a.internal;
    };

    // ---- the DEVICE trajectory, from the same start and never reading the host's ------------------
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<int> fixes, flag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const int fx = (aH.boundary[pi]->fixesValue()) ? 1 : 0;
        const int fl = (fvp[pi].type == "empty") ? 1 : ((fvp[pi].type == "wedge") ? 2 : 0);
        for (label i = 0; i < fvp[pi].size; ++i) { fixes.push_back(fx); flag.push_back(fl); }
    }
    DeviceBuffer<scalar> dPhiInt(phi.internal), dPhiBnd(flatten(phi.boundary));
    DeviceBuffer<int>    dFixes(fixes), dFlag(flag);

    DeviceAlphaStepInput din;
    din.phiInt = &dPhiInt;
    din.phiBnd = &dPhiBnd;
    din.phiCNInt = &dPhiInt;
    din.phiCNBnd = &dPhiBnd;
    din.cAlpha = ic.cAlpha;
    din.deltaT = dt;
    din.rho1 = rho1;
    din.rho2 = rho2;
    din.deltaN = ip::deltaN(g.V());
    din.alphaScheme  = DeviceAlphaScheme::vanLeer;
    din.alpharScheme = DeviceAlphaScheme::linear;
    DeviceMulesControls dmcBase;
    dmcBase.nLimiterIter = mctl.nLimiterIter;

    // `freezeNHatf` is arm 5's control: it keeps the interface normal at its INITIAL value instead of
    // letting the step's own mixture.correct() rewrite it, which is the one ordering fact this file
    // owns. Both runs are otherwise identical.
    scalar worstPreSolveExcursion = 0;      // filled by the MULESCorr path, read by arm 6
    auto runDevice = [&](int steps, bool freezeNHatf, scalar cAlpha, bool mulesCorr = false,
                         scalar preSolveTol = scalar(1e-12), int nLimiterIter = 0)
    {
        DeviceMulesControls dmc = dmcBase;
        if (nLimiterIter > 0) dmc.nLimiterIter = nLimiterIter;
        DeviceAlphaStepInput li = din;
        li.cAlpha = cAlpha;
        li.MULESCorr = mulesCorr;

        // The boundary is evaluated on the host from the DEVICE's own field, which is the split
        // device_alpha_step.cuh states: branchy per-patch dispatch stays off the GPU. It is rebuilt
        // from `work` every step, and `work` is only ever filled from the device.
        GeometricField<scalar> work;
        work.internal = a0;
        for (const FvPatch& q : fvp)
            work.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        work.evaluateBoundary();

        SurfaceScalarField nH;
        std::vector<scalar> K0;
        ip::calculateK(work, ic, m, g, fvp, false, nH, K0);

        DeviceBuffer<scalar> dA(a0), dAOld(a0), dNHatf(nH.internal), dK(K0);
        DeviceBuffer<scalar> dABnd(flatten2(work, fvp)), dNHatfBnd(flatten(nH.boundary));
        DeviceBuffer<scalar> aPhiInt, aPhiBnd, rPhiInt, rPhiBnd;

        // the host's evaluate step, run on whatever the device last wrote -- alpha1's patch values and
        // the boundary interface normal, which is where correctContactAngle would act on a case that
        // had one. `freezeNHatf` is arm 5's control and skips only the normal.
        // ...followed by mixture.correct() -- alphaEqn.H:225 -- whose INTERNAL half runs on the
        // device and whose boundary normals come from the host, where correctContactAngle lives.
        auto correctBoundaryAndMixture = [&]()
        {
            dA.copyTo(work.internal);
            work.evaluateBoundary();
            dABnd.copyFrom(flatten2(work, fvp));
            if (freezeNHatf) return;      // arm 5: leave nHatf where it was

            SurfaceScalarField nHb;
            std::vector<scalar> Kb;
            ip::calculateK(work, ic, m, g, fvp, false, nHb, Kb);
            dNHatfBnd.copyFrom(flatten(nHb.boundary));
            deviceInterfaceCorrect(dm, dA, dABnd, dNHatfBnd, li.deltaN, dNHatf, dK);
        };

        for (int s = 0; s < steps; ++s)
        {
            // alphaEqn.H resets alpha1 to its old time ONCE per sub-cycle, not once per corrector.
            dAOld.copyFrom(work.internal);
            dA.copyFrom(work.internal);

            if (mulesCorr)
            {
                // alphaEqn.H:103-155, ONCE before the correctors: the implicit upwind pre-solve, then
                // a mixture.correct() of its own. The boundary coefficients come from the host's
                // fvm::div, as deviceAlphaPreSolve's header requires.
                GeometricField<scalar> ab;
                ab.internal = work.internal;
                for (const FvPatch& q : fvp)
                    ab.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
                ab.evaluateBoundary();
                FvScalarMatrix Mb = fvm::div<scalar>(phi.internal, phi.boundary, ab, m, fvp);
                std::vector<scalar> iCv, bCv;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    for (label i = 0; i < fvp[pi].size; ++i)
                    {
                        iCv.push_back(Mb.internalCoeffs[pi][i]);
                        bCv.push_back(Mb.boundaryCoeffs[pi][i]);
                    }
                DeviceBuffer<scalar> dIC(iCv), dBC(bCv);
                DeviceAlphaSolverControls psc;
                psc.tol     = preSolveTol;
                psc.relTol  = 0;
                psc.maxIter = 2000;
                deviceAlphaPreSolve(dm, dA, dAOld, dPhiInt, dIC, dBC, dt, psc, aPhiInt, aPhiBnd);
                {
                    std::vector<scalar> pv;
                    dA.copyTo(pv);
                    failures += brae::gatecheck::nonFinite("pv", pv);
                    for (label c = 0; c < nC; ++c)
                        worstPreSolveExcursion = std::fmax(worstPreSolveExcursion,
                            std::fmax(-pv[c], pv[c] - scalar(1)));
                }
                correctBoundaryAndMixture();          // alphaEqn.H:151-153
            }

            for (int aCorr = 0; aCorr < nAlphaCorr; ++aCorr)
            {
                li.aCorr = aCorr;
                DeviceAlphaBoundary db;
                db.alpha1     = &dABnd;
                db.nHatfBnd   = &dNHatfBnd;
                db.fixesValue = &dFixes;
                db.flag       = &dFlag;
                deviceAlphaCorrector(dm, dA, dAOld, li, db, dmc, dNHatf, aPhiInt, aPhiBnd);
                correctBoundaryAndMixture();
            }
            deviceMassFlux(nIf, aPhiInt, dPhiInt, rho1, rho2, rPhiInt);
            deviceMassFlux(nBf, aPhiBnd, dPhiBnd, rho1, rho2, rPhiBnd);
        }
        std::vector<scalar> out;
        dA.copyTo(out);
        failures += brae::gatecheck::nonFinite("out", out);
        return out;
    };

    const std::vector<scalar> dev = runDevice(nSteps, /*freezeNHatf=*/false, ic.cAlpha);
    const std::vector<scalar> hst = runHost(nSteps, /*mulesCorr=*/false);
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }

    // ---- 1. THE PROPERTY, asserted exactly: MULES keeps alpha in [0,1] ----------------------------
    {
        scalar exc = 0;
        for (label c = 0; c < nC; ++c) exc = std::fmax(exc, std::fmax(-dev[c], dev[c] - scalar(1)));
        std::printf("  after %d device steps: worst excursion from [0,1] = %.3e\n", nSteps, (double)exc);
        check("the device step keeps alpha in [0,1] -- MULES' whole contract", exc <= scalar(1e-14));
    }

    // ---- 2. it actually advected, so arms 1 and 3 are not satisfied by doing nothing --------------
    {
        scalar moved = 0;
        for (label c = 0; c < nC; ++c) moved = std::fmax(moved, std::fabs(dev[c] - a0[c]));
        std::printf("  ...and the field moved by up to %.4f\n", (double)moved);
        check("the interface actually moved", moved > scalar(0.5));
    }

    // ---- 3a. ONE step, where nothing has had time to drift -----------------------------------------
    // This is the arm that says the step is RIGHT. After a single step the only difference between the
    // two sides is the order MULES' budgets are summed in, so the gap is at round-off; anything larger
    // here would be a term, not an accumulation, and arm 3 below could not tell the two apart.
    {
        const std::vector<scalar> d1 = runDevice(1, false, ic.cAlpha);
        const std::vector<scalar> h1 = runHost(1, /*mulesCorr=*/false);
        scalar worst = 0;
        for (label c = 0; c < nC; ++c) worst = std::fmax(worst, std::fabs(d1[c] - h1[c]));
        std::printf("  after ONE step: worst |device - host| = %.4e\n", (double)worst);
        check("one step agrees with the host at round-off", worst < scalar(1e-14));
    }

    // ---- 3. agreement with the HOST step, measured --------------------------------------------------
    {
        scalar worst = 0;
        label iw = -1;
        for (label c = 0; c < nC; ++c)
        {
            const scalar e = std::fabs(dev[c] - hst[c]);
            if (e > worst) { worst = e; iw = c; }
        }
        std::printf("  alpha after %d steps: worst |device - host| = %.4e", nSteps, (double)worst);
        if (iw >= 0) std::printf("  (device %.17g, host %.17g)", (double)dev[iw], (double)hst[iw]);
        std::printf("\n");
        // MEASURED 9.99e-16 after 40 steps of two correctors each -- four ULP of a field of order 1,
        // which is to say the two trajectories have not drifted apart at all. The bound is 1e-13, two
        // orders of headroom over that and twelve below the 2.3e-01 arms 4 and 5 need to discriminate.
        // It was 1e-10 while this file's first version had the curvature update inside the corrector;
        // the bound tightened when that was fixed, and does not loosen again.
        check("the device trajectory matches the host's to 1e-13", worst < scalar(1e-13));
    }

    // ---- 4. THE COMPRESSION IS A REAL PART OF WHAT IS BEING COMPARED ------------------------------
    // With cAlpha = 0 the whole nested flux vanishes and the step is plain advection. If that gave the
    // same answer, arms 1-3 would be passing on a code path that never ran.
    {
        const std::vector<scalar> noComp = runDevice(nSteps, false, scalar(0));
        scalar d = 0;
        for (label c = 0; c < nC; ++c) d = std::fmax(d, std::fabs(noComp[c] - dev[c]));
        std::printf("  turning the compression off moves the answer by %.4e\n", (double)d);
        check("the compressive term is a real part of the answer", d > scalar(1e-2));

        scalar exc = 0;
        for (label c = 0; c < nC; ++c) exc = std::fmax(exc, std::fmax(-noComp[c], noComp[c] - scalar(1)));
        check("...and alpha is bounded without it too, so arm 1 is not the compression's doing",
              exc <= scalar(1e-14));
    }

    // ---- 5. THE FAIL-PROOF FOR THIS FILE'S OWN CLAIM: the normal is carried, not recomputed -------
    // Put nHatf back to its t = 0 value at the top of every step, so each step's first corrector
    // compresses towards where the interface was at the start of the run rather than where the last
    // corrector left it. The answer must then move by far more than arm 3's bound -- which is to say
    // that mixture.correct() at the BOTTOM of the corrector is load-bearing, and that a version of
    // this step which computed the normal at the TOP from a stale field would be caught here. That
    // ordering is the one thing this file adds over the four component gates underneath it.
    {
        const std::vector<scalar> frozen = runDevice(nSteps, /*freezeNHatf=*/true, ic.cAlpha);
        scalar d = 0;
        for (label c = 0; c < nC; ++c) d = std::fmax(d, std::fabs(frozen[c] - dev[c]));
        std::printf("  freezing nHatf at its initial value moves the answer by %.4e\n", (double)d);
        check("rewriting nHatf inside the corrector is load-bearing, so the order is under test",
              d > scalar(1e-2));
    }

    // ---- 6. damBreak's OWN PATH: MULESCorr, the implicit pre-solve and CMULES -------------------
    // 13 of the 44 shipped tutorials set `MULESCorr yes`; damBreak is one, at nAlphaCorr 2 and
    // nLimiterIter 5, which is what this arm runs. Every piece is separately gated -- the pre-solve in
    // tests/test_device_alpha_presolve.cu, CMULES in tests/test_device_mules_corr.cu -- so what is
    // under test here is again the ORDER: the pre-solve once per sub-cycle and not once per corrector,
    // a mixture.correct() of its own between it and the first corrector (alphaEqn.H:151-153), the
    // correction taken against the flux the pre-solve left rather than against nothing, and the
    // under-relaxation of BOTH the field and the flux from the second corrector onward.
    {
        worstPreSolveExcursion = 0;
        const std::vector<scalar> devC = runDevice(nSteps, false, ic.cAlpha, /*mulesCorr=*/true);
        const scalar excPre = worstPreSolveExcursion;
        worstPreSolveExcursion = 0;
        const std::vector<scalar> devLoose =
            runDevice(nSteps, false, ic.cAlpha, /*mulesCorr=*/true, /*preSolveTol=*/scalar(1e-6));
        const scalar excPreLoose = worstPreSolveExcursion;
        scalar excLoose = 0;
        for (label c = 0; c < nC; ++c)
            excLoose = std::fmax(excLoose, std::fmax(-devLoose[c], devLoose[c] - scalar(1)));
        std::printf("  pre-solve's own worst excursion: %.3e at tol 1e-12, %.3e at tol 1e-6 "
                    "(final field at 1e-6: %.3e)\n",
                    (double)excPre, (double)excPreLoose, (double)excLoose);
        check("the implicit pre-solve is EXACTLY bounded when it is solved exactly",
              excPre <= scalar(1e-14));
        check("...and its excursion tracks the linear solver's tolerance, so that is where a loose "
              "pre-solve leaks", excPreLoose > scalar(1e-9));

        // ...and the same sweep in nLimiterIter, which is where the FINAL field's excursion comes
        // from: MULES is a fixed-point iteration and its bound is exact only in the limit.
        const std::vector<scalar> devIter =
            runDevice(nSteps, false, ic.cAlpha, true, scalar(1e-12), /*nLimiterIter=*/40);
        scalar excIter = 0;
        for (label c = 0; c < nC; ++c)
            excIter = std::fmax(excIter, std::fmax(-devIter[c], devIter[c] - scalar(1)));
        const std::vector<scalar> hstC = runHost(nSteps, /*mulesCorr=*/true);

        scalar exc = 0, excH = 0, worst = 0, moved = 0;
        for (label c = 0; c < nC; ++c)
        {
            exc   = std::fmax(exc, std::fmax(-devC[c], devC[c] - scalar(1)));
            excH  = std::fmax(excH, std::fmax(-hstC[c], hstC[c] - scalar(1)));
            worst = std::fmax(worst, std::fabs(devC[c] - hstC[c]));
            moved = std::fmax(moved, std::fabs(devC[c] - a0[c]));
        }
        std::printf("  MULESCorr, %d steps x %d correctors: excursion device %.3e / host %.3e, "
                    "worst |device - host| %.4e, the field moved %.4f\n",
                    nSteps, nAlphaCorr, (double)exc, (double)excH, (double)worst, (double)moved);
        std::printf("  ...and at nLimiterIter 40 instead of %d the excursion is %.3e\n",
                    (int)mctl.nLimiterIter, (double)excIter);
        // NOT "exactly bounded", because it is not, and the host says the same number to three
        // digits. CMULES bounds the CORRECTION against a budget it reaches by fixed-point iteration,
        // so the bound is exact only in the limit -- unlike the explicit path in arm 1, which comes
        // back at 0.000e+00 and whose assertion stays exact. What is asserted here is what holds: the
        // excursion is small, it is the host's, and it SHRINKS when the iteration is given more
        // passes. An arm that demanded [0,1] exactly here would have to be satisfied by weakening
        // something real, and an arm that merely bounded it at 1e-5 would not notice a limiter that
        // had stopped iterating at all.
        check("the semi-implicit path holds alpha to within 1e-5 of [0,1]", exc <= scalar(1e-5));
        check("...the SAME excursion the host reaches, so it is CMULES' and not the device's",
              std::fabs(exc - excH) < scalar(1e-3)*exc);
        check("...and more limiter passes shrink it, which is what makes it the iteration's",
              excIter < scalar(0.5)*exc);
        check("...and tracks the host over forty steps", worst < scalar(1e-9));
        check("...having actually advected", moved > scalar(0.5));

        // THE DISCRIMINATOR: the two paths must not agree. If they did, this arm would be re-testing
        // the explicit one -- the implicit pre-solve advances alpha with first-order upwind before the
        // correctors ever run, so the answer after forty steps is a different field.
        scalar apart = 0;
        for (label c = 0; c < nC; ++c) apart = std::fmax(apart, std::fabs(devC[c] - dev[c]));
        std::printf("  ...and it differs from the explicit path by %.4e\n", (double)apart);
        check("MULESCorr gives a different answer from the explicit path, so this arm is not a "
              "second copy of arm 3", apart > scalar(1e-2));
    }

    std::printf("test_device_alpha_step: %d failures\n", failures);
    return failures ? 1 : 0;
}
