// runInterFoamDevice -- the same interFoam time loop, on the GPU.
//
// See inter_driver_cpp.cuh for why the hooks live here rather than in the gate. Everything this calls
// is separately landed and separately gated; what this file owns is the wiring, and that wiring is
// what tests/test_device_inter_dambreak_alpha.cu measures against the host driver on damBreak.
#include "inter_driver_cpp.cuh"
#include "inter_case_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "inter_ueqn_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include "two_phase_mixture_cpp.cuh"
#include "solution_directions.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "device_inter_step.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

std::vector<scalar> flattenPatches(const std::vector<std::vector<scalar>>& b)
{
    std::vector<scalar> v;
    for (const auto& p : b) v.insert(v.end(), p.begin(), p.end());
    return v;
}

std::vector<scalar> patchValues(const GeometricField<scalar>& f, const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> v;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& b = f.boundary[pi]->value();
        v.insert(v.end(), b.begin(), b.end());
    }
    return v;
}

std::vector<scalar> fullFace(const SurfaceScalarField& f, const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> v(f.internal);
    for (std::size_t pi = 0; pi < fvp.size() && pi < f.boundary.size(); ++pi)
        v.insert(v.end(), f.boundary[pi].begin(), f.boundary[pi].end());
    return v;
}

// OpenFOAM's directionMixed evaluate for the flux-conditional velocity patches. evaluateBoundary()
// alone does not resolve them, and their matrix coefficients are built from the value it leaves --
// see the note in pressureCorrector, and tools/dumpInterFoam for OpenFOAM's own numbers.
void updateVelocityPatches(GeometricField<vector>& U, const std::vector<FvPatch>& fvp)
{
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        std::vector<vector> Uc(static_cast<std::size_t>(fvp[pi].size), vector{0, 0, 0});
        for (label i = 0; i < fvp[pi].size; ++i)
            Uc[i] = U.internal[fvp[pi].faceCells[i]];
        U.boundary[pi]->updateFromPatchVelocity(U.boundary[pi]->value(), Uc, {});
    }
}

}   // namespace


RunReport runInterFoamDevice(const std::string&          caseDir,
                             const std::string&          startDir,
                             const PrimitiveMesh&        m,
                             const FvGeometry&           g,
                             const std::vector<FvPatch>& fvp,
                             label                       nSteps,
                             bool                        verbose,
                             InterFields*                fieldsOut)
{
    InterFields f = buildInterFields(caseDir, startDir, m, g, fvp);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const label nFaces = static_cast<label>(g.magSf().size());
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    if (f.timeCtl.base.adjustTimeStep)
        throw std::runtime_error(
            "brae interFoam (device): `adjustTimeStep yes` is not wired into the device loop yet. The "
            "VoF Courant number itself is ported (deviceAlphaCourantNo) but nothing here feeds it back "
            "into deltaT, and running with a fixed step while the case asks for an adaptive one would "
            "silently solve a different problem.");

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // the masks the device needs that the mesh does not carry
    std::vector<int> aFixes, aFlag, takeU, uFixes;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const int fx = f.alpha1.boundary[pi]->fixesValue() ? 1 : 0;
        const int fl = (fvp[pi].type == "empty") ? 1 : ((fvp[pi].type == "wedge") ? 2 : 0);
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            aFixes.push_back(fx);
            aFlag.push_back(fl);
            takeU.push_back(f.U.boundary[pi]->assignable() ? 0 : 1);
            uFixes.push_back(f.U.boundary[pi]->fixesValue() ? 1 : 0);
        }
    }
    DeviceBuffer<int> dAFixes(aFixes), dAFlag(aFlag), dTakeU(takeU), dUFixes(uFixes);

    // ---- the hooks: every one is per-patch host work, and nothing else ----------------------------
    DeviceInterStepHooks H;
    H.alpha.updateBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd, DeviceBuffer<scalar>& nBnd)
    {
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        aBnd.copyFrom(patchValues(f.alpha1, fvp));
        SurfaceScalarField nHb;
        std::vector<scalar> Kb;
        interfaceProps::calculateK(f.alpha1, f.interface, m, g, fvp, false, nHb, Kb);
        nBnd.copyFrom(flattenPatches(nHb.boundary));
    };
    H.alpha.divCoeffs =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        FvScalarMatrix M = fvm::div<scalar>(f.phi.internal, f.phi.boundary, f.alpha1, m, fvp);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(M.internalCoeffs[pi][i]); b2.push_back(M.boundaryCoeffs[pi][i]); }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };
    H.updateUBoundary =
        [&](const DeviceBuffer<scalar>& ux, const DeviceBuffer<scalar>& uy,
            const DeviceBuffer<scalar>& uz, DeviceVectorBoundary& db, DeviceBuffer<scalar>* ubOut)
    {
        std::vector<scalar> x, y, z;
        ux.copyTo(x); uy.copyTo(y); uz.copyTo(z);
        for (label c = 0; c < nC; ++c) f.U.internal[c] = vector{x[c], y[c], z[c]};
        f.U.evaluateBoundary();
        updateVelocityPatches(f.U, fvp);
        db = buildDeviceVectorBoundary(f.U, fvp, g);
        if (!ubOut) return;
        std::vector<scalar> bx, by, bz;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (const vector& u : f.U.boundary[pi]->value())
            { bx.push_back(u.x); by.push_back(u.y); bz.push_back(u.z); }
        ubOut[0].copyFrom(bx);
        ubOut[1].copyFrom(by);
        ubOut[2].copyFrom(bz);
    };
    H.interfaceForces =
        [&](const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& Kd,
            const DeviceBuffer<scalar>& rd, DeviceBuffer<scalar>& stf, DeviceBuffer<scalar>& snRho,
            DeviceBuffer<scalar>& nuC, DeviceBuffer<scalar>& nuB, DeviceBuffer<scalar>& snP)
    {
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        Kd.copyTo(f.K);
        rd.copyTo(f.rho);

        // surfaceTensionForce() = interpolate(sigma*K)*snGrad(alpha1), boundary INCLUDED -- at a
        // contact-angle wall that boundary term is the contact angle's only route into the equations.
        std::vector<scalar> sK;
        interfaceProps::sigmaK(f.K, f.interface.sigma, sK);
        const SurfaceScalarField sKf = fvc::interpolate(sK, m, g, fvp);
        const SurfaceScalarField snA = fvc::snGrad(f.alpha1, m, g, fvp, false);
        SurfaceScalarField t;
        t.internal.resize(static_cast<std::size_t>(nIf));
        for (label i = 0; i < nIf; ++i) t.internal[i] = sKf.internal[i]*snA.internal[i];
        t.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                t.boundary[pi].push_back(sKf.boundary[pi][i]*snA.boundary[pi][i]);
        stf.copyFrom(fullFace(t, fvp));

        GeometricField<scalar> rhoF;
        rhoF.internal = f.rho;
        for (const FvPatch& q : fvp)
            rhoF.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        rhoF.evaluateBoundary();
        snRho.copyFrom(fullFace(fvc::snGrad(rhoF, m, g, fvp, false), fvp));

        // the mixture's own nu. NOTE mixtureNu's second argument is mu, not alpha2.
        cpu::twoPhase::mixtureMu(f.alpha1.internal, f.mixture.phases, f.mu);
        cpu::twoPhase::mixtureNu(f.alpha1.internal, f.mu, f.mixture.phases, f.nu);
        nuC.copyFrom(f.nu);
        updateMixtureBoundary(f, fvp);
        nuB.copyFrom(flattenPatches(f.nuBnd));

        // snGrad(p_rgh) is read ONLY when the case runs a momentum predictor; it is explicit there and
        // implicit in the pressure equation, and carrying it into both would count it twice.
        if (f.momentumPredictorOn)
        {
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                if (f.p_rgh.boundary[pi]->updateableSnGrad())
                    f.p_rgh.boundary[pi]->updateSnGrad(
                        std::vector<scalar>(static_cast<std::size_t>(fvp[pi].size), scalar(0)));
            snP.copyFrom(fullFace(fvc::snGrad(f.p_rgh, m, g, fvp, false), fvp));
        }
        else snP.resize(0);
    };
    H.pressure.pressureCoeffs =
        [&](const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>& phiHB,
            const DeviceBuffer<scalar>& rAUfAll, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        std::vector<scalar> hB, rA;
        phiHB.copyTo(hB);
        rAUfAll.copyTo(rA);
        // constrainPressure: a fixedFluxPressure gradient is PRESCRIBED from phiHbyA, and brae refuses
        // to assemble one that has not been set. rAUf is taken PER FACE from the full array.
        label off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const FvPatch& q = fvp[pi];
            if (f.p_rgh.boundary[pi]->updateableSnGrad())
            {
                std::vector<scalar> sn(static_cast<std::size_t>(q.size));
                for (label i = 0; i < q.size; ++i)
                    sn[i] = (hB[off + i] - f.phi.boundary[pi][i]) / (q.magSf[i] * rA[nIf + off + i]);
                f.p_rgh.boundary[pi]->updateSnGrad(sn);
            }
            off += q.size;
        }
        SurfaceScalarField rf;
        rf.internal.assign(rA.begin(), rA.begin() + nIf);
        rf.boundary.resize(fvp.size());
        { label o = nIf;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i) rf.boundary[pi].push_back(rA[o++]); }
        FvScalarMatrix pe = fvm::laplacian<scalar>(rf, f.p_rgh, m, g, fvp, false);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(pe.internalCoeffs[pi][i]); b2.push_back(pe.boundaryCoeffs[pi][i]); }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };
    H.pressure.updateBoundary = [&](const DeviceBuffer<scalar>& pr)
    {
        pr.copyTo(f.p_rgh.internal);
        f.p_rgh.evaluateBoundary();
    };

    // ---- controls --------------------------------------------------------------------------------
    DeviceInterStepControls C;
    C.alpha.nAlphaSubCycles = static_cast<int>(f.alphaCtl.nAlphaSubCycles);
    C.alpha.nAlphaCorr      = static_cast<int>(f.alphaCtl.nAlphaCorr);
    C.alpha.MULESCorr       = f.alphaCtl.MULESCorr;
    C.alpha.preSolve.tol    = scalar(1e-12);
    C.alpha.preSolve.maxIter = 2000;
    C.mules = DeviceMulesControls{f.mulesCtl.nLimiterIter, f.mulesCtl.smoothLimiter,
                                  f.mulesCtl.extremaCoeff, f.mulesCtl.boundaryExtremaCoeff};
    C.alphaInput.cAlpha = f.interface.cAlpha;
    C.alphaInput.deltaN = interfaceProps::deltaN(g.V());
    C.alphaInput.alphaScheme  = (f.divPhiAlpha == AlphaFluxScheme::linear) ? DeviceAlphaScheme::linear
                              : (f.divPhiAlpha == AlphaFluxScheme::upwind) ? DeviceAlphaScheme::upwind
                              : DeviceAlphaScheme::vanLeer;
    C.alphaInput.alpharScheme = (f.divPhirbAlpha == AlphaFluxScheme::vanLeer) ? DeviceAlphaScheme::vanLeer
                              : (f.divPhirbAlpha == AlphaFluxScheme::upwind) ? DeviceAlphaScheme::upwind
                              : DeviceAlphaScheme::linear;
    switch (f.divRhoPhiU)
    {
        case DivScheme::linearUpwind:   C.divScheme = brae::cpu::DivScheme::linearUpwind;   break;
        case DivScheme::linearUpwindV:  C.divScheme = brae::cpu::DivScheme::linearUpwindV;  break;
        case DivScheme::limitedLinear:  C.divScheme = brae::cpu::DivScheme::limitedLinear;  break;
        case DivScheme::limitedLinearV: C.divScheme = brae::cpu::DivScheme::limitedLinearV; break;
        case DivScheme::LUST:           C.divScheme = brae::cpu::DivScheme::LUST;           break;
        default:                        C.divScheme = brae::cpu::DivScheme::upwind;         break;
    }
    C.divSchemeCoeff = f.divRhoPhiUCoeff;
    C.nCorrectors = static_cast<int>(f.pimple.nCorrectors);
    C.momentumPredictor = f.momentumPredictorOn;
    C.relaxU = f.relaxU;
    C.relaxEquationU = f.relaxEquationU;
    C.pressure.tol = f.tolP;
    C.pressure.relTol = f.relTolP;
    C.pressure.maxIter = f.maxIterP;
    C.momentum.tol = scalar(1e-12);
    C.takeUAtBoundary = &dTakeU;
    { const SolutionDirections sd = solutionDirections(fvp);
      for (int k = 0; k < 3; ++k) C.solutionD[k] = sd.d[k]; }

    DevicePhaseProperties props{f.mixture.phases.rho1, f.mixture.phases.nu1,
                                f.mixture.phases.rho2, f.mixture.phases.nu2};

    // ---- the state on the device -----------------------------------------------------------------
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    { ux[c] = f.U.internal[c].x; uy[c] = f.U.internal[c].y; uz[c] = f.U.internal[c].z; }

    DeviceBuffer<scalar> dA(f.alpha1.internal), dAOld(f.alpha1.internal);
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dUox(ux), dUoy(uy), dUoz(uz);
    DeviceBuffer<scalar> dPhiI(f.phi.internal), dPhiB(flattenPatches(f.phi.boundary));
    DeviceBuffer<scalar> dPrgh(f.p_rgh.internal), dP;
    DeviceBuffer<scalar> dNH(f.nHatf.internal), dNHB(flattenPatches(f.nHatf.boundary));
    DeviceBuffer<scalar> dABnd(patchValues(f.alpha1, fvp)), dK(f.K);
    DeviceBuffer<scalar> dGh(f.gh), dGhf, dMagSf(g.magSf());
    { SurfaceScalarField gf; gf.internal = f.ghfInternal; gf.boundary = f.ghfBoundary;
      dGhf.copyFrom(fullFace(gf, fvp)); }
    DeviceBuffer<scalar> dRho, dMu, dNu, dRpI, dRpB;
    DeviceVectorBoundary dbU = buildDeviceVectorBoundary(f.U, fvp, g);

    RunReport rep;
    rep.deltaT = f.deltaT;
    for (label s = 0; s < nSteps; ++s)
    {
        std::vector<scalar> ca, cx, cy, cz;
        dA.copyTo(ca);   dAOld.copyFrom(ca);
        dUx.copyTo(cx);  dUy.copyTo(cy);  dUz.copyTo(cz);
        dUox.copyFrom(cx); dUoy.copyFrom(cy); dUoz.copyFrom(cz);
        std::vector<scalar> poi, pob;
        dPhiI.copyTo(poi); dPhiB.copyTo(pob);
        DeviceBuffer<scalar> dPhiOI(poi), dPhiOB(pob);

        deviceInterStep(dm, f.deltaT, C, props, H, dGh, dGhf, dMagSf,
                        dA, dAOld, dUx, dUy, dUz, dUox, dUoy, dUoz,
                        dPhiI, dPhiB, dPhiOI, dPhiOB, dUFixes, dPrgh, dP,
                        dNH, dNHB, dABnd, dK, dAFixes, dAFlag, dbU,
                        dRho, dMu, dNu, dRpI, dRpB);
        rep.steps = s + 1;
        rep.time += f.deltaT;

        if (verbose)
        {
            std::vector<scalar> av;
            dA.copyTo(av);
            scalar lo = av.empty() ? 0 : av[0], hi = lo;
            for (scalar v : av) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
            std::printf("   t = %.6f  dt = %.3e  [device]  alpha [%.3e, %.6f]\n",
                        (double)rep.time, (double)f.deltaT, (double)lo, (double)hi);
        }
    }

    // hand the device's answer back through the host fields, so a caller compares the same objects
    dA.copyTo(f.alpha1.internal);
    f.alpha1.evaluateBoundary();
    { std::vector<scalar> x, y, z;
      dUx.copyTo(x); dUy.copyTo(y); dUz.copyTo(z);
      for (label c = 0; c < nC; ++c) f.U.internal[c] = vector{x[c], y[c], z[c]};
      f.U.evaluateBoundary(); }
    dPrgh.copyTo(f.p_rgh.internal);
    f.p_rgh.evaluateBoundary();
    dP.copyTo(f.p);
    dRho.copyTo(f.rho);
    dPhiI.copyTo(f.phi.internal);

    // ...and the REPORT's own numbers. Leaving maxU and worstDivPhi at their defaults printed
    // "max|U| 0 m/s, worst |div(phi)| 0.000e+00" on a run whose U reaches 0.27 -- a false statement in
    // the solver's own output, and the kind a reader trusts because the rest of the line is right.
    rep.maxU = 0;
    for (label c = 0; c < nC; ++c)
        rep.maxU = std::fmax(rep.maxU, std::sqrt(f.U.internal[c].x*f.U.internal[c].x
                                               + f.U.internal[c].y*f.U.internal[c].y
                                               + f.U.internal[c].z*f.U.internal[c].z));
    {
        std::vector<scalar> pb;
        dPhiB.copyTo(pb);
        SurfaceScalarField phiOut;
        phiOut.internal = f.phi.internal;
        phiOut.boundary.resize(fvp.size());
        { label o = 0;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          { for (label i = 0; i < fvp[pi].size; ++i) phiOut.boundary[pi].push_back(pb[o + i]);
            o += fvp[pi].size; } }
        f.phi.boundary = phiOut.boundary;
        const std::vector<scalar> d = fvc::div(phiOut, m, g, fvp);
        rep.worstDivPhi = 0;
        for (label c = 0; c < nC; ++c) rep.worstDivPhi = std::fmax(rep.worstDivPhi, std::fabs(d[c]));
    }

    std::vector<scalar> av;
    dA.copyTo(av);
    rep.alphaMin = av.empty() ? 0 : av[0];
    rep.alphaMax = rep.alphaMin;
    rep.alphaMass = 0;
    for (label c = 0; c < nC; ++c)
    {
        rep.alphaMin = std::fmin(rep.alphaMin, av[c]);
        rep.alphaMax = std::fmax(rep.alphaMax, av[c]);
        rep.alphaMass += av[c]*g.V()[c];
    }
    (void)nFaces;
    (void)nBf;
    if (fieldsOut) *fieldsOut = std::move(f);
    return rep;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
