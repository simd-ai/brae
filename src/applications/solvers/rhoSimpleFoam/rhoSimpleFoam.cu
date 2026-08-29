// CUDA driver for rhoSimpleFoam. See rhoSimpleFoam.cuh for the provenance, the order and the contract.
#include "rhoSimpleFoam.cuh"
#include "pEqn.cuh"              // correctVelocity, relaxField -- the stages that ARE shared
#include "device_pcg.cuh"
#include "device_blas.cuh"
#include "device_amg.cuh"
#include "device_simple.cuh"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

DeviceLduView foldedView(const DeviceMesh& dm, const PressureMatrix& P, const DeviceBuffer<scalar>& diagC)
{
    DeviceLduView A{};
    A.nCells = dm.nCells;
    A.nInternalFaces = dm.nInternalFaces;
    A.diag = diagC.data();
    A.upper = P.upper.data();
    A.lower = P.lower.data();
    A.owner = dm.owner.data();
    A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data();
    A.losortStart = dm.losortStart.data();
    return A;
}

DeviceLduView foldedViewM(const DeviceMesh& dm, const MomentumMatrix& M, const DeviceBuffer<scalar>& diagC)
{
    DeviceLduView A{};
    A.nCells = dm.nCells;
    A.nInternalFaces = dm.nInternalFaces;
    A.diag = diagC.data();
    A.upper = M.upper.data();
    A.lower = M.lower.data();
    A.owner = dm.owner.data();
    A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data();
    A.losortStart = dm.losortStart.data();
    return A;
}


// phi = phiHbyA + pEqn.flux(). PLUS, and gpu::correctFlux cannot be reused for it.
//
// rhoSimpleFoam writes the pressure equation as `fvc::div(phiHbyA) - fvm::laplacian(...) == 0` and the
// reference negates the ENTIRE assembled matrix -- diag, off-diagonals, source, both boundary coefficient
// arrays and the face-flux correction -- to match. The incompressible solver writes
// `fvm::laplacian(...) == fvc::div(phiHbyA)` and subtracts. Same physics, opposite sign, and the sign is
// what makes phi discretely conservative rather than merely plausible: a wrong one leaves div(phi) != 0
// while the pressure equation still solves happily.
void correctFluxCompressible(
    DeviceBuffer<scalar>&       phiInt,
    DeviceBuffer<scalar>&       phiBnd,
    const DeviceBuffer<scalar>& phiHbyAInt,
    const DeviceBuffer<scalar>& phiHbyABnd,
    const PressureMatrix&       P,
    const DeviceMesh&           dm,
    const DeviceBoundary&       dbP,
    const DeviceBuffer<scalar>& pSolved)
{
    DeviceBuffer<scalar> fInt, fBnd;
    deviceMatrixFluxInternal(P.view(dm), pSolved, fInt);
    deviceMatrixFluxBoundary(dbP, P.iC, P.bC, pSolved, fBnd);
    // fvMatrix.C:1688 -- `if (faceFluxCorrectionPtr_) fieldFlux += *faceFluxCorrectionPtr_;`
    if (P.faceFluxCorr.size() > 0) deviceAxpy(1.0, P.faceFluxCorr, fInt);

    deviceCopy(phiInt, phiHbyAInt);
    deviceAxpy(1.0, fInt, phiInt);
    deviceCopy(phiBnd, phiHbyABnd);
    deviceAxpy(1.0, fBnd, phiBnd);
}


// The closed-volume correction: p += (initialMass - domainIntegrate(psi*p))/domainIntegrate(psi).
// Two reductions and a scalar add; done on the host because it is two numbers, and the alternative is a
// device reduction whose result has to come back anyway.
void closedVolumeCorrection(
    DeviceBuffer<scalar>&       p,
    const DeviceBuffer<scalar>& psi,
    const DeviceMesh&           dm,
    double                      initialMass)
{
    const std::vector<scalar> hp = p.host(), hpsi = psi.host(), V = dm.V.host();
    double num = 0.0, den = 0.0;
    for (int c = 0; c < dm.nCells; ++c)
    {
        num += (double)hpsi[c] * (double)hp[c] * (double)V[c];
        den += (double)hpsi[c] * (double)V[c];
    }
    if (!(den > 0.0)) return;
    const scalar dp = (scalar)((initialMass - num) / den);
    std::vector<scalar> out(hp);
    for (int c = 0; c < dm.nCells; ++c) out[c] += dp;
    p.copyFrom(out);
}

} // namespace


Residuals rhoSimpleStep(
    RhoSolverFields&            f,
    RhoSolverWorkspace&         w,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    DeviceBoundary&             dbP,
    DeviceBoundary&             dbHe,
    const RhoStepInput&         in)
{
    Residuals res;
    const int nC = dm.nCells;

    if (!in.muEffCell || !in.muEffBndFace || !in.alphaEffCell || !in.alphaEffBndFace)
    {
        throw std::runtime_error(
            "rhoSimpleFoam(cuda): muEff and alphaEff are required on cells AND boundary faces. They are "
            "the ONLY place the closure enters the momentum and energy equations, and the boundary value "
            "is the patch's, not the owner cell's -- on a wall with an alphat wall function the two "
            "differ by the whole of alphat.");
    }
    if (!in.thermoCorrect || !in.updateRho)
    {
        throw std::runtime_error(
            "rhoSimpleFoam(cuda): thermoCorrect and updateRho are required hooks. EEqn.H ends in "
            "thermo.correct(), which moves T and therefore psi, and every consumer below that point "
            "reads the result; pcEqn.H opens with rho = thermo.rho(). Running without them would solve "
            "the whole iteration against the state it started with.");
    }

    if (w.ones.size() != static_cast<std::size_t>(nC))
    {
        w.ones.copyFrom(std::vector<scalar>(nC, scalar(1.0)));
    }

    // storePrevIter(): OpenFOAM banks prevIter at the TOP of the iteration, so p.relax() below relaxes
    // against the value p had before this iteration touched it -- not against the value it had at the
    // start of the pressure solve.
    DeviceBuffer<scalar> pPrev;
    deviceCopy(pPrev, f.p);

    // ---- UEqn.H ------------------------------------------------------------------------------
    RhoMomentumInput uin;
    uin.phiInt = &f.phiInt;          uin.phiBnd = &f.phiBnd;
    uin.rhoCell = &f.rho;            uin.rhoBndFace = &f.rhoBnd;
    // THE DYNAMIC SLOT, NOT THE KINEMATIC ONE. RhoMomentumInput carries both: muEffCell/muEffBndFace are
    // used verbatim, while nuEffCell/nuEffBndFace are KINEMATIC and the module forms rho*nuEff from them
    // (linearViscousStress.C:107-117). Feeding the dynamic muEff into the kinematic slot multiplies it by
    // rho a second time -- measured on rhoBox as a constant 1.161 on the whole diffusion term, which is
    // exactly p/(R*T) = 100000/(287.1*300) there, and it reached the converged velocity as a drift of
    // 5.4e-04 at iteration 1 growing to 3.9e-03 by iteration 8 while p, T and he all stayed at ~1e-7.
    uin.muEffCell = in.muEffCell;    uin.muEffBndFace = in.muEffBndFace;
    uin.relaxU = in.relaxU;
    uin.relaxEquationU = in.relaxEquationU;
    uin.bounded = in.boundedU;
    uin.scheme = in.schemeU;
    uin.schemeCoeff = in.schemeCoeffU;
    uin.gradULimitK = in.gradULimitK;
    uin.correctedLaplacian = in.correctedLaplacian;
    uin.snGradLimitCoeff = in.snGradLimitCoeff;
    uin.hasMRF = in.hasMRF;
    uin.hasFvOptions = in.hasFvOptions;
    uin.hasCoupledPatches = in.hasCoupledPatches;
    uin.fvOptionUnsupported = in.fvOptionUnsupported;

    MomentumMatrix UEqn;
    assembleUEqn(UEqn, dm, dbU, f.Ux, f.Uy, f.Uz, uin);

    {
        // solve(UEqn == -fvc::grad(p)) on a COPY. The pressure equation needs the ORIGINAL for A(), H()
        // and H1(); adding grad(p) here would leave the source carrying it and move rAU and HbyA with it.
        MomentumMatrix Mp;
        deviceCopy(Mp.diag, UEqn.diag);
        deviceCopy(Mp.upper, UEqn.upper);
        deviceCopy(Mp.lower, UEqn.lower);
        deviceCopy(Mp.relaxedDiag, UEqn.relaxedDiag);
        Mp.relaxed = UEqn.relaxed;
        for (int k = 0; k < 3; ++k)
        {
            deviceCopy(Mp.source[k], UEqn.source[k]);
            deviceCopy(Mp.iC[k], UEqn.iC[k]);
            deviceCopy(Mp.bC[k], UEqn.bC[k]);
        }

        DeviceBuffer<scalar> gpx, gpy, gpz;
        deviceBCValue(dbP, f.p, f.pBnd);
        deviceGaussGrad(dm, f.p, f.pBnd, gpx, gpy, gpz);
        addPressureGradient(Mp, dm, gpx, gpy, gpz);

        DeviceBuffer<scalar>* U[3] = {&f.Ux, &f.Uy, &f.Uz};
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> diagC, b;
            deviceFold(dm, Mp.relaxed ? Mp.relaxedDiag : Mp.diag, Mp.source[k], Mp.iC[k], Mp.bC[k], diagC, b);
            const DeviceLduView A = foldedViewM(dm, Mp, diagC);
            const scalar nf = deviceNormFactor(A, *U[k], b, w.ones);
            DeviceSolverPerf perf;
            if (in.uSymGaussSeidel)
                deviceSymGaussSeidel(A, b, *U[k], nf, in.tolU, in.relTolU, in.maxIter, &perf);
            else
                perf = deviceJacobiBiCGStab(A, b, *U[k], nf, in.tolU, in.relTolU, in.maxIter);
            // U is reported under the FIRST component's initial residual, NOT a cmptMax over three.
            // OF's cmptMax runs over a performance object whose skipped components were never solved
            // (fvMatrixSolve.C:157-164 continues on validComponents == -1) while this path solves all
            // three unconditionally -- so a cmptMax here would include the degenerate empty/wedge
            // direction and block convergence on every 2D case.
            if (k == 0) res["U"] = perf.initialResidual;
        }
    }

    // ---- EEqn.H ------------------------------------------------------------------------------
    {
        RhoEnergyInput ein;
        ein.phiInt = &f.phiInt;           ein.phiBnd = &f.phiBnd;
        ein.alphaEffCell = in.alphaEffCell;
        ein.alphaEffBndFace = in.alphaEffBndFace;
        ein.Ux = &f.Ux; ein.Uy = &f.Uy; ein.Uz = &f.Uz;
        ein.pCell = &f.p; ein.rhoCell = &f.rho;
        // Refreshed here, from the just-solved U: EEqn.H's kinetic-energy source is evaluated on
        // boundary faces as well as cells, and the momentum solve above has moved every one of them.
        deviceBCValue(dbU.comp[0], f.Ux, f.UxBnd);
        deviceBCValue(dbU.comp[1], f.Uy, f.UyBnd);
        deviceBCValue(dbU.comp[2], f.Uz, f.UzBnd);
        ein.UxBnd = &f.UxBnd; ein.UyBnd = &f.UyBnd; ein.UzBnd = &f.UzBnd;
        ein.pBnd = &f.pBnd; ein.rhoBnd = &f.rhoBnd;
        ein.isE = in.isE;
        ein.relaxHe = in.relaxHe;
        ein.relaxEquationHe = in.relaxEquationHe;
        ein.boundedHe = in.boundedHe;
        ein.boundedKE = in.boundedKE;
        ein.schemeHe = in.schemeHe;
        ein.schemeKE = in.schemeKE;
        ein.gradHeLimitK = in.gradHeLimitK;
        ein.gradKELimitK = in.gradKELimitK;
        ein.correctedLaplacian = in.correctedLaplacian;
        ein.snGradLimitCoeff = in.snGradLimitCoeff;
        ein.hasMRF = in.hasMRF;
        ein.hasFvOptions = in.hasFvOptions;
        ein.hasCoupledPatches = in.hasCoupledPatches;

        PressureMatrix E;
        assembleEEqn(E, dm, dbHe, f.he, ein);

        DeviceBuffer<scalar> diagC, b;
        deviceFold(dm, E.diag, E.source, E.iC, E.bC, diagC, b);
        const DeviceLduView A = foldedView(dm, E, diagC);
        const scalar nf = deviceNormFactor(A, f.he, b, w.ones);
        const DeviceSolverPerf perf =
            deviceJacobiBiCGStab(A, b, f.he, nf, in.tolHe, in.relTolHe, in.maxIter);
        res[in.isE ? "e" : "h"] = perf.initialResidual;
        deviceBCValue(dbHe, f.he, f.heBnd);
    }

    // EEqn.H ends with thermo.correct(): T, and therefore psi, move HERE and everything below sees them.
    in.thermoCorrect();

    // ---- pEqn.H or pcEqn.H -------------------------------------------------------------------
    RhoPressureInput pin;
    pin.rhoCell = &f.rho;            pin.rhoBndFace = &f.rhoBnd;
    pin.psiCell = &f.psi;            pin.psiBndFace = &f.psiBnd;
    pin.transonic = in.transonic;
    pin.relaxP = in.relaxPEqn;
    pin.relaxPSpecified = in.relaxPEqnSpecified;
    pin.pRefCell = in.pRefCell;      pin.pRefValue = in.pRefValue;
    pin.correctedLaplacian = in.correctedLaplacian;
    pin.snGradLimitCoeff = in.snGradLimitCoeff;
    pin.takeUAtBoundary = in.takeUAtBoundary;
    pin.adjustable = in.adjustable;
    pin.hasMRF = in.hasMRF;
    pin.hasFvOptions = in.hasFvOptions;
    pin.hasFixedFluxPressure = in.hasFixedFluxPressure;
    pin.hasCoupledPatches = in.hasCoupledPatches;
    pin.fvOptionUnsupported = in.fvOptionUnsupported;

    RhoPressureStages        st;
    ConsistentPressureStages cst;
    bool closedVolume = false;

    if (in.consistent)
    {
        // pcEqn.H OPENS with `rho = thermo.rho()`. pEqn.H does not -- so the SIMPLEC pressure equation is
        // built from a density that already reflects the just-solved T and the plain SIMPLE one is not.
        in.updateRho();
        consistentPressurePredictor(cst, dm, dbU, dbP, UEqn, f.Ux, f.Uy, f.Uz, f.p, pin);
        closedVolume = cst.closedVolume;
    }
    else
    {
        pressurePredictor(st, dm, dbU, dbP, UEqn, f.Ux, f.Uy, f.Uz, f.p, pin);
        closedVolume = st.closedVolume;
    }

    // The non-orthogonal corrector loop. solutionControlI.H:78-95 runs it nNonOrth+1 times, and only the
    // FINAL pass writes phi (simple.finalNonOrthogonalIter()).
    const label nCorr = in.nNonOrthogonalCorrectors + 1;
    for (label corr = 1; corr <= nCorr; ++corr)
    {
        PressureMatrix& P = w.P;                          // persistent -- see RhoSolverWorkspace
        if (in.consistent) assemblePcEqn(P, cst, dm, dbP, f.p, pin);
        else               assemblePEqn(P, st, dm, dbP, f.p, pin);

        DeviceBuffer<scalar>& diagC = w.diagC;
        DeviceBuffer<scalar>& b     = w.b;
        deviceFold(dm, P.diag, P.source, P.iC, P.bC, diagC, b);
        const DeviceLduView A = foldedView(dm, P, diagC);
        const scalar nf = deviceNormFactor(A, f.p, b, w.ones);

        DeviceSolverPerf perf;
        if (in.transonic)
        {
            // fvm::div(phid, p) makes lower = -w*phi and upper = lower + phi, so upper != lower at every
            // face with flow through it. A symmetric solver on that matrix is not slow, it is wrong: CG
            // burned the full 3000-iteration cap and the case stalled before printing iteration 1.
            perf = deviceJacobiBiCGStab(A, b, f.p, nf, in.tolP, in.relTolP, in.maxIter, in.pcgCheckEvery);
        }
        else
        {
            if (!w.amgBuilt)
            {
                // Face weights SLICED TO INTERNAL FACES. DeviceMesh::magSf is |Sf| over ALL faces, laid
                // out [internal | non-cyclic boundary], while owner/nei are internal-only -- handing the
                // whole array to buildAMG pairs an internal-face addressing with a weight array that
                // runs on into the boundary, and every decision the agglomeration makes (which faces are
                // strong, hence which cells merge) is downstream of that pairing.
                const std::vector<label>  own = dm.owner.host(), nei = dm.nei.host();
                const std::vector<scalar> magSfAll = dm.magSf.host();
                const std::size_t nIf = static_cast<std::size_t>(dm.nInternalFaces);
                const std::vector<label>  ownInt(own.begin(), own.begin() + std::min(nIf, own.size()));
                const std::vector<label>  neiInt(nei.begin(), nei.begin() + std::min(nIf, nei.size()));
                const std::vector<scalar> fw(magSfAll.begin(),
                                             magSfAll.begin() + std::min(nIf, magSfAll.size()));
                // The hierarchy is a function of the MESH: only the STRUCTURE is serialised, and
                // cDiag/cUpper/cLower are Galerkin-rebuilt every step. So a cache a simpleFoam run wrote
                // for this mesh is valid here, and until this line existed the compressible path started
                // cold on every run even when one was sitting next to constant/polyMesh.
                w.amg = in.amgCacheDir.empty()
                      ? buildAMG(ownInt, neiInt, fw, dm.nCells)
                      : buildOrLoadAMG(ownInt, neiInt, fw, dm.nCells, in.amgCacheDir, true);
                w.amgBuilt = true;
            }
            amgGalerkin(w.amg, diagC, P.upper, P.lower);
            perf = deviceAMGPCG(A, w.amg, b, f.p, nf, in.tolP, in.relTolP, in.maxIter,
                                in.captureVcycle, in.pcgCheckEvery);
        }
        // solutionControl.C:230-233 takes sp.first() -- the FIRST solve of the iteration, not the last.
        if (corr == 1) res["p"] = perf.initialResidual;

        if (corr == nCorr)
        {
            correctFluxCompressible(f.phiInt, f.phiBnd,
                                    in.consistent ? cst.phiHbyAInt : st.phiHbyAInt,
                                    in.consistent ? cst.phiHbyABnd : st.phiHbyABnd,
                                    P, dm, dbP, f.p);
        }
    }

    // p.relax() -- the FIELD factor, not the equation one. Both are spelled `p` in fvSolution and they
    // live in different sub-dictionaries; using the equation factor here relaxes the wrong thing.
    // AFTER the flux correction and BEFORE the velocity correction, so phi is built from the unrelaxed
    // pressure and U from the relaxed one.
    relaxField(f.p, pPrev, in.relaxP);
    deviceBCValue(dbP, f.p, f.pBnd);

    // U = HbyA - rAtU*fvc::grad(p), with rAtU on the SIMPLEC path and rAU otherwise.
    {
        DeviceBuffer<scalar> gpx, gpy, gpz;
        deviceGaussGrad(dm, f.p, f.pBnd, gpx, gpy, gpz);
        PressureStages shim;
        if (in.consistent)
        {
            for (int k = 0; k < 3; ++k) deviceCopy(shim.HbyA[k], cst.HbyA[k]);
            deviceCopy(shim.rAtU, cst.rAtU);
        }
        else
        {
            for (int k = 0; k < 3; ++k) deviceCopy(shim.HbyA[k], st.HbyA[k]);
            deviceCopy(shim.rAtU, st.rAU);
        }
        correctVelocity(f.Ux, f.Uy, f.Uz, shim, gpx, gpy, gpz);
    }

    // The closed-volume mass correction. `closedVolume` is set by the predictor, on the same condition
    // adjustPhi and pRefCell are: no patch fixes a pressure value, so the level is undetermined and the
    // total mass is what pins it.
    if (closedVolume)
    {
        closedVolumeCorrection(f.p, f.psi, dm, f.initialMass);
        deviceBCValue(dbP, f.p, f.pBnd);
    }

    // rho = thermo.rho(), then rho.relax() -- only when NOT transonic.
    //
    // THE BOUNDARY IS RELAXED TOO. GeometricField::relax is operator==(prevIter + alpha*(*this -
    // prevIter)) and operator== assigns BOTH halves. Relaxing only the internal field leaves rho's patch
    // values at the unrelaxed thermo value, which is a different field from the one OpenFOAM carries --
    // and it matters directly, because flowRateInletVelocity holds the prescribed mass flow against
    // rho's PATCH values, so an unrelaxed boundary sets a different inlet velocity every iteration.
    {
        DeviceBuffer<scalar> rhoPrev, rhoBndPrev;
        deviceCopy(rhoPrev, f.rho);
        deviceCopy(rhoBndPrev, f.rhoBnd);
        in.updateRho();
        if (!in.transonic)
        {
            relaxField(f.rho, rhoPrev, in.relaxRho);
            relaxField(f.rhoBnd, rhoBndPrev, in.relaxRho);
        }
    }

    // turbulence->correct() -- LAST, so the NEXT iteration's momentum equation uses this iteration's
    // closure. OpenFOAM's lagged coupling.
    if (in.correct) in.correct();

    return res;
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
