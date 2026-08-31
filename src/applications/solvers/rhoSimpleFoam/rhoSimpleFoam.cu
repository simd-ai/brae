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


// fvOptions.correct(he) for limitTemperature: clamp he between he(p,Tmin) and he(p,Tmax). A CORRECTION,
// so nothing in the assembly changes -- it acts on the solved field and then thermo.correct() turns it
// into a temperature. The bounds arrive already in energy; see the note in RhoStepInput.
__global__ void limitEnergyKernel(
    int    nC,
    scalar heMin,
    scalar heMax,
    scalar* __restrict__ he)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    he[c] = fmin(fmax(he[c], heMin), heMax);
}


// pressureControl::limit -- a clamp, applied in place. OpenFOAM returns true on `limitMaxP || limitMinP`
// rather than on whether any value actually moved, and the caller re-evaluates p's boundary on that
// return, so the boundary refresh below is keyed the same way.
__global__ void limitPressureKernel(
    int    nC,
    int    doMax,
    int    doMin,
    scalar pMax,
    scalar pMin,
    scalar* __restrict__ p)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    if (doMax) p[c] = fmin(p[c], pMax);
    if (doMin) p[c] = fmax(p[c], pMin);
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


// updateCoeffs() for the boundary conditions whose coefficients are a function of the SOLUTION.
//
// A named function rather than a run of statements inside the step, because it has a contract of its own
// that is worth testing on its own: given a flux and a boundary density, it must produce the patch
// coefficients OpenFOAM's updateCoeffs() would. The driver's gate exercises it directly -- doubling the
// boundary density must halve a flowRateInletVelocity's velocity, and reversing the flux must flip an
// inletOutlet face between fixedValue and zeroGradient -- and neither of those is visible from a
// whole-iteration comparison on a fixture whose patches have no coefficients that move.
void updateBoundaryCoeffs(
    RhoSolverFields&      f,
    DeviceVectorBoundary& dbU,
    DeviceBoundary&       dbP,
    DeviceBoundary&       dbHe,
    DeviceBoundary&       dbT,
    const RhoStepInput&   in)
{
    // OpenFOAM runs this inside the fvMatrix constructor, so it has happened before any coefficient is
    // read. Here the device boundary objects are a snapshot and the driver has to do it by hand; the
    // order is the reference driver's, which is OpenFOAM's.
    //
    // 1. The FLUX SWITCH. inletOutlet/outletInlet pick fixedValue or zeroGradient per face from the sign
    //    of phi, and OpenFOAM lags it: the flux used is the one this iteration STARTS with. U, he and T
    //    are the fields that carry one on a compressible case.
    //
    //    dbT is refreshed even though nothing in THIS function reads it. T's boundary is consumed by
    //    thermo.correct(), which is the caller's hook; a host thermo evaluates T's patches on its own
    //    host field and will not notice, but a device-resident one reads dbT and would otherwise get a
    //    flux switch frozen at its seeded state. Refreshing it here keeps the two thermo implementations
    //    interchangeable, which is the whole point of the hook being a hook.
    deviceUpdateInletOutlet(dbU, f.phiBnd);
    deviceUpdateInletOutlet(dbHe, f.phiBnd);
    deviceUpdateInletOutlet(dbT, f.phiBnd);

    // 2. The FREESTREAM BLEND, a different rule from the switch above: valueFraction is rebuilt from the
    //    current flow ANGLE, 0.5 - 0.5*(Up & nf)/mag(Up), and freestreamPressure follows the velocity
    //    patch. Left alone, every far-field face keeps the half-and-half blend it was seeded with.
    if (in.hasMixed)
    {
        deviceUpdateMixedFreestream(dbU, dbP, f.phiBnd, f.Ux, f.Uy, f.Uz, &f.rhoBnd);
        deviceBCValue(dbP, f.p, f.pBnd);
    }

    // 2b. pressureInletOutletVelocity and totalPressure -- the two patches whose value is a function of
    //     the patch VELOCITY. Both device kernels have existed and been called by the incompressible
    //     driver all along (device_simple_foam.cu:940 and :992); this driver called neither, and the
    //     host classes carried a comment saying the DEVICE recomputes them each step. validation/rhoTP
    //     carries both, and DIVERGED through this lineage -- T 3.79e+38 -- while the legacy path, which
    //     does compute them, converges against OpenFOAM.
    //
    //     THE COMPRESSIBLE FORM TAKES rho. OpenFOAM's totalPressure has three branches
    //     (totalPressureFvPatchScalarField.C): p in Pa with psi unnamed is p0 - 0.5*rho*neg(phi)*|U|^2;
    //     p in Pa with psi NAMED is the isentropic high-speed form; p/rho dimensions is the
    //     incompressible p0 - 0.5*neg(phi)*|U|^2. This solver is always the first of those, so rhoBnd is
    //     passed -- omitting it silently selects the incompressible form, wrong by a factor of rho.
    {
        DeviceBuffer<scalar> ubx, uby, ubz;
        deviceBCValue(dbU.comp[0], f.Ux, ubx);
        deviceBCValue(dbU.comp[1], f.Uy, uby);
        deviceBCValue(dbU.comp[2], f.Uz, ubz);
        deviceUpdatePressureInletOutletVelocity(dbU, f.phiBnd, f.Ux, f.Uy, f.Uz);
        deviceUpdateTotalPressure(dbP, f.phiBnd, ubx, uby, ubz, &f.rhoBnd);
        deviceBCValue(dbP, f.p, f.pBnd);
    }

    // 3. flowRateInletVelocity, and WHICH rho matters: avgU = -mdot/gSum(rho*magSf) is held against the
    //    boundary density the flux is actually carrying -- the solver's relaxed rho, which is what
    //    f.rhoBnd holds here -- not thermo.rho(). Feeding it the other one is the angledDuct defect,
    //    where the inlet quietly lost the prescribed mass flow. Last, because it reads that rho.
    if (in.frMagSf && in.frMdot && in.frNx && in.frNy && in.frNz)
    {
        for (std::size_t k = 0; k < in.frMagSf->size() && k < in.frMdot->size(); ++k)
        {
            const scalar sumRhoA = deviceDot(f.rhoBnd, (*in.frMagSf)[k]);
            if (sumRhoA <= scalar(0)) continue;
            deviceUpdateFlowRateInlet(dbU, (*in.frMagSf)[k], -(*in.frMdot)[k] / sumRhoA,
                                      *in.frNx, *in.frNy, *in.frNz);
        }
    }
}


Residuals rhoSimpleStep(
    RhoSolverFields&            f,
    RhoSolverWorkspace&         w,
    const DeviceMesh&           dm,
    // NON-const: the updateCoeffs() block at the top of the step rewrites refValue and valueFraction on
    // the patches that switch on the flux, blend on the flow angle, or carry a prescribed mass flow.
    DeviceVectorBoundary&       dbU,
    DeviceBoundary&             dbP,
    DeviceBoundary&             dbHe,
    DeviceBoundary&             dbT,
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

    updateBoundaryCoeffs(f, dbU, dbP, dbHe, dbT, in);

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
    // The porosity the momentum module has always been able to apply, and which the driver never passed.
    uin.porosity = in.porosity;
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

        // fvOptions.correct(he), EEqn.H:27 -- AFTER the solve and BEFORE thermo.correct(), which is what
        // makes it reach T at all. Applying it later would clamp an energy the thermo had already turned
        // into a temperature, and applying it earlier would clamp the field the solve is about to
        // overwrite.
        if (in.limitHe)
        {
            limitEnergyKernel<<<(nC + 255) / 256, 256>>>(nC, in.heMin, in.heMax, f.he.data());
            cudaCheck(cudaGetLastError(), "rhoSimpleFoam limitEnergy");
        }
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

        // U.correctBoundaryConditions() -- pEqn.H:87 and pcEqn.H:100, on the line IMMEDIATELY after
        // `U = HbyA - rAU*fvc::grad(p)`. The driver refreshed U's boundary once, before the energy
        // equation, and never again: from the velocity correction onward f.UxBnd/UyBnd/UzBnd held the
        // values U had BEFORE the pressure correction, for the rest of the iteration and into the next.
        //
        // Found by comparing every field against the host reference rather than the handful the driver
        // gate reports. At the end of iteration 1 on sbMatched every reported field agreed to ~1e-12
        // while UxBnd was out by 6.81e-01 and UyBnd/UzBnd by 1.00e+00 -- entirely different values, not
        // a drift. It fed forward through everything that reads U's patch values: fvc::div(phi, Ekp)
        // evaluates Ekp on boundary faces, the closure's production and its turbulentIntensity inlet
        // both read U_b, and the next iteration's flux switch reads the boundary flux built from it.
        deviceBCValue(dbU.comp[0], f.Ux, f.UxBnd);
        deviceBCValue(dbU.comp[1], f.Uy, f.UyBnd);
        deviceBCValue(dbU.comp[2], f.Uz, f.UzBnd);
    }

    // pressureControl.limit(p), HERE and not earlier: pEqn.H applies it after the velocity correction, so
    // U is built from the unclipped pressure and only p carries the clip.
    const bool pLimited = in.limitMaxP || in.limitMinP;
    if (pLimited)
    {
        limitPressureKernel<<<(nC + 255) / 256, 256>>>(nC, in.limitMaxP ? 1 : 0, in.limitMinP ? 1 : 0,
                                                       in.pMaxLimit, in.pMinLimit, f.p.data());
        cudaCheck(cudaGetLastError(), "rhoSimpleFoam limitPressure");
    }

    // The closed-volume mass correction. `closedVolume` is set by the predictor, on the same condition
    // adjustPhi and pRefCell are: no patch fixes a pressure value, so the level is undetermined and the
    // total mass is what pins it.
    if (closedVolume)
    {
        closedVolumeCorrection(f.p, f.psi, dm, f.initialMass);
    }

    // ONE refresh for both, keyed as OpenFOAM keys it: `if (pLimited || closedVolume)`.
    if (pLimited || closedVolume)
    {
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
