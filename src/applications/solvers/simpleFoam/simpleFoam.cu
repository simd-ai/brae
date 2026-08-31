// CUDA DRIVER -- see simpleFoam.cuh for provenance and the ordering notes.
#include "simpleFoam.cuh"
#include "device_blas.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"

namespace brae {
namespace gpu {

namespace {

// The LduView for a FOLDED system: the boundary internalCoeffs have been added to the diagonal, which is
// the matrix the linear solver actually sees. OpenFOAM folds at solve time for the same reason -- keeping
// internalCoeffs separate until then is what lets A(), H() and flux() each use the raw form they need.
DeviceLduView foldedView(const DeviceMesh& dm, const MomentumMatrix& M, const DeviceBuffer<scalar>& diagC)
{
    DeviceLduView A{};
    A.nCells = dm.nCells; A.nInternalFaces = dm.nInternalFaces;
    A.diag = diagC.data(); A.upper = M.upper.data(); A.lower = M.lower.data();
    A.owner = dm.owner.data(); A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data(); A.losortStart = dm.losortStart.data();
    return A;
}

DeviceLduView foldedView(const DeviceMesh& dm, const PressureMatrix& P, const DeviceBuffer<scalar>& diagC)
{
    DeviceLduView A{};
    A.nCells = dm.nCells; A.nInternalFaces = dm.nInternalFaces;
    A.diag = diagC.data(); A.upper = P.upper.data(); A.lower = P.lower.data();
    A.owner = dm.owner.data(); A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data(); A.losortStart = dm.losortStart.data();
    return A;
}

} // namespace


Residuals simpleStep(
    SolverFields&                f,
    SolverWorkspace&             w,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    const DeviceBoundary&        dbP,
    const StepInput&             in)
{
    Residuals res;
    if (w.ones.size() != static_cast<std::size_t>(dm.nCells))
    {
        w.ones.copyFrom(std::vector<scalar>(dm.nCells, 1.0));
    }

    // storePrevIterFields(): OpenFOAM banks prevIter at the TOP of the iteration, so p.relax() below
    // relaxes against the value p had before this iteration touched it.
    DeviceBuffer<scalar> pPrev;
    deviceCopy(pPrev, f.p);

    // ---- UEqn.H ------------------------------------------------------------------------------
    MomentumInput mi;
    mi.phiInt = &f.phiInt;          mi.phiBnd = &f.phiBnd;
    mi.nuEffCell = in.nuEffCell;    mi.nuEffFace = in.nuEffFace;
    mi.nuEffBndFace = in.nuEffBndFace;
    mi.relaxU = in.relaxU;
    mi.bounded = in.bounded;
    mi.linearUpwind = in.linearUpwind;
    mi.scheme       = in.scheme;
    mi.schemeCoeff  = in.schemeCoeff;
    mi.porosity     = in.porosity;
    mi.mrf          = in.mrf;
    mi.gradULimitK  = in.gradULimitK;
    mi.snGradLimitCoeff = in.snGradLimitCoeff;
    mi.rotor        = in.rotor;
    mi.actuationDisk = in.actuationDisk;
    mi.nuLaminar    = in.nuLaminar;
    mi.correctedLaplacian = in.correctedLaplacian;
    mi.hasMRF = in.hasMRF;          mi.hasFvOptions = in.hasFvOptions;

    MomentumMatrix MU;
    assembleUEqn(MU, dm, dbU, f.Ux, f.Uy, f.Uz, mi);

    if (in.momentumPredictor)
    {
        // solve(UEqn == -fvc::grad(p)) on a COPY -- pEqn.H needs the original for A() and H().
        MomentumMatrix Mp;
        deviceCopy(Mp.diag, MU.diag);
        deviceCopy(Mp.upper, MU.upper);
        deviceCopy(Mp.lower, MU.lower);
        deviceCopy(Mp.relaxedDiag, MU.relaxedDiag);
        Mp.relaxed = MU.relaxed;
        for (int k = 0; k < 3; ++k)
        {
            deviceCopy(Mp.source[k], MU.source[k]);
            deviceCopy(Mp.iC[k], MU.iC[k]);
            deviceCopy(Mp.bC[k], MU.bC[k]);
        }

        DeviceBuffer<scalar> gpx, gpy, gpz, pb;
        deviceBCValue(dbP, f.p, pb);
        deviceGaussGrad(dm, f.p, pb, gpx, gpy, gpz);
        addPressureGradient(Mp, dm, gpx, gpy, gpz);

        DeviceBuffer<scalar>* U[3] = {&f.Ux, &f.Uy, &f.Uz};
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> diagC, b;
            deviceFold(dm, Mp.relaxed ? Mp.relaxedDiag : Mp.diag, Mp.source[k], Mp.iC[k], Mp.bC[k], diagC, b);
            const DeviceLduView A = foldedView(dm, Mp, diagC);
            const scalar nf = deviceNormFactor(A, *U[k], b, w.ones);
            // The solver the CASE asked for. deviceSymGaussSeidel is OpenFOAM's smoothSolver +
            // symGaussSeidelSmoother (multicolor, internal-face LDU only -- no coupled interfaces, which
            // this path refuses anyway).
            DeviceSolverPerf perf;
            if (in.uSymGaussSeidel)
                deviceSymGaussSeidel(A, b, *U[k], nf, in.tolU, in.relTolU, in.maxIter, &perf);
            else
                perf = deviceJacobiBiCGStab(A, b, *U[k], nf, in.tolU, in.relTolU, in.maxIter);
            // OpenFOAM's residualControl watches the FIRST component's initial residual under the field
            // name; keep that convention so simpleControl's criteria mean the same thing here.
            if (k == 0) res["U"] = perf.initialResidual;
            if (std::getenv("BRAE_SOLVER_ITERS"))
                std::printf("    [U%d] nIter=%d init=%.3e final=%.3e\n",
                            k, perf.nIterations, perf.initialResidual, perf.finalResidual);
        }
    }

    // ---- pEqn.H ------------------------------------------------------------------------------
    PressureInput pin;
    pin.relaxP = in.relaxP;
    pin.pRefCell = in.pRefCell;   pin.pRefValue = in.pRefValue;
    pin.consistent = in.consistent;
    pin.correctedLaplacian = in.correctedLaplacian;
    pin.snGradLimitCoeff   = in.snGradLimitCoeff;
    pin.hasMRF = in.hasMRF;       pin.hasFvOptions = in.hasFvOptions;
    pin.mrf = in.mrf;
    pin.adjustable = in.adjustable;
    pin.takeUAtBoundary = in.takeUAtBoundary;

    PressureStages st;
    pressurePredictor(st, dm, dbU, MU, f.Ux, f.Uy, f.Uz, pin, &dbP, &f.p);

    DeviceBuffer<scalar> rAUface;
    // rAtU, not rAU: they are the same buffer unless SIMPLEC is on, and pEqn.H's laplacian takes rAtU
    // in both cases.
    deviceInterpolate(dm, st.rAtU, rAUface);

    if (in.probe)
    {
        in.probe->rAU = st.rAU.host();
        in.probe->rAUface = rAUface.host();
        for (int k = 0; k < 3; ++k)
        { in.probe->HbyA[k] = st.HbyA[k].host(); in.probe->HbyAb[k] = st.HbyAb[k].host(); }
        in.probe->phiHbyAInt = st.phiHbyAInt.host();
        in.probe->phiHbyABnd = st.phiHbyABnd.host();
    }

    // Non-orthogonal corrector loop -- solutionControlI.H:78-95 runs it nNonOrth+1 times, and only the
    // final pass writes phi (simple.finalNonOrthogonalIter()).
    const label nCorr = in.nNonOrthogonalCorrectors + 1;
    for (label corr = 1; corr <= nCorr; ++corr)
    {
        PressureMatrix& P = w.P;                       // persistent -- see SolverWorkspace
        assemblePEqn(P, st, dm, dbP, rAUface, pin, &f.p);

        DeviceBuffer<scalar>& diagC = w.diagC;
        DeviceBuffer<scalar>& b     = w.b;
        deviceFold(dm, P.diag, P.source, P.iC, P.bC, diagC, b);
        const DeviceLduView A = foldedView(dm, P, diagC);

        if (!w.amgBuilt)
        {
            // Face weights SLICED TO INTERNAL FACES, matching how the existing GPU driver builds its
            // hierarchy (device_simple_foam.cu:341). DeviceMesh::magSf is documented "|Sf|(all faces)" and
            // is laid out [internal | non-cyclic boundary], while owner/nei are internal-only -- so
            // handing the whole array to buildAMG pairs an internal-face addressing with a face-weight
            // array that continues into the boundary. Everything the agglomeration decides (which faces
            // are strong, hence which cells merge) is downstream of that.
            const std::vector<label> own = dm.owner.host(), nei = dm.nei.host();
            const std::vector<scalar> magSfAll = dm.magSf.host();
            const std::size_t nIf = static_cast<std::size_t>(dm.nInternalFaces);
            const std::vector<label> ownInt(own.begin(), own.begin() + std::min(nIf, own.size()));
            const std::vector<label> neiInt(nei.begin(), nei.begin() + std::min(nIf, nei.size()));
            const std::vector<scalar> fw(magSfAll.begin(),
                                         magSfAll.begin() + std::min(nIf, magSfAll.size()));
            w.amg = in.amgCacheDir.empty()
                  ? buildAMG(ownInt, neiInt, fw, dm.nCells)
                  : buildOrLoadAMG(ownInt, neiInt, fw, dm.nCells, in.amgCacheDir, true);
            w.amgBuilt = true;
        }
        amgGalerkin(w.amg, diagC, P.upper, P.lower);

        const scalar nf = deviceNormFactor(A, f.p, b, w.ones);
        const DeviceSolverPerf perf =
            deviceAMGPCG(A, w.amg, b, f.p, nf, in.tolP, in.relTolP, in.maxIter,
                         in.captureVcycle, in.pcgCheckEvery);
        if (corr == 1) res["p"] = perf.initialResidual;   // the FIRST solve's residual, as OpenFOAM reports
        // Solver-iteration counts, on demand. The wall-clock question on this path turned out to be "how
        // many Krylov iterations", not "how fast is a kernel": every iteration ends in a scalar
        // device-to-host read to test convergence, and those reads are blocking.
        if (std::getenv("BRAE_SOLVER_ITERS"))
            std::printf("    [p] nIter=%d init=%.3e final=%.3e\n",
                        perf.nIterations, perf.initialResidual, perf.finalResidual);

        if (corr == nCorr)
        {
            if (in.probe)
            {
                in.probe->pDiag = P.diag.host();     in.probe->pUpper = P.upper.host();
                in.probe->pLower = P.lower.host();   in.probe->pSource = P.source.host();
                in.probe->pIC = P.iC.host();         in.probe->pBC = P.bC.host();
                in.probe->pSolved = f.p.host();
            }
            correctFlux(f.phiInt, f.phiBnd, st, P, dm, dbP, f.p);
            if (in.probe)
            { in.probe->phiInt = f.phiInt.host(); in.probe->phiBnd = f.phiBnd.host(); }
        }
    }

    // p.relax(), then the momentum corrector -- in that order.
    relaxField(f.p, pPrev, in.relaxP);

    {
        DeviceBuffer<scalar> gpx, gpy, gpz, pb;
        deviceBCValue(dbP, f.p, pb);
        deviceGaussGrad(dm, f.p, pb, gpx, gpy, gpz);
        correctVelocity(f.Ux, f.Uy, f.Uz, st, gpx, gpy, gpz);
    }

    // simpleFoam.C:93-94 -- laminarTransport.correct(); turbulence->correct().
    if (in.correct) in.correct();

    return res;
}

} // namespace gpu
} // namespace brae
