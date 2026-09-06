// CUDA DRIVER -- see simpleFoam.cuh for provenance and the ordering notes.
#include "simpleFoam.cuh"
#include "device_blas.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

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

// Instrument: BRAE_STAGE_DUMP_DIR=<dir> (+ BRAE_STAGE_DUMP_ITER=n, default 1) writes this step's
// stages as plain columns, under the SAME names tools/dumpSimpleFoam writes as OpenFOAM fields, so the
// two can be held against each other coefficient by coefficient at one iteration. Cell fields and
// INTERNAL faces only: both read the same polyMesh, so those two orderings are OpenFOAM's by
// construction, while the flattened boundary array is brae's own and would need a patch map first.
// Costs nothing unless the variable is set.
struct DeviceStageDump
{
    std::string dir;
    bool        on = false;

    void scalars(const char* name, const DeviceBuffer<scalar>& v) const
    {
        if (!on) return;
        const std::vector<scalar> h = v.host();
        std::FILE* fp = std::fopen((dir + "/" + name).c_str(), "w");
        if (!fp) return;
        for (scalar x : h) std::fprintf(fp, "%.17g\n", (double)x);
        std::fclose(fp);
    }
    void vectors(const char* name,
                 const DeviceBuffer<scalar>& x,
                 const DeviceBuffer<scalar>& y,
                 const DeviceBuffer<scalar>& z) const
    {
        if (!on) return;
        const std::vector<scalar> hx = x.host(), hy = y.host(), hz = z.host();
        std::FILE* fp = std::fopen((dir + "/" + name).c_str(), "w");
        if (!fp) return;
        for (std::size_t i = 0; i < hx.size(); ++i)
            std::fprintf(fp, "%.17g %.17g %.17g\n", (double)hx[i], (double)hy[i], (double)hz[i]);
        std::fclose(fp);
    }
};

DeviceStageDump deviceStageDump()
{
    DeviceStageDump d;
    const char* dd = std::getenv("BRAE_STAGE_DUMP_DIR");
    if (!dd) return d;
    static int calls = 0;
    const char* it = std::getenv("BRAE_STAGE_DUMP_ITER");
    d.dir = dd;
    d.on  = (++calls == (it ? std::atoi(it) : 1));
    return d;
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
    w.report.clear();   // this step's solves, reported below in the order they run

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
    mi.gradUSchemeLimitK = in.gradUSchemeLimitK;
    mi.snGradLimitCoeff = in.snGradLimitCoeff;
    mi.rotor        = in.rotor;
    mi.actuationDisk = in.actuationDisk;
    mi.nuLaminar    = in.nuLaminar;
    mi.correctedLaplacian = in.correctedLaplacian;
    mi.hasMRF = in.hasMRF;          mi.hasFvOptions = in.hasFvOptions;

    const DeviceStageDump sd = deviceStageDump();
    sd.vectors("stage_Uass", f.Ux, f.Uy, f.Uz);
    sd.scalars("stage_phiU", f.phiInt);
    sd.scalars("stage_V", dm.V);
    if (in.nuEffCell) sd.scalars("stage_nuEff", *in.nuEffCell);

    MomentumMatrix MU;
    assembleUEqn(MU, dm, dbU, f.Ux, f.Uy, f.Uz, mi);

    // The momentum system as assembled and relaxed. MU.diag is OpenFOAM's PRE-relax diagonal (relax()
    // writes MU.relaxedDiag beside it rather than over it) and MU.source is its POST-relax source, so
    // these line up with dumpSimpleFoam's stage_UDiag0 and stage_USrc respectively.
    if (sd.on)
    {
        sd.scalars("stage_UDiag0", MU.diag);
        sd.scalars("stage_UDiag", MU.relaxed ? MU.relaxedDiag : MU.diag);
        sd.scalars("stage_UUpper", MU.upper);
        sd.scalars("stage_ULower", MU.lower.size() ? MU.lower : MU.upper);
        // The source with boundaryCoeffs folded into their face cells -- dumpSimpleFoam's stage_USrc is
        // defined the same way, and a capture that folded differently would measure the capture.
        DeviceBuffer<scalar> fs[3];
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> diagC;
            deviceFold(dm, MU.relaxed ? MU.relaxedDiag : MU.diag, MU.source[k], MU.iC[k], MU.bC[k],
                       diagC, fs[k]);
        }
        sd.vectors("stage_USrc", fs[0], fs[1], fs[2]);
    }

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
        // U's residual is cmptMax over the components OpenFOAM SOLVES, and the components it solves are
        // the ones polyMesh::solutionD() leaves valid: solveSegregated skips every -1
        // (fvMatrixSolve.C:164) and solutionControl compares cmptMax over the per-component vector it
        // stores (solutionControl.C:232, simpleControl.C:67-71), where a skipped component is Zero.
        //
        // This loop solved all three and reported component 0, and the two errors hid each other.
        // Reporting component 0 is wrong whenever Uy's initial residual exceeds Ux's -- on
        // validation/simpleBoxIO that is every iteration from the second (OpenFOAM at iteration 2:
        // Ux 3.226738e-01, Uy 6.040719e-01) -- so residualControl fires at a different iteration from
        // OpenFOAM's. A max over three components SOLVED unconditionally is the opposite error: the
        // empty direction's system has a ~0 right-hand side and a ~0 field, so its normFactor-scaled
        // residual never leaves O(0.1) and would block convergence on every 2D case. Hence the mask AND
        // the max, not either alone. The skipped solve was also 26% of this fixture's momentum
        // linear algebra (497 of 1912 BiCGStab iterations over 15 outer steps) for an answer of 4.9e-17.
        // THE SOLVE IS NOT SKIPPED, and that is a KNOWN DEVIATION, not an oversight. OpenFOAM's Uz on a
        // 2D case is BIT-EXACTLY zero -- emptyFvPatch::size() is 0, so no z quantity is ever formed --
        // and `U = HbyA - rAtU*grad(p)` therefore reproduces zero forever without the solve. brae's z
        // quantities are round-off nonzero instead (measured on pitzDaily at iteration 1: the momentum
        // source's z norm is 3.5e-21 against 6.1e-04 in x), and the z map amplifies at ~1.15 per
        // iteration, so dropping the solve let Uz reach 13% of |U| by iteration 200 and turned seven
        // end-to-end gates red. Today the z solve is what holds Uz down. Removing it needs the knocked-
        // out direction to be exactly zero first; that is queued, and until then this loop solves a
        // component OpenFOAM does not.
        scalar uInitialResidual = 0.0;
        // Every solved component's system first: the fold (fvMatrixSolve.C's addBoundaryDiag per
        // component into its own diagonal, its own source), the view over the SHARED upper/lower, and
        // its normFactor -- none of which reads another component's psi, so building them all before
        // any solve is the same arithmetic as building each just before its own.
        DeviceBuffer<scalar> diagC[3], b[3], dnf[3];
        DeviceLduView A[3];
        int solved[3];
        int nSolved = 0;
        for (int k = 0; k < 3; ++k)
        {
            if (in.solutionD[k] < 0) continue;   // TRIAL: item 35's skip
            deviceFold(dm, Mp.relaxed ? Mp.relaxedDiag : Mp.diag, Mp.source[k], Mp.iC[k], Mp.bC[k], diagC[k], b[k]);
            A[k] = foldedView(dm, Mp, diagC[k]);
            deviceNormFactorInto(A[k], *U[k], b[k], w.ones, dnf[k]);   // stays on the device (item 66)
            solved[nSolved++] = k;
        }
        // The solver the case asked for, algorithm included: deviceSymGaussSeidel is OpenFOAM's
        // smoothSolver stopping rule around symGaussSeidelSmoother.C's own index-order sweep,
        // level-scheduled onto the device (device_sym_gauss_seidel.cuh), and tests/gs_ladder holds
        // it to OpenFOAM's per-sweep residual at 2.8e-12. It used to be a MULTICOLOUR sweep, which
        // reached T3A's relTol 0.1 in 9-10 sweeps against OpenFOAM's 4-5 and made the case
        // limit-cycle. Only `GaussSeidel` (ascending only in OF) is still substituted, and announced.
        // Internal-face LDU only -- no coupled interfaces, which this path refuses anyway.
        //
        // The components' walks are FUSED (item 60a): one level walk updates every still-active
        // component at each level, so the per-level latency is paid once per sweep instead of once
        // per component. Each keeps its own diagonal, source, normFactor, residual and stop, so the
        // result is the per-component solves' to the bit (tests/gs_fused_identity); BRAE_GS_FUSED=0
        // restores those.
        DeviceSolverPerf perfs[3];
        if (in.uSymGaussSeidel)
        {
            GSFusedComponent comps[3];
            DeviceSolverPerf fp[3];
            for (int i = 0; i < nSolved; ++i)
            {
                const int k = solved[i];
                comps[i] = {&A[k], &b[k], U[k], 1.0, dnf[k].data()};
            }
            deviceSymGaussSeidelFused(nSolved, comps, in.tolU, in.relTolU, in.maxIterU, in.minIterU, in.nSweepsU,
                                      in.uGaussSeidelSymmetric, fp);
            for (int i = 0; i < nSolved; ++i) perfs[solved[i]] = fp[i];
        }
        else
        {
            for (int i = 0; i < nSolved; ++i)
            {
                const int k = solved[i];
                perfs[k] = deviceJacobiBiCGStab(A[k], b[k], *U[k], dnf[k].data(), in.tolU, in.relTolU, in.maxIterU,
                                                /*checkEvery*/1, in.minIterU);
            }
        }
        for (int i = 0; i < nSolved; ++i)
        {
            const int k = solved[i];
            const DeviceSolverPerf perf = perfs[k];
            // The REPORT is masked even where the solve is not: the empty direction's system has a ~0
            // right-hand side and a ~0 field, so its normFactor-scaled residual never leaves O(0.1)
            // (measured on simpleBoxIO: Uz 7.945e-01 at iteration 1, still 9.724e-02 at 15) and a max
            // over three would block residualControl on every 2D case. OpenFOAM's own max is over the
            // components it solved, with the rest at Zero -- which is this.
            if (in.solutionD[k] > 0)
                uInitialResidual = std::max(uInitialResidual, perf.initialResidual);
            // Inside the masked loop deliberately: OpenFOAM prints no `Solving for Uz` line for a
            // component it did not solve, so a skipped k must leave no [Uk] line either.
            if (std::getenv("BRAE_SOLVER_ITERS"))
                std::printf("    [U%d] nIter=%d init=%.3e final=%.3e\n",
                            k, perf.nIterations, perf.initialResidual, perf.finalResidual);
            w.report.push_back({std::string("U") + "xyz"[k], perf.initialResidual, perf.finalResidual,
                                perf.nIterations});
        }
        res["U"] = uInitialResidual;
    }

    // U out of the momentum predictor. UEqn.H() below is built from THIS field, so it sits between
    // the matrix and HbyA and separates a solver difference from an assembly one.
    sd.vectors("stage_Upred", f.Ux, f.Uy, f.Uz);

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
    for (int cmpt = 0; cmpt < 3; ++cmpt) pin.solutionD[cmpt] = in.solutionD[cmpt];

    PressureStages st;
    pressurePredictor(st, dm, dbU, MU, f.Ux, f.Uy, f.Uz, pin, &dbP, &f.p);

    // rAU and rAtU, the pair the whole SIMPLEC question is about. dumpSimpleFoam writes the reciprocal
    // of rAtU as stage_rowSum -- 1/rAU - H1, which is the relaxed matrix row sum over V -- so H1 needs no
    // separate capture here: V/rAtU is the same quantity and is what pEqn.cu actually inverts.
    sd.scalars("stage_rAU", st.rAU);
    sd.scalars("stage_rAtU", st.rAtU);
    sd.vectors("stage_HbyA", st.HbyA[0], st.HbyA[1], st.HbyA[2]);
    sd.scalars("stage_phiHbyA", st.phiHbyAInt);

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

        DeviceBuffer<scalar> dnf;
        deviceNormFactorInto(A, f.p, b, w.ones, dnf);                 // stays on the device (item 66)
        const DeviceSolverPerf perf =
            deviceAMGPCG(A, w.amg, b, f.p, dnf.data(), in.tolP, in.relTolP, in.maxIterP,
                         in.captureVcycle, in.pcgCheckEvery, /*corrScaling*/false, in.minIterP);
        if (corr == 1) res["p"] = perf.initialResidual;   // the FIRST solve's residual, as OpenFOAM reports
        w.report.push_back({"p", perf.initialResidual, perf.finalResidual, perf.nIterations});
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

    // The end of the step, before turbulence->correct() moves nut: p and U after the corrector and the
    // conservative flux the next assembly convects by.
    sd.scalars("stage_pOut", f.p);
    sd.vectors("stage_Uout", f.Ux, f.Uy, f.Uz);
    sd.scalars("stage_phiOut", f.phiInt);

    // simpleFoam.C:93-94 -- laminarTransport.correct(); turbulence->correct().
    if (in.correct) in.correct();

    return res;
}

} // namespace gpu
} // namespace brae
