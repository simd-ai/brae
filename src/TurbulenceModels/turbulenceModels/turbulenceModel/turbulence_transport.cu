#include "turbulence_transport.cuh"
#include "device_mesh.cuh"     // deviceDivUpwindCoeffs / deviceDivLimitedCoeffs / deviceLaplacian*
#include "device_kepsilon.cuh" // deviceGaussGrad, deviceCellLimitGrad, deviceBCValue
#include "device_blas.cuh"     // deviceAxpy
#include "device_fvoptions.cuh"   // deviceSetValues: fvMatrix::setValues
#include "device_pcg.cuh"
#include "device_amg.cuh"      // deviceSymGaussSeidel, when the case names a smoothSolver
#include "device_simple.cuh"    // deviceRelaxDiag -- fvMatrix::relax
#include <cstdio>
#include <string>
#include <cmath>

namespace brae {
namespace gpu {
namespace turbulence {

namespace {
// Each device module in the tree carries its own copy of this two-liner (rhoEEqn.cu, rhoPEqn.cu,
// rhoPcEqn.cu, kEpsilon.cu). Kept local here for the same reason rather than adding a public symbol
// for a memset.
void zeroed(DeviceBuffer<scalar>& b, int n)
{
    b.resize(static_cast<std::size_t>(n));
    cudaCheck(cudaMemsetAsync(b.data(), 0, static_cast<std::size_t>(n) * sizeof(scalar), cudaStreamPerThread),
              "turbulence transport zero");
}
}

void assembleScalarTransport(
    PressureMatrix&             M,
    const DeviceMesh&           dm,
    const DeviceBoundary&       db,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& gammaFace,
    const DeviceBuffer<scalar>& gammaBnd,
    const TransportScheme&      sc)
{
    const int nC = dm.nCells;

    // fvm::div(phi, field). The boundary half carries the flux-conditional switch the caller has
    // already applied to db.
    //
    // limitedLinear is a WEIGHT change, not a correction, so it replaces the upwind coefficients rather
    // than adding to the source -- the same shape the host closures' divWithScheme takes. The limiter's
    // gradient is the field's own Gauss gradient, limited by the case's grad(<field>) cellLimited
    // coefficient when it names one; the corrected-laplacian block below builds the same three buffers
    // the same way, and this is deliberately the identical call sequence so the two cannot drift.
    if (sc.limitedLinear)
    {
        DeviceBuffer<scalar> bval, gx, gy, gz;
        deviceBCValue(db, field, bval);
        deviceGaussGrad(dm, field, bval, gx, gy, gz);
        if (sc.limGradK > scalar(0)) deviceCellLimitGrad(dm, field, bval, gx, gy, gz, sc.limGradK);
        deviceDivLimitedCoeffs(dm, *sc.phiInt, field, gx, gy, gz,
                               scalar(2) / std::fmax(sc.limiterCoeff, scalar(1e-15)),
                               M.diag, M.upper, M.lower);
    }
    else
    {
        deviceDivUpwindCoeffs(dm, *sc.phiInt, M.diag, M.upper, M.lower);
    }
    zeroed(M.source, nC);
    deviceBCDivCoeffs(db, *sc.phiBnd, M.iC, M.bC);

    // - fvm::laplacian(gamma, field).
    {
        DeviceBuffer<scalar> lDiag, lUp, lLo, lIC, lBC;
        deviceLaplacianCoeffs(dm, gammaFace, lDiag, lUp, lLo, sc.correctedLaplacian);
        deviceBCLaplacianCoeffsFace(db, gammaBnd, lIC, lBC);
        deviceAxpy(-1.0, lDiag, M.diag);
        deviceAxpy(-1.0, lUp, M.upper);
        deviceAxpy(-1.0, lLo, M.lower);
        deviceAxpy(-1.0, lIC, M.iC);
        deviceAxpy(-1.0, lBC, M.bC);

        if (sc.correctedLaplacian)
        {
            DeviceBuffer<scalar> bval, gx, gy, gz, ffc, corr;
            deviceBCValue(db, field, bval);
            deviceGaussGrad(dm, field, bval, gx, gy, gz);
            // correctedSnGrad's correction takes the field's OWN grad scheme (correctedSnGrad.C:52-55).
            if (sc.gradFieldLimitK > scalar(0))
                deviceCellLimitGrad(dm, field, bval, gx, gy, gz, sc.gradFieldLimitK);
            if (sc.snGradLimitCoeff > scalar(0.0))
            {
                deviceLaplacianCorrFluxLimited(dm, gammaFace, field, gx, gy, gz, sc.snGradLimitCoeff, ffc);
                deviceFaceDivSource(dm, ffc, corr);
            }
            else
            {
                deviceLaplacianCorr(dm, gammaFace, gx, gy, gz, corr);
            }
            // deviceLaplacianCorr returns -V*div(faceFluxCorr) -- already negated -- and the laplacian
            // itself enters this equation with -1, so its explicit source does too. The two signs
            // compose to the reference's `L.source -= corr` followed by `M -= L`.
            deviceAxpy(-1.0, corr, M.source);
        }
    }
}

void solveScalarEqn(
    PressureMatrix&             M,
    DeviceBuffer<scalar>&       field,
    const DeviceMesh&           dm,
    bool                        relaxEquation,
    scalar                      alpha,
    const DeviceBuffer<label>*  fvoMask,
    const DeviceBuffer<scalar>* fvoVal,
    const DeviceBuffer<label>*  wallMask,
    const DeviceBuffer<scalar>* wallVal,
    const SolveControls&        sv,
    scalar&                     residualOut,
    const std::string&          dumpPrefix,   // "" = no dump; else <dir>/<name> path prefix
    bool                        gs)           // this field's own solver: the case's smoothSolver, or BiCGStab
{
    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nB  = dm.nBndFaces;

    // The guard is "the case NAMES a factor", not "the factor is below 1": fvMatrix::relax early-returns
    // only on alpha <= 0, so relax(1.0) still applies the dominance clamp and adds (D - D0)*psi.
    if (relaxEquation && alpha > scalar(0.0))
    {
        DeviceBuffer<scalar> relaxedDiag, delta, t;
        deviceRelaxDiag(M.view(dm), dm, M.iC, alpha, relaxedDiag, delta);
        deviceCopy(M.diag, relaxedDiag);
        deviceHadamard(t, delta, field);
        deviceAxpy(1.0, t, M.source);
    }

    // Both constraints go through the SAME four kernels, because in OpenFOAM they are the same call --
    // FixedValueConstraint::constrain and epsilonWallFunction::manipulateMatrix both end in
    // fvMatrix::setValues. Order matters and is OpenFOAM's: whichever runs first is the one whose value
    // reaches the neighbours, since setValues zeroes the coefficient it just transferred through.
    auto applySetValues = [&](const DeviceBuffer<label>* mask, const DeviceBuffer<scalar>* val)
    {
        if (!mask || !val) return;
        deviceSetValues(dm, *mask, *val, M.diag, M.upper, M.lower, M.source, M.iC, M.bC, field);
    };
    applySetValues(fvoMask, fvoVal);
    applySetValues(wallMask, wallVal);

    // Fold the boundary coefficients in exactly as fvMatrix::solve does, then solve. BiCGStab, not PCG:
    // upwind convection makes upper != lower, so the matrix is asymmetric and a symmetric solver would
    // be solving a different system.
    DeviceBuffer<scalar> diagC, b, ones;
    deviceFold(dm, M.diag, M.source, M.iC, M.bC, diagC, b);

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

    // Instrument (BRAE_STAGE_DUMP_DIR, see correct()): the FOLDED system as the solver sees it -- diag
    // with the boundary diagonal folded in, source with the boundary source folded in, the two
    // off-diagonals (<prefix>D/Src/Upper/Lower), and the field before and after the solve
    // (<prefix>SolveIn/SolveOut) -- the same convention the host's
    // captureSystem uses (kEpsilon_cpp.cu), so the two arms' assembled systems diff directly.
    auto dump = [&](const char* what, const DeviceBuffer<scalar>& v)
    {
        if (dumpPrefix.empty()) return;
        const std::vector<scalar> h = v.host();
        std::FILE* fp = std::fopen((dumpPrefix + what).c_str(), "w");
        if (!fp) return;
        for (scalar x : h) std::fprintf(fp, "%.17g\n", (double)x);
        std::fclose(fp);
    };
    dump("D", diagC);
    dump("Src", b);
    dump("Upper", M.upper);
    dump("Lower", M.lower);
    dump("SolveIn", field);

    // the ones vector kept across calls (item 63) and the normFactor kept on the device (item 66)
    DeviceBuffer<scalar> dnf;
    deviceNormFactorInto(A, field, b, deviceOnes(nC), dnf);
    // The solver the case asked for (item 58). The view above is internal-face only, which is what the
    // level-scheduled sweep needs; there is no interface to drop silently.
    DeviceSolverPerf perf;
    if (gs)
        deviceSymGaussSeidel(A, b, field, dnf.data(), sv.tol, sv.relTol, sv.maxIter, &perf, sv.minIter,
                             sv.nSweeps, sv.gsSymmetric);
    else
        perf = deviceJacobiBiCGStab(A, b, field, dnf.data(), sv.tol, sv.relTol, sv.maxIter, /*checkEvery=*/1, sv.minIter,
                                    sv.precon, /*amg=*/nullptr, sv.polyDeg);
    residualOut = perf.initialResidual;
    dump("SolveOut", field);
}


} // namespace turbulence
} // namespace gpu
} // namespace brae
