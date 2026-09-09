#include "turbulence_transport.cuh"
#include "device_mesh.cuh"     // deviceDivUpwindCoeffs / deviceDivLimitedCoeffs / deviceLaplacian*
#include "device_kepsilon.cuh" // deviceGaussGrad, deviceCellLimitGrad, deviceBCValue
#include "device_blas.cuh"     // deviceAxpy
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

} // namespace turbulence
} // namespace gpu
} // namespace brae
