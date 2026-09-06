// rhoTurbulenceHook.cu -- see the header for what this replaces and why it is shared.
#include "rhoTurbulenceHook.cuh"

#include "device_blas.cuh"        // deviceDivide, deviceCopy -- already gated, not re-written here
#include "device_kepsilon.cuh"    // deviceGatherWallNu: boundary-face -> wall-face ordering
#include "transport_model.cuh"   // transportMu -- the SAME function the host reference calls
#include <stdexcept>
#include <vector>

namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// nu = mu(T)/rho, the LAMINAR kinematic viscosity. The host reference computes exactly this
// (rhoSimpleFoam_cpp.cu, nuLam/nuLamBnd) with the same transportMu, so cells and boundary faces run one
// kernel. The rho guard is the boundary's: a boundary face of a patch brae pads rather than solves can
// carry rho 0, and dividing by it would put an inf into the closure's diffusivity.
__global__ void nuFromTKernel(
    int                        n,
    const scalar* __restrict__ T,
    const scalar* __restrict__ rho,
    ThermoCoeffs               c,
    scalar* __restrict__       nu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar r = rho[i];
    nu[i] = transportMu(T[i], c) / (r > scalar(0) ? r : scalar(1));
}

// The face-wise divide and the wall-face gather are NOT here: deviceDivide (device_blas.cuh, out = a./b
// with 0 where b == 0 -- the same guard) and deviceGatherWallNu (device_kepsilon.cuh, nuWall[i] =
// nuBnd[idx[i]]) already exist and are already gated. A private copy of either would be a second
// implementation of arithmetic the tree has one of.

} // namespace

void correctTurbulence(
    RhoSolverFields&             f,
    const RhoDeviceFields&       dev,
    const DeviceMesh&            dm,
    DeviceVectorBoundary&        dbU,
    const ThermoCoeffs&          thermo,
    const TurbulenceHookOptions& opt,
    TurbulenceHookBuffers&       buf)
{
    const int nC  = static_cast<int>(f.rho.size());
    const int nBF = dm.nBndFaces;
    const int nIF = dm.nInternalFaces;
    if (nC == 0) return;

    // ---- nu, cells and boundary faces -----------------------------------------------------------
    if (static_cast<int>(buf.nuCell.size()) != nC)  buf.nuCell.resize(nC);
    if (static_cast<int>(buf.nuBnd.size()) != nBF)  buf.nuBnd.resize(nBF);
    nuFromTKernel<<<nBlocks(nC), TPB>>>(nC, f.T.data(), f.rho.data(), thermo, buf.nuCell.data());
    cudaCheck(cudaGetLastError(), "rho turbulence hook: nu cells");
    if (nBF > 0)
    {
        nuFromTKernel<<<nBlocks(nBF), TPB>>>(nBF, f.TBnd.data(), f.rhoBnd.data(), thermo,
                                             buf.nuBnd.data());
        cudaCheck(cudaGetLastError(), "rho turbulence hook: nu boundary");
    }

    // ---- nu on the WALL faces, gathered ---------------------------------------------------------
    // The index map is uploaded ONCE: it is mesh topology and does not change between iterations.
    const int nWF = static_cast<int>(dev.wfFaceOfBnd.size());
    if (nWF > 0)
    {
        if (static_cast<int>(buf.wallFaceOfBnd.size()) != nWF) buf.wallFaceOfBnd.copyFrom(dev.wfFaceOfBnd);
        if (static_cast<int>(buf.nuWall.size()) != nWF)        buf.nuWall.resize(nWF);
        deviceGatherWallNu(buf.wallFaceOfBnd, buf.nuBnd, buf.nuWall);
    }

    // ---- phiByRho: compressibleTurbulenceModel::phi(), the VOLUMETRIC flux -----------------------
    // divU is a dilatation and must come from this, not from the mass flux the div operator uses --
    // they differ by rho, and using the mass flux there puts a density into a dilatation.
    if (static_cast<int>(buf.phiByRhoInt.size()) != nIF) buf.phiByRhoInt.resize(nIF);
    if (static_cast<int>(buf.phiByRhoBnd.size()) != nBF) buf.phiByRhoBnd.resize(nBF);
    if (nIF > 0)
    {
        deviceInterpolate(dm, f.rho, buf.rhoFace);   // fvc::interpolate(rho), internal faces
        deviceDivide(buf.phiByRhoInt, f.phiInt, buf.rhoFace);
    }
    if (nBF > 0)
    {
        // The PATCH rho, not an interpolated face value: effectiveFaceViscosity replaces the
        // interpolated boundary with the field's own boundary values, and the host reference divides
        // the boundary flux by exactly that (rhoSimpleFoam_cpp.cu, the phiByRho block).
        deviceDivide(buf.phiByRhoBnd, f.phiBnd, f.rhoBnd);
    }

    // ---- the entering wall viscosity, SNAPSHOT ---------------------------------------------------
    // nutBnd is also the output correct() overwrites; aliasing the two makes the closure read a value
    // it has already replaced partway through.
    if (f.nutBnd.size() > 0) deviceCopy(buf.nutBndIn, f.nutBnd);
    // ...and in WALL-face order, for the wall functions: OpenFOAM's G0 reads the STORED nut patch
    // value (nutw[facei]), which is this snapshot, not a recomputation from the current k and nu_w.
    if (nWF > 0 && buf.nutBndIn.size() > 0)
    {
        if (static_cast<int>(buf.nutWallIn.size()) != nWF) buf.nutWallIn.resize(nWF);
        deviceGatherWallNu(buf.wallFaceOfBnd, buf.nutBndIn, buf.nutWallIn);
    }

    // ---- the closure ------------------------------------------------------------------------------
    kEpsilonRAS::KEpsilonInput kin;
    kin.phiInt = &f.phiInt;              kin.phiBnd = &f.phiBnd;
    kin.phiByRhoInt = &buf.phiByRhoInt;  kin.phiByRhoBnd = &buf.phiByRhoBnd;
    kin.rhoCell = &f.rho;                kin.rhoBndFace = &f.rhoBnd;
    kin.nuCell = &buf.nuCell;            kin.nuBndFace = &buf.nuBnd;
    kin.nuWallFace = &buf.nuWall;
    kin.nutBndFace = &buf.nutBndIn;
    kin.nutWallFace = (nWF > 0) ? &buf.nutWallIn : nullptr;
    kin.wfBndMask = &dev.wfBndMask;      kin.wallYBndFace = &dev.wallYBndFace;
    kin.Ux = &f.Ux; kin.Uy = &f.Uy; kin.Uz = &f.Uz;
    // Passing null here is NOT "no turbulent inlet" -- it is silently no turbulent inlet at all, on a
    // case whose 0/k asks for one.
    if (dev.hasTurbulentInlet)
    {
        kin.turbInletKMask   = &dev.turbInletKMask;
        kin.turbInletKInt    = &dev.turbInletKInt;
        kin.turbInletEpsMask = &dev.turbInletEpsMask;
        kin.turbInletEpsLen  = &dev.turbInletEpsLen;
    }
    kin.alphatWallMask = &dev.alphatWallMask;
    kin.alphatPrtFace  = &dev.alphatPrtFace;
    kin.co  = opt.co;
    kin.Prt = opt.Prt;
    kin.boundedK = kin.boundedEps = opt.bounded;
    kin.hasNonUpwindDivScheme = !opt.divSchemeUnsupported.empty();
    kin.divSchemeUnsupported  = opt.divSchemeUnsupported;
    kin.correctedLaplacian = opt.correctedLaplacian;
    kin.relaxEquationK   = opt.relaxEquationK;   kin.relaxK   = opt.relaxK;
    kin.relaxEquationEps = opt.relaxEquationEps; kin.relaxEps = opt.relaxEps;
    kin.tol = opt.tol; kin.relTol = opt.relTol; kin.maxIter = opt.maxIter; kin.minIter = opt.minIter;
    kin.gsK = opt.gsK; kin.gsEps = opt.gsEps; kin.gsSymmetric = opt.gsSymmetric; kin.nSweepsKE = opt.nSweepsKE;
    kin.precon     = opt.precon;
    kin.fvoKMask   = opt.fvoKMask;    kin.fvoKVal   = opt.fvoKVal;
    kin.fvoEpsMask = opt.fvoEpsMask;  kin.fvoEpsVal = opt.fvoEpsVal;

    kEpsilonRAS::correct(f.k, f.epsilon, f.nut, f.nutBnd, &f.alphat, &f.alphatBnd,
                         buf.stages, dm, dbU,
                         const_cast<DeviceBoundary&>(dev.dbK),
                         const_cast<DeviceBoundary&>(dev.dbEps),
                         const_cast<DeviceWallData&>(dev.wall), kin);
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
