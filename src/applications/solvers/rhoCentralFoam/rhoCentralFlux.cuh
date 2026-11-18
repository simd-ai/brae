#pragma once
// rhoCentralFoam's central-upwind flux assembly on the device: the twin of rhoCentralFlux_cpp.cu.
//
//   aux, cells and patches       rPsi = 1/psi, c = sqrt(Cp/Cv*rPsi)                       rhoCentralFoam.C:107, 135
//   gradients                    Gauss linear grad of rho, rhoU (3), rPsi, e, c           deviceGaussGradFused (shared)
//   internal faces               directed reconstruction + centralFaceFlux                :102-210
//   patch faces                  the patch value in both directions + centralFaceFlux
// The cell-side gather of these fluxes (fvc::div) is fused into the cell update (rhoCentralStep.cu).
#include "rhoCentralControls.cuh"
#include "rhoCentralState.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"

namespace brae {
namespace gpu {
namespace rhoCentral {

using brae::rhoCentral::Case;

// The device state: cells (nC) and the flat boundary (nB), vectors as components.
struct DeviceFields
{
    DeviceBuffer<scalar> rho, rhoUx, rhoUy, rhoUz, rhoE, e, T, p, psi, Ux, Uy, Uz;
    DeviceBuffer<scalar> rhoB, rhoUxB, rhoUyB, rhoUzB, rhoEB, eB, TB, pB, psiB, UxB, UyB, UzB;
};

// Work the flux assembly leaves for the cell update.
struct DeviceFluxes
{
    DeviceBuffer<scalar> rPsi, c, rPsiB, cB;
    DeviceBuffer<scalar> gx[7], gy[7], gz[7];                       // rho, rhoUx, rhoUy, rhoUz, rPsi, e, c
    DeviceBuffer<scalar> phi, phiUx, phiUy, phiUz, phiEp, amaxSf;   // internal faces
    DeviceBuffer<scalar> phiB, phiUxB, phiUyB, phiUzB, phiEpB, amaxSfB;   // patch faces (0 on empty ones)
    // viscous runs only (milestone G): a_pos*U_pos + a_neg*U_neg on internal and patch faces, for sigmaDotU
    DeviceBuffer<scalar> aUx, aUy, aUz, aUxB, aUyB, aUzB;

    void allocate(int nC, int nIF, int nB);
};

// rhoCentralFoam.C:102-210 on the device. `outflow` (10 doubles: the run's totals, then one step's sums) accumulates
// deltaT * the patch-face fluxes of phi, phiUp and phiEp, a step at a time as the host arm does (diagnostics only;
// atomics, so not bit-reproducible, and never fed back).
// meshPhiI/meshPhiB (milestone K): mesh.phi() on the internal and the boundary faces of a moving mesh, null on a
// static one
void centralFluxes(const DeviceMesh& dm, const Case& cs, int control, DeviceFields& F, DeviceFluxes& X,
                   double* outflow, scalar deltaT, const scalar* meshPhiI = nullptr, const scalar* meshPhiB = nullptr);

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
