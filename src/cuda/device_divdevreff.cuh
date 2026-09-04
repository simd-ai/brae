#pragma once
// cf GPU offload (#2): the explicit divDevReff stress source on device,
//   source += V * fvc::div( nuEff * dev2(T(grad U)) ).
// Steps: grad U (tensor, = 3 component gaussGrads) -> sigma = nuEff*dev2(transpose(gradU)) (cell) ;
// boundary gradient via snGrad correction (gradU_b = gradC + n (x) (snGrad - n & gradC), snGrad =
// (U_b - U_cell)*deltaCoeffs) -> sigma_b ; tensor divergence (Sf & sigma_face) gathered per cell.
// Tensors are packed component-major: comp (i,j) at index (i*3+j)*n + cell.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_cyclic.cuh"
#include "device_ami.cuh"
#include "device_interface.cuh"   // interface<Op>() overloads dispatching to the cyclic/AMI backends
#include "device_halo.cuh"
#include <vector>

namespace brae {

// Processor (multi-GPU) coupling for the stress term -- the distributed counterpart of the cyc/ami hooks.
// A processor face is a COUPLED interface, so unlike a real boundary face its stress is NOT dev2(T(gradB)):
// it is the halo-INTERPOLATED cell sigma (w*own + (1-w)*nbr), exactly as host parallelDivDevReff takes it
// from distributeFromCells<tensor>(sigC, P). Passing this hook also injects the halo U value into gradU.
//   weights[i]   -- procW of interface i (the face interpolation weight of the local side)
//   procStart[i] -- offset of interface i's faces in the flattened boundary array
struct DeviceProcStress
{
    DeviceHalo*                              halo      = nullptr;
    const std::vector<DeviceBuffer<scalar>>* weights   = nullptr;
    const std::vector<label>*                procStart = nullptr;
};

// nuCell (nC), nuBnd (nBndFaces), the effective viscosity at cells / boundary faces (nu for laminar).
// cyc (optional): a periodic interface, its faces contribute to gradU AND to the tensor divergence, so the
// boundary-column stress stays consistent with the interior (else x-invariance drifts on periodic meshes).
// proc (optional): processor interfaces (multi-GPU). nullptr on the single-GPU path -- nothing changes.
void deviceDivDevReff(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                      const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                      const DeviceBuffer<scalar>& nuCell, const DeviceBuffer<scalar>& nuBnd,
                      DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ,
                      const DeviceCyclic* cyc = nullptr, const DeviceAMI* ami = nullptr,
                      const DeviceProcStress* proc = nullptr,
                      // U's STORED boundary values, one per component, when the caller keeps them. OF's
                      // fvc::grad(U) reads U.boundaryField() -- the value the last evaluate left -- and
                      // does NOT re-derive it: updateCoeffs sets a flag and the fvMatrix constructor calls
                      // nothing else (fvPatchField.C, fvMatrix.C:396). Passing null makes this re-derive
                      // with deviceBCValue, which is the same number only while the caller evaluates U's
                      // boundary before every assembly. The OF-mirror does not (queue items 25, 30), so it
                      // passes its stored values; the drivers that do evaluate leave this null and are
                      // unchanged.
                      const DeviceBuffer<scalar>* const* UbStored = nullptr,
                      // grad(U) "cellLimited Gauss linear <k>" coefficient; 0 = unlimited.
                      //
                      // OF's linearViscousStress::divDevReff calls fvc::grad(U), which resolves the
                      // NAMED gradSchemes entry -- so a case asking for `grad(U) cellLimited Gauss
                      // linear 1` gets a limited gradient here too, not just in the linearUpwind
                      // correction. brae built a plain Gauss gradient and ignored the entry.
                      //
                      // It is invisible until nuEff is large, because the whole term carries a factor of
                      // it. Measured on pimpleFoam/RAS/oscillatingInletACMI2D as a FREE run (no restart,
                      // no probe), laminar, 10 steps: at the tutorial's nu = 1e-6 the difference from
                      // OpenFOAM is 6.5e-07 either way, but at nu = 1e-3 -- the size of a turbulent nut --
                      // it is 4.7e-04 unlimited against 6.7e-08 limited. A factor of 7000.
                      scalar gradULimitK = 0.0);

// Exported for the Maxwell model -- see the definitions in device_divdevreff.cu.
void deviceBoundaryGradU(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                         const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                         const DeviceBuffer<scalar>& gradU, DeviceBuffer<scalar>& gradB);
void deviceTensorDivSource(const DeviceMesh& dm,
                           const DeviceBuffer<scalar>& Tcell, const DeviceBuffer<scalar>& Tbnd,
                           DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ,
                           const DeviceCyclic* cyc = nullptr, const DeviceAMI* ami = nullptr);

} // namespace brae
