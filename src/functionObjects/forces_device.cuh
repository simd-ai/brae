#pragma once
// functionObjects::forces' face integration on the device (forces.C calcForcesMoments, the pressure-field branch):
//
//     fP = rhoP*Sf*(p_b - pRef)                              pressure force on each selected boundary face
//     fV = Sf & (-rhoV*nuEff_b*devTwoSymm(gradU_b))          viscous force
//     mP = (Cf - origin) ^ fP,  mV = (Cf - origin) ^ fV      their moments about the coordinate system's origin
//
// gradU_b is fvc::grad(U)'s boundary value: the cell's gradient under the case's grad(U) scheme, corrected in the
// face-normal direction to the patch's own snGrad (gaussGrad::correctBoundaryConditions):
//     gradU_b = gradU_c + n*((U_b - U_c)*deltaCoeffs_b - (n & gradU_c))
// where (U_b - U_c)*deltaCoeffs_b is fvPatchField::snGrad() for every non-coupled condition whose value is its own
// (fixedValue and its kin, zeroGradient, mixed, fixedGradient, slip and symmetry: each value carries its gradient).
//
// THE SUMS ARE OPENFOAM'S PER PATCH: addToPatchFields adds sum(fP) of one patch at a time, in the patch set's
// ascending order. Here each selected patch is one block: 256 threads each walk a contiguous run of the patch's
// faces in order, and the block adds the 256 partial sums in a fixed tree. The same faces in the same order give
// the same bits on every run (no atomics), and the caller adds the patches in order, as OpenFOAM does. A face's
// terms are not stored: on the official DrivAer's car (millions of faces) twelve doubles a face would be hundreds
// of MB for twelve numbers.
#include "cf_types.cuh"
#include "device_buffer.cuh"

#include <vector>

namespace brae
{

// One forces/forceCoeffs selection: its faces in OpenFOAM's summation order (the selected patches by ascending
// index, each patch's faces in order), uploaded once.
struct DeviceForceFaces
{
    int nFaces = 0;
    std::vector<int> patchOffset{0};   // host: where each selected patch's faces start in the lists below, then their end
    DeviceBuffer<label> bnd;           // each face's place in the solver's boundary arrays (non-coupled patches, in order)
    DeviceBuffer<scalar> Sf;           // 3 a face
    DeviceBuffer<scalar> Cf;           // 3 a face
};

// The fields a face integration reads, device pointers in the solver's own layouts.
struct DeviceForceFields
{
    const label* faceCellB = nullptr;  // every boundary face's cell (DeviceMesh::bndCell)
    const scalar* dcB = nullptr;       // every boundary face's patch deltaCoeffs, 1/(nf & (Cf - C)) (DeviceBoundary::deltaCoeffs)
    const scalar* pB = nullptr;        // p at every boundary face
    const scalar* UB[3] = {};          // U's components at every boundary face
    const scalar* U[3] = {};           // U's components at every cell
    const scalar* gradU = nullptr;     // the cells' fvc::grad(U), deviceGradU's layout: d U_j / d x_i at (3*i + j)*nCells + cell
    int nCells = 0;
    const scalar* nuEffB = nullptr;    // nuEff at every boundary face (nut's patch value + nu)
    const scalar* rhoB = nullptr;      // rho at every boundary face; null: rhoV everywhere
    scalar rhoP = 1;                   // the pressure force's density: 1 for a dynamic p, rhoInf for a kinematic one
    scalar rhoV = 1;                   // the viscous stress's density where rhoB is null (forces::rho(patchi): rhoInf)
    scalar pRef = 0;                   // pRef/rhoRef, the pressure the face pressure is taken from
    vector origin{0, 0, 0};
    // TEST CONTROL BRAE_CONTROL_FORCES_GRAD_CELL (false): the cell's gradient at the face, without gaussGrad's boundary correction
    bool correctGrad = true;
};

// Per selected patch, twelve sums in order: fP (3), fV (3), mP (3), mV (3). Host vector of 12*patches.
std::vector<double> deviceForcePatchSums(
    const DeviceForceFaces& faces,
    const DeviceForceFields& fields);

} // namespace brae
