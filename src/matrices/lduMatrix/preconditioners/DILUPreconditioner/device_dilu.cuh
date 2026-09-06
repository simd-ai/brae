#pragma once
// DILU: OpenFOAM's diagonal-based incomplete-LU preconditioner for ASYMMETRIC lduMatrices, which is what
// `preconditioner DILU;` selects for U, nuTilda, k and epsilon in essentially every pimpleFoam tutorial.
//
// WHY IT IS NOT OPTIONAL. brae substituted Jacobi and said so (`NOTICE [approximated] solvers/U
// preconditioner`), which reads like a cost/benefit choice and is not one on a stiff mesh. Both
// preconditioners reach the requested relTol; what differs is WHERE the remaining error sits. Jacobi
// damps each cell by its own diagonal and leaves the error in the strongly-coupled directions, which on
// a boundary-layer mesh is precisely the wall-normal one. On pimpleFoam/LES/vortexShed (first cells
// ~1.6e-05 against nu 1e-05 and dt 0.01, a near-wall diffusion number of ~390) the case's own
// `relTol 0.1` therefore left the error concentrated in the stiffest modes, and the PIMPLE outer loop
// amplified rather than damped it: |U| reached 1.2e+10 against OpenFOAM's 0.0435.
//
// THE ALGORITHM IS SEQUENTIAL, which is the whole difficulty on a GPU. OF (DILUPreconditioner.C):
//
//     calcReciprocalD:  for face: rD[u[face]] -= upper[face]*lower[face]/rD[l[face]];   then rD = 1/rD
//     precondition:     w = rD*r
//                       forward,  faces in losort order: w[u[f]] -= rD[u[f]]*lower[f]*w[l[f]]
//                       backward, faces in reverse order: w[l[f]] -= rD[l[f]]*upper[f]*w[u[f]]
//
// Every one of those is a recurrence. The usual GPU answers -- multi-colour or Chow-Patel iterative ILU
// -- change the arithmetic and give a DIFFERENT preconditioner, which would leave brae unable to say it
// runs the case OpenFOAM was given. So this uses LEVEL SCHEDULING instead, which is exact:
//
//   * In OF's face ordering every internal face has owner < neighbour, and faces are sorted by owner. So
//     all faces with upper == c precede all faces with lower == c, and each recurrence above is a
//     well-defined DAG over cells: forward, cell c needs every lower-neighbour l < c; backward, every
//     upper-neighbour u > c.
//   * Cells at the same DAG depth are mutually independent, so a level can be evaluated in parallel.
//   * Evaluating level by level yields the SAME VALUES as the sequential sweep, and -- because each
//     cell's contributions are gathered in increasing face index, which is the order losort visits them
//     -- the same floating-point summation order too. test_dilu asserts bit-equality against a direct
//     transcription of the loops above.
//
// The cost is nLevels kernel launches per sweep. That is the price of being exact; a wrong-but-parallel
// preconditioner is not a cheaper version of this, it is a different solver.
//
// The interfaces (cyclic/AMI) are deliberately absent: OF's preconditioners see only the local matrix,
// so the interface coupling is not in rD there either. A preconditioner does not have to be a good
// approximation of the whole operator, only a consistent one, and matching OF here is what keeps the
// iterate comparable.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include <vector>

namespace brae {

// The level schedule + the preconditioned diagonal. Structure depends only on the mesh, so build() is
// called once; update() recomputes rD from the current matrix and must be called whenever the matrix
// changes (i.e. every solve -- the momentum diagonal moves every outer corrector).
struct DeviceDilu
{
    int nCells = 0, nFaces = 0;
    bool valid = false;

    DeviceBuffer<scalar> rD;        // 1/diag of the incomplete factorisation (OF rD_)
    DeviceBuffer<scalar> work;      // the un-inverted diagonal during update()

    // cells ordered by level, with per-level [begin,end) in the *Off arrays (host-side: they drive the
    // launch loop). fwd = forward sweep order, bwd = backward.
    DeviceBuffer<label> fwdCells, bwdCells;
    std::vector<int>    fwdOff, bwdOff;
    // ...and on the device, for the single-block walk (item 70): when no level outgrows one block the
    // whole sweep is ONE launch with __syncthreads between levels, instead of a launch per level.
    DeviceBuffer<label> fwdOffD, bwdOffD;
    int maxLevelWidth = 0;

    int levels() const { return (int)fwdOff.size() - 1; }
};

// Build the level schedule from the mesh addressing (host arrays: the DAG is a host computation and the
// result is uploaded once). owner/nei are the internal-face owner/neighbour lists, owner < nei.
DeviceDilu buildDeviceDilu(const std::vector<label>& owner, const std::vector<label>& nei, label nCells);

// Recompute rD from the current matrix (OF DILUPreconditioner::calcReciprocalD).
void diluUpdate(const DeviceLduView& A, DeviceDilu& d);

// w = M^-1 r (OF DILUPreconditioner::precondition). Safe to call with w aliasing nothing else.
void diluApply(const DeviceLduView& A, const DeviceDilu& d, const DeviceBuffer<scalar>& r,
               DeviceBuffer<scalar>& w);

}   // namespace brae
