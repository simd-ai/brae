#pragma once
// brae::attachProcessorCoupling, the SETUP-TIME coupling of a rank's `processor` patches (InterFoamStep.md WP1,
// MeshBoundaryIO.md item 3). One collective, called by every rank in the same order right after the mesh and its
// FvGeometry exist:
//   1. every rank sends, per neighbour, the 9 doubles a face of OpenFOAM's processorPolyPatch::initGeometry
//      sends (Cf, Sf, owner-cell centre) -- ProcComm::exchangeFaceData;
//   2. the HaloPlan is built from the patches (ProcComm::buildPlanFromPatches): distinct neighbour cells become
//      halo slots nC + k, grouped per neighbour;
//   3. every processor patch gets its coupled half (coupleProcessorPair: weights, delta, deltaCoeffs,
//      nonOrthDeltaCoeffs, nonOrthCorrectionVectors with processorFvPatch's formulas) and its nbrFaceCells =
//      the halo slot of the neighbour cell of each face.
// Zero-size processor patches (decomposePar keeps none of them for a pair without a cut, but a reader may) are
// skipped on both sides: sizes of face i of my patch and of the neighbour's patch agree or the exchange throws.
#include "fv_patch.cuh"
#include "fv_geometry.cuh"
#include "proc_comm.cuh"

#include <memory>
#include <vector>

namespace brae {

struct ProcessorCoupling
{
    std::unique_ptr<HaloPlan> plan;           // the halo plan of this rank (no neighbours at np == 1)
    std::vector<label> patches;               // indices into the FvPatch list of the coupled processor patches
    label nCells = 0;                         // owned cells (nC)
    label nHalo = 0;                          // halo slots; cell arrays that processor operators read have nC + nHalo
    label nProcFaces = 0;                     // faces of all those patches, in patch order = plan processor-face order
    label firstProcFace = 0;                  // mesh face index of the first of them (they are the LAST faces of the mesh)
};

// COLLECTIVE. Couples every `processor` patch with faces in `patches` in place and returns the plan. Throws (on
// every rank, through collectiveAbort) when a processor patch is not of this rank, names a rank outside the
// communicator, repeats a neighbour (a processorCyclic is refused elsewhere), or the processor faces are not the
// last faces of the mesh.
ProcessorCoupling attachProcessorCoupling(
    std::vector<FvPatch>& patches,
    const FvGeometry& g,
    label nCells,
    ProcComm& comm);

} // namespace brae
