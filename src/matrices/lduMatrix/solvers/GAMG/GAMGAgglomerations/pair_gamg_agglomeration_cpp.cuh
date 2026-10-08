#pragma once
// OpenFOAM's faceAreaPair GAMG agglomeration, the host reference: which cells each coarse level
// merges, and -- as much a part of the answer -- how the coarse cells and faces are NUMBERED.
//
// provenance:
//   openfoam:  src/finiteVolume/fvMatrices/solvers/GAMGSymSolver/GAMGAgglomerations/
//                  faceAreaPairGAMGAgglomeration/faceAreaPairGAMGAgglomeration.C:58-101 (the weights)
//              src/OpenFOAM/matrices/lduMatrix/solvers/GAMG/GAMGAgglomerations/
//                  pairGAMGAgglomeration/pairGAMGAgglomerate.C:34-154 (the level loop),
//                      :159-330 (one pairing pass)
//                  pairGAMGAgglomeration/pairGAMGAgglomeration.C:36 (forward_, a STATIC)
//                  GAMGAgglomeration/GAMGAgglomeration.C:215-235 (continueAgglomerating),
//                      :248-286 (maxLevels 50, nCellsInCoarsestLevel 10 clamped to nCells/2),
//                      :83-212 (printLevels, the oracle this is gated against)
//                  GAMGAgglomeration/GAMGAgglomerateLduAddressing.C:36-225 (coarse addressing)
//                  GAMGAgglomeration/GAMGAgglomerationTemplates.C:98-127 (restrictFaceField)
//              src/OpenFOAM/matrices/lduMatrix/lduAddressing/lduAddressing.C:272-299 (band)
//   tests:     tests/test_inter_gamg_vs_openfoam.cu holds every level's cell count, face count and band
//              profile against the table OpenFOAM prints under `DebugSwitches { GAMGAgglomeration 1; }`
//
// WHY THE NUMBERING IS PART OF THE ANSWER. Every smoother OpenFOAM runs on a coarse level walks that
// level's faces in order -- DIC's two substitution passes, Gauss-Seidel's ascent -- so two hierarchies
// that merge the same cells and number them differently are two different preconditioners, and the
// V-cycle they make stops in a different place. brae::gamg (src/OpenFOAM/matrices/gamg.cu) numbers
// coarse faces through a sorted map, which is a valid multigrid and not OpenFOAM's.
//
// FOUR THINGS THAT ARE EASY TO GET WRONG, all read from the source above:
//
//   THE WEIGHTS ARE PERTURBED. Not magSf but |Sf/sqrt(magSf) * (1, 1.01, 1.02)| component-wise, "to
//   avoid jagged agglomerations on axis-aligned meshes": on a mesh of equal squares every face weighs
//   the same and the pairing would follow the face order alone. requiresUpdate() is
//   `timeIndex % updateInterval == 0` with updateInterval 1 by default, so this branch is the one a
//   static mesh takes; the other (plain magSf) needs `updateInterval` set and is refused.
//
//   THE DIRECTION ALTERNATES, AND THE FLAG IS STATIC. forward_ flips after EVERY pairing pass --
//   including the last one, whose result continueAgglomerating() throws away -- and it belongs to the
//   class, not the object: a second agglomeration built in the same run starts where the first left
//   it. The caller owns the flag here for that reason.
//
//   A REJECTED LEVEL IS DISCARDED WHOLE. The pass that would take the coarsest level under
//   nCellsInCoarsestLevel is computed and dropped, so the coarsest level has AT LEAST that many cells.
//
//   COARSE FACES ARE NUMBERED BY OWNER, THEN BY FIRST APPEARANCE. Walking the fine faces in order,
//   each new (coarse owner, coarse neighbour) pair is appended to its owner's list; the final order is
//   owner-major with each owner's faces in the order the walk met them -- NOT sorted by neighbour.
//
// NOT PORTED, refused where the entry is read: mergeLevels above 1 (combineLevels), any agglomerator
// but faceAreaPair, processor agglomeration (this is a serial solver), coupled interfaces.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "primitive_mesh.cuh"
#include <utility>
#include <vector>

namespace brae {

// lduAddressing of one level: lowerAddr is the owner side, upperAddr the neighbour side
struct GamgLduAddressing
{
    label nCells = 0;
    std::vector<label> lowerAddr;
    std::vector<label> upperAddr;
};

// Indexed as OpenFOAM indexes them: entry i describes the step from level i to level i+1, level 0
// being the mesh itself. meshLevels[i] is OpenFOAM's meshLevels_[i], the addressing of level i+1.
struct GamgAgglomeration
{
    GamgLduAddressing fineMesh;
    std::vector<GamgLduAddressing> meshLevels;
    std::vector<label> nCells;
    std::vector<std::vector<label>> restrictAddressing;
    std::vector<label> nFaces;
    // a fine face maps to a coarse face (>= 0) or, when both its cells merged, to -(coarse cell + 1)
    std::vector<std::vector<label>> faceRestrictAddressing;
    std::vector<std::vector<char>> faceFlipMap;

    label size() const
    {
        return static_cast<label>(nCells.size());
    }
    // GAMGAgglomeration::meshLevel: 0 is the mesh, i > 0 is meshLevels[i-1]
    const GamgLduAddressing& meshLevel(label i) const
    {
        if (i == 0) return fineMesh;
        return meshLevels[static_cast<std::size_t>(i) - 1];
    }
};

// pairGAMGAgglomeration::agglomerate, one pass. Flips `forward` on the way out.
std::vector<label> gamgPairAgglomerate(
    label& nCoarseCells,
    const GamgLduAddressing& fine,
    const std::vector<scalar>& faceWeights,
    bool& forward);

// faceAreaPairGAMGAgglomeration's constructor with the defaults a static mesh takes. `forward` is
// OpenFOAM's static pairGAMGAgglomeration::forward_, true at program start.
GamgAgglomeration faceAreaPairGamgAgglomeration(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    label nCellsInCoarsestLevel,
    bool& forward);

// lduAddressing::band(): (bandwidth, profile). printLevels prints the second, and it moves when the
// numbering does, which is what makes the level table a test of the numbering.
std::pair<label, scalar> gamgLduBand(const GamgLduAddressing& addr);

} // namespace brae
