#pragma once
// brae::PrimitiveMesh, OpenFOAM polyMesh primitive topology.
// Mirrors src/OpenFOAM/meshes/polyMesh: points, faces (CSR), owner, neighbour, boundary
// patches. Internal faces occupy [0, nInternalFaces); boundary faces follow, grouped by
// patch. owner is non-decreasing (upper-triangular face ordering).
#include "cf_types.cuh"
#include "function1.cuh"   // cyclicACMI `scale`: the interface open-area fraction as a function of time
#include "codedPatchFunction1.cuh"   // ...or a per-face coded one
#include "foam_token_reader.cuh"
#include <string>
#include <stdexcept>
#include <vector>

namespace brae {

struct PatchInfo
{
    std::string name;
    std::string type;
    label       start = 0;  // first face index (startFace)
    label       size  = 0;  // number of faces (nFaces)
    std::string neighbourPatch;  // cyclic: paired patch name
    // cyclicACMI: the patch (a wall) that takes the UNCOVERED area fraction. The two are
    // geometrically COINCIDENT -- OF's blockMeshDict gives them the same faces -- and the AMI
    // overlap fraction splits the area between them (cyclicACMIPolyPatch.C:226,252). Empty for
    // every other patch type.
    std::string nonOverlapPatch;
    // cyclicPeriodicAMI: the coupled patch whose transform TILES this interface. The two sides of a
    // periodic AMI need not span the same sector, so OF covers the target by applying this patch's
    // transform to the source repeatedly until the weights sum to 1 (cyclicPeriodicAMIPolyPatch::
    // resetAMI). maxIter bounds that loop (OF default 36); matchTolerance is when to stop (OF 1e-4).
    std::string periodicPatch;
    label       maxIter = 36;
    scalar      matchTolerance = 1e-4;
    std::vector<std::string> inGroups;  // polyBoundaryMesh groups this patch belongs to (boundaryField group match)
    std::string transform;       // cyclic: "translational" / "rotational" / "unknown"
    vector      rotationAxis{0, 0, 1};    // cyclic rotational: axis direction
    vector      rotationCentre{0, 0, 0};  // cyclic rotational: a point on the axis
    // cyclicACMI `scale` (OF cyclicACMIPolyPatch: PatchFunction1<scalar>::NewIfPresent(*this, "scale")):
    // a PRESCRIBED open-area fraction multiplying the geometric overlap mask, re-evaluated every step:
    //     scaledMask = min(1 - tol, max(tol, scale(t)*mask))
    // TJunctionSwitching closes a branch with `table ((0 1)(0.2 1)(0.3 0))`. Empty = no scaling, and
    // then the interface is the geometric overlap alone, exactly as before.
    Function1   acmiScale;
    // ...or `scale { type coded; code #{ ... #}; }`, a PatchFunction1 evaluated per FACE (RAS/
    // damBreakLeakage opens two faces of its baffle at t > 0.5). Only captured here: the one consumer
    // that evaluates it is the interFoam host loop's cyclic_acmi_cpp; applyACMIAreaScaling refuses it.
    bool                     acmiScaleCoded = false;
    CodedPatchFunction1Spec  acmiScaleCodedSpec;
};

// A cyclicACMI's `scale` belongs to the PAIR, not to one patch. OF keeps it on the OWNER (the half with
// the lower patch index, cyclicAMIPolyPatch::owner) and clones it onto the neighbour
// (cyclicACMIPolyPatch.C:107-110); a scale written on the slave half is discarded with a warning
// (line 838). Applied after the boundary file is read, and exposed here so a test can drive the same
// rule on an in-memory mesh instead of a copy of it.
inline void propagateACMIScale(std::vector<PatchInfo>& patches)
{
    for (std::size_t i = 0; i < patches.size(); ++i)
    {
        if (patches[i].type != "cyclicACMI") continue;
        std::size_t nbr = patches.size();
        for (std::size_t j = 0; j < patches.size(); ++j)
            if (patches[j].name == patches[i].neighbourPatch) { nbr = j; break; }
        if (nbr >= patches.size()) continue;             // dangling neighbourPatch: reported elsewhere
        if (i < nbr)                                     // this half is the owner
        {
            if (!patches[i].acmiScale.empty()) patches[nbr].acmiScale = patches[i].acmiScale;
            // a coded scale is cloned onto the neighbour PATCH (tgtScalePtr_ = srcScalePtr_.clone(
            // neighbPatch())), so the neighbour evaluates the same code on its own faces
            if (patches[i].acmiScaleCoded)
            {
                patches[nbr].acmiScaleCoded = true;
                patches[nbr].acmiScaleCodedSpec = patches[i].acmiScaleCodedSpec;
                patches[nbr].acmiScale = Function1();
            }
        }
        else if (!patches[i].acmiScale.empty() && !patches[nbr].acmiScale.empty())
        {
            patches[i].acmiScale = patches[nbr].acmiScale;   // both carry one: the owner's wins
        }
    }
}

class PrimitiveMesh
{
public:
    // Read points/faces/owner/neighbour/boundary from a constant/polyMesh directory.
    void read(const std::string& polyMeshDir);

    // Construct in memory (e.g. a decomposePar local mesh). Faces must be ordered internal-first
    // (upper-triangular by owner) then boundary faces grouped by patch, exactly as read().
    void assign(
        std::vector<vector> points,
        std::vector<label> faceVerts,
        std::vector<label> faceOffsets,
        std::vector<label> owner,
        std::vector<label> neighbour,
        std::vector<PatchInfo> patches,
        label nCells)
    {
        points_ = std::move(points);
        faceVerts_ = std::move(faceVerts);
        faceOffsets_ = std::move(faceOffsets);
        owner_ = std::move(owner);
        neighbour_ = std::move(neighbour);
        patches_ = std::move(patches);
        nCells_ = nCells;
    }

    label nPoints()        const { return static_cast<label>(points_.size()); }
    label nFaces()         const { return static_cast<label>(owner_.size()); }
    label nInternalFaces() const { return static_cast<label>(neighbour_.size()); }
    label nCells()         const { return nCells_; }

    const std::vector<vector>&    points()      const { return points_; }

    // OF polyMesh::movePoints(newPoints) -- replace the point positions, keeping the topology
    // (faces, owner, neighbour, patches) untouched. The CALLER must then rebuild FvGeometry, exactly
    // as OF's fvMesh::movePoints follows polyMesh::movePoints with updateGeomNotOldVol() and
    // surfaceInterpolation::clearOut(): every derived quantity -- Sf, magSf, V, C, Cf, weights,
    // deltaCoeffs -- is a function of the points and is stale the moment they move.
    //
    // `points0` (the ORIGINAL positions) must be kept by the caller: OF's solidBodyMotionSolver
    // transforms points0 by an absolute function of t (points0MotionSolver), never the current points.
    // Transforming the current points each step would compound round-off and drift.
    void movePoints(std::vector<vector> newPoints)
    {
        if (newPoints.size() != points_.size())
            throw std::runtime_error("brae: movePoints with a different point count -- topology change "
                                     "is not supported (OF polyMesh::movePoints keeps topology fixed).");
        points_ = std::move(newPoints);
    }
    const std::vector<label>&     faceVerts()   const { return faceVerts_; }
    const std::vector<label>&     faceOffsets() const { return faceOffsets_; }
    const std::vector<label>&     owner()       const { return owner_; }
    const std::vector<label>&     neighbour()   const { return neighbour_; }
    const std::vector<PatchInfo>& patches()     const { return patches_; }

    label faceSize(label f)  const { return faceOffsets_[f + 1] - faceOffsets_[f]; }
    label faceVert(label f, label k) const { return faceVerts_[faceOffsets_[f] + k]; }

    // Public so a test can exercise the boundary parse on its own: the patch dictionary has grown
    // enough shapes (sub-dictionaries, wordLists, ACMI keys) to be worth testing without having to
    // synthesise a whole consistent polyMesh around it.
    void readBoundary(const std::string& dir);

private:
    void readPoints(const std::string& dir);
    void readFaces(const std::string& dir);
    void readOwner(const std::string& dir);
    void readNeighbour(const std::string& dir);
    // Binary mesh cache (BRAE_MESH_CACHE): the ASCII polyMesh parse is the #1 startup cost (millions of string tokens).
    // On a warm run we reload the parsed topology from a binary blob (raw fread of the POD arrays), like OpenFOAM
    // reusing its decomposePar processor* dirs. Auto-invalidated when the polyMesh/owner file is newer than the cache.
    bool loadBinary(const std::string& path);
    void writeBinary(const std::string& path) const;

    std::vector<vector>    points_;
    std::vector<label>     faceVerts_;    // CSR values
    std::vector<label>     faceOffsets_;  // CSR offsets, size nFaces+1
    std::vector<label>     owner_;
    std::vector<label>     neighbour_;
    std::vector<PatchInfo> patches_;
    label                  nCells_ = 0;
};

} // namespace brae
