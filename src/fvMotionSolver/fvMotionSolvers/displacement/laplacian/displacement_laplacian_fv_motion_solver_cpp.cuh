#pragma once
// OpenFOAM's displacementLaplacian motion solver with the inverseDistance diffusivity: the point
// displacement a mesh takes from its point boundary conditions, by way of a Laplace equation for the
// displacement of the cell centres. The host reference.
//
// provenance:
//   openfoam: src/fvMotionSolver/fvMotionSolvers/displacement/laplacian/
//                 displacementLaplacianFvMotionSolver.C:45-120 (constructor), :199-282 (curPoints),
//                 :285-318 (solve)
//             src/dynamicMesh/motionSolvers/motionSolver/motionSolver.C:200-204 (newPoints)
//             src/dynamicMesh/motionSolvers/displacement/displacement/displacementMotionSolver.C
//             src/fvMotionSolver/fvMotionSolvers/fvMotionSolver/fvMotionSolverTemplates.C
//                 (cellMotionBoundaryTypes)
//             src/fvMotionSolver/fvPatchFields/derived/cellMotion/cellMotionFvPatchField.C (updateCoeffs)
//             src/fvMotionSolver/motionDiffusivity/inverseDistance/inverseDistanceDiffusivity.C:70-86
//             src/finiteVolume/fvMesh/wallDist/wallDist/wallDist.C, patchDistMethods/meshWave/
//                 meshWavePatchDistMethod.C:78-110 (y and its boundary values)
//             src/fvMotionSolver/motionInterpolation/motionInterpolation/motionInterpolation.C:104-129
//             src/finiteVolume/interpolation/volPointInterpolation/volPointInterpolate.C:346-368
//                 (interpolate), pointConstraintsTemplates.C:130-156 (constrain)
//             src/OpenFOAM/fields/pointPatchFields/basic/value/valuePointPatchField.C (updateCoeffs and
//                 evaluate both write the patch's values into the point field)
//             src/finiteVolume/fvMatrices/fvMatrix/fvMatrixSolve.C (solveSegregated)
//             src/finiteVolume/finiteVolume/laplacianSchemes/gaussLaplacianScheme/gaussLaplacianSchemes.C
//
// ONE newPoints(), IN OpenFOAM's ORDER, all of it on the mesh as it stands BEFORE the move:
//   1. the diffusivity: 1/interpolate(y), y the meshWave distance to the named patches. y's boundary
//      value is the wave's own on those patches and the next cell's on every other (zeroGradient);
//   2. the point boundary conditions' updateCoeffs, in patch order, each writing its values straight
//      into the point field -- so a point two patches share takes the later patch's value;
//   3. the cell boundary conditions' updateCoeffs: on every patch whose point condition fixes a value,
//      cellMotion, the face average of THAT point field;
//   4. laplacian(diffusivity, cellDisplacement) with the case's `Gauss linear corrected`, solved one
//      component at a time from the last step's solution, the empty direction skipped;
//   5. the cell displacement to the points: cells to interior points, boundary faces to patch points,
//      then the point boundary conditions evaluated again in patch order;
//   6. points0 + pointDisplacement, then the 2-D correction.
//
// NOT PORTED, refused by name where the dictionaries are read: every diffusivity but inverseDistance;
// `interpolation`, `frozenPointsZone`, a cellZone or cellSet, a pointLocation field, a cellDisplacement
// field to start from; point conditions other than fixedValue, zeroGradient, empty and waveMaker; a
// solver for cellDisplacement other than GAMG; and any scheme on the equation's path other than Gauss
// linear corrected, Gauss linear and linear.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "gamg_solver_cpp.cuh"
#include "primitive_mesh.cuh"
#include "two_d_point_corrector_cpp.cuh"
#include "vol_point_interpolation_cpp.cuh"
#include "wave_maker_point_patch_vector_field_cpp.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {

// what one solve of the displacement equation did, per component x, y, z. A component the mesh does not
// solve -- the empty direction of a 2-D mesh -- is not solved.
struct DisplacementSolveRecord
{
    bool solved[3] = {false, false, false};
    SolverPerformance perf[3];
};

class DisplacementLaplacianFvMotionSolver
{
public:
    // the constructor, from the dictionaries alone: `coeffs` is displacementLaplacianCoeffs, and the
    // rest is the start time's pointDisplacement, fvSolution, fvSchemes and constant/g
    static std::unique_ptr<DisplacementLaplacianFvMotionSolver> New(
        const FoamDict& coeffs,
        const std::string& caseDir,
        const std::string& startDir);

    // the mesh as it stands: points0, each point patch's points, the 2-D corrector
    void attach(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);

    // motionSolver::newPoints(): solve() at `time`, then curPoints(). finalIteration selects the
    // cellDisplacementFinal entry, as fvMatrix::solverDict() does under PIMPLE's last iteration.
    std::vector<vector> newPoints(
        scalar time,
        bool finalIteration,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches,
        GamgAgglomerationCache& agglomeration);

    const std::vector<vector>& pointDisplacement() const
    {
        return pointDisplacement_;
    }
    const std::vector<vector>& cellDisplacement() const
    {
        return cellDisplacement_;
    }
    // per patch, per face; empty patches carry nothing
    const std::vector<std::vector<vector>>& cellDisplacementBoundary() const
    {
        return cellDisplacementBoundary_;
    }
    // the wall distance the last solve's diffusivity came from
    const std::vector<scalar>& y() const
    {
        return y_;
    }
    const DisplacementSolveRecord& lastSolve() const
    {
        return lastSolve_;
    }

private:
    DisplacementLaplacianFvMotionSolver() = default;

    enum class PointPatchType
    {
        fixedValue,
        zeroGradient,
        empty,
        waveMaker
    };

    struct PointPatch
    {
        std::string name;
        PointPatchType type = PointPatchType::zeroGradient;
        // meshPoints(), and the fixed values per point where the type fixes them
        std::vector<label> meshPoints;
        std::vector<vector> value;
        std::unique_ptr<WaveMakerPointPatchVectorField> waveMaker;
        // the dictionary entry, kept until attach() has the patch's points
        const FoamDict* dict = nullptr;
        // cellMotionBoundaryTypes: fixedValue and what derives from it give the cell field cellMotion
        bool fixesValue() const
        {
            return type == PointPatchType::fixedValue || type == PointPatchType::waveMaker;
        }
    };

    void diffusivityCorrect(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);
    void solve(
        scalar time,
        bool finalIteration,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches,
        GamgAgglomerationCache& agglomeration);
    std::vector<vector> curPoints(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);

    std::string caseDir_;
    FoamDict pointDisplacementDict_;
    std::vector<std::string> diffusivityPatches_;
    GamgControls controls_;
    GamgControls controlsFinal_;
    vector g_{0, 0, 0};
    scalar startTime_ = 0;

    std::vector<vector> points0_;
    std::vector<vector> pointDisplacement_;
    std::vector<PointPatch> pointPatches_;
    std::vector<vector> cellDisplacement_;
    std::vector<std::vector<vector>> cellDisplacementBoundary_;
    std::vector<std::vector<label>> pointCells_;
    std::vector<label> diffusivityPatchIDs_;
    std::vector<scalar> y_;
    SurfaceScalarField faceDiffusivity_;
    TwoDPointCorrector twoDCorrector_;
    VolPointInterpolation interpolation_;
    DisplacementSolveRecord lastSolve_;
    bool attached_ = false;
};

} // namespace brae
