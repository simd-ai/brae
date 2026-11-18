#pragma once
// OpenFOAM's velocityComponentLaplacian motion solver: one component of the point velocity from a Laplace equation for
// the cell-centre velocity of that component, the points moved by deltaT times it. The host reference.
//
// provenance:
//   openfoam: src/fvMotionSolver/fvMotionSolvers/componentVelocity/laplacian/
//                 velocityComponentLaplacianFvMotionSolver.C (constructor: cellMotionU<c> READ_IF_PRESENT with the
//                 boundary types cellMotionBoundaryTypes gives; solve; curPoints)
//             src/dynamicMesh/motionSolvers/componentVelocity/componentVelocityMotionSolver.C (`component`, the
//                 MUST_READ point field pointMotionU<c>)
//             src/dynamicMesh/motionSolvers/motionSolver/motionSolver.C (newPoints = solve(); curPoints();
//                 twoDCorrectPoints)
//             src/fvMotionSolver/fvMotionSolvers/fvMotionSolver/fvMotionSolverTemplates.C (cellMotionBoundaryTypes:
//                 a fixedValue-derived point patch gives cellMotion, any other keeps its type name)
//             src/fvMotionSolver/fvPatchFields/derived/cellMotion/cellMotionFvPatchField.C (updateCoeffs: the face
//                 average of the point field, face::average)
//             src/fvMotionSolver/motionDiffusivity/uniform, directional/directionalDiffusivity.C (correct:
//                 n & cmptMultiply(D, n), n = Sf/magSf)
//             src/finiteVolume/fields/fvPatchFields/basic/transform/transformFvPatchFields.C:42-58 (a scalar on a
//                 slip, symmetry or wedge patch: valueInternalCoeffs 1, gradientInternalCoeffs 0 -- zeroGradient)
//             src/finiteVolume/interpolation/volPointInterpolation/volPointInterpolate.C (interpolate:
//                 internal, boundary, then pointConstraints::constrain -> correctBoundaryConditions)
//             src/OpenFOAM/fields/pointPatchFields/derived/uniformFixedValue (updateCoeffs: Function1(t))
//             src/meshTools/twoDPointCorrector (the wedge: snapToWedge)
//   tests:    tests/rho_central_moving_vs_openfoam.sh (movingCone: the points, meshPhi and cellMotionUx against
//             OpenFOAM's at every write).
//
// ONE newPoints(), IN OpenFOAM's ORDER, on the mesh as it stands BEFORE the move (velocity solvers are incremental:
// the points move from where they are, not from points0):
//   1. movePoints(points): the interpolation's weights for the mesh as it stands;
//   2. the diffusivity: 1 (uniform), or n & cmptMultiply(D, n) on every face (directional);
//   3. pointMotionU<c>'s boundary updateCoeffs, patch by patch: uniformFixedValue and fixedValue write their values
//      into the point field, so a point two patches share takes the later patch's;
//   4. fvm::laplacian(1*diffusivity, cellMotionU<c>) with the case's `laplacian(diffusivity,cellMotionU)` (its
//      construction runs cellMotion's updateCoeffs: the face average of the point field over the current points),
//      solved with the fvSolution entry for cellMotionU<c> (PCG, DIC), then correctBoundaryConditions;
//   5. curPoints: volPointInterpolation (cells, then the boundary faces of real patches), the fixed point patches'
//      values written back in patch order (constrain: a scalar's slip and wedge evaluate to itself), points[c] +=
//      deltaT*pointMotionU<c>, then twoDCorrectPoints.
//
// NOT PORTED, refused by name where the dictionaries are read: a diffusivity other than uniform and directional;
// `interpolation`; a cellZone or cellSet; point conditions other than fixedValue, uniformFixedValue with a constant
// Function1, slip, zeroGradient, wedge, symmetryPlane, symmetry and empty; a solver for cellMotionU<c> other than PCG
// with DIC; a laplacian scheme other than Gauss linear uncorrected, Gauss linear orthogonal and Gauss linear limited 0;
// coupled patches.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include "pcg.cuh"
#include "primitive_mesh.cuh"
#include "two_d_point_corrector_cpp.cuh"
#include "vol_point_interpolation_cpp.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {

class VelocityComponentLaplacianFvMotionSolver
{
public:
    // the constructor, from the dictionaries alone: `coeffs` is velocityComponentLaplacianCoeffs (or the top level),
    // the rest the start time's pointMotionU<c> (MUST_READ) and cellMotionU<c> (READ_IF_PRESENT), fvSolution and
    // fvSchemes
    static std::unique_ptr<VelocityComponentLaplacianFvMotionSolver> New(
        const FoamDict& coeffs,
        const std::string& caseDir,
        const std::string& startDir);

    // the mesh as it stands: each point patch's points, the cell field's patches, the 2-D corrector
    void attach(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);

    // motionSolver::newPoints(): solve() at `time`, then curPoints() with `deltaT`
    std::vector<vector> newPoints(
        scalar time,
        scalar deltaT,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);

    // the component (0, 1, 2) and its name (x, y, z)
    int cmpt() const
    {
        return cmpt_;
    }
    const std::string& cmptName() const
    {
        return cmptName_;
    }
    // the last cellMotionU<c> solve, for the solver's log line (DICPCG: Solving for cellMotionU<c>, ...)
    const SolverPerformance& lastSolve() const
    {
        return lastSolve_;
    }

    enum class PointPatchType
    {
        fixedValue,
        uniformFixedValue,
        zeroGradient,
        slip,
        symmetryPlane,
        symmetry,
        wedge,
        empty
    };
    struct PointPatch
    {
        std::string name;
        std::string typeName;   // as written back
        PointPatchType type = PointPatchType::zeroGradient;
        std::vector<label> meshPoints;
        scalar value = 0;       // fixedValue: its uniform value; uniformFixedValue: the constant Function1's
        bool fixesValue() const
        {
            return type == PointPatchType::fixedValue || type == PointPatchType::uniformFixedValue;
        }
    };
    // the fields as the writer echoes them (AUTO_WRITE): pointMotionU<c> and cellMotionU<c> with their patches
    const std::vector<PointPatch>& pointPatches() const
    {
        return pointPatches_;
    }
    const std::vector<scalar>& pointMotionU() const
    {
        return pointMotionU_;
    }
    const std::vector<scalar>& cellMotionU() const
    {
        return cellMotionU_;
    }
    // per patch, per face (empty patches carry nothing); the type each patch has on the cell field
    const std::vector<std::vector<scalar>>& cellMotionUBoundary() const
    {
        return cellMotionUBoundary_;
    }
    const std::vector<std::string>& cellMotionUTypes() const
    {
        return cellMotionUTypes_;
    }

private:
    VelocityComponentLaplacianFvMotionSolver() = default;

    void diffusivityCorrect(
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);
    void solve(
        scalar time,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);
    std::vector<vector> curPoints(
        scalar deltaT,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);

    std::string caseDir_;
    std::string startDir_;
    int cmpt_ = 0;
    std::string cmptName_;
    bool directional_ = false;
    vector diffusivityVector_{1, 1, 1};
    // fvSolution's cellMotionU<c>: PCG with DIC
    scalar tolerance_ = 1e-6;
    scalar relTol_ = 0;
    int maxIter_ = 1000;
    int minIter_ = 0;
    // fvSchemes' laplacian(diffusivity,cellMotionU): every accepted form takes nonOrthDeltaCoeffs with no correction
    // (uncorrected, limited 0) or deltaCoeffs (orthogonal)
    bool orthogonal_ = false;

    std::vector<PointPatch> pointPatches_;
    std::vector<scalar> pointMotionU_;
    // the start time's cellMotionU<c> when it exists (a restart): the solve's first initial guess
    std::vector<scalar> cellMotionU_;
    std::vector<std::vector<scalar>> cellMotionUBoundary_;
    std::vector<std::string> cellMotionUTypes_;
    bool cellMotionURead_ = false;
    std::vector<std::vector<label>> pointCells_;
    SurfaceScalarField faceDiffusivity_;
    TwoDPointCorrector twoDCorrector_;
    VolPointInterpolation interpolation_;
    SolverPerformance lastSolve_;
    bool attached_ = false;
};

} // namespace brae
