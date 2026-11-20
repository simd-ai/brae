#pragma once
// OpenFOAM's compressible RAS turbulence model as a compressible solver constructs it -- compressible::turbulenceModel::
// New(rho, U, phi, thermo) (rhoCentralFoam's createFields.H) -- the case half: what the model reads at construction.
//
// provenance (OpenFOAM v2412):
//   src/TurbulenceModels/turbulenceModels/turbulenceModel.C, RASModel.C:61-99 (the RAS dict: model, turbulence,
//       kMin/omegaMin, the <model>Coeffs sub-dictionary read by optionalSubDict)
//   src/TurbulenceModels/compressible/EddyDiffusivity/EddyDiffusivity.C:34-40, :70-95 (alphat MUST_READ, Prt from the
//       coeffDict, default 1.0)
//   src/TurbulenceModels/turbulenceModels/Base/kOmegaSST/kOmegaSSTBase.C (the constructor: k_, omega_ MUST_READ; nut_
//       from eddyViscosity, MUST_READ; F1's wall distance from wallDist::New(mesh), meshWave)
//   the patch fields' own classes: kqRWallFunction (zeroGradient), omegaWallFunction, nutkWallFunction,
//       compressible::alphatWallFunction (its own Prt), calculated, fixedValue, inletOutlet, the constraint types
//   src/finiteVolume/fvMesh/wallDist/nearWallDist (the wall functions' y), patchDistMethods meshWave (the cells' y)
//
// The model itself -- kOmegaSSTBase -- is brae's shared closure (kOmegaSST.cuh), which OpenFOAM's every compressible
// solver runs too; compressibleRAS.cuh is the layer OpenFOAM's CompressibleTurbulenceModel and EddyDiffusivity are.
// This file builds what both read, from the case, and refuses what it does not port by name.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "komega_sst_coeffs.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {
namespace compressibleRAS {

// one equation's div(phi,<field>) as the closure takes it
struct DivScheme
{
    bool   bounded = false;
    bool   limitedLinear = false;
    scalar limiterCoeff = 1.0;
    bool   linearUpwind = false;
};

struct SolveCtl
{
    scalar tol = 1e-10, relTol = 0;
    int    maxIter = 1000, minIter = 0, nSweeps = 1;
    bool   symmetric = false;   // symGaussSeidel; GaussSeidel otherwise
};

struct Case
{
    bool   on = false;                 // simulationType RAS
    std::string model;                 // kOmegaSST
    KOmegaSSTCoeffs co;
    scalar Prt = 1.0;                  // EddyDiffusivity, from kOmegaSSTCoeffs
    // the fields as read, with their patch classes (fv_patch_field.cuh), and their patch values flat
    GeometricField<scalar> k, omega, nut, alphat;
    std::vector<scalar> kB, omegaB, nutB, alphatB;
    // wall distance: meshWave for the cells (F1, F2), nearWallDist for the wall-function faces
    std::vector<scalar> yCell;
    // per boundary face (flat, patch order, no coupled patch)
    std::vector<label>  wfMask, f1One, nutCalc, nutKind, alphatMask;
    // nut's and alphat's constraint patches (symmetryPlane, symmetry, wedge, slip, empty): their correctBoundary-
    // Conditions() after correctNut sets the patch to the face cell's value (a scalar's transform is the identity)
    std::vector<label>  constraintFace, bCell;
    std::vector<scalar> yBnd, nutCmu25, nutKappa, nutE, nutYplLam, alphatPrt;
    std::vector<char>   wfPatch;       // per patch: omega's patch is a turbulence wall function
    // fvSchemes, fvSolution
    DivScheme kDiv, omegaDiv;
    bool correctedLaplacian = true;
    SolveCtl kSolve, omegaSolve;
};

// Reads constant/turbulenceProperties; on `simulationType RAS` builds the model's case (fields from `startDir`), refusing
// what the port does not run. `U` is the velocity field as the solver built it (its wall patches give the wall
// velocity). `refuse` throws with a message naming the input.
Case readCase(const std::string& caseDir, const std::string& startDir, const PrimitiveMesh& m, const FvGeometry& g,
              const std::vector<FvPatch>& patches, const GeometricField<vector>& U, bool lts);

} // namespace compressibleRAS
} // namespace brae
