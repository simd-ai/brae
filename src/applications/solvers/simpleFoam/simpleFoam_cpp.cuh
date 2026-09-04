#pragma once
// _cpp REFERENCE DRIVER -- one simpleFoam SIMPLE iteration, composed of the ported components.
//
// provenance:
//   openfoam: applications/solvers/incompressible/simpleFoam/simpleFoam.C:79-97 (the time loop body)
//   brae:     src/applications/solvers/simpleFoam/simpleFoam_cpp.cu
//   tests:    tests/test_simple_step_cpp.cu   (end-to-end vs OpenFOAM's dumpSimpleStep, step.dat)
//
// THE DRIVER OWNS NO NUMERICS. That is the whole point of the rebuild: everything below is a call into a
// shared component that has its own OpenFOAM provenance and its own test.
//
//     assembleUEqn          UEqn_cpp            (fvm::div + divDevReff + relax)
//     addPressureGradient   UEqn_cpp            (-fvc::grad(p))
//     solveVector           OpenFOAM/matrices   (the momentum solve)
//     pressurePredictor     pEqn_cpp            (rAU, HbyA, phiHbyA, adjustPhi)
//     assemblePEqn          pEqn_cpp            (laplacian == div(phiHbyA), setReference)
//     gamg                  OpenFOAM/matrices   (the pressure solve)
//     correctFlux           pEqn_cpp            (phi = phiHbyA - pEqn.flux())
//     relaxField            pEqn_cpp            (p.relax())
//     correctVelocity       pEqn_cpp            (U = HbyA - rAU*grad(p))
//     correctNonOrthogonal  simpleControl_cpp   (the corrector loop)
//     kepsilon::correct     TurbulenceModels    (turbulence->correct(), after the pressure corrector)
//
// Compare with what it replaces: src/applications/solvers/simpleFoam/device_simple_foam.cu, 3578 lines,
// included by pimpleFoam, rhoSimpleFoam and five headers in solvers/common.
//
// ORDER IS OPENFOAM'S. Two orderings matter and neither is obvious from the equations:
//   * the momentum predictor solves a COPY of UEqn with -grad(p) added, because pEqn.H then needs the
//     original UEqn for A() and H(). Adding grad(p) to UEqn itself changes rAU and HbyA.
//   * p is relaxed AFTER the flux correction and BEFORE the velocity correction, so phi is built from the
//     unrelaxed pressure and U from the relaxed one.
#include "kOmegaSSTLM_cpp.cuh"
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "UEqn_cpp.cuh"        // DivScheme: the div(phi,U) scheme, shared with the CUDA driver
#include "kepsilon_coeffs.cuh"
#include "komega_sst_coeffs.cuh"
#include "SpalartAllmaras_cpp.cuh"
#include "geometric_field.cuh"
#include <map>
#include <string>
#include <vector>

namespace brae {
namespace cpu {

// Turbulence coupling. simpleFoam.C:93-94 calls laminarTransport.correct() then turbulence->correct()
// AFTER the pressure corrector, so the momentum equation of iteration n uses nut from iteration n-1 --
// a LAGGED coupling. Running correct() before UEqn instead is a different algorithm that still converges
// to something plausible, which is exactly why the order is written down here rather than assumed.
//
// nuEff is DERIVED rather than supplied: nuEff = nu + nut, with boundary values from nut's own boundary
// field (nut_wall on a wall function), never the owner cell. That boundary rule has bitten brae before;
// see effectiveFaceViscosity.
struct TurbulenceState
{
    GeometricField<scalar>* k = nullptr;
    GeometricField<scalar>* epsilon = nullptr;
    GeometricField<scalar>* nut = nullptr;
    KEpsilonCoeffs coeffs{};

    // kOmegaSST rides the SAME slots: `epsilon` above holds omega. It additionally needs the CELL wall
    // distance -- F1/F2 blend on it at every cell, which is a different quantity from the near-wall face
    // distance the wall functions use -- and its own coefficients.
    bool sst = false;
    KOmegaSSTCoeffs sstCoeffs{};
    std::vector<scalar> y;

    // div(phi,k) / div(phi,omega). `bounded Gauss limitedLinear 1` in the SST tutorials, plain
    // `Gauss upwind` in pitzDaily's kEpsilon -- a different matrix, not a different tolerance.
    bool   boundedTurb = false;
    bool   limitedLinearTurb = false;
    bool   linearUpwindTurb = false;   // `Gauss linearUpwind grad(<var>)` on the turbulence scalar
    scalar turbLimiterCoeff = 1.0;

    // kOmegaSSTLM is kOmegaSST plus two transported scalars and three overrides of it. gammaIntEff is
    // NOT read from a file: OpenFOAM constructs it as zero and the first outer iteration runs with no
    // turbulent production at all, so seeding it here would change where transition lands.
    bool lm = false;
    GeometricField<scalar>* ReThetat = nullptr;
    GeometricField<scalar>* gammaInt = nullptr;
    std::vector<scalar>     gammaIntEff;
    kOmegaSSTLM::Coeffs     lmCoeffs{};
    kOmegaSSTLM::Residuals* lmRes = nullptr;

    // SpalartAllmaras transports ONE scalar, nuTilda, and holds it in the `k` slot; `epsilon` is unused.
    // Its own coefficients, and the same cell wall distance the SST needs.
    bool     sa = false;
    SA::Coeffs saCoeffs{};
    SA::Residuals* saRes = nullptr;   // optional: the nuTilda INITIAL residual, OF normalisation

    scalar relaxK = 0.7, relaxEpsilon = 0.7;
    scalar tol = 1e-10, relTol = 0.0;
    int    maxIter = 2000;
};

struct StepInput
{
    // Laminar viscosity. With `turb` set, nuEff is built from nu + nut each iteration and the two arrays
    // below are ignored; without it they are used as given (the laminar path).
    scalar nu = 1e-5;
    TurbulenceState* turb = nullptr;

    std::vector<scalar>              nuEff;      // cells    (laminar path only)
    std::vector<std::vector<scalar>> nuEffBnd;   // [patch][face]
    scalar relaxU = 0.7;                         // relaxationFactors/equations U
    scalar relaxP = 0.3;                         // relaxationFactors/fields p
    scalar tolU = 1e-10, relTolU = 0.0;
    scalar tolP = 1e-10, relTolP = 0.0;
    // PER FIELD -- lduMatrix::solver::readControls (lduMatrixSolver.C:190-208) reads minIter and maxIter
    // from that field's OWN entry (defaults 0 and lduMatrix::defaultMaxIter = 1000). See the device
    // twin's note: one shared maxIter, read from `p`, ran T3A's momentum solve at 1000 against
    // OpenFOAM's 10.
    int    maxIterU = 1000, minIterU = 0;
    int    maxIterP = 1000, minIterP = 0;
    bool   correctedLaplacian = false;           // `corrected` laplacianSchemes
    scalar snGradLimitCoeff = 0.0;               // `limited <k> corrected` (OF limitedSnGrad)
    bool   bounded = false;                      // div(phi,U) `bounded`
    bool   linearUpwind = false;                 // div(phi,U) `linearUpwind`
    scalar gradULimitK = 0.0;                    // `grad(U) cellLimited Gauss linear <k>` (0 = off)
    DivScheme scheme = DivScheme::upwind;
    scalar    schemeCoeff = 1.0;
    // MRF zones, already resolved against the mesh. hasMRF WITHOUT these is still a refusal.
    const std::vector<MRF::Zone>* mrf = nullptr;
    bool   hasMRF = false;                       // refused downstream when `mrf` is null
    bool   hasFvOptions = false;                 // refused downstream
};

// field name -> initial residual of its first solve this iteration (what simpleControl checks).
using Residuals = std::map<std::string, scalar>;

// One SIMPLE iteration, in place on f.p / f.U / f.phi.
Residuals simpleStep(
    SimpleFields&               f,
    SimpleControl&              ctl,
    const StepInput&            in,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

} // namespace cpu
} // namespace brae
