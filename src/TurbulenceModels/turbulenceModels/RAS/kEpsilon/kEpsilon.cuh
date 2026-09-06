#pragma once
// CUDA implementation of the compressible standard k-epsilon closure -- the device twin of kEpsilon_cpp.
//
// provenance:
//   openfoam:  src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C:225-292
//     wall:    .../wallFunctions/epsilonWallFunctions/epsilonWallFunction/
//              epsilonWallFunctionFvPatchScalarField.C  (the patch write, G0, the cell overwrite, and
//              manipulateMatrix -> setValues)
//     nut:     .../nutkWallFunctions/nutkWallFunction/nutkWallFunctionFvPatchScalarField.C
//     alphat:  src/TurbulenceModels/compressible/EddyDiffusivity/EddyDiffusivity.C:33-40
//     bound:   src/finiteVolume/cfdTools/general/bound/bound.C:38-62
//   reference: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu
//              (gated against tools/dumpKEpsilon by tests/rho_kepsilon_vs_openfoam.sh)
//   cuda:      src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu
//   tests:     tests/test_rho_kepsilon_cuda.cu
//
// TWO FLUXES, NOT ONE, and this is the trap the incompressible lineage cannot show. fvm::div takes the
// MASS flux; divU is a DILATATION and takes the VOLUMETRIC one, phi/interpolate(rho)
// (compressibleTurbulenceModel.C). `bounded` subtracts the divergence of the EQUATION's own flux, which
// is the mass one. Where rho is uniform all three coincide and nothing discriminates them.
//
// THE EPSILON PRODUCTION IS C1*rho*GbyNu*Cmu*k, NOT C1*G/k. The two are equal only where
// nut == Cmu k^2/eps, which is exactly what is not true in a wall cell.
//
// nu IS A FIELD, NOT A SCALAR. rhoSimpleFoam_cpp.cu passes /*nu=*/0.0 to the reference precisely because
// the compressible lineage has no case-constant viscosity, and every fallback that read that scalar was
// a divide-by-zero. This module therefore has NO scalar-nu path: nuCell and nuBndFace are required, and
// so is nuWallFace, because the wall functions are written in terms of nu AT THE WALL FACE and not the
// adjacent cell's.
//
// THE DIFFUSIVITY'S BOUNDARY IS nut's OWN BOUNDARY, not the adjacent cell's. DkEff()/DepsilonEff() are
// volScalarFields, so fvm::laplacian takes their PATCH values. At an inlet OpenFOAM's nut_b is
// Cmu*k_b^2/eps_b, which is far from the cell's -- so this module takes the FACE variant of the boundary
// laplacian coefficients, never the cell one.
//
// WHAT THE CALLER MUST HAVE DONE BEFORE ENTERING (this module refreshes none of it -- the same contract
// rhoUEqn.cuh states):
//   1. buildDeviceWallData with the 5-ARG overload, wfPatch[pi] = epsilon.boundary[pi]->
//      isTurbulenceWallFunction(). The 4-arg overload falls back to the patch TYPE alone, and a `wall`
//      carrying a plain fixedValue epsilon then gets silently constrained.
//   2. rho, and nu = mu(T)/rho on cells AND boundary faces, from THIS iteration's temperature.
//   3. phiByRho = phi / interpolate(rho), on internal AND boundary faces.
//
// ORDER IS LOAD-BEARING, in three places:
//   - G is captured BEFORE the wall replacement, because that is where OpenFOAM writes it.
//   - relax() -> fvOptions.constrain() -> setValues(wall), in that order (kEpsilon.C:265-267). setValues
//     transfers source_[nei] -= coeff*value and then ZEROES that coeff, so only the FIRST setValues
//     touching a cell moves anything into its neighbours.
//   - the k equation has NO wall setValues and NO boundaryManipulate (kEpsilon.C:286-288); kqRWallFunction
//     is zeroGradient. Constraining k in wall cells the way epsilon is constrained is a different equation.
//
// REFUSED, not ignored -- a comment is not a refusal, and every flag below throws and names itself:
//   realizable / RNG (the reference implements neither, and the coefficient set has already been
//   substituted by the time it arrives), coupled patches, unported fvOptions, a div scheme other than
//   upwind, a turbulence wall function on a non-`wall` patch, and a case that bounds one of the two
//   convection terms but not the other.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"    // DeviceWallData, deviceGradU, deviceGByNuFromGradU, deviceWallEpsG0
#include "device_dilu.cuh"        // the case's DILU preconditioner for the k and epsilon solves
#include "kepsilon_coeffs.cuh"
#include "pEqn.cuh"               // PressureMatrix -- the assembled scalar object, shared not redefined
#include <string>

namespace brae {
namespace gpu {
namespace kEpsilonRAS {

struct KEpsilonInput
{
    // --- the two fluxes. See the header: they are not interchangeable. ---
    const DeviceBuffer<scalar>* phiInt      = nullptr;   // MASS flux, kg/s -- convection and `bounded`
    const DeviceBuffer<scalar>* phiBnd      = nullptr;
    const DeviceBuffer<scalar>* phiByRhoInt = nullptr;   // VOLUMETRIC flux -- divU ONLY
    const DeviceBuffer<scalar>* phiByRhoBnd = nullptr;

    // --- the compressible instantiation: alpha = 1, rho = the solver's relaxed density ---
    const DeviceBuffer<scalar>* rhoCell    = nullptr;
    const DeviceBuffer<scalar>* rhoBndFace = nullptr;
    const DeviceBuffer<scalar>* nuCell     = nullptr;    // mu(T)/rho per cell.        REQUIRED.
    const DeviceBuffer<scalar>* nuBndFace  = nullptr;    // mu_b/rho_b per bnd face.   REQUIRED.
    const DeviceBuffer<scalar>* nuWallFace = nullptr;    // the same, in WALL-face order
                                                         // (deviceGatherWallNu).      REQUIRED.

    // nut's boundary as it stands ENTERING correct(): what DkEff(patchi)/DepsilonEff(patchi) are built
    // from. NOT the owner cell's nut -- see the header.
    const DeviceBuffer<scalar>* nutBndFace = nullptr;
    // The same, gathered into WALL-face order, for the wall functions' G0: OpenFOAM's
    // epsilonWallFunction reads nutw[facei] from the nut patch field as stored, not a fresh
    // nutkWallFunction of the current k and nu_w -- the two differ wherever rho_b has moved since the
    // value was stored (5.4e-05 on rhoKE at iteration 1, worth 1e-06 in k). Null = recompute (the
    // old arithmetic), so a caller that cannot gather still runs; the rho hook gathers it.
    const DeviceBuffer<scalar>* nutWallFace = nullptr;

    // Which BOUNDARY FACES carry a turbulence wall function, and the near-wall distance on each. The
    // predicate is per FACE and not per cell: a cell can touch a wall patch and an inlet at once, and
    // correctNut must give those two faces different values. DeviceWallData answers the CELL question,
    // which is the one the epsilon constraint needs; this is the other one.
    const DeviceBuffer<label>*  wfBndMask    = nullptr;
    const DeviceBuffer<scalar>* wallYBndFace = nullptr;   // nearWallDist y, per boundary face

    // --- velocity, for the production term and the wall functions ---
    const DeviceBuffer<scalar>* Ux = nullptr;
    const DeviceBuffer<scalar>* Uy = nullptr;
    const DeviceBuffer<scalar>* Uz = nullptr;

    // --- schemes ---
    // Only `Gauss upwind` (with or without `bounded`) is ported, because that is all the reference
    // assembles: its two div terms are plain fvm::div with no scheme argument. Anything else refuses.
    // The solver the case named for each transported scalar (item 58): `smoothSolver` + a
    // GaussSeidel-family smoother runs OpenFOAM's own sweep under its stopping rule; anything else keeps
    // BiCGStab. Per field, because fvSolution can name one for k and another for epsilon; ONE smoother
    // variant and ONE nSweeps for the pair, which linear_solver_setup refuses to resolve when they differ.
    bool   gsK = false, gsEps = false, gsSymmetric = true;
    int    nSweepsKE = 1;
    bool   boundedK   = false;
    bool   boundedEps = false;    // separate fvSchemes entries; the reference carries ONE bool for both,
                                  // so a case that bounds one and not the other is refused here rather
                                  // than quietly bounded twice or not at all.
    bool   correctedLaplacian = false;   // BOTH halves: the implicit coefficient AND the explicit source
    scalar snGradLimitCoeff   = 0.0;

    // --- relaxation. relax(1.0) is NOT the identity: fvMatrix::relax early-returns only on alpha <= 0,
    //     so at 1.0 it still applies the dominance clamp and moves the source. Hence a flag, not a
    //     sentinel value -- the same argument rhoUEqn.cuh makes. ---
    bool   relaxEquationEps = false;
    scalar relaxEps         = 1.0;
    bool   relaxEquationK   = false;
    scalar relaxK           = 1.0;

    // --- linear solve ---
    scalar tol     = 1e-12;
    scalar relTol  = 0.0;
    int    maxIter = 2000;
    int    minIter = 0;      // fvSolution solvers/<field>/minIter, OF's floor on the iteration count
    // fvSolution's `preconditioner DILU` on k and epsilon, which is what every compressible fixture in
    // the tree names and what the host reference has always run (pbicgstab.cuh is DILU-preconditioned).
    // Null keeps Jacobi.
    //
    // NOT A COST KNOB, and the k equation is where it shows. On sbMatched at iteration 2 the two arms'
    // assembled k systems agree to 1e-11 and both solutions satisfy their own system to
    // sum|Ax-b|/sum|b| ~ 1e-12, yet the solved k differed by 3.6e-06 per entry in the wall cells: the
    // device's Jacobi BiCGStab stalls about ten times short of the host's DILU on that ill-conditioned
    // system, and neither tolerance 1e-15 nor maxIter 40000 moves it. OpenFOAM converges along the DILU
    // path, so the arm that substitutes Jacobi lands somewhere else -- queue item 27.
    const DeviceDilu* precon = nullptr;

    KEpsilonCoeffs co{};
    scalar         Prt = 1.0;            // EddyDiffusivity: alphat = rho*nut/Prt

    // --- the flux-conditional and turbulent inlets, refreshed where OpenFOAM refreshes them:
    //     epsilon_.boundaryFieldRef().updateCoeffs() fires on EVERY patch immediately before the
    //     equation, and it is the ONLY place per iteration where turbulentMixingLengthDissipationRateInlet
    //     recomputes its refValue from k's CURRENT patch values, and
    //     turbulentIntensityKineticEnergyInlet from U's. Freezing them at the case file's `value` is a
    //     silent no-turbulence inlet. A null mask means the case has no such patch; supplying one is the
    //     caller's job because the mask is a property of the case, not of the equation. ---
    // compressible::alphatWallFunction, per boundary face. OF ends every EddyDiffusivity::correctNut with
    // alphat_.correctBoundaryConditions(), and on such a patch that is operator==(rhow*tnutw/Prt_) with
    // the PATCH's own Prt_ (default 0.85) -- not the model's `Prt` above, whose default is 1.0. Two
    // different turbulent Prandtl numbers in one case, so the wall one is carried per face.
    // Null => no alphat boundary is written, which is what every caller got until now.
    const DeviceBuffer<label>*  alphatWallMask = nullptr;
    const DeviceBuffer<scalar>* alphatPrtFace  = nullptr;

    const DeviceBuffer<label>*  turbInletEpsMask = nullptr;
    const DeviceBuffer<scalar>* turbInletEpsLen  = nullptr;   // mixing length, per boundary face
    const DeviceBuffer<label>*  turbInletKMask   = nullptr;
    const DeviceBuffer<scalar>* turbInletKInt    = nullptr;   // turbulent intensity, per boundary face

    // --- fvOptions: a scalarFixedValueConstraint naming k or epsilon forces them on its cells through
    //     setValues, which is not the same as overwriting the field afterwards -- it also cuts the
    //     coupling out of the neighbours' equations. Resolved to a per-cell mask by the caller. ---
    const DeviceBuffer<label>*  fvoEpsMask = nullptr;
    const DeviceBuffer<scalar>* fvoEpsVal  = nullptr;
    const DeviceBuffer<label>*  fvoKMask   = nullptr;
    const DeviceBuffer<scalar>* fvoKVal    = nullptr;

    // --- refusals ---
    bool        hasCoupledPatches      = false;
    bool        hasUnportedFvOption    = false;
    bool        hasNonUpwindDivScheme  = false;
    bool        hasNonWallTurbWallFunc = false;
    std::string fvOptionUnsupported;
    std::string divSchemeUnsupported;
};

// Every intermediate kEpsilon::correct() forms, kept separate so a gate can name the FIRST divergent
// stage rather than reporting one number for the whole closure. Same argument as the other four modules'
// Stages structs: a wall defect and a relaxation defect are the same number once they are folded.
struct KEpsilonStages
{
    DeviceBuffer<scalar> gradU;        // 9*nC, OF column convention
    DeviceBuffer<scalar> gByNu;        // gradU && devTwoSymm(gradU)
    DeviceBuffer<scalar> divU;         // from the VOLUMETRIC flux
    DeviceBuffer<scalar> divPhi;       // from the MASS flux, for `bounded`
    DeviceBuffer<scalar> G;            // nut*gByNu, BEFORE the wall replacement
    DeviceBuffer<scalar> eps0, G0;     // epsilonWallFunction's near-wall values
    DeviceBuffer<label>  isWallCell;   // nw[c] != 0 -- the cells setValues will constrain
    DeviceBuffer<scalar> DepsilonEff;  // cells, WITHOUT the rho the compressible form multiplies in
    DeviceBuffer<scalar> DkEff;
    DeviceBuffer<scalar> gammaEpsFace, gammaKFace;   // the assembled face diffusivity, rho included
    DeviceBuffer<scalar> gammaEpsBnd,  gammaKBnd;

    scalar epsResidual = 0.0;
    scalar kResidual   = 0.0;
    label  wallCells   = 0;
};

// Stage 1: gradU, gByNu, divU, divPhi and G. Touches no matrix. G is left PRE-wall, which is the state
// OpenFOAM's own dump carries.
void production(
    KEpsilonStages&              st,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    const DeviceBuffer<scalar>&  nut,
    const KEpsilonInput&         in);

// Stage 2: epsilonWallFunction. Fills eps0/G0/isWallCell and then applies OpenFOAM's overwrite --
// G[c] = G0[c] and epsilon[c] = eps0[c] on every wall-adjacent cell. Mutates G and epsilon in place,
// exactly where OpenFOAM mutates them.
void wallTreatment(
    KEpsilonStages&              st,
    DeviceBuffer<scalar>&        epsilon,
    const DeviceMesh&            dm,
    const DeviceWallData&        wall,
    const DeviceBuffer<scalar>&  k,
    const KEpsilonInput&         in);

// Stage 3: the epsilon system, up to and including relax -> constrain -> setValues. Solves nothing, so
// it can be compared against the reference's captured system without either side running a solver.
void assembleEpsEqn(
    PressureMatrix&              E,
    KEpsilonStages&              st,
    const DeviceMesh&            dm,
    DeviceBoundary&              dbEps,
    const DeviceBoundary&        dbK,      // turbulentMixingLengthDissipationRateInlet reads k's patch
    const DeviceBuffer<scalar>&  epsilon,
    const DeviceBuffer<scalar>&  k,
    const DeviceBuffer<scalar>&  nut,
    const KEpsilonInput&         in);

// Stage 4: the k system. No wall setValues and no boundaryManipulate -- see the header.
void assembleKEqn(
    PressureMatrix&              K,
    KEpsilonStages&              st,
    const DeviceMesh&            dm,
    DeviceBoundary&              dbK,
    const DeviceVectorBoundary&  dbU,      // turbulentIntensityKineticEnergyInlet reads U's patch
    const DeviceBuffer<scalar>&  k,
    const DeviceBuffer<scalar>&  epsilon,
    const DeviceBuffer<scalar>&  nut,
    const KEpsilonInput&         in);

// Foam::bound(vsf, lowerBound): a cell that solved BELOW the floor takes the AREA-WEIGHTED average of
// its bounded neighbours, not a hard clamp (bound.C:38-56 via fvc::average = surfaceSum(magSf*ssf)/
// surfaceSum(magSf)). The legacy deviceBoundField takes an unweighted face-count mean and reads the CELL
// value on boundary faces, so it is not reused here.
void boundField(
    DeviceBuffer<scalar>&        x,
    const DeviceMesh&            dm,
    const DeviceBoundary&        db,
    scalar                       floor);

// Stage 5: correctNut. nut = Cmu k^2/eps on cells; a turbulence-wall-function FACE takes
// nutkWallFunction; every other boundary face takes Cmu*k_b^2/eps_b -- NOT the owner cell's nut, which
// is what a `calculated` nut patch was getting wrong. Then alphat = rho*nut/Prt, unconditionally and
// whole-field, as EddyDiffusivity::correctNut has it.
void correctNut(
    DeviceBuffer<scalar>&        nut,
    DeviceBuffer<scalar>&        nutBnd,
    DeviceBuffer<scalar>*        alphat,
    DeviceBuffer<scalar>*        alphatBnd,      // null => the boundary is left alone (see alphatWallMask)
    const DeviceMesh&            dm,
    const DeviceBoundary&        dbK,
    const DeviceBoundary&        dbEps,
    const DeviceWallData&        wall,
    const DeviceBuffer<scalar>&  k,
    const DeviceBuffer<scalar>&  epsilon,
    const KEpsilonInput&         in);

// One kEpsilon::correct(): production -> wall -> epsilon eqn -> solve -> bound -> k eqn -> solve ->
// bound -> correctNut. Updates k, epsilon, nut and alphat in place.
//
// Throws (host-side) on every refusal in KEpsilonInput, matching the reference's contract.
void correct(
    DeviceBuffer<scalar>&        k,
    DeviceBuffer<scalar>&        epsilon,
    DeviceBuffer<scalar>&        nut,
    DeviceBuffer<scalar>&        nutBnd,
    DeviceBuffer<scalar>*        alphat,
    DeviceBuffer<scalar>*        alphatBnd,      // null => the boundary is left alone (see alphatWallMask)
    KEpsilonStages&              st,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    DeviceBoundary&              dbK,
    DeviceBoundary&              dbEps,
    const DeviceWallData&        wall,
    const KEpsilonInput&         in);

} // namespace kEpsilonRAS
} // namespace gpu
} // namespace brae
