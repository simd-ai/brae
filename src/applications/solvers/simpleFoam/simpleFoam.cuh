#pragma once
// CUDA DRIVER -- one simpleFoam SIMPLE iteration, device-resident, composed of the ported components.
//
// provenance:
//   openfoam:  applications/solvers/incompressible/simpleFoam/simpleFoam.C:79-97 (the time-loop body)
//   reference: src/applications/solvers/simpleFoam/simpleFoam_cpp.cu
//   cuda:      src/applications/solvers/simpleFoam/simpleFoam.cu
//   tests:     tests/test_simple_step_cuda.cu
//
// THE DRIVER OWNS NO NUMERICS -- the same contract as the _cpp driver it mirrors:
//
//     gpu::assembleUEqn         UEqn.cu           (fvm::div + divDevReff + relax)
//     gpu::addPressureGradient  UEqn.cu           (-fvc::grad(p))
//     deviceJacobiBiCGStab      lduMatrix/solvers (the momentum solve, non-symmetric)
//     gpu::pressurePredictor    pEqn.cu           (rAU, HbyA, phiHbyA, adjustPhi)
//     gpu::assemblePEqn         pEqn.cu           (laplacian == div(phiHbyA), setReference)
//     deviceAMGPCG              GAMGPreconditioner(the pressure solve)
//     gpu::correctFlux          pEqn.cu           (phi = phiHbyA - pEqn.flux())
//     gpu::relaxField           pEqn.cu           (p.relax())
//     gpu::correctVelocity      pEqn.cu           (U = HbyA - rAU*grad(p))
//     correct()                 caller-supplied   (turbulence->correct(), simpleFoam.C:94)
//
// Compare with what it replaces: device_simple_foam.cu, 3578 lines, included by pimpleFoam,
// rhoSimpleFoam and five headers in solvers/common.
//
// ORDERING, taken from OpenFOAM and not rearranged:
//   * the momentum predictor solves a COPY of UEqn with -grad(p) added; pEqn.H needs the original for
//     A() and H(), and adding grad(p) to UEqn itself changes rAU and HbyA;
//   * p is relaxed AFTER the flux correction and BEFORE the velocity correction, so phi is built from
//     the unrelaxed pressure and U from the relaxed one;
//   * turbulence->correct() runs at the END, so the next iteration's UEqn uses this iteration's nut --
//     a lagged coupling, not a simultaneous one.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_amg.cuh"
#include "UEqn.cuh"
#include "actuation_disk.cuh"
#include "device_MRF.cuh"
#include "pEqn.cuh"
#include <functional>
#include <map>
#include <string>

namespace brae {
namespace gpu {

// The device-resident solution state the loop carries between iterations.
struct SolverFields
{
    DeviceBuffer<scalar> Ux, Uy, Uz;
    DeviceBuffer<scalar> p;
    DeviceBuffer<scalar> phiInt, phiBnd;
};

// Optional instrumentation of the driver's internals. The per-stage tests compare each kernel against the
// _cpp reference and pass at 1e-16, yet the driver's phi output is wrong by 1.17 relative from identical
// inputs -- so the defect is in what the driver FEEDS a stage, not in the stage. That is invisible from
// outside, hence this. Host-side vectors, filled only when a probe is attached; nullptr costs nothing.
struct StepProbe
{
    std::vector<scalar> rAU, rAUface;
    std::vector<scalar> HbyA[3], HbyAb[3];
    std::vector<scalar> phiHbyAInt, phiHbyABnd;
    std::vector<scalar> pDiag, pUpper, pLower, pSource, pIC, pBC;
    std::vector<scalar> pSolved, phiInt, phiBnd;
};

struct StepInput
{
    // nuEff for THIS iteration. Supplied by the caller because it comes from the turbulence model, which
    // the caller owns (see `correct` below): nuEff = nu + nut from the PREVIOUS iteration.
    const DeviceBuffer<scalar>* nuEffCell = nullptr;
    const DeviceBuffer<scalar>* nuEffFace = nullptr;
    const DeviceBuffer<scalar>* nuEffBndFace = nullptr;

    scalar relaxU = 0.7;
    scalar relaxP = 0.3;
    scalar tolU = 1e-10, relTolU = 0.0;
    // OF fvVectorMatrix::solveSegregated solves each U component with the `U` lduMatrix solver, and
    // pitzDaily -- like most tutorials -- asks for smoothSolver/symGaussSeidel. Routing that through
    // Jacobi-BiCGStab is a silent solver substitution AND the dominant cost: measured ~42 BiCGStab
    // iterations per component per outer step, ~90% of all linear-algebra time on this path.
    bool   uSymGaussSeidel = false;
    scalar tolP = 1e-10, relTolP = 0.0;
    int    maxIter = 2000;
    // V-cycle graph replay and PCG residual-read cadence. NOTE: on CUDA >= 13 deviceAMGPCG dispatches
    // first to its DEVICE-RESIDENT conditional-graph PCG (BRAE_PCG_DEVICE, default on), which captures
    // the whole Krylov loop and ignores both of these -- measured A/B: 17.9 s on vs 18.7 s off at 440k.
    // These therefore only reach the host-driven fallback. They are passed explicitly rather than left
    // to defaults so the fallback is not silently the slow one.
    // Reuse the AMG hierarchy STRUCTURE across runs. The agglomeration is the build cost and depends only
    // on the mesh (the matrix VALUES are re-Galerkined every iteration regardless), so it serializes next
    // to constant/polyMesh and is reloaded when it is newer than `owner`. Empty = no caching.
    std::string amgCacheDir;
    bool   captureVcycle = true;
    int    pcgCheckEvery = 1;
    bool   momentumPredictor = true;
    bool   bounded = false;              // div(phi,U) `bounded`
    bool   linearUpwind = false;         // div(phi,U) `linearUpwind`: deferred correction, upwind matrix
    scalar gradULimitK = 0.0;            // `grad(U) cellLimited Gauss linear <k>` (0 = plain Gauss)
    cpu::DivScheme scheme = cpu::DivScheme::upwind;   // the div(phi,U) scheme, shared with the reference
    scalar         schemeCoeff = 1.0;                 // the `k` of `limitedLinear k`
    const DevicePorosity* porosity = nullptr;         // explicitPorositySource/DarcyForchheimer
    const DeviceRotorDisk* rotor = nullptr;           // rotorDiskSource (Froude blade-element momentum)
    const DeviceActuationDisk* actuationDisk = nullptr;   // actuationDiskSource (Froude actuator disk)
    // MRF zones, already resolved against the mesh and uploaded. Null with hasMRF set is a REFUSAL:
    // a case that declares MRF and gets none of it converges to a confidently wrong answer.
    const std::vector<DeviceMRFZone>* mrf = nullptr;
    scalar         nuLaminar = 0.0;
    bool   correctedLaplacian = false;   // `corrected` laplacianSchemes
    // `limited <k> corrected` (OF limitedSnGrad): caps the non-orth correction against the orthogonal
    // part of the same snGrad. 0 = no cap, which is what `corrected` and `limited 1` both mean.
    scalar snGradLimitCoeff = 0.0;
    label  nNonOrthogonalCorrectors = 0;

    label  pRefCell = -1;
    scalar pRefValue = 0.0;

    const DeviceBuffer<label>* adjustable = nullptr;        // adjustPhi mask   (!fixesValue)
    const DeviceBuffer<label>* takeUAtBoundary = nullptr;   // constrainHbyA mask (!assignable)

    bool hasMRF = false, hasFvOptions = false, consistent = false;   // all refused downstream

    StepProbe* probe = nullptr;   // optional; see StepProbe

    // turbulence->correct(), called at the END of the iteration exactly where simpleFoam.C:94 calls it.
    // A hook rather than a hard dependency because the model is runtime-selected: the driver must not
    // know which one it is, and a driver that hard-codes kEpsilon is the god object starting over.
    std::function<void()> correct;
};

using Residuals = std::map<std::string, scalar>;


// Scratch that persists across iterations (the AMG hierarchy is built once per mesh, and `ones` is the
// unit vector deviceNormFactor needs). Kept out of StepInput so the loop cannot accidentally rebuild it.
struct SolverWorkspace
{
    AMGData amg;
    bool    amgBuilt = false;
    DeviceBuffer<scalar> ones;
    // The pressure system's PERSISTENT buffers. The existing GPU driver keeps these as members and
    // re-Galerkins the hierarchy against the same allocations every iteration; allocating them fresh each
    // iteration instead gives the AMG state a different fine matrix to attach to each time, and the
    // cached V-cycle/PCG graphs are keyed on that matrix. Keeping them here matches the pattern that
    // works and removes a whole class of stale-pointer question.
    PressureMatrix       P;
    DeviceBuffer<scalar> diagC, b;
};

// One SIMPLE iteration, in place on `f`.
Residuals simpleStep(
    SolverFields&                f,
    SolverWorkspace&             w,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    const DeviceBoundary&        dbP,
    const StepInput&             in);

} // namespace gpu
} // namespace brae
