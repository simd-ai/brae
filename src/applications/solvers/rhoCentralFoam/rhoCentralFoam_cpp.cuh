#pragma once
// rhoCentralFoam, first stage: the case, the conservative state, and the HOST REFERENCE step.
//
// The host step is a transcription of OpenFOAM v2412's applications/solvers/compressible/rhoCentralFoam/
// rhoCentralFoam.C (lines 102-292), term by term and in its order, restricted to what this stage runs:
//
//   single GPU, static mesh, hePsiThermo + pureMixture + const + hConst + perfectGas + sensibleInternalEnergy,
//   laminar and inviscid (every mu = 0), fixed deltaT (adjustTimeStep no), Euler, fluxScheme Kurganov or
//   Tadmor, reconstruct(rho|T) vanLeer and reconstruct(U) vanLeerV, every gradient Gauss linear, boundary
//   conditions zeroGradient / fixedValue / empty on U, p and T.
//
// Everything outside that is REFUSED at start-up by name (readRhoCentralCase), with the OpenFOAM file it comes
// from. The host step is the oracle the CUDA arm (rhoCentralFoam_device.cu) is held to, and both are held to
// OpenFOAM by tests/rho_central_sod_vs_openfoam.sh.
//
// Not SIMPLE, not PIMPLE: there is no pressure equation. rho, rhoU and rhoE advance explicitly through
// `solve(fvm::ddt(q) + fvc::div(flux))` with the diagonal solver, and p comes from p = rho/psi.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "solution_directions.cuh"
#include "rhoCentralThermo.cuh"

#include <string>
#include <vector>

namespace brae {
namespace rhoCentral {

// The boundary conditions this stage runs, per patch and field. Anything else is refused by name.
enum BcKind : int
{
    bcEmpty        = 0,   // emptyFvPatchField: zero-sized in OpenFOAM, in no surface sum
    bcZeroGradient = 1,   // value = patchInternalField
    bcFixedValue   = 2    // value kept
};

struct Case
{
    std::string caseDir;
    std::string startName;    // the start time directory, as Time resolved it
    scalar      startTime = 0;
    scalar      deltaT    = 0;
    long        nSteps    = 0;
    int         writePrecision = 6;

    PrimitiveMesh       m;
    FvGeometry          g;
    std::vector<FvPatch> patches;
    SolutionDirections  sd;
    GasCoeffs           gas;
    bool                tadmor = false;   // fluxScheme Tadmor (readFluxScheme.H); Kurganov otherwise

    // flat boundary addressing: every boundary face in patch order (no coupled patch is accepted, so flat index k
    // is mesh face nInternalFaces + k), empty faces included and flagged
    label nC = 0, nIF = 0, nB = 0;
    std::vector<label> bCell;      // owner cell of boundary face k
    std::vector<int>   bPatch;     // patch of boundary face k
    std::vector<char>  bEmpty;     // 1 on an empty patch's face

    // per-patch boundary condition kind of U, p and T (the energy's follows T: basicThermo::heBoundaryTypes,
    // fixedValue T -> fixedEnergy, zeroGradient T -> gradientEnergy)
    std::vector<int> bcU, bcP, bcT;
};

// The solved state. Cells are nC long, boundary arrays nB long (flat, see Case).
struct State
{
    std::vector<scalar> rho, rhoE, e, T, p, psi;
    std::vector<vector> rhoU, U;
    std::vector<scalar> rhoB, rhoEB, eB, TB, pB, psiB;
    std::vector<vector> rhoUB, UB;
};

// Per-step diagnostics both arms report.
struct StepReport
{
    scalar meanCoNum = 0;     // centralCourantNo.H, on this step's amaxSf
    scalar CoNum     = 0;
    // boundary fluxes of this step, summed over non-empty boundary faces: what leaves the domain
    scalar bndMass   = 0;
    vector bndMom    {0, 0, 0};
    scalar bndEnergy = 0;
    int    status    = 0;     // nonzero: a thermo inversion OpenFOAM would have stopped on (see gasTEs)
};

// Reads and checks the case: refuses everything outside this stage's scope by name, before any field is read.
Case readRhoCentralCase(const std::string& caseDir);

// createFields.H + the thermo's construction (heThermo::init, hePsiThermo's constructor calculate), on the host.
// Both arms start from this state.
State createFields(const Case& cs);

// One time step of rhoCentralFoam.C on the host: lines 102-292, the !LTS branch, with runTime already advanced.
StepReport stepCpp(const Case& cs, State& s);

// Integral diagnostics of a state: totals of V*rho, V*rhoU, V*rhoE, and the extrema the positivity check reads.
struct Integrals
{
    long double mass = 0, energy = 0;
    long double mom[3] = {0, 0, 0};
    scalar minRho = 0, minP = 0, minT = 0, minE = 0;
    scalar maxRho = 0, maxP = 0, maxT = 0, maxMagU = 0;
};
Integrals integrals(const Case& cs, const State& s);

// Writes U, p, T, rho to <caseDir>/<timeName>, as OpenFOAM's AUTO_WRITE fields, at the case's writePrecision.
void writeTime(const Case& cs, const State& s, const std::string& timeName);

} // namespace rhoCentral
} // namespace brae
