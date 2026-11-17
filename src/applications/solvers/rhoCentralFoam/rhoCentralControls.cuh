#pragma once
// rhoCentralFoam's case: the controls OpenFOAM reads (controlDict, fvSchemes' fluxScheme and reconstruct entries,
// fvSolution, thermophysicalProperties, turbulenceProperties), the mesh, and every start-up refusal.
//
// What runs (everything else is refused here, by name, before a field is read): single GPU, static mesh,
// hePsiThermo + pureMixture + const + hConst + perfectGas + sensibleInternalEnergy, laminar with every mu = 0
// (inviscid), Euler with a fixed deltaT or adjustTimeStep (setDeltaT.H on the central Courant number), any purgeWrite
// (milestone D), fluxScheme Kurganov or Tadmor (readFluxScheme.H), reconstruct(rho|T) vanLeer, vanAlbada, Minmod, MUSCL,
// SuperBee, limitedLinear k or upwind and reconstruct(U) their V forms or upwind (milestone F), Gauss linear gradients; on U, p and T the patch fields zeroGradient, fixedValue, slip, inletOutlet on the
// flux `phi` (milestone D), and the constraint types empty, symmetryPlane, symmetry and wedge (milestone C) on their
// own mesh patch types.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "solution_directions.cuh"
#include "rhoCentralThermo.cuh"
#include "rhoCentralPatchUpdate.cuh"
#include "run_clock.cuh"
#include "time_controls.cuh"

#include <optional>

#include <string>
#include <vector>

namespace brae {
namespace rhoCentral {

struct Case
{
    std::string caseDir;
    std::string startName;    // the start time directory, as Time resolves it
    scalar      startTime = 0;
    int         writePrecision = 6;
    int         purgeWrite = 0;         // controlDict purgeWrite (TimeIO.C:322-333; negative read as 0, warned)
    TimeControls timeControls;          // adjustTimeStep, maxCo, maxDeltaT (createTimeControls.H)
    std::optional<RunClock> clock;      // the clock at the start: deltaT, write cadence, endTime (Time::setControls)

    PrimitiveMesh        m;
    FvGeometry           g;
    std::vector<FvPatch> patches;
    SolutionDirections   sd;
    GasCoeffs            gas;
    bool                 tadmor = false;   // fluxScheme Tadmor (readFluxScheme.H); Kurganov otherwise
    // interpolationSchemes reconstruct(rho), reconstruct(T) (rPsi, e, c) and reconstruct(U) (rhoU): the limiter of each,
    // as device_limiter.cuh's limiterOfR selects it (device_mesh.cuh's sentinels; limitedLinear's 2/max(k,SMALL) > 0),
    // and the statement as the case wrote it (milestone F)
    scalar               limRho = -1.0, limT = -1.0, limU = -1.0;   // kVanLeerTwoByk
    std::string          reconRho = "vanLeer", reconT = "vanLeer", reconU = "vanLeerV";

    // flat boundary addressing: every boundary face in patch order (no coupled patch is accepted, so flat index k is
    // mesh face nInternalFaces + k), empty faces included and flagged
    label nC = 0, nIF = 0, nB = 0;
    std::vector<label> bCell;      // owner cell of boundary face k
    std::vector<int>   bPatch;     // patch of boundary face k
    std::vector<char>  bEmpty;     // 1 on an empty patch's face

    // the transform patches (milestone C), face by face; see rhoCentralPatchUpdate.cuh for what each one changes
    std::vector<char>   bRhoConstraint;   // 1: the mesh patch is symmetryPlane, symmetry or wedge
    std::vector<char>   bTseedOwn;        // 1: T's patch field is a transform (its energy BC is the same transform)
    std::vector<int>    bUTransform;      // transformNone, transformMirror (symmetry kinds, slip) or transformRotate
    std::vector<tensor> bUR;              // I - 2.0*sqr(nHat) for a mirror, the wedge's faceT for a rotation

    // inletOutlet (milestone D): which fields carry one anywhere (each such field's patches are switched on this
    // step's phi before they are evaluated), and the faces where any of U, p, T does (the inflow diagnostic)
    bool ioU = false, ioP = false, ioT = false;
    std::vector<char>   bInletOutlet;
};

// Reads and checks the case; refuses everything outside this stage's scope by name, before any field is read.
Case readRhoCentralCase(const std::string& caseDir);

} // namespace rhoCentral
} // namespace brae
