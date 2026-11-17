#pragma once
// rhoCentralFoam's case: the controls OpenFOAM reads (controlDict, fvSchemes' fluxScheme and reconstruct entries,
// fvSolution, thermophysicalProperties, turbulenceProperties), the mesh, and every start-up refusal.
//
// What runs (everything else is refused here, by name, before a field is read): single GPU, static mesh,
// hePsiThermo + pureMixture + const + hConst + perfectGas + sensibleInternalEnergy, laminar with every mu = 0
// (inviscid), Euler with a fixed deltaT or adjustTimeStep (setDeltaT.H on the central Courant number), purgeWrite 0, fluxScheme Kurganov or Tadmor (readFluxScheme.H),
// reconstruct(rho|T) vanLeer and reconstruct(U) vanLeerV, Gauss linear gradients, zeroGradient / fixedValue / empty
// patch fields on U, p and T.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "solution_directions.cuh"
#include "rhoCentralThermo.cuh"
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
    TimeControls timeControls;          // adjustTimeStep, maxCo, maxDeltaT (createTimeControls.H)
    std::optional<RunClock> clock;      // the clock at the start: deltaT, write cadence, endTime (Time::setControls)

    PrimitiveMesh        m;
    FvGeometry           g;
    std::vector<FvPatch> patches;
    SolutionDirections   sd;
    GasCoeffs            gas;
    bool                 tadmor = false;   // fluxScheme Tadmor (readFluxScheme.H); Kurganov otherwise

    // flat boundary addressing: every boundary face in patch order (no coupled patch is accepted, so flat index k is
    // mesh face nInternalFaces + k), empty faces included and flagged
    label nC = 0, nIF = 0, nB = 0;
    std::vector<label> bCell;      // owner cell of boundary face k
    std::vector<int>   bPatch;     // patch of boundary face k
    std::vector<char>  bEmpty;     // 1 on an empty patch's face
};

// Reads and checks the case; refuses everything outside this stage's scope by name, before any field is read.
Case readRhoCentralCase(const std::string& caseDir);

} // namespace rhoCentral
} // namespace brae
