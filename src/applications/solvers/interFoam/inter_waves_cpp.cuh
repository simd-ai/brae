#pragma once
// interFoam's wave boundary conditions -- waveAlpha on alpha, waveVelocity on U -- and WHEN each asks
// its wave model to update.
//
// provenance:
//   openfoam:  src/waveModels/derivedFvPatchFields/waveAlpha/waveAlphaFvPatchScalarField.C:95-118
//              src/waveModels/derivedFvPatchFields/waveVelocity/waveVelocityFvPatchVectorField.C:95-118
//              src/waveModels/waveModel/waveModelNew.C:82-100 (lookupOrCreate: one model per patch,
//                  created at the FIRST updateCoeffs that asks for it)
//              src/finiteVolume/fvMatrices/solvers/MULES/MULESTemplates.C:168 (the explicit solve's
//                  correctBoundaryConditions, AHEAD of the limiter)
//              src/OpenFOAM/db/Time/Time.C:993-1014, :1057-1066 (the sub-cycle's time and time index)
//   model:     src/waveModels/waveModel/wave_model_cpp.cuh
//   tests:     tests/interfoam_waves_vs_openfoam.sh
//
// BOTH CONDITIONS ARE fixedValue PATCHES WHOSE VALUE IS THE MODEL'S, re-assigned at every updateCoeffs.
// brae's patch-field factory REFUSES both type names, and that stays: a factory that accepted them
// would hand every other solver a wave inlet frozen at the case file's `value`. buildInterFields
// rewrites the two types to fixedValue on its own copy of the file data and records the patches here,
// so the only driver that can build one is the driver that drives it.
//
// THE TIMING IS THE PORT. Read against OpenFOAM's log on laminar/waves/stokesI (nAlphaSubCycles 3):
//
//     Updating StokesI wave model for patch inlet      <- sub-cycle 1, ahead of its MULES solve
//     MULES: Solving for alpha.water
//     Updating StokesI ...    MULES: ...               <- sub-cycles 2 and 3
//     Updating StokesI wave model for patch inlet      <- once more: U's first updateCoeffs, in UEqn
//     Selecting waveModel shallowWaterAbsorption       <- the OUTLET's model is only now CREATED
//     Updating shallowWaterAbsorption wave model for patch outlet
//
//   so alpha's inlet value is evaluated at each SUB-CYCLE's time; the inlet velocity is evaluated
//   from the alpha the sub-cycles LEFT; and the outlet's reference depth, where the case names none,
//   comes from alpha after the first step's advance -- not from the initial field.
//
// WHAT WAS MEASURED, each by breaking it once on stokesI at full amplitude (tests/
// interfoam_waves_vs_openfoam.sh, where brae is 1.2e-10 of alpha from OpenFOAM): every sub-cycle
// evaluated at the step's time 1.0e-04; no fourth update in UEqn 9.5e-03, and the update ORDER arm
// fails; the active absorption left out 2.1e-01.
//
// UNDER MULESCorr the first updateCoeffs of a sub-cycle is the pre-solve's matrix construction instead,
// and THAT position is gated (the gate's `mulescorr` profile): updating after the pre-solve reads
// alpha 8.9e-07 from OpenFOAM, against 1.1e-10.
//
// THE DEVICE LOOP calls the same two functions through its alpha and velocity hooks, at the same clock
// and in the same order (device_inter_alpha_step.cuh, DeviceAlphaBoundary::updateModelled), and is held
// to OpenFOAM on every profile of the gate at the host's bounds.
//
// AND WHAT WAS NOT. The update opens MULES::explicitSolve (MULESTemplates.C:168), so it sits BETWEEN
// the high-order flux, built on the value the last update left, and the limiter. That position is
// transcribed and it agrees with the log -- and moving the update to AFTER the solve changed no digit
// on any fixture tried, up to the tutorial's own mesh at its largest time step (Co 1.9). The explicit
// limiter never reduces lambda on an uncoupled boundary face (it treats wedge and coupled patches
// only, MULESTemplates.C:533-564), so the boundary flux is the high-order one either way; the new
// value reaches the adjacent cell's extrema (:338) and nothing here made that bite. Likewise the
// outlet model's creation time: a case that starts from rest has the same alpha at the outlet on
// both sides of the first step.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "primitive_mesh.cuh"
#include "wave_model_cpp.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

struct InterWaves
{
    bool any = false;
    FoamDict waveProperties;
    // per patch: does alpha carry waveAlpha, does U carry waveVelocity
    std::vector<char> alphaPatch;
    std::vector<char> UPatch;
    // per patch, null until the first update that asks for it -- waveModel::lookupOrCreate
    std::vector<std::shared_ptr<waveModels::WaveModel>> model;
    vector gravity{0, 0, 0};
    std::string alphaName;
    // every update that RAN, in order, as "<patch>@<timeIndex>": OpenFOAM's "Updating ..." lines
    std::vector<std::string> updateLog;
};

// Rewrites `waveAlpha` / `waveVelocity` to fixedValue on the caller's copy of the file data and
// records the patches. Reads constant/waveProperties when there is one to read. Refuses a `waveDict`
// other than the default, and a restart (<startDir>/uniform/waveProperties.<patch>).
InterWaves readInterWaves(
    const std::string& caseDir,
    const std::string& startDir,
    FieldData<scalar>& alphaData,
    FieldData<vector>& UData,
    const std::vector<FvPatch>& patches,
    const vector& gravity,
    const std::string& alphaName);

// waveAlpha::updateCoeffs on every such patch, at OpenFOAM's time `t` and time index.
void updateWaveAlpha(
    InterWaves& w,
    GeometricField<scalar>& alpha1,
    const GeometricField<vector>& U,
    scalar t,
    label timeIndex,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

// waveVelocity::updateCoeffs on every such patch.
void updateWaveVelocity(
    InterWaves& w,
    const GeometricField<scalar>& alpha1,
    GeometricField<vector>& U,
    scalar t,
    label timeIndex,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

// Time::subCycle's clock: the time and time index of sub-cycle k (1-based) of step n (1-based), and
// the step's own when the case does not sub-cycle. The time is ACCUMULATED as OpenFOAM accumulates it
// -- (tNew - deltaT), then += deltaT/nSubCycles per sub-cycle -- because a wave phase is a function of it.
struct SubCycleClock
{
    scalar t = 0;
    label timeIndex = 0;
};

SubCycleClock subCycleClock(
    scalar tNew,
    scalar deltaT,
    label stepIndex,
    label nSubCycles,
    label k);

} // namespace interFoam
} // namespace cpu
} // namespace brae
