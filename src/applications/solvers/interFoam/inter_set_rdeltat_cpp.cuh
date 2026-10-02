#pragma once
// setRDeltaT.H -- the local time step of a localEuler (LTS) VoF case. Host reference.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/VoF/setRDeltaT.H:1-136
//              applications/solvers/multiphase/interFoam/interFoam.C:90-105 (called before ++runTime, in
//                  place of CourantNo/alphaCourantNo/setDeltaT)
//              src/finiteVolume/cfdTools/general/include/createRDeltaT.H:1-26 (the field: `rDeltaT`, 1/s,
//                  READ_IF_PRESENT, AUTO_WRITE, extrapolatedCalculated patches)
//              src/finiteVolume/finiteVolume/fvc/fvcAverage.C:45-116 (fvc::average of linearInterpolate)
//              src/finiteVolume/finiteVolume/fvc/fvcSurfaceIntegrate.C:163-182 (surfaceSum)
//   brae:      fvc_smooth_cpp.cuh (the smoothing wave)
//   tests:     tests/interfoam_dtchull_vs_openfoam.sh -- RAS/DTCHull, rDeltaT against the field OpenFOAM
//              writes at every step
//
// WHAT IT COMPUTES, per cell, in this order:
//   S1  rDeltaT = max(1/maxDeltaT, surfaceSum(|rhoPhi|)/(((2*maxCo)*V)*rho))
//   S2  when maxAlphaCo < maxCo:
//       rDeltaT = max(rDeltaT, ((pos0(aBar - alphaSpreadMin)*pos0(alphaSpreadMax - aBar))*surfaceSum(|phi|))
//                              /((2*maxAlphaCo)*V)),  aBar = fvc::average(alpha1)
//   S5  fvc::smooth(rDeltaT, rDeltaTSmoothingCoeff) when that coefficient is below 1
//   S8  rDeltaT = max(rDeltaT, (1 - rDeltaTDampingCoeff)*rDeltaT0) when that coefficient is below 1 and
//       runTime.timeIndex() > runTime.startTimeIndex() + 1 -- tested BEFORE ++runTime, so from the THIRD
//       step of every run, a restart included
//
// FOUR THINGS THAT ARE NOT WHAT THEY LOOK LIKE:
//
//   a. THE CONTROLS ARE fvSolution's PIMPLE, NOT controlDict's. setRDeltaT.H:4 reads pimple.dict(), and
//      its maxCo and maxDeltaT shadow the controlDict keys setDeltaT.H reads. The defaults are OpenFOAM's
//      (:6-54): nAlphaSpreadIter 1 and nAlphaSweepIter 5 are ON unless the case says 0, and maxDeltaT is
//      GREAT, 1e15 in a double build (doubleScalar.H:57).
//   b. S1 READS rhoPhi AND S2 READS phi, and they are not the same step's. rhoPhi is the last alpha
//      step's mass flux, built from phiCN BEFORE that step's pressure correctors (alphaEqn.H:248) -- or,
//      at the first step, createFields.H's interpolate(rho)*phi, from phi as it was read, BEFORE
//      initCorrectPhi. phi is the flux AFTER the last pressure corrector.
//   c. THE INTERFACE MASK IS ON aBar, the face-area-weighted average of alpha1's LINEAR face values --
//      alpha1's STORED patch values on the boundary -- not on the cell's alpha and not nearInterface().
//   d. rDeltaT's PATCH VALUE IS THE FACE CELL'S, on every patch type the field can hold: the
//      extrapolatedCalculated OpenFOAM asks for (evaluate: patchInternalField), and the symmetryPlane a
//      constraint patch gives it instead ((a + a)/2 == a). So only the cells are carried; the face field
//      localEulerDdtScheme interpolates takes the face cell's value on a patch.
//
// rDeltaT's READ_IF_PRESENT is INERT and is not read: S1 overwrites every cell, and the value read would
// seed only rDeltaT0, which only S8 reads, and S8 is off for the first two steps of a run.
//
// NOT PORTED, refused where the case is read (refuseUnportedLocalEuler): fvc::spread (nAlphaSpreadIter
// > 0, which is the DEFAULT), fvc::sweep (nAlphaSweepIter > 0, also the default), and a mesh with a
// coupled patch (the smoothing wave does not cross one).
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "primitive_mesh.cuh"
#include "fvc.cuh"
#include <set>
#include <string>
#include <vector>

#include "primitive_patch_cpp.cuh"

namespace brae {
namespace cpu {
namespace interFoam {

// setRDeltaT.H:6-54, pimpleDict.getOrDefault, OpenFOAM's defaults
struct LocalEulerControls
{
    scalar maxCo = 0.9;
    scalar maxAlphaCo = 0.2;
    scalar rDeltaTSmoothingCoeff = 0.1;
    label nAlphaSpreadIter = 1;
    scalar alphaSpreadDiff = 0.2;
    scalar alphaSpreadMax = 0.99;
    scalar alphaSpreadMin = 0.01;
    label nAlphaSweepIter = 5;
    scalar rDeltaTDampingCoeff = 1.0;
    scalar maxDeltaT = 1.0e15;
};

LocalEulerControls readLocalEulerControls(const FoamDict& pimple);

// The parts of setRDeltaT.H brae does not carry, refused by name with the case.
void refuseUnportedLocalEuler(
    const LocalEulerControls& c,
    const std::vector<FvPatch>& patches);

struct SetRDeltaTInput
{
    const SurfaceScalarField* rhoPhi = nullptr;         // note b
    const SurfaceScalarField* phi = nullptr;
    const GeometricField<scalar>* alpha1 = nullptr;     // cells AND stored patch values, note c
    const std::vector<scalar>* rho = nullptr;           // cells
    // runTime.timeIndex() > runTime.startTimeIndex() + 1, evaluated before ++runTime
    bool damp = false;
    // primitiveMesh::cells() for fvc::smooth's wave, when the caller keeps it across steps; null builds it
    // in the call. The caller owns its validity: it must be rebuilt whenever the mesh's addressing changes.
    const CellFaces* cells = nullptr;
};

// The three Info lines, gMin/gMax of 1/rDeltaT after S2, after S5 and after S8.
struct SetRDeltaTReport
{
    scalar flowMin = 0;
    scalar flowMax = 0;
    scalar smoothedMin = 0;
    scalar smoothedMax = 0;
    bool damped = false;
    scalar dampedMin = 0;
    scalar dampedMax = 0;
};

// setRDeltaT.H:1-136. `rDeltaT` holds the previous step's cells on entry -- rDeltaT0 -- and this step's on
// return.
SetRDeltaTReport setRDeltaT(
    std::vector<scalar>& rDeltaT,
    const LocalEulerControls& c,
    const SetRDeltaTInput& in,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

// A GATE'S CONTROL, never set by a solver: BRAE_CONTROL_LTS_SCALAR names the localEuler consumers -- any of
// alpha, ueqn, ddtcorr, turbulence, comma-separated -- that read 1/deltaT instead of the local rDeltaT. Each
// makes the answer WRONG. A LIST so a loop that carries only some consumers can be held against one that
// carries all of them with the rest switched off: the device port took them one at a time that way. An
// unknown name is refused, since it would make the control vacuous. Prints the CONTROL MODE line when set.
std::set<std::string> readLtsScalarControl();

// GATE CONTROLS on setRDeltaT itself, never set by a solver, both WRONG: BRAE_CONTROL_LTS_NODAMP never damps,
// BRAE_CONTROL_LTS_NOSMOOTH skips the smoothing wave. Applied to a copy of the case's controls and the
// step's damp flag; prints the CONTROL MODE line for each one set.
void applySetRDeltaTControls(
    LocalEulerControls& c,
    bool& damp);

// fvc::interpolate(rDeltaT) as localEulerDdtScheme forms it (localEulerDdtScheme.C:385): linear through
// interpolationSchemes' default, lambda*(P - N) + N on an internal face, and the face cell's value on a
// patch (note d).
SurfaceScalarField interpolateRDeltaT(
    const std::vector<scalar>& rDeltaT,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

} // namespace interFoam
} // namespace cpu
} // namespace brae
