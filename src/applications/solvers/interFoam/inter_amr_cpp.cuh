#pragma once
// interFoam on an ADAPTIVE mesh: the driver's mesh-update stage for `dynamicFvMesh dynamicRefineFvMesh`,
// and what the solver's own state has to do when the mesh under it changes. Unit 8c of the
// dynamicRefineFvMesh port.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/interFoam/interFoam.C:118-148 (mesh.update() and what the
//                  solver rebuilds when the mesh changed)
//              src/dynamicFvMesh/dynamicRefineFvMesh/dynamicRefineFvMesh.C (the driver itself)
//   brae:      the driver and every mapper are src/dynamicFvMesh/dynamicRefineFvMesh (units 7, 7b) and
//              the patch fields' own autoMap is src/finiteVolume/fields/fv_patch_field.cuh (unit 8a)
//   tests:     tests/interfoam_amr_vs_openfoam.sh
//
// WHAT IS MAPPED AND WHAT IS RECOMPUTED, and the split is OpenFOAM's rather than a convenience:
//   mapped      alpha1, U, p_rgh (cells and patch fields), phi and rhoPhi (ORIENTED surface fields),
//               nHatf (a surface field that is NOT oriented -- it carries an area, and its dimensions say
//               so), and the old-time volumes
//   recomputed  gh, ghf and p, which are functions of the CELL CENTRES, and alpha2, rho, mu, nu and the
//               curvature K, which mixture.correct() rebuilds from alpha1 -- interFoam.C:139-147 does
//               exactly that after any change, so mapping them would be overwritten work
//
// THE INVENTORY, member by member, because a mesh-sized member that is neither mapped nor refused is a
// silent defect and there are forty-odd of them:
//   MAPPED      alpha1, U, p_rgh (cells + patch fields), phi, rhoPhi (oriented surface scalars), nHatf
//               (a surface scalar that is NOT oriented -- it carries an area), Uf (a surface VECTOR, not
//               oriented -- a velocity), rAU (a cell field the next CorrectPhi interpolates), and the
//               old-time volumes V0
//   RECOMPUTED  gh, ghfInternal, ghfBoundary and p (functions of the new centres), and alpha2, rho, mu,
//               nu with their boundary lists and the curvature K (mixture.correct() rebuilds them from
//               the mapped alpha1) -- interFoam.C:139-147 does exactly that after any change
//   NOT SIZED BY THE MESH  every scheme, control, solver setting and coefficient; movingWallVelocityPatch
//               is per PATCH, and the patch count cannot change (checked where the patches are assigned)
//   REFUSED     the turbulence fields, the wave models, the MRF zones, the fvOptions cell sets, the
//               CrankNicolson ddt0 levels (both the solver's and the closure's), a pressure-reference
//               CELL INDEX, and a motion solver beside the refinement on a 2-D mesh (twoDCorrectPoints) or
//               under CrankNicolson or moveMeshOuterCorrectors. A 3-D solidBody motion of the whole mesh
//               RUNS: its points0 is carried (RefineUpdateState::points0), its V0 is the change's
//               (DynamicMotionSolverFvMesh::topoChanged), and the move follows the change
//               (tests/interfoam_amr_motion_vs_openfoam.sh, laminar/oscillatingBox)
//
// UF AND rAU ARE NOT MOVING-MESH EXTRAS HERE, and that was measured rather than assumed: `correctPhi`
// defaults to mesh.dynamic() and a REFINING mesh is dynamic, so OpenFOAM's own run of
// damBreakWithObstacle writes a Uf and an rAU beside every time directory and solves pcorr at every step.
//
// AND EVERYTHING ELSE IS REFUSED BY NAME. A solver carries more state than fields, and a piece of it that
// silently survives a topology change is the defect this project keeps finding: a pressure-reference CELL
// INDEX that now names a different cell, an MRF zone resolved against the old numbering, a
// CrankNicolson ddt0 level of the wrong length, a turbulence field nobody mapped. Each of those throws and
// names itself until its own unit lands.
#include "cf_types.cuh"
#include "dynamic_refine_fv_mesh_cpp.cuh"
#include "inter_case_cpp.cuh"
#include "crank_nicolson_ddt_scheme_cpp.cuh"   // CrankNicolsonDdt0, the levels a change maps
#include "inter_driver_cpp.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

// One zone of a polyMesh/{cell,face,point}Zones file, as the writer echoes it: its name and type, and how many
// members it has (a zone with members, or a face zone's flip map, is not written -- no oracle holds one).
struct ZoneEntry
{
    std::string name;
    std::string type;
    label nMembers = 0;
    // an entry key other than `type` and the members (inGroups, a flipMap): not echoed
    bool extraKeys = false;
};

struct InterAmr
{
    // false on every case that does not ask for dynamicRefineFvMesh, which is all but three of the
    // forty-four interFoam tutorials
    bool                                 active = false;
    dynamicRefine::RefineControls        controls;
    dynamicRefine::RefineUpdateState     state;
    // what the last step did, for the driver's report and for a gate to read
    label                                nRefined = 0;
    label                                nUnrefined = 0;
    // ...and the polyMesh directory, which the `cellSet` selection mode re-reads at every change -- as
    // OpenFOAM does, from the file, in the ORIGINAL numbering (cellSetOption.C:269-276)
    std::string                          polyMeshDir;
    // hexRef8's level0Edge (hexRef8.C:1939-1951): read if present, computed on the start mesh otherwise, and
    // the same at every write
    scalar level0Edge = 0;
    // the mesh's zones in FILE order, which is the order OpenFOAM writes them in
    std::vector<ZoneEntry> cellZoneEntries;
    std::vector<ZoneEntry> faceZoneEntries;
    std::vector<ZoneEntry> pointZoneEntries;
    // true from the first topology change on: polyMesh::updateMesh makes the mesh files AUTO_WRITE at that
    // instance (polyMeshUpdate.C:57-65) and nothing sets them back, so every later write carries them
    bool topoChanged = false;
    // the start directory's uniform/time `index` (0 without one): OpenFOAM's refine schedule tests the GLOBAL
    // time index, `timeIndex % refineInterval` (dynamicRefineFvMesh.C:1320), which a restart continues from
    // there. The driver's step count restarts at 1, so a restart with refineInterval > 1 refined on the wrong
    // steps -- MEASURED: restarted from OpenFOAM's 0.001 with refineInterval 2, OpenFOAM refined 32256 to
    // 42266 at step 2 and brae refined nothing, U 2.2e-01 off.
    label startTimeIndex = 0;
};

// Does the case ask for an adaptive mesh at all? ONE function, asked by everything that needs to know --
// buildInterFields (to build its motion with the LIST factory), the host loop (to build the state) and
// the DEVICE loop (to refuse it). Three inline dictionary reads would be three chances to disagree.
bool caseAsksForAdaptiveMesh(const std::string& caseDir);

// constant/dynamicMeshDict, and the initial state: the levels a fresh mesh starts at, the identity
// history, and dynamicRefineFvMesh::init's own protected-cell scan. Inactive when the dictionary names
// another mesh type, which is what leaves a static case untouched.
InterAmr readInterAmr(
    const std::string&          caseDir,
    // the polyMesh directory of the FACES instance, which is where cellLevel, pointLevel,
    // refinementHistory and the three zone lists live -- polyMesh reads every one of them at
    // faces_.instance(), so on a restart of a refined case they are NOT under constant/
    const std::string&          facesPolyMeshDir,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    const FvGeometry&           g);

// THE OLD-TIME LEVELS, which are NOT in InterFields: the driver keeps them as locals of its time loop, and
// OpenFOAM keeps them as REGISTERED FIELDS -- which is why its own output carries `alpha.water_0` and
// `U_0` beside every time directory. They are MAPPED, never re-captured: re-capturing them from the new
// field would make every ddt zero on the step that refined.
//
// MEASURED, and it is how this struct came to exist: without them the first refinement of
// damBreakWithObstacle ran to "UEqn: ddt field lengths disagree" -- 42,266 cells against old-time levels
// still 32,256 long.
struct InterAmrOldTime
{
    std::vector<scalar>*                    alphaOld = nullptr;
    std::vector<vector>*                    UOld     = nullptr;
    std::vector<std::vector<vector>>*       UOldBnd  = nullptr;
    std::vector<scalar>*                    rhoOld   = nullptr;
    std::vector<vector>*                    UOO      = nullptr;
    std::vector<std::vector<vector>>*       UOOBnd   = nullptr;
    std::vector<scalar>*                    rhoOO    = nullptr;
    // ...and ALPHA's old-old level, which is the DEVICE arm's representation of the same fact: its closure
    // rebuilds rho.oldTime().oldTime() from alpha1.oldTime().oldTime() (dCn.alpha1OO) where the host loop
    // keeps rhoOO directly. Null on the host arm, which has no such field.
    std::vector<scalar>*                    alphaOO  = nullptr;
    SurfaceScalarField*                     phiOld   = nullptr;
    // ...and Uf.oldTime(), which is what ddtCorr reads in phi.oldTime()'s place on a DYNAMIC mesh --
    // and a refining mesh is dynamic (fvcDdt.C:219-228 branches on mesh.dynamic(), not moving()). Left
    // unmapped it is the old face count at the next ddt term, which throws before it can be wrong.
    SurfaceVectorField*                     UfOld    = nullptr;
};

// THE CrankNicolson SCHEME'S OWN FIELDS, which are fields and not scheme state. Each ddt0 level is a
// registered GeometricField in OpenFOAM's mesh registry (CrankNicolsonDdtScheme's DDt0Field, REGISTER +
// AUTO_WRITE), so MapGeometricFields autoMaps its internal field AND every patch field with no help from
// the scheme -- there is no topoChanging branch in CrankNicolsonDdtScheme.C at all. brae keeps them as
// locals of its time loop, so the driver hands them here.
//
// WHICH ONES EXIST on a REFINING interFoam case, from OpenFOAM's own dispatch:
//   ddt0(rho,U)         volVectorField      UEqn.H's fvm::ddt(rho, U)
//   ddtCorrDdt0(U)      volVectorField      fvcDdtUfCorr
//   ddtCorrDdt0(Uf)     surfaceVectorField  fvcDdtUfCorr -- the Uf branch, because ddtCorr(U, phi, Uf)
//                                           routes on mesh.dynamic() and a refining mesh IS dynamic
// and two that do NOT: ddtCorrDdt0(phi) (the phi branch is never taken) and meshPhiCN_0 (only
// fvc::meshPhi creates it, and a refine-only mesh has no mesh flux). ddt(alpha) creates none at all --
// alphaEqn.H builds a throwaway CN scheme to read ocCoeff() and solves alpha with Euler.
//
// Beside them ride the old-OLD levels and the alpha flux's blend state, which are the driver's own:
// UfOO (a surface vector), phiOO (an oriented flux) and alphaPhi10End/alphaPhi10Old (oriented fluxes).
struct InterAmrCn
{
    fv::CrankNicolsonDdt0<vector>*  ddt0RhoU    = nullptr;
    fv::CrankNicolsonDdt0<vector>*  ddtCorrU    = nullptr;
    fv::CrankNicolsonDdt0<vector>*  ddtCorrUf   = nullptr;
    fv::CrankNicolsonDdt0<scalar>*  ddtCorrPhi  = nullptr;   // never created here; checked, not mapped
    SurfaceVectorField*             UfOO        = nullptr;
    SurfaceScalarField*             phiOO       = nullptr;
    SurfaceScalarField*             alphaPhiEnd = nullptr;
    SurfaceScalarField*             alphaPhiOld = nullptr;
};

// One mesh.update() for an adaptive mesh, at the top of an outer corrector. Returns true when the mesh
// changed -- which is what interFoam.C branches on to rebuild gh, the MRF and the mixture.
//
// The mesh, its geometry and its patches are updated THROUGH `mm`: the patch objects are assigned in
// place, because every patch field holds a reference into that vector (see updatePatchesInPlace).
bool interAmrUpdate(
    InterAmr&              amr,
    InterFields&           f,
    const MutableMesh&     mm,
    label                  timeIndex,
    const InterAmrOldTime& old = InterAmrOldTime{},
    const InterAmrCn&      cn = InterAmrCn{});

// interFoam.C:139-147: what the solver rebuilds once the mesh HAS changed, which is everything the
// inventory calls RECOMPUTED. Every one of these ASSIGNS its result rather than writing into a vector at
// the old size -- which is what the mesh change makes load-bearing, because the old size is wrong now.
//
// The GAMG agglomeration is CLEARED here and not rebuilt: it is keyed on the mesh it was built for, and a
// refined mesh is a different one. Keeping it would solve the pressure equation on the old coarse levels.
void interAfterMeshChange(
    InterFields&              f,
    const MutableMesh&        mm,
    GamgAgglomerationCache&   gamgCache,
    const CorrectPhiControls& cpc,
    RunReport&                rep,
    // the step being taken, which the TURBULENCE recompute needs: wallDist's schedule tests it modulo the
    // interval (wallDist.C:198). REQUIRED rather than defaulted or stashed on the AMR state -- a stashed
    // index is right only while every caller runs interAmrUpdate immediately before this, and that is the
    // kind of invariant this port keeps finding broken.
    label                     timeIndex,
    // THE MESH ALSO MOVES THIS STEP, right after the change (dynamicRefineFvMesh::update: topology first,
    // motion second). interFoam.C:112-148 is ONE block after mesh.update(): gh/ghf, MRF, and under
    // correctPhi `phi = Sf & Uf`, CorrectPhi, makeRelative and mixture.correct() -- all on the MOVED mesh.
    // The motion's own block (interMeshUpdate) runs that, with the mesh flux, so here only the
    // topology-only rebuild runs: a CorrectPhi here as well would solve pcorr twice, the first time
    // without the mesh flux, and a curvature pass here would be one OpenFOAM does not take.
    bool                      motionFollows,
    // the closure's wall-distance wave run elsewhere (updateMeshInterTurbulence): the device loop hands in its
    // GPU wave; null runs the host's
    const PatchWaveRunner*    waveRunner = nullptr);

} // namespace interFoam
} // namespace cpu
} // namespace brae
