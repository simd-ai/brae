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
//               CELL INDEX, and a motion solver beside the refinement
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

namespace brae {
namespace cpu {
namespace interFoam {

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
};

// Does the case ask for an adaptive mesh at all? ONE function, asked by everything that needs to know --
// buildInterFields (to keep the case away from the MOTION factory), the host loop (to build the state) and
// the DEVICE loop (to refuse it). Three inline dictionary reads would be three chances to disagree.
bool caseAsksForAdaptiveMesh(const std::string& caseDir);

// constant/dynamicMeshDict, and the initial state: the levels a fresh mesh starts at, the identity
// history, and dynamicRefineFvMesh::init's own protected-cell scan. Inactive when the dictionary names
// another mesh type, which is what leaves a static case untouched.
InterAmr readInterAmr(
    const std::string&          caseDir,
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
    RunReport&                rep);

} // namespace interFoam
} // namespace cpu
} // namespace brae
