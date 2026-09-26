#pragma once
// A RESTART UNDER CrankNicolson: the scheme's own state, read back off disk as OpenFOAM reads it.
//
// provenance:
//   openfoam: src/finiteVolume/finiteVolume/ddtSchemes/CrankNicolsonDdtScheme/CrankNicolsonDdtScheme.C
//               DDt0Field's READING constructor (:49-64), and ddt0_'s choice between it and the
//               creating one (:101-160)
//             applications/solvers/multiphase/VoF/createAlphaFluxes.H:1-23 (alphaPhi0, READ_IF_PRESENT)
//             applications/solvers/multiphase/VoF/alphaEqn.H:36-45 (alphaRestart into ocCoeff)
//   tests:    tests/interfoam_cn_vs_openfoam.sh -- profile `cnRestart`, whose oracle is OpenFOAM's own
//             warm restart and whose control is the same restart with the state removed
//
// TWO FACTS, and they are not the same fact.
//
// (1) THE ddt0 FIELDS ARE READ, and reading one makes the scheme WARM FROM THE FIRST STEP. ddt0_
//     (:101-160) looks the field up in the START directory -- and only while the run's index is the
//     start index or one past it, which for every ddt0 interFoam forms is the first step, since every
//     one of them is assembled there -- and when the file is present it builds the field with the
//     READING constructor, whose startTimeIndex_ is -2: "This field is for a restart and thus correct
//     so set the start time-index to correspond to a previous run". It then sets the field's OWN index
//     to time().startTimeIndex(), so the first step's evaluate() is true and the read ddt0 is
//     off-centred rather than left alone.
//     brae's clock counts steps from 1 (inter_driver_cpp.cu, Stage::advanceTime), so its start index is
//     0 and OpenFOAM's -2 is literally -2 here: coef (timeIndex > startTimeIndex) and coef0
//     (> startTimeIndex + 1) are both 1 + ocCoeff on step one. A cold start's Euler first step and its
//     Euler-estimated second step do not happen.
//
// (2) alphaPhi0's VALUES ARE NEVER USED -- only whether the file is there. alphaEqn.H ASSIGNS
//     alphaPhi10 (:131 under MULESCorr, :208 without) before anything reads its old time;
//     GeometricField::operator= stores old times through ref(), and on the first step there is no
//     level to store; the un-blend's alphaPhi10.oldTime() (:256) then CREATES the level from the value
//     just assigned. So the read numbers are overwritten before anything can read them. What the file
//     changes is `alphaRestart` (createAlphaFluxes.H:10-11), which alphaEqn.H:36-45 ORs with the
//     warm-up test -- so ddt(alpha)'s off-centring, and the un-blend that follows from it, are live on
//     the first step instead of from the third.
//     ASSERTED, not assumed: the gate runs OpenFOAM's warm restart a second time with alphaPhi0's
//     numbers replaced by zeros and requires the answer to be bitwise the oracle's.
//
// NOT HERE, and still refused by name in inter_case_cpp.cu: ddtCorrDdt0(Uf) and meshPhiCN_0, the two a
// MOVING mesh writes. Both are surface fields of a moving-mesh branch that no shipped tutorial restarts
// -- RAS/floatingObject is the only interFoam tutorial that names CrankNicolson and it moves its mesh
// under rigidBodyMotion, which brae refuses before it gets here -- so there is no fixture to gate a
// seed against, and a seed nothing measures is worth less than a refusal.
#include "cf_types.cuh"
#include "crank_nicolson_ddt_scheme_cpp.cuh"
#include "device_crank_nicolson_ddt.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include <string>
#include <vector>

namespace brae {

// WHERE to look for the scheme's state. Empty means "do not look": the case is not under CrankNicolson,
// or the caller has no start directory. ONE FIELD, because that is all a seeder needs -- alphaPhi0's
// presence (fact (2)) is alphaEqn's own fact and lives on InterFields beside this, not in here where the
// closure's copy of the struct could never fill it.
struct InterCnRestart
{
    std::string dir;
};

// DDt0Field's READING constructor on one ddt0 field, from <r.dir>/<ddt0.name>. Nothing happens and
// false comes back when the file is not there -- that is ddt0_'s other branch, which lookupOrCreate
// already is. `nInternal` is the field's own internal count: cells for a volume field,
// internal faces for a surface one. The boundary is read per patch, by name.
//
// THE PATCH VALUES ARE READ BUT NOT WITNESSED by the gate's fixture, and that is measured, not assumed:
// OpenFOAM restarted from the same directory with every ddt0 patch value list replaced by zeros gives
// BITWISE the same answer (0 of 2268 cells, every field, 0.0e+00) -- RAS/damBreak's patches are walls
// whose U fixes a value, where fvcDdtPhiCoeff is zero, plus an empty pair. They are read because the
// field OpenFOAM builds holds them and a case whose patches do not fix U would read them; dropping them
// would be a silent substitution that this fixture cannot see. Said here rather than claimed as gated.
bool seedCnDdt0(
    cpu::fv::CrankNicolsonDdt0<scalar>& ddt0,
    const InterCnRestart& r,
    std::size_t nInternal,
    const std::vector<FvPatch>& patches);
bool seedCnDdt0(
    cpu::fv::CrankNicolsonDdt0<vector>& ddt0,
    const InterCnRestart& r,
    std::size_t nInternal,
    const std::vector<FvPatch>& patches);

// THE OLD-OLD LEVELS, <field>_0 in the start directory. GeometricField's reading constructor calls
// readOldTimeIfPresent (GeometricField.C:104-127 and :130-168), which reads <field>_0 when it is there,
// gives it the index BEFORE the start, and -- finding no <field>_0_0 -- creates the old-old level as a
// copy of it. So the first step's storeOldTimes rotates the READ level into oldTime().oldTime() while
// oldTime() becomes the field at the start time. A COLD start has neither, and both levels are created
// as copies of the field itself, which is what the drivers do when these return false.
//
// WHICH FIELDS HAVE ONE, and why this is CrankNicolson's business and no other scheme's: storeOldTime
// (GeometricField.C:922-939) hands the old-time field the parent's writeOpt only `if
// (field0Ptr_->field0Ptr_)` -- only once the old-time field has an old level of ITS own -- and only
// CrankNicolson ever asks for oldTime().oldTime(). So U_0, phi_0, k_0 and epsilon_0/omega_0 are written
// under this scheme and under no other. MEASURED on the gate's own fixture: those four and no more.
// rho_0 and alpha.water_0 are never written -- interFoam's rho is not AUTO_WRITE and alpha1 has one
// level -- so rho and alpha restart cold in OpenFOAM too.
// The CELLS alone, for the closure's two scalars: brae keeps their old-time snapshot as a cell vector and
// nothing reads a patch value off it, so the file's boundary is not carried where it could only go stale.
bool readCnOldOld(
    const InterCnRestart& r,
    const std::string& name,
    std::size_t nCells,
    std::vector<scalar>& cells);
bool readCnOldOld(
    const InterCnRestart& r,
    const std::string& name,
    std::size_t nCells,
    const std::vector<FvPatch>& patches,
    std::vector<vector>& cells,
    std::vector<std::vector<vector>>& patchValues);
// ...and a surfaceScalarField's (phi_0), whose "cells" are the internal faces
bool readCnOldOldSurface(
    const InterCnRestart& r,
    const std::string& name,
    label nInternalFaces,
    const std::vector<FvPatch>& patches,
    SurfaceScalarField& out);

// ...and the device twin, which holds the same numbers per component and whose boundary array is
// flattened over the NON-COUPLED patches only, as every other device boundary array in this solver is
// (see the note in inter_driver_device.cu where U's patch values are flattened).
// `nBoundary` is the size the CONSUMER will ask lookupOrCreate for, and it is 0 for an fvm::ddt field:
// deviceCnFvmDdt reads only the cells, so it sizes the boundary at nothing and would reject a field
// that carried one. The file's patch values are then dropped rather than kept where nothing reads them.
bool seedCnDdt0(
    DeviceCnDdt0& ddt0,
    const InterCnRestart& r,
    int nComp,
    std::size_t nInternal,
    std::size_t nBoundary,
    const std::vector<FvPatch>& patches);

}   // namespace brae
