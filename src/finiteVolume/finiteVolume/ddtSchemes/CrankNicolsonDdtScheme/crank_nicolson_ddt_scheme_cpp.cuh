#pragma once
// OpenFOAM's CrankNicolson ddt scheme, the host reference: fvm::ddt(rho, vf) and fvc::ddtCorr(U, phi)
// on a mesh that does not move, with a CONSTANT off-centring coefficient.
//
// provenance:
//   openfoam: src/finiteVolume/finiteVolume/ddtSchemes/CrankNicolsonDdtScheme/CrankNicolsonDdtScheme.C
//               DDt0Field (:44-70), ddt0_ (:93-160), evaluate/coef_/coef0_/rDtCoef_/rDtCoef0_/offCentre_
//               (:163-256), fvmDdt(rho, vf) (:1000-1087), fvcDdtPhiCorr(U, phi) (:1262-1322)
//             src/finiteVolume/finiteVolume/ddtSchemes/ddtScheme/ddtScheme.C:128-200, :308-330
//               (fvcDdtPhiCoeff, the two-argument form CrankNicolson calls)
//             src/finiteVolume/interpolation/surfaceInterpolation/surfaceInterpolationScheme/
//               surfaceInterpolationScheme.C:220-305 (dotInterpolate)
//   tests:    tests/interfoam_cn_vs_openfoam.sh -- RAS/damBreak with `default CrankNicolson 0.5`
//
// THE SCHEME IS A STATE, not a formula. Every equation that takes fvm::ddt under CrankNicolson keeps a
// field OpenFOAM calls ddt0 -- the previous step's time derivative -- on the object registry, named
// after the operands ("ddt0(rho,U)", "ddt0(rho,k)", "ddtCorrDdt0(U)", "ddtCorrDdt0(phi)"), created
// zero at the first assembly and advanced ONCE PER TIME STEP by the first assembly of that step
// (evaluate(): the field's time index against the run's). The three coefficients follow the time
// index too:
//     coef  = 1 + oc   once timeIndex > startTimeIndex        -- Euler on the step the field is born
//     coef0 = 1 + oc   once timeIndex > startTimeIndex + 1    -- the first ddt0 is an Euler estimate
//     offCentre(x) = oc*x when oc < 1, x itself at oc = 1
// so a cold start runs Euler for one step, an Euler-estimated ddt0 for the next, and CrankNicolson
// proper from the third. Two equations born on different steps keep different start indices; one
// assembled twice in a step (nOuterCorrectors > 1) advances its ddt0 once. Both are why the state
// lives here, per field, and not in the caller.
//
// THE ARITHMETIC is OpenFOAM's, term for term, on a static mesh:
//     fvm.diag()   =  (rDtCoef*rho)*V
//     ddt0        <-  rDtCoef0*(rhoOld*vfOld - rhoOO*vfOO) - offCentre(ddt0)      (when evaluated)
//     fvm.source() = ((rDtCoef*rhoOld)*vfOld + offCentre(ddt0))*V
// with rDtCoef = coef/deltaT and rDtCoef0 = coef0/deltaT0 as single divisions. The k and epsilon
// equations of an incompressible closure call fvm::ddt(vf), whose form is the same with rho a
// geometricOneField; multiplying by exactly 1 changes no bit, so one implementation serves both, with
// the density pointers null.
//
// A MOVING MESH is fvmDdt's V0/V00 arguments (the moving branch weights ddt0 by them), fvcDdtUfCorr and
// meshPhi below; a restart is inter_cn_restart.cuh's seed (startTimeIndex -2). NOT HERE, refused by the
// callers by name: an `ocCoeff` that is a Function1 of time, and fvc::ddt.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "ldu_matrix.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace fv {

// mesh().time() as the scheme reads it: Time::timeIndex(), deltaT(), deltaT0()
struct CrankNicolsonClock
{
    label  timeIndex = 0;   // the step being taken; Time::operator++ has already run
    scalar deltaT = 0;      // this step's
    scalar deltaT0 = 0;     // the previous step's (Time::deltaT0_, deltaTSave_ before the increment)
    scalar ocCoeff = 1;     // the coefficient after the scheme's name, in [0, 1]
};

// DDt0Field<GeoField>: the previous step's ddt of one operand set, with its two time indices. T is
// scalar or vector; a surface field's "boundary" is its patch faces as a vol field's is.
template <typename T>
struct CrankNicolsonDdt0
{
    std::string name;                         // OpenFOAM's registry name, for messages
    std::vector<T> internal;
    std::vector<std::vector<T>> boundary;
    label startTimeIndex = 0;
    label timeIndex = 0;
    bool exists = false;

    // ddt0_(): the registry lookup that CREATES the field, zero, at the clock's index, the first time
    void lookupOrCreate(
        const CrankNicolsonClock& clock,
        std::size_t nInternal,
        const std::vector<std::size_t>& patchSizes);

    // evaluate(): true once per time index, and the field takes the index
    bool evaluate(const CrankNicolsonClock& clock)
    {
        const bool evaluated = (timeIndex != clock.timeIndex);
        timeIndex = clock.timeIndex;
        return evaluated;
    }
    scalar coef(const CrankNicolsonClock& clock) const
    {
        return (clock.timeIndex > startTimeIndex) ? scalar(1) + clock.ocCoeff : scalar(1);
    }
    scalar coef0(const CrankNicolsonClock& clock) const
    {
        return (clock.timeIndex > startTimeIndex + 1) ? scalar(1) + clock.ocCoeff : scalar(1);
    }
    scalar rDtCoef(const CrankNicolsonClock& clock) const { return coef(clock)/clock.deltaT; }
    scalar rDtCoef0(const CrankNicolsonClock& clock) const { return coef0(clock)/clock.deltaT0; }
};

// offCentre_(x): ocCoeff*x below 1, x itself at 1 (CrankNicolsonDdtScheme.C:245-256)
inline scalar offCentre(
    const CrankNicolsonClock& clock,
    scalar x)
{
    return (clock.ocCoeff < scalar(1)) ? clock.ocCoeff*x : x;
}
inline vector offCentre(
    const CrankNicolsonClock& clock,
    const vector& x)
{
    return (clock.ocCoeff < scalar(1)) ? vector{clock.ocCoeff*x.x, clock.ocCoeff*x.y, clock.ocCoeff*x.z} : x;
}

// THE PATCH HALF of fvm::ddt's ddt0: the operands' STORED patch values at the two old levels, one list
// per patch. ddt0 is a whole GeometricField and OpenFOAM advances its patches beside its cells
// (CrankNicolsonDdtScheme.C:1040-1047 moving, :1069-1073 static):
//     ddt0_b <- rDtCoef0*(rhoOld_b*vfOld_b - rhoOO_b*vfOO_b) - offCentre(ddt0_b)
// Nothing in the solve reads them -- fvm.source() takes the primitiveField -- but the field is written
// with them at every write time. `rhoOld`/`rhoOO` null together mean rho = 1.
template <typename T>
struct CrankNicolsonDdt0Operands
{
    const std::vector<std::vector<scalar>>* rhoOld = nullptr;
    const std::vector<std::vector<scalar>>* rhoOO = nullptr;
    const std::vector<std::vector<T>>* vfOld = nullptr;
    const std::vector<std::vector<T>>* vfOO = nullptr;
};

// fvm::ddt(rho, vf) on a static mesh, added INTO M's diagonal and source (the caller has assembled the
// other terms; the fvMatrix constructor's `+` adds this one before relax). `rho`, `rhoOld` and `rhoOO`
// null together mean fvm::ddt(vf), rho = 1. The ddt0 field is created here if it does not exist, and
// evaluated here if this is the step's first call -- as ddt0_() and evaluate() do inside OpenFOAM's
// fvmDdt. The cells of ddt0 are always kept; its patches when `patchOperands` is handed (the writer's
// callers), since fvm.source() reads its primitiveField and nothing in the solve reads its patches.
void fvmDdt(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<vector>& ddt0,
    const std::vector<scalar>* rho,
    const std::vector<scalar>* rhoOld,
    const std::vector<scalar>* rhoOO,
    const std::vector<vector>& vfOld,
    const std::vector<vector>& vfOO,
    const std::vector<scalar>& V,
    FvVectorMatrix& M,
    // a MOVING mesh: mesh().V0() and mesh().V00(), which the scheme's moving branch
    // weights ddt0 and the source by (CrankNicolsonDdtScheme.C:1029-1065). Null together
    // means the static branch.
    const std::vector<scalar>* V0 = nullptr,
    const std::vector<scalar>* V00 = nullptr,
    const CrankNicolsonDdt0Operands<vector>* patchOperands = nullptr);
void fvmDdt(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<scalar>& ddt0,
    const std::vector<scalar>* rho,
    const std::vector<scalar>* rhoOld,
    const std::vector<scalar>* rhoOO,
    const std::vector<scalar>& vfOld,
    const std::vector<scalar>& vfOO,
    const std::vector<scalar>& V,
    FvScalarMatrix& M,
    // a MOVING mesh: mesh().V0() and mesh().V00(), which the scheme's moving branch
    // weights ddt0 and the source by (CrankNicolsonDdtScheme.C:1029-1065). Null together
    // means the static branch.
    const std::vector<scalar>* V0 = nullptr,
    const std::vector<scalar>* V00 = nullptr,
    const CrankNicolsonDdt0Operands<scalar>* patchOperands = nullptr);

// fvc::ddtCorr(U, phi) on a static mesh (fvcDdtPhiCorr):
//     ddt0    <- rDtCoef0*(U.oldTime() - U.oldTime().oldTime()) - offCentre(ddt0)      cells AND patches
//     dphidt0 <- rDtCoef0*(phi.oldTime() - phi.oldTime().oldTime()) - offCentre(dphidt0)
//     out = fvcDdtPhiCoeff(U.oldTime(), phi.oldTime())
//          *((rDtCoef*phi.oldTime() + offCentre(dphidt0)) - (Sf & interpolate(rDtCoef*U.oldTime() + offCentre(ddt0))))
// The coefficient is the two-argument fvcDdtPhiCoeff: 1 - min(|phiCorr|/(|phi| + SMALL), 1) with
// phiCorr = phi.oldTime() - (Sf & interpolate(U.oldTime())), zero on every patch whose U fixes a value
// and on every cyclicAMI patch, or the case's constant `ddtPhiCoeff` when that is not negative. The
// interpolations are surfaceInterpolationScheme::dotInterpolate's, Sf & (lambda*(P - N) + N) on an
// internal face, the patch value on an uncoupled boundary face, lerp(N, P, lambda) on a coupled one.
// `UOldBnd`/`UOOBnd` are U.oldTime()'s and U.oldTime().oldTime()'s STORED patch values.
void fvcDdtPhiCorr(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<vector>& ddt0,
    CrankNicolsonDdt0<scalar>& dphidt0,
    const std::vector<vector>& UOld,
    const std::vector<vector>& UOO,
    const std::vector<std::vector<vector>>& UOldBnd,
    const std::vector<std::vector<vector>>& UOOBnd,
    const SurfaceScalarField& phiOld,
    const SurfaceScalarField& phiOO,
    const std::vector<bool>& patchFixesU,
    scalar ddtPhiCoeff,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    SurfaceScalarField& out);

// fvc::ddtCorr(U, Uf) on a MOVING mesh (fvcDdtUfCorr, CrankNicolsonDdtScheme.C:1201-1257). A moving
// mesh carries Uf -- the face velocity the solver stores and moves with the mesh -- and OpenFOAM's
// correction is then built from Uf.oldTime() rather than from phi.oldTime():
//
//     ddt0   <- rDtCoef0*(U.oldTime()  - U.oldTime().oldTime())  - offCentre(ddt0)     cells AND patches
//     dUfdt0 <- rDtCoef0*(Uf.oldTime() - Uf.oldTime().oldTime()) - offCentre(dUfdt0)   a SURFACE vector
//     out = fvcDdtPhiCoeff(U.oldTime(), Sf & Uf.oldTime())
//          *(Sf & ((rDtCoef*Uf.oldTime() + offCentre(dUfdt0)) - interpolate(rDtCoef*U.oldTime() + offCentre(ddt0))))
//
// THREE DIFFERENCES from the static twin above, each of which changes the answer on a moving mesh:
//   * the flux side is `Sf & Uf.oldTime()`, not phi.oldTime() -- on a moving mesh those are not the
//     same number, because phi carries the mesh flux and Uf does not;
//   * the second ddt0 is a SURFACE VECTOR field (ddtCorrDdt0(Uf)), so it needs Uf.oldTime().oldTime();
//   * the interpolation is subtracted as a VECTOR and dotted with Sf once, rather than each term being
//     dotted separately -- the same arithmetic, in OpenFOAM's order.
//
// `UfOld`/`UfOO` are surface VECTOR fields: internal faces and patch faces, as Uf is stored.
void fvcDdtUfCorr(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<vector>& ddt0,
    CrankNicolsonDdt0<vector>& dUfdt0,
    const std::vector<vector>& UOld,
    const std::vector<vector>& UOO,
    const std::vector<std::vector<vector>>& UOldBnd,
    const std::vector<std::vector<vector>>& UOOBnd,
    const SurfaceVectorField& UfOld,
    const SurfaceVectorField& UfOO,
    const std::vector<bool>& patchFixesU,
    scalar ddtPhiCoeff,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    SurfaceScalarField& out);

// fvc::meshPhi(U) under CrankNicolson (CrankNicolsonDdtScheme.C:1626-1661). THE MESH FLUX IS
// OFF-CENTRED TOO, and this is the half of the scheme a moving mesh reaches without any ddt term of
// its own: every fvc::makeRelative / makeAbsolute and every movingWallVelocity patch asks
// fvc::meshPhi for the flux, fvc::meshPhi asks the ddt scheme named for `ddt(U)`, and Euler answers
// with mesh().phi() itself while CrankNicolson answers with
//
//     meshPhi0 <- coef0*mesh().phi().oldTime() - offCentre(meshPhi0)      (once per time step)
//     out       = coef *mesh().phi()           - offCentre(meshPhi0)
//
// The field is `meshPhiCN_0` on OpenFOAM's registry, born zero at the first move, so the first step
// returns mesh().phi() unchanged (coef = 1) and the off-centring appears from the second -- which is
// exactly where sloshing2DCN left OpenFOAM: step one 2.7e-14, step two 4.6e-04 with the raw mesh flux
// in its place, and the moving walls 5.1e-02 out because Uwall's normal component IS this flux.
//
// `phiOld` is mesh().phi().oldTime(): the flux of the PREVIOUS move, which fvMesh::movePoints grabs
// before overwriting phi (fvMesh.C:971-978) and only when the time index has advanced.
void meshPhi(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<scalar>& meshPhi0,
    const SurfaceScalarField& phi,
    const SurfaceScalarField& phiOld,
    SurfaceScalarField& out);

// The coefficient after `CrankNicolson` in an fvSchemes entry: a bare number in [0, 1] is the constant
// Function1 (CrankNicolsonDdtScheme.C:288-303); a dictionary is a Function1 of time and is refused by
// name; nothing at all is OpenFOAM's default of 1.
scalar readOcCoeff(const std::string& entry);

}   // namespace fv
}   // namespace cpu
}   // namespace brae
