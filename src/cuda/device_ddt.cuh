#pragma once
// Transient time-derivative fvm::ddt(rho, psi) for the PIMPLE / rho-transient solvers -- the ONE numerical module SIMPLE
// (steady) does not have. Adds the IMPLICIT ddt contribution to an already-assembled fvMatrix's diagonal + source, so a
// transient UEqn is  ddt + div(phi,U) + divDevReff == fvOptions  exactly as OpenFOAM assembles it (ddt added BEFORE
// UEqn.relax()). Reproduces OpenFOAM-2412 byte-for-byte on a fixed (non-moving) mesh, where Vsc == Vsc0 == V:
//
//   diag[c]   += coefft *rDeltaT*rho*V[c]
//   source[c] += rDeltaT*rho*V[c]*( coefft0*psi_old[c] - coefft00*psi_old2[c] )
//
// Euler (1st order)   : coefft=1,  coefft0=1,               coefft00=0            -> (rho/dt)(V) , source (rho/dt)(V) psi_old
// backward (2nd order): coefft=1+dt/(dt+dt0), coefft00=dt^2/(dt0(dt+dt0)), coefft0=coefft+coefft00
//
// OF source refs (/home/ghost/cudafoam/src/finiteVolume/finiteVolume/ddtSchemes):
//   EulerDdtScheme.C:381-391   fvmDdt(vf)        diag = rDeltaT*Vsc ; source = rDeltaT*vf.oldTime()*Vsc   (non-moving)
//   EulerDdtScheme.C:416-428   fvmDdt(rho,vf)    diag = rDeltaT*rho*Vsc ; source = rDeltaT*rho*vf.oldTime()*Vsc
//   backwardDdtScheme.C:26     diag = (coefft*rDeltaT)*V
//   backwardDdtScheme.C:39-43  source = rDeltaT*V*(coefft0*oldTime - coefft00*oldTime.oldTime())   (non-moving)
//   backwardDdtScheme.C:96-98  coefft / coefft00 / coefft0
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"   // DeviceMesh: deviceSurfaceSumMagPhi needs the owner/neighbour addressing

namespace brae {

// fvSchemes ddtSchemes.default. steadyState == SIMPLE (no ddt); Euler/backward/CrankNicolson == transient.
enum class DdtScheme { steadyState = 0, Euler = 1, backward = 2, CrankNicolson = 3 };

// The time-scheme weights, computed ONCE per time step (host side) exactly as OF does. `active == false` (steadyState,
// or before the first advance) makes deviceFvmDdt a no-op, so the same code path serves SIMPLE unchanged.
// CrankNicolson replaces the psi.old2 source term (-coefft00*psi_old2) with an auxiliary stored-old-ddt term
// (+ocCoeff*ddt0), where ddt0 is carried per transported variable and updated each step (OF CrankNicolsonDdtScheme).
struct DdtCoeffs
{
    scalar rDeltaT  = 0;      // 1/deltaT
    scalar coefft   = 1;      // implicit diagonal weight (backward: 1+dt/(dt+dt0); CrankNicolson: 1+ocCoeff)
    scalar coefft0  = 1;      // psi.oldTime()          source weight (CrankNicolson: 1+ocCoeff)
    scalar coefft00 = 0;      // psi.oldTime().oldTime() source weight (backward only)
    scalar ocCoeff  = 0;      // CrankNicolson off-centring: ddt0 source weight + ddt0 recurrence blend (0=Euler, 1=pure CN)
    scalar rDeltaT0 = 0;      // 1/deltaT0, for the ddt0 recurrence (CrankNicolson)
    scalar coefft0dd = 0;     // CrankNicolson ddt0-recurrence coef (OF coef0_): 1 on the FIRST ddt0 update (Euler ddt0), 1+oc after
    bool   cn       = false;  // CrankNicolson: source uses +ocCoeff*ddt0 instead of -coefft00*psi.old2
    bool   active   = false;
};

// Build the coefficients exactly as OF (backwardDdtScheme.C:96-98; CrankNicolsonDdtScheme rDtCoef/offCentre). First
// transient step (deltaT0<=0) falls back to Euler, matching OF (oldTime().oldTime()/ddt0 are uninitialised until two
// levels exist -> pimpleFoam bootstraps with Euler). ocCoeff is the CrankNicolson off-centring (fvSchemes "CrankNicolson <oc>").
// cnWarm=false on the FIRST ddt0 update (second overall step) -> Euler ddt0 (OF coef0_=1); true afterwards (OF coef0_=1+oc).
// This Euler-startup of the ddt0 recurrence is what makes CrankNicolson genuinely 2nd order (else the first ddt0 is 2x too
// large and, at oc=1, oscillates without decaying -> 1st order).
inline DdtCoeffs ddtCoeffs(DdtScheme scheme, scalar deltaT, scalar deltaT0, scalar ocCoeff = scalar(1), bool cnWarm = false)
{
    DdtCoeffs c;
    if (scheme == DdtScheme::steadyState || deltaT <= scalar(0))
        return c;                                   // active stays false -> no-op
    c.active  = true;
    c.rDeltaT = scalar(1) / deltaT;
    if (scheme == DdtScheme::backward && deltaT0 > scalar(0))
    {
        c.coefft   = scalar(1) + deltaT / (deltaT + deltaT0);
        c.coefft00 = deltaT * deltaT / (deltaT0 * (deltaT + deltaT0));
        c.coefft0  = c.coefft + c.coefft00;
    }
    else if (scheme == DdtScheme::CrankNicolson && deltaT0 > scalar(0))
    {
        c.cn        = true;
        c.ocCoeff   = ocCoeff;
        c.rDeltaT0  = scalar(1) / deltaT0;
        c.coefft    = scalar(1) + ocCoeff;          // diag = (1+oc)*rDeltaT*V   (OF coef_)
        c.coefft0   = scalar(1) + ocCoeff;          // source psi.old weight = (1+oc)*rDeltaT
        c.coefft0dd = cnWarm ? (scalar(1) + ocCoeff) : scalar(1);   // ddt0 recurrence coef (OF coef0_): Euler on first update
    }
    // else Euler: coefft=1, coefft0=1, coefft00=0, cn=false (struct defaults). CrankNicolson first step lands here.
    return c;
}

// Add the implicit ddt of ONE scalar component (a velocity component, or a transported scalar) to (diag, source).
// diag/source are the assembled fvMatrix arrays (size == nCells); this ACCUMULATES, matching ddt being one UEqn term.
// psiOld2 may be empty (Euler / bootstrap). rho is the constant density (1 for incompressible pimpleFoam). No-op when
// c.active is false. V is the cell-volume field.
//
// MOVING MESH. OF's EulerDdtScheme::fvmDdt (EulerDdtScheme.C:383-388) is
//     fvm.diag()   = rDeltaT*mesh().Vsc();                          // CURRENT volume
//     fvm.source() = rDeltaT*vf.oldTime()*(moving ? Vsc0() : Vsc()); // OLD volume when moving
// so the discrete term is (V^n*psi^n - V^{n-1}*psi^{n-1})/dt, NOT V*(psi^n - psi^{n-1})/dt. That
// difference IS the discrete space conservation law: with phi made relative, the two together keep
// continuity exact on a moving mesh. Using one volume for both leaves a continuity error that grows
// every step -- measured on pimpleFoam/RAS/rotatingFanInRoom as contLocal 8.1e-03 -> 8.0e-02 while the
// Courant number sat at 0.46, i.e. a divergence that is not a CFL problem.
//
// The split needs no new kernel: deviceFvmDdtDiag and deviceFvmDdtSource already take V separately, so
// a moving-mesh caller passes the CURRENT volume to Diag and the OLD volume to Source. On a fixed mesh
// the two buffers are the same and every existing call is unchanged.
//
// Use deviceFvmDdt when diag + source belong to the SAME matrix (a scalar transport, e.g. k/epsilon).
void deviceFvmDdt(
    const DeviceBuffer<scalar>& V,
    const DdtCoeffs&            c,
    scalar                      rho,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>&       diag,
    DeviceBuffer<scalar>&       source,
    const DeviceBuffer<scalar>* ddt0 = nullptr);   // CrankNicolson stored old ddt (else nullptr)

// Per-scalar transient-ddt bundle for a transported turbulence scalar (k / epsilon / omega / nuTilda). Threaded into
// deviceSolveScalarTransport so a transient (PIMPLE) turbulence solve folds fvm::ddt(f) into its matrix, while the steady
// (SIMPLE) path passes the default (c.active==false, old==nullptr) -> an exact no-op. `old`/`old2` are that scalar's
// oldTime()[.oldTime()] device fields (owned by the solver, rotated each time step); old2 may be null (Euler/bootstrap).
struct ScalarDdt
{
    DdtCoeffs                   c{};
    const DeviceBuffer<scalar>* old  = nullptr;
    const DeviceBuffer<scalar>* old2 = nullptr;
    const DeviceBuffer<scalar>* ddt0 = nullptr;   // CrankNicolson stored old ddt (that scalar's ddt0_ member); null otherwise
};

// The two halves of deviceFvmDdt, split for the vector momentum predictor where the ddt DIAGONAL is shared across the 3
// components (added once to the assembled diag, BEFORE relax) but the ddt SOURCE is per component (uses that component's
// oldTime). deviceFvmDdt(...) == Diag(...) then Source(...). Both no-op when !c.active.
//   diag[c]   += coefft*rDeltaT*rho*V[c]
void deviceFvmDdtDiag(
    const DeviceBuffer<scalar>& V,
    const DdtCoeffs&            c,
    scalar                      rho,
    DeviceBuffer<scalar>&       diag);
//   backward/Euler : source[c] += rDeltaT*rho*V[c]*( coefft0*psiOld[c] - coefft00*psiOld2[c] )
//   CrankNicolson  : source[c] += rho*V[c]*( coefft0*rDeltaT*psiOld[c] + ocCoeff*ddt0[c] )   (ddt0 non-null when c.cn)
void deviceFvmDdtSource(
    const DeviceBuffer<scalar>& V,
    const DdtCoeffs&            c,
    scalar                      rho,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>&       source,
    const DeviceBuffer<scalar>* ddt0 = nullptr);   // CrankNicolson stored old ddt (else nullptr)

// Euler fvm::ddt(rho, vf) and fvc::ddt(rho, vf) with a per-cell rho on a static mesh, in OpenFOAM's arithmetic
// (EulerDdtScheme.C:449-467 and :256; fvMatrix.C's operator- of a matrix and a field). For the corrector form
// `fvm::ddt(rho, vf) - fvc::ddt(rho, vf)` (rhoCentralFoam.C:238, 275) call both, in that order. One component (or
// one scalar field) a call; every product is rounded on its own, as OpenFOAM's field algebra stores it.
//   deviceEulerFvmDdtRho:    diag[c] = rDeltaT*rho[c]*V[c];  source[c] = rDeltaT*rho0[c]*vf0[c]*V[c]   (assigned)
//   deviceEulerFvcDdtRhoSub: source[c] += V[c]*(rDeltaT*(rho[c]*vf[c] - rho0[c]*vf0[c]))  (the matrix minus fvc::ddt)
void deviceEulerFvmDdtRho(scalar rDeltaT, const DeviceBuffer<scalar>& rho, const DeviceBuffer<scalar>& rho0,
                          const DeviceBuffer<scalar>& vf0, const DeviceBuffer<scalar>& V,
                          DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source);
void deviceEulerFvcDdtRhoSub(scalar rDeltaT, const DeviceBuffer<scalar>& rho, const DeviceBuffer<scalar>& vf,
                             const DeviceBuffer<scalar>& rho0, const DeviceBuffer<scalar>& vf0,
                             const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& source);

// CrankNicolson ddt0 recurrence (OF DDt0Field update), run ONCE per time step after the old-time rotation:
//   ddt0[c] = coefft*rDeltaT0*( psiOld[c] - psiOld2[c] ) - ocCoeff*ddt0[c]     (in place; uses the OLD ddt0)
// No-op unless c.cn (steady/Euler/backward carry no ddt0). psiOld2 must hold two old levels (skipped on the first step).
void deviceFvmDdtUpdateDdt0(
    const DdtCoeffs&            c,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>&       ddt0);

// fvc::ddtCorr(U, phi) -- the Rhie-Chow-style ddt flux correction pimpleFoam adds to phiHbyA:
//
//     phiHbyA += fvc::interpolate(rAU)*fvc::ddtCorr(U, phi, Uf)          (pEqn.H)
//
// EulerDdtScheme::fvcDdtPhiCorr, verified term for term against OpenFOAM v2412 on
// pimpleFoam/RAS/oscillatingInletACMI2D (21464 internal faces, agreement 1.2e-10 L2):
//
//     phiCorr = phi_old - (interpolate(U_old) & Sf)
//     coeff   = 1 - min(|phiCorr| / (|phi_old| + SMALL), 1)        <- ddtScheme::fvcDdtPhiCoeff
//     ddtCorr = coeff * (1/deltaT) * phiCorr
//
// The coefficient is what makes it a CORRECTION rather than a second convection term: it is 1 where the
// stored flux already agrees with the interpolated velocity and falls to 0 where they disagree by as
// much as the flux itself, so a face whose flux is pure Rhie-Chow contributes nothing. On that case it
// ran between 0.875 and 1.0.
//
// `coeffMask` is per BOUNDARY face: OF zeroes the coefficient wherever U fixesValue() or the patch is a
// cyclicAMI (ddtScheme.C, fvcDdtPhiCoeff), so pass 0 there and 1 elsewhere. Coupled-interface faces are
// not in this array at all and get no correction, which is the same thing OF's cyclicAMI branch does.
//
// out{Int,Bnd} are ACCUMULATED into (+=), so this adds straight onto phiHbyA.
void deviceDdtCorrFlux(
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& phiOldInt,   // phi.oldTime(), internal faces
    const DeviceBuffer<scalar>& phiOldBnd,   // phi.oldTime(), boundary faces (DeviceBoundary order)
    const DeviceBuffer<scalar>& fluxUoldInt, // (interpolate(U.oldTime()) & Sf), internal
    const DeviceBuffer<scalar>& fluxUoldBnd, // ... boundary
    const DeviceBuffer<scalar>& rAUf,        // fvc::interpolate(rAU), internal faces
    const DeviceBuffer<scalar>& rAUbnd,      // rAU on the boundary faces (adjacent-cell value)
    const DeviceBuffer<scalar>& coeffMask,   // per boundary face: 0 at fixesValue/coupled, else 1
    scalar                      rDeltaT,
    DeviceBuffer<scalar>&       outInt,      // += the correction
    DeviceBuffer<scalar>&       outBnd,
    const DeviceBuffer<scalar>* phiCorrEffInt = nullptr,   // backward: the two-level correction
    const DeviceBuffer<scalar>* phiCorrEffBnd = nullptr);

// fvc::correctUf (fvcMeshPhi.C) -- the face velocity a moving-mesh pimpleFoam carries alongside phi:
//
//     Uf  = fvc::interpolate(U);
//     n   = Sf/magSf;
//     Uf += n*(phi/magSf - (n & Uf));
//
// i.e. the interpolated cell velocity with its NORMAL component replaced by the one the conservative flux
// implies. `phi` must be the ABSOLUTE flux -- OF calls this at the end of pEqn.H, before makeRelative.
//
// `faceIdx` maps entry i to its index in Sf/magSf: pass the boundary gather (dm.bndGFace) for boundary
// faces, and nullptr when the arrays are already in the caller's own order (internal faces, or an
// interface whose Sf is stored per interface face).
void deviceCorrectUf(
    int n, const label* faceIdx,
    const DeviceBuffer<scalar>& Sfx, const DeviceBuffer<scalar>& Sfy, const DeviceBuffer<scalar>& Sfz,
    const DeviceBuffer<scalar>& magSf, const DeviceBuffer<scalar>& phi,
    DeviceBuffer<scalar>& ufx, DeviceBuffer<scalar>& ufy, DeviceBuffer<scalar>& ufz);   // in: interp(U), out: Uf

// Sf & Uf, per face. This is EulerDdtScheme::fvcDdtUfCorr's phiUf0 -- the CURRENT Sf dotted into the
// STORED face velocity, which is what makes the Uf form exact on a mesh whose areas change (a rotating
// mesh, or a cyclicACMI whose overlap mask rescales its coupled areas every step) where phi + mesh.phi()
// is only an approximation of it.
// fvc::surfaceSum(mag(phi)) per cell, coupled interfaces included. Pass nullptr for an absent interface.
void deviceSurfaceSumMagPhi(const DeviceMesh& dm,
                            const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
                            const DeviceBuffer<label>* cycOwn, const DeviceBuffer<scalar>* cycPhi,
                            const DeviceBuffer<label>* amiOwn, const DeviceBuffer<scalar>* amiPhi,
                            DeviceBuffer<scalar>& out);

void deviceDotSf(
    int n, const label* faceIdx,
    const DeviceBuffer<scalar>& Sfx, const DeviceBuffer<scalar>& Sfy, const DeviceBuffer<scalar>& Sfz,
    const DeviceBuffer<scalar>& ufx, const DeviceBuffer<scalar>& ufy, const DeviceBuffer<scalar>& ufz,
    DeviceBuffer<scalar>& out);

}  // namespace brae
