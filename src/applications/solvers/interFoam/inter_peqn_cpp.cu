// interFoam's pressure corrector -- see inter_peqn_cpp.cuh for the provenance and for the four things
// in it that are interFoam's own.
#include "inter_peqn_cpp.cuh"
#include "fvc_reconstruct_cpp.cuh"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "pbicgstab.cuh"
#include "pcg.cuh"

namespace brae {
namespace cpu {
namespace interFoam {

void buoyancyFlux(const std::vector<scalar>& surfaceTensionForce,
                  const std::vector<scalar>& ghf,
                  const std::vector<scalar>& snGradRho,
                  const std::vector<scalar>& rAUf,
                  const std::vector<scalar>& magSf,
                  std::vector<scalar>&       phig)
{
    const std::size_t n = surfaceTensionForce.size();
    if (ghf.size() != n || snGradRho.size() != n || rAUf.size() != n || magSf.size() < n)
        throw std::runtime_error("brae interFoam pEqn: phig's face fields differ in length.");
    phig.resize(n);
    for (std::size_t f = 0; f < n; ++f)
        phig[f] = (surfaceTensionForce[f] - ghf[f]*snGradRho[f]) * rAUf[f] * magSf[f];
}


void rhoRAUf(const std::vector<scalar>& rho,
             const std::vector<scalar>& rAU,
             const PrimitiveMesh&       m,
             const FvGeometry&          g,
             std::vector<scalar>&       out)
{
    if (rho.size() != rAU.size())
        throw std::runtime_error("brae interFoam pEqn: rho and rAU differ in length.");
    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<scalar>& w   = g.weights();

    // THE PRODUCT IS FORMED PER CELL AND INTERPOLATED ONCE. Not interpolate(rho)*interpolate(rAU):
    // linear interpolation does not commute with multiplication, and the gap is largest where the two
    // factors vary most -- which across a VoF interface is a factor of 1000 in one face.
    out.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const scalar po = rho[own[f]] * rAU[own[f]];
        const scalar pn = rho[nei[f]] * rAU[nei[f]];
        out[f] = w[f]*po + (scalar(1) - w[f])*pn;
    }
}


void correctVelocity(const std::vector<vector>&              HbyA,
                     const std::vector<scalar>&              rAU,
                     const std::vector<scalar>&              faceFlux,
                     const std::vector<scalar>&              rAUf,
                     const std::vector<std::vector<scalar>>& faceFluxBnd,
                     const std::vector<std::vector<scalar>>& rAUfBnd,
                     const PrimitiveMesh&                    m,
                     const FvGeometry&                       g,
                     const std::vector<FvPatch>&             patches,
                     std::vector<vector>&                    U)
{
    const label nIf = m.nInternalFaces();
    if (faceFlux.size() != static_cast<std::size_t>(nIf) || rAUf.size() != static_cast<std::size_t>(nIf))
        throw std::runtime_error("brae interFoam pEqn: the correction's face fields differ in length.");

    // (phig - p_rghEqn.flux())/rAUf, face by face, BEFORE the reconstruction. The division and the
    // later multiplication by rAU do not cancel on a non-uniform rAU -- see note 3.
    SurfaceScalarField ssf;
    ssf.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f) ssf.internal[f] = faceFlux[f] / rAUf[f];
    ssf.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::size_t np = static_cast<std::size_t>(patches[pi].size);
        ssf.boundary[pi].resize(np);
        for (std::size_t i = 0; i < np; ++i)
        {
            const scalar ff = (pi < faceFluxBnd.size() && i < faceFluxBnd[pi].size())
                            ? faceFluxBnd[pi][i] : scalar(0);
            const scalar rf = (pi < rAUfBnd.size() && i < rAUfBnd[pi].size())
                            ? rAUfBnd[pi][i] : scalar(1);
            ssf.boundary[pi][i] = ff / rf;
        }
    }

    using namespace cpu::fvcReconstruct;
    const label nC = m.nCells();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<vector>& Sf  = g.Sf();

    std::vector<tensor> T(static_cast<std::size_t>(nC), tensor{0,0,0,0,0,0,0,0,0});
    std::vector<vector> v(static_cast<std::size_t>(nC), vector{0,0,0});
    for (label f = 0; f < nIf; ++f)
    {
        accumulate(Sf[f], ssf.internal[f], T[own[f]], v[own[f]]);
        accumulate(Sf[f], ssf.internal[f], T[nei[f]], v[nei[f]]);
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        // EMPTY PATCHES ARE NOT IN surfaceSum -- emptyFvPatch::size() is 0 in OpenFOAM. This loop did
        // NOT skip them, and that was hiding a second defect: with the empty faces included the
        // tensor is invertible on a 2-D mesh, so the plain cofactor inverse below never divided by
        // zero. It is safeInv's job to handle the singular direction, not this loop's to avoid it.
        if (q.type == "empty") continue;
        for (label i = 0; i < q.size; ++i)
        {
            const label ci = own[q.start + i];
            accumulate(Sf[q.start + i], ssf.boundary[pi][i], T[ci], v[ci]);
        }
    }

    U.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        // safeInv, as OpenFOAM's inv(Field<tensor>) is (tensorField.C:55) -- see fvc_reconstruct_cpp.cuh
        const vector r = dot(safeInv(T[c]), v[c]);
        U[c] = vector{HbyA[c].x + rAU[c]*r.x,
                      HbyA[c].y + rAU[c]*r.y,
                      HbyA[c].z + rAU[c]*r.z};
    }
}


void staticPressure(const std::vector<scalar>& p_rgh,
                    const std::vector<scalar>& rho,
                    const std::vector<scalar>& gh,
                    std::vector<scalar>&       p)
{
    if (rho.size() != p_rgh.size() || gh.size() != p_rgh.size())
        throw std::runtime_error("brae interFoam pEqn: p_rgh, rho and gh differ in length.");
    p.resize(p_rgh.size());
    for (std::size_t c = 0; c < p_rgh.size(); ++c) p[c] = p_rgh[c] + rho[c]*gh[c];
}


void applyPressureReference(std::vector<scalar>&       p,
                            std::vector<scalar>&       p_rgh,
                            const std::vector<scalar>& rho,
                            const std::vector<scalar>& gh,
                            label                      pRefCell,
                            scalar                     pRefValue)
{
    if (pRefCell < 0 || static_cast<std::size_t>(pRefCell) >= p.size())
        throw std::runtime_error(
            "brae interFoam pEqn: pRefCell is outside the mesh. A p_rgh with no value-fixing patch "
            "needs a reference cell, and pEqn.H:74-83 reads p there.");
    const scalar shift = pRefValue - p[static_cast<std::size_t>(pRefCell)];
    for (scalar& v : p) v += shift;
    // ...AND p_rgh IS REBUILT FROM THE SHIFTED p. It does not keep what the solve gave it -- note 4.
    for (std::size_t c = 0; c < p.size(); ++c) p_rgh[c] = p[c] - rho[c]*gh[c];
}


void ddtCorr(const DdtCorrInput&           in,
             const GeometricField<vector>& U,
             const PrimitiveMesh&          m,
             const FvGeometry&             g,
             const std::vector<FvPatch>&   patches,
             SurfaceScalarField&           out)
{
    if (!in.phiOld || !in.UOld)
        throw std::runtime_error(
            "brae interFoam ddtCorr: phi.oldTime() and U.oldTime() are both required -- the "
            "correction is the difference between them, at the OLD time (note 3).");
    if (in.deltaT <= scalar(0))
        throw std::runtime_error("brae interFoam ddtCorr: deltaT must be positive.");

    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<scalar>& w   = g.weights();
    const std::vector<vector>& Sf  = g.Sf();
    const scalar rDeltaT = scalar(1) / in.deltaT;
    const scalar kSmall  = scalar(1e-37);                 // OF SMALL

    out.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        // phiCorr = phi.oldTime() - (interpolate(U.oldTime()) & Sf)
        const vector& uo = (*in.UOld)[own[f]];
        const vector& un = (*in.UOld)[nei[f]];
        const vector uf{w[f]*uo.x + (scalar(1) - w[f])*un.x,
                        w[f]*uo.y + (scalar(1) - w[f])*un.y,
                        w[f]*uo.z + (scalar(1) - w[f])*un.z};
        const scalar interpFlux = uf.x*Sf[f].x + uf.y*Sf[f].y + uf.z*Sf[f].z;
        const scalar phiCorr    = in.phiOld->internal[f] - interpFlux;

        // note 1: a NEGATIVE ddtPhiCoeff selects the limiter, which is the default. It switches the
        // correction OFF where it is large compared with the flux -- the opposite of what a constant 1
        // would do.
        const scalar coeff = (in.ddtPhiCoeff < scalar(0))
            ? scalar(1) - std::fmin(std::fabs(phiCorr)
                                  / (std::fabs(in.phiOld->internal[f]) + kSmall), scalar(1))
            : in.ddtPhiCoeff;

        out.internal[f] = coeff * rDeltaT * phiCorr;
    }

    // note 2: zero on every patch where U fixes a value. brae has no cyclicAMI in a VoF case yet; the
    // loop is per patch so adding one changes only that patch.
    out.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        out.boundary[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(0));
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (U.boundary[pi]->fixesValue()) continue;       // already zero, and stays zero
        const FvPatch& q = patches[pi];
        for (label i = 0; i < q.size; ++i)
        {
            const vector& uo = (*in.UOld)[q.faceCells[i]];
            const vector& S  = Sf[q.start + i];
            const scalar interpFlux = uo.x*S.x + uo.y*S.y + uo.z*S.z;
            const scalar pOld = (pi < in.phiOld->boundary.size()
                                 && static_cast<std::size_t>(i) < in.phiOld->boundary[pi].size())
                              ? in.phiOld->boundary[pi][i] : scalar(0);
            const scalar phiCorr = pOld - interpFlux;
            const scalar coeff = (in.ddtPhiCoeff < scalar(0))
                ? scalar(1) - std::fmin(std::fabs(phiCorr)/(std::fabs(pOld) + kSmall), scalar(1))
                : in.ddtPhiCoeff;
            out.boundary[pi][i] = coeff * rDeltaT * phiCorr;
        }
    }
}


void updatePressurePatchesFromVelocity(
    GeometricField<scalar>& p_rgh,
    const GeometricField<vector>& U,
    const std::vector<std::vector<scalar>>* rhoBnd,
    const std::vector<FvPatch>& patches)
{
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // bcCategory 7 is totalPressure, the one p_rgh patch type here that reads the velocity
        if (p_rgh.boundary[pi]->bcCategory() != 7) continue;
        const bool haveRho = rhoBnd && pi < rhoBnd->size()
                          && (*rhoBnd)[pi].size() == static_cast<std::size_t>(patches[pi].size);
        if (!haveRho)
        {
            throw std::runtime_error(
                "brae interFoam pEqn: patch '" + patches[pi].name + "' is totalPressure and no rho "
                "patch values were supplied. p_rgh has the dimensions of pressure, so OpenFOAM's form "
                "is p0 - 0.5*rho*neg(phi)*|U|^2; defaulting rho to 1 would be the incompressible "
                "form and wrong by the density.");
        }
        p_rgh.boundary[pi]->updateFromPatchVelocity(U.boundary[pi]->value(), {}, (*rhoBnd)[pi]);
    }
}

void pressureCorrector(GeometricField<scalar>&      p_rgh,
                       GeometricField<vector>&      U,
                       SurfaceScalarField&          phi,
                       std::vector<scalar>&         p,
                       const PressureStepInput&     in,
                       const PressureSolveControls& sc,
                       const PrimitiveMesh&         m,
                       const FvGeometry&            g,
                       const std::vector<FvPatch>&  patches)
{
    if (!in.UEqn || !in.rho || !in.gh || !in.ghf || !in.stf || !in.snGradRho)
        throw std::runtime_error("brae interFoam pEqn: a required field is missing.");

    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();

    // rAU = 1/UEqn.A(), rAUf = interpolate(rAU).
    const std::vector<scalar> A = matrixA(*in.UEqn, m, g, patches);
    std::vector<scalar> rAU(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c) rAU[c] = scalar(1) / A[c];
    const SurfaceScalarField rAUfField = fvc::interpolate(rAU, m, g, patches);

    // HbyA = constrainHbyA(rAU*UEqn.H(), U, p_rgh).
    const std::vector<vector> H = matrixH(*in.UEqn, U, m, g, patches);
    std::vector<vector> HbyA(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
        HbyA[c] = vector{rAU[c]*H[c].x, rAU[c]*H[c].y, rAU[c]*H[c].z};

    // phiHbyA = fvc::flux(HbyA) + interpolate(rho*rAU)*ddtCorr(U, phi, Uf).
    // NOTE the weighting is interpolate(rho*rAU) -- the PRODUCT -- not interpolate(rho)*rAUf; see
    // note 2 in the header, and the 250x it is worth across the interface.
    // constrainHbyA(HbyA, U, p_rgh), constrainHbyA.C:
    //     if (!U.boundaryField()[patchi].assignable()) HbyAbf[patchi] = U.boundaryField()[patchi];
    // Everywhere else HbyA keeps the extrapolated value H() gives it, which is the owner cell's.
    // assignable() is NOT fixesValue(): slip and inletOutlet are non-assignable without fixing one,
    // and damBreak's atmosphere is pressureInletOutletVelocity. simpleFoam's pEqn_cpp carries the same
    // three lines, and they are the same three lines on purpose.
    std::vector<std::vector<vector>> HbyAb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        HbyAb[pi].resize(static_cast<std::size_t>(q.size));
        const bool takeU = !U.boundary[pi]->assignable();
        const std::vector<vector>& ub = U.boundary[pi]->value();
        for (label i = 0; i < q.size; ++i)
            HbyAb[pi][i] = takeU ? ub[i] : HbyA[q.faceCells[i]];
    }
    SurfaceScalarField phiHbyA = fvc::flux(HbyA, HbyAb, m, g, patches);

    if (in.ddt)
    {
        std::vector<scalar> rhoRAU;
        rhoRAUf(*in.rho, rAU, m, g, rhoRAU);
        SurfaceScalarField corr;
        ddtCorr(*in.ddt, U, m, g, patches, corr);
        for (label f = 0; f < nIf; ++f) phiHbyA.internal[f] += rhoRAU[f] * corr.internal[f];
    }

    // phig = (surfaceTensionForce - ghf*snGrad(rho)) * rAUf * magSf -- and NOT snGrad(p_rgh), which is
    // the laplacian below. phiHbyA += phig.
    std::vector<scalar> phig;
    buoyancyFlux(in.stf->internal, *in.ghf, in.snGradRho->internal, rAUfField.internal, g.magSf(), phig);
    if (in.taps)
    {
        in.taps->A = A;
        in.taps->rAU = rAU;
        in.taps->HbyA = HbyA;
        in.taps->rAUf = rAUfField.internal;
        in.taps->phig = phig;
        in.taps->phiHbyA = phiHbyA.internal;
        in.taps->stf = in.stf->internal;
        in.taps->snGradRho = in.snGradRho->internal;
    }
    for (label f = 0; f < nIf; ++f) phiHbyA.internal[f] += phig[f];

    // ...ON THE BOUNDARY TOO -- see PressureStepInput::ghfBnd. rAUf at an uncoupled patch is the face
    // cell's rAU, which is what fvc::interpolate gives there.
    std::vector<std::vector<scalar>> phigBnd(patches.size());
    if (in.ghfBnd)
    {
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            std::vector<scalar> rAUfb(static_cast<std::size_t>(q.size));
            for (label i = 0; i < q.size; ++i) rAUfb[i] = rAU[q.faceCells[i]];
            buoyancyFlux(in.stf->boundary[pi], (*in.ghfBnd)[pi], in.snGradRho->boundary[pi],
                         rAUfb, q.magSf, phigBnd[pi]);
            for (label i = 0; i < q.size; ++i) phiHbyA.boundary[pi][i] += phigBnd[pi][i];
        }
    }
    else
    {
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            phigBnd[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(0));
    }

    // constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF): a fixedFluxPressure patch's gradient is
    // PRESCRIBED from the flux, and brae refuses to assemble one that has not been set.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!p_rgh.boundary[pi]->updateableSnGrad()) continue;
        const FvPatch& q = patches[pi];
        std::vector<scalar> sn(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i)
        {
            const scalar rf = rAU[q.faceCells[i]];        // interpolate(rAU) at an uncoupled patch
            const scalar ph = (pi < phiHbyA.boundary.size()
                               && static_cast<std::size_t>(i) < phiHbyA.boundary[pi].size())
                            ? phiHbyA.boundary[pi][i] : scalar(0);
            const scalar Uf = (pi < phi.boundary.size()
                               && static_cast<std::size_t>(i) < phi.boundary[pi].size())
                            ? phi.boundary[pi][i] : scalar(0);
            sn[i] = (ph - Uf) / (q.magSf[i] * rf);
        }
        p_rgh.boundary[pi]->updateSnGrad(sn);
    }

    for (label corr = 0; corr <= sc.nNonOrthogonalCorrectors; ++corr)
    {
        // the fvMatrix constructor's updateCoeffs -- see updatePressurePatchesFromVelocity
        updatePressurePatchesFromVelocity(p_rgh, U, in.rhoBnd, patches);
        FvScalarMatrix pe = fvm::laplacian<scalar>(rAUfField, p_rgh, m, g, patches, /*corrected=*/false);
        const std::vector<scalar> div = fvc::div(phiHbyA, m, g, patches);
        for (label c = 0; c < nC; ++c) pe.source[c] += div[c] * g.V()[c];

        if (sc.needReference)
        {
            // fvMatrix::setReference: source += diag*refValue, diag += diag -- pinning one cell in a
            // system that is otherwise singular because every patch is zeroGradient.
            pe.source[sc.pRefCell] += pe.diag[sc.pRefCell] * sc.pRefValue;
            pe.diag[sc.pRefCell]   += pe.diag[sc.pRefCell];
        }

        // p_rgh.select(pimple.finalInnerIter()) (pEqn.H:50): the Final entry on the last
        // non-orthogonal pass of the last corrector, the plain entry everywhere else. And the case's
        // own solver where brae has OpenFOAM's -- the choice is not cosmetic. On damBreak the over-1
        // alpha excursion is dt x the div(phi) this solve leaves behind (interFoam's alphaSuSp.H has
        // no divU), and tightening ONLY p_rgh removed the whole of it on both codes, from 1.16e-06
        // to 4.9e-13; with PBiCGStab standing in for PCG+DIC brae's excursion ran 0.12x to 3.18x
        // OpenFOAM's from one step to the next.
        const bool finalInner = sc.finalCorrector && corr == sc.nNonOrthogonalCorrectors;
        const scalar tol = finalInner ? sc.tolPFinal : sc.tolP;
        const scalar relTol = finalInner ? sc.relTolPFinal : sc.relTolP;
        const int maxIter = finalInner ? sc.maxIterPFinal : sc.maxIterP;
        const SolverPerformance sp = (finalInner ? sc.pcgDICFinal : sc.pcgDIC)
            ? pcg(pe, p_rgh.internal, m, patches, tol, relTol, maxIter)
            : pbicgstab(pe, p_rgh.internal, m, patches, tol, relTol, maxIter);
        if (in.solveLog)
        {
            in.solveLog->push_back(PressureSolveRecord{sp.initialResidual, sp.finalResidual, sp.nIterations});
        }
        p_rgh.evaluateBoundary();

        if (corr == sc.nNonOrthogonalCorrectors)
        {
            const SurfaceScalarField pFlux = matrixFlux(pe, p_rgh.internal, m, patches);

            // phi = phiHbyA - p_rghEqn.flux()
            phi = phiHbyA;
            for (label f = 0; f < nIf; ++f) phi.internal[f] -= pFlux.internal[f];
            for (std::size_t pi = 0; pi < phi.boundary.size(); ++pi)
                for (std::size_t i = 0; i < phi.boundary[pi].size(); ++i)
                    phi.boundary[pi][i] -= pFlux.boundary[pi][i];

            // THE NEW FLUX REACHES THE FLUX-CONDITIONAL PATCHES BEFORE U'S BOUNDARY IS EVALUATED, which
            // is the order pEqn.H gives: `phi = phiHbyA - p_rghEqn.flux()`, then
            // U.correctBoundaryConditions(), whose updateCoeffs looks phi up. p_rgh's patches take it
            // too, for the next corrector's laplacian. See pushFluxToPatches in inter_case_cpp.cu for
            // what leaving this out cost on capillaryRise.
            for (std::size_t pi = 0; pi < patches.size() && pi < phi.boundary.size(); ++pi)
            {
                U.boundary[pi]->updateFromFlux(phi.boundary[pi]);
                p_rgh.boundary[pi]->updateFromFlux(phi.boundary[pi]);
            }

            // U = HbyA + rAU*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf) -- the divide INSIDE the
            // reconstruction and the multiply OUTSIDE, which coincide only for a uniform rAU.
            std::vector<scalar> faceFlux(static_cast<std::size_t>(nIf));
            for (label f = 0; f < nIf; ++f) faceFlux[f] = phig[f] - pFlux.internal[f];
            std::vector<std::vector<scalar>> ffB(patches.size()), rB(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const FvPatch& q = patches[pi];
                ffB[pi].assign(static_cast<std::size_t>(q.size), scalar(0));
                rB[pi].resize(static_cast<std::size_t>(q.size));
                for (label i = 0; i < q.size; ++i)
                {
                    rB[pi][i]  = rAU[q.faceCells[i]];
                    // (phig - p_rghEqn.flux()) on the boundary, the same expression as inside.
                    ffB[pi][i] = phigBnd[pi][i] - pFlux.boundary[pi][i];
                }
            }
            correctVelocity(HbyA, rAU, faceFlux, rAUfField.internal, ffB, rB, m, g, patches, U.internal);
            U.evaluateBoundary();

            // ...AND THE FLUX-CONDITIONAL VELOCITY PATCHES, which evaluateBoundary() alone does not
            // resolve. pressureInletOutletVelocity is a directionMixed: OpenFOAM's evaluate() leaves
            // the patch value at patchInternalField on an OUTFLOW face and at the normal component of
            // it on an INFLOW one, and its matrix coefficients are built from THAT value --
            // valueBoundaryCoeffs = value - (1 - d)*pif, which is identically zero on outflow BECAUSE
            // value == pif there. Leaving the WRITTEN seed in place instead makes it value - pif,
            // which is not zero.
            //
            // MEASURED on damBreak with tools/dumpInterFoam, which dumps OpenFOAM's own assembled
            // UEqn.boundaryCoeffs(): OpenFOAM |bC| = 0 on the atmosphere, brae's host 3.3406e-06, and
            // brae's DEVICE path 0 -- the device was right and this was the defect. rhoSimpleFoam has
            // called updateFromPatchVelocity for this reason since its own pcEqn gate; interFoam never
            // did.
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const FvPatch& q = patches[pi];
                std::vector<vector> Ucell(static_cast<std::size_t>(q.size), vector{0, 0, 0});
                for (label i = 0; i < q.size; ++i)
                {
                    const label c = q.faceCells[i];
                    if (c >= 0 && c < static_cast<label>(U.internal.size())) Ucell[i] = U.internal[c];
                }
                U.boundary[pi]->updateFromPatchVelocity(U.boundary[pi]->value(), Ucell, {});
            }
        }
    }

    // p == p_rgh + rho*gh, then the reference shift if p_rgh needs one -- and BOTH fields move.
    staticPressure(p_rgh.internal, *in.rho, *in.gh, p);
    if (sc.needReference)
        applyPressureReference(p, p_rgh.internal, *in.rho, *in.gh, sc.pRefCell, sc.pRefValue);
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
