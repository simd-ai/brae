// interFoam's pressure corrector -- see inter_peqn_cpp.cuh for the provenance and for the four things
// in it that are interFoam's own.
#include "inter_peqn_cpp.cuh"
#include "fvc_reconstruct_cpp.cuh"
#include "inter_set_rdeltat_cpp.cuh"   // interpolateRDeltaT: localEuler's face rDeltaT
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "pbicgstab.cuh"
#include "pcg.cuh"

namespace brae {
namespace cpu {
namespace interFoam {

// fvc::makeRelative(phi, U) and makeAbsolute (fvcMeshPhi.C:76, :121): phi -= meshPhi and phi += meshPhi,
// on every face
void makeRelativeFlux(
    SurfaceScalarField& phi,
    const SurfaceScalarField& meshPhi)
{
    for (std::size_t f = 0; f < phi.internal.size(); ++f)
    {
        phi.internal[f] -= meshPhi.internal[f];
    }
    for (std::size_t pi = 0; pi < phi.boundary.size() && pi < meshPhi.boundary.size(); ++pi)
    {
        for (std::size_t i = 0; i < phi.boundary[pi].size() && i < meshPhi.boundary[pi].size(); ++i)
        {
            phi.boundary[pi][i] -= meshPhi.boundary[pi][i];
        }
    }
}

void makeAbsoluteFlux(
    SurfaceScalarField& phi,
    const SurfaceScalarField& meshPhi)
{
    for (std::size_t f = 0; f < phi.internal.size(); ++f)
    {
        phi.internal[f] += meshPhi.internal[f];
    }
    for (std::size_t pi = 0; pi < phi.boundary.size() && pi < meshPhi.boundary.size(); ++pi)
    {
        for (std::size_t i = 0; i < phi.boundary[pi].size() && i < meshPhi.boundary[pi].size(); ++i)
        {
            phi.boundary[pi][i] += meshPhi.boundary[pi][i];
        }
    }
}

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
    // CrankNicolson's fvcDdtPhiCorr, a different operator with two ddt0 fields of its own
    if (in.cn)
    {
        if (!in.cnDdt0U || !in.cnDdt0Phi || !in.UOO || !in.UOOBnd || !in.phiOO || !in.UOldBnd)
            throw std::runtime_error(
                "brae interFoam ddtCorr: CrankNicolson needs its two ddt0 fields, U.oldTime().oldTime() "
                "(cells and patches), U.oldTime()'s patches and phi.oldTime().oldTime(); the caller "
                "supplied fewer.");
        std::vector<bool> fixes(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            fixes[pi] = U.boundary[pi]->fixesValue();
        }
        // ON A MOVING MESH it is a DIFFERENT OPERATOR: fvcDdtUfCorr, built from Uf.oldTime() and its
        // own old-old level rather than from phi.oldTime() (CrankNicolsonDdtScheme.C:1201-1257).
        // interFoam calls ddtCorr(U, phi) on a static mesh and ddtCorr(U, Uf) on a moving one
        // (pEqn.H:23-27), and under CrankNicolson those two resolve to different member functions --
        // the same distinction the Euler branch below makes with `UfOld`.
        if (in.UfOld)
        {
            if (!in.UfOO || !in.cnDdt0Uf)
                throw std::runtime_error(
                    "brae interFoam ddtCorr: CrankNicolson on a moving mesh needs Uf.oldTime().oldTime() "
                    "and its own ddt0 field; the caller supplied the mesh's Uf.oldTime() alone.");
            fv::fvcDdtUfCorr(*in.cn, *in.cnDdt0U, *in.cnDdt0Uf, *in.UOld, *in.UOO, *in.UOldBnd, *in.UOOBnd,
                             *in.UfOld, *in.UfOO, fixes, in.ddtPhiCoeff, m, g, patches, out);
            return;
        }
        fv::fvcDdtPhiCorr(*in.cn, *in.cnDdt0U, *in.cnDdt0Phi, *in.UOld, *in.UOO, *in.UOldBnd, *in.UOOBnd,
                          *in.phiOld, *in.phiOO, fixes, in.ddtPhiCoeff, m, g, patches, out);
        return;
    }

    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<scalar>& w   = g.weights();
    const std::vector<vector>& Sf  = g.Sf();
    const scalar rDeltaT = scalar(1) / in.deltaT;
    // localEuler's face rDeltaT (DdtCorrInput::rDeltaT); empty under Euler
    SurfaceScalarField rDeltaTf;
    if (in.rDeltaT)
    {
        if (in.UfOld || in.cn)
            throw std::runtime_error(
                "brae interFoam ddtCorr: a local time step on a moving mesh, or beside CrankNicolson, is not "
                "ported.");
        rDeltaTf = interpolateRDeltaT(*in.rDeltaT, m, g, patches);
    }
    // OpenFOAM's SMALL IN A DOUBLE BUILD (doubleScalar.H:62). This was 1e-37, which is the FLOAT
    // build's VSMALL (floatScalar.H:64): the two differ where |phi| is at or below 1e-15, so the
    // limiter's ratio there was |phiCorr|/|phi| instead of |phiCorr|/1e-15.
    const scalar kSmall = scalar(1e-15);

    out.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        // phiCorr = phi.oldTime() - (interpolate(U.oldTime()) & Sf) -- or, on a moving mesh,
        // (Sf & Uf.oldTime()) in phi.oldTime()'s place, with the Sf and weights of the mesh as it
        // stands NOW (EulerDdtScheme.C, fvcDdtUfCorr)
        const vector& uo = (*in.UOld)[own[f]];
        const vector& un = (*in.UOld)[nei[f]];
        const vector uf{w[f]*uo.x + (scalar(1) - w[f])*un.x,
                        w[f]*uo.y + (scalar(1) - w[f])*un.y,
                        w[f]*uo.z + (scalar(1) - w[f])*un.z};
        const scalar interpFlux = uf.x*Sf[f].x + uf.y*Sf[f].y + uf.z*Sf[f].z;
        const scalar phiUf0 = in.UfOld ? dot(Sf[f], in.UfOld->internal[f]) : in.phiOld->internal[f];
        const scalar phiCorr = phiUf0 - interpFlux;

        // note 1: a NEGATIVE ddtPhiCoeff selects the limiter, which is the default. It switches the
        // correction OFF where it is large compared with the flux -- the opposite of what a constant 1
        // would do.
        const scalar coeff = (in.ddtPhiCoeff < scalar(0))
            ? scalar(1) - std::fmin(std::fabs(phiCorr)
                                  / (std::fabs(phiUf0) + kSmall), scalar(1))
            : in.ddtPhiCoeff;

        out.internal[f] = coeff * (in.rDeltaT ? rDeltaTf.internal[static_cast<std::size_t>(f)] : rDeltaT) * phiCorr;
    }

    // note 2: zero on every patch where U fixes a value, and on every cyclicAMI patch
    // (ddtScheme.C:178-181, isA<cyclicAMIFvPatch> -- a cyclicACMI patch is a coupledFvPatch, not one)
    out.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        out.boundary[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(0));
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (U.boundary[pi]->fixesValue()) continue;       // already zero, and stays zero
        const FvPatch& q = patches[pi];
        if (q.type == "cyclicAMI") continue;
        for (label i = 0; i < q.size; ++i)
        {
            const bool havePatch = in.UOldBnd && pi < in.UOldBnd->size()
                                && static_cast<std::size_t>(i) < (*in.UOldBnd)[pi].size();
            // ON A COUPLED PATCH fvc::dotInterpolate(Sf, U.oldTime()) is the two CELLS' old velocities
            // interpolated, as on an internal face, and never the stored patch value
            const vector uo = q.coupled ? coupledLinear(q, i, *in.UOld)
                            : havePatch ? (*in.UOldBnd)[pi][static_cast<std::size_t>(i)]
                                        : (*in.UOld)[q.faceCells[i]];
            const vector& S  = Sf[q.start + i];
            const scalar interpFlux = uo.x*S.x + uo.y*S.y + uo.z*S.z;
            const scalar pOld = in.UfOld
                              ? dot(S, in.UfOld->boundary[pi][i])
                              : ((pi < in.phiOld->boundary.size()
                                  && static_cast<std::size_t>(i) < in.phiOld->boundary[pi].size())
                                 ? in.phiOld->boundary[pi][i] : scalar(0));
            const scalar phiCorr = pOld - interpFlux;
            const scalar coeff = (in.ddtPhiCoeff < scalar(0))
                ? scalar(1) - std::fmin(std::fabs(phiCorr)/(std::fabs(pOld) + kSmall), scalar(1))
                : in.ddtPhiCoeff;
            const scalar rDT = in.rDeltaT ? rDeltaTf.boundary[pi][static_cast<std::size_t>(i)] : rDeltaT;
            out.boundary[pi][i] = coeff * rDT * phiCorr;
        }
    }
}


const std::vector<scalar>& namedPatchFlux(
    const std::string& fluxName,
    std::size_t patchIndex,
    const std::string& patchName,
    const SurfaceScalarField& phi,
    const SurfaceScalarField* rhoPhi)
{
    if (fluxName == "phi") return phi.boundary[patchIndex];
    if (fluxName == "rhoPhi")
    {
        if (!rhoPhi || patchIndex >= rhoPhi->boundary.size())
            throw std::runtime_error(
                "brae interFoam: patch `" + patchName + "` names `phi rhoPhi` and the caller has no "
                "rhoPhi to hand it.");
        return rhoPhi->boundary[patchIndex];
    }
    throw std::runtime_error(
        "brae interFoam: patch `" + patchName + "` names the flux `" + fluxName + "`. interFoam has "
        "`phi` and `rhoPhi`; OpenFOAM would look the named field up and stop when there is none.");
}


void updateVelocityPatchesFromCells(
    GeometricField<vector>& U,
    const std::vector<FvPatch>& patches)
{
    // THE CELLS' VELOCITY IS GATHERED FOR A PATCH THAT READS IT. It was gathered for every face of every patch
    // and handed to updateFromPatchVelocity, whose body is empty in every class but one (readsPatchCellVelocity)
    // -- on a 2-D mesh the two empty patches are a face a cell each. MEASURED on laminar/waves/streamFunction
    // (160,000 cells, 3 calls a step), 2026-10-05: most of the 1.9 ms of the U hook's patch updates.
    //   BRAE_CONTROL_U_PATCH_GATHER_ALL=1    gathered for every patch, as before
    //   BRAE_CONTROL_U_PATCH_GATHER_NONE=1   a gate's CONTROL, deliberately wrong: gathered for none, so the patch
    //                                        that reads it is handed zeros
    static const bool gatherAll = std::getenv("BRAE_CONTROL_U_PATCH_GATHER_ALL") != nullptr;
    static const bool gatherNone = std::getenv("BRAE_CONTROL_U_PATCH_GATHER_NONE") != nullptr;
    static const std::vector<vector> notRead;
    static bool said = false;
    if (!said && !gatherAll)
    {
        said = true;
        std::size_t nRead = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (U.boundary[pi]->readsPatchCellVelocity()) ++nRead;
        }
        std::printf("  U patches: the cells' velocity is gathered for the %zu of %zu patches that read it; "
                    "BRAE_CONTROL_U_PATCH_GATHER_ALL=1 gathers it for every patch\n", nRead, patches.size());
        if (gatherNone)
        {
            std::printf("  *** CONTROL MODE: the cells' velocity is gathered for no patch, and the ones that read "
                        "it are handed zeros. This run is deliberately wrong. ***\n");
        }
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        if (!gatherAll && !U.boundary[pi]->readsPatchCellVelocity())
        {
            U.boundary[pi]->updateFromPatchVelocity(U.boundary[pi]->value(), notRead, {});
            continue;
        }
        std::vector<vector> Ucell(static_cast<std::size_t>(q.size), vector{0, 0, 0});
        for (label i = 0; i < q.size && !gatherNone; ++i)
        {
            const label c = q.faceCells[i];
            if (c >= 0 && c < static_cast<label>(U.internal.size()))
            {
                Ucell[static_cast<std::size_t>(i)] = U.internal[static_cast<std::size_t>(c)];
            }
        }
        U.boundary[pi]->updateFromPatchVelocity(U.boundary[pi]->value(), Ucell, {});
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

bool adjustPhi(
    SurfaceScalarField& phi,
    const GeometricField<vector>& U,
    bool needReference,
    const std::vector<FvPatch>& patches)
{
    if (!needReference) return false;
    scalar massIn = 0;
    scalar fixedMassOut = 0;
    scalar adjustableMassOut = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "empty" || isCoupledInterfaceType(patches[pi].type)) continue;
        const fvPatchField<vector>& Up = *U.boundary[pi];
        const std::vector<scalar>& phip = phi.boundary[pi];
        // `Up.fixesValue() && !isA<inletOutletFvPatchVectorField>(Up)`: a fixed outflow is not
        // adjustable, an inletOutlet's is
        const bool fixed = Up.fixesValue() && !dynamic_cast<const InletOutletPatchField<vector>*>(&Up);
        for (const scalar v : phip)
        {
            if (v < 0)
            {
                massIn -= v;
            }
            else if (fixed)
            {
                fixedMassOut += v;
            }
            else
            {
                adjustableMassOut += v;
            }
        }
    }
    // totalFlux = VSMALL + sum(mag(phi)) (adjustPhi.C:91) -- and Foam::sum() of a GeometricField is
    // gSum(f1.primitiveField()) (GeometricFieldFunctions.C:470-497): the INTERNAL faces only, summed
    // from zero, VSMALL added after. This used to add every non-empty boundary patch too, which moved
    // all three relative tests below; on test_adjust_phi_guards' straddle (massIn 1, fixed outflow
    // 1 + 2e-8, internal flux 1) OpenFOAM stops and the inflated normaliser (3) let the run continue.
    // The simpleFoam and rhoSimpleFoam twins were corrected for the same thing; this one is shared by
    // the pressure corrector and CorrectPhi, on both arms.
    scalar sumMagPhi = 0;
    for (const scalar v : phi.internal)
    {
        sumMagPhi += std::fabs(v);
    }
    const scalar totalFlux = scalar(1e-300) + sumMagPhi;
    scalar massCorr = 1;
    const scalar magAdjustableMassOut = std::fabs(adjustableMassOut);
    // VSMALL, SMALL
    if (magAdjustableMassOut > scalar(1e-300) && magAdjustableMassOut/totalFlux > scalar(1e-15))
    {
        massCorr = (massIn - fixedMassOut)/adjustableMassOut;
    }
    else if (std::fabs(fixedMassOut - massIn)/totalFlux > scalar(1e-8))
    {
        // OpenFOAM's FatalError (adjustPhi.C:108-119), in its words and with its four numbers. They are
        // printed %.6e: std::to_string's %f wrote every flux on damBreak as 0.000000.
        char nums[256];
        std::snprintf(nums, sizeof(nums),
                      "Total flux              : %.6e\n"
                      "Specified mass inflow   : %.6e\n"
                      "Specified mass outflow  : %.6e\n"
                      "Adjustable mass outflow : %.6e",
                      (double)totalFlux, (double)massIn, (double)fixedMassOut, (double)adjustableMassOut);
        throw std::runtime_error(
            std::string("brae interFoam adjustPhi: Continuity error cannot be removed by adjusting the "
                        "outflow.\nPlease check the velocity boundary conditions and/or run potentialFoam "
                        "to initialise the outflow.\n") + nums);
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "empty" || isCoupledInterfaceType(patches[pi].type)) continue;
        const fvPatchField<vector>& Up = *U.boundary[pi];
        if (Up.fixesValue() && !dynamic_cast<const InletOutletPatchField<vector>*>(&Up)) continue;
        for (scalar& v : phi.boundary[pi])
        {
            if (v > 0)
            {
                v *= massCorr;
            }
        }
    }
    return std::fabs(massIn)/totalFlux < scalar(1e-15)
        && std::fabs(fixedMassOut)/totalFlux < scalar(1e-15)
        && std::fabs(adjustableMassOut)/totalFlux < scalar(1e-15);
}

// fvc::correctUf(Uf, U, phi), pEqn.H:70-72 on a moving mesh: Uf = interpolate(U), then its NORMAL
// component is replaced by the flux's, Uf += n*(phi/magSf - (n & Uf)). ONE COPY, shared with the
// device loop: Uf is a host field, and next step's ddtCorr reads (Sf & Uf.oldTime()) off it
// (EulerDdtScheme's fvcDdtUfCorr), so the device arm has to keep it too.
void correctUf(
    SurfaceVectorField&           Uf,
    const GeometricField<vector>& U,
    const SurfaceScalarField&     phi,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    const label nIf = m.nInternalFaces();
    std::vector<std::vector<vector>> Ub(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        Ub[pi] = U.boundary[pi]->value();
    }
    Uf = fvc::interpolate(U.internal, Ub, m, g, patches);
    for (label f = 0; f < nIf; ++f)
    {
        const vector n = g.Sf()[f]/g.magSf()[f];
        vector& uf = Uf.internal[f];
        uf += n*(phi.internal[f]/g.magSf()[f] - dot(n, uf));
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        if (q.type == "empty") continue;
        for (label i = 0; i < q.size; ++i)
        {
            const vector n = g.Sf()[q.start + i]/q.magSf[i];
            vector& uf = Uf.boundary[pi][i];
            uf += n*(phi.boundary[pi][i]/q.magSf[i] - dot(n, uf));
        }
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

    // EVERY PRESSURE TAP FROM THE SAME CORRECTOR, the first. The device loop already pins its own
    // there (device_inter_step.cu:494 and :640, with the reason written out); this one did not. Its
    // HbyA, phiHbyA, rAU, phig and the rest sat under a bare `if (in.taps)` and were overwritten by
    // every corrector, so `tapCorrector` reported 0 while the fields it labelled were the LAST
    // corrector's. On a case running nCorrectors 2 that made every host-against-device and
    // host-against-OpenFOAM dump compare two different correctors.
    // MEASURED on the permeable-moving case: it read HbyA 4.66e-01 and phiHbyA 3.64e-01 from
    // OpenFOAM while the momentum matrix those are built from -- upper, lower, diag and source --
    // agrees to 1.5e-09, and rAU to 3.1e-10. tools/dumpInterFoam now suffixes its own per-corrector
    // writes for the same reason, so a bare `<name>.dump` is the first corrector on both sides.
    const bool tapHere = (in.taps != nullptr) && (std::max(in.correctorIndex, 0) == tapCorrectorWanted());

    // rAU = 1/UEqn.A(), rAUf = interpolate(rAU).
    const std::vector<scalar> A = matrixA(*in.UEqn, m, g, patches);
    std::vector<scalar> rAU(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c) rAU[c] = scalar(1) / A[c];
    if (in.rAUOut)
    {
        *in.rAUOut = rAU;
    }
    const SurfaceScalarField rAUfField = fvc::interpolate(rAU, m, g, patches);

    // HbyA = constrainHbyA(rAU*UEqn.H(), U, p_rgh).
    std::vector<vector> hCoupled;
    const std::vector<vector> H = matrixH(*in.UEqn, U, m, g, patches,
                                          in.taps ? &hCoupled : nullptr);
    if (tapHere)
    {
        in.taps->uAtH = U.internal;
        in.taps->hPairX.assign(static_cast<std::size_t>(nC), scalar(0));
        in.taps->hNoPairX.assign(static_cast<std::size_t>(nC), scalar(0));
        for (label c = 0; c < nC; ++c)
        {
            // the pair's half divided by V, as H() divides it, so the two arms' taps are the same
            // quantity (the device's addHKernel divides by V inside the kernel)
            const scalar pair = (static_cast<std::size_t>(c) < hCoupled.size())
                              ? hCoupled[static_cast<std::size_t>(c)].x/g.V()[c] : scalar(0);
            in.taps->hPairX[static_cast<std::size_t>(c)] = pair;
            in.taps->hNoPairX[static_cast<std::size_t>(c)] = H[static_cast<std::size_t>(c)].x - pair;
        }
    }
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

    // Uf WITHOUT meshPhi IS A DYNAMIC MESH THAT DOES NOT MOVE -- a refining one. OpenFOAM creates Uf on
    // mesh.dynamic() (createUfIfPresent.H) and computes a mesh flux only on mesh.moving(), so the pair is
    // not symmetric: Uf is the wider condition. meshPhi without Uf is not reachable in OpenFOAM, because
    // moving implies dynamic, and is refused rather than run with a Uf nobody corrects.
    if (in.meshPhi != nullptr && in.Uf == nullptr)
        throw std::runtime_error("brae interFoam pEqn: a moving mesh has a mesh flux and no Uf.");
    if (in.ddt)
    {
        std::vector<scalar> rhoRAU;
        rhoRAUf(*in.rho, rAU, m, g, rhoRAU);
        SurfaceScalarField corr;
        ddtCorr(*in.ddt, U, m, g, patches, corr);
        // MRF.zeroFilter(...), pEqn.H:17: MRFZone::zero sets the flux to Zero on the zone's internal,
        // included and excluded faces (MRFZoneTemplates.C:213-247). The correction compares phi.oldTime()
        // with the flux of U.oldTime(), and inside the zone the first is RELATIVE to the frame and the
        // second is not, so their difference there is the frame flux and not a correction.
        if (in.mrf)
        {
            for (const MRF::Zone& z : *in.mrf)
            {
                if (!z.active) continue;
                for (label fi : z.internalFaces)
                {
                    corr.internal[fi] = scalar(0);
                }
                for (std::size_t pi = 0; pi < patches.size(); ++pi)
                {
                    for (const std::vector<std::vector<label>>* lists : {&z.includedFaces, &z.excludedFaces})
                    {
                        if (pi >= lists->size()) continue;
                        for (label fi : (*lists)[pi])
                        {
                            corr.boundary[pi][static_cast<std::size_t>(fi)] = scalar(0);
                        }
                    }
                }
            }
        }
        for (label f = 0; f < nIf; ++f) phiHbyA.internal[f] += rhoRAU[f] * corr.internal[f];
        // ...AND THE BOUNDARY. pEqn.H:16-17 adds a whole surfaceScalarField, and fvcDdtPhiCoeff zeroes
        // the coupling coefficient only on a patch where U FIXES A VALUE (ddtScheme.C, fvcDdtPhiCoeff:
        // `if (U.boundaryField()[patchi].fixesValue() ...) ccbf[patchi] = 0.0`). On any other patch --
        // a zeroGradient outlet -- the correction is live, with interpolate(rho*rAU)'s patch value:
        // rho's own and rAU's extrapolated one, 1/A of the face cell (fvMatrix::A() is
        // extrapolatedCalculated). ddtCorr above already computed it and this dropped it. Every open
        // patch gated before RAS/weirOverflow carried a value-fixing U, where it is zero; that case's
        // outlet is `U zeroGradient`, and once its flux turned outward at the third step the coefficient
        // left zero: phi 3.2e-04 out in the outlet's top corner after that step, U 3.6e-05 after ten.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (U.boundary[pi]->fixesValue()) continue;
            const FvPatch& q = patches[pi];
            bool any = false;
            for (label i = 0; i < q.size && !any; ++i)
            {
                any = corr.boundary[pi][static_cast<std::size_t>(i)] != scalar(0);
            }
            if (!any) continue;
            if (!in.rhoBnd || in.rhoBnd->size() <= pi)
                throw std::runtime_error(
                    "brae interFoam pEqn: patch `" + q.name + "` does not fix U, so ddtCorr is live on "
                    "it and needs rho's patch values (PressureStepInput::rhoBnd) for interpolate(rho*rAU).");
            for (label i = 0; i < q.size; ++i)
            {
                const std::size_t k = static_cast<std::size_t>(i);
                if (q.coupled)
                {
                    // interpolate(rho*rAU) on a coupled face: the PRODUCT per cell, interpolated once
                    const label P = q.faceCells[i];
                    scalar rrN = 0;
                    if (q.amiOffsets.empty())
                    {
                        const label N = q.nbrFaceCells[k];
                        rrN = (*in.rho)[N]*rAU[N];
                    }
                    else
                    {
                        for (label s = q.amiOffsets[k]; s < q.amiOffsets[k + 1]; ++s)
                        {
                            const label N = q.amiNbrCells[static_cast<std::size_t>(s)];
                            rrN += q.amiWeights[static_cast<std::size_t>(s)]*((*in.rho)[N]*rAU[N]);
                        }
                    }
                    const scalar rr = q.weights[k]*((*in.rho)[P]*rAU[P]) + (scalar(1) - q.weights[k])*rrN;
                    phiHbyA.boundary[pi][k] += rr * corr.boundary[pi][k];
                    continue;
                }
                phiHbyA.boundary[pi][k] += (*in.rhoBnd)[pi][k] * rAU[q.faceCells[i]] * corr.boundary[pi][k];
            }
        }
    }
    // MRF.makeRelative(phiHbyA), pEqn.H:19
    if (in.mrf && !in.mrf->empty())
    {
        MRF::makeRelative(phiHbyA, *in.mrf, g, patches);
    }

    // pEqn.H:21-26: on a closed case the boundary flux is balanced by adjustPhi, and on a moving mesh
    // it is the RELATIVE flux that is balanced -- makeRelative around it, makeAbsolute after
    if (sc.needReference)
    {
        if (in.meshPhi)
        {
            makeRelativeFlux(phiHbyA, *in.meshPhi);
        }
        adjustPhi(phiHbyA, U, true, patches);
        if (in.meshPhi)
        {
            makeAbsoluteFlux(phiHbyA, *in.meshPhi);
        }
    }

    // phig = (surfaceTensionForce - ghf*snGrad(rho)) * rAUf * magSf -- and NOT snGrad(p_rgh), which is
    // the laplacian below. phiHbyA += phig.
    std::vector<scalar> phig;
    buoyancyFlux(in.stf->internal, *in.ghf, in.snGradRho->internal, rAUfField.internal, g.magSf(), phig);
    if (tapHere)
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
    // Instrument: BRAE_STAGE_DUMP_DIR=<dir> (+ BRAE_STAGE_DUMP_ITER=n, default 1) writes the
    // momentum-to-pressure hand-off at the nth step's FIRST corrector, under the names
    // tools/dumpInterFoam writes OpenFOAM's -- rAU, HbyA, rAUf, phig, phiHbyA, stf, snGradRho. It goes
    // here rather than at the end of the step because every one of these feeds the next: ddtCorr reads
    // Uf.oldTime() and correctUf writes Uf back from U, so a field written at the end of a step cannot
    // say which of them moved first.
    {
        static int firstCorrectors = 0;
        const char* dd = std::getenv("BRAE_STAGE_DUMP_DIR");
        const bool countHere = (std::max(in.correctorIndex, 0) == tapCorrectorWanted());
        if (countHere) ++firstCorrectors;
        const char* it = std::getenv("BRAE_STAGE_DUMP_ITER");
        if (dd && countHere && firstCorrectors == (it && *it ? std::atoi(it) : 1))
        {
            std::error_code ec;
            std::filesystem::create_directories(dd, ec);
            const std::string dir(dd);
            const auto wS = [&dir](const char* n, const std::vector<scalar>& v)
            {
                std::ofstream o(dir + "/" + n);
                o.precision(17);
                for (const scalar x : v) o << x << "\n";
            };
            const auto wV = [&dir](const char* n, const std::vector<vector>& v)
            {
                std::ofstream o(dir + "/" + n);
                o.precision(17);
                for (const vector& x : v) o << x.x << " " << x.y << " " << x.z << "\n";
            };
            wS("peqnRAU", rAU);
            wV("peqnHbyA", HbyA);
            wS("peqnRAUf", rAUfField.internal);
            wS("peqnPhig", phig);
            wS("peqnPhiHbyA", phiHbyA.internal);
            wS("peqnStf", in.stf->internal);
            wS("peqnSnGradRho", in.snGradRho->internal);
        }
    }
    for (label f = 0; f < nIf; ++f) phiHbyA.internal[f] += phig[f];

    // ...ON THE BOUNDARY TOO -- see PressureStepInput::ghfBnd. rAUf at an uncoupled patch is the face
    // cell's rAU, which is what fvc::interpolate gives there.
    if (tapHere)
    {
        in.taps->phiHbyABndPrePhig = phiHbyA.boundary;
        in.taps->uEqnDiag  = in.UEqn->diag;
        in.taps->uEqnUpper = in.UEqn->upper;
        in.taps->uEqnLower = in.UEqn->lower;
        in.taps->uEqnSourceX.clear();
        for (const vector& v : in.UEqn->source) in.taps->uEqnSourceX.push_back(v.x);
        in.taps->uEqnICx.assign(patches.size(), std::vector<scalar>());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (const vector& v : in.UEqn->internalCoeffs[pi]) in.taps->uEqnICx[pi].push_back(v.x);
        }
        in.taps->uEqnBCx.assign(patches.size(), std::vector<scalar>());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (const vector& v : in.UEqn->boundaryCoeffs[pi])
            {
                in.taps->uEqnBCx[pi].push_back(v.x);
            }
        }
    }
    std::vector<std::vector<scalar>> phigBnd(patches.size());
    if (in.ghfBnd)
    {
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            std::vector<scalar> rAUfb(static_cast<std::size_t>(q.size));
            for (label i = 0; i < q.size; ++i)
            {
                // interpolate(rAU): the face cell's at an uncoupled patch, the two cells' at a coupled one
                rAUfb[i] = q.coupled ? coupledLinear(q, i, rAU) : rAU[q.faceCells[i]];
            }
            buoyancyFlux(in.stf->boundary[pi], (*in.ghfBnd)[pi], in.snGradRho->boundary[pi],
                         rAUfb, q.magSf, phigBnd[pi]);
            for (label i = 0; i < q.size; ++i) phiHbyA.boundary[pi][i] += phigBnd[pi][i];
        }
        if (tapHere)
        {
            in.taps->phigBnd = phigBnd;
            in.taps->rAUfBnd.assign(patches.size(), std::vector<scalar>());
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const FvPatch& q = patches[pi];
                in.taps->rAUfBnd[pi].resize(static_cast<std::size_t>(q.size));
                for (label i = 0; i < q.size; ++i)
                {
                    in.taps->rAUfBnd[pi][static_cast<std::size_t>(i)] =
                        q.coupled ? coupledLinear(q, i, rAU) : rAU[q.faceCells[i]];
                }
            }
        }
    }
    else
    {
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            phigBnd[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(0));
    }

    // constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF): a fixedFluxPressure patch's gradient is
    // PRESCRIBED from the flux, and brae refuses to assemble one that has not been set.
    //
    //     snGrad = (phiHbyA_b - (Sf_b & U_b)) / (magSf_b * rAUf_b)          constrainPressure.C:62-72
    //
    // THE FLUX SUBTRACTED IS THE VELOCITY'S, Sf & U_b -- NOT the stored phi_b, which is what this took.
    // The two are the same number on a wall that does not move, which is every fixedFluxPressure
    // patch this solver had met: there phi_b is whatever the last corrector left, 0 or 1.9e-37. On a
    // patch whose U_b CHANGES they are not: with phi_b the corrector hands back exactly the flux it
    // was given, so a flux that starts at zero stays at zero and the inlet never opens. Measured on
    // laminar/waves/stokesI after ten steps, against real OpenFOAM, with the wave model's own patch
    // values exact to 4e-16: U 260% out, and OpenFOAM's own answer with the wave switched OFF was
    // closer to it (64%) than brae was.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!p_rgh.boundary[pi]->updateableSnGrad()) continue;
        const FvPatch& q = patches[pi];
        const std::vector<vector>& ub = U.boundary[pi]->value();
        std::vector<scalar> sn(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i)
        {
            const scalar rf = rAU[q.faceCells[i]];        // interpolate(rAU) at an uncoupled patch
            const scalar ph = (pi < phiHbyA.boundary.size()
                               && static_cast<std::size_t>(i) < phiHbyA.boundary[pi].size())
                            ? phiHbyA.boundary[pi][i] : scalar(0);
            const scalar SfU = dot(g.Sf()[q.start + i], ub[i]);
            sn[i] = (ph - SfU) / (q.magSf[i] * rf);
        }
        // prghPermeableAlphaTotalPressure rebuilds its refValue and valueFraction INSIDE updateSnGrad,
        // from rho, phi and U on the patch and gh at the face centres (...FvPatchScalarField.C:151-212).
        // The phi it looks up is the field as it stands when constrainPressure runs -- the last
        // corrector's, not phiHbyA.
        if (p_rgh.boundary[pi]->isPrghPermeableAlphaTotalPressure())
        {
            if (!in.rhoBnd || !in.ghfBnd || in.rhoBnd->size() <= pi || in.ghfBnd->size() <= pi
             || phi.boundary.size() <= pi)
                throw std::runtime_error(
                    "brae interFoam pEqn: p_rgh patch `" + q.name + "` is a prghPermeableAlphaTotalPressure, "
                    "which needs rho's patch values, gh at the patch's face centres and phi on the patch "
                    "(PressureStepInput::rhoBnd, ::ghfBnd).");
            p_rgh.boundary[pi]->updatePermeableTotalPressure((*in.rhoBnd)[pi], phi.boundary[pi], ub,
                                                             (*in.ghfBnd)[pi]);
        }
        p_rgh.boundary[pi]->updateSnGrad(sn);
    }

    for (label corr = 0; corr <= sc.nNonOrthogonalCorrectors; ++corr)
    {
        // the fvMatrix constructor's updateCoeffs -- see updatePressurePatchesFromVelocity
        updatePressurePatchesFromVelocity(p_rgh, U, in.rhoBnd, patches);
        // porousBafflePressure::updateCoeffs, which fvMatrix's constructor runs at THIS assembly: the
        // owner's jump from phi as it stands -- the last corrector's, not phiHbyA -- and from the
        // stored patch values of the laminar nu and of rho; the other side takes the owner's
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].owner || !p_rgh.boundary[pi]->isPorousBafflePressure()) continue;
            if (!in.rhoBnd || !in.nuBnd || in.rhoBnd->size() <= pi || in.nuBnd->size() <= pi)
                throw std::runtime_error(
                    "brae interFoam pEqn: p_rgh patch `" + patches[pi].name + "` is a porousBafflePressure, "
                    "which needs the patch values of rho and of the mixture's laminar nu "
                    "(PressureStepInput::rhoBnd, ::nuBnd).");
            const std::vector<scalar> jump = p_rgh.boundary[pi]->porousBaffleJump(
                namedPatchFlux(p_rgh.boundary[pi]->fluxName(), pi, patches[pi].name, phi, in.rhoPhi),
                (*in.nuBnd)[pi], (*in.rhoBnd)[pi]);
            p_rgh.boundary[pi]->setOwnerJump(jump);
            p_rgh.boundary[static_cast<std::size_t>(patches[pi].nbrPatch)]->setOwnerJump(jump);
        }
        if (in.taps)
        {
            // jumpBnd is this corrector's, jumpHistory is every corrector's by design -- it is the
            // one tap here that a single corrector cannot answer, and the cyclic gates read it.
            if (tapHere) in.taps->jumpBnd.assign(patches.size(), std::vector<scalar>());
            std::vector<scalar> flatJump;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                if (const std::vector<scalar>* j = p_rgh.boundary[pi]->coupledJump())
                {
                    if (tapHere) in.taps->jumpBnd[pi] = *j;
                }
                if (!patches[pi].coupled) continue;
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const std::vector<scalar>* j = p_rgh.boundary[pi]->coupledJump();
                    flatJump.push_back((j && static_cast<std::size_t>(i) < j->size())
                                       ? (*j)[static_cast<std::size_t>(i)] : scalar(0));
                }
            }
            in.taps->jumpHistory.push_back(flatJump);
        }
        FvScalarMatrix pe = fvm::laplacian<scalar>(rAUfField, p_rgh, m, g, patches, sc.correctedLaplacian, sc.nonOrthCoeffs);
        // corrector 0 on BOTH arms: the device copies its pressure taps there, and a dump that
        // compares different correctors reads as a defect in whichever term moves between them.
        if (in.taps && corr == 0 && std::max(in.correctorIndex, 0) == tapCorrectorWanted())
        {
            in.taps->pLaplacianSource = pe.source;
            in.taps->tapCorrector = in.correctorIndex;
        }
        if (sc.correctedLaplacian)
        {
            // gaussLaplacianSchemes.C: source -= V*div(gammaMagSf*snGradCorrection(p_rgh)), the
            // correction from grad(p_rgh) through its own gradSchemes entry (Gauss linear, the only one
            // this solver admits) on the p_rgh of THIS pass, and the same flux kept for
            // p_rghEqn.flux(). The two are built from one face field so `phi = phiHbyA - flux` stays
            // conservative on a non-orthogonal mesh.
            const std::vector<vector> gradP = gradOf(p_rgh, sc.gradPrgh, m, g, patches);
            const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                rAUfField, p_rgh, gradP, m, g, patches, sc.snGradLimitCoeff);
            if (in.taps && std::max(in.correctorIndex, 0) == tapCorrectorWanted()) in.taps->pNonOrthSource = corr;
            for (label c = 0; c < nC; ++c)
            {
                pe.source[c] -= corr[c];
            }
            pe.faceFluxCorrection = fvm::laplacianCorrFlux<scalar, vector>(
                rAUfField, gradP, m, g, sc.snGradLimitCoeff, &p_rgh);
            pe.faceFluxCorrectionBoundary = fvm::laplacianCorrFluxCoupled<scalar, vector>(
                rAUfField, gradP, g, patches, sc.snGradLimitCoeff, p_rgh);
        }
        const std::vector<scalar> div = fvc::div(phiHbyA, m, g, patches);
        if (in.taps && corr == 0 && std::max(in.correctorIndex, 0) == tapCorrectorWanted()) in.taps->pDivPhiHbyA = div;
        for (label c = 0; c < nC; ++c) pe.source[c] += div[c] * g.V()[c];

        if (sc.needReference)
        {
            // p_rghEqn.setReference(pRefCell, getRefCellValue(p_rgh, pRefCell)) (pEqn.H:47): the cell
            // is pinned at ITS CURRENT p_rgh, not at pRefValue -- pRefValue is p's level, applied
            // below after the solve. fvMatrix::setReference: source += diag*value, diag += diag.
            // This used pRefValue here, on a path no gated case had ever taken.
            if (sc.pRefCell < 0 || sc.pRefCell >= nC)
                throw std::runtime_error("brae interFoam pEqn: pRefCell is outside the mesh.");
            const scalar refValue = p_rgh.internal[sc.pRefCell];
            pe.source[sc.pRefCell] += pe.diag[sc.pRefCell] * refValue;
            pe.diag[sc.pRefCell]   += pe.diag[sc.pRefCell];
        }

        // the assembled system, at the device arm's tap point: after the source, before the solve,
        // and on the FIRST non-orthogonal pass only
        if (in.taps && corr == 0 && std::max(in.correctorIndex, 0) == tapCorrectorWanted())
        {
            in.taps->pDiag = pe.diag;
            in.taps->pUpper = pe.upper;
            in.taps->pLower = pe.lower;
            in.taps->pSource = pe.source;
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
        const GamgControls* gamgCtl = finalInner ? sc.gamgFinal : sc.gamg;
        const GamgPreconditionerControls* pcgGamgCtl = finalInner ? sc.pcgGamgFinal : sc.pcgGamg;
        SolverPerformance sp;
        if (pcgGamgCtl)
        {
            // PCG with the GAMG preconditioner: the hierarchy is the mesh's, whichever entry built it
            if (!sc.gamgCache)
            {
                throw std::runtime_error(
                    "brae interFoam pEqn: the case names a GAMG preconditioner for p_rgh and the caller "
                    "handed in no agglomeration cache. The hierarchy is the mesh's and has to outlive the step.");
            }
            sp = pcgGamgSolve(pe, p_rgh.internal, m, g, patches, *sc.gamgCache, tol, relTol, maxIter, 0,
                              *pcgGamgCtl, in.gamgLog);
        }
        else if (gamgCtl)
        {
            // GAMG, where the case names it: OpenFOAM's own hierarchy, smoother and stopping rule.
            // See gamg_solver_cpp.cuh for what standing PCG in for it cost on the wave tank.
            if (!sc.gamgCache)
            {
                throw std::runtime_error(
                    "brae interFoam pEqn: the case names GAMG for p_rgh and the caller handed in no "
                    "agglomeration cache. The hierarchy is the mesh's and has to outlive the step.");
            }
            const GamgAgglomeration& agglomeration =
                sc.gamgCache->get(m, g, gamgCtl->nCellsInCoarsestLevel);
            sp = gamgSolve(pe, p_rgh.internal, m, patches, agglomeration, *gamgCtl, in.gamgLog);
        }
        else if (finalInner ? sc.pcgDICFinal : sc.pcgDIC)
        {
            // a jump cyclic's jump enters the solve through the interface update, and only there
            CoupledJumps jumps(patches.size(), nullptr);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                jumps[pi] = p_rgh.boundary[pi]->coupledJump();
            }
            sp = pcg(pe, p_rgh.internal, m, patches, tol, relTol, maxIter, 0, &jumps);
        }
        else
        {
            sp = pbicgstab(pe, p_rgh.internal, m, patches, tol, relTol, maxIter);
        }
        if (in.solveLog)
        {
            in.solveLog->push_back(PressureSolveRecord{sp.initialResidual, sp.finalResidual, sp.nIterations});
        }
        p_rgh.evaluateBoundary();

        if (corr == sc.nNonOrthogonalCorrectors)
        {
            // through the FIELD, so that a coupled patch's half is boundaryCoeffs*patchNeighbourField
            // with the patch's own jump in it
            const SurfaceScalarField pFlux = matrixFlux(pe, p_rgh, m, patches);

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
                // ...each the flux ITS OWN entry names -- see namedPatchFlux. NOT U's on a corrector
                // that starts with its patches still updated(): OpenFOAM's evaluate skips updateCoeffs
                // there and blends with the assembly-time valueFraction -- see uPatchesUpdatedAtEntry.
                // laminar/damBreakPermeable's wall flips faces from outflow to inflow in the first
                // corrector of the first step; taking the new flux at once put the second corrector's
                // initial residual 1.2e-03 out and U 1.5e-06 after one step, with HbyA and the internal
                // phiHbyA of that corrector exact against tools/dumpInterFoam.
                // A class whose updateCoeffs ENDS IN evaluate() is never left updated, and takes the new
                // flux in every corrector: pressureInletOutletVelocity. With the skip applied to it as
                // well, an atmosphere face turning from outflow to inflow at step 55 of the staged wet
                // wall put U 6.9e-03 out in one step, from 3.5e-13 the step before.
                if (!in.uPatchesUpdatedAtEntry || U.boundary[pi]->updateCoeffsEvaluates())
                {
                    U.boundary[pi]->updateFromFlux(
                        namedPatchFlux(U.boundary[pi]->fluxName(), pi, patches[pi].name, phi, in.rhoPhi));
                }
                p_rgh.boundary[pi]->updateFromFlux(
                    namedPatchFlux(p_rgh.boundary[pi]->fluxName(), pi, patches[pi].name, phi, in.rhoPhi));
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
                    rB[pi][i]  = q.coupled ? coupledLinear(q, i, rAU) : rAU[q.faceCells[i]];
                    // (phig - p_rghEqn.flux()) on the boundary, the same expression as inside.
                    ffB[pi][i] = phigBnd[pi][i] - pFlux.boundary[pi][i];
                }
            }
            if (tapHere) in.taps->ffBnd = ffB;
            correctVelocity(HbyA, rAU, faceFlux, rAUfField.internal, ffB, rB, m, g, patches, U.internal);
            // U.correctBoundaryConditions()'s updateCoeffs, for the patches that read the cells just set
            // -- skipped while they are still updated() from the assembly (uPatchesUpdatedAtEntry)
            if (in.uUpdateCoeffsFromCells && (!in.uPatchesUpdatedAtEntry || in.controlIgnoreUpdatedLag))
            {
                in.uUpdateCoeffsFromCells();
            }
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
            updateVelocityPatchesFromCells(U, patches);
        }
    }

    // pEqn.H:70-73 on a moving mesh: fvc::correctUf(Uf, U, phi) -- Uf = interpolate(U), then its
    // normal component replaced by the ABSOLUTE flux's, Uf += n*(phi/magSf - (n & Uf)) -- and then
    // phi is made relative to the motion. The flux that leaves here is the RELATIVE one, which the
    // next alpha equation convects with.
    // continuityErrs.H (pEqn.H:64), on the flux as it stands here -- absolute on a moving mesh, which the
    // lines below make relative. Taken after the corrector instead, a rigid motion's error read 1e-19
    // where OpenFOAM's swept volumes leave 7.0e-16 (sloshingTank2D), and a deforming mesh would lose
    // its volume change.
    if (in.continuityDivOut)
    {
        *in.continuityDivOut = fvc::div(phi, m, g, patches);
    }
    if (in.Uf)
    {
        // fvcMeshPhi.C:224 gates correctUf on mesh.DYNAMIC(), so it runs on a refining mesh too -- and
        // there the flux is already absolute, so nothing is made relative afterwards.
        correctUf(*in.Uf, U, phi, m, g, patches);
    }
    if (in.meshPhi)
    {
        makeRelativeFlux(phi, *in.meshPhi);
        // ...AND THE PRESSURE PATCHES ARE TOLD THE RELATIVE FLUX, because that is the one OpenFOAM's
        // LOOK UP. pEqn.H is included once per PISO corrector and each invocation ENDS here, so the
        // next corrector's `fvm::laplacian(rAUf, p_rgh)` -- whose constructor runs every p_rgh patch's
        // updateCoeffs -- reads a phi that has already been made relative. brae's patches are TOLD
        // instead, and were told only inside the corrector where phi is still ABSOLUTE, so they kept the
        // absolute flux for the rest of the step. U's patches are NOT re-told here and must not be:
        // OpenFOAM evaluates them at `U.correctBoundaryConditions()` INSIDE the corrector, where phi is
        // still absolute (see the push site above).
        //
        // MEASURED on RAS/electrostaticDeposition, step one, `side-02` (totalPressure, 225 faces), at
        // the second and third correctors' assemblies -- tools/dumpInterFoam's phiAtPrghAssembly against
        // the same instant in brae:
        //   OpenFOAM  phi [+3.554986e-04, +3.555726e-04]   inflow faces 0     p_rgh uniform 0
        //   brae      phi [-7.468970e-08, -7.384048e-10]   inflow faces 225   p_rgh [-1.708e-10, ...]
        // The two differ by exactly meshPhi (0.08 m/s x a 4.44e-03 face = 3.55e-04), so `neg(phip)`
        // took the opposite branch on every face and totalPressure added 0.5*rho*|U_b|^2 where OpenFOAM
        // added nothing. phiHbyA's boundary was NOT the cause: brae's matches OpenFOAM's to every digit
        // there ([-4.463313e-10, -5.922619e-12] at corrector two on both sides).
        for (std::size_t pi = 0; pi < patches.size() && pi < phi.boundary.size(); ++pi)
        {
            if (patches[pi].size == 0) continue;
            p_rgh.boundary[pi]->updateFromFlux(
                namedPatchFlux(p_rgh.boundary[pi]->fluxName(), pi, patches[pi].name, phi, in.rhoPhi));
        }
    }

    // p == p_rgh + rho*gh, then the reference shift if p_rgh needs one -- and BOTH fields move.
    staticPressure(p_rgh.internal, *in.rho, *in.gh, p);
    if (sc.needReference)
    {
        applyPressureReference(p, p_rgh.internal, *in.rho, *in.gh, sc.pRefCell, sc.pRefValue);
        p_rgh.evaluateBoundary();
    }
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
