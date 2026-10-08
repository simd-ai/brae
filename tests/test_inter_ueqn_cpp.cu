// interFoam's UEqn.H -- the three things in it that are not in rhoSimpleFoam's momentum predictor.
//
// The ORACLE here is OpenFOAM's own expressions, transcribed from the lines cited in
// inter_ueqn_cpp.cuh and evaluated in this file, on a mesh built in memory. There is no solve, and no
// tolerance beyond round-off: every arm is an identity or an exact coefficient.
//
// WHAT EACH ARM EXISTS TO CATCH. None of these is arithmetic.
//
// 1. ddt's SOURCE TAKES rho.oldTime(), ITS DIAGONAL TAKES rho (EulerDdtScheme.C:455-467). Two different
//    rho fields in one term. Everywhere else in brae the two are within a per-cent of each other and
//    reusing the new one reads as a rounding choice; HERE they differ by the water/air ratio in exactly
//    the cells the interface crossed. The control below builds the same source with the new rho and
//    requires it to differ by ~1000x, so the arm can tell the two apart -- on a smooth field it could
//    not, which is why no ordinary agreement gate would ever find this.
//
// 2. THE MOMENTUM SOURCE IS A RECONSTRUCTED FACE FLUX. reconstruct(-snGrad(p_rgh)*magSf) must return
//    exactly -grad(p_rgh) for a linear p_rgh, on every cell including the boundary ones. That single
//    identity pins the sign, the magSf factor and the same-sign surfaceSum together; arm 3 drops the
//    magSf and arm 4 flips the sign, and each breaks it.
//
// 3. `solve(UEqn == R)` PUTS R INTO THE SOURCE WITH A PLUS (fvMatrix::operator==, source += V*R).
//    rhoSimpleFoam's twin carries the minus inside R itself, so a transcription that copies that call
//    site gets the buoyancy backwards and still converges -- to a dam that breaks upwards.
//
// 4. THE VISCOSITY IS rho*nuEff, NOT nuEff. The incompressible lineage's form is off by the density,
//    which on this mixture is a factor of 1000 in the water. Arm 5 assembles it both ways and requires
//    them to disagree.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "inter_ueqn_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

void checkNum(const char* what, scalar got, scalar want, scalar tol = scalar(1e-12))
{
    const bool ok = std::fabs(got - want) <= tol * std::fmax(scalar(1), std::fabs(want));
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s (got %.17g want %.17g)\n", what, (double)got, (double)want);
    if (!ok) ++failures;
}

// damBreak's own phases.
constexpr scalar kRhoWater = 1000.0;
constexpr scalar kRhoAir   = 1.0;
}   // namespace

int main()
{
    std::printf("== interFoam UEqn ==\n");

    // CELLS OF 2 x 1 x 0.5, not a unit cube. On a cube every face has the same area and a term that
    // forgot its magSf factor is a uniform rescaling -- invisible to every identity in this file. Three
    // distinct cell sizes give three distinct face areas (1, 0.5, 2) while the mesh stays orthogonal, so
    // the identity in arm 3 is still exact and its magSf control can actually fail.
    const label N = 4;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N, /*shear=*/scalar(0),
                                       /*dx=*/scalar(2), /*dy=*/scalar(1), /*dz=*/scalar(0.5));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const std::vector<scalar>& V = g.V();
    const std::vector<vector>& C = g.C();

    // ---- 1. fvm::ddt(rho, U): the diagonal takes rho, the source takes rho.oldTime() ---------------
    // A stratified column with the interface HALF A CELL ABOVE y = 2 at the old time and half a cell
    // below it at the new one, so the middle layer flips from air to water in one step. That is the
    // ordinary situation in a VoF run, not a contrived one.
    {
        std::vector<scalar> rho(nC), rhoOld(nC);
        std::vector<vector> UOld(nC);
        for (label c = 0; c < nC; ++c)
        {
            rhoOld[c] = (C[c].y < scalar(2.0)) ? kRhoWater : kRhoAir;
            rho[c]    = (C[c].y < scalar(3.0)) ? kRhoWater : kRhoAir;   // the interface rose one layer
            UOld[c]   = vector{scalar(0.3), scalar(-0.7), scalar(0.1)};
        }
        const scalar dt = scalar(0.002);

        FvVectorMatrix M;
        M.diag.assign(static_cast<std::size_t>(nC), scalar(0));
        M.source.assign(static_cast<std::size_t>(nC), vector{0, 0, 0});
        addEulerDdtRhoU(M, rho, rhoOld, UOld, V, dt);

        // the cell that changed phase: its diagonal is water's, its source is still air's
        label flipped = -1;
        for (label c = 0; c < nC && flipped < 0; ++c)
            if (rho[c] != rhoOld[c]) flipped = c;
        check("the fixture actually flips a cell's phase", flipped >= 0);

        checkNum("diag  = rho_new*V/dt",
                 M.diag[flipped], kRhoWater * V[flipped] / dt);
        checkNum("source= rho_OLD*U_old*V/dt (x)",
                 M.source[flipped].x, kRhoAir * scalar(0.3) * V[flipped] / dt);
        checkNum("source= rho_OLD*U_old*V/dt (y)",
                 M.source[flipped].y, kRhoAir * scalar(-0.7) * V[flipped] / dt);

        // THE CONTROL: the same source built with the new rho, which is what a transcription that
        // carries one rho field produces.
        const scalar wrong = kRhoWater * scalar(0.3) * V[flipped] / dt;
        std::printf("  flipped cell: source with rho_old %.6g, with rho_new %.6g (ratio %.1f)\n",
                    (double)M.source[flipped].x, (double)wrong,
                    (double)(wrong / M.source[flipped].x));
        check("the two differ by the density ratio, so this arm discriminates",
              std::fabs(wrong / M.source[flipped].x - kRhoWater / kRhoAir) < scalar(1e-9));
        check("...and the implementation takes the OLD one",
              std::fabs(M.source[flipped].x - wrong) > scalar(1));

        // ...and where the phase did NOT change the two agree, which is why a smooth gate cannot see it
        label same = -1;
        for (label c = 0; c < nC && same < 0; ++c)
            if (rho[c] == rhoOld[c] && rho[c] == kRhoWater) same = c;
        checkNum("away from the interface both rhos agree",
                 M.source[same].x, kRhoWater * scalar(0.3) * V[same] / dt);

        // deltaT <= 0 is refused: interFoam has no steady path.
        bool threw = false;
        try
        {
            FvVectorMatrix Z;
            Z.diag.assign(static_cast<std::size_t>(nC), scalar(0));
            Z.source.assign(static_cast<std::size_t>(nC), vector{0, 0, 0});
            addEulerDdtRhoU(Z, rho, rhoOld, UOld, V, scalar(0));
        }
        catch (const std::exception&) { threw = true; }
        check("deltaT = 0 is refused", threw);
    }

    // ---- 2. the momentum source face flux: the three terms and their signs ------------------------
    // (surfaceTensionForce - ghf*snGrad(rho) - snGrad(p_rgh)) * magSf, with four distinct values so a
    // transposition cannot hide.
    {
        const std::vector<scalar> stf  {scalar(2.0)};
        const std::vector<scalar> ghf  {scalar(5.0)};
        const std::vector<scalar> sgR  {scalar(7.0)};
        const std::vector<scalar> sgP  {scalar(11.0)};
        const std::vector<scalar> aSf  {scalar(3.0)};
        std::vector<scalar> out;
        momentumSourceFlux(stf, ghf, sgR, sgP, aSf, out);
        checkNum("(stf - ghf*snGrad(rho) - snGrad(p_rgh))*magSf",
                 out[0], (scalar(2.0) - scalar(5.0)*scalar(7.0) - scalar(11.0)) * scalar(3.0));
        check("surface tension enters with a PLUS",
              out[0] != (-scalar(2.0) - scalar(5.0)*scalar(7.0) - scalar(11.0)) * scalar(3.0));
        check("gravity enters with a MINUS",
              out[0] != (scalar(2.0) + scalar(5.0)*scalar(7.0) - scalar(11.0)) * scalar(3.0));
        check("and magSf multiplies the WHOLE bracket, not one term",
              out[0] != scalar(2.0)*scalar(3.0) - scalar(5.0)*scalar(7.0) - scalar(11.0));
    }

    // ---- 3. THE IDENTITY: reconstruct(-snGrad(p_rgh)*magSf) == -grad(p_rgh), exactly ---------------
    // p_rgh linear, so snGrad is exact on internal AND boundary faces (a fixedValue patch's snGrad is
    // deltaCoeffs*(value - psi_c), which for a linear field is the true normal derivative). reconstruct
    // then has to return the gradient itself. This is the arm that pins the sign, the magSf factor and
    // the same-sign surfaceSum simultaneously.
    {
        const vector gp{scalar(3.0), scalar(-2.0), scalar(0.5)};
        auto lin = [&](const vector& x) { return gp.x*x.x + gp.y*x.y + gp.z*x.z + scalar(17.0); };

        GeometricField<scalar> pRgh;
        pRgh.internal.resize(nC);
        for (label c = 0; c < nC; ++c) pRgh.internal[c] = lin(C[c]);
        for (const FvPatch& q : fvp)
        {
            std::vector<scalar> vals(static_cast<std::size_t>(q.size));
            for (label i = 0; i < q.size; ++i) vals[i] = lin(q.Cf[i]);
            pRgh.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(
                q, false, scalar(0), vals));
        }
        pRgh.evaluateBoundary();

        const SurfaceScalarField sg = fvc::snGrad(pRgh, m, g, fvp, /*corrected=*/false);

        SurfaceScalarField flux;
        flux.internal.resize(sg.internal.size());
        for (std::size_t f = 0; f < sg.internal.size(); ++f)
            flux.internal[f] = -sg.internal[f] * g.magSf()[f];
        flux.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            flux.boundary[pi].resize(sg.boundary[pi].size());
            for (std::size_t i = 0; i < sg.boundary[pi].size(); ++i)
                flux.boundary[pi][i] = -sg.boundary[pi][i] * fvp[pi].magSf[i];
        }

        const std::vector<vector> R = reconstruct(flux, m, g, fvp);
        scalar worst = 0;
        for (label c = 0; c < nC; ++c)
            worst = std::fmax(worst, std::fmax(std::fabs(R[c].x + gp.x),
                              std::fmax(std::fabs(R[c].y + gp.y), std::fabs(R[c].z + gp.z))));
        std::printf("  worst |reconstruct(-snGrad(p_rgh)*magSf) + grad(p_rgh)| = %.3e\n", (double)worst);
        check("the identity holds on every cell", worst <= scalar(1e-12));

        // CONTROL A: drop magSf. This is why the fixture is 2 x 1 x 0.5 rather than a cube: with equal
        // areas the omission is a uniform rescaling that this identity cannot see at all.
        SurfaceScalarField noArea = flux;
        for (std::size_t f = 0; f < noArea.internal.size(); ++f)
            noArea.internal[f] = -sg.internal[f];
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (std::size_t i = 0; i < noArea.boundary[pi].size(); ++i)
                noArea.boundary[pi][i] = -sg.boundary[pi][i];
        const std::vector<vector> Rn = reconstruct(noArea, m, g, fvp);
        check("dropping magSf changes the answer (control)",
              std::fabs(Rn[0].x + gp.x) > scalar(1e-3));

        // CONTROL B: the sign. Getting it backwards is the transcription error the rhoSimpleFoam call
        // site invites, and it is a solution, not a divergence.
        SurfaceScalarField flipped = flux;
        for (scalar& s : flipped.internal) s = -s;
        for (std::vector<scalar>& b : flipped.boundary) for (scalar& s : b) s = -s;
        const std::vector<vector> Rf = reconstruct(flipped, m, g, fvp);
        checkNum("flipping the sign flips the reconstruction", Rf[0].x, -R[0].x);
        check("...so the two are distinguishable", std::fabs(Rf[0].x - R[0].x) > scalar(1));
    }

    // ---- 4. `solve(UEqn == R)` adds V*R to the source, with a PLUS -------------------------------
    {
        SurfaceScalarField flux;
        flux.internal.assign(static_cast<std::size_t>(m.nInternalFaces()), scalar(0));
        flux.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            flux.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        // a uniform vector's flux: reconstruct returns the vector itself (the defining identity)
        const vector F{scalar(1.5), scalar(-0.25), scalar(4.0)};
        for (label f = 0; f < m.nInternalFaces(); ++f)
        {
            const vector& Sf = g.Sf()[f];
            flux.internal[f] = F.x*Sf.x + F.y*Sf.y + F.z*Sf.z;
        }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const vector& Sf = g.Sf()[fvp[pi].start + i];
                flux.boundary[pi][i] = F.x*Sf.x + F.y*Sf.y + F.z*Sf.z;
            }

        FvVectorMatrix M;
        M.diag.assign(static_cast<std::size_t>(nC), scalar(0));
        M.source.assign(static_cast<std::size_t>(nC), vector{scalar(100), scalar(0), scalar(0)});
        addMomentumPredictorSource(M, flux, m, g, fvp);
        checkNum("source += V*R, x", M.source[0].x, scalar(100) + F.x * V[0]);
        checkNum("source += V*R, y", M.source[0].y,                F.y * V[0]);
        // The pre-existing source was (100, 0, 0), so each component moved by exactly +V*R. A minus
        // would move it by -V*R, and F's components have three different signs so no single flip hides.
        check("...a PLUS: every component moved in R's direction, not against it",
              M.source[0].x - scalar(100) > scalar(0) && M.source[0].y < scalar(0) && M.source[0].z > scalar(0));
        checkNum("source += V*R, z", M.source[0].z, F.z * V[0]);
    }

    // ---- 5. the stress term takes rho*nuEff, not nuEff ------------------------------------------
    // Assembled both ways on the same fields; on water they differ by 1000. The arm is the DIFFERENCE,
    // not the value: the operator itself is gated against OpenFOAM by rho_ueqn_vs_openfoam.sh, and what
    // is under test here is which viscosity interFoam hands it.
    {
        GeometricField<vector> U;
        U.internal.resize(nC);
        for (label c = 0; c < nC; ++c) U.internal[c] = vector{scalar(0.1)*C[c].y, scalar(0), scalar(0)};
        for (const FvPatch& q : fvp)
            U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                q, true, vector{0, 0, 0}, std::vector<vector>{}));
        U.evaluateBoundary();

        std::vector<scalar> rho(nC, kRhoWater), rhoOld(nC, kRhoWater), nuEff(nC, scalar(1e-6));
        std::vector<vector> UOld(nC, vector{0, 0, 0});
        std::vector<std::vector<scalar>> rhoBnd(fvp.size()), nuEffBnd(fvp.size()), phiBnd(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            rhoBnd[pi].assign(static_cast<std::size_t>(fvp[pi].size), kRhoWater);
            nuEffBnd[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(1e-6));
            phiBnd[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        }
        std::vector<scalar> rhoPhi(static_cast<std::size_t>(m.nInternalFaces()), scalar(0));

        InterMomentumInput in;
        in.rhoPhi = &rhoPhi;  in.rhoPhiBnd = &phiBnd;
        in.rho = &rho;        in.rhoOld = &rhoOld;   in.rhoBnd = &rhoBnd;
        in.UOld = &UOld;
        in.nuEff = &nuEff;    in.nuEffBnd = &nuEffBnd;
        in.deltaT = scalar(0.001);

        const FvVectorMatrix Mrho = assembleUEqn(U, in, m, g, fvp);

        // the incompressible lineage: nuEff handed straight to the operator
        InterMomentumInput inc = in;
        inc.muEff    = &nuEff;
        inc.muEffBnd = &nuEffBnd;
        const FvVectorMatrix Mnu = assembleUEqn(U, inc, m, g, fvp);

        scalar dmax = 0;
        for (label f = 0; f < m.nInternalFaces(); ++f)
            dmax = std::fmax(dmax, std::fabs(Mrho.upper[f] - Mnu.upper[f]));
        std::printf("  worst |upper(rho*nuEff) - upper(nuEff)| = %.6g\n", (double)dmax);
        check("the two viscosities give different matrices, so this arm discriminates", dmax > scalar(0));
        // and by the density, to round-off: the laplacian coefficient is linear in the viscosity, and
        // the ddt term contributes to the diagonal only, so the off-diagonals scale exactly by rho.
        scalar worstRatio = 0;
        for (label f = 0; f < m.nInternalFaces(); ++f)
            if (Mnu.upper[f] != scalar(0))
                worstRatio = std::fmax(worstRatio,
                    std::fabs(Mrho.upper[f] / Mnu.upper[f] - kRhoWater) / kRhoWater);
        checkNum("...and they differ by exactly rho", worstRatio, scalar(0), scalar(1e-12));
    }

    // ---- 6. refusals, and the control ------------------------------------------------------------
    {
        std::vector<scalar> rho(nC, kRhoWater), rhoOld(nC, kRhoWater), nuEff(nC, scalar(1e-6));
        std::vector<vector> UOld(nC, vector{0, 0, 0});
        std::vector<std::vector<scalar>> rhoBnd(fvp.size()), nuEffBnd(fvp.size()), phiBnd(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            rhoBnd[pi].assign(static_cast<std::size_t>(fvp[pi].size), kRhoWater);
            nuEffBnd[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(1e-6));
            phiBnd[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        }
        std::vector<scalar> rhoPhi(static_cast<std::size_t>(m.nInternalFaces()), scalar(0));

        GeometricField<vector> U;
        U.internal.assign(static_cast<std::size_t>(nC), vector{0, 0, 0});
        for (const FvPatch& q : fvp)
            U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                q, true, vector{0, 0, 0}, std::vector<vector>{}));
        U.evaluateBoundary();

        InterMomentumInput base;
        base.rhoPhi = &rhoPhi; base.rhoPhiBnd = &phiBnd;
        base.rho = &rho;       base.rhoOld = &rhoOld;  base.rhoBnd = &rhoBnd;
        base.UOld = &UOld;
        base.nuEff = &nuEff;   base.nuEffBnd = &nuEffBnd;
        base.deltaT = scalar(0.001);

        auto refuses = [&](InterMomentumInput in)
        {
            try { (void)assembleUEqn(U, in, m, g, fvp); } catch (const std::exception&) { return true; }
            return false;
        };

        InterMomentumInput cn = base;  cn.ddtScheme = DdtScheme::CrankNicolson;
        check("CrankNicolson ddt is refused by name", refuses(cn));
        InterMomentumInput lts = base; lts.ddtScheme = DdtScheme::localEuler;
        check("localEuler (LTS) ddt is refused by name", refuses(lts));
        InterMomentumInput mrf = base; mrf.hasMRF = true;
        check("a case declaring MRF is refused", refuses(mrf));
        InterMomentumInput opt = base; opt.hasFvOptions = true; opt.fvOptionUnsupported = "momentumSource";
        check("a case declaring fvOptions is refused", refuses(opt));
        InterMomentumInput noOld = base; noOld.rhoOld = nullptr;
        check("rho.oldTime() missing is refused, not silently replaced by rho", refuses(noOld));
        // `Gauss limitedLinear` on div(rhoPhi,U) is ported (tests/interfoam_limitedlinear_vs_openfoam.sh
        // holds it against OpenFOAM). Here: it assembles, and not as upwind. That needs a flux and a
        // U that varies -- on the zero U and zero flux above the two matrices are the same one.
        {
            const std::vector<vector>& Cc = g.C();
            GeometricField<vector> Uv;
            Uv.internal.resize(static_cast<std::size_t>(nC));
            for (label c = 0; c < nC; ++c)
            {
                Uv.internal[c] = vector{Cc[c].x*Cc[c].x, Cc[c].y, 0};
            }
            for (const FvPatch& q : fvp)
            {
                Uv.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                    q, true, vector{0, 0, 0}, std::vector<vector>{}));
            }
            Uv.evaluateBoundary();
            std::vector<scalar> flux(static_cast<std::size_t>(m.nInternalFaces()), scalar(1e-3));
            InterMomentumInput up = base;
            up.rhoPhi = &flux;
            up.scheme = DivScheme::upwind;
            InterMomentumInput ll = up;
            ll.scheme = DivScheme::limitedLinear;
            ll.schemeCoeff = scalar(0.2);
            bool ran = true;
            FvVectorMatrix Mu;
            FvVectorMatrix Ml;
            try
            {
                Mu = assembleUEqn(Uv, up, m, g, fvp);
                Ml = assembleUEqn(Uv, ll, m, g, fvp);
            }
            catch (const std::exception& e)
            {
                std::printf("    threw: %s\n", e.what());
                ran = false;
            }
            check("`Gauss limitedLinear` on div(rhoPhi,U) assembles", ran);
            scalar dUpper = 0;
            for (std::size_t f = 0; ran && f < Ml.upper.size(); ++f)
            {
                dUpper = std::fmax(dUpper, std::fabs(Ml.upper[f] - Mu.upper[f]));
            }
            std::printf("    limitedLinear 0.2 against upwind, largest upper-coefficient difference %.3e\n", dUpper);
            check("...and not as upwind", dUpper > scalar(1e-6));
        }

        // CONTROL: the ordinary damBreak configuration must assemble, or the refusals prove nothing.
        InterMomentumInput ok = base;
        ok.scheme         = DivScheme::linearUpwind;   // damBreak: `Gauss linearUpwind grad(U)`
        ok.relaxEquationU = true;                      // damBreak: `equations { ".*" 1; }`
        ok.relaxU         = scalar(1);
        check("damBreak's own configuration still assembles", !refuses(ok));
    }

    std::printf("test_inter_ueqn_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
