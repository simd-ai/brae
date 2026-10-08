// interFoam's pEqn.H -- the four things in it that are NOT shared with every other pressure corrector.
//
// The laplacian, the solve and the non-orthogonal loop are machinery brae already has and gates
// elsewhere. What this file covers is what makes this pEqn interFoam's: where the pressure gradient
// went, how ddtCorr is weighted, how the velocity is rebuilt, and what happens to p_rgh afterwards.
//
// TWO OF THE FOUR ARE INVISIBLE ON AN EASY FIXTURE, and the arms below are built to make them visible:
//
//   * note 2 (interpolate(rho*rAU) vs interpolate(rho)*rAUf) needs rho to actually VARY across a face.
//     On a single-phase fixture rho is constant and the two forms agree exactly. The arm uses a 1000:1
//     water/air jump, which is the whole point of the solver.
//
//   * note 3 (the divide by rAUf inside reconstruct, the multiply by rAU outside) needs rAU to VARY.
//     On a uniform mesh with a uniform momentum diagonal the two forms are identically equal, so the
//     arm keeps that case as the identity it is and adds a non-uniform one beside it.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "fvc_reconstruct_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include <cmath>
#include <cstdio>
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

constexpr scalar kRhoWater = 1000.0;
constexpr scalar kRhoAir   = 1.0;
}   // namespace

int main()
{
    std::printf("== interFoam pEqn ==\n");

    const label N = 6;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N, scalar(0), scalar(2), scalar(1), scalar(0.5));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();

    // ---- 1. phig, and WHERE THE PRESSURE GRADIENT WENT ---------------------------------------------
    // The relationship between the two expressions is the assertion, not the absence of a term:
    //
    //     UEqn source / magSf   =  stf - ghf*snGrad(rho) - snGrad(p_rgh)
    //     phig       / (rAUf*magSf) =  stf - ghf*snGrad(rho)
    //
    // so phig/(rAUf*magSf) minus UEqn's bracket must be EXACTLY snGrad(p_rgh), on every face. That
    // pins which of the two carries the pressure gradient without either file having to claim it.
    {
        const std::vector<scalar> stf{scalar(2.0),  scalar(-1.5), scalar(0.25)};
        const std::vector<scalar> ghf{scalar(5.0),  scalar(3.0),  scalar(-2.0)};
        const std::vector<scalar> sgR{scalar(7.0),  scalar(-4.0), scalar(0.5)};
        const std::vector<scalar> sgP{scalar(11.0), scalar(6.0),  scalar(-3.0)};
        const std::vector<scalar> rAUf{scalar(0.3), scalar(1.7),  scalar(0.05)};
        const std::vector<scalar> aSf{scalar(3.0),  scalar(2.0),  scalar(9.0)};

        std::vector<scalar> phig;
        buoyancyFlux(stf, ghf, sgR, rAUf, aSf, phig);
        for (std::size_t f = 0; f < stf.size(); ++f)
            checkNum("phig = (stf - ghf*snGrad(rho))*rAUf*magSf",
                     phig[f], (stf[f] - ghf[f]*sgR[f]) * rAUf[f] * aSf[f]);

        scalar worst = 0;
        for (std::size_t f = 0; f < stf.size(); ++f)
        {
            const scalar ueqnBracket = stf[f] - ghf[f]*sgR[f] - sgP[f];   // UEqn.H:19-27
            const scalar peqnBracket = phig[f] / (rAUf[f] * aSf[f]);      // pEqn.H:28-34
            worst = std::fmax(worst, std::fabs((peqnBracket - ueqnBracket) - sgP[f]));
        }
        std::printf("  worst |phig/(rAUf*magSf) - UEqn bracket - snGrad(p_rgh)| = %.3e\n", (double)worst);
        check("the two differ by EXACTLY snGrad(p_rgh): explicit in UEqn, implicit here", worst <= scalar(1e-13));

        // CONTROL: had phig kept snGrad(p_rgh), the difference would be zero instead.
        const scalar withP = (stf[0] - ghf[0]*sgR[0] - sgP[0]) * rAUf[0] * aSf[0];
        check("...and a phig that kept the term would be a different number (control)",
              std::fabs(withP - phig[0]) > scalar(1e-6));
    }

    // ---- 2. ddtCorr's weighting: interpolate(rho*rAU), not interpolate(rho)*rAUf -------------------
    // The arm that needs the density ratio. Water on one side of the interface, air on the other.
    {
        // rAU IS 1/UEqn.A(), AND UEqn's DIAGONAL CARRIES rho/dt -- so rAU is of order dt/rho and jumps
        // across the interface with it, the other way up. That correlation is the point: rho*rAU is
        // nearly CONSTANT there while interpolate(rho)*interpolate(rAU) is not, because linear
        // interpolation does not commute with a product whose factors move in opposite directions.
        //
        // This arm was first written with rho varying in y and rAU in x. No single face saw both vary,
        // the two forms agreed to 1e-14, and the arm proved nothing. Face-crossing variation in BOTH
        // factors is what the test needs, and it is also what the physics does.
        const scalar dt = scalar(1e-3);
        std::vector<scalar> rho(static_cast<std::size_t>(nC)), rAU(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            rho[c] = (g.C()[c].y < scalar(3)) ? kRhoWater : kRhoAir;
            rAU[c] = dt / rho[c];
        }

        std::vector<scalar> product;
        rhoRAUf(rho, rAU, m, g, product);

        // the two separately-interpolated factors, computed here as the oracle's rival
        const std::vector<scalar>& w = g.weights();
        scalar worstGap = 0;
        label  worstFace = -1;
        for (label f = 0; f < nIf; ++f)
        {
            const label o = m.owner()[f], n = m.neighbour()[f];
            const scalar rhoF = w[f]*rho[o] + (scalar(1) - w[f])*rho[n];
            const scalar rAUF = w[f]*rAU[o] + (scalar(1) - w[f])*rAU[n];
            const scalar rival = rhoF * rAUF;
            checkNum("interpolate(rho*rAU) is the product interpolated",
                     product[f], w[f]*(rho[o]*rAU[o]) + (scalar(1) - w[f])*(rho[n]*rAU[n]));
            if (std::fabs(product[f] - rival) > worstGap)
            {
                worstGap = std::fabs(product[f] - rival);
                worstFace = f;
            }
            if (f > 2) break;   // the per-face assertion only needs a few; the sweep below is the point
        }
        worstGap = 0;
        for (label f = 0; f < nIf; ++f)
        {
            const label o = m.owner()[f], n = m.neighbour()[f];
            const scalar rival = (w[f]*rho[o] + (scalar(1) - w[f])*rho[n])
                               * (w[f]*rAU[o] + (scalar(1) - w[f])*rAU[n]);
            const scalar d = std::fabs(product[f] - rival);
            if (d > worstGap) { worstGap = d; worstFace = f; }
        }
        std::printf("  across the interface: worst |interp(rho*rAU) - interp(rho)*interp(rAU)| = %.4g"
                    " (face %d)\n", (double)worstGap, (int)worstFace);
        // The product is dt on BOTH sides, so interpolate(rho*rAU) is dt exactly. The rival is
        // (rho_w+rho_a)/2 * (rAU_w+rAU_a)/2, which for 1000:1 is about 250*dt -- two and a half orders
        // of magnitude, in the one face that decides where the interface goes.
        checkNum("interpolate(rho*rAU) across the interface is dt exactly", product[worstFace], dt);
        std::printf("  ...the rival form is %.1fx that\n",
                    (double)((worstGap + dt) / dt));
        check("the two forms genuinely disagree across the density jump, so this arm discriminates",
              worstGap > scalar(100) * dt);

        // ...and on a UNIFORM rho they agree, which is why a single-phase fixture cannot see this.
        const std::vector<scalar> rhoUniform(static_cast<std::size_t>(nC), kRhoWater);
        // (rAU still varies here -- it is the SECOND factor being constant that would be cheating)
        std::vector<scalar> uniformProduct;
        rhoRAUf(rhoUniform, rAU, m, g, uniformProduct);
        scalar uniformGap = 0;
        for (label f = 0; f < nIf; ++f)
        {
            const label o = m.owner()[f], n = m.neighbour()[f];
            const scalar rival = (w[f]*rhoUniform[o] + (scalar(1) - w[f])*rhoUniform[n])
                               * (w[f]*rAU[o] + (scalar(1) - w[f])*rAU[n]);
            uniformGap = std::fmax(uniformGap, std::fabs(uniformProduct[f] - rival));
        }
        check("...while on a single-phase field the two agree exactly, which is why this needs VoF",
              uniformGap <= scalar(1e-12));
    }

    // ---- 3. the velocity correction: divide by rAUf inside, multiply by rAU outside ---------------
    {
        std::vector<vector> HbyA(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) HbyA[c] = vector{scalar(0.1)*g.C()[c].x, scalar(0), scalar(0)};

        // faceFlux = phig - p_rghEqn.flux(), chosen so that (faceFlux/rAUf) is the flux of a UNIFORM
        // vector -- then reconstruct returns that vector exactly and the whole arm is an identity.
        const vector R{scalar(2.5), scalar(-1.25), scalar(0.75)};

        auto runWith = [&](const std::vector<scalar>& rAU)
        {
            std::vector<scalar> rAUf(static_cast<std::size_t>(nIf));
            const std::vector<scalar>& w = g.weights();
            for (label f = 0; f < nIf; ++f)
                rAUf[f] = w[f]*rAU[m.owner()[f]] + (scalar(1) - w[f])*rAU[m.neighbour()[f]];

            std::vector<scalar> faceFlux(static_cast<std::size_t>(nIf));
            for (label f = 0; f < nIf; ++f)
            {
                const vector& Sf = g.Sf()[f];
                faceFlux[f] = (R.x*Sf.x + R.y*Sf.y + R.z*Sf.z) * rAUf[f];
            }
            std::vector<std::vector<scalar>> ffB(fvp.size()), rB(fvp.size());
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                const FvPatch& q = fvp[pi];
                ffB[pi].resize(static_cast<std::size_t>(q.size));
                rB[pi].assign(static_cast<std::size_t>(q.size), scalar(1));
                for (label i = 0; i < q.size; ++i)
                {
                    const vector& Sf = g.Sf()[q.start + i];
                    // the boundary rAUf is the face cell's rAU (fvc::interpolate at an uncoupled patch)
                    rB[pi][i]  = rAU[q.faceCells[i]];
                    ffB[pi][i] = (R.x*Sf.x + R.y*Sf.y + R.z*Sf.z) * rB[pi][i];
                }
            }
            std::vector<vector> U;
            correctVelocity(HbyA, rAU, faceFlux, rAUf, ffB, rB, m, g, fvp, U);
            return U;
        };

        // (a) UNIFORM rAU: (faceFlux/rAUf) is exactly R & Sf, so reconstruct gives R and
        //     U = HbyA + rAU*R, exactly. This is the identity the whole expression rests on.
        const std::vector<scalar> rAUuniform(static_cast<std::size_t>(nC), scalar(0.4));
        const std::vector<vector> Uu = runWith(rAUuniform);
        scalar wu = 0;
        for (label c = 0; c < nC; ++c)
            wu = std::fmax(wu, std::fmax(std::fabs(Uu[c].x - (HbyA[c].x + scalar(0.4)*R.x)),
                 std::fmax(std::fabs(Uu[c].y - (HbyA[c].y + scalar(0.4)*R.y)),
                           std::fabs(Uu[c].z - (HbyA[c].z + scalar(0.4)*R.z)))));
        std::printf("  uniform rAU: worst |U - (HbyA + rAU*R)| = %.3e\n", (double)wu);
        check("with a uniform rAU the correction is exactly HbyA + rAU*R", wu <= scalar(1e-12));

        // (b) NON-UNIFORM rAU, and the rival form that a uniform fixture cannot rule out:
        //     U = HbyA + rAU*reconstruct(faceFlux) -- i.e. the division by rAUf dropped.
        std::vector<scalar> rAUvar(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) rAUvar[c] = scalar(0.2) + scalar(0.3)*g.C()[c].y;
        const std::vector<vector> Uv = runWith(rAUvar);

        std::vector<scalar> rAUfVar(static_cast<std::size_t>(nIf));
        const std::vector<scalar>& w = g.weights();
        for (label f = 0; f < nIf; ++f)
            rAUfVar[f] = w[f]*rAUvar[m.owner()[f]] + (scalar(1) - w[f])*rAUvar[m.neighbour()[f]];

        using namespace brae::cpu::fvcReconstruct;
        std::vector<tensor> T(static_cast<std::size_t>(nC), tensor{0,0,0,0,0,0,0,0,0});
        std::vector<vector> v(static_cast<std::size_t>(nC), vector{0,0,0});
        for (label f = 0; f < nIf; ++f)
        {
            const vector& Sf = g.Sf()[f];
            const scalar ff = (R.x*Sf.x + R.y*Sf.y + R.z*Sf.z) * rAUfVar[f];   // NOT divided
            accumulate(Sf, ff, T[m.owner()[f]],     v[m.owner()[f]]);
            accumulate(Sf, ff, T[m.neighbour()[f]], v[m.neighbour()[f]]);
        }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const FvPatch& q = fvp[pi];
            for (label i = 0; i < q.size; ++i)
            {
                const vector& Sf = g.Sf()[q.start + i];
                const label ci = q.faceCells[i];
                accumulate(Sf, (R.x*Sf.x + R.y*Sf.y + R.z*Sf.z) * rAUvar[ci], T[ci], v[ci]);
            }
        }
        scalar wv = 0;
        for (label c = 0; c < nC; ++c)
        {
            const vector r = dot(inv(T[c]), v[c]);
            wv = std::fmax(wv, std::fabs(Uv[c].x - (HbyA[c].x + rAUvar[c]*r.x)));
        }
        std::printf("  non-uniform rAU: worst |brae - the form without the /rAUf| = %.4g\n", (double)wv);
        check("with a VARYING rAU the two forms disagree, so the division is pinned", wv > scalar(1e-3));
    }

    // ---- 4. p = p_rgh + rho*gh, and the reference that moves BOTH fields ---------------------------
    {
        std::vector<scalar> p_rgh{scalar(120), scalar(90), scalar(75)};
        const std::vector<scalar> rho{kRhoWater, kRhoAir, kRhoWater};
        const std::vector<scalar> gh{scalar(-9.81), scalar(-4.0), scalar(-1.0)};

        std::vector<scalar> p;
        staticPressure(p_rgh, rho, gh, p);
        checkNum("p = p_rgh + rho*gh, water cell", p[0], scalar(120) + kRhoWater*scalar(-9.81));
        checkNum("p = p_rgh + rho*gh, air cell",   p[1], scalar(90)  + kRhoAir  *scalar(-4.0));

        const std::vector<scalar> pBefore = p, prghBefore = p_rgh;
        const label  refCell  = 1;
        const scalar refValue = scalar(0);
        applyPressureReference(p, p_rgh, rho, gh, refCell, refValue);

        checkNum("after the shift p at the reference cell IS the reference value", p[refCell], refValue);
        // ...and p_rgh was REBUILT, not left alone.
        scalar worstConsistency = 0, worstMove = 0;
        for (std::size_t c = 0; c < p.size(); ++c)
        {
            worstConsistency = std::fmax(worstConsistency, std::fabs(p[c] - (p_rgh[c] + rho[c]*gh[c])));
            worstMove = std::fmax(worstMove, std::fabs(p_rgh[c] - prghBefore[c]));
        }
        std::printf("  reference shift: worst |p - (p_rgh + rho*gh)| = %.3e, p_rgh moved by %.4g\n",
                    (double)worstConsistency, (double)worstMove);
        check("p and p_rgh are still consistent afterwards", worstConsistency <= scalar(1e-12));
        check("...because p_rgh was REBUILT from the shifted p, not left where the solve put it",
              worstMove > scalar(1));

        // CONTROL: stopping after shifting p alone leaves the two inconsistent by exactly the shift.
        const scalar shift = refValue - pBefore[refCell];
        scalar wrongConsistency = 0;
        for (std::size_t c = 0; c < p.size(); ++c)
            wrongConsistency = std::fmax(wrongConsistency,
                                         std::fabs(p[c] - (prghBefore[c] + rho[c]*gh[c])));
        std::printf("  ...had p_rgh been left alone, they would disagree by %.4g (the shift is %.4g)\n",
                    (double)wrongConsistency, (double)std::fabs(shift));
        check("the control differs by the shift, so this arm discriminates",
              std::fabs(wrongConsistency - std::fabs(shift)) < scalar(1e-9));

        bool threw = false;
        try { applyPressureReference(p, p_rgh, rho, gh, 99, refValue); }
        catch (const std::exception&) { threw = true; }
        check("a pRefCell outside the mesh is refused", threw);
    }

    std::printf("test_inter_peqn_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
