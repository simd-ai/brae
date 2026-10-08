// The coupled cyclic patch field and porousBafflePressure's jump, face by face.
//
// WHY A UNIT TEST, when tests/interfoam_baffle_vs_openfoam.sh holds both against OpenFOAM: on
// RAS/damBreakPorousBaffle the baffle is `uniformJump true` and the flow crosses it ONE way in every gated
// run, so the per-face jump, the reversed flow and sign()'s value at zero are reached by no case. What is
// asserted here is OpenFOAM's own text (porousBafflePressureFvPatchField.C:125-189,
// jumpCyclicFvPatchField.C patchNeighbourField, fixedJumpFvPatchField.C jump(), coupledFvPatchField.C
// evaluate and snGrad):
//
//     Un   = phi_p/magSf                       and gAverage(Un) on every face under uniformJump
//     jump = -sign(Un)*(D*nu_p + I*0.5*|Un|)*|Un|*length * rho_p           sign(0) = +1
//     pnf  = psi[nbr] - jump on the OWNER side, psi[nbr] + jump on the other
//     value = w*psi[own] + (1 - w)*pnf;   snGrad = deltaCoeffs*(pnf - psi[own])
//
//   LEG 1  a plain coupled patch: value and snGrad from the two cells, live from the field handed in
//   LEG 2  the per-face jump, flow out of the owner: negative, and rho_p multiplies it
//   LEG 3  reversed flow gives the opposite sign; zero flow gives exactly zero
//   LEG 4  uniformJump: every face takes the AVERAGE Un, so faces with opposite fluxes share one sign
//   LEG 5  the owner SUBTRACTS its jump from the neighbour cell and the other side ADDS it, so the two
//          sides' face values differ by exactly the jump times the weights' complement
//   LEG 6  CONTROL: the kinematic form (no rho_p) differs wherever rho_p is not 1, so leg 2 can fail
//
// BROKEN ONCE EACH: the other side not negating the jump fails all three LEG 5 lines; the jump without
// rho_p fails LEG 2, LEG 4 and the control; uniformJump ignored fails both LEG 4 lines.
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

// two patches of n faces facing each other: cells 0..n-1 on the owner side, n..2n-1 on the other
FvPatch makeHalf(
    label n,
    bool owner)
{
    FvPatch p;
    p.name = owner ? "half0" : "half1";
    p.type = "cyclic";
    p.size = n;
    p.coupled = true;
    p.owner = owner;
    p.nbrPatch = owner ? 1 : 0;
    for (label i = 0; i < n; ++i)
    {
        p.faceCells.push_back(owner ? i : n + i);
        p.nbrFaceCells.push_back(owner ? n + i : i);
        p.nf.push_back(owner ? vector{1, 0, 0} : vector{-1, 0, 0});
        p.magSf.push_back(scalar(0.5));
        p.Cf.push_back(vector{0, scalar(i), 0});
        // NOT one half: the weight is this side's, and the other side's is its complement
        p.weights.push_back(owner ? scalar(0.25) : scalar(0.75));
        p.deltaCoeffs.push_back(scalar(4));
        p.nonOrthDeltaCoeffs.push_back(scalar(4));
        p.delta.push_back(owner ? vector{0.25, 0, 0} : vector{-0.25, 0, 0});
        p.nonOrthCorrectionVectors.push_back(vector{0, 0, 0});
    }
    return p;
}
}   // namespace

int main()
{
    const label n = 4;
    const FvPatch own = makeHalf(n, true);
    const FvPatch nbr = makeHalf(n, false);
    const std::vector<scalar> psi = {10, 20, 30, 40, 1, 2, 3, 4};

    std::printf("== a plain coupled cyclic ==\n");
    {
        CoupledCyclicPatchField<scalar> f(own);
        check("LEG 1  it is coupled, fixes no value and carries no jump",
              f.coupled() && !f.fixesValue() && f.coupledJump() == nullptr);
        f.evaluate(psi);
        bool value = true;
        bool sn = true;
        const std::vector<scalar> g = f.snGrad(psi);
        for (label i = 0; i < n; ++i)
        {
            value = value && f.value()[i] == scalar(0.25)*psi[i] + scalar(0.75)*psi[n + i];
            sn = sn && g[i] == scalar(4)*(psi[n + i] - psi[i]);
        }
        check("LEG 1  value = w*own + (1 - w)*nbr, with THIS side's weight", value);
        check("LEG 1  snGrad = deltaCoeffs*(nbr - own)", sn);
        std::vector<scalar> moved = psi;
        moved[n] = scalar(100);
        check("LEG 1  patchNeighbourField is LIVE: it follows the field it is handed",
              f.patchNeighbourField(moved)[0] == scalar(100) && f.patchNeighbourField(psi)[0] == scalar(1));
    }

    std::printf("== porousBafflePressure ==\n");
    const scalar D = 1000;
    const scalar I = 500;
    const scalar L = 0.15;
    const std::vector<scalar> nup = {1e-6, 1e-6, 1.48e-5, 1.48e-5};
    const std::vector<scalar> rhop = {1000, 1000, 1, 1};
    auto formula = [&](scalar Un, std::size_t i, bool withRho)
    {
        const scalar mu = std::fabs(Un);
        const scalar sgn = (Un >= scalar(0)) ? scalar(1) : scalar(-1);
        const scalar j = -sgn*(D*nup[i] + I*scalar(0.5)*mu)*mu*L;
        return withRho ? j*rhop[i] : j;
    };
    {
        PorousBafflePressurePatchField perFace(own, D, I, L, /*uniformJump=*/false, std::vector<scalar>(4, scalar(0)));
        // out of the owner, out, INTO the owner, and exactly nothing
        const std::vector<scalar> phip = {0.1, 0.2, -0.05, 0.0};
        const std::vector<scalar> j = perFace.porousBaffleJump(phip, nup, rhop);
        check("LEG 2  flow out of the owner gives a NEGATIVE jump, with rho_p in it",
              j[0] == formula(scalar(0.2), 0, true) && j[0] < scalar(0) && j[1] == formula(scalar(0.4), 1, true));
        check("LEG 3  reversed flow gives a POSITIVE jump", j[2] == formula(scalar(-0.1), 2, true) && j[2] > scalar(0));
        check("LEG 3  no flow gives exactly zero", j[3] == scalar(0));
        check("LEG 6  CONTROL: without rho_p the water faces differ a thousandfold, so leg 2 can fail",
              formula(scalar(0.2), 0, false) != j[0] && std::fabs(j[0]/formula(scalar(0.2), 0, false) - scalar(1000)) < 1e-9);

        PorousBafflePressurePatchField uniform(own, D, I, L, /*uniformJump=*/true, std::vector<scalar>(4, scalar(0)));
        const std::vector<scalar> ju = uniform.porousBaffleJump(phip, nup, rhop);
        const scalar avg = (scalar(0.2) + scalar(0.4) + scalar(-0.1) + scalar(0))/scalar(4);
        bool allAvg = true;
        for (std::size_t i = 0; i < 4; ++i)
        {
            allAvg = allAvg && ju[i] == formula(avg, i, true);
        }
        check("LEG 4  uniformJump: every face takes the AVERAGE Un", allAvg);
        check("LEG 4  ...so the face the flow ENTERS through takes the average's sign, not its own",
              ju[2] < scalar(0) && j[2] > scalar(0));
    }
    {
        PorousBafflePressurePatchField a(own, D, I, L, true, std::vector<scalar>(4, scalar(0)));
        PorousBafflePressurePatchField b(nbr, D, I, L, true, std::vector<scalar>(4, scalar(0)));
        const std::vector<scalar> ownerJump = {-7, -7, -7, -7};
        a.setOwnerJump(ownerJump);
        b.setOwnerJump(ownerJump);
        check("LEG 5  the owner holds the jump and the other side its negative",
              (*a.coupledJump())[0] == scalar(-7) && (*b.coupledJump())[0] == scalar(7));
        const std::vector<scalar> pa = a.patchNeighbourField(psi);
        const std::vector<scalar> pb = b.patchNeighbourField(psi);
        check("LEG 5  the owner SUBTRACTS it from the neighbour cell, the other side ADDS it",
              pa[0] == psi[n] - scalar(-7) && pb[0] == psi[0] + scalar(-7));
        a.evaluate(psi);
        b.evaluate(psi);
        // owner: 0.25*10 + 0.75*(1 + 7) = 8.5;  other: 0.75*1 + 0.25*(10 - 7) = 1.5;  apart by the jump
        std::printf("  the two sides' face values: %.3f and %.3f\n", (double)a.value()[0], (double)b.value()[0]);
        check("LEG 5  the two sides' face values are the jump apart", a.value()[0] - b.value()[0] == scalar(7));
    }

    std::printf("test_porous_baffle_pressure: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
