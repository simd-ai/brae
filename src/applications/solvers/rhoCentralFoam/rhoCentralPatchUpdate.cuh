#pragma once
// One boundary face's thermo-coupled update in a rhoCentralFoam step, for both arms (BRAE_HD). The patch values of U
// (232) and of T as the energy BCs' Tw.evaluate() sees it (265) come in already evaluated through brae's patch
// fields; this does the rest, in OpenFOAM's order:
//
//   rhoU.boundaryFieldRef() == rho.boundaryField()*U.boundaryField()          :233  rho_b is still last step's
//   e.correctBoundaryConditions()                                              :265
//       fixedEnergy (T fixesValue):      e_b = he(p_b, T_b)
//       gradientEnergy (zeroGradient T): Cpv*snGrad(T) + delta*(he(p_b,T_b) - he(p_b,T_b)) is exactly 0: e_b = e_P
//       mixedEnergy (T inletOutlet, milestone D): Tw.evaluate() switches on :195's phi and blends with the OLD cell
//                                        T; mixed fixesValue(), so thermo.correct() then sets e_b = HE(p_b, T_b) --
//                                        the fixedEnergy row's arithmetic, on that T_b
//   thermo.correct(), patch half (hePsiThermo.C:97-137)                        :266
//       T fixesValue: e_b = HE(p_b, T_b), T_b kept;  else T_b = THE(e_b, p_b, T_b), seeded by Tw.evaluate()'s T_b
//       psi_b = psi(p_b, T_b); p_b is still last step's
//   rhoE.boundaryFieldRef() == rho.boundaryField()*(e_b + 0.5*magSqr(U_b))     :267-271
// p's patches (287) and rho_b = psi_b*p_b (288) follow, after p.
//
// THE TRANSFORM PATCHES (milestone C). On a mesh patch of type symmetryPlane, symmetry or wedge EVERY field takes the
// constraint type -- rho, rhoU and rhoE too, though createFields.H builds them `calculated` (fvPatchFieldNew.C) -- so
// the three solves' correctBoundaryConditions() evaluate them, and two things change in the order above:
//   :224  solve(rho) ends in rho.correctBoundaryConditions(): rho_b = rho_P, the NEW cell value, so :233 and :267
//         multiply by it, not by last step's psi_b*p_b. The rhoU and rhoE evaluations are overwritten by :233/:267.
//   :265  T's patch field is the transform and so is he's (basicThermo::heBoundaryTypes keeps T's type when T is not
//         fixedValue, zeroGradient, fixedGradient or mixed): e_b = e_P with no Tw.evaluate(), and thermo.correct()
//         inverts T_b = THE(e_b, p_b, T_b) seeded with T_b as the last thermo.correct() left it.
// The second holds on any patch whose T is a transform, `slip` on a plain patch included; the first only on the
// constraint mesh patches. U's transform value is transformPatchU, below, in OpenFOAM's arithmetic.
#include "cf_types.cuh"
#include "rhoCentralState.cuh"
#include "rhoCentralThermo.cuh"

namespace brae {
namespace rhoCentral {

// U's transform patch fields, by what their evaluate() computes
enum PatchTransform : int
{
    transformNone   = 0,
    transformMirror = 1,   // symmetryPlane, symmetry, slip: (iF + transform(I - 2.0*sqr(nHat), iF))/2.0
    transformRotate = 2    // wedge: transform(faceT, iF)
};

// U.correctBoundaryConditions() on a transform face (rhoCentralFoam.C:232), from the face cell's new U:
//   mirror  symmetryPlaneFvPatchField.C:134-151 (nHat the patch's one normal), basicSymmetryFvPatchField.C:106-115
//           (nHat = patch().nf() per face; slip and symmetry), R = I - 2.0*sqr(nHat) precomputed per face
//   rotate  wedgeFvPatchField.C:118-131, R = wedgePolyPatch::faceT()
// transform(R, v) is R & v, row by row. NEGATIVE CONTROL controlNoTransform returns the cell value (zeroGradient).
BRAE_HD inline vector transformPatchU(int kind, const tensor& R, const vector& v, int ctrl)
{
    if (ctrl == controlNoTransform) return v;
    const vector Rv{R.xx*v.x + R.xy*v.y + R.xz*v.z, R.yx*v.x + R.yy*v.y + R.yz*v.z, R.zx*v.x + R.zy*v.y + R.zz*v.z};
    if (kind == transformRotate) return Rv;
    return vector{(v.x + Rv.x)/2.0, (v.y + Rv.y)/2.0, (v.z + Rv.z)/2.0};
}

// The rho_b that :233 and :267 multiply by: the new cell rho on a constraint patch, last step's psi_b*p_b elsewhere.
// NEGATIVE CONTROL controlLaggedRhoB keeps last step's everywhere.
BRAE_HD inline scalar patchRhoMid(int rhoConstraint, scalar rhoCell, scalar rhoB, int ctrl)
{
    return (rhoConstraint && ctrl != controlLaggedRhoB) ? rhoCell : rhoB;
}

struct PatchNew
{
    scalar rhoUx, rhoUy, rhoUz, e, T, psi, rhoE;
    int    status;
};

BRAE_HD inline PatchNew updatePatchFace(int TfixesValue, scalar rhoOld, scalar Ux, scalar Uy, scalar Uz,
                                        scalar eCell, scalar TbEval, scalar pOld, const GasCoeffs& gas)
{
    PatchNew b;
    b.status = 0;
    b.rhoUx = rhoOld*Ux;
    b.rhoUy = rhoOld*Uy;
    b.rhoUz = rhoOld*Uz;
    if (TfixesValue)
    {
        b.T = TbEval;
        b.e = gasEs(pOld, b.T, gas);
    }
    else
    {
        b.e = eCell;
        const GasTResult r = gasTEs(b.e, pOld, TbEval, gas);
        b.status = r.status | (r.limited ? statusJanafLimited : 0);
        b.T = r.T;
    }
    b.psi = gasPsi(pOld, b.T, gas);
    b.rhoE = rhoOld*(b.e + 0.5*(Ux*Ux + Uy*Uy + Uz*Uz));
    return b;
}

} // namespace rhoCentral
} // namespace brae
