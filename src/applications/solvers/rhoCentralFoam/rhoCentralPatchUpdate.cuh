#pragma once
// One boundary face's thermo-coupled update in a rhoCentralFoam step, for both arms (BRAE_HD). The patch values of U
// (232) and of T as the energy BCs' Tw.evaluate() sees it (265) come in already evaluated through brae's patch
// fields; this does the rest, in OpenFOAM's order:
//
//   rhoU.boundaryFieldRef() == rho.boundaryField()*U.boundaryField()          :233  rho_b is still last step's
//   e.correctBoundaryConditions()                                              :265
//       fixedEnergy (T fixesValue):      e_b = he(p_b, T_b)
//       gradientEnergy (zeroGradient T): Cpv*snGrad(T) + delta*(he(p_b,T_b) - he(p_b,T_b)) is exactly 0: e_b = e_P
//   thermo.correct(), patch half (hePsiThermo.C:97-137)                        :266
//       T fixesValue: e_b = HE(p_b, T_b), T_b kept;  else T_b = THE(e_b, p_b, T_b), seeded by Tw.evaluate()'s T_b
//       psi_b = psi(p_b, T_b); p_b is still last step's
//   rhoE.boundaryFieldRef() == rho.boundaryField()*(e_b + 0.5*magSqr(U_b))     :267-271
// p's patches (287) and rho_b = psi_b*p_b (288) follow, after p.
#include "cf_types.cuh"
#include "rhoCentralThermo.cuh"

namespace brae {
namespace rhoCentral {

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
        b.status = r.status;
        b.T = r.T;
    }
    b.psi = gasPsi(pOld, b.T, gas);
    b.rhoE = rhoOld*(b.e + 0.5*(Ux*Ux + Uy*Uy + Uz*Uz));
    return b;
}

} // namespace rhoCentral
} // namespace brae
