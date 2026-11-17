#pragma once
// One cell's share of a rhoCentralFoam step after the fluxes, rhoCentralFoam.C:222-286 (inviscid), for both arms
// (BRAE_HD), in OpenFOAM's four stages:
//
//   momentumStage   solve(fvm::ddt(rho) + fvc::div(phi))       :224   the diagonal solver: x = source/diag with the
//                   solve(fvm::ddt(rhoU) + fvc::div(phiUp))    :227   Euler matrix, diag = rDeltaT*V,
//                   U = rhoU/rho                               :229-231   source = rDeltaT*q0*V - V*div
//   energyStage     solve(fvm::ddt(rhoE) + fvc::div(phiEp))    :258-263   inviscid: div(sigmaDotU) adds V*0
//                   e = rhoE/rho - 0.5*magSqr(U)               :264
//   thermoStage     thermo.correct(), cell half                :266   T = THE(e, p_old, T_old), psi = psi(p_old, T)
//   pressureStage   p = rho/psi                                :284-286
// (EulerDdtScheme.C:381-392, fvMatrix.C:2283.) A component of rhoU knocked out by an empty patch is not solved
// (fvMatrixSolve.C:157-164) and keeps its value. Nothing here reads a neighbour or a patch.
//
// Why stages rather than one function: the host compiler contracts a*b + c into a fused multiply-add (g++ -O3 on
// aarch64 does), and which products it fuses depends on what one function body holds. The host arm runs each stage
// as a loop over the cells, OpenFOAM's field-at-a-time order; one fused body moved its results by one ulp (measured
// 2026-11-17: T in one Sod cell at step 5). The device arm runs the four stages back to back in one kernel.
#include "cf_types.cuh"
#include "rhoCentralState.cuh"
#include "rhoCentralThermo.cuh"

#include <cmath>

namespace brae {
namespace rhoCentral {

struct MomentumNew
{
    scalar rho, rhoUx, rhoUy, rhoUz, Ux, Uy, Uz;
};

// d0 = fvc::div(phi), d1..d3 = fvc::div(phiUp): surfaceIntegrate, already divided by V
BRAE_HD inline MomentumNew momentumStage(scalar rDeltaT, scalar Vc, scalar d0, scalar d1, scalar d2, scalar d3,
                                         int v0, int v1, int v2, scalar rho0, scalar rhoUx0, scalar rhoUy0,
                                         scalar rhoUz0)
{
    MomentumNew n;
    const scalar diag = rDeltaT*Vc;
    n.rho = (rDeltaT*rho0*Vc - Vc*d0)/diag;
    n.rhoUx = rhoUx0;
    n.rhoUy = rhoUy0;
    n.rhoUz = rhoUz0;
    if (v0) n.rhoUx = (rDeltaT*rhoUx0*Vc - Vc*d1)/diag;
    if (v1) n.rhoUy = (rDeltaT*rhoUy0*Vc - Vc*d2)/diag;
    if (v2) n.rhoUz = (rDeltaT*rhoUz0*Vc - Vc*d3)/diag;
    n.Ux = n.rhoUx/n.rho;
    n.Uy = n.rhoUy/n.rho;
    n.Uz = n.rhoUz/n.rho;
    return n;
}

struct EnergyNew
{
    scalar rhoE, e;
};

// d4 = fvc::div(phiEp), already divided by V
BRAE_HD inline EnergyNew energyStage(scalar rDeltaT, scalar Vc, scalar d4, scalar rhoE0, scalar rho, scalar Ux,
                                     scalar Uy, scalar Uz)
{
    EnergyNew n;
    const scalar diag = rDeltaT*Vc;
    n.rhoE = (rDeltaT*rhoE0*Vc - Vc*d4)/diag;
    n.e = n.rhoE/rho - 0.5*(Ux*Ux + Uy*Uy + Uz*Uz);
    return n;
}

struct ThermoNew
{
    scalar T, psi;
    int    status;
};

BRAE_HD inline ThermoNew thermoStage(scalar e, scalar pOld, scalar Told, const GasCoeffs& gas)
{
    ThermoNew n;
    const GasTResult r = gasTEs(e, pOld, Told, gas);
    n.T = r.T;
    n.psi = gasPsi(pOld, r.T, gas);
    n.status = r.status;
    return n;
}

BRAE_HD inline scalar pressureStage(scalar rho, scalar psi)
{
    return rho/psi;
}

// what OpenFOAM would stop on is the thermo's status; this adds what makes the state unphysical
BRAE_HD inline int physicalStatus(scalar rho, scalar p, scalar T, scalar e)
{
    return (!(rho > 0) || !(p > 0) || !(T > 0) || !isfinite(e)) ? statusNotPhysical : 0;
}

} // namespace rhoCentral
} // namespace brae
