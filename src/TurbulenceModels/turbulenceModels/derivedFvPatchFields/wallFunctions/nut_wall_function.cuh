#pragma once
// nutkWallFunction (OpenFOAM v2412, STEPWISE blender), turbulent viscosity nut at a wall.
//   yPlus = Cmu^0.25 * y * sqrt(k_nearWall) / nu       (y = near-wall distance)
//   nut   = (yPlus > yPlusLam) ? nu*yPlus*kappa/log(max(E*yPlus, 1+1e-4)) - nu : 0
// y is the near-wall distance (OF turbModel.y() = wallDist field), passed in by the caller.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include <vector>

namespace brae {

// Single source of truth for the OF nutkWallFunction value (the log-law wall viscosity, 0 in the viscous sublayer).
// Shared by the host nutkWallFunction below AND the device wall kernels (kEpsilon wallFnKernel/boundaryNutKernel,
// kOmegaSST wallOmegaG0Kernel) so the wall-nut physics has ONE definition, not four copies. BRAE_HD (__host__
// __device__) so it compiles identically on both; log/fmax resolve to the device intrinsics on the GPU and libm on
// the host (same result for double). yPlus = Cmu^0.25 * y * sqrt(k_nearWall) / nu (see yPlusWall).
BRAE_HD inline scalar nutkWallFunctionValue(scalar yPlus, scalar nu, scalar yplLam, scalar kappa, scalar E)
{
    return (yPlus > yplLam) ? (nu * yPlus * kappa / log(fmax(E * yPlus, scalar(1.0 + 1e-4))) - nu) : scalar(0.0);
}
BRAE_HD inline scalar yPlusWall(scalar Cmu25, scalar y, scalar kNearWall, scalar nu) { return Cmu25 * y * sqrt(kNearWall) / nu; }

// atmNutkWallFunction (OpenFOAM atmosphericModels): atmospheric ROUGH-wall nut using surface roughness length z0.
// Same k-based friction velocity as nutk (yPlus = Cmu^0.25*sqrt(k)*y/nu) but the log uses the roughness blend
// Edash = (y+z0)/z0 instead of the smooth-wall E*yPlus, and there is no viscous-sublayer cut-off:
//   nut = nu*(yPlus*kappa/log(max(Edash, 1+1e-4)) - 1);  boundNut -> max(nut, 0)  (OF atmNutkWallFunction.C).
BRAE_HD inline scalar atmNutkWallFunctionValue(scalar yPlus, scalar y, scalar z0, scalar nu, scalar kappa, bool boundNut)
{
    const scalar Edash = (y + z0) / fmax(z0, scalar(1e-300));
    const scalar nutw  = nu * (yPlus * kappa / log(fmax(Edash, scalar(1.0 + 1e-4))) - scalar(1.0));
    return boundNut ? fmax(nutw, scalar(0.0)) : nutw;
}
// Wall nut for the k-based family: z0>0 selects atmNutkWallFunction (rough), else nutkWallFunction (smooth log law).
BRAE_HD inline scalar kBasedWallNut(scalar yPlus, scalar y, scalar z0, bool atmBoundNut, scalar nu, scalar yplLam, scalar kappa, scalar E)
{
    return (z0 > scalar(0.0)) ? atmNutkWallFunctionValue(yPlus, y, z0, nu, kappa, atmBoundNut)
                              : nutkWallFunctionValue(yPlus, nu, yplLam, kappa, E);
}

// Velocity-based wall nut, SINGLE source of truth shared by the momentum wall kernels (spaldingNutKernel/
// blendedNutKernel) AND the near-wall production G0, so the wall shear and the k-production use the SAME nutw
// (exactly as OpenFOAM). magUp = |U_cell - U_wall|, magGradU = |snGrad U| = magUp*deltaCoeffs; nutSeed warm-starts
// the iteration (pass 0 for a cold start). BRAE_HD: identical on host + device.
//
// nutUSpaldingWallFunction: Newton on Spalding's law (OF nutUSpaldingWallFunctionFvPatchScalarField, 10 iters, tol 0.01).
BRAE_HD inline scalar spaldingNutValue(scalar magUp, scalar magGradU, scalar y, scalar nu, scalar kappa, scalar E, scalar nutSeed)
{
    scalar ut = sqrt((nutSeed + nu) * magGradU);   // warm seed
    if (ut > scalar(1e-300) && magGradU > scalar(1e-300))
    {
        for (int it = 0; it < 10; ++it)
        {
            const scalar kUu = fmin(kappa*magUp/ut, scalar(50.0));
            const scalar fkUu = exp(kUu) - scalar(1.0) - kUu*(scalar(1.0) + scalar(0.5)*kUu);
            const scalar f  = -ut*y/nu + magUp/ut + (scalar(1.0)/E)*(fkUu - (scalar(1.0)/scalar(6.0))*kUu*kUu*kUu);
            const scalar df =  y/nu + magUp/(ut*ut) + (scalar(1.0)/E)*kUu*fkUu/ut;
            const scalar utNew = ut + f/df;
            const scalar err = fabs((ut - utNew) / ut);
            ut = utNew;
            if (!(ut > scalar(1e-300)) || err < scalar(0.01)) break;
        }
    }
    const scalar uTau = fmax(scalar(0.0), ut);
    return fmax(scalar(0.0), uTau*uTau / (magGradU + scalar(1e-300)) - nu);
}
// nutUWallFunction (OF nutUWallFunctionFvPatchScalarField, STEPWISE blender -- the default set by
// wallFunctionBlenders(dict, blenderType::STEPWISE, 4)).
//
// yPlus comes from a fixed-point iteration on the log law rather than a Newton solve for uTau:
//     kappaRe = kappa*magUp*y/nuw
//     yp <- (kappaRe + yp)/(1 + log(E*yp))     seeded at yPlusLam, tol 0.01/yPlusLam, at most 10 passes
// then
//     nutVis = nuw
//     nutLog = nuw*yPlus*kappa/log(max(E*yPlus, 1 + 1e-4))
//     nutw   = (yPlus > yPlusLam ? nutLog : nutVis) - nutVis
// The trailing subtraction is OF's `nutw -= nutVis`, so the viscous branch returns exactly 0 and the log
// branch returns the TURBULENT part only.
// yPlusLam: the fixed point of yPlusLam = log(E*yPlusLam)/kappa (OF wallFunctionCoefficients). One
// definition, shared by the nut wall functions and epsilonWallFunction's lowReCorrection branch.
scalar yPlusLam(scalar kappa, scalar E);

BRAE_HD inline scalar nutUWallValue(scalar magUp, scalar y, scalar nu, scalar kappa, scalar E, scalar yPlusLam)
{
    if (!(magUp > scalar(0)) || !(y > scalar(0)) || !(nu > scalar(0))) return scalar(0);
    const scalar kappaRe = kappa * magUp * y / nu;
    scalar yp = yPlusLam;
    const scalar ryPlusLam = scalar(1) / yPlusLam;
    for (int it = 0; it < 10; ++it)
    {
        const scalar last = yp;
        yp = (kappaRe + yp) / (scalar(1) + log(fmax(E * yp, scalar(1e-300))));
        if (fabs(ryPlusLam * (yp - last)) <= scalar(0.01)) break;
    }
    const scalar yPlus = fmax(scalar(0), yp);
    if (yPlus <= yPlusLam) return scalar(0);                       // viscous branch: nutVis - nutVis
    const scalar nutLog = nu * yPlus * kappa / log(fmax(E * yPlus, scalar(1) + scalar(1e-4)));
    return fmax(scalar(0), nutLog - nu);
}

// nutUBlendedWallFunction: binomial n=4 blend of viscous/log velocity scales (OF, 10 iters, tol 1e-3, under-relaxed).
BRAE_HD inline scalar blendedNutValue(scalar magUp, scalar magGradU, scalar y, scalar nu, scalar kappa, scalar E, scalar nutSeed)
{
    scalar ut = sqrt((nutSeed + nu) * magGradU);   // warm seed
    if (ut > scalar(1e-300) && magGradU > scalar(1e-300) && magUp > scalar(1e-300))
    {
        scalar err = scalar(1e30);
        for (int it = 0; it < 10 && err > scalar(1e-3); ++it)
        {
            const scalar yPlus   = fmax(y * ut / nu, scalar(1e-300));
            const scalar uTauVis = magUp / yPlus;
            const scalar uTauLog = kappa * magUp / log(fmax(E * yPlus, scalar(1.0 + 1e-4)));
            const scalar v2 = uTauVis * uTauVis, l2 = uTauLog * uTauLog;
            const scalar utNew = sqrt(sqrt(v2*v2 + l2*l2));   // (uTauVis^4 + uTauLog^4)^(1/4)
            err = fabs(ut - utNew) / (ut + scalar(1e-300));
            ut  = scalar(0.5) * (ut + utNew);
        }
    }
    const scalar uTau = fmax(scalar(0.0), ut);
    return fmax(scalar(0.0), uTau*uTau / (magGradU + scalar(1e-300)) - nu);
}

// Shared near-wall production for the kEpsilon and kOmegaSST wall functions: compute nutw + |grad(U)|_wall and
// RETURN this face's turbulence-production contribution. The term is IDENTICAL for both models (this enforces
// that by construction, the komega wall kernel used to carry a hand-copy); each caller then adds only its
// distinct eps0 / omega0 term.
//
// It used to atomicAdd straight into G0[c]. It returns instead, so the caller -- which now owns a whole cell
// and iterates that cell's wall faces in a fixed order -- accumulates in a register and writes once. Same
// arithmetic, no atomics, and the result no longer depends on face scheduling order.
// nutWall: 0 = nutkWallFunction (k-based), 1 = nutUSpalding, 2 = nutUBlended (velocity-based),
// 3 = nutUWallFunction, 4 = nutLowRe (exactly zero) -- MUST match the
// momentum wall-shear choice (ctl.nutWall) so the near-wall production uses the same nutw OpenFOAM does.
__device__ inline scalar wallProductionG0(
    int c,
    int wf,
    scalar y,
    scalar dc,
    scalar kc,
    scalar invNwC,
    const scalar* wux,
    const scalar* wuy,
    const scalar* wuz,
    const scalar* Ux,
    const scalar* Uy,
    const scalar* Uz,
    scalar nu,
    scalar yplLam,
    scalar Cmu25,
    scalar kappa,
    scalar E,
    scalar atmZ0,        // >0 -> atmNutkWallFunction (rough) for the k-based path; 0 -> nutkWallFunction (smooth)
    bool   atmBoundNut,
    int nutWall)
{
    const scalar dux = (wux[wf]-Ux[c])*dc, duy = (wuy[wf]-Uy[c])*dc, duz = (wuz[wf]-Uz[c])*dc;
    const scalar magG = sqrt(dux*dux + duy*duy + duz*duz);   // |snGrad U| = magGradU
    scalar nutw;
    if (nutWall == 4)                                        // nutLowReWallFunction: calcNut() is Zero,
        nutw = scalar(0.0);                                  // unconditionally (OF .C:38-42)
    else if (nutWall == 0)                                   // k-based: nutk (smooth) or atmNutk (rough, z0>0)
        nutw = kBasedWallNut(yPlusWall(Cmu25, y, kc, nu), y, atmZ0, atmBoundNut, nu, yplLam, kappa, E);
    else if (nutWall == 3)                                   // nutUWallFunction (log-law yPlus, stepwise)
    {
        const scalar magUp = magG / fmax(dc, scalar(1e-300));
        nutw = nutUWallValue(magUp, y, nu, kappa, E, yplLam);
    }
    else                                                     // velocity-based: magUp = |U_cell - U_wall| = magG/dc
    {
        const scalar magUp = magG / fmax(dc, scalar(1e-300));
        nutw = (nutWall == 1) ? spaldingNutValue(magUp, magG, y, nu, kappa, E, scalar(0.0))
                              : blendedNutValue (magUp, magG, y, nu, kappa, E, scalar(0.0));
    }
    return invNwC * (nutw + nu) * magG * Cmu25 * sqrt(kc) / (kappa * y);
}

std::vector<scalar> nutkWallFunction(
    const FvPatch& wall,
    const std::vector<scalar>& y,
    const std::vector<scalar>& kInternal,
    scalar nu,
    scalar Cmu = 0.09,
    scalar kappa = 0.41,
    scalar E = 9.8);

// PER-FACE nu. The incompressible lineage has one kinematic viscosity for the whole domain; the
// compressible one has mu(T)/rho, which differs face by face along a wall with a temperature gradient --
// and nutkWallFunction is written in terms of nu_w, the value AT the face, not a case constant. An
// overload rather than a changed signature so every existing caller keeps the scalar it already passes.
std::vector<scalar> nutkWallFunction(
    const FvPatch& wall,
    const std::vector<scalar>& y,
    const std::vector<scalar>& kInternal,
    const std::vector<scalar>& nuFace,
    scalar Cmu = 0.09,
    scalar kappa = 0.41,
    scalar E = 9.8);

} // namespace brae
