// _cpp REFERENCE implementation -- see kOmegaSST_cpp.cuh for the OpenFOAM provenance and refusals.
#include "kOmegaSST_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "nut_wall_function.cuh"
#include "near_wall_dist.cuh"
#include "pbicgstab.cuh"
#include "limitedSchemes_cpp.cuh"
#include "bound_cpp.cuh"
#include "fvc.cuh"
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

namespace brae {
namespace cpu {
namespace kOmegaSST {

namespace {

namespace ls = limitedSchemes;

// gaussConvectionScheme with the SCHEME's weights. `limitedLinear <coeff>` limits on the transported
// scalar itself (NVDTVD on vf, not on magSqr as the vector form does), so vf and its gradient are the
// field being solved for.
inline scalar blend(scalar F1, scalar psi1, scalar psi2) { return F1 * (psi1 - psi2) + psi2; }

// DkEff/DomegaEff are volScalarFields in OpenFOAM -- alphaK(F1)*nut_ + nu() -- so their BOUNDARY comes
// from the operands' boundary fields, not from interpolating the cell value out to the face. nut's own
// boundary is what matters most: on a `calculated` patch OF carries the evaluated eddy viscosity, which
// at an inlet differs from the adjacent cell by more than 10x. Interpolating the cell value there is the
// same defect that put 90% of the kEpsilon epsilon residual on pitzDaily's inlet.
//
// F1 is likewise a volScalarField with its own boundary, and the blend on a patch face takes F1's PATCH
// value (F1Boundary) -- the owner cell's F1 was used until the inlet's 1% blend difference was found
// to seed kOmegaSST's iteration-2 residual on rhoSST.
SurfaceScalarField effectiveDiffusivity(
    const std::vector<scalar>&       DCell,
    const GeometricField<scalar>&    nutField,
    const std::vector<scalar>&       f1,
    scalar                           alpha1,
    scalar                           alpha2,
    scalar                           nu,
    const PrimitiveMesh&             m,
    const FvGeometry&                g,
    const std::vector<FvPatch>&      patches,
    // The compressible lineage multiplies the whole thing by rho -- fvm::laplacian(alpha*rho*DkEff(F1))
    // -- and carries a nu that varies with T. Null throughout is the incompressible reading and
    // reproduces the previous arithmetic exactly. The CELL PRODUCT is interpolated, as OpenFOAM does:
    // fvc::interpolate(rho*D), not interpolate(rho)*interpolate(D), which differ on non-uniform fields.
    const std::vector<scalar>*              rho    = nullptr,
    const std::vector<std::vector<scalar>>* rhoBnd = nullptr,
    const std::vector<std::vector<scalar>>* nuBnd  = nullptr,
    // F1 evaluated ON the patch faces (F1Boundary); null blends the owner cell's, the old arithmetic.
    const std::vector<std::vector<scalar>>* f1Bnd  = nullptr)
{
    static const bool dbg = std::getenv("BRAE_SST_DIFF_DEBUG") != nullptr;
    std::vector<scalar> DRho(DCell.size());
    for (std::size_t cc = 0; cc < DCell.size(); ++cc)
        DRho[cc] = DCell[cc] * (rho ? (*rho)[cc] : scalar(1.0));
    SurfaceScalarField sf = fvc::interpolate(DRho, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& nb = nutField.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            const scalar was = sf.boundary[pi][i];
            const scalar nuF = nuBnd ? (*nuBnd)[pi][i] : nu;
            const scalar f1Face = f1Bnd ? (*f1Bnd)[pi][i] : f1[c];
            sf.boundary[pi][i] = (blend(f1Face, alpha1, alpha2) * nb[i] + nuF)
                               * (rhoBnd ? (*rhoBnd)[pi][i] : scalar(1.0));
            if (dbg && i == 5)
                std::printf("      [Deff] patch %-12s face %d: interp %.6e -> nut_b %.6e (nut_b=%.3e)\n",
                            patches[pi].name.c_str(), i, was, sf.boundary[pi][i], nb[i]);
        }
    }
    return sf;
}

FvScalarMatrix divWithScheme(
    const SurfaceScalarField&        phi,
    const GeometricField<scalar>&    vf,
    bool                             limitedLinear,
    scalar                           limiterCoeff,
    const PrimitiveMesh&             m,
    const FvGeometry&                g,
    const std::vector<FvPatch>&      patches)
{
    if (!limitedLinear)
    {
        return fvm::div(phi.internal, phi.boundary, vf, m, patches);
    }

    std::vector<std::vector<scalar>> vfb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        vfb[pi] = vf.boundary[pi]->value();
    }
    const std::vector<vector> gradVf = fvc::gaussGrad(vf.internal, vfb, m, g, patches);
    return fvm::div(phi.internal, phi.boundary, vf,
                    ls::limitedLinearWeights(phi.internal, vf, gradVf, limiterCoeff, m, g),
                    m, patches);
}



} // namespace


std::vector<scalar> S2(const std::vector<tensor>& gradU)
{
    // 2*magSqr(symm(gradU)). symm(t)_ij = 0.5*(t_ij + t_ji); magSqr sums the squares of all 9 components.
    std::vector<scalar> out(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        const tensor& t = gradU[c];
        const scalar s[9] = {
            t.xx,                 0.5*(t.xy + t.yx),    0.5*(t.xz + t.zx),
            0.5*(t.yx + t.xy),    t.yy,                 0.5*(t.yz + t.zy),
            0.5*(t.zx + t.xz),    0.5*(t.zy + t.yz),    t.zz };
        scalar m = 0;
        for (int q = 0; q < 9; ++q) m += s[q] * s[q];
        out[c] = 2.0 * m;
    }
    return out;
}


std::vector<scalar> GbyNu0(const std::vector<tensor>& gradU)
{
    // gradU && devTwoSymm(gradU); devTwoSymm(t) = (t + t^T) - (2/3)*tr(t)*I.
    std::vector<scalar> out(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        const tensor& t = gradU[c];
        const scalar tr = t.xx + t.yy + t.zz;
        const scalar d[9] = {
            2*t.xx - (2.0/3.0)*tr,  t.xy + t.yx,            t.xz + t.zx,
            t.yx + t.xy,            2*t.yy - (2.0/3.0)*tr,  t.yz + t.zy,
            t.zx + t.xz,            t.zy + t.yz,            2*t.zz - (2.0/3.0)*tr };
        const scalar gg[9] = { t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz };
        scalar s = 0;
        for (int q = 0; q < 9; ++q) s += gg[q] * d[q];   // double inner product A && B = sum A_ij B_ij
        out[c] = s;
    }
    return out;
}


std::vector<scalar> CDkOmega(const std::vector<vector>& gradK, const std::vector<vector>& gradOmega,
                             const std::vector<scalar>& omega, const KOmegaSSTCoeffs& co)
{
    std::vector<scalar> out(omega.size());
    for (std::size_t c = 0; c < omega.size(); ++c)
        out[c] = (2.0 * co.alphaOmega2)
               * (gradK[c].x*gradOmega[c].x + gradK[c].y*gradOmega[c].y + gradK[c].z*gradOmega[c].z)
               / omega[c];
    return out;
}


std::vector<scalar> F1(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                       const std::vector<scalar>& y, const std::vector<scalar>& CD,
                       // PER CELL: the compressible lineage's nu is mu(T)/rho, a field. Both blenders
                       // use it in their viscous cross-over term, so a single constant is only right
                       // for the incompressible reading.
                       const std::vector<scalar>& nuC, const KOmegaSSTCoeffs& co)
{
    std::vector<scalar> out(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
    {
        const scalar CDp = std::fmax(CD[c], 1.0e-10);
        const scalar a = std::fmax((1.0/co.betaStar)*std::sqrt(k[c])/(omega[c]*y[c]),
                                   500.0*nuC[c]/(y[c]*y[c]*omega[c]));
        const scalar b = (4.0*co.alphaOmega2)*k[c]/(CDp*y[c]*y[c]);
        const scalar arg1 = std::fmin(std::fmin(a, b), 10.0);
        const scalar a4 = arg1*arg1*arg1*arg1;                 // pow4
        out[c] = std::tanh(a4);
    }
    return out;
}


std::vector<std::vector<scalar>> F1Boundary(
    const GeometricField<scalar>&           k,
    const GeometricField<scalar>&           omega,
    const std::vector<scalar>&              y,
    const std::vector<vector>&              gradK,
    const std::vector<vector>&              gradOmega,
    scalar                                  nu,
    const std::vector<std::vector<scalar>>* nuBnd,
    const std::vector<FvPatch>&             patches,
    const KOmegaSSTCoeffs&                  co)
{
    std::vector<std::vector<scalar>> out(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        out[pi].assign(static_cast<std::size_t>(p.size), scalar(1));
        if (p.type == "wall" || p.type == "empty") continue;   // wall: y = 0 -> arg1 = 10 -> F1 = 1
        const std::vector<scalar>& kb = k.boundary[pi]->value();
        const std::vector<scalar>& ob = omega.boundary[pi]->value();
        for (label i = 0; i < p.size; ++i)
        {
            const label c = p.faceCells[i];
            if (!(ob[i] > 0.0) || !(y[c] > 0.0)) continue;
            // The Gauss gradients' boundary values: gb = gc + n*(snGrad - n&gc), for k and omega.
            const vector& n = p.nf[i];
            const scalar snK = (kb[i] - k.internal[c]) * p.deltaCoeffs[i];
            const scalar snO = (ob[i] - omega.internal[c]) * p.deltaCoeffs[i];
            const vector& gKc = gradK[c];
            const vector& gOc = gradOmega[c];
            const scalar nK = n.x*gKc.x + n.y*gKc.y + n.z*gKc.z;
            const scalar nO = n.x*gOc.x + n.y*gOc.y + n.z*gOc.z;
            const scalar gKx = gKc.x + n.x*(snK - nK), gKy = gKc.y + n.y*(snK - nK), gKz = gKc.z + n.z*(snK - nK);
            const scalar gOx = gOc.x + n.x*(snO - nO), gOy = gOc.y + n.y*(snO - nO), gOz = gOc.z + n.z*(snO - nO);
            const scalar CDb  = (2.0*co.alphaOmega2) * (gKx*gOx + gKy*gOy + gKz*gOz) / ob[i];
            const scalar CDp  = std::fmax(CDb, 1.0e-10);
            const scalar nuB  = nuBnd ? (*nuBnd)[pi][i] : nu;
            const scalar yb   = y[c];
            const scalar a = std::fmax((1.0/co.betaStar)*std::sqrt(std::fmax(kb[i], 0.0))/(ob[i]*yb),
                                       500.0*nuB/(yb*yb*ob[i]));
            const scalar b = (4.0*co.alphaOmega2)*kb[i]/(CDp*yb*yb);
            const scalar arg1 = std::fmin(std::fmin(a, b), 10.0);
            const scalar a4 = arg1*arg1*arg1*arg1;
            out[pi][i] = std::tanh(a4);
        }
    }
    return out;
}


std::vector<scalar> F2(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                       const std::vector<scalar>& y, const std::vector<scalar>& nuC,
                       const KOmegaSSTCoeffs& co)
{
    std::vector<scalar> out(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
    {
        const scalar arg2 = std::fmin(std::fmax((2.0/co.betaStar)*std::sqrt(k[c])/(omega[c]*y[c]),
                                                500.0*nuC[c]/(y[c]*y[c]*omega[c])), 100.0);
        out[c] = std::tanh(arg2*arg2);
    }
    return out;
}


std::vector<scalar> correctNut(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                               const std::vector<scalar>& F23, const std::vector<scalar>& s2,
                               const KOmegaSSTCoeffs& co)
{
    std::vector<scalar> out(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
        out[c] = co.a1*k[c] / std::fmax(co.a1*omega[c], co.b1*F23[c]*std::sqrt(s2[c]));
    return out;
}



namespace {

// D() and the full right-hand side of an assembled system, in the SAME form tools/dumpKOmegaSST writes
// for OpenFOAM's: the diagonal including the boundary internalCoeffs, and the source with boundaryCoeffs
// folded into their face cells. Matching the capture point to the dump point is not optional -- comparing
// two different objects measures the capture, not the code.
void captureSSTSystem(
    const FvScalarMatrix&       M,
    const std::vector<FvPatch>& patches,
    std::vector<scalar>&        D,
    std::vector<scalar>&        S,
    std::vector<scalar>*        up = nullptr,
    std::vector<scalar>*        lo = nullptr)
{
    if (up) *up = M.upper;
    if (lo) *lo = M.lower;
    D = M.diag;
    S = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            D[c] += M.internalCoeffs[pi][i];
            S[c] += M.boundaryCoeffs[pi][i];
        }
}

}

void correct(
    const GeometricField<vector>&  U,
    GeometricField<scalar>&        k,
    GeometricField<scalar>&        omega,
    GeometricField<scalar>&        nutField,
    const SurfaceScalarField&      phi,
    const std::vector<scalar>&     y,
    scalar                         nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    scalar                         relaxOmega,
    scalar                         relaxK,
    scalar                         tol,
    scalar                         relTol,
    int                            maxIter,
    const KOmegaSSTCoeffs&         co,
    SSTResiduals*                  res,
    bool                           bounded,
    bool                           limitedLinear,
    scalar                         limiterCoeff,
    bool                           linearUpwind,
    bool                           correctedLaplacian,
    scalar                         snGradLimitCoeff,
    const LMHooks*                 lm,
    const Compressible*            comp,
    int                            minIter,
    bool                           relaxEquationOmega,
    bool                           relaxEquationK)
{
    if (co.F3)
        throw std::runtime_error(
            "kOmegaSST_cpp: the F3 near-wall switch is set. kOmegaSSTBase multiplies F23 by F3 "
            "(kOmegaSSTBase.C:F23), which changes both the eddy-viscosity limiter and the production "
            "limiter. Not implemented; refusing rather than silently running with F3 off.");

    const label nC = m.nCells();
    const scalar Cmu25 = std::pow(co.CmuWall, 0.25);    // the WALL FUNCTIONS' Cmu, not the model's betaStar
    std::vector<scalar>& nutF = nutField.internal;

    // alpha()*rho() multiplies every source in both equations; alpha is 1 for a single-phase model, so
    // this is rho or it is 1. Null comp is the incompressible reading, bit-for-bit as before.
    auto rhoAt = [&](label cc) { return (comp && comp->rho) ? (*comp->rho)[cc] : scalar(1.0); };
    // this->nu() varies with temperature in the compressible lineage; the incompressible one has a
    // single constant.
    auto nuAt = [&](label cc) { return (comp && comp->nu) ? (*comp->nu)[cc] : nu; };

    // ---- production, from the CURRENT nut (the previous outer iteration's correctNut) -------------
    // fvc::grad(U) through the case's grad(U) scheme (kOmegaSSTBase.C:522): cellLimited where fvSchemes
    // says so. Unlimited here, naca0012 read omega 5.4e-03 / nut 1.2e-03 against OpenFOAM at t = 1.
    std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    if (co.gradULimitK > 0.0) cellLimitGrad(gradU, U, co.gradULimitK, m, g, patches);
    const std::vector<scalar> s2  = S2(gradU);
    const std::vector<scalar> gb0 = GbyNu0(gradU);
    std::vector<scalar> G(nC);
    for (label c = 0; c < nC; ++c) G[c] = nutF[c] * gb0[c];      // RAW GbyNu0 -- the k equation's G

    // divU takes the VOLUMETRIC flux -- fvc::absolute(this->phi(), U) -- while fvm::div and the
    // `bounded` Sp below take the MASS flux. Two different fields in the compressible lineage.
    const std::vector<scalar> divU =
        fvc::div(comp && comp->phiByRho ? *comp->phiByRho : phi, m, g, patches);
    // fvc::div(alphaRhoPhi), for `bounded Gauss <scheme>`: boundedConvectionScheme subtracts
    // Sp(surfaceIntegrate(faceFlux), vf) with the flux the equation is CONVECTED by, which is the mass
    // flux. Identical to divU when comp is null.
    const std::vector<scalar> divPhi = fvc::div(phi, m, g, patches);

    if (res && res->captureStages)
    {
        res->gradU  = gradU;
        res->s2     = s2;
        res->gbyNu0 = gb0;
        res->divU   = divU;
        // G is captured AFTER the omegaWallFunction override below, not here: the wall function replaces
        // it in wall-adjacent cells and that is the G both equations are built from.
    }

    // ---- omegaWallFunction FIRST: OpenFOAM's order -----------------------------------------------
    // kOmegaSSTBase::correct() calls omega_.boundaryFieldRef().updateCoeffs() (kOmegaSSTBase.C:541) --
    // which writes omega0 into the wall cells and G0 into G -- BEFORE it builds CDkOmega from
    // fvc::grad(k) & fvc::grad(omega) (:556-559), F1 (:560) and F23 (:561). This block used to sit
    // after those, so grad(omega), CDkOmega and F1 were taken from the wall-cell omega of the PREVIOUS
    // iteration. Invisible at iteration 1 (k is uniform, so grad(k) = 0 and CDkOmega = 0 whatever
    // grad(omega) is) and on a converged state (the closure gate feeds one; the wall-cell omega no
    // longer moves), which is why it survived: measured on rhoSST at iteration 2, CDkOmega 2.7e-09 off
    // OpenFOAM's with gradU, S2, G, F1 and F23 all at 1e-15, feeding the omega cross-diffusion source
    // and growing into k 1.3e-03 by iteration 10 while kEpsilon stayed at 1e-12.
    // ---- omegaWallFunction: near-wall omega + the G override -------------------------------------
    // omega = sqrt(omegaVis^2 + omegaLog^2)  -- OF's DEFAULT blender is binomial with n = 2
    // (omegaWallFunctionFvPatchScalarField.C:445, wallFunctionBlenders(dict, BINOMIAL, 2)).
    std::vector<label> nw(nC, 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (patches[pi].type == "wall")
            for (label i = 0; i < patches[pi].size; ++i) ++nw[patches[pi].faceCells[i]];

    const std::vector<std::vector<scalar>> yWall = nearWallDist(m, g, patches);   // OF turbulence.y()
    std::vector<scalar> om0(nC, 0.0), G0(nC, 0.0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        const FvPatch& wp = patches[pi];
        const std::vector<scalar>& yw = yWall[pi];
        // PER-FACE nu along the wall. nutkWallFunction and omegaVis are written in terms of nu_w, the
        // value AT the face; the compressible lineage's nu is mu(T)/rho and varies face to face along a
        // wall with a temperature gradient. Handing the k-epsilon port a single scalar here produced a
        // NaN in the first turbulent iteration, so this takes the same overload.
        std::vector<scalar> nuFace(wp.size);
        for (label i = 0; i < wp.size; ++i)
            nuFace[i] = (comp && comp->nuBnd) ? (*comp->nuBnd)[pi][i] : nu;
        // THE STORED WALL nut, NOT A FRESH ONE. OpenFOAM's epsilon/omegaWallFunction::calculate takes
        // nutw = refCast<nutWallFunctionFvPatchScalarField>(turbModel.nut().boundaryField()[patchi])
        // and reads nutw[facei] -- the patch VALUES, last written by the previous correctNut() (or by
        // validate() at construction) from THAT call's k and nu_w. This recomputed nutkWallFunction from
        // the current k and nu_w instead: exact on a converged state (the closure gate feeds one, 1e-15)
        // and 5.4e-05 off at iteration 1 on rhoKE, where rho_b along the wall has moved since
        // construction -- OpenFOAM's stored value is uniform per wall there. That fed G0 in every wall
        // cell and left k 1e-06 off while p, T, U and the second scalar sat at 1e-12; on kOmegaSST it
        // compounded into a 1e-03 trajectory drift by iteration 10.
        const std::vector<scalar>& nutw = nutField.boundary[pi]->value();
        const std::vector<vector>& Uw = U.boundary[pi]->value();
        for (label i = 0; i < wp.size; ++i)
        {
            const label c = wp.faceCells[i];
            const scalar w = 1.0 / nw[c], kc = k.internal[c];
            const scalar omegaVis = 6.0*nuFace[i]/(co.beta1*yw[i]*yw[i]);
            const scalar omegaLog = std::sqrt(kc)/(Cmu25*co.kappa*yw[i]);
            const scalar magGradUw = mag((Uw[i] - U.internal[c]) * wp.deltaCoeffs[i]);
            om0[c] += w * std::sqrt(omegaVis*omegaVis + omegaLog*omegaLog);
            G0[c]  += w * (nutw[i] + nuFace[i]) * magGradUw * Cmu25 * std::sqrt(kc) / (co.kappa * yw[i]);
        }
    }
    std::vector<label> wallCells;
    std::vector<scalar> omVals;
    for (label c = 0; c < nC; ++c)
        if (nw[c] > 0)
        {
            G[c] = G0[c];
            omega.internal[c] = om0[c];
            wallCells.push_back(c);
            omVals.push_back(om0[c]);
        }
    if (res && res->captureStages) res->G = G;

    // ---- CDkOmega, F1, F2 ------------------------------------------------------------------------
    // The case's gradScheme on each, as OpenFOAM resolves grad(k) and grad(omega) separately.
    std::vector<vector> gradK  = fvc::gaussGrad(k, m, g, patches);
    std::vector<vector> gradOm = fvc::gaussGrad(omega, m, g, patches);
    if (co.gradKLimitK > 0.0)
    {
        cellLimitGrad(gradK,  k,     co.gradKLimitK, m, g, patches);
        cellLimitGrad(gradOm, omega, co.gradKLimitK, m, g, patches);
    }
    const std::vector<scalar> CD  = CDkOmega(gradK, gradOm, omega.internal, co);
    // A per-cell nu, so F1/F2's viscous cross-over terms see the field the compressible lineage has.
    std::vector<scalar> nuCell(nC);
    for (label c = 0; c < nC; ++c) nuCell[c] = nuAt(c);
    std::vector<scalar> f1 = F1(k.internal, omega.internal, y, CD, nuCell, co);
    if (res && res->captureStages) { res->CD = CD; res->f1 = f1; }
    // F1 on the boundary faces, for the two diffusivities' boundary coefficients (see the header).
    std::vector<std::vector<scalar>> f1Bnd =
        F1Boundary(k, omega, y, gradK, gradOm, nu, comp ? comp->nuBnd : nullptr, patches, co);
    if (res && res->captureStages) res->f1Bnd = f1Bnd;
    if (lm)
    {
        // kOmegaSSTLM::F1 = max(kOmegaSST::F1, F3), F3 = exp(-(Ry/120)^8), Ry = y*sqrt(k)/nu
        // (kOmegaSSTLM.C:43-52). Raising F1 toward 1 near the wall keeps the k-omega branch of the blend
        // through the laminar region, which is the point of a transition model.
        for (label c = 0; c < nC; ++c)
        {
            const scalar Ry = y[c] * std::sqrt(std::fmax(k.internal[c], 0.0)) / nu;
            const scalar r  = Ry / 120.0;
            const scalar r2 = r * r, r4 = r2 * r2;
            f1[c] = std::fmax(f1[c], std::exp(-(r4 * r4)));
        }
    }
    const std::vector<scalar> f23 = F2(k.internal, omega.internal, y, nuCell, co);
    if (res && res->captureStages) res->f23 = f23;

    // ---- the production limiter: omega uses the LIMITED GbyNu, k uses the raw G ------------------
    // kOmegaSSTBase.C reassigns GbyNu0 = GbyNu(GbyNu0, F23, S2) AFTER G was taken from the raw value.
    // Using one for both is the easy mistake here; they are different quantities.
    std::vector<scalar> gbLim(nC);
    for (label c = 0; c < nC; ++c)
        gbLim[c] = std::fmin(gb0[c],
                             (co.c1/co.a1)*co.betaStar*omega.internal[c]
                           * std::fmax(co.a1*omega.internal[c], co.b1*f23[c]*std::sqrt(s2[c])));


    // ---- omega equation --------------------------------------------------------------------------
    {
        std::vector<scalar> DomegaEff(nC);
        for (label c = 0; c < nC; ++c)
            DomegaEff[c] = blend(f1[c], co.alphaOmega1, co.alphaOmega2)*nutF[c] + nuAt(c);
        const SurfaceScalarField Df =
            effectiveDiffusivity(DomegaEff, nutField, f1, co.alphaOmega1, co.alphaOmega2,
                                 nu, m, g, patches,
                                 comp ? comp->rho : nullptr, comp ? comp->rhoBnd : nullptr,
                                 comp ? comp->nuBnd : nullptr, &f1Bnd);

        // fvMatrix's constructor calls psi.boundaryFieldRef().updateCoeffs(), and that is where
        // OpenFOAM's flux-conditional boundaries read the flux and set their valueFraction, and where the
        // turbulent inlets RECOMPUTE their refValue from the current fields. Without it every such patch
        // keeps whatever it was seeded with and contributes nothing to the system, whatever value it
        // carries -- which is two of the four defects the kEpsilon port turned up, found the same way.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // turbulentMixingLengthFrequencyInlet: refValue = sqrt(kp)/(Cmu^0.25*L), from k's CURRENT
            // patch values. Its Cmu is `coeffDict().getOrDefault("Cmu", 0.09)`
            // (turbulentMixingLengthFrequencyInletFvPatchScalarField.C:137-138) -- kOmegaSSTCoeffs
            // { Cmu } when the case names one, else 0.09, and NOT betaStar, which this passed: with
            // kOmegaSSTCoeffs { betaStar 0.11; } the inlet omega read 379.76 where OpenFOAM writes
            // 399.30 (the field 1.21e-03 at t=1), with { Cmu 0.12; } 399.30 against 371.59 (1.73e-03)
            // (rho_turbinlet_cmu_vs_openfoam).
            omega.boundary[pi]->updateTurbulentInlet({}, k.boundary[pi]->value(), co.Cmu, co.CmuInDict);
            omega.boundary[pi]->updateFromFlux(phi.boundary[pi]);
        }

        FvScalarMatrix M = divWithScheme(phi, omega, limitedLinear, limiterCoeff, m, g, patches);
        {
            // The laplacian with BOTH halves of `corrected`, then subtracted from the equation. The
            // explicit correction goes into the LAPLACIAN's own source first, so the -1.0 below carries
            // it into the transport equation with the right sign.
            FvScalarMatrix L = fvm::laplacian(Df, omega, m, g, patches, correctedLaplacian);
            if (correctedLaplacian)
            {
                std::vector<std::vector<scalar>> vb(patches.size());
                for (std::size_t pi = 0; pi < patches.size(); ++pi) vb[pi] = omega.boundary[pi]->value();
                std::vector<vector> gradVf = fvc::gaussGrad(omega.internal, vb, m, g, patches);   // grad(omega)'s own scheme
                if (co.gradKLimitK > 0.0) cellLimitGrad(gradVf, omega.internal, vb, co.gradKLimitK, m, g, patches);
                const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                    Df, omega, gradVf, m, g, patches, snGradLimitCoeff);
                for (label c = 0; c < nC; ++c) L.source[c] -= corr[c];
            }
            addEqual(M, L, -1.0);
        }
        for (label c = 0; c < nC; ++c)
        {
            const scalar gam  = blend(f1[c], co.gamma1, co.gamma2);
            const scalar beta = blend(f1[c], co.beta1,  co.beta2);
            const scalar V    = g.V()[c];
            // alpha()*rho() on EVERY source and Sp of the omega equation (kOmegaSSTBase.C). rhoAt is 1
            // for the incompressible lineage, so the arithmetic there is unchanged.
            const scalar rc   = rhoAt(c);
            M.source[c] += rc * gam * gbLim[c] * V;                  // == gamma*GbyNu
            // - SuSp((2/3)*gamma*divU, omega): implicit where the coefficient is positive.
            const scalar sp1 = (2.0/3.0) * rc * gam * divU[c];
            M.diag[c]   += V * std::fmax(sp1, 0.0);
            M.source[c] -= V * std::fmin(sp1, 0.0) * omega.internal[c];
            // - Sp(beta*omega, omega)
            M.diag[c]   += rc * beta * omega.internal[c] * V;
            // - SuSp((F1 - 1)*CDkOmega/omega, omega)
            const scalar sp2 = rc * (f1[c] - 1.0) * CD[c] / omega.internal[c];
            M.diag[c]   += V * std::fmax(sp2, 0.0);
            M.source[c] -= V * std::fmin(sp2, 0.0) * omega.internal[c];
            // `bounded`: - Sp(fvc::div(phi), omega). Vanishes where phi is conservative, so it cannot
            // move a converged answer -- it is there to keep the transported scalar bounded on the way.
            if (bounded) M.diag[c] -= divPhi[c] * V;
        }
        if (linearUpwind)
        {
            // linearUpwind's deferred correction. The caller SUBTRACTS what linearUpwindCorrection
            // returns -- see the sign note in fvm.cuh.
            std::vector<std::vector<scalar>> ob(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) ob[pi] = omega.boundary[pi]->value();
            std::vector<vector> gradVf = fvc::gaussGrad(omega.internal, ob, m, g, patches);
            if (co.gradKLimitK > 0.0) cellLimitGrad(gradVf, omega.internal, ob, co.gradKLimitK, m, g, patches);
            const std::vector<scalar> corr =
                fvm::linearUpwindCorrection<scalar, vector>(phi.internal, gradVf, m, g);
            for (label c = 0; c < nC; ++c) M.source[c] -= corr[c];
        }

        // Per-cell term trace. A single residual over the whole field cannot say WHICH term of the
        // omega equation disagrees, and on a resolved mesh the terms span eight orders of magnitude.
        if (const char* cs = std::getenv("BRAE_SST_CELL"))
        {
            const label c = std::atoi(cs);
            if (c >= 0 && c < nC)
            {
                const scalar gam  = blend(f1[c], co.gamma1, co.gamma2);
                const scalar beta = blend(f1[c], co.beta1,  co.beta2);
                const scalar V    = g.V()[c];
                std::printf("  [omega cell %d]  V %.4e  y %.4e  omega %.6e  k %.6e  nut %.6e\n",
                            (int)c, V, y[c], omega.internal[c], k.internal[c], nutF[c]);
                std::printf("    F1 %.6f  F23 %.6f  gamma %.6f  beta %.6f  S %.6e\n",
                            f1[c], f23[c], gam, beta, std::sqrt(s2[c]));
                std::printf("    GbyNu0 %.6e  GbyNu(limited) %.6e  %s\n",
                            gb0[c], gbLim[c], gbLim[c] < gb0[c] ? "LIMITER ACTIVE" : "unlimited");
                std::printf("    production gamma*GbyNu*V %.6e   destruction beta*omega^2*V %.6e\n",
                            gam*gbLim[c]*V, beta*omega.internal[c]*omega.internal[c]*V);
                std::printf("    CDkOmega %.6e   cross-diff SuSp (F1-1)*CD/omega %.6e\n",
                            CD[c], (f1[c] - 1.0)*CD[c]/omega.internal[c]);
                std::printf("    divU %.6e   matrix: diag %.6e  source %.6e\n",
                            divU[c], M.diag[c], M.source[c]);
            }
        }
        if (res && res->captureStages)
            captureSSTSystem(M, patches, res->omD0, res->omSrc0);
        if (relaxEquationOmega) relaxMatrix(M, omega, m, patches, relaxOmega);
        setValues(M, omega.internal, m, patches, wallCells, omVals);
        if (res && res->captureStages)
            captureSSTSystem(M, patches, res->omD, res->omSrc, &res->omUpper, &res->omLower);
        const SolverPerformance po = pbicgstab(M, omega.internal, m, patches, tol, relTol, maxIter, minIter);
        if (res) res->omega = po.initialResidual;
        if (std::getenv("BRAE_SST_DEBUG"))
            std::printf("    [omega] init=%.3e final=%.3e nIter=%d\n",
                        po.initialResidual, po.finalResidual, po.nIterations);
        static const bool dbgBound = std::getenv("BRAE_SST_BOUND_DEBUG") != nullptr;
        if (dbgBound)
        {
            scalar mn = omega.internal[0];
            int nNeg = 0;
            for (label c = 0; c < nC; ++c)
            {
                mn = std::fmin(mn, omega.internal[c]);
                if (omega.internal[c] < 0.0) ++nNeg;
            }
            std::printf("      [bound] omega min %.6e   negative cells %d\n", mn, nNeg);
        }
        // Foam::bound(omega_, omegaMin_). A FLOOR here is not the same thing and is not survivable:
        // the next iteration divides CDkOmega by this, so a floored cell contributes ~1e15. See
        // bound_cpp.cuh for the measurement.
        omega.evaluateBoundary();
        bound(omega, 1e-15, m, g, patches);
    }

    // ---- k equation ------------------------------------------------------------------------------
    {
        std::vector<scalar> DkEff(nC);
        for (label c = 0; c < nC; ++c)
            DkEff[c] = blend(f1[c], co.alphaK1, co.alphaK2)*nutF[c] + nuAt(c);
        const SurfaceScalarField Df =
            effectiveDiffusivity(DkEff, nutField, f1, co.alphaK1, co.alphaK2,
                                 nu, m, g, patches,
                                 comp ? comp->rho : nullptr, comp ? comp->rhoBnd : nullptr,
                                 comp ? comp->nuBnd : nullptr, &f1Bnd);

        // fvMatrix's constructor calls psi.boundaryFieldRef().updateCoeffs(), and that is where
        // OpenFOAM's flux-conditional boundaries read the flux and set their valueFraction, and where the
        // turbulent inlets RECOMPUTE their refValue from the current fields. Without it every such patch
        // keeps whatever it was seeded with and contributes nothing to the system, whatever value it
        // carries -- which is two of the four defects the kEpsilon port turned up, found the same way.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // turbulentIntensityKineticEnergyInlet reads U's patch values (and no Cmu at all).
            k.boundary[pi]->updateTurbulentInlet(U.boundary[pi]->value(), {}, co.Cmu, co.CmuInDict);
            k.boundary[pi]->updateFromFlux(phi.boundary[pi]);
        }

        FvScalarMatrix M = divWithScheme(phi, k, limitedLinear, limiterCoeff, m, g, patches);
        {
            // The laplacian with BOTH halves of `corrected`, then subtracted from the equation. The
            // explicit correction goes into the LAPLACIAN's own source first, so the -1.0 below carries
            // it into the transport equation with the right sign.
            FvScalarMatrix L = fvm::laplacian(Df, k, m, g, patches, correctedLaplacian);
            if (correctedLaplacian)
            {
                std::vector<std::vector<scalar>> vb(patches.size());
                for (std::size_t pi = 0; pi < patches.size(); ++pi) vb[pi] = k.boundary[pi]->value();
                std::vector<vector> gradVf = fvc::gaussGrad(k.internal, vb, m, g, patches);   // grad(k)'s own scheme
                if (co.gradKLimitK > 0.0) cellLimitGrad(gradVf, k.internal, vb, co.gradKLimitK, m, g, patches);
                const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                    Df, k, gradVf, m, g, patches, snGradLimitCoeff);
                for (label c = 0; c < nC; ++c) L.source[c] -= corr[c];
            }
            addEqual(M, L, -1.0);
        }
        for (label c = 0; c < nC; ++c)
        {
            const scalar V = g.V()[c];
            // == Pk(G) = min(G, (c1*betaStar)*k*omega), and for kOmegaSSTLM that whole thing scaled by
            // gammaIntEff (kOmegaSSTLM.C:55-62) -- the intermittency IS the switch that turns turbulent
            // production on as the boundary layer transitions.
            // alpha()*rho() on every source and Sp of the k equation, as for omega above.
            const scalar rc = rhoAt(c);
            scalar pk = std::fmin(G[c], (co.c1*co.betaStar)*k.internal[c]*omega.internal[c]);
            if (lm) pk *= (*lm->gammaIntEff)[c];
            M.source[c] += rc * pk * V;
            const scalar sp = (2.0/3.0) * rc * divU[c];              // - SuSp((2/3)*divU, k)
            M.diag[c]   += V * std::fmax(sp, 0.0);
            M.source[c] -= V * std::fmin(sp, 0.0) * k.internal[c];
            // - Sp(epsilonByk, k). kOmegaSSTLM scales epsilonByk by gammaIntEff CLAMPED to [0.1, 1]
            // (kOmegaSSTLM.C:65-76) -- a different clamp from the production above, so the destruction
            // never falls below a tenth even where the intermittency does.
            scalar ebk = co.betaStar * omega.internal[c];
            if (lm)
            {
                const scalar ge = (*lm->gammaIntEff)[c];
                ebk *= (ge < 0.1 ? 0.1 : (ge > 1.0 ? 1.0 : ge));
            }
            M.diag[c]   += rc * ebk * V;
            if (bounded) M.diag[c] -= divPhi[c] * V;                 // - Sp(fvc::div(alphaRhoPhi), k)
        }
        if (linearUpwind)
        {
            std::vector<std::vector<scalar>> kb(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) kb[pi] = k.boundary[pi]->value();
            std::vector<vector> gradVf = fvc::gaussGrad(k.internal, kb, m, g, patches);
            if (co.gradKLimitK > 0.0) cellLimitGrad(gradVf, k.internal, kb, co.gradKLimitK, m, g, patches);
            const std::vector<scalar> corr =
                fvm::linearUpwindCorrection<scalar, vector>(phi.internal, gradVf, m, g);
            for (label c = 0; c < nC; ++c) M.source[c] -= corr[c];
        }
        if (res && res->captureStages)
            captureSSTSystem(M, patches, res->kD0, res->kSrc0);
        if (relaxEquationK) relaxMatrix(M, k, m, patches, relaxK);
        if (res && res->captureStages)
            captureSSTSystem(M, patches, res->kD, res->kSrc, &res->kUpper, &res->kLower);
        const SolverPerformance pk = pbicgstab(M, k.internal, m, patches, tol, relTol, maxIter, minIter);
        if (res) res->k = pk.initialResidual;
        if (std::getenv("BRAE_SST_DEBUG"))
            std::printf("    [k] init=%.3e final=%.3e nIter=%d\n",
                        pk.initialResidual, pk.finalResidual, pk.nIterations);
        k.evaluateBoundary();
        bound(k, 1e-15, m, g, patches);   // Foam::bound(k_, kMin_)
    }

    // ---- correctNut(S2), boundary and EddyDiffusivity included -- ONE implementation, shared with
    // turbulence->validate() at construction (see correctNutField). S2 is the one computed at the TOP of
    // correct() from the old U, F23 inside reads the k and omega the two solves above have just written:
    // kOmegaSSTBase.C ends with correctNut(S2) and F23() reads the members.
    correctNutField(U, k, omega, nutField, gradU, y, yWall, nu, m, g, patches, co, comp);
}

void correctNutField(
    const GeometricField<vector>&           U,
    const GeometricField<scalar>&           k,
    const GeometricField<scalar>&           omega,
    GeometricField<scalar>&                 nutField,
    const std::vector<tensor>&              gradU,
    const std::vector<scalar>&              y,
    const std::vector<std::vector<scalar>>& yWall,
    scalar                                  nu,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches,
    const KOmegaSSTCoeffs&                  co,
    const Compressible*                     comp)
{
    (void)m; (void)g;
    const label nC = m.nCells();
    std::vector<scalar>& nutF = nutField.internal;
    std::vector<scalar> nuCell(nC);
    for (label c = 0; c < nC; ++c) nuCell[c] = (comp && comp->nu) ? (*comp->nu)[c] : nu;
    const std::vector<scalar> s2 = S2(gradU);
    const std::vector<scalar> f23New = F2(k.internal, omega.internal, y, nuCell, co);
    nutF = correctNut(k.internal, omega.internal, f23New, s2, co);   // a1*k/max(a1*omega, b1*F23*sqrt(S2))

    // EddyDiffusivity::correctNut, which the COMPRESSIBLE instantiation runs after the model's own:
    //     alphat = rho*nut/Prt
    // The energy equation's alphaEff reads this field, so a compressible run without it transports heat
    // with no turbulent contribution at all.
    if (comp && comp->alphat && comp->rho)
    {
        comp->alphat->resize(nC);
        for (label c = 0; c < nC; ++c)
            (*comp->alphat)[c] = (*comp->rho)[c] * nutF[c] / comp->Prt;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        nutField.boundary[pi]->setValue(
            nutkWallFunction(patches[pi], yWall[pi], k.internal,
                             comp && comp->nuBnd ? (*comp->nuBnd)[pi]
                                                 : std::vector<scalar>(patches[pi].size, nu),
                             co.CmuWall, co.kappa, co.E));
    }

    // OpenFOAM assigns nut_ as a FIELD -- nut_ = a1*k/max(a1*omega, b1*F23*sqrt(S2)) -- and a field
    // assignment writes the BOUNDARY as well, from the boundary k and omega. correctBoundaryConditions()
    // then leaves a `calculated` patch alone, so those patches carry the evaluated value rather than the
    // adjacent cell's. Only wall patches were being written here, which the single-iteration probe could
    // never catch: it reads nut's boundary from OpenFOAM's converged file, where it is already right.
    // Running from 0/ is what exposes it -- the tutorials ship `calculated; value uniform 0` at the
    // inlet, so the inlet eddy viscosity stayed ZERO for the entire run.
    //
    // Every operand is taken at the BOUNDARY, as a field expression does: k and omega from their patch
    // values, S2 from the boundary gradU (gaussGrad's boundary correction replaces the normal component
    // with the snGrad, gb = gc + n*(snGrad - n&gc)), and F2 from those with the owner cell's y.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "wall") continue;
        if (nutField.boundary[pi]->bcCategory() != 2) continue;   // fixedValue means the case PINNED it

        const std::vector<scalar>& kb = k.boundary[pi]->value();
        const std::vector<scalar>& ob = omega.boundary[pi]->value();
        const std::vector<vector>& ub = U.boundary[pi]->value();
        std::vector<scalar> vals(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            if (!(ob[i] > 0.0) || !(y[c] > 0.0))
            {
                vals[i] = nutF[c];
                continue;
            }

            const tensor& t = gradU[c];
            const scalar gc[9] = { t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz };
            const vector& n = patches[pi].nf[i];
            const scalar nv[3] = { n.x, n.y, n.z };
            const vector sng = (ub[i] - U.internal[c]) * patches[pi].deltaCoeffs[i];
            const scalar sn[3] = { sng.x, sng.y, sng.z };

            scalar ngc[3];
            for (int j = 0; j < 3; ++j)
            {
                ngc[j] = nv[0]*gc[0*3+j] + nv[1]*gc[1*3+j] + nv[2]*gc[2*3+j];
            }
            scalar gb[9];
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    gb[a*3+b] = gc[a*3+b] + nv[a]*(sn[b] - ngc[b]);
                }
            }
            scalar ss = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    const scalar sab = 0.5*(gb[a*3+b] + gb[b*3+a]);
                    ss += sab*sab;
                }
            }
            const scalar S2b = 2.0*ss;

            // nu AT THE FACE: the compressible caller passes the scalar nu as 0 and carries mu(T)/rho per
            // face in comp->nuBnd; the scalar was being used here, which zeroed F2's viscous term.
            const scalar nuB = (comp && comp->nuBnd) ? (*comp->nuBnd)[pi][i] : nu;
            const scalar arg2 = std::fmax(2.0*std::sqrt(std::fmax(kb[i], 0.0))/(co.betaStar*ob[i]*y[c]),
                                          500.0*nuB/(y[c]*y[c]*ob[i]));
            const scalar F2b  = std::tanh(arg2*arg2);
            vals[i] = co.a1*kb[i] / std::fmax(co.a1*ob[i], co.b1*F2b*std::sqrt(std::fmax(S2b, 0.0)));
        }
        nutField.boundary[pi]->setValue(vals);
    }
}

} // namespace kOmegaSST
} // namespace cpu
} // namespace brae
