// _cpp REFERENCE implementation -- see kOmegaSST_cpp.cuh for the OpenFOAM provenance and refusals.
#include "kOmegaSST_cpp.cuh"
#include "sst_stage_dump.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "nut_wall_function.cuh"
#include "near_wall_dist.cuh"
#include "pbicg.cuh"
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
        if (patches[pi].coupled)
        {
            // A COUPLED FACE KEEPS fvc::interpolate's VALUE, which is the two CELLS' DEff:
            // surfaceInterpolationScheme::interpolate takes pLambda*patchInternalField +
            // pY*patchNeighbourField wherever the patch field is coupled
            // (surfaceInterpolationScheme.C:186-191), and only there does it read the patch value.
            // Rebuilding it here as blend(F1_b)*nut_b + nu_b is a different number because the blend
            // is non-linear in F1: interp(alphaK(F1)*nut) is not alphaK(interp(F1))*interp(nut).
            // MEASURED on validation/interFoamCyclic `sst`, ten steps: omega 4.7e-11 against
            // OpenFOAM where keeping the interpolated value reads 8.8e-14, nut 7.7e-11 against
            // 2.4e-12, k 1.3e-11 against 4.8e-13. The kEpsilon closure has always kept it.
            continue;
        }
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
    const std::vector<FvPatch>&      patches,
    // The limiter's gradient is fvc::grad(lPhi) with lPhi the field itself (LimitedScheme.C:51-55,
    // limitFuncs::magSqr<scalar> returns phi), so it resolves `grad(k)` / `grad(omega)` -- the same
    // gradSchemes entry the closure's own gradients take: the same base scheme, and the same cellLimited
    // coefficient on top. This took the plain Gauss gradient whatever the case said; the driver resolved
    // the scheme and this function never received it. Measured on validation/rhoSST restarted from
    // OpenFOAM's own iteration 5 with `grad(k) leastSquares; grad(omega) leastSquares;` and limitedLinear
    // on both: exact at iteration 6, k 3.1e-06 / omega 6.4e-05 at iteration 7 with the Gauss gradient here.
    scalar                           limGradK = 0.0,
    bool                             limGradLeastSq = false,
    // INSTRUMENT: when set, the weights and the gradient THIS call limited with, under <pre>Asm*,
    // beside the device's columns of the same name. Recomputing them at the call site cannot witness
    // a gradient built differently in here, which is the class of gap being chased.
    const turbulence::SstStageDump*  sd = nullptr,
    const char*                      pre = "")
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
    std::vector<vector> gradVf = limGradLeastSq ? fvc::leastSquaresGrad(vf.internal, vfb, m, g, patches)
                                                : fvc::gaussGrad(vf.internal, vfb, m, g, patches);
    if (limGradK > 0.0) cellLimitGrad(gradVf, vf.internal, vfb, limGradK, m, g, patches);
    const std::vector<scalar> w = ls::limitedLinearWeights(phi.internal, vf, gradVf, limiterCoeff, m, g);
    if (sd && sd->on)
    {
        const std::string t(pre);
        sd->scalars((t + "AsmLimW").c_str(), w);
        std::vector<scalar> gx(gradVf.size()), gy(gradVf.size()), gz(gradVf.size());
        for (std::size_t c = 0; c < gradVf.size(); ++c)
        { gx[c] = gradVf[c].x; gy[c] = gradVf[c].y; gz[c] = gradVf[c].z; }
        sd->scalars((t + "AsmGradX").c_str(), gx);
        sd->scalars((t + "AsmGradY").c_str(), gy);
        sd->scalars((t + "AsmGradZ").c_str(), gz);
        std::vector<scalar> bv;
        for (const auto& pb : vfb) bv.insert(bv.end(), pb.begin(), pb.end());
        sd->scalars((t + "AsmBval").c_str(),  bv);
        sd->scalars((t + "AsmField").c_str(), vf.internal);
        sd->scalars((t + "AsmPhi").c_str(),   phi.internal);
        // ...and the geometry, beside the device's columns of the same name (see turbulence_transport.cu).
        sd->scalars((t + "AsmCd").c_str(), g.weights());
        const label nIf = m.nInternalFaces();
        std::vector<scalar> dx(nIf), dy(nIf), dz(nIf);
        for (label f = 0; f < nIf; ++f)
        {
            const label P = m.owner()[f], N = m.neighbour()[f];
            dx[f] = g.C()[N].x - g.C()[P].x;
            dy[f] = g.C()[N].y - g.C()[P].y;
            dz[f] = g.C()[N].z - g.C()[P].z;
        }
        sd->scalars((t + "AsmDX").c_str(), dx);
        sd->scalars((t + "AsmDY").c_str(), dy);
        sd->scalars((t + "AsmDZ").c_str(), dz);
    }
    // ...and the coupled patches' own weights, from the same limiter with the patch's two sides
    // (gaussConvectionScheme.C:105-108). fvm::div refuses a pair without them.
    const std::vector<std::vector<scalar>> pw =
        ls::limitedLinearPatchWeights(phi.boundary, vf.internal, gradVf, limiterCoeff, patches);
    if (sd && sd->on)
    {
        std::vector<scalar> flat, fphi, fcd;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].coupled) continue;
            flat.insert(flat.end(), pw[pi].begin(), pw[pi].end());
            fphi.insert(fphi.end(), phi.boundary[pi].begin(), phi.boundary[pi].end());
            fcd.insert(fcd.end(), patches[pi].weights.begin(), patches[pi].weights.end());
        }
        sd->scalars((std::string(pre) + "CycW").c_str(), flat);
        sd->scalars((std::string(pre) + "CycPhi").c_str(), fphi);
        sd->scalars((std::string(pre) + "CycCd").c_str(), fcd);
        std::vector<scalar> fo, fn, fdx;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& fp = patches[pi];
            if (!fp.coupled) continue;
            for (label i = 0; i < fp.size; ++i)
            {
                fo.push_back(vf.internal[static_cast<std::size_t>(fp.faceCells[i])]);
                fn.push_back(patchNeighbourValue(fp, i, vf.internal));
                fdx.push_back(fp.delta[i].x);
            }
        }
        sd->scalars((std::string(pre) + "CycFOwn").c_str(), fo);
        sd->scalars((std::string(pre) + "CycFNbr").c_str(), fn);
        sd->scalars((std::string(pre) + "CycDX").c_str(), fdx);
    }
    return fvm::div(phi.internal, phi.boundary, vf, w, m, patches, &pw);
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
        // THE PATCH'S OWN snGrad(), which is what gaussGrad::correctBoundaryConditions asks for
        // (gaussGrad.C:164-168) -- see correctNutField, where the inline (value - cell)*deltaCoeffs
        // cost interFoam's waterChannel its whole atmosphere
        const std::vector<scalar> snKPatch = k.boundary[pi]->snGrad(k.internal);
        const std::vector<scalar> snOPatch = omega.boundary[pi]->snGrad(omega.internal);
        for (label i = 0; i < p.size; ++i)
        {
            const label c = p.faceCells[i];
            if (!(ob[i] > 0.0) || !(y[c] > 0.0)) continue;
            // The Gauss gradients' boundary values: gb = gc + n*(snGrad - n&gc), for k and omega.
            const vector& n = p.nf[i];
            const scalar snK = snKPatch[i];
            const scalar snO = snOPatch[i];
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
    std::vector<scalar>*        lo = nullptr,
    // the COUPLED patches' boundaryCoeffs, flattened in patch order: the pair's off-diagonal, which
    // is where a div scheme reaches the interface
    std::vector<scalar>*        ifc = nullptr)
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
            // A COUPLED PATCH'S boundaryCoeffs ARE NOT A SOURCE: the solver multiplies them by the
            // NEIGHBOUR's psi every sweep, so folding them in here makes a constant out of the psi the
            // matrix happened to be assembled at -- and the device arm cannot do the same, because its
            // pair lives in cyc.ifCoeff and never reaches the boundary arrays deviceFold walks. The two
            // arms' source columns would then differ on every coupled face by construction, which is a
            // difference in the INSTRUMENT and not in the system. They are reported on their own
            // instead (`ifc` below), which is also the column that carries the div scheme across a pair.
            if (!patches[pi].coupled) S[c] += M.boundaryCoeffs[pi][i];
        }
    if (ifc)
    {
        ifc->clear();
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].coupled) continue;
            ifc->insert(ifc->end(), M.boundaryCoeffs[pi].begin(), M.boundaryCoeffs[pi].end());
        }
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
    bool                           relaxEquationK,
    const LinearSolverChoice*      which,
    const EqnSolveSetting*         omegaSolve,
    const EqnDivScheme*            omegaDiv,
    const EqnGradScheme*           omegaGrad,
    bool                           nonOrthCoeffs)
{
    // OMEGA'S OWN CONVECTION SCHEME, resolved ONCE so the assembly and the `bounded` term cannot read
    // different answers. Absent it, omega's is k's -- what a caller that still refuses a mismatch means.
    EqnDivScheme kDivS;
    kDivS.bounded = bounded;
    kDivS.limitedLinear = limitedLinear;
    kDivS.limiterCoeff = limiterCoeff;
    // EXPLICITLY FALSE, not left to the default: this closure assembles `Gauss linearUpwind` through
    // the positional `linearUpwind` flag and co.luGradLimitK, for BOTH equations -- its callers require
    // k's and omega's entries to agree (rhoSimpleFoam's driver, interFoam's readClosureDivScheme) --
    // and never through EqnDivScheme, so that half of the struct is not read here. The kEpsilon twin
    // builds the same struct from five parameters, and a reader of two builders that fill different
    // subsets cannot tell "not applicable here" from "forgotten" -- which is what the defaults audit
    // flagged.
    kDivS.linearUpwind = false;
    kDivS.luGradK = 0.0;
    const EqnDivScheme oDiv = omegaDiv ? *omegaDiv : kDivS;
    // OMEGA'S OWN GRADIENT SCHEME; `co.gradKLeastSq`/`gradKLimitK` are k's. CDkOmega takes grad(k) AND
    // grad(omega) in one expression (kOmegaSSTBase.C:548), so the two are resolved here, once, and every
    // site below reads the one for the field it differentiates.
    EqnGradScheme kGradS;
    kGradS.leastSquares = co.gradKLeastSq;
    kGradS.cellLimitK   = co.gradKLimitK;
    const EqnGradScheme oGrad = omegaGrad ? *omegaGrad : kGradS;
    // THE CASE'S LINEAR SOLVER, PER EQUATION -- `which`/`tol`/... are k's, `omegaSolve` is omega's, and
    // null means the caller has one setting for both. See the `omegaSolve` parameter.
    auto solveWith = [&](const FvScalarMatrix&      A,
                         std::vector<scalar>&       psi,
                         const LinearSolverChoice*  w,
                         scalar                     t,
                         scalar                     rt,
                         int                        mx,
                         int                        mn)
    {
        if (w && w->smoothSolver)
        {
            return smoothSolver(A, psi, m, patches, w->symmetric, t, rt, mx, mn, w->nSweeps);
        }
        // `solver PBiCG; preconditioner DILU;` -- OpenFOAM's PBiCG, not PBiCGStab, which is a
        // different recurrence and stops somewhere else at the same tolerance. The kEpsilon twin has
        // carried this branch since waves/mangroveInteraction; this one fell through to PBiCGStab.
        if (w && w->pbicgDILU)
        {
            return pbicgDILU(A, psi, m, patches, t, rt, mx, mn);
        }
        return pbicgstab(A, psi, m, patches, t, rt, mx, mn);
    };
    auto solveScalar = [&](const FvScalarMatrix& A, std::vector<scalar>& psi)
    {
        return solveWith(A, psi, which, tol, relTol, maxIter, minIter);
    };
    // omega's. kOmegaSSTBase solves omega FIRST and k second (kOmegaSSTBase.C:572 then :602).
    auto solveSecond = [&](const FvScalarMatrix& A, std::vector<scalar>& psi)
    {
        return omegaSolve ? solveWith(A, psi, &omegaSolve->which, omegaSolve->tol, omegaSolve->relTol,
                                      omegaSolve->maxIter, omegaSolve->minIter)
                          : solveWith(A, psi, which, tol, relTol, maxIter, minIter);
    };
    if (co.F3)
        throw std::runtime_error(
            "kOmegaSST_cpp: the F3 near-wall switch is set. kOmegaSSTBase multiplies F23 by F3 "
            "(kOmegaSSTBase.C:F23), which changes both the eddy-viscosity limiter and the production "
            "limiter. Not implemented; refusing rather than silently running with F3 off.");

    const label nC = m.nCells();
    const scalar Cmu25 = std::pow(co.CmuWall, 0.25);    // the WALL FUNCTIONS' Cmu, not the model's betaStar
    std::vector<scalar>& nutF = nutField.internal;

    // The closure instrument -- see sst_stage_dump.cuh. Same names as the device twin's.
    const turbulence::SstStageDump sd = turbulence::sstStageDump("host");
    sd.scalars("y", y);
    sd.scalars("kIn", k.internal);
    sd.scalars("omegaIn", omega.internal);
    sd.scalars("nutIn", nutF);
    if (sd.on)
    {
        // The gradient's two operands: the cell values and the PATCH values gaussGrad sums over. If the
        // cells agree and the faces do not, the gradient's disagreement is the boundary's.
        std::vector<scalar> ux, uy, uz;
        ux.reserve(U.internal.size());
        uy.reserve(U.internal.size());
        uz.reserve(U.internal.size());
        for (const vector& v : U.internal)
        {
            ux.push_back(v.x);
            uy.push_back(v.y);
            uz.push_back(v.z);
        }
        sd.scalars("Ux", ux);
        sd.scalars("Uy", uy);
        sd.scalars("Uz", uz);
        std::vector<scalar> bx, by, bz;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<vector>& vb = U.boundary[pi]->value();
            for (const vector& v : vb)
            {
                bx.push_back(v.x);
                by.push_back(v.y);
                bz.push_back(v.z);
            }
        }
        sd.scalars("UbX", bx);
        sd.scalars("UbY", by);
        sd.scalars("UbZ", bz);
    }

    // alpha()*rho() multiplies every source in both equations; alpha is 1 for a single-phase model, so
    // this is rho or it is 1. Null comp is the incompressible reading, bit-for-bit as before.
    auto rhoAt = [&](label cc) { return (comp && comp->rho) ? (*comp->rho)[cc] : scalar(1.0); };
    // fvm::ddt(alpha, rho, psi), EulerDdtScheme::fvmDdt: diag = rho*V/deltaT, source =
    // rho.oldTime()*psi.oldTime()*V/deltaT; psi.oldTime() is the field as this iteration started, taken
    // here before anything writes it. Zero under steadyState.
    // ...and under CRANKNICOLSON the term is the scheme's own, added where the Euler line would have
    // gone with rDeltaT left at zero; the caller keeps the old-old levels. Transcribed from the
    // kEpsilon reference's block (kEpsilon_cpp.cu:326-341), which is the gated shape for it.
    const bool cn = comp && comp->cn;
    if (cn)
    {
        if (!comp->cnDdt0Omega || !comp->cnDdt0K || !comp->omegaOO || !comp->kOO
         || (comp->rho && !comp->rhoOO) || (comp->rhoOld && !comp->rhoOO))
            throw std::runtime_error(
                "brae kOmegaSST: CrankNicolson needs the two ddt0 fields, k.oldTime().oldTime(), "
                "omega.oldTime().oldTime() and, with a density, rho.oldTime().oldTime(); the caller "
                "supplied fewer.");
        if (comp->V0)
            throw std::runtime_error(
                "brae kOmegaSST: CrankNicolson's fvm::ddt on a moving mesh is the scheme's moving "
                "branch, which brae does not carry.");
    }
    const scalar rDeltaT = (comp && !cn) ? comp->rDeltaT : scalar(0);
    // localEuler: the per-cell rDeltaT in the scalar's place (Compressible::rDeltaTCells)
    const std::vector<scalar>* rDeltaTCells = comp ? comp->rDeltaTCells : nullptr;
    if (rDeltaTCells)
    {
        if (cn || comp->V0)
            throw std::runtime_error(
                "brae kOmegaSST: a local time step (localEuler) beside CrankNicolson, or on a moving mesh, "
                "is not ported.");
        if (static_cast<label>(rDeltaTCells->size()) != nC)
            throw std::runtime_error("brae kOmegaSST: the local rDeltaT is not the mesh's cells.");
    }
    auto rDeltaTAt = [&](label cc) { return rDeltaTCells ? (*rDeltaTCells)[cc] : rDeltaT; };
    const std::vector<scalar> kOld     = (comp && comp->kOldIn)     ? *comp->kOldIn     : k.internal;
    const std::vector<scalar> omegaOld = (comp && comp->omegaOldIn) ? *comp->omegaOldIn : omega.internal;
    auto rhoOldAt = [&](label cc) { return (comp && comp->rhoOld) ? (*comp->rhoOld)[cc] : rhoAt(cc); };
    // this->nu() varies with temperature in the compressible lineage; the incompressible one has a
    // single constant.
    auto nuAt = [&](label cc) { return (comp && comp->nu) ? (*comp->nu)[cc] : nu; };

    // ---- production, from the CURRENT nut (the previous outer iteration's correctNut) -------------
    // fvc::grad(U) through the case's grad(U) scheme (kOmegaSSTBase.C:522): cellLimited where fvSchemes
    // says so. Unlimited here, naca0012 read omega 5.4e-03 / nut 1.2e-03 against OpenFOAM at t = 1.
    std::vector<tensor> gradU = co.gradULeastSq ? fvc::leastSquaresGrad(U, m, g, patches)
                                               : fvc::gaussGrad(U, m, g, patches);
    if (co.gradULimitK > 0.0) cellLimitGrad(gradU, U, co.gradULimitK, m, g, patches);
    const std::vector<scalar> s2  = S2(gradU);
    const std::vector<scalar> gb0 = GbyNu0(gradU);
    if (sd.on)
    {
        std::vector<scalar> flat;
        flat.reserve(gradU.size() * 9);
        for (const tensor& t : gradU)
        {
            flat.push_back(t.xx);
            flat.push_back(t.xy);
            flat.push_back(t.xz);
            flat.push_back(t.yx);
            flat.push_back(t.yy);
            flat.push_back(t.yz);
            flat.push_back(t.zx);
            flat.push_back(t.zy);
            flat.push_back(t.zz);
        }
        sd.components("gradU", flat, 9);
    }
    sd.scalars("S2", s2);
    sd.scalars("GbyNu0", gb0);
    std::vector<scalar> G(nC);
    for (label c = 0; c < nC; ++c) G[c] = nutF[c] * gb0[c];      // RAW GbyNu0 -- the k equation's G

    // divU takes the VOLUMETRIC flux -- fvc::absolute(this->phi(), U) -- while fvm::div and the
    // `bounded` Sp below take the MASS flux. Two different fields in the compressible lineage.
    const SurfaceScalarField& phiVol = (comp && comp->phiByRho) ? *comp->phiByRho : phi;
    std::vector<scalar> divU;
    if (comp && comp->meshPhi)
    {
        // fvc::absolute(phi, U): phi + mesh.phi() on every face, the boundary included
        SurfaceScalarField phiAbs = phiVol;
        for (std::size_t f = 0; f < phiAbs.internal.size(); ++f)
        {
            phiAbs.internal[f] += comp->meshPhi->internal[f];
        }
        for (std::size_t pi = 0; pi < phiAbs.boundary.size(); ++pi)
        {
            for (std::size_t i = 0; i < phiAbs.boundary[pi].size(); ++i)
            {
                phiAbs.boundary[pi][i] += comp->meshPhi->boundary[pi][i];
            }
        }
        divU = fvc::div(phiAbs, m, g, patches);
    }
    else
    {
        divU = fvc::div(phiVol, m, g, patches);
    }
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
        // mag(Uw.snGrad()), the wall patch's own (omegaWallFunctionFvPatchScalarField.C:253)
        const std::vector<vector> snUw = U.boundary[pi]->snGrad(U.internal);
        // THIS PATCH'S OWN COEFFICIENTS. `wallFunctionCoefficients` is constructed from the PATCH
        // dictionary in every wall function (wallFunctionCoefficients.C:68-80), with yPlusLam derived
        // from that patch's kappa and E, and omegaWallFunction reads its own `beta1` there too
        // (omegaWallFunctionFvPatchScalarField.C:404-409). None of them comes from the model dict. This
        // loop used the model-wide values and the interFoam reader REFUSED any patch that named its own
        // -- refusing a case OpenFOAM runs. The kEpsilon twin has read them per patch all along
        // (kEpsilon_cpp.cu:443).
        const WallFunctionCoeffs& owc = omega.boundary[pi]->wallCoeffs();
        const scalar oCmu25 = std::pow(owc.Cmu, 0.25);
        for (label i = 0; i < wp.size; ++i)
        {
            const label c = wp.faceCells[i];
            const scalar w = 1.0 / nw[c], kc = k.internal[c];
            // THE PARENTHESES ARE LOAD-BEARING. OpenFOAM is `6.0*nuw[facei]/(beta1_*sqr(y[facei]))`
            // (omegaWallFunctionFvPatchScalarField.C:218-225), i.e. beta1*(y*y); brae had (beta1*y)*y,
            // which is one ulp different on 5 of this fixture's 2268 cells. Reconstructed from
            // OpenFOAM's own dumped y and nu over the 154 single-wall-face cells: beta1*(y*y) reproduces
            // OpenFOAM on 154/154, (beta1*y)*y on 149/154. That ulp reaches the answer because the
            // pinned wall omega feeds limitedLinear's r = gradcf/gradf, and at call one the two cells of
            // one face are 3 and 4 ulp from equal, so the ratio is 0/0 in all but name: the limiter went
            // 0.295 against OpenFOAM's clipped 1.0 on face 2405 and moved that row's diagonal by
            // 2.3e-08. It showed up as 6.6e-09 of the first omega solve's normFactor -- the residual
            // itself agreed to 4.1e-15 -- and only at `writePrecision 17`; at the tutorial's 15 the two
            // omega values parse as identical.
            const scalar omegaVis = 6.0*nuFace[i]/(owc.beta1*(yw[i]*yw[i]));
            const scalar omegaLog = std::sqrt(kc)/(oCmu25*owc.kappa*yw[i]);
            const scalar magGradUw = mag(snUw[i]);
            om0[c] += w * std::sqrt(omegaVis*omegaVis + omegaLog*omegaLog);
            G0[c]  += w * (nutw[i] + nuFace[i]) * magGradUw * oCmu25 * std::sqrt(kc)
                        / (owc.kappa * yw[i]);
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
    // ...and the wall PATCH takes the fresh cell value here, not after the solve: OpenFOAM's
    // calculateTurbulenceFields ends with `opf == scalarField(omega0, opf.patch().faceCells())`
    // (omegaWallFunctionFvPatchScalarField.C:167-174), inside updateCoeffs, before the equation is
    // assembled. brae wrote the cells and left the patch's stored value at the previous evaluate -- the
    // file seed at iteration 1 -- so grad(omega) at the wall cells, which the corrected laplacian's
    // deferred correction interpolates onto the wall cells' inner faces, saw 10 where OpenFOAM saw 1598 on
    // naca0012, and the second ring of cells read omega 1.4e-04 off at t=1 (queue item 25). Invisible with
    // an orthogonal laplacian (5e-12 on the same case), which is what every SST fixture had.
    //
    // THE WALL-FUNCTION PATCHES AND NO OTHERS: that loop runs over the patches carrying
    // cornerWeights_, the omegaWallFunction ones. The inlet and outlet keep the values their last
    // evaluate left them, and CDkOmega's grad(omega) just below and the assembly's gradients read those.
    // This was omega.evaluateBoundary() -- every patch. Measured on squareBendLiq's geometry under
    // kOmegaSST with `linearUpwind limited`: omega 3.2e-06 off OpenFOAM at iteration 2 with U exact,
    // 1e-12 with only the wall patches updated. The kEpsilon twin carries the same fix and measurement.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (omega.boundary[pi]->isTurbulenceWallFunction())
        {
            omega.boundary[pi]->evaluate(omega.internal);
        }
    }
    if (res && res->captureStages) res->G = G;
    sd.scalars("G", G);

    // ---- CDkOmega, F1, F2 ------------------------------------------------------------------------
    // The case's gradScheme on each, as OpenFOAM resolves grad(k) and grad(omega) separately.
    // The base scheme first -- fvc::grad(k_) and fvc::grad(omega_) through the case's `grad(k)` and
    // `grad(omega)` entries (fvcGrad.C:149), Gauss linear or leastSquares -- from each field's STORED
    // patch values, then the cellLimited coefficient on top of either, as cellLimitedGrad wraps any base.
    std::vector<std::vector<scalar>> kbv(patches.size()), obv(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        kbv[pi] = k.boundary[pi]->value();
        obv[pi] = omega.boundary[pi]->value();
    }
    std::vector<vector> gradK  = co.gradKLeastSq ? fvc::leastSquaresGrad(k.internal, kbv, m, g, patches)
                                                 : fvc::gaussGrad(k.internal, kbv, m, g, patches);
    std::vector<vector> gradOm = oGrad.leastSquares ? fvc::leastSquaresGrad(omega.internal, obv, m, g, patches)
                                                 : fvc::gaussGrad(omega.internal, obv, m, g, patches);
    // EACH FIELD BY ITS OWN SCHEME: this block limited both with k's coefficient.
    if (co.gradKLimitK > 0.0)
    {
        cellLimitGrad(gradK,  k,     co.gradKLimitK, m, g, patches);
    }
    if (oGrad.cellLimitK > 0.0)
    {
        cellLimitGrad(gradOm, omega, oGrad.cellLimitK, m, g, patches);
    }
    // THE LIMITER'S OWN GRADIENT, dumped. CDkOmega is grad(k) & grad(omega) and grad(k) is exactly
    // zero while k is uniform, so CD matching says nothing about grad(omega) -- and grad(omega) is
    // what the TVD limiter's branch turns on. See interFoam/PORT.md, the 26-face finding.
    if (sd.on)
    {
        std::vector<scalar> fo(gradOm.size()*3), fk(gradK.size()*3);
        for (std::size_t c = 0; c < gradOm.size(); ++c)
        {
            fo[c*3+0] = gradOm[c].x; fo[c*3+1] = gradOm[c].y; fo[c*3+2] = gradOm[c].z;
            fk[c*3+0] = gradK[c].x;  fk[c*3+1] = gradK[c].y;  fk[c*3+2] = gradK[c].z;
        }
        sd.components("gradOmega", fo, 3);
        sd.components("gradKfield", fk, 3);
        sd.scalars("omegaAsm", omega.internal);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            sd.scalars(("omegaB_" + patches[pi].name).c_str(), obv[pi]);
    }
    const std::vector<scalar> CD  = CDkOmega(gradK, gradOm, omega.internal, co);
    // A per-cell nu, so F1/F2's viscous cross-over terms see the field the compressible lineage has.
    std::vector<scalar> nuCell(nC);
    for (label c = 0; c < nC; ++c) nuCell[c] = nuAt(c);
    std::vector<scalar> f1 = F1(k.internal, omega.internal, y, CD, nuCell, co);
    if (res && res->captureStages) { res->CD = CD; res->f1 = f1; }
    sd.scalars("CD", CD);
    sd.scalars("F1", f1);
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
    sd.scalars("F23", f23);

    // ---- the production limiter: omega uses the LIMITED GbyNu, k uses the raw G ------------------
    // kOmegaSSTBase.C reassigns GbyNu0 = GbyNu(GbyNu0, F23, S2) AFTER G was taken from the raw value.
    // Using one for both is the easy mistake here; they are different quantities.
    std::vector<scalar> gbLim(nC);
    for (label c = 0; c < nC; ++c)
        gbLim[c] = std::fmin(gb0[c],
                             (co.c1/co.a1)*co.betaStar*omega.internal[c]
                           * std::fmax(co.a1*omega.internal[c], co.b1*f23[c]*std::sqrt(s2[c])));
    sd.scalars("GbyNuLim", gbLim);


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

        // THE LIMITER'S WEIGHTS, beside the device's `omegaLimW`. Recomputed here with exactly what
        // divWithScheme computes internally -- its gradient is the field's own Gauss gradient from
        // the STORED patch values -- because the assembled matrix carries the laplacian and the
        // relaxation and cannot show the weights on their own.
        if (sd.on && limitedLinear)
        {
            std::vector<std::vector<scalar>> ob(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) ob[pi] = omega.boundary[pi]->value();
            const std::vector<vector> go = oGrad.leastSquares
                ? fvc::leastSquaresGrad(omega.internal, ob, m, g, patches)
                : fvc::gaussGrad(omega.internal, ob, m, g, patches);
            sd.scalars("omegaLimW",
                       ls::limitedLinearWeights(phi.internal, omega, go, limiterCoeff, m, g));
            sd.scalars("phiAsm", phi.internal);
        }
        FvScalarMatrix M = divWithScheme(phi, omega, oDiv.limitedLinear, oDiv.limiterCoeff, m, g, patches,
                                         oGrad.cellLimitK, oGrad.leastSquares, &sd, "omega");
        {
            // The laplacian with BOTH halves of `corrected`, then subtracted from the equation. The
            // explicit correction goes into the LAPLACIAN's own source first, so the -1.0 below carries
            // it into the transport equation with the right sign.
            FvScalarMatrix L = fvm::laplacian(Df, omega, m, g, patches, correctedLaplacian, nonOrthCoeffs);
            if (correctedLaplacian)
            {
                std::vector<std::vector<scalar>> vb(patches.size());
                for (std::size_t pi = 0; pi < patches.size(); ++pi) vb[pi] = omega.boundary[pi]->value();
                std::vector<vector> gradVf = oGrad.leastSquares ? fvc::leastSquaresGrad(omega.internal, vb, m, g, patches)
                                                             : fvc::gaussGrad(omega.internal, vb, m, g, patches);   // grad(omega)'s own scheme
                if (oGrad.cellLimitK > 0.0) cellLimitGrad(gradVf, omega.internal, vb, oGrad.cellLimitK, m, g, patches);
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
            const scalar rDTo = rDeltaTAt(c);
            if (rDTo > 0.0)   // fvm::ddt(alpha, rho, omega_), kOmegaSSTBase.C:572
            {
                M.diag[c]   += rDTo * rc * V;
                M.source[c] += rDTo * rhoOldAt(c) * omegaOld[c] * ((comp && comp->V0) ? (*comp->V0)[c] : V);
            }
            // - SuSp((F1 - 1)*CDkOmega/omega, omega)
            const scalar sp2 = rc * (f1[c] - 1.0) * CD[c] / omega.internal[c];
            M.diag[c]   += V * std::fmax(sp2, 0.0);
            M.source[c] -= V * std::fmin(sp2, 0.0) * omega.internal[c];
            // `bounded`: - Sp(fvc::div(phi), omega). Vanishes where phi is conservative, so it cannot
            // move a converged answer -- it is there to keep the transported scalar bounded on the way.
            if (oDiv.bounded) M.diag[c] -= divPhi[c] * V;
        }
        // fvm::ddt(alpha, rho, omega_) under CrankNicolson: "ddt0(rho,omega)" is the equation's own
        if (cn)
        {
            fv::fvmDdt(*comp->cn, *comp->cnDdt0Omega, comp->rho, comp->rhoOld ? comp->rhoOld : comp->rho,
                       comp->rhoOO, omegaOld, *comp->omegaOO, g.V(), M);
        }
        if (linearUpwind)
        {
            // linearUpwind's deferred correction. The caller SUBTRACTS what linearUpwindCorrection
            // returns -- see the sign note in fvm.cuh.
            std::vector<std::vector<scalar>> ob(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) ob[pi] = omega.boundary[pi]->value();
            std::vector<vector> gradVf = fvc::gaussGrad(omega.internal, ob, m, g, patches);
            // The gradient linearUpwind NAMES, where the caller resolved it (see luGradLimitK).
            const scalar luK = (co.luGradLimitK >= 0.0) ? co.luGradLimitK : oGrad.cellLimitK;
            if (luK > 0.0) cellLimitGrad(gradVf, omega.internal, ob, luK, m, g, patches);
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
            captureSSTSystem(M, patches, res->omD, res->omSrc, &res->omUpper, &res->omLower,
                             &res->omIfc);
        // ...and WRITTEN HERE, at the call the stage dump latched, not at every call from the driver.
        // The driver's own write overwrote its files on every closure call, so a ten-step run left the
        // system of step TEN beside device columns latched at step ONE: the comparison then reported
        // 28,000 differing diagonals that were only two different iterations.
        if (sd.on && res)
        {
            sd.scalars("omD", res->omD);       sd.scalars("omSrc", res->omSrc);
            sd.scalars("omUpper", res->omUpper); sd.scalars("omLower", res->omLower);
            sd.scalars("omIfc", res->omIfc);
            sd.scalars("omD0", res->omD0);     sd.scalars("omSrc0", res->omSrc0);
        }
        const SolverPerformance po = solveSecond(M, omega.internal);
        if (res)
        {
            res->omega = po.initialResidual;
            res->omegaPerf = po;
        }
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
        bound(omega, co.omegaMin, m, g, patches, "omega");
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

        FvScalarMatrix M = divWithScheme(phi, k, limitedLinear, limiterCoeff, m, g, patches,
                                         co.gradKLimitK, co.gradKLeastSq, &sd, "k");
        {
            // The laplacian with BOTH halves of `corrected`, then subtracted from the equation. The
            // explicit correction goes into the LAPLACIAN's own source first, so the -1.0 below carries
            // it into the transport equation with the right sign.
            FvScalarMatrix L = fvm::laplacian(Df, k, m, g, patches, correctedLaplacian, nonOrthCoeffs);
            if (correctedLaplacian)
            {
                std::vector<std::vector<scalar>> vb(patches.size());
                for (std::size_t pi = 0; pi < patches.size(); ++pi) vb[pi] = k.boundary[pi]->value();
                std::vector<vector> gradVf = co.gradKLeastSq ? fvc::leastSquaresGrad(k.internal, vb, m, g, patches)
                                                             : fvc::gaussGrad(k.internal, vb, m, g, patches);   // grad(k)'s own scheme
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
            const scalar rDTk = rDeltaTAt(c);
            if (rDTk > 0.0)   // fvm::ddt(alpha, rho, k_), kOmegaSSTBase.C:602
            {
                M.diag[c]   += rDTk * rc * V;
                M.source[c] += rDTk * rhoOldAt(c) * kOld[c] * ((comp && comp->V0) ? (*comp->V0)[c] : V);
            }
            if (bounded) M.diag[c] -= divPhi[c] * V;                 // - Sp(fvc::div(alphaRhoPhi), k)
        }
        // fvm::ddt(alpha, rho, k_) under CrankNicolson: "ddt0(rho,k)" is this equation's own
        if (cn)
        {
            fv::fvmDdt(*comp->cn, *comp->cnDdt0K, comp->rho, comp->rhoOld ? comp->rhoOld : comp->rho,
                       comp->rhoOO, kOld, *comp->kOO, g.V(), M);
        }
        if (linearUpwind)
        {
            std::vector<std::vector<scalar>> kb(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) kb[pi] = k.boundary[pi]->value();
            std::vector<vector> gradVf = fvc::gaussGrad(k.internal, kb, m, g, patches);
            const scalar luK = (co.luGradLimitK >= 0.0) ? co.luGradLimitK : co.gradKLimitK;
            if (luK > 0.0) cellLimitGrad(gradVf, k.internal, kb, luK, m, g, patches);
            const std::vector<scalar> corr =
                fvm::linearUpwindCorrection<scalar, vector>(phi.internal, gradVf, m, g);
            for (label c = 0; c < nC; ++c) M.source[c] -= corr[c];
        }
        if (res && res->captureStages)
            captureSSTSystem(M, patches, res->kD0, res->kSrc0);
        if (relaxEquationK) relaxMatrix(M, k, m, patches, relaxK);
        if (res && res->captureStages)
            captureSSTSystem(M, patches, res->kD, res->kSrc, &res->kUpper, &res->kLower,
                             &res->kIfc);
        if (sd.on && res)
        {
            sd.scalars("kD", res->kD);       sd.scalars("kSrc", res->kSrc);
            sd.scalars("kUpper", res->kUpper); sd.scalars("kLower", res->kLower);
            sd.scalars("kIfc", res->kIfc);
        }
        const SolverPerformance pk = solveScalar(M, k.internal);
        if (res)
        {
            res->k = pk.initialResidual;
            res->kPerf = pk;
        }
        if (std::getenv("BRAE_SST_DEBUG"))
            std::printf("    [k] init=%.3e final=%.3e nIter=%d\n",
                        pk.initialResidual, pk.finalResidual, pk.nIterations);
        k.evaluateBoundary();
        bound(k, co.kMin, m, g, patches, "k");   // Foam::bound(k_, kMin_)
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
        // nutkWallFunction's own coefficients, from the nut PATCH's dictionary -- as the kEpsilon twin
        // reads them (kEpsilon_cpp.cu:912)
        const WallFunctionCoeffs& nwc = nutField.boundary[pi]->wallCoeffs();
        // nutkRoughWallFunction dispatches on the PATCH's class, as OpenFOAM's virtual calcNut does. It reads
        // the patch's current value -- the previous correctNut's, or the file's before the first -- because
        // its limiter is relative to it; the stored value is never reset between calls (the fixedValue
        // operator= is empty, so the field assignment above does not touch it).
        const std::vector<scalar>* Ks = nutField.boundary[pi]->nutkRoughKs();
        // A GATE'S CONTROL, reachable only on a rough patch: run it as nutkWallFunction. WRONG.
        if (Ks && std::getenv("BRAE_CONTROL_NUTK_SMOOTH"))
        {
            std::printf("  *** CONTROL MODE: nutkRoughWallFunction on `%s` runs as nutkWallFunction. This run is "
                        "deliberately wrong. ***\n", patches[pi].name.c_str());
            Ks = nullptr;
        }
        if (Ks)
        {
            std::vector<scalar> nutPrev = nutField.boundary[pi]->value();
            // A GATE'S CONTROL: the limiter against nu_w alone -- the history dropped. WRONG.
            if (std::getenv("BRAE_CONTROL_NUTKROUGH_NOHISTORY"))
            {
                std::printf("  *** CONTROL MODE: nutkRoughWallFunction on `%s` limits against nu_w, not its "
                            "previous value. This run is deliberately wrong. ***\n", patches[pi].name.c_str());
                std::fill(nutPrev.begin(), nutPrev.end(), scalar(0));
            }
            nutField.boundary[pi]->setValue(
                nutkRoughWallFunction(patches[pi], yWall[pi], k.internal,
                                      comp && comp->nuBnd ? (*comp->nuBnd)[pi]
                                                          : std::vector<scalar>(patches[pi].size, nu),
                                      nutPrev, *Ks,
                                      *nutField.boundary[pi]->nutkRoughCs(),
                                      nwc.Cmu, nwc.kappa, nwc.E));
            continue;
        }
        nutField.boundary[pi]->setValue(
            nutkWallFunction(patches[pi], yWall[pi], k.internal,
                             comp && comp->nuBnd ? (*comp->nuBnd)[pi]
                                                 : std::vector<scalar>(patches[pi].size, nu),
                             nwc.Cmu, nwc.kappa, nwc.E));
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
        // an inletOutlet nut (or a relative): OpenFOAM's correctBoundaryConditions evaluates it against the
        // cell nut just assigned, inflow faces taking the inletValue. MEASURED on RAS/waterChannel under
        // kOmegaSST with `outlet inletOutlet; inletValue 0.002`: skipped, nut 8.5e-04 from OpenFOAM after
        // ten steps and U 9.2e-07 (tests/interfoam_waterchannel_vs_openfoam.sh `nutOutlet`).
        // A COUPLED PATCH TAKES THE TWO CELLS, not Cmu*k_b^2/eps_b. correctNut() ends with
        // nut_.correctBoundaryConditions() (kEpsilon.C:45-46), and on a constraint patch that is
        // coupledFvPatchField::evaluate -- lerp(patchNeighbourField, patchInternalField, weights),
        // i.e. w*nut_own + (1 - w)*nut_nbr of the nut JUST ASSIGNED. It is not the same number as the
        // assignment's: Cmu*k_b^2/eps_b is a non-linear function of k and epsilon's own interpolated
        // patch values, and interpolate(Cmu*k^2/eps) is the interpolation of the result.
        if (patches[pi].coupled)
        {
            nutField.boundary[pi]->evaluate(nutF);
            continue;
        }
        if (nutField.boundary[pi]->isInletOutlet())
        {
            if (!comp || !comp->nutPhi || comp->nutPhi->boundary.size() <= pi)
            {
                throw std::runtime_error(
                    "brae kOmegaSST: nut on patch `" + patches[pi].name + "` is flux-conditional (inletOutlet) and the "
                    "caller handed the closure no flux to decide inflow by (Compressible::nutPhi).");
            }
            nutField.boundary[pi]->updateFromFlux(comp->nutPhi->boundary[pi]);
            nutField.boundary[pi]->evaluate(nutF);
            continue;
        }
        // ...and every other patch that is not `calculated`: correctBoundaryConditions evaluates it --
        // a zeroGradient or symmetry takes the new cell nut, a fixedValue keeps the value the case pinned,
        // a coupled patch interpolates. MEASURED on RAS/waterChannel with the inlet's nut zeroGradient:
        // skipped, it kept the value it was built with (0) where OpenFOAM holds the cell's, and U was
        // 8.7e-05 from OpenFOAM after ten steps (tests/interfoam_waterchannel_vs_openfoam.sh `nutInletZeroGrad`).
        // An EMPTY patch is not among them: OpenFOAM's emptyFvPatchField has size 0, so there is nothing
        // to evaluate, and brae keeps the faces only for its addressing. Evaluating it copied the cell nut
        // into faces OpenFOAM does not have -- which the kEpsilon closure and the device closure leave
        // alone -- and put rhoSimpleFoam's device-against-host alphat boundary gate 2.1e-01 out on
        // validation/rhoSST's frontBack while every field agreed to 1e-13 (tests/rho_step_cuda_euler.sh).
        if (patches[pi].type == "empty")
        {
            continue;
        }
        if (nutField.boundary[pi]->bcCategory() != 2)
        {
            nutField.boundary[pi]->evaluate(nutF);
            continue;
        }

        const std::vector<scalar>& kb = k.boundary[pi]->value();
        const std::vector<scalar>& ob = omega.boundary[pi]->value();
        // THE PATCH'S OWN snGrad(): gaussGrad::correctBoundaryConditions replaces the boundary
        // gradient's normal component with `vsf.boundaryField()[patchi].snGrad()` (gaussGrad.C:164-168),
        // and that is a virtual. On a directionMixed patch it is built from the valueFraction and the
        // cell, NOT from the stored value: interFoam's waterChannel opens its top through a
        // pressureInletOutletVelocity whose file value is (0 0 0) over cells moving at (1 0 0), with a
        // valueFraction of zero at construction, so OpenFOAM's snGrad there is 0 and its nut k/omega =
        // 3.33e-02, where (value - cell)*deltaCoeffs read a shear of 1/d and gave 3.7e-05 -- the whole
        // patch 100% out, and the first p_rgh residual 7.1e-03 with it.
        const std::vector<vector> snU = U.boundary[pi]->snGrad(U.internal);
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
            const vector sng = snU[i];
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
