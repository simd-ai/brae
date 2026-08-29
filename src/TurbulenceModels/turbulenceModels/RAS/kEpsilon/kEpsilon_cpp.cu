// _cpp REFERENCE implementation -- see kEpsilon_cpp.cuh for the OpenFOAM provenance and the wall note.
#include "kEpsilon_cpp.cuh"
#include "bound_cpp.cuh"
#include "nut_wall_function.cuh"
#include "near_wall_dist.cuh"
#include "pbicgstab.cuh"
#include <cmath>
#include <algorithm>
#include <vector>

namespace brae {
namespace cpu {
namespace kEpsilonRef {

namespace {

// D() and the full right-hand side of an assembled system, in the SAME form tools/dumpKEpsilon writes for
// OpenFOAM's: the diagonal including the boundary internalCoeffs, and the source with boundaryCoeffs
// folded into their face cells.
void captureSystem(
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


// DkEff / DepsilonEff as OpenFOAM builds them: `nut_/sigma + nu()` is a volScalarField, so its BOUNDARY
// value comes from nut's OWN boundary, not from the adjacent cell. fvm::laplacian then takes that
// boundary value for the patch coefficients.
//
// brae's fvc::interpolate gives a surface field whose boundary is the CELL value, which is right for an
// extrapolatedCalculated field like rAU but wrong here: at pitzDaily's inlet OpenFOAM's nut_b is
// Cmu*k_b^2/eps_b = 8.5e-04 from the inlet's fixed k and epsilon, while the adjacent cell's nut is
// several times that. It is the same distinction the momentum path already makes with nuEffBnd -- the
// turbulence path simply never made it.
SurfaceScalarField effectiveDiffusivity(
    const std::vector<scalar>& nutCell,
    const std::vector<scalar>& nutBndFlatPerPatch,
    const GeometricField<scalar>& nutField,
    scalar sigma,
    scalar nu,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    // The compressible lineage multiplies the whole thing by rho and carries a nu that varies with T.
    // Null throughout is the incompressible reading and reproduces the previous arithmetic exactly.
    const std::vector<scalar>*              rho    = nullptr,
    const std::vector<std::vector<scalar>>* rhoBnd = nullptr,
    const std::vector<scalar>*              nuFld  = nullptr,
    const std::vector<std::vector<scalar>>* nuBnd  = nullptr,
    // OF's DkEff()/DepsilonEff(), i.e. WITHOUT the rho the compressible form multiplies in. Recorded for
    // comparison against the model's own, nothing more.
    std::vector<scalar>*                    DEffOut = nullptr)
{
    (void)nutBndFlatPerPatch;
    std::vector<scalar> D(nutCell.size());
    if (DEffOut) DEffOut->resize(nutCell.size());
    for (std::size_t c = 0; c < nutCell.size(); ++c)
    {
        const scalar nuc = nuFld ? (*nuFld)[c] : nu;
        if (DEffOut) (*DEffOut)[c] = nutCell[c] / sigma + nuc;
        D[c] = (nutCell[c] / sigma + nuc) * (rho ? (*rho)[c] : 1.0);
    }
    SurfaceScalarField sf = fvc::interpolate(D, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& nb = nutField.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const scalar nub = nuBnd ? (*nuBnd)[pi][i] : nu;
            sf.boundary[pi][i] = (nb[i] / sigma + nub) * (rhoBnd ? (*rhoBnd)[pi][i] : 1.0);
        }
    }
    return sf;
}

} // namespace

std::vector<scalar> GbyNu(const std::vector<tensor>& gradU)
{
    std::vector<scalar> out(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        const tensor& t = gradU[c];
        const scalar tr = t.xx + t.yy + t.zz;

        // devTwoSymm(t) = (t + t^T) - (2/3)*tr(t)*I
        const scalar d[9] =
        {
            2*t.xx - (2.0/3.0)*tr,
            t.xy + t.yx,
            t.xz + t.zx,
            t.yx + t.xy,
            2*t.yy - (2.0/3.0)*tr,
            t.yz + t.zy,
            t.zx + t.xz,
            t.zy + t.yz,
            2*t.zz - (2.0/3.0)*tr
        };
        const scalar gg[9] =
        {
            t.xx, t.xy, t.xz,
            t.yx, t.yy, t.yz,
            t.zx, t.zy, t.zz
        };

        scalar s = 0;
        for (int q = 0; q < 9; ++q)
        {
            s += gg[q] * d[q];
        }
        out[c] = s;
    }
    return out;
}


std::vector<scalar> correctNut(
    const std::vector<scalar>& k,
    const std::vector<scalar>& epsilon,
    const KEpsilonCoeffs& co)
{
    std::vector<scalar> out(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
    {
        out[c] = co.Cmu * k[c] * k[c] / epsilon[c];
    }
    return out;
}


void correct(
    const GeometricField<vector>& U,
    GeometricField<scalar>& k,
    GeometricField<scalar>& epsilon,
    GeometricField<scalar>& nutField,
    const SurfaceScalarField& phi,
    scalar nu,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar relaxEps,
    scalar relaxK,
    scalar tol,
    scalar relTol,
    int maxIter,
    const KEpsilonCoeffs& co,
    KEResiduals* res,
    bool bounded,
    int dropTerm,
    const Compressible* comp,
    const cpu::fvOptions::OptionList* fvOpts,
    bool relaxEquationEps,
    bool relaxEquationK)
{
    const label nC = m.nCells();
    const scalar Cmu25 = std::pow(co.Cmu, 0.25);
    const scalar Cmu75 = std::pow(co.Cmu, 0.75);
    std::vector<scalar>& nutF = nutField.internal;

    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    const std::vector<scalar> gByNu = GbyNu(gradU);
    // divU is the DILATATION, so it comes from the VOLUMETRIC flux -- which for the compressible lineage
    // is phi/interpolate(rho) and not the mass flux the div operator uses. `bounded` instead subtracts
    // the divergence of the EQUATION's own flux. In the incompressible lineage both are div(phi) and the
    // two lines below collapse to the one they replace.
    const std::vector<scalar> divU =
        (comp && comp->phiByRho) ? fvc::div(*comp->phiByRho, m, g, patches)
                                 : fvc::div(phi, m, g, patches);
    const std::vector<scalar> divPhi =
        (comp && comp->phiByRho) ? fvc::div(phi, m, g, patches) : divU;

    // alpha*rho on a cell: 1 in the incompressible lineage.
    auto rhoAt = [&](label c) { return (comp && comp->rho) ? (*comp->rho)[c] : scalar(1.0); };

    std::vector<scalar> G(nC);
    for (label c = 0; c < nC; ++c)
    {
        G[c] = nutF[c] * gByNu[c];
    }

    if (res && res->captureStages)
    {
        res->gradU  = gradU;
        res->divU   = divU;
        res->divPhi = divPhi;
        res->gByNu  = gByNu;
        res->G      = G;     // BEFORE the wall replacement below, which is where OpenFOAM writes it too
    }

    // createAveragingWeights: count the adjacent faces that carry an epsilonWallFunction PATCH FIELD.
    // brae's boundary factory maps epsilonWallFunction to zeroGradient and applies the constraint here,
    // so the discriminator available at this level is the patch's own wall-ness together with epsilon's
    // BC category; patch type alone would also count a `wall` carrying a plain fixedValue epsilon.
    std::vector<label> nw(nC, 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!epsilon.boundary[pi]->isTurbulenceWallFunction()) continue;
        for (label i = 0; i < patches[pi].size; ++i)
        {
            ++nw[patches[pi].faceCells[i]];
        }
    }

    const std::vector<std::vector<scalar>> yWall = nearWallDist(m, g, patches);
    std::vector<scalar> eps0(nC, 0.0);
    std::vector<scalar> G0(nC, 0.0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!epsilon.boundary[pi]->isTurbulenceWallFunction()) continue;

        const FvPatch& wp = patches[pi];
        const std::vector<scalar>& yw = yWall[pi];
        // nu AT THE WALL FACE. The incompressible lineage has one number for the whole domain; the
        // compressible one has mu(T)/rho, which differs face by face on a wall with a temperature
        // gradient -- and it is nu_w, not the cell's, that the wall function is written in terms of.
        auto nuAtFace = [&](label i)
        {
            return (comp && comp->nuBnd) ? (*comp->nuBnd)[pi][i] : nu;
        };
        std::vector<scalar> nuFace(wp.size);
        for (label i = 0; i < wp.size; ++i) nuFace[i] = nuAtFace(i);
        const std::vector<scalar> nutw =
            nutkWallFunction(wp, yw, k.internal, nuFace, co.Cmu, co.kappa, co.E);
        const std::vector<vector>& Uw = U.boundary[pi]->value();

        for (label i = 0; i < wp.size; ++i)
        {
            const label c = wp.faceCells[i];
            const scalar w = 1.0 / static_cast<scalar>(nw[c]);
            const scalar kc = k.internal[c];
            const scalar magGradUw = mag((Uw[i] - U.internal[c]) * wp.deltaCoeffs[i]);

            // epsilonWallFunction, STEPWISE blender (its default: wallFunctionBlenders(dict,
            // blenderType::STEPWISE, 2)). Without lowReCorrection the log branch is taken on every face,
            // which is what this did unconditionally before.
            const scalar yPlus = Cmu25 * yw[i] * std::sqrt(kc) / nuAtFace(i);
            const scalar yPlusLam = brae::yPlusLam(co.kappa, co.E);
            const bool   resolved = co.epsLowRe && (yPlus < yPlusLam);
            eps0[c] += resolved ? w * 2.0 * kc * nuAtFace(i) / (yw[i] * yw[i])               // epsilonVis
                                : w * Cmu75 * std::pow(kc, 1.5) / (co.kappa * yw[i]);        // epsilonLog
            // ...and the production override is SKIPPED ENTIRELY on a resolved face -- OF's guard is
            // `if (!lowReCorrection_ || (yPlus > yPlusLam))`, not a scaling of the same term.
            if (!resolved)
                G0[c] += w * (nutw[i] + nuAtFace(i)) * magGradUw * Cmu25 * std::sqrt(kc) / (co.kappa * yw[i]);
        }
    }

    std::vector<label> wallCells;
    std::vector<scalar> epsVals;
    for (label c = 0; c < nC; ++c)
    {
        if (nw[c] == 0) continue;
        G[c] = G0[c];
        epsilon.internal[c] = eps0[c];
        wallCells.push_back(c);
        epsVals.push_back(eps0[c]);
    }
    if (res) res->wallCells = static_cast<label>(wallCells.size());
    if (res && res->captureStages)
    {
        res->eps0  = eps0;
        res->G0    = G0;
        res->Gwall = G;      // AFTER the override -- the field the k equation actually transports
        res->isWallCell.assign(nC, 0);
        for (label c = 0; c < nC; ++c) res->isWallCell[c] = (nw[c] != 0) ? 1 : 0;
    }

    // epsilon equation
    {
        SurfaceScalarField Df =
            effectiveDiffusivity(nutF, {}, nutField, co.sigmaEps, nu, m, g, patches,
                                 comp ? comp->rho : nullptr, comp ? comp->rhoBnd : nullptr,
                                 comp ? comp->nu  : nullptr, comp ? comp->nuBnd  : nullptr,
                                 (res && res->captureStages) ? &res->DepsilonEff : nullptr);
        // The DIAGNOSTIC drop, applied to the assembled diffusivity rather than to the cell nut it is
        // built from. Zeroing `nutF` alone left the term standing: effectiveDiffusivity adds the LAMINAR
        // nu to every cell coefficient and reads the untouched nut BOUNDARY field for every patch face,
        // so the laplacian was still fully assembled and the sweep reported that dropping diffusion
        // barely moved the answer -- a diagnostic that could not name the term it was named after.
        if (dropTerm == 4)
        {
            std::fill(Df.internal.begin(), Df.internal.end(), 0.0);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                std::fill(Df.boundary[pi].begin(), Df.boundary[pi].end(), 0.0);
        }

        // fvMatrix's constructor calls psi.boundaryFieldRef().updateCoeffs() -- that is where OpenFOAM's
        // flux-conditional boundaries (inletOutlet, and the turbulentMixingLength/Intensity inlets derived
        // from it) read the flux and set their valueFraction. Without it every such patch keeps the
        // valueFraction it was seeded with and contributes nothing to the system, whatever value it
        // carries. The flux is the one this equation is convected by, which is the field the BC names.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // OF kEpsilon.C: epsilon_.boundaryFieldRef().updateCoeffs() immediately before the equation.
            // turbulentMixingLengthDissipationRateInlet reads k's CURRENT patch values there.
            epsilon.boundary[pi]->updateTurbulentInlet({}, k.boundary[pi]->value(), co.Cmu);
            epsilon.boundary[pi]->updateFromFlux(phi.boundary[pi]);
        }

        FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, epsilon, m, patches);
        if (res && res->captureStages)
        {
            captureSystem(M, patches, res->epsDivD, res->epsDivSrc, &res->epsDivUpper, &res->epsDivLower);
            // `bounded Gauss upwind` is ONE scheme: boundedConvectionScheme wraps the Gauss operator and
            // subtracts fvm::Sp(surfaceIntegrate(phi), vf). OpenFOAM's fvm::div returns the wrapped
            // matrix, so the capture has to include the bounded term to be the same object -- brae adds
            // it a few lines below, on the combined matrix.
            if (bounded)
                for (label c = 0; c < nC; ++c) res->epsDivD[c] -= divPhi[c] * g.V()[c];
            res->gammaEpsFace = Df.internal;
        }
        {
            // `Gauss linear corrected` changes TWO things, and kOmegaSST in this same directory already
            // does both: the implicit face coefficient becomes gamma*nonOrthDeltaCoeffs*magSf, and the
            // non-orthogonal part enters as an explicit source. Assembling the orthogonal laplacian
            // instead is a silent scheme substitution -- the case asked for `corrected` -- and on this
            // near-orthogonal mesh it moved the off-diagonals by only 2.2e-06 while moving epsilon by
            // 6.5e-06, small enough to look like round-off and large enough not to be.
            FvScalarMatrix L = fvm::laplacian(Df, epsilon, m, g, patches, co.correctedLaplacian);
            if (co.correctedLaplacian)
            {
                std::vector<std::vector<scalar>> vb(patches.size());
                for (std::size_t pi = 0; pi < patches.size(); ++pi) vb[pi] = epsilon.boundary[pi]->value();
                const std::vector<vector> gradVf = fvc::gaussGrad(epsilon.internal, vb, m, g, patches);
                const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                    Df, epsilon, gradVf, m, g, patches, co.snGradLimitCoeff);
                for (label c = 0; c < nC; ++c) L.source[c] -= corr[c];
            }
            if (res && res->captureStages) captureSystem(L, patches, res->epsLapD, res->epsLapSrc,
                                   &res->epsLapUpper, &res->epsLapLower);
            addEqual(M, L, -1.0);
        }

        for (label c = 0; c < nC; ++c)
        {
            const scalar V = g.V()[c];

            // == C1*GbyNu*Cmu*k
            if (dropTerm != 1) M.source[c] += co.C1 * rhoAt(c) * gByNu[c] * co.Cmu * k.internal[c] * V;

            // - SuSp(((2/3)*C1 - C3)*divU, epsilon)
            const scalar sp = (dropTerm == 2) ? 0.0 : ((2.0/3.0) * co.C1 - co.C3) * rhoAt(c) * divU[c];
            M.diag[c]   += V * std::fmax(sp, 0.0);
            M.source[c] -= V * std::fmin(sp, 0.0) * epsilon.internal[c];

            // - Sp(C2*epsilon/k, epsilon)
            if (dropTerm != 3) M.diag[c] += co.C2 * rhoAt(c) * epsilon.internal[c] / k.internal[c] * V;

            // `bounded`: - Sp(div(phi), epsilon). Vanishes where phi is conservative, so it cannot move
            // a converged state -- which is exactly why it needs its own measurement rather than being
            // assumed harmless.
            if (bounded) M.diag[c] -= divPhi[c] * V;
        }

        if (res && res->captureStages) captureSystem(M, patches, res->epsD0, res->epsSrc0);
        // kEpsilon.C:265-267 -- relax(), THEN fvOptions.constrain(), THEN boundaryManipulate(). These two
        // were the other way round here, and the difference is not the constrained cell itself: both
        // orders leave it at the fvOption's value, because setValues writes psi and OpenFOAM's
        // boundaryManipulate re-reads patchInternalField() after the option has written it. What differs
        // is the NEIGHBOURS. setValues does `source_[nei] -= coeff*value` and then zeroes that coeff
        // (fvMatrix.C:259-291), so only the FIRST setValues touching a cell transfers anything -- with
        // the wall first, a neighbour of a cell that is both wall-adjacent and fvOption-constrained
        // received -coeff*epsilon0 where OpenFOAM gives it -coeff*(the option's value).
        //
        // UNMEASURED, and saying so is the point. No validation case carries an fvOption on k or epsilon
        // -- sbMatched has no fvOptions file at all, and the three cases that do constrain momentum or
        // temperature (simpleCar explicitPorositySource, turbineSiting actuationDiskSource, naca0012
        // limitTemperature). So `fvOpts` is null on every fixture, this swap is a no-op on all of them,
        // and rho_kepsilon_vs_openfoam passing across it is evidence of no harm, not of a fix. The
        // discriminating fixture -- a wall-adjacent cell inside an fvOption cell set -- does not exist
        // yet; until it does this order is asserted by OpenFOAM's source and by nothing that runs.
        // "the case names a factor", not "the factor is below 1" -- see the header. Unconditional
        // relaxation applies a dominance clamp OpenFOAM does not apply when fvSolution names nothing.
        if (relaxEquationEps) relaxMatrix(M, epsilon, m, patches, relaxEps);
        if (fvOpts) cpu::fvOptions::constrain(*fvOpts, M, epsilon.internal, "epsilon", m, patches);
        setValues(M, epsilon.internal, m, patches, wallCells, epsVals);
        if (res)
        {
            // |b - A.psi| per cell, on the SAME assembled matrix the solve is about to use, before the
            // solve changes psi. Boundary coefficients are folded in exactly as fvMatrix::solve does.
            std::vector<scalar> diagC = M.diag;
            std::vector<scalar> b = M.source;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const label c = patches[pi].faceCells[i];
                    diagC[c] += M.internalCoeffs[pi][i];
                    b[c]     += M.boundaryCoeffs[pi][i];
                }
            }
            std::vector<scalar> r(nC);
            for (label c = 0; c < nC; ++c)
            {
                r[c] = b[c] - diagC[c] * epsilon.internal[c];
            }
            const std::vector<label>& own = m.owner();
            const std::vector<label>& nei = m.neighbour();
            for (label f = 0; f < m.nInternalFaces(); ++f)
            {
                r[own[f]] -= M.upper[f] * epsilon.internal[nei[f]];
                r[nei[f]] -= M.lower[f] * epsilon.internal[own[f]];
            }
            for (label c = 0; c < nC; ++c)
            {
                r[c] = std::fabs(r[c]);
            }
            res->epsCellResidual = r;
        }
        if (res && res->captureStages)
        {
            captureSystem(M, patches, res->epsD, res->epsSrc, &res->epsUpper, &res->epsLower);
        }
        const SolverPerformance p = pbicgstab(M, epsilon.internal, m, patches, tol, relTol, maxIter);
        if (res) res->epsilon = p.initialResidual;

        // Foam::bound(epsilon_, epsilonMin_): a cell that solved NEGATIVE takes its neighbours'
        // average, not a floor. Inert under upwind convection, which does not produce one -- see
        // bound_cpp.cuh for where a floor is fatal.
        epsilon.evaluateBoundary();
        bound(epsilon, 1e-15, m, g, patches);
    }

    // k equation
    {
        SurfaceScalarField Df =
            effectiveDiffusivity(nutF, {}, nutField, co.sigmaK, nu, m, g, patches,
                                 comp ? comp->rho : nullptr, comp ? comp->rhoBnd : nullptr,
                                 comp ? comp->nu  : nullptr, comp ? comp->nuBnd  : nullptr,
                                 (res && res->captureStages) ? &res->DkEff : nullptr);
        // The DIAGNOSTIC drop, applied to the assembled diffusivity rather than to the cell nut it is
        // built from. Zeroing `nutF` alone left the term standing: effectiveDiffusivity adds the LAMINAR
        // nu to every cell coefficient and reads the untouched nut BOUNDARY field for every patch face,
        // so the laplacian was still fully assembled and the sweep reported that dropping diffusion
        // barely moved the answer -- a diagnostic that could not name the term it was named after.
        if (dropTerm == 8)
        {
            std::fill(Df.internal.begin(), Df.internal.end(), 0.0);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                std::fill(Df.boundary[pi].begin(), Df.boundary[pi].end(), 0.0);
        }

        // fvMatrix's constructor calls psi.boundaryFieldRef().updateCoeffs() -- that is where OpenFOAM's
        // flux-conditional boundaries (inletOutlet, and the turbulentMixingLength/Intensity inlets derived
        // from it) read the flux and set their valueFraction. Without it every such patch keeps the
        // valueFraction it was seeded with and contributes nothing to the system, whatever value it
        // carries. The flux is the one this equation is convected by, which is the field the BC names.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // turbulentIntensityKineticEnergyInlet reads U's patch values.
            k.boundary[pi]->updateTurbulentInlet(U.boundary[pi]->value(), {}, co.Cmu);
            k.boundary[pi]->updateFromFlux(phi.boundary[pi]);
        }

        FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, k, m, patches);
        {
            // `Gauss linear corrected` changes TWO things, and kOmegaSST in this same directory already
            // does both: the implicit face coefficient becomes gamma*nonOrthDeltaCoeffs*magSf, and the
            // non-orthogonal part enters as an explicit source. Assembling the orthogonal laplacian
            // instead is a silent scheme substitution -- the case asked for `corrected` -- and on this
            // near-orthogonal mesh it moved the off-diagonals by only 2.2e-06 while moving epsilon by
            // 6.5e-06, small enough to look like round-off and large enough not to be.
            FvScalarMatrix L = fvm::laplacian(Df, k, m, g, patches, co.correctedLaplacian);
            if (co.correctedLaplacian)
            {
                std::vector<std::vector<scalar>> vb(patches.size());
                for (std::size_t pi = 0; pi < patches.size(); ++pi) vb[pi] = k.boundary[pi]->value();
                const std::vector<vector> gradVf = fvc::gaussGrad(k.internal, vb, m, g, patches);
                const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                    Df, k, gradVf, m, g, patches, co.snGradLimitCoeff);
                for (label c = 0; c < nC; ++c) L.source[c] -= corr[c];
            }
            addEqual(M, L, -1.0);
        }

        for (label c = 0; c < nC; ++c)
        {
            const scalar V = g.V()[c];

            // == G   (the wall-overridden value at wall cells)
            if (dropTerm != 5) M.source[c] += rhoAt(c) * G[c] * V;

            // - SuSp((2/3)*divU, k)
            const scalar sp = (dropTerm == 6) ? 0.0 : (2.0/3.0) * rhoAt(c) * divU[c];
            M.diag[c]   += V * std::fmax(sp, 0.0);
            M.source[c] -= V * std::fmin(sp, 0.0) * k.internal[c];

            // - Sp(alpha*rho*epsilon/k, k). The rho is NOT optional and its absence is not visible in
            // the incompressible lineage, where it is 1: here rho is ~0.38, so leaving it out made k's
            // destruction 2.6x too strong and k wrong across the whole field (3.3e-01 against OpenFOAM)
            // while epsilon -- solved first, and correctly rho-weighted -- looked far better.
            if (dropTerm != 7) M.diag[c] += rhoAt(c) * epsilon.internal[c] / k.internal[c] * V;

            if (bounded) M.diag[c] -= divPhi[c] * V;
        }

        if (res && res->captureStages) captureSystem(M, patches, res->kD0, res->kSrc0);
        if (relaxEquationK) relaxMatrix(M, k, m, patches, relaxK);
        // fvOptions.constrain(kEqn), kEpsilon.C -- after relax(), as OpenFOAM has it.
        if (fvOpts) cpu::fvOptions::constrain(*fvOpts, M, k.internal, "k", m, patches);
        if (res && res->captureStages)
        {
            captureSystem(M, patches, res->kD, res->kSrc, &res->kUpper, &res->kLower);
        }
        const SolverPerformance p = pbicgstab(M, k.internal, m, patches, tol, relTol, maxIter);
        if (res) res->k = p.initialResidual;

        k.evaluateBoundary();
        bound(k, 1e-15, m, g, patches);   // Foam::bound(k_, kMin_)
    }

    // correctNut.
    //
    // OpenFOAM writes this as `nut_ = Cmu*sqr(k_)/epsilon_`, and that is a GeometricField assignment: it
    // sets the BOUNDARY field as well, from k and epsilon's own boundary values. A patch whose nut is
    // `calculated` therefore gets Cmu*k_b^2/eps_b every iteration -- it does NOT keep its initial value.
    // Only a patch carrying a nut wall function has that overwritten afterwards.
    //
    // brae used to set the internal field and the wall patches only, leaving everything else at whatever
    // was read from disk. On pitzDaily the walls carry nutkWallFunction so the gap never showed. On
    // simpleCar a trailing ".*" entry makes nut `calculated` on the walls, and brae held it at the
    // initial uniform 0 while OpenFOAM had 0.6 to 2.5 there -- a wall viscosity wrong by everything,
    // which is what a 50% different flow looks like.
    nutF = correctNut(k.internal, epsilon.internal, co);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (epsilon.boundary[pi]->isTurbulenceWallFunction())
        {
            nutField.boundary[pi]->setValue(
                [&]
                {
                    // Per-face nu here too. This is the nut BOUNDARY update after correctNut, and it was
                    // the one call site still reading the case-constant nu -- which the compressible
                    // lineage does not have. Passing the scalar there put a 0 into a divisor and the
                    // whole solve went non-finite inside the first iteration.
                    if (!(comp && comp->nuBnd))
                        return nutkWallFunction(patches[pi], yWall[pi], k.internal, nu,
                                                co.Cmu, co.kappa, co.E);
                    std::vector<scalar> nf(patches[pi].size);
                    for (label i = 0; i < patches[pi].size; ++i) nf[i] = (*comp->nuBnd)[pi][i];
                    return nutkWallFunction(patches[pi], yWall[pi], k.internal, nf,
                                            co.Cmu, co.kappa, co.E);
                }());
            continue;
        }

        const std::vector<scalar>& kb = k.boundary[pi]->value();
        const std::vector<scalar>& eb = epsilon.boundary[pi]->value();
        std::vector<scalar> nb(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            nb[i] = co.Cmu * kb[i] * kb[i] / eb[i];
        }
        nutField.boundary[pi]->setValue(nb);
    }

    // EddyDiffusivity::correctNut -- alphat = rho*nut/Prt (EddyDiffusivity.C:36-38), unconditional and
    // whole-field, after nut's boundary values are in place. The momentum equation never asks for alphat;
    // the ENERGY equation does, through alphaEff = CpByCpv*(alpha + alphat), so it is produced here where
    // nut has just been corrected rather than derived again by the caller from a nut that may have moved.
    //
    // OUTSIDE the patch loop, and that is the whole point. This block used to sit inside the
    // `isTurbulenceWallFunction()` branch above, ahead of its `continue`, so it ran once per wall-function
    // patch and NOT AT ALL on a case carrying no such patch -- leaving the energy equation to run on
    // whatever alphat was read from disk, with nothing to catch it. OpenFOAM has no patch loop here.
    if (comp && comp->alphat)
    {
        comp->alphat->resize(nC);
        for (label c = 0; c < nC; ++c)
            (*comp->alphat)[c] = rhoAt(c) * nutF[c] / comp->Prt;
    }
}

} // namespace kEpsilonRef
} // namespace cpu
} // namespace brae
