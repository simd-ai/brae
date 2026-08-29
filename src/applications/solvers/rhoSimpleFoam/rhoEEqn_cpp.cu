// _cpp REFERENCE implementation -- see EEqn_cpp.cuh for the OpenFOAM provenance and the refusal contract.
#include "rhoEEqn_cpp.cuh"
#include "fvm.cuh"
#include "fvc.cuh"   // gaussGrad, for the laplacian non-orthogonal correction
#include "fv_matrix_ops.cuh"
#include "linearViscousStress_cpp.cuh"   // effectiveFaceViscosity -- the SAME face rule for alphaEff
#include "limitedSchemes_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include <stdexcept>

namespace brae {
namespace cpu {
namespace rhoSimple {

namespace {

void refuseUnsupported(const EnergyInput& in)
{
    if (in.heName != "e" && in.heName != "h")
        throw std::runtime_error(
            "rhoSimpleFoam EEqn_cpp: energy variable '" + in.heName + "' is neither 'e' nor 'h'. EEqn.H "
            "branches on he.name() and has a kinetic-energy source written for exactly those two "
            "(rhoSimpleFoam/EEqn.H:6-10); OpenFOAM's thermo.validate(.., \"h\", \"e\") refuses the same "
            "set. Refusing rather than solving a different energy equation.");
    if (!in.phi || !in.phiBnd)
        throw std::runtime_error("rhoSimpleFoam EEqn_cpp: the mass flux phi was not supplied.");
    if (!in.alphaEff || !in.alphaEffBnd)
        throw std::runtime_error(
            "rhoSimpleFoam EEqn_cpp: fvm::laplacian(turbulence->alphaEff(), he) needs alphaEff "
            "(EEqn.H:12). None was supplied.");
    if (in.hasMRF)
        throw std::runtime_error(
            "rhoSimpleFoam EEqn_cpp: the case declares MRF, which EEqn.H adds as "
            "`EEqn += fvc::div(MRF.phi(), p)` (EEqn.H:17-20). Not implemented; refusing rather than "
            "silently solving a different equation.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "rhoSimpleFoam EEqn_cpp: the case declares fvOptions, which EEqn.H applies as "
            "fvOptions(rho, he), fvOptions.constrain(EEqn) and fvOptions.correct(he) (EEqn.H:14,24,28). "
            "Not implemented; refusing rather than silently solving a different equation.");
    const bool luKE = (in.schemeKE == DivScheme::linearUpwind);
    const bool luHe = (in.schemeHe == DivScheme::linearUpwind);
    const bool okKE = (in.schemeKE == DivScheme::upwind) || luKE;
    const bool okHe = (in.schemeHe == DivScheme::upwind) || luHe;
    if (in.snGradLimitCoeff != 0.0)
        throw std::runtime_error(
            "rhoSimpleFoam EEqn_cpp: a `limited <k> corrected` laplacian was asked for. brae's laplacian "
            "here implements `corrected` (uncapped), which is what all five rhoSimpleFoam tutorials set; "
            "capping the non-orthogonal correction is a different discretisation. Refusing rather than "
            "running the uncapped form under the limited name.");
    if (!okKE || !okHe)
        throw std::runtime_error(
            "rhoSimpleFoam EEqn_cpp: only `Gauss upwind` and `Gauss linearUpwind <grad>` are ported for "
            "the energy convection terms -- those are what every rhoSimpleFoam tutorial that names "
            "div(phi,e|h) and div(phi,Ekp|K) asks for. A different scheme would be a different "
            "discretisation; refusing rather than substituting one brae does have.");
}

// gaussConvectionScheme::fvcDiv, plus boundedConvectionScheme::fvcDiv when `bounded`.
//
// Returned EXTENSIVE (V*div), because that is how it is consumed: adding a GeometricField to an fvMatrix
// means `source -= V*field` (fvMatrix.C), so carrying the V through and cancelling it is one fewer
// divide-and-remultiply, and one fewer place for a per-cell volume to go missing.
//
//     internal faces:  sum_f phi_f * vf_f      with the scheme's face value, owner +, neighbour -
//     boundary faces:  phi_b * vf_b            the patch value, as OpenFOAM's surfaceIntegrate does
//     bounded:        -(sum_f phi_f) * vf[c]   boundedConvectionScheme.C, the fvcDiv form
std::vector<scalar> explicitConvectionDivExtensive(
    const std::vector<scalar>&              phi,
    const std::vector<std::vector<scalar>>& phiBnd,
    const std::vector<scalar>&              vf,
    const std::vector<std::vector<scalar>>& vfBnd,
    DivScheme                               scheme,
    scalar                                  gradLimitK,
    bool                                    bounded,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    std::vector<scalar> d(nC, 0.0);

    // upwind face value: pos0(phi) -- OpenFOAM's `>= 0` convention, owner on a zero flux.
    for (label f = 0; f < nIf; ++f)
    {
        const scalar pf = phi[f];
        const scalar vfF = (pf >= 0.0) ? vf[own[f]] : vf[nei[f]];
        const scalar flux = pf * vfF;
        d[own[f]] += flux;
        d[nei[f]] -= flux;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
            d[patches[pi].faceCells[i]] += phiBnd[pi][i] * vfBnd[pi][i];
    }

    // linearUpwind's correction to the FACE VALUE. In the explicit form it is part of the value rather
    // than a deferred source, but the per-cell contribution is the same accumulation, so the same
    // validated helper computes it.
    if (scheme == DivScheme::linearUpwind)
    {
        GeometricField<scalar> shim;
        shim.internal = vf;
        std::vector<vector> gradVf = fvc::gaussGrad(vf, vfBnd, m, g, patches);
        cellLimitGrad(gradVf, shim, gradLimitK, m, g, patches);
        const std::vector<scalar> corr =
            fvm::linearUpwindCorrection<scalar, vector>(phi, gradVf, m, g);
        for (label c = 0; c < nC; ++c) d[c] += corr[c];
    }

    if (bounded)
    {
        // - surfaceIntegrate(phi)*vf, extensive: -(sum_f phi_f)*vf[c].
        std::vector<scalar> divPhi(nC, 0.0);
        for (label f = 0; f < nIf; ++f)
        {
            divPhi[own[f]] += phi[f];
            divPhi[nei[f]] -= phi[f];
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
                divPhi[patches[pi].faceCells[i]] += phiBnd[pi][i];
        for (label c = 0; c < nC; ++c) d[c] -= divPhi[c] * vf[c];
    }
    return d;
}

} // namespace


std::vector<scalar> kineticEnergy(
    const std::string&            heName,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho)
{
    const std::size_t nC = U.internal.size();
    std::vector<scalar> ke(nC);
    const bool isE = (heName == "e");
    for (std::size_t c = 0; c < nC; ++c)
    {
        const vector& u = U.internal[c];
        const scalar half = 0.5 * (u.x*u.x + u.y*u.y + u.z*u.z);
        // e carries the flow work p/rho explicitly; h already contains it.
        ke[c] = isE ? half + p.internal[c] / rho.internal[c] : half;
    }
    return ke;
}


std::vector<std::vector<scalar>> kineticEnergyBoundary(
    const std::string&            heName,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho,
    const std::vector<FvPatch>&   patches)
{
    const bool isE = (heName == "e");
    std::vector<std::vector<scalar>> ke(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<vector>& ub = U.boundary[pi]->value();
        const std::vector<scalar>& pb = p.boundary[pi]->value();
        const std::vector<scalar>& rb = rho.boundary[pi]->value();
        ke[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const scalar half = 0.5 * (ub[i].x*ub[i].x + ub[i].y*ub[i].y + ub[i].z*ub[i].z);
            ke[pi][i] = isE ? half + pb[i] / rb[i] : half;
        }
    }
    return ke;
}


std::vector<scalar> kineticEnergyDivergence(
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho,
    const EnergyInput&            in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    const std::vector<scalar>              ke    = kineticEnergy(in.heName, U, p, rho);
    const std::vector<std::vector<scalar>> keBnd = kineticEnergyBoundary(in.heName, U, p, rho, patches);
    return explicitConvectionDivExtensive(
        *in.phi, *in.phiBnd, ke, keBnd, in.schemeKE, in.gradKELimitK, in.boundedKE, m, g, patches);
}


FvScalarMatrix assembleEEqn(
    const GeometricField<scalar>& he,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho,
    const EnergyInput&            in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);
    const label nC = m.nCells();

    // fvm::div(phi, he) -- implicit convection of the energy variable by the MASS flux.
    FvScalarMatrix M = fvm::div(*in.phi, *in.phiBnd, he, m, patches);

    // linearUpwind's deferred correction on the IMPLICIT term, subtracted into the source exactly as in
    // the momentum equation (`fvm += ...` means `source -= ...`).
    if (in.schemeHe == DivScheme::linearUpwind)
    {
        std::vector<vector> gradHe = fvc::gaussGrad(he, m, g, patches);
        cellLimitGrad(gradHe, he, in.gradHeLimitK, m, g, patches);
        const std::vector<scalar> corr =
            fvm::linearUpwindCorrection<scalar, vector>(*in.phi, gradHe, m, g);
        for (label c = 0; c < nC; ++c) M.source[c] -= corr[c];
    }

    // `bounded` on div(phi,he): the fvmDiv form, -fvm::Sp(surfaceIntegrate(phi), he) -> diag -= div(phi)*V.
    if (in.boundedHe)
    {
        SurfaceScalarField phis;
        phis.internal = *in.phi;
        phis.boundary = *in.phiBnd;
        const std::vector<scalar> divPhi = fvc::div(phis, m, g, patches);
        for (label c = 0; c < nC; ++c) M.diag[c] -= divPhi[c] * g.V()[c];
    }

    // THE BRANCH: + fvc::div(phi, Ekp) for e, + fvc::div(phi, K) for h. An explicit field added to an
    // fvMatrix means `source -= V*field`, and kineticEnergyDivergence already carries the V.
    const std::vector<scalar> keDiv = kineticEnergyDivergence(U, p, rho, in, m, g, patches);
    for (label c = 0; c < nC; ++c) M.source[c] -= keDiv[c];

    // - fvm::laplacian(alphaEff, he). The face value of alphaEff follows the SAME rule as the momentum
    // equation's face viscosity -- boundary faces take the BOUNDARY field, not the owner cell -- which is
    // why the shared helper is reused rather than a second interpolation written here.
    // `corrected` is NOT optional: every rhoSimpleFoam tutorial sets `laplacianSchemes default Gauss
    // linear corrected`, and the correction is an explicit source term. Assembling the orthogonal
    // laplacian instead leaves the source short by the whole non-orthogonal contribution -- which on a
    // near-orthogonal mesh moves the DIAGONAL by only ~1e-06 while moving the SOURCE by ~19%, so the
    // diagonal comparison alone would not have caught it.
    //
    // `corrected` HAS TWO HALVES AND THIS EQUATION ONLY HAD ONE. Passing correctedLaplacian to
    // fvm::laplacian selects nonOrthDeltaCoeffs for the implicit coefficients; the EXPLICIT half --
    // source -= V*div(gamma*magSf*(corrVecs & interpolate(grad(vf)))) -- is a separate term
    // (gaussLaplacianScheme.C), and every other equation in the tree adds it: simpleFoam's pEqn,
    // kEpsilon, kOmegaSST, kOmegaSSTLM and linearViscousStress all call laplacianNonOrthSource. The
    // energy equation did not, so on any mesh with real non-orthogonality its source was short by the
    // whole correction while its DIAGONAL stayed exact -- which is precisely why it survived: every
    // gate that compares D() passed, and squareBend and sbMatched are near-orthogonal enough that the
    // source barely noticed.
    //
    // angledDuct is the case that shows it. At cell 629 on `walls` brae's assembled source was
    // 4.7983400746e-04 -- equal to its own -div(phi,Ekp)*V to seven digits, with every boundary
    // coefficient there exactly zero -- against OpenFOAM's 2.0967040959e-03. The whole 4.37x is the
    // missing correction. The gradient is the UNLIMITED Gauss linear one, because correctedSnGrad's
    // fullGradCorrection resolves grad(he) through the case's gradSchemes and angledDuct's default is
    // `Gauss linear`.
    {
        const SurfaceScalarField gammaf =
            effectiveFaceViscosity(*in.alphaEff, *in.alphaEffBnd, m, g, patches);
        FvScalarMatrix L = fvm::laplacian<scalar>(gammaf, he, m, g, patches, in.correctedLaplacian);
        if (in.correctedLaplacian)
        {
            std::vector<std::vector<scalar>> vb(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) vb[pi] = he.boundary[pi]->value();
            const std::vector<vector> gradHe = fvc::gaussGrad(he.internal, vb, m, g, patches);
            const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                gammaf, he, gradHe, m, g, patches, in.snGradLimitCoeff);
            for (label c = 0; c < nC; ++c) L.source[c] -= corr[c];
        }
        addEqual(M, L, -1.0);
    }

    // EEqn.relax().
    // THE GUARD IS "THE CASE NAMES A FACTOR", NOT "THE FACTOR IS BELOW 1".
    //
    // OpenFOAM's fvMatrix::relax() looks the name up -- `if (mesh.relaxEquation(name, coeff)) relax(coeff)`
    // (fvMatrix.C:1250-1263), and relaxEquation is `eqnRelaxDict_.found(name) || found("default")`
    // (solution.C:330-334). relax(alpha) itself returns early ONLY on alpha <= 0 (fvMatrix.C:1102-1107),
    // so a factor of exactly 1 still runs the whole body: the diagonal-dominance clamp
    // D = max(|D|, sumOff)/alpha and the matching source term S += (D - D0)*psi.
    //
    // The previous guard `relaxU > 0 && relaxU < 1` therefore skipped a case that NAMES 1 -- and
    // validation/sbMatched names `p 1` while validation/kEpsCorrect names `k 1` and `epsilon 1`, so this
    // is live rather than theoretical. rhoPEqn_cpp and rhoPcEqn_cpp already carry the correct form as
    // relaxPSpecified; U and he simply never got it, and the device path (rhoUEqn.cu's relaxEquationU)
    // has had it all along -- so this brings the reference up to the twin it is the oracle for.
    if (in.relaxEquationHe && in.relaxHe > 0.0)
    {
        relaxMatrix<scalar>(M, he, m, patches, in.relaxHe);
    }
    return M;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
