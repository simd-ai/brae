// _cpp REFERENCE implementation -- see linearViscousStress_cpp.cuh for the OpenFOAM provenance.
#include "linearViscousStress_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "fvm.cuh"

namespace brae {
namespace cpu {

// Declared in the header; see the note there. The boundary array is a required argument rather than
// something this function is allowed to invent.
SurfaceScalarField effectiveFaceViscosity(
    const std::vector<scalar>& nuEff,
    const std::vector<std::vector<scalar>>& nuEffBnd,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    SurfaceScalarField gf = fvc::interpolate(nuEff, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (pi < nuEffBnd.size() && nuEffBnd[pi].size() == gf.boundary[pi].size())
        {
            gf.boundary[pi] = nuEffBnd[pi];
        }
    }
    return gf;
}


std::vector<vector> divDevReffExplicit(
    const GeometricField<vector>& U,
    const std::vector<scalar>&    nuEff,
    const std::vector<std::vector<scalar>>& nuEffBnd,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    scalar                        gradULimitK)
{
    // fvc::grad(U) -- cell tensors, then the boundary tensors with OpenFOAM's gaussGrad boundary
    // correction (wall-normal component replaced by snGrad(U)).
    std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    // The case's gradScheme, applied to the SAME gradient the dev2 term is built from. OpenFOAM resolves
    // fvc::grad(U) here against gradSchemes/grad(U); running the unlimited base scheme where the case
    // names `cellLimited Gauss linear 1` is a different discretisation under the case's own scheme name.
    if (gradULimitK > 0.0) cellLimitGrad(gradU, U, gradULimitK, m, g, patches);
    const std::vector<std::vector<tensor>> gradUb =
        fvc::gradUBoundary(U, gradU, m, g, patches);

    // nuEff*dev2(T(grad(U))) as a volTensorField (cells + boundary faces).
    std::vector<tensor> tCell(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        tCell[c] = nuEff[c] * dev2(transpose(gradU[c]));
    }

    std::vector<std::vector<tensor>> tBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::size_t n = gradUb[pi].size();
        tBnd[pi].resize(n);
        for (std::size_t f = 0; f < n; ++f)
        {
            const scalar nu =
                (pi < nuEffBnd.size() && f < nuEffBnd[pi].size()) ? nuEffBnd[pi][f] : 0.0;
            tBnd[pi][f] = nu * dev2(transpose(gradUb[pi][f]));
        }
    }

    // OpenFOAM: -fvc::div(...). fvc::div already divides by the cell volume.
    std::vector<vector> d = fvc::div(tCell, tBnd, m, g, patches);
    for (auto& v : d)
    {
        v = {-v.x, -v.y, -v.z};
    }
    return d;
}


void addDivDevReff(
    FvVectorMatrix&               UEqn,
    const GeometricField<vector>& U,
    const std::vector<scalar>&    nuEff,
    const std::vector<std::vector<scalar>>& nuEffBnd,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    bool                          correctedLaplacian,
    scalar                        snGradLimitCoeff,
    scalar                        gradULimitK)
{
    // Implicit half: OpenFOAM writes `- fvm::laplacian(nuEff, U)` inside divDevReff, and UEqn.H adds
    // divDevReff to the equation -- so the laplacian enters with coefficient -1.
    const SurfaceScalarField gammaf = effectiveFaceViscosity(nuEff, nuEffBnd, m, g, patches);
    addEqual(UEqn, fvm::laplacian<vector>(gammaf, U, m, g, patches, correctedLaplacian), -1.0);

    // ...and, when `corrected`, its explicit deferred correction.
    //
    // SIGN. OpenFOAM's laplacian does `fvm.source() -= V*div(faceFluxCorrection)`. divDevReff returns
    // MINUS that laplacian, so the contribution reaching UEqn is `+V*div(...)`. Writing the laplacian's
    // own sign here instead made the momentum WORSE with the correction on than off (U error against real
    // OpenFOAM on shearedChannel: 1.69e-01 corrected vs 8.47e-02 uncorrected) while the pressure -- which
    // does carry the laplacian's own sign -- improved. That asymmetry is what identified it.
    if (correctedLaplacian)
    {
        // THE SAME gradient as the dev2 term's: grad(U) through the case's gradSchemes entry. An earlier
        // version of this block kept it unlimited on the reading that correctedSnGrad<Type>::correction
        // goes per COMPONENT and looks up `grad(U.component(0))`, which falls to `default`. That is the
        // generic template; OpenFOAM instantiates the VECTOR specialisation, correctedSnGrads.C:48,
        //     correctedSnGrad<vector>::correction(vvf) { return fullGradCorrection(vvf); }
        // and fullGradCorrection resolves mesh.gradScheme("grad(U)") (correctedSnGrad.C:52-55) -- cellLimited
        // Gauss linear 1 on aerofoilNACA0012. Measured there at iteration 1 with the unlimited gradient:
        // the momentum source on the 120 aerofoil-adjacent cells 2.36e-05 against OpenFOAM's, with the
        // gradient, the dev2 term, the diagonal and the off-diagonals all exact (queue item 25).
        std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
        if (gradULimitK > 0.0) cellLimitGrad(gradU, U, gradULimitK, m, g, patches);
        const std::vector<vector> corr =
            fvm::laplacianNonOrthSource<vector, tensor>(gammaf, U, gradU, m, g, patches,
                                                        snGradLimitCoeff);
        for (std::size_t c = 0; c < corr.size(); ++c)
        {
            UEqn.source[c].x += corr[c].x;
            UEqn.source[c].y += corr[c].y;
            UEqn.source[c].z += corr[c].z;
        }
    }

    // Explicit half. OpenFOAM's equation reads  M(U) + divDevReff(U) == 0, i.e. the explicit term sits on
    // the LEFT. brae's FvMatrix solves  M.psi = source, so a left-hand explicit term crosses to the right
    // with a sign change: source -= (explicit contribution) becomes source += (-...)*V.
    //
    // Two conversions happen here and both are easy to get silently wrong, so both are stated:
    //   1. the side change (left -> right) flips the sign;
    //   2. fvc::div returns a per-VOLUME quantity, while FvMatrix::source is an EXTENSIVE per-cell
    //      quantity, so it must be multiplied back by V.
    // tests/test_divdevreff_cpp.cu pins the result against an OpenFOAM dump; do not "simplify" the signs
    // here without re-running it.
    const std::vector<vector> expl =
        divDevReffExplicit(U, nuEff, nuEffBnd, m, g, patches, gradULimitK);
    const std::vector<scalar>& V = g.V();
    for (std::size_t c = 0; c < expl.size(); ++c)
    {
        UEqn.source[c].x -= expl[c].x * V[c];
        UEqn.source[c].y -= expl[c].y * V[c];
        UEqn.source[c].z -= expl[c].z * V[c];
    }
}

} // namespace cpu
} // namespace brae
