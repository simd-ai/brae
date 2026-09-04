#include "fvc.cuh"

namespace brae {
namespace fvc {

std::vector<vector> gaussGrad(
    const std::vector<scalar>& internal,
    const std::vector<std::vector<scalar>>& boundary,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    std::vector<vector> grad(nC, vector{0, 0, 0});
    for (label f = 0; f < nIf; ++f)
    {
        const label o = own[f], n = nei[f];
        const scalar pf = w[f] * internal[o] + (1.0 - w[f]) * internal[n];
        const vector Sfssf = Sf[f] * pf;
        grad[o] += Sfssf;
        grad[n] = grad[n] - Sfssf;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        if (pi >= boundary.size()) continue;
        for (label i = 0; i < fp.size && i < (label)boundary[pi].size(); ++i)
            grad[fp.faceCells[i]] += Sf[fp.start + i] * boundary[pi][i];
    }
    for (label c = 0; c < nC; ++c)
    {
        const scalar iv = 1.0 / g.V()[c];
        grad[c] = grad[c] * iv;
    }
    return grad;
}

std::vector<vector> gaussGrad(
    const GeometricField<scalar>& p,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    std::vector<vector> grad(nC, vector{0, 0, 0});

    // Internal faces: linear interpolation to the face, +owner / -neighbour.
    for (label f = 0; f < nIf; ++f)
    {
        const label o = own[f], n = nei[f];
        const scalar pf = w[f] * p.internal[o] + (1.0 - w[f]) * p.internal[n];
        const vector Sfssf = Sf[f] * pf;
        grad[o] += Sfssf;
        grad[n] = grad[n] - Sfssf;
    }

    // Boundary faces: use the evaluated boundary face value.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<scalar>& pv = p.boundary[pi]->value();
        for (label i = 0; i < fp.size; ++i)
            grad[fp.faceCells[i]] += Sf[fp.start + i] * pv[i];
    }

    for (label c = 0; c < nC; ++c)
        grad[c] = grad[c] / g.V()[c];
    return grad;
}

std::vector<tensor> gaussGrad(
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    std::vector<tensor> grad(nC, tensor{0,0,0,0,0,0,0,0,0});
    for (label f = 0; f < nIf; ++f)
    {
        const vector Uf = w[f] * U.internal[own[f]] + (1.0 - w[f]) * U.internal[nei[f]];
        const tensor SfUf = outer(Sf[f], Uf);
        grad[own[f]] += SfUf;
        grad[nei[f]] = grad[nei[f]] - SfUf;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<vector>& uv = U.boundary[pi]->value();
        for (label i = 0; i < fp.size; ++i)
            grad[fp.faceCells[i]] += outer(Sf[fp.start + i], uv[i]);
    }
    for (label c = 0; c < nC; ++c)
        grad[c] = grad[c] / g.V()[c];
    return grad;
}

// Array form. HbyA in pEqn.H is not a GeometricField in the _cpp reference -- it is an internal field
// plus a boundary field that constrainHbyA has partly overwritten from U -- so the flux operator is
// expressed over plain arrays and the GeometricField overload delegates to it. One implementation, so the
// solver path and the field path cannot drift apart.
SurfaceScalarField flux(
    const std::vector<vector>& internal,
    const std::vector<std::vector<vector>>& boundary,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    SurfaceScalarField phi;
    phi.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f)
    {
        const vector Uf = w[f] * internal[own[f]] + (1.0 - w[f]) * internal[nei[f]];
        phi.internal[f] = dot(Uf, Sf[f]);
    }
    phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        phi.boundary[pi].resize(fp.size);
        for (label i = 0; i < fp.size; ++i)
            phi.boundary[pi][i] = dot(boundary[pi][i], Sf[fp.start + i]);
    }
    return phi;
}

SurfaceScalarField flux(
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    std::vector<std::vector<vector>> bnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) bnd[pi] = U.boundary[pi]->value();
    return flux(U.internal, bnd, m, g, patches);
}

SurfaceScalarField rhoFlux(
    const std::vector<scalar>& rho,
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const SurfaceScalarField phiV = flux(U, m, g, patches);
    const SurfaceScalarField rhoF = interpolate(rho, m, g, patches);

    SurfaceScalarField phi;
    phi.internal.resize(phiV.internal.size());
    for (std::size_t f = 0; f < phiV.internal.size(); ++f)
    {
        phi.internal[f] = rhoF.internal[f] * phiV.internal[f];
    }
    phi.boundary.resize(phiV.boundary.size());
    for (std::size_t pi = 0; pi < phiV.boundary.size(); ++pi)
    {
        phi.boundary[pi].resize(phiV.boundary[pi].size());
        for (std::size_t i = 0; i < phiV.boundary[pi].size(); ++i)
        {
            phi.boundary[pi][i] = rhoF.boundary[pi][i] * phiV.boundary[pi][i];
        }
    }
    return phi;
}

SurfaceScalarField interpolate(
    const std::vector<scalar>& vol,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();

    SurfaceScalarField sf;
    sf.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f)
        sf.internal[f] = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
    sf.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        sf.boundary[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            sf.boundary[pi][i] = vol[patches[pi].faceCells[i]];
    }
    return sf;
}

std::vector<scalar> div(
    const SurfaceScalarField& phi,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    std::vector<scalar> d(nC, 0.0);
    for (label f = 0; f < nIf; ++f)
    {
        d[own[f]] += phi.internal[f];
        d[nei[f]] -= phi.internal[f];
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            d[patches[pi].faceCells[i]] += phi.boundary[pi][i];
    for (label c = 0; c < nC; ++c)
        d[c] /= g.V()[c];
    return d;
}

std::vector<std::vector<tensor>> gradUBoundary(
    const GeometricField<vector>& U,
    const std::vector<tensor>& gradUcell,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const std::vector<vector>& Sf    = g.Sf();
    const std::vector<scalar>& magSf = g.magSf();
    std::vector<std::vector<tensor>> gb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        gb[pi].assign(fp.size, tensor{0,0,0,0,0,0,0,0,0});
        if (fp.type == "empty") continue;   // empty patches contribute nothing to fvc operations
        // THE PATCH'S OWN snGrad(), as OF's gaussGrad::correctBoundaryConditions asks for it
        // (gaussGrad.C: `gGradbf[patchi] += n*(vsf.boundaryField()[patchi].snGrad() - (n & gGradbf))`).
        // This used to inline (U_b - U_c)*deltaCoeffs, which is the BASE class's formula and wrong on
        // every class that overrides it -- zeroGradient (exactly zero), fixedGradient (the prescribed
        // gradient) and the mixed family, whose snGrad uses the CURRENT valueFraction while value() still
        // carries the blend of the previous one. See fv_patch_field.cuh's snGrad for the measurement.
        const std::vector<vector> sn = U.boundary[pi]->snGrad(U.internal);
        for (label i = 0; i < fp.size; ++i)
        {
            const label c  = fp.faceCells[i];
            const label gf = fp.start + i;
            const vector n = (1.0 / magSf[gf]) * Sf[gf];                       // unit normal
            const tensor& gc = gradUcell[c];                                   // extrapolated cell grad
            gb[pi][i] = gc + outer(n, sn[i] - dot(n, gc));                     // normal comp -> snGrad
        }
    }
    return gb;
}

std::vector<vector> div(
    const std::vector<tensor>& tCell,
    const std::vector<std::vector<tensor>>& tBnd,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    std::vector<vector> d(nC, vector{0.0, 0.0, 0.0});
    for (label f = 0; f < nIf; ++f)
    {
        const tensor Tf = w[f] * tCell[own[f]] + (1.0 - w[f]) * tCell[nei[f]];  // linear interpolate
        const vector SfTf = dot(Sf[f], Tf);                                     // Sf & T_f
        d[own[f]] += SfTf;
        d[nei[f]] = d[nei[f]] - SfTf;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "empty") continue;   // empty patches excluded (OF fvc semantics)
        for (label i = 0; i < patches[pi].size; ++i)
            d[patches[pi].faceCells[i]] += dot(Sf[patches[pi].start + i], tBnd[pi][i]);
    }
    for (label c = 0; c < nC; ++c)
        d[c] = d[c] / g.V()[c];
    return d;
}

SurfaceScalarField snGrad(
    const GeometricField<scalar>& vf,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    bool                          corrected)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& dc = corrected ? g.nonOrthDeltaCoeffs() : g.deltaCoeffs();

    SurfaceScalarField sf;
    sf.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f)
        sf.internal[f] = dc[f] * (vf.internal[nei[f]] - vf.internal[own[f]]);

    // correctedSnGrad::fullGradCorrection -- linear interpolation of grad(vf) dotted with the correction
    // vectors, which are zero on boundary faces.
    if (corrected)
    {
        const std::vector<vector>  gradVf   = gaussGrad(vf, m, g, patches);
        const std::vector<vector>& corrVecs = g.nonOrthCorrectionVectors();
        const std::vector<scalar>& w        = g.weights();
        for (label f = 0; f < nIf; ++f)
        {
            const vector& go = gradVf[own[f]];
            const vector& gn = gradVf[nei[f]];
            const vector  gf { w[f] * go.x + (1.0 - w[f]) * gn.x,
                               w[f] * go.y + (1.0 - w[f]) * gn.y,
                               w[f] * go.z + (1.0 - w[f]) * gn.z };
            sf.internal[f] += corrVecs[f].x * gf.x + corrVecs[f].y * gf.y + corrVecs[f].z * gf.z;
        }
    }

    sf.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<scalar> gIC = vf.boundary[pi]->gradientInternalCoeffs();
        const std::vector<scalar> gBC = vf.boundary[pi]->gradientBoundaryCoeffs();
        sf.boundary[pi].resize(fp.size);
        for (label i = 0; i < fp.size; ++i)
            sf.boundary[pi][i] = gIC[i] * vf.internal[fp.faceCells[i]] + gBC[i];
    }
    return sf;
}

} // namespace fvc
} // namespace brae
