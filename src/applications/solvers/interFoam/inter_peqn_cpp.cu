// interFoam's pressure corrector -- see inter_peqn_cpp.cuh for the provenance and for the four things
// in it that are interFoam's own.
#include "inter_peqn_cpp.cuh"
#include "fvc_reconstruct_cpp.cuh"

namespace brae {
namespace cpu {
namespace interFoam {

void buoyancyFlux(const std::vector<scalar>& surfaceTensionForce,
                  const std::vector<scalar>& ghf,
                  const std::vector<scalar>& snGradRho,
                  const std::vector<scalar>& rAUf,
                  const std::vector<scalar>& magSf,
                  std::vector<scalar>&       phig)
{
    const std::size_t n = surfaceTensionForce.size();
    if (ghf.size() != n || snGradRho.size() != n || rAUf.size() != n || magSf.size() < n)
        throw std::runtime_error("brae interFoam pEqn: phig's face fields differ in length.");
    phig.resize(n);
    for (std::size_t f = 0; f < n; ++f)
        phig[f] = (surfaceTensionForce[f] - ghf[f]*snGradRho[f]) * rAUf[f] * magSf[f];
}


void rhoRAUf(const std::vector<scalar>& rho,
             const std::vector<scalar>& rAU,
             const PrimitiveMesh&       m,
             const FvGeometry&          g,
             std::vector<scalar>&       out)
{
    if (rho.size() != rAU.size())
        throw std::runtime_error("brae interFoam pEqn: rho and rAU differ in length.");
    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<scalar>& w   = g.weights();

    // THE PRODUCT IS FORMED PER CELL AND INTERPOLATED ONCE. Not interpolate(rho)*interpolate(rAU):
    // linear interpolation does not commute with multiplication, and the gap is largest where the two
    // factors vary most -- which across a VoF interface is a factor of 1000 in one face.
    out.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const scalar po = rho[own[f]] * rAU[own[f]];
        const scalar pn = rho[nei[f]] * rAU[nei[f]];
        out[f] = w[f]*po + (scalar(1) - w[f])*pn;
    }
}


void correctVelocity(const std::vector<vector>&              HbyA,
                     const std::vector<scalar>&              rAU,
                     const std::vector<scalar>&              faceFlux,
                     const std::vector<scalar>&              rAUf,
                     const std::vector<std::vector<scalar>>& faceFluxBnd,
                     const std::vector<std::vector<scalar>>& rAUfBnd,
                     const PrimitiveMesh&                    m,
                     const FvGeometry&                       g,
                     const std::vector<FvPatch>&             patches,
                     std::vector<vector>&                    U)
{
    const label nIf = m.nInternalFaces();
    if (faceFlux.size() != static_cast<std::size_t>(nIf) || rAUf.size() != static_cast<std::size_t>(nIf))
        throw std::runtime_error("brae interFoam pEqn: the correction's face fields differ in length.");

    // (phig - p_rghEqn.flux())/rAUf, face by face, BEFORE the reconstruction. The division and the
    // later multiplication by rAU do not cancel on a non-uniform rAU -- see note 3.
    SurfaceScalarField ssf;
    ssf.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f) ssf.internal[f] = faceFlux[f] / rAUf[f];
    ssf.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::size_t np = static_cast<std::size_t>(patches[pi].size);
        ssf.boundary[pi].resize(np);
        for (std::size_t i = 0; i < np; ++i)
        {
            const scalar ff = (pi < faceFluxBnd.size() && i < faceFluxBnd[pi].size())
                            ? faceFluxBnd[pi][i] : scalar(0);
            const scalar rf = (pi < rAUfBnd.size() && i < rAUfBnd[pi].size())
                            ? rAUfBnd[pi][i] : scalar(1);
            ssf.boundary[pi][i] = ff / rf;
        }
    }

    using namespace cpu::fvcReconstruct;
    const label nC = m.nCells();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<vector>& Sf  = g.Sf();

    std::vector<tensor> T(static_cast<std::size_t>(nC), tensor{0,0,0,0,0,0,0,0,0});
    std::vector<vector> v(static_cast<std::size_t>(nC), vector{0,0,0});
    for (label f = 0; f < nIf; ++f)
    {
        accumulate(Sf[f], ssf.internal[f], T[own[f]], v[own[f]]);
        accumulate(Sf[f], ssf.internal[f], T[nei[f]], v[nei[f]]);
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        for (label i = 0; i < q.size; ++i)
        {
            const label ci = own[q.start + i];
            accumulate(Sf[q.start + i], ssf.boundary[pi][i], T[ci], v[ci]);
        }
    }

    U.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector r = dot(inv(T[c]), v[c]);
        U[c] = vector{HbyA[c].x + rAU[c]*r.x,
                      HbyA[c].y + rAU[c]*r.y,
                      HbyA[c].z + rAU[c]*r.z};
    }
}


void staticPressure(const std::vector<scalar>& p_rgh,
                    const std::vector<scalar>& rho,
                    const std::vector<scalar>& gh,
                    std::vector<scalar>&       p)
{
    if (rho.size() != p_rgh.size() || gh.size() != p_rgh.size())
        throw std::runtime_error("brae interFoam pEqn: p_rgh, rho and gh differ in length.");
    p.resize(p_rgh.size());
    for (std::size_t c = 0; c < p_rgh.size(); ++c) p[c] = p_rgh[c] + rho[c]*gh[c];
}


void applyPressureReference(std::vector<scalar>&       p,
                            std::vector<scalar>&       p_rgh,
                            const std::vector<scalar>& rho,
                            const std::vector<scalar>& gh,
                            label                      pRefCell,
                            scalar                     pRefValue)
{
    if (pRefCell < 0 || static_cast<std::size_t>(pRefCell) >= p.size())
        throw std::runtime_error(
            "brae interFoam pEqn: pRefCell is outside the mesh. A p_rgh with no value-fixing patch "
            "needs a reference cell, and pEqn.H:74-83 reads p there.");
    const scalar shift = pRefValue - p[static_cast<std::size_t>(pRefCell)];
    for (scalar& v : p) v += shift;
    // ...AND p_rgh IS REBUILT FROM THE SHIFTED p. It does not keep what the solve gave it -- note 4.
    for (std::size_t c = 0; c < p.size(); ++c) p_rgh[c] = p[c] - rho[c]*gh[c];
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
