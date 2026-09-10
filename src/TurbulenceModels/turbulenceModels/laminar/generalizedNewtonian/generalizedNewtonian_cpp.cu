// _cpp REFERENCE implementation -- see generalizedNewtonian_cpp.cuh for the OpenFOAM provenance.
#include "generalizedNewtonian_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "fvc.cuh"

namespace brae {
namespace cpu {
namespace generalizedNewtonian {

void correctNu(
    const GeometricField<vector>&           U,
    const std::vector<scalar>&              nu0,
    const std::vector<std::vector<scalar>>& nu0Bnd,
    const PowerLawCoeffs&                   coeffs,
    scalar                                  gradULimitK,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches,
    std::vector<scalar>&                    nu,
    std::vector<std::vector<scalar>>&       nuBnd)
{
    // fvc::grad(this->U()) through gradSchemes/grad(U) -- the same gradient divDevRhoReff's dev2 term
    // takes (linearViscousStress_cpp.cu), limited when the case limits it.
    std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    if (gradULimitK > 0.0)
    {
        cellLimitGrad(gradU, U, gradULimitK, m, g, patches);
    }
    const std::vector<std::vector<tensor>> gradUb = fvc::gradUBoundary(U, gradU, m, g, patches);

    nu.resize(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        nu[c] = powerLawNu(nu0[c], strainRate(gradU[c]), coeffs.n, coeffs.nuMin, coeffs.nuMax);
    }
    nuBnd.assign(patches.size(), {});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        nuBnd[pi].resize(gradUb[pi].size());
        for (std::size_t i = 0; i < gradUb[pi].size(); ++i)
        {
            nuBnd[pi][i] = powerLawNu(nu0Bnd[pi][i], strainRate(gradUb[pi][i]), coeffs.n,
                                      coeffs.nuMin, coeffs.nuMax);
        }
    }
}

} // namespace generalizedNewtonian
} // namespace cpu
} // namespace brae
