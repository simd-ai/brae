#pragma once
// A gradSchemes entry as the host operators take it: the base scheme (Gauss linear or leastSquares) and
// an optional cellLimited coefficient, resolved ONCE from the case (scheme_parse.cuh) and handed to the
// operator that takes the gradient -- OpenFOAM resolves the entry by the name the call site asks for
// (fvc::grad(vf) reads `grad(<vf.name()>)`, interfaceProperties reads `nHat`), then `default`.
//
// provenance:
//   openfoam: src/finiteVolume/finiteVolume/gradSchemes/gaussGrad, leastSquaresGrad,
//             limitedGradSchemes/cellLimitedGrad
//   brae:     fvc.cu (gaussGrad, leastSquaresGrad), cellLimitedGrad_cpp.cu
#include "cellLimitedGrad_cpp.cuh"
#include "fvc.cuh"
#include "geometric_field.cuh"
#include <vector>

namespace brae {

struct GradChoice
{
    bool   leastSquares = false;
    // cellLimited's k; 0 is unlimited
    scalar cellLimitK = 0;

    bool gaussLinear() const
    {
        return !leastSquares && cellLimitK <= scalar(0);
    }
};

// fvc::grad(vf) under `choice`
inline std::vector<vector> gradOf(
    const GeometricField<scalar>& vf,
    const GradChoice& choice,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    std::vector<vector> gr = choice.leastSquares ? fvc::leastSquaresGrad(vf, m, g, patches)
                                                 : fvc::gaussGrad(vf, m, g, patches);
    if (choice.cellLimitK > scalar(0))
    {
        cpu::cellLimitGrad(gr, vf, choice.cellLimitK, m, g, patches);
    }
    return gr;
}

} // namespace brae
