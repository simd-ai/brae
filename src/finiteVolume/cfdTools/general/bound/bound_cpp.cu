#include "bound_cpp.cuh"
#include <cmath>

namespace brae {
namespace cpu {

scalar bound(
    GeometricField<scalar>&     vsf,
    scalar                      lowerBound,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells();
    if (nC == 0) return 0.0;

    scalar minVsf = vsf.internal[0];
    for (label c = 0; c < nC; ++c)
    {
        minVsf = std::fmin(minVsf, vsf.internal[c]);
    }
    if (minVsf >= lowerBound) return minVsf;

    // average(max(vsf, lowerBound)): linear interpolation to the faces, then the area-weighted mean
    // over each cell's faces -- fvc::average is surfaceSum(magSf*ssf)/surfaceSum(magSf).
    std::vector<scalar> capped(nC);
    for (label c = 0; c < nC; ++c)
    {
        capped[c] = std::fmax(vsf.internal[c], lowerBound);
    }

    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<scalar> num(nC, 0.0), den(nC, 0.0);
    for (label f = 0; f < nIf; ++f)
    {
        const scalar w  = g.weights()[f];
        const scalar vf = w * capped[own[f]] + (1.0 - w) * capped[nei[f]];
        const scalar a  = g.magSf()[f];
        num[own[f]] += a * vf;
        den[own[f]] += a;
        num[nei[f]] += a * vf;
        den[nei[f]] += a;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& b = vsf.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label  c = patches[pi].faceCells[i];
            const scalar a = patches[pi].magSf[i];
            num[c] += a * std::fmax(b[i], lowerBound);
            den[c] += a;
        }
    }

    for (label c = 0; c < nC; ++c)
    {
        // pos0(-vsf) selects the average ONLY where the solve went non-positive.
        const scalar avg  = (den[c] > 0.0) ? num[c] / den[c] : lowerBound;
        const scalar cand = (vsf.internal[c] <= 0.0) ? avg : 0.0;
        vsf.internal[c] = std::fmax(std::fmax(vsf.internal[c], cand), lowerBound);
    }

    // vsf.boundaryFieldRef() = max(vsf.boundaryField(), lowerBound)
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        std::vector<scalar> b = vsf.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            b[i] = std::fmax(b[i], lowerBound);
        }
        vsf.boundary[pi]->setValue(b);
    }
    return minVsf;
}

} // namespace cpu
} // namespace brae
