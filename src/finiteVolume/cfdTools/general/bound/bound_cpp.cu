#include "bound_cpp.cuh"
#include "bound_report.cuh"   // printBounding: one formatter for both arms
#include <cmath>

namespace brae {
namespace cpu {

scalar bound(
    GeometricField<scalar>&     vsf,
    scalar                      lowerBound,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    const char*                 fieldName)
{
    const label nC = m.nCells();
    if (nC == 0) return 0.0;

    // THE GUARD IS OpenFOAM'S min(vsf), WHICH INCLUDES THE BOUNDARY. min/max on a GeometricField are
    // UNARY_REDUCTION_FUNCTION_WITH_BOUNDARY (GeometricFieldFunctions.C:427): internal field AND every
    // patch field. This mined the cells only, so a field whose cells were all above the bound but whose
    // one patch face was below it did not trip the guard at all -- OpenFOAM bounds and prints there.
    // The `average:` in the message is the other way round: gAverage(vsf.primitiveField())
    // (FieldFunctions.C:647), internal cells only, arithmetic, not volume-weighted.
    scalar minVsf = vsf.internal[0];
    scalar maxVsf = vsf.internal[0];
    long double sumVsf = 0.0L;
    for (label c = 0; c < nC; ++c)
    {
        minVsf = std::fmin(minVsf, vsf.internal[c]);
        maxVsf = std::fmax(maxVsf, vsf.internal[c]);
        sumVsf += vsf.internal[c];
    }
    const scalar avgVsf = static_cast<scalar>(sumVsf / static_cast<long double>(nC));
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar> pv = vsf.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size && i < static_cast<label>(pv.size()); ++i)
        {
            minVsf = std::fmin(minVsf, pv[i]);
            maxVsf = std::fmax(maxVsf, pv[i]);
        }
    }
    if (minVsf >= lowerBound) return minVsf;   // bound.C:40, a STRICT less-than

    // The message, BEFORE the field is touched (bound.C:42 precedes bound.C:48) -- printing after the
    // clamp would report min == lowerBound every time. Shared with the device arm so the two cannot
    // drift on a line whose whole value is being comparable with OpenFOAM's own log.
    if (fieldName)
    {
        printBounding(fieldName, minVsf, maxVsf, avgVsf);
    }

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
