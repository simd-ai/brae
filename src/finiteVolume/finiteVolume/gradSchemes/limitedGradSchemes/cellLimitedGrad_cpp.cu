#include "cellLimitedGrad_cpp.cuh"

#include <algorithm>
#include <cmath>

namespace brae {
namespace cpu {

namespace {

constexpr scalar SMALL_ = 1e-15;

// OF cellLimitedGrad::limitFaceCmpt. The `else` branch RETURNS -- a face whose extrapolation is
// negligible contributes no constraint at all, which is not the same as contributing r = 1.
void limitFaceCmpt(scalar& limiter, scalar maxDelta, scalar minDelta, scalar extrapolate)
{
    scalar r;
    if (extrapolate > SMALL_)
    {
        r = maxDelta / extrapolate;
    }
    else if (extrapolate < -SMALL_)
    {
        r = minDelta / extrapolate;
    }
    else
    {
        return;
    }
    limiter = std::fmin(limiter, std::fmin(r, 1.0));   // minmod: limiter(r) = min(r, 1)
}

// One pass of the whole scheme over `nCmpt` components. The caller supplies accessors so the scalar and
// vector forms share this body rather than restating it -- the two differ only in how many components
// the field has and in how the limiter multiplies the gradient.
template <typename ValueAt, typename GradDotAt, typename ApplyLimiter>
void limitPass(
    label                       nC,
    int                         nCmpt,
    ValueAt                     valueAt,      // (cell, cmpt) -> scalar
    GradDotAt                   gradDotAt,    // (cell, cmpt, d) -> (d & grad_cmpt)
    ApplyLimiter                applyLimiter, // (cell, cmpt, limiter)
    scalar                      k,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    const std::vector<std::vector<std::vector<scalar>>>& patchValues)   // [patch][face][cmpt]
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    for (int cmpt = 0; cmpt < nCmpt; ++cmpt)
    {
        std::vector<scalar> maxVsf(nC), minVsf(nC);
        for (label c = 0; c < nC; ++c)
        {
            maxVsf[c] = valueAt(c, cmpt);
            minVsf[c] = maxVsf[c];
        }
        for (label f = 0; f < nIf; ++f)
        {
            const scalar vo = valueAt(own[f], cmpt), vn = valueAt(nei[f], cmpt);
            maxVsf[own[f]] = std::fmax(maxVsf[own[f]], vn);
            minVsf[own[f]] = std::fmin(minVsf[own[f]], vn);
            maxVsf[nei[f]] = std::fmax(maxVsf[nei[f]], vo);
            minVsf[nei[f]] = std::fmin(minVsf[nei[f]], vo);
        }
        // A boundary face contributes its PATCH VALUE to the cell's range. Leaving it out lets the
        // gradient overshoot precisely where the field is being driven from outside.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // EMPTY patches contribute NOTHING, because in OpenFOAM they cannot: emptyFvPatch::size() is
            // 0, so cellLimitedGrad's boundary loops never see those faces. brae keeps the mesh's face
            // count on an empty patch, and including them here is not harmless -- see the face loop below.
            if (patches[pi].type == "empty") continue;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const scalar vb = patchValues[pi][i][cmpt];
                maxVsf[c] = std::fmax(maxVsf[c], vb);
                minVsf[c] = std::fmin(minVsf[c], vb);
            }
        }

        for (label c = 0; c < nC; ++c)
        {
            maxVsf[c] -= valueAt(c, cmpt);
            minVsf[c] -= valueAt(c, cmpt);
        }
        if (k < 1.0)
        {
            for (label c = 0; c < nC; ++c)
            {
                const scalar w = (1.0 / k - 1.0) * (maxVsf[c] - minVsf[c]);
                maxVsf[c] += w;
                minVsf[c] -= w;
            }
        }

        std::vector<scalar> limiter(nC, 1.0);
        for (label f = 0; f < nIf; ++f)
        {
            const vector& Cf = g.Cf()[f];
            const label o = own[f], n = nei[f];
            limitFaceCmpt(limiter[o], maxVsf[o], minVsf[o], gradDotAt(o, cmpt, Cf - g.C()[o]));
            limitFaceCmpt(limiter[n], maxVsf[n], minVsf[n], gradDotAt(n, cmpt, Cf - g.C()[n]));
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // EMPTY patches again, and here it MATTERS. On a 2D mesh Cf - C for an empty face points out
            // of the plane, so `extrapolate` is the out-of-plane gradient -- round-off, not physics. It
            // still clears the 1e-15 threshold once the gradient itself is of order 1e5, and then
            // r = maxDelta/extrapolate is a ratio of a real number to noise, which can clamp the limiter
            // far below what any real face asks for. OpenFOAM never evaluates these faces at all.
            if (patches[pi].type == "empty") continue;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const vector& Cf = g.Cf()[patches[pi].start + i];
                limitFaceCmpt(limiter[c], maxVsf[c], minVsf[c], gradDotAt(c, cmpt, Cf - g.C()[c]));
            }
        }

        for (label c = 0; c < nC; ++c) applyLimiter(c, cmpt, limiter[c]);
    }
}

} // namespace

void cellLimitGrad(
    std::vector<vector>&          grad,
    const GeometricField<scalar>& vsf,
    scalar                        k,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    if (k < SMALL_) return;   // OF: `if (k_ < SMALL) return tGrad;` -- the scheme is off
    const label nC = m.nCells();

    std::vector<std::vector<std::vector<scalar>>> pv(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& b = vsf.boundary[pi]->value();
        pv[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i) pv[pi][i] = {b[i]};
    }

    limitPass(
        nC, 1,
        [&](label c, int) { return vsf.internal[c]; },
        [&](label c, int, const vector& d) { return dot(d, grad[c]); },
        [&](label c, int, scalar lim) { grad[c] = grad[c] * lim; },
        k, m, g, patches, pv);
}

void cellLimitGrad(
    std::vector<tensor>&          grad,
    const GeometricField<vector>& vsf,
    scalar                        k,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    if (k < SMALL_) return;
    const label nC = m.nCells();

    std::vector<std::vector<std::vector<scalar>>> pv(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<vector>& b = vsf.boundary[pi]->value();
        pv[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i) pv[pi][i] = {b[i].x, b[i].y, b[i].z};
    }

    // OF's grad(U)_ij = d(U_j)/d(x_i), so component j of the field owns COLUMN j of the tensor: the
    // limiter for U_j scales grad[c][*][j], not a row.
    auto col = [](tensor& t, int j) -> scalar*
    {
        scalar* p = &t.xx;
        return p + j;   // rows are contiguous, so column j is p[j], p[3+j], p[6+j]
    };

    limitPass(
        nC, 3,
        [&](label c, int cmpt) { return (&vsf.internal[c].x)[cmpt]; },
        [&](label c, int cmpt, const vector& d)
        {
            const scalar* t = &grad[c].xx;
            return d.x * t[0 * 3 + cmpt] + d.y * t[1 * 3 + cmpt] + d.z * t[2 * 3 + cmpt];
        },
        [&](label c, int cmpt, scalar lim)
        {
            scalar* p = col(grad[c], cmpt);
            p[0] *= lim;
            p[3] *= lim;
            p[6] *= lim;
        },
        k, m, g, patches, pv);
}

} // namespace cpu
} // namespace brae
