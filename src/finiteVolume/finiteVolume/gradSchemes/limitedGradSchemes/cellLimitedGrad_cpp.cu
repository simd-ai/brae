#include <stdexcept>
#include <string>
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
// OVER THE WHOLE MESH (`at` null) OR OVER A SUBSET'S FACES AND CELLS (fvc::GradSubset): the same loops handed
// the listed faces and cells, with the subset's kept arrays in place of fresh ones. A cell's limiter takes
// its own faces, the values across them and its own gradient, so at a listed cell the result is the whole
// mesh's (held by tests/test_grad_subset.cu: the compiler clones this pass for the two forms); an unlisted
// cell's entries are touched by the faces it shares with a listed one and mean nothing.
template <typename ValueAt, typename PatchValueAt, typename GradDotAt, typename ApplyLimiter>
void limitPass(
    label                       nC,
    int                         nCmpt,
    ValueAt                     valueAt,      // (cell, cmpt) -> scalar
    PatchValueAt patchValueAt,                // (patch, face, cmpt) -> scalar, an uncoupled patch's value
    GradDotAt                   gradDotAt,    // (cell, cmpt, d) -> (d & grad_cmpt)
    ApplyLimiter                applyLimiter, // (cell, cmpt, limiter)
    scalar                      k,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    const fvc::GradSubset* at)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const label nCells = at ? static_cast<label>(at->cells.size()) : nC;
    const label nFaces = at ? static_cast<label>(at->faces.size()) : nIf;
    std::vector<scalar> maxWhole;
    std::vector<scalar> minWhole;
    std::vector<scalar> limiterWhole;
    std::vector<scalar>& maxVsf = at ? at->maxVsf : maxWhole;
    std::vector<scalar>& minVsf = at ? at->minVsf : minWhole;
    std::vector<scalar>& limiter = at ? at->limiter : limiterWhole;
    maxVsf.resize(static_cast<std::size_t>(nC));
    minVsf.resize(static_cast<std::size_t>(nC));
    limiter.resize(static_cast<std::size_t>(nC));

    for (int cmpt = 0; cmpt < nCmpt; ++cmpt)
    {
        for (label j = 0; j < nCells; ++j)
        {
            const label c = at ? at->cells[static_cast<std::size_t>(j)] : j;
            maxVsf[c] = valueAt(c, cmpt);
            minVsf[c] = maxVsf[c];
        }
        for (label j = 0; j < nFaces; ++j)
        {
            const label f = at ? at->faces[static_cast<std::size_t>(j)] : j;
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
            // ...and a COUPLED patch its patchNeighbourField (cellLimitedGrad.C, the psf.coupled()
            // branch): the cell across a cyclic, the AMI's weighted sum across a cyclicAMI -- never a
            // stored patch value
            const FvPatch& fp = patches[pi];
            for (label i = 0; i < fp.size; ++i)
            {
                const label c = fp.faceCells[i];
                scalar vb = 0;
                if (!fp.coupled)
                {
                    vb = patchValueAt(pi, i, cmpt);
                }
                else if (fp.amiOffsets.empty())
                {
                    vb = valueAt(fp.nbrFaceCells[static_cast<std::size_t>(i)], cmpt);
                }
                else
                {
                    for (label sl = fp.amiOffsets[i]; sl < fp.amiOffsets[i + 1]; ++sl)
                    {
                        vb += fp.amiWeights[static_cast<std::size_t>(sl)]
                             *valueAt(fp.amiNbrCells[static_cast<std::size_t>(sl)], cmpt);
                    }
                }
                maxVsf[c] = std::fmax(maxVsf[c], vb);
                minVsf[c] = std::fmin(minVsf[c], vb);
            }
        }

        for (label j = 0; j < nCells; ++j)
        {
            const label c = at ? at->cells[static_cast<std::size_t>(j)] : j;
            maxVsf[c] -= valueAt(c, cmpt);
            minVsf[c] -= valueAt(c, cmpt);
        }
        if (k < 1.0)
        {
            for (label j = 0; j < nCells; ++j)
            {
                const label c = at ? at->cells[static_cast<std::size_t>(j)] : j;
                const scalar w = (1.0 / k - 1.0) * (maxVsf[c] - minVsf[c]);
                maxVsf[c] += w;
                minVsf[c] -= w;
            }
        }

        for (label j = 0; j < nCells; ++j)
        {
            limiter[at ? at->cells[static_cast<std::size_t>(j)] : j] = 1.0;
        }
        for (label j = 0; j < nFaces; ++j)
        {
            const label f = at ? at->faces[static_cast<std::size_t>(j)] : j;
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

        for (label j = 0; j < nCells; ++j)
        {
            const label c = at ? at->cells[static_cast<std::size_t>(j)] : j;
            applyLimiter(c, cmpt, limiter[c]);
        }
    }
}

// The scalar limiter, over the whole mesh or at a subset: ONE call of limitPass with one set of accessors, so
// both forms are the same instantiation of it. An uncoupled patch face's value is the caller's, or its face
// cell's where the caller's list is short (a patch handed no values).
void cellLimitScalar(
    std::vector<vector>& grad,
    const std::vector<scalar>& vsf,
    const std::vector<std::vector<scalar>>& vsfBnd,
    scalar k,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const fvc::GradSubset* at)
{
    limitPass(
        m.nCells(), 1,
        [&](
            label c,
            int)
        {
            return vsf[c];
        },
        [&](
            std::size_t pi,
            label i,
            int)
        {
            const std::vector<scalar>& b = vsfBnd[pi];
            return i < static_cast<label>(b.size()) ? b[i] : vsf[patches[pi].faceCells[i]];
        },
        [&](
            label c,
            int,
            const vector& d)
        {
            return dot(d, grad[c]);
        },
        [&](
            label c,
            int,
            scalar lim)
        {
            grad[c] = grad[c] * lim;
        },
        k, m, g, patches, at);
}

} // namespace

void cellLimitGrad(
    std::vector<vector>&                    grad,
    const std::vector<scalar>&              vsf,
    const std::vector<std::vector<scalar>>& vsfBnd,
    scalar                                  k,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches)
{
    if (k < SMALL_) return;   // OF: `if (k_ < SMALL) return tGrad;` -- the scheme is off
    // (the patch values are read where the pass wants one: this built a vector a boundary FACE for them at
    // every call -- 6,642 heap blocks a call on RAS/electrostaticDeposition)
    cellLimitScalar(grad, vsf, vsfBnd, k, m, g, patches, nullptr);
}

void cellLimitGradAt(
    const fvc::GradSubset& at,
    const std::vector<scalar>& vsf,
    const std::vector<std::vector<scalar>>& vsfBnd,
    scalar k,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (k < SMALL_) return;
    if (at.grad.size() != static_cast<std::size_t>(m.nCells()))
    {
        throw std::runtime_error("brae cellLimitGradAt: the subset holds no gradient of this mesh to limit.");
    }
    cellLimitScalar(at.grad, vsf, vsfBnd, k, m, g, patches, &at);
}

void cellLimitGrad(
    std::vector<vector>&          grad,
    const GeometricField<scalar>& vsf,
    scalar                        k,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    if (k < SMALL_) return;
    std::vector<std::vector<scalar>> bnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // a jump cyclic's patchNeighbourField is the cell LESS the jump, which the limiter's own
        // neighbour stencil does not subtract
        if (vsf.boundary[pi]->coupledJump())
        {
            throw std::runtime_error(
                "brae: a cellLimited gradient of a field with a jump across the coupled patch '"
                + patches[pi].name + "' is not ported.");
        }
        bnd[pi] = vsf.boundary[pi]->value();
    }
    cellLimitGrad(grad, vsf.internal, bnd, k, m, g, patches);
}

void cellLimitGrad(
    std::vector<tensor>&                    grad,
    const std::vector<vector>&              vsf,
    const std::vector<std::vector<vector>>& vsfBnd,
    scalar                                  k,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches)
{
    if (k < SMALL_) return;
    const label nC = m.nCells();

    std::vector<std::vector<std::vector<scalar>>> pv(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<vector>& b = vsfBnd[pi];
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
        [&](label c, int cmpt) { return (&vsf[c].x)[cmpt]; },
        [&](
            std::size_t pi,
            label i,
            int cmpt)
        {
            return pv[pi][i][cmpt];
        },
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
        k, m, g, patches, nullptr);
}

// ...and the field form, over the values the field currently holds.
void cellLimitGrad(
    std::vector<tensor>&          grad,
    const GeometricField<vector>& vsf,
    scalar                        k,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    if (k < SMALL_) return;
    std::vector<std::vector<vector>> bnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // a jump cyclic's patchNeighbourField is the cell LESS the jump, which the limiter's own
        // neighbour stencil does not subtract
        if (vsf.boundary[pi]->coupledJump())
        {
            throw std::runtime_error(
                "brae: a cellLimited gradient of a field with a jump across the coupled patch '"
                + patches[pi].name + "' is not ported.");
        }
        bnd[pi] = vsf.boundary[pi]->value();
    }
    cellLimitGrad(grad, vsf.internal, bnd, k, m, g, patches);
}

} // namespace cpu
} // namespace brae
