// A LEAST-SQUARES GRADIENT AND ITS CELL LIMITER AT A SUBSET OF THE CELLS, against the same functions over the
// whole mesh, cell for cell and bit for bit.
//
// fvc::leastSquaresGradAt and cpu::cellLimitGradAt are the whole-mesh functions' own bodies handed a list of
// cells and of the faces that touch them (fvc::GradSubset). The claim is that at a listed cell the result is
// the whole-mesh gradient's to the bit. It is one body in the source, but the compiler specialises it for
// the two call forms, so the claim is a measurement: this file makes it on the gradient ITSELF, at every
// listed cell, where the solver's own check sees it only through a patch face's normal -- which is zero
// wherever the patch has no normal gradient, whatever the cell's gradient is.
//
// What the fixtures are chosen for:
//   * a field that varies in every cell and patch values that are not the cells' (a normal gradient on every
//     patch), so no term is multiplied by a zero difference (the gradients that come out zero are the ones
//     the limiter took there: an extremum cell's);
//   * a SHEARED 3-D box of three different cell sizes: the mesh is non-orthogonal, so the boundary fit vector
//     is the projected one and the fit tensor has off-diagonal terms;
//   * the limiter at k = 1 and at k = 0.5 (the branch that widens the bounds), with a count of the listed
//     cells it changed -- a limiter that changed none would be compared as the identity;
//   * a 2-D box whose two z planes are `empty`: the fit tensor is singular in z there and safeInv is what
//     inverts it;
//   * each case twice on the SAME subset with another field: the subset keeps its arrays between calls and
//     clears the listed cells' entries only.
// The CONTROL leaves every other face out of the subset: listed cells must then differ.
// NOT HERE: a coupled patch (cyclic, cyclicAMI) -- the box has none; tests/test_grad_subset_coupled.cu holds
// the same on a pair.
#include "box_mesh.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

bool sameBits(
    const vector& a,
    const vector& b)
{
    return std::memcmp(&a, &b, sizeof(vector)) == 0;
}

struct Tally
{
    long compared = 0;
    long differing = 0;
    long limited = 0;
    long nonZero = 0;
};

// one mesh: the subset of the cells on a patch that is not empty, two fields, the fit alone and the fit
// limited at k = 1 and k = 0.5. `everyOtherFace` is the control.
Tally run(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    bool everyOtherFace)
{
    const label nC = m.nCells();
    std::vector<char> on(static_cast<std::size_t>(nC), 0);
    for (const FvPatch& q : fvp)
    {
        if (q.type == "empty") continue;
        for (label i = 0; i < q.size; ++i)
        {
            on[static_cast<std::size_t>(q.faceCells[i])] = 1;
        }
    }
    std::vector<label> cells;
    for (label c = 0; c < nC; ++c)
    {
        if (on[static_cast<std::size_t>(c)])
        {
            cells.push_back(c);
        }
    }
    fvc::GradSubset at = fvc::gradSubset(m, cells);
    if (everyOtherFace)
    {
        std::vector<label> kept;
        for (std::size_t k = 0; k < at.faces.size(); k += 2)
        {
            kept.push_back(at.faces[k]);
        }
        at.faces = kept;
    }
    Tally t;
    for (int field = 0; field < 2; ++field)
    {
        // a field with no flat spot, and patch values a fixed step off their cells'
        std::vector<scalar> a(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            const vector& x = g.C()[c];
            a[static_cast<std::size_t>(c)] = scalar(0.5)
                + scalar(0.4)*std::sin(scalar(1.3 + field)*x.x + scalar(0.7)*x.y + scalar(0.45)*x.z)
                + scalar(0.05)*std::cos(scalar(2.1)*x.y - scalar(0.3 + 0.2*field)*x.x);
        }
        std::vector<std::vector<scalar>> bnd(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            bnd[pi].resize(static_cast<std::size_t>(fvp[pi].size));
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                bnd[pi][static_cast<std::size_t>(i)] =
                    a[static_cast<std::size_t>(fvp[pi].faceCells[i])] + scalar(0.07)*scalar(1 + pi % 3);
            }
        }
        const std::vector<vector> whole = fvc::leastSquaresGrad(a, bnd, m, g, fvp);
        for (const scalar k : {scalar(0), scalar(1), scalar(0.5)})
        {
            std::vector<vector> wholeK = whole;
            fvc::leastSquaresGradAt(a, bnd, m, g, fvp, at);
            if (k > scalar(0))
            {
                cpu::cellLimitGrad(wholeK, a, bnd, k, m, g, fvp);
                cpu::cellLimitGradAt(at, a, bnd, k, m, g, fvp);
            }
            for (const label c : cells)
            {
                const std::size_t s = static_cast<std::size_t>(c);
                ++t.compared;
                t.differing += sameBits(at.grad[s], wholeK[s]) ? 0 : 1;
                t.limited += (k > scalar(0) && !sameBits(wholeK[s], whole[s])) ? 1 : 0;
                t.nonZero += (wholeK[s].x != scalar(0) || wholeK[s].y != scalar(0)) ? 1 : 0;
            }
        }
    }
    return t;
}

} // namespace

int main()
{
    std::printf("== a least-squares gradient and its cell limiter at a subset of the cells ==\n");
    // 3-D, sheared, three cell sizes
    {
        PrimitiveMesh m = boxtest::boxMesh(7, 6, 5, scalar(0.3), scalar(1), scalar(0.7), scalar(1.3));
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        const Tally t = run(m, g, fvp, false);
        std::printf("  sheared 3-D box: %ld gradients compared at the cells on a patch, %ld differ; %ld were "
                    "changed by the limiter, %ld are not zero\n", t.compared, t.differing, t.limited, t.nonZero);
        check("3-D, non-orthogonal: the subset's fit and its limiter (k = 1, k = 0.5) are the whole mesh's, bit "
              "for bit, with the limiter biting",
              t.compared > 0 && t.differing == 0 && t.limited > 0 && 2*t.nonZero > t.compared);
        const Tally wrong = run(m, g, fvp, true);
        std::printf("  CONTROL, every other face of the subset left out: %ld of %ld differ\n", wrong.differing,
                    wrong.compared);
        check("CONTROL: with faces left out of the subset the comparison fails", wrong.differing > 0);
    }
    // 2-D: the z planes empty
    {
        PrimitiveMesh m = boxtest::boxMesh(9, 7, 1, scalar(0.2), scalar(1), scalar(0.8), scalar(1));
        FvGeometry g;
        g.build(m);
        std::vector<FvPatch> fvp = buildPatches(m, g);
        long emptyFaces = 0;
        for (FvPatch& q : fvp)
        {
            if (q.name == "wallZmin" || q.name == "wallZmax")
            {
                q.type = "empty";
                emptyFaces += q.size;
            }
        }
        const Tally t = run(m, g, fvp, false);
        std::printf("  2-D box (%ld empty faces): %ld gradients compared, %ld differ; %ld changed by the limiter\n",
                    emptyFaces, t.compared, t.differing, t.limited);
        check("2-D, the z planes empty: the subset's fit and its limiter are the whole mesh's, bit for bit",
              emptyFaces > 0 && t.compared > 0 && t.differing == 0 && t.limited > 0);
    }
    std::printf("test_grad_subset: %d failures\n", failures);
    return failures ? 1 : 0;
}
