// A LEAST-SQUARES GRADIENT AND ITS CELL LIMITER AT A SUBSET OF THE CELLS, ON A MESH WITH A COUPLED PAIR,
// against the same functions over the whole mesh, cell for cell and bit for bit.
//
// tests/test_grad_subset.cu holds fvc::leastSquaresGradAt and cpu::cellLimitGradAt to the whole-mesh forms on
// meshes without a pair. Their coupled branches -- the fit's own (the pair's delta in the fit tensor, the
// value ACROSS the pair in the fit) and the limiter's (the value across the pair in a cell's range, through a
// cyclic's neighbour cell or a cyclicAMI's weighted sum) -- are in the same bodies, and nothing ran them with a
// subset. The solver does not reach them either: brae_interFoam refuses a least-squares `nHat` on a mesh with
// a coupled patch by name (inter_case_cpp.cu, the arm baffle_leastSquaresNHat of tests/interfoam_refusals.sh),
// so this file is the only thing that holds the subset form across a pair, and what it holds is
// subset == whole. That the WHOLE form is OpenFOAM's across a translational cyclic is ctest
// interfoam_cyclic_vs_openfoam's (sstLsq for the fit, sstLim for the limiter's range).
//
// THE FIXTURE: test_grad_subset's sheared 3-D box, its x-min and x-max sides made one translational cyclic pair
// (coupleTranslationalPair, the solver's own routine), and then the same pair with a cyclicAMI's addressing --
// two neighbour cells a face, weights 0.6 and 0.4 -- so that the weighted sums run. The subset is the cells of
// every patch, the pair's included: the cells across the pair are cells of the pair, which is what
// interfaceProps::nHatBoundaryStencil relies on and what the first check asserts of it.
// THE CONTROLS hand the subset form the pair as two UNCOUPLED patches while the whole form keeps it coupled:
// in the fit, and in the limiter alone (the fit left coupled), on the cyclic and on the cyclicAMI. Cells of
// the pair must differ in each. MEASURED 2026-10-06: 900 gradients compared in each arm, 360 at cells of the
// pair, none differing; the fit's control 354 of 360, the limiter's alone 8 (cyclic) -- the limiter's range
// reaches across the pair in few cells of this field, which is why it has a control of its own.
#include "box_mesh.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "interface_properties_cpp.cuh"
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
    // gradients compared at the listed cells, and how many differ
    long compared = 0;
    long differing = 0;
    // ...of them at cells of the pair
    long onPair = 0;
    long pairDiffering = 0;
    // listed cells the limiter changed
    long limited = 0;
};

// the cells of every patch that is not empty
std::vector<label> patchCells(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& fvp)
{
    std::vector<char> on(static_cast<std::size_t>(m.nCells()), 0);
    for (const FvPatch& q : fvp)
    {
        if (q.type == "empty") continue;
        for (label i = 0; i < q.size; ++i)
        {
            on[static_cast<std::size_t>(q.faceCells[i])] = 1;
        }
    }
    std::vector<label> cells;
    for (label c = 0; c < m.nCells(); ++c)
    {
        if (on[static_cast<std::size_t>(c)])
        {
            cells.push_back(c);
        }
    }
    return cells;
}

// the whole-mesh forms on `whole`'s patches against the subset forms: the fit on `fitPatches`, the limiter on
// `limitPatches` (both `whole` itself outside the controls). Two fields, k = 0 (the fit alone), 1 and 0.5.
Tally run(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& whole,
    const std::vector<FvPatch>& fitPatches,
    const std::vector<FvPatch>& limitPatches)
{
    const label nC = m.nCells();
    const std::vector<label> cells = patchCells(m, whole);
    std::vector<char> pairCell(static_cast<std::size_t>(nC), 0);
    for (const FvPatch& q : whole)
    {
        if (!q.coupled) continue;
        for (label i = 0; i < q.size; ++i)
        {
            pairCell[static_cast<std::size_t>(q.faceCells[i])] = 1;
        }
    }
    fvc::GradSubset at = fvc::gradSubset(m, cells);
    Tally t;
    for (int field = 0; field < 2; ++field)
    {
        // a field with no flat spot and no period of the box's, so the values across the pair are not the cells'
        std::vector<scalar> a(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            const vector& x = g.C()[c];
            a[static_cast<std::size_t>(c)] = scalar(0.5)
                + scalar(0.4)*std::sin(scalar(1.3 + field)*x.x + scalar(0.7)*x.y + scalar(0.45)*x.z)
                + scalar(0.05)*std::cos(scalar(2.1)*x.y - scalar(0.3 + 0.2*field)*x.x);
        }
        std::vector<std::vector<scalar>> bnd(whole.size());
        for (std::size_t pi = 0; pi < whole.size(); ++pi)
        {
            bnd[pi].resize(static_cast<std::size_t>(whole[pi].size));
            for (label i = 0; i < whole[pi].size; ++i)
            {
                bnd[pi][static_cast<std::size_t>(i)] =
                    a[static_cast<std::size_t>(whole[pi].faceCells[i])] + scalar(0.07)*scalar(1 + pi % 3);
            }
        }
        const std::vector<vector> fit = fvc::leastSquaresGrad(a, bnd, m, g, whole);
        for (const scalar k : {scalar(0), scalar(1), scalar(0.5)})
        {
            std::vector<vector> wholeK = fit;
            fvc::leastSquaresGradAt(a, bnd, m, g, fitPatches, at);
            if (k > scalar(0))
            {
                cpu::cellLimitGrad(wholeK, a, bnd, k, m, g, whole);
                cpu::cellLimitGradAt(at, a, bnd, k, m, g, limitPatches);
            }
            for (const label c : cells)
            {
                const std::size_t s = static_cast<std::size_t>(c);
                const bool same = sameBits(at.grad[s], wholeK[s]);
                ++t.compared;
                t.differing += same ? 0 : 1;
                t.onPair += pairCell[s] ? 1 : 0;
                t.pairDiffering += (pairCell[s] && !same) ? 1 : 0;
                t.limited += (k > scalar(0) && !sameBits(wholeK[s], fit[s])) ? 1 : 0;
            }
        }
    }
    return t;
}

} // namespace

int main()
{
    std::printf("== a least-squares gradient and its cell limiter at a subset of the cells, across a coupled "
                "pair ==\n");
    PrimitiveMesh m = boxtest::boxMesh(7, 6, 5, scalar(0.3), scalar(1), scalar(0.7), scalar(1.3));
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> walled = buildPatches(m, g);
    const bool sides = walled.size() > 1 && walled[0].name == "inlet" && walled[1].name == "outlet"
                    && walled[0].size == walled[1].size && walled[0].size > 0;
    if (!sides)
    {
        std::printf("  the box's first two patches are not its x sides\n");
        return 1;
    }
    // the pair: a translational cyclic, coupled by the solver's own routine
    std::vector<FvPatch> cyclic = walled;
    cyclic[0].type = "cyclic";
    cyclic[1].type = "cyclic";
    coupleTranslationalPair(cyclic[0], walled[1], 1, true, g);
    coupleTranslationalPair(cyclic[1], walled[0], 0, false, g);

    // 1: the cyclic pair
    {
        const cpu::interfaceProps::NHatBoundaryStencil st =
            cpu::interfaceProps::nHatBoundaryStencil(m, cyclic, true);
        const std::vector<label> cells = patchCells(m, cyclic);
        const Tally t = run(m, g, cyclic, cyclic, cyclic);
        std::printf("  cyclic pair (%ld faces a side): %ld gradients compared, %ld differ; %ld at cells of the "
                    "pair; the limiter changed %ld\n", static_cast<long>(cyclic[0].size), t.compared, t.differing,
                    t.onPair, t.limited);
        check("across a translational cyclic the subset's fit and limiter (k = 1, k = 0.5) are the whole mesh's, "
              "bit for bit, and the solver's own stencil lists these cells",
              t.compared > 0 && t.differing == 0 && t.onPair > 0 && t.limited > 0 && st.usable
              && st.cells == cells);
    }
    // 2: the same pair with a cyclicAMI's addressing: two neighbour cells a face
    std::vector<FvPatch> ami = cyclic;
    for (int side = 0; side < 2; ++side)
    {
        FvPatch& p = ami[static_cast<std::size_t>(side)];
        const FvPatch& q = cyclic[static_cast<std::size_t>(1 - side)];
        const label n = p.size;
        p.nbrFaceCells.clear();
        p.amiOffsets.assign(1, 0);
        for (label i = 0; i < n; ++i)
        {
            p.amiNbrFaces.push_back(i);
            p.amiNbrFaces.push_back((i + 1) % n);
            p.amiNbrCells.push_back(q.faceCells[i]);
            p.amiNbrCells.push_back(q.faceCells[(i + 1) % n]);
            p.amiWeights.push_back(scalar(0.6));
            p.amiWeights.push_back(scalar(0.4));
            p.amiOffsets.push_back(static_cast<label>(p.amiNbrCells.size()));
        }
    }
    {
        const Tally t = run(m, g, ami, ami, ami);
        // PREMISE: the weighted sums are another answer than the cyclic's one cell a face -- or this arm would
        // be the first again
        const Tally other = run(m, g, cyclic, ami, ami);
        std::printf("  the pair as a cyclicAMI (two neighbour cells a face): %ld compared, %ld differ; %ld at "
                    "cells of the pair; the limiter changed %ld; against the cyclic's own answer %ld of them "
                    "differ\n", t.compared, t.differing, t.onPair, t.limited, other.pairDiffering);
        check("across a cyclicAMI's weighted sums the subset's fit and limiter are the whole mesh's, bit for bit",
              t.compared > 0 && t.differing == 0 && t.onPair > 0 && t.limited > 0 && other.pairDiffering > 0);
    }
    // 3: the controls -- the subset form handed the pair as two uncoupled patches
    {
        std::vector<FvPatch> uncoupled = cyclic;
        for (int side = 0; side < 2; ++side)
        {
            uncoupled[static_cast<std::size_t>(side)] = walled[static_cast<std::size_t>(side)];
        }
        const Tally fitWrong = run(m, g, cyclic, uncoupled, uncoupled);
        const Tally limitWrong = run(m, g, cyclic, cyclic, uncoupled);
        const Tally limitWrongAmi = run(m, g, ami, ami, uncoupled);
        std::printf("  CONTROL, the subset form's pair uncoupled: in the fit %ld of %ld gradients at cells of "
                    "the pair differ; in the limiter alone %ld (cyclic) and %ld (cyclicAMI)\n",
                    fitWrong.pairDiffering, fitWrong.onPair, limitWrong.pairDiffering,
                    limitWrongAmi.pairDiffering);
        check("CONTROL: with the pair uncoupled in the subset's fit, and in its limiter alone on either kind of "
              "pair, cells of the pair differ",
              fitWrong.pairDiffering > 0 && limitWrong.pairDiffering > 0 && limitWrongAmi.pairDiffering > 0);
    }
    std::printf("test_grad_subset_coupled: %d failures\n", failures);
    return failures ? 1 : 0;
}
