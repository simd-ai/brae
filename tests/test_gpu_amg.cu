// GPU offload (#6): AMG preconditioner. Solve a pressure-like Laplacian on pitzDaily with both the device
// Jacobi-PCG and the device AMG-PCG; they must reach the SAME solution, and the AMG V-cycle must cut the
// iteration count (the #1 perf lever vs Jacobi).
//
// ...and, since the multicolour Gauss-Seidel smoother learned a COLOUR-MAJOR PERMUTED LAYOUT
// (device_amg_smoothers.cu, THE LAYOUT), the arms that hold that layout to the sweep it replaces.
//
// WHAT WAS MEASURED. squareBend's transonic pressure at 305,760 cells on a GB10, BEFORE the change:
// multicolour GS needed the fewest V-cycles of any smoother (2.5 per solve against the two-stage
// Gauss-Seidel's 2.7 and weighted Jacobi's 3.5) and was the slowest in wall time (25.1 ms per outer
// iteration against 19.9 and 23.0) -- 10.0 ms per cycle against the two-stage's 7.4. It lost on apply
// cost alone: gsColorT walks a cells[] indirection over the natural cell numbering, so every colour
// launch touches every cache line of every per-cell and per-face array and uses only the fraction
// belonging to its colour. AFTER is not measured here (this test runs on pitzDaily's 12,225 cells and
// times nothing); the integrator reads it from a 306k squareBend under
//     BRAE_AMG_GS=1 BRAE_PHASE_TIME=1        (and BRAE_AMG_GS_PERM=0 for the control)
// taking `p` from the "[phase] ... of which the linear solves" line and the `pIters` mean. What THIS
// test is for is the other half of that claim: that the two layouts are the same solver, so the cycle
// count and the iterate cannot move and only the time can.
//
//   (a) one sweep, both layouts, forward and backward, on every grid of a real hierarchy: the same
//       iterate BIT FOR BIT (memcmp), not to a tolerance. The permuted row carries its entries in
//       gsColorT's own accumulation order -- the owned faces' upper terms in face order, then the
//       neighboured faces' lower terms in losort order -- and accumulates them into the same `off`
//       before the same (b - off)/safeDiag(diag), so the summation order is not merely equivalent, it
//       is the same sequence of the same operands. (a2) is its fail-proof: the same comparison against
//       a sweep that ran the colours the other way must report a difference.
//   (b) the permutation is a bijection of [0, nCells) on every grid, and the permuted ADDRESSING
//       reproduces that grid's matrix-vector product: amgPermutedLayoutAmul against deviceAmul, again
//       bit for bit (the permuted matvec sums the diagonal then the row's entries in their layout
//       order, which is amulKernel's order).
//   (c) a CORRUPTED permutation is caught: one entry of the fine grid's rowNbr is moved, and the
//       layout checker must name it AND the matvec must then disagree with deviceAmul -- the second
//       half is what makes the first a test rather than a formality. Restoring the entry restores both.
//   (d) the coefficients FOLLOW the Galerkin update: the fine matrix is re-scaled, amgGalerkin is
//       re-run, and every grid's permuted matvec must again equal deviceAmul. This is the arm that
//       would fail if the gather were made conditional on a per-solve flag and frozen at the values of
//       the first iteration -- the failure mode the gather's placement in amgGalerkin exists to avoid.
//   (e) end-to-end: the whole AMG-PCG solve with the GS smoother (so the permuted sweep runs inside the
//       real V-cycle, graph capture included) must reach the Jacobi-PCG's solution.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"
#include "device_amg_internal.cuh"   // Coloring / greedyColor: the colouring the smoother is built on
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    GeometricField<scalar> f; f.internal.assign(nC, 0.0);
    for (const FvPatch& p : fvp) {
        if (p.type == "empty")     f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        else if (p.type == "wall") f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        else                       f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(p, true, 0.0, std::vector<scalar>{}));
    }
    f.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(f, 1.0, m, g, fvp);
    for (label c = 0; c < nC; ++c) M.source[c] = g.V()[c] * std::sin(0.01 * c + 0.4);

    std::vector<scalar> diagC = M.diag, b = M.source;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) { const label c = fvp[pi].faceCells[i]; diagC[c] += M.internalCoeffs[pi][i]; b[c] += M.boundaryCoeffs[pi][i]; }
    scalar normFactor = 0; for (label c = 0; c < nC; ++c) normFactor += std::fabs(b[c]); normFactor += 1e-20;
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    const scalar tol = 1e-10;

    DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.lower, ownerInt, m.neighbour(), nC);
    DeviceBuffer<scalar> db(b);

    // Jacobi-PCG.
    DeviceBuffer<scalar> xJ(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf sJ = deviceJacobiPCG(dM.view(), db, xJ, normFactor, tol, 0.0, 30000);

    // AMG-PCG (faceWeights = |Sf| over the internal faces).
    const std::vector<scalar> magSfInt(g.magSf().begin(), g.magSf().begin() + nIf);
    AMGData amg = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
    amgGalerkin(amg, dM.diag, dM.upper, dM.lower);
    DeviceBuffer<scalar> xA(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf sA = deviceAMGPCG(dM.view(), amg, db, xA, normFactor, tol, 0.0, 30000);

    const std::vector<scalar> xj = xJ.host(), xa = xA.host();
    scalar n = 0, d = 0; for (label c = 0; c < nC; ++c) { n = std::fmax(n, std::fabs(xa[c]-xj[c])); d = std::fmax(d, std::fabs(xj[c])); }
    const scalar solDiff = n / d;

    std::printf("GPU AMG preconditioner (nCells=%d -> %d coarse):\n", nC, amg.nCoarse);
    std::printf("  Jacobi-PCG : %d iters, res %.2e\n", sJ.nIterations, sJ.finalResidual);
    std::printf("  AMG-PCG    : %d iters, res %.2e\n", sA.nIterations, sA.finalResidual);
    std::printf("  solution diff (AMG vs Jacobi): %.3e\n", solDiff);
    std::printf("  iteration reduction          : %.2fx\n", (double)sJ.nIterations / std::max(1, sA.nIterations));
    bool pass = sA.finalResidual < tol && solDiff < 1e-6 && sA.nIterations < sJ.nIterations;

    // ---- THE COLOUR-MAJOR PERMUTED GAUSS-SEIDEL LAYOUT (see the header) --------------------------
    // A second hierarchy carrying a colouring per grid. Built by hand rather than through
    // BRAE_AMG_GS so the solve above keeps running the DEFAULT smoother: useGS() latches on its first
    // read, so setting the variable here would silently re-point the assertions above at another
    // smoother.
    AMGData amgGS = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
    amgGS.gsSmooth = true;
    {
        auto pushColouring = [&](const std::vector<label>& own, const std::vector<label>& nbr, int n)
        {
            const Coloring c = greedyColor(own, nbr, n);
            GridColoring gc;
            gc.nColors = c.nColors;
            gc.cells.copyFrom(c.cells);
            gc.start.copyFrom(c.start);
            gc.startH = c.start;
            amgGS.coloring.push_back(std::move(gc));
        };
        pushColouring(ownerInt, m.neighbour(), nC);                      // grid 0 = the fine matrix
        for (int k = 0; k < amgGS.nLevels(); ++k)
        {
            pushColouring(amgGS.level[k].cOwn.host(), amgGS.level[k].cNei.host(), amgGS.level[k].nCoarse);
        }
    }
    amgGalerkin(amgGS, dM.diag, dM.upper, dM.lower);
    const int nGrids = amgGS.nLevels();                                 // the SMOOTHED grids, 0 .. nLevels-1
    auto gridView = [&](int g) { return (g == 0) ? dM.view() : amgGS.level[g-1].coarseView(); };
    for (int g = 0; g < nGrids; ++g)
    {
        if (!amgEnsurePermutedGSLayout(amgGS.coloring[g], gridView(g)))
        {
            std::printf("  [layout] grid %d: the permuted layout could not be built\n", g);
            pass = false;
        }
    }

    // A field and a source with structure, so a wrong entry cannot cancel: every arm below reuses them.
    auto seedField = [](int n, double phase)
    {
        std::vector<scalar> v(n);
        for (int c = 0; c < n; ++c) v[c] = std::sin(0.037*c + phase) + 0.25*std::cos(0.011*c);
        return v;
    };
    auto sameBits = [](const std::vector<scalar>& a, const std::vector<scalar>& b)
    {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0;
    };

    int nBijectionBad = 0, nCheckBad = 0, nAmulBad = 0, nSweepBad = 0, nControlBlind = 0;
    for (int g = 0; g < nGrids; ++g)
    {
        const DeviceLduView Ag = gridView(g);
        const GridColoring& gc = amgGS.coloring[g];
        const int n = Ag.nCells;

        // (b) the permutation is a bijection of [0, nCells), and the layout is what this grid's
        // addressing says it should be (the checker walks every row's entries against ownerStart /
        // nei / losort / owner and refuses two cells of one colour sharing a face).
        {
            const std::vector<label> cells = gc.cells.host();
            std::vector<char> seen(n, 0);
            bool ok = static_cast<int>(cells.size()) == n;
            for (std::size_t i = 0; ok && i < cells.size(); ++i)
            {
                const label c = cells[i];
                if (c < 0 || c >= n || seen[c]) ok = false;
                else seen[c] = 1;
            }
            if (!ok)
            {
                ++nBijectionBad;
                std::printf("  [layout] grid %d: cells[] is not a permutation of [0, %d)\n", g, n);
            }
        }
        const std::string bad = amgCheckPermutedGSLayout(gc, Ag);
        if (!bad.empty())
        {
            ++nCheckBad;
            std::printf("  [layout] grid %d: %s\n", g, bad.c_str());
        }

        // (b) the permuted ADDRESSING reproduces this grid's matrix-vector product, bit for bit.
        DeviceBuffer<scalar> psi(seedField(n, 0.3*g));
        DeviceBuffer<scalar> yNat, yPerm;
        deviceAmul(Ag, psi, yNat);
        amgPermutedLayoutAmul(gc, Ag, psi, yPerm);
        if (!sameBits(yNat.host(), yPerm.host()))
        {
            ++nAmulBad;
            std::printf("  [layout] grid %d: the permuted matvec is not deviceAmul's bits\n", g);
        }

        // (a) one sweep of each layout, forward then backward, from the same iterate. Bit for bit.
        DeviceBuffer<scalar> bg(seedField(n, 1.7 + 0.3*g));
        const std::vector<scalar> x0 = seedField(n, 0.9*g);
        DeviceBuffer<scalar> xInd(x0), xPerm(x0), xRev(x0);
        amgGSSweepIndirect(Ag, bg, xInd, gc, true);
        amgGSSweepPermuted(Ag, bg, xPerm, gc, true);
        const bool fwdSame = sameBits(xInd.host(), xPerm.host());
        amgGSSweepIndirect(Ag, bg, xInd, gc, false);
        amgGSSweepPermuted(Ag, bg, xPerm, gc, false);
        const bool bothSame = fwdSame && sameBits(xInd.host(), xPerm.host());
        if (!bothSame)
        {
            ++nSweepBad;
            std::printf("  [layout] grid %d: the permuted sweep is not the indirection sweep's bits\n", g);
        }
        // (a2) the fail-proof: a sweep that ran the colours the other way must NOT compare equal, or
        // the memcmp above proves nothing. Skipped on a one-colour grid, where the two orders are the
        // same sweep and equality is the right answer.
        if (gc.nColors > 1)
        {
            amgGSSweepPermuted(Ag, bg, xRev, gc, false);
            if (sameBits(xRev.host(), xPerm.host()))
            {
                ++nControlBlind;
                std::printf("  [layout] grid %d: the reversed-colour control compares EQUAL; arm (a) is blind\n", g);
            }
        }
    }
    std::printf("  [layout] %d smoothed grids: bijection %s, checker %s, matvec %s, sweep %s, control %s\n",
                nGrids,
                nBijectionBad ? "FAIL" : "ok", nCheckBad ? "FAIL" : "ok", nAmulBad ? "FAIL" : "ok",
                nSweepBad ? "FAIL" : "ok", nControlBlind ? "FAIL" : "ok");
    pass = pass && !nBijectionBad && !nCheckBad && !nAmulBad && !nSweepBad && !nControlBlind;

    // (c) a CORRUPTED permutation must be caught. Move one entry of the fine grid's rowNbr to another
    // row (still in range, so the matvec runs and the fault has to be found by looking, not by
    // crashing): the checker must name it, and the matvec must then disagree with deviceAmul.
    {
        const GridColoring& gc = amgGS.coloring[0];
        const DeviceLduView A0 = dM.view();
        std::vector<label> rows = gc.rowNbr.host();
        const label good = rows[0];
        rows[0] = (good + 1) % nC;
        gc.rowNbr.copyFrom(rows);
        const std::string bad = amgCheckPermutedGSLayout(gc, A0);
        DeviceBuffer<scalar> psi(seedField(nC, 0.11));
        DeviceBuffer<scalar> yNat, yPerm;
        deviceAmul(A0, psi, yNat);
        amgPermutedLayoutAmul(gc, A0, psi, yPerm);
        const bool matvecMoved = !sameBits(yNat.host(), yPerm.host());
        rows[0] = good;
        gc.rowNbr.copyFrom(rows);
        const std::string good2 = amgCheckPermutedGSLayout(gc, A0);
        amgPermutedLayoutAmul(gc, A0, psi, yPerm);
        const bool restored = good2.empty() && sameBits(yNat.host(), yPerm.host());
        std::printf("  [layout] corrupted entry: checker %s (\"%s\"), matvec moved %s, restored %s\n",
                    bad.empty() ? "MISSED" : "caught", bad.empty() ? "" : bad.c_str(),
                    matvecMoved ? "yes" : "NO", restored ? "yes" : "NO");
        pass = pass && !bad.empty() && matvecMoved && restored;
    }

    // (d) the coefficients follow the Galerkin update. Re-scale the fine matrix, re-run amgGalerkin
    // (which re-gathers every grid's permuted coefficients), and every grid's permuted matvec must
    // again be deviceAmul's bits. A gather that had been frozen at the values the layout was built
    // with would fail here on every grid.
    {
        std::vector<scalar> dv = dM.diag.host(), uv = dM.upper.host(), lv = dM.lower.host();
        for (std::size_t i = 0; i < dv.size(); ++i) dv[i] *= 1.0 + 0.5*std::sin(0.019*static_cast<double>(i));
        for (std::size_t i = 0; i < uv.size(); ++i) uv[i] *= 1.0 + 0.3*std::cos(0.023*static_cast<double>(i));
        for (std::size_t i = 0; i < lv.size(); ++i) lv[i] *= 1.0 - 0.2*std::sin(0.017*static_cast<double>(i));
        dM.diag.copyFrom(dv);
        dM.upper.copyFrom(uv);
        dM.lower.copyFrom(lv);
        amgGalerkin(amgGS, dM.diag, dM.upper, dM.lower);
        int nStale = 0;
        for (int g = 0; g < nGrids; ++g)
        {
            const DeviceLduView Ag = gridView(g);
            DeviceBuffer<scalar> psi(seedField(Ag.nCells, 0.7 + 0.3*g));
            DeviceBuffer<scalar> yNat, yPerm;
            deviceAmul(Ag, psi, yNat);
            amgPermutedLayoutAmul(amgGS.coloring[g], Ag, psi, yPerm);
            if (!sameBits(yNat.host(), yPerm.host())) ++nStale;
        }
        std::printf("  [layout] after a re-Galerkin on changed values: %d of %d grids stale\n", nStale, nGrids);
        pass = pass && nStale == 0;
    }

    // (e) end-to-end, on the ORIGINAL matrix: the GS-smoothed V-cycle -- which is where the permuted
    // sweep actually runs, graph capture and all -- must reach the Jacobi-PCG's solution.
    {
        dM.diag.copyFrom(diagC);
        dM.upper.copyFrom(M.upper);
        dM.lower.copyFrom(M.lower);
        amgGalerkin(amgGS, dM.diag, dM.upper, dM.lower);
        DeviceBuffer<scalar> xG(std::vector<scalar>(nC, 0.0));
        const DeviceSolverPerf sG = deviceAMGPCG(dM.view(), amgGS, db, xG, normFactor, tol, 0.0, 30000);
        const std::vector<scalar> xg = xG.host();
        scalar ng = 0;
        for (label c = 0; c < nC; ++c) ng = std::fmax(ng, std::fabs(xg[c]-xj[c]));
        const scalar gsDiff = ng / d;
        std::printf("  [layout] AMG-PCG with the multicolour GS smoother: %d iters, res %.2e, diff vs Jacobi %.3e\n",
                    sG.nIterations, sG.finalResidual, gsDiff);
        pass = pass && sG.finalResidual < tol && gsDiff < 1e-6;
    }

    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
