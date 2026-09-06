// THE GAUSS-SEIDEL SWEEP, measured against OpenFOAM's own residual after exactly n sweeps.
//
// Run: gs_ladder <caseDir> <timeDir>
//
// brae runs a case's `smoothSolver` + a GaussSeidel-family smoother through deviceSymGaussSeidel, which
// is OpenFOAM's smoothSolver STOPPING RULE (smoothSolver.C:135-209) around symGaussSeidelSmoother.C's own
// sweep, level-scheduled onto the device (device_sym_gauss_seidel.cuh). This gate is what says the sweep
// really is OpenFOAM's and not merely called that.
//
// IT WAS NOT, until this measured it. brae ran a MULTICOLOUR sweep, which visits the same cells in a
// different order; Gauss-Seidel is order-dependent, so that was a different smoother wearing the same
// name, and the driver announced it as a substitution. What the substitution cost is the CONTROL below.
//
// THE ORACLE is tools/dumpSimpleFoam, which writes the momentum system in the FOLDED form the linear
// solver actually sees (fvMatrixSolve.C:149 `addBoundarySource`, :169 `addBoundaryDiag`), the field the
// solve starts from, and OpenFOAM's own initial/final residual after exactly n sweeps for n = 1..10 --
// obtained by solving a COPY with `tolerance 0; relTol 0; maxIter n`, which makes checkConvergence false
// forever so smoothSolver.C's do-while runs exactly n. Every number in stage_UsmoothLadder.dat is
// OpenFOAM's: its matrix, its fold, its normFactor, its smoother. The ladder is self-validating -- the
// case's own relTol-0.1 solve stopped after 5 sweeps at 0.0711018500747468, which is the n=5 rung to the
// last digit, and the gate script asserts that before this binary runs.
//
//   LEG 0  brae's initial residual on that system equals OpenFOAM's. This is the harness control: it
//          proves the matrix, the boundary fold and the normFactor are all faithful, so what LEG 2 finds
//          is the sweep and not the setup.
//   LEG 1  a transcription of symGaussSeidelSmoother.C:145-190 -- bPrime = source once per sweep, a
//          forward cell walk that gathers the upper and distributes the lower, a reverse walk that
//          re-reads the bPrime the forward half left -- reproduces OpenFOAM's ladder at every rung. This
//          is what lets LEG 2 be read as "the sweep", and not as "something in this harness".
//   LEG 2  deviceSymGaussSeidel with tol and relTol both 0 and maxIter n, so it too runs exactly n sweeps
//          from the same psi0, must EQUAL OpenFOAM at every rung.
//   CONTROL the multicolour sweep brae used to run, through the identical harness. It must MISS LEG 2's
//          bound by a wide margin, or that bound is not measuring the order.
//
// Measured on validation/T3A's first momentum solve from its own 0.orig, 26820 cells (the bounds below
// are this run, and tighten from here):
//   n     OpenFOAM      level-scheduled       colour order    colour/OpenFOAM
//   1    4.130e-01     4.130e-01              5.628e-01           1.36
//   5    7.110e-02     7.110e-02              1.960e-01           2.76
//  10    8.593e-03     8.593e-03              5.913e-02           6.88
// The colour order's gap GROWS with the sweep count, which is the shape a reordering has rather than a
// one-off offset: ten colour sweeps bought what OpenFOAM buys in about four. On T3A that is the
// difference between converging and not -- the case asks `relTol 0.1; maxIter 10`, OpenFOAM reaches
// relTol 0.1 after 5 sweeps, the colour order needs 9-10 and so always took the cap, and with the
// pressure solve tight in both codes real simpleFoam CONVERGES at 5 sweeps and DIVERGES when forced to
// take all ten (Ux 2.997e-01 at iteration 400).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "read_surface_field.cuh"
#include "device_mesh.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_amg.cuh"
#include "device_sym_gauss_seidel.cuh"   // gsLevelsFor + the sweep, for the timing line
#include "device_amg_internal.cuh"   // gsSweep + greedyColor: the colour order, as the CONTROL

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

// greedyColor has external linkage and no public declaration (device_amg_gauss_seidel.cu:23 declares it
// the same way); the CONTROL needs the colouring the old default path built.
namespace brae { Coloring greedyColor(const std::vector<label>& owner, const std::vector<label>& nei, int nC); }

// LEG 2 is an EQUALITY: the level-scheduled sweep performs OpenFOAM's operations in OpenFOAM's order, so
// only the residual sum's own round-off separates them. Set from the first green run (worst 2.9e-12).
static const scalar LADDER_TOL = 1e-11;
// The control must miss that by a wide margin. The colour order's worst rung is 6.88x OpenFOAM's
// residual; requiring 1.2x asserts the bound discriminates without pinning the greedy colouring's
// exact output, which is a mesh-ordering artefact and not a promise.
static const scalar CONTROL_MIN = 1.2;
// LEG 0 and LEG 1 both compare a 26820-term sum against gSumMag's, over a b - A.psi that cancels heavily;
// 1.2e-12 and 8.0e-13 are that round-off, and are the same floor from two independent directions.
static const scalar LEG0_TOL   = 1e-11;  // the folded system and normFactor are OpenFOAM's, to round-off
static const scalar LEG1_TOL   = 1e-11;  // the index-order transcription IS symGaussSeidelSmoother.C

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];
    const std::string tdir    = argv[2];

    std::vector<int> ln;
    std::vector<scalar> lInit, lFinal, gFinal;
    {
        std::ifstream lad(caseDir + "/stage_UsmoothLadder.dat");
        if (!lad) { std::printf("SKIP: no %s/stage_UsmoothLadder.dat\n", caseDir.c_str()); return 77; }
        int n; double a, b;
        while (lad >> n >> a >> b) { ln.push_back(n); lInit.push_back(a); lFinal.push_back(b); }
        // The ascending-only ladder, OpenFOAM's GaussSeidelSmoother under the same harness.
        std::ifstream gl(caseDir + "/stage_UgsLadder.dat");
        if (!gl) { std::printf("SKIP: no %s/stage_UgsLadder.dat\n", caseDir.c_str()); return 77; }
        while (gl >> n >> a >> b) gFinal.push_back(b);
    }
    if (ln.empty()) { std::printf("SKIP: the ladder file is empty\n"); return 77; }
    if (gFinal.size() != ln.size()) { std::printf("SKIP: the two ladders differ in length\n"); return 77; }

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();

    const std::string d = caseDir + "/" + tdir + "/";
    // The dumped system, read as raw internal lists: only the interior is a matrix, and the boundary is
    // already folded into the diagonal and the source by the writer. A cold start writes psi0 as
    // `uniform`, so a uniform entry is expanded rather than read as an empty list.
    auto internalOf = [&](const std::string& name)
    {
        const FieldData<scalar> fd = readField<scalar>(d + name);
        if (fd.internalUniform)
            return std::vector<scalar>(static_cast<std::size_t>(nC), fd.internalUniformValue);
        return fd.internalField;
    };
    const std::vector<scalar> diag  = internalOf("stage_UsolveDiag");
    const std::vector<scalar> src   = internalOf("stage_UsolveSrc");
    const std::vector<scalar> psi0  = internalOf("stage_Usolve0");
    const std::vector<scalar> upper = readSurfaceField(d + "stage_UsolveUpper", patches, nIf).internal;
    const std::vector<scalar> lower = readSurfaceField(d + "stage_UsolveLower", patches, nIf).internal;
    if (static_cast<label>(diag.size()) != nC || static_cast<label>(src.size()) != nC
        || static_cast<label>(psi0.size()) != nC)
    {
        std::printf("FAIL: the dumped system has %zu cells, the mesh has %d\n", diag.size(), (int)nC);
        return 1;
    }
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    // lduMatrix::Amul's traversal (device_ldu.cuh's own comment), on the host.
    auto amul = [&](const std::vector<scalar>& x, std::vector<scalar>& y)
    {
        for (label c = 0; c < nC; ++c) y[c] = diag[c] * x[c];
        for (label f = 0; f < nIf; ++f)
        {
            y[own[f]] += upper[f] * x[nei[f]];
            y[nei[f]] += lower[f] * x[own[f]];
        }
    };

    // lduMatrixSolver.C:251-264, DEFAULT_NORM: sumA (lduMatrix::sumA, the row sum) times the AVERAGE of
    // psi, then sum(|A.psi - that| + |b - that|) + solverPerformance::small_.
    std::vector<scalar> sumA(nC), Apsi(nC);
    for (label c = 0; c < nC; ++c) sumA[c] = diag[c];
    for (label f = 0; f < nIf; ++f) { sumA[own[f]] += upper[f]; sumA[nei[f]] += lower[f]; }
    amul(psi0, Apsi);
    scalar xRef = 0;
    for (label c = 0; c < nC; ++c) xRef += psi0[c];
    xRef /= static_cast<scalar>(nC);
    scalar normFactor = 0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar t = sumA[c] * xRef;
        normFactor += std::fabs(Apsi[c] - t) + std::fabs(src[c] - t);
    }
    normFactor += 1e-20;

    auto residual = [&](const std::vector<scalar>& x)
    {
        std::vector<scalar> y(static_cast<std::size_t>(nC));
        amul(x, y);
        scalar s = 0;
        for (label c = 0; c < nC; ++c) s += std::fabs(src[c] - y[c]);
        return s / normFactor;
    };

    int fails = 0;
    auto say = [&](const std::string& msg, bool ok)
    {
        std::printf("  %-70s %s\n", msg.c_str(), ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };

    // ---- LEG 0: the harness control ----------------------------------------------------------------
    const scalar init0 = residual(psi0);
    const scalar relInit = std::fabs(init0 - lInit[0]) / std::fabs(lInit[0]);
    std::printf("  LEG 0  initial residual  brae %.15e  OpenFOAM %.15e  rel %.3e\n",
                (double)init0, (double)lInit[0], (double)relInit);
    say("LEG 0  the folded system and the normFactor are OpenFOAM's", relInit < LEG0_TOL);

    // ---- LEG 1: symGaussSeidelSmoother.C, transcribed -----------------------------------------------
    // The forward walk reads the faces OWNED by the cell (ownerStart[c]..ownerStart[c+1]), gathering
    // upper*psi[neighbour] and distributing lower*psi_c forward into bPrime; the reverse walk gathers the
    // same faces off the bPrime the forward half left, and does NOT distribute again (:191).
    std::vector<label> ownStart(static_cast<std::size_t>(nC) + 1, 0);
    for (label f = 0; f < nIf; ++f) ++ownStart[own[f] + 1];
    for (label c = 0; c < nC; ++c) ownStart[c + 1] += ownStart[c];
    auto sequentialSweeps = [&](int nSweeps, bool symmetric)
    {
        std::vector<scalar> psi(psi0), bPrime(static_cast<std::size_t>(nC));
        for (int s = 0; s < nSweeps; ++s)
        {
            bPrime = src;
            for (label c = 0; c < nC; ++c)
            {
                scalar psii = bPrime[c];
                for (label f = ownStart[c]; f < ownStart[c + 1]; ++f) psii -= upper[f] * psi[nei[f]];
                psii /= diag[c];
                for (label f = ownStart[c]; f < ownStart[c + 1]; ++f) bPrime[nei[f]] -= lower[f] * psii;
                psi[c] = psii;
            }
            // GaussSeidelSmoother.C:145-176 stops after the ascending walk; symGaussSeidelSmoother.C
            // adds this descending one, which re-reads the bPrime the forward half left (:191 does not
            // distribute again).
            if (!symmetric) continue;
            for (label c = nC - 1; c >= 0; --c)
            {
                scalar psii = bPrime[c];
                for (label f = ownStart[c]; f < ownStart[c + 1]; ++f) psii -= upper[f] * psi[nei[f]];
                psii /= diag[c];
                psi[c] = psii;
            }
        }
        return residual(psi);
    };

    // ---- LEG 2: brae's own sweep, and the colour order it replaced --------------------------------
    DeviceMesh dm = buildDeviceMesh(m, g, patches);
    DeviceBuffer<scalar> dDiag, dUp, dLo, dB;
    dDiag.copyFrom(diag); dUp.copyFrom(upper); dLo.copyFrom(lower); dB.copyFrom(src);
    const DeviceLduView A = deviceLduView(dm, dDiag, dUp, dLo);

    // The CONTROL's colouring, built exactly as the old default path built it.
    GridColoring gc;
    {
        // owner() spans the boundary faces too; the colouring, like the LDU, sees only the internal ones.
        const std::vector<label> ownIn(own.begin(), own.begin() + nIf);
        const Coloring c = greedyColor(ownIn, nei, (int)nC);
        gc.nColors = c.nColors;
        gc.cells.copyFrom(c.cells);
        gc.start.copyFrom(c.start);
        gc.startH = c.start;
    }

    std::printf("  symGaussSeidel -- OpenFOAM's ascending-then-descending sweep\n");
    std::printf("  %-3s %-22s %-22s %-22s %-22s\n", "n", "OpenFOAM", "index order (LEG 1)",
                "brae (level-scheduled)", "colour order (CONTROL)");
    scalar worstSeq = 0, worstBrae = 0, worstControl = 0;
    for (std::size_t i = 0; i < ln.size(); ++i)
    {
        const int n = ln[i];
        const scalar seq = sequentialSweeps(n, true);

        DeviceBuffer<scalar> dPsi;
        dPsi.copyFrom(psi0);
        DeviceSolverPerf perf;
        deviceSymGaussSeidel(A, dB, dPsi, normFactor, 0.0, 0.0, n, &perf);

        DeviceBuffer<scalar> cPsi;
        cPsi.copyFrom(psi0);
        for (int s = 0; s < n; ++s)
        {
            gsSweep(A, dB, cPsi, gc, true);
            gsSweep(A, dB, cPsi, gc, false);
        }
        std::vector<scalar> cH;
        cPsi.copyTo(cH);
        const scalar ctl = residual(cH);

        std::printf("  %-3d %-22.15g %-22.15g %-22.15g %-22.15g\n",
                    n, (double)lFinal[i], (double)seq, (double)perf.finalResidual, (double)ctl);
        worstSeq     = std::fmax(worstSeq,     std::fabs(seq - lFinal[i]) / std::fabs(lFinal[i]));
        worstBrae    = std::fmax(worstBrae,    std::fabs(perf.finalResidual - lFinal[i]) / std::fabs(lFinal[i]));
        worstControl = std::fmax(worstControl, ctl / lFinal[i]);
    }

    // The OTHER smoother in the family. GaussSeidelSmoother.C's sweep loop is the ascending walk and
    // nothing else, so at every rung it leaves MORE residual than symGaussSeidel -- and answering a case
    // that named it with the symmetric sweep gives it about twice the smoothing it asked for. The
    // symmetric sweep is therefore this leg's control, exactly as the colour order is LEG 2's.
    std::printf("  GaussSeidel -- OpenFOAM's ascending-only sweep\n");
    std::printf("  %-3s %-22s %-22s %-22s %-22s\n", "n", "OpenFOAM", "index order (LEG 1)",
                "brae (level-scheduled)", "symmetric (CONTROL)");
    scalar worstSeqG = 0, worstBraeG = 0, worstControlG = 1e30;
    for (std::size_t i = 0; i < ln.size(); ++i)
    {
        const int n = ln[i];
        const scalar seq = sequentialSweeps(n, false);

        DeviceBuffer<scalar> dPsi;
        dPsi.copyFrom(psi0);
        DeviceSolverPerf perf;
        deviceSymGaussSeidel(A, dB, dPsi, normFactor, 0.0, 0.0, n, &perf,
                             /*minIter*/0, /*nSweeps*/1, /*symmetric*/false);

        std::printf("  %-3d %-22.15g %-22.15g %-22.15g %-22.15g\n",
                    n, (double)gFinal[i], (double)seq, (double)perf.finalResidual, (double)lFinal[i]);
        worstSeqG     = std::fmax(worstSeqG,  std::fabs(seq - gFinal[i]) / std::fabs(gFinal[i]));
        worstBraeG    = std::fmax(worstBraeG, std::fabs(perf.finalResidual - gFinal[i]) / std::fabs(gFinal[i]));
        worstControlG = std::fmin(worstControlG, gFinal[i] / lFinal[i]);
    }

    // WALL-CLOCK, reported and not asserted: ten symmetric sweeps of this system, so the gate script
    // can print the single-block walk against the forced per-level launches (BRAE_GS_PER_LEVEL=1)
    // side by side. The numbers above are the same either way; this is what the choice costs.
    {
        DeviceBuffer<scalar> tPsi;
        tPsi.copyFrom(psi0);
        const DeviceGaussSeidelLevels& lv = gsLevelsFor(A);
        // the walk-order operands once, as the solver refreshes them once per solve (a null hands the
        // sweep a scratch it re-permutes every call, which would time the permutation, not the walk)
        GSLevelCoefs lc;
        GSLevelCells cc;
        gsLevelCoefsRefresh(A, lv, lc);
        gsLevelCellsRefresh(A, dB, lv, cc);
        deviceSymGaussSeidelSweepExact(A, dB, tPsi, lv, true, &lc, &cc);   // warm the levels and the kernel
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < 10; ++s) deviceSymGaussSeidelSweepExact(A, dB, tPsi, lv, true, &lc, &cc);
        cudaDeviceSynchronize();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  TIMING  10 symmetric sweeps, %d levels (widest %d cells), %s: %.2f ms  (%.1f us per half-sweep)\n",
                    lv.levels(), lv.maxLevelWidth,
                    std::getenv("BRAE_GS_PER_LEVEL") ? "per-level launches" : "single-block walk",
                    ms, 1000.0 * ms / 20.0);
    }
    // ...and the same ten sweeps on the HOST (item 68's question): the transcription above IS the exact
    // sweep, so a host momentum smoother would cost this plus, per solve, the download of the folded
    // system and the field and the upload of the result. Reported beside the device number so the
    // choice is a measurement; single-threaded, because the sweep is a recurrence.
    {
        std::vector<scalar> psi(psi0), bPrime(static_cast<std::size_t>(nC));
        double msHost = 1e30;
        for (int rep = 0; rep < 3; ++rep)   // best of three: the host number moved 2x between runs once
        {
        psi = psi0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < 10; ++s)
        {
            bPrime = src;
            for (label c = 0; c < nC; ++c)
            {
                scalar psii = bPrime[c];
                for (label f = ownStart[c]; f < ownStart[c + 1]; ++f) psii -= upper[f] * psi[nei[f]];
                psii /= diag[c];
                for (label f = ownStart[c]; f < ownStart[c + 1]; ++f) bPrime[nei[f]] -= lower[f] * psii;
                psi[c] = psii;
            }
            for (label c = nC - 1; c >= 0; --c)
            {
                scalar psii = bPrime[c];
                for (label f = ownStart[c]; f < ownStart[c + 1]; ++f) psii -= upper[f] * psi[nei[f]];
                psii /= diag[c];
                psi[c] = psii;
            }
        }
        msHost = std::fmin(msHost, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        // Is the host transcription's psi the device walk's, BIT FOR BIT? Both subtract the same terms
        // in the same order and both contract to a fused multiply-add, so they should be; the residual
        // rows above cannot say (their norms are summed in different orders). Reported, not asserted:
        // this is item 68's admissibility question.
        {
            DeviceBuffer<scalar> dPsiH;
            dPsiH.copyFrom(psi0);
            const DeviceGaussSeidelLevels& lvH = gsLevelsFor(A);
            for (int s = 0; s < 10; ++s) deviceSymGaussSeidelSweepExact(A, dB, dPsiH, lvH);
            std::vector<scalar> devPsi;
            dPsiH.copyTo(devPsi);
            std::size_t nDiff = 0;
            double maxRel = 0;
            for (label c = 0; c < nC; ++c)
            {
                if (devPsi[c] != psi[c]) ++nDiff;
                maxRel = std::fmax(maxRel, std::fabs(devPsi[c] - psi[c]) / std::fmax(std::fabs(psi[c]), 1e-300));
            }
            std::printf("  HOST-VS-DEVICE  psi after 10 symmetric sweeps: %zu of %d cells differ, max rel %.3e\n", nDiff, (int)nC, maxRel);
        }
        // the per-solve traffic: diag, upper, lower, source and psi down, psi back up (pageable, as the
        // solver's own reads are)
        std::vector<scalar> hDiag(nC), hUp(nIf), hLo(nIf), hSrc(nC), hPsi(nC);
        DeviceBuffer<scalar> dPsiT;
        dPsiT.copyFrom(psi0);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        dDiag.copyTo(hDiag); dUp.copyTo(hUp); dLo.copyTo(hLo); dB.copyTo(hSrc); dPsiT.copyTo(hPsi);
        dPsiT.copyFrom(hPsi);
        cudaDeviceSynchronize();
        const double msXfer = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::printf("  TIMING  host: 10 symmetric sweeps %.2f ms (%.1f us per half-sweep); per-solve transfers %.2f ms (%d cells, %d faces)\n",
                    msHost, 1000.0 * msHost / 20.0, msXfer, (int)nC, (int)nIf);
    }

    char buf[220];
    std::snprintf(buf, sizeof buf,
                  "LEG 1  the index-order transcription IS symGaussSeidelSmoother.C (worst rel %.3e)",
                  (double)worstSeq);
    say(buf, worstSeq < LEG1_TOL);

    std::snprintf(buf, sizeof buf,
                  "LEG 2  brae's symGaussSeidel IS OpenFOAM's, at every rung (worst rel %.3e)",
                  (double)worstBrae);
    say(buf, worstBrae < LADDER_TOL);

    std::snprintf(buf, sizeof buf,
                  "CONTROL  the colour order misses that bound by %.2fx (need >= %.2fx)",
                  (double)worstControl, (double)CONTROL_MIN);
    say(buf, worstControl >= CONTROL_MIN);

    std::snprintf(buf, sizeof buf,
                  "LEG 3  the ascending-only transcription IS GaussSeidelSmoother.C (worst rel %.3e)",
                  (double)worstSeqG);
    say(buf, worstSeqG < LEG1_TOL);

    std::snprintf(buf, sizeof buf,
                  "LEG 4  brae's GaussSeidel IS OpenFOAM's, at every rung (worst rel %.3e)",
                  (double)worstBraeG);
    say(buf, worstBraeG < LADDER_TOL);

    std::snprintf(buf, sizeof buf,
                  "CONTROL  the symmetric sweep leaves %.2fx less residual (need >= %.2fx)",
                  (double)worstControlG, (double)CONTROL_MIN);
    say(buf, worstControlG >= CONTROL_MIN);

    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
