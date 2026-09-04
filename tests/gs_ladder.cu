// THE GAUSS-SEIDEL SWEEP, measured against OpenFOAM's own residual after exactly n sweeps.
//
// Run: gs_ladder <caseDir> <timeDir>
//
// brae routes a case's `smoothSolver` + a GaussSeidel-family smoother to deviceSymGaussSeidel, which is
// OpenFOAM's smoothSolver STOPPING RULE (smoothSolver.C:135-209) around a MULTICOLOUR sweep, where
// symGaussSeidelSmoother.C walks cells in strict index order (:145 forward, :175 reverse). Same algorithm
// under a permutation -- and Gauss-Seidel is order-dependent, so the iterate after n sweeps is not
// OpenFOAM's. Until now that substitution was asserted only as a NOTICE the driver must print
// (tests/gs_smoother_notice.sh). This measures what it costs.
//
// THE ORACLE is tools/dumpSimpleFoam, which writes the momentum system in the FOLDED form the linear
// solver actually sees (fvMatrixSolve.C:149 addBoundarySource, :169 addBoundaryDiag), the field the solve
// starts from, and OpenFOAM's own initial/final residual after exactly n sweeps for n = 1..10 -- obtained
// by solving a COPY with `tolerance 0; relTol 0; maxIter n`, which makes checkConvergence false forever so
// smoothSolver.C's do-while runs exactly n. Every number in stage_UsmoothLadder.dat is OpenFOAM's: its
// matrix, its fold, its normFactor, its smoother. The ladder is self-validating -- the case's own
// relTol-0.1 solve stopped after 5 sweeps at 0.0711018500747468, which is the n=5 rung to the last digit.
//
//   LEG 0  brae's initial residual on that system equals OpenFOAM's. This is the harness control: it
//          proves the matrix, the boundary fold and the normFactor are all faithful, so what LEG 2 finds
//          is the sweep and not the setup.
//   LEG 1  a transcription of symGaussSeidelSmoother.C:116-198 -- bPrime = source once per sweep, a
//          forward cell walk that gathers the upper and distributes the lower, a reverse walk that
//          re-reads the bPrime the forward half left -- reproduces OpenFOAM's ladder at every rung. This
//          is what lets LEG 2 be read as "the visiting order", and not as "something in this harness".
//   LEG 2  deviceSymGaussSeidel with tol and relTol both 0 and maxIter n, so it too runs exactly n sweeps
//          from the same psi0, against the same normFactor.
//
// Measured on validation/T3A's first momentum solve from its own 0.orig, 26820 cells (the bounds below
// are this run, and tighten from here):
//   n     OpenFOAM       brae       ratio
//   1    4.130e-01    5.628e-01     1.36
//   5    7.110e-02    1.960e-01     2.76
//  10    8.593e-03    5.913e-02     6.88
// The gap GROWS with the sweep count, which is the shape a reordering has rather than a one-off offset:
// every rung leaves more of the residual behind, so the deficit compounds. That is the cost of the
// substitution -- ten colour-order sweeps buy what OpenFOAM buys in about four.
//
// LEG 2 asserts two things about that: that brae is BEHIND OpenFOAM at every rung (a sweep that silently
// started matching index order is an improvement to record, not to pass over in silence), and that it
// stays within LADDER_MAX (a sweep that got worse is caught).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "read_surface_field.cuh"
#include "device_mesh.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_amg.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

// The measured ratios above, with headroom for the multicolour sweep's own reduction order.
static const scalar LADDER_MAX = 7.5;    // brae's residual after n sweeps over OpenFOAM's (worst 6.88)
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
    std::vector<scalar> lInit, lFinal;
    {
        std::ifstream lad(caseDir + "/stage_UsmoothLadder.dat");
        if (!lad) { std::printf("SKIP: no %s/stage_UsmoothLadder.dat\n", caseDir.c_str()); return 77; }
        int n; double a, b;
        while (lad >> n >> a >> b) { ln.push_back(n); lInit.push_back(a); lFinal.push_back(b); }
    }
    if (ln.empty()) { std::printf("SKIP: the ladder file is empty\n"); return 77; }

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
    auto sequentialSweeps = [&](int nSweeps)
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

    // ---- LEG 2: brae's multicolour sweep ------------------------------------------------------------
    DeviceMesh dm = buildDeviceMesh(m, g, patches);
    DeviceBuffer<scalar> dDiag, dUp, dLo, dB;
    dDiag.copyFrom(diag); dUp.copyFrom(upper); dLo.copyFrom(lower); dB.copyFrom(src);
    const DeviceLduView A = deviceLduView(dm, dDiag, dUp, dLo);

    std::printf("  %-3s %-22s %-22s %-22s %s\n", "n", "OpenFOAM", "index order (LEG 1)",
                "brae multicolour", "ratio");
    scalar worstSeq = 0, worstRatio = 0, bestRatio = 1e30;
    for (std::size_t i = 0; i < ln.size(); ++i)
    {
        const int n = ln[i];
        const scalar seq = sequentialSweeps(n);
        DeviceBuffer<scalar> dPsi;
        dPsi.copyFrom(psi0);
        DeviceSolverPerf perf;
        deviceSymGaussSeidel(A, dB, dPsi, normFactor, 0.0, 0.0, n, &perf);
        const scalar ratio = perf.finalResidual / lFinal[i];
        std::printf("  %-3d %-22.15g %-22.15g %-22.15g %.3f\n",
                    n, (double)lFinal[i], (double)seq, (double)perf.finalResidual, (double)ratio);
        worstSeq   = std::fmax(worstSeq, std::fabs(seq - lFinal[i]) / std::fabs(lFinal[i]));
        worstRatio = std::fmax(worstRatio, ratio);
        bestRatio  = std::fmin(bestRatio, ratio);
    }

    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "LEG 1  the index-order transcription IS OpenFOAM's smoother (worst rel %.3e)",
                  (double)worstSeq);
    say(buf, worstSeq < LEG1_TOL);

    std::snprintf(buf, sizeof buf,
                  "LEG 2  the colour order costs at most %.2fx the residual (worst %.2fx)",
                  (double)LADDER_MAX, (double)worstRatio);
    say(buf, worstRatio < LADDER_MAX);

    // The substitution's direction. A ratio below 1 would mean the multicolour sweep BEAT index order on
    // this system -- possible in principle, and a fact this gate must not report as a pass in silence.
    std::snprintf(buf, sizeof buf,
                  "LEG 2  it is behind OpenFOAM at every rung, as a reordering is (best %.2fx)",
                  (double)bestRatio);
    say(buf, bestRatio > 1.0);

    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
