// FUSED LEVEL PRODUCTS (amgFusedLevelProducts, device_amg_vcycle.cu), opt-in and off by default: in the double weighted-Jacobi
// cycle a grid with rows sums each row of A x in a register and applies residualT's and smoothT's expressions to it, where the
// cycle otherwise writes A x out and reads it back. The same sum in the same order, so every value must be the unfused cycle's
// to the bit. No caller in this repository turns it on: this test is its gate. On pitzDaily's Laplacian:
//   (a) PREMISE: vcycleAt's conditions for the fused branch hold where this test says they do: the double cycle
//       (BRAE_AMG_FP32=0), weighted Jacobi (no BRAE_AMG_GS, BRAE_CHEBYSHEV or BRAE_AMG_TSGS), one sweep a side (BRAE_NPRE and
//       BRAE_NPOST unset), plain aggregation (no BRAE_AMG_SA), no correction scaling, no coupled pair, the cycle from zero
//       (neither BRAE_CONTROL_AMG_ZERO_PRODUCT nor BRAE_CONTROL_AMG_FROM_ZERO_STALE), and rows: the finest grid's made for its
//       view (BRAE_AMG_FINE_CSR, on by default) and every coarse grid's (BRAE_AMG_CSR64_BELOW=100000000). The `finest` arm
//       (BRAE_AMG_CSR64_BELOW=0) holds the coarse grids to no rows, where the cycle keeps its two kernels.
//   (b) the cycle started on each grid above the coarsest, launched directly, its x poisoned first: mode 1 mode 0's bits.
//       CONTROL mode 2, deliberately wrong (the post-smooth's row summed from its last entry), must differ from a start grid
//       with rows; from a grid without them and none below (the `finest` arm's coarse grids) it must be mode 0's: the switch
//       acts through the fused branch alone
//   (c) the cycle from the finest grid captured and replayed (amgVCycleApply), a fresh hierarchy for each mode: mode 0 the
//       direct cycle's bits, mode 1 mode 0's, mode 2 not
//   (d) a whole AMG-PCG solve (deviceAMGPCG; on a CUDA 13 build its captured device loop), a fresh hierarchy for each mode:
//       mode 1 the same iterations, residuals and solution bits as mode 0; CONTROL mode 2 another solution
// The settings are set below before the first cycle (each is read once a process), so ctest runs the test once an arm.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_amg.cuh"
#include "device_amg_detail.cuh"   // the sweep counts and the smoother switches the premise reads
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace brae;

namespace {

int failures = 0;

void check(
    bool ok,
    const std::string& what)
{
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// how many values differ in their bits; a size mismatch counts every value
std::size_t bitsDiffer(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    if (a.size() != b.size()) return std::max(a.size(), b.size());
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::memcmp(&a[i], &b[i], sizeof(scalar)) != 0) ++n;
    }
    return n;
}

bool sameBits(
    scalar a,
    scalar b)
{
    return std::memcmp(&a, &b, sizeof(scalar)) == 0;
}

bool allFinite(
    const std::vector<scalar>& a)
{
    for (const scalar v : a)
    {
        if (!std::isfinite(v)) return false;
    }
    return !a.empty();
}

// grid g's view: the caller's on the finest grid, the hierarchy's own below it
DeviceLduView gridView(
    const AMGData& amg,
    int g,
    const DeviceLduView& A)
{
    return g == 0 ? A : amg.level[g - 1].coarseView();
}

// whether grid g holds the rows the fused branch reads (vcycleAt's levelRows): on the finest grid the rows made for this view,
// at its size and with no coupled faces; on a coarse grid the rows amgGatherCsr64 made
bool gridHasRows(
    const AMGData& amg,
    int g,
    const DeviceLduView& A)
{
    if (g == 0)
    {
        return amg.csr64FineKey != nullptr && amg.csr64FineKey == A.ownerStart && amg.csr64FineCells == A.nCells
            && amg.csr64FineFaces == A.nInternalFaces && A.nCyc == 0 && A.nAmi == 0 && !amg.csr64Val.empty()
            && amg.csr64Val[0].size() > 0;
    }
    return g < static_cast<int>(amg.csr64Val.size()) && amg.csr64Val[g].size() > 0;
}

// x = M^-1 b from grid g down, launched directly in the given mode. x is poisoned first (every byte 0xff, a NaN): a value the
// cycle does not write stays NaN, and no cycle's answer is
std::vector<scalar> cycleFrom(
    AMGData& amg,
    int g,
    const DeviceLduView& Ag,
    const DeviceBuffer<scalar>& b,
    int mode)
{
    amgFusedLevelProducts(mode);
    DeviceBuffer<scalar> x(b.size());
    cudaCheck(cudaMemset(x.data(), 0xff, b.size()*sizeof(scalar)), "poison x");
    cudaCheck(cudaDeviceSynchronize(), "poison x");
    vcycleAt(g, amg, Ag, b, x, false);
    cudaCheck(cudaDeviceSynchronize(), "direct cycle");
    return x.host();
}

}   // namespace

int main(
    int argc,
    char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    const std::string arm = argc > 2 ? argv[2] : "all";
    if (arm != "all" && arm != "finest")
    {
        std::printf("usage: test_amg_fused_level_products <case> [all|finest]\n");
        return 2;
    }
    const bool finestOnly = arm == "finest";
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev <= 0)
    {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    setenv("BRAE_AMG_FP32", "0", 1);
    setenv("BRAE_AMG_FINE_CSR", "1", 1);
    setenv("BRAE_AMG_CSR64_BELOW", finestOnly ? "0" : "100000000", 1);
    std::printf("AMG fused level products, rows on %s (%s):\n",
                finestOnly ? "the finest grid alone" : "every grid",
                caseDir.c_str());
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry geo;
    geo.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, geo);
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    GeometricField<scalar> f;
    f.internal.assign(nC, 0.0);
    for (const FvPatch& p : fvp)
    {
        if (p.type == "empty")
        {
            f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        }
        else if (p.type == "wall")
        {
            f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        }
        else
        {
            f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(p, true, 0.0, std::vector<scalar>{}));
        }
    }
    f.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(f, 1.0, m, geo, fvp);
    std::vector<scalar> diagC = M.diag;
    std::vector<scalar> bSrc(nC);
    for (label c = 0; c < nC; ++c) bSrc[c] = geo.V()[c] * std::sin(0.01 * c + 0.4);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label c = fvp[pi].faceCells[i];
            diagC[c] += M.internalCoeffs[pi][i];
            bSrc[c] += M.boundaryCoeffs[pi][i];
        }
    }
    scalar normFactor = 1e-20;
    for (label c = 0; c < nC; ++c) normFactor += std::fabs(bSrc[c]);
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    const std::vector<scalar> magSfInt(geo.magSf().begin(), geo.magSf().begin() + nIf);
    const DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.lower, ownerInt, m.neighbour(), nC);
    const DeviceLduView A = dM.view();
    const DeviceBuffer<scalar> db(bSrc);
    // a hierarchy as the pressure solve has it: built, its coarse matrices made (amgGalerkin gathers the coarse rows), and the
    // finest grid's rows made for A and filled
    const auto hierarchy = [&]()
    {
        AMGData h = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
        amgGalerkin(h, dM.diag, dM.upper, dM.lower);
        amgGatherFineCsr64(h, A);
        return h;
    };

    // (a) the premise
    AMGData H = hierarchy();
    const int G = H.nLevels();
    std::printf("  grids 0..%d (the last solved directly), cells:", G);
    for (int g = 0; g <= G; ++g) std::printf(" %d", g == 0 ? static_cast<int>(nC) : H.level[g - 1].nCoarse);
    std::printf("\n");
    const bool cycle = !amgSinglePrecisionCycle(H) && !H.gsSmooth && !useChebyshev() && !useTSGS() && nPreSweeps() == 1
        && nPostSweeps() == 1 && !H.saSmooth && !H.corrScaling && H.pair.n == 0 && A.nCyc == 0 && A.nAmi == 0;
    const bool fromZero = std::getenv("BRAE_CONTROL_AMG_ZERO_PRODUCT") == nullptr
        && std::getenv("BRAE_CONTROL_AMG_FROM_ZERO_STALE") == nullptr;
    check(cycle && fromZero,
          "(a) premise: the double weighted-Jacobi cycle, one sweep a side, from zero, plain aggregation, no pair");
    check(G >= 2, "(a) premise: at least two grids above the coarsest");
    bool rows = gridHasRows(H, 0, A);
    for (int g = 1; g < G; ++g) rows = rows && gridHasRows(H, g, gridView(H, g, A)) != finestOnly;
    check(rows, finestOnly ? "(a) premise: the finest grid holds its rows and no coarse grid any"
                           : "(a) premise: every grid above the coarsest holds its rows");

    // (b) the cycle from each grid above the coarsest, launched directly, on one hierarchy
    std::vector<scalar> zDirect;
    for (int g = 0; g < G; ++g)
    {
        const DeviceLduView v = gridView(H, g, A);
        std::vector<scalar> rhs(static_cast<std::size_t>(v.nCells));
        for (int c = 0; c < v.nCells; ++c) rhs[c] = std::sin(0.013*c + 0.7*g + 0.3) + 0.25*std::cos(0.0031*c);
        const DeviceBuffer<scalar> b(g == 0 ? bSrc : rhs);
        const std::vector<scalar> z0 = cycleFrom(H, g, v, b, 0);
        const std::vector<scalar> z1 = cycleFrom(H, g, v, b, 1);
        check(amgFusedLevelProductsMode() == 1, "(b) the switch reads back the mode set");
        const std::vector<scalar> z2 = cycleFrom(H, g, v, b, 2);
        if (g == 0) zDirect = z0;
        const std::size_t d1 = bitsDiffer(z1, z0);
        const std::size_t d2 = bitsDiffer(z2, z0);
        const bool fusedBelow = !finestOnly || g == 0;
        std::printf("  grid %d, %d cells: mode 1 differs in %zu values, mode 2 in %zu\n", g, v.nCells, d1, d2);
        const std::string at = "(b) from grid " + std::to_string(g) + ": ";
        check(allFinite(z0) && allFinite(z1) && allFinite(z2), at + "every value of each mode written and finite");
        check(d1 == 0, at + "mode 1 the unfused cycle's bits");
        if (fusedBelow)
        {
            check(d2 > 0, at + "CONTROL mode 2 another cycle (the fused branch ran)");
        }
        else
        {
            check(d2 == 0, at + "no rows here or below: mode 2 the unfused cycle's bits (the switch acts through the branch)");
        }
    }

    // (c) the cycle from the finest grid captured and replayed, a fresh hierarchy for each mode
    std::vector<std::vector<scalar>> zc(3);
    bool replays = true;
    for (int mode = 0; mode < 3; ++mode)
    {
        AMGData F = hierarchy();
        amgFusedLevelProducts(mode);
        DeviceBuffer<scalar> z(static_cast<std::size_t>(nC));
        amgVCycleApply(F, A, db, z, /*captureVcycle*/ true);
        cudaCheck(cudaDeviceSynchronize(), "captured cycle");
        const std::vector<scalar> captured = z.host();
        // the replay must write every value again: the cycle's output buffer (the graph's wA) and z poisoned first
        cudaCheck(cudaMemset(F.wA.data(), 0xff, F.wA.size()*sizeof(scalar)), "poison wA");
        cudaCheck(cudaMemset(z.data(), 0xff, static_cast<std::size_t>(nC)*sizeof(scalar)), "poison z");
        cudaCheck(cudaDeviceSynchronize(), "poison");
        amgVCycleApply(F, A, db, z, /*captureVcycle*/ true);
        cudaCheck(cudaDeviceSynchronize(), "replayed cycle");
        zc[mode] = z.host();
        replays = replays && bitsDiffer(captured, zc[mode]) == 0 && allFinite(zc[mode]);
    }
    std::printf("  captured: mode 1 differs in %zu values, mode 2 in %zu\n", bitsDiffer(zc[1], zc[0]), bitsDiffer(zc[2], zc[0]));
    check(replays, "(c) each replay the bits of its capture, every value finite");
    check(bitsDiffer(zc[0], zDirect) == 0, "(c) captured mode 0 the direct cycle's bits");
    check(bitsDiffer(zc[1], zc[0]) == 0, "(c) captured mode 1 mode 0's bits");
    check(bitsDiffer(zc[2], zc[0]) > 0, "(c) CONTROL captured mode 2 another cycle");

    // (d) a whole AMG-PCG solve, a fresh hierarchy for each mode
    std::vector<DeviceSolverPerf> perf(3);
    std::vector<std::vector<scalar>> xs(3);
    bool keyed = true;
    const scalar tol = 1e-10;
    for (int mode = 0; mode < 3; ++mode)
    {
        AMGData F = hierarchy();
        amgFusedLevelProducts(mode);
        DeviceBuffer<scalar> x(std::vector<scalar>(nC, 0.0));
        perf[mode] = deviceAMGPCG(A, F, db, x, normFactor, tol, 0.0, 30000, /*captureVcycle*/ false);
        cudaCheck(cudaDeviceSynchronize(), "solve");
        xs[mode] = x.host();
        keyed = keyed && gridHasRows(F, 0, A);
    }
    for (int mode = 0; mode < 3; ++mode)
    {
        std::printf("  solve, mode %d: %d iterations, residual %.17g -> %.17g\n", mode, perf[mode].nIterations,
                    perf[mode].initialResidual, perf[mode].finalResidual);
    }
    check(keyed && perf[0].nIterations > 1 && perf[0].finalResidual < tol && allFinite(xs[0]),
          "(d) premise: the solve converged, in more than one iteration, on the finest grid's rows made for its view");
    check(perf[1].nIterations == perf[0].nIterations && sameBits(perf[1].initialResidual, perf[0].initialResidual)
              && sameBits(perf[1].finalResidual, perf[0].finalResidual) && bitsDiffer(xs[1], xs[0]) == 0,
          "(d) mode 1: the same iterations, residuals and solution bits as mode 0");
    check(bitsDiffer(xs[2], xs[0]) > 0, "(d) CONTROL mode 2 another solution");

    amgFusedLevelProducts(0);
    check(amgFusedLevelProductsMode() == 0, "the switch left off");
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
