// AMG V-cycle: the recursive FP64 V-cycle (vcycleAt) + its mixed-precision FP32 mirror (amgCastFP32/vcycleAtF)
// + the public entry points (amgVCycleApply / amgPrepareFP32). Each level: pre-smooth, restrict residual, recurse,
// prolong+scale the correction, post-smooth; the coarsest level dispatches to the coarse solvers. Smoothers
// (ensureSpectrum/chebyshevSmooth/twoStageGSSmooth/gsSweep) + coarse solvers are called across TUs via the AMG
// headers. Verbatim split of device_amg.cu -- no logic change. The PCG drivers (device_amg.cu) call vcycleAt/
// vcycleAtF/amgCastFP32 (external linkage) via local decls there.
#include "device_amg.cuh"          // AMGData / AMGLevel / GridColoring / DeviceLduView / DeviceSolverPerf
#include "device_amg_split.cuh"    // BRAE_AMG_PCG_SPLIT: a lap after every part of a cycle
#include "device_amg_detail.cuh"   // constants (OMEGA/NPRE/NPOST/CHEB_*/SB_*/NCOARSE_CG) + env flags + nBlocks/TPB
#include "device_amg_internal.cuh" // LduF/lduF/cast_/amulF, gsSweep + smoother decls (ensureSpectrum/cheb/tsGS)
#include "device_amg_coarse.cuh"   // deviceCoarsePCG / deviceCoarseJacobiSingleBlock
#include "amg_kernels.cuh"         // zeroT/smoothT/residualT/restrictT/prolongT<T>
#include "device_ldu.cuh"          // DeviceLduView / deviceAmul
#include "device_blas.cuh"         // deviceCopy / deviceDot / deviceReciprocalV
#include <cuda_runtime.h>
#include <cstdio>
#include <limits>
#include <vector>
#include <cstdlib>
#include <stdexcept>

namespace brae {

// The V-cycle options that are only valid for a SYMMETRIC operator. Each names itself and what to set
// instead, and each throws on its own so a message never has to be read as a list. Called by vcycleAt on
// every asymmetric entry with (useChebyshev(), amg.corrScaling, amg.saSmooth).
void amgRefuseAsymmetric(
    bool chebyshev,
    bool corrScaling,
    bool smoothedAggregation)
{
    if (chebyshev)
    {
        // chebyshevSmooth runs the polynomial over [lambdaMax/CHEB_EIGRATIO, CHEB_UPPER_SAFETY*lambdaMax]
        // (device_amg_smoothers.cu:153-190), an interval estimated by a power iteration that converges to a
        // dominant REAL eigenvalue. An asymmetric operator's spectrum is complex, so the interval does not
        // cover it, and an under-covered Chebyshev polynomial AMPLIFIES the modes it misses.
        throw std::runtime_error(
            "brae AMG V-cycle: the Chebyshev smoother (BRAE_CHEBYSHEV) is not valid for an ASYMMETRIC "
            "matrix -- its spectral interval comes from a power iteration that presumes a dominant real "
            "eigenvalue, and an interval that under-covers the spectrum amplifies the modes it misses. "
            "Unset BRAE_CHEBYSHEV (weighted Jacobi, the default) or use BRAE_AMG_GS / BRAE_AMG_TSGS, "
            "which are asymmetric-safe.");
    }
    if (corrScaling)
    {
        // OpenFOAM makes the same refusal by construction: GAMGSolver.C:82 sets
        // scaleCorrection_(matrix.symmetric()), so the scaled correction is off for an asymmetric matrix.
        throw std::runtime_error(
            "brae AMG V-cycle: coarse-correction scaling is not valid for an ASYMMETRIC matrix -- the "
            "line search alpha = (r.c)/(c.Ac) rests on c.Ac being the A-norm of c, which needs a "
            "symmetric A. OpenFOAM disables it the same way (GAMGSolver.C:82, "
            "scaleCorrection_(matrix.symmetric())). Pass corrScaling = false.");
    }
    if (smoothedAggregation)
    {
        throw std::runtime_error(
            "brae AMG V-cycle: smoothed aggregation (BRAE_AMG_SA) is not supported for an ASYMMETRIC "
            "matrix -- its prolongator is smoothed with a SYMMETRIC proxy Laplacian built from the face "
            "weights (device_amg.cu:442-467), and its restriction scatters atomically, so the result is "
            "not reproducible (device_amg.cu:1036-1038). It is untested on an asymmetric operator. "
            "Unset BRAE_AMG_SA (pairwise agglomeration with injection, the default).");
    }
}

namespace {
__global__
void prolongToK(
    int nF,
    const label* __restrict__ map,
    const scalar* __restrict__ xc,
    scalar* __restrict__ pc)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) pc[c] = xc[map[c]];   // prolong INTO pc (not added yet)
}
// Smoothed-aggregation sparse prolongator apply (BRAE_AMG_SA): restrict = P^T, prolong = P. Templated on the
// value type, as the cycles' other work kernels are: T = scalar is the double-precision cycle's kernel as it
// was, T = float the single-precision cycle's, over the prolongator's values cast once (AMGLevel::PvalF).
// restrict: rc += P^T r  (each fine cell f scatters val*r[f] into its coarse columns, atomically).
template <typename T>
__global__
void restrictSparseT(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const T* __restrict__ val,
    const T* __restrict__ r,
    T* __restrict__ rc)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    const T rf = r[f];
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        atomicAdd(&rc[col[k]], val[k]*rf);
}
// prolong (ADD): x[f] += sum_k P[f][k]*xc[col], the smoothed-P twin of prolongT.
template <typename T>
__global__
void prolongSparseT(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const T* __restrict__ val,
    const T* __restrict__ xc,
    T* __restrict__ x)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    T s = T(0);
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        s += val[k]*xc[col[k]];
    x[f] += s;
}
// prolong (SET): pc[f] = sum_k P[f][k]*xc[col], the corrScaling line-search path (writes pc, not added yet).
__global__
void prolongToSparseK(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const scalar* __restrict__ val,
    const scalar* __restrict__ xc,
    scalar* __restrict__ pc)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    scalar s = 0.0;
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        s += val[k]*xc[col[k]];
    pc[f] = s;
}

// Energy-minimising coarse-correction scale (OF GAMG): alpha = (r . c)/(c . Ac), guarded since c.Ac = ||c||_A^2 >= 0.
__global__
void scaleFactorK(
    const scalar* __restrict__ num,
    const scalar* __restrict__ den,
    scalar* __restrict__ out)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar d = *den;
        *out = (d > 1e-30) ? (*num)/d : 1.0;
    }
}
} // anon

bool amgSinglePrecisionCycle(const AMGData& amg)
{
    // BRAE_CONTROL_AMG_SA_DOUBLE=1: a smoothed-aggregation hierarchy keeps the double-precision cycle, as it
    // did before the single-precision one had the sparse prolongator's transfers.
    static const bool saDouble = std::getenv("BRAE_CONTROL_AMG_SA_DOUBLE") != nullptr;
    if (amg.saSmooth && saDouble) return false;
    // A two-stage Gauss-Seidel asked for (BRAE_AMG_TSGS) is run where it exists, in the double-precision cycle.
    // It was SILENTLY IGNORED: the single-precision cycle was chosen all the same and has weighted Jacobi only,
    // so an arm with the switch measured Jacobi under another name -- on RAS/DTCHull, 1,072 iterations with it
    // and 1,072 without (853 once the cycle was forced to double precision beside it).
    return useFP32() && !amg.gsSmooth && !useChebyshev() && !useTSGS();
}

// Recursive V-cycle at grid g: x_g <- M^-1 b_g (x_g overwritten). g==nLevels is the coarsest grid (an approximate
// solve; single-block or fused-cluster when small); above it: pre-smooth, restrict the residual to g+1, recurse,
// prolong the correction, post-smooth.
void vcycleAt(
    int g,
    AMGData& amg,
    const DeviceLduView& Ag,
    const DeviceBuffer<scalar>& bg,
    DeviceBuffer<scalar>& xg,
    bool asymmetric)
{
    const int n = Ag.nCells;
    if (asymmetric) amgRefuseAsymmetric(useChebyshev(), amg.corrScaling, amg.saSmooth);
    // x starts at zero. A grid whose pre-smooth is weighted Jacobi writes its first sweep straight from b
    // (smoothFromZeroT: the sweep's own expression with the two zeros written in, so its result to the bit) and
    // needs neither the zeroing nor the product of a zero vector -- what vcycleAtF has done since the pressure
    // work. This cycle zeroed x and formed A*0 at every grid of every iteration: MEASURED 2026-10-04 with
    // BRAE_AMG_PCG_SPLIT on pcorr's smoothed hierarchy, waveMakerPiston at 896,000 cells, the pre-smooth of the
    // finest grid 454 us an iteration where its post-smooth, the same sweep, is 403.
    // BRAE_CONTROL_AMG_ZERO_PRODUCT=1 forms both, as before. BRAE_CONTROL_AMG_FROM_ZERO_STALE=1 is a gate's
    // CONTROL, deliberately wrong: neither the zeroing nor the from-zero sweep, so the first sweep starts from
    // whatever the last cycle left in x.
    static const bool zeroProduct = std::getenv("BRAE_CONTROL_AMG_ZERO_PRODUCT") != nullptr;
    static const bool staleStart = std::getenv("BRAE_CONTROL_AMG_FROM_ZERO_STALE") != nullptr;
    const bool jacobiPre = !useChebyshev() && !(asymmetric ? useTSGSAsym() : useTSGS()) && !amg.gsSmooth;
    const bool fromZero = !zeroProduct && g != amg.nLevels() && jacobiPre && nPreSweeps() > 0;
    if (!fromZero) zeroT<scalar><<<nBlocks(n),TPB>>>(n, xg.data());
    if (g == amg.nLevels())                                    // coarsest: approximate solve
    {
        // BRAE_NCOARSE_CG overrides the coarsest PCG iteration CAP (item 80: it converges, it does not count).
        static const int ncoarseCG = [](){ const char* e = std::getenv("BRAE_NCOARSE_CG"); return (e && std::atoi(e) > 0) ? std::atoi(e) : NCOARSE_CG; }();
        // The asymmetric coarsest solve iterates to COARSE_REL_TOL and this is only its CAP -- see the
        // constant's note: a fixed count there breaks the outer Krylov method.
        static const int ncoarseAsymCap = [](){ const char* e = std::getenv("BRAE_NCOARSE_CG"); return (e && std::atoi(e) > 0) ? std::atoi(e) : NCOARSE_ASYM_CAP; }();
        // The coarsest solve is the ONE part of the V-cycle that a nonsymmetric operator invalidates:
        // deviceCoarsePCG is a conjugate gradient, whose step length presumes p.Ap is an A-norm. Its
        // asymmetric twin is a BiCGStab, which is what OpenFOAM builds for an asymmetric coarsest level
        // (GAMGSolver.C:299-328). Every other branch below is weighted Jacobi and needs no distinction.
        // The DIRECT coarsest solve, when amgGalerkin factorised this grid (BRAE_AMG_COARSE_LU; the
        // test is against the grid size so a hierarchy whose coarsest changed, or a caller that never
        // ran a Galerkin update, falls through to the iterative solvers below). Exact for both the
        // symmetric and the asymmetric operator, so it precedes the CG/BiCGStab split.
        if (amg.coarseLUn == n) deviceCoarseLUSolve(n, amg.coarseLU, amg.coarsePiv, bg, xg);
        else if (n <= SB_CG_MAX && asymmetric) deviceCoarseBiCGStab(Ag, bg, xg, ncoarseAsymCap);
        else if (n <= SB_CG_MAX) deviceCoarsePCG(Ag, bg, xg, ncoarseCG);            // tiny coarsest: single-block Jacobi-PCG (cheap+accurate)
        else if (n <= SB_MAX) deviceCoarseJacobiSingleBlock(Ag, bg, xg, NCOARSE);   // larger: single-block many-sweep Jacobi
        else if (deviceCoarseFitsCluster(n) && n <= COARSE_FUSE_MAX) deviceCoarseJacobiFused(Ag, bg, xg, NCOARSE);
        else for (int s = 0; s < NCOARSE; ++s)
        {
            deviceAmul(Ag, xg, amg.vAx[g]);
            smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
        }
        amgSplit::lap(g, amgSplit::coarsest);
        return;
    }
    if (useChebyshev()) chebyshevSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], amg.lambdaMax[g], chebDeg());  // pre-smooth (x=0)
    else if (asymmetric ? useTSGSAsym() : useTSGS()) twoStageGSSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], nPreSweeps(), tsgsOrder(), true);     // OF v2606 twoStageGaussSeidel (fwd)
    else if (amg.gsSmooth) for (int s = 0; s < nPreSweeps(); ++s) gsSweep(Ag, bg, xg, amg.coloring[g], true);    // forward GS
    else for (int s = 0; s < nPreSweeps(); ++s)
    {
        if (s == 0 && fromZero && !staleStart)
        {
            smoothFromZeroT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), Ag.diag, xg.data());
            continue;
        }
        deviceAmul(Ag, xg, amg.vAx[g]);
        smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
    }
    amgSplit::lap(g, amgSplit::smoothPre);
    deviceAmul(Ag, xg, amg.vAx[g]);
    residualT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), amg.vR[g].data());
    amgSplit::lap(g, amgSplit::residual);
    const int nc = amg.level[g].nCoarse;
    const AMGLevel& Lg = amg.level[g];
    if (amg.saSmooth)                                            // restrict rc = P^T r (sparse smoothed prolongator)
    {
        // Sparse-prolongator restriction still scatters; the SA path is opt-in and remains nondeterministic.
        zeroT<scalar><<<nBlocks(nc),TPB>>>(nc, amg.vB[g+1].data());
        restrictSparseT<scalar><<<nBlocks(n),TPB>>>(
            n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vR[g].data(), amg.vB[g+1].data());
    }
    else                                          // fixed-order gather; writes rc, so no pre-zero needed
        restrictGatherT<scalar><<<nBlocks(nc),TPB>>>(
            nc, Lg.galCellStart.data(), Lg.galCellList.data(), amg.vR[g].data(), amg.vB[g+1].data());
    amgSplit::lap(g, amgSplit::restriction);
    vcycleAt(g+1, amg, amg.level[g].coarseView(), amg.vB[g+1], amg.vX[g+1], asymmetric);   // recurse to the next coarser grid
    if (amg.corrScaling)
    {
        // OF-GAMG-style scaled coarse correction: alpha = (r . A pc)/(A pc . A pc); xg += alpha*pc. The single
        // line-search per level fixes the magnitude of the prolongation. All scalars stay on the device
        // (deviceDotInto/scaleFactorK/AxpyDev) so it adds NO host sync and remains capturable into the V-cycle graph.
        if (amg.saSmooth) prolongToSparseK<<<nBlocks(n),TPB>>>(n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vX[g+1].data(), amg.vPc[g].data());
        else              prolongToK<<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vX[g+1].data(), amg.vPc[g].data());  // c = P*corr
        deviceAmul(Ag, amg.vPc[g], amg.vAx[g]);                                       // Ac
        deviceDotInto(amg.vR[g], amg.vPc[g], amg.sScNum.data());                      // r . c   (energy-min, OF GAMG)
        deviceDotInto(amg.vPc[g], amg.vAx[g], amg.sScDen.data());                     // c . Ac  (= ||c||_A^2 > 0)
        scaleFactorK<<<1,1>>>(amg.sScNum.data(), amg.sScDen.data(), amg.sScAlpha.data());
        deviceAxpyDev(amg.sScAlpha.data(), amg.vPc[g], xg);                           // xg += alpha * c
    }
    else if (amg.saSmooth)
        prolongSparseT<scalar><<<nBlocks(n),TPB>>>(
            n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vX[g+1].data(), xg.data());
    else
        prolongT<scalar><<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vX[g+1].data(), xg.data());
    amgSplit::lap(g, amgSplit::prolongation);
    if (useChebyshev()) chebyshevSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], amg.lambdaMax[g], chebDeg());  // post-smooth
    else if (asymmetric ? useTSGSAsym() : useTSGS()) twoStageGSSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], nPostSweeps(), tsgsOrder(), false);  // twoStageGaussSeidel (bwd -> symmetric)
    else if (amg.gsSmooth) for (int s = 0; s < nPostSweeps(); ++s) gsSweep(Ag, bg, xg, amg.coloring[g], false);  // backward GS (symmetric V-cycle)
    else for (int s = 0; s < nPostSweeps(); ++s)
    {
        deviceAmul(Ag, xg, amg.vAx[g]);
        smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
    }
    amgSplit::lap(g, amgSplit::smoothPost);
}

// The SYMMETRIC entry point, unchanged for every existing caller (device_amg.cu and device_amg_pcg.cu
// declare this 5-argument form themselves). It is a separate overload rather than a default argument on
// the function above because those local declarations would otherwise make their calls ambiguous.
void vcycleAt(
    int g,
    AMGData& amg,
    const DeviceLduView& Ag,
    const DeviceBuffer<scalar>& bg,
    DeviceBuffer<scalar>& xg)
{
    vcycleAt(g, amg, Ag, bg, xg, false);
}

// Mixed-precision (FP32) V-cycle: a mirror of the default V-cycle (weighted-Jacobi + map restrict/prolong) with the
// matrix values and work vectors in FP32 (half the bytes on the BW-bound SpMV+smooth); the topology (labels) is
// shared with the FP64 path. The coarsest level casts back to FP64 and reuses the exact FP64 coarse solve. Only the
// default smoother is supported here (GS/Chebyshev/corrScaling stay FP64: amgSinglePrecisionCycle and its callers).
// It reuses the templated zeroT/smoothT/residualT/restrictT/prolongT<float>; only the casts and the FP32 SpMV (amulF)
// are FP32-specific.
// A SMOOTHED-AGGREGATION hierarchy runs here too (2026-10-04): its coarse matrices are LDU views like any other
// level's and are cast with them; its transfers are restrictSparseT/prolongSparseT<float> over PvalF. It ran the
// double-precision cycle until then: MEASURED with BRAE_AMG_PCG_SPLIT on pcorr, waveMakerPiston at 896,000
// cells, us an iteration on the finest grid, double against this cycle's on the same mesh: post-smooth 405 / 222,
// residual 359 / 215.
// (amulFK/amulF -- the FP32 SpMV -- now live in device_amg_internal.cuh, shared with the FP32 GS solver.)
// ---- FP-12: the coarse operator as contiguous rows -------------------------------------------
//
// The FP32 SpMV reads its off-diagonals through two indirections -- upper[f] with psi[nei[f]] over the
// cell's owner faces, then lower[f] with psi[owner[f]] over losort -- into arrays ordered by FACE, not
// by cell. On the fine grid that is bandwidth-bound work and the order does not matter; on a coarse
// grid it is a few hundred threads chasing scattered addresses with nothing to hide the latency, and it
// shows: 4.2 us on a level under 1,000 cells where an elementwise kernel on the same grid takes 0.76,
// and 2.5 of the pressure phase's 6.6 ms per iteration across the hierarchy (nsys on
// gasMixing/injectorPipe, 74,650 cells, 11 levels, ~15 V-cycles per iteration).
//
// So each grid below a threshold gets its rows laid out contiguously: entry i of row c is a (value,
// column) pair, the cell's owner faces in ownerStart order first and then its neighbour faces in losort
// order -- the exact sequence amulFK sums in. Same terms, same order, same bits (amulCsrFK).
//
// The values are refilled once per solve from the FP64 face arrays (csrGatherValsK), which REPLACES the
// two cast_ launches that grid's upper and lower would otherwise need, so the layout costs no extra
// launch. The structure is built once per hierarchy: agglomeration is static for the life of the mesh.
//
// BRAE_AMG_CSR=0 restores the face form everywhere; BRAE_AMG_CSR_BELOW moves the threshold.
namespace {

int amgCsrOn()
{
    static const int on = []()
    {
        const char* e = std::getenv("BRAE_AMG_CSR");
        return (e && std::atoi(e) == 0) ? 0 : 1;
    }();
    return on;
}
// EVERY grid by default, the fine one included. Swept on gasMixing/injectorPipe (74,650 cells), p solve
// in ms per iteration: 4.3 with the face form everywhere, then 4.3 / 4.2 / 4.0 / 3.9 / 3.8 as the
// threshold rises through 512 / 2,048 / 8,192 / 16,384 / every grid. The fine grid gains too -- its
// owner half is already sequential in the face arrays but its neighbour half is read through losort,
// and the row layout makes both contiguous.
int amgCsrBelow()
{
    static const int n = []()
    {
        const char* e = std::getenv("BRAE_AMG_CSR_BELOW");
        return (e && std::atoi(e) > 0) ? std::atoi(e) : std::numeric_limits<int>::max();
    }();
    return n;
}

// label arrays come back from wherever they live -- a DeviceBuffer for a coarse grid, a raw device
// pointer for the fine one.
std::vector<label> pullLabels(const label* d, int n)
{
    std::vector<label> h(n);
    if (n > 0) cudaCheck(cudaMemcpy(h.data(), d, n*sizeof(label), cudaMemcpyDeviceToHost), "csr pull");
    return h;
}

// One grid's rows, built on the host from its addressing. Returns false when the grid keeps the face
// form (too large to gain, or nothing to lay out).
bool buildCsrForGrid(AMGData& amg, int g, const LduF& A)
{
    if (A.nCells <= 0 || A.nInternalFaces <= 0) return false;
    if (A.nCells > amgCsrBelow()) return false;

    const std::vector<label> ownerStart = pullLabels(A.ownerStart, A.nCells + 1);
    const std::vector<label> losortStart = pullLabels(A.losortStart, A.nCells + 1);
    const std::vector<label> losort = pullLabels(A.losort, A.nInternalFaces);
    const std::vector<label> nei = pullLabels(A.nei, A.nInternalFaces);
    const std::vector<label> owner = pullLabels(A.owner, A.nInternalFaces);

    std::vector<label> row(A.nCells + 1, 0), col, src;
    col.reserve(2*A.nInternalFaces);
    src.reserve(2*A.nInternalFaces);
    for (int c = 0; c < A.nCells; ++c)
    {
        row[c] = static_cast<label>(col.size());
        for (label f = ownerStart[c]; f < ownerStart[c+1]; ++f)      // upper[f] * psi[nei[f]]
        {
            col.push_back(nei[f]);
            src.push_back(f);
        }
        for (label k = losortStart[c]; k < losortStart[c+1]; ++k)    // lower[f] * psi[owner[f]]
        {
            const label f = losort[k];
            col.push_back(owner[f]);
            src.push_back(-f - 1);
        }
    }
    row[A.nCells] = static_cast<label>(col.size());

    amg.csrRow[g].copyFrom(row);
    amg.csrCol[g].copyFrom(col);
    amg.csrSrc[g].copyFrom(src);
    amg.csrVal[g].resize(col.size());
    return true;
}

}   // namespace

// Builds the CSR mirror for every grid that qualifies. Called once, from amgCastFP32, after the FP32
// buffers exist; the agglomeration behind it is static for the life of the mesh.
void amgBuildCsrFP32(AMGData& amg, const DeviceLduView& A)
{
    const int G = amg.nLevels();
    amg.csrRow.resize(G+1);
    amg.csrCol.resize(G+1);
    amg.csrSrc.resize(G+1);
    amg.csrVal.resize(G+1);
    for (int g = 0; g <= G; ++g)
    {
        const DeviceLduView v = (g==0) ? A : amg.level[g-1].coarseView();
        const LduF f = lduF(v, amg.fDiag[g], amg.fUpper[g], amg.fLower[g]);
        buildCsrForGrid(amg, g, f);
    }
    amg.csrBuilt = true;
}

// Cast the (current) FP64 fine + coarse matrices to their FP32 mirrors. Allocates the mirrors + FP32 work vectors once.
void amgCastFP32(
    AMGData& amg,
    const DeviceLduView& A)
{
    const int G = amg.nLevels();
    if (!amg.fp32Alloc)
    {
        amg.fDiag.resize(G+1);
        amg.fUpper.resize(G+1);
        amg.fLower.resize(G+1);
        amg.vAxF.resize(G+1);
        amg.vRF.resize(G+1);
        amg.vXF.resize(G+1);
        amg.vBF.resize(G+1);
        for (int g=0; g<=G; ++g)
        {
            const DeviceLduView v = (g==0) ? A : amg.level[g-1].coarseView();
            amg.fDiag[g].resize(v.nCells);
            amg.fUpper[g].resize(v.nInternalFaces);
            amg.fLower[g].resize(v.nInternalFaces);
            amg.vAxF[g].resize(v.nCells);
            amg.vRF[g].resize(v.nCells);
            amg.vXF[g].resize(v.nCells);
            amg.vBF[g].resize(v.nCells);
        }
        if (amg.saSmooth)
        {
            // the sparse prolongator's values, once: they are fixed for the life of the hierarchy.
            // BRAE_CONTROL_AMG_SA_P_NOT_CAST=1 is a gate's CONTROL, deliberately wrong: the single-precision
            // values are left zero, so the cycle transfers nothing and has no coarse correction.
            static const bool notCast = std::getenv("BRAE_CONTROL_AMG_SA_P_NOT_CAST") != nullptr;
            for (int g = 0; g < G; ++g)
            {
                AMGLevel& L = amg.level[g];
                const int nnz = static_cast<int>(L.Pval.size());
                L.PvalF.resize(L.Pval.size());
                if (nnz == 0) continue;
                if (notCast)
                {
                    zeroT<float><<<nBlocks(nnz),TPB>>>(nnz, L.PvalF.data());
                    continue;
                }
                cast_<scalar,float><<<nBlocks(nnz),TPB>>>(nnz, L.Pval.data(), L.PvalF.data());
            }
        }
        amg.fp32Alloc = true;
    }
    if (amgCsrOn() && !amg.csrBuilt) amgBuildCsrFP32(amg, A);       // FP-12, once per hierarchy
    for (int g=0; g<=G; ++g)
    {
        const DeviceLduView v = (g==0) ? A : amg.level[g-1].coarseView();
        cast_<scalar,float><<<nBlocks(v.nCells),TPB>>>(v.nCells, v.diag, amg.fDiag[g].data());
        if (v.nInternalFaces>0)
        {
            // A CSR grid takes its off-diagonals through the row layout instead, in ONE launch where
            // the face form needs two -- the same values, gathered into the order the SpMV reads them.
            const int nnz = amgCsrOn() ? static_cast<int>(amg.csrVal[g].size()) : 0;
            if (nnz > 0)
            {
                csrGatherValsK<<<nBlocks(nnz),TPB>>>(nnz, amg.csrSrc[g].data(), v.upper, v.lower,
                                                     amg.csrVal[g].data());
            }
            else
            {
                cast_<scalar,float><<<nBlocks(v.nInternalFaces),TPB>>>(v.nInternalFaces, v.upper, amg.fUpper[g].data());
                cast_<scalar,float><<<nBlocks(v.nInternalFaces),TPB>>>(v.nInternalFaces, v.lower, amg.fLower[g].data());
            }
        }
    }
}
// FP32 recursive V-cycle. topoG = grid g's FP64 view (topology + the FP64 coarsest solve); Ag = grid g's FP32 matrix.
// FP-12 TRIED FUSING THE COARSE HIERARCHY INTO ONE KERNEL AND IT IS SLOWER. The idea was the one that
// worked for device_dilu.cu's level walk: from the first level small enough for a single block, do the
// whole rest of the V-cycle -- down, the coarsest LU solve, and back up -- in ONE kernel with
// __syncthreads where the launches were. It was written, and it is exactly bit-identical (every
// residual line over 100 iterations of gasMixing/injectorPipe matched the separate-launch path). It
// was also measured, and every threshold lost: fusing below 64 / 128 / 256 / 512 / 1024 / 2048 cells
// gives a pressure phase of 6.7 / 6.7 / 6.9 / 6.9 / 7.1 / 7.0 ms per iteration against 6.5 for the
// separate launches.
//
// WHY, and this is the part worth keeping: these kernels are ALREADY graph nodes, so the node-to-node
// overhead is not what the V-cycle is paying. Fusing below 64 cells removes a dozen node boundaries on
// levels where one block of 256 threads is more than enough parallelism, and it still gained nothing --
// so the boundaries are cheap. What the coarse levels actually pay is the DEVICE-SIDE duration of a
// scattered indirect gather: the FP32 SpMV takes 4.2 us on a level under 1,000 cells where zeroT on the
// same grid takes 0.76, because owner/losort indirection with few threads has no parallelism to hide
// the latency. One block makes that strictly worse. The lever that remains is the coarse operator's
// LAYOUT -- a per-level CSR whose inner loop is one contiguous read instead of two indirections -- not
// the launch structure.
// The FP32 SpMV for grid g: the contiguous-row form where that grid has one, the face form otherwise.
// Same sum, same order, same bits either way (see amgBuildCsrFP32).
static inline void amgSpmvF(AMGData& amg, int g, const LduF& Ag, const float* x, float* y)
{
    if (!amg.csrVal.empty() && amg.csrVal[g].size() > 0)
    {
        amulCsrFK<<<nBlocks(Ag.nCells),TPB>>>(Ag.nCells, Ag.diag, amg.csrRow[g].data(),
                                              amg.csrCol[g].data(), amg.csrVal[g].data(), x, y);
        cudaCheck(cudaGetLastError(), "amulCsrF");
        return;
    }
    amulF(Ag, x, y);
}

void vcycleAtF(
    int g,
    AMGData& amg,
    const DeviceLduView& topoG,
    const LduF& Ag,
    const float* bg,
    float* xg,
    bool asymmetric)
{
    const int n = Ag.nCells;
    if (asymmetric) amgRefuseAsymmetric(useChebyshev(), amg.corrScaling, amg.saSmooth);
    // x starts at zero. A level that pre-smooths writes its first sweep straight from b (smoothFromZeroT) and
    // needs neither this nor the product of a zero vector; BRAE_CONTROL_AMG_ZERO_PRODUCT=1 forms both, as before.
    // BRAE_CONTROL_AMG_FROM_ZERO_STALE=1: the deliberately wrong control vcycleAt describes.
    static const bool zeroProduct = std::getenv("BRAE_CONTROL_AMG_ZERO_PRODUCT") != nullptr;
    static const bool staleStart = std::getenv("BRAE_CONTROL_AMG_FROM_ZERO_STALE") != nullptr;
    const bool fromZero = !zeroProduct && g != amg.nLevels() && nPreSweeps() > 0;
    if (!fromZero) zeroT<float><<<nBlocks(n),TPB>>>(n, xg);
    if (g == amg.nLevels())                                    // coarsest: cast to FP64, exact FP64 solve, cast back
    {
        cast_<float,scalar><<<nBlocks(n),TPB>>>(n, bg, amg.vB[g].data());
        static const int ncoarseCG = [](){ const char* e=std::getenv("BRAE_NCOARSE_CG"); return (e&&std::atoi(e)>0)?std::atoi(e):NCOARSE_CG; }();
        // The asymmetric coarsest solve iterates to COARSE_REL_TOL and this is only its CAP.
        static const int ncoarseAsymCap = [](){ const char* e=std::getenv("BRAE_NCOARSE_CG"); return (e&&std::atoi(e)>0)?std::atoi(e):NCOARSE_ASYM_CAP; }();
        // Same coarsest split as the FP64 V-cycle: CG is not a solve on a nonsymmetric operator.
        if (amg.coarseLUn == n) deviceCoarseLUSolve(n, amg.coarseLU, amg.coarsePiv, amg.vB[g], amg.vX[g]);
        else if (n <= SB_CG_MAX && asymmetric) deviceCoarseBiCGStab(topoG, amg.vB[g], amg.vX[g], ncoarseAsymCap);
        else if (n <= SB_CG_MAX) deviceCoarsePCG(topoG, amg.vB[g], amg.vX[g], ncoarseCG);
        else
        {
            zeroT<scalar><<<nBlocks(n),TPB>>>(n, amg.vX[g].data());
            for (int s=0;s<NCOARSE;++s)
            {
                deviceAmul(topoG, amg.vX[g], amg.vAx[g]);
                smoothT<scalar><<<nBlocks(n),TPB>>>(n, amg.vB[g].data(), amg.vAx[g].data(), topoG.diag, amg.vX[g].data());
            }
        }
        cast_<scalar,float><<<nBlocks(n),TPB>>>(n, amg.vX[g].data(), xg);
        amgSplit::lap(g, amgSplit::coarsest);
        return;
    }
    for (int s=0; s<nPreSweeps(); ++s)     // BRAE_NPRE, as the FP64 cycle reads it; NPRE by default
    {
        if (s == 0 && fromZero && !staleStart)
        {
            smoothFromZeroT<float><<<nBlocks(n),TPB>>>(n, bg, Ag.diag, xg);
            continue;
        }
        amgSpmvF(amg, g, Ag, xg, amg.vAxF[g].data());
        smoothT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), Ag.diag, xg);
    }
    amgSplit::lap(g, amgSplit::smoothPre);
    amgSpmvF(amg, g, Ag, xg, amg.vAxF[g].data());
    residualT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), amg.vRF[g].data());
    amgSplit::lap(g, amgSplit::residual);
    const AMGLevel& Lg = amg.level[g];
    const int nc = Lg.nCoarse;
    if (amg.saSmooth)                                            // rc = P^T r, the sparse smoothed prolongator
    {
        zeroT<float><<<nBlocks(nc),TPB>>>(nc, amg.vBF[g+1].data());
        restrictSparseT<float><<<nBlocks(n),TPB>>>(
            n, Lg.Prow.data(), Lg.Pcol.data(), Lg.PvalF.data(), amg.vRF[g].data(), amg.vBF[g+1].data());
    }
    else
    {
        restrictGatherT<float><<<nBlocks(nc),TPB>>>(
            nc, Lg.galCellStart.data(), Lg.galCellList.data(), amg.vRF[g].data(), amg.vBF[g+1].data());
    }
    amgSplit::lap(g, amgSplit::restriction);
    const DeviceLduView topoC = Lg.coarseView();
    const LduF Ac = lduF(topoC, amg.fDiag[g+1], amg.fUpper[g+1], amg.fLower[g+1]);
    vcycleAtF(g+1, amg, topoC, Ac, amg.vBF[g+1].data(), amg.vXF[g+1].data(), asymmetric);
    if (amg.saSmooth)
    {
        prolongSparseT<float><<<nBlocks(n),TPB>>>(
            n, Lg.Prow.data(), Lg.Pcol.data(), Lg.PvalF.data(), amg.vXF[g+1].data(), xg);
    }
    else
    {
        prolongT<float><<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vXF[g+1].data(), xg);
    }
    amgSplit::lap(g, amgSplit::prolongation);
    for (int s=0; s<nPostSweeps(); ++s)    // BRAE_NPOST; NPOST by default
    {
        amgSpmvF(amg, g, Ag, xg, amg.vAxF[g].data());
        smoothT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), Ag.diag, xg);
    }
    amgSplit::lap(g, amgSplit::smoothPost);
}

// The SYMMETRIC FP32 entry point, unchanged for its existing callers (same overload-not-default reason
// as vcycleAt above).
void vcycleAtF(
    int g,
    AMGData& amg,
    const DeviceLduView& topoG,
    const LduF& Ag,
    const float* bg,
    float* xg)
{
    vcycleAtF(g, amg, topoG, Ag, bg, xg, false);
}

// amgVCycleApply runs the FP32 V-cycle automatically -- the same mixed-precision path the single-GPU deviceAMGPCG
// uses, now available to the distributed Krylov (the pressure V-cycle's SpMV+smoother are bandwidth-bound, so
// FP32 halves their bytes ~= 2x, while the r->FP32/FP32->z casts and the FP64 coarsest solve keep it accurate).
void amgPrepareFP32(AMGData& amg, const DeviceLduView& A)
{
    if (amgSinglePrecisionCycle(amg)) amgCastFP32(amg, A);
}

// z = M^-1 r : ONE symmetric AMG V-cycle applied as a PRECONDITIONER, factored out of deviceAMGPCG so the
// DISTRIBUTED Krylov (device_pcg.cu) can precondition each rank's LOCAL block with AMG. The V-cycle is built on
// internal faces only, so it omits the processor-interface coupling -- exactly the block-Jacobi / additive-Schwarz
// design: the outer distributed matvec (deviceParallelAmul) supplies the interface, the local V-cycle need only
// approximate the local block. amg must be built (buildAMG) and current (amgGalerkin gives it this step's coarse
// operators). Runs the FP32 V-cycle when amgPrepareFP32 cast the matrices this solve, else the FP64 one.
void amgVCycleApply(AMGData& amg, const DeviceLduView& A,
                    const DeviceBuffer<scalar>& r, DeviceBuffer<scalar>& z, bool captureVcycle)
{
    ensureSpectrum(amg, A);                // one-time Chebyshev spectrum estimate (no-op unless the Chebyshev smoother is on)
    const bool fp32 = amg.fp32Alloc && amgSinglePrecisionCycle(amg);
    const int nC = A.nCells;

    if (!captureVcycle)
    {
        // DIRECT launch (default): every V-cycle kernel is a fresh launch.
        if (fp32)
        {
            const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);         // grid-0 FP32 matrix view
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, r.data(), amg.vBF[0].data());   // r -> FP32
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());              // FP32 V-cycle
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), z.data());   // FP32 -> z (FP64)
        }
        else vcycleAt(0, amg, A, r, z);    // FP64 V-cycle
        return;
    }

    // GRAPH REPLAY: the V-cycle is host-scalar-free and runs on FIXED buffers (amg.rA/wA), so capture it once
    // (keyed on A.diag; the coarse VALUES change each step but the graph references the buffers, read at replay) and
    // replay -- removing the launch overhead of the V-cycle's many small kernels (13 levels x SpMV+smoother per PCG
    // iter, the dominant per-iteration launch cost). Copy r -> persistent amg.rA, replay, amg.wA -> z. Same
    // capture/replay the single-GPU deviceAMGPCG uses; now available to the distributed pressure solve.
    deviceCopy(amg.rA, r);
    if (fp32)
    {
        const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);
        AMGGraphCache& gcf = *amg.gcacheF;
        if (!gcf.exec || gcf.key != A.diag || gcf.keyEpoch != deviceReductionScratchEpoch()
            || gcf.keyAddressingId != A.addressingId)
        {
            if (gcf.exec)  { cudaGraphExecDestroy(gcf.exec);  gcf.exec  = nullptr; }
            if (gcf.graph) { cudaGraphDestroy(gcf.graph);     gcf.graph = nullptr; }
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amgF capture begin");
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, amg.rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), amg.wA.data());
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gcf.graph), "amgF capture end");
            cudaCheck(cudaGraphInstantiate(&gcf.exec, gcf.graph, 0), "amgF graph instantiate");
            gcf.key = A.diag;
            gcf.keyEpoch = deviceReductionScratchEpoch();
            gcf.keyAddressingId = A.addressingId;
        }
        cudaCheck(cudaGraphLaunch(gcf.exec, cudaStreamPerThread), "amgF graph launch");
    }
    else
    {
        AMGGraphCache& gc = *amg.gcache;
        if (!gc.exec || gc.key != A.diag || gc.keyEpoch != deviceReductionScratchEpoch()
            || gc.keyAddressingId != A.addressingId)
        {
            if (gc.exec)  { cudaGraphExecDestroy(gc.exec);  gc.exec  = nullptr; }
            if (gc.graph) { cudaGraphDestroy(gc.graph);     gc.graph = nullptr; }
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amg capture begin");
            vcycleAt(0, amg, A, amg.rA, amg.wA);
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gc.graph), "amg capture end");
            cudaCheck(cudaGraphInstantiate(&gc.exec, gc.graph, 0), "amg graph instantiate");
            gc.key = A.diag;
            gc.keyEpoch = deviceReductionScratchEpoch();
            gc.keyAddressingId = A.addressingId;
        }
        cudaCheck(cudaGraphLaunch(gc.exec, cudaStreamPerThread), "amg graph launch");
    }
    deviceCopy(z, amg.wA);
}

} // namespace brae
