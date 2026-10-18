// ONE COARSE OFF-DIAGONAL FOR A FINE MATRIX HANDED WITH ONE (amgGalerkin, AMGLevel::coarseLower): upper and lower
// the same buffer, as simpleFoam's symmetric pressure and interFoam's pcorr hand them. Every injection level then
// keeps cUpper alone and hands it as the lower. Gated on pitzDaily's Laplacian:
//   (a) the premise: that hierarchy holds no cLower on any level, and one fed two buffers of the same values holds
//       one on every level;
//   (b) their coarse matrices are the same bits: cDiag, cUpper against cUpper, and cUpper against cLower;
//   (c) an AMG-PCG solve with its V-cycle captured gives the same bits and iterations on each;
//   (d) THE SWITCH: the one-off-diagonal hierarchy, its V-cycle captured, is handed a matrix with two different
//       off-diagonals. It takes its cLower and drops its graphs, and its next captured cycle is a fresh
//       hierarchy's bit for bit. CONTROL: BRAE_CONTROL_AMG_LOWER_GRAPHS_KEPT=1 keeps the graphs, and that cycle
//       must then differ -- the replay reads cUpper as the lower.
//   (e) THE WHOLE-LOOP PCG GRAPH's copy (deviceAMGPCGGraph, CUDA 13 builds): handed one off-diagonal it copies the
//       upper alone and its graph reads that copy as the lower. On one hierarchy, the fine view's lower moves from
//       its own buffer to the upper and back: each second solve is a fresh solver's of that view, bit for bit.
//       CONTROL: BRAE_CONTROL_PCG_GRAPH_LOWER_UNKEYED=1 leaves the lower out of the graph's key, and each second
//       solve must then differ -- the replay reads the other view's lower.
// The double-precision cycle throughout (BRAE_AMG_FP32=0, set before the first solve): its captured graph reads
// the coarse views' lower, where the single-precision one reads copies cast outside the graph.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_amg.cuh"
#include "device_amg_detail.cuh"   // BRAE_HAS_GS_DEVICE: leg (e) needs the whole-loop PCG graph (CUDA 13)
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
    const char* what)
{
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

bool sameBits(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0);
}

// The V-cycle applied once with its graph captured -- the second call of a pair replays what the first captured
std::vector<scalar> cycle(
    AMGData& amg,
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& r)
{
    DeviceBuffer<scalar> z(std::vector<scalar>(r.size(), 0.0));
    amgVCycleApply(amg, A, r, z, /*captureVcycle*/ true);
    cudaCheck(cudaDeviceSynchronize(), "cycle");
    return z.host();
}

}   // namespace

int main(int argc, char** argv)
{
    setenv("BRAE_AMG_FP32", "0", 1);
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    std::printf("AMG coarse matrices with one off-diagonal (%s):\n", caseDir.c_str());
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    GeometricField<scalar> f;
    f.internal.assign(nC, 0.0);
    for (const FvPatch& p : fvp)
    {
        if (p.type == "empty")     f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        else if (p.type == "wall") f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        else                       f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(p, true, 0.0, std::vector<scalar>{}));
    }
    f.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(f, 1.0, m, g, fvp);
    std::vector<scalar> diagC = M.diag;
    std::vector<scalar> bSrc(nC);
    for (label c = 0; c < nC; ++c) bSrc[c] = g.V()[c] * std::sin(0.01 * c + 0.4);
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
    const std::vector<scalar> magSfInt(g.magSf().begin(), g.magSf().begin() + nIf);
    // the same lower with the off-diagonal weighted face by face: two off-diagonals that differ on every level
    std::vector<scalar> lowerAsym(M.upper);
    for (label i = 0; i < nIf; ++i) lowerAsym[i] *= 1.0 + 0.5*std::sin(0.37*i);

    const DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.upper, ownerInt, m.neighbour(), nC);
    const DeviceLduMatrix dA = buildDeviceLdu(diagC, M.upper, lowerAsym, ownerInt, m.neighbour(), nC);
    DeviceLduView one = dM.view();
    one.lower = one.upper;
    const DeviceBuffer<scalar> db(bSrc);

    // (a) the premise
    AMGData S = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
    AMGData T = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
    amgGalerkin(S, dM.diag, dM.upper, dM.upper);
    amgGalerkin(T, dM.diag, dM.upper, dM.lower);
    bool sNone = S.nLevels() > 1;
    bool tAll = true;
    for (int k = 0; k < S.nLevels(); ++k)
    {
        sNone = sNone && (S.level[k].nCoarseFaces == 0 || S.level[k].cLower.size() == 0);
        tAll = tAll && static_cast<int>(T.level[k].cLower.size()) == T.level[k].nCoarseFaces;
    }
    std::printf("  %d levels, %zu cells and %d faces on the first coarse grid\n", S.nLevels(),
                static_cast<std::size_t>(S.level[0].nCoarse), S.level[0].nCoarseFaces);
    check(sNone, "(a) handed one off-diagonal, no level of several holds a cLower");
    check(tAll, "(a) handed two buffers, every level holds a cLower of its own");

    // (b) the coarse matrices
    bool same = true;
    for (int k = 0; k < S.nLevels(); ++k)
    {
        const std::vector<scalar> su = S.level[k].cUpper.host();
        same = same && sameBits(S.level[k].cDiag.host(), T.level[k].cDiag.host());
        same = same && sameBits(su, T.level[k].cUpper.host());
        same = same && sameBits(su, T.level[k].cLower.host());
    }
    check(same, "(b) cDiag, cUpper and the lower the same bits on every level");

    // (c) a captured solve on each
    DeviceBuffer<scalar> xS(std::vector<scalar>(nC, 0.0));
    DeviceBuffer<scalar> xT(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf pS = deviceAMGPCG(one, S, db, xS, normFactor, 1e-10, 0.0, 30000, /*captureVcycle*/ true);
    const DeviceSolverPerf pT = deviceAMGPCG(dM.view(), T, db, xT, normFactor, 1e-10, 0.0, 30000, /*captureVcycle*/ true);
    cudaCheck(cudaDeviceSynchronize(), "solves");
    std::printf("  solves: %d and %d iterations\n", pS.nIterations, pT.nIterations);
    check(pS.nIterations == pT.nIterations && pS.nIterations > 1 && sameBits(xS.host(), xT.host()),
          "(c) the captured solves the same bits in the same iterations");

    // (d) the switch, against a hierarchy that only ever saw the two off-diagonals
    const DeviceBuffer<scalar> r(bSrc);
    AMGData R = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
    amgGalerkin(R, dA.diag, dA.upper, dA.lower);
    const std::vector<scalar> zRef = cycle(R, dA.view(), r);
    bool asym = false;
    for (int k = 0; k < R.nLevels() && !asym; ++k)
    {
        const std::vector<scalar> u = R.level[k].cUpper.host();
        const std::vector<scalar> l = R.level[k].cLower.host();
        for (std::size_t i = 0; i < u.size() && !asym; ++i) asym = std::fabs(u[i] - l[i]) > 1e-3*std::fabs(u[i]);
    }
    check(asym, "(d) premise: the reference's coarse matrices are asymmetric");
    for (int arm = 0; arm < 2; ++arm)
    {
        const bool control = arm == 1;
        if (control) setenv("BRAE_CONTROL_AMG_LOWER_GRAPHS_KEPT", "1", 1);
        AMGData C = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
        amgGalerkin(C, dM.diag, dM.upper, dM.upper);
        const std::vector<scalar> zBefore = cycle(C, dA.view(), r);       // captured against the shared lower
        amgGalerkin(C, dA.diag, dA.upper, dA.lower);
        const std::vector<scalar> zAfter = cycle(C, dA.view(), r);
        bool took = true;
        for (int k = 0; k < C.nLevels(); ++k) took = took && static_cast<int>(C.level[k].cLower.size()) == C.level[k].nCoarseFaces;
        if (control)
        {
            unsetenv("BRAE_CONTROL_AMG_LOWER_GRAPHS_KEPT");
            check(!sameBits(zAfter, zRef), "(d) CONTROL, graphs kept: the replay is NOT the fresh hierarchy's cycle");
        }
        else
        {
            check(took, "(d) handed two off-diagonals, every level took its cLower");
            check(!sameBits(zBefore, zRef), "(d) premise: the cycle before the switch is another operator's");
            check(sameBits(zAfter, zRef), "(d) after the switch the cycle is the fresh hierarchy's, bit for bit");
        }
    }
    // (e) the whole-loop PCG graph's copy of the matrix, one view after the other on one hierarchy
#ifdef BRAE_HAS_GS_DEVICE
    {
        // the same addressing, the upper as the lower or the asymmetric lower beside it: only the lower moves
        DeviceLduView two = dM.view();
        two.lower = dA.lower.data();
        // eight iterations from zero into ONE psi per hierarchy: the graph is keyed on psi, so a second buffer
        // would recapture whatever the lower did, and the control below could not fail
        const auto solve = [&](AMGData& amg, const DeviceLduView& v, DeviceBuffer<scalar>& x)
        {
            cudaCheck(cudaMemsetAsync(x.data(), 0, static_cast<std::size_t>(nC)*sizeof(scalar), cudaStreamPerThread), "psi = 0");
            deviceAMGPCG(v, amg, db, x, normFactor, 1e-30, 0.0, 8, /*captureVcycle*/ true);
            cudaCheck(cudaDeviceSynchronize(), "graph solve");
            return x.host();
        };
        DeviceBuffer<scalar> xE(std::vector<scalar>(nC, 0.0));
        const auto fresh = [&]()
        {
            AMGData a = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
            amgGalerkin(a, dM.diag, dM.upper, dM.upper);
            return a;
        };
        AMGData F1 = fresh();
        const std::vector<scalar> refOne = solve(F1, one, xE);
        check(F1.pcgCache && F1.pcgCache->gLower.size() == 0 && F1.pcgCache->gUpper.size() == static_cast<std::size_t>(nIf),
              "(e) premise: handed one off-diagonal, the graph copies the upper alone");
        AMGData F2 = fresh();
        const std::vector<scalar> refTwo = solve(F2, two, xE);
        check(F2.pcgCache && F2.pcgCache->gLower.size() == static_cast<std::size_t>(nIf),
              "(e) premise: handed two, it copies the lower too");
        check(!sameBits(refOne, refTwo), "(e) premise: the two views' solves differ");
        for (int arm = 0; arm < 2; ++arm)
        {
            const bool control = arm == 1;
            if (control) setenv("BRAE_CONTROL_PCG_GRAPH_LOWER_UNKEYED", "1", 1);
            AMGData E = fresh();
            solve(E, two, xE);
            const std::vector<scalar> twoThenOne = solve(E, one, xE);
            AMGData G = fresh();
            solve(G, one, xE);
            const std::vector<scalar> oneThenTwo = solve(G, two, xE);
            if (control)
            {
                unsetenv("BRAE_CONTROL_PCG_GRAPH_LOWER_UNKEYED");
                check(!sameBits(twoThenOne, refOne), "(e) CONTROL, lower unkeyed: two then one replays the other lower");
                check(!sameBits(oneThenTwo, refTwo), "(e) CONTROL, lower unkeyed: one then two replays the upper as the lower");
            }
            else
            {
                check(sameBits(twoThenOne, refOne), "(e) two then one: the second solve is a fresh one-view solver's, bit for bit");
                check(sameBits(oneThenTwo, refTwo), "(e) one then two: the second solve is a fresh two-view solver's, bit for bit");
            }
        }
    }
#else
    std::printf("  (e) not run: no whole-loop PCG graph in this build (it needs CUDA 13)\n");
#endif

    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
