// THE FUSED COARSE-LEVEL FP32 V-CYCLE (device_amg_fused.cu) against the unfused cycle, bit for bit.
//
// On each mesh a real hierarchy is built and cast to single precision, and the V-cycle is applied with
// BRAE_AMG_FUSE_BELOW=0 (the kernels the cycle has always launched) and with fusion on at several thresholds --
// the default 1024, 4096, one that fuses only a few grids, one that fuses every grid including the finest -- for
// three right-hand sides, directly launched and as captured graphs. Every output is memcmp'd with the unfused
// direct launch's. Then:
//   * premise: the fused arms really launched the kernel (amgFusedStats) and the unfused arms did not;
//   * a captured graph reads its coefficients at replay: one coarse diagonal is changed in place and the
//     replay of each graph must differ from before, fused and unfused alike, and agree with each other;
//   * NEGATIVE CONTROL: the same comparison against a cycle whose last smoothed grid's diagonal was perturbed by 1e-6
//     relative must FAIL -- the test sees one coarse coefficient;
//   * TIMING: the captured cycle's time per call and its graph's node count, unfused and fused.
// BRAE_AMG_COARSE_SOLVE selects the coarsest solve (read once per process; ctest runs this under lu and tri): the
// inverse (the default) and the LU are fused, the warp substitutions and Cholesky are not, so that arm must keep
// the kernel out and still match.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_amg.cuh"
#include "device_amg_fused.cuh"
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

bool sameBits(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0);
}

// one application of the cycle; the graph, when captured, is the hierarchy's own (amg.gcacheF)
std::vector<scalar> cycle(
    AMGData& amg,
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& r,
    bool captured)
{
    DeviceBuffer<scalar> z(std::vector<scalar>(r.size(), 0.0));
    amgVCycleApply(amg, A, r, z, captured);
    cudaCheck(cudaDeviceSynchronize(), "cycle");
    return z.host();
}

// a fresh captured graph for the next arm: the graph's key does not hold the fusion switch
void dropGraph(
    AMGData& amg)
{
    amg.gcacheF = std::make_unique<AMGGraphCache>();
}

std::size_t graphNodes(
    AMGData& amg)
{
    std::size_t n = 0;
    cudaCheck(cudaGraphGetNodes(amg.gcacheF->graph, nullptr, &n), "graph nodes");
    return n;
}

void setBelow(
    int below)
{
    setenv("BRAE_AMG_FUSE_BELOW", std::to_string(below).c_str(), 1);
}

// microseconds a call of the captured cycle takes, over `reps` replays
double timeCaptured(
    AMGData& amg,
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& r,
    int reps)
{
    DeviceBuffer<scalar> z(std::vector<scalar>(r.size(), 0.0));
    for (int i = 0; i < 20; ++i) amgVCycleApply(amg, A, r, z, true);
    cudaCheck(cudaDeviceSynchronize(), "warm");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, cudaStreamPerThread);
    for (int i = 0; i < reps; ++i) amgVCycleApply(amg, A, r, z, true);
    cudaEventRecord(e1, cudaStreamPerThread);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1000.0*ms/reps;
}

void runCase(
    const std::string& caseDir)
{
    std::printf("fused coarse V-cycle on %s:\n", caseDir.c_str());
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
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        for (label i = 0; i < fvp[pi].size; ++i) diagC[fvp[pi].faceCells[i]] += M.internalCoeffs[pi][i];
    }
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    const std::vector<scalar> magSfInt(g.magSf().begin(), g.magSf().begin() + nIf);
    // off-diagonals that differ a little face by face, so the product's order is not hidden by symmetry
    std::vector<scalar> lowerW(M.upper);
    for (label i = 0; i < nIf; ++i) lowerW[i] *= 1.0 + 0.1*std::sin(0.37*i);

    const DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.upper, ownerInt, m.neighbour(), nC);
    const DeviceLduMatrix dA = buildDeviceLdu(diagC, M.upper, lowerW, ownerInt, m.neighbour(), nC);

    const char* sel = std::getenv("BRAE_AMG_COARSE_SOLVE");
    const std::string solve = sel ? sel : "";
    const bool expectFused = !(solve == "tri" || solve == "cholesky");

    for (int arm = 0; arm < 2; ++arm)                                    // symmetric, then two off-diagonals
    {
        const bool sym = arm == 0;
        const DeviceLduView A = sym ? dM.view() : dA.view();
        std::printf(" %s off-diagonals:\n", sym ? "one" : "two");
        AMGData amg = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
        const DeviceLduMatrix& D = sym ? dM : dA;
        amgGalerkin(amg, D.diag, D.upper, D.lower);
        amgPrepareFP32(amg, A);
        check(amg.fp32Alloc && amgSinglePrecisionCycle(amg), "premise: the single-precision cycle is the one applied");
        std::printf("  %d cells, %d levels, coarsest %d cells, coarsest solve '%s'\n", static_cast<int>(nC), amg.nLevels(),
                    amg.level.back().nCoarse, solve.empty() ? "default" : solve.c_str());

        std::vector<DeviceBuffer<scalar>> rhs;
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> r(nC);
            for (label c = 0; c < nC; ++c) r[c] = std::sin(0.013*(k + 1)*c + 0.4*k) + 0.2*std::cos(0.7*c*(k + 1));
            rhs.emplace_back(r);
        }

        // the reference: unfused, directly launched, three right-hand sides
        setBelow(0);
        const long long before = amgFusedStats().launches;
        std::vector<std::vector<scalar>> ref;
        for (int k = 0; k < 3; ++k) ref.push_back(cycle(amg, A, rhs[k], false));
        check(amgFusedStats().launches == before, "BRAE_AMG_FUSE_BELOW=0 launches no fused kernel");

        // the unfused captured graph against it
        dropGraph(amg);
        bool unfusedGraph = true;
        for (int k = 0; k < 3; ++k) unfusedGraph = unfusedGraph && sameBits(cycle(amg, A, rhs[k], true), ref[k]);
        check(unfusedGraph, "unfused: captured graph replay = direct launch, 3 right-hand sides");
        const std::size_t nodesOff = graphNodes(amg);

        const int thresholds[] = {1024, 4096, 700, 100000, 64};
        for (int below : thresholds)
        {
            setBelow(below);
            const long long l0 = amgFusedStats().launches;
            bool direct = true;
            for (int k = 0; k < 3; ++k) direct = direct && sameBits(cycle(amg, A, rhs[k], false), ref[k]);
            const FusedStats st = amgFusedStats();
            const bool ran = st.launches > l0;
            check(ran == expectFused, "BELOW=" + std::to_string(below) + ": the fused kernel "
                  + (expectFused ? "ran" : "stayed out") + " (first grid " + std::to_string(st.firstGrid) + ", "
                  + std::to_string(st.grids) + " grids)");
            check(direct, "BELOW=" + std::to_string(below) + ": direct launch = unfused, bit for bit, 3 right-hand sides");
            dropGraph(amg);
            bool graph = true;
            for (int k = 0; k < 3; ++k) graph = graph && sameBits(cycle(amg, A, rhs[k], true), ref[k]);
            check(graph, "BELOW=" + std::to_string(below) + ": captured graph replay = unfused, bit for bit");
            if (below == 1024) std::printf("  graph nodes: %zu unfused, %zu fused\n", nodesOff, graphNodes(amg));
        }
        // the default (BRAE_AMG_FUSE_BELOW unset)
        unsetenv("BRAE_AMG_FUSE_BELOW");
        {
            const long long l0 = amgFusedStats().launches;
            bool direct = true;
            for (int k = 0; k < 3; ++k) direct = direct && sameBits(cycle(amg, A, rhs[k], false), ref[k]);
            check((amgFusedStats().launches > l0) == expectFused, "default threshold: the fused kernel " + std::string(expectFused ? "ran" : "stayed out"));
            check(direct, "default threshold: direct launch = unfused, bit for bit");
        }

        // the replay reads live coefficients, fused or not; and the control: the comparison sees one coarse coefficient
        const int G = amg.nLevels() - 1;                                          // the last grid the cycle smooths (the coarsest's FP32 diagonal is unread)
        const std::vector<float> d0 = amg.fDiag[G].host();
        std::vector<float> d1 = d0;
        d1[0] *= 1.000001f;
        std::vector<std::vector<scalar>> perturbed;
        for (int below : {0, 1024})
        {
            setBelow(below);
            dropGraph(amg);
            const std::vector<scalar> before1 = cycle(amg, A, rhs[0], true);        // captured with the true coefficient
            amg.fDiag[G].copyFrom(d1);
            perturbed.push_back(cycle(amg, A, rhs[0], true));                       // replayed: reads the changed one
            amg.fDiag[G].copyFrom(d0);
            check(!sameBits(before1, perturbed.back()), std::string(below == 0 ? "unfused" : "fused")
                  + ": a graph replay reads the coarsest diagonal's current value");
        }
        check(sameBits(perturbed[0], perturbed[1]), "both replays of the perturbed coefficient agree, bit for bit");
        check(!sameBits(perturbed[1], ref[0]),
              "NEGATIVE CONTROL: the fused cycle with one coarsest diagonal off by 1e-6 is NOT the unperturbed unfused cycle");

        // timing: the captured cycle's time per call, unfused and fused at a sweep of thresholds (best of three runs)
        const int reps = 200;
        std::printf("  captured V-cycle call, us (grids fused):");
        double tOff = 0.0;
        double tDef = 0.0;
        for (int below : {0, 256, 1024, 4096, 16384, 65536})
        {
            setBelow(below);
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep)
            {
                dropGraph(amg);
                best = std::min(best, timeCaptured(amg, A, rhs[0], reps));
            }
            if (below == 0) tOff = best;
            if (below == 1024) tDef = best;
            std::printf("  %d: %.1f (%d)", below, best, below == 0 ? 0 : amgFusedStats().grids);
        }
        std::printf("\n  fused (default 1024) %.2fx the unfused cycle\n", tOff/tDef);
        unsetenv("BRAE_AMG_FUSE_BELOW");
    }
}

}   // namespace

int main(int argc, char** argv)
{
    unsetenv("BRAE_AMG_FP32");
    std::vector<std::string> cases;
    for (int i = 1; i < argc; ++i) cases.push_back(argv[i]);
    if (cases.empty()) cases.push_back("validation/pitzDaily");
    for (const std::string& c : cases) runCase(c);
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
