// np ranks of one decomposePar'd case in ONE process on ONE GPU (ProcComm loopback, a thread per rank): the
// processor-patch coupling and the device interface on REAL meshes.
//
//   test_proc_geometry_loopback <decomposed case dir>     (processor0..N-1 with constant/polyMesh; the case's own
//                                                          constant/polyMesh and the ProcAddressing files are used when present)
//
// Stage 1 (serial, host): every rank reads its processorN mesh through brae's reader and builds FvGeometry and the
//   patches. Stage 2 (rank threads): attachProcessorCoupling -- the setup exchange, the HaloPlan, the coupled half of
//   every processor patch. Stage 3 (serial, GPU): buildDeviceMesh + buildDeviceProcInterface. Stage 4 (rank threads):
//   the halo round trip -- the cell centres are exchanged and every halo slot must hold the centre of the right
//   NEIGHBOUR cell (named independently of the plan: face i of my patch to q is face i of q's patch to me), the face
//   exchange (Replace) must hand every face its twin's id.
// Checks, per rank:
//   - face i of my patch pairs with face i of the neighbour's: the face centres coincide, the areas oppose;
//   - the geometry the exchange produced equals, BIT FOR BIT, coupleProcessorPair fed the neighbour's own packed data
//     (the transport is lossless);
//   - halo slots: distinct, contiguous per neighbour, every slot used, slot of a face is a function of its neighbour cell;
//   - DeviceMesh: nHalo, nCellsExt, nBndFacesPhys; the DeviceCyclic's neighbour index = nC + slot = the plan's;
//   - when the global mesh and cellProcAddressing are there: the two cells of every interface face are adjacent in the
//     global mesh, and the halo value of the cell centre agrees with the GLOBAL cell centre (to 1e-10 of the mesh size).
// Prints interface faces / neighbours / halo cells per rank (compare with tools/mgpu_oracle/decomp_stats.py).
#include "device_processor.cuh"
#include "foam_token_reader.cuh"
#include "processor_coupling.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using namespace brae;

static int fails = 0;
#define CHECK(cond, ...)                                                                      \
    do {                                                                                      \
        if (!(cond)) { ++fails; std::printf("  FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond);  \
                       std::printf(__VA_ARGS__); std::printf("\n"); }                         \
    } while (0)

struct Rank
{
    PrimitiveMesh m;
    FvGeometry g;
    std::vector<FvPatch> patches;
    std::vector<label> cellProc;   // local -> global cell, empty without cellProcAddressing
    ProcessorCoupling pc;
};

static bool sameBits(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0);
}
static bool sameBits(const std::vector<vector>& a, const std::vector<vector>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()*sizeof(vector)) == 0);
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s <decomposed case dir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];
    int np = 0;
    while (fs::exists(caseDir + "/processor" + std::to_string(np))) ++np;
    if (np < 1) { std::printf("no processor0 in %s\n", caseDir.c_str()); return 2; }
    std::printf("test_proc_geometry_loopback: %s, np %d\n", caseDir.c_str(), np);

    // ---- stage 1: meshes, serial (the reader and FvGeometry keep caches and statics) ------------------------------
    std::vector<std::unique_ptr<Rank>> R;
    for (int r = 0; r < np; ++r)
    {
        R.emplace_back(new Rank);
        const std::string d = caseDir + "/processor" + std::to_string(r) + "/constant/polyMesh";
        R[r]->m.read(d, d, d);
        R[r]->g.build(R[r]->m);
        R[r]->patches = buildPatches(R[r]->m, R[r]->g);
        if (fs::exists(d + "/cellProcAddressing")) R[r]->cellProc = readLabelListFile(d + "/cellProcAddressing");
    }
    PrimitiveMesh gm;
    FvGeometry gg;
    std::unordered_set<unsigned long long> adj;
    const bool haveGlobal = fs::exists(caseDir + "/constant/polyMesh/owner") && !R[0]->cellProc.empty();
    if (haveGlobal)
    {
        const std::string d = caseDir + "/constant/polyMesh";
        gm.read(d, d, d);
        gg.build(gm);
        for (label f = 0; f < gm.nInternalFaces(); ++f)
        {
            const unsigned long long a = static_cast<unsigned long long>(std::min(gm.owner()[f], gm.neighbour()[f]));
            const unsigned long long b = static_cast<unsigned long long>(std::max(gm.owner()[f], gm.neighbour()[f]));
            adj.insert(a << 32 | b);
        }
    }

    // ---- stage 2: the coupling, a thread per rank ------------------------------------------------------------------
    std::vector<std::unique_ptr<ProcComm>> comms = ProcComm::createLoopback(np);
    std::vector<std::string> err(static_cast<std::size_t>(np));
    {
        std::vector<std::thread> th;
        for (int r = 0; r < np; ++r)
        {
            th.emplace_back([&, r] {
                try
                {
                    comms[r]->bindThread();
                    R[r]->pc = attachProcessorCoupling(R[r]->patches, R[r]->g, R[r]->m.nCells(), *comms[r]);
                }
                catch (const std::exception& e) { err[static_cast<std::size_t>(r)] = e.what(); }
            });
        }
        for (auto& t : th) t.join();
    }
    for (int r = 0; r < np; ++r)
    {
        CHECK(err[r].empty(), "rank %d: %s", r, err[r].c_str());
    }
    if (fails) { std::printf("coupling failed\n"); return 1; }

    // ---- host checks ------------------------------------------------------------------------------------------------
    auto patchTo = [&](int r, int q) -> const FvPatch* {
        for (const FvPatch& p : R[r]->patches)
            if (p.type == "processor" && p.size > 0 && p.neighbProcNo == q) return &p;
        return nullptr;
    };
    std::printf("%-5s %8s %9s %8s %5s %8s\n", "rank", "cells", "procFaces", "patches", "nbrs", "haloCells");
    // expected (rank, local cell) of every halo slot, named from the OTHER rank's patch
    std::vector<std::vector<std::pair<int, label>>> slotCell(static_cast<std::size_t>(np));
    double worstCentre = 0;
    for (int r = 0; r < np; ++r)
    {
        Rank& A = *R[r];
        const label nC = A.m.nCells();
        slotCell[r].assign(static_cast<std::size_t>(A.pc.nHalo), {-1, -1});
        for (const label pi : A.pc.patches)
        {
            const FvPatch& p = A.patches[static_cast<std::size_t>(pi)];
            const int q = p.neighbProcNo;
            const FvPatch* pq = patchTo(q, r);
            CHECK(pq != nullptr && pq->size == p.size, "rank %d patch %s has no twin of equal size on rank %d", r, p.name.c_str(), q);
            if (!pq || pq->size != p.size) continue;
            CHECK(p.coupled && p.myProcNo == r, "patch %s", p.name.c_str());
            // the transport is lossless: coupleProcessorPair on the neighbour's OWN packed data gives the same bits
            FvPatch copy = p;
            coupleProcessorPair(copy, unpackProcessorFaces(packProcessorFaces(*pq, R[q]->g), pq->size), A.g);
            CHECK(sameBits(copy.weights, p.weights) && sameBits(copy.deltaCoeffs, p.deltaCoeffs)
                  && sameBits(copy.nonOrthDeltaCoeffs, p.nonOrthDeltaCoeffs) && sameBits(copy.delta, p.delta)
                  && sameBits(copy.nonOrthCorrectionVectors, p.nonOrthCorrectionVectors) && sameBits(copy.nbrD, p.nbrD),
                  "rank %d patch %s: the exchanged geometry is not the neighbour's own bits", r, p.name.c_str());
            for (label i = 0; i < p.size; ++i)
            {
                const std::size_t u = static_cast<std::size_t>(i);
                // face i <-> face i: same place, opposite areas
                const vector& c0 = A.g.Cf()[p.start + i];
                const vector& c1 = R[q]->g.Cf()[pq->start + i];
                const vector& s0 = A.g.Sf()[p.start + i];
                const vector& s1 = R[q]->g.Sf()[pq->start + i];
                const scalar L = std::sqrt(A.g.magSf()[p.start + i]);
                CHECK(mag(c0 - c1) < 1e-9*L, "rank %d patch %s face %d: centres %g apart", r, p.name.c_str(), i, mag(c0 - c1));
                CHECK(mag(s0 + s1) < 1e-9*mag(s0), "rank %d patch %s face %d: areas do not oppose", r, p.name.c_str(), i);
                // the halo slot of the neighbour cell
                const label slot = p.nbrFaceCells[u] - nC;
                CHECK(slot >= 0 && slot < A.pc.nHalo && p.nbrFaceCells[u] >= p.haloBase
                      && p.nbrFaceCells[u] < p.haloBase + p.nHaloCells,
                      "rank %d patch %s face %d: slot %d outside [%d,%d)", r, p.name.c_str(), i, slot, p.haloBase - nC,
                      p.haloBase - nC + p.nHaloCells);
                if (slot < 0 || slot >= A.pc.nHalo) continue;
                const std::pair<int, label> want{q, pq->faceCells[u]};
                auto& have = slotCell[r][static_cast<std::size_t>(slot)];
                if (have.first < 0) have = want;
                CHECK(have == want, "rank %d slot %d holds two neighbour cells", r, slot);
                if (haveGlobal)
                {
                    const label ga = A.cellProc[static_cast<std::size_t>(p.faceCells[u])];
                    const label gb = R[q]->cellProc[static_cast<std::size_t>(pq->faceCells[u])];
                    const unsigned long long a = static_cast<unsigned long long>(std::min(ga, gb));
                    const unsigned long long b = static_cast<unsigned long long>(std::max(ga, gb));
                    CHECK(adj.count(a << 32 | b) == 1, "rank %d patch %s face %d: global cells %d and %d are not adjacent",
                          r, p.name.c_str(), i, ga, gb);
                    const scalar d = mag(R[q]->g.C()[pq->faceCells[u]] - gg.C()[gb]);
                    worstCentre = std::max(worstCentre, (double)d);
                    CHECK(d < 1e-10, "rank %d: neighbour cell centre is %g from the global cell's", r, d);
                }
            }
        }
        for (std::size_t s = 0; s < slotCell[r].size(); ++s)
        {
            CHECK(slotCell[r][s].first >= 0, "rank %d halo slot %zu is used by no face", r, s);
        }
        std::printf("%-5d %8ld %9ld %8zu %5d %8ld\n", r, (long)nC, (long)A.pc.nProcFaces, A.pc.patches.size(),
                    A.pc.plan->nNeighbours(), (long)A.pc.nHalo);
    }
    if (haveGlobal) std::printf("global check: interface cells adjacent in the global mesh; worst neighbour-centre difference %.3g\n", worstCentre);
    else std::printf("no global mesh / cellProcAddressing: adjacency and global centres not checked\n");

    // ---- stage 3: device meshes and interfaces, serial (the device pool is single-threaded) ---------------------------
    std::vector<DeviceMesh> dm(static_cast<std::size_t>(np));
    std::vector<DeviceProcInterface> dp(static_cast<std::size_t>(np));
    for (int r = 0; r < np; ++r)
    {
        dm[r] = buildDeviceMesh(R[r]->m, R[r]->g, R[r]->patches);
        dp[r] = buildDeviceProcInterface(dm[r], R[r]->g, R[r]->patches, R[r]->pc);
        const Rank& A = *R[r];
        label nBndPhysExpect = 0;
        for (const FvPatch& p : A.patches) if (p.type != "processor") nBndPhysExpect += p.size;
        CHECK(dm[r].nHalo == A.pc.nHalo && dm[r].nCellsExt == A.m.nCells() + A.pc.nHalo, "rank %d dm halo %d", r, dm[r].nHalo);
        CHECK(dm[r].nBndFacesPhys == nBndPhysExpect && dm[r].nBndFaces == nBndPhysExpect + A.pc.nProcFaces,
              "rank %d boundary prefix %d/%d", r, dm[r].nBndFacesPhys, dm[r].nBndFaces);
        CHECK(dp[r].nProcFaces == A.pc.nProcFaces && dp[r].cyc.n == A.pc.nProcFaces, "rank %d interface faces", r);
        if (A.pc.nProcFaces > 0)
        {
            std::vector<label> own(static_cast<std::size_t>(dp[r].cyc.n)), nbr(own.size()), bnd(static_cast<std::size_t>(dm[r].nBndFaces));
            dp[r].cyc.ownCell.copyTo(own);
            dp[r].cyc.nbrCell.copyTo(nbr);
            dm[r].bndCell.copyTo(bnd);
            const HaloPlanDesc& pd = A.pc.plan->desc();
            for (std::size_t j = 0; j < own.size(); ++j)
            {
                CHECK(own[j] == pd.faceCells[j] && own[j] == bnd[static_cast<std::size_t>(dm[r].nBndFacesPhys) + j],
                      "rank %d face %zu owner cell: cyc %d plan %d mesh %d", r, j, own[j], pd.faceCells[j],
                      bnd[static_cast<std::size_t>(dm[r].nBndFacesPhys) + j]);
                CHECK(nbr[j] == A.m.nCells() + pd.faceToSlot[j], "rank %d face %zu neighbour index %d", r, j, nbr[j]);
            }
            std::vector<int> viaPlan(own.size());
            CHECK(cudaMemcpy(viaPlan.data(), A.pc.plan->dFaceHaloCell(), viaPlan.size()*sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess, "copy");
            for (std::size_t j = 0; j < own.size(); ++j) CHECK(viaPlan[j] == nbr[j], "rank %d face %zu dFaceHaloCell", r, j);
        }
    }

    // ---- stage 4: the halo round trip and the face exchange, a thread per rank ----------------------------------------
    std::vector<std::string> err4(static_cast<std::size_t>(np));
    {
        std::vector<std::thread> th;
        for (int r = 0; r < np; ++r)
        {
            th.emplace_back([&, r] {
                try
                {
                    comms[r]->bindThread();
                    Rank& A = *R[r];
                    cudaStream_t s;
                    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) throw std::runtime_error("stream");
                    const std::size_t nC = static_cast<std::size_t>(A.m.nCells());
                    const std::size_t nExt = nC + static_cast<std::size_t>(A.pc.nHalo);
                    std::vector<double> h[3];
                    for (int k = 0; k < 3; ++k) h[k].assign(nExt, std::nan(""));
                    for (std::size_t c = 0; c < nC; ++c)
                    {
                        h[0][c] = A.g.C()[c].x; h[1][c] = A.g.C()[c].y; h[2][c] = A.g.C()[c].z;
                    }
                    double* d[3];
                    for (int k = 0; k < 3; ++k)
                    {
                        if (cudaMalloc(&d[k], nExt*sizeof(double)) != cudaSuccess) throw std::runtime_error("malloc");
                        if (cudaMemcpyAsync(d[k], h[k].data(), nExt*sizeof(double), cudaMemcpyHostToDevice, s) != cudaSuccess) throw std::runtime_error("h2d");
                    }
                    const std::size_t nF = static_cast<std::size_t>(A.pc.nProcFaces);
                    std::vector<double> face(nF), faceBack(nF);
                    for (std::size_t i = 0; i < nF; ++i) face[i] = 1.0e7*r + static_cast<double>(i);
                    double* df = nullptr;
                    if (nF)
                    {
                        if (cudaMalloc(&df, nF*sizeof(double)) != cudaSuccess) throw std::runtime_error("malloc");
                        cudaMemcpyAsync(df, face.data(), nF*sizeof(double), cudaMemcpyHostToDevice, s);
                    }
                    cudaStreamSynchronize(s);
                    comms[r]->exchangeVector(*A.pc.plan, d[0], d[1], d[2], s);
                    if (nF) comms[r]->exchangeFaces(*A.pc.plan, df, FaceOp::Replace, s);
                    for (int k = 0; k < 3; ++k) cudaMemcpyAsync(h[k].data(), d[k], nExt*sizeof(double), cudaMemcpyDeviceToHost, s);
                    if (nF) cudaMemcpyAsync(faceBack.data(), df, nF*sizeof(double), cudaMemcpyDeviceToHost, s);
                    cudaStreamSynchronize(s);
                    long bad = 0;
                    for (std::size_t slot = 0; slot < slotCell[r].size(); ++slot)
                    {
                        const int q = slotCell[r][slot].first;
                        const std::size_t lc = static_cast<std::size_t>(slotCell[r][slot].second);
                        const vector& want = R[q]->g.C()[lc];
                        if (std::memcmp(&h[0][nC + slot], &want.x, 8) || std::memcmp(&h[1][nC + slot], &want.y, 8)
                         || std::memcmp(&h[2][nC + slot], &want.z, 8))
                            ++bad;
                    }
                    for (std::size_t c = 0; c < nC; ++c)   // the owned part is untouched
                        if (h[0][c] != A.g.C()[c].x) ++bad;
                    // face i of my patch to q gets the id of face i of q's patch to me
                    std::size_t base = 0;
                    for (const HaloNeighbour& nb : A.pc.plan->desc().nbrs)
                    {
                        std::size_t qbase = 0;
                        for (const HaloNeighbour& nq : R[nb.rank]->pc.plan->desc().nbrs)
                            if (nq.rank == r) qbase = static_cast<std::size_t>(nq.faceStart);
                        for (int i = 0; i < nb.faceCount; ++i)
                            if (faceBack[base + static_cast<std::size_t>(i)] != 1.0e7*nb.rank + static_cast<double>(qbase + i)) ++bad;
                        base += static_cast<std::size_t>(nb.faceCount);
                    }
                    if (bad) err4[static_cast<std::size_t>(r)] = std::to_string(bad) + " halo/face values wrong";
                    for (int k = 0; k < 3; ++k) cudaFree(d[k]);
                    if (df) cudaFree(df);
                    cudaStreamDestroy(s);
                    comms[r]->barrier();   // no rank frees a plan while a neighbour may still read its staging
                }
                catch (const std::exception& e) { err4[static_cast<std::size_t>(r)] = e.what(); }
            });
        }
        for (auto& t : th) t.join();
    }
    for (int r = 0; r < np; ++r) CHECK(err4[r].empty(), "rank %d: %s", r, err4[r].c_str());

    for (int r = 0; r < np; ++r) R[r]->pc.plan.reset();
    dp.clear();
    comms.clear();
    std::printf(fails ? "FAILED: %d check(s)\n" : "OK: np %d, halo round trip, face exchange, device interface\n", fails ? fails : np);
    return fails ? 1 : 0;
}
