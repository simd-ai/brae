// AMG hierarchy cache I/O -- serialization of the static agglomeration STRUCTURE (values are re-Galerkined,
// so not stored). Pure host: writes/reads the DeviceBuffer contents via .host()/.copyFrom(). Split out of
// device_amg.cu (which owns the build + V-cycle solve) so cache I/O is its own small translation unit.
#include "device_amg.cuh"          // AMGData / AMGLevel / GridColoring + writeAMGCache/loadAMGCache decls
#include "device_amg_detail.cuh"   // useGS()/useSA() (smoother/aggregation mode) + finalizeAMG()
#include "device_buffer.cuh"
#include <cstdio>
#include <cstddef>
#include <string>
#include <vector>

namespace brae {

// AMG hierarchy cache. The agglomeration (greedy + sort per level) is the AMG-build cost and is static per mesh
// (only the matrix VALUES change each step, via Galerkin), so the static hierarchy STRUCTURE is serialized: a
// "partition" step builds it once and the run reloads it. cDiag/cUpper/cLower hold values (Galerkin re-fills them),
// so they are not serialized, only re-sized.
namespace {
constexpr unsigned AMG_CACHE_MAGIC = 0x43464131;          // "CFA1"
template<class T>
void wbuf(
    std::FILE* f,
    const DeviceBuffer<T>& b)
{
    std::vector<T> h = b.host();
    std::size_t n = h.size();
    std::fwrite(&n,sizeof(n),1,f);
    if (n) std::fwrite(h.data(),sizeof(T),n,f);
}
template<class T>
bool rbuf(
    std::FILE* f,
    DeviceBuffer<T>& b)
{
    std::size_t n;
    if (std::fread(&n,sizeof(n),1,f)!=1) return false;
    std::vector<T> h(n);
    if (n && std::fread(h.data(),sizeof(T),n,f)!=n) return false;
    b.copyFrom(h);
    return true;
}
}
void writeAMGCache(
    const AMGData& A,
    const std::string& path)
{
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    unsigned magic = AMG_CACHE_MAGIC;
    std::fwrite(&magic,sizeof(magic),1,f);
    int nFine = A.nFine, nLev = A.nLevels();
    char gs = A.gsSmooth, sa = A.saSmooth;
    std::fwrite(&nFine,sizeof(nFine),1,f);
    std::fwrite(&nLev,sizeof(nLev),1,f);
    std::fwrite(&gs,1,1,f);
    std::fwrite(&sa,1,1,f);
    for (const auto& L : A.level)
    {
        std::fwrite(&L.nFine,sizeof(int),1,f);
        std::fwrite(&L.nCoarse,sizeof(int),1,f);
        std::fwrite(&L.nCoarseFaces,sizeof(int),1,f);
        std::fwrite(&L.nTriples,sizeof(int),1,f);
        wbuf(f,L.map);
        wbuf(f,L.cOwn);
        wbuf(f,L.cNei);
        wbuf(f,L.cOwnerStart);
        wbuf(f,L.cLosort);
        wbuf(f,L.cLosortStart);
        wbuf(f,L.faceRestrict);
        wbuf(f,L.faceFlip);
        wbuf(f,L.Prow);
        wbuf(f,L.Pcol);
        wbuf(f,L.Pval);
        wbuf(f,L.rapSrcKind);
        wbuf(f,L.rapSrcIdx);
        wbuf(f,L.rapDstKind);
        wbuf(f,L.rapDstIdx);
        wbuf(f,L.rapW);
    }
    int nCol = static_cast<int>(A.coloring.size());
    std::fwrite(&nCol,sizeof(nCol),1,f);
    for (const auto& c : A.coloring)
    {
        std::fwrite(&c.nColors,sizeof(c.nColors),1,f);
        wbuf(f,c.cells);
        wbuf(f,c.start);
        std::size_t ns = c.startH.size();
        std::fwrite(&ns,sizeof(ns),1,f);
        if (ns) std::fwrite(c.startH.data(),sizeof(label),ns,f);
    }
    std::fwrite(&magic,sizeof(magic),1,f);                 // trailing sentinel (truncation/corruption check)
    std::fclose(f);
}
bool loadAMGCache(
    const std::string& path,
    AMGData& A)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto fail = [&]()
    {
        std::fclose(f);
        return false;
    };
    unsigned magic = 0;
    if (std::fread(&magic,sizeof(magic),1,f)!=1 || magic!=AMG_CACHE_MAGIC) return fail();
    int nFine=0, nLev=0;
    char gs=0, sa=0;
    if (std::fread(&nFine,sizeof(nFine),1,f)!=1 || std::fread(&nLev,sizeof(nLev),1,f)!=1
        || std::fread(&gs,1,1,f)!=1 || std::fread(&sa,1,1,f)!=1) return fail();
    if ((bool)gs != useGS() || (bool)sa != useSA()) return fail();   // smoother/aggregation mode (env) must match
    A = AMGData{};
    A.nFine = nFine;
    A.gsSmooth = gs;
    A.saSmooth = sa;
    A.level.resize(nLev);
    bool ok = true;
    for (auto& L : A.level)
    {
        ok = ok && std::fread(&L.nFine,sizeof(int),1,f)==1 && std::fread(&L.nCoarse,sizeof(int),1,f)==1
                && std::fread(&L.nCoarseFaces,sizeof(int),1,f)==1 && std::fread(&L.nTriples,sizeof(int),1,f)==1;
        ok = ok && rbuf(f,L.map) && rbuf(f,L.cOwn) && rbuf(f,L.cNei) && rbuf(f,L.cOwnerStart) && rbuf(f,L.cLosort)
                && rbuf(f,L.cLosortStart) && rbuf(f,L.faceRestrict) && rbuf(f,L.faceFlip)
                && rbuf(f,L.Prow) && rbuf(f,L.Pcol) && rbuf(f,L.Pval)
                && rbuf(f,L.rapSrcKind) && rbuf(f,L.rapSrcIdx) && rbuf(f,L.rapDstKind) && rbuf(f,L.rapDstIdx) && rbuf(f,L.rapW);
        if (!ok) break;
        L.cDiag.resize(L.nCoarse);
        L.cUpper.resize(L.nCoarseFaces);
        L.cLower.resize(L.nCoarseFaces);   // VALUES via Galerkin
        // ...and the gather lists the Galerkin re-fill indexes with. They are NOT in the file: they are
        // a pure function of map/faceRestrict/faceFlip, which are, so they are rebuilt through the same
        // builder the build path uses. Without this every cached run died on its first Galerkin.
        rebuildGalerkinGather(L, L.nFine);
    }
    int nCol = 0;
    ok = ok && std::fread(&nCol,sizeof(nCol),1,f)==1;
    if (ok) A.coloring.resize(nCol);
    for (int i = 0; ok && i < nCol; ++i)
    {
        auto& c = A.coloring[i];
        ok = ok && std::fread(&c.nColors,sizeof(c.nColors),1,f)==1 && rbuf(f,c.cells) && rbuf(f,c.start);
        std::size_t ns = 0;
        ok = ok && std::fread(&ns,sizeof(ns),1,f)==1;
        if (ok)
        {
            c.startH.resize(ns);
            ok = ok && (ns==0 || std::fread(c.startH.data(),sizeof(label),ns,f)==ns);
        }
    }
    unsigned tail = 0;
    ok = ok && std::fread(&tail,sizeof(tail),1,f)==1 && tail==AMG_CACHE_MAGIC;   // sentinel
    std::fclose(f);
    if (!ok) return false;
    A.nCoarse = A.level.empty() ? nFine : A.level.front().nCoarse;
    A.nCoarseFaces = A.level.empty() ? 0 : A.level.front().nCoarseFaces;
    finalizeAMG(A, nFine);
    return true;
}

} // namespace brae
