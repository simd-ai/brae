// AMG hierarchy cache I/O -- serialization of the static agglomeration STRUCTURE (values are re-Galerkined,
// so not stored). Pure host: writes/reads the DeviceBuffer contents via .host()/.copyFrom(). Split out of
// device_amg.cu (which owns the build + V-cycle solve) so cache I/O is its own small translation unit.
#include "device_amg.cuh"          // AMGData / AMGLevel / GridColoring + writeAMGCache/loadAMGCache decls
#include "device_amg_detail.cuh"   // useGS()/useSA() (smoother/aggregation mode) + finalizeAMG()
#include "device_buffer.cuh"
#include "device_ldu.cuh"          // nextDeviceAddressingId
#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// AMG hierarchy cache. The agglomeration (greedy + sort per level) is the AMG-build cost and is static per mesh
// (only the matrix VALUES change each step, via Galerkin), so the static hierarchy STRUCTURE is serialized: a
// "partition" step builds it once and the run reloads it. cDiag/cUpper/cLower hold values (Galerkin re-fills them),
// so they are not serialized, only re-sized.
namespace {
// "CFA2": the second form of the file, which carries the signature of what the hierarchy was built from. A
// "CFA1" file has none and is read as another build's.
constexpr unsigned AMG_CACHE_MAGIC = 0x43464132;
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
namespace {
template<class T>
void dcopy(
    DeviceBuffer<T>& dst,
    const DeviceBuffer<T>& src)
{
    dst.resize(src.size());
    if (src.size() == 0) return;
    const cudaError_t e = cudaMemcpy(dst.data(), src.data(), src.size()*sizeof(T), cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string("brae cloneAMG: the device copy failed: ") + cudaGetErrorString(e));
    }
}
template<class T>
bool sameBuffer(
    const DeviceBuffer<T>& a,
    const DeviceBuffer<T>& b)
{
    return a.size() == b.size() && a.host() == b.host();
}
}
AMGData cloneAMG(const AMGData& S)
{
    AMGData A;
    A.nFine = S.nFine;
    A.gsSmooth = S.gsSmooth;
    A.saSmooth = S.saSmooth;
    A.pairInCoarseMatrices = S.pairInCoarseMatrices;
    A.level.resize(S.level.size());
    for (std::size_t k = 0; k < S.level.size(); ++k)
    {
        const AMGLevel& s = S.level[k];
        AMGLevel& L = A.level[k];
        L.nFine = s.nFine;
        L.nCoarse = s.nCoarse;
        L.nCoarseFaces = s.nCoarseFaces;
        L.nTriples = s.nTriples;
        L.addressingId = nextDeviceAddressingId();
        dcopy(L.map, s.map);
        dcopy(L.cOwn, s.cOwn);
        dcopy(L.cNei, s.cNei);
        dcopy(L.cOwnerStart, s.cOwnerStart);
        dcopy(L.cLosort, s.cLosort);
        dcopy(L.cLosortStart, s.cLosortStart);
        dcopy(L.faceRestrict, s.faceRestrict);
        dcopy(L.faceFlip, s.faceFlip);
        dcopy(L.galCellStart, s.galCellStart);
        dcopy(L.galCellList, s.galCellList);
        dcopy(L.galDFaceStart, s.galDFaceStart);
        dcopy(L.galDFaceList, s.galDFaceList);
        dcopy(L.galFaceStart, s.galFaceStart);
        dcopy(L.galFaceList, s.galFaceList);
        dcopy(L.galFaceFlipList, s.galFaceFlipList);
        dcopy(L.Prow, s.Prow);
        dcopy(L.Pcol, s.Pcol);
        dcopy(L.Pval, s.Pval);
        dcopy(L.rapSrcKind, s.rapSrcKind);
        dcopy(L.rapSrcIdx, s.rapSrcIdx);
        dcopy(L.rapDstKind, s.rapDstKind);
        dcopy(L.rapDstIdx, s.rapDstIdx);
        dcopy(L.rapW, s.rapW);
        // ...and the fixed-order lists derived from them (amgSaFixedOrder); RvalF is cast with PvalF
        dcopy(L.Rrow, s.Rrow);
        dcopy(L.Rfine, s.Rfine);
        dcopy(L.Rval, s.Rval);
        L.RvalF.resize(0);
        L.Rterm.resize(s.Rterm.size());
        L.RtermF.resize(s.RtermF.size());
        L.rapTerm.resize(s.rapTerm.size());
        dcopy(L.rapStart, s.rapStart);
        // the VALUES are Galerkin's at every solve; only their sizes are the structure's
        L.cDiag.resize(s.nCoarse);
        L.cUpper.resize(s.nCoarseFaces);
        L.cLower.resize(s.nCoarseFaces);
    }
    A.coloring.resize(S.coloring.size());
    for (std::size_t i = 0; i < S.coloring.size(); ++i)
    {
        A.coloring[i].nColors = S.coloring[i].nColors;
        A.coloring[i].nCells = S.coloring[i].nCells;
        dcopy(A.coloring[i].cells, S.coloring[i].cells);
        dcopy(A.coloring[i].start, S.coloring[i].start);
        A.coloring[i].startH = S.coloring[i].startH;
    }
    A.nCoarse = S.nCoarse;
    A.nCoarseFaces = S.nCoarseFaces;
    finalizeAMG(A, A.nFine);
    return A;
}
const char* firstAMGDifference(
    const AMGData& A,
    const AMGData& B)
{
    if (A.nFine != B.nFine) return "the fine cell count";
    if (A.gsSmooth != B.gsSmooth || A.saSmooth != B.saSmooth) return "the smoother or aggregation mode";
    if (A.level.size() != B.level.size()) return "the number of levels";
    for (std::size_t k = 0; k < A.level.size(); ++k)
    {
        const AMGLevel& a = A.level[k];
        const AMGLevel& b = B.level[k];
        if (a.nFine != b.nFine || a.nCoarse != b.nCoarse || a.nCoarseFaces != b.nCoarseFaces
         || a.nTriples != b.nTriples) return "a level's sizes";
        if (!sameBuffer(a.map, b.map)) return "a level's cell map";
        if (!sameBuffer(a.cOwn, b.cOwn) || !sameBuffer(a.cNei, b.cNei)) return "a level's coarse addressing";
        if (!sameBuffer(a.cOwnerStart, b.cOwnerStart) || !sameBuffer(a.cLosort, b.cLosort)
         || !sameBuffer(a.cLosortStart, b.cLosortStart)) return "a level's coarse gather lists";
        if (!sameBuffer(a.faceRestrict, b.faceRestrict) || !sameBuffer(a.faceFlip, b.faceFlip))
            return "a level's face restriction";
        if (!sameBuffer(a.galCellStart, b.galCellStart) || !sameBuffer(a.galCellList, b.galCellList)
         || !sameBuffer(a.galDFaceStart, b.galDFaceStart) || !sameBuffer(a.galDFaceList, b.galDFaceList)
         || !sameBuffer(a.galFaceStart, b.galFaceStart) || !sameBuffer(a.galFaceList, b.galFaceList)
         || !sameBuffer(a.galFaceFlipList, b.galFaceFlipList)) return "a level's Galerkin gather lists";
        if (!sameBuffer(a.Prow, b.Prow) || !sameBuffer(a.Pcol, b.Pcol) || !sameBuffer(a.Pval, b.Pval))
            return "a level's prolongator";
        if (!sameBuffer(a.rapSrcKind, b.rapSrcKind) || !sameBuffer(a.rapSrcIdx, b.rapSrcIdx)
         || !sameBuffer(a.rapDstKind, b.rapDstKind) || !sameBuffer(a.rapDstIdx, b.rapDstIdx)
         || !sameBuffer(a.rapW, b.rapW)) return "a level's RAP recipe";
    }
    if (A.coloring.size() != B.coloring.size()) return "the number of colourings";
    for (std::size_t i = 0; i < A.coloring.size(); ++i)
    {
        if (A.coloring[i].nColors != B.coloring[i].nColors || A.coloring[i].startH != B.coloring[i].startH
         || !sameBuffer(A.coloring[i].cells, B.coloring[i].cells)) return "a colouring";
    }
    return nullptr;
}
bool writeAMGCache(
    const AMGData&     A,
    const std::string& path,
    unsigned long long signature)
{
    const std::string partial = path + ".partial";
    std::FILE* f = std::fopen(partial.c_str(), "wb");
    if (!f) return false;
    unsigned magic = AMG_CACHE_MAGIC;
    std::fwrite(&magic,sizeof(magic),1,f);
    std::fwrite(&signature,sizeof(signature),1,f);
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
    // a short write anywhere above sets the stream's error flag; the flush at the close can fail too
    const bool failed = std::ferror(f) != 0;
    const bool closed = std::fclose(f) == 0;
    if (failed || !closed || std::rename(partial.c_str(), path.c_str()) != 0)
    {
        std::remove(partial.c_str());
        return false;
    }
    return true;
}
AMGCacheRead readAMGCache(
    const std::string& path,
    AMGData&           A,
    unsigned long long signature,
    bool               smoothedAggregation,
    bool               compareSignature)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return AMGCacheRead::absent;
    auto fail = [&](AMGCacheRead why)
    {
        std::fclose(f);
        return why;
    };
    unsigned magic = 0;
    if (std::fread(&magic,sizeof(magic),1,f)!=1) return fail(AMGCacheRead::unreadable);
    // an older form of the file is another build's
    if (magic!=AMG_CACHE_MAGIC) return fail(AMGCacheRead::otherMeshOrBuild);
    unsigned long long written = 0;
    if (std::fread(&written,sizeof(written),1,f)!=1) return fail(AMGCacheRead::unreadable);
    if (compareSignature && written != signature) return fail(AMGCacheRead::otherMeshOrBuild);
    int nFine=0, nLev=0;
    char gs=0, sa=0;
    if (std::fread(&nFine,sizeof(nFine),1,f)!=1 || std::fread(&nLev,sizeof(nLev),1,f)!=1
        || std::fread(&gs,1,1,f)!=1 || std::fread(&sa,1,1,f)!=1) return fail(AMGCacheRead::unreadable);
    // the smoother and aggregation mode are in the signature too; told apart here so a file read with the
    // signature not compared still cannot hand back the other kind of hierarchy
    if ((bool)gs != useGS() || (bool)sa != smoothedAggregation) return fail(AMGCacheRead::otherMeshOrBuild);
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
        // an addressing id of its own, as a built level takes (buildAMG): a loaded level's was left 0
        L.addressingId = nextDeviceAddressingId();
        // the smoothed hierarchy's fixed-order lists are not in the file either: derived from what is
        amgSaFixedOrder(L);
        // ...and the gather lists the Galerkin re-fill indexes with. They are NOT in the file: they are
        // a pure function of map/faceRestrict/faceFlip, which are, so they are rebuilt through the same
        // builder the build path uses. Without this every cached run died on its first Galerkin.
        // A SMOOTHED level has none: its coarse matrix is the RAP recipe's (buildAMG builds no gather lists
        // for it), and lists made here would be the one thing a loaded hierarchy held that a built one did not.
        if (!sa) rebuildGalerkinGather(L, L.nFine);
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
    if (!ok)
    {
        A = AMGData{};
        return AMGCacheRead::unreadable;
    }
    A.nCoarse = A.level.empty() ? nFine : A.level.front().nCoarse;
    A.nCoarseFaces = A.level.empty() ? 0 : A.level.front().nCoarseFaces;
    finalizeAMG(A, nFine);
    return AMGCacheRead::loaded;
}

} // namespace brae
