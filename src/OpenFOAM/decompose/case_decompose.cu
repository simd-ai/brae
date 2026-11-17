// decomposePar / reconstructPar, for the cases brae's multi-GPU solver runs. See case_decompose.cuh for what is and is not done.
// The mesh half is a transcription of OpenFOAM v2606's applications/utilities/parallelProcessing/decomposePar
// (domainDecompositionMesh.C: decomposeMesh; domainDecomposition.C: writeComplete, which turns the addressing into meshes).
#include "decompose_detail.cuh"

#include "foam_token_reader.cuh"
#include "fv_geometry.cuh"
#include "scotch_decomposition.cuh"
#include "start_time.cuh"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <cerrno>
#include <csignal>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace brae {
namespace decompose {
namespace detail {

std::string numberText(double v)
{
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    return std::string(buf, r.ptr);
}

std::string quoted(const std::string& s)
{
    std::string o = "\"";
    for (const char c : s)
    {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o + '"';
}

std::string tokenText(const std::string& tok)
{
    double v = 0;
    if (TokenStream::decodeBinaryScalar(tok, v)) return numberText(v);
    if (tok == "(" || tok == ")" || tok == "[" || tok == "]" || tok == "{" || tok == "}") return tok;
    if (tok.size() > 1 && tok.front() == '[' && tok.back() == ']') return tok;     // dimensions, if the reader hands them over whole
    for (const char c : tok)
    {
        if (std::isspace(static_cast<unsigned char>(c)) || c == ';' || c == '"' || c == '(' || c == ')' || c == '|' || c == '*'
            || c == '{' || c == '}' || c == '[' || c == ']' || c == '^' || c == '\\')
        {
            return quoted(tok);
        }
    }
    return tok.empty() ? std::string("\"\"") : tok;
}

std::string foamHeader(const std::string& cls, const std::string& object, const std::string& location, const std::string& note)
{
    std::string h = "/*--------------------------------*- C++ -*----------------------------------*\\\n"
                    "| brae: written by `brae decompose` / `brae reconstruct`, in OpenFOAM's layout |\n"
                    "\\*---------------------------------------------------------------------------*/\n"
                    "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    arch        \"LSB;label=32;scalar=64\";\n";
    if (!note.empty()) h += "    note        " + quoted(note) + ";\n";
    h += "    class       " + cls + ";\n";
    if (!location.empty()) h += "    location    " + quoted(location) + ";\n";
    h += "    object      " + object + ";\n}\n"
         "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n";
    return h;
}

std::vector<label> readLabelListFile(const std::string& path)
{
    if (foamFormat(path) == "binary") return readBinaryLabelList(path);
    TokenStream ts(path);
    const label n = ts.nextLabel();
    std::vector<label> v;
    v.reserve(static_cast<std::size_t>(n));
    if (n == 0)
    {
        return v;
    }
    ts.expect("(");
    for (label i = 0; i < n; ++i) v.push_back(ts.nextLabel());
    ts.expect(")");
    return v;
}

void writeLabelListFile(const std::string& path, const std::string& object, const std::vector<label>& v, const std::string& note)
{
    std::ofstream o(path);
    if (!o) throw std::runtime_error("brae decompose: cannot write " + path);
    const std::string location = fs::path(path).parent_path().filename().string() == "polyMesh" ? "constant/polyMesh" : "";
    o << foamHeader("labelList", object, location, note) << v.size() << "\n(\n";
    for (const label x : v) o << x << '\n';
    o << ")\n";
}

std::vector<std::string> timeDirectories(const std::string& dir)
{
    std::vector<std::pair<double, std::string>> t;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec))
    {
        if (!e.is_directory()) continue;
        const std::string nm = e.path().filename().string();
        char* end = nullptr;
        const double v = std::strtod(nm.c_str(), &end);
        if (end == nm.c_str() || *end != '\0') continue;
        t.emplace_back(v, nm);
    }
    std::sort(t.begin(), t.end());
    std::vector<std::string> out;
    for (auto& p : t) out.push_back(p.second);
    return out;
}

}   // namespace detail

using namespace detail;

// ------------------------------------------------------------------------------------------------------------------------------
// which processor each cell goes to
// ------------------------------------------------------------------------------------------------------------------------------
namespace {

std::vector<vector> cellCentres(const PrimitiveMesh& m)
{
    FvGeometry g;
    g.build(m);
    return g.C();
}

double coordinate(const vector& v, int d) { return d == 0 ? v.x : d == 1 ? v.y : v.z; }

// geomDecomp::adjustPoints (v2606): the points are turned by a small rotation (delta, default 0.001) before they are sorted, so the
// cells of a regular mesh that share a coordinate are told apart by the others. localPosition: R^T applied to the point.
vector rotatePoint(const vector& p, double delta)
{
    if (delta == 0.0) return p;
    const double d = 1 - 0.5 * delta * delta, d2 = d * d, a = delta, a2 = a * a;
    const double R[3][3] = {{d2, -a * d, a},
                            {a * d - a2 * d, a * a2 + d2, -2 * a * d},
                            {a * d2 + a2, a * d - a2 * d, d2 - a2}};
    const double x[3] = {p.x, p.y, p.z};
    double o[3];
    for (int i = 0; i < 3; ++i) o[i] = R[0][i] * x[0] + R[1][i] * x[1] + R[2][i] * x[2];
    return vector{o[0], o[1], o[2]};
}

// assignToProcessorGroup (simpleGeomDecomp.C): the first (size mod n) groups take one cell more
std::vector<label> groupOf(std::size_t size, int nGroups)
{
    std::vector<label> g(size);
    const std::size_t jump = size / static_cast<std::size_t>(nGroups);
    const std::size_t first = size - jump * static_cast<std::size_t>(nGroups);
    std::size_t at = 0;
    for (int j = 0; j < nGroups; ++j)
    {
        const std::size_t n = j < static_cast<int>(first) ? jump + 1 : jump;
        for (std::size_t k = 0; k < n; ++k) g[at++] = j;
    }
    return g;
}

// OpenFOAM's `simple`: for each direction, all the cells sorted along it and cut into n equal parts (simpleGeomDecomp::decomposeOneProc,
// unweighted): the three part numbers are independent of each other. x fastest.
std::vector<label> simpleDecomposition(const std::vector<vector>& centres, const int n[3], double delta)
{
    const std::size_t nC = centres.size();
    std::vector<vector> pts(nC);
    for (std::size_t i = 0; i < nC; ++i) pts[i] = rotatePoint(centres[i], delta);
    std::vector<label> idx(nC), code(nC, 0);
    for (std::size_t i = 0; i < nC; ++i) idx[i] = static_cast<label>(i);
    int stride = 1;
    for (int d = 0; d < 3; ++d)
    {
        std::stable_sort(idx.begin(), idx.end(), [&](label a, label b)
        { return coordinate(pts[static_cast<std::size_t>(a)], d) < coordinate(pts[static_cast<std::size_t>(b)], d); });
        const std::vector<label> g = groupOf(nC, n[d]);
        for (std::size_t i = 0; i < nC; ++i) code[static_cast<std::size_t>(idx[i])] += g[i] * stride;
        stride *= n[d];
    }
    return code;
}

// `hierarchical`: the cells cut along the first direction of `order` into n equal parts, each of those along the next, and so on.
// brae's own nested split: OpenFOAM's hierarchGeomDecomp cuts by a weighted binary search and is not reproduced, so a cell next to a
// cut may fall on the other side of it.
std::vector<label> hierarchicalDecomposition(const std::vector<vector>& centres, const int n[3], const int order[3], double delta)
{
    const std::size_t nC = centres.size();
    std::vector<vector> pts(nC);
    for (std::size_t i = 0; i < nC; ++i) pts[i] = rotatePoint(centres[i], delta);
    std::vector<std::vector<label>> groups(1);
    groups[0].resize(nC);
    for (std::size_t i = 0; i < nC; ++i) groups[0][i] = static_cast<label>(i);
    std::vector<label> code(nC, 0);
    // the parts are numbered with the FIRST direction of `order` slowest
    int weight[3];
    {
        int w = 1;
        for (int k = 2; k >= 0; --k) { weight[order[k]] = w; w *= n[order[k]]; }
    }
    for (int pass = 0; pass < 3; ++pass)
    {
        const int d = order[pass];
        std::vector<std::vector<label>> next;
        for (auto& gr : groups)
        {
            std::stable_sort(gr.begin(), gr.end(), [&](label a, label b)
            { return coordinate(pts[static_cast<std::size_t>(a)], d) < coordinate(pts[static_cast<std::size_t>(b)], d); });
            const std::vector<label> g = groupOf(gr.size(), n[d]);
            std::size_t begin = 0;
            for (int s = 0; s < n[d]; ++s)
            {
                std::size_t end = begin;
                while (end < gr.size() && g[end] == s) ++end;
                next.emplace_back(gr.begin() + static_cast<std::ptrdiff_t>(begin), gr.begin() + static_cast<std::ptrdiff_t>(end));
                for (std::size_t i = begin; i < end; ++i) code[static_cast<std::size_t>(gr[i])] += s * weight[d];
                begin = end;
            }
        }
        groups.swap(next);
    }
    return code;
}

std::vector<label> readManualDecomposition(const std::string& file, label nCells, int nProcs)
{
    std::vector<label> v = readLabelListFile(file);
    if (static_cast<label>(v.size()) != nCells)
        throw std::runtime_error("brae decompose: the manual decomposition " + file + " has " + std::to_string(v.size())
                                 + " entries and the mesh has " + std::to_string(nCells) + " cells.");
    for (const label p : v)
        if (p < 0 || p >= nProcs)
            throw std::runtime_error("brae decompose: the manual decomposition " + file + " names processor " + std::to_string(p)
                                     + ", outside 0.." + std::to_string(nProcs - 1) + ".");
    return v;
}

}   // namespace

static std::vector<label> assignCellsOn(const PrimitiveMesh& m, const std::string& caseDir, int nProcs, const std::string& methodIn,
                                        int& nProcsOut, std::string& methodOut)
{
    FoamDict d;
    const std::string dictPath = caseDir + "/system/decomposeParDict";
    const bool haveDict = fs::exists(dictPath);
    if (haveDict) d = readDict(dictPath);
    nProcsOut = nProcs > 0 ? nProcs : d.intOr("numberOfSubdomains", 0);
    if (nProcsOut < 1)
        throw std::runtime_error("brae decompose: no number of processors: give `-n N` or set numberOfSubdomains in "
                                 "system/decomposeParDict.");
    if (nProcsOut == 1)
    {
        methodOut = "single";
        return std::vector<label>(static_cast<std::size_t>(m.nCells()), 0);
    }
    methodOut = !methodIn.empty() ? methodIn : d.wordOr("method", "scotch");
    if (methodOut == "scotch")
    {
        return scotchDecompose(m, nProcsOut);
    }
    if (methodOut == "simple" || methodOut == "hierarchical")
    {
        int n[3] = {nProcsOut, 1, 1};
        int order[3] = {0, 1, 2};
        const FoamDict* c = d.subDict(methodOut + "Coeffs");
        if (!c) c = &d;
        const std::vector<scalar> nv = c->scalarListOr("n", {});
        if (nv.size() == 3)
        {
            n[0] = static_cast<int>(nv[0]); n[1] = static_cast<int>(nv[1]); n[2] = static_cast<int>(nv[2]);
        }
        else if (methodOut == "simple" && haveDict)
        {
            throw std::runtime_error("brae decompose: method simple needs `n (nx ny nz)` in simpleCoeffs.");
        }
        if (n[0] * n[1] * n[2] != nProcsOut)
            throw std::runtime_error("brae decompose: n (" + std::to_string(n[0]) + " " + std::to_string(n[1]) + " "
                                     + std::to_string(n[2]) + ") does not multiply to " + std::to_string(nProcsOut) + " processors.");
        if (methodOut == "hierarchical")
        {
            const std::string ord = c->wordOr("order", "xyz");
            if (ord.size() != 3)
                throw std::runtime_error("brae decompose: hierarchical `order " + ord + "` is not three of x, y, z.");
            for (int i = 0; i < 3; ++i)
            {
                const char ch = ord[static_cast<std::size_t>(i)];
                if (ch != 'x' && ch != 'y' && ch != 'z')
                    throw std::runtime_error("brae decompose: hierarchical `order " + ord + "` is not three of x, y, z.");
                order[i] = ch - 'x';
            }
        }
        const double delta = c->scalarOr("delta", 0.001);
        const std::vector<vector> centres = cellCentres(m);
        return methodOut == "simple" ? simpleDecomposition(centres, n, delta) : hierarchicalDecomposition(centres, n, order, delta);
    }
    if (methodOut == "manual")
    {
        const FoamDict* c = d.subDict("manualCoeffs");
        std::string file = c ? c->wordOr("dataFile", "") : std::string();
        if (file.empty())
            throw std::runtime_error("brae decompose: method manual needs `dataFile` in manualCoeffs.");
        if (file[0] != '/') file = caseDir + "/" + file;
        return readManualDecomposition(file, m.nCells(), nProcsOut);
    }
    throw std::runtime_error("brae decompose: decomposition method `" + methodOut + "` is not one brae does (scotch, simple, "
                             "hierarchical, manual).");
}

std::vector<label> assignCells(const std::string& caseDir, int nProcs, const std::string& method)
{
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    int n = 0;
    std::string used;
    return assignCellsOn(m, caseDir, nProcs, method, n, used);
}

// ------------------------------------------------------------------------------------------------------------------------------
// the addressing (domainDecompositionMesh.C: decomposeMesh)
// ------------------------------------------------------------------------------------------------------------------------------
namespace {

void refuseUnsupportedPatches(const PrimitiveMesh& m)
{
    for (const PatchInfo& p : m.patches())
    {
        const std::string& t = p.type;
        if (t == "patch" || t == "wall" || t == "empty" || t == "symmetryPlane" || t == "symmetry" || t == "wedge") continue;
        throw std::runtime_error("brae decompose: patch `" + p.name + "` is of type `" + t + "`. Only patch, wall, empty, "
                                 "symmetryPlane, symmetry and wedge patches are decomposed (cyclic, AMI and processor patches "
                                 "are not).");
    }
}

Plan makePlan(const PrimitiveMesh& m, const std::vector<label>& cellToProc, int nProcs)
{
    if (static_cast<label>(cellToProc.size()) != m.nCells())
        throw std::runtime_error("brae decompose: the decomposition has " + std::to_string(cellToProc.size()) + " entries for "
                                 + std::to_string(m.nCells()) + " cells.");
    Plan plan;
    plan.nProcs = nProcs;
    plan.cellToProc = cellToProc;
    plan.proc.resize(static_cast<std::size_t>(nProcs));
    GlobalLayout& g = plan.global;
    g.nCells = m.nCells();
    g.nPoints = m.nPoints();
    g.nFaces = m.nFaces();
    g.nInternalFaces = m.nInternalFaces();
    g.owner = m.owner();
    g.neighbour = m.neighbour();
    for (const PatchInfo& p : m.patches())
    {
        PatchLayout l;
        l.name = p.name; l.type = p.type; l.inGroups = p.inGroups; l.start = p.start; l.size = p.size;
        g.patches.push_back(l);
    }
    const std::vector<label>& owner = m.owner();
    const std::vector<label>& nbr = m.neighbour();

    // cells: ascending global order within a processor
    std::vector<label> cellLocal(static_cast<std::size_t>(m.nCells()), -1);
    for (label c = 0; c < m.nCells(); ++c)
    {
        const label p = cellToProc[static_cast<std::size_t>(c)];
        if (p < 0 || p >= nProcs)
            throw std::runtime_error("brae decompose: cell " + std::to_string(c) + " is assigned to processor " + std::to_string(p));
        cellLocal[static_cast<std::size_t>(c)] = static_cast<label>(plan.proc[static_cast<std::size_t>(p)].cells.size());
        plan.proc[static_cast<std::size_t>(p)].cells.push_back(c);
    }

    // faces. 1: internal to a processor, ascending global face order, no turning index
    for (label f = 0; f < m.nInternalFaces(); ++f)
    {
        const label po = cellToProc[static_cast<std::size_t>(owner[static_cast<std::size_t>(f)])];
        const label pn = cellToProc[static_cast<std::size_t>(nbr[static_cast<std::size_t>(f)])];
        if (po == pn) plan.proc[static_cast<std::size_t>(po)].faces.push_back(f + 1);
    }
    for (ProcLayout& pl : plan.proc) pl.nInternal = static_cast<label>(pl.faces.size());

    // 2: each original patch's faces, where the cell next to the face lives. Every processor lists every patch, empty or not.
    for (ProcLayout& pl : plan.proc)
    {
        pl.patches.assign(g.patches.size(), PatchLayout());
        pl.boundary.resize(g.patches.size());
        pl.patchNbr.assign(g.patches.size(), -1);
        for (std::size_t k = 0; k < g.patches.size(); ++k)
        {
            pl.patches[k] = g.patches[k];
            pl.patches[k].start = static_cast<label>(pl.faces.size());
            pl.patches[k].size = 0;
            pl.boundary[k] = static_cast<label>(k);
        }
    }
    for (std::size_t k = 0; k < g.patches.size(); ++k)
    {
        for (label i = 0; i < g.patches[k].size; ++i)
        {
            const label f = g.patches[k].start + i;
            ProcLayout& pl = plan.proc[static_cast<std::size_t>(cellToProc[static_cast<std::size_t>(owner[static_cast<std::size_t>(f)])])];
            pl.faces.push_back(f + 1);
            ++pl.patches[k].size;
        }
        // a processor that got no face of this patch starts it where the faces of the patches before it end
        for (ProcLayout& pl : plan.proc)
            if (pl.patches[k].size == 0) pl.patches[k].start = static_cast<label>(pl.faces.size());
    }
    // the start of each patch is where its first face went in; recompute from the counts so an empty patch sits after its
    // predecessor, as OpenFOAM's procPatchStartIndex_ does
    for (ProcLayout& pl : plan.proc)
    {
        label at = pl.nInternal;
        for (PatchLayout& p : pl.patches) { p.start = at; at += p.size; }
    }

    // 3: the processor patches, from the internal faces between processors, ascending global face order
    std::vector<std::map<label, std::vector<label>>> inter(static_cast<std::size_t>(nProcs));
    long long nInterFaces = 0;
    for (label f = 0; f < m.nInternalFaces(); ++f)
    {
        const label po = cellToProc[static_cast<std::size_t>(owner[static_cast<std::size_t>(f)])];
        const label pn = cellToProc[static_cast<std::size_t>(nbr[static_cast<std::size_t>(f)])];
        if (po == pn) continue;
        inter[static_cast<std::size_t>(po)][pn].push_back(f + 1);        // the owner's side: as the face is
        inter[static_cast<std::size_t>(pn)][po].push_back(-(f + 1));     // the neighbour's side: turned
        ++nInterFaces;
    }
    for (int p = 0; p < nProcs; ++p)
    {
        ProcLayout& pl = plan.proc[static_cast<std::size_t>(p)];
        for (const auto& kv : inter[static_cast<std::size_t>(p)])            // std::map: neighbours ascending
        {
            PatchLayout l;
            l.name = "procBoundary" + std::to_string(p) + "to" + std::to_string(kv.first);
            l.type = "processor";
            l.inGroups = {"processor"};
            l.start = static_cast<label>(pl.faces.size());
            l.size = static_cast<label>(kv.second.size());
            pl.patches.push_back(l);
            pl.boundary.push_back(-1);
            pl.patchNbr.push_back(kv.first);
            for (const label a : kv.second) pl.faces.push_back(a);
        }
    }
    (void)nInterFaces;

    // points: the ones the processor's faces use, ascending
    for (ProcLayout& pl : plan.proc)
    {
        std::vector<char> used(static_cast<std::size_t>(m.nPoints()), 0);
        for (const label a : pl.faces)
        {
            const label f = (a > 0 ? a : -a) - 1;
            for (label k = 0; k < m.faceSize(f); ++k) used[static_cast<std::size_t>(m.faceVert(f, k))] = 1;
        }
        for (label pt = 0; pt < m.nPoints(); ++pt)
            if (used[static_cast<std::size_t>(pt)]) pl.points.push_back(pt);
    }
    return plan;
}

// ---- zones ------------------------------------------------------------------------------------------------------------------
using ZoneList = std::vector<std::pair<std::string, std::vector<label>>>;

// A polyMesh file as OpenFOAM may have written it: `name` or `name.gz`
std::string foamFilePath(const std::string& base)
{
    return (!fs::exists(base) && fs::exists(base + ".gz")) ? base + ".gz" : base;
}

// Does a faceZones / pointZones file hold no member at all? Its zones' lists are all `List<label> 0` (or the file is `0()`). A file
// this cannot read is taken to have members. (The file is ASCII text up to its first binary list, and an empty list has no bytes.)
bool zoneFileIsEmptyOfMembers(const std::string& path, bool& hasZones)
{
    const std::vector<char> bytes = gzSlurp(path);
    const std::string text(bytes.begin(), bytes.end());
    hasZones = false;
    const std::size_t banner = text.find("// * * *");
    if (banner == std::string::npos) return false;
    std::size_t i = text.find('\n', banner);
    while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
    if (i >= text.size()) return false;
    const long nZones = std::atol(text.c_str() + i);
    hasZones = nZones > 0;
    if (nZones == 0) return true;
    for (const char* key : {"List<label>", "List<bool>"})
    {
        for (std::size_t at = text.find(key); at != std::string::npos; at = text.find(key, at + 1))
        {
            std::size_t k = at + std::strlen(key);
            while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k]))) ++k;
            if (k >= text.size() || !std::isdigit(static_cast<unsigned char>(text[k])) || std::atol(text.c_str() + k) != 0) return false;
        }
    }
    return text.find("List<label>") != std::string::npos;
}

ZoneList readCellZonesInOrder(const std::string& polyMeshDir)
{
    const std::string path = polyMeshDir + "/cellZones";
    ZoneList out;
    if (!fs::exists(path) && !fs::exists(path + ".gz")) return out;
    if (foamFormat(path) == "binary") return readBinaryCellZones(path);
    TokenStream ts(path);
    const label nz = ts.nextLabel();
    if (nz == 0) return out;
    ts.expect("(");
    for (label z = 0; z < nz; ++z)
    {
        const std::string name = ts.next();
        ts.expect("{");
        std::vector<label> cells;
        while (ts.peek() != "}")
        {
            const std::string key = ts.next();
            if (key == "cellLabels")
            {
                ts.next();                                   // List<label>
                const label n = ts.nextLabel();
                if (n > 0)
                {
                    ts.expect("(");
                    for (label i = 0; i < n; ++i) cells.push_back(ts.nextLabel());
                    ts.expect(")");
                }
                ts.expect(";");
            }
            else
            {
                while (ts.peek() != ";") ts.next();
                ts.expect(";");
            }
        }
        ts.expect("}");
        out.emplace_back(name, std::move(cells));
    }
    return out;
}

// ---- writing -----------------------------------------------------------------------------------------------------------------
void writeProcessorMesh(const std::string& caseDir, int p, const Plan& plan, const PrimitiveMesh& m, const ZoneList& cellZones)
{
    const ProcLayout& pl = plan.proc[static_cast<std::size_t>(p)];
    const std::string dir = caseDir + "/processor" + std::to_string(p) + "/constant/polyMesh";
    fs::create_directories(dir);

    std::vector<label> pointLocal(static_cast<std::size_t>(m.nPoints()), -1);
    for (std::size_t i = 0; i < pl.points.size(); ++i) pointLocal[static_cast<std::size_t>(pl.points[i])] = static_cast<label>(i);
    std::vector<label> cellLocal(static_cast<std::size_t>(m.nCells()), -1);
    for (std::size_t i = 0; i < pl.cells.size(); ++i) cellLocal[static_cast<std::size_t>(pl.cells[i])] = static_cast<label>(i);

    const std::string note = "nPoints:" + std::to_string(pl.points.size()) + "  nCells:" + std::to_string(pl.cells.size())
                           + "  nFaces:" + std::to_string(pl.faces.size()) + "  nInternalFaces:" + std::to_string(pl.nInternal);

    {   // points
        std::ofstream o(dir + "/points");
        o << foamHeader("vectorField", "points", "constant/polyMesh") << pl.points.size() << "\n(\n";
        for (const label g : pl.points)
        {
            const vector& v = m.points()[static_cast<std::size_t>(g)];
            o << '(' << numberText(v.x) << ' ' << numberText(v.y) << ' ' << numberText(v.z) << ")\n";
        }
        o << ")\n";
    }
    std::vector<label> owner(pl.faces.size()), neighbour(static_cast<std::size_t>(pl.nInternal));
    {   // faces, with owner and neighbour made from the same list
        std::ofstream o(dir + "/faces");
        o << foamHeader("faceList", "faces", "constant/polyMesh") << pl.faces.size() << "\n(\n";
        for (std::size_t k = 0; k < pl.faces.size(); ++k)
        {
            const label a = pl.faces[k];
            const label f = (a > 0 ? a : -a) - 1;
            const label n = m.faceSize(f);
            std::vector<label> v(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
            {
                // reverseFace: the first vertex stays, the others run the other way
                const label src = (a > 0 || i == 0) ? i : n - i;
                v[static_cast<std::size_t>(i)] = pointLocal[static_cast<std::size_t>(m.faceVert(f, src))];
            }
            o << n << '(';
            for (label i = 0; i < n; ++i) o << (i ? " " : "") << v[static_cast<std::size_t>(i)];
            o << ")\n";
            if (a > 0)
            {
                owner[k] = cellLocal[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(f)])];
                if (static_cast<label>(k) < pl.nInternal)
                    neighbour[k] = cellLocal[static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(f)])];
            }
            else
            {
                owner[k] = cellLocal[static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(f)])];
            }
        }
        o << ")\n";
    }
    writeLabelListFile(dir + "/owner", "owner", owner, note);
    writeLabelListFile(dir + "/neighbour", "neighbour", neighbour, note);
    {   // boundary
        std::ofstream o(dir + "/boundary");
        o << foamHeader("polyBoundaryMesh", "boundary", "constant/polyMesh") << pl.patches.size() << "\n(\n";
        for (std::size_t k = 0; k < pl.patches.size(); ++k)
        {
            const PatchLayout& pt = pl.patches[k];
            o << "    " << pt.name << "\n    {\n        type            " << pt.type << ";\n";
            if (!pt.inGroups.empty())
            {
                o << "        inGroups        " << pt.inGroups.size() << '(';
                for (std::size_t i = 0; i < pt.inGroups.size(); ++i) o << (i ? " " : "") << pt.inGroups[i];
                o << ");\n";
            }
            o << "        nFaces          " << pt.size << ";\n        startFace       " << pt.start << ";\n";
            if (pt.type == "processor")
            {
                o << "        matchTolerance  0.0001;\n        transform       unknown;\n        myProcNo        " << p
                  << ";\n        neighbProcNo    " << pl.patchNbr[k] << ";\n";
            }
            o << "    }\n";
        }
        o << ")\n";
    }
    writeLabelListFile(dir + "/cellProcAddressing", "cellProcAddressing", pl.cells);
    writeLabelListFile(dir + "/faceProcAddressing", "faceProcAddressing", pl.faces);
    writeLabelListFile(dir + "/pointProcAddressing", "pointProcAddressing", pl.points);
    writeLabelListFile(dir + "/boundaryProcAddressing", "boundaryProcAddressing", pl.boundary);

    if (!cellZones.empty())
    {   // decomposePar: a zone keeps the processor's cells that are in it, in ascending local order (domainDecomposition.C:691-715)
        std::ofstream o(dir + "/cellZones");
        o << foamHeader("regIOobject", "cellZones", "constant/polyMesh") << cellZones.size() << "\n(\n";
        for (const auto& z : cellZones)
        {
            std::vector<char> in(static_cast<std::size_t>(m.nCells()), 0);
            for (const label c : z.second) in[static_cast<std::size_t>(c)] = 1;
            std::vector<label> local;
            for (std::size_t i = 0; i < pl.cells.size(); ++i)
                if (in[static_cast<std::size_t>(pl.cells[i])]) local.push_back(static_cast<label>(i));
            o << z.first << "\n{\n    type cellZone;\n    cellLabels List<label> " << local.size() << '(';
            for (std::size_t i = 0; i < local.size(); ++i) o << (i ? " " : "") << local[i];
            o << ");\n}\n";
        }
        o << ")\n";
    }
}

std::string timeFor(const std::string& caseDir)
{
    const std::string controlPath = caseDir + "/system/controlDict";
    if (!fs::exists(controlPath))
        throw std::runtime_error("brae decompose: " + controlPath + " does not exist.");
    setCaseEnvironment(caseDir);
    const FoamDict controlDict = readDict(controlPath);
    return resolveStartTime(caseDir, controlDict).name;
}

void say(bool quiet, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void say(bool quiet, const char* fmt, ...)
{
    if (quiet) return;
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
}

}   // namespace

std::string startTimeName(const std::string& caseDir) { return timeFor(caseDir); }

int processorDirectories(const std::string& caseDir)
{
    int n = 0;
    while (fs::is_directory(caseDir + "/processor" + std::to_string(n))) ++n;
    return n;
}

Report decomposeCaseWith(const std::string& caseDir, const std::vector<label>& cellToProc, int nProcs, const Options& opt)
{
    if (!fs::is_directory(caseDir + "/constant/polyMesh"))
        throw std::runtime_error("brae decompose: " + caseDir + "/constant/polyMesh does not exist.");
    if (processorDirectories(caseDir) > 0 || fs::exists(caseDir + "/processor0"))
    {
        if (!opt.force)
            throw std::runtime_error("brae decompose: " + caseDir + " already has processor directories. Remove them, or give "
                                     "-force to replace them.");
        for (int p = 0; fs::exists(caseDir + "/processor" + std::to_string(p)); ++p)
            fs::remove_all(caseDir + "/processor" + std::to_string(p));
    }
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    refuseUnsupportedPatches(m);
    const std::string pm = caseDir + "/constant/polyMesh";
    std::vector<std::string> emptyZoneFiles;          // faceZones / pointZones with zones that hold nothing: carried over as they are
    for (const char* z : {"faceZones", "pointZones"})
    {
        const std::string zpath = foamFilePath(pm + "/" + z);
        if (!fs::exists(zpath)) continue;
        bool hasZones = false;
        if (!zoneFileIsEmptyOfMembers(zpath, hasZones))
            throw std::runtime_error(std::string("constant/polyMesh/") + z + " has members, and only cellZones are decomposed "
                                     "(a face or point zone with members is not).");
        if (hasZones) emptyZoneFiles.push_back(fs::path(zpath).filename().string());
    }
    {
        std::string extras;
        for (const char* extra : {"cellLevel", "pointLevel", "level0Edge", "refinementHistory"})
            if (fs::exists(pm + "/" + extra)) extras += std::string(extras.empty() ? "" : ", ") + extra;
        if (!extras.empty())
            say(opt.quiet, "brae decompose: constant/polyMesh/{%s} are not decomposed (dynamic refinement is not run under -parallel)\n", extras.c_str());
    }
    const ZoneList cellZones = readCellZonesInOrder(pm);

    const Plan plan = makePlan(m, cellToProc, nProcs);
    for (int p = 0; p < nProcs; ++p)
        if (plan.proc[static_cast<std::size_t>(p)].cells.empty())
            throw std::runtime_error("brae decompose: processor " + std::to_string(p) + " gets no cells (" + std::to_string(nProcs)
                                     + " processors for " + std::to_string(m.nCells()) + " cells); a processor with no cells cannot run.");
    Report rep;
    rep.nProcs = nProcs;
    rep.nCells = m.nCells();
    for (const ProcLayout& pl : plan.proc) rep.cellsPerProc.push_back(static_cast<long long>(pl.cells.size()));
    for (label f = 0; f < m.nInternalFaces(); ++f)
        if (cellToProc[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(f)])]
            != cellToProc[static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(f)])]) ++rep.nInterProcFaces;

    for (int p = 0; p < nProcs; ++p)
    {
        writeProcessorMesh(caseDir, p, plan, m, cellZones);
        for (const std::string& z : emptyZoneFiles)
            fs::copy_file(pm + "/" + z, caseDir + "/processor" + std::to_string(p) + "/constant/polyMesh/" + z, fs::copy_options::overwrite_existing);
    }
    say(opt.quiet, "brae decompose: %lld cells into %d processors (", rep.nCells, nProcs);
    for (int p = 0; p < nProcs; ++p) say(opt.quiet, "%s%lld", p ? " " : "", rep.cellsPerProc[static_cast<std::size_t>(p)]);
    say(opt.quiet, " cells), %lld inter-processor faces\n", rep.nInterProcFaces);

    // the time directories
    std::vector<std::string> times = opt.times;
    if (opt.allTimes)
    {
        times = timeDirectories(caseDir);
    }
    else if (times.empty())
    {
        times.push_back(timeFor(caseDir));
    }
    for (const std::string& t : times)
    {
        if (!fs::is_directory(caseDir + "/" + t))
            throw std::runtime_error("brae decompose: the time directory " + caseDir + "/" + t + " does not exist.");
        decomposeFields(caseDir, plan, t, rep.fields, opt.quiet);
        if (opt.copyUniform && fs::is_directory(caseDir + "/" + t + "/uniform"))
            for (int p = 0; p < nProcs; ++p)
                fs::copy(caseDir + "/" + t + "/uniform", caseDir + "/processor" + std::to_string(p) + "/" + t + "/uniform",
                         fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        rep.times.push_back(t);
    }
    return rep;
}

Report decomposeCase(const std::string& caseDir, const Options& opt)
{
    if (!fs::is_directory(caseDir + "/constant/polyMesh"))
        throw std::runtime_error("brae decompose: " + caseDir + "/constant/polyMesh does not exist.");
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    refuseUnsupportedPatches(m);
    int n = 0;
    std::string method;
    const std::vector<label> cellToProc = assignCellsOn(m, caseDir, opt.nProcs, opt.method, n, method);
    Report rep = decomposeCaseWith(caseDir, cellToProc, n, opt);
    rep.method = method;
    return rep;
}

// ------------------------------------------------------------------------------------------------------------------------------
// reconstructPar
// ------------------------------------------------------------------------------------------------------------------------------
ReconstructReport reconstructCase(const std::string& caseDir, const ReconstructOptions& opt)
{
    const int nProcs = processorDirectories(caseDir);
    if (nProcs == 0) throw std::runtime_error("brae reconstruct: " + caseDir + " has no processor directories.");

    // the global mesh's patches, and the addressing of every processor
    PrimitiveMesh gm;
    gm.readBoundary(caseDir + "/constant/polyMesh");
    GlobalLayout global;
    for (const PatchInfo& p : gm.patches())
    {
        PatchLayout l;
        l.name = p.name; l.type = p.type; l.inGroups = p.inGroups; l.start = p.start; l.size = p.size;
        global.patches.push_back(l);
    }
    std::vector<ProcLayout> proc(static_cast<std::size_t>(nProcs));
    for (int p = 0; p < nProcs; ++p)
    {
        const std::string dir = caseDir + "/processor" + std::to_string(p) + "/constant/polyMesh";
        for (const char* f : {"cellProcAddressing", "faceProcAddressing", "boundaryProcAddressing"})
            if (!fs::exists(dir + "/" + f))
                throw std::runtime_error("brae reconstruct: " + dir + "/" + f + " does not exist: it is written by decomposePar and "
                                         "`brae decompose`, and says which global cells and faces a processor's are.");
        ProcLayout& pl = proc[static_cast<std::size_t>(p)];
        pl.cells = readLabelListFile(dir + "/cellProcAddressing");
        pl.faces = readLabelListFile(dir + "/faceProcAddressing");
        pl.boundary = readLabelListFile(dir + "/boundaryProcAddressing");
        PrimitiveMesh lm;
        lm.readBoundary(dir);
        for (const PatchInfo& pi : lm.patches())
        {
            PatchLayout l;
            l.name = pi.name; l.type = pi.type; l.inGroups = pi.inGroups; l.start = pi.start; l.size = pi.size;
            pl.patches.push_back(l);
        }
        if (pl.patches.size() != pl.boundary.size())
            throw std::runtime_error("brae reconstruct: " + dir + ": boundaryProcAddressing has " + std::to_string(pl.boundary.size())
                                     + " entries and the boundary file has " + std::to_string(pl.patches.size()) + " patches.");
        pl.nInternal = pl.patches.empty() ? static_cast<label>(pl.faces.size()) : pl.patches.front().start;
        global.nCells += static_cast<label>(pl.cells.size());
    }
    global.nFaces = global.patches.empty() ? 0 : global.patches.back().start + global.patches.back().size;
    global.nInternalFaces = global.patches.empty() ? 0 : global.patches.front().start;

    std::vector<std::string> times = opt.times;
    if (times.empty())
    {
        for (const std::string& t : timeDirectories(caseDir + "/processor0"))
            if (opt.force || !fs::exists(caseDir + "/" + t)) times.push_back(t);
        if (opt.latestTime && times.size() > 1) times.erase(times.begin(), times.end() - 1);
    }
    ReconstructReport rep;
    for (const std::string& t : times)
    {
        if (!fs::is_directory(caseDir + "/processor0/" + t))
            throw std::runtime_error("brae reconstruct: processor0 has no time directory " + t + ".");
        if (fs::exists(caseDir + "/" + t) && !opt.force)
            throw std::runtime_error("brae reconstruct: " + caseDir + "/" + t + " exists; give -force to replace its fields.");
        reconstructFields(caseDir, proc, global, t, rep.fields, opt.quiet);
        // reconstructPar takes the uniform/ directory (time index, cumulativeContErr) from processor0
        if (fs::is_directory(caseDir + "/processor0/" + t + "/uniform"))
            fs::copy(caseDir + "/processor0/" + t + "/uniform", caseDir + "/" + t + "/uniform",
                     fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        rep.times.push_back(t);
    }
    return rep;
}

// ------------------------------------------------------------------------------------------------------------------------------
// `brae -parallel` on a case nobody decomposed
// ------------------------------------------------------------------------------------------------------------------------------
bool decomposedByRun(const std::string& caseDir) { return fs::exists(caseDir + "/" + kAutoMarker); }

namespace {

std::string thisHost()
{
    char h[256] = {0};
    ::gethostname(h, sizeof h - 1);
    return h;
}

// A lock whose holder is gone: the same host, and no such process. (Another host's holder cannot be told from here: it is waited for.)
bool lockIsStale(const std::string& lock)
{
    std::ifstream f(lock);
    std::string host;
    long pid = 0;
    if (!(f >> host >> pid) || pid <= 0 || host != thisHost()) return false;
    return ::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH;
}

}   // namespace

bool decomposeForRun(const std::string& caseDir, int nProcs, bool quiet)
{
    const std::string lock = caseDir + "/.brae_decomposing";
    const std::string failed = caseDir + "/.brae_decompose_failed";
    const auto began = std::chrono::steady_clock::now();
    long long last = 0;
    for (;;)
    {
        // a finished decomposition, somebody's: nothing to do. (While the lock is there it is not finished: the directories appear
        // one at a time.)
        if (processorDirectories(caseDir) >= 1 && !fs::exists(lock)) return false;
        // The first process to create the lock decomposes; the others wait for it. O_EXCL is atomic on the file systems this is run on.
        const int fd = ::open(lock.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd >= 0)
        {
            const std::string who = thisHost() + " " + std::to_string(static_cast<long>(::getpid())) + "\n";
            if (::write(fd, who.data(), who.size()) < 0) { /* the lock only loses its owner's name */ }
            ::close(fd);
            std::error_code ec;
            fs::remove(failed, ec);
            try
            {
                if (processorDirectories(caseDir) >= 1)                 // another run finished while this one was starting
                {
                    fs::remove(lock, ec);
                    return false;
                }
                Options opt;
                opt.nProcs = nProcs;
                opt.quiet = quiet;
                opt.copyUniform = true;     // a run that starts from an earlier time keeps its time index
                std::printf("brae: -parallel on a case with no processor directories: decomposing it into %d (brae decompose)\n", nProcs);
                std::fflush(stdout);
                decomposeCase(caseDir, opt);
                // what a run's end looks at to know brae made these directories, and so puts the case back together
                std::ofstream(caseDir + "/" + kAutoMarker) << "decomposed by brae for a -parallel run\n";
            }
            catch (const std::exception& e)
            {
                std::ofstream(failed) << e.what() << "\n";
                for (int p = 0; fs::exists(caseDir + "/processor" + std::to_string(p)); ++p)
                    fs::remove_all(caseDir + "/processor" + std::to_string(p));      // a half-made decomposition must not pass for one
                fs::remove(lock, ec);
                throw;
            }
            fs::remove(lock, ec);
            return true;
        }
        if (errno != EEXIST)
            throw std::runtime_error("cannot decompose the case for this run: " + lock + " cannot be created (" + std::strerror(errno)
                                     + "). Decompose it where it can be written (`brae decompose`), or run on a case with processor directories.");
        // waiting for whoever holds the lock
        while (fs::exists(lock))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const long long waited = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - began).count();
            if (waited >= last + 60)
            {
                last = waited;
                if (!quiet) std::printf("brae: waiting for the decomposition of %s (%lld s)\n", caseDir.c_str(), waited);
                std::fflush(stdout);
            }
            if (lockIsStale(lock))
            {
                // the process that was decomposing is gone (killed). Every waiter sees it at about the same time, so each waits a
                // different moment, looks again, and takes the lock over by renaming it: one rename succeeds, and that one clears the
                // half-made directories. The others find the new lock and wait on it.
                std::this_thread::sleep_for(std::chrono::milliseconds(37 * (static_cast<long>(::getpid()) % 13)));
                if (lockIsStale(lock))
                {
                    const std::string gone = lock + ".gone." + std::to_string(static_cast<long>(::getpid()));
                    if (::rename(lock.c_str(), gone.c_str()) == 0)
                    {
                        std::printf("brae: the decomposition of %s was abandoned by a process that is gone; starting it again\n", caseDir.c_str());
                        std::fflush(stdout);
                        for (int p = 0; fs::exists(caseDir + "/processor" + std::to_string(p)); ++p)
                            fs::remove_all(caseDir + "/processor" + std::to_string(p));
                        std::error_code ec;
                        fs::remove(gone, ec);
                    }
                }
                break;
            }
            if (waited > 6 * 3600)
                throw std::runtime_error("brae: gave up after six hours waiting for the decomposition of " + caseDir + ".");
        }
        if (fs::exists(failed))
        {
            std::ifstream f(failed);
            std::string why((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            throw std::runtime_error("the decomposition of the case failed: " + why);
        }
        if (processorDirectories(caseDir) >= 1 && !fs::exists(lock)) return true;
    }
}

}   // namespace decompose
}   // namespace brae
