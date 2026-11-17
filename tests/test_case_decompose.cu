// brae decompose / brae reconstruct on cases built here (host only: no GPU, no OpenFOAM, no solver).
//
// A hexahedral block mesh is written in OpenFOAM's file layout, and its fields hold the number of the cell or face they sit on, so
// what the decomposition did to a value can be told from the value alone. A cell -> processor list is handed in (decomposeCaseWith),
// which keeps the cut out of the question; the cut itself is checked against OpenFOAM by tests/case_decompose_vs_openfoam.sh.
//
//   1. a vector, a symmTensor, a tensor and an oriented surface scalar field, with boundaryField entries named by patch, by patch group,
//      by pattern and by wildcard: every internal and patch value is the one of its own cell or face, a processor patch holds the value
//      from the other side of the face (the face's own value, negated on a reversed face, for the oriented surface field)
//   2. reconstruct gives every field back to the last bit
//   3. one processor (np 1) round-trips, and a processor without cells is refused by name
//   4. gzipped fields and zone files (OpenFOAM's writeCompression) are decomposed, not skipped
//   5. uniform/ follows with -copyUniform and not without, and reconstruct brings it back from processor0
//   6. a case that cannot be written to is refused by name, where it used to wait for a lock it could not create
//   7. empty face/point zones are carried over; a point zone with members is refused
//
//   test_case_decompose
#include "case_decompose.cuh"
#include "decompose_detail.cuh"
#include "foam_token_reader.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace brae;

static int failures = 0;
static void check(bool ok, const std::string& what)
{
    std::printf("  %s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

static std::string wr(const std::string& path, const std::string& body)
{
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream(path) << body;
    return path;
}

// ---- the mesh --------------------------------------------------------------------------------------------------------------------
constexpr int NX = 4, NY = 3, NZ = 2;
constexpr int NCELLS = NX * NY * NZ;

struct Mesh
{
    std::vector<std::array<int, 4>> faces;     // global point numbers
    std::vector<int>                owner, neighbour;
    int                             nInternal = 0;
    struct Patch { std::string name, type, groups; int start, size; };
    std::vector<Patch>              patches;
};

static int pt(int i, int j, int k) { return i + (NX + 1) * (j + (NY + 1) * k); }
static int cell(int i, int j, int k) { return i + NX * (j + NY * k); }

static Mesh buildMesh()
{
    Mesh m;
    // internal faces, upper triangular: the faces to the higher neighbours of each cell, in neighbour order (x+, y+, z+)
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const int c = cell(i, j, k);
                if (i + 1 < NX) { m.faces.push_back({pt(i + 1, j, k), pt(i + 1, j + 1, k), pt(i + 1, j + 1, k + 1), pt(i + 1, j, k + 1)}); m.owner.push_back(c); m.neighbour.push_back(cell(i + 1, j, k)); }
                if (j + 1 < NY) { m.faces.push_back({pt(i, j + 1, k), pt(i, j + 1, k + 1), pt(i + 1, j + 1, k + 1), pt(i + 1, j + 1, k)}); m.owner.push_back(c); m.neighbour.push_back(cell(i, j + 1, k)); }
                if (k + 1 < NZ) { m.faces.push_back({pt(i, j, k + 1), pt(i + 1, j, k + 1), pt(i + 1, j + 1, k + 1), pt(i, j + 1, k + 1)}); m.owner.push_back(c); m.neighbour.push_back(cell(i, j, k + 1)); }
            }
    m.nInternal = static_cast<int>(m.faces.size());
    auto patch = [&](const char* name, const char* type, const char* groups, auto&& add)
    {
        Mesh::Patch p{name, type, groups, static_cast<int>(m.faces.size()), 0};
        add();
        p.size = static_cast<int>(m.faces.size()) - p.start;
        m.patches.push_back(p);
    };
    patch("inlet", "patch", "", [&] { for (int k = 0; k < NZ; ++k) for (int j = 0; j < NY; ++j) { m.faces.push_back({pt(0, j, k), pt(0, j, k + 1), pt(0, j + 1, k + 1), pt(0, j + 1, k)}); m.owner.push_back(cell(0, j, k)); } });
    patch("outlet", "patch", "", [&] { for (int k = 0; k < NZ; ++k) for (int j = 0; j < NY; ++j) { m.faces.push_back({pt(NX, j, k), pt(NX, j + 1, k), pt(NX, j + 1, k + 1), pt(NX, j, k + 1)}); m.owner.push_back(cell(NX - 1, j, k)); } });
    patch("walls", "wall", "wall", [&] { for (int k = 0; k < NZ; ++k) for (int i = 0; i < NX; ++i) { m.faces.push_back({pt(i, 0, k), pt(i + 1, 0, k), pt(i + 1, 0, k + 1), pt(i, 0, k + 1)}); m.owner.push_back(cell(i, 0, k)); }
                                             for (int k = 0; k < NZ; ++k) for (int i = 0; i < NX; ++i) { m.faces.push_back({pt(i, NY, k), pt(i, NY, k + 1), pt(i + 1, NY, k + 1), pt(i + 1, NY, k)}); m.owner.push_back(cell(i, NY - 1, k)); } });
    patch("bottom", "wall", "wall", [&] { for (int j = 0; j < NY; ++j) for (int i = 0; i < NX; ++i) { m.faces.push_back({pt(i, j, 0), pt(i, j + 1, 0), pt(i + 1, j + 1, 0), pt(i + 1, j, 0)}); m.owner.push_back(cell(i, j, 0)); } });
    patch("top", "patch", "", [&] { for (int j = 0; j < NY; ++j) for (int i = 0; i < NX; ++i) { m.faces.push_back({pt(i, j, NZ), pt(i + 1, j, NZ), pt(i + 1, j + 1, NZ), pt(i, j + 1, NZ)}); m.owner.push_back(cell(i, j, NZ - 1)); } });
    return m;
}

static std::string header(const char* cls, const char* object, const char* location)
{
    return std::string("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class ") + cls + ";\n    location \"" + location + "\";\n    object " + object
           + ";\n}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n";
}

static void writeMesh(const std::string& dir, const Mesh& m)
{
    const std::string pm = dir + "/constant/polyMesh";
    std::ostringstream pts;
    pts << header("vectorField", "points", "constant/polyMesh") << (NX + 1) * (NY + 1) * (NZ + 1) << "\n(\n";
    for (int k = 0; k <= NZ; ++k) for (int j = 0; j <= NY; ++j) for (int i = 0; i <= NX; ++i) pts << '(' << i * 0.5 << ' ' << j * 0.25 + 0.01 * i << ' ' << k * 0.125 << ")\n";
    pts << ")\n";
    wr(pm + "/points", pts.str());
    std::ostringstream fc;
    fc << header("faceList", "faces", "constant/polyMesh") << m.faces.size() << "\n(\n";
    for (const auto& f : m.faces) fc << "4(" << f[0] << ' ' << f[1] << ' ' << f[2] << ' ' << f[3] << ")\n";
    fc << ")\n";
    wr(pm + "/faces", fc.str());
    std::ostringstream ow;
    ow << header("labelList", "owner", "constant/polyMesh") << m.owner.size() << "\n(\n";
    for (int c : m.owner) ow << c << '\n';
    ow << ")\n";
    wr(pm + "/owner", ow.str());
    std::ostringstream nb;
    nb << header("labelList", "neighbour", "constant/polyMesh") << m.neighbour.size() << "\n(\n";
    for (int c : m.neighbour) nb << c << '\n';
    nb << ")\n";
    wr(pm + "/neighbour", nb.str());
    std::ostringstream bd;
    bd << header("polyBoundaryMesh", "boundary", "constant/polyMesh") << m.patches.size() << "\n(\n";
    for (const auto& p : m.patches)
    {
        bd << "    " << p.name << "\n    {\n        type " << p.type << ";\n";
        if (!p.groups.empty()) bd << "        inGroups 1(" << p.groups << ");\n";
        bd << "        nFaces " << p.size << ";\n        startFace " << p.start << ";\n    }\n";
    }
    bd << ")\n";
    wr(pm + "/boundary", bd.str());
    wr(dir + "/system/controlDict", header("dictionary", "controlDict", "system") + "startFrom startTime;\nstartTime 0;\nendTime 1;\ndeltaT 1;\n");
}

// ---- the fields: every value is its own cell's or face's number --------------------------------------------------------------------
static std::string value(double id, int comps, double scale = 1.0)
{
    std::ostringstream o;
    if (comps > 1) o << '(';
    for (int c = 0; c < comps; ++c) o << (c ? " " : "") << scale * (id * (c + 1) + 0.5 * c);
    if (comps > 1) o << ')';
    return o.str();
}

static std::string list(const char* type, int comps, const std::vector<double>& ids, double scale = 1.0)
{
    std::ostringstream o;
    o << "nonuniform List<" << type << "> " << ids.size() << "\n(\n";
    for (double id : ids) o << value(id, comps, scale) << '\n';
    o << ")";
    return o.str();
}

static std::vector<double> iota(int from, int n)
{
    std::vector<double> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = from + i;
    return v;
}

static void writeFields(const std::string& dir, const Mesh& m, const std::string& time)
{
    const std::string d = dir + "/" + time + "/";
    auto patchIds = [&](const char* name) { for (const auto& p : m.patches) if (p.name == name) return iota(p.start, p.size); return std::vector<double>(); };
    // U: a patch by name, one by group (`wall` is the entry of `bottom`, which has none of its own), two by pattern
    std::ostringstream U2;
    U2 << header("volVectorField", "U", time.c_str()) << "dimensions [0 1 -1 0 0 0 0];\ninternalField " << list("vector", 3, iota(0, NCELLS)) << ";\nboundaryField\n{\n"
       << "    inlet\n    {\n        type fixedValue;\n        value " << list("vector", 3, patchIds("inlet")) << ";\n    }\n"
       << "    walls\n    {\n        type fixedValue;\n        value " << list("vector", 3, patchIds("walls"), 2.0) << ";\n    }\n"
       << "    wall\n    {\n        type noSlip;\n    }\n"                                                      // bottom takes the group's entry
       << "    \"(outlet|top)\"\n    {\n        type zeroGradient;\n    }\n}\n";
    wr(d + "U", U2.str());
    // S: a symmTensor, wildcard boundary
    std::ostringstream S;
    S << header("volSymmTensorField", "S", time.c_str()) << "dimensions [0 0 0 0 0 0 0];\ninternalField " << list("symmTensor", 6, iota(0, NCELLS)) << ";\nboundaryField\n{\n"
      << "    inlet\n    {\n        type fixedValue;\n        value " << list("symmTensor", 6, patchIds("inlet")) << ";\n    }\n"
      << "    \".*\"\n    {\n        type zeroGradient;\n    }\n}\n";
    wr(d + "S", S.str());
    // T: a tensor, uniform internal field
    std::ostringstream T;
    T << header("volTensorField", "T", time.c_str()) << "dimensions [0 0 0 0 0 0 0];\ninternalField uniform (1 2 3 4 5 6 7 8 9);\nboundaryField\n{\n"
      << "    \".*\"\n    {\n        type fixedValue;\n        value uniform (1 2 3 4 5 6 7 8 9);\n    }\n}\n";
    wr(d + "T", T.str());
    // phi: oriented surface scalar, every internal face and every boundary face carries its own number + 1
    std::ostringstream phi;
    phi << header("surfaceScalarField", "phi", time.c_str()) << "dimensions [0 3 -1 0 0 0 0];\noriented oriented;\ninternalField nonuniform List<scalar> " << m.nInternal << "\n(\n";
    for (int f = 0; f < m.nInternal; ++f) phi << f + 1 << '\n';
    phi << ");\nboundaryField\n{\n";
    for (const auto& p : m.patches)
        phi << "    " << p.name << "\n    {\n        type calculated;\n        value " << list("scalar", 1, iota(p.start + 1, p.size)) << ";\n    }\n";
    phi << "}\n";
    wr(d + "phi", phi.str());
}

// ---- reading a written field back (the test's own, from the tokens) ------------------------------------------------------------
struct Parsed { bool uniform = false; bool found = false; std::vector<double> v; std::string type; };

static std::vector<double> readValues(TokenStream& ts)
{
    std::vector<double> v;
    if (ts.peek() == "(")
    {
        // either a vector (a flat group) or a list of them: collect numbers until the matching close
        int depth = 0;
        do
        {
            const std::string t = ts.next();
            if (t == "(") ++depth;
            else if (t == ")") --depth;
            else v.push_back(std::strtod(t.c_str(), nullptr));
        } while (depth > 0);
    }
    else
    {
        v.push_back(ts.nextScalar());
    }
    return v;
}

// <patch> empty: the internalField; else boundaryField/<patch>/<key>
static Parsed read(const std::string& path, const std::string& patch, const std::string& key = "value")
{
    Parsed r;
    TokenStream ts(path, true);
    auto value = [&]()
    {
        const std::string mode = ts.next();
        r.found = true;
        if (mode == "uniform") { r.uniform = true; r.v = readValues(ts); return; }
        ts.next();                                              // List<type>
        const long n = ts.nextLabel();
        if (n == 0) return;
        ts.expect("(");
        for (long i = 0; i < n; ++i)
        {
            const std::vector<double> e = readValues(ts);
            r.v.insert(r.v.end(), e.begin(), e.end());
        }
        ts.expect(")");
    };
    while (!ts.eof())
    {
        const std::string t = ts.next();
        if (patch.empty() && t == "internalField") { value(); return r; }
        if (!patch.empty() && t == "boundaryField")
        {
            ts.expect("{");
            while (ts.peek() != "}")
            {
                const std::string name = ts.next();
                ts.expect("{");
                const bool mine = (name == patch);
                int depth = 1;
                while (depth > 0)
                {
                    const std::string k = ts.next();
                    if (k == "{") ++depth;
                    else if (k == "}") --depth;
                    else if (mine && depth == 1 && k == "type") { r.type = ts.next(); ts.expect(";"); }
                    else if (mine && depth == 1 && k == key) { value(); if (!ts.eof() && ts.peek() == ";") ts.next(); return r; }
                }
                if (mine) return r;
            }
        }
    }
    return r;
}

static std::vector<double> expected(const std::vector<double>& ids, int comps, double scale = 1.0)
{
    std::vector<double> v;
    for (double id : ids)
        for (int c = 0; c < comps; ++c) v.push_back(scale * (id * (c + 1) + 0.5 * c));
    return v;
}

static bool equalVec(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

static std::string P(const std::string& dir, int p, const std::string& rest) { return dir + "/processor" + std::to_string(p) + "/" + rest; }

static std::vector<label> readLabels(const std::string& path) { return decompose::detail::readLabelListFile(path); }

// ---- the tests -----------------------------------------------------------------------------------------------------------------
static std::string tmpDir(const char* name)
{
    const std::string d = (fs::temp_directory_path() / (std::string("case_decompose_") + name + "_" + std::to_string(::getpid()))).string();
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

static std::vector<label> halves()      // i < 2 -> processor 0, else 1
{
    std::vector<label> c(NCELLS);
    for (int k = 0; k < NZ; ++k) for (int j = 0; j < NY; ++j) for (int i = 0; i < NX; ++i) c[static_cast<std::size_t>(cell(i, j, k))] = i < 2 ? 0 : 1;
    return c;
}

int main()
{
    const Mesh mesh = buildMesh();
    const std::string root = tmpDir("t");
    int exitCode = 0;
    try
    {
        // ---- 1, 2 ----
        std::printf("== 1. values, patch entries by name / group / pattern / wildcard, processor patches\n");
        const std::string a = root + "/a";
        writeMesh(a, mesh);
        writeFields(a, mesh, "0");
        decompose::Options opt;
        opt.quiet = true;
        opt.times = {"0"};
        decompose::decomposeCaseWith(a, halves(), 2, opt);
        for (int p = 0; p < 2; ++p)
        {
            const std::string pm = P(a, p, "constant/polyMesh/");
            const std::vector<label> cells = readLabels(pm + "cellProcAddressing"), faces = readLabels(pm + "faceProcAddressing");
            const std::string who = "processor" + std::to_string(p) + ": ";
            std::vector<double> cid(cells.begin(), cells.end());
            check(equalVec(read(P(a, p, "0/U"), "").v, expected(cid, 3)), who + "U internal field is its cells'");
            check(equalVec(read(P(a, p, "0/S"), "").v, expected(cid, 6)), who + "S (symmTensor) internal field is its cells'");
            const Parsed t = read(P(a, p, "0/T"), "");
            check(t.uniform && equalVec(t.v, {1, 2, 3, 4, 5, 6, 7, 8, 9}), who + "T (tensor) stays uniform");
            // the processor's own patches, from its boundary file
            std::ifstream bf(pm + "boundary");
            std::string btxt((std::istreambuf_iterator<char>(bf)), std::istreambuf_iterator<char>());
            check(btxt.find("procBoundary" + std::to_string(p) + "to" + std::to_string(1 - p)) != std::string::npos, who + "has its processor patch");
            // physical patch values: inlet (named) holds the faces of this processor's cells only, in global order, each its own face number
            std::vector<double> inletIds;
            for (const auto& pt_ : mesh.patches) if (pt_.name == "inlet") for (int f = 0; f < pt_.size; ++f) if (halves()[static_cast<std::size_t>(mesh.owner[static_cast<std::size_t>(pt_.start + f)])] == p) inletIds.push_back(pt_.start + f);
            const Parsed inlet = read(P(a, p, "0/U"), "inlet");
            check(equalVec(inlet.v, expected(inletIds, 3)), who + "U on inlet: this processor's faces, each its own value");
            std::vector<double> wallIds;
            for (const auto& pt_ : mesh.patches) if (pt_.name == "walls") for (int f = 0; f < pt_.size; ++f) if (halves()[static_cast<std::size_t>(mesh.owner[static_cast<std::size_t>(pt_.start + f)])] == p) wallIds.push_back(pt_.start + f);
            check(equalVec(read(P(a, p, "0/U"), "walls").v, expected(wallIds, 3, 2.0)), who + "U on walls");
            // `wall` (the group) is the entry of `bottom`, which has no entry of its own; the pattern's entry is `outlet`'s and `top`'s
            check(read(P(a, p, "0/U"), "bottom").type == "noSlip", who + "bottom took the group's entry");
            check(read(P(a, p, "0/U"), "outlet").type == "zeroGradient" && read(P(a, p, "0/U"), "top").type == "zeroGradient", who + "outlet and top took the pattern's entry");
            check(read(P(a, p, "0/S"), "top").type == "zeroGradient" && read(P(a, p, "0/S"), "walls").type == "zeroGradient", who + "S: the wildcard's entry reaches the other patches");
            // the processor patch: a vol field holds the value of the cell across the face; phi its own face, turned over when the face is reversed
            std::vector<double> across, phiPatch;
            const int nb = 1 - p;
            // the faces of the processor patch, from faceProcAddressing: those after the physical patches
            std::vector<double> procFaceIds;
            for (label fa : faces) { const long g = (fa > 0 ? fa : -fa) - 1; if (g < mesh.nInternal && halves()[static_cast<std::size_t>(mesh.owner[static_cast<std::size_t>(g)])] != halves()[static_cast<std::size_t>(mesh.neighbour[static_cast<std::size_t>(g)])]) procFaceIds.push_back(static_cast<double>(fa)); }
            for (double fa : procFaceIds)
            {
                const long g = static_cast<long>(std::fabs(fa)) - 1;
                const long other = fa > 0 ? mesh.neighbour[static_cast<std::size_t>(g)] : mesh.owner[static_cast<std::size_t>(g)];
                across.push_back(static_cast<double>(other));
                phiPatch.push_back(fa > 0 ? static_cast<double>(g + 1) : -static_cast<double>(g + 1));
            }
            const std::string pn = "procBoundary" + std::to_string(p) + "to" + std::to_string(nb);
            const Parsed pu = read(P(a, p, "0/U"), pn);
            std::vector<double> want = expected(across, 3);
            check((pu.uniform ? pu.v.size() == 3 : equalVec(pu.v, want)), who + "U on the processor patch: the other side's cells");
            check(equalVec(read(P(a, p, "0/phi"), pn).v, phiPatch), who + "phi on the processor patch: the face's value, negated where reversed");
            // phi internal faces: this processor's, in the order of faceProcAddressing
            std::vector<double> internalFace;
            for (std::size_t i = 0; i < faces.size(); ++i) { const long g = faces[i] - 1; if (faces[i] > 0 && g < mesh.nInternal && halves()[static_cast<std::size_t>(mesh.owner[static_cast<std::size_t>(g)])] == halves()[static_cast<std::size_t>(mesh.neighbour[static_cast<std::size_t>(g)])]) internalFace.push_back(static_cast<double>(g + 1)); }
            check(equalVec(read(P(a, p, "0/phi"), "").v, internalFace), who + "phi internal faces are its own");
        }

        std::printf("== 2. reconstruct gives the global fields back\n");
        const std::string orig = root + "/a_orig";
        fs::create_directories(orig);
        fs::copy(a + "/0", orig + "/0", fs::copy_options::recursive);
        fs::rename(a + "/0", a + "/0.src");
        decompose::ReconstructOptions ro;
        ro.quiet = true;
        ro.times = {"0"};
        decompose::reconstructCase(a, ro);
        for (const char* f : {"U", "S", "T", "phi"})
        {
            // the reconstructed file is written in its own layout, so the values are read back and compared
            bool same = true;
            for (const char* patch : {"", "inlet", "walls"})
            {
                const std::string pn = patch;
                if (pn.empty() || std::string(f) == "U" || (std::string(f) == "S" && pn == "inlet") || std::string(f) == "phi")
                {
                    const Parsed x = read(orig + "/0/" + f, pn), y = read(a + "/0/" + f, pn);
                    if (!(x.found == y.found && x.uniform == y.uniform && equalVec(x.v, y.v))) same = false;
                }
            }
            check(same, std::string("reconstructed ") + f + " equals the original");
        }
        for (const char* patch : {"inlet", "outlet", "walls", "bottom", "top"})
            check(!read(a + "/0/phi", patch).v.empty(), std::string("reconstructed phi has its values on ") + patch);

        // ---- 3 ----
        std::printf("== 3. one processor round-trips; a processor without cells is refused\n");
        const std::string one = root + "/one";
        writeMesh(one, mesh);
        writeFields(one, mesh, "0");
        decompose::decomposeCaseWith(one, std::vector<label>(NCELLS, 0), 1, opt);
        check(equalVec(read(P(one, 0, "0/U"), "").v, expected(iota(0, NCELLS), 3)), "np 1: U internal field is the whole field");
        check(readLabels(P(one, 0, "constant/polyMesh/cellProcAddressing")).size() == NCELLS, "np 1: all cells on processor 0");
        {
            const std::string e = root + "/empty";
            writeMesh(e, mesh);
            writeFields(e, mesh, "0");
            std::string why;
            try { decompose::decomposeCaseWith(e, halves(), 3, opt); } catch (const std::exception& x) { why = x.what(); }
            check(why.find("processor 2 gets no cells") != std::string::npos, "a processor with no cells is refused by name: " + why.substr(0, 70));
            check(!fs::exists(e + "/processor0"), "...and nothing was written");
        }

        // ---- 4 ----
        std::printf("== 4. gzipped fields and zone files\n");
        if (std::system("gzip --version > /dev/null 2>&1") == 0)
        {
            const std::string g = root + "/gz";
            writeMesh(g, mesh);
            writeFields(g, mesh, "0");
            wr(g + "/constant/polyMesh/cellZones", header("regIOobject", "cellZones", "constant/polyMesh") + "1\n(\nleft\n{\n    type cellZone;\n    cellLabels List<label> 3(0 1 4);\n}\n)\n");
            check(std::system(("gzip " + g + "/0/U " + g + "/constant/polyMesh/cellZones").c_str()) == 0, "gzip the U file and cellZones");
            decompose::Report r = decompose::decomposeCaseWith(g, halves(), 2, opt);
            const std::vector<label> gc = readLabels(P(g, 0, "constant/polyMesh/cellProcAddressing"));
            check(equalVec(read(P(g, 0, "0/U"), "").v, expected(std::vector<double>(gc.begin(), gc.end()), 3)), "U.gz was decomposed");
            check(r.fields.size() == 4, "every field was found, the gzipped one once: " + std::to_string(r.fields.size()));
            std::ifstream zf(P(g, 0, "constant/polyMesh/cellZones"));
            std::string z((std::istreambuf_iterator<char>(zf)), std::istreambuf_iterator<char>());
            check(z.find("left") != std::string::npos && z.find("List<label> 3(0 1 2)") != std::string::npos,
                  "cellZones.gz was decomposed (global cells 0 1 4 are local 0 1 2 on processor 0)");
        }
        else
        {
            std::printf("  skip: no gzip\n");
        }

        // ---- 5 ----
        std::printf("== 5. uniform/\n");
        const std::string u = root + "/u";
        writeMesh(u, mesh);
        writeFields(u, mesh, "0");
        wr(u + "/0/uniform/time", "index 7;\n");
        decompose::decomposeCaseWith(u, halves(), 2, opt);
        check(!fs::exists(P(u, 0, "0/uniform/time")), "without -copyUniform the uniform/ directory is left (as decomposePar)");
        const std::string u2 = root + "/u2";
        writeMesh(u2, mesh);
        writeFields(u2, mesh, "0");
        wr(u2 + "/0/uniform/time", "index 7;\n");
        decompose::Options ou = opt;
        ou.copyUniform = true;
        decompose::decomposeCaseWith(u2, halves(), 2, ou);
        check(fs::exists(P(u2, 1, "0/uniform/time")), "with copyUniform every processor has it");
        fs::remove_all(u2 + "/0");
        decompose::reconstructCase(u2, ro);
        check(fs::exists(u2 + "/0/uniform/time"), "reconstruct brings processor0's back");

        // ---- 6 ----
        std::printf("== 6. a case that cannot be written to\n");
        if (::geteuid() != 0)
        {
            const std::string w = root + "/ro";
            writeMesh(w, mesh);
            writeFields(w, mesh, "0");
            fs::permissions(w, fs::perms::owner_read | fs::perms::owner_exec);
            std::string why;
            try { decompose::decomposeForRun(w, 2, true); } catch (const std::exception& x) { why = x.what(); }
            fs::permissions(w, fs::perms::owner_all);
            check(why.find("cannot be created") != std::string::npos, "decomposeForRun on a read-only case throws instead of waiting: " + why.substr(0, 60));
        }
        else
        {
            std::printf("  skip: running as root\n");
        }

        // ---- 7 ----
        std::printf("== 7. zone files\n");
        const std::string z1 = root + "/z1";
        writeMesh(z1, mesh);
        writeFields(z1, mesh, "0");
        wr(z1 + "/constant/polyMesh/pointZones", header("regIOobject", "pointZones", "constant/polyMesh") + "1\n(\nfrozen\n{\n    type pointZone;\n    pointLabels List<label> 0;\n}\n)\n");
        decompose::decomposeCaseWith(z1, halves(), 2, opt);
        check(fs::exists(P(z1, 1, "constant/polyMesh/pointZones")), "an empty point zone is carried over");
        const std::string z2 = root + "/z2";
        writeMesh(z2, mesh);
        writeFields(z2, mesh, "0");
        wr(z2 + "/constant/polyMesh/pointZones", header("regIOobject", "pointZones", "constant/polyMesh") + "1\n(\nfrozen\n{\n    type pointZone;\n    pointLabels List<label> 2(3 4);\n}\n)\n");
        std::string why;
        try { decompose::decomposeCaseWith(z2, halves(), 2, opt); } catch (const std::exception& x) { why = x.what(); }
        check(why.find("pointZones has members") != std::string::npos, "a point zone with members is refused by name: " + why.substr(0, 60));
    }
    catch (const std::exception& e)
    {
        std::printf("EXCEPTION: %s\n", e.what());
        exitCode = 1;
    }
    fs::remove_all(root);
    if (failures || exitCode)
    {
        std::printf("FAIL: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
