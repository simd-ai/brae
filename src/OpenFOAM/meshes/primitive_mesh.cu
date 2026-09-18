#include "primitive_mesh.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cctype>

namespace brae {

namespace {

// FAST polyMesh ASCII scan. The general TokenStream tokenises the whole file into a std::vector<std::string> --
// one heap-allocated string PER NUMBER -- so a 1M-cell mesh (owner/neighbour 3M ints each, points 3M floats,
// faces ~15M ints) does ~24M string allocations and took ~8.3s (measured, the dominant startup cost). The
// polyMesh numeric lists have a rigid shape (count, '(', values, ')'), so parse them straight off the file
// buffer with strtol/strtod and treat ( ) as delimiters -- no per-token allocation. Header + comments are
// skipped structurally. Cuts the 1M-cell read from ~8.3s to well under 1s.
struct FastScan
{
    std::string buf;   // whole file (null-terminated via std::string)
    const char* p = nullptr;
    const char* end = nullptr;
    explicit FastScan(const std::string& path)
    {
        const std::vector<char> b = gzSlurp(path);   // gz-transparent: reads <path> or <path>.gz (OF writeCompression on)
        buf.assign(b.begin(), b.end());
        p = buf.data();
        end = p + buf.size();
        skipHeader();
    }
    // Advance past the leading FoamFile{...} dictionary so the first value read is the payload count.
    void skipHeader()
    {
        static const char kw[] = "FoamFile";
        const char* f = std::search(p, end, kw, kw + 8);
        if (f == end) return;                       // no header -> payload at the top
        const char* b = std::find(f, end, '{');
        if (b == end) return;
        int depth = 0;
        for (const char* q = b; q < end; ++q)
        {
            if (*q == '{') ++depth;
            else if (*q == '}' && --depth == 0) { p = q + 1; return; }
        }
    }
    // Next integer, skipping any delimiters (whitespace, parens, comment punctuation) before it.
    long nextLong()
    {
        while (p < end && !(std::isdigit((unsigned char)*p) || *p == '-' || *p == '+')) ++p;
        char* q = nullptr;
        const long v = std::strtol(p, &q, 10);
        p = q;
        return v;
    }
    // Next float, skipping delimiters. '.' also starts a number (e.g. ".5").
    double nextDouble()
    {
        while (p < end && !(std::isdigit((unsigned char)*p) || *p == '-' || *p == '+' || *p == '.')) ++p;
        char* q = nullptr;
        const double v = std::strtod(p, &q);
        p = q;
        return v;
    }
};

constexpr unsigned BRAE_MESH_CACHE_MAGIC = 0x43464D34;   // "CFM4"
// Bumped again (CFM3 -> CFM4) when PatchInfo gained periodicPatch/maxIter/matchTolerance for
// cyclicPeriodicAMI -- same reason.
// Bumped CFM2 -> CFM3 when PatchInfo gained nonOverlapPatch: the record layout changed, so a cache
// written by the previous build must be REJECTED rather than read as garbage (loadBinary returns
// false on a magic mismatch and the caller re-reads the polyMesh).
template<class T>
void wvec(std::FILE* f, const std::vector<T>& v)
{
    std::size_t n = v.size();
    std::fwrite(&n, sizeof(n), 1, f);
    if (n) std::fwrite(v.data(), sizeof(T), n, f);
}
template<class T>
bool rvec(std::FILE* f, std::vector<T>& v)
{
    std::size_t n;
    if (std::fread(&n, sizeof(n), 1, f) != 1) return false;
    v.resize(n);
    return n == 0 || std::fread(v.data(), sizeof(T), n, f) == n;
}
void wstr(std::FILE* f, const std::string& s)
{
    std::size_t n = s.size();
    std::fwrite(&n, sizeof(n), 1, f);
    if (n) std::fwrite(s.data(), 1, n, f);
}
bool rstr(std::FILE* f, std::string& s)
{
    std::size_t n;
    if (std::fread(&n, sizeof(n), 1, f) != 1) return false;
    s.resize(n);
    return n == 0 || std::fread(&s[0], 1, n, f) == n;
}
} // namespace

void PrimitiveMesh::writeBinary(const std::string& path) const
{
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    unsigned magic = BRAE_MESH_CACHE_MAGIC;
    std::fwrite(&magic, sizeof(magic), 1, f);
    std::fwrite(&nCells_, sizeof(nCells_), 1, f);
    wvec(f, points_);
    wvec(f, faceVerts_);
    wvec(f, faceOffsets_);
    wvec(f, owner_);
    wvec(f, neighbour_);
    std::size_t np = patches_.size();
    std::fwrite(&np, sizeof(np), 1, f);
    for (const auto& p : patches_)
    {
        wstr(f, p.name);
        wstr(f, p.type);
        std::fwrite(&p.start, sizeof(p.start), 1, f);
        std::fwrite(&p.size, sizeof(p.size), 1, f);
        wstr(f, p.neighbourPatch);
        wstr(f, p.nonOverlapPatch);
        wstr(f, p.periodicPatch);
        std::fwrite(&p.maxIter, sizeof(p.maxIter), 1, f);
        std::fwrite(&p.matchTolerance, sizeof(p.matchTolerance), 1, f);
        std::size_t ng = p.inGroups.size();
        std::fwrite(&ng, sizeof(ng), 1, f);
        for (const auto& g : p.inGroups) wstr(f, g);
        wstr(f, p.transform);
        std::fwrite(&p.rotationAxis, sizeof(p.rotationAxis), 1, f);
        std::fwrite(&p.rotationCentre, sizeof(p.rotationCentre), 1, f);
    }
    std::fclose(f);
}

bool PrimitiveMesh::loadBinary(const std::string& path)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned magic = 0;
    bool ok = std::fread(&magic, sizeof(magic), 1, f) == 1 && magic == BRAE_MESH_CACHE_MAGIC;
    ok = ok && std::fread(&nCells_, sizeof(nCells_), 1, f) == 1;
    ok = ok && rvec(f, points_) && rvec(f, faceVerts_) && rvec(f, faceOffsets_) && rvec(f, owner_) && rvec(f, neighbour_);
    std::size_t np = 0;
    ok = ok && std::fread(&np, sizeof(np), 1, f) == 1;
    if (ok)
    {
        patches_.resize(np);
        for (auto& p : patches_)
        {
            ok = ok && rstr(f, p.name) && rstr(f, p.type)
                 && std::fread(&p.start, sizeof(p.start), 1, f) == 1 && std::fread(&p.size, sizeof(p.size), 1, f) == 1
                 && rstr(f, p.neighbourPatch) && rstr(f, p.nonOverlapPatch)
                 && rstr(f, p.periodicPatch)
                 && std::fread(&p.maxIter, sizeof(p.maxIter), 1, f) == 1
                 && std::fread(&p.matchTolerance, sizeof(p.matchTolerance), 1, f) == 1;
            std::size_t ng = 0;
            ok = ok && std::fread(&ng, sizeof(ng), 1, f) == 1;
            if (ok)
            {
                p.inGroups.resize(ng);
                for (auto& g : p.inGroups) ok = ok && rstr(f, g);
            }
            ok = ok && rstr(f, p.transform)
                 && std::fread(&p.rotationAxis, sizeof(p.rotationAxis), 1, f) == 1
                 && std::fread(&p.rotationCentre, sizeof(p.rotationCentre), 1, f) == 1;
            if (!ok) break;
        }
    }
    std::fclose(f);
    return ok;
}

void PrimitiveMesh::readPoints(const std::string& dir)
{
    const std::string path = dir + "/points";
    if (foamFormat(path) == "binary")
    {
        points_ = readBinaryPoints(path);
        return;
    }

    FastScan s(path);
    const label n = static_cast<label>(s.nextLong());   // ( and per-point ( ) are delimiters -> just read 3n floats
    points_.resize(n);
    for (label i = 0; i < n; ++i)
    {
        points_[i].x = s.nextDouble();
        points_[i].y = s.nextDouble();
        points_[i].z = s.nextDouble();
    }
}

void PrimitiveMesh::readFaces(const std::string& dir)
{
    const std::string path = dir + "/faces";
    if (foamFormat(path) == "binary")
    {
        readBinaryCompactFaces(path, faceOffsets_, faceVerts_);
        return;
    }

    FastScan s(path);
    const label n = static_cast<label>(s.nextLong());   // face count
    faceOffsets_.resize(n + 1);
    faceOffsets_[0] = 0;
    faceVerts_.clear();
    for (label f = 0; f < n; ++f)
    {
        const label k = static_cast<label>(s.nextLong());   // verts in this face; its ( ) are delimiters
        for (label j = 0; j < k; ++j) faceVerts_.push_back(static_cast<label>(s.nextLong()));
        faceOffsets_[f + 1] = faceOffsets_[f] + k;
    }
}

static std::vector<label> readLabelColumn(const std::string& path)
{
    FastScan s(path);
    const label n = static_cast<label>(s.nextLong());   // count; '(' then the n ints ')' are parsed as delimiters
    std::vector<label> v(n);
    for (label i = 0; i < n; ++i) v[i] = static_cast<label>(s.nextLong());
    return v;
}

static std::vector<label> readLabelFile(const std::string& path)
{
    return (foamFormat(path) == "binary") ? readBinaryLabelList(path) : readLabelColumn(path);
}

void PrimitiveMesh::readOwner(const std::string& dir)     { owner_     = readLabelFile(dir + "/owner"); }
void PrimitiveMesh::readNeighbour(const std::string& dir) { neighbour_ = readLabelFile(dir + "/neighbour"); }

// A Function1 `table` payload as blockMesh writes it: an optional leading count, then a list of
// (t value) pairs -- `3((0 1)(0.2 1)(0.3 0))`, with the count and the newlines both optional.
static std::vector<std::pair<scalar, scalar>> readAcmiScaleTable(TokenStream& ts)
{
    std::vector<std::pair<scalar, scalar>> pts;
    if (ts.peek() != "(") ts.next();                       // optional element count
    ts.expect("(");
    while (ts.peek() != ")")
    {
        ts.expect("(");
        const scalar t = ts.nextScalar();
        const scalar v = ts.nextScalar();
        ts.expect(")");
        pts.emplace_back(t, v);
    }
    ts.expect(")");
    return pts;
}

void PrimitiveMesh::readBoundary(const std::string& dir)
{
    TokenStream ts(dir + "/boundary");
    const label np = ts.nextLabel();
    ts.expect("(");
    patches_.clear();
    patches_.reserve(np);
    for (label p = 0; p < np; ++p)
    {
        PatchInfo pi;
        // cyclicACMI `scale`: blockMesh writes the Function1 in its expanded form,
        //     scale table;  scaleCoeffs { values 3((0 1)(0.2 1)(0.3 0)); }
        // so the selector word and the data arrive as two separate entries and have to be joined up
        // after the loop.
        std::string acmiScaleType;
        std::vector<std::pair<scalar, scalar>> acmiScaleTable;
        scalar acmiScaleConst = 1;
        pi.name = ts.next();
        ts.expect("{");
        while (ts.peek() != "}")
        {
            const std::string key = ts.next();
            if      (key == "scale" && ts.peek() == "{")
            {
                // THE DICTIONARY FORM, `scale { type coded; code #{ ... #}; }`. OpenFOAM reads `scale` as a
                // PatchFunction1 (cyclicACMIPolyPatch.C:625), so the selector may sit inside a block, and
                // RAS/damBreakLeakage writes a per-face coded one there. This branch took `{` for the
                // selector word and the parse died on "expected ';' got 'type'", naming nothing. `constant`
                // and `table` are read as their inline forms are; any other type reaches the refusal below
                // under its own name.
                ts.expect("{");
                while (ts.peek() != "}")
                {
                    const std::string k2 = ts.next();
                    if (k2 == "type")
                    {
                        acmiScaleType = ts.next();
                        ts.expect(";");
                    }
                    else if (k2 == "value" && ts.peek() != "{")
                    {
                        acmiScaleConst = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (k2 == "values")
                    {
                        acmiScaleTable = readAcmiScaleTable(ts);
                        ts.expect(";");
                    }
                    else if (ts.peek() == "{")
                    {
                        ts.expect("{");
                        for (int depth = 1; depth > 0; )
                        {
                            const std::string t = ts.next();
                            if (t == "{")
                            {
                                ++depth;
                            }
                            else if (t == "}")
                            {
                                --depth;
                            }
                        }
                    }
                    else
                    {
                        while (ts.peek() != ";")
                        {
                            ts.next();
                        }
                        ts.expect(";");
                    }
                }
                ts.expect("}");
                continue;
            }
            else if (key == "scale")
            {
                acmiScaleType = ts.next();
                // `scale constant 0.5;` carries its value inline; `scale table;` defers to scaleCoeffs.
                if (acmiScaleType == "constant") acmiScaleConst = ts.nextScalar();
                else if (acmiScaleType == "table" && ts.peek() == "(")   // inline table, no scaleCoeffs
                    acmiScaleTable = readAcmiScaleTable(ts);
                ts.expect(";");
                continue;
            }
            else if (key == "scaleCoeffs")
            {
                ts.expect("{");
                while (ts.peek() != "}")
                {
                    const std::string k2 = ts.next();
                    if (k2 == "values") { acmiScaleTable = readAcmiScaleTable(ts); ts.expect(";"); }
                    else if (ts.peek() == "{")   // nested sub-dict: skip balanced
                    {
                        ts.expect("{");
                        for (int depth = 1; depth > 0; )
                        {
                            const std::string t = ts.next();
                            if      (t == "{") ++depth;
                            else if (t == "}") --depth;
                        }
                    }
                    else { while (ts.peek() != ";") ts.next(); ts.expect(";"); }
                }
                ts.expect("}");
                continue;
            }
            if      (key == "type")           { pi.type  = ts.next();      ts.expect(";"); }
            else if (key == "nFaces")         { pi.size  = ts.nextLabel(); ts.expect(";"); }
            else if (key == "startFace")      { pi.start = ts.nextLabel(); ts.expect(";"); }
            else if (key == "neighbourPatch") { pi.neighbourPatch = ts.next(); ts.expect(";"); }
            else if (key == "nonOverlapPatch") { pi.nonOverlapPatch = ts.next(); ts.expect(";"); }
            else if (key == "periodicPatch")  { pi.periodicPatch = ts.next(); ts.expect(";"); }
            else if (key == "maxIter")        { pi.maxIter = (label)ts.nextScalar(); ts.expect(";"); }
            else if (key == "matchTolerance") { pi.matchTolerance = ts.nextScalar(); ts.expect(";"); }
            else if (key == "transform")      { pi.transform = ts.next();      ts.expect(";"); }
            else if (key == "rotationAxis")   { ts.expect("("); pi.rotationAxis   = {ts.nextScalar(), ts.nextScalar(), ts.nextScalar()}; ts.expect(")"); ts.expect(";"); }
            else if (key == "rotationCentre") { ts.expect("("); pi.rotationCentre = {ts.nextScalar(), ts.nextScalar(), ts.nextScalar()}; ts.expect(")"); ts.expect(";"); }
            else if (key == "inGroups")   // wordList: "N(g1 g2 ...)" or "(g1 ...)"
            {
                if (ts.peek() != "(") ts.next();                          // optional leading count token
                ts.expect("(");
                while (ts.peek() != ")") pi.inGroups.push_back(ts.next());
                ts.expect(")");
                ts.expect(";");
            }
            // An unsupported key whose value is a sub-DICTIONARY, e.g. cyclicACMI's
            //     scale table; scaleCoeffs { values 3((0 1)(0.2 1)(0.3 0)); }
            // Skipping to the next ';' lands INSIDE the block (on the list's terminator), after which the
            // block's own '}' is mistaken for the patch's and every later patch is off by one -- the real
            // error being reported miles away as "expected '{' got <the next patch's name>", which is how
            // pimpleFoam/RAS/TJunctionSwitching failed to load at all. Skip balanced braces instead.
            else if (ts.peek() == "{")
            {
                ts.expect("{");
                for (int depth = 1; depth > 0; )
                {
                    const std::string t = ts.next();
                    if      (t == "{") ++depth;
                    else if (t == "}") --depth;
                }
            }
            else { while (ts.peek() != ";") ts.next(); ts.expect(";"); }  // skip remaining unsupported keys
        }
        ts.expect("}");
        // `scale` makes the interface's open area a prescribed function of time (TJunctionSwitching
        // closes a branch with it). Only the forms brae evaluates are accepted; anything else is
        // refused rather than run as a fixed-overlap interface, which would be a different case.
        // OF constructs the scale Function1 in cyclicACMIPolyPatch's constructor and nowhere else, so on
        // any other patch type the entry is simply never looked up. Match that: read it on cyclicACMI,
        // ignore it elsewhere (refusing would reject a dictionary OpenFOAM accepts).
        if (!acmiScaleType.empty() && pi.type == "cyclicACMI")
        {
            if (acmiScaleType == "constant")        pi.acmiScale = Function1::constant(acmiScaleConst);
            else if (acmiScaleType == "table" && !acmiScaleTable.empty())
                                                    pi.acmiScale = Function1::table(acmiScaleTable);
            else
                throw std::runtime_error(
                    "brae: cyclicACMI '" + pi.name + "' has `scale " + acmiScaleType + "`, which brae "
                    "does not evaluate (it reads `constant` and `table`). The scale sets how far the "
                    "interface is open at each time, so substituting another function solves a "
                    "different case.");
        }
        patches_.push_back(std::move(pi));
    }
    ts.expect(")");

    // The scale belongs to the interface PAIR, not to one patch -- see propagateACMIScale.
    propagateACMIScale(patches_);
}

void PrimitiveMesh::read(const std::string& polyMeshDir)
{
    // BRAE_MESH_CACHE: skip the (slow) ASCII parse on a warm run by reloading a binary blob, provided it is newer than
    // the polyMesh/owner file (so an edited/regenerated mesh auto-invalidates the cache). Same idea as OF reusing
    // decomposePar's processor* dirs, one cold parse, then fast warm starts.
    const std::string cachePath = polyMeshDir + "/.brae_meshcache";
    {   // AUTO warm-load if a valid cache is present (newer than owner), no env needed, so a `-partition` run makes
        // the subsequent solve warm automatically. Stale/foreign caches are rejected (mtime + magic) -> cold parse.
        std::error_code ec;
        namespace fs = std::filesystem;
        const std::string ownerPath = polyMeshDir + "/owner";
        if (fs::exists(cachePath, ec) && fs::exists(ownerPath, ec)
            && fs::last_write_time(cachePath, ec) >= fs::last_write_time(ownerPath, ec)
            && loadBinary(cachePath))
            return;                                          // warm: reloaded from cache
    }
    const bool writeCache = std::getenv("BRAE_MESH_CACHE") != nullptr;   // write only when asked (-partition / env)
    readPoints(polyMeshDir);
    readFaces(polyMeshDir);
    readOwner(polyMeshDir);
    readNeighbour(polyMeshDir);
    readBoundary(polyMeshDir);

    // nCells = max cell index referenced by owner/neighbour, + 1.
    label maxCell = -1;
    for (const label c : owner_)     maxCell = std::max(maxCell, c);
    for (const label c : neighbour_) maxCell = std::max(maxCell, c);
    nCells_ = maxCell + 1;

    if (writeCache) writeBinary(cachePath);                  // cold: write the cache for next time
}

} // namespace brae
