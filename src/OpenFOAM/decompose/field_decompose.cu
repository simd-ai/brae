// The fields, both ways: decomposePar's fvFieldDecomposer and reconstructPar's fvFieldReconstructor, read from OpenFOAM v2606
// (src/parallel/decompose/decompose/fvFieldDecomposer.{cxx,txx}, src/parallel/reconstruct/reconstruct/fvFieldReconstructor*).
//
// A field file is read as a tree of entries without knowing what any boundary condition means: an entry is a dictionary, a plain
// run of tokens, `uniform <tokens>` or `nonuniform List<T> N (...)`. A nonuniform list is kept as doubles (a binary payload is
// decoded to them), so a 2.5M-cell field is 20 MB, not millions of strings. What is cut and merged is the internal field, and the
// lists of the patch entries whose length is the patch's face count. Everything else rides along unchanged.
#include "decompose_detail.cuh"

#include "foam_token_reader.cuh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace brae {
namespace decompose {
namespace detail {
namespace {

struct Value
{
    enum Kind { Plain, Uniform, List } kind = Plain;
    std::vector<std::string> tokens;      // Plain: every token of the value; Uniform: the tokens after `uniform`
    std::string              listType;    // List: `List<scalar>`
    int                      comps = 1;   // List: numbers per element
    long                     n = 0;       // List: elements
    std::vector<double>      data;        // List: n * comps
};

struct Entry
{
    std::string        key;
    bool               isDict = false;
    Value              v;
    std::vector<Entry> sub;

    const Entry* find(const std::string& k) const
    {
        for (const Entry& e : sub) if (e.key == k) return &e;
        return nullptr;
    }
    Entry* find(const std::string& k)
    {
        for (Entry& e : sub) if (e.key == k) return &e;
        return nullptr;
    }
};

struct FieldFile
{
    std::string        cls, object;
    std::vector<Entry> top;     // everything after the FoamFile header: dimensions, internalField, boundaryField, oriented, ...
};

bool isNumber(const std::string& t)
{
    double d;
    if (TokenStream::decodeBinaryScalar(t, d)) return true;
    if (t.empty()) return false;
    char* end = nullptr;
    std::strtod(t.c_str(), &end);
    return end != t.c_str() && *end == '\0';
}

void readList(TokenStream& ts, Value& v)
{
    v.kind = Value::List;
    if (ts.peek().compare(0, 5, "List<") == 0) v.listType = ts.next();
    v.n = static_cast<long>(ts.nextLabel());
    if (v.n == 0)
    {
        if (!ts.eof() && ts.peek() == "(") { ts.next(); ts.expect(")"); }
        if (!ts.eof() && ts.peek() == ";") ts.next();
        v.comps = 1;
        return;
    }
    ts.expect("(");
    v.data.reserve(static_cast<std::size_t>(v.n));
    for (long i = 0; i < v.n; ++i)
    {
        if (ts.peek() == "(")
        {
            ts.next();
            int c = 0;
            while (ts.peek() != ")") { v.data.push_back(ts.nextScalar()); ++c; }
            ts.next();
            if (i == 0) v.comps = c;
            else if (c != v.comps) throw std::runtime_error("brae decompose: a list whose elements have different sizes");
        }
        else
        {
            v.comps = 1;
            v.data.push_back(ts.nextScalar());
        }
    }
    ts.expect(")");
    if (!ts.eof() && ts.peek() == ";") ts.next();
}

Entry readEntry(TokenStream& ts)
{
    Entry e;
    e.key = ts.next();
    if (ts.peek() == "{")
    {
        ts.next();
        e.isDict = true;
        while (ts.peek() != "}") e.sub.push_back(readEntry(ts));
        ts.next();
        if (!ts.eof() && ts.peek() == ";") ts.next();
        return e;
    }
    if (ts.peek() == "nonuniform")
    {
        ts.next();
        readList(ts, e.v);
        return e;
    }
    if (ts.peek() == "uniform")
    {
        ts.next();
        e.v.kind = Value::Uniform;
    }
    int depth = 0;
    for (;;)
    {
        const std::string t = ts.next();
        if (t == ";" && depth == 0) break;
        if (t == "(" || t == "[" || t == "{") ++depth;
        if (t == ")" || t == "]" || t == "}") --depth;
        e.v.tokens.push_back(t);
    }
    return e;
}

// the FoamFile header's class and object. TokenStream drops the header, so they are read off the file's first bytes.
void readHeader(const std::string& path, std::string& cls, std::string& object)
{
    std::ifstream in(path, std::ios::binary);
    std::string head(2048, '\0');
    in.read(head.data(), static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<std::size_t>(in.gcount()));
    const std::size_t ff = head.find("FoamFile");
    if (ff == std::string::npos) return;
    const std::size_t end = head.find('}', ff);
    const std::string block = head.substr(ff, end == std::string::npos ? std::string::npos : end - ff);
    auto word = [&](const char* key) -> std::string
    {
        const std::size_t k = block.find(key);
        if (k == std::string::npos) return std::string();
        std::size_t i = k + std::strlen(key);
        while (i < block.size() && std::isspace(static_cast<unsigned char>(block[i]))) ++i;
        std::size_t j = i;
        while (j < block.size() && block[j] != ';' && !std::isspace(static_cast<unsigned char>(block[j]))) ++j;
        return block.substr(i, j - i);
    };
    cls = word("class");
    object = word("object");
}

FieldFile readFieldFile(const std::string& path)
{
    FieldFile f;
    readHeader(path, f.cls, f.object);
    // anything that is not a volume or surface field (cellLevel, a dictionary, a points file) is left to the caller to skip: it is
    // not an `entry value;` file and would not parse as one
    if (f.cls.compare(0, 3, "vol") != 0 && f.cls.compare(0, 7, "surface") != 0) return f;
    TokenStream ts(path, /*expandVars=*/true);
    while (!ts.eof())
    {
        Entry e = readEntry(ts);
        if (e.key == "FoamFile") continue;      // if the reader kept it
        f.top.push_back(std::move(e));
    }
    return f;
}

// "volScalarField" -> ("vol", "scalar"); anything else -> kind ""
struct FieldKind
{
    bool        vol = false, surface = false;
    std::string type;    // scalar, vector, sphericalTensor, symmTensor, tensor
    int         comps = 1;
};

FieldKind fieldKind(const std::string& cls)
{
    FieldKind k;
    std::string rest;
    if (cls.compare(0, 3, "vol") == 0) { k.vol = true; rest = cls.substr(3); }
    else if (cls.compare(0, 7, "surface") == 0) { k.surface = true; rest = cls.substr(7); }
    else return k;
    if (rest.size() < 6 || rest.compare(rest.size() - 5, 5, "Field") != 0) { k.vol = k.surface = false; return k; }
    rest.resize(rest.size() - 5);
    if (rest == "Scalar") { k.type = "scalar"; k.comps = 1; }
    else if (rest == "Vector") { k.type = "vector"; k.comps = 3; }
    else if (rest == "SphericalTensor") { k.type = "sphericalTensor"; k.comps = 1; }
    else if (rest == "SymmTensor") { k.type = "symmTensor"; k.comps = 6; }
    else if (rest == "Tensor") { k.type = "tensor"; k.comps = 9; }
    else { k.vol = k.surface = false; }
    return k;
}

// the numbers of a uniform value: `5`, `(1 2 3)`
std::vector<double> uniformNumbers(const Value& v)
{
    std::vector<double> d;
    for (const std::string& t : v.tokens)
    {
        if (t == "(" || t == ")") continue;
        if (!isNumber(t))
            throw std::runtime_error("brae decompose: the uniform value `" + t + "` is not a number (a $variable that was not expanded, or a "
                                     "function: not decomposed).");
        double x;
        if (!TokenStream::decodeBinaryScalar(t, x)) x = std::strtod(t.c_str(), nullptr);
        d.push_back(x);
    }
    return d;
}

// element i of a value as `comps` numbers: out[0..comps)
struct Access
{
    const Value*        v;
    int                 comps;
    std::vector<double> uni;

    Access(const Value& val, int c) : v(&val), comps(c)
    {
        if (val.kind == Value::Uniform)
        {
            uni = uniformNumbers(val);
            if (static_cast<int>(uni.size()) != c)
                throw std::runtime_error("brae decompose: a uniform value of " + std::to_string(uni.size()) + " numbers in a field of " + std::to_string(c));
        }
        else if (val.kind == Value::List)
        {
            if (val.n > 0 && val.comps != c)
                throw std::runtime_error("brae decompose: a list of " + std::to_string(val.comps) + "-number elements in a field of " + std::to_string(c));
        }
        else
        {
            throw std::runtime_error("brae decompose: a field value that is neither uniform nor nonuniform");
        }
    }
    long size() const { return v->kind == Value::List ? v->n : -1; }
    const double* at(long i) const
    {
        return v->kind == Value::Uniform ? uni.data() : v->data.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(comps);
    }
};

Value makeList(const std::string& type, int comps, std::vector<double>&& data)
{
    Value v;
    v.kind = Value::List;
    v.listType = "List<" + type + ">";
    v.comps = comps;
    v.n = static_cast<long>(data.size() / static_cast<std::size_t>(comps));
    v.data = std::move(data);
    return v;
}

// ---- writing ---------------------------------------------------------------------------------------------------------------
void writeValue(std::ostream& o, const Value& v, const std::string& indent)
{
    if (v.kind == Value::Plain)
    {
        for (std::size_t i = 0; i < v.tokens.size(); ++i) o << (i ? " " : "") << tokenText(v.tokens[i]);
        return;
    }
    if (v.kind == Value::Uniform)
    {
        o << "uniform";
        for (const std::string& t : v.tokens) o << ' ' << tokenText(t);
        return;
    }
    // OpenFOAM writes a list whose values are all equal as `uniform` (Field::writeEntry)
    bool same = v.n > 0;
    for (long i = 1; same && i < v.n; ++i)
        for (int c = 0; c < v.comps; ++c)
            if (v.data[static_cast<std::size_t>(i * v.comps + c)] != v.data[static_cast<std::size_t>(c)]) { same = false; break; }
    if (same)
    {
        o << "uniform ";
        if (v.comps > 1) o << '(';
        for (int c = 0; c < v.comps; ++c) o << (c ? " " : "") << numberText(v.data[static_cast<std::size_t>(c)]);
        if (v.comps > 1) o << ')';
        return;
    }
    o << "nonuniform " << (v.listType.empty() ? "List<scalar>" : v.listType) << ' ' << v.n;
    if (v.n == 0) { o << "()"; return; }
    o << "\n" << indent << "(\n";
    for (long i = 0; i < v.n; ++i)
    {
        o << indent;
        if (v.comps > 1) o << '(';
        for (int c = 0; c < v.comps; ++c) o << (c ? " " : "") << numberText(v.data[static_cast<std::size_t>(i * v.comps + c)]);
        if (v.comps > 1) o << ')';
        o << '\n';
    }
    o << indent << ")";
}

void writeEntry(std::ostream& o, const Entry& e, const std::string& indent)
{
    if (e.isDict)
    {
        o << indent << e.key << "\n" << indent << "{\n";
        for (const Entry& s : e.sub) writeEntry(o, s, indent + "    ");
        o << indent << "}\n";
        return;
    }
    o << indent << e.key << "    ";
    writeValue(o, e.v, indent);
    o << ";\n";
}

void writeFieldFile(const std::string& path, const std::string& cls, const std::string& object, const std::string& time,
                    const std::vector<Entry>& top)
{
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("brae decompose: cannot write " + path);
    o << foamHeader(cls, object, time);
    for (const Entry& e : top)
    {
        writeEntry(o, e, "");
        if (e.key == "dimensions" || e.key == "internalField") o << "\n";
    }
    o << "\n// ************************************************************************* //\n";
}

// ---- the boundaryField entry for a patch ---------------------------------------------------------------------------------------
// exact name, then a patch group, then a pattern (the last that matches wins, as OpenFOAM's patternEntries do)
const Entry* patchEntry(const Entry& boundaryField, const PatchLayout& p)
{
    if (const Entry* e = boundaryField.find(p.name)) return e;
    for (const std::string& g : p.inGroups)
        if (const Entry* e = boundaryField.find(g)) return e;
    for (auto it = boundaryField.sub.rbegin(); it != boundaryField.sub.rend(); ++it)
    {
        if (!it->isDict) continue;
        try
        {
            if (std::regex_match(p.name, std::regex(it->key))) return &*it;
        }
        catch (const std::regex_error&)
        {
        }
    }
    return nullptr;
}

const Entry* top(const FieldFile& f, const char* key)
{
    for (const Entry& e : f.top) if (e.key == key) return &e;
    return nullptr;
}

bool isOriented(const FieldFile& f)
{
    const Entry* e = top(f, "oriented");
    return e && !e->v.tokens.empty() && e->v.tokens[0] == "oriented";
}

}   // namespace

// ------------------------------------------------------------------------------------------------------------------------------
void decomposeFields(const std::string& caseDir, const Plan& plan, const std::string& time, std::vector<std::string>& done, bool quiet)
{
    const std::string src = caseDir + "/" + time;
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(src))
        if (e.is_regular_file()) names.push_back(e.path().filename().string());
    std::sort(names.begin(), names.end());

    const GlobalLayout& G = plan.global;
    for (const std::string& name : names)
    {
        if (name.empty() || name[0] == '.' || name.find(".orig") != std::string::npos) continue;
        FieldFile f;
        try
        {
            f = readFieldFile(src + "/" + name);
        }
        catch (const std::exception& e)
        {
            if (!quiet) std::printf("brae decompose: %s/%s is not read as a field (%s): skipped\n", time.c_str(), name.c_str(), e.what());
            continue;
        }
        const FieldKind k = fieldKind(f.cls);
        if (!k.vol && !k.surface)
        {
            if (!quiet) std::printf("brae decompose: %s/%s is a %s, not a volume or surface field: skipped\n", time.c_str(), name.c_str(),
                                    f.cls.empty() ? "file without a class" : f.cls.c_str());
            continue;
        }
        const Entry* internal = top(f, "internalField");
        const Entry* bfield = top(f, "boundaryField");
        if (!internal || !bfield || !bfield->isDict)
            throw std::runtime_error("brae decompose: " + src + "/" + name + " has no internalField or boundaryField.");
        const Access in(internal->v, k.comps);
        const long expected = k.vol ? G.nCells : G.nInternalFaces;
        if (in.size() >= 0 && in.size() != expected)
            throw std::runtime_error("brae decompose: " + src + "/" + name + ": internalField has " + std::to_string(in.size()) + " values and the mesh has "
                                     + std::to_string(expected) + (k.vol ? " cells." : " internal faces."));
        const bool oriented = k.surface && isOriented(f);

        // each global patch's entry, once
        std::vector<const Entry*> gEntry(G.patches.size(), nullptr);
        for (std::size_t g = 0; g < G.patches.size(); ++g)
        {
            gEntry[g] = patchEntry(*bfield, G.patches[g]);
            if (!gEntry[g])
                throw std::runtime_error("brae decompose: " + src + "/" + name + " has no boundaryField entry for patch `" + G.patches[g].name + "`.");
        }

        for (int p = 0; p < plan.nProcs; ++p)
        {
            const ProcLayout& pl = plan.proc[static_cast<std::size_t>(p)];
            std::vector<Entry> out;
            for (const Entry& t : f.top)
            {
                if (t.key == "internalField")
                {
                    Entry e;
                    e.key = t.key;
                    if (t.v.kind == Value::Uniform)
                    {
                        e = t;
                    }
                    else
                    {
                        std::vector<double> d;
                        const std::size_t n = k.vol ? pl.cells.size() : static_cast<std::size_t>(pl.nInternal);
                        d.reserve(n * static_cast<std::size_t>(k.comps));
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            const long src_i = k.vol ? pl.cells[i] : static_cast<long>(pl.faces[i]) - 1;
                            const double* v = in.at(src_i);
                            d.insert(d.end(), v, v + k.comps);
                        }
                        e.v = makeList(k.type, k.comps, std::move(d));
                    }
                    out.push_back(std::move(e));
                }
                else if (t.key == "boundaryField")
                {
                    Entry b;
                    b.key = t.key;
                    b.isDict = true;
                    for (std::size_t lp = 0; lp < pl.patches.size(); ++lp)
                    {
                        const PatchLayout& pt = pl.patches[lp];
                        Entry pe;
                        pe.key = pt.name;
                        pe.isDict = true;
                        if (pl.boundary[lp] >= 0)
                        {
                            const PatchLayout& gp = G.patches[static_cast<std::size_t>(pl.boundary[lp])];
                            const Entry& ge = *gEntry[static_cast<std::size_t>(pl.boundary[lp])];
                            for (const Entry& s : ge.sub)
                            {
                                Entry c = s;
                                if (!s.isDict && s.v.kind == Value::List && s.v.n == gp.size && gp.size > 0)
                                {
                                    std::vector<double> d;
                                    d.reserve(static_cast<std::size_t>(pt.size) * static_cast<std::size_t>(s.v.comps));
                                    for (label j = 0; j < pt.size; ++j)
                                    {
                                        const long gi = static_cast<long>(pl.faces[static_cast<std::size_t>(pt.start + j)]) - 1 - gp.start;
                                        const double* v = s.v.data.data() + static_cast<std::size_t>(gi) * static_cast<std::size_t>(s.v.comps);
                                        d.insert(d.end(), v, v + s.v.comps);
                                    }
                                    c.v = makeList(s.v.listType.empty() ? k.type : s.v.listType.substr(5, s.v.listType.size() - 6), s.v.comps, std::move(d));
                                }
                                pe.sub.push_back(std::move(c));
                            }
                        }
                        else
                        {
                            // a processor patch: the value from the other side of the face (processorVolPatchFieldDecomposer), the
                            // face's own value for a surface field, turned over for a reversed face of an oriented one
                            Entry ty;
                            ty.key = "type";
                            ty.v.tokens = {"processor"};
                            pe.sub.push_back(std::move(ty));
                            Entry val;
                            val.key = "value";
                            std::vector<double> d;
                            d.reserve(static_cast<std::size_t>(pt.size) * static_cast<std::size_t>(k.comps));
                            for (label j = 0; j < pt.size; ++j)
                            {
                                const label a = pl.faces[static_cast<std::size_t>(pt.start + j)];
                                const long ai = static_cast<long>(a > 0 ? a : -a) - 1;
                                if (k.vol)
                                {
                                    const long c = a > 0 ? G.neighbour[static_cast<std::size_t>(ai)] : G.owner[static_cast<std::size_t>(ai)];
                                    const double* v = in.at(c);
                                    d.insert(d.end(), v, v + k.comps);
                                }
                                else
                                {
                                    const double* v = in.at(ai);
                                    const double sgn = (oriented && a < 0) ? -1.0 : 1.0;
                                    for (int c = 0; c < k.comps; ++c) d.push_back(sgn * v[c]);
                                }
                            }
                            val.v = makeList(k.type, k.comps, std::move(d));
                            pe.sub.push_back(std::move(val));
                        }
                        b.sub.push_back(std::move(pe));
                    }
                    out.push_back(std::move(b));
                }
                else
                {
                    out.push_back(t);
                }
            }
            writeFieldFile(caseDir + "/processor" + std::to_string(p) + "/" + time + "/" + name, f.cls, name, time, out);
        }
        done.push_back(time + "/" + name);
        if (!quiet) std::printf("brae decompose: %s/%s\n", time.c_str(), name.c_str());
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
void reconstructFields(const std::string& caseDir, const std::vector<ProcLayout>& proc, const GlobalLayout& G, const std::string& time,
                       std::vector<std::string>& done, bool quiet)
{
    const int nProcs = static_cast<int>(proc.size());
    const std::string src0 = caseDir + "/processor0/" + time;
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(src0))
        if (e.is_regular_file()) names.push_back(e.path().filename().string());
    std::sort(names.begin(), names.end());

    for (const std::string& name : names)
    {
        if (name.empty() || name[0] == '.') continue;
        std::vector<FieldFile> fs_(static_cast<std::size_t>(nProcs));
        try
        {
            fs_[0] = readFieldFile(src0 + "/" + name);
        }
        catch (const std::exception& e)
        {
            if (!quiet) std::printf("brae reconstruct: %s/%s is not read as a field (%s): skipped\n", time.c_str(), name.c_str(), e.what());
            continue;
        }
        const FieldKind k = fieldKind(fs_[0].cls);
        if (!k.vol && !k.surface)
        {
            if (!quiet) std::printf("brae reconstruct: %s/%s is not a volume or surface field: skipped\n", time.c_str(), name.c_str());
            continue;
        }
        for (int p = 1; p < nProcs; ++p)
        {
            const std::string path = caseDir + "/processor" + std::to_string(p) + "/" + time + "/" + name;
            if (!fs::exists(path))
                throw std::runtime_error("brae reconstruct: " + path + " does not exist, and processor0 has it.");
            fs_[static_cast<std::size_t>(p)] = readFieldFile(path);
        }

        std::vector<Entry> out;
        for (const Entry& t : fs_[0].top)
        {
            if (t.key == "internalField")
            {
                std::vector<Access> in;
                bool allUniformSame = true;
                for (int p = 0; p < nProcs; ++p)
                {
                    const Entry* e = top(fs_[static_cast<std::size_t>(p)], "internalField");
                    if (!e) throw std::runtime_error("brae reconstruct: processor" + std::to_string(p) + "'s " + name + " has no internalField.");
                    in.emplace_back(e->v, k.comps);
                    if (e->v.kind != Value::Uniform || e->v.tokens != t.v.tokens || t.v.kind != Value::Uniform) allUniformSame = false;
                }
                Entry e;
                e.key = "internalField";
                if (allUniformSame)
                {
                    e = t;
                }
                else
                {
                    const long n = k.vol ? G.nCells : G.nInternalFaces;
                    std::vector<double> d(static_cast<std::size_t>(n) * static_cast<std::size_t>(k.comps), 0.0);
                    long filled = 0;
                    for (int p = 0; p < nProcs; ++p)
                    {
                        const ProcLayout& pl = proc[static_cast<std::size_t>(p)];
                        const std::size_t cnt = k.vol ? pl.cells.size() : static_cast<std::size_t>(pl.nInternal);
                        if (in[static_cast<std::size_t>(p)].size() >= 0 && static_cast<std::size_t>(in[static_cast<std::size_t>(p)].size()) != cnt)
                            throw std::runtime_error("brae reconstruct: processor" + std::to_string(p) + "'s " + name + " internalField has "
                                                     + std::to_string(in[static_cast<std::size_t>(p)].size()) + " values, its mesh has " + std::to_string(cnt) + ".");
                        for (std::size_t i = 0; i < cnt; ++i)
                        {
                            const long g = k.vol ? pl.cells[i] : static_cast<long>(pl.faces[i]) - 1;
                            const double* v = in[static_cast<std::size_t>(p)].at(static_cast<long>(i));
                            std::memcpy(d.data() + static_cast<std::size_t>(g) * static_cast<std::size_t>(k.comps), v, sizeof(double) * static_cast<std::size_t>(k.comps));
                            ++filled;
                        }
                    }
                    if (k.vol && filled != G.nCells)
                        throw std::runtime_error("brae reconstruct: the processors' cells are " + std::to_string(filled) + " and the case has " + std::to_string(G.nCells) + ".");
                    e.v = makeList(k.type, k.comps, std::move(d));
                    if (!k.vol)
                    {
                        // the faces between processors are internal faces of the global mesh: take the owner side's value
                        auto& dd = e.v.data;
                        for (int p = 0; p < nProcs; ++p)
                        {
                            const ProcLayout& pl = proc[static_cast<std::size_t>(p)];
                            for (std::size_t lp = 0; lp < pl.patches.size(); ++lp)
                            {
                                if (pl.boundary[lp] >= 0) continue;
                                const Entry* b = top(fs_[static_cast<std::size_t>(p)], "boundaryField");
                                const Entry* pe = b ? b->find(pl.patches[lp].name) : nullptr;
                                const Entry* val = pe ? pe->find("value") : nullptr;
                                if (!val) throw std::runtime_error("brae reconstruct: processor" + std::to_string(p) + "'s " + name + " has no value on " + pl.patches[lp].name + ".");
                                const Access a(val->v, k.comps);
                                for (label j = 0; j < pl.patches[lp].size; ++j)
                                {
                                    const label fa = pl.faces[static_cast<std::size_t>(pl.patches[lp].start + j)];
                                    if (fa < 0) continue;
                                    std::memcpy(dd.data() + static_cast<std::size_t>(fa - 1) * static_cast<std::size_t>(k.comps), a.at(j), sizeof(double) * static_cast<std::size_t>(k.comps));
                                }
                            }
                        }
                    }
                }
                out.push_back(std::move(e));
            }
            else if (t.key == "boundaryField")
            {
                Entry b;
                b.key = "boundaryField";
                b.isDict = true;
                for (std::size_t g = 0; g < G.patches.size(); ++g)
                {
                    const PatchLayout& gp = G.patches[g];
                    // the processors' sides of this patch: (processor, local patch)
                    struct Side { int p; std::size_t lp; const Entry* pe; };
                    std::vector<Side> sides;
                    for (int p = 0; p < nProcs; ++p)
                    {
                        const ProcLayout& pl = proc[static_cast<std::size_t>(p)];
                        for (std::size_t lp = 0; lp < pl.patches.size(); ++lp)
                        {
                            if (pl.boundary[lp] != static_cast<label>(g)) continue;
                            const Entry* bf = top(fs_[static_cast<std::size_t>(p)], "boundaryField");
                            const Entry* pe = bf ? bf->find(pl.patches[lp].name) : nullptr;
                            if (!pe) throw std::runtime_error("brae reconstruct: processor" + std::to_string(p) + "'s " + name + " has no entry for " + pl.patches[lp].name + ".");
                            sides.push_back({p, lp, pe});
                        }
                    }
                    if (sides.empty()) throw std::runtime_error("brae reconstruct: no processor has patch `" + gp.name + "`.");
                    const Side* tmpl = &sides[0];
                    for (const Side& s : sides)
                        if (proc[static_cast<std::size_t>(s.p)].patches[s.lp].size > 0) { tmpl = &s; break; }
                    Entry pe;
                    pe.key = gp.name;
                    pe.isDict = true;
                    for (const Entry& s : tmpl->pe->sub)
                    {
                        Entry c = s;
                        if (!s.isDict && (s.v.kind == Value::List || s.v.kind == Value::Uniform) && gp.size > 0)
                        {
                            bool uniformEverywhere = (s.v.kind == Value::Uniform);
                            int comps = s.v.kind == Value::List ? s.v.comps : 0;
                            for (const Side& sd : sides)
                            {
                                if (proc[static_cast<std::size_t>(sd.p)].patches[sd.lp].size == 0) continue;
                                const Entry* o = sd.pe->find(s.key);
                                if (!o) throw std::runtime_error("brae reconstruct: processor" + std::to_string(sd.p) + "'s " + name + "/" + gp.name + " lacks `" + s.key + "`.");
                                if (o->v.kind != Value::Uniform || o->v.tokens != s.v.tokens || s.v.kind != Value::Uniform) uniformEverywhere = false;
                                if (o->v.kind == Value::List && o->v.n > 0) comps = o->v.comps;
                            }
                            if (!uniformEverywhere)
                            {
                                if (comps == 0) comps = static_cast<int>(uniformNumbers(s.v).size());
                                std::vector<double> d(static_cast<std::size_t>(gp.size) * static_cast<std::size_t>(comps), 0.0);
                                for (const Side& sd : sides)
                                {
                                    const PatchLayout& lpt = proc[static_cast<std::size_t>(sd.p)].patches[sd.lp];
                                    if (lpt.size == 0) continue;
                                    const Entry* o = sd.pe->find(s.key);
                                    const Access a(o->v, comps);
                                    for (label j = 0; j < lpt.size; ++j)
                                    {
                                        const long gi = static_cast<long>(proc[static_cast<std::size_t>(sd.p)].faces[static_cast<std::size_t>(lpt.start + j)]) - 1 - gp.start;
                                        std::memcpy(d.data() + static_cast<std::size_t>(gi) * static_cast<std::size_t>(comps), a.at(j), sizeof(double) * static_cast<std::size_t>(comps));
                                    }
                                }
                                std::string lt = s.v.listType;
                                c.v = makeList(lt.size() > 6 ? lt.substr(5, lt.size() - 6) : k.type, comps, std::move(d));
                            }
                        }
                        pe.sub.push_back(std::move(c));
                    }
                    b.sub.push_back(std::move(pe));
                }
                out.push_back(std::move(b));
            }
            else
            {
                out.push_back(t);
            }
        }
        writeFieldFile(caseDir + "/" + time + "/" + name, fs_[0].cls, name, time, out);
        done.push_back(time + "/" + name);
        if (!quiet) std::printf("brae reconstruct: %s/%s\n", time.c_str(), name.c_str());
    }
}

}   // namespace detail
}   // namespace decompose
}   // namespace brae
