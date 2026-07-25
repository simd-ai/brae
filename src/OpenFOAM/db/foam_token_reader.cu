#include "foam_token_reader.cuh"
#include "foam_dict.cuh"   // expandDictVariables ($macro expansion) for the expandVars path

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iterator>
#include <filesystem>
#include <zlib.h>    // transparent gzip read of OF's default-gzipped polyMesh (points.gz, faces.gz, ...) and fields

namespace brae {
namespace {

// Read a file into bytes, transparently gunzipping. OF gzips polyMesh/fields by default (`points.gz`, ...); zlib's
// gzread reads BOTH gzip and plain files, so this also handles uncompressed files. Resolves <base> -> <base>.gz when
// <base> itself is absent (prefers the uncompressed file if both exist). maxBytes>0 stops early (header sniffing).
std::vector<char> gzSlurp(const std::string& base, std::size_t maxBytes = 0)
{
    std::string path = base;
    {
        std::ifstream a(base, std::ios::binary);
        if (!a.good())
        {
            std::ifstream g(base + ".gz", std::ios::binary);
            if (g.good()) path = base + ".gz";
        }
    }
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("brae: cannot open " + base + " (or " + base + ".gz)");
    std::vector<char> out;
    char buf[1 << 16];
    int n;
    while ((n = gzread(f, buf, sizeof(buf))) > 0)
    {
        out.insert(out.end(), buf, buf + n);
        if (maxBytes && out.size() >= maxBytes) break;
    }
    gzclose(f);
    return out;
}

std::string readWhole(const std::string& path)
{
    const std::vector<char> b = gzSlurp(path);
    return std::string(b.begin(), b.end());
}

std::string dirOf(const std::string& p)
{
    const auto s = p.find_last_of('/');
    return s == std::string::npos ? "." : p.substr(0, s);
}

// Strip a leading "FoamFile { ... }" header from an included fragment (most fragments have none; full dicts do).
std::string stripFoamHeader(const std::string& s)
{
    const auto f = s.find("FoamFile");
    if (f == std::string::npos) return s;
    const auto b = s.find('{', f);
    if (b == std::string::npos) return s;
    int depth = 0;
    std::size_t i = b;
    for (; i < s.size(); ++i)
    {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}') { if (--depth == 0) { ++i; break; } }
    }
    return s.substr(0, f) + s.substr(i);
}

// Resolve a #includeEtc path against the OpenFOAM etc tree (BRAE_FOAM_ETC / FOAM_ETC / $WM_PROJECT_DIR/etc / install).
std::string resolveEtc(const std::string& rel)
{
    std::vector<std::string> roots;
    if (const char* e = std::getenv("BRAE_FOAM_ETC"))    roots.emplace_back(e);
    if (const char* e = std::getenv("FOAM_ETC"))       roots.emplace_back(e);
    if (const char* e = std::getenv("WM_PROJECT_DIR")) roots.emplace_back(std::string(e) + "/etc");
    roots.emplace_back("/usr/lib/openfoam/openfoam2412/etc");
    for (const auto& r : roots)
    {
        std::ifstream t(r + "/" + rel);
        if (t.good()) return r + "/" + rel;
    }
    return "";
}

// Recursively search the OpenFOAM caseDicts library (etc roots) for a function-object template named `name`
// (OF's functionObjectList search path for #includeFunc). Returns the full path, or "" if not found.
std::string resolveFunc(const std::string& name)
{
    namespace fs = std::filesystem;
    std::vector<std::string> roots;
    if (const char* e = std::getenv("BRAE_FOAM_ETC"))    roots.emplace_back(e);
    if (const char* e = std::getenv("FOAM_ETC"))         roots.emplace_back(e);
    if (const char* e = std::getenv("WM_PROJECT_DIR"))   roots.emplace_back(std::string(e) + "/etc");
    roots.emplace_back("/usr/lib/openfoam/openfoam2412/etc");
    for (const auto& r : roots)
    {
        const std::string base = r + "/caseDicts";
        std::error_code ec;
        if (!fs::is_directory(base, ec)) continue;
        for (fs::recursive_directory_iterator it(base, ec), end; !ec && it != end; it.increment(ec))
            if (it->is_regular_file(ec) && it->path().filename() == name) return it->path().string();
    }
    return "";
}
// Which function objects brae RUNS during a steady simpleFoam solve: forces / force coefficients. Everything else
// (visualization, sampling, transient particle tracks, output-format writers, solverInfo, ...) is skipped and left
// to OpenFOAM's postProcess on brae's output.
bool isSteadyForceFO(const std::string& name)
{
    std::string l;
    l.reserve(name.size());
    for (char c : name) l += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return l.find("force") != std::string::npos;   // forces / forceCoeffs / forceCoeffsIncompressible
}

// Environment-variable substitution in an #include path, matching OF includeEntry::resolveFile's
// stringOps::inplaceExpand(fName, dict, allowEnv=true, allowEmpty=true): $NAME / ${NAME} -> getenv (empty if unset).
// (OF also expands parent-dict variables in the path; those are rare in include paths and need the partial dict,
// which is not available at this text-level pass, env vars cover $FOAM_CASE/$FOAM_TUTORIALS/$WM_PROJECT_DIR etc.)
std::string expandEnvVars(const std::string& s)
{
    if (s.find('$') == std::string::npos) return s;
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); )
    {
        if (s[i] == '$')
        {
            std::size_t j = i + 1;
            std::string name;
            if (j < s.size() && s[j] == '{')
            {
                ++j;
                while (j < s.size() && s[j] != '}') name += s[j++];
                if (j < s.size()) ++j;
            }
            else
            {
                while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) name += s[j++];
            }
            if (!name.empty()) { if (const char* v = std::getenv(name.c_str())) out += v; }   // allowEmpty: unset -> ""
            i = j;
        }
        else out += s[i++];
    }
    return out;
}

// Expand OpenFOAM #include directives (text-level, recursive), byte-faithful to includeEntry/includeEtcEntry:
//   resolveFile: expand $vars/env; if the (expanded) name is absolute use it as-is, else <includingDir>/name.
//   resolveEtcFile (#includeEtc): findEtcFile over the OF etc tree.
//   #include / #includeEtc are MANDATORY, a missing file is a hard error (OF FatalIOError). Only the optional
//   forms #sinclude / #includeIfPresent skip a missing file silently. Bounded recursion.
std::string expandIncludes(const std::string& text, const std::string& baseDir, int depth)
{
    if (depth > 20 || text.find('#') == std::string::npos) return text;
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size())
    {
        if (text[i] == '#')
        {
            std::size_t j = i + 1;
            while (j < text.size() && std::isalpha(static_cast<unsigned char>(text[j]))) ++j;
            const std::string dir = text.substr(i + 1, j - i - 1);
            // brae does not evaluate #calc/#eval/#codeStream: the raw expression string flows into stod/stoi and is
            // silently mis-parsed (e.g. endTime #calc "1000*2" -> 1000). Fail loud rather than run a wrong control value.
            if (dir == "calc" || dir == "eval" || dir == "codeStream")
                throw std::runtime_error("brae does not evaluate #" + dir + " directives -- the expression would be"
                    " silently mis-parsed into a wrong value. Replace it with a literal value.");
            if (dir == "includeFunc")
            {
                // #includeFunc <name>[(args)] (unquoted; OF function-template include). brae only RUNS force/forceCoeffs
                // for steady simpleFoam, resolve those from the case system/ dir then the OpenFOAM caseDicts library.
                // Every other function object (streamLines, cuttingPlane, ensightWrite, solverInfo, ...) is SKIPPED
                // gracefully (run those via OpenFOAM's postProcess on brae's output). Never a hard error.
                std::size_t k = j;
                while (k < text.size() && (text[k] == ' ' || text[k] == '\t')) ++k;
                const bool q = (k < text.size() && text[k] == '"');
                if (q) ++k;
                std::size_t e = k;
                while (e < text.size() && text[e] != '\n' && text[e] != '(' && text[e] != ';'
                       && (q ? text[e] != '"' : (text[e] != ' ' && text[e] != '\t'))) ++e;
                const std::string name = text.substr(k, e - k);
                std::size_t adv = e;
                while (adv < text.size() && text[adv] != '\n') ++adv;   // consume the rest of the line
                if (!name.empty())
                {
                    if (isSteadyForceFO(name))
                    {
                        std::string path = std::ifstream(baseDir + "/" + name).good() ? (baseDir + "/" + name) : resolveFunc(name);
                        std::ifstream f(path);
                        if (f.good())
                        {
                            std::ostringstream ss;
                            ss << f.rdbuf();
                            out += name + "\n{\n" + expandIncludes(stripFoamHeader(ss.str()), dirOf(path), depth + 1) + "\n}\n";
                        }
                        else std::fprintf(stderr, "brae: skipping #includeFunc %s (template not found)\n", name.c_str());
                    }
                    else
                        std::fprintf(stderr, "brae: skipping function object '%s' (not a steady simpleFoam FO brae runs; use OpenFOAM postProcess on the output)\n", name.c_str());
                }
                i = adv;
                continue;
            }
            if (dir == "include" || dir == "sinclude" || dir == "includeIfPresent" || dir == "includeEtc")
            {
                std::size_t k = j;
                while (k < text.size() && (text[k] == ' ' || text[k] == '\t')) ++k;
                if (k < text.size() && text[k] == '"')
                {
                    const std::size_t e = text.find('"', k + 1);
                    if (e != std::string::npos)
                    {
                        const std::string rel = expandEnvVars(text.substr(k + 1, e - k - 1));
                        const bool isEtc    = (dir == "includeEtc");
                        const bool optional = (dir == "sinclude" || dir == "includeIfPresent");
                        std::string path = isEtc ? resolveEtc(rel)
                                         : (!rel.empty() && rel[0] == '/') ? rel        // absolute (fName.isAbsolute())
                                         : (baseDir + "/" + rel);                       // relative -> dir/fName
                        std::ifstream f(path);
                        if (f.good())
                        {
                            std::ostringstream ss;
                            ss << f.rdbuf();
                            out += expandIncludes(stripFoamHeader(ss.str()), dirOf(path), depth + 1);
                        }
                        else if (!optional)
                        {
                            if (isEtc)
                                throw std::runtime_error("brae::TokenStream: cannot find #includeEtc file \"" + rel + "\" in the OpenFOAM etc tree");
                            // graceful: a bare name is almost always a function-object template. Route force FOs to the
                            // caseDicts library; skip anything else (and any genuinely missing file) with a warning
                            // rather than crashing, the functions{} block must never be fatal.
                            const bool bare = rel.find('/') == std::string::npos && rel.find('.') == std::string::npos;
                            if (bare && isSteadyForceFO(rel))
                            {
                                const std::string fp = resolveFunc(rel);
                                std::ifstream ff(fp);
                                if (ff.good())
                                {
                                    std::ostringstream ss;
                                    ss << ff.rdbuf();
                                    out += rel + "\n{\n" + expandIncludes(stripFoamHeader(ss.str()), dirOf(fp), depth + 1) + "\n}\n";
                                }
                                else std::fprintf(stderr, "brae: skipping #include \"%s\" (force FO template not found)\n", rel.c_str());
                            }
                            else if (bare)
                                std::fprintf(stderr, "brae: skipping function object '%s' (not a steady simpleFoam FO brae runs; use OpenFOAM postProcess on the output)\n", rel.c_str());
                            else
                                std::fprintf(stderr, "brae WARNING: #include file \"%s\" not found (resolved: %s), skipping\n", rel.c_str(), path.c_str());
                        }
                        // optional (#sinclude/#includeIfPresent) missing -> skip silently (OF includeIfPresentEntry)
                        i = e + 1;
                        continue;
                    }
                }
            }
        }
        out += text[i++];
    }
    return out;
}

// Remove // line comments and /* ... */ block comments (the banner is one block comment).
std::string stripComments(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    enum { CODE, LINE, BLOCK } st = CODE;
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        const char c = s[i];
        const char d = (i + 1 < s.size()) ? s[i + 1] : '\0';
        if (st == CODE)
        {
            if (c == '/' && d == '/') { st = LINE;  ++i; }
            else if (c == '/' && d == '*') { st = BLOCK; ++i; }
            else out += c;
        }
        else if (st == LINE)
        {
            if (c == '\n') { st = CODE; out += '\n'; }
        }
        else   // BLOCK
        {
            if (c == '*' && d == '/') { st = CODE; ++i; }
        }
    }
    return out;
}

// Split on whitespace; treat ( ) { } ; as standalone tokens even when glued to numbers.
// A "..." string is one atomic token (quotes stripped), OF keyword regexes like "(U|k|epsilon)"
// must NOT be split on their inner ( ), and quoted values (lib names, regexes) stay intact.
std::vector<std::string> tokenize(const std::string& s)
{
    std::vector<std::string> t;
    std::string cur;
    bool inQuote = false;
    auto flush = [&]()
    {
        if (!cur.empty()) { t.push_back(cur); cur.clear(); }
    };
    for (const char c : s)
    {
        if (inQuote)
        {
            if (c == '"') { t.push_back(cur); cur.clear(); inQuote = false; }   // emit quoted token (quotes stripped)
            else cur += c;
        }
        else if (c == '"')
        {
            flush();
            inQuote = true;
        }
        else if (c == '(' || c == ')' || c == '{' || c == '}' || c == ';')
        {
            flush();
            t.emplace_back(1, c);
        }
        else if (std::isspace(static_cast<unsigned char>(c)))
        {
            flush();
        }
        else
        {
            cur += c;
        }
    }
    flush();
    return t;
}

} // namespace

TokenStream::TokenStream(const std::string& path, bool expandVars)
{
    // Expand #include directives at the text level (no-op unless a '#' is present, polyMesh stays untouched/fast).
    std::string txt = expandIncludes(readWhole(path), dirOf(path), 0);
    // Then in-file $variable macros, if requested (field reader). Done AFTER includes so a var defined in an included
    // fragment is visible. expandDictVariables strips comments + substitutes $name/${name}; stripComments below is then
    // a no-op. polyMesh/non-field reads skip this entirely (expandVars=false).
    if (expandVars) txt = expandDictVariables(txt);
    std::vector<std::string> all = tokenize(stripComments(txt));

    // Skip the leading "FoamFile { ... }" header dict, if present.
    std::size_t i = 0;
    while (i < all.size() && all[i] != "FoamFile") ++i;
    if (i < all.size())
    {
        while (i < all.size() && all[i] != "{") ++i;
        int depth = 0;
        for (; i < all.size(); ++i)
        {
            if (all[i] == "{") ++depth;
            else if (all[i] == "}") { if (--depth == 0) { ++i; break; } }
        }
    }
    else
    {
        i = 0; // no header, payload starts at the top
    }
    toks_.assign(all.begin() + i, all.end());
}

const std::string& TokenStream::peek() const
{
    if (pos_ >= toks_.size()) throw std::runtime_error("brae::TokenStream: peek past end");
    return toks_[pos_];
}

std::string TokenStream::next()
{
    if (pos_ >= toks_.size()) throw std::runtime_error("brae::TokenStream: read past end");
    return toks_[pos_++];
}

void TokenStream::expect(const std::string& t)
{
    const std::string got = next();
    if (got != t)
        throw std::runtime_error("brae::TokenStream: expected '" + t + "' got '" + got + "'");
}

label  TokenStream::nextLabel()  { return static_cast<label>(std::strtol(next().c_str(), nullptr, 10)); }
scalar TokenStream::nextScalar() { return std::strtod(next().c_str(), nullptr); }

// Binary format support
namespace {

std::vector<char> readBytes(const std::string& path)
{
    return gzSlurp(path);   // gz-aware (binary polyMesh: owner.gz/faces.gz/...)
}

void skipSpaceAndComments(const std::vector<char>& b, std::size_t& i)
{
    while (i < b.size())
    {
        const char c = b[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++i; }
        else if (c == '/' && i + 1 < b.size() && b[i + 1] == '/')
        {
            while (i < b.size() && b[i] != '\n') ++i;
        }
        else break;
    }
}

// Byte offset of the payload: just past the FoamFile header dict and any comment divider.
std::size_t headerPayloadOffset(const std::vector<char>& b)
{
    const std::string head(b.begin(),
                           b.begin() + std::min(b.size(), static_cast<std::size_t>(8192)));
    std::size_t i = 0;
    const std::size_t hf = head.find("FoamFile");
    if (hf != std::string::npos)
    {
        std::size_t br = head.find('{', hf);
        int depth = 0;
        for (i = br; i < b.size(); ++i)
        {
            if (b[i] == '{') ++depth;
            else if (b[i] == '}') { if (--depth == 0) { ++i; break; } }
        }
    }
    skipSpaceAndComments(b, i);
    return i;
}

// Read an ASCII list count "N (" and leave the cursor just past '('.
std::size_t readAsciiCount(const std::vector<char>& b, std::size_t& i)
{
    skipSpaceAndComments(b, i);
    std::size_t n = 0;
    bool any = false;
    while (i < b.size() && b[i] >= '0' && b[i] <= '9')
    {
        n = n * 10 + (b[i] - '0');
        ++i;
        any = true;
    }
    if (!any) throw std::runtime_error("brae binary: expected list count");
    skipSpaceAndComments(b, i);
    if (i >= b.size() || b[i] != '(') throw std::runtime_error("brae binary: expected '(' after count");
    ++i;
    return n;
}

void expectCloseParen(const std::vector<char>& b, std::size_t& i)
{
    skipSpaceAndComments(b, i);
    if (i >= b.size() || b[i] != ')') throw std::runtime_error("brae binary: expected ')'");
    ++i;
}

template <typename T>
void readRaw(const std::vector<char>& b, std::size_t& i, T* dst, std::size_t n)
{
    const std::size_t nbytes = n * sizeof(T);
    if (i + nbytes > b.size()) throw std::runtime_error("brae binary: truncated list");
    std::memcpy(dst, b.data() + i, nbytes);
    i += nbytes;
}

} // namespace

std::string foamFormat(const std::string& path)
{
    const std::vector<char> b = gzSlurp(path, 4096);   // gz-aware; only the header is needed
    const std::string s(b.begin(), b.end());
    const std::size_t fp = s.find("format");
    if (fp != std::string::npos)
    {
        const std::size_t e = s.find(';', fp);
        if (e != std::string::npos && s.substr(fp, e - fp).find("binary") != std::string::npos)
            return "binary";
    }
    return "ascii";
}

std::vector<vector> readBinaryPoints(const std::string& path)
{
    const std::vector<char> b = readBytes(path);
    std::size_t i = headerPayloadOffset(b);
    const std::size_t n = readAsciiCount(b, i);
    std::vector<vector> pts(n);
    readRaw(b, i, pts.data(), n);      // vector == 3 contiguous fp64 (24 bytes), LSB host
    expectCloseParen(b, i);
    return pts;
}

std::vector<label> readBinaryLabelList(const std::string& path)
{
    const std::vector<char> b = readBytes(path);
    std::size_t i = headerPayloadOffset(b);
    const std::size_t n = readAsciiCount(b, i);
    std::vector<label> v(n);
    readRaw(b, i, v.data(), n);
    expectCloseParen(b, i);
    return v;
}

void readBinaryCompactFaces(const std::string& path, std::vector<label>& offsets, std::vector<label>& verts)
{
    const std::vector<char> b = readBytes(path);
    std::size_t i = headerPayloadOffset(b);
    const std::size_t m = readAsciiCount(b, i);   // nFaces + 1 (compact CSR offsets)
    offsets.resize(m);
    readRaw(b, i, offsets.data(), m);
    expectCloseParen(b, i);
    const std::size_t k = readAsciiCount(b, i);   // total vertices
    verts.resize(k);
    readRaw(b, i, verts.data(), k);
    expectCloseParen(b, i);
}

// Byte-level helpers for the mixed ASCII/binary cellZones parser below.
namespace {

// Skip whitespace/comments, then read a token up to the next whitespace or a ( ) { } ; delimiter.
// Reads compound tokens like "List<label>" whole (the '<' '>' are not delimiters).
std::string readWordBytes(const std::vector<char>& b, std::size_t& i)
{
    skipSpaceAndComments(b, i);
    std::string w;
    while (i < b.size())
    {
        const char c = b[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r'
            || c == '(' || c == ')' || c == '{' || c == '}' || c == ';') break;
        w += c;
        ++i;
    }
    return w;
}

// Skip whitespace/comments, then require and consume the single byte `ch`.
void expectByte(const std::vector<char>& b, std::size_t& i, char ch)
{
    skipSpaceAndComments(b, i);
    if (i >= b.size() || b[i] != ch)
        throw std::runtime_error(std::string("brae binary cellZones: expected '") + ch + "'");
    ++i;
}

// Advance past the next occurrence of `ch` (consuming it). Used to skip flat entries like "type cellZone;".
void skipPastByte(const std::vector<char>& b, std::size_t& i, char ch)
{
    while (i < b.size() && b[i] != ch) ++i;
    if (i < b.size()) ++i;
}

} // namespace

std::vector<std::pair<std::string, std::vector<label>>> readBinaryCellZones(const std::string& path)
{
    const std::vector<char> b = readBytes(path);
    std::size_t i = headerPayloadOffset(b);        // past the FoamFile header (incl. nested meta{})
    const std::size_t nz = readAsciiCount(b, i);   // nZones, cursor left just past the outer '('
    std::vector<std::pair<std::string, std::vector<label>>> zones;
    zones.reserve(nz);
    for (std::size_t z = 0; z < nz; ++z)
    {
        const std::string name = readWordBytes(b, i);   // zone name (glued after '(' / previous '}')
        expectByte(b, i, '{');
        std::vector<label> cells;
        for (;;)
        {
            skipSpaceAndComments(b, i);
            if (i < b.size() && b[i] == '}') { ++i; break; }
            const std::string key = readWordBytes(b, i);
            if (key == "cellLabels")
            {
                // Optional compound token "List<label>" (some writers omit it: `cellLabels N(...)`).
                skipSpaceAndComments(b, i);
                if (i < b.size() && !(b[i] >= '0' && b[i] <= '9'))
                    readWordBytes(b, i);
                const std::size_t n = readAsciiCount(b, i);   // count, cursor just past '('
                cells.resize(n);
                readRaw(b, i, cells.data(), n);               // n raw int32 labels
                expectCloseParen(b, i);
                skipPastByte(b, i, ';');
            }
            else
            {
                skipPastByte(b, i, ';');   // e.g. "type cellZone;"
            }
        }
        zones.emplace_back(name, std::move(cells));
    }
    return zones;
}

} // namespace brae
