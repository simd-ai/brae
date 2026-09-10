#include "foam_token_reader.cuh"
#include "eval_expression.cuh"   // OF #eval scalar expressions
#include "foam_dict.cuh"   // expandDictVariables ($macro expansion) for the expandVars path

#include <fstream>
#include <algorithm>
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

// gzSlurp is EXTERNAL (declared in foam_token_reader.cuh; the mesh reader FastScan uses it too), so it lives
// OUTSIDE the file-local anonymous namespace that begins after it.
// Read a file into bytes, transparently gunzipping. OF gzips polyMesh/fields by default (`points.gz`, ...); zlib's
// gzread reads BOTH gzip and plain files, so this also handles uncompressed files. Resolves <base> -> <base>.gz when
// <base> itself is absent (prefers the uncompressed file if both exist). maxBytes>0 stops early (header sniffing).
std::vector<char> gzSlurp(const std::string& base, std::size_t maxBytes)   // default in foam_token_reader.cuh
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

namespace {   // file-local parser helpers below

// Rewrite the RAW LIST PAYLOADS of a binary foam file as ASCII, leaving the dictionary structure around
// them untouched, so the ordinary TokenStream can read the file exactly as it reads an ASCII one.
//
// OF's binary format is a hybrid: everything is the usual ASCII dictionary except that a list of a
// PRIMITIVE type is written as
//     List<vector> \n 89728 \n ( <89728*3 raw doubles> )
// Word lists (`inGroups 1(wall)`) stay ASCII even in a binary file, so only the primitive types are
// transcoded -- rewriting a word list would corrupt it.
//
// brae already had binary readers for the polyMesh (points/faces/owner/cellZones), each a dedicated
// parser. Fields could not reuse those: a field file is a dictionary whose payloads happen to be binary,
// and it carries #includes, $macros and a boundaryField whose entries the ASCII path already handles.
// Transcoding here means all of that machinery is reused unchanged, and it covers any binary foam
// dictionary rather than just the one file type that prompted it.
// (pimpleFoam/LES/NACA4412 writes 0/U binary; brae stopped with "expected '(' got <raw bytes>".)
std::size_t foamComponentCount(const std::string& type)
{
    if (type == "scalar" || type == "sphericalTensor" || type == "label") return 1;
    if (type == "vector")     return 3;
    if (type == "symmTensor") return 6;
    if (type == "tensor")     return 9;
    return 0;   // word/string/anything else: NOT a raw payload, leave alone
}

std::string foamBinaryToAscii(const std::vector<char>& in)
{
    std::string out;
    out.reserve(in.size() + in.size()/2);
    const std::size_t n = in.size();
    std::size_t i = 0;
    while (i < n)
    {
        // find the next "List<"
        std::size_t p = std::string::npos;
        for (std::size_t q = i; q + 5 <= n; ++q)
            if (in[q]=='L' && in[q+1]=='i' && in[q+2]=='s' && in[q+3]=='t' && in[q+4]=='<') { p = q; break; }
        if (p == std::string::npos) { out.append(in.begin()+i, in.end()); break; }

        std::size_t close = p + 5;
        while (close < n && in[close] != '>') ++close;
        if (close >= n) { out.append(in.begin()+i, in.end()); break; }
        const std::string type(in.begin()+p+5, in.begin()+close);
        const std::size_t nCmpt = foamComponentCount(type);

        std::size_t j = close + 1;                       // just past '>'
        while (j < n && std::isspace((unsigned char)in[j])) ++j;
        std::size_t d0 = j;
        while (j < n && std::isdigit((unsigned char)in[j])) ++j;
        const bool haveCount = (j > d0);
        std::size_t count = 0;
        if (haveCount) count = std::strtoull(std::string(in.begin()+d0, in.begin()+j).c_str(), nullptr, 10);
        std::size_t k = j;
        while (k < n && std::isspace((unsigned char)in[k])) ++k;

        // Not a raw primitive payload (a word list, or no `N (` shape): copy through and continue.
        if (nCmpt == 0 || !haveCount || k >= n || in[k] != '(')
        {
            out.append(in.begin()+i, in.begin()+close+1);
            i = close + 1;
            continue;
        }

        const std::size_t width = (type == "label") ? sizeof(int32_t) : sizeof(double);
        const std::size_t bytes = count * nCmpt * width;
        if (k + 1 + bytes >= n)   // truncated / not actually binary: leave it be
        {
            out.append(in.begin()+i, in.begin()+close+1);
            i = close + 1;
            continue;
        }

        out.append(in.begin()+i, in.begin()+close+1);    // "...List<vector>"
        out += "\n";
        out += std::to_string(count);
        out += "\n(\n";
        const char* raw = in.data() + k + 1;
        char buf[64];
        for (std::size_t e = 0; e < count; ++e)
        {
            if (nCmpt > 1) out += '(';
            for (std::size_t c = 0; c < nCmpt; ++c)
            {
                if (c) out += ' ';
                if (type == "label")
                {
                    int32_t v;
                    std::memcpy(&v, raw + (e*nCmpt + c)*width, sizeof v);
                    out += std::to_string((long long)v);
                }
                else
                {
                    double v;
                    std::memcpy(&v, raw + (e*nCmpt + c)*width, sizeof v);
                    std::snprintf(buf, sizeof buf, "%.17g", v);
                    out += buf;
                }
            }
            if (nCmpt > 1) out += ')';
            out += '\n';
        }
        out += ")";
        i = k + 1 + bytes;                                // the ')' that closes the blob
        if (i < n && in[i] == ')') ++i;
    }
    return out;
}

// True when the FoamFile header of this buffer says `format binary`.
bool foamBufferIsBinary(const std::vector<char>& b)
{
    const std::size_t lim = std::min<std::size_t>(b.size(), 2048);
    const std::string head(b.begin(), b.begin() + lim);
    const std::size_t f = head.find("format");
    if (f == std::string::npos) return false;
    const std::size_t e = head.find(';', f);
    if (e == std::string::npos) return false;
    return head.substr(f, e - f).find("binary") != std::string::npos;
}

std::string readWhole(const std::string& path)
{
    const std::vector<char> b = gzSlurp(path);
    if (foamBufferIsBinary(b)) return foamBinaryToAscii(b);
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
            // #eval / #calc are handled in a LATER pass (expandEvalDirectives), after $macro
            // expansion -- their bodies routinely reference macros (`#eval{ $lRef*0.004 }`, NACA4412),
            // and evaluating here would see a literal '$'. Left untouched by this pass.
            if (dir == "eval") { out += text[i]; ++i; continue; }
            if (dir == "calc" || dir == "codeStream")
                throw std::runtime_error("brae does not evaluate #" + dir + " directives -- OpenFOAM"
                    " COMPILES these (calcEntry.C -> codeStream/dynamicCode), so they may contain"
                    " arbitrary C++, not just arithmetic. #eval IS evaluated; replace #" + dir +
                    " with #eval{...} if the body is a scalar expression, else with a literal value.");
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

// OpenFOAM reads `#{ ... #}` as ONE token, a verbatim string (ISstream::readVerbatim): its comments,
// quotes, `$` and `#include` lines belong to the code, and the first `#}` ends it. brae's pipeline would
// take one apart -- stripComments drops the code's `//` comments, tokenize strips the quotes off a string
// literal and splits on every brace, expandIncludes acts on the code's own `#include` -- so the body is
// lifted out here, before any of that runs, and replaced by a placeholder word the reader redeems through
// TokenStream::verbatim(). `#eval #{ ... #}` is left in place: expandEvalDirectives evaluates it after the
// $macros, which is where its body is meant to be expanded. A `#{` inside a comment or a quoted string is
// not a verbatim opener and is left alone.
std::string captureVerbatim(const std::string& text, std::vector<std::string>& store)
{
    if (text.find("#{") == std::string::npos) return text;
    std::string out;
    out.reserve(text.size());
    enum { CODE, LINE, BLOCK, QUOTE } st = CODE;
    for (std::size_t i = 0; i < text.size(); ++i)
    {
        const char c = text[i];
        const char d = (i + 1 < text.size()) ? text[i + 1] : '\0';
        if (st == LINE)
        {
            if (c == '\n') st = CODE;
            out += c;
            continue;
        }
        if (st == BLOCK)
        {
            if (c == '*' && d == '/') { st = CODE; out += c; out += d; ++i; continue; }
            out += c;
            continue;
        }
        if (st == QUOTE)
        {
            if (c == '\\' && d != '\0') { out += c; out += d; ++i; continue; }
            if (c == '"') st = CODE;
            out += c;
            continue;
        }
        if (c == '/' && d == '/') { st = LINE;  out += c; continue; }
        if (c == '/' && d == '*') { st = BLOCK; out += c; continue; }
        if (c == '"')             { st = QUOTE; out += c; continue; }
        if (c == '#' && d == '{')
        {
            const std::size_t e = text.find("#}", i + 2);
            if (e == std::string::npos)
                throw std::runtime_error("brae: unterminated #{ ... #} verbatim block");
            // #eval's own block: keep it, whole, for expandEvalDirectives.
            std::size_t b = out.size();
            while (b > 0 && (out[b - 1] == ' ' || out[b - 1] == '\t')) --b;
            const bool isEval = b >= 5 && out.compare(b - 5, 5, "#eval") == 0;
            if (isEval)
            {
                out.append(text, i, e + 2 - i);
            }
            else
            {
                out += " __brae_verbatim_" + std::to_string(store.size()) + "__ ";
                store.push_back(text.substr(i + 2, e - i - 2));
            }
            i = e + 1;
            continue;
        }
        out += c;
    }
    return out;
}

} // namespace

// #eval{...} / #calc "..." -- OF src/OpenFOAM/expressions. Run AFTER $macro expansion, because the
// bodies reference macros. An expression that cannot be evaluated exactly THROWS from evalExpression;
// it is never defaulted, since an #eval that quietly became 0 is a silently wrong control value.
std::string expandEvalDirectives(const std::string& text)
{
    if (text.find('#') == std::string::npos) return text;
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); )
    {
        if (text[i] == '#')
        {
            std::size_t j = i + 1;
            while (j < text.size() && std::isalpha(static_cast<unsigned char>(text[j]))) ++j;
            const std::string dir = text.substr(i + 1, j - i - 1);
            // #eval ONLY. OF's #calc is a DIFFERENT mechanism: calcEntry.C includes codeStream.H and
            // dynamicCode.H -- it COMPILES C++ and can contain arbitrary code, not just arithmetic.
            // Evaluating it as an expression would work for `#calc "2*3"` and silently diverge on
            // anything else, so it stays refused with #codeStream.
            if (dir == "eval")
            {
                std::size_t k = j;
                while (k < text.size() && (text[k] == ' ' || text[k] == '\t')) ++k;
                // Three delimiter forms, all of which OF accepts:
                //   #eval{ expr }      #eval "expr"      #eval #{ expr #}
                // The last is OF's verbatim-block form and is used by LES/planeChannel.
                std::size_t bodyStart = 0, bodyEnd = 0, next = 0;
                if (k + 1 < text.size() && text[k] == '#' && text[k + 1] == '{')
                {
                    const std::size_t e = text.find("#}", k + 2);
                    if (e == std::string::npos)
                        throw std::runtime_error("brae: unterminated #" + dir + " #{ ... #} expression");
                    bodyStart = k + 2; bodyEnd = e; next = e + 2;
                }
                else
                {
                    const char open = (k < text.size() ? text[k] : '\0');
                    const char close = (open == '{') ? '}' : (open == '"' ? '"' : '\0');
                    if (close == '\0')
                        throw std::runtime_error("brae: #" + dir + " must be followed by {expression}, "
                            "\"expression\" or #{ expression #}; brae evaluates scalar expressions only.");
                    std::size_t e = k + 1;
                    int depth = 1;
                    while (e < text.size())
                    {
                        if (open == '{' && text[e] == '{') ++depth;
                        else if (text[e] == close && (open != '{' || --depth == 0)) break;
                        ++e;
                    }
                    if (e >= text.size())
                        throw std::runtime_error("brae: unterminated #" + dir + " expression");
                    bodyStart = k + 1; bodyEnd = e; next = e + 1;
                }
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.17g",
                              static_cast<double>(evalExpression(text.substr(bodyStart, bodyEnd - bodyStart))));
                out += buf;
                i = next;
                continue;
            }
        }
        out += text[i];
        ++i;
    }
    return out;
}

TokenStream::TokenStream(const std::string& path, bool expandVars)
{
    // Verbatim blocks first -- see captureVerbatim -- and once more after the includes, for a block that
    // arrives inside an included fragment.
    std::string txt = captureVerbatim(readWhole(path), verbatim_);
    // Expand #include directives at the text level (no-op unless a '#' is present, polyMesh stays untouched/fast).
    txt = captureVerbatim(expandIncludes(txt, dirOf(path), 0), verbatim_);
    // Then in-file $variable macros, if requested (field reader). Done AFTER includes so a var defined in an included
    // fragment is visible. expandDictVariables strips comments + substitutes $name/${name}; stripComments below is then
    // a no-op. polyMesh/non-field reads skip this entirely (expandVars=false).
    if (expandVars) txt = expandDictVariables(txt);
    txt = expandEvalDirectives(txt);   // after $macros: #eval bodies reference them
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

bool TokenStream::verbatim(const std::string& token, std::string& body) const
{
    static const std::string head = "__brae_verbatim_";
    if (token.size() <= head.size() + 2 || token.compare(0, head.size(), head) != 0) return false;
    const std::size_t n = std::strtoul(token.c_str() + head.size(), nullptr, 10);
    if (n >= verbatim_.size()) return false;
    body = verbatim_[n];
    return true;
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
                // An EMPTY list is written as `cellLabels List<label> 0;` -- a count with NO
                // parenthesised body at all, which is what OF emits for a zone that matched no cells
                // (simpleFoam/turbineSiting after a serial topoSet). Demanding '(' unconditionally
                // threw "expected '(' after count" and took the whole case down on a legal file.
                {
                    std::size_t probe = i;
                    skipSpaceAndComments(b, probe);
                    std::size_t cnt = 0; bool any = false;
                    while (probe < b.size() && b[probe] >= '0' && b[probe] <= '9')
                    { cnt = cnt*10 + (b[probe]-'0'); ++probe; any = true; }
                    skipSpaceAndComments(b, probe);
                    if (any && cnt == 0 && probe < b.size() && b[probe] == ';')
                    {
                        i = probe;                 // empty zone: nothing to read
                        skipPastByte(b, i, ';');
                        continue;
                    }
                }
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
