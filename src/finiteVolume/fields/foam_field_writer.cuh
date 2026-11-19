#pragma once
// brae::writeVolField, write a converged volume field in OpenFOAM's own structure. Unlike a raw text splice, this
// emits a fully RESOLVED field like OpenFOAM does: the internalField is the solved nonuniform list, and the
// boundaryField is written per mesh patch with an explicit `type` (+ value) entry, patch groups expanded, #include /
// #includeEtc / $macros / $internalField all resolved. So the file loads in any OpenFOAM reader (paraFoam, the
// built-in ParaView reader, postProcess, foamToVTK) exactly like a case OpenFOAM wrote itself.
#include "cf_types.cuh"
#include "foam_field_reader.cuh"   // readField -> FieldData/PatchFieldData (resolves includes/macros/$internalField)
#include "foam_dict.cuh"           // compileFoamRegex, isConstraintPatchType (same matching as buildField)
#include "foam_token_reader.cuh"   // gzSlurp: the template field may be gzipped (OF writeCompression on)
#include "fv_patch.cuh"            // FvPatch (name / type / inGroups)
#include "field_write_format.cuh"   // FieldWriteFormat: binary, fast
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <thread>

namespace brae {

// THE WRITE PATH'S OPTIONS AND COUNTERS (2026-11-19, the 66 M cell DrivAer I/O study). A driver that takes the options sets them once,
// before its first write; unset, every file is written as it always was.
//   binary  OpenFOAM's `writeFormat binary`: each nonuniform list as its raw little-endian doubles, as v2412 writes one (" \nN\n(",
//           the bytes, ")"; an empty list " 0"), the header `format binary` with OpenFOAM's arch line, so a restart reads back the
//           exact doubles. Only a driver all of whose fields go through these functions sets it (the multi-GPU driver).
//   fast    an ASCII list's values formatted in blocks by std::to_chars instead of one operator<< a value (by BRAE_WRITE_THREADS
//           threads on a long list), the file built in memory and written in one call. The same bytes: to_chars general at a
//           precision is printf's %.*g, which is what an ostream prints a double as at that precision.
// Either keeps a template's header and boundary between writes (keyed by its path, size and modification time) instead of re-reading
// and re-parsing it at every write: 3.8 s a write a rank on the 66 M DrivAer's start fields.
// The counters, in milliseconds and bytes, are read and cleared by the caller's timing report.
struct FieldWriteStats
{
    double templateMs = 0.0;   // the template read and parsed, or found in the cache
    double formatMs = 0.0;     // the values formatted; on the streaming path formatted AND written, the two interleaved
    double fileMs = 0.0;       // the file opened, written and closed (the in-memory path only)
    std::uint64_t bytes = 0;
    int files = 0;
};
// a thread's own: a background writer (the multi-GPU driver's BRAE_WRITE_ASYNC) counts its writes apart from the step's thread
inline FieldWriteStats& fieldWriteStats()
{
    static thread_local FieldWriteStats s;
    return s;
}
inline double writeMsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
// the in-memory file: an ostream's bytes appended to a string, written in one call by writeFoamFile
class FoamFileBuffer : public std::streambuf
{
public:
    std::string data;

protected:
    int_type overflow(int_type c) override
    {
        if (c != traits_type::eof())
        {
            data.push_back(static_cast<char>(c));
        }
        return c;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override
    {
        data.append(s, static_cast<std::size_t>(n));
        return n;
    }
};
inline void writeFoamFile(const std::string& path, const std::string& data)
{
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path + ": " + std::strerror(errno));
    const std::size_t written = std::fwrite(data.data(), 1, data.size(), f);
    const int closed = std::fclose(f);
    if (written != data.size() || closed != 0)
        throw std::runtime_error("cannot write " + path + ": " + std::strerror(errno));
    FieldWriteStats& s = fieldWriteStats();
    s.fileMs += writeMsSince(t0);
    s.bytes += data.size();
    s.files += 1;
}
// a file's body done: the in-memory one written in one call (its formatting and its file system time counted apart), the streamed one
// closed (the two counted together, as they happen together)
inline void finishFoamFile(
    const std::string& path,
    bool inMemory,
    FoamFileBuffer& buffer,
    std::ofstream& file,
    std::chrono::steady_clock::time_point t0)
{
    FieldWriteStats& s = fieldWriteStats();
    if (inMemory)
    {
        s.formatMs += writeMsSince(t0);
        writeFoamFile(path, buffer.data);
        return;
    }
    const std::streamoff at = file.tellp();
    file.close();
    if (!file) throw std::runtime_error("cannot write " + path);
    s.formatMs += writeMsSince(t0);
    s.bytes += at > 0 ? static_cast<std::uint64_t>(at) : 0;
    s.files += 1;
}
// a FoamFile header made to say what follows it: `format binary` with OpenFOAM's arch line (added where the header has none), or
// `format ascii` (a binary template's header, the lists written as text)
inline std::string foamHeaderInFormat(const std::string& header, bool binary)
{
    if (!binary)
    {
        static const std::regex binRe("format\\s+binary\\s*;");
        return std::regex_replace(header, binRe, std::string("format      ascii;"));
    }
    static const std::regex fmtRe("format\\s+(ascii|binary)\\s*;");
    std::string h = std::regex_replace(header, fmtRe, std::string("format      binary;"));
    if (h.find("arch") == std::string::npos)
    {
        const std::size_t at = h.find("format      binary;");
        const std::string arch = "\n    arch        \"LSB;label=" + std::to_string(8 * sizeof(label)) + ";scalar="
                               + std::to_string(8 * sizeof(scalar)) + "\";";
        if (at != std::string::npos)
        {
            h.insert(at + std::strlen("format      binary;"), arch);
        }
    }
    return h;
}

inline void formatFoamValue(std::ostream& os, scalar v) { os << v; }
inline void formatFoamValue(std::ostream& os, const vector& v) { os << '(' << v.x << ' ' << v.y << ' ' << v.z << ')'; }
inline const char* foamListType(scalar) { return "List<scalar>"; }
inline const char* foamListType(const vector&) { return "List<vector>"; }
inline void formatFoamValue(std::ostream& os, const symmTensor& t)
{
    os << '(' << t.xx << ' ' << t.xy << ' ' << t.xz << ' ' << t.yy << ' ' << t.yz << ' ' << t.zz << ')';
}
inline const char* foamListType(const symmTensor&) { return "List<symmTensor>"; }
inline const char* foamClassName(const symmTensor&) { return "volSymmTensorField"; }
inline const char* foamClassName(scalar) { return "volScalarField"; }
inline const char* foamClassName(const vector&) { return "volVectorField"; }

// one value as text appended to a buffer: what `os << std::setprecision(p) << v` prints
inline void appendFoamValue(std::string& s, scalar v, int precision)
{
    char b[48];
    const std::to_chars_result r = std::to_chars(b, b + sizeof(b), v, std::chars_format::general, precision);
    s.append(b, r.ptr);
}
inline void appendFoamValue(std::string& s, const vector& v, int precision)
{
    s += '(';
    appendFoamValue(s, v.x, precision);
    s += ' ';
    appendFoamValue(s, v.y, precision);
    s += ' ';
    appendFoamValue(s, v.z, precision);
    s += ')';
}
inline void appendFoamValue(std::string& s, const symmTensor& t, int precision)
{
    s += '(';
    appendFoamValue(s, t.xx, precision);
    s += ' ';
    appendFoamValue(s, t.xy, precision);
    s += ' ';
    appendFoamValue(s, t.xz, precision);
    s += ' ';
    appendFoamValue(s, t.yy, precision);
    s += ' ';
    appendFoamValue(s, t.yz, precision);
    s += ' ';
    appendFoamValue(s, t.zz, precision);
    s += ')';
}
// a list's values as text, each followed by `sep`, at the stream's precision: one operator<< a value, or (fast) formatted in blocks
template <typename T>
inline void writeFoamValues(
    std::ostream& os,
    const T* v,
    std::size_t n,
    char sep)
{
    if (!fieldWriteFormat().fast)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            formatFoamValue(os, v[i]);
            os << sep;
        }
        return;
    }
    const int precision = static_cast<int>(os.precision());
    // BRAE_WRITE_THREADS (default 8): a long list's values formatted by that many threads, each its own contiguous slice into its own
    // text, the texts then written in order -- the same bytes as one thread's (a value's text depends on the value alone). The 66 M
    // DrivAer's write was 18 s of one core's formatting a rank in 4.2 GB (Arrhenius job 5496607), each rank with 72 cores idle
    static const unsigned threads = []()
    {
        const char* e = std::getenv("BRAE_WRITE_THREADS");
        const int t = e ? std::atoi(e) : 8;
        return static_cast<unsigned>(t < 1 ? 1 : (t > 64 ? 64 : t));
    }();
    const std::size_t perThread = std::size_t(1) << 18;
    if (threads > 1 && n >= 2 * perThread)
    {
        const std::size_t t = std::min<std::size_t>(threads, n / perThread);
        std::vector<std::string> text(t);
        std::vector<std::thread> pool;
        pool.reserve(t);
        for (std::size_t k = 0; k < t; ++k)
        {
            pool.emplace_back([&, k]()
            {
                const std::size_t lo = n * k / t;
                const std::size_t hi = n * (k + 1) / t;
                std::string& b = text[k];
                b.reserve((hi - lo) * 16);
                for (std::size_t i = lo; i < hi; ++i)
                {
                    appendFoamValue(b, v[i], precision);
                    b += sep;
                }
            });
        }
        for (std::thread& th : pool)
        {
            th.join();
        }
        for (const std::string& b : text)
        {
            os.write(b.data(), static_cast<std::streamsize>(b.size()));
        }
        return;
    }
    const std::size_t block = std::size_t(1) << 20;
    std::string b;
    b.reserve(block + 512);
    for (std::size_t i = 0; i < n; ++i)
    {
        appendFoamValue(b, v[i], precision);
        b += sep;
        if (b.size() >= block)
        {
            os.write(b.data(), static_cast<std::streamsize>(b.size()));
            b.clear();
        }
    }
    os.write(b.data(), static_cast<std::streamsize>(b.size()));
}
// a nonuniform list in OpenFOAM's binary form, after its `List<T>` word: " \nN\n(" the raw bytes ")"; an empty one " 0"
template <typename T>
inline void writeFoamBinaryList(
    std::ostream& os,
    const T* v,
    std::size_t n)
{
    static_assert(sizeof(scalar) == 8 && sizeof(T) % sizeof(scalar) == 0, "a binary list is written as doubles");
    if (n == 0)
    {
        os << " 0";
        return;
    }
    os << " \n" << n << "\n(";
    os.write(reinterpret_cast<const char*>(v), static_cast<std::streamsize>(n * sizeof(T)));
    os << ')';
}
// a nonuniform list entry from after its `List<T>` word to its ';' and the line's end: " \nN\n(\n" one value a line ")\n;\n" in ASCII,
// OpenFOAM's binary list then ";\n" in binary
template <typename T>
inline void writeFoamListEntry(
    std::ostream& os,
    const T* v,
    std::size_t n)
{
    if (fieldWriteFormat().binary)
    {
        writeFoamBinaryList(os, v, n);
        os << ";\n";
        return;
    }
    os << " \n" << n << "\n(\n";
    writeFoamValues(os, v, n, '\n');
    os << ")\n;\n";
}

// Write "uniform <v>;" or "nonuniform List<...> N (...);" for a boundary entry field.
// OpenFOAM's FoamFile `location` is the TIME DIRECTORY NAME -- `location "269";` -- never a path. brae's
// surface writer put the whole output path there, so two otherwise identical runs in different
// directories wrote different phi files (tests/gs_device_loop_identity caught it as a byte diff), and
// the vol writer echoed its 0/ template's `location "0"` into every later time. Both now write what
// OpenFOAM writes. Readers ignore the entry; a diff does not.
inline std::string timeDirName(const std::string& outPath)
{
    const std::size_t slash = outPath.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".") : outPath.substr(0, slash);
    const std::size_t s2 = dir.find_last_of('/');
    return (s2 == std::string::npos) ? dir : dir.substr(s2 + 1);
}

template <typename T>
inline void writeFieldValue(
    std::ostream& os,
    bool uniform,
    const T& uval,
    const std::vector<T>& vals)
{
    if (uniform)
    {
        os << "uniform ";
        formatFoamValue(os, uval);
        os << ";\n";
    }
    else if (fieldWriteFormat().binary)
    {
        os << "nonuniform " << foamListType(T{});
        writeFoamBinaryList(os, vals.data(), vals.size());
        os << ";\n";
    }
    else
    {
        os << "nonuniform " << foamListType(T{}) << " " << vals.size() << "(";
        writeFoamValues(os, vals.data(), vals.size(), ' ');
        os << ");\n";
    }
}

// One boundaryField patch entry in OpenFOAM structure: type, then the value entries the reader resolved (inletValue
// for inletOutlet/mixed, value for value-holding BCs). BCs that hold no value (zeroGradient/slip/symmetry/...) write
// just the type, matching OpenFOAM's output.
// The `expression` PatchFunction1 dictionary, written back as it was read: OpenFOAM's PatchExprField::
// writeData echoes dict_ (PatchFunction1Expression.C:147), so a restart from brae's output -- by OpenFOAM
// or by brae -- re-reads the same expression, variables and functions<> tables. The functions<> tokens
// are re-emitted one entry per line. The tokenizer strips quotes, so a token that could only have come
// from a quoted string -- one with whitespace, or a multi-character one holding a delimiter, like the
// regex "(?i).*walls" -- is quoted again; the single-character delimiters `{ } ( ) ;` are the
// tokenizer's own and are written bare. (OpenFOAM read brae's output with those quoted, at the
// functionObjectTrigger's `{`, as "Unexpected '}' while reading dictionary entry".)
inline void writePatchExprDict(std::ostream& os, const PatchExprSpec& e)
{
    auto quoteIfNeeded = [](const std::string& t)
    {
        const bool delimiter = t.size() == 1 && std::string("(){};").find(t[0]) != std::string::npos;
        const bool needs = t.empty()
                        || (!delimiter && t.find_first_of(" \t\r\n(){};\"$#") != std::string::npos);
        return needs ? "\"" + t + "\"" : t;
    };
    os << "        uniformValue\n        {\n            type            expression;\n";
    for (const PatchExprSpec::FunctionDict& fd : e.functionDicts)
    {
        os << "            " << fd.key << "\n            {\n";
        int depth = 1;
        bool lineStart = true;
        for (const std::string& t : fd.tokens)
        {
            if (t == "}") --depth;
            if (lineStart) os << std::string(static_cast<std::size_t>(12 + 4 * depth), ' ');
            else           os << ' ';
            os << quoteIfNeeded(t);
            lineStart = (t == ";" || t == "{" || t == "}");
            if (lineStart) os << '\n';
            if (t == "{") ++depth;
        }
        if (!lineStart) os << '\n';
        os << "            }\n";
    }
    os << "            variables\n            (\n";
    for (const std::string& v : e.variables) os << "                \"" << v << "\"\n";
    os << "            );\n"
       << "            expression\n            #{" << e.expression << "#};\n"
       << "        }\n";
}

template <typename T>
inline void writePatchEntry(
    std::ostream& os,
    const std::string& name,
    const PatchFieldData<T>& d,
    const T* computed = nullptr,     // solver's boundary values for this patch, or null to echo the input
    std::size_t nComputed = 0,
    const std::string* extra = nullptr)   // entries a solver writes for this patch, before `value`; null: none
{
    os << "    " << name << "\n    {\n";
    os << "        type            " << (d.type.empty() ? "zeroGradient" : d.type) << ";\n";
    // totalPressure, totalTemperature and waveTransmissive echo the entries OpenFOAM's readers require, under their own
    // names, so OpenFOAM restarts from brae's output. totalPressureFvPatchScalarField::write emits rho, psi, gamma and p0
    // in every form (.C:238-248), and its dictionary constructor REQUIRES p0: this wrote the psi form alone so (milestone
    // H), and every other form's p0 as `inletValue`, which OpenFOAM stops on ("Entry 'p0' not found"; movingCone's rho
    // form, found 2026-11-19, and any simpleFoam, pimpleFoam or rhoSimpleFoam case with a totalPressure outlet or inlet).
    // uniformTotalPressure writes the same entries with p0 a Function1 (uniformTotalPressureFvPatchScalarField.C:222-232),
    // and fanPressure totalPressure's plus its curve and direction (fanPressureFvPatchScalarField.C:223-235). Both were
    // written as `type` + `inletValue` + `value`: OpenFOAM stopped on either ("Missing or invalid Function1 entry: p0";
    // "Required entry 'p0' missing"), brae refused its own fanPressure output ("has no `fanCurve` entry"), and brae
    // restarted pimpleFoam/RAS/TJunction from its own output with p0 held at p0(0) = 10 where the table ramps it to 40,
    // saying nothing (2026-11-19).
    const bool uniformTP = d.type == "uniformTotalPressure";
    const bool totalPsi  = d.type == "totalPressure" || uniformTP || d.type == "fanPressure"
                        || d.type == "totalTemperature";
    if (totalPsi || d.type == "waveTransmissive")
    {
        if (d.type == "waveTransmissive" && !d.advFieldName.empty())
            os << "        field           " << d.advFieldName << ";\n";
        if (d.phiName != "phi") os << "        phi             " << d.phiName << ";\n";
        if (d.type != "totalTemperature") os << "        rho             " << d.flowRateRhoName << ";\n";
        os << "        psi             " << d.psiName << ";\n";
        os << "        gamma           " << d.gammaTP << ";\n";
        if (d.type == "waveTransmissive")
        {
            if (d.hasAdvFieldInf) os << "        fieldInf        " << d.advFieldInf << ";\n";
            if (d.advLInf > -1e299) os << "        lInf            " << d.advLInf << ";\n";
        }
    }
    // surfaceNormalFixedValue's scalar refValue (U_b = refValue * n): OpenFOAM REQUIRES it (surfaceNormalFixedValueFvPatchVectorField.C
    // reads it with the patch dictionary's get, and writes it before value), and so does brae's reader ("surfaceNormalFixedValue without
    // the refValue entry OpenFOAM requires"). It is an input the solve does not change, so it is echoed as read. Without it neither code
    // started from brae's own write: the 66 M AutoCFD5 DrivAer's inlet, from brae's steady write (Arrhenius job 5495841, 2026-11-19)
    if (d.hasNormalRef && d.type == "surfaceNormalFixedValue")
    {
        os << "        refValue        ";
        writeFieldValue(os, d.normalRefUniform, d.normalRefUniformValue, d.normalRefValues);
    }
    if (uniformTP && d.hasInletValue)
    {
        // Function1::writeData: a table as `p0 table;` + `p0Coeffs { values ...; }` (TableBase::writeEntries), which the
        // reader takes back (its p0Coeffs entry); any other p0 the reader accepts for this type is a constant
        const auto& pts = d.p0Function1.entries();
        if (d.hasP0Function1 && !d.p0Function1.isCoded() && pts.size() > 1)
        {
            os << "        p0              table;\n        p0Coeffs\n        {\n            values          "
               << pts.size() << "\n(\n";
            for (const auto& e : pts) os << '(' << e.first << ' ' << e.second << ")\n";
            os << ")\n;\n        }\n";
        }
        else
        {
            os << "        p0              constant ";
            formatFoamValue(os, d.inletUniformValue);
            os << ";\n";
        }
    }
    else if (d.hasInletValue)
    {
        os << "        " << ((totalPsi && !d.totalRefKey.empty()) ? d.totalRefKey + std::string(14 - d.totalRefKey.size(), ' ')
                                                                   : std::string("inletValue      "));
        writeFieldValue(os, d.inletUniform, d.inletUniformValue, d.inletValues);
    }
    if (d.type == "fanPressure")
    {
        // fanCurve_->writeData (`fanCurve table;` + `fanCurveCoeffs`) and `direction`, which pimpleFoam reads back from
        // the start time's p (fan_pressure.cuh) as OpenFOAM's constructor does (.C:91-107)
        if (!d.fanCurveType.empty() || !d.fanCurveFile.empty() || !d.fanCurveValues.empty())
        {
            os << "        fanCurve        " << (d.fanCurveType.empty() ? "table" : d.fanCurveType) << ";\n"
               << "        fanCurveCoeffs\n        {\n";
            if (!d.fanCurveFile.empty()) os << "            file            \"" << d.fanCurveFile << "\";\n";
            if (!d.fanCurveValues.empty())
            {
                os << "            values          " << d.fanCurveValues.size() << "\n(\n";
                for (const auto& e : d.fanCurveValues) os << '(' << e.first << ' ' << e.second << ")\n";
                os << ")\n;\n";
            }
            if (!d.fanCurveOutOfBounds.empty())   os << "            outOfBounds     " << d.fanCurveOutOfBounds << ";\n";
            if (!d.fanCurveInterpolation.empty()) os << "            interpolationScheme " << d.fanCurveInterpolation << ";\n";
            os << "        }\n";
        }
        if (!d.fanDirection.empty()) os << "        direction       " << d.fanDirection << ";\n";
    }
    // fixedGradient's gradient is REQUIRED by OF's reader, not optional like value: omitting it made
    // OF abort with "Required entry 'gradient' : missing for patch ..." on any attempt to read brae's
    // output back. The gradient is an input that the solve does not change, so echo what was read.
    if (d.hasGradient)
    {
        os << "        gradient        ";
        writeFieldValue(os, d.gradientUniform, d.gradientUniformValue, d.gradientValues);
    }
    // flowRateInletVelocity: OF REQUIRES one of volumetricFlowRate/massFlowRate and aborts without it,
    // and so does brae ("has neither 'volumetricFlowRate' nor 'massFlowRate'"). Dropping it made brae's
    // own output unrestartable -- by brae OR by OpenFOAM -- which only surfaced once brae started
    // writing phi and a brae->brae restart was attempted at all. Same class as the `gradient` entry
    // above: an INPUT the solve does not change, so echo what was read. OF's own output writes the
    // Function1 in its "constant <v>" form.
    // uniformFixedValue's `uniformValue` is a PatchFunction1 that OF's reader REQUIRES, exactly like the
    // gradient above: without it PatchFunction1::New aborts with "Missing or invalid PatchFunction1
    // entry: uniformValue", so brae's own output could be read back by neither OpenFOAM nor brae. It is
    // an INPUT the solve does not change, so echo what was read, in the `constant <v>` form OF writes.
    if (d.hasUniformFn1)
    {
        os << "        uniformValue    constant ";
        formatFoamValue(os, d.uniformFn1Value);
        os << ";\n";
    }
    if (d.hasPatchExpr) writePatchExprDict(os, d.patchExpr);
    // atmBoundaryLayerInlet{Velocity,K,Epsilon,Omega}: OF builds flowDir, zDir, Uref, Zref, z0 and d as
    // Function1/PatchFunction1 and REQUIRES every one of them -- reading a field back without them aborts
    // with "Missing or invalid Function1 entry: flowDir". The tutorial keeps them in an #include that the
    // written field cannot refer to, so they are echoed inline, in the `constant <v>` form OF writes.
    if (d.hasABL)
    {
        os << "        flowDir         constant ";
        formatFoamValue(os, d.ablFlowDir);
        os << ";\n        zDir            constant ";
        formatFoamValue(os, d.ablZDir);
        os << ";\n"
           << "        Uref            constant " << d.ablUref << ";\n"
           << "        Zref            constant " << d.ablZref << ";\n"
           << "        z0              constant " << d.ablZ0   << ";\n"
           << "        d               constant " << d.ablD    << ";\n"
           << "        kappa           " << d.ablKappa << ";\n"
           << "        Cmu             " << d.ablCmu   << ";\n"
           // OF always writes C1/C2 (atmBoundaryLayer.C write); losing a non-default pair on a
           // roundtrip would silently reset the YGCJ profile to the flat one.
           << "        C1              " << d.ablC1    << ";\n"
           << "        C2              " << d.ablC2    << ";\n";
    }
    // atmNutkWallFunction takes its roughness length as a PatchFunction1 that OF REQUIRES, and z0 IS the
    // terrain: a field written without it cannot be read back, and a z0 quietly lost would turn a rough
    // wall into a smooth one. `boundNut` is echoed because its default (true) is not what every case asks.
    if (d.type == "atmNutkWallFunction")
    {
        os << "        z0              constant " << d.ablZ0 << ";\n"
           << "        boundNut        " << (d.atmBoundNut ? "true" : "false") << ";\n";
    }
    if (d.hasFlowRate)
    {
        const char* rateKey = d.flowRateIsMass ? "massFlowRate" : "volumetricFlowRate";
        if (const CodedFunction1Spec* cs = d.flowRateFunction1.codedSpec())
        {
            // OpenFOAM writes the coded dictionary back as it read it (CodedFunction1.C writeData,
            // dict_.writeEntry), so a restart recompiles the same body. Writing the number the rate had
            // at this time instead would turn a function of time into a constant for every run that
            // restarts from here -- OpenFOAM's or brae's.
            os << "        " << rateKey << "\n        {\n"
               << "            type            coded;\n"
               << "            name            " << cs->name << ";\n"
               << "            code\n            #{" << cs->code << "#};\n"
               << "        }\n";
        }
        else
        {
            os << "        " << rateKey << "    constant " << d.flowRate << ";\n";
        }
        // OF guards both of these with `if (!volumetric_)` (flowRateInletVelocityFvPatchVectorField.C:
        // 245-249) -- i.e. on the MASS form -- and writes `rho` only when it is not the default, through
        // writeEntryIfDifferent. `rho none` is not cosmetic: it selects OpenFOAM's VOLUMETRIC branch for
        // a massFlowRate (.C:208), so losing it on a round-trip changes the prescribed inlet with no
        // message on either side. Measured on validation/rhoFR with `rho none`: brae runs the inlet at
        // 6.000 m/s, wrote the patch without it, and real OpenFOAM restarted from that file computed
        // 5.160413015 -- 14% of the flow rate.
        if (d.flowRateIsMass)
        {
            if (d.flowRateRhoName != "rho")
                os << "        rho             " << d.flowRateRhoName << ";\n";
            if (d.rhoInlet >= 0) os << "        rhoInlet        " << d.rhoInlet << ";\n";
        }
    }
    // turbulentIntensityKineticEnergyInlet and turbulentMixingLength{DissipationRate,Frequency}Inlet take
    // their coefficient with dict.get<scalar>(), which THROWS when the key is absent
    // (turbulentIntensityKineticEnergyInletFvPatchScalarField.C:76), and OpenFOAM's own write() emits it
    // unconditionally (.C:154-158). Dropping it made brae's output unreadable by OpenFOAM outright --
    // `--> FOAM FATAL IO ERROR: Entry 'intensity' not found in dictionary ".../k/boundaryField/inlet"`,
    // measured by restarting real rhoSimpleFoam from brae's own t=2 write of validation/rhoTI. brae could
    // read it back only because ITS reader defaults intensity to 0, which is a silently laminar inlet.
    // OF's `U`, `phi` and `k` entries beside these are getOrDefault, so their absence is not an error.
    if (d.type == "turbulentIntensityKineticEnergyInlet")
        os << "        intensity       " << d.intensity << ";\n";
    if (d.type == "turbulentMixingLengthDissipationRateInlet"
     || d.type == "turbulentMixingLengthFrequencyInlet")
        os << "        mixingLength    " << d.mixingLength << ";\n";
    if (extra) os << *extra;
    if (computed && nComputed)
    {
        // The SOLVED boundary values, not the ones the case was started from. Echoing the input made
        // every written boundaryField stale: a wall nut computed by the wall function was reported as
        // whatever 0/nut happened to say, so anything post-processing brae's output (yPlus, wall shear,
        // force decomposition, a restart) read a value the solve never used.
        // foamListType already IS "List<scalar>"/"List<vector>" -- wrapping it in another List<> emitted
        // "nonuniform List<List<scalar>>", which OF's reader rejects. Every scalar and vector field brae
        // wrote carried it, so no brae output could be restarted from or post-processed by OF; the
        // validation gates never caught it because they regex the internalField and never re-read the file.
        if (fieldWriteFormat().binary)
        {
            os << "        value           nonuniform " << foamListType(T{});
            writeFoamBinaryList(os, computed, nComputed);
            os << ";\n";
        }
        else
        {
            os << "        value           nonuniform " << foamListType(T{}) << " \n"
               << nComputed << "\n(\n";
            writeFoamValues(os, computed, nComputed, '\n');
            os << ")\n;\n";
        }
    }
    else if (d.hasValue)
    {
        os << "        value           ";
        writeFieldValue(os, d.valueUniform, d.uniformValue, d.values);
    }
    os << "    }\n";
}

// Splice the solved internalField into a fully-resolved OpenFOAM field written for `patches`.
// A DERIVED field -- one brae computes but the case has no 0/ file for, e.g. rho. It still needs a
// template for the FoamFile header, but taking the template's identity with it is wrong: rho written from
// 0/T inherited `object T`, `dimensions [0 0 0 1 0 0 0]` (temperature), and T's boundary TYPES and VALUES,
// so the inlet density read back as 300 and a fixedGradient T wall became a density gradient of 20000.
// OF writes a derived field as `calculated` + the computed values on every non-constraint patch; this
// says so explicitly instead of inheriting.
struct DerivedFieldSpec
{
    const char* object = nullptr;       // FoamFile `object` entry, e.g. "rho"
    const char* dimensions = nullptr;   // full entry text, e.g. "dimensions      [1 -3 0 0 0 0 0];"
    const char* patchType = nullptr;    // the non-constraint patches' type; null: `calculated` (rDeltaT's is
                                        // extrapolatedCalculated, createRDeltaT.H)
};

// A vol field's template as the writer takes it: the FoamFile header as written in the file, the dimensions line (empty: none) and
// the resolved boundary. Re-read at every write unless the write path is fast or binary, which keep it while the file's size and modification
// time are the ones it was read at (the multi-GPU driver's templates are the start's fields: 16.6 M values a part on the 66 M DrivAer,
// read and parsed again for every field of every write)
template <typename T>
struct VolFieldTemplate
{
    std::string header;
    std::string dims;
    FieldData<T> fd;
};
template <typename T>
inline VolFieldTemplate<T> readVolFieldTemplate(const std::string& origPath)
{
    // The ORIGINAL field is re-read here as a template: its FoamFile header, dimensions and boundaryField
    // shape are echoed into the new time directory. It may be gzipped -- OF writes `U.gz` whenever
    // writeCompression is on, and pimpleFoam/LES/periodicPlaneChannel ships its 0/ that way -- so this has
    // to go through gzSlurp like every other read, not a bare ifstream. The reader side already did;
    // only the writer still opened the file directly, and the run died at its FIRST write having done all
    // the solving.
    std::string text;
    try
    {
        const std::vector<char> b = gzSlurp(origPath);
        text.assign(b.begin(), b.end());
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("writeVolField: cannot read " + origPath + " (nor " + origPath + ".gz)");
    }
    // FoamFile header block (verbatim but for `location`, rewritten below to the output time as
    // OpenFOAM writes it, and `object` for a derived field) + the dimensions line.
    const std::size_t ff = text.find("FoamFile");
    const std::size_t hb = text.find('{', ff);
    int depth = 0;
    std::size_t he = hb;
    for (; he < text.size(); ++he)
    {
        if (text[he] == '{') ++depth;
        else if (text[he] == '}')
        {
            if (--depth == 0)
            {
                ++he;
                break;
            }
        }
    }
    VolFieldTemplate<T> t;
    t.header = (ff == std::string::npos) ? "" : text.substr(0, he);
    const std::size_t dk = text.find("dimensions", he == 0 ? 0 : he);
    const std::size_t ds = (dk == std::string::npos) ? std::string::npos : text.find(';', dk);
    t.dims = (ds == std::string::npos) ? std::string() : text.substr(dk, ds - dk + 1);
    // Resolved boundary (includes / #includeEtc / $macros / $internalField expanded); groups are matched per patch below.
    t.fd = readField<T>(origPath);
    return t;
}
template <typename T>
inline const VolFieldTemplate<T>& volFieldTemplate(
    const std::string& origPath,
    VolFieldTemplate<T>& scratch)
{
    if (!fieldWriteFormat().fast && !fieldWriteFormat().binary)
    {
        scratch = readVolFieldTemplate<T>(origPath);
        return scratch;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string file = fs::exists(origPath, ec) ? origPath : origPath + ".gz";
    const std::uintmax_t size = fs::file_size(file, ec);
    const fs::file_time_type mtime = fs::last_write_time(file, ec);
    struct Kept
    {
        std::uintmax_t size = 0;
        fs::file_time_type mtime{};
        VolFieldTemplate<T> t;
    };
    // a background writer and the step's thread may both write (one at a time is the driver's rule, the map is guarded anyway)
    static std::mutex guard;
    std::lock_guard<std::mutex> lock(guard);
    static std::map<std::string, Kept> kept;
    auto it = kept.find(origPath);
    if (it == kept.end() || ec || it->second.size != size || it->second.mtime != mtime)
    {
        Kept k;
        k.size = size;
        k.mtime = mtime;
        k.t = readVolFieldTemplate<T>(origPath);
        // the internal values are not written from the template: dropped, the boundary and header kept
        k.t.fd.internalField.clear();
        k.t.fd.internalField.shrink_to_fit();
        it = kept.insert_or_assign(origPath, std::move(k)).first;
    }
    return it->second.t;
}

template <typename T, typename Patch>
inline void writeVolField(
    const std::string& origPath,
    const std::string& outPath,
    const std::vector<T>& values,
    const std::vector<Patch>& patches,
    int precision = 16,
    // Flat boundary values in patch order, EXCLUDING cyclic/cyclicAMI -- the same flattening
    // DeviceBoundary uses, so a solver buffer can be handed over unchanged. Empty -> echo the input.
    const std::vector<T>& computedBoundary = {},
    const DerivedFieldSpec* derived = nullptr,
    // per patch name, entries the solver writes before `value` (rhoCentralFoam's maxwellSlipU and smoluchowskiJumpT:
    // the inputs and the mixed state OpenFOAM's write() emits); null: none, as before
    const std::map<std::string, std::string>* patchExtra = nullptr,
    // the file's `object` where it is not the template's (an old-time level, U_0, written from U's template): written so, rather
    // than the file rewritten after it is written
    const char* objectName = nullptr)
{
    const FieldWriteFormat& wf = fieldWriteFormat();
    FieldWriteStats& stats = fieldWriteStats();
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    VolFieldTemplate<T> scratch;
    const VolFieldTemplate<T>& tmpl = volFieldTemplate<T>(origPath, scratch);
    std::string header = tmpl.header;
    // The template may be a BINARY file (OF writeFormat binary; pimpleFoam/LES/NACA4412 ships 0/U that
    // way). brae writes ASCII unless the write path is binary, so the header it copies has to say what
    // follows it -- otherwise the written field is labelled binary and holds text, and nothing, OpenFOAM
    // included, can read it back.
    header = foamHeaderInFormat(header, wf.binary);
    {
        static const std::regex locRe("location\\s+\"[^\"]*\";");
        header = std::regex_replace(header, locRe, std::string("location    \"") + timeDirName(outPath) + "\";");
    }
    if (derived && derived->object)
    {
        // Rewrite the template's `object <name>;` so the file identifies as what it IS. Some OF readers
        // key on it, and a mismatched one is confusing even where it is tolerated.
        const std::regex objRe("object\\s+[A-Za-z0-9_.:]+\\s*;");
        header = std::regex_replace(header, objRe, std::string("object      ") + derived->object + ";");
    }
    else if (objectName)
    {
        // an old-time level named as the multi-GPU driver's renameFoamObject names it in place: the padding gives up the characters
        // the name adds (`object      U;` -> `object    U_0;`), so the file is the bytes the write-then-rename made; a padding too
        // short for that takes the standard one
        static const std::regex objRe("object([ \\t]+)([A-Za-z0-9_.:]+);");
        std::smatch m;
        if (std::regex_search(header, m, objRe))
        {
            const std::string to = objectName;
            const std::string from = m[2].str();
            const std::size_t pad = static_cast<std::size_t>(m[1].length());
            const std::string entry = (to.size() >= from.size() && pad > to.size() - from.size())
                                    ? "object" + std::string(pad - (to.size() - from.size()), ' ') + to + ";"
                                    : "object      " + to + ";";
            header = header.substr(0, static_cast<std::size_t>(m.position(0))) + entry
                   + header.substr(static_cast<std::size_t>(m.position(0) + m.length(0)));
        }
    }
    const std::string dims = (derived && derived->dimensions)
        ? std::string(derived->dimensions)
        : (tmpl.dims.empty() ? "dimensions      [0 0 0 0 0 0 0];" : tmpl.dims);
    const FieldData<T>& fd = tmpl.fd;
    stats.templateMs += writeMsSince(t0);

    // streaming, as it always was, or (fast or binary) the file built in memory and written in one call
    t0 = std::chrono::steady_clock::now();
    const bool inMemory = wf.fast || wf.binary;
    FoamFileBuffer buffer;
    std::ofstream file;
    std::ostream mem(&buffer);
    if (!inMemory)
    {
        file.open(outPath);
        if (!file) throw std::runtime_error("writeVolField: cannot write " + outPath);
    }
    std::ostream& out = inMemory ? mem : static_cast<std::ostream&>(file);
    out << std::setprecision(precision);
    if (!header.empty()) out << header << "\n\n";
    out << dims << "\n\n";
    if (wf.binary)
    {
        out << "internalField   nonuniform " << foamListType(T{});
        writeFoamBinaryList(out, values.data(), values.size());
        out << ";\n\n";
    }
    else
    {
        out << "internalField   nonuniform " << foamListType(T{}) << "\n" << values.size() << "\n(\n";
        writeFoamValues(out, values.data(), values.size(), '\n');
        out << ")\n;\n\n";
    }
    out << "boundaryField\n{\n";
    std::size_t bndOff = 0;   // index into computedBoundary (coupled patches excluded)
    for (const Patch& p : patches)   // per mesh patch, matched like buildField
    {
        const PatchFieldData<T>* d = nullptr;
        for (const auto& b : fd.boundary)   // pass 1: exact name
            if (b.name == p.name)
            {
                d = &b;
                break;
            }
        if (!d)
            for (const auto& b : fd.boundary)   // pass 2: group / regex
            {
                bool hit = false;
                for (const auto& g : p.inGroups)
                    if (b.name == g)
                    {
                        hit = true;
                        break;
                    }
                if (!hit)
                {
                    try
                    {
                        const std::regex re = compileFoamRegex(b.name);
                        if (std::regex_match(p.name, re)) hit = true;
                        else
                            for (const auto& g : p.inGroups)
                                if (std::regex_match(g, re))
                                {
                                    hit = true;
                                    break;
                                }
                    }
                    catch (const std::regex_error&) {}
                }
                if (hit) d = &b;
            }
        PatchFieldData<T> synth;                                                     // constraint patch (empty/cyclic/...) or safety
        if (!d || derived)
        {
            synth.name = p.name;
            // A derived field takes NO boundary type from the template -- constraint patches keep their own
            // (OF writes `empty` as `empty`), everything else is `calculated`, exactly as OF writes rho.
            synth.type = isConstraintPatchType(p.type)
                       ? p.type
                       : (derived ? (derived->patchType ? derived->patchType : "calculated") : "zeroGradient");
            d = &synth;
        }
        const bool coupled = (isCoupledInterfaceType(p.type));
        const bool haveComputed = !computedBoundary.empty() && !coupled && p.size > 0
                               && bndOff + static_cast<std::size_t>(p.size) <= computedBoundary.size()
                               && p.type != "empty";
        const std::string* extra = nullptr;
        if (patchExtra)
        {
            const auto it = patchExtra->find(p.name);
            if (it != patchExtra->end()) extra = &it->second;
        }
        writePatchEntry(out, p.name, *d,
                        haveComputed ? &computedBoundary[bndOff] : nullptr,
                        haveComputed ? static_cast<std::size_t>(p.size) : 0, extra);
        if (!coupled) bndOff += static_cast<std::size_t>(p.size);
    }
    out << "}\n\n\n// ************************************************************************* //\n";
    finishFoamFile(outPath, inMemory, buffer, file, t0);
}

// Write a surfaceScalarField (the face flux phi) in OpenFOAM structure, matching OF's own output: internalField = the
// internal-face flux, boundaryField per patch (`type calculated` + the face-flux value list, except `empty` patches
// which write only the type). phiBoundary is the FLAT boundary flux in patch order EXCLUDING cyclic/cyclicAMI patches
// (their flux lives on the interface, not in phiBnd) -- those coupled patches write just their constraint type (v1: no
// value). Loads in any OpenFOAM reader (paraFoam/postProcess) as the case's phi.
template <typename Patch>
inline void writeSurfaceField(
    const std::string& outPath,
    const std::vector<scalar>& phiInternal,   // nInternalFaces
    const std::vector<scalar>& phiBoundary,   // flat, patch order, EXCLUDING cyclic/cyclicAMI
    const std::vector<Patch>& patches,
    int precision = 16,
    // phi's dimensions differ by solver and OF writes the real ones: the incompressible solvers carry a
    // VOLUMETRIC flux [0 3 -1 0 0 0 0] (m3/s), the compressible ones a MASS flux [1 0 -1 0 0 0 0] (kg/s).
    // Defaulted to the volumetric form so the incompressible callers are unchanged. Getting this wrong
    // still loads, but every downstream tool that checks dimensions (postProcess, funkySetFields, a
    // restart into OF) then sees a field that claims to be something it is not.
    const std::string& dimensions = "[0 3 -1 0 0 0 0]",
    // Per-COUPLED-patch flux, keyed by patch name. cyclic/cyclicAMI/cyclicACMI faces are not in
    // phiBoundary at all (their flux lives on the interface object), so without this they were written
    // as a bare `type <patchType>;` with no value -- and a restart had nothing to resume from. OF writes
    // the values; see DeviceSimpleSolver::interfacePatchFlux for what the missing round-trip cost.
    // Absent/empty keeps the old value-less form, which is still right for a solver with no interface.
    const std::map<std::string, std::vector<scalar>>& coupledValues = {})
{
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    const bool inMemory = fieldWriteFormat().fast || fieldWriteFormat().binary;
    FoamFileBuffer buffer;
    std::ofstream file;
    std::ostream mem(&buffer);
    if (!inMemory)
    {
        file.open(outPath);
        if (!file) throw std::runtime_error("writeSurfaceField: cannot write " + outPath);
    }
    std::ostream& out = inMemory ? mem : static_cast<std::ostream&>(file);
    out << std::setprecision(precision);
    out << foamHeaderInFormat("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       surfaceScalarField;\n"
                              "    location    \"" + timeDirName(outPath) + "\";\n    object      phi;\n}", fieldWriteFormat().binary)
        << "\n\n";
    out << "dimensions      " << dimensions << ";\n\n";
    out << "internalField   nonuniform List<scalar>";
    writeFoamListEntry(out, phiInternal.data(), phiInternal.size());
    out << "\n";
    out << "boundaryField\n{\n";
    std::size_t off = 0;
    for (const auto& p : patches)
    {
        out << "    " << p.name << "\n    {\n";
        if (isCoupledInterfaceType(p.type))   // coupled: flux held on the interface, not in phiBoundary
        {
            out << "        type            " << p.type << ";\n";
            const auto cv = coupledValues.find(p.name);
            if (cv != coupledValues.end() && cv->second.size() == static_cast<std::size_t>(p.size))
            {
                out << "        value           nonuniform List<scalar>";
                writeFoamListEntry(out, cv->second.data(), cv->second.size());
            }
            out << "    }\n";
            continue;                                       // do NOT advance off (phiBoundary skips these)
        }
        const std::size_t n = static_cast<std::size_t>(p.size);
        if (p.type == "empty")
            out << "        type            empty;\n";
        else
        {
            // A constraint patch (symmetry, symmetryPlane, wedge) keeps its type, with the flux as its value, as OpenFOAM
            // writes it: fvsPatchField::New stops on any other ("inconsistent patch and patchField types for patch type
            // symmetry and patchField type calculated", fvsPatchFieldNew.C:127). This wrote `calculated` on them, so
            // OpenFOAM could not restart from brae's simpleFoam, pimpleFoam or rhoSimpleFoam output on such a mesh
            // (validation/windAroundBuildingsBox's frontAndBack, 2026-11-19).
            out << "        type            " << (isConstraintPatchType(p.type) ? p.type : std::string("calculated"))
                << ";\n        value           nonuniform List<scalar>";
            writeFoamListEntry(out, phiBoundary.data() + off, n);
        }
        off += n;
        out << "    }\n";
    }
    out << "}\n\n\n// ************************************************************************* //\n";
    finishFoamFile(outPath, inMemory, buffer, file, t0);
}

// Write a vol field with NO template (the CrankNicolson ddt0 fields, which have no 0/ prototype): a constructed OF header
// (class from T, the given object name + dimensions), the internalField, and a minimal boundaryField (constraint patches
// keep their type; every other patch is `calculated` with a zero uniform value). Loads in any OF reader; on a brae
// restart only the internalField is consumed (ddt0's boundary is unused by the pointwise recurrence).
template <typename T, typename Patch>
inline void writeVolFieldRaw(
    const std::string& outPath,
    const std::string& objectName,
    const std::string& dims,
    const std::vector<T>& values,
    const std::vector<Patch>& patches,
    int precision = 16,
    // Flat boundary values in patch order, EXCLUDING cyclic/cyclicAMI -- the same flattening
    // DeviceBoundary uses, so a solver buffer can be handed over unchanged. Empty -> echo the input.
    const std::vector<T>& computedBoundary = {})
{
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    const bool inMemory = fieldWriteFormat().fast || fieldWriteFormat().binary;
    FoamFileBuffer buffer;
    std::ofstream file;
    std::ostream mem(&buffer);
    if (!inMemory)
    {
        file.open(outPath);
        if (!file) throw std::runtime_error("writeVolFieldRaw: cannot write " + outPath);
    }
    std::ostream& out = inMemory ? mem : static_cast<std::ostream&>(file);
    out << std::setprecision(precision);
    out << foamHeaderInFormat(std::string("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       ") + foamClassName(T{})
                              + ";\n    object      " + objectName + ";\n}", fieldWriteFormat().binary)
        << "\n\n";
    out << "dimensions      " << dims << ";\n\n";
    if (fieldWriteFormat().binary)
    {
        out << "internalField   nonuniform " << foamListType(T{});
        writeFoamBinaryList(out, values.data(), values.size());
        out << ";\n\n";
    }
    else
    {
        out << "internalField   nonuniform " << foamListType(T{}) << "\n" << values.size() << "\n(\n";
        writeFoamValues(out, values.data(), values.size(), '\n');
        out << ")\n;\n\n";
    }
    out << "boundaryField\n{\n";
    for (const Patch& p : patches)
    {
        out << "    " << p.name << "\n    {\n";
        if (isConstraintPatchType(p.type))
            out << "        type            " << p.type << ";\n";
        else
        {
            out << "        type            calculated;\n        value           uniform ";
            formatFoamValue(out, T{});
            out << ";\n";
        }
        out << "    }\n";
    }
    out << "}\n\n\n// ************************************************************************* //\n";
    finishFoamFile(outPath, inMemory, buffer, file, t0);
}

} // namespace brae
