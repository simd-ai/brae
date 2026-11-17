#pragma once
// Shared coded-BC (codedFixedValue / codedMixed) driver-side setup for the steady (gpuSimpleFoam) and transient
// (gpuPimpleFoam) solvers. Both parse a case's field files for coded patches, map each to its flat-boundary face range,
// and register an NVRTC device kernel on the shared DeviceSimpleSolver. The solver machinery (addCodedBC + the
// momentum-predictor apply) lives in device_simple_foam; this header is only the parse + registration glue.

#include "fv_patch.cuh"                // FvPatch
#include "device_simple_foam.cuh"      // DeviceSimpleSolver + DeviceSimpleControls

#include <cctype>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// A coded (codedFixedValue / codedMixed) patch parsed from a field file: the patch name, the `code #{ ... #}` body, the
// kernel `name`, and whether it is a mixed (Robin) BC (the snippet also sets valueFraction `vf`).
struct CodedBCSpec { std::string patch, code, name; bool mixed = false; };

// Scan a field file's boundaryField for codedFixedValue / codedMixed patches, extracting each patch's code snippet +
// name. Handles the OF verbatim `#{ ... #}` block (its braces do not count toward the patch's brace depth). Empty if none.
inline std::vector<CodedBCSpec> parseCodedBCs(const std::string& fieldPath)
{
    std::vector<CodedBCSpec> out;
    std::ifstream f(fieldPath);
    if (!f) return out;
    const std::string t((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::size_t bf = t.find("boundaryField");
    if (bf == std::string::npos) return out;
    std::size_t p = t.find('{', bf);
    if (p == std::string::npos) return out;
    ++p;
    auto isws = [](char c) { return std::isspace((unsigned char)c) != 0; };
    while (p < t.size())
    {
        while (p < t.size() && isws(t[p])) ++p;
        if (p >= t.size() || t[p] == '}') break;                 // end of boundaryField
        const std::size_t ns = p;
        while (p < t.size() && !isws(t[p]) && t[p] != '{') ++p;
        std::string pname = t.substr(ns, p - ns);
        // A PREPROCESSOR DIRECTIVE is not a patch. OF tutorials routinely open boundaryField with
        //     #includeEtc "caseDicts/setConstraintTypes"
        // (pipeCyclic, and many others). Reading that as a patch name is not merely a wrong name: the
        // scan then runs forward to the NEXT '{', which belongs to the first real patch, so that
        // patch's body is consumed as the directive's and the patch itself disappears. pipeCyclic
        // failed with "coded BC: patch '#includeEtc' not in the mesh boundary" -- the visible half of
        // that. Directives are single-line, so skip to end of line.
        if (!pname.empty() && pname.front() == '#')
        {
            while (p < t.size() && t[p] != '\n') ++p;
            continue;
        }
        if (!pname.empty() && pname.front() == '"' && pname.back() == '"') pname = pname.substr(1, pname.size() - 2);
        while (p < t.size() && t[p] != '{') ++p;
        if (p >= t.size()) break;
        const std::size_t bodyStart = ++p;
        int depth = 1; std::size_t q = p;
        while (q < t.size() && depth > 0)                        // find the matching '}', skipping #{ ... #} verbatim
        {
            if (t.compare(q, 2, "#{") == 0) { const std::size_t e = t.find("#}", q + 2); q = (e == std::string::npos ? t.size() : e + 2); continue; }
            if (t[q] == '{') ++depth; else if (t[q] == '}') --depth;
            ++q;
        }
        const std::string body = t.substr(bodyStart, (q - 1) - bodyStart);
        p = q;
        const bool isMixed = body.find("codedMixed") != std::string::npos;
        if (isMixed || body.find("codedFixedValue") != std::string::npos)
        {
            CodedBCSpec s; s.patch = pname; s.mixed = isMixed;
            const std::size_t cs = body.find("#{");
            if (cs != std::string::npos) { const std::size_t ce = body.find("#}", cs + 2); if (ce != std::string::npos) s.code = body.substr(cs + 2, ce - (cs + 2)); }
            std::size_t nm = body.find("name");
            if (nm != std::string::npos) { nm += 4; while (nm < body.size() && isws(body[nm])) ++nm; std::size_t ne = nm; while (ne < body.size() && !isws(body[ne]) && body[ne] != ';') ++ne; s.name = body.substr(nm, ne - nm); }
            if (s.name.empty()) s.name = "coded_" + pname;
            out.push_back(std::move(s));
        }
    }
    return out;
}

// Register every coded (codedFixedValue / codedMixed) patch found in U / p / the turbulence scalar(s) as an NVRTC device
// kernel on `solver` (compiled once here; applied each step in the momentum predictor). Each patch's flat-boundary face
// range (offset,count) uses the same patch order (cyclic/cyclicAMI excluded) as the device boundary. `logPrefix` names the
// solver in the banner. Shared by gpuSimpleFoam (steady; time stays 0 -> position-based coded BCs) and gpuPimpleFoam.
inline void setupCodedBCs(DeviceSimpleSolver& solver,
                          const std::string& fieldDir,
                          const std::vector<FvPatch>& fvp,
                          const DeviceSimpleControls& ctl,
                          const std::string& secondName,
                          const char* logPrefix)
{
    auto registerCoded = [&](const std::string& field, int target)
    {
        for (const CodedBCSpec& s : parseCodedBCs(fieldDir + "/" + field))
        {
            int off = 0, cnt = -1;
            for (const FvPatch& q : fvp)
            {
                if (isCyclicFamilyType(q.type)) continue;
                if (q.name == s.patch) { cnt = (int)q.size; break; }
                off += (int)q.size;
            }
            if (cnt < 0) throw std::runtime_error("coded BC: patch '" + s.patch + "' not in the mesh boundary");
            solver.addCodedBC(s.name, s.code, off, cnt, target, s.mixed);
            std::printf("%s: %s '%s' on '%s' (field %s) -> NVRTC device kernel (%d faces)\n",
                        logPrefix, s.mixed ? "codedMixed" : "codedFixedValue", s.name.c_str(), s.patch.c_str(), field.c_str(), cnt);
        }
    };
    registerCoded("U", 0);                        // U (vector)
    registerCoded("p", 1);                        // p (scalar)
    if (ctl.turbulent && !ctl.les)                // turbulence scalars (none for pure-LES Smagorinsky)
    {
        registerCoded(ctl.sa ? "nuTilda" : "k", 2);
        if (!ctl.sa) registerCoded(secondName, 3);
    }
}

}  // namespace brae
