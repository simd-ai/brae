#pragma once
// `brae decompose` and `brae reconstruct`: decomposePar and reconstructPar without OpenFOAM, for the cases `brae -parallel` runs.
// The work is src/OpenFOAM/decompose/case_decompose.cuh; this is its command line.
#include "case_decompose.cuh"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cli {

inline std::vector<std::string> splitTimes(const std::string& s)
{
    std::vector<std::string> out;
    std::string cur;
    for (const char c : s)
    {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// the message with its command's name in front, unless it already has "brae ..." there
inline std::string withCommand(const char* cmd, const char* what)
{
    const std::string w = what;
    return w.compare(0, 5, "brae ") == 0 ? w : std::string("brae ") + cmd + ": " + w;
}

inline int decomposeUsage(const char* what, int code)
{
    std::FILE* f = code ? stderr : stdout;
    if (std::string(what) == "decompose")
        std::fprintf(f,
            "usage: brae decompose [-case <dir>] [-n N] [-method scotch|simple|hierarchical|manual] [-time T[,T..] | -allTimes]\n"
            "                      [-force] [-quiet]\n"
            "  Splits the case into processor<N>/ directories, as OpenFOAM's decomposePar does, for `brae -parallel`. -n and -method\n"
            "  default to system/decomposeParDict's numberOfSubdomains and method (scotch without one). The fields of the time the\n"
            "  run would start from are decomposed; -time and -allTimes name others. The solver does this itself when `-parallel`\n"
            "  finds no processor directories.\n");
    else
        std::fprintf(f,
            "usage: brae reconstruct [-case <dir>] [-time T[,T..] | -latestTime] [-force] [-quiet]\n"
            "  Puts the processor<N>/ time directories back into the case, as OpenFOAM's reconstructPar does. Without -time, every\n"
            "  time processor0 has that the case does not.\n");
    return code;
}

// argv[1] is the command word
inline int decomposeCommand(int argc, char** argv)
{
    try
    {
        decompose::Options opt;
        std::string caseDir = ".";
        for (int i = 2; i < argc; ++i)
        {
            const std::string a = argv[i];
            auto value = [&]() -> std::string
            {
                if (i + 1 >= argc) throw std::runtime_error("`" + a + "` needs a value");
                return argv[++i];
            };
            if (a == "-case") caseDir = value();
            else if (a == "-n" || a == "-np" || a == "-nprocs") opt.nProcs = std::atoi(value().c_str());
            else if (a == "-method") opt.method = value();
            else if (a == "-time") opt.times = splitTimes(value());
            else if (a == "-allTimes") opt.allTimes = true;
            else if (a == "-force") opt.force = true;
            else if (a == "-quiet") opt.quiet = true;
            else if (a == "-h" || a == "-help" || a == "--help") return decomposeUsage("decompose", 0);
            else if (!a.empty() && a[0] != '-') caseDir = a;
            else { std::fprintf(stderr, "brae decompose: unknown option `%s`\n", a.c_str()); return decomposeUsage("decompose", 2); }
        }
        const decompose::Report r = decompose::decomposeCase(caseDir, opt);
        std::printf("brae decompose: %d processors (%s), %zu field file(s) over %zu time(s)\n", r.nProcs, r.method.c_str(), r.fields.size(),
                    r.times.size());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", withCommand("decompose", e.what()).c_str());
        return 2;
    }
}

inline int reconstructCommand(int argc, char** argv)
{
    try
    {
        decompose::ReconstructOptions opt;
        std::string caseDir = ".";
        for (int i = 2; i < argc; ++i)
        {
            const std::string a = argv[i];
            auto value = [&]() -> std::string
            {
                if (i + 1 >= argc) throw std::runtime_error("`" + a + "` needs a value");
                return argv[++i];
            };
            if (a == "-case") caseDir = value();
            else if (a == "-time") opt.times = splitTimes(value());
            else if (a == "-latestTime") opt.latestTime = true;
            else if (a == "-force") opt.force = true;
            else if (a == "-quiet") opt.quiet = true;
            else if (a == "-h" || a == "-help" || a == "--help") return decomposeUsage("reconstruct", 0);
            else if (!a.empty() && a[0] != '-') caseDir = a;
            else { std::fprintf(stderr, "brae reconstruct: unknown option `%s`\n", a.c_str()); return decomposeUsage("reconstruct", 2); }
        }
        const decompose::ReconstructReport r = decompose::reconstructCase(caseDir, opt);
        std::printf("brae reconstruct: %zu field file(s) over %zu time(s)\n", r.fields.size(), r.times.size());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", withCommand("reconstruct", e.what()).c_str());
        return 2;
    }
}

}   // namespace cli
}   // namespace brae
