#pragma once
// Instrument: BRAE_SST_DUMP_DIR=<dir> (+ BRAE_SST_DUMP_ITER=n, default 1) writes the kOmegaSST closure's
// intermediates at ONE call as plain columns, under <dir>/<arm>/<name>, with the SAME names and layout
// from both mirror arms -- so the device can be held against the host reference stage by stage from the
// same in-memory trajectory, and the FIRST stage that differs names the term.
//
// WHY THIS EXISTS. The kEpsilon closure already had this, through the solver's StageDump: the host
// writes G, gByNu, epsD/epsSrc, kD/kSrc and their pre-relax twins (rhoSimpleFoam_cpp.cu) and the device
// writes the same names (rhoSimpleFoam.cu). kOmegaSST had NONE of it -- the solver's dump only reaches
// the kEpsilon branch -- so a device-only disagreement in this closure had no oracle short of the final
// field, which is how the iteration-1 gap on aerofoilNACA0012 stayed unlocalised.
//
// It writes nothing, and costs one getenv per closure call, unless the variable is set.
#include "cf_types.cuh"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace brae {
namespace turbulence {

struct SstStageDump
{
    std::string dir;
    bool        on = false;

    void scalars(const char* name, const std::vector<scalar>& v) const
    {
        if (!on) return;
        std::FILE* fp = std::fopen((dir + "/" + name).c_str(), "w");
        if (!fp) return;
        for (const scalar x : v) std::fprintf(fp, "%.17g\n", (double)x);
        std::fclose(fp);
    }

    // A tensor per cell as nine columns, a vector as three -- the layout the solver's StageDump uses,
    // so the same reader serves both.
    void components(const char* name, const std::vector<scalar>& flat, int ncomp) const
    {
        if (!on) return;
        std::FILE* fp = std::fopen((dir + "/" + name).c_str(), "w");
        if (!fp) return;
        for (std::size_t i = 0; i + ncomp <= flat.size(); i += ncomp)
        {
            for (int c = 0; c < ncomp; ++c) std::fprintf(fp, "%s%.17g", c ? " " : "", (double)flat[i + c]);
            std::fprintf(fp, "\n");
        }
        std::fclose(fp);
    }
};

// arm is "host" or "cuda". The counter is per arm and per process, so `<n>` means the n-th call of THIS
// closure -- iteration n of a run that enters it once per outer iteration.
inline SstStageDump sstStageDump(const char* arm)
{
    SstStageDump sd;
    const char* dd = std::getenv("BRAE_SST_DUMP_DIR");
    if (!dd || !*dd) return sd;
    const char* it = std::getenv("BRAE_SST_DUMP_ITER");
    const int want = (it && *it) ? std::atoi(it) : 1;
    static int calls = 0;
    if (++calls != want) return sd;
    sd.dir = std::string(dd) + "/" + arm;
    std::error_code ec;
    std::filesystem::create_directories(sd.dir, ec);
    sd.on = !ec;
    return sd;
}

}   // namespace turbulence
}   // namespace brae
