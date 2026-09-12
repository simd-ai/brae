#pragma once
// OF `startFrom` resolution, shared by the incompressible and compressible drivers.
//
// C6: gpuRhoSimpleFoam hardcoded `caseDir + "/0"` and never looked at startFrom at all, so a case with
// `startFrom latestTime` -- the standard way to CONTINUE a compressible run -- silently restarted from
// scratch. The run then converged perfectly to the right steady answer while having thrown away the
// restart, which is only harmless because the case was steady; the same input on a transient driver loses
// the history outright. The incompressible driver already resolved this; the logic lived inline there, so
// the compressible one could not share it. Extracted verbatim rather than reimplemented.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace brae {

// The number of SIMPLE/PIMPLE steps a run takes, stepping OpenFOAM's own inequality rather than rounding
// a quotient. Time::run() tests `value() < endTime_ - 0.5*deltaT_` (Time.C:785) and Time::operator++
// ACCUMULATES the value by deltaT (Time.C:1067), so N is the count of integers k >= 0 with
// startTime + k*deltaT < endTime - 0.5*deltaT.
//
// Every driver here derived it as `std::lround((endTime - startTime)/deltaT)`, which disagrees with
// OpenFOAM whenever that ratio lands on n + 0.5: measured at startTime 0 / endTime 1 / deltaT 0.4 (ratio
// exactly 2.5), real OpenFOAM runs 2 steps and lround gives 3. Accumulating rather than multiplying also
// reproduces OpenFOAM's own rounding drift on a ratio that is not exact in binary.
//
// No fixture in validation/ can tell the two apart -- all 159 have an integer (endTime - startTime)/deltaT
// -- which is exactly why seven drivers carried the same rounding for as long as they did.
inline long openFoamNSteps(double startTime, double endTime, double deltaT)
{
    if (deltaT <= 0.0) return 0;
    long n = 0;
    for (double t = startTime; t < endTime - 0.5 * deltaT; t += deltaT)
    {
        ++n;
    }
    return n;
}

// Resolve OF's startFrom against the case's numeric time directories. `startStr` is the controlDict
// startTime (the 'startTime' default, and the fallback when nothing matches). `probe` is a field that
// must exist for a directory to count as a field directory -- "U" for a flow solver -- so mesh-only
// directories written by snappyHexMesh are not mistaken for a restart point.
inline std::string resolveStartTime(
    const std::string& caseDir,
    const std::string& startFrom,
    const std::string& startStr,
    const char* probe = "U")
{
    if (startFrom != "latestTime" && startFrom != "firstTime") return startStr;

    namespace fs = std::filesystem;
    std::error_code ec;
    double best = 0.0;
    std::string bestName;
    for (const auto& e : fs::directory_iterator(caseDir, ec))
    {
        if (!e.is_directory()) continue;
        const std::string nm = e.path().filename().string();
        char* end = nullptr;
        const double t = std::strtod(nm.c_str(), &end);
        if (end == nm.c_str() || *end != '\0') continue;   // not a pure number -> skip constant/system/*.orig
        if (!(fs::exists(e.path() / probe) || fs::exists(e.path() / (std::string(probe) + ".gz")))) continue;
        if (bestName.empty() || (startFrom == "latestTime" ? t > best : t < best)) { best = t; bestName = nm; }
    }
    if (bestName.empty() || bestName == startStr) return startStr;

    std::fprintf(stderr, "brae: startFrom %s -> starting from time '%s'\n", startFrom.c_str(), bestName.c_str());
    return bestName;
}

}  // namespace brae
