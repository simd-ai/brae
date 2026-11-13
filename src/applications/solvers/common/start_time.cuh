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
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

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

// argList::setCasePaths (argList.C:866-895): the case directory, absolute and cleaned, into FOAM_CASE and its name
// into FOAM_CASENAME, overwriting whatever the shell had, before anything is read. A case's dictionaries name files
// through it: occDrivAerStaticMesh's turbulenceProperties, transportProperties and every 0/ field say `#include
// "${FOAM_CASE}/system/include/caseDefinition"`. brae never set it, so the path expanded to /system/include/..., the
// include was skipped with a warning, and $RASturbModel, $nu and every #eval of them were left undefined: `brae
// check` refused a case OpenFOAM runs.
inline void setCaseEnvironment(const std::string& caseDir)
{
    std::error_code ec;
    std::string abs = std::filesystem::absolute(caseDir, ec).lexically_normal().string();
    while (abs.size() > 1 && abs.back() == '/')
    {
        abs.pop_back();
    }
    const std::size_t slash = abs.find_last_of('/');
    const std::string name = (slash == std::string::npos) ? abs : abs.substr(slash + 1);
    setenv("FOAM_CASE", abs.c_str(), 1);
    setenv("FOAM_CASENAME", name.c_str(), 1);
}

// OpenFOAM's Time::timeName(t, precision) (Time.C:721-728): an ostringstream set to the controlDict's
// timeFormat -- `general` sets no floatfield flag (%g), `fixed` and `scientific` set theirs (IOstreamOption.C:42-47)
// -- at `precision` digits. `0.`, `0` and `0.0` all name the directory "0"; `1e-3` names "0.001".
enum class TimeFormat
{
    general,
    fixed,
    scientific
};

inline std::string openFoamTimeName(double t, TimeFormat format, int precision)
{
    std::ostringstream buf;
    if (format == TimeFormat::fixed)
    {
        buf.setf(std::ios_base::fixed, std::ios_base::floatfield);
    }
    else if (format == TimeFormat::scientific)
    {
        buf.setf(std::ios_base::scientific, std::ios_base::floatfield);
    }
    buf.precision(precision);
    buf << t;
    return buf.str();
}

// Time::format_ and Time::precision_ are STATIC in OpenFOAM (Time.C:80-81): one formatting for every time name a run
// reads or writes. controlDict's timeFormat and timePrecision set it (TimeIO.C:336-338); setControls raises the
// precision when the start directory is formatted with more digits (Time.C:236-243), and every later name keeps it.
struct TimeFormatting
{
    TimeFormat format          = TimeFormat::general;
    int        precision       = 6;
    int        raisedPrecision = -1;   // what setControls raised it to, which a later re-read of controlDict keeps
};

inline TimeFormatting& timeFormatting()
{
    static TimeFormatting f;
    return f;
}

// controlDict's timeFormat and timePrecision, into the run's one formatting. An unknown format name is warned about
// and general kept (IOstreamOption.C:62-96).
template <class Dict>
inline void readTimeFormatting(const Dict& controlDict)
{
    TimeFormatting& f = timeFormatting();
    const std::string fmt = controlDict.wordOr("timeFormat", "general");
    if (fmt == "general")
    {
        f.format = TimeFormat::general;
    }
    else if (fmt == "fixed")
    {
        f.format = TimeFormat::fixed;
    }
    else if (fmt == "scientific")
    {
        f.format = TimeFormat::scientific;
    }
    else
    {
        std::fprintf(stderr, "brae: controlDict timeFormat '%s' is not general, fixed or scientific; OpenFOAM warns "
                     "and keeps general, and so does brae\n", fmt.c_str());
    }
    f.precision = controlDict.intOr("timePrecision", 6);
    if (f.raisedPrecision > f.precision)
    {
        f.precision = f.raisedPrecision;
    }
}

// Time::timeName(t): the run's formatting
inline std::string openFoamTimeName(double t)
{
    return openFoamTimeName(t, timeFormatting().format, timeFormatting().precision);
}

// What Time::setControls (Time.C:146-250) settles on: the start time's VALUE, which every time value of the run is
// measured from, and the DIRECTORY its fields are read from, which is that value formatted by timeName() -- never
// the controlDict's own spelling of it.
struct StartTime
{
    double      value     = 0.0;
    std::string name;
    TimeFormat  format    = TimeFormat::general;
    int         precision = 6;
};

// Port of Time::setControls' start resolution, for any dictionary with found/wordOr/scalarOr/intOr.
//
// THE DIRECTORY WAS THE STARTTIME TEXT. This returned controlDict's `startTime` word as written, so `startTime 0.;`
// (as ANSA writes it) looked for a directory named "0." where OpenFOAM reads "0", and `startTime 1e-3;` looked for
// "1e-3" where OpenFOAM reads "0.001". OpenFOAM reads the entry as a scalar (Time.C:155-158) and names the directory
// timeName(value) at the case's timeFormat and timePrecision (TimeIO.C:336-338); when no such directory exists it
// tries every precision from maxPrecision (3 - log10(SMALL) = 18, Time.C:83) down, and adopts the lowest that finds
// one (Time.C:209-250).
//
// `startFrom` defaults to latestTime, OpenFOAM's (Time.C:149-153); the drivers passed startTime as the default. A word
// other than startTime/firstTime/latestTime, and `startFrom startTime` with no startTime, stop OpenFOAM (Time.C:155-
// 158, 181-186) and are refused here. A latestTime/firstTime search that finds nothing leaves the start at 0, the
// value Time's constructors give startTime_ (Time.C:454), not at controlDict's startTime.
//
// `probe` is brae's one addition: a field that must exist for a directory to count as a restart point -- "U" for a
// flow solver -- so mesh-only directories written by snappyHexMesh are not mistaken for one. OpenFOAM's findTimes
// takes any directory whose name reads as a scalar, and then fails to read the field there.
template <class Dict>
inline StartTime resolveStartTime(const std::string& caseDir, const Dict& controlDict, const char* probe = "U")
{
    namespace fs = std::filesystem;
    StartTime s;
    const std::string startFrom = controlDict.wordOr("startFrom", "latestTime");
    if (startFrom == "startTime")
    {
        if (!controlDict.found("startTime"))
        {
            throw std::runtime_error("system/controlDict sets `startFrom startTime` and no `startTime`; OpenFOAM reads it "
                                     "with no default (Time.C:155-158) and stops.");
        }
        s.value = static_cast<double>(controlDict.scalarOr("startTime", 0.0));
    }
    else if (startFrom == "firstTime" || startFrom == "latestTime")
    {
        std::error_code ec;
        bool any = false;
        for (const auto& e : fs::directory_iterator(caseDir, ec))
        {
            if (!e.is_directory())
            {
                continue;
            }
            const std::string nm = e.path().filename().string();
            char* end = nullptr;
            const double t = std::strtod(nm.c_str(), &end);
            if (end == nm.c_str() || *end != '\0')
            {
                continue;   // not a pure number: constant, system, 0.orig
            }
            if (!(fs::exists(e.path() / probe) || fs::exists(e.path() / (std::string(probe) + ".gz"))))
            {
                continue;
            }
            if (!any || (startFrom == "latestTime" ? t > s.value : t < s.value))
            {
                s.value = t;
            }
            any = true;
        }
    }
    else
    {
        throw std::runtime_error("system/controlDict sets `startFrom " + startFrom + "`; OpenFOAM expects startTime, "
                                 "firstTime or latestTime (Time.C:181-186) and stops.");
    }

    readTimeFormatting(controlDict);
    s.format = timeFormatting().format;
    s.precision = timeFormatting().precision;
    s.name = openFoamTimeName(s.value, s.format, s.precision);

    // Time.C:209-250: no directory at this formatting -> the lowest higher precision that names one
    if (!fs::is_directory(caseDir + "/" + s.name))
    {
        const int maxPrecision = 18;
        int required = -1;
        std::string oldTime = s.name;
        for (int prec = maxPrecision; prec > s.precision; --prec)
        {
            const std::string newTime = openFoamTimeName(s.value, s.format, prec);
            if (newTime == oldTime)
            {
                break;
            }
            oldTime = newTime;
            if (fs::is_directory(caseDir + "/" + newTime))
            {
                required = prec;
            }
        }
        if (required > 0)
        {
            std::fprintf(stderr, "brae: increasing the timePrecision from %d to %d to support the formatting of the "
                         "start directory, as OpenFOAM does (Time.C:236-243)\n", s.precision, required);
            s.precision = required;
            timeFormatting().precision = required;
            timeFormatting().raisedPrecision = required;
            s.name = openFoamTimeName(s.value, s.format, s.precision);
        }
    }
    if (startFrom != "startTime" && s.value != static_cast<double>(controlDict.scalarOr("startTime", 0.0)))
    {
        std::fprintf(stderr, "brae: startFrom %s -> starting from time '%s'\n", startFrom.c_str(), s.name.c_str());
    }
    return s;
}

}  // namespace brae
