#pragma once
// time_dict_writer.cuh -- what OpenFOAM writes under <time>/uniform at every write time, so a restart from brae's own
// output -- by brae or by OpenFOAM -- resumes the time state OpenFOAM's would:
//
//   uniform/time      Time::writeTimeDict (TimeIO.C:508-539), written first, before the fields:
//                       value    timeName(value(), maxPrecision_): the time's VALUE in the run's timeFormat at 18
//                                digits (maxPrecision_ = 3 - log10(SMALL), Time.C:83)
//                       name     the time directory's name, a quoted string
//                       index    the time index (the number of steps since 0, carried across restarts)
//                       deltaT   this step's deltaT    } scalars at the stream's precision, which controlDict's
//                       deltaT0  the step before's     } writePrecision sets: 6 digits in a case as shipped
//   uniform/functionObjects/functionObjectProperties
//                     functionObjectList's state dictionary, written at every write time even with no function
//                     object (then empty)
//
// What a restart reads back is Time::setControls' (Time.C:276-307, and RunClock): deltaT only under adjustTimeStep,
// deltaT0 and index always. The time value itself comes from the directory's NAME, not from `value`.
#include "foam_dict.cuh"    // readDict: uniform/time read back on a restart
#include "start_time.cuh"   // openFoamTimeName, timeFormatting

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brae {

namespace detail {

inline const char* foamBanner()
{
    return "/*--------------------------------*- C++ -*----------------------------------*\\\n"
           "| =========                 |                                                 |\n"
           "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
           "|  \\\\    /   O peration     | Version:  v2412                                 |\n"
           "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
           "|    \\\\/     M anipulation  |                                                 |\n"
           "\\*---------------------------------------------------------------------------*/\n";
}

inline void writeDictFile(const std::string& path, const std::string& location, const std::string& object,
                          const std::string& body)
{
    std::ofstream os(path);
    if (!os) throw std::runtime_error("brae: cannot write " + path);
    os << foamBanner()
       << "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    arch        \"LSB;label=32;scalar=64\";\n"
       << "    class       dictionary;\n    location    \"" << location << "\";\n    object      " << object
       << ";\n}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
       << body
       << "\n\n// ************************************************************************* //\n";
}

// a scalar as OpenFOAM's Ostream writes it: general format at the stream's precision
inline std::string foamScalar(double v, int precision)
{
    std::ostringstream s;
    s.precision(precision);
    s << v;
    return s.str();
}

inline std::string entry(const std::string& key, const std::string& value)
{
    std::string k = key;
    if (k.size() < 15) k.resize(15, ' ');
    return k + " " + value + ";\n\n";
}

} // namespace detail

// <caseDir>/<timeName>/uniform/time and uniform/functionObjects/functionObjectProperties, as OpenFOAM writes them
inline void writeUniformTime(const std::string& caseDir, const std::string& timeName, double value, long index,
                             double deltaT, double deltaT0, int writePrecision)
{
    namespace fs = std::filesystem;
    const std::string u = caseDir + "/" + timeName + "/uniform";
    fs::create_directories(u + "/functionObjects");
    const TimeFormatting& f = timeFormatting();
    std::string body;
    body += detail::entry("value", openFoamTimeName(value, f.format, 18));
    body += detail::entry("name", "\"" + timeName + "\"");
    body += detail::entry("index", std::to_string(index));
    body += detail::entry("deltaT", detail::foamScalar(deltaT, writePrecision));
    body += detail::entry("deltaT0", detail::foamScalar(deltaT0, writePrecision));
    body.resize(body.size() - 1);   // the last entry's blank line is the footer's
    detail::writeDictFile(u + "/time", timeName + "/uniform", "time", body);
    detail::writeDictFile(u + "/functionObjects/functionObjectProperties", timeName + "/uniform/functionObjects",
                          "functionObjectProperties", "");
}

// The time index a restart's Time::setControls reads from <startDir>/uniform/time (Time.C:302-307), or nothing when the
// start directory holds no such file (an older case, or a 0 directory): OpenFOAM then keeps the index setTime(startTime_, 0)
// gave it (Time.C:193), so a caller falls back to 0. `value` is not read: OpenFOAM's start time is the directory's name
// and `value` only feeds a warning (Time.C:310-346). A `.gz` file is found as OpenFOAM finds it (readDict, gzSlurp).
inline std::optional<long> readUniformTimeIndex(const std::string& startDir)
{
    namespace fs = std::filesystem;
    const std::string p = startDir + "/uniform/time";
    if (!fs::exists(p) && !fs::exists(p + ".gz"))
    {
        return std::nullopt;
    }
    const FoamDict td = readDict(p);
    if (!td.found("index"))
    {
        return std::nullopt;
    }
    return static_cast<long>(td.scalarOr("index", scalar(0)));
}

} // namespace brae
