// THE WHOLE CLOCK AGAINST OPENFOAM'S OWN LOG, one step a row: setDeltaT.H and then Time::adjustDeltaT -- Time's
// write cadence and every function object's (time_controls.cuh, WriteCadence and FunctionObjectCadence).
//
// tests/test_time_controls_openfoam_log.cu holds setDeltaT.H alone, on runs whose write control lands a step on
// nothing. Here the run's controlDict is READ, by the functions the solver's reader calls, and the clock is
// carried from row to row as the two loops carry it: the deltaT a row begins with and the two Courant numbers
// go to setDeltaTVoF with the time since the start and the cadence, the result must be the row's deltaT TO THE
// BIT, the time then advances by OpenFOAM's deltaT and the cadence's indices with it. A row is judged from
// OpenFOAM's state, so one wrong row does not make the next one wrong.
//
// The rows come from tests/interfoam_write/clock/function_object_write_times.sh and
// function_object_write_entries.sh, which stage the runs and read this program's RESULT line; the controls
// (BRAE_CONTROL_FUNCTION_OBJECT_STEP=<part>) are run through it too.
//
// usage: test_time_controls_cadence_log <rows file> <case directory>
//        a row: <deltaT before> <Courant number> <interface Courant number> <deltaT after>
#include "time_controls.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {

struct Row
{
    scalar before;
    scalar co;
    scalar alphaCo;
    scalar after;
};

bool sameBits(
    scalar a,
    scalar b)
{
    return std::memcmp(&a, &b, sizeof(scalar)) == 0;
}

} // namespace

int main(
    int argc,
    char** argv)
{
    if (argc != 3)
    {
        std::printf("usage: %s <rows file> <case directory>\n", argv[0]);
        return 2;
    }
    std::vector<Row> rows;
    {
        std::ifstream in(argv[1]);
        Row r;
        std::string a;
        std::string b;
        std::string c;
        std::string d;
        while (in >> a >> b >> c >> d)
        {
            r.before = std::strtod(a.c_str(), nullptr);
            r.co = std::strtod(b.c_str(), nullptr);
            r.alphaCo = std::strtod(c.c_str(), nullptr);
            r.after = std::strtod(d.c_str(), nullptr);
            rows.push_back(r);
        }
    }
    const FoamDict controlDict = readDict(std::string(argv[2]) + "/system/controlDict");
    const VoFTimeControls tc = VoFTimeControls::read(controlDict, false);
    WriteCadence w = WriteCadence::read(controlDict);
    // a cold start: the rows' runs begin at 0
    const scalar startTime = 0;
    w.startTime = startTime;

    // the same cadence with no object in it: what Time's own trimming leaves, to count the rows each decides
    WriteCadence timeAlone = w;
    timeAlone.functionObjects.clear();

    long wrong = 0;
    long byObject = 0;
    long byTime = 0;
    long byClip = 0;
    scalar worst = 0;
    long worstRow = -1;
    scalar t = startTime;
    for (std::size_t k = 0; k < rows.size(); ++k)
    {
        const Row& r = rows[k];
        const scalar got = setDeltaTVoF(r.before, r.co, r.alphaCo, tc, t - startTime, &w);
        const scalar untrimmed = setDeltaTVoF(r.before, r.co, r.alphaCo, tc);
        const scalar byTimeAlone = setDeltaTVoF(r.before, r.co, r.alphaCo, tc, t - startTime, &timeAlone);
        byTime += sameBits(byTimeAlone, untrimmed) ? 0 : 1;
        byObject += sameBits(got, byTimeAlone) ? 0 : 1;
        // a decrease an object held at half the step it was handed: the factor that is 2 and not Time's 5
        byClip += (!sameBits(got, byTimeAlone) && sameBits(got, scalar(0.5)*byTimeAlone)) ? 1 : 0;
        if (!sameBits(got, r.after))
        {
            ++wrong;
            const scalar gap = std::fabs(got - r.after)/std::fabs(r.after);
            if (gap > worst)
            {
                worst = gap;
                worstRow = static_cast<long>(k);
            }
        }
        // Time::operator++ with OpenFOAM's own step, and the indices as the loops move them
        t += r.after;
        w.advance(t - startTime, r.after);
        timeAlone.advance(t - startTime, r.after);
    }
    if (worstRow >= 0)
    {
        const Row& r = rows[static_cast<std::size_t>(worstRow)];
        std::printf("  the row furthest off: step %ld, deltaT before %.17g, Co %.17g, alphaCo %.17g, "
                    "OpenFOAM %.17g\n", worstRow + 1, (double)r.before, (double)r.co, (double)r.alphaCo,
                    (double)r.after);
    }
    std::string names;
    for (const FunctionObjectCadence& fo : w.functionObjects)
    {
        names += (names.empty() ? "" : ",") + fo.name;
    }
    std::printf("RESULT rows=%zu wrong=%ld worst=%.1e byTime=%ld byObject=%ld byClip=%ld objects=%zu "
                "names=%s endTime=%.17g\n", rows.size(), wrong, (double)worst, byTime, byObject, byClip,
                w.functionObjects.size(), names.empty() ? "-" : names.c_str(), (double)t);
    return 0;
}
