// WriteControl's write cadence from a controlDict: `writeInterval`, its older name `writeFrequency`, and
// neither.
// TimeIO.C:286-297 reads writeInterval and, where it is absent, writeFrequency; with neither OpenFOAM stops.
// brae read writeInterval alone with a default of 1e30 ("never"), so a case that spelt the entry
// writeFrequency ran to its end with no intermediate write and no word. It reads both now, and with neither
// it says so and writes the final state alone.
// THE CONTROL is the reader's own switch, checked at every construction: BRAE_CONTROL_WRITE_FREQUENCY_IGNORED.
// Also here, the same entry on the interFoam path: WriteCadence::read (time_controls.cuh) read writeInterval
// alone as well; and TimeIO.C:288-293's stop on `writeControl timeStep` with a writeInterval below 1.
#include "time_controls.cuh"
#include "write_control.cuh"
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

// the iterations of 1..20 a controlDict with these entries writes at, as a string
std::string writesAt(
    const std::string& dir,
    const std::string& entries)
{
    const std::string path = dir + "/controlDict";
    {
        std::ofstream f(path);
        f << "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
          << "application simpleFoam;\nstartTime 0;\nendTime 20;\ndeltaT 1;\nwriteControl timeStep;\n"
          << entries << "\n";
    }
    WriteControl wc(readDict(path));
    std::string out;
    for (int it = 1; it <= 20; ++it)
    {
        if (wc.isWriteTime(it, scalar(it)))
        {
            out += std::to_string(it) + " ";
        }
    }
    return out;
}

} // namespace

int main()
{
    std::printf("== WriteControl: writeInterval, writeFrequency, and neither\n");
    const std::string dir = (std::filesystem::temp_directory_path() / "brae_write_control_interval").string();
    std::filesystem::create_directories(dir);
    const std::string byInterval = writesAt(dir, "writeInterval 5;");
    const std::string byFrequency = writesAt(dir, "writeFrequency 5;");
    const std::string both = writesAt(dir, "writeInterval 10;\nwriteFrequency 5;");
    // the notice is said once a process (brae_notice.cuh keeps the ones said): none so far, one after `neither`
    const std::size_t noticesBefore = detail::noticeSeen().size();
    const std::string neither = writesAt(dir, "");
    const std::size_t noticesAfter = detail::noticeSeen().size();
    setenv("BRAE_CONTROL_WRITE_FREQUENCY_IGNORED", "1", 1);
    const std::string ignored = writesAt(dir, "writeFrequency 5;");
    unsetenv("BRAE_CONTROL_WRITE_FREQUENCY_IGNORED");
    const std::size_t noticesControl = detail::noticeSeen().size();
    // the interFoam loops' cadence, from the same entries
    const auto cadence = [&](const std::string& entries)
    {
        const std::string path = dir + "/controlDict";
        {
            std::ofstream f(path);
            f << "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
              << "writeControl runTime;\n" << entries << "\n";
        }
        return WriteCadence::read(readDict(path)).writeInterval;
    };
    const scalar cadenceInterval = cadence("writeInterval 0.05;");
    const scalar cadenceFrequency = cadence("writeFrequency 0.05;");
    const scalar cadenceBoth = cadence("writeInterval 0.1;\nwriteFrequency 0.05;");
    // `writeControl timeStep` with a writeInterval below 1
    bool refused = false;
    try
    {
        (void)writesAt(dir, "writeInterval 0.5;");
    }
    catch (const std::exception& e)
    {
        refused = std::string(e.what()).find("writeInterval") != std::string::npos;
    }
    std::filesystem::remove_all(dir);
    std::printf("  writes at: writeInterval 5 [%s]; writeFrequency 5 [%s]; both, 10 and 5 [%s]; neither [%s]\n",
                byInterval.c_str(), byFrequency.c_str(), both.c_str(), neither.c_str());
    check("writeFrequency is read where writeInterval is absent, and writeInterval wins where both are there",
          byInterval == "5 10 15 20 " && byFrequency == byInterval && both == "10 20 ");
    std::printf("  notices said: %zu before `neither`, %zu after, %zu after the control; the interFoam cadence "
                "reads %g, %g and %g; timeStep with writeInterval 0.5 %s\n", noticesBefore, noticesAfter,
                noticesControl, static_cast<double>(cadenceInterval), static_cast<double>(cadenceFrequency),
                static_cast<double>(cadenceBoth), refused ? "is refused" : "RUNS");
    check("with neither entry no intermediate state is written, and that is said (one notice, none before it); "
          "timeStep with a writeInterval below 1 is refused by name",
          neither.empty() && noticesAfter == noticesBefore + 1 && refused);
    check("the interFoam loops' cadence reads writeFrequency where writeInterval is absent too",
          cadenceInterval == scalar(0.05) && cadenceFrequency == scalar(0.05) && cadenceBoth == scalar(0.1));
    std::printf("  CONTROL, writeFrequency not read: writes at [%s]\n", ignored.c_str());
    check("CONTROL: with writeFrequency ignored, as before, the same case writes nothing -- and says no notice "
          "that the entry is absent", ignored.empty() && noticesControl == noticesAfter);
    std::printf("test_write_control_interval: %d failures\n", failures);
    return failures ? 1 : 0;
}
