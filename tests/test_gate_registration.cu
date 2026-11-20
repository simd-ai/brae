// D5: every script named validation/*_vs_openfoam.sh must be a REGISTERED, RUNNABLE, ASSERTING gate.
//
// D5 was logged as one dead script. It was 27. Every `*_vs_openfoam.sh` in validation/ that predated the
// cf -> brae rename was unregistered, still pointed at /home/ghost/cudafoam/cf, and printed its comparison
// without ever asserting a tolerance -- so none could run, and none would have failed if it had. A
// directory listing showed 27 OpenFOAM gates that did not exist.
//
// That is the same failure this project keeps finding, one level up: the validation suite was silently
// claiming coverage it did not have, exactly as the solver was silently ignoring inputs it had read. The
// per-run answer to the second was dict_audit; this is the build-time answer to the first.
//
// Three properties, and the third is the one that makes the other two worth anything:
//   1. REGISTERED  -- the name appears in a registration file (the root CMakeLists.txt, tests/CMakeLists.txt or
//                     tests/cmake/*.cmake), so ctest actually runs it.
//   2. RUNNABLE    -- no path from the retired `cf` project, which cannot resolve on any machine.
//   3. ASSERTING   -- contains an explicit failure path (exit 1 / sys.exit(1) / a FAIL branch). A gate that
//                     only prints passes forever and is worse than no gate, because it reads as coverage.
//
// Quarantined scripts live under validation/legacy_cf/ and are deliberately NOT checked: they are kept as a
// record of what was once verified, and validation/legacy_cf/README.md names which live gate covers each
// area and which are genuinely uncovered. Moving a script there is a decision to be made in the open, not a
// way to silence this test -- the README is the place that decision is written down.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>

namespace {

std::string slurp(const std::filesystem::path& p)
{
    std::ifstream in(p);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool contains(const std::string& hay, const std::string& needle)
{
    return hay.find(needle) != std::string::npos;
}

}   // namespace

int main(int argc, char** argv)
{
    // The repo root is passed by CMake; no guessing from argv[0] or the cwd.
    const std::filesystem::path root = (argc > 1) ? argv[1] : ".";
    const std::filesystem::path validation = root / "validation";
    const std::filesystem::path cmakeLists = root / "CMakeLists.txt";
    const std::filesystem::path testLists = root / "tests" / "CMakeLists.txt";
    const std::filesystem::path testGroups = root / "tests" / "cmake";

    if (!std::filesystem::is_directory(validation) || !std::filesystem::exists(cmakeLists)
        || !std::filesystem::exists(testLists))
    {
        std::printf("  FAIL cannot find %s, %s or %s -- this test is not looking at the repo it thinks it is\n",
                    validation.c_str(), cmakeLists.c_str(), testLists.c_str());
        return 1;
    }

    // The registrations live in the root file (the product), tests/CMakeLists.txt (the suite's entry point) and the
    // files it includes from tests/cmake/, one per subsystem. A gate wired in any of them is registered.
    std::vector<std::filesystem::path> registrationFiles = {cmakeLists, testLists};
    if (std::filesystem::is_directory(testGroups))
    {
        std::vector<std::filesystem::path> groups;
        for (const auto& e : std::filesystem::recursive_directory_iterator(testGroups))
        {
            if (e.is_regular_file() && e.path().extension() == ".cmake")
            {
                groups.push_back(e.path());
            }
        }
        std::sort(groups.begin(), groups.end());
        registrationFiles.insert(registrationFiles.end(), groups.begin(), groups.end());
    }

    // Only lines that actually WIRE a test count. Matching the whole file matched COMMENTS: the flatplate
    // oracle regenerator is named in a comment and was read as registered, which is the same
    // claimed-but-absent coverage this test exists to catch -- one level further up again.
    std::vector<std::string> wiringLines;
    for (const std::filesystem::path& f : registrationFiles)
    {
        std::istringstream in(slurp(f));
        std::string line;
        while (std::getline(in, line))
        {
            const std::size_t hash = line.find('#');
            const std::string code = (hash == std::string::npos) ? line : line.substr(0, hash);
            if (contains(code, "add_test") || contains(code, "COMMAND")) wiringLines.push_back(code);
        }
    }
    auto isRegistered = [&](const std::string& stem)
    {
        for (const std::string& l : wiringLines)
            if (contains(l, stem)) return true;
        return false;
    };
    int failures = 0, checked = 0;

    std::vector<std::string> names;
    for (const auto& e : std::filesystem::directory_iterator(validation))
    {
        if (!e.is_regular_file()) continue;
        const std::string n = e.path().filename().string();
        const std::string suffix = "_vs_openfoam.sh";
        if (n.size() <= suffix.size()) continue;
        if (n.compare(n.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        names.push_back(n);
    }
    std::sort(names.begin(), names.end());

    for (const std::string& n : names)
    {
        const std::string body = slurp(validation / n);
        const std::string stem = n.substr(0, n.size() - 3);   // drop ".sh"
        ++checked;

        if (!isRegistered(stem))
        {
            std::printf("  FAIL %s is not registered in any registration file -- ctest never runs it, so it is\n"
                        "       decoration, not coverage. Register it, or move it to validation/legacy_cf/\n"
                        "       and say in that README which live gate covers the area.\n", n.c_str());
            ++failures;
            continue;
        }
        if (contains(body, "cudafoam/cf") || contains(body, "cf_gpuSimpleFoam"))
        {
            std::printf("  FAIL %s references the retired `cf` project; it cannot run on any machine.\n", n.c_str());
            ++failures;
            continue;
        }
        // A gate may assert directly, or DELEGATE to a test binary whose exit status is the assertion --
        // energy_vs_openfoam ends in `"$BUILD/test_energy_frozen" ...` and is a perfectly good gate. Treating
        // that as unasserting was this check being too narrow, and it wrongly condemned a live gate.
        const bool asserts = contains(body, "exit 1")
                          || contains(body, "sys.exit(1")
                          || contains(body, "FAIL")
                          || contains(body, "$BUILD/test_");
        if (!asserts)
        {
            std::printf("  FAIL %s has no failure path (no `exit 1`, `sys.exit(1`, FAIL branch, or a\n"
                        "       delegated $BUILD/test_* whose exit status is the assertion).\n"
                        "       A gate that only prints passes forever and reads as coverage.\n", n.c_str());
            ++failures;
            continue;
        }
    }

    if (checked == 0)
    {
        std::printf("  FAIL found no *_vs_openfoam.sh at all -- the glob is wrong, not the suite\n");
        return 1;
    }

    std::printf("gate_registration: %d failures over %d gates, %zu registration files read\n", failures, checked,
                registrationFiles.size());
    return failures ? 1 : 0;
}
