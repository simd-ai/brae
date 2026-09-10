#pragma once
// `brae benchmark [sample]` -- run the standard workload and write a comparable result.
//
// Samples are not built into the binary. They live as BRANCHES of the template repository
// github.com/simd-ai/brae-bench, one branch per case, and are pulled and cached on first use:
//
//   brae benchmark                            the default sample
//   brae benchmark pimplefoam/pitzDaily-1M    that branch
//   brae benchmark --list                     what the repo offers
//
// Branches are grouped by the solver that runs them: pimplefoam/<case>, simplefoam/<case>.
//
// A branch ships a normal OpenFOAM case plus a `brae-bench.json` manifest naming it, its cell count and the
// VRAM it needs, so the result is self-describing and nothing has to be parsed out of solver output.
//
// SECURITY -- read before adding anything here. A brae case is not inert data: `codedFixedValue` / `codedMixed`
// boundary conditions carry CUDA source that brae compiles and runs on the device (coded_bc_setup.cuh). Running
// a pulled case is therefore running someone's code, so this file holds two hard limits:
//
//   1. The repository is PINNED to a constant below. No server, no case file and no command-line argument can
//      redirect it. Only a sample NAME is ever variable, and it is validated as a branch name.
//   2. A pulled case containing coded boundary conditions or #codeStream is REFUSED, not compiled. Benchmarks
//      measure the solver; they have no reason to ship code, and a benchmark that could would be a remote
//      code-execution path onto every contributor's machine.
//
// BRAE_BENCH_REPO overrides the URL for the test suite ONLY (tests/brae_benchmark.sh builds a local fixture
// repo). It is deliberately not documented in --help: a contributor who can be talked into setting it can be
// talked into running anything.
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"   // readInvariants: evidence the run computed something
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace brae {
namespace bench {

// The template repository. Pinned: see the security note above.
inline constexpr const char* kBenchRepo = "https://github.com/simd-ai/brae-bench.git";
inline constexpr const char* kDefaultSample = "pimplefoam/pitzDaily-1M";

inline std::string repoUrl()
{
    const char* over = std::getenv("BRAE_BENCH_REPO");
    return over && *over ? over : kBenchRepo;
}

inline std::filesystem::path cacheRoot()
{
    if (const char* c = std::getenv("BRAE_BENCH_CACHE")) return std::filesystem::path(c);
    if (const char* x = std::getenv("XDG_CACHE_HOME"))   return std::filesystem::path(x) / "brae" / "bench";
    if (const char* h = std::getenv("HOME"))             return std::filesystem::path(h) / ".cache" / "brae" / "bench";
    return std::filesystem::path("/tmp/brae-bench");
}

// A sample name becomes a git branch AND a cache directory, so it must look like a branch and nothing else.
// Samples are grouped by solver -- `simplefoam/kepsilon-pitzDaily`, `pimplefoam/pitzDaily-1M` -- so '/' is
// allowed, which means path traversal has to be excluded explicitly rather than by banning the separator:
// no '..' anywhere, no empty/leading/trailing component, and no component starting with '-' or '.'.
// Rejecting rather than sanitising -- a name we would have to rewrite is a name we do not understand.
inline void validateSampleName(const std::string& s)
{
    const auto bad = [&](const std::string& why) {
        throw std::runtime_error("benchmark sample '" + s + "' " + why);
    };
    if (s.empty() || s.size() > 96) bad("must be 1-96 characters");
    if (s.find("..") != std::string::npos) bad("may not contain '..'");
    if (s.front() == '/' || s.back() == '/') bad("may not start or end with '/'");

    std::size_t start = 0;
    while (start <= s.size())
    {
        const std::size_t slash = s.find('/', start);
        const std::string part = s.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (part.empty()) bad("has an empty path component");
        if (part.front() == '-' || part.front() == '.')
            bad("has a component starting with '-' or '.' (" + part + ")");
        for (const char c : part)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.'))
                bad("may only contain letters, digits, '-', '_', '.' and '/'");
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
}

// Run argv, no shell, optionally capturing stdout. Returns the exit status.
inline int run(const std::vector<std::string>& argv, std::string* captured = nullptr, bool quiet = false)
{
    int pipefd[2] = {-1, -1};
    if (captured && ::pipe(pipefd) != 0) throw std::runtime_error("pipe() failed");

    const pid_t pid = ::fork();
    if (pid < 0) throw std::runtime_error("fork() failed");
    if (pid == 0)
    {
        if (captured)
        {
            ::close(pipefd[0]);
            ::dup2(pipefd[1], STDOUT_FILENO);
            ::close(pipefd[1]);
        }
        if (quiet)
        {
            const int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0)
            {
                ::dup2(devnull, STDERR_FILENO);
                if (!captured) ::dup2(devnull, STDOUT_FILENO);   // else `git --version` lands in brae's output
                ::close(devnull);
            }
        }
        std::vector<char*> av;
        for (const std::string& a : argv) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        ::execvp(av[0], av.data());          // explicit argv, never a shell
        ::_exit(127);
    }
    if (captured)
    {
        ::close(pipefd[1]);
        char buf[4096];
        ssize_t n;
        while ((n = ::read(pipefd[0], buf, sizeof buf)) > 0) captured->append(buf, static_cast<std::size_t>(n));
        ::close(pipefd[0]);
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

inline void requireGit()
{
    if (run({"git", "--version"}, nullptr, true) != 0)
        throw std::runtime_error("git is needed to fetch benchmark samples but is not installed "
                                 "(Ubuntu/Debian: sudo apt-get install -y git)");
}

// Branch names in the template repo == the samples on offer.
inline std::vector<std::string> listSamples()
{
    requireGit();
    std::string out;
    if (run({"git", "ls-remote", "--heads", repoUrl()}, &out, true) != 0)
        throw std::runtime_error("cannot reach the benchmark repository " + repoUrl() +
                                 " (no network, or the repository is private)");
    std::vector<std::string> names;
    std::istringstream is(out);
    std::string line;
    while (std::getline(is, line))
    {
        const std::size_t at = line.find("refs/heads/");
        if (at == std::string::npos) continue;
        std::string name = line.substr(at + 11);
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) name.pop_back();
        if (!name.empty() && name != "main" && name != "master") names.push_back(name);
    }
    return names;
}

// Pull the sample's branch into the cache (shallow), or reuse what is already there. BRAE_BENCH_REFRESH=1 forces
// a re-pull, which is also how a sample is updated.
inline std::filesystem::path fetchSample(const std::string& sample)
{
    validateSampleName(sample);
    requireGit();
    const std::filesystem::path dir = cacheRoot() / sample;
    const bool refresh = std::getenv("BRAE_BENCH_REFRESH") != nullptr;

    if (std::filesystem::exists(dir / ".git") && !refresh)
        return dir;

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir.parent_path(), ec);
    std::printf("  fetching sample '%s' from %s\n", sample.c_str(), repoUrl().c_str());
    if (run({"git", "clone", "--quiet", "--depth", "1", "--branch", sample, repoUrl(), dir.string()}) != 0)
    {
        std::filesystem::remove_all(dir, ec);
        throw std::runtime_error("no benchmark sample '" + sample + "' (it is a branch of " + repoUrl() +
                                 "). Run `brae benchmark --list` to see what is available.");
    }
    return dir;
}

// SECURITY: refuse a pulled case that carries code. brae compiles codedFixedValue / codedMixed bodies with NVRTC
// and runs them on the device, and a `type coded;` Function1 with the host C++ compiler, so accepting them
// here would make `brae benchmark <anything>` an arbitrary-code path onto a contributor's machine.
// Benchmarks measure the solver and have no reason to ship code.
inline void refuseExecutableCase(const std::filesystem::path& dir)
{
    static const char* kCodeMarkers[] = {"codedFixedValue", "codedMixed", "#codeStream", "codedFunctionObject"};
    // `type coded;` with any spacing -- the Function1 form carries no distinctive word of its own.
    static const std::regex kCodedFunction1(R"(\btype\s+coded\s*;)");
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(dir, ec);
         it != std::filesystem::recursive_directory_iterator(); ++it)
    {
        if (it->is_directory() && it->path().filename() == ".git") { it.disable_recursion_pending(); continue; }
        if (!it->is_regular_file(ec)) continue;
        if (std::filesystem::file_size(it->path(), ec) > (4u << 20)) continue;   // mesh data, not a dictionary
        std::ifstream f(it->path());
        if (!f) continue;
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (const char* marker : kCodeMarkers)
            if (text.find(marker) != std::string::npos)
                throw std::runtime_error(
                    std::string("benchmark sample carries executable content (") + marker + " in " +
                    std::filesystem::relative(it->path(), dir).string() +
                    "). brae compiles coded boundary conditions on the device, so a benchmark is not allowed to "
                    "contain them. Refusing to run it.");
        if (std::regex_search(text, kCodedFunction1))
            throw std::runtime_error(
                "benchmark sample carries executable content (a `type coded;` Function1 in " +
                std::filesystem::relative(it->path(), dir).string() +
                "). brae compiles coded Function1s with the host C++ compiler, so a benchmark is not allowed "
                "to contain them. Refusing to run it.");
    }
}

struct Manifest
{
    std::string sample, description, solver;
    long long cells = 0;
    int minVramMb = 0;
};

// brae-bench.json in the branch root: what this sample is. Absent -> the sample is not a benchmark case.
inline Manifest readManifest(const std::filesystem::path& dir, const std::string& sample)
{
    const std::filesystem::path path = dir / "brae-bench.json";
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("sample '" + sample + "' has no brae-bench.json manifest, so it is not a "
                                 "benchmark case. Every branch of the template repo must ship one.");
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto str = [&](const char* key) -> std::string {
        const std::size_t k = text.find(std::string("\"") + key + "\"");
        if (k == std::string::npos) return "";
        const std::size_t q1 = text.find('"', text.find(':', k) + 1);
        if (q1 == std::string::npos) return "";
        const std::size_t q2 = text.find('"', q1 + 1);
        return q2 == std::string::npos ? "" : text.substr(q1 + 1, q2 - q1 - 1);
    };
    auto num = [&](const char* key) -> long long {
        const std::size_t k = text.find(std::string("\"") + key + "\"");
        if (k == std::string::npos) return 0;
        return std::strtoll(text.c_str() + text.find(':', k) + 1, nullptr, 10);
    };

    Manifest m;
    m.sample = str("sample").empty() ? sample : str("sample");
    m.description = str("description");
    m.solver = str("solver");
    m.cells = num("cells");
    m.minVramMb = static_cast<int>(num("min_vram_mb"));
    return m;
}

inline std::string commitOf(const std::filesystem::path& dir)
{
    std::string sha;
    if (run({"git", "-C", dir.string(), "rev-parse", "HEAD"}, &sha, true) != 0) return "";
    while (!sha.empty() && std::isspace(static_cast<unsigned char>(sha.back()))) sha.pop_back();
    return sha;
}

// Numbers that say the run computed the right thing, not merely that it finished.
//
// Deliberately NOT a hash. Two GPUs sum in different orders, so a digest of the field values differs between
// machines that are both perfectly correct -- a checksum would flag the whole network. These are a handful of
// scalar invariants instead, compared numerically against a tolerance by whoever collects them
// (brae-cloud docs/08-benchmark-job.md).
struct Invariants
{
    bool ok = false;
    long long cells = 0;
    double uMean = 0, uMax = 0;
    double pMean = 0, pMin = 0, pMax = 0;
};

// The newest numeric time directory holding a p field: what the run finished with.
inline std::string lastWrittenTimeDir(const std::filesystem::path& caseDir)
{
    std::error_code ec;
    std::string best;
    double bestValue = -1;
    for (const auto& e : std::filesystem::directory_iterator(caseDir, ec))
    {
        if (!e.is_directory()) continue;
        const std::string name = e.path().filename().string();
        char* end = nullptr;
        const double t = std::strtod(name.c_str(), &end);
        if (end == name.c_str() || *end != '\0') continue;          // not a pure number
        if (!std::filesystem::exists(e.path() / "p") && !std::filesystem::exists(e.path() / "p.gz")) continue;
        if (t > bestValue) { bestValue = t; best = e.path().string(); }
    }
    return best;
}

inline Invariants readInvariants(const std::filesystem::path& caseDir)
{
    Invariants inv;
    const std::string dir = lastWrittenTimeDir(caseDir);
    if (dir.empty()) return inv;                                    // nothing written: not a usable result

    try
    {
        const FieldData<scalar> p = readField<scalar>(dir + "/p");
        const FieldData<vector> U = readField<vector>(dir + "/U");
        if (p.internalUniform || U.internalUniform) return inv;      // a uniform field means it never solved
        if (p.internalField.empty() || U.internalField.size() != p.internalField.size()) return inv;

        inv.cells = static_cast<long long>(p.internalField.size());
        double pSum = 0, uSum = 0;
        inv.pMin = inv.pMax = static_cast<double>(p.internalField[0]);
        for (const scalar v : p.internalField)
        {
            const double d = static_cast<double>(v);
            pSum += d;
            if (d < inv.pMin) inv.pMin = d;
            if (d > inv.pMax) inv.pMax = d;
        }
        for (const vector& v : U.internalField)
        {
            const double m = std::sqrt(double(v.x) * double(v.x) + double(v.y) * double(v.y)
                                       + double(v.z) * double(v.z));
            uSum += m;
            if (m > inv.uMax) inv.uMax = m;
        }
        inv.pMean = pSum / double(inv.cells);
        inv.uMean = uSum / double(inv.cells);
        inv.ok = true;
    }
    catch (const std::exception&)
    {
        // A field we cannot read is a missing invariant, not a failed benchmark: the run still happened, and
        // the runtime is still worth reporting. The absence is visible in the result.
        inv.ok = false;
    }
    return inv;
}

inline std::string jsonEscape(const std::string& s)
{
    std::string o;
    for (const char c : s)
    {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n')        o += "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) continue;
        else                       o += c;
    }
    return o;
}

// `brae benchmark [sample|--list]`. argv is everything after the `benchmark` word.
inline int runBenchmark(const std::vector<std::string>& args, const std::string& braeExe)
{
    std::string sample;
    for (const std::string& a : args)
    {
        if (a == "--list")
        {
            const std::vector<std::string> names = listSamples();
            if (names.empty()) { std::printf("no samples published yet at %s\n", repoUrl().c_str()); return 0; }
            std::printf("benchmark samples (%s):\n", repoUrl().c_str());
            for (const std::string& n : names)
                std::printf("  %s%s\n", n.c_str(), n == kDefaultSample ? "   (default)" : "");
            std::printf("\nrun one with:  brae benchmark <sample>\n");
            return 0;
        }
        if (a == "--help" || a == "-h")
        {
            std::printf("brae benchmark [sample]   run the standard workload and write brae-benchmark.json\n"
                        "brae benchmark --list     list the samples published at %s\n\n"
                        "Samples are branches of the template repository, pulled and cached on first use.\n"
                        "With no sample, '%s' runs.\n", kBenchRepo, kDefaultSample);
            return 0;
        }
        if (a[0] == '-') throw std::runtime_error("unknown option '" + a + "' (brae benchmark --help)");
        if (!sample.empty()) throw std::runtime_error("give at most one sample name");
        sample = a;
    }
    if (sample.empty()) sample = kDefaultSample;

    const std::filesystem::path dir = fetchSample(sample);
    refuseExecutableCase(dir);                     // before anything in the case is read
    const Manifest man = readManifest(dir, sample);
    const std::string commit = commitOf(dir);

    std::printf("brae benchmark | sample=%s | %s\n", man.sample.c_str(), man.description.c_str());
    if (man.cells) std::printf("  %lld cells", man.cells);
    if (!man.solver.empty()) std::printf("  |  %s", man.solver.c_str());
    if (man.cells || !man.solver.empty()) std::printf("\n");

    // The run is a normal solve of the sample case, through this same binary -- so the benchmark measures
    // exactly what a user gets, including the solver the case's `application` selects.
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = run({braeExe, "-case", dir.string()});
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Read what the run produced, so the result carries evidence of correctness and not just a duration.
    const Invariants inv = (rc == 0) ? readInvariants(dir) : Invariants{};

    const std::string outPath = "brae-benchmark.json";
    {
        std::ofstream j(outPath);
        j.precision(12);
        j << "{\n"
          << "  \"sample\": \"" << jsonEscape(man.sample) << "\",\n"
          << "  \"sample_commit\": \"" << jsonEscape(commit) << "\",\n"
          << "  \"cells\": " << man.cells << ",\n"
          << "  \"solver\": \"" << jsonEscape(man.solver) << "\",\n"
          << "  \"runtime_s\": " << seconds << ",\n"
          << "  \"success\": " << (rc == 0 ? "true" : "false") << ",\n"
          << "  \"exit_code\": " << rc;
        if (inv.ok)
            j << ",\n  \"invariants\": {"
              << "\"cells\": " << inv.cells
              << ", \"u_mean\": " << inv.uMean
              << ", \"u_max\": " << inv.uMax
              << ", \"p_mean\": " << inv.pMean
              << ", \"p_min\": " << inv.pMin
              << ", \"p_max\": " << inv.pMax << "}";
        j << "\n}\n";
    }
    if (rc == 0) std::printf("\nbenchmark complete: %.2f s   ->  %s\n", seconds, outPath.c_str());
    else         std::fprintf(stderr, "\nbenchmark FAILED (brae exited %d) after %.2f s   ->  %s\n",
                              rc, seconds, outPath.c_str());
    return rc == 0 ? 0 : 1;
}

}  // namespace bench
}  // namespace brae
