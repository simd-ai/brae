#pragma once
// The two roots of a case under `-parallel`, and which rank this process is.
//
// OpenFOAM's decomposed case keeps what every rank shares in the CASE -- system/, the dictionaries of
// constant/ (transportProperties, g, turbulenceProperties, ...) -- and what belongs to one rank in
// `processor<rank>/`: constant/polyMesh and its zones, the time directories, the output
// (IOobject::globalFilePath for the first, objectPath for the second). The single-GPU solvers have always
// had ONE directory, which is both. CaseRoots names the two:
//
//   dictDir  the case: system/, constant/<dictionaries>, postProcessing/ (written by the master alone)
//   procDir  processor<rank>/ under -parallel; the case itself otherwise. constant/polyMesh, the cell/face/
//            point zones, the start and write time directories, the AMG cache and every field read.
//
// Without -parallel the two are the same string and every run is what it was before. The converting
// constructors are deliberate: an interface that took a case directory takes a CaseRoots, and every caller
// that still passes the one directory is a serial caller.
//
// RANK AND SIZE come from the launcher's environment (no MPI_Init is needed to read them): Open MPI's
// OMPI_COMM_WORLD_RANK/SIZE, PMI/Hydra's PMI_RANK/PMI_SIZE, Slurm's SLURM_PROCID/SLURM_NTASKS -- or from
// explicit `-rank N -nprocs M`, which a harness that runs several virtual ranks in one process needs and
// which take precedence over the environment.
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace brae {

struct CaseRoots
{
    std::string dictDir;
    std::string procDir;
    int         rank = 0;
    int         nProcs = 1;
    bool        parallel = false;

    CaseRoots() = default;
    // serial: both roots are the case directory
    CaseRoots(const std::string& caseDir) : dictDir(caseDir), procDir(caseDir) {}
    CaseRoots(const char* caseDir) : dictDir(caseDir), procDir(caseDir) {}

    bool isMaster() const { return rank == 0; }

    static std::string procName(int r) { return "processor" + std::to_string(r); }

    // processor0: OpenFOAM's master. `startFrom latestTime` reads the time list from here alone so that every
    // rank agrees on the start even when a rank's directory lags.
    std::string masterProcDir() const
    {
        return parallel ? dictDir + "/" + procName(0) : dictDir;
    }
};

// The rank/size a launcher put in the environment, or false when none did.
inline bool rankFromEnvironment(int& rank, int& nProcs)
{
    struct Pair { const char* rank; const char* size; };
    static const Pair pairs[] = {
        {"OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"},
        {"PMI_RANK", "PMI_SIZE"},
        {"SLURM_PROCID", "SLURM_NTASKS"},
    };
    for (const Pair& p : pairs)
    {
        const char* r = std::getenv(p.rank);
        const char* s = std::getenv(p.size);
        if (r && s && *r && *s)
        {
            rank = std::atoi(r);
            nProcs = std::atoi(s);
            return true;
        }
    }
    return false;
}

// Number of `processor<digits>` directories in the case, as OpenFOAM counts them to check numberOfSubdomains
// against the run's size (argList.C, "number of processor directories = N is not equal to the number of
// processors = M").
inline int countProcessorDirs(const std::string& caseDir)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    int n = 0;
    for (const auto& e : fs::directory_iterator(caseDir, ec))
    {
        const std::string nm = e.path().filename().string();
        if (!e.is_directory() || nm.size() <= 9 || nm.compare(0, 9, "processor") != 0) continue;
        bool digits = true;
        for (std::size_t i = 9; i < nm.size(); ++i) digits = digits && nm[i] >= '0' && nm[i] <= '9';
        if (digits) ++n;
    }
    return n;
}

// The roots for a run. `parallel` is the `-parallel` flag; rankFlag/nProcsFlag are `-rank`/`-nprocs` (-1 when
// absent). Throws, naming what is wrong, for anything OpenFOAM would stop on.
inline CaseRoots makeCaseRoots(
    const std::string& caseDir,
    bool               parallel,
    int                rankFlag = -1,
    int                nProcsFlag = -1)
{
    if (!parallel)
    {
        if (rankFlag >= 0 || nProcsFlag >= 0)
            throw std::runtime_error("`-rank`/`-nprocs` select a processor directory and only mean something with `-parallel`.");
        return CaseRoots(caseDir);
    }
    if ((rankFlag >= 0) != (nProcsFlag >= 0))
        throw std::runtime_error("`-rank` and `-nprocs` go together: give both, or neither and let the launcher's environment "
                                 "(OMPI_COMM_WORLD_RANK/SIZE, PMI_RANK/SIZE, SLURM_PROCID/NTASKS) name them.");
    CaseRoots r;
    r.dictDir = caseDir;
    r.parallel = true;
    if (rankFlag >= 0)
    {
        r.rank = rankFlag;
        r.nProcs = nProcsFlag;
    }
    else if (!rankFromEnvironment(r.rank, r.nProcs))
    {
        throw std::runtime_error("`-parallel` needs to know the rank: run it under mpirun (OMPI_COMM_WORLD_RANK/SIZE, "
                                 "PMI_RANK/SIZE or SLURM_PROCID/NTASKS) or give `-rank N -nprocs M`.");
    }
    if (r.nProcs < 1 || r.rank < 0 || r.rank >= r.nProcs)
        throw std::runtime_error("rank " + std::to_string(r.rank) + " of " + std::to_string(r.nProcs)
                                 + " processes is not a valid rank.");
    const int dirs = countProcessorDirs(caseDir);
    if (dirs != r.nProcs)
        throw std::runtime_error(
            "the case has " + std::to_string(dirs) + " processor directories and the run has "
            + std::to_string(r.nProcs) + " processes (decomposePar's numberOfSubdomains must equal the size of the run).");
    r.procDir = caseDir + "/" + CaseRoots::procName(r.rank);
    return r;
}

}  // namespace brae
