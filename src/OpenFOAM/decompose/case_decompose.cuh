#pragma once
// A case split into processor<N>/ directories and put back together, without OpenFOAM: decomposePar and reconstructPar
// for the cases brae's multi-GPU solver runs (`-parallel`).
//
//   decomposeCase     constant/polyMesh and the chosen time directories of <case>  ->  processor<N>/
//   reconstructCase   the time directories of processor<N>/                          ->  <case>/<time>/
//
// THE LAYOUT IS OpenFOAM's (domainDecomposition::decomposeMesh, v2606), so the output is read by OpenFOAM's own tools and the
// other way round -- tests/case_decompose_vs_openfoam.sh holds both directions to OpenFOAM's own files:
//
//   cells        a processor's cells are its cells of the global mesh in ascending global order (cellProcAddressing)
//   faces        1. the faces whose two cells are both on the processor, in ascending global face order;
//                2. each original patch's faces on the processor, in the patch's own order (a patch with none stays, empty);
//                3. one `procBoundary<me>to<nbr>` patch per neighbour processor, neighbours ascending, whose faces are the
//                   inter-processor faces in ascending global face order. The processor that owns the global face keeps
//                   it as it is; the other takes it with its vertices reversed (`reverseFace`: first vertex stays, the rest
//                   run backwards), so its normal points out of ITS cell. faceProcAddressing carries the global face + 1,
//                   negative for a reversed one.
//   points       the points the processor's faces use, in ascending global order (pointProcAddressing)
//   patch fields a patch's `nonuniform` lists are cut with its faces; a `processor` patch holds the value of the cell on the
//                OTHER side (surface fields: the face value, sign-flipped for a reversed face), as decomposePar sets it
//
// WHAT IS NOT DONE, said rather than guessed: cyclic, cyclicAMI, cyclicACMI and processor patches in the mesh, face zones and
// non-empty point zones, coded boundary conditions, point fields, and any decomposition method but scotch, simple,
// hierarchical (order xyz) and manual. The output is ascii whatever the input's format. Refusals name the entry.
#include "cf_types.cuh"

#include <string>
#include <vector>

namespace brae {
namespace decompose {

struct Options
{
    int                      nProcs = 0;          // 0: decomposeParDict's numberOfSubdomains
    std::string              method;              // "": decomposeParDict's `method`, else scotch
    bool                     force = false;       // replace the processor<N>/ directories that are there
    std::vector<std::string> times;               // time directories to decompose; empty: the time a run would start from
    bool                     allTimes = false;    // every numeric time directory
    bool                     quiet = false;
};

struct Report
{
    int                      nProcs = 0;
    std::string              method;
    long long                nCells = 0;
    long long                nInterProcFaces = 0;     // faces between processors (each counted once)
    std::vector<long long>   cellsPerProc;
    std::vector<std::string> times;                   // the time directories decomposed
    std::vector<std::string> fields;                  // the field files decomposed, over all times
};

// The cell -> processor assignment alone (decomposeParDict's method, or `method` when given). `nProcs` 0 reads the dictionary.
std::vector<label> assignCells(const std::string& caseDir, int nProcs, const std::string& method);

Report decomposeCase(const std::string& caseDir, const Options& opt);

// The same, from a cell -> processor assignment the caller made (a test's, OpenFOAM's `manual` file)
Report decomposeCaseWith(const std::string& caseDir, const std::vector<label>& cellToProc, int nProcs, const Options& opt);

struct ReconstructOptions
{
    std::vector<std::string> times;               // time directories to reconstruct; empty: every one processor0 has
                                                  // that the case does not (reconstructPar's default)
    bool                     latestTime = false;  // only the newest of them
    bool                     force = false;       // replace a time directory that is already in the case
    bool                     quiet = false;
};

struct ReconstructReport
{
    std::vector<std::string> times;
    std::vector<std::string> fields;
};

ReconstructReport reconstructCase(const std::string& caseDir, const ReconstructOptions& opt);

// processor<N> directories under the case, as `processor0 .. processor<N-1>`; 0 when none
int processorDirectories(const std::string& caseDir);

// The time directory a run of the case would start from (controlDict startFrom/startTime)
std::string startTimeName(const std::string& caseDir);

// What `brae -parallel` does when the case has no processor directories: decompose into `nProcs` and report where the
// run's own output is to be reconstructed. Called by every rank; the first to take the lock decomposes, the others wait for it.
// Returns true when this call found no processor directories and made them.
bool decomposeForRun(const std::string& caseDir, int nProcs, bool quiet = false);

// The marker decomposeForRun leaves in the case: a run that finds it puts the case back together when it ends (reconstructCase),
// since the one who started it never saw a processor directory.
inline constexpr const char* kAutoMarker = ".brae_decomposed";
bool decomposedByRun(const std::string& caseDir);

}   // namespace decompose
}   // namespace brae
