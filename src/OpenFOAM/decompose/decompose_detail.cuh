#pragma once
// What case_decompose.cu (the mesh and the orchestration) and field_decompose.cu (the fields, both ways) share. Not an interface:
// include case_decompose.cuh.
#include "case_decompose.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"

#include <string>
#include <vector>

namespace brae {
namespace decompose {
namespace detail {

struct PatchLayout
{
    std::string              name;
    std::string              type;
    std::vector<std::string> inGroups;
    label                    start = 0;
    label                    size = 0;
};

// One processor's mesh as addressing into the global one (OpenFOAM's *ProcAddressing files)
struct ProcLayout
{
    std::vector<label>       cells;         // local cell  -> global cell
    std::vector<label>       faces;         // local face  -> global face + 1, negative for a reversed face
    std::vector<label>       points;        // local point -> global point
    std::vector<label>       boundary;      // local patch -> global patch, -1 for a processor patch
    std::vector<PatchLayout> patches;       // the physical patches in the global order, then procBoundary<me>to<nbr>
    std::vector<label>       patchNbr;      // per local patch: the neighbouring processor, -1 for a physical one
    label                    nInternal = 0; // internal faces
};

struct GlobalLayout
{
    label                    nCells = 0, nPoints = 0, nFaces = 0, nInternalFaces = 0;
    std::vector<PatchLayout> patches;
    std::vector<label>       owner, neighbour;   // the global mesh's, for the processor patch values of a field
};

struct Plan
{
    int                      nProcs = 0;
    GlobalLayout             global;
    std::vector<ProcLayout>  proc;
    std::vector<label>       cellToProc;
};

// ---- text ---------------------------------------------------------------------------------------------------------------------
// the shortest text a double reads back from exactly
std::string numberText(double v);
// a token as it is written: a brae binary payload token as its number, a string that needs quotes quoted
std::string tokenText(const std::string& tok);
std::string quoted(const std::string& s);
std::string foamHeader(const std::string& cls, const std::string& object, const std::string& location,
                       const std::string& note = std::string());

// ---- the fields ---------------------------------------------------------------------------------------------------------------
// every field file of the case's `time` directory, cut for each processor and written to processor<p>/<time>/
void decomposeFields(const std::string& caseDir, const Plan& plan, const std::string& time, std::vector<std::string>& done,
                     bool quiet);

// the same read back: one field of every processor's `time` directory merged into <case>/<time>/
void reconstructFields(const std::string& caseDir, const std::vector<ProcLayout>& proc, const GlobalLayout& global,
                       const std::string& time, std::vector<std::string>& done, bool quiet);

// ---- files --------------------------------------------------------------------------------------------------------------------
std::vector<label> readLabelListFile(const std::string& path);
void               writeLabelListFile(const std::string& path, const std::string& object, const std::vector<label>& v,
                                      const std::string& note = std::string());

// numeric time directories of a directory, ascending by value
std::vector<std::string> timeDirectories(const std::string& dir);

}   // namespace detail
}   // namespace decompose
}   // namespace brae
