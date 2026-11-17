#pragma once
// The per-rank text dump of a coupled processor patch's geometry, in the format tools/dumpInterFoam writes from
// OpenFOAM (procGeometryDump.H) so tools/mgpu_oracle/compare_proc_geometry.py can lay the two side by side:
//
//   rank <r> nCells <n> nProcPatches <k>
//   patch <name> myProcNo <a> neighbProcNo <b> size <s> start <st>
//   f <i> <cell> <nbrCell> <slot> <w> <dc> <nodc> <magSf> <Sfx Sfy Sfz> <dx dy dz> <cx cy cz>
//
// cell = my owner cell, nbrCell = the NEIGHBOUR's local owner cell of the face (its own cell numbering; OpenFOAM
// learns it with processorFvPatch::internalFieldTransfer, brae with one exchangeIntLists), slot = brae's halo slot
// of that cell (-1 in OpenFOAM's file), w/dc/nodc = weights, deltaCoeffs, nonOrthDeltaCoeffs, delta = fvPatch::delta(),
// c = nonOrthCorrectionVectors. Every double is %.17g, which round-trips.
#include "processor_coupling.cuh"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// COLLECTIVE (one integer exchange): every rank of the communicator must call it.
inline void writeProcessorGeometryDump(
    const std::string& path,
    const std::vector<FvPatch>& patches,
    const FvGeometry& g,
    const ProcessorCoupling& pc,
    ProcComm& comm)
{
    std::vector<int> nbrRanks;
    std::vector<std::vector<int>> mine;
    for (const label pi : pc.patches)
    {
        const FvPatch& p = patches[static_cast<std::size_t>(pi)];
        nbrRanks.push_back(p.neighbProcNo);
        mine.emplace_back(p.faceCells.begin(), p.faceCells.end());
    }
    const std::vector<std::vector<int>> theirs = comm.exchangeIntLists(nbrRanks, mine);

    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("brae: cannot write " + path);
    std::fprintf(f, "rank %d nCells %ld nProcPatches %zu\n", comm.rank(), static_cast<long>(pc.nCells), pc.patches.size());
    for (std::size_t k = 0; k < pc.patches.size(); ++k)
    {
        const FvPatch& p = patches[static_cast<std::size_t>(pc.patches[k])];
        std::fprintf(f, "patch %s myProcNo %d neighbProcNo %d size %ld start %ld\n", p.name.c_str(), (int)p.myProcNo,
                     (int)p.neighbProcNo, static_cast<long>(p.size), static_cast<long>(p.start));
        for (label i = 0; i < p.size; ++i)
        {
            const std::size_t u = static_cast<std::size_t>(i);
            const label gf = p.start + i;
            const vector& sf = g.Sf()[gf];
            std::fprintf(f, "f %ld %ld %d %ld %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                         static_cast<long>(i), static_cast<long>(p.faceCells[u]), theirs[k][u],
                         static_cast<long>(p.nbrFaceCells[u] - pc.nCells), p.weights[u], p.deltaCoeffs[u],
                         p.nonOrthDeltaCoeffs[u], g.magSf()[gf], sf.x, sf.y, sf.z, p.delta[u].x, p.delta[u].y,
                         p.delta[u].z, p.nonOrthCorrectionVectors[u].x, p.nonOrthCorrectionVectors[u].y,
                         p.nonOrthCorrectionVectors[u].z);
        }
    }
    std::fclose(f);
}

} // namespace brae
