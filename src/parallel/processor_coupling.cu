#include "processor_coupling.cuh"

#include <set>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

ProcessorCoupling attach(std::vector<FvPatch>& patches, const FvGeometry& g, label nCells, ProcComm& comm)
{
    ProcessorCoupling out;
    out.nCells = nCells;

    // the processor patches with faces, in patch order
    std::vector<label> idx;
    label firstFace = -1;
    label end = -1;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        if (p.type != "processor")
        {
            if (firstFace >= 0 && p.size > 0)
            {
                throw std::runtime_error(
                    "brae: patch '" + p.name + "' (" + p.type + ") has faces after the processor patches. "
                    "decomposePar writes the processor patches last; the multi-GPU interface needs the "
                    "processor faces to be the last faces of the mesh.");
            }
            continue;
        }
        if (p.size == 0) continue;
        if (p.myProcNo != comm.rank())
        {
            throw std::runtime_error(
                "brae: processor patch '" + p.name + "' is of rank " + std::to_string(p.myProcNo)
                + " but this is rank " + std::to_string(comm.rank()) + ".");
        }
        if (p.neighbProcNo < 0 || p.neighbProcNo >= comm.size() || p.neighbProcNo == comm.rank())
        {
            throw std::runtime_error(
                "brae: processor patch '" + p.name + "' joins rank " + std::to_string(p.myProcNo) + " to rank "
                + std::to_string(p.neighbProcNo) + ", outside a communicator of " + std::to_string(comm.size()) + ".");
        }
        if (firstFace < 0)
        {
            firstFace = p.start;
        }
        else if (p.start != end)
        {
            throw std::runtime_error("brae: processor patch '" + p.name + "' does not follow the previous one.");
        }
        end = p.start + p.size;
        idx.push_back(static_cast<label>(pi));
    }
    if (firstFace >= 0 && end != static_cast<label>(g.Cf().size()))
    {
        throw std::runtime_error("brae: the processor patches do not end at the last face of the mesh.");
    }
    {
        std::set<label> seen;
        for (const label pi : idx)
        {
            if (!seen.insert(patches[static_cast<std::size_t>(pi)].neighbProcNo).second)
            {
                throw std::runtime_error(
                    "brae: two processor patches join rank " + std::to_string(comm.rank()) + " to rank "
                    + std::to_string(patches[static_cast<std::size_t>(pi)].neighbProcNo)
                    + " (a processorCyclic cut? put preservePatches in decomposeParDict).");
            }
        }
    }

    // 1. the neighbours' face data, once
    std::vector<int> nbrRanks;
    std::vector<std::vector<double>> send;
    std::vector<ProcessorPatchDesc> descs;
    for (const label pi : idx)
    {
        const FvPatch& p = patches[static_cast<std::size_t>(pi)];
        nbrRanks.push_back(p.neighbProcNo);
        send.push_back(packProcessorFaces(p, g));
        ProcessorPatchDesc d;
        d.nbrRank = p.neighbProcNo;
        d.faceCells.assign(p.faceCells.begin(), p.faceCells.end());
        descs.push_back(std::move(d));
    }
    const std::vector<std::vector<double>> recv = comm.exchangeFaceData(nbrRanks, send);

    // 2. the plan (fetches the neighbours' face cells itself)
    out.plan = comm.buildPlanFromPatches(nCells, std::move(descs));
    const HaloPlanDesc& pd = out.plan->desc();
    out.nHalo = pd.nHalo;
    out.nProcFaces = pd.nProcFaces;
    out.firstProcFace = firstFace < 0 ? static_cast<label>(g.Cf().size()) : firstFace;
    out.patches = idx;

    // 3. the coupled half of every patch
    for (std::size_t k = 0; k < idx.size(); ++k)
    {
        FvPatch& p = patches[static_cast<std::size_t>(idx[k])];
        const HaloNeighbour& nb = pd.nbrs[k];
        if (p.neighbProcNo != nb.rank || nb.faceCount != p.size)
        {
            throw std::runtime_error("brae: the halo plan's neighbour " + std::to_string(k) + " is not processor patch '"
                                     + p.name + "'.");
        }
        coupleProcessorPair(p, unpackProcessorFaces(recv[k], p.size), g);
        p.haloBase = nCells + nb.recvStart;
        p.nHaloCells = nb.recvCount;
        p.nbrFaceCells.resize(static_cast<std::size_t>(p.size));
        for (label i = 0; i < p.size; ++i)
        {
            p.nbrFaceCells[static_cast<std::size_t>(i)] = nCells + pd.faceToSlot[static_cast<std::size_t>(nb.faceStart + i)];
        }
    }
    return out;
}

} // namespace

ProcessorCoupling attachProcessorCoupling(
    std::vector<FvPatch>& patches,
    const FvGeometry& g,
    label nCells,
    ProcComm& comm)
{
    ProcessorCoupling out;
    comm.guarded([&] { out = attach(patches, g, nCells, comm); });
    return out;
}

} // namespace brae
