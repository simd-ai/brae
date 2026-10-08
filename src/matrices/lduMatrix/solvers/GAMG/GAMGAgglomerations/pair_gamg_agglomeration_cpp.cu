#include "pair_gamg_agglomeration_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

// GAMGAgglomeration.C:248
constexpr label maxLevels = 50;
// the GREAT of OpenFOAM's scalar.H
constexpr scalar great = 1.0e+15;

// GAMGAgglomeration::continueAgglomerating, serial
bool continueAgglomerating(
    label nCellsInCoarsestLevel,
    label nFineCells,
    label nCoarseCells)
{
    if (nCoarseCells < nCellsInCoarsestLevel) return false;
    return nCoarseCells < nFineCells;
}

// GAMGAgglomeration::agglomerateLduAddressing without its interface half: the coarse level's
// addressing, the fine-to-coarse face map and the flip map, appended to `a`.
void agglomerateLduAddressing(
    GamgAgglomeration& a,
    label fineLevelIndex)
{
    const GamgLduAddressing& fineMeshAddr = a.meshLevel(fineLevelIndex);
    const std::vector<label>& upperAddr = fineMeshAddr.upperAddr;
    const std::vector<label>& lowerAddr = fineMeshAddr.lowerAddr;
    const std::size_t nFineFaces = upperAddr.size();
    const std::vector<label>& restrictMap = a.restrictAddressing[static_cast<std::size_t>(fineLevelIndex)];
    const label nCoarseCells = a.nCells[static_cast<std::size_t>(fineLevelIndex)];

    // Guess initial maximum number of neighbours in coarse cell
    std::size_t maxNnbrs = 10;
    // Number of faces for each coarse-cell
    std::vector<label> cCellnFaces(static_cast<std::size_t>(nCoarseCells), 0);
    // packed storage for coarse-cell faces
    std::vector<label> cCellFaces(maxNnbrs*static_cast<std::size_t>(nCoarseCells));
    std::vector<label> faceRestrictAddr(nFineFaces);
    // Initial neighbour array (not in upper-triangle order)
    std::vector<label> initCoarseNeighb(nFineFaces);
    label nCoarseFaces = 0;

    for (std::size_t fineFacei = 0; fineFacei < nFineFaces; ++fineFacei)
    {
        const label rmUpperAddr = restrictMap[static_cast<std::size_t>(upperAddr[fineFacei])];
        const label rmLowerAddr = restrictMap[static_cast<std::size_t>(lowerAddr[fineFacei])];
        if (rmUpperAddr == rmLowerAddr)
        {
            // inside a coarse cell: keep the cell's address as a negative index
            faceRestrictAddr[fineFacei] = -(rmUpperAddr + 1);
            continue;
        }

        const label cOwn = std::min(rmUpperAddr, rmLowerAddr);
        const label cNei = std::max(rmUpperAddr, rmLowerAddr);

        // check the neighbour to see if this face has already been found
        bool nbrFound = false;
        label& ccnFaces = cCellnFaces[static_cast<std::size_t>(cOwn)];
        for (label i = 0; i < ccnFaces; ++i)
        {
            const label cf = cCellFaces[maxNnbrs*static_cast<std::size_t>(cOwn) + static_cast<std::size_t>(i)];
            if (initCoarseNeighb[static_cast<std::size_t>(cf)] == cNei)
            {
                nbrFound = true;
                faceRestrictAddr[fineFacei] = cf;
                break;
            }
        }
        if (nbrFound) continue;

        if (static_cast<std::size_t>(ccnFaces) >= maxNnbrs)
        {
            const std::size_t oldMaxNnbrs = maxNnbrs;
            maxNnbrs *= 2;
            cCellFaces.resize(maxNnbrs*static_cast<std::size_t>(nCoarseCells));
            for (std::size_t i = static_cast<std::size_t>(nCoarseCells); i-- > 0;)
            {
                for (label j = 0; j < cCellnFaces[i]; ++j)
                {
                    cCellFaces[maxNnbrs*i + static_cast<std::size_t>(j)] =
                        cCellFaces[oldMaxNnbrs*i + static_cast<std::size_t>(j)];
                }
            }
        }

        cCellFaces[maxNnbrs*static_cast<std::size_t>(cOwn) + static_cast<std::size_t>(ccnFaces)] = nCoarseFaces;
        initCoarseNeighb[static_cast<std::size_t>(nCoarseFaces)] = cNei;
        faceRestrictAddr[fineFacei] = nCoarseFaces;
        ++ccnFaces;
        ++nCoarseFaces;
    }

    // Renumber into upper-triangular order: owner-major, each owner's faces as the walk met them
    GamgLduAddressing coarse;
    coarse.nCells = nCoarseCells;
    coarse.lowerAddr.resize(static_cast<std::size_t>(nCoarseFaces));
    coarse.upperAddr.resize(static_cast<std::size_t>(nCoarseFaces));
    std::vector<label> coarseFaceMap(static_cast<std::size_t>(nCoarseFaces));
    label coarseFacei = 0;
    for (std::size_t cci = 0; cci < cCellnFaces.size(); ++cci)
    {
        for (label i = 0; i < cCellnFaces[cci]; ++i)
        {
            const label cf = cCellFaces[maxNnbrs*cci + static_cast<std::size_t>(i)];
            coarse.lowerAddr[static_cast<std::size_t>(coarseFacei)] = static_cast<label>(cci);
            coarse.upperAddr[static_cast<std::size_t>(coarseFacei)] = initCoarseNeighb[static_cast<std::size_t>(cf)];
            coarseFaceMap[static_cast<std::size_t>(cf)] = coarseFacei;
            ++coarseFacei;
        }
    }
    for (std::size_t fineFacei = 0; fineFacei < nFineFaces; ++fineFacei)
    {
        if (faceRestrictAddr[fineFacei] >= 0)
        {
            faceRestrictAddr[fineFacei] = coarseFaceMap[static_cast<std::size_t>(faceRestrictAddr[fineFacei])];
        }
    }

    // face-flip status: true where the fine face's UPPER cell fell in the coarse OWNER
    std::vector<char> faceFlipMap(nFineFaces, 0);
    for (std::size_t fineFacei = 0; fineFacei < nFineFaces; ++fineFacei)
    {
        const label cFace = faceRestrictAddr[fineFacei];
        if (cFace < 0) continue;
        const label cOwn = coarse.lowerAddr[static_cast<std::size_t>(cFace)];
        const label cNei = coarse.upperAddr[static_cast<std::size_t>(cFace)];
        const label rmUpperAddr = restrictMap[static_cast<std::size_t>(upperAddr[fineFacei])];
        const label rmLowerAddr = restrictMap[static_cast<std::size_t>(lowerAddr[fineFacei])];
        if (cOwn == rmUpperAddr && cNei == rmLowerAddr)
        {
            faceFlipMap[fineFacei] = 1;
        }
        else if (!(cOwn == rmLowerAddr && cNei == rmUpperAddr))
        {
            throw std::runtime_error(
                "brae GAMG agglomeration: fine face " + std::to_string(fineFacei) + " of level " +
                std::to_string(fineLevelIndex) + " maps to a coarse face that joins neither of its cells' "
                "clusters. GAMGAgglomerateLduAddressing.C:218 stops on the same condition.");
        }
    }

    a.nFaces.push_back(nCoarseFaces);
    a.faceRestrictAddressing.push_back(std::move(faceRestrictAddr));
    a.faceFlipMap.push_back(std::move(faceFlipMap));
    a.meshLevels.push_back(std::move(coarse));
}

} // namespace

std::vector<label> gamgPairAgglomerate(
    label& nCoarseCells,
    const GamgLduAddressing& fine,
    const std::vector<scalar>& faceWeights,
    bool& forward)
{
    const label nFineCells = fine.nCells;
    const std::vector<label>& upperAddr = fine.upperAddr;
    const std::vector<label>& lowerAddr = fine.lowerAddr;
    const std::size_t nFaces = upperAddr.size();

    // For each cell its faces: those it is the UPPER cell of first, then those it owns, each in
    // face order. The order decides ties, which a strict `>` below resolves to the first met.
    std::vector<label> cellFaces(2*nFaces);
    std::vector<label> cellFaceOffsets(static_cast<std::size_t>(nFineCells) + 1);
    {
        std::vector<label> nNbrs(static_cast<std::size_t>(nFineCells), 0);
        for (std::size_t facei = 0; facei < nFaces; ++facei)
        {
            ++nNbrs[static_cast<std::size_t>(upperAddr[facei])];
        }
        for (std::size_t facei = 0; facei < nFaces; ++facei)
        {
            ++nNbrs[static_cast<std::size_t>(lowerAddr[facei])];
        }
        cellFaceOffsets[0] = 0;
        for (std::size_t celli = 0; celli < nNbrs.size(); ++celli)
        {
            cellFaceOffsets[celli + 1] = cellFaceOffsets[celli] + nNbrs[celli];
        }
        std::fill(nNbrs.begin(), nNbrs.end(), 0);
        for (std::size_t facei = 0; facei < nFaces; ++facei)
        {
            const std::size_t c = static_cast<std::size_t>(upperAddr[facei]);
            cellFaces[static_cast<std::size_t>(cellFaceOffsets[c] + nNbrs[c])] = static_cast<label>(facei);
            ++nNbrs[c];
        }
        for (std::size_t facei = 0; facei < nFaces; ++facei)
        {
            const std::size_t c = static_cast<std::size_t>(lowerAddr[facei]);
            cellFaces[static_cast<std::size_t>(cellFaceOffsets[c] + nNbrs[c])] = static_cast<label>(facei);
            ++nNbrs[c];
        }
    }

    // go through the faces and create clusters
    std::vector<label> coarseCellMap(static_cast<std::size_t>(nFineCells), -1);
    nCoarseCells = 0;
    for (label cellfi = 0; cellfi < nFineCells; ++cellfi)
    {
        // Change cell ordering depending on direction for this level
        const std::size_t celli = static_cast<std::size_t>(forward ? cellfi : nFineCells - cellfi - 1);
        if (coarseCellMap[celli] >= 0) continue;

        label matchFaceNo = -1;
        scalar maxFaceWeight = -great;
        // find the ungrouped neighbour with the largest face weight
        for (label faceOs = cellFaceOffsets[celli]; faceOs < cellFaceOffsets[celli + 1]; ++faceOs)
        {
            const std::size_t facei = static_cast<std::size_t>(cellFaces[static_cast<std::size_t>(faceOs)]);
            if
            (
                coarseCellMap[static_cast<std::size_t>(upperAddr[facei])] < 0
             && coarseCellMap[static_cast<std::size_t>(lowerAddr[facei])] < 0
             && faceWeights[facei] > maxFaceWeight
            )
            {
                matchFaceNo = static_cast<label>(facei);
                maxFaceWeight = faceWeights[facei];
            }
        }

        if (matchFaceNo >= 0)
        {
            // Make a new group
            coarseCellMap[static_cast<std::size_t>(upperAddr[static_cast<std::size_t>(matchFaceNo)])] = nCoarseCells;
            coarseCellMap[static_cast<std::size_t>(lowerAddr[static_cast<std::size_t>(matchFaceNo)])] = nCoarseCells;
            ++nCoarseCells;
            continue;
        }

        // No match. Find the best neighbouring cluster and put the cell there
        label clusterMatchFaceNo = -1;
        scalar clusterMaxFaceCoeff = -great;
        for (label faceOs = cellFaceOffsets[celli]; faceOs < cellFaceOffsets[celli + 1]; ++faceOs)
        {
            const std::size_t facei = static_cast<std::size_t>(cellFaces[static_cast<std::size_t>(faceOs)]);
            if (faceWeights[facei] > clusterMaxFaceCoeff)
            {
                clusterMatchFaceNo = static_cast<label>(facei);
                clusterMaxFaceCoeff = faceWeights[facei];
            }
        }
        if (clusterMatchFaceNo >= 0)
        {
            const std::size_t mf = static_cast<std::size_t>(clusterMatchFaceNo);
            coarseCellMap[celli] = std::max(
                coarseCellMap[static_cast<std::size_t>(upperAddr[mf])],
                coarseCellMap[static_cast<std::size_t>(lowerAddr[mf])]);
        }
    }

    // cells that are part of no cluster become single-cell clusters
    for (label cellfi = 0; cellfi < nFineCells; ++cellfi)
    {
        const std::size_t celli = static_cast<std::size_t>(forward ? cellfi : nFineCells - cellfi - 1);
        if (coarseCellMap[celli] < 0)
        {
            coarseCellMap[celli] = nCoarseCells;
            ++nCoarseCells;
        }
    }

    if (!forward)
    {
        --nCoarseCells;
        for (label& c : coarseCellMap)
        {
            c = nCoarseCells - c;
        }
        ++nCoarseCells;
    }

    // Reverse the map ordering for the next level
    forward = !forward;
    return coarseCellMap;
}

GamgAgglomeration faceAreaPairGamgAgglomeration(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    label nCellsInCoarsestLevel,
    bool& forward)
{
    const std::size_t nIf = static_cast<std::size_t>(m.nInternalFaces());
    GamgAgglomeration a;
    a.fineMesh.nCells = m.nCells();
    a.fineMesh.lowerAddr.assign(m.owner().begin(), m.owner().begin() + static_cast<std::ptrdiff_t>(nIf));
    a.fineMesh.upperAddr.assign(m.neighbour().begin(), m.neighbour().begin() + static_cast<std::ptrdiff_t>(nIf));

    // GAMGAgglomeration.C:285
    nCellsInCoarsestLevel = std::max<label>(1, std::min<label>(m.nCells()/2, nCellsInCoarsestLevel));

    // mag(cmptMultiply(Sf/sqrt(magSf), vector(1, 1.01, 1.02)))
    std::vector<scalar> faceWeights(nIf);
    for (std::size_t f = 0; f < nIf; ++f)
    {
        const scalar rootMagSf = std::sqrt(g.magSf()[f]);
        const vector s = g.Sf()[f];
        const scalar wx = s.x/rootMagSf;
        const scalar wy = (s.y/rootMagSf)*1.01;
        const scalar wz = (s.z/rootMagSf)*1.02;
        faceWeights[f] = std::sqrt(wx*wx + wy*wy + wz*wz);
    }

    label nCreatedLevels = 0;
    while (nCreatedLevels < maxLevels - 1)
    {
        const GamgLduAddressing& fineMesh = a.meshLevel(nCreatedLevels);
        label nCoarseCells = -1;
        std::vector<label> finalAgglom = gamgPairAgglomerate(nCoarseCells, fineMesh, faceWeights, forward);
        if (!continueAgglomerating(nCellsInCoarsestLevel, static_cast<label>(finalAgglom.size()), nCoarseCells))
        {
            break;
        }
        a.nCells.push_back(nCoarseCells);
        a.restrictAddressing.push_back(std::move(finalAgglom));

        agglomerateLduAddressing(a, nCreatedLevels);

        // restrictFaceField: the next level's weights are the sums over the merged faces
        const std::vector<label>& fineToCoarse = a.faceRestrictAddressing[static_cast<std::size_t>(nCreatedLevels)];
        std::vector<scalar> aggFaceWeights(static_cast<std::size_t>(a.nFaces[static_cast<std::size_t>(nCreatedLevels)]), 0.0);
        for (std::size_t ffacei = 0; ffacei < fineToCoarse.size(); ++ffacei)
        {
            const label cFace = fineToCoarse[ffacei];
            if (cFace >= 0)
            {
                aggFaceWeights[static_cast<std::size_t>(cFace)] += faceWeights[ffacei];
            }
        }
        faceWeights = std::move(aggFaceWeights);

        // mergeLevels is 1, so `nPairLevels % mergeLevels_` is never set and no level is combined
        ++nCreatedLevels;
    }
    return a;
}

std::pair<label, scalar> gamgLduBand(const GamgLduAddressing& addr)
{
    std::vector<label> cellBandwidth(static_cast<std::size_t>(addr.nCells), 0);
    for (std::size_t facei = 0; facei < addr.upperAddr.size(); ++facei)
    {
        const label nei = addr.upperAddr[facei];
        const label diff = nei - addr.lowerAddr[facei];
        cellBandwidth[static_cast<std::size_t>(nei)] = std::max(cellBandwidth[static_cast<std::size_t>(nei)], diff);
    }
    label bandwidth = 0;
    scalar profile = 0.0;
    for (const label b : cellBandwidth)
    {
        bandwidth = std::max(bandwidth, b);
        profile += 1.0*b;
    }
    return {bandwidth, profile};
}

} // namespace brae
