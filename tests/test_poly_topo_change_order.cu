// polyTopoChange's face ordering against REAL OpenFOAM's, with no instrumentation and no oracle file.
//
// THE ORACLE IS THE MESH ITSELF. Any mesh OpenFOAM has written is already in the order getFaceOrder
// produces, so feeding its own owner/neighbour/patch-id back in must return the IDENTITY permutation,
// with patchStarts equal to `boundary`'s startFace and patchSizes equal to its nFaces. That is a
// stronger oracle than a dump: it cannot drift, and it exists for every mesh in the tree.
//
// THE MESH IS RUN THROUGH TWICE: once as written, and once after a topology change, because the two
// exercise different halves. A blockMesh mesh has its cells in a lattice order that makes the
// within-cell neighbour sort nearly monotone; a refined mesh does not, and it is the refined one that
// the producing half will actually be asked to reproduce.
//
// WHAT THIS DOES NOT COVER, and says so rather than implying otherwise: the cell ordering
// (getCellOrder/makeCellCells, the Cuthill-McKee half), the point compaction, the flip rule, zones,
// coupled-patch reordering, and the mapPolyMesh construction. This is the face numbering alone.
#include "primitive_mesh.cuh"
#include "poly_topo_change_cpp.cuh"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::polyTopoChange;

namespace {
int failures = 0;

void check(
    const char* what,
    bool        ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

// The polyTopoChange state a NO-OP round trip starts from: the mesh's own addressing, every cell and
// face active, each boundary face carrying its patch id.
OrderInput inputFromMesh(
    const PrimitiveMesh& m)
{
    OrderInput in;
    in.cellMapSize = m.nCells();
    in.faceOwner = m.owner();
    in.faceNeighbour.assign(static_cast<std::size_t>(m.nFaces()), label(-1));
    for (label f = 0; f < m.nInternalFaces(); ++f)
    {
        in.faceNeighbour[static_cast<std::size_t>(f)] = m.neighbour()[static_cast<std::size_t>(f)];
    }
    in.region.assign(static_cast<std::size_t>(m.nFaces()), label(-1));
    in.nPatches = static_cast<label>(m.patches().size());
    for (std::size_t pi = 0; pi < m.patches().size(); ++pi)
    {
        const auto& p = m.patches()[pi];
        for (label i = 0; i < p.size; ++i)
        {
            in.region[static_cast<std::size_t>(p.start + i)] = static_cast<label>(pi);
        }
    }
    return in;
}

bool isIdentity(
    const std::vector<label>& v)
{
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (v[i] != static_cast<label>(i)) return false;
    }
    return true;
}

// one arm: build the input, order it, and hold the result against the mesh's own numbering
void arm(
    const std::string&   caseDir,
    const std::string&   label_)
{
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    std::printf("== %s: %ld cells, %ld faces (%ld internal), %zu patches ==\n",
                label_.c_str(), (long)m.nCells(), (long)m.nFaces(),
                (long)m.nInternalFaces(), m.patches().size());

    const OrderInput in = inputFromMesh(m);
    const label nActiveFaces = m.nFaces();

    std::vector<label> cellFaces, cellFaceOffsets;
    makeCells(in, nActiveFaces, cellFaces, cellFaceOffsets);

    // makeCells' own shape, before the ordering reads it
    check((label_ + ": every face appears once as an owner and internal faces twice").c_str(),
          static_cast<label>(cellFaces.size()) == m.nFaces() + m.nInternalFaces());
    check((label_ + ": the offsets are a prefix sum ending at the list length").c_str(),
          static_cast<label>(cellFaceOffsets.size()) == m.nCells() + 1
          && cellFaceOffsets[0] == 0
          && cellFaceOffsets[static_cast<std::size_t>(m.nCells())] == static_cast<label>(cellFaces.size()));

    std::vector<label> oldToNew, patchSizes, patchStarts;
    getFaceOrder(in, nActiveFaces, cellFaces, cellFaceOffsets, oldToNew, patchSizes, patchStarts);

    // THE ORACLE
    check((label_ + ": the face order OpenFOAM's own mesh is already in is the IDENTITY").c_str(),
          isIdentity(oldToNew));
    if (!isIdentity(oldToNew))
    {
        std::size_t nBad = 0, firstBad = 0;
        for (std::size_t i = 0; i < oldToNew.size(); ++i)
        {
            if (oldToNew[i] != static_cast<label>(i))
            {
                if (nBad == 0) firstBad = i;
                ++nBad;
            }
        }
        std::printf("    %zu of %zu faces moved; first at %zu (%ld -> %ld), which is %s\n",
                    nBad, oldToNew.size(), firstBad, (long)firstBad,
                    (long)oldToNew[firstBad],
                    firstBad < static_cast<std::size_t>(m.nInternalFaces()) ? "internal" : "a boundary face");
    }

    // ...and the patch layout against `boundary`'s own startFace / nFaces
    bool startsOk = (patchStarts.size() == m.patches().size());
    bool sizesOk  = startsOk;
    for (std::size_t pi = 0; pi < m.patches().size() && startsOk; ++pi)
    {
        if (patchStarts[pi] != m.patches()[pi].start) startsOk = false;
        if (patchSizes[pi]  != m.patches()[pi].size)  sizesOk  = false;
    }
    check((label_ + ": patchStarts are `boundary`'s own startFace values").c_str(), startsOk);
    check((label_ + ": patchSizes are `boundary`'s own nFaces values").c_str(), sizesOk);
    check((label_ + ": the first patch starts at nInternalFaces").c_str(),
          !patchStarts.empty() && patchStarts[0] == m.nInternalFaces());

    // the arm must not be vacuous: a mesh with no internal faces would pass the identity trivially
    check((label_ + ": the mesh has internal faces, so the ordering half is exercised").c_str(),
          m.nInternalFaces() > 0);
    check((label_ + ": the mesh has boundary faces, so the patch half is exercised").c_str(),
          m.nFaces() > m.nInternalFaces());
}
}   // namespace

// UNIT 2: the compaction, on the path dynamicRefineFvMesh takes (orderCells == false,
// orderPoints == false). Two arms, because the two halves of AMR take different paths through it.
TopoState stateFromMesh(
    const PrimitiveMesh& m)
{
    TopoState s;
    s.points.assign(m.points().begin(), m.points().end());
    s.faces.resize(static_cast<std::size_t>(m.nFaces()));
    for (label f = 0; f < m.nFaces(); ++f)
    {
        const label k = m.faceSize(f);
        s.faces[static_cast<std::size_t>(f)].resize(static_cast<std::size_t>(k));
        for (label v = 0; v < k; ++v) s.faces[static_cast<std::size_t>(f)][static_cast<std::size_t>(v)] = m.faceVert(f, v);
    }
    const OrderInput in = inputFromMesh(m);
    s.faceOwner     = in.faceOwner;
    s.faceNeighbour = in.faceNeighbour;
    s.region        = in.region;
    s.nPatches      = in.nPatches;
    s.cellMap.resize(static_cast<std::size_t>(m.nCells()));
    std::iota(s.cellMap.begin(), s.cellMap.end(), label(0));
    s.pointMap.resize(static_cast<std::size_t>(m.nPoints()));
    std::iota(s.pointMap.begin(), s.pointMap.end(), label(0));
    s.faceMap.resize(static_cast<std::size_t>(m.nFaces()));
    std::iota(s.faceMap.begin(), s.faceMap.end(), label(0));
    s.flipFaceFlux.assign(static_cast<std::size_t>(m.nFaces()), char(0));
    return s;
}

void compactArm(
    const std::string& caseDir,
    const std::string& label_)
{
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");

    // ---- the NO-OP round trip: nothing removed, so every map is the identity and the flip is skipped
    {
        TopoState s = stateFromMesh(m);
        const std::vector<label> ownBefore = s.faceOwner;
        const std::vector<label> neiBefore = s.faceNeighbour;
        const CompactResult r = compactNoOrder(s);

        check((label_ + " [no-op]: the point map is the identity").c_str(),
              isIdentity(r.localPointMap) && r.nActivePoints == m.nPoints());
        check((label_ + " [no-op]: the face map is the identity").c_str(),
              isIdentity(r.localFaceMap) && r.nActiveFaces == m.nFaces());
        check((label_ + " [no-op]: the cell map is the identity").c_str(),
              isIdentity(r.localCellMap) && r.nActiveCells == m.nCells());
        check((label_ + " [no-op]: nInternalPoints is -1, as orderPoints == false leaves it").c_str(),
              r.nInternalPoints == -1);
        // polyTopoChange.C:1222 -- with no cell removed the renumber, and so the flip, never runs
        check((label_ + " [no-op]: no cell went, so the renumber and the flip are SKIPPED").c_str(),
              !r.cellsRenumbered && r.nFlipped == 0);
        check((label_ + " [no-op]: owner and neighbour are untouched").c_str(),
              s.faceOwner == ownBefore && s.faceNeighbour == neiBefore);
    }

    // ---- ONE CELL REMOVED: the renumber runs, and it must flip exactly the faces whose neighbour
    //      became the lower cell. This is the unrefinement half's path.
    {
        TopoState s = stateFromMesh(m);
        const label gone = m.nCells()/2;             // a cell in the middle, so both sides exist
        s.cellMap[static_cast<std::size_t>(gone)] = -2;
        const std::vector<label> ownBefore = s.faceOwner;
        const std::vector<label> neiBefore = s.faceNeighbour;
        const CompactResult r = compactNoOrder(s);

        check((label_ + " [one cell removed]: the cell map compacts past it").c_str(),
              r.nActiveCells == m.nCells() - 1
              && r.localCellMap[static_cast<std::size_t>(gone)] == -1);
        check((label_ + " [one cell removed]: the renumber RAN").c_str(), r.cellsRenumbered);
        check((label_ + " [one cell removed]: the point and face maps are still identities").c_str(),
              isIdentity(r.localPointMap) && isIdentity(r.localFaceMap));

        // every internal face must end owner < neighbour, and a face flipped exactly when the
        // renumbering put its neighbour below its owner
        label expectedFlips = 0, badOrder = 0;
        for (label f = 0; f < m.nInternalFaces(); ++f)
        {
            const label o = ownBefore[static_cast<std::size_t>(f)];
            const label n = neiBefore[static_cast<std::size_t>(f)];
            if (o == gone || n == gone) continue;    // its faces are the removed cell's business
            const label no = r.localCellMap[static_cast<std::size_t>(o)];
            const label nn = r.localCellMap[static_cast<std::size_t>(n)];
            if (nn < no) ++expectedFlips;
            const label ao = s.faceOwner[static_cast<std::size_t>(f)];
            const label an = s.faceNeighbour[static_cast<std::size_t>(f)];
            if (ao >= 0 && an >= 0 && an < ao) ++badOrder;
        }
        std::printf("    %s: %ld cells -> %ld, flips %ld, faces needing one %ld\n",
                    label_.c_str(), (long)m.nCells(), (long)r.nActiveCells,
                    (long)r.nFlipped, (long)expectedFlips);
        check((label_ + " [one cell removed]: every internal face still has owner < neighbour").c_str(),
              badOrder == 0);
        // THE FLIP IS UNREACHABLE ON THIS PATH, and asserting `nFlipped >= 0` would have hidden that.
        // With orderCells == false the cell map is MONOTONE -- a compaction only closes gaps, it never
        // reorders a pair -- so localCellMap[own] < localCellMap[nei] whenever own < nei, and
        // polyTopoChange.C:1258-1261's `faceNeighbour_ < faceOwner_` is never true. The flip block
        // exists for the orderCells == TRUE path, where Cuthill-McKee reorders arbitrarily, and
        // dynamicRefineFvMesh never takes it (changeMesh(*this, false), defaults).
        // So this asserts the fact rather than testing the branch: NO face may need a flip, and none
        // may get one. The flip itself is therefore NOT COVERED here, and cannot be on the AMR path.
        check((label_ + " [one cell removed]: a monotone compaction inverts NO pair, so none needs a flip").c_str(),
              expectedFlips == 0);
        check((label_ + " [one cell removed]: and none is applied").c_str(), r.nFlipped == 0);
    }
}

int main(
    int argc,
    char** argv)
{
    std::printf("== brae polyTopoChange face ordering vs OpenFOAM's own mesh numbering ==\n");
    if (argc < 2)
    {
        std::printf("  SKIP: usage: %s <caseDir> [<refinedCaseDir>]\n", argv[0]);
        return 77;
    }
    arm(argv[1], "as written");
    if (argc > 2) arm(argv[2], "after a topology change");
    compactArm(argv[1], "as written");
    if (argc > 2) compactArm(argv[2], "after a topology change");

    std::printf("test_poly_topo_change_order: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
