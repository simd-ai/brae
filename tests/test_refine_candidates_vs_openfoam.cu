// brae's refinement candidate selection against REAL OpenFOAM's own, on laminar/damBreakWithObstacle.
//
// WHAT IS UNDER TEST: the four field operations that stand between a volScalarField and the set of
// cells an adaptive mesh would refine -- cellToPoint, error, maxPointField and the composition
// selectRefineCandidates -- plus maxCellField, which is not on the refine path but is the point field
// the unrefinement side is handed, and is the mirror of the same cell-point walk.
//
// THE ORACLE is tools/dumpRefineCandidates, which does NOT transcribe those functions: they are
// protected members of dynamicRefineFvMesh, so it derives a mesh class and re-exports them with
// `using`. Every number it prints comes out of OpenFOAM's own code on OpenFOAM's own pointCells().
//
// THE ORDER IS THE CONTENT, and it is the one thing this gate had to measure rather than assume.
// cellToPoint accumulates in pointCells order, and OpenFOAM's primitiveMesh::calcPointCells has THREE
// orders, chosen by what the mesh has already computed. brae carries two of them, so this runs BOTH
// and says which one reproduces OpenFOAM -- and, where the case cannot tell them apart, says that too
// rather than claiming a match it did not measure.
//
// usage: test_refine_candidates_vs_openfoam <caseDir> <meshDir> <fieldDir> <oracleDump>
#include "primitive_mesh.cuh"
#include "primitive_patch_cpp.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "dynamic_refine_fv_mesh_cpp.cuh"
#include "foam_field_reader.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf("  %s:   %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

struct Dump
{
    std::vector<scalar> cellToPoint, error, maxPointField, maxCellField;
    std::vector<label>  candidate, consistent, selected, protectedCells;
    std::vector<label>  cellLevel, pointLevel, nFacesOfCell;
    std::vector<label>  extended1, extended2;
    label nCells = -1, nPoints = -1, nCandidates = -1;
    label nConsistent = -1, nSelected = -1, nProtectedCells = -1;
    label maxCells = -1, maxRefinement = -1, nTotalCells = -1;
    scalar lower = 0, upper = 0;
    std::string field;
    bool complete = false;
};

// `[brae] <tag> <index> <value>` for the fields, `[brae] <tag> <value>` for the scalars
Dump readDump(const std::string& path)
{
    Dump d;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.rfind("[brae] ", 0) != 0) continue;
        std::istringstream is(line.substr(7));
        std::string tag;
        is >> tag;
        auto indexed = [&](std::vector<scalar>& v)
        {
            label i = 0;
            scalar x = 0;
            is >> i >> x;
            if (static_cast<std::size_t>(i) >= v.size()) v.resize(static_cast<std::size_t>(i) + 1);
            v[static_cast<std::size_t>(i)] = x;
        };
        if (tag == "cellToPoint")        indexed(d.cellToPoint);
        else if (tag == "error")         indexed(d.error);
        else if (tag == "maxPointField") indexed(d.maxPointField);
        else if (tag == "maxCellField")  indexed(d.maxCellField);
        else if (tag == "candidate")     { label c = 0; is >> c; d.candidate.push_back(c); }
        else if (tag == "consistent")    { label c = 0; is >> c; d.consistent.push_back(c); }
        else if (tag == "protected")     { label c = 0; is >> c; d.protectedCells.push_back(c); }
        else if (tag == "extended")
        {
            label n = 0, c = 0;
            is >> n >> c;
            if (n == 1) d.extended1.push_back(c);
            else if (n == 2) d.extended2.push_back(c);
        }
        else if (tag == "pointLevel")
        {
            label i = 0, v = 0;
            is >> i >> v;
            if (static_cast<std::size_t>(i) >= d.pointLevel.size())
                d.pointLevel.resize(static_cast<std::size_t>(i) + 1);
            d.pointLevel[static_cast<std::size_t>(i)] = v;
        }
        else if (tag == "nFacesOfCell")
        {
            label i = 0, v = 0;
            is >> i >> v;
            if (static_cast<std::size_t>(i) >= d.nFacesOfCell.size())
                d.nFacesOfCell.resize(static_cast<std::size_t>(i) + 1);
            d.nFacesOfCell[static_cast<std::size_t>(i)] = v;
        }
        else if (tag == "selected")      { label c = 0; is >> c; d.selected.push_back(c); }
        else if (tag == "cellLevel")
        {
            label i = 0, v = 0;
            is >> i >> v;
            if (static_cast<std::size_t>(i) >= d.cellLevel.size())
                d.cellLevel.resize(static_cast<std::size_t>(i) + 1);
            d.cellLevel[static_cast<std::size_t>(i)] = v;
        }
        else if (tag == "nConsistent")   is >> d.nConsistent;
        else if (tag == "nSelected")     is >> d.nSelected;
        else if (tag == "nProtectedCells") is >> d.nProtectedCells;
        else if (tag == "maxCells")      is >> d.maxCells;
        else if (tag == "maxRefinement") is >> d.maxRefinement;
        else if (tag == "nTotalCells")   is >> d.nTotalCells;
        else if (tag == "nCells")        is >> d.nCells;
        else if (tag == "nPoints")       is >> d.nPoints;
        else if (tag == "nCandidates")   is >> d.nCandidates;
        else if (tag == "lowerRefineLevel") is >> d.lower;
        else if (tag == "upperRefineLevel") is >> d.upper;
        else if (tag == "field")         is >> d.field;
        else if (tag == "END")           d.complete = true;
    }
    return d;
}

struct Diff
{
    scalar worst = 0;
    scalar refMax = 0;
    std::size_t nAbove = 0;
    scalar rel() const { return worst/std::fmax(refMax, scalar(1e-300)); }
};

Diff compare(const std::vector<scalar>& mine, const std::vector<scalar>& of)
{
    Diff d;
    for (std::size_t i = 0; i < mine.size() && i < of.size(); ++i)
    {
        const scalar w = std::fabs(mine[i] - of[i]);
        d.worst = std::fmax(d.worst, w);
        d.refMax = std::fmax(d.refMax, std::fabs(of[i]));
        if (w > scalar(0)) ++d.nAbove;
    }
    return d;
}

// the five quantities, from one pointCells ordering
struct Answer
{
    std::vector<scalar> cellToPoint, error, maxPointField, maxCellField;
    std::vector<char>   candidate;
    std::size_t nCandidates = 0;
};

Answer runAll(
    const std::vector<scalar>& alpha,
    const std::vector<std::vector<label>>& pointCells,
    label nCells,
    scalar lower,
    scalar upper)
{
    Answer a;
    a.cellToPoint = dynamicRefine::cellToPoint(alpha, pointCells);
    a.error = dynamicRefine::error(a.cellToPoint, lower, upper);
    a.maxPointField = dynamicRefine::maxPointField(a.error, pointCells, nCells);
    a.maxCellField = dynamicRefine::maxCellField(alpha, pointCells);
    a.candidate.assign(static_cast<std::size_t>(nCells), 0);
    dynamicRefine::selectRefineCandidates(lower, upper, alpha, pointCells, nCells, a.candidate);
    for (const char c : a.candidate) a.nCandidates += (c != 0);
    return a;
}

}   // namespace


int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::printf("usage: %s <caseDir> <meshDir> <fieldDir> <oracleDump>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string meshDir = argv[2];
    const std::string fieldDir = argv[3];
    const std::string dumpPath = argv[4];
    (void)caseDir;

    std::printf("== brae refine candidates vs OpenFOAM: %s ==\n", dumpPath.c_str());

    const Dump of = readDump(dumpPath);
    check("the oracle ran to completion", of.complete && of.nCells > 0 && of.nPoints > 0);
    if (!of.complete)
    {
        std::printf("test_refine_candidates_vs_openfoam: %d failures\n", ++failures);
        return 1;
    }

    PrimitiveMesh m;
    m.read(meshDir);
    const label nCells = m.nCells();
    const label nPoints = m.nPoints();
    std::printf("  mesh %s: %d cells, %d points; OpenFOAM read %d and %d\n",
                meshDir.c_str(), (int)nCells, (int)nPoints, (int)of.nCells, (int)of.nPoints);
    check("brae read the same mesh OpenFOAM did", nCells == of.nCells && nPoints == of.nPoints);

    const FieldData<scalar> fd = readField<scalar>(fieldDir + "/" + of.field);
    const std::vector<scalar> alpha = fd.internalUniform
        ? std::vector<scalar>(static_cast<std::size_t>(nCells), fd.internalUniformValue)
        : fd.internalField;
    check("the refinement field has one value per cell",
          alpha.size() == static_cast<std::size_t>(nCells));
    std::printf("  field `%s`, band (%.17g %.17g)\n", of.field.c_str(),
                (double)of.lower, (double)of.upper);
    check("the band is the case's own, not a default", of.lower < of.upper);

    // the patch list, for the 2:1 closure's coupled-patch refusal
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    // THE TWO ORDERS. brae carries the pointFaces branch already; the cells branch is new here.
    const std::vector<std::vector<label>> cells = meshCells(m);
    const std::vector<std::vector<label>> pcCells = pointCellsFromCells(m, cells);
    const std::vector<std::vector<label>> pcFaces = pointCellsFromPointFaces(m, meshPointFaces(m));

    // ...and they must at least be the same SETS, or one of them is not pointCells at all
    {
        bool sameSets = (pcCells.size() == pcFaces.size());
        for (std::size_t p = 0; sameSets && p < pcCells.size(); ++p)
        {
            std::vector<label> a = pcCells[p], b = pcFaces[p];
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            sameSets = (a == b);
        }
        check("the two pointCells orders are the same sets, differing only in order", sameSets);
    }

    const Answer aCells = runAll(alpha, pcCells, nCells, of.lower, of.upper);
    const Answer aFaces = runAll(alpha, pcFaces, nCells, of.lower, of.upper);

    const Diff dCells = compare(aCells.cellToPoint, of.cellToPoint);
    const Diff dFaces = compare(aFaces.cellToPoint, of.cellToPoint);
    std::printf("  cellToPoint: cells-order %.4e (%zu of %zu points differ), pointFaces-order %.4e "
                "(%zu differ)\n",
                (double)dCells.worst, dCells.nAbove, of.cellToPoint.size(),
                (double)dFaces.worst, dFaces.nAbove);
    check("the CELLS-branch order reproduces OpenFOAM's cellToPoint exactly", dCells.worst == scalar(0));
    if (dFaces.nAbove == 0)
    {
        std::printf("  (on this field the pointFaces order is bit-identical too -- a 0/1 field sums "
                    "the same in any order, so this arm does NOT discriminate the ordering here)\n");
    }
    else
    {
        check("...and the pointFaces order does not, so the branch is what this measures",
              dFaces.worst > scalar(0));
    }

    // The rest, on the order that matched
    const Diff dErr = compare(aCells.error, of.error);
    const Diff dMax = compare(aCells.maxPointField, of.maxPointField);
    const Diff dMcf = compare(aCells.maxCellField, of.maxCellField);
    std::printf("  error %.4e | maxPointField %.4e | maxCellField %.4e\n",
                (double)dErr.worst, (double)dMax.worst, (double)dMcf.worst);
    check("brae's error field is OpenFOAM's", dErr.worst == scalar(0));
    check("...its maxPointField", dMax.worst == scalar(0));
    check("...and its maxCellField", dMcf.worst == scalar(0));

    // THE ANSWER: the candidate set, as a set
    {
        std::vector<char> ofSet(static_cast<std::size_t>(nCells), 0);
        for (const label c : of.candidate)
        {
            if (c >= 0 && c < nCells) ofSet[static_cast<std::size_t>(c)] = 1;
        }
        std::size_t nDiff = 0;
        for (std::size_t c = 0; c < ofSet.size(); ++c)
        {
            if (ofSet[c] != aCells.candidate[c]) ++nDiff;
        }
        std::printf("  candidates: brae %zu, OpenFOAM %d (%zu cells classified differently)\n",
                    aCells.nCandidates, (int)of.nCandidates, nDiff);
        check("OpenFOAM's candidate count is the one it printed",
              of.candidate.size() == static_cast<std::size_t>(of.nCandidates));
        check("the case actually selects cells, so this is not a comparison of empty sets",
              of.nCandidates > 0);
        check("brae selects exactly OpenFOAM's cells", nDiff == 0);
    }

    // UNIT 2: the 2:1 closure and the selection the solver would act on.
    {
        check("the oracle printed the levels and the budget",
              of.cellLevel.size() == static_cast<std::size_t>(nCells)
           && of.maxCells > 0 && of.maxRefinement > 0 && of.nTotalCells == nCells);
        check("...and its protected-cell count",  of.nProtectedCells >= 0);
        if (of.nProtectedCells != 0)
        {
            std::printf("  (%d protected cells -- this arm DOES exercise the protected path)\n",
                        (int)of.nProtectedCells);
        }
        else
        {
            std::printf("  (no protected cells on this mesh, so the protected-cell path is NOT "
                        "discriminated by this arm)\n");
        }
        // THE PROTECTED SET selectRefineCells is given is `protectedCell_`, as init() computed it --
        // not an empty list. Passing an empty one here made this block pass on every all-hex arm and
        // fail only on the wedge, which is precisely what the wedge fixture exists for.
        const std::vector<char> protectedSet =
            of.pointLevel.empty()
          ? std::vector<char>()
          : dynamicRefine::initProtectedCells(of.cellLevel, of.pointLevel, pcCells, cells, m, patches);

        // the closure on the RAW candidate set, with the budget out of the way
        std::vector<label> candList;
        for (std::size_t c = 0; c < aCells.candidate.size(); ++c)
        {
            if (aCells.candidate[c]) candList.push_back(static_cast<label>(c));
        }
        const std::vector<label> mineConsistent =
            dynamicRefine::consistentRefinement(of.cellLevel, candList, /*maxSet=*/true, m, patches);
        std::printf("  2:1 closure: brae %zu, OpenFOAM %d (from %zu candidates, so it added %d)\n",
                    mineConsistent.size(), (int)of.nConsistent, candList.size(),
                    (int)(of.nConsistent - static_cast<label>(candList.size())));
        check("OpenFOAM's closure list is the length it printed",
              of.consistent.size() == static_cast<std::size_t>(of.nConsistent));
        check("brae's 2:1 closure is OpenFOAM's, cell for cell and in order",
              mineConsistent == of.consistent);

        // ...and the whole selection
        const std::vector<label> mineSelected =
            dynamicRefine::selectRefineCells(of.maxCells, of.maxRefinement, aCells.candidate,
                                             of.cellLevel, protectedSet, of.nTotalCells, m, patches);
        std::printf("  selectRefineCells: brae %zu, OpenFOAM %d (maxCells %d, maxRefinement %d, "
                    "budget %d)\n",
                    mineSelected.size(), (int)of.nSelected, (int)of.maxCells, (int)of.maxRefinement,
                    (int)((of.maxCells - of.nTotalCells)/7));
        check("OpenFOAM's selection is the length it printed",
              of.selected.size() == static_cast<std::size_t>(of.nSelected));
        check("brae selects exactly the cells OpenFOAM would refine", mineSelected == of.selected);

        // CONTROL: the closure with maxSet FALSE must REMOVE where the other ADDS. The two directions
        // are the whole difference between the refinement superset and the unrefinement subset, and a
        // port that ignored the flag would pass the arm above on any already-consistent set.
        const std::vector<label> shrunk =
            dynamicRefine::consistentRefinement(of.cellLevel, candList, /*maxSet=*/false, m, patches);
        std::printf("  CONTROL: the same closure with maxSet false: %zu cells against %zu\n",
                    shrunk.size(), mineConsistent.size());
        check("...never grows the set", shrunk.size() <= candList.size());
        if (mineConsistent.size() != candList.size())
        {
            check("...and is a DIFFERENT set from the maxSet-true one, so the flag is measured here",
                  shrunk != mineConsistent);
        }
        else
        {
            std::printf("  (the candidate set was already 2:1 consistent, so the closure added "
                        "nothing and this arm cannot separate the two directions)\n");
        }

        // CONTROL: the level cap. Raising maxRefinement can only admit more cells, never fewer.
        const std::vector<label> higherCap =
            dynamicRefine::selectRefineCells(of.maxCells, of.maxRefinement + 1, aCells.candidate,
                                             of.cellLevel, protectedSet, of.nTotalCells, m, patches);
        std::printf("  CONTROL: maxRefinement %d instead of %d selects %zu instead of %zu\n",
                    (int)of.maxRefinement + 1, (int)of.maxRefinement, higherCap.size(),
                    mineSelected.size());
        check("...and the cap only ever removes cells", higherCap.size() >= mineSelected.size());
    }

    // UNIT 3: the buffer-layer dilation, and the cells refinement must not touch.
    if (!of.extended1.empty() || !of.extended2.empty())
    {
        // ONE layer, then TWO, each against OpenFOAM's own. The dilation is cell-face-cell, so a port
        // that walked points instead would be right on a structured hex mesh's interior and wrong at
        // every diagonal -- which is most of a refined mesh.
        for (int layers = 1; layers <= 2; ++layers)
        {
            const std::vector<label>& ofExt = (layers == 1) ? of.extended1 : of.extended2;
            if (ofExt.empty()) continue;
            std::vector<char> mine = aCells.candidate;
            for (int i = 0; i < layers; ++i)
            {
                dynamicRefine::extendMarkedCells(m, patches, cells, mine);
            }
            std::vector<label> mineList;
            for (std::size_t c = 0; c < mine.size(); ++c)
            {
                if (mine[c]) mineList.push_back(static_cast<label>(c));
            }
            std::printf("  buffer layers x%d: brae %zu cells, OpenFOAM %zu (from %zu candidates)\n",
                        layers, mineList.size(), ofExt.size(), aCells.nCandidates);
            check(layers == 1 ? "brae's one-layer buffer is OpenFOAM's, cell for cell"
                              : "...and its two-layer buffer",
                  mineList == ofExt);
            const std::size_t before = (layers == 1) ? aCells.nCandidates : of.extended1.size();
            if (before < static_cast<std::size_t>(nCells))
            {
                check(layers == 1 ? "...and it actually grew the set, so the dilation is measured"
                                  : "...and the second layer grew it again",
                      ofExt.size() > before);
            }
            else
            {
                // the `budget` arm makes EVERY cell a candidate, so there is nothing left to dilate
                // into and this arm cannot witness the growth
                std::printf("  (every cell is already marked, so the dilation has nowhere to grow "
                            "and this arm cannot witness it)\n");
            }
        }

        // CONTROL: the dilation must be MONOTONE -- two layers contain one layer contains the
        // candidates. A port that rebuilt the marker instead of extending it would break this while
        // still producing a plausible count.
        if (!of.extended1.empty() && !of.extended2.empty())
        {
            std::vector<char> two(static_cast<std::size_t>(nCells), 0);
            for (const label c : of.extended2) two[static_cast<std::size_t>(c)] = 1;
            bool nested = true;
            for (const label c : of.extended1)
            {
                nested = nested && (two[static_cast<std::size_t>(c)] != 0);
            }
            for (std::size_t c = 0; c < aCells.candidate.size(); ++c)
            {
                nested = nested && (!aCells.candidate[c] || two[c]);
            }
            check("CONTROL: the buffers nest -- candidates inside one layer inside two", nested);
        }
    }

    // THE PROTECTED CELLS: brae's whole init() scan against the set OpenFOAM detected.
    if (!of.pointLevel.empty())
    {
        check("the oracle printed a point level per point",
              of.pointLevel.size() == static_cast<std::size_t>(nPoints));
        const std::vector<char> mine =
            dynamicRefine::initProtectedCells(of.cellLevel, of.pointLevel, pcCells, cells, m, patches);
        std::vector<label> mineList;
        for (std::size_t c = 0; c < mine.size(); ++c)
        {
            if (mine[c]) mineList.push_back(static_cast<label>(c));
        }
        const std::size_t nProt = of.protectedCells.size();
        std::printf("  protected cells: brae %zu, OpenFOAM %zu (it printed %d)\n",
                    mineList.size(), nProt, (int)of.nProtectedCells);
        check("OpenFOAM's protected list is the length it printed",
              of.nProtectedCells < 0 || nProt == static_cast<std::size_t>(of.nProtectedCells));
        check("brae's init scan finds exactly OpenFOAM's protected cells", mineList == of.protectedCells);
        // ...and the SENTINEL: OpenFOAM clears the marker to size ZERO when nothing is protected, and
        // four other sites read that size rather than the bits. A port that returned a zeroed array
        // of the mesh's size would take the wrong branch in all four.
        check("...and an empty result is EMPTY, not a zeroed array of the mesh's size",
              nProt != 0 || mine.empty());

        if (nProt == 0)
        {
            std::printf("  (every cell here is a hex with eight anchor points and six quad faces, so "
                        "the protected path is NOT discriminated by this arm)\n");
        }
        else
        {
            // the 2:1 cascade selectRefineCells applies internally
            const std::vector<char> cascade =
                dynamicRefine::calculateProtectedCells(mine, of.cellLevel, m, patches);
            std::size_t nCascade = 0;
            for (const char c : cascade) nCascade += (c != 0);
            std::printf("  ...and the 2:1 cascade of them reaches %zu cells\n", nCascade);
            check("the cascade never shrinks the protected set", nCascade >= nProt);

            // CONTROL: the `less than hex` pass alone. On a mesh whose only non-hexes are prisms it
            // is what finds them, and dropping it must lose cells -- if it does not, this fixture is
            // protecting cells for some other reason and the arm is not measuring what it says.
            std::size_t nFewFaces = 0, nManyFaces = 0;
            for (std::size_t c = 0; c < cells.size(); ++c)
            {
                if (cells[c].size() < 6) ++nFewFaces;
                else if (cells[c].size() > 6) ++nManyFaces;
            }
            std::printf("  CONTROL: %zu cells have fewer than six faces, %zu have more\n",
                        nFewFaces, nManyFaces);
            // The two fixtures protect cells for DIFFERENT reasons and the gate says which: a prism
            // trips the `< 6 faces` pass, a split hex trips the >4-anchor FACE pass and the under-8
            // anchor count. If neither kind of cell is present the protected set came from somewhere
            // this arm is not describing.
            check("...and the mesh carries the non-hex cells that explain it",
                  nFewFaces > 0 || nManyFaces > 0);
        }
    }

    // CONTROL 1: the band edge. `error` writes an exact edge as 0 (`>= 0`) and selectRefineCandidates
    // rejects it (`> 0`), so the band is OPEN. A port that wrote `>= 0` at the second test would
    // select every cell whose points sit exactly on an edge -- and on a VoF field, where whole
    // regions are exactly 0 and exactly 1, that is most of the mesh.
    {
        std::vector<char> mark(static_cast<std::size_t>(nCells), 0);
        dynamicRefine::selectRefineCandidates(scalar(0), scalar(1), alpha, pcCells, nCells, mark);
        std::size_t n = 0;
        for (const char c : mark) n += (c != 0);
        std::printf("  CONTROL: the band widened to [0, 1] -- the VoF field's own extremes -- selects "
                    "%zu cells of %d\n", n, (int)nCells);
        check("...and it is the interface cells, not the whole mesh: the band is OPEN at both ends",
              n < static_cast<std::size_t>(nCells));
        check("...and not empty either", n > 0);
    }

    // CONTROL 2: the average is an average. Dropping the divisor is the one mistake cellToPoint's
    // shape invites, and on a 0/1 field it multiplies the interface points by their cell count.
    {
        std::vector<scalar> unscaled(static_cast<std::size_t>(nPoints), scalar(0));
        for (std::size_t p = 0; p < pcCells.size(); ++p)
        {
            scalar sum = 0;
            for (const label c : pcCells[p]) sum += alpha[static_cast<std::size_t>(c)];
            unscaled[p] = sum;
        }
        const Diff du = compare(unscaled, of.cellToPoint);
        std::printf("  CONTROL: the cell-to-point sum WITHOUT its divisor: worst %.4e of %.4e\n",
                    (double)du.worst, (double)du.refMax);
        check("...is a different field, so the divisor is measured here", du.worst > scalar(0.5));
    }

    // CONTROL 3: the sentinels. `error` seeds -1 and the max fields seed -GREAT; a port that seeded
    // either with 0 would leave a cell outside the band looking like a cell exactly on the edge.
    {
        const std::vector<scalar> errZero = [&]
        {
            std::vector<scalar> e = aCells.error;
            for (scalar& x : e) { if (x < scalar(0)) x = scalar(0); }
            return e;
        }();
        const Diff dz = compare(errZero, of.error);
        std::size_t nOutside = 0;
        for (const scalar x : aCells.error) nOutside += (x < scalar(0));
        std::printf("  CONTROL: `error` seeded 0 instead of -1: worst %.4e (%zu of %zu points lie "
                    "outside the band)\n", (double)dz.worst, nOutside, aCells.error.size());
        if (nOutside > 0)
        {
            check("...is a different field", dz.worst > scalar(0));
        }
        else
        {
            // the `budget` arm widens the band to [-1, 2], so every point is inside it and the
            // sentinel is never written -- the control has nothing to change and says so
            std::printf("  (every point is inside the band on this arm, so the sentinel is never "
                        "written and this control cannot witness it here)\n");
        }
    }

    std::printf("test_refine_candidates_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
