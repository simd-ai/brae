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
    std::vector<label>  candidate;
    label nCells = -1, nPoints = -1, nCandidates = -1;
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
        std::printf("  CONTROL: `error` seeded 0 instead of -1: worst %.4e\n", (double)dz.worst);
        check("...is a different field", dz.worst > scalar(0));
    }

    std::printf("test_refine_candidates_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
