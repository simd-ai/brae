// brae's hexRef8::setRefinement against REAL OpenFOAM's -- unit 5b of the dynamicRefineFvMesh port.
//
// THE ORACLE is tools/dumpHexRef8, which constructs OpenFOAM's own hexRef8 on the mesh, runs its 2:1
// closure on a requested cell set, calls setRefinement with the result and dumps everything that is
// PUBLIC afterwards. That is more than it looks: setRefinement RETURNS cellAddedCells, and
// cellLevel()/pointLevel() are accessors -- so sections 1-8 and 10 of setRefinement can be held against
// OpenFOAM WITHOUT instrumenting anything and WITHOUT the face machinery existing.
//
// SO THIS GATE HAS TWO STAGES, and today it runs the first:
//   5b-1  cellAddedCells, cellLevel and pointLevel after setRefinement, and the count of points and
//         cells the split added. No face work is needed for any of it.
//   5b-2  the mapPolyMesh and the mesh, which cannot be right until section 9 exists. Those comparisons
//         are SKIPPED here by name, printed as such, and are what unit 5b-2 turns on.
// A skipped comparison that says why is not the same as one nobody wrote; the count of skips is printed
// so it cannot quietly become zero-coverage.
#include "primitive_mesh.cuh"
#include "hex_ref8_cpp.cuh"
#include "mesh_cell_cells_cpp.cuh"
#include "remove_faces_cpp.cuh"
#include "mesh_edges_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include "poly_topo_change_cpp.cuh"
#include "fv_geometry.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu;
using namespace brae::cpu::hexRef8;

namespace {

int failures = 0;
int skipped = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

void skip(const char* what)
{
    std::printf("  SKIP: %s\n", what);
    ++skipped;
}

struct Dump
{
    std::map<std::string, label> scalars;
    std::map<std::string, std::vector<label>> lists;
    std::map<std::string, std::vector<std::vector<label>>> listLists;
};

Dump readDump(const std::string& path)
{
    Dump d;
    std::ifstream is(path);
    if (!is) { std::printf("  FAIL: cannot open %s\n", path.c_str()); ++failures; return d; }
    std::string line;
    while (std::getline(is, line))
    {
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        label n = 0;
        if (!(ls >> n)) continue;
        std::string tail;
        std::getline(ls, tail);
        const bool nothingAfterN = (tail.find_first_not_of(" \t\r") == std::string::npos);
        // A SCALAR IS `name value` AND NOTHING ELSE. The name test alone was a trap: removeFaces'
        // `nFacesPerEdge` is a LIST of 14,386 entries whose name begins with n and a capital, and the
        // name rule swallowed it as the scalar 14386 -- silently, because the comparison that wanted the
        // list then said "the dump has no nFacesPerEdge". So the shape decides first and the name only
        // separates a scalar from a one-line BLOCK, which have the same shape.
        if (nothingAfterN
         && ((key.rfind('n', 0) == 0 && key.size() > 1
           && std::isupper(static_cast<unsigned char>(key[1])))
          || key == "historyActive"))
        {
            d.scalars[key] = n;
            continue;
        }
        // BLOCK OR FLAT LIST, decided by the dump's own SHAPE and not by a list of key names. The tool
        // writes a flat list as `name N v0 v1 ...` on one line and a block as `name N` followed by N
        // lines. So: nothing left on the line after N means a block.
        //
        // IT USED TO BE A LIST OF NAMES, and that cost two rounds of the same bug: a block whose key was
        // not in the list was read as a flat list and silently dropped, and the comparison that wanted it
        // then reported "the dump has no ..." or, worse, a size mismatch that looked like a port defect.
        // `historyAddedCellsAfterSet` was the second one. A structural test cannot be forgotten.
        const bool isBlock = nothingAfterN;
        if (!isBlock)
        {
            std::vector<label> v(static_cast<std::size_t>(n));
            std::istringstream vs(line);
            std::string k2; label n2 = 0;
            vs >> k2 >> n2;
            for (label i = 0; i < n; ++i) vs >> v[static_cast<std::size_t>(i)];
            d.lists[key] = v;
            continue;
        }
        // ONE GENUINE AMBIGUITY: a FLAT list with zero entries (`flipFaceFlux 0`) is written exactly like
        // a BLOCK with zero entries, and no structural test can separate them. So a zero-length one is
        // stored as BOTH -- an empty flat list and an empty block -- which is correct for either reader.
        // Measured: `flipFaceFlux` is empty on every refinement arm, and treating it as a block alone made
        // all four report "the dump has no flipFaceFlux".
        if (n == 0)
        {
            d.lists[key] = std::vector<label>();
            d.listLists[key] = std::vector<std::vector<label>>();
            continue;
        }
        // an objectMap block is `index k v...` per line; every one of their names carries `From`
        const bool isObjectMap = (key.find("From") != std::string::npos);
        std::vector<std::vector<label>> v;
        v.reserve(static_cast<std::size_t>(n));
        for (label i = 0; i < n; ++i)
        {
            if (!std::getline(is, line)) break;
            std::istringstream rs(line);
            if (isObjectMap)
            {
                label idx = 0, k = 0;
                rs >> idx >> k;
                std::vector<label> e;
                e.reserve(static_cast<std::size_t>(k) + 1);
                e.push_back(idx);
                for (label j = 0; j < k; ++j) { label q = 0; rs >> q; e.push_back(q); }
                v.push_back(e);
            }
            else if (key == "patches")
            {
                std::string nm; label st = 0, sz = 0;
                rs >> nm >> st >> sz;
                v.push_back({st, sz});
            }
            else if (key == "points")
            {
                // three scalars per line; not compared by this harness, so the line is consumed and
                // dropped rather than parsed into labels
                v.push_back({});
            }
            else
            {
                label k = 0;
                rs >> k;
                std::vector<label> e(static_cast<std::size_t>(k));
                for (label j = 0; j < k; ++j) rs >> e[static_cast<std::size_t>(j)];
                v.push_back(e);
            }
        }
        d.listLists[key] = v;
    }
    return d;
}

void compareList(const char* name, const std::vector<label>& mine, const Dump& d, const char* key)
{
    const auto it = d.lists.find(key);
    if (it == d.lists.end()) { std::printf("  FAIL: the dump has no `%s`\n", key); ++failures; return; }
    if (mine.size() != it->second.size())
    {
        std::printf("  FAIL: %s has %zu entries, OpenFOAM %zu\n", name, mine.size(), it->second.size());
        ++failures;
        return;
    }
    std::size_t firstBad = mine.size();
    for (std::size_t i = 0; i < mine.size(); ++i)
    {
        if (mine[i] != it->second[i]) { firstBad = i; break; }
    }
    if (firstBad < mine.size())
    {
        std::printf("  FAIL: %s differs first at %zu (brae %d, OpenFOAM %d)\n", name, firstBad,
                    (int)mine[firstBad], (int)it->second[firstBad]);
        ++failures;
    }
    else
    {
        std::printf("  ok:   %s is OpenFOAM's (%zu entries)\n", name, mine.size());
    }
}

}   // namespace


void compareListList(
    const char* name,
    const std::vector<std::vector<label>>& mine,
    const Dump& d,
    const char* key)
{
    const auto it = d.listLists.find(key);
    if (it == d.listLists.end()) { std::printf("  FAIL: the dump has no `%s`\n", key); ++failures; return; }
    if (mine.size() != it->second.size())
    {
        std::printf("  FAIL: %s has %zu entries, OpenFOAM %zu\n", name, mine.size(), it->second.size());
        ++failures;
        return;
    }
    for (std::size_t i = 0; i < mine.size(); ++i)
    {
        if (mine[i] != it->second[i])
        {
            std::printf("  FAIL: %s differs at %zu (brae %zu entries, OpenFOAM %zu)\n", name, i,
                        mine[i].size(), it->second[i].size());
            ++failures;
            return;
        }
    }
    std::printf("  ok:   %s is OpenFOAM's (%zu entries)\n", name, mine.size());
}

int main(int argc, char** argv)
{
    std::printf("== brae hexRef8::setRefinement vs OpenFOAM's ==\n");
    if (argc < 4)
    {
        std::printf("  SKIP: usage: %s <caseDir> <cellsFile> <dump>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string cellsPath = argv[2];
    const std::string dumpPath = argv[3];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);

    // the cell set OpenFOAM refined, read from the file it wrote -- so both codes refine the same one
    std::vector<label> cellsToRefine;
    {
        std::ifstream is(cellsPath);
        std::string all((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
        // OpenFOAM writes a labelList as `N ( a b c )` or `N{v}` for a uniform one
        std::size_t lp = all.find('(');
        if (lp != std::string::npos)
        {
            std::istringstream ls(all.substr(lp + 1));
            label v = 0;
            while (ls >> v) cellsToRefine.push_back(v);
        }
    }
    const Dump d = readDump(dumpPath);
    // THE UNREFINE ARM is a different dump and a different comparison: it removes cells, so the reverse
    // maps stop being the identity and unit 6's remapping half is witnessed for the first time. Only the
    // pieces brae has are compared; removeFaces (6b-2 and 6b-3) is what turns the rest on.
    const bool unrefine = (d.lists.count("splitPoints") != 0);
    if (unrefine)
    {
        std::printf("  UNREFINE arm: %zu split points\n", d.lists.at("splitPoints").size());
        PrimitiveMesh um;
        um.read(caseDir + "/constant/polyMesh");
        const MeshEdges ume = buildMeshEdges(um);
        const std::vector<std::vector<label>> ucells = meshCells(um);
        const std::vector<std::vector<label>> uptCells = pointCellsFromCells(um, ucells);
        const std::vector<std::vector<label>> ufEdges = buildFaceEdges(um, ume);
        MeshView uv;
        uv.m = &um; uv.edges = &ume; uv.faceEdges = &ufEdges;
        const std::vector<std::vector<label>> ueFaces = buildEdgeFaces(um, ufEdges);
        const std::vector<std::vector<label>> ucellPts = cellPointsFromCells(um, ucells);
        const std::vector<std::vector<label>> ucellEdg = buildCellEdges(ucells, ufEdges);
        FvGeometry ug;
        ug.build(um);
        uv.edgeFaces = &ueFaces; uv.cells = &ucells; uv.cellPoints = &ucellPts;
        uv.pointCells = &uptCells; uv.cellEdges = &ucellEdg;
        uv.cellCentres = &ug.C(); uv.faceCentres = &ug.Cf();

        check("the pre-unrefinement mesh is the one OpenFOAM unrefined",
              um.nCells() == d.scalars.at("nOldCells") && um.nFaces() == d.scalars.at("nOldFaces"));
        Levels ulv;
        ulv.cellLevel = d.lists.at("preCellLevel");
        ulv.pointLevel = d.lists.at("prePointLevel");
        History uh;
        uh.visibleCells = d.lists.at("preHistoryVisibleCells");
        uh.parent = d.lists.at("preHistoryParent");
        uh.addedCells = d.listLists.at("preHistoryAddedCells");
        uh.active = !uh.visibleCells.empty();
        check("...and its levels and history are OpenFOAM's, handed over",
              (label)ulv.cellLevel.size() == um.nCells() && uh.active);

        setUnrefinementLevels(uv, ulv, uh, d.lists.at("splitPoints"));
        compareList("cellLevel after setUnrefinement", ulv.cellLevel, d, "cellLevelAfterSet");
        compareList("pointLevel after setUnrefinement (untouched)", ulv.pointLevel, d, "pointLevelAfterSet");
        compareList("the history's visibleCells after setUnrefinement", uh.visibleCells, d,
                    "historyVisibleCellsAfterSet");
        compareList("the history's parent list after setUnrefinement", uh.parent, d,
                    "historyParentAfterSet");
        {
            const auto it = d.listLists.find("historyAddedCellsAfterSet");
            bool ok = (it != d.listLists.end()) && (uh.addedCells.size() == it->second.size());
            std::size_t firstBad = uh.addedCells.size();
            for (std::size_t i = 0; ok && i < uh.addedCells.size(); ++i)
            {
                if (uh.addedCells[i] != it->second[i]) { ok = false; firstBad = i; }
            }
            if (!ok && firstBad < uh.addedCells.size())
            {
                std::printf("  FAIL: the history's addedCells after setUnrefinement differs at split "
                            "cell %zu (brae %zu entries)\n", firstBad, uh.addedCells[firstBad].size());
                ++failures;
            }
            else
            {
                check("the history's addedCells after setUnrefinement is OpenFOAM's", ok);
            }
        }
        // UNIT 6's REMAPPING HALF, witnessed at last: these maps are NOT the identity here
        {
            const std::vector<label>& rcm = d.lists.at("reverseCellMap");
            const std::vector<label>& rpm = d.lists.at("reversePointMap");
            bool cIdent = true, pIdent = true;
            for (std::size_t i = 0; i < rcm.size(); ++i) if (rcm[i] != (label)i) cIdent = false;
            for (std::size_t i = 0; i < rpm.size(); ++i) if (rpm[i] != (label)i) pIdent = false;
            check("the reverse maps are NOT the identity here -- which is the whole point of this arm",
                  !cIdent && !pIdent);
            Levels uafter = ulv;
            History uh2 = uh;
            updateLevels(uafter, rcm, rpm, d.lists.at("cellMap"), d.lists.at("pointMap"),
                         d.scalars.at("nCells"), d.scalars.at("nPoints"));
            historyUpdateMesh(uh2, rcm, d.scalars.at("nCells"));
            compareList("cellLevel after changeMesh (the remap, now witnessed)", uafter.cellLevel, d,
                        "cellLevelFinal");
            compareList("pointLevel after changeMesh (the remap, now witnessed)", uafter.pointLevel, d,
                        "pointLevelFinal");
            compareList("the history's visibleCells after changeMesh", uh2.visibleCells, d,
                        "historyVisibleCells");
        }
        // UNIT 6b-2: removeFaces::compatibleRemoves, on OpenFOAM's own face set and on a reduced one.
        // The face list is handed over in THE ORDER OpenFOAM passed it (labelHashSet::toc), because
        // region 0 is the region the first face created -- see the tool's own note. brae has no
        // labelHashSet and the numbering is what is being compared, so the order is data, not a detail.
        {
            const std::vector<std::vector<label>> ucc = buildCellCells(um);
            std::vector<label> cr, crm, ftr;
            const label nUsed = removeFaces::compatibleRemoves(
                um, ucc, d.lists.at("splitFacesToc"), cr, crm, ftr);
            compareList("cellRegion", cr, d, "cellRegion");
            compareList("cellRegionMaster", crm, d, "cellRegionMaster");
            compareList("facesToRemove", ftr, d, "facesToRemove");
            check("compatibleRemoves returns OpenFOAM's used-region count",
                  nUsed == d.scalars.at("nUsedRegions"));
            // ...AND THE REDUCED SET, which is the only arm where the recount does anything: on
            // hexRef8's own set the twelve faces at a split point ARE its block's twelve internal
            // faces, so the walk over the internal faces returns the input unchanged (measured:
            // the same LIST, not merely the same size, on all three arms). With the lowest-numbered
            // face of each block dropped the region is unchanged and the recount must put it back.
            std::vector<label> dcr, dcrm, dftr;
            const label nUsedD = removeFaces::compatibleRemoves(
                um, ucc, d.lists.at("splitFacesDropped"), dcr, dcrm, dftr);
            compareList("cellRegion on the reduced set", dcr, d, "cellRegionDropped");
            compareList("cellRegionMaster on the reduced set", dcrm, d, "cellRegionMasterDropped");
            compareList("facesToRemove on the reduced set -- THE RECOUNT", dftr, d,
                        "facesToRemoveDropped");
            check("...and the recount really added faces, so this comparison is not the input",
                  dftr.size() > d.lists.at("splitFacesDropped").size());
            check("compatibleRemoves returns OpenFOAM's used-region count on the reduced set",
                  nUsedD == d.scalars.at("nUsedRegionsDropped"));
        }
        // UNIT 6b-3a: removeFaces::setRefinement's DECISIONS -- which edges go, which faces merge with
        // which, which points go, which faces are touched. Every one is a LOCAL of OpenFOAM's function,
        // so the oracle is a copy of its own class with writes added (tools/dumpHexRef8/
        // removeFacesDump.C), called on the same three inputs hexRef8 hands it -- the ones just compared
        // above. Run TWICE, because minCos decides whether a whole branch executes:
        //   GREAT      what hexRef8 constructs its own faceRemover with (hexRef8.C:1967), where
        //              `minCos_ < 1` is false and the feature-angle guard never runs
        //   cos(45deg) a configuration hexRef8 never asks for, but removeFaces has other callers, so the
        //              guard is transcribed and this arm is what holds it against OpenFOAM
        {
            const std::vector<std::vector<label>> upointFaces = meshPointFaces(um);
            removeFaces::RemoveFacesView rv;
            rv.m = &um;
            rv.edges = &ume;
            rv.faceEdges = &ufEdges;
            rv.edgeFaces = &ueFaces;
            rv.cells = &ucells;
            rv.pointFaces = &upointFaces;
            rv.faceAreas = &ug.Sf();
            const std::vector<label>& ftr = d.lists.at("facesToRemove");
            const std::vector<label>& crg = d.lists.at("cellRegion");
            const std::vector<label>& crm2 = d.lists.at("cellRegionMaster");
            // OpenFOAM's GREAT (1e15), which is hexRef8's own minCos_
            const std::pair<const char*, scalar> profiles[] = {
                {".removeFaces",   scalar(1e15)},
                {".removeFaces45", std::cos(scalar(45.0)*scalar(M_PI)/scalar(180.0))},
            };
            for (const auto& prof : profiles)
            {
                const Dump rf = readDump(dumpPath + prof.first);
                if (rf.lists.find("nFacesPerEdge") == rf.lists.end())
                {
                    std::printf("  FAIL: %s has no nFacesPerEdge -- the oracle did not write it\n",
                                prof.first);
                    ++failures;
                    continue;
                }
                const removeFaces::RemoveFacesDecisions dec =
                    removeFaces::setRefinementDecisions(rv, ftr, crg, crm2, prof.second);
                std::printf("  -- removeFaces::setRefinement's decisions at minCos %g (%s)\n",
                            (double)prof.second, prof.first);
                compareList("nFacesPerEdge", dec.nFacesPerEdge, rf, "nFacesPerEdge");
                compareList("edgesToRemove", dec.edgesToRemove, rf, "edgesToRemove");
                compareList("faceRegion", dec.faceRegion, rf, "faceRegion");
                check("the number of face regions is OpenFOAM's",
                      dec.nFaceRegions == rf.scalars.at("nFaceRegions"));
                compareList("pointsToRemove", dec.pointsToRemove, rf, "pointsToRemove");
                std::vector<label> aff(dec.affectedFace.size());
                for (std::size_t i = 0; i < aff.size(); ++i) aff[i] = dec.affectedFace[i] ? 1 : 0;
                compareList("affectedFace", aff, rf, "affectedFace");
                compareListList("regionToFaces", dec.regionToFaces, rf, "regionToFaces");
            }
        }
        skip("removeFaces::setRefinement -- the actions, the map and the mesh, unit 6b-3b");
        std::printf("test_hex_ref8_vs_openfoam: %d failures, %d skipped\n", failures, skipped);
        return failures == 0 ? 0 : 1;
    }
    std::printf("  refining %zu cells\n", cellsToRefine.size());
    check("the cell set was read and is not empty", !cellsToRefine.empty());
    check("...and it is the set OpenFOAM refined",
          d.lists.count("cellsToRefine") && d.lists.at("cellsToRefine") == cellsToRefine);
    check("the dump carries what this stage compares",
          d.lists.count("cellLevelAfterSet") && d.lists.count("pointLevelAfterSet")
       && d.listLists.count("cellAddedCells"));

    // ---- brae's own setRefinement, sections 1-8 and 10 -------------------------------------------
    const MeshEdges me = buildMeshEdges(m);
    const std::vector<std::vector<label>> cells = meshCells(m);
    const std::vector<std::vector<label>> cellPts = cellPointsFromCells(m, cells);
    const std::vector<std::vector<label>> ptCells = pointCellsFromCells(m, cells);
    const std::vector<std::vector<label>> fEdges = buildFaceEdges(m, me);
    const std::vector<std::vector<label>> eFaces = buildEdgeFaces(m, fEdges);
    const std::vector<std::vector<label>> cellEdg = buildCellEdges(cells, fEdges);

    Levels lv;
    // THE LEVELS ARE THE DUMP'S, not assumed zero. A mesh straight out of blockMesh has every level 0 --
    // and at level 0 every one of setRefinement's level formulas collapses to the same number, so a
    // first refinement CANNOT witness any of them (measured: three fail-proofs on those very lines all
    // stayed green). The `twice` arm hands over the mesh and the levels as they stood after one
    // refinement, and there the arithmetic is live.
    if (d.lists.count("preCellLevel") && d.lists.count("prePointLevel"))
    {
        lv.cellLevel = d.lists.at("preCellLevel");
        lv.pointLevel = d.lists.at("prePointLevel");
    }
    else
    {
        lv.cellLevel.assign(static_cast<std::size_t>(m.nCells()), label(0));
        lv.pointLevel.assign(m.points().size(), label(0));
    }
    check("the levels are the mesh's size",
          (label)lv.cellLevel.size() == m.nCells() && lv.pointLevel.size() == m.points().size());
    {
        const label maxC = *std::max_element(lv.cellLevel.begin(), lv.cellLevel.end());
        std::printf("  input levels: max cellLevel %d\n", (int)maxC);
    }

    MeshView v;
    v.m = &m;
    v.edges = &me;
    v.cells = &cells;
    v.cellPoints = &cellPts;
    v.pointCells = &ptCells;
    v.cellEdges = &cellEdg;
    v.faceEdges = &fEdges;
    v.edgeFaces = &eFaces;
    v.cellCentres = &g.C();
    v.faceCentres = &g.Cf();

    std::vector<label> starts, sizes;
    std::vector<std::string> types;
    for (const auto& p : m.patches()) { starts.push_back(p.start); sizes.push_back(p.size); types.push_back(p.type); }
    std::vector<std::vector<label>> faces(static_cast<std::size_t>(m.nFaces()));
    for (label f = 0; f < m.nFaces(); ++f)
    {
        faces[static_cast<std::size_t>(f)].assign(m.faceVerts().begin() + m.faceOffsets()[f],
                                                 m.faceVerts().begin() + m.faceOffsets()[f + 1]);
    }
    std::vector<label> nbr(static_cast<std::size_t>(m.nFaces()), label(-1));
    for (label f = 0; f < m.nInternalFaces(); ++f) nbr[static_cast<std::size_t>(f)] = m.neighbour()[f];

    polyTopoChange::TopoActions a;
    polyTopoChange::addMesh(a, m.points(), faces, m.owner(), nbr, m.nCells(), starts, sizes);
    const std::size_t pointsBefore = a.state.points.size();
    const std::size_t cellsBefore = a.state.cellMap.size();

    RefinementMarks marks;
    const std::vector<std::vector<label>> refinedCells =
        setRefinementPointsAndCells(v, lv, cellsToRefine, types, a, marks);

    const std::size_t pointsAdded = a.state.points.size() - pointsBefore;
    const std::size_t cellsAdded = a.state.cellMap.size() - cellsBefore;
    std::printf("  brae added %zu points and %zu cells\n", pointsAdded, cellsAdded);
    // 19 points per split hex (1 centre + 6 face mids + 12 edge mids) only when no neighbour shares
    // them; the counts against OpenFOAM's are what is actually held
    check("the point count is OpenFOAM's",
          (label)(pointsBefore + pointsAdded) == d.scalars.at("nPoints"));
    check("the cell count is OpenFOAM's",
          (label)(cellsBefore + cellsAdded) == d.scalars.at("nCells"));

    // cellAddedCells, per REQUESTED cell, with the original at element 0
    {
        const auto it = d.listLists.find("cellAddedCells");
        bool ok = (it != d.listLists.end()) && (refinedCells.size() == it->second.size());
        std::size_t firstBad = refinedCells.size();
        for (std::size_t i = 0; ok && i < refinedCells.size(); ++i)
        {
            if (refinedCells[i] != it->second[i]) { ok = false; firstBad = i; }
        }
        if (!ok && firstBad < refinedCells.size())
        {
            std::printf("  FAIL: cellAddedCells differs at requested cell %zu (brae %zu entries)\n",
                        firstBad, refinedCells[firstBad].size());
            ++failures;
        }
        else
        {
            check("cellAddedCells is OpenFOAM's, cell for cell and in OpenFOAM's own order", ok);
        }
    }

    compareList("cellLevel after setRefinement", marks.newCellLevel, d, "cellLevelAfterSet");
    compareList("pointLevel after setRefinement", marks.newPointLevel, d, "pointLevelAfterSet");

    // ---- SECTION 9: THE FACES, and then the mesh and the map (unit 5b-2) --------------------------
    setRefinementFaces(v, lv, marks, a);
    std::printf("  after the faces: %zu faces accumulated\n", a.state.faces.size());

    polyTopoChange::ChangeMeshInput ci;
    ci.nOldPoints = (label)m.points().size();
    ci.nOldFaces = m.nFaces();
    ci.nOldCells = m.nCells();
    ci.oldPatchStarts = starts;
    ci.oldPatchSizes = sizes;
    ci.oldPatchNMeshPoints.assign(starts.size(), label(0));
    ci.patchTypes = types;
    ci.nZones = 0;
    polyTopoChange::ChangedMesh out;
    polyTopoChange::TopoChangeMap map;
    polyTopoChange::changeMesh(a, ci, out, map);
    std::printf("  brae's mesh: %zu points, %zu faces (%d internal), %d cells\n",
                out.points.size(), out.faces.size(), (int)out.nInternalFaces, (int)out.nCells);

    check("the mesh has OpenFOAM's face and internal-face counts",
          (label)out.faces.size() == d.scalars.at("nFaces")
       && out.nInternalFaces == d.scalars.at("nInternalFaces"));
    compareList("owner", out.faceOwner, d, "owner");
    compareList("neighbour", out.faceNeighbour, d, "neighbour");
    compareList("pointMap", map.pointMap, d, "pointMap");
    compareList("faceMap", map.faceMap, d, "faceMap");
    compareList("cellMap", map.cellMap, d, "cellMap");
    compareList("reversePointMap", map.reversePointMap, d, "reversePointMap");
    compareList("reverseFaceMap", map.reverseFaceMap, d, "reverseFaceMap");
    compareList("reverseCellMap", map.reverseCellMap, d, "reverseCellMap");
    compareList("flipFaceFlux", map.flipFaceFlux, d, "flipFaceFlux");
    {
        const auto it = d.listLists.find("faces");
        bool ok = (it != d.listLists.end()) && (out.faces.size() == it->second.size());
        std::size_t firstBad = out.faces.size();
        for (std::size_t i = 0; ok && i < out.faces.size(); ++i)
        {
            if (out.faces[i] != it->second[i]) { ok = false; firstBad = i; }
        }
        if (!ok && firstBad < out.faces.size())
        {
            std::printf("  FAIL: face %zu differs -- brae %zu vertices, OpenFOAM %zu\n",
                        firstBad, out.faces[firstBad].size(), it->second[firstBad].size());
            ++failures;
        }
        else
        {
            check("every face's vertex list is OpenFOAM's", ok);
        }
    }
    {
        const auto it = d.listLists.find("patches");
        bool ok = (it != d.listLists.end()) && (out.patchStarts.size() == it->second.size());
        for (std::size_t i = 0; ok && i < out.patchStarts.size(); ++i)
        {
            ok = (out.patchStarts[i] == it->second[i][0]) && (out.patchSizes[i] == it->second[i][1]);
        }
        check("the patch slicing is OpenFOAM's", ok);
    }
    // ---- UNIT 6: hexRef8::updateMesh and the refinement history --------------------------------
    // the history is built BEFORE updateMesh remaps the levels, because section 11 runs inside
    // setRefinement and this harness calls the two stages in OpenFOAM's own order
    // THE HISTORY IS THE DUMP'S when the dump carries one -- for the same reason the levels are: after an
    // earlier refinement it is NOT the fresh identity, and brae cannot reconstruct it from the mesh. The
    // `twice` arm caught this: starting fresh there gave 9340 split cells where OpenFOAM has 9436, short
    // by exactly the 96 cells pass 1 had refined.
    History h;
    if (d.lists.count("preHistoryVisibleCells") && d.lists.count("preHistoryParent"))
    {
        h.visibleCells = d.lists.at("preHistoryVisibleCells");
        h.parent = d.lists.at("preHistoryParent");
        h.addedCells = d.listLists.count("preHistoryAddedCells")
                     ? d.listLists.at("preHistoryAddedCells")
                     : std::vector<std::vector<label>>(h.parent.size());
        h.active = !h.visibleCells.empty();
    }
    else
    {
        h = freshHistory(m.nCells());
    }
    check("the history is active -- as OpenFOAM's is, even on a mesh never refined",
          h.active && d.scalars.count("historyActive") && d.scalars.at("historyActive") == 1);
    storeRefinementHistory(h, marks.cellAddedCells, (label)marks.newCellLevel.size());

    Levels after;
    after.cellLevel = marks.newCellLevel;
    after.pointLevel = marks.newPointLevel;
    updateLevels(after, map.reverseCellMap, map.reversePointMap, map.cellMap, map.pointMap,
                 out.nCells, (label)out.points.size());
    historyUpdateMesh(h, map.reverseCellMap, out.nCells);

    compareList("cellLevel after changeMesh", after.cellLevel, d, "cellLevelFinal");
    compareList("pointLevel after changeMesh", after.pointLevel, d, "pointLevelFinal");
    compareList("the history's visibleCells", h.visibleCells, d, "historyVisibleCells");
    compareList("the history's parent list", h.parent, d, "historyParent");
    {
        const auto it = d.listLists.find("historyAddedCells");
        bool ok = (it != d.listLists.end()) && (h.addedCells.size() == it->second.size());
        std::size_t firstBad = h.addedCells.size();
        for (std::size_t i = 0; ok && i < h.addedCells.size(); ++i)
        {
            if (h.addedCells[i] != it->second[i]) { ok = false; firstBad = i; }
        }
        if (!ok && firstBad < h.addedCells.size())
        {
            std::printf("  FAIL: the history's addedCells differs at split cell %zu "
                        "(brae %zu entries, OpenFOAM %zu)\n", firstBad, h.addedCells[firstBad].size(),
                        firstBad < it->second.size() ? it->second[firstBad].size() : 0);
            ++failures;
        }
        else
        {
            check("the history's addedCells is OpenFOAM's, split cell for split cell", ok);
        }
    }

    std::printf("test_hex_ref8_vs_openfoam: %d failures, %d skipped\n", failures, skipped);
    return failures == 0 ? 0 : 1;
}
