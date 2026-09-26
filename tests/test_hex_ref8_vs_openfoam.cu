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
#include "mesh_edges_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include "poly_topo_change_cpp.cuh"
#include "fv_geometry.cuh"
#include <algorithm>
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
    const char* blocks[] = {"cellAddedCells", "pointsFromPointsMap", "facesFromPointsMap",
                            "facesFromEdgesMap", "facesFromFacesMap", "cellsFromPointsMap",
                            "cellsFromEdgesMap", "cellsFromFacesMap", "cellsFromCellsMap",
                            "points", "faces", "patches"};
    // 5b-2 reads the mesh the refinement produced, so the face and patch blocks are parsed now
    (void)0;
    std::string line;
    while (std::getline(is, line))
    {
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        bool isBlock = false;
        for (const char* b : blocks) if (key == b) isBlock = true;
        label n = 0;
        if (!(ls >> n)) continue;
        if (isBlock)
        {
            // one entry per following line; only cellAddedCells is read back as labels here, the
            // others are 5b-2's and are consumed as raw lines
            std::vector<std::vector<label>> v;
            v.reserve(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
            {
                std::getline(is, line);
                std::istringstream rs(line);
                if (key == "cellAddedCells" || key == "faces")
                {
                    label k = 0;
                    rs >> k;
                    std::vector<label> e(static_cast<std::size_t>(k));
                    for (label j = 0; j < k; ++j) rs >> e[static_cast<std::size_t>(j)];
                    v.push_back(e);
                }
                else if (key == "patches")
                {
                    std::string nm; label st = 0, sz = 0;
                    rs >> nm >> st >> sz;
                    v.push_back({st, sz});
                }
            }
            if (key == "cellAddedCells" || key == "faces" || key == "patches") d.listLists[key] = v;
            continue;
        }
        if (key.rfind('n', 0) == 0 && key.size() > 1
         && std::isupper(static_cast<unsigned char>(key[1])))
        {
            d.scalars[key] = n;
            continue;
        }
        std::vector<label> v(static_cast<std::size_t>(n));
        for (label i = 0; i < n; ++i) ls >> v[static_cast<std::size_t>(i)];
        d.lists[key] = v;
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
    skip("cellLevelFinal / pointLevelFinal after changeMesh -- needs hexRef8::updateMesh, unit 6");

    std::printf("test_hex_ref8_vs_openfoam: %d failures, %d skipped\n", failures, skipped);
    return failures == 0 ? 0 : 1;
}
