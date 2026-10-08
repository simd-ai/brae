// brae's polyTopoChange action surface and changeMesh against REAL OpenFOAM's, on a scripted action
// list both codes read from the same file -- units 3 and 4 of the dynamicRefineFvMesh port.
//
// THE ORACLE is tools/dumpTopoChange. polyTopoChange's accumulated state is private, so it cannot be
// dumped and replayed; the ACTION LIST is the input instead. The tool authors one, writes it out, plays
// it into OpenFOAM's own engine and dumps the mapPolyMesh that comes out. This reads the same list,
// replays it through brae's action surface and changeMesh, and must produce the same map AND the same
// mesh. tests/topo_change_vs_openfoam.sh carries the scenarios and their measurements.
//
// WHY UNITS 3 AND 4 SHARE ONE GATE: the actions have no OpenFOAM-visible output of their own. What can
// be checked on the action surface alone is the addMesh ROUND TRIP -- every map the identity, the state
// the mesh -- and that is asserted here before any scenario runs.
#include "primitive_mesh.cuh"
#include "poly_topo_change_cpp.cuh"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::polyTopoChange;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

// the dump's own vocabulary: `name N v0 v1 ...` for a label list, and a block for an objectMap list
struct Dump
{
    std::map<std::string, std::vector<label>> lists;
    std::map<std::string, std::vector<ObjectMap>> objectMaps;
    std::map<std::string, label> scalars;
    std::vector<vector> points;
    std::vector<std::vector<label>> faces;
    std::vector<std::string> patchNames;
    std::vector<label> patchStarts, patchSizes;
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
        if (key == "scenario") continue;
        if (key.rfind("nOld", 0) == 0 || key == "nPoints" || key == "nFaces"
         || key == "nInternalFaces" || key == "nCells")
        {
            label v = 0; ls >> v; d.scalars[key] = v; continue;
        }
        if (key == "points")
        {
            label n = 0; ls >> n;
            d.points.resize(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
            {
                std::getline(is, line);
                std::istringstream ps(line);
                ps >> d.points[static_cast<std::size_t>(i)].x
                   >> d.points[static_cast<std::size_t>(i)].y
                   >> d.points[static_cast<std::size_t>(i)].z;
            }
            continue;
        }
        if (key == "faces")
        {
            label n = 0; ls >> n;
            d.faces.resize(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
            {
                std::getline(is, line);
                std::istringstream fs(line);
                label k = 0; fs >> k;
                d.faces[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(k));
                for (label j = 0; j < k; ++j) fs >> d.faces[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }
            continue;
        }
        if (key == "patches")
        {
            label n = 0; ls >> n;
            for (label i = 0; i < n; ++i)
            {
                std::getline(is, line);
                std::istringstream ps(line);
                std::string nm; label st = 0, sz = 0;
                ps >> nm >> st >> sz;
                d.patchNames.push_back(nm); d.patchStarts.push_back(st); d.patchSizes.push_back(sz);
            }
            continue;
        }
        // the objectMap BLOCKS are the eight inflation lists and every one of their names carries
        // `From`. Keying on the "Map" suffix instead would swallow pointMap, faceMap and cellMap --
        // which it did, and every comparison then reported "the dump has no `pointMap`".
        if (key.find("From") != std::string::npos)
        {
            label n = 0; ls >> n;
            std::vector<ObjectMap> oms(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
            {
                std::getline(is, line);
                std::istringstream os2(line);
                label idx = 0, k = 0;
                os2 >> idx >> k;
                oms[static_cast<std::size_t>(i)].index = idx;
                oms[static_cast<std::size_t>(i)].masterObjects.resize(static_cast<std::size_t>(k));
                for (label j = 0; j < k; ++j)
                    os2 >> oms[static_cast<std::size_t>(i)].masterObjects[static_cast<std::size_t>(j)];
            }
            d.objectMaps[key] = oms;
            continue;
        }
        // a plain label list
        label n = 0;
        if (!(ls >> n)) continue;
        std::vector<label> v(static_cast<std::size_t>(n));
        for (label i = 0; i < n; ++i) ls >> v[static_cast<std::size_t>(i)];
        d.lists[key] = v;
    }
    return d;
}

// the action list, in the tool's own order and vocabulary
struct Action
{
    std::string kind;
    vector pt{0, 0, 0};
    std::vector<label> verts;
    std::vector<label> args;
};

std::vector<Action> readActions(const std::string& path)
{
    std::vector<Action> out;
    std::ifstream is(path);
    if (!is) { std::printf("  FAIL: cannot open %s\n", path.c_str()); ++failures; return out; }
    std::string line;
    while (std::getline(is, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string kind;
        ls >> kind;
        if (kind == "nActions") continue;
        Action a;
        a.kind = kind;
        if (kind == "addPoint") { ls >> a.pt.x >> a.pt.y >> a.pt.z; }
        if (kind == "addFace" || kind == "modifyFace")
        {
            label k = 0; ls >> k;
            a.verts.resize(static_cast<std::size_t>(k));
            for (label i = 0; i < k; ++i) ls >> a.verts[static_cast<std::size_t>(i)];
        }
        label v = 0;
        while (ls >> v) a.args.push_back(v);
        out.push_back(a);
    }
    return out;
}

void replay(TopoActions& ta, const std::vector<Action>& acts)
{
    for (const Action& a : acts)
    {
        const std::vector<label>& g = a.args;
        if (a.kind == "addPoint")        addPoint(ta, a.pt, g[0], bool(g[2]));
        else if (a.kind == "addCell")    addCell(ta, g[0], g[1], g[2], g[3]);
        else if (a.kind == "addFace")    addFace(ta, a.verts, g[0], g[1], g[2], g[3], g[4], bool(g[5]), g[6]);
        // polyModifyFace carries `removeFromZone` between patchID and zoneID; brae keeps no zones, so
        // g[5] is read and dropped -- and the gate refuses a case with zones, so it is always 0 here
        else if (a.kind == "modifyFace") modifyFace(ta, a.verts, g[0], g[1], g[2], bool(g[3]), g[4]);
        else if (a.kind == "removePoint") removePoint(ta, g[0], g[1]);
        else if (a.kind == "removeFace")  removeFace(ta, g[0], g[1]);
        else if (a.kind == "removeCell")  removeCell(ta, g[0], g[1]);
        else { std::printf("  FAIL: unknown action `%s`\n", a.kind.c_str()); ++failures; }
    }
}

bool sameLists(const std::vector<label>& a, const std::vector<label>& b)
{
    return a == b;
}

void compareList(const char* name, const std::vector<label>& mine, const Dump& d, const char* key)
{
    const auto it = d.lists.find(key);
    if (it == d.lists.end())
    {
        std::printf("  FAIL: the dump has no `%s`\n", key);
        ++failures;
        return;
    }
    const bool ok = sameLists(mine, it->second);
    if (!ok)
    {
        std::size_t first = 0;
        while (first < mine.size() && first < it->second.size() && mine[first] == it->second[first]) ++first;
        std::printf("  FAIL: %s differs -- brae %zu entries, OpenFOAM %zu; first at %zu (%d vs %d)\n",
                    name, mine.size(), it->second.size(), first,
                    first < mine.size() ? (int)mine[first] : -999,
                    first < it->second.size() ? (int)it->second[first] : -999);
        ++failures;
    }
    else
    {
        std::printf("  ok:   %s is OpenFOAM's (%zu entries)\n", name, mine.size());
    }
}

void compareObjectMaps(const char* name, const std::vector<ObjectMap>& mine, const Dump& d,
                       const char* key)
{
    const auto it = d.objectMaps.find(key);
    if (it == d.objectMaps.end())
    {
        std::printf("  FAIL: the dump has no `%s`\n", key);
        ++failures;
        return;
    }
    bool ok = (mine.size() == it->second.size());
    for (std::size_t i = 0; ok && i < mine.size(); ++i)
    {
        ok = (mine[i].index == it->second[i].index)
          && (mine[i].masterObjects == it->second[i].masterObjects);
    }
    std::printf(ok ? "  ok:   %s is OpenFOAM's (%zu sets)\n" : "  FAIL: %s differs (brae %zu sets)\n",
                name, mine.size());
    if (!ok) ++failures;
}

}   // namespace


int main(int argc, char** argv)
{
    std::printf("== brae polyTopoChange vs OpenFOAM: the action surface and changeMesh ==\n");
    if (argc < 4)
    {
        std::printf("  SKIP: usage: %s <caseDir> <actions> <dump>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string actPath = argv[2];
    const std::string dumpPath = argv[3];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");

    // brae keeps faces as CSR and `neighbour()` over the internal faces only; the action surface takes
    // OpenFOAM's shapes, so the caller converts -- which is what dynamicRefineFvMesh's caller will do
    std::vector<std::vector<label>> faces(static_cast<std::size_t>(m.nFaces()));
    for (label f = 0; f < m.nFaces(); ++f)
    {
        faces[static_cast<std::size_t>(f)].assign(m.faceVerts().begin() + m.faceOffsets()[f],
                                                  m.faceVerts().begin() + m.faceOffsets()[f + 1]);
    }
    std::vector<label> nbr(static_cast<std::size_t>(m.nFaces()), label(-1));
    for (label f = 0; f < m.nInternalFaces(); ++f) nbr[static_cast<std::size_t>(f)] = m.neighbour()[f];
    std::vector<label> starts, sizes;
    std::vector<std::string> types;
    for (const auto& p : m.patches()) { starts.push_back(p.start); sizes.push_back(p.size); types.push_back(p.type); }

    // ---- THE ADD-MESH ROUND TRIP, which is unit 3 on its own ------------------------------------
    {
        TopoActions a;
        addMesh(a, m.points(), faces, m.owner(), nbr, m.nCells(), starts, sizes);
        bool pid = true, fid = true, cid = true, rp = true, rf = true, rc = true;
        bool fv = true, own = true, nei = true, reg = true;
        for (std::size_t i = 0; i < a.state.pointMap.size(); ++i)
        {
            if (a.state.pointMap[i] != (label)i) pid = false;
            if (a.reversePointMap[i] != (label)i) rp = false;
        }
        for (std::size_t i = 0; i < a.state.faceMap.size(); ++i)
        {
            if (a.state.faceMap[i] != (label)i) fid = false;
            if (a.reverseFaceMap[i] != (label)i) rf = false;
            if (a.state.faces[i] != faces[i]) fv = false;
            if (a.state.faceOwner[i] != m.owner()[i]) own = false;
            if (a.state.faceNeighbour[i] != nbr[i]) nei = false;
        }
        for (std::size_t i = 0; i < a.state.cellMap.size(); ++i)
        {
            if (a.state.cellMap[i] != (label)i) cid = false;
            if (a.reverseCellMap[i] != (label)i) rc = false;
        }
        for (label f = 0; f < m.nInternalFaces(); ++f)
            if (a.state.region[static_cast<std::size_t>(f)] != -1) reg = false;
        for (std::size_t pi = 0; pi < starts.size(); ++pi)
            for (label i = 0; i < sizes[pi]; ++i)
                if (a.state.region[static_cast<std::size_t>(starts[pi] + i)] != (label)pi) reg = false;
        std::printf("  addMesh: %zu points, %zu faces, %zu cells\n",
                    a.state.points.size(), a.state.faces.size(), a.state.cellMap.size());
        check("addMesh took every point, face and cell",
              (label)a.state.points.size() == (label)m.points().size()
           && (label)a.state.faces.size() == m.nFaces()
           && (label)a.state.cellMap.size() == m.nCells());
        check("...pointMap, faceMap and cellMap are the identity", pid && fid && cid);
        check("...and so are the three reverse maps", rp && rf && rc);
        check("...the face vertex lists, owner and neighbour are the mesh's", fv && own && nei);
        check("...region is -1 inside and the patch id out", reg);
        check("...nothing is retired and nothing is inflated",
              a.state.retiredPoints.empty() && a.faceFromPoint.empty() && a.faceFromEdge.empty()
           && a.cellFromPoint.empty() && a.cellFromEdge.empty() && a.cellFromFace.empty());
        check("...and the patch count is the boundary's", a.state.nPatches == (label)starts.size());
    }

    // ---- THE SCENARIO: replay OpenFOAM's own action list and compare the map --------------------
    const std::vector<Action> acts = readActions(actPath);
    const Dump d = readDump(dumpPath);
    std::printf("  replaying %zu actions from %s\n", acts.size(), actPath.c_str());
    // the fail-proof: an empty parse would make every comparison below vacuous
    check("the dump carries OpenFOAM's map", d.lists.count("cellMap") && d.lists.count("faceMap")
       && d.objectMaps.count("cellsFromCellsMap"));

    TopoActions a;
    addMesh(a, m.points(), faces, m.owner(), nbr, m.nCells(), starts, sizes);
    replay(a, acts);

    ChangeMeshInput in;
    in.nOldPoints = (label)m.points().size();
    in.nOldFaces = m.nFaces();
    in.nOldCells = m.nCells();
    in.oldPatchStarts = starts;
    in.oldPatchSizes = sizes;
    in.oldPatchNMeshPoints.assign(starts.size(), label(0));   // not compared; see the script
    in.patchTypes = types;
    in.nZones = 0;

    ChangedMesh out;
    TopoChangeMap map;
    changeMesh(a, in, out, map);

    std::printf("  brae: %zu points, %zu faces (%d internal), %d cells\n",
                out.points.size(), out.faces.size(), (int)out.nInternalFaces, (int)out.nCells);
    check("the mesh has OpenFOAM's point, face and cell counts",
          (label)out.points.size() == d.scalars.at("nPoints")
       && (label)out.faces.size() == d.scalars.at("nFaces")
       && out.nInternalFaces == d.scalars.at("nInternalFaces")
       && out.nCells == d.scalars.at("nCells"));

    compareList("pointMap", map.pointMap, d, "pointMap");
    compareList("faceMap", map.faceMap, d, "faceMap");
    compareList("cellMap", map.cellMap, d, "cellMap");
    compareList("reversePointMap", map.reversePointMap, d, "reversePointMap");
    compareList("reverseFaceMap", map.reverseFaceMap, d, "reverseFaceMap");
    compareList("reverseCellMap", map.reverseCellMap, d, "reverseCellMap");
    compareList("flipFaceFlux", map.flipFaceFlux, d, "flipFaceFlux");
    compareList("oldPatchStarts", map.oldPatchStarts, d, "oldPatchStarts");
    compareList("oldPatchSizes", map.oldPatchSizes, d, "oldPatchSizes");

    compareObjectMaps("pointsFromPoints", map.pointsFromPoints, d, "pointsFromPointsMap");
    compareObjectMaps("facesFromFaces", map.facesFromFaces, d, "facesFromFacesMap");
    compareObjectMaps("cellsFromCells", map.cellsFromCells, d, "cellsFromCellsMap");
    // the four the unit REFUSES must be empty on both sides, which is the assertion that the refusal
    // is not hiding a case the fixture actually reaches
    compareObjectMaps("facesFromPoints (refused, must be empty)", map.facesFromPoints, d, "facesFromPointsMap");
    compareObjectMaps("facesFromEdges (refused, must be empty)", map.facesFromEdges, d, "facesFromEdgesMap");
    compareObjectMaps("cellsFromPoints (refused, must be empty)", map.cellsFromPoints, d, "cellsFromPointsMap");
    compareObjectMaps("cellsFromEdges (refused, must be empty)", map.cellsFromEdges, d, "cellsFromEdgesMap");
    compareObjectMaps("cellsFromFaces (refused, must be empty)", map.cellsFromFaces, d, "cellsFromFacesMap");

    // ---- THE MESH ITSELF, not only the map ------------------------------------------------------
    compareList("owner", out.faceOwner, d, "owner");
    compareList("neighbour", out.faceNeighbour, d, "neighbour");
    {
        bool sameFaces = (out.faces.size() == d.faces.size());
        std::size_t firstBad = out.faces.size();
        for (std::size_t i = 0; i < out.faces.size() && i < d.faces.size(); ++i)
        {
            if (out.faces[i] != d.faces[i]) { sameFaces = false; firstBad = i; break; }
        }
        if (!sameFaces && firstBad < out.faces.size())
        {
            std::printf("  FAIL: face %zu differs -- brae %zu vertices, OpenFOAM %zu\n",
                        firstBad, out.faces[firstBad].size(), d.faces[firstBad].size());
            ++failures;
        }
        else
        {
            check("every face's vertex list is OpenFOAM's", sameFaces);
        }
        scalar worst = 0;
        for (std::size_t i = 0; i < out.points.size() && i < d.points.size(); ++i)
        {
            worst = std::fmax(worst, mag(out.points[i] - d.points[i]));
        }
        std::printf("  points: worst |dx| %.4e\n", (double)worst);
        check("every point is OpenFOAM's, bitwise", worst == scalar(0));
    }
    check("the patch slicing is OpenFOAM's",
          out.patchStarts == d.patchStarts && out.patchSizes == d.patchSizes);

    std::printf("test_topo_change_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
