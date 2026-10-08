// brae's faceEdges(), edgeFaces() and cellEdges() against REAL OpenFOAM's primitiveMesh -- unit 5a of
// the dynamicRefineFvMesh port, and the prerequisite hexRef8::setRefinement reads.
//
// THE ORACLE is tools/dumpMeshEdges, which dumps BOTH of OpenFOAM's implementations of each -- the
// cached whole-mesh form and the on-demand one -- having asked the on-demand forms FIRST so the cache
// could not answer for them. tests/mesh_edges_addressing_vs_openfoam.sh carries what that measured.
//
// WHAT IS HELD, and it is not the same for all three:
//   faceEdges   EXACTLY, element for element. It is indexed by face POSITION at its call site
//               (hexRef8.C:4062-4080, `fEdges[fp]`), so the order is the answer.
//   edgeFaces   EXACTLY. Its call site only feeds a bitSet (:3888-3896), so the order is inert -- but
//               OpenFOAM's is deterministic and free to match, so it is matched and checked.
//   cellEdges   AS A SET, sorted. OpenFOAM's on-demand form walks a labelHashSet, so its order is
//               bucket order and cannot be reproduced; its only call site MARKS per edge (:3415-3433)
//               and cannot tell. The gate asserts brae's sorted list equals OpenFOAM's SORTED -- and
//               asserts separately that OpenFOAM's two forms really do disagree, so the day they stop
//               disagreeing this comment stops being true and somebody finds out.
#include "primitive_mesh.cuh"
#include "mesh_edges_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

struct Dump
{
    std::map<std::string, label> scalars;
    std::map<std::string, std::vector<std::vector<label>>> lists;
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
        if (key.rfind('n', 0) == 0 && key.size() > 1 && std::isupper(static_cast<unsigned char>(key[1])))
        {
            d.scalars[key] = n;
            continue;
        }
        if (key.find("EqualsCached") != std::string::npos)
        {
            d.scalars[key] = n;
            continue;
        }
        if (key == "edges")
        {
            // start/end pairs, one per line; kept as two-element lists so one reader serves all
            std::vector<std::vector<label>> v(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
            {
                std::getline(is, line);
                std::istringstream es(line);
                label s = 0, e = 0;
                es >> s >> e;
                v[static_cast<std::size_t>(i)] = {s, e};
            }
            d.lists[key] = v;
            continue;
        }
        std::vector<std::vector<label>> v(static_cast<std::size_t>(n));
        for (label i = 0; i < n; ++i)
        {
            std::getline(is, line);
            std::istringstream rs(line);
            label k = 0;
            rs >> k;
            v[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(k));
            for (label j = 0; j < k; ++j) rs >> v[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
        d.lists[key] = v;
    }
    return d;
}

void compare(const char* name, const std::vector<std::vector<label>>& mine, const Dump& d,
             const char* key, bool sortBoth)
{
    const auto it = d.lists.find(key);
    if (it == d.lists.end())
    {
        std::printf("  FAIL: the dump has no `%s`\n", key);
        ++failures;
        return;
    }
    if (mine.size() != it->second.size())
    {
        std::printf("  FAIL: %s has %zu entries, OpenFOAM %zu\n", name, mine.size(), it->second.size());
        ++failures;
        return;
    }
    std::size_t firstBad = mine.size();
    for (std::size_t i = 0; i < mine.size(); ++i)
    {
        std::vector<label> a = mine[i];
        std::vector<label> b = it->second[i];
        if (sortBoth) { std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); }
        if (a != b) { firstBad = i; break; }
    }
    if (firstBad < mine.size())
    {
        std::printf("  FAIL: %s differs first at %zu (brae %zu entries, OpenFOAM %zu)\n",
                    name, firstBad, mine[firstBad].size(), it->second[firstBad].size());
        ++failures;
    }
    else
    {
        std::printf("  ok:   %s is OpenFOAM's%s (%zu entries)\n", name,
                    sortBoth ? ", as a set" : ", in order", mine.size());
    }
}

}   // namespace


int main(int argc, char** argv)
{
    std::printf("== brae faceEdges / edgeFaces / cellEdges vs OpenFOAM's primitiveMesh ==\n");
    if (argc < 3)
    {
        std::printf("  SKIP: usage: %s <caseDir> <dump>\n", argv[0]);
        return 77;
    }
    PrimitiveMesh m;
    m.read(std::string(argv[1]) + "/constant/polyMesh");
    const Dump d = readDump(argv[2]);

    // the fail-proof: an empty parse would make every comparison below vacuous
    check("the dump carries OpenFOAM's addressing",
          d.lists.count("faceEdges") && d.lists.count("edgeFaces") && d.lists.count("cellEdges")
       && d.lists.count("pointEdges") && d.lists.count("edges"));
    check("...and it is this mesh's",
          d.scalars.count("nFaces") && d.scalars.at("nFaces") == m.nFaces()
       && d.scalars.at("nCells") == m.nCells());

    const MeshEdges me = buildMeshEdges(m);
    std::printf("  brae: %zu edges\n", me.start.size());
    check("brae's edge count is OpenFOAM's",
          d.scalars.count("nEdges") && (label)me.start.size() == d.scalars.at("nEdges"));
    {
        // the edge list itself, since faceEdges' indices mean nothing without it
        bool same = (me.start.size() == d.lists.at("edges").size());
        for (std::size_t i = 0; same && i < me.start.size(); ++i)
        {
            same = (me.start[i] == d.lists.at("edges")[i][0])
                && (me.end[i] == d.lists.at("edges")[i][1]);
        }
        check("...and its edges are OpenFOAM's, in OpenFOAM's numbering", same);
    }
    // pointEdges is held compact (compact_list_list.cuh); the comparison is on the list of lists it unpacks to
    compare("pointEdges", me.pointEdges.unpack(), d, "pointEdges", /*sortBoth=*/false);

    const std::vector<std::vector<label>> fe = buildFaceEdges(m, me);
    const std::vector<std::vector<label>> ef = buildEdgeFaces(m, fe);
    const std::vector<std::vector<label>> cells = meshCells(m);
    const std::vector<std::vector<label>> ce = buildCellEdges(cells, fe);

    // faceEdges IS ORDER, because its call site indexes it by face position
    compare("faceEdges", fe, d, "faceEdges", /*sortBoth=*/false);
    // edgeFaces' order is inert at its call site but deterministic, so it is held in order anyway
    compare("edgeFaces", ef, d, "edgeFaces", /*sortBoth=*/false);
    // cellEdges AS A SET -- see the file header
    compare("cellEdges", ce, d, "cellEdges", /*sortBoth=*/true);

    // ...AND THE PREMISE THAT LETS cellEdges BE COMPARED AS A SET: OpenFOAM's own two forms disagree,
    // so there is no single order to match. Asserted rather than asserted-in-a-comment.
    check("OpenFOAM's faceEdges agrees with itself between its two forms",
          d.scalars.count("faceEdgesOnDemandEqualsCached")
       && d.scalars.at("faceEdgesOnDemandEqualsCached") == 1);
    check("...its edgeFaces too",
          d.scalars.count("edgeFacesOnDemandEqualsCached")
       && d.scalars.at("edgeFacesOnDemandEqualsCached") == 1);
    check("...and its cellEdges DOES NOT, which is why brae matches that one as a set only",
          d.scalars.count("cellEdgesOnDemandEqualsCached")
       && d.scalars.at("cellEdgesOnDemandEqualsCached") == 0);
    // ...and brae's cellEdges must equal the ON-DEMAND form as a set too, which is the form hexRef8
    // is handed on a mesh whose cache is unbuilt
    compare("cellEdges against OpenFOAM's ON-DEMAND form", ce, d, "cellEdgesOnDemand", true);

    std::printf("test_mesh_edges_addressing_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
