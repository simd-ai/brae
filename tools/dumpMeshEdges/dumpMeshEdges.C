/*---------------------------------------------------------------------------*\
    dumpMeshEdges -- primitiveMesh's edge addressing, and the ORDER question
    settled by OpenFOAM itself.

    hexRef8::setRefinement reads cellEdges(celli), edgeFaces(edgeI) and
    faceEdges(facei) (hexRef8.C:3416, :3892, :4066), and each of those has TWO
    implementations in primitiveMesh: a cached whole-mesh form and an on-demand
    form that is used when the cache has not been built. They need not agree in
    ORDER, and setRefinement's answer depends on the order -- so which one a
    given run takes is not a detail.

    This dumps both forms of all three, plus edges() and pointEdges(), and
    reports whether each pair agrees. brae's port is then written against the
    form OpenFOAM actually hands hexRef8, with this as the record of which.

    Reads only. No OpenFOAM source is edited.
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "polyMesh.H"
#include "OFstream.H"

using namespace Foam;

namespace
{

void writeListList(Ostream& os, const word& name, const labelListList& l)
{
    os << name << ' ' << l.size() << nl;
    for (const labelList& e : l)
    {
        os << "  " << e.size();
        for (const label v : e) os << ' ' << v;
        os << nl;
    }
}

}   // namespace


int main(int argc, char *argv[])
{
    argList::addNote("primitiveMesh's edge addressing, both forms");
    argList::addOption("out", "file", "where to write the dump (default meshEdges.dump)");

    #include "setRootCase.H"
    #include "createTime.H"

    const fileName outFile(args.getOrDefault<fileName>("out", "meshEdges.dump"));

    polyMesh mesh
    (
        IOobject("region0", runTime.timeName(), runTime, IOobject::MUST_READ)
    );

    // THE ON-DEMAND FORMS FIRST, before anything asks for a cached one -- primitiveMesh's on-demand
    // faceEdges/cellEdges/edgeFaces return the CACHE when hasFaceEdges()/hasCellEdges()/hasEdgeFaces()
    // is already true, so asking in the other order would measure the cache twice and prove nothing.
    labelListList faceEdgesOnDemand(mesh.nFaces());
    {
        DynamicList<label> storage;
        forAll(faceEdgesOnDemand, facei)
        {
            faceEdgesOnDemand[facei] = mesh.faceEdges(facei, storage);
        }
    }
    labelListList cellEdgesOnDemand(mesh.nCells());
    {
        labelHashSet set;
        DynamicList<label> storage;
        forAll(cellEdgesOnDemand, celli)
        {
            cellEdgesOnDemand[celli] = mesh.cellEdges(celli, set, storage);
        }
    }
    labelListList edgeFacesOnDemand(mesh.nEdges());
    {
        DynamicList<label> storage;
        forAll(edgeFacesOnDemand, edgei)
        {
            edgeFacesOnDemand[edgei] = mesh.edgeFaces(edgei, storage);
        }
    }

    Info<< "on-demand forms read;"
        << " hasFaceEdges " << Switch(mesh.hasFaceEdges())
        << " hasCellEdges " << Switch(mesh.hasCellEdges())
        << " hasEdgeFaces " << Switch(mesh.hasEdgeFaces()) << endl;

    // ...and now the cached whole-mesh forms
    const labelListList& faceEdgesCached = mesh.faceEdges();
    const labelListList& cellEdgesCached = mesh.cellEdges();
    const labelListList& edgeFacesCached = mesh.edgeFaces();

    // do the two forms agree, element for element and in ORDER?
    auto agree = [](const labelListList& a, const labelListList& b, label& firstBad)
    {
        if (a.size() != b.size()) { firstBad = -2; return false; }
        forAll(a, i)
        {
            if (a[i] != b[i]) { firstBad = i; return false; }
        }
        return true;
    };
    label bad = -1;
    const bool feSame = agree(faceEdgesOnDemand, faceEdgesCached, bad);
    Info<< "faceEdges  on-demand vs cached: " << Switch(feSame)
        << (feSame ? "" : " first differing at ") << (feSame ? "" : Foam::name(bad)) << endl;
    bad = -1;
    const bool ceSame = agree(cellEdgesOnDemand, cellEdgesCached, bad);
    Info<< "cellEdges  on-demand vs cached: " << Switch(ceSame)
        << (ceSame ? "" : " first differing at ") << (ceSame ? "" : Foam::name(bad)) << endl;
    bad = -1;
    const bool efSame = agree(edgeFacesOnDemand, edgeFacesCached, bad);
    Info<< "edgeFaces  on-demand vs cached: " << Switch(efSame)
        << (efSame ? "" : " first differing at ") << (efSame ? "" : Foam::name(bad)) << endl;

    OFstream os(outFile);
    os << "nPoints " << mesh.nPoints() << nl;
    os << "nFaces " << mesh.nFaces() << nl;
    os << "nCells " << mesh.nCells() << nl;
    os << "nEdges " << mesh.nEdges() << nl;
    os << "faceEdgesOnDemandEqualsCached " << (feSame ? 1 : 0) << nl;
    os << "cellEdgesOnDemandEqualsCached " << (ceSame ? 1 : 0) << nl;
    os << "edgeFacesOnDemandEqualsCached " << (efSame ? 1 : 0) << nl;

    os << "edges " << mesh.edges().size() << nl;
    for (const edge& e : mesh.edges()) os << "  " << e.start() << ' ' << e.end() << nl;
    writeListList(os, "pointEdges", mesh.pointEdges());
    writeListList(os, "faceEdges", faceEdgesCached);
    writeListList(os, "cellEdges", cellEdgesCached);
    writeListList(os, "edgeFaces", edgeFacesCached);
    writeListList(os, "faceEdgesOnDemand", faceEdgesOnDemand);
    writeListList(os, "cellEdgesOnDemand", cellEdgesOnDemand);
    writeListList(os, "edgeFacesOnDemand", edgeFacesOnDemand);

    Info<< "wrote " << outFile << nl << "End" << nl << endl;
    return 0;
}
