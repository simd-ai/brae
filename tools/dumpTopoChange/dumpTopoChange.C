/*---------------------------------------------------------------------------*\
    dumpTopoChange -- OpenFOAM's polyTopoChange::changeMesh on a SCRIPTED
    action list, and a dump of the mapPolyMesh it produces.

    THE ORACLE FOR UNIT 3+4 of the dynamicRefineFvMesh port. brae must produce
    OpenFOAM's own map, and until hexRef8::setRefinement exists on brae's side
    there is nothing to produce one FROM -- polyTopoChange's accumulated state
    is private, so it cannot be dumped and replayed either.

    So the ACTION LIST is the input, authored here and written to disk, and both
    codes read the same file. That makes the comparison a real one: OpenFOAM
    applies the list and writes its map; brae applies the same list and must
    write the same map.

    WHAT IT DOES NOT COVER, said here rather than left to be discovered: a legal
    8-way hex split. Authoring one by hand is writing hexRef8::setRefinement,
    which is unit 5 -- so the ADD path's inflation maps (facesFromEdges,
    facesFromPoints, cellsFromPoints, cellsFromEdges) are not witnessed by these
    scenarios. `mergeTwoCells` witnesses the REMOVAL path end to end:
    cellsFromCells, facesFromFaces, the compaction's holes and the flip rule.

    No OpenFOAM source is edited and no arithmetic is changed: the tool calls
    setAction and changeMesh and reads the public map.

    Scenarios (-scenario):
      noop           no actions at all. changeMesh must return the identity.
      mergeTwoCells  merge cell `-cell` into the neighbour across its first
                     internal face: remove the shared face, remove the cell with
                     the neighbour as its merge master, and modify every other
                     face of the removed cell onto the survivor.
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "polyMesh.H"
#include "polyTopoChange.H"
#include "polyAddPoint.H"
#include "polyAddFace.H"
#include "polyAddCell.H"
#include "polyModifyFace.H"
#include "polyRemovePoint.H"
#include "polyRemoveFace.H"
#include "polyRemoveCell.H"
#include "mapPolyMesh.H"
#include "OFstream.H"
#include "IOmanip.H"

using namespace Foam;

namespace
{

// one scripted action, in the file's own vocabulary
struct Action
{
    word kind;
    // addPoint
    point pt = point::zero;
    // the label arguments, in the order the file writes them
    labelList args;
    // addFace / modifyFace only
    labelList verts;
};

void writeAction(Ostream& os, const Action& a)
{
    os << a.kind;
    if (a.kind == "addPoint")
    {
        os << ' ' << setprecision(17) << a.pt.x() << ' ' << a.pt.y() << ' ' << a.pt.z();
    }
    if (a.kind == "addFace" || a.kind == "modifyFace")
    {
        os << ' ' << a.verts.size();
        for (const label v : a.verts) os << ' ' << v;
    }
    for (const label v : a.args) os << ' ' << v;
    os << nl;
}

// setAction, in the file's order. Returns what OpenFOAM returned (a new index, or -1).
label apply(polyTopoChange& meshMod, const Action& a)
{
    const labelList& g = a.args;
    if (a.kind == "addPoint")
    {
        // pt masterPointID zoneID inCell
        return meshMod.setAction(polyAddPoint(a.pt, g[0], g[1], bool(g[2])));
    }
    if (a.kind == "addCell")
    {
        // masterPointID masterEdgeID masterFaceID masterCellID zoneID
        return meshMod.setAction(polyAddCell(g[0], g[1], g[2], g[3], g[4]));
    }
    if (a.kind == "addFace")
    {
        // own nei masterPointID masterEdgeID masterFaceID flipFaceFlux patchID zoneID zoneFlip
        return meshMod.setAction
        (
            polyAddFace
            (
                face(a.verts), g[0], g[1], g[2], g[3], g[4],
                bool(g[5]), g[6], g[7], bool(g[8])
            )
        );
    }
    if (a.kind == "modifyFace")
    {
        // facei own nei flipFaceFlux patchID removeFromZone zoneID zoneFlip -- polyModifyFace carries a
        // `removeFromZone` bool between patchID and zoneID that polyAddFace does not, so the two action
        // records are NOT the same shape
        return meshMod.setAction
        (
            polyModifyFace
            (
                face(a.verts), g[0], g[1], g[2], bool(g[3]), g[4], bool(g[5]), g[6], bool(g[7])
            )
        );
    }
    if (a.kind == "removePoint")
    {
        return meshMod.setAction(polyRemovePoint(g[0], g[1]));
    }
    if (a.kind == "removeFace")
    {
        return meshMod.setAction(polyRemoveFace(g[0], g[1]));
    }
    if (a.kind == "removeCell")
    {
        return meshMod.setAction(polyRemoveCell(g[0], g[1]));
    }
    FatalErrorInFunction << "unknown action `" << a.kind << '`' << exit(FatalError);
    return -1;
}

void writeLabels(Ostream& os, const word& name, const labelUList& l)
{
    os << name << ' ' << l.size();
    for (const label v : l) os << ' ' << v;
    os << nl;
}

void writeObjectMaps(Ostream& os, const word& name, const List<objectMap>& l)
{
    os << name << ' ' << l.size() << nl;
    for (const objectMap& om : l)
    {
        os << "  " << om.index() << ' ' << om.masterObjects().size();
        for (const label v : om.masterObjects()) os << ' ' << v;
        os << nl;
    }
}

}   // namespace


int main(int argc, char *argv[])
{
    argList::addNote("polyTopoChange::changeMesh on a scripted action list");
    argList::addOption("scenario", "word", "noop | mergeTwoCells");
    argList::addOption("cell", "label", "the cell mergeTwoCells removes (default 0)");
    argList::addOption("out", "file", "where to write the dump (default topoChange.dump)");
    argList::addOption("actions", "file", "where to write the action list (default topoChange.actions)");

    #include "setRootCase.H"
    #include "createTime.H"

    const word scenario(args.getOrDefault<word>("scenario", "noop"));
    const label cellToRemove(args.getOrDefault<label>("cell", 0));
    const fileName outFile(args.getOrDefault<fileName>("out", "topoChange.dump"));
    const fileName actFile(args.getOrDefault<fileName>("actions", "topoChange.actions"));

    polyMesh mesh
    (
        IOobject("region0", runTime.timeName(), runTime, IOobject::MUST_READ)
    );

    const label nOldPoints = mesh.nPoints();
    const label nOldFaces = mesh.nFaces();
    const label nOldCells = mesh.nCells();

    // ---- author the action list ----------------------------------------------------------------
    DynamicList<Action> actions;

    if (scenario == "mergeTwoCells")
    {
        // The survivor is the cell across `cellToRemove`'s first INTERNAL face. Every other face of
        // the removed cell is modified onto the survivor; the shared face goes away. This is a legal
        // mesh change: no face is left pointing at a cell that no longer exists.
        const cell& cFaces = mesh.cells()[cellToRemove];
        label shared = -1;
        label survivor = -1;
        for (const label facei : cFaces)
        {
            if (mesh.isInternalFace(facei))
            {
                shared = facei;
                survivor =
                (
                    mesh.faceOwner()[facei] == cellToRemove
                  ? mesh.faceNeighbour()[facei]
                  : mesh.faceOwner()[facei]
                );
                break;
            }
        }
        if (shared == -1)
        {
            FatalErrorInFunction
                << "cell " << cellToRemove << " has no internal face" << exit(FatalError);
        }
        Info<< "mergeTwoCells: removing cell " << cellToRemove
            << " into " << survivor << " across face " << shared << endl;

        // the shared face first, as a removal with no merge master
        {
            Action a; a.kind = "removeFace"; a.args = labelList({shared, -1});
            actions.append(a);
        }
        // every other face of the removed cell moves onto the survivor
        for (const label facei : cFaces)
        {
            if (facei == shared) continue;
            label own = mesh.faceOwner()[facei];
            label nei = mesh.isInternalFace(facei) ? mesh.faceNeighbour()[facei] : -1;
            if (own == cellToRemove) own = survivor;
            if (nei == cellToRemove) nei = survivor;
            // OpenFOAM requires owner < neighbour on an internal face; flip when the merge broke it
            face f(mesh.faces()[facei]);
            bool flip = false;
            if (nei != -1 && own > nei)
            {
                f = f.reverseFace();
                Swap(own, nei);
                flip = true;
            }
            const label patchi = mesh.isInternalFace(facei)
                               ? -1 : mesh.boundaryMesh().whichPatch(facei);
            Action a;
            a.kind = "modifyFace";
            a.verts = labelList(f);
            // facei own nei flipFaceFlux patchID removeFromZone zoneID zoneFlip
            a.args = labelList({facei, own, nei, label(flip), patchi, 0, -1, 0});
            actions.append(a);
        }
        // and the cell itself, with the survivor as its merge master
        {
            Action a; a.kind = "removeCell"; a.args = labelList({cellToRemove, survivor});
            actions.append(a);
        }
    }
    else if (scenario != "noop")
    {
        FatalErrorInFunction << "unknown scenario `" << scenario << '`' << exit(FatalError);
    }

    // ---- write the list, so brae reads the same input ------------------------------------------
    {
        OFstream os(actFile);
        os << "# dumpTopoChange action list, scenario " << scenario << nl;
        os << "nActions " << actions.size() << nl;
        for (const Action& a : actions) writeAction(os, a);
        Info<< "wrote " << actions.size() << " actions to " << actFile << endl;
    }

    // ---- play them into OpenFOAM's own engine ---------------------------------------------------
    polyTopoChange meshMod(mesh);
    for (const Action& a : actions) apply(meshMod, a);

    autoPtr<mapPolyMesh> mapPtr(meshMod.changeMesh(mesh, false));
    const mapPolyMesh& map = mapPtr();

    // ---- dump the map and the mesh it produced --------------------------------------------------
    OFstream os(outFile);
    os.precision(17);
    os << "scenario " << scenario << nl;
    os << "nOldPoints " << nOldPoints << nl;
    os << "nOldFaces " << nOldFaces << nl;
    os << "nOldCells " << nOldCells << nl;
    os << "nPoints " << mesh.nPoints() << nl;
    os << "nFaces " << mesh.nFaces() << nl;
    os << "nInternalFaces " << mesh.nInternalFaces() << nl;
    os << "nCells " << mesh.nCells() << nl;

    writeLabels(os, "pointMap", map.pointMap());
    writeLabels(os, "faceMap", map.faceMap());
    writeLabels(os, "cellMap", map.cellMap());
    writeLabels(os, "reversePointMap", map.reversePointMap());
    writeLabels(os, "reverseFaceMap", map.reverseFaceMap());
    writeLabels(os, "reverseCellMap", map.reverseCellMap());
    writeLabels(os, "oldPatchStarts", map.oldPatchStarts());
    writeLabels(os, "oldPatchNMeshPoints", map.oldPatchNMeshPoints());
    writeLabels(os, "oldPatchSizes", map.oldPatchSizes());

    // flipFaceFlux is a labelHashSet in the map; written as a sorted list
    {
        labelList flips(map.flipFaceFlux().sortedToc());
        writeLabels(os, "flipFaceFlux", flips);
    }

    writeObjectMaps(os, "pointsFromPointsMap", map.pointsFromPointsMap());
    writeObjectMaps(os, "facesFromPointsMap", map.facesFromPointsMap());
    writeObjectMaps(os, "facesFromEdgesMap", map.facesFromEdgesMap());
    writeObjectMaps(os, "facesFromFacesMap", map.facesFromFacesMap());
    writeObjectMaps(os, "cellsFromPointsMap", map.cellsFromPointsMap());
    writeObjectMaps(os, "cellsFromEdgesMap", map.cellsFromEdgesMap());
    writeObjectMaps(os, "cellsFromFacesMap", map.cellsFromFacesMap());
    writeObjectMaps(os, "cellsFromCellsMap", map.cellsFromCellsMap());

    // the mesh the change produced, so brae's resetPrimitives can be compared too
    os << "points " << mesh.points().size() << nl;
    for (const point& p : mesh.points())
    {
        os << "  " << p.x() << ' ' << p.y() << ' ' << p.z() << nl;
    }
    os << "faces " << mesh.faces().size() << nl;
    for (const face& f : mesh.faces())
    {
        os << "  " << f.size();
        for (const label v : f) os << ' ' << v;
        os << nl;
    }
    writeLabels(os, "owner", mesh.faceOwner());
    writeLabels(os, "neighbour", mesh.faceNeighbour());
    os << "patches " << mesh.boundaryMesh().size() << nl;
    forAll(mesh.boundaryMesh(), patchi)
    {
        const polyPatch& pp = mesh.boundaryMesh()[patchi];
        os << "  " << pp.name() << ' ' << pp.start() << ' ' << pp.size() << nl;
    }

    Info<< "wrote " << outFile << nl
        << "cells " << nOldCells << " -> " << mesh.nCells()
        << ", faces " << nOldFaces << " -> " << mesh.nFaces()
        << ", points " << nOldPoints << " -> " << mesh.nPoints() << endl;

    Info<< "End" << nl << endl;
    return 0;
}
