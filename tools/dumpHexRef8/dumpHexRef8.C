/*---------------------------------------------------------------------------*\
    dumpHexRef8 -- OpenFOAM's hexRef8::setRefinement on a chosen cell set, and
    a dump of everything it produced.

    THE ORACLE FOR UNIT 5b of the dynamicRefineFvMesh port, which is ~1,900
    lines of OpenFOAM (setRefinement 1,011 plus 911 of private helpers). A
    transcription that size with only an end-to-end comparison would be very
    hard to localise when it disagrees, so this dumps everything hexRef8 makes
    PUBLIC after the call, which is more than it first appears:

      setRefinement RETURNS cellAddedCells -- per refined cell, the eight cells
        it became (element 0 is the original). That is section 7-8's whole output.
      cellLevel() and pointLevel() afterwards -- the refinement levels of every
        cell and point including the new ones, which encode the anchor and
        mid-point structure of sections 2-7.
      changeMesh's mapPolyMesh and the new mesh -- the end-to-end answer.

    So no copy of hexRef8.C is needed. If a disagreement still will not localise,
    THEN copy the class and add writes (the of-instrument pattern) -- not before.

    THE INPUT is a cell set, written to disk so brae refines the same one.
    `-cells <file>` reads one; otherwise every cell whose centre lies in a box
    given by -box, or cell 0 alone. Whatever is requested is passed through
    hexRef8::consistentRefinement FIRST -- the 2:1 closure, which brae already
    has and which is gated separately -- and the RESULT is what is written and
    what setRefinement is given, so the two codes refine an identical, legal set.

    Reads only: no OpenFOAM source is edited and no arithmetic is changed.
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "polyMesh.H"
#include "hexRef8.H"
#include "refinementHistory.H"
#include "polyTopoChange.H"
#include "mapPolyMesh.H"
#include "OFstream.H"
#include "IFstream.H"
#include "boundBox.H"
#include "IStringStream.H"
#include "OSspecific.H"

using namespace Foam;

namespace
{

void writeLabels(Ostream& os, const word& name, const labelUList& l)
{
    os << name << ' ' << l.size();
    for (const label v : l) os << ' ' << v;
    os << nl;
}

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
    argList::addNote("hexRef8::setRefinement on a chosen cell set");
    argList::addOption("cells", "file", "a label list of cells to refine");
    argList::addOption("box", "((minx miny minz) (maxx maxy maxz))",
                       "refine cells whose centre is inside");
    argList::addOption("out", "file", "where to write the dump (default hexRef8.dump)");
    argList::addOption("cellsOut", "file", "where to write the consistent set (default hexRef8.cells)");
    argList::addOption("times", "int", "how many refinements to run; the dump describes the LAST");
    argList::addOption("meshOut", "dir", "write the mesh as it stood BEFORE the last refinement here");

    #include "setRootCase.H"
    #include "createTime.H"

    const fileName outFile(args.getOrDefault<fileName>("out", "hexRef8.dump"));
    const fileName cellsOut(args.getOrDefault<fileName>("cellsOut", "hexRef8.cells"));

    polyMesh mesh
    (
        IOobject("region0", runTime.timeName(), runTime, IOobject::MUST_READ)
    );

    const label nOldPoints = mesh.nPoints();
    const label nOldFaces = mesh.nFaces();
    const label nOldCells = mesh.nCells();

    hexRef8 meshCutter(mesh);

    // ---- what to refine ---------------------------------------------------------------------------
    labelList requested;
    if (args.found("cells"))
    {
        IFstream is(args.get<fileName>("cells"));
        is >> requested;
    }
    else if (args.found("box"))
    {
        // read as two points rather than as a boundBox: boundBox's own Istream operator wants a form
        // argList does not hand it back cleanly from a command-line string
        IStringStream bis(args.get<string>("box"));
        point lo, hi;
        bis >> lo >> hi;
        const boundBox bb(lo, hi);
        DynamicList<label> sel;
        forAll(mesh.cellCentres(), celli)
        {
            if (bb.contains(mesh.cellCentres()[celli])) sel.append(celli);
        }
        requested = sel;
    }
    else
    {
        requested = labelList(1, label(0));
    }
    Info<< "requested " << requested.size() << " cells" << endl;

    // THE 2:1 CLOSURE FIRST, so the set handed to setRefinement is a legal one. brae has this function
    // already (consistentRefinement, gated by refine_candidates_vs_openfoam), and the RESULT is what
    // both codes are given -- so this gate is about setRefinement and not about the closure.
    labelList cellsToRefine(meshCutter.consistentRefinement(requested, true));
    Info<< "consistent set " << cellsToRefine.size() << " cells" << endl;
    {
        OFstream cos(cellsOut);
        cos << cellsToRefine << endl;
    }

    // ---- EARLIER REFINEMENTS, when -times > 1 -----------------------------------------------------
    // WHY THIS EXISTS. A mesh straight out of blockMesh has every level 0, and at level 0 every one of
    // setRefinement's level formulas collapses to the same number: faceAnchorLevel+1, the master point's
    // level+1 and max(edge end levels)+1 are all 1. So a FIRST refinement cannot witness any of them --
    // measured, three fail-proofs on brae's port of those very lines all stayed green. Refining twice
    // makes the levels vary, and the second refinement is then a real test of the arithmetic.
    //
    // The mesh as it stood before the LAST refinement is written out, with its cellLevel and pointLevel,
    // so brae can read exactly that state and do one refinement from it -- brae cannot yet produce the
    // intermediate mesh itself (that needs section 9 and updateMesh), and waiting for it would mean the
    // arithmetic stayed ungated until then.
    const label nTimes = args.getOrDefault<label>("times", 1);
    for (label pass = 1; pass < nTimes; ++pass)
    {
        polyTopoChange mod(mesh);
        meshCutter.setRefinement(cellsToRefine, mod);
        autoPtr<mapPolyMesh> m2(mod.changeMesh(mesh, false));
        mesh.updateMesh(*m2);
        meshCutter.updateMesh(*m2);
        Info<< "pass " << pass << ": " << mesh.nCells() << " cells" << endl;
        // the next pass refines the CHILDREN of what was just refined, so the levels stay non-uniform
        // and the 2:1 closure has something to do
        DynamicList<label> next;
        forAll(meshCutter.cellLevel(), celli)
        {
            if (meshCutter.cellLevel()[celli] == pass) next.append(celli);
        }
        requested = next;
        cellsToRefine = meshCutter.consistentRefinement(requested, true);
        Info<< "pass " << (pass+1) << " requested " << requested.size()
            << ", consistent " << cellsToRefine.size() << endl;
        {
            OFstream cos(cellsOut);
            cos << cellsToRefine << endl;
        }
    }
    // the state brae will read: this mesh and these levels, before the last refinement
    if (args.found("meshOut"))
    {
        mesh.setInstance("constant");
        mesh.polyMesh::write();
        Info<< "wrote the pre-refinement mesh: " << mesh.nCells() << " cells, "
            << mesh.nFaces() << " faces, " << mesh.nPoints() << " points" << endl;
    }
    // ...and the levels it stood at, which are the INPUT to the refinement dumped below. Written into
    // the dump rather than as files beside the mesh: hexRef8 owns them as labelIOList and the reader on
    // brae's side already parses this file.
    const labelList preCellLevel(meshCutter.cellLevel());
    const labelList prePointLevel(meshCutter.pointLevel());
    // ...and the HISTORY as it stood, for the same reason the levels are dumped: after an earlier
    // refinement it is NOT the fresh identity, and brae cannot reconstruct it from the mesh alone.
    labelList preHistoryVisible(meshCutter.history().visibleCells());
    labelList preHistoryParent(meshCutter.history().splitCells().size());
    labelListList preHistoryAdded(meshCutter.history().splitCells().size());
    forAll(meshCutter.history().splitCells(), i)
    {
        const auto& sc = meshCutter.history().splitCells()[i];
        preHistoryParent[i] = sc.parent_;
        if (sc.addedCellsPtr_)
        {
            preHistoryAdded[i].setSize(8);
            forAll(sc.addedCellsPtr_(), j) preHistoryAdded[i][j] = sc.addedCellsPtr_()[j];
        }
    }
    const label nPreCells = mesh.nCells();
    const label nPreFaces = mesh.nFaces();
    const label nPrePoints = mesh.nPoints();

    // ---- OpenFOAM's own refinement ----------------------------------------------------------------
    polyTopoChange meshMod(mesh);
    const labelListList cellAddedCells(meshCutter.setRefinement(cellsToRefine, meshMod));

    // the levels AFTER setRefinement but BEFORE changeMesh: they are indexed by the pre-change
    // numbering, which is what brae's own port produces at the same point
    const labelList cellLevelAfterSet(meshCutter.cellLevel());
    const labelList pointLevelAfterSet(meshCutter.pointLevel());

    autoPtr<mapPolyMesh> mapPtr(meshMod.changeMesh(mesh, false));
    const mapPolyMesh& map = mapPtr();

    // ---- the dump ---------------------------------------------------------------------------------
    OFstream os(outFile);
    os.precision(17);
    os << "nOldPoints " << nPrePoints << nl;
    os << "nOldFaces " << nPreFaces << nl;
    os << "nOldCells " << nPreCells << nl;
    os << "nPoints " << mesh.nPoints() << nl;
    os << "nFaces " << mesh.nFaces() << nl;
    os << "nInternalFaces " << mesh.nInternalFaces() << nl;
    os << "nCells " << mesh.nCells() << nl;

    writeLabels(os, "preCellLevel", preCellLevel);
    writeLabels(os, "prePointLevel", prePointLevel);
    writeLabels(os, "preHistoryVisibleCells", preHistoryVisible);
    writeLabels(os, "preHistoryParent", preHistoryParent);
    writeListList(os, "preHistoryAddedCells", preHistoryAdded);
    writeLabels(os, "cellsToRefine", cellsToRefine);
    writeListList(os, "cellAddedCells", cellAddedCells);
    writeLabels(os, "cellLevelAfterSet", cellLevelAfterSet);
    writeLabels(os, "pointLevelAfterSet", pointLevelAfterSet);

    writeLabels(os, "pointMap", map.pointMap());
    writeLabels(os, "faceMap", map.faceMap());
    writeLabels(os, "cellMap", map.cellMap());
    writeLabels(os, "reversePointMap", map.reversePointMap());
    writeLabels(os, "reverseFaceMap", map.reverseFaceMap());
    writeLabels(os, "reverseCellMap", map.reverseCellMap());
    writeLabels(os, "oldPatchStarts", map.oldPatchStarts());
    writeLabels(os, "oldPatchSizes", map.oldPatchSizes());
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

    // the mesh the refinement produced
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
    // ...and the levels of the mesh that now exists, which is what hexRef8::updateMesh leaves behind
    writeLabels(os, "cellLevelFinal", meshCutter.cellLevel());
    writeLabels(os, "pointLevelFinal", meshCutter.pointLevel());
    // ...and the REFINEMENT HISTORY, which is active even on a mesh that was never refined: its
    // constructor makes visibleCells the identity when there is no file, and active_ follows from that.
    // `parent` is splitCells_[i].parent_ and `addedCells` its eight children (empty where it has none).
    {
        const refinementHistory& h = meshCutter.history();
        os << "historyActive " << (h.active() ? 1 : 0) << nl;
        writeLabels(os, "historyVisibleCells", h.visibleCells());
        os << "historyParent " << h.splitCells().size();
        for (const auto& sc : h.splitCells()) os << ' ' << sc.parent_;
        os << nl;
        os << "historyAddedCells " << h.splitCells().size() << nl;
        for (const auto& sc : h.splitCells())
        {
            if (sc.addedCellsPtr_)
            {
                os << "  8";
                for (const label v : sc.addedCellsPtr_()) os << ' ' << v;
            }
            else
            {
                os << "  0";
            }
            os << nl;
        }
    }

    Info<< "wrote " << outFile << nl
        << "cells " << nOldCells << " -> " << mesh.nCells()
        << ", faces " << nOldFaces << " -> " << mesh.nFaces()
        << ", points " << nOldPoints << " -> " << mesh.nPoints() << endl;

    Info<< "End" << nl << endl;
    return 0;
}
