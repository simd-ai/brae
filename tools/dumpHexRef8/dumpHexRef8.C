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
#include "removeFacesDump.H"
#include "refinementHistory.H"
#include "removeFaces.H"
#include "unitConversion.H"
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
    argList::addBoolOption("unrefine", "after the refinements, UNREFINE the split points and dump that");
    argList::addOption("unrefineStride", "int",
                       "unrefine only every Nth split point, so survivors are RENUMBERED");

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

    // ---- THE UNREFINEMENT ARM ---------------------------------------------------------------------
    // Refine first (the passes above have already run), then unrefine the split points -- which is the
    // only way to reach a change that REMOVES cells, and so the only way the reverse maps stop being the
    // identity. A refinement leaves them the identity, measured, so unit 6's remapping half cannot be
    // gated without this.
    if (args.found("unrefine"))
    {
        // one refinement first, so there is something to undo
        {
            polyTopoChange mod(mesh);
            meshCutter.setRefinement(cellsToRefine, mod);
            autoPtr<mapPolyMesh> m2(mod.changeMesh(mesh, false));
            mesh.updateMesh(*m2);
            meshCutter.updateMesh(*m2);
        }
        if (args.found("meshOut"))
        {
            mesh.setInstance("constant");
            mesh.polyMesh::write();
            Info<< "wrote the pre-unrefinement mesh: " << mesh.nCells() << " cells" << endl;
        }
        const labelList uPreCellLevel(meshCutter.cellLevel());
        const labelList uPrePointLevel(meshCutter.pointLevel());
        labelList uPreHistVisible(meshCutter.history().visibleCells());
        labelList uPreHistParent(meshCutter.history().splitCells().size());
        labelListList uPreHistAdded(meshCutter.history().splitCells().size());
        forAll(meshCutter.history().splitCells(), i)
        {
            const auto& sc = meshCutter.history().splitCells()[i];
            uPreHistParent[i] = sc.parent_;
            if (sc.addedCellsPtr_)
            {
                uPreHistAdded[i].setSize(8);
                forAll(sc.addedCellsPtr_(), j) uPreHistAdded[i][j] = sc.addedCellsPtr_()[j];
            }
        }
        const label uNOldCells = mesh.nCells();
        const label uNOldFaces = mesh.nFaces();
        const label uNOldPoints = mesh.nPoints();

        // which points can be unsplit, through OpenFOAM's own two steps -- both already ported in brae
        // and gated elsewhere, so this arm is about setUnrefinement and not about the selection
        labelList allSplit(meshCutter.getSplitPoints());
        // UNREFINING EVERY SPLIT POINT PUTS THE MESH BACK EXACTLY, and then every surviving cell keeps its
        // own index: the children that go are the ones appended at the end, so reverseCellMap is the
        // IDENTITY on its live part and the remap is indistinguishable from a truncation. MEASURED --
        // fail-proofs on updateLevels' remap and historyUpdateMesh's renumber both stayed green on such an
        // arm. Unrefining only every Nth split point leaves removed cells INTERLEAVED with survivors, and
        // then the survivors really are renumbered.
        const label stride = args.getOrDefault<label>("unrefineStride", 1);
        if (stride > 1)
        {
            DynamicList<label> some;
            forAll(allSplit, i)
            {
                if ((i % stride) == 0) some.append(allSplit[i]);
            }
            Info<< "unrefineStride " << stride << ": " << some.size() << " of "
                << allSplit.size() << " split points" << endl;
            allSplit = some;
        }
        const labelList splitPoints(meshCutter.consistentUnrefinement(allSplit, false));
        Info<< "unrefine: " << allSplit.size() << " split points, "
            << splitPoints.size() << " consistent" << endl;

        // compatibleRemoves is PUBLIC on removeFaces, so its three outputs can be dumped and gated on
        // their own rather than only through the mesh they produce
        // GREAT is what hexRef8 constructs ITS faceRemover with (hexRef8.C:1967, "merge boundary faces
        // wherever possible"), so that is what this probe uses. compatibleRemoves does not read minCos_
        // at all; setRefinement does, and the value decides whether its feature-angle guard runs.
        removeFaces faceRemover(mesh, GREAT);
        labelHashSet splitFaces(12*splitPoints.size());
        for (const label pointi : splitPoints)
        {
            splitFaces.insert(mesh.pointFaces()[pointi]);
        }
        // THE ORDER THE SET IS PASSED IN IS PART OF THE ANSWER, so it is dumped rather than left to be
        // guessed: region 0 is the region the FIRST face of this list created, so cellRegion's values
        // and cellRegionMaster's indices follow labelHashSet's iteration order. hexRef8 itself passes
        // splitFaces.toc() (hexRef8.C:5673), and brae has no labelHashSet, so the gate hands brae THIS
        // list and compares the numbering exactly. Nothing in the final mesh depends on it -- hexRef8
        // overwrites each region's master with its own, and removeFaces walks regions, not region
        // labels -- but a comparison of cellRegion does.
        const labelList splitFacesToc(splitFaces.toc());
        labelList cellRegion, cellRegionMaster, facesToRemove;
        const label nUsedRegions = faceRemover.compatibleRemoves
        (
            splitFacesToc, cellRegion, cellRegionMaster, facesToRemove
        );

        // ...AND A SECOND CALL ON A DELIBERATELY REDUCED SET, because on hexRef8's own set the recount
        // finds nothing new: the twelve faces at a split point ARE the twelve internal faces of its
        // block, and hexRef8 FatalErrors if the recount returns any other number (hexRef8.C:5686-5710,
        // "can be more (but never less) than splitFaces provided"). So the walk over the internal faces
        // is indistinguishable from `newFacesToRemove = facesToRemove` on every arm this tool can build
        // through setUnrefinement. Dropping the lowest-numbered face of each block gives it something to
        // find: the block stays connected through the other eleven, so the region is unchanged and the
        // recount must put the dropped face back. Not fed to setUnrefinement -- it is a direct call on
        // OpenFOAM's own function, which is what makes it a usable oracle.
        labelList droppedFaces;
        labelList dCellRegion, dCellRegionMaster, dFacesToRemove;
        label nUsedRegionsDropped = 0;
        {
            labelHashSet drop(splitPoints.size());
            for (const label pointi : splitPoints)
            {
                label lowest = -1;
                for (const label facei : mesh.pointFaces()[pointi])
                {
                    if (splitFaces.found(facei) && (lowest == -1 || facei < lowest)) lowest = facei;
                }
                if (lowest != -1) drop.insert(lowest);
            }
            DynamicList<label> kept(splitFacesToc.size());
            for (const label facei : splitFacesToc)
            {
                if (!drop.found(facei)) kept.append(facei);
            }
            droppedFaces.transfer(kept);
            nUsedRegionsDropped = faceRemover.compatibleRemoves
            (
                droppedFaces, dCellRegion, dCellRegionMaster, dFacesToRemove
            );
            Info<< "compatibleRemoves on a reduced set: " << droppedFaces.size() << " faces in, "
                << dFacesToRemove.size() << " out (the full set gives " << splitFacesToc.size()
                << " in, " << facesToRemove.size() << " out)" << endl;
        }

        // UNIT 6b-3's ORACLE. removeFaces::setRefinement decides which EDGES go, which FACES merge into
        // which, which POINTS go and which faces are affected -- and every one of those is a LOCAL. So a
        // second, INSTRUMENTED copy of OpenFOAM's own class (removeFacesDump.C, writes only) is called on
        // the SAME inputs hexRef8 hands it: compatibleRemoves' three outputs, unchanged. hexRef8's "Redo
        // the region master" block is a CHECK and not a change -- it FatalErrors unless the master
        // already is min(pointCells) -- so there is nothing between compatibleRemoves and setRefinement
        // for this call to miss.
        //
        // It writes into a polyTopoChange of ITS OWN and that change is never played: the resulting mesh
        // comes from the real path below, which is what brae's actions are compared against end to end.
        {
            OFstream rfos(outFile + ".removeFaces");
            rfos.precision(17);
            rfDump::os = &rfos;
            removeFacesDump instrumented(mesh, GREAT);
            polyTopoChange rfMod(mesh);
            instrumented.setRefinement(facesToRemove, cellRegion, cellRegionMaster, rfMod);
            rfDump::os = nullptr;
            // ...AND A SECOND CALL AT cos(45 deg), because at GREAT the feature-angle guard is
            // UNREACHABLE: `minCos_ < 1 && minCos_ > -1` (removeFaces.C:997) is false at 1e15, so the
            // only thing that stops a boundary merge on hexRef8's own path is a patch boundary. brae has
            // to transcribe the guard all the same -- removeFaces has other callers -- and this arm is
            // what holds it against OpenFOAM. Written to its own file; never played.
            OFstream rfos45(outFile + ".removeFaces45");
            rfos45.precision(17);
            rfDump::os = &rfos45;
            removeFacesDump instrumented45(mesh, Foam::cos(degToRad(45.0)));
            polyTopoChange rfMod45(mesh);
            instrumented45.setRefinement(facesToRemove, cellRegion, cellRegionMaster, rfMod45);
            rfDump::os = nullptr;
            Info<< "wrote " << outFile << ".removeFaces (removeFaces::setRefinement's own decisions)"
                << endl;
        }

        polyTopoChange uMod(mesh);
        meshCutter.setUnrefinement(splitPoints, uMod);
        const labelList uCellLevelAfterSet(meshCutter.cellLevel());
        const labelList uPointLevelAfterSet(meshCutter.pointLevel());
        labelList uHistVisible(meshCutter.history().visibleCells());
        labelList uHistParent(meshCutter.history().splitCells().size());
        labelListList uHistAdded(meshCutter.history().splitCells().size());
        forAll(meshCutter.history().splitCells(), i)
        {
            const auto& sc = meshCutter.history().splitCells()[i];
            uHistParent[i] = sc.parent_;
            if (sc.addedCellsPtr_)
            {
                uHistAdded[i].setSize(8);
                forAll(sc.addedCellsPtr_(), j) uHistAdded[i][j] = sc.addedCellsPtr_()[j];
            }
        }
        autoPtr<mapPolyMesh> uMapPtr(uMod.changeMesh(mesh, false));
        const mapPolyMesh& uMap = uMapPtr();
        meshCutter.updateMesh(uMap);

        OFstream uos(outFile);
        uos.precision(17);
        uos << "mode unrefine" << nl;
        uos << "nOldPoints " << uNOldPoints << nl;
        uos << "nOldFaces " << uNOldFaces << nl;
        uos << "nOldCells " << uNOldCells << nl;
        uos << "nPoints " << mesh.nPoints() << nl;
        uos << "nFaces " << mesh.nFaces() << nl;
        uos << "nInternalFaces " << mesh.nInternalFaces() << nl;
        uos << "nCells " << mesh.nCells() << nl;
        writeLabels(uos, "preCellLevel", uPreCellLevel);
        writeLabels(uos, "prePointLevel", uPrePointLevel);
        writeLabels(uos, "preHistoryVisibleCells", uPreHistVisible);
        writeLabels(uos, "preHistoryParent", uPreHistParent);
        writeListList(uos, "preHistoryAddedCells", uPreHistAdded);
        writeLabels(uos, "allSplitPoints", allSplit);
        writeLabels(uos, "splitPoints", splitPoints);
        writeLabels(uos, "splitFaces", splitFaces.sortedToc());
        writeLabels(uos, "splitFacesToc", splitFacesToc);
        writeLabels(uos, "cellRegion", cellRegion);
        writeLabels(uos, "cellRegionMaster", cellRegionMaster);
        writeLabels(uos, "facesToRemove", facesToRemove);
        uos << "nUsedRegions " << nUsedRegions << nl;
        writeLabels(uos, "splitFacesDropped", droppedFaces);
        writeLabels(uos, "cellRegionDropped", dCellRegion);
        writeLabels(uos, "cellRegionMasterDropped", dCellRegionMaster);
        writeLabels(uos, "facesToRemoveDropped", dFacesToRemove);
        uos << "nUsedRegionsDropped " << nUsedRegionsDropped << nl;
        writeLabels(uos, "cellLevelAfterSet", uCellLevelAfterSet);
        writeLabels(uos, "pointLevelAfterSet", uPointLevelAfterSet);
        writeLabels(uos, "historyVisibleCellsAfterSet", uHistVisible);
        writeLabels(uos, "historyParentAfterSet", uHistParent);
        writeListList(uos, "historyAddedCellsAfterSet", uHistAdded);
        writeLabels(uos, "pointMap", uMap.pointMap());
        writeLabels(uos, "faceMap", uMap.faceMap());
        writeLabels(uos, "cellMap", uMap.cellMap());
        writeLabels(uos, "reversePointMap", uMap.reversePointMap());
        writeLabels(uos, "reverseFaceMap", uMap.reverseFaceMap());
        writeLabels(uos, "reverseCellMap", uMap.reverseCellMap());
        {
            labelList flips(uMap.flipFaceFlux().sortedToc());
            writeLabels(uos, "flipFaceFlux", flips);
        }
        writeObjectMaps(uos, "cellsFromCellsMap", uMap.cellsFromCellsMap());
        writeObjectMaps(uos, "facesFromFacesMap", uMap.facesFromFacesMap());
        writeObjectMaps(uos, "pointsFromPointsMap", uMap.pointsFromPointsMap());
        // ...AND THE MESH ITSELF, which the refinement branch already dumps and this one did not: unit
        // 6b-3's answer IS the mesh an unrefinement leaves, so the comparison needs it.
        uos << "points " << mesh.points().size() << nl;
        for (const point& pt : mesh.points())
        {
            uos << "  " << pt.x() << ' ' << pt.y() << ' ' << pt.z() << nl;
        }
        uos << "faces " << mesh.faces().size() << nl;
        for (const face& f : mesh.faces())
        {
            uos << "  " << f.size();
            for (const label v : f) uos << ' ' << v;
            uos << nl;
        }
        writeLabels(uos, "owner", mesh.faceOwner());
        writeLabels(uos, "neighbour", mesh.faceNeighbour());
        uos << "patches " << mesh.boundaryMesh().size() << nl;
        for (const polyPatch& pp : mesh.boundaryMesh())
        {
            uos << "  " << pp.name() << ' ' << pp.start() << ' ' << pp.size() << nl;
        }
        writeLabels(uos, "cellLevelFinal", meshCutter.cellLevel());
        writeLabels(uos, "pointLevelFinal", meshCutter.pointLevel());
        {
            const refinementHistory& h = meshCutter.history();
            uos << "historyActive " << (h.active() ? 1 : 0) << nl;
            writeLabels(uos, "historyVisibleCells", h.visibleCells());
            uos << "historyParent " << h.splitCells().size();
            for (const auto& sc : h.splitCells()) uos << ' ' << sc.parent_;
            uos << nl;
            uos << "historyAddedCells " << h.splitCells().size() << nl;
            for (const auto& sc : h.splitCells())
            {
                if (sc.addedCellsPtr_)
                {
                    uos << "  8";
                    for (const label v : sc.addedCellsPtr_()) uos << ' ' << v;
                }
                else { uos << "  0"; }
                uos << nl;
            }
        }
        Info<< "wrote " << outFile << nl
            << "cells " << uNOldCells << " -> " << mesh.nCells()
            << ", faces " << uNOldFaces << " -> " << mesh.nFaces()
            << ", points " << uNOldPoints << " -> " << mesh.nPoints() << endl;
        Info<< "End" << nl << endl;
        return 0;
    }

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
