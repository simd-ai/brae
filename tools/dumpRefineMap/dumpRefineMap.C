/*---------------------------------------------------------------------------*\
    brae instrument, not an OpenFOAM application.

    The FIELD MAPPING half of OpenFOAM's dynamicRefineFvMesh: the mapPolyMesh
    that a refine or an unrefine step produces, and the cell values every
    registered field carries once fvMesh::mapFields has run against it.

        dynamicRefineFvMesh::refine      dynamicRefineFvMesh.C:442-535
        dynamicRefineFvMesh::unrefine    dynamicRefineFvMesh.C:537-716
        dynamicRefineFvMesh::mapFields   dynamicRefineFvMesh.C:197-440
        fvMesh::updateMesh               fvMesh.C:1012-1091
        fvMesh::mapFields                fvMesh.C:838-940

    Why this exists: the solver goes on to advance the fields after update()
    returns, so the fields written into a <time> directory are the POST-SOLVE
    state and cannot witness the mapping. The only place the post-map state
    exists is inside the step, which is where this instrument reads it.

    HOW THE MAP IS OBTAINED, and whether it changes any arithmetic.

    dynamicRefineFvMesh::update() does not return the map, and updateTopology()
    drops unrefine()'s return value on the floor (dynamicRefineFvMesh.C:1440).
    But refine() and unrefine() are themselves `virtual` protected members
    (dynamicRefineFvMesh.H:126,129), so a derived mesh can override them, call
    OpenFOAM's own implementation by its qualified name, and read the autoPtr it
    hands back. That is what this instrument does. It changes no arithmetic:

      - the override body is `dynamicRefineFvMesh::refine(arg)` and nothing
        else, so the whole topology change, the field mapping and the flux
        correction are OpenFOAM's, executed in OpenFOAM's order;
      - everything printed is read from the returned map and from the mesh
        AFTER the base call has finished, so no print can be seen by the code
        being measured;
      - the extra cellMapper built below is a second, throwaway instance of the
        class fvMeshMapper already built inside fvMesh::mapFields
        (fvMeshMapper.H:118 volMap() returns its cellMapper). It is read-only
        and it is constructed from the same mapPolyMesh against the same
        post-change mesh, so it reproduces the addressing the mapping used.

    No OpenFOAM file is edited, and no OpenFOAM class is copied.

    TWO TRAPS IN THE SETUP, both of which make update() silently do nothing or
    abort, and neither of which is visible in the answer.

    1. dynamicRefineFvMesh::updateTopology() refines only when
           time().timeIndex() > 0 && time().timeIndex() % refineInterval == 0
       (dynamicRefineFvMesh.C:1317). timeSelector::select0 with -time <t>
       returns a ONE-entry list, so the usual `setTime(timeDirs.last(),
       timeDirs.size() - 1)` leaves timeIndex 0 and the step is skipped without
       a word. The index used here is the case's own: Time::setTime(instant,
       label) reads <t>/uniform/time and takes its `index` entry over the
       argument (Time.C:1060-1069), which is the timeIndex the solver held when
       it wrote that directory. Both it and the index update() is called at are
       printed, so a silent skip cannot be mistaken for a step that did
       nothing.

    2. fvMesh::updateMesh only creates V0 when VPtr_ already exists and when
       curTimeIndex_ < time().timeIndex() (fvMesh.C:1022-1026, storeOldVol
       fvMesh.C:164-167), and dynamicRefineFvMesh::mapFields then calls V0()
       unconditionally (dynamicRefineFvMesh.C:228), which aborts with "V0 is
       not available" if it was not created. A freshly built mesh has
       curTimeIndex_ == time().timeIndex() and no V. So this instrument does
       what the solver does: it builds the mesh at the index of the directory it
       reads, touches V(), and only then advances the time INDEX by one -- the
       runTime++ that precedes mesh.update() in the solver. The time NAME is
       left alone, because moving it would send the mesh reader to the next
       directory.

    The state read at <t> is therefore the state the solver held when it called
    update() for the step that FOLLOWS <t>: -time 0.002 measures the mesh change
    the log reports under "Time = 0.003".

    Old-time fields are not registered here. MapGeometricFields only stores an
    old time that already exists (MapGeometricFields.H:98), and the mapping of
    an oldTime field is the same operation on the same mapper, so registering
    alpha.water_0 would add a copy of a row, not a new oracle.

    Usage:  dumpRefineMap -case <dir> -time <t> [-field alpha.water]
                [-timeIndex <n>]

    Writes bare space-separated rows at setprecision(18), prefixed [brae]:
        [brae] time <name>
        [brae] timeDirIndex <n>                  position in the time series
        [brae] timeIndexAtConstruction <n>       what curTimeIndex_ becomes
        [brae] timeIndexAtUpdate <n>             what updateTopology tests
        [brae] refineInterval <n>
        [brae] field <name>
        [brae] hasU|hasP_rgh <true|false>
        [brae] pre  nCells|nPoints|nFaces|nInternalFaces <n>
        [brae] pre  cellLevel|pointLevel <i> <level>
        [brae] pre  V <celli> <value>
        [brae] pre  <fieldName> <celli> <value...>
        [brae] pre  <fieldName>Patch <patchi> <facei> <value...>
        [brae] updateReturn <0|1>
        [brae] nCaptured <n>
        [brae] map <k> phase <refine|unrefine>
        [brae] map <k> nOldCells|nOldFaces|nOldPoints <n>
        [brae] map <k> nCells|nFaces|nPoints <n>
        [brae] map <k> cellMap|pointMap|faceMap <newi> <oldi>
        [brae] map <k> reverseCellMap|reversePointMap|reverseFaceMap <oldi> <v>
        [brae] map <k> cellsFromCells <newi> <nMasters> <old...>
        [brae] map <k> ncellsFromCells <n>
        [brae] map <k> cellsFromFaces|cellsFromEdges|cellsFromPoints <newi> ...
        [brae] map <k> facesFromFaces|facesFromEdges|facesFromPoints <newi> ...
        [brae] map <k> pointsFromPoints <newi> <nMasters> <old...>
        [brae] map <k> oldCellVolumes <oldi> <value>
        [brae] map <k> flipFaceFlux <facei>
        [brae] map <k> nFlipFaceFlux <n>
        [brae] map <k> hasOldCellVolumes|hasMotionPoints <true|false>
        [brae] map <k> oldPatchStarts|oldPatchSizes <patchi> <n>
        [brae] map <k> oldPatchNMeshPoints <patchi> <n>
        [brae] map <k> cellMapperDirect <true|false>
        [brae] map <k> cellMapperSize|cellMapperSizeBeforeMapping <n>
        [brae] map <k> cellMapperHasUnmapped <true|false>
        [brae] map <k> cellDirectAddressing <newi> <oldi>   (direct only)
        [brae] map <k> cellAddressing <newi> <n> <old...>   (interp only)
        [brae] map <k> cellWeights <newi> <n> <w...>        (interp only)
        [brae] map <k> cellInserted <newi>
        [brae] map <k> nCellInserted <n>
        [brae] post nCells|nPoints|nFaces|nInternalFaces <n>
        [brae] post cellLevel|pointLevel <i> <level>
        [brae] post V <celli> <value>
        [brae] post V0 <celli> <value>
        [brae] post <fieldName> <celli> <value...>
        [brae] post <fieldName>Patch <patchi> <facei> <value...>
        [brae] END
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "timeSelector.H"
#include "IOmanip.H"
#include "IOdictionary.H"
#include "Switch.H"
#include "volFields.H"
#include "cellMapper.H"
#include "mapPolyMesh.H"
#include "objectMap.H"
#include "dynamicRefineFvMesh.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * printing * * * * * * * * * * * * * * * * * //

// A bare space-separated row per entry, not OpenFOAM's "4877 ( a b ... )", so a
// gate can split on whitespace without stripping the list length and brackets.
static void writeLabels
(
    const std::string& tag,
    const labelUList& lst
)
{
    forAll(lst, i)
    {
        Info<< "[brae] " << tag.c_str() << ' ' << i << ' ' << lst[i] << nl;
    }
}


static void writeScalars
(
    const std::string& tag,
    const scalarField& f
)
{
    forAll(f, i)
    {
        Info<< "[brae] " << tag.c_str() << ' ' << i << ' ' << f[i] << nl;
    }
}


// The objectMap lists (cellsFromCells and friends). The row carries its own
// master count so a reader never has to guess where the list ends.
static void writeObjectMaps
(
    const std::string& prefix,
    const char* const name,
    const List<objectMap>& maps
)
{
    for (const objectMap& om : maps)
    {
        Info<< "[brae] " << prefix.c_str() << ' ' << name << ' '
            << om.index() << ' ' << om.masterObjects().size();

        for (const label m : om.masterObjects())
        {
            Info<< ' ' << m;
        }

        Info<< nl;
    }

    // The count is on its own row because an EMPTY list is the answer on most
    // of these eight, and a reader has to be able to tell an empty list from a
    // list that was never printed.
    Info<< "[brae] " << prefix.c_str() << " n" << name << ' '
        << maps.size() << nl;
}


// The cell values and then the patch values of one field. Both halves are
// mapped -- the internal field through cellMapper, every patch through
// fvPatchField::autoMap -- so a port that only reproduces the cells is half
// checked.
template<class Type>
static void writeGeometricField
(
    const std::string& prefix,
    const GeometricField<Type, fvPatchField, volMesh>& fld
)
{
    const label nCmpt = pTraits<Type>::nComponents;

    forAll(fld, celli)
    {
        Info<< "[brae] " << prefix.c_str() << ' ' << fld.name() << ' '
            << celli;

        for (label cmpt = 0; cmpt < nCmpt; ++cmpt)
        {
            Info<< ' ' << component(fld[celli], cmpt);
        }

        Info<< nl;
    }

    const auto& bf = fld.boundaryField();

    forAll(bf, patchi)
    {
        forAll(bf[patchi], facei)
        {
            Info<< "[brae] " << prefix.c_str() << ' ' << fld.name()
                << "Patch " << patchi << ' ' << facei;

            for (label cmpt = 0; cmpt < nCmpt; ++cmpt)
            {
                Info<< ' ' << component(bf[patchi][facei], cmpt);
            }

            Info<< nl;
        }
    }
}


// * * * * * * * * * * * * * * * * the mesh * * * * * * * * * * * * * * * * * //

class refineMapProbeFvMesh
:
    public dynamicRefineFvMesh
{
    // Private Data

        //- How many maps have been seen in this update()
        label nCaptured_;


    // Private Member Functions

        //- Everything the returned map holds, plus the cell addressing that
        //  fvMesh::mapFields derived from it. Read-only on the mesh and on
        //  the map; the only thing it writes is its own capture counter.
        void report(const word& phase, const mapPolyMesh& map);


public:

    // doInit MUST be false here, then init(true): that is the order
    // dynamicFvMesh::New uses (dynamicFvMeshNew.C:84-87). Constructing with
    // the default doInit=true instead runs dynamicMotionSolverListFvMesh's own
    // constructor-init, which is the MANDATORY one (init(doInit) ->
    // init(doInit, true), dynamicMotionSolverListFvMesh.C:137-141) and demands
    // a `motionSolver` entry a pure-refinement dynamicMeshDict does not have.
    // dynamicRefineFvMesh::init passes mandatory=false (:1106).
    explicit refineMapProbeFvMesh(const IOobject& io)
    :
        dynamicRefineFvMesh(io, false),
        nCaptured_(0)
    {
        init(true);
    }

    label nCaptured() const noexcept
    {
        return nCaptured_;
    }


protected:

    //- OpenFOAM's refine, then a read of the map it returns
    virtual autoPtr<mapPolyMesh> refine(const labelList& cellsToRefine)
    {
        autoPtr<mapPolyMesh> map
        (
            dynamicRefineFvMesh::refine(cellsToRefine)
        );

        report("refine", map());

        return map;
    }

    //- OpenFOAM's unrefine, then a read of the map it returns. updateTopology
    //  discards this map (:1440); it is the only copy there is.
    virtual autoPtr<mapPolyMesh> unrefine(const labelList& splitPoints)
    {
        autoPtr<mapPolyMesh> map
        (
            dynamicRefineFvMesh::unrefine(splitPoints)
        );

        report("unrefine", map());

        return map;
    }
};


void refineMapProbeFvMesh::report
(
    const word& phase,
    const mapPolyMesh& map
)
{
    const label k = nCaptured_++;

    const std::string m("map " + std::to_string(k));
    const char* const mc = m.c_str();

    Info<< "[brae] " << mc << " phase " << phase << nl
        << "[brae] " << mc << " nOldCells " << map.nOldCells() << nl
        << "[brae] " << mc << " nOldFaces " << map.nOldFaces() << nl
        << "[brae] " << mc << " nOldPoints " << map.nOldPoints() << nl
        << "[brae] " << mc << " nOldInternalFaces "
        << map.nOldInternalFaces() << nl
        << "[brae] " << mc << " nCells " << nCells() << nl
        << "[brae] " << mc << " nFaces " << nFaces() << nl
        << "[brae] " << mc << " nInternalFaces " << nInternalFaces() << nl
        << "[brae] " << mc << " nPoints " << nPoints() << nl;

    writeLabels(m + " cellMap", map.cellMap());
    writeLabels(m + " pointMap", map.pointMap());
    writeLabels(m + " faceMap", map.faceMap());
    writeLabels(m + " reverseCellMap", map.reverseCellMap());
    writeLabels(m + " reversePointMap", map.reversePointMap());
    writeLabels(m + " reverseFaceMap", map.reverseFaceMap());

    // The merge lists. cellsFromCells is the one an unrefine fills; the other
    // three are printed because a port has to know they are EMPTY here, and an
    // empty print is the only thing that says so.
    writeObjectMaps(m, "cellsFromCells", map.cellsFromCellsMap());
    writeObjectMaps(m, "cellsFromFaces", map.cellsFromFacesMap());
    writeObjectMaps(m, "cellsFromEdges", map.cellsFromEdgesMap());
    writeObjectMaps(m, "cellsFromPoints", map.cellsFromPointsMap());
    writeObjectMaps(m, "facesFromFaces", map.facesFromFacesMap());
    writeObjectMaps(m, "facesFromEdges", map.facesFromEdgesMap());
    writeObjectMaps(m, "facesFromPoints", map.facesFromPointsMap());
    writeObjectMaps(m, "pointsFromPoints", map.pointsFromPointsMap());

    // oldCellVolumes is what fvMesh::updateMesh hands storeOldVol, so it is
    // the source of V0 and it is sized on the OLD mesh
    // (polyTopoChange.C:3850 takes it from mesh.cellVolumes()).
    Info<< "[brae] " << mc << " hasOldCellVolumes "
        << Switch(map.hasOldCellVolumes()) << nl;

    if (map.hasOldCellVolumes())
    {
        writeScalars(m + " oldCellVolumes", map.oldCellVolumes());
    }

    Info<< "[brae] " << mc << " hasMotionPoints "
        << Switch(map.hasMotionPoints()) << nl;

    // sortedToc, not toc: a HashSet's iteration order is the table's, and an
    // oracle has to be byte-comparable between runs.
    const labelList flipped(map.flipFaceFlux().sortedToc());
    forAll(flipped, i)
    {
        Info<< "[brae] " << mc << " flipFaceFlux " << flipped[i] << nl;
    }
    Info<< "[brae] " << mc << " nFlipFaceFlux " << flipped.size() << nl;

    writeLabels(m + " oldPatchStarts", map.oldPatchStarts());
    writeLabels(m + " oldPatchSizes", map.oldPatchSizes());
    writeLabels(m + " oldPatchNMeshPoints", map.oldPatchNMeshPoints());

    // The cell addressing fvMesh::mapFields actually mapped through: the
    // same class, on the same map, against the same post-change mesh.
    // direct() and addressing()/weights() are mutually exclusive -- each of
    // the two accessors FatalErrors in the other mode
    // (cellMapper.C:222-231, :244-253) -- so the branch is not a convenience.
    const cellMapper cellMap(map);

    Info<< "[brae] " << mc << " cellMapperDirect "
        << Switch(cellMap.direct()) << nl
        << "[brae] " << mc << " cellMapperSize " << cellMap.size() << nl
        << "[brae] " << mc << " cellMapperSizeBeforeMapping "
        << cellMap.sizeBeforeMapping() << nl
        << "[brae] " << mc << " cellMapperHasUnmapped "
        << Switch(cellMap.hasUnmapped()) << nl;

    if (cellMap.direct())
    {
        writeLabels(m + " cellDirectAddressing", cellMap.directAddressing());
    }
    else
    {
        const labelListList& addr = cellMap.addressing();
        const scalarListList& w = cellMap.weights();

        forAll(addr, celli)
        {
            Info<< "[brae] " << mc << " cellAddressing " << celli << ' '
                << addr[celli].size();

            for (const label a : addr[celli])
            {
                Info<< ' ' << a;
            }

            Info<< nl;

            Info<< "[brae] " << mc << " cellWeights " << celli << ' '
                << w[celli].size();

            for (const scalar s : w[celli])
            {
                Info<< ' ' << s;
            }

            Info<< nl;
        }
    }

    // Both counts, because they DISAGREE on an unrefine step and only one of
    // them is the list the mapping used. cellMapper's constructor counts
    // nInsertedObjects_ by unsetting bits at the VALUES of cellMap in a bitSet
    // indexed by NEW cell (cellMapper.C:325), so on a merge it counts cells
    // that calcAddressing then finds are perfectly well mapped, and the list
    // it finally builds is shorter (:322-327). hasUnmapped() reports the
    // constructor's number; the list is the truth about what got no value.
    writeLabels(m + " cellInserted", cellMap.insertedObjectLabels());
    Info<< "[brae] " << mc << " nCellInserted "
        << cellMap.insertedObjectLabels().size() << nl;

    Info<< flush;
}


// * * * * * * * * * * * * * * * * * main * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "the mapPolyMesh of one dynamicRefineFvMesh step, and the cell"
        " values the mapping leaves behind"
    );
    argList::addOption("field", "name", "the volScalarField to refine on");
    argList::addOption
    (
        "timeIndex",
        "label",
        "the time index update() is called at (default: the position of the"
        " selected directory in the case's time series, plus one)"
    );

    timeSelector::addOptions();

    #include "setRootCase.H"
    #include "createTime.H"

    const instantList timeDirs = timeSelector::select0(runTime, args);
    if (timeDirs.empty())
    {
        FatalErrorInFunction << "no time directory selected" << exit(FatalError);
    }
    const instant readInstant(timeDirs.last());

    // Which STEP the selected directory is. The solver's timeIndex at the
    // moment it wrote <t> is this number; the update() that follows runs at
    // this number plus one. It is taken from the case's own time series, not
    // from timeDirs, because -time <t> makes timeDirs one entry long.
    label stepIndex = -1;
    {
        const instantList allTimes(runTime.times());
        label n = 0;

        forAll(allTimes, i)
        {
            if (allTimes[i].name() == runTime.constant())
            {
                continue;
            }

            if (allTimes[i].name() == readInstant.name())
            {
                stepIndex = n;
            }

            ++n;
        }
    }

    if (stepIndex < 0)
    {
        FatalErrorInFunction
            << "cannot place time " << readInstant.name()
            << " in the case's time series" << exit(FatalError);
    }

    // The mesh is read at the SELECTED time and constructed at the index the
    // solver held when it wrote that directory. Both matter: the name so
    // facesInstance resolves to <t>/polyMesh instead of constant/, and the
    // index so that the bump below makes curTimeIndex_ < timeIndex() and V0
    // gets created (see the traps in the header).
    //
    // Note that this overload of setTime OVERWRITES newIndex from the case's
    // own <t>/uniform/time `index` entry (Time.C:1060-1069), so stepIndex is
    // only a fallback and the number below is the solver's own. That is the
    // better authority of the two, so it is kept, not fought.
    runTime.setTime(readInstant, stepIndex);

    const label constructionIndex = runTime.timeIndex();

    const label updateIndex
    (
        args.getOrDefault<label>("timeIndex", constructionIndex + 1)
    );

    Info<< setprecision(18);

    refineMapProbeFvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.timeName(),
            runTime,
            IOobject::MUST_READ
        )
    );

    const dictionary refineDict
    (
        IOdictionary
        (
            IOobject
            (
                "dynamicMeshDict",
                runTime.constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                IOobject::NO_REGISTER
            )
        ).optionalSubDict(dynamicRefineFvMesh::typeName + "Coeffs")
    );

    const word fieldName
    (
        args.getOrDefault<word>("field", refineDict.get<word>("field"))
    );
    const label refineInterval(refineDict.get<label>("refineInterval"));

    // The fields to be mapped, on the mesh's own registry so that
    // fvMesh::mapFields finds them (MapGeometricFields.H:78 walks the
    // registry, and skips anything whose mesh is not the mapper's). The
    // refinement field itself is MUST_READ because updateTopology looks it up
    // by name (:1343) and cannot proceed without it. NO_WRITE throughout:
    // this instrument must not leave fields in the case.
    volScalarField refineField
    (
        IOobject
        (
            fieldName,
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    // ...and the two the solver would also be carrying. Absent ones are
    // reported as absent rather than defaulted, so a run cannot quietly
    // measure fewer fields than the reader thinks.
    autoPtr<volVectorField> UPtr;
    autoPtr<volScalarField> pRghPtr;

    {
        IOobject Uio
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        );

        if (Uio.typeHeaderOk<volVectorField>(true))
        {
            UPtr.reset(new volVectorField(Uio, mesh));
        }

        IOobject pio
        (
            "p_rgh",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        );

        if (pio.typeHeaderOk<volScalarField>(true))
        {
            pRghPtr.reset(new volScalarField(pio, mesh));
        }
    }

    Info<< "[brae] time " << runTime.timeName() << nl
        << "[brae] timeDirIndex " << stepIndex << nl
        << "[brae] timeIndexAtConstruction " << constructionIndex << nl
        << "[brae] timeIndexAtUpdate " << updateIndex << nl
        << "[brae] refineInterval " << refineInterval << nl
        << "[brae] field " << fieldName << nl
        << "[brae] hasU " << Switch(bool(UPtr)) << nl
        << "[brae] hasP_rgh " << Switch(bool(pRghPtr)) << nl;

    // PRE-update state.
    Info<< "[brae] pre nCells " << mesh.nCells() << nl
        << "[brae] pre nPoints " << mesh.nPoints() << nl
        << "[brae] pre nFaces " << mesh.nFaces() << nl
        << "[brae] pre nInternalFaces " << mesh.nInternalFaces() << nl;

    writeLabels("pre cellLevel", mesh.meshCutter().cellLevel());
    writeLabels("pre pointLevel", mesh.meshCutter().pointLevel());

    // Touching V() here is not only a print: fvMesh::updateMesh skips
    // storeOldVol entirely when VPtr_ is null (fvMesh.C:1024), and
    // dynamicRefineFvMesh::mapFields then aborts on V0() (:228). The solver
    // has touched V long before it calls update(); this is that touch.
    writeScalars("pre V", mesh.V().field());

    writeGeometricField("pre", refineField);
    if (UPtr)
    {
        writeGeometricField("pre", UPtr());
    }
    if (pRghPtr)
    {
        writeGeometricField("pre", pRghPtr());
    }

    // The runTime++ that precedes mesh.update() in the solver, with the time
    // NAME held still so nothing re-reads from the next directory. This is the
    // scalar overload deliberately: the instant overload would go back to
    // <t>/uniform/time and put the index back where it was, leaving
    // curTimeIndex_ == timeIndex() and V0 uncreated.
    runTime.setTime(readInstant.value(), updateIndex);

    // ...and that overload rebuilds the NAME by formatting the value
    // (Time.C:1073-1080), which is only the directory's name back again when
    // the case's timeFormat/timePrecision can round-trip it. Refuse rather
    // than read the wrong directory from here on.
    if (runTime.timeName() != readInstant.name())
    {
        FatalErrorInFunction
            << "advancing the time index renamed the instant from "
            << readInstant.name() << " to " << runTime.timeName()
            << ": the case's timeFormat/timePrecision cannot round-trip it"
            << exit(FatalError);
    }

    Info<< flush;

    const bool changed = mesh.update();

    Info<< "[brae] updateReturn " << (changed ? 1 : 0) << nl
        << "[brae] nCaptured " << mesh.nCaptured() << nl;

    // POST-update state: the mapped fields, before anything solves.
    Info<< "[brae] post nCells " << mesh.nCells() << nl
        << "[brae] post nPoints " << mesh.nPoints() << nl
        << "[brae] post nFaces " << mesh.nFaces() << nl
        << "[brae] post nInternalFaces " << mesh.nInternalFaces() << nl;

    writeLabels("post cellLevel", mesh.meshCutter().cellLevel());
    writeLabels("post pointLevel", mesh.meshCutter().pointLevel());

    writeScalars("post V", mesh.V().field());
    writeScalars("post V0", mesh.V0().field());

    writeGeometricField("post", refineField);
    if (UPtr)
    {
        writeGeometricField("post", UPtr());
    }
    if (pRghPtr)
    {
        writeGeometricField("post", pRghPtr());
    }

    Info<< "[brae] END" << nl;
    Info<< flush;

    return 0;
}


// ************************************************************************* //
