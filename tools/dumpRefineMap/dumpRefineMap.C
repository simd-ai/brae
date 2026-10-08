/*---------------------------------------------------------------------------*\
    brae instrument, not an OpenFOAM application.

    The FIELD MAPPING half of OpenFOAM's dynamicRefineFvMesh: the mapPolyMesh
    that a refine or an unrefine step produces, and the cell AND face values
    every registered field carries once fvMesh::mapFields, the V0 correction,
    the correctFluxes_ correction and mapNewInternalFaces have run against it.

        dynamicRefineFvMesh::refine      dynamicRefineFvMesh.C:442-535
        dynamicRefineFvMesh::unrefine    dynamicRefineFvMesh.C:537-716
        dynamicRefineFvMesh::mapFields   dynamicRefineFvMesh.C:197-438
          the V0 correction               :203-254
          the flux correction             :256-422
          mapNewInternalFaces             :424-437
        dynamicRefineFvMesh::mapNewInternalFaces
                                         dynamicRefineFvMeshTemplates.C:32-183
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

    FOUR TRAPS IN THE SETUP, each of which makes update() silently do nothing,
    abort, or take a branch other than the one being measured, and none of which
    is visible in the answer.

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

    3. The flux correction walks the mesh's own objectRegistry
       (dynamicRefineFvMesh.C:298-301, and mapNewInternalFaces
       dynamicRefineFvMeshTemplates.C:109), so a surfaceScalarField that is not
       registered BEFORE update() is not corrected, not warned about and not
       visible. phi and Uf are therefore constructed here, on the mesh, before
       the update; hasPhi/hasUf say whether each one was found, so a run cannot
       quietly measure a correction that had nothing to correct.

    4. Which BRANCH of mapNewInternalFaces a surface field takes is decided by
       is_oriented() (dynamicRefineFvMeshTemplates.C:118,158), and the oriented
       flag is a FILE entry: DimensionedField reads the optional "oriented"
       keyword on construction (DimensionedFieldIO.C:48-51) and writes it only
       when the field is ORIENTED (orientedType.C:123-133). interFoam's phi
       carries `oriented oriented;` and its Uf does not, so phi takes the
       oriented branch -- fFld = phi*Sf/sqr(magSf), map, then `sFld = (fFld &
       Sf)` (:169-175), a whole-field assignment whose round trip is NOT the
       identity in floating point and moves every internal AND boundary face by
       of order one ulp -- while Uf takes the plain branch and only injected
       internal faces move. isOriented is printed per field for that reason.

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
        [brae] hasU|hasP_rgh|hasPhi|hasUf <true|false>
        [brae] correctFluxes <fluxName> <UName>      in the dict's own order
        [brae] nCorrectFluxes <n>
        [brae] isOriented <fieldName> <true|false>
        [brae] phiUFrom <fluxName> <UName>
        [brae] pre  nCells|nPoints|nFaces|nInternalFaces <n>
        [brae] pre  cellLevel|pointLevel <i> <level>
        [brae] pre  V <celli> <value>
        [brae] pre  <fieldName> <celli> <value...>
        [brae] pre  <fieldName>Patch <patchi> <facei> <value...>
        [brae] pre  <surfName> <facei> <value...>            internal faces
        [brae] pre  <surfName>Patch <patchi> <facei> <value...>
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
        [brae] map <k> faceMapperDirect <true|false>
        [brae] map <k> faceMapperSize|faceMapperSizeBeforeMapping <n>
        [brae] map <k> faceMapperHasUnmapped <true|false>
        [brae] map <k> faceDirectAddressing <newi> <oldi>   (direct only)
        [brae] map <k> faceAddressing <newi> <n> <old...>   (interp only)
        [brae] map <k> faceWeights <newi> <n> <w...>        (interp only)
        [brae] map <k> faceInserted <newi>
        [brae] map <k> nFaceInserted <n>
      derived rows -- transcriptions of the correction's own loops, NOT captured:
        [brae] map <k> masterFace <facei>
        [brae] map <k> nMasterFace <n>
        [brae] map <k> nMasterFaceNegative <n>
        [brae] map <k> fluxInflated <facei>
        [brae] map <k> nFluxInflated <n>
        [brae] map <k> fluxFromMaster <facei>
        [brae] map <k> nFluxFromMaster <n>
        [brae] map <k> nFluxOverwritten <n>
        [brae] map <k> newInternalFaceHull <facei> <n> <hullFace...>
        [brae] map <k> nNewInternalFaceHull <n>
        [brae] map <k> nNewInternalFaceNoHull <n>
      unrefine only -- the SECOND correction site:
        [brae] map <k> splitPoint <pointi>            captured (the argument)
        [brae] map <k> nSplitPoint <n>
        [brae] map <k> faceToSplitPoint <oldFacei> <oldPointi>
        [brae] map <k> nFaceToSplitPoint <n>
        [brae] map <k> unrefineFluxFace <facei>
        [brae] map <k> nUnrefineFluxFace <n>
        [brae] post nCells|nPoints|nFaces|nInternalFaces <n>
        [brae] post cellLevel|pointLevel <i> <level>
        [brae] post V <celli> <value>
        [brae] post V0 <celli> <value>
        [brae] post <fieldName> <celli> <value...>
        [brae] post <fieldName>Patch <patchi> <facei> <value...>
        [brae] post <surfName> <facei> <value...>
        [brae] post <surfName>Patch <patchi> <facei> <value...>
        [brae] post faceOwner|faceNeighbour <facei> <celli>
        [brae] post patchStart|patchSize <patchi> <n>
        [brae] post cellFaces <celli> <n> <facei...>
        [brae] post Sf <facei> <x> <y> <z>       and SfPatch
        [brae] post magSf <facei> <value>        and magSfPatch
        [brae] post phiU <facei> <value>         and phiUPatch  (reproduced)
        [brae] END

    THERE ARE TWO FLUX CORRECTIONS, not one, and they do not share a criterion.

      * mapFields :256-422, reached by both phases, keyed on faceMap /
        reverseFaceMap / masterFaces. MEASURED to be a complete no-op on the
        unrefine step of damBreakWithObstacle: nFluxInflated, nFluxFromMaster
        and nMasterFace are all 0 there, and OpenFOAM's own debug print agrees
        ("Found 0 split faces").
      * unrefine :610-689, reached by the unrefine phase ONLY, keyed on
        faceToSplitPoint and reversePointMap, and run AFTER mapFields --
        so after mapNewInternalFaces. It has a `none` branch but NO `NaN`
        branch, and it is the only flux correction an unrefine actually
        performs on this case.

    WHAT THE FLUX CORRECTION READS, and which of it is CAPTURED here.

    Captured (read off OpenFOAM's own objects after its own code has run):
    faceMap and reverseFaceMap, the pre and post values of phi and Uf on both
    halves of the field, the post mesh's Sf/magSf/faceOwner/faceNeighbour/cells,
    and fvSurfaceMapper -- the addressing fvMesh::mapFields mapped phi's
    internal field through, built here as a second read-only instance of the
    same class on the same map (fvMeshMapper.H:97 surfaceMap_(mesh, faceMap_)).

    DERIVED (recomputed here, so a defect in the transcription would be shared
    by oracle and port and neither would see it):

      * masterFaces. OpenFOAM does NOT expose it. It is a bitSet declared local
        to mapFields (:268) and destroyed with the block; nothing on mapPolyMesh
        or on the mesh carries it. It is derived here from faceMap and
        reverseFaceMap by the loop at :270-291. The count is cross-checkable
        against OpenFOAM's own: `-debug-switch dynamicRefineFvMesh=1` makes
        :293-296 print "Found <n> split faces", and that number must equal
        nMasterFace.
      * fluxInflated and fluxFromMaster, the two inline conditions of the
        internal loop (:359, :364) and the boundary loop (:386, :391).
      * newInternalFaceHull, the already-mapped owner-plus-neighbour face list
        that mapNewInternalFaces averages (Templates:72-90). Its ORDER is the
        summation order, and it is cells()'s order, which is not ascending
        face index: primitiveMesh::calcCells fills every cell's owner faces
        first and its neighbour faces second, each ascending
        (primitiveMeshCells.C:82-97).
      * faceToSplitPoint and unrefineFluxFace, the second site's key. The table
        is a Map local to unrefine() (:555) built from the PRE-change mesh's
        pointEdges/edges/pointFaces, so it is rebuilt in the unrefine override
        before the base call; splitPoints_ beside it IS captured, being the
        argument.
      * phiU. The correction's `fvc::interpolate(U) & Sf()` (:345-352), rebuilt
        here after update() returns with the same expression on the same fields.
        It reproduces the in-step value rather than capturing it, and it does so
        exactly: mapFields clears the interpolation weights at :306 and the
        rebuild that follows is cached on the post-change mesh, U is not touched
        again by mapFields, and the scheme is the case's own
        `interpolate(U)` entry (surfaceInterpolate.C:256 builds that name,
        schemesLookup.C:221-225 resolves it).
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "timeSelector.H"
#include "IOmanip.H"
#include "IOdictionary.H"
#include "Switch.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "surfaceInterpolate.H"
#include "cellMapper.H"
#include "faceMapper.H"
#include "fvSurfaceMapper.H"
#include "bitSet.H"
#include "Pair.H"
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


// The face values and then the patch values of one surface field, in the same
// two row shapes writeGeometricField uses for a volField, so a reader needs one
// parser for both halves of both kinds. The name is a parameter rather than
// fld.name() because Sf() and magSf() are the mesh's own fields and are named
// "S" and "magSf" (fvMeshGeometry.C:58, :95), which would make the row for the
// face areas read "post S".
template<class Type>
static void writeSurfaceField
(
    const std::string& prefix,
    const word& name,
    const GeometricField<Type, fvsPatchField, surfaceMesh>& fld
)
{
    const label nCmpt = pTraits<Type>::nComponents;

    forAll(fld, facei)
    {
        Info<< "[brae] " << prefix.c_str() << ' ' << name << ' ' << facei;

        for (label cmpt = 0; cmpt < nCmpt; ++cmpt)
        {
            Info<< ' ' << component(fld[facei], cmpt);
        }

        Info<< nl;
    }

    const auto& bf = fld.boundaryField();

    forAll(bf, patchi)
    {
        forAll(bf[patchi], facei)
        {
            Info<< "[brae] " << prefix.c_str() << ' ' << name
                << "Patch " << patchi << ' ' << facei;

            for (label cmpt = 0; cmpt < nCmpt; ++cmpt)
            {
                Info<< ' ' << component(bf[patchi][facei], cmpt);
            }

            Info<< nl;
        }
    }
}


// One row per set member plus a count row, for a set that is EMPTY on one of
// the two phases. Without the count a reader cannot tell empty from unprinted,
// and on an unrefine step every one of these is empty.
static void writeFaceSet
(
    const std::string& prefix,
    const char* const name,
    const char* const countName,
    const bitSet& set
)
{
    for (const label facei : set)
    {
        Info<< "[brae] " << prefix.c_str() << ' ' << name << ' ' << facei << nl;
    }

    Info<< "[brae] " << prefix.c_str() << ' ' << countName << ' '
        << set.count() << nl;
}


// * * * * * * * * * * * * * * * * the mesh * * * * * * * * * * * * * * * * * //

class refineMapProbeFvMesh
:
    public dynamicRefineFvMesh
{
    // Private Data

        //- How many maps have been seen in this update()
        label nCaptured_;

        //- The splitPoints unrefine() was called with. Captured: it is the
        //  argument, and unrefine's own flux correction is keyed on it.
        labelList splitPoints_;

        //- old face -> the split point on it, DERIVED from the PRE-change mesh
        //  by the loop at dynamicRefineFvMesh.C:555-574. It has to be built
        //  before the base call, because it reads pointEdges/edges/pointFaces
        //  of the mesh unrefine is about to change.
        Map<label> faceToSplitPoint_;


    // Private Member Functions

        //- Everything the returned map holds, plus the cell addressing that
        //  fvMesh::mapFields derived from it. Read-only on the mesh and on
        //  the map; the only thing it writes is its own capture counter.
        void report(const word& phase, const mapPolyMesh& map);

        //- The face sets the flux correction overwrites and the hull
        //  mapNewInternalFaces averages, derived from the map by the
        //  correction's own loops. Read-only on the mesh and on the map.
        void reportFluxFaces
        (
            const std::string& m,
            const word& phase,
            const mapPolyMesh& map
        );


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
    //
    //  unrefine() carries a SECOND flux correction of its own (:610-689),
    //  separate from the one in mapFields and keyed on faceToSplitPoint rather
    //  than on masterFaces. That table is local to unrefine and is read off the
    //  PRE-change mesh, so it is rebuilt here before the base call. This is the
    //  one place the instrument does work ahead of OpenFOAM's, and it is
    //  read-only on the mesh: pointEdges(), edges() and pointFaces() are
    //  demand-driven addressing the base call is about to ask for anyway.
    virtual autoPtr<mapPolyMesh> unrefine(const labelList& splitPoints)
    {
        splitPoints_ = splitPoints;

        faceToSplitPoint_ = Map<label>(3*splitPoints.size());

        for (const label pointi : splitPoints)
        {
            const labelList& pEdges = pointEdges()[pointi];

            for (const label edgei : pEdges)
            {
                const label otherPointi = edges()[edgei].otherVertex(pointi);

                const labelList& pFaces = pointFaces()[otherPointi];

                for (const label facei : pFaces)
                {
                    faceToSplitPoint_.insert(facei, otherPointi);
                }
            }
        }

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

    // The face addressing the SURFACE half of fvMesh::mapFields mapped through
    // (fvMesh.C:825-837 MapGeometricFields over fvsPatchField/surfaceMesh), as
    // fvMeshMapper builds it: a faceMapper off the map, then an fvSurfaceMapper
    // off the mesh and that faceMapper (fvMeshMapper.H:95,97). fvSurfaceMapper
    // is the INTERNAL half only -- size() is nInternalFaces (fvSurfaceMapper.H:
    // 117-120) -- and phi's patch values map through fvPatchMapper instead.
    const faceMapper faceMapper_(map);
    const fvSurfaceMapper surfMap(*this, faceMapper_);

    Info<< "[brae] " << mc << " faceMapperDirect "
        << Switch(surfMap.direct()) << nl
        << "[brae] " << mc << " faceMapperSize " << surfMap.size() << nl
        << "[brae] " << mc << " faceMapperSizeBeforeMapping "
        << surfMap.sizeBeforeMapping() << nl
        << "[brae] " << mc << " faceMapperHasUnmapped "
        << Switch(surfMap.hasUnmapped()) << nl;

    if (surfMap.direct())
    {
        writeLabels(m + " faceDirectAddressing", surfMap.directAddressing());
    }
    else
    {
        const labelListList& addr = surfMap.addressing();
        const scalarListList& w = surfMap.weights();

        forAll(addr, facei)
        {
            Info<< "[brae] " << mc << " faceAddressing " << facei << ' '
                << addr[facei].size();

            for (const label a : addr[facei])
            {
                Info<< ' ' << a;
            }

            Info<< nl;

            Info<< "[brae] " << mc << " faceWeights " << facei << ' '
                << w[facei].size();

            for (const scalar s : w[facei])
            {
                Info<< ' ' << s;
            }

            Info<< nl;
        }
    }

    writeLabels(m + " faceInserted", surfMap.insertedObjectLabels());
    Info<< "[brae] " << mc << " nFaceInserted "
        << surfMap.insertedObjectLabels().size() << nl;

    reportFluxFaces(m, phase, map);

    Info<< flush;
}


// The three face sets the flux correction overwrites, and the hull
// mapNewInternalFaces averages. Every row here is DERIVED: see the header. The
// mesh is the POST-change mesh, which is the mesh the correction ran against.
void refineMapProbeFvMesh::reportFluxFaces
(
    const std::string& m,
    const word& phase,
    const mapPolyMesh& map
)
{
    const char* const mc = m.c_str();

    const labelList& faceMap = map.faceMap();
    const labelList& reverseFaceMap = map.reverseFaceMap();

    // dynamicRefineFvMesh.C:268-291, with one difference stated rather than
    // hidden: OpenFOAM aborts on a negative masterFacei (:278-285, "should not
    // have removed faces when refining"). An instrument that aborted there
    // would destroy the dump it exists to produce, so the count is printed
    // instead and it must be 0 for the derivation to mean what OpenFOAM means.
    bitSet masterFaces(nFaces());
    label nNegative = 0;

    forAll(faceMap, facei)
    {
        const label oldFacei = faceMap[facei];

        if (oldFacei >= 0)
        {
            const label masterFacei = reverseFaceMap[oldFacei];

            if (masterFacei < 0)
            {
                ++nNegative;
            }
            else if (masterFacei != facei)
            {
                masterFaces.set(masterFacei);
            }
        }
    }

    // The two inline conditions, split apart so a gate can see WHICH of them
    // claimed a face. The internal loop (:355-369) and the boundary loop
    // (:374-399) test the same pair, so they are walked in OpenFOAM's order and
    // recorded in one pair of sets indexed by mesh face.
    bitSet inflated(nFaces());
    bitSet fromMaster(nFaces());

    for (label facei = 0; facei < nInternalFaces(); ++facei)
    {
        const label oldFacei = faceMap[facei];

        if (oldFacei == -1)
        {
            inflated.set(facei);
        }
        else if (reverseFaceMap[oldFacei] != facei)
        {
            fromMaster.set(facei);
        }
    }

    forAll(boundary(), patchi)
    {
        label facei = boundary()[patchi].start();

        forAll(boundary()[patchi], i)
        {
            const label oldFacei = faceMap[facei];

            if (oldFacei == -1)
            {
                inflated.set(facei);
            }
            else if (reverseFaceMap[oldFacei] != facei)
            {
                fromMaster.set(facei);
            }

            ++facei;
        }
    }

    writeFaceSet(m, "masterFace", "nMasterFace", masterFaces);
    Info<< "[brae] " << mc << " nMasterFaceNegative " << nNegative << nl;

    writeFaceSet(m, "fluxInflated", "nFluxInflated", inflated);
    writeFaceSet(m, "fluxFromMaster", "nFluxFromMaster", fromMaster);

    // The union, because the three sets overlap: the siblings of a split face
    // are fromMaster and their master is masterFace, so a port that counted the
    // three separately would over-count the faces that actually changed.
    bitSet overwritten(masterFaces);
    overwritten |= inflated;
    overwritten |= fromMaster;

    Info<< "[brae] " << mc << " nFluxOverwritten "
        << overwritten.count() << nl;

    // dynamicRefineFvMeshTemplates.C:59-97. cells() order is the summation
    // order and it is owner faces then neighbour faces, each ascending
    // (primitiveMeshCells.C:82-97), so the list is printed rather than left for
    // a port to guess.
    const labelUList& own = faceOwner();
    const labelUList& nei = faceNeighbour();
    const cellList& cs = cells();

    label nHull = 0;
    label nNoHull = 0;

    for (label facei = 0; facei < nInternalFaces(); ++facei)
    {
        if (faceMap[facei] != -1)
        {
            continue;
        }

        DynamicList<label> hull;

        for (const label ownFacei : cs[own[facei]])
        {
            if (faceMap[ownFacei] != -1)
            {
                hull.append(ownFacei);
            }
        }

        for (const label neiFacei : cs[nei[facei]])
        {
            if (faceMap[neiFacei] != -1)
            {
                hull.append(neiFacei);
            }
        }

        Info<< "[brae] " << mc << " newInternalFaceHull " << facei << ' '
            << hull.size();

        for (const label h : hull)
        {
            Info<< ' ' << h;
        }

        Info<< nl;

        if (hull.empty())
        {
            // counter == 0 leaves the face alone (Templates:92-95), which is
            // the one case where an injected face keeps its mapped value
            ++nNoHull;
        }
        else
        {
            ++nHull;
        }
    }

    Info<< "[brae] " << mc << " nNewInternalFaceHull " << nHull << nl
        << "[brae] " << mc << " nNewInternalFaceNoHull " << nNoHull << nl;

    if (phase != "unrefine")
    {
        return;
    }

    // THE SECOND CORRECTION SITE (:610-689), which lives in unrefine() and not
    // in mapFields, runs AFTER mapFields has finished, and is keyed on a face's
    // split point rather than on masterFaces. On this case the mapFields
    // correction is a complete no-op on an unrefine step -- all three sets
    // above are empty -- so this is the only flux correction an unrefine
    // performs, and a port that implemented only the mapFields half would be
    // green on the refine arm and silently wrong here.
    writeLabels(m + " splitPoint", splitPoints_);
    Info<< "[brae] " << mc << " nSplitPoint " << splitPoints_.size() << nl;

    const labelList oldFaces(faceToSplitPoint_.sortedToc());

    for (const label oldFacei : oldFaces)
    {
        Info<< "[brae] " << mc << " faceToSplitPoint " << oldFacei << ' '
            << faceToSplitPoint_[oldFacei] << nl;
    }

    Info<< "[brae] " << mc << " nFaceToSplitPoint " << oldFaces.size() << nl;

    // The faces :659-687 writes phiU onto: an old face whose split point is
    // gone (reversePointMap < 0) and which still exists (reverseFaceMap >= 0).
    // Printed sorted by NEW face so the row order does not inherit the hash
    // table's.
    const labelList& reversePointMap = map.reversePointMap();
    const labelList& reverseFaceMap2 = map.reverseFaceMap();

    bitSet unrefineFlux(nFaces());

    for (const label oldFacei : oldFaces)
    {
        const label oldPointi = faceToSplitPoint_[oldFacei];

        if (reversePointMap[oldPointi] < 0)
        {
            const label facei = reverseFaceMap2[oldFacei];

            if (facei >= 0)
            {
                unrefineFlux.set(facei);
            }
        }
    }

    writeFaceSet(m, "unrefineFluxFace", "nUnrefineFluxFace", unrefineFlux);
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

    // ...and the two SURFACE fields the flux correction and
    // mapNewInternalFaces act on. They have to exist before update() or none of
    // the three passes can see them: the correction takes its list from the
    // registry (:298-301) and so does mapNewInternalFaces (Templates:109).
    // Trap 4 in the header: which branch phi takes is decided by the "oriented"
    // entry in the FILE, so these must be read, never rebuilt from U.
    autoPtr<surfaceScalarField> phiPtr;
    autoPtr<surfaceVectorField> UfPtr;

    {
        IOobject phiio
        (
            "phi",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        );

        if (phiio.typeHeaderOk<surfaceScalarField>(true))
        {
            phiPtr.reset(new surfaceScalarField(phiio, mesh));
        }

        IOobject Ufio
        (
            "Uf",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        );

        if (Ufio.typeHeaderOk<surfaceVectorField>(true))
        {
            UfPtr.reset(new surfaceVectorField(Ufio, mesh));
        }
    }

    Info<< "[brae] time " << runTime.timeName() << nl
        << "[brae] timeDirIndex " << stepIndex << nl
        << "[brae] timeIndexAtConstruction " << constructionIndex << nl
        << "[brae] timeIndexAtUpdate " << updateIndex << nl
        << "[brae] refineInterval " << refineInterval << nl
        << "[brae] field " << fieldName << nl
        << "[brae] hasU " << Switch(bool(UPtr)) << nl
        << "[brae] hasP_rgh " << Switch(bool(pRghPtr)) << nl
        << "[brae] hasPhi " << Switch(bool(phiPtr)) << nl
        << "[brae] hasUf " << Switch(bool(UfPtr)) << nl;

    // The correctFluxes table in the dict's OWN order. readDict reworks this
    // list into a HashTable (:184-191) whose iteration order is the table's, so
    // the list is the only ordered authority on it. It is printed whole because
    // an entry this instrument does not register is still an entry a port has to
    // honour, and because "none" and an ABSENT name are different code paths
    // (:312-328): absent warns and skips, "none" skips silently.
    {
        const auto fluxVelocities
        (
            refineDict.get<List<Pair<word>>>("correctFluxes")
        );

        for (const auto& pr : fluxVelocities)
        {
            Info<< "[brae] correctFluxes " << pr.first() << ' '
                << pr.second() << nl;
        }

        Info<< "[brae] nCorrectFluxes " << fluxVelocities.size() << nl;
    }

    if (phiPtr)
    {
        Info<< "[brae] isOriented phi "
            << Switch(phiPtr().is_oriented()) << nl;
    }
    if (UfPtr)
    {
        Info<< "[brae] isOriented Uf "
            << Switch(UfPtr().is_oriented()) << nl;
    }

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
    if (phiPtr)
    {
        writeSurfaceField("pre", "phi", phiPtr());
    }
    if (UfPtr)
    {
        writeSurfaceField("pre", "Uf", UfPtr());
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
    if (phiPtr)
    {
        writeSurfaceField("post", "phi", phiPtr());
    }
    if (UfPtr)
    {
        writeSurfaceField("post", "Uf", UfPtr());
    }

    // The post-change mesh the correction ran against. owner/neighbour and
    // cells() are here because mapNewInternalFaces walks the hull through them
    // and the ORDER of cells()[celli] is the order it sums in; Sf and magSf
    // because phiU is `... & Sf()` (:351) and because the oriented branch
    // divides by sqr(magSf) (Templates:169).
    writeLabels("post faceOwner", mesh.faceOwner());
    writeLabels("post faceNeighbour", mesh.faceNeighbour());

    // The boundary loop of the correction walks patch().start() upwards
    // (:380,:397) and the masterFace loop converts the other way with
    // whichPatch() and start() (:410-411), so the POST-change patch starts are
    // what turns a (patchi, i) row into a mesh face and back. map <k>
    // oldPatchStarts is the OLD mesh's and cannot do it.
    forAll(mesh.boundary(), patchi)
    {
        Info<< "[brae] post patchStart " << patchi << ' '
            << mesh.boundary()[patchi].start() << nl
            << "[brae] post patchSize " << patchi << ' '
            << mesh.boundary()[patchi].size() << nl;
    }

    {
        const cellList& cs = mesh.cells();

        forAll(cs, celli)
        {
            Info<< "[brae] post cellFaces " << celli << ' ' << cs[celli].size();

            for (const label facei : cs[celli])
            {
                Info<< ' ' << facei;
            }

            Info<< nl;
        }
    }

    writeSurfaceField("post", "Sf", mesh.Sf());
    writeSurfaceField("post", "magSf", mesh.magSf());

    // phiU, REPRODUCED and not captured: the correction's own expression
    // (:345-352) evaluated after update() returned, on the same U and the same
    // post-change weights. It is printed for the flux named FIRST in
    // correctFluxes that this instrument actually holds and whose velocity is
    // neither "none" nor "NaN", and the pairing is printed with it so a reader
    // never has to assume which flux and which velocity it belongs to.
    if (phiPtr)
    {
        word UName(word::null);

        for (const auto& pr : refineDict.get<List<Pair<word>>>("correctFluxes"))
        {
            if (pr.first() == "phi")
            {
                UName = pr.second();
            }
        }

        if (UName.empty() || UName == "none" || UName == "NaN")
        {
            // Not a defect and not a substitution: it is the case telling the
            // correction to leave phi alone, and the reader has to be able to
            // see that the interpolating branch did not run.
            Info<< "[brae] phiUFrom phi " << (UName.empty() ? "absent" : UName)
                << nl;
        }
        else if (!mesh.foundObject<volVectorField>(UName))
        {
            Info<< "[brae] phiUFrom phi " << UName << "-unregistered" << nl;
        }
        else
        {
            Info<< "[brae] phiUFrom phi " << UName << nl;

            const surfaceScalarField phiU
            (
                fvc::interpolate
                (
                    mesh.lookupObject<volVectorField>(UName)
                )
              & mesh.Sf()
            );

            writeSurfaceField("post", "phiU", phiU);
        }
    }

    Info<< "[brae] END" << nl;
    Info<< flush;

    return 0;
}


// ************************************************************************* //
