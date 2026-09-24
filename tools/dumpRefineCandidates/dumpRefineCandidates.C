/*---------------------------------------------------------------------------*\
    brae instrument, not an OpenFOAM application.

    The refinement CANDIDATE SELECTION of OpenFOAM's dynamicRefineFvMesh, run on
    a fixed mesh with no topology change: the four field operations that stand
    between a volScalarField and the bitSet of candidate cells.

        cellToPoint(vFld)                      dynamicRefineFvMesh.C:754-771
        error(pFld, lowerRefineLevel, upper)   dynamicRefineFvMesh.C:773-793
        maxPointField(errPointField)           dynamicRefineFvMesh.C:718-733
        maxCellField(vFld)                     dynamicRefineFvMesh.C:736-751
        selectRefineCandidates(...)            dynamicRefineFvMesh.C:795-827

    None of these is a private static: they are PROTECTED const member functions
    of dynamicRefineFvMesh (dynamicRefineFvMesh.H:145-166). So this instrument
    does not transcribe them -- it derives a mesh class from dynamicRefineFvMesh
    and re-exports them with `using`, and every number below is produced by
    OpenFOAM's own code on OpenFOAM's own pointCells() addressing.

    maxCellField is printed too: it is not part of the refine path, it is the
    point field selectUnrefinePoints is given (dynamicRefineFvMesh.C:1433), and
    it is the mirror image of maxPointField, so a port gets both directions of
    the same cell-point walk checked from one run.

    The mesh is constructed at the SELECTED time, so -time 0.001 on a run that
    has already refined reads 0.001/polyMesh and its cellLevel/pointLevel, and
    -time 0 reads the base mesh from constant/polyMesh with cellLevel all zero.

    The case must carry a constant/dynamicMeshDict: the dynamicRefineFvMesh
    constructor calls readDict(), which insists on `correctFluxes` and
    `dumpLevel` (dynamicRefineFvMesh.C:166-194). Its `dynamicFvMesh` entry is
    NOT consulted here -- the class is named directly, not through New -- so a
    staticFvMesh case cannot be probed without adding those two entries.

    Usage:  dumpRefineCandidates -case <dir> [-time <t>] [-field alpha.water]
                [-lower 0.001] [-upper 0.999]

    With no -lower/-upper the values come from the case's own dynamicMeshDict,
    so the instrument cannot quietly substitute a threshold the case never set.

    Writes, one row per entry, bare space-separated at setprecision(18):
        [brae] nCells <n>
        [brae] nPoints <n>
        [brae] cellToPoint  <pointi> <value>
        [brae] error        <pointi> <value>
        [brae] maxPointField <celli> <value>
        [brae] maxCellField  <pointi> <value>
        [brae] candidate     <celli>
        [brae] nCandidates <n>
        [brae] END
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "timeSelector.H"
#include "IOmanip.H"
#include "IOdictionary.H"
#include "volFields.H"
#include "bitSet.H"
#include "dynamicRefineFvMesh.H"

using namespace Foam;

// dynamicRefineFvMesh's selection functions are protected, so a derived class
// is the only way to call them without editing OpenFOAM. Nothing is overridden
// and nothing is recomputed here: the `using` declarations only widen access.
class refineCandidateProbeFvMesh
:
    public dynamicRefineFvMesh
{
public:

    // doInit MUST be false here, then init(true): that is the order
    // dynamicFvMesh::New uses (dynamicFvMeshNew.C:84-87). Constructing with
    // the default doInit=true instead runs dynamicMotionSolverListFvMesh's own
    // constructor-init, which is the MANDATORY one (init(doInit) ->
    // init(doInit, true), dynamicMotionSolverListFvMesh.C:140-144) and demands
    // a `motionSolver` entry a pure-refinement dynamicMeshDict does not have.
    explicit refineCandidateProbeFvMesh(const IOobject& io)
    :
        dynamicRefineFvMesh(io, false)
    {
        init(true);
    }

    using dynamicRefineFvMesh::cellToPoint;
    using dynamicRefineFvMesh::error;
    using dynamicRefineFvMesh::maxCellField;
    using dynamicRefineFvMesh::maxPointField;
    using dynamicRefineFvMesh::selectRefineCandidates;
    using dynamicRefineFvMesh::selectRefineCells;
};


// A bare space-separated row per entry, not OpenFOAM's "4877 ( a b ... )", so a
// gate can split on whitespace without stripping the list length and brackets.
static void writeIndexedField
(
    const word& tag,
    const scalarField& f
)
{
    forAll(f, i)
    {
        Info<< "[brae] " << tag << ' ' << i << ' ' << f[i] << nl;
    }
    Info<< flush;
}


int main(int argc, char *argv[])
{
    argList::addNote
    (
        "dynamicRefineFvMesh's own refinement candidate selection,"
        " on a fixed mesh"
    );
    argList::addOption("field", "name", "the volScalarField to refine on");
    argList::addOption("lower", "scalar", "lowerRefineLevel");
    argList::addOption("upper", "scalar", "upperRefineLevel");
    argList::addOption("maxCells", "label", "the cell ceiling the budget uses");
    argList::addOption("maxRefinement", "label", "the refinement level cap");

    timeSelector::addOptions();

    #include "setRootCase.H"
    #include "createTime.H"

    // The instant the mesh and the field are read at. A refined run writes
    // <time>/polyMesh, and the mesh must be constructed AFTER the time is set
    // or facesInstance resolves back to constant/.
    const instantList timeDirs = timeSelector::select0(runTime, args);
    if (timeDirs.empty())
    {
        FatalErrorInFunction << "no time directory selected" << exit(FatalError);
    }
    runTime.setTime(timeDirs.last(), timeDirs.size() - 1);

    Info<< setprecision(18);

    refineCandidateProbeFvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.timeName(),
            runTime,
            IOobject::MUST_READ
        )
    );

    // The case's own settings. dynamicRefineFvMesh::update reads them from the
    // dictionary at every refine step (dynamicRefineFvMesh.C:1341-1350); an
    // unset -lower/-upper must come from there, never from a built-in number.
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
    const scalar lowerRefineLevel
    (
        args.getOrDefault<scalar>
        (
            "lower",
            refineDict.get<scalar>("lowerRefineLevel")
        )
    );
    const scalar upperRefineLevel
    (
        args.getOrDefault<scalar>
        (
            "upper",
            refineDict.get<scalar>("upperRefineLevel")
        )
    );

    const volScalarField vFld
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

    Info<< "[brae] time " << runTime.timeName() << nl
        << "[brae] field " << fieldName << nl
        << "[brae] lowerRefineLevel " << lowerRefineLevel << nl
        << "[brae] upperRefineLevel " << upperRefineLevel << nl
        << "[brae] nCells " << mesh.nCells() << nl
        << "[brae] nPoints " << mesh.nPoints() << nl
        << "[brae] nFaces " << mesh.nFaces() << nl;

    // The three stages of selectRefineCandidates, each one printed, so a port
    // is held against the intermediate and not only against the answer.
    const scalarField pFld(mesh.cellToPoint(vFld));
    writeIndexedField("cellToPoint", pFld);

    const scalarField errFld
    (
        mesh.error(pFld, lowerRefineLevel, upperRefineLevel)
    );
    writeIndexedField("error", errFld);

    const scalarField cellError(mesh.maxPointField(errFld));
    writeIndexedField("maxPointField", cellError);

    // Not on the refine path: the point field selectUnrefinePoints is handed
    // (dynamicRefineFvMesh.C:1427-1436).
    const scalarField maxCellFld(mesh.maxCellField(vFld));
    writeIndexedField("maxCellField", maxCellFld);

    // ...and the whole thing again through OpenFOAM's own composition, so the
    // stage prints above are checked to be the stages this actually runs.
    bitSet candidateCell(mesh.nCells());
    mesh.selectRefineCandidates
    (
        lowerRefineLevel,
        upperRefineLevel,
        vFld,
        candidateCell
    );

    for (const label celli : candidateCell)
    {
        Info<< "[brae] candidate " << celli << nl;
    }
    Info<< "[brae] nCandidates " << candidateCell.count() << nl;

    // UNIT 2: from the candidates to the cells that will actually be refined --
    // hexRef8's 2:1 closure and dynamicRefineFvMesh's budget/level selection.
    // Both are printed, and separately, because they are different selections:
    // the closure ADDS cells (maxSet true) where the budget only removes them.
    {
        const labelList& cellLevel = mesh.meshCutter().cellLevel();
        forAll(cellLevel, celli)
        {
            Info<< "[brae] cellLevel " << celli << ' ' << cellLevel[celli] << nl;
        }

        // ...overridable, because the case's own maxCells is 200000 against 4032 cells: the budget
        // is never binding here, so the whole-level truncation branch (dynamicRefineFvMesh.C:871-889)
        // is unreachable on the tutorial as shipped and a gate on it needs a smaller ceiling.
        const label maxCells
        (
            args.getOrDefault<label>("maxCells", refineDict.get<label>("maxCells"))
        );
        const label maxRefinement
        (
            args.getOrDefault<label>("maxRefinement", refineDict.get<label>("maxRefinement"))
        );
        Info<< "[brae] maxCells " << maxCells << nl
            << "[brae] maxRefinement " << maxRefinement << nl
            << "[brae] nTotalCells " << mesh.globalData().nTotalCells() << nl;

        // the closure on the RAW candidate set, so a port is held against it
        // without the budget in the way
        const labelList consistentSet
        (
            mesh.meshCutter().consistentRefinement(candidateCell.toc(), true)
        );
        forAll(consistentSet, i)
        {
            Info<< "[brae] consistent " << consistentSet[i] << nl;
        }
        Info<< "[brae] nConsistent " << consistentSet.size() << nl;

        // ...and the whole selection OpenFOAM would act on
        const labelList selected
        (
            mesh.selectRefineCells(maxCells, maxRefinement, candidateCell)
        );
        forAll(selected, i)
        {
            Info<< "[brae] selected " << selected[i] << nl;
        }
        Info<< "[brae] nSelected " << selected.size() << nl;
    }

    // The levels the next unit will need, and the only thing on this mesh that
    // says whether it has been refined already.
    Info<< "[brae] level0Edge " << mesh.meshCutter().level0EdgeLength() << nl
        << "[brae] maxCellLevel " << max(mesh.meshCutter().cellLevel()) << nl
        << "[brae] maxPointLevel " << max(mesh.meshCutter().pointLevel()) << nl
        << "[brae] nProtectedCells " << mesh.protectedCell().count() << nl;

    Info<< "[brae] END" << endl;

    return 0;
}

// ************************************************************************* //
