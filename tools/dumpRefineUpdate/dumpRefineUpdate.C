/*---------------------------------------------------------------------------*\
    dumpRefineUpdate -- OpenFOAM's own dynamicRefineFvMesh::updateTopology(),
    stepped, with everything it decides written out.

    Unit 7 of brae's dynamicRefineFvMesh port. The mesh half is gated one step
    at a time by tools/dumpHexRef8; what this adds is the DRIVER: which cells
    the field selects, what the buffer layers do to the refineCell set, which
    points are unrefined, and the state after each step.

    THE FIELD IS ANALYTIC, and that is the point. The tutorial's alpha.water
    comes out of a solve, so a comparison driven by it would measure the solve.
    Here the field is set from the CELL CENTRES after every topology change --
    1 inside a sphere that moves in x, 0 outside -- so brae is handed a field it
    can reproduce exactly, and what is compared is the refinement driver alone.
    The sphere moves so that both refinement (ahead of it) and unrefinement
    (behind it) happen, which one pass cannot exercise.

    The class it drives is dynamicRefineFvMeshDump: a copy of OpenFOAM's own
    with writes added and nothing else changed. The case's dynamicMeshDict must
    name it, which the gate script does.

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "dynamicFvMesh.H"
#include "volFields.H"
#include "OFstream.H"
#include "dynamicRefineFvMeshDump.H"
#include "hexRef8.H"
#include "refinementHistory.H"

using namespace Foam;

namespace
{

void writeLabels(Ostream& os, const word& name, const labelUList& l)
{
    os << name << ' ' << l.size();
    for (const label v : l) os << ' ' << v;
    os << nl;
}

// the mesh as tools/dumpHexRef8 writes it, so the same reader serves both
void writeMesh(Ostream& os, const polyMesh& mesh, const word& prefix)
{
    os << prefix << "nPoints " << mesh.nPoints() << nl;
    os << prefix << "nFaces " << mesh.nFaces() << nl;
    os << prefix << "nInternalFaces " << mesh.nInternalFaces() << nl;
    os << prefix << "nCells " << mesh.nCells() << nl;
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
    for (const polyPatch& pp : mesh.boundaryMesh())
    {
        os << "  " << pp.name() << ' ' << pp.start() << ' ' << pp.size() << nl;
    }
}

}   // namespace

int main(int argc, char *argv[])
{
    argList::addOption("steps", "int", "how many update() steps to run (default 4)");
    argList::addOption("out", "file", "where to write the dump (default refineUpdate.dump)");
    argList::addOption("centre", "(x y z)", "where the sphere starts (default the mesh centre)");
    argList::addOption("velocity", "(x y z)", "how far the sphere moves per step");
    argList::addOption("radius", "scalar", "the sphere's radius");

    #include "setRootCase.H"
    #include "createTime.H"

    autoPtr<dynamicFvMesh> meshPtr(dynamicFvMesh::New(args, runTime));
    dynamicFvMesh& mesh = meshPtr();

    // TOUCH THE CELL VOLUMES BEFORE ANYTHING CHANGES. fvMesh::updateMesh only stores old-time volumes
    // when the current ones already exist (fvMesh.C:1020-1024, `if (VPtr_)`), and
    // dynamicRefineFvMesh::mapFields then reads V0() unconditionally (:229) -- so without this the first
    // refinement aborts with "V0 is not available". A solver reaches the same state by building its
    // fields; this tool has to ask for it.
    (void)mesh.V();

    const label nSteps = args.getOrDefault<label>("steps", 4);
    const fileName outFile(args.getOrDefault<fileName>("out", "refineUpdate.dump"));

    // the sphere's path, defaulted from the mesh's own bounding box so the fixture does not depend on
    // the case's units
    const boundBox& bb = mesh.bounds();
    vector centre = bb.min() + 0.25*(bb.max() - bb.min());
    vector velocity = 0.12*(bb.max() - bb.min())/scalar(nSteps > 0 ? nSteps : 1);
    scalar radius = 0.18*mag(bb.max() - bb.min());
    if (args.found("centre"))   centre = args.get<vector>("centre");
    if (args.found("velocity")) velocity = args.get<vector>("velocity");
    if (args.found("radius"))   radius = args.get<scalar>("radius");

    // the field the dictionary names. Read so its boundary types are the case's, then overwritten.
    const word fieldName
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
        ).optionalSubDict(mesh.type() + "Coeffs").get<word>("field")
    );

    volScalarField fld
    (
        IOobject
        (
            fieldName,
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    // set the field from the cell centres: 1 inside the sphere, 0 outside
    const auto setField = [&](const vector& c)
    {
        forAll(fld, celli)
        {
            fld[celli] = (mag(mesh.C()[celli] - c) < radius) ? scalar(1) : scalar(0);
        }
        // NOT correctBoundaryConditions(): the case's alpha.water carries an inletOutlet on the
        // atmosphere, which looks phi up, and there is no phi here. Nothing in the selection reads a
        // boundary value -- cellToPoint takes the INTERNAL field (dynamicRefineFvMesh.C:754-772) and so
        // does maxCellField -- so the boundary is left as read.
    };

    // THE PASSIVE FIELDS, which is what unit 7b is about. They are set ONCE, before the first step, to a
    // value that says which cell each value came from -- the cell INDEX -- and then never touched again:
    // every later value is OpenFOAM's own mapping of them, accumulated over the steps. A refinement
    // copies a parent's value to its children; an unrefinement averages the children, VOLUME-WEIGHTED
    // where the map carries old cell volumes (cellMapper.C:104-160), which is exactly the behaviour a
    // 0/1 driving field cannot show.
    volScalarField passiveScalar
    (
        IOobject("braeScalar", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedScalar(dimless, Foam::zero{}),
        fvPatchFieldBase::calculatedType()
    );
    // ...AND A THIRD, RE-SET AT THE START OF EVERY STEP. The two above are set once, which is what tests
    // a mapping COMPOSED over three steps -- but it also makes the merge averaging invisible: a
    // refinement copies a parent's value to all eight children, so by the time they are merged back they
    // all hold the same number and their weighted mean equals any one of them. This one is written fresh
    // each step, so the cells a merge combines hold EIGHT DIFFERENT values and the average is a real one.
    volScalarField passiveFresh
    (
        IOobject("braeFresh", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedScalar(dimless, Foam::zero{}),
        fvPatchFieldBase::calculatedType()
    );
    volVectorField passiveVector
    (
        IOobject("braeVector", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedVector(dimless, Foam::zero{}),
        fvPatchFieldBase::calculatedType()
    );
    forAll(passiveScalar, celli)
    {
        passiveScalar[celli] = scalar(celli);
        passiveVector[celli] = vector(scalar(celli), scalar(2*celli), scalar(3*celli));
    }

    OFstream os(outFile);
    os.precision(17);
    os << "mode refineUpdate" << nl;
    os << "steps " << nSteps << nl;
    os << "sphereRadius " << radius << nl;
    os << "sphereCentre " << centre.x() << ' ' << centre.y() << ' ' << centre.z() << nl;
    os << "sphereVelocity " << velocity.x() << ' ' << velocity.y() << ' ' << velocity.z() << nl;
    os << "startCells " << mesh.nCells() << nl;
    Info<< "sphere radius " << radius << " from " << centre << " moving " << velocity
        << " per step, " << nSteps << " steps, starting from " << mesh.nCells() << " cells" << endl;

    const hexRef8& cutter =
        refCast<const dynamicRefineFvMeshDump>(mesh).meshCutter();

    for (label step = 1; step <= nSteps; ++step)
    {
        ++runTime;
        // the field is set BEFORE the update, on this step's mesh, at this step's centre
        setField(centre + scalar(step)*velocity);
        forAll(passiveFresh, celli) passiveFresh[celli] = scalar(celli);
        os << "step " << step << nl;
        os << "timeIndex " << runTime.timeIndex() << nl;
        writeLabels(os, "fieldOne", labelList());   // placeholder, kept so the reader sees the key
        {
            // the field itself, as the 0/1 pattern brae must reproduce -- a cell list, not scalars,
            // because the field IS 0 or 1 and a label list cannot be read differently on the two sides
            labelList ones;
            forAll(fld, celli)
            {
                if (fld[celli] > 0.5) ones.append(celli);
            }
            writeLabels(os, "fieldOneCells", ones);
        }
        druDump::os = &os;
        const bool changed = mesh.update();
        druDump::os = nullptr;
        os << "hasChanged " << (changed ? 1 : 0) << nl;
        writeLabels(os, "cellLevel", cutter.cellLevel());
        writeLabels(os, "pointLevel", cutter.pointLevel());
        {
            const refinementHistory& h = cutter.history();
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
                else { os << "  0"; }
                os << nl;
            }
        }
        // ...and the passive fields as OpenFOAM's own mapping left them, plus the old-time volumes,
        // which fvMesh::mapFields maps by a rule of their OWN (gather + inject the merged cells,
        // fvMesh.C:851-891) and dynamicRefineFvMesh::mapFields then corrects on split and merged cells
        os << "braeScalar " << passiveScalar.size();
        forAll(passiveScalar, celli) os << ' ' << passiveScalar[celli];
        os << nl;
        os << "braeFresh " << passiveFresh.size();
        forAll(passiveFresh, celli) os << ' ' << passiveFresh[celli];
        os << nl;
        os << "braeVector " << passiveVector.size();
        forAll(passiveVector, celli)
        {
            os << ' ' << passiveVector[celli].x() << ' ' << passiveVector[celli].y()
               << ' ' << passiveVector[celli].z();
        }
        os << nl;
        {
            const scalarField& V0 = mesh.V0();
            os << "V0 " << V0.size();
            for (const scalar v : V0) os << ' ' << v;
            os << nl;
            const scalarField& V = mesh.V();
            os << "V " << V.size();
            for (const scalar v : V) os << ' ' << v;
            os << nl;
        }
        writeMesh(os, mesh, "step");
        Info<< "step " << step << ": " << mesh.nCells() << " cells, changed " << changed << endl;
    }

    Info<< "wrote " << outFile << nl << "End" << nl << endl;
    return 0;
}
