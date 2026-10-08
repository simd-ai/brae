/*---------------------------------------------------------------------------*\
    brae instrument, not an OpenFOAM application.

    OpenFOAM's own pointConstraints::constrainDisplacement on a pointDisplacement
    the caller staged -- the last line of rigidBodyMeshMotion::solve
    (rigidBodyMeshMotion.C:385-388), run on inputs brae controls rather than on a
    displacement the rigid body happened to produce.

    Prints, at setprecision(18) and prefixed [brae]:
      normal <patch> (x y z)          every symmetryPlane patch's n(), the frozen
                                      normal both the patch field and the corner
                                      constraint use (symmetryPlanePolyPatch.C:46-57)
      corner <point> <count> T(9)     every patchPatchPointConstraint point with its
                                      constraint count and tensor
                                      (pointConstraints.C:310-340)
    and writes two pointVectorFields next to the input:
      pointDisplacement.evaluated     after correctBoundaryConditions() alone
      pointDisplacement.constrained   after constrainDisplacement() on the same input,
                                      with the default overrideFixedValue (false), as
                                      rigidBodyMeshMotion calls it
    and, with -synthetic, pointDisplacement.input: the field both start from.

    usage: dumpPointConstraints -case <dir> [-time <t>] [-synthetic]
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pointMesh.H"
#include "pointFields.H"
#include "pointConstraints.H"
#include "symmetryPlanePolyPatch.H"
#include "IOmanip.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "OpenFOAM's pointConstraints::constrainDisplacement on a staged"
        " pointDisplacement, and the constraint it builds"
    );
    argList::addBoolOption
    (
        "synthetic",
        "replace the internal field by a smooth polynomial of the point position first"
    );
    timeSelector::addOptions();

    #include "setRootCase.H"
    #include "createTime.H"

    const instantList timeDirs = timeSelector::select0(runTime, args);
    runTime.setTime(timeDirs.last(), timeDirs.size() - 1);

    #include "createMesh.H"

    Info<< setprecision(18);

    const pointMesh& pMesh = pointMesh::New(mesh);

    forAll(mesh.boundaryMesh(), patchi)
    {
        const polyPatch& pp = mesh.boundaryMesh()[patchi];
        if (isA<symmetryPlanePolyPatch>(pp))
        {
            const vector& n = refCast<const symmetryPlanePolyPatch>(pp).n();
            Info<< "[brae] normal " << pp.name() << ' ' << n << endl;
        }
    }

    const pointConstraints& pcs = pointConstraints::New(pMesh);
    const labelList& cPoints = pcs.patchPatchPointConstraintPoints();
    const tensorField& cTensors = pcs.patchPatchPointConstraintTensors();
    // patchPatchPointConstraints() is NOT aligned with the two lists above: pointConstraints.C:314-340
    // compacts the points and tensors past every zero-constraint entry and then only TRUNCATES the constraint
    // list, so its i-th entry is not the i-th point's. The count is read off the tensor instead:
    // constraintTransformation gives I - nn (trace 2) for one constraint, ll (trace 1) for two, 0 for three.
    Info<< "[brae] nCorner " << cPoints.size() << endl;
    forAll(cPoints, i)
    {
        const tensor& T = cTensors[i];
        const scalar tr = T.xx() + T.yy() + T.zz();
        const label count = (tr > 1.5) ? 1 : (tr > 0.5) ? 2 : 3;
        Info<< "[brae] corner " << cPoints[i] << ' ' << count
            << ' ' << T.xx() << ' ' << T.xy() << ' ' << T.xz()
            << ' ' << T.yx() << ' ' << T.yy() << ' ' << T.yz()
            << ' ' << T.zx() << ' ' << T.zy() << ' ' << T.zz() << endl;
    }

    const IOobject io
    (
        "pointDisplacement",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE,
        IOobject::NO_REGISTER
    );

    // -synthetic: the internal field replaced by a smooth polynomial of the point position, nonzero in all
    // three components on every patch, and written as pointDisplacement.input so the port reads the same
    // bits; the patch entries (types and fixedValue values) stay the file's
    pointVectorField input(io, pMesh);
    if (args.found("synthetic"))
    {
        const pointField& p = mesh.points();
        vectorField& d = input.primitiveFieldRef();
        forAll(d, i)
        {
            const scalar x = p[i].x();
            const scalar y = p[i].y();
            const scalar z = p[i].z();
            d[i] = vector
            (
                0.01*x + 0.02*(y*z) + 0.003,
                0.03*y - 0.01*(x*x) + 0.002,
                0.02*z + 0.005*(x*y) - 0.004
            );
        }
        pointVectorField w
        (
            IOobject("pointDisplacement.input", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
            input
        );
        w.write();
    }

    {
        pointVectorField pd(input);
        pd.correctBoundaryConditions();
        pointVectorField out
        (
            IOobject("pointDisplacement.evaluated", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
            pd
        );
        out.write();
    }
    {
        pointVectorField pd(input);
        pcs.constrainDisplacement(pd);
        pointVectorField out
        (
            IOobject("pointDisplacement.constrained", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
            pd
        );
        out.write();
    }
    Info<< "[brae] wrote pointDisplacement.evaluated and pointDisplacement.constrained at "
        << runTime.timeName() << endl;

    Info<< "End\n" << endl;
    return 0;
}
