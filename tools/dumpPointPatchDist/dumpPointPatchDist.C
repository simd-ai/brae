/*---------------------------------------------------------------------------*\
    brae instrument, not an OpenFOAM application.

    OpenFOAM's own pointPatchDist for a named set of patches, written as a
    pointScalarField -- and with it the scale field rigidBodyMeshMotion builds
    from it, which that class computes and does not write (rigidBodyMeshMotion.C:
    the `//scale.write();` two lines below it).

    It is the oracle for brae's point-to-patch distance: the wave that carries it
    (PointEdgeWave over externalPointEdgePoint) stops propagating an improvement
    smaller than 1% of the squared distance it already has, so the answer is NOT
    the exact nearest-patch-point distance and cannot be checked against one.

    usage: dumpPointPatchDist -patches '(floatingObject)' [-di 0.05 -do 0.35]
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pointMesh.H"
#include "pointFields.H"
#include "pointPatchDist.H"
#include "pointConstraints.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::addNote("write OpenFOAM's pointPatchDist, and the rigidBodyMeshMotion scale from it");
    argList::addOption("patches", "wordRes", "the patches to measure the distance to");
    argList::addOption("di", "scalar", "innerDistance, for the scale field");
    argList::addOption("do", "scalar", "outerDistance, for the scale field");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    const wordRes patchNames(args.getList<wordRe>("patches"));
    const labelHashSet patchSet(mesh.boundaryMesh().patchSet(patchNames));
    if (patchSet.empty())
    {
        FatalErrorInFunction << "no patch matched " << patchNames << exit(FatalError);
    }

    const pointMesh& pMesh = pointMesh::New(mesh);
    pointPatchDist pDist(pMesh, patchSet, mesh.points());

    pointScalarField dist
    (
        IOobject("pointPatchDist.dump", runTime.timeName(), mesh,
                 IOobject::NO_READ, IOobject::NO_WRITE),
        pMesh,
        dimensionedScalar(dimLength, Zero)
    );
    dist.primitiveFieldRef() = pDist.primitiveField();
    dist.write();
    Info<< "[brae] wrote pointPatchDist.dump for " << patchNames
        << " (" << pDist.nUnset() << " points the wave never reached)" << endl;

    if (args.found("di") && args.found("do"))
    {
        const scalar di = args.get<scalar>("di");
        const scalar dOut = args.get<scalar>("do");
        pointScalarField scale
        (
            IOobject("rbmScale.dump", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            pMesh,
            dimensionedScalar(dimless, Zero)
        );
        // rigidBodyMeshMotion.C: 1 up to di, then linear down to 0 at do, then made a cosine
        scale.primitiveFieldRef() =
            min(max((dOut - pDist.primitiveField())/(dOut - di), scalar(0)), scalar(1));
        scale.primitiveFieldRef() =
            min(max(0.5 - 0.5*cos(scale.primitiveField()*constant::mathematical::pi), scalar(0)),
                scalar(1));
        pointConstraints::New(pMesh).constrain(scale);
        scale.write();
        Info<< "[brae] wrote rbmScale.dump for innerDistance " << di
            << " outerDistance " << dOut << endl;
    }

    Info<< "End\n" << endl;
    return 0;
}

// ************************************************************************* //
