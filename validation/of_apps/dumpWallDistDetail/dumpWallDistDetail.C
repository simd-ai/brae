// dumpWallDistDetail: OpenFOAM's correctWalls, with its CHOICES exposed.
//
// WHY. brae's cellWallDist agrees with wallDist::New(mesh).y() to L2 1.7e-04 overall, but disagrees by
// more than 1% on 963 cells of the motorBike mesh -- 917 of them exactly one face-layer off the wall,
// which is correctBoundaryPointCells' territory. Reading cellDistFuncs.C and porting the ordering it
// describes (first (patch, meshPoint) wins, that point's patch-local faces only, then the cell is locked)
// made brae WORSE, not better: it moved from reading a median 0.894 of OpenFOAM to 1.239, i.e. from too
// small to too large. So OpenFOAM's answer sits BETWEEN the honest global minimum and the literal reading
// of its own source, and no amount of further reading settles which faces it actually used.
//
// This asks it directly. cellDistFuncs' correction routines are public and take the nearestFace Map by
// reference, so calling them exactly as patchWave::correct() does and then writing that Map out gives, per
// cell, THE WALL FACE OPENFOAM CHOSE -- not an inference about the rule, the rule's output.
//
// Writes two fields at the current time:
//   yCorr       the distance after correctBoundaryFaceCells + correctBoundaryPointCells
//   yCorrFace   the global face index OpenFOAM settled on, or -1 for a cell neither pass claimed
#include "fvCFD.H"
#include "cellDistFuncs.H"
#include "wallPolyPatch.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    const cellDistFuncs wallUtils(mesh);
    // wallDist's own patch selection: every patch of type wall.
    const labelHashSet wallPatchIDs(wallUtils.getPatchIDs<wallPolyPatch>());

    Info<< "wall patches: " << wallPatchIDs.size() << endl;

    // patchWave::correct() seeds distance_ from the wave; here only the corrections matter, so start
    // from GREAT and report which cells the two passes actually touch.
    scalarField dist(mesh.nCells(), GREAT);
    Map<label> nearestFace(2*wallUtils.sumPatchSize(wallPatchIDs));

    wallUtils.correctBoundaryFaceCells(wallPatchIDs, dist, nearestFace);
    const label nFaceCells = nearestFace.size();
    wallUtils.correctBoundaryPointCells(wallPatchIDs, dist, nearestFace);

    Info<< "cells claimed by correctBoundaryFaceCells : " << nFaceCells << nl
        << "cells claimed by correctBoundaryPointCells: "
        << (nearestFace.size() - nFaceCells) << nl
        << "cells claimed in total                    : " << nearestFace.size() << endl;

    volScalarField yCorr
    (
        IOobject("yCorr", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar(dimLength, GREAT)
    );
    volScalarField yCorrFace
    (
        IOobject("yCorrFace", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar(dimless, -1)
    );
    forAll(dist, celli)
    {
        yCorr[celli] = dist[celli];
    }
    forAllConstIters(nearestFace, iter)
    {
        yCorrFace[iter.key()] = scalar(iter.val());
    }
    yCorr.write();
    yCorrFace.write();

    Info<< "dumpWallDistDetail: wrote yCorr + yCorrFace" << endl;
    return 0;
}
