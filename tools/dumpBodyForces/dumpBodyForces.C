/*---------------------------------------------------------------------------*\
    brae instrument, not an OpenFOAM application.

    The fluid force and moment on a body's patches, computed by OpenFOAM's own
    functionObjects::forces with the dictionary rigidBodyMeshMotion::solve builds
    for it (rigidBodyMeshMotion.C:296-307): type forces, the body's patches,
    rhoInf, rho and CofR (0 0 0).

    Prints forceEff() and momentEff() -- the two numbers that become the body's
    spatial force -- and writes the per-face pressure and viscous parts beside
    them, so a mismatch can be taken apart rather than guessed at.

    usage: dumpBodyForces -patches '(floatingObject)' [-rho rho] [-rhoInf 1]
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "timeSelector.H"
#include "forces.H"
#include "turbulentTransportModel.H"
#include "incompressibleInterPhaseTransportModel.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::addNote("OpenFOAM's own forces on a body's patches, as rigidBodyMeshMotion asks for them");
    argList::addOption("patches", "wordRes", "the body's patches");
    argList::addOption("rho", "word", "the density field name (default rho)");
    argList::addOption("rhoInf", "scalar", "the reference density when rho is rhoInf (default 1)");

    timeSelector::addOptions();

    #include "setRootCase.H"
    #include "createTime.H"
    // the instant to measure at: -time <t> picks the written directory, and the mesh is created at
    // it so a MOVED mesh is read from <t>/polyMesh rather than from constant/
    const instantList timeDirs = timeSelector::select0(runTime, args);
    if (timeDirs.empty())
    {
        FatalErrorInFunction << "no time directory selected" << exit(FatalError);
    }
    runTime.setTime(timeDirs.last(), timeDirs.size() - 1);
    #include "createMesh.H"

    const wordRes patchNames(args.getList<wordRe>("patches"));
    const word rhoName(args.getOrDefault<word>("rho", "rho"));
    const scalar rhoInf(args.getOrDefault<scalar>("rhoInf", 1.0));

    // The fields forces looks up, and the transport model devRhoReff asks for. interFoam registers
    // them in createFields.H; this instrument has to do it by hand because it is not that solver.
    Info<< "Reading field p_rgh, U, alpha and the two-phase mixture" << endl;
    volScalarField p_rgh
    (
        IOobject("p_rgh", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    volVectorField U
    (
        IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    #include "createPhi.H"
    immiscibleIncompressibleTwoPhaseMixture mixture(U, phi);
    volScalarField& alpha1(mixture.alpha1());
    volScalarField& alpha2(mixture.alpha2());
    const dimensionedScalar& rho1 = mixture.rho1();
    const dimensionedScalar& rho2 = mixture.rho2();
    volScalarField rho
    (
        IOobject("rho", runTime.timeName(), mesh, IOobject::READ_IF_PRESENT),
        alpha1*rho1 + alpha2*rho2
    );
    rho.oldTime();
    #include "readGravitationalAcceleration.H"
    #include "readhRef.H"
    #include "gh.H"
    volScalarField p
    (
        IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    surfaceScalarField rhoPhi
    (
        IOobject("rhoPhi", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        fvc::interpolate(rho)*phi
    );
    incompressibleInterPhaseTransportModel<immiscibleIncompressibleTwoPhaseMixture>
        turbulence(rho, U, phi, rhoPhi, mixture);

    dictionary forcesDict;
    forcesDict.add("type", functionObjects::forces::typeName);
    forcesDict.add("patches", patchNames);
    forcesDict.add("rhoInf", rhoInf);
    forcesDict.add("rho", rhoName);
    forcesDict.add("CofR", vector::zero);
    forcesDict.add("writeFields", true);

    functionObjects::forces f("forces", mesh.thisDb(), forcesDict);
    f.calcForcesMoments();

    const vector fEff = f.forceEff();
    const vector mEff = f.momentEff();
    Info<< setprecision(18)
        << "[brae] forceEff  " << fEff.x() << " " << fEff.y() << " " << fEff.z() << nl
        << "[brae] momentEff " << mEff.x() << " " << mEff.y() << " " << mEff.z() << endl;

    // ...AND THE PER-FACE FORCE AND MOMENT, by forces' own `writeFields`, which writes the
    // `force` and `moment` volVectorFields whose boundary values on the body's patches are fP + fV
    // and mP + mV. Taken from OpenFOAM rather than recomputed here: devRhoReff picks its viscous
    // model by which turbulence object is registered (forces.C:770-820), and an instrument that
    // guessed that branch would be checking its own guess.
    f.write();

    // ...AND grad(U), the one input to the viscous half that is not a field on disk. forces takes it
    // from fvc::grad(U) and then gaussGrad::correctBoundaryConditions replaces the wall-normal
    // component of the boundary value by snGrad, so the patch value is not the face cell's gradient.
    {
        volTensorField gradU("gradU.dump", fvc::grad(U));
        gradU.write();
        // the model forces::devRhoReff itself finds: an incompressible::turbulenceModel registered
        // under `turbulenceProperties`, which is what incompressibleInterPhaseTransportModel puts
        // there. The wrapper has no nuEff of its own.
        const incompressible::turbulenceModel& turb =
            mesh.lookupObject<incompressible::turbulenceModel>(
                incompressible::turbulenceModel::propertiesName);
        volScalarField nuEff("nuEff.dump", turb.nuEff()());
        nuEff.write();
        volScalarField rhoOut("rhoAtForces.dump", rho);
        rhoOut.write();
    }

    Info<< "End\n" << endl;
    return 0;
}

// ************************************************************************* //
