/*---------------------------------------------------------------------------*\
    dumpInterfaceK -- write OpenFOAM's OWN interface curvature and unit-normal
    flux, so brae's can be compared field by field.

    WHY THIS IS A UTILITY AND NOT A PATCHED COPY. The of-instrument pattern is
    to copy OpenFOAM's source and add writes, because the quantity wanted is
    usually private and computed mid-solve. interfaceProperties is not like
    that: nHatf() and sigmaK() are PUBLIC (interfaceProperties.H:139-144), and
    immiscibleIncompressibleTwoPhaseMixture inherits them. So the class runs
    UNMODIFIED here and what comes out is what OpenFOAM computes, with no
    transcription of its source at all -- which is a stronger guarantee than a
    copy with writes added.

    K is not exposed directly, but sigmaK() is, and sigma is a constant of the
    case; K = sigmaK/sigma is exact.

    Usage:  dumpInterfaceK -case <dir> [-time <t>]
    Writes: <time>/K.dump, <time>/nHatf.dump
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"

int main(int argc, char *argv[])
{
    argList::addNote("Write interfaceProperties' K and nHatf, from OpenFOAM's own class.");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    Info<< "Reading field alpha/U at t = " << runTime.timeName() << endl;

    volVectorField U
    (
        IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );

    #include "createPhi.H"

    // The mixture OWNS the interfaceProperties, and constructing it runs calculateK() -- the same
    // call interFoam makes, on the same fields, with the same dictionaries.
    immiscibleIncompressibleTwoPhaseMixture mixture(U, phi);

    // ...and correct() runs it again, which is what interFoam does every alpha corrector. Both are
    // written so a comparison can tell a construction-time curvature from a corrected one.
    // interfaceProperties' curvature is a FIXED POINT in its own passes -- each calculateK reads the
    // wall gradient the previous one wrote through correctContactAngle. The constructor runs pass 1;
    // BRAE_K_PASSES asks for more, so a comparison can be made pass for pass instead of guessing how
    // many the solver has run by the time it needs the force.
    {
        const char* np = getenv("BRAE_K_PASSES");
        const int n = np ? atoi(np) : 1;
        for (int i = 0; i < n; ++i) mixture.correct();
        Info<< "calculateK passes: " << (n + 1) << " (constructor + " << n << " correct)" << endl;
    }

    // sigma is not exposed by the mixture; read it from the same dictionary interfaceProperties does.
    const IOdictionary transportProperties
    (
        IOobject("transportProperties", runTime.constant(), mesh,
                 IOobject::MUST_READ_IF_MODIFIED, IOobject::NO_WRITE)
    );
    const dimensionedScalar sigma("sigma", dimMass/sqr(dimTime), transportProperties);
    Info<< "sigma = " << sigma.value() << ", cAlpha = " << mixture.cAlpha()
        << ", deltaN = " << mixture.deltaN().value() << endl;

    volScalarField K
    (
        IOobject("K.dump", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mixture.sigmaK()/sigma
    );
    K.write();

    surfaceScalarField nHatf
    (
        IOobject("nHatf.dump", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mixture.nHatf()
    );
    nHatf.write();

    // ...and the surface tension force itself, which is what the momentum equation actually sees.
    surfaceScalarField stf
    (
        IOobject("stf.dump", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mixture.surfaceTensionForce()
    );
    stf.write();

    // ...and alpha ITSELF, after correct(). Its boundaryField now carries whatever the contact-angle
    // patch's evaluate() left there, which is the quantity that feeds back into the next gradient --
    // and the one thing a comparison against the 0/ file cannot see.
    const volScalarField& a1 = mixture.alpha1();
    volScalarField alphaOut
    (
        IOobject("alphaAfterCorrect.dump", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        a1
    );
    alphaOut.write();

    Info<< "wrote K.dump, nHatf.dump, stf.dump, alphaAfterCorrect.dump in " << runTime.timeName() << endl;
    Info<< "End\n" << endl;
    return 0;
}
