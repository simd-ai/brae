// BRAE_STAGE_DUMP=1: the first step's EXPLICIT stages on the state the run starts from (run with -latestTime on a
// time directory a previous run wrote), written as s*.dump fields in that time directory, then the program stops.
// No linear solve and no limiter is involved, so serial and -parallel differ by round-off only and brae's
// per-rank stage dump (BRAE_MGPU_STAGE_DUMP, src/applications/solvers/interFoam/inter_stage_dump.cuh) is held
// against them quantity by quantity. Included from interFoam.C after createFields/createAlphaFluxes.
{
    #include "alphaControls.H"

    const word alphaScheme("div(phi,alpha)");
    const word alpharScheme("div(phirb,alpha)");

    if (icAlpha > 0 || scAlpha > 0)
    {
        FatalErrorInFunction << "the stage dump takes icAlpha = scAlpha = 0" << exit(FatalError);
    }

    // the state itself (diagnostics: what both sides read)
    volScalarField("sAlpha.dump", alpha1).write();
    surfaceScalarField("sPhi.dump", phi).write();

    // what mixture.correct() left at construction: the interface normal flux and the curvature of this alpha
    surfaceScalarField("sNHatf.dump", mixture.nHatf()).write();
    volScalarField("sSigmaK.dump", mixture.sigmaK()()).write();
    surfaceScalarField("sStf.dump", mixture.surfaceTensionForce()()).write();

    // rho = alpha1*rho1 + alpha2*rho2 and its snGrad (pEqn.H's phig reads it)
    volScalarField("sRho.dump", rho).write();
    surfaceScalarField("sSnGradRho.dump", fvc::snGrad(rho)).write();

    // alphaEqn.H:59-89 and :164-176 on the state, without the pre-solve: phic, phir, alphaPhiUn
    surfaceScalarField phic(mixture.cAlpha()*mag(phi/mesh.magSf()));
    surfaceScalarField::Boundary& phicBf = phic.boundaryFieldRef();
    forAll(phic.boundaryField(), patchi)
    {
        if (!phicBf[patchi].coupled())
        {
            phicBf[patchi] == 0;
        }
    }
    surfaceScalarField phir(phic*mixture.nHatf());
    surfaceScalarField alphaPhiUnS
    (
        fvc::flux(phi, alpha1, alphaScheme)
      + fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme), alpha1, alpharScheme)
    );
    surfaceScalarField("sPhic.dump", phic).write();
    surfaceScalarField("sPhir.dump", phir).write();
    surfaceScalarField("sAlphaPhiUn.dump", alphaPhiUnS).write();

    Info<< "[brae] stage dump written at time " << runTime.timeName() << endl;
}
