tmp<volScalarField> rAU;

if (correctPhi)
{
    rAU = new volScalarField
    (
        IOobject
        (
            "rAU",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("rAU", dimTime/dimDensity, 1)
    );

    if (!getenv("BRAE_STAGE_ALPHA"))
    {
        #include "correctPhi.H"
    }
}
else
{
    if (!getenv("BRAE_STAGE_ALPHA"))
    {
        CorrectPhi
        (
            U,
            phi,
            p_rgh,
            dimensionedScalar("rAUf", dimTime/rho.dimensions(), 1),
            geometricZeroField(),
            pimple
        );
    }

    #include "continuityErrs.H"
}
