{
    if (correctPhi)
    {
        rAU.ref() = 1.0/UEqn.A();
    }
    else
    {
        rAU = 1.0/UEqn.A();
    }

    surfaceScalarField rAUf("rAUf", fvc::interpolate(rAU()));
    volVectorField HbyA(constrainHbyA(rAU()*UEqn.H(), U, p_rgh));
    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::flux(HbyA)
      + MRF.zeroFilter(fvc::interpolate(rho*rAU())*fvc::ddtCorr(U, phi, Uf))
    );
    MRF.makeRelative(phiHbyA);

    if (p_rgh.needReference())
    {
        fvc::makeRelative(phiHbyA, U);
        adjustPhi(phiHbyA, U, p_rgh);
        fvc::makeAbsolute(phiHbyA, U);
    }

    surfaceScalarField phig
    (
        (
            mixture.surfaceTensionForce()
          - ghf*fvc::snGrad(rho)
        )*rAUf*mesh.magSf()
    );

    // ---- INSTRUMENTATION: writes only, nothing above or below is altered ----------------------
    // Gated on BRAE_DUMP_ITER so the solver runs normally and dumps one time step. Everything the
    // momentum-to-pressure hand-off is built from, because a 6% velocity difference with an exact
    // alpha, an exact curvature and a converged pressure solve has to be in one of these.
    if (getenv("BRAE_DUMP_ITER") && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
    {
        volScalarField("rAU.dump", rAU()).write();
        volVectorField("HbyA.dump", HbyA).write();
        surfaceScalarField("rAUf.dump", rAUf).write();
        surfaceScalarField("phig.dump", phig).write();
        surfaceScalarField("phiHbyA.dump", phiHbyA).write();
        surfaceScalarField("stf.dump", mixture.surfaceTensionForce()()).write();
        surfaceScalarField("snGradRho.dump", fvc::snGrad(rho)()).write();
        volScalarField("rho.dump", rho).write();
        volScalarField("UEqnA.dump", UEqn.A()()).write();
        volScalarField("muEff.dump", rho*mixture.nu()).write();
        volScalarField("nuMix.dump", mixture.nu()()).write();
        Info<< "[brae] dumped pEqn stage at timeIndex " << runTime.timeIndex() << endl;
    }
    // ------------------------------------------------------------------------------------------

    phiHbyA += phig;

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF);

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix p_rghEqn
        (
            fvm::laplacian(rAUf, p_rgh) == fvc::div(phiHbyA)
        );

        p_rghEqn.setReference(pRefCell, getRefCellValue(p_rgh, pRefCell));

        p_rghEqn.solve(p_rgh.select(pimple.finalInnerIter()));

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA - p_rghEqn.flux();

            p_rgh.relax();

            U = HbyA + rAU()*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf);
            U.correctBoundaryConditions();
            fvOptions.correct(U);
        }
    }

    #include "continuityErrs.H"

    // Correct Uf if the mesh is moving
    fvc::correctUf(Uf, U, phi);

    // Make the fluxes relative to the mesh motion
    fvc::makeRelative(phi, U);

    p == p_rgh + rho*gh;

    if (p_rgh.needReference())
    {
        p += dimensionedScalar
        (
            "p",
            p.dimensions(),
            pRefValue - getRefCellValue(p, pRefCell)
        );
        p_rgh = p - rho*gh;
    }

    if (!correctPhi)
    {
        rAU.clear();
    }
}
