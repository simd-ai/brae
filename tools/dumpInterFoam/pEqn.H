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
        // ONE SUFFIX PER PRESSURE CORRECTOR. Every field written here is rebuilt by each corrector,
        // and this block used to write the same eleven names on all of them -- so the files left on
        // disk were the LAST corrector's while brae's own taps are the FIRST's (inter_peqn_cpp.cu:851
        // pins them to corrector 0 for exactly this reason, between the two brae arms). Comparing
        // those two reads as a defect in whichever term moves between correctors: it put HbyA
        // 3.556e-01 and phiHbyA 1.786e-01 out on permeable-moving while the momentum MATRIX behind
        // them agrees to 1.5e-09. corrPISO() is 1 on the first corrector
        // (pimpleControlI.H:59), and that one keeps the bare name so every existing probe now reads
        // the corrector its brae side tapped.
        const Foam::word sfx =
            (pimple.corrPISO() <= 1) ? Foam::word() : ("c" + Foam::name(pimple.corrPISO()));
        volScalarField("rAU" + sfx + ".dump", rAU()).write();
        volVectorField("HbyA" + sfx + ".dump", HbyA).write();
        surfaceScalarField("rAUf" + sfx + ".dump", rAUf).write();
        surfaceScalarField("phig" + sfx + ".dump", phig).write();
        surfaceScalarField("phiHbyA" + sfx + ".dump", phiHbyA).write();
        surfaceScalarField("stf" + sfx + ".dump", mixture.surfaceTensionForce()()).write();
        surfaceScalarField("snGradRho" + sfx + ".dump", fvc::snGrad(rho)()).write();
        volScalarField("rho" + sfx + ".dump", rho).write();
        volScalarField("UEqnA" + sfx + ".dump", UEqn.A()()).write();
        volScalarField("muEff" + sfx + ".dump", rho*mixture.nu()).write();
        volScalarField("nuMix" + sfx + ".dump", mixture.nu()()).write();
        Info<< "[brae] dumped pEqn stage at timeIndex " << runTime.timeIndex()
            << " corrector " << pimple.corrPISO() << endl;
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
