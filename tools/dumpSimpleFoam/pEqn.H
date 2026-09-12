{
    volScalarField rAU(1.0/UEqn.A());
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));

    // Stage harness: the SIMPLE chain's first three quantities, before MRF or adjustPhi moves the flux.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volScalarField("stage_A", UEqn.A()()).write();
        volScalarField("stage_rAU", rAU).write();
        volScalarField UD
        (
            IOobject("stage_UD", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        UD.primitiveFieldRef() = UEqn.D()();
        UD.write();
        volVectorField("stage_UH", UEqn.H()()).write();
        volVectorField("stage_HbyA0", HbyA).write();
        surfaceScalarField("stage_phiHbyA0", phiHbyA).write();
    }

    MRF.makeRelative(phiHbyA);
    adjustPhi(phiHbyA, U, p);

    if (runTime.timeIndex() == braeDumpIter())
    {
        surfaceScalarField("stage_phiHbyA1", phiHbyA).write();
    }

    tmp<volScalarField> rAtU(rAU);

    if (simple.consistent())
    {
        rAtU = 1.0/(1.0/rAU - UEqn.H1());
        phiHbyA +=
            fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p)*mesh.magSf();
        HbyA -= (rAU - rAtU())*fvc::grad(p);
    }

    // Stage harness: SIMPLEC's own quantities. rAtU is V over the RELAXED row sum of the momentum
    // matrix, and that sum is a near-total cancellation -- the relaxed diagonal is O(sumMagOffDiag/alpha)
    // while the row sum is O((1/alpha - 1)*sumMagOffDiag). So H1 is dumped in its own right and not left
    // to be inferred from rAtU: an absolute coefficient error that is invisible in rAU reaches rAtU
    // magnified by 1/(1/alpha - 1), which is 9 at the alpha 0.9 this fixture asks for.
    //
    // stage_rowSum is written as the literal 1/rAU - H1 that pEqn.H inverts, so a comparison does not
    // have to reconstruct it from two fields and inherit their difference twice.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volScalarField("stage_H1", UEqn.H1()()).write();
        volScalarField("stage_rowSum", 1.0/rAU - UEqn.H1()).write();
        volScalarField("stage_rAtU", rAtU()).write();
        volVectorField("stage_HbyA", HbyA).write();
        surfaceScalarField("stage_phiHbyA", phiHbyA).write();
        surfaceScalarField("stage_snGradP", fvc::snGrad(p)()).write();
        volVectorField("stage_gradP", fvc::grad(p)()).write();
    }

    tUEqn.clear();

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p, U, phiHbyA, rAtU(), MRF);

    // Non-orthogonal pressure corrector loop
    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
        );

        pEqn.setReference(pRefCell, pRefValue);

        // The pressure system as assembled, before setReference has any effect on the solve and before
        // the solver touches it. The laplacian's diffusivity is rAtU, so a rAtU that is wrong shows up
        // here as an off-diagonal that is wrong by the same factor.
        if (runTime.timeIndex() == braeDumpIter() && simple.finalNonOrthogonalIter())
        {
            volScalarField pDiag
            (
                IOobject("stage_pDiag", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
            );
            pDiag.primitiveFieldRef() = pEqn.D()();
            pDiag.write();

            volScalarField pSrc
            (
                IOobject("stage_pSrc", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
            );
            scalarField rhsP(pEqn.source());
            forAll(mesh.boundary(), pj)
            {
                const labelUList& fc = mesh.boundary()[pj].faceCells();
                const scalarField& bcp = pEqn.boundaryCoeffs()[pj];
                forAll(fc, i) rhsP[fc[i]] += bcp[i];
            }
            pSrc.primitiveFieldRef() = rhsP;
            pSrc.write();

            surfaceScalarField pUp
            (
                IOobject("stage_pUpper", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero)
            );
            if (pEqn.hasUpper()) pUp.primitiveFieldRef() = pEqn.upper();
            pUp.write();

            volScalarField pIC
            (
                IOobject("stage_pIC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), calculatedFvPatchScalarField::typeName
            );
            volScalarField pBC
            (
                IOobject("stage_pBC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), calculatedFvPatchScalarField::typeName
            );
            forAll(mesh.boundary(), pi)
            {
                pIC.boundaryFieldRef()[pi] = pEqn.internalCoeffs()[pi];
                pBC.boundaryFieldRef()[pi] = pEqn.boundaryCoeffs()[pi];
            }
            pIC.write(); pBC.write();
        }

        pEqn.solve();

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA - pEqn.flux();
        }
    }

    #include "continuityErrs.H"

    // Explicitly relax pressure for momentum corrector
    p.relax();

    // Momentum corrector
    U = HbyA - rAtU()*fvc::grad(p);
    U.correctBoundaryConditions();
    fvOptions.correct(U);

    // The end of the step: p and U after the corrector, and the conservative flux the next momentum
    // assembly convects by.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volScalarField("stage_pOut", p).write();
        volVectorField("stage_Uout", U).write();
        surfaceScalarField("stage_phiOut", phi).write();
    }
}
