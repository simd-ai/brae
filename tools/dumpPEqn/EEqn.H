{
    volScalarField& he = thermo.he();

    fvScalarMatrix EEqn
    (
        fvm::div(phi, he)
      + (
            he.name() == "e"
          ? fvc::div(phi, volScalarField("Ekp", 0.5*magSqr(U) + p/rho))
          : fvc::div(phi, volScalarField("K", 0.5*magSqr(U)))
        )
      - fvm::laplacian(turbulence->alphaEff(), he)
     ==
        fvOptions(rho, he)
    );

    if (MRF.active())
    {
        EEqn += fvc::div(MRF.phi(), p);
    }

    // THE SYSTEM BEFORE relax(), as well as after. fvMatrix::relax adds the boundary internalCoeffs to
    // the diagonal, clamps it to max(|D|, sumMagOffDiag), divides by alpha, takes the boundary part back
    // off, and finally does S += (D - D0)*psi. Where that clamp is ACTIVE the relaxed D lands on sumOff
    // and carries no trace of D0 -- so a wrong unrelaxed diagonal shows up ONLY in the source. On
    // angledDuct's walls it does exactly that: D agrees to eleven digits while the source is 7.3e-05
    // out, and D0 was not being written for anything to compare.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volScalarField eD0
        (
            IOobject("stage_eD0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        volScalarField eSrc0(eD0); eSrc0.rename("stage_eSrc0");
        eD0.primitiveFieldRef() = EEqn.D();
        scalarField rhs0(EEqn.source());
        forAll(mesh.boundary(), pj)
        {
            const labelUList& fc = mesh.boundary()[pj].faceCells();
            const scalarField& bcp = EEqn.boundaryCoeffs()[pj];
            forAll(fc, i) rhs0[fc[i]] += bcp[i];
        }
        eSrc0.primitiveFieldRef() = rhs0;
        eD0.write(); eSrc0.write();
    }

    EEqn.relax();

    // Stage harness: the kinetic term and the assembled energy system at iteration 1.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volScalarField Ekp
        (
            IOobject("stage_Ekp", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            he.name() == "e" ? 0.5*magSqr(U) + p/rho : 0.5*magSqr(U)
        );
        Ekp.write();
        // The KINETIC TERM ITSELF, as OpenFOAM's own scheme produced it -- `div(phi,Ekp) bounded Gauss
        // upwind`, both halves. The field Ekp already agrees to 1e-16, so a disagreement in the assembled
        // source is this divergence and nothing else, and its boundary faces are where it would live.
        volScalarField keDiv
        (
            IOobject("stage_keDiv", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            he.name() == "e"
          ? fvc::div(phi, volScalarField("Ekp", 0.5*magSqr(U) + p/rho))
          : fvc::div(phi, volScalarField("K", 0.5*magSqr(U)))
        );
        keDiv.write();
        // phi as the energy equation is convected by, so the comparison is not measuring a different flux.
        surfaceScalarField("stage_phiE", phi).write();
        volScalarField heDump
        (
            IOobject("stage_he", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            he
        );
        heDump.write();
        volScalarField eD
        (
            IOobject("stage_eD", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        volScalarField eSrc(eD); eSrc.rename("stage_eSrc");
        eD.primitiveFieldRef() = EEqn.D();
        scalarField rhs(EEqn.source());
        forAll(mesh.boundary(), pj)
        {
            const labelUList& fc = mesh.boundary()[pj].faceCells();
            const scalarField& bcp = EEqn.boundaryCoeffs()[pj];
            forAll(fc, i) rhs[fc[i]] += bcp[i];
        }
        eSrc.primitiveFieldRef() = rhs;
        eD.write(); eSrc.write();
        volScalarField aEff
        (
            IOobject("stage_alphaEff", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            turbulence->alphaEff()()
        );
        aEff.write();
        // thermo.alphahe() == CpByCpv*alpha, the LAMINAR half of alphaEff with no turbulence in it, so
        // the solver's own transport can be asserted against OpenFOAM's without also needing the model's
        // nut and alphat at this exact point.
        volScalarField aHe
        (
            IOobject("stage_alphaEffLam", runTime.timeName(), mesh, IOobject::NO_READ,
                     IOobject::NO_WRITE),
            thermo.alphahe()
        );
        aHe.write();
    }



    fvOptions.constrain(EEqn);

    EEqn.solve();

    fvOptions.correct(he);

    thermo.correct();
}
