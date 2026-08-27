    // Solve the Momentum equation

    MRF.correctBoundaryVelocity(U);

    // Stage harness: U EXACTLY as UEqn.H assembles from -- after correctBoundaryVelocity and after the
    // boundary conditions have been evaluated, before any matrix is built. Without this, a brae-side
    // comparison of the momentum matrix also measures brae's flowRateInletVelocity (which sbMatched's
    // inlet uses, and whose value depends on which rho it is fed). Those are separate components and a
    // single number covering both cannot be attributed to either.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField("stage_Uass", U).write();
        // rho EXACTLY as it stands when the momentum equation is assembled -- the field
        // flowRateInletVelocity::updateCoeffs looks up, so brae's inlet can be held against the same
        // density OpenFOAM held it against. It is the RELAXED solver rho from the previous iteration,
        // which is not thermo.rho() and not the face-cell values.
        volScalarField("stage_rhoU", rho).write();
        // phi EXACTLY as the momentum equation is convected by. From iteration 2 on this is the
        // CONSERVATIVE flux the previous pressure solve left, which is not what compressibleCreatePhi.H
        // would recompute from rho and U -- so a brae-side phi rebuilt at this point is a different field.
        surfaceScalarField("stage_phiU", phi).write();
    }

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(U)
     ==
        fvOptions(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    // The source BEFORE relax(), so the assembly can be separated from the relaxation. relax() replaces
    // the diagonal with max(|D|, sumMagOffDiag)/alpha and adds (D - D0)*psi to the source, so a pre-relax
    // diagonal difference can be invisible in A() and visible only here.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField USrc0
        (
            IOobject("stage_USrc0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedVector(dimless, Zero), zeroGradientFvPatchVectorField::typeName
        );
        vectorField rhs0(UEqn.source());
        forAll(mesh.boundary(), pj)
        {
            const labelUList& fc = mesh.boundary()[pj].faceCells();
            const vectorField& bcp = UEqn.boundaryCoeffs()[pj];
            forAll(fc, i) rhs0[fc[i]] += bcp[i];
        }
        USrc0.primitiveFieldRef() = rhs0;
        USrc0.write();
        volScalarField UDiag0
        (
            IOobject("stage_UDiag0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        UDiag0.primitiveFieldRef() = UEqn.diag();
        UDiag0.write();
    }

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (runTime.timeIndex() == braeDumpIter())
    {
        // The momentum matrix's SOURCE and OFF-DIAGONALS. rAU compares A() and stage_UIC/UBC compare the
        // boundary coefficients, but H() -- which is the whole of HbyA -- is built from the source and
        // the off-diagonals, and neither had an oracle. On aerofoilNACA0012 HbyA is 8.6e-06 out with A()
        // and both coefficient sets exact, so it has to be one of these.
        volVectorField USrcDump
        (
            IOobject("stage_USrc", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedVector(dimless, Zero), zeroGradientFvPatchVectorField::typeName
        );
        vectorField rhsU(UEqn.source());
        forAll(mesh.boundary(), pj)
        {
            const labelUList& fc = mesh.boundary()[pj].faceCells();
            const vectorField& bcp = UEqn.boundaryCoeffs()[pj];
            forAll(fc, i) rhsU[fc[i]] += bcp[i];
        }
        USrcDump.primitiveFieldRef() = rhsU;
        USrcDump.write();

        surfaceScalarField UUp
        (
            IOobject("stage_UUpper", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero)
        );
        surfaceScalarField ULo(UUp); ULo.rename("stage_ULower");
        if (UEqn.hasUpper()) UUp.primitiveFieldRef() = UEqn.upper();
        if (UEqn.hasLower()) ULo.primitiveFieldRef() = UEqn.lower();
        else if (UEqn.hasUpper()) ULo.primitiveFieldRef() = UEqn.upper();
        UUp.write(); ULo.write();

        // fvc::grad(U) as the momentum source's explicit dev2 term is built from, and the term itself.
        // The source is 1.0e-06 out on aerofoilNACA0012 and 1.2e-04 on wall-adjacent cells, so it is one
        // of these two; the gradient's BOUNDARY carries gaussGrad's correction (normal component replaced
        // by the patch snGrad), which is exactly what a no-slip wall makes large.
        volTensorField gradUDump
        (
            IOobject("stage_UgradU", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            fvc::grad(U)
        );
        gradUDump.write();

        // The dev2 TENSOR itself -- (rho*nuEff)*dev2(T(grad(U))) -- and then its divergence. The tensor
        // needs no scheme lookup, so it isolates the two halves: if the tensor agrees and the divergence
        // does not, the difference is in the div operator's BOUNDARY faces rather than in the field.
        // The names are forced to the ones the case's divSchemes entry uses
        // (`div(((rho*nuEff)*dev2(T(grad(U)))))`), because fvc::div looks its scheme up by the ARGUMENT's
        // name and a freshly-composed expression gets a different one.
        volScalarField rhoNuEff("(rho*nuEff)", turbulence->muEff());
        volTensorField devArg
        (
            IOobject("((rho*nuEff)*dev2(T(grad(U))))", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            rhoNuEff*dev2(T(fvc::grad(U)))
        );
        volTensorField devCopy
        (
            IOobject("stage_UdevTensor", runTime.timeName(), mesh, IOobject::NO_READ,
                     IOobject::NO_WRITE),
            devArg
        );
        devCopy.write();
        volVectorField devDiv
        (
            IOobject("stage_UdevDiv", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            fvc::div(devArg)
        );
        devDiv.write();


        // cell volume, so the porosity's V*tr(Cd) contribution can be decomposed from D = V/rAU
        volScalarField VV
        (
            IOobject("stage_V", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        VV.primitiveFieldRef() = mesh.V();
        VV.write();
    }

    // Stage harness: the MOMENTUM equation's boundary coefficients, per patch, AFTER relax() and
    // constrain() -- the same state brae's iC/bC are in. internalCoeffs goes to the adjacent cell's
    // diagonal, boundaryCoeffs to its source. With the diagonal proven right and the source proven wrong,
    // these are the whole remaining question.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField UIC
        (
            IOobject("stage_UIC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedVector(dimless, Zero), calculatedFvPatchVectorField::typeName
        );
        volVectorField UBC
        (
            IOobject("stage_UBC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedVector(dimless, Zero), calculatedFvPatchVectorField::typeName
        );
        forAll(mesh.boundary(), pi)
        {
            UIC.boundaryFieldRef()[pi] = UEqn.internalCoeffs()[pi];
            UBC.boundaryFieldRef()[pi] = UEqn.boundaryCoeffs()[pi];
        }
        UIC.write();
        UBC.write();
        forAll(mesh.boundary(), pi)
        {
            const fvPatch& pp = mesh.boundary()[pi];
            if (!pp.size()) continue;
            vector si(Zero), sb(Zero);
            forAll(UEqn.internalCoeffs()[pi], i)
            {
                si += cmptMag(UEqn.internalCoeffs()[pi][i]);
                sb += cmptMag(UEqn.boundaryCoeffs()[pi][i]);
            }
            Info<< "  UEqn patch " << pp.name() << " (" << pp.size() << " faces)"
                << "  sum|iC| = " << si << "  sum|bC| = " << sb << endl;
        }
    }

    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);

    // Stage harness: U straight out of the MOMENTUM PREDICTOR. UEqn.H() below uses THIS U, not the
    // initial field, so it is a stage in its own right and sits between rAU and HbyA in the chain.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField("stage_Upred", U).write();
        volScalarField("stage_muEff", turbulence->muEff()).write();
        // thermo.mu(), the LAMINAR half of muEff with no turbulence in it -- the twin of
        // stage_alphaEffLam. muEff is INJECTED into the momentum gate so that its assembly number stays
        // attributable, which also means brae's own viscosity had never been compared to OpenFOAM's.
        // Its sibling alphaEff turned out to be six orders wrong behind exactly that injection.
        volScalarField("stage_muLam", thermo.mu()).write();
    }
