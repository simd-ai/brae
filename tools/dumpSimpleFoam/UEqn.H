    // Momentum predictor

    MRF.correctBoundaryVelocity(U);

    // Stage harness: U and phi EXACTLY as the momentum equation is assembled from -- after
    // correctBoundaryVelocity, before any matrix is built. Held separately from stage_Upost below
    // because fvMatrix's constructor calls updateCoeffs() on every boundary, so a patch OpenFOAM
    // recomputes per iteration still carries the PREVIOUS iteration's value here.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField("stage_Uass", U).write();
        surfaceScalarField("stage_phiU", phi).write();
        volScalarField("stage_pass", p).write();
        // The KINEMATIC eddy and effective viscosities -- simpleFoam's divDevReff carries nuEff, not
        // muEff. Both are written so a comparison can separate the closure from the assembly.
        volScalarField("stage_nut", turbulence->nut()()).write();
        volScalarField("stage_nuEff", turbulence->nuEff()()).write();
    }

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(U)
      + turbulence->divDevReff(U)
     ==
        fvOptions(U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    // The assembled system BEFORE relax(), so assembly and relaxation can be separated. relax()
    // replaces the diagonal with max(|D + cmptMax(cmptMag(iC))|, sumMagOffDiag)/alpha - cmptMin(iC)
    // and adds (D - D0)*psi to the source, so an assembly error can be invisible in A() -- which the
    // clamp absorbs -- and visible only here.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField("stage_Upost", U).write();

        volScalarField UDiag0
        (
            IOobject("stage_UDiag0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        UDiag0.primitiveFieldRef() = UEqn.diag();
        UDiag0.write();

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

        surfaceScalarField UUp0
        (
            IOobject("stage_UUpper0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero)
        );
        surfaceScalarField ULo0(UUp0); ULo0.rename("stage_ULower0");
        if (UEqn.hasUpper()) UUp0.primitiveFieldRef() = UEqn.upper();
        if (UEqn.hasLower()) ULo0.primitiveFieldRef() = UEqn.lower();
        else if (UEqn.hasUpper()) ULo0.primitiveFieldRef() = UEqn.upper();
        UUp0.write(); ULo0.write();

        volVectorField UIC0
        (
            IOobject("stage_UIC0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedVector(dimless, Zero), calculatedFvPatchVectorField::typeName
        );
        volVectorField UBC0
        (
            IOobject("stage_UBC0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedVector(dimless, Zero), calculatedFvPatchVectorField::typeName
        );
        forAll(mesh.boundary(), pi)
        {
            UIC0.boundaryFieldRef()[pi] = UEqn.internalCoeffs()[pi];
            UBC0.boundaryFieldRef()[pi] = UEqn.boundaryCoeffs()[pi];
        }
        UIC0.write(); UBC0.write();

        // sumMagOffDiag: the clamp relax() compares the diagonal against. Where the clamp binds, the
        // relaxed diagonal is a function of THIS and not of the assembled diagonal at all, so a
        // diagonal comparison that disagrees means nothing until this one is known to agree.
        volScalarField UsumOff0
        (
            IOobject("stage_UsumOff0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        scalarField so0(mesh.nCells(), Zero);
        if (UEqn.hasUpper() || UEqn.hasLower()) UEqn.sumMagOffDiag(so0);
        UsumOff0.primitiveFieldRef() = so0;
        UsumOff0.write();

        // The RAW row sum, diag + sum(offdiag), before relax(). SIMPLEC's rAtU = 1/(1/rAU - H1) is
        // V over the RELAXED row sum, and for a conservative convection scheme with `bounded` that sum
        // is a near-total cancellation -- the relaxed diagonal is O(sumMagOffDiag/alpha) while the sum
        // is O((1/alpha - 1)*sumMagOffDiag). Any absolute error in a coefficient therefore lands on
        // rAtU magnified by 1/(1/alpha - 1), which is 9 at alpha = 0.9. Writing the raw sum here says
        // whether the cancellation is already different before relaxation touches it.
        volScalarField URowSum0
        (
            IOobject("stage_URowSum0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        scalarField rs0(UEqn.diag());
        if (UEqn.hasUpper() || UEqn.hasLower())
        {
            const labelUList& lo = mesh.lduAddr().lowerAddr();
            const labelUList& up = mesh.lduAddr().upperAddr();
            const scalarField& uc = UEqn.upper();
            const scalarField& lc = UEqn.hasLower() ? UEqn.lower() : UEqn.upper();
            forAll(lo, f)
            {
                rs0[lo[f]] += uc[f];
                rs0[up[f]] += lc[f];
            }
        }
        URowSum0.primitiveFieldRef() = rs0;
        URowSum0.write();
    }

    UEqn.relax();

    fvOptions.constrain(UEqn);

    // The same system AFTER relax() and constrain() -- the state pEqn.H reads A(), H(), H1() from.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volScalarField UDiag
        (
            IOobject("stage_UDiag", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        UDiag.primitiveFieldRef() = UEqn.diag();
        UDiag.write();

        volVectorField USrc
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
        USrc.primitiveFieldRef() = rhsU;
        USrc.write();

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
        UIC.write(); UBC.write();

        // The mesh factors every coefficient above is a product of. When each factor agrees and the
        // product does not, a different factor is being multiplied -- which is what that reads as.
        volScalarField VV
        (
            IOobject("stage_V", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        VV.primitiveFieldRef() = mesh.V();
        VV.write();
        surfaceScalarField("stage_magSf", mesh.magSf()).write();
        surfaceScalarField("stage_weights", mesh.weights()).write();
        surfaceScalarField("stage_deltaCoeffs", mesh.deltaCoeffs()).write();
        surfaceScalarField("stage_nonOrthDeltaCoeffs", mesh.nonOrthDeltaCoeffs()).write();

        // fvc::grad(U), the explicit dev2 tensor and its divergence -- the three halves of the
        // viscous term that carry no implicit coefficient, so a source that disagrees with the
        // off-diagonals agreeing is one of these. The tensor's name is forced to the one the case's
        // divSchemes entry uses, because fvc::div looks its scheme up by the ARGUMENT's name.
        volTensorField gradUDump
        (
            IOobject("stage_UgradU", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            fvc::grad(U)
        );
        gradUDump.write();
        volScalarField nuEffDump("nuEff", turbulence->nuEff()());
        volTensorField devArg
        (
            IOobject("(nuEff*dev2(T(grad(U))))", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            nuEffDump*dev2(T(fvc::grad(U)))
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
                << "  type " << U.boundaryField()[pi].type()
                << "  sum|iC| = " << si << "  sum|bC| = " << sb << endl;
        }
    }

    // The ladder's matrix and starting field, captured HERE (where they are correct) and used AFTER the
    // real solve (where the logging is harmless). autoPtr so a non-dump iteration copies nothing.
    autoPtr<fvVectorMatrix> ladderEqn;
    autoPtr<volVectorField> ladderPsi0;

    // THE SMOOTHER LADDER. A COPY of the system the momentum predictor is about to solve -- the pressure
    // gradient included, which stage_USrc above does not carry -- written in the FOLDED form the linear
    // solver actually sees (fvMatrixSolve.C adds boundaryCoeffs into the source and internalCoeffs into
    // the diagonal before lduMatrix::solver::New), plus OpenFOAM's OWN residual after exactly n sweeps
    // for n = 1..LADDER_N.
    //
    // `tolerance 0; relTol 0; maxIter n` makes checkConvergence false forever (SolverPerformance.C), so
    // smoothSolver.C's do-while runs exactly n sweeps and still reports its own normFactor-scaled
    // residuals. Every number here is OpenFOAM's: its matrix, its fold, its normFactor, its smoother.
    // The real solve below is untouched -- the copy carries its own matrix and U is restored each rung.
    if (runTime.timeIndex() == braeDumpIter())
    {
        fvVectorMatrix UEqnP(UEqn == -fvc::grad(p));

        volScalarField sD
        (
            IOobject("stage_UsolveDiag", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        volScalarField sS(sD); sS.rename("stage_UsolveSrc");
        volScalarField s0(sD); s0.rename("stage_Usolve0");
        scalarField d(UEqnP.diag());
        scalarField b(UEqnP.source().component(0));
        forAll(mesh.boundary(), pj)
        {
            const labelUList& fc = mesh.boundary()[pj].faceCells();
            forAll(fc, i)
            {
                d[fc[i]] += UEqnP.internalCoeffs()[pj][i].x();
                b[fc[i]] += UEqnP.boundaryCoeffs()[pj][i].x();
            }
        }
        sD.primitiveFieldRef() = d;
        sS.primitiveFieldRef() = b;
        s0.primitiveFieldRef() = U.component(0)();
        sD.write(); sS.write(); s0.write();

        surfaceScalarField sU
        (
            IOobject("stage_UsolveUpper", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero)
        );
        surfaceScalarField sL(sU); sL.rename("stage_UsolveLower");
        if (UEqnP.hasUpper()) sU.primitiveFieldRef() = UEqnP.upper();
        if (UEqnP.hasLower()) sL.primitiveFieldRef() = UEqnP.lower();
        else if (UEqnP.hasUpper()) sL.primitiveFieldRef() = UEqnP.upper();
        sU.write(); sL.write();

        ladderEqn.reset(new fvVectorMatrix(UEqnP));
        ladderPsi0.reset(new volVectorField("stage_ladderPsi0", U));
    }

    if (simple.momentumPredictor())
    {
        solve(UEqn == -fvc::grad(p));

        fvOptions.correct(U);
    }

    // The ladder runs HERE, after the real solve, and not where its matrix was built: fvMatrixSolve.C
    // prints every solve unconditionally, and eleven gates parse this log -- several of them by FIRST
    // occurrence. Running the rungs after the predictor leaves the real solve's line first. Each rung
    // starts from the psi0 dumped as stage_Usolve0, so the ladder and that field correspond, and U is
    // put back afterwards so nothing downstream can tell this happened.
    if (runTime.timeIndex() == braeDumpIter() && ladderEqn)
    {
        const volVectorField Usolved("stage_ladderUsolved", U);
        OFstream os(runTime.path()/"stage_UsmoothLadder.dat");
        os.precision(16);
        for (label n = 1; n <= 10; ++n)
        {
            U == ladderPsi0();
            fvVectorMatrix Urung(ladderEqn());
            dictionary dct;
            dct.add("solver",    word("smoothSolver"));
            dct.add("smoother",  word("symGaussSeidel"));
            dct.add("tolerance", scalar(0));
            dct.add("relTol",    scalar(0));
            dct.add("maxIter",   n);
            const SolverPerformance<vector> sp = Urung.solve(dct);
            os << n << ' ' << sp.initialResidual().x() << ' ' << sp.finalResidual().x() << nl;
        }
        U == Usolved;
    }

    // U straight out of the momentum predictor. UEqn.H() below is built from THIS U, not the field the
    // matrix was assembled from, so it is a stage in its own right between rAU and HbyA.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField("stage_Upred", U).write();
    }
