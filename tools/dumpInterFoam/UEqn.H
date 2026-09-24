    MRF.correctBoundaryVelocity(U);

    fvVectorMatrix UEqn
    (
        fvm::ddt(rho, U) + fvm::div(rhoPhi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(rho, U)
     ==
        fvOptions(rho, U)
    );

    UEqn.relax();

    // ---- INSTRUMENTATION: writes only, nothing above or below is altered ----------------------
    // THE QUESTION THIS SETTLES. brae's HOST and DEVICE paths disagree completely on UEqn's
    // boundaryCoeffs at damBreak's pressureInletOutletVelocity atmosphere -- host 3.3406e-06, device
    // identically zero, every other patch agreeing at zero -- and it carries HbyA's whole remaining
    // error. transformFvPatchField gives bC = value - cmptMultiply(iC, patchInternalField), and
    // whether that is zero depends on whether the STORED patch value equals the extrapolation at this
    // instant. Two brae implementations cannot settle it; OpenFOAM's own number can.
    //
    // AFTER relax(), because that is where brae reads it: fvMatrix::relax touches internalCoeffs
    // (fvMatrix.C:1102-1107) and a dump taken before it would be a different matrix.
    if (getenv("BRAE_DUMP_ITER") && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
    {
        // U AS THE ASSEMBLY SEES IT. A mixed condition writes its refValue, refGradient and
        // valueFraction (mixedFvPatchField::write), so a copy written here carries the pair the
        // boundaryCoeffs above were built from -- which is what nothing written at the END of a step
        // can show, because the patch is evaluated again between the two moments.
        volVectorField
        (
            IOobject("Uasm.dump", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            U
        ).write();
        forAll(mesh.boundary(), patchi)
        {
            const fvPatch& pp = mesh.boundary()[patchi];
            if (pp.size() == 0) continue;
            const vectorField& bc = UEqn.boundaryCoeffs()[patchi];
            const vectorField& ic = UEqn.internalCoeffs()[patchi];
            const fvPatchVectorField& Ub = U.boundaryField()[patchi];
            const vectorField pif(Ub.patchInternalField());

            scalar mxBC = 0, mxIC = 0, mxUb = 0, mxPif = 0, mxResid = 0;
            forAll(bc, i)
            {
                mxBC    = max(mxBC,    mag(bc[i]));
                mxIC    = max(mxIC,    mag(ic[i]));
                mxUb    = max(mxUb,    mag(Ub[i]));
                mxPif   = max(mxPif,   mag(pif[i]));
                // NOT the transformFvPatchField identity. internalCoeffs/boundaryCoeffs on the
                // ASSEMBLED matrix are the sum of the div and laplacian contributions, each already
                // scaled by its own face coefficient -- they are not the per-BC valueInternalCoeffs /
                // valueBoundaryCoeffs that identity relates. This is kept only as a magnitude, and the
                // |bC| column is what answers the question.
                const vector r = Ub[i] - cmptMultiply(ic[i], pif[i]) - bc[i];
                mxResid = max(mxResid, mag(r));
            }
            Info<< "[brae] UEqn bnd  " << pp.name()
                << "  type " << Ub.type()
                << "  |bC| "   << mxBC
                << "  |iC| "   << mxIC
                << "  |U_b| "  << mxUb
                << "  |pif| "  << mxPif
                << "  (|U_b - iC*pif - bC| " << mxResid << ", see note)"
                << endl;
        }
        Info<< "[brae] dumped UEqn boundary at timeIndex " << runTime.timeIndex() << endl;
    }
    // ------------------------------------------------------------------------------------------

    fvOptions.constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                    mixture.surfaceTensionForce()
                  - ghf*fvc::snGrad(rho)
                  - fvc::snGrad(p_rgh)
                ) * mesh.magSf()
            )
        );

        fvOptions.correct(U);
    }

    // ---- INSTRUMENTATION: writes only, nothing above is altered -------------------------------
    // THE MOMENTUM MATRIX ITSELF, as pEqn.H reads it. UEqn.H() = (lduMatrix::H(psi) + source
    // + boundarySource)/V. On the permeable-moving case every other term of that expression has
    // been measured exact -- V, A, the ddt source, the patch's refValue/valueFraction and the
    // assembled internalCoeffs/boundaryCoeffs -- while HbyA is still 3.556e-01 out. What is left is
    // the OFF-DIAGONALS and the explicit half of the source, and nothing written at the end of a
    // step can show either.
    //
    // THROUGH A CONST REFERENCE, deliberately. The non-const lduMatrix::lower() ALLOCATES lowerPtr_
    // as a copy of upper (lduMatrix.C:333-342), which turns a symmetric matrix asymmetric and
    // changes the solver chosen for it. An instrument must not move the thing it measures.
    if (getenv("BRAE_DUMP_ITER") && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
    {
        const fvVectorMatrix& UEqnC = UEqn;

        surfaceScalarField UEqnUpper
        (
            IOobject("UEqnUpper.dump", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            mesh,
            dimensionedScalar(dimless, Zero)
        );
        UEqnUpper.primitiveFieldRef() = UEqnC.upper();
        UEqnUpper.write();

        surfaceScalarField UEqnLower
        (
            IOobject("UEqnLower.dump", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            mesh,
            dimensionedScalar(dimless, Zero)
        );
        UEqnLower.primitiveFieldRef() = UEqnC.lower();
        UEqnLower.write();

        volScalarField UEqnDiag
        (
            IOobject("UEqnDiag.dump", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            mesh,
            dimensionedScalar(dimless, Zero)
        );
        UEqnDiag.primitiveFieldRef() = UEqnC.diag();
        UEqnDiag.write();

        volVectorField UEqnSource
        (
            IOobject("UEqnSource.dump", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            mesh,
            dimensionedVector(dimless, Zero)
        );
        UEqnSource.primitiveFieldRef() = UEqnC.source();
        UEqnSource.write();

        // ...and the two FACE COEFFICIENTS the off-diagonals are built from, so a difference in
        // upper can be told from a difference in what it was assembled out of: rhoPhi carries the
        // div, muEff the laplacian. muEff.dump is already written by pEqn.H.
        surfaceScalarField("rhoPhi.dump", rhoPhi).write();

        Info<< "[brae] dumped the UEqn matrix at timeIndex " << runTime.timeIndex()
            << " (symmetric " << UEqnC.symmetric() << ", asymmetric " << UEqnC.asymmetric() << ")"
            << endl;
    }
    // ------------------------------------------------------------------------------------------
