    MRF.correctBoundaryVelocity(U);

    fvVectorMatrix UEqn
    (
        fvm::ddt(rho, U) + fvm::div(rhoPhi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(rho, U)
     ==
        fvOptions(rho, U)
    );

    // ---- INSTRUMENTATION: writes only, nothing above or below is altered ----------------------
    // UEqn's SOURCE, TERM BY TERM, and BEFORE relax(). With the matrix, V, rAU and psi all measured
    // exact on permeable-moving, the source is the only term of UEqn.H() still out (3.82e-04
    // relative, 6.526e-10 absolute against a field whose largest value is 1.710e-06) -- so which of
    // the three terms carries a quantity of that size is the question, and the assembled source
    // cannot answer it.
    //
    // THE PRE-RELAX SOURCE IS TAKEN FROM THE REAL MATRIX, not rebuilt: fvMatrix::relax adds
    // (D - D0)*psi to it, and this case gets that contribution even at `"U.*" 1` because relax runs
    // max(D, D, sumOff) before dividing by alpha (fvMatrix.C). The difference between this dump and
    // UEqnSource.dump below IS the relaxation contribution.
    //
    // THE THREE PIECES ARE RE-EVALUATED, which is safe for fvm::div and divDevRhoReff (both are pure
    // functions of their arguments) but NOT for every ddt scheme: CrankNicolsonDdtScheme keeps a
    // stored ddt0 field. The ddt piece is therefore dumped only under a stateless scheme, and the log
    // says so when it is skipped rather than leaving the file silently absent.
    if (getenv("BRAE_DUMP_ITER") && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
    {
        const fvVectorMatrix& UEqnPre = UEqn;

        volVectorField srcPreRelax
        (
            IOobject("UEqnSourcePreRelax.dump", runTime.timeName(), mesh,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            mesh,
            dimensionedVector(dimless, Zero)
        );
        srcPreRelax.primitiveFieldRef() = UEqnPre.source();
        srcPreRelax.write();

        auto dumpSource = [&](const word& name, const fvVectorMatrix& mtx)
        {
            volVectorField f
            (
                IOobject(name, runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh,
                dimensionedVector(dimless, Zero)
            );
            f.primitiveFieldRef() = mtx.source();
            f.write();
            scalar mx = 0;
            forAll(mtx.source(), celli) mx = max(mx, mag(mtx.source()[celli]));
            Info<< "[brae] " << name << "  max|source| " << mx << endl;
        };

        {
            const tmp<fvVectorMatrix> tdiv(fvm::div(rhoPhi, U));
            dumpSource("UEqnSourceDiv.dump", tdiv());
        }
        {
            const tmp<fvVectorMatrix> tdev(turbulence->divDevRhoReff(rho, U));
            dumpSource("UEqnSourceDev.dump", tdev());
        }

        // ONE LEVEL DEEPER INTO divDevRhoReff, which the bisection above lands on: brae's dev source
        // is 9.330e-10 of OpenFOAM's 2.167e-09 at step two, 43 per cent, while ddt is exact to
        // 5.3e-14 and div(rhoPhi,U) contributes no source at all. Its explicit half is
        //     - fvc::div((alpha*rho*nuEff)*dev2(T(fvc::grad(U))))    linearViscousStress.C:127-131
        // so grad(U), the viscosity it is weighted by, and the divergence of the product are the
        // three things that can carry it. This case is `simulationType laminar`, where nuEff() is
        // the mixture's own nu and alpha is one, so rho*mixture.nu() IS the weight OpenFOAM uses.
        {
            // NAMED `(rho*nuEff)` DELIBERATELY: fvc::div looks its scheme up by the composed field
            // name, and this case's divSchemes lists
            // `div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear` with NO default entry -- any other
            // name is a fatal lookup, not a silent fallback.
            const volScalarField muA("(rho*nuEff)", rho*mixture.nu());
            volScalarField("muEffAsm.dump", muA).write();
            volTensorField("gradUasm.dump", fvc::grad(U)()).write();
            volVectorField("divDevExpl.dump", fvc::div(muA*dev2(T(fvc::grad(U))))()).write();
        }

        const token& ddtTok = mesh.ddtScheme("ddt(rho,U)").peekFirst();
        const word ddtName(ddtTok.isWord() ? ddtTok.wordToken() : word("<not-a-word>"));
        if (ddtName == "Euler" || ddtName == "backward" || ddtName == "steadyState")
        {
            const tmp<fvVectorMatrix> tddt(fvm::ddt(rho, U));
            dumpSource("UEqnSourceDdt.dump", tddt());
        }
        else
        {
            Info<< "[brae] UEqnSourceDdt.dump SKIPPED: ddt scheme `" << ddtName
                << "` keeps stored state, and re-evaluating it here would move it" << endl;
        }

        scalar mxPre = 0;
        forAll(UEqnPre.source(), celli) mxPre = max(mxPre, mag(UEqnPre.source()[celli]));
        Info<< "[brae] UEqnSourcePreRelax.dump  max|source| " << mxPre
            << "  (ddt scheme " << ddtName << ")" << endl;
    }
    // ------------------------------------------------------------------------------------------

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
