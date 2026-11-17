{
    word alphaScheme("div(phi,alpha)");
    word alpharScheme("div(phirb,alpha)");

    // Set the off-centering coefficient according to ddt scheme
    scalar ocCoeff = 0;
    {
        tmp<fv::ddtScheme<scalar>> tddtAlpha
        (
            fv::ddtScheme<scalar>::New
            (
                mesh,
                mesh.ddtScheme("ddt(alpha)")
            )
        );
        const fv::ddtScheme<scalar>& ddtAlpha = tddtAlpha();

        if
        (
            isType<fv::EulerDdtScheme<scalar>>(ddtAlpha)
         || isType<fv::localEulerDdtScheme<scalar>>(ddtAlpha)
        )
        {
            ocCoeff = 0;
        }
        else if (isType<fv::CrankNicolsonDdtScheme<scalar>>(ddtAlpha))
        {
            if (nAlphaSubCycles > 1)
            {
                FatalErrorInFunction
                    << "Sub-cycling is not supported "
                       "with the CrankNicolson ddt scheme"
                    << exit(FatalError);
            }

            if
            (
                alphaRestart
             || mesh.time().timeIndex() > mesh.time().startTimeIndex() + 1
            )
            {
                ocCoeff =
                    refCast<const fv::CrankNicolsonDdtScheme<scalar>>(ddtAlpha)
                   .ocCoeff();
            }
        }
        else
        {
            FatalErrorInFunction
                << "Only Euler and CrankNicolson ddt schemes are supported"
                << exit(FatalError);
        }
    }

    // Set the time blending factor, 1 for Euler
    scalar cnCoeff = 1.0/(1.0 + ocCoeff);

    // Standard face-flux compression coefficient
    surfaceScalarField phic(mixture.cAlpha()*mag(phi/mesh.magSf()));

    // Add the optional isotropic compression contribution
    if (icAlpha > 0)
    {
        phic *= (1.0 - icAlpha);
        phic += (mixture.cAlpha()*icAlpha)*fvc::interpolate(mag(U));
    }

    // Add the optional shear compression contribution
    if (scAlpha > 0)
    {
        phic +=
            scAlpha*mag(mesh.delta() & fvc::interpolate(symm(fvc::grad(U))));
    }


    surfaceScalarField::Boundary& phicBf =
        phic.boundaryFieldRef();

    // Do not compress interface at non-coupled boundary faces
    // (inlets, outlets etc.)
    forAll(phic.boundaryField(), patchi)
    {
        fvsPatchScalarField& phicp = phicBf[patchi];

        if (!phicp.coupled())
        {
            phicp == 0;
        }
    }

    tmp<surfaceScalarField> phiCN(phi);

    // Calculate the Crank-Nicolson off-centred volumetric flux
    if (ocCoeff > 0)
    {
        phiCN = cnCoeff*phi + (1.0 - cnCoeff)*phi.oldTime();
    }

    if (MULESCorr)
    {
        #include "alphaSuSp.H"

        fvScalarMatrix alpha1Eqn
        (
            (
                LTS
              ? fv::localEulerDdtScheme<scalar>(mesh).fvmDdt(alpha1)
              : fv::EulerDdtScheme<scalar>(mesh).fvmDdt(alpha1)
            )
          + fv::gaussConvectionScheme<scalar>
            (
                mesh,
                phiCN,
                upwind<scalar>(mesh, phiCN)
            ).fvmDiv(phiCN, alpha1)
       // - fvm::Sp(fvc::ddt(dimensionedScalar("1", dimless, 1), mesh)
       //           + fvc::div(phiCN), alpha1)
         ==
            Su + fvm::Sp(Sp + divU, alpha1)
        );
        if (getenv("BRAE_STAGE_ALPHA"))
        {
            volScalarField rg
            (
                IOobject("sAlphaRefGradInitial.dump", runTime.timeName(), alpha1.mesh(), IOobject::NO_READ, IOobject::NO_WRITE),
                alpha1.mesh(), dimensionedScalar(dimless, Zero)
            );
            forAll(alpha1.boundaryField(), pi)
            {
                const auto* ca = dynamic_cast<const constantAlphaContactAngleFvPatchScalarField*>(&alpha1.boundaryField()[pi]);
                if (ca) rg.boundaryFieldRef()[pi] == ca->gradient();
            }
            rg.write();
        }

        alpha1Eqn.solve();
        if (getenv("BRAE_STAGE_ALPHA"))
        {
            forAll(alpha1.boundaryField(), pi)
            {
                const auto* ca = dynamic_cast<const constantAlphaContactAngleFvPatchScalarField*>(&alpha1.boundaryField()[pi]);
                if (ca && ca->gradient().size()) Info<< "oracle pre-solve gradient[0] " << ca->gradient()[0] << endl;
            }
        }
        if (getenv("BRAE_STAGE_ALPHA")) braeDumpMatrix("sAlphaEqn", alpha1Eqn);

        Info<< "Phase-1 volume fraction = "
            << alpha1.weightedAverage(mesh.Vsc()).value()
            << "  Min(" << alpha1.name() << ") = " << min(alpha1).value()
            << "  Max(" << alpha1.name() << ") = " << max(alpha1).value()
            << endl;

        tmp<surfaceScalarField> talphaPhi1UD(alpha1Eqn.flux());
        alphaPhi10 = talphaPhi1UD();

        if (alphaApplyPrevCorr && talphaPhi1Corr0.valid())
        {
            Info<< "Applying the previous iteration compression flux" << endl;
            MULES::correct
            (
                geometricOneField(),
                alpha1,
                alphaPhi10,
                talphaPhi1Corr0.ref(),
                oneField(),
                zeroField()
            );

            alphaPhi10 += talphaPhi1Corr0();
        }

        // Cache the upwind-flux
        talphaPhi1Corr0 = talphaPhi1UD;

        // ---- ORACLE INSTRUMENTATION: the semi-implicit upwind predictor's alpha and flux ----
        if (braeOracleStep())
        {
            surfaceScalarField("oAlphaPhiUD.dump", talphaPhi1Corr0()).write();
            volScalarField("oAlphaPredictor.dump", alpha1).write();
            if (getenv("BRAE_STAGE_ALPHA")) volScalarField("sAlphaPredictor.dump", alpha1).write();
        }
        // ------------------------------------------------------------------------------------

        alpha2 = 1.0 - alpha1;

        mixture.correct();
    }


    for (int aCorr=0; aCorr<nAlphaCorr; aCorr++)
    {
        #include "alphaSuSp.H"

        surfaceScalarField phir(phic*mixture.nHatf());

        alphaPhiUn =
            fvc::flux
            (
                phi,
                alpha1,
                alphaScheme
            )
          + fvc::flux
            (
               -fvc::flux(-phir, alpha2, alpharScheme),
                alpha1,
                alpharScheme
            );

        if (MULESCorr)
        {
            tmp<surfaceScalarField> talphaPhi1Corr(alphaPhiUn - alphaPhi10);
            volScalarField alpha10("alpha10", alpha1);

            // ---- ORACLE INSTRUMENTATION: the unlimited MULESCorr inputs, one set per alpha corrector ----
            if (braeOracleStep())
            {
                const Foam::word sfxM = "a" + Foam::name(aCorr);
                surfaceScalarField("oPhir" + sfxM + ".dump", phir).write();
                surfaceScalarField("oPhic" + sfxM + ".dump", phic).write();
                surfaceScalarField("oAlphaPhiUn" + sfxM + ".dump", alphaPhiUn).write();
                surfaceScalarField("oAlphaPhi10Pre" + sfxM + ".dump", alphaPhi10).write();
                surfaceScalarField("oAlphaPhiCorrUnlim" + sfxM + ".dump", talphaPhi1Corr()).write();
                volScalarField("oAlphaPreCorr" + sfxM + ".dump", alpha1).write();
            }
            // ---------------------------------------------------------------------------------------------

            MULES::correct
            (
                geometricOneField(),
                alpha1,
                alphaPhiUn,
                talphaPhi1Corr.ref(),
                Sp,
                (-Sp*alpha1)(),
                oneField(),
                zeroField()
            );

            // ---- ORACLE INSTRUMENTATION (tools/mgpu_oracle): writes only, gated on BRAE_ORACLE_ITERS ----
            // MULESCorr's limited correction flux; lambda_eff = phiCorrLim/phiCorrUnlim face by face
            // (MULES::correct scales the correction by its limiter lambda, in place).
            if (braeOracleStep())
            {
                const Foam::word sfxM = "a" + Foam::name(aCorr);
                surfaceScalarField("oAlphaPhiCorrLim" + sfxM + ".dump", talphaPhi1Corr()).write();
                volScalarField("oAlphaCorr" + sfxM + ".dump", alpha1).write();
                if (getenv("BRAE_STAGE_ALPHA")) volScalarField("sAlphaCorr" + sfxM + ".dump", alpha1).write();
            }
            // ---------------------------------------------------------------------------------------------

            // Under-relax the correction for all but the 1st corrector
            if (aCorr == 0)
            {
                alphaPhi10 += talphaPhi1Corr();
            }
            else
            {
                alpha1 = 0.5*alpha1 + 0.5*alpha10;
                alphaPhi10 += 0.5*talphaPhi1Corr();
            }
        }
        else
        {
            // ---- INSTRUMENTATION: writes only -------------------------------------------------
            if (getenv("BRAE_DUMP_ITER") && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
            {
                surfaceScalarField("phiCN.dump", phiCN()).write();
                surfaceScalarField("phic.dump", phic).write();
                surfaceScalarField("phir.dump", phir).write();
                surfaceScalarField("alphaPhiUn.dump", alphaPhiUn).write();
                volScalarField("alphaPreMULES.dump", alpha1).write();
            }
            // -----------------------------------------------------------------------------------

            alphaPhi10 = alphaPhiUn;

            MULES::explicitSolve
            (
                geometricOneField(),
                alpha1,
                phiCN,
                alphaPhi10,
                Sp,
                (Su + divU*min(alpha1(), scalar(1)))(),
                oneField(),
                zeroField()
            );
        }

        if (getenv("BRAE_DUMP_ITER") && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
        {
            volScalarField("alphaPostMULES.dump", alpha1).write();
            surfaceScalarField("alphaPhi10Post.dump", alphaPhi10).write();
        }

        alpha2 = 1.0 - alpha1;
        if (getenv("BRAE_STAGE_ALPHA")) volScalarField("sAlphaBeforeMixture" + Foam::name(aCorr) + ".dump", alpha1).write();

        mixture.correct();
    }

    if (alphaApplyPrevCorr && MULESCorr)
    {
        talphaPhi1Corr0 = alphaPhi10 - talphaPhi1Corr0;
        talphaPhi1Corr0.ref().rename("alphaPhi1Corr0");
    }
    else
    {
        talphaPhi1Corr0.clear();
    }

    #include "rhofs.H"

    if
    (
        word(mesh.ddtScheme("ddt(rho,U)"))
     == fv::EulerDdtScheme<vector>::typeName
     || word(mesh.ddtScheme("ddt(rho,U)"))
     == fv::localEulerDdtScheme<vector>::typeName
    )
    {
        rhoPhi = alphaPhi10*(rho1f - rho2f) + phiCN*rho2f;
    }
    else
    {
        if (ocCoeff > 0)
        {
            // Calculate the end-of-time-step alpha flux
            alphaPhi10 =
                (alphaPhi10 - (1.0 - cnCoeff)*alphaPhi10.oldTime())/cnCoeff;
        }

        // Calculate the end-of-time-step mass flux
        rhoPhi = alphaPhi10*(rho1f - rho2f) + phi*rho2f;
    }

    Info<< "Phase-1 volume fraction = "
        << alpha1.weightedAverage(mesh.Vsc()).value()
        << "  Min(" << alpha1.name() << ") = " << min(alpha1).value()
        << "  Max(" << alpha1.name() << ") = " << max(alpha1).value()
        << endl;
}
