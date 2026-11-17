/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2020 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    interFoam

Group
    grpMultiphaseSolvers

Description
    Solver for two incompressible, isothermal immiscible fluids using a VOF
    (volume of fluid) phase-fraction based interface capturing approach,
    with optional mesh motion and mesh topology changes including adaptive
    re-meshing.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "CMULES.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "CrankNicolsonDdtScheme.H"
#include "subCycle.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "incompressibleInterPhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "CorrectPhi.H"
#include "fvcSmooth.H"
#include "processorFvPatch.H"
#include "twoPhaseProperties/alphaContactAngle/constantAlphaContactAngle/constantAlphaContactAngleFvPatchScalarField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Solver for two incompressible, isothermal immiscible fluids"
        " using VOF phase-fraction based interface capturing.\n"
        "With optional mesh motion and mesh topology changes including"
        " adaptive re-meshing."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"

    // BRAE_PROC_GEOMETRY=1: write the processor patches' geometry (procGeometryDump.H) and stop, before any field
    // is read -- tools/mgpu_oracle/proc_geometry_gate.sh
    if (getenv("BRAE_PROC_GEOMETRY"))
    {
        #include "procGeometryDump.H"
        return 0;
    }
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createAlphaFluxes.H"
    // BRAE_STAGE_DUMP=1: the explicit stages of the first step on THIS state (before initCorrectPhi, which would
    // re-solve the flux), written, then stop (stageDump.H)
    if (getenv("BRAE_STAGE_DUMP"))
    {
        #include "stageDump.H"
        return 0;
    }

    #include "initCorrectPhi.H"
    if (getenv("BRAE_STAGE_STATE"))
    {
        runTime.writeNow();
        return 0;
    }
    #include "createUfIfPresent.H"

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "alphaCourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            // U's STORED PATCH VALUES AT THREE POINTS OF THE STEP. brae holds the value its last
            // evaluate left and OpenFOAM does not agree with it at the momentum assembly -- 346 of the
            // 1050 permeable faces, OpenFOAM exactly zero where brae carries the extrapolation -- while
            // refValue and valueFraction agree on every face. So something between the previous step's
            // pEqn and this assembly evaluates the patch in OpenFOAM and not in brae, and only a dump
            // on each side of the mesh update and the alpha step can say which.
            if (getenv("BRAE_DUMP_ITER")
             && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
            {
                volVectorField(IOobject("UloopEntry.dump", runTime.timeName(), mesh,
                                        IOobject::NO_READ, IOobject::NO_WRITE), U).write();
            }

            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                mesh.update();
                if (getenv("BRAE_DUMP_ITER")
                 && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
                {
                    volVectorField(IOobject("UafterMove.dump", runTime.timeName(), mesh,
                                            IOobject::NO_READ, IOobject::NO_WRITE), U).write();
                }

                if (mesh.changing())
                {
                    // Do not apply previous time-step mesh compression flux
                    // if the mesh topology changed
                    if (mesh.topoChanging())
                    {
                        talphaPhi1Corr0.clear();
                    }

                    gh = (g & mesh.C()) - ghRef;
                    ghf = (g & mesh.Cf()) - ghRef;

                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & Uf();

                        // ---- INSTRUMENTATION: writes only ------------------------------------
                        // The mesh-update block's three flux stages. brae is exact on this case
                        // with `correctPhi no` and 1.2e-03 out at step two with it on, so the gap
                        // is between these three writes. Gated on BRAE_DUMP_ITER.
                        const bool braeDump =
                            getenv("BRAE_DUMP_ITER")
                         && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER"));
                        if (braeDump)
                        {
                            surfaceScalarField("phiAbsPre.dump", phi).write();
                            surfaceScalarField("rAUfCorr.dump",
                                               fvc::interpolate(rAU())()).write();
                            // ...only where there IS one. fvMesh::phi() is a FATAL ERROR on a mesh that
                            // changes topology without moving its points (fvMeshGeometry.C:421, "mesh flux
                            // field does not exist"), and this block runs for a REFINING mesh too.
                            if (mesh.moving())
                            {
                                surfaceScalarField("meshPhiU.dump", fvc::meshPhi(U)()).write();
                            }
                            surfaceVectorField("UfIn.dump", Uf()).write();
                        }
                        // ----------------------------------------------------------------------

                        #include "correctPhi.H"

                        if (braeDump)
                        {
                            surfaceScalarField("phiAbsPost.dump", phi).write();
                        }

                        // Make the flux relative to the mesh motion
                        fvc::makeRelative(phi, U);

                        if (braeDump)
                        {
                            surfaceScalarField("phiRel.dump", phi).write();
                        }

                        mixture.correct();

                        // ...and what mixture.correct() LEAVES for the alpha step: the interface
                        // normal it compresses along and the curvature the surface tension force is
                        // built from. calculateK is a fixed point -- it reads the wall gradient the
                        // previous call wrote -- so an extra call here is not a no-op.
                        if (braeDump)
                        {
                            // ...and alpha AS calculateK SAW IT, boundary values included: the only
                            // other input it has once the mesh is fixed
                            volScalarField("alphaAtK.dump", alpha1).write();
                            surfaceScalarField("nHatf.dump", mixture.nHatf()).write();
                            volScalarField("sigmaK.dump", mixture.sigmaK()()).write();
                            // ...and THE MOVED MESH'S OWN GEOMETRY, which is what calculateK is a
                            // function of once alpha is fixed. A deforming mesh makes faces
                            // non-planar, where the face-centre/area decomposition is a choice.
                            surfaceScalarField("weights.dump", mesh.weights()).write();
                            surfaceScalarField("deltaCoeffs.dump", mesh.deltaCoeffs()).write();
                            {
                                surfaceVectorField Sfd
                                (
                                    IOobject("Sf.dump", runTime.timeName(), mesh),
                                    mesh,
                                    dimensionedVector(dimArea, Zero)
                                );
                                Sfd.primitiveFieldRef() = mesh.Sf().primitiveField();
                                forAll(Sfd.boundaryField(), pi)
                                {
                                    Sfd.boundaryFieldRef()[pi] == mesh.Sf().boundaryField()[pi];
                                }
                                Sfd.write();

                                volVectorField Cd
                                (
                                    IOobject("C.dump", runTime.timeName(), mesh),
                                    mesh,
                                    dimensionedVector(dimLength, Zero)
                                );
                                Cd.primitiveFieldRef() = mesh.C().primitiveField();
                                Cd.write();
                            }
                            volScalarField
                            (
                                IOobject("V.dump", runTime.timeName(), mesh),
                                mesh,
                                dimensionedScalar(dimVolume, Zero)
                            ).write();
                            {
                                volScalarField Vf
                                (
                                    IOobject("V.dump", runTime.timeName(), mesh),
                                    mesh,
                                    dimensionedScalar(dimVolume, Zero)
                                );
                                Vf.primitiveFieldRef() = mesh.V();
                                Vf.write();
                            }
                            Info<< "[brae] dumped the mesh-update stages at timeIndex "
                                << runTime.timeIndex() << endl;
                        }
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            #include "alphaControls.H"
            #include "alphaEqnSubCycle.H"

            // ...and what the alpha step LEAVES for the momentum: the interface normal flux and the
            // curvature. Written HERE and not in the mesh-update block, because a case with
            // `correctPhi no` -- RAS/electrostaticDeposition -- never reaches that block.
            if (getenv("BRAE_DUMP_ITER")
             && runTime.timeIndex() == atoi(getenv("BRAE_DUMP_ITER")))
            {
                surfaceScalarField("nHatfA.dump", mixture.nHatf()).write();
                volVectorField(IOobject("UafterAlpha.dump", runTime.timeName(), mesh,
                                        IOobject::NO_READ, IOobject::NO_WRITE), U).write();
                volScalarField("sigmaKA.dump", mixture.sigmaK()()).write();
                // THE OLD CELL VOLUMES, which the moving-mesh ddt's source is built on
                // (rho*V0/dt*U.oldTime()). Nothing written at the end of a step can show them, which
                // is why six comparisons of step-one state missed the permeable-moving gap.
                // ONLY ON A MESH THAT MOVES: fvMesh::V0() has no field to return otherwise and
                // aborts with "V0 is not available" (fvMeshGeometry.C:208-215), which is how this
                // block took interfoam_capillaryrise_vs_openfoam down -- a static case whose gate
                // runs this tool, and which the scoped run that unit chose did not cover.
                if (mesh.moving())
                {
                    volScalarField V0dump
                    (
                        IOobject
                        (
                            "V0.dump",
                            runTime.timeName(),
                            mesh,
                            IOobject::NO_READ,
                            IOobject::NO_WRITE
                        ),
                        mesh,
                        dimensionedScalar(dimVolume, Zero)
                    );
                    V0dump.primitiveFieldRef() = mesh.V0();
                    V0dump.write();
                    volScalarField Vdump
                    (
                        IOobject("V.dump", runTime.timeName(), mesh,
                                 IOobject::NO_READ, IOobject::NO_WRITE),
                        mesh,
                        dimensionedScalar(dimVolume, Zero)
                    );
                    Vdump.primitiveFieldRef() = mesh.V();
                    Vdump.write();
                }
                Info<< "[brae] dumped nHatf and sigmaK after the alpha step at timeIndex "
                    << runTime.timeIndex() << endl;
            }

            if (getenv("BRAE_STAGE_ALPHA")) volScalarField("sAlphaBeforeFinalMixture.dump", alpha1).write();
            if (getenv("BRAE_STAGE_ALPHA")) volScalarField("sAlpha2BeforeFinalMixture.dump", alpha2).write();
            mixture.correct();

            if (getenv("BRAE_STAGE_ALPHA")) volScalarField("sAlpha2AfterFinalMixture.dump", alpha2).write();
            // ---- ORACLE INSTRUMENTATION (tools/mgpu_oracle): what the alpha step hands to momentum/pressure
            if (braeOracleStep())
            {
                surfaceScalarField("oNHatf.dump", mixture.nHatf()).write();
                volScalarField("oSigmaK.dump", mixture.sigmaK()()).write();
                volScalarField("oAlphaEnd.dump", alpha1).write();
                surfaceScalarField("oAlphaPhi10End.dump", alphaPhi10).write();
                surfaceScalarField("oRhoPhi.dump", rhoPhi).write();
                volScalarField("oRhoEnd.dump", rho).write();
                surfaceScalarField("oPhiEntry.dump", phi).write();
            }
            if (getenv("BRAE_STAGE_ALPHA"))
            {
                volScalarField rg
                (
                    IOobject("sAlphaRefGrad.dump", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                    mesh, dimensionedScalar(dimless, Zero)
                );
                forAll(alpha1.boundaryField(), pi)
                {
                    const auto* ca = dynamic_cast<const constantAlphaContactAngleFvPatchScalarField*>(&alpha1.boundaryField()[pi]);
                    if (ca) rg.boundaryFieldRef()[pi] == ca->gradient();
                }
                rg.write();
            }
            if (getenv("BRAE_STAGE_ALPHA"))
            {
                volScalarField("sAlphaAfter.dump", alpha1).write();
                surfaceScalarField("sAlphaPhi.dump", alphaPhi10).write();
                volScalarField("sRhoAfter.dump", rho).write();
                Info<< "[brae] alpha stage dump written at time " << runTime.timeName() << endl;
                return 0;
            }

            if (pimple.frozenFlow())
            {
                continue;
            }

            #include "UEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

        runTime.write();

        // ---- ORACLE INSTRUMENTATION (tools/mgpu_oracle): write the full state at chosen step indices
        // (BRAE_WRITE_ITERS="1,20") and stop after step BRAE_STOP_ITER, so serial and -parallel runs
        // end on the same step whatever adjustTimeStep does with deltaT.
        {
            const char* stopEnv = getenv("BRAE_STOP_ITER");
            const bool stopNow = stopEnv && runTime.timeIndex() >= atoi(stopEnv);
            if (stopNow)
            {
                runTime.writeAndEnd();
            }
            else if (braeInList("BRAE_WRITE_ITERS") && !runTime.writeTime())
            {
                runTime.writeNow();
            }
        }

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
