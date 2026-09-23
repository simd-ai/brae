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
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createAlphaFluxes.H"
    #include "initCorrectPhi.H"
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
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                mesh.update();

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
                            surfaceScalarField("meshPhiU.dump", fvc::meshPhi(U)()).write();
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
                volScalarField("sigmaKA.dump", mixture.sigmaK()()).write();
                Info<< "[brae] dumped nHatf and sigmaK after the alpha step at timeIndex "
                    << runTime.timeIndex() << endl;
            }

            mixture.correct();

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

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
