/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
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
    rhoSimpleFoam

Group
    grpCompressibleSolvers

Description
    Steady-state solver for compressible turbulent flow.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "wallDist.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "simpleControl.H"
#include "pressureControl.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Stage harness: which SIMPLE iteration to dump (BRAE_DUMP_STAGE_ITER, default 1).
static int braeDumpIter()
{
    const char* e = ::getenv("BRAE_DUMP_STAGE_ITER");
    const int v = (e && *e) ? ::atoi(e) : 1;
    return v > 0 ? v : 1;
}

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state solver for compressible turbulent flow."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createFieldRefs.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // Pressure-velocity SIMPLE corrector
        #include "UEqn.H"
        #include "EEqn.H"

        if (simple.consistent())
        {
            #include "pcEqn.H"
        }
        else
        {
            #include "pEqn.H"
        }

        // Stage harness: turbulence->correct() as a BLACK BOX -- every input it reads, then every field
        // it writes. That is enough to gate another code's closure against it without reaching inside
        // OpenFOAM's turbulence library, which the solver cannot do: the k and epsilon matrices are
        // assembled inside the model and never surface here.
        if (runTime.timeIndex() == braeDumpIter())
        {
            volScalarField("stage_kIn",      turbulence->k()()).write();
            volScalarField("stage_epsIn",    turbulence->epsilon()()).write();
            volScalarField("stage_nutIn",    turbulence->nut()()).write();
            volScalarField("stage_alphatIn", turbulence->alphat()()).write();
            volVectorField("stage_Uturb",    U).write();
            surfaceScalarField("stage_phiTurb", phi).write();
            volScalarField("stage_rhoTurb",  rho).write();
            volScalarField("stage_muTurb",   thermo.mu()()).write();
        }

        turbulence->correct();

        if (runTime.timeIndex() == braeDumpIter())
        {
            volScalarField("stage_kOut",      turbulence->k()()).write();
            volScalarField("stage_epsOut",    turbulence->epsilon()()).write();
            volScalarField("stage_nutOut",    turbulence->nut()()).write();
            volScalarField("stage_alphatOut", turbulence->alphat()()).write();
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
