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
#include "clockTime.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "simpleControl.H"
#include "pressureControl.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

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

    // brae tools/timeRhoSimpleFoam: OpenFOAM's rhoSimpleFoam with WALL-CLOCK timers around each block
    // (and around each block's linear solve, set inside the copied *Eqn.H), reduced to the MAX over
    // ranks. Nothing else is changed. One PHASE line per iteration, in seconds.
    clockTime phaseClock;
    scalar tUsol = 0, tEsol = 0, tPsol = 0;
    auto phase = [&](scalar& acc) { acc += phaseClock.timeIncrement(); };
    auto maxRank = [](scalar v) { reduce(v, maxOp<scalar>()); return v; };

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        scalar tU = 0, tE = 0, tP = 0, tTurb = 0, tRest = 0;
        tUsol = tEsol = tPsol = 0;
        phaseClock.timeIncrement();

        // Pressure-velocity SIMPLE corrector
        #include "UEqn.H"
        phase(tU);
        #include "EEqn.H"
        phase(tE);

        if (simple.consistent())
        {
            #include "pcEqn.H"
        }
        else
        {
            #include "pEqn.H"
        }
        phase(tP);

        turbulence->correct();
        phase(tTurb);

        runTime.write();

        runTime.printExecutionTime(Info);
        phase(tRest);

        Info<< "PHASE UEqn " << maxRank(tU) << " (solve " << maxRank(tUsol) << ")"
            << " EEqn " << maxRank(tE) << " (solve " << maxRank(tEsol) << ")"
            << " pEqn " << maxRank(tP) << " (solve " << maxRank(tPsol) << ")"
            << " turbulence " << maxRank(tTurb)
            << " rest " << maxRank(tRest)
            << " total " << maxRank(tU + tE + tP + tTurb + tRest) << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
