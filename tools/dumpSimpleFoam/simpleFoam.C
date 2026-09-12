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
    dumpSimpleFoam

Group
    grpIncompressibleSolvers

Description
    Steady-state solver for incompressible, turbulent flows.

    \heading Solver details
    The solver uses the SIMPLE algorithm to solve the continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \div \left( \vec{U} \vec{U} \right) - \div \gvec{R}
          = - \grad p + \vec{S}_U
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
        \vec{R} | Stress tensor
        \vec{S}_U | Momentum source
    \endvartable

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
        \<turbulence fields\> | As required by user selection
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Stage harness: which SIMPLE iteration to dump (BRAE_DUMP_STAGE_ITER, default 1). The twin of
// tools/dumpPEqn's, so a compressible and an incompressible dump are driven by the same variable.
// The transported turbulence fields a dump names one by one. turbulence->k() and ->nut() are on the
// model's own interface; omega, epsilon, nuTilda, gammaInt and ReThetat are not, so they are looked up
// in the registry by name and skipped when the case's model does not carry them.
static const Foam::wordList TURBFIELDS
({
    "omega",
    "epsilon",
    "nuTilda",
    "gammaInt",
    "ReThetat"
});

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
        "Steady-state solver for incompressible, turbulent flows, carrying a stage harness:"
        " at SIMPLE iteration BRAE_DUMP_STAGE_ITER it writes the momentum system and every"
        " quantity the SIMPLE/SIMPLEC pressure step is built from."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // Do any mesh changes
        mesh.controlledUpdate();

        if (mesh.changing())
        {
            MRF.update();
        }

        // --- Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "pEqn.H"
        }

        // The closure's INPUTS, exactly as turbulence->correct() reads them: phi is the CONSERVATIVE
        // flux the pressure step just left, not the one the momentum equation was convected by, and nut
        // is whatever the previous correct() (or validate()) put there. Held separately from the outputs
        // below so a closure comparison cannot be blamed on an input the two codes disagree about.
        if (runTime.timeIndex() == braeDumpIter())
        {
            volScalarField("stage_nutIn", turbulence->nut()()).write();
            volScalarField("stage_kIn",   turbulence->k()()).write();
            surfaceScalarField("stage_phiIn", phi).write();
            forAll(TURBFIELDS, i)
            {
                const volScalarField* f = mesh.findObject<volScalarField>(TURBFIELDS[i]);
                if (f) volScalarField("stage_" + TURBFIELDS[i] + "In", *f).write();
            }
        }

        laminarTransport.correct();
        turbulence->correct();

        // The closure's OUTPUTS. Every transported turbulence field the case registers, by name, so a
        // model with more than two of them (kOmegaSSTLM carries gammaInt and ReThetat as well) is dumped
        // whole rather than through the two the solver happens to hold a handle on. runTime.write()
        // below writes k, omega and nut anyway, but not at a name a comparison can key on when the case
        // writes at a different interval -- and never gammaInt/ReThetat on the brae side.
        if (runTime.timeIndex() == braeDumpIter())
        {
            volScalarField("stage_nutPost", turbulence->nut()()).write();
            volScalarField("stage_kPost",   turbulence->k()()).write();
            forAll(TURBFIELDS, i)
            {
                const volScalarField* f = mesh.findObject<volScalarField>(TURBFIELDS[i]);
                if (f) volScalarField("stage_" + TURBFIELDS[i] + "Post", *f).write();
            }
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
