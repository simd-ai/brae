/*---------------------------------------------------------------------------*\
    dumpLimitedGrad

    Prints OpenFOAM's own gradient of a scalar field through a NAMED gradSchemes
    entry -- exactly the call linearUpwind makes for its correction
    (linearUpwind.C:61-72):

        fv::gradScheme<scalar>::New(mesh, mesh.gradScheme(name))().grad(vf, name)

    so a brae gradient (Gauss linear, cellLimited, ...) can be held against it
    cell by cell at 17 digits, on OpenFOAM's own field and patch values.

    The field is read as written, patch values included, and NOT re-evaluated:
    the comparison is of the gradient operator, not of a boundary evaluation.

    Usage:  dumpLimitedGrad -case <dir> -field <name> [-scheme <gradSchemes key>]
            [-time <t>]
    Writes: GRAD <cell> <gx> <gy> <gz>   one row per cell, then END.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "IOmanip.H"

int main(int argc, char *argv[])
{
    argList::addOption("field", "name", "scalar field to take the gradient of");
    argList::addOption("scheme", "name", "gradSchemes entry (default grad(<field>))");
    timeSelector::addOptions();

    #include "setRootCase.H"
    #include "createTime.H"

    instantList timeDirs = timeSelector::select0(runTime, args);
    if (timeDirs.empty())
    {
        FatalErrorInFunction << "no time directory selected" << exit(FatalError);
    }
    runTime.setTime(timeDirs.last(), timeDirs.size() - 1);

    #include "createMesh.H"

    const word fieldName = args.get<word>("field");
    const word schemeName =
        args.getOrDefault<word>("scheme", "grad(" + fieldName + ")");

    volScalarField vf
    (
        IOobject
        (
            fieldName,
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    tmp<fv::gradScheme<scalar>> gs
    (
        fv::gradScheme<scalar>::New(mesh, mesh.gradScheme(schemeName))
    );
    const volVectorField g(gs().grad(vf, schemeName));

    Info<< "SCHEME " << schemeName << nl;
    forAll(g, celli)
    {
        Info<< "GRAD " << celli
            << ' ' << setprecision(17) << g[celli].x()
            << ' ' << setprecision(17) << g[celli].y()
            << ' ' << setprecision(17) << g[celli].z() << nl;
    }
    Info<< "END" << endl;

    return 0;
}

// ************************************************************************* //
