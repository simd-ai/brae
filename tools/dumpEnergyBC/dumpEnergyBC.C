/*---------------------------------------------------------------------------*\
    dumpEnergyBC

    Prints the ENERGY boundary conditions' coefficients as OpenFOAM's own
    fixedEnergy / gradientEnergy / mixedEnergy build them, so brae's
    updateEnergyBoundaryCoeffs can be compared against them coefficient by
    coefficient rather than through a converged field.

    THE ORACLE HAS TO BE THE COEFFICIENTS, not a solution. For perfectGas +
    hConst the live update is arithmetically identical to the static
    construction-time mapping it replaces -- he is p-independent and Cpv is a
    constant -- so an end-to-end comparison on a gas case cannot tell a correct
    transcription from a wrong one. These four numbers per face can:
    Tw.refValue() and Tw.value() differ on a mixed patch, Cpv and the affine
    he(T) differ on a gradient, and either mistake shows up here immediately.

    What it does is exactly what the energy matrix assembly does and nothing
    else: construct the case's own fluidThermo, then call updateCoeffs() on
    he's boundary field.  Each condition evaluates T's patch itself
    (fixedEnergy .C:108, gradientEnergy .C:109, mixedEnergy .C:97) and rebuilds
    its coefficients from the p and T standing in the time directory read.

    Run it in a time directory holding a DEVELOPED state, not a uniform one:
    on a uniform field every patch's coefficients collapse onto the same number
    and the comparison stops discriminating.

    Usage:  dumpEnergyBC -case <dir> [-time <t>]
    Writes: the rows below on stdout, 17 significant digits.

        PATCHES <n>
        PATCH <i> <name> <heType> <nFaces>
        MIXED <i> <face> <refValue> <refGrad> <valueFraction> <value>
        GRAD  <i> <face> <gradient> <value>
        VALUE <i> <face> <value>
        OTHER <i> <face> <value>
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fluidThermo.H"
#include "IOmanip.H"
#include "fixedValueFvPatchFields.H"
#include "fixedGradientFvPatchFields.H"
#include "mixedFvPatchFields.H"

int main(int argc, char *argv[])
{
    // timeSelector rather than a bare `-time` option: `-time`, `-latestTime` and `-newTimes` all come
    // from it, and the thermo has to be constructed AFTER the time is set or it reads 0/ whatever the
    // caller asked for.
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

    Info<< "TIME " << runTime.timeName() << endl;

    // phi has to be on the registry before updateCoeffs runs: inletOutlet -- which basicThermo maps to
    // mixedEnergy -- looks it up by name to build its flux switch (inletOutletFvPatchField.C). Without it
    // the utility dies with "failed lookup of phi" on any case carrying an outlet, which is most of them.
    // MUST_READ rather than a zero default: a zero flux would silently make every face an outflow face
    // and the valueFraction this dumps would not be the one OpenFOAM computes.
    surfaceScalarField phi
    (
        IOobject
        (
            "phi",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    autoPtr<fluidThermo> pThermo(fluidThermo::New(mesh));
    fluidThermo& thermo = pThermo();
    thermo.validate(args.executable(), "h", "e");

    volScalarField& he = thermo.he();

    // The one call the energy matrix makes before it reads a single coefficient
    // (fvMatrix.C:396 -- updateCoeffs is not an evaluate; it is what BUILDS the
    // coefficients the matrix then asks for).
    he.boundaryFieldRef().updateCoeffs();

    Info<< "HENAME " << he.name() << nl
        << "PATCHES " << mesh.boundary().size() << endl;

    forAll(mesh.boundary(), patchi)
    {
        const fvPatchScalarField& hp = he.boundaryField()[patchi];

        Info<< "PATCH " << patchi << ' ' << mesh.boundary()[patchi].name()
            << ' ' << hp.type() << ' ' << hp.size() << endl;

        // The order mirrors basicThermo::heBoundaryTypes: mixed is tested before
        // fixedGradient here only because mixedEnergy derives from mixed and not
        // from fixedGradient, so the two sets are disjoint either way.
        if (isA<mixedFvPatchScalarField>(hp))
        {
            const mixedFvPatchScalarField& m =
                refCast<const mixedFvPatchScalarField>(hp);

            forAll(m, facei)
            {
                Info<< "MIXED " << patchi << ' ' << facei
                    << ' ' << setprecision(17) << m.refValue()[facei]
                    << ' ' << setprecision(17) << m.refGrad()[facei]
                    << ' ' << setprecision(17) << m.valueFraction()[facei]
                    << ' ' << setprecision(17) << m[facei] << endl;
            }
        }
        else if (isA<fixedGradientFvPatchScalarField>(hp))
        {
            const fixedGradientFvPatchScalarField& gp =
                refCast<const fixedGradientFvPatchScalarField>(hp);

            forAll(gp, facei)
            {
                Info<< "GRAD " << patchi << ' ' << facei
                    << ' ' << setprecision(17) << gp.gradient()[facei]
                    << ' ' << setprecision(17) << gp[facei] << endl;
            }
        }
        else if (isA<fixedValueFvPatchScalarField>(hp))
        {
            forAll(hp, facei)
            {
                Info<< "VALUE " << patchi << ' ' << facei
                    << ' ' << setprecision(17) << hp[facei] << endl;
            }
        }
        else
        {
            forAll(hp, facei)
            {
                Info<< "OTHER " << patchi << ' ' << facei
                    << ' ' << setprecision(17) << hp[facei] << endl;
            }
        }
    }

    Info<< "END" << endl;

    return 0;
}

// ************************************************************************* //
