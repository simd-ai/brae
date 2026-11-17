#include "createRDeltaT.H"

Info<< "Reading field p_rgh\n" << endl;
volScalarField p_rgh
(
    IOobject
    (
        "p_rgh",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhi.H"


Info<< "Reading transportProperties\n" << endl;
immiscibleIncompressibleTwoPhaseMixture mixture(U, phi);

volScalarField& alpha1(mixture.alpha1());
volScalarField& alpha2(mixture.alpha2());

const dimensionedScalar& rho1 = mixture.rho1();
const dimensionedScalar& rho2 = mixture.rho2();


// Need to store rho for ddt(rho, U)
volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT
    ),
    alpha1*rho1 + alpha2*rho2
);
rho.oldTime();


// Mass flux
surfaceScalarField rhoPhi
(
    IOobject
    (
        "rhoPhi",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    fvc::interpolate(rho)*phi
);

typedef incompressibleInterPhaseTransportModel
    <
        immiscibleIncompressibleTwoPhaseMixture
    > transportModelType;

autoPtr<transportModelType> turbulence
(
    new transportModelType
    (
        rho, U, phi, rhoPhi, mixture
    )
);

#include "readGravitationalAcceleration.H"
#include "readhRef.H"
#include "gh.H"


volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    p_rgh + rho*gh
);

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell
(
    p,
    p_rgh,
    pimple.dict(),
    pRefCell,
    pRefValue
);

if (p_rgh.needReference())
{
    p += dimensionedScalar
    (
        "p",
        p.dimensions(),
        pRefValue - getRefCellValue(p, pRefCell)
    );
    p_rgh = p - rho*gh;
}

mesh.setFluxRequired(p_rgh.name());
mesh.setFluxRequired(alpha1.name());

#include "createMRF.H"
#include "createFvOptions.H"

// ---- ORACLE INSTRUMENTATION (tools/mgpu_oracle) ---------------------------------------------------
// braeInList("VAR"): is runTime.timeIndex() in the comma list in $VAR. The per-step operator dumps of
// tools/mgpu_oracle are gated on BRAE_ORACLE_ITERS (e.g. "1,20") and nothing else changes in the solve.
auto braeInList = [&runTime](const char* env) -> bool
{
    const char* s = getenv(env);
    if (!s) return false;
    const int ti = runTime.timeIndex();
    while (*s)
    {
        char* e;
        const long v = strtol(s, &e, 10);
        if (e != s && v == ti) return true;
        s = (*e == ',') ? e + 1 : (e != s ? e : s + 1);
    }
    return false;
};
auto braeOracleStep = [&]() -> bool { return braeInList("BRAE_ORACLE_ITERS"); };

// braeDumpMatrix: a scalar fvMatrix as <name>Diag/Source (vol), <name>Upper/Lower (surface, internal
// faces in lduAddressing order) and <name>IntCoeffs/BouCoeffs (surface, per-patch values: for a
// processor patch these are the interface coefficients).
auto braeDumpMatrix = [&](const word& name, const fvScalarMatrix& m)
{
    const fvMesh& msh = m.psi().mesh();
    auto volF = [&](const word& n, const scalarField& v)
    {
        volScalarField f
        (
            IOobject(name + n + ".dump", runTime.timeName(), msh, IOobject::NO_READ, IOobject::NO_WRITE),
            msh, dimensionedScalar(dimless, Zero)
        );
        f.primitiveFieldRef() = v;
        f.write();
    };
    auto surfF = [&](const word& n, const scalarField* inner, const FieldField<Field, scalar>* bnd)
    {
        surfaceScalarField f
        (
            IOobject(name + n + ".dump", runTime.timeName(), msh, IOobject::NO_READ, IOobject::NO_WRITE),
            msh, dimensionedScalar(dimless, Zero)
        );
        if (inner) f.primitiveFieldRef() = *inner;
        if (bnd)
        {
            forAll(f.boundaryField(), patchi)
            {
                f.boundaryFieldRef()[patchi] == (*bnd)[patchi];
            }
        }
        f.write();
    };
    volF("Diag", m.diag());
    volF("Source", m.source());
    if (m.hasUpper()) surfF("Upper", &m.upper(), nullptr);
    if (m.hasLower()) surfF("Lower", &m.lower(), nullptr);
    surfF("IntCoeffs", nullptr, &m.internalCoeffs());
    surfF("BouCoeffs", nullptr, &m.boundaryCoeffs());
};
