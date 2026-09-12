volScalarField rAU(1.0/UEqn.A());
surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
tUEqn.clear();

bool closedVolume = false;
bool dumpedPCoeffsS = false;   // stage harness, subsonic branch

surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho)*fvc::flux(HbyA));

// dumpPEqn stage harness -- same quantities as pcEqn.H, so a case can be compared with `consistent no`.
// rAtU does not exist on this path (no SIMPLEC), so rAU stands in for both.
if (runTime.timeIndex() == braeDumpIter())
{
    volScalarField("stage_rAU",  rAU).write();
    volScalarField("stage_rAtU", rAU).write();
    volVectorField("stage_HbyA", HbyA).write();
    surfaceScalarField("stage_phi", phi).write();
    surfaceScalarField("stage_rhorAUf", rhorAUf).write();
    volScalarField("stage_rhorAU", rho*rAU).write();
    volScalarField("stage_rhoP", rho).write();
    // psi AS pEqn.H SEES IT. EEqn.H ends with thermo.correct(), which recomputes T from the just-solved
    // he and then psi from that T -- so the psi the pressure equation uses is one energy solve newer than
    // the one createFields produced. The transonic branch divides by it.
    volScalarField("stage_psi", psi).write();
    volScalarField("stage_rAUp", rAU).write();
    surfaceScalarField("phiHbyA0", phiHbyA).write();
}
MRF.makeRelative(fvc::interpolate(rho), phiHbyA);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf, MRF);

// Stage harness: phiHbyA as pEqn.H builds it, BEFORE either branch modifies it. The transonic branch
// subtracts a psi*p term from it and the subsonic branch runs adjustPhi over it, so a single "phiHbyA"
// would mean two different things depending on the branch.
if (runTime.timeIndex() == braeDumpIter())
{
    surfaceScalarField("stage_phiHbyA0", phiHbyA).write();
}

if (simple.transonic())
{
    surfaceScalarField phid
    (
        "phid",
        (fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA
    );

    phiHbyA -= fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho);

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          + fvm::div(phid, p)
          - fvm::laplacian(rhorAUf, p)
          ==
            fvOptions(psi, p, rho.name())
        );

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();

        // Stage harness: the TRANSONIC branch's own quantities. This branch had no dump at all, so
        // nothing could be compared against it -- and it is the branch that carries fvm::div(phid, p),
        // the term that makes the pressure equation convective. Dumped AFTER relax(), because unlike the
        // subsonic branch this one relaxes before solving.
        if (runTime.timeIndex() == braeDumpIter())
        {
            surfaceScalarField("stage_phid", phid).write();
            surfaceScalarField("stage_phiHbyA", phiHbyA).write();
            volScalarField pICt
            (
                IOobject("stage_pICt", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), calculatedFvPatchScalarField::typeName
            );
            volScalarField pBCt
            (
                IOobject("stage_pBCt", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), calculatedFvPatchScalarField::typeName
            );
            forAll(mesh.boundary(), pi)
            {
                pICt.boundaryFieldRef()[pi] = pEqn.internalCoeffs()[pi];
                pBCt.boundaryFieldRef()[pi] = pEqn.boundaryCoeffs()[pi];
            }
            pICt.write(); pBCt.write();
            volScalarField pDt
            (
                IOobject("stage_pDt", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
            );
            volScalarField pSrct
            (
                IOobject("stage_pSrct", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
            );
            pDt.primitiveFieldRef() = pEqn.D();
            scalarField rhst(pEqn.source());
            forAll(mesh.boundary(), pj)
            {
                const labelUList& fc = mesh.boundary()[pj].faceCells();
                const scalarField& bcp = pEqn.boundaryCoeffs()[pj];
                forAll(fc, i) rhst[fc[i]] += bcp[i];
            }
            pSrct.primitiveFieldRef() = rhst;
            pDt.write(); pSrct.write();
        }

        pEqn.setReference
        (
            pressureControl.refCell(),
            pressureControl.refValue()
        );

        pEqn.solve();

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}
else
{
    closedVolume = adjustPhi(phiHbyA, U, p);

    // Stage harness: phiHbyA AFTER adjustPhi, plus what adjustPhi decided. closedVolume drives the
    // psi-weighted mass correction below, so it is part of the branch's observable behaviour.
    if (runTime.timeIndex() == braeDumpIter())
    {
        surfaceScalarField("stage_phiHbyA", phiHbyA).write();
        Info<< "  pEqn closedVolume = " << closedVolume << endl;
    }

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          - fvm::laplacian(rhorAUf, p)
          ==
            fvOptions(psi, p, rho.name())
        );

        // Stage harness: the pEqn coefficients and assembled system on the SUBSONIC branch too.
        if (runTime.timeIndex() == braeDumpIter())
        {
            dumpedPCoeffsS = true;
            volScalarField pIC
            (
                IOobject("stage_pIC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), calculatedFvPatchScalarField::typeName
            );
            volScalarField pBC
            (
                IOobject("stage_pBC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), calculatedFvPatchScalarField::typeName
            );
            forAll(mesh.boundary(), pi)
            {
                pIC.boundaryFieldRef()[pi] = pEqn.internalCoeffs()[pi];
                pBC.boundaryFieldRef()[pi] = pEqn.boundaryCoeffs()[pi];
            }
            pIC.write(); pBC.write();
            volScalarField pD
            (
                IOobject("stage_pD", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
            );
            volScalarField pSrc
            (
                IOobject("stage_pSrc", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
            );
            pD.primitiveFieldRef() = pEqn.D();
            scalarField rhs(pEqn.source());
            forAll(mesh.boundary(), pj)
            {
                const labelUList& fc = mesh.boundary()[pj].faceCells();
                const scalarField& bcp = pEqn.boundaryCoeffs()[pj];
                forAll(fc, i) rhs[fc[i]] += bcp[i];
            }
            pSrc.primitiveFieldRef() = rhs;
            pD.write(); pSrc.write();
            forAll(mesh.boundary(), pi)
            {
                const fvPatch& pp = mesh.boundary()[pi];
                if (!pp.size()) continue;
                scalar si = 0, sb = 0;
                forAll(pEqn.internalCoeffs()[pi], i)
                {
                    si += mag(pEqn.internalCoeffs()[pi][i]);
                    sb += mag(pEqn.boundaryCoeffs()[pi][i]);
                }
                Info<< "  pEqn patch " << pp.name() << " (" << pp.size() << " faces)"
                    << "  sum|iC| = " << si << "  sum|bC| = " << sb << endl;
            }
        }

        pEqn.setReference
        (
            pressureControl.refCell(),
            pressureControl.refValue()
        );

        pEqn.solve();

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}

#include "incompressible/continuityErrs.H"

if (runTime.timeIndex() == braeDumpIter())
{
    volScalarField("stage_pSolved", p).write();     // straight out of the linear solver
    surfaceScalarField("stage_phiSolved", phi).write();
}

// Explicitly relax pressure for momentum corrector
p.relax();

if (runTime.timeIndex() == braeDumpIter()) volScalarField("stage_pRelaxed", p).write();

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
fvOptions.correct(U);

bool pLimited = pressureControl.limit(p);

// For closed-volume cases adjust the pressure and density levels
// to obey overall mass continuity
if (closedVolume)
{
    p += (initialMass - fvc::domainIntegrate(psi*p))
        /fvc::domainIntegrate(psi);
}

if (pLimited || closedVolume)
{
    p.correctBoundaryConditions();
}

rho = thermo.rho();

if (!simple.transonic())
{
    rho.relax();
}
