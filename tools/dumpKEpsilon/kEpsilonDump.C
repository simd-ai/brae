/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2023 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "kEpsilonDump.H"
#include "fvOptions.H"
#include "bound.H"
#include "zeroGradientFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// Which SIMPLE iteration to dump (BRAE_DUMP_STAGE_ITER, default 1) -- the same control dumpPEqn uses, so
// one environment variable lines the two harnesses up on the same iteration.
static int braeDumpKEIter()
{
    const char* e = ::getenv("BRAE_DUMP_STAGE_ITER");
    const int v = (e && *e) ? ::atoi(e) : 1;
    return v > 0 ? v : 1;
}

// D() and the full right-hand side of an assembled scalar system -- the two observables of a matrix, in
// the same form the pEqn and EEqn harnesses already write.
static void braeDumpSystem
(
    const fvMesh& mesh,
    const fvScalarMatrix& eqn,
    const word& dName,
    const word& sName
)
{
    volScalarField D
    (
        IOobject(dName, mesh.time().timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
    );
    volScalarField S(D); S.rename(sName);
    D.primitiveFieldRef() = eqn.D();
    scalarField rhs(eqn.source());
    forAll(mesh.boundary(), pj)
    {
        const labelUList& fc = mesh.boundary()[pj].faceCells();
        const scalarField& bcp = eqn.boundaryCoeffs()[pj];
        forAll(fc, i) rhs[fc[i]] += bcp[i];
    }
    S.primitiveFieldRef() = rhs;
    D.write(); S.write();

    // The off-diagonals, the only part of the system a per-cell D/source comparison cannot see. Written on
    // the internal faces they belong to; an asymmetric matrix has both, a symmetric one repeats upper.
    surfaceScalarField up
    (
        IOobject(dName + "Upper", mesh.time().timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh, dimensionedScalar(dimless, Zero)
    );
    surfaceScalarField lo(up); lo.rename(dName + "Lower");
    if (eqn.hasUpper()) up.primitiveFieldRef() = eqn.upper();
    if (eqn.hasLower()) lo.primitiveFieldRef() = eqn.lower();
    else if (eqn.hasUpper()) lo.primitiveFieldRef() = eqn.upper();
    up.write(); lo.write();
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void kEpsilonDump<BasicTurbulenceModel>::correctNut()
{
    this->nut_ = Cmu_*sqr(k_)/epsilon_;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kEpsilonDump<BasicTurbulenceModel>::kSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        k_,
        dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
    );
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kEpsilonDump<BasicTurbulenceModel>::epsilonSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        epsilon_,
        dimVolume*this->rho_.dimensions()*epsilon_.dimensions()/dimTime
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kEpsilonDump<BasicTurbulenceModel>::kEpsilonDump
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    Cmu_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "Cmu",
            this->coeffDict_,
            0.09
        )
    ),
    C1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "C1",
            this->coeffDict_,
            1.44
        )
    ),
    C2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "C2",
            this->coeffDict_,
            1.92
        )
    ),
    C3_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "C3",
            this->coeffDict_,
            0
        )
    ),
    sigmak_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "sigmak",
            this->coeffDict_,
            1.0
        )
    ),
    sigmaEps_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "sigmaEps",
            this->coeffDict_,
            1.3
        )
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    epsilon_
    (
        IOobject
        (
            IOobject::groupName("epsilon", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    )
{
    bound(k_, this->kMin_);
    bound(epsilon_, this->epsilonMin_);

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool kEpsilonDump<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        Cmu_.readIfPresent(this->coeffDict());
        C1_.readIfPresent(this->coeffDict());
        C2_.readIfPresent(this->coeffDict());
        C3_.readIfPresent(this->coeffDict());
        sigmak_.readIfPresent(this->coeffDict());
        sigmaEps_.readIfPresent(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void kEpsilonDump<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    const volScalarField& nut = this->nut_;

    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))().v()
    );

    tmp<volTensorField> tgradU = fvc::grad(U);
    const volScalarField::Internal GbyNu
    (
        IOobject::scopedName(this->type(), "GbyNu"),
        tgradU().v() && devTwoSymm(tgradU().v())
    );
    const volScalarField::Internal G(this->GName(), nut()*GbyNu);
    // brae oracle: the velocity gradient itself, before it is discarded. GbyNu is built from nothing else,
    // so a disagreement in GbyNu is either here or in the invariant, and this write separates the two.
    if (this->mesh_.time().timeIndex() == braeDumpKEIter())
    {
        volTensorField dumpGradU
        (
            IOobject("stage_gradU", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            tgradU()
        );
        dumpGradU.write();
    }

    tgradU.clear();

    // brae oracle: the three quantities the epsilon and k equations are built from. Nothing above is
    // altered -- this is OpenFOAM's own kEpsilon with writes added, so what comes out is what OpenFOAM
    // used and not a reconstruction of it.
    if (this->mesh_.time().timeIndex() == braeDumpKEIter())
    {
        volScalarField dumpDivU
        (
            IOobject("stage_divU", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            this->mesh_, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        volScalarField dumpGbyNu(dumpDivU); dumpGbyNu.rename("stage_GbyNu");
        volScalarField dumpG(dumpDivU);     dumpG.rename("stage_G");
        dumpDivU.primitiveFieldRef()  = divU;
        dumpGbyNu.primitiveFieldRef() = GbyNu;
        dumpG.primitiveFieldRef()     = G;
        dumpDivU.write(); dumpGbyNu.write(); dumpG.write();

        // The VOLUMETRIC flux the dilatation is taken from, so brae's own phi/interpolate(rho) can be
        // compared against the one compressibleTurbulenceModel::phi() actually returns.
        surfaceScalarField("stage_phiVol", fvc::absolute(this->phi(), U)).write();
    }


    // Update epsilon and G at the wall
    epsilon_.boundaryFieldRef().updateCoeffs();
    // Push any changed cell values to coupled neighbours
    epsilon_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();

    // brae oracle: the epsilon equation's two fvm terms on their own, plus the diffusivities they are
    // built from. G and divU already agree, so a disagreement in the assembled system is convection,
    // diffusion or a source, and separating them here says which without altering the equation below.
    if (this->mesh_.time().timeIndex() == braeDumpKEIter())
    {
        tmp<fvScalarMatrix> tdiv(fvm::div(alphaRhoPhi, epsilon_));
        braeDumpSystem(this->mesh_, tdiv.ref(), "stage_epsDivD", "stage_epsDivSrc");
        tmp<fvScalarMatrix> tlap(fvm::laplacian(alpha*rho*DepsilonEff(), epsilon_));
        braeDumpSystem(this->mesh_, tlap.ref(), "stage_epsLapD", "stage_epsLapSrc");

        volScalarField dEps
        (
            IOobject("stage_DepsilonEff", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            DepsilonEff()
        );
        dEps.write();
        volScalarField dK
        (
            IOobject("stage_DkEff", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            DkEff()
        );
        dK.write();
        // The mesh quantities the laplacian's face coefficient is built from. gamma_f*ndc*magSf is the
        // whole of it, so if the diffusivity agrees and the coefficient does not, these say which factor.
        surfaceScalarField ndc
        (
            IOobject("stage_nonOrthDeltaCoeffs", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            this->mesh_.nonOrthDeltaCoeffs()
        );
        ndc.write();
        surfaceScalarField dc
        (
            IOobject("stage_deltaCoeffs", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            this->mesh_.deltaCoeffs()
        );
        dc.write();
        surfaceScalarField gam
        (
            IOobject("stage_gammaEpsFace", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            fvc::interpolate(alpha*rho*DepsilonEff())
        );
        gam.write();
        surfaceScalarField wts
        (
            IOobject("stage_weights", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            this->mesh_.weights()
        );
        wts.write();
        surfaceScalarField msf
        (
            IOobject("stage_magSf", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            this->mesh_.magSf()
        );
        msf.write();

        surfaceScalarField arp
        (
            IOobject("stage_alphaRhoPhi", this->mesh_.time().timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE),
            alphaRhoPhi
        );
        arp.write();
    }

    // Dissipation equation
    tmp<fvScalarMatrix> epsEqn
    (
        fvm::ddt(alpha, rho, epsilon_)
      + fvm::div(alphaRhoPhi, epsilon_)
      - fvm::laplacian(alpha*rho*DepsilonEff(), epsilon_)
     ==
        C1_*alpha()*rho()*GbyNu*Cmu_*k_()
      - fvm::SuSp(((2.0/3.0)*C1_ - C3_)*alpha()*rho()*divU, epsilon_)
      - fvm::Sp(C2_*alpha()*rho()*epsilon_()/k_(), epsilon_)
      + epsilonSource()
      + fvOptions(alpha, rho, epsilon_)
    );

    if (this->mesh_.time().timeIndex() == braeDumpKEIter())
        braeDumpSystem(this->mesh_, epsEqn.ref(), "stage_epsD0", "stage_epsSrc0");

    epsEqn.ref().relax();
    fvOptions.constrain(epsEqn.ref());
    epsEqn.ref().boundaryManipulate(epsilon_.boundaryFieldRef());
    if (this->mesh_.time().timeIndex() == braeDumpKEIter())
        braeDumpSystem(this->mesh_, epsEqn.ref(), "stage_epsD", "stage_epsSrc");

    solve(epsEqn);
    fvOptions.correct(epsilon_);
    bound(epsilon_, this->epsilonMin_);

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(), k_)
     ==
        alpha()*rho()*G
      - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
      - fvm::Sp(alpha()*rho()*epsilon_()/k_(), k_)
      + kSource()
      + fvOptions(alpha, rho, k_)
    );

    if (this->mesh_.time().timeIndex() == braeDumpKEIter())
        braeDumpSystem(this->mesh_, kEqn.ref(), "stage_kD0", "stage_kSrc0");

    kEqn.ref().relax();
    fvOptions.constrain(kEqn.ref());
    if (this->mesh_.time().timeIndex() == braeDumpKEIter())
        braeDumpSystem(this->mesh_, kEqn.ref(), "stage_kD", "stage_kSrc");

    solve(kEqn);
    fvOptions.correct(k_);
    bound(k_, this->kMin_);

    correctNut();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
