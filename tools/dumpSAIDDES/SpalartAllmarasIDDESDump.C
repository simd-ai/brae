// INSTRUMENTED COPY of OpenFOAM v2412's SpalartAllmarasIDDES (src/TurbulenceModels/turbulenceModels/DES/
// SpalartAllmarasIDDES), renamed SpalartAllmarasIDDESDump so a case selects it beside the original:
//     LESModel SpalartAllmarasIDDESDump;   libs ("libdumpSAIDDES.so");
// The model's statements are OpenFOAM's, unaltered. ADDED: dTilda() writes its own terms, cell by cell at 17
// digits, to <case>/saIddes_dump.txt when BRAE_DUMP_ITER names the time index (the last call of that step
// wins). Its coefficient sub-dictionary is SpalartAllmarasIDDESDumpCoeffs. brae's oracle for
// tests/sa_iddes_vs_openfoam.sh.
/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2015-2022 OpenCFD Ltd.
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

#include "SpalartAllmarasIDDESDump.H"
#include "OFstream.H"
#include <cstdlib>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
const IDDESDelta& SpalartAllmarasIDDESDump<BasicTurbulenceModel>::setDelta() const
{
    if (!isA<IDDESDelta>(this->delta_()))
    {
        FatalErrorInFunction
            << "The delta function must be set to a " << IDDESDelta::typeName
            << " -based model" << exit(FatalError);
    }

    return refCast<const IDDESDelta>(this->delta_());
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasIDDESDump<BasicTurbulenceModel>::alpha() const
{
    return max(0.25 - this->y_/IDDESDelta_.hmax(), scalar(-5));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasIDDESDump<BasicTurbulenceModel>::ft
(
    const volScalarField& magGradU
) const
{
    return tanh(pow3(sqr(Ct_)*this->r(this->nut_, magGradU, this->y_)));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasIDDESDump<BasicTurbulenceModel>::fl
(
    const volScalarField& magGradU
) const
{
    return tanh(pow(sqr(Cl_)*this->r(this->nu(), magGradU, this->y_), 10));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasIDDESDump<BasicTurbulenceModel>::fdt
(
    const volScalarField& magGradU
) const
{
    return 1 - tanh(pow(Cdt1_*this->r(this->nut_, magGradU, this->y_), Cdt2_));
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasIDDESDump<BasicTurbulenceModel>::dTilda
(
    const volScalarField& chi,
    const volScalarField& fv1,
    const volTensorField& gradU
) const
{
    const volScalarField magGradU(mag(gradU));
    const volScalarField psi(this->psi(chi, fv1));

    const volScalarField& lRAS = this->y_;
    const volScalarField lLES(psi*this->CDES_*this->delta());

    const volScalarField alpha(this->alpha());
    const volScalarField expTerm(exp(sqr(alpha)));

    tmp<volScalarField> fB = min(2*pow(expTerm, -9.0), scalar(1));
    const volScalarField fdTilda(max(1 - fdt(magGradU), fB));

    if (fe_)
    {
        tmp<volScalarField> fe1 =
            2*lerp(pow(expTerm, -9.), pow(expTerm, -11.09), pos0(alpha));
        tmp<volScalarField> fe2 = 1 - max(ft(magGradU), fl(magGradU));
        // ADDED: copies for the dump, taken before the next statement consumes the two tmps
        const volScalarField fe1Dump(fe1());
        const volScalarField fe2Dump(fe2());
        tmp<volScalarField> fe = max(fe1 - 1, scalar(0))*psi*fe2;
        const volScalarField feDump(fe());

        // Original formulation from Shur et al. paper (2008)
        tmp<volScalarField> tdTilda = max
        (
            fdTilda*(1 + fe)*lRAS + (1 - fdTilda)*lLES,
            dimensionedScalar("SMALL", dimLength, SMALL)
        );
        dump(gradU, magGradU, psi, lLES, alpha, fdTilda, fe1Dump, fe2Dump, feDump, tdTilda());
        return tdTilda;
    }


    // Simplified formulation from Gritskevich et al. paper (2011) where fe = 0
    tmp<volScalarField> tdTilda = max
    (
        lerp(lLES, lRAS, fdTilda),
        dimensionedScalar("SMALL", dimLength, SMALL)
    );
    {
        const volScalarField zero(0*alpha);
        dump(gradU, magGradU, psi, lLES, alpha, fdTilda, zero, zero, zero, tdTilda());
    }
    return tdTilda;
}


// ADDED: the dump. One header line (the constructed model's constants), then a line a cell:
//   y hmax delta gradU[9] nuTilda nut nu psi alpha fB fdt ft fl fdTilda fe1 fe2 fe lLES dTilda
template<class BasicTurbulenceModel>
void SpalartAllmarasIDDESDump<BasicTurbulenceModel>::dump
(
    const volTensorField& gradU,
    const volScalarField& magGradU,
    const volScalarField& psi,
    const volScalarField& lLES,
    const volScalarField& alpha,
    const volScalarField& fdTilda,
    const volScalarField& fe1,
    const volScalarField& fe2,
    const volScalarField& fe,
    const volScalarField& dTilda
) const
{
    if
    (
        !getenv("BRAE_DUMP_ITER")
     || this->runTime_.timeIndex() != atoi(getenv("BRAE_DUMP_ITER"))
    )
    {
        return;
    }
    const volScalarField expTerm(exp(sqr(alpha)));
    const volScalarField fB(min(2*pow(expTerm, -9.0), scalar(1)));
    const volScalarField fdtF(fdt(magGradU));
    const volScalarField ftF(ft(magGradU));
    const volScalarField flF(fl(magGradU));
    const volScalarField& hmax = IDDESDelta_.hmax();
    const volScalarField& delta = this->delta();
    const volScalarField nu(this->nu());

    OFstream os(this->runTime_.path()/"saIddes_dump.txt");
    os.precision(17);
    os  << "# cells " << this->mesh_.nCells()
        << " Cdt1 " << Cdt1_.value() << " Cdt2 " << Cdt2_.value()
        << " Cl " << Cl_.value() << " Ct " << Ct_.value()
        << " fe " << (fe_ ? 1 : 0)
        << " CDES " << this->CDES_.value() << " fwStar " << this->fwStar_.value()
        << " lowReCorrection " << (this->lowReCorrection_ ? 1 : 0)
        << " kappa " << this->kappa_.value() << " Cv1 " << this->Cv1_.value()
        << " Cb1 " << this->Cb1_.value() << " Cw1 " << this->Cw1_.value()
        << nl;
    forAll(dTilda, celli)
    {
        const tensor& g = gradU[celli];
        os  << this->y_[celli] << ' ' << hmax[celli] << ' ' << delta[celli];
        for (direction k = 0; k < 9; ++k)
        {
            os  << ' ' << g[k];
        }
        os  << ' ' << this->nuTilda_[celli] << ' ' << this->nut_[celli]
            << ' ' << nu[celli] << ' ' << psi[celli] << ' ' << alpha[celli]
            << ' ' << fB[celli] << ' ' << fdtF[celli] << ' ' << ftF[celli]
            << ' ' << flF[celli] << ' ' << fdTilda[celli] << ' ' << fe1[celli]
            << ' ' << fe2[celli] << ' ' << fe[celli] << ' ' << lLES[celli]
            << ' ' << dTilda[celli] << nl;
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SpalartAllmarasIDDESDump<BasicTurbulenceModel>::SpalartAllmarasIDDESDump
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
    SpalartAllmarasDES<BasicTurbulenceModel>
    (
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName,
        type
    ),

    Cdt1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "Cdt1",
            this->coeffDict_,
            8
        )
    ),
    Cdt2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "Cdt2",
            this->coeffDict_,
            3
        )
    ),
    Cl_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "Cl",
            this->coeffDict_,
            3.55
        )
    ),
    Ct_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "Ct",
            this->coeffDict_,
            1.63
        )
    ),
    fe_
    (
        Switch::getOrAddToDict
        (
            "fe",
            this->coeffDict_,
            true
        )
    ),

    IDDESDelta_(setDelta())
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool SpalartAllmarasIDDESDump<BasicTurbulenceModel>::read()
{
    if (SpalartAllmarasDES<BasicTurbulenceModel>::read())
    {
        Cdt1_.readIfPresent(this->coeffDict());
        Cdt2_.readIfPresent(this->coeffDict());
        Cl_.readIfPresent(this->coeffDict());
        Ct_.readIfPresent(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasIDDESDump<BasicTurbulenceModel>::fd() const
{
    const volScalarField alpha(this->alpha());
    const volScalarField expTerm(exp(sqr(alpha)));

    tmp<volScalarField> fB = min(2*pow(expTerm, -9.0), scalar(1));
    return max(1 - fdt(mag(fvc::grad(this->U_))), fB);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //
