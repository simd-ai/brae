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
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2019-2022 OpenCFD Ltd.
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

Class
    Foam::LESModels::SpalartAllmarasIDDESDump

Group
    grpDESTurbulence

Description
    SpalartAllmaras IDDES turbulence model
    for incompressible and compressible flows.

    Reference:
    \verbatim
        Shur, M. L., Spalart, P. R., Strelets, M. K., & Travin, A. K. (2008).
        A hybrid RANS-LES approach with delayed-DES
        and wall-modelled LES capabilities.
        International journal of heat and fluid flow, 29(6), 1638-1649.
        DOI:10.1016/j.ijheatfluidflow.2008.07.001
    \endverbatim

SourceFiles
    SpalartAllmarasIDDESDump.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_SpalartAllmarasIDDESDump_H
#define Foam_SpalartAllmarasIDDESDump_H

#include "SpalartAllmarasDES.H"
#include "IDDESDelta.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

/*---------------------------------------------------------------------------*\
                    Class SpalartAllmarasIDDESDump Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class SpalartAllmarasIDDESDump
:
    public SpalartAllmarasDES<BasicTurbulenceModel>
{
    // Private Member Functions

        //- Check that the supplied delta is an IDDESDelta
        const IDDESDelta& setDelta() const;

        tmp<volScalarField> alpha() const;

        tmp<volScalarField> ft(const volScalarField& magGradU) const;

        tmp<volScalarField> fl(const volScalarField& magGradU) const;

        //- Delay function
        tmp<volScalarField> fdt(const volScalarField& magGradU) const;

        //- ADDED: write dTilda's terms (see the note at the top of this file)
        void dump
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
        ) const;

        //- No copy construct
        SpalartAllmarasIDDESDump(const SpalartAllmarasIDDESDump&) = delete;

        //- No copy assignment
        void operator=(const SpalartAllmarasIDDESDump&) = delete;


protected:

    // Protected Data

        // Model coefficients

            dimensionedScalar Cdt1_;
            dimensionedScalar Cdt2_;
            dimensionedScalar Cl_;
            dimensionedScalar Ct_;
            Switch fe_;


        //- IDDES delta
        const IDDESDelta& IDDESDelta_;


    // Protected Member Functions

        //- Return the length scale
        virtual tmp<volScalarField> dTilda
        (
            const volScalarField& chi,
            const volScalarField& fv1,
            const volTensorField& gradU
        ) const;


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("SpalartAllmarasIDDESDump");


    // Constructors

        //- Construct from components
        SpalartAllmarasIDDESDump
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~SpalartAllmarasIDDESDump() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read();

        //- Return the shielding function
        virtual tmp<volScalarField> fd() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "SpalartAllmarasIDDESDump.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
