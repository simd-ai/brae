/*
    Trace of OpenFOAM's own boundary updates on RAS/TJunction, for the comparison with brae's.

    Each class below DERIVES from the OpenFOAM v2412 boundary condition of the same name and calls that
    class's own updateCoeffs() and evaluate(). Nothing is copied and nothing is recomputed: the trace
    prints, per face, what the update read (the flux, U's or k's patch values) and what it left (the
    value, refValue, valueFraction), tagged with the time index and the PIMPLE outer and pressure
    corrector it ran in. Reading the arithmetic is the whole point, so none of it may change: a run
    with these types writes the same files as the run with OpenFOAM's own (checked by the job that
    uses it).

    Use: rename the patch type in 0/ to <type>Trace, add libs ("<path>/libtraceBoundaryUpdates.so")
    to controlDict, and set BRAE_TRACE_BC_STEPS=<n> to print while the time index is <= n. Unset,
    the library prints nothing.

    Lines: BCT of ti=<index> t=<time> oc=<outer> pc=<pressure corrector> field=<f> patch=<p>
           site=<update|evaluate> f=<face> <name>=<value> ...
    oc and pc are pimpleControl's corr() and corrPISO(): pc is 0 in the momentum phase and in the
    turbulence correction, 1..nCorrectors inside pEqn.H.
*/

#include "uniformTotalPressureFvPatchScalarField.H"
#include "pressureInletOutletVelocityFvPatchVectorField.H"
#include "inletOutletFvPatchField.H"
#include "turbulentIntensityKineticEnergyInletFvPatchScalarField.H"
#include "turbulentMixingLengthDissipationRateInletFvPatchScalarField.H"
#include "pimpleControl.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvPatchFieldMapper.H"
#include "addToRunTimeSelectionTable.H"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace Foam
{
namespace bcTrace
{

inline label maxStep()
{
    static const label n = []()
    {
        const char* e = std::getenv("BRAE_TRACE_BC_STEPS");
        return e ? label(std::atoi(e)) : label(0);
    }();
    return n;
}

inline bool on(const objectRegistry& db)
{
    return db.time().timeIndex() <= maxStep();
}

inline std::string num(const scalar v)
{
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", double(v));
    return b;
}

template<class Type>
inline std::string num(const Type& v)
{
    std::string s;
    for (direction d = 0; d < pTraits<Type>::nComponents; ++d)
    {
        if (d)
        {
            s += ',';
        }
        s += num(component(v, d));
    }
    return s;
}

inline std::string tag
(
    const objectRegistry& db,
    const word& field,
    const word& patch,
    const char* site
)
{
    label oc = -1;
    label pc = -1;
    const pimpleControl* pimple = db.cfindObject<pimpleControl>("solutionControl");
    if (pimple)
    {
        oc = pimple->corr();
        pc = pimple->corrPISO();
    }
    char b[512];
    std::snprintf
    (
        b,
        sizeof b,
        "BCT of ti=%d t=%.17g oc=%d pc=%d field=%s patch=%s site=%s",
        int(db.time().timeIndex()),
        double(db.time().timeOutputValue()),
        int(oc),
        int(pc),
        field.c_str(),
        patch.c_str(),
        site
    );
    return b;
}

// One line per face. Info writes to std::cout as well, so the lines keep their place in the log.
inline void out(const std::string& s)
{
    std::cout << s << '\n' << std::flush;
}

} // End namespace bcTrace


// uniformTotalPressure: p_b = p0(t) - 0.5 neg(phi_b) |U_b|^2. Prints phi_b and U_b as the update read
// them, and the value it left.
class uniformTotalPressureTraceFvPatchScalarField
:
    public uniformTotalPressureFvPatchScalarField
{
public:

    TypeName("uniformTotalPressureTrace");

    uniformTotalPressureTraceFvPatchScalarField
    (
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF
    )
    :
        uniformTotalPressureFvPatchScalarField(p, iF)
    {}

    uniformTotalPressureTraceFvPatchScalarField
    (
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const dictionary& dict
    )
    :
        uniformTotalPressureFvPatchScalarField(p, iF, dict)
    {}

    uniformTotalPressureTraceFvPatchScalarField
    (
        const uniformTotalPressureTraceFvPatchScalarField& ptf,
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const fvPatchFieldMapper& mapper
    )
    :
        uniformTotalPressureFvPatchScalarField(ptf, p, iF, mapper)
    {}

    uniformTotalPressureTraceFvPatchScalarField
    (
        const uniformTotalPressureTraceFvPatchScalarField& ptf
    )
    :
        uniformTotalPressureFvPatchScalarField(ptf)
    {}

    uniformTotalPressureTraceFvPatchScalarField
    (
        const uniformTotalPressureTraceFvPatchScalarField& ptf,
        const DimensionedField<scalar, volMesh>& iF
    )
    :
        uniformTotalPressureFvPatchScalarField(ptf, iF)
    {}

    virtual tmp<fvPatchField<scalar>> clone() const
    {
        return fvPatchField<scalar>::Clone(*this);
    }

    virtual tmp<fvPatchField<scalar>> clone
    (
        const DimensionedField<scalar, volMesh>& iF
    ) const
    {
        return fvPatchField<scalar>::Clone(*this, iF);
    }

    using uniformTotalPressureFvPatchScalarField::updateCoeffs;

    virtual void updateCoeffs()
    {
        if (updated())
        {
            return;
        }
        uniformTotalPressureFvPatchScalarField::updateCoeffs();
        if (!bcTrace::on(db()))
        {
            return;
        }
        const fvPatchVectorField& Up = patch().lookupPatchField<volVectorField>("U");
        const fvsPatchScalarField& phip = patch().lookupPatchField<surfaceScalarField>("phi");
        const std::string t = bcTrace::tag(db(), internalField().name(), patch().name(), "update");
        forAll(*this, facei)
        {
            bcTrace::out
            (
                t + " f=" + std::to_string(facei)
              + " phi=" + bcTrace::num(phip[facei])
              + " Ub=" + bcTrace::num(Up[facei])
              + " value=" + bcTrace::num((*this)[facei])
            );
        }
    }
};

defineTypeNameAndDebug(uniformTotalPressureTraceFvPatchScalarField, 0);
addToPatchFieldRunTimeSelection
(
    fvPatchScalarField,
    uniformTotalPressureTraceFvPatchScalarField
);


// pressureInletOutletVelocity: valueFraction = neg(phi_b)(I - n n), then evaluate (its updateCoeffs
// evaluates too). Prints the flux the switch read, the valueFraction's diagonal, the adjacent cell and
// the value, at the update and at every evaluate.
class pressureInletOutletVelocityTraceFvPatchVectorField
:
    public pressureInletOutletVelocityFvPatchVectorField
{
    void print(const char* site) const
    {
        const fvsPatchScalarField& phip = patch().lookupPatchField<surfaceScalarField>("phi");
        const vectorField Uc(patchInternalField());
        const std::string t = bcTrace::tag(db(), internalField().name(), patch().name(), site);
        forAll(*this, facei)
        {
            const symmTensor& vf = valueFraction()[facei];
            bcTrace::out
            (
                t + " f=" + std::to_string(facei)
              + " phi=" + bcTrace::num(phip[facei])
              + " vf=" + bcTrace::num(vf.xx()) + "," + bcTrace::num(vf.yy()) + "," + bcTrace::num(vf.zz())
              + " Uc=" + bcTrace::num(Uc[facei])
              + " value=" + bcTrace::num((*this)[facei])
            );
        }
    }

public:

    TypeName("pressureInletOutletVelocityTrace");

    pressureInletOutletVelocityTraceFvPatchVectorField
    (
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF
    )
    :
        pressureInletOutletVelocityFvPatchVectorField(p, iF)
    {}

    pressureInletOutletVelocityTraceFvPatchVectorField
    (
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF,
        const dictionary& dict
    )
    :
        pressureInletOutletVelocityFvPatchVectorField(p, iF, dict)
    {}

    pressureInletOutletVelocityTraceFvPatchVectorField
    (
        const pressureInletOutletVelocityTraceFvPatchVectorField& ptf,
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF,
        const fvPatchFieldMapper& mapper
    )
    :
        pressureInletOutletVelocityFvPatchVectorField(ptf, p, iF, mapper)
    {}

    pressureInletOutletVelocityTraceFvPatchVectorField
    (
        const pressureInletOutletVelocityTraceFvPatchVectorField& ptf
    )
    :
        pressureInletOutletVelocityFvPatchVectorField(ptf)
    {}

    pressureInletOutletVelocityTraceFvPatchVectorField
    (
        const pressureInletOutletVelocityTraceFvPatchVectorField& ptf,
        const DimensionedField<vector, volMesh>& iF
    )
    :
        pressureInletOutletVelocityFvPatchVectorField(ptf, iF)
    {}

    virtual tmp<fvPatchField<vector>> clone() const
    {
        return fvPatchField<vector>::Clone(*this);
    }

    virtual tmp<fvPatchField<vector>> clone
    (
        const DimensionedField<vector, volMesh>& iF
    ) const
    {
        return fvPatchField<vector>::Clone(*this, iF);
    }

    virtual void updateCoeffs()
    {
        if (updated())
        {
            return;
        }
        pressureInletOutletVelocityFvPatchVectorField::updateCoeffs();
        if (bcTrace::on(db()))
        {
            print("update");
        }
    }

    virtual void evaluate(const Pstream::commsTypes commsType)
    {
        pressureInletOutletVelocityFvPatchVectorField::evaluate(commsType);
        if (bcTrace::on(db()))
        {
            print("evaluate");
        }
    }
};

defineTypeNameAndDebug(pressureInletOutletVelocityTraceFvPatchVectorField, 0);
addToPatchFieldRunTimeSelection
(
    fvPatchVectorField,
    pressureInletOutletVelocityTraceFvPatchVectorField
);


// inletOutlet: valueFraction = neg(phi_b). Prints the flux and the valueFraction at the update, the
// adjacent cell and the value at every evaluate.
template<class Type>
class inletOutletTraceFvPatchField
:
    public inletOutletFvPatchField<Type>
{
    void print(const char* site) const
    {
        const fvsPatchScalarField& phip =
            this->patch().template lookupPatchField<surfaceScalarField>("phi");
        const Field<Type> pc(this->patchInternalField());
        const std::string t =
            bcTrace::tag(this->db(), this->internalField().name(), this->patch().name(), site);
        forAll(*this, facei)
        {
            bcTrace::out
            (
                t + " f=" + std::to_string(facei)
              + " phi=" + bcTrace::num(phip[facei])
              + " vf=" + bcTrace::num(this->valueFraction()[facei])
              + " cell=" + bcTrace::num(pc[facei])
              + " value=" + bcTrace::num(this->operator[](facei))
            );
        }
    }

public:

    TypeName("inletOutletTrace");

    inletOutletTraceFvPatchField
    (
        const fvPatch& p,
        const DimensionedField<Type, volMesh>& iF
    )
    :
        inletOutletFvPatchField<Type>(p, iF)
    {}

    inletOutletTraceFvPatchField
    (
        const fvPatch& p,
        const DimensionedField<Type, volMesh>& iF,
        const dictionary& dict
    )
    :
        inletOutletFvPatchField<Type>(p, iF, dict)
    {}

    inletOutletTraceFvPatchField
    (
        const inletOutletTraceFvPatchField<Type>& ptf,
        const fvPatch& p,
        const DimensionedField<Type, volMesh>& iF,
        const fvPatchFieldMapper& mapper
    )
    :
        inletOutletFvPatchField<Type>(ptf, p, iF, mapper)
    {}

    inletOutletTraceFvPatchField
    (
        const inletOutletTraceFvPatchField<Type>& ptf
    )
    :
        inletOutletFvPatchField<Type>(ptf)
    {}

    inletOutletTraceFvPatchField
    (
        const inletOutletTraceFvPatchField<Type>& ptf,
        const DimensionedField<Type, volMesh>& iF
    )
    :
        inletOutletFvPatchField<Type>(ptf, iF)
    {}

    virtual tmp<fvPatchField<Type>> clone() const
    {
        return fvPatchField<Type>::Clone(*this);
    }

    virtual tmp<fvPatchField<Type>> clone
    (
        const DimensionedField<Type, volMesh>& iF
    ) const
    {
        return fvPatchField<Type>::Clone(*this, iF);
    }

    virtual void updateCoeffs()
    {
        if (this->updated())
        {
            return;
        }
        inletOutletFvPatchField<Type>::updateCoeffs();
        if (bcTrace::on(this->db()))
        {
            print("update");
        }
    }

    virtual void evaluate(const Pstream::commsTypes commsType)
    {
        inletOutletFvPatchField<Type>::evaluate(commsType);
        if (bcTrace::on(this->db()))
        {
            print("evaluate");
        }
    }
};

makePatchTypeFieldTypedefs(inletOutletTrace);
makePatchFields(inletOutletTrace);


// turbulentIntensityKineticEnergyInlet: refValue = 1.5 I^2 |U_b|^2, valueFraction = neg(phi_b). Prints
// U_b as the update read it, the flux, refValue and valueFraction; the value at every evaluate.
class turbulentIntensityKineticEnergyInletTraceFvPatchScalarField
:
    public turbulentIntensityKineticEnergyInletFvPatchScalarField
{
    void print(const char* site) const
    {
        const fvPatchVectorField& Up = patch().lookupPatchField<volVectorField>("U");
        const fvsPatchScalarField& phip = patch().lookupPatchField<surfaceScalarField>("phi");
        const scalarField kc(patchInternalField());
        const std::string t = bcTrace::tag(db(), internalField().name(), patch().name(), site);
        forAll(*this, facei)
        {
            bcTrace::out
            (
                t + " f=" + std::to_string(facei)
              + " phi=" + bcTrace::num(phip[facei])
              + " Ub=" + bcTrace::num(Up[facei])
              + " refValue=" + bcTrace::num(refValue()[facei])
              + " vf=" + bcTrace::num(valueFraction()[facei])
              + " cell=" + bcTrace::num(kc[facei])
              + " value=" + bcTrace::num((*this)[facei])
            );
        }
    }

public:

    TypeName("turbulentIntensityKineticEnergyInletTrace");

    turbulentIntensityKineticEnergyInletTraceFvPatchScalarField
    (
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF
    )
    :
        turbulentIntensityKineticEnergyInletFvPatchScalarField(p, iF)
    {}

    turbulentIntensityKineticEnergyInletTraceFvPatchScalarField
    (
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const dictionary& dict
    )
    :
        turbulentIntensityKineticEnergyInletFvPatchScalarField(p, iF, dict)
    {}

    turbulentIntensityKineticEnergyInletTraceFvPatchScalarField
    (
        const turbulentIntensityKineticEnergyInletTraceFvPatchScalarField& ptf,
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const fvPatchFieldMapper& mapper
    )
    :
        turbulentIntensityKineticEnergyInletFvPatchScalarField(ptf, p, iF, mapper)
    {}

    turbulentIntensityKineticEnergyInletTraceFvPatchScalarField
    (
        const turbulentIntensityKineticEnergyInletTraceFvPatchScalarField& ptf
    )
    :
        turbulentIntensityKineticEnergyInletFvPatchScalarField(ptf)
    {}

    turbulentIntensityKineticEnergyInletTraceFvPatchScalarField
    (
        const turbulentIntensityKineticEnergyInletTraceFvPatchScalarField& ptf,
        const DimensionedField<scalar, volMesh>& iF
    )
    :
        turbulentIntensityKineticEnergyInletFvPatchScalarField(ptf, iF)
    {}

    virtual tmp<fvPatchField<scalar>> clone() const
    {
        return fvPatchField<scalar>::Clone(*this);
    }

    virtual tmp<fvPatchField<scalar>> clone
    (
        const DimensionedField<scalar, volMesh>& iF
    ) const
    {
        return fvPatchField<scalar>::Clone(*this, iF);
    }

    virtual void updateCoeffs()
    {
        if (updated())
        {
            return;
        }
        turbulentIntensityKineticEnergyInletFvPatchScalarField::updateCoeffs();
        if (bcTrace::on(db()))
        {
            print("update");
        }
    }

    virtual void evaluate(const Pstream::commsTypes commsType)
    {
        turbulentIntensityKineticEnergyInletFvPatchScalarField::evaluate(commsType);
        if (bcTrace::on(db()))
        {
            print("evaluate");
        }
    }
};

defineTypeNameAndDebug(turbulentIntensityKineticEnergyInletTraceFvPatchScalarField, 0);
addToPatchFieldRunTimeSelection
(
    fvPatchScalarField,
    turbulentIntensityKineticEnergyInletTraceFvPatchScalarField
);


// turbulentMixingLengthDissipationRateInlet: refValue = Cmu^0.75/L k_b^1.5, valueFraction = neg(phi_b).
// Prints k's patch values as the update read them, the flux, refValue and valueFraction; the value at
// every evaluate.
class turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField
:
    public turbulentMixingLengthDissipationRateInletFvPatchScalarField
{
    void print(const char* site) const
    {
        const fvPatchScalarField& kp = patch().lookupPatchField<volScalarField>("k");
        const fvsPatchScalarField& phip = patch().lookupPatchField<surfaceScalarField>("phi");
        const scalarField ec(patchInternalField());
        const std::string t = bcTrace::tag(db(), internalField().name(), patch().name(), site);
        forAll(*this, facei)
        {
            bcTrace::out
            (
                t + " f=" + std::to_string(facei)
              + " phi=" + bcTrace::num(phip[facei])
              + " kb=" + bcTrace::num(kp[facei])
              + " refValue=" + bcTrace::num(refValue()[facei])
              + " vf=" + bcTrace::num(valueFraction()[facei])
              + " cell=" + bcTrace::num(ec[facei])
              + " value=" + bcTrace::num((*this)[facei])
            );
        }
    }

public:

    TypeName("turbulentMixingLengthDissipationRateInletTrace");

    turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField
    (
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF
    )
    :
        turbulentMixingLengthDissipationRateInletFvPatchScalarField(p, iF)
    {}

    turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField
    (
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const dictionary& dict
    )
    :
        turbulentMixingLengthDissipationRateInletFvPatchScalarField(p, iF, dict)
    {}

    turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField
    (
        const turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField& ptf,
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const fvPatchFieldMapper& mapper
    )
    :
        turbulentMixingLengthDissipationRateInletFvPatchScalarField(ptf, p, iF, mapper)
    {}

    turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField
    (
        const turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField& ptf
    )
    :
        turbulentMixingLengthDissipationRateInletFvPatchScalarField(ptf)
    {}

    turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField
    (
        const turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField& ptf,
        const DimensionedField<scalar, volMesh>& iF
    )
    :
        turbulentMixingLengthDissipationRateInletFvPatchScalarField(ptf, iF)
    {}

    virtual tmp<fvPatchField<scalar>> clone() const
    {
        return fvPatchField<scalar>::Clone(*this);
    }

    virtual tmp<fvPatchField<scalar>> clone
    (
        const DimensionedField<scalar, volMesh>& iF
    ) const
    {
        return fvPatchField<scalar>::Clone(*this, iF);
    }

    virtual void updateCoeffs()
    {
        if (updated())
        {
            return;
        }
        turbulentMixingLengthDissipationRateInletFvPatchScalarField::updateCoeffs();
        if (bcTrace::on(db()))
        {
            print("update");
        }
    }

    virtual void evaluate(const Pstream::commsTypes commsType)
    {
        turbulentMixingLengthDissipationRateInletFvPatchScalarField::evaluate(commsType);
        if (bcTrace::on(db()))
        {
            print("evaluate");
        }
    }
};

defineTypeNameAndDebug(turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField, 0);
addToPatchFieldRunTimeSelection
(
    fvPatchScalarField,
    turbulentMixingLengthDissipationRateInletTraceFvPatchScalarField
);

} // End namespace Foam
