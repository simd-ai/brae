#pragma once
// liquid_thermo.cuh -- ONE branch point between the gas and liquid property paths.
//
// OpenFOAM asks its mixture for every property through the same seven calls whatever the thermo is:
// `mixture_.mu(p,T)`, `.Cp(p,T)`, `.alphah(p,T)`, `.rho(p,T)`, `.psi(p,T)`, `.He(p,T)`, `.Cpv(p,T)`
// (heRhoThermo.C:60-75, heThermo.C). The selection happens ONCE, when the mixture is constructed, and
// no call site knows which thermo it is talking to. These accessors are the mirror of that: the model
// is a field of ThermoCoeffs, so a call site names the property and nothing else.
//
// WHY THIS EXISTS AS A HEADER RATHER THAN AN `if` AT EACH USE. The gas mirror recomputes mu and alpha
// from T at ~20 places. Adding `if (liquid)` at each is ~20 chances to add it in nineteen and miss one,
// and a missed one is a liquid running a gas correlation -- silently, because the number it returns is
// finite and plausible. The branch is uniform across a launch (it is a case-level setting), so on the
// GPU it costs nothing.
//
// EVERY BRANCH CITES ITS OpenFOAM LINE. The liquid arms come from liquidProperties/H2O/H2O.C and
// liquidPropertiesI.H; the gas arms are byte-identical to the calls they replace, which is what lets
// stage H3.1 be a pure refactor with every existing compressible gate unmoved.
#include "cf_types.cuh"
#include "thermo_types.cuh"
#include "thermo_model.cuh"
#include "transport_model.cuh"
#include "equation_of_state.cuh"
#include "nsrds_functions.cuh"

namespace brae {

// A liquid's correlations are functions of T alone (liquidProperties treats these substances as
// pressure-independent -- see nsrds_functions.cuh), so `p` is unused on the liquid arm and carried only
// because OpenFOAM's own signature carries it.

// mu(p,T). liquid: NSRDSfunc1 (H2O.C mu_). gas: Sutherland or the constant mu0.
BRAE_HD inline scalar thermoMuOf(scalar /*p*/, scalar T, const ThermoCoeffs& c)
{
    return (c.model == ThermoModel::liquidH2O) ? H2OLiquid::mu(T) : transportMu(T, c);
}

// Cp(p,T). liquid: NSRDSfunc0 (H2O.C Cp_). gas: the constant Cp of hConst.
BRAE_HD inline scalar thermoCpOf(scalar /*p*/, scalar T, const ThermoCoeffs& c)
{
    return (c.model == ThermoModel::liquidH2O) ? H2OLiquid::Cp(T) : c.Cp;
}

// alphah(p,T) [kg/(m s)] -- OF's "thermal diffusivity for ENTHALPY", kappa/Cp.
// liquid: liquidPropertiesI.H:130-133, `alphah = kappa(p,T)/Cp(p,T)`, both correlations.
// gas: the transport model's own alphah, unchanged.
BRAE_HD inline scalar thermoAlphaOf(scalar /*p*/, scalar T, const ThermoCoeffs& c)
{
    if (c.model == ThermoModel::liquidH2O) return H2OLiquid::kappa(T) / H2OLiquid::Cp(T);
    return transportAlpha(transportMu(T, c), c);
}

// rho(p,T). liquid: NSRDSfunc5 (H2O.C rho_) -- a function of T ONLY, so it does not respond to p at
// all, which is what makes the closed-volume pressure correction degenerate (see thermoPsiOf).
BRAE_HD inline scalar thermoRhoOf(scalar p, scalar T, const ThermoCoeffs& c)
{
    return (c.model == ThermoModel::liquidH2O) ? H2OLiquid::rho(T) : perfectGasRho(p, T, c);
}

// psi(p,T) = d(rho)/d(p)|T. liquid: EXACTLY ZERO -- liquidPropertiesI.H:100-103 is `return 0;`, and
// OpenFOAM's own header says "currently it is assumed the liquid is incompressible". Verified live with
// `rhoSimpleFoam -postProcess -func 'writeObjects(thermo:psi)'` on squareBendLiq: `uniform 0`.
// This must return a REAL zero rather than be left unset: the transonic pEqn interpolates psi and the
// closed-volume correction sums psi*V, and both have to see it (stage H3.0's guards).
BRAE_HD inline scalar thermoPsiOf(scalar /*p*/, scalar T, const ThermoCoeffs& c)
{
    return (c.model == ThermoModel::liquidH2O) ? scalar(0) : perfectGasPsi(T, c);
}

// He(p,T) -- the energy variable the solver transports, sensibleEnthalpy or sensibleInternalEnergy.
// liquid: h2oEnergy, which integrates the h_ ROW (not Cp*(T-Tref)) -- see nsrds_functions.cuh on why
// those differ. gas: hConst's closed form.
BRAE_HD inline scalar thermoHeOf(scalar p, scalar T, const ThermoCoeffs& c)
{
    return (c.model == ThermoModel::liquidH2O) ? h2oEnergy(c.internalEnergy ? EnergyForm::sensibleInternalEnergy
                                                                        : EnergyForm::sensibleEnthalpy, p, T)
                                               : hConstTToHe(T, c);
}

// Cpv(p,T) -- OF's Cpv: the Newton inversion's derivative and the energy BCs' coefficient.
// liquid: Cp for BOTH energy forms, because liquidProperties::CpMCv is 0 (liquidPropertiesI.H:104) and
// Cv = Cp - CpMCv -- see h2oCpv, which also records that this is deliberately NOT the exact dEs/dT and
// why reproducing OpenFOAM's choice is the point. gas: Cp for enthalpy, Cv for internal energy.
BRAE_HD inline scalar thermoCpvOf(scalar p, scalar T, const ThermoCoeffs& c)
{
    if (c.model == ThermoModel::liquidH2O)
        return h2oCpv(c.internalEnergy ? EnergyForm::sensibleInternalEnergy
                                       : EnergyForm::sensibleEnthalpy, p, T);
    return c.internalEnergy ? thermoCv(c) : c.Cp;
}

// The temperatures the thermo's correlations are defined on, so a call site can refuse a field that
// leaves them without knowing which thermo it holds. Gas: no range -- hConst and perfectGas are
// closed forms, and the ten compressible gates run them exactly as before. Liquid: H2O's [Tt, Tc]
// (see H2OLiquid::inRange for why above Tc is a NaN and not merely an extrapolation).
struct ThermoTRange
{
    bool        bounded;
    scalar      lo;
    scalar      hi;
    const char* substance;
};

BRAE_HD inline ThermoTRange thermoTRangeOf(const ThermoCoeffs& c)
{
    if (c.model == ThermoModel::liquidH2O) return ThermoTRange{true, H2OLiquid::Tt, H2OLiquid::Tc, "H2O"};
    return ThermoTRange{false, scalar(0), scalar(0), "perfectGas"};
}

BRAE_HD inline bool thermoTInRange(scalar T, const ThermoCoeffs& c)
{
    return (c.model == ThermoModel::liquidH2O) ? H2OLiquid::inRange(T) : true;
}

// THE(he, p, T0) -- the inverse of thermoHeOf, and the eighth accessor. OpenFOAM asks its mixture for it
// through the same one call whatever the thermo is: `mixture_.THE(hCells[celli], pCells[celli],
// TCells[celli])` (heRhoThermo.C:76-82, and the same line on the boundary at :133).
//
// NOTE THE THIRD ARGUMENT. It is the CURRENT temperature, and it is not a convenience: OpenFOAM's
// inversion is a do-while whose tolerance Ttol = T0*1e-4 is computed once from the initial guess and
// never updated (species::thermo<>::T, thermoI.H:43-88), so THE ANSWER DEPENDS ON T0 -- measured with
// tools/liqref as a 2.4e-08 K spread over six starting guesses at the same (p, he). A caller that passes
// a fixed seed instead of the field's own previous value is running a different function from OpenFOAM's,
// however close its answer looks. See nsrds_functions.cuh for the transcription and the two other things
// about that loop that a from-scratch inversion would get wrong.
//
// The gas branch is p- and T0-independent because hConst's he is affine in T, so the closed form is the
// fixed point OpenFOAM's loop lands on; it is what the ten compressible gates measure and it is left
// exactly as it was.
BRAE_HD inline HeToTResult thermoHeToT(
    scalar he,
    scalar p,
    scalar T0,
    const ThermoCoeffs& c)
{
    if (c.model == ThermoModel::liquidH2O)
        return h2oEnergyToT(c.internalEnergy ? EnergyForm::sensibleInternalEnergy
                                             : EnergyForm::sensibleEnthalpy,
                            he, p, T0);
    HeToTResult r;
    r.T         = hConstHeToT(he, c);
    r.converged = true;
    return r;
}

}   // namespace brae
