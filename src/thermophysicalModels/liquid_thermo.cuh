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

}   // namespace brae
