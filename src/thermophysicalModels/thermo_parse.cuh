#pragma once
// thermo_parse.cuh -- read constant/thermophysicalProperties into a ThermoCoeffs.
//
// Only the perfectGas + hConst + (sutherland | const) combination is supported today. Anything else is
// refused by name, with the supported set listed, rather than silently falling back to a default: a wrong
// equation of state does not announce itself in the residuals, it just gives the wrong answer. This is the
// same contract solver_dispatch.cuh applies to an unknown controlDict `application`.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "thermo_types.cuh"
#include "thermo_model.cuh"   // thermoCv
#include "foam_constants.cuh"   // foamRR(): OF's DimensionedConstants NA*k
#include <stdexcept>
#include <string>

namespace brae {

// The universal gas constant lives in foam_constants.cuh (foamRR()), resolved at RUNTIME the way OF
// resolves it: RR = 1e3*NA*k from `DimensionedConstants` in OpenFOAM's etc/controlDict, falling back to
// OF v2412's own value when that file is not reachable.
//
// It was a compile-time constant here, and wrong once already: brae carried the CODATA-2018
// 8314.46261815324 while OF v2412 computes 8314.47006650545. That 8.958e-07 relative gap propagated to R,
// psi and rho in every compressible case, and was found as an unexplained 9.06e-07 floor on rho in
// hf_vs_openfoam that SURVIVED after brae was made to stop at OF's own iteration count -- OF's written rho
// missed OF's own p/(R*T) by 8.958e-07 while brae's matched its own to 1e-11, which pointed at the constant
// rather than at convergence. brae's number was the more physically current one, and that is precisely why
// it was wrong: the contract is to reproduce OpenFOAM.

inline void thermoRequire(
    bool ok,
    const std::string& key,
    const std::string& got,
    const std::string& supported)
{
    if (ok) return;
    throw std::runtime_error(
        "brae: thermophysicalProperties thermoType." + key + " = '" + got
        + "' is not supported. brae supports: " + supported
        + ". (rhoSimpleFoam scope today is subsonic perfectGas.)");
}

// rhoMin/rhoMax/pMin and the rho relaxation factor are SOLVER controls, not thermo properties, so they
// come from fvSolution and apply to every thermo model. Shared by the gas and liquid paths rather than
// duplicated: the liquid path needs relaxRho just as much (squareBendLiq relaxes rho), and a second copy
// is a second thing to forget when a control is added.
inline void readCommonSolutionControls(
    ThermoCoeffs& c,
    const std::string& caseDir,
    const FoamDict* fvSolutionIn)
{
    const FoamDict fvSolutionOwn = fvSolutionIn ? FoamDict{} : readDict(caseDir + "/system/fvSolution");
    const FoamDict& fvSolution = fvSolutionIn ? *fvSolutionIn : fvSolutionOwn;
    const FoamDict* simple = fvSolution.subDict("SIMPLE");
    if (simple)
    {
        c.rhoMin = simple->scalarOr("rhoMin", c.rhoMin);
        c.rhoMax = simple->scalarOr("rhoMax", c.rhoMax);
        c.pMin   = simple->scalarOr("pMin", c.pMin);
    }
    // rho.relax() factor lives under relaxationFactors.fields.rho, as in OF.
    const FoamDict* relax = fvSolution.subDict("relaxationFactors");
    if (relax)
    {
        const FoamDict* fields = relax->subDict("fields");
        if (fields) c.relaxRho = fields->scalarOr("rho", c.relaxRho);
    }
}

// Reads the dictionary and fills ThermoCoeffs. Throws on anything outside the supported set.
// `fvSolutionIn` lets the caller hand over the fvSolution it ALREADY read. That is not just a saved file
// read: FoamDict records which keys were queried so dict_audit can name the entries brae ignored, and a
// second independent readDict of the same file records those lookups on an object nobody audits. rhoMin,
// rhoMax and relaxationFactors/fields/rho were reported unread for exactly that reason while being read
// perfectly well here -- an audit that cries wolf is an audit people stop reading.
inline ThermoCoeffs readThermoCoeffs(const std::string& caseDir, const FoamDict* fvSolutionIn = nullptr)
{
    const FoamDict dict = readDict(caseDir + "/constant/thermophysicalProperties");
    ThermoCoeffs c;

    const FoamDict* tt = dict.subDict("thermoType");
    if (!tt)
    {
        throw std::runtime_error(
            "brae: constant/thermophysicalProperties has no thermoType dictionary.");
    }

    // Refuse the unsupported combinations up front, before any number is read.
    const std::string type = tt->wordOr("type", "");
    const std::string mixture = tt->wordOr("mixture", "");
    const std::string transport = tt->wordOr("transport", "");
    const std::string thermo = tt->wordOr("thermo", "");
    const std::string eos = tt->wordOr("equationOfState", "");
    const std::string energy = tt->wordOr("energy", "");

    // heRhoThermo is accepted ONLY with perfectGas, where it is provably the same arithmetic:
    //   hePsiThermo::calculate  -> psi = mixture.psi(p,T);           rho comes out as psi*p
    //   heRhoThermo::calculate  -> psi = mixture.psi(p,T);  rho = mixture.rho(p,T)
    //   perfectGas              -> rho = p/(R T),  psi = 1/(R T)  =>  rho == psi*p exactly
    // Both then take mu and alphah from the same mixture functions. So for perfectGas the two thermo
    // types are bit-identical and refusing heRhoThermo blocks real cases for no reason.
    //
    // For ANY other equationOfState (rhoConst, perfectFluid, icoPolynomial, Boussinesq, liquids) rho is
    // NOT psi*p, the two genuinely differ, and heRhoThermo stays refused -- accepting it there would run
    // the wrong density and converge.
    // LIQUID PATH. `properties liquid` selects OF's liquidProperties: there is no equationOfState, no
    // transport and no thermo entry at all, because every property is a temperature correlation instead.
    // OF builds it as (basic/rhoThermo/liquidThermo.H)
    //   heRhoThermo<rhoThermo, pureMixture<species::thermo<
    //       thermophysicalPropertiesSelector<liquidProperties>, sensibleInternalEnergy>>>
    //
    // EXACTLY ONE COMBINATION IS ACCEPTED, and deliberately not one more. OF also registers the
    // sensibleEnthalpy variant, and liquidProperties carries dozens of substances; neither has been run
    // end to end here, and "mathematically reachable" is not the same as "validated". Anything broader
    // keeps falling through to the refusal below with a message naming what it found.
    const std::string properties = tt->wordOr("properties", "");
    if (properties == "liquid")
    {
        thermoRequire(type == "heRhoThermo", "type", type,
                      "heRhoThermo (the only thermo type OF builds `properties liquid` with)");
        thermoRequire(mixture == "pureMixture", "mixture", mixture, "pureMixture");
        thermoRequire(energy == "sensibleInternalEnergy", "energy", energy,
                      "sensibleInternalEnergy for `properties liquid` (the sensibleEnthalpy variant "
                      "exists in OpenFOAM but has not been validated here)");
        const FoamDict* lm = dict.subDict("mixture");
        const bool isH2O = lm && lm->found("H2O");
        if (!isH2O)
            throw std::runtime_error(
                "brae: `properties liquid` supports the H2O mixture only so far. The NSRDS correlation "
                "forms are general but each substance carries its own coefficient set, and an unvalidated "
                "one would run silently wrong rather than fail.");
        c.model          = ThermoModel::liquidH2O;
        c.internalEnergy = true;
        c.rhoThermoType  = true;    // heRhoThermo: rho is the STORED field, lagging the pressure solve
        // Cp/mu/kappa/rho are per-cell correlations on this path; the scalar members stay at their
        // defaults and are not read. relaxRho / rhoMin / rhoMax / pMin below still apply.
        readCommonSolutionControls(c, caseDir, fvSolutionIn);
        return c;
    }

    thermoRequire(
        type == "hePsiThermo" || (type == "heRhoThermo" && eos == "perfectGas"),
        "type",
        type,
        "hePsiThermo, or heRhoThermo with equationOfState perfectGas");
    thermoRequire(mixture == "pureMixture", "mixture", mixture, "pureMixture");
    thermoRequire(thermo == "hConst", "thermo", thermo, "hConst");
    thermoRequire(eos == "perfectGas", "equationOfState", eos, "perfectGas");
    thermoRequire(
        energy == "sensibleEnthalpy" || energy == "sensibleInternalEnergy",
        "energy",
        energy,
        "sensibleEnthalpy, sensibleInternalEnergy");
    c.internalEnergy = (energy == "sensibleInternalEnergy");
    c.rhoThermoType  = (type == "heRhoThermo");   // same arithmetic as hePsiThermo, DIFFERENT rho timing
    thermoRequire(
        transport == "sutherland" || transport == "const",
        "transport",
        transport,
        "sutherland, const");

    c.sutherland = (transport == "sutherland");

    const FoamDict* mix = dict.subDict("mixture");
    if (!mix)
    {
        throw std::runtime_error(
            "brae: constant/thermophysicalProperties has no mixture dictionary.");
    }

    // specie: molWeight [kg/kmol] -> specific gas constant
    const FoamDict* specie = mix->subDict("specie");
    const scalar W = specie ? specie->scalarOr("molWeight", 28.9) : 28.9;
    if (W <= 0.0)
    {
        throw std::runtime_error("brae: mixture.specie.molWeight must be positive.");
    }
    c.R = foamRR() / W;   // OF's DimensionedConstants when reachable, else thermoRRfallback

    // thermodynamics: hConst wants Cp, the heat of formation, and the reference point of the sensible
    // enthalpy. OF's hConstThermo reads Tref/Href here too (hConstThermo.C:40-41), defaulting Tref to
    // Tstd and Href to 0 -- and Hs = Cp*(T - Tref) + Href is the he the energy equation transports, so
    // silently assuming Tref = 0 shifts he by Cp*Tstd (~3.0e5 J/kg for air).
    c.Tref = foamTstd();
    const FoamDict* thermoDict = mix->subDict("thermodynamics");
    if (thermoDict)
    {
        c.Cp   = thermoDict->scalarOr("Cp", c.Cp);
        c.Hf   = thermoDict->scalarOr("Hf", c.Hf);
        c.Tref = thermoDict->scalarOr("Tref", c.Tref);
        c.Href = thermoDict->scalarOr("Href", c.Href);
    }

    // transport: sutherland wants (As, Ts), const wants (mu, Pr). Pr lives here in both cases.
    const FoamDict* transDict = mix->subDict("transport");
    if (transDict)
    {
        c.Pr = transDict->scalarOr("Pr", c.Pr);
        if (c.sutherland)
        {
            c.As = transDict->scalarOr("As", c.As);
            c.Ts = transDict->scalarOr("Ts", c.Ts);
        }
        else
        {
            c.mu0 = transDict->scalarOr("mu", c.mu0);
        }
    }

    if (c.Cp <= 0.0 || c.Pr <= 0.0)
    {
        throw std::runtime_error("brae: mixture Cp and Pr must be positive.");
    }

    // Cv = Cp - CpMCv, and perfectGas::CpMCv = R (OF HtoEthermo.H + perfectGasI.H). Guarded because a
    // molWeight/Cp pair giving Cp <= R is not a gas -- it would make gamma negative and the energy
    // equation quietly nonsense rather than obviously broken.
    // Gas-only, and skipped rather than merely harmless on the liquid path: Cv there is Cp(T) from the
    // correlation, not Cp - R, so this test would be reading gas defaults the liquid case never sets.
    if (c.model == ThermoModel::perfectGas && thermoCv(c) <= 0.0)
    {
        throw std::runtime_error(
            "brae: Cv = Cp - R must be positive, but the given Cp and molWeight make it <= 0. "
            "That is not a gas: it would make gamma negative and the sutherland kappa nonsense.");
    }

    readCommonSolutionControls(c, caseDir, fvSolutionIn);

    // Prt, exactly where OF looks for it: the turbulence model's own coeffs dict
    // (EddyDiffusivity::correctNut -> Prt_.readIfPresent(this->coeffDict())). Absent, OF's 1.0 stands.
    // OF names the dict after the model, e.g. RAS { RASModel kOmegaSST; kOmegaSSTCoeffs { Prt 0.85; } }.
    try
    {
        const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");
        for (const char* sub : {"RAS", "LES"})
        {
            const FoamDict* d = turbProps.subDict(sub);
            if (!d) continue;
            const std::string model = d->wordOr(std::string(sub) + "Model", "");
            // optionalSubDict (RASModel.C:72, LESModel likewise): without a `<model>Coeffs` sub-dictionary
            // OpenFOAM's coeffDict() IS the RAS/LES dictionary, so a flat `Prt 0.7;` reaches it.
            const FoamDict* coeffs = model.empty() ? nullptr : d->optionalSubDict(model + "Coeffs");
            if (coeffs) c.Prt = coeffs->scalarOr("Prt", c.Prt);
        }
    }
    catch (const std::exception&)
    {
        // No turbulenceProperties (a laminar case may omit it): alphat is zero anyway, so Prt is unused.
    }

    return c;
}

} // namespace brae
