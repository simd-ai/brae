// interFoam's createFields -- see inter_case_cpp.cuh for the provenance and for the four things the
// order of this file encodes.
#include "inter_case_cpp.cuh"
#include "foam_field_reader.cuh"
#include "read_surface_field.cuh"
#include "scheme_parse.cuh"
#include <filesystem>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

// fvSchemes' divSchemes entry for div(rhoPhi,U). The shipped tutorials ask for `Gauss linearUpwind
// grad(U)` (24), `Gauss vanLeerV` (8), `Gauss upwind` (6), `Gauss linear` (3) and
// `Gauss limitedLinear 0.2` (1). Anything else -- and vanLeerV, which is not ported -- is refused by
// name rather than run as something similar.
DivScheme parseMomentumDiv(const std::string& entry, scalar& coeff)
{
    coeff = scalar(1);
    std::vector<std::string> tok;
    {
        std::string cur;
        for (char ch : entry)
        {
            if (std::isspace(static_cast<unsigned char>(ch))) { if (!cur.empty()) { tok.push_back(cur); cur.clear(); } }
            else cur += ch;
        }
        if (!cur.empty()) tok.push_back(cur);
    }
    if (tok.empty() || tok[0] != "Gauss")
        throw std::runtime_error(
            "brae interFoam: `div(rhoPhi,U) " + entry + "` -- only `Gauss <scheme>` is ported.");
    const std::string s = (tok.size() > 1) ? tok[1] : "";
    if (s == "upwind")        return DivScheme::upwind;
    if (s == "linear")        return DivScheme::linear;
    if (s == "LUST")          return DivScheme::LUST;
    if (s == "linearUpwind")  return DivScheme::linearUpwind;
    if (s == "linearUpwindV") return DivScheme::linearUpwindV;
    if (s == "limitedLinearV")
    {
        if (tok.size() > 2) coeff = std::stod(tok[2]);
        return DivScheme::limitedLinearV;
    }
    if (s == "limitedLinear")
    {
        if (tok.size() > 2) coeff = std::stod(tok[2]);
        return DivScheme::limitedLinear;    // refused downstream, by name, with the reason
    }
    throw std::runtime_error(
        "brae interFoam: `div(rhoPhi,U) " + entry + "` is not ported. `vanLeerV` (8 shipped tutorials) "
        "is the V-variant of the alpha limiter and limits along the direction of steepest change "
        "rather than per component -- it is a different scheme, not a variant of vanLeer.");
}

AlphaFluxScheme parseAlphaDiv(const std::string& entry, const char* key)
{
    const std::string e = entry;
    if (e.find("vanLeer") != std::string::npos)   return AlphaFluxScheme::vanLeer;
    if (e.find("upwind")  != std::string::npos)   return AlphaFluxScheme::upwind;
    if (e.find("interfaceCompression") != std::string::npos)
        return AlphaFluxScheme::interfaceCompression;   // refused downstream, by name
    if (e.find("linear")  != std::string::npos)   return AlphaFluxScheme::linear;
    throw std::runtime_error(
        std::string("brae interFoam: `") + key + " " + entry + "` is not ported.");
}

AlphaDdt parseAlphaDdt(const std::string& entry)
{
    if (entry.find("CrankNicolson") != std::string::npos) return AlphaDdt::CrankNicolson;
    if (entry.find("localEuler")    != std::string::npos) return AlphaDdt::localEuler;
    if (entry.find("Euler")         != std::string::npos) return AlphaDdt::Euler;
    return AlphaDdt::other;                              // refused by offCentringCoeff, by name
}

void refuseUnportedTurbulence(const std::string& caseDir)
{
    const std::string path = caseDir + "/constant/momentumTransport";
    const std::string alt  = caseDir + "/constant/turbulenceProperties";
    const std::string p = std::filesystem::exists(path) ? path
                        : (std::filesystem::exists(alt) ? alt : std::string());
    if (p.empty()) return;                                // no dictionary at all -> laminar
    const FoamDict d = readDict(p);
    const std::string sim = d.wordOr("simulationType", "laminar");
    if (sim != "laminar")
        throw std::runtime_error(
            "brae interFoam: constant/" + std::filesystem::path(p).filename().string()
            + " asks for simulationType `" + sim + "`. interFoam's turbulence is "
              "incompressibleInterPhaseTransportModel, which selects a MIXTURE model and hands it the "
              "blended rho -- not the single-phase model brae already has. Refused rather than run "
              "with the wrong density in the closure.");
}

}   // namespace


InterFields buildInterFields(const std::string&          caseDir,
                             const std::string&          startDir,
                             const PrimitiveMesh&        m,
                             const FvGeometry&           g,
                             const std::vector<FvPatch>& patches)
{
    InterFields f;
    const label nC = m.nCells();

    refuseUnportedTurbulence(caseDir);

    // --- the dictionaries ---------------------------------------------------------------------
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict fvSchemes   = readDict(caseDir + "/system/fvSchemes");

    f.mixture   = cpu::twoPhase::readTransportProperties(caseDir);
    // 1: the alpha field's NAME comes from `phases (water air)`, not from a convention.
    f.alphaName = "alpha." + f.mixture.phase1Name;
    f.interface = interfaceProps::readInterfaceCoeffs(fvSolution, f.alphaName, f.mixture.sigma);
    f.alphaCtl  = readAlphaControls(fvSolution, f.alphaName);
    f.mulesCtl  = f.alphaCtl.MULESCorr ? MULES::readControlsCorr(fvSolution, f.alphaName)
                                       : MULES::readControls(fvSolution, f.alphaName);
    f.timeCtl   = VoFTimeControls::read(controlDict);
    f.deltaT    = controlDict.scalarOr("deltaT", scalar(1e-3));

    {
        // fvSchemes' KEYS CONTAIN PARENTHESES AND COMMAS -- `div(rhoPhi,U)` is one token to OpenFOAM
        // and several to a dictionary tokenizer, so FoamDict cannot look them up. brae already reads
        // them as TEXT, scoped to the block (scheme_parse.cuh's fvSchemesBlock), and this uses the
        // same route rather than a second one that would parse the same file differently.
        const std::string all = readFvSchemesText(caseDir);
        std::string div = fvSchemesBlock(all, "divSchemes");
        if (div.empty())
            throw std::runtime_error("brae interFoam: fvSchemes has no divSchemes block.");
        auto entry = [&](const std::string& key, const std::string& dflt)
        {
            const std::size_t k = div.find(key);
            if (k == std::string::npos) return dflt;
            const std::size_t e = div.find(';', k);
            std::string st = div.substr(k + key.size(),
                                        e == std::string::npos ? std::string::npos : e - k - key.size());
            // trim
            std::size_t b = st.find_first_not_of(" \t\n\r");
            std::size_t f2 = st.find_last_not_of(" \t\n\r");
            return (b == std::string::npos) ? dflt : st.substr(b, f2 - b + 1);
        };
        const std::string uEntry = entry("div(rhoPhi,U)", "");
        if (uEntry.empty())
            throw std::runtime_error(
                "brae interFoam: fvSchemes names no `div(rhoPhi,U)`. interFoam's momentum convection "
                "has no default to fall back to -- every shipped tutorial names one.");
        f.divRhoPhiU   = parseMomentumDiv(uEntry, f.divRhoPhiUCoeff);
        f.divPhiAlpha  = parseAlphaDiv(entry("div(phi,alpha)",   "Gauss vanLeer"), "div(phi,alpha)");
        f.divPhirbAlpha= parseAlphaDiv(entry("div(phirb,alpha)", "Gauss linear"),  "div(phirb,alpha)");

        const std::string ddtBlock = fvSchemesBlock(all, "ddtSchemes");
        auto ddtEntry = [&](const std::string& key, const std::string& dflt)
        {
            const std::size_t k = ddtBlock.find(key);
            if (k == std::string::npos) return dflt;
            const std::size_t e = ddtBlock.find(';', k);
            std::string st = ddtBlock.substr(k + key.size(),
                                             e == std::string::npos ? std::string::npos : e - k - key.size());
            std::size_t b = st.find_first_not_of(" \t\n\r");
            std::size_t f2 = st.find_last_not_of(" \t\n\r");
            return (b == std::string::npos) ? dflt : st.substr(b, f2 - b + 1);
        };
        const std::string dflt = ddtEntry("default", "Euler");
        f.ddtAlpha = parseAlphaDdt(ddtEntry("ddt(alpha)", dflt));
        f.ddtU     = (dflt.find("Euler") != std::string::npos && dflt.find("localEuler") == std::string::npos)
                   ? DdtScheme::Euler
                   : (dflt.find("CrankNicolson") != std::string::npos ? DdtScheme::CrankNicolson
                   : (dflt.find("localEuler")    != std::string::npos ? DdtScheme::localEuler
                   : (dflt.find("backward")      != std::string::npos ? DdtScheme::backward
                                                                      : DdtScheme::steadyState)));
    }

    // fvSolution's PIMPLE block -- see InterFields::pimple for why momentumPredictor is read rather
    // than assumed.
    {
        const FoamDict* pim = fvSolution.subDict("PIMPLE");
        if (!pim)
            throw std::runtime_error(
                "brae interFoam: fvSolution has no PIMPLE block. interFoam is a PIMPLE solver and every "
                "shipped tutorial carries one; there is no default to fall back to.");
        f.pimple.nOuterCorrectors = static_cast<label>(pim->scalarOr("nOuterCorrectors", scalar(1)));
        f.pimple.nCorrectors      = static_cast<label>(pim->scalarOr("nCorrectors", scalar(1)));
        f.nNonOrthogonalCorrectors =
            static_cast<label>(pim->scalarOr("nNonOrthogonalCorrectors", scalar(0)));
        const std::string mp = pim->wordOr("momentumPredictor", "yes");
        f.momentumPredictorOn = !(mp == "no" || mp == "false" || mp == "off" || mp == "0");
        f.pimple.frozenFlow = false;
        f.pimple.turbOnFinalIterOnly = true;
    }

    // relaxationFactors/equations -- see InterFields::relaxEquationU.
    {
        const FoamDict* rf = fvSolution.subDict("relaxationFactors");
        const FoamDict* eq = rf ? rf->subDict("equations") : nullptr;
        if (eq)
        {
            // OpenFOAM resolves the name through the same regex machinery fvSolution uses everywhere;
            // `".*" 1` matches U, and an explicit `U` entry wins over a `default`.
            const scalar u   = eq->scalarOr("U", scalar(-1));
            const scalar any = eq->scalarOr("\".*\"", scalar(-1));
            const scalar def = eq->scalarOr("default", scalar(-1));
            const scalar v = (u >= 0) ? u : ((any >= 0) ? any : def);
            if (v >= 0) { f.relaxEquationU = true; f.relaxU = v; }
        }
    }

    // --- gravity ------------------------------------------------------------------------------
    f.g          = readGravity(caseDir);
    f.hRef       = readHRef(caseDir);
    f.ghRefValue = ghRef(f.g, f.hRef);

    // --- the read fields ----------------------------------------------------------------------
    f.alpha1 = buildField<scalar>(readField<scalar>(startDir + "/" + f.alphaName), patches, nC);
    f.U      = buildField<vector>(readField<vector>(startDir + "/U"),              patches, nC);
    f.p_rgh  = buildField<scalar>(readField<scalar>(startDir + "/p_rgh"),          patches, nC);
    f.alpha1.evaluateBoundary();
    f.U.evaluateBoundary();
    f.p_rgh.evaluateBoundary();

    // 3: createAlphaFluxes.H READS phi when it is there. A restart continues from the WRITTEN flux,
    // which is not fvc::flux(U) -- U and phi disagree by the continuity error the pressure corrector
    // has just driven down, and recomputing it discards exactly that.
    f.phi = readPhiIfPresent(startDir, patches, m.nInternalFaces(),
                             fvc::flux(f.U, m, g, patches), &f.phiWasRead);

    // --- the mixture --------------------------------------------------------------------------
    // 2: rho from the RAW alpha, mu and nu from the CLAMPED one -- see two_phase_mixture_cpp.cuh.
    f.alpha2.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c) f.alpha2[c] = scalar(1) - f.alpha1.internal[c];
    cpu::twoPhase::mixtureRho(f.alpha1.internal, f.alpha2, f.mixture.phases, f.rho);
    cpu::twoPhase::mixtureMu (f.alpha1.internal, f.mixture.phases, f.mu);
    cpu::twoPhase::mixtureNu (f.alpha1.internal, f.mu, f.mixture.phases, f.nu);

    // --- gh, ghf and p ------------------------------------------------------------------------
    ghField(f.g, f.ghRefValue, g.C(), f.gh);
    {
        std::vector<vector> Cf(static_cast<std::size_t>(m.nInternalFaces()));
        for (label i = 0; i < m.nInternalFaces(); ++i) Cf[i] = g.Cf()[i];
        ghField(f.g, f.ghRefValue, Cf, f.ghfInternal);
    }
    staticPressure(f.p_rgh.internal, f.rho, f.gh, f.p);

    // rhoPhi starts at the mass flux implied by the read alpha and phi: alphaEqn overwrites it every
    // step, but UEqn would read it on the very first outer iteration if it did not exist.
    {
        SurfaceScalarField alphaPhi;
        fluxWithScheme(f.phi, f.alpha1, f.divPhiAlpha, m, g, patches, alphaPhi);
        massFlux(alphaPhi, f.phi, f.mixture.phases.rho1, f.mixture.phases.rho2, f.rhoPhi);
    }

    return f;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
