#pragma once
// pimpleFoam's ddtSchemes.default and its fvSolution PIMPLE block, each read by one function that the run (gpuPimpleFoam.cu) and
// `brae check` both call, so that check's audit of the block (preflightAlgorithmControls) sees the reads the run makes; with the
// ddt names it takes for `brae capabilities`.
#include "device_ddt.cuh"
#include "foam_dict.cuh"
#include "scheme_parse.cuh"   // ddtSchemeFor: a named ddt entry resolved as OpenFOAM's schemesLookup resolves it
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// the ddtSchemes.default words parsePimpleDdtScheme takes
inline const std::vector<std::string>& pimpleDdtSchemeNames()
{
    static const std::vector<std::string> n = {"Euler", "backward", "CrankNicolson [<ocCoeff>]", "bounded Euler"};
    return n;
}

// fvSchemes holds multi-token scheme values + $-vars (not a plain key/value dict), so text-parse ddtSchemes.default.
inline DdtScheme parsePimpleDdtScheme(const std::string& fvSchemesPath, scalar& ocCoeff)
{
    ocCoeff = 1.0;                                    // CrankNicolson off-centring; default = pure CN (1)
    std::ifstream f(fvSchemesPath);
    if (!f) throw std::runtime_error("cannot read " + fvSchemesPath);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::size_t blk = text.find("ddtSchemes");
    if (blk == std::string::npos)
        throw std::runtime_error("fvSchemes has no ddtSchemes block (transient pimpleFoam needs Euler|backward).");
    std::size_t d = text.find("default", blk);
    if (d == std::string::npos) throw std::runtime_error("fvSchemes ddtSchemes has no 'default' entry.");
    d += 7;
    auto nextWord = [&]()
    {
        while (d < text.size() && std::isspace((unsigned char)text[d])) ++d;
        std::string w;
        while (d < text.size() && !std::isspace((unsigned char)text[d]) && text[d] != ';') w += text[d++];
        return w;
    };
    std::string w = nextWord();
    if (w == "steadyState")
        throw std::runtime_error("fvSchemes ddtSchemes.default = steadyState -> use the steady solver 'brae'.");
    if (w == "bounded")
    {
        // OpenFOAM's boundedDdtScheme wraps the scheme it names and subtracts fvm::Sp(fvc::ddt(one)), which is zero for this
        // incompressible solver: `bounded Euler` is Euler. The word after `bounded` was not read, so `bounded backward` ran
        // Euler under its name; any scheme but Euler after it is refused now.
        const std::string inner = nextWord();
        if (inner != "Euler")
            throw std::runtime_error("fvSchemes ddtSchemes.default 'bounded " + inner + "': only 'bounded Euler' is implemented "
                                     "(it is Euler for this incompressible solver); refused rather than run Euler under its name.");
        return DdtScheme::Euler;
    }
    if (w == "backward") return DdtScheme::backward;
    if (w == "Euler")    return DdtScheme::Euler;
    if (w == "CrankNicolson")
    {
        // read the off-centring coefficient (OF "CrankNicolson <ocCoeff>", oc in [0,1]); absent -> pure CN (1).
        const std::string oc = nextWord();
        if (!oc.empty()) { try { ocCoeff = std::stod(oc); } catch (...) {} }
        return DdtScheme::CrankNicolson;
    }
    throw std::runtime_error("fvSchemes ddtSchemes.default '" + w + "' unsupported (Euler|backward|CrankNicolson|steadyState).");
}

// One ddtSchemes entry's value, as parsePimpleDdtScheme reads `default`: false when it is not one of the forms that reader takes.
inline bool parsePimpleDdtEntry(
    const std::string& entry,
    DdtScheme& scheme,
    scalar& ocCoeff)
{
    std::istringstream in(entry);
    std::vector<std::string> w;
    for (std::string t; in >> t; )
    {
        w.push_back(t);
    }
    ocCoeff = 1.0;
    if (w.size() == 1 && w[0] == "Euler")
    {
        scheme = DdtScheme::Euler;
        return true;
    }
    if (w.size() == 2 && w[0] == "bounded" && w[1] == "Euler")
    {
        scheme = DdtScheme::Euler;
        return true;
    }
    if (w.size() == 1 && w[0] == "backward")
    {
        scheme = DdtScheme::backward;
        return true;
    }
    if (!w.empty() && w.size() <= 2 && w[0] == "CrankNicolson")
    {
        if (w.size() == 2)
        {
            try
            {
                ocCoeff = std::stod(w[1]);
            }
            catch (...)
            {
                return false;
            }
        }
        scheme = DdtScheme::CrankNicolson;
        return true;
    }
    return false;
}

// Named ddt entries. OpenFOAM looks each fvm::ddt up by its own name first -- ddt(U) for the momentum (UEqn.H, and fvc::ddtCorr in
// pEqn.H), ddt(k), ddt(epsilon), ddt(omega), ddt(nuTilda) ... for the turbulence fields (fvm::ddt(alpha, rho, k) is fvm::ddt(k) for
// an incompressible model, fvmDdt.C) -- and falls back to `default` only when the dictionary has no entry for that name, literal or
// regex (schemesLookupDetail.C:76-89). brae applies ddtSchemes.default to every field it integrates, so an entry that resolves a
// name brae solves to another scheme would run the default under its name. `names` are the ddt names of the fields this run
// integrates; an entry for a field the run does not solve is not looked up by OpenFOAM either and is left alone.
//   BRAE_CONTROL_DDT_NAMED_UNREAD=1: a gate's CONTROL, deliberately wrong -- the named entries are not looked at, as before.
inline void refuseNamedDdtSchemes(
    const std::string& caseDir,
    const std::vector<std::string>& names,
    DdtScheme defaultScheme,
    scalar defaultOcCoeff)
{
    if (std::getenv("BRAE_CONTROL_DDT_NAMED_UNREAD") != nullptr)
    {
        std::printf("  *** CONTROL MODE: named ddtSchemes entries are not checked. This run is deliberately wrong. ***\n");
        return;
    }
    for (const std::string& name : names)
    {
        const std::string entry = ddtSchemeFor(caseDir, name);
        DdtScheme s = DdtScheme::Euler;
        scalar oc = 1.0;
        const bool known = parsePimpleDdtEntry(entry, s, oc);
        if (known && s == defaultScheme && (s != DdtScheme::CrankNicolson || oc == defaultOcCoeff))
        {
            continue;
        }
        throw std::runtime_error(
            "brae: system/fvSchemes ddtSchemes resolves `" + name + "` to `" + entry + "`, which is not the `default` entry "
            "brae applies. OpenFOAM integrates that field with the scheme its own entry names (schemesLookupDetail.C:76-89); "
            "brae integrates every field with ddtSchemes.default, so it would run the default under this name. Remove the "
            "entry, or make it the same as `default`: per-field ddt schemes are not ported.");
    }
}

// The fvSolution PIMPLE controls pimpleFoam runs, each with OpenFOAM's default (pimpleControl::read, createDyMControls.H). Read
// together, before preflightAlgorithmControls names what is left. pRefCell and pRefValue are not here: they are read only where the
// pressure needs a reference, where OpenFOAM's setRefCell reads them, and the audit treats them as conditional.
struct PimpleAlgorithm
{
    struct ResidualControl
    {
        std::string field;
        scalar absTol;
        scalar relTol;
    };
    int nOuter = 1;
    int nCorr = 1;
    int nNonOrth = 0;
    bool ddtCorr = true;
    bool momentumPredictor = true;
    bool turbOnFinalIterOnly = true;
    bool solveFlow = true;
    bool simpleRho = false;
    bool moveMeshOuterCorrectors = false;
    bool correctPhi = false;
    std::vector<ResidualControl> residualControl;
};

// `dynamicMesh`: a motion solver is active. correctPhi's default is mesh.dynamic() (createDyMControls.H), and it is applied only
// on a moving mesh.
inline PimpleAlgorithm readPimpleAlgorithm(
    const FoamDict& fvSolution,
    bool dynamicMesh)
{
    PimpleAlgorithm a;
    const FoamDict* pimple = fvSolution.subDict("PIMPLE");
    if (!pimple)
    {
        a.correctPhi = dynamicMesh;
        return a;
    }
    auto yes = [](const std::string& v)
    {
        return !(v == "no" || v == "false" || v == "off" || v == "0");
    };
    a.nOuter = std::max(1, pimple->intOr("nOuterCorrectors", 1));
    a.nCorr = std::max(1, pimple->intOr("nCorrectors", 1));
    // OF pimpleControl.C:55, default true. pEqn.H guards the whole fvc::ddtCorr term with it.
    a.ddtCorr = yes(pimple->wordOr("ddtCorr", "yes"));
    a.nNonOrth = pimple->intOr("nNonOrthogonalCorrectors", 0);
    // OF assembles UEqn either way (rAU/HbyA come from it) and only skips the SOLVE. Default true, as pimpleControl's.
    a.momentumPredictor = yes(pimple->wordOr("momentumPredictor", "yes"));
    // the PIMPLE loop contract (OF pimpleControl::read): turbOnFinalIterOnly TRUE, solveFlow TRUE, SIMPLErho FALSE
    a.turbOnFinalIterOnly = yes(pimple->wordOr("turbOnFinalIterOnly", "yes"));
    a.solveFlow = yes(pimple->wordOr("solveFlow", "yes"));
    a.simpleRho = yes(pimple->wordOr("SIMPLErho", "no"));
    a.moveMeshOuterCorrectors = yes(pimple->wordOr("moveMeshOuterCorrectors", "no"));
    // residualControl: outer-loop convergence, parsed as OF does -- each entry is a field name whose sub-dict carries `tolerance`
    // (absolute) and/or `relTol`
    if (const FoamDict* rc = pimple->subDict("residualControl"))
        for (const auto& e : rc->subs)
            a.residualControl.push_back({e.first, e.second.scalarOr("tolerance", 0.0), e.second.scalarOr("relTol", 0.0)});
    a.correctPhi = dynamicMesh && yes(pimple->wordOr("correctPhi", dynamicMesh ? "yes" : "no"));
    return a;
}

}  // namespace brae
