#pragma once
// pimpleFoam's ddtSchemes.default and its fvSolution PIMPLE block, each read by one function that the run (gpuPimpleFoam.cu) and
// `brae check` both call, so that check's audit of the block (preflightAlgorithmControls) sees the reads the run makes; with the
// ddt names it takes for `brae capabilities`.
#include "device_ddt.cuh"
#include "foam_dict.cuh"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
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
