// DISPATCH for the rebuilt simpleFoam -- see simpleFoamV2.cuh for why the guard exists.
#include "simpleFoamV2.cuh"
#include "simpleFoam.cuh"
#include "createFields_cpp.cuh"
#include "frozen_bc_guard.cuh"
#include "simpleControl_cpp.cuh"
#include "linearViscousStress_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "solution_directions.cuh"
#include "start_time.cuh"     // OF Time::setControls startFrom (Time.C:146-193)
#include "write_control.cuh"  // OF Time::timeName (Time.C:721-728)   // polyMesh::solutionD(): which U components OpenFOAM solves
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "scheme_parse.cuh"   // readFvSchemesText: fusedGauss -> Gauss
#include "fvc.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"
#include "device_blas.cuh"        // deviceCopy -- the nutLowRe wall zeroing keeps the other faces
#include "turbulence_setup.cuh"   // selectNutWall: the nut wall function the 0/nut BC chooses
#include "device_komega_sst.cuh"    // deviceKOmegaSSTCorrect building blocks + KOmegaSSTCoeffs
#include "komega_sst_coeffs.cuh"
#include "kOmegaSSTLM_cpp.cuh"   // the LM coefficient reader; the reference this port was gated against
#include "realizableKE_cpp.cuh"   // RealizableKECoeffs + readRealizableKECoeffs
#include "SpalartAllmaras_cpp.cuh"
#include "spalart_coeffs.cuh"
#include "MRF_cpp.cuh"
#include "device_MRF.cuh"
#include "mrf_read.cuh"          // readCellZones
#include "fv_options.cuh"        // readFvOptions -> RotorDiskParams
#include "rotor_disk.cuh"
#include "fvOptions_cpp.cuh"       // the fvOptions framework + explicitPorositySource    // readKOmegaSSTCoeffs (RAS.kOmegaSSTCoeffs, OF defaults when absent)
#include "cell_wall_dist.cuh"       // cellWallDist: kOmegaSST's F1/F2 need y at every CELL, not just walls
#include "near_wall_dist.cuh"
#include "solver_dispatch.cuh"   // readDdtSchemeWord: the same steady/transient test the dispatcher uses

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <regex>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace brae {
namespace gpu {

namespace {

bool fileExists(const std::string& p)
{
    return std::filesystem::exists(p) || std::filesystem::exists(p + ".gz");
}

// The div SCHEME the case asks for on div(phi,U). Returned as the selectable keyword, i.e. the last word
// that is not `Gauss`, `bounded` or a numeric coefficient -- the same rule ofscan's case layer applies.
// Does div(phi,U) carry the `bounded` prefix? OpenFOAM's `bounded Gauss <scheme>` adds
//     - fvm::Sp(fvc::div(phi), U)
// to the momentum equation (boundedConvectionScheme). It vanishes at convergence, where div(phi) -> 0,
// which is exactly why dropping it is invisible in a converged comparison and very visible in the
// approach to it: it is what keeps the equation diagonally dominant while the flux is not yet
// conservative. The rebuilt UEqn does not implement it.
bool divUBounded(const std::string& caseDir);

// The div(phi,U) ENTRY, as OpenFOAM's schemesLookup resolves it: the explicit entry when the dictionary
// has one, else `default` (schemesLookupDetail.C:76-88). Shared by the scheme lookup and the gradient
// lookup below so the two cannot drift apart -- the gradient lookup used to search the WHOLE divSchemes
// block for `linearUpwind <name>`, which takes whichever statement comes first: a case with
// `div(phi,k) bounded Gauss linearUpwind grad(k)` written ABOVE `div(phi,U) bounded Gauss linearUpwind
// grad(U)` was measured reporting grad(k)'s coefficient for div(phi,U).
std::string divUEntry(const std::string& text)
{
    const std::size_t blk = text.find("divSchemes");
    if (blk == std::string::npos) return "";
    const std::size_t open = text.find('{', blk);
    const std::size_t close = text.find('}', open == std::string::npos ? blk : open);
    if (open == std::string::npos) return "";
    const std::string blkText = text.substr(open, (close == std::string::npos ? text.size() : close) - open);
    static const std::regex re(R"(div\(phi,U\)\s*([^;]*);)");
    std::smatch mm;
    if (std::regex_search(blkText, mm, re)) return mm[1].str();
    static const std::regex rd(R"(default\s+([^;]*);)");
    if (std::regex_search(blkText, mm, rd)) return mm[1].str();
    return "";
}


std::string divUScheme(const std::string& caseDir)
{
    std::string text;
    try { text = readFvSchemesText(caseDir); } catch (...) { return ""; }
    const std::string entry = divUEntry(text);
    std::string last;
    std::regex tok(R"([^\s]+)");
    for (std::sregex_iterator it(entry.begin(), entry.end(), tok), e; it != e; ++it)
    {
        const std::string w = it->str();
        if (w == "Gauss" || w == "bounded" || w == "none") continue;
        if (std::regex_match(w, std::regex(R"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"))) continue;
        last = w;
        break;      // the FIRST such word is the scheme; anything after is its coefficient
    }
    return last;
}

// div(phi,k) / div(phi,epsilon) / div(phi,omega). The turbulence scalars get their OWN entry, and the
// SST tutorials point it at `bounded Gauss limitedLinear 1` while pitzDaily's kEpsilon asks for plain
// `Gauss upwind`. That is a different MATRIX, not a looser tolerance: measured against OpenFOAM's own
// initial residual on pitzDailySST, running upwind where the case asks for limitedLinear is 8.3x out on
// omega and 82x on k. The entries are usually written through a `$turbulence` macro, which
// readFileExpanded resolves before this sees them.
struct TurbDivScheme
{
    std::string scheme;          // the scheme word: `upwind`, `limitedLinear`, ...
    scalar      coeff = 1.0;     // its coefficient, `limitedLinear 1` -> 1
    bool        bounded = false; // the `bounded` prefix -> -fvm::Sp(fvc::div(phi), var)
    bool        found = false;
};

TurbDivScheme divTurbScheme(const std::string& caseDir, const std::string& key)
{
    TurbDivScheme out;
    std::string text;
    try { text = readFvSchemesText(caseDir); } catch (...) { return out; }
    const std::size_t blk = text.find("divSchemes");
    if (blk == std::string::npos) return out;
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return out;
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);

    std::string esc;
    for (char c : key)
    {
        if (std::strchr("().[]{}*+?^$|\\", c)) esc += '\\';
        esc += c;
    }
    std::smatch mm;
    std::string entry;
    if (std::regex_search(b, mm, std::regex(esc + R"(\s*([^;]*);)"))) entry = mm[1].str();
    else if (std::regex_search(b, mm, std::regex(R"(default\s+([^;]*);)"))) entry = mm[1].str();
    if (entry.empty()) return out;

    out.found = true;
    static const std::regex tok(R"([^\s]+)");
    for (std::sregex_iterator it(entry.begin(), entry.end(), tok), e; it != e; ++it)
    {
        const std::string w = it->str();
        if (w == "bounded")
        {
            out.bounded = true;
            continue;
        }
        if (w == "Gauss" || w == "none") continue;
        if (std::regex_match(w, std::regex(R"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)")))
        {
            if (!out.scheme.empty()) out.coeff = std::atof(w.c_str());
            continue;
        }
        if (out.scheme.empty()) out.scheme = w;
    }
    return out;
}

// The gradient `linearUpwind` NAMES, resolved through gradSchemes -- and whether it is one we compute.
//
// `div(phi,U) bounded Gauss linearUpwind grad(U);` does not mean "use fvc::grad". linearUpwind's
// constructor reads the word after the scheme (linearUpwind.H, gradSchemeName_(schemeData)) and looks it
// up with mesh.gradScheme(name), which falls back to gradSchemes `default`. brae computes a plain Gauss
// linear gradient; a case naming cellLimited or leastSquares there would get a DIFFERENT correction,
// silently, and this correction does NOT vanish at convergence -- so the answer would simply be wrong.
// Returns the offending scheme word, or empty when the resolved scheme is one brae computes: `Gauss
// linear`, or `cellLimited Gauss linear <k>` -- the limited form is now implemented on both paths
// (cellLimitedGrad_cpp.cu, deviceCellLimitGrad), and `limitK` receives its k when it is.
std::string gradSchemeUnsupported(const std::string& text, const std::string& gradName,
                                 scalar* limitK);   // defined below, shared by both lookups

std::string linearUpwindGradUnsupported(const std::string& caseDir, scalar* limitK)
{
    if (limitK) *limitK = 0.0;
    std::string text;
    try { text = readFvSchemesText(caseDir); } catch (...) { return ""; }

    // 1. the gradient the SCHEME names -- the word straight after it on div(phi,U)'s own stream.
    //
    // This was `std::regex(R"(linearUpwind\s+([^\s;]+))")` searched over the whole divSchemes block, and
    // it had three holes. It cannot match `linearUpwindV grad(U)` (the next character is `V`, not
    // whitespace) or `LUST grad(U)` at all, so both fell back to `default`: measured on staged pitzDaily
    // copies sharing `grad(U) cellLimited Gauss linear 1`, the linearUpwind copy reported the case's
    // coefficient and the linearUpwindV and LUST copies reported nothing, character-for-character
    // identical to a case with no entry. And searching the block rather than the entry takes whichever
    // statement comes first. All three schemes read the name the same way -- linearUpwind.H:105/:117,
    // linearUpwindV.H:117-135, and LUST.H, whose ctor hands the stream straight to linearUpwind -- and
    // no other scheme brae implements reads one at all, so anything else stops here with no gradient
    // rather than reporting `default` under a scheme name that never asked for one.
    std::string gradName;
    {
        const std::string entry = divUEntry(text);
        static const std::regex tokRe(R"([^\s]+)");
        static const std::regex numRe(R"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)");
        std::string scheme;
        for (std::sregex_iterator it(entry.begin(), entry.end(), tokRe), e; it != e; ++it)
        {
            const std::string w = it->str();
            if (scheme.empty())
            {
                if (w == "Gauss" || w == "bounded" || w == "none") continue;
                if (std::regex_match(w, numRe)) continue;
                scheme = w;
                continue;
            }
            gradName = w;      // the word straight after the scheme is its gradSchemeName_
            break;
        }
        if (scheme != "linearUpwind" && scheme != "linearUpwindV" && scheme != "LUST") return "";
        if (gradName.empty()) gradName = "default";
    }

    // 2-3. that name's entry, classified. Shared with gradUSchemeUnsupported below.
    return gradSchemeUnsupported(text, gradName, limitK);
}


// Steps 2 and 3 of the lookup above, on their own so the `grad(U)` entry can be resolved by the same
// code: OpenFOAM's schemesLookup returns the named entry if the dictionary has it and falls back to
// `default` otherwise (schemesLookupDetail.C:76-88), and the accepted forms are `Gauss linear` and
// `cellLimited Gauss linear <k>`.
std::string gradSchemeUnsupported(const std::string& text, const std::string& gradName, scalar* limitK)
{
    if (limitK) *limitK = 0.0;
    const std::size_t gblk = text.find("gradSchemes");
    if (gblk == std::string::npos) return "";          // no block -> OpenFOAM's default is Gauss linear
    const std::size_t open = text.find('{', gblk);
    if (open == std::string::npos) return "";
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);

    std::string entry;
    {
        // The name contains parentheses (`grad(U)`), so every regex metacharacter in it is escaped rather
        // than pasted in raw -- `grad(U)` as a pattern would match the bare word `gradU`.
        std::string esc;
        for (char c : gradName)
        {
            if (std::strchr("().[]{}*+?^$|\\", c)) esc += '\\';
            esc += c;
        }
        std::smatch mm;
        if (std::regex_search(b, mm, std::regex(esc + R"(\s+([^;]*);)"))) entry = mm[1].str();
        else if (std::regex_search(b, mm, std::regex(R"(default\s+([^;]*);)"))) entry = mm[1].str();
    }
    if (entry.empty()) return "";

    // 3. accept `Gauss linear` and `cellLimited Gauss linear <k>`; anything else is named and refused.
    static const std::regex tok(R"([^\s]+)");
    static const std::regex num(R"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)");
    bool limited = false;
    for (std::sregex_iterator it(entry.begin(), entry.end(), tok), e; it != e; ++it)
    {
        const std::string w = it->str();
        if (w == "Gauss" || w == "linear") continue;
        if (w == "cellLimited")
        {
            limited = true;
            continue;
        }
        // The coefficient, and ONLY when a limiter claimed it -- a bare number after `Gauss linear`
        // would belong to some other scheme this does not implement.
        if (limited && std::regex_match(w, num))
        {
            if (limitK) *limitK = std::atof(w.c_str());
            continue;
        }
        return w;
    }
    // `cellLimited Gauss linear` with no number is k = 1 in OpenFOAM's stream reading.
    if (limited && limitK && *limitK == 0.0) *limitK = 1.0;
    return "";
}


// The `grad(U)` ENTRY's own limiter -- a DIFFERENT lookup from linearUpwindGradUnsupported's, which
// resolves the word linearUpwind names. On validation/rotorDisk the two differ (`div(phi,U) ...
// linearUpwind unlimited` beside `grad(U) $limited`), so neither can stand in for the other.
//
// This is the gradient every fvc::grad(U) in the solver takes: the closures' strain and production
// (kOmegaSSTBase.C:522 tgradU, kEpsilon.C:237, SpalartAllmarasBase.C:461) and divDevReff's explicit
// dev2 term (linearViscousStress.C:114). All four were running the UNLIMITED Gauss gradient while the
// driver printed the case's cellLimited coefficient. Measured on validation/windAroundBuildingsBox at
// its own converged t=400, limited against unlimited through OpenFOAM's own postProcess grad(U):
// |gradU| relative L2 3.21e+01, the SST/kEpsilon production term GbyNu peak 1.05e-03 against 5.10e-01
// (487x), SpalartAllmaras's Omega 22x, and 34% of the cells carry a clipped gradient.
std::string gradUSchemeUnsupported(const std::string& caseDir, scalar* limitK)
{
    if (limitK) *limitK = 0.0;
    std::string text;
    try { text = readFvSchemesText(caseDir); } catch (...) { return ""; }
    return gradSchemeUnsupported(text, "grad(U)", limitK);
}

// The numeric coefficient a limited scheme carries: the `1` of `limitedLinear 1`. OpenFOAM reads it off
// the scheme stream, so it is the first number after the scheme word. It is NOT cosmetic: twoByk = 2/k
// scales the limiter, and `limitedLinear 0.2` is a materially different scheme from `limitedLinear 1`.
scalar divUSchemeCoeff(const std::string& caseDir, scalar def)
{
    std::string text;
    try { text = readFvSchemesText(caseDir); } catch (...) { return def; }
    const std::size_t blk = text.find("divSchemes");
    if (blk == std::string::npos) return def;
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return def;
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);
    static const std::regex re(R"(div\(phi,U\)\s*([^;]*);)");
    std::smatch mm;
    std::string entry;
    if (std::regex_search(b, mm, re)) entry = mm[1].str();
    else
    {
        static const std::regex rd(R"(default\s+([^;]*);)");
        if (std::regex_search(b, mm, rd)) entry = mm[1].str();
    }
    static const std::regex num(R"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)");
    std::smatch nm;
    if (std::regex_search(entry, nm, num)) return std::atof(nm[0].str().c_str());
    return def;
}

bool divUBounded(const std::string& caseDir)
{
    std::string text;
    try { text = readFvSchemesText(caseDir); } catch (...) { return false; }
    const std::size_t blk = text.find("divSchemes");
    if (blk == std::string::npos) return false;
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return false;
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);
    static const std::regex re(R"(div\(phi,U\)\s*([^;]*);)");
    std::smatch mm;
    std::string entry;
    if (std::regex_search(b, mm, re)) entry = mm[1].str();
    else
    {
        static const std::regex rd(R"(default\s+([^;]*);)");
        if (std::regex_search(b, mm, rd)) entry = mm[1].str();
    }
    return entry.find("bounded") != std::string::npos;
}

// Does the LAPLACIAN carry the non-orthogonal correction?
//
// The laplacian's OWN scheme decides. `laplacianSchemes { default Gauss linear orthogonal; }` builds an
// orthogonal laplacian whatever `snGradSchemes` says -- snGradSchemes governs EXPLICIT fvc::snGrad, which
// in simpleFoam appears only in the SIMPLEC branch (already refused). An earlier version of this check
// blocked on either block and so refused pitzDailyTurb, a case whose laplacian is orthogonal and which
// the rebuilt path handles exactly.
//
} // anonymous namespace (laplacianScheme has EXTERNAL linkage -- declared in simpleFoamV2.cuh)

// LaplacianScheme is declared in simpleFoamV2.cuh -- hoisted so the parse is unit-testable
// (tests/test_v2_laplacian_parse): the `limited 0` conflation below survived precisely because
// nothing outside this file could call it.

// Both halves of `corrected` are now implemented on both paths (fvm.cuh; UEqn.cu; pEqn.cu), so this READS
// the scheme instead of refusing it. OpenFOAM's default when the word is absent IS corrected, so an absent
// block returns true.
//
// `limited <k>` is limitedSnGrad: it caps the correction against the ORTHOGONAL part of the same snGrad,
//     limiter = min( k*|orth| / ((1 - k)*|corr| + SMALL), 1 )
// so `limited 1` is exactly `corrected` and `limited 0` is `uncorrected`. Implemented in
// fvm::laplacianCorrFlux and gated by tests/limitedsngrad_vs_openfoam.sh; the coefficient is read out
// here rather than refused, because running the UNCAPPED correction under a capped name applies a larger
// correction than the case asked for and does not vanish at convergence.
LaplacianScheme laplacianScheme(const std::string& caseDir)
{
    LaplacianScheme r;
    std::string text;
    try { text = readFvSchemesText(caseDir); } catch (...) { r.corrected = false; return r; }
    const std::size_t blk = text.find("laplacianSchemes");
    if (blk == std::string::npos) return r;                 // absent -> OpenFOAM default is `corrected`
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return r;
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);

    if (b.find("limited") != std::string::npos)
    {
        // OpenFOAM writes it BOTH ways: `limited <k>` and `limited <scheme> <k>` -- turbineSiting has
        // `Gauss linear limited corrected 0.33`, where the coefficient follows the scheme word. Reading
        // only the token straight after `limited` finds "corrected" there and refuses a case brae can run.
        const std::size_t k = b.find("limited");
        std::istringstream ls(b.substr(k + 7));
        scalar coeff = -1.0;
        std::string tok;
        bool got = false;
        while (ls >> tok)
        {
            if (tok.empty()) continue;
            if (tok.back() == ';') tok.pop_back();
            try
            {
                std::size_t used = 0;
                const scalar v = std::stod(tok, &used);
                if (used == tok.size())
                {
                    coeff = v;
                    got = true;
                    break;
                }
            }
            catch (...) { /* a scheme word, e.g. `corrected` -- keep looking */ }
            if (tok == "corrected" || tok == "uncorrected" || tok == "orthogonal") continue;
            break;   // anything else is a scheme this does not implement
        }
        if (!got || coeff < 0.0 || coeff > 1.0)
            r.unsupported = "limited";
        else if (coeff <= 1e-12)
            r.corrected = false;    // `limited 0`: the limiter is identically zero -- `uncorrected`
        else if (std::fabs(coeff - 1.0) > 1e-12)
            r.limitCoeff = coeff;   // `limited 1` needs no limiter at all
        return r;
    }
    if (b.find("uncorrected") != std::string::npos) { r.corrected = false; return r; }
    if (b.find("orthogonal") != std::string::npos && b.find("nonOrthogonal") == std::string::npos)
    { r.corrected = false; return r; }
    return r;   // `corrected` or unspecified
}

namespace {   // reopened

bool switchOn(const FoamDict& d, const std::string& key, bool def)
{
    const auto* v = d.find(key);
    if (!v || v->empty()) return def;
    const std::string& s = v->back();
    if (s == "true" || s == "on" || s == "yes" || s == "y" || s == "1") return true;
    if (s == "false" || s == "off" || s == "no" || s == "n" || s == "0") return false;
    return def;
}

} // namespace


EnvelopeReport simpleFoamV2Envelope(const std::string& caseDir)
{
    EnvelopeReport r;

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict turbProps  = readDict(caseDir + "/constant/turbulenceProperties");

    // --- things that change the equations and are not implemented ---------------------------
    // MRF is IMPLEMENTED: correctBoundaryVelocity (UEqn.H:3), DDt (UEqn.H:8) and makeRelative
    // (pEqn.H:5), on the same cpu::MRF::Zone the _cpp reference is gated on. update() is moving-mesh
    // only and inert on a steady static mesh; constrainPressure (pEqn.H:21) is ported, but WITHOUT the
    // MRF.relative term -- pEqn.cu refuses the MRF + fixedFluxPressure combination by name at the call
    // site. What IS still refused here is a zone naming a cellZone the
    // mesh does not carry -- silently applying no rotation there is the failure mode this guards.
    for (const cpu::MRF::ZoneSpec& sp : cpu::MRF::readMRFProperties(caseDir + "/constant"))
    {
        const std::map<std::string, std::vector<label>> zoneMap =
            readCellZones(caseDir + "/constant/polyMesh");
        if (zoneMap.find(sp.cellZone) == zoneMap.end())
            r.blockers.push_back("MRFProperties names cellZone `" + sp.cellZone +
                                 "`, which is not in constant/polyMesh/cellZones. Running would apply "
                                 "no rotation at all rather than the rotation the case asks for.");
    }
    // fvOptions: the FRAMEWORK is ported (dictionary, cellSetOption selection, and `== fvOptions(U)`),
    // and so is explicitPorositySource/DarcyForchheimer. ofscan counts 46 fv::option implementations, so
    // the list is read and any option whose TYPE is not implemented is refused BY NAME -- reading the
    // framework as "fvOptions is supported" would be the silent substitution this guard exists for.
    // Still not implemented for ANY option: fvOptions.constrain(UEqn) and fvOptions.correct(U).
    if (fileExists(caseDir + "/constant/fvOptions") || fileExists(caseDir + "/system/fvOptions"))
    {
        PrimitiveMesh mo;
        bool meshOk = true;
        try { mo.read(caseDir + "/constant/polyMesh"); } catch (...) { meshOk = false; }
        if (meshOk)
        {
            const cpu::fvOptions::OptionList ol = cpu::fvOptions::read(caseDir, mo);
            const std::string bad = ol.firstUnsupported();
            if (!bad.empty())
                r.blockers.push_back("fvOptions declares `" + bad + "`, which the rebuilt path does not "
                                     "implement. Ported: explicitPorositySource/DarcyForchheimer, "
                                     "rotorDiskSource and actuationDiskSource, applied "
                                     "as UEqn.H's `== fvOptions(U)`. OpenFOAM registers 46 fv::option "
                                     "implementations (ofscan: impls option).");
        }
    }

    if (const FoamDict* s = fvSolution.subDict("SIMPLE"))
    {
        // SIMPLEC is implemented (matrixH1 + fvc::snGrad), and so is the constrainPressure pEqn.H
        // calls with rAtU right after (deviceConstrainPressure, gated by ffpi_vs_openfoam).
        (void)s;
    }

    // --- this is the STEADY solver ------------------------------------------------------------
    {
        // NOT a blocker, because it is not one for OpenFOAM either: simpleFoam's UEqn is
        // div(phi,U) + MRF.DDt(U) + divDevReff(U) == fvOptions(U) -- there is no fvm::ddt term, so
        // ddtSchemes is never consulted and the entry is inert. OpenFOAM's OWN squareBend tutorial ships
        // `application simpleFoam` with `ddtSchemes default Euler` and runs it; refusing here blocked a
        // case the reference solver accepts, which is the opposite of matching it.
        const std::string ddt = readDdtSchemeWord(caseDir + "/system/fvSchemes");
        if (!ddt.empty() && ddt != "steadyState")
            r.notices.push_back("ddtSchemes.default is `" + ddt + "`, not steadyState. simpleFoam assembles "
                                "no ddt term, so this entry is inert here -- as it is in OpenFOAM. The run "
                                "is STEADY regardless of what it asks for.");
    }

    // --- the convection scheme ----------------------------------------------------------------
    // The rebuilt UEqn implements OpenFOAM's `upwind` implicit weights. Running a case that asks for
    // limitedLinear/linearUpwind/LUST on those weights would silently solve a different discretisation --
    // which is exactly how brae's LUST implicit-weight defect stayed hidden.
    {
        // `bounded` FIRST: the scheme word after it is still `upwind`, so a guard that only looked at
        // the scheme word accepted `bounded Gauss upwind` and ran it without the Sp term. Measured on
        // pitzDaily: the existing solver's Ux residual fell 1 -> 0.022 over 20 iterations while the
        // rebuilt path plateaued at ~0.5. The term vanishes at convergence, so a converged comparison
        // cannot see it -- which is precisely why the guard has to.
        const LaplacianScheme lap = laplacianScheme(caseDir);
        if (!lap.unsupported.empty())
            r.blockers.push_back("laplacianSchemes asks for `" + lap.unsupported + "`, which caps the "
                                 "non-orthogonal correction against the orthogonal part (limitedSnGrad). "
                                 "Only the uncapped `corrected` is ported; running it would apply a larger "
                                 "correction than the case asked for.");

        const std::string sc = divUScheme(caseDir);
        // OpenFOAM registers 78 surfaceInterpolationSchemes (ofscan: impls surfaceInterpolationScheme).
        // These five are the ones ported and gated; anything else is refused by name.
        if (!sc.empty() && sc != "upwind" && sc != "linearUpwind" && sc != "limitedLinear"
            && sc != "limitedLinearV" && sc != "LUST" && sc != "linearUpwindV")
            r.blockers.push_back("div(phi,U) asks for `" + sc + "`; the rebuilt UEqn implements `upwind`, "
                                 "`linearUpwind`, `linearUpwindV`, `limitedLinear`, `limitedLinearV` and "
                                 "`LUST`. Running it "
                                 "anyway would solve a different discretisation than the case specifies.");
        // LUST reads a gradient name too -- its ctor hands the stream straight to linearUpwind (LUST.H) --
        // so a LUST case naming a gradient brae cannot compute was RUN rather than refused.
        if (sc == "linearUpwind" || sc == "linearUpwindV" || sc == "LUST")
        {
            const std::string bad = linearUpwindGradUnsupported(caseDir, nullptr);
            if (!bad.empty())
                r.blockers.push_back("div(phi,U) is `" + sc + "`, whose named gradient resolves to `" +
                                     bad + "` in gradSchemes; brae computes a plain Gauss linear gradient. "
                                     "This correction does not vanish at convergence, so running it would "
                                     "be wrong rather than merely slower to converge.");
        }
    }

    // The `grad(U)` ENTRY, which is a different lookup from linearUpwind's named gradient and reaches
    // more of the solver: divDevReff's dev2 term and every closure's strain and production all take
    // fvc::grad(U). Blocked on the same terms -- `Gauss linear` and `cellLimited Gauss linear <k>` are
    // computed, anything else would be run as plain Gauss under another limiter's name.
    {
        const std::string bad = gradUSchemeUnsupported(caseDir, nullptr);
        if (!bad.empty())
            r.blockers.push_back("gradSchemes resolves `grad(U)` to `" + bad + "`; brae computes `Gauss "
                                 "linear` and `cellLimited Gauss linear <k>` only. fvc::grad(U) feeds "
                                 "divDevReff's explicit dev2 term and the turbulence closures' strain "
                                 "and production, so running it as plain Gauss would be a different "
                                 "model, not a slower one.");
    }

    // --- turbulence ---------------------------------------------------------------------------
    {
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType == "RAS")
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            if (model != "kEpsilon" && model != "kOmegaSST" && model != "realizableKE"
                && model != "SpalartAllmaras" && model != "kOmegaSSTLM")
                r.blockers.push_back("RASModel is `" + model + "`; the rebuilt path wires kEpsilon, "
                                     "realizableKE, kOmegaSST, kOmegaSSTLM and SpalartAllmaras only. "
                                     "OpenFOAM "
                                     "registers 26 incompressible turbulence models (ofscan: impls "
                                     "incompressible::turbulenceModel).");
        }
        else if (simType != "laminar")
        {
            r.blockers.push_back("simulationType is `" + simType + "`; the rebuilt path supports laminar "
                                 "and RAS/kEpsilon only.");
        }

        if (simType == "RAS")
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            // SpalartAllmaras transports nuTilda and names no k or epsilon entry at all.
            std::vector<std::string> keys;
            if (model == "SpalartAllmaras")
            {
                keys.push_back("div(phi,nuTilda)");
            }
            else
            {
                const std::string second = (model == "kOmegaSST") ? "omega" : "epsilon";
                keys.push_back("div(phi,k)");
                keys.push_back("div(phi," + second + ")");
            }
            for (const std::string& key : keys)
            {
                const TurbDivScheme ts = divTurbScheme(caseDir, key);
                // An EMPTY scheme word means the case has no such entry and none was inherited -- a
                // SpalartAllmaras case transports nuTilda and never names div(phi,k). Refusing on that
                // would be refusing a field the model does not solve.
                if (ts.found && !ts.scheme.empty()
                    && ts.scheme != "upwind" && ts.scheme != "limitedLinear"
                    && ts.scheme != "linearUpwind")
                    r.blockers.push_back("`" + key + "` is `" + ts.scheme + "`; the rebuilt turbulence "
                                         "transport implements `upwind`, `limitedLinear` and "
                                         "`linearUpwind` only. Running a different scheme's matrix under "
                                         "this name would be wrong, not merely slower.");
            }
        }
    }

    // --- coupled patches ----------------------------------------------------------------------
    {
        PrimitiveMesh m;
        try { m.read(caseDir + "/constant/polyMesh"); }
        catch (const std::exception& e)
        {
            r.blockers.push_back(std::string("cannot read the mesh: ") + e.what());
            r.supported = r.blockers.empty();
            return r;
        }
        for (const auto& b : m.patches())
            if (isCoupledInterfaceType(b.type) || b.type == "processor")
                r.blockers.push_back("patch `" + b.name + "` is of coupled type `" + b.type +
                                     "`; the rebuilt components handle no coupled interfaces.");
    }

    // fixedFluxPressure is SUPPORTED here now: the factory builds the real FixedFluxPressurePatchField
    // (which itself refuses any driver that never calls constrainPressure), pEqn.cu runs
    // deviceConstrainPressure at pEqn.H:21, and ffpi_vs_openfoam gates this path against real OpenFOAM
    // on validation/simpleBoxP. The envelope used to refuse the whole case on a raw substring of the
    // 0/p text -- the OVER-BROAD refusal the census called it.

    // --- substitutions that are supported but must be SAID ------------------------------------
    if (const FoamDict* solvers = fvSolution.subDict("solvers"))
        if (const FoamDict* ps = solvers->subDict("p"))
        {
            const std::string sel = ps->wordOr("solver", "");
            if (sel == "GAMG")
                r.notices.push_back("system/fvSolution asks for `GAMG` on p; brae runs an "
                                    "AMG-preconditioned PCG instead. Same operator, different Krylov "
                                    "method and iteration count.");
        }
    // The SMOOTHER substitution, which had no notice anywhere. brae routes a `smoothSolver` with a
    // GaussSeidel-family smoother to deviceSymGaussSeidel, which is OpenFOAM's smoothSolver STOPPING RULE
    // around a MULTICOLOUR sweep where symGaussSeidelSmoother.C sweeps in index order. Same algorithm
    // under a permutation, but Gauss-Seidel is order-dependent, so at the loose relTol a SIMPLE step asks
    // for the two stop in different places: measured on validation/T3A, OpenFOAM cut Ux 1.6186e-05 ->
    // 6.940e-07 in ONE sweep where brae took SEVEN to reach 1.278e-06, both on the case's own relTol 0.1.
    // It was announced nowhere -- the shared reader suppresses the smoother notice precisely when brae
    // takes this path (linear_solver_setup.cuh), and the run log said "as the case asks".
    if (const FoamDict* solvers = fvSolution.subDict("solvers"))
    {
        std::string fields;
        for (const char* f : {"U", "k", "epsilon", "omega", "nuTilda"})
        {
            const FoamDict* fs = solvers->subDict(f);
            if (!fs) continue;
            const std::string sm = fs->wordOr("smoother", "");
            if (fs->wordOr("solver", "") != "smoothSolver") continue;
            if (sm != "symGaussSeidel" && sm != "GaussSeidel") continue;
            if (!fields.empty()) fields += ", ";
            fields += f;
        }
        // The turbulence nSweeps notice that stood here is GONE, not silenced: solvers/<field>/nSweeps
        // now reaches the transported scalars' solve (queue item 47), and a case that sets it
        // differently on the two fields a model carries is refused by name rather than announced.
        if (!fields.empty())
            r.notices.push_back("system/fvSolution asks for `smoothSolver` with a symGaussSeidel smoother "
                                "on " + fields + "; brae runs that smoothSolver around a MULTICOLOUR "
                                "Gauss-Seidel sweep, which SMOOTHS LESS per sweep than OpenFOAM's "
                                "index-order sweep. Same stopping rule, more sweeps, and a loose solve "
                                "stops somewhere else.");
    }

    r.supported = r.blockers.empty();
    return r;
}


bool simpleFoamV2Selected()
{
    const char* e = std::getenv("BRAE_SIMPLEFOAM_V2");
    return e && e[0] == '1';
}


namespace { double g_hookSeconds = 0.0; double g_refreshSeconds = 0.0; }

int runSimpleFoamV2(const std::string& caseDir)
{
    const EnvelopeReport env = simpleFoamV2Envelope(caseDir);
    for (const std::string& n : env.notices)
        std::printf("NOTICE (simpleFoam v2): %s\n", n.c_str());
    if (!env.supported)
    {
        std::string msg =
            "BRAE_SIMPLEFOAM_V2=1 selected the rebuilt simpleFoam, but this case is outside what it "
            "implements:\n";
        for (const std::string& b : env.blockers) msg += "  - " + b + "\n";
        msg += "Refusing rather than solving a different problem. Unset BRAE_SIMPLEFOAM_V2 to run the "
               "existing solver.";
        throw std::runtime_error(msg);
    }

    // ---- case setup --------------------------------------------------------------------------
    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");

    const scalar nu = transport.scalarOr("nu", 1e-5);
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    // OF Time::setControls (Time.C:146-193): only `startFrom startTime` reads the dict entry;
    // `latestTime` scans the case's own numeric time directories and starts from the LAST. This returned
    // the literal "0", so a case restarted that way silently reread the initial field -- measured on
    // simpleBoxIO staged with 0/ and 7/ and endTime 10, real OpenFOAM starts at 7 and takes 3 steps
    // while this took 10 from 0. resolveStartTime is the helper the legacy driver already uses, so the
    // two paths cannot disagree about which directory a case restarts from.
    const std::string startTime = resolveStartTime(caseDir,
                                                   controlDict.wordOr("startFrom", "startTime"),
                                                   controlDict.wordOr("startTime", "0"));
    const scalar startTimeVal = static_cast<scalar>(std::strtod(startTime.c_str(), nullptr));
    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + startTime, simpleDict, m, g, fvp);

    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    cpu::SimpleControl ctl(cd);

    // OF relaxes a FIELD only when the case NAMES a factor: GeometricField::relax() starts at
    // `scalar relaxCoeff = 1` and applies one only `if (mesh().relaxField(name, relaxCoeff))`, which is
    // `found(name) || found("default")` (solution.C:320-327). These defaulted to 0.7/0.3 instead, so a
    // case with no `relaxationFactors/fields` entry -- backwardFacingStep2D is one, as most SIMPLEC
    // cases are -- had its pressure correction cut to 0.3 of what OpenFOAM applies. Measured: after one
    // iteration that left p at 8.47 against OpenFOAM's 28.07 and |U| at 4.51 against 14.72, a ratio of
    // 3.31 = 1/0.3, and it cost roughly 3x the iterations to converge (V2 needed 3000 to reach the
    // agreement OpenFOAM has at ~1000). This is the same "the case NAMES a factor, not the factor is
    // below 1" rule the tree already applies to EQUATION relaxation, on the field side.
    scalar relaxU = 1.0, relaxP = 1.0;
    if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
    {
        // `default` is part of the same lookup in OpenFOAM, so a case that names only a default gets it.
        if (const FoamDict* eq = rf->subDict("equations"))
            relaxU = eq->scalarOr("U", eq->scalarOr("default", relaxU));
        if (const FoamDict* fl = rf->subDict("fields"))
            relaxP = fl->scalarOr("p", fl->scalarOr("default", relaxP));
    }
    // SAID, not assumed. The other drivers print their relaxation and this one printed nothing, so the
    // 0.3 it was applying to a case that names no field factor was invisible in the log.
    std::printf("  relaxation: U %g (equations), p %g (fields)%s\n", (double)relaxU, (double)relaxP,
                relaxP == 1.0 ? "  -- the case names none, so OpenFOAM relaxes p not at all" : "");

    // endTime is an ABSOLUTE TIME and deltaT is the step between iterations -- not a run length and not
    // a count. Time::run() tests `value() < endTime_ - 0.5*deltaT_` (Time.C:785) and Time::operator++
    // advances the value by deltaT (Time.C:1067). This read endTime as the iteration COUNT and never
    // read deltaT at all: a restart at 269 with `endTime 1000` ran 1000 iterations where OpenFOAM runs
    // 731, and it invalidated a whole afternoon of restart measurements before anyone noticed
    // (REFUSALS.md item 39). Measured against real OpenFOAM on simpleBoxIO with residualControl emptied:
    // startTime 5 / endTime 8 / deltaT 1 -> OpenFOAM 3 iterations (Time = 6, 7, 8) against this driver's
    // 8; startTime 0 / endTime 2.5 / deltaT 0.5 -> OpenFOAM 5 ending at 2.5/ against 2 ending at 2/.
    const scalar endTimeVal = controlDict.scalarOr("endTime", 100);
    const scalar deltaT     = controlDict.scalarOr("deltaT", 1.0);
    if (deltaT <= 0.0)
        throw std::runtime_error(
            "controlDict deltaT = " + std::to_string(static_cast<double>(deltaT))
            + ". OpenFOAM advances the time by deltaT every iteration (Time.C:1067), so a non-positive "
              "step never reaches endTime.");
    // OF's own inequality, stepped OF's own way, rather than a rounded quotient: the two disagree at
    // (endTime - startTime)/deltaT = n + 0.5. Measured with startTime 0 / endTime 1 / deltaT 0.4 (ratio
    // exactly 2.5): OpenFOAM runs 2 iterations where std::lround gives 3. Accumulating deltaT rather
    // than multiplying it also reproduces OpenFOAM's own rounding drift.
    const label nSteps = static_cast<label>(openFoamNSteps(static_cast<double>(startTimeVal),
                                                          static_cast<double>(endTimeVal),
                                                          static_cast<double>(deltaT)));
    if (nSteps < 1)
        throw std::runtime_error(
            "controlDict endTime (" + WriteControl::timeName(endTimeVal) + ") is not beyond the start "
            "time (" + startTime + "): there is nothing to run. endTime is an ABSOLUTE time, not a "
            "number of iterations -- on a restart set it past the time you are restarting from.");
    // SAID, not assumed. The driver printed nothing about how long it was going to run, which is how an
    // 8-iteration run of a 3-iteration case went unnoticed.
    std::printf("  time: startTime %s, endTime %s, deltaT %g -> %d iterations\n",
                startTime.c_str(), WriteControl::timeName(endTimeVal).c_str(),
                static_cast<double>(deltaT), static_cast<int>(nSteps));

    // ---- MRF ---------------------------------------------------------------------------------
    // HOOK 1, UEqn.H:3 -- MRF.correctBoundaryVelocity(U). It runs HERE, before the device boundary is
    // built, because the included patch faces take the frame velocity Omega x (Cf - origin) and that
    // value has to be in the field the DeviceVectorBoundary is built from. brae's noSlip is a fixedValue
    // whose matrix coefficients come from its live value, exactly as OpenFOAM's does, so the assignment
    // reaches internalCoeffs/boundaryCoeffs.
    std::vector<cpu::MRF::Zone>  mrfZones;
    std::vector<DeviceMRFZone>   dMrf;
    {
        const std::vector<cpu::MRF::ZoneSpec> specs = cpu::MRF::readMRFProperties(caseDir + "/constant");
        if (!specs.empty())
        {
            const std::map<std::string, std::vector<label>> zoneMap =
                readCellZones(caseDir + "/constant/polyMesh");
            for (const cpu::MRF::ZoneSpec& sp : specs)
            {
                const auto it = zoneMap.find(sp.cellZone);
                if (it == zoneMap.end())
                    throw std::runtime_error(
                        "simpleFoam v2: MRF cellZone `" + sp.cellZone +
                        "` is not in constant/polyMesh/cellZones.");
                mrfZones.push_back(cpu::MRF::buildZone(sp, it->second, m, fvp));
            }
            cpu::MRF::correctBoundaryVelocity(f.U, mrfZones, fvp);
            for (const cpu::MRF::Zone& z : mrfZones)
            {
                dMrf.push_back(buildDeviceMRFZone(z, m, g, fvp));
            }
            std::printf("  MRF: %zu zone(s)", mrfZones.size());
            for (const cpu::MRF::Zone& z : mrfZones)
            {
                std::printf("   Omega (%.4g %.4g %.4g) over %zu cells",
                            z.Omega.x, z.Omega.y, z.Omega.z, z.cells.size());
            }
            std::printf("\n");
        }
    }

    // ---- device state ------------------------------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    // NOT const: the freestream family's per-face valueFraction is recomputed every iteration from the
    // flow angle, which writes into these (deviceUpdateMixedFreestream).
    DeviceVectorBoundary dbU = buildDeviceVectorBoundary(f.U, fvp, g);
    DeviceBoundary dbP = buildDeviceBoundary(f.p, fvp, g);
    // Does any patch actually carry a mixed BC? Only then is the per-step update worth running.
    bool hasMixedBC = false;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (f.U.boundary[pi]->bcCategory() == 5 || f.p.boundary[pi]->bcCategory() == 5) hasMixedBC = true;
    }

    SolverFields gf;
    {
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        { ux[c] = f.U.internal[c].x; uy[c] = f.U.internal[c].y; uz[c] = f.U.internal[c].z; }
        gf.Ux.copyFrom(ux); gf.Uy.copyFrom(uy); gf.Uz.copyFrom(uz);
        gf.p.copyFrom(f.p.internal);
        gf.phiInt.copyFrom(f.phi.internal);
        std::vector<scalar> pb;
        for (const auto& v : f.phi.boundary) for (scalar x : v) pb.push_back(x);
        pb.resize(dm.nBndFaces, 0.0);
        gf.phiBnd.copyFrom(pb);
    }

    const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");
    const bool ras = (turbProps.wordOr("simulationType", "laminar") == "RAS");

    std::vector<scalar> nuEffC(nC, nu);
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nuEffB[pi].assign(fvp[pi].size, nu);
    const SurfaceScalarField nuFace = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);

    std::vector<scalar> nuBndFlat;
    for (const auto& v : nuEffB) for (scalar x : v) nuBndFlat.push_back(x);
    nuBndFlat.resize(dm.nBndFaces, nu);

    DeviceBuffer<scalar> dNuCell(nuEffC), dNuFace(nuFace.internal), dNuBnd(nuBndFlat);

    std::vector<label> takeU, adjustable;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            takeU.push_back(f.U.boundary[pi]->assignable() ? 0 : 1);
            // adjustPhi's predicate needs BOTH halves (adjustPhi.C:59): mixed fixesValue() is TRUE
            // and inletOutlet inherits it, so fixesValue alone marked an inletOutlet outlet as FIXED
            // outflow -- deviceAdjustPhi then had nothing adjustable and refused a case OpenFOAM
            // solves (measured on simpleBoxIO: "adjustable mass outflow 0.000000"). The rho mirror's
            // mask builder (rhoCreateFields.cu) had it right; this one predates it.
            adjustable.push_back(
                (f.U.boundary[pi]->fixesValue() && !f.U.boundary[pi]->isInletOutlet()) ? 0 : 1);
        }
    takeU.resize(dm.nBndFaces, 0);
    adjustable.resize(dm.nBndFaces, 0);
    DeviceBuffer<label> dTakeU(takeU), dAdjust(adjustable);

    // ---- turbulence: k-epsilon on the device, behind the driver's hook ------------------------
    //
    // The hook exists so the driver never names a model (that is what made the old solver a god object).
    // Everything the model needs is set up here and captured; the driver only knows that SOMETHING runs
    // at the end of the iteration, which is where simpleFoam.C:94 calls turbulence->correct().
    //
    // The hook also owns the nuEff REFRESH. The driver reads nuEff through pointers, so updating those
    // buffers here is what makes the coupling lagged in the right direction: iteration n+1's momentum
    // equation sees the nut this correct() just produced, and nothing else in the loop has to know.
    GeometricField<scalar> kF, epsF, nutF;
    DeviceBuffer<scalar> dK, dEps, dNut;
    DeviceWallData wall;
    DeviceBoundary dbK, dbEps, dbNut;
    DeviceBuffer<label> bndIsWall;
    DeviceBuffer<scalar> bndY;
    // THE NUT WALL FUNCTION THE CASE ASKS FOR. This driver had no selector at all: refreshBoundaryNut
    // ran the k-based deviceBoundaryNut whatever 0/nut said, and every turbulence correct() below was
    // handed `nutWall = 0` (nutk) for its near-wall production. So a nutUBlendedWallFunction case got
    // BOTH the wall viscosity and the near-wall G0 from the wrong formula -- measured on
    // backwardFacingStep2D as a wall nut of 0.000e+00 where the dispatching path gives up to 1.5e-01.
    // Derived through the SAME selectNutWall the other drivers use (turbulence_setup.cuh), so the two
    // cannot disagree about what the case asked for.
    NutWall v2NutWall = NutWall::Nutk;
    scalar relaxK = 0.7, relaxEps = 0.7;
    // k/epsilon linear-solver settings, read from fvSolution below. Declared here because the turbulence
    // hook is a lambda defined above that point and captures them by reference; it does not run until the
    // outer loop, by which time they are set.
    scalar tolKE = 1e-10, relTolKE = 0.0;
    bool   gsKE  = false;
    // div(phi,k) / div(phi,<second>): scheme, its coefficient as the device's twoByk, and `bounded`.
    bool   limitedK = false, limitedEps = false;
    bool   boundedK = false, boundedEps = false;
    // `Gauss linearUpwind grad(<var>)` on the turbulence scalar: upwind's MATRIX plus a deferred gradient
    // correction. Module 2 reads it; declared here with the rest of the scheme state.
    bool   linearUpwindK = false, linearUpwindEps = false;
    // The case's laplacianScheme, read ONCE here because both the turbulence hook and the momentum /
    // pressure equations need it. OpenFOAM applies it to every laplacian in the case, the turbulence
    // ones included; this driver applied it only to momentum and pressure.
    bool   nonOrthLaplacian = false;
    scalar snGradLimitK = 0.0;
    {
        const LaplacianScheme lapS = laplacianScheme(caseDir);
        nonOrthLaplacian = lapS.corrected;
        snGradLimitK     = lapS.limitCoeff;
    }
    // The `grad(U)` ENTRY's limiter coefficient, read once here because everything that takes
    // fvc::grad(U) needs it: divDevReff's explicit dev2 term (linearViscousStress.C:114), the closures'
    // strain and production (kOmegaSSTBase.C:522, kEpsilon.C:237, SpalartAllmarasBase.C:461), the
    // startup correctNut, and the SST's `calculated` nut boundary. Every one of them ran the UNLIMITED
    // gradient while the driver printed the case's coefficient. NOT the same lookup as
    // linearUpwindGradUnsupported's, which resolves the word linearUpwind names -- on rotorDisk they
    // differ. Any unsupported form is refused by the envelope above, so a nonzero here is a coefficient
    // brae computes.
    scalar gradUSchemeLimit = 0.0;
    gradUSchemeUnsupported(caseDir, &gradUSchemeLimit);
    // The TRANSPORTED scalars' own `grad(<var>)` entries, which feed their limitedLinear weight, their
    // linearUpwind correction and the non-orthogonal laplacian correction. brae carries ONE coefficient
    // for the pair, so a case that limits them differently is refused by name rather than run with
    // whichever one was read first. windAroundBuildingsBox limits all three at 1.
    // Declared here so the turbulence hook can capture it; RESOLVED below, once the model is known --
    // the two entries to compare are the two variables the model actually transports, and `grad(omega)`
    // means nothing to a kEpsilon case.
    scalar gradScalarSchemeLimit = 0.0;
    // fvSolution solvers/<field>/nSweeps for the TRANSPORTED turbulence scalars. brae holds ONE value
    // for the pair the model carries, so a case that sets them differently is refused rather than run
    // with whichever was read first.
    int nSweepsKE = 1;
    scalar twoBykK = 2.0, twoBykEps = 2.0;
    bool   sstModel = false;
    // SpalartAllmaras transports ONE scalar, nuTilda, and rides the k slot; the epsilon slot is unused.
    bool   saModel = false;
    // kOmegaSSTLM rides the kOmegaSST slots (omega in the `epsilon` slot) and adds two transported
    // scalars of its own plus gammaIntEff, which is NOT read from a file: OpenFOAM constructs it zero.
    bool   lmModel = false;
    DeviceBuffer<scalar> dReThetat, dGammaInt, dGammaIntEff;
    DeviceBoundary       dbReThetat, dbGammaInt;
    cpu::kOmegaSSTLM::Coeffs lmCoeffs{};
    cpu::SA::Coeffs saCoeffs;
    // nutUSpaldingWallFunction faces, and their wall distance. See where they are built for why the mask
    // is keyed off nut's BC rather than the patch type.
    bool   hasSpaldingWall = false;
    DeviceBuffer<label>  spaldingIsWall;
    DeviceBuffer<scalar> spaldingY;
    std::string secondField = "epsilon";
    KOmegaSSTCoeffs sstCoeffs;
    KEpsilonCoeffs  keCoeffs;
    DeviceBuffer<scalar> dY;             // cell wall distance -- kOmegaSST's F1/F2 need it per CELL

    // OpenFOAM prints an Initial residual for every turbulence solve; comparing those against its log is
    // how this path is checked against the oracle (tests/boundary_nut_vs_openfoam.sh).
    const bool printTurbResid = std::getenv("BRAE_TURB_RESID") != nullptr;

    // atmNutkWallFunction parameters, read from 0/nut's own boundary below (OF decides the wall model
    // per BC, not per turbulence model).
    scalar atmZ0       = 0.0;
    bool   atmBoundNut = true;

    // nut at boundary FACES, read twice per outer iteration: once as the k/epsilon patch diffusivity
    // (before the solve) and once for the momentum nuEff (after it). Declared out here with the rest of
    // the hook's state because the `if (ras)` block below only FILLS it -- the hook itself runs later.
    bool hasNutCalc = false;
    bool keNut = false;
    DeviceBuffer<label>  nutCalcMask;
    DeviceBuffer<scalar> nb, nutKb, nutEb, sstGradU;
    // SA's nut = nuTilda*fv1 field assignment, kept apart from nb so nb can carry the previous wall nut
    // into the Spalding warm start.
    DeviceBuffer<scalar> saNutAssigned;
    auto refreshBoundaryNut = [&]()
    {
        if (keNut)
        {
            deviceBCValue(dbK, dK, nutKb);
            deviceBCValue(dbEps, dEps, nutEb);
        }
        // SpalartAllmaras does NOT go through the k-based wall nut: its wall treatment is
        // nutUSpaldingWallFunction, and its non-wall faces take the nut = nuTilda*fv1 field assignment.
        // Running deviceBoundaryNut first would overwrite nb with cell values, destroying the previous
        // WALL nut that Spalding's Newton warm-starts from -- OpenFOAM seeds it from nut_ itself, and
        // seeding from the adjacent cell instead leaves the wall 14% out at convergence.
        // The BC's own function, not the k-based one for everything. Each kernel rewrites every WALL
        // face and leaves the rest to the branch below, exactly as the dispatching driver does.
        if (!saModel && v2NutWall == NutWall::Spalding)
        {
            SpalartAllmarasCoeffs dsp;
            dsp.kappa = keCoeffs.kappa;
            dsp.E     = keCoeffs.E;
            deviceBoundaryNutSpalding(dbU, bndIsWall, bndY, gf.Ux, gf.Uy, gf.Uz,
                                      dNut, nu, dsp, nb, nullptr, nullptr);
        }
        else if (!saModel && v2NutWall == NutWall::Blended)
            deviceBoundaryNutBlended(dbU, bndIsWall, bndY, gf.Ux, gf.Uy, gf.Uz, dNut, nu,
                                     keCoeffs.kappa, keCoeffs.E, nb, nullptr, nullptr);
        else if (!saModel && v2NutWall == NutWall::NutU)
            deviceBoundaryNutU(dbU, bndIsWall, bndY, gf.Ux, gf.Uy, gf.Uz, dNut, nu,
                               keCoeffs.kappa, keCoeffs.E, nb, nullptr, nullptr);
        else if (!saModel)
        deviceBoundaryNut(dbNut, bndIsWall, bndY, dK, dNut, nu, nb,
                          keCoeffs, atmZ0, atmBoundNut,
                          /*nuFace*/nullptr,
                          keNut ? &nutCalcMask : nullptr,
                          keNut ? &nutKb : nullptr,
                          keNut ? &nutEb : nullptr);
        // nutLowReWallFunction: calcNut() is Zero UNCONDITIONALLY on every model. The k-based branch
        // above still runs in full so the calculated/inlet faces keep their values; only the WALL faces
        // are then zeroed, which is what OpenFOAM writes there.
        if (!saModel && v2NutWall == NutWall::LowRe && bndIsWall.size() == nb.size())
        {
            DeviceBuffer<scalar> keepNb;
            deviceCopy(keepNb, nb);
            cudaCheck(cudaMemsetAsync(nb.data(), 0, nb.size()*sizeof(scalar), cudaStreamPerThread),
                      "v2 nutLowRe wall nut");
            deviceSelectFixedFlux(bndIsWall, keepNb, nb);   // non-wall faces keep what they had
        }

        // SpalartAllmaras: nut_ = nuTilda*fv1 is a FIELD ASSIGNMENT, so EVERY boundary face takes
        // nuTilda's own patch value -- not the adjacent cell's, which is what deviceBoundaryNut leaves
        // there for a non-wall patch. Measured on airFoil2D: the outlet's nuEff was 2.88e-01 away from
        // what the nut field itself carried, on a nut of order 1e-03. Same defect class as kEpsilon's
        // DkEff(patchi) and the SST's; SA just reaches it through a different expression.
        // Into its OWN buffer, not into nb: nb still holds the previous iteration's wall nut, and the
        // Spalding Newton below warm-starts from it exactly as OpenFOAM does. Overwriting nb here cold-
        // started that iteration at ~0 (nuTilda is fixedValue ZERO at a wall) and its 10 iterations with a
        // 1% early-out then landed on 4.86e-06 where OpenFOAM has 4.55e-03 -- a thousandfold.
        if (saModel)
        {
            // nb carries the previous iteration's boundary nut -- the Spalding seed. Seeded once from
            // the nut field the case shipped, exactly what OpenFOAM starts from.
            if (nb.size() != static_cast<std::size_t>(dm.nBndFaces))
            {
                std::vector<scalar> nb0;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    if (isCoupledInterfaceType(fvp[pi].type)) continue;
                    const std::vector<scalar>& fb = nutF.boundary[pi]->value();
                    nb0.insert(nb0.end(), fb.begin(), fb.end());
                }
                nb0.resize(dm.nBndFaces, 0.0);
                nb.copyFrom(nb0);
            }
            DeviceBuffer<scalar> ntB;
            deviceBCValue(dbK, dK, ntB);
            deviceNutSABoundary(ntB, nullptr, nu, saCoeffs.Cv1, saNutAssigned);
        }

        // ...and THEN correctBoundaryConditions(): correctNut writes nut_ = nuTilda*fv1, and nuTilda is
        // fixedValue ZERO at a wall -- so the assignment leaves the wall with NO eddy viscosity. OF then
        // runs correctBoundaryConditions(), and nutUSpaldingWallFunction overwrites that from Spalding's
        // law (~4.5e-03 on airFoil2D). Measured on the host reference, taking the assignment instead cost
        // 268x on U, 598x on p and 35x on nuTilda end to end.
        if (saModel && hasSpaldingWall)
        {
            SpalartAllmarasCoeffs dsaW;
            dsaW.kappa = saCoeffs.nutKappa;
            dsaW.E     = saCoeffs.E;
            // The kernel rewrites EVERY face: wall faces from Spalding's law seeded by whatever nb
            // already holds, and every other face from `nutFile` -- or, without it, from the adjacent
            // CELL, which would discard the field assignment.
            deviceBoundaryNutSpalding(dbU, spaldingIsWall, spaldingY, gf.Ux, gf.Uy, gf.Uz,
                                      dNut, nu, dsaW, nb, nullptr, &saNutAssigned);
        }

        // kOmegaSST's `calculated` patches carry a1*k_b/max(a1*om_b, b1*F2_b*sqrt(S2_b)) -- a different
        // expression from kEpsilon's Cmu*k_b^2/eps_b, and one that needs the BOUNDARY F2 and S2 (hence
        // the boundary gradU). Overwrites those faces only; walls keep the wall-function value above.
        // The tutorials ship the inlet as `calculated; value uniform 0`, so without this a run started
        // from 0/ carries zero eddy viscosity on the inlet for its whole length.
        if (sstModel && hasNutCalc)
        {
            deviceBCValue(dbK, dK, nutKb);
            deviceBCValue(dbEps, dEps, nutEb);
            deviceGradU(dm, dbU, gf.Ux, gf.Uy, gf.Uz, sstGradU);
            // cellLimitedGrad limits the INTERIOR and rebuilds the boundary from it
            // (cellLimitedGrad.C:224-226), so a limited interior against an unlimited boundary would be
            // neither scheme. backwardFacingStep2D ships `calculated` nut on two patches, which is
            // exactly where that would show.
            if (gradUSchemeLimit > 0.0)
                deviceCellLimitGradU(dm, dbU, gf.Ux, gf.Uy, gf.Uz, sstGradU, gradUSchemeLimit);
            deviceSSTNutBoundary(dbU, nutKb, nutEb, dY, nullptr, nu,
                                 sstGradU, static_cast<int>(nC), gf.Ux, gf.Uy, gf.Uz,
                                 nutCalcMask, sstCoeffs, dNut, nb);
        }
    };

    StepInput in;
    in.mrf = dMrf.empty() ? nullptr : &dMrf;
    // The momentum components OpenFOAM SOLVES, from the same EMPTY patches polyMesh::calcDirections
    // reads (polyMesh.C:75-118). Derived here, where the patches are, because gpu::simpleStep sees only
    // the DeviceMesh and cannot recover a patch type from it. Wedge patches are deliberately not
    // consulted: they knock out geometricD_ only (polyMesh.C:128-141), so an axisymmetric case still
    // solves all three.
    {
        const SolutionDirections sd = solutionDirections(fvp);
        for (int cmpt = 0; cmpt < 3; ++cmpt) in.solutionD[cmpt] = sd.d[cmpt];
        if (!sd.valid(0) || !sd.valid(1) || !sd.valid(2))
            std::printf("  empty patches knock out a solution direction: U is solved in (%s%s%s) only, "
                        "as fvMatrix<vector>::solveSegregated does\n",
                        sd.valid(0) ? "x" : "", sd.valid(1) ? "y" : "", sd.valid(2) ? "z" : "");
    }
    if (ras)
    {
        // kOmegaSST's second transport variable is omega, and brae holds it in the same slot epsilon
        // uses -- the fused device correct() for each model takes that slot and knows what it means.
        // `ras` here is the simulationType switch, not the dict -- the RAS sub-dictionary is re-fetched.
        const FoamDict* rasDict = turbProps.subDict("RAS");
        lmModel  = rasDict && rasDict->wordOr("RASModel", "") == "kOmegaSSTLM";
        // The LM model IS kOmegaSST plus the transition transport, so every kOmegaSST path below is
        // taken as well -- the two flags are deliberately not exclusive.
        sstModel = lmModel || (rasDict && rasDict->wordOr("RASModel", "") == "kOmegaSST");
        // realizableKE rides the SAME fused device kEpsilon through KEpsilonCoeffs::realizable: variable
        // Cmu, the strain-based epsilon production and the C2*eps^2/(k + sqrt(nu*eps)) destruction are
        // all selected by that flag, so this is a coefficient change and not a second model path.
        if (rasDict && rasDict->wordOr("RASModel", "") == "realizableKE")
        {
            keCoeffs.realizable = true;
            RealizableKECoeffs rc;
            readRealizableKECoeffs(rasDict, rc);
            keCoeffs.A0 = rc.A0;  keCoeffs.C2 = rc.C2;
            keCoeffs.sigmaK = rc.sigmak;  keCoeffs.sigmaEps = rc.sigmaEps;
            keCoeffs.kappa = rc.kappa;  keCoeffs.E = rc.E;
        }
        saModel = rasDict && rasDict->wordOr("RASModel", "") == "SpalartAllmaras";
        if (saModel) cpu::SA::readCoeffs(rasDict, saCoeffs);

        // SpalartAllmaras has no k and no epsilon: nuTilda is the whole model, and it rides the k slot
        // exactly as it does in the _cpp reference. Reading a `k` that the case does not ship would fail
        // before the model ever ran.
        secondField = sstModel ? "omega" : "epsilon";
        const std::string firstField = saModel ? "nuTilda" : "k";
        // The transported scalars' own `grad(<var>)` entries, resolved for the variables THIS model
        // carries. brae holds ONE coefficient for the pair, so a case that limits them differently is
        // refused by name rather than run with whichever was read first. Resolved here and not with
        // grad(U) above because `grad(omega)` is not a question a kEpsilon case answers.
        {
            std::string txt;
            try { txt = readFvSchemesText(caseDir); } catch (...) { txt.clear(); }
            if (!txt.empty())
            {
                gradSchemeUnsupported(txt, "grad(" + firstField + ")", &gradScalarSchemeLimit);
                if (!saModel)
                {
                    scalar k2 = 0.0;
                    gradSchemeUnsupported(txt, "grad(" + secondField + ")", &k2);
                    if (std::fabs(k2 - gradScalarSchemeLimit) > 1e-12)
                        throw std::runtime_error(
                            "brae: gradSchemes resolves `grad(" + firstField + ")` and `grad("
                            + secondField + ")` to different cellLimited coefficients ("
                            + std::to_string((double)gradScalarSchemeLimit) + " and "
                            + std::to_string((double)k2) + "). This driver carries one coefficient for "
                            "the transported pair, so running would apply one variable's limiter under "
                            "the other's name.");
                }
                if (gradScalarSchemeLimit > 0.0)
                    std::printf("  gradSchemes grad(%s)/grad(%s): cellLimited Gauss linear %g\n",
                                firstField.c_str(), secondField.c_str(), (double)gradScalarSchemeLimit);
            }
        }
        // V2 maintains NO per-step boundary (fixedMean, fanPressure, coded) -- the legacy driver's
        // NVRTC and collect* hooks live in DeviceSimpleSolver, which this path does not use. Check the
        // types here, where they still exist; buildField erases them (frozen_bc_guard.cuh).
        const FieldData<scalar> kFd = readField<scalar>(caseDir + "/" + startTime + "/" + firstField);
        refuseFrozenPerStepBC(kFd, firstField, "simpleFoamV2", false);
        kF   = buildField<scalar>(kFd, fvp, nC);
        if (!saModel)
        {
            const FieldData<scalar> eFd = readField<scalar>(caseDir + "/" + startTime + "/" + secondField);
            refuseFrozenPerStepBC(eFd, secondField, "simpleFoamV2", false);
            epsF = buildField<scalar>(eFd, fvp, nC);
        }
        const FieldData<scalar> nutFd = readField<scalar>(caseDir + "/" + startTime + "/nut");
        refuseFrozenPerStepBC(nutFd, "nut", "simpleFoamV2", false);
        nutF = buildField<scalar>(nutFd, fvp, nC);
        // atmNutkWallFunction: the ATMOSPHERIC rough-wall nut, nut = nu*(y+*kappa/log(max(Edash,1+1e-4)) - 1)
        // with Edash built from the surface roughness z0. Running it as the smooth nutkWallFunction is not a
        // small error on a terrain case -- z0 IS the terrain, and the whole boundary layer hangs off it.
        // brae carries ONE z0 for the solver, so two different roughnesses have to be refused rather than
        // averaged into a landscape the case never described.
        for (const PatchFieldData<scalar>& pb : nutFd.boundary)
        {
            if (pb.type != "atmNutkWallFunction") continue;
            if (atmZ0 > 0 && std::fabs(pb.ablZ0 - atmZ0) > 1e-12*std::max<scalar>(atmZ0, pb.ablZ0))
                throw std::runtime_error(
                    "brae: 0/nut has atmNutkWallFunction patches with different roughness lengths (z0 "
                    + std::to_string(atmZ0) + " and " + std::to_string(pb.ablZ0) + "). This driver applies "
                    "one z0 to every rough wall, so running would give patches a roughness the case does "
                    "not ask for.");
            atmZ0       = pb.ablZ0;
            atmBoundNut = pb.atmBoundNut;
        }
        if (atmZ0 > 0)
            std::printf("  nut wall function: atmNutkWallFunction (rough, z0=%g, boundNut=%s)\n",
                        (double)atmZ0, atmBoundNut ? "true" : "false");
        selectNutWall(nutFd, fvp, saModel, saModel ? "SpalartAllmaras" : (sstModel ? "kOmegaSST" : "kEpsilon"),
                      v2NutWall, atmZ0, atmBoundNut);
        kF.evaluateBoundary();
        if (!saModel) epsF.evaluateBoundary();
        nutF.evaluateBoundary();

        // MUST_READ in OpenFOAM: kOmegaSSTLM refuses to construct without them, and there is no default.
        if (lmModel)
        {
            GeometricField<scalar> reF =
                buildField<scalar>(readField<scalar>(caseDir + "/" + startTime + "/ReThetat"), fvp, nC);
            GeometricField<scalar> giF =
                buildField<scalar>(readField<scalar>(caseDir + "/" + startTime + "/gammaInt"), fvp, nC);
            reF.evaluateBoundary();
            giF.evaluateBoundary();
            dReThetat.copyFrom(reF.internal);
            dGammaInt.copyFrom(giF.internal);
            dbReThetat = buildDeviceBoundary(reF, fvp, g);
            dbGammaInt = buildDeviceBoundary(giF, fvp, g);
            // gammaIntEff starts at ZERO, as OpenFOAM constructs it -- so the first outer iteration has
            // no turbulent production at all. Seeding it to 1 would make the first iteration fully
            // turbulent and move where transition lands.
            dGammaIntEff.copyFrom(std::vector<scalar>(nC, 0.0));
            std::printf("  kOmegaSSTLM (Langtry-Menter transition): + ReThetat + gammaInt transport, "
                        "gammaIntEff seeded at 0 as OpenFOAM constructs it\n");
        }

        dK.copyFrom(kF.internal); dNut.copyFrom(nutF.internal);
        dbK   = buildDeviceBoundary(kF,   fvp, g);
        dbNut = buildDeviceBoundary(nutF, fvp, g);
        if (!saModel)
        {
            dEps.copyFrom(epsF.internal);
            dbEps = buildDeviceBoundary(epsF, fvp, g);
        }

        // OF fills nut by FIELD ASSIGNMENT (nut_ = Cmu*sqr(k_)/epsilon_), which writes the BOUNDARY from
        // the boundary k and epsilon; correctBoundaryConditions() then leaves a `calculated` patch alone.
        // So such a patch carries Cmu*k_b^2/eps_b, NOT the adjacent cell value -- and that boundary nut is
        // what sets the patch diffusivity DkEff(patchi)/DepsilonEff(patchi) in the k and epsilon
        // laplacians. Measured on pitzDaily: the inlet's nut_b is 8.52e-04 against a cell value 12x
        // larger, and taking the cell value put 90.5% of the whole epsilon residual on that one patch --
        // the 10x baseline against OpenFOAM's own log. realizableKE's nut carries a variable rCmu and the
        // SST's is a different expression again, so only kEpsilon may evaluate it this way.
        {
            std::vector<label> cm;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;

                // A non-wall `fixedValue` nut is PINNED by the case -- conventionally to 0 -- and must not
                // be lumped in with `calculated`, which means "correctNut filled this in".
                const bool calc = (fvp[pi].type != "wall") && (nutF.boundary[pi]->bcCategory() == 2);
                if (calc) hasNutCalc = true;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    cm.push_back(calc ? 1 : 0);
                }
            }
            if (hasNutCalc) nutCalcMask.copyFrom(cm);
            keNut = hasNutCalc && !sstModel && !keCoeffs.realizable;
        }

        // Wall geometry for the wall functions. wallU is the patch velocity; a static mesh with no
        // movingWallVelocity means the field's own boundary value is what the solver imposes.
        std::vector<std::vector<vector>> wallU(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) wallU[pi] = f.U.boundary[pi]->value();
        wall = buildDeviceWallData(m, g, fvp, wallU);

        if (sstModel)
        {
            // F1/F2 blend on the wall distance at EVERY cell (kOmegaSSTBase.C), not the near-wall face
            // distance the wall functions use. Two different quantities with the same symbol in OpenFOAM.
            dY.copyFrom(cellWallDist(m, g, fvp));
            readKOmegaSSTCoeffs(turbProps.subDict("RAS"), sstCoeffs);
            // kOmegaSSTLM's own coeffDict. The device carried OpenFOAM's DEFAULTS hardcoded, so a case
            // that overrode ca1 or cThetat ran the defaults silently.
            if (lmModel) cpu::kOmegaSSTLM::readCoeffs(turbProps.subDict("RAS"), lmCoeffs);
        }
        // SA needs the same CELL wall distance: dTilda is y everywhere, not only near the wall.
        if (saModel) dY.copyFrom(cellWallDist(m, g, fvp));

        {
            const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
            std::vector<label> isW; std::vector<scalar> yv;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;

                // OpenFOAM keys the epsilon/omega wall treatment on the BC TYPE, not on the mesh patch
                // type: createAveragingWeights counts the faces whose epsilon field carries an
                // epsilonWallFunction (epsilonWallFunctionFvPatchScalarField.C:100-118). The two coincide
                // on pitzDaily, where the walls carry the wall function, so keying on `type == "wall"`
                // looked right for a long time. They do NOT coincide on simpleCar: its 0/epsilon sets
                // "(body|upperWall|lowerWall)" to epsilonWallFunction but then a trailing ".*" entry
                // overrides it to zeroGradient -- last matching regex wins, in OpenFOAM and here alike --
                // so OpenFOAM applies NO wall function there and brae was applying one on three patches.
                // That was a real defect, but it is NOT the whole of the 50% simpleCar disagreement: with
                // it fixed that case's epsilon residual is still 127x OpenFOAM's, and 76% of what remains
                // now sits in the INTERIOR, so the rest of the cause is not a boundary treatment at all.
                // SpalartAllmaras has no epsilon field and no epsilon wall function: the wall treatment
                // it needs is on nut (nutUSpaldingWallFunction), not on the transported scalar, so the
                // epsilon-wall mask is simply empty here.
                const bool isWall = !saModel && epsF.boundary[pi]->isTurbulenceWallFunction();
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    isW.push_back(isWall ? 1 : 0);
                    yv.push_back(isWall ? yW[pi][i] : 0.0);
                }
            }
            bndIsWall.copyFrom(isW);

            // SpalartAllmaras puts its wall treatment on NUT, not on the transported scalar: a wall
            // carrying nutUSpaldingWallFunction recomputes nut from Spalding's law every correctNut.
            // Keyed off nut's own BC, exactly as the host reference is.
            if (saModel)
            {
                std::vector<label> spW;
                std::vector<scalar> spY;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    if (isCoupledInterfaceType(fvp[pi].type)) continue;
                    const bool sp = nutF.boundary[pi]->isNutUSpalding();
                    if (sp) hasSpaldingWall = true;
                    for (label i = 0; i < fvp[pi].size; ++i)
                    {
                        spW.push_back(sp ? 1 : 0);
                        spY.push_back(sp ? yW[pi][i] : 0.0);
                        if (sp && i == 0 && std::getenv("BRAE_DUMP_NUEFF"))
                            std::printf("  [wallY] patch %s face0: nearWallDist %.6e   1/deltaCoeffs %.6e\n",
                                        fvp[pi].name.c_str(), yW[pi][i], 1.0 / fvp[pi].deltaCoeffs[i]);
                    }
                }
                if (hasSpaldingWall)
                {
                    spaldingIsWall.copyFrom(spW);
                    spaldingY.copyFrom(spY);
                }
            }
            bndY.copyFrom(yv);
        }

        if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
            if (const FoamDict* eq = rf->subDict("equations"))
            {
                relaxK   = eq->scalarOr("k", relaxK);
                relaxEps = eq->scalarOr(secondField, relaxEps);   // `omega` under kOmegaSST
            }

        // div(phi,k) and div(phi,<second>) carry their own scheme; the envelope above has already
        // refused anything but upwind and limitedLinear. limitedLinear's device form takes
        // twoByk = 2/coeff, and reduces to upwind at limiter 0, so upwind stays bit-identical.
        {
            const TurbDivScheme tk =
                divTurbScheme(caseDir, saModel ? "div(phi,nuTilda)" : "div(phi,k)");
            const TurbDivScheme te = saModel ? TurbDivScheme{}
                                             : divTurbScheme(caseDir, "div(phi," + secondField + ")");
            limitedK    = (tk.scheme == "limitedLinear");
            limitedEps  = (te.scheme == "limitedLinear");
            // linearUpwind keeps upwind's matrix and adds a deferred gradient correction, so it is a
            // separate flag rather than another value of the same one.
            linearUpwindK   = (tk.scheme == "linearUpwind");
            linearUpwindEps = (te.scheme == "linearUpwind");
            twoBykK     = 2.0 / std::max(tk.coeff, static_cast<scalar>(1e-15));
            twoBykEps   = 2.0 / std::max(te.coeff, static_cast<scalar>(1e-15));
            boundedK    = tk.bounded;
            boundedEps  = te.bounded;
            if (saModel)
                std::printf("  div(phi,nuTilda): %s%s\n",
                            tk.bounded ? "bounded " : "",
                            tk.scheme.empty() ? "upwind" : tk.scheme.c_str());
            else
                std::printf("  div(phi,k): %s%s   div(phi,%s): %s%s\n",
                            tk.bounded ? "bounded " : "", tk.scheme.empty() ? "upwind" : tk.scheme.c_str(),
                            secondField.c_str(),
                            te.bounded ? "bounded " : "", te.scheme.empty() ? "upwind" : te.scheme.c_str());
        }

        // turbulence->validate() (simpleFoam.C:92), which is NOT a no-op: eddyViscosity::validate() is
        // correctNut(), so OpenFOAM enters its FIRST momentum assembly with nut rebuilt from the initial
        // k and omega rather than with whatever 0/nut holds. The tutorials ship `nut uniform 0`; on T3A
        // correctNut gives 1.8e-04 against nu 1.5e-05, so reading the file left brae's first momentum
        // equation THIRTEEN TIMES less viscous than OpenFOAM's -- a laminar first iteration under a
        // turbulent name. Measured against tools/dumpSimpleFoam at iteration 1: nuEff 9.2e-01, the
        // momentum diagonal 3.6e-01, rAU 3.8e-01, HbyA 4.9e-02. With this in place all four read 1e-13,
        // and one whole SIMPLEC step from OpenFOAM's converged field agrees to 8e-09 on the velocity
        // increment and 2.7e-07 on the pressure increment.
        //
        // The strain takes the case's own `grad(U)` scheme, exactly as the per-iteration correct() below
        // now does -- kOmegaSSTBase.C:132 is correctNut(2*magSqr(symm(fvc::grad(U)))), and fvc::grad
        // resolves the gradSchemes entry. Startup and the loop have to agree, so these move together.
        if (ras)
        {
            if (sstModel)
            {
                // correctNut(2*magSqr(symm(grad(U)))): nut = a1*k/max(a1*omega, b1*F23*sqrt(S2)),
                // kOmegaSSTBase.C. dEps carries omega on this model.
                DeviceBuffer<scalar> gradU, S2, F2;
                deviceGradU(dm, dbU, gf.Ux, gf.Uy, gf.Uz, gradU);
                if (gradUSchemeLimit > 0.0)
                    deviceCellLimitGradU(dm, dbU, gf.Ux, gf.Uy, gf.Uz, gradU, gradUSchemeLimit);
                deviceS2(gradU, static_cast<int>(nC), S2);
                deviceF2(dK, dEps, dY, nu, sstCoeffs, F2);
                deviceNutSST(dK, dEps, F2, S2, sstCoeffs, dNut);
            }
            else if (saModel)
            {
                // SpalartAllmaras::correctNut: nut = nuTilda*fv1(nuTilda/nu). nuTilda rides dK.
                deviceNutSA(dK, nu, saCoeffs.Cv1, dNut);
            }
            else if (keCoeffs.realizable)
            {
                DeviceBuffer<scalar> gradU, rCmu, magS;
                deviceGradU(dm, dbU, gf.Ux, gf.Uy, gf.Uz, gradU);
                if (gradUSchemeLimit > 0.0)
                    deviceCellLimitGradU(dm, dbU, gf.Ux, gf.Uy, gf.Uz, gradU, gradUSchemeLimit);
                deviceRealizableStrain(gradU, dK, dEps, keCoeffs.A0, static_cast<int>(nC), rCmu, magS);
                deviceRealizableNut(rCmu, dK, dEps, dNut);
            }
            else
            {
                deviceNut(dK, dEps, dNut, keCoeffs);   // kEpsilon: nut = Cmu*k^2/epsilon
            }
        }

        // Seed nuEff from the nut validate() has just rebuilt, so iteration 1 already uses it.
        {
            refreshBoundaryNut();
            const std::vector<scalar> nutC = dNut.host(), nutB = nb.host();
            for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nutC[c];
            std::size_t j = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (label i = 0; i < fvp[pi].size; ++i, ++j)
                    if (j < nutB.size()) nuEffB[pi][i] = nu + nutB[j];
            }
            const SurfaceScalarField nf2 = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
            dNuCell.copyFrom(nuEffC);
            dNuFace.copyFrom(nf2.internal);
            std::vector<scalar> flat;
            for (const auto& v : nuEffB) for (scalar x : v) flat.push_back(x);
            flat.resize(dm.nBndFaces, nu);
            dNuBnd.copyFrom(flat);

            // What the driver seeds nuEff's BOUNDARY with, against what the field itself carries. They
            // are the same quantity by two routes -- the wall-function recomputation here, and whatever
            // the previous correctNut left on nut -- and any gap is an input difference, not an assembly
            // one (the two momentum assemblies are bit-identical; see tests/ueqn_localize.cu).
            if (std::getenv("BRAE_DUMP_NUEFF"))
            {
                const std::vector<label> hm = spaldingIsWall.host();
                label nW = 0;
                for (label v : hm) nW += v;
                std::printf("  [nuEff] saModel %d  hasSpaldingWall %d  spalding faces %d of %zu\n",
                            (int)saModel, (int)hasSpaldingWall, (int)nW, hm.size());
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    if (isCoupledInterfaceType(fvp[pi].type)) continue;
                    const std::vector<scalar>& fb = nutF.boundary[pi]->value();
                    scalar mx = 0;
                    for (label i = 0; i < fvp[pi].size; ++i)
                        mx = std::fmax(mx, std::fabs(nuEffB[pi][i] - (nu + fb[i])));
                    scalar dmx = 0, fmx = 0;
                    for (label i = 0; i < fvp[pi].size; ++i)
                    {
                        dmx = std::fmax(dmx, nuEffB[pi][i] - nu);
                        fmx = std::fmax(fmx, fb[i]);
                    }
                    std::printf("  [nuEff] patch %-14s (%-7s) driver nut max %.6e  field nut max %.6e"
                                "  max|diff| %.6e\n",
                                fvp[pi].name.c_str(), fvp[pi].type.c_str(), dmx, fmx, mx);
                }
            }
        }

        in.correct = [&]()
        {
            // Phase timing, on demand: the wall-clock investigation needed to know how much of an outer
            // iteration is the turbulence hook (which round-trips nut to the host) versus the solve.
            const auto tHook0 = std::chrono::steady_clock::now();
            clearTurbulenceReport();
            // Solver settings from the CASE, exactly as for U and p. This was hardcoded to tol=1e-10
            // with relTol 0 and BiCGStab, so k and epsilon were driven to near machine precision every
            // outer iteration while the case asks for 1e-05 / relTol 0.1 / symGaussSeidel. Measured: the
            // turbulence hook was 168 ms of a ~300 ms outer iteration on 440k cells -- 56% of the run --
            // and almost none of that was the nuEff host copy (6 ms); it was over-solving these two.
            if (saModel)
            {
                // SpalartAllmarasBase::correct(): chi/fv1/fv2/ft2 -> Stilda -> the nuTilda transport
                // (div - laplacian(DnuTildaEff) - Cb2/sigmaNut*magSqr(grad) == Cb1*Stilda*nuTilda
                // - Sp(Cw1*fw*nuTilda/d^2)) -> bound(nuTilda, 0) -> correctNut. nuTilda rides dK.
                SpalartAllmarasCoeffs dsa;
                dsa.sigmaNut = saCoeffs.sigmaNut;
                dsa.kappa    = saCoeffs.kappa;
                dsa.Cb1      = saCoeffs.Cb1;
                dsa.Cb2      = saCoeffs.Cb2;
                dsa.Cw2      = saCoeffs.Cw2;
                dsa.Cw3      = saCoeffs.Cw3;
                dsa.Cv1      = saCoeffs.Cv1;
                dsa.Cs       = saCoeffs.Cs;
                dsa.E        = saCoeffs.E;
                deviceSpalartAllmarasCorrect(dm, dbU, dbK, gf.Ux, gf.Uy, gf.Uz,
                                             dK, dNut, dY, gf.phiInt, gf.phiBnd,
                                             nu, relaxK, tolKE,
                                             boundedK, limitedK, twoBykK,
                                             dsa, relTolKE, /*checkEvery*/1,
                                             // SpalartAllmaras's own laplacian takes the case's scheme
                                             // too. validation/airFoil2D names
                                             // `div(phi,nuTilda) bounded Gauss linearUpwind grad(nuTilda)`
                                             // beside `laplacianSchemes default Gauss linear corrected`,
                                             // and sa_cuda_vs_openfoam bounds nuTilda at 3e-02 with U at
                                             // 3e-04 -- so this false was live on a gated fixture.
                                             linearUpwindK, nonOrthLaplacian, gsKE,
                                             /*ami*/nullptr, /*cyc*/nullptr, /*ntDdt*/{},
                                             /*des*/false, /*iddes*/false, /*hmax*/nullptr,
                                             /*hwn*/nullptr, /*lesDelta*/nullptr, /*wallN*/nullptr,
                                             // fvc::grad(U) through the case's own scheme, which SA's
                                             // Omega is built from (SpalartAllmarasBase.C:103, :461).
                                             gradUSchemeLimit,
                                             // solvers/nuTilda/nSweeps -- validation/airFoil2D really
                                             // does ship `nSweeps 2` here.
                                             nSweepsKE);
            }
            else if (sstModel)
            {
                // Mirrors kOmegaSSTBase::correct(): GbyNu0 -> F1/F2/CDkOmega/S2 -> omega eqn (loose solve,
                // omega-wall setValues) -> bound -> k eqn -> bound -> correctNut (Bradshaw limiter).
                deviceKOmegaSSTCorrect(dm, wall, dbEps, dbK, dbU, gf.Ux, gf.Uy, gf.Uz,
                                       dK, dEps, dNut, dY, gf.phiInt, gf.phiBnd,
                                       nu, relaxEps, relaxK, tolKE,
                                       boundedK, boundedEps,
                                       limitedK, limitedEps, twoBykK, twoBykEps,
                                       sstCoeffs, relTolKE, /*keCheckEvery*/1,
                                       // The case's OWN turbulence div and laplacian schemes. These were
                                       // hardcoded off while the setup line above printed what the case
                                       // asked for -- so a case naming `linearUpwind` or `corrected` got
                                       // upwind and orthogonal. On the _cpp reference honouring
                                       // linearUpwind was worth a factor of 12 on T3A's end-to-end error.
                                       linearUpwindK, linearUpwindEps,
                                       // The case's own `grad(U)` scheme, which OpenFOAM's tgradU
                                       // takes (kOmegaSSTBase.C:522). This was a hardcoded 0.0 while
                                       // the driver printed the case's coefficient: on
                                       // windAroundBuildingsBox the production term GbyNu peaks at
                                       // 5.10e-01 unlimited against 1.05e-03 limited, a factor 487.
                                       nonOrthLaplacian, gradUSchemeLimit, gsKE, gsKE,
                                       /*ami*/nullptr, /*cyc*/nullptr,
                                       // kOmegaSSTLM: F1 = max(F1, F3), Pk *= gammaIntEff and
                                       // epsilonByk *= clamp(gammaIntEff, 0.1, 1). Null on plain SST.
                                       lmModel ? dGammaIntEff.data() : nullptr,
                                       static_cast<int>(v2NutWall), atmZ0, atmBoundNut,   // near-wall G0 uses the BC-chosen wall nut
                                       /*kDdt*/{}, /*sDdt*/{},
                                       /*des*/false, /*iddes*/false, /*hmax*/nullptr, /*hwn*/nullptr,
                                       /*rho*/nullptr, /*muLam*/nullptr,
                                       /*nuWallFace*/nullptr, /*rhoBnd*/nullptr,
                                       // DkEff(patchi)/DomegaEff(patchi) = alphaK(F1)*nut_b + nu, from
                                       // nut's OWN boundary rather than the interpolated cell value.
                                       &nb,
                                       /*muBnd*/nullptr,
                                       // `grad(k)`/`grad(omega)`, the transported scalars' own entries.
                                       gradScalarSchemeLimit,
                                       /*fvoKMask*/nullptr, /*fvoKVal*/nullptr,
                                       /*fvoEMask*/nullptr, /*fvoEVal*/nullptr,
                                       /*lesDelta*/nullptr,
                                       // solvers/k|omega/nSweeps -- the case's own sweep count.
                                       nSweepsKE);

            // OF kOmegaSSTLM::correct() runs kOmegaSST::correct() FIRST and the transition transport
            // SECOND, so k and omega always advance on the PREVIOUS iteration's gammaIntEff. Reversing
            // these two lines is a different (more implicit) scheme.
            if (lmModel)
                deviceKOmegaSSTLMCorrect(dm, dbU, dbReThetat, dbGammaInt, gf.Ux, gf.Uy, gf.Uz,
                                         dK, dEps, dNut, dY, dReThetat, dGammaInt, dGammaIntEff,
                                         gf.phiInt, gf.phiBnd, nu, relaxEps, tolKE, relTolKE,
                                         /*keCheckEvery*/1, boundedEps, nonOrthLaplacian, gsKE,
                                         /*ami*/nullptr, /*cyc*/nullptr, /*reDdt*/{}, /*giDdt*/{},
                                         limitedEps, linearUpwindEps);
            }
            else
            deviceKEpsilonCorrect(dm, wall, dbEps, dbK, dbU, gf.Ux, gf.Uy, gf.Uz,
                                  dK, dEps, dNut, gf.phiInt, gf.phiBnd,
                                  nu, relaxEps, relaxK, tolKE,
                                  boundedK, boundedEps,
                                  limitedK, limitedEps, twoBykK, twoBykEps,
                                  keCoeffs, relTolKE, /*keCheckEvery*/1,
                                  // The case's OWN turbulence div and laplacian schemes. These were three
                                  // hardcoded `false`s while the SST call forty lines above forwarded the
                                  // very same variables, and while the driver printed what the case asked
                                  // for. Measured on validation/pitzDaily at 1000 iterations with every
                                  // linear solve pinned: brae reproduced the SUBSTITUTED equations to
                                  // eleven digits (U 1.29e-11, k 3.01e-11 against OpenFOAM run with
                                  // upwind + orthogonal) while sitting U 5.98e-03 / k 2.08e-02 from
                                  // OpenFOAM running the case's own dictionary -- a ratio of 4.6e+08. The
                                  // written k, epsilon, nut, U and p were BYTE-IDENTICAL with
                                  // `linearUpwind` and with `upwind` on div(phi,k), 5 iterations of 5.
                                  linearUpwindK, linearUpwindEps, nonOrthLaplacian,
                                  gsKE, gsKE,
                                  /*ami*/nullptr, /*cyc*/nullptr,
                                  static_cast<int>(v2NutWall), atmZ0, atmBoundNut,   // near-wall G0 uses the BC-chosen wall nut
                                  /*kDdt*/{}, /*eDdt*/{},
                                  /*rho*/nullptr, /*muLam*/nullptr,
                                  /*rhoBnd*/nullptr, /*nuWallFace*/nullptr,
                                  // DkEff(patchi)/DepsilonEff(patchi) from nut's OWN boundary.
                                  &nb,
                                  /*muBnd*/nullptr,
                                  gradScalarSchemeLimit,
                                  // kEpsilon::correct builds GbyNu from fvc::grad(U) (kEpsilon.C:237),
                                  // which resolves the case's `grad(U)` entry. These three arguments
                                  // used to fall through to their defaults, so the limiter never
                                  // reached the production term at all.
                                  gradUSchemeLimit,
                                  /*fvoKMask*/nullptr, /*fvoKVal*/nullptr,
                                  /*fvoEMask*/nullptr, /*fvoEVal*/nullptr,
                                  // solvers/k|epsilon/nSweeps -- the case's own sweep count.
                                  nSweepsKE);

            // OpenFOAM prints an Initial residual for every turbulence solve, and comparing those against
            // its log is how this path is checked against the oracle. Off unless asked for, so that the
            // gates that parse this log keep seeing the output they were written against.
            if (printTurbResid)
            {
                for (const ScalarSolveEntry& e : turbulenceReport())
                {
                    std::printf("    Solving for %s, Initial residual = %.9e, No Iterations %d\n",
                                e.field.c_str(), e.perf.initialResidual, e.perf.nIterations);
                }
            }

            const auto tRefresh0 = std::chrono::steady_clock::now();
            // nuEff for the NEXT iteration: nu + nut, with the boundary value from the wall function --
            // NOT the owner cell's. That distinction is the defect that once made boundary viscosity
            // 2000x too small; deviceBoundaryNut is what applies the wall function per face.
            refreshBoundaryNut();

            const std::vector<scalar> nutC = dNut.host(), nutB = nb.host();
            for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nutC[c];
            std::size_t j = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (label i = 0; i < fvp[pi].size; ++i, ++j)
                    if (j < nutB.size()) nuEffB[pi][i] = nu + nutB[j];
            }
            const SurfaceScalarField nf2 = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
            dNuCell.copyFrom(nuEffC);
            dNuFace.copyFrom(nf2.internal);
            std::vector<scalar> flat;
            for (const auto& v : nuEffB) for (scalar x : v) flat.push_back(x);
            flat.resize(dm.nBndFaces, nu);
            dNuBnd.copyFrom(flat);
            const auto tNow = std::chrono::steady_clock::now();
            g_hookSeconds    += std::chrono::duration<double>(tNow - tHook0).count();
            g_refreshSeconds += std::chrono::duration<double>(tNow - tRefresh0).count();
        };
    }

    in.nuEffCell = &dNuCell; in.nuEffFace = &dNuFace; in.nuEffBndFace = &dNuBnd;
    in.relaxU = relaxU; in.relaxP = relaxP;

    // ---- linear-solver controls, from the case ------------------------------------------------
    // These were HARDCODED at tolerance 1e-10 / relTol 0, which is four to five orders tighter than any
    // case asks for. SIMPLE deliberately solves each inner system LOOSELY -- pitzDaily says
    // `relTol 0.1` -- because the outer iteration is what converges, not the inner solve. Ignoring that
    // does not give a wrong answer, it gives the right answer for roughly ten times the linear-algebra
    // work: measured 614 fine-grid SpMV per SIMPLE iteration against the existing solver's 62.
    //
    // maxIter and minIter follow OpenFOAM's lduMatrix::solver defaults (1000 and 0) and are READ FROM
    // EACH FIELD'S OWN ENTRY, because lduMatrix::solver::readControls does (lduMatrixSolver.C:190-208)
    // and because an `maxIter 10` in a case is a cap on the ANSWER, not a performance hint: two solvers
    // stopped by a cap hold two different residuals. This used to read maxIter from the `p` entry alone
    // and hand it to the momentum solve as well: validation/T3A caps
    // `"(U|k|omega|gammaInt|ReThetat)"` at 10 and says nothing about p, so brae ran U at 1000 against
    // OpenFOAM's 10. Honouring it does NOT change T3A's fate -- measured cold, iteration 100 reads
    // 2.53e-02 against 2.58e-02 before, and the case still leaves its basin by 200. The 5.2e-04 that an
    // earlier experiment saw came from capping `p` at 10, which OpenFOAM does not do; that is a lead
    // about brae's pressure step, not a fidelity fix, and it is not applied here.
    if (const FoamDict* solvers = fvSolution.subDict("solvers"))
    {
        // subDict resolves OpenFOAM regex keys, so `"(U|k|epsilon|omega|f|v2)"` is found by "U".
        if (const FoamDict* sp = solvers->subDict("p"))
        {
            in.tolP     = sp->scalarOr("tolerance", in.tolP);
            in.relTolP  = sp->scalarOr("relTol", 0.0);
            in.maxIterP = static_cast<int>(sp->scalarOr("maxIter", 1000));
            in.minIterP = static_cast<int>(sp->scalarOr("minIter", 0));
        }
        // k and epsilon take their own entry, which in most tutorials is the same regex key as U's.
        // The TRANSPORTED field's own entry, not the literal "k". SpalartAllmaras has no k, so this
        // block was skipped entirely on an SA case and tolerance, relTol and the smoothSolver selection
        // all stayed at brae's defaults -- validation/airFoil2D asks for `smoothSolver` with a
        // GaussSeidel smoother and `nSweeps 2` on nuTilda and got Jacobi-BiCGStab at brae's own
        // tolerance, silently. `subDict` resolves OpenFOAM's regex keys, so a case writing
        // `"(U|k|omega)"` is still found by whichever name the model carries.
        const FoamDict* sk = solvers->subDict(saModel ? "nuTilda" : "k");
        if (sk)
        {
            tolKE    = sk->scalarOr("tolerance", tolKE);
            relTolKE = sk->scalarOr("relTol", 0.0);
            gsKE     = (sk->wordOr("solver", "") == "smoothSolver" &&
                        (sk->wordOr("smoother", "") == "symGaussSeidel" ||
                         sk->wordOr("smoother", "") == "GaussSeidel"));
            // nSweeps for the transported scalars. `subDict` resolves OpenFOAM's regex keys, so a case
            // writing `"(U|k|omega)"` is found by either name; the cross-check below therefore only
            // fires when the case really did give the two fields separate entries with different
            // values. A NEGATIVE nSweeps is refused for the same reason as on U: smoothSolver.C:96-119
            // deletes the convergence test and reports a residual of ZERO, which residualControl reads
            // as converged.
            const std::string firstFld  = saModel ? "nuTilda" : "k";
            const std::string secondFld = sstModel ? "omega" : "epsilon";
            nSweepsKE = static_cast<int>(sk->scalarOr("nSweeps", 1));
            if (nSweepsKE < 0)
                throw std::runtime_error(
                    "system/fvSolution asks for `nSweeps " + std::to_string(nSweepsKE) + "` on "
                    + firstFld + ". A negative nSweeps is OpenFOAM's FIXED-SWEEP mode "
                    "(smoothSolver.C:96-119): it deletes the convergence test and reports a residual of "
                    "zero, which residualControl reads as converged. brae has no such mode.");
            if (nSweepsKE < 1) nSweepsKE = 1;
            if (!saModel)
                if (const FoamDict* ss = solvers->subDict(secondFld))
                {
                    const int n2 = static_cast<int>(ss->scalarOr("nSweeps", 1));
                    if (n2 != nSweepsKE)
                        throw std::runtime_error(
                            "system/fvSolution sets `nSweeps` to " + std::to_string(nSweepsKE) + " on "
                            + firstFld + " and " + std::to_string(n2) + " on " + secondFld
                            + ". This driver carries one sweep count for the transported pair, so "
                              "running would apply one field's smoothSolver setting under the other's "
                              "name.");
                }
            if (keCoeffs.realizable)
            std::printf("  RASModel realizableKE: variable Cmu (A0=%g C2=%g sigmaEps=%g)\n",
                        keCoeffs.A0, keCoeffs.C2, keCoeffs.sigmaEps);
        std::printf("  k/epsilon solves: tol=%.1e relTol=%.3g  solver=%s\n",
                    tolKE, relTolKE,
                    gsKE ? "smoothSolver + MULTICOLOUR symGaussSeidel (OpenFOAM sweeps in index order)"
                         : "Jacobi-BiCGStab");
        }
        if (const FoamDict* su = solvers->subDict("U"))
        {
            in.tolU     = su->scalarOr("tolerance", in.tolU);
            in.relTolU  = su->scalarOr("relTol", 0.0);
            in.maxIterU = static_cast<int>(su->scalarOr("maxIter", 1000));
            in.minIterU = static_cast<int>(su->scalarOr("minIter", 0));
            // nSweeps (smoothSolver.C:78, default 1). A NEGATIVE value is a different solver, not a
            // tuning of this one: smoothSolver.C:96-119 smooths -nSweeps times with no normFactor, no
            // residual and no convergence test, and reports a residual of ZERO -- which residualControl
            // then reads as converged (measured on validation/airFoil2D: "SIMPLE solution converged in
            // 1 iterations"). Substituting a convergence-tested loop there would change how many outer
            // iterations the whole run takes, i.e. the answer, so it is refused rather than approximated.
            const int nSweepsU = static_cast<int>(su->scalarOr("nSweeps", 1));
            if (nSweepsU < 0)
                throw std::runtime_error(
                    "system/fvSolution asks for `nSweeps " + std::to_string(nSweepsU) + "` on U. A "
                    "negative nSweeps is OpenFOAM's FIXED-SWEEP mode (smoothSolver.C:96-119): it deletes "
                    "the convergence test and reports a residual of zero, which residualControl reads as "
                    "converged. brae has no such mode, and running its convergence-tested loop instead "
                    "would change how many SIMPLE iterations the case takes.");
            in.nSweepsU = (nSweepsU > 0) ? nSweepsU : 1;
            // OpenFOAM's SELECTION, exactly: `solver smoothSolver` + a GaussSeidel-family smoother.
            // The selection is matched; the ALGORITHM is not -- brae's sweep is multicolour where
            // OpenFOAM's is index order (see the envelope notice above, and device_amg.cuh). Anything
            // else (PBiCG[Stab], GAMG, ...) keeps Jacobi-BiCGStab, announced below as the substitution
            // it is.
            const std::string usolv = su->wordOr("solver", "");
            const std::string usm   = su->wordOr("smoother", "");
            in.uSymGaussSeidel = (usolv == "smoothSolver" &&
                                  (usm == "symGaussSeidel" || usm == "GaussSeidel"));
            if (!in.uSymGaussSeidel && !usolv.empty())
                std::printf("NOTICE (simpleFoam v2): system/fvSolution asks for `%s` on U; brae runs a "
                            "Jacobi-preconditioned BiCGStab instead. Same matrix, different Krylov "
                            "method and iteration count.\n", usolv.c_str());
        }
    }
    // minIter is printed beside maxIter because it is the same kind of statement: a floor on how far the
    // solve runs, read from the field's own entry, and a case that sets one has said something about the
    // answer it wants.
    std::printf("  linear solves: p tol=%.1e relTol=%.3g maxIter=%d minIter=%d   "
                "U tol=%.1e relTol=%.3g maxIter=%d minIter=%d\n",
                in.tolP, in.relTolP, in.maxIterP, in.minIterP,
                in.tolU, in.relTolU, in.maxIterU, in.minIterU);
    // Both are pure execution strategy -- same matrix, same stopping criterion -- so they are on by
    // default and env-overridable for A/B measurement rather than being case settings.
    // The AMG hierarchy is already reused across ITERATIONS (SolverWorkspace::amgBuilt); this reuses it
    // across RUNS. Opt-in, because it writes a file into the case.
    if (const char* e = std::getenv("BRAE_AMG_CACHE"))
        if (e[0] == '1') in.amgCacheDir = caseDir + "/constant/polyMesh";
    if (const char* e = std::getenv("BRAE_VCYCLE_GRAPH")) in.captureVcycle = (e[0] == '1');
    if (const char* e = std::getenv("BRAE_PCG_CHECK_EVERY")) in.pcgCheckEvery = std::max(1, std::atoi(e));
    std::printf("  pressure: V-cycle graph %s, residual read every %d PCG iteration(s)\n",
                in.captureVcycle ? "ON" : "off", in.pcgCheckEvery);
    // ---- fvOptions: upload the porous zone ----------------------------------------------------
    // The device kernel takes DIAGONAL d/f and applies the 0.5 itself, so the RAW f goes across and a
    // ROTATED coordinate system has to be refused rather than silently flattened -- dropping the
    // off-diagonals of a rotated D would apply the resistance along the wrong axes.
    DevicePorosity porosity;
    {
        const cpu::fvOptions::OptionList ol = cpu::fvOptions::read(caseDir, m);
        for (const auto& o : ol.options)
        {
            if (!o.active || !o.unsupported.empty()) continue;
            if (o.rotorDisk || o.actuationDisk) continue;   // built below, from readFvOptions + the mesh
            const scalar offD = std::fabs(o.D.xy) + std::fabs(o.D.xz) + std::fabs(o.D.yz)
                              + std::fabs(o.D.yx) + std::fabs(o.D.zx) + std::fabs(o.D.zy);
            const scalar offF = std::fabs(o.F.xy) + std::fabs(o.F.xz) + std::fabs(o.F.yz)
                              + std::fabs(o.F.yx) + std::fabs(o.F.zx) + std::fabs(o.F.zy);
            const scalar sc = std::fabs(o.D.xx) + std::fabs(o.D.yy) + std::fabs(o.D.zz) + 1.0;
            if (offD + offF > 1e-10*sc)
                throw std::runtime_error(
                    "fvOptions `" + o.name + "`: the DarcyForchheimer coordinateSystem is rotated, so D "
                    "and F have off-diagonal entries. The device porosity kernel takes diagonal "
                    "coefficients only; refusing rather than dropping them.");
            porosity.active = true;
            porosity.cells.copyFrom(o.cells);
            porosity.d = vector{o.D.xx, o.D.yy, o.D.zz};
            // F already carries the 0.5 (calcTransformModelData); the kernel applies it again, so the
            // RAW f is what it wants back.
            porosity.f = vector{2.0*o.F.xx, 2.0*o.F.yy, 2.0*o.F.zz};
            std::printf("  fvOptions `%s`: explicitPorositySource/DarcyForchheimer on %d cells, "
                        "d=(%.4g %.4g %.4g)\n", o.name.c_str(), (int)o.cells.size(),
                        porosity.d.x, porosity.d.y, porosity.d.z);
        }
    }
    in.porosity  = porosity.active ? &porosity : nullptr;

    // rotorDiskSource. The parameters come from readFvOptions (the same reader the shipped solver uses)
    // and the per-cell geometry from the mesh. The force itself is the blade-element calculation gated
    // against OpenFOAM's own reported drag/lift/AOA in tests/rotordisk_vs_openfoam.sh.
    DeviceRotorDisk     rotor;
    DeviceActuationDisk actuationDisk;
    {
        std::map<std::string, std::vector<label>> rzones;
        {
            std::ifstream a(caseDir + "/constant/polyMesh/cellZones");
            std::ifstream b(caseDir + "/constant/polyMesh/cellZones.gz");
            if (a.good() || b.good()) rzones = readCellZones(caseDir + "/constant/polyMesh");
        }
        const FvOptionsData fvo = readFvOptions(caseDir, rzones, g.V(), nC, g.C());
        // A source brae recognises by name but cannot apply here would otherwise run as if it were not
        // in the case at all -- a turbine site with no turbines converges perfectly well.
        if (!fvo.unsupported.empty())
        {
            std::string msg = "brae: fvOptions this driver cannot apply:";
            for (const std::string& u : fvo.unsupported) msg += "\n  - " + u;
            throw std::runtime_error(msg);
        }
        if (fvo.rotor.active)
        {
            rotor = buildDeviceRotorDisk(fvo.rotor, g.C(), g.Sf(), m.owner(), m.neighbour(),
                                         m.nInternalFaces());
            std::printf("  fvOptions rotorDisk: %d cells, rMax %.4g, omega %.4g rad/s, %d blades\n",
                        rotor.n, rotor.rMax, rotor.omega, (int)rotor.nBlades);
        }
        if (fvo.adActive)
        {
            actuationDisk = buildDeviceActuationDisk(fvo, dm.V, nC);
            for (std::size_t di = 0; di < actuationDisk.disks.size(); ++di)
            {
                const auto& d = actuationDisk.disks[di];
                std::printf("  fvOptions actuationDisk %zu: area %.4g, a %.6g, disk volume %.6g, "
                            "%d monitor cell(s)\n", di + 1, d.area, d.a, d.vtot, (int)d.nmon);
            }
        }
    }
    in.rotor         = rotor.active ? &rotor : nullptr;
    in.actuationDisk = actuationDisk.active ? &actuationDisk : nullptr;

    in.nuLaminar = nu;
    // The gradSchemes `grad(U)` entry, a SEPARATE lookup from linearUpwind's named gradient above. It is
    // printed on its own line because the two can differ (validation/rotorDisk) and because the single
    // line this replaced named the SCHEME while carrying linearUpwind's value -- the display string
    // asserting a capability the code did not have.
    in.gradUSchemeLimitK = gradUSchemeLimit;
    if (gradUSchemeLimit > 0.0)
        std::printf("  gradSchemes grad(U): cellLimited Gauss linear %g "
                    "(fvc::grad(U): divDevReff + the closures)\n", gradUSchemeLimit);

    std::printf("  U solver: %s\n",
                // "as the case asks" was the display string this line used to carry, and it was the
                // shared-capability-notice defect: the SELECTION is the case's, the sweep order is not.
                in.uSymGaussSeidel ? "smoothSolver + MULTICOLOUR symGaussSeidel (OpenFOAM sweeps in index order)"
                                   : "Jacobi-BiCGStab");
    in.momentumPredictor = cd.momentumPredictor;
    in.nNonOrthogonalCorrectors = cd.nNonOrthogonalCorrectors;
    in.pRefCell = f.pRefCell; in.pRefValue = f.pRefValue;
    in.takeUAtBoundary = &dTakeU; in.adjustable = &dAdjust;
    in.consistent = cd.consistent;
    if (in.consistent)
        std::printf("  SIMPLE/consistent: SIMPLEC, rAtU = 1/(1/rAU - UEqn.H1())\n");
    // `bounded Gauss <scheme>`: -fvm::Sp(fvc::div(phi), U), implemented on both paths and matched to
    // 2.9e-16 (tests/test_ueqn_cuda.cu), so it is READ rather than refused.
    {
        const std::string sc = divUScheme(caseDir);
        in.scheme = sc == "linearUpwind"   ? cpu::DivScheme::linearUpwind
                  : sc == "linearUpwindV"  ? cpu::DivScheme::linearUpwindV
                  : sc == "limitedLinear"  ? cpu::DivScheme::limitedLinear
                  : sc == "limitedLinearV" ? cpu::DivScheme::limitedLinearV
                  : sc == "LUST"           ? cpu::DivScheme::LUST
                                           : cpu::DivScheme::upwind;
        in.linearUpwind = (in.scheme == cpu::DivScheme::linearUpwind);
        in.schemeCoeff = divUSchemeCoeff(caseDir, 1.0);
        // The gradient linearUpwind NAMES: `cellLimited Gauss linear <k>` limits it, and the envelope
        // above has already refused any other resolution. Measured on windAroundBuildings, running the
        // plain Gauss gradient where the case names cellLimited leaves the momentum residual 272x
        // OpenFOAM's own instead of 1.4x -- the correction does not vanish at convergence.
        linearUpwindGradUnsupported(caseDir, &in.gradULimitK);
        if (in.gradULimitK > 0.0)
            std::printf("  div(phi,U) %s's named gradient: cellLimited Gauss linear %g\n",
                        sc.empty() ? "linearUpwind" : sc.c_str(), in.gradULimitK);
        std::printf("  div(phi,U) scheme: %s", sc.empty() ? "upwind" : sc.c_str());
        if (in.scheme == cpu::DivScheme::limitedLinear || in.scheme == cpu::DivScheme::limitedLinearV)
            std::printf(" %g", in.schemeCoeff);
        std::printf("\n");
    }
    in.bounded = divUBounded(caseDir);
    if (in.bounded) std::printf("  div(phi,U) is bounded: applying -fvm::Sp(fvc::div(phi), U)\n");

    // The non-orthogonal correction, both halves: nonOrthDeltaCoeffs in the implicit coefficients and the
    // deferred correction in the source, in the momentum equation and the pressure equation alike. Matched
    // to the reference to 5e-16 (tests/test_ueqn_cuda.cu, tests/test_peqn_cuda.cu) and gated against real
    // OpenFOAM on a non-orthogonal mesh (tests/nonorth_vs_openfoam.sh), so it is READ rather than refused.
    {
        in.correctedLaplacian = nonOrthLaplacian;
        in.snGradLimitCoeff   = snGradLimitK;
        if (snGradLimitK > 0.0)
            std::printf("  laplacianSchemes: `limited %g corrected` -- the non-orthogonal correction is "
                        "capped against the orthogonal part\n", snGradLimitK);
    }
    std::printf("  laplacianSchemes: non-orthogonal correction %s\n",
                in.correctedLaplacian ? "ON (`corrected`)" : "OFF (`uncorrected`/`orthogonal`)");

    // ---- the SIMPLE loop ---------------------------------------------------------------------
    SolverWorkspace ws;
    std::map<std::string, scalar> residuals;
    label iter = 0;
    scalar timeValue = startTimeVal;
    while (ctl.loop(iter + 1, nSteps, residuals))
    {
        ++iter;
        timeValue += deltaT;   // Time::operator++ ACCUMULATES the value, it does not multiply (Time.C:1067)
        // inletOutlet: resolve each face to fixedValue|zeroGradient from the PREVIOUS iteration's
        // boundary flux, as OF's updateCoeffs does. The rebuilt driver did NONE of this -- the mask was
        // built once and never consulted -- so an inletOutlet outlet stayed pinned at its inletValue with
        // full convection and laplacian coupling where OF switches to zeroGradient on outflow. On
        // airFoil2D that is nuTilda's whole far field (`freestream` derives from inletOutlet), and the
        // run diverged to nuTilda ~1e+36 while residualControl still reported convergence.
        deviceUpdateInletOutlet(dbU, gf.phiBnd);
        deviceUpdateInletOutlet(dbP, gf.phiBnd);
        if (ras)
        {
            deviceUpdateInletOutlet(dbK, gf.phiBnd);
            if (!saModel) deviceUpdateInletOutlet(dbEps, gf.phiBnd);
        }

        // OF freestreamVelocity/freestreamPressure updateCoeffs: valueFraction = 0.5 -/+ 0.5*(U.n)/|U|,
        // rebuilt from the (lagged) flow angle every iteration. Leaving it at the 0.5 seed makes every
        // far-field face a half-and-half blend whether it is inflow or outflow; on airFoil2D, whose whole
        // far field is freestream, that and the mixed evaluate() blend were worth 372x -> 47x on the
        // momentum residual and 3551x -> 136x on pressure.
        if (hasMixedBC)
            deviceUpdateMixedFreestream(dbU, dbP, gf.phiBnd, gf.Ux, gf.Uy, gf.Uz, nullptr);

        residuals = simpleStep(gf, ws, dm, dbU, dbP, in);
        // OF prints the TIME NAME (simpleFoam.C:100, runTime.timeName()), not the iteration index. The
        // two coincide only at startTime 0 with deltaT 1 -- which every fixture in validation/ happens
        // to be, so no gate could see this.
        std::printf("Time = %s   U initial residual = %.6e   p initial residual = %.6e\n",
                    WriteControl::timeName(timeValue).c_str(),
                    residuals.count("U") ? residuals.at("U") : 0.0,
                    residuals.count("p") ? residuals.at("p") : 0.0);
    }
    if (ctl.converged())
        std::printf("SIMPLE solution converged in %d iterations\n", static_cast<int>(iter));
    if (std::getenv("BRAE_PHASE_TIME"))
        std::printf("  [phase] turbulence hook %.3f s (%.1f ms/iter), of which the nuEff host "
                    "round-trip %.3f s (%.1f ms/iter), over %d iterations\n",
                    g_hookSeconds,    iter ? 1e3 * g_hookSeconds    / (double)iter : 0.0,
                    g_refreshSeconds, iter ? 1e3 * g_refreshSeconds / (double)iter : 0.0,
                    static_cast<int>(iter));

    // ---- write -------------------------------------------------------------------------------
    {
        // OF names a time directory by its TIME VALUE (Time::timeName, Time.C:721-728), not by an
        // iteration index. Measured: startTime 0 / endTime 2.5 / deltaT 0.5 -- OpenFOAM writes 2.5/,
        // this driver wrote 2/.
        const std::string outDir = caseDir + "/" + WriteControl::timeName(startTimeVal + iter * deltaT);
        std::filesystem::create_directories(outDir);
        const std::string src = caseDir + "/" + startTime;

        const std::vector<scalar> pOut = gf.p.host();
        const std::vector<scalar> ux = gf.Ux.host(), uy = gf.Uy.host(), uz = gf.Uz.host();
        std::vector<vector> UOut(nC);
        for (label c = 0; c < nC; ++c) UOut[c] = {ux[c], uy[c], uz[c]};

        // THE SOLVED BOUNDARY VALUES, not the start directory's. Every call here used to omit
        // writeVolField's `computedBoundary` argument, which makes it ECHO the template it read -- so a
        // V2 time directory carried the 0/ boundary on every patch the solve had moved. Measured on
        // backwardFacingStep2D: the walls' nut came out `value uniform 0` (the 0/ placeholder) where
        // the legacy driver writes the solved `nonuniform List<scalar>`, i.e. the wall function's
        // viscosity was absent from the output. A restart from such a directory restarts from the wrong
        // state, and any wall post-processing reads a zero that was never the answer.
        DeviceBuffer<scalar> pB, uxB, uyB, uzB;
        deviceBCValue(dbP, gf.p, pB);
        deviceBCValue(dbU.comp[0], gf.Ux, uxB);
        deviceBCValue(dbU.comp[1], gf.Uy, uyB);
        deviceBCValue(dbU.comp[2], gf.Uz, uzB);
        const std::vector<scalar> hpB = pB.host();
        const std::vector<scalar> hxB = uxB.host(), hyB = uyB.host(), hzB = uzB.host();
        std::vector<vector> UB(hxB.size());
        for (std::size_t i = 0; i < hxB.size(); ++i) UB[i] = vector{hxB[i], hyB[i], hzB[i]};

        writeVolField<scalar>(src + "/p", outDir + "/p", pOut, fvp, 16, hpB);
        writeVolField<vector>(src + "/U", outDir + "/U", UOut, fvp, 16, UB);
        if (ras)
        {
            // SpalartAllmaras transports nuTilda in the k slot and has no second field.
            const std::string first = saModel ? "nuTilda" : "k";
            DeviceBuffer<scalar> kB;
            deviceBCValue(dbK, dK, kB);
            writeVolField<scalar>(src + "/" + first, outDir + "/" + first, dK.host(), fvp, 16, kB.host());
            if (!saModel)
            {
                DeviceBuffer<scalar> eB;
                deviceBCValue(dbEps, dEps, eB);
                writeVolField<scalar>(src + "/" + secondField, outDir + "/" + secondField,
                                      dEps.host(), fvp, 16, eB.host());
            }
            // nut's boundary is NOT a BC evaluation: the wall faces carry what the wall function
            // computed, which is exactly what `nb` holds (deviceBoundaryNut writes the whole boundary
            // there). Evaluating dbNut instead would put the `calculated` placeholder back on the walls.
            writeVolField<scalar>(src + "/nut", outDir + "/nut", dNut.host(), fvp, 16,
                                  nb.size() ? nb.host() : std::vector<scalar>());
        }
        // The second field is `omega` under kOmegaSST; the file was always written under its right name,
        // but this line claimed `epsilon` for every model.
        const std::string turbFields =
            ras ? (saModel ? std::string(",nuTilda,nut") : ",k," + secondField + ",nut") : std::string();
        std::printf("written %s/{U,p%s}\n", outDir.c_str(), turbFields.c_str());
    }
    return static_cast<int>(iter);
}

} // namespace gpu
} // namespace brae
