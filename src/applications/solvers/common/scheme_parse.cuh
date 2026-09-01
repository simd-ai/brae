#pragma once
// Shared fvSchemes div/laplacian/grad scheme parse -> DeviceSimpleControls flags. Used by BOTH the steady
// (gpuSimpleFoam) and transient (gpuPimpleFoam) drivers so the scheme detection lives in ONE place. Reads
// system/fvSchemes ($-expanded): div(phi,U) bounded/linearUpwind[V]/LUST + div(phi,{k,epsilon,omega,nuTilda})
// limitedLinear/linearUpwind, laplacian/snGrad corrected/limited (non-orth), grad(U) cellLimited. Extracted
// verbatim from gpuSimpleFoam; throws (OF-style) on a missing/unsupported div(phi,U) scheme.
#include "solver_controls.cuh"
#include "foam_dict.cuh"    // readFileExpanded
#include "brae_notice.cuh"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <sstream>
#include <utility>
#include <vector>
#include <stdexcept>
#include <set>
#include <string>

namespace brae {

// system/fvSchemes, $-expanded, with OpenFOAM's `fusedGauss` family rewritten to `Gauss`.
//
// fusedGauss is NOT a different discretisation. src/fused/finiteVolume/ holds fusedGaussConvectionScheme,
// fusedGaussDivScheme, fusedGaussGrad and fusedGaussLaplacianScheme, and each is the plain Gauss scheme
// with the field-expression temporaries replaced by fused loops -- OpenFOAM even leaves the original
// lines in as comments right above the fused calls:
//     //fvm.lower() = -weights.primitiveField()*faceFlux.primitiveField();
//     multiplySubtract(fvm.lower(), weights.primitiveField(), faceFlux.primitiveField());
// fvmDiv, fvcDiv, calcGrad and fvmLaplacian all compute the same numbers as their Gauss counterparts;
// fusedGaussLaplacianScheme::fvmLaplacian is line-for-line identical to gaussLaplacianScheme's.
// pitzDaily_fused is stock pitzDaily with `libs (fusedFiniteVolume)` and the schemes renamed.
//
// So this is a rename, not an approximation -- but it is ANNOUNCED rather than silently aliased, because
// "brae ran a scheme the case did not name" is exactly the class of substitution this codebase refuses.
// Only the leading scheme word is rewritten; whatever interpolation follows is parsed as it always was,
// so `fusedGauss <something brae cannot do>` is still refused by name.
inline std::string readFvSchemesText(const std::string& caseDir)
{
    std::string text = readFileExpanded(caseDir + "/system/fvSchemes");
    if (text.find("fusedGauss") == std::string::npos) return text;

    static bool announced = false;
    if (!announced)
    {
        announced = true;
        noticeEquivalent("system/fvSchemes",
                   "the case names OpenFOAM's `fusedGauss` schemes; brae runs `Gauss`. fusedGauss is the "
                   "same discretisation written without field temporaries (src/fused/finiteVolume), so "
                   "the matrices are identical -- it is a performance variant, not a numerical one.");
    }
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); )
    {
        // Only a WHOLE token: a key like `div(fusedGaussThing)` must not be rewritten underneath us.
        const bool boundaryBefore = (i == 0) || !(std::isalnum(static_cast<unsigned char>(text[i-1]))
                                                 || text[i-1] == '_');
        if (boundaryBefore && text.compare(i, 10, "fusedGauss") == 0)
        {
            const std::size_t j = i + 10;
            const bool boundaryAfter = (j >= text.size())
                                    || !(std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '_');
            if (boundaryAfter)
            {
                out += "Gauss";
                i = j;
                continue;
            }
        }
        out += text[i++];
    }
    return out;
}

// Fill ctl's convection/laplacian/grad scheme flags from caseDir/system/fvSchemes.
// Every div(...) statement brae actually CONSUMED, recorded at the point of consumption rather than
// from a hand-written list. That distinction is the whole point: a maintained list drifts, and a stale
// entry marks an ignored input as read -- a false NEGATIVE in the audit, which is worse than no audit.
// checkDiv is a one-to-one choke point (six div branches, six calls), so this cannot miss one.
inline std::set<std::string>& divSchemesConsumed()
{
    static std::set<std::string> s;
    return s;
}

// The `div(...)` token of a statement, paren-balanced so div((nuEff*dev2(T(grad(U))))) survives intact.
inline std::string divKeyOf(const std::string& ln)
{
    const std::size_t b = ln.find("div(");
    if (b == std::string::npos) return std::string();
    int depth = 0;
    for (std::size_t i = b + 3; i < ln.size(); ++i)
    {
        if (ln[i] == '(') ++depth;
        else if (ln[i] == ')') { if (--depth == 0) return ln.substr(b, i - b + 1); }
    }
    return std::string();
}

// The div scheme for ONE arbitrary field, resolved the way OF's scalarTransport does:
//
//     word divScheme("div(phi," + schemesField_ + ")");        scalarTransport.C:249
//
// so a tracer is discretised by the case's OWN entry, not by whatever the solver happens to use for U.
// Hardcoding it instead produced a measured 5.59 on a passive tracer bounded by 1.0 in pitzDaily's shear
// layer -- an unbounded central scheme where the case never asked for one.
//
// ABSENCE IS AN ERROR when divSchemes `default` is `none`, which is what OF does: a case with
// `default none` and no div(phi,tracer0) entry does not run at all. brae ran it and produced a
// plausible wrong answer, which is worse than refusing.
struct FieldDivScheme
{
    bool   bounded      = false;
    bool   limited      = false;   // limitedLinear
    bool   linearUpwind = false;
    // TWO currencies for the same `limitedLinear <k>` coefficient, because the two consumers transform
    // it in different places: the device kernels take twoByk pre-computed (solvePassiveScalar,
    // deviceSolveScalarTransport), while the host weights functions take the RAW k and compute
    // 2/max(k,SMALL) themselves (limitedSchemes_cpp.cu:53). Handing twoByk where raw is expected runs
    // limitedLinear 2 under the case's limitedLinear 1 -- the limiter becomes clamp01(1*r) instead of
    // clamp01(2*r) -- which three rho harnesses did until the turbulence-scheme port made it visible.
    scalar coeff        = 1.0;     // the raw k of `limitedLinear k` -- host weights functions
    scalar twoByk       = 0;       // 2/max(k,SMALL)                 -- device kernels
    // laplacian(D<field>,<field>) -- OF scalarTransport.C:250, where Dname = "D" + field name.
    // `corrected`/`limited` -> non-orthogonal correction on; `orthogonal`/`uncorrected` -> off.
    // Unlike divSchemes, laplacianSchemes almost always carries a usable `default`, which OF resolves
    // normally, so absence falls back to it rather than refusing.
    bool   nonOrth      = true;
};

// Extract the brace-delimited body of a top-level fvSchemes block ("divSchemes", "laplacianSchemes").
// Needed because searching the whole file for `default` finds ddtSchemes' entry first -- every
// fvSchemes has several -- and that misreports why a lookup failed.
inline std::string fvSchemesBlock(const std::string& all, const std::string& name)
{
    const std::size_t b = all.find(name);
    if (b == std::string::npos) return std::string();
    const std::size_t o = all.find('{', b);
    if (o == std::string::npos) return std::string();
    int depth = 0;
    std::size_t i = o;
    for (; i < all.size(); ++i)
    {
        if (all[i] == '{') ++depth;
        else if (all[i] == '}') { if (--depth == 0) break; }
    }
    return all.substr(o, (i < all.size() ? i - o : std::string::npos));
}

inline FieldDivScheme parseFieldDivScheme(const std::string& caseDir, const std::string& field)
{
    // Same source as parseFvSchemesControls: $-expanded, so `div(phi,tracer0) $turbulence;` resolves.
    const std::string all = readFvSchemesText(caseDir);

    // Scope to the divSchemes BLOCK. Searching the whole file for `default` finds ddtSchemes' entry
    // first and misreports why a lookup failed -- every fvSchemes has several `default` lines.
    std::string raw = fvSchemesBlock(all, "divSchemes");
    if (raw.empty()) raw = all;
    const std::string key = "div(phi," + field + ")";

    // Find the statement for this field: from the key to its terminating ';'.
    const std::size_t k = raw.find(key);
    if (k == std::string::npos)
    {
        // OF: `default none` means an unlisted scheme is a fatal error, not a silent fallback.
        const std::size_t d = raw.find("default");
        const bool defaultNone = (d != std::string::npos
                                  && raw.find("none", d) != std::string::npos
                                  && raw.find("none", d) < raw.find(';', d));
        if (defaultNone)
            throw std::runtime_error(
                "brae: fvSchemes divSchemes has `default none` and no `" + key + "` entry, so " +
                field + " has no convection scheme. OpenFOAM refuses this case; brae will not run it "
                "with a substituted scheme.");
        throw std::runtime_error(
            "brae: fvSchemes divSchemes has no `" + key + "` entry and brae does not resolve the "
            "divSchemes `default`; add the entry explicitly.");
    }
    const std::size_t end = raw.find(';', k);
    const std::string st = raw.substr(k, end == std::string::npos ? std::string::npos : end - k);

    FieldDivScheme fs;
    divSchemesConsumed().insert(key);   // recorded here too: the tracer's own div(phi,<field>)
    fs.bounded      = st.find("bounded")       != std::string::npos;
    fs.limited      = st.find("limitedLinear") != std::string::npos;
    fs.linearUpwind = st.find("linearUpwind")  != std::string::npos;
    if (fs.limited)
    {
        double kc = 1.0;
        const std::size_t q = st.find("limitedLinear");
        std::sscanf(st.c_str() + q + 13, "%lf", &kc);
        fs.coeff  = static_cast<scalar>(kc);
        fs.twoByk = static_cast<scalar>(2.0 / std::max(kc, 1e-30));
    }

    // laplacian(D<field>,<field>), else the laplacianSchemes `default`.
    //
    // Named-then-default is OF's own direction -- schemesLookup.H:112 documents lookup() as "Lookup
    // named scheme from dictionary, or return default". What OF does when NEITHER exists is not
    // readable here (only lnInclude headers ship, no schemesLookup.C), but populate() takes a
    // `mandatory` flag and fallback() returns the default "(if any)", so the absent case is plainly an
    // error there and not a silent value. So it is an error here too: inventing a default would be
    // exactly the substitution this whole path exists to avoid.
    {
        const std::string lap = fvSchemesBlock(all, "laplacianSchemes");
        const std::string lkey = "laplacian(D" + field + "," + field + ")";
        std::size_t q = lap.find(lkey);
        bool viaDefault = false;
        if (q == std::string::npos) { q = lap.find("default"); viaDefault = true; }
        if (q == std::string::npos)
            throw std::runtime_error(
                "brae: fvSchemes laplacianSchemes has neither `" + lkey + "` nor a `default`, so " +
                field + " has no laplacian scheme.");
        const std::size_t e2 = lap.find(';', q);
        const std::string ls = lap.substr(q, e2 == std::string::npos ? std::string::npos : e2 - q);
        // `none` is a real token OF would try to construct a scheme from, and fail. Treating it as
        // "found" and quietly running `corrected` would be a substituted discretisation.
        if (ls.find("none") != std::string::npos)
            throw std::runtime_error(
                "brae: fvSchemes laplacianSchemes " + std::string(viaDefault ? "`default`" : lkey) +
                " is `none`, so " + field + " has no laplacian scheme. OpenFOAM refuses this; brae "
                "will not substitute one.");
        if (ls.find("uncorrected") != std::string::npos
         || ls.find("orthogonal")  != std::string::npos) fs.nonOrth = false;
    }
    return fs;
}

// fvSchemes `wallDist { method <m>; }`. brae's cellWallDist is a byte-for-byte port of OF's DEFAULT
// method, meshWave (patchDistMethods::meshWave -> patchWave -> FaceCellWave<wallPoint>), and every
// rhoSimpleFoam tutorial asks for exactly that. OF offers five others (Poisson, advectionDiffusion,
// directionalMeshWave, meshWaveAddressing, patchDistMethod), and until now a case naming one of those
// silently got meshWave: the dict was read by nothing at all.
//
// y feeds kOmegaSST's F1/F2/F3 blending and Spalart-Allmaras' dTilda destruction term (~1/y^2), so a
// different wall distance is a different turbulence model, not a detail. Absent means meshWave, which
// is OF's own default.
inline void checkWallDistMethod(const std::string& caseDir, DeviceSimpleControls& ctl)
{
    const std::string all = readFvSchemesText(caseDir);
    const std::string blk = fvSchemesBlock(all, "wallDist");
    if (blk.empty()) return;                       // no entry -> OF's default, which is what brae runs
    const std::size_t m = blk.find("method");
    if (m == std::string::npos) return;
    const std::size_t e = blk.find(';', m);
    std::string w = blk.substr(m + 6, (e == std::string::npos ? std::string::npos : e - m - 6));
    // trim
    const std::size_t a = w.find_first_not_of(" \t\n\r");
    const std::size_t b = w.find_last_not_of(" \t\n\r");
    if (a == std::string::npos) return;
    w = w.substr(a, b - a + 1);
    if (w == "meshWave") return;
    if (w == "exactDistance") { ctl.exactWallDist = true; return; }
    throw std::runtime_error(
        "brae: fvSchemes wallDist method '" + w + "' is not implemented; brae computes OF's default "
        "meshWave (and `exactDistance`). The wall distance feeds kOmegaSST F1/F2/F3 and "
        "Spalart-Allmaras dTilda (~1/y^2), so running a different method would be a different turbulence "
        "model, not an approximation.");
}

inline void parseFvSchemesControls(const std::string& caseDir, DeviceSimpleControls& ctl)
{
    // Refuse an unimplemented wall-distance method here, where every solver already passes.
    checkWallDistMethod(caseDir, ctl);

            std::string schemesText = readFileExpanded(caseDir + "/system/fvSchemes");   // $var expanded ($turbulence)
            // Statements are ';'-terminated and each one belongs to a SUB-DICTIONARY. Both halves matter.
            //
            // Splitting on ';' (not newlines) is what stops one entry's flags leaking into the next: OF does
            // not care about layout, so "div(phi,U) upwind; div(phi,e) linearUpwind;" on one line is two
            // different schemes, and matching per LINE once gave div(phi,U) the energy equation's scheme.
            //
            // Tracking the enclosing sub-dictionary is what stops a rule firing on a statement from a
            // DIFFERENT group. The rules used to sniff keywords across a flat token stream, so any statement
            // containing "Gauss" was treated as a candidate laplacian/snGrad entry. aerofoilNACA0012 uses OF's
            // standard macro idiom:
            //     gradSchemes { limited  cellLimited Gauss linear 1;  grad(U) $limited; ... }
            //     divSchemes  { div(phi,U)  bounded Gauss linearUpwind limited; ... }
            // where "limited" is the NAME of a gradient scheme and has nothing to do with the laplacian. Both
            // statements matched `hasWord(ln, "limited")` and set ctl.nonOrth, so a case whose laplacianSchemes
            // and snGradSchemes both say `orthogonal` ran WITH non-orthogonal correction. Measured on a
            // laplacian-orthogonal case: nonOrth 0 -> 1 purely from the divSchemes line.
            struct Stmt { std::string block, text; };
            std::vector<Stmt> stmts;
            {
                std::string buf, cur;
                std::vector<std::string> stack;
                auto lastWord = [](const std::string& b)
                {
                    std::size_t e = b.size();
                    while (e > 0 && std::isspace((unsigned char)b[e-1])) --e;
                    std::size_t st = e;
                    while (st > 0 && !std::isspace((unsigned char)b[st-1]) && b[st-1] != '{' && b[st-1] != '}') --st;
                    return b.substr(st, e - st);
                };
                for (char c : schemesText)
                {
                    if (c == '{')      { stack.push_back(lastWord(buf)); buf.clear(); }
                    else if (c == '}') { if (!stack.empty()) stack.pop_back(); buf.clear(); }
                    else if (c == ';') { stmts.push_back({stack.empty() ? std::string() : stack.front(), buf}); buf.clear(); }
                    else                 buf += c;
                }
            }
            // OF looks div(phi,U) up in divSchemes and falls back to the `default` entry when it is not
            // listed (dictionary lookup with a default); `default none` is the idiom that says "there is
            // no fallback, name every scheme", and there OF itself throws. brae refused BOTH alike, which
            // turned an ordinary `default Gauss linear` case (laminar/cylinder2D) into a hard stop over a
            // scheme it implements. Resolve the default the way OF does by synthesising the statement the
            // file would have contained, so every branch below sees it as if it had been written out; the
            // refusal survives for `none` and for no default at all.
            {
                bool explicitDivU = false;
                std::string defaultDiv;
                for (const Stmt& st : stmts)
                {
                    if (st.block != "divSchemes") continue;
                    if (st.text.find("div(phi,U)") != std::string::npos) explicitDivU = true;
                    std::size_t b = st.text.find_first_not_of(" \t\n\r");
                    if (b == std::string::npos) continue;
                    if (st.text.compare(b, 7, "default") == 0
                        && (b + 7 >= st.text.size() || std::isspace((unsigned char)st.text[b + 7])))
                        defaultDiv = st.text.substr(b + 7);
                }
                const std::size_t v = defaultDiv.find_first_not_of(" \t\n\r");
                if (v != std::string::npos) defaultDiv = defaultDiv.substr(v);
                if (!explicitDivU && !defaultDiv.empty() && defaultDiv.compare(0, 4, "none") != 0)
                    stmts.push_back({"divSchemes", "div(phi,U) " + defaultDiv});
            }
            bool foundDivU = false;   // set by the div(phi,U) branch below (explicit, or resolved from `default`)
            bool warnedLeastSq = false, warnedCellMD = false;   // #14: warn-once on grad schemes brae approximates
            auto hasWord = [](const std::string& s, const std::string& w)   // whole-word match (so "uncorrected" != "corrected")
            {
                for (std::size_t p = s.find(w); p != std::string::npos; p = s.find(w, p + 1))
                {
                    const bool lb = (p == 0 || !std::isalpha((unsigned char)s[p-1]));
                    const bool rb = (p + w.size() >= s.size() || !std::isalpha((unsigned char)s[p+w.size()]));
                    if (lb && rb) return true;
                }
                return false;
            };
            // "limitedLinear <k_>" on a div line -> twoByk = 2/max(k_,SMALL); returns 0 if the scheme is absent.
            auto limitedTwoByk = [](const std::string& s) -> scalar
            {
                const std::size_t p = s.find("limitedLinear");
                if (p == std::string::npos) return 0.0;
                scalar kc = 1.0;
                std::sscanf(s.c_str() + p + 13, "%lf", &kc);   // coefficient after "limitedLinear"
                return 2.0 / std::max(kc, (scalar)1e-30);
            };
            // the interpolation-scheme word after "Gauss" on a div(phi,*) line (OF: "[bounded] Gauss <scheme> [args]").
            auto divSchemeWord = [](const std::string& s) -> std::string
            {
                const std::size_t g = s.find("Gauss");
                if (g == std::string::npos) return std::string();
                const char* p = s.c_str() + g + 5;
                while (*p && std::isspace((unsigned char)*p)) ++p;
                std::string w;
                while (*p && !std::isspace((unsigned char)*p) && *p != ';') { w += *p; ++p; }
                return w;
            };
            // OF-faithful fail-fast (mirrors surfaceInterpolationScheme::New "Unknown discretisation scheme ... Valid
            // schemes are : (...)" + exit(FatalIOError)): throw on a scheme cf models nothing close to; warn loudly on
            // one cf only APPROXIMATES (so the route is detected, never silently covered). `ok` = exact; `approx` = warned.
            auto checkDiv = [&](
                const std::string& s,
                const char* field,
                std::initializer_list<const char*> ok,
                std::initializer_list<const char*> approx)
            {
                { const std::string dk = divKeyOf(s); if (!dk.empty()) divSchemesConsumed().insert(dk); }
                const std::string w = divSchemeWord(s);
                for (const char* o : ok)
                    if (w == o) return;
                for (const char* o : approx)
                    if (w == o)
                    {
                        std::fprintf(stderr, "brae WARNING: div(phi,%s) 'Gauss %s' has no exact cf kernel -- run as a near-equivalent "
                                     "(NOT OF-bit-identical). Set the scheme to an exact one to avoid this.\n", field, w.c_str());
                        return;
                    }
                std::string valid;
                for (const char* o : ok)     (valid += " ") += o;
                for (const char* o : approx) (valid += " ") += o;
                throw std::runtime_error(std::string("brae: unknown/unsupported div(phi,") + field +
                    ") scheme 'Gauss " + w + "'; cf supports : (" + valid + " )");
            };
            // The cellLimited coefficient of every NAMED gradSchemes entry, keyed by its name.
            //
            // OF's linearUpwind takes the gradient scheme as an ARGUMENT and looks that name up in
            // gradSchemes (linearUpwind.H: gradSchemeName_(schemeData) -> mesh.gradScheme(name)), so
            //     gradSchemes { limited cellLimited Gauss linear 1; }
            //     divSchemes  { div(phi,e) bounded Gauss linearUpwind limited; }
            // gives the energy's deferred correction a CELL-LIMITED gradient even though the case has no
            // `grad(e)` entry at all. brae keyed the limiter off `grad(e)`/`grad(h)` alone, found nothing,
            // and ran unlimited. Measured on aerofoilNACA0012: forcing OF to use an unlimited gradient
            // reproduces brae's first-iteration T[233.71, 301.64] against OF's own T[297.95, 298.01].
            std::map<std::string, scalar> gradLimitByName;
            for (const Stmt& gs : stmts)
            {
                if (gs.block != "gradSchemes") continue;
                const char* p = gs.text.c_str();
                while (*p && std::isspace((unsigned char)*p)) ++p;
                std::string key;
                while (*p && !std::isspace((unsigned char)*p)) key += *p++;
                if (key.empty()) continue;
                scalar kc = 0.0;                                     // not cellLimited -> unlimited
                if (hasWord(gs.text, "cellLimited"))
                {
                    kc = 1.0;
                    const char* c = gs.text.c_str() + gs.text.find("cellLimited") + 11;
                    while (*c && !(std::isdigit((unsigned char)*c) || *c == '.')) ++c;
                    if (std::sscanf(c, "%lf", &kc) != 1) kc = 1.0;
                }
                gradLimitByName[key] = kc;
            }
            // The cellLimited coefficient linearUpwind on THIS div statement will use: the word after
            // "linearUpwind" resolved through the table above, else the gradSchemes `default`. Returns
            // -1 when the statement is not linearUpwind at all, so a caller can keep its own fallback.
            auto luGradLimit = [&](const std::string& s) -> scalar
            {
                const std::size_t p = s.find("linearUpwind");
                if (p == std::string::npos) return -1.0;
                const char* c = s.c_str() + p + 12;
                while (*c && (std::isalpha((unsigned char)*c)))  ++c;   // skip the V of linearUpwindV
                while (*c && std::isspace((unsigned char)*c))    ++c;
                std::string name;
                while (*c && !std::isspace((unsigned char)*c) && *c != ';') name += *c++;
                if (name.empty()) name = "default";
                const auto it = gradLimitByName.find(name);
                if (it != gradLimitByName.end()) return it->second;
                const auto d = gradLimitByName.find("default");
                return d != gradLimitByName.end() ? d->second : 0.0;
            };
            for (const Stmt& st : stmts)
            {
                const std::string& ln = st.text;
                const bool inDiv    = (st.block == "divSchemes");
                const bool inGrad   = (st.block == "gradSchemes");
                const bool inLap    = (st.block == "laplacianSchemes" || st.block == "snGradSchemes");
                const bool inInterp = (st.block == "interpolationSchemes");
                (void)inInterp;
                if (inDiv && ln.find("div(phi,U)") != std::string::npos)
                {
                    foundDivU = true;
                    // DEShybrid takes POSITIONAL arguments -- two sub-schemes, the delta field name, then
                    // six or seven numbers (DEShybrid.H's Istream constructor). Parsed here rather than
                    // through checkDiv, whose vocabulary is single-word schemes.
                    if (hasWord(ln, "DEShybrid"))
                    {
                        ctl.desHybrid = true;
                        const std::size_t p0 = ln.find("DEShybrid") + 9;
                        // OF blends scheme1 (low dissipation) with scheme2 (upwind-biased). brae implements
                        // the pair every DES tutorial uses; anything else would silently run a different
                        // discretisation, so it is named.
                        const std::string rest = ln.substr(p0);
                        if (!hasWord(rest, "linear") || !hasWord(rest, "linearUpwind"))
                            throw std::runtime_error(
                                "brae: div(phi,U) Gauss DEShybrid blends two named schemes; brae implements "
                                "`linear` (scheme 1) with `linearUpwind` (scheme 2), which is what the "
                                "OpenFOAM DES tutorials use. This case asks for something else:\n  " + ln);
                        ctl.linearUpwind = true;   // scheme 2 supplies the gradient reconstruction
                        { const scalar g = luGradLimit(rest); if (g >= 0.0) ctl.gradULULimitK = g; }
                        // the numbers, in OF's read order: CDES U0 L0 sigmaMin sigmaMax OmegaLim [nutLim]
                        std::vector<scalar> num;
                        for (const char* c = rest.c_str(); *c; )
                        {
                            if (std::isdigit((unsigned char)*c)
                                || ((*c == '-' || *c == '.') && std::isdigit((unsigned char)c[1])))
                            {
                                char* e = nullptr;
                                num.push_back(std::strtod(c, &e));
                                c = e ? e : c + 1;
                            }
                            else ++c;
                        }
                        if (num.size() < 6)
                            throw std::runtime_error(
                                "brae: div(phi,U) Gauss DEShybrid needs at least six coefficients (CDES U0 L0 "
                                "sigmaMin sigmaMax OmegaLim [nutLim]); found " + std::to_string(num.size())
                                + " in:\n  " + ln);
                        DesHybridCoeffs& d = ctl.desCoeffs;
                        d.CDES = num[0]; d.U0 = num[1]; d.L0 = num[2];
                        d.sigmaMin = num[3]; d.sigmaMax = num[4]; d.OmegaLim = num[5];
                        d.nutLim = (num.size() > 6) ? num[6] : scalar(1);
                        if (d.U0 <= 0 || d.L0 <= 0)
                            throw std::runtime_error("brae: DEShybrid U0 and L0 must be > 0 (OF checkValues).");
                        if (d.sigmaMin < 0 || d.sigmaMin > 1 || d.sigmaMax < 0 || d.sigmaMax > 1)
                            throw std::runtime_error("brae: DEShybrid sigmaMin/sigmaMax must lie in [0,1] (OF checkValues).");
                        continue;   // the generic single-word checks below do not apply
                    }
                    checkDiv(ln, "U", {"upwind", "linearUpwind", "linearUpwindV", "LUST", "linear", "limitedLinearV"}, {"limitedLinear"});
                    if (ln.find("bounded") != std::string::npos)      ctl.bounded = true;
                    // Gauss LINEAR = central differencing: OF's `linear` scheme returns the plain
                    // geometric weights (linear.H:106), so fvmDiv builds lower=-w*phi, upper=lower+phi
                    // with w from the mesh instead of pos0(phi). divSchemeWord() returns the token right
                    // after "Gauss", so this is only true for a BARE `linear` -- "linearUpwind" and
                    // "limitedLinear" are different words and are matched by their own branches above.
                    if (divSchemeWord(ln) == "linear")               ctl.divULinear = true;
                    if (ln.find("linearUpwind") != std::string::npos) ctl.linearUpwind = true;   // linearUpwindV contains this -> upwind matrix + gradients
                    if (ln.find("linearUpwindV") != std::string::npos) ctl.linearUpwindV = true; // + OF vector direction limiter
                    if (ln.find("LUST") != std::string::npos)         ctl.lust = true;   // 0.75 linear + 0.25 linearUpwind
                    // `Gauss limitedLinearV k`: the vector (NVDVTVDV) limiter. k is the number right after
                    // the scheme word, and OF REFUSES k outside [0,1] (limitedLinearLimiter's ctor), so do
                    // the same rather than clamp -- a k of 2 is a typo, not a request.
                    if (divSchemeWord(ln) == "limitedLinearV")
                    {
                        ctl.divULimitedV = true;
                        const std::size_t w = ln.find("limitedLinearV") + 14;
                        const scalar k = std::strtod(ln.c_str() + w, nullptr);
                        if (k < 0.0 || k > 1.0)
                            throw std::runtime_error("brae: div(phi,U) limitedLinearV coefficient must lie in "
                                                     "[0,1] (OF limitedLinearLimiter):\n  " + ln);
                        ctl.divUTwoBykV = 2.0/std::max(k, scalar(1e-15));   // OF twoByk_ = 2/max(k_, SMALL)
                    }
                    { const scalar g = luGradLimit(ln); if (g >= 0.0) ctl.gradULULimitK = g; }   // linearUpwind's named grad(U)
                }
                // div(phi,sigma) -- the Maxwell viscoelastic stress transport. Both Maxwell tutorials
                // write `Gauss vanAlbada`, and brae HAS that limiter, but nothing ever SET the flag that
                // switches it on: divSigmaVanAlbada was declared, defaulted false, read once in
                // correctMaxwell, and assigned nowhere. The stress equation therefore ran UNLIMITED
                // while the case asked for a TVD limiter, and neither tutorial caught it because both
                // flows are smooth enough that the limiter barely engages -- the same false comfort a
                // degenerate Bird-Carreau parameterisation gave on TJunctionArrhenius.
                //
                // Found by the coverage manifest, not by a case: `vanAlbada` appeared as a type the
                // tutorials DEMAND and brae never names in a quoted comparison, which is exactly the
                // signature of a control that is plumbed but never selected.
                if (inDiv && ln.find("div(phi,sigma)") != std::string::npos)
                {
                    const std::string sw = divSchemeWord(ln);
                    if (sw == "vanAlbada") ctl.divSigmaVanAlbada = true;
                    else if (!sw.empty() && sw != "linear" && sw != "upwind")
                        throw std::runtime_error(
                            "brae: div(phi,sigma) scheme '" + sw + "' is not implemented (brae has "
                            "`vanAlbada`, `linear` and `upwind`). The limiter bounds the viscoelastic "
                            "stress transport, so substituting another one solves a different "
                            "equation:\n  " + ln);
                }
                // grad(U) cellLimited Gauss linear <k> (OF cellLimitedGrad<minmod>): k is the first number after
                // "cellLimited" (the basicScheme between has no digits). 0 = unlimited. cellMDLimited not yet handled.
                if (inGrad && ln.find("grad(U)") != std::string::npos && hasWord(ln, "cellLimited"))
                {
                    ctl.gradULimitK = 1.0;
                    const char* s = ln.c_str() + ln.find("cellLimited") + 11;
                    while (*s && !(std::isdigit((unsigned char)*s) || *s == '.')) ++s;
                    scalar kc;
                    if (std::sscanf(s, "%lf", &kc) == 1) ctl.gradULimitK = kc;
                }
                // C2: the same cellLimited rule for the TURBULENCE and ENERGY gradients. Previously only
                // grad(U) was scanned, so `grad(k) cellLimited Gauss linear 1` was read and discarded.
                if (inGrad && hasWord(ln, "cellLimited"))
                {
                    scalar kc = 1.0;
                    const char* c = ln.c_str() + ln.find("cellLimited") + 11;
                    while (*c && !(std::isdigit((unsigned char)*c) || *c == '.')) ++c;
                    if (std::sscanf(c, "%lf", &kc) != 1) kc = 1.0;
                    if (ln.find("grad(k)") != std::string::npos || ln.find("grad(omega)") != std::string::npos
                     || ln.find("grad(epsilon)") != std::string::npos || ln.find("grad(nuTilda)") != std::string::npos)
                        ctl.gradKLimitK = kc;
                    if (ln.find("grad(h)") != std::string::npos || ln.find("grad(e)") != std::string::npos)
                        ctl.gradHeLimitK = kc;
                }
                // #14: brae computes gradients via Gauss linear only. These are valid ALTERNATIVE discretisations
                // (not wrong answers), so warn-once rather than fail -- the user should know brae is approximating.
                if (inGrad && !warnedLeastSq && ln.find("leastSquares") != std::string::npos)
                { warnedLeastSq = true; std::fprintf(stderr, "brae WARNING: gradScheme 'leastSquares' is approximated as Gauss linear (differs on skewed meshes)\n"); }
                if (inGrad && !warnedCellMD && ln.find("cellMDLimited") != std::string::npos)
                { warnedCellMD = true; std::fprintf(stderr, "brae WARNING: grad limiter 'cellMDLimited' is not applied (runs unlimited)\n"); }
                if (std::getenv("BRAE_SCHEME_DEBUG") && ln.find("div(phi,") != std::string::npos) std::fprintf(stderr, "[scheme] %s\n", ln.c_str());
                if (inDiv && ln.find("div(phi,k)") != std::string::npos)
                {
                    checkDiv(ln, "k", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedK = true; ctl.twoBykK = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luK = true;
                    if (hasWord(ln, "bounded")) ctl.boundedK = true;
                    { const scalar g = luGradLimit(ln); if (g >= 0.0) ctl.gradKLULimitK = g; }   // linearUpwind's named grad(k|eps)
                }
                if (inDiv && (ln.find("div(phi,epsilon)") != std::string::npos || ln.find("div(phi,omega)") != std::string::npos))
                {
                    checkDiv(ln, "epsilon|omega", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedEps = true; ctl.twoBykEps = t; }   // 2nd turb scalar (eps|omega)
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luEps = true;
                    if (hasWord(ln, "bounded")) ctl.boundedEps = true;
                    { const scalar g = luGradLimit(ln); if (g >= 0.0) ctl.gradKLimitK = g; }   // EXPERIMENT
                }
                // Energy: OF names the field "h" for sensibleEnthalpy and "e" for sensibleInternalEnergy,
                // and the kinetic term "K" or "Ekp" to match. Any of them sets the same flags.
                // Energy: OF names the field "h" for sensibleEnthalpy and "e" for sensibleInternalEnergy.
                if (inDiv && (ln.find("div(phi,h)") != std::string::npos || ln.find("div(phi,e)") != std::string::npos))
                {
                    checkDiv(ln, "h|e", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedHe = true; ctl.twoBykHe = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luHe = true;
                    if (hasWord(ln, "bounded")) ctl.boundedHe = true;
                    const scalar g = luGradLimit(ln);   // the gradient linearUpwind NAMES, not grad(e)
                    if (g >= 0.0) ctl.gradHeLimitK = g;
                }
                // The KINETIC term, named "K" alongside h and "Ekp" alongside e. Its own fvSchemes entry and
                // its own slots -- see DeviceSimpleControls::luKin for why sharing the He slots was wrong.
                if (inDiv && (ln.find("div(phi,K)") != std::string::npos || ln.find("div(phi,Ekp)") != std::string::npos))
                {
                    checkDiv(ln, "K|Ekp", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    ctl.foundKinScheme = true;
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedKin = true; ctl.twoBykKin = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luKin = true;
                    if (hasWord(ln, "bounded")) ctl.boundedKin = true;
                    const scalar g = luGradLimit(ln);
                    if (g >= 0.0) ctl.gradKinLimitK = g;
                }
                if (inDiv && ln.find("div(phi,nuTilda)") != std::string::npos)   // SA: nuTilda uses the k-slot scheme flags
                {
                    checkDiv(ln, "nuTilda", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedK = true; ctl.twoBykK = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luK = true;
                    if (hasWord(ln, "bounded")) ctl.boundedK = true;   // SA: nuTilda uses the k slot
                }
                if (inLap)
                {
                    if (hasWord(ln, "corrected")) ctl.nonOrth = true;     // unlimited non-orth correction (psi = 1)
                    // OF fv::limitedSnGrad "limited [<correctedScheme>] <psi>" (psi in [0,1]): non-orth correction
                    // capped per-face. hasWord avoids matching "unlimited" and "limitedLinear" (a div scheme); the coeff
                    // is the next numeric token after "limited" (skip an optional scheme word like "corrected").
                    if (hasWord(ln, "limited"))
                    {
                        ctl.nonOrth = true;
                        scalar psi = 1.0;
                        const char* s = ln.c_str() + ln.find("limited") + 7;
                        while (*s && !(std::isdigit((unsigned char)*s) || *s == '.')) ++s;   // skip to the coefficient
                        if (std::sscanf(s, "%lf", &psi) == 1) ctl.nonOrthLimit = psi;
                    }
                }
            }
            // No explicit div(phi,K|Ekp): OF would fall through to the divSchemes `default`. brae keeps its
            // previous behaviour (the energy scheme) rather than inventing one, but says so, because that is
            // an assumption and not what the file asked for.
            if (!ctl.foundKinScheme)
            {
                ctl.boundedKin = ctl.boundedHe;
                ctl.limitedKin = ctl.limitedHe;
                ctl.luKin      = ctl.luHe;
                ctl.twoBykKin  = ctl.twoBykHe;
                ctl.gradKinLimitK = ctl.gradHeLimitK;
            }
            // C1: interpolationSchemes was never parsed at all -- zero hits in src/. brae interpolates
            // linearly everywhere, which IS OF's default, so a `default linear` case was right by accident.
            // Anything else silently ran linear instead, which is a different discretisation, so it is
            // refused rather than noticed (the project rule: notice when brae does LESS than asked,
            // throw when the answer would be WRONG).
            for (const Stmt& st : stmts)
            {
                if (st.block != "interpolationSchemes") continue;
                const std::string& ln = st.text;
                std::istringstream is(ln);
                std::string key, scheme;
                is >> key >> scheme;
                if (key.empty() || scheme.empty() || scheme == "linear") continue;
                if (key == "default")
                    throw std::runtime_error(
                        "brae: interpolationSchemes default is '" + scheme + "'; brae interpolates linearly "
                        "and would silently run 'linear' instead. Set it to linear, or use a case that does.");
                noticeIgnored("interpolationSchemes",
                              key + " " + scheme + " -- brae interpolates linearly; only the `default` entry is enforced");
            }
            if (!schemesText.empty() && !foundDivU)
                throw std::runtime_error("fvSchemes: no div(phi,U) scheme, and the divSchemes `default` is"
                    " `none` or absent -- so there is nothing to resolve it to, exactly as OpenFOAM would"
                    " report. Add e.g. 'div(phi,U)  bounded Gauss linearUpwind grad(U);'.");

            // What brae CONCLUDED, not what the file said. BRAE_SCHEME_DEBUG used to echo the raw input
            // line -- which is exactly the thing that was ambiguous: a one-line divSchemes looked correct
            // in that echo while the flags leaked between entries, and div(phi,U) silently picked up the
            // energy scheme. A measurement that never states which group its input landed in cannot tell
            // "hypothesis wrong" from "input misread", and I read one as the other.
            if (std::getenv("BRAE_SCHEME_DEBUG"))
            {
                auto sname = [](bool lu, bool lim) { return lu ? "linearUpwind" : (lim ? "limitedLinear" : "upwind"); };
                std::fprintf(stderr,
                    "[scheme] RESOLVED  div(phi,U)=%s%s  div(phi,k|omega)=%s  div(phi,h|e)=%s  "
                    "div(phi,K|Ekp)=%s%s  nonOrth=%d gradLimit(U=%g,k|omega=%g,h|e=%g)\n",
                    ctl.linearUpwind ? "linearUpwind" : "upwind", ctl.linearUpwindV ? "V" : "",
                    sname(ctl.luK, ctl.limitedK), sname(ctl.luHe, ctl.limitedHe),
                    sname(ctl.luKin, ctl.limitedKin), ctl.foundKinScheme ? "" : "(inherited)",
                    (int)ctl.nonOrth, ctl.gradULimitK, ctl.gradKLimitK, ctl.gradHeLimitK);
            }
}

}  // namespace brae
