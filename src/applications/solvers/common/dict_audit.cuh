#pragma once
// dict_audit.cuh -- report every dictionary entry brae READ OFF DISK AND THEN IGNORED.
//
// This exists because of what the rhoSimpleFoam audit kept finding. The dangerous class of defect in this
// project has never been "brae cannot parse X" -- that fails loudly and gets fixed. It is "brae parsed X,
// stored it, and then never used it", which converges to a plausible answer and says nothing. fvOptions
// (A2), the turbulent-inlet recompute (A3), the kinetic-term scheme (A4), the Function1 uniformValue (A5),
// tangentialVelocity (A9), interpolationSchemes (C1), grad(k) cellLimited (C2), the kinetic slot (C4),
// residualControl (C5), startFrom (C6) -- ten separate instances of one shape.
//
// Each was found by reading OF's source and noticing an entry brae never mentions. That does not scale and
// it does not generalise to the next solver. So: FoamDict records every key it is ASKED for, and this walks
// the parsed dict afterwards and names everything that was never asked. The complement is the gap list.
//
// It reports; it does not throw. An unread key is EVIDENCE, not proof -- some entries are legitimately for
// other tools (functionObject sub-entries, writeFormat), and a few are consumed by paths that ran before the
// audit point. Treating it as fatal would make it useless. Silence is the thing being fixed.
//
// On by default so it is visible during development. BRAE_DICT_AUDIT=0 silences it.
#include "foam_dict.cuh"
#include "scheme_parse.cuh"   // divSchemesConsumed(): recorded on the consumption path
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brae {

namespace detail {

// Entries that are genuinely not brae's to consume, so naming them would be noise that trains the reader
// to ignore the whole report. Kept deliberately short: when in doubt, REPORT it. A false positive costs a
// line of output; a false negative is the bug class this file exists to catch.
inline bool auditIgnorable(const std::string& key)
{
    static const char* skip[] = {
        "FoamFile", "version", "format", "class", "location", "object", "note", "arch",
        "writeFormat", "writePrecision", "writeCompression", "timeFormat", "timePrecision",
        "runTimeModifiable", "graphFormat", "libs", "functions", "DebugSwitches", "InfoSwitches",
        // Consumed by solver_dispatch on its own FoamDict instance before the driver ever starts, so it is
        // read by brae -- just not by this object. Listing it would be a permanent false positive.
        "application",
    };
    for (const char* s : skip)
        if (key == s) return true;
    return false;
}

inline void auditWalk(
    const FoamDict& d,
    const std::string& path,
    std::vector<std::string>& out)
{
    for (const auto& l : d.leaves)
    {
        if (auditIgnorable(l.first)) continue;
        if (d.queried.count(l.first)) continue;
        out.push_back(path + l.first);
    }
    for (const auto& s : d.subs)
    {
        if (auditIgnorable(s.first)) continue;
        if (!d.queried.count(s.first))
        {
            // A whole sub-dictionary nobody opened. Report it as one line rather than every leaf inside
            // it -- "brae never looked at PIMPLE" is the finding; its twelve entries are one fact.
            out.push_back(path + s.first + "/ (whole sub-dictionary never read)");
            continue;
        }
        auditWalk(s.second, path + s.first + "/", out);
    }
}

}   // namespace detail

// PREFLIGHT: an unread key in the ALGORITHM-CONTROL block is FATAL, not a notice.
//
// auditDict above reports and does not throw, for a good reason -- an unread key elsewhere may belong to
// another tool. That reasoning does not extend to fvSolution's PIMPLE/SIMPLE sub-dictionary. Every entry
// there selects part of the pressure-velocity algorithm, none of them belong to anyone else, and reading
// one and ignoring it means brae is running a DIFFERENT ALGORITHM than the case asked for -- silently,
// and with a converged-looking field at the end.
//
// This is not hypothetical. `turbOnFinalIterOnly` defaults to TRUE in OpenFOAM's pimpleControl, so
// turbulence is corrected once per time step on the final outer corrector. brae corrected it on EVERY
// outer corrector and never read the key. A 30-case tutorial sweep did not catch it, because most
// tutorials use nOuterCorrectors 1, where the two cadences are the same; the cases that would have shown
// it (vortexShed 5, propeller 2, axialTurbine 10) were failing for what looked like other reasons.
// `residualControl`, `solveFlow`, `SIMPLErho` and `moveMeshOuterCorrectors` are the same shape.
//
// So the rule for this one block is inverted: brae must consume every key or refuse. A gap becomes a
// startup error naming the key, never a discovery made halfway through a simulation.
inline void preflightAlgorithmControls(const FoamDict& fvSolution, const std::string& dictName)
{
    const FoamDict* alg = fvSolution.subDict(dictName);
    if (!alg) return;
    // Controls brae consumes CONDITIONALLY, so an unread one is not a gap. Each needs a reason, and the
    // reason has to be that OpenFOAM ignores it under the same condition -- otherwise this list is just a
    // way to silence the check. Keep it short: anything added here stops being enforced.
    auto conditional = [](const std::string& k)
    {
        //  pRefCell/pRefValue : read only when the pressure needs a reference (no fixedValue-p patch).
        //                       OpenFOAM's setRefCell does nothing on a case with a fixed-pressure
        //                       boundary either, so an unread pair there is inert in both codes.
        //  transonic          : consumed by the COMPRESSIBLE pEqn. Incompressible pimpleFoam never reads
        //                       it in OpenFOAM either -- a case that sets it is setting a no-op.
        return k == "pRefCell" || k == "pRefValue" || k == "transonic";
    };
    std::vector<std::string> unread;
    for (const auto& l : alg->leaves)
        if (!alg->queried.count(l.first) && !conditional(l.first)) unread.push_back(l.first);
    for (const auto& sb : alg->subs)
        if (!alg->queried.count(sb.first)) unread.push_back(sb.first + " (sub-dictionary)");
    if (unread.empty()) return;

    std::string msg = "brae: system/fvSolution " + dictName + " contains "
                    + std::to_string(unread.size()) + " entr"
                    + (unread.size() == 1 ? "y" : "ies") + " brae never read:";
    for (const std::string& k : unread) msg += "\n    " + dictName + "/" + k;
    msg += "\n  Every entry in this block selects part of the pressure-velocity algorithm, so running "
           "with one ignored means solving a different algorithm than the case specifies -- which "
           "converges to a plausible wrong answer instead of failing. Refused at startup rather than "
           "discovered mid-run. Implement the control, or delete it from the case if it is not wanted.";
    throw std::runtime_error(msg);
}


// Name every entry in `d` that brae never looked up. `label` is the file, e.g. "system/fvSolution".
// `partial` marks a report taken from a run that STOPPED EARLY -- see DictAuditScope.
inline void auditDict(const FoamDict& d, const std::string& label, bool partial = false)
{
    if (const char* e = std::getenv("BRAE_DICT_AUDIT"))
        if (std::atoi(e) == 0) return;

    std::vector<std::string> unread;
    detail::auditWalk(d, "", unread);
    if (unread.empty()) return;

    std::fprintf(stderr, "brae NOTICE [unread] %s -- %zu entr%s brae never looked at%s:\n",
                 label.c_str(), unread.size(), unread.size() == 1 ? "y" : "ies",
                 partial ? " BEFORE THE RUN STOPPED" : "");
    for (const std::string& k : unread)
        std::fprintf(stderr, "brae NOTICE [unread]   %s\n", k.c_str());
    if (partial)
        std::fprintf(stderr,
                     "brae NOTICE [unread]   (PARTIAL: the run ended early, so an entry here may simply not have\n"
                     "brae NOTICE [unread]    been REACHED yet -- residualControl is read inside the iteration loop,\n"
                     "brae NOTICE [unread]    for instance. Compare against a case that runs to completion.)\n");
    else
        std::fprintf(stderr,
                     "brae NOTICE [unread]   (an entry here is EVIDENCE of an unimplemented input, not proof --\n"
                     "brae NOTICE [unread]    some belong to other tools. Set BRAE_DICT_AUDIT=0 to silence.)\n");
}
// fvSchemes, audited at BLOCK granularity.
//
// WHY NOT LEAF GRANULARITY. Every other dict brae reads goes through FoamDict, which records each key on
// the lookup path itself -- so the unread set cannot drift from what the code consumes. fvSchemes does
// not: parseFvSchemesControls reads it as raw TEXT (readFileExpanded), because keys like
// `div(phi,U)` and `$turbulence` expansion are awkward through the dict reader. Auditing its leaves
// would therefore mean maintaining a hand-written list of "keys the text parser touches", and that list
// would drift -- a stale entry marks an ignored input as read, which is precisely the failure this file
// exists to catch, now with a green light on it. A false NEGATIVE here is worse than no audit.
//
// So this reports whole blocks brae never looks at. That is what caught `fluxRequired`: not a wrong
// answer (it is an assertion OF sets in code, createFields.H:43) but an entry nothing in brae mentions.
// Leaf-level fvSchemes coverage stays an open gap, and the notice says so rather than implying it.
inline void auditFvSchemes(const std::string& caseDir)
{
    if (const char* e = std::getenv("BRAE_DICT_AUDIT"))
        if (std::atoi(e) == 0) return;

    FoamDict fvSchemes;
    try { fvSchemes = readDict(caseDir + "/system/fvSchemes"); }
    catch (const std::exception&) { return; }   // absent/unparseable is the parser's problem, not ours

    // The blocks brae resolves. ddtSchemes goes through setDdtScheme rather than parseFvSchemesControls;
    // wallDist is checked by checkWallDistMethod.
    static const char* read[] = {
        "ddtSchemes", "gradSchemes", "divSchemes", "laplacianSchemes",
        "interpolationSchemes", "snGradSchemes", "wallDist"
    };

    std::vector<std::string> unread;
    for (const auto& sub : fvSchemes.subs)
    {
        bool known = false;
        for (const char* r : read) if (sub.first == r) { known = true; break; }
        if (!known) unread.push_back(sub.first + "/ (whole block)");
    }
    for (const auto& leaf : fvSchemes.leaves)
    {
        bool known = false;
        for (const char* r : read) if (leaf.first == r) { known = true; break; }
        if (!known) unread.push_back(leaf.first + " (entry)");
    }
    // divSchemes at LEAF granularity. Possible here and nowhere else in fvSchemes because div
    // consumption has a single choke point (checkDiv), so the consumed set is recorded on the
    // consumption path and cannot drift. A div entry the case declares and brae never reads means the
    // field is being discretised by something other than what was asked for -- the exact defect that
    // put a passive tracer at 6.32 against a bound of 1.0.
    // Read the RAW block, not FoamDict's leaves: the tokeniser splits on parentheses, so `div(phi,U)`
    // arrives as the key `div` with `( phi , U )` among its values. Comparing against leaf keys silently
    // matched nothing and reported nothing -- a false negative, the failure mode this audit exists to
    // prevent. Parsing the same text parseFvSchemesControls reads keeps the two in step by construction.
    try
    {
        const std::string blk =
            fvSchemesBlock(readFileExpanded(caseDir + "/system/fvSchemes"), "divSchemes");
        for (std::size_t i = blk.find("div("); i != std::string::npos; i = blk.find("div(", i + 1))
        {
            const std::string key = divKeyOf(blk.substr(i));
            if (key.empty() || divSchemesConsumed().count(key)) continue;
            unread.push_back("divSchemes/" + key);
        }
    }
    catch (const std::exception&) {}

    if (unread.empty()) return;

    std::fprintf(stderr,
                 "brae NOTICE [unread] system/fvSchemes -- %zu entr%s brae never looked at:\n",
                 unread.size(), unread.size() == 1 ? "y" : "ies");
    for (const std::string& k : unread)
        std::fprintf(stderr, "brae NOTICE [unread]   %s\n", k.c_str());
    std::fprintf(stderr,
                 "brae NOTICE [unread]   (divSchemes is audited per ENTRY -- its consumption has a single choke\n"
                 "brae NOTICE [unread]    point. Other blocks are audited whole: fvSchemes is text-parsed, so their\n"
                 "brae NOTICE [unread]    individual entries are not tracked.)\n");
}


// E5: audit on EVERY exit, including a refusal.
//
// The audit used to sit at the end of the run, so any case brae refuses was never audited -- and the cases
// brae refuses are the biggest and most realistic ones. aerofoilNACA0012 and angledDuctExplicitFixedCoeff
// both stop on fvOptions (A2), so the two stock tutorials most worth auditing were precisely the two that
// never were. A scope guard fires on both paths: normal return and thrown exception.
//
// The report is marked PARTIAL on the exception path, and that distinction is not cosmetic. On an early
// stop, an unread entry may simply not have been REACHED -- residualControl is queried inside the
// iteration loop -- so "unread" stops meaning "unimplemented". Saying so is the difference between a
// gap list and a pile of false positives.
//
// LIFETIME: every registered dict must outlive this object, so declare it AFTER all of them. It holds
// pointers, not copies, because `queried` keeps filling right up to the moment of the report.
class DictAuditScope
{
public:
    void add(const FoamDict& d, std::string label) { entries_.push_back({&d, std::move(label)}); }

    // fvSchemes is audited at SCOPE EXIT, not where it is parsed. Entries are consumed at different
    // points in the run -- the tracer's div(phi,<field>) is read by the functionObject factory, long
    // after parseFvSchemesControls -- so auditing at the parse point reported a consumed entry as
    // unread. Same reason every other dict here reports from the destructor.
    void addFvSchemes(std::string caseDir) { caseDir_ = std::move(caseDir); }

    ~DictAuditScope()
    {
        // The report goes to stderr, which is unbuffered, while the solver's log goes to stdout, which
        // is block-buffered when it is a file. With `2>&1` the two interleave at the flush boundary,
        // which can fall INSIDE a stdout line: the rho mirror's `Time = 30  U 1.2e-03 ...` summary
        // was found with "-- 9 entries brae never looked at" spliced into it, and a gate's parser
        // read `--` as a residual (rho_smoothsolver_vs_openfoam, item 15). Flushing stdout first
        // makes the report land between whole lines, on every driver that owns one of these.
        std::fflush(stdout);
        // Non-zero only when we are unwinding, i.e. brae threw. C++17's uncaught_exceptions() is the
        // supported way to tell a destructor which path it is on.
        const bool partial = std::uncaught_exceptions() > 0;
        for (const auto& e : entries_) auditDict(*e.first, e.second, partial);
        if (!caseDir_.empty()) auditFvSchemes(caseDir_);
    }

private:
    std::vector<std::pair<const FoamDict*, std::string>> entries_;
    std::string caseDir_;
};


}   // namespace brae
