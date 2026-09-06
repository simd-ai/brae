#pragma once
// brae_notice.cuh -- say it out loud when brae ignores, drops, approximates or defaults something.
//
// The rhoSimpleFoam audit found that the expensive bugs were not missing features. They were features
// that were PRESENT AND PARSED but never applied, and that said nothing about it:
//
//   - `inletOutlet` on T was read, masked, and then never resolved -- the outlet enthalpy stayed pinned
//     for the whole run.
//   - constant/fvOptions was never opened by the compressible driver, so a porosity source that dominated
//     the momentum balance simply vanished.
//   - div(phi,K) accepted `linearUpwind` and ran upwind.
//   - a Function1-valued uniformFixedValue silently reused a stale scalar.
//
// Every one of those produced a converged, plausible-looking, wrong field. None printed a word. The
// common failure is not the bug itself -- it is that nothing in the output pointed at the cause, so the
// search started from the field error instead of from the dropped input.
//
// So: any code path that does less than the case asked for MUST call one of these. They are cheap, they
// de-duplicate, and they write to stderr with a stable, greppable prefix:
//
//   brae NOTICE [ignored]      fvOptions: constant/fvOptions present but not read by this solver
//   brae NOTICE [approximated] div(phi,U): limitedLinearV -> upwind (no limitedLinear path for U)
//   brae NOTICE [defaulted]    Prt: patch "(?i).*walls" not matched by name -> model Prt 1.0
//
// Rules of thumb:
//   ignored      -- the case asked for something and brae does nothing with it
//   approximated -- brae does something related but not equal (a scheme downgrade, a formula substitution)
//   defaulted    -- brae could not read a value and fell back
//   applied      -- only for things worth confirming positively (opt-in, quiet by default)
//
// If the difference would change the converged answer materially and brae cannot approximate it honestly,
// THROW instead. A notice is for "less than asked, but still a defensible answer"; a refusal is for
// "this answer would be wrong". When in doubt, throw -- brae's contract is that it never guesses.

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

namespace brae {

namespace detail {

inline std::set<std::string>& noticeSeen()
{
    static std::set<std::string> s;
    return s;
}

// One line per distinct (kind, subject, detail). Repeats are swallowed so a per-face or per-iteration
// call site cannot flood the log.
inline void notice(const char* kind, const std::string& subject, const std::string& detail)
{
    const std::string key = std::string(kind) + '|' + subject + '|' + detail;
    if (!noticeSeen().insert(key).second) return;
    std::fprintf(stderr, "brae NOTICE [%s] %s: %s\n", kind, subject.c_str(), detail.c_str());
}

}   // namespace detail

// The case asked for something and brae does nothing with it.
inline void noticeIgnored(const std::string& subject, const std::string& detail)
{
    detail::notice("ignored", subject, detail);
}

// brae does something related but not equal. Say what it ran INSTEAD, not just that it differs.
inline void noticeApproximated(const std::string& subject, const std::string& detail)
{
    detail::notice("approximated", subject, detail);
}

// The case named one thing and brae runs another that computes the SAME numbers -- a rename, not an
// approximation. Distinct from `approximated` on purpose: reporting an exact equivalence as an
// approximation teaches the reader to discount the approximated lines that really do differ.
inline void noticeEquivalent(const std::string& subject, const std::string& detail)
{
    detail::notice("equivalent", subject, detail);
}

// A value could not be read and brae fell back. Say the fallback value.
inline void noticeDefaulted(const std::string& subject, const std::string& detail)
{
    detail::notice("defaulted", subject, detail);
}

// Positive confirmation, off unless BRAE_NOTICE_APPLIED is set. For the cases where "did it actually
// take effect?" has been the question -- a boundary update, a scheme flag reaching the solver.
inline void noticeApplied(const std::string& subject, const std::string& detail)
{
    if (std::getenv("BRAE_NOTICE_APPLIED")) detail::notice("applied", subject, detail);
}

}   // namespace brae
