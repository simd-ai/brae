// _cpp REFERENCE implementation -- see simpleControl_cpp.cuh for the OpenFOAM provenance.
#include "simpleControl_cpp.cuh"
#include <regex>

namespace brae {
namespace cpu {

namespace {

// OpenFOAM reads these as `Switch`, which accepts on/off, yes/no, true/false and 1/0.
bool switchOr(const FoamDict& d, const std::string& key, bool def)
{
    const auto* v = d.find(key);
    if (!v || v->empty()) return def;
    const std::string& s = v->back();
    if (s == "true" || s == "on"  || s == "yes" || s == "y" || s == "1") return true;
    if (s == "false"|| s == "off" || s == "no"  || s == "n" || s == "0") return false;
    return def;
}

} // namespace


SimpleControlDict readSimpleControl(const FoamDict& fvSolution)
{
    SimpleControlDict c;
    const FoamDict* s = fvSolution.subDict("SIMPLE");
    if (!s) return c;                      // subOrEmptyDict: an absent block is all-defaults, not an error

    c.nNonOrthogonalCorrectors = s->intOr("nNonOrthogonalCorrectors", 0);
    c.momentumPredictor        = switchOr(*s, "momentumPredictor", true);
    c.transonic                = switchOr(*s, "transonic", false);
    c.consistent               = switchOr(*s, "consistent", false);
    c.frozenFlow               = switchOr(*s, "frozenFlow", false);

    // absTolOnly: the entries are `name value;`, not sub-dictionaries. Read the LEAVES of
    // residualControl directly, in file order -- order is semantics, applyToField returns the first match.
    if (const FoamDict* rc = s->subDict("residualControl"))
    {
        for (const auto& leaf : rc->leaves)
        {
            if (leaf.second.empty()) continue;
            rc->queried.insert(leaf.first);      // consumed, so the unread-key audit does not flag it
            c.residualControl.emplace_back(leaf.first, std::stod(leaf.second.back()));
        }
    }
    return c;
}


label applyToField(const std::vector<std::pair<std::string, scalar>>& ctrl,
                   const std::string& fieldName)
{
    for (std::size_t i = 0; i < ctrl.size(); ++i)
    {
        if (ctrl[i].first == fieldName) return static_cast<label>(i);
        try
        {
            // keyType::match with useRegEx: a quoted key like "(k|epsilon)" is a regex over field names.
            if (std::regex_match(fieldName, compileFoamRegex(ctrl[i].first)))
                return static_cast<label>(i);
        }
        catch (...) {}
    }
    return -1;
}


bool SimpleControl::correctNonOrthogonal()
{
    ++corrNonOrtho_;
    if (corrNonOrtho_ <= d_.nNonOrthogonalCorrectors + 1) return true;
    corrNonOrtho_ = 0;
    return false;
}


bool SimpleControl::criteriaSatisfied(const std::map<std::string, scalar>& initialResiduals) const
{
    if (d_.residualControl.empty()) return false;

    bool achieved = true;
    bool checked  = false;      // OpenFOAM's safety that some check was indeed performed
    for (const auto& kv : initialResiduals)
    {
        const label fieldi = applyToField(d_.residualControl, kv.first);
        if (fieldi == -1) continue;
        checked = true;
        // The INITIAL residual, not the final one -- see the header note.
        achieved = achieved && (kv.second < d_.residualControl[fieldi].second);
    }
    return checked && achieved;
}


bool SimpleControl::loop(label iteration, label maxIters,
                         const std::map<std::string, scalar>& initialResiduals)
{
    // simpleControl.C:136-160. The criteria are only consulted once an iteration has been completed
    // (`initialised_`), so a case can never "converge" before it has computed a single residual.
    if (initialised_ && criteriaSatisfied(initialResiduals))
    {
        converged_ = true;
        return false;               // runTime.writeAndEnd()
    }
    initialised_ = true;
    return iteration <= maxIters;   // runTime.loop()
}

} // namespace cpu
} // namespace brae
