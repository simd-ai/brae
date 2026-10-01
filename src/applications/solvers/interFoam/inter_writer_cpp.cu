// interFoam's time directories -- see inter_writer_cpp.cuh.
#include "inter_writer_cpp.cuh"
#include "inter_turbulence_cpp.cuh"
#include "inter_case_cpp.cuh"
#include "inter_amr_cpp.cuh"
#include "inter_waves_cpp.cuh"
#include "displacement_laplacian_fv_motion_solver_cpp.cuh"
#include "dynamic_motion_solver_fv_mesh_cpp.cuh"
#include "foam_token_reader.cuh"
#include "brae_notice.cuh"
#include <zlib.h>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

namespace fs = std::filesystem;

// OpenFOAM's double SMALL (doubleScalar.H)
constexpr scalar kSmall = 1.0e-15;
// Time::maxPrecision_ = 3 - log10(SMALL) (Time.C:83)
constexpr int kMaxTimePrecision = 18;
// UList::writeList's shortLen for a contiguous type (UListIO.C:80-135): ten entries or fewer on one line
constexpr std::size_t kShortList = 10;

std::string fmt(
    scalar v,
    int precision)
{
    std::ostringstream o;
    o.precision(precision);
    o << v;
    return o.str();
}

std::string fmt(
    const vector& v,
    int precision)
{
    return "(" + fmt(v.x, precision) + " " + fmt(v.y, precision) + " " + fmt(v.z, precision) + ")";
}

const char* listTypeName(scalar) { return "List<scalar>"; }
const char* listTypeName(const vector&) { return "List<vector>"; }

bool sameValue(
    scalar a,
    scalar b)
{
    return a == b;
}

bool sameValue(
    const vector& a,
    const vector& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Ostream::writeKeyword: the indent, the keyword, and spaces to column 16 past the indent (at least one)
void keyword(
    std::ostringstream& os,
    int indent,
    const std::string& k)
{
    os << std::string(static_cast<std::size_t>(indent), ' ') << k;
    const int pad = std::max(1, 16 - static_cast<int>(k.size()));
    os << std::string(static_cast<std::size_t>(pad), ' ');
}

// Field::writeEntry (Field.C:727-748): `uniform v` when every entry compares equal (UList::uniform, which
// needs at least one), else `nonuniform List<T> ` and the list -- one line of ten or fewer, else the count,
// the values one per line at column zero, and the closing parenthesis on its own line before the `;`.
template <typename T>
void listEntry(
    std::ostringstream& os,
    int indent,
    const std::string& k,
    const std::vector<T>& values,
    int precision)
{
    keyword(os, indent, k);
    bool uniform = !values.empty();
    for (std::size_t i = 1; uniform && i < values.size(); ++i)
    {
        uniform = sameValue(values[i], values[0]);
    }
    if (uniform)
    {
        os << "uniform " << fmt(values[0], precision) << ";\n";
        return;
    }
    os << "nonuniform " << listTypeName(T{}) << " ";
    if (values.size() <= kShortList)
    {
        os << values.size() << "(";
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (i > 0)
            {
                os << " ";
            }
            os << fmt(values[i], precision);
        }
        os << ");\n";
        return;
    }
    os << "\n" << values.size() << "\n(\n";
    for (const T& v : values)
    {
        os << fmt(v, precision) << "\n";
    }
    os << ")\n;\n";
}

void wordEntry(
    std::ostringstream& os,
    int indent,
    const std::string& k,
    const std::string& w)
{
    keyword(os, indent, k);
    os << w << ";\n";
}

void scalarEntry(
    std::ostringstream& os,
    int indent,
    const std::string& k,
    scalar v,
    int precision)
{
    keyword(os, indent, k);
    os << fmt(v, precision) << ";\n";
}

// the patch types OpenFOAM treats as constraints: a patch of one of these takes its field entry from the
// mesh (fvPatchFieldNew.C), so a field file need not name it
bool isConstraintType(const std::string& t)
{
    return t == "empty" || t == "wedge" || t == "symmetryPlane" || t == "symmetry" || t == "cyclic"
        || t == "cyclicAMI" || t == "cyclicACMI" || t == "cyclicSlip" || t == "processor"
        || t == "processorCyclic" || t == "nonuniformTransformCyclic";
}

const std::vector<std::string>* leaf(
    const FoamDict& d,
    const std::string& k)
{
    return d.find(k);
}

std::string leafWord(
    const FoamDict& d,
    const std::string& k,
    const std::string& dflt)
{
    const std::vector<std::string>* v = leaf(d, k);
    return (v && !v->empty()) ? (*v)[0] : dflt;
}

bool leafScalar(
    const FoamDict& d,
    const std::string& k,
    scalar& out)
{
    const std::vector<std::string>* v = leaf(d, k);
    if (!v || v->size() != 1)
    {
        return false;
    }
    char* end = nullptr;
    out = static_cast<scalar>(std::strtod((*v)[0].c_str(), &end));
    return end && *end == '\0';
}

bool leafBool(
    const FoamDict& d,
    const std::string& k)
{
    const std::string w = leafWord(d, k, "false");
    return w == "true" || w == "on" || w == "yes" || w == "y" || w == "t";
}

// `uniform (x y z)` as the tokenizer leaves it -- `(`, three values, `)` or `(x`, `y`, `z)` depending on
// spacing -- joined and re-read here rather than guessed at
bool uniformVector(
    const std::vector<std::string>& toks,
    vector& out)
{
    if (toks.empty() || toks[0] != "uniform")
    {
        return false;
    }
    std::string joined;
    for (std::size_t i = 1; i < toks.size(); ++i)
    {
        joined += toks[i] + " ";
    }
    for (char& c : joined)
    {
        if (c == '(' || c == ')')
        {
            c = ' ';
        }
    }
    std::istringstream in(joined);
    double x = 0, y = 0, z = 0;
    if (!(in >> x >> y >> z))
    {
        return false;
    }
    std::string rest;
    if (in >> rest)
    {
        return false;
    }
    out = vector{static_cast<scalar>(x), static_cast<scalar>(y), static_cast<scalar>(z)};
    return true;
}

// a plain `(x y z)` -- a point read by lookup, as rotatingWallVelocity's origin and axis are -- through
// uniformVector's joining of however the tokenizer split it
bool plainVector(
    const std::vector<std::string>& toks,
    vector& out)
{
    std::vector<std::string> u{"uniform"};
    u.insert(
        u.end(),
        toks.begin(),
        toks.end());
    return uniformVector(
        u,
        out);
}

// A constant Function1 or PatchFunction1 as the template spells it: `constant v`, a bare v, and for a
// PatchFunction1 also `uniform v` (Function1New.C:83-98, PatchFunction1New.C:73-96). False for any other
// form -- a table, a coded function, a sub-dictionary -- which is refused, not echoed.
bool constantFunction1(
    const std::vector<std::string>& toks,
    bool allowUniform,
    scalar& out)
{
    std::size_t first = 0;
    if (!toks.empty() && (toks[0] == "constant" || (allowUniform && toks[0] == "uniform")))
    {
        first = 1;
    }
    if (toks.size() != first + 1)
    {
        return false;
    }
    char* end = nullptr;
    out = static_cast<scalar>(std::strtod(toks[first].c_str(), &end));
    return end && end != toks[first].c_str() && *end == '\0';
}

bool constantFunction1(
    const std::vector<std::string>& toks,
    bool allowUniform,
    vector& out)
{
    std::size_t first = 0;
    if (!toks.empty() && (toks[0] == "constant" || (allowUniform && toks[0] == "uniform")))
    {
        first = 1;
    }
    std::vector<std::string> asUniform{"uniform"};
    asUniform.insert(
        asUniform.end(),
        toks.begin() + static_cast<std::ptrdiff_t>(first),
        toks.end());
    return uniformVector(
        asUniform,
        out);
}

// PatchFunction1::writeData (PatchFunction1.C:180-187) writes coordinateScaling's entries first, read from
// the dictionary the function was built from -- the patch dictionary, or `<key>Coeffs` for the `constant`
// word form (PatchFunction1New.C:138-145, coordinateScaling.C:40-62). brae neither applies nor writes it.
bool hasCoordinateScaling(
    const FoamDict& d,
    const std::string& k)
{
    if (d.subDict(k + "Coeffs"))
    {
        return true;
    }
    for (const char* s : {"coordinateSystem", "scale1", "scale2", "scale3"})
    {
        if (d.find(s) || d.subDict(s))
        {
            return true;
        }
    }
    return false;
}

// OpenFOAM's Switch as a bool entry is read (Switch.C:233-236): its words, and a label 0/1 -- which is how
// a bool is WRITTEN (bool.C:60-66). -1 for anything else.
int ofSwitchToken(const std::string& w);

// `useImplicit` as fvPatchFieldBase::readDict reads it (readIfPresent<bool>, fvPatchFieldBase.C:109): any
// Switch spelling, `1` and `any` included. An unreadable one counts as set -- OpenFOAM would stop on it.
bool setsUseImplicit(const FoamDict& pd)
{
    return pd.find("useImplicit") && ofSwitchToken(leafWord(pd, "useImplicit", "false")) != 0;
}

int ofSwitchToken(const std::string& w)
{
    if (w == "true" || w == "on" || w == "yes" || w == "y" || w == "t" || w == "1" || w == "any")
    {
        return 1;
    }
    if (w == "false" || w == "off" || w == "no" || w == "n" || w == "f" || w == "0" || w == "none")
    {
        return 0;
    }
    return -1;
}

// One token of a dictionary OpenFOAM writes back as it holds it (ISstream.C:712-790): a label as its
// parsed value (`007` comes back `7`, `-0` as `0`), any other number as a scalar at writePrecision (`3.0`
// comes back `3`, `0.05` as 0.050000000000000003, a label too big for int32 as a scalar), a word
// unchanged. A leading `+` does not start a number there, so `+1` stays the word `+1`.
std::string dictToken(
    const std::string& t,
    int precision)
{
    if (t.empty() || t[0] == '+')
    {
        return t;
    }
    static const std::regex label("^-?[0-9]+$");
    if (std::regex_match(t, label))
    {
        // int32IO.C:42-58: strtoimax base 10, then the int32 range
        errno = 0;
        const long long v = std::strtoll(t.c_str(), nullptr, 10);
        if (errno == 0
            && v >= std::numeric_limits<int32_t>::min()
            && v <= std::numeric_limits<int32_t>::max())
        {
            return std::to_string(v);
        }
    }
    char* end = nullptr;
    const double v = std::strtod(t.c_str(), &end);
    if (end && end != t.c_str() && *end == '\0')
    {
        // readScalar rounds |x| <= VSMALL to zero on the re-read (ISstream.C:782, Scalar.C:104-110)
        if (v >= -1.0e-300 && v <= 1.0e-300)
        {
            return "0";
        }
        return fmt(static_cast<scalar>(v), precision);
    }
    return t;
}

// Whether a dictionary token re-emits through dictToken as OpenFOAM re-emits it. ISstream makes each of
// `[ ] , : = + * /` a punctuation token of its own (ISstream.C:577-594) where brae's tokenizer keeps it in
// the word; `<` opens a compound (`List<scalar> N(...)`, ISstream.C:806-809); and strtod reads `0x10`,
// `inf`, `nan` and `infinity` as numbers ISstream does not.
bool echoableToken(const std::string& t)
{
    if (t.find_first_of("[],:=+*/<>") != std::string::npos)
    {
        return false;
    }
    static const std::regex notNumber("^-?(0[xX]|[iI][nN][fF]|[nN][aA][nN])");
    return !std::regex_search(t, notNumber);
}

// A coupled patch's value after a field expression. Every GeometricField function ends in
// correctLocalBoundaryConditions() (GeometricFieldFunctionsM.C:50, localConsistency 1 by default), and a
// coupled patch's evaluateLocal IS its evaluate (coupledFvPatchField.H:198-205): w*f_P + (1 - w)*f_N of the
// RESULT's cells. So `1.0/UEqn.A()` and `p_rgh + rho*gh` on a cyclicAMI are that, and not the arithmetic of
// their operands' patch values -- MEASURED on RAS/mixerVesselAMI: the reciprocal of A's patch value put rAU
// 1.6e-02 off OpenFOAM's on the AMI pair, and p_rgh_b + rho_b*gh_b put p 6.9e-05 off (13.5 Pa).
std::vector<scalar> coupledLocalValue(
    const FvPatch& p,
    const std::vector<scalar>& cells)
{
    CoupledCyclicPatchField<scalar> local(p);
    local.evaluate(cells);
    return local.value();
}

// UList::writeList for a label list (UListIO.C:82-178), the rule every labelList file goes by: more than one
// entry, all equal, as `N{v}`; ten or fewer on one line, `N(a b c)`; otherwise a newline, the count, and one
// entry per line between parentheses on lines of their own, ending in a newline.
std::string labelListText(const std::vector<label>& v)
{
    const std::size_t n = v.size();
    bool uniform = n > 1;
    for (std::size_t i = 1; uniform && i < n; ++i)
    {
        uniform = v[i] == v[0];
    }
    std::ostringstream os;
    if (uniform)
    {
        os << n << "{" << v[0] << "}";
        return os.str();
    }
    if (n <= kShortList)
    {
        os << n << "(";
        for (std::size_t i = 0; i < n; ++i)
        {
            os << (i ? " " : "") << v[i];
        }
        os << ")";
        return os.str();
    }
    os << "\n" << n << "\n(";
    for (const label x : v)
    {
        os << "\n" << x;
    }
    os << "\n)\n";
    return os.str();
}

// ...and a list of elements that are not contiguous (a face, a splitCell8): one line only when it holds at
// most one, otherwise the multi-line form, each element as its own text
std::string compoundListText(const std::vector<std::string>& elements)
{
    std::ostringstream os;
    if (elements.size() <= 1)
    {
        os << elements.size() << "(";
        for (const std::string& e : elements)
        {
            os << e;
        }
        os << ")";
        return os.str();
    }
    os << "\n" << elements.size() << "\n(";
    for (const std::string& e : elements)
    {
        os << "\n" << e;
    }
    os << "\n)\n";
    return os.str();
}

// A scalar list as a dictionary entry holds it: dict.add streams UList::writeList's form into a primitive
// entry, re-read as tokens (primitiveEntryTemplates.C:36-46) and written back joined by single spaces
// (primitiveEntryIO.C:280-314). So `N ( a b )` on one line at any length, `0 ( )` when empty, and -- for
// more than one entry, all equal -- `N { v }` (UListIO.C:119-123), measured on floatingObject's body at
// rest. Each number is a token re-emitted, so dictToken. BRAE_CONTROL_RBSTATE_PAREN=1 writes the uniform
// list in the paren form instead, for the write gate's control.
std::string dictScalarList(
    const std::vector<scalar>& v,
    int precision)
{
    bool uniform = v.size() > 1;
    for (std::size_t i = 1; uniform && i < v.size(); ++i)
    {
        uniform = v[i] == v[0];
    }
    if (uniform && std::getenv("BRAE_CONTROL_RBSTATE_PAREN") != nullptr)
    {
        uniform = false;
    }
    std::string out = std::to_string(v.size());
    if (uniform)
    {
        return out + " { " + dictToken(fmt(v[0], precision), precision) + " }";
    }
    out += " (";
    for (const scalar x : v)
    {
        out += " " + dictToken(fmt(x, precision), precision);
    }
    return out + " )";
}

// The patch's entry in the template's boundaryField, matched as the reader and writeVolField match it:
// the exact name first, then a group or a regex, the LAST matching entry winning.
const FoamDict* patchDict(
    const FoamDict& boundary,
    const FvPatch& p)
{
    for (const auto& s : boundary.subs)
    {
        if (s.first == p.name)
        {
            return &s.second;
        }
    }
    const FoamDict* hit = nullptr;
    for (const auto& s : boundary.subs)
    {
        bool match = false;
        for (const std::string& g : p.inGroups)
        {
            if (s.first == g)
            {
                match = true;
            }
        }
        if (!match)
        {
            try
            {
                const std::regex re = compileFoamRegex(s.first);
                if (std::regex_match(p.name, re))
                {
                    match = true;
                }
                for (const std::string& g : p.inGroups)
                {
                    if (std::regex_match(g, re))
                    {
                        match = true;
                    }
                }
            }
            catch (const std::regex_error&)
            {
            }
        }
        if (match)
        {
            hit = &s.second;
        }
    }
    return hit;
}

[[noreturn]] void refuseWrite(
    const std::string& field,
    const FvPatch& p,
    const std::string& what)
{
    throw std::runtime_error(
        "brae interFoam writer: " + field + " on patch `" + p.name + "`: " + what
        + ". OpenFOAM writes it through the condition's own write(); brae writes only the ones it has "
          "transcribed (inter_writer_cpp.cu).");
}

// `<key> constant <value>;` -- Function1::writeData then Constant::writeData (Function1.C:163-166,
// Constant.C:116-122), or ConstantField::writeData for a PatchFunction1 (ConstantField.C:314-329): the
// `constant` word whichever of the accepted forms the file used.
template <typename T>
void constantFunction1Entry(
    std::ostringstream& os,
    const std::string& field,
    const FvPatch& p,
    const FoamDict& pd,
    const std::string& key,
    bool allowUniform,
    int precision)
{
    const std::vector<std::string>* toks = leaf(pd, key);
    T c{};
    if (!toks || !constantFunction1(*toks, allowUniform, c))
    {
        refuseWrite(field, p, "`" + key + "` is not a constant Function1; only `constant <value>` is written");
    }
    keyword(os, 8, key);
    os << "constant " << fmt(c, precision) << ";\n";
}

// fvPatchField::write (fvPatchField.C:379-391) for a condition whose dictionary constructor runs
// fvPatchFieldBase::readDict (fvPatchFieldBase.C:106-110): the type, then `patchType` when the file set one.
// `useImplicit` would follow; brae does not honour it, so it is refused.
void baseEntries(
    std::ostringstream& os,
    const std::string& field,
    const FvPatch& p,
    const FoamDict& pd,
    const std::string& type)
{
    if (setsUseImplicit(pd))
    {
        refuseWrite(field, p, "it sets `useImplicit`, which brae does not honour");
    }
    wordEntry(os, 8, "type", type);
    const std::string patchType = leafWord(pd, "patchType", "");
    if (!patchType.empty())
    {
        wordEntry(os, 8, "patchType", patchType);
    }
}

// wallFunctionBlenders::writeEntries (wallFunctionBlenders.C:88-96): the blending word always, `n` for
// binomial only. Both defaults belong to the condition, not to the blender (the constructors at
// epsilonWallFunctionFvPatchScalarField.C:413 STEPWISE/2, omegaWallFunctionFvPatchScalarField.C:405
// BINOMIAL/2, nutkWallFunctionFvPatchScalarField.C:216 STEPWISE/4): a shared `stepwise` default wrote
// `blending stepwise` for RAS/waterChannel's omega walls where OpenFOAM writes `binomial` and `n 2`.
void blendingEntries(
    std::ostringstream& os,
    const FoamDict& pd,
    const std::string& defaultBlending,
    scalar defaultN,
    int precision)
{
    const std::string b = leafWord(pd, "blending", defaultBlending);
    wordEntry(os, 8, "blending", b);
    if (b == "binomial")
    {
        scalar n = defaultN;
        leafScalar(pd, "n", n);
        scalarEntry(os, 8, "n", n, precision);
    }
}

// wallFunctionCoefficients::writeEntries (wallFunctionCoefficients.C:85-93): each only where it differs
void wallCoefficientEntries(
    std::ostringstream& os,
    const FoamDict& pd,
    int precision)
{
    const struct
    {
        const char* k;
        scalar dflt;
    } coeffs[] = {{"Cmu", 0.09}, {"kappa", 0.41}, {"E", 9.8}};
    for (const auto& c : coeffs)
    {
        scalar v = c.dflt;
        if (leafScalar(pd, c.k, v) && v != c.dflt)
        {
            scalarEntry(os, 8, c.k, v, precision);
        }
    }
}

// One patch of a VOLUME field, through its condition's write(). `derived` is p: every non-constraint
// patch `calculated` with the given values, exactly as OpenFOAM writes a field it built itself.
template <typename T>
void volPatch(
    std::ostringstream& os,
    const std::string& field,
    const FvPatch& p,
    const FoamDict* pd,
    const fvPatchField<T>& bc,
    const std::vector<T>* derived,
    const std::vector<T>* stored,
    const std::vector<T>* storedGradient,
    int precision)
{
    os << "    " << p.name << "\n    {\n";
    std::string type = pd ? leafWord(*pd, "type", "") : std::string();
    if (derived || type.empty())
    {
        type = isConstraintType(p.type) ? p.type : (derived ? std::string("calculated") : std::string());
    }
    if (type.empty())
    {
        refuseWrite(field, p, "the start directory's field file has no entry for this patch");
    }
    // `stored`: an OLD-TIME level's values through this patch's own type -- the level is a copy of the
    // field, conditions and all (GeometricField.C:960-972), so only the values differ
    const std::vector<T>& value = derived ? *derived : (stored ? *stored : bc.value());
    // fvPatchField::write emits `patchType` and `useImplicit` for every condition whose dictionary
    // constructor runs fvPatchFieldBase::readDict (fvPatchField.C:383-390). The branches below that carry
    // them say so; everywhere else a file that sets one is refused rather than written without it.
    const bool writesPatchType = type == "slip" || type == "turbulentIntensityKineticEnergyInlet"
                              || type == "cyclicACMI" || type == "porousBafflePressure"
                              || type == "permeableAlphaPressureInletOutletVelocity";
    if (pd && !derived && !writesPatchType && (leaf(*pd, "patchType") || setsUseImplicit(*pd)))
    {
        refuseWrite(field, p, "it sets `patchType` or `useImplicit`, which this condition's branch does not write");
    }

    // fvPatchField::write (fvPatchField.C:379-391) and the overrides that add nothing: the type alone.
    // wedge/symmetryPlane/symmetry/zeroGradient have no write() of their own; empty and noSlip and cyclic
    // write the base's (emptyFvPatchField.C:127-131, noSlipFvPatchVectorField.C:90-93,
    // cyclicFvPatchField.C:244-247).
    if (type == "empty" || type == "wedge" || type == "symmetryPlane" || type == "symmetry"
     || type == "cyclic" || type == "zeroGradient" || type == "noSlip")
    {
        wordEntry(os, 8, "type", type);
    }
    // calculatedFvPatchField.C:212-216, fixedValueFvPatchField.C:154-158, kqRWallFunctionFvPatchField.C:
    // 103-107 (zeroGradient's write, then the value), movingWallVelocityFvPatchVectorField.C:150-154 (the
    // base's write and the value -- the Uwall its updateCoeffs assigned, :128-146)
    else if (type == "calculated" || type == "fixedValue" || type == "kqRWallFunction"
          || type == "movingWallVelocity")
    {
        wordEntry(os, 8, "type", type);
        listEntry(os, 8, "value", value, precision);
    }
    // inletOutletFvPatchField.C:131-137: [phi], inletValue (the refValue), value
    else if (type == "inletOutlet")
    {
        wordEntry(os, 8, "type", type);
        const std::string phiName = leafWord(*pd, "phi", "phi");
        if (phiName != "phi")
        {
            wordEntry(os, 8, "phi", phiName);
        }
        listEntry(os, 8, "inletValue", bc.refValues(), precision);
        listEntry(os, 8, "value", value, precision);
    }
    // fixedFluxPressureFvPatchScalarField.C:167-171 over fixedGradientFvPatchField.C:237-241: the
    // gradient the last assembly SET (constrainPressure), then the value
    else if (type == "fixedFluxPressure")
    {
        const std::vector<T>* grad = bc.refGradPtr();
        if (!grad)
        {
            refuseWrite(field, p, "a fixedFluxPressure patch object carries no gradient");
        }
        wordEntry(os, 8, "type", type);
        // BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT=1 writes the gradient the dictionary constructor
        // leaves when the file has none (fixedFluxPressureFvPatchScalarField.C:57-61, Zero) -- the shared
        // writer's defect, echoing construction state for the gradient the solver set. The control
        // tests/interfoam_write_vs_openfoam.sh must go red on; never the default.
        const char* constructionGradient = std::getenv("BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT");
        if (constructionGradient && std::string(constructionGradient) == "1")
        {
            listEntry(os, 8, "gradient", std::vector<T>(grad->size(), T{}), precision);
        }
        else
        {
            listEntry(os, 8, "gradient", *grad, precision);
        }
        listEntry(os, 8, "value", value, precision);
    }
    else if (type == "totalPressure")
    {
        // totalPressureFvPatchScalarField.C:238-248: [U], [phi], rho, psi, gamma always, p0, value
        wordEntry(os, 8, "type", type);
        const std::string UName = leafWord(*pd, "U", "U");
        const std::string phiName = leafWord(*pd, "phi", "phi");
        if (UName != "U")
        {
            wordEntry(os, 8, "U", UName);
        }
        if (phiName != "phi")
        {
            wordEntry(os, 8, "phi", phiName);
        }
        wordEntry(os, 8, "rho", leafWord(*pd, "rho", "rho"));
        wordEntry(os, 8, "psi", leafWord(*pd, "psi", "none"));
        scalar gamma = 1;
        leafScalar(*pd, "gamma", gamma);
        scalarEntry(os, 8, "gamma", gamma, precision);
        // p0 as the patch object holds it (TotalPressurePatchField::refValues), which is the file's p0,
        // uniform or not
        listEntry(os, 8, "p0", bc.refValues(), precision);
        listEntry(os, 8, "value", value, precision);
    }
    else if (type == "pressureInletOutletVelocity")
    {
        // pressureInletOutletVelocityFvPatchVectorField.C:187-200: [phi], [tangentialVelocity], value
        wordEntry(os, 8, "type", type);
        const std::string phiName = leafWord(*pd, "phi", "phi");
        if (phiName != "phi")
        {
            wordEntry(os, 8, "phi", phiName);
        }
        if (const std::vector<std::string>* tv = leaf(*pd, "tangentialVelocity"))
        {
            vector t{0, 0, 0};
            if (!uniformVector(*tv, t))
            {
                refuseWrite(field, p, "tangentialVelocity is not `uniform (x y z)`; only that form is written");
            }
            keyword(os, 8, "tangentialVelocity");
            os << "uniform " << fmt(t, precision) << ";\n";
        }
        listEntry(os, 8, "value", value, precision);
    }
    else if (type == "flowRateInletVelocity")
    {
        // flowRateInletVelocityFvPatchVectorField.C:241-255: the flow rate's Function1, then for a MASS
        // flow rate [rho] and [rhoInlet], then [extrapolateProfile], then value
        wordEntry(os, 8, "type", type);
        const bool volumetric = leaf(*pd, "volumetricFlowRate") != nullptr;
        const std::string key = volumetric ? "volumetricFlowRate" : "massFlowRate";
        const std::vector<std::string>* fr = leaf(*pd, key);
        if (!fr)
        {
            refuseWrite(field, p, "flowRateInletVelocity names neither volumetricFlowRate nor massFlowRate "
                                  "as a plain entry (a Function1 sub-dictionary is not written)");
        }
        // Function1::writeData then Constant::writeData: `<key> constant <value>;` -- the only Function1
        // written here; a table, a file or a coded one is refused rather than echoed
        std::string v;
        if (fr->size() == 2 && (*fr)[0] == "constant")
        {
            v = (*fr)[1];
        }
        else if (fr->size() == 1)
        {
            v = (*fr)[0];
        }
        else
        {
            refuseWrite(field, p, "the flow rate's Function1 is not `constant <value>`; only that is written");
        }
        char* end = nullptr;
        const scalar q = static_cast<scalar>(std::strtod(v.c_str(), &end));
        if (!end || *end != '\0')
        {
            refuseWrite(field, p, "the flow rate `" + v + "` is not a number");
        }
        keyword(os, 8, key);
        os << "constant " << fmt(q, precision) << ";\n";
        if (!volumetric)
        {
            const std::string rhoName = leafWord(*pd, "rho", "rho");
            if (rhoName != "rho")
            {
                wordEntry(os, 8, "rho", rhoName);
            }
            scalar rhoInlet = 0;
            if (leafScalar(*pd, "rhoInlet", rhoInlet))
            {
                scalarEntry(os, 8, "rhoInlet", rhoInlet, precision);
            }
        }
        if (leafBool(*pd, "extrapolateProfile"))
        {
            wordEntry(os, 8, "extrapolateProfile", "true");
        }
        listEntry(os, 8, "value", value, precision);
    }
    else if (type == "epsilonWallFunction")
    {
        // epsilonWallFunctionFvPatchScalarField.C:351-360 and :665-673: blending, [lowReCorrection],
        // the coefficients that differ, value
        wordEntry(os, 8, "type", type);
        blendingEntries(
            os,
            *pd,
            "stepwise",
            scalar(2),
            precision);
        if (leafBool(*pd, "lowReCorrection"))
        {
            wordEntry(os, 8, "lowReCorrection", "true");
        }
        wallCoefficientEntries(os, *pd, precision);
        listEntry(os, 8, "value", value, precision);
    }
    else if (type == "omegaWallFunction")
    {
        // omegaWallFunctionFvPatchScalarField.C:346-355 and :655-663: blending, [beta1], coefficients,
        // value
        wordEntry(os, 8, "type", type);
        blendingEntries(
            os,
            *pd,
            "binomial",
            scalar(2),
            precision);
        scalar beta1 = 0.075;
        if (leafScalar(*pd, "beta1", beta1) && beta1 != scalar(0.075))
        {
            scalarEntry(os, 8, "beta1", beta1, precision);
        }
        wallCoefficientEntries(os, *pd, precision);
        listEntry(os, 8, "value", value, precision);
    }
    else if (type == "nutkWallFunction")
    {
        // nutkWallFunctionFvPatchScalarField.C:296-304: nutWallFunction's write -- the type, [U] and the
        // coefficients that differ (nutWallFunctionFvPatchScalarField.C:73-80, :188-195) -- then its own
        // blending, then the value. writeLocalEntries is not virtual, so the blending is written once
        // (OpenFOAM's own RAS/damBreak output shows it once).
        wordEntry(os, 8, "type", type);
        const std::string UName = leafWord(*pd, "U", "");
        if (!UName.empty())
        {
            wordEntry(os, 8, "U", UName);
        }
        wallCoefficientEntries(os, *pd, precision);
        blendingEntries(
            os,
            *pd,
            "stepwise",
            scalar(4),
            precision);
        listEntry(os, 8, "value", value, precision);
    }
    // slip and its bases basicSymmetry and transform declare no write() (slipFvPatchField.H,
    // basicSymmetryFvPatchField.H, transformFvPatchField.H) for any Type: the base's, no value
    else if (type == "slip")
    {
        baseEntries(
            os,
            field,
            p,
            *pd,
            type);
    }
    // constantAlphaContactAngleFvPatchScalarField.C:113-121 over alphaContactAngleTwoPhaseFvPatchScalarField.C:
    // 154-161 over fixedGradientFvPatchField.C:237-241: type, gradient, limit, theta0, value. No
    // patchType: the dictionary constructor bypasses fixedGradient's (alphaContactAngleTwoPhase...C:70). The
    // gradient is the one correctContactAngle last set; an OLD-TIME level carries the one copied when the
    // level was created (GeometricField.C:949), later assignments being values only (fvPatchField.C:552-558).
    else if (type == "constantAlphaContactAngle")
    {
        const std::vector<T>* grad = stored ? storedGradient : bc.refGradPtr();
        if (!grad || grad->size() != value.size())
        {
            refuseWrite(field, p, stored ? "the old-time level's contact-angle gradient was not captured"
                                         : "the contact-angle patch object carries no gradient");
        }
        // alphaContactAngleTwoPhaseFvPatchScalarField.C:71: `limit` is mandatory
        const std::string limit = leafWord(*pd, "limit", "");
        if (limit != "none" && limit != "gradient" && limit != "zeroGradient" && limit != "alpha")
        {
            refuseWrite(field, p, "`limit` is missing or not one of none/gradient/zeroGradient/alpha");
        }
        // constantAlphaContactAngleFvPatchScalarField.C:57 reads it with get<scalar>
        scalar theta0 = 0;
        if (!leafScalar(*pd, "theta0", theta0))
        {
            refuseWrite(field, p, "`theta0` is missing or not a plain number");
        }
        wordEntry(os, 8, "type", type);
        listEntry(os, 8, "gradient", *grad, precision);
        wordEntry(os, 8, "limit", limit);
        scalarEntry(os, 8, "theta0", theta0, precision);
        listEntry(os, 8, "value", value, precision);
    }
    // variableHeightFlowRateFvPatchField.C:169-176: fvPatchField's write (the type), [phi], lowerBound,
    // upperBound, value -- not mixed's, so no refValue or valueFraction. Both bounds are read without a
    // default (.C:81-82) and never changed.
    else if (type == "variableHeightFlowRate")
    {
        scalar lowerBound = 0;
        scalar upperBound = 0;
        if (!leafScalar(*pd, "lowerBound", lowerBound) || !leafScalar(*pd, "upperBound", upperBound))
        {
            refuseWrite(field, p, "`lowerBound` or `upperBound` is not a plain number");
        }
        wordEntry(os, 8, "type", type);
        const std::string phiName = leafWord(*pd, "phi", "phi");
        if (phiName != "phi")
        {
            wordEntry(os, 8, "phi", phiName);
        }
        scalarEntry(os, 8, "lowerBound", lowerBound, precision);
        scalarEntry(os, 8, "upperBound", upperBound, precision);
        listEntry(os, 8, "value", value, precision);
    }
    // variableHeightFlowRateInletVelocityFvPatchVectorField.C:145-154: the type, the flow rate's Function1,
    // alpha, then the value updateCoeffs last stored (.C:131-141)
    else if (type == "variableHeightFlowRateInletVelocity")
    {
        const std::string alphaName = leafWord(*pd, "alpha", "");
        if (alphaName.empty())
        {
            refuseWrite(field, p, "it names no `alpha`");
        }
        wordEntry(os, 8, "type", type);
        constantFunction1Entry<scalar>(
            os,
            field,
            p,
            *pd,
            "flowRate",
            false,
            precision);
        wordEntry(os, 8, "alpha", alphaName);
        listEntry(os, 8, "value", value, precision);
    }
    // turbulentIntensityKineticEnergyInletFvPatchScalarField.C:149-159: fvPatchField's write (the type,
    // [patchType]; its constructor runs readDict at :79), intensity, [U], [phi], value -- NOT inletOutlet's,
    // so no inletValue: the refValue is rebuilt from U at every updateCoeffs (:142)
    else if (type == "turbulentIntensityKineticEnergyInlet")
    {
        scalar intensity = 0;
        if (!leafScalar(*pd, "intensity", intensity) || intensity != bc.turbulentInletCoefficient())
        {
            refuseWrite(field, p, "`intensity` is missing, not a number, or not the one the patch ran with");
        }
        // brae's reader takes no `U` entry, so the solve read `U` whatever the file says
        if (leafWord(*pd, "U", "U") != "U")
        {
            refuseWrite(field, p, "it names a U other than `U`, which brae does not honour");
        }
        baseEntries(
            os,
            field,
            p,
            *pd,
            type);
        scalarEntry(os, 8, "intensity", intensity, precision);
        const std::string phiName = leafWord(*pd, "phi", "phi");
        if (phiName != "phi")
        {
            wordEntry(os, 8, "phi", phiName);
        }
        listEntry(os, 8, "value", value, precision);
    }
    // turbulentMixingLengthDissipationRateInletFvPatchScalarField.C:166-176: the type, then mixingLength,
    // phi and k ALWAYS, then value. Its constructor never runs readDict (:84-99), so no patchType.
    else if (type == "turbulentMixingLengthDissipationRateInlet")
    {
        scalar mixingLength = 0;
        if (!leafScalar(*pd, "mixingLength", mixingLength) || mixingLength != bc.turbulentInletCoefficient())
        {
            refuseWrite(field, p, "`mixingLength` is missing, not a number, or not the one the patch ran with");
        }
        // brae's reader takes no `k` entry, so the solve read `k` whatever the file says
        if (leafWord(*pd, "k", "k") != "k")
        {
            refuseWrite(field, p, "it names a k other than `k`, which brae does not honour");
        }
        wordEntry(os, 8, "type", type);
        scalarEntry(os, 8, "mixingLength", mixingLength, precision);
        wordEntry(os, 8, "phi", leafWord(*pd, "phi", "phi"));
        wordEntry(os, 8, "k", "k");
        listEntry(os, 8, "value", value, precision);
    }
    // cyclicACMIFvPatchField.C:974-983 and cyclicAMIFvPatchField.C:1010-1019: fvPatchField's write (type,
    // [patchType]), the stored coupled value -- coupledFvPatchField::evaluate's lerp, which only a coupled
    // patch object holds -- and `neighbourValue` only while patchNeighbourFieldPtr_ is set, which serially
    // is only a start file's (cyclicAMIFvPatchField.C:75-86), refused here
    else if (type == "cyclicACMI" || type == "cyclicAMI")
    {
        if (!derived && !stored && !bc.coupled())
        {
            refuseWrite(field, p, "the patch object is not coupled, so it holds no coupled value");
        }
        // pd is null for a field brae derives (p) or a file with no entry for the patch: the patch type's
        // own constraint field, type and value only
        if (pd && leaf(*pd, "neighbourValue"))
        {
            refuseWrite(field, p, "the start file carries `neighbourValue`, which OpenFOAM echoes until an "
                                  "assignment drops it (cyclicAMIFvPatchField.C:75-86, :1025-1066)");
        }
        if (pd)
        {
            baseEntries(
                os,
                field,
                p,
                *pd,
                type);
        }
        else
        {
            wordEntry(os, 8, "type", type);
        }
        listEntry(os, 8, "value", value, precision);
    }
    // porousBafflePressureFvPatchField.C:200-209 over fixedJumpFvPatchField.C:243-271: type, patchType
    // (the file's, else interfaceFieldType() -- cyclic), the OWNER's jump, value, [phi], [rho], D, I,
    // length, uniformJump. relax/jump0/minJump are fixedJump state brae refuses at construction.
    else if (type == "porousBafflePressure")
    {
        if (!bc.isPorousBafflePressure())
        {
            refuseWrite(field, p, "the template names porousBafflePressure but the patch object is not one");
        }
        if (setsUseImplicit(*pd) || leaf(*pd, "relax") || leaf(*pd, "minJump"))
        {
            refuseWrite(field, p, "useImplicit, relax or minJump is set; none is ported");
        }
        const int uniformJump = ofSwitchToken(leafWord(*pd, "uniformJump", "false"));
        scalar length = 0;
        if (uniformJump < 0 || !leafScalar(*pd, "length", length))
        {
            refuseWrite(field, p, "`uniformJump` is not a switch, or `length` is missing or not a number");
        }
        wordEntry(os, 8, "type", type);
        wordEntry(os, 8, "patchType", leafWord(*pd, "patchType", "cyclic"));
        if (p.owner)
        {
            const std::vector<T>* jump = bc.coupledJump();
            if (!jump)
            {
                refuseWrite(field, p, "the owner side's patch object carries no jump");
            }
            listEntry(os, 8, "jump", *jump, precision);
        }
        listEntry(os, 8, "value", value, precision);
        for (const char* k : {"phi", "rho"})
        {
            const std::string w = leafWord(*pd, k, k);
            if (w != k)
            {
                wordEntry(os, 8, k, w);
            }
        }
        constantFunction1Entry<scalar>(
            os,
            field,
            p,
            *pd,
            "D",
            false,
            precision);
        constantFunction1Entry<scalar>(
            os,
            field,
            p,
            *pd,
            "I",
            false,
            precision);
        scalarEntry(os, 8, "length", length, precision);
        // a bool is written as a label (bool.C:60-66)
        keyword(os, 8, "uniformJump");
        os << uniformJump << ";\n";
    }
    // prghPermeableAlphaTotalPressureFvPatchScalarField.C:262-278 over mixedFvPatchField.C:315-323: type,
    // refValue, refGradient, valueFraction, source, value, [phi], [rho], [U], [alpha], [alphaMin], p. Its
    // constructor is mixedFvPatchField(p, iF) (.C:65) without readDict, so no patchType. The coefficients
    // are the ones the last updateSnGrad SET (.C:215-226), stored.
    else if (type == "prghPermeableAlphaTotalPressure")
    {
        if (hasCoordinateScaling(*pd, "p"))
        {
            refuseWrite(field, p, "its `p` carries coordinate scaling or a pCoeffs dictionary, not written");
        }
        const std::vector<scalar>* fraction = bc.valueFractionPtr();
        scalar alphaMin = 1;
        if (!fraction || (leaf(*pd, "alphaMin") && !leafScalar(*pd, "alphaMin", alphaMin)))
        {
            refuseWrite(field, p, "the patch object carries no valueFraction, or `alphaMin` is not a number");
        }
        // the dictionary constructor's refGrad is 0 (.C:75) until the first updateSnGrad sets it (.C:217);
        // brae allocates refGrad_ only there, so a null pointer IS that 0 -- write() never stops
        // (the fatal at .C:252-257 is updateCoeffs')
        const std::vector<T> zeros(static_cast<std::size_t>(p.size), T{});
        const std::vector<T>* grad = bc.refGradPtr();
        wordEntry(os, 8, "type", type);
        listEntry(os, 8, "refValue", bc.refValues(), precision);
        listEntry(os, 8, "refGradient", grad ? *grad : zeros, precision);
        listEntry(os, 8, "valueFraction", *fraction, precision);
        // source_ is Zero in the (p, iF) constructor (mixedFvPatchField.C:93) and never assigned
        listEntry(os, 8, "source", zeros, precision);
        listEntry(os, 8, "value", value, precision);
        for (const char* k : {"phi", "rho", "U"})
        {
            const std::string w = leafWord(*pd, k, k);
            if (w != k)
            {
                wordEntry(os, 8, k, w);
            }
        }
        const std::string alphaName = leafWord(*pd, "alpha", "none");
        if (alphaName != "none")
        {
            wordEntry(os, 8, "alpha", alphaName);
        }
        if (alphaMin != scalar(1))
        {
            scalarEntry(os, 8, "alphaMin", alphaMin, precision);
        }
        constantFunction1Entry<scalar>(
            os,
            field,
            p,
            *pd,
            "p",
            true,
            precision);
    }
    // pressurePermeableAlphaInletOutletVelocityFvPatchVectorField.C:180-190 over mixedFvPatchField.C:315-323:
    // type, [patchType] (its constructor runs readDict, .C:86), refValue, refGradient, valueFraction,
    // source, value, [phi], [rho], [alpha], [alphaMin]. refGrad and source are the constructor's Zero.
    else if (type == "permeableAlphaPressureInletOutletVelocity")
    {
        const std::vector<scalar>* fraction = bc.valueFractionPtr();
        scalar alphaMin = 1;
        if (!fraction || (leaf(*pd, "alphaMin") && !leafScalar(*pd, "alphaMin", alphaMin)))
        {
            refuseWrite(field, p, "the patch object carries no valueFraction, or `alphaMin` is not a number");
        }
        const std::vector<T> zeros(static_cast<std::size_t>(p.size), T{});
        const std::vector<T>* grad = bc.refGradPtr();
        baseEntries(
            os,
            field,
            p,
            *pd,
            type);
        listEntry(os, 8, "refValue", bc.refValues(), precision);
        listEntry(os, 8, "refGradient", grad ? *grad : zeros, precision);
        listEntry(os, 8, "valueFraction", *fraction, precision);
        listEntry(os, 8, "source", zeros, precision);
        listEntry(os, 8, "value", value, precision);
        for (const char* k : {"phi", "rho"})
        {
            const std::string w = leafWord(*pd, k, k);
            if (w != k)
            {
                wordEntry(os, 8, k, w);
            }
        }
        const std::string alphaName = leafWord(*pd, "alpha", "none");
        if (alphaName != "none")
        {
            wordEntry(os, 8, "alpha", alphaName);
        }
        if (alphaMin != scalar(1))
        {
            scalarEntry(os, 8, "alphaMin", alphaMin, precision);
        }
    }
    // waveAlphaFvPatchScalarField.C:122-129 and waveVelocityFvPatchVectorField.C:122-129: the type, then
    // waveDictName -- read from `waveDict`, default waveProperties (waveModel.C:46, :68 of each) -- then
    // the value the model's last update stored
    else if (type == "waveAlpha" || type == "waveVelocity")
    {
        wordEntry(os, 8, "type", type);
        wordEntry(os, 8, "waveDictName", leafWord(*pd, "waveDict", leafWord(*pd, "waveDictName", "waveProperties")));
        listEntry(os, 8, "value", value, precision);
    }
    // uniformFixedValueFvPatchField.C:186-194: fvPatchField's write, the PatchFunction1's writeData --
    // ConstantField's `constant v` for `constant v`, `uniform v` and a bare v alike (ConstantField.C:
    // 314-329) -- then the value, which updateCoeffs sets to that constant (.C:180)
    else if (type == "uniformFixedValue")
    {
        if (hasCoordinateScaling(*pd, "uniformValue"))
        {
            refuseWrite(field, p, "uniformValue carries coordinate scaling or a Coeffs dictionary, not written");
        }
        const std::vector<std::string>* uv = leaf(*pd, "uniformValue");
        T c{};
        if (!uv || !constantFunction1(*uv, true, c))
        {
            refuseWrite(field, p, "uniformValue is not a constant; only ConstantField's `constant v` is written");
        }
        // brae's reader keeps whichever of `uniformValue` and `value` comes last (foam_field_reader.cuh);
        // OpenFOAM's updateCoeffs sets the constant. A stored value off it is that substitution: refused.
        for (const T& v : value)
        {
            if (!sameValue(v, c))
            {
                refuseWrite(field, p, "the stored value is not uniformValue's constant, which OpenFOAM applies");
            }
        }
        wordEntry(os, 8, "type", type);
        keyword(os, 8, "uniformValue");
        os << "constant " << fmt(c, precision) << ";\n";
        listEntry(os, 8, "value", value, precision);
    }
    // rotatingWallVelocityFvPatchVectorField.C:141-148: type, origin, axis, omega (`constant v`), value
    else if (type == "rotatingWallVelocity")
    {
        vector origin{0, 0, 0};
        vector axis{0, 0, 0};
        const std::vector<std::string>* o = leaf(*pd, "origin");
        const std::vector<std::string>* a = leaf(*pd, "axis");
        if (!o || !plainVector(*o, origin) || !a || !plainVector(*a, axis))
        {
            refuseWrite(field, p, "origin or axis is not a plain `(x y z)`");
        }
        wordEntry(os, 8, "type", type);
        keyword(os, 8, "origin");
        os << fmt(origin, precision) << ";\n";
        keyword(os, 8, "axis");
        os << fmt(axis, precision) << ";\n";
        constantFunction1Entry<scalar>(
            os,
            field,
            p,
            *pd,
            "omega",
            false,
            precision);
        listEntry(os, 8, "value", value, precision);
    }
    // outletPhaseMeanVelocityFvPatchVectorField.C:161-171: fvPatchField's write (the type), Umean, alpha,
    // value -- not mixed's. Both read without a default (.C:78-79) and never changed.
    else if (type == "outletPhaseMeanVelocity")
    {
        scalar Umean = 0;
        const std::string alphaName = leafWord(*pd, "alpha", "");
        if (!leafScalar(*pd, "Umean", Umean) || alphaName.empty())
        {
            refuseWrite(field, p, "`Umean` is not a plain number or `alpha` is missing");
        }
        wordEntry(os, 8, "type", type);
        scalarEntry(os, 8, "Umean", Umean, precision);
        wordEntry(os, 8, "alpha", alphaName);
        listEntry(os, 8, "value", value, precision);
    }
    // nutkRoughWallFunctionFvPatchScalarField.C:238-246: nutWallFunction's write -- type, [U], the
    // coefficients that differ -- then its own Cs and Ks (per-face fields, .C:130-137), then the value.
    // nutkWallFunction's write is skipped, so there is NO blending entry.
    else if (type == "nutkRoughWallFunction")
    {
        const std::vector<scalar>* Ks = bc.nutkRoughKs();
        const std::vector<scalar>* Cs = bc.nutkRoughCs();
        if (!Ks || !Cs)
        {
            refuseWrite(field, p, "the patch object carries no Ks or Cs");
        }
        wordEntry(os, 8, "type", type);
        const std::string UName = leafWord(*pd, "U", "");
        if (!UName.empty())
        {
            wordEntry(os, 8, "U", UName);
        }
        wallCoefficientEntries(os, *pd, precision);
        listEntry(os, 8, "Cs", *Cs, precision);
        listEntry(os, 8, "Ks", *Ks, precision);
        listEntry(os, 8, "value", value, precision);
    }
    else
    {
        refuseWrite(field, p, "its condition `" + type + "` has no transcribed write()");
    }
    os << "    }\n";
}

} // namespace


InterWriter::InterWriter(
    const std::string& caseDir,
    const std::string& startDir,
    const std::vector<FvPatch>& patches,
    const std::string& phase1Name)
  : caseDir_(caseDir),
    startDir_(startDir),
    patches_(patches),
    phase1_(phase1Name)
{
    const FoamDict cd = readDict(caseDir + "/system/controlDict");

    // Time::writeControlNames (Time.C:63-73). clockTime and cpuTime write on elapsed seconds, which no two
    // runs share; brae does not implement them and says so rather than never writing.
    control_ = cd.wordOr("writeControl", "timeStep");
    if (control_ != "timeStep" && control_ != "runTime" && control_ != "adjustable"
     && control_ != "adjustableRunTime" && control_ != "none")
    {
        throw std::runtime_error(
            "brae interFoam: controlDict's writeControl `" + control_ + "` is not written by brae -- "
            "timeStep, runTime, adjustable(RunTime) and none are. OpenFOAM's clockTime and cpuTime write "
            "on elapsed seconds (Time.C:1132-1160).");
    }
    // TimeIO.C:284-297: writeInterval, else writeFrequency, else fatal
    const bool haveInterval = cd.find("writeInterval") || cd.find("writeFrequency");
    if (control_ != "none" && !haveInterval)
    {
        throw std::runtime_error(
            "brae interFoam: controlDict has no writeInterval (nor writeFrequency). OpenFOAM stops on it "
            "(TimeIO.C:284-297).");
    }
    interval_ = cd.find("writeInterval") ? cd.scalarOr("writeInterval", scalar(0))
                                         : cd.scalarOr("writeFrequency", scalar(0));
    if (control_ == "timeStep" && interval_ < scalar(1))
    {
        throw std::runtime_error(
            "brae interFoam: `writeControl timeStep` with writeInterval " + fmt(interval_, 6)
            + " -- OpenFOAM stops on an interval below one step (TimeIO.C:286-293).");
    }
    if ((control_ == "runTime" || control_ == "adjustable" || control_ == "adjustableRunTime")
     && !(interval_ > scalar(0)))
    {
        throw std::runtime_error("brae interFoam: writeControl `" + control_ + "` needs a positive writeInterval.");
    }
    purge_ = std::max(0, cd.intOr("purgeWrite", 0));
    precision_ = cd.intOr("writePrecision", 6);
    const std::string comp = cd.wordOr("writeCompression", "off");
    compress_ = (comp == "on" || comp == "true" || comp == "yes" || comp == "compressed");
    // brae writes ascii whatever writeFormat says; OpenFOAM reads a file's format from its own header
    // (IOobjectReadHeader.C:51), so the output is honestly labelled and readable, only larger and at
    // writePrecision rather than exact. And OpenFOAM switches compression off for binary (TimeIO.C:392-420).
    if (cd.wordOr("writeFormat", "ascii") == "binary")
    {
        noticeApproximated(
            "controlDict writeFormat binary",
            "brae writes ascii at writePrecision " + std::to_string(precision_)
                + ", labelled `format ascii;` so OpenFOAM reads it; not bit-exact as binary would be");
        compress_ = false;
    }
    timeFormat_ = cd.wordOr("timeFormat", "general");
    if (timeFormat_ != "general" && timeFormat_ != "fixed" && timeFormat_ != "scientific")
    {
        throw std::runtime_error("brae interFoam: controlDict timeFormat `" + timeFormat_
                                 + "` -- OpenFOAM knows general, fixed and scientific (IOstreamOption.H:102-110).");
    }
    timePrecision_ = cd.intOr("timePrecision", 6);
    deltaTSave_ = cd.scalarOr("deltaT", scalar(0));
    deltaT0_ = deltaTSave_;

    if (const FoamDict* fns = cd.subDict("functions"))
    {
        if (!fns->subs.empty())
        {
            std::string names;
            for (const auto& s : fns->subs)
            {
                names += (names.empty() ? "" : ", ") + s.first;
            }
            noticeIgnored(
                "controlDict functions",
                "brae runs no function objects (" + names + "): their postProcessing/ output and the fields "
                "they write are not produced; the solution does not depend on them");
        }
    }

    // A RESTART continues OpenFOAM's time index and cumulative continuity error from the start directory
    // (Time.C:304-307, initContinuityErrs.H:40-52). deltaT and deltaT0 OpenFOAM also reads there
    // (Time.C:291-302); brae's solver starts from controlDict's deltaT, and says so.
    const std::string ut = startDir + "/uniform/time";
    if (fs::exists(ut))
    {
        const FoamDict td = readDict(ut);
        startTimeIndex_ = static_cast<label>(td.scalarOr("index", scalar(0)));
        noticeIgnored(
            "uniform/time deltaT/deltaT0",
            "the restart starts from controlDict's deltaT; OpenFOAM reads the stored deltaT under "
            "adjustTimeStep and deltaT0 always (Time.C:291-302)");
    }
    const std::string startAlphaOld = startDir + "/alpha." + phase1Name + "_0";
    startHoldsAlphaOld_ = fs::exists(startAlphaOld) || fs::exists(startAlphaOld + ".gz");
    const std::string cce = startDir + "/uniform/cumulativeContErr";
    if (fs::exists(cce) || fs::exists(cce + ".gz"))
    {
        const FoamDict cd2 = readDict(fs::exists(cce) ? cce : cce + ".gz");
        cumulativeContErr_ = cd2.scalarOr("value", scalar(0));
    }
}

bool InterWriter::startHolds(const std::string& file) const
{
    const std::string path = startDir_ + "/" + file;
    return fs::exists(path) || fs::exists(path + ".gz");
}

void InterWriter::noteAlphaOldCreation(const GeometricField<scalar>& alpha1)
{
    if (oldLevelNoted_)
    {
        return;
    }
    oldLevelGrad_.assign(alpha1.boundary.size(), std::vector<scalar>());
    for (std::size_t pi = 0; pi < alpha1.boundary.size(); ++pi)
    {
        const fvPatchField<scalar>& bc = *alpha1.boundary[pi];
        if (bc.contactAngleTheta0() >= scalar(0) && bc.refGradPtr())
        {
            oldLevelGrad_[pi] = *bc.refGradPtr();
        }
    }
    oldLevelNoted_ = true;
}

void InterWriter::refuseAtFirstWrite(
    const std::string& file,
    const std::string& why)
{
    refused_.emplace_back(file, why);
    std::fprintf(stderr, "brae interFoam: %s will not be written (%s) -- the run stops at its first write "
                         "time rather than leave it out\n", file.c_str(), why.c_str());
}

bool InterWriter::isWriteTime(
    label timeIndex,
    bool runTimeIndexMoved) const
{
    if (control_ == "timeStep")
    {
        // Time.C:1111-1113, on the ABSOLUTE index -- a restart continues it
        const label n = static_cast<label>(interval_);
        return (timeIndex % n) == 0;
    }
    if (control_ == "none")
    {
        return false;
    }
    return runTimeIndexMoved;
}

void InterWriter::stepTaken(scalar deltaT)
{
    // Time::operator++ (Time.C:1059-1060): deltaT0_ = deltaTSave_; deltaTSave_ = deltaT_
    deltaT0_ = deltaTSave_;
    deltaTSave_ = deltaT;
}

void InterWriter::addContinuityError(
    scalar deltaT,
    const std::vector<scalar>& divPhi,
    const std::vector<scalar>& V)
{
    // continuityErrs.H: globalContErr = deltaT*contErr.weightedAverage(V) (DimensionedField.C
    // weightedAverage: gSum(V*f)/gSum(V) where gSum(V) > SMALL)
    scalar sumV = 0;
    scalar sumVf = 0;
    for (std::size_t c = 0; c < divPhi.size() && c < V.size(); ++c)
    {
        sumV += V[c];
        sumVf += V[c]*divPhi[c];
    }
    if (sumV > kSmall)
    {
        cumulativeContErr_ += deltaT*(sumVf/sumV);
    }
}

std::string InterWriter::timeName(
    scalar t,
    int precision) const
{
    // Time::timeName (Time.C:721-728)
    std::ostringstream buf;
    if (timeFormat_ == "fixed")
    {
        buf.setf(std::ios_base::fixed, std::ios_base::floatfield);
    }
    else if (timeFormat_ == "scientific")
    {
        buf.setf(std::ios_base::scientific, std::ios_base::floatfield);
    }
    buf.precision(precision);
    buf << t;
    return buf.str();
}

std::string InterWriter::header(
    const std::string& className,
    const std::string& location,
    const std::string& object) const
{
    return header(className, location, object, "", "");
}

// ...with the two header entries a mesh file can carry (IOobjectWriteHeader.C:164-190): `note` between
// arch and class (owner and neighbour: polyMeshInitMesh.C:96-105), `meta` after object (a ZoneMesh with
// names: ZoneMesh.C:1058-1069, a dictionary re-emitted, so its list is `N ( a b )`).
std::string InterWriter::header(
    const std::string& className,
    const std::string& location,
    const std::string& object,
    const std::string& note,
    const std::string& metaNames) const
{
    std::ostringstream os;
    os << "/*--------------------------------*- C++ -*----------------------------------*\\\n"
          "| =========                 |                                                 |\n"
          "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
          "|  \\\\    /   O peration     | Version:  2412                                  |\n"
          "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
          "|    \\\\/     M anipulation  |                                                 |\n"
          "\\*---------------------------------------------------------------------------*/\n"
          "FoamFile\n{\n"
          "    version     2.0;\n"
          "    format      ascii;\n"
          "    arch        \"LSB;label=32;scalar=64\";\n";
    if (!note.empty())
    {
        os << "    note        \"" << note << "\";\n";
    }
    os << "    class       " << className << ";\n";
    os << "    location    \"" << location << "\";\n";
    os << "    object      " << object << ";\n";
    if (!metaNames.empty())
    {
        os << "    meta\n    {\n        names           " << metaNames << ";\n    }\n";
    }
    os << "}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n";
    return os.str();
}

void InterWriter::emit(
    const std::string& path,
    const std::string& text,
    bool compressible) const
{
    pending_.push_back(PendingFile{path, text, compressible});
}

void InterWriter::writeFile(
    const std::string& path,
    const std::string& text,
    bool compressible) const
{
    // fstreamPointers.C:147-170: a compressed write goes to <file>.gz and removes the plain file, and the
    // other way round, so a directory never holds both
    const bool gz = compress_ && compressible;
    const std::string target = gz ? path + ".gz" : path;
    std::error_code ec;
    fs::remove(gz ? path : path + ".gz", ec);
    if (gz)
    {
        gzFile f = gzopen(target.c_str(), "wb");
        if (!f)
        {
            throw std::runtime_error("brae interFoam writer: cannot write " + target);
        }
        const int n = gzwrite(f, text.data(), static_cast<unsigned>(text.size()));
        gzclose(f);
        if (n != static_cast<int>(text.size()))
        {
            throw std::runtime_error("brae interFoam writer: short write to " + target);
        }
        return;
    }
    std::ofstream out(target, std::ios::binary);
    out << text;
    if (!out)
    {
        throw std::runtime_error("brae interFoam writer: cannot write " + target);
    }
}

const InterWriter::Template& InterWriter::templateFor(const std::string& fieldName)
{
    auto it = templates_.find(fieldName);
    if (it != templates_.end())
    {
        return it->second;
    }
    Template t;
    std::string path = startDir_ + "/" + fieldName;
    if (!fs::exists(path) && fs::exists(path + ".gz"))
    {
        path += ".gz";
    }
    if (fs::exists(path))
    {
        const FoamDict d = readDict(path);
        if (const FoamDict* b = d.subDict("boundaryField"))
        {
            t.boundary = *b;
        }
        const std::vector<std::string>* dims = d.find("dimensions");
        if (dims)
        {
            std::string joined;
            for (const std::string& s : *dims)
            {
                joined += (joined.empty() ? "" : " ") + s;
            }
            // `[0 1 -1 0 0 0 0]` whatever the tokenizer did with the brackets
            std::string inner;
            for (char c : joined)
            {
                if (c != '[' && c != ']')
                {
                    inner += c;
                }
            }
            std::istringstream in(inner);
            std::string tok;
            std::string out;
            while (in >> tok)
            {
                out += (out.empty() ? "" : " ") + tok;
            }
            t.dimensions = "[" + out + "]";
        }
        t.present = true;
    }
    return templates_.emplace(fieldName, std::move(t)).first->second;
}

namespace {

template <typename T>
std::string volFieldText(
    const std::string& head,
    const std::string& dimensions,
    const std::string& field,
    const std::vector<T>& internal,
    const std::vector<FvPatch>& patches,
    const FoamDict* boundary,
    const std::vector<std::unique_ptr<fvPatchField<T>>>& bcs,
    const std::vector<std::vector<T>>* derived,
    const std::vector<std::vector<T>>* stored,
    const std::vector<std::vector<T>>* storedGradient,
    int precision)
{
    std::ostringstream os;
    os << head;
    keyword(os, 0, "dimensions");
    os << dimensions << ";\n\n";
    listEntry(os, 0, "internalField", internal, precision);
    os << "\nboundaryField\n{\n";
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        const FoamDict* pd = boundary ? patchDict(*boundary, p) : nullptr;
        if (pi >= bcs.size() || !bcs[pi])
        {
            refuseWrite(field, p, "the solver holds no patch object for it");
        }
        volPatch(
            os,
            field,
            p,
            pd,
            *bcs[pi],
            derived ? &(*derived)[pi] : nullptr,
            stored ? &(*stored)[pi] : nullptr,
            (storedGradient && pi < storedGradient->size() && !(*storedGradient)[pi].empty())
                ? &(*storedGradient)[pi]
                : nullptr,
            precision);
    }
    os << "}\n\n\n// ************************************************************************* //\n";
    return os.str();
}

// A surface field's file: phi and alphaPhi0 (fluxes, `oriented`, DimensionedFieldIO.C:160-163), meshPhi
// (oriented too) and Uf (a surfaceVectorField, not oriented). Empty patches write their type alone
// (emptyFvsPatchField.C); every other patch its constraint type or `calculated` and the value
// (calculatedFvsPatchField.C, coupledFvsPatchField.C, wedgeFvsPatchField.C, symmetryPlaneFvsPatchField.C).
template <typename Field>
std::string surfaceFieldText(
    const std::string& head,
    const Field& f,
    const std::string& dimensions,
    bool oriented,
    const std::vector<FvPatch>& patches,
    int precision)
{
    std::ostringstream os;
    os << head;
    keyword(os, 0, "dimensions");
    os << dimensions << ";\n\n";
    if (oriented)
    {
        keyword(os, 0, "oriented");
        os << "oriented;\n\n";
    }
    listEntry(os, 0, "internalField", f.internal, precision);
    os << "\nboundaryField\n{\n";
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        os << "    " << p.name << "\n    {\n";
        if (p.type == "empty")
        {
            wordEntry(os, 8, "type", "empty");
        }
        else
        {
            const bool constraint = (p.type == "wedge" || p.type == "symmetryPlane" || p.type == "symmetry"
                                  || p.type == "cyclic" || p.type == "cyclicAMI" || p.type == "cyclicACMI");
            wordEntry(os, 8, "type", constraint ? p.type : std::string("calculated"));
            const std::vector<typename std::decay<decltype(f.internal[0])>::type> none;
            listEntry(os, 8, "value", pi < f.boundary.size() ? f.boundary[pi] : none, precision);
        }
        os << "    }\n";
    }
    os << "}\n\n\n// ************************************************************************* //\n";
    return os.str();
}

} // namespace

template <typename T>
std::string InterWriter::fieldText(
    const std::string& className,
    const std::string& location,
    const std::string& field,
    const std::string& templateName,
    const std::vector<T>& cells,
    const std::vector<std::unique_ptr<fvPatchField<T>>>& bcs,
    const std::vector<std::vector<T>>* derived,
    const std::vector<std::vector<T>>* stored,
    const std::vector<std::vector<T>>* storedGradient)
{
    const Template& t = templateFor(templateName);
    // a derived field takes the template's dimensions and nothing of its patches
    const FoamDict* boundary = (!derived && t.present) ? &t.boundary : nullptr;
    return volFieldText<T>(
        header(className, location, field),
        t.dimensions,
        field,
        cells,
        patches_,
        boundary,
        bcs,
        derived,
        stored,
        storedGradient,
        precision_);
}

template <typename T>
void InterWriter::probeConditions(
    const std::string& field,
    const GeometricField<T>& fld)
{
    if (!templateFor(field).present)
    {
        return;
    }
    try
    {
        fieldText<T>(
            "volField",
            "probe",
            field,
            field,
            std::vector<T>(),
            fld.boundary,
            nullptr,
            nullptr,
            nullptr);
    }
    catch (const std::runtime_error& e)
    {
        refuseAtFirstWrite(field, e.what());
    }
}

void InterWriter::write(const InterWriteState& s)
{
    if (!refused_.empty())
    {
        std::string list;
        for (const auto& r : refused_)
        {
            list += "\n  " + r.first + ": " + r.second;
        }
        throw std::runtime_error(
            "this is a write time, and OpenFOAM would write files brae does not write for "
            "this case:" + list + "\nStopping here rather than leave a time directory OpenFOAM could not "
            "restart from.");
    }
    if (!s.alpha1 || !s.U || !s.p_rgh || !s.p || !s.pBoundary || !s.phi || !s.alphaPhi0)
    {
        throw std::runtime_error("brae interFoam writer: a field the time directory needs was not handed in.");
    }

    // Time::operator++'s precision raise (Time.C:1189-1266): the name must read back as the previous time
    // plus the step, to within 10^-precision (or a tenth of the step, or SMALL)
    std::string name = timeName(s.time, timePrecision_);
    {
        const scalar oldTimeValue = s.time - s.deltaT;
        const scalar userDeltaT = s.time - (s.time - s.deltaT);
        const scalar timeTol =
            std::max(std::min(std::pow(scalar(10), -timePrecision_), scalar(0.1)*userDeltaT), kSmall);
        auto off = [&](const std::string& n)
        {
            const scalar v = static_cast<scalar>(std::strtod(n.c_str(), nullptr));
            return std::fabs(v - oldTimeValue - userDeltaT) > timeTol;
        };
        const int oldPrecision = timePrecision_;
        while (timePrecision_ < kMaxTimePrecision && off(name))
        {
            ++timePrecision_;
            name = timeName(s.time, timePrecision_);
        }
        if (timePrecision_ != oldPrecision)
        {
            std::fprintf(stderr, "brae interFoam: increased the timePrecision from %d to %d to distinguish "
                                 "between timeNames at time %s\n", oldPrecision, timePrecision_, name.c_str());
        }
    }

    const std::string dir = caseDir_ + "/" + name;
    pending_.clear();

    // uniform/time first, always ascii and never compressed (TimeIO.C:508-539)
    {
        std::ostringstream os;
        os << header("dictionary", name + "/uniform", "time");
        keyword(os, 0, "value");
        os << timeName(s.time, kMaxTimePrecision) << ";\n\n";
        keyword(os, 0, "name");
        os << "\"" << name << "\";\n\n";
        keyword(os, 0, "index");
        os << s.timeIndex << ";\n\n";
        scalarEntry(os, 0, "deltaT", s.deltaT, precision_);
        os << "\n";
        scalarEntry(os, 0, "deltaT0", deltaT0_, precision_);
        os << "\n\n// ************************************************************************* //\n";
        emit(dir + "/uniform/time", os.str(), false);
    }

    const std::string alphaName = "alpha." + phase1_;
    emit(
        dir + "/" + alphaName,
        fieldText<scalar>(
            "volScalarField",
            name,
            alphaName,
            alphaName,
            s.alpha1Cells ? *s.alpha1Cells : s.alpha1->internal,
            s.alpha1->boundary,
            nullptr,
            nullptr,
            nullptr),
        true);
    if (alphaOld_)
    {
        if (!s.alpha1OldCells || !s.alpha1OldBoundary || !s.alpha1SubCycleBoundary)
        {
            throw std::runtime_error(
                "brae interFoam writer: alpha is sub-cycled and the loop handed no old-level state for "
                + alphaName + "_0");
        }
        if (!oldLevelNoted_)
        {
            throw std::runtime_error(
                "brae interFoam writer: " + alphaName + "_0's creation state (the contact-angle gradient) was "
                "never recorded");
        }
        // The old level's patches after the step: the sub-cycle's storeOldTime force-copies alpha into
        // it at every sub-cycle, patches included (GeometricField.C:932), and ~subCycleField restores it
        // with `gf0_ = gf_0_` (subCycle.H:89-99) -- each patch's OWN operator=, which is
        //   a copy of the step's start on an assigning patch (fvPatchField.C:407-413),
        //   a NO-OP on the fixedValue and mixed families (fixedValueFvPatchField.H:202-204,
        //     mixedFvPatchField.H:303-305), which so keep the last sub-cycle's copy -- alpha as that
        //     sub-cycle began, MEASURED on laminar/waves/stokesI's waveAlpha inlet,
        //   inletOutlet's re-blend vf*refValue + (1 - vf)*start (inletOutletFvPatchField.C:143-152) with
        //     the old level's OWN valueFraction -- the dictionary constructor's 0 (:80), since nothing
        //     evaluates alpha's inletOutlet patches before the level is cloned (the first alpha1.oldTime(),
        //     subCycle.H:78) and every later store is values only (GeometricField.C:932). So it is a plain
        //     copy of the start, as on an assigning patch. (This writer first blended with the valueFraction
        //     brae pushes from the initial flux, which differs on any start with inflow there.)
        const std::vector<std::vector<scalar>>& start = *s.alpha1OldBoundary;
        // BRAE_CONTROL_ALPHA_OLD_START=1 takes the step's start on every patch -- this writer's first
        // form, 5.3e-03 off OpenFOAM on RAS/weirOverflow's variableHeightFlowRate inlet. The control
        // tests/interfoam_write_vs_openfoam.sh must go red on; never the default.
        const char* startOnly = std::getenv("BRAE_CONTROL_ALPHA_OLD_START");
        const bool controlStartOnly = startOnly && std::string(startOnly) == "1";
        const std::vector<std::vector<scalar>>& lastSub =
            controlStartOnly ? start : *s.alpha1SubCycleBoundary;
        std::vector<std::vector<scalar>> old(s.alpha1->boundary.size());
        for (std::size_t pi = 0; pi < old.size(); ++pi)
        {
            const fvPatchField<scalar>& bc = *s.alpha1->boundary[pi];
            if (!bc.coupled() && bc.bcCategory() == 3)
            {
                old[pi] = start[pi];
            }
            else if (!bc.coupled() && !bc.ofAssignmentWritesValue())
            {
                old[pi] = lastSub[pi];
            }
            else
            {
                old[pi] = start[pi];
            }
        }
        emit(
            dir + "/" + alphaName + "_0",
            fieldText<scalar>(
                "volScalarField",
                name,
                alphaName + "_0",
                alphaName,
                *s.alpha1OldCells,
                s.alpha1->boundary,
                nullptr,
                &old,
                &oldLevelGrad_),
            true);
    }
    emit(
        dir + "/U",
        fieldText<vector>(
            "volVectorField",
            name,
            "U",
            "U",
            s.UCells ? *s.UCells : s.U->internal,
            s.U->boundary,
            nullptr,
            nullptr,
            nullptr),
        true);
    emit(
        dir + "/p_rgh",
        fieldText<scalar>(
            "volScalarField",
            name,
            "p_rgh",
            "p_rgh",
            s.p_rghCells ? *s.p_rghCells : s.p_rgh->internal,
            s.p_rgh->boundary,
            nullptr,
            nullptr,
            nullptr),
        true);
    // p: NO_READ, AUTO_WRITE (createFields.H:91-102), `calculated` on every non-constraint patch, with
    // p_rgh's dimensions -- p_rgh's template, whose patch entries the derived values override
    // ...and on a coupled patch that writes a value, `p == p_rgh + rho*gh` (pEqn.H) is the sum's own cells
    // evaluated, as every field expression is (coupledLocalValue)
    std::vector<std::vector<scalar>> pPatchValues = *s.pBoundary;
    for (std::size_t pi = 0; pi < patches_.size() && pi < pPatchValues.size(); ++pi)
    {
        const FvPatch& q = patches_[pi];
        if (q.coupled && (q.type == "cyclicAMI" || q.type == "cyclicACMI")
            && std::getenv("BRAE_CONTROL_COUPLED_OPERAND_VALUES") == nullptr)
        {
            pPatchValues[pi] = coupledLocalValue(q, *s.p);
        }
    }
    emit(
        dir + "/p",
        fieldText<scalar>(
            "volScalarField",
            name,
            "p",
            "p_rgh",
            *s.p,
            s.p_rgh->boundary,
            &pPatchValues,
            nullptr,
            nullptr),
        true);
    emit(
        dir + "/phi",
        surfaceFieldText(
            header("surfaceScalarField", name, "phi"),
            *s.phi,
            "[0 3 -1 0 0 0 0]",
            true,
            patches_,
            precision_),
        true);
    const std::string aPhiName = "alphaPhi0." + phase1_;
    emit(
        dir + "/" + aPhiName,
        surfaceFieldText(
            header("surfaceScalarField", name, aPhiName),
            *s.alphaPhi0,
            "[0 3 -1 0 0 0 0]",
            true,
            patches_,
            precision_),
        true);

    if (s.turbulence && s.turbulence->on)
    {
        const InterTurbulence& tb = *s.turbulence;
        std::vector<std::pair<std::string, const GeometricField<scalar>*>> tf;
        tf.emplace_back("k", &tb.k);
        if (tb.model == InterRasModel::KEpsilon)
        {
            tf.emplace_back("epsilon", &tb.epsilon);
        }
        else if (tb.model == InterRasModel::KOmegaSST)
        {
            tf.emplace_back("omega", &tb.omega);
        }
        tf.emplace_back("nut", &tb.nut);
        for (const auto& e : tf)
        {
            emit(
                dir + "/" + e.first,
                fieldText<scalar>(
                    "volScalarField",
                    name,
                    e.first,
                    e.first,
                    e.second->internal,
                    e.second->boundary,
                    nullptr,
                    nullptr,
                    nullptr),
                true);
        }
    }

    // A solidBody-moved mesh. polyMesh/points: polyMesh::movePoints makes them AUTO_WRITE at the current
    // instance (polyMesh.C:1215, :1232-1233), a pointIOField of class vectorField, the list alone. meshPhi:
    // created at the first move (fvMesh.C:951-970) and written whenever it exists (fvMesh.C:1103-1105),
    // the swept volumes of the step's move over deltaT. Uf: createUfIfPresent.H's AUTO_WRITE face velocity,
    // correctUf's result after the last pressure corrector (pEqn.H:67).
    if (meshMotion_)
    {
        if (!s.Uf || !s.meshPhi || !s.points)
        {
            throw std::runtime_error("brae interFoam writer: a moving mesh handed no Uf, meshPhi or points");
        }
        emit(
            dir + "/Uf",
            surfaceFieldText(
                header("surfaceVectorField", name, "Uf"),
                *s.Uf,
                "[0 1 -1 0 0 0 0]",
                false,
                patches_,
                precision_),
            true);
        emit(
            dir + "/meshPhi",
            surfaceFieldText(
                header("surfaceScalarField", name, "meshPhi"),
                *s.meshPhi,
                "[0 3 -1 0 0 0 0]",
                true,
                patches_,
                precision_),
            true);
        std::ostringstream os;
        os << header("vectorField", name + "/polyMesh", "points") << "\n";
        os << s.points->size() << "\n(\n";
        for (const vector& x : *s.points)
        {
            os << fmt(x, precision_) << "\n";
        }
        os << ")\n\n\n// ************************************************************************* //\n";
        emit(dir + "/polyMesh/points", os.str(), true);
    }

    // A displacementLaplacian motion's own fields. pointDisplacement: MUST_READ/AUTO_WRITE
    // (displacementMotionSolver.C:50-61), the total displacement from points0 after curPoints; its patches
    // write as pointPatchFields do -- fixedValue its value, zeroGradient and empty their type, waveMaker
    // every member and the value (waveMakerPointPatchVectorField.C:449-464). cellDisplacement:
    // READ_IF_PRESENT/AUTO_WRITE (displacementLaplacianFvMotionSolver.C:71-84), this step's Laplace
    // solution; a value-fixing point patch makes its patch cellMotion (cellMotionBoundaryTypes), type and
    // value (cellMotionFvPatchField.C:126-131), the others keep their type.
    if (displacement_)
    {
        const DisplacementLaplacianFvMotionSolver* dm = s.displacement;
        if (!dm)
        {
            throw std::runtime_error("brae interFoam writer: a displacementLaplacian mesh handed no motion solver");
        }
        const auto& pps = dm->pointPatches();
        if (pps.size() != patches_.size())
        {
            throw std::runtime_error("brae interFoam writer: the point patches do not follow the mesh's patches");
        }
        using PT = DisplacementLaplacianFvMotionSolver::PointPatchType;
        // MUST_READ, so the dimensions are the start file's (displacementMotionSolver.C:50-61)
        const Template& pdt = templateFor("pointDisplacement");
        if (!pdt.present)
        {
            throw std::runtime_error("brae interFoam writer: the start time's pointDisplacement was not read");
        }
        std::ostringstream os;
        os << header("pointVectorField", name, "pointDisplacement");
        keyword(os, 0, "dimensions");
        os << pdt.dimensions << ";\n\n";
        listEntry(os, 0, "internalField", dm->pointDisplacement(), precision_);
        os << "\nboundaryField\n{\n";
        for (std::size_t pi = 0; pi < pps.size(); ++pi)
        {
            const auto& pp = pps[pi];
            if (pp.name != patches_[pi].name)
            {
                refuseWrite("pointDisplacement", patches_[pi], "the point patch order is not the mesh's");
            }
            if (pp.dict && (leaf(*pp.dict, "patchType") || setsUseImplicit(*pp.dict)))
            {
                refuseWrite("pointDisplacement", patches_[pi], "it sets patchType or useImplicit, not written");
            }
            os << "    " << pp.name << "\n    {\n";
            if (pp.type == PT::fixedValue)
            {
                wordEntry(os, 8, "type", "fixedValue");
                listEntry(os, 8, "value", pp.value, precision_);
            }
            else if (pp.type == PT::zeroGradient || pp.type == PT::empty)
            {
                wordEntry(os, 8, "type", pp.type == PT::empty ? "empty" : "zeroGradient");
            }
            else
            {
                const WaveMakerPointPatchVectorField* w = pp.waveMaker.get();
                scalar wavePhase = 0;
                if (!w || !pp.dict || !leafScalar(*pp.dict, "wavePhase", wavePhase))
                {
                    refuseWrite("pointDisplacement", patches_[pi], "a waveMaker without its model or wavePhase");
                }
                wordEntry(os, 8, "type", "waveMaker");
                wordEntry(os, 8, "motionType", w->motionTypeName());
                keyword(os, 8, "n");
                os << fmt(w->n(), precision_) << ";\n";
                scalarEntry(os, 8, "initialDepth", w->initialDepth(), precision_);
                scalarEntry(os, 8, "wavePeriod", w->wavePeriod(), precision_);
                scalarEntry(os, 8, "waveHeight", w->waveHeight(), precision_);
                scalarEntry(os, 8, "wavePhase", wavePhase, precision_);
                scalarEntry(os, 8, "waveAngle", w->waveAngle(), precision_);
                scalarEntry(os, 8, "startTime", w->startTime(), precision_);
                scalarEntry(os, 8, "rampTime", w->rampTime(), precision_);
                // a bool is written as a label (bool.C:60-66)
                keyword(os, 8, "secondOrder");
                os << (w->secondOrder() ? 1 : 0) << ";\n";
                keyword(os, 8, "nPaddle");
                os << w->nPaddle() << ";\n";
                listEntry(os, 8, "value", pp.value, precision_);
            }
            os << "    }\n";
        }
        os << "}\n\n\n// ************************************************************************* //\n";
        emit(dir + "/pointDisplacement", os.str(), true);

        std::ostringstream oc;
        oc << header("volVectorField", name, "cellDisplacement");
        keyword(oc, 0, "dimensions");
        oc << "[0 1 0 0 0 0 0];\n\n";
        listEntry(oc, 0, "internalField", dm->cellDisplacement(), precision_);
        oc << "\nboundaryField\n{\n";
        const auto& cb = dm->cellDisplacementBoundary();
        for (std::size_t pi = 0; pi < pps.size(); ++pi)
        {
            const auto& pp = pps[pi];
            oc << "    " << pp.name << "\n    {\n";
            if (pp.fixesValue())
            {
                if (pi >= cb.size())
                {
                    refuseWrite("cellDisplacement", patches_[pi], "no cellMotion value for this patch");
                }
                wordEntry(oc, 8, "type", "cellMotion");
                listEntry(oc, 8, "value", cb[pi], precision_);
            }
            else
            {
                wordEntry(oc, 8, "type", pp.type == PT::empty ? "empty" : "zeroGradient");
            }
            oc << "    }\n";
        }
        oc << "}\n\n\n// ************************************************************************* //\n";
        emit(dir + "/cellDisplacement", oc.str(), true);
    }

    // A rigidBodyMotion's own state. pointDisplacement: the same MUST_READ/AUTO_WRITE field
    // (displacementMotionSolver.C:50-61), transformPoints(weight, points0) - points0 after
    // constrainDisplacement (rigidBodyMeshMotion.C:360-389). Its patches write as their pointPatchFields do:
    // fixedValue its stored value, which constrainDisplacement never changes (pointConstraints.C:395-429);
    // calculated and symmetryPlane their type alone (calculatedPointPatchField.H:50-59,
    // basicSymmetryPointPatchField.H:53-55). uniform/rigidBodyMotionState: rigidBodyMeshMotion::writeObject
    // forces ASCII and writes model_.state() -- motionState_, the step's solved state, never motionState0_
    // -- into an IOdictionary (rigidBodyMeshMotion.C:392-417): q, qDot, qDdot, t, deltaT
    // (rigidBodyModelStateIO.C:33-40), compressed as the run's writeCompression says (regIOobjectWrite.C:
    // 134-140). BRAE_CONTROL_RBSTATE_OLD=1 writes the state the step started from, for the gate's control.
    if (rigidBody_)
    {
        const RigidBodyMeshMotion* rb = s.rigidBody;
        if (!rb)
        {
            throw std::runtime_error("brae interFoam writer: a rigidBodyMotion mesh handed no motion solver");
        }
        const auto& pcs = rb->pointConstraints().patchConstraints();
        if (pcs.size() != patches_.size())
        {
            throw std::runtime_error("brae interFoam writer: the point patches do not follow the mesh's patches");
        }
        using Kind = PointPatchConstraint::Kind;
        // MUST_READ, so the dimensions are the start file's (displacementMotionSolver.C:50-61)
        const Template& pdt = templateFor("pointDisplacement");
        if (!pdt.present)
        {
            throw std::runtime_error("brae interFoam writer: the start time's pointDisplacement was not read");
        }
        std::ostringstream os;
        os << header("pointVectorField", name, "pointDisplacement");
        keyword(os, 0, "dimensions");
        os << pdt.dimensions << ";\n\n";
        listEntry(os, 0, "internalField", rb->pointDisplacement(), precision_);
        os << "\nboundaryField\n{\n";
        for (std::size_t pi = 0; pi < pcs.size(); ++pi)
        {
            const PointPatchConstraint& c = pcs[pi];
            if (c.name != patches_[pi].name)
            {
                refuseWrite("pointDisplacement", patches_[pi], "the point patch order is not the mesh's");
            }
            os << "    " << c.name << "\n    {\n";
            if (c.kind == Kind::fixedValue)
            {
                wordEntry(os, 8, "type", "fixedValue");
                // the stored Field, one value per patch point: `uniform v`, or an empty list when the
                // patch has no points (Field.C:727-748)
                listEntry(os, 8, "value", std::vector<vector>(c.meshPoints.size(), c.value), precision_);
            }
            else
            {
                wordEntry(os, 8, "type", c.kind == Kind::symmetryPlane ? "symmetryPlane" : "calculated");
            }
            os << "    }\n";
        }
        os << "}\n\n\n// ************************************************************************* //\n";
        emit(dir + "/pointDisplacement", os.str(), true);

        const RBD::ModelState& st =
            std::getenv("BRAE_CONTROL_RBSTATE_OLD") != nullptr ? rb->state0() : rb->state();
        std::ostringstream ob;
        ob << header("dictionary", name + "/uniform", "rigidBodyMotionState");
        for (const auto& entry : {std::make_pair("q", &st.q),
                                  std::make_pair("qDot", &st.qDot),
                                  std::make_pair("qDdot", &st.qDdot)})
        {
            keyword(ob, 0, entry.first);
            ob << dictScalarList(*entry.second, precision_) << ";\n\n";
        }
        // t and deltaT at writePrecision, as the tokens they are -- uniform/time writes the same double at
        // the maximum precision (TimeIO.C:528), this file does not
        keyword(ob, 0, "t");
        ob << dictToken(fmt(st.t, precision_), precision_) << ";\n\n";
        keyword(ob, 0, "deltaT");
        ob << dictToken(fmt(st.deltaT, precision_), precision_) << ";\n\n";
        ob << "\n// ************************************************************************* //\n";
        emit(dir + "/uniform/rigidBodyMotionState", ob.str(), true);
    }

    // A refining mesh (dynamicRefineFvMesh, hexRef8). hexRef8's own files at EVERY write -- writeObject
    // forces them to this instance and calls hexRef8::write whatever the mesh did (dynamicRefineFvMesh.C:
    // 1478-1491, hexRef8.C:5808-5825): cellLevel and pointLevel (labelList), level0Edge
    // (uniformDimensionedScalarField) and, while the history is active, refinementHistory -- which
    // operator<< COMPACTS first (refinementHistory.C), done by the driver on the live history before this
    // call. dumpLevel's volScalarField cellLevel beside them, every patch `calculated; value uniform 0`
    // (dynamicRefineFvMesh.C:1494-1519). From the first topology change on, the mesh itself
    // (polyMesh::updateMesh -> setInstance, AUTO_WRITE ever after): faces as a plain faceList in ascii
    // (CompactIOList.C:165-200), owner and neighbour with their `note`, boundary from the live patches, the
    // three zone lists, points, and a moving mesh's points0 (points0MotionSolver.C:152-218). Uf for a mesh
    // that only refines (createUfIfPresent.H: a dynamic mesh has one); a moving one writes it with meshPhi.
    if (refine_)
    {
        const InterAmr* amr = s.amr;
        const PrimitiveMesh* m = s.mesh;
        if (!amr || !m)
        {
            throw std::runtime_error("brae interFoam writer: a refining mesh handed no mesh or refinement state");
        }
        const std::string pm = name + "/polyMesh";
        const std::string end = "\n\n// ************************************************************************* //\n";
        const cpu::hexRef8::Levels& lv = amr->state.levels;
        const cpu::hexRef8::History& hist = amr->state.history;
        emit(dir + "/polyMesh/cellLevel", header("labelList", pm, "cellLevel") + labelListText(lv.cellLevel) + end,
             true);
        emit(dir + "/polyMesh/pointLevel", header("labelList", pm, "pointLevel") + labelListText(lv.pointLevel)
             + end, true);
        {
            std::ostringstream os;
            os << header("uniformDimensionedScalarField", pm, "level0Edge");
            keyword(os, 0, "dimensions");
            os << "[0 1 0 0 0 0 0];\n";
            keyword(os, 0, "value");
            os << fmt(amr->level0Edge, precision_) << ";\n\n" << end;
            emit(dir + "/polyMesh/level0Edge", os.str(), true);
        }
        if (hist.active)
        {
            std::vector<std::string> split;
            split.reserve(hist.parent.size());
            for (std::size_t i = 0; i < hist.parent.size(); ++i)
            {
                if (hist.parent[i] < -1)
                {
                    throw std::runtime_error("brae interFoam writer: the refinement history holds a freed entry; "
                                             "it must be compacted before it is written");
                }
                split.push_back(std::to_string(hist.parent[i]) + " " + labelListText(hist.addedCells[i]));
            }
            emit(dir + "/polyMesh/refinementHistory",
                 header("refinementHistory", pm, "refinementHistory") + "// splitCells\n" + compoundListText(split)
                 + "\n// visibleCells\n" + labelListText(hist.visibleCells) + end,
                 true);
        }
        if (amr->controls.dumpLevel)
        {
            std::ostringstream os;
            os << header("volScalarField", name, "cellLevel");
            keyword(os, 0, "dimensions");
            os << "[0 0 0 0 0 0 0];\n\n";
            listEntry(os, 0, "internalField", std::vector<scalar>(lv.cellLevel.begin(), lv.cellLevel.end()),
                      precision_);
            os << "\nboundaryField\n{\n";
            for (const FvPatch& p : patches_)
            {
                os << "    " << p.name << "\n    {\n";
                wordEntry(os, 8, "type", "calculated");
                listEntry(os, 8, "value", std::vector<scalar>(static_cast<std::size_t>(p.size), scalar(0)),
                          precision_);
                os << "    }\n";
            }
            os << "}\n" << end;
            emit(dir + "/cellLevel", os.str(), true);
        }
        if (amr->topoChanged)
        {
            std::vector<std::string> faces;
            faces.reserve(static_cast<std::size_t>(m->nFaces()));
            for (label fi = 0; fi < m->nFaces(); ++fi)
            {
                std::vector<label> verts(static_cast<std::size_t>(m->faceSize(fi)));
                for (label k = 0; k < m->faceSize(fi); ++k)
                {
                    verts[static_cast<std::size_t>(k)] = m->faceVert(fi, k);
                }
                faces.push_back(labelListText(verts));
            }
            emit(dir + "/polyMesh/faces", header("faceList", pm, "faces") + compoundListText(faces) + end, true);
            std::ostringstream note;
            note << "nPoints:" << m->nPoints() << "  nCells:" << m->nCells() << "  nFaces:" << m->nFaces()
                 << "  nInternalFaces:" << m->nInternalFaces();
            emit(dir + "/polyMesh/owner", header("labelList", pm, "owner", note.str(), "")
                 + labelListText(m->owner()) + end, true);
            emit(dir + "/polyMesh/neighbour", header("labelList", pm, "neighbour", note.str(), "")
                 + labelListText(m->neighbour()) + end, true);
            // polyBoundaryMesh::writeObject forces UNCOMPRESSED (polyBoundaryMesh.C)
            {
                std::ostringstream os;
                os << header("polyBoundaryMesh", pm, "boundary") << patches_.size() << "\n(\n";
                for (const FvPatch& p : patches_)
                {
                    os << "    " << p.name << "\n    {\n";
                    wordEntry(os, 8, "type", p.type);
                    // the read groups, and a wall's own type added to them (wallPolyPatch.C:57, addGroup)
                    std::vector<std::string> groups = p.inGroups;
                    if (p.type == "wall" && std::find(groups.begin(), groups.end(), "wall") == groups.end())
                    {
                        groups.push_back("wall");
                    }
                    if (!groups.empty())
                    {
                        // writeList(os, 0): flat, `N(a b)` (patchIdentifier.C:139-144)
                        std::string g = std::to_string(groups.size()) + "(";
                        for (std::size_t i = 0; i < groups.size(); ++i)
                        {
                            g += (i ? " " : "") + groups[i];
                        }
                        wordEntry(os, 8, "inGroups", g + ")");
                    }
                    wordEntry(os, 8, "nFaces", std::to_string(p.size));
                    wordEntry(os, 8, "startFace", std::to_string(p.start));
                    os << "    }\n";
                }
                os << ")" << end;
                emit(dir + "/polyMesh/boundary", os.str(), false);
            }
            // the zone lists (ZoneMesh.C:1155-1180): `0()` when empty; motorBike's empty pointZone as
            // N ( name { type; pointLabels List<label> 0(); } ), its names in the header's meta
            {
                const std::vector<std::pair<std::string, const std::vector<ZoneEntry>*>> kinds{
                    {"cellZones", &amr->cellZoneEntries},
                    {"faceZones", &amr->faceZoneEntries},
                    {"pointZones", &amr->pointZoneEntries}};
                for (const auto& kind : kinds)
                {
                    const std::vector<ZoneEntry>& zones = *kind.second;
                    if (zones.empty())
                    {
                        emit(dir + "/polyMesh/" + kind.first, header("regIOobject", pm, kind.first) + "0()" + end,
                             true);
                        continue;
                    }
                    std::string names = std::to_string(zones.size()) + " (";
                    std::ostringstream os;
                    os << zones.size() << "\n(";
                    for (const ZoneEntry& z : zones)
                    {
                        names += " " + z.name;
                        os << z.name << "\n{\n";
                        wordEntry(os, 4, "type", z.type);
                        wordEntry(os, 4, "pointLabels", "List<label> 0()");
                        os << "}\n";
                    }
                    os << ")";
                    emit(dir + "/polyMesh/" + kind.first,
                         header("regIOobject", pm, kind.first, "", names + " )") + os.str() + end, true);
                }
            }
            if (!refineMoves_)
            {
                std::ostringstream os;
                os << header("vectorField", pm, "points") << "\n";
                os << m->nPoints() << "\n(\n";
                for (const vector& x : m->points())
                {
                    os << fmt(x, precision_) << "\n";
                }
                os << ")\n" << end;
                emit(dir + "/polyMesh/points", os.str(), true);
            }
            else
            {
                if (!s.points0)
                {
                    throw std::runtime_error("brae interFoam writer: a moving refining mesh handed no points0");
                }
                std::ostringstream os;
                os << header("vectorField", pm, "points0") << "\n";
                os << s.points0->size() << "\n(\n";
                for (const vector& x : *s.points0)
                {
                    os << fmt(x, precision_) << "\n";
                }
                os << ")\n" << end;
                emit(dir + "/polyMesh/points0", os.str(), true);
            }
        }
        if (!refineMoves_)
        {
            if (!s.Uf)
            {
                throw std::runtime_error("brae interFoam writer: a refining mesh handed no Uf");
            }
            emit(
                dir + "/Uf",
                surfaceFieldText(
                    header("surfaceVectorField", name, "Uf"),
                    *s.Uf,
                    "[0 1 -1 0 0 0 0]",
                    false,
                    patches_,
                    precision_),
                true);
        }
    }

    // rAU (initCorrectPhi.H:3-17): `rAU.ref() = 1.0/UEqn.A()` assigns the whole field (pEqn.H:4), and A()
    // is extrapolatedCalculated (fvMatrix.C:1314-1328), so every non-coupled patch holds its face cells'
    // values -- measured bit-exact on six cases. Constraint patches write their type alone.
    if (rAU_)
    {
        if (!s.rAU || s.rAU->empty())
        {
            throw std::runtime_error("brae interFoam writer: correctPhi handed no rAU");
        }
        const std::vector<scalar>& r = *s.rAU;
        std::ostringstream os;
        os << header("volScalarField", name, "rAU");
        keyword(os, 0, "dimensions");
        os << "[-1 3 1 0 0 0 0];\n\n";
        listEntry(os, 0, "internalField", r, precision_);
        os << "\nboundaryField\n{\n";
        for (const FvPatch& p : patches_)
        {
            os << "    " << p.name << "\n    {\n";
            // cyclic writes its type alone (cyclicFvPatchField.C:244-247 overrides coupled's value entry)
            if (p.type == "empty" || p.type == "wedge" || p.type == "symmetryPlane" || p.type == "symmetry"
                || p.type == "cyclic")
            {
                wordEntry(os, 8, "type", p.type);
            }
            else if (p.type == "cyclicAMI" && p.coupled)
            {
                // rAU.ref() = 1.0/UEqn.A() (pEqn.H:4) assigns the expression's patch values too, and on a
                // coupled patch those are the result's own cells evaluated (coupledLocalValue)
                // BRAE_CONTROL_COUPLED_OPERAND_VALUES=1 writes the operands' arithmetic instead -- the
                // reciprocal of A's patch value here, p_rgh_b + rho_b*gh_b for p -- for the write gate's control
                std::vector<scalar> v = coupledLocalValue(p, r);
                if (std::getenv("BRAE_CONTROL_COUPLED_OPERAND_VALUES") != nullptr)
                {
                    std::vector<scalar> aCells(r.size());
                    for (std::size_t c = 0; c < r.size(); ++c)
                    {
                        aCells[c] = scalar(1)/r[c];
                    }
                    v = coupledLocalValue(p, aCells);
                    for (scalar& x : v)
                    {
                        x = scalar(1)/x;
                    }
                }
                wordEntry(os, 8, "type", p.type);
                listEntry(os, 8, "value", v, precision_);
            }
            else if (isConstraintType(p.type))
            {
                refuseWrite("rAU", p, "a coupled patch's rAU is the coupled 1/A, which brae does not keep");
            }
            else
            {
                std::vector<scalar> v(p.faceCells.size());
                for (std::size_t i = 0; i < v.size(); ++i)
                {
                    v[i] = r[static_cast<std::size_t>(p.faceCells[i])];
                }
                wordEntry(os, 8, "type", "calculated");
                listEntry(os, 8, "value", v, precision_);
            }
            os << "    }\n";
        }
        os << "}\n\n\n// ************************************************************************* //\n";
        emit(dir + "/rAU", os.str(), true);
    }

    // rDeltaT (createRDeltaT.H): AUTO_WRITE, 1/s, built on extrapolatedCalculated -- which the mesh's
    // constraint patches replace with their own type (fvPatchFieldNew.C:57-64). extrapolatedCalculated
    // writes calculated's type and value (calculatedFvPatchField.C:212-216), the face cells' values its
    // evaluate extrapolates (extrapolatedCalculatedFvPatchField.C:81-92) once setRDeltaT's
    // correctBoundaryConditions has run; a constraint patch writes as it does for any field.
    if (rDeltaT_)
    {
        if (!s.rDeltaT || s.rDeltaT->empty())
        {
            throw std::runtime_error("brae interFoam writer: local time stepping handed no rDeltaT");
        }
        const std::vector<scalar>& r = *s.rDeltaT;
        std::ostringstream os;
        os << header("volScalarField", name, "rDeltaT");
        keyword(os, 0, "dimensions");
        os << "[0 0 -1 0 0 0 0];\n\n";
        listEntry(os, 0, "internalField", r, precision_);
        os << "\nboundaryField\n{\n";
        for (const FvPatch& p : patches_)
        {
            os << "    " << p.name << "\n    {\n";
            if (p.type == "empty" || p.type == "wedge" || p.type == "symmetryPlane" || p.type == "symmetry"
                || p.type == "cyclic")
            {
                wordEntry(os, 8, "type", p.type);
            }
            else if (isConstraintType(p.type))
            {
                refuseWrite("rDeltaT", p, "its constraint type is not transcribed for rDeltaT");
            }
            else
            {
                std::vector<scalar> v(p.faceCells.size());
                for (std::size_t i = 0; i < v.size(); ++i)
                {
                    v[i] = r[static_cast<std::size_t>(p.faceCells[i])];
                }
                wordEntry(os, 8, "type", "extrapolatedCalculated");
                listEntry(os, 8, "value", v, precision_);
            }
            os << "    }\n";
        }
        os << "}\n\n\n// ************************************************************************* //\n";
        emit(dir + "/rDeltaT", os.str(), true);
    }

    // uniform/waveProperties.<patch>: each wave model is an AUTO_WRITE IOdictionary (waveModel.C:252-263),
    // written by regIOobject with the run's format and compression (regIOobjectWrite.C:134-137); its header
    // class is the model's type (IOobjectWriteHeader.C:280-283). The body is the dictionary as the model
    // holds it -- the case's sub-dictionary merged over a restart's stored file -- then the computed
    // waterDepthRef when neither named it (waveModel.C:322-343). Only models that exist are written: one is
    // created at the first update that looks it up (waveModelNew.C:85-104). The models only READ their
    // lists (irregularMultiDirectionalWaveModel.C:268-271, streamFunctionWaveModel.C:231-232), so a list
    // entry is written as primitiveEntry writes any entry: its tokens, each re-emitted, joined by single
    // spaces on one line (primitiveEntryIO.C:280-314) -- `57 ( ( 15.367000000000001 ... ) ... )`.
    for (std::size_t pi = 0; waves_ && pi < waves_->model.size(); ++pi)
    {
        const waveModels::WaveModel* wm = waves_->model[pi].get();
        if (!wm)
        {
            continue;
        }
        const FoamDict& d = wm->dict();
        const std::string object = "waveProperties." + wm->patchName();
        if (!d.subs.empty())
        {
            throw std::runtime_error(
                "brae interFoam writer: " + object + " holds a sub-dictionary, which is not echoed");
        }
        std::ostringstream os;
        os << header(wm->type(), name + "/uniform", object);
        bool hasDepth = false;
        for (const auto& leafEntry : d.leaves)
        {
            // an entry with no tokens (`key;`) is a legal primitiveEntry, written as the keyword and `;`
            // (primitiveEntryIO.C:126-131, 280-307)
            hasDepth = hasDepth || leafEntry.first == "waterDepthRef";
            std::string value;
            for (const std::string& token : leafEntry.second)
            {
                value += (value.empty() ? "" : " ") + dictToken(token, precision_);
            }
            wordEntry(os, 0, leafEntry.first, value);
            os << "\n";
        }
        if (!hasDepth)
        {
            scalarEntry(os, 0, "waterDepthRef", wm->waterDepthRef(), precision_);
            os << "\n";
        }
        os << "\n// ************************************************************************* //\n";
        emit(dir + "/uniform/" + object, os.str(), true);
    }

    // uniform/cumulativeContErr (initContinuityErrs.H:40-52), and the function objects' state file,
    // empty because brae runs none
    {
        std::ostringstream os;
        os << header("uniformDimensionedScalarField", name + "/uniform", "cumulativeContErr");
        keyword(os, 0, "dimensions");
        os << "[0 0 0 0 0 0 0];\n";
        scalarEntry(os, 0, "value", cumulativeContErr_, precision_);
        os << "\n\n\n// ************************************************************************* //\n";
        emit(dir + "/uniform/cumulativeContErr", os.str(), true);
    }
    {
        std::ostringstream os;
        os << header("dictionary", name + "/uniform/functionObjects", "functionObjectProperties");
        os << "\n\n// ************************************************************************* //\n";
        emit(dir + "/uniform/functionObjects/functionObjectProperties", os.str(), true);
    }

    // BRAE_CONTROL_WRITE_REFUSE_LATE=1 refuses here, with every file built and none written -- the write
    // gate asserts it leaves no time directory. A case cannot reach this point refused: the start-up check
    // (refuseAtFirstWrite) mirrors every refusal above, so without the control the queue goes unwitnessed.
    if (std::getenv("BRAE_CONTROL_WRITE_REFUSE_LATE") != nullptr)
    {
        throw std::runtime_error("brae interFoam writer: BRAE_CONTROL_WRITE_REFUSE_LATE refuses " + dir
                                 + " after building its " + std::to_string(pending_.size()) + " files");
    }

    // every file is built: only now does the time directory appear
    for (const PendingFile& pf : pending_)
    {
        std::error_code ec;
        fs::create_directories(fs::path(pf.path).parent_path(), ec);
        if (ec)
        {
            throw std::runtime_error("brae interFoam writer: cannot create the directory of " + pf.path + ": "
                                     + ec.message());
        }
        writeFile(pf.path, pf.text, pf.compressible);
    }
    pending_.clear();

    // purgeWrite (TimeIO.C:559-582): the directories this run wrote, oldest removed past the limit; the
    // start directory and anything from before the run are never touched
    written_.push_back(name);
    while (purge_ > 0 && static_cast<int>(written_.size()) > purge_)
    {
        std::error_code rec;
        fs::remove_all(caseDir_ + "/" + written_.front(), rec);
        written_.pop_front();
    }
}

void registerUnwritten(
    InterWriter& w,
    const InterFields& f)
{
    const std::string a = f.alphaName;
    // the conditions: 16 types across the shipped interFoam tutorials have no transcribed write() yet
    // (contact angles, moving and rotating walls, waves, the permeable and porous pairs, the turbulent
    // inlets, variableHeightFlowRate, nutkRoughWallFunction, ...)
    w.probeConditions(a, f.alpha1);
    w.probeConditions("U", f.U);
    w.probeConditions("p_rgh", f.p_rgh);
    // k and nut are AUTO_WRITE under every closure brae carries (kEqn.C:92-98, eddyViscosity.C:60-69); LES
    // kEqn has no second scalar, and its filter width is NO_WRITE (LESdelta.C:51-57)
    if (f.turbulence.on)
    {
        w.probeConditions("k", f.turbulence.k);
        if (f.turbulence.model == InterRasModel::KOmegaSST)
        {
            w.probeConditions("omega", f.turbulence.omega);
        }
        else if (f.turbulence.model == InterRasModel::KEpsilon)
        {
            w.probeConditions("epsilon", f.turbulence.epsilon);
        }
        w.probeConditions("nut", f.turbulence.nut);
    }
    // A start directory holding a written field's `_0` level: OpenFOAM's read constructor takes it as AUTO_WRITE
    // (readOldTimeIfPresent, GeometricField.C:120, :131-160) and writes it at every write time. brae writes
    // only a sub-cycled alpha's, below; any other is refused by name rather than dropped.
    {
        std::vector<std::string> names{a, "U", "p_rgh"};
        if (f.turbulence.on)
        {
            names.insert(names.end(), {"k", "epsilon", "omega", "nut"});
        }
        for (const std::string& n : names)
        {
            if (n == a && f.alphaCtl.nAlphaSubCycles > 1)
            {
                continue;
            }
            if (w.startHolds(n + "_0"))
            {
                w.refuseAtFirstWrite(n + "_0", "the start directory holds it, and OpenFOAM reads it as an old-time "
                                               "level and writes it back at every write time");
            }
        }
    }
    // a sub-cycled alpha's old time is AUTO_WRITE (GeometricField::storeOldTime gives the level the
    // field's writeOpt once the sub-cycle has made it an old-old one, GeometricField.C:935-938) -- at every
    // write time, the first included. On a refining mesh OpenFOAM maps that level with the mesh
    // (MapGeometricFields: storeOldTimes, then the same cell mapper as alpha), so it is alpha as the mesh
    // update left it: the driver takes the level's copy after the update, not at the step's start.
    if (f.alphaCtl.nAlphaSubCycles > 1)
    {
        w.writeAlphaOld();
        // ...taken at the step's first mesh update. Under moveMeshOuterCorrectors a refining mesh updates again
        // at every corrector, and each change maps the restored old level anew (MapGeometricFields at the same
        // time index); that re-mapping is not carried, so the file is refused rather than written from the
        // first update alone.
        if (f.amr && f.amr->active && f.moveMeshOuterCorrectors)
        {
            w.refuseAtFirstWrite(a + "_0", "a sub-cycled alpha's old time on a refining mesh updated at every "
                                           "outer corrector (moveMeshOuterCorrectors), which OpenFOAM maps at "
                                           "each change");
        }
        // a restart whose start directory holds alpha_0 gives OpenFOAM's old level that FILE's contact-angle
        // gradient (readGradientEntry, alphaContactAngleTwoPhaseFvPatchScalarField.C:73-77), which brae
        // does not read
        for (std::size_t pi = 0; w.startHoldsAlphaOld() && pi < f.alpha1.boundary.size(); ++pi)
        {
            if (f.alpha1.boundary[pi]->contactAngleTheta0() >= scalar(0))
            {
                w.refuseAtFirstWrite(a + "_0", "a restart from a stored old level with a contact angle, whose "
                                               "gradient OpenFOAM reads from that file and brae does not");
                break;
            }
        }
    }
    if (f.ddtU == DdtScheme::CrankNicolson)
    {
        w.refuseAtFirstWrite("ddt0(rho,U), ddtCorrDdt0(U), U_0, phi_0",
                             "CrankNicolson's ddt0 fields and old-time levels (CrankNicolsonDdtScheme.C:137,156)");
    }
    if (f.ddtU == DdtScheme::backward)
    {
        w.refuseAtFirstWrite("the old-old time levels", "backward keeps a second old-time level, whose written "
                                                        "form is not transcribed");
    }
    if (f.lts || f.ddtU == DdtScheme::localEuler)
    {
        w.writeRDeltaT();
    }
    // a dynamic mesh's state, by what moves it. A solidBody motion writes the mesh itself -- points, meshPhi,
    // Uf -- and nothing of its own. A refining mesh writes its topology and hexRef8's state too; the other
    // motion solvers their own fields. Each is refused by name until it is written.
    if (f.meshIsDynamic)
    {
        if (f.amr && f.amr->active)
        {
            // what the writer can echo, each alternative named here and not inside write(): the patch
            // types whose polyPatch::write is type/inGroups/nFaces/startFace alone (coupled, cyclic,
            // processor and generic add entries of their own, polyPatch.C:437-443), no physicalType (brae
            // does not keep it), and the zone forms an oracle holds -- `0()`, and motorBike's empty
            // pointZone. A zone with members, or a cell or face zone entry at all, is not written.
            std::string why;
            for (const FvPatch& p : w.patches())
            {
                if (p.type != "patch" && p.type != "wall")
                {
                    why = "patch `" + p.name + "` is `" + p.type + "`";
                }
            }
            std::ifstream boundaryIn(f.amr->polyMeshDir + "/boundary");
            const std::string boundaryText((std::istreambuf_iterator<char>(boundaryIn)),
                                           std::istreambuf_iterator<char>());
            if (boundaryText.find("physicalType") != std::string::npos)
            {
                why = "a patch sets physicalType";
            }
            if (!f.amr->cellZoneEntries.empty() || !f.amr->faceZoneEntries.empty())
            {
                why = "the mesh has cell or face zone entries";
            }
            for (const ZoneEntry& z : f.amr->pointZoneEntries)
            {
                if (z.type != "pointZone" || z.nMembers != 0 || z.extraKeys)
                {
                    why = "pointZone `" + z.name + "` has members or entries beyond its type";
                }
            }
            if (!why.empty())
            {
                w.refuseAtFirstWrite("polyMesh/{boundary,cellZones,faceZones,pointZones}", why + ", which the "
                                     "refining mesh's writer does not echo");
            }
            // the motion a refining mesh may also carry (dynamicRefineFvMesh is a
            // dynamicMotionSolverListFvMesh): its points, meshPhi and Uf, and points0 after a change
            const bool moves = f.dynamicMesh != nullptr;
            if (moves && !f.dynamicMesh->solidBodyOnly())
            {
                w.refuseAtFirstWrite("the motion solver's own state", "a refining mesh moved by a solver other "
                                     "than solidBody");
            }
            if (moves)
            {
                w.writeMeshMotion();
            }
            w.writeRefineMesh(moves);
        }
        else if (f.dynamicMesh && f.dynamicMesh->solidBodyOnly())
        {
            w.writeMeshMotion();
        }
        else if (f.dynamicMesh && f.dynamicMesh->displacement())
        {
            w.writeMeshMotion();
            w.writeDisplacement();
        }
        else if (f.dynamicMesh && f.dynamicMesh->rigidBody())
        {
            w.writeMeshMotion();
            w.writeRigidBody();
        }
        else
        {
            w.refuseAtFirstWrite("the motion solver's own state",
                                 "a dynamic mesh the writer does not know (dynamicMotionSolverFvMesh.C)");
        }
    }
    // correctPhi's rAU: written where every patch holds its face cells' value, is a type-only constraint
    // (cyclic included), or is a cyclicAMI the host loop coupled -- whose value is the coupled 1/A,
    // 1/(w*A_P + (1 - w)*A_N), evaluated at the write. Any other coupled patch's is not kept.
    if (f.correctPhi)
    {
        bool coupled = false;
        for (const FvPatch& p : w.patches())
        {
            coupled = coupled || (p.type != "empty" && p.type != "wedge" && p.type != "symmetryPlane"
                                  && p.type != "symmetry" && p.type != "cyclic"
                                  && !(p.type == "cyclicAMI" && p.coupled) && isConstraintType(p.type));
        }
        if (coupled)
        {
            w.refuseAtFirstWrite("rAU", "correctPhi's field on a coupled patch (the coupled 1/A)");
        }
        else
        {
            w.writeRAU();
        }
    }
    // the wave models' state files: written when every model's entry is one the writer can echo -- words,
    // numbers and lists of them. Not echoed, each named here rather than at the write: a sub-dictionary
    // (OpenFOAM writes a nested block, which no shipped wave tutorial holds and so nothing could gate), a
    // quoted string (the tokenizer drops the quotes), a leaf the dictionary reader split at a `{` (a
    // uniform list `N{v}`), and a token OpenFOAM tokenizes differently (echoableToken).
    if (f.waves.any)
    {
        bool echoable = !f.waves.quotedString;
        for (std::size_t pi = 0; pi < f.waves.alphaPatch.size(); ++pi)
        {
            if (!f.waves.alphaPatch[pi] && !(pi < f.waves.UPatch.size() && f.waves.UPatch[pi]))
            {
                continue;
            }
            std::vector<const FoamDict*> dicts{f.waves.waveProperties.subDict(w.patches()[pi].name)};
            if (pi < f.waves.stored.size() && f.waves.stored[pi])
            {
                dicts.push_back(f.waves.stored[pi].get());
            }
            for (const FoamDict* d : dicts)
            {
                if (!d)
                {
                    continue;
                }
                echoable = echoable && d->subs.empty();
                for (const auto& leafEntry : d->leaves)
                {
                    echoable = echoable && leafEntry.first != "{" && leafEntry.first != "}";
                    for (const std::string& token : leafEntry.second)
                    {
                        echoable = echoable && echoableToken(token);
                    }
                }
            }
        }
        if (echoable)
        {
            w.writeWaveState(&f.waves);
        }
        else
        {
            w.refuseAtFirstWrite("uniform/waveProperties.<patch>", "a wave model's entry holds a "
                                 "sub-dictionary, a quoted string, a uniform `N{v}` list or a token OpenFOAM "
                                 "splits, which the writer does not echo");
        }
    }
}

std::vector<std::vector<scalar>> staticPressureBoundary(
    const GeometricField<scalar>& p_rgh,
    const std::vector<std::vector<scalar>>& rhoBnd,
    const std::vector<std::vector<scalar>>& ghfBoundary)
{
    std::vector<std::vector<scalar>> pb(p_rgh.boundary.size());
    for (std::size_t pi = 0; pi < p_rgh.boundary.size(); ++pi)
    {
        const std::vector<scalar>& prghB = p_rgh.boundary[pi]->value();
        pb[pi].assign(prghB.size(), scalar(0));
        for (std::size_t k = 0; k < prghB.size() && pi < rhoBnd.size() && k < rhoBnd[pi].size()
                                && pi < ghfBoundary.size() && k < ghfBoundary[pi].size(); ++k)
        {
            pb[pi][k] = prghB[k] + rhoBnd[pi][k]*ghfBoundary[pi][k];
        }
    }
    return pb;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
