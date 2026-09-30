// interFoam's time directories -- see inter_writer_cpp.cuh.
#include "inter_writer_cpp.cuh"
#include "inter_turbulence_cpp.cuh"
#include "inter_case_cpp.cuh"
#include "foam_token_reader.cuh"
#include "brae_notice.cuh"
#include <zlib.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

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
    const std::vector<T>& value = derived ? *derived : bc.value();

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
    // 103-107 (zeroGradient's write, then the value)
    else if (type == "calculated" || type == "fixedValue" || type == "kqRWallFunction")
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
    const std::string cce = startDir + "/uniform/cumulativeContErr";
    if (fs::exists(cce) || fs::exists(cce + ".gz"))
    {
        const FoamDict cd2 = readDict(fs::exists(cce) ? cce : cce + ".gz");
        cumulativeContErr_ = cd2.scalarOr("value", scalar(0));
    }
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
    os << "    class       " << className << ";\n";
    os << "    location    \"" << location << "\";\n";
    os << "    object      " << object << ";\n";
    os << "}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n";
    return os.str();
}

void InterWriter::emit(
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
        volPatch(os, field, p, pd, *bcs[pi], derived ? &(*derived)[pi] : nullptr, precision);
    }
    os << "}\n\n\n// ************************************************************************* //\n";
    return os.str();
}

std::string surfaceFieldText(
    const std::string& head,
    const SurfaceScalarField& f,
    const std::vector<FvPatch>& patches,
    int precision)
{
    std::ostringstream os;
    os << head;
    keyword(os, 0, "dimensions");
    os << "[0 3 -1 0 0 0 0];\n\n";
    // DimensionedFieldIO.C:160-163: a flux says it is oriented
    keyword(os, 0, "oriented");
    os << "oriented;\n\n";
    listEntry(os, 0, "internalField", f.internal, precision);
    os << "\nboundaryField\n{\n";
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        os << "    " << p.name << "\n    {\n";
        // emptyFvsPatchField.C: the type alone; calculated/coupled/wedge/symmetryPlane: type and value
        // (calculatedFvsPatchField.C, coupledFvsPatchField.C, wedgeFvsPatchField.C,
        // symmetryPlaneFvsPatchField.C)
        if (p.type == "empty")
        {
            wordEntry(os, 8, "type", "empty");
        }
        else
        {
            const bool constraint = (p.type == "wedge" || p.type == "symmetryPlane" || p.type == "symmetry"
                                  || p.type == "cyclic");
            wordEntry(os, 8, "type", constraint ? p.type : std::string("calculated"));
            const std::vector<scalar> none;
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
    const std::vector<std::vector<T>>* derived)
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
    std::error_code ec;
    fs::create_directories(dir + "/uniform/functionObjects", ec);
    if (ec)
    {
        throw std::runtime_error("brae interFoam writer: cannot create " + dir + ": " + ec.message());
    }

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
            nullptr),
        true);
    emit(
        dir + "/U",
        fieldText<vector>(
            "volVectorField",
            name,
            "U",
            "U",
            s.UCells ? *s.UCells : s.U->internal,
            s.U->boundary,
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
            nullptr),
        true);
    // p: NO_READ, AUTO_WRITE (createFields.H:91-102), `calculated` on every non-constraint patch, with
    // p_rgh's dimensions -- p_rgh's template, whose patch entries the derived values override
    emit(
        dir + "/p",
        fieldText<scalar>(
            "volScalarField",
            name,
            "p",
            "p_rgh",
            *s.p,
            s.p_rgh->boundary,
            s.pBoundary),
        true);
    emit(dir + "/phi", surfaceFieldText(header("surfaceScalarField", name, "phi"), *s.phi, patches_, precision_),
         true);
    const std::string aPhiName = "alphaPhi0." + phase1_;
    emit(dir + "/" + aPhiName,
         surfaceFieldText(header("surfaceScalarField", name, aPhiName), *s.alphaPhi0, patches_, precision_), true);

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
                    nullptr),
                true);
        }
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
    if (f.turbulence.on && f.turbulence.model != InterRasModel::KEqnLES)
    {
        w.probeConditions("k", f.turbulence.k);
        if (f.turbulence.model == InterRasModel::KOmegaSST)
        {
            w.probeConditions("omega", f.turbulence.omega);
        }
        else
        {
            w.probeConditions("epsilon", f.turbulence.epsilon);
        }
        w.probeConditions("nut", f.turbulence.nut);
    }
    if (f.alphaCtl.nAlphaSubCycles > 1)
    {
        w.refuseAtFirstWrite(a + "_0", "a sub-cycled alpha keeps its old time, which OpenFOAM writes from the "
                                       "second step (subCycle.H:76-84, GeometricField.C:933-936)");
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
        w.refuseAtFirstWrite("rDeltaT", "local time stepping's field (createRDeltaT.H)");
    }
    if (f.meshIsDynamic)
    {
        w.refuseAtFirstWrite("Uf, meshPhi, polyMesh/points (and a refining mesh's cellLevel, pointLevel, "
                             "refinementHistory)",
                             "a dynamic mesh's state (createUfIfPresent.H, fvMesh.C:1103-1114, polyMesh.C:1232)");
    }
    if (f.correctPhi)
    {
        w.refuseAtFirstWrite("rAU", "correctPhi's field (initCorrectPhi.H:3-17)");
    }
    if (f.waves.any)
    {
        w.refuseAtFirstWrite("uniform/waveProperties.<patch>", "the wave models' state (waveModel.C:250-261)");
    }
    if (f.turbulence.on && f.turbulence.model == InterRasModel::KEqnLES)
    {
        w.refuseAtFirstWrite("k and nut of LES kEqn", "their written form is not gated yet");
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
