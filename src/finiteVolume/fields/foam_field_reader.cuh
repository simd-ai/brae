#pragma once
// Reads an OpenFOAM field file: dimensions / internalField (uniform|nonuniform) /
// boundaryField { patch { type; value; ... } }. Templated on the value type (scalar/vector).
// ASCII for now; binary field values land with the binary field reader (later increment).
#include "cf_types.cuh"
#include "function1.cuh"   // OF Function1: constant / table
#include "foam_token_reader.cuh"
#include <string>
#include <type_traits>
#include <vector>
#include <filesystem>

namespace brae {

template <typename T> inline T readFoamValue(TokenStream& ts);
template <> inline scalar readFoamValue<scalar>(TokenStream& ts) { return ts.nextScalar(); }
template <> inline vector readFoamValue<vector>(TokenStream& ts)
{
    ts.expect("(");
    vector v{ts.nextScalar(), ts.nextScalar(), ts.nextScalar()};
    ts.expect(")");
    return v;
}
// OF SymmTensor's I/O order, which is also its component order: (xx xy xz yy yz zz).
template <> inline symmTensor readFoamValue<symmTensor>(TokenStream& ts)
{
    ts.expect("(");
    symmTensor t{ts.nextScalar(), ts.nextScalar(), ts.nextScalar(),
                 ts.nextScalar(), ts.nextScalar(), ts.nextScalar()};
    ts.expect(")");
    return t;
}

// Skip an unhandled dict entry's tokens up to (not including) its terminating ';', paren-aware so nested "(...)" lists
// / tables / Function1 entries are consumed whole. initialDepth accounts for a leading '(' the caller already read.
// Leaves the ';' in place for the caller's ts.expect(";"). Single source for the reader's 3 table/skip fallbacks.
// Skip an entry's VALUE. Returns true if it consumed a sub-dictionary (which has no trailing ';', so the
// caller must not expect one) and false for an ordinary value terminated by ';'.
//
// Brace tracking is not decoration. A patch entry can carry a sub-DICTIONARY value -- fanPressure's
//     fanCurve { type table; file "<constant>/FluxVsdP.dat"; }
// is one -- and a paren-only scan stops at the ';' INSIDE that block. The block's own '}' is then taken
// for the patch's, and every later patch of the field is swallowed: the failure surfaced as "no
// boundaryField entry for patch outlet1" on pimpleFoam/RAS/TJunctionFan, naming a patch that is plainly
// there and pointing nowhere near the fanCurve two entries above it. The polyMesh boundary parser had the
// identical bug for cyclicACMI's `scaleCoeffs` block.
inline bool skipToSemicolon(TokenStream& ts, int initialDepth = 0)
{
    if (initialDepth == 0 && ts.peek() == "{")   // sub-dictionary value: skip it balanced, no ';' follows
    {
        ts.expect("{");
        for (int d = 1; d > 0; )
        {
            const std::string t = ts.next();
            if      (t == "{") ++d;
            else if (t == "}") --d;
        }
        return true;
    }
    int depth = initialDepth;
    while (!(depth == 0 && ts.peek() == ";"))
    {
        const std::string s = ts.next();
        if (s == "(") ++depth;
        else if (s == ")") --depth;
    }
    return false;
}

template <typename T>
struct PatchFieldData
{
    std::string    name;
    std::string    type;
    bool           hasValue     = false;
    // uniformFixedValue whose uniformValue is a Function1 (table / polynomial / coded / expression)
    // rather than a constant. brae cannot evaluate those, and the danger is that the entry ALSO carries a
    // stale `value` from an overridden fixedValue entry, so the field silently takes that constant
    // instead. Recorded here and refused at construction rather than guessed at.
    std::string    unsupportedFunction1;
    Function1      p0Function1;              // uniformTotalPressure p0(t); empty unless hasP0Function1
    bool           hasP0Function1 = false;
    // pressureInletOutletVelocity's optional `tangentialVelocity`. OF sets
    // refValue = tv - n*(n & tv) (pressureInletOutletVelocityFvPatchVectorField.C:135); brae leaves the
    // tangential refValue at zero, so honouring the key would need per-face storage it does not have.
    // Recorded so it can be refused instead of quietly changing the boundary condition.
    bool           hasTangentialVelocity = false;
    // fixedGradient (and the heat-flux BCs derived from it): the prescribed normal gradient.
    // Plain `mixed` (Robin) carries refValue + refGradient + valueFraction. refGradient shares the
    // gradient* slots below; these two are its own. Distinct from inletValue*, which inletOutlet and the
    // freestream family use for a refValue whose blend the DEVICE recomputes each step.
    bool           hasRefValue      = false;
    bool           refValueUniform  = false;
    T              refValueUniformValue{};
    std::vector<T> refValues;
    bool           hasValueFraction = false;
    bool           vfUniform        = false;
    scalar         vfUniformValue   = 0;
    std::vector<scalar> vfValues;

    bool           hasGradient    = false;
    bool           gradientUniform = false;
    T              gradientUniformValue{};
    std::vector<T> gradientValues;
    bool           valueUniform = false;
    T              uniformValue{};
    std::vector<T> values;
    // uniformFixedValue's `uniformValue` PatchFunction1, kept in its OWN slot. It shares the `value`
    // slot above only in the sense that OF writes both and they agree; brae must write it back because
    // OF's PatchFunction1::New REQUIRES it -- an output missing it cannot be read by OpenFOAM at all.
    bool           hasUniformFn1 = false;
    T              uniformFn1Value{};
    // inletOutlet (and similar mixed BCs): the inflow value. Used as the device refValue.
    bool           hasInletValue   = false;
    bool           inletUniform    = false;
    T              inletUniformValue{};
    std::vector<T> inletValues;
    // turbulent-inlet BCs: k = 1.5*(intensity*|U|)^2; eps = Cmu^0.75 k^1.5/mixingLength; omega = sqrt(k)/(Cmu^0.25 mixingLength).
    scalar         intensity    = 0;
    // compressible::alphatWallFunction: alphat_w = rho_w*nut_w/Prt. OF's default here is 0.85
    // (alphatWallFunctionFvPatchScalarField.C: dict.getOrDefault<scalar>("Prt", 0.85)) and is NOT the
    // turbulence model's own Prt, whose default is 1.0. Two different numbers in the same case.
    scalar         Prt          = 0.85;
    // totalPressure: OF picks its formula from these. psi "none" (the default) -> the low-speed form
    // p0 - 0.5*rho*neg(phi)*|U|^2; a NAMED psi -> the isentropic high-speed form with gamma, which brae
    // does not implement. Recorded so it can be refused by name instead of silently running low-speed.
    std::string    psiName      = "none";
    scalar         gammaTP      = 1.0;
    // flowRateInletVelocity (OF flowRateInletVelocityFvPatchVectorField). OF selects the branch by which
    // key is present: "volumetricFlowRate" -> volumetric_ = true; otherwise "massFlowRate" (default
    // rhoName "rho"). rhoInlet is only the FALLBACK used when the rho field is not registered -- in
    // rhoSimpleFoam it is, so the real patch rho is used and rhoInlet is ignored, exactly as OF does.
    bool           hasFlowRate  = false;
    bool           flowRateIsMass = false;
    scalar         flowRate     = 0.0;
    scalar         rhoInlet     = -1.0;   // OF default -VGREAT ("not given")
    bool           extrapolateProfile = false;
    scalar         mixingLength = 0;
    // surfaceNormalFixedValue / uniformNormalFixedValue: SCALAR refValue; the BC builds U_b = refValue * face_normal.
    bool                hasNormalRef     = false;
    bool                normalRefUniform = false;
    scalar              normalRefUniformValue = 0;
    std::vector<scalar> normalRefValues;
    // timeVaryingMappedFixedValue: the external boundaryData points + values (read here; the BC maps them to the faces).
    bool                hasMapData = false;
    std::vector<vector> mapPoints;
    std::vector<T>      mapValues;
    // atmBoundaryLayerInlet{Velocity,K,Epsilon,Omega}: log-law ABL profile params (the BC evaluates the profile per
    // face from Cf). u* = kappa*|Uref|/ln((Zref+z0)/z0); U(z)=(u*/kappa)ln((z-d+z0)/z0)*flowDir; k=u*^2/sqrt(Cmu);
    // eps=u*^3/(kappa(z-d+z0)); omega=u*/(sqrt(Cmu)kappa(z-d+z0)); z = Cf.zDir.
    bool   hasABL = false;
    scalar ablUref = 0, ablZref = 0, ablZ0 = 0.1, ablD = 0, ablKappa = 0.41, ablCmu = 0.09;
    // YGCJ curve-fit coefficients (atmBoundaryLayer.C:70-71 getOrDefault; .H:178-179). The DEFAULTS
    // make sqrt(C1*log(..)+C2) exactly 1, which is the only profile brae computed before these were
    // parsed -- a case setting either got the default silently.
    scalar ablC1 = 0.0, ablC2 = 1.0;
    bool   atmBoundNut = true;   // atmNutkWallFunction boundNut option (clamp nut>=0); z0 is stored in ablZ0.
    // epsilonWallFunction `lowReCorrection` (epsilonWallFunctionFvPatchScalarField.C:414,
    // getOrDefault("lowReCorrection", false)). On a face with y+ < yPlusLam it switches epsilon from the
    // log form to the VISCOUS one AND drops that face's wall production entirely -- see kEpsilon_cpp.
    bool   epsLowRe = false;
    vector ablFlowDir{1, 0, 0}, ablZDir{0, 0, 1};
};

template <typename T>
struct FieldData
{
    bool                          internalUniform = true;
    T                             internalUniformValue{};
    std::vector<T>                internalField;   // when nonuniform
    std::vector<PatchFieldData<T>> boundary;
};

// Read "uniform <v>" or "nonuniform List<...> N ( ... )".
template <typename T>
inline void readUniformOrList(
    TokenStream& ts,
    bool& uniform,
    T& uval,
    std::vector<T>& vals)
{
    const std::string mode = ts.next();
    if (mode == "uniform")
    {
        uniform = true;
        uval = readFoamValue<T>(ts);
    }
    else   // nonuniform
    {
        uniform = false;
        ts.next();                 // List<scalar> / List<vector>
        const label n = ts.nextLabel();
        ts.expect("(");
        vals.resize(n);
        for (label i = 0; i < n; ++i)
            vals[i] = readFoamValue<T>(ts);
        ts.expect(")");
    }
}

// Read "uniform <v>" / "nonuniform List<...>" OR the OF self-reference "$internalField" (copy the internalField
// entry). Used by value / inletValue / freestreamValue, any of which may be written as $internalField.
// Does this token start a numeric literal? Used to spot a bare (keyword-less) value entry.
inline bool isFoamNumber(const std::string& t)
{
    if (t.empty()) return false;
    const char c = t[0];
    return (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.';
}

// OF v2412 writes atmBoundaryLayer's Uref/Zref/z0/d as Function1 and flowDir/zDir as PatchFunction1,
// so a field WRITTEN by OpenFOAM reads `flowDir constant (1 0 0);` where the tutorial's own
// include/ABLConditions has the bare `flowDir (1 0 0);`. Both are the same entry: OF's Function1 parser
// accepts a bare value as shorthand for `constant <v>`. Consume the keyword so the caller sees the value.
// Anything else (table / polynomial / sine / coded) IS time-varying: name it so dispatch can refuse,
// because reading past it would leave the caller's scalar/vector at its default and look like a run.
inline bool takeConstantFunction1(TokenStream& ts, std::string& unsupported, const std::string& key)
{
    const std::string m = ts.peek();
    if (m == "constant" || m == "uniform")
    {
        ts.next();
        return true;
    }
    if (m == "(" || isFoamNumber(m)) return true;      // bare value: OF's own shorthand
    unsupported = key + " " + m;
    skipToSemicolon(ts, 0);
    return false;
}
// OF Function1 accepts a BARE value as shorthand for `constant <v>`: `uniformValue (0 0 0);` and
// `uniformValue 5;` are constants, not tables. brae required the keyword, so a bare vector was
// classified as an unsupported Function1 and the case refused -- simpleFoam/turbineSiting's terrain
// patch is exactly `uniformFixedValue` with `uniformValue (0 0 0)`.
template <typename T>
inline T readBareFoamValue(TokenStream& ts, const std::string& first);

template <>
inline scalar readBareFoamValue<scalar>(TokenStream& ts, const std::string& first)
{
    (void)ts;
    return static_cast<scalar>(std::strtod(first.c_str(), nullptr));
}

template <>
inline vector readBareFoamValue<vector>(TokenStream& ts, const std::string& first)
{
    // `first` is already the '('; the three components and the ')' remain.
    (void)first;
    vector v{};
    v.x = ts.nextScalar();
    v.y = ts.nextScalar();
    v.z = ts.nextScalar();
    ts.expect(")");
    return v;
}

template <>
inline symmTensor readBareFoamValue<symmTensor>(TokenStream& ts, const std::string& first)
{
    // `first` is already the '('; the six components (xx xy xz yy yz zz) and the ')' remain.
    (void)first;
    symmTensor t{};
    t.xx = ts.nextScalar();
    t.xy = ts.nextScalar();
    t.xz = ts.nextScalar();
    t.yy = ts.nextScalar();
    t.yz = ts.nextScalar();
    t.zz = ts.nextScalar();
    ts.expect(")");
    return t;
}


template <typename T>
inline void readValueOrInternal(
    TokenStream& ts,
    const FieldData<T>& fd,
    bool& uniform,
    T& uval,
    std::vector<T>& vals)
{
    if (ts.peek() == "$internalField")
    {
        ts.next();
        uniform = fd.internalUniform;
        uval = fd.internalUniformValue;
        vals = fd.internalField;
    }
    // A BARE value, i.e. no `uniform`/`nonuniform` keyword: `inletValue (0 0 0);`. OpenFOAM tolerates
    // these because a boundary condition only reads the entries it cares about -- gasMixing's
    // pressureInletOutletVelocity carries an `inletValue` that OF's implementation never looks at, so OF
    // runs the case fine. brae parsed every known key unconditionally and died with
    // "TokenStream: expected '(' got '0'", failing on a file OpenFOAM accepts and, worse, failing BEFORE
    // reaching its own legitimate refusal (nutUWallFunction), so the message pointed at the wrong thing.
    else if (ts.peek() == "(" || isFoamNumber(ts.peek()))
    {
        uniform = true;
        uval = readFoamValue<T>(ts);
        vals.clear();
    }
    else readUniformOrList(ts, uniform, uval, vals);
}

// timeVaryingMappedFixedValue boundaryData
// Read an OF boundaryData list file: [comments] N ( v1 v2 ... ). Comments are stripped by the tokenizer.
template <typename V>
inline std::vector<V> readBoundaryDataList(const std::string& file)
{
    TokenStream ts(file);
    const label n = ts.nextLabel();
    ts.expect("(");
    std::vector<V> vals(n);
    for (label i = 0; i < n; ++i)
        vals[i] = readFoamValue<V>(ts);
    return vals;   // closing ')' not required by the mapper
}
// Read constant/boundaryData/<patch>/{points, <earliestTime>/<field>}. Steady simpleFoam: the earliest boundaryData
// time IS the (constant) profile (no time interpolation). The BC then maps these points->faces (nearest).
template <typename T>
inline void readTimeVaryingMapped(
    const std::string& fieldPath,
    const std::string& patchName,
    PatchFieldData<T>& p)
{
    namespace fs = std::filesystem;
    const std::size_t s1 = fieldPath.rfind('/');
    const std::string field   = fieldPath.substr(s1 + 1);
    const std::string timeP   = fieldPath.substr(0, s1);
    const std::string caseDir = timeP.substr(0, timeP.rfind('/'));
    const std::string bd = caseDir + "/constant/boundaryData/" + patchName;
    std::string best;
    double bestT = 1e300;
    for (const auto& e : fs::directory_iterator(bd))
    {
        if (!e.is_directory()) continue;
        try
        {
            const double t = std::stod(e.path().filename().string());
            if (t < bestT)
            {
                bestT = t;
                best = e.path().filename().string();
            }
        }
        catch (...) {}
    }
    if (best.empty()) return;
    p.mapPoints = readBoundaryDataList<vector>(bd + "/points");
    p.mapValues = readBoundaryDataList<T>(bd + "/" + best + "/" + field);
    p.hasMapData = (p.mapPoints.size() == p.mapValues.size() && !p.mapPoints.empty());
}

// One COMPONENT of a symmTensor field, as a scalar FieldData -- so a `sigma` read from 0/sigma can go
// through the ordinary scalar machinery (buildField, DeviceBoundary, the scalar transport) six times.
//
// The Maxwell tutorials use exactly three boundary kinds on sigma: fixedValue with a value, zeroGradient,
// and the constraint types (empty/symmetry/cyclic), which carry no value. Anything that DOES carry data
// this does not split is refused rather than silently dropped -- a fixedGradient sigma boundary quietly
// becoming zeroGradient is the class of bug that never shows up as an error.
inline FieldData<scalar> symmTensorComponent(const FieldData<symmTensor>& fd, int k)
{
    auto comp = [](const symmTensor& t, int i) -> scalar
    {
        switch (i)
        {
            case 0: return t.xx;
            case 1: return t.xy;
            case 2: return t.xz;
            case 3: return t.yy;
            case 4: return t.yz;
            default: return t.zz;
        }
    };
    FieldData<scalar> out;
    out.internalUniform      = fd.internalUniform;
    out.internalUniformValue = comp(fd.internalUniformValue, k);
    out.internalField.reserve(fd.internalField.size());
    for (const symmTensor& t : fd.internalField) out.internalField.push_back(comp(t, k));

    for (const PatchFieldData<symmTensor>& p : fd.boundary)
    {
        if (p.hasGradient || p.hasInletValue || p.hasRefValue || p.hasValueFraction || p.hasMapData
         || p.hasNormalRef || p.hasFlowRate || !p.unsupportedFunction1.empty())
            throw std::runtime_error(
                "brae: sigma patch '" + p.name + "' is a '" + p.type + "', whose data brae does not know "
                "how to split into components. The Maxwell stress supports fixedValue, zeroGradient and "
                "the constraint types; anything else would be read and then dropped.");
        PatchFieldData<scalar> q;
        q.name         = p.name;
        q.type         = p.type;
        q.hasValue     = p.hasValue;
        q.valueUniform = p.valueUniform;
        q.uniformValue = comp(p.uniformValue, k);
        q.hasUniformFn1   = p.hasUniformFn1;
        q.uniformFn1Value = comp(p.uniformFn1Value, k);
        q.values.reserve(p.values.size());
        for (const symmTensor& t : p.values) q.values.push_back(comp(t, k));
        out.boundary.push_back(std::move(q));
    }
    return out;
}

template <typename T>
inline FieldData<T> readField(const std::string& path)
{
    TokenStream ts(path, /*expandVars=*/true);   // expand in-file $macros (e.g. Uinlet (0 1 0); ... $Uinlet)
    FieldData<T> fd;

    while (!ts.eof())
    {
        const std::string t = ts.next();
        if (t == "internalField")
        {
            readUniformOrList(ts, fd.internalUniform, fd.internalUniformValue, fd.internalField);
            ts.expect(";");
        }
        else if (t == "boundaryField")
        {
            ts.expect("{");
            while (ts.peek() != "}")
            {
                const std::string pname = ts.next();
                if (ts.peek() != "{")
                {
                    // Non-sub-dict entry at the boundaryField level: a leftover variable definition spliced from an
                    // #include'd ABLConditions/initialConditions at that scope (e.g. "Uref 10.0;", referenced later
                    // via $z0 -- turbineSiting). OpenFOAM keeps these as dict variables; brae's $-expansion has already
                    // resolved the references, so skip the definition to its ';'. Real patches are always "name { ... }".
                    skipToSemicolon(ts);
                    ts.expect(";");
                    continue;
                }
                PatchFieldData<T> p;
                p.name = pname;
                ts.expect("{");
                while (ts.peek() != "}")
                {
                    const std::string key = ts.next();
                    if (key == ";") continue;                // stray ';' left by a subdict-macro ($intakeType1;) expansion
                    if (key == "type")
                    {
                        p.type = ts.next();
                        ts.expect(";");
                        if (p.type.rfind("atmBoundaryLayer", 0) == 0) p.hasABL = true;
                    }
                    // atmBoundaryLayerInlet* params (from the case's include/ABLConditions, #include-expanded). z0-specific
                    // keys parse always; the generic-named ones (d/kappa/Cmu) only when this is an ABL entry.
                    else if (key == "Uref")
                    {
                        if (takeConstantFunction1(ts, p.unsupportedFunction1, key))
                        {
                            p.ablUref = ts.nextScalar();
                            ts.expect(";");
                        }
                    }
                    else if (key == "Zref")
                    {
                        if (takeConstantFunction1(ts, p.unsupportedFunction1, key))
                        {
                            p.ablZref = ts.nextScalar();
                            ts.expect(";");
                        }
                    }
                    else if (key == "z0")
                    {
                        if (takeConstantFunction1(ts, p.unsupportedFunction1, key))
                        {
                            p.ablZ0 = ts.nextScalar();
                            ts.expect(";");
                        }
                    }
                    else if (key == "boundNut")   // atmNutkWallFunction: clamp nut>=0 (true/false)
                    {
                        const std::string v = ts.next();
                        p.atmBoundNut = (v == "true" || v == "yes" || v == "on" || v == "1");
                        ts.expect(";");
                    }
                    else if (key == "lowReCorrection")   // epsilonWallFunction: resolved-sublayer branch
                    {
                        const std::string v = ts.next();
                        p.epsLowRe = (v == "true" || v == "yes" || v == "on" || v == "1");
                        ts.expect(";");
                    }
                    else if (key == "flowDir" || key == "zDir")
                    {
                        if (takeConstantFunction1(ts, p.unsupportedFunction1, key))
                        {
                            ts.expect("(");
                            const vector v{ts.nextScalar(), ts.nextScalar(), ts.nextScalar()};
                            ts.expect(")");
                            ts.expect(";");
                            if (key == "flowDir") p.ablFlowDir = v;
                            else p.ablZDir = v;
                        }
                    }
                    else if (key == "d" && p.hasABL)
                    {
                        if (takeConstantFunction1(ts, p.unsupportedFunction1, key))
                        {
                            p.ablD = ts.nextScalar();
                            ts.expect(";");
                        }
                    }
                    else if ((key == "C1" || key == "C2") && p.hasABL)
                    {
                        // GATED on hasABL: bare C1/C2 are also kEpsilon coefficient names, and an
                        // ungated parse would swallow an unrelated entry that today skips harmlessly.
                        const scalar v = ts.nextScalar();
                        if (key == "C1") p.ablC1 = v;
                        else             p.ablC2 = v;
                        ts.expect(";");
                    }
                    else if ((key == "kappa" || key == "Cmu") && p.hasABL)
                    {
                        const scalar v = ts.nextScalar();
                        ts.expect(";");
                        if (key == "kappa") p.ablKappa = v;
                        else p.ablCmu = v;
                    }
                    // `ramp` multiplies the normal-velocity BCs' value by a Function1 of time every
                    // updateCoeffs (surfaceNormalFixedValueFvPatchVectorField.C:63-65). brae evaluates
                    // no Function1 here, so the key is MARKED and the factory refuses by name -- it
                    // used to fall into the unhandled-key skip, a constant inlet where the case asked
                    // for a ramp.
                    else if (key == "ramp" && (p.type == "surfaceNormalFixedValue"
                                            || p.type == "uniformNormalFixedValue"))
                    {
                        p.unsupportedFunction1 = "ramp";
                        skipToSemicolon(ts, 0);
                        ts.expect(";");
                    }
                    // surfaceNormalFixedValue refValue / uniformNormalFixedValue uniformValue: SCALAR (U_b = refValue * n).
                    else if ((key == "refValue" && p.type == "surfaceNormalFixedValue") ||
                             (key == "uniformValue" && p.type == "uniformNormalFixedValue"))
                    {
                        const std::string m = ts.next();     // uniform <s> | constant <s> | nonuniform List<scalar> | table(...)
                        if (m == "uniform" || m == "constant")
                        {
                            p.normalRefUniform = true;
                            p.normalRefUniformValue = ts.nextScalar();
                            p.hasNormalRef = true;
                        }
                        else if (m == "nonuniform")
                        {
                            p.normalRefUniform = false;
                            ts.next();
                            const label n = ts.nextLabel();
                            ts.expect("(");
                            p.normalRefValues.resize(n);
                            for (label i = 0; i < n; ++i)
                                p.normalRefValues[i] = ts.nextScalar();
                            ts.expect(")");
                            p.hasNormalRef = true;
                        }
                        else   // table/expression/... -- a Function1 brae cannot evaluate
                        {
                            // MARK it, so the factory refuses by name. This branch used to skip
                            // silently ("ramp handles it"), which nothing did: the patch built with an
                            // EMPTY value array and every face got U_b = 0*n -- a zero inlet where the
                            // case prescribed a ramp.
                            p.unsupportedFunction1 = (m == "(" ? "inline Function1" : m);
                            skipToSemicolon(ts, m == "(" ? 1 : 0);
                        }
                        ts.expect(";");
                    }
                    else if (key == "inletValue"     // inletOutlet inflow value (may be $internalField)
                          || key == "outletValue")   // outletInlet OUTflow value -- the same slot, the
                    {                                // opposite flux branch (see OutletInletPatchField)
                        readValueOrInternal(ts, fd, p.inletUniform, p.inletUniformValue, p.inletValues);
                        p.hasInletValue = true;
                        ts.expect(";");
                    }
                    else if (key == "p0")   // totalPressure reference p0 (reuse the inletValue slot)
                    {
                        // p0 is a Function1 on uniformTotalPressure: it may be `table (...)`,
                        // `polynomial`, `csvFile`, an expression -- not just a field value.
                        // readValueOrInternal only knows uniform/nonuniform, so a table made the
                        // TOKENISER fail mid-parse ("expected '(' got '0'", pimpleFoam/RAS/TJunction).
                        // That is a raw parser error where this codebase's rule is that unsupported
                        // input is NAMED. Record it and let the dispatch refuse by name instead.
                        const std::string m = ts.peek();
                        if (m == "uniform" || m == "nonuniform")
                        {
                            readValueOrInternal(ts, fd, p.inletUniform, p.inletUniformValue, p.inletValues);
                            p.hasInletValue = true;
                        }
                        else if (m == "constant")
                        {
                            ts.next();
                            p.inletUniformValue = readFoamValue<T>(ts);
                            p.inletUniform = true;
                            p.hasInletValue = true;
                        }
                        else if (isFoamNumber(m))
                        {
                            p.inletUniformValue = readBareFoamValue<T>(ts, ts.next());
                            p.inletUniform = true;
                            p.hasInletValue = true;
                        }
                        else if (m == "table")
                        {
                            // OF Function1 `table ((t v) (t v) ...)`: linear between entries, CLAMPed
                            // outside (TableBase.C:76). pimpleFoam/RAS/TJunction ramps p0 this way.
                            ts.next();                       // "table"
                            ts.expect("(");
                            std::vector<std::pair<scalar, scalar>> pts;
                            while (!ts.eof() && ts.peek() != ")")
                            {
                                ts.expect("(");
                                const scalar tt = ts.nextScalar();
                                const scalar vv = ts.nextScalar();
                                ts.expect(")");
                                pts.emplace_back(tt, vv);
                            }
                            ts.expect(")");
                            p.p0Function1 = Function1::table(std::move(pts));
                            p.hasP0Function1 = true;

                            // Seed the constant slot with t = 0 so a solver that never advances time
                            // still has a defined p0 rather than zero.
                            // p0 is a PRESSURE: scalar only. The reader is templated on T, so guard
                            // rather than cast -- a vector field has no p0 and must not silently get one.
                            if constexpr (std::is_same_v<T, scalar>)
                            {
                                p.inletUniformValue = p.p0Function1.value(0);
                                p.inletUniform = true;
                                p.hasInletValue = true;
                            }
                        }
                        else
                        {
                            p.unsupportedFunction1 = m;
                            ts.next();
                            skipToSemicolon(ts, 0);
                        }
                        ts.expect(";");
                    }
                    else if (key == "uniformValue")   // uniformFixedValue: steady PatchFunction1 "constant <v>"
                    {
                        const std::string m = ts.next();     // "constant" | "uniform" | a BARE value
                        if (m == "constant" || m == "uniform")
                        {
                            p.uniformValue = readFoamValue<T>(ts);
                            p.valueUniform = true;
                            p.hasValue = true;
                            p.hasUniformFn1 = true;
                            p.uniformFn1Value = p.uniformValue;
                        }
                        else if (m == "(" || isFoamNumber(m))
                        {
                            // Bare constant (see readBareFoamValue): OF's Function1 shorthand.
                            p.uniformValue = readBareFoamValue<T>(ts, m);
                            p.valueUniform = true;
                            p.hasValue = true;
                            p.hasUniformFn1 = true;
                            p.uniformFn1Value = p.uniformValue;
                        }
                        else   // table / polynomial / coded / expression: skip the entry, then REFUSE.
                        {
                            // Relying on "dispatch throws when there is no value" is not enough: a case
                            // that overrides an earlier `type fixedValue; value uniform X;` still has
                            // hasValue == true, so the Function1 silently degrades to the constant X.
                            // squareBendLiq does exactly that (T walls: expression, stale value 350).
                            // A dict form ({ type expression; ... }) names its Function1 inside, so peek
                            // the `type` keyword -- "a dictionary" is a much worse error message than
                            // "expression".
                            p.unsupportedFunction1 = m;
                            if (m == "{")
                            {
                                const std::string t1 = ts.peek();
                                if (t1 == "type")
                                {
                                    ts.next();
                                    p.unsupportedFunction1 = ts.peek();
                                }
                            }
                            skipToSemicolon(ts, m == "(" ? 1 : 0);
                        }
                        ts.expect(";");
                    }
                    else if (key == "value")
                    {
                        readValueOrInternal(ts, fd, p.valueUniform, p.uniformValue, p.values);   // uniform/nonuniform/$internalField
                        p.hasValue = true;
                        ts.expect(";");
                    }
                    else if (key == "intensity")   // turbulentIntensityKineticEnergyInlet
                    {
                        p.intensity = ts.nextScalar();
                        ts.expect(";");
                    }
                    // OF takes a Function1 here; "constant <v>" and a bare "<v>" are the steady forms.
                    // Anything else (table/polynomial/...) is refused by name rather than approximated.
                    else if (key == "volumetricFlowRate" || key == "massFlowRate")
                    {
                        p.hasFlowRate = true;
                        p.flowRateIsMass = (key == "massFlowRate");
                        std::string w = ts.next();
                        if (w == "constant") w = ts.next();
                        else if (w == "{")
                            throw std::runtime_error(
                                "brae: flowRateInletVelocity '" + key + "' given as a Function1 dictionary on patch "
                                + p.name + "; only 'constant <value>' (or a bare value) is supported.");
                        p.flowRate = std::stod(w);
                        ts.expect(";");
                    }
                    else if (key == "rhoInlet")
                    {
                        p.rhoInlet = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "extrapolateProfile")
                    {
                        const std::string w = ts.next();
                        p.extrapolateProfile = (w == "true" || w == "yes" || w == "on" || w == "1");
                        ts.expect(";");
                    }
                    else if (key == "psi")    // totalPressure: selects OF's isentropic branch when != none
                    {
                        p.psiName = ts.next();
                        ts.expect(";");
                    }
                    else if (key == "gamma")   // totalPressure isentropic exponent
                    {
                        p.gammaTP = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "Prt")   // compressible::alphatWallFunction turbulent Prandtl number
                    {
                        p.Prt = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "mixingLength")   // turbulentMixingLength*Inlet
                    {
                        p.mixingLength = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "freestreamValue")   // freestream/freestreamPressure farfield value (may be $internalField)
                    {
                        readValueOrInternal(ts, fd, p.valueUniform, p.uniformValue, p.values);
                        p.hasValue = true;
                        p.inletUniform = p.valueUniform;
                        p.inletUniformValue = p.uniformValue;
                        p.inletValues = p.values;
                        p.hasInletValue = true;
                        ts.expect(";");
                    }
                    // `gradient` is fixedGradient's key, `refGradient` is mixed's. They fill the same slot
                    // because they mean the same thing to the discretisation -- the difference is the (1-vf)
                    // weight the kernels apply on a mixed patch (OF mixedFvPatchField.C:279-310).
                    else if (key == "gradient" || key == "refGradient")
                    {
                        readValueOrInternal(ts, fd, p.gradientUniform, p.gradientUniformValue, p.gradientValues);
                        p.hasGradient = true;
                        ts.expect(";");
                    }
                    // mixedEnergy is OpenFOAM's own spelling for a mixed he patch (basicThermo maps
                    // T `mixed` onto it), and it is what OF WRITES into an he output file -- so a
                    // restart that read `type mixedEnergy` through a mixed-only gate lost its refValue
                    // and rebuilt the patch around zero.
                    else if (key == "refValue" && (p.type == "mixed" || p.type == "mixedEnergy"))
                    {
                        readValueOrInternal(ts, fd, p.refValueUniform, p.refValueUniformValue, p.refValues);
                        p.hasRefValue = true;
                        ts.expect(";");
                    }
                    else if (key == "valueFraction")
                    {
                        // Always a SCALAR field, even on a vector patch (OF blends component-wise with one
                        // fraction), so it cannot go through readValueOrInternal<T>.
                        const std::string w = ts.peek();
                        if (w == "uniform")
                        {
                            ts.next();
                            p.vfUniform = true;
                            p.vfUniformValue = std::stod(ts.next());
                        }
                        else
                        {
                            ts.next();                      // nonuniform
                            if (ts.peek() == "List<scalar>") ts.next();
                            const int n = std::stoi(ts.next());
                            ts.expect("(");
                            p.vfValues.resize(static_cast<std::size_t>(n));
                            for (int i = 0; i < n; ++i) p.vfValues[static_cast<std::size_t>(i)] = std::stod(ts.next());
                            ts.expect(")");
                        }
                        p.hasValueFraction = true;
                        ts.expect(";");
                    }
                    else if (key == "tangentialVelocity")   // pressureInletOutletVelocity, optional
                    {
                        p.hasTangentialVelocity = true;
                        skipToSemicolon(ts);
                    }
                    else
                    {
                        // skip any other (unhandled) entry: a plain value up to its ';', or a whole
                        // sub-dictionary (which carries no ';' of its own)
                        if (!skipToSemicolon(ts)) ts.expect(";");
                    }
                }
                ts.expect("}");
                if (p.type == "timeVaryingMappedFixedValue")
                {
                    // OF requires constant/boundaryData/<patch>; do NOT swallow a read failure into a silent
                    // zeroGradient/fixedValue fallback -- rethrow with context so the misconfiguration is detected.
                    try
                    {
                        readTimeVaryingMapped(path, p.name, p);
                    }
                    catch (const std::exception& e)
                    {
                        throw std::runtime_error("brae: timeVaryingMappedFixedValue on patch '" + p.name +
                            "' requires constant/boundaryData/" + p.name + " (points + a time dir); read failed: " + e.what());
                    }
                }
                fd.boundary.push_back(std::move(p));
            }
            ts.expect("}");
        }
        // dimensions and other top-level entries are skipped token-by-token.
    }
    return fd;
}

} // namespace brae
