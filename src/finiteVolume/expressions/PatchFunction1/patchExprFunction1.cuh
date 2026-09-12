#pragma once
// OpenFOAM's `type expression;` PatchFunction1 -- a per-face value written as an expression over the
// registered fields -- HOST ONLY, and only the subset squareBendLiq's T walls need.
//
// provenance:
//   openfoam:
//     class:    Foam::PatchFunction1Types::PatchExprField<Type>
//     file:     src/finiteVolume/expressions/PatchFunction1/PatchFunction1Expression.C
//               :45     valueExpr_("expression", dict_) -- the `expression` entry is the expression
//               :50-55  an empty expression is fatal
//               :57     driver_.readDict(dict_): `variables` (exprDriver.C:304) and the Function1
//                       tables `functions<scalar>` / `functions<vector>` (exprDriverFunctions.C:212-213)
//               :93-105 value(x): clearVariables() re-evaluates every `variables` entry in order, then
//                       the expression, at every call -- the fields as they stand at that call
//     driver:   src/finiteVolume/expressions/patch/patchExprDriverTemplates.C
//               :108-222 getField(name): a variable of that name first, else the registered volField's
//                        boundaryField()[patchIndex] -- the patch's CURRENT value
//               :226-319 patchInternalField(name): variable first, else boundaryField()[patch].patchInternalField()
//               :398-470 patchNormalField(name): variable first, else boundaryField()[patch].snGrad() -- the
//                        patch class's own virtual, deltaCoeffs*(value - patchInternalField) on fixedValue
//     grammar:  src/finiteVolume/expressions/patch/patchExprLemonParser.lyy-m4 with
//               src/OpenFOAM/include/m4/lemon/operator-precedence.m4 (+ - at 4, * / at 3 left to right,
//               unary minus at 2 right to left) and rules-standard.m4:61 `max` = Foam::max(a, b) on fields
//     scalars:  src/OpenFOAM/primitives/ints/int/int.H MAXMIN: max(a, b) = (b < a) ? a : b, min = (a < b) ? a : b
//               src/OpenFOAM/primitives/VectorSpace/VectorSpaceI.H:464-482 mag = sqrt(x*x + y*y + z*z), left to right
//     time:     src/OpenFOAM/expressions/exprDriver/exprDriver.C:268-294 time() = Time::value(), deltaT() =
//               Time::deltaTValue(); patchExprLemonParser.lyy-m4:136-138 arg() = the value(x) argument
//     scanner:  src/finiteVolume/expressions/patch/patchExprScanner.rl:399-459 the function keywords; an
//               identifier that names a `functions<scalar>` entry becomes a Function1 call (:586-590)
//   brae:
//     reference: this file (parser and evaluator in patchExprFunction1.cu); UniformFixedValueExprPatchField
//                (fv_patch_field.cuh) owns one per patch; the rhoSimpleFoam steps evaluate it where
//                OpenFOAM's fixedEnergy::updateCoeffs evaluates T's patch (energy_boundary.cuh)
//     tests:     tests/test_patch_expr.cu, tests/rho_patch_expression_vs_openfoam.sh
//
// WHAT IS CARRIED, and nothing else: numbers, `+ - * /`, unary minus, parentheses, identifiers that are
// `variables` entries or registered fields (the patch value), internalField(x), snGrad(x), mag(vector),
// max(a, b), min(a, b), time(), deltaT(), arg(), pi(). Every other token -- another function name, a
// comparison, `?:`, `.x` component access, a `functions<scalar>` entry referenced by name, vector
// arithmetic -- is REFUSED BY NAME at parse or at the first evaluation. Never a substitute: OpenFOAM's
// own parser has ~60 more rules, and an expression brae evaluates differently from OpenFOAM would be a
// converged run at the wrong wall temperature.
//
// WHY THE HOST. The expression is data read from the case; the CUDA arm evaluates it on the host from the
// cells the patch touches (downloaded at the evaluation) and pushes the result into the device patch's
// refValue, exactly as the flowRate Function1 is evaluated on the host and pushed as frMdot.
#include "cf_types.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {

struct PatchExprSpec
{
    std::string expression;                  // the `expression` entry, verbatim (#{ #} delimiters excluded)
    std::vector<std::string> variables;      // the `variables` list, in file order, each "name = expr"
    // `functions<scalar>` / `functions<vector>`: OpenFOAM builds a Function1 per entry and lets the
    // expression call it by name. brae builds none; the NAMES are kept so a reference is refused by
    // name, and the raw TOKENS so the writer echoes the dictionary and OpenFOAM can read brae's output.
    struct FunctionDict
    {
        std::string              key;        // "functions<scalar>" or "functions<vector>"
        std::vector<std::string> names;      // its first-level entry names
        std::vector<std::string> tokens;     // everything between its braces, as read
    };
    std::vector<FunctionDict> functionDicts;
    std::string origin;                      // "<file>: patch '<name>', uniformValue" for messages
    bool empty() const { return expression.empty(); }
};

// The registry the expression reads: what OpenFOAM's driver finds through the mesh's objectRegistry.
// Each lookup returns false when no field of that name and type is registered; the evaluator then
// refuses by name, as OpenFOAM's getField fatals with "No field". The values are those of the patch the
// expression belongs to.
class PatchExprContext
{
public:
    virtual ~PatchExprContext() = default;
    virtual scalar timeValue() const = 0;   // NaN when the caller has no Time: time() then refuses
    virtual scalar deltaT() const = 0;      // NaN likewise
    virtual bool scalarPatchValue(const std::string& name, std::vector<scalar>& out) const = 0;
    virtual bool scalarPatchInternal(const std::string& name, std::vector<scalar>& out) const = 0;
    virtual bool scalarPatchSnGrad(const std::string& name, std::vector<scalar>& out) const = 0;
    virtual bool vectorPatchValue(const std::string& name, std::vector<vector>& out) const = 0;
    virtual bool vectorPatchInternal(const std::string& name, std::vector<vector>& out) const = 0;
    virtual std::string registeredNames() const = 0;   // for the refusal message
};

class PatchExprFunction1
{
public:
    struct Node;   // the parsed tree; defined in the .cu, where the parser and evaluator live
    // Parses the expression and every variable. Structural refusals (an unsupported token or function,
    // a variable without `=`, an empty expression) fire here, at read time; name refusals (a field the
    // context does not register, a `functions<>` entry referenced) fire at the first value().
    explicit PatchExprFunction1(PatchExprSpec spec);
    ~PatchExprFunction1();
    PatchExprFunction1(PatchExprFunction1&&) noexcept;
    PatchExprFunction1& operator=(PatchExprFunction1&&) noexcept;
    PatchExprFunction1(const PatchExprFunction1&) = delete;
    PatchExprFunction1& operator=(const PatchExprFunction1&) = delete;

    // PatchExprField::value(x): the variables re-evaluated in order, then the expression, per face.
    std::vector<scalar> value(scalar arg, const PatchExprContext& ctx, label patchSize) const;

    const PatchExprSpec& spec() const { return spec_; }

private:
    PatchExprSpec spec_;
    std::unique_ptr<Node> expr_;
    std::vector<std::pair<std::string, std::unique_ptr<Node>>> vars_;
};

} // namespace brae
