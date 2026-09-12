// The `expression` PatchFunction1 evaluator (patchExprFunction1.cuh) on the host, without OpenFOAM: the
// grammar's precedence and associativity as operator-precedence.m4 declares them, the field lookups in
// OpenFOAM's order (variable, then registered field), Foam::max/min's NaN side, mag() of a vector, the
// variables re-evaluated per call, and every token outside the subset refused BY NAME.
//
// WHAT THIS GATE DOES NOT CLAIM. No number here came from OpenFOAM; the expected values are hand
// computations of the same double operations. The OpenFOAM comparison is
// tests/rho_patch_expression_vs_openfoam.sh, which runs squareBendLiq's own T-wall expression on both
// mirror arms against real OpenFOAM's written wall values, iteration by iteration, and this test runs
// where that one exits 77 for want of an OpenFOAM installation.
//
// THE CONTROL for every refusal below is the arms that evaluate: a parser that threw on everything
// would pass all the refusals, and fails the first arm.
#include "patchExprFunction1.cuh"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
using namespace brae;

namespace {

// Two faces. T patch {350, 360}, T cells {300, 310}, deltaCoeffs {2, 4}; U cells {(3,4,0), (0,0,12)}.
struct Ctx : PatchExprContext
{
    scalar t = 2.0, dt = 0.5;
    scalar timeValue() const override { return t; }
    scalar deltaT() const override { return dt; }
    bool scalarPatchValue(const std::string& n, std::vector<scalar>& o) const override
    {
        if (n == "T") { o = {350.0, 360.0}; return true; }
        if (n == "p") { o = {1e5, 2e5}; return true; }
        return false;
    }
    bool scalarPatchInternal(const std::string& n, std::vector<scalar>& o) const override
    {
        if (n == "T") { o = {300.0, 310.0}; return true; }
        if (n == "p") { o = {1.5e5, 2.5e5}; return true; }
        return false;
    }
    bool scalarPatchSnGrad(const std::string& n, std::vector<scalar>& o) const override
    {
        if (n == "T") { o = {2.0 * (350.0 - 300.0), 4.0 * (360.0 - 310.0)}; return true; }
        if (n == "p") { o = {0.0, 0.0}; return true; }
        return false;
    }
    bool vectorPatchValue(const std::string& n, std::vector<vector>& o) const override
    {
        if (n == "U") { o = {vector{0, 0, 0}, vector{0, 0, 0}}; return true; }
        return false;
    }
    bool vectorPatchInternal(const std::string& n, std::vector<vector>& o) const override
    {
        if (n == "U") { o = {vector{3, 4, 0}, vector{0, 0, 12}}; return true; }
        return false;
    }
    std::string registeredNames() const override { return "U, T, p"; }
};

int fails = 0;

PatchExprSpec specFor(const std::string& expr, std::vector<std::string> vars = {},
                      std::vector<std::string> fnNames = {})
{
    PatchExprSpec s;
    s.expression = expr;
    s.variables  = std::move(vars);
    s.origin     = "test";
    if (!fnNames.empty())
    {
        PatchExprSpec::FunctionDict fd;
        fd.key   = "functions<scalar>";
        fd.names = std::move(fnNames);
        s.functionDicts.push_back(fd);
    }
    return s;
}

void expect(const char* nm, const std::string& expr, std::vector<scalar> want,
            std::vector<std::string> vars = {}, scalar arg = std::numeric_limits<scalar>::quiet_NaN())
{
    try
    {
        PatchExprFunction1 fn(specFor(expr, std::move(vars)));
        Ctx ctx;
        const std::vector<scalar> got = fn.value(arg, ctx, 2);
        bool ok = got.size() == want.size();
        for (std::size_t i = 0; ok && i < got.size(); ++i)
        {
            const bool bothNaN = std::isnan(got[i]) && std::isnan(want[i]);
            ok = bothNaN || std::fabs(got[i] - want[i]) <= 1e-13 * std::fmax(1.0, std::fabs(want[i]));
        }
        if (!ok) ++fails;
        std::printf("  %-58s got {%.15g, %.15g}  exp {%.15g, %.15g}  %s\n", nm,
                    got.size() > 0 ? got[0] : NAN, got.size() > 1 ? got[1] : NAN,
                    want.size() > 0 ? want[0] : NAN, want.size() > 1 ? want[1] : NAN, ok ? "OK" : "FAIL");
    }
    catch (const std::exception& e)
    {
        ++fails;
        std::printf("  %-58s THREW: %s  FAIL\n", nm, e.what());
    }
}

void refuses(const char* nm, const PatchExprSpec& spec, const char* mustSay, bool atParse)
{
    try
    {
        PatchExprFunction1 fn(spec);
        if (atParse)
        {
            ++fails;
            std::printf("  %-58s parsed, expected a refusal naming `%s`  FAIL\n", nm, mustSay);
            return;
        }
        Ctx ctx;
        fn.value(1.0, ctx, 2);
        ++fails;
        std::printf("  %-58s evaluated, expected a refusal naming `%s`  FAIL\n", nm, mustSay);
    }
    catch (const std::exception& e)
    {
        const bool named = std::string(e.what()).find(mustSay) != std::string::npos;
        if (!named) ++fails;
        std::printf("  %-58s %s  %s\n", nm, named ? "refused by name" : e.what(), named ? "OK" : "FAIL");
    }
}

} // namespace

int main()
{
    std::printf("-- arm 1: the tutorial's expression on the fixture, and the grammar\n");
    // squareBendLiq's T walls: par1 = |U_c|/snGrad(T); T = Tcrit + par1*T_c*max((Tcrit-T)/Tcrit*dt/t, 0)
    // face 0: |U| = 5, snGrad = 100, par1 = 0.05; (500-350)/500 = 0.3, *0.5/2 = 0.075; 500 + 0.05*300*0.075 = 501.125
    // face 1: |U| = 12, snGrad = 200, par1 = 0.06; (500-360)/500 = 0.28, *0.25 = 0.07;  500 + 0.06*310*0.07 = 501.302
    expect("squareBendLiq T walls (variables, mag, snGrad, max, dt, t)",
           "Tcrit + par1*internalField(T) * max((Tcrit-T)/(Tcrit)*deltaT()/time(),0)",
           {500.0 + 0.05 * 300.0 * (((500.0 - 350.0) / 500.0) * 0.5 / 2.0),
            500.0 + 0.06 * 310.0 * (((500.0 - 360.0) / 500.0) * 0.5 / 2.0)},
           {"Tcrit = 500", "par1 = mag(internalField(U))/snGrad(T)"});
    expect("+ - left to right: 1-2-3", "1-2-3", {-4, -4});
    expect("* / left to right: 8/2/2", "8/2/2", {2, 2});
    expect("* / bind tighter than + -: 2*3+4*5", "2*3+4*5", {26, 26});
    expect("unary minus binds tighter than *: -2*3", "-2*3", {-6, -6});
    expect("unary minus of a field: -T", "-T", {-350, -360});
    expect("parentheses: (1+2)*3", "(1+2)*3", {9, 9});
    expect("a bare number is a uniform field", "1.5e2", {150, 150});
    expect("a field name is its PATCH value", "T", {350, 360});
    expect("internalField(name) is the face-cell value", "internalField(T)", {300, 310});
    expect("snGrad(name) is the patch class's snGrad", "snGrad(T)", {100, 200});
    expect("mag of a vector is sqrt(x*x + y*y + z*z)", "mag(internalField(U))", {5, 12});
    expect("mag of a scalar", "mag(0-T)", {350, 360});
    expect("max(a, b) = (b < a) ? a : b", "max(T-355, 0)", {0, 5});
    expect("min(a, b) = (a < b) ? a : b", "min(T, 355)", {350, 355});
    expect("max(NaN, 0) is 0, as Foam::max reads it", "max(snGrad(p)/snGrad(p), 0)", {0, 0});
    expect("time() and deltaT()", "time()*10 + deltaT()", {20.5, 20.5});
    expect("arg() is value(x)'s argument", "arg()", {7, 7}, {}, 7.0);
    expect("pi()", "pi()", {3.14159265358979323846, 3.14159265358979323846});
    expect("a variable shadows a field of the same name", "T", {1, 1}, {"T = 1"});
    expect("internalField(var) returns the variable itself", "internalField(x)", {2, 2}, {"x = 2"});
    expect("variables chain in order", "b", {12, 12}, {"a = 4", "b = a*3"});
    expect("semicolon-separated inline variables", "b", {12, 12}, {"a = 4; b = a*3"});
    expect("a variable that is a field", "v*2", {700, 720}, {"v = T"});
    expect("0 is the ZERO token", "0", {0, 0});

    std::printf("-- arm 2: refusals, each by name\n");
    refuses("an empty expression", specFor("  "), "not defined", true);
    refuses("a function outside the subset: sqrt", specFor("sqrt(T)"), "sqrt", true);
    refuses("neighbourField is outside the subset", specFor("neighbourField(T)"), "neighbourField", true);
    refuses("component access U.x", specFor("internalField(U.x)"), "component access", true);
    refuses("a comparison operator", specFor("T > 300"), "`>`", true);
    refuses("the ternary operator", specFor("T ? 1 : 0"), "`?`", true);
    refuses("a functions<scalar> entry referenced by name", specFor("T*trigger", {}, {"trigger"}), "trigger", true);
    refuses("a functions<scalar> entry called", specFor("T*trigger()", {}, {"trigger"}), "trigger", true);
    refuses("a variable without =", specFor("T", {"Tcrit 500"}), "no `=`", true);
    refuses("a remote variable name{where}", specFor("T", {"x{patch} = 1"}), "remote", true);
    refuses("internalField of an expression, not a name", specFor("internalField(T+1)"), "takes a field or variable name", true);
    refuses("vector arithmetic: U*2", specFor("internalField(U)*2"), "vector", false);
    refuses("a field that is not registered", specFor("Tfoo"), "Tfoo", false);
    refuses("snGrad of a field that is not registered", specFor("snGrad(Tfoo)"), "Tfoo", false);
    refuses("snGrad of a vector", specFor("snGrad(U)"), "vector", false);
    refuses("a vector result on a scalar patch", specFor("internalField(U)"), "evaluates to a vector", false);
    {
        // time() with no Time: NaN refuses rather than reading 0.
        struct NoTime : Ctx { scalar timeValue() const override { return std::numeric_limits<scalar>::quiet_NaN(); } } ctx;
        try
        {
            PatchExprFunction1 fn(specFor("time()"));
            fn.value(1.0, ctx, 2);
            ++fails;
            std::printf("  %-58s evaluated  FAIL\n", "time() with no Time");
        }
        catch (const std::exception& e)
        {
            const bool named = std::string(e.what()).find("time()") != std::string::npos;
            if (!named) ++fails;
            std::printf("  %-58s %s  %s\n", "time() with no Time", named ? "refused by name" : e.what(), named ? "OK" : "FAIL");
        }
    }
    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
