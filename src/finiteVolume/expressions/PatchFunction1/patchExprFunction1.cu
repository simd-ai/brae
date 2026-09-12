// The parser and evaluator for the `expression` PatchFunction1 subset -- see the header for what is
// carried and the OpenFOAM lines each rule follows.
#include "patchExprFunction1.cuh"
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace brae {

struct PatchExprFunction1::Node
{
    enum Kind
    {
        Number,          // a literal (the ZERO token is the literal 0)
        Ident,           // a variable or a registered field's patch value
        Neg,             // unary minus
        Add, Sub, Mul, Div,
        Mag,             // mag(vector or scalar)
        Max, Min,        // Foam::max / Foam::min on fields
        InternalField,   // internalField(name)
        SnGrad,          // snGrad(name)
        Time, DeltaT, Arg, Pi
    };
    Kind                  kind;
    scalar                num = 0;
    std::string           name;
    std::unique_ptr<Node> a, b;
};

namespace {

using Node = PatchExprFunction1::Node;

// A value mid-evaluation. OpenFOAM promotes a scalar literal or a single-value variable to a field of
// the patch's size before any operation (rule_field_from_value), and every operation is then
// element-wise; keeping the uniform ones as one number gives the same doubles with fewer copies.
struct Val
{
    bool                isVector = false;
    bool                uniform  = true;
    scalar              s        = 0;
    std::vector<scalar> f;
    vector              v{0, 0, 0};
    std::vector<vector> vf;

    static Val scalarUniform(scalar x) { Val r; r.s = x; return r; }
    static Val scalarField(std::vector<scalar> x) { Val r; r.uniform = false; r.f = std::move(x); return r; }
    static Val vectorField(std::vector<vector> x) { Val r; r.isVector = true; r.uniform = false; r.vf = std::move(x); return r; }
    scalar at(std::size_t i) const { return uniform ? s : f[i]; }
};

struct Tok
{
    enum Kind { Num, Id, Op, End };
    Kind        kind;
    std::string text;
    scalar      num = 0;
};

[[noreturn]] void refuse(const std::string& origin, const std::string& what)
{
    throw std::runtime_error("brae: " + origin + ": " + what);
}

// The scanner. OpenFOAM's (patchExprScanner.rl) accepts a great deal more; anything outside the subset
// is named here rather than skipped.
std::vector<Tok> scan(const std::string& s, const std::string& origin)
{
    std::vector<Tok> out;
    std::size_t i = 0;
    while (i < s.size())
    {
        const char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c)))
        {
            ++i;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))
            || (c == '.' && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1]))))
        {
            std::size_t j = i;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '.')) ++j;
            if (j < s.size() && (s[j] == 'e' || s[j] == 'E'))
            {
                std::size_t k = j + 1;
                if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
                if (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k])))
                {
                    while (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) ++k;
                    j = k;
                }
            }
            Tok t;
            t.kind = Tok::Num;
            t.text = s.substr(i, j - i);
            t.num  = std::stod(t.text);
            out.push_back(t);
            i = j;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
        {
            std::size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
            Tok t;
            t.kind = Tok::Id;
            t.text = s.substr(i, j - i);
            out.push_back(t);
            i = j;
            // `U.x`: OpenFOAM's component access (rule_vector_components). Not carried.
            if (j < s.size() && s[j] == '.')
                refuse(origin, "the expression uses `" + t.text + "." + "` component access, which brae's "
                               "expression evaluator does not carry. Refusing rather than evaluating "
                               "something else.");
            continue;
        }
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '(' || c == ')' || c == ',')
        {
            Tok t;
            t.kind = Tok::Op;
            t.text = std::string(1, c);
            out.push_back(t);
            ++i;
            continue;
        }
        refuse(origin, std::string("the expression uses `") + c + "`, which brae's expression evaluator "
                       "does not carry (only + - * / ( ) , numbers, names and the functions internalField, "
                       "snGrad, mag, max, min, time, deltaT, arg, pi). Refusing rather than evaluating "
                       "something else.");
    }
    Tok e;
    e.kind = Tok::End;
    out.push_back(e);
    return out;
}

class Parser
{
public:
    Parser(std::vector<Tok> toks, const std::string& origin, const std::vector<std::string>& functionNames)
        : t_(std::move(toks)), origin_(origin), functionNames_(functionNames) {}

    std::unique_ptr<Node> parseAll()
    {
        std::unique_ptr<Node> n = additive();
        if (peek().kind != Tok::End)
            refuse(origin_, "unexpected `" + peek().text + "` in the expression.");
        return n;
    }

private:
    const Tok& peek() const { return t_[pos_]; }
    Tok next() { return t_[pos_++]; }
    bool isOp(const char* s) const { return peek().kind == Tok::Op && peek().text == s; }
    void expectOp(const char* s)
    {
        if (!isOp(s)) refuse(origin_, std::string("expected `") + s + "` in the expression, found `" + peek().text + "`.");
        next();
    }
    static std::unique_ptr<Node> mk(Node::Kind k) { auto n = std::make_unique<Node>(); n->kind = k; return n; }

    // %left PLUS MINUS (operator-precedence.m4, level 4): left to right.
    std::unique_ptr<Node> additive()
    {
        std::unique_ptr<Node> l = multiplicative();
        while (isOp("+") || isOp("-"))
        {
            const bool plus = next().text == "+";
            auto n = mk(plus ? Node::Add : Node::Sub);
            n->a = std::move(l);
            n->b = multiplicative();
            l = std::move(n);
        }
        return l;
    }
    // %left TIMES DIVIDE (level 3): left to right, tighter than + -.
    std::unique_ptr<Node> multiplicative()
    {
        std::unique_ptr<Node> l = unary();
        while (isOp("*") || isOp("/"))
        {
            const bool mul = next().text == "*";
            auto n = mk(mul ? Node::Mul : Node::Div);
            n->a = std::move(l);
            n->b = unary();
            l = std::move(n);
        }
        return l;
    }
    // %right NEGATE (level 2): binds tighter than * /, so -a*b is (-a)*b.
    std::unique_ptr<Node> unary()
    {
        if (isOp("-"))
        {
            next();
            auto n = mk(Node::Neg);
            n->a = unary();
            return n;
        }
        if (isOp("+"))
        {
            // OpenFOAM's grammar has no unary plus.
            refuse(origin_, "the expression uses a unary `+`, which OpenFOAM's grammar does not have.");
        }
        return primary();
    }
    std::unique_ptr<Node> primary()
    {
        const Tok t = next();
        if (t.kind == Tok::Num)
        {
            auto n = mk(Node::Number);
            n->num = t.num;
            return n;
        }
        if (t.kind == Tok::Op && t.text == "(")
        {
            std::unique_ptr<Node> n = additive();
            expectOp(")");
            return n;
        }
        if (t.kind == Tok::Id)
        {
            if (isOp("("))
            {
                next();
                return call(t.text);
            }
            for (const std::string& fn : functionNames_)
                if (fn == t.text)
                    refuse(origin_, "the expression references `" + fn + "`, a `functions<scalar>`/"
                                    "`functions<vector>` Function1 entry. brae builds none of those; refusing "
                                    "rather than evaluating the name as a field.");
            auto n = mk(Node::Ident);
            n->name = t.text;
            return n;
        }
        refuse(origin_, "unexpected `" + (t.kind == Tok::End ? std::string("end of expression") : t.text)
                        + "` in the expression.");
    }
    std::unique_ptr<Node> call(const std::string& fn)
    {
        auto nullary = [&](Node::Kind k)
        {
            expectOp(")");
            return mk(k);
        };
        if (fn == "time")   return nullary(Node::Time);
        if (fn == "deltaT") return nullary(Node::DeltaT);
        if (fn == "arg")    return nullary(Node::Arg);
        if (fn == "pi")     return nullary(Node::Pi);
        if (fn == "internalField" || fn == "snGrad")
        {
            // rule_get_patchfields: the argument is an IDENTIFIER (a variable or a field name), never
            // an expression -- OpenFOAM's grammar is INTERNAL_FIELD LPAREN SCALAR_ID RPAREN.
            const Tok id = next();
            if (id.kind != Tok::Id || !isOp(")"))
                refuse(origin_, fn + "(...) takes a field or variable name, found `"
                                + (id.kind == Tok::Id ? id.text + " " + peek().text : id.text) + "`.");
            expectOp(")");
            auto n = mk(fn == "internalField" ? Node::InternalField : Node::SnGrad);
            n->name = id.text;
            for (const std::string& f : functionNames_)
                if (f == id.text)
                    refuse(origin_, "the expression references `" + f + "`, a `functions<>` Function1 entry, "
                                    "inside " + fn + "(). brae builds none of those.");
            return n;
        }
        if (fn == "mag")
        {
            auto n = mk(Node::Mag);
            n->a = additive();
            expectOp(")");
            return n;
        }
        if (fn == "max" || fn == "min")
        {
            auto n = mk(fn == "max" ? Node::Max : Node::Min);
            n->a = additive();
            expectOp(",");
            n->b = additive();
            expectOp(")");
            return n;
        }
        for (const std::string& f : functionNames_)
            if (f == fn)
                refuse(origin_, "the expression calls `" + fn + "()`, a `functions<scalar>`/`functions<vector>` "
                                "Function1 entry. brae builds none of those; refusing rather than "
                                "evaluating it as something else.");
        refuse(origin_, "the expression calls `" + fn + "(...)`, which brae's expression evaluator does not "
                        "carry (it carries internalField, snGrad, mag, max, min, time, deltaT, arg and pi). "
                        "Refusing rather than evaluating something else.");
    }

    std::vector<Tok>                t_;
    std::size_t                     pos_ = 0;
    std::string                     origin_;
    const std::vector<std::string>& functionNames_;
};

std::unique_ptr<Node> parseExpr(const std::string& text, const std::string& origin,
                                const std::vector<std::string>& functionNames)
{
    Parser p(scan(text, origin), origin, functionNames);
    return p.parseAll();
}

struct Evaluator
{
    const PatchExprContext&      ctx;
    scalar                       arg;
    label                        n;
    const std::string&           origin;
    std::map<std::string, Val>   vars;

    [[noreturn]] void noField(const std::string& name, const char* what) const
    {
        refuse(origin, std::string("the expression reads `") + name + "` (" + what + "), and no variable "
                       "or field of that name is registered. Registered: " + ctx.registeredNames()
                       + ". OpenFOAM's driver fatals with \"No field\" there; refusing rather than "
                         "substituting.");
    }
    void sized(const char* what, const std::vector<scalar>& f) const
    {
        if (static_cast<label>(f.size()) != n)
            refuse(origin, std::string(what) + " returned " + std::to_string(f.size()) + " values for a patch of "
                           + std::to_string(n) + " faces.");
    }
    void sized(const char* what, const std::vector<vector>& f) const
    {
        if (static_cast<label>(f.size()) != n)
            refuse(origin, std::string(what) + " returned " + std::to_string(f.size()) + " values for a patch of "
                           + std::to_string(n) + " faces.");
    }

    Val ident(const std::string& name) const
    {
        auto it = vars.find(name);
        if (it != vars.end()) return it->second;
        std::vector<scalar> sf;
        if (ctx.scalarPatchValue(name, sf)) { sized("the patch value", sf); return Val::scalarField(std::move(sf)); }
        std::vector<vector> vf;
        if (ctx.vectorPatchValue(name, vf)) { sized("the patch value", vf); return Val::vectorField(std::move(vf)); }
        noField(name, "the patch value");
    }
    Val internalField(const std::string& name) const
    {
        // patchInternalField(name): a VARIABLE of that name is returned as it is (patchExprDriverTemplates.C:231-235).
        auto it = vars.find(name);
        if (it != vars.end()) return it->second;
        std::vector<scalar> sf;
        if (ctx.scalarPatchInternal(name, sf)) { sized("internalField()", sf); return Val::scalarField(std::move(sf)); }
        std::vector<vector> vf;
        if (ctx.vectorPatchInternal(name, vf)) { sized("internalField()", vf); return Val::vectorField(std::move(vf)); }
        noField(name, "internalField()");
    }
    Val snGrad(const std::string& name) const
    {
        auto it = vars.find(name);
        if (it != vars.end()) return it->second;   // patchNormalField: the variable itself (:403-407)
        std::vector<scalar> sf;
        if (ctx.scalarPatchSnGrad(name, sf)) { sized("snGrad()", sf); return Val::scalarField(std::move(sf)); }
        std::vector<vector> vf;
        if (ctx.vectorPatchValue(name, vf))
            refuse(origin, "snGrad(" + name + ") on a vector field is not carried by brae's expression evaluator.");
        noField(name, "snGrad()");
    }

    template <class F>
    Val binary(const Val& x, const Val& y, const char* op, F f) const
    {
        if (x.isVector || y.isVector)
            refuse(origin, std::string("the expression applies `") + op + "` to a vector, which brae's "
                           "expression evaluator does not carry (vectors reach it only through mag()).");
        if (x.uniform && y.uniform) return Val::scalarUniform(f(x.s, y.s));
        std::vector<scalar> r(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = f(x.at(i), y.at(i));
        return Val::scalarField(std::move(r));
    }

    Val eval(const Node& nd) const
    {
        switch (nd.kind)
        {
            case Node::Number: return Val::scalarUniform(nd.num);
            case Node::Ident:  return ident(nd.name);
            case Node::InternalField: return internalField(nd.name);
            case Node::SnGrad: return snGrad(nd.name);
            case Node::Time:
            {
                const scalar t = ctx.timeValue();
                if (std::isnan(t)) refuse(origin, "the expression calls time(), and this caller has no Time.");
                return Val::scalarUniform(t);
            }
            case Node::DeltaT:
            {
                const scalar dt = ctx.deltaT();
                if (std::isnan(dt)) refuse(origin, "the expression calls deltaT(), and this caller has no Time.");
                return Val::scalarUniform(dt);
            }
            case Node::Arg:
                if (std::isnan(arg)) refuse(origin, "the expression calls arg(), and this caller passed none.");
                return Val::scalarUniform(arg);
            case Node::Pi: return Val::scalarUniform(scalar(3.14159265358979323846264338327950288));
            case Node::Neg:
            {
                Val x = eval(*nd.a);
                if (x.isVector)
                    refuse(origin, "the expression negates a vector, which brae's expression evaluator does not carry.");
                if (x.uniform) return Val::scalarUniform(-x.s);
                for (scalar& v : x.f) v = -v;
                return x;
            }
            case Node::Add: return binary(eval(*nd.a), eval(*nd.b), "+", [](scalar p, scalar q) { return p + q; });
            case Node::Sub: return binary(eval(*nd.a), eval(*nd.b), "-", [](scalar p, scalar q) { return p - q; });
            case Node::Mul: return binary(eval(*nd.a), eval(*nd.b), "*", [](scalar p, scalar q) { return p * q; });
            case Node::Div: return binary(eval(*nd.a), eval(*nd.b), "/", [](scalar p, scalar q) { return p / q; });
            // Foam::max(a, b) = (b < a) ? a : b and min = (a < b) ? a : b (int.H MAXMIN), so a NaN in a
            // returns b. Transcribed, not std::max, for that reason.
            case Node::Max: return binary(eval(*nd.a), eval(*nd.b), "max", [](scalar p, scalar q) { return (q < p) ? p : q; });
            case Node::Min: return binary(eval(*nd.a), eval(*nd.b), "min", [](scalar p, scalar q) { return (p < q) ? p : q; });
            case Node::Mag:
            {
                Val x = eval(*nd.a);
                if (x.isVector)
                {
                    // VectorSpaceI.H:464-482: magSqr summed component by component from x, then sqrt.
                    std::vector<scalar> r(x.vf.size());
                    for (std::size_t i = 0; i < r.size(); ++i)
                    {
                        const vector& v = x.vf[i];
                        scalar ms = v.x * v.x;
                        ms += v.y * v.y;
                        ms += v.z * v.z;
                        r[i] = std::sqrt(ms);
                    }
                    return Val::scalarField(std::move(r));
                }
                if (x.uniform) return Val::scalarUniform(std::fabs(x.s));
                for (scalar& v : x.f) v = std::fabs(v);
                return x;
            }
        }
        refuse(origin, "internal: unknown expression node.");
    }
};

// exprDriver::addVariables (exprDriver.C:378-472): each entry is `name = expr`; `name{where} = expr`
// (a remote evaluation) is refused rather than evaluated here.
std::pair<std::string, std::string> splitVariable(const std::string& entry, const std::string& origin)
{
    const std::size_t eq = entry.find('=');
    if (eq == std::string::npos)
        refuse(origin, "variables entry \"" + entry + "\" has no `=` (OpenFOAM fatals there too).");
    auto trim = [](std::string s)
    {
        const std::size_t a = s.find_first_not_of(" \t\r\n");
        const std::size_t b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    };
    const std::string lhs = trim(entry.substr(0, eq));
    const std::string rhs = trim(entry.substr(eq + 1));
    if (lhs.find('{') != std::string::npos)
        refuse(origin, "variables entry \"" + entry + "\" names a remote location (`name{where}`), which brae's "
                       "expression evaluator does not carry.");
    if (lhs.empty() || rhs.empty())
        refuse(origin, "variables entry \"" + entry + "\" has an empty side.");
    for (const char c : lhs)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
            refuse(origin, "variables entry \"" + entry + "\" has a name OpenFOAM's word::validate would alter.");
    return {lhs, rhs};
}

} // namespace

PatchExprFunction1::PatchExprFunction1(PatchExprSpec spec)
    : spec_(std::move(spec))
{
    if (spec_.expression.find_first_not_of(" \t\r\n") == std::string::npos)
        throw std::runtime_error("brae: " + spec_.origin + ": the expression was not defined "
                                 "(PatchFunction1Expression.C:50-55 fatals there too).");
    std::vector<std::string> fnNames;
    for (const auto& fd : spec_.functionDicts)
        for (const std::string& nm : fd.names) fnNames.push_back(nm);
    // A semicolon-separated inline list is also accepted by OpenFOAM (exprDriver.C:389-391).
    for (const std::string& entry : spec_.variables)
    {
        std::size_t start = 0;
        while (start <= entry.size())
        {
            const std::size_t semi = entry.find(';', start);
            const std::string one = entry.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
            if (one.find_first_not_of(" \t\r\n") != std::string::npos)
            {
                auto [name, rhs] = splitVariable(one, spec_.origin);
                vars_.emplace_back(name, parseExpr(rhs, spec_.origin + ", variable `" + name + "`", fnNames));
            }
            if (semi == std::string::npos) break;
            start = semi + 1;
        }
    }
    expr_ = parseExpr(spec_.expression, spec_.origin, fnNames);
}

PatchExprFunction1::~PatchExprFunction1() = default;
PatchExprFunction1::PatchExprFunction1(PatchExprFunction1&&) noexcept = default;
PatchExprFunction1& PatchExprFunction1::operator=(PatchExprFunction1&&) noexcept = default;

std::vector<scalar> PatchExprFunction1::value(scalar arg, const PatchExprContext& ctx, label patchSize) const
{
    Evaluator ev{ctx, arg, patchSize, spec_.origin, {}};
    // clearVariables(): every variable re-evaluated, in order, against the fields as they stand now.
    for (const auto& [name, node] : vars_) ev.vars[name] = ev.eval(*node);
    const Val r = ev.eval(*expr_);
    if (r.isVector)
        throw std::runtime_error("brae: " + spec_.origin + ": the expression evaluates to a vector on a scalar patch.");
    if (r.uniform) return std::vector<scalar>(static_cast<std::size_t>(patchSize), r.s);
    return r.f;
}

} // namespace brae
