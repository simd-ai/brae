#pragma once
// inter_function_objects.cuh -- the runtime function objects of interFoam's controlDict `functions { }`.
//
// brae ran none of them: the writer said so in one notice and the case's postProcessing/ never appeared.
// This file runs the subset a refill/slosh study reads its answers from, with OpenFOAM's own semantics and
// OpenFOAM's own file layout, so the usual post-processing scripts read brae's output unchanged:
//
//   volFieldValue      regionType all | cellZone,  operation volIntegrate | volAverage | sum | min | max
//   surfaceFieldValue  regionType patch,           operation areaAverage | areaIntegrate | sum | min | max
//                      (the phase fraction's patch values; no other field has patch values on the host)
//   probes             fixedLocations, cell values (interpolationScheme cell) of the phase fraction, U, p_rgh
//   fieldMinMax        mode magnitude | component, of U and of any scalar field
//   exprField          action new, a scalar expression of the phase fraction, p_rgh, U (.x() .y() .z(),
//                      mag, magSqr), pos().x()/y()/z(), other expression fields and numbers
//   runTimeControl     conditions of type minMax on another object's result, satisfiedAction end
//
// WHEN THEY RUN. At the end of each time step, in the controlDict's order -- OpenFOAM runs them at the top
// of the next Time::run(), on the same state. An object's `writeControl` (timeStep | runTime |
// adjustableRunTime | writeTime | none) and `writeInterval` say when it computes and writes; there is no
// output at the start time, as in OpenFOAM (the execution index starts at 0). runTimeControl evaluates every
// step. `enabled false` removes an object, as in OpenFOAM, and `foamDictionary -entry
// functions.<name>.enabled -set false` between two runs therefore works as it does there.
//
// WHERE THE STATE COMES FROM. The driver hands over lazy host views (Source): a loop that keeps its fields on
// the device downloads a field the first time an object asks for it in a step, and not at all in a step in
// which nothing is due. Expression fields are evaluated on demand, from those views, once per step. A device
// loop may instead answer all due volIntegrate requests in one batch from a compiled expression tree, with
// deterministic GPU reductions. Products/sums/quotients, unary minus, the phase fraction, U components,
// pos().x()/y()/z(), magSqr(U), constants and cellZone indices are supported there. The device tree's sum has a
// different rounding order from the serial host loop; any other expression, a missing answer, or the CPU loop
// uses this original host evaluator. runTimeControl reads the same result entry after the batch, as before.
//
// UNDER -parallel every rank runs the list on ITS cells and faces, and the objects answer as ONE OpenFOAM run
// would (the rules are OpenFOAM v2606's, read from its source):
//   volFieldValue      the rank's sum over its cells (the region's cells in zone order) is reduced over the ranks
//                      (gSum: rank order); min/max are gMin/gMax; `# Cells` and `# Volume` are the global ones
//   surfaceFieldValue  the patch's faces on each rank (a rank may hold none), areas and sums reduced
//   probes             the lowest rank whose cells hold the point owns the probe (listCombineReduce with
//                      isNotEqOp keeps the first set value in rank order); a point in no rank's cell is skipped;
//                      the master writes the columns
//   fieldMinMax        each rank's extremum (the first cell on a tie), then the first rank with a strictly better
//                      value wins -- value, location and the `processor` column come from that rank
//   runTimeControl     reads the other objects' reduced results, so every rank decides the same
//   files              postProcessing/ is written by the master alone, in the single-GPU format
//
// WHAT IS NOT HERE, said rather than guessed: an entry of a ported type that asks for something unported
// (writeFields, weightField, a vector field in volFieldValue, interpolationScheme cellPoint, exprField
// autowrite, a minMax mode other than maximum/minimum ...) stops the run at start-up and names itself. An
// object of a type that is not ported at all (surfaces, forces, fieldAverage ...) is noticed and left alone,
// as before.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "brae_notice.cuh"
#include "proc_globals.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace fo {

// ---------------------------------------------------------------------------------------------------------------
// what an object may read, per step
// ---------------------------------------------------------------------------------------------------------------

struct Step
{
    label  timeIndex = 0;        // the absolute index, as Time::timeIndex() (a restart continues it)
    scalar time = 0;
    scalar deltaT = 0;
    scalar sinceStart = 0;       // time - startTime
    bool   runTimeWrite = false; // the run writes its fields at this step (writeControl writeTime)
};


// A device-resident field is named by a member offset in DeviceArrays. Keeping this vocabulary free of CUDA
// types lets the host-only unit test exercise a device Source. `cells == nullptr` means regionType all;
// otherwise the reduction visits exactly the cellZone's indices, in dictionary order.
struct DeviceArrays
{
    const scalar* alpha = nullptr;
    const scalar* ux = nullptr;
    const scalar* uy = nullptr;
    const scalar* uz = nullptr;
    const scalar* V = nullptr;
    const scalar* Cx = nullptr;
    const scalar* Cy = nullptr;
    const scalar* Cz = nullptr;
    label         nCells = 0;
};


// The IR is also the Source-independent description of what the device reduction can evaluate.
struct DeviceExpr
{
    enum Kind { Constant, Field, Unary, Binary, MagSqrU } kind = Constant;
    enum Leaf { Alpha, Ux, Uy, Uz, PosX, PosY, PosZ } leaf = Alpha;
    char        op = 0;
    scalar      constant = 0;
    int         left = -1, right = -1, third = -1;
};

struct DeviceIntegral
{
    std::string             phaseField;   // the case's phase name, so alpha.water is not hard-coded
    std::string             expression;   // the original field name, for diagnostics
    std::vector<DeviceExpr> nodes;        // used when no prepareDeviceIntegral callback is supplied
    std::shared_ptr<void>   plan;         // opaque immutable state made once by prepareDeviceIntegral
    DeviceArrays            fields;
    const label*            cells = nullptr;
    label                   n = 0;
    std::vector<scalar>*    result = nullptr; // one entry per request, stable until the next step
};

using DeviceIntegrals = std::function<bool(std::vector<DeviceIntegral>&)>;
using PrepareDeviceIntegral = std::function<std::shared_ptr<void>(
    const std::string&, const std::vector<DeviceExpr>&, const label*, label)>;

// Host views of the solution at the step just taken. Every getter may download on first use; `newStep` is
// called once at the start of each execute() so the driver can drop what it cached.
struct Source
{
    std::function<void()>                                       newStep;
    std::function<const std::vector<scalar>&()>                 alpha;
    std::function<const std::vector<scalar>&()>                 p_rgh;
    std::function<const std::vector<scalar>&()>                 Ux;
    std::function<const std::vector<scalar>&()>                 Uy;
    std::function<const std::vector<scalar>&()>                 Uz;
    std::function<const std::vector<scalar>&(std::size_t)>      alphaPatch;   // the phase fraction's patch values
    // Optional. All due requests are submitted together; false, or an empty result entry, selects the host
    // path for every request left unanswered. A Source must not answer a request it cannot evaluate exactly.
    DeviceIntegrals                                             deviceIntegrals;
    std::function<DeviceArrays()>                               deviceFields;
    PrepareDeviceIntegral                                       prepareDeviceIntegral;
};
struct DeviceValue { bool vec = false; int s = -1, x = -1, y = -1, z = -1; };

struct DeviceScope
{
    std::string                                       phaseField;
    std::vector<DeviceExpr>*                          nodes = nullptr;
    std::function<bool(const std::string&, int&)>      scalarField;
};

inline int deviceConstant(DeviceScope& sc, scalar v)
{
    sc.nodes->push_back(DeviceExpr{});
    sc.nodes->back().constant = v;
    return static_cast<int>(sc.nodes->size() - 1);
}

inline int deviceLeaf(DeviceScope& sc, DeviceExpr::Leaf leaf)
{
    DeviceExpr e;
    e.kind = DeviceExpr::Field;
    e.leaf = leaf;
    sc.nodes->push_back(e);
    return static_cast<int>(sc.nodes->size() - 1);
}

inline int deviceNode(DeviceScope& sc, DeviceExpr::Kind kind, char op, int a, int b = -1)
{
    DeviceExpr e;
    e.kind = kind;
    e.op = op;
    e.left = a;
    e.right = b;
    sc.nodes->push_back(e);
    return static_cast<int>(sc.nodes->size() - 1);
}

struct Services
{
    std::string                                        caseDir;
    std::string                                        startName;     // postProcessing/<object>/<startName>/
    int                                                precision = 6; // controlDict writePrecision
    std::string                                        phaseField = "alpha.water";
    const PrimitiveMesh*                               mesh = nullptr;
    const FvGeometry*                                  geom = nullptr;
    const std::vector<FvPatch>*                        patches = nullptr;
    const std::map<std::string, std::vector<label>>*   cellZones = nullptr;
    Source                                             src;
    // false on every rank but 0 under -parallel: OpenFOAM writes postProcessing/ from the master alone
    bool                                               isMaster = true;
    // The reduction/gather layer over the ranks (pg:: over ProcComm). The default is a single rank: every
    // function is the identity and communicates nothing. createFunctionObjects installs the ProcComm one when
    // the process has a multi-rank communicator.
    struct Reduce
    {
        int  rank = 0;
        int  size = 1;
        std::function<void(double*, int)>                                          sum, min, max;
        std::function<std::vector<std::vector<double>>(const std::vector<double>&)> gatherAll;
        std::function<void(const std::string&)>                                    fail;   // collective error
        bool parallel() const { return size > 1; }
        bool master() const { return rank == 0; }
        // every function below is collective when parallel() and the identity otherwise
        scalar reduceSum(scalar v) const { if (parallel()) { double d = v; sum(&d, 1); return d; } return v; }
        scalar reduceMin(scalar v) const { if (parallel()) { double d = v; min(&d, 1); return d; } return v; }
        scalar reduceMax(scalar v) const { if (parallel()) { double d = v; max(&d, 1); return d; } return v; }
        void   reduceSum(std::vector<double>& v) const { if (parallel() && !v.empty()) sum(v.data(), static_cast<int>(v.size())); }
        void   reduceMin(std::vector<double>& v) const { if (parallel() && !v.empty()) min(v.data(), static_cast<int>(v.size())); }
        void   reduceMax(std::vector<double>& v) const { if (parallel() && !v.empty()) max(v.data(), static_cast<int>(v.size())); }
        // every rank's list, on every rank, in rank order (the master writes from it)
        std::vector<std::vector<double>> gatherToMaster(const std::vector<double>& local) const
        {
            return parallel() ? gatherAll(local) : std::vector<std::vector<double>>(1, local);
        }
        // a condition only some ranks can see: every rank stops with the message
        [[noreturn]] void abortAll(const std::string& msg) const
        {
            if (parallel() && fail) fail(msg);
            throw std::runtime_error(msg);
        }
        static Reduce procComm()
        {
            Reduce r;
            if (!pg::parallel()) return r;
            r.rank = pg::rank();
            r.size = pg::nProcs();
            r.sum = [](double* v, int n) { pg::sum(v, n); };
            r.min = [](double* v, int n) { pg::min(v, n); };
            r.max = [](double* v, int n) { pg::max(v, n); };
            r.gatherAll = [](const std::vector<double>& v) { return pg::gatherAll(v); };
            r.fail = [](const std::string& m) { pg::fail(m); };
            return r;
        }
    }                                                  red;
};

// ---------------------------------------------------------------------------------------------------------------
// scalar / vector field values and the expression evaluator
// ---------------------------------------------------------------------------------------------------------------

// A scalar field that is a constant, a borrowed host vector or an owned one.
struct SF
{
    scalar                               c = 0;
    const std::vector<scalar>*           ref = nullptr;
    std::shared_ptr<std::vector<scalar>> own;
    bool isConst() const { return !ref && !own; }
    const std::vector<scalar>& v() const { return own ? *own : *ref; }
};

struct VF { SF x, y, z; };

struct ExprVal
{
    bool vec = false;
    SF   s;
    VF   v;
};

inline SF sfConst(scalar c) { SF r; r.c = c; return r; }
inline SF sfBorrow(const std::vector<scalar>& v) { SF r; r.ref = &v; return r; }

inline SF sfBinary(const SF& a, const SF& b, char op, std::size_t n)
{
    auto apply = [op](scalar x, scalar y) -> scalar
    {
        switch (op)
        {
            case '+': return x + y;
            case '-': return x - y;
            case '*': return x*y;
            default:  return x/y;
        }
    };
    if (a.isConst() && b.isConst())
    {
        return sfConst(apply(a.c, b.c));
    }
    auto out = std::make_shared<std::vector<scalar>>(n);
    const scalar* pa = a.isConst() ? &a.c : a.v().data();
    const scalar* pb = b.isConst() ? &b.c : b.v().data();
    const std::size_t sa = a.isConst() ? 0 : 1;
    const std::size_t sb = b.isConst() ? 0 : 1;
    scalar* po = out->data();
    switch (op)
    {
        case '+': for (std::size_t i = 0; i < n; ++i) po[i] = pa[i*sa] + pb[i*sb]; break;
        case '-': for (std::size_t i = 0; i < n; ++i) po[i] = pa[i*sa] - pb[i*sb]; break;
        case '*': for (std::size_t i = 0; i < n; ++i) po[i] = pa[i*sa]*pb[i*sb];   break;
        default:  for (std::size_t i = 0; i < n; ++i) po[i] = pa[i*sa]/pb[i*sb];   break;
    }
    SF r;
    r.own = out;
    return r;
}

template <class F>
inline SF sfMap(const SF& a, F f, std::size_t n)
{
    if (a.isConst())
    {
        return sfConst(f(a.c));
    }
    auto out = std::make_shared<std::vector<scalar>>(n);
    const scalar* pa = a.v().data();
    for (std::size_t i = 0; i < n; ++i)
    {
        (*out)[i] = f(pa[i]);
    }
    SF r;
    r.own = out;
    return r;
}

// What the expression evaluator may name.
struct Scope
{
    std::size_t                                          n = 0;
    std::function<bool(const std::string&, SF&)>         scalarField;
    std::function<bool(const std::string&, VF&)>         vectorField;
    std::function<VF()>                                  pos;
};

struct Tok
{
    enum Kind { Num, Id, Op, End } k = End;
    std::string t;
    scalar      num = 0;
};

inline std::vector<Tok> tokenizeExpression(const std::string& s)
{
    std::vector<Tok> out;
    std::size_t i = 0;
    while (i < s.size())
    {
        const char ch = s[i];
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            ++i;
            continue;
        }
        const bool afterClose = !out.empty() && out.back().k == Tok::Op && out.back().t == ")";
        if (std::isdigit(static_cast<unsigned char>(ch))
         || (ch == '.' && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1])) && !afterClose))
        {
            char* end = nullptr;
            const scalar v = std::strtod(s.c_str() + i, &end);
            Tok t;
            t.k = Tok::Num;
            t.num = v;
            t.t = s.substr(i, static_cast<std::size_t>(end - (s.c_str() + i)));
            i = static_cast<std::size_t>(end - s.c_str());
            out.push_back(t);
        }
        else if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_')
        {
            std::size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_'))
            {
                ++j;
            }
            Tok t;
            t.k = Tok::Id;
            t.t = s.substr(i, j - i);
            out.push_back(t);
            i = j;
        }
        else if (std::strchr("+-*/(),.", ch))
        {
            Tok t;
            t.k = Tok::Op;
            t.t = std::string(1, ch);
            out.push_back(t);
            ++i;
        }
        else
        {
            throw std::runtime_error(std::string("brae function objects: the expression `") + s
                                     + "` contains `" + ch + "`, which the expression evaluator does not read");
        }
    }
    out.push_back(Tok());   // End
    return out;
}

// Compile the device-safe subset with the same tokenisation, precedence and dot/component rules as ExprParser.
// Every unsupported construct throws before a request reaches the Source; this is an optimisation gate, not a
// replacement for ExprParser's full validation and host evaluation.
class DeviceExprParser
{
public:
    DeviceExprParser(const std::string& text, DeviceScope& sc) : text_(text), t_(tokenizeExpression(text)), sc_(sc) {}

    int parse()
    {
        const DeviceValue v = expr();
        if (peek().k != Tok::End || v.vec) fail("not a supported scalar expression");
        return v.s;
    }

private:
    std::string        text_;
    std::vector<Tok>   t_;
    std::size_t        p_ = 0;
    DeviceScope&       sc_;

    [[noreturn]] void fail(const std::string& why) const
    {
        throw std::runtime_error("brae function objects: device expression `" + text_ + "`: " + why);
    }
    const Tok& peek(std::size_t o = 0) const { return t_[std::min(p_ + o, t_.size() - 1)]; }
    bool isOp(const char* s, std::size_t o = 0) const { return peek(o).k == Tok::Op && peek(o).t == s; }
    void expectOp(const char* s) { if (!isOp(s)) fail(std::string("expected `") + s + "`"); ++p_; }
    int scalar(const DeviceValue& v) const { if (v.vec) fail("vector arithmetic"); return v.s; }
    static int componentIndex(const std::string& s) { return s == "x" ? 0 : s == "y" ? 1 : s == "z" ? 2 : -1; }

    DeviceValue expr()
    {
        DeviceValue l = term();
        while (isOp("+") || isOp("-"))
        {
            const char op = t_[p_++].t[0];
            const DeviceValue r = term();
            l.s = deviceNode(sc_, DeviceExpr::Binary, op, scalar(l), scalar(r));
            l.vec = false;
        }
        return l;
    }
    DeviceValue term()
    {
        DeviceValue l = unary();
        while (isOp("*") || isOp("/"))
        {
            const char op = t_[p_++].t[0];
            const DeviceValue r = unary();
            l.s = deviceNode(sc_, DeviceExpr::Binary, op, scalar(l), scalar(r));
            l.vec = false;
        }
        return l;
    }
    DeviceValue unary()
    {
        if (isOp("-"))
        {
            ++p_;
            DeviceValue v;
            v.s = deviceNode(sc_, DeviceExpr::Unary, '-', scalar(unary()));
            return v;
        }
        if (isOp("+")) { ++p_; return unary(); }
        return primary();
    }
    DeviceValue postfix(DeviceValue v)
    {
        while (isOp(".") && peek(1).k == Tok::Id && componentIndex(peek(1).t) >= 0
            && isOp("(", 2) && isOp(")", 3))
        {
            if (!v.vec) fail("component of a scalar");
            const int c = componentIndex(peek(1).t);
            p_ += 4;
            DeviceValue s;
            s.s = c == 0 ? v.x : c == 1 ? v.y : v.z;
            v = s;
        }
        return v;
    }
    DeviceValue call(const std::string& name, const std::vector<DeviceValue>& args)
    {
        if (name == "pos" && args.empty())
        {
            DeviceValue v;
            v.vec = true;
            v.x = deviceLeaf(sc_, DeviceExpr::PosX);
            v.y = deviceLeaf(sc_, DeviceExpr::PosY);
            v.z = deviceLeaf(sc_, DeviceExpr::PosZ);
            return v;
        }
        if (name == "magSqr" && args.size() == 1 && args[0].vec)
        {
            DeviceValue v;
            v.s = deviceNode(sc_, DeviceExpr::MagSqrU, 0, args[0].x, args[0].y);
            sc_.nodes->back().third = args[0].z;
            return v;
        }
        fail("only pos(), U components and magSqr(U) are supported on the device");
    }
    DeviceValue primary()
    {
        const Tok tk = peek();
        if (tk.k == Tok::Num) { ++p_; DeviceValue v; v.s = deviceConstant(sc_, tk.num); return v; }
        if (isOp("(")) { ++p_; DeviceValue v = expr(); expectOp(")"); return postfix(v); }
        if (tk.k != Tok::Id) fail(tk.k == Tok::End ? "ends early" : "unexpected `" + tk.t + "`");
        std::vector<std::string> parts{tk.t};
        ++p_;
        while (isOp(".") && peek(1).k == Tok::Id && !(componentIndex(peek(1).t) >= 0 && isOp("(", 2)))
        {
            parts.push_back(peek(1).t);
            p_ += 2;
        }
        if (parts.size() == 1 && isOp("("))
        {
            ++p_;
            std::vector<DeviceValue> args;
            if (!isOp(")"))
            {
                args.push_back(expr());
                while (isOp(",")) { ++p_; args.push_back(expr()); }
            }
            expectOp(")");
            return postfix(call(parts[0], args));
        }
        std::string name = parts[0];
        for (std::size_t i = 1; i < parts.size(); ++i) name += "." + parts[i];
        int node = -1;
        if (sc_.scalarField(name, node)) { DeviceValue v; v.s = node; return v; }
        if (name == "U")
        {
            DeviceValue v;
            v.vec = true;
            v.x = deviceLeaf(sc_, DeviceExpr::Ux);
            v.y = deviceLeaf(sc_, DeviceExpr::Uy);
            v.z = deviceLeaf(sc_, DeviceExpr::Uz);
            return postfix(v);
        }
        fail("unknown field `" + name + "`");
    }
};

class ExprParser
{
public:
    ExprParser(const std::string& text, Scope& sc)
      : text_(text), t_(tokenizeExpression(text)), sc_(sc)
    {}

    SF parseScalar()
    {
        ExprVal v = expr();
        if (peek().k != Tok::End)
        {
            fail("unexpected `" + peek().t + "`");
        }
        if (v.vec)
        {
            fail("it is a vector; an exprField here is a scalar field");
        }
        return v.s;
    }

private:
    std::string       text_;
    std::vector<Tok>  t_;
    std::size_t       p_ = 0;
    Scope&            sc_;

    [[noreturn]] void fail(const std::string& why) const
    {
        throw std::runtime_error("brae function objects: expression `" + text_ + "`: " + why);
    }
    const Tok& peek(std::size_t o = 0) const { return t_[std::min(p_ + o, t_.size() - 1)]; }
    bool isOp(const char* s, std::size_t o = 0) const { return peek(o).k == Tok::Op && peek(o).t == s; }
    void expectOp(const char* s)
    {
        if (!isOp(s))
        {
            fail(std::string("expected `") + s + "`");
        }
        ++p_;
    }

    ExprVal scalarVal(const SF& s) const { ExprVal v; v.s = s; return v; }
    const SF& needScalar(const ExprVal& v) const
    {
        if (v.vec)
        {
            fail("vector arithmetic is not supported; use .x() .y() .z(), mag() or magSqr()");
        }
        return v.s;
    }

    ExprVal expr()
    {
        ExprVal l = term();
        while (isOp("+") || isOp("-"))
        {
            const char op = t_[p_++].t[0];
            ExprVal r = term();
            l = scalarVal(sfBinary(needScalar(l), needScalar(r), op, sc_.n));
        }
        return l;
    }
    ExprVal term()
    {
        ExprVal l = unary();
        while (isOp("*") || isOp("/"))
        {
            const char op = t_[p_++].t[0];
            ExprVal r = unary();
            l = scalarVal(sfBinary(needScalar(l), needScalar(r), op, sc_.n));
        }
        return l;
    }
    ExprVal unary()
    {
        if (isOp("-"))
        {
            ++p_;
            ExprVal v = unary();
            return scalarVal(sfMap(needScalar(v), [](scalar x) { return -x; }, sc_.n));
        }
        if (isOp("+"))
        {
            ++p_;
            return unary();
        }
        return primary();
    }

    static int componentIndex(const std::string& s)
    {
        return s == "x" ? 0 : s == "y" ? 1 : s == "z" ? 2 : -1;
    }
    ExprVal component(const ExprVal& v, int c) const
    {
        if (!v.vec)
        {
            fail("a component of a scalar");
        }
        return scalarVal(c == 0 ? v.v.x : c == 1 ? v.v.y : v.v.z);
    }
    ExprVal postfix(ExprVal v)
    {
        while (isOp(".") && peek(1).k == Tok::Id && componentIndex(peek(1).t) >= 0 && isOp("(", 2) && isOp(")", 3))
        {
            const int c = componentIndex(peek(1).t);
            p_ += 4;
            v = component(v, c);
        }
        return v;
    }

    ExprVal callFunction(const std::string& name, const std::vector<ExprVal>& args)
    {
        const std::size_t n = sc_.n;
        auto sq = [](scalar x) { return x*x; };
        if (name == "pos")
        {
            if (!args.empty())
            {
                fail("pos() takes no argument");
            }
            ExprVal v;
            v.vec = true;
            v.v = sc_.pos();
            return v;
        }
        if (args.size() != 1)
        {
            fail("function `" + name + "` is not supported here (mag, magSqr, sqr, sqrt and pos() are)");
        }
        const ExprVal& a = args[0];
        if (name == "magSqr" || name == "mag")
        {
            SF s;
            if (a.vec)
            {
                s = sfBinary(sfBinary(sfMap(a.v.x, sq, n), sfMap(a.v.y, sq, n), '+', n), sfMap(a.v.z, sq, n), '+', n);
            }
            else
            {
                s = sfMap(a.s, sq, n);
            }
            return scalarVal(name == "mag" ? sfMap(s, [](scalar x) { return std::sqrt(x); }, n) : s);
        }
        if (name == "sqr")
        {
            return scalarVal(sfMap(needScalar(a), sq, n));
        }
        if (name == "sqrt")
        {
            return scalarVal(sfMap(needScalar(a), [](scalar x) { return std::sqrt(x); }, n));
        }
        fail("function `" + name + "` is not supported here (mag, magSqr, sqr, sqrt and pos() are)");
    }

    ExprVal primary()
    {
        const Tok tk = peek();
        if (tk.k == Tok::Num)
        {
            ++p_;
            return scalarVal(sfConst(tk.num));
        }
        if (isOp("("))
        {
            ++p_;
            ExprVal v = expr();
            expectOp(")");
            return postfix(v);
        }
        if (tk.k != Tok::Id)
        {
            fail(tk.k == Tok::End ? "it ends early" : "unexpected `" + tk.t + "`");
        }
        std::vector<std::string> parts;
        parts.push_back(tk.t);
        ++p_;
        while (isOp(".") && peek(1).k == Tok::Id && !(componentIndex(peek(1).t) >= 0 && isOp("(", 2)))
        {
            parts.push_back(peek(1).t);
            p_ += 2;
        }
        if (parts.size() == 1 && isOp("("))
        {
            ++p_;
            std::vector<ExprVal> args;
            if (!isOp(")"))
            {
                args.push_back(expr());
                while (isOp(","))
                {
                    ++p_;
                    args.push_back(expr());
                }
            }
            expectOp(")");
            return postfix(callFunction(parts[0], args));
        }
        std::string name = parts[0];
        for (std::size_t i = 1; i < parts.size(); ++i)
        {
            name += "." + parts[i];
        }
        SF s;
        if (sc_.scalarField(name, s))
        {
            return scalarVal(s);
        }
        VF vf;
        if (sc_.vectorField(name, vf))
        {
            ExprVal v;
            v.vec = true;
            v.v = vf;
            return postfix(v);
        }
        fail("`" + name + "` is neither a field nor a function here");
    }
};

// ---------------------------------------------------------------------------------------------------------------
// controls and files
// ---------------------------------------------------------------------------------------------------------------

// Foam::timeControl::execute() for the write cadences brae supports.
struct Control
{
    enum Kind { TimeStep, RunTime, Adjustable, WriteTime, None } kind = TimeStep;
    scalar interval = 1;
    label  executionIndex = 0;

    static Control read(const FoamDict& d, const std::string& object, const std::string& prefix)
    {
        Control c;
        const std::string w = d.wordOr(prefix + "Control", "timeStep");
        if (w == "timeStep")             c.kind = TimeStep;
        else if (w == "runTime")         c.kind = RunTime;
        else if (w == "adjustableRunTime") c.kind = Adjustable;
        else if (w == "writeTime" || w == "outputTime") c.kind = WriteTime;
        else if (w == "none")            c.kind = None;
        else
        {
            throw std::runtime_error("brae function objects: `" + object + "` has " + prefix + "Control " + w
                                     + "; brae has timeStep, runTime, adjustableRunTime, writeTime and none");
        }
        c.interval = d.scalarOr(prefix + "Interval", 1);
        if ((c.kind == TimeStep || c.kind == RunTime || c.kind == Adjustable) && !(c.interval > 0))
        {
            throw std::runtime_error("brae function objects: `" + object + "` has " + prefix + "Interval "
                                     + std::to_string(c.interval) + ", which is not positive");
        }
        return c;
    }

    bool due(const Step& s)
    {
        switch (kind)
        {
            case TimeStep:
            {
                const label n = std::max<label>(1, static_cast<label>(interval));
                return (s.timeIndex % n) == 0;
            }
            case RunTime:
            case Adjustable:
            {
                const label idx = static_cast<label>((s.sinceStart + scalar(0.5)*s.deltaT)/interval);
                if (idx > executionIndex)
                {
                    executionIndex = idx;
                    return true;
                }
                return false;
            }
            case WriteTime:
                return s.runTimeWrite;
            default:
                return false;
        }
    }
};

inline std::string fmtNum(scalar v, int precision)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", precision, v);
    return buf;
}

// postProcessing/<object>/<startTime>/<file>, created with its first row, flushed after every row so a
// running case can be read by a script while it runs.
class DatFile
{
public:
    DatFile() = default;
    DatFile(std::string dir, std::string file, bool enabled = true)
      : dir_(std::move(dir)), file_(std::move(file)), enabled_(enabled) {}

    void header(const std::string& line) { head_.push_back(line); }
    void row(const std::string& line)
    {
        if (!enabled_) return;   // a rank that is not the master keeps no file
        if (!os_.is_open())
        {
            std::filesystem::create_directories(dir_);
            os_.open(dir_ + "/" + file_, std::ios::out | std::ios::trunc);
            if (!os_)
            {
                throw std::runtime_error("brae function objects: cannot write " + dir_ + "/" + file_);
            }
            for (const std::string& h : head_)
            {
                os_ << h << '\n';
            }
        }
        os_ << line << '\n';
        os_.flush();
    }

private:
    std::string               dir_, file_;
    bool                      enabled_ = true;
    std::vector<std::string>  head_;
    std::ofstream             os_;
};

// ---------------------------------------------------------------------------------------------------------------
// the run-time state every object shares
// ---------------------------------------------------------------------------------------------------------------

class Runtime
{
public:
    explicit Runtime(Services s) : svc(std::move(s)) {}

    Services svc;
    label    stamp = 0;                                                  // one per execute()
    std::map<std::string, std::string>                    exprs;         // exprField name -> expression
    std::map<std::string, std::map<std::string, scalar>> results;        // object -> "operation(field)" -> value

    std::size_t nCells() const { return svc.geom->V().size(); }

    DatFile datFile(const std::string& object, const std::string& file) const
    {
        return DatFile(dir(object), file, svc.isMaster);
    }

    std::string dir(const std::string& object) const
    {
        return svc.caseDir + "/postProcessing/" + object + "/" + svc.startName;
    }

    bool isScalarName(const std::string& name) const
    {
        return name == svc.phaseField || name == "p_rgh" || exprs.count(name) > 0;
    }

    SF scalarField(const std::string& name)
    {
        SF s;
        if (!lookupScalar(name, s))
        {
            throw std::runtime_error("brae function objects: the field `" + name + "` is not one brae can give a "
                                     "function object (" + svc.phaseField + ", p_rgh, U and exprField results are)");
        }
        return s;
    }

    VF velocity()
    {
        VF v;
        v.x = sfBorrow(svc.src.Ux());
        v.y = sfBorrow(svc.src.Uy());
        v.z = sfBorrow(svc.src.Uz());
        return v;
    }

    VF cellCentres()
    {
        if (!posX_)
        {
            const std::vector<vector>& C = svc.geom->C();
            posX_ = std::make_shared<std::vector<scalar>>(C.size());
            posY_ = std::make_shared<std::vector<scalar>>(C.size());
            posZ_ = std::make_shared<std::vector<scalar>>(C.size());
            for (std::size_t i = 0; i < C.size(); ++i)
            {
                (*posX_)[i] = C[i].x;
                (*posY_)[i] = C[i].y;
                (*posZ_)[i] = C[i].z;
            }
        }
        VF v;
        v.x.ref = posX_.get();
        v.y.ref = posY_.get();
        v.z.ref = posZ_.get();
        return v;
    }

    scalar result(const std::string& object, const std::string& key) const
    {
        const auto o = results.find(object);
        if (o == results.end() || !o->second.count(key))
        {
            throw std::runtime_error("brae function objects: `" + key + "` of object `" + object + "` is not "
                                     "available at this step (the object is absent, disabled or not due yet)");
        }
        return o->second.at(key);
    }

    bool hasResult(const std::string& object, const std::string& key) const
    {
        const auto o = results.find(object);
        return o != results.end() && o->second.count(key) != 0;
    }

    // Compile an expression once, after all exprFields have been read. An empty cached tree means the host
    // evaluator owns this field; no unsupported construct is silently approximated.
    const std::vector<DeviceExpr>* deviceExpression(const std::string& field)
    {
        const auto made = deviceIr_.find(field);
        if (made != deviceIr_.end()) return made->second.empty() ? nullptr : &made->second;
        std::vector<DeviceExpr> out;
        DeviceScope sc;
        sc.phaseField = svc.phaseField;
        sc.nodes = &out;
        std::vector<std::string> active;
        std::function<bool(const std::string&, int&)> lookup;
        lookup = [&](const std::string& name, int& node)
        {
            if (name == svc.phaseField) { node = deviceLeaf(sc, DeviceExpr::Alpha); return true; }
            const auto e = exprs.find(name);
            if (e == exprs.end()) return false;
            if (std::find(active.begin(), active.end(), name) != active.end()) return false;
            active.push_back(name);
            try { node = DeviceExprParser(e->second, sc).parse(); }
            catch (...) { active.pop_back(); throw; }
            active.pop_back();
            return true;
        };
        sc.scalarField = lookup;
        try
        {
            int root = -1;
            if (field == svc.phaseField) deviceLeaf(sc, DeviceExpr::Alpha);
            else if (!lookup(field, root)) return nullptr;
        }
        catch (const std::exception&)
        {
            deviceIr_[field] = {};
            return nullptr;
        }
        deviceIr_[field] = std::move(out);
        return &deviceIr_[field];
    }

    const std::vector<scalar>* queueDeviceIntegral(const std::string& field, const label* cells, label n,
                                                    std::shared_ptr<void>& plan)
    {
        const std::vector<DeviceExpr>* nodes = deviceExpression(field);
        if (!svc.src.deviceIntegrals || !svc.src.deviceFields || !nodes) return nullptr;
        if (!plan && svc.src.prepareDeviceIntegral)
        {
            plan = svc.src.prepareDeviceIntegral(field, *nodes, cells, n);
            if (!plan) return nullptr;
        }
        DeviceIntegral q;
        q.phaseField = svc.phaseField;
        q.expression = field;
        if (!svc.src.prepareDeviceIntegral) q.nodes = *nodes;
        q.plan = plan;
        q.fields = svc.src.deviceFields();
        q.cells = cells;
        q.n = n;
        q.result = &deviceValues_.emplace_back();
        deviceIntegrals_.push_back(std::move(q));
        return q.result;
    }

    void beginDeviceStep()
    {
        deviceIntegrals_.clear();
        deviceValues_.clear();
    }

    void flushDeviceIntegrals()
    {
        if (deviceIntegrals_.empty()) return;
        if (!svc.src.deviceIntegrals(deviceIntegrals_))
        {
            for (DeviceIntegral& q : deviceIntegrals_) q.result->clear();
        }
        deviceIntegrals_.clear();
    }

    std::map<std::string, std::vector<DeviceExpr>>       deviceIr_;
    std::vector<DeviceIntegral>                          deviceIntegrals_;
    std::deque<std::vector<scalar>>                      deviceValues_;

private:
    std::map<std::string, std::pair<label, SF>> cache_;
    std::vector<std::string>                    active_;
    std::shared_ptr<std::vector<scalar>>        posX_, posY_, posZ_;

    bool lookupScalar(const std::string& name, SF& out)
    {
        if (name == svc.phaseField)
        {
            out = sfBorrow(svc.src.alpha());
            return true;
        }
        if (name == "p_rgh")
        {
            out = sfBorrow(svc.src.p_rgh());
            return true;
        }
        const auto e = exprs.find(name);
        if (e == exprs.end())
        {
            return false;
        }
        const auto c = cache_.find(name);
        if (c != cache_.end() && c->second.first == stamp)
        {
            out = c->second.second;
            return true;
        }
        if (std::find(active_.begin(), active_.end(), name) != active_.end())
        {
            throw std::runtime_error("brae function objects: the expression fields refer to each other in a "
                                     "loop at `" + name + "`");
        }
        active_.push_back(name);
        Scope sc;
        sc.n = nCells();
        sc.scalarField = [this](const std::string& nm, SF& o) { return lookupScalar(nm, o); };
        sc.vectorField = [this](const std::string& nm, VF& o)
        {
            if (nm != "U")
            {
                return false;
            }
            o = velocity();
            return true;
        };
        sc.pos = [this]() { return cellCentres(); };
        SF r;
        try
        {
            r = ExprParser(e->second, sc).parseScalar();
        }
        catch (...)
        {
            active_.pop_back();
            throw;
        }
        active_.pop_back();
        cache_[name] = std::make_pair(stamp, r);
        out = r;
        return true;
    }
};

// The uniform/functionObjects/functionObjectProperties file is a restart input as well as an output:
// runTimeControl reads its condition properties and condition queries read these results. Keep the same
// three-level result dictionary as OpenFOAM's functionObjects::properties (SHA1("results"), object,
// value type), rather than a flat map that OpenFOAM would silently ignore.
struct FunctionObjectState
{
    struct Value
    {
        std::string type;
        std::string text;
        int         width = 0;
    };
    std::map<std::string, std::map<std::string, std::map<std::string, Value>>> results;
    std::map<std::string, std::vector<std::string>> properties;

    void scalar(const std::string& object, const std::string& key, double value, int width = 0)
    {
        std::ostringstream os;
        os.precision(16);
        os << value;
        results[object]["scalar"][key] = Value{"scalar", os.str(), width};
    }
    void vector(const std::string& object, const std::string& key, double x, double y, double z)
    {
        std::ostringstream os;
        os.precision(12);
        os << "( " << x << ' ' << y << ' ' << z << " )";
        results[object]["vector"][key] = Value{"vector", os.str(), 0};
    }
    void label(const std::string& object, const std::string& key, long long value)
    {
        results[object]["label"][key] = Value{"label", std::to_string(value), 0};
    }
};

// ---------------------------------------------------------------------------------------------------------------
// the objects
// ---------------------------------------------------------------------------------------------------------------

class Object
{
public:
    virtual ~Object() = default;
    std::string name;
    bool        enabled = true;
    Control     ctl;
    bool        log = true;

    // compute and write, when ctl says it is due
    virtual void write(const Step&, Runtime&) = 0;
    // every step, after the writes; true asks the run to end (and write) at this step
    virtual bool evaluatesEveryStep() const { return false; }
    virtual bool evaluate(const Step&, Runtime&) { return false; }
    // Integrals that were queued during write() are emitted after the Source's one batched device call.
    virtual void afterDeviceBatch(const Step&, Runtime&) {}
    // Latest values that OpenFOAM persists for a restart (not the time-series output).
    virtual void saveState(FunctionObjectState&) const {}
};

inline void refuseKey(const FoamDict& d, const std::string& object, const char* key, const char* why)
{
    if (d.found(key))
    {
        throw std::runtime_error("brae function objects: `" + object + "` sets `" + key + "` -- " + why);
    }
}

inline std::vector<std::string> readFieldNames(const FoamDict& d, const std::string& key)
{
    // `fields (alpha.water alphaUz)` and `fields (volIntegrate(alphaUz))`: split at the top level only
    const auto* t = d.find(key);
    std::vector<std::string> out;
    if (!t)
    {
        return out;
    }
    // The reader hands parentheses over as tokens of their own, so `volIntegrate(alphaUz)` is three tokens.
    // A word followed by `(` is a call and stays one word; words are otherwise separated by a space.
    std::string joined;
    for (const std::string& s : *t)
    {
        const bool paren = (s == "(" || s == ")");
        if (paren)
        {
            joined += s;
        }
        else
        {
            joined += ((joined.empty() || joined.back() == '(') ? "" : " ") + s;
        }
    }
    const std::size_t a = joined.find('(');
    const std::size_t b = joined.rfind(')');
    if (a == std::string::npos || b == std::string::npos || b < a)
    {
        throw std::runtime_error("brae function objects: `" + key + "` is not a list");
    }
    std::string cur;
    int depth = 0;
    for (std::size_t i = a + 1; i < b; ++i)
    {
        const char ch = joined[i];
        if (ch == '(')
        {
            ++depth;
        }
        else if (ch == ')')
        {
            --depth;
        }
        if (depth == 0 && std::isspace(static_cast<unsigned char>(ch)))
        {
            if (!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
        }
        else if (!std::isspace(static_cast<unsigned char>(ch)))
        {
            cur += ch;
        }
    }
    if (!cur.empty())
    {
        out.push_back(cur);
    }
    return out;
}

// the region's cells (empty `all`) and the sums every volume operation needs
struct CellRegion
{
    bool                all = true;
    std::vector<label>  cells;
    std::string         description;
    std::string         zone;
    scalar              volume = 0;
    label               n = 0;
};

inline CellRegion readCellRegion(const FoamDict& d, const std::string& object, Runtime& rt)
{
    CellRegion r;
    const std::string type = d.wordOr("regionType", "");
    const std::vector<scalar>& V = rt.svc.geom->V();
    if (type == "all")
    {
        r.all = true;
        r.description = "all";
        r.n = static_cast<label>(V.size());
        for (scalar v : V)
        {
            r.volume += v;
        }
    }
    else if (type == "cellZone")
    {
        const std::string zone = d.wordOr("name", "");
        const auto it = rt.svc.cellZones ? rt.svc.cellZones->find(zone) : std::map<std::string, std::vector<label>>::const_iterator();
        if (!rt.svc.cellZones || it == rt.svc.cellZones->end())
        {
            rt.svc.red.abortAll("brae function objects: `" + object + "` names the cellZone `" + zone
                                + "`, which is not in constant/polyMesh/cellZones");
        }
        r.all = false;
        r.cells = it->second;
        r.description = "cellZone " + zone;
        r.zone = zone;
        r.n = static_cast<label>(r.cells.size());
        for (label c : r.cells)
        {
            r.volume += V[static_cast<std::size_t>(c)];
        }
    }
    else
    {
        throw std::runtime_error("brae function objects: `" + object + "` has regionType `" + type
                                 + "`; brae has all and cellZone for volFieldValue");
    }
    // volRegion::calculateCache (volRegion.C): nCells_ = returnReduce(count), V_ reduced from each rank's own
    // sequential sum -- the header's `# Cells` and `# Volume` are the whole mesh's. A zone nobody holds is an
    // error there ("Region has no cells").
    if (rt.svc.red.parallel())
    {
        std::vector<double> nv = {static_cast<double>(r.n), static_cast<double>(r.volume)};
        rt.svc.red.reduceSum(nv);
        r.n = static_cast<label>(nv[0]);
        r.volume = nv[1];
        if (r.n == 0)
        {
            throw std::runtime_error("brae function objects: `" + object + "` selects a region with no cells "
                                     "on any rank (OpenFOAM stops on it too: volRegion, `Region has no cells`)");
        }
    }
    return r;
}

class VolFieldValue : public Object
{
public:
    VolFieldValue(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        regionType_ = d.wordOr("regionType", "");
        region_ = readCellRegion(d, name, rt);
        op_ = d.wordOr("operation", "");
        if (op_ != "volIntegrate" && op_ != "volAverage" && op_ != "sum" && op_ != "min" && op_ != "max")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has operation `" + op_
                                     + "`; brae has volIntegrate, volAverage, sum, min and max");
        }
        fields_ = readFieldNames(d, "fields");
        if (fields_.empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` lists no fields");
        }
        for (const std::string& f : fields_)
        {
            if (f == "U" || !rt.isScalarName(f))
            {
                throw std::runtime_error("brae function objects: `" + name + "` takes the field `" + f
                                         + "`, which is not a scalar field brae knows (" + rt.svc.phaseField
                                         + ", p_rgh and exprField results are)");
            }
        }
        writeToFile_ = d.switchOr("writeToFile", true);
        if (d.switchOr("writeFields", false))
        {
            throw std::runtime_error("brae function objects: `" + name + "` sets writeFields; brae does not write "
                                     "the region's fields");
        }
        refuseKey(d, name, "weightField", "weighted operations are not ported");
        refuseKey(d, name, "postOperation", "post operations are not ported");
    }

    void open(Runtime& rt)
    {
        if (!writeToFile_)
        {
            return;
        }
        dat_ = rt.datFile(name, "volFieldValue.dat");
        dat_.header("# Region type : " + region_.description);
        dat_.header("# Cells       : " + std::to_string(region_.n));
        dat_.header("# Volume      : " + fmtNum(region_.volume, rt.svc.precision));
        std::string h = "# Time       ";
        for (const std::string& f : fields_)
        {
            h += " " + op_ + "(" + f + ")";
        }
        dat_.header(h);
    }

    void write(const Step& s, Runtime& rt) override
    {
        if (op_ != "volIntegrate") { emit(s, rt); return; }
        deviceDue_ = true;
        deviceResults_.clear();
        // one plan per field, made on the first write and kept: a plan holds the expression nodes and the
        // region's cell list on the device, so a fresh one at every write is a leak of the cell list per step
        devicePlans_.resize(fields_.size());
        for (std::size_t i = 0; i < fields_.size(); ++i)
        {
            deviceResults_.push_back(rt.queueDeviceIntegral(fields_[i], region_.all ? nullptr : region_.cells.data(),
                                                             region_.all ? static_cast<label>(rt.nCells())
                                                                         : static_cast<label>(region_.cells.size()),
                                                             devicePlans_[i]));
        }
    }


    void saveState(FunctionObjectState& st) const override
    {
        for (const auto& result : latest_)
        {
            const std::string key = op_ + "(" + (region_.all ? std::string() : region_.zone + ",")
                                  + result.first + ")";
            st.scalar(name, key, result.second);
        }
    }

    void afterDeviceBatch(const Step& s, Runtime& rt) override
    {
        if (deviceDue_) emit(s, rt);
    }

private:
    std::string               regionType_, op_;
    std::vector<std::string>  fields_;
    CellRegion                region_;
    bool                      writeToFile_ = true;
    DatFile                   dat_;
    std::map<std::string, scalar> latest_;
    std::vector<const std::vector<scalar>*> deviceResults_;
    std::vector<std::shared_ptr<void>>    devicePlans_;
    bool                      deviceDue_ = false;
    // What one rank adds up over its part of the region
    struct Part
    {
        scalar acc = 0, vol = 0, mn = 0, mx = 0;
    };

    void emit(const Step& s, Runtime& rt)
    {
        const std::vector<scalar>& V = rt.svc.geom->V();
        const Services::Reduce& red = rt.svc.red;
        std::vector<Part> parts(fields_.size());
        std::vector<char> fromDevice(fields_.size(), 0);
        for (std::size_t i = 0; i < fields_.size(); ++i)
        {
            const bool device = i < deviceResults_.size() && deviceResults_[i] && deviceResults_[i]->size() == 1;
            fromDevice[i] = device;
            if (device) parts[i].acc = (*deviceResults_[i])[0];
            else parts[i] = localPart(rt.scalarField(fields_[i]), V);
        }
        std::vector<scalar> values(fields_.size());
        if (!red.parallel())
        {
            for (std::size_t i = 0; i < fields_.size(); ++i) values[i] = fromDevice[i] ? parts[i].acc : finish(parts[i]);
        }
        else
        {
            // volFieldValue::processValues: gSum(V*values) (each rank's sequential sum, then the ranks' in rank
            // order), gSum(V) for the average, gMin/gMax -- one collective for all the fields
            std::vector<double> pack;
            const bool average = op_ == "volAverage";
            for (const Part& p : parts)
            {
                if (op_ == "min") pack.push_back(p.mn);
                else if (op_ == "max") pack.push_back(p.mx);
                else
                {
                    pack.push_back(p.acc);
                    if (average) pack.push_back(p.vol);
                }
            }
            if (op_ == "min") red.reduceMin(pack);
            else if (op_ == "max") red.reduceMax(pack);
            else red.reduceSum(pack);
            std::size_t k = 0;
            for (std::size_t i = 0; i < fields_.size(); ++i)
            {
                if (average)
                {
                    values[i] = pack[k]/pack[k + 1];
                    k += 2;
                }
                else values[i] = pack[k++];
            }
        }
        std::string line = fmtNum(s.time, rt.svc.precision);
        for (std::size_t i = 0; i < fields_.size(); ++i)
        {
            rt.results[name][op_ + "(" + fields_[i] + ")"] = values[i];
            latest_[fields_[i]] = values[i];
            line += "\t" + fmtNum(values[i], rt.svc.precision);
        }
        if (writeToFile_) dat_.row(line);
        deviceResults_.clear();
        deviceDue_ = false;
    }

    Part localPart(const SF& f, const std::vector<scalar>& V) const
    {
        auto at = [&](std::size_t c) { return f.isConst() ? f.c : f.v()[c]; };
        Part p;
        p.mn = std::numeric_limits<scalar>::max();
        p.mx = -std::numeric_limits<scalar>::max();
        auto visit = [&](std::size_t c)
        {
            const scalar x = at(c);
            p.acc += (op_ == "volIntegrate" || op_ == "volAverage") ? V[c]*x : x;
            p.vol += V[c];
            p.mn = std::min(p.mn, x);
            p.mx = std::max(p.mx, x);
        };
        if (region_.all)
        {
            for (std::size_t c = 0; c < V.size(); ++c) visit(c);
        }
        else
        {
            for (label c : region_.cells) visit(static_cast<std::size_t>(c));
        }
        return p;
    }

    scalar finish(const Part& p) const
    {
        if (op_ == "volAverage") return p.acc/p.vol;
        if (op_ == "min") return p.mn;
        if (op_ == "max") return p.mx;
        return p.acc;
    }
};

class SurfaceFieldValue : public Object
{
public:
    SurfaceFieldValue(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        const std::string type = d.wordOr("regionType", "");
        if (type != "patch")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has regionType `" + type
                                     + "`; brae has patch for surfaceFieldValue");
        }
        patchName_ = d.wordOr("name", "");
        const std::vector<FvPatch>& patches = *rt.svc.patches;
        patch_ = -1;
        for (std::size_t i = 0; i < patches.size(); ++i)
        {
            if (patches[i].name == patchName_)
            {
                patch_ = static_cast<int>(i);
            }
        }
        if (patch_ < 0)
        {
            rt.svc.red.abortAll("brae function objects: `" + name + "` names the patch `" + patchName_
                                + "`, which the mesh does not have");
        }
        for (scalar a : patches[static_cast<std::size_t>(patch_)].magSf)
        {
            area_ += a;
        }
        nFaces_ = static_cast<label>(patches[static_cast<std::size_t>(patch_)].magSf.size());
        // the patch's faces lie on several ranks (and on none of some): `# Faces` is returnReduce(count) and
        // the area gSum(magSf) of the rank's own sequential sums (surfaceFieldValue.C:203, :560). A processor
        // patch is never selected: the name is a physical patch's.
        if (rt.svc.red.parallel())
        {
            std::vector<double> fa = {static_cast<double>(nFaces_), static_cast<double>(area_)};
            rt.svc.red.reduceSum(fa);
            nFaces_ = static_cast<label>(fa[0]);
            area_ = fa[1];
        }
        op_ = d.wordOr("operation", "");
        if (op_ != "areaAverage" && op_ != "areaIntegrate" && op_ != "sum" && op_ != "min" && op_ != "max")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has operation `" + op_
                                     + "`; brae has areaAverage, areaIntegrate, sum, min and max");
        }
        fields_ = readFieldNames(d, "fields");
        if (fields_.empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` lists no fields");
        }
        for (const std::string& f : fields_)
        {
            if (f != rt.svc.phaseField)
            {
                throw std::runtime_error("brae function objects: `" + name + "` takes the field `" + f
                                         + "`; only " + rt.svc.phaseField + " has patch values here");
            }
        }
        writeToFile_ = d.switchOr("writeToFile", true);
        if (d.switchOr("writeFields", false))
        {
            throw std::runtime_error("brae function objects: `" + name + "` sets writeFields; brae does not write "
                                     "the region's fields");
        }
        refuseKey(d, name, "weightField", "weighted operations are not ported");
        refuseKey(d, name, "postOperation", "post operations are not ported");
    }

    void open(Runtime& rt)
    {
        if (!writeToFile_)
        {
            return;
        }
        dat_ = rt.datFile(name, "surfaceFieldValue.dat");
        dat_.header("# Region type : patch " + patchName_);
        dat_.header("# Faces       : " + std::to_string(nFaces_));
        dat_.header("# Area        : " + fmtNum(area_, rt.svc.precision));
        std::string h = "# Time       ";
        for (const std::string& f : fields_)
        {
            h += " " + op_ + "(" + f + ")";
        }
        dat_.header(h);
    }

    void saveState(FunctionObjectState& st) const override
    {
        for (const auto& result : latest_) st.scalar(name, op_ + "(" + patchName_ + "," + result.first + ")", result.second);
    }

    void write(const Step& s, Runtime& rt) override
    {
        const std::vector<scalar>& a = (*rt.svc.patches)[static_cast<std::size_t>(patch_)].magSf;
        std::string line = fmtNum(s.time, rt.svc.precision);
        for (const std::string& f : fields_)
        {
            const std::vector<scalar>& v = rt.svc.src.alphaPatch(static_cast<std::size_t>(patch_));
            if (v.size() != a.size())
            {
                throw std::runtime_error("brae function objects: `" + name + "`: the patch values of " + f
                                         + " have " + std::to_string(v.size()) + " faces, the patch has "
                                         + std::to_string(a.size()));
            }
            scalar acc = 0;
            scalar mn = std::numeric_limits<scalar>::max();
            scalar mx = -std::numeric_limits<scalar>::max();
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                acc += (op_ == "areaAverage" || op_ == "areaIntegrate") ? a[i]*v[i] : v[i];
                mn = std::min(mn, v[i]);
                mx = std::max(mx, v[i]);
            }
            // gSum / gMin / gMax over the ranks' parts of the patch (a rank without faces adds 0, and the
            // identity of min and max)
            const Services::Reduce& red = rt.svc.red;
            if (op_ == "min") mn = red.reduceMin(mn);
            else if (op_ == "max") mx = red.reduceMax(mx);
            else acc = red.reduceSum(acc);
            scalar value = acc;
            if (op_ == "areaAverage") value = acc/area_;
            if (op_ == "min") value = mn;
            if (op_ == "max") value = mx;
            rt.results[name][op_ + "(" + f + ")"] = value;
            latest_[f] = value;
            line += "\t" + fmtNum(value, rt.svc.precision);
        }
        if (writeToFile_)
        {
            dat_.row(line);
        }
    }

private:
    std::string               patchName_, op_;
    int                       patch_ = -1;
    std::vector<std::string>  fields_;
    scalar                    area_ = 0;
    label                     nFaces_ = 0;
    bool                      writeToFile_ = true;
    DatFile                   dat_;
    std::map<std::string, scalar> latest_;
};

// OpenFOAM's polyMesh::findCell with its default decomposition (CELL_TETS), for a handful of points. A cell is
// split into tetrahedra -- the cell centre, the face centre and each edge of the face -- and holds the points that
// lie in one of them; a point on a face shared by two cells belongs to both, and the tolerance admits it. The face
// planes (FACE_PLANES) are not used: a snappyHexMesh cell with a concave face fails that test for points inside it.
// Tried on the nearest cell centre first, then on the cells whose centres are close, then on all of them.
inline std::vector<label> locatePoints(
    const std::vector<vector>& pts,
    const PrimitiveMesh&       m,
    const FvGeometry&          g,
    const std::string&         object)
{
    const label nC = m.nCells();
    const label nF = m.nFaces();
    const label nI = m.nInternalFaces();
    const std::vector<label>& owner = m.owner();
    const std::vector<label>& nei   = m.neighbour();
    const std::vector<vector>& P    = m.points();
    std::vector<label> start(static_cast<std::size_t>(nC) + 1, 0);
    for (label f = 0; f < nF; ++f) ++start[static_cast<std::size_t>(owner[f]) + 1];
    for (label f = 0; f < nI; ++f) ++start[static_cast<std::size_t>(nei[f]) + 1];
    for (label c = 0; c < nC; ++c) start[static_cast<std::size_t>(c) + 1] += start[static_cast<std::size_t>(c)];
    std::vector<label> faces(static_cast<std::size_t>(start[static_cast<std::size_t>(nC)]));
    std::vector<label> cursor(start.begin(), start.end() - 1);
    for (label f = 0; f < nF; ++f) faces[static_cast<std::size_t>(cursor[static_cast<std::size_t>(owner[f])]++)] = f;
    for (label f = 0; f < nI; ++f) faces[static_cast<std::size_t>(cursor[static_cast<std::size_t>(nei[f])]++)] = f;

    auto vol = [](const vector& a, const vector& b, const vector& c, const vector& d)
    {
        return dot(b - a, cross(c - a, d - a));
    };
    // p inside the tetrahedron (a b c d) of either orientation, a point on a face of it included
    auto inTet = [&](const vector& a, const vector& b, const vector& c, const vector& d, const vector& p)
    {
        const scalar v0 = vol(a, b, c, d);
        if (v0 == scalar(0))
        {
            return false;
        }
        const scalar s = v0 > scalar(0) ? scalar(1) : scalar(-1);
        const scalar eps = scalar(1e-9)*std::fabs(v0);
        return s*vol(p, b, c, d) >= -eps && s*vol(a, p, c, d) >= -eps
            && s*vol(a, b, p, d) >= -eps && s*vol(a, b, c, p) >= -eps;
    };
    auto inCell = [&](const vector& p, label c)
    {
        const vector& cc = g.C()[static_cast<std::size_t>(c)];
        for (label k = start[static_cast<std::size_t>(c)]; k < start[static_cast<std::size_t>(c) + 1]; ++k)
        {
            const label f = faces[static_cast<std::size_t>(k)];
            const vector& fc = g.Cf()[static_cast<std::size_t>(f)];
            const label n = m.faceSize(f);
            for (label i = 0; i < n; ++i)
            {
                if (inTet(cc, P[static_cast<std::size_t>(m.faceVert(f, i))],
                          P[static_cast<std::size_t>(m.faceVert(f, (i + 1) % n))], fc, p))
                {
                    return true;
                }
            }
        }
        return false;
    };

    std::vector<label> out;
    for (const vector& p : pts)
    {
        label best = -1;
        scalar bd = std::numeric_limits<scalar>::max();
        for (label c = 0; c < nC; ++c)
        {
            const vector d = g.C()[static_cast<std::size_t>(c)] - p;
            const scalar dd = dot(d, d);
            if (dd < bd)
            {
                bd = dd;
                best = c;
            }
        }
        label found = (best >= 0 && inCell(p, best)) ? best : -1;
        if (found < 0 && best >= 0)
        {
            // the cells whose centres are within three cell sizes of the point, in index order
            const scalar R = scalar(3)*std::cbrt(g.V()[static_cast<std::size_t>(best)]);
            for (label c = 0; c < nC && found < 0; ++c)
            {
                const vector d = g.C()[static_cast<std::size_t>(c)] - p;
                if (dot(d, d) < R*R && inCell(p, c))
                {
                    found = c;
                }
            }
        }
        for (label c = 0; c < nC && found < 0; ++c)
        {
            if (inCell(p, c))
            {
                found = c;
            }
        }
        // found < 0: in no cell -- the caller says what to do about it
        out.push_back(found);
    }
    return out;
}

class Probes : public Object
{
public:
    Probes(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        if (!d.switchOr("fixedLocations", true))
        {
            throw std::runtime_error("brae function objects: `" + name + "` has fixedLocations false; the probes "
                                     "are fixed points here");
        }
        const std::string scheme = d.wordOr("interpolationScheme", "cell");
        if (scheme != "cell")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has interpolationScheme " + scheme
                                     + "; brae reads the cell value (cell)");
        }
        fields_ = readFieldNames(d, "fields");
        for (const std::string& f : fields_)
        {
            if (f != rt.svc.phaseField && f != "p_rgh" && f != "U")
            {
                throw std::runtime_error("brae function objects: `" + name + "` probes `" + f
                                         + "`; brae probes " + rt.svc.phaseField + ", p_rgh and U");
            }
        }
        refuseKey(d, name, "sourceName", "probing a surface is not ported");
        refuseKey(d, name, "patchName", "probing a patch is not ported");
        const std::vector<scalar> xyz = d.scalarListOr("probeLocations", std::vector<scalar>());
        if (xyz.empty() || xyz.size() % 3 != 0)
        {
            throw std::runtime_error("brae function objects: `" + name + "` has no probeLocations, or a list that "
                                     "is not made of points");
        }
        for (std::size_t i = 0; i + 2 < xyz.size(); i += 3)
        {
            points_.push_back(vector{xyz[i], xyz[i + 1], xyz[i + 2]});
        }
        // OpenFOAM's probes skip a location that lies in no cell ("Did not find location ... in any cell. Skipping
        // location.") and write the files for the others, numbered from 0 again: the columns after a skipped probe
        // shift down. brae does the same, and says which locations it left out.
        //
        // UNDER -parallel (internalFieldProbe::findElements, ::sample): every rank looks for the point in ITS
        // cells; the probe is skipped when no rank has it; the VALUE is the first set one in rank order
        // (listCombineReduce with isNotEqOp), i.e. the lowest rank whose cells hold the point -- a point on a
        // processor face is in a cell of both and the lower rank owns it (OpenFOAM warns about this case).
        // The master writes every column; the other ranks send their owned values.
        {
            const std::vector<label> found = locatePoints(points_, *rt.svc.mesh, *rt.svc.geom, name);
            const Services::Reduce& red = rt.svc.red;
            std::vector<int> owner(points_.size(), 0);
            if (red.parallel())
            {
                std::vector<double> flags(points_.size());
                for (std::size_t i = 0; i < points_.size(); ++i) flags[i] = found[i] >= 0 ? 1.0 : 0.0;
                const std::vector<std::vector<double>> all = red.gatherToMaster(flags);
                for (std::size_t i = 0; i < points_.size(); ++i)
                {
                    owner[i] = -1;
                    int holders = 0;
                    for (std::size_t r = 0; r < all.size(); ++r)
                    {
                        if (all[r][i] != 0.0)
                        {
                            if (owner[i] < 0) owner[i] = static_cast<int>(r);
                            ++holders;
                        }
                    }
                    if (holders > 1 && red.master())
                    {
                        noticeIgnored("functions/" + name + " probe " + std::to_string(i),
                                      "is in cells of " + std::to_string(holders) + " ranks (a point on a processor "
                                      "face); rank " + std::to_string(owner[i]) + " owns it, as in OpenFOAM");
                    }
                }
            }
            else
            {
                for (std::size_t i = 0; i < points_.size(); ++i) owner[i] = found[i] >= 0 ? 0 : -1;
            }
            std::vector<vector> kept;
            std::vector<label>  keptCells;
            std::vector<int>    keptOwner;
            for (std::size_t i = 0; i < points_.size(); ++i)
            {
                if (owner[i] < 0)
                {
                    if (red.master())
                    {
                        noticeIgnored("functions/" + name + " probe " + std::to_string(i) + " at ("
                                      + fmtNum(points_[i].x, 8) + " " + fmtNum(points_[i].y, 8) + " "
                                      + fmtNum(points_[i].z, 8) + ")",
                                      "lies in no cell; skipped, as OpenFOAM's probes skip it. The files have one column "
                                      "fewer, and the probes after this one are numbered one lower");
                    }
                }
                else
                {
                    kept.push_back(points_[i]);
                    // this rank's cell when it owns the probe, -1 when another rank does
                    keptCells.push_back(owner[i] == red.rank ? found[i] : label(-1));
                    keptOwner.push_back(owner[i]);
                }
            }
            if (kept.empty())
            {
                throw std::runtime_error("brae function objects: none of the locations of `" + name
                                         + "` lies in a cell of the mesh");
            }
            points_ = kept;
            cells_ = keptCells;
            owner_ = keptOwner;
        }
        for (const std::string& f : fields_)
        {
            DatFile df = rt.datFile(name, f);
            for (std::size_t i = 0; i < points_.size(); ++i)
            {
                df.header("# Probe " + std::to_string(i) + " (" + fmtNum(points_[i].x, rt.svc.precision) + " "
                          + fmtNum(points_[i].y, rt.svc.precision) + " " + fmtNum(points_[i].z, rt.svc.precision) + ")");
            }
            std::string h = "#               Probe";
            for (std::size_t i = 0; i < points_.size(); ++i)
            {
                char b[32];
                std::snprintf(b, sizeof b, " %13zu", i);
                h += b;
            }
            df.header(h);
            df.header("#                Time");
            files_.push_back(std::move(df));
        }
    }

    void saveState(FunctionObjectState& st) const override
    {
        for (const auto& item : latest_)
        {
            const std::string& f = item.first;
            const std::vector<double>& v = item.second;
            const int nComp = (f == "U") ? 3 : 1;
            if (v.empty()) continue;
            auto stat = [&](int component, bool wantMin, bool average)
            {
                double x = wantMin ? std::numeric_limits<double>::max()
                                   : average ? 0.0 : -std::numeric_limits<double>::max();
                for (std::size_t i = static_cast<std::size_t>(component); i < v.size(); i += static_cast<std::size_t>(nComp))
                {
                    if (average) x += v[i];
                    else if (wantMin) x = std::min(x, v[i]);
                    else x = std::max(x, v[i]);
                }
                return average ? x/static_cast<double>(v.size()/static_cast<std::size_t>(nComp)) : x;
            };
            st.label(name, "size(" + f + ")", static_cast<long long>(v.size()/static_cast<std::size_t>(nComp)));
            if (nComp == 1)
            {
                st.scalar(name, "average(" + f + ")", stat(0, false, true), 17);
                st.scalar(name, "min(" + f + ")", stat(0, true, false), 13);
                st.scalar(name, "max(" + f + ")", stat(0, false, false), 13);
            }
            else
            {
                st.vector(name, "average(" + f + ")", stat(0, false, true), stat(1, false, true), stat(2, false, true));
                st.vector(name, "min(" + f + ")", stat(0, true, false), stat(1, true, false), stat(2, true, false));
                st.vector(name, "max(" + f + ")", stat(0, false, false), stat(1, false, false), stat(2, false, false));
            }
        }
    }

    void write(const Step& s, Runtime& rt) override
    {
        const Services::Reduce& red = rt.svc.red;
        for (std::size_t k = 0; k < fields_.size(); ++k)
        {
            const std::string& f = fields_[k];
            const int nComp = (f == "U") ? 3 : 1;
            // the probes this rank owns, in probe order
            std::vector<double> mine;
            if (f == "U")
            {
                const std::vector<scalar>& ux = rt.svc.src.Ux();
                const std::vector<scalar>& uy = rt.svc.src.Uy();
                const std::vector<scalar>& uz = rt.svc.src.Uz();
                for (label c : cells_)
                {
                    if (c < 0) continue;
                    const std::size_t i = static_cast<std::size_t>(c);
                    mine.push_back(ux[i]);
                    mine.push_back(uy[i]);
                    mine.push_back(uz[i]);
                }
            }
            else
            {
                const std::vector<scalar>& v = (f == "p_rgh") ? rt.svc.src.p_rgh() : rt.svc.src.alpha();
                for (label c : cells_)
                {
                    if (c < 0) continue;
                    mine.push_back(v[static_cast<std::size_t>(c)]);
                }
            }
            // everyone takes part; only the master has a file to write
            const std::vector<std::vector<double>> all = red.gatherToMaster(mine);
            std::string line = fmtNum(s.time, rt.svc.precision);
            std::vector<std::size_t> cursor(all.size(), 0);
            std::vector<double> stateValues;
            stateValues.reserve(cells_.size()*static_cast<std::size_t>(nComp));
            for (std::size_t pi = 0; pi < cells_.size(); ++pi)
            {
                const std::size_t r = static_cast<std::size_t>(owner_[pi]);
                const double* v = all[r].data() + cursor[r];
                cursor[r] += static_cast<std::size_t>(nComp);
                for (int j = 0; j < nComp; ++j) stateValues.push_back(v[j]);
                if (f == "U")
                {
                    line += " (" + fmtNum(v[0], rt.svc.precision) + " " + fmtNum(v[1], rt.svc.precision) + " "
                          + fmtNum(v[2], rt.svc.precision) + ")";
                }
                else
                {
                    line += " " + fmtNum(v[0], rt.svc.precision);
                }
            }
            latest_[f] = std::move(stateValues);
            files_[k].row(line);
        }
    }

private:
    std::vector<std::string> fields_;
    std::vector<vector>      points_;
    std::vector<label>       cells_;     // this rank's cell of each kept probe, -1 when another rank owns it
    std::vector<int>         owner_;     // the owning rank of each kept probe
    std::vector<DatFile>     files_;
    std::map<std::string, std::vector<double>> latest_;
};

class FieldMinMax : public Object
{
public:
    FieldMinMax(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        fields_ = readFieldNames(d, "fields");
        mode_ = d.wordOr("mode", "magnitude");
        if (mode_ != "magnitude" && mode_ != "component")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has mode " + mode_
                                     + "; brae has magnitude and component");
        }
        for (const std::string& f : fields_)
        {
            if (f != "U" && !rt.isScalarName(f))
            {
                throw std::runtime_error("brae function objects: `" + name + "` takes the field `" + f
                                         + "`, which is not one brae knows");
            }
        }
        refuseKey(d, name, "location", "only the default (location written) is ported");
        dat_ = rt.datFile(name, "fieldMinMax.dat");
        dat_.header("# Field minima and maxima");
        dat_.header("# Time          field           min             location(min)           processor       "
                    "max             location(max)           processor");
    }

    void saveState(FunctionObjectState& st) const override
    {
        for (const auto& item : latest_)
        {
            const std::string& f = item.first;
            const std::vector<double>& v = item.second;
            st.scalar(name, "min(" + f + ")", v[0], 13);
            st.scalar(name, "max(" + f + ")", v[1], 13);
            st.vector(name, "min(" + f + ")_position", v[2], v[3], v[4]);
            st.vector(name, "max(" + f + ")_position", v[5], v[6], v[7]);
            st.label(name, "min(" + f + ")_cell", static_cast<long long>(v[8]));
            st.label(name, "max(" + f + ")_cell", static_cast<long long>(v[9]));
            st.label(name, "min(" + f + ")_processor", static_cast<long long>(v[10]));
            st.label(name, "max(" + f + ")_processor", static_cast<long long>(v[11]));
        }
    }

    void write(const Step& s, Runtime& rt) override
    {
        const std::vector<vector>& C = rt.svc.geom->C();
        auto vec3 = [&](std::size_t c)
        {
            return "(" + fmtNum(C[c].x, rt.svc.precision) + " " + fmtNum(C[c].y, rt.svc.precision) + " "
                 + fmtNum(C[c].z, rt.svc.precision) + ")";
        };
        for (const std::string& f : fields_)
        {
            auto save = [&](const std::string& label, double mn, double mx,
                            const vector& cmin, const vector& cmax, long long imin, long long imax, int pmin, int pmax)
            {
                latest_[label] = {mn, mx, cmin.x, cmin.y, cmin.z, cmax.x, cmax.y, cmax.z,
                                  static_cast<double>(imin), static_cast<double>(imax),
                                  static_cast<double>(pmin), static_cast<double>(pmax)};
            };
            auto emit = [&](const std::string& label, const std::vector<scalar>& val)
            {
                const Services::Reduce& red = rt.svc.red;
                if (!red.parallel())
                {
                    std::size_t imin = 0, imax = 0;
                    for (std::size_t i = 1; i < val.size(); ++i)
                    {
                        if (val[i] < val[imin]) imin = i;
                        if (val[i] > val[imax]) imax = i;
                    }
                    dat_.row(fmtNum(s.time, rt.svc.precision) + "\t" + label + "\t" + fmtNum(val[imin], rt.svc.precision)
                             + "\t" + vec3(imin) + "\t0\t" + fmtNum(val[imax], rt.svc.precision) + "\t" + vec3(imax) + "\t0");
                    save(label, val[imin], val[imax], C[imin], C[imax], imin, imax, 0, 0);
                    return;
                }
                // fieldMinMax::calcMinMaxFieldType: each rank's extremum (the FIRST cell on a tie; a rank with
                // no cells offers +max / -max and the centre 0), gathered; the winner is the first rank in
                // rank order whose value is STRICTLY better, and its value, location and rank are the row
                const scalar big = std::numeric_limits<scalar>::max();
                std::vector<double> mine = {big, 0, 0, 0, -big, 0, 0, 0, -1, -1};
                if (!val.empty())
                {
                    std::size_t imin = 0, imax = 0;
                    for (std::size_t i = 1; i < val.size(); ++i)
                    {
                        if (val[i] < val[imin]) imin = i;
                        if (val[i] > val[imax]) imax = i;
                    }
                    mine = {val[imin], C[imin].x, C[imin].y, C[imin].z, val[imax], C[imax].x, C[imax].y, C[imax].z,
                            static_cast<double>(imin), static_cast<double>(imax)};
                }
                const std::vector<std::vector<double>> all = red.gatherToMaster(mine);
                std::size_t pmin = 0, pmax = 0;
                for (std::size_t r = 1; r < all.size(); ++r)
                {
                    if (all[r][0] < all[pmin][0]) pmin = r;
                    if (all[pmax][4] < all[r][4]) pmax = r;
                }
                auto at = [&](const std::vector<double>& v, int o)
                {
                    return "(" + fmtNum(v[o], rt.svc.precision) + " " + fmtNum(v[o + 1], rt.svc.precision) + " "
                         + fmtNum(v[o + 2], rt.svc.precision) + ")";
                };
                dat_.row(fmtNum(s.time, rt.svc.precision) + "\t" + label + "\t" + fmtNum(all[pmin][0], rt.svc.precision)
                         + "\t" + at(all[pmin], 1) + "\t" + std::to_string(pmin) + "\t"
                         + fmtNum(all[pmax][4], rt.svc.precision) + "\t" + at(all[pmax], 5) + "\t"
                         + std::to_string(pmax));
                save(label, all[pmin][0], all[pmax][4],
                     vector{all[pmin][1], all[pmin][2], all[pmin][3]},
                     vector{all[pmax][5], all[pmax][6], all[pmax][7]},
                     static_cast<long long>(all[pmin][8]), static_cast<long long>(all[pmax][8]),
                     static_cast<int>(pmin), static_cast<int>(pmax));
            };
            if (f == "U")
            {
                const std::vector<scalar>& ux = rt.svc.src.Ux();
                const std::vector<scalar>& uy = rt.svc.src.Uy();
                const std::vector<scalar>& uz = rt.svc.src.Uz();
                if (mode_ == "magnitude")
                {
                    std::vector<scalar> m(ux.size());
                    for (std::size_t i = 0; i < m.size(); ++i)
                    {
                        m[i] = std::sqrt(ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i]);
                    }
                    emit("mag(U)", m);
                }
                else
                {
                    emit("U.x", ux);
                    emit("U.y", uy);
                    emit("U.z", uz);
                }
            }
            else
            {
                const SF sf = rt.scalarField(f);
                if (sf.isConst())
                {
                    emit(f, std::vector<scalar>(rt.nCells(), sf.c));
                }
                else
                {
                    emit(f, sf.v());
                }
            }
        }
    }

private:
    std::vector<std::string> fields_;
    std::string              mode_;
    DatFile                  dat_;
    std::map<std::string, std::vector<double>> latest_;
};

class ExprField : public Object
{
public:
    ExprField(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        const std::string field = d.wordOr("field", "");
        if (field.empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` names no field");
        }
        const std::string action = d.wordOr("action", "new");
        if (action != "new")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has action " + action
                                     + "; brae has new");
        }
        const auto* e = d.find("expression");
        if (!e || e->empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` has no expression");
        }
        std::string text;
        for (const std::string& t : *e)
        {
            text += t;
        }
        // strip a quote pair the dictionary reader kept
        if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
        {
            text = text.substr(1, text.size() - 2);
        }
        if (d.found("autowrite") && d.switchOr("autowrite", false))
        {
            throw std::runtime_error("brae function objects: `" + name + "` has autowrite true; brae keeps expression "
                                     "fields in memory and does not write them");
        }
        if (!d.found("autowrite") && rt.svc.isMaster)
        {
            noticeDefaulted("functions/" + name + " autowrite", "false: brae does not write expression fields");
        }
        d.find("dimensions");   // read, and unused: the evaluator is not dimensioned
        rt.exprs[field] = text;
        // evaluate once at start-up on zero-size views? No: a syntax error should stop the run now, so parse it
        {
            Scope sc;
            sc.n = 1;
            static const std::vector<scalar> one(1, 1.0);
            sc.scalarField = [&](const std::string& nm, SF& o)
            {
                if (rt.isScalarName(nm) || nm == field)
                {
                    o = sfBorrow(one);
                    return true;
                }
                return false;
            };
            sc.vectorField = [&](const std::string& nm, VF& o)
            {
                if (nm != "U")
                {
                    return false;
                }
                o.x = o.y = o.z = sfBorrow(one);
                return true;
            };
            sc.pos = [&]() { VF v; v.x = v.y = v.z = sfBorrow(one); return v; };
            ExprParser(text, sc).parseScalar();
        }
    }

    void write(const Step&, Runtime&) override {}   // evaluated where it is read
};

class RunTimeControl : public Object
{
public:
    RunTimeControl(const std::string& nm, const FoamDict& d, Runtime&)
    {
        name = nm;
        action_ = d.wordOr("satisfiedAction", "end");
        if (action_ != "end")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has satisfiedAction " + action_
                                     + "; brae has end");
        }
        const FoamDict* conds = d.subDict("conditions");
        if (!conds || conds->subs.empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` has no conditions");
        }
        for (const auto& c : conds->subs)
        {
            Cond cd;
            cd.name = c.first;
            const std::string type = c.second.wordOr("type", "");
            if (type != "minMax")
            {
                throw std::runtime_error("brae function objects: `" + name + "` condition `" + c.first
                                         + "` has type " + type + "; brae has minMax");
            }
            cd.object = c.second.wordOr("functionObject", "");
            cd.fields = readFieldNames(c.second, "fields");
            const std::string mode = c.second.wordOr("mode", "");
            if (mode != "maximum" && mode != "minimum")
            {
                throw std::runtime_error("brae function objects: `" + name + "` condition `" + c.first
                                         + "` has mode " + mode + "; brae has maximum and minimum");
            }
            cd.maximum = (mode == "maximum");
            cd.threshold = c.second.scalarOr("value", std::numeric_limits<scalar>::quiet_NaN());
            if (cd.object.empty() || cd.fields.empty() || std::isnan(cd.threshold))
            {
                throw std::runtime_error("brae function objects: `" + name + "` condition `" + c.first
                                         + "` needs functionObject, fields and value");
            }
            conds_.push_back(cd);
        }
    }

    void saveState(FunctionObjectState& st) const override
    {
        std::string text = name + "\n{\n";
        for (const Cond& c : conds_) text += "    " + c.name + "\n    {\n    }\n";
        text += "}\n";
        st.properties["__control_" + name] = {text};
    }

    bool canEnd() const { return true; }
    bool evaluatesEveryStep() const override { return true; }
    void write(const Step&, Runtime&) override {}

    bool evaluate(const Step&, Runtime& rt) override
    {
        const int prec = std::max(rt.svc.precision, 6);
        const bool say = log && rt.svc.isMaster;   // Info is the master's
        if (say)
        {
            std::printf("runTimeControl %s output:\n", name.c_str());
        }
        bool met = true;
        for (const Cond& c : conds_)
        {
            // minMaxCondition::apply skips an unavailable saved result (and warns), rather than applying
            // the condition to zero. This matters on the first step of a restart whose monitor is not due.
            bool available = true;
            for (const std::string& f : c.fields) available = available && rt.hasResult(c.object, f);
            if (!available)
            {
                met = false;
                if (say)
                {
                    std::printf("    minMax: %s %s: saved result unavailable; condition not applied\n",
                                c.maximum ? "maximum" : "minimum", c.fields.front().c_str());
                }
                continue;
            }
            scalar value = c.maximum ? -std::numeric_limits<scalar>::max() : std::numeric_limits<scalar>::max();
            for (const std::string& f : c.fields)
            {
                const scalar v = rt.result(c.object, f);
                value = c.maximum ? std::max(value, v) : std::min(value, v);
            }
            const bool satisfied = c.maximum ? (value > c.threshold) : (value < c.threshold);
            met = met && satisfied;
            if (say)
            {
                std::printf("    minMax: %s %s: value = %s, threshold value = %s, satisfied = %s\n",
                            c.maximum ? "maximum" : "minimum", c.fields.front().c_str(), fmtNum(value, prec).c_str(),
                            fmtNum(c.threshold, prec).c_str(), satisfied ? "true" : "false");
            }
        }
        if (say)
        {
            std::printf(met ? "    Conditions met\n" : "    Conditions not met\n");
            std::fflush(stdout);
        }
        return met;   // satisfiedAction end
    }

private:
    struct Cond
    {
        std::string               name, object;
        std::vector<std::string>  fields;
        bool                      maximum = true;
        scalar                    threshold = 0;
    };
    std::string       action_;
    std::vector<Cond> conds_;
};

// ---------------------------------------------------------------------------------------------------------------
// the list
// ---------------------------------------------------------------------------------------------------------------

class FunctionObjectList
{
public:
    // Builds the list from controlDict `functions`; null when it has none that brae runs.
    static std::shared_ptr<FunctionObjectList> read(const FoamDict& controlDict, Services svc,
                                                    const std::string& startDir = std::string())
    {
        const FoamDict* fns = controlDict.subDict("functions");
        if (!fns || fns->subs.empty())
        {
            return nullptr;
        }
        std::shared_ptr<FunctionObjectList> list(new FunctionObjectList(std::move(svc)));
        Runtime& rt = list->rt_;
        // functionObjectList::createPropertiesDict reads the start time's file. Only scalar results are needed
        // by the runTimeControl conditions brae runs; vector/label results are still persisted for OpenFOAM.
        if (!startDir.empty())
        {
            const std::string statePath = startDir + "/uniform/functionObjects/functionObjectProperties";
            if (std::filesystem::exists(statePath))
            {
                const FoamDict state = readDict(statePath);
                if (const FoamDict* saved = state.subDict("cdf7e925f5746741c316f5fbcf39ad0dfca90775"))
                {
                    for (const auto& object : saved->subs)
                    {
                        if (const FoamDict* scalars = object.second.subDict("scalar"))
                        {
                            for (const auto& value : scalars->leaves)
                            {
                                const std::string& key = value.first;
                                const std::size_t open = key.find('('), comma = key.find(','), close = key.rfind(')');
                                if (open == std::string::npos || close == std::string::npos) continue;
                                const std::string op = key.substr(0, open);
                                const std::string field = comma == std::string::npos
                                    ? key.substr(open + 1, close - open - 1)
                                    : key.substr(comma + 1, close - comma - 1);
                                rt.results[object.first][op + "(" + field + ")"] =
                                    value.second.empty() ? scalar(0) : std::strtod(value.second[0].c_str(), nullptr);
                            }
                        }
                    }
                }
            }
        }
        std::vector<std::string> notRun;
        for (const auto& e : fns->subs)
        {
            const FoamDict& d = e.second;
            const std::string type = d.wordOr("type", "");
            const bool enabled = d.switchOr("enabled", true);
            d.find("libs");
            std::unique_ptr<Object> o;
            if (type == "volFieldValue" || type == "surfaceFieldValue" || type == "probes" || type == "fieldMinMax"
             || type == "exprField" || type == "runTimeControl")
            {
                if (!enabled)
                {
                    list->disabled_.push_back(e.first);
                    continue;
                }
                refuseKey(d, e.first, "timeStart", "an active window (timeStart/timeEnd) is not ported");
                refuseKey(d, e.first, "timeEnd", "an active window (timeStart/timeEnd) is not ported");
                if (type == "volFieldValue")
                {
                    auto v = std::unique_ptr<VolFieldValue>(new VolFieldValue(e.first, d, rt));
                    v->open(rt);
                    o = std::move(v);
                }
                else if (type == "surfaceFieldValue")
                {
                    auto v = std::unique_ptr<SurfaceFieldValue>(new SurfaceFieldValue(e.first, d, rt));
                    v->open(rt);
                    o = std::move(v);
                }
                else if (type == "probes")
                {
                    o.reset(new Probes(e.first, d, rt));
                }
                else if (type == "fieldMinMax")
                {
                    o.reset(new FieldMinMax(e.first, d, rt));
                }
                else if (type == "exprField")
                {
                    o.reset(new ExprField(e.first, d, rt));
                }
                else
                {
                    o.reset(new RunTimeControl(e.first, d, rt));
                    list->canStop_ = true;
                }
                o->log = d.switchOr("log", true);
                o->ctl = Control::read(d, e.first, "write");
                list->objects_.push_back(std::move(o));
                list->ran_.push_back(e.first);
            }
            else
            {
                notRun.push_back(e.first + " (" + type + ")");
            }
        }
        std::string line = "  function objects: run";
        for (const std::string& n : list->ran_) line += " `" + n + "`";
        if (list->ran_.empty()) line += " none";
        if (!list->disabled_.empty())
        {
            line += "; disabled";
            for (const std::string& n : list->disabled_) line += " `" + n + "`";
        }
        if (list->rt_.svc.isMaster) std::printf("%s\n", line.c_str());
        if (!notRun.empty() && list->rt_.svc.isMaster)
        {
            std::string nr;
            for (const std::string& n : notRun) nr += " " + n;
            noticeIgnored("controlDict functions", "not run by brae (a type it does not port):" + nr);
        }
        if (list->objects_.empty())
        {
            return nullptr;
        }
        return list;
    }

    bool canStop() const { return canStop_; }

    // One call per completed step. True when a runTimeControl asks the run to end at this step.
    bool execute(const Step& s)
    {
        ++rt_.stamp;
        if (rt_.svc.src.newStep)
        {
            rt_.svc.src.newStep();
        }
        rt_.beginDeviceStep();
        for (auto& o : objects_)
        {
            if (!o->evaluatesEveryStep() && o->ctl.due(s))
            {
                o->write(s, rt_);
            }
        }
        rt_.flushDeviceIntegrals();
        for (auto& o : objects_) o->afterDeviceBatch(s, rt_);
        bool stop = false;
        for (auto& o : objects_)
        {
            if (o->evaluatesEveryStep() && o->evaluate(s, rt_))
            {
                stop = true;
            }
        }
        return stop;
    }

    // Latest results and runTimeControl properties for the time directory's restart state. Values are already
    // globally reduced / owner-selected, so every rank holds identical state and every rank writes it.
    FunctionObjectState state() const
    {
        FunctionObjectState st;
        for (const auto& o : objects_) o->saveState(st);
        return st;
    }

private:
    explicit FunctionObjectList(Services svc) : rt_(std::move(svc)) {}

    Runtime                               rt_;
    std::vector<std::unique_ptr<Object>>  objects_;
    std::vector<std::string>              ran_, disabled_;
    bool                                  canStop_ = false;
};

// What both time loops call once, before the first step: the list for the case's own controlDict, or null when
// it has no function object brae runs. `src` is the loop's way of showing its fields on the host.
inline std::shared_ptr<FunctionObjectList> createFunctionObjects(
    const std::string&                                caseDir,
    const std::string&                                startDir,
    const PrimitiveMesh&                              m,
    const FvGeometry&                                 g,
    const std::vector<FvPatch>&                       patches,
    const std::map<std::string, std::vector<label>>&  zones,
    const std::string&                                phaseField,
    bool                                              meshChanges,
    const Source&                                     src,
    bool                                              isMaster)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    Services s;
    s.caseDir = caseDir;
    s.startName = std::filesystem::path(startDir).filename().string();
    s.precision = std::max(controlDict.intOr("writePrecision", 6), 6);
    s.phaseField = phaseField;
    s.mesh = &m;
    s.geom = &g;
    s.patches = &patches;
    s.cellZones = &zones;
    s.src = src;
    s.isMaster = isMaster;
    s.red = Services::Reduce::procComm();   // identity without a multi-rank communicator
    std::shared_ptr<FunctionObjectList> list = FunctionObjectList::read(controlDict, s, startDir);
    if (list && meshChanges)
    {
        throw std::runtime_error(
            "brae interFoam: the case has function objects brae runs and a mesh that moves or refines. They are "
            "evaluated on the start mesh's cells and faces; on a changing mesh that would be a wrong answer, so "
            "the run stops here. Remove the function objects, or run the case with a static mesh.");
    }
    return list;
}

}   // namespace fo
}   // namespace brae
