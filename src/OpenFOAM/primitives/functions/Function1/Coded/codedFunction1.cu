// OpenFOAM's `type coded;` Function1<scalar> -- see codedFunction1.cuh for the provenance and the scope.
#include "codedFunction1.cuh"
#include "coded_library.cuh"
#include "io_precision.cuh"   // ioPrecision(): OpenFOAM's Info takes writePrecision, so does the shim's
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace brae {

namespace {

// The Foam scope a coded scalar Function1 reaches, transcribed. Every entry names the OpenFOAM file it
// comes from; anything absent is a compile error the caller reports, never a stand-in.
const char* kShim = R"SHIM(// brae's Foam scope for a coded Function1 -- generated, do not edit.
#pragma once
#include <cmath>
#include <iostream>

namespace Foam
{

typedef double scalar;        // WM_DP (etc/bashrc WM_PRECISION_OPTION=DP)
typedef int    label;         // WM_LABEL_SIZE=32

// db/IOstreams/IOstreams/Ostream.H:55
constexpr char nl = '\n';

namespace braeDetail
{
    inline int& infoPrecision() { static int p = 6; return p; }
}

// Foam::Info, a messageStream onto stdout, scalars at the stream precision (writePrecision). The
// solver's own precision is set on std::cout for the one write and put back, so the host's other
// output is untouched.
class braeInfoStream
{
public:
    template<class T>
    braeInfoStream& operator<<(const T& v)
    {
        const std::streamsize old = std::cout.precision(braeDetail::infoPrecision());
        std::cout << v;
        std::cout.precision(old);
        return *this;
    }
    braeInfoStream& operator<<(braeInfoStream& (*manip)(braeInfoStream&)) { return manip(*this); }
};
static braeInfoStream Info;

// db/IOstreams/IOstreams/Ostream.H:409-413 -- a newline, then a flush
inline braeInfoStream& endl(braeInfoStream& os) { std::cout << '\n'; std::cout.flush(); return os; }

// primitives/Scalar/doubleScalar/doubleScalar.H:76-89 and the transFunc macro at :93-97
inline scalar mag(const scalar s) { return ::fabs(s); }
inline scalar hypot(const scalar x, const scalar y) { return ::hypot(x, y); }
inline scalar atan2(const scalar y, const scalar x) { return ::atan2(y, x); }
#define BRAE_TRANSFUNC(func) inline scalar func(const scalar s) { return ::func(s); }
// primitives/Scalar/Scalar.H:187-211
BRAE_TRANSFUNC(sqrt)
BRAE_TRANSFUNC(cbrt)
BRAE_TRANSFUNC(exp)
BRAE_TRANSFUNC(log)
BRAE_TRANSFUNC(log10)
BRAE_TRANSFUNC(sin)
BRAE_TRANSFUNC(cos)
BRAE_TRANSFUNC(tan)
BRAE_TRANSFUNC(asin)
BRAE_TRANSFUNC(acos)
BRAE_TRANSFUNC(atan)
BRAE_TRANSFUNC(sinh)
BRAE_TRANSFUNC(cosh)
BRAE_TRANSFUNC(tanh)
BRAE_TRANSFUNC(asinh)
BRAE_TRANSFUNC(acosh)
BRAE_TRANSFUNC(atanh)
BRAE_TRANSFUNC(erf)
BRAE_TRANSFUNC(erfc)
BRAE_TRANSFUNC(lgamma)
BRAE_TRANSFUNC(tgamma)
#undef BRAE_TRANSFUNC

// primitives/ints/int/int.H:50-62 (MAXMIN) and primitives/Scalar/doubleFloat.H:38-58 (MAXMINPOW), the
// double overloads
#define BRAE_MAXMINPOW(RetType, Type1, Type2)                                   \
inline RetType min(const Type1 s1, const Type2 s2) { return (s1 < s2) ? s1 : s2; } \
inline RetType max(const Type1 s1, const Type2 s2) { return (s2 < s1) ? s1 : s2; } \
inline double pow(const Type1 base, const Type2 expon) { return ::pow(double(base), double(expon)); }
BRAE_MAXMINPOW(double, double, double)
BRAE_MAXMINPOW(double, double, int)
BRAE_MAXMINPOW(double, int, double)
BRAE_MAXMINPOW(double, double, long)
BRAE_MAXMINPOW(double, long, double)
#undef BRAE_MAXMINPOW

// primitives/Scalar/Scalar.H:236-376
inline scalar sign(const scalar s) noexcept { return (s >= 0)? 1: -1; }
inline scalar pos(const scalar s) noexcept { return (s > 0)? 1: 0; }
inline scalar pos0(const scalar s) noexcept { return (s >= 0)? 1: 0; }
inline scalar neg(const scalar s) noexcept { return (s < 0)? 1: 0; }
inline scalar neg0(const scalar s) noexcept { return (s <= 0)? 1: 0; }
inline scalar posPart(const scalar s) noexcept { return (s > 0)? s: 0; }
inline scalar negPart(const scalar s) noexcept { return (s < 0)? s: 0; }
inline scalar clamp(const scalar val, const scalar lower, const scalar upper)
{
    return (val < lower) ? lower : (upper < val) ? upper : val;
}
inline scalar limit(const scalar s1, const scalar s2) { return (mag(s1) < mag(s2)) ? s1: 0.0; }
inline scalar minMod(const scalar s1, const scalar s2) { return (mag(s1) < mag(s2)) ? s1: s2; }
inline constexpr scalar lerp(const scalar a, const scalar b, const scalar t) { return (scalar{1}-t)*a + t*b; }
inline scalar magSqr(const scalar s) { return s*s; }
inline scalar sqr(const scalar s) { return s*s; }
inline scalar pow3(const scalar s) { return s*sqr(s); }
inline scalar pow4(const scalar s) { return sqr(sqr(s)); }
inline scalar pow5(const scalar s) { return s*pow4(s); }
inline scalar pow6(const scalar s) { return pow3(sqr(s)); }
inline scalar pow025(const scalar s) { return sqrt(sqrt(s)); }
inline scalar inv(const scalar s) { return 1.0/s; }

// global/constants/mathematical/mathematicalConstants.H:54-57
namespace constant
{
namespace mathematical
{
    constexpr scalar e(M_E);
    constexpr scalar pi(M_PI);
    constexpr scalar twoPi(2*M_PI);
    constexpr scalar piByTwo(0.5*M_PI);
}
}

// global/constants/unitConversion.H:48-106 -- codedFunction1Template.C includes it
inline constexpr scalar degToRad(const scalar deg) noexcept { return (deg*M_PI/180.0); }
inline constexpr scalar radToDeg(const scalar rad) noexcept { return (rad*180.0/M_PI); }
inline constexpr scalar degToRad() noexcept { return (M_PI/180.0); }
inline constexpr scalar radToDeg() noexcept { return (180.0/M_PI); }
inline constexpr scalar rpmToRads(const scalar rpm) noexcept { return (rpm*M_PI/30.0); }
inline constexpr scalar radsToRpm(const scalar rads) noexcept { return (rads*30.0/M_PI); }
inline constexpr scalar rpmToRads() noexcept { return (M_PI/30.0); }
inline constexpr scalar radsToRpm() noexcept { return (30.0/M_PI); }
inline constexpr scalar atmToPa(const scalar atm) noexcept { return (atm*101325.0); }
inline constexpr scalar barToPa(const scalar bar) noexcept { return (bar*100000.0); }

} // End namespace Foam
)SHIM";

} // namespace


const char* codedFunction1ScalarShim()
{
    return kShim;
}


CodedFunction1::CodedFunction1(CodedFunction1Spec spec)
    : spec_(std::move(spec))
{
    const std::string what = "brae: coded Function1 '" + spec_.name + "' (" + spec_.origin + ")";
    if (!spec_.unsupportedKeys.empty())
        throw std::runtime_error(
            what + " carries " + spec_.unsupportedKeys + ", which OpenFOAM compiles against its own "
            "headers and libraries; brae compiles the `code` body against a scalar shim only. Refusing "
            "rather than dropping them.");
    const std::string body = codedLibrary::trim(spec_.code);
    if (body.empty())
        throw std::runtime_error(
            what + " has no `code`. OpenFOAM refuses the same (CodedFunction1.C:100-106).");
    if (spec_.code.find('$') != std::string::npos)
        throw std::runtime_error(
            what + " uses `$` in its code. OpenFOAM expands those against the dictionary before compiling "
            "(dynamicCodeContext); brae does not reproduce that scope. Refusing rather than compiling a "
            "different body.");
    codedLibrary::refuseRoot(what);

    const std::string id = codedLibrary::identifier(spec_.name);
    const std::string cls = id + "Function1_scalar";
    std::ostringstream tu;
    tu << "// Generated by brae from " << spec_.origin << ". Regenerated whenever the code changes.\n"
       << "#include \"braeCodedFunction1.H\"\n\n"
       << "namespace Foam\n{\nnamespace Function1Types\n{\n\n"
       << "// The class codedFunction1Template.H declares, so the body's unqualified names resolve through\n"
       << "// the same scopes (Foam::Function1Types, then Foam).\n"
       << "class " << cls << "\n{\npublic:\n    scalar value(const scalar x) const;\n};\n\n"
       << "} // End namespace Function1Types\n} // End namespace Foam\n\n"
       << "Foam::scalar Foam::Function1Types::" << cls << "::value\n(\n    const scalar x\n) const\n{\n"
       << "#line 1 \"" << codedLibrary::cLiteral(spec_.origin) << "\"\n"
       << spec_.code << "\n}\n\n";
    const std::string key = codedLibrary::contentKey(std::string(kShim) + tu.str());
    tu << "extern \"C\" double brae_coded_function1_value_" << key << "(double x)\n{\n"
       << "    static const Foam::Function1Types::" << cls << " fn{};\n    return fn.value(x);\n}\n\n"
       << "extern \"C\" void brae_coded_function1_precision_" << key << "(int p)\n{\n"
       << "    Foam::braeDetail::infoPrecision() = p;\n}\n";

    const char* envDir = std::getenv("BRAE_DYNAMIC_CODE_DIR");
    const std::filesystem::path dir = std::filesystem::path(envDir && *envDir ? envDir : spec_.codeDir)
                                    / (id + "_" + key);
    codedLibrary::Build b;
    b.what = what;
    b.dir = dir.string();
    b.shimFile = "braeCodedFunction1.H";
    b.shimText = kShim;
    b.sourceFile = id + ".C";
    b.sourceText = tu.str();
    b.libStem = id + "_" + key;
    b.key = key;
    b.shimDescription = "the scalar shim";
    void* handle = codedLibrary::compileAndLoad(b);
    const std::string lib = (dir / ("lib" + b.libStem + ".so")).string();

    fn_ = reinterpret_cast<double (*)(double)>(
        codedLibrary::symbol(handle, "brae_coded_function1_value_" + key, what, lib));
    auto setPrecision = reinterpret_cast<void (*)(int)>(
        codedLibrary::symbol(handle, "brae_coded_function1_precision_" + key, what, lib));
    setPrecision(ioPrecision());
}


scalar CodedFunction1::value(scalar x) const
{
    return fn_(x);
}

} // namespace brae
