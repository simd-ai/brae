// OpenFOAM's `type coded;` PatchFunction1<scalar> -- see codedPatchFunction1.cuh for the scope.
#include "codedPatchFunction1.cuh"
#include "codedFunction1.cuh"      // codedFunction1ScalarShim(): the scalar scope this one extends
#include "coded_library.cuh"
#include "io_precision.cuh"
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace brae {

namespace {

// OpenFOAM's field types, as far as a coded PatchFunction1<scalar> reaches them, transcribed. Every entry
// names the OpenFOAM file it comes from; anything absent is a compile error the caller reports.
const char* kFieldShim = R"SHIM(
// ---- brae's field scope for a coded PatchFunction1 -- generated, do not edit.
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Foam
{

// primitives/traits/zero.H:110
class zero {};
static constexpr const zero Zero;

// primitives/Vector/Vector.H, VectorI.H: the three components and the algebra on them
template<class Cmpt>
class Vector
{
public:
    Vector() : v_{Cmpt(0), Cmpt(0), Cmpt(0)} {}
    Vector(const zero&) : v_{Cmpt(0), Cmpt(0), Cmpt(0)} {}
    Vector(const Cmpt& vx, const Cmpt& vy, const Cmpt& vz) : v_{vx, vy, vz} {}
    const Cmpt& x() const { return v_[0]; }
    const Cmpt& y() const { return v_[1]; }
    const Cmpt& z() const { return v_[2]; }
    Cmpt& x() { return v_[0]; }
    Cmpt& y() { return v_[1]; }
    Cmpt& z() { return v_[2]; }
    // VectorI.H:153-158
    Cmpt inner(const Vector<Cmpt>& v2) const { return (x()*v2.x() + y()*v2.y() + z()*v2.z()); }
    // VectorI.H:83-97
    scalar magSqr() const { return (x()*x() + y()*y() + z()*z()); }
    scalar mag() const { return ::sqrt(magSqr()); }
private:
    Cmpt v_[3];
};
// VectorI.H:293-296
template<class Cmpt>
inline Cmpt operator&(const Vector<Cmpt>& v1, const Vector<Cmpt>& v2) { return v1.inner(v2); }
// VectorSpaceI.H, componentwise
template<class Cmpt>
inline Vector<Cmpt> operator+(const Vector<Cmpt>& a, const Vector<Cmpt>& b)
{ return Vector<Cmpt>(a.x() + b.x(), a.y() + b.y(), a.z() + b.z()); }
template<class Cmpt>
inline Vector<Cmpt> operator-(const Vector<Cmpt>& a, const Vector<Cmpt>& b)
{ return Vector<Cmpt>(a.x() - b.x(), a.y() - b.y(), a.z() - b.z()); }
template<class Cmpt>
inline Vector<Cmpt> operator-(const Vector<Cmpt>& a) { return Vector<Cmpt>(-a.x(), -a.y(), -a.z()); }
template<class Cmpt>
inline Vector<Cmpt> operator*(const scalar s, const Vector<Cmpt>& a)
{ return Vector<Cmpt>(s*a.x(), s*a.y(), s*a.z()); }
template<class Cmpt>
inline Vector<Cmpt> operator*(const Vector<Cmpt>& a, const scalar s)
{ return Vector<Cmpt>(a.x()*s, a.y()*s, a.z()*s); }
template<class Cmpt>
inline Vector<Cmpt> operator/(const Vector<Cmpt>& a, const scalar s)
{ return Vector<Cmpt>(a.x()/s, a.y()/s, a.z()/s); }
template<class Cmpt>
inline scalar magSqr(const Vector<Cmpt>& a) { return a.magSqr(); }
template<class Cmpt>
inline scalar mag(const Vector<Cmpt>& a) { return a.mag(); }

// primitives/Vector/floats/vector.H, primitives/Vector/point/point.H
typedef Vector<scalar> vector;
typedef vector point;

namespace braeDetail
{
    template<class T> inline T zeroOf() { return T(Zero); }
    template<> inline scalar zeroOf<scalar>() { return 0; }
    template<> inline label zeroOf<label>() { return 0; }
}

// memory/tmp/tmp.H: a managed pointer to a field. OpenFOAM's tmp can also wrap a const reference;
// nothing in this scope hands one out, so only the owning form is carried.
template<class T>
class tmp
{
public:
    tmp() = default;
    explicit tmp(T* p) : p_(p) {}
    // tmp.H New: construct the managed object from the arguments
    template<class... Args>
    static tmp<T> New(Args&&... args) { return tmp<T>(new T(std::forward<Args>(args)...)); }
    T& ref() const { return *p_; }
    const T& cref() const { return *p_; }
    const T& operator()() const { return *p_; }
    T* operator->() const { return p_.get(); }
    bool good() const noexcept { return bool(p_); }
    bool valid() const noexcept { return bool(p_); }
    operator const T&() const { return *p_; }
private:
    std::shared_ptr<T> p_;
};

// fields/Fields/Field/Field.H: a list of values with the field algebra
template<class Type>
class Field
{
public:
    Field() = default;
    explicit Field(const label n) : v_(static_cast<std::size_t>(n)) {}
    Field(const label n, const zero&) : v_(static_cast<std::size_t>(n), braeDetail::zeroOf<Type>()) {}
    Field(const label n, const Type& t) : v_(static_cast<std::size_t>(n), t) {}
    Field(const tmp<Field<Type>>& tf) : v_(tf().v_) {}
    label size() const noexcept { return static_cast<label>(v_.size()); }
    bool empty() const noexcept { return v_.empty(); }
    Type& operator[](const label i) { return v_[static_cast<std::size_t>(i)]; }
    const Type& operator[](const label i) const { return v_[static_cast<std::size_t>(i)]; }
    typename std::vector<Type>::iterator begin() { return v_.begin(); }
    typename std::vector<Type>::iterator end() { return v_.end(); }
    typename std::vector<Type>::const_iterator begin() const { return v_.begin(); }
    typename std::vector<Type>::const_iterator end() const { return v_.end(); }
    void operator=(const Type& t) { for (Type& e : v_) e = t; }
    void operator=(const zero&) { for (Type& e : v_) e = braeDetail::zeroOf<Type>(); }
private:
    std::vector<Type> v_;
};
typedef Field<scalar> scalarField;
typedef Field<vector> vectorField;
typedef Field<point> pointField;

// include/stdFoam.H:353
#define forAll(list, i) for (Foam::label i=0; i<(list).size(); ++i)

// fields/Fields/Field/FieldFunctions.H:405 PRODUCT_OPERATOR(innerProduct, &, dot): f1[i] & s2, and s1 & f2[i]
inline tmp<Field<scalar>> operator&(const Field<vector>& f1, const vector& s2)
{
    auto tres = tmp<Field<scalar>>::New(f1.size());
    Field<scalar>& res = tres.ref();
    forAll(res, i) { res[i] = f1[i] & s2; }
    return tres;
}
inline tmp<Field<scalar>> operator&(const vector& s1, const Field<vector>& f2)
{
    auto tres = tmp<Field<scalar>>::New(f2.size());
    Field<scalar>& res = tres.ref();
    forAll(res, i) { res[i] = s1 & f2[i]; }
    return tres;
}
inline tmp<Field<scalar>> operator&(const tmp<Field<vector>>& tf1, const vector& s2) { return tf1() & s2; }
inline tmp<Field<scalar>> operator&(const vector& s1, const tmp<Field<vector>>& tf2) { return s1 & tf2(); }

// meshes/polyMesh/polyPatches/polyPatch/polyPatch.H: the patch as the snippet sees it -- its name, its
// size and its face centres (polyPatch.C:308-311, the mesh's face centres, which ACMI scaling does not
// touch)
class polyPatch
{
public:
    polyPatch(const std::string& n, const Field<point>& cf) : name_(n), Cf_(cf) {}
    const std::string& name() const { return name_; }
    label size() const { return Cf_.size(); }
    const Field<point>& faceCentres() const { return Cf_; }
private:
    std::string name_;
    const Field<point>& Cf_;
};

// db/Time/TimeState.H: value() and timeOutputValue() (TimeStateI.H:31-34). NOT timeIndex(): OpenFOAM's
// continues across a restart from <start>/uniform/time, which brae does not read, so a snippet that asks
// for it is a compile error and refused rather than handed a count from the wrong origin.
class Time
{
public:
    Time(const scalar v, const scalar vOut) : v_(v), vOut_(vOut) {}
    scalar value() const { return v_; }
    scalar timeOutputValue() const { return vOut_; }
private:
    scalar v_;
    scalar vOut_;
};

} // End namespace Foam
)SHIM";

} // namespace


CodedPatchFunction1::CodedPatchFunction1(CodedPatchFunction1Spec spec)
    : spec_(std::move(spec))
{
    const std::string what = "brae: coded PatchFunction1 '" + spec_.name + "' (" + spec_.origin + ")";
    if (!spec_.unsupportedKeys.empty())
    {
        throw std::runtime_error(
            what + " carries " + spec_.unsupportedKeys + ", which OpenFOAM compiles against its own "
            "headers and libraries; brae compiles the `code` body against a shim of OpenFOAM's scope "
            "only. Refusing rather than dropping them.");
    }
    if (codedLibrary::trim(spec_.code).empty())
    {
        throw std::runtime_error(
            what + " has no `code`. OpenFOAM refuses the same (CodedField.C:100-107).");
    }
    if (spec_.code.find('$') != std::string::npos)
    {
        throw std::runtime_error(
            what + " uses `$` in its code. OpenFOAM expands those against the dictionary before compiling "
            "(dynamicCodeContext); brae does not reproduce that scope. Refusing rather than compiling a "
            "different body.");
    }
    codedLibrary::refuseRoot(what);

    const std::string shim = std::string(codedFunction1ScalarShim()) + kFieldShim;
    const std::string id = codedLibrary::identifier(spec_.name);
    const std::string cls = id + "PatchFunction1scalar";
    std::ostringstream tu;
    tu << "// Generated by brae from " << spec_.origin << ". Regenerated whenever the code changes.\n"
       << "#include \"braeCodedPatchFunction1.H\"\n\n"
       << "namespace Foam\n{\nnamespace PatchFunction1Types\n{\n\n"
       << "// The class codedPatchFunction1Template.H declares, so the body's unqualified names resolve\n"
       << "// through the same scopes (Foam::PatchFunction1Types, then Foam), and this->patch() and\n"
       << "// this->time() are what patchFunction1Base gives it.\n"
       << "class " << cls << "\n{\npublic:\n"
       << "    " << cls << "(const polyPatch& pp, const Time& t) : pp_(pp), t_(t) {}\n"
       << "    const polyPatch& patch() const { return pp_; }\n"
       << "    const Time& time() const { return t_; }\n"
       << "    tmp<Field<scalar>> value(const scalar x) const;\n"
       << "private:\n    const polyPatch& pp_;\n    const Time& t_;\n};\n\n"
       << "} // End namespace PatchFunction1Types\n} // End namespace Foam\n\n"
       << "Foam::tmp<Foam::Field<Foam::scalar>> Foam::PatchFunction1Types::" << cls
       << "::value\n(\n    const scalar x\n) const\n{\n"
       << "#line 1 \"" << codedLibrary::cLiteral(spec_.origin) << "\"\n"
       << spec_.code << "\n}\n\n";
    const std::string key = codedLibrary::contentKey(shim + tu.str());
    tu << "extern \"C\" int brae_coded_patchfunction1_value_" << key
       << "(double x, const char* name, int n, const double* cf, double tv, double tOut, double* out)\n"
       << "{\n"
       << "    Foam::Field<Foam::point> Cf(n);\n"
       << "    for (int i = 0; i < n; ++i) { Cf[i] = Foam::point(cf[3*i], cf[3*i + 1], cf[3*i + 2]); }\n"
       << "    const Foam::polyPatch pp(name, Cf);\n"
       << "    const Foam::Time t(tv, tOut);\n"
       << "    const Foam::PatchFunction1Types::" << cls << " fn(pp, t);\n"
       << "    const Foam::tmp<Foam::Field<Foam::scalar>> r = fn.value(x);\n"
       << "    if (!r.good() || r().size() != n) { return r.good() ? int(r().size()) : -1; }\n"
       << "    for (int i = 0; i < n; ++i) { out[i] = r()[i]; }\n"
       << "    return n;\n"
       << "}\n\n"
       << "extern \"C\" void brae_coded_patchfunction1_precision_" << key << "(int p)\n{\n"
       << "    Foam::braeDetail::infoPrecision() = p;\n}\n";

    const char* envDir = std::getenv("BRAE_DYNAMIC_CODE_DIR");
    const std::filesystem::path dir = std::filesystem::path(envDir && *envDir ? envDir : spec_.codeDir)
                                    / (id + "_" + key);
    codedLibrary::Build b;
    b.what = what;
    b.dir = dir.string();
    b.shimFile = "braeCodedPatchFunction1.H";
    b.shimText = shim;
    b.sourceFile = id + ".C";
    b.sourceText = tu.str();
    b.libStem = id + "_" + key;
    b.key = key;
    b.shimDescription = "the scalar and field shim";
    void* handle = codedLibrary::compileAndLoad(b);
    const std::string lib = (dir / ("lib" + b.libStem + ".so")).string();

    fn_ = reinterpret_cast<int (*)(double, const char*, int, const double*, double, double, double*)>(
        codedLibrary::symbol(handle, "brae_coded_patchfunction1_value_" + key, what, lib));
    auto setPrecision = reinterpret_cast<void (*)(int)>(
        codedLibrary::symbol(handle, "brae_coded_patchfunction1_precision_" + key, what, lib));
    setPrecision(ioPrecision());
}


std::vector<scalar> CodedPatchFunction1::value(
    scalar x,
    const std::string& patchName,
    const std::vector<vector>& faceCentres,
    scalar timeValue,
    scalar timeOutputValue) const
{
    const int n = static_cast<int>(faceCentres.size());
    std::vector<double> cf(3*faceCentres.size());
    for (std::size_t i = 0; i < faceCentres.size(); ++i)
    {
        cf[3*i] = faceCentres[i].x;
        cf[3*i + 1] = faceCentres[i].y;
        cf[3*i + 2] = faceCentres[i].z;
    }
    std::vector<double> out(faceCentres.size());
    const int got = fn_(x, patchName.c_str(), n, cf.data(), timeValue, timeOutputValue, out.data());
    if (got != n)
    {
        throw std::runtime_error(
            "brae: coded PatchFunction1 '" + spec_.name + "' (" + spec_.origin + ") returned " +
            (got < 0 ? std::string("no field") : std::to_string(got) + " values") + " for patch '" +
            patchName + "' of " + std::to_string(n) + " faces. OpenFOAM would carry the wrong-sized field "
            "into the mask and fail there; brae refuses here.");
    }
    return std::vector<scalar>(out.begin(), out.end());
}

} // namespace brae
