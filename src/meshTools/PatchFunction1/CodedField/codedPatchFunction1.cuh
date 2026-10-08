#pragma once
// OpenFOAM's `type coded;` PatchFunction1<scalar>, compiled and loaded at run time -- HOST ONLY.
//
// provenance:
//   openfoam:
//     class:    Foam::PatchFunction1Types::CodedField<Type>
//     file:     src/meshTools/PatchFunction1/CodedField/CodedField.C
//               :94-107  prepare() refuses an empty `code`
//               :158     redirectName_ defaults to the entry name
//               :238-249 value(x) updates the library, then redirects to its value(x)
//     template: etc/codeTemplates/dynamicCode/codedPatchFunction1Template.C -- `code` is pasted as the
//               BODY of `tmp<Field<scalar>> <name>PatchFunction1scalar::value(const scalar x) const`, a
//               member of a PatchFunction1, so `this->patch()` is the polyPatch and `this->time()` the
//               Time (patchFunction1Base.H:140)
//   brae:
//     reference: this file; the compile-and-load half is coded_library.cuh, the scalar scope
//                codedFunction1.cuh's shim, which this one extends with OpenFOAM's field types
//     users:     cyclicACMI's `scale` (cyclic_acmi_cpp.cuh), which evaluates it once per time step
//     tests:     tests/interfoam_leakage_vs_openfoam.sh against real OpenFOAM on RAS/damBreakLeakage
//
// WHAT THE SNIPPET SEES: the scalar shim, plus Vector/vector/point, Field/scalarField/vectorField,
// tmp with New/ref/cref/operator(), Zero, forAll, the field-vector inner product, and a polyPatch whose
// name(), size() and faceCentres() are the patch's, and a Time whose value() and timeOutputValue() are
// the solver's -- not timeIndex(), which OpenFOAM continues across a restart from <start>/uniform/time.
// Each is transcribed with its OpenFOAM file. faceAreas() is NOT in the scope: on a cyclicACMI patch
// OpenFOAM returns the areas the mask scaled, which depend on where in the step the call lands. Anything the shim lacks is a compile error, refused with the compiler's output.
#include "cf_types.cuh"
#include <string>
#include <vector>

namespace brae {

struct CodedPatchFunction1Spec
{
    std::string name;       // the dictionary's `name`, defaulting to the entry name (CodedField.C:158)
    std::string code;       // the verbatim #{ ... #} body
    std::string origin;     // "constant/polyMesh/boundary: patch 'coupled_half0', scale"
    std::string codeDir;    // where the generated source and library are written
    // Keys OpenFOAM would compile in and brae cannot -- non-empty means refuse, naming them.
    std::string unsupportedKeys;
};

class CodedPatchFunction1
{
public:
    // Compiles (or reuses the library a previous run built from the same code) and loads it, as
    // OpenFOAM's constructor does.
    explicit CodedPatchFunction1(CodedPatchFunction1Spec spec);

    // The loaded value(x), for one patch: its name and face centres, and the time the snippet reads
    // through this->time(). Throws if the snippet returns a field of the wrong length.
    std::vector<scalar> value(
        scalar x,
        const std::string& patchName,
        const std::vector<vector>& faceCentres,
        scalar timeValue,
        scalar timeOutputValue) const;

    const CodedPatchFunction1Spec& spec() const { return spec_; }

private:
    CodedPatchFunction1Spec spec_;
    int (*fn_)(double, const char*, int, const double*, double, double, double*) = nullptr;
};

} // namespace brae
