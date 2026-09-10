#pragma once
// OpenFOAM's `type coded;` Function1<scalar>, compiled and loaded at run time -- HOST ONLY.
//
// provenance:
//   openfoam:
//     class:    Foam::Function1Types::CodedFunction1<Type>
//     file:     src/OpenFOAM/primitives/functions/Function1/Coded/CodedFunction1.C
//               :153     redirectName_ defaults to the entry name
//               :159     the constructor compiles and loads the library (updateLibrary)
//               :100-106 prepare() refuses an empty `code`
//               :221-231 value(x) updates the library, then redirects to its value(x)
//     template: etc/codeTemplates/dynamicCode/codedFunction1Template.C -- `code` is pasted as the BODY of
//               `Foam::scalar Foam::Function1Types::<name>Function1_scalar::value(const scalar x) const`
//     compile:  wmake/rules/General/Gcc/c++ (CC = g++ -std=c++17) + linuxARM64Gcc/c++Opt (-O3), -fPIC,
//               linked with `$(CC) $(c++FLAGS) -shared` (General/Gcc/link-c++)
//   brae:
//     reference: this file; the flow-rate Function1 of flowRateInletVelocity (function1.cuh)
//     tests:     tests/test_coded_function1.cu, tests/rho_coded_function1_vs_openfoam.sh
//
// WHY THE HOST. OpenFOAM compiles host C++ and dlopens it, and both mirror arms form the inlet velocity
// on the host (the device arm reduces gSum(rho*magSf) on the GPU and divides on the host). Compiling the
// snippet with the system C++ compiler keeps what OpenFOAM's snippet can rely on -- a function-local
// `static` lives for the process, `Info` writes to stdout, pow/exp are glibc's -- where NVRTC would change
// all three and make the host reference depend on a GPU.
//
// WHAT THE SNIPPET SEES. OpenFOAM's template puts the code inside a const member function of a class in
// Foam::Function1Types, so unqualified names resolve through Foam. brae generates the same class and the
// same function signature, and replaces OpenFOAM's headers with a SHIM that carries the scalar subset
// transcribed from them (the list is in codedFunction1.cu, each entry with its OpenFOAM file). A name the
// shim does not carry -- `this->time()`, a vector, an objectRegistry lookup, Pout -- is a COMPILE ERROR,
// and brae throws with the compiler's message and the snippet's origin. Never a substitute.
//
// REFUSED BY NAME, before compiling, because brae cannot reproduce what OpenFOAM would do with them:
//   codeInclude, localCode, codeOptions, codeLibs  -- OpenFOAM compiles them against its own headers
//   a `$` anywhere in `code`                        -- OpenFOAM expands it against the dictionary first
//                                                      (dynamicCodeContext), a scope brae does not have
//   an empty `code`                                 -- OpenFOAM refuses it too (CodedFunction1.C:100-106)
//   running as root                                 -- OpenFOAM refuses dynamic code then (dynamicCode.C:71-79)
#include "cf_types.cuh"
#include <memory>
#include <string>

namespace brae {

struct CodedFunction1Spec
{
    std::string name;       // the dictionary's `name`, defaulting to the entry name (CodedFunction1.C:147)
    std::string code;       // the verbatim #{ ... #} body
    std::string origin;     // "<file>: patch 'inlet', massFlowRate" -- in every message and the #line
    std::string codeDir;    // where the generated source and library are written
    // Keys OpenFOAM would compile in and brae cannot -- non-empty means refuse, naming them.
    std::string unsupportedKeys;
};

class CodedFunction1
{
public:
    // Compiles (or reuses the library a previous run built from the same code) and loads it, as
    // OpenFOAM's constructor does -- a snippet that does not compile fails the case at construction.
    explicit CodedFunction1(CodedFunction1Spec spec);

    // The loaded value(x). x is whatever the caller passes; flowRateInletVelocity passes the time.
    scalar value(scalar x) const;

    const CodedFunction1Spec& spec() const { return spec_; }

private:
    CodedFunction1Spec spec_;
    double (*fn_)(double) = nullptr;
};

} // namespace brae
