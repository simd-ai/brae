#pragma once
// The compile-and-load half of OpenFOAM's dynamic code, shared by every coded object brae runs on the
// host: the coded Function1 (codedFunction1.cu) and the coded PatchFunction1 (codedPatchFunction1.cu).
//
// provenance:
//   openfoam: src/OpenFOAM/db/dynamicLibrary/codedBase/codedBase.C (updateLibrary: a library keyed on a
//             SHA1 of the code, built once and reused, loaded for the life of the process),
//             src/OpenFOAM/db/dynamicLibrary/dynamicCode/dynamicCode.C:71-79 (checkSecurity: no dynamic
//             code as root)
//   brae:     codedFunction1.cu (the first user; its generated file names and keys are unchanged)
//
// WHY brae DOES NOT CALL wmake. The snippet is compiled against a SHIM of OpenFOAM's scope, never
// against OpenFOAM's headers, so the result does not depend on an OpenFOAM installation. What the shim
// does not carry is a compile error the caller reports with the compiler's output.
#include <string>

namespace brae {
namespace codedLibrary {

// FNV-1a, 64 bit, as 16 hex digits -- a content key stable across processes, so a second run reuses
// the library the first one built (OpenFOAM keys its dynamicCode on a SHA1 of the same inputs).
std::string contentKey(const std::string& text);

// OpenFOAM's `name` is a word and becomes part of a class name; the identifier characters are kept.
std::string identifier(const std::string& name);

// the text between the ends, whitespace trimmed
std::string trim(const std::string& s);

// `s` escaped for a C string literal (the #line directive's file name)
std::string cLiteral(const std::string& s);

// dynamicCode.C checkSecurity: throws, naming `what`, when the process runs as root
void refuseRoot(const std::string& what);

struct Build
{
    // who is compiling, for every message: "brae: coded Function1 'x' (<origin>)"
    std::string what;
    // the generated names: <dir>/<shimFile> and <dir>/<sourceFile>, library lib<libStem>.so
    std::string dir;
    std::string shimFile;
    std::string shimText;
    std::string sourceFile;
    std::string sourceText;
    std::string libStem;
    // the key the library is cached under in this process
    std::string key;
    // for the compile-error message: where the shim is, and what it carries
    std::string shimDescription;
};

// Compiles <dir>/<sourceFile> (unless the library for this key exists already) with $BRAE_CXX or g++
// -std=c++17 -O3 -fPIC -shared, loads it once per process and returns the dlopen handle. A compile
// error throws with the compiler's output.
void* compileAndLoad(const Build& b);

// dlsym on a handle compileAndLoad returned; throws, naming `what`, when the symbol is missing
void* symbol(
    void* handle,
    const std::string& name,
    const std::string& what,
    const std::string& lib);

} // namespace codedLibrary
} // namespace brae
