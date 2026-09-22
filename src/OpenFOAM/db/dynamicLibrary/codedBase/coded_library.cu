// The compile-and-load half of OpenFOAM's dynamic code -- see coded_library.cuh.
#include "coded_library.cuh"
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <unistd.h>

namespace brae {
namespace codedLibrary {

namespace {

std::string quoted(const std::string& s)
{
    std::string out = "'";
    for (const char c : s)
    {
        if (c == '\'')
        {
            out += "'\\''";
        }
        else
        {
            out += c;
        }
    }
    return out + "'";
}

// Libraries stay loaded for the life of the process, as OpenFOAM's do: a function-local static in the
// snippet must keep its value across every call.
std::map<std::string, void*>& loadedLibraries()
{
    static std::map<std::string, void*> libs;
    return libs;
}

} // namespace


std::string contentKey(const std::string& text)
{
    std::uint64_t h = 1469598103934665603ull;
    for (const unsigned char c : text)
    {
        h ^= c;
        h *= 1099511628211ull;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}


std::string identifier(const std::string& name)
{
    std::string out;
    for (const char c : name)
    {
        out += (std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
    }
    if (out.empty() || std::isdigit(static_cast<unsigned char>(out[0])))
    {
        out = "f" + out;
    }
    return out;
}


std::string trim(const std::string& s)
{
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
    {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
    {
        --e;
    }
    return s.substr(b, e - b);
}


std::string cLiteral(const std::string& s)
{
    std::string out;
    for (const char c : s)
    {
        if (c == '"' || c == '\\')
        {
            out += '\\';
        }
        out += c;
    }
    return out;
}


void refuseRoot(const std::string& what)
{
    if (::geteuid() == 0)
    {
        throw std::runtime_error(
            what + ": OpenFOAM does not execute dynamic code with administrator rights (dynamicCode.C "
            "checkSecurity), and neither does brae.");
    }
}


void* compileAndLoad(const Build& b)
{
    std::map<std::string, void*>& libs = loadedLibraries();
    const auto found = libs.find(b.key);
    if (found != libs.end())
    {
        return found->second;
    }

    const std::filesystem::path dir(b.dir);
    const std::filesystem::path lib = dir / ("lib" + b.libStem + ".so");
    if (!std::filesystem::exists(lib))
    {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec)
        {
            throw std::runtime_error(b.what + ": cannot create " + dir.string() + " (" + ec.message() +
                                     "). Set BRAE_DYNAMIC_CODE_DIR to a writable directory.");
        }
        {
            std::ofstream(dir / b.shimFile) << b.shimText;
            std::ofstream(dir / b.sourceFile) << b.sourceText;
        }
        // Built under a temporary name and renamed, so an interrupted compile never leaves a library a
        // later run would load.
        const char* envCxx = std::getenv("BRAE_CXX");
        const std::string cxx = (envCxx && *envCxx) ? envCxx : "g++";
        const std::filesystem::path tmp = dir / ("lib.tmp." + std::to_string(::getpid()) + ".so");
        const std::filesystem::path log = dir / "compile.log";
        const std::string cmd = cxx + " -std=c++17 -pthread -O3 -fPIC -shared -o " + quoted(tmp.string())
                              + " " + quoted((dir / b.sourceFile).string())
                              + " > " + quoted(log.string()) + " 2>&1";
        const int rc = std::system(cmd.c_str());
        if (rc != 0 || !std::filesystem::exists(tmp))
        {
            std::ifstream lf(log);
            std::string text((std::istreambuf_iterator<char>(lf)), std::istreambuf_iterator<char>());
            if (text.size() > 6000)
            {
                text = text.substr(0, 6000) + "\n[...]";
            }
            throw std::runtime_error(
                b.what + " did not compile (`" + cxx + "`, source in " + dir.string() + "). OpenFOAM fails "
                "the case the same way on a compile error; brae's snippet scope is " + b.shimDescription +
                " in " + b.shimFile + " there, so a name OpenFOAM's headers provide and the shim does not "
                "is refused here. Compiler output:\n" + text);
        }
        std::filesystem::rename(tmp, lib, ec);
        if (ec)
        {
            throw std::runtime_error(b.what + ": cannot install " + lib.string() + " (" + ec.message() + ").");
        }
    }
    void* handle = ::dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
    {
        throw std::runtime_error(b.what + ": cannot load " + lib.string() + " (" + ::dlerror() + ").");
    }
    libs[b.key] = handle;
    return handle;
}


void* symbol(
    void* handle,
    const std::string& name,
    const std::string& what,
    const std::string& lib)
{
    void* s = ::dlsym(handle, name.c_str());
    if (!s)
    {
        throw std::runtime_error(what + ": " + lib + " does not export " + name + "; delete the directory "
                                 "and rerun.");
    }
    return s;
}

} // namespace codedLibrary
} // namespace brae
