// CodedFunction1's mechanism, on the host, without OpenFOAM: the snippet compiles, x reaches it, a
// function-local static lives for the process, and every key brae cannot reproduce is refused by name.
//
// WHAT THIS GATE DOES NOT CLAIM. It carries no OpenFOAM oracle -- no number here came from OpenFOAM.
// That comparison is tests/rho_coded_function1_vs_openfoam.sh, which runs the squareBendLiq tutorial's
// own coded massFlowRate on both mirror arms against real OpenFOAM compiling the same body, iteration by
// iteration, and asserts the printed-once property against OpenFOAM's own library. This gate covers the
// mechanism underneath it, and runs where that one exits 77 for want of an OpenFOAM installation.
//
// THE CONTROL for every refusal below is arm 1: a snippet that compiles, loads and returns the right
// number. Without it a CodedFunction1 that threw on everything would pass all five refusals.
#include "codedFunction1.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>

using namespace brae;

int main()
{
    int fails = 0;

    auto chk = [&](const char* nm, double got, double exp, double tol)
    {
        const bool ok = std::fabs(got - exp) <= tol*std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-52s got %.12g  exp %.12g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };
    auto flag = [&](const char* nm, bool ok)
    {
        if (!ok) ++fails;
        std::printf("  %-52s %s\n", nm, ok ? "OK" : "FAIL");
    };

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("brae_coded_f1_" + std::to_string(::getpid()));

    auto specFor = [&](const std::string& name, const std::string& code)
    {
        CodedFunction1Spec s;
        s.name = name;
        s.code = code;
        s.origin = "tests/test_coded_function1.cu: " + name;
        s.codeDir = dir.string();
        return s;
    };

    // Every refusal reports what it threw, so a message that stops naming the thing turns this red rather
    // than passing on the throw alone.
    auto refuses = [&](const char* nm, CodedFunction1Spec spec, const char* mustSay)
    {
        std::string msg;
        try
        {
            CodedFunction1 fn(std::move(spec));
            (void)fn.value(0);
        }
        catch (const std::exception& e)
        {
            msg = e.what();
        }
        if (msg.empty())
        {
            ++fails;
            std::printf("  %-52s RAN                                      FAIL\n", nm);
            return;
        }
        const bool named = msg.find(mustSay) != std::string::npos;
        if (!named) ++fails;
        std::printf("  %-52s %s\n", nm, named ? "refused, by name                         OK"
                                              : "refused, but not by name                 FAIL");
        if (!named) std::printf("      threw: %.200s\n", msg.c_str());
    };

    // ---- arm 1: the snippet compiles, and x reaches it -------------------------------------------
    // The body varies with x on purpose. A CodedFunction1 that compiled the code and then evaluated it
    // once, or fed it a frozen time, returns 5 for every call and passes a body that returns a constant;
    // it cannot pass this one. Same discriminator the .sh gate's second snippet carries.
    std::printf("a coded body that varies with x:\n");
    try
    {
        CodedFunction1 fn(specFor("rate", "    return 5*(1 + 0.05*x);"));
        chk("value(0)   = 5", fn.value(0), 5.00, 1e-15);
        chk("value(1)   = 5.25", fn.value(1), 5.25, 1e-15);
        chk("value(20)  = 10", fn.value(20), 10.0, 1e-15);
        flag("value(1) is not value(0) -- x reaches the snippet", fn.value(1) != fn.value(0));
    }
    catch (const std::exception& e)
    {
        ++fails;
        std::printf("  arm 1 threw: %s\n", e.what());
    }

    // ---- arm 2: the Foam scope the shim carries --------------------------------------------------
    // Unqualified, as OpenFOAM's template leaves them: the snippet is the body of a member function of a
    // class in Foam::Function1Types.
    std::printf("the shim's Foam scope, unqualified:\n");
    try
    {
        CodedFunction1 fn(specFor("scope", "    return sqr(x) + mag(-2.0) + pow(2.0, 3)"
                                           " + constant::mathematical::pi;"));
        chk("sqr(x) + mag + pow + pi at x = 1", fn.value(1), 1.0 + 2.0 + 8.0 + M_PI, 1e-15);
    }
    catch (const std::exception& e)
    {
        ++fails;
        std::printf("  arm 2 threw: %s\n", e.what());
    }

    // ---- arm 3: a function-local static lives for the process ------------------------------------
    // The tutorial's own snippet relies on this ("static bool reported"), and the .sh gate measures the
    // consequence -- the line printed exactly once by OpenFOAM and once by brae. Here: the static counts
    // across calls, and across a SECOND object built from the same code, because one library backs both.
    std::printf("a function-local static in the loaded library:\n");
    const std::string counter = "    static int calls = 0;\n    ++calls;\n    return calls;";
    try
    {
        CodedFunction1 a(specFor("counter", counter));
        chk("first  call", a.value(0), 1, 1e-15);
        chk("second call", a.value(0), 2, 1e-15);
        CodedFunction1 b(specFor("counter", counter));
        chk("a second object on the same code continues the count", b.value(0), 3, 1e-15);
        // The control for that sharing: a DIFFERENT body is a different library and a fresh static, so
        // the count above is keyed on the code and not merely a process-wide counter.
        CodedFunction1 c(specFor("counter2", "    static int calls = 0;\n    ++calls;\n"
                                             "    return 100 + calls;"));
        chk("a different body gets its own static", c.value(0), 101, 1e-15);
    }
    catch (const std::exception& e)
    {
        ++fails;
        std::printf("  arm 3 threw: %s\n", e.what());
    }

    // ---- arm 4: the refusals, each by name -------------------------------------------------------
    std::printf("refusals:\n");
    refuses("empty code, as OpenFOAM refuses it", specFor("empty", "   \n  "), "has no `code`");
    refuses("$ expansion inside code", specFor("dollar", "    return $rhoInlet;"), "uses `$` in its code");
    {
        CodedFunction1Spec s = specFor("incl", "    return 5;");
        s.unsupportedKeys = "`codeInclude`";
        refuses("codeInclude is not compiled in", std::move(s), "carries `codeInclude`");
    }
    // A name OpenFOAM's headers provide and the shim does not is a compile error, never a substitute.
    refuses("this->time(): not in brae's scope",
            specFor("thisTime", "    return this->time().value();"), "did not compile");
    refuses("an objectRegistry lookup: not in brae's scope",
            specFor("lookup", "    return db().lookupObject<volScalarField>(\"p\")[0];"), "did not compile");

    // The compile-error message carries the compiler's own output, so a snippet that fails for a reason
    // brae did not anticipate still tells the user what the compiler said.
    {
        std::string msg;
        try
        {
            CodedFunction1 fn(specFor("thisTime", "    return this->time().value();"));
        }
        catch (const std::exception& e)
        {
            msg = e.what();
        }
        flag("the compile refusal quotes the compiler", msg.find("error") != std::string::npos);
        flag("the compile refusal names the snippet's origin",
             msg.find("test_coded_function1.cu") != std::string::npos);
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    std::printf("%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
