// `$name` resolves in the dictionary it is WRITTEN in, then in that dictionary's ancestors.
//
// OpenFOAM's expandVariable (dictionary::lookupScopedEntryPtr) searches the current dictionary and then
// walks up its parents. brae's expander built ONE FLAT MAP -- every entry under its bare name, last
// definition in the file winning -- so a name defined twice at different depths resolved by POSITION.
// OpenFOAM's own angledDuctExplicitFixedCoeff is the case that breaks on it:
//
//     solvers { U { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-06; relTol 0.1; }
//               "(k|epsilon)" { $U; tolerance 1e-07; relTol 0.1; } }
//     relaxationFactors { equations { U 0.7; ... } }
//
// The relaxation leaf comes later, so `$U` expanded to `0.7`: k and epsilon named NO solver, fell back
// to brae's default BiCGStab, and no notice fired -- the substitution notice keys on `solver` being
// present and different from what brae runs. Silent, and worth 2.56x on k's residual against OpenFOAM
// (queue item 73, measured in tests/rho_smoothsolver_vs_openfoam).
//
//   LEG 1  the sibling wins over a same-named leaf in another block, whichever comes later in the file
//   LEG 2  ...and the merged entries are the referenced block's, with the referring block's OWN entries
//          still overriding them (OpenFOAM's merge: `$U;` first, then `tolerance 1e-07;`)
//   LEG 3  an ANCESTOR is still found when there is no sibling -- fvSchemes' `div(phi,k) $turbulence;`
//   LEG 4  a name with '-' and '.', which OF's word::valid allows and cases use
//   LEG 5  `z0 $z0;` -- an inner entry pulling from an outer of the same name keeps the outer value
//   LEG 6  ${/name} -- the root-scoped form a functionObject uses to reach controlDict's endTime
#include "foam_dict.cuh"
#include <cstdio>
#include <fstream>
#include <string>

using namespace brae;

namespace {
int failures = 0;
void say(const char* what, bool ok)
{
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}
std::string write(const std::string& dir, const char* name, const char* body)
{
    const std::string p = dir + "/" + name;
    std::ofstream f(p);
    f << "FoamFile { version 2.0; format ascii; class dictionary; object " << name << "; }\n" << body;
    return p;
}
}   // namespace

int main(int argc, char** argv)
{
    const std::string dir = (argc > 1) ? argv[1] : ".";

    // ---- LEG 1 and 2: the shape angledDuct ships ---------------------------------------------------
    {
        const std::string p = write(dir, "fvSolution",
            "solvers\n{\n"
            "    U { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-06; relTol 0.1; }\n"
            "    \"(k|epsilon)\" { $U; tolerance 1e-07; relTol 0.1; }\n"
            "}\n"
            "relaxationFactors\n{\n    equations { U 0.7; e 0.5; }\n}\n");
        const FoamDict d = readDict(p);
        const FoamDict* sol = d.subDict("solvers");
        const FoamDict* k = sol ? sol->subDict("k") : nullptr;
        say("LEG 1  the pair's block resolves $U to the SIBLING solvers/U, not to relaxationFactors/U",
            k && k->wordOr("solver", "") == "smoothSolver" && k->wordOr("smoother", "") == "GaussSeidel");
        say("LEG 1  ...including the entries only the referenced block carries (nSweeps 2)",
            k && (int)k->scalarOr("nSweeps", -1) == 2);
        say("LEG 2  ...and the block's OWN tolerance overrides the merged one (1e-07, not 1e-06)",
            k && std::abs((double)k->scalarOr("tolerance", -1) - 1e-07) < 1e-20);
        say("LEG 2  ...while the relaxation factor is untouched by the merge",
            d.subDict("relaxationFactors") && d.subDict("relaxationFactors")->subDict("equations")
            && std::abs((double)d.subDict("relaxationFactors")->subDict("equations")->scalarOr("U", -1) - 0.7) < 1e-12);
    }
    // ---- LEG 3: the ancestor, which is what fvSchemes uses ------------------------------------------
    {
        const std::string p = write(dir, "fvSchemes",
            "divSchemes\n{\n"
            "    turbulence      bounded Gauss limitedLinear 1;\n"
            "    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
            "    div(phi,k)      $turbulence;\n"
            "}\n");
        const FoamDict d = readDict(p);
        const FoamDict* ds = d.subDict("divSchemes");
        // The parser splits a parenthesised key: `div(phi,k) bounded ...` is stored as the leaf `div`
        // with values `( phi,k ) bounded ...`, so this reads the LAST such leaf rather than looking up
        // a name the dictionary never holds.
        bool ok = false;
        if (ds)
            for (const auto& l : ds->leaves)
                if (l.first == "div" && l.second.size() >= 7 && l.second[1] == "phi,k")
                    ok = (l.second[3] == "bounded" && l.second[5] == "limitedLinear");
        say("LEG 3  an ancestor/sibling entry is still found when no nearer one exists", ok);
    }
    // ---- LEG 4: '-' and '.' in a name --------------------------------------------------------------
    {
        const std::string p = write(dir, "fvSolution2",
            "relaxationFactors-SIMPLE { fields { p 0.3; } equations { U 0.7; } }\n"
            "relaxationFactors { $relaxationFactors-SIMPLE }\n");
        const FoamDict d = readDict(p);
        const FoamDict* rf = d.subDict("relaxationFactors");
        say("LEG 4  a macro name may carry '-' (relaxationFactors-SIMPLE)",
            rf && rf->subDict("fields") && std::abs((double)rf->subDict("fields")->scalarOr("p", -1) - 0.3) < 1e-12);
    }
    // ---- LEG 5: the self-reference ------------------------------------------------------------------
    {
        const std::string p = write(dir, "ablConditions",
            "z0              uniform 0.1;\n"
            "inlet\n{\n    z0              $z0;\n    other           1;\n}\n");
        const FoamDict d = readDict(p);
        const FoamDict* in = d.subDict("inlet");
        const std::vector<std::string>* v = in ? in->find("z0") : nullptr;
        say("LEG 5  `z0 $z0;` keeps the outer definition rather than resolving to itself",
            v && v->size() >= 2 && (*v)[0] == "uniform");
    }
    // ---- LEG 6: the root-scoped form ---------------------------------------------------------------
    {
        const std::string p = write(dir, "controlDictLike",
            "endTime         0.5;\n"
            "functions\n{\n    probe { timeStart ${/endTime}; }\n}\n");
        const FoamDict d = readDict(p);
        const FoamDict* fn = d.subDict("functions");
        const FoamDict* pr = fn ? fn->subDict("probe") : nullptr;
        say("LEG 6  ${/name} reaches the file's root scope",
            pr && std::abs((double)pr->scalarOr("timeStart", -1) - 0.5) < 1e-12);
    }
    std::printf(failures ? "FAIL: %d check(s)\n" : "PASS: $name resolves by scope, as OpenFOAM resolves it\n", failures);
    return failures ? 1 : 0;
}
