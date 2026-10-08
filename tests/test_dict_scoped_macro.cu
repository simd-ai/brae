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
//   LEG 7  a `$name;` ENTRY is found through a PATTERN key: dictionary::substituteScopedKeyword searches
//          REGEX_RECURSIVE (dictionary.C:415-443), so `p_rgh { $pcorr; }` takes the parent's
//          `"(pcorr|pcorrFinal)"` block -- the two multi-paddle waveMakers write exactly that, and brae
//          read no solver for p_rgh there
//   LEG 8  ...but a `$name` VALUE is not: primitiveEntry::expandVariable searches LITERAL_RECURSIVE
//          (primitiveEntry.C:127-139), so a `".*"` block cannot capture `value $internalField;`
//   LEG 9  ...and the search is per level, literal then patterns, before the parent (dictionarySearch.C,
//          csearch): an inner pattern wins over an outer literal of the same name
//   LEG 10 ...and it sees only what has ALREADY been read: OpenFOAM substitutes while it reads
//          (entryIO.C:274) and a block is added at its closing brace, so a trailing `".*"` cannot
//          capture a reference above it -- blockMesh/pipe's three edge patches became the surface one.
//          A reference BELOW it is captured, and gets the pattern block's EXPANDED content
//   LEG 11 `${name}` is one token, a reference. Read as a block called `$` it DEFINED `name` as
//          nothing, and every later `$name` expanded empty
//   LEG 12 a braced form this does not resolve -- `${a/b}` -- stays whole; resolving `a` alone left
//          `/b}` behind (sphereDrop's centreOfMass)
//   LEG 13 an UNDEFINED `$name;` entry is dropped, as dictionary::substituteScopedKeyword drops it
//          (dictionary.C:415-443, its return ignored at entryIO.C:274); an undefined VALUE, and an
//          undefined name that opens a block, are fatal in OpenFOAM and are NOT dropped
//   LEG 14 a directive inside a comment or a string is not a directive: `// #include "x"` includes
//          nothing, and `note "see the #calc entry";` is not refused as a #calc
//
// Measured over OpenFOAM's 9,500 tutorial dictionaries against `foamDictionary -expand`, at every entry
// the change moved: 2,958 now agree with OpenFOAM and 17 agreed only before. Those 17 are forms brae
// resolves in neither version (`${_${FOAM_EXECUTABLE}}`, `${mergeType:-default}`, `#remove "__.*"`).
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
    // LEG 7: a keyword reference found through a pattern key
    {
        const std::string p = write(dir, "fvSolutionPatterns",
            "solvers\n{\n"
            "    \"(pcorr|pcorrFinal)\"\n    {\n        solver GAMG; tolerance 1e-5; relTol 0;\n"
            "        smoother DICGaussSeidel; nCellsInCoarsestLevel 10;\n    }\n"
            "    p_rgh\n    {\n        $pcorr;\n        tolerance 1e-07;\n        relTol 0.05;\n    }\n"
            "    p_rghFinal\n    {\n        $p_rgh;\n        tolerance 1e-07;\n        relTol 0;\n    }\n"
            "}\n");
        const FoamDict d = readDict(p);
        const FoamDict* sol = d.subDict("solvers");
        const FoamDict* pr = sol ? sol->subDict("p_rgh") : nullptr;
        const FoamDict* pf = sol ? sol->subDict("p_rghFinal") : nullptr;
        say("LEG 7  `p_rgh { $pcorr; }` takes the parent's \"(pcorr|pcorrFinal)\" block (solver GAMG)",
            pr && pr->wordOr("solver", "") == "GAMG" && pr->wordOr("smoother", "") == "DICGaussSeidel");
        say("LEG 7  ...with p_rgh's own tolerance and relTol overriding the merged ones",
            pr && std::abs((double)pr->scalarOr("tolerance", -1) - 1e-07) < 1e-20
               && std::abs((double)pr->scalarOr("relTol", -1) - 0.05) < 1e-15);
        say("LEG 7  ...and `p_rghFinal { $p_rgh; }` carries it on, relTol 0",
            pf && pf->wordOr("solver", "") == "GAMG" && std::abs((double)pf->scalarOr("relTol", -1)) < 1e-20);
    }
    // LEG 8: a value reference is literal only
    {
        const std::string p = write(dir, "fieldLike",
            "internalField   uniform 3;\n"
            "boundaryField\n{\n"
            "    \".*\" { type zeroGradient; }\n"
            "    movingWall { type fixedValue; value $internalField; }\n"
            "}\n");
        const FoamDict d = readDict(p);
        const FoamDict* bf = d.subDict("boundaryField");
        const FoamDict* mw = bf ? bf->subDict("movingWall") : nullptr;
        const std::vector<std::string>* v = mw ? mw->find("value") : nullptr;
        say("LEG 8  `value $internalField;` is the root's uniform 3, not the \".*\" block",
            v && v->size() == 2 && (*v)[0] == "uniform" && (*v)[1] == "3");
    }
    // LEG 9: per level, patterns before the parent
    {
        const std::string p = write(dir, "levels",
            "bb { v 2; }\n"
            "a\n{\n    \"b.*\" { v 1; }\n    inner { $bb; }\n}\n");
        const FoamDict d = readDict(p);
        const FoamDict* a = d.subDict("a");
        const FoamDict* in = a ? a->subDict("inner") : nullptr;
        say("LEG 9  an inner level's pattern (\"b.*\", v 1) wins over the root's literal bb (v 2)",
            in && std::abs((double)in->scalarOr("v", -1) - 1) < 1e-15);
    }
    // LEG 10: a pattern is visible only once it has been read
    {
        const std::string p = write(dir, "pointDisplacementLike",
            "__edge { type edgeSlip; }\n"
            "__surf { type surfaceSlip; }\n"
            "boundaryField\n{\n"
            "    e1 { ${__edge}; }\n"
            "    \".*\" { ${__surf}; }\n"
            "    late { ${__edge}; }\n"
            "}\n");
        const FoamDict d = readDict(p);
        const FoamDict* bf = d.subDict("boundaryField");
        const FoamDict* e1 = bf ? bf->subDict("e1") : nullptr;
        const FoamDict* late = bf ? bf->subDict("late") : nullptr;
        say("LEG 10 a reference ABOVE the trailing \".*\" is the root's __edge (edgeSlip)",
            e1 && e1->wordOr("type", "") == "edgeSlip");
        say("LEG 10 ...and one BELOW it is captured by it (surfaceSlip), as OpenFOAM captures it",
            late && late->wordOr("type", "") == "surfaceSlip");
    }
    // LEG 11: a braced reference does not define its own name as nothing
    {
        const std::string p = write(dir, "bracedRef",
            "Uin uniform (1 0 0);\n"
            "__edge { type edgeSlip; }\n"
            "boundaryField\n{\n"
            "    a { ${__edge}; }\n"
            "    inlet { type fixedValue; value ${Uin}; }\n"
            "    outlet { type inletOutlet; inletValue $Uin; }\n"
            "}\n");
        const FoamDict d = readDict(p);
        const FoamDict* bf = d.subDict("boundaryField");
        const FoamDict* a = bf ? bf->subDict("a") : nullptr;
        const FoamDict* in = bf ? bf->subDict("inlet") : nullptr;
        const FoamDict* ou = bf ? bf->subDict("outlet") : nullptr;
        const std::vector<std::string>* vi = in ? in->find("value") : nullptr;
        const std::vector<std::string>* vo = ou ? ou->find("inletValue") : nullptr;
        say("LEG 11 `a { ${__edge}; }` merges the block",
            a && a->wordOr("type", "") == "edgeSlip");
        say("LEG 11 `value ${Uin};` is the root's `uniform (1 0 0)`",
            vi && !vi->empty() && (*vi)[0] == "uniform" && vi->size() > 1);
        say("LEG 11 ...and the `$Uin` after it is too: `${Uin}` did not define Uin as nothing",
            vo && !vo->empty() && (*vo)[0] == "uniform" && vo->size() > 1);
    }
    // LEG 12: an unresolved braced path is left whole
    {
        const std::string out = expandDictVariables(
            "blk { h 0.147; }\n"
            "centreOfMass (0 ${/blk/h} 0);\n");
        say("LEG 12 `${/blk/h}` is left as written, not half-resolved to `blk`'s body plus `/h}`",
            out.find("(0 ${/blk/h} 0)") != std::string::npos && out.find("0.147; /h}") == std::string::npos);
    }
    // LEG 13: an undefined keyword reference is dropped; nothing else is
    {
        const std::string p = write(dir, "fragmentLike",
            "fo\n{\n    ${__nowhere}\n    field U;\n}\n"
            "fo2\n{\n    $__nowhere;\n    field p;\n}\n");
        const FoamDict d = readDict(p);
        const FoamDict* fo = d.subDict("fo");
        const FoamDict* fo2 = d.subDict("fo2");
        say("LEG 13 `fo { ${__nowhere} field U; }` holds `field U` and nothing else",
            fo && fo->subs.empty() && fo->leaves.size() == 1 && fo->wordOr("field", "") == "U");
        say("LEG 13 ...and so does the unbraced `$__nowhere;`",
            fo2 && fo2->subs.empty() && fo2->leaves.size() == 1 && fo2->wordOr("field", "") == "p");
        const std::string out = expandDictVariables(
            "kappa $__nowhere;\n"
            "$__nowhere\n{\n    type x;\n}\n");
        say("LEG 13 an undefined VALUE reference is not dropped (OpenFOAM: fatal, primitiveEntry.C:156)",
            out.find("kappa $__nowhere;") != std::string::npos);
        say("LEG 13 ...nor an undefined name that opens a block (OpenFOAM: fatal, entryIO.C:256)",
            out.find("$__nowhere\n{") != std::string::npos);
    }
    // LEG 14: a commented-out directive
    {
        write(dir, "fragX", "x 1;\n");
        const std::string off = write(dir, "includeOff",
            "// #include \"fragX\"\n"
            "/* #include \"fragMissing\" */\n"
            "note \"see the #calc entry\";\n"
            "y 2;\n");
        const std::string on = write(dir, "includeOn",
            "#include \"fragX\"\n"
            "y 2;\n");
        const FoamDict dOff = readDict(off);
        const FoamDict dOn = readDict(on);
        say("LEG 14 `// #include \"fragX\"` includes nothing",
            dOff.find("x") == nullptr && std::abs((double)dOff.scalarOr("y", -1) - 2) < 1e-15);
        say("LEG 14 ...nor is one inside a string: `note \"see the #calc entry\";` is read, not refused",
            dOff.find("note") != nullptr);
        say("LEG 14 ...and the live `#include \"fragX\"` still does (x 1)",
            std::abs((double)dOn.scalarOr("x", -1) - 1) < 1e-15);
    }
    std::printf(failures ? "FAIL: %d check(s)\n" : "PASS: $name resolves by scope, as OpenFOAM resolves it\n", failures);
    return failures ? 1 : 0;
}
