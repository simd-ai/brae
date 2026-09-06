#!/usr/bin/env bash
# brae SAYS which smoother it is running -- and says nothing where there is nothing to say.
#
# `smoothSolver` with a `symGaussSeidel` smoother is what almost every turbulent OpenFOAM tutorial asks
# for on U, k and omega; `GaussSeidel` is what much of the rest asks for. brae used to route BOTH to a
# MULTICOLOUR sweep -- a different smoother wearing each name -- and this gate asserted the notice that
# admitted it. Neither is substituted now: device_sym_gauss_seidel.cuh runs OpenFOAM's own index-order
# sweep, level-scheduled, ascending-then-descending for `symGaussSeidel` and ascending-only for
# `GaussSeidel` (GaussSeidelSmoother.C's sweep loop has no second half). tests/gs_ladder holds both to
# OpenFOAM's own per-sweep residual, at 2.8e-12 and 5.2e-13. So both notices are GONE, and their absence
# is now the assertion.
#
# WHY THE SILENCE IS WORTH ASSERTING. On validation/T3A the colour order and the index order stop on
# opposite sides of the case's own `relTol 0.1; maxIter 10`: OpenFOAM reaches relTol 0.1 in 4-5
# index-order sweeps, the colour order needed 9-10 and so always took the cap, and brae limit-cycled at
# U 3.83e-01 where it now converges in 406 iterations at U 6.7e-05 against OpenFOAM's own answer. A
# notice that came back would mean that regressed.
#
#   POSITIVE  a case naming a smoother brae does NOT run -- `DILUGaussSeidel`, a real OpenFOAM smoother
#             for asymmetric matrices -- must still be announced. That notice comes from the shared
#             READER (linear_solver_setup.cuh), not from the envelope, so it is asserted in ARM 4 on the
#             real driver's stderr; tests/test_solver_notices.cu asserts the same thing at reader level
#             without a GPU. It is what proves the machinery still speaks, so the silences below mean
#             "nothing to say" rather than "gone quiet".
#   CONTROL A symGaussSeidel must NOT produce a smoother notice.
#   CONTROL B GaussSeidel must NOT produce one either -- it is exact now, ascending walk and all.
#   CONTROL C PBiCGStab/DILU must not produce one, or the notice is unconditional.
#   CONTROL 2 the GAMG-on-p notice must appear in ALL FOUR, so a silence is about the smoother and not
#             about the envelope having gone quiet.
#
# Fail-proof, 2026-09-04: with the notices.push_back wrapped in `if (false)`, POSITIVE fails and the
# controls still pass.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="${BUILD:-$ROOT/build}/v2_envelope"
SRC="${1:-$ROOT/validation/T3A}"

[ -x "$PROBE" ] || { echo "SKIP: $PROBE not built"; exit 77; }
[ -d "$SRC" ]   || { echo "SKIP: fixture $SRC missing"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

cp -r "$SRC" "$W/gsym"; cp -r "$SRC" "$W/gfwd"; cp -r "$SRC" "$W/other"; cp -r "$SRC" "$W/krylov"
# CONTROL B: OpenFOAM's ascending-only smoother, which brae now runs exactly.
python3 - "$W/gfwd" <<'PYX'
import re, sys
f = sys.argv[1] + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'smoother\s+symGaussSeidel;', 'smoother        GaussSeidel;', s)
open(f, 'w').write(s)
PYX
# THE POSITIVE: a real OpenFOAM smoother for asymmetric matrices that brae does not implement.
python3 - "$W/other" <<'PYX'
import re, sys
f = sys.argv[1] + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'smoother\s+symGaussSeidel;', 'smoother        DILUGaussSeidel;', s)
open(f, 'w').write(s)
PYX
grep -q "smoother        DILUGaussSeidel;" "$W/other/system/fvSolution" \
    || { echo "FAIL: the positive fixture does not name DILUGaussSeidel"; exit 1; }
# CONTROL C: a solver brae substitutes in a DIFFERENT way, so the smoother entry disappears entirely.
python3 - "$W/krylov" <<'PYX'
import re, sys
f = sys.argv[1] + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'solver\s+smoothSolver;', 'solver          PBiCGStab;', s)
s = re.sub(r'smoother\s+symGaussSeidel;', 'preconditioner  DILU;', s)
open(f, 'w').write(s)
PYX
grep -q "smoothSolver" "$W/krylov/system/fvSolution" && { echo "FAIL: the control still names smoothSolver"; exit 1; }

for c in gsym gfwd other krylov; do
    "$PROBE" "$W/$c" > "$W/$c.out" 2>&1 || { echo "FAIL: probe crashed on $c"; exit 1; }
done

grep -qE "smoother" "$W/other.out" \
    && say "CONTROL D the envelope stays out of it -- the READER announces an unrun smoother" FAIL \
    || say "CONTROL D the envelope stays out of it -- the READER announces an unrun smoother" ok
grep -qE "smoother" "$W/gsym.out" \
    && say "CONTROL A symGaussSeidel announces NO smoother substitution" FAIL \
    || say "CONTROL A symGaussSeidel announces NO smoother substitution" ok
grep -qE "smoother" "$W/gfwd.out" \
    && say "CONTROL B GaussSeidel announces none either -- it is exact now" FAIL \
    || say "CONTROL B GaussSeidel announces none either -- it is exact now" ok
grep -qE "ASCENDING ONLY|MULTICOLOUR" "$W/krylov.out" \
    && say "CONTROL C PBiCGStab/DILU announces no sweep substitution" FAIL \
    || say "CONTROL C PBiCGStab/DILU announces no sweep substitution" ok
if grep -q "AMG-preconditioned PCG" "$W/gsym.out" && grep -q "AMG-preconditioned PCG" "$W/gfwd.out" \
   && grep -q "AMG-preconditioned PCG" "$W/other.out" && grep -q "AMG-preconditioned PCG" "$W/krylov.out"; then
    say "CONTROL 2 the GAMG notice fires in all four, so the envelope is reporting" ok
else
    say "CONTROL 2 the GAMG notice fires in all four, so the envelope is reporting" FAIL
fi

# ARM 4 -- THE RUN LOG ITSELF. The legacy driver prints an OpenFOAM-FORMAT per-iteration report so a log
# can be diffed against one, and its solver prefixes used to read `smoothSolver:` and `GAMG:` whatever
# brae actually ran: a log asserting a capability the code does not have. Every gate that parses this log
# matches on `Solving for <field>, Initial residual = ...` and none anchors on the prefix, so the prefix
# now names what brae runs. Needs a GPU because it is a real solve; the three arms above do not.
if command -v nvidia-smi > /dev/null 2>&1 && [ -d "$SRC/0.orig" ]; then
    cp -r "$SRC" "$W/legacy"; rm -rf "$W"/legacy/[1-9]* "$W"/legacy/0 "$W"/legacy/log.*
    cp -r "$W/legacy/0.orig" "$W/legacy/0"
    python3 - "$W/legacy" <<'PYEOF'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;', 'endTime         2;', s)
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
PYEOF
    ( cd "$W/legacy" && "${BRAE_BIN:-$ROOT/build/brae}" "$W/legacy" > log.out 2> log.err ) \
        || { echo "FAIL: the legacy driver crashed"; tail -8 "$W/legacy/log.err"; exit 1; }
    if grep -q "Solving for Ux, Initial residual = " "$W/legacy/log.out"; then
        say "ARM 4    the OpenFOAM-format report still parses (Solving for Ux, ...)" ok
    else
        say "ARM 4    the OpenFOAM-format report still parses (Solving for Ux, ...)" FAIL
    fi
    if grep -qE '^GAMG: *Solving for p' "$W/legacy/log.out"; then
        say "ARM 4    the p line does NOT claim GAMG (brae runs AMG-PCG)" FAIL
    else
        say "ARM 4    the p line does NOT claim GAMG (brae runs AMG-PCG)" ok
    fi
    if grep -qE '^smoothSolver: *Solving' "$W/legacy/log.out"; then
        say "ARM 4    no line claims OpenFOAM's bare smoothSolver" FAIL
    else
        say "ARM 4    no line claims OpenFOAM's bare smoothSolver" ok
    fi
    # The fixture asks for symGaussSeidel, which brae now runs exactly, so the driver must NOT claim any
    # smoother substitution.
    if grep -q "solvers/U smoother" "$W/legacy/log.err"; then
        say "ARM 4    no smoother notice on symGaussSeidel, because there is nothing to say" FAIL
    else
        say "ARM 4    no smoother notice on symGaussSeidel, because there is nothing to say" ok
    fi
    # THE POSITIVE, at driver level: a smoother brae does NOT run must still be announced, or the
    # silence above is the reader having gone quiet rather than having nothing to report.
    cp -r "$W/legacy" "$W/legacy_dilugs"; rm -rf "$W"/legacy_dilugs/[1-9]* "$W/legacy_dilugs/log.out" "$W/legacy_dilugs/log.err"
    python3 - "$W/legacy_dilugs" <<'PYEOF'
import re, sys
f = sys.argv[1] + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'smoother\s+symGaussSeidel;', 'smoother        DILUGaussSeidel;', s)
open(f, 'w').write(s)
PYEOF
    ( cd "$W/legacy_dilugs" && "${BRAE_BIN:-$ROOT/build/brae}" "$W/legacy_dilugs" > log.out 2> log.err ) || true
    if grep -q "solvers/U smoother" "$W/legacy_dilugs/log.err"; then
        say "ARM 4    POSITIVE: a smoother brae does not run IS announced (DILUGaussSeidel)" ok
    else
        say "ARM 4    POSITIVE: a smoother brae does not run IS announced (DILUGaussSeidel)" FAIL
    fi
    # ARM 5 -- the legacy driver names WHICH GaussSeidel it ran. There is no notice to assert any more
    # (neither name is substituted), so the run log is the only observable that says the variant reached
    # the solver at all. It is also the only check that the legacy arm's turbulence calls forward it: they
    # take the flag from the same ctl_ this line prints.
    if grep -q "smoothSolver\[symGaussSeidel\]" "$W/legacy/log.out"; then
        say "ARM 5    the log names symGaussSeidel when the case asked for it" ok
    else
        say "ARM 5    the log names symGaussSeidel when the case asked for it" FAIL
    fi
    cp -r "$W/legacy" "$W/legacy_gs"; rm -rf "$W"/legacy_gs/[1-9]* "$W/legacy_gs/log.out" "$W/legacy_gs/log.err"
    python3 - "$W/legacy_gs" <<'PYEOF'
import re, sys
f = sys.argv[1] + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'smoother\s+symGaussSeidel;', 'smoother        GaussSeidel;', s)
open(f, 'w').write(s)
PYEOF
    ( cd "$W/legacy_gs" && "${BRAE_BIN:-$ROOT/build/brae}" "$W/legacy_gs" > log.out 2> log.err ) || true
    if grep -q "smoothSolver\[GaussSeidel\]" "$W/legacy_gs/log.out"; then
        say "ARM 5    ...and names GaussSeidel when the case asked for THAT" ok
    else
        say "ARM 5    ...and names GaussSeidel when the case asked for THAT" FAIL
    fi
else
    echo "  ARM 4    skipped: no GPU or no 0.orig in the fixture"
fi

[ $fail -eq 0 ] || { for c in gsym gfwd other krylov; do echo "--- $c ---"; cat "$W/$c.out"; done; }
[ $fail -eq 0 ] && echo "PASS: brae announces the smoother it cannot run, and nothing where it substitutes nothing"
exit $fail
