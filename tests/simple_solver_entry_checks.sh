#!/usr/bin/env bash
# Three small truths of the legacy simpleFoam driver's solver reading, found on the boxy5 UAV case.
#
#   1. AN ENTRY THAT NAMES NO SOLVER IS REFUSED, as OpenFOAM refuses it (lduMatrix::solver::New reads `solver`
#      with no default, lduMatrixSolver.C:148). An edit had left `"(U|nuTilda)" { }` and brae ran nuTilda on a
#      default of its own, silently. Arms: an explicit `nuTilda { tolerance 1e-8; relTol 0.1; }` and the
#      emptied regex block; CONTROL: the shipped entry runs.
#   2. residualControl's turbulence targets are READ when the run starts. The convergence test read nuTilda's
#      only after p and U had passed, so a run that never got there ended with
#      `SIMPLE/residualControl/nuTilda` in the unread notice although the test does gate on it. Arms: no
#      such unread line on a run that does not converge; nuTilda 1e-12 with p and U at 2 must run to endTime
#      (the target gates), nuTilda 2 must stop at iteration 1 -- the first iteration from a uniform start reads exactly 1, which a target of 1 does not pass (CONTROL: the test can pass at all).
#   3. The `Solving for U?` label names what RUNS: `DILUPBiCGStab` for an honoured `preconditioner DILU`
#      (it read `Jacobi-BiCGStab`), `smoothSolver[colourGaussSeidel]` under the colour rule.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
AF="$ROOT/validation/airFoil2D"
PD="$ROOT/validation/pitzDailyTurb"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$AF" ] && [ -d "$PD" ] || { echo "SKIP: fixtures missing"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-72s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1
    return 0
}

# stage <src> <dir> <endTime> [python edit of fvSolution text s]
stage()
{
    mkdir -p "$2"
    cp -r "$1/constant" "$1/system" "$1/0" "$2/"
    sed -i -e "s/^endTime .*/endTime         $3;/" -e "s/^writeInterval .*/writeInterval   $3;/" "$2/system/controlDict"
    if [ -n "${4:-}" ]
    then
        python3 - "$2/system/fvSolution" "$4" <<'PY'
import re
import sys
p, code = sys.argv[1], sys.argv[2]
s = open(p).read()
exec(code)
open(p, 'w').write(s)
PY
    fi
}
run()   # run <dir> [VAR=value ...]
{
    local d=$1
    shift
    ( cd "$d" && env "$@" "$BRAE" -case . > log 2> err )
    echo $? > "$d/rc"
}

# 1. the refusal
NUT='{\n        tolerance       1e-08;\n        relTol          0.1;\n    }'
stage "$AF" "$W/e1" 2 "s = re.sub(r'(\n    nuTilda\n    )\{[^}]*\}', r'\g<1>' + '$NUT', s)"
grep -A3 '^    nuTilda' "$W/e1/system/fvSolution" | grep -q solver && say "PREMISE e1 staged without a solver keyword" FAIL
run "$W/e1"
[ "$(cat "$W/e1/rc")" != 0 ] && grep -q "names no \`solver\`" "$W/e1/err" \
    && say "an explicit nuTilda entry without \`solver\` is refused, by name" ok \
    || { say "an explicit nuTilda entry without \`solver\` is refused, by name" FAIL; tail -3 "$W/e1/err"; }
stage "$AF" "$W/e2" 2 "s = re.sub(r'\n    U\n    \{[^}]*\}', '', s); s = re.sub(r'\n    nuTilda\n    \{[^}]*\}', '\n    \"(U|nuTilda)\"\n    {\n    }', s)"
grep -q '"(U|nuTilda)"' "$W/e2/system/fvSolution" || say "PREMISE e2 staged with the emptied regex block" FAIL
run "$W/e2"
[ "$(cat "$W/e2/rc")" != 0 ] && grep -q "names no \`solver\`" "$W/e2/err" \
    && say "the emptied \"(U|nuTilda)\" block is refused, by name" ok \
    || { say "the emptied \"(U|nuTilda)\" block is refused, by name" FAIL; tail -3 "$W/e2/err"; }
stage "$AF" "$W/e0" 2
run "$W/e0" BRAE_SIMPLE_CASE_SOLVER=1
[ "$(cat "$W/e0/rc")" = 0 ] && ! grep -q "names no \`solver\`" "$W/e0/err" \
    && say "CONTROL the shipped entries run" ok || say "CONTROL the shipped entries run" FAIL

# 2. residualControl
grep -q "residualControl/nuTilda" "$W/e0/err" \
    && say "no unread notice for SIMPLE/residualControl/nuTilda on an unconverged run" FAIL \
    || say "no unread notice for SIMPLE/residualControl/nuTilda on an unconverged run" ok
RC='s = re.sub(r"residualControl\s*\{[^}]*\}", "residualControl { p 2; U 2; nuTilda %s; }", s)'
stage "$AF" "$W/r1" 5 "${RC/\%s/1e-12}"
run "$W/r1"
grep -q "SIMPLE solution converged" "$W/r1/log" \
    && say "nuTilda 1e-12 (p, U at 2) gates convergence: runs to endTime" FAIL \
    || say "nuTilda 1e-12 (p, U at 2) gates convergence: runs to endTime" ok
stage "$AF" "$W/r2" 5 "${RC/\%s/2}"
run "$W/r2"
grep -q "SIMPLE solution converged in 1 iterations" "$W/r2/log" \
    && say "CONTROL nuTilda 2 converges at iteration 1" ok \
    || { say "CONTROL nuTilda 2 converges at iteration 1" FAIL; grep -m1 converged "$W/r2/log"; }

# 3. the labels
stage "$PD" "$W/l1" 2
run "$W/l1" BRAE_SIMPLE_CASE_SOLVER=1
grep -q '^DILUPBiCGStab:  Solving for Ux' "$W/l1/log" \
    && say "U on PBiCGStab + DILU is labelled DILUPBiCGStab" ok \
    || { say "U on PBiCGStab + DILU is labelled DILUPBiCGStab" FAIL; grep -m1 "Solving for Ux" "$W/l1/log"; }
stage "$AF" "$W/l2" 2
run "$W/l2" -u BRAE_SIMPLE_CASE_SOLVER
grep -q '^smoothSolver\[colourGaussSeidel\]:  Solving for Ux' "$W/l2/log" \
    && say "U under the colour rule is labelled smoothSolver[colourGaussSeidel]" ok \
    || { say "U under the colour rule is labelled smoothSolver[colourGaussSeidel]" FAIL; grep -m1 "Solving for Ux" "$W/l2/log"; }
grep -q '^smoothSolver\[GaussSeidel\]:  Solving for Ux' "$W/e0/log" \
    && say "CONTROL U in OpenFOAM's order is labelled smoothSolver[GaussSeidel]" ok \
    || say "CONTROL U in OpenFOAM's order is labelled smoothSolver[GaussSeidel]" FAIL

[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
