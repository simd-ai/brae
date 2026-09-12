#!/usr/bin/env bash
# The Spalart-Allmaras branch of the shared linear-solver reader never set the PRECONDITIONER.
#
# readLinearSolverControls assigns ctl.diluKE only in its k/epsilon branch. On the SA branch the field
# kept its default of false, so a case naming `solver PBiCGStab; preconditioner DILU;` on nuTilda -- which
# 37 of the 43 SpalartAllmaras tutorials shipped with OpenFOAM v2412 do (4 name smoothSolver, 2 PBiCG,
# none GAMG) -- had that entry read and then ignored, and brae solved with the plain diagonal.
#
# It announced nothing, and WHY is the defect: noticeSolverChoice compares the case's `preconditioner`
# against what diluHere() answers; diluHere() wires "nuTilda" and answered DILU; so prec == braePrecon and
# the notice stayed silent over a solve that was not doing it. A capability the shared reader reports and
# one caller never applied -- the shape item 58 found on the rho mirror, here on the SA path.
#
# THE ORACLE is real simpleFoam on the same case. What it pins is the per-iteration nuTilda solve: with
# the case's own preconditioner both codes take the SAME number of BiCGStab iterations and land within a
# few percent of each other, and without it brae takes three to four times as many and stops five to
# eight times further out. That gap is the fail-proof.
#
# Fixture: validation/airFoil2D (the SA gate case), its nuTilda entry rewritten to the PBiCGStab/DILU form
# OpenFOAM's own tutorials use -- the shipped fixture names a smoothSolver, which takes no preconditioner
# and so cannot see this at all. Arm 4 runs the shipped form as the control.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/airFoil2D"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set +u; source "$OFBASHRC" > /dev/null 2>&1 || true; set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
ITERS=6

fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage <dir> <nuTilda solver body, or "" for the fixture's own>
stage() {
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$1/0"; else cp -r "$SRC/0" "$1/0"; fi
    python3 - "$1" "$ITERS" "${2:-}" <<'PYEOF'
import re, sys
d, it, body = sys.argv[1], sys.argv[2], sys.argv[3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % it, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % it, s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
if body:
    m = re.search(r'nuTilda\s*\n?\s*\{[^{}]*\}', s)
    assert m, 'the fixture no longer carries a nuTilda solver block'
    s = s.replace(m.group(0), 'nuTilda\n    {\n' + body + '\n    }', 1)
open(f, 'w').write(s)
PYEOF
}
# nuiters <log> -> the per-iteration nuTilda BiCGStab counts, space separated
nuiters() { grep -oE 'Solving for nuTilda, Initial residual = [0-9.e+-]+, Final residual = [0-9.e+-]+, No Iterations [0-9]+' "$1" \
            | grep -oE 'No Iterations [0-9]+' | grep -oE '[0-9]+' | tr '\n' ' '; }
nures()   { grep -oE 'Solving for nuTilda, Initial residual = [0-9.e+-]+' "$1" | grep -oE '= [0-9.e+-]+' | cut -c3- | tr '\n' ' '; }

DILU_BODY='        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-08;
        relTol          0.1;'

# ---- the ORACLE ----------------------------------------------------------------------------------
stage "$W/of" "$DILU_BODY"
( cd "$W/of" && simpleFoam > run.log 2>&1 ) || { tail -5 "$W/of/run.log"; echo "FAIL: OpenFOAM did not run"; exit 1; }
OF_IT=$(nuiters "$W/of/run.log"); OF_RES=$(nures "$W/of/run.log")
[ -n "$OF_IT" ] || { echo "SKIP: OpenFOAM logged no nuTilda solve"; exit 77; }
printf '        (OpenFOAM nuTilda iterations: %s)\n' "$OF_IT"

# ---- ARM 1: the case's own preconditioner is honoured, and the trajectory is OpenFOAM's -----------
stage "$W/on" "$DILU_BODY"
( cd "$W/on" && "$BRAE" -case "$W/on" > run.log 2>&1 ) || { tail -5 "$W/on/run.log"; say "brae runs the case" FAIL; }
BR_IT=$(nuiters "$W/on/run.log"); BR_RES=$(nures "$W/on/run.log")
[ "$BR_IT" = "$OF_IT" ] && say "brae takes OpenFOAM's nuTilda iteration count, every iteration" ok \
                        || say "brae takes OpenFOAM's nuTilda iteration count, every iteration" FAIL
printf '        (brae: %s)\n' "$BR_IT"
python3 - "$OF_RES" "$BR_RES" <<'PYEOF' || fail=1
import sys
o = [float(x) for x in sys.argv[1].split()]
b = [float(x) for x in sys.argv[2].split()]
n = min(len(o), len(b))
assert n, 'no residuals'
worst = max(abs(b[i] - o[i]) / o[i] for i in range(n))
ok = worst <= 0.10
print('  %-70s %s' % ("...and its initial residuals within 10%% of OpenFOAM's (worst %.1f%%)" % (100*worst),
                      'ok' if ok else 'FAIL'))
raise SystemExit(0 if ok else 1)
PYEOF

# ---- ARM 2 (FAIL-PROOF): ignoring the preconditioner must NOT clear arm 1 -------------------------
# This is the behaviour that shipped. If it passed arm 1's checks, arm 1 would be proving nothing.
stage "$W/off" "$DILU_BODY"
( cd "$W/off" && BRAE_DILU_KE=0 BRAE_POLY_KE=1 "$BRAE" -case "$W/off" > run.log 2>&1 ) || true
OFF_IT=$(nuiters "$W/off/run.log")
[ "$OFF_IT" != "$OF_IT" ] && say "...and the diagonal does NOT reproduce it (fail-proof)" ok \
                          || say "...and the diagonal does NOT reproduce it (fail-proof)" FAIL
printf '        (diagonal: %s)\n' "$OFF_IT"

# ---- ARM 3: the log names the preconditioner, as OpenFOAM's own prefix does -----------------------
grep -q "^DILUPBiCGStab:  Solving for nuTilda" "$W/on/run.log" \
    && say "the solve line reads DILUPBiCGStab, the prefix OpenFOAM prints" ok \
    || { grep -m1 "Solving for nuTilda" "$W/on/run.log"; say "the solve line reads DILUPBiCGStab, the prefix OpenFOAM prints" FAIL; }
grep -q "^Jacobi-BiCGStab:  Solving for nuTilda" "$W/off/run.log" \
    && say "...and reads Jacobi-BiCGStab when it is one (the label tracks the solve)" ok \
    || { grep -m1 "Solving for nuTilda" "$W/off/run.log"; say "...and reads Jacobi-BiCGStab when it is one (the label tracks the solve)" FAIL; }

# ---- ARM 4 (control): the SHIPPED fixture names a smoothSolver and is untouched -------------------
stage "$W/ship" ""
( cd "$W/ship" && "$BRAE" -case "$W/ship" > run.log 2>&1 ) || true
grep -q "^smoothSolver\[.*\]:  Solving for nuTilda" "$W/ship/run.log" \
    && say "the shipped smoothSolver fixture still runs a smoothSolver (control)" ok \
    || { grep -m1 "Solving for nuTilda" "$W/ship/run.log"; say "the shipped smoothSolver fixture still runs a smoothSolver (control)" FAIL; }

# ---- ARM 5: a solver that carries NO preconditioner takes the Neumann series ----------------------
# No OpenFOAM SA tutorial writes GAMG on nuTilda, but a user can, and it leaves the same blank the
# k/epsilon pair's does. airFoil2D relaxes nuTilda by 0.7, so fvMatrix::relax bounds the series.
stage "$W/gamg" '        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-08;
        relTol          0.1;'
( cd "$W/gamg" && "$BRAE" -case "$W/gamg" > run.log 2>&1 ) || true
# The degree is DERIVED from the case's own relaxation, d = ceil(ln(0.1)/ln(alpha)); airFoil2D relaxes
# nuTilda by 0.7, so 7 -- a different number from the 22 the alpha-0.9 fixtures derive, which is the
# point: pinning it here asserts the rule rather than a constant.
grep -q "^Neumann7-BiCGStab:  Solving for nuTilda" "$W/gamg/run.log" \
    && say "a GAMG entry on nuTilda takes the degree-7 series (derived from alpha 0.7)" ok \
    || { grep -m1 "Solving for nuTilda" "$W/gamg/run.log"; say "a GAMG entry on nuTilda takes the degree-7 series (derived from alpha 0.7)" FAIL; }

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
