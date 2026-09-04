#!/usr/bin/env bash
# brae SAYS that it substitutes OpenFOAM's Gauss-Seidel sweep.
#
# `smoothSolver` with a `symGaussSeidel` smoother is what almost every turbulent OpenFOAM tutorial asks
# for on U, k and omega. brae routes it to deviceSymGaussSeidel, which is OpenFOAM's smoothSolver
# STOPPING RULE (smoothSolver.C:135-209) around a MULTICOLOUR sweep -- where symGaussSeidelSmoother.C
# walks cells in strict index order (:147 forward, :176 reverse). That is the same algorithm under a
# permutation, but Gauss-Seidel is order-dependent, so the iterate after n sweeps is not OpenFOAM's:
# measured on validation/T3A restarted from its 269 fixture, on the case's own relTol 0.1, OpenFOAM cut
# Ux 1.6186e-05 -> 6.940e-07 in ONE sweep while brae took SEVEN to reach 1.278e-06.
#
# It was announced NOWHERE. The shared reader (linear_solver_setup.cuh) suppressed the smoother notice
# precisely when brae took that path, tests/test_solver_notices.cu asserted that silence, and the V2 run
# log said "smoothSolver/symGaussSeidel (as the case asks)".
#
#   POSITIVE  a case naming smoothSolver+symGaussSeidel must produce the notice, naming the smoother and
#             the substitution.
#   CONTROL   the same case with PBiCGStab/DILU instead must NOT produce it -- otherwise the notice is
#             unconditional and says nothing.
#   CONTROL 2 the GAMG-on-p notice must appear in BOTH, so the control's silence is about the smoother
#             and not about the envelope having gone quiet.
#
# Fail-proof, 2026-09-04: with the notices.push_back wrapped in `if (false)`, POSITIVE fails and both
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

cp -r "$SRC" "$W/gs"; cp -r "$SRC" "$W/krylov"
# The control: the same fields, asking for a solver brae substitutes in a DIFFERENT way, so the smoother
# entry disappears entirely and only the Krylov notice is left to fire.
python3 - "$W/krylov" <<'PY'
import re, sys
f = sys.argv[1] + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'solver\s+smoothSolver;', 'solver          PBiCGStab;', s)
s = re.sub(r'smoother\s+symGaussSeidel;', 'preconditioner  DILU;', s)
open(f, 'w').write(s)
PY
grep -q "smoothSolver" "$W/krylov/system/fvSolution" && { echo "FAIL: the control still names smoothSolver"; exit 1; }

"$PROBE" "$W/gs"     > "$W/gs.out"     2>&1 || { echo "FAIL: probe crashed on the fixture"; exit 1; }
"$PROBE" "$W/krylov" > "$W/krylov.out" 2>&1 || { echo "FAIL: probe crashed on the control"; exit 1; }

grep -q "symGaussSeidel" "$W/gs.out" && grep -q "MULTICOLOUR" "$W/gs.out" && grep -q "SMOOTHS LESS" "$W/gs.out" \
    && say "POSITIVE  the smoother substitution is announced" ok \
    || say "POSITIVE  the smoother substitution is announced" FAIL
grep -q "MULTICOLOUR" "$W/krylov.out" \
    && say "CONTROL   PBiCGStab/DILU does NOT announce a multicolour sweep" FAIL \
    || say "CONTROL   PBiCGStab/DILU does NOT announce a multicolour sweep" ok
grep -q "AMG-preconditioned PCG" "$W/gs.out" && grep -q "AMG-preconditioned PCG" "$W/krylov.out" \
    && say "CONTROL 2 the GAMG notice still fires in both, so the envelope is reporting" ok \
    || say "CONTROL 2 the GAMG notice still fires in both, so the envelope is reporting" FAIL

[ $fail -eq 0 ] || { echo; echo "--- fixture ---"; cat "$W/gs.out"; echo "--- control ---"; cat "$W/krylov.out"; }
[ $fail -eq 0 ] && echo "PASS: brae announces the Gauss-Seidel sweep substitution"
exit $fail
