#!/usr/bin/env bash
# ONE RULE, TWO DRIVERS. What preconditions a substituted PBiCGStab on the transported turbulence scalars
# is decided by turbPreconFor (solvers/common/linear_solver_setup.cuh). The V2 simpleFoam driver used to
# decide it again, separately, and the two copies had drifted apart in three ways by the time this was
# written: V2 read `preconditioner == "DILU"` and nothing else, keyed its escape hatch on BRAE_DILU where
# every other driver reads BRAE_DILU_KE, and had no notion of the Neumann series at all -- so a V2 case
# naming GAMG on k and epsilon still got the bare diagonal that item 78 removed everywhere else.
#
# A copied policy does not announce that it has drifted; it just answers differently. So this runs the SAME
# case through BOTH drivers in three configurations and requires the same answer from each. The fail-proof
# is BRAE_DILU_KE: it is the hatch V2 did not read, so if V2 ever goes back to its own copy, the arm that
# forces DILU through it fails.
#
# Fixture: validation/pitzDailyTurb (kEpsilon, 12,225 cells). One second of wall time per run.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDailyTurb"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage <dir> <gamg|dilu|smooth>
stage() {
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$1/0"; else cp -r "$SRC/0" "$1/0"; fi
    python3 - "$1" "$2" <<'PYEOF'
import re, sys
d, mode = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 8;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 8;', s)
open(c, 'w').write(s + '\n')
body = {
  'gamg':   '{\n        solver          GAMG;\n        smoother        GaussSeidel;\n'
            '        tolerance       1e-08;\n        relTol          0.1;\n    }',
  'dilu':   '{\n        solver          PBiCGStab;\n        preconditioner  DILU;\n'
            '        tolerance       1e-08;\n        relTol          0.1;\n    }',
  'smooth': '{\n        solver          smoothSolver;\n        smoother        GaussSeidel;\n'
            '        nSweeps         1;\n        tolerance       1e-08;\n        relTol          0.1;\n    }',
}[mode]
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
n = 0
for fld in ('k', 'epsilon'):
    s, c2 = re.subn(r'(^\s*%s\s*\n?\s*)\{[^{}]*\}' % fld, lambda m: m.group(1) + body, s, flags=re.M)
    n += c2
assert n == 2, 'expected one k and one epsilon solver block, rewrote %d' % n
open(f, 'w').write(s)
PYEOF
}
# what each driver SAYS it preconditions with, normalised to one token
legacy_tok() { grep -oE '^(DILUPBiCGStab|Neumann[0-9]+-BiCGStab|Jacobi-BiCGStab|smoothSolver\[[a-zA-Z]+\]):  Solving for epsilon' "$1" \
               | head -1 | sed 's/:.*//; s/Neumann[0-9]*-BiCGStab/series/; s/DILUPBiCGStab/DILU/; s/Jacobi-BiCGStab/diagonal/; s/smoothSolver\[.*\]/smoothSolver/'; }
v2_tok()     { grep -oE 'k/epsilon solves:.*solver=.*' "$1" | head -1 \
               | sed -E 's/.*solver=//; s/.*degree-[0-9]+ truncated Neumann series.*/series/; s/^DILUPBiCGStab$/DILU/; s/^diagonalPBiCGStab$/diagonal/; s/^smoothSolver \+.*/smoothSolver/'; }

for mode in gamg dilu smooth; do
    stage "$W/l_$mode" "$mode"; stage "$W/v_$mode" "$mode"
    ( cd "$W/l_$mode" &&                      "$BRAE" -case "$W/l_$mode" > run.log 2>&1 ) || true
    ( cd "$W/v_$mode" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case "$W/v_$mode" > run.log 2>&1 ) || true
    L=$(legacy_tok "$W/l_$mode/run.log"); V=$(v2_tok "$W/v_$mode/run.log")
    [ -n "$L" ] && [ "$L" = "$V" ] \
        && say "\`$mode\` on k/epsilon: both drivers choose the same preconditioner" ok \
        || say "\`$mode\` on k/epsilon: both drivers choose the same preconditioner" FAIL
    printf '        (legacy %s | V2 %s)\n' "${L:-<none>}" "${V:-<none>}"
done

# ---- what each configuration must actually resolve to, or the agreement above is agreement on junk ----
stage "$W/x_gamg" gamg
( cd "$W/x_gamg" && "$BRAE" -case "$W/x_gamg" > run.log 2>&1 ) || true
[ "$(legacy_tok "$W/x_gamg/run.log")" = "series" ] \
    && say "...and a GAMG entry resolves to the Neumann series, not to the diagonal" ok \
    || say "...and a GAMG entry resolves to the Neumann series, not to the diagonal" FAIL

# ---- FAIL-PROOF: BRAE_DILU_KE is the hatch V2's own copy did not read ------------------------------
stage "$W/f_l" gamg; stage "$W/f_v" gamg
( cd "$W/f_l" &&                      BRAE_DILU_KE=1 "$BRAE" -case "$W/f_l" > run.log 2>&1 ) || true
( cd "$W/f_v" && BRAE_SIMPLEFOAM_V2=1 BRAE_DILU_KE=1 "$BRAE" -case "$W/f_v" > run.log 2>&1 ) || true
FL=$(legacy_tok "$W/f_l/run.log"); FV=$(v2_tok "$W/f_v/run.log")
[ "$FL" = "DILU" ] && [ "$FV" = "DILU" ] \
    && say "BRAE_DILU_KE reaches BOTH drivers (the hatch V2's own copy ignored)" ok \
    || say "BRAE_DILU_KE reaches BOTH drivers (the hatch V2's own copy ignored)" FAIL
printf '        (legacy %s | V2 %s)\n' "${FL:-<none>}" "${FV:-<none>}"

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
