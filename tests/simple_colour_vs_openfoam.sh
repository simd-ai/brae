#!/usr/bin/env bash
# simpleFoam's COLOUR RULE against real OpenFOAM: the legacy driver sweeps a Gauss-Seidel smoothSolver on U
# and on the transported turbulence fields in colour order on the GPU (device_simple_foam.cu,
# decideColourRule), relTol entries included -- the user's decision, 2026-10-16, on the boxy5 UAV, where the
# case-order sweeps on one CPU core were 1,670 of a 1,769 ms momentum phase and 437 of a 720 ms turbulence
# phase per iteration.
#
# THE ORACLE is OpenFOAM's CONVERGED state, validation/pitzDailyTurb/1576 (kEpsilon, residualControl 1e-6 on
# every field). A steady fixed point does not depend on the linear solver that reached it, so the fixture is
# run with its U, k and epsilon entries rewritten to the form the boxy5 case ships -- smoothSolver,
# GaussSeidel, nSweeps 2, relTol 0.1 -- which is the form the rule takes (with this driver's own sweep count,
# one sweep between residual evaluations, as the case-order path runs), and brae must CONVERGE (its own
# residualControl line, asserted) and land on OpenFOAM's answer.
#   ARM       the rule as shipped (BRAE_SIMPLE_CASE_SOLVER unset): it must announce the colour order for U, k
#             and epsilon, converge, and hold U, p, k, epsilon, nut to the bound.
#   CASE      BRAE_SIMPLE_CASE_SOLVER=1, OpenFOAM's order: no colour line; the reference distance.
#   FLOOR     BRAE_SIMPLE_COLOUR_MIN_CELLS above the mesh: the rule's floor line, no colour line.
#   CONTROL   BRAE_CONTROL_SIMPLE_COLOUR_SWAP=1, deliberately wrong: the fused U solve writes Ux's solution
#             into Uy's field and back. It must fail -- not converge, or miss the bound.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDailyTurb"
REF="$SRC/1576"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$REF" ] || { echo "SKIP: no OpenFOAM converged state at $REF"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-66s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1
    return 0
}

stage()
{
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    cp -r "$SRC/0" "$1/0"
    python3 - "$1/system/fvSolution" "$1/system/controlDict" <<'PY'
import re
import sys
fs, cd = sys.argv[1], sys.argv[2]
s = open(fs).read()
gs = '''{
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-08;
        relTol          0.1;
    }'''
for f in ('U', 'k', 'epsilon'):
    s, n = re.subn(r'(\n    %s\n    )\{[^}]*\}' % f, r'\g<1>' + gs, s)
    assert n == 1, f
# residualControl just above where these loose solves plateau. MEASURED on the GB10, 4000 iterations, both
# orders: p 1.5e-05 (OpenFOAM's order) / 2.4e-05 (colour), epsilon 5.0e-06 / 5.6e-06, U and k below 1e-07 --
# 1e-06 is never reached by either, so the converged line would be absent from both.
s, n = re.subn(r'residualControl\s*\{[^}]*\}', '''residualControl
    {
        p               5e-5;
        U               1e-5;
        "(k|epsilon)"   1e-5;
    }''', s)
assert n == 1
open(fs, 'w').write(s)
c = open(cd).read()
c = re.sub(r'(?m)^endTime\s+[^;]*;', 'endTime         4000;', c)
c = re.sub(r'(?m)^writeInterval\s+[^;]*;', 'writeInterval   4000;', c)
c = re.sub(r'(?m)^writePrecision\s+[^;]*;', 'writePrecision  15;', c)
open(cd, 'w').write(c)
PY
}

run()   # run <arm> [VAR=value ...]
{
    local a=$1
    shift
    stage "$W/$a"
    ( cd "$W/$a" && env -u BRAE_SIMPLE_CASE_SOLVER "$@" "$BRAE" -case . > log 2> err )
    echo $? > "$W/$a/rc"
}

run arm
run case BRAE_SIMPLE_CASE_SOLVER=1
run floor BRAE_SIMPLE_COLOUR_MIN_CELLS=100000000
run ctl BRAE_CONTROL_SIMPLE_COLOUR_SWAP=1

# what ran, by its own words
grep -q "colour rule: U, k, epsilon swept in colour order on the GPU" "$W/arm/log" \
    && say "ARM announces U, k and epsilon in colour order" ok || say "ARM announces U, k and epsilon in colour order" FAIL
for f in U k epsilon
do
    grep -q "solvers/$f smoother: case asks 'GaussSeidel' in OpenFOAM's index order; brae sweeps GaussSeidel in COLOUR order" "$W/arm/err" \
        && say "ARM notice for $f" ok || say "ARM notice for $f" FAIL
done
grep -q "^smoothSolver\[colourGaussSeidel\]:  Solving for Ux" "$W/arm/log" \
    && say "ARM's log names the colour sweep on U" ok || say "ARM's log names the colour sweep on U" FAIL
for a in case floor
do
    grep -qE "swept in colour order|sweeps it in COLOUR order|colourGaussSeidel" "$W/$a/log" "$W/$a/err" \
        && say "$(echo $a | tr a-z A-Z) runs OpenFOAM's order (no colour line)" FAIL \
        || say "$(echo $a | tr a-z A-Z) runs OpenFOAM's order (no colour line)" ok
done
grep -q "colour rule: 12225 cells keep OpenFOAM's order" "$W/floor/log" \
    && say "FLOOR names the rule's floor" ok || say "FLOOR names the rule's floor" FAIL

# converged, by its own line, then the fields at each arm's last time
lastdir()
{
    ls -d "$W/$1"/[0-9]* 2> /dev/null | grep -v '/0$' | awk -F/ '{print $NF, $0}' | sort -g | tail -1 | cut -d' ' -f2
}
l2()   # l2 <dir a> <dir b>: L2(a - b)/L2(b) per field
{
    python3 - "$1" "$2" <<'PY'
import re
import sys

import numpy as np


def read(fn):
    b = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', b)
    n = int(m.group(2))
    nc = 3 if m.group(1) == 'vector' else 1
    body = b[m.end():]
    body = body[:body.index('\n)')]
    return np.array(body.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)


a, b = sys.argv[1], sys.argv[2]
print(' '.join('%s=%.3e' % (f, np.linalg.norm(read(a + '/' + f) - read(b + '/' + f)) / np.linalg.norm(read(b + '/' + f)))
               for f in ('U', 'p', 'k', 'epsilon', 'nut')))
PY
}
for a in arm case
do
    grep -q "SIMPLE solution converged" "$W/$a/log" && say "$(echo $a | tr a-z A-Z) converged ($(grep -o 'in [0-9]* iterations' "$W/$a/log" | head -1))" ok \
        || { say "$(echo $a | tr a-z A-Z) converged" FAIL; echo FAIL; exit 1; }
done
ARM_OF=$(l2 "$(lastdir arm)" "$REF")
CASE_OF=$(l2 "$(lastdir case)" "$REF")
ARM_CASE=$(l2 "$(lastdir arm)" "$(lastdir case)")
# the control either writes a wrong answer or never gets to write one (diverged, the NaN guard): both fail it
CTL_DIR=$(lastdir ctl)
if [ "$(cat "$W/ctl/rc")" = 0 ] && [ -n "$CTL_DIR" ]
then
    CTL_OF=$(l2 "$CTL_DIR" "$REF")
else
    CTL_OF="U=1e+30 p=1e+30 k=1e+30 epsilon=1e+30 nut=1e+30"
fi
echo "  arm  vs OpenFOAM: $ARM_OF"
echo "  case vs OpenFOAM: $CASE_OF"
echo "  arm  vs case:     $ARM_CASE"
echo "  ctl  vs OpenFOAM: $CTL_OF   (exit $(cat "$W/ctl/rc") after $(grep -c '^Time = ' "$W/ctl/log") iterations; 1e+30 = no field written)"

# MEASURED on the GB10 (2026-10-16). OpenFOAM's order converges in 1323 iterations and the colour order in
# 1361, and both land as far from OpenFOAM's 1576 as this driver lands on this case with the fixture's own
# PBiCGStab entries (U 5.1e-03 p 1.2e-02 k 2.2e-02 epsilon 3.4e-02 nut 6.2e-02 -- a gap of the driver's own on
# pitzDailyTurb, not of the order): arm U 5.00e-03 p 1.06e-02 k 2.26e-02 epsilon 3.37e-02 nut 6.22e-02, case
# U 4.98e-03 p 1.02e-02 k 2.27e-02 epsilon 3.37e-02 nut 6.23e-02. Arm against case: U 7.7e-05 p 4.5e-04 k 3.0e-04
# epsilon 2.4e-04 nut 3.1e-04 -- two residualControl stopping points 1e-05 apart. So the bounds: the colour
# order no further from OpenFOAM than 1.05x OpenFOAM's order is, and within 1e-03 of it.
python3 - "$ARM_OF" "$CASE_OF" "$ARM_CASE" "$CTL_OF" <<'PY'
import sys


def parse(s):
    return {k: float(v) for k, v in (t.split('=') for t in s.split())}


arm_of, case_of, arm_case, ctl_of = (parse(s) for s in sys.argv[1:5])
rc = 0
for f in ('U', 'p', 'k', 'epsilon', 'nut'):
    ok = arm_case[f] < 1e-03
    print('  %-66s %s' % ('ARM vs CASE %-8s %.3e < 1e-03' % (f, arm_case[f]), 'ok' if ok else 'FAIL'))
    rc |= 0 if ok else 1
    ok = arm_of[f] < 1.05 * case_of[f]
    print('  %-66s %s' % ('ARM vs OpenFOAM %-8s %.3e < 1.05 x %.3e' % (f, arm_of[f], case_of[f]), 'ok' if ok else 'FAIL'))
    rc |= 0 if ok else 1
bad = ctl_of['U'] > 1.05 * case_of['U'] + 1e-03
print('  %-66s %s' % ('CONTROL (components swapped) misses: U %.3e' % ctl_of['U'], 'ok' if bad else 'FAIL'))
rc |= 0 if bad else 1
sys.exit(rc)
PY
[ $? -eq 0 ] || fail=1
[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
