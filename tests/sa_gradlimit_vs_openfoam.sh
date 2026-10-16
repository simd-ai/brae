#!/usr/bin/env bash
# The case's gradient limiters on Spalart-Allmaras, legacy simpleFoam driver, against real OpenFOAM v2412.
#
# The driver ran nuTilda's equation with every gradient UNLIMITED whatever the case asked: linearUpwind's named
# grad(nuTilda) (the parser read the limiter for div(phi,k) only), the non-orthogonal laplacian correction's
# grad(nuTilda) (correctedSnGrad.C:52-55 takes the field's own gradSchemes entry), and the grad(U) behind
# Omega and Stilda (SpalartAllmarasBase.C:461), which the call passed as 0. On the boxy5 UAV, whose fvSchemes
# limit all three (`cellLimited Gauss linear 1`), nuTilda blew up at -3, -6 and -9 degrees on the cells of its
# four highly skewed faces (skewness 4.85) by iteration ~150; OpenFOAM v2412 and OpenFOAM-13 run on.
#
# THE ORACLE is serial OpenFOAM v2412 simpleFoam on validation/airFoil2D with those three entries added,
# every solve pinned: OpenFOAM runs 0 -> 3, both codes restart from ITS time 3 and run to 6.
# THE CONTROL is BRAE_CONTROL_SA_GRAD_UNLIMITED=1, the unlimited gradients this replaced, which must miss.
set -u
SRC="${1:?case dir (validation/airFoil2D)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
OFRC=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -f "$OFRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFRC"; exit 77; }
set +u
set --
source "$OFRC" > /dev/null 2>&1
set -u
command -v simpleFoam > /dev/null || { echo "SKIP: simpleFoam not on PATH"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "$W"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
rc=0
say()
{
    printf '  %-60s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && rc=1
    return 0
}

# stage <dir> <endTime>: the fixture with every solve pinned (tolerance 1e-12, relTol 0), no residualControl,
# no function objects, ascii at 17 digits, a write every iteration
stage()
{
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    cp -r "$SRC/0" "$1/0"
    # the boxy5 UAV's three limiters
    python3 - "$1/system/fvSchemes" <<'PY'
import re
import sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(gradSchemes\s*\{[^}]*?default\s+[^;]*;)',
               r'\1\n    grad(U)         cellLimited Gauss linear 1;\n    grad(nuTilda)   cellLimited Gauss linear 1;', s)
assert n == 1
open(p, 'w').write(s)
PY
    grep -q 'grad(nuTilda)   cellLimited' "$1/system/fvSchemes" || { echo "FAIL: limiters not staged"; exit 1; }
    python3 - "$1/system/controlDict" "$1/system/fvSolution" "$2" <<'PY'
import re
import sys
cd, fs, end = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(cd).read()
for k, v in (('endTime', end), ('writeInterval', '1'), ('writePrecision', '17'), ('writeFormat', 'ascii'),
             ('writeCompression', 'off'), ('startFrom', 'latestTime')):
    s, n = re.subn(r'(?m)^%s\s+[^;]*;' % k, '%-16s %s;' % (k, v), s)
    if n == 0:
        s = s.replace('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //',
                      '%-16s %s;\n' % (k, v), 1)
i = s.find('functions')
if i >= 0:
    j = s.index('{', i)
    depth = 0
    for e in range(j, len(s)):
        depth += {'{': 1, '}': -1}.get(s[e], 0)
        if depth == 0:
            break
    s = s[:i] + 'functions {}' + s[e + 1:]
open(cd, 'w').write(s)
f = open(fs).read()
f = re.sub(r'(?m)^(\s*tolerance\s+)[^;]*;', r'\g<1>1e-12;', f)
f = re.sub(r'(?m)^(\s*relTol\s+)[^;]*;', r'\g<1>0;', f)
f = re.sub(r'(?m)^(\s*smoother\s+GaussSeidel;)', r'\1\n        maxIter         5000;', f)
f = re.sub(r'residualControl\s*\{[^}]*\}', '', f)
open(fs, 'w').write(f)
PY
}

# 1. OpenFOAM 0 -> 3, the restart state
stage "$W/seed" 3
( cd "$W/seed" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM's seed run"; tail -5 "$W/seed/log"; exit 1; }
[ -d "$W/seed/3" ] || { echo "FAIL: the seed run wrote no time 3"; exit 1; }
arm()
{
    stage "$W/$1" 6
    rm -rf "${W:?}/$1/0"
    cp -r "$W/seed/3" "$W/$1/3"
    rm -rf "${W:?}/$1/3/uniform"
}
arm of
( cd "$W/of" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM 3 -> 6"; tail -5 "$W/of/log"; exit 1; }
for a in brae ctl
do
    arm "$a"
done
( cd "$W/brae" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2> err ) \
    || { echo "FAIL: brae 3 -> 6"; tail -5 "$W/brae/err"; exit 1; }
( cd "$W/ctl" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_SA_GRAD_UNLIMITED=1 "$BRAE" -case . > log 2> err ) \
    || { echo "FAIL: brae control 3 -> 6"; tail -5 "$W/ctl/err"; exit 1; }

diffs()
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
               for f in ('U', 'p', 'nuTilda')))
PY
}
B4=$(diffs "$W/brae/4" "$W/of/4")
B6=$(diffs "$W/brae/6" "$W/of/6")
C4=$(diffs "$W/ctl/4" "$W/of/4")
C6=$(diffs "$W/ctl/6" "$W/of/6")
echo "  arm     t4: $B4"
echo "  arm     t6: $B6"
echo "  control t4: $C4"
echo "  control t6: $C6"
# MEASURED on the GB10 (2026-10-16), L2 relative to OpenFOAM: the arm nuTilda 5.0e-05 at t4 and 4.3e-03 at t6
# (U 2.9e-05 and p 2.6e-05 at t4, the turbulent wall rows' own gap, the same in both arms); the control nuTilda
# 2.3e-01 at t4 and 4.5e+00 at t6. Bounds at about twice the arm, the control three orders past the first.
python3 - "$B4" "$B6" "$C4" "$C6" <<'PY'
import sys


def parse(s):
    return {k: float(v) for k, v in (t.split('=') for t in s.split())}


b4, b6, c4, c6 = (parse(s) for s in sys.argv[1:5])
rc = 0
for name, v, bound in (('ARM nuTilda t4', b4['nuTilda'], 1e-4), ('ARM nuTilda t6', b6['nuTilda'], 1e-2),
                       ('ARM U t6', b6['U'], 3e-4), ('ARM p t6', b6['p'], 6e-4)):
    ok = v < bound
    print('  %-40s %.3e < %.0e   %s' % (name, v, bound, 'ok' if ok else 'FAIL'))
    rc |= 0 if ok else 1
ok = c4['nuTilda'] > 1e-2
print('  %-40s %.3e > 1e-02   %s' % ('CONTROL nuTilda t4 misses', c4['nuTilda'], 'ok' if ok else 'FAIL'))
rc |= 0 if ok else 1
print('PASS' if rc == 0 else 'FAIL')
sys.exit(rc)
PY
