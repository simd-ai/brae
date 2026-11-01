#!/usr/bin/env bash
# The transported scalars' laplacian under `uncorrected`. uncorrectedSnGrad.H returns nonOrthDeltaCoeffs() with no correction; the scalar scaffold read only `nonOrth`, which is the correction, so
# k, epsilon and omega took orthogonal's deltaCoeffs under that name, while momentum and pressure read `nonOrth || nonOrthCoeffs`. MEASURED (JUPITER job 2637170), pitzDailyTurb, two tight iterations
# from OpenFOAM's iteration 100: epsilon 1.0e-04, k 1.8e-05, nut 2.2e-05, U 2.3e-06, p 9.5e-06 from OpenFOAM.
#
# ARMS: pitzDailyTurb (kEpsilon) and pitzDailySST (kOmegaSST, BRAE_SST_F1_PATCH=1: without it the inlet's F1 is its own gap), laplacian and snGrad each `uncorrected`, then each `orthogonal` and
# `corrected`, whose answers this must not move. OpenFOAM to iteration 50, then two iterations from its bytes in each code, every solve tight (p 1e-12, the rest 1e-13), OpenFOAM's sweep order.
# THE ORACLE: real OpenFOAM on the same bytes. On JUPITER use a build that passes brae-multigpu's scripts/jsc/of_oracle.sh.
# THE BOUND: every field within 1e-10 relative L2, each arm.
# THE `corrected` ARMS FIX THE OUTLET of epsilon or omega at the cells' own values at 50, in both codes. KNOWN GAP, not this gate's: OpenFOAM's wall function overwrites the cell in the wall and outlet
# corner, and its zeroGradient outlet face keeps the value stored before that, which grad(<field>) reads; brae re-evaluates the face from the overwritten cell, and only `corrected` reads that gradient.
# MEASURED (JUPITER jobs 2637505, 2638428), two iterations: zeroGradient epsilon 2.3e-08, omega 2.4e-08, 100% and 99.8% of it in the outlet cells and their neighbours; fixed, 9.0e-13 and 1.8e-12.
# CONTROL, asserted red: BRAE_CONTROL_SCALAR_ORTHOGONAL_DELTA=1 on kEpsilon's `uncorrected`, the old coefficients.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
of()
{
    bash -c "source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"
}
cat > "$W/gate.py" <<'PY'
import re
import sys

import numpy as np


def stage(case, scheme, turb):
    p = case + '/system/fvSchemes'
    t = open(p).read()
    t, a = re.subn(r'(laplacianSchemes\s*\{\s*default\s+Gauss linear )\w+;', r'\g<1>%s;' % scheme, t)
    t, b = re.subn(r'(snGradSchemes\s*\{\s*default\s+)\w+;', r'\g<1>%s;' % scheme, t)
    assert a == 1 and b == 1, (a, b)
    open(p, 'w').write(t)
    p = case + '/system/fvSolution'
    s = open(p).read()
    s, n = re.subn(r'solvers\s*\{.*?\n\}', '''solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0; maxIter 5000; }
    "(U|k|%s)" { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-13; relTol 0; maxIter 2000; }
}''' % turb, s, count=1, flags=re.S)
    assert n == 1
    s = re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl\n    {\n    }', s)
    open(p, 'w').write(s)
    control(case, 0, 50)


def control(case, start, end):
    p = case + '/system/controlDict'
    c = open(p).read()
    for key, val in (('startTime', start), ('endTime', end), ('writeInterval', 1), ('writePrecision', 17)):
        c, n = re.subn(r'(?m)^%s\s+[^;]*;' % key, '%s %s;' % (key, val), c)
        assert n == 1, key
    c = re.sub(r'(?m)^startFrom\s+[^;]*;', 'startFrom startTime;', c)
    c = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', c, flags=re.S)
    open(p, 'w').write(c)


def read(fn):
    b = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', b)
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    body = b[m.end():]
    body = body[:body.index('\n)')]
    return np.array(body.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)


def fix_outlet(case, turb):
    p = case + '/50/' + turb
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n?\(', s)
    body = s[m.end():]
    cells = body[:body.index('\n)')].split()
    own = open(case + '/constant/polyMesh/owner').read()
    o = re.search(r'\n(\d+)\s*\n\(', own)
    rest = own[o.end():]
    own = rest[:rest.index('\n)')].split()
    bd = open(case + '/constant/polyMesh/boundary').read()
    b = re.search(r'\n\s+outlet\n\s+\{[^}]*?nFaces\s+(\d+);[^}]*?startFace\s+(\d+);', bd)
    n, st = int(b.group(1)), int(b.group(2))
    vals = ' '.join(cells[int(x)] for x in own[st:st + n])
    i = s.index('boundaryField')
    t, k = re.subn(r'(\n\s+outlet\n\s+\{)[^}]*(\})', lambda q: q.group(1) + '\n        type fixedValue;\n        value nonuniform List<scalar> %d(%s);\n    ' % (n, vals) + q.group(2), s[i:])
    assert k == 1
    open(p, 'w').write(s[:i] + t)


what = sys.argv[1]
if what == 'stage':
    stage(*sys.argv[2:])
elif what == 'fix_outlet':
    fix_outlet(*sys.argv[2:])
elif what == 'control':
    control(sys.argv[2], sys.argv[3], sys.argv[4])
else:
    of, br, turb, want = sys.argv[2:6]
    gaps = {f: np.linalg.norm(read('%s/52/%s' % (br, f)) - read('%s/52/%s' % (of, f))) / np.linalg.norm(read('%s/52/%s' % (of, f))) for f in ('U', 'p', 'k', turb, 'nut')}
    ok = max(gaps.values()) <= 1e-10
    print('%s %s' % ('within' if ok else 'MISSES', ' '.join('%s %.2e' % kv for kv in gaps.items())))
    sys.exit(0 if ok == (want != 'miss') else 1)
PY
fail=0
for arm in "pitzDailyTurb epsilon" "pitzDailySST omega"
do
    read -r fixture turb <<< "$arm"
    for scheme in uncorrected orthogonal corrected
    do
        a=$W/$fixture-$scheme
        dev=$a/dev
        mkdir -p "$dev"
        cp -r "$ROOT/validation/$fixture/0" "$ROOT/validation/$fixture/constant" "$ROOT/validation/$fixture/system" "$dev/"
        find "$dev" -name '.brae_*' -delete
        python3 -B "$W/gate.py" stage "$dev" "$scheme" "$turb" || { echo "FAIL staging $fixture $scheme"; fail=1; continue; }
        of "cd '$dev' && simpleFoam > log 2>&1" < /dev/null || { echo "FAIL OpenFOAM's development run, $fixture $scheme"; fail=1; continue; }
        for who in of brae ctl
        do
            c=$a/$who
            mkdir "$c"
            cp -r "$dev/constant" "$dev/system" "$dev/50" "$c/"
            rm -rf "${c:?}/50/uniform"
            python3 -B "$W/gate.py" control "$c" 50 52
            [ "$scheme" = corrected ] && python3 -B "$W/gate.py" fix_outlet "$c" "$turb"
        done
        of "cd '$a/of' && simpleFoam > log 2>&1" < /dev/null || { echo "FAIL OpenFOAM $fixture $scheme"; fail=1; continue; }
        ( cd "$a/brae" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_SST_F1_PATCH=1 "$BRAE" -case . > log 2> err < /dev/null ) || { echo "FAIL brae $fixture $scheme"; fail=1; continue; }
        printf '  %-14s %-12s ' "$fixture" "$scheme"
        python3 -B "$W/gate.py" compare "$a/of" "$a/brae" "$turb" hold || fail=1
        if [ "$fixture" = pitzDailyTurb ] && [ "$scheme" = uncorrected ]
        then
            ( cd "$a/ctl" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_SCALAR_ORTHOGONAL_DELTA=1 "$BRAE" -case . > log 2> err < /dev/null ) || { echo "FAIL control"; fail=1; continue; }
            printf '  CONTROL old coefficients      '
            python3 -B "$W/gate.py" compare "$a/of" "$a/ctl" "$turb" miss || fail=1
        fi
    done
done
[ "$fail" = 0 ] && echo "PASS: the transported scalars' laplacian takes the case's coefficients" || { echo FAIL; exit 1; }
