#!/usr/bin/env bash
# simpleFoam's COLOUR RULE against real OpenFOAM: the legacy driver sweeps a Gauss-Seidel smoothSolver on U and on the transported turbulence fields in colour order on the GPU
# (device_simple_foam.cu, decideColourRule), relTol entries included -- the user's decision, 2026-10-16, on the boxy5 UAV, where the case-order sweeps on one CPU core were 1,670 of a
# 1,769 ms momentum phase and 437 of a 720 ms turbulence phase per iteration.
#
# A steady fixed point does not depend on the linear solver that reached it, so pitzDailyTurb runs with its U, k and epsilon entries in the form the boxy5 case ships -- smoothSolver,
# GaussSeidel, nSweeps 2, relTol 0.1 -- which is the form the rule takes.
# THE ORACLE is OpenFOAM's own converged state at residualControl 1e-10, run here on the fixture's Krylov entries, every linear tolerance 100 times under the stopping point.
#   ARM       the rule as shipped (BRAE_SIMPLE_CASE_SOLVER unset), residualControl 1e-8: announces the colour order for U, k and epsilon and converges
#   CASE      BRAE_SIMPLE_CASE_SOLVER=1, OpenFOAM's order, residualControl 1e-8: no colour line; the reference distance
#   FLOOR     BRAE_SIMPLE_COLOUR_MIN_CELLS above the mesh: the rule's floor line, no colour line
# THE BOUNDS, each field
#   the order rule  ARM no further from OpenFOAM than 1.05 x CASE
#   between orders  ARM within 1e-7 of CASE. MEASURED on JUPITER (jobs 2618805, 2618806), 10 runs, identical: U 8.0e-09 p 1.9e-08 k 3.4e-08 epsilon 1.9e-08 nut 4.1e-08
#   residuals       OpenFOAM's first initial residuals on ARM's and CASE's written fields, one simpleFoam iteration restarted from them, each at most 1.05 x the stopping point: no brae code
#                   measures that number. MEASURED, the same runs: worst p, 1.02e-08 on ARM, 9.86e-09 on CASE
# MEASURED, why this form (job 2615279): both orders and OpenFOAM reach one fixed point, every distance falling 10 times a decade from 1e-5 to 1e-9, and the order rule holding throughout, worst
# ratio 1.016. Until 2026-11-01 this gate stopped at 5e-5 and 1e-5 and held a reference at 1e-6, where stopping-point differences are the size of the distance itself (p 1.55e-03 against 1.44e-03,
# ratio 1.08, once epsilon stopped running on div(phi,omega)).
# CONTROLS, each must fail
#   BRAE_CONTROL_SIMPLE_COLOUR_SWAP=1      the fused U solve writes Ux's solution into Uy's field and back. MEASURED: diverges at iteration 91
#   BRAE_CONTROL_SCHEME_SECOND_BOTH=1      epsilon on omega's scheme, the parser's old defect. MEASURED: never converges; nut 6.19e-02 from the oracle at any tolerance
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDailyTurb"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
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
of()
{
    bash -c "source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"
}

cat > "$W/gate.py" <<'PY'
import os
import re
import shutil
import sys

import numpy as np

FIELDS = ('U', 'p', 'k', 'epsilon', 'nut')
ENTRY = {'gs': '{\n        solver smoothSolver;\n        smoother GaussSeidel;\n        nSweeps 2;\n        tolerance %s;\n        relTol 0.1;\n    }',
         'krylov': '{\n        solver PBiCGStab;\n        preconditioner DILU;\n        tolerance %s;\n        relTol 0;\n    }'}


def controls(case, end, rc):
    fs = case + '/system/fvSolution'
    s = open(fs).read()
    body = '' if rc is None else 'p %s;\n        U %s;\n        "(k|epsilon)" %s;' % (rc, rc, rc)
    s, n = re.subn(r'residualControl\s*\{[^}]*\}', 'residualControl\n    {\n        %s\n    }' % body, s)
    assert n == 1
    open(fs, 'w').write(s)
    cd = case + '/system/controlDict'
    c = open(cd).read()
    for key, value in (('endTime', end), ('writeInterval', end), ('writePrecision', 17)):
        c, n = re.subn(r'(?m)^%s\s+[^;]*;' % key, '%s %s;' % (key, value), c)
        assert n == 1, key
    open(cd, 'w').write(c)


def stage(src, case, rc, kind):
    os.mkdir(case)
    for d in ('constant', 'system', '0'):
        shutil.copytree(src + '/' + d, case + '/' + d)
    tol = '%.0e' % (float(rc) * 1e-2)
    fs = case + '/system/fvSolution'
    s = open(fs).read()
    for f in ('U', 'k', 'epsilon'):
        s, n = re.subn(r'(\n    %s\n    )\{[^}]*\}' % f, lambda m: m.group(1) + ENTRY[kind] % tol, s)
        assert n == 1, f
    s, n = re.subn(r'(\n    p\n    \{[^}]*?tolerance\s+)[^;]*;', lambda m: m.group(1) + tol + ';', s)
    assert n == 1
    open(fs, 'w').write(s)
    controls(case, 20000, rc)


def last(case):
    t = [d for d in os.listdir(case) if re.fullmatch(r'[0-9.e+-]+', d) and d != '0' and os.path.isfile(case + '/' + d + '/U')]
    return max(t, key=float) if t else None


def recheck(case):
    t = last(case)
    if t is None:
        return 1
    out = case + '.recheck'
    os.mkdir(out)
    for d in ('constant', 'system'):
        shutil.copytree(case + '/' + d, out + '/' + d)
    shutil.copytree(case + '/' + t, out + '/' + t)
    controls(out, int(float(t)) + 1, None)
    c = open(out + '/system/controlDict').read()
    open(out + '/system/controlDict', 'w').write(re.sub(r'(?m)^startFrom\s+[^;]*;', 'startFrom latestTime;', c))
    return 0


def read(fn):
    b = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', b)
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    body = b[m.end():]
    body = body[:body.index('\n)')]
    return np.array(body.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)


def fields(case):
    t = last(case)
    return None if t is None else {f: read('%s/%s/%s' % (case, t, f)) for f in FIELDS}


def dist(a, b, f):
    return np.linalg.norm(a[f] - b[f]) / np.linalg.norm(b[f])


def residuals(case):
    found = {}
    log = case + '.recheck/log'
    if os.path.isfile(log):
        for line in open(log, errors='replace'):
            m = re.search(r'Solving for (Ux|Uy|p|k|epsilon), Initial residual = ([0-9.e+-]+)', line)
            if m and m.group(1) not in found:
                found[m.group(1)] = float(m.group(2))
    return found


def converged(case):
    m = re.search(r'SIMPLE solution converged in (\d+) iterations', open(case + '/log', errors='replace').read())
    return int(m.group(1)) if m else None


def judge(w):
    fails = 0

    def say(ok, text):
        nonlocal fails
        print('  %-108s %s' % (text, 'ok' if ok else 'FAIL'))
        fails += 0 if ok else 1

    for name, text in (('ref', 'the reference, OpenFOAM at 1e-10'), ('arm', 'ARM'), ('case', 'CASE')):
        say(converged(w + '/' + name) is not None, '%s converged, %s iterations' % (text, converged(w + '/' + name)))
    ref, arm, case = fields(w + '/ref'), fields(w + '/arm'), fields(w + '/case')
    if ref is None or arm is None or case is None:
        print('FAIL')
        return 1
    for f in FIELDS:
        a, c = dist(arm, ref, f), dist(case, ref, f)
        say(a < 1.05 * c, 'the order rule %-8s ARM %.3e < 1.05 x CASE %.3e (ratio %.3f)' % (f, a, c, a / c))
    for f in FIELDS:
        d = dist(arm, case, f)
        say(d < 1e-7, 'between orders %-8s %.3e < 1e-07' % (f, d))
    for name, limit in (('arm', 1.05e-8), ('case', 1.05e-8), ('ref', 1.05e-10)):
        r = residuals(w + '/' + name)
        say(len(r) == 5 and max(r.values()) <= limit, "OpenFOAM's residuals on %-5s %s (at most %.3g)" % (name, ' '.join('%s %.2e' % kv for kv in sorted(r.items())), limit))
    for name, text in (('ctlswap', 'components swapped'), ('ctlparser', 'epsilon on omega\'s scheme')):
        c = fields(w + '/' + name)
        passed = converged(w + '/' + name) is not None and c is not None
        if passed:
            passed = all(dist(c, ref, f) < 1.05 * dist(case, ref, f) for f in FIELDS) and all(dist(c, case, f) < 1e-7 for f in FIELDS)
        far = 'no fields' if c is None else 'nut %.2e' % dist(c, ref, 'nut')
        say(not passed, 'CONTROL %s misses: converged %s, %s' % (text, converged(w + '/' + name), far))
    print('PASS' if not fails else 'FAIL')
    return 1 if fails else 0


what = sys.argv[1]
if what == 'stage':
    stage(*sys.argv[2:6])
elif what == 'recheck':
    sys.exit(recheck(sys.argv[2]))
else:
    sys.exit(judge(sys.argv[2]))
PY
python3 -B "$W/gate.py" stage "$SRC" "$W/ref" 1e-10 krylov
for a in arm case floor ctlswap ctlparser
do
    python3 -B "$W/gate.py" stage "$SRC" "$W/$a" 1e-8 gs
done
of "cd '$W/ref' && simpleFoam > log 2>&1" < /dev/null
run()   # run <arm> [VAR=value ...]
{
    local a=$1
    shift
    ( cd "$W/$a" && env -u BRAE_SIMPLE_CASE_SOLVER "$@" "$BRAE" -case . > log 2> err < /dev/null )
    echo $? > "$W/$a/rc"
}
run arm
run case BRAE_SIMPLE_CASE_SOLVER=1
run floor BRAE_SIMPLE_COLOUR_MIN_CELLS=100000000
run ctlswap BRAE_CONTROL_SIMPLE_COLOUR_SWAP=1
run ctlparser BRAE_CONTROL_SCHEME_SECOND_BOTH=1

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

for a in arm case ctlswap ctlparser ref
do
    if python3 -B "$W/gate.py" recheck "$W/$a"
    then
        of "cd '$W/$a.recheck' && simpleFoam > log 2>&1" < /dev/null
    fi
done
python3 -B "$W/gate.py" judge "$W" || fail=1
[ "$fail" = 0 ] && echo "PASS: both sweep orders reach OpenFOAM's fixed point" || { echo FAIL; exit 1; }
