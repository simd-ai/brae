#!/usr/bin/env bash
# brae's interFoam against REAL OpenFOAM's interFoam, on damBreak, field by field.
#
# THE METHOD IS "EXACTLY N IDENTICAL STEPS", not "run both to the end". An adaptive time step makes the
# two solvers take DIFFERENT steps the moment their Courant numbers differ by anything at all, and then
# every field is compared at a different physical time -- which produces a disagreement that looks like
# a discretisation error and is actually a clock. So the case is rewritten with `adjustTimeStep no` and
# a fixed deltaT, both run the same count, and the comparison is at one instant.
#
# The case is damBreak's own, prepared the way the tutorial does (blockMesh, then setFields), because
# it ships 0.orig and no mesh.
#
# IT RUNS AT TWO TIME STEPS, because the two things it measures want opposite fixtures. At dt 1e-4 the
# interface barely moves (3.7e-03) and the fields agree to 3.6e-14, which is where the tight FIELD
# bounds live -- but the alpha pre-solve is then so diagonally dominant (Co ~ 1e-3) that every solver
# lands on the exact solution in one iteration: measured, PBiCGStab's final residuals were 2.6e-03 from
# OpenFOAM's symGaussSeidel's, beside the device's honest 1.5e-03, so a solver-log arm there cannot
# tell solvers apart. At dt 5e-3 the interface moves 0.69, OpenFOAM's smoother takes 0, 5, 2, 2, 2
# sweeps, brae takes the same and leaves its final residuals to 9e-10 -- and the PBiCGStab CONTROL gets
# two counts of five and is 100% out. The `bigstep` profile is where those arms are asserted.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_dambreak_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
STEPS=${STEPS:-5}
DT=${DT:-1e-4}
DT_BIG=${DT_BIG:-5e-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

# run_at <deltaT> <profile>: stage the tutorial at a FIXED step, run real OpenFOAM, run the gate.
run_at()
{
    local dt="$1" profile="$2"
    local C="$W/$profile"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"

    # A FIXED time step, and write exactly once at step N. writePrecision 15 because the comparison is
    # against brae's fp64 and an ascii round-trip at the default 6 digits would dominate the difference.
    STEPS="$STEPS" DT="$dt" python3 - "$C" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
n  = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'^adjustTimeStep .*', 'adjustTimeStep  no;',        s, flags=re.M)
s = re.sub(r'^deltaT .*',         'deltaT          %s;' % dt,   s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         %.10g;' % (n*float(dt)), s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;',  s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   %d;' % n,    s, flags=re.M)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',     s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',        s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; return 1; }

    local end
    end=$(python3 -c "print('%.10g' % ($STEPS*float('$dt')))")
    [ -d "$C/$end" ] || { echo "FAIL: OpenFOAM wrote no $end directory"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $dt to t = $end   [$profile]"

    # THE CONTROL CASE for the alpha solve-log arm: identical, except that the alpha entry names
    # PBiCGStab -- the solver brae's host ran in place of the case's symGaussSeidel until it had one.
    # OpenFOAM is NOT re-run on it; the test compares brae-on-PBiCGStab against
    # OpenFOAM-on-symGaussSeidel and, under `bigstep`, requires that comparison to FAIL.
    cp -r "$C" "$C.control"
    python3 - "$C.control" <<'PYEOF'
import os, re, sys
q = os.path.join(sys.argv[1], 'system/fvSolution')
t = open(q).read()
m = re.search(r'("alpha\.water\.\*"\s*\{)([^}]*)\}', t)
assert m, 'no alpha.water.* entry'
body = m.group(2)
body = re.sub(r'solver\s+\w+;', 'solver          PBiCGStab;', body)
body = re.sub(r'smoother\s+\w+;', 'preconditioner  DILU;', body)
t = t[:m.start(2)] + body + t[m.end(2):]
open(q, 'w').write(t)
PYEOF
    grep -q "PBiCGStab" "$C.control/system/fvSolution" || { echo "FAIL: the control case was not rewritten"; return 1; }

    "$BIN" "$C" "$C/0" "$C/$end" "$STEPS" "$C/log.interFoam" "$C.control" "$profile"
}

rc=0
run_at "$DT" small || rc=1
run_at "$DT_BIG" bigstep || rc=1
exit $rc
