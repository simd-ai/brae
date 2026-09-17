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
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_dambreak_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
STEPS=${STEPS:-5}
DT=${DT:-1e-4}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

cp -r "$W/case/0.orig" "$W/case/0"

# A FIXED time step, and write exactly once at step N. writePrecision 15 because the comparison is
# against brae's fp64 and an ascii round-trip at the default 6 digits would dominate the difference.
STEPS="$STEPS" DT="$DT" python3 - "$W/case" <<'PYEOF'
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

( cd "$W/case" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -20 "$W/case/log.blockMesh"; exit 1; }
( cd "$W/case" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; tail -20 "$W/case/log.setFields"; exit 1; }
( cd "$W/case" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam"; tail -30 "$W/case/log.interFoam"; exit 1; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")
[ -d "$W/case/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory"; ls "$W/case"; exit 1; }
echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END"

"$BIN" "$W/case" "$W/case/0" "$W/case/$END" "$STEPS" "$W/case/log.interFoam"
