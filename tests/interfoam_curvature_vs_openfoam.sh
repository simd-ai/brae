#!/usr/bin/env bash
# brae's interface curvature against OpenFOAM's own, at several pass counts.
#
# calculateK is a FIXED POINT: it reads the wall gradient its own previous pass wrote through
# correctContactAngle, so the curvature at a contact line depends on how many times it has run. This
# sweeps the count on BOTH sides and requires agreement at each -- which is a much stronger statement
# than agreement at one, and it is what separates "the formula is right" from "the solver happens to
# call it the same number of times".
#
# The oracle is tools/dumpInterfaceK, which runs OpenFOAM's UNMODIFIED interfaceProperties.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_curvature_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/capillaryRise"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: capillaryRise not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/log.*

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

command -v blockMesh       > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v dumpInterfaceK  > /dev/null 2>&1 || {
    echo "SKIP: dumpInterfaceK not built -- run 'wmake' in tools/dumpInterfaceK"; exit 77; }

# writePrecision 15: the dump is ascii, and at the default 6 digits the comparison measures the WRITE
# and not the code -- K is of order 4e+04, so six significant figures is an absolute precision of 0.01
# and the first run of this gate sat at exactly that.
python3 - "$W/case" <<'PYEOF'
import re, sys
c = sys.argv[1] + '/system/controlDict'
s = open(c).read()
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;', s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',    s, flags=re.M)
open(c, 'w').write(s)
PYEOF

cp -r "$W/case/0.orig" "$W/case/0"
( cd "$W/case" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) \
    || { echo "FAIL: mesh/setFields"; exit 1; }

rc=0
for N in 1 2 3 4; do
    ( cd "$W/case" && BRAE_K_PASSES=$((N-1)) dumpInterfaceK > log.dumpK 2>&1 ) \
        || { echo "FAIL: dumpInterfaceK"; tail -20 "$W/case/log.dumpK"; exit 1; }
    "$BIN" "$W/case" "$W/case/0" "$N" || rc=1
done
exit $rc
