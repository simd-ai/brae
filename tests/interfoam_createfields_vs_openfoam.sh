#!/usr/bin/env bash
# interFoam's createFields against REAL OpenFOAM's own damBreak, prepared the way the tutorial does.
#
# damBreak ships 0.orig and no mesh: blockMesh builds it and setFields paints the initial column of
# water. So the fixture is not a directory this repo can check in -- it is a PROCEDURE, and running
# OpenFOAM's own two utilities is the only way to get the case brae is supposed to read. That is why
# this is a shell gate rather than a plain ctest binary, and why it SKIPS rather than fails where
# OpenFOAM is not installed.
#
# What is compared: brae's InterFields against the numbers damBreak's own dictionaries and 0 directory
# carry -- the phase order and properties, cAlpha through its regex key, nAlphaSubCycles, MULESCorr,
# nLimiterIter, maxCo/maxAlphaCo, g, the alpha field's NAME, and the derived rho/mu/gh/p. The arms live
# in tests/test_inter_case_cpp.cu; this script only builds the case for them.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_case_cpp"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"

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
command -v setFields > /dev/null 2>&1 || { echo "SKIP: setFields not on PATH"; exit 77; }

cp -r "$W/case/0.orig" "$W/case/0"
( cd "$W/case" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -20 "$W/case/log.blockMesh"; exit 1; }
( cd "$W/case" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; tail -20 "$W/case/log.setFields"; exit 1; }

echo "damBreak prepared: $(grep -c . "$W/case/constant/polyMesh/owner" > /dev/null 2>&1; echo ok)"
"$BIN" "$W/case" "$W/case/0"
