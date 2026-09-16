#!/usr/bin/env bash
# brae's DEVICE alpha step against brae's HOST solver, on damBreak's own mesh and case.
#
# WHY THE CASE AND NOT A FIXTURE. Every device gate before this ran on a synthetic box: a rotating blob
# on uniform cells, built so each arm could discriminate. Those cannot bring what a real case brings --
# damBreak's mesh is graded, its atmosphere patch is inletOutlet, its frontAndBack are EMPTY (which
# MULES must skip and no box fixture here had), and its fvSolution asks for MULESCorr with
# nLimiterIter 5 and nAlphaCorr 2. This gate runs those.
#
# THE CHAIN. tests/interfoam_dambreak_vs_openfoam.sh holds the HOST solver against real OpenFOAM at
# alpha 3.4346e-09; this holds the DEVICE against that host. Two links, each measured. Comparing the
# device straight to OpenFOAM would fold both differences into one number and make neither readable.
#
# No OpenFOAM RUN is needed here -- only its mesh generator, because damBreak ships 0.orig and no mesh.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_device_inter_dambreak_alpha"
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

cp -r "$W/case/0.orig" "$W/case/0"
( cd "$W/case" && blockMesh > log.blockMesh 2>&1 ) || { echo "SKIP: blockMesh failed"; exit 77; }
if command -v setFields > /dev/null 2>&1; then
    ( cd "$W/case" && setFields > log.setFields 2>&1 ) || { echo "SKIP: setFields failed"; exit 77; }
fi

"$BIN" "$W/case" "$W/case/0" "$STEPS" "$DT"
