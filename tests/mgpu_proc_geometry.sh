#!/bin/bash
# Processor geometry against OpenFOAM v2606 -parallel, and the loopback harness, on real decomposePar meshes.
#   tests/mgpu_proc_geometry.sh <brae_interFoam> <test_proc_geometry_loopback> [cases] [nps] [methods]
# For every (case, np, method): decomposePar; OpenFOAM's weights/delta/deltaCoeffs/nonOrthDeltaCoeffs/
# nonOrthCorrectionVectors/Sf/magSf of the processor faces (tools/dumpInterFoam, BRAE_PROC_GEOMETRY) against brae's
# (BRAE_MGPU_GEOMETRY_ONLY=1, one MPI process per rank), face by face, ULP-gated (compare_proc_geometry.py); then the
# loopback harness (all ranks as threads of one process) on the same directory.
# Skips (77) without OpenFOAM v2606 or the oracle build ($ORACLE_ROOT/bin/dumpInterFoam, tools/mgpu_oracle/setup.sh).
set -eo pipefail
BRAE=${1:?usage: $0 <brae_interFoam> <test_proc_geometry_loopback> [cases] [nps] [methods]}
LOOP=${2:?loopback test binary}
CASES=${3:-"tank tankGraded open"}
NPS=${4:-"2 4 8"}
METHODS=${5:-"scotch simple hier"}
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
export ORACLE_ROOT=${ORACLE_ROOT:-/tmp/brae-oracle}
[ -f /usr/lib/openfoam/openfoam2606/etc/bashrc ] || { echo "SKIP: no OpenFOAM v2606"; exit 77; }
[ -x "$ORACLE_ROOT/bin/dumpInterFoam" ] || { echo "SKIP: no $ORACLE_ROOT/bin/dumpInterFoam (tools/mgpu_oracle/setup.sh)"; exit 77; }
grep -q procGeometry "$ORACLE_ROOT/dumpInterFoam/interFoam.C" 2>/dev/null \
    || { echo "SKIP: the oracle build at $ORACLE_ROOT predates procGeometryDump.H (rerun setup.sh)"; exit 77; }
export OUT_ROOT=${OUT_ROOT:-$(mktemp -d /tmp/mgpu_proc_geometry.XXXXXX)}
mkdir -p "$OUT_ROOT"
export KEEP=1
trap '[ -n "$KEEP_OUT" ] || rm -rf "$OUT_ROOT"' EXIT
for CASE in $CASES; do
    for NP in $NPS; do
        for METHOD in $METHODS; do
            D=$OUT_ROOT/${CASE}_np${NP}_${METHOD}
            DICT=$HERE/tools/mgpu_oracle/cases/$CASE/system/decomposeParDict.$METHOD$NP
            [ -f "$DICT" ] || { echo "-- $CASE np $NP $METHOD: no decomposeParDict, skipped"; continue; }
            if ! "$HERE/tools/mgpu_oracle/proc_geometry_gate.sh" "$BRAE" "$CASE" "$NP" "$METHOD" > "$OUT_ROOT/log.gate" 2>&1; then
                cat "$OUT_ROOT/log.gate"; KEEP_OUT=1; echo "FAIL: kept $D"; exit 1
            fi
            grep -a "^==" "$OUT_ROOT/log.gate"
            printf '== loopback %-10s np %d %-7s  ' "$CASE" "$NP" "$METHOD"
            "$LOOP" "$D" > "$D/log.loopback" 2>&1 || { tail -20 "$D/log.loopback"; echo "FAIL: kept $D"; KEEP_OUT=1; exit 1; }
            tail -n 1 "$D/log.loopback"
            rm -rf "$D"   # one decomposition at a time: /tmp is RAM
        done
    done
done
echo "mgpu_proc_geometry: PASS"
