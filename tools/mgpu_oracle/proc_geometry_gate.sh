#!/bin/bash
# The processor-patch geometry gate: OpenFOAM's weights / delta / deltaCoeffs / nonOrthDeltaCoeffs /
# nonOrthCorrectionVectors / Sf / magSf of every processor face against brae's, per rank, for the oracle cases on
# real decomposePar meshes.
#
#   tools/mgpu_oracle/proc_geometry_gate.sh <brae_interFoam> [cases] [nps] [methods]
#       cases    default "tank tankGraded open"      (tools/mgpu_oracle/cases/)
#       nps      default "2 4 8"
#       methods  default "scotch simple hier"        (system/decomposeParDict.<method><np>)
#   env  ORACLE_ROOT  directory with of_env.sh and bin/dumpInterFoam built from tools/dumpInterFoam WITH procGeometryDump.H
#                     (setup.sh builds it; default /tmp/brae-oracle)
#        OUT_ROOT     default /tmp/brae-oracle_geometry (scratch; the run directories are removed after
#                     a passing combination unless KEEP=1)
#   one process per rank (mpirun --oversubscribe) on the one GPU; the brae side is
#   BRAE_MGPU_GEOMETRY_ONLY=1 brae_interFoam -parallel, which couples the patches over ProcComm and stops.
set -eo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
BRAE=$(readlink -f "${1:?usage: proc_geometry_gate.sh <brae_interFoam> [cases] [nps] [methods]}")
CASES=${2:-"tank tankGraded open"}
NPS=${3:-"2 4 8"}
METHODS=${4:-"scotch simple hier"}
ORACLE_ROOT=${ORACLE_ROOT:-/tmp/brae-oracle}
OUT_ROOT=${OUT_ROOT:-/tmp/brae-oracle_geometry}
[ -x "$BRAE" ] || { echo "no brae binary at $BRAE" >&2; exit 1; }
[ -x "$ORACLE_ROOT/bin/dumpInterFoam" ] || { echo "SKIP: no $ORACLE_ROOT/bin/dumpInterFoam (tools/mgpu_oracle/setup.sh builds the OpenFOAM oracle)"; exit 77; }
source "$ORACLE_ROOT/of_env.sh"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_btl_vader_single_copy_mechanism=none
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$OUT_ROOT"
fail=0
for CASE in $CASES; do
    MESH=$OUT_ROOT/mesh_$CASE
    if [ ! -d "$MESH/constant/polyMesh" ]; then
        rm -rf "$MESH"; cp -r "$HERE/cases/$CASE" "$MESH"
        ( cd "$MESH"; rm -f system/decomposeParDict.*
          blockMesh > log.blockMesh 2>&1
          cp -r 0.orig 0; mv 0/alpha.water.orig 0/alpha.water; rm -rf 0.orig )
    fi
    for NP in $NPS; do
        for METHOD in $METHODS; do
            D=$OUT_ROOT/${CASE}_np${NP}_${METHOD}
            DICT=$HERE/cases/$CASE/system/decomposeParDict.$METHOD$NP
            [ -f "$DICT" ] || { echo "-- $CASE np $NP $METHOD: no decomposeParDict.$METHOD$NP, skipped"; continue; }
            rm -rf "$D"; cp -r "$MESH" "$D"; cp "$DICT" "$D/system/decomposeParDict"
            ( cd "$D" && decomposePar -force > log.decompose 2>&1 )
            ( cd "$D" && BRAE_PROC_GEOMETRY=1 mpirun --bind-to none -np "$NP" "$ORACLE_ROOT/bin/dumpInterFoam" -parallel > log.of 2>&1 )
            ( cd "$D" && BRAE_MGPU_GEOMETRY_ONLY=1 mpirun --bind-to none -np "$NP" "$BRAE" -case "$D" -parallel > log.brae 2>&1 ) \
                || { echo "brae_interFoam failed: $D/log.brae"; grep -a -v "^    " "$D/log.brae" | tail -25; fail=1; continue; }
            printf '== %-10s np %d %-7s  ' "$CASE" "$NP" "$METHOD"
            if python3 "$HERE/compare_proc_geometry.py" "$D" -q; then
                [ "${KEEP:-0}" = 1 ] || rm -rf "$D"
            else
                fail=1
            fi
        done
    done
done
[ $fail = 0 ] && echo "PROC GEOMETRY: ALL MATCH" || { echo "PROC GEOMETRY: FAILURES (run directories kept)"; exit 1; }
