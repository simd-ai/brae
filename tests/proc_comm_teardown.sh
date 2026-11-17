#!/bin/bash
# The np=8 explicit-stage run drives enough host collectives and pinned staging to make the
# abort side channel's request-lifetime bug visible at process exit. Before restoring the
# MPI_Test completion guard this aborts in glibc/PMIx AFTER every rank prints End.
# Optional BRAE_MGPU_TEARDOWN_CASE is a prepared <case>/bd tree. Otherwise the normal oracle
# harness creates that tree once, then this script repeats only the relevant brae launch.
set -euo pipefail
BRAE=${1:?usage: $0 <brae_interFoam>}
LAUNCHES=${LAUNCHES:-20}
MPIEXEC=${MPIEXEC:-mpirun}
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
OUT=$(mktemp -d /tmp/proc_comm_teardown.XXXXXX)
trap '[ -n "${KEEP_OUT:-}" ] || rm -rf "$OUT"' EXIT
export MALLOC_CHECK_=3
export OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_btl_vader_single_copy_mechanism=none OMP_NUM_THREADS=1
if [ -n "${BRAE_MGPU_TEARDOWN_CASE:-}" ]; then
    CASE=$BRAE_MGPU_TEARDOWN_CASE
else
    [ -f /usr/lib/openfoam/openfoam2606/etc/bashrc ] || { echo "SKIP: no OpenFOAM v2606"; exit 77; }
    [ -x "${ORACLE_ROOT:-/tmp/brae-oracle}/bin/dumpInterFoam" ] || {
        echo "SKIP: no stage-dump oracle (set ORACLE_ROOT or BRAE_MGPU_TEARDOWN_CASE)"; exit 77; }
    OUT_ROOT=$OUT/setup KEEP=1 "$HERE/tools/mgpu_oracle/stage_gate.sh" "$BRAE" tank 8 scotch > "$OUT/setup.log" 2>&1 || {
        cat "$OUT/setup.log"; exit 1; }
    CASE=$OUT/setup/tank_np8
fi
[ -d "$CASE/bd" ] || { echo "FAIL: no prepared bd directory in $CASE" >&2; exit 1; }
cd "$CASE"
for i in $(seq 1 "$LAUNCHES"); do
    export MALLOC_PERTURB_=$((40 + i))
    log=$OUT/$(printf '%02d' "$i").log
    set +e
    BRAE_MGPU_STAGE_DUMP=$CASE/bd timeout 90 "$MPIEXEC" --bind-to none -np 8 \
        -x MALLOC_CHECK_ -x MALLOC_PERTURB_ "$BRAE" -case bd -parallel -device >"$log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        cat "$log"
        echo "FAIL: np=8 explicit-stage teardown launch $i/$LAUNCHES exited $rc; log: $log" >&2
        KEEP_OUT=1
        exit 1
    fi
    rm -f "$log"
done
echo "PASS: $LAUNCHES consecutive np=8 explicit-stage teardown launches"
