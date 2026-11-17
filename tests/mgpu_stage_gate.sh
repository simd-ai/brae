#!/bin/bash
# The explicit stages of the first step (nHatf, sigma*K, the surface-tension force, rho and snGrad(rho), phic, phir,
# alphaPhiUn) through brae's processor-face operators at np = 1, 2, 4, 8 against OpenFOAM v2606 -parallel's own
# (tools/mgpu_oracle/stage_gate.sh: both read the same step-1 state; maxrel <= 1e-11 per quantity), plus the
# control BRAE_CONTROL_PROC_NO_EXCHANGE=1 (halos never refreshed), which must FAIL the gate.
#   tests/mgpu_stage_gate.sh <brae_interFoam> [cases] [specs]     spec = np[:method], default "1 2 2:simple 4 4:hier 8 8:simple 8:hier"
# Skips (77) without OpenFOAM v2606 or an oracle build that has stageDump.H (tools/mgpu_oracle/setup.sh).
set -eo pipefail
BRAE=${1:?usage: $0 <brae_interFoam> [cases] [specs]}
CASES=${2:-"tank tankGraded open"}
SPECS=${3:-"1 2 2:simple 4 4:hier 8 8:simple 8:hier"}
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
export ORACLE_ROOT=${ORACLE_ROOT:-/tmp/brae-oracle}
[ -f /usr/lib/openfoam/openfoam2606/etc/bashrc ] || { echo "SKIP: no OpenFOAM v2606"; exit 77; }
[ -x "$ORACLE_ROOT/bin/dumpInterFoam" ] || { echo "SKIP: no $ORACLE_ROOT/bin/dumpInterFoam (tools/mgpu_oracle/setup.sh)"; exit 77; }
grep -q stageDump "$ORACLE_ROOT/dumpInterFoam/interFoam.C" 2>/dev/null \
    || { echo "SKIP: the oracle build at $ORACLE_ROOT predates stageDump.H (rerun setup.sh)"; exit 77; }
export OUT_ROOT=${OUT_ROOT:-$(mktemp -d /tmp/mgpu_stage.XXXXXX)}
mkdir -p "$OUT_ROOT"
trap '[ -n "$KEEP_OUT" ] || rm -rf "$OUT_ROOT"' EXIT
for CASE in $CASES; do
    for S in $SPECS; do
        IFS=: read NP METHOD <<< "$S"; METHOD=${METHOD:-scotch}
        if ! "$HERE/tools/mgpu_oracle/stage_gate.sh" "$BRAE" "$CASE" "$NP" "$METHOD" > "$OUT_ROOT/log.gate" 2>&1; then
            cat "$OUT_ROOT/log.gate"; KEEP_OUT=1; echo "FAIL: $CASE np $NP $METHOD (kept $OUT_ROOT)"; exit 1
        fi
        worst=$(grep "^| 0" "$OUT_ROOT/log.gate" | awk -F'|' '{print $8}' | sort -g | tail -n 1)
        printf '== %-10s np %d %-7s worst maxrel %s\n' "$CASE" "$NP" "$METHOD" "$worst"
        rm -rf "$OUT_ROOT"/${CASE}_np*
    done
done
# the control: the same gate with the halos never refreshed must fail
if BRAE_CONTROL_PROC_NO_EXCHANGE=1 "$HERE/tools/mgpu_oracle/stage_gate.sh" "$BRAE" tank 2 scotch > "$OUT_ROOT/log.ctl" 2>&1; then
    cat "$OUT_ROOT/log.ctl"; echo "FAIL: the control (BRAE_CONTROL_PROC_NO_EXCHANGE=1) passed the gate"; exit 1
fi
echo "== control BRAE_CONTROL_PROC_NO_EXCHANGE=1 fails the gate, as it must"
if [ "${BRAE_STAGE_ALPHA:-0}" = 1 ]; then
    if BRAE_STAGE_ALPHA=1 BRAE_CONTROL_PROC_NO_LAMBDA_SYNC=1 "$HERE/tools/mgpu_oracle/stage_gate.sh" "$BRAE" tank 2 scotch > "$OUT_ROOT/log.lambda_ctl" 2>&1; then
        cat "$OUT_ROOT/log.lambda_ctl"; echo "FAIL: BRAE_CONTROL_PROC_NO_LAMBDA_SYNC=1 passed the alpha gate"; exit 1
    fi
    echo "== control BRAE_CONTROL_PROC_NO_LAMBDA_SYNC=1 fails the alpha gate, as it must"
fi
echo PASS
