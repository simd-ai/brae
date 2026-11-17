#!/bin/bash
# Post-alpha-step gate: alpha, alphaPhi10 and rho after alphaEqnSubCycle + mixture.correct() against
# OpenFOAM v2606 -parallel. The same cases exercise MULESCorr (default), explicit MULES, and two sub-cycles.
#   tests/mgpu_alpha_stage_gate.sh <brae_interFoam> [cases] [specs]
# spec = np[:method], default "1 2 2:simple 4 4:simple 4:hier 8 8:simple 8:hier".
# Skips (77) without OpenFOAM v2606 or the stage-dump oracle build.
set -eo pipefail
BRAE=${1:?usage: $0 <brae_interFoam> [cases] [specs]}
CASES=${2:-"tank tankGraded open"}
SPECS=${3:-"1 2 2:simple 4 4:simple 4:hier 8 8:simple 8:hier"}
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
export ORACLE_ROOT=${ORACLE_ROOT:-/tmp/brae-oracle}
[ -f /usr/lib/openfoam/openfoam2606/etc/bashrc ] || { echo "SKIP: no OpenFOAM v2606"; exit 77; }
[ -x "$ORACLE_ROOT/bin/dumpInterFoam" ] || { echo "SKIP: no $ORACLE_ROOT/bin/dumpInterFoam (tools/mgpu_oracle/setup.sh)"; exit 77; }
export OUT_ROOT=${OUT_ROOT:-$(mktemp -d /tmp/mgpu_alpha_stage.XXXXXX)}
mkdir -p "$OUT_ROOT"
trap '[ -n "$KEEP_OUT" ] || rm -rf "$OUT_ROOT"' EXIT
run_gate() {
    local tag=$1 case=$2 np=$3 method=$4
    if ! BRAE_STAGE_ALPHA=1 "$HERE/tools/mgpu_oracle/stage_gate.sh" "$BRAE" "$case" "$np" "$method" > "$OUT_ROOT/log.$tag" 2>&1; then
        cat "$OUT_ROOT/log.$tag"; KEEP_OUT=1; echo "FAIL: $tag (kept $OUT_ROOT)"; exit 1
    fi
    grep '^| 0' "$OUT_ROOT/log.$tag" | grep -E 'sAlphaAfter|sAlphaPhi\.dump|sRhoAfter' || {
        echo "FAIL: $tag did not compare the post-alpha fields"; exit 1; }
    for F in sAlphaAfter.dump sAlphaPhi.dump sRhoAfter.dump; do
        grep -F "$F" "$OUT_ROOT/log.$tag" >/dev/null || { echo "FAIL: $tag omitted $F"; exit 1; }
    done
    printf '== %-24s PASS\n' "$tag"
}
for CASE in $CASES; do
    for S in $SPECS; do
        IFS=: read NP METHOD <<< "$S"; METHOD=${METHOD:-scotch}
        run_gate "${CASE}_np${NP}_${METHOD}" "$CASE" "$NP" "$METHOD"
        rm -rf "$OUT_ROOT"/${CASE}_np*
    done
done
# Explicit MULES and nAlphaSubCycles 2 use the same harness, but with a temporary case copy made by
# stage_gate's caller through its case directory. Keep the production case untouched.
WORK=$OUT_ROOT/variants; mkdir -p "$WORK"
cp -r "$HERE/tools/mgpu_oracle/cases/tank" "$WORK/tankExplicit"
sed -i 's/MULESCorr       yes/MULESCorr       no /' "$WORK/tankExplicit/system/fvSolution"
cp -r "$HERE/tools/mgpu_oracle/cases/tank" "$WORK/tankSubcycles"
sed -i 's/nAlphaSubCycles 1/nAlphaSubCycles 2/' "$WORK/tankSubcycles/system/fvSolution"
for V in tankExplicit tankSubcycles; do
    for NP in 1 2; do run_gate "${V}_np${NP}" "$WORK/$V" "$NP" scotch; rm -rf "$OUT_ROOT"/${V}_np*; done
done
# Negative control: without the processor lambda min sync, the two sides of a cut face are limited
# independently and the post-alpha gate must fail.
if BRAE_STAGE_ALPHA=1 BRAE_CONTROL_PROC_NO_LAMBDA_SYNC=1 "$HERE/tools/mgpu_oracle/stage_gate.sh" "$BRAE" tank 2 scotch > "$OUT_ROOT/log.lambda_control" 2>&1; then
    cat "$OUT_ROOT/log.lambda_control"; echo "FAIL: BRAE_CONTROL_PROC_NO_LAMBDA_SYNC=1 passed"; exit 1
fi
echo "== control BRAE_CONTROL_PROC_NO_LAMBDA_SYNC=1 fails the gate, as it must"
echo PASS
