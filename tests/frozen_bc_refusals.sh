#!/bin/bash
# The frozen-per-step-BC refusals are REACHABLE from the drivers that need them.
#
# The unit gate (test_frozen_bc_guard) proves what the guard refuses; this one proves the drivers call
# it. A fixedMean U inlet on backwardFacingStep2D must be refused by BOTH incompressible paths in the
# `brae` binary -- simpleFoamV2 (BRAE_SIMPLEFOAM_V2=1, no per-step hooks at all) and the
# DeviceSimpleSolver path (which wires the coded pair but not fixedMean) -- and the UNMUTATED case must
# still start, or the guard is refusing ordinary cases.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/backwardFacingStep2D"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$W/"
# one iteration: the control only needs to prove the case STARTS (createFields is behind us by then)
sed -i 's/^endTime.*/endTime 1;/' "$W/system/controlDict"
sed -i 's/writeInterval.*/writeInterval 1;/' "$W/system/controlDict"

fail=0
say() { printf '  %-64s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# the mutation: the inlet asks for a maintained mean
sed -i 's/type            fixedValue;/type            fixedMean;\n        meanValue       uniform (44.2 0 0);/' "$W/0/U"
grep -q fixedMean "$W/0/U" || { echo "FAIL: mutation did not apply"; exit 1; }

out=$(cd "$W" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "fixedMean" && echo "$out" | grep -q "never updates" \
    && say "simpleFoamV2 refuses a fixedMean inlet by name" ok \
    || say "simpleFoamV2 refuses a fixedMean inlet by name" FAIL

out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "fixedMean" && echo "$out" | grep -q "never updates" \
    && say "gpuSimpleFoam refuses a fixedMean inlet by name" ok \
    || say "gpuSimpleFoam refuses a fixedMean inlet by name" FAIL

# NEGATIVE CONTROL: the case's own fixedValue inlet must still run (one iteration).
cp "$SRC/0/U" "$W/0/U"
out=$(cd "$W" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case "$W" 2>&1) \
    && say "the unmutated case still starts under V2 (negative control)" ok \
    || { echo "$out" | tail -5; say "the unmutated case still starts under V2 (negative control)" FAIL; }

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
