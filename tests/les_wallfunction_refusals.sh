#!/bin/bash
# Audit finding #14: the algebraic-LES path honoured ONLY nutUSpaldingWallFunction while printing any
# other nut wall function as honoured, labelled with a model ("kEpsilon") that was not running -- an
# LES case with a k-based wall function ran with no wall model at all and the log said otherwise.
#
# Now: any nut wall function other than nutUSpalding on an algebraic LES model REFUSES BY NAME (the
# k-based family cannot be ported there either way -- algebraic LES has no k field, and OpenFOAM feeds
# those functions the model's own sgs k() estimate), and every model print carries the model word the
# case names (ctl.modelName) instead of a label derived from two flags.
#
# The fixture is backwardFacingStep2D mutated to LES/Smagorinsky through the transient dispatch
# (the steady driver refuses simulationType LES outright, which arm 4 pins). Its shipped 0/nut walls
# carry nutUBlendedWallFunction, which is exactly the function the refusal must name.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/backwardFacingStep2D"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$W/"

fail=0
say() { printf '  %-64s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# ---- arm 3 FIRST, on the unmutated RAS case: the model label survives the modelName change ---------
sed -i 's/^endTime.*/endTime 1;/;s/writeInterval.*/writeInterval 1;/' "$W/system/controlDict"
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "honoured on kOmegaSST per the BC" \
    && say "RAS still labels its own model (kOmegaSST) in the wall-fn print" ok \
    || { echo "$out" | grep "nut wall function" | head -2; \
         say "RAS still labels its own model (kOmegaSST) in the wall-fn print" FAIL; }

# ---- the LES mutation: Smagorinsky through the transient dispatch ----------------------------------
cat > "$W/constant/turbulenceProperties" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType      LES;
LES
{
    LESModel        Smagorinsky;
    turbulence      on;
    delta           cubeRootVol;
}
EOF
sed -i 's/^application.*/application     pimpleFoam;/' "$W/system/controlDict"
sed -i 's/^endTime.*/endTime 1e-05;/;s/^deltaT.*/deltaT 1e-05;/' "$W/system/controlDict"
sed -i 's/default *steadyState;/default         Euler;/' "$W/system/fvSchemes"
grep -q "Smagorinsky" "$W/constant/turbulenceProperties" || { echo "FAIL: LES mutation did not apply"; exit 1; }

# ---- arm 1: the shipped nutUBlendedWallFunction must refuse, naming the function AND the model -----
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "asks for nutUBlendedWallFunction on LESModel Smagorinsky" \
    && echo "$out" | grep -q "honours only" \
    && say "LES + nutUBlendedWallFunction refused, naming function and model" ok \
    || { echo "$out" | tail -4; say "LES + nutUBlendedWallFunction refused, naming function and model" FAIL; }
# ...and the refused run must NOT have printed the function as honoured (the pre-fix defect)
echo "$out" | grep -q "nutUBlendedWallFunction (velocity-based, honoured" \
    && say "the refused run does not print the function as honoured" FAIL \
    || say "the refused run does not print the function as honoured" ok

# ---- arm 2 (negative control): LES + nutUSpalding runs, with the honest model label ----------------
sed -i 's/nutUBlendedWallFunction/nutUSpaldingWallFunction/' "$W/0/nut"
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "^Time = " \
    && echo "$out" | grep -q "honoured on Smagorinsky per the BC" \
    && ! echo "$out" | grep -q "honoured on kEpsilon" \
    && say "LES + nutUSpalding runs and the label names Smagorinsky, not kEpsilon" ok \
    || { echo "$out" | tail -4; say "LES + nutUSpalding runs and the label names Smagorinsky, not kEpsilon" FAIL; }

# ---- arm 4: the steady dispatch still refuses simulationType LES outright --------------------------
sed -i 's/^application.*/application     simpleFoam;/' "$W/system/controlDict"
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "unsupported simulationType 'LES'" \
    && say "the steady dispatch refuses simulationType LES by name" ok \
    || { echo "$out" | tail -3; say "the steady dispatch refuses simulationType LES by name" FAIL; }

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
