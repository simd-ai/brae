#!/bin/bash
# Audit finding #15: the SA device path hard-forced nutUSpaldingWallFunction (`ctl_.sa || ...` in
# device_simple_foam.cu) whatever 0/nut asked for. bump2D:SpalartAllmaras ships nutLowReWallFunction,
# whose calcNut() returns Zero UNCONDITIONALLY on every model
# (nutLowReWallFunctionFvPatchScalarField.C:38-42) -- and got a Newton uTau instead, inside a green
# tutorial gate whose loose nut bound (1e-01) absorbed it.
#
# Now the BC selects: Spalding and LowRe are honoured on SA; the k-based family refuses by name
# (OpenFOAM feeds it SpalartAllmarasBase::k(), a derived estimate brae does not carry); a concrete wall
# patch whose nut names no wall function refuses (the device would overwrite the BC); and two different
# wall functions refuse (one selector drives every wall).
#
# The fixture is validation/airFoil2D -- the SA gate case, walls = nutUSpaldingWallFunction.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/airFoil2D"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary"; exit 77; }
[ -d "$SRC/constant" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$W/"
sed -i 's/^endTime.*/endTime         1;/;s/^writeInterval.*/writeInterval   1;/' "$W/system/controlDict"

fail=0
say() { printf '  %-66s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# ---- arm 1 (negative control): the shipped Spalding case runs, honestly labelled -------------------
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "nutUSpaldingWallFunction" \
    && echo "$out" | grep -q "honoured on SpalartAllmaras per the BC" \
    && echo "$out" | grep -qE "^Time = |iter" \
    && say "SA + nutUSpalding runs, labelled SpalartAllmaras (control)" ok \
    || { echo "$out" | tail -4; say "SA + nutUSpalding runs, labelled SpalartAllmaras (control)" FAIL; }

# ---- arm 2: a k-based wall function on SA refuses, naming function, model and the OF mechanism -----
sed -i 's/nutUSpaldingWallFunction/nutkWallFunction/' "$W/0/nut"
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "asks for nutkWallFunction on SpalartAllmaras" \
    && echo "$out" | grep -q "derived k() estimate" \
    && say "SA + nutkWallFunction refused by name" ok \
    || { echo "$out" | tail -4; say "SA + nutkWallFunction refused by name" FAIL; }

# ---- arm 3: nutLowRe on SA is HONOURED (the bump2D BC), not refused and not Spalding ---------------
sed -i 's/nutkWallFunction/nutLowReWallFunction/' "$W/0/nut"
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "nut wall function: nutLowReWallFunction (honoured on SpalartAllmaras per the BC)" \
    && echo "$out" | grep -qE "^Time = |iter" \
    && say "SA + nutLowRe runs and the label names LowRe, not Spalding" ok \
    || { echo "$out" | tail -4; say "SA + nutLowRe runs and the label names LowRe, not Spalding" FAIL; }

# ---- arm 4: two DIFFERENT wall functions refuse (one selector drives every wall) -------------------
# airFoil2D has a single wall patch, so the second member of the family rides a dict entry with no
# matching mesh patch: the reader keeps it, patchGeoType resolves it to nothing (so the wall-typed
# guards skip it), and the one-selector guard is exactly the check that still has to see it. A wall
# function on a real non-wall patch cannot stage this -- guardWallFn refuses that first, by design.
cp "$SRC/0/nut" "$W/0/nut"    # walls back to Spalding, so the phantom LowRe entry disagrees
python3 - "$W/0/nut" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
entry = "    phantomWall\n    {\n        type            nutLowReWallFunction;\n        value           uniform 0;\n    }\n"
i = s.rstrip().rfind('}')          # the boundaryField closer is the last brace in the file
open(p, 'w').write(s[:i] + entry + s[i:])
PYEOF
grep -q "phantomWall" "$W/0/nut" || { echo "FAIL: arm-4 mutation did not apply"; exit 1; }
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "carries both" \
    && say "two different SA wall functions refused" ok \
    || { echo "$out" | tail -4; say "two different SA wall functions refused" FAIL; }

# ---- arm 5: a concrete WALL patch with a plain (non-wall-function) nut BC refuses ------------------
cp "$SRC/0/nut" "$W/0/nut"
python3 - "$W/0/nut" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("""        type            nutUSpaldingWallFunction;""",
              """        type            fixedValue;""")
open(p, 'w').write(s)
PYEOF
grep -q "fixedValue" "$W/0/nut" || { echo "FAIL: arm-5 mutation did not apply"; exit 1; }
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "no wall" \
    && echo "$out" | grep -q "Refusing rather than substituting Spalding" \
    && say "a plain nut BC on a concrete SA wall patch refuses" ok \
    || { echo "$out" | tail -4; say "a plain nut BC on a concrete SA wall patch refuses" FAIL; }

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
