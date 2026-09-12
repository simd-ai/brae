#!/bin/bash
# Item 16h-b. OpenFOAM's four blended wall functions -- epsilonWallFunction, omegaWallFunction,
# nutkWallFunction and nutUWallFunction, the only classes that reference blender_ -- read a `blending`
# word and a binomial exponent `n` from the PATCH dictionary (wallFunctionBlenders.C:59-82). Each has
# its OWN default: STEPWISE for the three, BINOMIAL with n = 2 for omega
# (omegaWallFunctionFvPatchScalarField.C:405). brae implements each one's default and nothing else, and
# used to skip the entry silently -- a case asking `blending tanh` got stepwise with no notice.
#
# The refusal must be exactly as wide as the gap: it fires on a blender brae does not run, it does NOT
# fire on the wall function's own default (arms 4 and 6, which is why one global default would be wrong
# -- `stepwise` is right on nut and WRONG on omega), it does NOT fire on a derived wall function that
# overrides calcNut and never consults the blender, and it must not have taught the reader to swallow
# `n` where `n` is a vector.
#
# Fixture: validation/pitzDaily (nutkWallFunction + epsilonWallFunction on kEpsilon, omegaWallFunction
# with the model switched to kOmegaSST). One time step is enough: the refusal is in the field reader.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDaily"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary"; exit 77; }
[ -d "$SRC/constant" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
stage() {   # stage() <model>: a clean copy of the fixture, one step, the named RAS model
    rm -rf "$W/c"; mkdir -p "$W/c"
    cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$W/c/"
    sed -i 's/^endTime.*/endTime         1;/;s/^writeInterval.*/writeInterval   1;/' "$W/c/system/controlDict"
    sed -i "s/RASModel        kEpsilon;/RASModel        $1;/" "$W/c/constant/turbulenceProperties"
}
add() {     # add() <field> <wall function type> <entry>: put an entry under every patch of that type
    python3 - "$W/c/0/$1" "$2" "$3" <<'PYEOF'
import sys
path, wf, entry = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
old = "type            %s;" % wf
assert old in s, "no %s in %s" % (wf, path)
open(path, 'w').write(s.replace(old, old + "\n        " + entry))
PYEOF
}
run() { (cd "$W/c" && "$BRAE" -case "$W/c" 2>&1 || true); }
ran()     { echo "$1" | grep -qE "^Time = "; }
refused() { echo "$1" | grep -q "brae ERROR"; }

# ---- arm 1 (negative control): the shipped case carries no blending entry and runs -----------------
stage kEpsilon
out=$(run)
ran "$out" && ! refused "$out" \
    && say "pitzDaily kEpsilon with no blending entry runs (control)" ok \
    || { echo "$out" | tail -3; say "pitzDaily kEpsilon with no blending entry runs (control)" FAIL; }

# ---- arm 2: a blender brae does not run, on nut, refuses BY NAME ----------------------------------
stage kEpsilon; add nut nutkWallFunction "blending        max;"
out=$(run)
echo "$out" | grep -q "nutkWallFunction) sets \`blending max\`" \
    && echo "$out" | grep -q "stepwise" \
    && say "nutkWallFunction + \`blending max\` refuses, naming patch, type and blender" ok \
    || { echo "$out" | tail -3; say "nutkWallFunction + \`blending max\` refuses, naming patch, type and blender" FAIL; }

# ---- arm 3: the same on epsilon ------------------------------------------------------------------
stage kEpsilon; add epsilon epsilonWallFunction "blending        tanh;"
out=$(run)
echo "$out" | grep -q "epsilonWallFunction) sets \`blending tanh\`" \
    && say "epsilonWallFunction + \`blending tanh\` refuses by name" ok \
    || { echo "$out" | tail -3; say "epsilonWallFunction + \`blending tanh\` refuses by name" FAIL; }

# ---- arm 4 (control): the wall function's OWN default is accepted, not refused --------------------
stage kEpsilon; add nut nutkWallFunction "blending        stepwise;"
out=$(run)
ran "$out" && ! refused "$out" \
    && say "nutkWallFunction + \`blending stepwise\` runs (its own default, not blanket)" ok \
    || { echo "$out" | tail -3; say "nutkWallFunction + \`blending stepwise\` runs (its own default, not blanket)" FAIL; }

# ---- arm 5: stepwise is WRONG on omega -- the default is per wall function, not global ------------
stage kOmegaSST; add omega omegaWallFunction "blending        stepwise;"
out=$(run)
echo "$out" | grep -q "omegaWallFunction) sets \`blending stepwise\`" \
    && echo "$out" | grep -q "binomial" \
    && say "omegaWallFunction + \`blending stepwise\` refuses (omega's default is binomial)" ok \
    || { echo "$out" | tail -3; say "omegaWallFunction + \`blending stepwise\` refuses (omega's default is binomial)" FAIL; }

# ---- arm 6 (control): binomial IS omega's default and runs ----------------------------------------
stage kOmegaSST; add omega omegaWallFunction "blending        binomial;"
out=$(run)
ran "$out" && ! refused "$out" \
    && say "omegaWallFunction + \`blending binomial\` runs (its own default)" ok \
    || { echo "$out" | tail -3; say "omegaWallFunction + \`blending binomial\` runs (its own default)" FAIL; }

# ---- arm 7: `n` is live on omega even with no blending entry, because binomial is its default -----
stage kOmegaSST; add omega omegaWallFunction "n               4;"
out=$(run)
echo "$out" | grep -q "omegaWallFunction) sets \`n 4" \
    && echo "$out" | grep -q "exponent 2" \
    && say "omegaWallFunction + \`n 4\` refuses with no blending entry present" ok \
    || { echo "$out" | tail -3; say "omegaWallFunction + \`n 4\` refuses with no blending entry present" FAIL; }

# ---- arm 8 (control): the same `n` under stepwise is inert in OpenFOAM, so brae is silent ---------
stage kEpsilon; add nut nutkWallFunction "n               4;"
out=$(run)
ran "$out" && ! refused "$out" \
    && say "nutkWallFunction + \`n 4\` runs (stepwise never reads it, nor does OpenFOAM)" ok \
    || { echo "$out" | tail -3; say "nutkWallFunction + \`n 4\` runs (stepwise never reads it, nor does OpenFOAM)" FAIL; }

# ---- arm 9 (control): a wall function with NO blender is untouched -------------------------------
# nutUSpaldingWallFunction derives from nutWallFunction, not nutUWallFunction, and is one of the four
# classes that do NOT reference blender_ -- `blending` there is a dead key in OpenFOAM too, so brae
# must let it through. (nutkRoughWallFunction would be the same argument, but brae refuses that type
# outright, which would make the arm pass without ever reaching the blending check.)
stage kEpsilon
python3 - "$W/c/0/nut" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
open(p, 'w').write(s.replace("type            nutkWallFunction;",
                             "type            nutUSpaldingWallFunction;\n        blending        max;"))
PYEOF
out=$(run)
ran "$out" && ! refused "$out" \
    && say "nutUSpaldingWallFunction + \`blending max\` runs (it has no blender in OpenFOAM)" ok \
    || { echo "$out" | tail -3; say "nutUSpaldingWallFunction + \`blending max\` runs (it has no blender in OpenFOAM)" FAIL; }

# ---- arm 10 (control): `n` as a VECTOR still falls through to the unhandled-key skip --------------
# The new scalar parse is gated on the value being a number precisely so that `n (0 0 1)` -- a normal,
# not an exponent -- keeps skipping. Without the gate the reader would try to read a scalar and the
# whole file would fail to parse.
stage kEpsilon; add nut nutkWallFunction "n               (0 0 1);"
out=$(run)
ran "$out" && ! refused "$out" \
    && say "a VECTOR \`n (0 0 1)\` still parses and runs (the scalar gate)" ok \
    || { echo "$out" | tail -3; say "a VECTOR \`n (0 0 1)\` still parses and runs (the scalar gate)" FAIL; }

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
