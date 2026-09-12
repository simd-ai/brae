#!/bin/bash
# Audit finding #16: the turbulence-setup guards compared boundaryField keys to patch names EXACTLY, so
# a regex or group key -- `"(upperWall|lowerWall)"` is how backwardFacingStep2D writes every wall BC --
# resolved to nothing and the guard silently skipped it. buildField has always resolved keys OF's way
# (exact name, then group, then regex, last pattern wins; patch_entry_lookup.cuh), so the fields were
# attached correctly while the guards checking them were blind.
#
# The guards now resolve through the same machinery (patchesResolvingTo / findPatchEntry). Two arms are
# discriminating -- they RAN SILENTLY before the fix:
#   arm 1: a wall function whose regex key resolves to a non-'wall' patch must refuse (guardWallFn).
#   arm 3: a regex-keyed PLAIN nut BC on an SA wall must refuse instead of getting the Spalding
#          hard-force (the #15 plain-BC check, previously reachable only through concrete keys).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
BFS="$ROOT/validation/backwardFacingStep2D"
AF="$ROOT/validation/airFoil2D"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary"; exit 77; }
[ -d "$BFS/constant" ] || { echo "SKIP: fixture $BFS missing"; exit 77; }
[ -d "$AF/constant" ]  || { echo "SKIP: fixture $AF missing"; exit 77; }

fail=0
say() { printf '  %-68s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# ---- arms 1-2: guardWallFn sees through the regex key (backwardFacingStep2D) -----------------------
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$BFS/0" "$BFS/constant" "$BFS/system" "$W/"
sed -i 's/^endTime.*/endTime 1;/;s/writeInterval.*/writeInterval 1;/' "$W/system/controlDict"

# arm 2 FIRST (control): the unmutated case, regex keys everywhere, still starts.
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -qE "^Time = |iter" \
    && say "bfs2D with its regex-keyed wall BCs still runs (control)" ok \
    || { echo "$out" | tail -4; say "bfs2D with its regex-keyed wall BCs still runs (control)" FAIL; }

# arm 1: retype lowerWall wall -> patch. The 0/nut and 0/omega wall functions are keyed
# "(upperWall|lowerWall)", which the exact-name compare could never see -- this ran silently pre-fix.
python3 - "$W/constant/polyMesh/boundary" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
i = s.find('lowerWall\n')                      # the concrete patch block (not lowerWallStartup)
j = s.find('type', i)
s = s[:j] + s[j:].replace('type            wall;', 'type            patch;', 1)
open(p, 'w').write(s)
PYEOF
grep -A2 "^    lowerWall$" "$W/constant/polyMesh/boundary" | grep -q "patch;" \
    || { echo "FAIL: the lowerWall retype did not apply"; exit 1; }
out=$(cd "$W" && "$BRAE" -case "$W" 2>&1 || true)
echo "$out" | grep -q "resolves to patch 'lowerWall'" \
    && echo "$out" | grep -q "not 'wall'" \
    && say "a regex-keyed wall function on a non-wall patch refuses by name" ok \
    || { echo "$out" | tail -4; say "a regex-keyed wall function on a non-wall patch refuses by name" FAIL; }

# ---- arms 3-4: the SA checks resolve regex keys (airFoil2D) ----------------------------------------
W2=$(mktemp -d)
cp -r "$AF/0" "$AF/constant" "$AF/system" "$W2/"
sed -i 's/^endTime.*/endTime         1;/;s/^writeInterval.*/writeInterval   1;/' "$W2/system/controlDict"

# arm 4 (control): the walls entry re-keyed as the regex "wal.*", Spalding kept -- resolution must
# feed the selection and the case must run with the honest label.
sed -i 's/^    walls$/    "wal.*"/' "$W2/0/nut"
grep -q '"wal\.\*"' "$W2/0/nut" || { echo "FAIL: the wal.* re-key did not apply"; exit 1; }
out=$(cd "$W2" && "$BRAE" -case "$W2" 2>&1 || true)
echo "$out" | grep -q "honoured on SpalartAllmaras per the BC" \
    && echo "$out" | grep -qE "^Time = |iter" \
    && say "SA walls keyed by regex still run with Spalding honoured (control)" ok \
    || { echo "$out" | tail -4; say "SA walls keyed by regex still run with Spalding honoured (control)" FAIL; }

# arm 3: the same regex key carrying a PLAIN BC -- pre-fix the exact-name compare missed it and the SA
# path ran the Spalding hard-force over a fixedValue the case asked for.
sed -i 's/type            nutUSpaldingWallFunction;/type            fixedValue;/' "$W2/0/nut"
out=$(cd "$W2" && "$BRAE" -case "$W2" 2>&1 || true)
echo "$out" | grep -q "resolves its nut BC to key 'wal.\*'" \
    && echo "$out" | grep -q "Refusing rather than substituting Spalding" \
    && say "a regex-keyed plain nut BC on an SA wall refuses by name" ok \
    || { echo "$out" | tail -4; say "a regex-keyed plain nut BC on an SA wall refuses by name" FAIL; }
rm -rf "$W2"

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
