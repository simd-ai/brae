#!/usr/bin/env bash
# brae's pointPatchDist against REAL OpenFOAM's where the wave does NOT reach every point: a mesh of two
# blocks that share no vertex, measured to a patch of the first.
# pointPatchDist.C:52 constructs its field at GREAT (1e15) and :126-136 overwrites the points the wave reached,
# counting the rest. rigidBodyMeshMotion.C:181-206 makes a motion scale of exactly 0 of GREAT: a point with no
# edge path to the body's patches does not move. brae left such a point at 0, whose scale is exactly 1 -- it
# would have moved rigidly with the body (found 2026-10-05 by a sweep for constants; no shipped tutorial has
# such a point, and the gate beside this one, point_patch_dist_vs_openfoam, holds meshes where none is).
# THE FIXTURE: two hex blocks, (0..1)^3 and x in [2, 3], 4 x 2 x 2 cells each, 45 points each; the patch `body`
# is the x = 0 face of the first. The oracle is tools/dumpPointPatchDist, as in the other gate: OpenFOAM's own
# pointPatchDist, the scale rigidBodyMeshMotion makes of it (inner distance 0.2, outer 0.9, so the first
# block's x = 0.25, 0.5 and 0.75 planes lie in the blend band), and its own count of unreached points.
# PREMISE, a number read from OpenFOAM's log: 45 of the 90 points are unreached.
# HELD: brae's count is OpenFOAM's, and brae's distance and scale are OpenFOAM's at all 90 points (the
# unreached ones 1e15 and 0).
# The CONTROL leaves them at 0 (BRAE_CONTROL_POINT_PATCH_DIST_UNSET_ZERO=1): 45 distances must be 1e15 off and
# 45 scales 1.0 off.
# DOES NOT CLAIM: the scale as rigidBodyMeshMotion's own object computes it -- the oracle is the tool's retyping
# of rigidBodyMeshMotion.C:181-206, since that class does not write its scale; RigidBodyMeshMotion::attach on
# such a mesh (the binary calls pointPatchDist and the scale function, as the other gate does); a whole run of
# brae_interFoam on a mesh of two regions.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_point_patch_dist_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/floatingObject"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: floatingObject tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

for t in blockMesh dumpPointPatchDist; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

# the tutorial's case is the shell (controlDict, schemes); the mesh is this script's own
C="$W/case"
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.* "$C"/dynamicCode
mkdir -p "$C/0"
sed -i 's/^writePrecision .*/writePrecision  18;/' "$C/system/controlDict"
sed -i 's/^writeFormat .*/writeFormat     ascii;/' "$C/system/controlDict"
grep -q '^writePrecision  18;' "$C/system/controlDict" || { echo "FAIL: writePrecision not set"; exit 1; }
cat > "$C/system/blockMeshDict" <<'DICT'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
scale 1;
vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1)
    (2 0 0) (3 0 0) (3 1 0) (2 1 0) (2 0 1) (3 0 1) (3 1 1) (2 1 1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (4 2 2) simpleGrading (1 1 1)
    hex (8 9 10 11 12 13 14 15) (4 2 2) simpleGrading (1 1 1)
);
edges ();
boundary
(
    body
    {
        type wall;
        faces ((0 4 7 3));
    }
    walls
    {
        type wall;
        faces
        (
            (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)
            (8 12 15 11) (9 10 14 13) (8 9 13 12) (11 15 14 10) (8 11 10 9) (12 13 14 15)
        );
    }
);
mergePatchPairs ();
DICT
( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -20 "$C/log.blockMesh"; exit 1; }
( cd "$C" && dumpPointPatchDist -patches '(body)' -di 0.2 -do 0.9 > log.dump 2>&1 ) \
    || { echo "FAIL: dumpPointPatchDist"; tail -20 "$C/log.dump"; exit 1; }
N=$(sed -n -E 's/.*\(([0-9]+) points the wave never reached\).*/\1/p' "$C/log.dump" | head -1)
# ...and the dump itself: how many of OpenFOAM's distances are GREAT, and how many of those points have scale 0
read -r NG NZ <<< "$(python3 - "$C/0/pointPatchDist.dump" "$C/0/rbmScale.dump" <<'PY'
import re, sys
def values(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\d+\s*\((.*?)\)\s*;', s, flags=re.S)
    return [float(x) for x in m.group(1).split()] if m else []
d, sc = values(sys.argv[1]), values(sys.argv[2])
great = [i for i, v in enumerate(d) if v == 1e15]
print(len(great), sum(1 for i in great if i < len(sc) and sc[i] == 0.0))
PY
)"

rc=0
say() { printf '  %-5s %s\n' "$1" "$2"; [ "$1" = "ok:" ] || rc=1; }
[ "${N:-0}" = 45 ] && [ "${NG:-0}" = 45 ] && [ "${NZ:-0}" = 45 ] \
    && say "ok:" "PREMISE  OpenFOAM leaves $N of the 90 points unreached: $NG hold 1e15, $NZ of them have scale 0" \
    || say "FAIL:" "PREMISE  OpenFOAM: ${N:-?} unreached, ${NG:-?} at 1e15, ${NZ:-?} with scale 0; 45 expected of each"

"$BIN" "$C" "$C/0" body 0.2 0.9 "unset=${N:-0}" > "$W/log.brae" 2>&1
e=$?
sed -n '/points the wave never reached\|pointPatchDist: worst\|rigidBodyMeshMotion scale/p' "$W/log.brae"
[ $e -eq 0 ] && ! grep -q "CONTROL MODE" "$W/log.brae" \
    && grep -q "pointPatchDist: worst .* 0 of 90 points above 1e-12" "$W/log.brae" \
    && grep -q "rigidBodyMeshMotion scale .* 0 above" "$W/log.brae" \
    && say "ok:" "brae's count, distance and scale are OpenFOAM's on a mesh the wave does not cover" \
    || say "FAIL:" "brae's count, distance or scale is not OpenFOAM's (exit $e)"

BRAE_CONTROL_POINT_PATCH_DIST_UNSET_ZERO=1 "$BIN" "$C" "$C/0" body 0.2 0.9 "unset=${N:-0}" > "$W/log.control" 2>&1
e=$?
sed -n '/pointPatchDist: worst\|rigidBodyMeshMotion scale/p' "$W/log.control" | sed 's/^/  CONTROL/'
[ $e -ne 0 ] && grep -q "CONTROL MODE: pointPatchDist leaves a point the wave never reached at 0" "$W/log.control" \
    && grep -q "pointPatchDist: worst 1.0000e+15 .* 45 of 90 points above 1e-12" "$W/log.control" \
    && grep -q "rigidBodyMeshMotion scale .*: worst 1.0000e+00 .* 45 above" "$W/log.control" \
    && say "ok:" "CONTROL  unreached points left at 0: 45 distances 1e15 off, 45 scales 1.0 off" \
    || say "FAIL:" "CONTROL  unreached points left at 0 did not fail on the 45 points (exit $e)"

echo "point_patch_dist_unreached_vs_openfoam: rc $rc"
exit $rc
