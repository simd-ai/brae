#!/usr/bin/env bash
# brae's pointPatchDist against REAL OpenFOAM's, on the mesh RAS/floatingObject builds -- and the
# scale field rigidBodyMeshMotion makes of it.
#
# WHY THIS EXISTS. rigidBodyMeshMotion moves the points inside innerDistance of a body rigidly with it,
# leaves those beyond outerDistance alone, and blends between the two by a cosine of the point's
# distance to the body's patches. That distance is pointPatchDist, and it is NOT a nearest-point
# search: PointEdgeWave propagates an ORIGIN point along the mesh's edges, and externalPointEdgePoint
# refuses to propagate an improvement smaller than 1% of the squared distance a point already holds
# (PointEdgeWaveBase.C:40). The answer therefore depends on the edge NUMBERING and on each point's
# pointEdges order, which is why brae reproduces primitiveMesh::calcEdges rather than any equivalent.
#
# THE ORACLE is tools/dumpPointPatchDist: OpenFOAM's own pointPatchDist and, from it, the two clamped
# expressions rigidBodyMeshMotion.C:178-205 turns it into. That class computes the scale and does not
# write it (the `//scale.write();` under it), so the tool writes both.
#
# THE MESH is the tutorial's, built as its Allrun does: blockMesh, topoSet, then subsetMesh carving the
# floating cuboid out and naming its surface `floatingObject` -- 13,461 points, 36,694 faces, 38,513
# edges, and calcPointOrder reports its points UNORDERED, which is the branch brae ports.
#
# TWO ARMS, and the second is what makes the first mean something:
#   floatingObject  the patch rigidBodyMeshMotion measures to, with the tutorial's own innerDistance
#                   0.05 and outerDistance 0.35. MEASURED, at writePrecision 18: the distance and the
#                   scale are OpenFOAM's EXACTLY -- 0.0000e+00 on all 13,461 points. On THIS patch the
#                   wave happens to reach the exact nearest patch point everywhere, so the
#                   brute-force control below cannot separate the two; that is said rather than left
#                   to look like a pass.
#   atmosphere      the same mesh measured to the far boundary, where they part. MEASURED: brae is
#                   0.0000e+00 from OpenFOAM, while the EXACT nearest patch point is 5.539e-03 away
#                   from OpenFOAM's answer on 60 of the 13,461 points -- OpenFOAM leaves
#                   0.905538513813742 where the true nearest is 0.9. That is the arm that says this is
#                   a port of the wave and not a distance function that happens to agree.
#
# CONTROLS: the exact nearest patch point (above), and the scale WITHOUT its cosine -- the linear ramp
# both expressions start from, which agrees outside the blend band and differs by 1.0525e-01 on 6,554
# of the 13,461 points, 7,271 of which lie in the band.
# FAIL-PROOF: the seed origins moved by one part in 10,000 reads 1.042e-04 on 13,111 points in the
# first arm and 1.035e-04 on 13,020 in the second, with four arms red.
#
# NOT DISCRIMINATED, each broken deliberately and the gate still green, so none of it is claimed:
#   * the 1% propagation tolerance (propagationTol set to 0);
#   * the edge NUMBERING (the upper-triangular sort replaced by the creation order);
#   * an edge's own stored distance (taken at the POINT's position rather than the edge centre).
# The wave converges to a FIXED POINT on this mesh -- each point ends at the smallest |p - origin| any
# edge-connected neighbour could hand it -- and a fixed point does not depend on the order it is
# reached in. What the gate does hold is the SEEDS (every mesh point of the patch, itself, at zero),
# the EDGE TOPOLOGY the origins travel along, and that the distance is measured to an ORIGIN POINT and
# not to a face. That the answer is still not the nearest patch point is the `atmosphere` arm's 60.

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

for t in blockMesh topoSet subsetMesh dumpPointPatchDist; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

C="$W/case"
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.* "$C"/dynamicCode
cp -r "$C/0.orig" "$C/0"
# the tutorial's own two distances, read here so the arm below cannot drift from the case
# writePrecision 18: the comparison is against brae's fp64, and the tutorial's default of 6 digits
# would BE the difference -- it read 5.0e-13 instead of 5.0e-16 before this line existed.
sed -i 's/^writePrecision .*/writePrecision  18;/' "$C/system/controlDict"
sed -i 's/^writeFormat .*/writeFormat     ascii;/' "$C/system/controlDict"
grep -q '^writePrecision  18;' "$C/system/controlDict" || { echo "FAIL: writePrecision not set"; exit 1; }
DI=$(sed -n 's/^ *innerDistance *\([0-9.eE+-]*\);.*/\1/p' "$C/constant/dynamicMeshDict" | head -1)
DO=$(sed -n 's/^ *outerDistance *\([0-9.eE+-]*\);.*/\1/p' "$C/constant/dynamicMeshDict" | head -1)
[ -n "$DI" ] && [ -n "$DO" ] || { echo "FAIL: the tutorial no longer names innerDistance/outerDistance"; exit 1; }

( cd "$C" && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 \
      && subsetMesh -overwrite c0 -patch floatingObject > log.subsetMesh 2>&1 ) \
    || { echo "FAIL: meshing"; tail -20 "$C"/log.* 2>/dev/null; exit 1; }

# unreached <log>: the number of points OpenFOAM's own wave never reached, from the tool's log. Both arms hold
# it to 0 -- the mesh is one region -- and hand it to the binary, which holds brae's count to it (a mesh where
# it is not 0 is point_patch_dist_unreached_vs_openfoam.sh's)
unreached() { sed -n -E 's/.*\(([0-9]+) points the wave never reached\).*/\1/p' "$1" | head -1; }
rc=0
( cd "$C" && dumpPointPatchDist -patches '(floatingObject)' -di "$DI" -do "$DO" > log.dumpBody 2>&1 ) \
    || { echo "FAIL: dumpPointPatchDist (floatingObject)"; tail -20 "$C/log.dumpBody"; exit 1; }
N=$(unreached "$C/log.dumpBody")
[ "${N:-x}" = 0 ] \
    || { echo "  FAIL: OpenFOAM's wave left ${N:-?} points unreached on floatingObject, 0 expected"; rc=1; }
"$BIN" "$C" "$C/0" floatingObject "$DI" "$DO" "unset=${N:-0}" || rc=1

( cd "$C" && dumpPointPatchDist -patches '(atmosphere)' > log.dumpAtm 2>&1 ) \
    || { echo "FAIL: dumpPointPatchDist (atmosphere)"; tail -20 "$C/log.dumpAtm"; exit 1; }
N=$(unreached "$C/log.dumpAtm")
[ "${N:-x}" = 0 ] \
    || { echo "  FAIL: OpenFOAM's wave left ${N:-?} points unreached to atmosphere, 0 expected"; rc=1; }
"$BIN" "$C" "$C/0" atmosphere "unset=${N:-0}" || rc=1

echo "point_patch_dist_vs_openfoam: rc $rc"
exit $rc
