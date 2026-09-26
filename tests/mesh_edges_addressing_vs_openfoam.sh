#!/usr/bin/env bash
# brae's faceEdges(), edgeFaces() and cellEdges() against REAL OpenFOAM's primitiveMesh -- unit 5a of the
# dynamicRefineFvMesh port. These three are what hexRef8::setRefinement reads, and brae had none of them.
#
# THE ORDER QUESTION, AND WHY IT NEEDED AN ORACLE. Each of the three has TWO implementations in
# primitiveMesh: a cached whole-mesh form and an on-demand one taken while the cache is unbuilt. They need
# not agree, and setRefinement is handed whichever the run happens to have. tools/dumpMeshEdges asks the
# ON-DEMAND forms first -- so the cache cannot answer for them -- then the cached ones, and reports both.
#
# MEASURED on laminar/damBreak's own blockMesh (2268 cells, 9176 faces, 4746 points, 12043 edges):
#     faceEdges   on-demand == cached
#     edgeFaces   on-demand == cached
#     cellEdges   on-demand != cached, differing already at CELL 0
#
# THE cellEdges DISAGREEMENT IS A HASH ORDER: the on-demand form inserts each face's edges into a
# labelHashSet and walks THE SET (primitiveMeshEdges.C:650-668), so its order is bucket order -- a
# function of OpenFOAM's hash and the set's capacity. brae cannot reproduce that and does not try.
#
# AND IT DOES NOT MATTER, read off hexRef8's own call sites rather than hoped for:
#     cellEdges(celli)   hexRef8.C:3415-3433 -- the loop MARKS, `edgeMidPoint[edgeI] = 12345`, under a
#                        condition on the edge's own two point levels. Idempotent, per edge. ANY order.
#     edgeFaces(edgeI)   :3888-3896 -- `affectedFace.set(eFaces)`, a bitSet. ANY order.
#     faceEdges(facei)   :4062-4080 -- `fEdges[fp]`, indexed BY FACE POSITION. THE ORDER IS THE ANSWER.
# So this gate holds faceEdges and edgeFaces EXACTLY and cellEdges AS A SET -- against both of OpenFOAM's
# forms -- and asserts the premise itself: that OpenFOAM's two cellEdges forms really do disagree while
# the other two agree. The day that changes, the set-comparison stops being justified and this says so.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_mesh_edges_addressing_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v dumpMeshEdges > /dev/null 2>&1 \
    || { echo "SKIP: dumpMeshEdges not built (cd tools/dumpMeshEdges && wmake)"; exit 77; }

rc=0

# arm <name> <tutorial path under $TUT> [blockMeshDict.m4 needed]
arm()
{
    local name="$1" rel="$2"
    local C="$W/$name"
    [ -d "$TUT/$rel" ] || { echo "SKIP: no tutorial $TUT/$rel"; return 0; }
    rm -rf "$C"
    cp -r "$TUT/$rel" "$C" || return 1
    rm -rf "$C"/0 "$C"/processor* "$C"/log.*
    [ -d "$C/0.orig" ] && cp -r "$C/0.orig" "$C/0"
    [ -f "$C/system/blockMeshDict.m4" ] && ( cd "$C" && m4 system/blockMeshDict.m4 > system/blockMeshDict )
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && dumpMeshEdges -case . -out d.dump > log.dump 2>&1 ) \
        || { echo "FAIL: dumpMeshEdges [$name]"; tail -20 "$C/log.dump"; return 1; }
    echo "--- $name: $(grep -m1 'on-demand forms read' "$C/log.dump")"
    grep -E "on-demand vs cached" "$C/log.dump" | sed 's/^/      /'
    "$BIN" "$C" "$C/d.dump" || return 1
}

# TWO MESHES, because one of them cannot witness a wedge or a non-hex face. damBreak is the hex fixture
# every other unit of this port is measured on; nozzleFlow2D carries WEDGE patches, whose faces are
# triangles, so faceEdges' per-position order is exercised on a face that is not a quad.
arm damBreak   multiphase/interFoam/laminar/damBreak/damBreak || rc=1
arm nozzle     multiphase/interFoam/LES/nozzleFlow2D          || rc=1

echo "mesh_edges_addressing_vs_openfoam: rc $rc"
exit $rc
