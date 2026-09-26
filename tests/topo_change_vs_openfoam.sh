#!/usr/bin/env bash
# brae's polyTopoChange ACTION SURFACE and changeMesh against REAL OpenFOAM's -- units 3 and 4 of the
# dynamicRefineFvMesh port.
#
# THE PROBLEM THIS GATE SOLVES. polyTopoChange's accumulated state is PRIVATE (only points(), faces(),
# region(), faceOwner() and faceNeighbour() are public) and it is consumed by changeMesh in the same
# call, so it cannot be dumped out of OpenFOAM and replayed. A `#define private public` instrument was
# considered and rejected: it mangles every transitive include.
#
# SO THE ACTION LIST IS THE INPUT. tools/dumpTopoChange authors one, WRITES IT TO DISK, plays it into
# OpenFOAM's own polyTopoChange and dumps the mapPolyMesh that comes out -- every map and its reverse,
# flipFaceFlux, all eight inflation lists, the old patch description -- plus the mesh itself. brae reads
# the same file, replays it through its own action surface and changeMesh, and must produce the same map
# AND the same mesh. No OpenFOAM source is edited and no arithmetic is changed.
#
# THREE ARMS, on laminar/damBreak's own blockMesh (2268 cells, 9176 faces, 4746 points):
#   noop        no actions. changeMesh must return the IDENTITY, with the patch slicing equal to
#               constant/polyMesh/boundary's. 2268 -> 2268, 9176 -> 9176, 4746 -> 4746.
#   merge0      `mergeTwoCells -cell 0`: the REMOVAL path. The shared face is removed, every other face
#               of the cell is modified onto the survivor, and the cell is removed with the survivor as
#               its merge master. 2268 -> 2267, 9176 -> 9175, cellsFromCellsMap 1 set -- and ZERO flips.
#   merge2267   the last cell, whose first internal face leads to cell 2248 -- a LOWER survivor, so the
#               ACTION carries flipFaceFlux and the map must too. 1 flipped face.
#
# WHY TWO MERGE ARMS, measured rather than asserted -- and NOT for the reason first written here. The
# two differ in what they can witness:
#   * merge0 alone witnesses the `-master-2` RE-ENCODE in renumberReverseMap: its master is cell 1,
#     which becomes cell 0, so the encoding has to be decoded and rebuilt. On merge2267 the master is
#     cell 2248 and nothing below it is removed, so the re-encode is the identity and the arm is BLIND
#     to it -- measured, that fail-proof reads 2 failures on merge0 and 0 on merge2267.
#   * merge2267 alone witnesses flipFaceFlux surviving the two face REORDERS: 1 failure there, 0 on
#     merge0, which has no flagged face at all.
#
# AND THE COMPACTION'S FLIP RULE IS UNREACHABLE ON THIS PATH, which is a fact about OpenFOAM and not a
# hole in the gate. `localCellMap` is built by walking cells ascending and handing the survivors
# consecutive indices, so it is strictly MONOTONIC: `own < nei` before implies `own < nei` after, and no
# face can invert. The block exists in OpenFOAM because `orderCells` can be true (the Cuthill-McKee
# branch), and dynamicRefineFvMesh never passes it. MEASURED: deleting the flip's face reversal, and
# deleting its flag toggle, BOTH leave all three arms green. Said here so the next reader does not take
# those two green fail-proofs for a weak gate.
#
# AND THE addMesh ROUND TRIP runs before every arm: `polyTopoChange meshMod(mesh)` must reproduce the
# mesh exactly -- all three maps and all three reverse maps the identity, the face lists, owner,
# neighbour and region the mesh's, nothing retired and nothing inflated. That is unit 3 checked on its
# own, and it is the only thing about the action surface that OpenFOAM cannot be asked about directly.
#
# WHAT THESE ARMS CANNOT SEE is the 8-way hex split itself -- authoring one by hand IS writing
# hexRef8::setRefinement, which is unit 5. But NOT the inflation maps, and that was written wrongly here
# at first: `setRefinement` NEVER FILLS THEM. Its four action sites pass, measured in hexRef8.C:
#   polyAddPoint  a POINT master (the edge's first vertex, :3484-3494)
#   polyAddCell   a CELL master (:3838-3846)
#   addFace       a FACE master (:2790-2840)
#   addInternalFace  NO master at all, in BOTH branches -- OpenFOAM's own comment is "For now create out
#                 of nothing" (:2880-2930), so the face simply is not mapped
# CONFIRMED against a real refinement step of laminar/damBreakWithObstacle (tools/dumpRefineMap, 24,185 ->
# 24,815 cells): ncellsFromCells, ncellsFromFaces, ncellsFromEdges, ncellsFromPoints, nfacesFromFaces,
# nfacesFromEdges, nfacesFromPoints and npointsFromPoints are ALL ZERO. So changeMesh's inflation refusal
# guards a path OpenFOAM does not take on refinement -- it is a correct refusal, not a deferred one, and
# unit 5 does not lift it. The gate asserts the five lists are empty on BOTH sides rather than skipping
# them: an empty comparison that is empty for a measured reason is not one nobody looked at.
#
# ALSO REFUSED by changeMesh, each for want of a fixture and not for difficulty: a coupled patch
# (reorderCoupledFaces is a parallel exchange), zones (resetZones, 369 lines), `inflate == true`, and
# `orderCells`/`orderPoints` -- dynamicRefineFvMesh passes the defaults for all four.
#
# SIX FAIL-PROOFS, and what each one says:
#   the upper-triangular + patch re-order dropped        noop 0, merge0 4, merge2267 6   <- FOUND A DEFECT
#   getMergeSets' element 0 = the master's old label      noop 0, merge0 1, merge2267 1
#   renumberReverseMap's -master-2 branch dropped        noop 0, merge0 2, merge2267 0
#   flipFaceFlux dropped from the face reorder           noop 0, merge0 0, merge2267 1
#   the compaction flip's face reversal dropped          all three 0 -- unreachable, see above
#   the compaction flip's flag toggle dropped            all three 0 -- unreachable, see above
# THE FIRST ONE IS WHY THIS GATE EXISTS. compact's LAST step is makeCells + getFaceOrder +
# reorderCompactFaces (polyTopoChange.C:1288-1308), putting the faces into upper-triangular and patch
# order. Units 1 and 2 ported both functions and gated them -- and compactNoOrder never CALLED them.
# `noop` could not see it (nothing moves when nothing is removed); the merge arms read faceMap[0] = 1
# where OpenFOAM has 2, neighbour[0] = 22 where OpenFOAM has 1, and a different vertex list on face 0.
#
# `oldPatchNMeshPoints` is dumped but NOT compared: it is patch.meshPoints().size() per patch, which
# only patch POINT fields consume, and brae's MapPolyMesh does not carry one. The harness passes zeros
# and says so rather than inventing a value.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_topo_change_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
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
command -v dumpTopoChange > /dev/null 2>&1 \
    || { echo "SKIP: dumpTopoChange not built (cd tools/dumpTopoChange && wmake)"; exit 77; }

# the base mesh, once
B="$W/base"
cp -r "$SRC" "$B" || exit 1
rm -rf "$B"/0 "$B"/processor* "$B"/log.*
cp -r "$B/0.orig" "$B/0"
( cd "$B" && blockMesh > log.blockMesh 2>&1 ) \
    || { echo "FAIL: blockMesh"; tail -20 "$B/log.blockMesh"; exit 1; }
# the fixture is the one this gate's measurements were taken on
grep -q "nCells:  *2268" "$B/log.blockMesh" \
    || { echo "FAIL: damBreak's blockMesh is no longer 2268 cells, so the measurements below are stale"; exit 1; }

rc=0

# arm <name> <extra dumpTopoChange args...>
arm()
{
    local name="$1"; shift
    local C="$W/$name"
    rm -rf "$C"
    cp -r "$B" "$C" || return 1
    ( cd "$C" && dumpTopoChange -case . -out d.dump -actions a.actions "$@" > log.dump 2>&1 ) \
        || { echo "FAIL: dumpTopoChange [$name]"; tail -20 "$C/log.dump"; return 1; }
    echo "--- $name: $(grep -m1 '^cells ' "$C/log.dump")"
    "$BIN" "$C" "$C/a.actions" "$C/d.dump" || return 1
}

arm noop      -scenario noop                     || rc=1
arm merge0    -scenario mergeTwoCells -cell 0     || rc=1
arm merge2267 -scenario mergeTwoCells -cell 2267  || rc=1

# THE ARMS TOOK THE PATHS THEIR NAMES CLAIM, read out of the oracle's own dumps rather than assumed
python3 - "$W" <<'PYEOF' || rc=1
import re, sys
W = sys.argv[1]
def val(path, key):
    for line in open(path):
        if line.startswith(key + ' '):
            return line.split()[1:]
    return None
bad = 0
n0 = val(W + '/noop/d.dump', 'cellsFromCellsMap')
if n0 is None or int(n0[0]) != 0:
    print("  FAIL: the noop arm produced merge sets, so it is not the identity"); bad = 1
for name, want in (('merge0', 0), ('merge2267', 1)):
    sets = val(W + '/%s/d.dump' % name, 'cellsFromCellsMap')
    flips = val(W + '/%s/d.dump' % name, 'flipFaceFlux')
    if sets is None or int(sets[0]) != 1:
        print("  FAIL: %s produced %s cell merge sets, not 1" % (name, sets)); bad = 1
    if flips is None or int(flips[0]) != want:
        print("  FAIL: %s produced %s flipped faces, not %d -- the arm no longer means what its name "
              "says" % (name, flips, want)); bad = 1
    # ...and the flag comes from the ACTION, not from the compaction's flip rule, which is unreachable
    # while orderCells is false -- see the header. Asserted so a future OpenFOAM that started flipping
    # here would surface rather than quietly change what the arm covers.
    acts = open(W + '/%s/a.actions' % name).read()
    actionFlips = sum(1 for l in acts.splitlines()
                      if l.startswith('modifyFace') and l.split()[-5] == '1')
    if actionFlips != want:
        print("  FAIL: %s has %d action-level flips but %d in the map -- the flip is no longer the "
              "action's" % (name, actionFlips, want)); bad = 1
    else:
        print("  PROPERTY: %s -- 1 cell merge set, %d flipped face(s)" % (name, want))
if not bad:
    print("  PROPERTY: the noop arm is the identity; merge0 witnesses the merge re-encode and merge2267 the flag")
sys.exit(bad)
PYEOF

echo "topo_change_vs_openfoam: rc $rc"
exit $rc
