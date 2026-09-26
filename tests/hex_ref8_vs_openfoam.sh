#!/usr/bin/env bash
# brae's hexRef8::setRefinement against REAL OpenFOAM's -- unit 5b of the dynamicRefineFvMesh port, and
# the last piece of its producing half.
#
# THE ORACLE is tools/dumpHexRef8: it builds OpenFOAM's own hexRef8 on the mesh, runs hexRef8's 2:1
# closure (consistentRefinement) on a requested cell set, WRITES THE RESULTING SET so brae refines the
# same one, calls setRefinement, then changeMesh, and dumps everything hexRef8 leaves public.
#
# WHY NO COPY OF hexRef8.C. setRefinement's intermediates are locals, but three of its outputs are not:
# it RETURNS cellAddedCells, and cellLevel()/pointLevel() are accessors. Those three cover sections 1-8
# and 10 -- the points, the cells and the levels -- so the first half of a 1,900-line transcription can be
# held against OpenFOAM without instrumenting anything. If a later disagreement will not localise, THEN
# copy the class and add writes (the of-instrument pattern); not before.
#
# TWO STAGES, and today the gate runs the first:
#   5b-1  cellAddedCells, cellLevel, pointLevel, and the point and cell counts. No face work needed.
#   5b-2  the mapPolyMesh and the new mesh. SKIPPED by name until section 9 exists, and the harness
#         prints the skip count so it cannot quietly become zero coverage.
#
# THREE ARMS on laminar/damBreak's own blockMesh (2268 cells, 9176 faces, 4746 points):
#   one       cell 0 alone. One 8-way split: 2268 -> 2275 cells, 4746 -> 4765 points (1 cell centre + 6
#             face mids + 12 edge mids), 9176 -> 9206 faces (12 internal + 6 faces becoming 4).
#   block     96 cells in a box -- a CONTIGUOUS block, so refined cells share faces, edges and points
#             with each other, which one cell cannot exercise. 2268 -> 2940, 4746 -> 5787, 9176 -> 11540.
#   scattered every 97th cell, so almost every refined cell is surrounded by UNREFINED ones and the
#             refined/unrefined interface is maximised -- the opposite configuration from `block`.
#   twice     the block refined, then ITS CHILDREN refined. The three arms above all start from level 0,
#             and AT LEVEL 0 EVERY LEVEL FORMULA IN setRefinement GIVES 1 -- faceAnchorLevel+1, the master
#             point's level+1, max(edge end levels)+1 -- so none of them can be told apart. MEASURED:
#             fail-proofs on each of those three lines stayed GREEN on all three arms, which is why this
#             arm exists. The tool refines once, writes the resulting mesh, puts its cellLevel and
#             pointLevel in the dump, and refines the children: the levels are then non-uniform, the 2:1
#             closure adds cells of its own (768 requested -> 800 consistent) and the arithmetic is live.
# FOUR FAIL-PROOFS, and the whole point is WHICH ARM SEES THEM. Every one is blind on `one`, `block` and
# `scattered` -- the level-0 arms -- and caught on `twice`:
#   the face mid point's level taken as its master's +1, not faceAnchorLevel+1
#                                     twice: pointLevel[10031] brae 1, OpenFOAM 2
#   the edge mid point's level taken as e0's +1, not max(e0, e1)+1
#                                     twice: pointLevel[6590]  brae 1, OpenFOAM 2
#   findMaxLevel finding the MINIMUM point level instead of the maximum
#                                     twice: pointLevel[10031] brae 1, OpenFOAM 2
#   the anchor test `pointLevel <= cellLevel` made strict
#                                     twice: brae's OWN refusal fires -- "cell 7 has 0 anchor points, not
#                                     8" -- so it is CAUGHT, but as a crash and not as a wrong number,
#                                     which is the weaker of the two demonstrations. Recorded as such.
# That is the measurement that justifies the `twice` arm existing, and it is why the other three are not
# enough on their own even though they are the realistic configurations.
# UNIT 5b-2's FAIL-PROOFS, on the faces. Unlike 5b-1's these bite on EVERY arm, because a face is a face
# whatever the levels are:
#   case 1 ADDS all four split faces instead of modifying the original and adding three
#                                     7 failures on all four arms, starting with the face count
#   addFace/modFace do not reverse the face when the neighbour is the lower cell
#                                     1 failure on all four: face 6 / 53 / 6 / 62
#   storeMidPointInfo does not flip the new internal face when the anchors swap
#                                     1 failure on all four: face 4441 / 4816 / 4777 / 7612
#   case 2 picks the face's MAX-level point as the anchor instead of the min
#                                     GREEN on all four -- NOT WITNESSED
# The last one is not witnessed and that is said rather than left: a face that does not split but has a
# split edge sits on the edge of the refined region, and on these fixtures it carries exactly ONE anchor
# of each split cell -- so min and max pick the same point and getAnchorCell returns the same child. A
# configuration where such a face has two anchors of one split cell would tell them apart; none of these
# four produces one.
# UNIT 6's FAIL-PROOFS -- and only ONE of six is witnessed by a refinement, each for a reason read off
# OpenFOAM rather than guessed:
#   allocateSplitCell does not register the child on its parent
#                                     RED on all four arms: the history's addedCells at split cell 0/8/0/7
#   updateLevels takes the GATHER branch (cellMap) instead of the reorder
#                                     green -- and OpenFOAM's own warning about it does not apply here:
#                                     section 10 already set the level to +1 at BOTH the original cell and
#                                     its seven added ones, so gathering the master's level gives the same
#                                     number. The warning is about a caller that is not hexRef8.
#   allocateSplitCell pops the free list from the FRONT
#                                     green -- a refinement FREES no split cell, so the list is always
#                                     empty and the branch is unreachable
#   storeSplit does not clear the parent's own visibleCells
#                                     green -- and OpenFOAM says why in its own comment: the cell "gets
#                                     alive again below since is addedCells[0]", so the clear is overwritten
#   historyUpdateMesh keeps the old cell index instead of renumbering
#   updateLevels does not remap at all
#                                     both green, and this is the one that matters: MEASURED from
#                                     OpenFOAM's own dumps, a pure refinement leaves reverseCellMap AND
#                                     reversePointMap the IDENTITY (2275/4765 on `one`, 8540/12929 on
#                                     `twice`), because the compaction only renumbers when cells are
#                                     REMOVED. So unit 6's REMAPPING half cannot be witnessed by any
#                                     refinement arm; the unrefinement arm is what turns it on.
# So what unit 6 holds today is the history's PRODUCTION -- visibleCells, parent and addedCells against
# OpenFOAM's on four arms, with addedCells' fail-proof red on all of them -- and the levels after
# changeMesh. Its remapping half is carried but ungated, and that is stated rather than implied.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_hex_ref8_vs_openfoam"
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
command -v dumpHexRef8 > /dev/null 2>&1 \
    || { echo "SKIP: dumpHexRef8 not built (cd tools/dumpHexRef8 && wmake)"; exit 77; }

B="$W/base"
cp -r "$SRC" "$B" || exit 1
rm -rf "$B"/0 "$B"/processor* "$B"/log.*
cp -r "$B/0.orig" "$B/0"
( cd "$B" && blockMesh > log.blockMesh 2>&1 ) \
    || { echo "FAIL: blockMesh"; tail -20 "$B/log.blockMesh"; exit 1; }
grep -q "nCells:  *2268" "$B/log.blockMesh" \
    || { echo "FAIL: damBreak's blockMesh is no longer 2268 cells, so the counts above are stale"; exit 1; }
# a mesh that has never been refined carries no levels, which is what makes every level 0
[ ! -e "$B/constant/polyMesh/cellLevel" ] \
    || { echo "FAIL: the fresh mesh already has a cellLevel, so the arms do not start from level 0"; exit 1; }

rc=0

# arm <name> <dumpHexRef8 args...>
arm()
{
    local name="$1"; shift
    local C="$W/$name"
    rm -rf "$C"
    cp -r "$B" "$C" || return 1
    ( cd "$C" && dumpHexRef8 -case . -out d.dump -cellsOut d.cells "$@" > log.dump 2>&1 ) \
        || { echo "FAIL: dumpHexRef8 [$name]"; tail -20 "$C/log.dump"; return 1; }
    echo "--- $name: $(grep -m1 'consistent set' "$C/log.dump"), $(grep -m1 '^cells ' "$C/log.dump")"
    "$BIN" "$C" "$C/d.cells" "$C/d.dump" || return 1
}

# every 97th cell of 2268: 24 cells, almost all of them isolated
python3 -c "print('%d(%s)' % (24, ' '.join(str(97*i) for i in range(24))))" > "$W/scattered.cells"

arm one       || rc=1
arm block     -box '(0.1 0.0 -1) (0.2 0.1 1)' || rc=1
arm scattered -cells "$W/scattered.cells"      || rc=1
# ...AND THE ARM THAT CAN WITNESS THE LEVEL ARITHMETIC -- see the header. The three above all refine from
# level 0, where every level formula in setRefinement gives 1, so none of them can tell those formulas
# apart; fail-proofs on each stayed GREEN on all three. This one refines the block, hands brae the
# resulting mesh AND its levels, and refines the children.
arm twice     -box '(0.1 0.0 -1) (0.2 0.1 1)' -times 2 -meshOut . || rc=1

echo "hex_ref8_vs_openfoam: rc $rc"
exit $rc
