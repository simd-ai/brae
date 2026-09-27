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
# TWO STAGES, and both now run on the refinement arms:
#   5b-1  cellAddedCells, cellLevel, pointLevel, and the point and cell counts. No face work needed.
#   5b-2  the mapPolyMesh and the new mesh, from section 9's faces.
# The unrefinement arms still skip ONE check by name -- removeFaces::setRefinement, the mesh an
# unrefinement produces -- and the harness prints the skip count so it cannot quietly become zero
# coverage.
#
# SEVEN ARMS on laminar/damBreak's own blockMesh (2268 cells, 9176 faces, 4746 points):
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
#   unrefine, unrefine3, unrefineTwice
#             the arms that REMOVE cells, each described at its own call site below. They are what gate
#             setUnrefinement and unit 6's remapping half, and each exists because the one before it was
#             measured to be blind to something.
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
#                                     green on every REFINEMENT arm, and that is a property of the
#                                     fixture and not of the code: MEASURED from OpenFOAM's own dumps, a
#                                     pure refinement leaves reverseCellMap AND reversePointMap the
#                                     IDENTITY (2275/4765 on `one`, 8540/12929 on `twice`), because the
#                                     compaction only renumbers when cells are REMOVED. The three
#                                     unrefinement arms are what turn them on -- see unit 6b-1 below.
# UNIT 6b-1's FAIL-PROOFS -- setUnrefinement's levels and history, and unit 6's remapping half with them.
# Every one is green on `one` and `twice` (a refinement removes nothing), and the interesting column is
# WHICH unrefinement arm sees it:
#                                                             unrefine unrefine3 unrefineTwice
#   combineCells takes the master's own index as the parent       3 F      3 F        3 F
#   setUnrefinementLevels does not decrement the cell level       2 F      2 F        2 F
#   the master cell is the MAX of the eight, not the min          2 F      2 F        2 F
#   combineCells reads the parent index AFTER freeing the
#     children (OpenFOAM's own ordering, :1652-1673)              3 F      3 F        3 F
#   the parent's addedCells is left as it stands, not cleared    throw    throw      throw
#   historyUpdateMesh keeps the old cell index                   GREEN     1 F        1 F
#   updateLevels does not remap the CELL levels                  GREEN     1 F        1 F
#   updateLevels does not remap the POINT levels                 GREEN    GREEN       1 F
#   freeSplitCell ERASES the child instead of writing -1 in its
#     positional slot                                            GREEN    GREEN      GREEN
# Two of those need saying rather than leaving. The `unrefine` column is green on the three remapping
# lines because undoing EVERY split puts the mesh back exactly and every survivor keeps its own index --
# that measurement is why `unrefine3` exists. `unrefine3` is green on the POINT remap because a
# once-refined mesh's new points are one uniform level-1 block and removals only shuffle within it, so a
# resize reproduces OpenFOAM's pointLevelFinal exactly -- that measurement is why `unrefineTwice` exists,
# and there 558 of 6054 renumbered points land in a slot that held a different level. The last line is
# NOT witnessed by any arm and will not be: setUnrefinement frees eight SIBLINGS, whose parent's
# addedCells is then cleared outright, so the positional -1 is overwritten either way. It is transcribed
# because refinementHistory's other callers (compact, and a partial free) do read those slots
# positionally; a fixture that reaches one of them would tell them apart, and none of these seven does.
# So unit 6 now holds BOTH halves: the history's production on seven arms and its remapping on three, and
# unit 6b-1 holds setUnrefinement's levels and history.
# UNIT 6b-2: removeFaces::compatibleRemoves, which answers WHICH CELLS MERGE with which. Its three
# outputs and its return value are compared on the three unrefinement arms, and TWICE each:
#   the full set     the faces hexRef8 itself asks about -- every face at a split point. The list is
#                    handed over in OpenFOAM'S OWN ORDER (labelHashSet::toc, dumped as `splitFacesToc`),
#                    because region 0 is whichever region the FIRST face created: cellRegion's values and
#                    cellRegionMaster's indices are a function of that order and brae has no HashSet.
#                    Nothing in the resulting MESH depends on it -- hexRef8 overwrites every region's
#                    master with its own and removeFaces walks regions rather than region labels -- but a
#                    comparison of cellRegion does, so it is data and it is dumped.
#   a reduced set    the same faces with the LOWEST-NUMBERED face of each block dropped. This exists
#                    because the recount cannot otherwise be told from returning its input: the twelve
#                    faces at a split point ARE the twelve internal faces of its 2x2x2 block, and hexRef8
#                    FatalErrors if the recount returns any other number at all (hexRef8.C:5686-5710).
#                    MEASURED: 1152 faces in and 1152 out on `unrefine`, the same SET; with one face per
#                    block dropped, 1056 in and 1152 out, so the walk has to find 96 faces nobody asked
#                    for. The block stays connected through its other eleven faces, so the regions are
#                    unchanged and only the face list moves.
# EIGHT FAIL-PROOFS, all three arms alike -- there is no arm-dependence here, which is itself the finding:
# a merge of two regions happens on every one (1 freed region on `unrefine`, 0 on `unrefine3`, 133 on
# `unrefineTwice`, and `unrefine3`'s reduced set frees one).
#   a new region's master is the NEIGHBOUR, not the owner              throws: brae's own transcribed
#   joining an owner to a region does not take min() of the master     check fires, "the master is not
#   changeCellRegion changes one cell instead of the component        the lowest-numbered cell" / "region
#                                                                     N has only 1 cell". Caught, but as
#                                                                     a crash and not a wrong number --
#                                                                     the weaker demonstration, recorded
#                                                                     as such
#   the merge keeps the HIGHER-numbered region       4 / 2 / 4 failures, first at cellRegion
#   the freed region's master is left standing       4 / 2 / 4 failures, first at cellRegionMaster
#   the recount returns its input                    3 failures on all three -- and it fails on the FULL
#                                                    set too, because the input is in hash order and the
#                                                    recount's answer is ASCENDING
#   the recount returns its input SORTED             2 failures, and ONLY on the reduced set: that is the
#                                                    one comparison that can see a face being ADDED
#   each cellCells row walked BACKWARDS              GREEN on all three, and that is the measurement
#                                                    behind porting OpenFOAM's recursion as an explicit
#                                                    stack: the flood assigns one region to a connected
#                                                    component, so no traversal order can change it
# UNIT 6b-3a: removeFaces::setRefinement's DECISIONS -- which edges go, which faces merge with which,
# which points go, which faces are touched at all. Every one of those is a LOCAL of OpenFOAM's function,
# so the oracle is the of-instrument pattern: tools/dumpHexRef8/removeFacesDump.C is a COPY of
# OpenFOAM's own class with WRITES ONLY, called on the three inputs hexRef8 hands it -- compatibleRemoves'
# outputs, which the arms above already compare. It writes to <dump>.removeFaces.
# TWO PROFILES, and the reason is a constant nobody would guess: hexRef8 constructs its faceRemover with
# GREAT, not an angle (hexRef8.C:1967, "merge boundary faces wherever possible"), so `minCos_ < 1` is
# FALSE and setRefinement's feature-angle guard never runs on the unrefinement path at all.
#   GREAT      what hexRef8 itself does. The guard is unreachable.
#   cos(45deg) a configuration hexRef8 never asks for. removeFaces has other callers and brae has to
#              transcribe the guard, so this arm is what holds it against OpenFOAM: <dump>.removeFaces45.
# On these fixtures the two profiles produce the SAME answer -- OpenFOAM's own log prints zero "not
# merging faces" warnings on every arm -- because every pair of boundary faces that merges here is
# coplanar. What the second profile gates is that the guard is REACHED and compares the right way round.
# NINE FAIL-PROOFS:
#                                                             unrefine unrefine3 unrefineTwice
#   the edge count starts at edgeFaces().size(), not size()-1     throws   throws     throws
#   an untouched edge with exactly 2 faces is set to 2 instead
#     of being left at -1                                          8 F      8 F        8 F
#   a single-face region is left at -1 instead of marked -2       HANGS    HANGS      HANGS
#   a point with 2 remaining edges is kept                         4 F      4 F        4 F
#   the face walk also crosses edges left at -1                    6 F      6 F        6 F
#   the feature-angle guard's comparison is reversed             7 F on the cos45 profile ONLY, all three
#                                                               arms, and GREEN on the GREAT profile --
#                                                               which is how the two profiles are proved
#                                                               to run different code
#   the two internal survivors' region pairs compared as
#     ORDERED pairs (Foam::edge equality is SYMMETRIC)            12 F     GREEN      12 F
#   getFacesAffected marks the region MASTER's faces too         GREEN      2 F        2 F
#   two boundary survivors on DIFFERENT patches are merged
#     anyway                                                     GREEN    GREEN      GREEN
# Four of those want saying rather than leaving:
#   The -2 one HANGS instead of failing: without the marker the walk finds the same unvisited face again
#   and never advances. So it IS caught, and as the worst kind of catch. That the branch is live is read
#   off OpenFOAM's own dump instead: 8,772 faces marked -2 on `unrefine`, 10,848 on `unrefine3` and
#   24,681 on `unrefineTwice`.
#   The ORDERED-pair one is green on `unrefine3` because its merge regions are ISOLATED -- every third
#   split point -- so no two internal faces run between the same pair of regions in opposite senses.
#   getFacesAffected's master test is green on `unrefine` because there every face of a region master is
#   also a face of a sibling being removed, or touches a removed edge or point, so marking it changes
#   nothing; with survivors interleaved it does.
#   The patch-boundary one is NOT WITNESSED on any arm, and OpenFOAM says why: it logs zero "not merging"
#   warnings. An edge whose two survivors lie on DIFFERENT patches is never reduced to 2 faces by a
#   hexRef8 unrefinement -- the edges that are reduced are interior to one parent face. Transcribed
#   because removeFaces has other callers; recorded as unwitnessed rather than counted.
# What is still skipped, by name, is the rest of removeFaces::setRefinement -- the ACTIONS it plays into
# the polyTopoChange, mergeFaces' own choice of master face, and the mesh that comes out (unit 6b-3b).
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
# ...AND THE UNREFINE ARM, which is the only one that REMOVES cells. A refinement leaves reverseCellMap and
# reversePointMap the IDENTITY (measured: 2275/4765 on `one`, 8540/12929 on `twice`), so unit 6's remapping
# half was carried but could not be witnessed by any arm above. Here the maps have 2940 and 5787 entries and
# neither is the identity, and cellsFromCellsMap carries 96 merge sets. MEASURED: 2940 -> 2268 cells,
# 11540 -> 9176 faces, 5787 -> 4746 points -- the block refined and then put back.
arm unrefine  -box '(0.1 0.0 -1) (0.2 0.1 1)' -unrefine -meshOut . || rc=1
# ...AND A PARTIAL ONE, because `unrefine` above is still not enough. Undoing EVERY split puts the mesh back
# exactly, and then every surviving cell keeps its own index -- the children that go were appended at the
# end -- so reverseCellMap is the IDENTITY on its live part and the remap cannot be told from a truncation.
# MEASURED: fail-proofs on updateLevels' remap and historyUpdateMesh's renumber stayed GREEN on `unrefine`.
# Undoing every THIRD split leaves the removed cells INTERLEAVED with survivors: 2940 -> 2716 cells, and
# 448 cells and 905 points are genuinely RENUMBERED. That is the arm that gates unit 6's remapping half.
arm unrefine3 -box '(0.1 0.0 -1) (0.2 0.1 1)' -unrefine -unrefineStride 3 -meshOut . || rc=1
# ...AND ONE MORE, because `unrefine3` still cannot witness the POINT half of the remap. MEASURED off
# OpenFOAM's own maps: on unrefine3 all 905 renumbered points are level 1 and every slot they move into
# held level 1 too, so a resize that keeps the old ordering gives OpenFOAM's pointLevelFinal exactly --
# a once-refined mesh appends all its new points in one uniform-level block, and removals only shuffle
# within it. Refine TWICE first and the split points span both levels (level-1 splits left by the 2:1
# closure and level-2 splits), so removals sit in the level-1 block too and level-2 survivors shift
# across the boundary: 8540 -> 6748 cells, 12929 -> 11844 points, 6054 points and 3787 cells renumbered,
# of which 558 points and 1435 cells land in a slot that held a DIFFERENT level. That is what turns the
# pointLevel remap on.
arm unrefineTwice -box '(0.1 0.0 -1) (0.2 0.1 1)' -times 2 -unrefine -unrefineStride 3 -meshOut . || rc=1

echo "hex_ref8_vs_openfoam: rc $rc"
exit $rc
