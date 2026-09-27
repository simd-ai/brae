#!/usr/bin/env bash
# brae's dynamicRefineFvMesh::updateTopology() against REAL OpenFOAM's, STEPPED -- unit 7 of the
# dynamicRefineFvMesh port, and the first arm that runs the whole adaptive chain rather than one stage.
#
# THE ORACLE is tools/dumpRefineUpdate: OpenFOAM's own dynamicRefineFvMesh with WRITES ONLY
# (dynamicRefineFvMeshDump.C, the of-instrument pattern), stepped over the tutorial's own mesh and
# dictionary. Everything updateTopology decides is a local of that function -- the cells it selects, the
# refineCell marker it rebuilds through the map and extends with buffer layers, the points it unrefines --
# so instrumenting the class is the only way to hold brae's wiring against it.
#
# THE FIELD IS ANALYTIC, and that is the whole reason this gate can exist. The tutorial's alpha.water
# comes out of a solve, so a comparison driven by it would measure interFoam and not the driver. Here the
# field is 1 inside a sphere and 0 outside, recomputed from the CELL CENTRES on whatever mesh each step
# starts from, so both sides are driven by an input that is bit-for-bit the same -- and the harness
# compares the field's own cell list first, so a divergence in the INPUT cannot be read as a defect in the
# output. The sphere MOVES, which is what makes refinement happen ahead of it and unrefinement behind.
#
# WHAT IS COMPARED, per step: the cells selected for refinement; the refinement's six maps; refineCell
# after the map rebuild AND after the buffer layers; the points selected for unrefinement; the
# unrefinement's six maps; the mesh (counts, owner, neighbour and every face's vertex list); cellLevel and
# pointLevel; the refinement history's visibleCells, parent and addedCells; whether the history was
# compacted; and nRefinementIterations.
#
# THREE STEPS, on laminar/damBreakWithObstacle's own mesh (32,256 cells after subsetMesh), MEASURED:
#   step 1  142 cells refined, nothing unrefined      32,256 -> 33,250 cells
#   step 2  470 refined, 44 split points unrefined     33,250 -> 36,540 -> 36,232
#   step 3  361 refined, 286 unrefined                 36,232 -> 38,759 -> 36,757
# So refinement, unrefinement and BOTH-IN-ONE-STEP are all exercised, and the second and third steps
# refine cells that are already children -- which is what the first step cannot do.
#
# WHAT THE SECOND STEP FOUND, and it is why one step is not enough: brae's driver handed
# storeRefinementHistory the COMPACTED per-requested-cell cellAddedCells instead of the per-CELL form
# section 11 indexes (hexRef8.C:4295). On a FIRST refinement that still agreed with OpenFOAM, because a
# fresh history's visibleCells is the identity, so passing a request index in place of a cell label still
# reuses "its own" entry -- and the compaction at the end of the step dropped the difference. On the
# second it came apart: 470 splits allocated 470 new parents where OpenFOAM allocated 64 and reused 406.
#
# THE COMPACTION RUNS ON THE FIRST STEP, which is not obvious and is gated: nRefinementIterations_ starts
# at 0 and the test is `(n % 10) == 0`, so step 1 compacts and steps 2 and 3 do not.
#
# NINE FAIL-PROOFS, five red and four green for reasons that are measured rather than assumed:
#   refineCell is not rebuilt through the map, only resized      22 F, from step 1
#   the rebuild does not mark a non-master child                  4 F, from STEP 2 -- and green on step
#                                                                1, because there every refined cell was
#                                                                also a candidate, so the third clause
#                                                                (`the old cell was marked`) already
#                                                                marked its children. The clause only
#                                                                bites on cells the 2:1 CLOSURE added,
#                                                                which step 1 had none of.
#   no buffer layers                                             19 F, from step 1
#   the history is never compacted                               10 F, from step 1
#   the history is compacted on EVERY step                        8 F, green on step 1 -- which is the
#                                                                one step that really does compact
#   storeRefinementHistory is given the COMPACTED
#     cellAddedCells instead of the per-CELL form                 6 F, from STEP 2. This is the defect
#                                                                the unit found in brae's own wiring;
#                                                                see the note above for why a first
#                                                                refinement cannot see it.
#   the rebuild does not mark a NEW cell (cellMap < 0)           GREEN -- a refinement never produces a
#                                                                cell with cellMap -1, every added cell
#                                                                has a master, so the clause is
#                                                                unreachable from here
#   the driving field is not mapped through cellMap after a
#     refinement                                                 GREEN, and this is the fixture's own
#                                                                doing: `unrefineLevel 10` against a 0/1
#                                                                field makes selectUnrefinePoints' field
#                                                                test unfailable, so the mapped VALUES
#                                                                never reach the answer. Unit 7b gates
#                                                                the mapping itself.
#   markSplit walks the children before the parent               GREEN -- the compaction is gated (the
#                                                                two lines above) but its internal visit
#                                                                ORDER is not discriminated by these
#                                                                three steps
# AND ONE THING THIS FIXTURE CANNOT WITNESS AT ALL: protectedCell is EMPTY at construction here (the
# harness prints it), because every cell of a blockMesh hex mesh has eight anchor points and six faces.
# So the renumbering of protectedCell through each change is carried and ungated. A prism or a
# snappyHexMesh case would fill it; this one does not, and that is said rather than left.
#
# UNIT 7b-1: THE FIELD MAPPING, on three fields the oracle carries and on the old-time volumes.
#   braeScalar   set ONCE, before the first step, to the cell index -- so after three steps it holds
#                THREE MAPPINGS COMPOSED, and each value still says which original cell it came from
#   braeVector   the same, as (i, 2i, 3i), which is the vector path through the same mapper
#   braeFresh    re-set to the cell index at the START OF EVERY STEP. This one exists because the carried
#                fields cannot witness the merge AVERAGING: a refinement copies a parent's value to all
#                eight children, so by the time they are merged back they all hold the same number and
#                their weighted mean equals any one of them. Written fresh, the cells a merge combines
#                hold eight DIFFERENT values.
#   V0           mapped by a rule of its OWN (gather, then ADD each merged part into the master,
#                fvMesh.C:851-891) and then corrected on split and merged cells (correctOldVolumes)
# TWO ARMS, and the difference is which cell volumes weight a merge:
#   OpenFOAM's, injected   the default. The mapping is then arithmetic on identical inputs, so the bound
#                          is ZERO and anything but exactness is a defect. MEASURED: the mapped fields
#                          agree EXACTLY on all three steps.
#   brae's own V           `BRAE_REFINE_UPDATE_OWN_V=1`, which is what the shipped path does. Bound 1e-12.
# V0 is held at 1e-12 on both arms because brae's own V is what overwrites it on split and merged cells;
# MEASURED worst 2.4e-15.
# FIVE FAIL-PROOFS:
#   correctOldVolumes is not applied                      3 F on BOTH arms, V0 out by a factor 8
#                                                         (7.000e+00 relative) -- which is the number
#                                                         unit 6's header predicted for a split cell
#                                                         keeping its parent's volume
#   the mapper is always DIRECT (a merged cell takes its
#     master's value instead of the mean)                  4 F injected / 2 F own, on braeFresh -- and
#                                                         GREEN on braeScalar, which is exactly why
#                                                         braeFresh exists
#   the merge weights are UNIFORM, not volume-weighted     3 F on the INJECTED arm only, at 2.1e-16.
#                                                         MEASURED off OpenFOAM's own dumped volumes:
#                                                         the eight cells of a merge set have volumes
#                                                         equal to within 1.2e-15, so on this uniform
#                                                         mesh the two weightings differ by round-off
#                                                         and only the exact arm can see it. A graded
#                                                         mesh would separate them by more.
#   V0's merged parts are not injected into the master    GREEN on both -- correctOldVolumes OVERWRITES
#                                                         every merged cell's V0 with its own new V, so
#                                                         the injection is invisible behind it. OpenFOAM
#                                                         does it anyway and so does brae.
#   an inserted cell reads nothing instead of cell 0      GREEN -- neither a refinement nor an
#                                                         unrefinement produces a cell with cellMap -1,
#                                                         so cellMapper's inserted-object branch is
#                                                         unreachable from here
#
# WHAT THIS GATE STILL DOES NOT COVER: the SURFACE fields and the FLUX correction (unit 7b-2). And the
# reason that needs its own fixture is worth stating: ALL THREE interFoam tutorials with adaptive meshes
# -- damBreakWithObstacle, oscillatingBox and RAS/motorBike -- map EVERY flux to `none` in their own
# correctFluxes, so the flux correction is unreachable from any of them.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_refine_update_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreakWithObstacle"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreakWithObstacle tutorial not found at $SRC"; exit 77; }
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
command -v dumpRefineUpdate > /dev/null 2>&1 \
    || { echo "SKIP: dumpRefineUpdate not built (cd tools/dumpRefineUpdate && wmake)"; exit 77; }

C="$W/case"
rm -rf "$C"
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/0 "$C"/processor* "$C"/log.*
cp -r "$C/0.orig" "$C/0"
# the tutorial's own four preparation steps, so the mesh is the one the tutorial runs on
( cd "$C" && blockMesh > log.blockMesh 2>&1 \
          && topoSet > log.topoSet 2>&1 \
          && subsetMesh -overwrite c0 -patch walls > log.subsetMesh 2>&1 \
          && setFields > log.setFields 2>&1 ) \
    || { echo "FAIL: preparing the case"; tail -20 "$C"/log.*; exit 1; }
grep -q "Subset 32256 of 32768 cells" "$C/log.subsetMesh" \
    || { echo "FAIL: damBreakWithObstacle is no longer 32256 cells, so the counts above are stale"; exit 1; }
[ ! -e "$C/constant/polyMesh/cellLevel" ] \
    || { echo "FAIL: the fresh mesh already carries a cellLevel, so it does not start from level 0"; exit 1; }

# ...and the instrumented class in place of OpenFOAM's own. Everything else in the dictionary is the
# tutorial's: refineInterval 1, maxRefinement 2, maxCells 200000, nBufferLayers 1, unrefineLevel 10.
sed -i 's/^dynamicFvMesh   dynamicRefineFvMesh;/dynamicFvMesh   dynamicRefineFvMeshDump;/' \
    "$C/constant/dynamicMeshDict"
grep -q "dynamicRefineFvMeshDump" "$C/constant/dynamicMeshDict" \
    || { echo "FAIL: could not select the instrumented mesh class"; exit 1; }

# the sphere: small enough that the gate runs in seconds, moving a cell or so per step
( cd "$C" && dumpRefineUpdate -case . -steps 3 -radius 0.06 -centre '(0.28 0.28 0.14)' \
                              -velocity '(0.05 0 0)' -out d.dump > log.dru 2>&1 ) \
    || { echo "FAIL: dumpRefineUpdate"; tail -25 "$C/log.dru"; exit 1; }
grep -E "^(Refined|Unrefined) from" "$C/log.dru" | sed 's/^/  OpenFOAM: /'
grep -q "^Unrefined from" "$C/log.dru" \
    || { echo "FAIL: OpenFOAM unrefined nothing, so half of the driver is not being measured"; exit 1; }

rc=0
echo "--- arm: OpenFOAM's own old cell volumes injected, so the mapping must be EXACT"
"$BIN" "$C" "$C/d.dump" || rc=1
echo "--- arm: brae's own FvGeometry::V() weights the merges, which is what the shipped path does"
BRAE_REFINE_UPDATE_OWN_V=1 "$BIN" "$C" "$C/d.dump" || rc=1
echo "refine_update_vs_openfoam: rc $rc"
exit $rc
