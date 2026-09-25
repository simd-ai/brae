#!/usr/bin/env bash
# brae's polyTopoChange face ordering against REAL OpenFOAM's own mesh numbering. Unit 1 of
# dynamicRefineFvMesh's PRODUCING half.
#
# THE ORACLE IS THE MESH, not a dump. `getFaceOrder` is the function that decides where every face of a
# changed mesh ends up, and any mesh OpenFOAM has written is ALREADY in that order -- so feeding its own
# owner/neighbour/patch-id back in must return the IDENTITY permutation, with `patchStarts` equal to
# `constant/polyMesh/boundary`'s startFace values and `patchSizes` equal to its nFaces. No
# instrumentation, no dumped file, nothing to drift.
#
# VERIFIED BEFORE A LINE OF THE PORT WAS WRITTEN, on a real 172,230-face mesh: 0 violations of
# ascending-owner, 0 of within-cell-ascending-neighbour, 0 of owner < neighbour, and patchStarts[0] ==
# nInternalFaces with the patches tiling exactly to nFaces. That is the property this gate rests on.
#
# TWO ARMS, because they exercise different halves:
#   as written              laminar/damBreakWithObstacle's blockMesh mesh -- a lattice, whose
#                           within-cell neighbour order is nearly monotone
#   after a topology change the SAME case after OpenFOAM has refined it -- the mesh the producing half
#                           will actually be asked to reproduce, whose cell numbering is not a lattice
#
# WHY THIS TUTORIAL. It is the one brae's AMR decision side is already gated on
# (tests/refine_candidates_vs_openfoam.sh, tests/field_map_vs_openfoam.sh), so the staging is known to
# refine, and the same 16^3 coarsening keeps it inside the validation ceiling.
#
# UNIT 2, IN THE SAME BINARY: the COMPACTION, on the path dynamicRefineFvMesh actually takes.
# `changeMesh(*this, false)` (dynamicRefineFvMesh.C:462, :585) takes the DEFAULTS, so
# `orderCells == false` and `orderPoints == false` (polyTopoChange.H:720-738). Two consequences that
# decide what is worth porting at all:
#   * `getCellOrder`/`makeCellCells` -- the Cuthill-McKee half -- are NEVER REACHED here
#     (polyTopoChange.C:1190 is the `if (orderCells)` branch), so they are NOT ported. They would be
#     dead code for AMR.
#   * the cell renumber runs only `if (orderCells || newCelli != cellMap_.size())`
#     (polyTopoChange.C:1222): a pure REFINEMENT removes no cell and skips it; an UNREFINEMENT does not.
# Arms: the no-op round trip (every map the identity, the renumber skipped, owner/neighbour untouched)
# and one cell removed (the map compacts past it, the renumber runs).
#
# AND THE FLIP IS UNREACHABLE ON THIS PATH -- asserted as a fact, not tested as a branch. With
# `orderCells == false` the cell map is MONOTONE: a compaction closes gaps and never reorders a pair, so
# `localCellMap[own] < localCellMap[nei]` whenever `own < nei` and polyTopoChange.C:1258-1261's
# `faceNeighbour_ < faceOwner_` is never true. MEASURED on both meshes with a cell removed: 0 faces
# need a flip and 0 get one. The flip block belongs to the `orderCells == true` path, which AMR never
# takes; it is NOT covered here and cannot be.
#
# WHAT IS NOT COVERED, and this gate does not imply otherwise: the point compaction's retired-point
# half (no fixture retires a point), zones, coupled-patch reordering, the face-vertex renumbering, and
# the mapPolyMesh construction. This is the face numbering and the compaction's maps.
#
# BROKEN ONCE EACH, by editing poly_topo_change_cpp.cu and re-running against the same two meshes:
#
#   a face emitted from its HIGHER-numbered cell      2 failures (the identity, on BOTH arms)
#   patch faces walked DESCENDING                     2 failures (the identity, on BOTH arms)
#   patchStarts[0] forced to 0                        6 failures (identity + patchStarts + the
#                                                     nInternalFaces check, on both arms)
#   the within-cell sort made UNSTABLE (std::sort)    0 failures -- THIS ARM CANNOT WITNESS IT
#
# THE FOURTH IS NOT A PASS, it is a gap, and it is stated here rather than left to look like coverage.
# `sortedOrder` is stable (List.C:474-487 is identity() then stableSort) and the stability decides the
# order only where one cell meets another across TWO faces, so the sort keys within that cell tie.
# MEASURED on both fixtures: 11,296 internal faces over 11,296 distinct cell pairs, and 52,798 over
# 52,798 -- **0 pairs share more than one face**, so every within-cell key is distinct and any correct
# sort gives the same answer. A mesh that does have such a pair is what would witness it; none of the
# AMR tutorials produces one, and a synthesised fixture for it is its own unit.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_poly_topo_change_order"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreakWithObstacle"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreakWithObstacle tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh topoSet subsetMesh setFields interFoam; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

C="$W/case"
rm -rf "$C"
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
cp -r "$C/0.orig" "$C/0"

# The same 16^3 coarsening and the same fixed clock the AMR decision gates use, so the mesh stays inside
# the validation ceiling and the case is known to refine within a few steps.
python3 - "$C" <<'PYEOF' || { echo "FAIL: staging"; exit 1; }
import re, sys
d = sys.argv[1]
b = d + '/system/blockMeshDict'
s = open(b).read()
s2, n = re.subn(r'hex\s*\(([^)]*)\)\s*\(\s*\d+\s+\d+\s+\d+\s*\)', lambda m: 'hex (%s) (16 16 16)' % m.group(1), s)
assert n >= 1, 'no hex block to coarsen'
open(b, 'w').write(s2)
c = d + '/system/controlDict'
t = open(c).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', '0.001'), ('endTime', '0.004'),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii')]:
    t, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), t, flags=re.M)
    assert k == 1, key
open(c, 'w').write(t)
PYEOF

( cd "$C" && blockMesh  > log.blockMesh  2>&1 ) || { echo "FAIL: blockMesh";  tail -15 "$C/log.blockMesh";  exit 1; }
( cd "$C" && topoSet    > log.topoSet    2>&1 ) || { echo "FAIL: topoSet";    tail -15 "$C/log.topoSet";    exit 1; }
( cd "$C" && subsetMesh -overwrite c0 -patch walls > log.subsetMesh 2>&1 ) \
    || { echo "FAIL: subsetMesh"; tail -15 "$C/log.subsetMesh"; exit 1; }
( cd "$C" && setFields  > log.setFields  2>&1 ) || { echo "FAIL: setFields";  tail -15 "$C/log.setFields";  exit 1; }
( cd "$C" && interFoam  > log.interFoam  2>&1 ) || { echo "FAIL: interFoam";  tail -25 "$C/log.interFoam";  exit 1; }

# THE SECOND ARM'S MESH: the one OpenFOAM wrote after it refined. A time directory only carries a
# polyMesh when the topology changed there, which is also the assertion that the arm is not vacuous.
REFINED=""
for t in 0.004 0.003 0.002 0.001; do
    if [ -d "$C/$t/polyMesh" ]; then REFINED="$t"; break; fi
done
[ -n "$REFINED" ] || { echo "FAIL: OpenFOAM wrote no refined polyMesh -- the case did not change topology"; grep -iE "refined|unrefined" "$C/log.interFoam" | head -5; exit 1; }
grep -qiE "Refined from [0-9]+ to [0-9]+" "$C/log.interFoam" \
    || { echo "FAIL: the log shows no refinement"; exit 1; }
echo "OpenFOAM refined the mesh: $(grep -oiE 'Refined from [0-9]+ to [0-9]+' "$C/log.interFoam" | head -1)   (mesh at t = $REFINED)"

# the refined mesh, presented as a case directory the reader can open
R="$W/refined"
rm -rf "$R"; mkdir -p "$R/constant"
cp -r "$C/$REFINED/polyMesh" "$R/constant/polyMesh"

rc=0
"$BIN" "$C" "$R" || rc=1

echo "poly_topo_change_order_vs_openfoam: rc $rc"
exit $rc
