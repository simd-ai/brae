#!/usr/bin/env bash
# brae's refinement candidate selection against REAL OpenFOAM's own, on laminar/damBreakWithObstacle --
# unit 1 of the dynamicRefineFvMesh port.
#
# WHAT IS UNDER TEST, and what is NOT. Given a field and a band, which cells would an adaptive mesh
# refine? That is `cellToPoint` -> `error` -> `maxPointField` -> a strictly-positive test, and it is
# the one part of AMR that is pure arithmetic on a FIXED mesh: it changes no topology, allocates no
# polyTopoChange and calls nothing on hexRef8. What comes AFTER it -- the 2:1 closure, the level cap,
# the protected cells, and the refinement itself -- is not here and is refused as before: brae still
# refuses `dynamicFvMesh dynamicRefineFvMesh` by name, and interfoam_refusals.sh holds that on both
# arms. This gate reaches the module directly.
#
# THE ORACLE is tools/dumpRefineCandidates. It does not transcribe OpenFOAM's functions: they are
# PROTECTED members of dynamicRefineFvMesh, so it derives a mesh class and re-exports them with
# `using`. Every number it prints is produced by OpenFOAM's own code on OpenFOAM's own pointCells().
#
# THE FIXTURE is laminar/damBreakWithObstacle with the (16 16 16) block its interIsoFoam twin ships,
# so 4,032 cells rather than 32,256, and fixed time stepping. Five steps of 1e-3 take 0.7 s and the
# case refines TWICE: 4032 -> 6650 -> 17080 cells. Everything else -- `refineInterval 1`,
# `lowerRefineLevel 0.001`, `upperRefineLevel 0.999`, `maxRefinement 2`, `field alpha.water` -- is the
# tutorial's own.
#
# THREE ARMS, one per mesh state the run leaves behind, and the third is the one that matters:
#   t0       the base mesh, 4,032 cells all at level 0. OpenFOAM selects 374 candidates, and
#            `selectRefineCells` refines exactly those 374 -- so this arm is also a free end-to-end
#            confirmation that candidate selection alone is what OpenFOAM acted on.
#   t0.001   6,650 cells, levels 0 and 1, a developed alpha. OpenFOAM selects 1,490, and again
#            refines exactly those.
#   t0.002   17,080 cells at levels 0-2. OpenFOAM selects 6,266 candidates and refines NONE of them,
#            because `selectRefineCells` drops every cell already at `maxRefinement`. That is the arm
#            that separates this unit from the next: a port that fused the two stages reads 0 here.
#
# CONTROLS, in the binary: the band widened to the VoF field's own extremes [0, 1] (which must select
# the interface and not the whole mesh, because `error` writes an edge as 0 and the candidate test is
# STRICTLY > 0); the cell-to-point sum without its divisor; and `error` seeded 0 instead of -1.
#
# THE ORDERING, measured rather than assumed. `cellToPoint` accumulates in pointCells order, and
# OpenFOAM's primitiveMesh::calcPointCells has three orders chosen by what the mesh has already
# computed. The binary runs BOTH of the two brae carries and reports which reproduces OpenFOAM. On a
# VoF field whose values are mostly exactly 0 and exactly 1 the two orders can agree bitwise, and the
# binary SAYS SO rather than claiming the arm discriminated an ordering it could not see.
#
# WHAT THE PROTECTED-CELL ARMS DO AND DO NOT ISOLATE, measured with five separate defects rather
# than argued. brae reproduces OpenFOAM's protected sets exactly on both fixtures -- 8 on the wedge,
# 1,886 on the split hex -- but init()'s four passes are REDUNDANT on them: every protected cell there
# is caught by at least two, so removing any ONE changes nothing and the gate stays green.
#   * `cells[c].size() < 6` weakened to `< 5`            -> green (the same pass's `face < 4 points`
#                                                           else-branch catches the prisms instead)
#   * the face-anchor pass removed entirely               -> green (pass a's `> 8` catches the same cells)
#   * checkEightAnchorPoints' `!= 8` weakened to `> 8`    -> green (pass a catches them)
#   * the face-anchor threshold `> 4` raised to `> 5`     -> green (those faces carry 9 anchors)
# What IS caught, and so what these arms measure: the ANCHOR TEST ITSELF and the composition --
# `pointLevel <= cellLevel` made strict reads 40 protected against 8 on the wedge and 17,080 against
# 1,886 on the split hex, and fails selectRefineCells with it. Isolating the four passes needs a
# fixture where each is the only one that fires, and no OpenFOAM tutorial mesh here provides one.
#
# THE BUFFER DILATION IS independently discriminated: setting only the owner on an internal face reads
# 7,815 cells against OpenFOAM's 11,981 on the split hex and fails both layers on the wedge too.
#
# NOT DISCRIMINATED by this gate, each stated rather than implied: the 2:1 consistency closure, the
# cellLevel cap, protected cells, the buffer layers, unrefinement, and the mesh change itself. None of
# it is ported and all of it is still refused.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_refine_candidates_vs_openfoam"
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

for t in blockMesh topoSet subsetMesh setFields interFoam dumpRefineCandidates; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

grep -q "dynamicFvMesh *dynamicRefineFvMesh" "$SRC/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial is no longer a dynamicRefineFvMesh case"; exit 1; }
grep -q "field *alpha.water" "$SRC/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial no longer refines on alpha.water"; exit 1; }

C="$W/case"
rm -rf "$C"
cp -r "$SRC" "$C" || exit 1
chmod -R u+w "$C"
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
# the interIsoFoam twin's block: 16^3 instead of 32^3, so the gate is seconds rather than minutes
sed -i 's/(32 32 32)/(16 16 16)/' "$C/system/blockMeshDict"
grep -q "(16 16 16)" "$C/system/blockMeshDict" \
    || { echo "FAIL: the blockMeshDict no longer carries the (32 32 32) this gate reduces"; exit 1; }
python3 - "$C" <<'PYEOF' || { echo "FAIL: staging"; exit 1; }
import re, sys
d = sys.argv[1]
c = d + '/system/controlDict'
s = open(c).read()
for key, val in [('endTime', '0.005'), ('deltaT', '0.001'), ('adjustTimeStep', 'no'),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'),
                 ('writeFormat', 'ascii'), ('writePrecision', '18'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
PYEOF

( cd "$C" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 \
      && subsetMesh -overwrite c0 -patch walls > log.subsetMesh 2>&1 \
      && setFields > log.setFields 2>&1 && interFoam > log.interFoam 2>&1 ) \
    || { echo "FAIL: running the case"; tail -30 "$C"/log.interFoam 2>/dev/null; exit 1; }

# ...AND IT REFINED. A staging that silently left the case static would make every arm below a
# comparison on one mesh, and the gate would be green and vacuous.
grep -q "^Refined from 4032 to" "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM did not refine the base mesh"; grep -c Refined "$C/log.interFoam"; exit 1; }
grep -q "^Selected 374 cells for refinement out of 4032." "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM's first selection is not the 374 this gate was measured against"; exit 1; }
echo "OpenFOAM ran 5 steps and refined: $(grep -c '^Refined from' "$C/log.interFoam") refinements"

rc=0
# armIn <name> <case dir> <time> <mesh dir> [extra oracle options...]
armIn()
{
    local name="$1" case="$2" t="$3" meshDir="$4"
    shift 4
    local dump="$W/dump.$name.txt"
    ( cd "$case" && dumpRefineCandidates -case "$case" -time "$t" "$@" > "$dump" 2>&1 ) \
        || { echo "FAIL: dumpRefineCandidates [$name]"; tail -20 "$dump"; rc=1; return; }
    echo "--- $name (t = $t)${*:+ [$*]}"
    "$BIN" "$case" "$case/$meshDir" "$case/$t" "$dump" || rc=1
}

# arm <name> <time> <mesh dir> [extra oracle options...] -- on the damBreak case
arm()
{
    local name="$1" t="$2" meshDir="$3"
    shift 3
    armIn "$name" "$C" "$t" "$meshDir" "$@"
}

arm t0     0     constant/polyMesh
arm t0.001 0.001 0.001/polyMesh
arm t0.002 0.002 0.002/polyMesh
# THE BUDGET, which the tutorial as shipped can never exercise: `maxCells 200000` against 17,080 cells
# leaves a budget of 26,131 and the whole-level truncation branch is unreachable. Widening the band to
# the field's own extremes makes EVERY cell a candidate at three different levels, and a ceiling just
# above the current count makes the budget bind. MEASURED: OpenFOAM selects 17,080 with the shipped
# ceiling and 3,658 with this one -- exactly the level-0 cells, because the branch takes WHOLE LEVELS
# coarsest first and stops after the first one that carries it past the budget.
arm budget 0.002 0.002/polyMesh -lower -1 -upper 2 -maxRefinement 3 -maxCells 17100


# THE PROTECTED CELLS need a mesh that is not all hexes, and damBreakWithObstacle is. Two fixtures are
# built here, and they cover different branches of init()'s scan:
#
#   wedge     a 2.5-degree axisymmetric wedge, 40 cells, whose innermost radial row collapses to
#             PRISMS -- five faces, two of them triangles. `mergeType points` is what makes blockMesh
#             merge the axis vertices; without it the axis cells stay collapsed hexes with zero-area
#             faces and nothing is protected. MEASURED: 8 protected cells, exactly the 8 prisms, and
#             OpenFOAM writes constant/polyMesh/sets/protectedCells.
#             ...and it is the ONLY arm here where the protected set CHANGES THE ANSWER: two of its
#             fourteen candidates are protected prisms, so OpenFOAM selects twelve. A port that
#             computes protectedCell_ correctly but never feeds it through calculateProtectedCells
#             into selectRefineCells passes every other arm and fails this one. That defect was real:
#             selectRefineCells took the raw set until this fixture existed.
#   splitHex  the refined 17,080-cell mesh with its cellLevel/pointLevel/refinementHistory/level0Edge
#             files REMOVED, so every cell reads level 0 and a split hex stops being a legal refined
#             hex: 1,886 protected, covering the >4-anchor FACE branch and its owner/neighbour
#             propagation, which the wedge cannot reach.
WC="$W/wedge"
rm -rf "$WC"
mkdir -p "$WC/system" "$WC/constant" "$WC/0"
python3 - "$WC" <<'PYEOF' || { echo "FAIL: staging the wedge fixture"; exit 1; }
import os, sys
d = sys.argv[1]
hdr = ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
       "    class       %s;\n    object      %s;\n}\n")
# tan(2.5 deg) = 0.0436609. `mergeType points` is mandatory: without it the axis hexes stay
# collapsed instead of becoming prisms, and nothing is protected.
open(d + '/system/blockMeshDict', 'w').write(
    hdr % ('dictionary', 'blockMeshDict') + """
mergeType points;
scale   1;
vertices
(
    (0 0 0)
    (1 0 0)
    (0 1 -0.0436609)
    (1 1 -0.0436609)
    (0 1  0.0436609)
    (1 1  0.0436609)
);
blocks
(
    hex (0 1 3 2 0 1 5 4) (8 5 1) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet  { type patch; faces ((0 2 4 0)); }
    outlet { type patch; faces ((1 3 5 1)); }
    top    { type wall;  faces ((2 3 5 4)); }
    front  { type wedge; faces ((0 1 5 4)); }
    back   { type wedge; faces ((0 2 3 1)); }
);
mergePatchPairs ();
""")
open(d + '/system/controlDict', 'w').write(
    hdr % ('dictionary', 'controlDict') + """
application     interFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     ascii;
writePrecision  18;
""")
open(d + '/system/fvSchemes', 'w').write(
    hdr % ('dictionary', 'fvSchemes') + """
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes{ default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
""")
open(d + '/system/fvSolution', 'w').write(hdr % ('dictionary', 'fvSolution') + "\nsolvers { }\n")
open(d + '/constant/dynamicMeshDict', 'w').write(
    hdr % ('dictionary', 'dynamicMeshDict') + """
dynamicFvMesh   dynamicRefineFvMesh;
refineInterval  1;
field           alpha.water;
lowerRefineLevel 0.001;
upperRefineLevel 0.999;
unrefineLevel   10;
nBufferLayers   1;
maxRefinement   2;
maxCells        200000;
correctFluxes ( (phi none) );
dumpLevel       true;
""")
# a fixed pattern, chosen so the candidate set INTERSECTS the protected prisms
vals = ['1' if (i < 24 and i % 8 < 4) else '0' for i in range(40)]
open(d + '/0/alpha.water', 'w').write(
    hdr % ('volScalarField', 'alpha.water')
    + "\ndimensions      [0 0 0 0 0 0 0];\n\ninternalField   nonuniform List<scalar>\n40\n(\n"
    + "\n".join(vals)
    + "\n)\n;\n\nboundaryField\n{\n"
    # a wedge PATCH demands a wedge patchFIELD -- zeroGradient there is a fatal, not a warning
    + "    front { type wedge; }\n    back  { type wedge; }\n"
    + "    \".*\"\n    {\n        type            zeroGradient;\n    }\n}\n")
PYEOF
( cd "$WC" && blockMesh > log.blockMesh 2>&1 ) \
    || { echo "FAIL: blockMesh [wedge]"; tail -20 "$WC/log.blockMesh"; exit 1; }
grep -q "prisms:" "$WC/log.blockMesh" 2>/dev/null || ( cd "$WC" && checkMesh > log.checkMesh 2>&1 || true )
armIn wedge "$WC" 0 constant/polyMesh
grep -q "^\[brae\] nProtectedCells 8$" "$W/dump.wedge.txt" \
    || { echo "FAIL: the wedge fixture no longer protects 8 cells -- the mesh is not prisms"; rc=1; }
[ -f "$WC/constant/polyMesh/sets/protectedCells" ] \
    || { echo "FAIL: OpenFOAM wrote no protectedCells cellSet on the wedge"; rc=1; }

SC="$W/splitHex"
rm -rf "$SC"
mkdir -p "$SC/constant/polyMesh" "$SC/system" "$SC/0"
cp "$C/0.002/polyMesh/points" "$C/0.002/polyMesh/faces" "$C/0.002/polyMesh/owner" \
   "$C/0.002/polyMesh/neighbour" "$C/0.002/polyMesh/boundary" "$SC/constant/polyMesh/" \
    || { echo "FAIL: the refined mesh is not where splitHex expects it"; exit 1; }
cp "$C/0.002/alpha.water" "$SC/0/alpha.water"
cp "$WC/system/controlDict" "$WC/system/fvSchemes" "$WC/system/fvSolution" "$SC/system/"
cp "$WC/constant/dynamicMeshDict" "$SC/constant/"
# the level files are deliberately NOT copied: with them the split hexes are legal refined hexes and
# nothing is protected (the same mesh at t = 0.002 reports 0), without them every cell reads level 0
# and the nine-faced ones stop being hexes
armIn splitHex "$SC" 0 constant/polyMesh
grep -q "^\[brae\] nProtectedCells 1886$" "$W/dump.splitHex.txt" \
    || { echo "FAIL: the splitHex fixture no longer protects 1886 cells"; rc=1; }

echo "refine_candidates_vs_openfoam: rc $rc"
exit $rc
