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
# arm <name> <time> <mesh dir> [extra oracle options...]
arm()
{
    local name="$1" t="$2" meshDir="$3"
    shift 3
    local dump="$W/dump.$name.txt"
    ( cd "$C" && dumpRefineCandidates -case "$C" -time "$t" "$@" > "$dump" 2>&1 ) \
        || { echo "FAIL: dumpRefineCandidates [$name]"; tail -20 "$dump"; rc=1; return; }
    echo "--- $name (t = $t)${*:+ [$*]}"
    "$BIN" "$C" "$C/$meshDir" "$C/$t" "$dump" || rc=1
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

echo "refine_candidates_vs_openfoam: rc $rc"
exit $rc
