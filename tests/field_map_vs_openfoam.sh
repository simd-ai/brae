#!/usr/bin/env bash
# brae's cell mapper against REAL OpenFOAM's own, through one refinement step and one unrefinement step
# of laminar/damBreakWithObstacle -- unit 5 of the dynamicRefineFvMesh port.
#
# WHAT IS UNDER TEST: how a field's CELL VALUES move when the mesh changes under it. This is the one
# part of adaptive refinement that decides whether the answer is right, and it is gated BEFORE the
# machinery that produces the mesh change exists, because both its input (the map) and its output (the
# mapped values) can be read out of OpenFOAM.
#
# THE TWO STEPS TAKE DIFFERENT BRANCHES, and that is the whole point:
#   refine    24,185 -> 24,815 cells. Every cellsFrom*Map is empty, so cellMapper is DIRECT: a pure
#             gather, `f[i] = old[cellMap[i]]`, no arithmetic. Each of eight children takes its
#             parent's value verbatim -- piecewise constant, NOT conservative. Expect exactly 0.
#   unrefine  24,815 -> 24,444 cells, 53 merges of eight. cellsFromCellsMap is non-empty, so the WHOLE
#             field goes interpolative: an untouched cell maps from itself with weight 1, a merged cell
#             from its masters with VOLUME weights taken from the map's own oldCellVolumes.
# A port that used one branch everywhere would look plausible and be wrong in one direction only, so
# the gate checks that brae reaches OPENFOAM'S OWN verdict on the branch, not merely its numbers.
#
# THE ORACLE is tools/dumpRefineMap. `dynamicRefineFvMesh::update()` does not hand back the map, and
# `updateTopology` DISCARDS the unrefine one -- so the tool derives a mesh, overrides the VIRTUAL
# protected `refine`/`unrefine` with a body that calls the base implementation and nothing else, and
# reads the map out of the autoPtr it returns. It then builds the very `cellMapper` that
# `fvMesh::mapFields` uses. No OpenFOAM file is edited and no arithmetic changes.
#
# THE WRITTEN <time> FIELDS CANNOT SERVE AS THE ORACLE, measured rather than assumed: the solver
# advances the fields after `update()` returns, so the post-map state and the written state differ on
# 13,774 of 24,815 cells by up to 3.12e-01. Every value compared here is a post-MAP value.
#
# THE ONE-STEP OFFSET is real: the state written at <t> is what the solver held when it called
# update() for the step that FOLLOWS <t>. So `-time 0.002` measures the change the log reports under
# `Time = 0.003`, and `-time 0.003` the one under `Time = 0.004`.
#
# CONTROLS, in the binary: the identity map instead of cellMap (without it the refinement arm would
# pass on a port that never read the map, because the mesh is mostly cells it does not move and the
# eight children of an interface cell all take the same parent value); and uniform 1/n weights instead
# of volume weights, live on the interpolative arm only -- the volume weights of eight equal children
# are 1/8 to within an ulp, printing as 0.125000000000000083, so hardcoding 1.0/8.0 is one ulp per
# child out.
#
# THE FLUX HALF runs here too, in test_flux_map_vs_openfoam, on the same two dumps: THREE correction
# sites, not one -- mapFields' four write sites (:257-422), mapNewInternalFaces on the ORIENTED branch
# (:424-437), and the site only unrefinement has (:610-689). MEASURED: the first overwrites 2,156 faces
# on the refine step and ZERO on the unrefine step, where the third overwrites 158. A port that finds
# only the first is green on every refinement and silently wrong on every unrefinement.
#
# NOT COVERED, each stated rather than implied -- a green run here does NOT mean "mapping works":
#   * the FACE MAPPER itself. Its direct() predicate and addressing build are their own unit; step one
#     is taken from OpenFOAM's own addressing in the dump, and the boundary half of step one goes
#     through faceMap rather than fvPatchMapper, which the oracle does not dump. Enough on both arms
#     here; not enough on a mesh where a patch's own faces are split.
#   * `phi[facei] = phiU[facei]` at :362 for an INJECTED internal face. The oriented round trip's final
#     whole-field assignment overwrites it, so its value is provably discarded and no oracle can witness
#     whether a port wrote it. Median relative difference from the final value: 9.6e-01.
#   * the `NaN` branch (:329-336) and the absent-entry warning (:312-324), unreachable on this fixture.
#   * old-time fields, surface fields other than phi and Uf, and anything parallel.
#   * old-time fields, the boundary half, and anything parallel.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_field_map_vs_openfoam"
FLUXBIN="${BUILD:-$ROOT/build}/test_flux_map_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreakWithObstacle"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -x "$FLUXBIN" ]  || { echo "SKIP: $FLUXBIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreakWithObstacle tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

for t in blockMesh topoSet subsetMesh setFields interFoam dumpRefineMap; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

C="$W/case"
rm -rf "$C"
cp -r "$SRC" "$C" || exit 1
chmod -R u+w "$C"
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
sed -i 's/(32 32 32)/(16 16 16)/' "$C/system/blockMeshDict"
python3 - "$C" <<'PYEOF' || { echo "FAIL: staging"; exit 1; }
import re, sys
d = sys.argv[1]
c = d + '/system/controlDict'
s = open(c).read()
for key, val in [('endTime', '0.004'), ('deltaT', '0.001'), ('adjustTimeStep', 'no'),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'),
                 ('writeFormat', 'ascii'), ('writePrecision', '18'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
# the water starts moving, so the interface sweeps out of cells and the case UNREFINES within four
# steps; without it a just-refined octet can never unsplit, because one of its children always still
# carries the interface and vetoes the point
f = d + '/system/setFieldsDict'
t = open(f).read()
t, k = re.subn(r'(volScalarFieldValue alpha\.water 1)',
               r'\1\n                volVectorFieldValue U (2 0 0)', t)
assert k == 1, 'the setFields water block was not found'
open(f, 'w').write(t)
# ...and the FLUX correction's interpolating branch, which is unreachable as the tutorial ships: all
# four interFoam AMR tutorials set every correctFluxes entry to `none`, so `phi` is never recreated on
# a new face. Naming U instead makes the branch live. MEASURED: the refine and unrefine cell counts are
# unchanged by it, which the two asserts below check.
m = d + '/constant/dynamicMeshDict'
u = open(m).read()
assert '(phi none)' in u, 'the tutorial no longer says `(phi none)`'
u = u.replace('(phi none)', '(phi U)', 1)
open(m, 'w').write(u)
PYEOF

( cd "$C" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 \
      && subsetMesh -overwrite c0 -patch walls > log.subsetMesh 2>&1 \
      && setFields > log.setFields 2>&1 && interFoam > log.interFoam 2>&1 ) \
    || { echo "FAIL: running the case"; tail -30 "$C"/log.interFoam 2>/dev/null; exit 1; }

# ...AND IT BOTH REFINED AND UNREFINED. Either half missing makes an arm below a comparison on one
# mesh, and the gate would be green and vacuous.
grep -q "^Refined from 24185 to 24815 cells.$" "$C/log.interFoam" \
    || { echo "FAIL: the fixture no longer refines 24185 -> 24815"; exit 1; }
grep -q "^Unrefined from 24815 to 24444 cells.$" "$C/log.interFoam" \
    || { echo "FAIL: the fixture no longer unrefines 24815 -> 24444"; exit 1; }
echo "OpenFOAM refined 24185 -> 24815 and unrefined 24815 -> 24444"

rc=0
# arm <name> <time> <expected phase>
arm()
{
    local name="$1" t="$2" phase="$3"
    local dump="$W/map.$name.txt"
    ( cd "$C" && dumpRefineMap -case "$C" -time "$t" > "$dump" 2>&1 ) \
        || { echo "FAIL: dumpRefineMap [$name]"; tail -20 "$dump"; rc=1; return; }
    grep -q "^\[brae\] map 0 phase $phase$" "$dump" \
        || { echo "FAIL: [$name] the oracle reports no `$phase` phase at t = $t"; \
             grep -m2 "phase" "$dump"; rc=1; return; }
    echo "--- $name (t = $t, $phase)"
    "$BIN" "$dump" "$phase" || rc=1
    "$FLUXBIN" "$dump" "$phase" || rc=1
}

arm refine   0.002 refine
arm unrefine 0.003 unrefine

echo "field_map_vs_openfoam: rc $rc"
exit $rc
