#!/usr/bin/env bash
# brae's interFoam on an ADAPTIVE mesh against REAL OpenFOAM's -- unit 8e, the last unit of the
# dynamicRefineFvMesh port and the first arm that runs the WHOLE chain through the solver:
# selection -> hexRef8 -> removeFaces -> changeMesh -> the six maps -> every field's autoMap -> the
# solver's own rebuild -> the next step's ddt terms.
#
# THE ORACLE is real OpenFOAM v2412's interFoam on laminar/damBreakWithObstacle, two fixed steps of
# deltaT 1e-3, written at `writePrecision 18` so the comparison floor is floating point and not the
# file. A refining case writes its own polyMesh into every time directory, so the comparison is on
# OpenFOAM's mesh, cell by cell -- and the mesh arms assert that numbering first, because a field
# difference on a differently-numbered mesh would be unreadable.
#
# WHAT THE TWO STEPS EXERCISE, measured from OpenFOAM's own log:
#   step 1   1,430 cells refined   32,256 -> 42,266 cells   pcorr: initial residual 0, 0 iterations
#   step 2   5,714 cells refined   42,266 -> 82,264 cells   pcorr: initial residual 1, 8 iterations
# So step 1 maps fields onto a refined mesh and needs no flux correction, and step 2 needs the
# correction -- 8 GAMG iterations of it. One step alone would have passed with the flux rebuild missing.
#
# WHAT IT MEASURED on 2026-09-27, every arm green. These are FLOATING-POINT FLOORS, not tolerances:
# on this case brae reproduces OpenFOAM's own arithmetic, and the bounds in the .cu sit just above them.
#   alpha    1.8848e-15 absolute on 82,264 cells      patch values: alpha 0.0e+00 (exact)
#   p_rgh    2.7285e-11 / 3.7065e-15 relative                       U     1.4268e-16
#   U        2.2053e-13 / 2.4051e-13 relative                       p_rgh 2.7285e-11
#   p        2.7626e-11 / 2.2797e-14 relative
#   rAU      1.5132e-15 / 1.5133e-12 relative
#   phi      1.3878e-17 / 2.6343e-13 relative
#   Uf       2.2738e-13 / 2.5089e-13 relative
#   the 6 p_rgh solves and the 3 pcorr solves: EVERY iteration count identical (1/3/7 2/5/5 and 0/0/8),
#   and every initial and final residual identical to the bit -- 0.0e+00 relative difference on all nine
#   sum local continuity error 8.138623e-09 against OpenFOAM's 8.138623e-09 (3.0e-11 relative)
# The residuals are the sharper statement of the two: a field at 1e-13 could still be a solver that
# stopped somewhere else, and an iteration count cannot be.
#
# TWO CONTROLS, one per half of the unit, each caught:
#   BRAE_CONTROL_AMR_NO_CORRECTPHI     the flux is left as the MAPPER wrote it -- no `phi = Sf & Uf` and
#                                      no pcorr solve. A split face inherits its parent's WHOLE flux
#                                      through a quarter of the area (Field::map copies, it does not
#                                      divide), so the mapped flux is not divergence-free and interFoam's
#                                      alpha equation has nothing to absorb that. MEASURED: alpha
#                                      2.8570e-01 out, U 3.2980e-01 relative, and max(alpha.water)
#                                      1.009594 where OpenFOAM stays at 1.
#   BRAE_CONTROL_AMR_RESIZE_NOT_MAP    the carried CELL fields are resized to the new count instead of
#                                      mapped -- the old values kept by index, the new cells at zero.
#                                      That is the defect class this unit exists to prevent. MEASURED:
#                                      alpha 3.6032e-02 out, U 6.0920e-02 relative.
#
# ...AND ONE ARM THAT CANNOT WITNESS, kept because the measurement is the point.
#   BRAE_CONTROL_AMR_NO_OLDTIME_MAP    re-captures the old-time levels from the fields after the change
#                                      instead of mapping them. It changes NOTHING here -- 0.0e+00 on
#                                      alpha and on U -- and the reason is structural: at the top of a
#                                      step every old-time level IS a copy of its own field (OpenFOAM's
#                                      storeOldTimes runs on the step's first access; brae's loop assigns
#                                      them at the end of the previous step), and the change happens
#                                      before anything has solved. So mapping them and re-capturing them
#                                      are the same numbers. The gate asserts they are identical to the
#                                      BIT, so this stays a statement about the fixture rather than an
#                                      arm that looks like a test of the old-time mapping. A case with
#                                      `moveMeshOuterCorrectors yes` is what would part them.
#
# WHAT THIS FIXTURE CANNOT WITNESS, and each is named rather than left to be assumed:
#   * THE FLUX CORRECTION. damBreakWithObstacle's dynamicMeshDict maps every flux to `none`
#     (`(phi none) (nHatf none) (rhoPhi none) (alphaPhi0.water none) (ghf none) (alphaPhiUn none)`), so
#     mapFields' own re-interpolation never runs. It is gated on an injected (braePhi braeU) pair in
#     tests/refine_update_vs_openfoam.sh, and a case that names a velocity is REFUSED by name here.
#   * UNREFINEMENT. The tutorial selects 0 split points at both steps -- and at every step of a 60-step
#     run of the same case at the tutorial's own resolution, measured. hexRef8's unrefinement path is
#     gated on its own in tests/hex_ref8_vs_openfoam.sh (unrefine, unrefine3, unrefineTwice) and the
#     driver's in tests/refine_update_vs_openfoam.sh, whose sphere moves so cells behind it coarsen.
#   * THE OLD-TIME LEVELS' MAPPING -- see the third control above, which measures that it cannot.
#   * A MOVING mesh beside refinement, turbulence, MRF, fvOptions, CrankNicolson, a pressure reference,
#     `correctPhi no` and a CN restart directory: each is REFUSED by name
#     (tests/interfoam_refusals.sh), so no arm here can silently stand in for one.
#   * THE DEVICE ARM, which refuses an adaptive case by name (`device_mesh_dynamic`).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_amr_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreakWithObstacle"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreakWithObstacle tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for app in blockMesh topoSet subsetMesh setFields interFoam; do
    command -v "$app" > /dev/null 2>&1 || { echo "SKIP: $app not on PATH"; exit 77; }
done

DT=0.001
N=2
END=$(python3 -c "print('%.10g' % ($N*float('$DT')))")

C="$W/case"
rm -rf "$C"
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/0 "$C"/processor* "$C"/log.*
cp -r "$C/0.orig" "$C/0"

# The case's own controlDict, with only what an oracle needs changed: fixed steps, every step written,
# and `writePrecision 18` -- at the tutorial's own 6 the comparison floor is the FILE. MEASURED: p_rgh
# reaches 7.3e+03 on this case, so six significant digits resolve it to 5e-03, and every field came out
# exactly at that floor before this line went in.
python3 - "$C/system/controlDict" "$DT" "$END" <<'PYEOF'
import re, sys
path, dt, end = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('stopAt', 'endTime'),
                 ('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', end),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '18'), ('writeCompression', 'off'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(path, 'w').write(s)
PYEOF
[ $? -eq 0 ] || { echo "FAIL: rewriting controlDict"; exit 1; }

grep -q "dynamicFvMesh   dynamicRefineFvMesh" "$C/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial no longer selects dynamicRefineFvMesh, so this gate tests nothing"; exit 1; }

key=$(oracleKey "$C" "interfoam_amr" "$DT" "$N")
if oracleRestore "$C" "$key" "$END"; then
    echo "OpenFOAM's $N steps of deltaT $DT to t = $END reused from the oracle cache"
else
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 \
              && topoSet > log.topoSet 2>&1 \
              && subsetMesh -overwrite c0 -patch walls > log.subsetMesh 2>&1 \
              && setFields > log.setFields 2>&1 ) \
        || { echo "FAIL: preparing the case"; tail -20 "$C"/log.*; exit 1; }
    grep -q "Subset 32256 of 32768 cells" "$C/log.subsetMesh" \
        || { echo "FAIL: damBreakWithObstacle is no longer 32256 cells, so the counts above are stale"; exit 1; }
    [ ! -e "$C/constant/polyMesh/cellLevel" ] \
        || { echo "FAIL: the fresh mesh already carries a cellLevel, so it does not start from level 0"; exit 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; exit 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory"; ls "$C"; exit 1; }
    oracleStore "$C" "$key"
fi

# THE FIXTURE MUST BE ABLE TO WITNESS, asserted on OpenFOAM's own log rather than assumed: two
# refinements, at these counts, and a pcorr solve that does real work at the second one.
grep -q "Refined from 32256 to 42266 cells." "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM's first refinement is not 32256 -> 42266, so the measurements above are stale"
         grep -n "Refined from" "$C/log.interFoam"; exit 1; }
grep -q "Refined from 42266 to 82264 cells." "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM's second refinement is not 42266 -> 82264"
         grep -n "Refined from" "$C/log.interFoam"; exit 1; }
grep -q "Solving for pcorr, Initial residual = 1," "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM's pcorr solve does no work on this fixture, so the flux correction cannot"
         echo "      be witnessed and the control below would be vacuous"
         grep -n "pcorr" "$C/log.interFoam"; exit 1; }
[ -f "$C/$END/polyMesh/owner" ] \
    || { echo "FAIL: OpenFOAM wrote no polyMesh into $END, so there is no mesh to compare on"; exit 1; }
[ -f "$C/$END/Uf" ] \
    || { echo "FAIL: OpenFOAM wrote no Uf, so a refining mesh is not dynamic after all"; exit 1; }

rc=0
"$BIN" "$C" "$C/0" "$C/$END" "$N" "$C/log.interFoam" || rc=1
echo "interfoam_amr_vs_openfoam: rc $rc"
exit $rc
