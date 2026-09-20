#!/usr/bin/env bash
# brae interFoam on a FULLY PERIODIC mesh against REAL OpenFOAM's, on validation/interFoamCyclic.
#
# THE CASE is brae's own, not a tutorial: 800 cells, x cyclic, y walled, z one cell thick, with the
# water surface staged as a STEP at the periodic boundary (y = 0.3 on one side, 0.15 on the other), so
# alpha, its gradient, nHatf and the pressure all carry a jump through the pair from the first step and
# gravity drives flow across it for the whole run. The baffle gate
# (tests/interfoam_baffle_vs_openfoam.sh) already couples an INTERNAL pair; what is new here is a pair
# on the DOMAIN boundary with no wall behind it, and a case small enough to run the device arm on.
#
# THE ORACLE is OpenFOAM's own fields after ten FIXED steps of 2e-3, written ASCII at 17 digits.
# THE CONTROL is OpenFOAM's own answer with the pair replaced by two WALLS: same block, same cells.
#
# WHAT IT ASSERTS: brae's host loop is OpenFOAM's on this mesh; OpenFOAM moves fluid through the pair,
# so the comparison is not vacuous; walling the pair is a different answer by orders of magnitude; and
# the DEVICE loop refuses the pair by name, because its pressure corrector carries neither phig nor its
# share of fvc::reconstruct there.
#
# MEASURED, ten steps of 2e-3: alpha 1.9e-13, p_rgh 9.8e-12 relative of 1.7e+03, U 7.5e-13 relative of
# 0.73, all 30 p_rgh iteration counts OpenFOAM's. CONTROL: OpenFOAM with the pair two walls, alpha
# 2.1e-01 and U 100% away -- and its step one runs 64/50/1 iterations where the periodic case runs
# 70/32/4, so the iteration-count arm sees the coupling too.
#
# BROKEN ONCE (at t = 0.02): the pair's laplacian deltaCoeffs 0.1% out -- alpha 2.1e-03, p_rgh 3.6e-03,
# U 7.9e-03, and 2 of 30 iteration counts lost. NOT MEASURABLE, because brae refuses it outright: the
# pair left uncoupled, which every other cyclic gate calls its control -- the host driver throws and
# names the patch rather than running it as two walls.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_cyclic_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC="$ROOT/validation/interFoamCyclic"
STEPS=${STEPS:-10}
DT=${DT:-0.002}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: $SRC not found"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v setFields > /dev/null 2>&1 || { echo "SKIP: setFields not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the fixture, apply the profile, mesh it, set the fields, run OpenFOAM
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[0-9]* "$C"/processor* "$C"/log.* "$C"/constant/polyMesh
    cp -r "$SRC/0.orig" "$C/0.orig"
    cp -r "$C/0.orig" "$C/0"
    grep -q "type cyclic; neighbourPatch right;" "$C/system/blockMeshDict" \
        || { echo "FAIL: the fixture's left patch is no longer the cyclic this gate stages"; return 1; }
    if [ "$profile" = walls ]; then
        # THE CONTROL: the pair replaced by two walls, in the mesh AND in every field that names it.
        # blockMesh numbers the cells from the block, so the two runs' cells are the same cells.
        sed -i 's/type cyclic; neighbourPatch right;/type wall;/; s/type cyclic; neighbourPatch left; */type wall;/' \
            "$C/system/blockMeshDict"
        grep -q "type cyclic" "$C/system/blockMeshDict" \
            && { echo "FAIL: the control's mesh still names a cyclic"; return 1; }
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type noSlip; }/' "$C/0/U"
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type zeroGradient; }/' "$C/0/alpha.water"
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type fixedFluxPressure; value uniform 0; }/' \
            "$C/0/p_rgh"
        grep -q "type cyclic" "$C/0/U" "$C/0/alpha.water" "$C/0/p_rgh" \
            && { echo "FAIL: the control's fields still name a cyclic"; return 1; }
    fi
    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '17')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) \
        || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

stage cyclic || { echo "interfoam_cyclic_vs_openfoam: staging failed"; exit 1; }
stage walls  || { echo "interfoam_cyclic_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path: OpenFOAM's own mesh carries the pair on one arm and not on the other
grep -q "type  *cyclic" "$W/cyclic/constant/polyMesh/boundary" \
    || { echo "FAIL: OpenFOAM's mesh has no cyclic patch"; exit 1; }
grep -q "type  *cyclic" "$W/walls/constant/polyMesh/boundary" \
    && { echo "FAIL: the control's mesh still has a cyclic patch"; exit 1; }

"$BIN" "$W/cyclic" "$W/cyclic/0" "$W/cyclic/$END" "$STEPS" "$W/cyclic/log.interFoam" "$W/walls/$END"
rc=$?
echo "interfoam_cyclic_vs_openfoam: rc $rc"
exit $rc
