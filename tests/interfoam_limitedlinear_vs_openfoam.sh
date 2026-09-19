#!/usr/bin/env bash
# brae's interFoam with `Gauss limitedLinear 0.2` on div(rhoPhi,U) against REAL OpenFOAM's, on
# laminar/vofToLagrangian/eulerianInjection, field by field and solve by solve.
#
# WHAT IS PORTED: limitedLinear on a vector is NVDTVD + limitFuncs::magSqr (LimitedScheme.H:188-189,
# LimitFuncs.C:34-39) -- ONE scalar limiter per face, built on magSqr(U) and grad(magSqr(U)), carrying all
# three components. Not per component and not the V form.
#
# MEASURED, 60 steps of 2e-5 on the 45^3 block (90627 cells), brae host against OpenFOAM:
#   alpha Linf 9.2e-14, p_rgh 8.8e-12, U 2.7e-12; all 180 p_rgh solves take OpenFOAM's iteration count.
# CONTROLS, OpenFOAM against itself at the same instant:
#   `Gauss upwind` 8.0e-01, `Gauss limitedLinear 1` 7.8e-01.
# FAIL-PROOFS, each decision broken once in a scratch copy, the gate red every time:
#   the V form's weights               U 8.5e-01
#   magSqr's patch values from cells   U 4.3e-01
#   the coefficient ignored (k = 1)    U 7.8e-01   (lands on OpenFOAM's own control)
#   upwind in place of the scheme      U 8.0e-01   (lands on OpenFOAM's own control)
#   the limiter on mag(U), not magSqr  U 8.2e-01
#
# WHY 60 STEPS: at 100 the gap is U 1.2e-08. It opens at step 68 in one near-air cell at the jet edge
# (alpha 3.4e-4), where the density ratio of 1000 turns alpha's 4e-13 into a rho difference of 3e-10.
# brae against brae with only the face-interpolation arithmetic reordered reads U 2.9e-09 at 100 steps,
# so the growth is the case's conditioning, not the scheme; 60 steps measures the scheme before it.
#
# THE DEVICE LOOP REFUSES the scheme: its limitedLinear branch reads U 5.1e-01 against OpenFOAM on a
# DIC-smoother twin of this case at 50 steps, where the host reads 3.9e-12 and the device under upwind
# 1.1e-13. That is an open device finding, not ported here.
#
# NOT CLAIMED: the tutorial's own 75^3 mesh (420k cells, a bench size), its Lagrangian function object
# (stripped, brae runs none), and limitedLinear across a coupled patch (refused).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_limitedlinear_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/vofToLagrangian/eulerianInjection"
STEPS=${STEPS:-60}
DT=${DT:-2e-5}
# the tutorial's 75^3 block is 420k cells after subsetMesh -- a bench size, not a validation one; 45^3 is
# 90627 cells with the injector still about 2.7 cells across and the blockage still two cells thick
NCELL=${NCELL:-45}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: eulerianInjection tutorial not found at $SRC"; exit 77; }
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

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the tutorial, fix the step, mesh it as Allrun does (serial), apply the profile, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    grep -q "div(rhoPhi,U)  *Gauss limitedLinear 0.2;" "$C/system/fvSchemes" \
        || { echo "FAIL: the tutorial no longer names Gauss limitedLinear 0.2 on div(rhoPhi,U)"; return 1; }
    grep -q "(75 75 75)" "$C/system/blockMeshDict" || { echo "FAIL: the tutorial's block is no longer 75^3"; return 1; }
    sed -i "s/(75 75 75)/($NCELL $NCELL $NCELL)/" "$C/system/blockMeshDict"
    case "$profile" in
        upwind) sed -i 's/Gauss limitedLinear 0.2;/Gauss upwind;/' "$C/system/fvSchemes" ;;
        ll1)    sed -i 's/Gauss limitedLinear 0.2;/Gauss limitedLinear 1;/' "$C/system/fvSchemes" ;;
    esac

    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
# extractEulerianParticles writes nothing a field comparison reads, and brae runs no function objects
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 \
          && topoSet -dict system/topoSetDict.createBlockage > log.topoSet.blockage 2>&1 \
          && subsetMesh -overwrite blockage -patch bottom > log.subsetMesh 2>&1 \
          && topoSet -dict system/topoSetDict.createPatch > log.topoSet.patch 2>&1 \
          && topoSet -dict system/topoSetDict.createCollector > log.topoSet.collector 2>&1 \
          && cp -r 0.orig 0 && setFields > log.setFields 2>&1 ) \
        || { echo "FAIL: meshing [$profile]"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in upwind ll1 ll; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_limitedlinear_vs_openfoam: staging failed"; exit 1; }

"$BIN" "$W/ll" "$W/ll/0" "$W/ll/$END" "$STEPS" "$W/ll/log.interFoam" "$W/upwind/$END" "$W/ll1/$END" || rc=1

echo "interfoam_limitedlinear_vs_openfoam: rc $rc"
exit $rc
