#!/usr/bin/env bash
# brae's fluid force and moment on a body's patches against REAL OpenFOAM's own
# functionObjects::forces, on RAS/floatingObject -- unit 3 of the rigidBodyMotion port.
#
# WHAT IT IS FOR. rigidBodyMeshMotion::solve builds `spatialVector(f.momentEff(), f.forceEff())` from a
# forces function object it constructs itself (rigidBodyMeshMotion.C:296-307: type forces, the body's
# patches, rhoInf, rho, CofR (0 0 0)) and hands it to the body's dynamics. Those two vectors are the
# whole interface between the fluid and the body, so they are what this gate holds.
#
# THE ORACLE is tools/dumpBodyForces: OpenFOAM's own forces with that dictionary, writing forceEff and
# momentEff, the pressure/viscous split (postProcessing/forces/0/force.dat) and -- through forces' own
# `writeFields` -- the per-face `force` and `moment` fields.
#
# THREE THINGS THIS SHAPE DOES NOT SHARE with the single-phase wallForces already in the tree, each
# found by measuring rather than reading:
#   * the pressure is REAL. forces::rho(const volScalarField&) returns 1 whenever p carries pressure
#     dimensions, so fP is Sf*(p - pRef) with no density in it.
#   * the viscous stress carries the PER-FACE density and effective viscosity, not one rhoInf. Reading
#     alpha.water's patch `value` on the body gives ZERO -- its entry there is `zeroGradient` with
#     nothing written under it -- which is air where there is water and a viscous force a thousand
#     times too small.
#   * U's patch value on a movingWallVelocity wall is the WALL's velocity. brae's class seeds it at
#     zero without a mesh motion to compute it from, and that zero took grad(U)'s wall-normal row 13%
#     out and the viscous force with it: forceEff read (-1.4698 -1.2434 257.1153) against OpenFOAM's
#     (-1.4660 -1.2334 257.1153).
#
# MEASURED, with all three right, at t = 0.05 of the ten staged steps:
#   force  (-1.46600542572389 -1.23337807571694 257.115317322973) against OpenFOAM's
#          (-1.466005425723887 -1.233378075716942 2.571153173229728e+02)
#   the pressure and viscous halves and the moment likewise, and the per-face force to 4.8e-14 of 10.7.
# CONTROLS: rhoInf = 1 in place of the per-face density reads the viscous half 9.987e-01 wrong; the
# viscous half dropped altogether moves the force by 3.6e-05 of the whole -- the body is BUOYANT, so
# the total is 257 N of pressure in z -- and by 9.1e-04 of its x component, which is the direction the
# Py joint leaves it free in. Both are stated at their measured size.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rigid_body_forces_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/floatingObject"
STEPS=${STEPS:-10}
DT=${DT:-5e-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: floatingObject tutorial not found at $SRC"; exit 77; }
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
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.* "$C"/dynamicCode

grep -q "accelerationRelaxation table" "$C/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial no longer relaxes the acceleration by a table"; exit 1; }
grep -q "type            composite" "$C/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial's joint is no longer a composite"; exit 1; }

STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging"; exit 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS']); dt = os.environ['DT']
end = 0.0
for _ in range(n):
    end += float(dt)
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('adjustTimeStep', 'no'),
                 ('deltaT', dt), ('endTime', '%.10g' % end), ('writeControl', 'timeStep'),
                 ('writeInterval', str(n)), ('writeFormat', 'ascii'), ('writePrecision', '18'),
                 ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
# the table's own final relaxation, applied from t = 0 so the body moves inside the gate's ten steps
m = os.path.join(d, 'constant/dynamicMeshDict')
t = open(m).read()
t, k = re.subn(r'accelerationRelaxation table\s*\((?:[^()]|\([^()]*\))*\);',
               'accelerationRelaxation 0.7;', t, flags=re.S)
assert k == 1, 'accelerationRelaxation table not matched'
open(m, 'w').write(t)
PYEOF

END=$(python3 -c "
t = 0.0
for i in range($STEPS):
    t += float('$DT')
print('%.10g' % t)")

( cd "$C" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 \
      && subsetMesh -overwrite c0 -patch floatingObject > log.subsetMesh 2>&1 \
      && setFields > log.setFields 2>&1 ) \
    || { echo "FAIL: meshing"; tail -20 "$C"/log.* 2>/dev/null; exit 1; }
( cd "$C" && interFoam > log.interFoam 2>&1 ) \
    || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; exit 1; }
[ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory"; ls "$C"; exit 1; }
echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END"

# the oracle: OpenFOAM's own forces, with the dictionary rigidBodyMeshMotion builds
command -v dumpBodyForces > /dev/null 2>&1 || { echo "SKIP: dumpBodyForces not on PATH"; exit 77; }
( cd "$C" && dumpBodyForces -time "$END" -patches '(floatingObject)' > log.dumpBodyForces 2>&1 ) \
    || { echo "FAIL: dumpBodyForces"; tail -20 "$C/log.dumpBodyForces"; exit 1; }
grep -q "\[brae\] forceEff" "$C/log.dumpBodyForces" \
    || { echo "FAIL: the oracle printed no forceEff"; exit 1; }

rc=0
"$BIN" "$C" "$C/$END" floatingObject || rc=1

echo "rigid_body_forces_vs_openfoam: rc $rc"
exit $rc
