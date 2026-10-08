#!/usr/bin/env bash
# brae's rigid-body point transform against REAL OpenFOAM's own pointDisplacement, on
# RAS/floatingObject -- unit 2 of the rigidBodyMotion port.
#
# WHAT IS UNDER TEST, and what is not. The DYNAMICS are OpenFOAM's: the joint state `q` is read from
# <time>/uniform/rigidBodyMotionState, so this arm holds the chain a `composite (Py Ry)` joint makes,
# X0(bodyID).inv() & X00(bodyID), the septernion slerp that weights it by the mesh-motion scale, and
# the point-patch constraint rigidBodyMeshMotion::solve ends in. The Newmark integrator, the body's
# inertia and the fluid force are the units after this one.
#
# THE CHAIN IS NOT ONE BODY. rigidBodyModel::join adds a massless jointBody for every joint of a
# composite but the last, so `composite (Py Ry)` is TWO links -- root through Py with the body's own
# `transform`, then that jointBody through Ry with the identity. Reading it as one body with two
# degrees of freedom gives a different X0 and a different mesh.
#
# THE STAGING, and why the tutorial as shipped cannot be used: its `accelerationRelaxation` table is
# ZERO until t = 4, so the body does not move for the first four seconds and a transform gate would be
# measuring the identity. It is set to the table's own final value, 0.7, from t = 0, and the case runs
# ten fixed steps of 5e-3. MEASURED at t = 0.05: q = (-6.458e-05 -1.288e-04), which is a translation
# along y and a rotation about it, both small and both non-zero.
#
# MEASURED, brae against OpenFOAM's pointDisplacement at writePrecision 18: EXACTLY OpenFOAM's,
# 0.0000e+00 on all 13,461 points, against a largest displacement of 9.799e-05.
# CONTROLS, each a reading of the same code that a careful person could have written:
#   the transform applied to EVERY point (the weight dropped)      1.496e+00 on 12,657 points
#   a LINEAR blend of the displacement instead of the slerp        6.493e-01 on 6,941 points
#   the rest state q = 0                                           moves the mesh by exactly 0
# ...and the constraint is load-bearing: without it OpenFOAM holds 291 of the tank's own wall points
# at zero where the blend moved them, and the arm reads 6.403e-05.
# FAIL-PROOF: the transform composed the other way round, X0 & inv(X00) instead of inv(X0) & X00,
# reads 2.275e-04 on 7,486 points with three arms red.
#
# THE REFUSAL ARM: the same case with the joint changed to `Rx`, which this unit does not carry. The
# reader must name it rather than run a chain it cannot build.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rigid_body_transform_vs_openfoam"
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

rc=0
"$BIN" "$C" "$C/$END" || rc=1

# THE REFUSAL ARM: a joint this unit does not carry must be named, not approximated.
R="$W/rx"
rm -rf "$R"
cp -r "$C" "$R"
sed -i '0,/type Py;/s//type Rx;/' "$R/constant/dynamicMeshDict"
grep -q "type Rx;" "$R/constant/dynamicMeshDict" || { echo "FAIL: the Rx edit did not take"; exit 1; }
if out=$("$BIN" "$R" "$R/$END" 2>&1); then
    echo "FAIL: the reader RAN a case whose joint is Rx"
    rc=1
else
    if printf '%s' "$out" | grep -q "the joint \`Rx\` is not ported"; then
        echo "  ok:   a joint this unit does not carry is refused BY NAME"
    else
        echo "FAIL: it failed on the Rx case without naming the joint:"
        printf '%s\n' "$out" | tail -3
        rc=1
    fi
fi

echo "rigid_body_transform_vs_openfoam: rc $rc"
exit $rc
