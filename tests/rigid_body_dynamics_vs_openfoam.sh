#!/usr/bin/env bash
# brae's articulated-body dynamics against REAL OpenFOAM's own RBD::rigidBodyMotion::solve, on
# RAS/floatingObject -- unit 4b, the last piece of the rigidBodyMotion port.
#
# WHAT IS UNDER TEST: forwardDynamics, the articulated-body algorithm that turns one spatial force on
# the body into the joint accelerations of the chain, together with the relaxation and the Newmark
# update that the previous unit gated on their own. Three passes over the chain
# (rigidBodyModel::forwardDynamics, rigidBodyModel.C:456-528): joint transforms and velocities down,
# articulated inertias and bias forces up, accelerations down again.
#
# THE ORACLE is tools/dumpRigidBodySolve: OpenFOAM's OWN rigidBodyMotion, built from the case's own
# dictionary, seeded with a state read off disk, handed a spatial force on the command line, and asked
# to solve. Comparing against it isolates the dynamics from the fluid and from the mesh -- which an
# end-to-end run cannot, because with `nOuterCorrectors 3` and `moveMeshOuterCorrectors yes` this case
# solves the body THREE times per step from the same frozen state, and the force of correctors 2 and 3
# was evaluated on mid-step fields that are never written. MEASURED (an independent transcription of
# the algorithm, run against the solver's own `report on` block): the force `dumpBodyForces` prints at
# time t, paired with the state written at t, reproduces corrector ONE of the next step BITWISE, while
# the written end-of-step state is off by up to 100% on the first step. So the written state is NOT a
# usable oracle for the dynamics, and this gate does not use it.
#
# THE FORCE is the fluid load from tools/dumpBodyForces, which unit 3 gated against OpenFOAM's own
# functionObjects::forces to a relative 0.
#
# TWO ARMS, because the tutorial's own ten steps barely move the body (q1 reaches 1e-4 rad, so the Ry
# rotation is the identity to twelve digits and gravity's moment about the joint axis is zero):
#   asRun    the states the run actually wrote, at three times.
#   rotated  the same force applied to a state overridden to q = (0.05, 0.3), qDot = (0.02, -0.1),
#            qDdot = (-0.05, 0.12). A 0.3 rad rotation makes Xry live, the velocities make the
#            `v ^* (I & v)` bias force live, and qDdot0 makes the relaxation and both start-of-step
#            Newmark weights live. None of those is exercised by the run as it stands.
#
# CONTROLS, in the binary: gravity dropped, the fluid force dropped, and the moment and the force
# exchanged between the two halves of the spatial vector -- the one mistake the (angular, linear)
# ordering invites, since both halves are three-vectors of plausible size.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rigid_body_dynamics_vs_openfoam"
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

for t in blockMesh topoSet subsetMesh setFields interFoam dumpBodyForces dumpRigidBodySolve; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

grep -q "type Newmark" "$SRC/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial's solver is no longer Newmark"; exit 1; }
grep -q "composite" "$SRC/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial's joint is no longer a composite, so the chain is not two links"; exit 1; }

C="$W/case"
rm -rf "$C"
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.* "$C"/dynamicCode
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
                 ('writeInterval', '1'), ('writeFormat', 'ascii'), ('writePrecision', '18'),
                 ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
m = os.path.join(d, 'constant/dynamicMeshDict')
t = open(m).read()
# the table's own final relaxation, applied from t = 0: the shipped table is ZERO until t = 4, so the
# body would not move at all over the ten steps this gate runs
t, k = re.subn(r'accelerationRelaxation\s+table\s*\((?:[^()]|\([^()]*\))*\)\s*;',
               'accelerationRelaxation 0.7;', t, flags=re.S)
assert k == 1, 'accelerationRelaxation table not matched'
open(m, 'w').write(t)
PYEOF

( cd "$C" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 \
      && subsetMesh -overwrite c0 -patch floatingObject > log.subsetMesh 2>&1 \
      && setFields > log.setFields 2>&1 && interFoam > log.interFoam 2>&1 ) \
    || { echo "FAIL: running the case"; tail -30 "$C"/log.interFoam 2>/dev/null; exit 1; }
echo "OpenFOAM ran $STEPS steps of deltaT $DT"

# solve <label> <time> <nextTime> [state overrides...]
rc=0
solve()
{
    local label="$1" t="$2" t2="$3"
    shift 3
    local fl="$W/force.$label" ol="$W/oracle.$label"
    dumpBodyForces -case "$C" -time "$t" -patches '(floatingObject)' > "$fl" 2>&1 \
        || { echo "FAIL: dumpBodyForces at $t"; tail -20 "$fl"; rc=1; return; }
    local f m
    f=$(awk '/\[brae\] forceEff/  {printf "(%s %s %s)", $3, $4, $5}' "$fl")
    m=$(awk '/\[brae\] momentEff/ {printf "(%s %s %s)", $3, $4, $5}' "$fl")
    [ -n "$f" ] && [ -n "$m" ] || { echo "FAIL: no force printed at $t"; tail -20 "$fl"; rc=1; return; }
    dumpRigidBodySolve -case "$C" -time "$t" -newTime -t "$t2" -deltaT "$DT" \
        -moment "$m" -force "$f" "$@" > "$ol" 2>&1 \
        || { echo "FAIL: dumpRigidBodySolve at $t"; tail -20 "$ol"; rc=1; return; }
    echo "--- $label: state at $t, force $f, solved to $t2"
    "$BIN" "$C" "$ol" || rc=1
}

solve asRun.0.005 0.005 0.01
solve asRun.0.025 0.025 0.03
solve asRun.0.05  0.05  0.055

# the same force on a state the run never reaches: a real rotation, real velocities, a real qDdot0
solve rotated 0.05 0.055 -q '(0.05 0.3)' -qDot '(0.02 -0.1)' -qDdot '(-0.05 0.12)' \
      -t0 0.05 -deltaT0 "$DT"

# --- DTCHullMoving: the body model a SHIPPED top-level dictionary names ---------------------------
# The same oracle on RAS/DTCHullMoving's own constant/dynamicMeshDict, which is everything
# floatingObject is not: the coefficients at the TOP LEVEL beside `motionSolver` (optionalSubDict's
# other branch, with the integrator's `solver { type Newmark; }` sitting where the legacy name would),
# a `rigidBody` body with its own `inertia` about the centre of mass, a composite of Pz and Ry, and
# two RESTRAINTS -- a linearDamper and a sphericalAngularDamper -- which Newmark::solve adds to fx
# before the dynamics. The dynamics does not need the fluid, so no mesh is built: the hull load is
# handed over on the command line at the size a floating hull carries (its weight, 412.73*9.81, is
# 4049 N).
#   rest     q = qDot = 0: the dampers read a zero velocity and must add EXACTLY nothing.
#   moving   heave 0.02 m and pitch -0.03 rad, both moving: the dampers are live, the pitch makes Xry
#            live, and the body's offset (2.93, 0, 0.2) makes the X0.T() transport of the damper force
#            carry a moment.
DSRC="$TUT/multiphase/interFoam/RAS/DTCHullMoving"
if [ -d "$DSRC" ]; then
    grep -q "^motionSolver *rigidBodyMotion" "$DSRC/constant/dynamicMeshDict" \
        || { echo "FAIL: DTCHullMoving no longer names motionSolver rigidBodyMotion"; exit 1; }
    grep -q "rigidBodyMotionCoeffs" "$DSRC/constant/dynamicMeshDict" \
        && { echo "FAIL: DTCHullMoving's coefficients are no longer at the top level"; exit 1; }
    grep -q "linearDamper" "$DSRC/constant/dynamicMeshDict" \
        || { echo "FAIL: DTCHullMoving no longer names a linearDamper"; exit 1; }
    D="$W/dtchm"
    rm -rf "$D"
    mkdir -p "$D/constant" "$D/system" "$D/0"
    cp "$DSRC/constant/dynamicMeshDict" "$DSRC/constant/g" "$D/constant/" || exit 1
    cat > "$D/system/controlDict" <<'CDEOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application interFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1e-4;
writeControl timeStep;
writeInterval 1;
CDEOF
    DDT=1e-4
    dsolve()
    {
        local label="$1"
        shift
        local ol="$W/oracle.$label"
        dumpRigidBodySolve -case "$D" -time 0 -newTime -deltaT "$DDT" "$@" > "$ol" 2>&1 \
            || { echo "FAIL: dumpRigidBodySolve on DTCHullMoving ($label)"; tail -20 "$ol"; rc=1; return; }
        echo "--- DTCHullMoving $label"
        "$BIN" "$D" "$ol" || rc=1
    }
    dsolve rest -q '(0 0)' -qDot '(0 0)' -qDdot '(0 0)' -t0 0 -deltaT0 "$DDT" -t 1e-4 \
        -force '(-12.5 3.25 4100.75)' -moment '(1.5 -85.25 0.75)'
    dsolve moving -q '(0.02 -0.03)' -qDot '(0.1 -0.2)' -qDdot '(0.3 0.5)' -t0 0.5 -deltaT0 "$DDT" \
        -t 0.5001 -force '(-12.5 3.25 4100.75)' -moment '(1.5 -85.25 0.75)'
else
    echo "SKIP (arm): DTCHullMoving tutorial not found at $DSRC"
fi

echo "rigid_body_dynamics_vs_openfoam: rc $rc"
exit $rc
