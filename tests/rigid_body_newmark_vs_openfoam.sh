#!/usr/bin/env bash
# brae's Newmark integrator and acceleration relaxation against REAL OpenFOAM's own joint state, on
# RAS/floatingObject -- the integrator half of unit 4 of the rigidBodyMotion port.
#
# WHAT IS UNDER TEST, and what is not. Given the state at the start of a step and the acceleration
# OpenFOAM's dynamics produced for it, reproduce the q and qDot OpenFOAM wrote. The acceleration
# itself -- the articulated-body algorithm that turns the fluid force into qDdot -- is the other half.
#
# THE INTEGRATOR (rigidBodySolvers/Newmark/Newmark.C:86-104):
#     qDot = qDot0 + dt*(gamma*qDdot + (1 - gamma)*qDdot0)
#     q    = q0 + dt*qDot0 + dt^2*(beta*qDdot + (0.5 - beta)*qDdot0)
# with gamma defaulting to 0.5 and beta = max(0.25*(gamma + 0.5)^2, the dictionary's beta) -- the
# entry is a FLOOR, not the value. Both updates read the acceleration at BOTH ends of the step, and
# `q0`/`qDot0`/`qDdot0` are the state at the START OF THE TIME STEP: with `moveMeshOuterCorrectors
# yes` this case solves three times per step, each from that same frozen state, so the three are an
# iteration and not three sub-steps.
#
# TWO ARMS, and the second exists because the first cannot tell the weights apart:
#   default   gamma 0.5, beta 0.25 -- at which gamma == 1 - gamma and beta == 0.5 - beta, so a port
#             that swapped the two acceleration weights would pass. MEASURED: q and qDot reproduced
#             EXACTLY, 0.0 on every step of the run.
#   gamma09   `gamma 0.9` added to the solver sub-dict, which makes beta = max(0.25*1.96, 0.25) = 0.49
#             through the clamp. The weights are then 0.9/0.1 and 0.49/0.01 and the swap control is
#             live. MEASURED: exact again, and the swapped-weights control 100% wrong.
#
# THE STAGING, and why the tutorial as shipped cannot be used: `accelerationRelaxation` is a table
# that is ZERO until t = 4, and with aRelax = 0 the relaxation returns the PREVIOUS acceleration --
# which starts at zero -- so the body does not move at all for the first four seconds. The gate
# applies the table's own final value, 0.7, from t = 0, writes every step, and runs ten steps of 5e-3.
#
# CONTROLS: the two acceleration weights swapped (live on the gamma09 arm only, and the binary says
# so on the other rather than passing quietly); the start-of-step acceleration dropped; and the body's
# own motion, so that none of the above is a comparison of zeros.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rigid_body_newmark_vs_openfoam"
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

grep -q "accelerationRelaxation table" "$SRC/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial no longer relaxes the acceleration by a table"; exit 1; }
grep -q "type Newmark" "$SRC/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial's solver is no longer Newmark"; exit 1; }

# stage <name> <gamma or empty>
stage()
{
    local name="$1" gamma="$2"
    local C="$W/$name"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.* "$C"/dynamicCode
    STEPS="$STEPS" DT="$DT" GAMMA="$gamma" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS']); dt = os.environ['DT']; gamma = os.environ['GAMMA']
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
# the table's own final relaxation, applied from t = 0, or the body never leaves rest
t, k = re.subn(r'accelerationRelaxation\s+table\s*\((?:[^()]|\([^()]*\))*\)\s*;',
               'accelerationRelaxation 0.7;', t, flags=re.S)
assert k == 1, 'accelerationRelaxation table not matched'
if gamma:
    t, k = re.subn(r'(type\s+Newmark;)', r'\1\n        gamma %s;' % gamma, t)
    assert k == 1, 'the Newmark solver entry was not found'
open(m, 'w').write(t)
PYEOF
    ( cd "$C" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 \
          && subsetMesh -overwrite c0 -patch floatingObject > log.subsetMesh 2>&1 \
          && setFields > log.setFields 2>&1 && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: running [$name]"; tail -30 "$C"/log.interFoam 2>/dev/null; return 1; }
    local n0
    n0=$(ls -d "$C"/[0-9]*/uniform/rigidBodyMotionState 2>/dev/null | wc -l)
    [ "$n0" -ge 5 ] || { echo "FAIL: only $n0 states written [$name]"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT, $n0 states written   [$name]"
}

rc=0
stage default "" || exit 1
"$BIN" "$W/default" 0.5 0.25 || rc=1

stage gamma09 "0.9" || exit 1
"$BIN" "$W/gamma09" 0.9 0.49 || rc=1

echo "rigid_body_newmark_vs_openfoam: rc $rc"
exit $rc
