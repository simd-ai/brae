#!/usr/bin/env bash
# angledDuctExplicitFixedCoeff: OpenFOAM's OWN rhoSimpleFoam tutorial, unmodified, through brae's _cpp
# driver.
#
# WHY THIS ONE. It is the only rhoSimpleFoam tutorial that exercises four things at once, and each of them
# was a defect this port found:
#
#   explicitPorositySource / fixedCoeff   alpha (500 -1000 -1000) -- NEGATIVE components, which OpenFOAM's
#                                         adjustNegativeResistance rewrites as val*(-maxCmpt) rather than
#                                         leaving signed. A porosity that keeps the sign drives the duct
#                                         the wrong way and still converges.
#   fixedTemperatureConstraint            and scalarFixedValueConstraint on the same cellZone
#   slip on porosityWall                  a TRANSFORM patch field: OpenFOAM writes no `value` for it, so no
#                                         gate that compares FIELDS can see its coefficients at all. Its
#                                         four transform coefficients read exactly 1.0 wrong here while
#                                         every other patch sat at 1e-14 -- see PORT.md.
#   flowRateInletVelocity                 fed the LIVE rho, not rhoInlet and not the face cells
#
# squareBend covers none of them: no fvOptions, no transform patch, and its own gate already runs the
# inlet. So this is not a second copy of that gate, it is the only end-to-end cover for the porosity and
# transform-patch paths.
#
# THE COMPARISON POINT IS OPENFOAM'S OWN CONVERGENCE, not a fixed iteration count. Comparing two codes at
# an endTime compares TRAJECTORIES unless both have arrived: squareBend reads U 2.8e-02 at its shipped
# endTime and 1.8e-03 where it converges -- same code, same case, a factor of 15 from nothing but where
# the comparison was taken. The tutorial's residualControl decides, and if it never fires this reports
# that instead of reporting a number.
#
# The mesh is generated rather than committed: blockMesh is deterministic and 28000 cells is not something
# to carry in the repo twice. The three named blocks ARE the porosity cellZone -- the case runs no
# topoSet, so a mesh without them would silently drop the porosity, which is asserted below.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cpp"
MIRRORBIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
cd "$W/case" || exit 1
rm -rf 0 [1-9]* processor* log.*

blockMesh > blockMesh.log 2>&1 \
    || { echo "FAIL: blockMesh did not run"; tail -20 blockMesh.log; exit 1; }
[ -f constant/polyMesh/owner ] || { echo "FAIL: blockMesh wrote no mesh"; exit 1; }

# THE POROSITY CELLZONE MUST EXIST. Without it fvOptions selects nothing, the porosity contributes
# nothing, and the case still runs and still converges -- to the wrong answer. A gate that let that pass
# would be comparing two duct flows with no porous block in either.
[ -f constant/polyMesh/cellZones ] || { echo "FAIL: no cellZones -- the porosity would be silently absent"; exit 1; }
grep -q "porosity" constant/polyMesh/cellZones \
    || { echo "FAIL: no 'porosity' cellZone -- fvOptions would select nothing"; exit 1; }

# The case's OWN fvOptions must still carry the negative resistance this gate is partly here to cover.
grep -qE "alpha .*-" constant/fvOptions \
    || { echo "FAIL: fvOptions no longer carries a negative alpha component"; exit 1; }
# ...and U's porosityWall must still be the transform patch.
grep -q "slip" 0.orig/U || { echo "FAIL: porosityWall is no longer a slip patch"; exit 1; }

cp -r 0.orig 0

# ASCII at 15 digits so the comparison is not bottomed out by the write format -- at the shipped 6 it
# floors near 1e-06 and hides real disagreement underneath. Function objects removed: they compute no
# field this reads. endTime raised so residualControl, not the shipped endTime, is what stops the run.
python3 - <<'PYEOF'
import re
c = 'system/controlDict'
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', '5000'),
             ('writeInterval', '5000'), ('writeControl', 'timeStep')]:
    s = re.sub(r'^%s .*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
open(c, 'w').write(s)
PYEOF

rhoSimpleFoam > of.log 2>&1 || { echo "FAIL: OpenFOAM did not run"; tail -25 of.log; exit 1; }
grep -q "^End" of.log || { echo "FAIL: OpenFOAM did not finish"; tail -25 of.log; exit 1; }

# WHERE OpenFOAM stopped. If residualControl did not fire, the run hit endTime and is NOT converged --
# comparing there would compare trajectories, so this says so rather than reporting a number.
N=$(grep -oE "SIMPLE solution converged in [0-9]+ iterations" of.log | grep -oE "[0-9]+" | head -1)
[ -n "$N" ] || { echo "FAIL: OpenFOAM did not converge within 5000 iterations -- comparing there would"; \
                 echo "      compare trajectories, not answers"; exit 1; }
[ -d "$N" ] || { echo "FAIL: OpenFOAM wrote no $N/"; exit 1; }
echo "  OpenFOAM converged in $N iterations"

rm -rf 0 && cp -r 0.orig 0
# The host end-to-end comparison's status is CAPTURED, not left as the script's exit code: the CUDA arm
# below runs after it, and appending anything after the last command silently makes that command's
# failure invisible. (This gate is not registered in CMakeLists, so nothing was watching it.)
"$BIN" "$W/case" 0 "$N"
hostrc=$?

# ---- THE CUDA ARM: the same tutorial, with the device modules ------------------------------------
# This case declares THREE fvOptions -- explicitPorositySource/fixedCoeff, fixedTemperatureConstraint
# and scalarFixedValueConstraint(k, epsilon) -- and the CUDA arm used to refuse ALL of them ("the case
# declares an fvOption this port does not implement"), even though rhoUEqn has applied a porous zone
# since it was written. The porosity is projected now (buildDeviceStepInput) and the two CONSTRAINTS
# are applied with fvMatrix::setValues on the device (deviceSetValues, shared with the kEpsilon
# closure), so the tutorial runs on this arm at all.
#
# The check is the CONSTRAINT ITSELF, not just "it ran": k must come out pinned to 1 on every cell of
# the porous zone, and on the SAME number of cells as the host arm -- which is the arm the end-to-end
# gates measure against OpenFOAM. A run that quietly dropped the constraint would still finish.
CU="$W/cuda"; rm -rf "$CU"; cp -r "$W/case" "$CU"
rm -rf "$CU"/[1-9]* "$CU"/0; cp -r "$CU/0.orig" "$CU/0" 2>/dev/null || true
python3 - "$CU/system/controlDict" <<'PYEOF'
import re, sys
c = sys.argv[1]; s = open(c).read()
for k, v in [('endTime', '5'), ('writeInterval', '5'), ('writeControl', 'timeStep'),
             ('writeFormat', 'ascii'), ('writePrecision', '15'), ('startFrom', 'startTime')]:
    s = re.sub(r'^%s .*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
open(c, 'w').write(s)
PYEOF
# AS THE TUTORIAL SHIPS, the device arm must REFUSE it: `porosityWall` is `type wall;` in the mesh and
# `slip` in 0/U, on a plane at 45 degrees, and the device's segregated symmetry treatment is exact only
# for axis-aligned planes. That refusal keyed on the MESH patch type and so never saw this patch --
# measured at iteration 1, with the interiors still agreeing to 1e-11, the device put
# (1.04804 1.04919 ...) on those faces where the host and OpenFOAM put (3.58007 3.58007 ...).
cuout=$( cd "$CU" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$MIRRORBIN" -case "$CU" 2>&1 || true )
echo "$cuout" | grep -q "porosityWall" && echo "$cuout" | grep -q "not aligned to a coordinate axis" \
    && echo "  cuda arm: the tutorial's TILTED SLIP wall is refused by name   ok" \
    || { echo "$cuout" | tail -4; echo "FAIL: the CUDA arm did not refuse the tilted slip patch"; exit 1; }

# ...and with that patch made axis-independent (noSlip), the SAME case runs on the device and its three
# fvOptions are applied. The mutation is one BC: everything the fvOptions machinery does is unchanged.
CU2="$W/cuda2"; rm -rf "$CU2"; cp -r "$CU" "$CU2"; rm -rf "$CU2"/[1-9]*
python3 "$ROOT/tests/rho_angledduct_noslip.py" "$CU2/0/U"
grep -q "noSlip" "$CU2/0/U" || { echo "FAIL: the noSlip mutation did not apply"; exit 1; }
cuout=$( cd "$CU2" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$MIRRORBIN" -case "$CU2" 2>&1 )
echo "$cuout" | grep -q "^End" \
    && echo "$cuout" | grep -q "fixedTemperatureConstraint T=" \
    && echo "$cuout" | grep -q "scalarFixedValueConstraint k=" \
    && echo "$cuout" | grep -q "explicitPorositySource/fixedCoeff" \
    && echo "  cuda arm: with an axis-independent wall it runs, all three fvOptions applied   ok" \
    || { echo "$cuout" | tail -4; echo "FAIL: the CUDA arm did not run the mutated tutorial with its fvOptions"; exit 1; }
CU="$CU2"

HO="$W/host"; rm -rf "$HO"; cp -r "$CU" "$HO"; rm -rf "$HO"/[1-9]*   # $CU is the mutated case
( cd "$HO" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$MIRRORBIN" -case "$HO" > host.log 2>&1 ) \
    || { tail -4 "$HO/host.log"; echo "FAIL: the host arm did not run the tutorial"; exit 1; }
CUDA_K="$CU/5/k" HOST_K="$HO/5/k" python3 - <<'PYEOF' || exit 1
import os, re, sys
import numpy as np
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\((.*?)\n\)\s*;', s, re.S)
    return None if not m else np.array([float(x) for x in m.group(2).split()])
a, b = read(os.environ['CUDA_K']), read(os.environ['HOST_K'])
if a is None or b is None or a.shape != b.shape:
    print('  k missing or shape mismatch   FAIL'); sys.exit(1)
na = int(np.sum(np.abs(a - 1.0) < 1e-9)); nb = int(np.sum(np.abs(b - 1.0) < 1e-9))
print('  cells with k pinned to the constrained 1.0: cuda %d, host %d' % (na, nb))
if na == 0:
    print('  the CUDA arm pinned NOTHING -- the constraint was dropped   FAIL'); sys.exit(1)
if na != nb:
    print('  the two arms pin different cells   FAIL'); sys.exit(1)
sys.exit(0)
PYEOF
echo "  cuda arm: the k constraint pins the same cells as the host arm   ok"

[ "$hostrc" = 0 ] || { echo "FAIL: the host end-to-end comparison above did not pass (exit $hostrc)"; exit "$hostrc"; }
echo PASS
