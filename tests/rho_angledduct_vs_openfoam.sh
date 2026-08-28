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
"$BIN" "$W/case" 0 "$N"
