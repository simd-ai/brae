#!/usr/bin/env bash
# The device's epsilon at a WALL-FUNCTION-CONSTRAINED cell that also touches a CYCLIC PAIR, against
# REAL OpenFOAM's own epsilon after one step.
#
# THE CASE is the tutorial's own RAS/damBreakPorousBaffle, meshed the way its Allrun meshes it:
# blockMesh, setFields, then createBaffles, which turns 13 internal faces at x = 0.3042 into the cyclic
# pair porous_half0/porous_half1. The baffle stands ON lowerWall, so two cells touch both the pair and a
# wall -- the only cells in the tree where fvMatrix::setValues has to cut a periodic connection.
#
# EVERY SOLVE IS PINNED AT 1e-16 IN BOTH CODES, and adjustTimeStep is off. At the case's own tolerances
# this comparison would measure where two Krylov paths stopped; pinned, it measures the discretisation.
# One step, because the defect this gates is visible in the first assembly and a trajectory is not.
#
# WHAT IT ASSERTS: the device's epsilon at those cells is as close to OpenFOAM's as brae's own host arm
# is, and -- the control -- that the pair's UNCONSTRAINED cells agree on both arms too, so the
# assertion is about the constrained rows and not about the case having run.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_device_wallfn_cyclic_vs_openfoam"
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }

OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1     || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v setFields > /dev/null 2>&1     || { echo "SKIP: setFields not on PATH"; exit 77; }
command -v createBaffles > /dev/null 2>&1 || { echo "SKIP: createBaffles not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1     || { echo "SKIP: interFoam not on PATH"; exit 77; }

TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/damBreakPorousBaffle"
[ -d "$SRC" ] || { echo "SKIP: $SRC not found"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
C="$W/case"
cp -r "$SRC" "$C" || { echo "FAIL: could not stage the case"; exit 1; }
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
cp -r "$C/0.orig" "$C/0"

DT=0.0001
sed -i "s/^endTime .*/endTime         $DT;/; s/^deltaT .*/deltaT          $DT;/; \
        s/^adjustTimeStep .*/adjustTimeStep  no;/; s/^writeControl .*/writeControl    timeStep;/; \
        s/^writeInterval .*/writeInterval   1;/; s/^writeFormat .*/writeFormat     ascii;/; \
        s/^writePrecision .*/writePrecision  17;/; s/^writeCompression .*/writeCompression off;/" \
    "$C/system/controlDict"
# EVERY solve, in both codes: the case's own 1e-07 would make this a comparison of stopping points
sed -i 's/^\( *\)tolerance .*/\1tolerance       1e-16;/; s/^\( *\)relTol .*/\1relTol          0;/' \
    "$C/system/fvSolution"
grep -q "tolerance       1e-16;" "$C/system/fvSolution" \
    || { echo "FAIL: the solves were not pinned"; exit 1; }

( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -20 "$C/log.blockMesh"; exit 1; }
( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; tail -20 "$C/log.setFields"; exit 1; }
( cd "$C" && createBaffles -overwrite > log.createBaffles 2>&1 ) \
    || { echo "FAIL: createBaffles"; tail -20 "$C/log.createBaffles"; exit 1; }
grep -q "porous_half0" "$C/constant/polyMesh/boundary" \
    || { echo "FAIL: createBaffles left no cyclic pair in the mesh"; exit 1; }
( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; exit 1; }
# OpenFOAM names the directory from the time VALUE under the case's timeFormat/timePrecision, so the
# step is spelled here the way OpenFOAM will spell it -- and the gate says so rather than skipping if
# the two ever part.
[ -f "$C/$DT/epsilon" ] || { echo "FAIL: OpenFOAM wrote no epsilon at $DT"; ls "$C"; exit 1; }
echo "OpenFOAM ran one step to t = $DT with every solve pinned at 1e-16"

"$BIN" "$C" "$C/$DT"
rc=$?
echo "interfoam_wallfn_cyclic_vs_openfoam: rc $rc"
exit $rc
