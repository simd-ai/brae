#!/usr/bin/env bash
# The device's interface normal flux ON A CYCLIC PAIR against REAL OpenFOAM's own nHatf.
#
# THE ORACLE IS OpenFOAM'S FIELD. interfaceProperties registers `nHatf` in the objectRegistry
# (interfaceProperties.C:200-210), so a writeObjects function object writes it into the time directory
# with everything else -- cyclic patches included. The device is then handed OpenFOAM's alpha at that
# same instant and has to reproduce what OpenFOAM wrote on those faces. Nothing here compares brae to
# brae.
#
# THE CASE is the tutorial's own RAS/damBreakPorousBaffle, meshed the way its Allrun meshes it:
# blockMesh, then createBaffles, which turns 13 internal faces at x = 0.3042 into the cyclic pair
# porous_half0/porous_half1. A mesh without that step has no coupled patch at all and the gate says so
# rather than passing vacuously.
#
# WHAT IT ASSERTS: the device's nHatf on those 26 faces is OpenFOAM's, and -- the control -- that
# leaving the pair out of grad(alpha), which is what the device did before this unit and what a mesh
# with a wall there would give, is a DIFFERENT answer.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_device_cyclic_nhatf_vs_openfoam"
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }

OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1     || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v createBaffles > /dev/null 2>&1 || { echo "SKIP: createBaffles not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1     || { echo "SKIP: interFoam not on PATH"; exit 77; }

TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/damBreakPorousBaffle"
[ -d "$SRC" ] || { echo "SKIP: $SRC not found"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
C="$W/case"
cp -r "$SRC" "$C" || { echo "FAIL: could not stage the case"; exit 1; }
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor*
cp -r "$C/0.orig" "$C/0"

# three fixed steps, written at 15 digits, with nHatf asked for by name
END=0.003
sed -i "s/^endTime.*/endTime         $END;/; s/^writeInterval.*/writeInterval   $END;/; \
        s/^writeFormat.*/writeFormat     ascii;/; s/^writePrecision.*/writePrecision  17;/; \
        s/^adjustTimeStep.*/adjustTimeStep  no;/" "$C/system/controlDict"
cat >> "$C/system/controlDict" <<'EOF'
functions { nh { type writeObjects; libs (utilityFunctionObjects); objects (nHatf); writeOption anyWrite; } }
EOF

( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -20 "$C/log.blockMesh"; exit 1; }
( cd "$C" && createBaffles -overwrite > log.createBaffles 2>&1 ) \
    || { echo "FAIL: createBaffles"; tail -20 "$C/log.createBaffles"; exit 1; }
grep -q "porous_half0" "$C/constant/polyMesh/boundary" \
    || { echo "FAIL: createBaffles left no cyclic pair in the mesh"; exit 1; }
# THE WATER COLUMN STAGED ACROSS THE BAFFLE, as the baffle gate's `cyclicWet` profile stages it
# (tests/interfoam_baffle_vs_openfoam.sh:178). With the tutorial's own box the interface is nowhere near
# the pair at this time and nHatf there is 1e-32: MEASURED, the control -- the pair left out of
# grad(alpha) -- then agreed with OpenFOAM to 7.4e-47 too, so the gate could not tell the two apart.
sed -i 's/box  *(0 0 -1) (0.1461 0.292 1);/box (0 0 -1) (0.4 0.13 1);/' "$C/system/setFieldsDict"
grep -q "box (0 0 -1) (0.4 0.13 1);" "$C/system/setFieldsDict" \
    || { echo "FAIL: the tutorial's setFieldsDict box is not the one this gate rewrites"; exit 1; }
( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; tail -20 "$C/log.setFields"; exit 1; }
( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; exit 1; }
[ -f "$C/$END/nHatf" ] || { echo "FAIL: OpenFOAM wrote no nHatf at $END"; ls "$C/$END"; exit 1; }
echo "OpenFOAM ran to t = $END and wrote nHatf"

"$BIN" "$C" "$C/$END"
rc=$?
echo "interfoam_cyclic_nhatf_vs_openfoam: rc $rc"
exit $rc
