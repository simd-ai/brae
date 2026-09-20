#!/usr/bin/env bash
# brae's interFoam with MRF against REAL OpenFOAM's, on laminar/mixerVessel2D AS SHIPPED, field by field
# and solve by solve.
#
# THE METHOD is the other interFoam gates': exactly N identical FIXED steps at the tutorial's own
# deltaT (1e-3), both codes read at one instant. The mesh is the tutorial's own -- blockMeshDict.m4,
# topoSet and setsToZones for the `rotor` cellZone -- 3072 cells.
#
# WHAT interFoam DOES WITH MRF, and so what is ported (UEqn.H:1, :6; pEqn.H:17, :19):
#   MRF.correctBoundaryVelocity(U)   the rotor patch, a noSlip, takes Omega x (Cf - origin)
#   + MRF.DDt(rho, U)                rho*(Omega x U) on the zone's cells, explicit in U
#   MRF.zeroFilter(ddtCorr flux)     zeroed on the zone's faces: inside the zone phi.oldTime() is
#                                    relative to the frame and the flux of U.oldTime() is not
#   MRF.makeRelative(phiHbyA)        the frame flux taken off the zone's faces, the interface included
# The vessel is CLOSED (p_rgh zeroGradient on both walls, pRefCell 0), `div(rhoPhi,U) Gauss linear`,
# three correctors, no momentum predictor. `omega constant 6.2831853;` is a Function1, which the shared
# MRF reader took as a bare number.
#
# THE CONTROL is OpenFOAM's own answer with the zone `active no`: nothing then moves at all.
#
# MEASURED, twenty steps of 1e-3: all 60 p_rgh iteration counts OpenFOAM's, initial residuals within
# 1.2e-11; alpha 8.2e-15, p_rgh 4.9e-14, U 1.9e-15. The zone is 1536 of 3072 cells with 3024 internal
# faces (the interface among them) and 192 rotor faces that move with the frame.
#
# BROKEN ONCE EACH, at t = 0.02 (U, p_rgh, p_rgh iteration counts equal):
#   MRF.makeRelative(phiHbyA) dropped               1.1e+00, 8.7e-01, 24 of 60
#   the ddtCorr flux not zero-filtered              1.7e-01, 2.2e-01, 30 of 60
#   MRF.correctBoundaryVelocity skipped             1.6e-01, 6.1e-01, 29 of 60
#   MRF.DDt dropped                                 2.7e-02, 2.9e-01, 42 of 60
#   MRF.DDt(U) for MRF.DDt(rho, U)                  2.7e-02, 2.9e-01, 42 of 60
# The rotor's frame velocity has no oracle of its own -- OpenFOAM writes a noSlip patch as its type
# alone -- so it is held through U, by the third line.
#
# NOT CLAIMED, each refused by name: MRF under a moving mesh, MRF under RAS, MRF beside a
# fixedFluxPressure patch (constrainPressure's MRF.relative), an omega that varies in time, and the
# device loop.
# THE DEVICE LOOP RUNS THIS CASE NOW, at the same bounds: alpha 7.3e-15, p_rgh 3.0e-14, U 2.1e-15, and
# all 60 of its own p_rgh solves take OpenFOAM's iteration counts (initial residuals within 1.5e-11).
# It took two modules, both transcribed from the host arm's lines:
#   MRF, all four calls -- correctBoundaryVelocity(U) in the U-boundary hook (UEqn.H:1, on the HOST field
#   the boundary snapshot is taken from), DDt(rho, U) rho-weighted in the shared assembler before relax
#   (UEqn.H:6), zeroFilter on the ddtCorr term (pEqn.H:18) and makeRelative(phiHbyA) BETWEEN that term and
#   phig (pEqn.H:19, which is why the device pEqn splits the two adds).
#   THE PRESSURE REFERENCE -- setReference at the cell's CURRENT p_rgh (pEqn.H:47) and the level shift of p
#   with p_rgh rebuilt from it (pEqn.H:74-83). Three controls in the device driver were never set while the
#   case was refused, so the step never pinned at all: p_rgh was a constant -2.17e+01 out with a spread of
#   only 1.5e-02 about it, which is how a missing reference reads.
# BROKEN ONCE EACH, device against OpenFOAM (alpha, p_rgh, U):
#   correctBoundaryVelocity skipped        8.5e-02, 6.1e-01, 1.6e-01
#   MRF.DDt dropped                        1.7e-02, 2.9e-01, 2.7e-02
#   MRF.DDt not rho-weighted               1.7e-02, 2.9e-01, 2.7e-02
#   zeroFilter dropped                     1.7e-01, 2.2e-01, 1.7e-01
#   makeRelative dropped                   9.9e-01, 8.7e-01, 1.1e+00
#   the level shift dropped                    --  , 3.1e-01,   --
# NOT DISCRIMINATED: the value setReference pins at -- g is (0 0 0) here, so p == p_rgh and the cell's own
# p_rgh IS pRefValue, to the last digit of every field and every iteration count. test_device_inter_peqn.cu
# asserts that one bit-level, with the old behaviour as its control.
# STILL REFUSED on the device: a case that needs a reference AND has an adjustable boundary face, because
# adjustPhi (pEqn.H:21-26) is not on the device. Every face of this closed vessel has its flux fixed by U,
# where OpenFOAM's own massCorr stays 1, so the device matches it by doing nothing.
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_mrf_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/mixerVessel2D"
STEPS=${STEPS:-20}
DT=${DT:-1e-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: laminar/mixerVessel2D tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1   || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v setsToZones > /dev/null 2>&1 || { echo "SKIP: setsToZones not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1   || { echo "SKIP: interFoam not on PATH"; exit 77; }
command -v m4 > /dev/null 2>&1          || { echo "SKIP: m4 not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the tutorial, apply the profile, fix the step, mesh it as Allrun.pre does, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "omega  *constant " "$C/constant/MRFProperties" \
        || { echo "FAIL: the tutorial no longer writes omega as a \`constant\` Function1"; return 1; }
    grep -q "active  *yes;" "$C/constant/MRFProperties" \
        || { echo "FAIL: the tutorial's zone is no longer \`active yes\`"; return 1; }
    if [ "$profile" = inactive ]; then
        sed -i 's/active  *yes;/active      no;/' "$C/constant/MRFProperties"
    fi

    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && m4 system/blockMeshDict.m4 > system/blockMeshDict ) || { echo "FAIL: m4 [$profile]"; return 1; }
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && topoSet > log.topoSet 2>&1 ) || { echo "FAIL: topoSet [$profile]"; tail -20 "$C/log.topoSet"; return 1; }
    ( cd "$C" && setsToZones -noFlipMap > log.setsToZones 2>&1 ) || { echo "FAIL: setsToZones [$profile]"; tail -20 "$C/log.setsToZones"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in inactive mrf; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_mrf_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path
grep -q "Creating MRF zone list from MRFProperties" "$W/mrf/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log does not build the MRF zone list"; exit 1; }

"$BIN" "$W/mrf" "$W/mrf/0" "$W/mrf/$END" "$STEPS" "$W/mrf/log.interFoam" "$W/inactive/$END" || rc=1

echo "interfoam_mrf_vs_openfoam: rc $rc"
exit $rc
