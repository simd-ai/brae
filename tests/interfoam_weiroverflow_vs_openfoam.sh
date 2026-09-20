#!/usr/bin/env bash
# brae's interFoam with a free-level inlet against REAL OpenFOAM's, on RAS/weirOverflow AS SHIPPED, field
# by field and solve by solve.
#
# THE METHOD is the other interFoam gates': exactly N identical FIXED steps at the tutorial's own
# deltaT (1e-3), both codes read at one instant. The mesh is the tutorial's own blockMesh, 5080 cells,
# non-orthogonal to 26.4 degrees; setFields puts the water behind the weir.
#
# THE TWO CONDITIONS, both on `inlet`:
#   U      variableHeightFlowRateInletVelocity   updateCoeffs rebuilds n*avgU*alpha_p with
#                                                avgU = -flowRate/gSum(magSf*alpha_p), alpha_p the phase
#                                                field's STORED patch values clipped to [0, 1]
#                                                (...InletVelocityFvPatchVectorField.C:103-139): the
#                                                prescribed VOLUME rate through the wet part of the inlet
#   alpha  variableHeightFlowRate                mixed, refGrad 0 (...FvPatchField.C:125-164): where
#                                                phi < -SMALL the face cell's value clipped at
#                                                lowerBound and upperBound, FIXED; elsewhere zeroGradient
# brae's patches cannot look a field up: the driver hands U's the phase fraction at the momentum
# assembly, where OpenFOAM's updateCoeffs runs, and alpha's gets the flux through updateFromFlux and
# the cells through evaluate().
#
# THE CONTROL is OpenFOAM's own answer with the inlet U a fixedValue at the file's (0 0 0) -- what a
# loop that never rebuilt the condition runs, which is what brae's interFoam did to
# flowRateInletVelocity until tests/interfoam_angledduct_vs_openfoam.sh found it.
#
# THE ALPHA CONDITION HAS NO CONTROL ON THIS CASE, and the reason is the condition. With alpha inside
# [lowerBound, upperBound] = [0, 1] its value is the face cell's on inflow AND on outflow; it differs
# from zeroGradient only in clipping an overshoot and in the coefficients an implicit alpha solve would
# take, and this case has no MULESCorr. An inletOutlet of 0 in its place is not a control either:
# OpenFOAM ITSELF goes to NaN at the third step, because the velocity condition then divides the flow
# rate by a wet area of zero -- the division brae's class refuses by name. The per-face logic is held
# by tests/test_variable_height_flow_rate.cu instead; the header of that file says against what.
#
# MEASURED, ten steps of 1e-3, as shipped: all 30 p_rgh, 10 epsilon and 10 k iteration counts
# OpenFOAM's; alpha 4.0e-13, p_rgh 8.0e-13, U 1.8e-11, k 5.1e-14, epsilon 6.2e-14, nut 8.8e-14, the
# inlet's velocity 2.7e-13 of its largest value, face by face against the `value` OpenFOAM writes.
#
# WHAT THE GATE FOUND, and it was not the new conditions. brae's pressure equation added fvc::ddtCorr to
# phiHbyA on the INTERNAL faces only. OpenFOAM adds a whole surface field, and fvcDdtPhiCoeff zeroes the
# coupling coefficient only on a patch where U FIXES A VALUE (ddtScheme.C). This case's outlet is
# `U zeroGradient` -- the first gated open patch that fixes nothing -- and once its flux turned outward,
# at the third step, the limiter's coefficient left zero: phi 3.2e-04 out in the outlet's top corner
# after that step, U 3.6e-05 after ten, 22 of 30 p_rgh counts. It was localised by WHERE and WHEN: every
# cell field, every stored patch value and phi agreed to 3e-10 after two steps, and the first
# disagreement was at the outlet. Converging the pressure solves to 1e-13 changed nothing, which ruled
# the stopping point out. The term reads U.oldTime()'s PATCH value (fvc::dotInterpolate), not the face
# cell's: the same number on a zeroGradient patch and on no other -- on RAS/angledDuct's tilted slip
# wall the cell's velocity made a correction OpenFOAM does not have, and that gate's floor arm went from
# 4e-15 to 1.4e-11 until the patch value was used.
#
# BROKEN ONCE EACH, at t = 0.01 (U, the inlet's velocity, p_rgh iteration counts equal):
#   the inlet never rebuilt (the control)                       1.0e+00
#   U_p = n*avgU, without the alpha_p weight                    7.6e-01, 3.6e+00, 8 of 30
#   avgU over the whole inlet area, not the wet part            6.0e-01, 2.2e+00, 8 of 30
#   alpha_p from the face CELLS of the old alpha                4.7e-03, 3.2e-02, 24 of 30
#   ddtCorr's boundary half dropped (what was found)            3.6e-05, 1.2e-09, 22 of 30
#   the alpha condition never updated                           the inlet stays at the file's alpha 0
#                                                               and the velocity condition REFUSES a
#                                                               dry inlet by name
#
# NOT CLAIMED: a flowRate that varies in time (refused), and the device loop (refused by name -- for
# the two conditions, and for ddtCorr's boundary half on any open patch whose U fixes no value).
# THE DEVICE LOOP RUNS THIS CASE NOW, at the same bounds: alpha 2.4e-13, p_rgh 4.3e-13, U 5.1e-11, and
# its inlet 5.6e-13 face by face. Three things it refused, each transcribed from the host arm:
#   ddtCorr's BOUNDARY half. pEqn.H:16-17 adds a whole surfaceScalarField and fvcDdtPhiCoeff zeroes the
#   coupling coefficient only where U FIXES a value; this outlet is a zeroGradient U, so the correction
#   is live there. deviceDdtCorr already computed it (and already zeroed it where U fixes a value) and
#   the step simply dropped it. Added with interpolate(rho*rAU)'s patch value -- rho's own boundary value
#   and rAU's extrapolated one, 1/A of the face cell (inter_peqn_cpp.cu:531-573). BROKEN: U 3.6e-05,
#   p_rgh 1.3e-06 against a bound of 5.0e-10.
#   the variableHeightFlowRateInletVelocity, rebuilt at every momentum assembly from the phase fraction
#   on its patch (variableHeightFlowRateInletVelocityFvPatchVectorField.C:103-139), where the host driver
#   rebuilds it (inter_driver_cpp.cu:722-735). BROKEN: U 1.0000e+00, the inlet itself 3.6e+00 out.
#   alpha's own variableHeightFlowRate, a mixed condition whose valueFraction is 1 on an inflow face and
#   0 on the rest. The device's MULES mask was the patch's single fixesValue(), taken once.
# AND ONE THE ARM FOUND ELSEWHERE. Adding the boundary half put RAS/angledDuct's `inactive` arm 1.4e-11
# from OpenFOAM where it had been 7.6e-15: the device computed that half from the face CELL's
# U.oldTime(), while fvc::dotInterpolate(Sf, U.oldTime()) on an uncoupled patch is
# `pSf & vf.boundaryField()[pi]` -- the STORED patch value (surfaceInterpolationScheme.C:296-298), which
# is what the host arm reads (inter_peqn_cpp.cu:263-270). The two coincide wherever the patch value
# follows the cell and differ on a SLIP wall, which angledDuct's `porosityWall` is. The device now
# snapshots U's patch values with its cells at the top of the step.
# NOT CLAIMED: that last one. The mask does vary here (13 of 44 faces outflow at the end, asserted), but
# the answer does not depend on it -- dropping the refresh changes no digit, and forcing the mask to zero
# over the whole patch moves p_rgh from 4.2981e-13 to 4.3078e-13 and nothing else. The faithful form is
# in; a fixture where MULES's limiter is live on that patch would be what tests it.
# STILL RED, AND NOT FROM THIS WORK: the k FINAL-residual arm, where OpenFOAM reports 8.189e-14 and brae
# 8.092e-14 -- both at round-off, the comparison is of two stopping points, and it awaits a decision.
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_weiroverflow_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/weirOverflow"
STEPS=${STEPS:-10}
DT=${DT:-1e-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/weirOverflow tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the tutorial, apply the profile, fix the step, mesh it as Allrun does, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "type  *variableHeightFlowRateInletVelocity;" "$C/0/U" \
        || { echo "FAIL: the tutorial's U no longer carries a variableHeightFlowRateInletVelocity"; return 1; }
    grep -q "type  *variableHeightFlowRate;" "$C/0/alpha.water" \
        || { echo "FAIL: the tutorial's alpha no longer carries a variableHeightFlowRate"; return 1; }

    PROFILE="$profile" STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
profile = os.environ['PROFILE']
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
if profile == 'frozen':
    q = os.path.join(d, '0/U')
    t = open(q).read()
    t, k = re.subn(r'type\s+variableHeightFlowRateInletVelocity;\s*flowRate[^;]*;\s*alpha[^;]*;',
                   'type            fixedValue;', t)
    assert k == 1, 'the U inlet was not frozen'
    open(q, 'w').write(t)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in frozen weir; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_weiroverflow_vs_openfoam: staging failed"; exit 1; }

"$BIN" "$W/weir" "$W/weir/0" "$W/weir/$END" "$STEPS" "$W/weir/log.interFoam" \
       "$W/frozen/$END" || rc=1

echo "interfoam_weiroverflow_vs_openfoam: rc $rc"
exit $rc
