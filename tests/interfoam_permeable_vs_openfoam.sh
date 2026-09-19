#!/usr/bin/env bash
# brae's interFoam with a permeable wall against REAL OpenFOAM's, on laminar/damBreakPermeable, field by
# field and solve by solve.
#
# THE METHOD is the other interFoam gates': exactly N identical FIXED steps at the tutorial's own
# deltaT (1e-3), both codes read at one instant. The mesh is the tutorial's own blockMesh, 2268 cells;
# the case is RAS kEpsilon in the `density variable` lineage, with MULESCorr and two alpha correctors.
#
# THE TWO CONDITIONS, both on `rightWall`, each switching FACE BY FACE on alpha at the patch:
#   U      permeableAlphaPressureInletOutletVelocity   mixed, scalar valueFraction, refGrad 0
#          (pressurePermeableAlphaInletOutletVelocityFvPatchVectorField.C:127-178 -- the class is named
#          the other way round from its TypeName): valueFraction = max(pos(alpha_p - alphaMin), neg(phi)),
#          refValue 0 wherever that is 1. A wet face, or one the flux enters through, holds U = 0; a dry
#          face the flux leaves through is zeroGradient. pos is STRICT in this OpenFOAM (s > 0).
#   p_rgh  prghPermeableAlphaTotalPressure             mixed, driven by constrainPressure through
#          updateSnGrad as fixedFluxPressure is (...FvPatchScalarField.C:151-212): refGrad the
#          flux-consistent gradient, refValue p0 - 0.5 rho neg(phi) |U|^2 - rho ((g & Cf) - ghRef),
#          valueFraction 1 - pos(alpha_p - alphaMin). Wet: a wall's gradient. Dry: the total pressure.
# brae's patches cannot look alpha up: the case reader hands both the phase field's stored patch values
# wherever it hands out the flux, and the pressure equation hands the second rho, phi, U and gh inside
# constrainPressure, where OpenFOAM's updateSnGrad looks them up.
#
# TWO PROFILES, because the shipped case cannot show the switch in a short run:
#   shipped   as it is. The water is 0.34 m from the wall, so every face of it is DRY for the gated run:
#             the open branch alone.
#   wetWall   STAGED: setFields puts the water column against rightWall instead of leftWall. The wall
#             starts wet over 27 of its 50 faces and two of them go DRY as the column falls -- both
#             branches, and the per-face switch between them. 140 steps, and not one more: see below.
# THE CONTROL for each is OpenFOAM with rightWall CLOSED: noSlip and fixedFluxPressure.
#
# WHY wetWall STOPS AT 140 STEPS. On a wet face the pressure condition is a wall's: its gradient is
# built to cancel the flux, so phi there is (rAUf magSf)*(x/(magSf rAUf)) - x, which is 0 or ONE UNIT OF
# ROUND-OFF in x. The velocity condition's valueFraction is max(pos(alpha_p - alphaMin), neg(phi)), so
# at the assembly where a face has just gone dry, OpenFOAM reads THE SIGN OF THAT ROUND-OFF to choose
# between a wall and an open face. MEASURED at step 142, where the third face goes dry (alpha_p
# 0.00977): brae's phi on it is -3.4e-21, so it stays a wall one more assembly; OpenFOAM's UEqn.A() in
# that one cell, from tools/dumpInterFoam, is 2.018795e+04 against brae's 2.019174e+04 -- it took the
# open branch -- and every other cell's agrees to 3e-17. From there U is 7.6e-05 apart. No second code
# can follow that except by reproducing OpenFOAM's arithmetic to the last bit; the first two switches
# agree because the flux on both faces is exactly 0.0 in both codes. If this gate ever fails at a
# switch, look at the sign of phi on the newly dry face before looking anywhere else.
#
# WHAT THE GATE FOUND, in shared code. (1) U's patches are still updated() at the FIRST pressure
# corrector: the fvMatrix constructor runs updateCoeffs at the momentum assembly and, with the predictor
# off, nothing evaluates U until that corrector's correctBoundaryConditions, whose evaluate then SKIPS
# updateCoeffs and blends with the assembly-time valueFraction (mixedFvPatchField.C). brae handed the
# new flux over in every corrector. This wall flips faces from outflow to inflow in the first corrector
# of the first step: the second corrector's initial residual 1.2e-03 out with HbyA and the internal
# phiHbyA exact, U 1.5e-06 after one step, 8.6e-08 after twenty. (2) It is PER CLASS:
# pressureInletOutletVelocity's updateCoeffs ends in evaluate(), which clears the flag, so that
# condition does take the new flux at once -- applied to it too, an atmosphere face turning inward at
# step 55 put U 6.9e-03 out in one step.
#
# MEASURED: shipped, 20 steps -- alpha 8.5e-15, p_rgh 8.0e-15, U 2.7e-15, k 2.6e-15, epsilon 6.6e-15, nut
# 3.7e-15, every p_rgh residual OpenFOAM's to the last printed digit. wetWall, 140 steps -- alpha
# 4.0e-13, p_rgh 5.0e-14, U 1.4e-14, all 420 p_rgh counts OpenFOAM's, 27 faces wet at the start and 25
# at the end. The wall's own U and p_rgh are held face by face against the values OpenFOAM writes.
#
# BROKEN ONCE EACH (U on shipped, U on wetWall, wetWall's p_rgh counts equal):
#   the hydrostatic term left out of the pressure's refValue     1.0e+00, 1.0e+00, 282 of 420
#   the pressure condition open on every face, wet or dry        nothing, 1.7e+01,  91 of 420
#   the velocity condition ignoring alpha                        nothing, 1.3e-01, 206 of 420
#   the wet/dry threshold at 0.5 for the case's alphaMin 0.01    nothing, 2.2e-02, 363 of 420
#   U's patches handed the new flux in EVERY corrector           8.6e-08, 3.4e-06, 420 of 420
#   the 0.5 rho neg(phi) |U|^2 term left out                     8.6e-08, 3.4e-06, 420 of 420
#   ...the first corrector's lag applied to pressureInletOutletVelocity too
#                                                                nothing, 3.9e-02, 407 of 420
# The fifth and sixth lines carry the same numbers because they remove the same thing: the dynamic term
# is non-zero only on a face the flux has just turned inward on, while U_b there is still the cell's --
# which is exactly the one corrector the lag exists for. COUNTED, not argued: the term is non-zero 51
# times in shipped's 20 steps (largest 7.3e-02 Pa) and 23 times in wetWall's 140 (4.1e-01 Pa), and not
# once in either with the lag removed, because an updated patch holds U_b = 0 wherever phi < 0. `shipped` cannot see four of the seven, which
# is why wetWall exists.
#
# THE WALL'S OWN VELOCITY is held differently by the two profiles. On `shipped` 40 of the wall's faces
# carry a velocity at t = 0.02 (up to 0.50 m/s) and brae's agree to 5.3e-16. On `wetWall` OpenFOAM writes
# `uniform (0 0 0)` at t = 0.14 -- the wet faces are closed and the dry ones take inflow -- so the check
# there holds brae to exactly zero: it read 4.5e-01 with the velocity ignoring alpha and 4.4e-01 with
# the threshold at 0.5. The per-face logic, the `alpha none` half included, is held by
# tests/test_permeable_wall_conditions.cu against OpenFOAM's text.
#
# NOT ASSERTED, and why: epsilon's solver RESIDUALS. rightWall is a `wall` with an epsilonWallFunction
# that here carries a flux; OpenFOAM's condition derives from fixedValue and brae's is zeroGradient
# beside constrained cells, so the wall cells' diagonals differ before setValues overwrites their rows.
# The solution is the same (epsilon 7e-15) and the residual's normalisation is not: 1.8e-06 per step,
# against 0.0 with the wall closed. The iteration counts are asserted.
#
# NOT CLAIMED: a `p` that varies in space or time, a mass flux named in `phi` (both refused), the
# conditions without an `alpha` entry (implemented, ungated), and the device loop (refused by name).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_permeable_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreakPermeable"
DT=${DT:-1e-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: laminar/damBreakPermeable tutorial not found at $SRC"; exit 77; }
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

# stage <name> <nSteps> <wetWall: yes|no> <closed: yes|no>
stage()
{
    local name="$1" n="$2" wet="$3" closed="$4"
    local C="$W/$name"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "type  *permeableAlphaPressureInletOutletVelocity;" "$C/0/U" \
        || { echo "FAIL: the tutorial's U no longer carries the permeable velocity condition"; return 1; }
    grep -q "type  *prghPermeableAlphaTotalPressure;" "$C/0/p_rgh" \
        || { echo "FAIL: the tutorial's p_rgh no longer carries the permeable pressure condition"; return 1; }

    WET="$wet" CLOSED="$closed" N="$n" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['N'])
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
if os.environ['WET'] == 'yes':
    # the column against rightWall: the tutorial's box mirrored in the tank's 0.584 m width
    q = os.path.join(d, 'system/setFieldsDict')
    t = open(q).read()
    t, k = re.subn(r'box \(0 0 -1\) \(0\.2461 0\.292 1\);', 'box (0.3379 0 -1) (0.584 0.292 1);', t)
    assert k == 1, 'the setFields box was not found'
    open(q, 'w').write(t)
if os.environ['CLOSED'] == 'yes':
    q = os.path.join(d, '0/U')
    t = open(q).read()
    t, k = re.subn(r'rightWall\s*\{[^}]*\}', 'rightWall\n    {\n        type            noSlip;\n    }', t)
    assert k == 1, 'U rightWall was not closed'
    open(q, 'w').write(t)
    q = os.path.join(d, '0/p_rgh')
    t = open(q).read()
    t, k = re.subn(r'rightWall\s*\{[^}]*\}',
                   'rightWall\n    {\n        type            fixedFluxPressure;\n        value           uniform 0;\n    }', t)
    assert k == 1, 'p_rgh rightWall was not closed'
    open(q, 'w').write(t)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$name]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$name]"; tail -30 "$C/log.interFoam"; return 1; }
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$DT')))")
    [ -d "$C/$end" ] || { echo "FAIL: OpenFOAM wrote no $end directory [$name]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $n steps of deltaT $DT to t = $end   [$name]"
}

NS=${NS:-20}
NW=${NW:-140}
rc=0
stage shippedClosed "$NS" no  yes || rc=1
stage shipped       "$NS" no  no  || rc=1
stage wetWallClosed "$NW" yes yes || rc=1
stage wetWall       "$NW" yes no  || rc=1
[ $rc = 0 ] || { echo "interfoam_permeable_vs_openfoam: staging failed"; exit 1; }

ENDS=$(python3 -c "print('%.10g' % ($NS*float('$DT')))")
ENDW=$(python3 -c "print('%.10g' % ($NW*float('$DT')))")
"$BIN" "$W/shipped" "$W/shipped/0" "$W/shipped/$ENDS" "$NS" "$W/shipped/log.interFoam" \
       "$W/shippedClosed/$ENDS" shipped || rc=1
"$BIN" "$W/wetWall" "$W/wetWall/0" "$W/wetWall/$ENDW" "$NW" "$W/wetWall/log.interFoam" \
       "$W/wetWallClosed/$ENDW" wetWall || rc=1

echo "interfoam_permeable_vs_openfoam: rc $rc"
exit $rc
