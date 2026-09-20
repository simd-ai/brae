#!/usr/bin/env bash
# brae interFoam on a FULLY PERIODIC mesh against REAL OpenFOAM's, on validation/interFoamCyclic.
#
# THE CASE is brae's own, not a tutorial: 800 cells, x cyclic, y walled, z one cell thick, with the
# water surface staged as a STEP at the periodic boundary (y = 0.3 on one side, 0.15 on the other), so
# alpha, its gradient, nHatf and the pressure all carry a jump through the pair from the first step and
# gravity drives flow across it for the whole run. The baffle gate
# (tests/interfoam_baffle_vs_openfoam.sh) already couples an INTERNAL pair; what is new here is a pair
# on the DOMAIN boundary with no wall behind it, and a case small enough to run the device arm on.
#
# THE ORACLE is OpenFOAM's own fields after ten FIXED steps of 2e-3, written ASCII at 17 digits.
# THE CONTROL is OpenFOAM's own answer with the pair replaced by two WALLS: same block, same cells.
#
# WHAT IT ASSERTS: brae's host loop is OpenFOAM's on this mesh; OpenFOAM moves fluid through the pair,
# so the comparison is not vacuous; walling the pair is a different answer by orders of magnitude; and
# the DEVICE arm agrees with the same OpenFOAM fields, and with brae's own host arm.
#
# MEASURED, ten steps of 2e-3: alpha 1.9e-13, p_rgh 9.8e-12 relative of 1.7e+03, U 7.5e-13 relative of
# 0.73, all 30 p_rgh iteration counts OpenFOAM's. CONTROL: OpenFOAM with the pair two walls, alpha
# 2.1e-01 and U 100% away -- and its step one runs 64/50/1 iterations where the periodic case runs
# 70/32/4, so the iteration-count arm sees the coupling too.
#
# TWO PROFILES, because they are two alpha equations: `MULESCorr` as the fixture ships, and `explicit`
# with MULESCorr off -- no implicit pre-solve, and so no mixture.correct() before the first corrector,
# which is what makes its phir read the nHatf the PREVIOUS TIME STEP left.
#
# MEASURED, ten steps, host then device against OpenFOAM:
#   MULESCorr  host alpha 1.9e-13, p_rgh 9.8e-12, U 7.5e-13; device 4.2e-11, 1.98e-11, 5.7e-11
#   explicit   host alpha 5.9e-14, p_rgh 2.9e-12, U 5.7e-13; device 4.3e-11, 3.1e-11, 3.0e-11
#   jump       host alpha 4.1e-13, p_rgh 1.0e-11, U 4.4e-12; device 6.2e-12, 2.4e-11, 7.5e-11
# and the device against brae's own host arm, 4.2e-11 / 1.02e-11 / 5.8e-11 and 4.3e-11 / 2.9e-11 /
# 3.0e-11. The alpha figure is the device alpha solver's stopping point: pinned at 1e-16 the two arms
# agree to 6.1e-14.
#
# BROKEN ONCE EACH (at t = 0.02):
#   the pair's laplacian deltaCoeffs 0.1% out          host alpha 2.1e-03, p_rgh 3.6e-03, U 7.9e-03,
#                                                      and 2 of 30 iteration counts lost
#   divDevReff handed a null pair (as it was)          device alpha 4.4e-06, p_rgh 6.2e-06, U 5.4e-04
#   the Gauss-Seidel sweep not applying the interface  device alpha 6.6e-01, p_rgh 1.2e+00, U 1.6e+01
#   the jump left out of the device entirely           device alpha 4.9e-02, U 45% -- the control's
#                                                      own distance, i.e. the plain-cyclic answer
#   the pair's mixture boundary one stage stale        device alpha 6.6e-05, p_rgh 1.1e-04, U 3.8e-03
#                                                      (porousBafflePressure reads nu and rho THERE,
#                                                      and they are the two cells' interpolated)
#   the pair's nHatf in a buffer local to one step     the `explicit` profile ABORTS by name (its
#                                                      first corrector asks for a normal nothing has
#                                                      written yet); `MULESCorr` still passes, which
#                                                      is why that profile alone did not cover it
# NOT DISCRIMINATED, measured: the pair's nHatf SEEDED AT ZERO rather than from calculateK. This
# fixture starts from rest, so phi and therefore phic are zero at step one and phir = phic*nHatf is
# zero whatever the normal is; only a restart with a live flux would read that seed.
# NOT MEASURABLE, because brae refuses it outright: the pair left uncoupled, which every other cyclic
# gate calls its control -- the host driver throws and names the patch rather than running it as two
# walls.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_cyclic_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC="$ROOT/validation/interFoamCyclic"
STEPS=${STEPS:-10}
DT=${DT:-0.002}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: $SRC not found"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v setFields > /dev/null 2>&1 || { echo "SKIP: setFields not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the fixture, apply the profile, mesh it, set the fields, run OpenFOAM
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[0-9]* "$C"/processor* "$C"/log.* "$C"/constant/polyMesh
    cp -r "$SRC/0.orig" "$C/0.orig"
    cp -r "$C/0.orig" "$C/0"
    grep -q "type cyclic; neighbourPatch right;" "$C/system/blockMeshDict" \
        || { echo "FAIL: the fixture's left patch is no longer the cyclic this gate stages"; return 1; }
    if [ "$profile" = explicitMules ] || [ "$profile" = explicitWalls ]; then
        # THE EXPLICIT MULES BRANCH. Without MULESCorr there is no implicit pre-solve and no
        # mixture.correct() before the correctors, so the FIRST corrector's phir reads the nHatf the
        # PREVIOUS TIME STEP left -- on the pair as everywhere else. The device kept that normal in a
        # buffer local to one step, which is empty when the first corrector asks for it, and refused.
        sed -i 's/MULESCorr       yes;/MULESCorr       no;/' "$C/system/fvSolution"
        grep -q "MULESCorr       no;" "$C/system/fvSolution" \
            || { echo "FAIL: the $profile profile did not turn MULESCorr off"; return 1; }
    fi
    if [ "$profile" = jump ]; then
        # A POROUS BAFFLE ON THE PERIODIC BOUNDARY. p_rgh's pair becomes a porousBafflePressure -- a
        # fixedJump cyclic whose jump is rebuilt at every assembly from that assembly's flux and
        # laminar viscosity (porousBafflePressureFvPatchField.C:125-189). The coupling is unchanged;
        # what is added is a pressure drop the matrix has to carry.
        #
        # THE COEFFICIENTS ARE NOT THE TUTORIAL'S, and the control is why. `uniformJump true` averages
        # Un over the patch, and on a gravity-driven periodic boundary that average is ~0: with the
        # tutorial's D 1000, I 500, length 0.15 the jump came out so small that OpenFOAM's own answer
        # with it and without it differed by 1.1e-13 in alpha -- the gate would have passed on a jump
        # that did nothing. Per-face (`uniformJump false`) at the tutorial's coefficients diverges in
        # OPENFOAM ITSELF (alpha 1e+286 by step 10, then a `nan` in its own solver log). I 5 over
        # length 0.05 is the one that is both live and stable, and the control below measures it.
        python3 - "$C" <<'PYEOF' || { echo "FAIL: the jump profile was not staged"; return 1; }
import sys
p = sys.argv[1] + '/0/p_rgh'
s = open(p).read()
old = '    "(left|right)" { type cyclic; }'
assert s.count(old) == 1, 'the fixture no longer writes p_rgh a plain cyclic'
s = s.replace(old,
"""    "(left|right)"
    {
        type            porousBafflePressure;
        patchType       cyclic;
        D               0;
        I               5;
        length          0.05;
        uniformJump     false;
        jump            uniform 0;
        value           uniform 0;
    }""")
open(p, 'w').write(s)
PYEOF
        grep -q "porousBafflePressure" "$C/0/p_rgh" \
            || { echo "FAIL: the jump profile did not reach p_rgh"; return 1; }
    fi
    if [ "$profile" = walls ] || [ "$profile" = explicitWalls ]; then
        # THE CONTROL: the pair replaced by two walls, in the mesh AND in every field that names it.
        # blockMesh numbers the cells from the block, so the two runs' cells are the same cells.
        sed -i 's/type cyclic; neighbourPatch right;/type wall;/; s/type cyclic; neighbourPatch left; */type wall;/' \
            "$C/system/blockMeshDict"
        grep -q "type cyclic" "$C/system/blockMeshDict" \
            && { echo "FAIL: the control's mesh still names a cyclic"; return 1; }
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type noSlip; }/' "$C/0/U"
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type zeroGradient; }/' "$C/0/alpha.water"
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type fixedFluxPressure; value uniform 0; }/' \
            "$C/0/p_rgh"
        grep -q "type cyclic" "$C/0/U" "$C/0/alpha.water" "$C/0/p_rgh" \
            && { echo "FAIL: the control's fields still name a cyclic"; return 1; }
    fi
    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '17')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) \
        || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

for p in cyclic walls explicitMules explicitWalls jump; do
    stage "$p" || { echo "interfoam_cyclic_vs_openfoam: staging failed"; exit 1; }
done

# the oracle took the path: OpenFOAM's own mesh carries the pair on one arm and not on the other
grep -q "type  *cyclic" "$W/cyclic/constant/polyMesh/boundary" \
    || { echo "FAIL: OpenFOAM's mesh has no cyclic patch"; exit 1; }
grep -q "type  *cyclic" "$W/walls/constant/polyMesh/boundary" \
    && { echo "FAIL: the control's mesh still has a cyclic patch"; exit 1; }
grep -q "MULESCorr       no;" "$W/explicitMules/system/fvSolution" \
    || { echo "FAIL: the explicit profile ran with MULESCorr still on"; exit 1; }
grep -q "MULESCorr       yes;" "$W/cyclic/system/fvSolution" \
    || { echo "FAIL: the shipped profile no longer sets MULESCorr"; exit 1; }

rc=0
"$BIN" "$W/cyclic" "$W/cyclic/0" "$W/cyclic/$END" "$STEPS" "$W/cyclic/log.interFoam" \
       "$W/walls/$END" MULESCorr || rc=1
# ...and the same case with MULESCorr off, which is a different alpha equation and a different place
# for the pair's nHatf to come from
"$BIN" "$W/explicitMules" "$W/explicitMules/0" "$W/explicitMules/$END" "$STEPS" \
       "$W/explicitMules/log.interFoam" "$W/explicitWalls/$END" explicit || rc=1
# ...and with a JUMP on the pair: a porousBafflePressure, which the matrix, the flux and the gradient
# all have to carry. ITS control is the PLAIN-CYCLIC run -- the pair coupled and nothing else -- so
# what the control measures is the jump itself and not the coupling.
"$BIN" "$W/jump" "$W/jump/0" "$W/jump/$END" "$STEPS" \
       "$W/jump/log.interFoam" "$W/cyclic/$END" jump || rc=1
echo "interfoam_cyclic_vs_openfoam: rc $rc"
exit $rc
