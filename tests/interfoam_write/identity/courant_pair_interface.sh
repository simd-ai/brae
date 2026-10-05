#!/usr/bin/env bash
# The gate: the device loop's INTERFACE Courant number with the faces of a coupled pair in each cell's flux
# sum, on RAS/damBreakPorousBaffle's W row restaged with the water ACROSS the baffle.
# alphaCourantNo.H is CourantNo.H's surfaceSum(mag(phi)) times the nearInterface mask, so the pair's faces are
# in it as they are in the other (courant_pair_faces.sh has the defect and the fix). As the tutorial ships,
# nothing but air reaches the baffle and no cell of the pair is in the 0.01-0.99 band: the pair's term of this
# number is multiplied by a zero mask and the other gate cannot see it. Here the column is the box
# (0 0 -1) (0.4 0.13 1), the staging tests/interfoam_baffle_vs_openfoam.sh uses: the free surface crosses the
# baffle at x = 0.3042. And the row runs STEPS steps where a W row runs two: setFields leaves alpha 0 or 1, so
# after one step no cell is in the band at all (OpenFOAM prints an interface number of 0 at the second step);
# by the last of these, faces of the pair carry a flux into cells of the band.
# TWO ORACLES, as there: OpenFOAM's log of the restaged row (`Interface Courant Number mean: .. max: ..`, 17
# digits) for brae's second step's, and the host's masked surfaceSumMagPhi cell for cell
# (BRAE_CONTROL_COURANT_CHECK=1). PREMISE, counted by the check: at least one face of the pair carries a flux
# into a cell of the band.
# The CONTROL leaves the pair out of the masked sum ALONE (BRAE_CONTROL_COURANT_PAIR_LEFT_OUT=interface): the
# whole sum still holds, so what stops the run is the band's compare, and the interface number it prints is off
# OpenFOAM's. A call that forgot to hand the pair to the second number is this control.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakPorousBaffle of > "$W/ci_stage.txt" 2>&1
[ -d "$W/w_of_damBreakPorousBaffle" ] \
    || { say "damBreakPorousBaffle did not stage" FAIL; finish "interface Courant number with the pair"; }
# the wet row: the W row's mesh and pinned controls, the column restaged, OpenFOAM run on it (cached)
o="$W/ci_of"
rm -rf "${o:?}"
mkdir -p "$o"
cp -r "$W/w_of_damBreakPorousBaffle/0" "$W/w_of_damBreakPorousBaffle/constant" \
      "$W/w_of_damBreakPorousBaffle/system" "$o/"
sed -i 's/box  *(0 0 -1) (0.1461 0.292 1);/box (0 0 -1) (0.4 0.13 1);/' "$o/system/setFieldsDict"
grep -q "box (0 0 -1) (0.4 0.13 1);" "$o/system/setFieldsDict" && ( cd "$o" && setFields > log.setFields 2>&1 ) \
    || { say "the water column was not restaged across the baffle" FAIL; finish "interface Courant number"; }
STEPS=12
dt=$(sed -n -E 's/^deltaT +([^;]+);/\1/p' "$o/system/controlDict")
sed -i -E "s/^endTime .*/endTime         $(python3 -c "print('%.12g' % ($STEPS*$dt))");/" "$o/system/controlDict"
sed -i -E "s/^writeInterval .*/writeInterval   $STEPS;/" "$o/system/controlDict"
runof "$o"
for v in wet leftout; do
    e="$W/ci_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        wet)     runbrae "$e" device BRAE_CONTROL_COURANT_CHECK=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        leftout) ( cd "$e" && BRAE_CONTROL_COURANT_CHECK=1 BRAE_CONTROL_COURANT_PAIR_LEFT_OUT=interface \
                       "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
# MEASURED 2026-10-05 at the twelfth step, the same in three runs: the interface number 2.3e-13 and the Courant
# number 9.1e-13 from OpenFOAM's log (what eleven steps' flux differs by); the bounds are a decade above. The
# control's interface number is 1.4e-03 off, its band compare 8.5e-03 of the largest sum.
BOUND_BAND=3e-12
BOUND_ALL=1e-11
mark="the coupled pair's .* faces are in each cell's flux sum"
line="Courant check: [1-9][0-9]* calls with a flux compared"
faces=$(grep -a "$line" "$W/ci_wet/log.brae" | tail -1 | sed -E 's/.*; ([0-9]+) faces of the pair carry.*/\1/')
worst=$(grep -a "$line" "$W/ci_wet/log.brae" | tail -1 | sed -E 's/.*worst cell is ([0-9.e+-]+) of.*/\1/')
what="[device] PREMISE and cells: ${faces:-?} faces of the pair carry a flux into the band; every cell within"
what="$what ${worst:-?} of the host's"
grep -q "$mark" "$W/ci_wet/log.brae" && [ -n "$worst" ] && [ "${faces:-0}" -ge 1 ] \
    && [ "$(timedirs "$W/ci_wet")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
gap=$(courantOff "$W/ci_wet/log.brae" "$o/log.interFoam" "Interface Courant Number")
gapAll=$(courantOff "$W/ci_wet/log.brae" "$o/log.interFoam" "Courant Number")
what="[device] against OpenFOAM's log: the interface number within $gap, the Courant number within $gapAll"
within "$gap" "$BOUND_BAND" && within "$gapAll" "$BOUND_ALL" && say "$what" ok || say "$what" FAIL
far=$(grep -a "COURANT_CHECK: cell .* interface band flux sum is" "$W/ci_leftout/log.brae" | head -1 \
      | sed -E 's/.*faces: ([0-9.e+-]+) of the largest.*/\1/')
gap=$(courantOff "$W/ci_leftout/log.brae" "$o/log.interFoam" "Interface Courant Number")
what="CONTROL  the pair out of the band's sum alone: the check stops ${far:-?} out, the number $gap off"
what="$what OpenFOAM's"
[ "$(cat "$W/ci_leftout/exit.txt")" != 0 ] \
    && grep -q "CONTROL MODE: the coupled pair's faces are left out of the interface Courant" \
            "$W/ci_leftout/log.brae" \
    && grep -q "$mark" "$W/ci_leftout/log.brae" && [ -n "$far" ] && above "$far" 4e-12 \
    && above "$gap" "$BOUND_BAND" && say "$what" ok || say "$what" FAIL
finish "the device loop's interface Courant number sums a coupled pair's faces as surfaceSum does"
