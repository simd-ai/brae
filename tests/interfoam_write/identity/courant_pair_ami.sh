#!/usr/bin/env bash
# The gate: the device loop's Courant number with a coupled pair's faces in each cell's flux sum
# (courant_pair_faces.sh has the defect and the fix, on a plain cyclic) where the pair is NOT a plain cyclic:
# RAS/damBreakLeakage's W row (a cyclicACMI pair, whose faces share their cells with a non-overlap wall's) and
# RAS/mixerVesselAMI's (a cyclicAMI pair on a mesh that turns: the pair's flux is made relative with its own
# mesh flux, and each face has several neighbours).
# The cell check of that gate holds the device's sum to the host's sum OF THE SAME ARRAYS, the pair's flux
# among them: it sees a face left out or misplaced, and cannot see a pair flux that is not OpenFOAM's -- which
# is what differs between the three kinds of pair (the review's finding, 2026-10-05). So here the oracle that
# matters is OpenFOAM's own log, `Courant Number mean: .. max: ..` at the top of every step at the row's 17
# digits: brae's second step's (BRAE_CONTROL_COURANT_CHECK=1 prints them) is held to it, with the cell check
# running beside it.
# THREE CONTROLS, one line. The pair left out (BRAE_CONTROL_COURANT_PAIR_LEFT_OUT=1) on each row: the cell
# check must stop the run and the mean it prints must be off OpenFOAM's. And on the AMI row the pair's flux
# left ABSOLUTE after the move (BRAE_CONTROL_DEVICE_PAIR_PHI_ABSOLUTE=1, the pressure step's own control): the
# cell check PASSES -- both of its sums read that array -- and it is OpenFOAM's log that the number is off.
# DOES NOT CLAIM the interface number on these rows (courant_pair_interface.sh, on the cyclic), nor deltaT
# under adjustTimeStep: the W rows pin the step.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakLeakage of > "$W/ca_stage.txt" 2>&1
wcase mixerVesselAMI of >> "$W/ca_stage.txt" 2>&1
[ -d "$W/w_of_damBreakLeakage" ] && [ -d "$W/w_of_mixerVesselAMI" ] \
    || { say "damBreakLeakage or mixerVesselAMI did not stage" FAIL; finish "Courant number on AMI pairs"; }
for v in acmi:damBreakLeakage ami:mixerVesselAMI leftout:damBreakLeakage amileftout:mixerVesselAMI \
         amiabsolute:mixerVesselAMI; do
    e="$W/ca_${v%%:*}"
    o="$W/w_of_${v#*:}"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case ${v%%:*} in
        acmi|ami) runbrae "$e" device BRAE_CONTROL_COURANT_CHECK=1 ;;
        # the controls are expected to stop: run them directly, runbrae would end the gate
        leftout|amileftout)
            ( cd "$e" && BRAE_CONTROL_COURANT_CHECK=1 BRAE_CONTROL_COURANT_PAIR_LEFT_OUT=1 \
                  "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
        # ...but not this one, which the cell check cannot see
        amiabsolute)
            ( cd "$e" && BRAE_CONTROL_COURANT_CHECK=1 BRAE_CONTROL_DEVICE_PAIR_PHI_ABSOLUTE=1 \
                  "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
# MEASURED 2026-10-05, the mean's gap to OpenFOAM's log (the max's is smaller). damBreakLeakage 1.6e-08, the
# same in four runs: the column is at rest and the number is 4.3e-11, the flux being what two pinned solves leave
# of nothing -- the row's own fields are held at 4e-06 for the same reason. mixerVesselAMI 6.3e-14, 2.0e-14,
# 4.9e-14 and 1.3e-14 with the max up to 1.8e-13: the moving pair's sums do not reproduce run to run. Each
# bound is a decade above the worst seen. The controls: left out, the number is 1.5e-02 off on the ACMI row
# (its cell 4.1e-01 of the largest sum) and 7.2e-03 on the AMI row (1.8e-02); the AMI pair's flux left
# absolute, 9.9e-04 off with every cell held.
BOUND_ACMI=2e-07
BOUND_AMI=2e-12
mark="the coupled pair's .* faces are in each cell's flux sum"
line="Courant check: [1-9][0-9]* calls with a flux compared"
# row <tag> <key> <bound> <what>: one row's two oracles
row()
{
    local log="$W/ca_$1/log.brae" o="$W/w_of_$2" worst gap
    worst=$(grep -a "$line" "$log" | tail -1 | sed -E 's/.*worst cell is ([0-9.e+-]+) of.*/\1/')
    gap=$(courantOff "$log" "$o/log.interFoam" "Courant Number")
    local what="$4: cells within ${worst:-?} of the host's, the number within $gap of OpenFOAM's log"
    grep -q "$mark" "$log" && [ -n "$worst" ] && within "$gap" "$3" \
        && [ "$(timedirs "$W/ca_$1")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
}
row acmi damBreakLeakage "$BOUND_ACMI" "[device] damBreakLeakage (cyclicACMI)"
row ami mixerVesselAMI "$BOUND_AMI" "[device] mixerVesselAMI (cyclicAMI, turning)"
# leftout <tag> <key> <bound>: "ok <far> <gap>" where the left-out control of a row stopped at the cell check,
# said so, named no pair as summed, and printed a number off OpenFOAM's; "no <far> <gap>" otherwise
leftout()
{
    local log="$W/ca_$1/log.brae" far gap
    far=$(grep -a "COURANT_CHECK: cell .* whole flux sum is" "$log" | head -1 \
          | sed -E 's/.*faces: ([0-9.e+-]+) of the largest.*/\1/')
    gap=$(courantOff "$log" "$W/w_of_$2/log.interFoam" "Courant Number")
    [ "$(cat "$W/ca_$1/exit.txt")" != 0 ] \
        && grep -q "CONTROL MODE: the coupled pair's faces are left out of the Courant" "$log" \
        && ! grep -q "$mark" "$log" && above "${far:-}" 4e-12 && above "$gap" "$3" \
        && echo "ok ${far:-?} $gap" || echo "no ${far:-?} $gap"
}
read -r v1 far1 gap1 <<< "$(leftout leftout damBreakLeakage "$BOUND_ACMI")"
read -r v2 far2 gap2 <<< "$(leftout amileftout mixerVesselAMI "$BOUND_AMI")"
log="$W/ca_amiabsolute/log.brae"
gap3=$(courantOff "$log" "$W/w_of_mixerVesselAMI/log.interFoam" "Courant Number")
v3=no
[ "$(cat "$W/ca_amiabsolute/exit.txt")" = 0 ] && grep -q "$line" "$log" && ! grep -q "COURANT_CHECK:" "$log" \
    && above "$gap3" "$BOUND_AMI" && v3=ok
what="CONTROL  the pair left out: the number $gap1 (ACMI) and $gap2 (AMI) off OpenFOAM's; its flux left"
what="$what absolute: cells pass, $gap3 off"
[ "$v1" = ok ] && [ "$v2" = ok ] && [ "$v3" = ok ] && say "$what" ok || say "$what" FAIL
finish "the device loop's Courant number sums a cyclicACMI's and a cyclicAMI's faces as surfaceSum does"
