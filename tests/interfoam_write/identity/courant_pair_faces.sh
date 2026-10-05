#!/usr/bin/env bash
# The gate: the device loop's Courant number with the faces of a COUPLED PAIR in each cell's flux sum, on
# RAS/damBreakPorousBaffle's W row (a cyclic pair with a porous jump) and, as the mesh without one,
# laminar/capillaryRise's.
# CourantNo.H and alphaCourantNo.H take fvc::surfaceSum(mag(phi)), which adds every patch's faces to their face
# cells -- a cyclic, cyclicAMI or cyclicACMI patch included (fvcSurfaceIntegrate.C:170-182). The device keeps a
# pair's faces apart from its boundary list, and its Courant numbers were summed without them: a cell on the
# pair had a short sum, so a step limited there would have been longer than OpenFOAM's. brae's host loop always
# had them. FOUND 2026-10-05 by reading: every gate on a mesh with a pair pins the step (adjustTimeStep no), and
# none compared a Courant number with OpenFOAM's.
# TWO ORACLES. OpenFOAM's own log, which prints `Courant Number mean: .. max: ..` at the top of every step
# whatever adjustTimeStep says, at the row's 17 digits: brae's second step's (BRAE_CONTROL_COURANT_CHECK=1
# prints them) held to it. MEASURED 2026-10-05, the same in three runs: 2.4e-14 on the baffle row and 3.7e-13
# on capillaryRise (the mean; the max 2.3e-16 and 1.9e-14) -- what one step's flux differs by; the bounds are
# a decade above, 3e-13 and 4e-12. And, cell for cell, the host's surfaceSumMagPhi of the same flux with every
# patch in face order: each cell's sum on the device held to it at every call that has a flux (one, on these
# rows: the first step starts from rest), to 4e-15 of the largest cell's -- the faces are added in another
# order, 2.0e-16 here and 3.6e-16 on mixerVesselAMI's 83,656 coupled faces, a row this gate does not run.
# The CONTROL leaves the pair out again (BRAE_CONTROL_COURANT_PAIR_LEFT_OUT=1): the cell check must stop the
# run far above its bound (8.8e-02 of the largest sum) and the mean it prints is off OpenFOAM's (4.6e-03).
# DOES NOT CLAIM: the INTERFACE number's pair term -- as shipped nothing but air reaches the baffle in these
# two steps, so no cell of the pair is in the band; courant_pair_interface.sh stages water across it. Nor a
# cyclicAMI or cyclicACMI pair, whose flux is formed differently (courant_pair_ami.sh). Nor deltaT on a pair
# under adjustTimeStep: the W row pins the step, and no gate lets one move on such a mesh.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakPorousBaffle of > "$W/cp_stage.txt" 2>&1
wcase capillaryRise of >> "$W/cp_stage.txt" 2>&1
[ -d "$W/w_of_damBreakPorousBaffle" ] && [ -d "$W/w_of_capillaryRise" ] \
    || { say "damBreakPorousBaffle or capillaryRise did not stage" FAIL; finish "Courant number with the pair"; }
for v in pair:damBreakPorousBaffle nopair:capillaryRise leftout:damBreakPorousBaffle; do
    e="$W/cp_${v%%:*}"
    o="$W/w_of_${v#*:}"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case ${v%%:*} in
        pair|nopair) runbrae "$e" device BRAE_CONTROL_COURANT_CHECK=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        leftout)     ( cd "$e" && BRAE_CONTROL_COURANT_CHECK=1 BRAE_CONTROL_COURANT_PAIR_LEFT_OUT=1 \
                           "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
mark="the coupled pair's .* faces are in each cell's flux sum"
line="Courant check: [1-9][0-9]* calls with a flux compared"
worst=$(grep -a "$line" "$W/cp_pair/log.brae" | tail -1 | sed -E 's/.*worst cell is ([0-9.e+-]+) of.*/\1/')
gap=$(courantOff "$W/cp_pair/log.brae" "$W/w_of_damBreakPorousBaffle/log.interFoam" "Courant Number")
what="[device] the baffle's pair in the sum: cells within ${worst:-?} of the host's, the number within $gap"
what="$what of OpenFOAM's log"
grep -q "$mark" "$W/cp_pair/log.brae" && [ -n "$worst" ] && within "$gap" 3e-13 \
    && [ "$(timedirs "$W/cp_pair")" = "$(timedirs "$W/w_of_damBreakPorousBaffle")" ] \
    && say "$what" ok || say "$what" FAIL
worst=$(grep -a "$line" "$W/cp_nopair/log.brae" | tail -1 | sed -E 's/.*worst cell is ([0-9.e+-]+) of.*/\1/')
gap=$(courantOff "$W/cp_nopair/log.brae" "$W/w_of_capillaryRise/log.interFoam" "Courant Number")
what="[device] a mesh without a pair, none named: cells within ${worst:-?} of the host's, the number"
what="$what within $gap"
! grep -q "$mark" "$W/cp_nopair/log.brae" && [ -n "$worst" ] && within "$gap" 4e-12 \
    && [ "$(timedirs "$W/cp_nopair")" = "$(timedirs "$W/w_of_capillaryRise")" ] \
    && say "$what" ok || say "$what" FAIL
# the control's distance, from the check's own message, and its printed number against OpenFOAM's
far=$(grep -a "COURANT_CHECK: cell .* whole flux sum is" "$W/cp_leftout/log.brae" | head -1 \
      | sed -E 's/.*faces: ([0-9.e+-]+) of the largest.*/\1/')
gap=$(courantOff "$W/cp_leftout/log.brae" "$W/w_of_damBreakPorousBaffle/log.interFoam" "Courant Number")
what="CONTROL  the pair left out: the check stops ${far:-?} of the largest sum out, the number $gap off"
what="$what OpenFOAM's"
[ "$(cat "$W/cp_leftout/exit.txt")" != 0 ] \
    && grep -q "CONTROL MODE: the coupled pair's faces are left out of the Courant" "$W/cp_leftout/log.brae" \
    && ! grep -q "$mark" "$W/cp_leftout/log.brae" && [ -n "$far" ] && above "$far" 4e-12 && above "$gap" 3e-10 \
    && say "$what" ok || say "$what" FAIL
finish "the device loop's Courant number sums a coupled pair's faces as surfaceSum does"
