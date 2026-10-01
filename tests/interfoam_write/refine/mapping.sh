#!/usr/bin/env bash
# The write gate: alpha.water_0 is alpha mapped with the mesh (Y1), with its two controls (Y2).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase damBreakWithObstacle
wcase oscillatingBox
wcase motorBike
for key in damBreakWithObstacle oscillatingBox motorBike; do
    [ -d "$W/w_of_$key" ] || { say "ARM Y  $key did not run in arm W" FAIL; continue; }
    amrule "$W/w_of_$key" && say "ARM Y  premise: OpenFOAM's $key alpha.water_0 is alpha mapped with the mesh" ok \
                           || say "ARM Y  premise: OpenFOAM's $key alpha.water_0 is alpha mapped with the mesh" FAIL
    for arm in $ARMS; do
        d="$W/w_br_${key}_$arm"
        [ -n "$(timedirs "$d")" ] || continue
        amrule "$d" && say "ARM Y  [$arm] $key: brae's alpha.water_0 is alpha mapped with the mesh" ok \
                    || say "ARM Y  [$arm] $key: brae's alpha.water_0 is alpha mapped with the mesh" FAIL
    done
done
if [ -d "$W/w_of_damBreakWithObstacle" ]; then
    for ctl in BRAE_CONTROL_AMR_NO_WRITE_COMPACT BRAE_CONTROL_AMR_ALPHA0_START; do
        want=polyMesh/refinementHistory
        [ $ctl = BRAE_CONTROL_AMR_ALPHA0_START ] && want=alpha.water_0
        d="$W/y_ctl_$ctl"
        mkdir -p "$d"
        cp -r "$W/w_of_damBreakWithObstacle/0" "$W/w_of_damBreakWithObstacle/constant" "$W/w_of_damBreakWithObstacle/system" "$d/"
        runbrae "$d" host "$ctl=1"
        python3 "$CMP" "$W/w_of_damBreakWithObstacle" "$d" $(timedirs "$W/w_of_damBreakWithObstacle") > "$W/cmp_y_$ctl.txt" 2>&1
        grep -qE "/$want +structure BAD" "$W/cmp_y_$ctl.txt" \
            && say "CONTROL  $ctl=1 fails $want" ok \
            || say "CONTROL  $ctl=1 fails $want" FAIL
    done
    # ...and Y1's rule itself goes red on the start-of-step copy
    amrule "$W/y_ctl_BRAE_CONTROL_AMR_ALPHA0_START" > /dev/null \
        && say "CONTROL  BRAE_CONTROL_AMR_ALPHA0_START=1 fails Y1's mapping rule" FAIL \
        || say "CONTROL  BRAE_CONTROL_AMR_ALPHA0_START=1 fails Y1's mapping rule" ok
fi
finish "arms Y1, Y2: the old level is mapped as OpenFOAM maps it"
