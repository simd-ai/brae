#!/usr/bin/env bash
# The write gate, a rigid body on the device: the control of the atmosphere's tangentialVelocity -- its refValue left off the device.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
x3_case noref
runbrae "$X3" device BRAE_CONTROL_DEVICE_PIOV_NOREF=1
python3 "$CMP" "$W/w_of_DTCHullMovingCoarse" "$X3" $x3t > "$W/cmp_x3_noref.txt" 2>&1
judge "control no refValue" "$W/cmp_x3_noref.txt" "$BOUND_X3_FIELDS" "$W/w_of_DTCHullMovingCoarse/log.interFoam" > "$W/x3_noref.txt"
grep -qE "over the bound: [0-9.e+-]+/U " "$W/x3_noref.txt" \
    && say "CONTROL  BRAE_CONTROL_DEVICE_PIOV_NOREF=1 puts DTCHullMoving's U over the bound" ok \
    || { say "CONTROL  BRAE_CONTROL_DEVICE_PIOV_NOREF=1 puts DTCHullMoving's U over the bound" FAIL; cat "$W/x3_noref.txt"; }
finish "arm X3 control: a missing refValue is seen"
