#!/usr/bin/env bash
# The write gate, a rigid body on the device: the control of load.sh -- the load taken from the stale host copies.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
x3_case stale
runbrae "$X3" device BRAE_CONTROL_DEVICE_BODY_STALE=1
python3 "$CMP" "$W/w_of_DTCHullMovingCoarse" "$X3" $x3t > "$W/cmp_x3_stale.txt" 2>&1
x3state "$W/cmp_x3_stale.txt" > "$W/x3_stale.txt" \
    && { say "CONTROL  BRAE_CONTROL_DEVICE_BODY_STALE=1 puts the body's state over the bound" FAIL; cat "$W/x3_stale.txt"; } \
    || { say "CONTROL  BRAE_CONTROL_DEVICE_BODY_STALE=1 puts the body's state over the bound" ok; cat "$W/x3_stale.txt"; }
finish "arm X3 control: a stale load is seen"
