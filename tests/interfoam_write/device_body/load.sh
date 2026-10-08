#!/usr/bin/env bash
# The write gate, a rigid body on the device: the body's state and every file after a step with no write between.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
x3_case run
runbrae "$X3" device
[ "$(echo $(timedirs "$X3"))" = "$x3t" ] \
    && say "ARM X3 [device] DTCHullMoving written at the second step alone" ok \
    || say "ARM X3 [device] DTCHullMoving written at the second step alone" FAIL
python3 "$CMP" "$W/w_of_DTCHullMovingCoarse" "$X3" $x3t > "$W/cmp_x3.txt" 2>&1
x3state "$W/cmp_x3.txt" \
    && say "ARM X3 [device] the body's state after a step with no write between: OpenFOAM's within $BOUND_X3_STATE" ok \
    || say "ARM X3 [device] the body's state after a step with no write between: OpenFOAM's within $BOUND_X3_STATE" FAIL
judge "DTCHullMoving device, one write" "$W/cmp_x3.txt" "$BOUND_X3_FIELDS" "$W/w_of_DTCHullMovingCoarse/log.interFoam" \
    && say "ARM X3 [device] DTCHullMoving: every file's structure is OpenFOAM's, every value within $BOUND_X3_FIELDS" ok \
    || { say "ARM X3 [device] DTCHullMoving: every file's structure is OpenFOAM's, every value within $BOUND_X3_FIELDS" FAIL; grep -v RESULT "$W/cmp_x3.txt" | grep -B1 "^      " | head -12; }
finish "arm X3: the device hands the body the load it solved with"
