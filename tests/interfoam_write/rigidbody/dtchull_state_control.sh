#!/usr/bin/env bash
# The write gate, a rigid body's state file: the control of dtchull_state.sh -- the state as the step began.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase DTCHullMovingCoarse of
[ -d "$W/w_of_DTCHullMovingCoarse" ] || { say "ARM Q  DTCHullMoving did not stage" FAIL; finish "arm Q"; }
d="$W/w_ctl_rbold"
mkdir -p "$d"
cp -r "$W/w_of_DTCHullMovingCoarse/0" "$W/w_of_DTCHullMovingCoarse/constant" "$W/w_of_DTCHullMovingCoarse/system" "$d/"
runbrae "$d" host BRAE_CONTROL_RBSTATE_OLD=1
# the VALUE check must be what trips: at 0.0001 the step's start is rest, `2 { 0 }`, which the text
# check alone would fail -- so the control is held to an entry over the bound at 0.0002, where both
# sides are paren lists
rbstate "$W/w_of_DTCHullMovingCoarse" "$d" "$BOUND_RB_DTC" > "$W/rbold.txt"
grep -qE "^ +0\.0002/(q|qDot|qDdot): .* over " "$W/rbold.txt" \
    && { say "CONTROL  BRAE_CONTROL_RBSTATE_OLD=1 puts DTCHullMoving's 0.0002 entries over the bound" ok; grep -E "^ +0\.0002/" "$W/rbold.txt" | head -3; } \
    || { say "CONTROL  BRAE_CONTROL_RBSTATE_OLD=1 puts DTCHullMoving's 0.0002 entries over the bound" FAIL; cat "$W/rbold.txt"; }
finish "arm Q control: the old state is seen"
