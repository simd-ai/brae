#!/usr/bin/env bash
# The write gate: the control of stored_u/outlet.sh -- constrainHbyA re-evaluating U's patches.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
v=$(outlet_face BRAE_CONTROL_DEVICE_HBYA_REEVALUATED=1)
grep -q "CONTROL MODE" "$W/taps.txt" \
    && say "CONTROL  BRAE_CONTROL_DEVICE_HBYA_REEVALUATED=1 ran as a control" ok \
    || say "CONTROL  BRAE_CONTROL_DEVICE_HBYA_REEVALUATED=1 ran as a control" FAIL
within "$CONTROL_OUTLET_FLOOR" "$v" \
    && say "CONTROL  ...and the outlet's phiHbyA leaves the host's: $v (floor $CONTROL_OUTLET_FLOOR)" ok \
    || say "CONTROL  ...and the outlet's phiHbyA leaves the host's: $v (floor $CONTROL_OUTLET_FLOOR)" FAIL
finish "stored_u control: a re-evaluated outlet is seen"
