#!/usr/bin/env bash
# The write gate: the device's constrainHbyA takes U's stored patch values (DTCHull's outlet).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
v=$(outlet_face BRAE_NO_CONTROL_SET=1)
grep -q "outletPhaseMeanVelocity" "$W/w_of_DTCHull/0/U" \
    && say "the fixture's outlet is an outletPhaseMeanVelocity, whose coefficients the assembly moves" ok \
    || say "the fixture's outlet is an outletPhaseMeanVelocity, whose coefficients the assembly moves" FAIL
within "$v" "$BOUND_OUTLET_FACE" \
    && say "[device] the outlet's phiHbyA at iteration two is the host's, face by face: $v (bound $BOUND_OUTLET_FACE)" ok \
    || say "[device] the outlet's phiHbyA at iteration two is the host's, face by face: $v (bound $BOUND_OUTLET_FACE)" FAIL
finish "the device's constrainHbyA takes U's stored patch values"
