#!/usr/bin/env bash
# The cyclicAMI pair on the device loop: rotating pair, controls on the pair's refresh after a move, its relative flux and its non-orthogonal correction.
. "$(dirname "$0")/run.sh"
ami_control rotating BRAE_CONTROL_DEVICE_PAIR_STALE=1 BRAE_CONTROL_DEVICE_PAIR_PHI_ABSOLUTE=1 BRAE_CONTROL_DEVICE_PAIR_NO_NONORTH=1
