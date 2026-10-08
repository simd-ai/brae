#!/usr/bin/env bash
# The cyclicAMI pair on the device loop: static pair, controls on the pair's non-orthogonal correction, the predictor's force and the limited grad(U).
. "$(dirname "$0")/run.sh"
ami_control static BRAE_CONTROL_DEVICE_PAIR_NO_NONORTH=1 BRAE_CONTROL_DEVICE_PREDICTOR_NO_PAIR_FORCE=1 BRAE_CONTROL_DEVICE_NONORTH_GRADU_UNLIMITED=1
