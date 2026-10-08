#!/usr/bin/env bash
# The cyclicAMI pair on the device loop: static pair, controls on alpha's boundary, the pair's sub-cycled mass flux and ddtCorr.
. "$(dirname "$0")/run.sh"
ami_control static BRAE_CONTROL_DEVICE_ALPHA_TOP_EVALUATE=1 BRAE_CONTROL_DEVICE_PAIR_RHOPHI_LAST=1 BRAE_CONTROL_DEVICE_AMI_DDTCORR=1
