#!/usr/bin/env bash
# The DTCHull gate (localEuler): device controls on the momentum and ddtCorr consumers and the outlet (frozen).
. "$(dirname "$0")/run.sh"
dtchull_gate control laminar:device:BRAE_CONTROL_LTS_SCALAR=ueqn laminar:device:BRAE_CONTROL_LTS_SCALAR=ddtcorr laminar:device:BRAE_CONTROL_OPMV_FROZEN=1
