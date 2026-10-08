#!/usr/bin/env bash
# The DTCHull gate (localEuler): host controls on the momentum and ddtCorr consumers and the mass flux.
. "$(dirname "$0")/run.sh"
dtchull_gate control laminar:host:BRAE_CONTROL_LTS_SCALAR=ueqn laminar:host:BRAE_CONTROL_LTS_SCALAR=ddtcorr laminar:host:BRAE_CONTROL_RHOPHI_ALPHAFLUX=1
