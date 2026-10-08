#!/usr/bin/env bash
# The DTCHull gate (localEuler): host controls on setRDeltaT (no smoothing, no damping) and the alpha consumer.
. "$(dirname "$0")/run.sh"
dtchull_gate control laminar:host:BRAE_CONTROL_LTS_NOSMOOTH=1 laminar:host:BRAE_CONTROL_LTS_NODAMP=1 laminar:host:BRAE_CONTROL_LTS_SCALAR=alpha
