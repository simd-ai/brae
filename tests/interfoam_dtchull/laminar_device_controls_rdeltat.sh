#!/usr/bin/env bash
# The DTCHull gate (localEuler): device controls on setRDeltaT and the alpha consumer.
. "$(dirname "$0")/run.sh"
dtchull_gate control laminar:device:BRAE_CONTROL_LTS_NOSMOOTH=1 laminar:device:BRAE_CONTROL_LTS_NODAMP=1 laminar:device:BRAE_CONTROL_LTS_SCALAR=alpha
