#!/usr/bin/env bash
# The DTCHull gate (localEuler): device control on the outlet lag.
. "$(dirname "$0")/run.sh"
dtchull_gate control laminar:device:BRAE_CONTROL_OPMV_NOLAG=1
