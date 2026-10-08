#!/usr/bin/env bash
# The DTCHull gate (localEuler): device controls on the closure under localEuler and its linearUpwind.
. "$(dirname "$0")/run.sh"
dtchull_gate control ras:device:BRAE_CONTROL_LTS_SCALAR=turbulence ras:device:BRAE_CONTROL_SST_LU_OFF=1 ras:device:BRAE_CONTROL_SST_LU_UNLIMITED=1
