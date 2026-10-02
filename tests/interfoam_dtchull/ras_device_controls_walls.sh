#!/usr/bin/env bash
# The DTCHull gate (localEuler): device controls on the rough wall function and the cached grad(U).
. "$(dirname "$0")/run.sh"
dtchull_gate control ras:device:BRAE_CONTROL_NUTK_SMOOTH_DEVICE=1 ras:device:BRAE_CONTROL_NUTKROUGH_NOHISTORY=1 ras:device:BRAE_CONTROL_GRADU_UNCACHED=1
