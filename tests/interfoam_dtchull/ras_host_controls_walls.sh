#!/usr/bin/env bash
# The DTCHull gate (localEuler): host controls on the rough wall function and the outlet (frozen).
. "$(dirname "$0")/run.sh"
dtchull_gate control ras:host:BRAE_CONTROL_NUTK_SMOOTH=1 ras:host:BRAE_CONTROL_NUTKROUGH_NOHISTORY=1 ras:host:BRAE_CONTROL_OPMV_FROZEN=1
