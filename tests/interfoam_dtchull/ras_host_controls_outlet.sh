#!/usr/bin/env bash
# The DTCHull gate (localEuler): host controls on the outlet lag and the cached grad(U).
. "$(dirname "$0")/run.sh"
dtchull_gate control ras:host:BRAE_CONTROL_OPMV_NOLAG=1 ras:host:BRAE_CONTROL_GRADU_UNCACHED=1
