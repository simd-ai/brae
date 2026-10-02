#!/usr/bin/env bash
# The DTCHull gate (localEuler): device control on a stale cached grad(U).
. "$(dirname "$0")/run.sh"
dtchull_gate control ras:device:BRAE_CONTROL_GRADU_STALE=1
