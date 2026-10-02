#!/usr/bin/env bash
# The DTCHull gate (localEuler): the laminar staging on the device loop, ten steps and one.
. "$(dirname "$0")/run.sh"
dtchull_gate run laminar:device
