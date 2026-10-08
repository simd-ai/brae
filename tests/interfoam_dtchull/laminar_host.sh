#!/usr/bin/env bash
# The DTCHull gate (localEuler): the laminar staging on the host loop.
. "$(dirname "$0")/run.sh"
dtchull_gate run laminar:host
