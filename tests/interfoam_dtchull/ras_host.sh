#!/usr/bin/env bash
# The DTCHull gate (localEuler): the tutorial as shipped (kOmegaSST) on the host loop.
. "$(dirname "$0")/run.sh"
dtchull_gate run ras:host
