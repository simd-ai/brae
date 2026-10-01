#!/usr/bin/env bash
# The moving-mesh gate: the closed dam break: adjustPhi on a closed domain.
. "$(dirname "$0")/../run.sh"
moving_gate closedDamBreak:closedRef1e5 closedDamBreakInitU:closedDamBreak
