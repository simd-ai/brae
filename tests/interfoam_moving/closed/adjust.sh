#!/usr/bin/env bash
# The moving-mesh gate: the dam with an adjustable outflow, zeroGradient and inletOutlet.
. "$(dirname "$0")/../run.sh"
moving_gate closedAdjZG:closedDamBreak closedAdjIO:closedAdjZG
