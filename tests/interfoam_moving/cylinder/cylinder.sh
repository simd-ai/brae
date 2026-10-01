#!/usr/bin/env bash
# The moving-mesh gate: the sloshing cylinder, and under correctPhi.
. "$(dirname "$0")/../run.sh"
moving_gate cylinder:cylinderStatic cylinderCorrectPhi:cylinderStatic
