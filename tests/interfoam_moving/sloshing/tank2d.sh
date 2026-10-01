#!/usr/bin/env bash
# The moving-mesh gate: the 2D sloshing tank: Euler, CrankNicolson, correctPhi.
. "$(dirname "$0")/../run.sh"
moving_gate sloshing2D:sloshing2DStatic sloshing2DCN:sloshing2D sloshing2DCorrectPhi:sloshing2DStatic
