#!/usr/bin/env bash
# The moving-mesh gate: the control of piston/outer_once -- the device moving the mesh at every outer corrector.
. "$(dirname "$0")/../run.sh"
moving_control pistonOuterOnce:piston
