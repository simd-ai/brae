#!/usr/bin/env bash
# The moving-mesh gate: the device control of floating/cn -- the closure's CrankNicolson ddt on the static branch.
. "$(dirname "$0")/../run.sh"
moving_control floatingShipped:floatingShippedStatic
