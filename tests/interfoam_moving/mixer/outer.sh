#!/usr/bin/env bash
# The moving-mesh gate: the tube mixer under outer correctors and under correctPhi.
. "$(dirname "$0")/../run.sh"
moving_gate mixerOuter:mixerStatic mixerOuterOnce:mixerStatic mixerCorrectPhi:mixerStatic
