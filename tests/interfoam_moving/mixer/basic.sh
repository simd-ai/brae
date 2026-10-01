#!/usr/bin/env bash
# The moving-mesh gate: the tube mixer under solidBody motion: as shipped, with correctors, with a predictor.
. "$(dirname "$0")/../run.sh"
moving_gate mixer:mixerStatic mixerCorr:mixerStatic mixerPred:mixerStatic
