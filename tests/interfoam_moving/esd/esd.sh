#!/usr/bin/env bash
# The moving-mesh gate: electrostaticDeposition: the relaxed alpha corrector, and without it.
. "$(dirname "$0")/../run.sh"
moving_gate esd:esdNoCorr esdNoCorr:esd
