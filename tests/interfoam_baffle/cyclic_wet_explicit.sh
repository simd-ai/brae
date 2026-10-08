#!/usr/bin/env bash
# The baffle gate: the wet cyclic pair under the explicit MULES.
PROFILES="wallsWetExplicit cyclicWetExplicit" BAFFLE_GATE="cyclicWetExplicit" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_baffle_vs_openfoam.sh"
