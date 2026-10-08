#!/usr/bin/env bash
# The baffle gate: the porous pair with the column across it.
PROFILES="cyclicWet porousWet" BAFFLE_GATE="porousWet" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_baffle_vs_openfoam.sh"
