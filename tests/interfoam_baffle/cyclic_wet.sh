#!/usr/bin/env bash
# The baffle gate: the cyclic pair with the column across it.
PROFILES="wallsWet cyclicWet" BAFFLE_GATE="cyclicWet" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_baffle_vs_openfoam.sh"
