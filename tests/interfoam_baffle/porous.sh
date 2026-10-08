#!/usr/bin/env bash
# The baffle gate: a porousBafflePressure pair, against the plain cyclic.
PROFILES="cyclic porous" BAFFLE_GATE="porous" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_baffle_vs_openfoam.sh"
