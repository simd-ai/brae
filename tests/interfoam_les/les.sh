#!/usr/bin/env bash
# The les gate: LES kEqn with the smooth delta, as shipped.
LES_PART="les" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_les_vs_openfoam.sh"
