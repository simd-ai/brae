#!/usr/bin/env bash
# The mangrove gate: the density-weighted k-epsilon.
MANGROVE_PART="densityVariable" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_mangrove_vs_openfoam.sh"
