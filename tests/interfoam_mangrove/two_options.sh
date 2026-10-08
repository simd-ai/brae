#!/usr/bin/env bash
# The mangrove gate: two options of each type over one cellZone.
MANGROVE_PART="twoOptions" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_mangrove_vs_openfoam.sh"
