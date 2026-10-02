#!/usr/bin/env bash
# The mangrove gate: the mangroves' momentum and turbulence sources, against both switched off.
MANGROVE_PART="mangrove" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_mangrove_vs_openfoam.sh"
