#!/usr/bin/env bash
# The leakage gate: a restart across the opening.
LEAKAGE_PART="restart" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_leakage_vs_openfoam.sh"
