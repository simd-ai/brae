#!/usr/bin/env bash
# The leakage gate: the baffle that opens at t = 0.5, against the closed and the all-open controls.
LEAKAGE_PART="leak" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_leakage_vs_openfoam.sh"
