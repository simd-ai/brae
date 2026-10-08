#!/usr/bin/env bash
# The leakage gate: the same baffle on a shaken mesh.
LEAKAGE_PART="moving" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_leakage_vs_openfoam.sh"
