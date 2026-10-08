#!/usr/bin/env bash
# The les gate: k solved with PBiCG.
LES_PART="pbicg" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_les_vs_openfoam.sh"
