#!/usr/bin/env bash
# The les gate: grad(pcorr) under leastSquares.
LES_PART="pcorrGrad" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_les_vs_openfoam.sh"
