#!/usr/bin/env bash
# The les gate: the uncorrected snGrad against the orthogonal one.
LES_PART="uncorrected" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_les_vs_openfoam.sh"
