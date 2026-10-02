#!/usr/bin/env bash
# The baffle gate: a cyclic baffle pair, against the walled dam.
PROFILES="walls cyclic" BAFFLE_GATE="cyclic" exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_baffle_vs_openfoam.sh"
