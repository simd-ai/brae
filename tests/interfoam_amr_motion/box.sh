#!/usr/bin/env bash
# The refine-and-move gate: oscillatingBox over twenty steps, with OpenFOAM's own no-motion control.
AMR_MOTION_PART=box exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_amr_motion_vs_openfoam.sh"
