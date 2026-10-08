#!/usr/bin/env bash
# The refine-and-move gate: oscillatingBox over sixty steps, through OpenFOAM's first unrefinement.
AMR_MOTION_PART=unrefine exec "$(cd "$(dirname "$0")/.." && pwd)/interfoam_amr_motion_vs_openfoam.sh"
