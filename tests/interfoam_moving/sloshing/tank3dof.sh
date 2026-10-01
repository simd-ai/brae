#!/usr/bin/env bash
# The moving-mesh gate: the sloshing tanks under three and six degrees of freedom.
. "$(dirname "$0")/../run.sh"
moving_gate sloshing2D3DoF:sloshing2D3DoFStatic sloshing3D3DoF:sloshing3D3DoFStatic sloshing3D6DoF:sloshing3D6DoFStatic
