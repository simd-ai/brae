#!/usr/bin/env bash
# The moving-mesh gate: the control of solitary/cn_outer -- phi.oldTime() left at the flux of the previous step.
. "$(dirname "$0")/../run.sh"
moving_control solitaryOuterCN:solitaryShort
