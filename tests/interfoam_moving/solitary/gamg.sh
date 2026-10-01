#!/usr/bin/env bash
# The moving-mesh gate: the solitary wave maker with a GAMG pressure solve.
. "$(dirname "$0")/../run.sh"
moving_gate solitaryGamg:solitaryStatic
