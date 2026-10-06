#!/usr/bin/env bash
# The turbulence rule on RAS/weirOverflow (2-D, kEpsilon): k and epsilon swept in colour order on the GPU,
# against real OpenFOAM. MEASURED 2026-10-06: the worst file 2.6e-12 (p), so the bound is 3e-11; the control
# (one sweep a solve) 2.5e-05 on epsilon. What the rule is for, on this case refined to 81,280 cells: the step
# 1,520 -> 75 ms, 0.12x -> 2.47x of 20-core OpenFOAM.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
turbcolour_gate weirOverflow 3e-11 k epsilon
finish "weirOverflow's k and epsilon in colour order are OpenFOAM's"
