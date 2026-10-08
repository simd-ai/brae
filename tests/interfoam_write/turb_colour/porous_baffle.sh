#!/usr/bin/env bash
# The turbulence rule ACROSS A COUPLED PAIR, on RAS/damBreakPorousBaffle (a cyclic baffle through the column):
# k and epsilon swept in colour order on the GPU with the pair's contribution moved to the right-hand side at
# the top of every sweep, as OpenFOAM's sweep moves it (device_colour_gauss_seidel.cuh, bEffP), against real
# OpenFOAM. MEASURED 2026-10-06: the worst file 1.3e-10 (nut), so the bound is 1.5e-09. The CONTROL is the
# pair's: BRAE_CONTROL_COLOUR_GS_PAIR_DROPPED=1 forms its term with a zero neighbour, so the two sides are
# solved uncoupled -- 3.1e-01 on nut. What this is for, on RAS/damBreakLeakage refined to 36,288 cells: the
# pair kept the one-core order there, 0.24x of 20-core OpenFOAM.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
TC_CONTROL=BRAE_CONTROL_COLOUR_GS_PAIR_DROPPED=1
TC_CONTROL_SAYS="the pair's term left out"
turbcolour_gate damBreakPorousBaffle 1.5e-09 k epsilon
finish "damBreakPorousBaffle's k and epsilon in colour order, across its baffle, are OpenFOAM's"
