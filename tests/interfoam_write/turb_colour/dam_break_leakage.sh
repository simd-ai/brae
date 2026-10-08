#!/usr/bin/env bash
# The turbulence rule on RAS/damBreakLeakage (a cyclicACMI pair whose open fraction moves): k and epsilon swept
# in colour order on the GPU, against real OpenFOAM. MEASURED 2026-10-06: the worst file 4.1e-07 (phi), beside
# 3.6e-07 in the case's own cell order -- this case's distance from OpenFOAM is its own (lib.sh: its column
# stands at rest behind the shut baffle at step 2), so the bound is the row's, 4e-06. The control is one sweep a
# solve, 1.1e-01 on nut. The PAIR is not witnessed here -- in the row's two steps its term is too small to
# move a file (measured: left out, the worst file is the same 4.1e-07) -- porous_baffle.sh holds it.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
turbcolour_gate damBreakLeakage 4e-06 k epsilon
finish "damBreakLeakage's k and epsilon in colour order are OpenFOAM's"
