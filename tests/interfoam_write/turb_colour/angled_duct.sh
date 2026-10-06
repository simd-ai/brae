#!/usr/bin/env bash
# The turbulence rule on RAS/angledDuct (3-D, kEpsilon, a porous zone): k and epsilon swept in colour order on
# the GPU, against real OpenFOAM. MEASURED 2026-10-06: the worst file 1.0e-09 (phi), which is this case's own
# distance from OpenFOAM -- the case's own cell order measures 1.0e-09 on the same file -- so the bound is
# 1e-08; the control (one sweep a solve) 3.0e-02 on nut.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
turbcolour_gate angledDuct 1e-08 k epsilon
finish "angledDuct's k and epsilon in colour order are OpenFOAM's"
