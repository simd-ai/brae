#!/usr/bin/env bash
# The turbulence rule on RAS/motorBike (snappyHexMesh, kOmegaSST): k and omega swept in colour order on the
# GPU, against real OpenFOAM -- the SST closure's share of the rule, on a mesh that is not a block. MEASURED
# 2026-10-06: the worst file 1.3e-11 (nut), so the bound is 1.5e-10; the control (one sweep a solve) 1.1e-06
# on k. At the row's pinned 1e-13 the k solve takes 1,000 sweeps in either order (OpenFOAM's stalls at 1.08e-13).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
turbcolour_gate motorBike 1.5e-10 k omega
finish "motorBike's k and omega in colour order are OpenFOAM's"
