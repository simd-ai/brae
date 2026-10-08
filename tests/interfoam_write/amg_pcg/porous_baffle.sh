#!/usr/bin/env bash
# The pressure rule's gate on RAS/damBreakPorousBaffle: a cyclic pair with a JUMP (porousBafflePressure) inside
# the AMG-PCG's captured solve. MEASURED 2026-10-03: worst 6.2e-11 (phi). Control (one AMG-PCG iteration a
# solve): phi 7.0e+00. FAIL-PROOF, the defect this file exists for: the captured graph kept the FIRST solve's
# pointer to the pair's jump, which this solver's caller rebuilds every solve (PCGGraphCache's key and copies,
# device_amg.cuh) -- every later solve read a stale jump and phi stood 5.6e-03 from OpenFOAM while each solve
# reported 1e-13.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
amgpcg_gate damBreakPorousBaffle 6.2e-10
finish "damBreakPorousBaffle's pressure with brae's AMG-PCG is within its bound of OpenFOAM's DICPCG"
