#!/usr/bin/env bash
# The pressure rule's gate on laminar/capillaryRise, whose p_rgh and pcorr name `PCG` with `DIC`: brae's AMG-PCG in
# their place, against OpenFOAM's DICPCG. MEASURED 2026-10-03: worst 3.8e-13 (phi at the first step). Control (one
# AMG-PCG iteration a solve): phi 3.1e-01. On this 8,000-cell mesh the case's own DIC on the device was 175 ms a
# step, OpenFOAM serial 16.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
amgpcg_gate capillaryRise 3.8e-12
finish "capillaryRise's pressure with brae's AMG-PCG is within its bound of OpenFOAM's DICPCG"
