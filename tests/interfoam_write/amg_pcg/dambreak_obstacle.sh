#!/usr/bin/env bash
# The pressure rule's gate on laminar/damBreakWithObstacle (GAMG on p_rgh, dynamicRefineFvMesh): the mesh's
# addressing changes, so the AMG hierarchy is rebuilt (DeviceAmgPcgCache, keyed on DeviceMesh::addressingId).
# MEASURED 2026-10-03: worst 1.8e-09 (rAU at the second step; phi 1.2e-09). Control: phi 6.6e+00.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
amgpcg_gate damBreakWithObstacle 1.8e-08
finish "damBreakWithObstacle's pressure with brae's AMG-PCG is within its bound of OpenFOAM's GAMG"
