#!/usr/bin/env bash
# pcorr on a smoothed-aggregation hierarchy of its own, on laminar/waves/waveMakerPiston: the mesh moves and
# keeps its topology, so the hierarchy is built once (DevicePcorrSolver::fixedTopology).
# MEASURED 2026-10-04 on the tutorial refined to 896,000 cells: pcorr 173 -> 48 iterations a solve, CorrectPhi
# 360 -> 179 ms a step, the same 30 steps to the same time. Here, at the write gate's pinned tolerances:
# 99 pcorr iterations against 196 on the hierarchy p_rgh uses, and the fields 6.4e-07 from OpenFOAM's DICPCG
# (Uf and phi; 6.5e-07 on the plain hierarchy). Control (pcorr's solve stopped after one iteration):
# alphaPhi0.water 2.3e+00.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
amgpcg_pcorr_sa_gate waveMakerPiston 6.4e-06
finish "waveMakerPiston's pcorr runs on its smoothed-aggregation hierarchy"
