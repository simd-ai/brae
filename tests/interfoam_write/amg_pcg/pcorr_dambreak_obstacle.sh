#!/usr/bin/env bash
# The pressure rule for pcorr, on laminar/damBreakWithObstacle: CorrectPhi assembles pcorr on the host and the
# device arm solves its GAMG entry with the AMG-PCG on the GPU (DevicePcorrSolver), on the refined mesh before
# the device loop has rebuilt its own. MEASURED 2026-10-03: pcorr's solver changes 9 written files against the
# GAMG port; the fields stay 1.8e-09 from OpenFOAM (dambreak_obstacle.sh holds them). Control (pcorr's solve
# alone stopped after one iteration): rAU 5.9e-01. sloshingTank2D cannot witness pcorr: its pcorr converges in
# one iteration, and the control reads the default's 1.6e-06.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
amgpcg_pcorr_gate damBreakWithObstacle 1.8e-08
finish "damBreakWithObstacle's pcorr runs brae's AMG-PCG, and the gate sees it"
