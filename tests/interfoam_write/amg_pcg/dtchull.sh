#!/usr/bin/env bash
# The pressure rule's gate on RAS/DTCHull (845,536 cells, GAMG with DIC on p_rgh, localEuler): brae's AMG-PCG
# against OpenFOAM's GAMG. MEASURED 2026-10-03: worst 2.3e-09 (omega at step 2; k 1.8e-09, U 5.7e-10) where
# the GAMG port is 5.9e-13. Control (one iteration a solve): p_rgh 4.8e+00. The pressure solve 517 -> 190 ms
# a step, and the benchmark 0.87 -> 0.48 s an iteration (6.5x serial OpenFOAM).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
amgpcg_gate DTCHull 2.3e-08
finish "DTCHull's pressure with brae's AMG-PCG is within its bound of OpenFOAM's GAMG"
