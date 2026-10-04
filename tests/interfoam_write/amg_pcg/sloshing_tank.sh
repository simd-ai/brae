#!/usr/bin/env bash
# The pressure rule's gate on laminar/sloshingTank2D: a moving mesh, p_rgh on `solver GAMG` and p_rghFinal on
# `solver PCG; preconditioner GAMG` -- both of the device's GAMG branches. MEASURED 2026-10-03: worst 1.2e-06
# (phi at the first step, 1.4e-05 absolute), 1.6e-06 once pcorr runs the AMG-PCG too; part of it OpenFOAM's:
# its pinned GAMG stops at the 1000-iteration cap at 1.8e-13 where the AMG-PCG reaches 1e-13. Control: p_rgh
# 2.2e+00.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
amgpcg_gate sloshingTank2D 1.2e-05
finish "sloshingTank2D's pressure with brae's AMG-PCG is within its bound of OpenFOAM's GAMG"
