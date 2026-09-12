#!/usr/bin/env bash
# SpalartAllmaras on the MIRRORED _cpp path, end to end, against real OpenFOAM. No CUDA in this gate.
#
# airFoil2D is the SA tutorial: freestreamVelocity/freestreamPressure far field, a fixedValue nuTilda
# wall with nutUSpaldingWallFunction on nut, and `bounded Gauss linearUpwind grad(...)` on BOTH div(phi,U)
# and div(phi,nuTilda) -- the turbulence scalar's own linearUpwind, which no other case here exercises.
#
# THE ORACLE IS REAL OPENFOAM: validation/airFoil2D/log.simpleFoam and its converged 500.
#
# WHAT THIS GATE COST, because every one of these was invisible until the case was run END TO END and each
# was found by localizing a residual rather than by reading code:
#
#   1. nut's WALL value. correctNut writes nut_ = nuTilda*fv1 as a field assignment, and nuTilda is
#      fixedValue ZERO at a wall -- so the assignment leaves nut_wall = 0. OpenFOAM then runs
#      correctBoundaryConditions(), and nutUSpaldingWallFunction OVERWRITES that with Spalding's law
#      (~4.5e-03 here). Taking the assignment removes the wall's entire eddy viscosity and with it the
#      wall shear: 25% of the momentum residual sat on those 78 faces, and fixing it moved the end-to-end
#      agreement by 268x on U, 598x on p and 35x on nuTilda in one step.
#   2. The freestream valueFraction was never recomputed. OF rebuilds vf = 0.5 -/+ 0.5*(U.n)/|U| every
#      updateCoeffs; brae left it at the 0.5 seed, making every far-field face a half-and-half blend
#      regardless of whether it was inflow or outflow.
#   3. mixed evaluate() is a BLEND, lerp(patchInternalField, refValue, vf), not refValue outright. Taking
#      refValue put the whole far field at freestreamValue.
#
# 2 and 3 together took the momentum residual from 372x OpenFOAM's to 47x and pressure from 3551x to 136x,
# BEFORE 1 closed the rest.
#
# THE CONTROL is laminar: the model dropped, nuEff held at the molecular value. It must fail, or the gate
# is measuring a case that does not care about its closure.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SA_CPP_BIN:-$ROOT/build/test_simple_sa_cpp}"
[ -x "$BIN" ] || { echo "SKIP: no test_simple_sa_cpp at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/500" ] || { echo "SKIP: no OpenFOAM converged state at $SRC/500"; exit 77; }

echo "== SA on: must match OpenFOAM end to end =="
# The bound is set just above where brae actually lands (U 6.2e-05, p 7.5e-05, nuTilda/nut 1.3e-02), with
# nuTilda looser than U/p because it is the transported scalar rather than a derived one.
SA_CPP_TOL=3e-02 "$BIN" "$SRC" 0 500 500 || { echo "FAIL: the _cpp SA did not match OpenFOAM end to end"; exit 1; }

echo "== control: laminar must NOT match =="
out=$(SIMPLE_SA_LAMINAR=1 SA_CPP_TOL=3e-02 "$BIN" "$SRC" 0 500 500 2>&1)
rc=$?
echo "$out" | tail -2
if [ $rc -eq 0 ]; then
    echo "FAIL: the case passes WITHOUT the turbulence model -- this gate measures nothing"
    exit 1
fi
echo "  ok:   the control fails, so the gate is measuring the model"
