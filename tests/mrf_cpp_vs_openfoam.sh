#!/usr/bin/env bash
# MRF on the MIRRORED _cpp path, end to end, against real OpenFOAM. No CUDA anywhere in this gate.
#
# mixerVessel2D is a rotating-zone case: cellZone `rotor`, omega 104.72 rad/s about z, kEpsilon,
# `bounded Gauss limitedLinearV 1` momentum and `bounded Gauss limitedLinear 1` turbulence, every patch a
# wall or empty. simpleFoam reaches MRF in three places and this exercises all of them:
# correctBoundaryVelocity (UEqn.H:3), DDt (UEqn.H:8) and makeRelative (pEqn.H:5).
#
# THE CONTROL IS THE POINT. With the zones dropped and everything else identical the case has NO driving
# force at all -- the flow is quiescent and every field goes to zero, so the comparison against OpenFOAM
# reads exactly 1.000. MRF is not a correction here, it is 100% of the physics, and a gate that passes
# with it removed would be measuring nothing. Each hook was checked the same way while porting: removing
# DDt alone costs 158x on the momentum residual and removing makeRelative 30x on pressure.
#
# WHAT THE BOUND CAUGHT. It was first set at 2e-01 because brae reached only U 4.8e-02 / p 1.2e-02 /
# epsilon 1.7e-01, with initial residuals at OpenFOAM's own converged state 10x (U) and 52x (p) its own.
# Localizing that residual put 93.8% of it on ONE patch -- the rotor wall -- and the cause was brae's
# noSlip boundary condition, which returned a hardcoded ZERO from valueBoundaryCoeffs and re-zeroed its
# value on every evaluate(). That bakes in "this wall is stationary", which is true of every case that
# does not rotate and false of every case that does: MRF makes the rotor wall MOVE at Omega x r.
# OpenFOAM's noSlip is a fixedValue whose coefficients come from the live patch value, so the assignment
# sticks. Fixing it took U from 1.60e-04 to 1.53e-05 against OpenFOAM's 1.54e-05 -- 0.7% -- and the
# rotor's share of the residual from 93.8% to 3.6%.
#
# WHAT THIS CASE CANNOT TEST, and the precise reason: OpenFOAM's internalFaces are the faces with EITHER
# cell in the zone, not both. This zone DOES have an internal interface -- 96 faces of the 3024, so the
# two readings select different face sets -- yet they produce bit-identical answers, because those 96
# faces carry a frame flux of 6.4e-13 against 1.5e-04 on the zone's interior faces. The interface is a
# circle of constant radius about the rotation axis: its normals are radial, Omega x r is circumferential,
# and the dot product vanishes by geometry. Gating OR-vs-AND therefore needs a zone whose boundary is NOT
# a surface of revolution about its own axis, and this case must not be read as having done so.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MRF_CPP_BIN:-$ROOT/build/test_simple_mrf_cpp}"
[ -x "$BIN" ] || { echo "SKIP: no test_simple_mrf_cpp at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/500" ] || { echo "SKIP: no OpenFOAM converged state at $SRC/500"; exit 77; }

echo "== MRF on: must match OpenFOAM =="
MRF_CPP_TOL=3e-02 "$BIN" "$SRC" 0 500 500 || { echo "FAIL: the _cpp MRF did not match OpenFOAM end to end"; exit 1; }

echo "== control: MRF off must NOT match =="
out=$(SIMPLE_MRF_OFF=1 MRF_CPP_TOL=3e-02 "$BIN" "$SRC" 0 500 500 2>&1)
rc=$?
echo "$out" | tail -2
if [ $rc -eq 0 ]; then
    echo "FAIL: the case passes WITHOUT the MRF terms -- this gate measures nothing"
    exit 1
fi
echo "  ok:   the control fails, so the gate is measuring the frame terms"
