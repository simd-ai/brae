#!/usr/bin/env bash
# The write gate, a rigid body on the device: the control of the smoothSolver's loop residual.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase DTCHullMovingCoarse of
o="$W/w_of_DTCHullMovingCoarse"
[ -d "$o" ] || { say "DTCHullMovingCoarse did not stage" FAIL; finish "k residual control"; }
# OpenFOAM's k solve at the second step never gets under its tolerance: it stalls at 1.9e-13 against 1e-13 and
# runs all 1000 sweeps. THE FIXTURE HOLDS THAT, or the control below measures nothing.
[ "$(grep -c 'Solving for k,.*No Iterations 1000' "$o/log.interFoam")" -ge 2 ] \
    && say "OpenFOAM's k solves run to the 1000-sweep cap on this fixture" ok \
    || say "OpenFOAM's k solves run to the 1000-sweep cap on this fixture" FAIL
# With source - A.psi in the loop (the device's residual until 2026-10-02) the device reads 3.2e-14 there and
# stops after 2 sweeps: MEASURED k 2.9e-10, against a worst file of 3.1e-12 with lduMatrix::residual.
e="$W/kres_ctl"
mkdir -p "$e"
cp -r "$o/0" "$o/constant" "$o/system" "$e/"
runbrae "$e" device BRAE_CONTROL_DEVICE_GS_RESIDUAL_AMUL=1
grep -q "CONTROL MODE" "$e/log.brae" \
    && say "CONTROL  BRAE_CONTROL_DEVICE_GS_RESIDUAL_AMUL=1 ran as a control" ok \
    || say "CONTROL  BRAE_CONTROL_DEVICE_GS_RESIDUAL_AMUL=1 ran as a control" FAIL
python3 "$CMP" "$o" "$e" $(timedirs "$o") > "$W/cmp_kres_ctl.txt" 2>&1
judge "control source - A.psi" "$W/cmp_kres_ctl.txt" 3.1e-11 "$o/log.interFoam" > "$W/kres_ctl.txt" \
    && { say "CONTROL  ...and it puts k over the coarse hull's device bound" FAIL; cat "$W/kres_ctl.txt"; } \
    || { grep -qE "over the bound: [0-9.e+-]+/k " "$W/kres_ctl.txt" \
             && say "CONTROL  ...and it puts k over the coarse hull's device bound" ok \
             || { say "CONTROL  ...and it puts k over the coarse hull's device bound" FAIL; cat "$W/kres_ctl.txt"; }; }
finish "the device smoothSolver's loop residual is lduMatrix::residual"
