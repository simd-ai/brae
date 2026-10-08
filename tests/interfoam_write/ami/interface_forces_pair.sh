#!/usr/bin/env bash
# The interface-forces hook's face loops on the GPU on a mesh with a COUPLED pair, on RAS/mixerVesselAMI (894,950
# cells). Where a pair existed the hook formed the surface-tension flux and snGrad(rho) on the host whole, twice
# a step, and the momentum predictor's snGrad(p_rgh) was the host's on every mesh: MEASURED 2026-10-05, 33.8,
# 39.0 and 38.6 ms a step with their gradients; the hook 133.3 -> 28.5 with all three on the GPU. Now the
# internal faces are the device kernels' as on any mesh, the gradient their correction reads takes the pair's
# terms, the host forms the uncoupled patch faces alone, and the pair's own faces are the device's: snGrad across
# the pair (deviceCyclicSnGrad, fvc::snGrad's coupled branch) and sigma*K interpolated on the pair's weights.
# A coupled mesh's runs do not reproduce, so the check is in the code: BRAE_CONTROL_FORCES_CHECK=1 forms the
# host's whole as well at every call and compares every face -- internal, patch and pair -- within 3e-12 of each
# field's largest value (measured widest 2.7e-13, snGrad(rho) on the pair; the flux agrees to 5e-17), a pair's
# snGrad with an allowance for the rounding of a uniform field's weighted sum (p_rgh at rest at the first step).
# The CONTROL drops the non-orthogonal correction from the device's face loops
# (BRAE_CONTROL_FORCES_NO_CORRECTION=1): the flux is then 2.2e-01 of its largest value off.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase mixerVesselAMI of > "$W/fp_stage.txt" 2>&1
o="$W/w_of_mixerVesselAMI"
[ -d "$o" ] || { say "mixerVesselAMI did not stage" FAIL; finish "interface forces across a pair"; }
# arm <name> [env...]: brae on a copy of the staged case, the comparison on; never ends the gate
arm()
{
    local e="$W/fp_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_CONTROL_FORCES_CHECK=1 "$@" "$BIN" -case . -device > log.brae 2>&1
      echo $? > exit.txt )
}
arm device BRAE_X=1
arm nocorrection BRAE_CONTROL_FORCES_NO_CORRECTION=1
arm host BRAE_CONTROL_FORCES_PAIR_HOST=1
what="[device] the flux and snGrad(rho) from the GPU are the host's on every face, the pair's included"
[ "$(cat "$W/fp_device/exit.txt")" = 0 ] && [ "$(timedirs "$W/fp_device")" = "$(timedirs "$o")" ] \
    && grep -aq "forces check: the surface-tension flux on the pair" "$W/fp_device/log.brae" \
    && grep -aq "forces check: snGrad(rho) on the pair" "$W/fp_device/log.brae" \
    && grep -aq "forces check: snGrad(p_rgh)" "$W/fp_device/log.brae" \
    && say "$what" ok || say "$what" FAIL
what="[device] BRAE_CONTROL_FORCES_PAIR_HOST=1 keeps a coupled mesh's forces on the host: nothing to compare"
[ "$(cat "$W/fp_host/exit.txt")" = 0 ] && ! grep -aq "forces check: " "$W/fp_host/log.brae" \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  the correction dropped from the device's face loops: the check stops the run and names the face"
[ "$(cat "$W/fp_nocorrection/exit.txt")" != 0 ] \
    && grep -aq "FORCES_CHECK: the surface-tension flux, face" "$W/fp_nocorrection/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the interface forces across a coupled pair are formed on the GPU"
