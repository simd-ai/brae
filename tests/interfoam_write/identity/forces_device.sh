#!/usr/bin/env bash
# The write gate: the interface-forces hook's face loops on the device against the host's, electrostaticDeposition.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-03 the device loop computes the internal faces of surfaceTensionForce() and of snGrad(rho) on
# the device, where no coupled pair is in them (the patch faces stay the host's): 57 ms a step to 11 on
# RAS/DTCHull. MEASURED, device-face run against host-face run: DTCHull, capillaryRise, damBreakPermeable,
# nozzleFlow2D and weirOverflow bit-identical; electrostaticDeposition 2.9e-15 (nut) -- its gradients are
# `cellLimited leastSquares 1`, the device's own fit. Held to one decade above that, on that case.
BOUND_FORCES_IDENTITY=2.9e-14
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase electrostaticDeposition of > "$W/forces_stage.txt" 2>&1
o="$W/w_of_electrostaticDeposition"
[ -d "$o" ] || { say "electrostaticDeposition did not stage" FAIL; finish "forces identity"; }
grep -qE "default +corrected;" "$o/system/fvSchemes" && grep -qE "default +cellLimited leastSquares 1;" "$o/system/fvSchemes" \
    && say "the fixture's snGrad is corrected and its gradients cellLimited leastSquares" ok \
    || say "the fixture's snGrad is corrected and its gradients cellLimited leastSquares" FAIL
for v in device host control; do
    e="$W/forces_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device)  runbrae "$e" device ;;
        host)    runbrae "$e" device BRAE_CONTROL_FORCES_HOST=1 ;;
        control) runbrae "$e" device BRAE_CONTROL_FORCES_NO_CORRECTION=1 ;;
    esac
done
python3 "$CMP" "$W/forces_host" "$W/forces_device" $(timedirs "$o") > "$W/cmp_forces.txt" 2>&1
judge "forces device against host" "$W/cmp_forces.txt" "$BOUND_FORCES_IDENTITY" "$o/log.interFoam" \
    && say "[device] the device's face loops give the host's run, every file within $BOUND_FORCES_IDENTITY" ok \
    || { say "[device] the device's face loops give the host's run, every file within $BOUND_FORCES_IDENTITY" FAIL; grep -v RESULT "$W/cmp_forces.txt" | head -8; }
# THE CONTROL: the device's face loop without the non-orthogonal correction. MEASURED p_rgh 2.5e-02.
python3 "$CMP" "$W/forces_host" "$W/forces_control" $(timedirs "$o") > "$W/cmp_forces_control.txt" 2>&1
judge "forces without correction" "$W/cmp_forces_control.txt" "$BOUND_FORCES_IDENTITY" "$o/log.interFoam" > "$W/forces_control.txt" \
    && { say "CONTROL  the device's face loop without its correction leaves the bound" FAIL; cat "$W/forces_control.txt"; } \
    || say "CONTROL  the device's face loop without its correction leaves the bound" ok
finish "the interface-forces hook's device face loops are the host's"
