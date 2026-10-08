#!/usr/bin/env bash
# The write gate: the displacement equation's interior assembled on the device against the host assembly, on
# waveMakerFlap.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 the device loop assembles the interior of displacementLaplacian's equation on the GPU
# (device_displacement_laplacian_assembly.cu): the face coefficients and the diagonal, grad(cellDisplacement), the
# non-orthogonal correction and its per-cell sum, each cell folding its faces in the host's order and every fused
# product the host's (read off the host binary). BRAE_CONTROL_MOTION_ASSEMBLY_CHECK=1 has the host assemble too
# and stops on one bit's difference anywhere in the matrix, at every step.
# MEASURED on waveMakerPiston refined to 896,000 cells: the matrix, gradient and correction 94 -> 5.3 ms a step.
# With the check on: waveMakerFlap over 50 steps and the 3-D waveMakerMultiPaddleFlap (448,000 cells) over 5, not
# one bit; every written file BYTE-IDENTICAL on waveMakerFlap, waveMakerPiston and waveMakerSolitary. The flap is
# the case here because its paddle ROTATES: from the second step the mesh is non-orthogonal and the correction is
# not zero. The control rounds the gradient's products before adding them
# (BRAE_CONTROL_MOTION_ASSEMBLY_UNFUSED=1): one rounding's difference, and the check names the entry -- measured,
# source entry 9, 1.4252799660516666e-13 against the host's 1.4252799660516659e-13.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/ma_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "motion assembly identity"; }
for v in device host control; do
    e="$W/ma_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device)  runbrae "$e" device BRAE_CONTROL_MOTION_ASSEMBLY_CHECK=1 ;;
        host)    runbrae "$e" device BRAE_CONTROL_MOTION_ASSEMBLY_HOST=1 ;;
        control) ( cd "$e" && BRAE_CONTROL_MOTION_ASSEMBLY_CHECK=1 BRAE_CONTROL_MOTION_ASSEMBLY_UNFUSED=1 \
                       "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/ma_control.rc" ;;
    esac
done
mark="interior is assembled on the GPU"
grep -q "$mark" "$W/ma_device/log.brae" && ! grep -q "$mark" "$W/ma_host/log.brae" \
    && say "the default run assembled on the GPU, the other arm on the host" ok \
    || say "the default run assembled on the GPU, the other arm on the host" FAIL
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/ma_host/$t" && find . -type f | sort); do
        cmp -s "$W/ma_host/$t/$f" "$W/ma_device/$t/$f" || n=$((n + 1))
    done
done
what="[device] the GPU assembly is the host's matrix at every step, and writes its files byte for byte"
[ "$n" = 0 ] && [ "$(timedirs "$W/ma_device")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what ($n differ)" FAIL
what="CONTROL  with the gradient's products rounded before they are added, the check names the entry and stops"
[ "$(cat "$W/ma_control.rc")" != 0 ] && grep -q "assembly run elsewhere is not the host's" "$W/ma_control/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the displacement equation's interior assembled on the device is the host's"
