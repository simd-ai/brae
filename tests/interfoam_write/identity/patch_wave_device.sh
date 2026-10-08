#!/usr/bin/env bash
# The write gate: the motion solver's wall-distance wave on the device against the host wave, on waveMakerPiston.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-03 the device loop runs the FaceCellWave<wallPoint> behind displacementLaplacian's inverseDistance
# diffusivity on the GPU (device_patch_wave.cu): each cell and face folds its own visits in the host's order, and
# the squared distance is formed as the host's compiled wave forms it -- fma(dz, dz, fma(dx, dx, dy*dy)).
# MEASURED on waveMakerPiston refined to 896,000 cells: the diffusivity 804 -> 174 ms a step; GPU wave against
# host wave, every written file BYTE-IDENTICAL on waveMakerPiston, waveMakerFlap and waveMakerSolitary. The
# control drops the neighbour side's visits (BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1): the wave no longer reaches
# the mesh and the solver refuses by name.
# THE GPU WAVE IS FORCED IN ITS ARMS (BRAE_PATCH_WAVE_MIN_FRONT=0): since 2026-10-06 a front as thin as this
# case's takes the host wave from its second call (patch_wave_front_rule.sh); this file is the GPU wave's.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerPiston of > "$W/pw_stage.txt" 2>&1
o="$W/w_of_waveMakerPiston"
[ -d "$o" ] || { say "waveMakerPiston did not stage" FAIL; finish "patch wave identity"; }
for v in device host control; do
    e="$W/pw_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device)  runbrae "$e" device BRAE_PATCH_WAVE_MIN_FRONT=0 ;;
        host)    runbrae "$e" device BRAE_CONTROL_PATCH_WAVE_HOST=1 ;;
        control) ( cd "$e" && BRAE_PATCH_WAVE_MIN_FRONT=0 BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1 \
                       "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/pw_control.rc" ;;
    esac
done
mark="the motion solver's wave runs on the GPU"
grep -q "$mark" "$W/pw_device/log.brae" && ! grep -q "$mark" "$W/pw_host/log.brae" \
    && say "the default run took the GPU wave, the other arm the host wave" ok \
    || say "the default run took the GPU wave, the other arm the host wave" FAIL
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/pw_host/$t" && find . -type f | sort); do
        cmp -s "$W/pw_host/$t/$f" "$W/pw_device/$t/$f" || n=$((n + 1))
    done
done
[ "$n" = 0 ] && [ "$(timedirs "$W/pw_device")" = "$(timedirs "$o")" ] \
    && say "[device] the GPU wave writes the host wave's files, byte for byte" ok \
    || say "[device] the GPU wave writes the host wave's files, byte for byte ($n differ)" FAIL
what="CONTROL  without the neighbour side's visits the wave does not reach the mesh, and brae refuses by name"
[ "$(cat "$W/pw_control.rc")" != 0 ] && grep -q "the wall distance did not reach" "$W/pw_control/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the motion solver's wall-distance wave on the device is the host wave"
