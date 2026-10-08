#!/usr/bin/env bash
# The write gate: the mesh flux of a move computed on the GPU, against the host's face loop, on waveMakerFlap.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 the device loop computes every face's swept volume on the GPU (device_swept_volumes.cu):
# face::sweptVol's central decomposition and triangle::sweptVol's six terms, every fused product the host's --
# read off the host binary instruction by instruction. BRAE_CONTROL_SWEPT_VOLUME_CHECK=1 has the host's loop run
# too and stops on one bit's difference on any face, at every move.
# MEASURED on waveMakerPiston refined to 896,000 cells: 97 -> 8.8 ms a step, the step 1,282 -> 1,192. With the
# check on, not one bit: 25 steps of waveMakerFlap, sloshingTank3D6DoF, floatingObject and DTCHullMovingCoarse
# (polyhedral faces), and waveMakerPiston, testTubeMixer, sloshingCylinder, sloshingTank2D, mixerVesselAMI and
# oscillatingBox (a refinement between moves). The control rounds one product of the triple product before
# adding it (BRAE_CONTROL_SWEPT_VOLUME_UNFUSED=1): measured, internal face 1067, 3.4694499816265408e-20 against
# the host's 3.4694499816265396e-20. waveMakerFlap because its paddle rotates: no face moves rigidly.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/sv_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "swept volume identity"; }
for v in device host control; do
    e="$W/sv_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device)  runbrae "$e" device BRAE_CONTROL_SWEPT_VOLUME_CHECK=1 ;;
        host)    runbrae "$e" device BRAE_CONTROL_SWEPT_VOLUME_HOST=1 ;;
        control) ( cd "$e" && BRAE_CONTROL_SWEPT_VOLUME_CHECK=1 BRAE_CONTROL_SWEPT_VOLUME_UNFUSED=1 \
                       "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/sv_control.rc" ;;
    esac
done
mark="swept volumes of each move are computed on the GPU"
grep -q "$mark" "$W/sv_device/log.brae" && ! grep -q "$mark" "$W/sv_host/log.brae" \
    && say "the default run computes the swept volumes on the GPU, the other arm on the host" ok \
    || say "the default run computes the swept volumes on the GPU, the other arm on the host" FAIL
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/sv_host/$t" && find . -type f | sort); do
        cmp -s "$W/sv_host/$t/$f" "$W/sv_device/$t/$f" || n=$((n + 1))
    done
done
what="[device] the GPU's mesh flux is the host's on every face at every move, and the files are the same"
[ "$n" = 0 ] && [ "$(timedirs "$W/sv_device")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what ($n differ)" FAIL
what="CONTROL  with one product rounded before it is added, the check names the face and stops"
[ "$(cat "$W/sv_control.rc")" != 0 ] \
    && grep -q "the swept volumes computed elsewhere are not the host's" "$W/sv_control/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the mesh flux computed on the GPU is the host's face loop's"
