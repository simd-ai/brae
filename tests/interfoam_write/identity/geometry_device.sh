#!/usr/bin/env bash
# The write gate: the moved mesh's geometry built on the GPU, and the device mesh refreshed from it there,
# against the host's build and its upload, on sloshingTank3D6DoF.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 the device loop builds FvGeometry on the GPU after each move (device_fv_geometry.cu): face
# centres and areas, cell centres and volumes, the interpolation factors, every fused product the host's -- read
# off the host binary -- and each cell folding its faces in the host's order. The nine arrays come down into the
# host's geometry, and the device mesh's geometric buffers are formed from the device's copy instead of being
# rebuilt on the host and uploaded.
# MEASURED on waveMakerPiston refined to 896,000 cells: the build 87 -> 13.7 ms a step, the device mesh's
# refresh 52 -> 3.4, the step 1,192 -> 1,087.
# TWO ORACLES, both at every move: BRAE_CONTROL_GEOMETRY_CHECK=1 has the host build the geometry too and compares
# the nine arrays' bytes; BRAE_CONTROL_DEVICE_MESH_CHECK=1 builds the device mesh from the host too and compares
# its twenty geometric buffers. Not one bit on waveMakerFlap, waveMakerPiston, sloshingTank2D, sloshingTank3D6DoF,
# floatingObject, DTCHullMovingCoarse (polyhedral), testTubeMixer, sloshingCylinder, mixerVesselAMI and
# oscillatingBox; 25 steps on four of them.
# THIS CASE because its tank ROTATES, so every face is slanted in all three directions: the control that rounds
# the pyramid volume's products before adding them (BRAE_CONTROL_GEOMETRY_UNFUSED=1) changes NOTHING on
# waveMakerFlap or floatingObject, where one term of every product pair is zero, and is caught here -- measured,
# C entry 10, -12.686457213021972 against the host's -12.686457213021971. The second control leaves the device
# mesh's internal offsets and correction vectors as the last mesh's (BRAE_CONTROL_DEVICE_MESH_STALE_OFFSETS=1).
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase sloshingTank3D6DoF of > "$W/gd_stage.txt" 2>&1
o="$W/w_of_sloshingTank3D6DoF"
[ -d "$o" ] || { say "sloshingTank3D6DoF did not stage" FAIL; finish "geometry device identity"; }
for v in device host unfused stale; do
    e="$W/gd_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device)  runbrae "$e" device BRAE_CONTROL_GEOMETRY_CHECK=1 BRAE_CONTROL_DEVICE_MESH_CHECK=1 ;;
        host)    runbrae "$e" device BRAE_CONTROL_GEOMETRY_HOST=1 ;;
        unfused) ( cd "$e" && BRAE_CONTROL_GEOMETRY_CHECK=1 BRAE_CONTROL_GEOMETRY_UNFUSED=1 \
                       "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/gd_unfused.rc" ;;
        stale)   ( cd "$e" && BRAE_CONTROL_DEVICE_MESH_CHECK=1 BRAE_CONTROL_DEVICE_MESH_STALE_OFFSETS=1 \
                       "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/gd_stale.rc" ;;
    esac
done
m1="mesh geometry: after each move it is built on the GPU"
m2="device mesh: after each move its geometry is taken from the GPU's own"
what="the default run builds the geometry and refreshes the device mesh on the GPU, the other arm on the host"
grep -q "$m1" "$W/gd_device/log.brae" && grep -q "$m2" "$W/gd_device/log.brae" \
    && ! grep -q "$m1" "$W/gd_host/log.brae" && ! grep -q "$m2" "$W/gd_host/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/gd_host/$t" && find . -type f | sort); do
        cmp -s "$W/gd_host/$t/$f" "$W/gd_device/$t/$f" || n=$((n + 1))
    done
done
what="[device] the GPU's geometry and device mesh are the host's at every move, and the files are the same"
[ "$n" = 0 ] && [ "$(timedirs "$W/gd_device")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what ($n differ)" FAIL
what="CONTROL  one rounding in the pyramid volumes, and stale offsets in the device mesh, are each named and stop"
[ "$(cat "$W/gd_unfused.rc")" != 0 ] \
    && grep -q "the geometry built elsewhere is not the host's" "$W/gd_unfused/log.brae" \
    && [ "$(cat "$W/gd_stale.rc")" != 0 ] \
    && grep -q "the device mesh refreshed on the GPU is not the one built from the host" "$W/gd_stale/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the moved mesh's geometry built on the GPU is the host's"
