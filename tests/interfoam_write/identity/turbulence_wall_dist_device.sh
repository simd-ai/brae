#!/usr/bin/env bash
# The write gate: the turbulence closure's wall distance on a moved mesh with its wave on the GPU and its
# near-wall correction's topology kept between calls, against the host's wave and a rebuild at every call, on
# DTCHullMovingCoarse.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 moveInterTurbulence takes kOmegaSST's wall-distance wave from the device (devicePatchWave,
# the wave the motion solver's diffusivity already ran there) and cellWallDist keeps the correction's
# point-and-face lists between calls (CellWallDistCache) -- it rebuilt them, with every mesh point's cells, at
# every move.
# MEASURED on RAS/DTCHullMoving (845,536 cells), ms a step: the closure's distances 455 -> 148, the step
# 2,269 -> 1,946; here (108,833 cells) 84 -> 36, the correction 46 -> 5 ms a call.
# ORACLE: BRAE_CONTROL_PATCH_WAVE_CHECK=1 runs the host's wave too and compares every cell's squared distance to
# the bit. The control drops the neighbour side's visits (BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1) and the check
# must name a cell. A moving hull under kOmegaSST is the only shipped combination that asks for this distance at
# every step.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase DTCHullMovingCoarse of > "$W/tw_stage.txt" 2>&1
o="$W/w_of_DTCHullMovingCoarse"
[ -d "$o" ] || { say "DTCHullMovingCoarse did not stage" FAIL; finish "turbulence wall distance identity"; }
for v in device host control; do
    e="$W/tw_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device)  runbrae "$e" device BRAE_CONTROL_PATCH_WAVE_CHECK=1 ;;
        host)    runbrae "$e" device BRAE_CONTROL_TURBULENCE_WAVE_HOST=1 BRAE_CONTROL_WALL_DIST_NO_CACHE=1 ;;
        control) ( cd "$e" && BRAE_CONTROL_PATCH_WAVE_CHECK=1 BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1 \
                       "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/tw_control.rc" ;;
    esac
done
mark="the turbulence closure's wave runs on the GPU"
grep -q "$mark" "$W/tw_device/log.brae" && ! grep -q "$mark" "$W/tw_host/log.brae" \
    && say "the default run takes the closure's wave from the GPU, the other arm from the host" ok \
    || say "the default run takes the closure's wave from the GPU, the other arm from the host" FAIL
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/tw_host/$t" && find . -type f | sort); do
        cmp -s "$W/tw_host/$t/$f" "$W/tw_device/$t/$f" || n=$((n + 1))
    done
done
what="[device] the GPU's wave is the host's in every cell, and with the kept correction the files are the same"
[ "$n" = 0 ] && [ "$(timedirs "$W/tw_device")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what ($n differ)" FAIL
what="CONTROL  without the neighbour side's visits the check names the cell and stops"
[ "$(cat "$W/tw_control.rc")" != 0 ] \
    && grep -q "the wave run elsewhere is not the host's" "$W/tw_control/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the closure's wall distance with its wave on the GPU is the host's"
