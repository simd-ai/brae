#!/usr/bin/env bash
# The write gate: U's boundary entries on the empty patches kept on the GPU, against the host building and
# uploading them, on waveMakerFlap.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 the momentum hook leaves an `empty` patch's faces out of its key, its state refresh and its
# values: they stay on the device and a kernel mirrors them from the device's own U at the calls where the host
# patch evaluates (inter_driver_device.cu, mirrorEmptyFacesKernel). On a 2-D mesh they are two faces a cell.
# MEASURED on waveMakerPiston refined to 896,000 cells, ms a step: the key 29 -> 0.1, the state refresh 58 -> 2.9,
# the values 22 -> 7.0, the hook 191 -> 90. BRAE_CONTROL_U_EMPTY_CHECK=1 has the host build every face's state and
# values at every call and stops on one bit's difference from what the device holds: not one over 40 steps of
# waveMakerFlap and of weirOverflow, nor on capillaryRise, stokesI, waveMakerPiston, sloshingTank2D,
# damBreakLeakage, damBreakPermeable and damBreakPorousBaffle. THE WRITTEN FILES are not the oracle for the
# mirror: the kernels multiply an empty face's entries by zero, and a run with the mirror skipped writes the
# same bytes. So the control is the check itself, with the mirror skipped
# (BRAE_CONTROL_U_EMPTY_NO_MIRROR=1): measured, refValue at boundary face 400, the device 0, the host
# -3.239234835391752. waveMakerFlap because its mesh moves: the boundary is rebuilt whole once a step and
# refreshed run by run at the other calls.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/ue_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "U empty on device identity"; }
for v in device host control; do
    e="$W/ue_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device)  runbrae "$e" device BRAE_CONTROL_U_EMPTY_CHECK=1 ;;
        host)    runbrae "$e" device BRAE_CONTROL_U_EMPTY_FROM_HOST=1 ;;
        control) ( cd "$e" && BRAE_CONTROL_U_EMPTY_CHECK=1 BRAE_CONTROL_U_EMPTY_NO_MIRROR=1 \
                       "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/ue_control.rc" ;;
    esac
done
mark="an empty patch's entries stay on the GPU"
grep -q "$mark" "$W/ue_device/log.brae" && ! grep -q "$mark" "$W/ue_host/log.brae" \
    && say "the default run keeps the empty patches' entries on the GPU, the other arm builds them on the host" ok \
    || say "the default run keeps the empty patches' entries on the GPU, the other arm builds them on the host" FAIL
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/ue_host/$t" && find . -type f | sort); do
        cmp -s "$W/ue_host/$t/$f" "$W/ue_device/$t/$f" || n=$((n + 1))
    done
done
what="[device] the GPU-kept entries are the host's at every call, and the files are the same byte for byte"
[ "$n" = 0 ] && [ "$(timedirs "$W/ue_device")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what ($n differ)" FAIL
what="CONTROL  with the mirror skipped the check names the entry and stops"
[ "$(cat "$W/ue_control.rc")" != 0 ] \
    && grep -q "U's boundary kept on the GPU is not the host's" "$W/ue_control/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "U's empty-patch entries kept on the GPU are the ones the host built"
