#!/usr/bin/env bash
# The write gate: seven buffers copied ON THE GPU at two places where they went down to the host and up again,
# and the cells' velocity gathered only for the patch class that reads it, against the old forms, on
# laminar/capillaryRise's W row (`div(phirb,alpha) Gauss linear`; its inlet and its atmosphere are
# pressureInletOutletVelocity, the class that reads the gather).
#   the linear scheme's weights    device -> host -> device twice a corrector; now deviceCopy
#   the old-time levels            alpha, Ux, Uy, Uz and phi's two halves, down and up at every step; now deviceCopy
#   updateVelocityPatchesFromCells gathered U at the face cells of EVERY patch for a call whose body is empty in
#                                  every class but one; now for that class alone (host loop and device loop)
# MEASURED 2026-10-05, ms a step: laminar/waves/streamFunction (160,000 cells) 39.2-40.0 -> 37.3-37.9 over three
# pairs; solitaryGrimshaw (411,600) 64.7 -> 61.5.
# Copies of the same bytes, and a call that reads nothing: nothing written may change. Held to the old paths:
# the DEFAULT device run (no switch: its copies are not followed by a read-back), the device run with
# BRAE_CONTROL_DEVICE_COPIES_CHECK=1 (both buffers read back after every copy and compared), and the default
# HOST run (the gather is the host loop's code too). TWO CONTROLS. A buffer left as it was
# (BRAE_CONTROL_DEVICE_COPIES_STALE=1) must be stopped by the check -- it can bite on alpha's and U's old levels
# only, the other three being fresh buffers at every call. The gather skipped for every patch
# (BRAE_CONTROL_U_PATCH_GATHER_NONE=1) must finish and CHANGE U, which is what shows a patch here reads it.
# NOT RUN HERE: the CrankNicolson branch of the step preparation (phi's host copies). brae stops a
# CrankNicolson run on a mesh that does not move at its first write, so there are no files to compare here; that
# branch is held by interfoam_cn_vs_openfoam's device arm at its own bound, by the cn_moving device gates on a
# mesh that moves, and by a size check in the code.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase capillaryRise of > "$W/hr_stage.txt" 2>&1
o="$W/w_of_capillaryRise"
[ -d "$o" ] || { say "capillaryRise did not stage" FAIL; finish "host round trips identity"; }
OLD="BRAE_CONTROL_COPIES_VIA_HOST=1 BRAE_CONTROL_U_PATCH_GATHER_ALL=1"
for v in old new checked hold hnew stale none; do
    e="$W/hr_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        old)     runbrae "$e" device $OLD ;;
        new)     runbrae "$e" device ;;
        checked) runbrae "$e" device BRAE_CONTROL_DEVICE_COPIES_CHECK=1 ;;
        hold)    runbrae "$e" host BRAE_CONTROL_U_PATCH_GATHER_ALL=1 ;;
        hnew)    runbrae "$e" host ;;
        # the controls are expected to stop or to differ: run directly, runbrae would end the gate
        stale)   ( cd "$e" && BRAE_CONTROL_DEVICE_COPIES_CHECK=1 BRAE_CONTROL_DEVICE_COPIES_STALE=1 \
                       "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
        none)    ( cd "$e" && BRAE_CONTROL_U_PATCH_GATHER_NONE=1 "$BIN" -case . -device > log.brae 2>&1
                   echo $? > exit.txt ) ;;
    esac
done
# same <a> <b>: how many files of arm a are not arm b's, byte for byte; "-" unless both hold OpenFOAM's times
# and the same files at each
same()
{
    local a="$W/hr_$1" b="$W/hr_$2" n=0 d f
    [ "$(timedirs "$a")" = "$(timedirs "$o")" ] && [ "$(timedirs "$b")" = "$(timedirs "$o")" ] || { echo "-"; return; }
    for d in $(timedirs "$o"); do
        [ "$(filesets "$a" "$d")" = "$(filesets "$b" "$d")" ] || { echo "-"; return; }
        for f in $(cd "$a/$d" && find . -type f | sort); do
            cmp -s "$a/$d/$f" "$b/$d/$f" || n=$((n + 1))
        done
    done
    echo $n
}
# only <mark> <arm with> <arm without>
only() { grep -q "$1" "$W/hr_$2/log.brae" && ! grep -q "$1" "$W/hr_$3/log.brae"; }
copies="a buffer copied whole stays on the GPU"
gather="the cells' velocity is gathered for the"
t=$(for d in $(timedirs "$W/hr_old"); do find "$W/hr_old/$d" -type f; done | wc -l)
n1=$(same old new)
n2=$(same old checked)
n3=$(same hold hnew)
what="[device] the default run and the checked one, [host] the default run, against the old paths: $t files an"
what="$what arm; $n1, $n2 and $n3 differ"
only "$copies" new old && only "$copies" checked old && only "$gather" new old && only "$gather" hnew hold \
    && [ "$t" -gt 0 ] && [ "$n1" = 0 ] && [ "$n2" = 0 ] && [ "$n3" = 0 ] && say "$what" ok || say "$what" FAIL
what="CONTROL  a buffer that has the size left as it was: the check stops the run and names the buffer"
[ "$(cat "$W/hr_stale/exit.txt")" != 0 ] \
    && grep -q "DEVICE_COPIES_CHECK: .* copied on the device is not the buffer" "$W/hr_stale/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=$(same new none)
last=$(timedirs "$o" | tr ' ' '\n' | grep . | tail -1)
what="CONTROL  the cells' velocity gathered for no patch: the run ends, and $n of its files are not the default's"
[ "$(cat "$W/hr_none/exit.txt")" = 0 ] && grep -q "CONTROL MODE: the cells' velocity" "$W/hr_none/log.brae" \
    && [ "$n" != "-" ] && [ "$n" -gt 0 ] && ! cmp -s "$W/hr_new/$last/U" "$W/hr_none/$last/U" \
    && say "$what" ok || say "$what" FAIL
finish "buffers copied on the GPU and a gather for the one class that reads it write what the host round trips wrote"
