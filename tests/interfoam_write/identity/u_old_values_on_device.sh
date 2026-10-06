#!/usr/bin/env bash
# The write gate: U's OLD-TIME patch values with an `empty` patch's entries mirrored ON THE GPU, against every
# boundary face flattened on the host, on laminar/waves/streamFunction (320,000 of 324,160 boundary faces an
# empty patch's). At the top of every step the driver flattened U's patch values, three components of every
# face by push_back, and uploaded them. An empty patch's are its cells' (EmptyPatchField::evaluate) and
# nothing moves U between a step's last evaluate and the next step's start, so they are mirrored on the device
# from the U the step starts on, and the other patches' go up run by run.
# MEASURED 2026-10-06 on this case: 1.5 -> 0.1 ms a step, the step 27.7 -> 26.3.
# THE ORACLE is brae's own host list: BRAE_CONTROL_U_EMPTY_CHECK=1 compares every entry bitwise at every step,
# and the written files against BRAE_CONTROL_U_EMPTY_FROM_HOST=1 byte for byte. The files alone cannot be it:
# an old-time value on an empty face enters ddtCorr times a face area with no in-plane part. The CONTROL
# skips this mirror alone, so the entries keep the step before's, and the check has to name this array.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase streamFunction of > "$W/uo_stage.txt" 2>&1
o="$W/w_of_streamFunction"
[ -d "$o" ] || { say "streamFunction did not stage" FAIL; finish "U's old-time patch values identity"; }
for v in host check control; do
    e="$W/uo_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        host)    runbrae "$e" device BRAE_CONTROL_U_EMPTY_FROM_HOST=1 ;;
        check)   runbrae "$e" device BRAE_CONTROL_U_EMPTY_CHECK=1 ;;
        # this control is expected to stop: run it directly, runbrae would end the gate
        control) ( cd "$e" && BRAE_CONTROL_U_EMPTY_CHECK=1 BRAE_CONTROL_U_OLD_VALUES_NO_MIRROR=1 \
                       "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/uo_host/$t" && find . -type f | sort); do
        cmp -s "$W/uo_host/$t/$f" "$W/uo_check/$t/$f" || n=$((n + 1))
    done
done
steps=$(grep -a "U old-time patch values check:" "$W/uo_check/log.brae" | tail -1 \
    | sed -E 's/.*check: ([0-9]+) steps.*/\1/')
what="[device] the old-time values kept on the GPU are the host's, bitwise (${steps:-0}+ steps); $n files differ"
[ "${steps:-0}" -ge 1 ] && ! grep -q "U old-time patch values check" "$W/uo_host/log.brae" \
    && [ "$n" = 0 ] && [ "$(timedirs "$W/uo_check")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
what="CONTROL  with this mirror skipped the check stops the run and names the old-time patch values"
[ "$(cat "$W/uo_control/exit.txt")" != 0 ] \
    && grep -q "not the host's: the old-time patch values" "$W/uo_control/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "U's old-time patch values with the empty faces mirrored on the GPU are the ones the host flattened"
