#!/usr/bin/env bash
# The write gate: U's boundary hook sends its staged arrays to their patches in ONE launch, and at the momentum
# assembly mirrors an empty patch's stored values on the GPU -- on weirOverflow (2-D; a
# variableHeightFlowRateInletVelocity, an inletOutlet, a noSlip and a pressureInletOutletVelocity patch).
. "$(dirname "$0")/../lib.sh"
# The hook runs three times a step. Each call refreshed the device boundary's eighteen state arrays with a
# device-to-device copy an array a patch (72 on stokesII), and the assembly call uploaded every boundary face's
# value because the step's buffer is new at every step. Since 2026-10-06 the arrays go in one launch a block
# (device_boundary.cuh, deviceScatterRuns) and the assembly mirrors like the other calls
# (inter_driver_device.cu). MEASURED, ms a step, three runs an arm: stokesII 7.77 -> 7.03, stokesI 9.17 -> 8.40,
# stokesV 10.93 -> 10.13, streamFunction 26.5 -> 24.9, damBreak 6.0 -> 5.3, the 3-D angledDuct 19.8 -> 19.7; the
# hook itself 1.4 -> 0.7 on stokesII, 4.4 -> 2.8 on streamFunction.
# ORACLE: the old path (BRAE_CONTROL_SCATTER_RUNS_BY_COPIES=1 BRAE_CONTROL_U_EMPTY_ASSEMBLY_WHOLE=1), byte for
# byte in the written files; and the in-code check (BRAE_CONTROL_U_EMPTY_CHECK=1), which builds every face's
# state and values on the host at every call and stops on one bit -- the files cannot see an empty face's entry.
# CONTROLS: the staged entries landed in reverse order (BRAE_CONTROL_SCATTER_RUNS_REVERSED=1) -- the files have
# to change, and under the check the run has to stop; the assembly's mirror skipped
# (BRAE_CONTROL_U_EMPTY_ASSEMBLY_STALE=1) -- the check has to stop it AT THE ASSEMBLY.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase weirOverflow of > "$W/uo_stage.txt" 2>&1
o="$W/w_of_weirOverflow"
[ -d "$o" ] || { say "weirOverflow did not stage" FAIL; finish "U hook in one launch"; }
dt=$(grep -m1 '^deltaT' "$o/system/controlDict" | tr -d ';' | awk '{print $2}')
t4=$(python3 -c "print('%.12g' % (4*float('$dt')))")
old="BRAE_CONTROL_SCATTER_RUNS_BY_COPIES=1 BRAE_CONTROL_U_EMPTY_ASSEMBLY_WHOLE=1"
chk=BRAE_CONTROL_U_EMPTY_CHECK=1
for v in new old check reversed stopped stale; do
    e="$W/uo_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i -E "s/^endTime .*/endTime         $t4;/; s/^writeInterval .*/writeInterval   4;/" "$e/system/controlDict"
    case $v in
        new)      runbrae "$e" device ;;
        old)      runbrae "$e" device $old ;;
        check)    runbrae "$e" device $chk ;;
        reversed) runbrae "$e" device BRAE_CONTROL_SCATTER_RUNS_REVERSED=1 ;;
        # these two are expected to stop: run directly, runbrae would end the gate
        stopped)  ( cd "$e" && env $chk BRAE_CONTROL_SCATTER_RUNS_REVERSED=1 "$BIN" -case . -device > log.brae 2>&1
                    echo $? > exit.txt ) ;;
        stale)    ( cd "$e" && env $chk BRAE_CONTROL_U_EMPTY_ASSEMBLY_STALE=1 "$BIN" -case . -device > log.brae 2>&1
                    echo $? > exit.txt ) ;;
    esac
done
differs()
{
    local n=0 t f
    for t in $(timedirs "$1"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
one="go to their patches in one launch"
asm="at the momentum assembly an empty patch's stored values are mirrored"
nt=$(for t in $(timedirs "$W/uo_old"); do find "$W/uo_old/$t" -type f; done | wc -l)
n=$(( $(differs "$W/uo_new" "$W/uo_old") + $(differs "$W/uo_check" "$W/uo_old") ))
what="[device] one launch and the assembly's mirror, with and without the check, write the old path's $nt files"
grep -aq "$one" "$W/uo_new/log.brae" && grep -aq "$asm" "$W/uo_new/log.brae" \
    && ! grep -aq "$one" "$W/uo_old/log.brae" && ! grep -aq "$asm" "$W/uo_old/log.brae" \
    && [ "$n" = 0 ] && [ "$nt" -gt 0 ] && [ "$(timedirs "$W/uo_new")" = "$t4 " ] \
    && say "$what" ok || say "$what ($n differ)" FAIL
nr=$(differs "$W/uo_reversed" "$W/uo_old")
what="CONTROL  the staged entries in reverse order: $nr written files change, and the check stops the run"
[ "$nr" -gt 0 ] && [ "$(cat "$W/uo_stopped/exit.txt")" != 0 ] \
    && grep -aq "U's boundary kept on the GPU is not the host's" "$W/uo_stopped/log.brae" \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  the assembly's mirror skipped: the check stops the run at the assembly"
[ "$(cat "$W/uo_stale/exit.txt")" != 0 ] \
    && grep -aq "not the host's: the stored patch values at the assembly" "$W/uo_stale/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "U's hook in one launch a block, mirrored at the assembly, is the old hook"
