#!/usr/bin/env bash
# The write gate: U's boundary hook brings U down as one block, and the host no longer evaluates an empty patch
# the GPU mirrors -- on weirOverflow (2-D, 5,080 cells, 10,160 empty faces).
. "$(dirname "$0")/../lib.sh"
# The hook runs three times a step. It brought U down in three copies and a walk over the cells, and at two of
# the three calls the host's EmptyPatchField::evaluate copied a cell's value to each of its two empty faces --
# values the device boundary has not taken from the host since it mirrors them itself. Since 2026-10-06 the
# device interleaves the components for one copy, and the hook evaluates the other patches only; a reader of the
# host's empty-patch values asks first (uEmptyCurrent, inter_driver_device.cu).
# ORACLE: the old hook (BRAE_CONTROL_U_DOWN_BY_COMPONENT=1 BRAE_CONTROL_U_EMPTY_HOST_EVALUATES=1), byte for byte
# in the written files; the bitwise check (BRAE_CONTROL_U_EMPTY_CHECK=1) for the device's entries; and the
# POISON (BRAE_CONTROL_U_EMPTY_POISON=1), NaN over the host values the hook leaves, for a reader that does not
# ask -- under the check the host is whole, so the check cannot see one.
# THE READER THAT ASKS is exercised by building the device boundary whole at every call
# (BRAE_CONTROL_U_BOUNDARY_FULL=1), which reads every patch's host values: with the poison on, that run has to
# write the oracle's files. CONTROL: the same run with the asking switched off
# (BRAE_CONTROL_U_EMPTY_NOT_ASKED=1) -- the NaN has to reach the written files or stop the run.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase weirOverflow of > "$W/uh_stage.txt" 2>&1
o="$W/w_of_weirOverflow"
[ -d "$o" ] || { say "weirOverflow did not stage" FAIL; finish "U hook, the host's share"; }
dt=$(grep -m1 '^deltaT' "$o/system/controlDict" | tr -d ';' | awk '{print $2}')
t4=$(python3 -c "print('%.12g' % (4*float('$dt')))")
old="BRAE_CONTROL_U_DOWN_BY_COMPONENT=1 BRAE_CONTROL_U_EMPTY_HOST_EVALUATES=1"
poi=BRAE_CONTROL_U_EMPTY_POISON=1
full=BRAE_CONTROL_U_BOUNDARY_FULL=1
for v in new old check poison asks notasked; do
    e="$W/uh_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i -E "s/^endTime .*/endTime         $t4;/; s/^writeInterval .*/writeInterval   4;/" "$e/system/controlDict"
    case $v in
        new)      runbrae "$e" device ;;
        old)      runbrae "$e" device $old ;;
        check)    runbrae "$e" device BRAE_CONTROL_U_EMPTY_CHECK=1 ;;
        poison)   runbrae "$e" device $poi ;;
        asks)     runbrae "$e" device $poi $full ;;
        # this control is expected to carry NaN or to stop: run it directly
        notasked) ( cd "$e" && env $poi $full BRAE_CONTROL_U_EMPTY_NOT_ASKED=1 "$BIN" -case . -device \
                        > log.brae 2>&1
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
# withnan <dir>: the written files that hold a nan -- through zcat, a compressed file's bytes can spell one
withnan()
{
    local n=0 t f
    for t in $(timedirs "$1"); do
        for f in $(find "$1/$t" -type f); do
            zcat -f "$f" | grep -aqiw "nan" && n=$((n + 1))
        done
    done
    echo $n
}
mark="the host does not evaluate an empty patch the GPU mirrors"
nt=$(for t in $(timedirs "$W/uh_old"); do find "$W/uh_old/$t" -type f; done | wc -l)
n=$(( $(differs "$W/uh_new" "$W/uh_old") + $(differs "$W/uh_check" "$W/uh_old") \
    + $(differs "$W/uh_poison" "$W/uh_old") ))
what="[device] one block down and the empty patches left, plain / checked / poisoned, write the old hook's $nt files"
grep -aq "$mark" "$W/uh_new/log.brae" && grep -aq "POISONED" "$W/uh_poison/log.brae" \
    && ! grep -aq "$mark" "$W/uh_old/log.brae" && ! grep -aq "$mark" "$W/uh_check/log.brae" \
    && [ "$n" = 0 ] && [ "$nt" -gt 0 ] && [ "$(withnan "$W/uh_poison")" = 0 ] \
    && [ "$(timedirs "$W/uh_new")" = "$t4 " ] \
    && say "$what" ok || say "$what ($n differ)" FAIL
na=$(differs "$W/uh_asks" "$W/uh_old")
what="[device] a reader that asks first (the boundary built whole at every call), poisoned: $na of $nt files differ"
[ "$na" = 0 ] && [ "$(withnan "$W/uh_asks")" = 0 ] && say "$what" ok || say "$what" FAIL
if [ "$(cat "$W/uh_notasked/exit.txt")" = 0 ]; then
    nn="$(withnan "$W/uh_notasked") files hold a nan"
    [ "$(withnan "$W/uh_notasked")" -gt 0 ] && bad=1 || bad=0
else
    nn="the run stops"
    bad=1
fi
what="CONTROL  the same reader not asking: $nn"
[ "$bad" = 1 ] && say "$what" ok || say "$what" FAIL
finish "U's hook with one block down and the host's empty patches left is the old hook"
