#!/usr/bin/env bash
# The write gate: hooks_empty_on_device.sh for a RELAXED corrector -- MULESCorr with two alpha correctors, the
# damBreak family -- which the first cut of that work left on the host: the relaxation assigns each patch
# 0.5*(its evaluate on the post-MULES cells) + 0.5*(alpha10's value) and reads alpha10's patch values off the
# host, and the MULESCorr pre-solve's divergence coefficients were built for every face there.
# On an `empty` patch both operands of that average are the cell's, and a half of a number is exact, so the
# entry is the relaxed cell's whichever way the two products are fused: it is mirrored like any other. The
# divergence coefficients of an empty face are written by a kernel from the device's flux (flux*1 and
# (-flux)*0, the host's own products). MEASURED 2026-10-06 on damBreak refined to 580,608 cells, ms a step:
# the alpha hooks 19.4 -> 2.3, the forces hook 13.0 -> 3.7, the step 233 -> 213.
# THE FIXTURE: laminar/damBreakPermeable (2-D, MULESCorr, two correctors, a permeable wall). The oracle is
# brae's own host build (BRAE_CONTROL_HOOKS_EMPTY_FROM_HOST=1): the written files byte for byte, the in-code
# bitwise check of every array at every call, and NaN in the host lists the hooks no longer fill.
# THE PATH IS FORCED HERE (BRAE_HOOKS_EMPTY_MIN_FACES=0): a mesh of the tutorial's size, 4,536 empty faces,
# builds these arrays on the host by default -- the kept path costs more than it saves below 16,000 empty
# faces (damBreak 6.7 for 6.4 ms a step) -- and no shipped tutorial of this family is larger. The arithmetic
# does not depend on the size; what this file holds is that arithmetic.
# TWO CONTROLS, one for each new array: the divergence coefficients written as 1 on the empty faces change the
# written files; the relaxed values written as 1 are stopped by the check, which names them.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakPermeable of > "$W/hr_stage.txt" 2>&1
o="$W/w_of_damBreakPermeable"
[ -d "$o" ] || { say "damBreakPermeable did not stage" FAIL; finish "relaxed corrector's empty faces identity"; }
for v in host check plain poison wrong stop; do
    e="$W/hr_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    forced=BRAE_HOOKS_EMPTY_MIN_FACES=0
    case $v in
        host)   runbrae "$e" device BRAE_CONTROL_HOOKS_EMPTY_FROM_HOST=1 ;;
        check)  runbrae "$e" device $forced BRAE_CONTROL_HOOKS_EMPTY_CHECK=1 ;;
        plain)  runbrae "$e" device $forced ;;
        poison) runbrae "$e" device $forced BRAE_CONTROL_HOOKS_EMPTY_POISON=1 ;;
        wrong)  runbrae "$e" device $forced BRAE_CONTROL_HOOKS_EMPTY_WRONG=divCoeffs ;;
        # this control is expected to stop: run it directly, runbrae would end the gate
        stop)   ( cd "$e" && env $forced BRAE_CONTROL_HOOKS_EMPTY_WRONG=alphaRelaxed \
                      BRAE_CONTROL_HOOKS_EMPTY_CHECK=1 "$BIN" -case . -device > log.brae 2>&1
                  echo $? > exit.txt ) ;;
    esac
done
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
held()   # held <the array's name in the check's line>: how many calls of it the check arm made
{
    grep -a "hooks empty check: $1" "$W/hr_check/log.brae" | tail -1 | sed -E 's/.*\(([0-9]+) calls\).*/\1/'
}
relaxed=$(held "alpha's patch values, relaxed")
divs=$(held "alpha's divergence coefficients, the diagonal's")
n=$(differs "$W/hr_host" "$W/hr_check")
m=$(differs "$W/hr_host" "$W/hr_plain")
p=$(differs "$W/hr_host" "$W/hr_poison")
nan=$(grep -raiwl "nan" $(for t in $(timedirs "$o"); do echo "$W/hr_poison/$t"; done) | wc -l)
what="[device] relaxed values (${relaxed:-0}+ calls) and divergence coefficients (${divs:-0}+) the host's, bitwise;"
what="$what files differ: checked $n, plain $m, poisoned $p ($nan with a nan)"
grep -q "hooks: an empty patch's entries" "$W/hr_plain/log.brae" \
    && ! grep -q "hooks.*an empty patch's entries" "$W/hr_host/log.brae" \
    && grep -q "hooks \[the host's lists there POISONED" "$W/hr_poison/log.brae" \
    && [ "${relaxed:-0}" -ge 1 ] && [ "${divs:-0}" -ge 1 ] && [ "$n" = 0 ] && [ "$m" = 0 ] && [ "$p" = 0 ] \
    && [ "$nan" = 0 ] && [ "$(timedirs "$W/hr_plain")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
n=$(differs "$W/hr_host" "$W/hr_wrong")
what="CONTROL  the divergence coefficients written as 1 on the empty faces: $n written files change"
[ "$n" != 0 ] && say "$what" ok || say "$what" FAIL
what="CONTROL  the relaxed values written as 1 on the empty faces: the check stops the run and names them"
[ "$(cat "$W/hr_stop/exit.txt")" != 0 ] \
    && grep -q "HOOKS_EMPTY_CHECK: alpha's patch values, relaxed, boundary face" "$W/hr_stop/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "a relaxed corrector's boundary arrays with the empty patches' entries on the GPU are the host's"
