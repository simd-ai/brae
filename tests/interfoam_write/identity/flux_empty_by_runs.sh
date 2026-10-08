#!/usr/bin/env bash
# The write gate: the flux push to the host with an `empty` patch's faces LEFT ON THE DEVICE, against the push
# of every boundary face, on damBreakPermeable -- 2-D, MULESCorr, so alpha's divergence hook runs.
# The hooks pushed phi's whole boundary down six times a step; on a 2-D mesh that array is mostly the empty
# patches' faces (320,000 of streamFunction's 324,160). MEASURED 2026-10-06 on laminar/waves/streamFunction:
# the two `flux to the patches` rows 0.6 + 0.6 -> 0.0 + 0.0 ms a step, the step 35.6 -> 34.2.
# The first attempt at this (2026-10-05) was withdrawn: three readers take an empty patch's host list. Each
# now has what it reads -- alpha's divCoeffs hook the device's own flux, the host closure and a dynamic mesh a
# whole push -- and the claim that NOTHING ELSE reads the list is held by writing NaN into it after every push.
# The oracle is brae's own whole push (BRAE_CONTROL_FLUX_EMPTY_FROM_HOST=1), byte for byte on the written files.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakPermeable of > "$W/fx_stage.txt" 2>&1
o="$W/w_of_damBreakPermeable"
[ -d "$o" ] || { say "damBreakPermeable did not stage" FAIL; finish "flux push by runs identity"; }
for v in host check poison control; do
    e="$W/fx_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        host)    runbrae "$e" device BRAE_CONTROL_FLUX_EMPTY_FROM_HOST=1 ;;
        check)   runbrae "$e" device BRAE_CONTROL_FLUX_DIVCOEFFS_CHECK=1 ;;
        poison)  runbrae "$e" device BRAE_CONTROL_FLUX_EMPTY_POISON=1 ;;
        # this control is expected to end in NaN: run it directly, runbrae would end the gate
        control) ( cd "$e" && BRAE_CONTROL_FLUX_EMPTY_POISON=1 BRAE_CONTROL_FLUX_DIVCOEFFS_FROM_HOST_COPY=1 \
                       "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
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
n=$(differs "$W/fx_host" "$W/fx_check")
calls=$(grep -a "flux divCoeffs check:" "$W/fx_check/log.brae" | tail -1 | sed -E 's/.*check: ([0-9]+) calls.*/\1/')
big=$(grep -a "flux divCoeffs check:" "$W/fx_check/log.brae" | tail -1 | sed -E 's/.*empty face ([0-9.e+-]+).*/\1/')
what="[device] divCoeffs' flux from the device is the host's whole push (${calls:-0}+ calls; on an empty face"
what="$what at most ${big:-?}); $n files differ"
grep -q "flux push: the patches that are not empty" "$W/fx_check/log.brae" \
    && grep -q "flux push: every boundary face" "$W/fx_host/log.brae" \
    && [ "${calls:-0}" -ge 1 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/fx_check")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
n=$(differs "$W/fx_host" "$W/fx_poison")
nan=$(grep -raiwl "nan" $(for t in $(timedirs "$o"); do echo "$W/fx_poison/$t"; done) | wc -l)
what="[device] NaN in the empty patches' host lists after every push: $n files differ, $nan hold a nan"
grep -q "flux push \[an empty patch's host list POISONED" "$W/fx_poison/log.brae" \
    && [ "$n" = 0 ] && [ "$nan" = 0 ] && say "$what" ok || say "$what" FAIL
what="CONTROL  divCoeffs on the host's stale copy, poisoned: the run ends in NaN"
grep -q "CONTROL MODE: a reader of an empty patch's flux" "$W/fx_control/log.brae" \
    && grep -Eq "alpha (in )?\[-?nan" "$W/fx_control/log.brae" && say "$what" ok || say "$what" FAIL
finish "the flux push by runs writes what the whole push writes, and no reader takes an empty patch's host list"
