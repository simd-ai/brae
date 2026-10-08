#!/usr/bin/env bash
# The write gate: the flux push by runs (flux_empty_by_runs.sh) on a MOVING 2-D mesh, waveMakerFlap, whose two
# empty patches split the other patches into runs and whose mesh update reads the boundary flux as the last
# push left it -- the reader the first attempt at this missed. That push is made whole; the hooks' go by runs.
# The device does not hold zeros on these faces (-8.0e-42 on `front`, the mesh flux's rounding), so a list that
# is left is not equal to the one that was pushed: the claim is that it is NOT READ, held by NaN.
# The oracle is brae's own whole push (BRAE_CONTROL_FLUX_EMPTY_FROM_HOST=1), byte for byte on the written files.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/fm_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "flux push by runs on a moving mesh identity"; }
for v in host poison control; do
    e="$W/fm_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        host)    runbrae "$e" device BRAE_CONTROL_FLUX_EMPTY_FROM_HOST=1 ;;
        poison)  runbrae "$e" device BRAE_CONTROL_FLUX_EMPTY_POISON=1 ;;
        # this control is expected to end in NaN: run it directly, runbrae would end the gate. FOUR steps, not
        # the row's two: the poisoned push is read by the NEXT step's mesh update, which hands the boundary
        # flux back to the device, and the NaN shows in the step after that (MEASURED: the third)
        control) sed -i -E 's/^endTime .*/endTime 0.02;/' "$e/system/controlDict"
                 ( cd "$e" && BRAE_CONTROL_FLUX_EMPTY_POISON=1 BRAE_CONTROL_FLUX_DYNAMIC_PUSH_BY_RUNS=1 \
                       "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/fm_host/$t" && find . -type f | sort); do
        cmp -s "$W/fm_host/$t/$f" "$W/fm_poison/$t/$f" || n=$((n + 1))
    done
done
nan=$(grep -raiwl "nan" $(for t in $(timedirs "$o"); do echo "$W/fm_poison/$t"; done) | wc -l)
what="[device] by runs with NaN in the empty patches' host lists: $n files differ from the whole push, $nan nan"
grep -q "flux push \[an empty patch's host list POISONED" "$W/fm_poison/log.brae" \
    && grep -q "flux push: every boundary face" "$W/fm_host/log.brae" \
    && [ "$n" = 0 ] && [ "$nan" = 0 ] && [ "$(timedirs "$W/fm_poison")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
at=$(grep -a "^ *t = " "$W/fm_control/log.brae" | grep -n "nan" | head -1 | cut -d: -f1)
what="CONTROL  the push before the mesh update by runs too, poisoned: NaN from step ${at:-none} of 4"
grep -q "CONTROL MODE: a reader of an empty patch's flux" "$W/fm_control/log.brae" \
    && [ -n "$at" ] && say "$what" ok || say "$what" FAIL
finish "on a moving mesh the flux push by runs writes what the whole push writes"
