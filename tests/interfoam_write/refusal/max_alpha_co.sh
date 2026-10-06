#!/usr/bin/env bash
# The write gate: a controlDict with no `maxAlphaCo` is refused, at a FIXED time step too -- because real
# OpenFOAM stops on it.
. "$(dirname "$0")/../lib.sh"
# alphaCourantNo.H:34-37 reads maxAlphaCo with get<scalar>, no default, and interFoam.C:100-102 includes it at
# every step of a run that is not local-time-stepped, whatever adjustTimeStep says. brae refused the missing
# entry only under `adjustTimeStep yes` until 2026-10-06 (time_controls.cuh said "a fixed-step case never reads
# it") and ran a fixed-step case OpenFOAM stops on.
# ORACLE: real interFoam on laminar/damBreakPermeable's pinned row (adjustTimeStep no) with the entry removed:
# it has to stop and name the entry. THE PAIR, which is the control: the same case with the entry in place
# runs on both loops -- the refusal keys on the entry and on nothing else in the fixture.
wcase damBreakPermeable of > "$W/ma_stage.txt" 2>&1
o="$W/w_of_damBreakPermeable"
[ -d "$o" ] || { say "damBreakPermeable did not stage" FAIL; finish "maxAlphaCo is mandatory"; }
grep -q '^adjustTimeStep  *no;' "$o/system/controlDict" && grep -q '^maxAlphaCo' "$o/system/controlDict" \
    || { say "the fixture is not a fixed-step case that names maxAlphaCo" FAIL; finish "maxAlphaCo is mandatory"; }
for v in of host device; do
    e="$W/ma_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i '/^maxAlphaCo/d' "$e/system/controlDict"
    # each of these is expected to stop: run directly
    case $v in
        of)     ( cd "$e" && interFoam > log.run 2>&1; echo $? > exit.txt ) ;;
        host)   ( cd "$e" && "$BIN" -case . > log.run 2>&1; echo $? > exit.txt ) ;;
        device) ( cd "$e" && "$BIN" -case . -device > log.run 2>&1; echo $? > exit.txt ) ;;
    esac
done
what="ORACLE  real interFoam stops on the fixed-step case without maxAlphaCo and names the entry"
[ "$(cat "$W/ma_of/exit.txt")" != 0 ] && grep -aq "maxAlphaCo" "$W/ma_of/log.run" \
    && grep -aqi "not found\|undefined" "$W/ma_of/log.run" && [ -z "$(timedirs "$W/ma_of")" ] \
    && say "$what" ok || say "$what" FAIL
ok=1
for v in host device; do
    [ "$(cat "$W/ma_$v/exit.txt")" != 0 ] && grep -aq "controlDict has no \`maxAlphaCo\`" "$W/ma_$v/log.run" \
        && [ -z "$(timedirs "$W/ma_$v")" ] || ok=0
done
what="[host, device] brae refuses the same case by name, before its first step, having written nothing"
[ $ok = 1 ] && say "$what" ok || say "$what" FAIL
for v in host device; do
    e="$W/ma_with_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    runbrae "$e" $v
done
what="CONTROL  with the entry in place the same case runs to its end on both loops"
[ "$(timedirs "$W/ma_with_host")" = "$(timedirs "$o")" ] && [ "$(timedirs "$W/ma_with_device")" = "$(timedirs "$o")" ] \
    && [ -n "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
finish "a controlDict without maxAlphaCo is refused wherever OpenFOAM stops on it"
