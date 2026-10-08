#!/usr/bin/env bash
# The gate: the ENTRIES a function object's write times are read from, and what several objects do to one
# step. clock/function_object_write_times.sh has the rule and both loops; here two more rows of laminar/
# damBreak under its own clock, each replayed from OpenFOAM's log TO THE BIT
# (tests/test_time_controls_cadence_log.cu, the staged controlDict read by the solver's own readers):
#   two      `first` (the short name `adjustable`, 0.0131) then `second` (adjustableRunTime 0.013,
#            nStepsToStartTimeChange 6). Each trims what the one before left, so the LAST one decides a step
#            both reach for: the run lands on 0.013, where `first`'s write time is a tenth of a millisecond
#            away -- it asks for a step a twentieth of the one it is handed and is held at HALF of it, an
#            object's own factor where Time::adjustDeltaT's is 5.
#   window   `windowed` (0.013, timeStart 0.03, timeEnd 0.07): trims only while active, and its index stands
#            still outside; `off` (`enabled no`): no object; `older` (outputControl / outputInterval, the names
#            before v1606): read as the write control; `byRunTime` (writeControl runTime): lands on nothing.
# Three checks: each row's log held to the bit, with the objects read (their names, in order), the rows they
# decide and, on `two`, the rows held at half counted, and the host loop's own run taking OpenFOAM's steps;
# and the CONTROLS, BRAE_CONTROL_FUNCTION_OBJECT_STEP=<rule broken> through the replay -- `reversed`, `clip5`
# and `near3` on `two`, `window` on `window` -- each putting rows off the log.
# MEASURED 2026-10-07: `two` 45 steps, 37 the objects', 1 held at half; `window` 38 steps, 20 the objects'
# (`windowed` and `older` read, `off` and `byRunTime` not); every row OpenFOAM's bits and the host loop's own
# steps OpenFOAM's exactly. Controls, rows off the log: `reversed` 21 (worst 1.1e+00), `clip5` 1 (2.6e-02),
# `near3` 10 (1.1e-01), `window` 11 (5.7e-01). 2 s.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
STOP=0.12
BOUND=1e-13
name="the entries of a function object's write times"
PROBE='type probes; libs (sampling); fields (p); probeLocations ((0.292 0.05 0.0073));'
cat > "$W/fe_two.txt" <<FO
functions
{
    first
    {
        $PROBE
        writeControl    adjustable;
        writeInterval   0.0131;
    }
    second
    {
        $PROBE
        writeControl    adjustableRunTime;
        writeInterval   0.013;
        nStepsToStartTimeChange 6;
    }
}
FO
cat > "$W/fe_window.txt" <<FO
functions
{
    windowed
    {
        $PROBE
        writeControl    adjustableRunTime;
        writeInterval   0.013;
        timeStart       0.03;
        timeEnd         0.07;
    }
    off
    {
        $PROBE
        enabled         no;
        writeControl    adjustableRunTime;
        writeInterval   0.007;
    }
    older
    {
        $PROBE
        outputControl   adjustableRunTime;
        outputInterval  0.017;
    }
    byRunTime
    {
        $PROBE
        writeControl    runTime;
        writeInterval   0.009;
    }
}
FO
for row in two window; do
    clock_stage "$W/fe_$row" "$STOP" "$W/fe_$row.txt" || finish "$name"
    clock_replay "$W/fe_$row" "$W/fe_${row}_replay.txt"
    clock_arm "$W/fe_$row" "$W/fe_${row}_host" host BRAE_X=1
done
r="$W/fe_two_replay.txt"
read -r so sh gh <<< "$(clock_steps "$W/fe_two/log.interFoam" "$W/fe_two_host/log.brae")"
what="[two] $(got "$r" rows) steps, objects $(got "$r" names): $(got "$r" byObject) rows theirs,"
what="$what $(got "$r" byClip) held at half, $(got "$r" wrong) not OpenFOAM's bits; [host] $sh steps for $so,"
what="$what deltaT within $gh (bound $BOUND)"
[ "$(got "$r" names)" = first,second ] && [ "$(got "$r" byObject)" -ge 10 ] && [ "$(got "$r" byClip)" -ge 1 ] \
    && [ "$(got "$r" wrong)" = 0 ] && [ "$sh" = "$so" ] && within "$gh" $BOUND && say "$what" ok || say "$what" FAIL
r="$W/fe_window_replay.txt"
read -r so sh gh <<< "$(clock_steps "$W/fe_window/log.interFoam" "$W/fe_window_host/log.brae")"
what="[window] $(got "$r" rows) steps, objects $(got "$r" names): $(got "$r" byObject) rows theirs,"
what="$what $(got "$r" wrong) not OpenFOAM's bits; [host] $sh steps for $so, deltaT within $gh (bound $BOUND)"
[ "$(got "$r" names)" = windowed,older ] && [ "$(got "$r" byObject)" -ge 5 ] && [ "$(got "$r" wrong)" = 0 ] \
    && [ "$sh" = "$so" ] && within "$gh" $BOUND && say "$what" ok || say "$what" FAIL
clock_replay "$W/fe_two" "$W/fe_two_reversed.txt" reversed
clock_replay "$W/fe_two" "$W/fe_two_clip5.txt" clip5
clock_replay "$W/fe_two" "$W/fe_two_near3.txt" near3
clock_replay "$W/fe_window" "$W/fe_window_window.txt" window
a="$W/fe_two_reversed.txt"
b="$W/fe_two_clip5.txt"
n="$W/fe_two_near3.txt"
c="$W/fe_window_window.txt"
what="CONTROL  rows off the log: the objects in reverse order $(got "$a" wrong) (worst $(got "$a" worst)), a"
what="$what decrease held within 5 $(got "$b" wrong) ($(got "$b" worst)), nStepsToStartTimeChange not read"
what="$what $(got "$n" wrong) ($(got "$n" worst)), the window ignored $(got "$c" wrong) ($(got "$c" worst))"
[ "$(got "$a" names)" = second,first ] && [ "$(got "$a" wrong)" -ge 1 ] && above "$(got "$a" worst)" 1e-6 \
    && [ "$(got "$b" wrong)" -ge 1 ] && above "$(got "$b" worst)" 1e-6 \
    && [ "$(got "$n" wrong)" -ge 1 ] && above "$(got "$n" worst)" 1e-6 \
    && [ "$(got "$c" wrong)" -ge 1 ] && above "$(got "$c" worst)" 1e-6 && say "$what" ok || say "$what" FAIL
finish "two objects, the active window and the older names trim the time step as OpenFOAM's do"
