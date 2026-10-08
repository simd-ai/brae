#!/usr/bin/env bash
# The gate: a function object's WRITE TIMES in the time step. Time::adjustDeltaT ends with
# functionObjects_.adjustTimeStep() (Time.C:142), and an object whose `writeControl` is adjustableRunTime trims
# deltaT there so that its own write times are landed on (timeControlFunctionObject.C:560-643) -- after Time's
# own cadence has, by other rules: only when the write time is under `nStepsToStartTimeChange` steps away (3),
# a decrease held within a factor of 2. brae runs no function object and took none of this: the step was the
# Courant number's and Time's alone.
# FOUND 2026-10-07 by a reviewer; no shipped tutorial has such an object (their write controls are timeStep,
# writeTime and onEnd). The row is laminar/damBreak as shipped -- adjustTimeStep, `writeControl adjustable;
# writeInterval 0.05` -- with a `probes` object at `writeControl adjustableRunTime; writeInterval 0.013`, to
# t = STOP.
# Three checks. (1) OPENFOAM'S LOG, a step a row: tests/test_time_controls_cadence_log.cu reads the staged
# controlDict with the solver's own readers and replays the clock -- every row's deltaT must be OpenFOAM's TO
# THE BIT, with rows the object decides and rows Time's own cadence decides both counted. (2) THE TWO LOOPS'
# OWN RUNS take OpenFOAM's number of steps, each step within BOUND of its deltaT, write the time directories
# OpenFOAM writes -- no longer multiples of Time's interval: the object trims last -- and say which object trims.
# (3) CONTROLS, BRAE_CONTROL_FUNCTION_OBJECT_STEP=<rule broken>: `untrimmed` (the clock as it was) puts rows
# off the log and gives the host run another number of steps; `everyStep` (the object trims as Time does, at
# every step) puts rows off the log.
# clock/function_object_write_entries.sh has two objects, the active window and the older entry names.
# MEASURED 2026-10-07: 38 steps, 20 decided by the object and all 38 by Time's own cadence first, every one
# OpenFOAM's bits; both loops 38 steps, each step's printed deltaT OpenFOAM's exactly (the step is the growth
# cap's and the cadences' here, not a Courant number's; the bound is the clock gates' 1e-13), written at 0.052
# and 0.0998333 as OpenFOAM's are (fields there: phi 1.1e-13, U 5.8e-14 on the host). Before the fix brae took
# 37 steps and wrote at 0.05 and 0.1. Controls: `untrimmed` 20 rows off (worst 3.1e-01), 37 steps; `everyStep`
# 18 rows off (1.1e-01). 3 s.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
. "$(dirname "$0")/_common.sh"
STOP=0.12
BOUND=1e-13
name="a function object's write times in the time step"
o="$W/fo_of"
cat > "$W/fo_functions.txt" <<'FO'
functions
{
    probes1
    {
        type            probes;
        libs            (sampling);
        writeControl    adjustableRunTime;
        writeInterval   0.013;
        fields          (p);
        probeLocations  ((0.292 0.05 0.0073));
    }
}
FO
clock_stage "$o" "$STOP" "$W/fo_functions.txt" || finish "$name"
clock_replay "$o" "$W/fo_replay.txt"
r="$W/fo_replay.txt"
what="[OpenFOAM's log] $(got "$r" rows) steps, $(got "$r" byObject) decided by the object ($(got "$r" names)),"
what="$what $(got "$r" byTime) by Time's own cadence: $(got "$r" wrong) not its bits"
[ "$(got "$r" rows)" = "$(grep -c "^Time = " "$o/log.interFoam")" ] && [ "$(got "$r" objects)" = 1 ] \
    && [ "$(got "$r" byObject)" -ge 10 ] && [ "$(got "$r" byTime)" -ge 10 ] && [ "$(got "$r" wrong)" = 0 ] \
    && say "$what" ok || say "$what" FAIL
clock_arm "$o" "$W/fo_host" host BRAE_X=1
clock_arm "$o" "$W/fo_device" device BRAE_X=1
clock_arm "$o" "$W/fo_untrimmed" host BRAE_CONTROL_FUNCTION_OBJECT_STEP=untrimmed
read -r so sh gh <<< "$(clock_steps "$o/log.interFoam" "$W/fo_host/log.brae")"
read -r so sd gd <<< "$(clock_steps "$o/log.interFoam" "$W/fo_device/log.brae")"
ot=$(timedirs "$o")
what="[host] [device] $sh and $sd steps for OpenFOAM's $so, a step's deltaT within $gh and $gd (bound $BOUND),"
what="$what written at $(timedirs "$W/fo_host")and $(timedirs "$W/fo_device")for OpenFOAM's $ot"
[ "$sh" = "$so" ] && [ "$sd" = "$so" ] && within "$gh" $BOUND && within "$gd" $BOUND \
    && [ "$(echo $ot | wc -w)" = 2 ] && [ "$(timedirs "$W/fo_host")" = "$ot" ] \
    && [ "$(timedirs "$W/fo_device")" = "$ot" ] \
    && grep -aq "time step: the write times of function object \`probes1\`" "$W/fo_host/log.brae" \
    && grep -aq "time step: the write times of function object \`probes1\`" "$W/fo_device/log.brae" \
    && say "$what" ok || say "$what" FAIL
clock_replay "$o" "$W/fo_replay_untrimmed.txt" untrimmed
clock_replay "$o" "$W/fo_replay_everystep.txt" everyStep
read -r so su gu <<< "$(clock_steps "$o/log.interFoam" "$W/fo_untrimmed/log.brae")"
u="$W/fo_replay_untrimmed.txt"
v="$W/fo_replay_everystep.txt"
what="CONTROL  no object trims: $(got "$u" wrong) rows off the log (worst $(got "$u" worst)), the host run $su"
what="$what steps for $so, written at $(timedirs "$W/fo_untrimmed"); the object trims at every step:"
what="$what $(got "$v" wrong) rows off (worst $(got "$v" worst))"
[ "$(got "$u" objects)" = 0 ] && [ "$(got "$u" wrong)" -ge 10 ] && [ "$su" != "$so" ] \
    && [ "$(timedirs "$W/fo_untrimmed")" != "$ot" ] \
    && ! grep -aq "time step: the write times" "$W/fo_untrimmed/log.brae" \
    && [ "$(got "$v" objects)" = 1 ] && [ "$(got "$v" wrong)" -ge 1 ] && above "$(got "$v" worst)" 1e-6 \
    && say "$what" ok || say "$what" FAIL
finish "a function object's adjustableRunTime write times trim the time step as OpenFOAM's do, on both loops"
