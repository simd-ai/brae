#!/usr/bin/env bash
# The gate: WHEN a function object writes. functionObjectList::execute calls execute() and write() on every
# object at every step; an object with a timing entry is wrapped (functionObjects::timeControl,
# timeControlFunctionObject.C) and the wrapper passes the call on only inside the active window and when its
# control fires (Foam::timeControl::execute, timeControl.C:170-278).
# The row is laminar/damBreak as shipped (adjustTimeStep, `writeControl adjustable; writeInterval 0.05`) to
# t = STOP, with a probes object for each rule -- the rows a file holds ARE the steps the rule fired at:
#   everyThird  writeControl timeStep; writeInterval 3     Time's index a multiple of 3
#   byRunTime   writeControl runTime; writeInterval 0.02   the half-step index moves; the step is NOT trimmed
#   adjusted    writeControl adjustableRunTime; writeInterval 0.013   the same index, and the step IS trimmed
#               (the clock's copy of the index must be the wrapper's: the solver stops if they part)
#   windowed    timeStart 0.03; timeEnd 0.07               every step inside the window, half a step each way
#   atEnd       writeControl onEnd                          the step that ends the run
#   older       outputControl outputTime; outputInterval 2  the older names: every second write time
#   disabled    enabled false                               not built: no directory
# Three checks. (1) THE HOST LOOP: each object's file has OpenFOAM's rows -- its time column as text -- and its
# values within HOST_BOUND; `disabled` leaves nothing. (2) THE GPU LOOP, the same. (3) CONTROLS,
# BRAE_CONTROL_FUNCTION_OBJECT_TIMING=<rule broken>: `noHalfStep` (a runTime index without the half step) and
# `window` (active whatever timeStart and timeEnd say) each give a file other rows.
# MEASURED 2026-10-08: 38 steps; OpenFOAM's rows everyThird 12, byRunTime 6, adjusted 9, windowed 8, atEnd 1,
# older 1, and brae's the same on both loops with the values within 3.1e-14 (host) and 1.4e-14 (GPU).
# Controls: `noHalfStep` puts 5 of byRunTime's 6 rows at another time; `window` writes 38 rows for 8.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
. "$(dirname "$0")/_common.sh"
STOP=0.12
HOST_BOUND=4e-13
DEVICE_BOUND=4e-13
name="when a function object writes"
o="$W/tm_of"
{
    echo "functions"
    echo "{"
    while read -r obj entries; do
        echo "    $obj"
        echo "    {"
        echo "        type            probes;"
        echo "        libs            (sampling);"
        echo "        fields          (p);"
        echo "        probeLocations  ((0.1 0.1 0.0073));"
        echo "        $entries"
        echo "    }"
    done <<'OBJECTS'
everyThird writeControl timeStep; writeInterval 3;
byRunTime writeControl runTime; writeInterval 0.02;
adjusted writeControl adjustableRunTime; writeInterval 0.013;
windowed timeStart 0.03; timeEnd 0.07;
atEnd writeControl onEnd;
older outputControl outputTime; outputInterval 2;
disabled enabled false;
OBJECTS
    echo "}"
} > "$W/tm_functions.txt"
fo_stage "$o" "$LAM" "$STOP" adjustable 0.05 "$W/tm_functions.txt" || finish "$name"
objects="everyThird byRunTime adjusted windowed atEnd older"
rowsof() { grep -vc '^#' "$1/postProcessing/$2/0/p" 2>/dev/null || echo 0; }
steps=$(grep -c "^Time = " "$o/log.interFoam")
counts=""
for x in $objects; do counts="$counts $x $(rowsof "$o" $x),"; done
[ "$(rowsof "$o" atEnd)" -ge 1 ] && [ "$(rowsof "$o" older)" = 1 ] && [ ! -e "$o/postProcessing/disabled" ] \
    && [ "$(rowsof "$o" windowed)" -ge 5 ] && [ "$(rowsof "$o" windowed)" -lt "$steps" ] \
    && [ "$(rowsof "$o" byRunTime)" -ge 5 ] && [ "$(rowsof "$o" adjusted)" -ge 8 ] \
    && [ "$(echo $(timedirs "$o") | wc -w)" -ge 2 ] \
    || { say "PREMISE  OpenFOAM's rows of $steps steps:$counts each rule fires on some steps and not all" FAIL
         finish "$name"; }
arm()   # arm <run dir> <label> <bound>
{
    local e="$1" ok=1 worst=0 x r
    for x in $objects; do
        read -r ro rb hd td gap <<< "$(fo_file "$o/postProcessing/$x/0/p" "$e/postProcessing/$x/0/p")"
        [ "$rb" = - ] && { rb=0; gap=1; }
        [ "$ro" = "$rb" ] && [ "$hd" = 0 ] && [ "$td" = 0 ] && within "$gap" "$3" || ok=0
        worst=$(python3 -c "print('%.1e' % max(float('$worst'), float('$gap')))")
    done
    [ ! -e "$e/postProcessing/disabled" ] || ok=0
    what="[$2] OpenFOAM's rows of $steps steps in every file ($(echo $counts | sed 's/,$//')), values within"
    what="$what $worst (bound $3); the disabled object leaves nothing"
    [ $ok = 1 ] && say "$what" ok || say "$what" FAIL
}
fo_arm "$o" "$W/tm_host" host BRAE_X=1
arm "$W/tm_host" host $HOST_BOUND
fo_arm "$o" "$W/tm_device" device BRAE_X=1
arm "$W/tm_device" device $DEVICE_BOUND
fo_arm "$o" "$W/tm_nohalf" host BRAE_CONTROL_FUNCTION_OBJECT_TIMING=noHalfStep
fo_arm "$o" "$W/tm_window" host BRAE_CONTROL_FUNCTION_OBJECT_TIMING=window
read -r ro nh hd tdn gap <<< "$(fo_file "$o/postProcessing/byRunTime/0/p" "$W/tm_nohalf/postProcessing/byRunTime/0/p")"
what="CONTROL  a runTime index without the half step: byRunTime's time column off OpenFOAM's in $tdn rows of"
what="$what $ro; an object active whatever its window: windowed writes $(rowsof "$W/tm_window" windowed) rows"
what="$what for $(rowsof "$o" windowed)"
[ "${tdn/-/0}" -ge 1 ] && [ "$(rowsof "$W/tm_window" windowed)" = "$steps" ] && say "$what" ok || say "$what" FAIL
finish "a function object's timing entries decide its rows as OpenFOAM's wrapper does, on both loops"
