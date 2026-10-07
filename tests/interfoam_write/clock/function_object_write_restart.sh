#!/usr/bin/env bash
# The gate: a function object's write times on a RESTART. Its write control counts from the run's START TIME
# -- the index begins at 0 and the next write time is (index + 1)*interval after the start (timeControl.C
# execute(), timeControlFunctionObject.C:575-584) -- so a continuation lands on other instants than the run it
# continues would have. clock/function_object_write_times.sh has the rule from a cold start.
# The row: laminar/damBreak under its own clock with a `probes` object, run by OpenFOAM to its first write;
# then both codes continue from OPENFOAM'S directory with the object's interval at 0.011, to t = STOP.
# PREMISE: the start time is not a multiple of 0.011, so "from the start" and "from zero" are other instants.
# Two checks: the host loop takes OpenFOAM's steps, each within BOUND of its deltaT, and writes its time
# directories; and the CONTROL, BRAE_CONTROL_FUNCTION_OBJECT_STEP=untrimmed, takes another number of steps.
# MEASURED 2026-10-07: from 0.052, 22 steps for OpenFOAM's 22, every deltaT its printed digits, written at
# 0.102 as OpenFOAM's is; the control 23 steps, a deltaT 3.4e-01 away. (By hand the same day, a continuation
# from 0.0998333 to 0.2 at the first interval: 50 steps for 50.) 2 s.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
STOP=0.12
BOUND=1e-13
name="a function object's write times on a restart"
cat > "$W/fr_functions.txt" <<'FO'
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
first="$W/fr_first"
clock_stage "$first" 0.06 "$W/fr_functions.txt" || finish "$name"
t0=$(timedirs "$first" | awk '{print $1}')
python3 -c "
import sys
t = float('$t0')
sys.exit(0 if t > 0 and abs(t/0.011 - round(t/0.011)) > 0.05 else 1)" \
    || { say "PREMISE  OpenFOAM wrote a start time that is no multiple of 0.011 [$t0]" FAIL; finish "$name"; }
o="$W/fr_of"
mkdir -p "$o"
cp -r "$first/0" "$first/$t0" "$first/constant" "$first/system" "$o/"
sed -i -E "s/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         $STOP;/" "$o/system/controlDict"
sed -i -E "s/^( *writeInterval +)0\.013;/\10.011;/" "$o/system/controlDict"
grep -q "writeInterval  *0.011;" "$o/system/controlDict" && grep -q "^startFrom  *latestTime;" "$o/system/controlDict" \
    || { say "PREMISE  the continuation was staged" FAIL; finish "$name"; }
runof "$o"
# arm <name> [env...]: brae's host loop continuing from OpenFOAM's directory
arm()
{
    local e="$W/fr_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$first/0" "$first/$t0" "$first/constant" "$o/system" "$e/"
    runbrae "$e" host "$@"
}
arm host BRAE_X=1
arm untrimmed BRAE_CONTROL_FUNCTION_OBJECT_STEP=untrimmed
read -r so sh gh <<< "$(clock_steps "$o/log.interFoam" "$W/fr_host/log.brae")"
ot=$(timedirs "$o")
what="[host] from $t0: $sh steps for OpenFOAM's $so, a step's deltaT within $gh (bound $BOUND), written at"
what="$what $(timedirs "$W/fr_host")for OpenFOAM's $ot"
[ "$so" -ge 20 ] && [ "$sh" = "$so" ] && within "$gh" $BOUND && [ "$(timedirs "$W/fr_host")" = "$ot" ] \
    && say "$what" ok || say "$what" FAIL
read -r so su gu <<< "$(clock_steps "$o/log.interFoam" "$W/fr_untrimmed/log.brae")"
what="CONTROL  no object trims: $su steps for $so, a step's deltaT $gu away"
[ "$su" != "$so" ] && above "$gu" 1e-6 && say "$what" ok || say "$what" FAIL
finish "a continuation counts a function object's write times from its own start, as OpenFOAM's does"
