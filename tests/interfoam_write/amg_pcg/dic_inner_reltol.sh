#!/usr/bin/env bash
# The pressure rule's stop on a non-final entry that names `PCG` with `DIC`, on laminar/waves/stokesI AS SHIPPED:
# the tutorial's own fvSolution (p_rgh PCG+DIC relTol 0.1, p_rghFinal GAMG) and its own time-step control
# (adjustTimeStep, maxCo 0.65, maxDeltaT 0.05), to t = 0.5. No other gate runs the AMG-PCG under a tutorial's own
# tolerances: the others pin every solve at 1e-13 (_common.sh), where the stop never matters.
# THE ORACLE is the number of time steps serial OpenFOAM takes to the same instant. An AMG-PCG stopped at the
# entry's relTol leaves its error at the interface; it becomes air velocity and maxCo cuts the step. brae caps the
# stop at relTol 1e-3 (AmgPcgKnobs::dicInnerRelTol, device_inter_pressure_step.cuh) and says so.
# MEASURED 2026-10-04: OpenFOAM 14 steps, brae 14; the CONTROL BRAE_PRESSURE_DIC_INNER_RELTOL=case (the entry's
# own relTol) 50. OpenFOAM itself with GAMG named for p_rgh has the control's Courant number: 1.897 at step 2
# against brae's 1.923, where its own DIC-PCG has 0.001.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
src="$TUT/multiphase/interFoam/laminar/waves/stokesI"
stage_allrun "$src" "$W/dir_stage" "" > "$W/dir_stage.txt" 2>&1 \
    || { say "stokesI did not stage" FAIL; finish "the non-final PCG+DIC entry's stop on the AMG-PCG"; }
# arm <name>: the staged mesh with the tutorial's OWN fvSolution and controlDict, run to t = 0.5 and not written
arm()
{
    local d="$W/dir_$1"
    rm -rf "${d:?}"
    mkdir -p "$d"
    cp -r "$W/dir_stage/0" "$W/dir_stage/constant" "$W/dir_stage/system" "$d/"
    cp "$src/system/fvSolution" "$d/system/fvSolution"
    python3 - "$src/system/controlDict" "$d/system/controlDict" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}', '\nfunctions\n{\n}', s, flags=re.S)
for k, v in [('writeControl', 'timeStep'), ('writeInterval', '1000000'), ('startFrom', 'startTime'),
             ('stopAt', 'endTime'), ('endTime', '0.5')]:
    s = re.sub(r'^%s\s.*' % k, '%s %s;' % (k.ljust(15), v), s, flags=re.M)
open(sys.argv[2], 'w').write(s)
PY
}
arm of
runof "$W/dir_of"
arm cap
runbrae "$W/dir_cap" device
arm case
runbrae "$W/dir_case" device BRAE_PRESSURE_DIC_INNER_RELTOL=case
no=$(grep -c "^Time = " "$W/dir_of/log.interFoam")
nb=$(grep -ac "^ *t = .* dt = " "$W/dir_cap/log.brae")
nc=$(grep -ac "^ *t = .* dt = " "$W/dir_case/log.brae")
what="[device] stokesI as shipped to t = 0.5: the AMG-PCG takes $nb time steps, serial OpenFOAM $no (within one)"
[ "$no" -ge 10 ] && [ "$nb" -le $((no + 1)) ] && [ "$nb" -ge $((no - 1)) ] && say "$what" ok || say "$what" FAIL
mark="the AMG-PCG stops it at relTol 0.001"
what="[device] the log announces the cap on the non-final PCG+DIC entry, and the control arm does not take it"
grep -q "$mark" "$W/dir_cap/log.brae" && ! grep -q "$mark" "$W/dir_case/log.brae" \
    && grep -q "brae runs its AMG-preconditioned PCG" "$W/dir_case/log.brae" && say "$what" ok || say "$what" FAIL
what="CONTROL  stopped at the entry's own relTol 0.1 the same run takes $nc steps, over twice OpenFOAM's $no"
[ "$nc" -gt $((2 * no)) ] && say "$what" ok || say "$what" FAIL
finish "a non-final PCG+DIC pressure entry on the AMG-PCG takes OpenFOAM's time steps"
