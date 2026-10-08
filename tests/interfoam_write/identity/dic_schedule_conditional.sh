#!/usr/bin/env bash
# The write gate: the DIC level schedule built ONLY where a pressure solve reads it, on damBreakWithObstacle --
# which refines at both steps of its W row, so the schedule would be built three times.
# The schedule is OpenFOAM's DIC sweep's, and three solvers take it, all the case's own: PCG with DIC, GAMG (its
# fine level) and PCG preconditioned by GAMG. brae's default pressure path is the AMG-PCG (smoother: weighted
# Jacobi), which never reads it; the device loop built it all the same at start-up and after every change of
# topology. MEASURED 2026-10-04, ms a step: damBreakWithObstacle 272 -> 257, RAS/motorBike 197 -> 189.
# Three arms: the default path with the schedule not built against the same path with it built
# (BRAE_CONTROL_DIC_SCHEDULE_ALWAYS=1), byte for byte; the case's own solver (GAMG here), which must still get
# it; and the CONTROL, the case's own solver with the schedule withheld
# (BRAE_CONTROL_DIC_SCHEDULE_NEVER=1), which must refuse by name and not solve without it.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakWithObstacle of > "$W/ds_stage.txt" 2>&1
o="$W/w_of_damBreakWithObstacle"
[ -d "$o" ] || { say "damBreakWithObstacle did not stage" FAIL; finish "DIC schedule conditional identity"; }
# arm <name> <default|case> [env...]: lib.sh exports BRAE_PRESSURE_CASE_SOLVER=1; `default` takes it away
arm()
{
    local e="$W/ds_$1" path="$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    if [ "$path" = default ]; then
        ( unset BRAE_PRESSURE_CASE_SOLVER
          cd "$e" && env "$@" "$BIN" -case . -device > log.brae 2>&1
          echo $? > exit.txt )
    else
        ( cd "$e" && env "$@" "$BIN" -case . -device > log.brae 2>&1
          echo $? > exit.txt )
    fi
}
arm always default BRAE_CONTROL_DIC_SCHEDULE_ALWAYS=1
arm skipped default BRAE_X=1
arm case case BRAE_X=1
arm never case BRAE_CONTROL_DIC_SCHEDULE_NEVER=1
mark="DIC schedule: not built"
pcg="brae runs its AMG-preconditioned PCG"
n=0
t=0
for d in $(timedirs "$o"); do
    for f in $(cd "$W/ds_always/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/ds_always/$d/$f" "$W/ds_skipped/$d/$f" || n=$((n + 1))
    done
done
what="[device] the AMG-PCG path with no DIC schedule built: $t written files, $n differ from the run that builds it"
grep -q "$mark" "$W/ds_skipped/log.brae" && ! grep -q "$mark" "$W/ds_always/log.brae" \
    && grep -q "$pcg" "$W/ds_skipped/log.brae" && grep -q "$pcg" "$W/ds_always/log.brae" \
    && [ "$t" -gt 0 ] && [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
what="[device] the case's own GAMG still gets the schedule and runs to OpenFOAM's last time"
[ "$(cat "$W/ds_case/exit.txt")" = 0 ] && ! grep -q "$mark" "$W/ds_case/log.brae" \
    && ! grep -q "$pcg" "$W/ds_case/log.brae" && [ "$(timedirs "$W/ds_case")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  with the schedule withheld the case's own GAMG refuses by name"
[ "$(cat "$W/ds_never/exit.txt")" != 0 ] \
    && grep -q "the case asks for GAMG and the caller handed in no fine-level DIC schedule" "$W/ds_never/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the DIC schedule is built where a pressure solve reads it and nowhere else"
