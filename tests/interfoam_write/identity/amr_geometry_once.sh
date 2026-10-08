#!/usr/bin/env bash
# The write gate: the solver's geometry after a change of topology COPIED from the one the refinement has just
# built of the same mesh, against building it again (BRAE_CONTROL_AMR_GEOMETRY_BUILT=1), on
# damBreakWithObstacle's W row -- which refines at both of its steps.
# After a change the refinement's kept addressing holds an FvGeometry of the new mesh (buildAddressing); the
# adapter then assigned that mesh to the solver's and ran FvGeometry::build on it a second time.
# MEASURED 2026-10-05, ms a step: damBreakWithObstacle 5.0 -> 1.0, the step 155.9 -> 151.8; RAS/motorBike
# 2.5 -> 0.5, 127.2 -> 124.6. The same function of the same points and faces gives the same bits, so nothing
# written may change. The copied arm runs with BRAE_CONTROL_AMR_GEOMETRY_CHECK=1: the geometry is built as well
# and its nine arrays compared with the copy, bitwise, at every change. The CONTROL copies the geometry of the
# mesh with its first point moved (BRAE_CONTROL_AMR_GEOMETRY_WRONG=1): the check must stop the run.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakWithObstacle of > "$W/ag_stage.txt" 2>&1
o="$W/w_of_damBreakWithObstacle"
[ -d "$o" ] || { say "damBreakWithObstacle did not stage" FAIL; finish "geometry once a change identity"; }
for v in built copied wrong; do
    e="$W/ag_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        built)  runbrae "$e" device BRAE_CONTROL_AMR_GEOMETRY_BUILT=1 ;;
        copied) runbrae "$e" device BRAE_CONTROL_AMR_GEOMETRY_CHECK=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        wrong)  ( cd "$e" && BRAE_CONTROL_AMR_GEOMETRY_WRONG=1 BRAE_CONTROL_AMR_GEOMETRY_CHECK=1 \
                      "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
mark="the solver's geometry after a change is the refinement's own, copied"
nref=$(grep -a "cells refined" "$W/ag_copied/log.brae" | grep -avc " 0 cells refined")
what="the default arm copies the geometry at $nref refinements, the other arm builds it again"
grep -q "$mark" "$W/ag_copied/log.brae" && ! grep -q "$mark" "$W/ag_built/log.brae" \
    && [ "$nref" -gt 0 ] && say "$what" ok || say "$what" FAIL
n=0
t=0
for d in $(timedirs "$o"); do
    for f in $(cd "$W/ag_built/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/ag_built/$d/$f" "$W/ag_copied/$d/$f" || n=$((n + 1))
    done
done
what="[device] the geometry copied, checked against a build at every change: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/ag_copied")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  the geometry of the mesh with one point moved: the check stops the run and names the array"
[ "$(cat "$W/ag_wrong/exit.txt")" != 0 ] \
    && grep -q "AMR_GEOMETRY_CHECK: of the geometry copied from the refinement's" "$W/ag_wrong/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "after a change the solver's geometry is the refinement's own, and is what building it again gives"
