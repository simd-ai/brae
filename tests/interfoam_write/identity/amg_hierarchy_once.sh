#!/usr/bin/env bash
# The write gate: a changed mesh's AMG hierarchy built ONCE and copied to the second solve that asks for it,
# against each solve building its own, on damBreakWithObstacle -- which refines at both steps of its W row.
# After a change of topology pcorr's solve (CorrectPhi) and then p_rgh's each asked deviceAmgPcgHierarchy for the
# same mesh's hierarchy and each built it. Its inputs are the internal faces' owner, neighbour and |Sf| and the
# cell count -- nothing of either matrix -- so the second is the first: it now gets a device-to-device copy of
# the structure (AmgHierarchyMemo, cloneAMG) and keeps its own per-solve state.
# MEASURED 2026-10-04, ms a step with the refinement's addressing already kept: damBreakWithObstacle 430 -> 402,
# RAS/motorBike 266 -> 250; the hierarchy's share 68 -> 40 and 42 -> 25.
# This gate runs brae's DEFAULT pressure path: under BRAE_PRESSURE_CASE_SOLVER (what lib.sh exports) no hierarchy
# is built at all. The default arm runs with BRAE_CONTROL_AMG_HIERARCHY_CHECK=1: every copy handed out is built
# as well and compared buffer by buffer. The CONTROL hands the held structure out without asking whether it is
# this mesh's (BRAE_CONTROL_AMG_HIERARCHY_STALE=1): the check must stop the run.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakWithObstacle of > "$W/ho_stage.txt" 2>&1
o="$W/w_of_damBreakWithObstacle"
[ -d "$o" ] || { say "damBreakWithObstacle did not stage" FAIL; finish "AMG hierarchy once identity"; }
for v in rebuilt once stale; do
    e="$W/ho_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        rebuilt) runbrae "$e" device BRAE_CONTROL_AMG_HIERARCHY_REBUILT=1 ;;
        once)    runbrae "$e" device BRAE_CONTROL_AMG_HIERARCHY_CHECK=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        stale)   ( cd "$e" && BRAE_CONTROL_AMG_HIERARCHY_STALE=1 BRAE_CONTROL_AMG_HIERARCHY_CHECK=1 \
                       "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
mark="a changed mesh's is built once and copied"
pcg="brae runs its AMG-preconditioned PCG"
what="both arms run the AMG-PCG; the default one copies the hierarchy after a change, the other builds it twice"
grep -q "$mark" "$W/ho_once/log.brae" && ! grep -q "$mark" "$W/ho_rebuilt/log.brae" \
    && grep -q "$pcg" "$W/ho_once/log.brae" && grep -q "$pcg" "$W/ho_rebuilt/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=0
t=0
for d in $(timedirs "$o"); do
    for f in $(cd "$W/ho_rebuilt/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/ho_rebuilt/$d/$f" "$W/ho_once/$d/$f" || n=$((n + 1))
    done
done
what="[device] the hierarchy built once, each copy checked against a build: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/ho_once")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  handed the hierarchy of the mesh before the change, the check stops the run"
[ "$(cat "$W/ho_stale/exit.txt")" != 0 ] \
    && grep -q "AMG_HIERARCHY_CHECK: of the copied hierarchy" "$W/ho_stale/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "a changed mesh's AMG hierarchy built once is the one each solve built"
