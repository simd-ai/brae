#!/usr/bin/env bash
# The write gate: the AMG hierarchy's coarse faces found in LINEAR time (buckets by the lower coarse cell) against
# the global sort and per-face binary search they replace, on damBreakWithObstacle -- which builds a hierarchy at
# start-up and after each of the two refinements of its W row, each of about ten levels.
# agglomerate() needs, a level, the unique (lower, higher) coarse-cell pairs in owner-sorted order and each fine
# face's coarse face. It sorted every crossing face's pair and binary-searched each face into the result.
# MEASURED 2026-10-04 on the 26-step benchmark interval: 27 of the build's 37 ms a step were that; with the
# buckets 3.5, the step 254 -> 231 ms; RAS/motorBike 189 -> 176; RAS/DTCHull's one build at 845,536 cells
# 594 -> 301 ms. The lists are the same, entry for entry, so the hierarchy and the run are the same bytes.
# This gate runs brae's DEFAULT pressure path: under BRAE_PRESSURE_CASE_SOLVER (what lib.sh exports) no
# hierarchy is built. The default arm runs with BRAE_CONTROL_AMG_COARSE_FACES_CHECK=1: both forms at every level
# of every build, compared entry for entry. The CONTROL leaves each bucket unsorted
# (BRAE_CONTROL_AMG_COARSE_FACES_UNSORTED=1): the check must stop the run.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakWithObstacle of > "$W/cf_stage.txt" 2>&1
o="$W/w_of_damBreakWithObstacle"
[ -d "$o" ] || { say "damBreakWithObstacle did not stage" FAIL; finish "AMG coarse faces linear identity"; }
for v in sort bucket unsorted; do
    e="$W/cf_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        sort)     runbrae "$e" device BRAE_CONTROL_AMG_COARSE_FACES_SORT=1 ;;
        bucket)   runbrae "$e" device BRAE_CONTROL_AMG_COARSE_FACES_CHECK=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        unsorted) ( cd "$e" && BRAE_CONTROL_AMG_COARSE_FACES_UNSORTED=1 BRAE_CONTROL_AMG_COARSE_FACES_CHECK=1 \
                        "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
mark="coarse faces are found by the global sort"
pcg="brae runs its AMG-preconditioned PCG"
what="both arms build AMG hierarchies; one finds the coarse faces by the global sort, the default by buckets"
grep -q "$mark" "$W/cf_sort/log.brae" && ! grep -q "$mark" "$W/cf_bucket/log.brae" \
    && grep -q "$pcg" "$W/cf_sort/log.brae" && grep -q "$pcg" "$W/cf_bucket/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=0
t=0
for d in $(timedirs "$o"); do
    for f in $(cd "$W/cf_sort/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/cf_sort/$d/$f" "$W/cf_bucket/$d/$f" || n=$((n + 1))
    done
done
what="[device] the bucket form, checked against the sort at every level: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/cf_bucket")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  with the buckets left unsorted the check stops the run and names the coarse faces"
[ "$(cat "$W/cf_unsorted/exit.txt")" != 0 ] \
    && grep -q "AMG_COARSE_FACES_CHECK: the coarse faces from the buckets are not" "$W/cf_unsorted/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the AMG hierarchy's coarse faces found by buckets are the ones the global sort finds"
