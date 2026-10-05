#!/usr/bin/env bash
# The write gate: what the turbulence closure recomputes after a TOPOLOGY CHANGE -- the cell wall distance and
# the wall faces' distance -- measured once and on the fast paths a mesh MOVE already takes, against the paths
# it took, on RAS/motorBike's W row (kOmegaSST, a mesh that refines at every step).
# After a change the host ran its own wave and measured every wall face against its neighbours in one loop; the
# device closure, built again, measured the same faces twice more through nearWallDist. MEASURED 2026-10-05, ms
# a step: the closure built again 14.3 -> 3.2, the wall distance after the change 13.7 -> 10.5, the step
# 159 -> 144. Now: the wave is the GPU's (devicePatchWave, as after a move), the faces are measured ONCE by
# nearWallDistKept for the cell distance's first pass, and the closure's build is handed that measure.
# Every number is the minimum of the same distances and the wave is the host's to the bit, so nothing written
# may change. The `before` arm takes every old path; the default arm runs with the four in-code checks on (the
# faces against nearWallDist, the first pass against its loop, the wave against the host's, the cell-face list
# against the list of lists). The two CONTROLS measure a face against itself alone -- one in the first pass, one
# in the closure's build -- and each check must stop its run.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase motorBike of > "$W/wd_stage.txt" 2>&1
o="$W/w_of_motorBike"
[ -d "$o" ] || { say "motorBike did not stage" FAIL; finish "wall distance after a change identity"; }
CHECKS="BRAE_CONTROL_NEAR_WALL_CHECK=1 BRAE_CONTROL_WALL_DIST_PASS1_CHECK=1 BRAE_CONTROL_PATCH_WAVE_CHECK=1"
CHECKS="$CHECKS BRAE_CONTROL_CELL_FACES_CHECK=1"
for v in before after pass1 build; do
    e="$W/wd_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        before) runbrae "$e" device BRAE_CONTROL_CLOSURE_REFRESH_TWICE=1 BRAE_CONTROL_WALL_DIST_PASS1_SERIAL=1 \
                    BRAE_CONTROL_NEAR_WALL_REMEASURE=1 BRAE_CONTROL_TURBULENCE_WAVE_HOST=1 \
                    BRAE_CONTROL_CELL_FACES_NESTED=1 ;;
        after)  runbrae "$e" device $CHECKS ;;
        # the controls are expected to stop: run directly, runbrae would end the gate
        pass1)  ( cd "$e" && env $CHECKS BRAE_CONTROL_WALL_DIST_PASS1_SELF_ONLY=1 \
                      "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
        build)  ( cd "$e" && env $CHECKS BRAE_CONTROL_NEAR_WALL_SELF_ONLY=1 \
                      "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
shared="a second reader of the same wall geometry is handed the first one's measure"
wave="the turbulence closure's wave runs on the GPU"
nref=$(grep -a "cells refined" "$W/wd_after/log.brae" | grep -avc " 0 cells refined")
what="the default arm takes the GPU wave and the shared measure through $nref refinements, the other arm neither"
grep -q "$shared" "$W/wd_after/log.brae" && ! grep -q "$shared" "$W/wd_before/log.brae" \
    && grep -q "$wave" "$W/wd_after/log.brae" && ! grep -q "$wave" "$W/wd_before/log.brae" \
    && [ "$nref" -gt 0 ] && say "$what" ok || say "$what" FAIL
n=0
t=0
for d in $(timedirs "$o"); do
    for f in $(cd "$W/wd_before/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/wd_before/$d/$f" "$W/wd_after/$d/$f" || n=$((n + 1))
    done
done
what="[device] the new paths with their four checks on, against the old ones: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/wd_after")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROLS a face measured against itself alone, in the first pass and in the closure's build: each is stopped"
[ "$(cat "$W/wd_pass1/exit.txt")" != 0 ] && grep -q "WALL_DIST_PASS1_CHECK: patch" "$W/wd_pass1/log.brae" \
    && [ "$(cat "$W/wd_build/exit.txt")" != 0 ] && grep -q "NEAR_WALL_CHECK: patch" "$W/wd_build/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "after a topology change the closure's wall distances are measured once, on the paths a move takes"
