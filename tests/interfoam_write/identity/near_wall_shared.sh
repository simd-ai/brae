#!/usr/bin/env bash
# The write gate: ONE measure of the wall faces' distance a mesh move for its two readers -- the cell wall
# distance's first pass (host) and the closure's wall functions (device) -- against each reader measuring, on
# DTCHullMovingCoarse (a mesh that moves; kOmegaSST with wall functions on the hull).
# Both read the smallest distance from a wall face's cell centre to the face and its vertex neighbours over the
# combined wall patch. nearWallDistKept keeps what it measured with what it measured it ON -- the faces'
# vertices and the cells' centres, by content -- and hands the same numbers to a second reader of the same wall.
# MEASURED 2026-10-05 on RAS/motorBike: the closure's own measure 1.3 ms a step, 0.2 handed.
# The shared arm runs with BRAE_CONTROL_NEAR_WALL_CHECK=1 and BRAE_CONTROL_WALL_DIST_PASS1_CHECK=1 (each
# reader's numbers against its own old loop, bitwise, at every move). The CONTROL hands the kept measure out on
# the faces alone, without asking whether the wall moved (BRAE_CONTROL_NEAR_WALL_STALE=1): a check must stop it.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase DTCHullMovingCoarse of > "$W/ns_stage.txt" 2>&1
o="$W/w_of_DTCHullMovingCoarse"
[ -d "$o" ] || { say "DTCHullMovingCoarse did not stage" FAIL; finish "near-wall distance shared identity"; }
CHECKS="BRAE_CONTROL_NEAR_WALL_CHECK=1 BRAE_CONTROL_WALL_DIST_PASS1_CHECK=1"
for v in each shared stale; do
    e="$W/ns_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        each)   runbrae "$e" device BRAE_CONTROL_NEAR_WALL_REMEASURE=1 ;;
        shared) runbrae "$e" device $CHECKS ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        stale)  ( cd "$e" && env $CHECKS BRAE_CONTROL_NEAR_WALL_STALE=1 \
                      "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
mark="a second reader of the same wall geometry is handed the first one's measure"
what="the shared arm hands the host's measure to the closure, the other arm measures for each reader"
grep -q "$mark" "$W/ns_shared/log.brae" && ! grep -q "$mark" "$W/ns_each/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=0
t=0
for d in $(timedirs "$o"); do
    for f in $(cd "$W/ns_each/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/ns_each/$d/$f" "$W/ns_shared/$d/$f" || n=$((n + 1))
    done
done
what="[device] one measure a move, each reader checked against its own loop: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/ns_shared")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  the kept measure handed out without asking whether the wall moved: a check stops the run"
[ "$(cat "$W/ns_stale/exit.txt")" != 0 ] \
    && grep -q "NEAR_WALL_CHECK: patch\|WALL_DIST_PASS1_CHECK: patch" "$W/ns_stale/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the wall faces' distance measured once a move serves both of its readers"
