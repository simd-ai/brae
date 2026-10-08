#!/usr/bin/env bash
# The write gate: the turbulence closure's near-wall distance measured ONCE a mesh move, with its neighbour lists
# kept and the distances on the host's threads, against the two measurements it replaces, on DTCHullMovingCoarse
# (a mesh that moves; kOmegaSST with wall functions on the hull).
# After every move the device's closure rebuilt its wall data, which measures each wall face's cell centre
# against the face and every wall face sharing a vertex, and then measured the same distance AGAIN for the
# faces' y -- each time rebuilding the map of which wall faces share a point. MEASURED 2026-10-05 on
# RAS/mixerVesselAMI (894,950 cells): 56.5 + 52.0 ms a step, 8.6 with this, the step 1,297.9 -> 1,210.5.
# The minimum of the same distances is the same number in any order, so nothing written may change.
# The comparison arm runs with BRAE_CONTROL_NEAR_WALL_CHECK=1 (every face's distance compared with
# nearWallDist's, bitwise, at every move), once on the default thread count and once on three. The CONTROL
# measures each face against itself alone (BRAE_CONTROL_NEAR_WALL_SELF_ONLY=1), which the check must stop.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase DTCHullMovingCoarse of > "$W/nw_stage.txt" 2>&1
o="$W/w_of_DTCHullMovingCoarse"
[ -d "$o" ] || { say "DTCHullMovingCoarse did not stage" FAIL; finish "near-wall distance once identity"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/nw_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
# differs <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$W/nw_$1/$t" && find . -type f | sort); do
            cmp -s "$W/nw_$1/$t/$f" "$W/nw_$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
arm twice BRAE_CONTROL_CLOSURE_REFRESH_TWICE=1
arm once BRAE_CONTROL_NEAR_WALL_CHECK=1
arm three BRAE_CONTROL_NEAR_WALL_CHECK=1 BRAE_NEAR_WALL_THREADS=3
arm selfonly BRAE_CONTROL_NEAR_WALL_CHECK=1 BRAE_CONTROL_NEAR_WALL_SELF_ONLY=1
n=$(differs twice once)
what="[device] the distance measured once, checked against nearWallDist at every move: $n written files differ"
[ "$(cat "$W/nw_twice/exit.txt")" = 0 ] && [ "$(cat "$W/nw_once/exit.txt")" = 0 ] \
    && [ "$(timedirs "$W/nw_once")" = "$(timedirs "$o")" ] && [ -n "$(timedirs "$o")" ] && [ "$n" = 0 ] \
    && say "$what" ok || say "$what" FAIL
n=$(differs twice three)
what="[device] ...and on three threads, whose ranges divide the wall faces unevenly: $n written files differ"
[ "$(cat "$W/nw_three/exit.txt")" = 0 ] && [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
what="CONTROL  a wall face measured against itself alone: the check stops the run and names the face"
[ "$(cat "$W/nw_selfonly/exit.txt")" != 0 ] \
    && grep -aq "NEAR_WALL_CHECK: patch .* face" "$W/nw_selfonly/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the closure's near-wall distance measured once a move is the one it measured twice"
