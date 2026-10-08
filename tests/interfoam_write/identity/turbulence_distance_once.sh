#!/usr/bin/env bash
# The write gate: the closure's wall distance computed ONCE a mesh move, against the two computations there were,
# on DTCHullMovingCoarse.
. "$(dirname "$0")/../lib.sh"
# The host mesh update (interMeshUpdate) calls moveInterTurbulence itself, ahead of CorrectPhi -- where
# wallDist::movePoints sits in fvMesh::movePoints -- and the device loop's refresh called it AGAIN for the same
# moved mesh: every move computed kOmegaSST's wall distance twice, the first time with the host's wave. Since
# 2026-10-04 the first call takes the GPU wave (interMeshUpdate's waveRunner) and the second is gone.
# MEASURED on RAS/DTCHullMoving (845,536 cells), ms a step with start-up subtracted: the distance in the host
# update 399 -> 64, in the refresh 107 -> 46, the step 1,678 -> 1,274.
# The default arm runs with BRAE_CONTROL_PATCH_WAVE_CHECK=1, the host's wave compared in every cell. The control
# drops the one call that is left (BRAE_CONTROL_TURBULENCE_DISTANCE_STALE=1): the closure keeps the old mesh's
# distances, and the files must change -- measured, 15 of 36.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase DTCHullMovingCoarse of > "$W/t1_stage.txt" 2>&1
o="$W/w_of_DTCHullMovingCoarse"
[ -d "$o" ] || { say "DTCHullMovingCoarse did not stage" FAIL; finish "turbulence distance once identity"; }
for v in once twice stale; do
    e="$W/t1_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        once)  runbrae "$e" device BRAE_CONTROL_PATCH_WAVE_CHECK=1 ;;
        twice) runbrae "$e" device BRAE_CONTROL_TURBULENCE_DISTANCE_TWICE=1 ;;
        stale) runbrae "$e" device BRAE_CONTROL_TURBULENCE_DISTANCE_STALE=1 ;;
    esac
done
# differs <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
mark="the closure's is computed once a mesh move"
grep -q "$mark" "$W/t1_once/log.brae" && ! grep -q "$mark" "$W/t1_twice/log.brae" \
    && say "the default run computes the closure's distance once a move, the other arm twice" ok \
    || say "the default run computes the closure's distance once a move, the other arm twice" FAIL
n=$(differs "$W/t1_once" "$W/t1_twice")
[ "$n" = 0 ] && [ "$(timedirs "$W/t1_once")" = "$(timedirs "$o")" ] \
    && say "[device] one computation a move writes the files two did, byte for byte" ok \
    || say "[device] one computation a move writes the files two did, byte for byte ($n differ)" FAIL
n=$(differs "$W/t1_stale" "$W/t1_twice")
[ "$n" != 0 ] \
    && say "CONTROL  with no computation after the move the closure's stale distance changes $n written files" ok \
    || say "CONTROL  with no computation after the move the closure's stale distance changes the written files" FAIL
finish "the closure's wall distance computed once a move is the one computed twice"
