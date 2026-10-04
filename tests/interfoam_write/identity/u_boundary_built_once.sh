#!/usr/bin/env bash
# The write gate: U's device boundary built whole ONCE after a mesh move, against the two builds there were, on
# waveMakerFlap.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 the boundary the device loop builds whole after a mesh change is recorded as the momentum
# hook's own (its key, the geometry count, the addressing): the hook used to find its key changed and build the
# same boundary whole again, and now refreshes the state on it. MEASURED on waveMakerPiston refined to 896,000
# cells: each whole build 44 ms, two a step; the momentum hook 90 -> 48 ms a step, the step 1,327 -> 1,282.
# Every written file BYTE-IDENTICAL against the hook building it again (BRAE_CONTROL_U_BOUNDARY_REBUILT_IN_HOOK=1)
# on eleven cases: three wave makers, sloshingTank2D and 3D6DoF, floatingObject, testTubeMixer, sloshingCylinder,
# DTCHullMovingCoarse, damBreakWithObstacle (refinement) and oscillatingBox (refinement and motion) -- the
# default arm with BRAE_CONTROL_U_EMPTY_CHECK=1 on, the host comparing the device's state at every call.
# The control records a boundary that was NOT built again after the move
# (BRAE_CONTROL_U_BOUNDARY_OLD_GEOMETRY=1), so the hook keeps the old mesh's patch deltas, areas and normals: what
# a wrong record would do, and the files must change. (Leaving the STATE stale, BRAE_CONTROL_U_BOUNDARY_STALE=1,
# changes nothing on this case: the build after the move already holds the step's wall velocity.)
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/ub1_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "U boundary built once identity"; }
for v in once twice old; do
    e="$W/ub1_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        once)  runbrae "$e" device BRAE_CONTROL_U_EMPTY_CHECK=1 ;;
        twice) runbrae "$e" device BRAE_CONTROL_U_BOUNDARY_REBUILT_IN_HOOK=1 ;;
        old)   runbrae "$e" device BRAE_CONTROL_U_BOUNDARY_OLD_GEOMETRY=1 ;;
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
mark="built whole after a mesh change is the momentum hook's too"
grep -q "$mark" "$W/ub1_once/log.brae" && ! grep -q "$mark" "$W/ub1_twice/log.brae" \
    && say "the default run keeps the build made after the move, the other arm builds it again in the hook" ok \
    || say "the default run keeps the build made after the move, the other arm builds it again in the hook" FAIL
n=$(differs "$W/ub1_once" "$W/ub1_twice")
[ "$n" = 0 ] \
    && say "[device] one whole build a move writes the files two did, byte for byte" ok \
    || say "[device] one whole build a move writes the files two did, byte for byte ($n differ)" FAIL
n=$(differs "$W/ub1_old" "$W/ub1_twice")
[ "$n" != 0 ] \
    && say "CONTROL  a boundary recorded without being built for the moved mesh changes $n written files" ok \
    || say "CONTROL  a boundary recorded without being built for the moved mesh changes the written files" FAIL
finish "U's device boundary built once after a mesh move is the one built twice"
