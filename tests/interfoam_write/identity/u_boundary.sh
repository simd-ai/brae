#!/usr/bin/env bash
# The write gate: the momentum hook's refreshed device boundary against a full rebuild, on weirOverflow.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-03 the device loop's updateUBoundary hook rebuilds U's device boundary whole only on its first
# call, after a topology change, or on a mesh that moves; otherwise it builds and uploads the patches' state alone
# (refreshDeviceVectorBoundaryState, one upload a type) -- 76 ms a step to 32 on RAS/DTCHull. MEASURED, refresh against full
# rebuild, every written file BYTE-IDENTICAL on DTCHull, weirOverflow, angledDuct, streamFunction,
# damBreakPermeable and capillaryRise. weirOverflow carries inletOutlet and variableHeightFlowRateInletVelocity.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase weirOverflow of > "$W/ub_stage.txt" 2>&1
o="$W/w_of_weirOverflow"
[ -d "$o" ] || { say "weirOverflow did not stage" FAIL; finish "U boundary identity"; }
for v in refresh full stale; do
    e="$W/ub_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        refresh) runbrae "$e" device ;;
        full)    runbrae "$e" device BRAE_CONTROL_U_BOUNDARY_FULL=1 ;;
        stale)   runbrae "$e" device BRAE_CONTROL_U_BOUNDARY_STALE=1 ;;
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
n=$(differs "$W/ub_refresh" "$W/ub_full")
[ "$n" = 0 ] \
    && say "[device] the refreshed boundary writes the full rebuild's files, byte for byte" ok \
    || say "[device] the refreshed boundary writes the full rebuild's files, byte for byte ($n differ)" FAIL
n=$(differs "$W/ub_stale" "$W/ub_full")
[ "$n" != 0 ] \
    && say "CONTROL  a boundary left unrefreshed changes $n written files" ok \
    || say "CONTROL  a boundary left unrefreshed changes the written files" FAIL
finish "the momentum hook's refreshed device boundary is the full rebuild"
