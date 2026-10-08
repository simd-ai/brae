#!/usr/bin/env bash
# RAS/mixerVesselAMI's GPU loop writes the same bytes twice. It did not: a kernel that adds a pair face's term
# to its own cell did it with atomicAdd, so a cell that owns SEVERAL faces of the pair was summed in the order
# the threads arrived, and this mesh has 21,392 such cells (6,742 with their faces in different thread blocks).
# MEASURED 2026-10-07 on this row, two runs of one binary: 25 of 34 files differed from the first step on, the
# host loop's none; the first operation that was not the same twice was the alpha corrector's gradient
# (deviceCyclicAddGrad -- found with BRAE_INTER_STEP_CHECK=1's stage hashes). Now a cell's faces are summed in
# face order: the pair's sixteen assembly kernels and the products' interface terms a launch a place
# (DeviceCyclic::ifRank), the AMG's grids by a list of the faces by own cell (AMGPair::ownCells).
# What the answer is held to is not here: ami/ and the tutorial's own row hold this case to OpenFOAM.
# Three checks. (1) brae's default paths (the AMG-PCG pressure, the pair on every grid), twice: the same bytes.
# (2) the tests' paths (the case's own pressure solver), twice: the same bytes. (3) CONTROL:
# BRAE_CONTROL_PAIR_SUMS_REVERSED=1 takes a cell's faces last first -- another fixed order -- and the written
# files have to change: the comparison of (1) sees the order of a sum. The premise, asserted: the log says the
# mesh has cells that own several faces of the pair.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase mixerVesselAMI of > "$W/ps_stage.txt" 2>&1
o="$W/w_of_mixerVesselAMI"
[ -d "$o" ] || { say "mixerVesselAMI did not stage" FAIL; finish "the pair's sums in a fixed order"; }
# arm <name> <default|tests> [env...]: brae's GPU loop on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/ps_$1" paths="$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    if [ "$paths" = default ]; then
        ( cd "$e" && env -u BRAE_PRESSURE_CASE_SOLVER -u BRAE_TURBULENCE_CASE_SOLVER "$@" "$BIN" -case . -device \
              > log.brae 2>&1; echo $? > exit.txt )
    else
        ( cd "$e" && env "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
    fi
}
# differs <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$W/ps_$1/$t" && find . -type f | sort); do
            cmp -s "$W/ps_$1/$t/$f" "$W/ps_$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
ran() { [ "$(cat "$W/ps_$1/exit.txt")" = 0 ] && [ "$(timedirs "$W/ps_$1")" = "$(timedirs "$o")" ]; }
amgpcg() { grep -ac "brae runs its AMG-preconditioned PCG" "$W/ps_$1/log.brae"; }
arm first default BRAE_X=1
arm second default BRAE_X=1
arm tests_first tests BRAE_X=1
arm tests_second tests BRAE_X=1
arm reversed default BRAE_CONTROL_PAIR_SUMS_REVERSED=1
n=$(differs first second)
what="[device] brae's default paths, twice: $n written files differ; the AMG-PCG's notice $(amgpcg first) times"
ran first && ran second && [ "$n" = 0 ] && [ "$(amgpcg first)" -ge 1 ] \
    && grep -aq "coupled pair: [0-9]* cells own several of its faces" "$W/ps_first/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=$(differs tests_first tests_second)
what="[device] the case's own pressure solver, twice: $n written files differ"
ran tests_first && ran tests_second && [ "$n" = 0 ] && [ "$(amgpcg tests_first)" = 0 ] \
    && say "$what" ok || say "$what" FAIL
n=$(differs first reversed)
what="CONTROL  a cell's faces summed last first: $n written files change"
ran reversed && [ "$n" != 0 ] && say "$what" ok || say "$what" FAIL
finish "a cell that owns several faces of a coupled pair is summed in face order"
