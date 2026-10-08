#!/usr/bin/env bash
# BRAE_AMG_TSGS on a symmetric pressure: the two-stage Gauss-Seidel smoother it asks for is RUN. It was silently
# ignored -- the single-precision cycle was chosen whatever the switch said, and that cycle has weighted Jacobi
# only -- so an arm with BRAE_AMG_TSGS=1 measured Jacobi under another name: on RAS/DTCHull 1,072 iterations with
# it and 1,072 without, 853 once BRAE_AMG_FP32=0 was set beside it (2026-10-04). The switch now takes the
# double-precision cycle, where the smoother exists (amgSinglePrecisionCycle). And BRAE_AMG_TSGS=0, which turns
# the smoother off on an asymmetric matrix, read as "set" on a symmetric one and turned it ON under
# BRAE_AMG_FP32=0; 0 is off on both now.
# On capillaryRise (a mesh at rest; p_rgh's AMG-PCG reproduces byte for byte), brae's default pressure path.
# Measured here, p_rgh's six solves: Jacobi 64 52 46 62 54 47, two-stage GS 61 48 44 58 51 43.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase capillaryRise of > "$W/ts_stage.txt" 2>&1
o="$W/w_of_capillaryRise"
[ -d "$o" ] || { say "capillaryRise did not stage" FAIL; finish "two-stage GS honoured"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/ts_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
# differs <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$W/ts_$1/$t" && find . -type f | sort); do
            cmp -s "$W/ts_$1/$t/$f" "$W/ts_$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
its() { grep -a "Solving for p_rgh" "$W/ts_$1/log.brae" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
arm jacobi BRAE_X=1
arm asked BRAE_AMG_TSGS=1
arm asked_double BRAE_AMG_TSGS=1 BRAE_AMG_FP32=0
arm off BRAE_AMG_TSGS=0
arm double BRAE_AMG_FP32=0
arm off_double BRAE_AMG_TSGS=0 BRAE_AMG_FP32=0
n=$(differs asked asked_double)
what="[device] BRAE_AMG_TSGS=1 runs the double-precision cycle's two-stage GS: $n files from that cycle forced"
[ "$(cat "$W/ts_asked/exit.txt")" = 0 ] && [ "$(timedirs "$W/ts_asked")" = "$(timedirs "$o")" ] \
    && [ -n "$(its asked)" ] && [ "$(its asked)" = "$(its asked_double)" ] && [ "$n" = 0 ] \
    && say "$what" ok || say "$what" FAIL
n=$(( $(differs off jacobi) + $(differs off_double double) ))
what="[device] BRAE_AMG_TSGS=0 is the default smoother in both precisions: $n written files differ"
[ "$(its off)" = "$(its jacobi)" ] && [ "$(its off_double)" = "$(its double)" ] && [ "$n" = 0 ] \
    && say "$what" ok || say "$what" FAIL
n=$(differs asked jacobi)
what="CONTROL  the smoother is seen: two-stage GS $(its asked)against Jacobi $(its jacobi)($n files differ)"
[ "$(its asked)" != "$(its jacobi)" ] && [ "$n" != 0 ] && say "$what" ok || say "$what" FAIL
finish "a two-stage Gauss-Seidel asked for on a symmetric pressure is run, and 0 turns it off"
