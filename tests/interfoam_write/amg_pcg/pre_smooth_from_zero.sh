#!/usr/bin/env bash
# The V-cycle's first pre-smoothing sweep written straight from b, against the sweep as it was: x zeroed, A*x
# formed for that zero x, then the weighted-Jacobi update. A cycle's x starts at zero, so the product is a zero
# vector and the sweep is omega*b/diag; smoothFromZeroT is the sweep's own expression with the two zeros written
# in, and its result to the bit. The single-precision cycle has done this since the pressure work (ungated until
# now); the double-precision one -- what pcorr's smoothed hierarchy runs -- had not.
# MEASURED 2026-10-04 with BRAE_AMG_PCG_SPLIT, pcorr on waveMakerPiston at 896,000 cells, the same 1,427
# iterations in 31 solves: the pre-smooth of the finest grid 454 -> 69 us an iteration, of all grids 30.3 -> 5.0 ms
# a step; pcorr's iterations 136.0 -> 111.6 ms a step.
# On waveMakerFlap, brae's default pressure path, BRAE_PCORR_AMG=plain (the smoothed hierarchy does not
# reproduce run to run; the sweep is the same code on either), the double-precision cycle forced with
# BRAE_AMG_FP32=0. The CONTROL is the defect this change could have been: the zeroing dropped and the from-zero
# sweep not put in its place (BRAE_CONTROL_AMG_FROM_ZERO_STALE=1), so a cycle starts from the last one's x.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/fz_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "AMG pre-smooth from zero"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/fz_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PCORR_AMG=plain BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1
      echo $? > exit.txt )
}
# differs <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$W/fz_$1/$t" && find . -type f | sort); do
            cmp -s "$W/fz_$1/$t/$f" "$W/fz_$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
# its <arm>: the iteration count of every pressure solve, p_rgh's and pcorr's, in the order they ran
its() { grep -a "Solving for p" "$W/fz_$1/log.brae" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
# same <before> <after>: both ran to the end, wrote the same bytes and took the same iterations at every solve
same()
{
    [ "$(cat "$W/fz_$1/exit.txt")" = 0 ] && [ "$(cat "$W/fz_$2/exit.txt")" = 0 ] \
        && [ "$(timedirs "$W/fz_$2")" = "$(timedirs "$o")" ] && [ -n "$(its $1)" ] \
        && [ "$(its $1)" = "$(its $2)" ] && [ "$(differs $1 $2)" = 0 ]
}
arm d_before BRAE_AMG_FP32=0 BRAE_CONTROL_AMG_ZERO_PRODUCT=1
arm d_zero BRAE_AMG_FP32=0
arm s_before BRAE_CONTROL_AMG_ZERO_PRODUCT=1
arm s_zero BRAE_X=1
arm d_stale BRAE_AMG_FP32=0 BRAE_CONTROL_AMG_FROM_ZERO_STALE=1
what="[device] double-precision cycle, the first sweep from zero: $(differs d_before d_zero) written files differ,"
what="$what the same iterations"
same d_before d_zero && say "$what" ok || say "$what" FAIL
what="[device] single-precision cycle, the first sweep from zero: $(differs s_before s_zero) written files differ,"
what="$what the same iterations"
same s_before s_zero && [ "$(differs d_zero s_zero)" != 0 ] && say "$what" ok || say "$what" FAIL
n=$(differs d_before d_stale)
what="CONTROL  a cycle started from the last one's x: $n written files change, the iterations do too"
[ "$(cat "$W/fz_d_stale/exit.txt")" = 0 ] && [ "$n" != 0 ] && [ "$(its d_stale)" != "$(its d_before)" ] \
    && say "$what" ok || say "$what" FAIL
finish "the V-cycle's first sweep from zero is the sweep it replaces, in both precisions"
