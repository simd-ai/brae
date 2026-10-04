#!/usr/bin/env bash
# A smoothed-aggregation hierarchy on the single-precision V-cycle -- pcorr's, on a mesh that moves and keeps its
# topology (waveMakerFlap), brae's default pressure path. The single-precision cycle had the plain hierarchy's
# transfers only, so a smoothed one ran the double-precision cycle: every product, sweep and transfer on 8-byte
# values. It now has the sparse prolongator's transfers (restrictSparseT/prolongSparseT<float>, the values cast
# once a hierarchy); the Krylov loop around it, its residual and its stopping test stay double precision.
# MEASURED 2026-10-04, the same iterations at every solve in both arms: pcorr's iterations 110.4 -> 75.5 ms a step
# on waveMakerPiston at 896,000 cells (1,427 iterations in 31 solves), 250.8 -> 160.9 on the 845,536-cell moving
# hull; with BRAE_AMG_PCG_SPLIT, the finest grid's post-smooth 405 -> 206 us an iteration, its residual 359 -> 210,
# its two transfers 397 -> 235.
# What the answer is held to is not here: amg_pcg/pcorr_wave_maker_piston.sh holds this default to OpenFOAM, with
# its own control. A preconditioner cannot change what a converged solve returns, only how many iterations it
# takes -- so this file holds THE ITERATIONS, solve for solve, to the double-precision cycle's
# (BRAE_CONTROL_AMG_SA_DOUBLE=1). One iteration either way is allowed: the smoothed hierarchy's restriction and
# coarse matrices are summed in an order that changes run to run (five runs here and 31 solves at 896,000 cells
# gave equal counts). The CONTROL leaves the single-precision prolongator at zero
# (BRAE_CONTROL_AMG_SA_P_NOT_CAST=1): no coarse correction, and the solves run 23 times as long (1,894 for 81).
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/ss_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "smoothed hierarchy in single precision"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/ss_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
# its <arm>: pcorr's iteration counts, a solve each;  sum <arm>: their total
its() { grep -a "Solving for pcorr" "$W/ss_$1/log.brae" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
sum() { its $1 | tr ' ' '\n' | awk '{s += $1} END {print s + 0}'; }
# near <a> <b>: the two arms' pcorr solves, one for one, each within one iteration of the other
near()
{
    local a=($(its $1)) b=($(its $2)) i d
    [ ${#a[@]} -gt 0 ] && [ ${#a[@]} = ${#b[@]} ] || return 1
    for i in "${!a[@]}"; do
        d=$(( a[i] - b[i] ))
        [ ${d#-} -le 1 ] || return 1
    done
}
split="BRAE_INTER_PHASE_TIME=1 BRAE_AMG_PCG_SPLIT=pcorr"
arm single BRAE_X=1
arm double BRAE_CONTROL_AMG_SA_DOUBLE=1
arm single_split $split
arm double_split $split BRAE_CONTROL_AMG_SA_DOUBLE=1
arm notcast BRAE_CONTROL_AMG_SA_P_NOT_CAST=1
what="[device] pcorr's smoothed hierarchy runs the single-precision cycle, and the double one under its switch"
grep -aq "pcorr: its AMG hierarchy is a smoothed-aggregation one of its own" "$W/ss_single/log.brae" \
    && grep -aq "smoothed aggregation, single-precision cycle" "$W/ss_single_split/log.brae" \
    && grep -aq "pcorr split: Krylov, r and w cast" "$W/ss_single_split/log.brae" \
    && grep -aq "smoothed aggregation, double-precision cycle" "$W/ss_double_split/log.brae" \
    && ! grep -aq "pcorr split: Krylov, r and w cast" "$W/ss_double_split/log.brae" \
    && say "$what" ok || say "$what" FAIL
what="[device] it takes the double-precision cycle's iterations, solve for solve: $(its single)(double $(its double))"
[ "$(cat "$W/ss_single/exit.txt")" = 0 ] && [ "$(cat "$W/ss_double/exit.txt")" = 0 ] \
    && [ "$(timedirs "$W/ss_single")" = "$(timedirs "$o")" ] && [ "$(sum single)" -gt 0 ] \
    && near single double && near single single_split && say "$what" ok || say "$what" FAIL
what="CONTROL  the single-precision prolongator left at zero: $(sum notcast) iterations for $(sum single)"
[ "$(cat "$W/ss_notcast/exit.txt")" = 0 ] && [ "$(sum notcast)" -ge $(( 10 * $(sum single) )) ] \
    && say "$what" ok || say "$what" FAIL
finish "a smoothed-aggregation hierarchy runs the single-precision cycle at the double one's iterations"
