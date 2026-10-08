#!/usr/bin/env bash
# The AMG-PCG's plain loop replays its V-cycle from a CUDA graph. A capture bakes every pointer the cycle was
# handed, and the double-precision cycle is handed the caller's matrix itself -- diag, upper, lower -- while the
# guard compared `diag` alone. interFoam's pressure step folds its diagonal into a new buffer at every solve
# and the pool hands the three blocks back in another combination, so the replayed cycle smoothed with the
# off-diagonals of the solve before. FOUND 2026-10-06 through BRAE_CORR_SCALING=1, an experiment switch that
# takes this loop: laminar/waves/stokesI ran every Final solve from its second step to the cap (25,224
# iterations in 27 steps for 309). The default path is another one (deviceAMGPCGGraph copies the matrix into
# buffers of its own) and was never wrong; the two switches that reach this loop were.
# ORACLE: the same loop with the cycle NOT captured (BRAE_USE_GRAPH=0) -- the same kernels in the same order,
# launched one by one, so every solver line has to be the same to the last digit and every written file the
# same bytes. Held twice: the double-precision cycle (BRAE_PCG_DEVICE=0 BRAE_AMG_FP32=0), and the scaled
# coarse correction (BRAE_CORR_SCALING=1), whose notice is the proof that arm took it.
# CONTROL: BRAE_CONTROL_AMG_GRAPH_VIEW_NOT_COMPARED=1 keys the capture on the diagonal alone again: a solve runs
# to its cap and the written files change.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase stokesI of > "$W/cv_stage.txt" 2>&1
o="$W/w_of_stokesI"
[ -d "$o" ] || { say "stokesI did not stage" FAIL; finish "captured V-cycle and the view it read"; }
PLAIN="BRAE_PCG_DEVICE=0 BRAE_AMG_FP32=0 BRAE_PCG_CHECK_EVERY=1"
SCALED="BRAE_CORR_SCALING=1 BRAE_PCG_CHECK_EVERY=1"
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/cv_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1
      echo $? > exit.txt )
}
# differs <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$W/cv_$1/$t" && find . -type f | sort); do
            cmp -s "$W/cv_$1/$t/$f" "$W/cv_$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
lines() { grep -a "Solving for p_rgh" "$W/cv_$1/log.brae"; }
its() { lines "$1" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
cap=$(python3 - "$o/system/fvSolution" <<'PY'
import re, sys
m = re.search(r'maxIter\s+(\d+)', open(sys.argv[1]).read())
print(m.group(1) if m else 1000)
PY
)
# same <a> <b>: both ran to the end, wrote OpenFOAM's times, and agree in every solver line and every byte
same()
{
    [ "$(cat "$W/cv_$1/exit.txt")" = 0 ] && [ "$(cat "$W/cv_$2/exit.txt")" = 0 ] \
        && [ "$(timedirs "$W/cv_$1")" = "$(timedirs "$o")" ] && [ "$(timedirs "$W/cv_$2")" = "$(timedirs "$o")" ] \
        && [ -n "$(lines "$1")" ] && [ "$(lines "$1")" = "$(lines "$2")" ] && [ "$(differs "$1" "$2")" = 0 ] \
        && ! its "$1" | tr ' ' '\n' | grep -qx "$cap"
}
arm captured $PLAIN
arm launched $PLAIN BRAE_USE_GRAPH=0
arm scaled $SCALED
arm scaled_launched $SCALED BRAE_USE_GRAPH=0
arm control $PLAIN BRAE_CONTROL_AMG_GRAPH_VIEW_NOT_COMPARED=1
what="[device] the double-precision cycle captured is the cycle launched: iterations $(its captured)"
what="$what(launched $(its launched)), $(differs captured launched) written files differ"
same captured launched && say "$what" ok || say "$what" FAIL
n=$(grep -ac "p_rgh EXPERIMENT: the AMG-PCG's coarse correction is scaled" "$W/cv_scaled/log.brae")
m=$(grep -ac "p_rgh EXPERIMENT: the AMG-PCG's coarse correction is scaled" "$W/cv_captured/log.brae")
what="[device] the scaled coarse correction captured is the one launched: iterations $(its scaled)"
what="$what(launched $(its scaled_launched)); its notice $n time in that arm, $m in the unscaled one"
same scaled scaled_launched && [ "$n" = 1 ] && [ "$m" = 0 ] && say "$what" ok || say "$what" FAIL
n=$(differs captured control)
what="CONTROL  the capture keyed on the diagonal alone: a solve at the cap of $cap ($(its control)),"
what="$what $n written files change"
[ "$(cat "$W/cv_control/exit.txt")" = 0 ] && its control | tr ' ' '\n' | grep -qx "$cap" && [ "$n" != 0 ] \
    && say "$what" ok || say "$what" FAIL
finish "a captured V-cycle is replayed only on the matrix buffers it was captured on"
