#!/usr/bin/env bash
# BRAE_AMG_PCG_SPLIT, the instrument that says where an AMG-PCG solve's time goes: the solve named leaves the
# captured graph for the plain loop, synchronised after every part, and its parts are charged grid by grid in
# BRAE_INTER_PHASE_TIME's table. What it measures is only worth reading if the solve it takes apart IS the
# graph's: the same iterations, the same answer. On waveMakerFlap, brae's default pressure path,
# BRAE_PCORR_AMG=plain (the smoothed hierarchy a moving mesh takes by default does not reproduce run to run).
# MEASURED with it 2026-10-04, pcorr on waveMakerPiston at 896,000 cells, 47.6 iterations a solve, 135 ms a step:
# the mesh's own grid 57% (smoothing 41 ms, residual 16, the two transfers 20), grid 1 11%, grids 2-5 8%, the
# Krylov loop 23% (A p 14, the vector updates 13) -- and the parts waited for one by one sum to the graph's time.
# Three checks: pcorr split and the pressure split each write the graph run's bytes with pcorr's iteration
# counts, each splitting its own solve only; a misspelt name and a split without the table are refused by name;
# and the CONTROL, pcorr cut to one iteration, which the byte comparison must see.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/sp_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "AMG-PCG split instrument"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/sp_$1"
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
        for f in $(cd "$W/sp_$1/$t" && find . -type f | sort); do
            cmp -s "$W/sp_$1/$t/$f" "$W/sp_$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
# its <arm>: pcorr's iteration counts, a solve each
its() { grep -a "Solving for pcorr" "$W/sp_$1/log.brae" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
arm graph BRAE_INTER_PHASE_TIME=1
arm pcorr BRAE_INTER_PHASE_TIME=1 BRAE_AMG_PCG_SPLIT=pcorr
arm pressure BRAE_INTER_PHASE_TIME=1 BRAE_AMG_PCG_SPLIT=pressure
arm misspelt BRAE_INTER_PHASE_TIME=1 BRAE_AMG_PCG_SPLIT=pcor
arm notable BRAE_AMG_PCG_SPLIT=pcorr
arm one BRAE_INTER_PHASE_TIME=1 BRAE_AMG_PCG_SPLIT=pcorr BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION=1
n=$(( $(differs graph pcorr) + $(differs graph pressure) ))
what="[device] the split solve is the graph's: pcorr's iterations $(its pcorr)(graph $(its graph)),"
what="$what $n written files differ"
[ "$(cat "$W/sp_graph/exit.txt")" = 0 ] && [ "$(cat "$W/sp_pcorr/exit.txt")" = 0 ] \
    && [ "$(cat "$W/sp_pressure/exit.txt")" = 0 ] && [ "$(timedirs "$W/sp_graph")" = "$(timedirs "$o")" ] \
    && [ -n "$(its graph)" ] && [ "$(its pcorr)" = "$(its graph)" ] && [ "$(its pressure)" = "$(its graph)" ] \
    && [ "$n" = 0 ] \
    && grep -aq "pcorr split: grid 0 pre-smooth" "$W/sp_pcorr/log.brae" \
    && grep -aq "pcorr split: ALL PARTS" "$W/sp_pcorr/log.brae" \
    && ! grep -aq "pressure split: " "$W/sp_pcorr/log.brae" \
    && grep -aq "pressure split: Krylov, A p" "$W/sp_pressure/log.brae" \
    && ! grep -aq "pcorr split: " "$W/sp_pressure/log.brae" \
    && ! grep -aq " split: " "$W/sp_graph/log.brae" && say "$what" ok || say "$what" FAIL
what="[device] a misspelt solve name, and a split without the phase table, each stop the run and name the variable"
[ "$(cat "$W/sp_misspelt/exit.txt")" != 0 ] \
    && grep -aq "BRAE_AMG_PCG_SPLIT=pcor is not one of pcorr, pressure, all" "$W/sp_misspelt/log.brae" \
    && [ "$(cat "$W/sp_notable/exit.txt")" != 0 ] \
    && grep -aq "BRAE_AMG_PCG_SPLIT is reported in BRAE_INTER_PHASE_TIME's table" "$W/sp_notable/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=$(differs graph one)
what="CONTROL  pcorr cut to one iteration ($(its one)): the written files change ($n of them)"
[ "$(cat "$W/sp_one/exit.txt")" = 0 ] && [ "$n" != 0 ] && [ "$(its one)" != "$(its graph)" ] \
    && say "$what" ok || say "$what" FAIL
finish "the AMG-PCG split instrument takes apart the solve the graph runs"
