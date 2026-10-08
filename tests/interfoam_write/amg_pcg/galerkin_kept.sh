#!/usr/bin/env bash
# The write gate: a pressure solve whose matrix is the solve before's keeps that solve's coarse matrices and
# their single-precision copies (device_amg.cuh, amgFineCompare; device_inter_pressure_step.cu), on
# damBreakPermeable -- three correctors a step, so two of every three solves.
# In a PISO loop every corrector after the first of a step solves the first one's matrix, bit for bit: MEASURED
# 2026-10-06, 28 of 56 solves on stokesII, 60 of 90 on damBreak. The matrix is compared on the GPU, as bit
# patterns, at every solve; where it is the same the Galerkin products and the casts are left out. The step,
# rebuilt at every solve / kept: stokesII 7.97 / 7.80 ms, capillaryRise 8.30 / 8.03, damBreak 6.17 / 6.03.
# ORACLE: the run that rebuilds at every solve (BRAE_CONTROL_PRESSURE_GALERKIN_ALWAYS=1). What is kept is what
# a rebuild would write, so the two runs have to agree to the byte: every written file and every p_rgh line.
# PREMISE, as numbers from the phase table: the kept arm rebuilds once a step, the other three times.
# CONTROL: the coarse matrices stand WITHOUT the comparison (BRAE_CONTROL_PRESSURE_GALERKIN_UNASKED=1), so the
# first solve's serve every later step's matrix -- the preconditioner is another matrix's and the solve's
# iterates are not the oracle's.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakPermeable of > "$W/gk_stage.txt" 2>&1
o="$W/w_of_damBreakPermeable"
[ -d "$o" ] || { say "damBreakPermeable did not stage" FAIL; finish "coarse matrices kept"; }
dt=$(grep -m1 '^deltaT' "$o/system/controlDict" | tr -d ';' | awk '{print $2}')
t6=$(python3 -c "print('%.12g' % (6*float('$dt')))")
for v in kept always unasked; do
    e="$W/gk_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i -E "s/^endTime .*/endTime         $t6;/; s/^writeInterval .*/writeInterval   6;/" "$e/system/controlDict"
    case $v in
        kept)    runbrae "$e" device BRAE_INTER_PHASE_TIME=1 BRAE_PRINT_TURB_SOLVES=1 ;;
        always)  runbrae "$e" device BRAE_INTER_PHASE_TIME=1 BRAE_PRINT_TURB_SOLVES=1 \
                     BRAE_CONTROL_PRESSURE_GALERKIN_ALWAYS=1 ;;
        unasked) runbrae "$e" device BRAE_PRINT_TURB_SOLVES=1 BRAE_CONTROL_PRESSURE_GALERKIN_UNASKED=1 ;;
    esac
    grep -a "Solving for p_rgh" "$e/log.brae" > "$W/gk_$v.solves"
done
# rebuilt <dir>: how many times a step the coarse matrices were rebuilt, from the phase table's row
rebuilt()
{
    grep -a "pressure: the coarse matrices rebuilt" "$1/log.brae" | sed -E 's/.*\(([0-9.]+) calls\/step\).*/\1/'
}
differs()
{
    local n=0 t f
    for t in $(timedirs "$1"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
mark="keeps that solve's coarse matrices"
rk=$(rebuilt "$W/gk_kept")
ra=$(rebuilt "$W/gk_always")
what="the premise: the coarse matrices rebuilt ${rk:-?} times a step where the matrix is compared, ${ra:-?} where not"
[ "${rk:-0}" = 1.0 ] && [ "${ra:-0}" = 3.0 ] && grep -aq "$mark" "$W/gk_kept/log.brae" \
    && ! grep -aq "$mark" "$W/gk_always/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=$(differs "$W/gk_kept" "$W/gk_always")
nt=$(for t in $(timedirs "$W/gk_always"); do find "$W/gk_always/$t" -type f; done | wc -l)
ns=$(wc -l < "$W/gk_always.solves")
nd=$(diff "$W/gk_kept.solves" "$W/gk_always.solves" | grep -c '^<')
what="[device] kept against rebuilt: $nt written files, $n differ; $ns p_rgh solves, $nd lines differ"
[ "$n" = 0 ] && [ "$nt" -gt 0 ] && [ "$ns" -ge 18 ] && [ "$nd" = 0 ] \
    && [ "$(timedirs "$W/gk_kept")" = "$t6 " ] && say "$what" ok || say "$what" FAIL
nc=$(diff "$W/gk_unasked.solves" "$W/gk_always.solves" | grep -c '^<')
fc=$(differs "$W/gk_unasked" "$W/gk_always")
what="CONTROL  the first solve's coarse matrices kept for good: $nc of $ns p_rgh lines and $fc files change"
[ "$nc" -gt 0 ] && [ "$fc" -gt 0 ] && grep -aq "CONTROL MODE: the coarse matrices stand" "$W/gk_unasked/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "a solve on the solve before's matrix keeps its coarse matrices, and is the rebuilt solve"
