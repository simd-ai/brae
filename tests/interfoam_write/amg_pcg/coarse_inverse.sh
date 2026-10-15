#!/usr/bin/env bash
# The write gate: the AMG coarsest grid solved by its precomputed inverse (BRAE_AMG_COARSE_SOLVE=inverse) in a real
# pressure solve, against the LU it stands in for (=lu), both named so the gate does not depend on the default.
# Two W rows, six steps each: capillaryRise, and damBreakPorousBaffle, whose cyclic pair adds dense entries to the
# coarsest matrix (the preparation's second site, device_amg_vcycle.cu's pair). The numbers behind the choice:
# tests/test_amg_coarse_inverse.cu (the oracle at the level: a long double LU) and bench_amg_coarse_solve.
# ORACLE: the LU arm. The default cycle is single precision and takes the coarsest solve's FP64 answer cast to
# float, so the inverse's rounding (2.8e-14 of the LU's at worst) does not reach it: every p_rgh / pcorr line and
# every written file must be the LU arm's to the byte (MEASURED 2026-10-15 on seven tutorials to their end time,
# capillaryRise's 51,000 steps apart: the same end state).
# PREMISE: the inverse arm's report (BRAE_AMG_COARSE_REPORT=1) says its coarsest grids were solved by the inverse,
# none by the LU.
# CONTROL: the inverse halved after its check (BRAE_CONTROL_COARSE_INVERSE=half): the solves' lines must change.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
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
for key in capillaryRise damBreakPorousBaffle; do
    wcase $key of > "$W/ci_stage_$key.txt" 2>&1
    o="$W/w_of_$key"
    [ -d "$o" ] || { say "$key did not stage" FAIL; continue; }
    dt=$(grep -m1 '^deltaT' "$o/system/controlDict" | tr -d ';' | awk '{print $2}')
    t6=$(python3 -c "print('%.12g' % (6*float('$dt')))")
    for v in lu inverse half; do
        e="$W/ci_${key}_$v"
        mkdir -p "$e"
        cp -r "$o/0" "$o/constant" "$o/system" "$e/"
        sed -i -E "s/^endTime .*/endTime         $t6;/; s/^writeInterval .*/writeInterval   6;/" "$e/system/controlDict"
        case $v in
            lu)      runbrae "$e" device BRAE_PRINT_TURB_SOLVES=1 BRAE_AMG_COARSE_SOLVE=lu ;;
            inverse) runbrae "$e" device BRAE_PRINT_TURB_SOLVES=1 BRAE_AMG_COARSE_SOLVE=inverse BRAE_AMG_COARSE_REPORT=1 ;;
            half)    runbrae "$e" device BRAE_PRINT_TURB_SOLVES=1 BRAE_AMG_COARSE_SOLVE=inverse BRAE_CONTROL_COARSE_INVERSE=half ;;
        esac
        grep -aE "Solving for (p_rgh|pcorr)" "$e/log.brae" > "$W/ci_${key}_$v.solves"
    done
    ni=$(grep -ac "solved by the inverse" "$W/ci_${key}_inverse/log.brae")
    nl=$(grep -ac "solved by the LU" "$W/ci_${key}_inverse/log.brae")
    what="[device] $key: the premise -- $ni coarsest grid(s) solved by the inverse, $nl by the LU, the substitution named"
    [ "$ni" -gt 0 ] && [ "$nl" -eq 0 ] && grep -aq "coarsest direct solve is 'inverse'" "$W/ci_${key}_inverse/log.brae" \
        && say "$what" ok || say "$what" FAIL
    n=$(differs "$W/ci_${key}_lu" "$W/ci_${key}_inverse")
    nt=$(for t in $(timedirs "$W/ci_${key}_lu"); do find "$W/ci_${key}_lu/$t" -type f; done | wc -l)
    ns=$(wc -l < "$W/ci_${key}_lu.solves")
    what="[device] $key: the inverse arm writes the LU arm's bytes ($n of $nt files differ) and its $ns pressure solves' lines"
    [ "$nt" -gt 0 ] && [ "$ns" -gt 0 ] && [ "$n" -eq 0 ] && cmp -s "$W/ci_${key}_lu.solves" "$W/ci_${key}_inverse.solves" \
        && say "$what" ok || say "$what" FAIL
    what="CONTROL  the inverse halved after its check changes $key's pressure solves"
    grep -aq "CONTROL coarse-inverse-half INJECTED" "$W/ci_${key}_half/log.brae" && ! cmp -s "$W/ci_${key}_lu.solves" "$W/ci_${key}_half.solves" \
        && say "$what" ok || say "$what" FAIL
done
finish "the AMG coarsest grid's precomputed inverse"
