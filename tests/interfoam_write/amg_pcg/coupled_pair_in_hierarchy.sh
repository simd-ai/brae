#!/usr/bin/env bash
# A coupled pair (cyclic, cyclicAMI) carried on every grid of the AMG hierarchy, on RAS/mixerVesselAMI -- 894,950
# cells, a rotor and a stator joined by a cyclicAMI pair -- brae's default pressure path.
# The hierarchy is built on internal faces, so its cycle preconditioned the two sides as uncoupled blocks and
# left the Krylov loop everything that crosses the pair. Now each grid holds the pair with its cells mapped to
# that grid's (the Galerkin coarse pair under aggregation), every product of the cycle adds its term, and the
# coarsest grid's dense LU has its entries (AMGPair, amgCouplePair).
# MEASURED 2026-10-04 on the tutorial, 30 steps, p_rgh tolerance 1e-6: 6,798 -> 1,926 iterations in 60 solves,
# p_rgh's iterations 624.1 -> 226.9 ms a step, the step 2,508.5 -> 2,093.3. Here, every solve pinned at 1e-13:
# p_rgh's four solves 482 409 467 398 -> 146 122 134 134.
# A preconditioner changes the iteration count and not what a converged solve returns: against OpenFOAM the
# worst file is 1.2e-09 (p) with the pair carried and without (the two runs are 2.5e-12 apart). THE CONTROL
# negates the hierarchy's copy of the pair's coefficients (BRAE_CONTROL_AMG_PAIR_WRONG_SIGN=1): the cycle then
# couples the sides with the opposite sign to the matrix's and is worse than no coupling at all.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase mixerVesselAMI of > "$W/cp_stage.txt" 2>&1
o="$W/w_of_mixerVesselAMI"
[ -d "$o" ] || { say "mixerVesselAMI did not stage" FAIL; finish "coupled pair in the hierarchy"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/cp_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
# its <arm>: p_rgh's iteration counts, a solve each;  sum <arm>: their total
its() { grep -a "Solving for p_rgh" "$W/cp_$1/log.brae" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
sum() { its $1 | tr ' ' '\n' | awk '{s += $1} END {print s + 0}'; }
mark="the matrix's coupled pair .* is carried on every grid of the hierarchy"
arm coupled BRAE_X=1
arm uncoupled BRAE_CONTROL_AMG_PAIR_UNCOUPLED=1
arm wrong BRAE_CONTROL_AMG_PAIR_WRONG_SIGN=1
what="[device] the pair carried on every grid: p_rgh in $(its coupled)iterations (without it $(its uncoupled))"
[ "$(cat "$W/cp_coupled/exit.txt")" = 0 ] && [ "$(cat "$W/cp_uncoupled/exit.txt")" = 0 ] \
    && grep -aq "$mark" "$W/cp_coupled/log.brae" && ! grep -aq "$mark" "$W/cp_uncoupled/log.brae" \
    && [ "$(sum coupled)" -gt 0 ] && [ $(( 2 * $(sum coupled) )) -le "$(sum uncoupled)" ] \
    && say "$what" ok || say "$what" FAIL
python3 "$CMP" "$o" "$W/cp_coupled" $(timedirs "$o") > "$W/cmp_cp_coupled.txt" 2>&1
judge "mixerVesselAMI with the pair in the hierarchy" "$W/cmp_cp_coupled.txt" 1.2e-08 "$o/log.interFoam" \
    && say "[device] ...and every file within 1.2e-08 of OpenFOAM, as without it" ok \
    || say "[device] ...and every file within 1.2e-08 of OpenFOAM, as without it" FAIL
what="CONTROL  the hierarchy's pair with the wrong sign: $(sum wrong) iterations, more than without it"
what="$what ($(sum uncoupled))"
[ "$(sum wrong)" -gt "$(sum uncoupled)" ] && say "$what" ok || say "$what" FAIL
finish "a coupled pair is carried on every grid of the AMG hierarchy"
