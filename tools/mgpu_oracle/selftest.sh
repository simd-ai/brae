#!/bin/bash
# Self-test of the oracle chain on the small sealed tank with every linear solver at 1e-12: OpenFOAM serial vs
# OpenFOAM -parallel (hierarchical 4-way), vs reconstructPar of that run, vs a second decomposition (simple 8-way).
#   tools/mgpu_oracle/selftest.sh        (~1 min, needs setup.sh done; uses OUT_ROOT=/tmp/mgpu_oracle_selftest)
set -eo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
export OUT_ROOT=${OUT_ROOT:-/tmp/mgpu_oracle_selftest} TIGHT=1
CMP="python3 $HERE/compare_fields.py"
"$HERE/run_oracle.sh" tank 4 hier 20 > /dev/null
"$HERE/run_oracle.sh" tank 8 simple 20 > /dev/null
"$HERE/run_oracle.sh" open 4 scotch 20 > /dev/null
A=$OUT_ROOT/tank_np4_hier_tight; B=$OUT_ROOT/tank_np8_simple_tight; O=$OUT_ROOT/open_np4_tight
F=alpha.water,p_rgh,p,U,phi
fail=0
step() {   # <label> <cmd...>: runs it, prints the worst maxrel row, fails the self-test on a non-zero exit
    echo -n "== $1: "; shift
    local rc=0
    "$@" > /tmp/selftest.out || rc=$?
    awk -F'|' 'NR>2 && $8+0 >= m {m=$8+0; w=$4 "/" $5 " @" $2} END {printf "worst maxrel %.2e (%s)", m, w}' /tmp/selftest.out
    [ $rc = 0 ] && echo "  ok" || { echo "  FAIL"; grep EXCEEDS /tmp/selftest.out; fail=1; }
}
# 1. addressing round trip: processorN mapped to global vs OpenFOAM's own reconstructPar: bit-identical
step "decomposed vs reconstructPar (identical up to the two sides' round-off of a cut face)" $CMP $A/par $A/par_rec --fields $F --tol 1e-14
# 2. serial vs parallel, step 1 and step 20, solver-converged: operators agree to round-off / solver tolerance
step "serial vs np=4 hier, step 1 (tol 1e-8)" $CMP $A/serial_d1 $A/par_d1 --fields $F --tol 1e-8
step "serial vs np=4 hier, step 20 (tol 1e-5)" $CMP $A/serial $A/par --fields $F --tol 1e-5
# 3. two decompositions of the same case against each other (processorN vs processorM)
step "np=4 hier vs np=8 simple, step 20 (tol 1e-5)" $CMP $A/par $B/par --fields $F --tol 1e-5
# 4. the per-rank operator dumps, step 1, serial vs np=4: the p_rgh matrix diag/source of the sealed tank differ at the
#    reference cell by design (fvMatrix::setReference doubles the LOCAL diag, which lacks the cut faces), so the sealed
#    tank is checked without Diag/Source and the open case (no reference cell) with them
SKIP='UEqnDiag|IntCoeffs|BouCoeffs|^o?[Ww]eights|Sf\.dump|C\.dump'
step "operator dumps tank, serial vs np=4 hier, step 1 (tol 1e-6)" $CMP $A/serial_d1 $A/par_d1 --dumps --skip "$SKIP|oPrghEqn.*(Diag|Source)" --tol 1e-6
step "operator dumps open (incl. p_rgh Diag/Source/Upper + interface coefficients), step 1 (tol 1e-6)" $CMP $O/serial_d1 $O/par_d1 --dumps --skip "$SKIP" --tol 1e-6
[ $fail = 0 ] && echo "SELFTEST PASS" || { echo "SELFTEST FAIL"; exit 1; }
