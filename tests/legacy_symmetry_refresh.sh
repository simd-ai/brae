#!/usr/bin/env bash
# The boundary velocity the legacy driver's turbulence closure reads on a symmetry plane is the slip
# projection of the CURRENT cell velocity -- n.U_b == 0 -- at every iteration, as OpenFOAM's is.
#
# OpenFOAM's basicSymmetryFvPatchField::evaluate() sets U_b = U_c - n(n.U_c) from whatever U_c is at the
# moment of the call, and solve() and pEqn.H's U.correctBoundaryConditions() call it after the momentum
# solve and after the velocity correction. brae holds a symmetry patch as a MIXED refValue built from
# U_c (symUpdateKernel: vf_k = |n_k|, r_k = U_c,k - sign(n_k)(n.U_c)) and device_simple_foam.cu
# rebuilt it ONCE, at the top of the momentum predictor; only the wedge was refreshed after the solve
# and after the corrector. The pressure stage never reads the stale value (HbyA_b is projected afresh),
# but correctTurbulence runs after the corrector and hands that dbU_ to grad(U) and the closures. The
# blend of the old refValue with the new U_c then carries n.U_b = sum_k n_k(1-|n_k|) dU_k -- exactly zero
# on an axis-aligned plane, which is why no fixture ever saw it, and 0.25 dU_x + 0.12 dU_y on the 30-degree
# plane of validation/slipTurb. Queue item 13. The rho driver had the same defect (fixed earlier); its gate
# uses the wall FLUX as the oracle, which cannot see this one, so this gate uses the VALUE.
#
#   ORACLE   n.U_b == 0 on every symmetry face at closure time (OpenFOAM's evaluate, exactly).
#   ARM      the legacy driver on slipTurb (2,700 cells, kEpsilon, started from REST so dU is O(1) in
#            the first iterations), BRAE_SYM_CHECK=1 printing max|n.U_b| and max|U_b| each iteration.
#   CONTROLS the check must print once per iteration (a silent check passes on nothing), must report
#            symmetry faces, and the closure must actually have run (`Solving for k`).
#
# FAIL-PROOF, RUN before the two refreshes existed, slipTurb from rest, 8 iterations:
#     iteration   1        2        3        4        5        6        7        8
#     max|n.U_b|  1.01e-01 7.62e-02 6.03e-02 1.59e-02 3.12e-02 2.57e-02 8.98e-03 2.34e-02
#     max|U_b|    0.266    0.707    0.924    1.102    1.120    1.057    1.015    1.004
# i.e. 38% penetration at iteration 1 and still 2.3% at 8; the Ux residual trajectory differed from
# iteration 2 (0.5886 vs 0.6658 at iteration 4). With the refreshes: 3.6e-17 .. 2.2e-16, every line.
# The same fixture with its ORIGINAL uniform aligned inflow (its own exact solution) read only 4.7e-10
# pre-fix -- dU per iteration was 1e-9 -- which is why the case starts from rest.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/slipTurb}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
cp -r "$SRC" "$W/case"
N=$(grep -oE "^endTime\s+[0-9]+" "$W/case/system/controlDict" | grep -oE "[0-9]+$")
( cd "$W/case" && BRAE_SYM_CHECK=1 "$BRAE" "$W/case" > log 2>&1 ) || { tail -5 "$W/case/log"; say "the run finished" FAIL; exit 1; }
grep -q "Solving for k" "$W/case/log" && say "the closure ran (Solving for k)" ok || say "the closure ran (Solving for k)" FAIL
nLines=$(grep -c "symmetry closure check" "$W/case/log")
[ "$nLines" -eq "$N" ] && say "the check printed once per iteration ($nLines of $N)" ok \
                        || say "the check printed once per iteration ($nLines of $N)" FAIL
nSym=$(grep -m1 "symmetry closure check" "$W/case/log" | grep -oE "\([0-9]+ symmetry faces\)" | grep -oE "[0-9]+")
[ -n "${nSym:-}" ] && [ "$nSym" -gt 0 ] && say "the fixture has symmetry faces ($nSym)" ok \
                                          || say "the fixture has symmetry faces" FAIL
worst=$(grep "symmetry closure check" "$W/case/log" | python3 -c "
import sys, re
w = 0.0
for l in sys.stdin:
    m = re.search(r'max\|n\.U_b\| = ([0-9.e+-]+)\s+max\|U_b\| = ([0-9.e+-]+)', l)
    if not m: continue
    pen, u = float(m.group(1)), float(m.group(2))
    w = max(w, pen / u if u > 0 else pen)
print(f'{w:.3e}')")
printf '  worst max|n.U_b| / max|U_b| over %s iterations: %s\n' "$N" "$worst"
python3 -c "import sys; sys.exit(0 if $worst <= 1e-12 else 1)" \
    && say "the closure's boundary velocity does not penetrate the symmetry plane (<= 1e-12)" ok \
    || say "the closure's boundary velocity does not penetrate the symmetry plane (<= 1e-12)" FAIL
[ $fail -eq 0 ] && echo "PASS: the legacy driver hands its closure OpenFOAM's symmetry value at every iteration"
exit $fail
