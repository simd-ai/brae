#!/usr/bin/env bash
# div(phi,k)/div(phi,epsilon) `Gauss limitedLinear 1` END TO END: brae's _cpp driver against real
# OpenFOAM, both codes running validation/sbMatched MUTATED to the scheme -- the same mutation applied
# to both, so the oracle converges under the scheme being gated, not under the fixture's upwind.
# `Gauss limitedLinear 1` (unbounded) is the exact entry validation/squareBend ships on both scalars.
#
# WHY A SEPARATE SCRIPT: rho_simple_end_to_end_vs_openfoam.sh already pays one 400-iteration OpenFOAM
# run; this arm needs a second one (the oracle must be run UNDER the mutation), and carrying both in
# one script doubles the gate everyone waits on. The e2e script keeps the cheap refusal arms for what
# still refuses (linearUpwind, disagreeing k/epsilon entries); this one gates what now assembles.
#
# THE CONTROL, and why no upwind-oracle run is needed for non-vacuity: the fail-proof arm reverts ONLY
# the scheme word (limitedLinear 1 -> upwind, both unbounded) and re-runs brae against the SAME
# limitedLinear oracle. brae-upwind agrees with OpenFOAM-upwind to 5e-05 on this fixture (the e2e
# gate), so brae-upwind failing the bound here is the statement that the two schemes' converged answers
# sit further apart than the bound -- the gate can see the scheme, and a closure that silently
# assembled upwind under the case's limitedLinear would fail the main arm the same way. Measured: the
# substitution moves k by 8.99e-02 and epsilon by 7.29e-02, and it is not confined to turbulence -- U
# reads 5.86e-03 and fails its own built-in bound too.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cpp"
SRC="${CASE:-$ROOT/validation/sbMatched}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-400}
# Measured on this arm's first run: k 5.648e-04, epsilon 1.894e-03, nut 6.910e-04, alphat 7.325e-04
# (relL2 vs OpenFOAM at 400/400 iterations; U 1.73e-05, p 6.03e-06). Wider than the upwind fixture's
# 3.46e-06/1.37e-04 because this mutation does not converge as deep: OpenFOAM's OWN k residual plateaus
# at 1.75e-04 from ~iteration 300 (5.2e-03 at 100, 1.4e-03 at 200, 1.77e-04 at 350) -- both codes sit
# on the same plateau, so 400 is an answer-level comparison for this scheme, and running longer buys
# nothing. Pinned at ~2x the worst (epsilon); the fail-proof arm reads k 8.99e-02 / epsilon 7.29e-02
# against it -- a 22x/18x margin. Tightens as the closure improves; never loosens.
LLBOUND=${LLBOUND:-4e-03}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*
[ -d "$W/case/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }
cp -r "$W/case/0.orig" "$W/case/0"

# THE MUTATION, applied before either code runs. The grep is load-bearing: a sed that silently missed
# would leave both codes on upwind and this whole gate vacuously green.
sed -i 's/turbulence          bounded Gauss upwind;/turbulence          Gauss limitedLinear 1;/' "$W/case/system/fvSchemes"
grep -q "Gauss limitedLinear 1;" "$W/case/system/fvSchemes" \
    || { echo "FAIL: the limitedLinear mutation did not apply"; exit 1; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

# Same controlDict/fvSolution rewrite as the e2e script: ascii at 15 digits, function objects off,
# residualControl removed so BOTH codes run exactly ITERS iterations from the same start.
ITERS="$ITERS" python3 - "$W/case" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
n = os.environ['ITERS']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', n),
             ('writeInterval', n), ('writeControl', 'timeStep')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution')
s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PYEOF

( cd "$W/case" && rhoSimpleFoam > run.log 2>&1 ) \
    || { echo "FAIL: OpenFOAM's rhoSimpleFoam did not run"; tail -25 "$W/case/run.log"; exit 1; }
[ -d "$W/case/$ITERS" ] \
    || { echo "FAIL: OpenFOAM wrote no $ITERS/"; tail -25 "$W/case/run.log"; exit 1; }
grep -q "^End" "$W/case/run.log" \
    || { echo "FAIL: OpenFOAM did not finish"; tail -25 "$W/case/run.log"; exit 1; }

# THE FAIL-PROOF FIXTURE is built from the mutated case BEFORE the main arm touches it: only the scheme
# word reverts (limitedLinear 1 -> upwind, both unbounded), the limitedLinear oracle in $ITERS/ stays.
FP="$W/fp"
cp -r "$W/case" "$FP"
sed -i 's/turbulence          Gauss limitedLinear 1;/turbulence          Gauss upwind;/' "$FP/system/fvSchemes"
grep -q "limitedLinear" "$FP/system/fvSchemes" \
    && { echo "FAIL: the fail-proof reversion did not apply"; exit 1; }

echo "== limitedLinear arm: brae assembling Gauss limitedLinear 1 vs OpenFOAM under the same scheme =="
BRAE_TURB_BOUND="$LLBOUND" "$BIN" "$W/case" 0 "$ITERS" \
    || { echo "FAIL: brae's limitedLinear does not match OpenFOAM's"; exit 1; }

echo "== fail-proof: upwind assembly under the SAME limitedLinear oracle must miss the bound =="
fout=$(BRAE_TURB_BOUND="$LLBOUND" "$BIN" "$FP" 0 "$ITERS" 2>&1) \
    && { echo "FAIL: upwind under the limitedLinear oracle fit the bound -- the gate cannot see the scheme"; exit 1; }
echo "$fout" | grep -Eq "^ *(k|epsilon) .*FAIL" \
    || { echo "$fout" | tail -15; echo "FAIL: the fail-proof run failed for some other reason than the k/epsilon bound"; exit 1; }
echo "  fail-proof: upwind misses the k/epsilon bound against the limitedLinear oracle  ok"
echo "PASS"
