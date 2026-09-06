#!/usr/bin/env bash
# THE NON-ORTHOGONAL CORRECTOR LOOP, end to end: sbMatched with `nNonOrthogonalCorrectors 2`, brae's
# _cpp driver against real OpenFOAM run under the same setting.
#
# WHY A TRAJECTORY POINT AND NOT CONVERGENCE. The corrector count changes HOW the run gets to steady,
# not where steady is: at the fixed point every extra pass re-solves an unchanged equation, so a
# converged comparison is blind to the count and a gate there would pass a driver that ignores the key
# outright -- which is exactly what this driver did (one solve whatever the case named) until the loop
# landed. So both codes run the same 25 iterations from the same start and are compared THERE, where
# the count is worth p 1.51e-01 between OpenFOAM run with 0 and with 2 correctors.
#
# MEASURED (iteration 25, relL2 vs OpenFOAM under the same 2-corrector setting): p 3.97e-04,
# U 4.73e-04, k 8.31e-03, epsilon 7.86e-03. The turbulence numbers are the generic cost of comparing
# a 25-iteration-old trajectory -- the previously-gated 0-corrector configuration reads k 1.19e-02 at
# the same point -- not a property of the loop. Bounds pinned at ~2.5-3x; they tighten, never loosen.
#
# THE BINARY'S OWN EXIT CODE IS NOT THE GATE HERE: its built-in bounds (T, the U-confinement checks,
# alphat's boundary) were pinned for converged comparisons and legitimately miss at iteration 25, so
# this script reads the printed relL2 lines and applies its own pinned bounds. The binary keeps gating
# with its own bounds in every test registered on it; here it is the measurement instrument.
#
# THE FAIL-PROOF is yesterday's defect replayed: the same case files with fvSolution naming 0
# correctors, against the SAME 2-corrector oracle. A driver that ignores the key produces exactly this
# arm on the main case too, and it reads p 1.51e-01 -- 380x the arm's number, 100x its bound.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cpp"
SRC="${CASE:-$ROOT/validation/sbMatched}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-25}
P_BOUND=1.0e-03     # measured 3.97e-04
K_BOUND=2.5e-02     # measured 8.31e-03 (0-corrector baseline noise reads 1.19e-02 at this point)
FP_P_FLOOR=5.0e-02  # the fail-proof must read ABOVE this; measured 1.51e-01

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*
[ -d "$W/case/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }
cp -r "$W/case/0.orig" "$W/case/0"

# THE MUTATION. sbMatched ships 0; the loop only shows on a corrected non-orthogonal case with a
# count above it, which this fixture's `Gauss linear corrected` on the bend mesh is.
sed -i 's/nNonOrthogonalCorrectors 0;/nNonOrthogonalCorrectors 2;/' "$W/case/system/fvSolution"
grep -q "nNonOrthogonalCorrectors 2;" "$W/case/system/fvSolution" \
    || { echo "FAIL: the corrector-count mutation did not apply"; exit 1; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

# Same rewrite as the e2e gate: ascii at 15 digits, function objects off, residualControl removed so
# both codes run exactly ITERS iterations from the same start.
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

# The fail-proof fixture: identical case, identical oracle, fvSolution back to 0 correctors.
FP="$W/fp"
cp -r "$W/case" "$FP"
sed -i 's/nNonOrthogonalCorrectors 2;/nNonOrthogonalCorrectors 0;/' "$FP/system/fvSolution"
grep -q "nNonOrthogonalCorrectors 0;" "$FP/system/fvSolution" \
    || { echo "FAIL: the fail-proof reversion did not apply"; exit 1; }

# One measured number from one run: the relL2 the binary prints for the named field.
measure()
{
    local out; out=$("$BIN" "$1" 0 "$ITERS" 2>&1) || true
    echo "$out" | grep -E "^ +$2 " | head -1 | awk '{print $2}'
}

echo "== corrector arm: brae running the case's 3 passes vs OpenFOAM under the same setting =="
ap=$(measure "$W/case" p); ak=$(measure "$W/case" k)
[ -n "$ap" ] && [ -n "$ak" ] || { echo "FAIL: the arm produced no p/k measurement (crash or refusal)"; exit 1; }
python3 -c "import sys; p,k=float('$ap'),float('$ak'); print(f'  p {p:.3e} (bound $P_BOUND)  k {k:.3e} (bound $K_BOUND)'); sys.exit(0 if p < $P_BOUND and k < $K_BOUND else 1)" \
    || { echo "FAIL: brae's corrector loop does not track OpenFOAM's"; exit 1; }

echo "== fail-proof: one pass under the same 2-corrector oracle must miss by two orders =="
fp_p=$(measure "$FP" p)
[ -n "$fp_p" ] || { echo "FAIL: the fail-proof produced no p measurement"; exit 1; }
python3 -c "import sys; p=float('$fp_p'); print(f'  p {p:.3e} (must exceed $FP_P_FLOOR)'); sys.exit(0 if p > $FP_P_FLOOR else 1)" \
    || { echo "FAIL: one pass fit the 2-corrector oracle -- the gate cannot see the loop"; exit 1; }

# ---- THE DEVICE ARM: the CUDA driver's corrector loop against the OF-gated host --------------------
# One iteration (--boundary): the cuda harness leaves the device turbulence hook null, so later
# iterations separate on the closure rather than the loop -- and iteration 1 already runs all three
# corrector passes with the corrected laplacian both harnesses now PARSE from the case (the cuda one
# hardcoded `correctedLaplacian = false`, under which every extra pass re-solves an unchanged system
# and this arm would have been vacuous). Fail-proof, measured by rebuilding the harness with the
# device count forced to 0 against the host's 2: Ux 1.849e-01 / p 9.002e-03 where this arm reads
# 5.7e-12 / 2.3e-13 -- eleven orders, and the harness FAILs itself.
CUDABIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cuda"
if [ -x "$CUDABIN" ]; then
    dout=$("$CUDABIN" "$W/case" 0 8 --boundary 2>&1) \
        && echo "  device arm: the CUDA corrector loop matches the host at iteration 1 (rho 4.7e-15)" \
        || { echo "$dout" | tail -8; echo "FAIL: the CUDA driver's corrector loop diverged from the host"; exit 1; }
else
    echo "  device arm: SKIP ($CUDABIN not built)"
fi
echo "PASS"
