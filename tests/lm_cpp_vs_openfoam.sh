#!/usr/bin/env bash
# fv kOmegaSSTLM (Langtry-Menter gamma-ReThetat) on the MIRRORED _cpp path, run END TO END against real
# OpenFOAM. No CUDA anywhere: this is the step that has to pass before any of it reaches the GPU.
#
# THE ORACLE IS REAL OPENFOAM: validation/T3A is the ERCOFTAC T3A flat-plate transition tutorial on its
# own blockMesh (26820 cells), run by simpleFoam v2412 to the case's own residualControl at t=269.
#
# WHY T3A AND NOT A PROBE. kOmegaSSTLM is a LOOP: gammaIntEff gates k's production, k and omega set RT
# and Rev, those set Fonset, Fonset drives gammaInt, and gammaInt becomes the next iteration's
# gammaIntEff. A single-iteration residual never turns that loop over once. Worse, the loop is LAGGED --
# OpenFOAM runs kOmegaSST::correct() BEFORE correctReThetatGammaInt(), so k and omega always advance on
# the previous iteration's transition state, and gammaIntEff starts at ZERO, so the very first iteration
# has no turbulent production at all. Only a full run exercises any of that.
#
# THE CONTROL IS THE POINT. Running the same case with plain kOmegaSST -- identical solver, identical
# schemes, transition model removed -- must be far worse against OpenFOAM's kOmegaSSTLM answer. A
# transition model that made no difference would pass any bound placed on the LM run alone. Measured:
# 167x on U, 437x on k, 114x on nut.
#
# WHAT THIS GATE ALSO PINS, because both were found by writing it:
#   * The turbulence scalars' `linearUpwind`. T3A asks for `bounded Gauss linearUpwind grad` on k, omega,
#     gammaInt and ReThetat. The _cpp kOmegaSST silently ran UPWIND -- it had no linearUpwind parameter
#     at all -- and honouring it moved U from 2.4e-03 to 2.0e-04, a factor of 12.
#   * The case's own LINEAR-SOLVER settings. T3A asks for `relTol 0.1` on U and p; solving each outer
#     iteration to 1e-10 instead is a different iteration, not a stricter one, and on this mesh it walks
#     away -- U grows from 1.5e-03 at iteration 20 to 3.0e-01 at 400 while OpenFOAM falls monotonically.
set -u
SRC="${1:?case dir}"
REF="${2:-269}"
ITERS="${3:-500}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${LM_CPP_BIN:-$ROOT/build/test_simple_lm_cpp}"
[ -x "$BIN" ] || { echo "SKIP: no test_simple_lm_cpp at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/$REF" ] || { echo "SKIP: no OpenFOAM reference at $SRC/$REF"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$SRC/$REF" "$W/"
cp -r "$W/0.orig" "$W/0"
# The reference time needs ReThetat and gammaInt present for the reader; the run starts from 0/ and never
# reads them from there, but the harness opens both time directories the same way.
for f in ReThetat gammaInt; do
    [ -f "$W/$REF/$f" ] || [ -f "$W/$REF/$f.gz" ] || cp "$W/0/$f" "$W/$REF/$f"
done

run() {   # run <label> <extra env>
    local out
    out=$(env LM_SHOW=0 $2 "$BIN" "$W" 0 "$REF" "$ITERS" 2>&1) || { echo "FAIL: $1 crashed"; echo "$out" | tail -5; return 1; }
    echo "$out"
}

LM=$(run "the LM run" "") || exit 1
SST=$(run "the control" "LM_PLAIN_SST=1") || exit 1
echo "$LM" | grep -E 'it +[0-9]+ +gammaInt|it +[0-9]+ +U' | tail -2 | sed 's/^/ /'

python3 - <<PY
import re, sys
lm  = """$LM"""
sst = """$SST"""

def fields(txt):
    m = re.search(r'vs OpenFOAM \S+ \(L2 rel\):\s+U ([0-9.e+-]+)\s+p ([0-9.e+-]+)\s+k ([0-9.e+-]+)'
                  r'\s+omega ([0-9.e+-]+)\s+nut ([0-9.e+-]+)', txt)
    if not m: return None
    return dict(zip(('U', 'p', 'k', 'omega', 'nut'), (float(x) for x in m.groups())))

a, b = fields(lm), fields(sst)
if a is None: print("  FAIL: the LM run printed no comparison"); sys.exit(1)
if b is None: print("  FAIL: the control printed no comparison"); sys.exit(1)

# The run must also have SETTLED. A run that is merely passing through the right answer on its way
# somewhere else would satisfy the bounds below on the right iteration.
mres = re.findall(r'it +[0-9]+ +U ([0-9.e+-]+) +p ([0-9.e+-]+)', lm)
rc = 0
if mres:
    U, P = float(mres[-1][0]), float(mres[-1][1])
    ok = (U < 1e-05) and (P < 1e-04)
    print("  settled  final residuals: U %.3e  p %.3e   %s" % (U, P, "ok" if ok else "FAIL (not converged)"))
    if not ok: rc = 1

# Set just above where the _cpp reference lands. k, omega and nut are the transported/derived turbulence
# quantities and would normally sit an order looser than U -- here they do not, which is the sign that
# the transition model is being reproduced and not merely approximated.
BOUND = {'U': 5e-04, 'p': 3e-03, 'k': 4e-03, 'omega': 2e-04, 'nut': 4e-03}
for f in ('U', 'p', 'k', 'omega', 'nut'):
    ok = a[f] < BOUND[f]
    print("  %-6s LM %.3e   bound %.1e   %s" % (f, a[f], BOUND[f], "ok" if ok else "FAIL"))
    if not ok: rc = 1

# THE CONTROL. Plain kOmegaSST against the SAME OpenFOAM kOmegaSSTLM answer.
print("  control  plain kOmegaSST against the same reference:")
for f, want in (('U', 20.0), ('k', 20.0), ('nut', 20.0)):
    ratio = b[f] / max(a[f], 1e-30)
    ok = ratio >= want
    print("    %-6s SST %.3e   %6.1fx the LM error   %s" % (f, b[f], ratio, "ok" if ok else "FAIL (want >=%.0fx)" % want))
    if not ok: rc = 1

if rc == 0:
    print("  ok:   the _cpp kOmegaSSTLM runs T3A end to end and matches real OpenFOAM")
sys.exit(rc)
PY
