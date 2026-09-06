#!/bin/bash
# Incompressible fixedFluxPressure through the V2 driver vs real OpenFOAM, on validation/simpleBoxP.
#
# This is the gate that let the V2 envelope's substring refusal be lifted, and the end-to-end oracle
# for simpleFoam's OWN constrainPressure position (pEqn.H:21 -- AFTER adjustPhi and SIMPLEC, divisor
# rAtU; the compressible gate cannot see that ordering). Converged-vs-converged, each side on its own
# residualControl.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/simpleBoxP" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_ffpi_vs_of}
if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
rm -rf "$WORK" "$WORK.brae"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
( cd "$WORK" && simpleFoam > log.simpleFoam 2>&1 )
OFLAST=$(cd "$WORK" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
[ -n "$OFLAST" ] || { echo "FAIL: OpenFOAM produced no output"; exit 1; }

cp -r "$WORK" "$WORK.brae"; ( cd "$WORK.brae" && rm -rf [1-9]* log.simpleFoam )
( cd "$WORK.brae" && BRAE_SIMPLEFOAM_V2=1 "$BUILD/brae" -case . > log.brae 2>&1 ) \
    || { tail -20 "$WORK.brae/log.brae"; echo "FAIL: the V2 driver refused or died"; exit 1; }
BRLAST=$(cd "$WORK.brae" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
[ -n "$BRLAST" ] || { echo "FAIL: brae produced no output"; exit 1; }

# LEGACY ARM: the DeviceSimpleSolver drivers never run constrainPressure, so they must REFUSE this
# case by name rather than keep the construction-time gradient (zeroGradient under ffp's name). The
# refusal fires in the solver constructor, before any iteration.
lout=$(cd "$WORK.brae" && "$BUILD/brae" -case . 2>&1) && { echo "FAIL(legacy): ran with an unmaintained fixedFluxPressure"; exit 1; }
echo "$lout" | grep -q "zeroGradient under fixedFluxPressure" \
    && echo "PASS(legacy-refused)" \
    || { echo "$lout" | tail -4; echo "FAIL(legacy): refusal does not name the substitution"; exit 1; }

python3 - "$WORK/$OFLAST" "$WORK.brae/$BRLAST" <<'PYEOF'
import re, math, sys
of, br = sys.argv[1], sys.argv[2]
def internal(path, vec=False):
    t = open(path).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\(\n(.*?)\n\)\s*;', t, re.S)
    rows = m.group(1).strip().split('\n')
    if vec: return [tuple(float(x) for x in r.strip('()').split()) for r in rows]
    return [float(r) for r in rows]
def rel(a, b):
    num = den = 0.0
    for x, y in zip(a, b):
        if isinstance(x, tuple):
            for i in range(3): num += (x[i]-y[i])**2; den += y[i]**2
        else: num += (x-y)**2; den += y*y
    return math.sqrt(num/den) if den > 0 else math.sqrt(num)
fails = 0
def check(what, val, bound):
    global fails
    ok = val < bound
    print(f"  {what:44s} {val:.6e}   {'ok' if ok else 'FAIL'} (bound {bound:g})")
    if not ok: fails += 1
# Bounds 10x the measured converged-vs-converged gap (U 3.3e-09, p 4.9e-05 -- p is a small-norm
# kinematic field at the two runs' residual floors). The zeroGradient substitution is the thing to
# catch; its measured distance is recorded in the fail-proof section of PORT.md.
check("U vs OpenFOAM (converged)", rel(internal(br+"/U", True), internal(of+"/U", True)), 5e-8)
check("p vs OpenFOAM (converged)", rel(internal(br+"/p"), internal(of+"/p")), 5e-4)
# NON-VACUITY: the vent's converged gradient must be real, or zeroGradient passes this whole gate.
m = re.search(r'coldWall\s*\{[^}]*?gradient\s+nonuniform[^(]*\(\n(.*?)\n\)', open(of+"/p").read(), re.S)
g = [abs(float(r)) for r in m.group(1).strip().split('\n')] if m else []
gmax = max(g) if g else 0.0
print(f"  {'OF vent gradient max|g| (non-vacuity)':44s} {gmax:.6e}   {'ok' if gmax > 0.01 else 'FAIL'}")
if gmax <= 0.01: fails += 1
print("PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(0 if fails == 0 else 1)
PYEOF
