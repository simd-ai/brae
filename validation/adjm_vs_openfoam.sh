#!/bin/bash
# adjustPhi's inletOutlet classification, both arms vs real OpenFOAM -- see validation/simpleBoxIO/README.md.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/simpleBoxIO" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_adjm_vs_of}
if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
rm -rf "$WORK" "$WORK.brae"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
( cd "$WORK" && simpleFoam > log.simpleFoam 2>&1 )
OFLAST=$(cd "$WORK" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
[ -n "$OFLAST" ] || { echo "FAIL: OpenFOAM produced no output"; exit 1; }

# HOST ARM: the mirror loop, with the in-binary continuity check and engagement guards.
"$BUILD/test_simple_adjustphi_cpp" "$WORK" "$OFLAST" 200 || { echo "FAIL(host)"; exit 1; }
echo "PASS(host)"

# V2 ARM: the device mask. With the inletOutlet half missing this REFUSED the case outright
# (deviceAdjustPhi: "adjustable mass outflow 0.000000"), so the run itself is the discriminator.
cp -r "$WORK" "$WORK.brae"; ( cd "$WORK.brae" && rm -rf [1-9]* log.simpleFoam )
( cd "$WORK.brae" && BRAE_SIMPLEFOAM_V2=1 "$BUILD/brae" -case . > log.brae 2>&1 ) \
    || { tail -6 "$WORK.brae/log.brae"; echo "FAIL(v2): refused or died"; exit 1; }
BRLAST=$(cd "$WORK.brae" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
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
# measured U 4.6e-11 / p 1.4e-08 converged-vs-converged; bounds 10x
dU = rel(internal(br+"/U", True), internal(of+"/U", True))
dP = rel(internal(br+"/p"), internal(of+"/p"))
print(f"  V2 vs OF: U {dU:.3e}  p {dP:.3e}")
ok = dU < 5e-10 and dP < 2e-7
print("PASS(v2)" if ok else "FAIL(v2)")
sys.exit(0 if ok else 1)
PYEOF
echo PASS
