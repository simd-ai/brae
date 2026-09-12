#!/usr/bin/env bash
# kOmegaSSTLM through the CUDA V2 driver, end to end on T3A, against real OpenFOAM.
#
# The companion to tests/lm_cpp_vs_openfoam.sh: the same case, the same oracle, the CUDA path. The order
# is the point -- the _cpp reference ran this case first, then the CUDA modules were compared against it
# one at a time (tests/lm_cuda_probe.cu, which holds Fthetat, the two reaction preps and gammaIntEff to
# 1e-12 and the two solved transports to 1e-06), and only then was the driver wired. A CUDA path that
# agrees with OpenFOAM end to end but not with the reference stage by stage is agreeing by cancellation.
#
# THE ORACLE IS REAL OPENFOAM: validation/T3A run by simpleFoam v2412 to the case's own residualControl.
#
# THE CONTROL: the same case with RASModel switched to plain kOmegaSST -- identical driver, identical
# schemes, transition model removed -- must be far worse against OpenFOAM's kOmegaSSTLM answer.
#
# WHAT THIS ALSO PINS, found by wiring it: the V2 driver hardcoded `linearUpwindK`, `linearUpwindOmega`
# and the laplacian `nonOrth` flag to FALSE for the turbulence equations while its own setup line printed
# what the case had asked for. So a case naming `bounded Gauss linearUpwind grad` and
# `Gauss linear corrected` got upwind and orthogonal, silently, on every turbulence scalar.
set -u
SRC="${1:?case dir}"
REF="${2:-269}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
PROBE="${LM_CUDA_PROBE:-$ROOT/build/lm_cuda_probe}"
[ -x "$BRAE" ]  || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -x "$PROBE" ] || { echo "SKIP: no lm_cuda_probe at $PROBE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/$REF" ] || { echo "SKIP: no OpenFOAM reference at $SRC/$REF"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }

# 1. THE MODULES, against the _cpp reference, before anything is run end to end.
"$PROBE" "$SRC" "$REF" | sed 's/^/  /' || { echo "FAIL: a CUDA module disagrees with the _cpp reference"; exit 1; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
prep() {   # prep <dir> <lm|sst>
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$1/"
    cp -r "$1/0.orig" "$1/0"
    python3 - "$1" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
open(c, 'w').write(re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S))
PY
    if [ "$2" = "sst" ]; then
        for f in "$1"/constant/turbulenceProperties "$1"/constant/momentumTransport; do
            [ -f "$f" ] && sed -i 's/RASModel *kOmegaSSTLM;/RASModel        kOmegaSST;/' "$f"
        done
    fi
}
prep "$W/lm"  lm
prep "$W/sst" sst
BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/lm"  > "$W/lm.log"  2>&1 || { echo "FAIL: brae crashed on the LM case"; tail -12 "$W/lm.log"; exit 1; }
BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/sst" > "$W/sst.log" 2>&1 || { echo "FAIL: brae crashed on the control"; tail -12 "$W/sst.log"; exit 1; }
grep -q 'kOmegaSSTLM' "$W/lm.log"  || { echo "FAIL: brae did not report running kOmegaSSTLM"; exit 1; }
grep -q 'kOmegaSSTLM' "$W/sst.log" && { echo "FAIL: the control reported kOmegaSSTLM"; exit 1; }
# The case asks for these; the driver used to print them and then not apply them.
grep -q 'div(phi,k): bounded linearUpwind' "$W/lm.log" || { echo "FAIL: brae did not read div(phi,k) as bounded linearUpwind"; exit 1; }
grep -q 'non-orthogonal correction ON' "$W/lm.log"     || { echo "FAIL: brae did not read the laplacian as corrected"; exit 1; }

python3 - "$W/lm" "$W/sst" "$SRC/$REF" <<'PY'
import gzip, os, re, sys
import numpy as np

def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r'[0-9]+(\.[0-9]+)?', x) and x != '0']
    return None if not ts else max(ts, key=float)

def read(d, f):
    for c in (os.path.join(d, f), os.path.join(d, f + '.gz')):
        if os.path.exists(c):
            b = (gzip.open(c, 'rb') if c.endswith('.gz') else open(c, 'rb')).read()
            break
    else:
        return None
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m: return None
    typ = m.group(1).decode(); n = int(m.group(2)); start = m.end()
    nc = 3 if typ == 'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary':
        return np.frombuffer(b[start:start+n*nc*8], dtype='<f8').reshape(n, nc)
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

lm, sst, ref = sys.argv[1], sys.argv[2], sys.argv[3]
for d in (lm, sst):
    t = lastTime(d)
    if t is None: print("  FAIL: %s wrote no result" % d); sys.exit(1)
for name, d in (('lm', lm), ('sst', sst)):
    globals()[name] = os.path.join(d, lastTime(d))

def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

# Set just above where the CUDA driver lands, which is TIGHTER than the _cpp reference it was ported
# from (U 7.9e-05 against 2.0e-04) -- the driver reads the case's linear-solver settings while the
# reference harness is asked to run a fixed iteration count.
BOUND = {'U': 3e-04, 'p': 3e-04, 'k': 4e-03, 'omega': 2e-05, 'nut': 4e-03}
rc, err = 0, {}
for f in ('U', 'p', 'k', 'omega', 'nut'):
    a, b = read(lm, f), read(ref, f)
    if a is None or b is None: print("  FAIL: could not read %s" % f); rc = 1; continue
    e = rel(a, b); err[f] = e
    ok = e < BOUND[f]
    print("  %-6s CUDA %.3e   bound %.1e   %s" % (f, e, BOUND[f], "ok" if ok else "FAIL"))
    if not ok: rc = 1

print("  control  plain kOmegaSST through the same driver, same reference:")
for f, want in (('U', 20.0), ('k', 20.0), ('nut', 20.0)):
    a, b = read(sst, f), read(ref, f)
    e = rel(a, b)
    ratio = e / max(err.get(f, 1e-30), 1e-30)
    ok = ratio >= want
    print("    %-6s SST %.3e   %6.1fx the LM error   %s" % (f, e, ratio, "ok" if ok else "FAIL (want >=%.0fx)" % want))
    if not ok: rc = 1

if rc == 0:
    print("  ok:   the CUDA kOmegaSSTLM runs T3A end to end and matches real OpenFOAM")
sys.exit(rc)
PY
