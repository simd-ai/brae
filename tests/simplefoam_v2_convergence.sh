#!/usr/bin/env bash
# CONVERGENCE GATE for the rebuilt simpleFoam: run to convergence from the initial fields and compare the
# converged solution with OpenFOAM's own.
#
# Every other gate on this path is per-iteration or per-stage. This is the only one that asks the question
# a user asks -- does it get the right answer -- and it is the criterion the rebuild has been missing:
# the _cpp reference was validated for ONE iteration against dumpSimpleStep and never for convergence.
#
# THE ORACLE IS REAL OPENFOAM. validation/pitzDailyTurb/1576 is simpleFoam v2412 output
# (log.simpleFoam: "Build : _e5c6ccc3-20250814 OPENFOAM=2412", "SIMPLE solution converged in 1576
# iterations"), not a brae result.
#
# THE BOUNDS ARE THE EXISTING SOLVER'S. On this case brae's established path reaches U 1.311e-01,
# p 1.989e-01, k 1.183e-02, eps 2.684e-02, nut 6.907e-03 against the same reference and that is the
# passing `simple_turbulent_full` gate -- the disagreement is localised (6 of 12225 cells above 0.5 m/s,
# worst at the step corner). So the bar here is "no worse than the solver we already ship", not a number
# chosen to fit. Tightening it would be asserting something untrue about the case.
#
# usage: simplefoam_v2_convergence.sh <pitzDailyTurbCase> <ofTimeDir>
set -u
SRC="${1:?case dir}"; OFT="${2:?OpenFOAM time dir}"
BRAE="${BRAE_BIN:-$(cd "$(dirname "$0")/.." && pwd)/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/$OFT" ] || { echo "SKIP: no OpenFOAM reference at $SRC/$OFT"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case"
find "$W/case" -mindepth 1 -maxdepth 1 -type d ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null

( cd "$W/case" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 )
rc=$?
if [ $rc -ne 0 ]; then echo "FAIL: solver exited $rc"; sed -n '1,6p' "$W/case/run.log"; exit 1; fi
grep -q "converged" "$W/case/run.log" || { echo "FAIL: never reported convergence"; tail -2 "$W/case/run.log"; exit 1; }
echo "  ok:   reached convergence -- $(grep converged "$W/case/run.log" | head -1)"

python3 - "$W/case" "$SRC/$OFT" <<'PY'
import re, sys, os, glob
brdir, ofdir = sys.argv[1], sys.argv[2]
times = sorted((int(os.path.basename(d)) for d in glob.glob(brdir + '/[0-9]*')
                if os.path.basename(d).isdigit()), reverse=True)
if not times:
    print("FAIL: the run wrote no time directory"); sys.exit(1)
last = str(times[0])

def vals(p, n=None):
    """Field values, handling BOTH forms. A `uniform` initial field silently returned None here and the
    control below vanished without a word -- a control that can disappear is not a control."""
    try: s = open(p).read()
    except OSError: return None
    m = re.search(r'internalField\s+nonuniform\s+List<\w+>\s*\n?\d+\s*\(\s*(.*?)\n\)\s*;', s, re.S)
    if m: return [float(x) for x in re.findall(r'[-+0-9.eE]+', m.group(1))]
    m = re.search(r'internalField\s+uniform\s+([^;]+);', s)
    if m and n:
        comp = [float(x) for x in re.findall(r'[-+0-9.eE]+', m.group(1))]
        return comp * (n // len(comp)) if comp else None
    return None

# The established solver's own agreement with this reference, from the passing simple_turbulent_full gate.
BOUND = {'U': 1.45e-01, 'p': 2.20e-01, 'k': 1.40e-02, 'epsilon': 3.00e-02, 'nut': 8.00e-03}
fails, checked = 0, 0
for f, tol in BOUND.items():
    a = vals(os.path.join(brdir, last, f)); b = vals(os.path.join(ofdir, f))
    if a is None or b is None or len(a) != len(b):
        print("  FAIL: %s missing or size mismatch" % f); fails += 1; continue
    checked += 1
    mx = max(abs(x - y) for x, y in zip(a, b))
    mg = max(abs(x) for x in b) or 1.0
    rel = mx / mg
    ok = rel <= tol
    if not ok: fails += 1
    print("  %-8s max-rel=%.4e  (bound %.2e)  %s" % (f, rel, tol, "ok" if ok else "FAIL"))

if checked < len(BOUND):
    print("FAIL: only %d of %d fields compared" % (checked, len(BOUND))); sys.exit(1)

# CONTROL: the reference must not be trivially reproduced. Compare OpenFOAM's converged U against the
# INITIAL field -- if that also came in under the bound, the fields never moved and every line above is
# meaningless.
b = vals(os.path.join(ofdir, 'U'))
a0 = vals(os.path.join(brdir, '0', 'U'), len(b) if b else None)
if not a0 or not b or len(a0) != len(b):
    print("  FAIL: could not read the initial field, so the control never ran"); fails += 1
else:
    mx = max(abs(x - y) for x, y in zip(a0, b)); mg = max(abs(x) for x in b) or 1.0
    print("  %-8s max-rel=%.4e  (initial field vs OpenFOAM -- must EXCEED the bound)" % ('control', mx/mg))
    if mx/mg <= BOUND['U']:
        print("  FAIL: the initial field already satisfies the bound (control)"); fails += 1

print("PASS" if not fails else "FAIL"); sys.exit(1 if fails else 0)
PY
