#!/usr/bin/env bash
# linearUpwind vs REAL OPENFOAM, on pitzDaily's own discretisation.
#
# `bounded Gauss linearUpwind grad(U)` is what stock pitzDaily specifies. linearUpwind derives from
# `upwind`, so the MATRIX is identical to plain upwind and the entire scheme is a deferred source
# correction -- which means a port that dropped it still runs, still converges, and still looks
# reasonable. Only a comparison against OpenFOAM can see it, and only if the case is one where the
# scheme actually changes the answer. Hence the control below.
#
# WHY THE CASE IS MODIFIED. Stock pitzDaily sets `consistent yes` (SIMPLEC), which is not ported. This
# gate therefore runs plain SIMPLE with pressure under-relaxation -- SIMPLEC does not relax pressure, so
# the stock relaxation factors diverge under SIMPLE, in OpenFOAM as well as here. Everything the gate is
# about is untouched: the mesh, the turbulence model, and every discretisation scheme.
#
# WHY residualControl IS TIGHTENED. Stock is `p 1e-2`, at which both solvers stop early at visibly
# different states and the comparison measures where each happened to stop, not the discretisation. At
# 1e-6 both settle: OpenFOAM's answer moves by 4e-05 (L2, U) between iteration 2000 and 20000, two orders
# below the effect being measured. OpenFOAM's own p residual plateaus near 2.7e-03 and never reaches
# 1e-6 -- normFactor collapses once the field stops moving -- so it runs to endTime by design.
#
# THE MEASURE IS L2, NOT MAX. pitzDaily's max-norm error sits in a handful of cells at the step corner
# and is ~9.3e-02 whatever the convection scheme is; it cannot see this term. The L2 norm can.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$BRAE" ]                 || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

# One case definition, used by all three runs -- the oracle and both brae runs must differ ONLY in the
# scheme under test. Building them from separate edits is how a harness ends up measuring itself.
mkcase() {
    rm -rf "$1"; cp -r "$SRC" "$1"
    find "$1" -mindepth 1 -maxdepth 1 -type d ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null
    rm -f "$1"/log.* 2>/dev/null
    python3 - "$1" <<'PY'
import re, sys
d = sys.argv[1]
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'consistent\s+\S+;', 'consistent      no;', s)
s = re.sub(r'relaxationFactors\s*\{.*?\n\}',
           'relaxationFactors\n{\n    fields\n    {\n        p               0.3;\n    }\n'
           '    equations\n    {\n        U               0.7;\n        ".*"            0.7;\n    }\n}',
           s, flags=re.S)
s = re.sub(r'residualControl\s*\{.*?\n    \}',
           'residualControl\n    {\n        p               1e-6;\n        U               1e-6;\n'
           '        "(k|epsilon|omega|f|v2)" 1e-6;\n    }', s, flags=re.S)
open(p, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;',       'endTime         2000;', s)
s = re.sub(r'writeInterval\s+\S+;', 'writeInterval   2000;', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)   # FOs are not what is measured
open(c, 'w').write(s)
PY
}

mkcase "$W/of"
export WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2412
export FOAM_ETC="$WM_PROJECT_DIR/etc"
export PATH="$OFBIN/bin:$PATH"
export LD_LIBRARY_PATH="$OFBIN/lib:$OFBIN/lib/dummy:${LD_LIBRARY_PATH:-}"
( cd "$W/of" && timeout 900 simpleFoam > log.of 2>&1 ) || { echo "FAIL: OpenFOAM did not run"; tail -3 "$W/of/log.of"; exit 1; }
OFT=$(ls -d "$W/of"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
[ "$OFT" = "0" ] && { echo "FAIL: OpenFOAM wrote no result"; exit 1; }
echo "  ok:   OpenFOAM reference generated at t=$OFT"

run_brae() {   # run_brae <dir> <scheme-entry>
    mkcase "$1"
    python3 - "$1" "$2" <<'PY'
import re, sys
p = sys.argv[1] + '/system/fvSchemes'; s = open(p).read()
n = len(re.findall(r'div\(phi,U\)[^;]*;', s))
assert n == 1, "expected exactly one div(phi,U) entry, found %d" % n
open(p, 'w').write(re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      ' + sys.argv[2] + ';', s))
PY
    ( cd "$1" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
        echo "FAIL: brae did not run ($2)"; tail -3 "$1/run.log"; exit 1; }
    ls -d "$1"/[0-9]* | xargs -n1 basename | sort -n | tail -1
}

LUT=$(run_brae "$W/lu"  "bounded Gauss linearUpwind grad(U)") || exit 1
grep -q "linearUpwind" "$W/lu/run.log" || { echo "FAIL: brae did not report applying linearUpwind"; exit 1; }
UPT=$(run_brae "$W/upw" "bounded Gauss upwind") || exit 1
echo "  ok:   brae ran with linearUpwind (t=$LUT) and with upwind (t=$UPT)"

python3 - "$W/lu/$LUT" "$W/upw/$UPT" "$W/of/$OFT" <<'PY'
import re, sys, os, gzip
def read(f):
    for c in (f, f + '.gz'):
        if os.path.exists(c): f = c; break
    else: return None
    s = (gzip.open(f, 'rt', errors='ignore') if f.endswith('.gz') else open(f, errors='ignore')).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(', s)
    if not m: return None
    typ, n, body = m.group(1), int(m.group(2)), s[m.end():]
    if typ == 'scalar':
        return [float(x) for x in body.replace('\n', ' ').split(')')[0].split()][:n]
    return [tuple(float(x) for x in t.split()) for t in re.findall(r'\(([^()]*)\)', body)[:n]]
def l2(a, b):
    if a is None or b is None or len(a) != len(b): return None
    if isinstance(b[0], tuple):
        d = [max(abs(x[i] - y[i]) for i in range(3)) for x, y in zip(a, b)]
        g = [max(abs(y[i]) for i in range(3)) for y in b]
    else:
        d = [abs(x - y) for x, y in zip(a, b)]; g = [abs(y) for y in b]
    return (sum(x * x for x in d) / max(sum(x * x for x in g), 1e-300)) ** 0.5

lu, upw, of = sys.argv[1], sys.argv[2], sys.argv[3]
fails = 0
print("  %-9s %13s %13s %8s" % ("field", "linearUpwind", "upwind(ctl)", "ratio"))
ratios = {}
for f in ['U', 'p', 'k', 'epsilon', 'nut']:
    ref = read(of + '/' + f)
    a, b = l2(read(lu + '/' + f), ref), l2(read(upw + '/' + f), ref)
    if a is None or b is None: print("  %-9s (missing)" % f); fails += 1; continue
    ratios[f] = b / a if a > 0 else float('inf')
    print("  %-9s %13.4e %13.4e %8.2fx" % (f, a, b, ratios[f]))
    if f == 'U':
        # Absolute bar. Measured 5.20e-03; 8e-03 is above it without being so loose that the
        # discretisation could be wrong and still pass.
        if a <= 8e-3: print("  ok:   U L2 %.3e <= 8e-03 vs OpenFOAM" % a)
        else:         print("  FAIL: U L2 %.3e > 8e-03 vs OpenFOAM" % a); fails += 1

# THE CONTROL, and the substance of this gate: brae's linearUpwind must land CLOSER to OpenFOAM's
# linearUpwind answer than brae's plain upwind does. Without this the test would pass with the deferred
# correction deleted, since both schemes converge and both look plausible. Measured ratios are
# 1.9x (U), 3.2x (k), 2.3x (epsilon), 2.0x (nut); the bar is 1.5x on all four.
for f in ['U', 'k', 'epsilon', 'nut']:
    r = ratios.get(f)
    if r is None: continue
    if r >= 1.5: print("  ok:   %-7s linearUpwind is %.2fx closer to OpenFOAM than upwind (control)" % (f, r))
    else:        print("  FAIL: %-7s only %.2fx -- this case cannot see the correction" % (f, r)); fails += 1
print("PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
PY
