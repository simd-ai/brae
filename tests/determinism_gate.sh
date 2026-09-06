#!/usr/bin/env bash
# DETERMINISM GATE: the same binary on the same case twice must produce bit-identical fields.
#
# This is a foundation property, not a nicety. Every `_cpp` vs CUDA comparison in the rebuild is only as
# sharp as the run-to-run noise floor, and a defect smaller than the floor cannot be seen -- which is how
# brae's LUST implicit-weight bug hid behind a plausible residual. Before the gather rewrites this case
# drifted like this, from the same binary and the same input:
#
#     iterations    1        5        20
#     worst rel     1.9e-08  1.4e-03  3.6e-02
#
# Three scatter sites caused it, all fixed by gathering in a fixed order instead:
#   * AMG restriction        rc[map[c]] += r[c]                 (every level, every V-cycle, every PCG iter)
#   * wall functions         G0[c] / eps0[c] / omega0[c]        (cells with >1 wall face)
#   * eps setValues          source[nei] -= lower[f]*eps0[own]  (cells with >1 constrained face)
# The last two are RARE -- the case was bit-identical at 1, 5, 8, 10 and 15 iterations and differed at 12 --
# so this gate runs long enough to catch an intermittent regression, not just a systematic one.
#
# usage: determinism_gate.sh <caseDir> <iterations>
set -u
CASE="${1:?case dir}"; N="${2:-20}"
# Absolute: the runs cd into a temp case directory, so a relative path would not survive.
BRAE="${BRAE_BIN:-$(cd "$(dirname "$0")/.." && pwd)/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
CASE="$(cd "$CASE" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
for d in A B; do
    cp -r "$CASE" "$W/$d" || exit 1
    find "$W/$d" -mindepth 1 -maxdepth 1 -type d \
        ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null
    python3 - "$W/$d/system/controlDict" "$N" <<'PY'
import re, sys
p, n = sys.argv[1], sys.argv[2]
s = open(p).read()
s = re.sub(r'^endTime\s+\S+;',       'endTime         %s;' % n, s, flags=re.M)
s = re.sub(r'^writeInterval\s+\S+;', 'writeInterval   %s;' % n, s, flags=re.M)
open(p, 'w').write(s)
PY
    ( cd "$W/$d" && "$BRAE" > run.log 2>&1 ) || { echo "FAIL: run $d did not complete"; tail -5 "$W/$d/run.log"; exit 1; }
done

python3 - "$W" "$N" <<'PY'
import re, sys, os
W, N = sys.argv[1], sys.argv[2]

def vals(p):
    try: s = open(p).read()
    except OSError: return None
    m = re.search(r'internalField\s+nonuniform\s+List<\w+>\s*\n?\d+\s*\(\s*(.*?)\n\)\s*;', s, re.S)
    return [float(x) for x in re.findall(r'[-+0-9.eE]+', m.group(1))] if m else None

fails, checked = 0, 0
for f in ('U', 'p', 'k', 'epsilon', 'omega', 'nut', 'nuTilda', 'phi'):
    a = vals(os.path.join(W, 'A', N, f)); b = vals(os.path.join(W, 'B', N, f))
    if a is None or b is None or len(a) != len(b):
        continue
    checked += 1
    mx = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    mg = max((abs(x) for x in a), default=0.0) or 1.0
    ok = (mx == 0.0)
    print("  %-8s n=%6d  rel=%.3e  %s" % (f, len(a), mx / mg, "bit-identical" if ok else "DIFFERS"))
    if not ok: fails += 1

if checked == 0:
    print("FAIL: no comparable fields were written -- the gate proved nothing")
    sys.exit(1)

# NEGATIVE CONTROL. A gate that can only pass is not a gate. Perturb one value by a single ULP and require
# the same comparison to call it a difference; if this does not fire, the comparator is broken and every
# "bit-identical" line above is meaningless.
import math
a = vals(os.path.join(W, 'A', N, 'p')) or vals(os.path.join(W, 'A', N, 'U'))
b = list(a); b[0] = math.nextafter(b[0], math.inf) if b[0] == b[0] else b[0]
detected = any(x != y for x, y in zip(a, b))
print("  %-8s %s" % ("control", "OK (1-ULP perturbation detected)" if detected
                     else "FAIL (comparator cannot detect a difference)"))
if not detected: fails += 1

print("PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
PY
