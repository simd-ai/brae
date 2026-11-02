#!/usr/bin/env bash
# limitedLinear on k and omega from a uniform start cannot be an exact gate, and this test says how far it can be held. On a uniform field grad() is round-off, so round-off picks the limiter's branch on
# every face: every cell of k and omega one ulp up or not, on a coin, moves OpenFOAM against itself by U 1.2e-04, nut 1.8e-02 after 40 tight iterations (job 4835375). An exact gate on this start measures that coin.
# So the oracle here is OpenFOAM's own spread: three OpenFOAM runs, straight and two coins, on the 3-D duct of tests/simple_sst_start_vs_openfoam.sh, kOmegaSST, tight, iteration 40.
#   floor      OpenFOAM against OpenFOAM: the spread is there, or this test measures nothing (premise, above 1e-6 on U)
#   brae       brae F1 on (BRAE_SST_F1_PATCH=1), straight and a coin, against OpenFOAM's three: each pair inside twice the largest OpenFOAM pair, field by field. Twice, because brae carries its own coin. MEASURED: U 8.4e-05 against 1.2e-04
# CONTROL, asserted red: `upwind` on brae's side only, a scheme that is not the case's. MEASURED: `limitedLinear 0.5` is INSIDE this spread (U 1.07e-04, nut 1.11e-02), so this test does not see an
# error of that size, and it is a net for a wrong branch of the code rather than a check of the limiter. That check is tests/simple_sst_start_vs_openfoam.sh's restarted arm, whose start is not uniform.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
sed -n '/^cat > "$W\/gate.py" <<.PY.$/,/^PY$/p' "$ROOT/tests/simple_sst_start_vs_openfoam.sh" | sed '1d;$d' > "$W/gate.py"
grep -q '^def stage' "$W/gate.py" || { echo "FAIL: the duct's generator is not where this test expects it"; exit 1; }
python3 -B "$W/gate.py" stage "$W/base" io
( set +u; source "$OFBASHRC" > /dev/null 2>&1; cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) < /dev/null || { echo "FAIL blockMesh"; exit 1; }
stage() {   # <dir> <coin seed, 0: straight> [limiter coefficient]
    cp -r "$W/base" "$1"
    python3 -B - "$1" "$2" "${3:-1}" <<'PYEOF'
import math, random, re, sys
c, seed, lim = sys.argv[1:]
p = c + '/system/controlDict'; s = open(p).read()
for k, v in (('endTime', '40'), ('writeInterval', '40'), ('writeFormat', 'ascii'), ('writePrecision', '17')):
    s, n = re.subn(r'%s\s+[^;]*;' % k, '%s %s;' % (k, v), s, count=1)
open(p, 'w').write(s)
p = c + '/system/fvSchemes'; s = open(p).read()
if lim != '1':
    s, n = re.subn(r'limitedLinear 1', lim, s)
    assert n > 0
open(p, 'w').write(s)
if seed == '0':
    raise SystemExit(0)
n = int(re.search(r'nCells:\s*(\d+)', open(c + '/constant/polyMesh/owner', errors='replace').read()).group(1))
random.seed(int(seed))
for f in ('k', 'omega'):
    p = '%s/0/%s' % (c, f)
    b = open(p).read()
    m = re.search(r'internalField\s+uniform\s+([-0-9.eE+]+);', b)
    x = float(m.group(1))
    v = [repr(math.nextafter(x, math.inf)) if random.random() < 0.5 else repr(x) for _ in range(n)]
    open(p, 'w').write(b[:m.start()] + 'internalField nonuniform List<scalar> %d\n(\n%s\n);' % (n, '\n'.join(v)) + b[m.end():])
PYEOF
}
of() { ( set +u; source "$OFBASHRC" > /dev/null 2>&1; cd "$1" && timeout 1500 simpleFoam > log 2>&1 < /dev/null ); }
br() { (cd "$1" && env BRAE_AMG_CACHE=0 BRAE_SIMPLE_CASE_SOLVER=1 BRAE_SST_F1_PATCH=1 "$BRAE" -case . > log 2>&1 < /dev/null); }
for a in of-0:0 of-1:1 of-2:2 br-0:0 br-1:1 ctl:0
do
    n=${a%:*}
    s=${a#*:}
    case $n in ctl) stage "$W/$n" "$s" upwind ;; *) stage "$W/$n" "$s" ;; esac
done
of "$W/of-0" || { echo "FAIL OpenFOAM"; exit 1; }
of "$W/of-1" || { echo "FAIL OpenFOAM"; exit 1; }
of "$W/of-2" || { echo "FAIL OpenFOAM"; exit 1; }
br "$W/br-0" || { echo "FAIL brae"; exit 1; }
br "$W/br-1" || { echo "FAIL brae"; exit 1; }
br "$W/ctl" || { echo "FAIL the control's brae"; exit 1; }
cat > "$W/cmp.py" <<'PYEOF'
import itertools, sys
import re
import numpy as np
W = sys.argv[1]
FL = ('U', 'p', 'k', 'omega', 'nut')
def rd(a, f):
    s = open('%s/%s/40/%s' % (W, a, f), errors='replace').read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', s)
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    b = s[m.end():]
    b = b[:b.index('\n)')]
    return np.array(b.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)
F = {a: {f: rd(a, f) for f in FL} for a in ('of-0', 'of-1', 'of-2', 'br-0', 'br-1', 'ctl')}
def rel(a, b, f):
    return np.linalg.norm(F[a][f] - F[b][f]) / np.linalg.norm(F[b][f])
floor = {f: max(rel(a, b, f) for a, b in itertools.combinations(('of-0', 'of-1', 'of-2'), 2)) for f in FL}
print('        OpenFOAM against OpenFOAM  ' + ' '.join('%s %.2e' % (f, floor[f]) for f in FL))
inside = {}
for x in ('br-0', 'br-1', 'ctl'):
    worst = {f: max(rel(x, o, f) for o in ('of-0', 'of-1', 'of-2')) for f in FL}
    inside[x] = all(worst[f] <= 2 * floor[f] for f in FL)
    print('        %-4s against OpenFOAM        ' % x + ' '.join('%s %.2e' % (f, worst[f]) for f in FL))
ok = floor['U'] > 1e-6
print('  %-72s %s' % ('the spread is there (premise)', 'ok' if ok else 'FAIL'))
b = inside['br-0'] and inside['br-1']
print('  %-72s %s' % ("brae inside twice OpenFOAM's own spread, field by field", 'ok' if b else 'FAIL'))
c = not inside['ctl']
print('  %-72s %s' % ('CONTROL: a scheme that is not the case'"'"'s is out of it', 'ok' if c else 'FAIL'))
raise SystemExit(0 if ok and b and c else 1)
PYEOF
python3 "$W/cmp.py" "$W" || fail=1
[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
