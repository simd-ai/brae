#!/usr/bin/env bash
# THE PRESSURE REFERENCE on the transient driver, from a DEVELOPED start, against real OpenFOAM pimpleFoam.
#
# WHY: tests/pimple_pressure_reference_vs_openfoam.sh asserts the same reference from a UNIFORM start, and its bounds
# (U 1e-3, p 5e-3) are one GB10 measurement rounded up. On a uniform k every internal face has k[N] - k[P] = 0 exactly,
# so NVDTVD::r takes its sign rule and limitedLinear chooses linear or upwind by the ROUNDING SIGN of d & grad(k). That
# is a coin flip in OpenFOAM itself: its serial and np2 runs choose differently on 11,823 of 24,170 faces at step one
# (measured 2026-11-21). The old gate passed on the GB10 because brae and OpenFOAM's packaged binary round
# the same way there, and failed on the GH200 because nvcc 13.0 rounds gradKernel's x update differently for sm_90. It
# measured which compiler ran, not the discretisation. This gate starts where every face has a real gradient, and
# derives its bound from OpenFOAM's own variability in each run. The old gate stays until this one has passed on both
# the GB10 and the GH200.
#
# THE CASE: the committed pitzDaily fixture staged exactly as the old gate stages it (transient, PIMPLE 2x2, the outlet's
# p and U zeroGradient so p.needReference() is true, pRefCell 0, U started at the inlet value). OpenFOAM serial runs it
# 100 steps of 2e-5 to t0 = 0.002 (the precursor); from that one written state OpenFOAM and brae run the same 10 steps
# with every solve to tolerance 1e-12, relTol 0, so where a solver stopped is not part of the comparison.
#
# THE REFERENCE: OpenFOAM serial. ITS OWN VARIABILITY: the same window decomposed in two (simple, 2 1 1) and in four
# (scotch), reconstructed. A decomposition changes the summation order and the solver's path and nothing in the
# discretisation, which is the class of difference between two correct implementations.
#
# THE BOUND, per field and per write time, from OpenFOAM alone: 10 x the larger of OpenFOAM's np2 and np4 differences
# from its serial run, and never below 1e-13. The measure is max|difference| / max|OpenFOAM serial| over the internal
# field; p is compared with each field's mean removed (np4's reference cell lands on another cell). The factor is the
# envelope rule of the rhoCentralFoam gates. MEASURED on the GB10 (2026-11-21): OpenFOAM's own scatter over np2-np8
# (simple and scotch) reaches 4.5-70x this envelope at six and eight ranks, so the bound is tighter than OpenFOAM's
# agreement with itself there; brae reads 1e-4 to 0.053 of it at every write time (p and U closest, at the first steps).
#
# PREMISES: OpenFOAM's t0 state has no internal face with k[N] == k[P] (no face is decided by a rounding sign: the
# reason for this gate), and brae reports that the case needs a pressure reference.
#
# THE CONTROLS, each must be RED by more than 100x the bound, or the gate could not tell a wrong code from a right one:
#   no reference  BRAE_NO_PREF=1 (the old gate's control): diverges, or ends far from OpenFOAM;
#   gradient      BRAE_CONTROL_LIMITER=grad:0.5: limitedLinear reads half the cell gradient;
#   limiter       BRAE_CONTROL_LIMITER=k:0.5: limitedLinear 0.5 where the case asks for 1.
#
# NOT CLAIMED: the uniform start (the old gate's), adjustTimeStep (refused in its own gates), any decomposed brae run,
# the case's shipped solver tolerances (where each code stops is a different question).
set -u
BIN="${1:?brae_pimpleFoam binary}"
SRC="${2:?committed pitzDaily case dir}"
W="${3:?work dir}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
FLOOR=${BRAE_PREF_DEVELOPED_FLOOR:-1e-13}

[ -x "$BIN" ] || { echo "SKIP: no brae_pimpleFoam at $BIN"; exit 125; }
[ -f "$SRC/constant/polyMesh/points" ] || { echo "SKIP: fixture '$SRC' not present"; exit 125; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 125; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 125; }
set +u
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for c in pimpleFoam decomposePar reconstructPar mpirun; do
    command -v $c > /dev/null 2>&1 || { echo "SKIP: $c not on PATH"; exit 125; }
done
export PYTHONDONTWRITEBYTECODE=1
fails=0
passes=0
ok()   { echo "  ok:   $*"; passes=$((passes + 1)); }
fail() { echo "  FAIL: $*"; fails=$((fails + 1)); }

# ---- the case, staged as tests/pimple_pressure_reference_vs_openfoam.sh stages it ------------------------------------
stage()   # stage <dir>
{
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    rm -rf "$1"/[1-9]* "$1"/log.*
    cat > "$1/system/controlDict" <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.002;
deltaT          2e-5;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  17;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
adjustTimeStep  no;
EOF
    python3 - "$1" <<'PYEOF' || return 1
import re, sys
d = sys.argv[1]
s = open(d + '/system/fvSchemes').read()
s = re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default         Euler;\n}', s, flags=re.S)
open(d + '/system/fvSchemes', 'w').write(s)
s = open(d + '/system/fvSolution').read()
m = re.search(r'\n(SIMPLE|PIMPLE)\s*\{', s)
if m:
    i, depth = m.end() - 1, 0
    while i < len(s):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    s = s[:m.start()] + s[i + 1:]
s += ('\nPIMPLE\n{\n    nOuterCorrectors 2;\n    nCorrectors     2;\n'
      '    nNonOrthogonalCorrectors 0;\n    pRefCell        0;\n    pRefValue       0;\n}\n')
def block(text, key):
    m = re.search(r'\n(\s*)%s\s*\n?\s*\{' % re.escape(key), text)
    if not m:
        return None
    i, depth = text.index('{', m.start()), 0
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[text.index('{', m.start()):i + 1]
        i += 1
    return None
finals = ''
for key in ('p', '"(U|k|epsilon|omega|f|v2)"'):
    b = block(s, key)
    if b:
        finals += '\n    %s\n    %s\n' % (key[:-1] + 'Final"' if key.startswith('"') else key + 'Final', b)
assert finals.count('Final') >= 2, 'the fixture\'s fvSolution has no p and momentum solver blocks to copy'
m = re.search(r'\nsolvers\s*\n?\s*\{', s)
assert m, 'the fixture has no solvers dictionary'
i, depth = s.index('{', m.start()), 0
while i < len(s):
    if s[i] == '{':
        depth += 1
    elif s[i] == '}':
        depth -= 1
        if depth == 0:
            break
    i += 1
s = s[:i] + finals + s[i:]
open(d + '/system/fvSolution', 'w').write(s)
p = d + '/0/p'
t = open(p).read()
m = re.search(r'(\n    outlet\s*\{)([^}]*)(\})', t, re.S)
assert m and 'fixedValue' in m.group(2), 'the fixture\'s outlet p is no longer fixedValue'
t = t[:m.start(2)] + '\n        type            zeroGradient;\n    ' + t[m.end(2):]
open(p, 'w').write(t)
assert 'fixedValue' not in open(p).read(), 'another p patch still fixes a value -- the case would not need a reference'
u = d + '/0/U'
t = open(u).read()
m = re.search(r'(\n    outlet\s*\{)([^}]*)(\})', t, re.S)
assert m, 'the fixture no longer has an `outlet` U patch'
t = t[:m.start(2)] + '\n        type            zeroGradient;\n    ' + t[m.end(2):]
t, n = re.subn(r'^internalField\s+uniform\s+\([^)]*\);', 'internalField   uniform (10 0 0);', t, flags=re.M)
assert n == 1, 'U has no uniform internalField to start from the inlet value'
open(u, 'w').write(t)
PYEOF
}

# window <dir> [np method]: the precursor's t0 state, 10 steps of 2e-5 written at every step, every solve to 1e-12
window()
{
    local d=$1
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$W/pre/constant" "$W/pre/system" "$W/pre/0.002" "$d/"
    python3 - "$d" "${2:-}" "${3:-}" <<'PYEOF' || return 1
import re, sys
d, n, method = sys.argv[1], sys.argv[2], sys.argv[3]
c = open(d + '/system/controlDict').read()
for k, v in (('startFrom', 'latestTime'), ('endTime', '0.0022'), ('writeInterval', '1')):
    c, k_n = re.subn(r'^%s\s+[^;]+;' % k, '%s %s;' % (k, v), c, flags=re.M)
    assert k_n == 1, k
open(d + '/system/controlDict', 'w').write(c)
s = open(d + '/system/fvSolution').read()
s, nr = re.subn(r'relTol\s+[^;]+;', 'relTol 0;', s)
s, nt = re.subn(r'tolerance\s+[^;]+;', 'tolerance 1e-12;', s)
assert nr >= 4 and nt >= 4, 'fvSolution lost its solver blocks'
open(d + '/system/fvSolution', 'w').write(s)
if n:
    open(d + '/system/decomposeParDict', 'w').write(
        'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\n'
        'numberOfSubdomains %s;\nmethod %s;\ncoeffs{n (%s 1 1);}\n' % (n, method, n))
PYEOF
}

# ---- OpenFOAM: the precursor, then the window serial, np2 and np4 ----------------------------------------------------
stage "$W/pre" || { echo "FAIL: staging"; exit 1; }
( cd "$W/pre" && pimpleFoam > log.pimpleFoam 2>&1 ) || { echo "FAIL: OpenFOAM precursor"; tail -20 "$W/pre/log.pimpleFoam"; exit 1; }
[ -d "$W/pre/0.002" ] || { echo "FAIL: the precursor wrote no 0.002"; exit 1; }
window "$W/of" && window "$W/of2" 2 simple && window "$W/of4" 4 scotch || { echo "FAIL: staging the window"; exit 1; }
( cd "$W/of" && pimpleFoam > log.pimpleFoam 2>&1 ) || { echo "FAIL: OpenFOAM serial window"; exit 1; }
for n in 2 4; do
    ( cd "$W/of$n" && decomposePar > log.decomposePar 2>&1 && mpirun -np $n pimpleFoam -parallel > log.pimpleFoam 2>&1 \
          && reconstructPar > log.reconstructPar 2>&1 ) || { echo "FAIL: OpenFOAM np$n window"; tail -5 "$W/of$n/log.pimpleFoam"; exit 1; }
done
echo "OpenFOAM: precursor 0 -> 0.002, window 0.002 -> 0.0022 serial, np2 (simple), np4 (scotch)"

# ---- brae, and the controls -------------------------------------------------------------------------------------------
window "$W/brae" && ( cd "$W/brae" && "$BIN" -case . > log.brae 2>&1 ) || { echo "FAIL: brae_pimpleFoam"; tail -20 "$W/brae/log.brae"; exit 1; }
ctl()   # ctl <name> <env assignment>: 0 when it ran to the end, 2 when it stopped on divergence, 1 otherwise
{
    window "$W/$1" || return 1
    ( cd "$W/$1" && env "$2" "$BIN" -case . > log.brae 2>&1 ) && return 0
    grep -q "diverged" "$W/$1/log.brae" && return 2
    return 1
}
ctl noref BRAE_NO_PREF=1; rcNoref=$?
ctl grad BRAE_CONTROL_LIMITER=grad:0.5; rcGrad=$?
ctl lim BRAE_CONTROL_LIMITER=k:0.5; rcLim=$?

# ---- premises -----------------------------------------------------------------------------------------------------------
grep -q "pressure needs reference" "$W/brae/log.brae" \
    && ok "premise: brae reports that the case needs a pressure reference" \
    || fail "premise: brae did not report that the case needs a pressure reference"
grep -q "BRAE_NO_PREF: the pressure reference is SUPPRESSED" "$W/noref/log.brae" \
    && ok "control no-reference announced itself" || fail "control no-reference did not announce itself"
for c in grad lim; do
    grep -q "BRAE_CONTROL_LIMITER=.*WRONG on purpose" "$W/$c/log.brae" \
        && ok "control $c announced itself" || fail "control $c did not announce itself"
done
prem=$(python3 - "$W/pre" <<'PYEOF'
import re, sys
import numpy as np
d = sys.argv[1]
def lst(p, kind):
    t = open(p).read(); t = t[t.index('FoamFile'):]; body = t[t.index('}') + 1:]
    m = re.search(r'(\d+)\s*\(', body); n = int(m.group(1))
    return np.array(body[m.end():].split(')')[0].split()[:n], dtype=kind)
own, nei = lst(d + '/constant/polyMesh/owner', np.int64), lst(d + '/constant/polyMesh/neighbour', np.int64)
s = open(d + '/0.002/k').read()
k = np.array(re.search(r'internalField\s+nonuniform List<scalar>\s*\d+\s*\((.*?)\n\)\s*;', s, re.S).group(1).split(), dtype=float)
print(len(nei), int(np.sum(k[own[:len(nei)]] == k[nei])))
PYEOF
)
read -r nIf nEq <<< "$prem"
if [ "${nEq:-1}" = 0 ]; then
    ok "premise: at t0 no internal face of $nIf has k[N] == k[P] (no limiter decided by a rounding sign)"
else
    fail "premise: $nEq of $nIf internal faces have k[N] == k[P] at t0: the start is not developed"
fi

# ---- the comparison -----------------------------------------------------------------------------------------------------
res=$(python3 - "$W" "$FLOOR" "$rcNoref" "$rcGrad" "$rcLim" <<'PYEOF'
import os, re, sys
import numpy as np
W, floor = sys.argv[1], float(sys.argv[2])
rc = dict(zip(('noref', 'grad', 'lim'), sys.argv[3:6]))
F = ('U', 'p', 'k', 'epsilon', 'nut')
def rd(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform List<(vector|scalar)>\s*(\d+)\s*\((.*?)\n\)\s*;', s, re.S)
    if not m:
        raise SystemExit('BAD %s has no nonuniform internalField' % p)
    if m.group(1) == 'vector':
        return np.array([[float(x) for x in t.split()] for t in re.findall(r'\(([^()]*)\)', m.group(3))])
    return np.array([float(x) for x in m.group(3).split()])
def rel(a, b, f):
    if f == 'p':
        a, b = a - a.mean(), b - b.mean()
    return float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
times = sorted((t for t in os.listdir(W + '/of') if re.fullmatch(r'[0-9.e-]+', t) and float(t) > 0.002), key=float)
if len(times) != 10:
    raise SystemExit('BAD OpenFOAM wrote %d window times, not 10' % len(times))
bound = {}
worst = (0.0, '')
over = []
for t in times:
    for f in F:
        o = rd('%s/of/%s/%s' % (W, t, f))
        env = max(rel(o, rd('%s/of2/%s/%s' % (W, t, f)), f), rel(o, rd('%s/of4/%s/%s' % (W, t, f)), f))
        b = max(10 * env, floor)
        bound[(t, f)] = b
        if not os.path.isdir('%s/brae/%s' % (W, t)):
            raise SystemExit('BAD brae wrote no %s' % t)
        v = rel(o, rd('%s/brae/%s/%s' % (W, t, f)), f)
        if v > b:
            over.append('%s:%s(%.2e>%.2e)' % (t, f, v, b))
        if v / b > worst[0]:
            worst = (v / b, '%s at t = %s: %.2e against a bound of %.2e (OpenFOAM np2/np4 %.2e)' % (f, t, v, b, env))
print('BRAE', 'FAIL' if over else 'OK', '%.3f' % worst[0], ' '.join(over) if over else worst[1])
for c in ('noref', 'grad', 'lim'):
    if rc[c] == '2':
        print('CTL', c, 'DIVERGED', 'inf', 'stopped on divergence'); continue
    if rc[c] != '0':
        print('CTL', c, 'BROKEN', '0', 'the control run failed for a reason other than divergence'); continue
    best = (0.0, '')
    for t in times:
        if not os.path.isdir('%s/%s/%s' % (W, c, t)):
            continue
        for f in F:
            v = rel(rd('%s/of/%s/%s' % (W, t, f)), rd('%s/%s/%s/%s' % (W, c, t, f)), f)
            if v / bound[(t, f)] > best[0]:
                best = (v / bound[(t, f)], '%s at t = %s: %.2e, %.3g x its bound' % (f, t, v, v / bound[(t, f)]))
    print('CTL', c, 'RED' if best[0] > 100 else 'WEAK', '%.4g' % best[0], best[1])
PYEOF
)
[ -n "$res" ] || { echo "FAIL: the comparison did not run"; exit 1; }
while read -r kind a b c rest; do
    case "$kind $a" in
        "BAD "*)     fail "comparison: $a $b $c $rest" ;;
        "BRAE OK")   ok "brae within 10x OpenFOAM's own np2/np4 spread at all 10 write times, U p k epsilon nut; closest ($b of its bound): $c $rest" ;;
        "BRAE FAIL") fail "brae above the bound: $b $c $rest" ;;
        CTL*)        case $b in
                         RED|DIVERGED) ok "control $a is red: $rest" ;;
                         *)            fail "control $a is $b: $rest" ;;
                     esac ;;
    esac
done <<< "$res"
echo "pimple_pressure_reference_developed_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 9 ]
