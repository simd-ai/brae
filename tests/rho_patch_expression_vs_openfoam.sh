#!/usr/bin/env bash
# OpenFOAM's `type expression;` PatchFunction1 -- squareBendLiq's T walls -- on both rhoSimpleFoam mirror
# arms, against real OpenFOAM v2412 evaluating the same expression. Iteration by iteration, and on the
# WALL VALUES themselves, not only on the cells they reach.
#
# WHAT RUNS VERBATIM: the tutorial's T walls,
#     uniformValue { type expression;
#         functions<scalar> { trigger { type functionObjectTrigger; triggers (2 4); defaultValue true; } }
#         variables ( "Tcrit = 500" "par1 = mag(internalField(U))/snGrad(T)" );
#         expression #{ Tcrit + par1*internalField(T) * max((Tcrit-T)/(Tcrit)*deltaT()/time(),0) #}; }
# with the coded massFlowRate inlet it ships beside it (tests/rho_coded_function1_vs_openfoam.sh). OpenFOAM
# parses it with its Lemon grammar and evaluates it in fixedEnergy::updateCoeffs -> Tw.evaluate() ->
# uniformFixedValue::updateCoeffs, once per iteration, over U after the momentum solve and T's cells and
# patch value from the previous iteration; brae parses the same text (patchExprFunction1.cu) and evaluates
# it at the same point of its step, on the host on both arms (the CUDA arm gathers the patch's cells and
# pushes the result into the device patch's refValue).
#
# THE ORACLE IS THE WRITTEN WALL. At iteration 1 the walls read 500 + par1*T_c*0.3 -- 22400 distinct values
# around 500.003, every one of them the expression's own arithmetic on that face's |U_c|, snGrad(T), T_c --
# and from iteration 2 exactly 500, because max((500 - 500.003)/500, 0) is 0. The gate compares brae's
# written wall values with OpenFOAM's at every iteration (relative L2 over the patch), then the cells
# (U p T k epsilon nut) as the tutorial gate does. Both codes' linear solvers are tightened to 1e-14 relTol
# 0 because at the tutorial's own tolerances they stop their U solves at different points and the wall
# values, which read |U_c|, would compare solver noise.
#
# BOUNDS, measured, both arms (the CUDA arm evaluates the same host code on the same downloaded cells):
#   walls at iteration 1   host 1.21e-13   CUDA 1.21e-13   (iterations 2 and 3: exactly 0, both 500)
#   cells over 1-3         host 1.86e-12   CUDA 1.86e-12   (worst: U at iteration 3)
# Bounds 1e-12 on the walls and 1e-11 on the cells.
#
# THE CONTROL: the same case with `uniformValue constant 500;` on the walls, run on the host arm, must
# MISS OpenFOAM's expression walls by at least CONTROL_MIN at iteration 1 -- measured 3.63e-07 (500
# against 500.0029 on the faces the flow has reached; most of the 22400 read ~500 at iteration 1), six
# orders above the bound. A gate whose oracle a constant wall could satisfy would not be measuring the
# expression.
#
# ALSO ASSERTED: the oracle itself is non-uniform and off 500 at iteration 1 (an OpenFOAM that had not
# evaluated the expression would leave a uniform wall); brae's written T carries the expression dictionary
# and OpenFOAM restarts from it; and each refusal below fires BY NAME while the verbatim arm -- its
# control -- runs: a function outside the carried subset (sqrt), a functions<scalar> entry referenced
# (trigger), an unregistered field (Tfoo), the expression on U (a vector), the expression on p (a field
# whose updateCoeffs no brae step evaluates), and the entry-less form (no `value`, which OpenFOAM
# evaluates at construction).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-3}
WALLBOUND_HOST=${WALLBOUND_HOST:-1e-12}
WALLBOUND_CUDA=${WALLBOUND_CUDA:-1e-12}
BOUND_HOST=${BOUND_HOST:-1e-11}
BOUND_CUDA=${BOUND_CUDA:-1e-11}
CONTROL_MIN=${CONTROL_MIN:-1e-07}
ARMS=${ARMS:-"1 cuda"}
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true   # before set -u: the bashrc reads unset variables
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/squareBendLiq"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

cp -r "$SRC" "$W/mesh" || exit 1
( cd "$W/mesh" && rm -rf 0 [1-9]* processor* log.* dynamicCode && cp -r 0.orig 0 \
    && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 ) \
    || { echo "FAIL: blockMesh/topoSet did not run"; exit 1; }

# stage <dir> <variant> -- tutorial | const500 | sqrt | trigger | unknown | onU | onP | novalue
stage() {
    rm -rf "$1"; cp -r "$W/mesh" "$1"; rm -rf "$1"/0 "$1"/log.* "$1"/dynamicCode
    ITERS="$ITERS" VARIANT="$2" python3 - "$1" <<'PYEOF'
import os, re, sys
d, n, v = sys.argv[1], os.environ['ITERS'], os.environ['VARIANT']
def edit(rel, fn):
    p = os.path.join(d, rel); s = open(p).read(); open(p, 'w').write(fn(s))
def fixSol(s):
    s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s)
    s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
    return re.sub(r'residualControl\s*\{[^}]*\}', '', s)
edit('system/fvSolution', fixSol)
def fixCtl(s):
    for k, val in (('startFrom', 'latestTime'), ('endTime', n), ('writeInterval', '1'),
                   ('writeFormat', 'ascii'), ('writePrecision', '15'), ('writeCompression', 'off')):
        s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, val), s, flags=re.M) \
            if re.search(r'^%s\s+' % k, s, re.M) else s + '\n%-15s %s;\n' % (k, val)
    return s
edit('system/controlDict', fixCtl)
def wallsDict(s):
    i = s.index('uniformValue'); j = s.index('{', i); depth = 0; k = j
    while True:
        if s[k] == '{': depth += 1
        elif s[k] == '}':
            depth -= 1
            if depth == 0: break
        k += 1
    return i, k + 1
if v == 'const500':
    def f(s):
        i, k = wallsDict(s); return s[:i] + 'uniformValue    constant 500;' + s[k:]
    edit('0.orig/T', f)
elif v == 'sqrt':
    edit('0.orig/T', lambda s: re.sub(r'expression\s*#\{.*?#\};', 'expression #{ Tcrit + sqrt(T) #};', s, count=1, flags=re.S))
elif v == 'trigger':
    edit('0.orig/T', lambda s: re.sub(r'expression\s*#\{.*?#\};', 'expression #{ Tcrit + trigger() #};', s, count=1, flags=re.S))
elif v == 'unknown':
    edit('0.orig/T', lambda s: re.sub(r'expression\s*#\{.*?#\};', 'expression #{ Tcrit + internalField(Tfoo) #};', s, count=1, flags=re.S))
elif v == 'novalue':
    def f(s):
        i = s.index('"(?i).*walls"')
        return s[:i] + s[i:].replace('        value           uniform 350;\n', '', 1)
    edit('0.orig/T', f)
elif v == 'onU':
    edit('0.orig/U', lambda s: s.replace('        type            noSlip;',
        '        type            uniformFixedValue;\n        value           uniform (0 0 0);\n'
        '        uniformValue    { type expression; expression #{ vector(0, 0, 0) #}; }', 1))
elif v == 'onP':
    edit('0.orig/p', lambda s: s.replace('        type            fixedValue;\n        value           $internalField;',
        '        type            uniformFixedValue;\n        value           $internalField;\n'
        '        uniformValue    { type expression; expression #{ 1.0e5 #}; }', 1))
PYEOF
    cp -r "$1/0.orig" "$1/0"
}

# walls <T file> -> prints the walls patch values, one per line (a uniform entry expanded to N)
walls() {
    N="$2" python3 - "$1" <<'PYEOF'
import os, re, sys
s = open(sys.argv[1]).read()
bf = s[s.index('boundaryField'):]
m = re.search(r'\n\s*(walls|"\(\?i\)\.\*walls")\s*\{', bf)
if not m: sys.exit(2)
body = bf[m.end():]
mv = re.search(r'\bvalue\s+(uniform\s+([-\d.eE+]+)|nonuniform\s+List<scalar>\s*(\d+)\s*\(([^)]*)\))', body, re.S)
if not mv: sys.exit(3)
if mv.group(2):
    for _ in range(int(os.environ['N'])): print(mv.group(2))
else:
    for x in mv.group(4).split(): print(x)
PYEOF
}
NWALLS=$(python3 - "$W/mesh/constant/polyMesh/boundary" <<'PYEOF'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r'\n\s*walls\s*\{(.*?)\}', s, re.S)
print(re.search(r'nFaces\s+(\d+)\s*;', m.group(1)).group(1))
PYEOF
)
[ -n "$NWALLS" ] || { echo "FAIL: no walls patch in the mesh"; exit 1; }

# relL2 <fileA> <fileB> -> relative L2 of A against B (one value per line)
relL2() {
    python3 - "$1" "$2" <<'PYEOF'
import math, sys
a = [float(x) for x in open(sys.argv[1])]
b = [float(x) for x in open(sys.argv[2])]
if len(a) != len(b): print('LENGTH %d vs %d' % (len(a), len(b))); sys.exit(1)
den = math.sqrt(sum(y*y for y in b))
print('%.6e' % (math.sqrt(sum((x-y)**2 for x, y in zip(a, b))) / den if den > 0 else 0.0))
PYEOF
}

# compareCells <brae> <of> <label> <bound>: the tutorial gate's comparison, U p T k epsilon nut, iterations 1..ITERS
compareCells() {
    LABEL="$3" BOUND="$4" LAST="$ITERS" python3 - "$1" "$2" <<'PYEOF'
import math, os, re, sys
brae, of = sys.argv[1], sys.argv[2]
label, bound, last = os.environ['LABEL'], float(os.environ['BOUND']), int(os.environ['LAST'])
def internal(p, n=None):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if m:
        return [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', m.group(1))]
    m = re.search(r'internalField\s+uniform\s+([^;]+);', s)
    v = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', m.group(1))]
    return v * (n // len(v)) if n and len(v) and n % len(v) == 0 else v
def rel(a, b):
    den = sum(y * y for y in b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / den) if den > 0 else 0.0
bad, worst, worstname = 0, 0.0, ''
for it in range(1, last + 1):
    for f in ('U', 'p', 'T', 'k', 'epsilon', 'nut'):
        pb, po = '%s/%d/%s' % (brae, it, f), '%s/%d/%s' % (of, it, f)
        if not (os.path.exists(pb) and os.path.exists(po)): continue
        a = internal(pb); b = internal(po, len(a))
        if len(a) != len(b):
            print('     %s it %d  %-7s LENGTH MISMATCH %d vs %d   FAIL' % (label, it, f, len(a), len(b))); bad = 1; continue
        e = rel(a, b)
        if e > worst: worst, worstname = e, '%s it %d' % (f, it)
        if e >= bound:
            print('     %s it %d  %-7s vs OpenFOAM (L2 rel)  %.6e   FAIL (bound %g)' % (label, it, f, e, bound)); bad = 1
print('     %s cells: worst %s %.3e (bound %g)   %s' % (label, worstname, worst, bound, 'FAIL' if bad else 'ok'))
sys.exit(1 if bad else 0)
PYEOF
}

runBrae() {   # runBrae <dir> <mirror>
    BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BRAE" -case "$1" > "$1/log.brae" 2>&1
}

# ---- the oracle ---------------------------------------------------------------------------------------
stage "$W/of" tutorial
( cd "$W/of" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )
if [ ! -d "$W/of/$ITERS" ]; then
    if grep -q "Failed wmake\|dynamicCode" "$W/of/log.rhoSimpleFoam"; then
        echo "SKIP: OpenFOAM cannot compile the tutorial's coded inlet here"; exit 77
    fi
    echo "FAIL: OpenFOAM did not reach iteration $ITERS"; tail -5 "$W/of/log.rhoSimpleFoam"; exit 1
fi
echo '== squareBendLiq T walls, `type expression` -- the OpenFOAM oracle =='
for it in $(seq 1 "$ITERS"); do walls "$W/of/$it/T" "$NWALLS" > "$W/of_walls_$it" || { echo "FAIL: no walls entry in OpenFOAM's $it/T"; exit 1; }; done
python3 - "$W/of_walls_1" <<'PYEOF' || fail=1
import sys
v = [float(x) for x in open(sys.argv[1])]
lo, hi = min(v), max(v)
ok = hi > lo and hi > 500.0
print('     OpenFOAM walls at iteration 1: %d faces, min %.15g max %.15g   %s' % (len(v), lo, hi,
      'non-uniform and above 500 (the expression was evaluated)  ok' if ok else 'UNIFORM: the oracle did not exercise the expression  FAIL'))
sys.exit(0 if ok else 1)
PYEOF

# ---- both arms ----------------------------------------------------------------------------------------
for MIRROR in $ARMS; do
    arm=$([ "$MIRROR" = cuda ] && echo CUDA || echo host)
    wb=$([ "$MIRROR" = cuda ] && echo "$WALLBOUND_CUDA" || echo "$WALLBOUND_HOST")
    cb=$([ "$MIRROR" = cuda ] && echo "$BOUND_CUDA" || echo "$BOUND_HOST")
    BR="$W/br_$MIRROR"
    stage "$BR" tutorial
    echo "== squareBendLiq T walls, \`type expression\` -- $arm arm =="
    if ! runBrae "$BR" "$MIRROR" || [ ! -d "$BR/$ITERS" ]; then
        echo "     $arm: DID NOT RUN                                                            FAIL"
        grep -v '^brae NOTICE' "$BR/log.brae" | tail -6; fail=1; continue
    fi
    for it in $(seq 1 "$ITERS"); do
        if ! walls "$BR/$it/T" "$NWALLS" > "$W/br_walls_${MIRROR}_$it"; then
            echo "     $arm it $it: no walls entry in brae's written T                              FAIL"; fail=1; continue
        fi
        e=$(relL2 "$W/br_walls_${MIRROR}_$it" "$W/of_walls_$it")
        if python3 -c "import sys; sys.exit(0 if float('$e') < float('$wb') else 1)" 2>/dev/null; then
            printf '     %s it %d  T walls vs OpenFOAM (L2 rel)  %s   ok (bound %s)\n' "$arm" "$it" "$e" "$wb"
        else
            printf '     %s it %d  T walls vs OpenFOAM (L2 rel)  %s   FAIL (bound %s)\n' "$arm" "$it" "$e" "$wb"; fail=1
        fi
    done
    compareCells "$BR" "$W/of" "$arm" "$cb" || fail=1
done

# ---- the control: a constant 500 K wall must be rejected by the same comparison ----------------------
echo "== control: uniformValue constant 500 on the walls, host arm =="
stage "$W/br_const500" const500
if ! runBrae "$W/br_const500" 1 || [ ! -d "$W/br_const500/1" ]; then
    echo "     control: DID NOT RUN                                                            FAIL"
    grep -v '^brae NOTICE' "$W/br_const500/log.brae" | tail -4; fail=1
else
    walls "$W/br_const500/1/T" "$NWALLS" > "$W/c_walls_1"
    e=$(relL2 "$W/c_walls_1" "$W/of_walls_1")
    if python3 -c "import sys; sys.exit(0 if float('$e') >= float('$CONTROL_MIN') else 1)" 2>/dev/null; then
        printf '     constant wall vs OpenFOAM expression walls at it 1  %s   rejected (>= %s)  ok\n' "$e" "$CONTROL_MIN"
    else
        printf '     constant wall vs OpenFOAM expression walls at it 1  %s   NOT rejected (< %s)  FAIL\n' "$e" "$CONTROL_MIN"; fail=1
    fi
fi

# ---- the writer round trip: brae's written T carries the dictionary, and OpenFOAM restarts from it ----
RT="$W/br_1"
if [ -d "$RT/$ITERS" ]; then
    echo "== round trip: brae's written state -> OpenFOAM =="
    if grep -q "type            expression;" "$RT/$ITERS/T" && grep -q "par1 = mag(internalField(U))/snGrad(T)" "$RT/$ITERS/T" \
       && grep -q "functionObjectTrigger" "$RT/$ITERS/T"; then
        echo "     brae's $ITERS/T carries the expression dictionary                                ok"
    else
        echo "     brae's $ITERS/T lost the expression dictionary                                    FAIL"; fail=1
    fi
    ( cd "$RT" && rm -rf dynamicCode && for t in [0-9]*; do [ "$t" = "$ITERS" ] || rm -rf "$t"; done \
        && sed -i "s/^endTime .*/endTime         $((ITERS + 1));/" system/controlDict \
        && rhoSimpleFoam > log.of_restart 2>&1 )
    if [ $? -eq 0 ] && [ -d "$RT/$((ITERS + 1))" ]; then
        echo "     OpenFOAM restarted from brae's written state and ran                          ok"
    else
        echo "     OpenFOAM could not restart from brae's written state                            FAIL"
        tail -8 "$RT/log.of_restart"; fail=1
    fi
fi

# ---- refusals, each by name; the verbatim arms above are their control --------------------------------
echo "== refusals by name (host arm) =="
refuses() {   # refuses <variant> <mustSay> <what>
    local d="$W/br_$1"
    stage "$d" "$1"
    if runBrae "$d" 1; then
        printf '     %-52s RAN                                              FAIL\n' "$3"; fail=1
    elif grep -q -- "$2" "$d/log.brae"; then
        printf '     %-52s refused by name                                  ok\n' "$3"
    else
        printf '     %-52s refused, but not by name                         FAIL\n' "$3"
        grep -v '^brae NOTICE' "$d/log.brae" | tail -2; fail=1
    fi
}
refuses sqrt    "sqrt"                          "a function outside the subset (sqrt)"
refuses trigger "trigger"                       "a functions<scalar> entry referenced (trigger)"
refuses unknown "Tfoo"                          "a field that is not registered (Tfoo)"
refuses onU     "non-scalar field"              "the expression on U"
refuses onP     "p on patch"                    "the expression on p"
refuses novalue "no \`value\` entry"            "no value entry (evaluated at construction)"

echo
[ "$fail" = 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"
