#!/usr/bin/env bash
# squareBendLiq -- OpenFOAM's own TURBULENT LIQUID rhoSimpleFoam tutorial, through brae's host mirror,
# against real OpenFOAM v2412, iteration by iteration. Stage H3.5.
#
# TWO LINES OF THE TUTORIAL ARE REPLACED, and both by values the tutorial itself supplies:
#   0/T walls        the file writes `fixedValue; value uniform 350;` and then OVERRIDES it with a
#                    uniformFixedValue `expression` PatchFunction1 under the comment "For general testing
#                    purposes". brae refuses expression PatchFunction1 by name; the gate keeps the
#                    fixedValue the file states first.
#   0/U inlet        massFlowRate is a `coded` Function1 returning 5. The tutorial's own Allrun replaces
#                    it with `constant 5` whenever dynamic code is unavailable; so does the gate.
# Neither is thermo, turbulence or a scheme, and neither changes a number the solver computes. Both
# refusals stay in force on the unmodified tutorial -- this gate does not make the tutorial run, it
# measures everything else in it.
#
# WHY THIS CASE, AND WHY ITERATION BY ITERATION. The 1200-cell liqBox fixtures are orthogonal and
# upwind, and squareBendLiq is neither: `bounded Gauss linearUpwind limited` on U, e, Ekp, k and epsilon
# with `limited cellLimited Gauss linear 1` beside an UNLIMITED `default`, on a mesh whose curved bend
# blocks meet the straight ones non-orthogonally. Four generic defects lived exactly there, each visible
# at iteration 1, 2 or 3 and each measured here first (tests/liquid_turbulence_vs_openfoam.sh covers the
# thermo half and could see none of them):
#
#   1. linearUpwind on div(phi,k)/div(phi,epsilon) was REFUSED by the host closures -- the liquid
#      tutorials both name it. Ported: kEpsilon_cpp's divWithScheme and kOmegaSST_cpp's own, over the
#      gradient the entry NAMES (FieldDivScheme::luGradName -> parseNamedGradScheme).
#   2. grad(k)/grad(epsilon) for the corrected laplacian came from a shared-parser slot that ALSO
#      receives linearUpwind's named gradient (the line marked EXPERIMENT in scheme_parse.cuh), so the
#      laplacian correction ran cellLimited where OpenFOAM's correctedSnGrad takes grad(epsilon)'s own,
#      unlimited scheme: iteration 1, epsilon source 1.56e-04 off in 144 cells at the block junction.
#   3. The momentum linearUpwind correction took grad(U)'s own scheme instead of the one it names:
#      U 1.3e-03 at iteration 2 (invisible at 1 only because the start state is uniform).
#   4. After the wall-function update the closures re-evaluated EVERY epsilon/omega patch, where
#      OpenFOAM's calculateTurbulenceFields assigns only the wall-function ones and leaves inlet and
#      outlet at their last evaluate: epsilon 1.9e-06 at iteration 2 in the 68 outlet-layer cells;
#      omega 3.2e-06 under kOmegaSST.
#
# kEpsilon is the tutorial's own model; kOmegaSST runs on the same case with an omega field built from
# the tutorial's k and epsilon (omega = epsilon/(Cmu k), a mixing-length inlet of the tutorial's own
# 0.005 m, omegaWallFunction on the walls), because the fourth defect lived in both closures and a fix
# proven on one would be a patch for a tutorial rather than a port.
#
# Measured, OpenFOAM v2412, every solver at tolerance 1e-14 relTol 0, worst field per iteration:
#     kEpsilon   it 1  epsilon 1.47e-12   it 2  epsilon 1.29e-12   it 3  U 1.87e-12
#     kOmegaSST  it 1  omega   1.50e-12   it 2  U       1.79e-12   it 3  omega 1.89e-12
# and before the four fixes, on the same staging: k 2.5e-07 at iteration 1 (defect 2), U 1.3e-03 at
# iteration 2 (defect 3), epsilon 1.9e-06 / omega 3.2e-06 at iteration 2 once 2 and 3 were fixed
# (defect 4), and a refusal before any of it (defect 1). Bounds: 1e-10, 1e-11 on T.
# FAIL-PROOFS, each defect put back in the source and the gate re-run -- every one turns it red, on the
# fixture and at the iteration where it lives:
#   1  linearUpwind refused on the turbulence pair  -> brae refuses, both closures
#   2  gradKLimitK from the shared slot again       -> iteration 1  k 2.54e-07, epsilon 2.59e-06
#   3  U's correction on grad(U)'s own scheme       -> iteration 2  U 1.31e-03, p 3.09e-06
#   4  every epsilon/omega patch re-evaluated       -> iteration 2  epsilon 1.85e-06; omega 3.15e-06
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-3}

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/squareBendLiq"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

# The mesh once, for every arm.
cp -r "$SRC" "$W/mesh" || exit 1
( cd "$W/mesh" && rm -rf 0 [1-9]* processor* log.* && cp -r 0.orig 0 \
    && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 ) \
    || { echo "FAIL: blockMesh/topoSet did not run"; tail -20 "$W/mesh/log.blockMesh"; exit 1; }

# stage <dir> <model>
stage() {
    rm -rf "$1"; cp -r "$W/mesh" "$1"; rm -rf "$1"/[1-9]* "$1"/log.* "$1"/0
    ITERS="$ITERS" MODEL="$2" python3 - "$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]; n = os.environ['ITERS']; model = os.environ['MODEL']
def edit(rel, fn):
    p = os.path.join(d, rel); s = open(p).read(); s2 = fn(s)
    if s2 == s: raise SystemExit('stage: no change made to ' + rel)
    open(p, 'w').write(s2)
# 1. T walls: the fixedValue the file states first, without the testing overlay that follows it.
# The overlay nests one dictionary deeper than U's inlet: after the expression's `#};` come the
# closing braces of `uniformValue` and then of the walls entry itself, and both go.
def fixT(s):
    i = s.index('"(?i).*walls"'); j = s.index('#};', i)
    j = s.index('}', j + 3) + 1          # uniformValue
    j = s.index('}', j) + 1              # the walls entry
    return s[:i] + '"(?i).*walls"\n    {\n        type            fixedValue;\n        value           uniform 350;\n    }' + s[j:]
edit('0.orig/T', fixT)
# 2. massFlowRate: `constant 5`, the Allrun's own substitution for the coded Function1.
def fixU(s):
    i = s.index('        massFlowRate\n        {'); j = s.index('#};', i); j = s.index('}', j + 3) + 1
    return s[:i] + '        massFlowRate    constant 5;' + s[j:]
edit('0.orig/U', fixU)
# 3. solvers pinned, residualControl out, ITERS iterations written every one, 15 digits.
def fixSol(s):
    s = re.sub(r'solvers\s*\{.*?\n\}', 'solvers\n{\n    p { solver PBiCGStab; preconditioner DIC; '
               'tolerance 1e-14; relTol 0; }\n    "(U|e|k|epsilon|omega)" { solver PBiCGStab; '
               'preconditioner DILU; tolerance 1e-14; relTol 0; }\n}', s, count=1, flags=re.S)
    s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
    return re.sub(r'(epsilon\s+0\.7;)', r'\1\n        omega           0.7;', s)
edit('system/fvSolution', fixSol)
def fixCtl(s):
    s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
    for k, v in (('endTime', n), ('writeInterval', '1'), ('writeFormat', 'ascii'), ('writePrecision', '15')):
        s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
    return s
edit('system/controlDict', fixCtl)
if model == 'kOmegaSST':
    edit('constant/turbulenceProperties', lambda s: re.sub(r'RASModel\s+\S+;', 'RASModel            kOmegaSST;', s))
    edit('system/fvSchemes', lambda s: s.replace('    div(phi,k)          $turbulence;',
                                                 '    div(phi,k)          $turbulence;\n    div(phi,omega)      $turbulence;')
                                        .rstrip() + '\n\nwallDist\n{\n    method meshWave;\n}\n')
    # omega = epsilon/(Cmu k) from the tutorial's own 200 and 1; the inlet the omega twin of its
    # epsilon mixing-length inlet, at its own 0.005 m.
    open(os.path.join(d, '0.orig/omega'), 'w').write(
        'FoamFile { version 2.0; format ascii; class volScalarField; object omega; }\n'
        'dimensions [0 0 -1 0 0 0 0];\ninternalField uniform 2222.2;\nboundaryField\n{\n'
        '    inlet { type turbulentMixingLengthFrequencyInlet; mixingLength 0.005; value $internalField; }\n'
        '    outlet { type inletOutlet; inletValue $internalField; value $internalField; }\n'
        '    "(?i).*walls" { type omegaWallFunction; value $internalField; }\n}\n')
PYEOF
    cp -r "$1/0.orig" "$1/0"
}

for MODEL in kEpsilon kOmegaSST; do
    second=$([ "$MODEL" = kOmegaSST ] && echo omega || echo epsilon)
    stage "$W/of_$MODEL" "$MODEL" || { echo "FAIL: $MODEL -- staging"; fail=1; continue; }
    ( cd "$W/of_$MODEL" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ) || {
        echo "FAIL: $MODEL -- OpenFOAM did not run"; tail -15 "$W/of_$MODEL/log.rhoSimpleFoam"; fail=1; continue; }
    stage "$W/br_$MODEL" "$MODEL" || { fail=1; continue; }
    BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BRAE" -case "$W/br_$MODEL" > "$W/br_$MODEL/log.brae" 2>&1 || {
        echo "FAIL: $MODEL -- brae did not run"; grep -v '^brae NOTICE' "$W/br_$MODEL/log.brae" | tail -5
        fail=1; continue; }
    echo "== squareBendLiq, $MODEL =="
    ITERS="$ITERS" SECOND="$second" python3 - "$W/br_$MODEL" "$W/of_$MODEL" <<'PYEOF' || fail=1
import math, os, re, sys
brae, of = sys.argv[1], sys.argv[2]
n = int(os.environ['ITERS']); second = os.environ['SECOND']
bad = 0
def sc(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;', s, re.S)
    if not m: raise SystemExit('%s: not a nonuniform field' % p)
    return [float(x) for x in m.group(1).split()]
def vc(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\(\s*(.*?)\s*\)\s*;\s*\n\s*boundaryField', s, re.S)
    return [float(x) for x in re.findall(r'-?[\d.]+(?:[eE][-+]?\d+)?', m.group(1))]
def rel(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b)); den = sum(y * y for y in b)
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)
# The case must be the liquid and the scheme it claims: rho of water, and the linearUpwind the four
# defects lived in (a staging that silently fell back to upwind would pass everything below).
rho = sc(of + '/1/rho')
ok = 980 < min(rho) and max(rho) < 1010
print('     %-52s %s' % ('OpenFOAM rho is a LIQUID: %.1f..%.1f' % (min(rho), max(rho)), 'ok' if ok else 'FAIL')); bad |= not ok
sch = open(of + '/system/fvSchemes').read()
ok = 'turbulence          bounded Gauss linearUpwind limited' in sch and 'limited             cellLimited Gauss linear 1' in sch
print('     %-52s %s' % ('the tutorial names linearUpwind limited + cellLimited', 'ok' if ok else 'FAIL')); bad |= not ok
for it in range(1, n + 1):
    for f, bound in (('U', 1e-10), ('p', 1e-10), ('T', 1e-11), ('k', 1e-10), (second, 1e-10), ('nut', 1e-10)):
        a = vc('%s/%d/U' % (brae, it)) if f == 'U' else sc('%s/%d/%s' % (brae, it, f))
        b = vc('%s/%d/U' % (of, it))   if f == 'U' else sc('%s/%d/%s' % (of, it, f))
        e = rel(a, b)
        v = 'ok' if e < bound else 'FAIL (bound %g)' % bound
        print('     iteration %d  %-8s vs OpenFOAM (L2 rel)          %.6e   %s' % (it, f, e, v))
        bad |= not (e < bound)
sys.exit(1 if bad else 0)
PYEOF
done

[ "$fail" -eq 0 ] && echo "PASSED" || echo "FAILED"
exit $fail
