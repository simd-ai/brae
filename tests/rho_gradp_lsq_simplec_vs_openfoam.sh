#!/usr/bin/env bash
# `grad(p) leastSquares` through SIMPLEC's flux correction, both rhoSimpleFoam mirror arms against real
# OpenFOAM v2412 on validation/sbMatched (`consistent yes; transonic yes;`, laplacian and snGrad
# `corrected`, 112000 cells), iterations 1 and 2.
#
# WHY THIS FIXTURE. pcEqn.H:27 corrects phiHbyA with fvc::interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)
# *magSf, and fvc::snGrad(p) under `corrected` is correctedSnGrad, whose fullGradCorrection resolves
# gradScheme::New(mesh, mesh.gradScheme("grad(p)")) (correctedSnGrad.C:52-55) -- grad(p)'s OWN entry.
# The host reference's fvc::snGrad built that correction from a hardcoded Gauss gradient. Neither
# tests/rho_leastsquares_closure_vs_openfoam.sh's lsqp arm (rhoSST: not consistent, not transonic) nor
# the device-vs-host harness at iteration 1 (p is `uniform 110000` there, so every grad(p) is ~0 until
# the first pressure solve) could see it. Iteration 2 is where grad(p) first acts on this fixture.
#
# MEASURED, worst field at iteration 2 vs OpenFOAM, with the case's solvers tightened to 1e-14 relTol 0:
#   shipped (Gauss grad(p)):   host U 4.1e-12  p 2.8e-12     CUDA U 5.6e-12  p 5.6e-12
#   grad(p) leastSquares:      host U 1.98e-09 p 2.8e-10 BEFORE the fvc::snGrad fix (the fail-proof:
#                              revert the leastSquares argument in fvc.cu and this arm goes red)
#                              CUDA U 5.7e-12  p 5.7e-12 -- the device took grad(p)'s scheme from its port
#                              AFTER the fix: host worst 2.02e-11, CUDA worst 2.48e-11 (epsilon at
#                              iteration 1, the shipped control's own floor: 2.29e-11 / 2.66e-11)
#   the two schemes must DIFFER on OpenFOAM's own runs (the control that the arm can discriminate).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=2
BOUND=${BOUND:-1e-10}      # measured: worst 5.7e-12 (CUDA, U at iteration 2); the host defect read 1.98e-09
DIFFER=${DIFFER:-1e-08}    # OpenFOAM lsq vs OpenFOAM Gauss must exceed this

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
[ -f "$ROOT/validation/sbMatched/constant/polyMesh/owner" ] || { echo "SKIP: sbMatched ships no mesh"; exit 77; }
grep -q "consistent *yes" "$ROOT/validation/sbMatched/system/fvSolution" || { echo "FAIL: sbMatched is no longer SIMPLEC"; exit 1; }
grep -q "transonic *yes"  "$ROOT/validation/sbMatched/system/fvSolution" || { echo "FAIL: sbMatched is no longer transonic"; exit 1; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

stage() {   # stage <dest> <ship|lsq>
    rm -rf "$1"; cp -r "$ROOT/validation/sbMatched" "$1"; rm -rf "$1/0"; cp -r "$1/0.orig" "$1/0"
    V="$2" ITERS="$ITERS" python3 - "$1" <<'PYEOF'
import os, re, sys
d, v = sys.argv[1], os.environ['V']
p = os.path.join(d, 'system/controlDict'); s = open(p).read()
for k, val in (('startFrom','latestTime'), ('endTime',os.environ['ITERS']), ('writeInterval','1'),
               ('writeFormat','ascii'), ('writePrecision','15'), ('writeCompression','off')):
    s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k,val), s, flags=re.M) if re.search(r'^%s\s+'%k, s, re.M) else s + '\n%-15s %s;\n' % (k,val)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S); open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSolution'); s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s); s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s); open(p, 'w').write(s)
if v == 'lsq':
    p = os.path.join(d, 'system/fvSchemes'); s = open(p).read()
    s2, n = re.subn(r'(gradSchemes\s*\{\s*default\s+Gauss linear;)', r'\1\n    grad(p)         leastSquares;', s)
    assert n == 1, 'gradSchemes default Gauss linear not found once'
    open(p, 'w').write(s2)
PYEOF
}

compare() {   # compare <brae> <of> <label> <bound> <mustDiffer 0|1>
    LABEL="$3" BOUND="$4" MUST="$5" ITERS="$ITERS" python3 - "$1" "$2" <<'PYEOF'
import math, os, re, sys
b, o = sys.argv[1], sys.argv[2]
label, bound, must, n = os.environ['LABEL'], float(os.environ['BOUND']), os.environ['MUST'] == '1', int(os.environ['ITERS'])
def rd(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if m: return [float(x) for x in m.group(1).replace('(',' ').replace(')',' ').split()], 'n'
    m = re.search(r'internalField\s+uniform\s+\(?([^;)]+)\)?;', s); return [float(x) for x in m.group(1).split()], 'u'
worst, wn, bad = 0.0, '', False
for it in range(1, n + 1):
    for f in ('U', 'p', 'T', 'k', 'epsilon', 'nut'):
        pb, po = f'{b}/{it}/{f}', f'{o}/{it}/{f}'
        if not (os.path.exists(pb) and os.path.exists(po)): print('     %s it %d %s MISSING' % (label, it, f)); bad = True; continue
        a, ka = rd(pb); c, kc = rd(po)
        if kc == 'u': c = c * (len(a)//len(c))
        if ka == 'u': a = a * (len(c)//len(a))
        den = math.sqrt(sum(y*y for y in c))
        if den == 0.0: print('     %s it %d %s DEGENERATE (zero-norm oracle)' % (label, it, f)); bad = True; continue
        e = math.sqrt(sum((x-y)**2 for x,y in zip(a,c)))/den
        if e > worst: worst, wn = e, '%s it %d' % (f, it)
        if not must and e >= bound: bad = True
if must:
    ok = worst > bound
    print('     %-34s CONTROL: worst %s %.3e must exceed %g   %s' % (label, wn, worst, bound, 'ok' if ok else 'FAIL (the fixture cannot tell the schemes apart)'))
    sys.exit(0 if ok else 1)
print('     %-34s worst %s %.3e  (bound %g)   %s' % (label, wn, worst, bound, 'FAIL' if bad else 'ok'))
sys.exit(1 if bad else 0)
PYEOF
}

for v in ship lsq; do
    stage "$W/of_$v" $v; ( cd "$W/of_$v" && rhoSimpleFoam > log 2>&1 )
    [ -d "$W/of_$v/$ITERS" ] || { echo "SKIP: OpenFOAM did not reach iteration $ITERS ($v)"; tail -3 "$W/of_$v/log"; exit 77; }
    echo "== $v: $( [ $v = ship ] && echo 'shipped, Gauss grad(p) -- the control' || echo 'grad(p) leastSquares' ) =="
    for M in 1 cuda; do
        stage "$W/br_${v}_$M" $v
        if BRAE_RHOSIMPLEFOAM_MIRROR=$M "$BRAE" -case "$W/br_${v}_$M" > "$W/br_${v}_$M/log" 2>&1 && [ -d "$W/br_${v}_$M/$ITERS" ]; then
            compare "$W/br_${v}_$M" "$W/of_$v" "$( [ $M = cuda ] && echo CUDA || echo host ) vs OpenFOAM" "$BOUND" 0 || fail=1
        else
            echo "     $M: DID NOT RUN   FAIL"; grep -v '^brae NOTICE' "$W/br_${v}_$M/log" | tail -3; fail=1
        fi
    done
done
# the schemes must be distinguishable on OpenFOAM's own runs, or the lsq arm proves nothing
compare "$W/of_ship" "$W/of_lsq" "OpenFOAM Gauss vs OpenFOAM lsq" "$DIFFER" 1 || fail=1

[ $fail = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit $fail
