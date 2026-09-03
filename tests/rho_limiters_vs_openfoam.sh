#!/usr/bin/env bash
# THE CASE'S GRADIENT LIMITERS AND LAPLACIAN LIMITER REACH THE MIRROR -- both arms against OpenFOAM.
#
# parseFvSchemesControls has carried every one of these for a long time; buildStepInput forwarded only
# nonOrth, grad(U) and grad(k). So the energy's linearUpwind ran an UNLIMITED deferred correction under a
# case that limits it, and `limited <psi>` ran the uncapped correction under the limited name.
#
#   ARM A  rhoKE2 (laminar, e-thermo, linearUpwind on div(phi,e) and div(phi,Ekp) with named gradients)
#          with `grad(e)` and `grad(Ekp)` made `cellLimited Gauss linear 1` -> both arms match OpenFOAM.
#          CONTROL: OpenFOAM's own limited and unlimited answers must differ by >= 100x the bound.
#   ARM B  rhoBoxSym (4 degrees non-orthogonal, laminar) with `limited 0.5`: the HOST refuses by name
#          (its energy and pressure laplacians implement `corrected` only) and the DEVICE matches
#          OpenFOAM -- its laplacianCorrFlux is OpenFOAM's limitedSnGrad limiter exactly.
#          CONTROL: OpenFOAM's `limited 0.5` and `corrected` answers differ by >= 100x the bound.
#   ARM C  rhoBoxSym with `limited 0`: BOTH arms refuse by name. OpenFOAM's `limited 0` keeps the
#          non-orthogonal implicit coefficients and zeroes the correction -- a third regime, U 1.3e-04
#          from `orthogonal` and 1.9e-03 from `corrected` -- which the mirror does not represent.
#   ARM D  rhoBoxSym with `corrected`: both arms match OpenFOAM (the baseline arm B is measured against).
#
# Every linear solver is pinned to 1e-12/0 and the comparison is at 20 iterations, so the trajectories
# are comparable and a discretisation difference cannot hide behind convergence.
# Measured (t=20, both arms vs OpenFOAM): A T 1.5e-12 / U 4.5e-12 (host), T 1.3e-12 / U 5.9e-12 (cuda)
# against a limited-vs-unlimited control of T 2.35e-02; D and B T 1.3e-09 / p 1.4e-10 against a
# limited-vs-corrected control of T 1.1e-07 / p 4.9e-08. Fail-proof (the old forwarding: nonOrth alone,
# no gradient limiters): A T 2.35e-02 on both arms -- the whole control -- B's device arm p 4.9e-08, and
# neither refusal fires. The forwarding also uncovered a latent crash: the host kinetic-energy path
# handed cellLimitGrad a boundary-less shim (SIGSEGV the first time the coefficient was non-zero),
# now a values overload.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=20
# ARM A (rhoKE2) at the linear-solver floor. ARMS B/D (rhoBoxSym): T and p at ~10x their measured
# 1.3e-09 / 1.4e-10; U at 3e-05 because rhoBoxSym carries a 5.2e-06 residual in U on BOTH arms whatever
# the laplacian says (the tilted symmetry plane's own, see validation/rhoBoxSym/README.md) -- so U is
# not the field that discriminates the laplacian regime there, T and p are. The control ratio is 10:
# a device that ignored the cap would read OpenFOAM's corrected answer, 49x the p bound away.
A_BOUND=${A_BOUND:-1e-09}; BT_BOUND=1e-08; BP_BOUND=1e-09; BU_BOUND=3e-05
CONTROL_RATIO=10

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # $1 fixture  $2 dst  $3 python mutation on the fvSchemes text (a `s = ...` snippet)
{
    rm -rf "$2"; cp -r "$ROOT/validation/$1" "$2"; rm -rf "$2"/[1-9]* "$2"/0; cp -r "$2/0.orig" "$2/0"
    ( cd "$2" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on $1"; exit 1; }
    MUT="$3" python3 - "$2" "$N" <<'PYEOF'
import os, re, sys
d, iters = sys.argv[1], sys.argv[2]
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', iters), ('writeInterval', iters),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
open(f, 'w').write(s)
f = os.path.join(d, 'system/fvSchemes'); s = open(f).read()
exec(os.environ['MUT'])
open(f, 'w').write(s)
PYEOF
}
NONE=""
GRADLIM="s, n = re.subn(r'gradSchemes\s*\{[^}]*\}', 'gradSchemes { default Gauss linear; grad(e) cellLimited Gauss linear 1; grad(Ekp) cellLimited Gauss linear 1; }', s); assert n == 1"
lap() { echo "s, n = re.subn(r'laplacianSchemes\s*\{[^}]*\}', 'laplacianSchemes { default Gauss linear $1; }', s); assert n == 1; s, n = re.subn(r'snGradSchemes\s*\{[^}]*\}', 'snGradSchemes { default $1; }', s); assert n == 1"; }

runOF()   { ( cd "$1" && rhoSimpleFoam > of.log 2>&1 ) || { tail -5 "$1/of.log"; echo "FAIL: OpenFOAM did not run ($1)"; exit 1; }; }
runBrae() { ( cd "$2" && BRAE_RHOSIMPLEFOAM_MIRROR=$1 "$BIN" -case "$2" > run.log 2>&1 ); }
cmp()     # $1 label  $2 brae dir  $3 of dir  $4 T bound  $5 U bound  $6 p bound
{
    A="$2" B="$3" N="$N" TB="$4" UB="$5" PB="$6" LABEL="$1" python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
ok = True
for f, key in (('T', 'TB'), ('U', 'UB'), ('p', 'PB')):
    a = read(os.path.join(os.environ['A'], os.environ['N'], f)); b = read(os.path.join(os.environ['B'], os.environ['N'], f))
    r = float(np.linalg.norm(a - b) / np.linalg.norm(b)); bnd = float(os.environ[key]); good = r < bnd
    print('     %-40s %-2s brae vs OpenFOAM %.4e   (bound %.1e)   %s' % (os.environ['LABEL'], f, r, bnd, 'ok' if good else 'FAIL'))
    ok = ok and good
sys.exit(0 if ok else 1)
PYEOF
}
control()   # $1 label  $2 of dir  $3 of reference dir  $4 T bound  $5 U bound  $6 p bound
{
    A="$2" B="$3" N="$N" TB="$4" UB="$5" PB="$6" RATIO="$CONTROL_RATIO" LABEL="$1" python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
rels = {f: float(np.linalg.norm(read(os.path.join(os.environ['A'], os.environ['N'], f)) - read(os.path.join(os.environ['B'], os.environ['N'], f))) / np.linalg.norm(read(os.path.join(os.environ['B'], os.environ['N'], f)))) for f in ('T', 'U', 'p')}
bounds = {'T': float(os.environ['TB']), 'U': float(os.environ['UB']), 'p': float(os.environ['PB'])}
best = max((r / bounds[f], f) for f, r in rels.items())
good = best[0] > float(os.environ['RATIO'])
print('     control: %-30s OpenFOAM T %.2e U %.2e p %.2e = %.0fx the bound on %s   %s' % (os.environ['LABEL'], rels['T'], rels['U'], rels['p'], best[0], best[1], 'ok' if good else 'FAIL (inert)'))
sys.exit(0 if good else 1)
PYEOF
}

# ---- ARM A: the energy gradient limiters ---------------------------------------------------------
stage rhoKE2 "$W/A_of"     "$GRADLIM"; runOF "$W/A_of"
stage rhoKE2 "$W/A_ofRef"  "$NONE";    runOF "$W/A_ofRef"
control "grad(e|Ekp) cellLimited vs unlimited" "$W/A_of" "$W/A_ofRef" "$A_BOUND" "$A_BOUND" "$A_BOUND"
for arm in 1 cuda; do
    stage rhoKE2 "$W/A_$arm" "$GRADLIM"; runBrae $arm "$W/A_$arm" || { tail -12 "$W/A_$arm/run.log"; echo "FAIL: arm $arm did not run arm A"; exit 1; }
    cmp "A grad(e|Ekp) cellLimited, arm $arm" "$W/A_$arm" "$W/A_of" "$A_BOUND" "$A_BOUND" "$A_BOUND"
done
say "the energy gradient limiters reach the linearUpwind corrections, both arms" "$([ $fail = 0 ] && echo ok || echo FAIL)"

# ---- ARM D then B: the laplacian regimes on the tilted mesh --------------------------------------
stage rhoBoxSym "$W/D_of" "$(lap corrected)";   runOF "$W/D_of"
stage rhoBoxSym "$W/B_of" "$(lap 'limited 0.5')"; runOF "$W/B_of"
control "limited 0.5 vs corrected" "$W/B_of" "$W/D_of" "$BT_BOUND" "$BU_BOUND" "$BP_BOUND"
for arm in 1 cuda; do
    stage rhoBoxSym "$W/D_$arm" "$(lap corrected)"; runBrae $arm "$W/D_$arm" || { tail -4 "$W/D_$arm/run.log"; echo "FAIL: arm $arm did not run arm D"; exit 1; }
    cmp "D corrected, arm $arm" "$W/D_$arm" "$W/D_of" "$BT_BOUND" "$BU_BOUND" "$BP_BOUND"
done
stage rhoBoxSym "$W/B_1" "$(lap 'limited 0.5')"; out=$( cd "$W/B_1" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$W/B_1" 2>&1 || true )
echo "$out" | grep -q "limited <k> corrected" && ! [ -d "$W/B_1/$N" ] \
    && say "B limited 0.5: the host arm refuses by name (corrected only there)" ok \
    || { echo "$out" | tail -3; say "B limited 0.5: the host arm refuses by name (corrected only there)" FAIL; }
stage rhoBoxSym "$W/B_cuda" "$(lap 'limited 0.5')"; runBrae cuda "$W/B_cuda" || { tail -4 "$W/B_cuda/run.log"; echo "FAIL: the CUDA arm did not run arm B"; exit 1; }
cmp "B limited 0.5, arm cuda" "$W/B_cuda" "$W/B_of" "$BT_BOUND" "$BU_BOUND" "$BP_BOUND"
say "the capped non-orthogonal correction reaches the device laplacians" "$([ $fail = 0 ] && echo ok || echo FAIL)"

# ---- ARM C: limited 0 is refused on both arms ---------------------------------------------------
for arm in 1 cuda; do
    stage rhoBoxSym "$W/C_$arm" "$(lap 'limited 0')"; out=$( cd "$W/C_$arm" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BIN" -case "$W/C_$arm" 2>&1 || true )
    echo "$out" | grep -q "limited 0" && ! [ -d "$W/C_$arm/$N" ] \
        && say "C limited 0 is refused by name (arm $arm)" ok \
        || { echo "$out" | tail -3; say "C limited 0 is refused by name (arm $arm)" FAIL; }
done
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
