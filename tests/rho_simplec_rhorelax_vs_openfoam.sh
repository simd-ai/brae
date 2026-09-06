#!/usr/bin/env bash
# rho.relax() ON THE SIMPLEC BRANCH relaxes against rho.prevIter() -- the density the ITERATION STARTED
# with -- not the one pcEqn.H:1 just recomputed. Both mirror arms against real OpenFOAM.
#
# OpenFOAM: simpleControl::loop() calls storePrevIterFields() at the start of the iteration
# (simpleControl.C:157); GeometricField::relax is prevIter + alpha*(this - prevIter)
# (GeometricField.C:1089-1095); pcEqn.H opens with `rho = thermo.rho()` (:1) and closes with another
# `rho = thermo.rho()` then rho.relax() (:118,:122). brae captured its relaxation base at the tail, after
# the pcEqn opening had already moved rho -- exact on the pEqn branch, wrong under `consistent yes`.
#
# WHY NO FIXTURE SAW IT: every consistent+subsonic fixture in validation/ relaxes rho at 1.0, and the
# stock tutorials that relax rho (angledDuct, NACA: 0.01) are not consistent. And WHY THIS GATE IS TAKEN
# MID-TRANSIENT: at 200 iterations rhoBox is converged and the fixed point does not depend on the
# relaxation path -- OpenFOAM's own rho-0.5 and rho-1.0 answers agree to 1e-12 there. At 10 iterations
# they differ by 1.35e-03 in rho, which is the control below; linear solvers are pinned to 1e-12/0 on
# every run so the trajectories are comparable (see converged-not-iteration-count).
#
# Measured (rhoBox, `consistent yes`, `rho 0.5`, 10 iterations, both arms vs OpenFOAM):
#   fixed       rho 2.59e-12   p 2.88e-12   T 1.44e-12   U 4.93e-11
#   fail-proof  rho 1.35e-03 on BOTH arms (the tail capture: the whole effect of the factor, lost),
#               p/T/U unchanged -- which is why rho carries the bound that matters.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
SRC="$ROOT/validation/rhoBox"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=10
RHO_BOUND=1e-10; P_BOUND=1e-10; T_BOUND=1e-10; U_BOUND=1e-09
CONTROL_MIN=1e-04   # OF rho-0.5 vs OF rho-1.0 must differ by at least this in rho, else the factor is inert

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-64s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # $1 dst  $2 rho factor -- SIMPLEC, residualControl off, tight solvers, N iterations
{
    RHO="$2" python3 - "$SRC" "$1" "$N" <<'PYEOF'
import os, re, shutil, sys
src, dst, iters = sys.argv[1], sys.argv[2], sys.argv[3]; rho = os.environ['RHO']
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
for d in ('0',) + tuple(x for x in os.listdir(dst) if re.fullmatch(r'[1-9][0-9]*', x)):
    shutil.rmtree(os.path.join(dst, d), ignore_errors=True)
shutil.copytree(os.path.join(dst, '0.orig'), os.path.join(dst, '0'))
c = os.path.join(dst, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', iters), ('writeInterval', iters),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
f = os.path.join(dst, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'(SIMPLE\s*\{)', r'\1 consistent yes;', s, count=1)
s = re.sub(r'\brho\s+[0-9.]+\s*;', 'rho %s;' % rho, s, count=1)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
assert 'consistent yes' in s and ('rho %s;' % rho) in s
open(f, 'w').write(s)
PYEOF
}

stage "$W/of05" 0.5; stage "$W/of10" 1.0
( cd "$W/of05" && rhoSimpleFoam > of.log 2>&1 ) || { tail -5 "$W/of05/of.log"; echo "FAIL: OpenFOAM (rho 0.5) did not run"; exit 1; }
( cd "$W/of10" && rhoSimpleFoam > of.log 2>&1 ) || { tail -5 "$W/of10/of.log"; echo "FAIL: OpenFOAM (rho 1.0) did not run"; exit 1; }
for arm in 1 cuda; do
    stage "$W/brae_$arm" 0.5
    ( cd "$W/brae_$arm" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BIN" -case "$W/brae_$arm" > run.log 2>&1 ) \
        || { tail -5 "$W/brae_$arm/run.log"; echo "FAIL: the mirror (arm $arm) did not run"; exit 1; }
done

W="$W" N="$N" RHO_BOUND="$RHO_BOUND" P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" U_BOUND="$U_BOUND" CONTROL_MIN="$CONTROL_MIN" \
python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], os.environ['N']
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
def rel(a, b, f):
    x = read(os.path.join(W, a, N, f)); y = read(os.path.join(W, b, N, f))
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))
ok = True
c = rel('of05', 'of10', 'rho')
good = c > float(os.environ['CONTROL_MIN'])
print('     control: OpenFOAM rho 0.5 vs rho 1.0 differ in rho by %.3e   %s' % (c, 'ok' if good else 'FAIL (the factor is inert here)'))
ok = ok and good
for arm in ('1', 'cuda'):
    for f, key in (('rho', 'RHO_BOUND'), ('p', 'P_BOUND'), ('T', 'T_BOUND'), ('U', 'U_BOUND')):
        r = rel('brae_' + arm, 'of05', f); b = float(os.environ[key]); good = r < b
        print('     arm %-4s %-4s vs OpenFOAM %.4e   (bound %.1e)   %s' % (arm, f, r, b, 'ok' if good else 'FAIL'))
        ok = ok and good
sys.exit(0 if ok else 1)
PYEOF
say "SIMPLEC rho.relax() relaxes against the iteration's starting density, both arms" "$([ $fail = 0 ] && echo ok || echo FAIL)"
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
