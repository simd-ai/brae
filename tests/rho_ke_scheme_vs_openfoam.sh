#!/usr/bin/env bash
# div(phi,K) / div(phi,Ekp) IS ITS OWN fvSchemes ENTRY. EEqn.H builds fvc::div(phi, K) on an h-thermo and
# fvc::div(phi, Ekp) on an e-thermo, and OpenFOAM resolves each under its own key. brae's buildStepInput
# copied the energy entry onto it ("follows the energy entry in every tutorial") -- true of every tutorial
# and every fixture, which is why a case that separates the two was never seen. Both mirror arms.
#
# ARM A  div(phi,K) linearUpwind, div(phi,h) upwind   -> brae must match OpenFOAM
# ARM B  div(phi,h) linearUpwind, div(phi,K) upwind   -> brae must match OpenFOAM (the other direction)
# ARM C  div(phi,K) limitedLinear, div(phi,h) upwind  -> both arms must REFUSE, naming div(phi,Ekp|K)
# CONTROL: OpenFOAM's own answers for A, B and the base must differ from one another by at least 100x
#          the bound on some field, else the K scheme is inert on this case and A/B could pass with the
#          entry ignored. It is stated against the BOUND rather than as an absolute: on rhoBox the K
#          scheme moves OpenFOAM's T by only 9.8e-07 (a low-speed case, K << h) -- 300x the T bound, so
#          the arm discriminates, while the h scheme moves it by 5.1e-02.
#
# Measured on rhoBox at 200 iterations (converged; the scheme changes the fixed point so no transient
# trick is needed), both arms vs OpenFOAM: A p 2.9e-12 T 2.5e-12 (host) / 1.2e-11 (cuda); B T 3.0e-11
# (host) / 1.6e-10 (cuda). Fail-proof (schemeKE copied from the energy entry): A T 9.83e-07, B T
# 2.61e-06 on both arms -- the whole control gap -- and the limitedLinear-on-K case RUNS instead of
# refusing. Every fixture and stock tutorial names the K/Ekp entry explicitly under `default none`
# (census 2026-09-02), so the parse cannot newly refuse a case OpenFOAM runs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
SRC="$ROOT/validation/rhoBox"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=200
P_BOUND=1e-10; T_BOUND=3e-09; U_BOUND=1e-09; RHO_BOUND=3e-09   # the mirror gate's rhoBox bounds
CONTROL_RATIO=100   # the control must clear the bound by this factor on at least one field

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
say() { printf '  %-66s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # $1 dst  $2 div(phi,h) scheme  $3 div(phi,K) scheme
{
    HS="$2" KS="$3" python3 - "$SRC" "$1" "$N" <<'PYEOF'
import os, re, shutil, sys
src, dst, iters = sys.argv[1], sys.argv[2], sys.argv[3]
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
open(f, 'w').write(s)
f = os.path.join(dst, 'system/fvSchemes'); s = open(f).read()
s, n1 = re.subn(r'div\(phi,h\)\s+[^;]*;', 'div(phi,h) %s;' % os.environ['HS'], s)
s, n2 = re.subn(r'div\(phi,K\)\s+[^;]*;', 'div(phi,K) %s;' % os.environ['KS'], s)
assert n1 == 1 and n2 == 1, (n1, n2)
open(f, 'w').write(s)
PYEOF
}
UP='bounded Gauss upwind'; LUH='bounded Gauss linearUpwind grad(h)'; LUK='bounded Gauss linearUpwind grad(K)'
stage "$W/of_base" "$UP"  "$UP"
stage "$W/of_A"    "$UP"  "$LUK"
stage "$W/of_B"    "$LUH" "$UP"
for c in of_base of_A of_B; do
    ( cd "$W/$c" && rhoSimpleFoam > of.log 2>&1 ) || { tail -5 "$W/$c/of.log"; echo "FAIL: OpenFOAM did not run ($c)"; exit 1; }
done
for arm in 1 cuda; do
    stage "$W/A_$arm" "$UP"  "$LUK"
    stage "$W/B_$arm" "$LUH" "$UP"
    for c in A B; do
        ( cd "$W/${c}_$arm" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BIN" -case "$W/${c}_$arm" > run.log 2>&1 ) \
            || { tail -5 "$W/${c}_$arm/run.log"; echo "FAIL: the mirror (arm $arm) did not run arm $c"; exit 1; }
    done
done

W="$W" N="$N" P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" U_BOUND="$U_BOUND" RHO_BOUND="$RHO_BOUND" CONTROL_RATIO="$CONTROL_RATIO" \
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
bounds = {'p': 'P_BOUND', 'T': 'T_BOUND', 'U': 'U_BOUND', 'rho': 'RHO_BOUND'}
ratio = float(os.environ['CONTROL_RATIO'])
for a, b in (('of_A', 'of_base'), ('of_B', 'of_base'), ('of_A', 'of_B')):
    best = max((rel(a, b, f) / float(os.environ[k]), f) for f, k in bounds.items())
    good = best[0] > ratio
    print('     control: OpenFOAM %-7s vs %-7s differ by %.0fx the %s bound   %s'
          % (a, b, best[0], best[1], 'ok' if good else 'FAIL (inert)'))
    ok = ok and good
for arm in ('1', 'cuda'):
    for c in ('A', 'B'):
        for f, key in (('p', 'P_BOUND'), ('T', 'T_BOUND'), ('U', 'U_BOUND'), ('rho', 'RHO_BOUND')):
            r = rel('%s_%s' % (c, arm), 'of_' + c, f); bnd = float(os.environ[key]); good = r < bnd
            print('     arm %-4s %s %-4s vs OpenFOAM %.4e   (bound %.1e)   %s' % (arm, c, f, r, bnd, 'ok' if good else 'FAIL'))
            ok = ok and good
sys.exit(0 if ok else 1)
PYEOF
say "div(phi,K) is read from its own entry, both directions, both arms" "$([ $fail = 0 ] && echo ok || echo FAIL)"

# ARM C: an unported scheme on K alone must refuse BY NAME on both arms, with the energy entry ported.
stage "$W/C" "$UP" 'bounded Gauss limitedLinear 1'
for arm in 1 cuda; do
    out=$( cd "$W/C" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BIN" -case "$W/C" 2>&1 || true )
    echo "$out" | grep -q "div(phi,Ekp|K)" && ! [ -d "$W/C/$N" ] \
        && say "limitedLinear on div(phi,K) alone refuses by name (arm $arm)" ok \
        || { echo "$out" | tail -3; say "limitedLinear on div(phi,K) alone refuses by name (arm $arm)" FAIL; }
    rm -rf "$W/C"/[1-9]*
done
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
