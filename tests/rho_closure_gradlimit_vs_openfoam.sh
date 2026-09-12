#!/usr/bin/env bash
# THE CLOSURES TAKE THE CASE'S grad(U), grad(k) AND grad(omega|epsilon) SCHEMES -- cellLimited where
# fvSchemes says so -- on both arms (queue item 24, found under item 23 on naca0012).
#
# OpenFOAM's kEpsilon and kOmegaSST compute fvc::grad(U) for the production (kEpsilon.C:237,
# kOmegaSSTBase.C:522 and :132 in correctNut) through the gradSchemes entry `grad(U)`, and every corrected
# laplacian's non-orthogonal correction takes the field's OWN entry (correctedSnGrad.C:52-55,
# mesh.gradScheme("grad(" + vf.name() + ')')). brae's closures took plain Gauss gradients at all of those,
# so a case naming `grad(U) cellLimited Gauss linear 1` (the aerofoilNACA0012 tutorial does, for U, k and
# omega) ran an unlimited production and unlimited corrections: naca0012 read k 3.3e-04, omega 5.4e-03, nut
# 1.2e-03 against OpenFOAM at t = 1. The limiter coefficients travel on the coefficient structs
# (KEpsilonCoeffs / KOmegaSSTCoeffs gradULimitK, gradKLimitK), filled from the shared parse on both arms.
#
#   ARM KE   validation/rhoKE  mutated: grad(U), grad(k), grad(epsilon) cellLimited Gauss linear 1,
#            laplacian Gauss linear corrected -- host and device, t=1..10 vs real rhoSimpleFoam.
#   ARM SST  validation/rhoSST mutated: grad(U), grad(k), grad(omega) cellLimited Gauss linear 1,
#            laplacian Gauss linear corrected -- host (the device refuses kOmegaSST).
#   CONTROL  OpenFOAM's limited-gradient run must differ from its unlimited run by >= 100x the bound at
#            t=10, or the limiter never clipped on the fixture and the arm is inert.
# Every linear solver pinned to 1e-12 / 0 / 2000; residualControl emptied; from 0/.
# Measured floors (max over t=1..10, 2026-09-03): host k 8.7e-13, epsilon 2.0e-12, omega 7.1e-13, U 6.7e-13,
# p 2.9e-12, T 9.8e-13; device k 3.0e-12, epsilon 4.6e-12, U 7.8e-13. Bounds ~30x the larger arm's floor.
# OpenFOAM's limited-vs-unlimited control: 1.2e8x (kEpsilon), 1.8e9x (kOmegaSST) the bounds.
# Fail-proof: the closures' limiter coefficients forced to 0 (unlimited gradients) -> the arms FAIL.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-10}
K_BOUND=${K_BOUND:-1e-10}; E_BOUND=${E_BOUND:-2e-10}; W_BOUND=${W_BOUND:-3e-11}
U_BOUND=${U_BOUND:-3e-11}; P_BOUND=${P_BOUND:-1e-10}; T_BOUND=${T_BOUND:-3e-11}
CONTROL_RATIO=${CONTROL_RATIO:-100}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoKE" ] && [ -d "$ROOT/validation/rhoSST" ] || { echo "SKIP: fixtures missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-76s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # $1 dst  $2 fixture  $3 second scalar (epsilon|omega)  $4 limited: yes|no
{
    rm -rf "$1"; cp -r "$ROOT/validation/$2" "$1"; rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    SECOND="$3" LIMITED="$4" python3 - "$1" "$N" <<'CGSTAGE'
import os, re, sys
d, n = sys.argv[1], sys.argv[2]
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', n), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'maxIter\s+[0-9]+;', 'maxIter 2000;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
s = re.sub(r'relTol 0;(?![^}]*maxIter)', 'relTol 0; maxIter 2000;', s)
open(f, 'w').write(s)
if os.environ['LIMITED'] == 'yes':
    sc = os.path.join(d, 'system/fvSchemes'); s = open(sc).read()
    second = os.environ['SECOND']
    # the three closure gradients cellLimited, and corrected laplacians so the corrections read them too
    s, k = re.subn(r'gradSchemes\s*\{', 'gradSchemes\n{\n    grad(U) cellLimited Gauss linear 1;\n    grad(k) cellLimited Gauss linear 1;\n    grad(%s) cellLimited Gauss linear 1;' % second, s, count=1)
    assert k == 1, 'no gradSchemes block'
    s, k = re.subn(r'laplacianSchemes\s*\{[^}]*\}', 'laplacianSchemes\n{\n    default Gauss linear corrected;\n}', s, count=1)
    assert k == 1, 'no laplacianSchemes block'
    s, k = re.subn(r'snGradSchemes\s*\{[^}]*\}', 'snGradSchemes\n{\n    default corrected;\n}', s, count=1)
    assert k == 1, 'no snGradSchemes block'
    open(sc, 'w').write(s)
CGSTAGE
    ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh in $1"; exit 1; }
}
run()
{
    case "$2" in
        of) ( cd "$1" && rhoSimpleFoam > run.log 2>&1 ) ;;
        *)  ( cd "$1" && BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BIN" -case "$1" > run.log 2>&1 ) ;;
    esac || { tail -6 "$1/run.log"; echo "FAIL: $2 did not run in $1"; exit 1; }
    [ -d "$1/$N" ] || { echo "FAIL: $2 wrote no $N/ in $1"; exit 1; }
}
clone() { rm -rf "$2"; cp -r "$1" "$2"; rm -rf "$2"/[1-9]* "$2"/run.log; }

stage "$W/ke_of_lim"  rhoKE  epsilon yes; stage "$W/ke_of_unl"  rhoKE  epsilon no
stage "$W/sst_of_lim" rhoSST omega   yes; stage "$W/sst_of_unl" rhoSST omega   no
for c in ke_of_lim ke_of_unl sst_of_lim sst_of_unl; do run "$W/$c" of; done
clone "$W/ke_of_lim" "$W/ke_host";  run "$W/ke_host" 1
clone "$W/ke_of_lim" "$W/ke_cuda";  run "$W/ke_cuda" cuda
clone "$W/sst_of_lim" "$W/sst_host"; run "$W/sst_host" 1
grep -q "grad(U) cellLimited k=1" "$W/ke_host/run.log" && say "the mirror resolved grad(U) cellLimited 1 from the case" ok || say "the mirror resolved grad(U) cellLimited 1 from the case" FAIL

W="$W" N="$N" K_BOUND="$K_BOUND" E_BOUND="$E_BOUND" W_BOUND="$W_BOUND" U_BOUND="$U_BOUND" P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" \
CONTROL_RATIO="$CONTROL_RATIO" python3 - <<'CGCMP' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], int(os.environ['N']); ratio = float(os.environ['CONTROL_RATIO'])
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'internalField\s+uniform\s+\(?([^);]+)\)?;', s)
    return np.array([float(x) for x in u.group(1).split()])
def rel(a, b, t, f):
    x = read(os.path.join(W, a, str(t), f)); y = read(os.path.join(W, b, str(t), f))
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))
B = {'k': 'K_BOUND', 'epsilon': 'E_BOUND', 'omega': 'W_BOUND', 'U': 'U_BOUND', 'p': 'P_BOUND', 'T': 'T_BOUND'}
bounds = {f: float(os.environ[k]) for f, k in B.items()}
ok = True
for fx, fields in (('ke', ('k', 'epsilon', 'U', 'p', 'T')), ('sst', ('k', 'omega', 'U', 'p', 'T'))):
    best = max((rel('%s_of_lim' % fx, '%s_of_unl' % fx, N, f) / bounds[f], f) for f in fields)
    good = best[0] > ratio
    print('     control: OpenFOAM limited vs unlimited gradients (%-3s) %10.0fx the %s bound at t=%d   %s' % (fx, best[0], best[1], N, 'ok' if good else 'FAIL (limiter inert)'))
    ok = ok and good
for arm, ref, fields in (('ke_host', 'ke_of_lim', ('k', 'epsilon', 'U', 'p', 'T')), ('ke_cuda', 'ke_of_lim', ('k', 'epsilon', 'U', 'p', 'T')),
                        ('sst_host', 'sst_of_lim', ('k', 'omega', 'U', 'p', 'T'))):
    for f in fields:
        worst = max(rel(arm, ref, t, f) for t in range(1, N + 1))
        good = worst < bounds[f]
        print('     %-8s %-7s max over t=1..%d vs OpenFOAM %.4e   (bound %.1e)   %s' % (arm, f, N, worst, bounds[f], 'ok' if good else 'FAIL'))
        ok = ok and good
sys.exit(0 if ok else 1)
CGCMP
say "the closures' limited gradients match OpenFOAM on both arms" "$([ $fail = 0 ] && echo ok || echo FAIL)"
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
