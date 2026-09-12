#!/usr/bin/env bash
# p's BOUNDARY AFTER p.relax(), WITHOUT A PRESSURE LIMITER -- the blend OpenFOAM carries into the next
# momentum assembly, on both arms; and the freestream pressure's valueFraction refreshed where OpenFOAM
# refreshes it (queue item 23).
#
# GeometricField::relax is operator==(prevIter + alpha*(this - prevIter)) on BOTH halves (GeometricField.C:
# 1089-1094, :1420), after the solve's own correctBoundaryConditions (fvMatrixSolve.C:309). Nothing evaluates
# p's boundary again until the next pressure solve unless pressureControl.limit() or a closed volume runs
# pEqn.H:100-103's correctBoundaryConditions -- and every compressible fixture in this tree sets pMin/pMax or
# pMinFactor/pMaxFactor, so that recompute hid three deviations of the mirror:
#   1. the freestream pressure's valueFraction was rebuilt at the TOP of the iteration from the corrected U
#      and p's boundary re-evaluated there, where OpenFOAM rebuilds it inside the pressure fvMatrix
#      constructor (fvMatrix.C:396; freestreamPressureFvPatchScalarField.C:109-121, from U's patch value
#      as the momentum solve left it) and reads p's boundary AS IT STANDS in -fvc::grad(p);
#   2. the device re-derived p's boundary from the cells before the momentum gradient;
#   3. relaxBoundary re-evaluated the mixed family instead of storing the blend.
#
#   ARM A  validation/rhoBox, pMin removed        both arms   (zeroGradient walls, fixedValue in/outlet)
#   ARM B  validation/rhoTP,  pMin removed        both arms   (totalPressure inlet, no limiter recompute)
#   ARM C  validation/naca0012 as shipped         host        (freestreamPressure, pMinFactor/pMaxFactor)
#   ARM D  validation/naca0012, factors removed   host        (freestreamPressure, NO limiter). Real
#          OpenFOAM itself aborts on it at iteration 3 ("Negative initial temperature T0: -11.57"), so this
#          arm runs to N_UNL=2: the blend of the first p.relax() is what the second momentum assembly reads.
#   kOmegaSST is refused on the device arm, so C and D are host-only and the device's mixed-p path is
#   verified at the source only.
#   CONTROL  OpenFOAM's own limited and unlimited answers must differ (C vs D, and each fixture's
#            unlimited run vs its shipped run) by >= 100x the bound at t=10, or the limiter never bound
#            and the arm proves nothing about it.
#
# Every linear solver pinned to 1e-12 / 0 / 2000; residualControl emptied; from 0/; t = 1..10.
# Measured floors on rhoBox and rhoTP unlimited (max over t=1..10, both arms, 2026-09-03): p 3.0e-12,
# T 1.0e-12, U 1.6e-11, rho 2.6e-12. Bounds ~30x.
#
# THE NACA ARMS ARE ASSERTED, under the SAME bounds as the box and totalPressure ones -- item 25 closed
# 2026-09-03. They were reported-only through items 23, 24, 26, 27 and 28, reading p 1.8e-05 / U 8.9e-06
# at worst and still p 1.1e-09 / U 9.9e-10 after 26. What closed them was two things OpenFOAM does that
# brae did not: gaussGrad's boundary correction asks the PATCH for snGrad() rather than using the base
# class's (value - patchInternalField)*deltaCoeffs, and updateCoeffs is NOT an evaluate -- OF's
# fvPatchField::updateCoeffs sets a flag and the fvMatrix constructor calls nothing else (fvMatrix.C:396),
# so a mixed patch assembles holding the NEW valueFraction beside the value its last evaluate left. brae
# evaluated U's boundary before the assembly and collapsed that lag: the freestreamVelocity inlet read
# 3.2e-06 from OpenFOAM's at iteration 2 while every internal input was exact to 1e-14. Measured now,
# t=1..10 limited: p 1.8e-11, T 1.1e-12, U 1.2e-12, rho 2.8e-12 -- the write-precision floor this fixture
# writes at, and the same floor the other two arms sit on.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-10}
N_UNL=${N_UNL:-2}   # the unlimited NACA horizon (see ARM D)
P_BOUND=${P_BOUND:-1e-10}; T_BOUND=${T_BOUND:-3e-11}; U_BOUND=${U_BOUND:-5e-10}; RHO_BOUND=${RHO_BOUND:-1e-10}
CONTROL_RATIO=${CONTROL_RATIO:-100}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
for fx in rhoBox rhoTP naca0012; do [ -d "$ROOT/validation/$fx" ] || { echo "SKIP: fixture $fx missing"; exit 77; }; done
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
[ -f "$FOAM_TUTORIALS/resources/geometry/NACA0012.obj.gz" ] || { echo "SKIP: the NACA0012 geometry is not in this OpenFOAM install"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

mesh()   # $1 dst (a fixture copy with 0/ in place)
{
    if [ "$(basename "$1" | cut -d_ -f1)" = naca ]; then
        ( cd "$1" && mkdir -p constant/geometry && cp -f "$FOAM_TUTORIALS/resources/geometry/NACA0012.obj.gz" constant/geometry/ \
            && blockMesh > log.blockMesh 2>&1 && transformPoints -scale '(1 0 1)' > log.transformPoints 2>&1 \
            && extrudeMesh > log.extrudeMesh 2>&1 && topoSet > log.topoSet 2>&1 ) || { echo "FAIL: naca mesh"; exit 1; }
        # The tutorial's limitTemperature fvOptions: brae refuses an fvOptions file rather than dropping the
        # constraint, so the gate removes it on both sides, as naca_vs_openfoam does.
        rm -f "$1/system/fvOptions"
    else
        ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh in $1"; exit 1; }
    fi
}
stage()   # $1 dst  $2 fixture  $3 limiter: keep|drop  $4 endTime (default N)
{
    local src="$ROOT/validation/$2"
    rm -rf "$1"; cp -r "$src" "$1"; rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    LIM="$3" python3 - "$1" "${4:-$N}" <<'PUSTAGE'
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
if os.environ['LIM'] == 'drop':
    # rhoMin/rhoMax too: pressureControl infers pMax/pMin from them for backward compatibility
    # (pressureControl.C:55-125 `else if (dict.found("rhoMax"))`), and brae mirrors that inference.
    s, k = re.subn(r'\b(pMin|pMax|pMinFactor|pMaxFactor|rhoMin|rhoMax)\s+[^;]*;', '', s)
    assert k >= 1, 'the fixture names no pressure limiter to drop'
open(f, 'w').write(s)
PUSTAGE
    mesh "$1"
}
run()   # $1 dir  $2 of|1|cuda  $3 last time expected (default N)
{
    local last="${3:-$N}"
    case "$2" in
        of) ( cd "$1" && rhoSimpleFoam > run.log 2>&1 ) ;;
        *)  ( cd "$1" && BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BIN" -case "$1" > run.log 2>&1 ) ;;
    esac || { tail -6 "$1/run.log"; echo "FAIL: $2 did not run in $1"; exit 1; }
    [ -d "$1/$last" ] || { echo "FAIL: $2 wrote no $last/ in $1"; exit 1; }
}
clone() { rm -rf "$2"; cp -r "$1" "$2"; rm -rf "$2"/[1-9]* "$2"/run.log; }

# OpenFOAM: the shipped (limited) and the unlimited form of each fixture.
stage "$W/box_of_lim"  rhoBox    keep; stage "$W/box_of_unl"  rhoBox    drop
stage "$W/tp_of_lim"   rhoTP     keep; stage "$W/tp_of_unl"   rhoTP     drop
stage "$W/naca_of_lim" naca0012  keep; stage "$W/naca_of_unl" naca0012  drop "$N_UNL"
for c in box_of_lim box_of_unl tp_of_lim tp_of_unl naca_of_lim; do run "$W/$c" of; done
run "$W/naca_of_unl" of "$N_UNL"
grep -q "^pressureControl" "$W/box_of_lim/run.log"   && say "OpenFOAM's shipped rhoBox reports a pressureControl (the limiter is real)" ok || say "OpenFOAM's shipped rhoBox reports a pressureControl (the limiter is real)" FAIL
grep -q "^pressureControl" "$W/box_of_unl/run.log"   && say "OpenFOAM's unlimited rhoBox reports no pressureControl" FAIL || say "OpenFOAM's unlimited rhoBox reports no pressureControl" ok
grep -q "^pressureControl" "$W/tp_of_unl/run.log"    && say "OpenFOAM's unlimited rhoTP reports no pressureControl" FAIL || say "OpenFOAM's unlimited rhoTP reports no pressureControl" ok
grep -q "^pressureControl" "$W/naca_of_unl/run.log"  && say "OpenFOAM's unlimited naca0012 reports no pressureControl" FAIL || say "OpenFOAM's unlimited naca0012 reports no pressureControl" ok
# brae: arms A and B on both arms, C and D on the host.
for c in box_of_unl tp_of_unl; do
    clone "$W/$c" "$W/${c/of/host}"; run "$W/${c/of/host}" 1
    clone "$W/$c" "$W/${c/of/cuda}"; run "$W/${c/of/cuda}" cuda
done
clone "$W/naca_of_lim" "$W/naca_host_lim"; run "$W/naca_host_lim" 1
clone "$W/naca_of_unl" "$W/naca_host_unl"; run "$W/naca_host_unl" 1 "$N_UNL"

W="$W" N="$N" N_UNL="$N_UNL" P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" U_BOUND="$U_BOUND" RHO_BOUND="$RHO_BOUND" CONTROL_RATIO="$CONTROL_RATIO" \
python3 - <<'PUCMP' || fail=1
import os, re, sys
import numpy as np
W, N, N_UNL = os.environ['W'], int(os.environ['N']), int(os.environ['N_UNL']); ratio = float(os.environ['CONTROL_RATIO'])
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
bounds = {f: float(os.environ[k]) for f, k in (('p', 'P_BOUND'), ('T', 'T_BOUND'), ('U', 'U_BOUND'), ('rho', 'RHO_BOUND'))}
ok = True
# CONTROLS: OpenFOAM's limited and unlimited answers differ, so the unlimited arms test a path the limiter hid.
# rhoBox: every p patch is zeroGradient or fixedValue, where the post-relax re-evaluation and the blend are
# the same arithmetic on the same operands, so OpenFOAM's limited and unlimited runs must be IDENTICAL --
# an OpenFOAM-side check of the argument that lets those classes be re-derived. rhoTP (totalPressure,
# recomputed under the limiter) and naca0012 (freestreamPressure) must differ, or the arm proves nothing.
ident = max(rel('box_of_unl', 'box_of_lim', t, f) for t in range(1, N + 1) for f in bounds)
print('     control: OpenFOAM unlimited == limited on rhoBox (zeroGradient/fixedValue only) max relL2 %.3e   %s' % (ident, 'ok' if ident < 1e-14 else 'FAIL'))
ok = ok and ident < 1e-14
for fx in ('tp', 'naca'):
    tn = N_UNL if fx == 'naca' else N
    best = max((rel('%s_of_unl' % fx, '%s_of_lim' % fx, tn, f) / bounds[f], f) for f in bounds)
    good = best[0] > ratio
    print('     control: OpenFOAM unlimited vs limited (%-4s) %10.0fx the %s bound at t=%d   %s' % (fx, best[0], best[1], tn, 'ok' if good else 'FAIL (limiter never bound)'))
    ok = ok and good
for arm, ref in (('box_host_unl', 'box_of_unl'), ('box_cuda_unl', 'box_of_unl'), ('tp_host_unl', 'tp_of_unl'), ('tp_cuda_unl', 'tp_of_unl'),
                 ('naca_host_lim', 'naca_of_lim'), ('naca_host_unl', 'naca_of_unl')):
    tn = N_UNL if arm == 'naca_host_unl' else N
    for f, b in bounds.items():
        worst = max(rel(arm, ref, t, f) for t in range(1, tn + 1))
        good = worst < b
        print('     %-14s %-4s max over t=1..%d vs OpenFOAM %.4e   (bound %.1e)   %s' % (arm, f, tn, worst, b, 'ok' if good else 'FAIL'))
        ok = ok and good
sys.exit(0 if ok else 1)
PUCMP
say "p's relaxed boundary and the freestream pressure's valueFraction track OpenFOAM without a limiter" "$([ $fail = 0 ] && echo ok || echo FAIL)"
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
