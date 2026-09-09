#!/usr/bin/env bash
# THE OF-MIRROR kOmegaSST CLOSURE ON THE DEVICE, AGAINST REAL OpenFOAM, AT THE ROUND-OFF FLOOR.
#
# The device closure (src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST.cu) is a separate
# lineage from the legacy deviceKOmegaSSTCorrect: it must agree with its OWN host reference, and that
# host reference sits on OpenFOAM at 1e-13. So does this one, now, and the bound says so.
#
# WHY THE LINEAR TOLERANCES ARE PINNED TO 1e-14 / relTol 0, AND WHY BRAE_U_SOLVER=ofOrder.
# validation/rhoSST ships `relTol 0.1` on U/h/k/omega, i.e. every solve stops after ONE decade of
# residual reduction. Two solvers that reach that gate stop at completely different iterates, and SIMPLE
# feeds the difference forward. Comparing the CUDA arm to OpenFOAM there measures the linear solver's
# stopping point and nothing else: the arm reads k 1.1e-02 at 20 iterations with a closure whose real
# error is 6e-13, and the ALREADY-SHIPPED device kEpsilon closure reads eps 1.3e-02 on the same mesh --
# worse -- while being exact at tight tolerance. A loose-tolerance bound cannot tell a correct closure
# from a broken one. BRAE_U_SOLVER=ofOrder is the same argument for the momentum equation, whose CUDA
# default is a multicolour Gauss-Seidel smoothSolver (rhoSimpleFoamDriver.cu, item 79): at relTol 0.1 it
# leaves a different-SHAPED error than the case's PBiCGStab, largest in the near-wall rows, which is
# exactly where the SST production lives. Both arms assert from their own log that no substitution ran.
#
#   ARM 1  rhoSST, device mirror vs OpenFOAM at t=1 and t=20      -> every field at the 1e-11 floor
#   ARM 2  CONTROL: the LEGACY device SST closure, same harness, same oracle, same bound -> must FAIL
#   ARM 3  rhoTI (computed turbulent inlets), device vs OpenFOAM  -> k/omega/nut inside 2e-3
#   ARM 4  rhoTI: k's and omega's INLET PATCH VALUES must equal OpenFOAM's -- the direct oracle for
#          turbulentIntensityKineticEnergyInlet and turbulentMixingLengthFrequencyInlet. At the floor at
#          t=1, where their inputs are still exact; at the field bound at t=N, where they cannot be
#          sharper than the U and k they are computed from.
#
# WHAT EACH ARM CAUGHT, measured in-session with the fix reverted (the fail-proofs):
#   wall G0 read a RECOMPUTED nutkWallFunction instead of the STORED nut patch value that
#     omegaWallFunctionFvPatchScalarField.C:199-200 reads -> rhoSST k 1.981e-06 at t=1 (now 2.788e-13),
#     confined to the 160 wall cells, sign-flipped between the hot and cold walls.
#   F1 on the boundary was the owner cell's F1 instead of the blender evaluated ON the patch face
#     -> rhoSST k 2.095e-08 at t=2 growing to 1.625e-07 at t=20, all of it at the inlet, decaying
#     downstream; zero at t=1 because a uniform start makes the two agree.
#   the turbulent inlets never refreshed -> rhoTI k 7.519e-02 at t=1, inlet frozen at the case file's
#     placeholder 0.1 against OpenFOAM's 9.63471.
#   the second-scalar inlet mask only ever matched mixingLengthEpsilon, so EVERY kOmegaSST case had an
#     empty omega mask -> rhoTI omega inlet frozen at 100 against OpenFOAM's 115.47.
#   ARM 2's control is live, not historical: the legacy closure reads k 7.5e-05 at t=1 and 2.7e-04 at
#     t=20 through this same harness, 1e7 x the bound.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
LEGACY="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam_legacy"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-20}
FLOOR=${FLOOR:-1e-11}          # rhoSST: device mirror vs OpenFOAM, every field
TI_BOUND=${TI_BOUND:-2e-3}     # rhoTI: k/omega/nut, the fixture with computed inlets
INLET_BOUND=${INLET_BOUND:-1e-12}
CONTROL_RATIO=${CONTROL_RATIO:-1e3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -x "$LEGACY" ]   || { echo "SKIP: $LEGACY not built"; exit 77; }
[ -d "$ROOT/validation/rhoSST" ] || { echo "SKIP: fixture rhoSST missing"; exit 77; }
[ -d "$ROOT/validation/rhoTI" ]  || { echo "SKIP: fixture rhoTI missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# One meshed base per fixture; every arm is a copy of it, with the linear tolerances pinned so the
# comparison is of the DISCRETISATION and not of where two Krylov solvers happened to stop.
for fx in rhoSST rhoTI; do
    cp -r "$ROOT/validation/$fx" "$W/base_$fx"
    rm -rf "$W/base_$fx"/[1-9]* "$W/base_$fx"/0
    cp -r "$W/base_$fx/0.orig" "$W/base_$fx/0"
    ( cd "$W/base_$fx" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh on $fx"; exit 1; }
done

stage()   # $1 fixture  $2 dst
{
    python3 - "$W/base_$1" "$2" "$N" <<'PYEOF'
import os, re, shutil, sys
src, dst, iters = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
c = os.path.join(dst, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', iters), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
s = re.sub(r'\bfunctions\s*\{.*\}\s*$', '', s, flags=re.S)
open(c, 'w').write(s)
f = os.path.join(dst, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
open(f, 'w').write(s)
PYEOF
}

for fx in rhoSST rhoTI; do
    stage "$fx" "$W/of_$fx"
    ( cd "$W/of_$fx" && rhoSimpleFoam > of.log 2>&1 ) \
        || { tail -5 "$W/of_$fx/of.log"; echo "FAIL: OpenFOAM did not run ($fx)"; exit 1; }
    stage "$fx" "$W/br_$fx"
    ( cd "$W/br_$fx" && BRAE_U_SOLVER=ofOrder BRAE_SST_DEVICE=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda \
        "$BIN" -case "$W/br_$fx" > run.log 2>&1 ) \
        || { tail -5 "$W/br_$fx/run.log"; echo "FAIL: the device mirror did not run ($fx)"; exit 1; }
done
stage rhoSST "$W/leg_rhoSST"
( cd "$W/leg_rhoSST" && BRAE_U_SOLVER=ofOrder "$LEGACY" -case "$W/leg_rhoSST" > run.log 2>&1 ) \
    || { tail -5 "$W/leg_rhoSST/run.log"; echo "FAIL: the legacy closure did not run"; exit 1; }

# --- the arms RAN what they claim to have run -----------------------------------------------------
# Without this the three arms silently collapse into one measurement: a refused SST run writes nothing
# and a substituted momentum solver turns the whole gate into a linear-solver comparison.
for fx in rhoSST rhoTI; do
    grep -q "OF-mirror, CUDA" "$W/br_$fx/run.log" && grep -q "kOmegaSST" "$W/br_$fx/run.log" \
        && say "$fx: the device OF-mirror arm reached the kOmegaSST closure" ok \
        || say "$fx: the device OF-mirror arm reached the kOmegaSST closure" FAIL
    grep -q "solvers/U solver" "$W/br_$fx/run.log" \
        && { grep -m1 "solvers/U solver" "$W/br_$fx/run.log"; say "$fx: no momentum-solver substitution on the device arm" FAIL; } \
        || say "$fx: no momentum-solver substitution on the device arm" ok
done
grep -q "solvers/U solver" "$W/leg_rhoSST/run.log" \
    && say "control arm: no momentum-solver substitution on the legacy arm" FAIL \
    || say "control arm: no momentum-solver substitution on the legacy arm" ok

W="$W" N="$N" FLOOR="$FLOOR" TI_BOUND="$TI_BOUND" INLET_BOUND="$INLET_BOUND" CONTROL_RATIO="$CONTROL_RATIO" \
python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], os.environ['N']
FLOOR, TI, INLET = float(os.environ['FLOOR']), float(os.environ['TI_BOUND']), float(os.environ['INLET_BOUND'])
RATIO = float(os.environ['CONTROL_RATIO'])

def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if not m:
        u = re.search(r'internalField\s+uniform\s+(\(.*?\)|[-+0-9.eE]+)\s*;', s)
        v = u.group(1)
        return np.array([float(x) for x in v.strip('()').split()]) if v.startswith('(') else np.array([float(v)])
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])

def patch(p, name):
    s = open(p).read()
    b = s[s.find('boundaryField'):]
    m = re.search(r'\n    %s\s*\n    \{(.*?)\n    \}' % name, b, re.S)
    if not m: return None
    v = re.search(r'value\s+nonuniform\s+List<scalar>\s*\n?\d+\s*\n?\(([^)]*)\)', m.group(1), re.S)
    if v: return np.array([float(x) for x in v.group(1).split()])
    v = re.search(r'value\s+uniform\s+([-+0-9.eE]+)', m.group(1))
    return np.array([float(v.group(1))]) if v else None

def rel(a, b, t, f):
    x, y = read(os.path.join(W, a, t, f)), read(os.path.join(W, b, t, f))
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))

ok = True
FIELDS = ('k', 'omega', 'nut', 'U', 'T', 'p')

# ARM 1 -- the device mirror sits on OpenFOAM at the round-off floor.
for t in ('1', N):
    for f in FIELDS:
        r = rel('br_rhoSST', 'of_rhoSST', t, f); good = r < FLOOR
        print('     rhoSST t=%-3s %-6s device mirror vs OpenFOAM %.4e   (bound %.1e)   %s'
              % (t, f, r, FLOOR, 'ok' if good else 'FAIL'))
        ok = ok and good

# ARM 2 -- CONTROL. A DIFFERENT closure through the same harness must fail the same bound, or the
# bound is not measuring the closure.
worst = max(rel('leg_rhoSST', 'of_rhoSST', t, f) / FLOOR for t in ('1', N) for f in ('k', 'omega', 'nut'))
good = worst > RATIO
print('     control: the LEGACY device closure misses the same bound by %.0e x        %s'
      % (worst, 'ok' if good else 'FAIL (the bound cannot see a closure change)'))
ok = ok and good

# ARM 3 -- the second fixture, whose inlets OpenFOAM recomputes every iteration.
for t in ('1', N):
    for f in ('k', 'omega', 'nut'):
        r = rel('br_rhoTI', 'of_rhoTI', t, f); good = r < TI
        print('     rhoTI  t=%-3s %-6s device mirror vs OpenFOAM %.4e   (bound %.1e)   %s'
              % (t, f, r, TI, 'ok' if good else 'FAIL'))
        ok = ok and good

# ARM 4 -- the inlets themselves, which is where a frozen BC shows up undiluted.
# The BOUND DIFFERS BY TIME on purpose. At t=1 every input to the inlet -- U for
# turbulentIntensityKineticEnergyInlet, k for turbulentMixingLengthFrequencyInlet -- is still exact, so
# the patch value is a direct oracle for the BC's own arithmetic and must agree at the floor. By t=N the
# inlet is a FUNCTION of fields that have themselves drifted (k 2.0e-05 on this fixture), so it cannot be
# tighter than they are; what it still catches there is a frozen inlet, which reads ~1e0 off.
for t, bound in (('1', INLET), (N, TI)):
    for f in ('k', 'omega'):
        a = patch(os.path.join(W, 'br_rhoTI', t, f), 'inlet')
        b = patch(os.path.join(W, 'of_rhoTI', t, f), 'inlet')
        if a is None or b is None:
            print('     rhoTI  t=%-3s %-6s inlet patch value NOT FOUND' % (t, f)); ok = False; continue
        if a.size == 1 and b.size > 1: a = np.full(b.shape, a[0])
        if b.size == 1 and a.size > 1: b = np.full(a.shape, b[0])
        r = float(np.linalg.norm(a - b) / np.linalg.norm(b)); good = r < bound
        print('     rhoTI  t=%-3s %-6s INLET patch value vs OpenFOAM %.4e   (bound %.1e)   %s'
              % (t, f, r, bound, 'ok' if good else 'FAIL'))
        ok = ok and good
sys.exit(0 if ok else 1)
PYEOF
say "the device kOmegaSST closure reproduces OpenFOAM on both fixtures" "$([ $fail = 0 ] && echo ok || echo FAIL)"
exit $fail
