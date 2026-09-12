#!/usr/bin/env bash
# THE totalPressure + pressureInletOutletVelocity INLET TRACKS OPENFOAM TRANSIENTLY, on both mirror arms --
# validation/rhoTP at t=1..N from 0/, against real rhoSimpleFoam's written output at every t.
#
# The converged tp_vs_openfoam gate is blind to this: at t=1 (2026-09-02, solvers pinned) the host arm read
# U 2.15e-02 / p 1.75e-04 relL2 and the device U 2.33e-01 / p 4.81e-04 against OpenFOAM, and BOTH arms
# wrote the inlet at its (5 0 0) seed with p = p0 - 0.5*rho*|5|^2 = 100185, where OpenFOAM writes
# (13.44 0 0) and 100095. What OpenFOAM does, at the source:
#   * pressureInletOutletVelocity::updateCoeffs sets valueFraction = neg(phi)*(I - nn) and then calls
#     directionMixed::evaluate ITSELF (pressureInletOutletVelocityFvPatchVectorField.C:180-183), so the
#     patch value is n(n & U_cell) at the momentum assembly (fvMatrix.C:396), after the solve
#     (fvMatrixSolve.C:242) and after the velocity correction (pEqn.H:87) -- the last with the NEW phi.
#   * totalPressure::updateCoeffs (totalPressureFvPatchScalarField.C:152-225) runs inside the PRESSURE
#     equation's constructor, from U's post-solve patch value, and again after the limiter
#     (pEqn.H:100-103, keyed on pressureControl::limit's `limitMaxP_ || limitMinP_`; rhoTP sets pMin).
#   * p.relax() relaxes the BOUNDARY too (GeometricField.C:1094, :1420); the velocity corrector's grad(p)
#     reads that blend.
#   * directionMixed is a transform patch: valueInternalCoeffs = 1 - sqrt|vf_kk| (transformFvPatchField.C:95-100,
#     directionMixedFvPatchField.C:180-200), i.e. the tangential components are fixedValue 0 on an inflow
#     face while the normal one is zeroGradient. brae's host class extrapolated every component and the
#     device typed every inflow component fixedValue at n(n.U_cell).
#
#   ARM 1  p, T, U, rho internal fields, host and device against OpenFOAM at every t=1..N (relL2).
#   ARM 2  the INLET patch values of U and p at t=1 and t=N, parsed from the written boundaryField by brace
#          matching -- the direct oracle for the two patches.
#   CONTROLS  the start state 0/ misses every field bound by >= CONTROL_RATIO x against OpenFOAM at t=N
#             (non-vacuity), and the SEEDED inlet (5 0 0) / 100200 misses the inlet bounds by the same
#             factor against OpenFOAM's written inlet at t=1 -- the arm can tell the seed from the projection.
#
# Before the fix (2026-09-02, same staging): host U 2.15e-02 / p 1.75e-04 / T 5.26e-05, device U 2.33e-01 /
# p 4.81e-04 at t=1; the inlet U 6.28e-01 (the seed) on both arms. After the VALUE steps alone (the patch
# values OpenFOAM's, the coefficients still extrapolated on the host) U 9.79e-03 / p 6.43e-05 at t=1 and the
# inlet's normal velocity 13.13 against 13.44: the rest was the matrix side.
# Fail-proof (2026-09-03): with the post-solve piov recompute removed on the host, the gate exits 1: host t=1 U 1.56e-02 / p 2.62e-04 /
# T 5.54e-05 / rho 2.94e-04 (max over t=1..10 U 6.57e-02), the inlet U 5.33e-03 at t=1 and 2.65e-02 at t=10, the
# inlet p 1.12e-05 / 3.24e-05; the device arm stays at its floor. Restored: PASS.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
SRC="$ROOT/validation/rhoTP"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-10}
# Floors measured 2026-09-03 (max over t=1..10, against OpenFOAM): host p 2.9e-12, T 1.0e-12, U 3.7e-12,
# rho 2.6e-12; device p 2.9e-12, T 1.0e-12, U 3.9e-12, rho 2.6e-12; the inlet U and p at t=1 and t=10 3.1e-12
# on both arms. Bounds ~30x, the same for the two arms -- they sit on the same floor here.
HOST_P_BOUND=${HOST_P_BOUND:-1e-10}; HOST_T_BOUND=${HOST_T_BOUND:-3e-11}
HOST_U_BOUND=${HOST_U_BOUND:-1.2e-10}; HOST_RHO_BOUND=${HOST_RHO_BOUND:-8e-11}
CUDA_P_BOUND=${CUDA_P_BOUND:-1e-10}; CUDA_T_BOUND=${CUDA_T_BOUND:-3e-11}
CUDA_U_BOUND=${CUDA_U_BOUND:-1.2e-10}; CUDA_RHO_BOUND=${CUDA_RHO_BOUND:-8e-11}
INLET_U_BOUND=${INLET_U_BOUND:-1e-10}; INLET_P_BOUND=${INLET_P_BOUND:-1e-10}
CONTROL_RATIO=${CONTROL_RATIO:-10}

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
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

cp -r "$SRC" "$W/base"; rm -rf "$W"/base/[1-9]* "$W"/base/0 "$W"/base/log.*
cp -r "$W/base/0.orig" "$W/base/0"

stage()   # $1 dst
{
    python3 - "$W/base" "$1" "$N" <<'TPSTAGE'
import os, re, shutil, sys
src, dst, n = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
c = os.path.join(dst, 'system/controlDict'); s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', n), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(dst, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0; maxIter 2000;', s)
open(f, 'w').write(s)
TPSTAGE
}

run()   # $1 dir  $2 arm (of|1|cuda)
{
    case "$2" in
        of)   ( cd "$1" && rhoSimpleFoam > run.log 2>&1 ) ;;
        *)    ( cd "$1" && BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BIN" -case "$1" > run.log 2>&1 ) ;;
    esac || { tail -5 "$1/run.log"; echo "FAIL: $2 did not run in $1"; exit 1; }
}

stage "$W/of"; stage "$W/host"; stage "$W/cuda"
run "$W/of" of; run "$W/host" 1; run "$W/cuda" cuda

W="$W" N="$N" CONTROL_RATIO="$CONTROL_RATIO" \
HOST_P_BOUND="$HOST_P_BOUND" HOST_T_BOUND="$HOST_T_BOUND" HOST_U_BOUND="$HOST_U_BOUND" HOST_RHO_BOUND="$HOST_RHO_BOUND" \
CUDA_P_BOUND="$CUDA_P_BOUND" CUDA_T_BOUND="$CUDA_T_BOUND" CUDA_U_BOUND="$CUDA_U_BOUND" CUDA_RHO_BOUND="$CUDA_RHO_BOUND" \
INLET_U_BOUND="$INLET_U_BOUND" INLET_P_BOUND="$INLET_P_BOUND" \
python3 - <<'TPCMP' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], int(os.environ['N'])
ratio = float(os.environ['CONTROL_RATIO'])
bounds = {arm: {f: float(os.environ['%s_%s_BOUND' % (arm.upper(), f.upper())]) for f in ('p', 'T', 'U', 'rho')}
          for arm in ('host', 'cuda')}
inlet_bounds = {'U': float(os.environ['INLET_U_BOUND']), 'p': float(os.environ['INLET_P_BOUND'])}
NFACES = 20   # rhoTP's inlet, constant/polyMesh/boundary

def read_internal(path):
    s = open(path).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'internalField\s+uniform\s+\(?([^);]+)\)?;', s)
    return None if not u else np.array([float(x) for x in u.group(1).split()])

def patch_block(s, name):
    # The brace-matched block of patch `name` inside boundaryField: OpenFOAM nests sub-dictionaries in a
    # patch block (a uniformTotalPressure writes `p0 table; p0Coeffs { ... }`), so a regex to the next
    # closing brace is not enough.
    i = s.find('boundaryField')
    j = re.search(r'\n\s*' + re.escape(name) + r'\s*\n?\s*\{', s[i:])
    if not j: return None
    k = i + j.end() - 1
    depth, p = 0, k
    while p < len(s):
        if s[p] == '{': depth += 1
        elif s[p] == '}':
            depth -= 1
            if depth == 0: return s[k:p + 1]
        p += 1
    return None

def patch_value(path, name):
    blk = patch_block(open(path).read(), name)
    if blk is None: return None
    m = re.search(r'\bvalue\s+nonuniform\s+List<(scalar|vector)>\s*(\d+)\s*\((.*?)\)\s*;', blk, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'\bvalue\s+uniform\s+\(?([^();]+)\)?\s*;', blk)
    if u:
        v = np.array([float(x) for x in u.group(1).split()])
        return np.tile(v, (NFACES, 1)) if v.size > 1 else np.full(NFACES, v[0])
    return None

def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

ok = True
# ---- arm 1: the fields at every t ----
worst = {arm: {f: 0.0 for f in ('p', 'T', 'U', 'rho')} for arm in ('host', 'cuda')}
for t in range(1, N + 1):
    row = '     t=%2d' % t
    for f in ('p', 'T', 'U', 'rho'):
        of = read_internal(os.path.join(W, 'of', str(t), f))
        for arm in ('host', 'cuda'):
            a = read_internal(os.path.join(W, arm, str(t), f))
            if of is None or a is None or a.shape != of.shape:
                row += '  %s/%s MISSING' % (f, arm[0]); ok = False; continue
            r = rel(a, of)
            worst[arm][f] = max(worst[arm][f], r)
            row += '  %s/%s %.2e%s' % (f, arm[0], r, '' if r < bounds[arm][f] else ' FAIL')
            ok = ok and r < bounds[arm][f]
    print(row)
for arm in ('host', 'cuda'):
    print('     %s max over t=1..%d: %s' % (arm, N, '  '.join('%s %.3e (bound %.1e)' % (f, worst[arm][f], bounds[arm][f])
                                                                for f in ('p', 'T', 'U', 'rho'))))
# CONTROL: the start state must miss the field bounds by >= ratio x, or a solver that did nothing would pass.
of_end = {f: read_internal(os.path.join(W, 'of', str(N), f)) for f in ('p', 'T', 'U', 'rho')}
def start_state(f):
    if f != 'rho':
        return read_internal(os.path.join(W, 'of', '0', f))
    # 0/ carries no rho: createFields builds it as thermo.rho() = p/(R T) (perfectGas), which is the
    # state a solver that did nothing would write. Reconstructed from 0/p, 0/T and the case's molWeight.
    p0, T0 = read_internal(os.path.join(W, 'of', '0', 'p')), read_internal(os.path.join(W, 'of', '0', 'T'))
    mw = float(re.search(r'molWeight\s+([0-9.eE+-]+)', open(os.path.join(W, 'of', 'constant/thermophysicalProperties')).read()).group(1))
    return p0 / (8314.47 / mw * T0)
for f in ('p', 'T', 'U', 'rho'):
    s0 = start_state(f)
    if s0 is None or of_end[f] is None: continue
    if s0.ndim < of_end[f].ndim or s0.shape[0] != of_end[f].shape[0]:
        s0 = np.tile(s0, (of_end[f].shape[0], 1)) if s0.ndim < of_end[f].ndim else s0
    r0 = rel(s0, of_end[f])
    b = max(bounds['host'][f], bounds['cuda'][f])
    good = r0 >= ratio * b
    print('     control: 0/ %-3s vs OpenFOAM at t=%d  %.3e  (>= %gx the bound %.1e)   %s' % (f, N, r0, ratio, b, 'ok' if good else 'FAIL (vacuous)'))
    ok = ok and good

# ---- arm 2: the inlet patch values ----
for t in (1, N):
    for f in ('U', 'p'):
        of = patch_value(os.path.join(W, 'of', str(t), f), 'inlet')
        if of is None:
            print('     inlet %s at t=%d: OpenFOAM patch block not parsed   FAIL' % (f, t)); ok = False; continue
        for arm in ('host', 'cuda'):
            a = patch_value(os.path.join(W, arm, str(t), f), 'inlet')
            if a is None or a.shape != of.shape:
                print('     inlet %s at t=%d: %s patch block not parsed   FAIL' % (f, t, arm)); ok = False; continue
            r = rel(a, of)
            good = r < inlet_bounds[f]
            mean = np.array2string(of.mean(axis=0), precision=4) if of.ndim > 1 else '%.4f' % of.mean()
            print('     inlet %s at t=%2d  %s vs OpenFOAM %.3e  (OF mean %s, bound %.1e)   %s' % (f, t, arm, r, mean, inlet_bounds[f], 'ok' if good else 'FAIL'))
            ok = ok and good
# CONTROL: the seed the two arms used to write must miss the inlet bounds by >= ratio x.
for f in ('U', 'p'):
    seed = patch_value(os.path.join(W, 'of', '0', f), 'inlet')
    of1 = patch_value(os.path.join(W, 'of', '1', f), 'inlet')
    if seed is None or of1 is None or seed.shape != of1.shape:
        print('     control: seeded inlet %s not parsed   FAIL' % f); ok = False; continue
    r0 = rel(seed, of1)
    good = r0 >= ratio * inlet_bounds[f]
    print('     control: seeded inlet %s vs OpenFOAM at t=1  %.3e  (>= %gx the bound %.1e)   %s' % (f, r0, ratio, inlet_bounds[f], 'ok' if good else 'FAIL (vacuous)'))
    ok = ok and good
sys.exit(0 if ok else 1)
TPCMP
say "arm 1+2: rhoTP fields at t=1..$N and the inlet patch values, both arms vs OpenFOAM" "$([ $fail = 0 ] && echo ok || echo FAIL)"

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
