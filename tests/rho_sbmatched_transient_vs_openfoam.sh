#!/usr/bin/env bash
# sbMatched TRANSIENT, both arms: kEpsilon with epsilonWallFunction on a NON-ORTHOGONAL mesh under a
# corrected laplacian, a flowRateInletVelocity inlet and an inletOutlet outlet on U, T, k and epsilon,
# t=1..3 from 0/ against real rhoSimpleFoam with the linear solvers pinned (1e-12, relTol 0).
#
# WHY. Every closure gate on a box mesh is blind to what a wall-function closure does near a
# non-orthogonal wall: the wall patch must take the fresh wall-cell value inside updateCoeffs, before
# the equation is assembled (epsilonWallFunctionFvPatchScalarField.C:168-175 `epf == scalarField(epsilon0,
# faceCells)`; the omega twin at omegaWallFunctionFvPatchScalarField.C:167-174), because the corrected
# laplacian's deferred correction takes grad(epsilon) at the wall cells -- and on an orthogonal mesh that
# correction is identically zero. Found on naca0012 (queue item 25); sbMatched is the fixture in the tree
# whose wall cells are non-orthogonal. Staging it transiently then found two more (item 26), both
# invisible to every converged gate and to the file-restart harnesses (which rebuild an iteration from
# OpenFOAM's WRITTEN state and so inherit its boundary values):
#
#   26a  flowRateInletVelocity::updateCoeffs runs again inside pEqn.H:100's U.correctBoundaryConditions,
#        against rho's PATCH value as it stands then. The mirror kept the assembly-time value: inlet U
#        523.09 against OpenFOAM's 488.17 at iteration 2, k 1.8e-02 at t=2.
#   26b  heRhoThermo::calculate() (heRhoThermo.C:102-142) KEEPS T's patch value on a fixesValue patch --
#        fixedValue and every mixed one, inletOutlet included (mixedFvPatchField.H:197) -- and builds
#        he_b, psi_b, rho_b, mu_b, alpha_b from it; T's mixed patch is evaluated from the cells only
#        inside the energy conditions' updateCoeffs, at the energy assembly. The mirror re-evaluated T's
#        outlet from the just-corrected cells: T_b 1000.82 K against OpenFOAM's 1000.00 at iteration 2,
#        rho_b 0.38203 vs 0.38235, phiHbyA on the outlet 8e-4 off with every internal face at 2e-11,
#        k 1.2e-2 in the outlet cells at t=2 (relL2 2.9e-05 over the domain).
#
#   27   The device arm substituted Jacobi for the case's DILU on every solve. With 26a and 26b in it
#        still sat two orders above the host (k 5.4e-09 against 8.4e-12) while the two arms' assembled k
#        and epsilon systems agreed to 1e-11 and each solution satisfied its OWN system to
#        sum|Ax-b|/sum|b| ~ 1e-12: the gap was where the solve STOPPED. fvSolution asks
#        `preconditioner DILU` on U, e, k and epsilon, the host reference has always run DILU
#        (pbicgstab.cuh), and Jacobi stalls about ten times short of it on this ill-conditioned k system
#        -- neither tolerance 1e-15 nor maxIter 40000 moves it, and a direct solve sits 3e-06 per entry
#        from BOTH answers. Worse, it was SILENT on U and the turbulence pair: linear_solver_setup.cuh
#        exempted those three from its substitution notice on the grounds that DILU was wired, which was
#        true of the legacy driver and not of this one. The device now takes the case's DILU on the
#        momentum, energy and turbulence solves, through the one level-scheduled DeviceDilu the driver
#        builds (device_dilu.cuh -- exact against a sequential sweep, not a parallel approximation of it).
#
# BOUNDS are per arm, ~30x the floors measured 2026-09-03 over t=1..3 with all three fixes in:
#   host    k 8.4e-12  epsilon 2.1e-11  U 6.7e-12  p 2.2e-11  T 2.6e-12
#   device  k 1.8e-11  epsilon 9.6e-11  U 7.2e-12  p 1.8e-11  T 2.6e-12
# k, epsilon, U and T are bit-reproducible run to run on both arms; p is not (four runs read 1.8e-11,
# 8.8e-12, 3.6e-12, 4.6e-12 -- the pressure solve's own nondeterminism), so its bound is taken off the
# worst seen. p is NOT preconditioned on the device arm and does not need to be: it fell from 1.2e-10 to
# 1.8e-11 on its inputs alone, and the reader still discloses the Jacobi substitution there.
#
# FAIL-PROOFS (measured, both arms, this gate): without 26a's post-correction recompute k 1.8414e-02,
# epsilon 3.5566e-02, U 4.8483e-04; with 26a but T's outlet re-evaluated in thermo.correct() (26b) k
# 2.8581e-05, U 1.4472e-04, p 2.8345e-05. Both sit 5 to 8 orders above the bounds.
#
# THE DILU CONTROL is an ARM of this gate rather than a recorded measurement, because item 27's fix is
# invisible to every other check here: BRAE_DILU=0 BRAE_DILU_KE=0 forces the device arm back onto Jacobi
# for the momentum and turbulence solves (the escape hatches linear_solver_setup.cuh already carries; the
# host arm is unaffected, its pbicgstab being DILU whatever the dict says), and that run must MISS the
# device k and U bounds. Measured with them set: k 5.3950e-09 (9.0x the bound), U 2.3670e-09 (11.8x).
# Without this arm the device bounds could be met by a build that ignored the case's preconditioner.
# NON-VACUITY: the fields at t=N must have moved from 0/ by >= 10x the bound, or a solver that did
# nothing would pass.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-3}
# host arm
K_BOUND=${K_BOUND:-3e-10}; E_BOUND=${E_BOUND:-6e-10}; U_BOUND=${U_BOUND:-2e-10}; P_BOUND=${P_BOUND:-7e-10}; T_BOUND=${T_BOUND:-8e-11}
# device arm (the same floors as the host now, epsilon apart -- see the header)
KC_BOUND=${KC_BOUND:-6e-10}; EC_BOUND=${EC_BOUND:-3e-9}; UC_BOUND=${UC_BOUND:-2e-10}; PC_BOUND=${PC_BOUND:-6e-10}; TC_BOUND=${TC_BOUND:-8e-11}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC/constant/polyMesh" ] || { echo "SKIP: sbMatched ships no mesh"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
cp -r "$SRC" "$W/of"; rm -rf "$W"/of/[1-9]* "$W"/of/0 "$W"/of/processor* "$W"/of/log.*; cp -r "$W/of/0.orig" "$W/of/0"
python3 - "$W/of" "$N" <<'SBSTAGE'
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
SBSTAGE
grep -q "corrected" "$W/of/system/fvSchemes" || { echo "FAIL: sbMatched no longer names a corrected laplacian"; exit 1; }
grep -q "epsilonWallFunction" "$W/of/0/epsilon" || { echo "FAIL: sbMatched lost its epsilon wall function"; exit 1; }
( cd "$W/of" && rhoSimpleFoam > run.log 2>&1 ) || { tail -5 "$W/of/run.log"; echo "FAIL: OpenFOAM did not run"; exit 1; }
for arm in 1 cuda; do
    rm -rf "$W/b$arm"; cp -r "$W/of" "$W/b$arm"; rm -rf "$W/b$arm"/[1-9]* "$W/b$arm/run.log"
    ( cd "$W/b$arm" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BIN" -case "$W/b$arm" > run.log 2>&1 ) || { tail -5 "$W/b$arm/run.log"; echo "FAIL: arm $arm did not run"; exit 1; }
    [ -d "$W/b$arm/$N" ] || { echo "FAIL: arm $arm wrote no $N/"; exit 1; }
done
# The DILU control arm -- see the header. The same device run with the case's preconditioner forced off.
rm -rf "$W/bjac"; cp -r "$W/of" "$W/bjac"; rm -rf "$W"/bjac/[1-9]* "$W/bjac/run.log"
( cd "$W/bjac" && BRAE_DILU=0 BRAE_DILU_KE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/bjac" > run.log 2>&1 ) \
    || { tail -5 "$W/bjac/run.log"; echo "FAIL: the Jacobi control arm did not run"; exit 1; }
[ -d "$W/bjac/$N" ] || { echo "FAIL: the Jacobi control arm wrote no $N/"; exit 1; }
W="$W" N="$N" K_BOUND="$K_BOUND" E_BOUND="$E_BOUND" U_BOUND="$U_BOUND" P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" \
KC_BOUND="$KC_BOUND" EC_BOUND="$EC_BOUND" UC_BOUND="$UC_BOUND" PC_BOUND="$PC_BOUND" TC_BOUND="$TC_BOUND" python3 - <<'SBCMP' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], int(os.environ['N'])
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'internalField\s+uniform\s+\(?([^);]+)\)?;', s)
    return np.array([float(x) for x in u.group(1).split()])
bounds = {'1':    {'k': float(os.environ['K_BOUND']),  'epsilon': float(os.environ['E_BOUND']),  'U': float(os.environ['U_BOUND']),  'p': float(os.environ['P_BOUND']),  'T': float(os.environ['T_BOUND'])},
          'cuda': {'k': float(os.environ['KC_BOUND']), 'epsilon': float(os.environ['EC_BOUND']), 'U': float(os.environ['UC_BOUND']), 'p': float(os.environ['PC_BOUND']), 'T': float(os.environ['TC_BOUND'])}}
ok = True
for arm in ('1', 'cuda'):
    for f, b in bounds[arm].items():
        worst = 0.0
        for t in range(1, N + 1):
            x = read(os.path.join(W, 'b' + arm, str(t), f)); y = read(os.path.join(W, 'of', str(t), f))
            worst = max(worst, float(np.linalg.norm(x - y) / np.linalg.norm(y)))
        good = worst < b
        print('     arm %-4s %-7s max over t=1..%d vs OpenFOAM %.4e   (bound %.1e)   %s' % (arm, f, N, worst, b, 'ok' if good else 'FAIL'))
        ok = ok and good
# THE DILU CONTROL: with the case's preconditioner forced off, the device arm must MISS the k and U
# bounds. Those are the two rows Jacobi moves by more than the bound (epsilon and T move less, and p is
# not preconditioned at all), so they are the ones asserted; the others are printed for the record.
for f in ('k', 'U'):
    b = bounds['cuda'][f]
    worst = 0.0
    for t in range(1, N + 1):
        x = read(os.path.join(W, 'bjac', str(t), f)); y = read(os.path.join(W, 'of', str(t), f))
        worst = max(worst, float(np.linalg.norm(x - y) / np.linalg.norm(y)))
    good = worst > b
    print('     control %-7s device forced onto Jacobi %.4e = %.1fx the bound   %s'
          % (f, worst, worst / b, 'ok' if good else 'FAIL (the bound does not need the preconditioner)'))
    ok = ok and good

# NON-VACUITY: OpenFOAM's t=N must sit >= 10x the (looser, device) bound away from the start state.
for f in bounds['cuda']:
    y = read(os.path.join(W, 'of', str(N), f)); s0 = read(os.path.join(W, 'of', '0', f))
    s0 = np.broadcast_to(s0, y.shape) if s0.shape != y.shape else s0
    r0 = float(np.linalg.norm(s0 - y) / np.linalg.norm(y))
    good = r0 >= 10 * bounds['cuda'][f]
    print('     control %-7s start state vs OpenFOAM t=%d %.3e (needs >= 10x the bound)   %s' % (f, N, r0, 'ok' if good else 'FAIL (vacuous)'))
    ok = ok and good
sys.exit(0 if ok else 1)
SBCMP
say "sbMatched tracks OpenFOAM transiently on both arms" "$([ $fail = 0 ] && echo ok || echo FAIL)"
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
