#!/usr/bin/env bash
# gasMixing/injectorPipe -- OpenFOAM's own tutorial, as shipped -- on the rhoSimpleFoam host arm against
# real OpenFOAM v2412, at a DEVELOPED state, iteration by iteration. This is the case the leastSquares
# work was for, and the one that found four defects the orthogonal fixtures could not see. It is the only
# rhoSimpleFoam tutorial with: `ddtSchemes default Euler` (the closures' fvm::ddt is live, kEpsilon.C:254),
# `gradSchemes default leastSquares` beside `grad(U) cellLimited Gauss linear 0.99`, `div(phi,U) Gauss
# limitedLinearV 1`, a snappyHexMesh mesh whose boundary faces are skewed, and `laplacianSchemes Gauss
# linear corrected` on that mesh.
#
# THE PROTOCOL: BOTH CODES RESTART from OpenFOAM's OWN written iteration 5. Two reasons, each measured.
#   1. From rest the case starts with k `uniform 6`, epsilon `uniform 100` and T uniform, and every
#      limitedLinear limiter sits in NVDTVD's 0/0 branch, where r is decided by the sign of round-off:
#      the codes part at iteration 2 (T 9.85e-05, k 9.6e-06) on nobody's defect. Iteration 1 from rest is
#      exact (every field <= 2.8e-12) and tests/rho_tutorials_vs_openfoam.sh asserts that one.
#   2. Under `Euler`, OpenFOAM's OWN continuous run and its OWN restart differ at the first restarted
#      iteration by k 3.8e-07 / epsilon 1.7e-07 / nut 3.7e-07 and nothing in U, p or T: GeometricField::
#      oldTime() creates the _0 field as a copy at FIRST USE, so on a fresh process rho.oldTime() inside
#      the closure's ddt is the closure-time rho, while in a continuous run it is the start-of-step rho
#      stored at the pressure tail. brae reproduces both (StepInput::firstIteration): restarted brae vs
#      restarted OpenFOAM 2.2e-12, brae under the continuous rule vs continuous OpenFOAM 2.2e-12, and each
#      against the other 3.8e-07. A gate that restarted brae against a continuous OpenFOAM would be
#      asserting OpenFOAM's restart non-reproducibility, so OpenFOAM is restarted too.
#
# MEASURED, host arm, iterations 6-8 under that protocol, worst over all fields:
#     it 6  U 1.7e-12  p 2.7e-12  T 9.6e-13  k 2.2e-12  epsilon 1.3e-12  nut 1.8e-12
#     it 8  U 2.0e-11  p 2.9e-12  T 9.8e-13  k 3.5e-12  epsilon 2.2e-12  nut 1.2e-11
# ...and the CUDA arm, same protocol:
#     it 6  U 1.7e-12  p 2.7e-12  T 9.6e-13  k 1.7e-12  epsilon 1.2e-12  nut 1.5e-12
#     it 8  U 1.2e-11  p 2.9e-12  T 9.7e-13  k 2.6e-12  epsilon 1.7e-12  nut 9.1e-12
# WHAT THE CUDA ARM FOUND ON THE WAY (each named by the FIRST divergent stage of iteration 6, both arms
# dumped with BRAE_STAGE_DUMP_DIR, then split by patch):
#     momentum off-diagonals 3.3e-03: limitedLinearV's limiter read the boundary AFTER updateCoeffs, where
#         the two flowRateInletVelocity inlets had just rewritten it (RhoMomentumInput::UxPreUpdateBnd)
#     closure off-diagonals 1.6e-04, 100% on outlet-adjacent faces: the device seeded an inletOutlet face
#         as fixedValue(inletValue) at construction, so its first evaluate was not OpenFOAM's stored value
#         (DeviceBoundary::ioStored); U 7.0e-04 at iteration 6 before the two, 1.7e-12 after
#
# FAIL-PROOFS, each the same protocol with ONE fix reverted (recorded in the files that carry them):
#     fvPatch::delta() raw instead of projected (fvc.cu)        U 2.43e-02 at iteration 1 from rest
#     limitedLinearV off the post-updateCoeffs boundary          U 9.35e-04 at iteration 8
#     energy non-orth correction off a hardcoded Gauss gradient  T 8.58e-07 at iteration 6
#     no fvm::ddt in the closure (steadyState assumed)           k 4.69e-05, epsilon 4.55e-05 at iteration 6
#
# BOTH ARMS, under the same protocol. The CUDA arm reached this case one module at a time, each gated
# against the host reference and OpenFOAM on its own: the closures' fvm::ddt under Euler
# (rho_kepsilon_cuda, rho_step_cuda_euler), grad(k)/grad(epsilon) leastSquares
# (rho_leastsquares_closure_vs_openfoam), grad(p) leastSquares at its five consumers
# (rho_step_cuda_lsq, rho_gradp_lsq_simplec_vs_openfoam) and the energy correction's gradient scheme
# (this gate: T is the field that sees it).
#
# WHAT THIS GATE DOES NOT CLAIM: three iterations from a developed state, not convergence -- the linear
# solvers are tightened to 1e-14 relTol 0 so the codes do not part on solver noise; schemes, boundary
# conditions, thermo and fvSolution are otherwise the tutorial's own. The tutorial's function objects
# (scalarTransport of tracer0) are removed; brae does not run them.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
DEV=5                     # OpenFOAM's own iteration both codes restart from
LAST=8
BOUND=${BOUND:-1e-10}     # measured: worst 2.0e-11 (U at iteration 8)
CUDABOUND=${CUDABOUND:-1e-10}   # measured: worst 1.2e-11 (U at iteration 8), see the header

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/gasMixing/injectorPipe"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

# the tutorial's OWN meshing; snappyHexMesh runs in parallel and is reconstructed to serial
cp -r "$SRC" "$W/mesh" || exit 1
( cd "$W/mesh" && ./Allrun.pre > log.pre 2>&1 ) || { echo "SKIP: the tutorial's own meshing did not run here"; exit 77; }
if [ -d "$W/mesh/processor0/constant/polyMesh" ]; then
    ( cd "$W/mesh" && reconstructParMesh -constant > log.rpm 2>&1 && rm -rf processor* )
fi
[ -f "$W/mesh/constant/polyMesh/owner" ] || { echo "SKIP: no mesh produced"; exit 77; }

stage() {   # stage <dest> <endTime>
    rm -rf "$1"; cp -r "$W/mesh" "$1"
    rm -rf "$1"/0 "$1"/[1-9]* "$1"/processor* "$1"/log.* "$1"/postProcessing "$1"/dynamicCode
    cp -r "$1/0.orig" "$1/0"
    N="$2" python3 - "$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
p = os.path.join(d, 'system/controlDict'); s = open(p).read()
for k, v in (('startFrom','latestTime'), ('endTime',os.environ['N']), ('writeInterval','1'),
             ('writeFormat','ascii'), ('writePrecision','15'), ('writeCompression','off')):
    s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k,v), s, flags=re.M) if re.search(r'^%s\s+'%k, s, re.M) else s + '\n%-15s %s;\n' % (k,v)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSolution'); s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s)
s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
PYEOF
}

# 1. OpenFOAM develops the field to iteration DEV, and its written state is what both codes restart from.
stage "$W/dev" "$DEV"
( cd "$W/dev" && rhoSimpleFoam > log 2>&1 )
[ -d "$W/dev/$DEV" ] || { echo "SKIP: OpenFOAM did not reach iteration $DEV here"; tail -3 "$W/dev/log"; exit 77; }
# ...and that state must be DEVELOPED, or this gate measures the 0/0 limiter branch and not the code.
grep -q nonuniform "$W/dev/$DEV/k" && grep -q nonuniform "$W/dev/$DEV/T" \
    || { echo "FAIL: the restart state is uniform -- this protocol cannot discriminate"; exit 1; }

restart() {   # restart <dest>
    stage "$1" "$LAST"; rm -rf "$1/0"; cp -r "$W/dev/$DEV" "$1/$DEV"
}

# 2. OpenFOAM restarted from its own DEV (see the header for why it is not the continuous run).
restart "$W/of"; ( cd "$W/of" && rhoSimpleFoam > log 2>&1 )
[ -d "$W/of/$LAST" ] || { echo "SKIP: OpenFOAM's restart did not reach $LAST"; exit 77; }

compareArm() {   # compareArm <brae case> <ARM label> <bound>
    ARM="$2" BOUND="$3" DEV="$DEV" LAST="$LAST" python3 - "$1" "$W/of" <<'PYEOF'
import math, os, re, sys
b, o = sys.argv[1], sys.argv[2]
bound = float(os.environ['BOUND']); dev, last = int(os.environ['DEV']), int(os.environ['LAST'])
def rd(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if m: return [float(x) for x in m.group(1).replace('(',' ').replace(')',' ').split()], 'n'
    m = re.search(r'internalField\s+uniform\s+\(?([^;)]+)\)?;', s)
    return ([float(x) for x in m.group(1).split()], 'u') if m else (None, None)
ok, worst, wn = True, 0.0, ''
for it in range(dev + 1, last + 1):
    for f in ('U', 'p', 'T', 'k', 'epsilon', 'nut'):
        pb, po = f'{b}/{it}/{f}', f'{o}/{it}/{f}'
        if not (os.path.exists(pb) and os.path.exists(po)):
            print('     it %d %-8s MISSING   FAIL' % (it, f)); ok = False; continue
        a, ka = rd(pb); c, kc = rd(po)
        if kc == 'u': c = c * (len(a) // len(c))
        if ka == 'u': a = a * (len(c) // len(a))
        if len(a) != len(c):
            print('     it %d %-8s LENGTH %d vs %d   FAIL' % (it, f, len(a), len(c))); ok = False; continue
        den = math.sqrt(sum(y*y for y in c))
        if den == 0.0:
            print('     it %d %-8s DEGENERATE (zero-norm oracle)   FAIL' % (it, f)); ok = False; continue
        e = math.sqrt(sum((x-y)**2 for x, y in zip(a, c))) / den
        if e > worst: worst, wn = e, '%s it %d' % (f, it)
        if e >= bound:
            print('     it %d %-8s vs OpenFOAM (L2 rel)  %.3e   FAIL (bound %g)' % (it, f, e, bound)); ok = False
print('     %s: worst %s %.3e  (bound %g)   %s' % (os.environ['ARM'], wn, worst, bound, 'ok' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PYEOF
}

# 3. brae, host arm, restarted from the same state.
restart "$W/br"
if ! BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BRAE" -case "$W/br" > "$W/br/log" 2>&1 || [ ! -d "$W/br/$LAST" ]; then
    echo "     host: DID NOT RUN the tutorial"; grep -v '^brae NOTICE' "$W/br/log" | tail -3; fail=1
else
    compareArm "$W/br" host "$BOUND" || fail=1
fi

# 4. brae, CUDA arm, restarted from the same state, against the same OpenFOAM restart.
restart "$W/cuda"
if ! BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/cuda" > "$W/cuda/log" 2>&1 || [ ! -d "$W/cuda/$LAST" ]; then
    echo "     CUDA: DID NOT RUN the tutorial"; grep -v '^brae NOTICE' "$W/cuda/log" | tail -3; fail=1
else
    compareArm "$W/cuda" CUDA "$CUDABOUND" || fail=1
fi

[ $fail = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit $fail
