#!/usr/bin/env bash
# OpenFOAM's own aerofoilNACA0012 tutorial on BOTH rhoSimpleFoam OF-mirror arms, against real OpenFOAM
# v2412, iteration by iteration. It is the only rhoSimpleFoam tutorial whose fvSchemes names
# `cellLimited Gauss linear 1` for grad(k) and grad(omega) as well as grad(U), and that is what it is
# here to hold: CDkOmega takes fvc::grad(k_) & fvc::grad(omega_) (kOmegaSSTBase.C:555-558), and
# fvc::grad(vf) resolves its scheme by the FIELD's name through gradSchemes (fvcGrad.C:149), so this
# case limits both gradients where every other fixture leaves them `Gauss linear`.
#
# WHAT THE DEVICE ARM DID BEFORE: took the plain Gauss gradient for both, whatever the case said. The
# host reference limited them (kOmegaSST_cpp.cu:458-461), so the two arms disagreed on a case no gate
# ran -- naca_vs_openfoam.sh drives the PRE-MIRROR path and limit_temperature_vs_openfoam.sh runs this
# tutorial on the host arm only. Measured, CUDA arm against OpenFOAM, before and after:
#     iteration 2   omega 9.1e-04 -> 7.2e-10     k 1.1e-05 -> 7.7e-09     nut 2.7e-05 -> 1.5e-08
#     iteration 3   omega 3.7e-03 -> 3.8e-09     k 6.5e-05 -> 1.3e-08     nut 1.1e-04 -> 3.2e-08
# The host arm holds 2.8e-12 on every field at every iteration throughout, which is what makes this
# fixture able to discriminate at all: a case whose own host reference could not reach 1e-12 would pass
# whichever gradient the device took.
#
# THE SECOND DEFECT, FOUND WITH THE CLOSURE INSTRUMENT AND FIXED (sst_stage_dump.cuh). At iteration 1 k
# and omega are still uniform, so CDkOmega's gradients are zero and the limiter above cannot be what
# moves -- yet the CUDA arm read k 7.0e-09 / nut 5.4e-09 against the host's 2.6e-12. Dumping the closure
# stage by stage put it upstream of everything the closure computes: the cell velocity agreed to 3e-12
# while U's PATCH values on the freestream inlet differed by up to 8.1e-04, all 200 faces, with the
# noSlip aerofoil bit-identical. OpenFOAM's freestreamVelocity sets
#     valueFraction() = 0.5 - 0.5*(Up & patch().nf())/mag(Up),  Up = *this
# (freestreamVelocityFvPatchVectorField.C) -- `Up` is the patch field's OWN STORED value, what its last
# evaluate wrote. The host passes exactly that (U.boundary[pi]->value(), rhoSimpleFoam_cpp.cu:463); the
# device re-evaluated the patch against the cells AS THEY STAND, which is the same number only while the
# cells have not moved since. It carries f.UxBnd/UyBnd/UzBnd -- the stored patch velocity, refreshed
# exactly where the host calls U.evaluateBoundary -- and deviceUpdateMixedFreestream now takes those.
# The control: replacing freestreamVelocity with fixedValue on all three codes collapsed the whole
# disagreement to 2.5e-12 at every iteration, before any fix.
#
# THE THIRD DEFECT, THE SAME CLASS, ON THE MOMENTUM AND PRESSURE SIDES. With the closure exact the CUDA arm
# still read U 4.4e-09 and p 1.5e-07 from iteration 2. The solver's stage dump (BRAE_STAGE_DUMP_DIR) put
# the first disagreement in the momentum SOURCE -- every coefficient at 1e-14, the source 8.8e-05 off on
# 182 of the inlet's 200 cells -- and, once that was closed, in the pressure equation as a near-uniform
# LEVEL shift (p 1.3e-01 on ~1e5, mean -9.6e-03, std 1.6e-02: the farfield freestreamPressure patch pins
# the level). Both were the device re-evaluating a patch value against the cells as they stand where
# OpenFOAM and the host read the field's STORED value: rhoUEqn.cu's five gradient sites (the host's
# fvc::gaussGrad sums U.boundary[pi]->value(), fvc.cu:189), rhoPEqn.cu's and rhoPcEqn.cu's HbyA
# boundary, constrainPressure's Sf&U_b, the psi*p boundary and the non-orthogonal grad(p)
# (rhoPEqn_cpp.cu:184, :221, :256, :303), and the driver's pre-assembly p evaluate, which touched every
# face where the host's updateTotalPressure(evaluateAll = false) touches only the totalPressure faces --
# "a mixed or zeroGradient face keeps the blend p.relax() left". Each now takes the stored value the
# driver carries (f.UxBnd/UyBnd/UzBnd, f.pBnd). After: every field on the CUDA arm at 1.2e-12 to 2.8e-12
# at every iteration, the host's own numbers.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=3
HOSTBOUND=1e-11      # measured: every field <= 2.8e-12 at every iteration
CUDABOUND=1e-11      # the same, at every iteration: nothing is left to bound around on this tutorial
CUDABOUND1=1e-11
OMEGABOUND=1e-11     # was 1e-07 while the iteration-2 gap stood; the limiter's signature is 3.7e-03 without it

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/aerofoilNACA0012"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }
[ -f "${FOAM_TUTORIALS}/resources/geometry/NACA0012.obj.gz" ] || { echo "SKIP: NACA0012 geometry not shipped"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

# The tutorial's OWN meshing, its Allrun.pre (blockMesh, transformPoints, extrudeMesh) then topoSet.
cp -r "$SRC" "$W/mesh" || exit 1
( cd "$W/mesh" && ./Allrun.pre > log.pre 2>&1 && topoSet > log.topoSet 2>&1 ) \
    || { echo "SKIP: the tutorial's own meshing did not run here"; tail -3 "$W/mesh/log.pre"; exit 77; }
[ -f "$W/mesh/constant/polyMesh/points" ] || { echo "SKIP: no mesh produced"; exit 77; }

# stage <dest>. Nothing is touched but the write settings and the LINEAR SOLVER stopping criterion: at
# the tutorial's own tolerance each code stops its solve at a different point and the two drift apart on
# solver noise rather than on the discretisation this measures. Schemes, boundary conditions, thermo and
# fvOptions are the tutorial's own.
stage() {
    rm -rf "$1"
    cp -r "$W/mesh" "$1"
    rm -rf "$1"/0 "$1"/[1-9]* "$1"/log.* "$1"/postProcessing
    cp -r "$1/0.orig" "$1/0"
    ITERS="$ITERS" python3 - "$1" <<'PYEOF'
import os, re, sys
d, n = sys.argv[1], os.environ['ITERS']
p = os.path.join(d, 'system/controlDict')
s = open(p).read()
for k, v in (('startFrom', 'latestTime'), ('endTime', n), ('writeInterval', '1'),
             ('writeFormat', 'ascii'), ('writePrecision', '15'), ('writeCompression', 'off')):
    if re.search(r'^%s\s+' % k, s, re.M):
        s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
    else:
        s += '\n%-15s %s;\n' % (k, v)
open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSolution')
s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s)
s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
PYEOF
}

# compare <braeDir> <ofDir> <label> <bound> <omegaBound> <iteration-1 bound>
compare() {
    LABEL="$3" BOUND="$4" OMEGABOUND="$5" BOUND1="$6" ITERS="$ITERS" python3 - "$1" "$2" <<'PYEOF'
import math, os, re, sys
brae, of = sys.argv[1], sys.argv[2]
label = os.environ['LABEL']
bound, ombound = float(os.environ['BOUND']), float(os.environ['OMEGABOUND'])
bound1 = float(os.environ['BOUND1'])
n = int(os.environ['ITERS'])
def internal(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if not m:
        m = re.search(r'internalField\s+uniform\s+([^;]+);', s)
    return [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', m.group(1))]
def rel(a, b):
    den = sum(y * y for y in b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / den) if den > 0 else 0.0
bad = 0
for it in range(1, n + 1):
    for f in ('U', 'p', 'T', 'k', 'omega', 'nut'):
        a, b = internal('%s/%d/%s' % (brae, it, f)), internal('%s/%d/%s' % (of, it, f))
        if len(a) != len(b):
            print('     %s it %d  %-6s LENGTH MISMATCH %d vs %d   FAIL' % (label, it, f, len(a), len(b)))
            bad = 1
            continue
        e = rel(a, b)
        # Iteration 1 is held to the tight bound on both arms; it is what the freestream fix owns.
        lim = bound1 if it == 1 else (ombound if f == 'omega' else bound)
        ok = e < lim
        print('     %s it %d  %-6s vs OpenFOAM (L2 rel)  %.6e   %s'
              % (label, it, f, e, 'ok' if ok else 'FAIL (bound %g)' % lim))
        if not ok:
            bad = 1
sys.exit(1 if bad else 0)
PYEOF
}

echo "== aerofoilNACA0012, kOmegaSST with cellLimited grad(U|k|omega) -- OpenFOAM =="
stage "$W/of"
( cd "$W/of" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )
[ -d "$W/of/$ITERS" ] || { echo "SKIP: OpenFOAM did not reach $ITERS iterations"; tail -4 "$W/of/log.rhoSimpleFoam"; exit 77; }
echo "     OpenFOAM ran $ITERS iterations                                              ok"

for MIRROR in 1 cuda
do
    arm=$([ "$MIRROR" = cuda ] && echo 'CUDA arm' || echo 'host arm')
    bound=$([ "$MIRROR" = cuda ] && echo "$CUDABOUND" || echo "$HOSTBOUND")
    ombound=$([ "$MIRROR" = cuda ] && echo "$OMEGABOUND" || echo "$HOSTBOUND")
    bound1=$([ "$MIRROR" = cuda ] && echo "$CUDABOUND1" || echo "$HOSTBOUND")
    d="$W/br_$MIRROR"
    stage "$d"
    if ! BRAE_RHOSIMPLEFOAM_MIRROR=$MIRROR "$BRAE" -case "$d" > "$d/log.brae" 2>&1 || [ ! -d "$d/$ITERS" ]
    then
        echo "FAIL: the $arm did not run the tutorial"
        grep -v '^brae NOTICE' "$d/log.brae" | tail -5
        fail=1
        continue
    fi
    echo "== aerofoilNACA0012 -- $arm =="
    compare "$d" "$W/of" "$arm" "$bound" "$ombound" "$bound1" || fail=1
done

[ "$fail" -eq 0 ] && echo "PASSED" || echo "FAILED"
exit $fail
