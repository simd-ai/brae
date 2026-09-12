#!/bin/bash
# THE TILTED SYMMETRY PLANE, both arms -- rhoBoxSym/README.md.
#
# An axis-aligned symmetry plane can never discriminate a symmetry defect on a segregated solver: with n
# along one axis the per-component treatment (vf = |n_k|) decouples, ref_normal is identically 0 whatever
# the cell velocity is, and a STALE ref equals a fresh one exactly. rhoBoxSym tilts the plane ~4 degrees
# so the components couple, and this is the only fixture in the tree that can see it.
#
# ARM 1 (host)  the _cpp mirror against real OpenFOAM, converged.
# ARM 2 (cuda)  the device modules against the _cpp mirror, per iteration, on the tilted plane. The
#               device used to REFUSE this case; it runs now, and this arm is what earned the lift.
# ARM 3 (cuda)  the physical oracle: max |phi| on the symmetry patch, at iterations 1..3, against ZERO.
# ARM 4 (cuda)  the whole solve, device against host, at a matched iteration count.
#
# WHY ARM 3 EXISTS AND ARM 4 IS NOT ENOUGH. The defect this gate was built around -- the driver refreshing
# U's boundary after the momentum solve and after the velocity correction without rebuilding the
# symmetry refValue those refreshes blend towards -- is TRANSIENT. It leaked 7.13e-04 of flux through
# `slant` at iteration 2 and the run still converged to the same fixed point: arm 4 reads U 7.2e-11
# against the host with the defect present and 6.8e-11 with it fixed. Only arm 3 separates them, because
# no-penetration is an oracle at every iteration and convergence is not.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBoxSym" && pwd)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-$ROOT/build}
WORK=${WORK:-/tmp/brae_sym_vs_of}
# The staged copies live OUTSIDE $WORK: `cp -r "$WORK" "$STAGE/tr"` copies a directory into itself.
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
MIRRORBIN="$BUILD/brae_rhoSimpleFoam"
ITERS=${ITERS:-200}
if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
[ -n "$OFLAST" ] || { echo "FAIL: OpenFOAM produced no output"; exit 1; }

# ---- ARM 1: the host mirror against real OpenFOAM ------------------------------------------------
out=$("$BUILD/test_rho_simple_step_cpp" "$WORK" 0 "$OFLAST" 2>&1) || { echo "$out" | tail -15; echo "FAIL(host)"; exit 1; }
echo "$out" | grep -E "^     [UTp] " | head -3
echo "$out" | grep -q "^PASS" || { echo "$out" | tail -5; echo "FAIL(host)"; exit 1; }
echo "PASS(host)"

# ---- ARM 2: the device modules against the host mirror, per iteration ----------------------------
# This case was REFUSED on this arm ("not aligned to a coordinate axis") until the refresh was fixed.
# The harness's own Ux bound is 1e-9; with the refresh removed iteration 1 reads Ux 2.77e-06.
dout=$("$BUILD/test_rho_simple_step_cuda" "$WORK" 0 3 2>&1) \
    || { echo "$dout" | tail -8; echo "FAIL(cuda): the tilted case did not run on the device arm"; exit 1; }
echo "$dout" | grep -E "^     iter" | head -3
echo "$dout" | grep -q "^PASS" || { echo "$dout" | tail -6; echo "FAIL(cuda): per-iteration agreement"; exit 1; }
echo "PASS(cuda-step)"

# ---- ARM 3: no penetration, the oracle that sees the transient -----------------------------------
stage()   # $1 dst  $2 endTime  $3 writeInterval -- residualControl removed so the count is exact
{
    rm -rf "$1"; cp -r "$WORK" "$1"; rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    ENDT="$2" WINT="$3" python3 - "$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
for k, v in [('endTime', os.environ['ENDT']), ('writeInterval', os.environ['WINT']),
             ('writeControl', 'timeStep'), ('writeFormat', 'ascii'), ('writePrecision', '15'),
             ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PYEOF
}
stage "$STAGE/tr" 3 1
( cd "$STAGE/tr" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$MIRRORBIN" -case "$STAGE/tr" > tr.log 2>&1 ) \
    || { tail -5 "$STAGE/tr/tr.log"; echo "FAIL(cuda): the transient run did not finish"; exit 1; }
# 1e-15 is three orders BELOW real OpenFOAM's own 1.45e-16 on this patch and eleven above brae's 1.96e-19.
python3 "$ROOT/tests/sym_patch_flux.py" "$STAGE/tr" slant 1e-15 1 2 3 \
    || { echo "FAIL(cuda): the symmetry plane is carrying flux"; exit 1; }
echo "PASS(cuda-noPenetration)"

# ---- ARM 4: the whole solve, device against host, matched iteration count ------------------------
stage "$STAGE/cu" "$ITERS" "$ITERS"
stage "$STAGE/ho" "$ITERS" "$ITERS"
( cd "$STAGE/cu" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$MIRRORBIN" -case "$STAGE/cu" > cu.log 2>&1 ) \
    || { tail -5 "$STAGE/cu/cu.log"; echo "FAIL(cuda): the end-to-end run did not finish"; exit 1; }
( cd "$STAGE/ho" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$MIRRORBIN" -case "$STAGE/ho" > ho.log 2>&1 ) \
    || { tail -5 "$STAGE/ho/ho.log"; echo "FAIL(host): the end-to-end run did not finish"; exit 1; }
# Measured at 200 iterations: p 3.00e-13, T 1.13e-10, U 6.83e-11, rho 1.12e-10. Bounds ~30x, and the
# OpenFOAM column is not asserted here -- arm 1 owns that comparison and the host mirror carries its own
# 8.2e-06 offset on this fixture, which the device arm inherits exactly (8.1874e-06) rather than adding to.
CUDA_P_BOUND=1e-11 CUDA_T_BOUND=4e-09 CUDA_U_BOUND=2e-09 CUDA_RHO_BOUND=4e-09 \
BRAE_DIR="$STAGE/cu/$ITERS" HOST_DIR="$STAGE/ho/$ITERS" OF_DIR="" \
python3 "$ROOT/tests/rho_mirror_compare.py" --host-only \
    || { echo "FAIL(cuda): the end-to-end device solve does not track the host mirror"; exit 1; }
echo "PASS(cuda-endToEnd)"
echo PASS
