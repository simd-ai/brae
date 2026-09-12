#!/usr/bin/env bash
# brae's _cpp limitedLinearV / limitedLinear for div(phi,U), against OpenFOAM's OWN momentum matrix, on a
# DEVELOPED field. This is the gate that did not exist: rho_ueqn_cuda_scheme_limitedLinearV runs
# test_rho_ueqn_cuda on validation/pitzDailyTurb at 1576 and compares the DEVICE against the _cpp
# reference -- so the device is held to _cpp, and _cpp was held to nothing. The scheme's own
# transcription had no OpenFOAM oracle at all.
#
# THE ORACLE is tools/dumpPEqn, an instrumented copy of OpenFOAM's own rhoSimpleFoam with writes added
# and its equations untouched. It is the momentum matrix's OFF-DIAGONALS that are compared
# (stage_UUpper / stage_ULower), and they are the right quantity for a div scheme:
# fvMatrix::relax() rewrites only the diagonal and the source (D /= alpha; S += (D - D0)*psi), so upper
# and lower ARE the scheme's face weights times the flux, with no relaxation or boundary-coefficient
# convention to reconcile. The diagonal is NOT compared here: OpenFOAM's diag() excludes internalCoeffs
# and brae's includes them, so a diagonal comparison would measure that convention rather than the scheme.
#
# WHY A DEVELOPED FIELD. Restarted from OpenFOAM's OWN iteration 5. From the tutorial's 0.orig the case
# starts at rest with k `uniform 6` and epsilon `uniform 100`, and a limiter on a uniform field is in
# NVDTVD's 0/0 branch, where r is decided by the sign of round-off. A gate run from rest measures that
# and not the scheme.
#
# THE CONTROL, and it is what makes the measurement attributable: the SAME harness, the same restart, the
# same oracle, with div(phi,U) set to `Gauss upwind` reads 1.06e-14 -- machine precision. So the
# comparison, the restart, the dump-point alignment and the flux are all sound, and what is left is the
# limiter itself. Without this arm a broken harness and a broken limiter look identical.
#
# MEASURED, brae _cpp against OpenFOAM, gasMixing/injectorPipe restarted at iteration 5:
#     div(phi,U) Gauss upwind           UUpper 1.06e-14   ULower 9.77e-15    <- the control, exact
#     div(phi,U) Gauss limitedLinearV 1 UUpper 3.31e-03   ULower 3.08e-03    <- the case's own scheme
#     div(phi,U) Gauss limitedLinear 1  UUpper 4.63e-02   ULower 4.33e-02
# A _cpp component against OpenFOAM's own intermediates is held to machine precision; 3.3e-03 and 4.6e-02
# were defects, not tolerances, and this gate was registered WILL_FAIL until they were found in OpenFOAM's
# own source and fixed (the bound was never loosened):
#     limitedLinearV  3.31e-03 -> 1.76e-14   the limiter's gradient is taken from U's patch values BEFORE
#                                            updateCoeffs: gaussConvectionScheme.C:84 takes the weights,
#                                            then builds the fvMatrix at :89 whose constructor is what
#                                            calls updateCoeffs (fvMatrix.C:396)
#     limitedLinear   4.63e-02 -> 3.22e-13   `Gauss limitedLinear` on a VECTOR is NVDTVD + limitFuncs::
#                                            magSqr (LimitedScheme.H:188), so its gradient is
#                                            fvc::grad(magSqr(U)) resolved under `grad(magSqr(U))`, which
#                                            falls to `default` -- leastSquares here, not a hardcoded Gauss
#
# WHAT THIS GATE DOES NOT CLAIM. It says nothing about the CUDA arm: the device computes no leastSquares
# gradient and refuses this case by name, so no device arm can run here at all. It covers div(phi,U) only
# -- the limiters on div(phi,k|epsilon|e|K) are the same code but are not measured by this script. It
# uses gasMixing's snappyHexMesh mesh (about a minute to build); compressible/rhoSimpleFoam/
# angledDuctExplicitFixedCoeff, a blockMesh, was measured to discriminate the same defect at 1.3e-03
# against a 2.0e-15 control and would make a faster arm.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
MATCH=${MATCH:-1e-12}        # a _cpp component against OpenFOAM's own intermediates
DEV=5                        # restart from OpenFOAM's own iteration 5

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v dumpPEqn      > /dev/null 2>&1 || { echo "SKIP: tools/dumpPEqn not built (see the of-instrument skill)"; exit 77; }
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

stage() {   # stage <dest> <endTime> <div(phi,U) scheme>
    rm -rf "$1"; cp -r "$W/mesh" "$1"
    rm -rf "$1"/0 "$1"/[1-9]* "$1"/processor* "$1"/log.* "$1"/postProcessing "$1"/dynamicCode
    cp -r "$1/0.orig" "$1/0"
    N="$2" USCH="$3" python3 - "$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
p = os.path.join(d, 'system/controlDict'); s = open(p).read()
for k, v in (('startFrom','latestTime'), ('endTime',os.environ['N']), ('writeInterval','1'),
             ('writeFormat','ascii'), ('writePrecision','15'), ('writeCompression','off')):
    s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k,v), s, flags=re.M) if re.search(r'^%s\s+'%k, s, re.M) else s + '\n%-15s %s;\n' % (k,v)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(p, 'w').write(s)
# Only the linear-solver stopping criterion is tightened, so the two codes do not part on solver noise.
p = os.path.join(d, 'system/fvSolution'); s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s)
s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSchemes'); s = open(p).read()
s = re.sub(r'div\(phi,U\)\s+[^;]+;', 'div(phi,U) %s;' % os.environ['USCH'], s)
open(p, 'w').write(s)
PYEOF
}

# Develop the field with the case's OWN scheme, so every arm restarts from the same state.
stage "$W/dev" "$DEV" "Gauss limitedLinearV 1"
( cd "$W/dev" && rhoSimpleFoam > log 2>&1 )
[ -d "$W/dev/$DEV" ] || { echo "SKIP: could not develop the field to iteration $DEV"; exit 77; }

# ...and assert it is actually developed. A gate that silently restarted from a uniform state would be
# measuring NVDTVD's 0/0 branch, which is what this restart exists to avoid.
grep -q "nonuniform" "$W/dev/$DEV/U" \
    || { echo "FAIL: the restart state is uniform -- this fixture cannot discriminate"; exit 1; }

# arm <label> <div(phi,U) scheme> <bound|CONTROL>
arm() {
    stage "$W/of" $((DEV+1)) "$2"; rm -rf "$W/of/0"; cp -r "$W/dev/$DEV" "$W/of/$DEV"
    # dumpPEqn's guard is runTime.timeIndex(), which is ABSOLUTE -- restarting at 5 makes the first
    # assembled iteration index 6, not 1.
    ( cd "$W/of" && BRAE_DUMP_STAGE_ITER=$((DEV+1)) dumpPEqn > log.dump 2>&1 )
    [ -f "$W/of/$((DEV+1))/stage_UUpper" ] || { echo "     $1: the instrument wrote no stage_UUpper"; fail=1; return; }
    stage "$W/br" $((DEV+1)) "$2"; rm -rf "$W/br/0"; cp -r "$W/dev/$DEV" "$W/br/$DEV"; mkdir -p "$W/br/dump"
    ( cd "$W/br" && BRAE_STAGE_DUMP_DIR="$W/br/dump" BRAE_STAGE_DUMP_ITER=1 \
        BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BRAE" -case "$W/br" > log.brae 2>&1 )
    [ -f "$W/br/dump/UUpper" ] || { echo "     $1: brae wrote no UUpper: $(grep -v '^brae NOTICE' "$W/br/log.brae" | tail -1 | cut -c1-100)"; fail=1; return; }
    LABEL="$1" BOUND="$3" python3 - "$W" <<'PYEOF' || fail=1
import math, os, re, sys
W = sys.argv[1]
label, bound = os.environ['LABEL'], float(os.environ['BOUND'])
def offield(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if m:
        return [float(x) for x in m.group(1).replace('(',' ').replace(')',' ').split()]
    m = re.search(r'internalField\s+uniform\s+\(?([^;)]+)\)?;', s)
    return [float(x) for x in m.group(1).split()] if m else None
def braef(p):
    return [float(x) for ln in open(p) for x in ln.split()]
ok, out = True, []
for bn, on in (('UUpper','stage_UUpper'), ('ULower','stage_ULower')):
    a = braef(os.path.join(W, 'br/dump', bn))
    b = offield(os.path.join(W, 'of/6', on))
    if b is None or len(a) != len(b):
        out.append('%s LENGTH %d/%s' % (bn, len(a), len(b) if b else 'none')); ok = False; continue
    nb = math.sqrt(sum(y*y for y in b))
    if nb == 0.0:
        # a zero-norm oracle reports 0 for any brae field at all; that is not agreement
        out.append('%s DEGENERATE (oracle norm 0)' % bn); ok = False; continue
    e = math.sqrt(sum((x-y)**2 for x, y in zip(a, b))) / nb
    out.append('%s %.3e' % (bn, e))
    ok = ok and (e < bound)
print('     %-34s %s   (< %.0e)   %s' % (label, '   '.join(out), bound, 'ok' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PYEOF
}

echo "== brae _cpp div(phi,U) against OpenFOAM's own momentum matrix, restarted at iteration $DEV =="
# THE CONTROL FIRST: if this does not read machine precision, nothing below is attributable to a limiter.
arm "CONTROL Gauss upwind"        "Gauss upwind"           "$MATCH"
arm "Gauss limitedLinearV 1"      "Gauss limitedLinearV 1" "$MATCH"
arm "Gauss limitedLinear 1"       "Gauss limitedLinear 1"  "$MATCH"

[ $fail = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit $fail
