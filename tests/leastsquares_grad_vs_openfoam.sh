#!/usr/bin/env bash
# fvc::leastSquaresGrad and deviceLeastSquaresGrad against OpenFOAM's OWN leastSquares gradient.
#
# OpenFOAM's leastSquares is an inverse-distance-weighted least-squares fit (leastSquaresGrad.C,
# leastSquaresVectors.C), not a Gauss sum, and a case naming it must get it: on validation/rhoLU at a
# developed state, swapping the limitedLinear limiter's gradient from Gauss linear to this moves the
# assembled energy diagonal by 9.1e-03. rhoSimpleFoam's OF-mirror refused such a case by name.
#
# THE ORACLE is OpenFOAM computing grad(T) on the same mesh and the same field, twice -- once with
# `gradSchemes default leastSquares` and once with `Gauss linear` -- via postProcess -func "grad(T)",
# which resolves through the case's own gradSchemes. That gives a 2x2 rather than a single number:
#
#                     OF leastSquares     OF Gauss linear
#   brae leastSquares      MATCH               differ
#   brae gaussGrad         differ              MATCH
#
# The off-diagonal is the control. Without it the gate would pass on any mesh where the two schemes
# happen to agree, which is exactly the mesh on which the port would never be exercised.
#
# THE MESH IS 2-D ON PURPOSE. An emptyFvPatch has size 0, so it contributes nothing to the least-squares
# fit and leaves the dd tensor SINGULAR in the empty direction. OpenFOAM inverts it with
# SymmTensor::safeInv (SymmTensorI.H:368-421), which detects the near-zero diagonal component, adds one,
# inverts, and subtracts it again. A plain cofactor inverse divides by ~0 here and on most of
# validation/. This gate is the only thing that exercises that path.
#
# FAIL-PROOF, measured in-session: with a plain inverse in place of safeInv the least-squares arm
# returns non-finite values on this fixture; with the volume division that gaussGrad ends with (the easy
# transcription mistake -- the fit vectors already carry the normalisation) it reads O(1) instead of
# 1e-13.
#
# NOT an end-to-end gate. The gradient is validated here; the CASE it unblocks is not -- gasMixing still
# parts from OpenFOAM by U 1.2e-01 with the gradient matched on both sides. rhoSimpleFoam therefore still
# reaches leastSquares by default; the BRAE_LEASTSQUARES=1 opt-in that once gated it is gone (its case, gasMixing, now matches -- tests/rho_gasmixing_vs_openfoam.sh).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_leastsquares_grad"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
MATCH=${MATCH:-1e-10}      # brae vs OpenFOAM, same scheme
DIFFER=${DIFFER:-1e-3}     # brae vs OpenFOAM, the OTHER scheme -- must be far above MATCH
DEVHOST=${DEVHOST:-1e-13}  # the CUDA port against its own host reference

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoSST" ] || { echo "SKIP: fixture rhoSST missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v postProcess   > /dev/null 2>&1 || { echo "SKIP: postProcess not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

for sc in lsq gauss; do
    d="$W/$sc"; mkdir -p "$d"
    cp -r "$ROOT/validation/rhoSST/constant" "$ROOT/validation/rhoSST/system" "$d/"
    cp -r "$ROOT/validation/rhoSST/0.orig" "$d/0"
    case $sc in lsq) G="leastSquares" ;; gauss) G="Gauss linear" ;; esac
    G="$G" python3 - "$d" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
f = os.path.join(d, 'system/fvSchemes'); s = open(f).read()
s = re.sub(r'gradSchemes\s*\{[^}]*\}', 'gradSchemes { default %s; }' % os.environ['G'], s, flags=re.S)
open(f, 'w').write(s)
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
for k, v in [('endTime', '1'), ('writeInterval', '1'), ('writeControl', 'timeStep'),
             ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1'),
             ('writeFormat', 'ascii'), ('writePrecision', '15')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
s = re.sub(r'\bfunctions\s*\{.*\}\s*$', '', s, flags=re.S)
open(c, 'w').write(s)
PYEOF
    ( cd "$d" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
    ( cd "$d" && rhoSimpleFoam > log 2>&1 )       || { tail -5 "$d/log"; echo "FAIL: OpenFOAM ($sc)"; exit 1; }
    ( cd "$d" && postProcess -func "grad(T)" -time 1 > log.pp 2>&1 ) \
        || { tail -5 "$d/log.pp"; echo "FAIL: postProcess grad(T) ($sc)"; exit 1; }
    [ -f "$d/1/grad(T)" ] || { echo "FAIL: no grad(T) written ($sc)"; exit 1; }
    # ...and the VECTOR form. grad(U) is a volTensorField and its own arm: lsGrad_ij = ownLs_i*deltaVsf_j,
    # a different expression from the scalar one, with its own consumers (divDevRhoReff's dev2 term and
    # the closure's production). It had no oracle until those consumers existed.
    ( cd "$d" && postProcess -func "grad(U)" -time 1 > log.ppU 2>&1 ) \
        || { tail -5 "$d/log.ppU"; echo "FAIL: postProcess grad(U) ($sc)"; exit 1; }
    [ -f "$d/1/grad(U)" ] || { echo "FAIL: no grad(U) written ($sc)"; exit 1; }
done

MATCH="$MATCH" DIFFER="$DIFFER" DEVHOST="$DEVHOST" python3 - "$BIN" "$W" <<'PYEOF' || fail=1
import os, subprocess, sys
BIN, W = sys.argv[1], sys.argv[2]
MATCH, DIFFER, DEVHOST = (float(os.environ[k]) for k in ('MATCH', 'DIFFER', 'DEVHOST'))

def run(case, field='T'):
    out = subprocess.run([BIN, os.path.join(W, case), '1', field], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout, out.stderr); sys.exit(1)
    v = {}
    for line in out.stdout.splitlines():
        p = line.split()
        if len(p) >= 3 and p[-2] == 'relL2':
            v[' '.join(p[:-2])] = float(p[-1])
    return v

lsq, gauss = run('lsq'), run('gauss')
lsqU, gaussU = run('lsq', 'U'), run('gauss', 'U')
ok = True
def check(label, got, bound, want_below):
    global ok
    good = (got < bound) if want_below else (got > bound)
    print('     %-56s %.4e  (%s %.1e)  %s'
          % (label, got, '<' if want_below else '>', bound, 'ok' if good else 'FAIL'))
    ok = ok and good

check('brae leastSquares vs OpenFOAM leastSquares',      lsq['leastSquares'],  MATCH,  True)
check('brae gaussGrad    vs OpenFOAM Gauss linear',      gauss['gaussLinear'], MATCH,  True)
check('CONTROL brae leastSquares vs OpenFOAM Gauss linear', lsq['gaussLinear'], DIFFER, False)
check('CONTROL brae gaussGrad vs OpenFOAM leastSquares', gauss['leastSquares'], DIFFER, False)
check('device leastSquares vs OpenFOAM leastSquares',    lsq['device lsq'],    MATCH,  True)
check('device leastSquares vs its own host reference',   lsq['device-host'],   DEVHOST, True)
# The VECTOR form, against OpenFOAM's own grad(U), with the same 2x2. No device twin: the CUDA mirror
# arm refuses a leastSquares grad(U) by name (rhoSimpleFoamDriver.cu) until that module is ported.
check('brae leastSquares grad(U) vs OpenFOAM leastSquares',      lsqU['leastSquares'],  MATCH,  True)
check('brae gaussGrad    grad(U) vs OpenFOAM Gauss linear',      gaussU['gaussLinear'], MATCH,  True)
check('CONTROL brae leastSquares grad(U) vs OpenFOAM Gauss linear', lsqU['gaussLinear'], DIFFER, False)
check('CONTROL brae gaussGrad grad(U) vs OpenFOAM leastSquares', gaussU['leastSquares'], DIFFER, False)
sys.exit(0 if ok else 1)
PYEOF

# ---- ARM 2: a mesh whose BOUNDARY FACES ARE SKEWED ---------------------------------------------------
# Everything above runs on validation/rhoSST, an orthogonal blockMesh, and an orthogonal mesh CANNOT SEE
# the defect this arm exists for. fvPatch::delta() on an uncoupled patch is the patch-normal projection
# `nHat*(nHat & (Cf() - Cn()))` (fvPatch.C), which leastSquaresVectors uses for both the dd accumulation
# and the boundary fit vectors. brae used the raw Cf - Cn. The two are the SAME VECTOR wherever Cf - Cn is
# already normal to the face -- every orthogonal blockMesh -- so rhoSST read 2.5e-13 and this gate passed
# green while the code was wrong, on host and device alike.
#
# MEASURED, brae's leastSquares grad(p) against OpenFOAM's own, same case, same scheme:
#     validation/pitzDaily          raw Cf-Cn 1.18e-01   projected 1.8e-12   <- discriminates
#     validation/squareBend         raw 4.6e-15          projected 4.2e-15   <- cannot tell them apart
#     validation/windAroundBuildings raw 3.7e-15         projected 3.4e-15   <- cannot tell them apart
# Two of the three candidate fixtures are useless as a gate for this, which is the whole reason it hid.
# It was found end to end on gasMixing/injectorPipe's snappyHexMesh mesh (grad(p) 5.8e-02, 90% of the
# squared error in 77 of 74650 cells, every one touching a boundary) and pitzDaily is the committed
# fixture that reproduces it in 12225 cells.
PD="$W/pitz"
rm -rf "$PD"; mkdir -p "$PD"
cp -r "$ROOT/validation/pitzDaily/constant" "$ROOT/validation/pitzDaily/system" "$PD/"
cp -r "$ROOT/validation/pitzDaily/0" "$PD/0"
python3 - "$PD" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
f = os.path.join(d, 'system/fvSchemes'); s = open(f).read()
s = re.sub(r'gradSchemes\s*\{[^}]*\}', 'gradSchemes { default leastSquares; }', s, flags=re.S)
open(f, 'w').write(s)
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
s = re.sub(r'\bfunctions\s*\{.*\}\s*$', '', s, flags=re.S)
# five iterations, so p is NONUNIFORM: at time 0 it is uniform and OpenFOAM writes grad(p) as
# `uniform (0 0 0)`, which is a zero-denominator oracle that reports 0 for any brae field at all.
for k, v in (('writeFormat','ascii'), ('writePrecision','15'), ('writeCompression','off'),
             ('endTime','5'), ('writeInterval','5'), ('writeControl','timeStep'),
             ('startFrom','startTime'), ('startTime','0'), ('deltaT','1')):
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s) if re.search(r'\b%s\s+' % k, s) else s + '\n%s %s;\n' % (k, v)
open(c, 'w').write(s)
p = os.path.join(d, 'system/fvSolution'); s = open(p).read()
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s); open(p, 'w').write(s)
PYEOF
if ( cd "$PD" && simpleFoam > log.simpleFoam 2>&1 ) && [ -d "$PD/5" ] \
   && ( cd "$PD" && postProcess -func "grad(p)" -time 5 > log.pp 2>&1 ) \
   && grep -q nonuniform "$PD/5/grad(p)"
then
    MATCH="$MATCH" DIFFER="$DIFFER" python3 - "$BIN" "$PD" <<'PYEOF' || fail=1
import os, subprocess, sys
BIN, PD = sys.argv[1], sys.argv[2]
MATCH, DIFFER = float(os.environ['MATCH']), float(os.environ['DIFFER'])
out = subprocess.run([BIN, PD, '5', 'p'], capture_output=True, text=True)
if out.returncode != 0:
    print(out.stdout, out.stderr); sys.exit(1)
v = {}
for line in out.stdout.splitlines():
    q = line.split()
    if len(q) >= 3 and q[-2] == 'relL2':
        v[' '.join(q[:-2])] = float(q[-1])
ok = True
def check(label, got, bound, below):
    global ok
    good = (got < bound) if below else (got > bound)
    print('     %-56s %.4e  (%s %.1e)  %s'
          % (label, got, '<' if below else '>', bound, 'ok' if good else 'FAIL'))
    ok = ok and good
check('pitzDaily: brae leastSquares vs OpenFOAM leastSquares', v['leastSquares'], MATCH, True)
check('pitzDaily: device leastSquares vs OpenFOAM leastSquares', v['device lsq'], MATCH, True)
# The CONTROL. Without it this arm would pass on a brae whose leastSquares had silently become Gauss.
check('pitzDaily CONTROL brae gaussGrad vs OpenFOAM leastSquares', v['gaussLinear'], DIFFER, False)
sys.exit(0 if ok else 1)
PYEOF
    say "leastSquares holds on a mesh with SKEWED boundary faces (fvPatch::delta projection)" \
        "$([ $fail = 0 ] && echo ok || echo FAIL)"
else
    echo "     pitzDaily arm: simpleFoam or postProcess did not produce a nonuniform grad(p) -- SKIPPED"
fi

say "leastSquares reproduces OpenFOAM's own gradient, on host and device" "$([ $fail = 0 ] && echo ok || echo FAIL)"
exit $fail
