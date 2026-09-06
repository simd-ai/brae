#!/usr/bin/env bash
# The DIV SCHEME's face weight at a COUPLED interface (cyclic / cyclicAMI), against real OpenFOAM.
#
# WHAT WAS WRONG. OpenFOAM assembles a coupled patch with the interpolation weights of the scheme the
# case NAMED -- gaussConvectionScheme::fvmDiv writes internalCoeffs = phi*w and boundaryCoeffs =
# -phi*(1-w) on every coupled patch, with w from limitedSurfaceInterpolationScheme::weights, and
# LimitedScheme::calcLimiter has a coupled() branch that limits an interface face exactly as it limits an
# internal one (only an UNCOUPLED patch is given the constant limiter 1.0). brae hardcoded UPWIND there --
# ifCoeff = -lap + min(phi,0), diag += lap + max(phi,0), which is the special case w = pos0(phi) -- for
# every case and every field, momentum and all six turbulence scalars alike.
#
# pipeCyclic asks for `bounded Gauss limitedLinearV 1` on U and `bounded Gauss limitedLinear 1` on k and
# epsilon, so brae was solving a different equation from the one the case describes at 500 of its faces.
#
# TWO CASES RUN THIS SCRIPT, because the defect is one thing and the schemes that expose it are not:
#   validation/pipeCyclic    `bounded Gauss limitedLinearV 1`   turbulent, rotational AMI, 3500 cells
#   validation/implicitAMI   `bounded Gauss linear`             laminar, non-conformal AMI, 24 cells
# The second is the harder weight to get wrong quietly: central differencing puts w at ~0.5, the furthest
# a weight can sit from upwind's 0 or 1, and it took U from 2.894e-01 to 1.780e-02. Everything below is
# written in terms of "the scheme the case names" rather than either of them.
#
# WHY IT SURVIVED SO LONG. brae's device AMI and its _cpp reference agreed to 1e-16 through sixteen stage
# gates -- because both implemented the same wrong scheme. Only OpenFOAM's own fvMatrix could tell them
# apart, and it did: L2 6.86e-01 on the interface off-diagonal, with the gap equal to phi*(1-w) face by
# face. A stage gate against your own reference cannot find a defect the reference shares.
#
# THE FOUR RUNS, and why fewer would not do. Matching OpenFOAM on the named-scheme case is necessary but not
# sufficient: a case where the two schemes happen to agree would pass it while proving nothing. So the
# same case is run under BOTH schemes by BOTH codes:
#
#     1. brae limited  vs  OpenFOAM limited     must AGREE   (the fix)
#     2. brae upwind   vs  OpenFOAM upwind      must AGREE   (upwind was never broken -- and this is what
#                                                             says the weight is scheme-driven, not tuned)
#     3. OpenFOAM limited vs OpenFOAM upwind    must DIFFER  (the case discriminates the two schemes at
#                                                             all -- if it did not, 1 and 2 are vacuous)
#     4. brae limited  vs  brae upwind          must DIFFER  (brae's answer actually FOLLOWS the scheme
#                                                             word; before this work these two runs were
#                                                             identical at the interface by construction)
#
# and the last one is the control the old code fails: it produced 4's "differ" only from the internal
# faces, and the interface contribution to it was identically zero.
#
# THE SYMMETRY CHECK is independent of OpenFOAM entirely. pipeCyclic is a 45-degree sector of a round pipe
# closed by a ROTATIONAL cyclicAMI, so the net transverse pressure force on the walls must vanish by
# symmetry. Assembling the interface upwind breaks that symmetry -- it is the one term that does not
# respect the periodicity -- and the measured force is a direct read of how much: -1.92e-02 before,
# -2.22e-06 after, against a 1.06 axial force.
#
# THE GATE IS PER CASE, not per scheme: it runs whatever div(phi,U) the case names against upwind, so the
# same script covers `bounded Gauss limitedLinearV 1` on pipeCyclic and `bounded Gauss linear` on
# implicitAMI. The bounds are the caller's because they are properties of the CASE -- two independently
# converged steady states of a turbulent 3500-cell pipe do not agree as tightly as a laminar 24-cell duct.
set -u
SRC="${1:?case dir with a coupled interface and a non-upwind div(phi,U)}"
BOUNDS="${2:-U=5e-03,p=2e-02,k=8e-02,epsilon=5e-02,nut=8e-02}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$BRAE" ]                 || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
grep -qE '^\s*type\s+(cyclic|cyclicAMI|cyclicACMI)' "$SRC/constant/polyMesh/boundary" \
    || { echo "FAIL: $SRC has no coupled interface -- this gate would test nothing"; exit 1; }
# The case must name something OTHER than upwind, or the two runs below are the same run twice and every
# check passes while measuring nothing.
NAMED=$(grep -E '^\s*div\(phi,U\)' "$SRC/system/fvSchemes" | head -1)
case "$NAMED" in
    *upwind*) echo "FAIL: $SRC already asks for upwind ($NAMED) -- the scheme comparison would be vacuous"; exit 1;;
    *Gauss*)  ;;
    *)        echo "FAIL: cannot read a div(phi,U) Gauss scheme from $SRC/system/fvSchemes"; exit 1;;
esac
echo "  case $(basename "$SRC"):$NAMED"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
# The full OpenFOAM environment, not a hand-assembled subset of it. pipeCyclic's momentum and
# coordinate-transform functionObjects resolve $WM_OPTIONS through OpenFOAM's own etc/controlDict, so the
# PATH+LD_LIBRARY_PATH shortcut the older gates use dies here with "Unknown variable 'WM_OPTIONS'".
# `set +u` around it: OpenFOAM's bashrc reads unset variables by design, so sourcing it under -u aborts
# the script before the first run.
set +u
# shellcheck disable=SC1091
source /usr/lib/openfoam/openfoam2412/etc/bashrc > /dev/null 2>&1 || true
set -u

# One case builder, two schemes. `upwind` rewrites BOTH the momentum and the turbulence div schemes, so
# the two runs differ in the scheme and in nothing else.
mkcase()
{
    cp -r "$SRC" "$1"
    rm -rf "$1"/[1-9]* "$1"/0.[0-9]* "$1"/postProcessing "$1"/log.* "$1"/processor*
    python3 - "$1" "$2" <<'PY'
import re, sys
d, scheme = sys.argv[1], sys.argv[2]
p = d + '/system/fvSchemes'; s = open(p).read()
if scheme == 'upwind':
    s = re.sub(r'div\(phi,U\)\s+[^;]+;',  'div(phi,U)      bounded Gauss upwind;', s)
    s = re.sub(r'turbulence\s+[^;]+;',    'turbulence      bounded Gauss upwind;', s)
open(p, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'writeFormat\s+\S+;',   'writeFormat     ascii;', s)
s = re.sub(r'writePrecision\s+\S+;','writePrecision  15;', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
# RUN TO CONVERGENCE, whatever the tutorial's own endTime is. A case with residualControl stops itself; one
# without (implicitAMI ships endTime 100) would otherwise be compared MID-TRAJECTORY, and the two codes are
# not at the same point of theirs -- at iteration 100 on that case OpenFOAM is at residual 1e-04 while brae
# is at 1e-07, so the comparison measures the difference in convergence RATE and reports it as a
# discretisation error. Check 0 below asserts both actually arrived.
s = re.sub(r'^endTime .*',       'endTime         20000;', s, flags=re.M)
s = re.sub(r'^writeInterval .*', 'writeInterval   20000;', s, flags=re.M)
open(c, 'w').write(s)
PY
}

for s in limited upwind; do
    mkcase "$W/of_$s" "$s"
    ( cd "$W/of_$s" && timeout 1800 "$OFBIN/bin/simpleFoam" > log.of 2>&1 ) \
        || { echo "FAIL: OpenFOAM did not run the $s case"; tail -5 "$W/of_$s/log.of"; exit 1; }
    mkcase "$W/brae_$s" "$s"
    ( cd "$W/brae_$s" && "$BRAE" . > run.log 2>&1 ) \
        || { echo "FAIL: brae refused or crashed on the $s case"; tail -15 "$W/brae_$s/run.log"; exit 1; }
done

GATE_BOUNDS="$BOUNDS" python3 - "$W" <<'PY'
import os, re, sys
import numpy as np
W = sys.argv[1]

def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r'[0-9]+(\.[0-9]+)?', x) and x != '0']
    return None if not ts else max(ts, key=float)

def read(d, f):
    b = open(os.path.join(d, f), 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m:
        m2 = re.search(rb'internalField\s+uniform\s+([^;]+);', b)
        return None if not m2 else np.array([[float(x) for x in re.findall(rb'[-+0-9.eE]+', m2.group(1))]])
    typ = m.group(1).decode(); n = int(m.group(2)); start = m.end()
    nc = 3 if typ == 'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary':
        return np.frombuffer(b[start:start+n*nc*8], dtype='<f8').reshape(n, nc)
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

# The fields the case actually wrote -- implicitAMI is laminar and has no k/epsilon/nut, and demanding
# them would turn a legitimate case into a failure.
BOUND = dict(kv.split('=') for kv in os.environ['GATE_BOUNDS'].split(','))
BOUND = {k: float(v) for k, v in BOUND.items()}
dirs = {}
for k in ('of_limited', 'of_upwind', 'brae_limited', 'brae_upwind'):
    t = lastTime(os.path.join(W, k))
    if t is None:
        print("  FAIL: %s wrote no result" % k); sys.exit(1)
    dirs[k] = os.path.join(W, k, t)

def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

rc = 0

# 0. BOTH CODES MUST HAVE CONVERGED. Every comparison below is between steady states, and a run still
#    moving is not one. Without this the gate degrades silently into a measurement of relative convergence
#    rate -- which is exactly how this investigation first misread implicitAMI.
#    A run ends steady in one of two legitimate ways, and both count: it satisfied the case's own
#    residualControl (pipeCyclic does, at its U tolerance of 1e-03, and says so), or it ran to endTime with
#    the residual already down. Demanding a fixed residual of both would fail a case for MEETING its own
#    stated criterion, which is why this asks the question the two ways round.
print("  0. both codes reached a steady state")
for k in ('of_limited', 'of_upwind', 'brae_limited', 'brae_upwind'):
    log = os.path.join(W, k, 'log.of' if k.startswith('of') else 'run.log')
    txt = open(log, errors='ignore').read()
    stopped = 'solution converged' in txt
    res = re.findall(r'Solving for Ux, Initial residual = ([0-9.eE+-]+)', txt)
    if not res:
        print("     %-14s FAIL: no Ux residual in the log" % k); rc = 1; continue
    v = float(res[-1])
    ok = stopped or v < 1e-05
    why = "residualControl" if stopped else "ran to endTime"
    print("     %-14s Ux %.3e  (%s)   %s"
          % (k, v, why, "ok" if ok else "FAIL: hit endTime still moving, so this is a trajectory not a steady state"))
    if not ok: rc = 1

# 1 and 2: brae must match OpenFOAM under EACH scheme. The limited bounds are the loose ones -- these are
# two independently converged steady states of a turbulent case, not one step from a shared start -- but
# they are far tighter than the upwind-interface run could reach: it sat at U 5.9e-02, p 2.4e-01,
# k 2.3e-01, so a regression to the old assembly fails the U bound by a factor of 12.
FIELDS = [f for f in BOUND if os.path.exists(os.path.join(dirs['of_limited'], f))]
missing = [f for f in BOUND if f not in FIELDS]
if missing:
    print("     (not written by this case, not checked: %s)" % ', '.join(sorted(missing)))
if not FIELDS:
    print("  FAIL: none of the named fields were written"); sys.exit(1)
for s in ('limited', 'upwind'):
    print("  %d. brae vs OpenFOAM, div(phi,U) = %s" % (1 if s == 'limited' else 2,
                                                       'the scheme the case names' if s == 'limited' else 'upwind'))
    for f in FIELDS:
        e = rel(read(dirs['brae_' + s], f), read(dirs['of_' + s], f))
        ok = e < BOUND[f]
        print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL (> %.0e)" % BOUND[f]))
        if not ok: rc = 1

# 3 and 4: the two schemes must actually give different answers -- in OpenFOAM, so the comparison above
# is not vacuous, and in brae, so brae's answer is shown to follow the scheme word rather than ignore it.
print("  3. OpenFOAM limited vs OpenFOAM upwind (the case must discriminate the schemes)")
for f in FIELDS[:2]:
    e = rel(read(dirs['of_limited'], f), read(dirs['of_upwind'], f))
    ok = e > 1e-03
    print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL: the schemes agree, so 1 and 2 prove nothing"))
    if not ok: rc = 1
print("  4. brae limited vs brae upwind (brae's answer must FOLLOW the scheme word)")
for f in FIELDS[:2]:
    e = rel(read(dirs['brae_limited'], f), read(dirs['brae_upwind'], f))
    ok = e > 1e-03
    print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL: brae gives the same answer either way"))
    if not ok: rc = 1

sys.exit(rc)
PY
rc=$?

# THE SYMMETRY CHECK, for a ROTATIONAL sector only -- it is a statement about the case's geometry, not
# about the scheme, so it applies where the mesh is a wedge of a body of revolution closed by a rotational
# interface and nowhere else. brae prints the wall force itself, so this reads brae's own output and never
# consults OpenFOAM.
if grep -q 'transform *rotational' "$SRC/constant/polyMesh/boundary"; then
    FY=$(grep '^forces' "$W/brae_limited/run.log" | tail -1 | sed -n 's/.*pressure=(\([^)]*\)).*/\1/p' | awk '{print $2}')
    FZ=$(grep '^forces' "$W/brae_limited/run.log" | tail -1 | sed -n 's/.*pressure=(\([^)]*\)).*/\1/p' | awk '{print $3}')
    echo "  5. rotational symmetry: transverse pressure force $FY against axial $FZ"
    python3 -c "
import sys
fy, fz = abs(float('$FY')), abs(float('$FZ'))
r = fy/max(fz,1e-300)
ok = r < 1e-04
print('     |Fy|/|Fz| = %.3e   %s' % (r, 'ok' if ok else 'FAIL (>1e-04: the interface breaks the periodicity)'))
sys.exit(0 if ok else 1)
" || rc=1
else
    echo "  5. rotational symmetry: skipped (this interface does not transform rotationally)"
fi

[ $rc -eq 0 ] && echo "  ok:   brae assembles a coupled interface with the scheme the case names"
exit $rc
