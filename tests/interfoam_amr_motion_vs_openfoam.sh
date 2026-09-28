#!/bin/bash
# brae interFoam vs real OpenFOAM interFoam on a mesh that REFINES AND MOVES: laminar/oscillatingBox, the
# tutorial as shipped but for fixed steps and pinned solves.
#
# In v2412 dynamicRefineFvMesh IS a dynamicMotionSolverListFvMesh (dynamicRefineFvMesh.H:92-95), and this
# tutorial names both halves: refinement on alpha.water at every step (refineInterval 1, maxRefinement 2)
# and a solidBody oscillatingLinearMotion of the whole mesh under `solvers { VF { ... } }`. update() changes
# the topology FIRST and moves the points AFTER (dynamicRefineFvMesh.C:1468-1474), and each change maps the
# motion's points0 (points0MotionSolver::updateMesh, points0MotionSolver.C:152-218), stores V0 once per time
# index and corrects it for split cells (fvMesh.C:1026, dynamicRefineFvMesh.C:203-253), and recreates the
# mesh flux as zero before the move sweeps it (fvMesh.C:1057-1077).
#
# brae REFUSED this case until this unit, and before the refusal it DROPPED the motion: it refined exactly
# as OpenFOAM did and read max|U| 1.2e-04 where OpenFOAM reads 2.21.
#
# STAGING: the tutorial's own Allrun.pre (restore0Dir, blockMesh, setFields) and interFoam run SERIALLY --
# the shipped Allrun decomposes, and that is not a request: the oracle is a serial run. Twenty fixed steps
# of 1e-3 (the shipped adaptive step starts at 1.2e-05 and would take 35 steps to reach t = 0.02), every
# solve pinned to 1e-14 / relTol 0 so what is compared is the discretisation and not two Krylov stopping
# points (as shipped, OpenFOAM's own U moves 3.4e-04 between the two). OpenFOAM refines 1000 -> 2400 ->
# 8000 cells at steps 1 and 2, 8000 -> 8420 at step 11 and 8420 -> 8700 at step 17; unrefinement first
# appears at step 57 and is NOT in this profile.
#
# THE FIXTURE CAN WITNESS THE MOTION, and that is asserted on OpenFOAM against itself: the same staging with
# the `solvers` block removed (the historical silent drop) reads U 100% and alpha 5e-01 different at
# t = 0.01, the last step where the two meshes still have the same cells. And it does not amplify: one ulp
# on one alpha cell moves OpenFOAM's own alpha 5.7e-15 by step 20, so the bounds are round-off floors.
#
# WHAT ONLY THIS PROFILE HOLDS: points and points0 against OpenFOAM's written files, BITWISE, and the mesh
# flux. What it CANNOT witness, said here rather than implied: the scale factor span(points0)/span(points),
# which is exactly 1 under a translation; the V0 correction for split cells beyond round-off, since a rigid
# box keeps V == V0; unrefinement; a 2-D mesh (twoDCorrectPoints, refused); and the list-form move
# p + (q - p), which is exact on this case.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_amr_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/oscillatingBox"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: oscillatingBox tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for app in blockMesh setFields interFoam; do
    command -v "$app" > /dev/null 2>&1 || { echo "SKIP: $app not on PATH"; exit 77; }
done

DT=0.001
N=20
END=$(python3 -c "print('%.10g' % ($N*float('$DT')))")

# stage <dir> [nomo]: the tutorial's Allrun.pre, fixed steps, pinned solves; `nomo` removes the motion
stage()
{
    local C="$1" variant="${2:-}"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/0 "$C"/processor* "$C"/log.* "$C"/0.[0-9]*

    python3 - "$C" "$DT" "$END" "$variant" <<'PYEOF' || return 1
import re, sys
C, dt, end, variant = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
p = C + '/system/controlDict'
s = open(p).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('stopAt', 'endTime'),
                 ('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', end),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '18'), ('writeCompression', 'off'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(p, 'w').write(s)

p = C + '/system/fvSolution'
s = open(p).read()
s, n = re.subn(r'tolerance\s+\S+;', 'tolerance       1e-14;', s)
assert n >= 4, 'expected at least 4 tolerance entries, found %d' % n
s, m = re.subn(r'relTol\s+\S+;', 'relTol          0;', s)
assert m >= 4, 'expected at least 4 relTol entries, found %d' % m
open(p, 'w').write(s)

p = C + '/constant/dynamicMeshDict'
s = open(p).read()
assert re.search(r'dynamicFvMesh\s+dynamicRefineFvMesh;', s), 'oscillatingBox no longer refines'
assert re.search(r'motionSolver\s+solidBody;', s), 'oscillatingBox no longer moves by solidBody'
assert re.search(r'solidBodyMotionFunction\s+oscillatingLinearMotion;', s), \
    'oscillatingBox no longer oscillates linearly'
if variant == 'nomo':
    # THE HISTORICAL SILENT DROP, on OpenFOAM: the same case with no motion solver
    s, k = re.subn(r'\nsolvers\s*\{.*?\n\}\n', '\n', s, flags=re.S)
    assert k == 1, 'the solvers block was not removed'
    open(p, 'w').write(s)
PYEOF

    ( cd "$C" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) \
        || { echo "FAIL: preparing the mesh"; tail -20 "$C"/log.* 2>/dev/null; return 1; }

    python3 - "$C" <<'PYEOF' || return 1
import re, sys, os
C = sys.argv[1]
own = open(C + '/constant/polyMesh/owner').read()
n = int(re.search(r'nCells:\s*(\d+)', own).group(1))
assert n == 1000, 'oscillatingBox is %d cells, not the 1000 every count in this gate is stated against' % n
assert not os.path.exists(C + '/constant/polyMesh/cellLevel'), \
    'the fresh mesh already carries a cellLevel, so it does not start from level 0'
PYEOF
}

runOf()
{
    local C="$1" tag="$2" key
    key=$(oracleKey "$C" "interfoam_amr_motion" "$tag" "$DT" "$N")
    if oracleRestore "$C" "$key" "$END"; then
        echo "[$tag] OpenFOAM's $N steps of deltaT $DT to t = $END reused from the oracle cache"
        return 0
    fi
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam ($tag)"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory ($tag)"; return 1; }
    oracleStore "$C" "$key"
}

rc=0

G="$W/box"
stage "$G" || exit 1
runOf "$G" "box" || exit 1

# THE ORACLE TOOK THE PATH: the refinement sequence, the motion, and a pcorr that does work
for want in "Refined from 1000 to 2400 cells." "Refined from 2400 to 8000 cells." \
            "Refined from 8000 to 8420 cells." "Refined from 8420 to 8700 cells."; do
    grep -qF "$want" "$G/log.interFoam" \
        || { echo "FAIL: OpenFOAM's refinement sequence is not the one this gate is stated against"
             echo "      missing: $want"; grep -n "Refined from" "$G/log.interFoam"; rc=1; }
done
grep -q "Applying motion to entire mesh" "$G/log.interFoam" \
    || { echo "FAIL: OpenFOAM's solidBody motion is not moving the whole mesh"; rc=1; }
grep -q "Solving for pcorr, Initial residual = 1," "$G/log.interFoam" \
    || { echo "FAIL: OpenFOAM's pcorr does no work on this fixture"; rc=1; }
for f in polyMesh/points polyMesh/points0 meshPhi Uf; do
    [ -f "$G/$END/$f" ] || { echo "FAIL: OpenFOAM wrote no $END/$f, which this gate compares against"; rc=1; }
done
[ $rc -eq 0 ] || { echo "interfoam_amr_motion_vs_openfoam: rc $rc"; exit $rc; }

# THE FIXTURE CAN WITNESS THE MOTION: OpenFOAM with the motion removed, against OpenFOAM with it, at the
# last step where both meshes have the same cells (both refine 1000 -> 2400 -> 8000 by step 2)
NM="$W/nomo"
stage "$NM" nomo || exit 1
runOf "$NM" "nomo" || exit 1
python3 - "$G" "$NM" <<'PYEOF' || rc=1
import re, sys
G, NM = sys.argv[1], sys.argv[2]
def internal(path, vec):
    s = open(path).read()
    i = s.find('internalField')
    m = re.match(r'internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(', s[i:])
    n = int(m.group(1)); body = s[i + m.end():]
    if not vec:
        return [float(x) for x in body.split(None, n)[:n]]
    j = body.find('\n)')
    v = [float(x) for x in re.findall(r'[-+0-9.eE]+', body[:j])]
    return [v[3*k:3*k+3] for k in range(n)]
t = '0.01'
own = lambda c: open(c + '/' + t + '/polyMesh/owner').read()
assert re.search(r'nCells:\s*(\d+)', own(G)).group(1) == re.search(r'nCells:\s*(\d+)', own(NM)).group(1), \
    'the two meshes differ at t = 0.01'
uG = internal(G + '/' + t + '/U', True); uN = internal(NM + '/' + t + '/U', True)
aG = internal(G + '/' + t + '/alpha.water', False); aN = internal(NM + '/' + t + '/alpha.water', False)
mag = lambda v: (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
dU = max(mag([a - b for a, b in zip(x, y)]) for x, y in zip(uG, uN)) / max(mag(x) for x in uG)
dA = max(abs(a - b) for a, b in zip(aG, aN))
print('  CONTROL: OpenFOAM without the motion against OpenFOAM with it at t = 0.01: U rel %.4e, alpha %.4e'
      % (dU, dA))
if dU < 0.5:
    raise SystemExit('FAIL: removing the motion moves OpenFOAM\'s own U by only %.4e, so this fixture '
                     'cannot witness the motion' % dU)
PYEOF

"$BIN" "$G" "$G/0" "$G/$END" "$N" "$G/log.interFoam" box || rc=1

# boxUnrefine: SIXTY steps, where OpenFOAM first UNREFINES (step 57). Unrefinement removes points and merges
# cells and faces, so it is the only profile here on which points0 loses points, a merged cell's V0 is reset,
# and a carry with no internal half (U's old-time patch values) is mapped through MERGED faces -- which read an
# empty list in mapSurfaceField's interpolative branch and segfaulted until this unit.
N=60
END=$(python3 -c "print('%.10g' % ($N*float('$DT')))")
GU="$W/boxUnrefine"
stage "$GU" || exit 1
runOf "$GU" "boxUnrefine" || exit 1
grep -qF "Unrefined from 9540 to 9400 cells." "$GU/log.interFoam" \
    || { echo "FAIL: OpenFOAM does not unrefine 9540 -> 9400 in this profile, so it cannot witness removal"
         grep -n "Unrefined from" "$GU/log.interFoam"; rc=1; }
"$BIN" "$GU" "$GU/0" "$GU/$END" "$N" "$GU/log.interFoam" boxUnrefine || rc=1

echo "interfoam_amr_motion_vs_openfoam: rc $rc"
exit $rc
