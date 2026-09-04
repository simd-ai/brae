#!/usr/bin/env bash
# WHICH U COMPONENTS THE INCOMPRESSIBLE V2 DRIVER SOLVES, AND WHICH RESIDUAL IT REPORTS.
#
# fvMatrix<vector>::solveSegregated walks the components and `continue`s on every one that
# polyMesh::solutionD() knocks out (fvMatrixSolve.C:164; fvMesh::validComponents<vector>() IS solutionD,
# fvMeshTemplates.C:33-44). solutionD_ is knocked out by the EMPTY patches alone (polyMesh.C:75-118), so
# on a 2D case OpenFOAM never solves the empty direction and prints no `Solving for Uz` line. What
# residualControl then compares is cmptMax over the stored per-component vector (solutionControl.C:232,
# simpleControl.C:67-71), in which a skipped component is Zero.
#
# The V2 step did neither: it solved all three and reported component 0. The two errors hid each other --
# a max over three components solved unconditionally is the OPPOSITE error, because the empty direction's
# system has a ~0 right-hand side and a ~0 field and its normFactor-scaled residual never leaves O(0.1).
#
#   ARM 1  THE MASK IS APPLIED TO THE REPORT. brae still SOLVES the knocked-out component -- a known
#          deviation with its own queue entry, because brae's z quantities are round-off nonzero where
#          OpenFOAM's are bit-exactly zero (emptyFvPatch::size() is 0) and the z solve is what holds Uz
#          down; dropping it took Uz to 13% of |U| on pitzDaily by iteration 200. So the arm asserts what
#          IS true today: the reported residual is the max over the SOLVED-BY-OPENFOAM components and
#          ignores the empty direction, whose own residual is O(0.1) and would dominate.
#   ARM 2  cmptMax. Both codes take ONE iteration from OpenFOAM's OWN field at iteration 12, so the
#          trajectories cannot have drifted, and brae's reported U residual must be OpenFOAM's
#          max(Ux,Uy). CONTROL: OpenFOAM's Uy/Ux at that step must exceed 1.5, or the arm could not tell
#          cmptMax from component 0. Measured 2026-09-04: Ux 2.68734607717799e-02,
#          Uy 5.13410544976933e-02, ratio 1.910 -- a component-0 report reads 47.7% low.
#   ARM 3  THE MASK IS DERIVED, not hardcoded off. The same binary on a 3D fixture (validation/cav3d_of,
#          no empty patch) must solve ALL THREE and must not print the knock-out line.
#
# Fail-proof, 2026-09-04: with `if (in.solutionD[k] < 0) continue;` removed, arm 1 fails (brae prints a
# [U2] line OpenFOAM has no counterpart for). With `res["U"]` back to component 0, arm 2 fails at 4.8e-01
# relative against a 2e-06 bound.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/simpleBoxIO}"
SRC3D="${2:-$ROOT/validation/cav3d_of}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SEED=${SEED:-12}
# brae prints the residual with six decimals (%.6e): 5e-07 relative resolution; the bound is 4x that.
U_RES_BOUND=${U_RES_BOUND:-2e-06}
# Below this the arm cannot distinguish cmptMax from component 0.
CONTROL_RATIO=${CONTROL_RATIO:-1.5}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"; SRC3D="$(cd "$SRC3D" && pwd)"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# --- stage the 2D fixture and drive OpenFOAM to iteration $SEED -------------------------------------
cp -r "$SRC" "$W/seed"; rm -rf "$W"/seed/[1-9]* "$W"/seed/0 "$W"/seed/log.*
cp -r "$W/seed/0.orig" "$W/seed/0"
if [ -f "$W/seed/system/blockMeshDict" ] && command -v blockMesh > /dev/null 2>&1; then
    ( cd "$W/seed" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
fi
python3 - "$W/seed" "$SEED" <<'PY'
import re, sys
d, n = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
# simpleBoxIO's controlDict is written on ONE line, so these must not be line-anchored.
s = re.sub(r'endTime\s+\S+;',        'endTime         %s;' % n, s)
s = re.sub(r'writeInterval\s+\S+;',  'writeInterval   %s;' % n, s)
s = re.sub(r'writePrecision\s+\S+;', 'writePrecision  15;',     s)
s = re.sub(r'writeCompression\s+\S+;', 'writeCompression off;', s)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
open(f, 'w').write(re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S))
PY
( cd "$W/seed" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM did not run the seed"; exit 1; }
[ -d "$W/seed/$SEED" ] || { echo "FAIL: OpenFOAM wrote no $SEED/"; exit 1; }

# --- one iteration each, from that identical state --------------------------------------------------
for side in of br; do
    mkdir -p "$W/$side/$SEED"
    cp -r "$W/seed/constant" "$W/seed/system" "$W/$side/"
    cp "$W/seed/$SEED"/[A-Za-z]* "$W/$side/$SEED/" 2>/dev/null || true
    rm -rf "$W/$side/$SEED/uniform"
    # brae's V2 driver counts ITERATIONS in endTime where OpenFOAM steps from startTime to endTime, so
    # `one iteration` is spelt differently on the two sides. That is itself a divergence and is queued;
    # writing it out here keeps this gate measuring the residual rule and not that.
    python3 - "$W/$side" "$SEED" "$side" <<'PY'
import re, sys
d, n, side = sys.argv[1], sys.argv[2], sys.argv[3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'startTime\s+\S+;', 'startTime       %s;' % n, s)
s = re.sub(r'endTime\s+\S+;',   'endTime         %s;' % (str(int(n) + 1) if side == 'of' else '1'), s)
open(c, 'w').write(s)
PY
done
( cd "$W/of" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM did not take its step"; exit 1; }
( cd "$W/br" && BRAE_SIMPLEFOAM_V2=1 BRAE_SOLVER_ITERS=1 "$BRAE" "$W/br" > run.log 2>&1 ) \
    || { echo "FAIL: brae crashed"; tail -15 "$W/br/run.log"; exit 1; }

# --- the 3D control ---------------------------------------------------------------------------------
if [ -d "$SRC3D" ]; then
    cp -r "$SRC3D" "$W/br3d"; rm -rf "$W"/br3d/[1-9]* "$W"/br3d/log.*
    [ -d "$W/br3d/0.orig" ] && cp -r "$W/br3d/0.orig" "$W/br3d/0"
    python3 - "$W/br3d" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'endTime\s+\S+;', 'endTime         1;', s)
open(c, 'w').write(s)
PY
    ( cd "$W/br3d" && BRAE_SIMPLEFOAM_V2=1 BRAE_SOLVER_ITERS=1 "$BRAE" "$W/br3d" > run.log 2>&1 ) \
        || { echo "FAIL: brae crashed on the 3D control"; tail -15 "$W/br3d/run.log"; exit 1; }
fi

python3 - "$W" "$U_RES_BOUND" "$CONTROL_RATIO" <<'PY'
import os, re, sys
W, bound, ctrl = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ofl = open(os.path.join(W, 'of', 'run.log')).read()
brl = open(os.path.join(W, 'br', 'run.log')).read()
fail = 0
def say(msg, ok):
    global fail
    print('  %-70s %s' % (msg, 'ok' if ok else 'FAIL'))
    if not ok: fail = 1

# ARM 1 -- the report is the max over the components OpenFOAM solved, and excludes the empty direction.
ofset = set(re.findall(r'Solving for U([xyz]),', ofl))
per = dict((int(k), float(v)) for k, v in re.findall(r'\[U(\d)\] nIter=\d+ init=([0-9.eE+-]+)', brl))
br = float(re.search(r'U initial residual = ([0-9.eE+-]+)', brl).group(1))
say('ARM 1  OpenFOAM solved {%s}; brae reports max over its x,y and not z (z = %.3e)'
    % (','.join(sorted(ofset)), per.get(2, float('nan'))),
    ofset == {'x', 'y'} and 2 in per and abs(br - max(per[0], per[1])) <= 2e-3 * max(per[0], per[1])
    and per[2] > max(per[0], per[1]))
say('ARM 1  brae names the knocked-out direction in its log',
    'empty patches knock out a solution direction' in brl)

# ARM 2 -- cmptMax, from the identical state.
ux = float(re.search(r'Solving for Ux, Initial residual = ([0-9.eE+-]+)', ofl).group(1))
uy = float(re.search(r'Solving for Uy, Initial residual = ([0-9.eE+-]+)', ofl).group(1))
rel = abs(br - max(ux, uy)) / max(ux, uy)
print('         OpenFOAM Ux %.8e  Uy %.8e  cmptMax %.8e   brae %.8e' % (ux, uy, max(ux, uy), br))
say('ARM 2  brae reports OpenFOAM cmptMax   rel %.3e (bound %.0e)' % (rel, bound), rel < bound)
r = max(ux, uy) / min(ux, uy)
say('CONTROL  Uy/Ux = %.3f >= %.2f, so component 0 would read %.1f%% low'
    % (r, ctrl, 100.0 * (1.0 - min(ux, uy) / max(ux, uy))), r >= ctrl)

# ARM 3 -- the 3D control.
p3 = os.path.join(W, 'br3d', 'run.log')
if os.path.exists(p3):
    b3 = open(p3).read()
    p3v = dict((int(k), float(v)) for k, v in re.findall(r'\[U(\d)\] nIter=\d+ init=([0-9.eE+-]+)', b3))
    b3r = float(re.search(r'U initial residual = ([0-9.eE+-]+)', b3).group(1))
    say('ARM 3  3D fixture: brae reports the max over ALL THREE (no direction masked)',
        len(p3v) == 3 and abs(b3r - max(p3v.values())) <= 2e-3 * max(p3v.values()))
    say('ARM 3  3D fixture: no direction is knocked out',
        'empty patches knock out a solution direction' not in b3)
else:
    print('  ARM 3  skipped: no 3D fixture')
sys.exit(1 if fail else 0)
PY
rc=$?
[ $rc -eq 0 ] && echo "PASS: the V2 driver reports cmptMax over the components OpenFOAM solves"
exit $rc
