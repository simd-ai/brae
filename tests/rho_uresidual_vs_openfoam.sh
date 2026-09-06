#!/usr/bin/env bash
# U'S REPORTED RESIDUAL IS cmptMax OVER THE COMPONENTS OPENFOAM SOLVES, on both mirror arms -- against
# OpenFOAM's own log, per iteration.
#
# fvMatrix<vector>::solveSegregated solves only the components polyMesh::solutionD() leaves valid
# (fvMatrixSolve.C:157-164; solutionD_ is knocked out by the EMPTY patches alone, polyMesh.C:53-59) and
# stores a per-component SolverPerformance; residualControl compares cmptMax over it
# (solutionControl.C:232). Both mirror arms reported COMPONENT 0. On rhoBox OpenFOAM's Uy initial
# residual exceeds Ux's at iterations 2, 3 and 8 (iteration 2: Ux 4.912e-01, Uy 6.043e-01; the mirror
# printed 4.9122e-01), so a residualControl on U fired at a different iteration from OpenFOAM's. And a
# max over three components SOLVED unconditionally is the opposite error: the empty direction's system
# has a ~0 right-hand side and a zero field, and its normFactor-scaled residual reads 1.000e+00 on every
# iteration (measured on rhoBox, both arms), which would block convergence on every 2D case.
#
#   ARM 1  the printed U residual, both arms, against max(Ux, Uy) from OpenFOAM's log at t=1..N.
#          brae prints %.4e, so the bound is the print's own resolution; a component-0 report misses
#          by 23% at iteration 2.
#          CONTROL: OpenFOAM's Uy must exceed Ux at some iteration by more than the bound, or the arm
#          could not tell cmptMax from component 0.
#   ARM 2  residualControl on U alone (p and h set to 1 so they never bind), at a threshold between the
#          two sequences: OpenFOAM stops at iteration N_OF, and both mirror arms must stop at the SAME
#          iteration. CONTROL: computed from OpenFOAM's own log, the iteration a component-0 report
#          would have stopped at must differ from N_OF, or the threshold is inert.
#
# Fail-proof, 2026-09-03: with both steps reporting component 0 again, arm 1 FAILS at iteration 2
# (host and device 4.9122e-01 against OpenFOAM's 6.0426e-01, 1.9e-01 relative) and arm 2 stops early.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
SRC="$ROOT/validation/rhoBox"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-10}
# brae prints the residual with four decimals (%.4e): 5e-05 relative resolution; the bound is 4x that.
# Measured max over t=1..10, both arms: 2.3e-05.
U_RES_BOUND=${U_RES_BOUND:-2e-04}
# The residualControl threshold for arm 2. Pinned from OpenFOAM's rhoBox sequences (see the control).
U_RC=${U_RC:-0.55}

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
if [ -f "$W/base/system/blockMeshDict" ] && command -v blockMesh > /dev/null 2>&1; then
    ( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on rhoBox"; exit 1; }
fi

stage()   # $1 dst  $2 residualControl body ("" = none)
{
    RC="$2" python3 - "$W/base" "$1" "$N" <<'URESSTAGE'
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
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { %s }' % os.environ['RC'], s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0; maxIter 2000;', s)
open(f, 'w').write(s)
URESSTAGE
}

run()   # $1 dir  $2 arm (of|1|cuda)
{
    case "$2" in
        of)   ( cd "$1" && rhoSimpleFoam > run.log 2>&1 ) ;;
        *)    ( cd "$1" && BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BIN" -case "$1" > run.log 2>&1 ) ;;
    esac || { tail -5 "$1/run.log"; echo "FAIL: $2 did not run in $1"; exit 1; }
}

# ---- arm 1: the printed residual against OpenFOAM's per-component log -----------------------------
stage "$W/of"   "";  stage "$W/host" "";  stage "$W/cuda" ""
run "$W/of" of; run "$W/host" 1; run "$W/cuda" cuda

W="$W" N="$N" U_RES_BOUND="$U_RES_BOUND" U_RC="$U_RC" python3 - <<'URESCMP' || fail=1
import os, re, sys
W, N = os.environ['W'], int(os.environ['N'])
bound, rc = float(os.environ['U_RES_BOUND']), float(os.environ['U_RC'])

def of_seq(path):
    # Per Time: the initial residual of every U component OpenFOAM SOLVED (it prints only those).
    seq, comps, cur = {}, {}, None
    for line in open(path):
        m = re.match(r'^Time = (\d+)\s*$', line)
        if m:
            cur = int(m.group(1)); continue
        m = re.search(r'Solving for U([xyz]), Initial residual = ([0-9.eE+-]+)', line)
        if m and cur is not None:
            comps.setdefault(cur, {})[m.group(1)] = float(m.group(2))
    for t, c in comps.items():
        seq[t] = max(c.values())
    return seq, comps

def brae_seq(path):
    seq = {}
    for line in open(path):
        m = re.match(r'^Time = (\d+)\s+U ([0-9.eE+-]+)', line)
        if m:
            seq[int(m.group(1))] = float(m.group(2))
    return seq

of, comps = of_seq(os.path.join(W, 'of', 'run.log'))
ok = True
print('     t   OF Ux        OF Uy        OF max       host         cuda')
worst = 0.0
for t in range(1, N + 1):
    if t not in of:
        print('     %2d  OpenFOAM printed no U residual   FAIL' % t); ok = False; continue
    row = '     %2d  %.4e   %.4e   %.4e' % (t, comps[t].get('x', float('nan')), comps[t].get('y', float('nan')), of[t])
    for arm in ('host', 'cuda'):
        b = brae_seq(os.path.join(W, arm, 'run.log'))
        if t not in b:
            row += '   MISSING'; ok = False; continue
        r = abs(b[t] - of[t]) / of[t]
        worst = max(worst, r)
        row += '   %.4e' % b[t]
        if r >= bound:
            row += ' FAIL(%.1e)' % r; ok = False
    print(row)
print('     max relative gap over t=1..%d, both arms: %.3e   (bound %.1e)   %s' % (N, worst, bound, 'ok' if worst < bound else 'FAIL'))
# CONTROL: the two components must be told apart by more than the bound at some iteration.
sep = max((comps[t]['y'] - comps[t]['x']) / of[t] for t in of if 'y' in comps[t] and 'x' in comps[t])
print('     control: OpenFOAM Uy exceeds Ux by %.3e relative at its best iteration (needs > 10x the bound)   %s'
      % (sep, 'ok' if sep > 10 * bound else 'FAIL (inert)'))
ok = ok and sep > 10 * bound
# Arm 2's control, from OpenFOAM's own log: first iteration at which each sequence drops below U_RC.
def first_below(s):
    for t in sorted(s):
        if s[t] < rc: return t
    return None
n_max = first_below(of)
n_x   = first_below({t: comps[t]['x'] for t in comps if 'x' in comps[t]})
print('     control: at residualControl U %.3g, cmptMax first drops below at t=%s, component 0 at t=%s   %s'
      % (rc, n_max, n_x, 'ok' if n_max is not None and n_x != n_max else 'FAIL (inert threshold)'))
ok = ok and n_max is not None and n_x != n_max
open(os.path.join(W, 'n_expected'), 'w').write(str(n_max))
sys.exit(0 if ok else 1)
URESCMP
say "arm 1: the printed U residual is cmptMax over the solved components, both arms" "$([ $fail = 0 ] && echo ok || echo FAIL)"
[ $fail = 0 ] || { echo FAIL; exit 1; }

# ---- arm 2: residualControl on U stops at OpenFOAM's iteration, both arms -------------------------
# OpenFOAM's simpleControl checks the criteria at the top of the loop against the previous iteration's
# residuals, so the run ends one iteration after the first below-threshold report; the mirror's stop
# is compared as the LAST `Time =` each log printed.
RC="p 1; h 1; U $U_RC;"
stage "$W/of_rc" "$RC"; stage "$W/host_rc" "$RC"; stage "$W/cuda_rc" "$RC"
run "$W/of_rc" of; run "$W/host_rc" 1; run "$W/cuda_rc" cuda
last() { grep -E '^Time = [0-9]+' "$1/run.log" | tail -1 | sed -E 's/^Time = ([0-9]+).*/\1/'; }
n_of=$(last "$W/of_rc"); n_host=$(last "$W/host_rc"); n_cuda=$(last "$W/cuda_rc")
grep -q "converged" "$W/of_rc/run.log" || say "arm 2: OpenFOAM reports convergence under the U criterion" FAIL
[ "$n_of" -lt "$N" ] || say "arm 2: OpenFOAM stopped before endTime ($n_of)" FAIL
[ "$n_host" = "$n_of" ] && say "arm 2: the host arm stops at OpenFOAM's iteration ($n_host = $n_of)" ok \
    || say "arm 2: the host arm stops at OpenFOAM's iteration (host $n_host, OpenFOAM $n_of)" FAIL
[ "$n_cuda" = "$n_of" ] && say "arm 2: the device arm stops at OpenFOAM's iteration ($n_cuda = $n_of)" ok \
    || say "arm 2: the device arm stops at OpenFOAM's iteration (device $n_cuda, OpenFOAM $n_of)" FAIL

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
