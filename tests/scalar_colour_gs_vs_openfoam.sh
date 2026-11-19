#!/usr/bin/env bash
# The colour-order smoothSolver on the ENERGY field and the turbulence pair (FP-1, the default since
# 2026-09-12) against real OpenFOAM -- the scalar counterpart of tests/u_colour_gs_vs_openfoam.sh.
#
# Where a case names `smoothSolver` with a GaussSeidel-family smoother on e/h, k, epsilon or omega, the
# rhoSimpleFoam mirror's CUDA arm now sweeps those systems in COLOUR order through the momentum engine
# with one component (deviceColourGaussSeidelFused, nComp 1; tests/test_colour_gs_fused arm (m)) under
# smoothSolver::solve's stop rule, where OpenFOAM sweeps in index order. Gauss-Seidel is order-dependent:
# the two leave different iterates after n sweeps and converge to the same linear solution. That is what
# this gate holds the change to: EXACT where the linear solves are converged, DIFFERENT where they are
# not, and ANNOUNCED per field either way. Before FP-1 those entries ran OpenFOAM's own order on one CPU
# thread (the host smoother); the REF arm runs that order, on the device loop since 2026-11-19.
#
#   FIXTURE     validation/rhoKE (3,200 cells, kEpsilon, 2D), blockMesh at staging, every linear solver
#               at 1e-14 / relTol 0, 5 iterations. The energy and the pair get smoothSolver entries in
#               BOTH codes -- symGaussSeidel on "(h|e)", GaussSeidel on "(k|epsilon)", so both of
#               OpenFOAM's smoothers are exercised -- with the relTol/maxIter each arm names below. U stays
#               on the fixture's PBiCGStab/DILU entry and is pinned to OpenFOAM's own solver
#               (BRAE_U_SOLVER=ofOrder) in EVERY brae arm, so the momentum contributes no order effect and
#               BRAE_GS_ORDER alone selects what this gate measures.
#   ARM EXACT   relTol 0, maxIter 5000, BRAE_GS_ORDER=colour: every scalar solve converges to roundoff in
#               both codes, so U, p, T, k and epsilon at 5 must agree within BOUND (rel L2).
#   ARM REF     the same case under BRAE_GS_ORDER=ofOrder (OpenFOAM's own index-order sweep): must also
#               pass BOUND, so the bound is one this fixture meets and EXACT's pass is not a loose bound.
#   ARM SAID    EXACT carries the colour-order notice for e (or h), k and epsilon and the start-up line;
#               REF carries none of them and announces the smoothSolver path it took instead.
#   ARM DEFAULT the same case with NEITHER variable set (BRAE_U_SOLVER unset too): the colour notices
#               must be present, which proves the default; its fields are compared at the momentum
#               gate's DEF bound since U runs colour order there as well.
#   CONTROL     relTol 0.1 on the three scalar entries in BOTH codes: the two orders stop at different
#               iterates, so the fields at 5 must differ by MORE than BOUND. This proves the comparison
#               can see a solver difference, and documents the approximation the notices announce.
#   FAIL-PROOF  colour order with maxIter 1 on the scalars in brae ONLY (OpenFOAM keeps 5000): one colour
#               sweep per outer iteration against a converged solve must miss BOUND.
#   BOUND       1e-9, the momentum gate's bound on this fixture; every arm prints its numbers.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/rhoKE}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
BOUND=${BOUND:-1e-9}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM at $OFBASHRC"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-92s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
N=5
stage() {   # stage <dir> <relTol> <maxIter>
    local d=$1 rel=$2 mx=$3
    rm -rf "$d"; cp -r "$SRC" "$d"; rm -rf "$d"/[1-9]* 2>/dev/null; [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    ( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$d" && blockMesh > log.blockMesh 2>&1 ) || { tail -3 "$d/log.blockMesh"; echo "FAIL: blockMesh"; exit 1; }
    python3 - "$d" "$N" "$rel" "$mx" <<'PY'
import re, sys
d, n, rel, mx = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
s = re.sub(r'functions\s*\{.*?\n\}', '', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'tolerance\s+[^;]*;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[^;]*;', 'relTol 0;', s)
# U keeps the fixture's PBiCGStab/DILU (tight); the energy and the pair get smoothSolver entries
new = ('U { solver PBiCGStab; preconditioner DILU; tolerance 1e-14; relTol 0; }\n'
       '    "(h|e)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-14; relTol %s; maxIter %s; }' % (rel, mx))
s, k = re.subn(r'"\(U\|h\|e\)"\s*\{[^}]*\}', new, s)
assert k == 1, 'expected one "(U|h|e)" block, found %d' % k
newke = '"(k|epsilon)" { solver smoothSolver; smoother GaussSeidel; tolerance 1e-14; relTol %s; maxIter %s; }' % (rel, mx)
s, k = re.subn(r'"\(k\|epsilon\)"\s*\{[^}]*\}', newke, s)
assert k == 1, 'expected one "(k|epsilon)" block, found %d' % k
open(f, 'w').write(s)
PY
}
run() {   # run <dir> <of|brae> [VAR=value ...]
    local d=$1 sel=$2; shift 2
    if [ "$sel" = of ]; then ( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$d" && rhoSimpleFoam > run.log 2>&1 )
    else ( cd "$d" && env BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$@" "$BRAE" -case "$d" > run.log 2>&1 ); fi
}
stage "$W/of_exact" 0 5000;   run "$W/of_exact" of || { tail -5 "$W/of_exact/run.log"; echo "FAIL: OpenFOAM exact"; exit 1; }
stage "$W/of_ctrl"  0.1 5000; run "$W/of_ctrl"  of || { tail -5 "$W/of_ctrl/run.log";  echo "FAIL: OpenFOAM control"; exit 1; }
stage "$W/brae_exact" 0 5000;   run "$W/brae_exact" brae BRAE_U_SOLVER=ofOrder BRAE_GS_ORDER=colour  || { tail -5 "$W/brae_exact/run.log"; say "EXACT       the colour run finished" FAIL; }
stage "$W/brae_ref"   0 5000;   run "$W/brae_ref"   brae BRAE_U_SOLVER=ofOrder BRAE_GS_ORDER=ofOrder || { tail -5 "$W/brae_ref/run.log";   say "REF         the index-order run finished" FAIL; }
stage "$W/brae_dflt"  0 5000;   run "$W/brae_dflt"  brae                                             || { tail -5 "$W/brae_dflt/run.log";  say "DEFAULT     the run with nothing set finished" FAIL; }
stage "$W/brae_ctrl"  0.1 5000; run "$W/brae_ctrl"  brae BRAE_U_SOLVER=ofOrder BRAE_GS_ORDER=colour  || { tail -5 "$W/brae_ctrl/run.log";  say "CONTROL     the colour relTol 0.1 run finished" FAIL; }
stage "$W/brae_fp"    0 1;      run "$W/brae_fp"    brae BRAE_U_SOLVER=ofOrder BRAE_GS_ORDER=colour  || { tail -5 "$W/brae_fp/run.log";    say "FAIL-PROOF  the colour maxIter 1 run finished" FAIL; }

has() { grep -q -- "$2" "$W/$1/run.log"; }
HE=$(grep -o "brae NOTICE \[approximated\] solvers/[he] smoother" "$W/brae_exact/run.log" | head -1 | sed 's/.*solvers\///; s/ smoother//')
[ -n "$HE" ] || HE=e
for f in "$HE" k epsilon; do
    smoo=$([ "$f" = "$HE" ] && echo symGaussSeidel || echo GaussSeidel)
    has brae_exact "brae NOTICE \[approximated\] solvers/$f smoother: case asks '$smoo' in OpenFOAM's index order; brae sweeps in COLOUR order" \
        && say "SAID  EXACT carries the colour-order notice for $f ($smoo)" ok \
        || say "SAID  EXACT carries the colour-order notice for $f ($smoo)" FAIL
    has brae_dflt "brae NOTICE \[approximated\] solvers/$f smoother: case asks '$smoo' in OpenFOAM's index order; brae sweeps in COLOUR order" \
        && say "DEFAULT  with nothing set, $f is swept in colour order (the notice proves the default)" ok \
        || say "DEFAULT  with nothing set, $f is swept in colour order (the notice proves the default)" FAIL
    has brae_ref "solvers/$f smoother" \
        && say "SAID  REF (BRAE_GS_ORDER=ofOrder) carries no colour-order notice for $f" FAIL \
        || say "SAID  REF (BRAE_GS_ORDER=ofOrder) carries no colour-order notice for $f" ok
done
has brae_exact "multicolour Gauss-Seidel smoothSolver, [0-9]* colours, the momentum colouring" \
    && say "SAID  EXACT prints the start-up line naming the fields and the colouring" ok \
    || say "SAID  EXACT prints the start-up line naming the fields and the colouring" FAIL
has brae_ref "smoothSolver: host smoother\|smoothSolver: device" \
    && say "SAID  REF announces the index-order smoothSolver path it took instead" ok \
    || say "SAID  REF announces the index-order smoothSolver path it took instead" FAIL
for arm in brae_exact brae_ref brae_dflt; do
    if has $arm "solvers/$HE solver: case asks .*brae runs PBiCGStab" || has $arm "solvers/k solver: case asks .*brae runs PBiCGStab"; then
        say "SAID  $arm: the shared reader printed no contradictory PBiCGStab line for the scalar entries" FAIL
    else
        say "SAID  $arm: the shared reader printed no contradictory PBiCGStab line for the scalar entries" ok
    fi
done
# The OpenFOAM side of the premise: every scalar solve driven to roundoff in the EXACT case.
python3 - "$W/of_exact/run.log" <<'PY' && say "EXACT  OpenFOAM's smoothSolvers drove every e/h, k and epsilon solve below 1e-12" ok \
                                 || say "EXACT  OpenFOAM's smoothSolvers drove every e/h, k and epsilon solve below 1e-12" FAIL
import re, sys
s = open(sys.argv[1]).read()
rows = re.findall(r'smoothSolver:\s+Solving for (?:e|h|k|epsilon), Initial residual = ([-+0-9.eE]+), Final residual = ([-+0-9.eE]+), No Iterations (\d+)', s)
assert rows, 'no smoothSolver e/h/k/epsilon lines in the OpenFOAM log'
worst = max(float(r[1]) for r in rows); most = max(int(r[2]) for r in rows)
print('  OpenFOAM smoothSolver on e/h, k, epsilon: %d solves, worst final residual %.3e, most sweeps %d' % (len(rows), worst, most))
sys.exit(0 if worst < 1e-12 else 1)
PY

python3 - "$W" "$N" "$BOUND" <<'PY' || fail=1
import re, sys, os, numpy as np
W, N, BOUND = sys.argv[1], sys.argv[2], float(sys.argv[3])
def internal(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m:
        m2 = re.search(rb'internalField\s+uniform\s+([^;]+);', b)
        return np.array([float(x) for x in re.findall(r'[-+0-9.eE]+', m2.group(1).decode())])
    typ, n, start = m.group(1).decode(), int(m.group(2)), m.end(); nc = 3 if typ == 'vector' else 1
    txt = b[start:].decode('latin-1'); vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0])
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)
def rel(a, b):
    x, y = internal(a), internal(b)
    if x.shape != y.shape: return float('nan')
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))
fail = 0
def arm(name, d, ref, mustPass, bound=BOUND):
    global fail
    out = []
    worst = 0.0
    for f in ('U', 'p', 'T', 'k', 'epsilon'):
        e = rel('%s/%s/%s/%s' % (W, d, N, f), '%s/%s/%s/%s' % (W, ref, N, f))
        out.append('%s %.3e' % (f, e)); worst = max(worst, e)
    ok = (worst < bound) if mustPass else (worst > bound)
    if not ok: fail = 1
    print('  %-11s %s   worst %.3e %s %g   %s' % (name, '  '.join(out), worst, '<' if mustPass else '>', bound, 'ok' if ok else 'FAIL'))
arm('EXACT', 'brae_exact', 'of_exact', True)
arm('REF', 'brae_ref', 'of_exact', True)
arm('DEFAULT', 'brae_dflt', 'of_exact', True)
arm('CONTROL', 'brae_ctrl', 'of_exact', False)
arm('FAIL-PROOF', 'brae_fp', 'of_exact', False)
# and the documented size of the approximation on OpenFOAM's own side: its loose solve vs its converged one
out = []
for f in ('U', 'p', 'T', 'k', 'epsilon'):
    out.append('%s %.3e' % (f, rel('%s/of_ctrl/%s/%s' % (W, N, f), '%s/of_exact/%s/%s' % (W, N, f))))
print('  OpenFOAM relTol 0.1 vs OpenFOAM converged at %s: %s' % (N, '  '.join(out)))
sys.exit(fail)
PY
echo
[ "$fail" = 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"
