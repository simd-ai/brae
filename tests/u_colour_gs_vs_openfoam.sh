#!/usr/bin/env bash
# The multicolour Gauss-Seidel momentum solver (the DEFAULT since 2026-09-08) against real OpenFOAM.
#
# The rhoSimpleFoam mirror's CUDA arm solves the momentum equation with a
# multicolour Gauss-Seidel smoothSolver: smoothSolver::solve's stop rule (smoothSolver.C:159-209) over
# GaussSeidelSmoother.C:104-173's cell update, sweeping the cells in COLOUR order where OpenFOAM sweeps
# them in index order. Gauss-Seidel is order-dependent, so the two leave different iterates after n
# sweeps (tests/gs_ladder: 1.36x / 2.76x / 6.88x behind after 1 / 5 / 10 sweeps on T3A) and converge
# to the same linear solution. That is exactly what this gate holds the experiment to: EXACT where the
# linear solve is converged, DIFFERENT where it is not, and ANNOUNCED either way.
#
#   FIXTURE     validation/rhoKE (3,200 cells, kEpsilon, 2D: Uz is knocked out by the empty patch and
#               not solved), blockMesh at staging, every linear solver at 1e-14 / relTol 0, 5 iterations.
#               U gets its OWN entry, split out of the fixture's "(U|h|e)" block:
#                   U { solver smoothSolver; smoother GaussSeidel; tolerance 1e-14; relTol R; maxIter M; }
#               in BOTH codes, so OpenFOAM runs its GaussSeidel smoothSolver against brae's colour one.
#   ARM EXACT   relTol 0, maxIter 5000: both orders converge each momentum solve to roundoff (1e-14, or
#               the Ux floor above it -- see MEASURED), so U, p, k and epsilon at 5 must agree within
#               BOUND (rel L2) -- brae with no BRAE_U_SOLVER set, i.e. the default.
#   ARM REF     the same case under BRAE_U_SOLVER=ofOrder (the opt-out, which on this entry runs OpenFOAM's
#               own index-order sweep, level-scheduled): must also pass BOUND, so the bound is one this
#               fixture can meet and the EXACT arm's pass is not a loose bound.
#   ARM SAID    the EXACT arm runs with NO BRAE_U_SOLVER set, so its lines prove the DEFAULT: the
#               colour-order notice, the start-up line, and a `Solving
#               for Ux` line (BRAE_OF_LOG=1) named colourGaussSeidel; the REF arm's log carries none of
#               those and names its own sweep. Neither carries the shared reader's `brae runs PBiCGStab`
#               or `not running a smoothSolver` lines for U: exactly one truthful set of lines.
#   ARM KRYLOV  brae-only: the fixture's ORIGINAL "(U|h|e)" PBiCGStab/DILU entry (pinned tight) under
#               colourGS must announce `solvers/U solver: case asks 'PBiCGStab'` and nothing
#               contradictory about U's preconditioner or smoother.
#   CONTROL     relTol 0.1 on U in BOTH codes: the two orders stop at different iterates, so the fields
#               at 5 must differ by MORE than BOUND. This proves the comparison can see a solver
#               difference at all, and documents the approximation the notice announces.
#   FAIL-PROOF  colourGS with U's maxIter 1 in brae ONLY (OpenFOAM keeps 5000): one colour sweep per
#               outer iteration against a converged solve must miss BOUND.
#   BOUND       1e-9, from tests/wall_coeffs_vs_openfoam.sh on the same fixture at the same N (its CUDA
#               arm tracks OpenFOAM inside that); every arm prints its numbers.
#   MEASURED    OpenFOAM side, 2026-09-07: at tolerance 1e-14 Uy converges in 29-31 sweeps (final 4e-15
#               to 9e-15) but Ux sits on a ROUNDOFF FLOOR above the tolerance -- 28 sweeps reach 1e-12,
#               the full 5000-sweep cap reaches only 2.9e-14 to 7.0e-13 -- so the Ux solve takes the cap
#               in both codes, converged to roundoff. The premise check below therefore asserts the
#               floor (< 1e-12), not the sweep count. OpenFOAM relTol 0.1 vs OpenFOAM converged at 5:
#               U 1.97e-04, p 2.31e-04, k 9.37e-05, epsilon 1.49e-04 -- what the CONTROL has to see.
#               brae side (first run, 2026-09-07), rel L2 on U / p / k / epsilon at 5:
#                 EXACT colourGS   9.363e-13 / 1.185e-12 / 6.858e-13 / 5.054e-13
#                 REF today's path 8.790e-13 / 1.198e-12 / 6.269e-13 / 4.686e-13
#                 CONTROL relTol 0.1     2.029e-04 / 1.550e-04 / 3.001e-04 / 5.452e-04  (> BOUND, as it must)
#                 FAIL-PROOF maxIter 1   3.637e-03 / 4.036e-03 / 4.436e-03 / 9.288e-03  (> BOUND, as it must)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/rhoKE}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM at $OFBASHRC"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-86s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
N=5
stage() {   # stage <dir> <gs <relTolU> <maxIterU> | krylov>
    local d=$1 mode=$2 rel=${3:-0} mx=${4:-5000}
    rm -rf "$d"; cp -r "$SRC" "$d"; rm -rf "$d"/[1-9]* 2>/dev/null; [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    ( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$d" && blockMesh > log.blockMesh 2>&1 ) || { tail -3 "$d/log.blockMesh"; echo "FAIL: blockMesh"; exit 1; }
    python3 - "$d" "$N" "$mode" "$rel" "$mx" <<'PY'
import re, sys
d, n, mode, rel, mx = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
s = re.sub(r'functions\s*\{.*?\n\}', '', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
# every solver tight FIRST, so the U entry written below keeps its own relTol
s = re.sub(r'tolerance\s+[^;]*;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[^;]*;', 'relTol 0;', s)
if mode == 'gs':
    # U out of the "(U|h|e)" block into its own entry: a literal key wins over a regex one in both codes
    new = ('"(h|e)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-14; relTol 0; }\n'
           '    U { solver smoothSolver; smoother GaussSeidel; tolerance 1e-14; relTol %s; maxIter %s; }' % (rel, mx))
    s, k = re.subn(r'"\(U\|h\|e\)"\s*\{[^}]*\}', new, s)
    assert k == 1, 'expected one "(U|h|e)" block, found %d' % k
else:
    assert re.search(r'"\(U\|h\|e\)"\s*\{[^}]*PBiCGStab[^}]*DILU', s), 'the fixture no longer names PBiCGStab/DILU on U'
open(f, 'w').write(s)
PY
}
run() {   # run <dir> <of|brae> [VAR=value ...]
    local d=$1 sel=$2; shift 2
    if [ "$sel" = of ]; then ( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$d" && rhoSimpleFoam > run.log 2>&1 )
    else ( cd "$d" && env BRAE_RHOSIMPLEFOAM_MIRROR=cuda BRAE_OF_LOG=1 "$@" "$BRAE" -case "$d" > run.log 2>&1 ); fi
}
# OpenFOAM: the converged solve and the loose one.
stage "$W/of_exact" gs 0 5000;   run "$W/of_exact" of || { tail -5 "$W/of_exact/run.log"; echo "FAIL: OpenFOAM exact"; exit 1; }
stage "$W/of_ctrl"  gs 0.1 5000; run "$W/of_ctrl"  of || { tail -5 "$W/of_ctrl/run.log";  echo "FAIL: OpenFOAM control"; exit 1; }
# brae: the experiment, today's path, the loose control, the fail-proof, and the Krylov-entry notice.
stage "$W/brae_exact"  gs 0 5000;   run "$W/brae_exact"  brae || { tail -5 "$W/brae_exact/run.log";  say "EXACT       the colourGS run finished" FAIL; }
stage "$W/brae_ref"    gs 0 5000;   run "$W/brae_ref"    brae BRAE_U_SOLVER=ofOrder || { tail -5 "$W/brae_ref/run.log";    say "REF         today's path finished" FAIL; }
stage "$W/brae_ctrl"   gs 0.1 5000; run "$W/brae_ctrl"   brae || { tail -5 "$W/brae_ctrl/run.log";   say "CONTROL     the colourGS relTol 0.1 run finished" FAIL; }
stage "$W/brae_fp"     gs 0 1;      run "$W/brae_fp"     brae || { tail -5 "$W/brae_fp/run.log";     say "FAIL-PROOF  the colourGS maxIter 1 run finished" FAIL; }
stage "$W/brae_krylov" krylov;      run "$W/brae_krylov" brae || { tail -5 "$W/brae_krylov/run.log"; say "KRYLOV      the colourGS run on the PBiCGStab entry finished" FAIL; }
# SUBST: the GaussSeidel entry with the smoothSolver path switched OFF (BRAE_RHO_SMOOTHSOLVER=0) and no
# colourGS: the shared reader must print exactly the substitution lines the SAID arms assert ABSENT,
# so those absence checks are shown able to match (review, round 2).
stage "$W/brae_subst"  gs 0 5000;   run "$W/brae_subst"  brae BRAE_RHO_SMOOTHSOLVER=0 || { tail -5 "$W/brae_subst/run.log";  say "SUBST       the substituted run finished" FAIL; }
# DEFAULT: the EXACT arm above already runs with no BRAE_U_SOLVER at all, so its colour lines prove the
# default; this arm proves the opt-out is real -- the same case under ofOrder must run OpenFOAM's order.

# ARM SAID: what the logs say. The notices go to stderr, the start-up and residual lines to stdout;
# run() folds both into run.log.
has() { grep -q -- "$2" "$W/$1/run.log"; }
has brae_subst "solvers/U solver: case asks 'smoothSolver', brae runs PBiCGStab" \
    && has brae_subst "solvers/U smoother: 'GaussSeidel' -- brae is not running a smoothSolver" \
    && say "SUBST   the substitution lines print where the smoothSolver path is off (the SAID absences can match)" ok \
    || say "SUBST   the substitution lines print where the smoothSolver path is off (the SAID absences can match)" FAIL
has brae_exact "brae NOTICE \[approximated\] solvers/U smoother: case asks 'GaussSeidel' in OpenFOAM's index order; brae sweeps in COLOUR order" \
    && say "SAID  EXACT carries the colour-order notice for the case's GaussSeidel smoother" ok \
    || say "SAID  EXACT carries the colour-order notice for the case's GaussSeidel smoother" FAIL
has brae_exact "momentum: multicolour Gauss-Seidel smoothSolver, [0-9]* colours (sizes " \
    && say "SAID  EXACT prints the start-up line with the colour count and sizes" ok \
    || say "SAID  EXACT prints the start-up line with the colour count and sizes" FAIL
has brae_exact "^colourGaussSeidel:  Solving for Ux, Initial residual = " \
    && say "SAID  EXACT's Ux residual line names colourGaussSeidel" ok \
    || say "SAID  EXACT's Ux residual line names colourGaussSeidel" FAIL
has brae_exact "Solving for Uz" \
    && say "SAID  EXACT prints no Uz line (2D: Uz is not solved, and OpenFOAM prints none)" FAIL \
    || say "SAID  EXACT prints no Uz line (2D: Uz is not solved, and OpenFOAM prints none)" ok
for arm in brae_exact brae_ctrl brae_fp brae_krylov; do
    if has $arm "solvers/U solver: case asks .*brae runs PBiCGStab" || has $arm "solvers/U smoother: .*brae is not running a smoothSolver" || has $arm "solvers/U preconditioner"; then
        say "SAID  $arm: the shared reader printed no contradictory PBiCGStab / smoother / preconditioner line for U" FAIL
    else
        say "SAID  $arm: the shared reader printed no contradictory PBiCGStab / smoother / preconditioner line for U" ok
    fi
done
# ...and the REF arm, today's path, says none of it and names its own sweep.
if has brae_ref "momentum: multicolour" || has brae_ref "colourGaussSeidel"; then
    say "SAID  REF (BRAE_U_SOLVER=ofOrder) carries no colour-order notice, start-up line or log name" FAIL
else
    say "SAID  REF (BRAE_U_SOLVER=ofOrder) carries no colour-order notice, start-up line or log name" ok
fi
has brae_ref "^GaussSeidel:  Solving for Ux, Initial residual = " \
    && say "SAID  REF's Ux residual line names GaussSeidel (OpenFOAM's own ascending sweep)" ok \
    || say "SAID  REF's Ux residual line names GaussSeidel (OpenFOAM's own ascending sweep)" FAIL
# ARM KRYLOV: a non-smoothSolver entry is announced as a different SOLVER, once, by the driver.
has brae_krylov "brae NOTICE \[approximated\] solvers/U solver: case asks 'PBiCGStab' with preconditioner 'DILU', brae runs a multicolour symGaussSeidel smoothSolver to the same tolerance, relTol, maxIter and minIter, nSweeps 1 -- a different solver" \
    && say "KRYLOV  the PBiCGStab/DILU entry is announced as a different solver under colourGS" ok \
    || say "KRYLOV  the PBiCGStab/DILU entry is announced as a different solver under colourGS" FAIL
has brae_krylov "solvers/U smoother" \
    && say "KRYLOV  no smoother notice on an entry that names no smoother" FAIL \
    || say "KRYLOV  no smoother notice on an entry that names no smoother" ok
# The OpenFOAM side of the premise: its GaussSeidel smoothSolver drove every momentum solve to its
# roundoff floor, or the EXACT arm compares two unconverged iterates and the bound means nothing. The
# floor, not the tolerance: Ux cannot reach 1e-14 on this fixture (see MEASURED) and takes the cap there.
python3 - "$W/of_exact/run.log" <<'PY' && say "EXACT  OpenFOAM's GaussSeidel smoothSolver drove every Ux/Uy solve to its roundoff floor (< 1e-12)" ok \
                                 || say "EXACT  OpenFOAM's GaussSeidel smoothSolver drove every Ux/Uy solve to its roundoff floor (< 1e-12)" FAIL
import re, sys
s = open(sys.argv[1]).read()
rows = re.findall(r'smoothSolver:\s+Solving for U[xy], Initial residual = ([-+0-9.eE]+), Final residual = ([-+0-9.eE]+), No Iterations (\d+)', s)
assert rows, 'no smoothSolver Ux/Uy lines in the OpenFOAM log'
worst = max(float(r[1]) for r in rows); most = max(int(r[2]) for r in rows)
print('  OpenFOAM smoothSolver on U: %d solves, worst final residual %.3e, most sweeps %d (5000 = the cap, on the Ux floor)' % (len(rows), worst, most))
sys.exit(0 if worst < 1e-12 else 1)
PY

python3 - "$W" "$N" <<'PY' || fail=1
import re, sys, os, numpy as np
W, N = sys.argv[1], sys.argv[2]
def internal(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m: return None
    n = int(m.group(2)); start = m.end(); comps = 3 if m.group(1) == b'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary': return np.frombuffer(b[start:start+n*comps*8], dtype='<f8')
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0])
    return np.array([float(x) for x in vals[:n*comps]])
bad = 0
def say(ok, what):
    global bad
    print('  %-86s %s' % (what, 'ok' if ok else 'FAIL'))
    if not ok: bad = 1
FIELDS = ('U', 'p', 'k', 'epsilon')
def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
def load(arm):
    out = {}
    for f in FIELDS:
        fn = '%s/%s/%s/%s' % (W, arm, N, f)
        out[f] = internal(fn) if os.path.exists(fn) else None
    return out
def worst(a, b, label):
    parts = []; w = 0.0
    for f in FIELDS:
        if a[f] is None or b[f] is None or a[f].size != b[f].size:
            parts.append('%s missing' % f); w = float('inf'); continue
        e = rel(a[f], b[f]); parts.append('%s %.3e' % (f, e)); w = max(w, e)
    print('  %-28s %s' % (label + ':', '  '.join(parts)))
    return w
BOUND = 1e-9
of = load('of_exact'); ofc = load('of_ctrl')
print('  (information) OpenFOAM relTol 0.1 vs OpenFOAM converged: worst %.3e' % worst(ofc, of, 'OF ctrl vs OF exact'))
e = worst(load('brae_exact'), of, 'EXACT colourGS vs OF')
say(e <= BOUND, 'EXACT       colourGS at tolerance 1e-14 tracks OpenFOAM on U, p, k, epsilon (bound %.0e)' % BOUND)
e = worst(load('brae_ref'), of, 'REF today vs OF')
say(e <= BOUND, "REF         today's path (OpenFOAM's own sweep) also meets the bound" )
e = worst(load('brae_ctrl'), ofc, 'CONTROL colourGS vs OF')
say(e > BOUND, 'CONTROL     at relTol 0.1 the two orders stop elsewhere: fields differ by MORE than the bound')
e = worst(load('brae_fp'), of, 'FAIL-PROOF maxIter 1 vs OF')
say(e > BOUND, 'FAIL-PROOF  colourGS capped at one sweep (brae only) misses the bound')
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: the default colour Gauss-Seidel momentum solve is exact where it converges, different where it does not, and says so"
exit $fail
