#!/usr/bin/env bash
# The transonic pressure BiCGStab can take the case's `preconditioner DILU`, says which one runs, and
# stops where the other did.
#
# A transonic pressure matrix is asymmetric (fvm::div(phid, p)), so the CUDA mirror runs BiCGStab on it.
# sbMatched's p entry is `solver PBiCGStab; preconditioner DILU;`; brae keeps the diagonal by default
# and announces it, because the value is the same and the diagonal is faster -- measured on this
# fixture at 20 iterations: DILU 202 against 668 BiCGStab iterations on the first p solve, and 474
# against 68 ms per iteration on the p phase (tolerance 1e-12 / relTol 0, so both converge fully and the
# residual trajectories agree to the printed digits); at 896k cells on the squareBend tutorial 567
# against 315 ms (bench/results/rhoSimpleFoam_squareBend_gb10.md). BRAE_DILU_P=1 opts in.
#   ARM 1   the notices are truthful: the default run announces "case asks 'DILU', brae preconditions
#           with diagonal"; the BRAE_DILU_P=1 run announces NOTHING on solvers/p (nothing substituted).
#   ARM 2   it is a preconditioner, not a different equation: both runs' p initial residuals track
#           each other over the first iterations (bound 1e-6 against the measured 0.000e+00).
#   ARM 3   DILU does the work in fewer solver iterations (the pIters the summary line now carries),
#           the CONTROL that the opt-in actually ran.
# Fixture: validation/sbMatched (squareBend as it ships, transonic yes), 3 iterations, CUDA mirror.
# MEASURED: pIters 202 / 163 / 163 (DILU) against 612-668 / 498-517 / 494-503 (diagonal, which varies
# run to run with the reductions it sums); p initial residuals
# identical to the printed digits at iterations 2 and 3.
# THE COMPARISON ARM PINS BRAE_P_SOLVER=diagonal because the DEFAULT transonic preconditioner is no
# longer the diagonal but brae's AMG V-cycle (2026-09-08, item 77b): unpinned, this arm would measure
# the AMG and the gate would compare DILU against a preconditioner it is not about (measured then:
# "diagonal" 53/50/43 iterations, which is the AMG, against DILU's 202/161/163).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/sbMatched}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
grep -qE "transonic\s+yes" "$SRC/system/fvSolution" || { echo "SKIP: $SRC is not transonic"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
N=3
for arm in "dilu BRAE_DILU_P=1" "diag BRAE_P_SOLVER=diagonal"; do
    set -- $arm; label=$1; envs=${2:-}
    d="$W/$label"; rm -rf "$d"; cp -r "$SRC" "$d"; rm -rf "$d"/[1-9]* 2>/dev/null; [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    python3 - "$d" "$N" <<'PY'
import re, sys
d, n = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s); s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s); s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
PY
    ( cd "$d" && env $envs BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > log 2>&1 ) || { tail -5 "$d/log"; say "$label  the run finished" FAIL; }
done
grep -qE "preconditioner\s+DILU" "$SRC/system/fvSolution" || { echo "SKIP: $SRC does not ask DILU on p"; exit 77; }
grep -q "solvers/p preconditioner: case asks 'DILU', brae preconditions with diagonal" "$W/diag/log" \
    && say "ARM 1  the default run announces the diagonal substitution" ok \
    || say "ARM 1  the default run announces the diagonal substitution" FAIL
grep -q "NOTICE \[approximated\] solvers/p " "$W/dilu/log" \
    && say "ARM 1  the BRAE_DILU_P=1 run announces nothing on solvers/p (the case's DILU runs)" FAIL \
    || say "ARM 1  the BRAE_DILU_P=1 run announces nothing on solvers/p (the case's DILU runs)" ok
python3 - "$W" "$N" <<'PY' || fail=1
import re, sys
W, N = sys.argv[1], int(sys.argv[2])
def parse(p):
    out = []
    for l in open(p):
        m = re.match(r'Time = (\d+)\s+(.*)$', l)
        if m: out.append({k: float(v) for k, v in re.findall(r'(\w+) ([\d.eE+-]+)', m.group(2))})
    return out
a, b = parse(W + '/dilu/log'), parse(W + '/diag/log')
bad = 0
def say(ok, what):
    global bad
    print('  %-74s %s' % (what, 'ok' if ok else 'FAIL'))
    if not ok: bad = 1
say(len(a) == N and len(b) == N, 'both runs reached %d iterations' % N)
if len(a) == N and len(b) == N:
    for i in range(N):
        print('  iteration %d: p initial residual DILU %.4e / diagonal %.4e   pIters %d / %d' % (i + 1, a[i]['p'], b[i]['p'], a[i].get('pIters', -1), b[i].get('pIters', -1)))
    # iteration 1 is identical by construction (same initial state); with tolerance 1e-12 / relTol 0
    # both solves converge fully, so 2 and 3 agree to the solver floor -- measured 0.000e+00 at the
    # printed digits
    worst = max(abs(a[i]['p'] - b[i]['p']) / max(abs(b[i]['p']), 1e-300) for i in range(1, N))
    print('  worst relative difference of the p initial residual, iterations 2..%d: %.3e' % (N, worst))
    say(worst <= 1e-6, 'ARM 2  the two preconditioners stop the same equation at the same rule (<= 1e-6 on the residual trajectory)')
    say(all(a[i].get('pIters', 1e9) < b[i].get('pIters', 0) for i in range(N)), 'ARM 3  DILU takes fewer solver iterations on every p solve (control that it ran)')
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: the transonic pressure is DILU-preconditioned, announced, and stops where the diagonal did"
exit $fail
