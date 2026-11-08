#!/usr/bin/env bash
# The SIMPLE loop's divergence detector (solvers/common/solution_health.cuh) on a real run. Each thing it claims to read is broken in one cell of a healthy
# pitzDailySST at one iteration (BRAE_TEST_CORRUPT), and a run blows up by itself with the two defects tests/simple_linearupwind_turbulence_vs_openfoam.sh names put back.
#   clean      nothing injected: exit 0, every iteration, every time written
#   injected   U, p, phi, k, omega and nut NaN, U Inf, k and omega negative: an error exit naming the field and the iteration, and nothing written at or after it
#   natural    BRAE_CONTROL_BOUND_EMPTY=1 BRAE_CONTROL_TURB_GRAD_LIVE=1, linearUpwind on k and omega, F1 off, the inlet one ulp down: job 4835375 froze here at omega 1e14 with k at
#              its floor in 2,083 cells, every residual near 1e-9, under the old tripwire's bar, and wrote the fields. The run must stop by itself. Whether a staging
#              freezes is the GPU's as much as the case's, so the inlet's last bits are tried in turn and every run is replayed through the detector's rule (below)
#   healthy    both fixes in, F1 on: a run that bounds hundreds of times reaches its end. A run that goes over the bar and comes back cannot be staged here, because where a chaotic run stands decides
#              it (job 4845295, F1 on, inlet one ulp up: omega 4.8e7 at iteration 363, 5,600 times the limit, and on to 10000) and a spike by hand does not come back (omega 1e12 in one cell went to 1e21):
#              tests/test_solution_health.cu holds that decision on its trajectory
# CONTROL, asserted red: BRAE_ALLOW_NONFINITE=1 on the U NaN arm, the opt-out: the detector says nothing, so the injected arms are the detector's. The natural arm's own
# control: the same run with the opt-out reaches its end, where the fields are the frozen ones.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDailySST"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
stage() {   # <dir> <end> <write> <lu|-> <up|dn|->
    local c=$1
    rm -rf "$c"; mkdir -p "$c"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$c/"
    rm -f "$c"/constant/polyMesh/.brae_*
    python3 - "$c" "$2" "$3" "$4" "$5" <<'PYEOF'
import math, re, sys
c, end, wr, lu, tw = sys.argv[1:]
p = c + '/system/controlDict'; s = open(p).read()
for k, v in (('startFrom', 'startTime'), ('startTime', '0'), ('endTime', end), ('writeInterval', wr)):
    s, n = re.subn(r'(?m)(^|\s)%s\s+[^;]*;' % k, r'\1%s %s;' % (k, v), s)
    if n == 0:
        s += '\n%s %s;\n' % (k, v)
s = re.sub(r'\nlibs\s[^;]*;', '\n', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(p, 'w').write(s)
p = c + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
if lu == 'lu':
    # The settings of the measured runs: solves a decade under a residual control of 1e-6, relaxation 0.9
    s, n = re.subn(r'solvers\s*\{.*?\}\s*\n\s*(?=SIMPLE)', '''solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-08; relTol 0; maxIter 5000; }
    Phi { $p; }
    "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; nSweeps 1; tolerance 1e-08; relTol 0; maxIter 5000; }
}
''', s, count=1, flags=re.S)
    assert n == 1
    s = re.sub(r'consistent\s+[^;]*;', 'consistent yes;', s)
    s, n = re.subn(r'relaxationFactors\s*\{.*?\n\}', 'relaxationFactors\n{\n    equations\n    {\n        U 0.9;\n        ".*" 0.9;\n    }\n}', s, count=1, flags=re.S)
    assert n == 1
open(p, 'w').write(s)
if lu == 'lu':
    p = c + '/system/fvSchemes'; s = open(p).read()
    s, b = re.subn(r'div\(phi,k\)\s+[^;]*;', 'div(phi,k) bounded Gauss linearUpwind grad(k);', s)
    s, d = re.subn(r'div\(phi,omega\)\s+[^;]*;', 'div(phi,omega) bounded Gauss linearUpwind grad(omega);', s)
    assert b == 1 and d == 1
    s = re.sub(r'div\(phi,epsilon\)[^;]*;', '', s)
    s = re.sub(r'\n\s*turbulence\s+bounded[^;]*;', '', s)
    open(p, 'w').write(s)
if tw != '-':
    # up, dn: one ulp; up2, dn2: two
    p = c + '/0/U'; b = open(p).read()
    m = re.search(r'inlet\s*\{[^}]*?value\s+uniform \(([-0-9.eE+]+) ', b, flags=re.S)
    assert m and tw in ('up', 'dn', 'up2', 'dn2')
    v = float(m.group(1))
    for _ in range(2 if tw.endswith('2') else 1):
        v = math.nextafter(v, math.inf if tw.startswith('up') else -math.inf)
    open(p, 'w').write(b[:m.start(1)] + repr(v) + b[m.end(1):])
PYEOF
}
run() { local c=$1; shift; (cd "$c" && env BRAE_AMG_CACHE=0 "$@" "$BRAE" -case . > log 2>&1 < /dev/null; echo $? > rc); }
stage "$W/clean" 8 1 - -
run "$W/clean"
[ "$(cat "$W/clean/rc")" = 0 ] && [ -d "$W/clean/8" ] && [ "$(grep -a -c '^Time = ' "$W/clean/log")" = 8 ] && say "clean: exit 0, every iteration, every time written" ok || say "clean: exit 0, every iteration, every time written" FAIL
for spec in U:nan U:inf p:nan phi:nan k:nan omega:nan nut:nan k:neg omega:neg
do
    a=${spec%:*}-${spec#*:}
    f=${spec%:*}
    stage "$W/$a" 8 1 - -
    run "$W/$a" "BRAE_TEST_CORRUPT=$spec:4"
    case $f in U) name=Ux ;; omega) name=epsilon.omega ;; *) name=$f ;; esac
    if [ "$(cat "$W/$a/rc")" != 0 ] && grep -aqE "solution diverged: .*$name.* at iteration 4" "$W/$a/log" && [ ! -d "$W/$a/4" ] && [ ! -d "$W/$a/5" ] && [ -d "$W/$a/3" ]
    then
        say "injected $spec at iteration 4: stops there, names it, writes nothing" ok
    else
        say "injected $spec at iteration 4: stops there, names it, writes nothing (exit $(cat "$W/$a/rc"))" FAIL
        grep -a -m2 ERROR "$W/$a/log" | cut -c1-160
    fi
done
stage "$W/opt" 8 1 - -
run "$W/opt" BRAE_TEST_CORRUPT=U:nan:4 BRAE_ALLOW_NONFINITE=1
grep -aq 'solution diverged' "$W/opt/log" && say "CONTROL: BRAE_ALLOW_NONFINITE=1 silences the detector" FAIL || say "CONTROL: BRAE_ALLOW_NONFINITE=1 silences the detector" ok
# natural: where a chaotic run stands decides whether it freezes (job 4845295), and the GPU it runs on is part of where it stands: on the
# GB10 (sm_121, CI run 42453796953) the one-ulp-down staging that froze on a GH200 (Arrhenius) ran its 3000 iterations unstopped. So the
# stagings are tried in turn, the measured one first: each with BRAE_HEALTH_LOG, its omega against its limit every iteration, which
# replay_detector replays through the detector's own rule (HealthAdmissible: above kExcess for two windows of kWindow, the second's low
# no lower than kNewLow times the first's), written here again, not read from brae
#   a staging the replay stops must be one brae stopped, writing nothing: a freeze brae let through is a FAIL, whichever staging it is
#   a staging the replay does not stop and brae ran to its end did not freeze on this machine: the next is tried
#   the first staging brae stops ends the arm; if none of them stops, this machine cannot witness the arm, and that is a FAIL too
replay_detector() {   # <log>: prints the iteration the rule stops at, or nothing
    python3 - "$1" <<'PYEOF'
import re, sys
excess, window, newlow = 1e3, 50, 0.9
count = above = 0
low = previous = 0.0
for i, m in enumerate(re.finditer(r'(?m)^health: \w+ max (\S+) limit (\S+) ratio \S+$', open(sys.argv[1], errors='replace').read()), start=1):
    v, lim = float(m.group(1)), float(m.group(2))
    if not lim > 0:
        continue
    last = v / lim
    if last <= excess:
        previous, count, above = 0.0, 0, 0
        continue
    above += 1
    low = last if count == 0 else min(low, last)
    count += 1
    if count < window:
        continue
    if previous > 0 and low >= newlow * previous:
        print(i)
        break
    previous, count = low, 0
PYEOF
}
natural=
for tw in dn - up dn2 up2
do
    c=$W/natural-$tw
    stage "$c" 3000 3000 lu "$tw"
    run "$c" BRAE_SIMPLE_CASE_SOLVER=1 BRAE_SCALAR_LINEARUPWIND=1 BRAE_CONTROL_BOUND_EMPTY=1 BRAE_CONTROL_TURB_GRAD_LIVE=1 BRAE_HEALTH_LOG=1
    stop=$(replay_detector "$c/log")
    its=$(grep -a -c '^Time = ' "$c/log")
    if [ "$(cat "$c/rc")" != 0 ] && grep -aq 'solution diverged' "$c/log" && [ ! -d "$c/3000" ]
    then
        printf '        inlet %-3s: stopped by brae after %s iterations (the replay: %s): %s\n' "$tw" "$its" "${stop:-no stop}" "$(grep -a -m1 -o 'solution diverged: [^.]*' "$c/log" | cut -c1-110)"
        natural=$tw
        break
    elif [ -n "$stop" ]
    then
        say "natural, inlet $tw: the replay stops at iteration $stop and brae did not stop (exit $(cat "$c/rc"), $its iterations)" FAIL
        natural=missed
        break
    else
        printf '        inlet %-3s: did not freeze here: %s iterations, exit %s, the replay never stops\n' "$tw" "$its" "$(cat "$c/rc")"
    fi
done
case $natural in
    "") say "natural: no staging froze on this machine, so the detector's natural stop is not witnessed" FAIL ;;
    missed) ;;
    *) say "natural: a frozen blow-up (inlet $natural) stops by itself and writes nothing" ok ;;
esac
stage "$W/healthy" 1000 1000 lu -
run "$W/healthy" BRAE_SIMPLE_CASE_SOLVER=1 BRAE_SCALAR_LINEARUPWIND=1 BRAE_SST_F1_PATCH=1 BRAE_HEALTH_LOG=1
[ "$(cat "$W/healthy/rc")" = 0 ] && [ -d "$W/healthy/1000" ] && say "healthy: the same run with both fixes in reaches its end" ok || say "healthy: the same run with both fixes in reaches its end (exit $(cat "$W/healthy/rc"))" FAIL
python3 - "$W/healthy/log" <<'PYEOF' && say "...and read the limit every iteration (premise)" ok || say "...and read the limit every iteration (premise)" FAIL
import re, sys
r = [float(m.group(1)) for m in re.finditer(r'(?m)^health: .* ratio (\S+)$', open(sys.argv[1], errors='replace').read())]
print('        %d iterations, largest ratio %.3e, last %.3e' % (len(r), max(r) if r else 0.0, r[-1] if r else 0.0))
raise SystemExit(0 if len(r) == 1000 and max(r) > 0 else 1)
PYEOF
[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
