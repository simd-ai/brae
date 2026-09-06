#!/usr/bin/env bash
# The device-side PBiCGStab loop is the host loop, byte for byte.
#
# deviceJacobiBiCGStab now runs OpenFOAM's PBiCGStab recurrence with its two per-iteration tests -- the
# mid-iteration exit on |s| and the end-of-iteration test on |r| plus the breakdown flag -- inside one
# conditional graph (a WHILE whose body ends in an IF/ELSE), four host syncs per solve where the host loop
# paid two per iteration. Item 55/57 measured those reads as 114 of 158 ms per outer iteration on
# squareBend, where every field is BiCGStab. The device loop must be the SAME loop -- same recurrence,
# same tests, same stop -- or it is a faster different solver, which this project refuses.
#
#   ARM 1  pitzDailyTurb (12,225 cells, kEpsilon; U, k, epsilon on PBiCGStab+DILU at tolerance 1e-10,
#          relTol 0 -- a deep solve every iteration): 30 iterations under the device loop and under
#          BRAE_BICG_HOST_LOOP=1, every `Time =` and every `Solving for` line (Initial / Final /
#          No Iterations per solve) byte-identical, and the written fields identical files.
#   ARM 2  rhoBox (1,200 cells, laminar, U/h on PBiCGStab+DILU at relTol 0.1) through the rho mirror,
#          50 iterations: the same identity on its log and fields. Its solves converge inside the
#          host-driven iteration 0, so this arm holds the PROLOGUE identical and never launches the graph.
#   ARM 3  sbMatched (112,000 cells, kEpsilon, every field PBiCGStab at 1e-12/relTol 0) through the rho
#          mirror, 5 iterations: hundreds of graph iterations per solve, DILU and diagonal preconditioners
#          both inside the graph; the device arm must announce the graph launch.
#   CONTROL pitzDailyTurb at relTol 0.01 on U must DIFFER from ARM 1's host run in its Solving-for lines,
#          so "identical" is not "the diff never looked".
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in pitzDailyTurb rhoBox; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <fixture> <endTime> [uRelTol]
    mkdir -p "$1"; cp -r "$ROOT/validation/$2/constant" "$ROOT/validation/$2/system" "$1/"
    if [ -d "$ROOT/validation/$2/0.orig" ]; then cp -r "$ROOT/validation/$2/0.orig" "$1/0"; else cp -r "$ROOT/validation/$2/0" "$1/0"; fi
    python3 - "$1" "$3" "${4:-}" <<'PY'
import re, sys
d, n, rt = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
# rhoBox keeps several entries on one line, so match the entry, not the line
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
if rt:
    f = d + '/system/fvSolution'; s = open(f).read()
    s = re.sub(r'(\bU\s*\{[^}]*relTol\s+)[0-9.eE+-]+;', r'\g<1>%s;' % rt, s, count=1)
    open(f, 'w').write(s)
PY
}
lines() { grep -E "^Time = |Solving for|residual" "$1/log" | grep -vE "^  BiCGStab: (device|host) loop"; }
run() { ( cd "$1" && env $2 "$BRAE" "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }

# ARM 1
prep "$W/pd_dev" pitzDailyTurb 30; prep "$W/pd_host" pitzDailyTurb 30; prep "$W/pd_ctl" pitzDailyTurb 30 0.01
run "$W/pd_dev"  "BRAE_SIMPLEFOAM_V2=1"
run "$W/pd_host" "BRAE_BICG_HOST_LOOP=1 BRAE_SIMPLEFOAM_V2=1"
run "$W/pd_ctl"  "BRAE_BICG_HOST_LOOP=1 BRAE_SIMPLEFOAM_V2=1"
# The two arms must have run DIFFERENT loops, or "identical" compares the host loop with itself -- which
# is exactly what happened when the graph code was compiled out behind a macro this file did not see.
grep -q "BiCGStab: device loop" "$W/pd_dev/log"  && say "ARMS   the device arm announces the device loop (the graph path ran)" ok  || say "ARMS   the device arm announces the device loop (the graph path ran)" FAIL
grep -q "BiCGStab: host loop"   "$W/pd_host/log" && ! grep -q "BiCGStab: device loop" "$W/pd_host/log" \
    && say "ARMS   the host arm announces the host loop and never the device loop" ok || say "ARMS   the host arm announces the host loop and never the device loop" FAIL
n=$(lines "$W/pd_dev" | wc -l); echo "  pitzDailyTurb: $n log lines compared"
[ "$n" -ge 100 ] || say "ARM 1  the runs produced solve report lines to compare" FAIL
if diff <(lines "$W/pd_dev") <(lines "$W/pd_host") > /dev/null; then say "ARM 1  pitzDailyTurb: device loop and host loop, every Time=/Solving-for line identical" ok
else say "ARM 1  pitzDailyTurb: device loop and host loop, every Time=/Solving-for line identical" FAIL; diff <(lines "$W/pd_dev") <(lines "$W/pd_host") | head -6; fi
same=1; for f in U p k epsilon nut phi; do [ -f "$W/pd_dev/30/$f" ] || continue; cmp -s "$W/pd_dev/30/$f" "$W/pd_host/30/$f" || { same=0; echo "  differs: 30/$f"; }; done
[ $same -eq 1 ] && say "ARM 1  ...and the written fields at 30 are identical files" ok || say "ARM 1  ...and the written fields at 30 are identical files" FAIL
# CONTROL
if diff <(lines "$W/pd_host") <(lines "$W/pd_ctl") > /dev/null; then say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" FAIL
else say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" ok; fi

# ARM 2 -- the rho mirror
prep "$W/rb_dev" rhoBox 50; prep "$W/rb_host" rhoBox 50
run "$W/rb_dev"  "BRAE_RHOSIMPLEFOAM_MIRROR=cuda"
run "$W/rb_host" "BRAE_BICG_HOST_LOOP=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda"
n=$(lines "$W/rb_dev" | wc -l); echo "  rhoBox: $n log lines compared"
[ "$n" -ge 50 ] || say "ARM 2  the rho runs produced residual lines to compare" FAIL
if diff <(lines "$W/rb_dev") <(lines "$W/rb_host") > /dev/null; then say "ARM 2  rhoBox (rho mirror): device loop and host loop, every residual line identical" ok
else say "ARM 2  rhoBox (rho mirror): device loop and host loop, every residual line identical" FAIL; diff <(lines "$W/rb_dev") <(lines "$W/rb_host") | head -6; fi
same=1; t=$(ls -d "$W"/rb_dev/[1-9]* 2>/dev/null | xargs -n1 basename | sort -n | tail -1); for f in U p T rho phi; do [ -f "$W/rb_dev/$t/$f" ] || continue; cmp -s "$W/rb_dev/$t/$f" "$W/rb_host/$t/$f" || { same=0; echo "  differs: $t/$f"; }; done
[ $same -eq 1 ] && say "ARM 2  ...and the written fields at $t are identical files" ok || say "ARM 2  ...and the written fields at $t are identical files" FAIL

# ARM 3 -- the rho mirror with DEEP solves: sbMatched (112,000 cells, kEpsilon; p, U, e, k, epsilon all
# PBiCGStab at tolerance 1e-12, relTol 0 -- hundreds of iterations per solve, U/e/k/epsilon under DILU and p
# under the diagonal), 5 iterations. rhoBox above converges inside the host-driven iteration 0 at its
# relTol 0.1 and never launches the graph, so this is the arm that exercises the WHILE body, both IF
# branches and the DILU preconditioner inside the graph.
if [ -d "$ROOT/validation/sbMatched" ]; then
    prep "$W/sb_dev" sbMatched 5; prep "$W/sb_host" sbMatched 5
    run "$W/sb_dev"  "BRAE_RHOSIMPLEFOAM_MIRROR=cuda"
    run "$W/sb_host" "BRAE_BICG_HOST_LOOP=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda"
    grep -q "BiCGStab: device loop" "$W/sb_dev/log" && say "ARM 3  sbMatched: the device arm launched the graph (deep solves, DILU inside it)" ok || say "ARM 3  sbMatched: the device arm launched the graph (deep solves, DILU inside it)" FAIL
    if diff <(lines "$W/sb_dev") <(lines "$W/sb_host") > /dev/null; then say "ARM 3  sbMatched (rho mirror): device loop and host loop, every residual line identical" ok
    else say "ARM 3  sbMatched (rho mirror): device loop and host loop, every residual line identical" FAIL; diff <(lines "$W/sb_dev") <(lines "$W/sb_host") | head -6; fi
    # sbMatched is NOT bit-reproducible run to run under EITHER loop (measured: two host-loop runs differ in
    # U by 1.6e-12 relative L2, two device-loop runs by 3.4e-12; the source is elsewhere in the rho CUDA
    # arm, see REFUSALS.md item 69), so byte identity of its fields is not available here. The assertion
    # is floor-relative instead: the device-vs-host difference must sit within 10x the case's own
    # run-to-run floor, measured IN THIS RUN as the larger of a host-vs-host and a device-vs-device repeat.
    # Three runs read the ratio at 0.9x, 2.4x and 4.6x on U (single samples of a noisy quantity on both
    # sides), so 5x would flake on the case's own noise; 10x of a 1e-12 floor is 1e-11, and a loop that
    # stopped at a different iterate shows up at 1e-6 or worse -- five orders above the bound. The
    # residual-line identity above already says it does not; this arm says the fields agree to the noise
    # the case has anyway. The ratios are printed so drift is visible.
    prep "$W/sb_dev2" sbMatched 5; prep "$W/sb_host2" sbMatched 5
    run "$W/sb_dev2"  "BRAE_RHOSIMPLEFOAM_MIRROR=cuda"
    run "$W/sb_host2" "BRAE_BICG_HOST_LOOP=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda"
    python3 - "$W" <<'PY'
import re, sys, numpy as np
W = sys.argv[1]
def read(p):
    b = open(p, errors='latin-1').read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    n = int(m.group(2)); body = b[m.end():].split(')\n;')[0]
    return np.array([float(x) for x in re.findall(r'[-+0-9.eE]+', body)]).reshape(n, -1)
def rel(a, b):
    x, y = read(a), read(b); return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))
fail = 0
for f in ('U', 'p', 'T', 'k', 'epsilon'):     # the mirror writes T, not e
    floor = max(rel(f'{W}/sb_host/5/{f}', f'{W}/sb_host2/5/{f}'), rel(f'{W}/sb_dev/5/{f}', f'{W}/sb_dev2/5/{f}'), 1e-16)
    gap = rel(f'{W}/sb_dev/5/{f}', f'{W}/sb_host/5/{f}')
    ok = gap <= 10.0 * floor
    print("  ARM 3  %-8s dev vs host %.2e   run-to-run floor %.2e   (%.1fx, bound 10x)  %s" % (f, gap, floor, gap / floor, "ok" if ok else "FAIL"))
    if not ok: fail = 1
sys.exit(fail)
PY
    [ $? -eq 0 ] || fail=1
else
    echo "  ARM 3  skipped: validation/sbMatched missing"
fi

[ $fail -eq 0 ] && echo "PASS: the device-side PBiCGStab loop is the host loop -- byte for byte where the case is reproducible, within the case's own noise where it is not"
exit $fail
