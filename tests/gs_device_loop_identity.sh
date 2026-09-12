#!/usr/bin/env bash
# The device-side smoothSolver loop is the host loop, byte for byte.
#
# deviceSymGaussSeidel runs OpenFOAM's smoothSolver stopping rule on the device by default -- sweep,
# residual and `(finalRes < tol || finalRes < relTol*initRes) && sweeps >= minIter` inside one
# conditional-graph WHILE node, two host syncs per solve where the host loop paid one per sweep. Item 55
# measured those reads as 31 of the turbulence hook's ~43 ms per outer iteration on T3A, against 11.5 ms
# of kernels. The device loop must therefore be the SAME loop: the same sweeps in the same order, the
# same residual, the same stop -- or it is a faster different solver, which this project refuses.
#
#   ARM 1  50 T3A iterations under the device loop and under BRAE_GS_HOST_LOOP=1: every `Time =` line and
#          every `Solving for` line (item 52's Initial / Final / No Iterations per solve) byte-identical.
#   ARM 2  the two runs differ in nothing else either: the written fields at 50 are identical files.
#   CONTROL the comparison can fail: the host-loop run against a third run at a different relTol on U
#          must DIFFER in its Solving-for lines, so "identical" is not "the diff never looked".
# Since item 68 the DEFAULT smoother runs on the host; BRAE_GS_HOST_SMOOTHER=0 pins the device loop this
# gate is about (its announce assertions would fail loudly otherwise -- the arms would compare the host
# smoother with itself).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/T3A}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {
    mkdir -p "$1"; cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$1/"; cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "${2:-}" <<'PY'
import re, sys
d, rt = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^endTime .*', 'endTime         50;', s, flags=re.M)
s = re.sub(r'^writeInterval .*', 'writeInterval   50;', s, flags=re.M)
open(c, 'w').write(s)
if rt:
    f = d + '/system/fvSolution'; s = open(f).read()
    s = s.replace('solvers\n{', 'solvers\n{\n    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol %s; maxIter 1000; }\n' % rt, 1)
    open(f, 'w').write(s)
PY
}
prep "$W/dev"; prep "$W/host"; prep "$W/ctl" 0.01
( cd "$W/dev"  && BRAE_GS_HOST_SMOOTHER=0 BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/dev"  > log 2>&1 ) || { echo "FAIL: device-loop run crashed"; tail -5 "$W/dev/log"; exit 1; }
( cd "$W/host" && BRAE_GS_HOST_LOOP=1 BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/host" > log 2>&1 ) || { echo "FAIL: host-loop run crashed"; tail -5 "$W/host/log"; exit 1; }
( cd "$W/ctl"  && BRAE_GS_HOST_LOOP=1 BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/ctl"  > log 2>&1 ) || { echo "FAIL: control run crashed"; tail -5 "$W/ctl/log"; exit 1; }
lines() { grep -E "^Time = |Solving for" "$1/log"; }
n=$(lines "$W/dev" | wc -l); echo "  $n log lines compared per run"
[ "$n" -ge 200 ] || say "ARM 1  the runs produced solve report lines to compare" FAIL
if diff <(lines "$W/dev") <(lines "$W/host") > /dev/null; then say "ARM 1  device loop and host loop: every Time= and Solving-for line identical" ok
else say "ARM 1  device loop and host loop: every Time= and Solving-for line identical" FAIL; diff <(lines "$W/dev") <(lines "$W/host") | head -6; fi
same=1; for f in U p k omega nut phi; do cmp -s "$W/dev/50/$f" "$W/host/50/$f" || { same=0; echo "  differs: 50/$f"; }; done
[ $same -eq 1 ] && say "ARM 2  the written fields at 50 are identical files" ok || say "ARM 2  the written fields at 50 are identical files" FAIL
if diff <(lines "$W/host") <(lines "$W/ctl") > /dev/null; then say "CONTROL  a different relTol on U changes the Solving-for lines (so ARM 1 can fail)" FAIL
else say "CONTROL  a different relTol on U changes the Solving-for lines (so ARM 1 can fail)" ok; fi
[ $fail -eq 0 ] && echo "PASS: the device-side smoothSolver loop is the host loop, byte for byte"
exit $fail
