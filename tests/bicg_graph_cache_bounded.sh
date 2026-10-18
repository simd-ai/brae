#!/usr/bin/env bash
# The BiCGStab graph cache keyed on the field it solves holds one entry per solved field, not one per iteration.
#
# deviceJacobiBiCGStabGraph (CUDA 13 builds) keeps a captured graph per psi.data() and never evicts one. The legacy
# simpleFoam driver's velocity corrector wrote each U component into a new block and moved it in. Once boxy5's
# memory step 4 (6625cb9) handed the momentum source back after the pressure phase, those addresses stopped
# recurring and the cache took an entry an iteration: implicitAMI 101 after 100 iterations, every entry a graph and
# its work vectors. The corrector now writes into U's own block, so each field's graph is captured once.
#
#   ARM      validation/implicitAMI (laminar, cyclicAMI: U by Jacobi-BiCGStab, no colour sweep), BRAE_CACHE_STATS=1,
#            100 iterations: the bicg-graph cache's high mark is at most the number of fields the log solves with
#            Jacobi-BiCGStab, and it is the high mark of a 20-iteration run (it does not grow with the iterations).
#   PREMISE  the run solved with Jacobi-BiCGStab and reported the cache (a CUDA 12 build has no BiCGStab graph and
#            reports none: SKIP).
#   CONTROL  BRAE_CONTROL_U_CORRECTOR_NEW_BLOCK=1 (the new block moved in, as before) must exceed the bound at 100
#            iterations: the fail-proof that a cache taking an entry an iteration is seen here.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
[ -d "$ROOT/validation/implicitAMI" ] || { echo "SKIP: fixture implicitAMI missing"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-90s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <endTime>
    mkdir -p "$1"; cp -r "$ROOT/validation/implicitAMI/constant" "$ROOT/validation/implicitAMI/system" "$1/"
    cp -r "$ROOT/validation/implicitAMI/0" "$1/0"
    python3 - "$1" "$2" <<'PY'
import re, sys
d, n = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
open(c, 'w').write(s)
PY
}
run() { ( cd "$1" && env BRAE_CACHE_STATS=1 $2 stdbuf -oL "$BRAE" "$1" > log 2> err ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/err"; exit 1; }; }
mark() { sed -n 's/^\[cache\] bicg-graph entries=\([0-9]*\)$/\1/p' "$1/err" | tail -1; }
prep "$W/a100" 100; prep "$W/a20" 20; prep "$W/ctl" 100
run "$W/a100" ""
nf=$(grep -E "^Jacobi-BiCGStab: +Solving for " "$W/a100/log" | awk '{print $4}' | sort -u | wc -l)
m100=$(mark "$W/a100")
[ "$nf" -gt 0 ] && [ -n "$m100" ] || { echo "SKIP: no BiCGStab graph in this build ($nf fields by Jacobi-BiCGStab, cache mark '${m100}')"; exit 77; }
run "$W/a20" ""
run "$W/ctl" "BRAE_CONTROL_U_CORRECTOR_NEW_BLOCK=1"
m20=$(mark "$W/a20"); mc=$(mark "$W/ctl")
echo "  implicitAMI: $nf fields by Jacobi-BiCGStab; bicg-graph entries at 20 its ${m20:-none}, at 100 its $m100, control at 100 its ${mc:-none}"
[ "$m100" -le "$nf" ] && say "ARM      at most one cached graph per field BiCGStab solves, after 100 iterations" ok \
                      || say "ARM      at most one cached graph per field BiCGStab solves, after 100 iterations" FAIL
[ -n "$m20" ] && [ "$m100" -eq "$m20" ] && say "...and as many as after 20 iterations: none taken per iteration" ok \
                                        || say "...and as many as after 20 iterations: none taken per iteration" FAIL
[ -n "$mc" ] && [ "$mc" -gt "$nf" ] && say "CONTROL  U into a new block each iteration: the cache outgrows the bound (so the ARM can fail)" ok \
                                    || say "CONTROL  U into a new block each iteration: the cache outgrows the bound (so the ARM can fail)" FAIL
[ $fail -eq 0 ] && echo "PASS: the BiCGStab graph cache holds one graph per solved field"
exit $fail
