#!/usr/bin/env bash
# The AMG's single-precision CSR values refilled through each grid's own addressing -- byte for byte the
# entry-by-entry gather through a stored source list, which they replace.
#
# amgCastFP32 refills every CSR grid's values from its double-precision face arrays at every solve. It read
# where each value comes from in a stored list (csrSrc) as long as the columns: 0.42 GiB on boxy5 at 4.94 M
# cells. It now puts each face's upper and lower value at the slot the row layout gives it -- row c is its owner
# faces and then its losort faces (buildCsrForGrid) -- one thread a slot (csrGatherFacesK); BRAE_AMG_CSR_SRC=1
# keeps the list and its gather. The same casts into the same slots, so every solve line and every field must
# be identical, or the gather put a value where the layout does not.
#
#   ARM     validation/pitzDaily (simpleFoam, the AMG-PCG pressure with the single-precision cycle on CSR
#           grids), 50 iterations: the face gather against BRAE_AMG_CSR_SRC=1, lines and fields identical.
#   PREMISE under BRAE_POOL_STATS=1 the list arm asks the driver for more blocks (cudaMalloc calls): one list a
#           CSR grid. Not the footprint: on pitzDaily the lists are ~0.2 MB, below the report's 1 MB figure.
#   CONTROL BRAE_AMG_FP32=0 (the double-precision cycle, no CSR values at all) must CHANGE the solve lines
#           against the list arm: the fail-proof that a difference in the cycle is seen here.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
[ -d "$ROOT/validation/pitzDaily" ] || { echo "SKIP: fixture pitzDaily missing"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-80s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <endTime>
    mkdir -p "$1"; cp -r "$ROOT/validation/pitzDaily/constant" "$ROOT/validation/pitzDaily/system" "$1/"
    cp -r "$ROOT/validation/pitzDaily/0" "$1/0"
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
lines() { grep -E "^Time = |Solving for" "$1/log"; }
mallocs() { sed -n 's/^\[pool\] loop end   cudaMalloc .* in \([0-9]*\) calls.*/\1/p' "$1/err"; }
# stdout and stderr apart: the pool report goes to stderr and, interleaved, cuts a solve line in two
run() { ( cd "$1" && env $2 stdbuf -oL "$BRAE" "$1" > log 2> err ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/err"; exit 1; }; }
N=50
prep "$W/rows" $N; prep "$W/list" $N; prep "$W/fp64" $N
run "$W/rows" "BRAE_POOL_STATS=1"
run "$W/list" "BRAE_POOL_STATS=1 BRAE_AMG_CSR_SRC=1"
run "$W/fp64" "BRAE_AMG_FP32=0"
nl=$(lines "$W/rows" | grep -c "Solving for p")
echo "  pitzDaily: $nl pressure solve lines compared"
[ "$nl" -ge $N ] || say "the runs produced a pressure solve line an iteration" FAIL
if diff <(lines "$W/rows") <(lines "$W/list") > /dev/null; then say "the face gather and the stored list: every Time=/Solving-for line identical" ok
else say "the face gather and the stored list: every Time=/Solving-for line identical" FAIL; diff <(lines "$W/rows") <(lines "$W/list") | head -4; fi
same=1
for f in U p k epsilon nut phi
do
    [ -f "$W/rows/$N/$f" ] || { same=0; echo "  missing: $N/$f"; continue; }
    cmp -s "$W/rows/$N/$f" "$W/list/$N/$f" || { same=0; echo "  differs: $N/$f"; }
done
[ $same -eq 1 ] && say "...and the written fields at $N are identical files" ok || say "...and the written fields at $N are identical files" FAIL
mr=$(mallocs "$W/rows"); ml=$(mallocs "$W/list")
echo "  pool cudaMalloc calls: face gather $mr, stored list $ml"
if [ -n "$mr" ] && [ -n "$ml" ] && [ "$mr" -lt "$ml" ]
then say "PREMISE  the face gather keeps no source list (it takes fewer blocks)" ok
else say "PREMISE  the face gather keeps no source list (it takes fewer blocks)" FAIL; fi
if diff <(lines "$W/list") <(lines "$W/fp64") > /dev/null; then say "CONTROL  the double-precision cycle changes the solve lines (so the ARM can fail)" FAIL
else say "CONTROL  the double-precision cycle changes the solve lines (so the ARM can fail)" ok; fi
[ $fail -eq 0 ] && echo "PASS: the CSR values gathered by face are the stored list's, byte for byte"
exit $fail
