#!/usr/bin/env bash
# The pressure AMG-PCG's double-precision row layout: the PCG's product A p and the double cycle's finest grid on the finest grid's
# rows (amgGatherFineCsr64, BRAE_AMG_FINE_CSR, on by default), and every coarse grid of the double cycle on its own when asked
# (amgGatherCsr64, BRAE_AMG_CSR64_BELOW, off by default, the private multi-GPU build's setting),
# byte for byte the face form they replace: each row is the cell's faces in amulKernel's order with its multiply and add apart, so the
# same sum to the bit, and every solve line and every field must be identical.
#
#   ARMS    validation/pitzDaily (simpleFoam, the AMG-PCG pressure), 50 iterations written at 17 digits, the rows against
#           BRAE_AMG_FINE_CSR=0 BRAE_AMG_CSR64_BELOW=0 (amulKernel on every grid):
#             fp32   the default single-precision cycle: the PCG's product on the finest grid's rows
#             fp64   BRAE_AMG_FP32=0 BRAE_AMG_CSR64_BELOW=100000000: the double cycle as well, its finest and coarse grids on rows
#   PREMISE the rows' arms say they were made (each NOTICE), the fallback's do not
#   CONTROL BRAE_CONTROL_AMG_CSR64_ORDER=1 (each row owner faces first: the same sum in another order) and
#           BRAE_CONTROL_AMG_CSR64_STALE=1 (the values gathered once and never again) must each change the fp64 arm's fields
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
[ -d "$ROOT/validation/pitzDaily" ] || { echo "SKIP: fixture pitzDaily missing"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-80s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
N=50
prep() {   # prep <dir>
    mkdir -p "$1"; cp -r "$ROOT/validation/pitzDaily/constant" "$ROOT/validation/pitzDaily/system" "$1/"
    cp -r "$ROOT/validation/pitzDaily/0" "$1/0"
    rm -f "$1"/constant/polyMesh/.brae_*
    python3 - "$1" "$N" <<'PY'
import re, sys
d, n = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 17;', s)
open(c, 'w').write(s)
PY
}
lines() { grep -E "^Time = |Solving for" "$1/log"; }
run() { prep "$1"; ( cd "$1" && env $2 stdbuf -oL "$BRAE" "$1" > log 2> err ); echo $? > "$1/rc"; }
same() {   # same <a> <b>: every written field the same file
    local f
    for f in U p k epsilon nut phi
    do
        [ -f "$1/$N/$f" ] && [ -f "$2/$N/$f" ] || return 1
        cmp -s "$1/$N/$f" "$2/$N/$f" || return 1
    done
    return 0
}
OFF="BRAE_AMG_FINE_CSR=0 BRAE_AMG_CSR64_BELOW=0"
run "$W/fp32-rows" ""
run "$W/fp32-off" "$OFF"
run "$W/fp64-rows" "BRAE_AMG_FP32=0 BRAE_AMG_CSR64_BELOW=100000000"
run "$W/fp64-off" "BRAE_AMG_FP32=0 $OFF"
run "$W/ctl-order" "BRAE_AMG_FP32=0 BRAE_AMG_CSR64_BELOW=100000000 BRAE_CONTROL_AMG_CSR64_ORDER=1"
run "$W/ctl-stale" "BRAE_AMG_FP32=0 BRAE_AMG_CSR64_BELOW=100000000 BRAE_CONTROL_AMG_CSR64_STALE=1"
for a in fp32-rows fp32-off fp64-rows fp64-off
do
    [ "$(cat "$W/$a/rc")" = 0 ] || { say "$a ran (exit $(cat "$W/$a/rc"))" FAIL; tail -3 "$W/$a/err"; }
done
nl=$(lines "$W/fp64-rows" | grep -c "Solving for p")
echo "  pitzDaily: $nl pressure solve lines a run compared"
[ "$nl" -ge $N ] || say "the runs produced a pressure solve line an iteration" FAIL
for p in fp32 fp64
do
    if diff <(lines "$W/$p-rows") <(lines "$W/$p-off") > /dev/null
    then say "$p: the rows and amulKernel, every Time=/Solving-for line identical" ok
    else say "$p: the rows and amulKernel, every Time=/Solving-for line identical" FAIL; diff <(lines "$W/$p-rows") <(lines "$W/$p-off") | head -4; fi
    same "$W/$p-rows" "$W/$p-off" && say "$p: ...and every field at $N the same file (17 digits)" ok || say "$p: ...and every field at $N the same file (17 digits)" FAIL
done
grep -q "the finest grid's row layout" "$W/fp32-rows/log" && grep -q "the finest grid's row layout" "$W/fp64-rows/log" \
    && grep -q "the double cycle's row layout on [1-9]" "$W/fp64-rows/log" \
    && ! grep -q "the double cycle's row layout" "$W/fp32-rows/log" \
    && ! grep -q "row layout" "$W/fp32-off/log" && ! grep -q "row layout" "$W/fp64-off/log" \
    && say "PREMISE  the rows' arms made the finest and the coarse grids' rows, the fallback's none" ok \
    || say "PREMISE  the rows' arms made the finest and the coarse grids' rows, the fallback's none" FAIL
for c in order stale
do
    if same "$W/ctl-$c" "$W/fp64-off"
    then say "CONTROL  $c: the fields are the face form's (the ARM cannot see it)" FAIL
    else say "CONTROL  $c: the fields differ from the face form's (exit $(cat "$W/ctl-$c/rc"))" ok; fi
done
[ $fail -eq 0 ] && echo "PASS: the pressure's double-precision rows are the face form, byte for byte"
exit $fail
