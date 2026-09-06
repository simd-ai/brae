#!/usr/bin/env bash
# The level-ordered Gauss-Seidel gather is the index gather, byte for byte.
#
# Item 60b: per walk position, a cell's terms are stored as a CSR in exactly the order gsCellUpdate
# subtracts them (its losort terms in losort order, then its owner terms in face order), built by replaying
# that gather's loops on the view's own losort/ownerStart, with the coefficients, diagonal and source
# permuted into walk order once per solve. Per level the walk then chases three dependent loads where the
# index gather chased five. Same terms, same order, same arithmetic -- so every solve line and every field
# must come out identical, or it is a different smoother, which this project refuses.
#
#   ARM 1   validation/T3A under the V2 driver, 50 iterations: U through the fused walk, k/omega/ReThetat/
#           gammaInt through the scalar device loop, all single-block -- under the level-ordered gather and
#           under BRAE_GS_LEVEL_GATHER=0, every `Time =`/`Solving for` line identical, every field identical.
#   ARM 2   the same case with BRAE_GS_PER_LEVEL=1 in BOTH arms, 20 iterations: the per-level kernels.
#   ARM 3   validation/duct3d_cf (72,000 cells, three components, GaussSeidel), 20 iterations.
#   ARMS    the announces name the gather each arm ran ("level-ordered gather" / "index gather").
#   CONTROL T3A at relTol 0.01 on U under the index gather must differ from ARM 1's index arm.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in T3A duct3d_cf; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-80s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <fixture> <endTime> [uRelTol]
    mkdir -p "$1"; cp -r "$ROOT/validation/$2/constant" "$ROOT/validation/$2/system" "$1/"
    if [ -d "$ROOT/validation/$2/0.orig" ]; then cp -r "$ROOT/validation/$2/0.orig" "$1/0"; else cp -r "$ROOT/validation/$2/0" "$1/0"; fi
    python3 - "$1" "$3" "${4:-}" <<'PY'
import re, sys
d, n, rt = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
if rt:
    f = d + '/system/fvSolution'; s = open(f).read()
    s = s.replace('solvers\n{', 'solvers\n{\n    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol %s; maxIter 10; }\n' % rt, 1)
    open(f, 'w').write(s)
PY
}
lines() { grep -E "^Time = |Solving for" "$1/log"; }
run() { ( cd "$1" && env $2 "$BRAE" "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }
arm() {   # arm <label> <fixture> <endTime> <extraEnv> <minLines> <fields...>
    local label=$1 fx=$2 n=$3 extra=$4 minl=$5; shift 5
    local dl="$W/${label// /_}_${fx}_l" di="$W/${label// /_}_${fx}_i"
    prep "$dl" "$fx" "$n"; prep "$di" "$fx" "$n"
    run "$dl" "BRAE_SIMPLEFOAM_V2=1 $extra"
    run "$di" "BRAE_GS_LEVEL_GATHER=0 BRAE_SIMPLEFOAM_V2=1 $extra"
    grep -qE "level-ordered gather" "$dl/log" && ! grep -qE "index gather" "$dl/log" && say "$label  the gather arm announces the level-ordered gather only" ok || say "$label  the gather arm announces the level-ordered gather only" FAIL
    grep -qE "index gather" "$di/log" && ! grep -qE "level-ordered gather" "$di/log" && say "$label  the BRAE_GS_LEVEL_GATHER=0 arm announces the index gather only" ok || say "$label  the BRAE_GS_LEVEL_GATHER=0 arm announces the index gather only" FAIL
    local nl; nl=$(lines "$dl" | wc -l); echo "  $label  $fx: $nl log lines compared"
    [ "$nl" -ge "$minl" ] || say "$label  the runs produced the expected number of report lines" FAIL
    if diff <(lines "$dl") <(lines "$di") > /dev/null; then say "$label  $fx: level-ordered and index gathers, every Time=/Solving-for line identical" ok
    else say "$label  $fx: level-ordered and index gathers, every Time=/Solving-for line identical" FAIL; echo "  ($(diff <(lines "$dl") <(lines "$di") | grep -c '^<') lines differ)"; diff <(lines "$dl") <(lines "$di") | head -4; fi
    local same=1 f
    for f in "$@"; do [ -f "$dl/$n/$f" ] || { same=0; echo "  missing: $n/$f"; continue; }; cmp -s "$dl/$n/$f" "$di/$n/$f" || { same=0; echo "  differs: $n/$f"; }; done
    [ $same -eq 1 ] && say "$label  ...and the written fields at $n are identical files" ok || say "$label  ...and the written fields at $n are identical files" FAIL
}
arm "ARM 1" T3A 50 "" 200 U p k omega nut ReThetat gammaInt phi
arm "ARM 2" T3A 20 "BRAE_GS_PER_LEVEL=1" 80 U p k omega nut phi
grep -q "per-level launches" "$W/ARM_2_T3A_l/log" && say "ARM 2  ...and that arm walked per-level launches" ok || say "ARM 2  ...and that arm walked per-level launches" FAIL
arm "ARM 3" duct3d_cf 20 "" 100 U p phi
prep "$W/T3A_c" T3A 50 0.01; run "$W/T3A_c" "BRAE_GS_LEVEL_GATHER=0 BRAE_SIMPLEFOAM_V2=1"
if diff <(lines "$W/ARM_1_T3A_i") <(lines "$W/T3A_c") > /dev/null; then say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" FAIL
else say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" ok; fi
[ $fail -eq 0 ] && echo "PASS: the level-ordered Gauss-Seidel gather is the index gather, byte for byte"
exit $fail
