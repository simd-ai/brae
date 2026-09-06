#!/usr/bin/env bash
# The single-block DILU walk is the per-level walk, byte for byte.
#
# Item 70. A DILU apply ran one kernel launch per dependency level per half-sweep -- 537 of them on
# validation/sbMatched, whose 112,000 cells fall into 268 levels of at most 800 -- and each launch carried
# a few microseconds of work behind an overhead of about the same size. Measured on that matrix with a
# benchmark on the real system: 2213 us per apply as launches against 1460 as ONE BLOCK per half-sweep,
# and on rhoBox (79 levels, widest 20) 651 against 156. A cooperative grid-barrier walk was measured too
# and is NOT the answer: 268 grid barriers cost what 268 launches cost (1111 us against 1106).
#
# Which walk a mesh TAKES is decided on the mean level width (128), because in production the apply is
# captured in the BiCGStab graph, where a launch is a cheap graph node and one block is one SM: measured
# end to end, the single block wins on pitzDailyTurb (mean 47) and loses on sbMatched (mean 418). That is
# a performance switch; this gate is about the arithmetic, so it FORCES each walk with BRAE_DILU_SINGLE
# and BRAE_DILU_PER_LEVEL and asserts from the log which one ran.
#
# Correct for the reason the Gauss-Seidel single-block walk is (item 60): cells at one level share no
# face, so no thread reads what another writes inside a level, and __syncthreads orders the levels
# exactly as the separate launches did. Same statements, same order, so the numbers must be identical --
# BRAE_DILU_PER_LEVEL=1 restores the launches and this gate holds the two together.
#
#   ARM 1   validation/sbMatched through the rho mirror, 20 iterations: U, e, k and epsilon all solve on
#           PBiCGStab + DILU at 1e-12, so every one of them walks this preconditioner hundreds of times.
#   ARM 2   validation/rhoBox through the rho mirror, 50 iterations.
# There is no incompressible arm, and the reason is worth recording: the V2 driver announces a
# substitution on pitzDailyTurb's `preconditioner DILU` and runs Jacobi, so it builds no DILU at all. An
# arm there would have compared two runs that never entered this code -- it did, and passed vacuously,
# until each arm was made to prove from the log which walk it took.
#   CONTROL a run with p's tolerance loosened must differ from ARM 1, so the comparison can fail.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in sbMatched rhoBox; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-76s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <fixture> <endTime> [loosenP]
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$ROOT/validation/$2/constant" "$ROOT/validation/$2/system" "$1/"
    if [ -d "$ROOT/validation/$2/0.orig" ]; then cp -r "$ROOT/validation/$2/0.orig" "$1/0"; else cp -r "$ROOT/validation/$2/0" "$1/0"; fi
    python3 - "$1" "$3" "${4:-}" <<'PY'
import re, sys
d, n, loose = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom startTime;', s)
s = re.sub(r'\bstartTime\s+[^;]*;', 'startTime 0;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
if loose:
    f = d + '/system/fvSolution'; s = open(f).read()
    s2 = re.sub(r'(\n\s*U\s+)0\.9\s*;', r'\g<1>0.7;', s, count=1)
    assert s2 != s, 'the U relaxation factor is not the one this control was written against'
    open(f, 'w').write(s2)
PY
}
lines() { grep -E "^Time = |Solving for" "$1/log"; }
run() { ( cd "$1" && env $2 "$BRAE" $3 "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }
arm() {   # arm <label> <fixture> <endTime> <env> <flag> <fields...>
    local label=$1 fx=$2 n=$3 envs=$4 flag=$5; shift 5
    local a="$W/${fx}_s" b="$W/${fx}_l"
    prep "$a" "$fx" "$n"; prep "$b" "$fx" "$n"
    run "$a" "BRAE_DILU_SINGLE=1 $envs" "$flag"
    run "$b" "BRAE_DILU_PER_LEVEL=1 $envs" "$flag"
    grep -q "DILU: single-block walk" "$a/log" && say "$label  the first arm walked one block" ok || say "$label  the first arm walked one block" FAIL
    grep -q "DILU: per-level walk"   "$b/log" && say "$label  the second arm walked per level" ok || say "$label  the second arm walked per level" FAIL
    local nl; nl=$(lines "$a" | wc -l); echo "  $label  $fx: $nl lines compared"
    [ "$nl" -ge "$n" ] || say "$label  the runs produced the expected report lines" FAIL
    diff <(lines "$a") <(lines "$b") > /dev/null \
        && say "$label  $fx: single-block and per-level DILU, every line identical" ok \
        || { say "$label  $fx: single-block and per-level DILU, every line identical" FAIL; diff <(lines "$a") <(lines "$b") | head -4; }
    local same=1 f t
    t=$(ls -d "$a"/[1-9]* 2>/dev/null | xargs -n1 basename | sort -n | tail -1)
    for f in "$@"; do [ -f "$a/$t/$f" ] || { same=0; echo "  missing: $t/$f"; continue; }; cmp -s "$a/$t/$f" "$b/$t/$f" || { same=0; echo "  differs: $t/$f"; }; done
    [ $same -eq 1 ] && say "$label  ...and every written field at $t is an identical file" ok \
                    || say "$label  ...and every written field at $t is an identical file" FAIL
}
arm "ARM 1" sbMatched     20 "BRAE_RHOSIMPLEFOAM_MIRROR=cuda" "-case" U p T rho phi k epsilon
arm "ARM 2" rhoBox        50 "BRAE_RHOSIMPLEFOAM_MIRROR=cuda" "-case" U p T rho phi
prep "$W/ctl" sbMatched 20 loose; run "$W/ctl" "BRAE_DILU_SINGLE=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda" "-case"
diff <(lines "$W/sbMatched_s") <(lines "$W/ctl") > /dev/null \
    && say "CONTROL  a changed momentum relaxation changes the run (so the arms can fail)" FAIL \
    || say "CONTROL  a changed momentum relaxation changes the run (so the arms can fail)" ok
[ $fail -eq 0 ] && echo "PASS: the single-block DILU walk is the per-level walk, byte for byte"
exit $fail
