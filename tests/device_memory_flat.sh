#!/usr/bin/env bash
# Device memory, and the caches that hold it, must not grow with the iteration count.
#
# brae caches per-solve state on a pointer because a captured CUDA graph BAKES the pointers it saw: the
# entry is only replayable for the same field. That key is safe only while the field's address is stable.
# The legacy driver allocates its psi and its U fresh every outer iteration, so a cache keyed on one of
# those gains an entry per iteration and never gives the memory back. It has happened twice: item 75 on
# the host (the smoother's pinned scratch, 29 MB/iteration) and item 76 on the device -- grad(U)'s memo,
# keyed on Ux.data(), nine gradient buffers plus the boundary values per entry, measured at 28 MiB per
# iteration on the composed 209,825-cell flat plate: 2.0 GiB by iteration 33, 17.8 GiB by 596. It took
# the machine down. Both fixes were the same shape: key on what does not move, or do not key at all.
#
# TWO INSTRUMENTS, because either alone can be fooled. Peak device memory is what actually killed the
# box, but it is polled and has a large constant baseline that dilutes a small leak. The cache census
# (BRAE_CACHE_STATS=1 prints a line whenever a named cache reaches a new high-water mark) is exact and
# catches a leak of any size, including one in a cache that does not hold much -- before it grows.
#
#   ARM 1   the legacy driver, the path that moves the pointers and the path the tutorial gates run.
#   ARM 2   the V2 mirror, which solves into persistent fields; a regression there is caught too.
#   For each arm, at 100 and at 400 iterations:
#     (a) every cache's high-water entry count is the SAME at 400 as at 100 -- bounded, not growing;
#     (b) peak device memory at 400 is within 1.25x of the peak at 100.
#   CONTROLS the runs must reach their iteration counts (two short runs would compare nothing), and the
#           census must actually name grad(U)'s memo (an empty census would pass (a) on nothing).
#
# FAIL-PROOF, run by hand against the leaking code (revert the key in deviceGradUShared to Ux.data()):
#   ARM 1 read gradu-memo 101 entries at 100 iterations and 401 at 400, and peak device memory 629 MiB
#   -> 1631 MiB (2.59x); both assertions failed and the gate exited 1. ARM 2 stayed at 1 entry and 1.00x,
#   which is the point of having it: the leak was on the legacy driver only.
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
run() {   # run <dir> <iterations> <env> -> writes log, cache.log, gpu (peak MiB), iters
    local d="$1" n="$2" envs="$3"
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$SRC/constant" "$SRC/system" "$d/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$d/0"; else cp -r "$SRC/0" "$d/0"; fi
    python3 - "$d" "$n" <<'PY'
import re, sys
d, n = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
# residualControl would stop the short and the long run at the same iteration and the slope would be
# measured on nothing; both runs must go the full distance.
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PY
    # exec so the pid we watch IS brae's: nvidia-smi reports memory per compute-app pid, and a wrapper
    # shell would never appear in that list.
    ( cd "$d" && exec env BRAE_CACHE_STATS=1 $envs "$BRAE" "$d" > log 2> cache.log ) &
    local pid=$! peak=0 m
    while kill -0 "$pid" 2>/dev/null; do
        m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
            | awk -v p="$pid" -F', ' '$1==p{print $2}')
        [ -n "${m:-}" ] && [ "$m" -gt "$peak" ] 2>/dev/null && peak=$m
        sleep 0.2
    done
    wait "$pid" 2>/dev/null
    echo "$peak" > "$d/gpu"
    grep -c '^Time = ' "$d/log" > "$d/iters" 2>/dev/null || echo 0 > "$d/iters"
}
high() {   # high <dir> -> "name count" per cache, the high-water mark it reached
    awk '/^\[cache\]/ {split($3,a,"="); if (a[2]+0 > m[$2]) m[$2]=a[2]+0}
         END {for (k in m) print k, m[k]}' "$1/cache.log" | sort
}
arm() {   # arm <label> <env>
    local label="$1" envs="$2" a="$W/${1// /_}_100" b="$W/${1// /_}_400"
    run "$a" 100 "$envs"; run "$b" 400 "$envs"
    local ia ib ga gb
    ia=$(cat "$a/iters"); ib=$(cat "$b/iters"); ga=$(cat "$a/gpu"); gb=$(cat "$b/gpu")
    [ "$ia" -ge 100 ] && [ "$ib" -ge 400 ] && say "$label  both runs reached their iteration count" ok \
                                           || say "$label  both runs reached their iteration count" FAIL
    high "$a" | grep -q '^gradu-memo ' && say "$label  the census names grad(U)'s memo" ok \
                                       || say "$label  the census names grad(U)'s memo" FAIL
    local bad=0
    while read -r name n400; do
        local n100; n100=$(high "$a" | awk -v k="$name" '$1==k{print $2}')
        [ -z "$n100" ] && n100=0
        printf '  %s  cache %-16s %s entries at %s iterations, %s at %s\n' "$label" "$name" "$n100" "$ia" "$n400" "$ib"
        [ "$n400" -gt "$n100" ] && bad=1
    done < <(high "$b")
    [ "$bad" -eq 0 ] && say "$label  no cache grew between the short and the long run" ok \
                     || say "$label  no cache grew between the short and the long run" FAIL
    if [ "$ga" -gt 0 ] 2>/dev/null; then
        printf '  %s  peak device memory %s MiB at %s iterations, %s MiB at %s -> %sx\n' "$label" \
            "$ga" "$ia" "$gb" "$ib" "$(python3 -c "print(f'{$gb/$ga:.2f}')")"
        python3 -c "import sys; sys.exit(0 if $gb <= 1.25*$ga else 1)" \
            && say "$label  peak device memory is flat in the iteration count (<= 1.25x)" ok \
            || say "$label  peak device memory is flat in the iteration count (<= 1.25x)" FAIL
    else
        say "$label  the poll saw the run's device memory" FAIL
    fi
}
arm "ARM 1  legacy" ""
arm "ARM 2  mirror" "BRAE_SIMPLEFOAM_V2=1"
[ $fail -eq 0 ] && echo "PASS: device memory and every pointer-keyed cache are flat in the iteration count"
exit $fail
