#!/usr/bin/env bash
# The host smoother's scratch is scratch: its memory must not grow with the iteration count.
#
# Item 68 gave the smoother a cache keyed on the field it solves, copying the graph paths, which key on
# psi because a captured GRAPH bakes pointers. The host smoother bakes nothing -- everything it holds is
# working storage for one call -- and the legacy driver hands it a psi whose address MOVES between
# iterations, so every iteration added an entry holding about 27 MB of PINNED host memory that is never
# freed. Measured on the composed 209,825-cell flat plate: 29 MB per iteration, 6.16 GB by iteration 200,
# 11.88 GB by 400, and the turbulentFlatPlate tutorial gate's run reached 54 GB before it was stopped --
# on a 121 GB machine, with page-locked memory, which comes out of the machine rather than the process.
# It is now one leaked scratch, resized when the mesh size changes.
#
# THE GATE IS THE SHAPE, not a number: peak resident memory must be FLAT in the iteration count. A leak
# of any size shows as a slope, and a bound in gigabytes would pass a smaller leak silently.
#
#   ARM 1   the legacy driver (the path the tutorial gates run) on validation/T3A: peak RSS at 400
#           iterations is within 1.25x of the peak at 100. A per-iteration leak of even 1 MB would read
#           1.9x here.
#   ARM 2   the same for the V2 mirror, which solves into persistent fields and never showed the leak --
#           so a regression there is caught too.
#   CONTROL the arms must actually reach their iteration counts; a run that stopped early would compare
#           two short runs and pass on nothing.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/T3A}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
command -v /usr/bin/time >/dev/null 2>&1 || { echo "SKIP: /usr/bin/time not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
peak() {   # peak <dir> <iterations> <env>  -> peak RSS in kB on stdout, iterations on fd 3
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
# residualControl would stop the short and the long run at the SAME iteration and the slope would be
# measured on nothing; both arms must run the full count.
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PY
    ( cd "$d" && { /usr/bin/time -f "PEAK %M" env $envs "$BRAE" "$d" > log; } 2> t ) || true
    grep -c '^Time = ' "$d/log" > "$d/iters" 2>/dev/null || echo 0 > "$d/iters"
    grep "^PEAK" "$d/t" | awk '{print $2}'
}
arm() {   # arm <label> <env>
    local label="$1" envs="$2"
    local a b ia ib
    a=$(peak "$W/${label// /_}_100" 100 "$envs"); ia=$(cat "$W/${label// /_}_100/iters")
    b=$(peak "$W/${label// /_}_400" 400 "$envs"); ib=$(cat "$W/${label// /_}_400/iters")
    if [ -z "$a" ] || [ -z "$b" ] || [ "$a" -le 0 ] 2>/dev/null; then say "$label  the runs reported a peak" FAIL; return; fi
    local r; r=$(python3 -c "print(f'{$b/$a:.2f}')")
    printf '  %s  peak RSS %.2f GB at %s iterations, %.2f GB at %s -> %sx\n' "$label" \
        "$(python3 -c "print($a/1048576)")" "$ia" "$(python3 -c "print($b/1048576)")" "$ib" "$r"
    [ "$ia" -ge 100 ] && [ "$ib" -ge 400 ] && say "$label  both runs reached their iteration count" ok \
                                           || say "$label  both runs reached their iteration count" FAIL
    python3 -c "import sys; sys.exit(0 if $b <= 1.25*$a else 1)" \
        && say "$label  peak memory is flat in the iteration count (<= 1.25x)" ok \
        || say "$label  peak memory is flat in the iteration count (<= 1.25x)" FAIL
}
arm "ARM 1  legacy" ""
arm "ARM 2  mirror" "BRAE_SIMPLEFOAM_V2=1"
[ $fail -eq 0 ] && echo "PASS: the smoother's scratch does not grow with the run"
exit $fail
