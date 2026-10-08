#!/usr/bin/env bash
# Shared by the clock gates that replay OpenFOAM's log through the WHOLE clock -- setDeltaT.H, Time's write
# cadence and the function objects' (function_object_write_times.sh, function_object_write_entries.sh).
# Sourced after ../lib.sh. tests/test_time_controls_cadence_log.cu is the replay.
REPLAY="${BUILD:-$ROOT/build}/test_time_controls_cadence_log"
[ -x "$REPLAY" ] || { echo "SKIP: $REPLAY not built"; exit 77; }
# got <file> <key>: a field of the replay's RESULT line
got() { grep -a "^RESULT " "$1" | tail -1 | tr ' ' '\n' | sed -n "s/^$2=//p"; }
# clock_rows <log> <controlDict's deltaT>: a row a step -- deltaT before, Co, alphaCo, deltaT after
clock_rows()
{
    python3 - "$1" "$2" <<'PY'
import re, sys
before = sys.argv[2]
co = alpha = None
for line in open(sys.argv[1], errors='replace'):
    m = re.match(r'^Courant Number mean: \S+ max: (\S+)', line)
    if m:
        co = m.group(1)
        continue
    m = re.match(r'^Interface Courant Number mean: \S+ max: (\S+)', line)
    if m:
        alpha = m.group(1)
        continue
    m = re.match(r'^deltaT = (\S+)', line)
    if m and co is not None and alpha is not None:
        print(before, co, alpha, m.group(1))
        before = m.group(1)
        co = alpha = None
PY
}
# clock_steps <OpenFOAM log> <brae log>: "<OpenFOAM's steps> <brae's steps> <worst relative gap of a deltaT>"
clock_steps()
{
    python3 - "$1" "$2" <<'PY'
import re, sys
of = [float(x) for x in re.findall(r'^deltaT = (\S+)', open(sys.argv[1], errors='replace').read(), re.M)]
br = [float(x) for x in re.findall(r'\bdt = (\S+)', open(sys.argv[2], errors='replace').read())]
n = min(len(of), len(br))
print(len(of), len(br), '%.1e' % max([abs(a - b)/abs(a) for a, b in zip(of[:n], br[:n])] or [1.0]))
PY
}
# clock_stage <dir> <endTime> <file holding a `functions { ... }` entry>: laminar/damBreak with its own clock
# -- adjustTimeStep, `writeControl adjustable; writeInterval 0.05` -- and that entry, run by OpenFOAM (cached);
# its log's rows land in <dir>.rows. Returns 1 with the premise that failed said.
clock_stage()
{
    local o="$1" c="$1/system/controlDict"
    stage "$LAM" "$o" "$2" adjustable 0.05 0 > "$o.stage.txt" 2>&1 || { say "laminar/damBreak staged" FAIL; return 1; }
    python3 - "$c" "$3" <<'PY' || { say "PREMISE  the functions entry was staged" FAIL; return 1; }
import sys
p, fo = sys.argv[1:3]
s = open(p).read()
assert s.count('functions\n{\n}') == 1
open(p, 'w').write(s.replace('functions\n{\n}', open(fo).read().strip(), 1))
PY
    runof "$o"
    grep -q "^adjustTimeStep  *yes;" "$c" && grep -q "^writeControl  *adjustable;" "$c" \
        && ! grep -q "loading function object" "$o/log.interFoam" \
        || { say "PREMISE  the step is adjusted, Time's cadence adjustable, every object loaded" FAIL; return 1; }
    clock_rows "$o/log.interFoam" "$(sed -n -E 's/^deltaT +([^;]+);.*/\1/p' "$c")" > "$o.rows"
    [ "$(wc -l < "$o.rows")" = "$(grep -c "^Time = " "$o/log.interFoam")" ] \
        || { say "PREMISE  every step of OpenFOAM's log is a row" FAIL; return 1; }
}
# clock_replay <dir> <out> [rule broken]: the rows through brae's clock, as it is or with one rule broken
clock_replay()
{
    if [ -n "${3:-}" ]; then
        BRAE_CONTROL_FUNCTION_OBJECT_STEP="$3" "$REPLAY" "$1.rows" "$1" > "$2" 2>&1
    else
        "$REPLAY" "$1.rows" "$1" > "$2" 2>&1
    fi
}
# clock_arm <staged dir> <run dir> <host|device> [env...]: brae's own run of the staged case
clock_arm()
{
    local o="$1" e="$2" loop="$3"
    shift 3
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    runbrae "$e" "$loop" "$@"
}
