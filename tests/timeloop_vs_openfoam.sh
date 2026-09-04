#!/usr/bin/env bash
# endTime IS AN ABSOLUTE TIME, deltaT IS THE STEP, AND THE TIME DIRECTORY IS NAMED BY THE VALUE.
#
# OpenFOAM's Time::run() tests `value() < endTime_ - 0.5*deltaT_` (Time.C:785), Time::operator++
# advances the value by deltaT (Time.C:1067) and Time::timeName formats that value (Time.C:721-728).
# `startFrom latestTime` resolves against the case's own numeric directories (Time.C:146-193).
#
# The V2 driver read endTime as an ITERATION COUNT, never read deltaT at all, hardcoded latestTime to
# "0", printed the iteration index where OpenFOAM prints the time name, and named its output directory
# by the index. Every fixture in validation/ is startTime 0 with deltaT 1, where all four coincide --
# which is why no gate could see it, and why a restart at 269 with endTime 1000 ran 1000 iterations
# instead of 731 and invalidated a set of measurements before anyone noticed.
#
#   ARM  the ORDERED LIST of time names, brae against OpenFOAM's own log, on four (startTime, endTime,
#        deltaT) shapes. One assertion covering the count, the naming and the per-iteration print.
#        The ratio-2.5 shape is deliberate: OpenFOAM runs 2 steps there where a rounded quotient gives 3.
#   ARM  the written time DIRECTORY name matches OpenFOAM's last one.
#   REFUSAL  endTime not beyond startTime must be refused by name, not silently run.
#   CONTROL  at least two shapes must have a time sequence that is NOT 1..N, or every arm would pass on
#            a driver that still counted iterations.
#
# writeInterval is forced to 1 on both sides because V2 reads no write cadence and writes only its final
# time; without it OpenFOAM writes nothing and the directory oracle is the start directory.
#
# Fail-proof, 2026-09-04: with the loop bounded by endTime again, the 5->8 shape reads "6 7 8 9 10 11 12
# 13" against OpenFOAM's "6 7 8"; with the directory named by the iteration index, the 0->2.5 shape
# writes 5/ against OpenFOAM's 2.5/.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/simpleBoxIO}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # stage <dir> <startTime> <endTime> <deltaT>
{
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    if [ -f "$1/system/blockMeshDict" ] && command -v blockMesh > /dev/null 2>&1; then
        ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
    fi
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import re, sys
d, st, et, dt = sys.argv[1:5]
c = d + '/system/controlDict'; s = open(c).read()
# simpleBoxIO's controlDict is on ONE line, so none of these may be line-anchored.
s = re.sub(r'startFrom\s+\S+;',      'startFrom       startTime;', s)
s = re.sub(r'startTime\s+\S+;',      'startTime       %s;' % st, s)
s = re.sub(r'endTime\s+\S+;',        'endTime         %s;' % et, s)
s = re.sub(r'deltaT\s+\S+;',         'deltaT          %s;' % dt, s)
s = re.sub(r'writeInterval\s+\S+;',  'writeInterval   1;', s)
s = re.sub(r'writeControl\s+\S+;',   'writeControl    timeStep;', s)
s = re.sub(r'writePrecision\s+\S+;', 'writePrecision  15;', s)
s = re.sub(r'functions\s*\{.*\n\}',  'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
open(f, 'w').write(re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S))
PY
}
seq_of()  { grep -oP '^Time = \S+' "$1/run.log" | sed 's/Time = //' | tr '\n' ' '; }
lastdir() { ( cd "$1" && ls -d [0-9]* 2>/dev/null | sort -g | tail -1 ); }

nontrivial=0
for shape in "0 5 1" "0 2.5 0.5" "0 1 0.4" "5 8 1"; do
    set -- $shape
    st=$1; et=$2; dt=$3
    stage "$W/of" "$st" "$et" "$dt"; stage "$W/br" "$st" "$et" "$dt"
    if [ "$st" != "0" ]; then
        # A restart needs the start directory to exist. Seed it with real OpenFOAM rather than copying
        # 0/, so both codes restart from a field OpenFOAM actually produced.
        for d in "$W/of" "$W/br"; do
            python3 - "$d" "$st" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
# The seed run has to START at 0 -- the directory the restart shape names does not exist yet.
s = re.sub(r'startTime\s+\S+;', 'startTime       0;', s)
s = re.sub(r'endTime\s+\S+;',   'endTime         %s;' % sys.argv[2], s)
open(c, 'w').write(s)
PY
            ( cd "$d" && simpleFoam > seed.log 2>&1 ) || { echo "FAIL: seeding $st"; exit 1; }
            python3 - "$d" "$st" "$et" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'startTime\s+\S+;', 'startTime       %s;' % sys.argv[2], s)
s = re.sub(r'endTime\s+\S+;',   'endTime         %s;' % sys.argv[3], s)
open(c, 'w').write(s)
PY
        done
    fi
    ( cd "$W/of" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM on $shape"; exit 1; }
    ( cd "$W/br" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/br" > run.log 2>&1 ) \
        || { echo "FAIL: brae crashed on $shape"; tail -8 "$W/br/run.log"; exit 1; }
    o="$(seq_of "$W/of")"; b="$(seq_of "$W/br")"
    say "start $st end $et dt $dt  times [$b] == OpenFOAM [$o]" "$([ "$o" = "$b" ] && echo ok || echo FAIL)"
    od="$(lastdir "$W/of")"; bd="$(lastdir "$W/br")"
    say "start $st end $et dt $dt  brae wrote ${bd}/ , OpenFOAM's last is ${od}/" \
        "$([ "$od" = "$bd" ] && echo ok || echo FAIL)"
    # A shape whose sequence is 1..N cannot tell a time from an iteration index.
    n=$(echo "$o" | wc -w); expect="$(seq 1 "$n" | tr '\n' ' ')"
    [ "$o" != "$expect" ] && nontrivial=$((nontrivial + 1))
done
say "CONTROL  $nontrivial shapes have a time sequence that is not 1..N" \
    "$([ "$nontrivial" -ge 2 ] && echo ok || echo FAIL)"

# ARM 5 -- THE LEGACY DRIVER'S STEP COUNT (queue item 46). Seven drivers derived it as
# `std::lround((endTime - startTime)/deltaT)`, which disagrees with OpenFOAM's `value() < endTime -
# 0.5*deltaT` whenever that ratio lands on n + 0.5. No fixture in validation/ can tell them apart -- all
# 159 have an integer ratio -- which is why the rounding survived in seven places at once. Only the COUNT
# is asserted: the legacy driver prints an iteration INDEX rather than OpenFOAM's time name, a separate
# hole that is not what this arm is about. Its own fixture, because the legacy driver refuses
# simpleBoxIO outright (a deviceAdjustPhi refusal, nothing to do with the step count).
LEGACY_SRC="${LEGACY_SRC:-$ROOT/validation/pitzDaily}"
if [ -d "$LEGACY_SRC" ]; then
    for side in of_l br_l; do
        rm -rf "$W/$side"; cp -r "$LEGACY_SRC" "$W/$side"
        rm -rf "$W/$side"/[1-9]* "$W/$side"/log.* "$W/$side"/postProcessing
        [ -d "$W/$side/0.orig" ] && { rm -rf "$W/$side/0"; cp -r "$W/$side/0.orig" "$W/$side/0"; }
        python3 - "$W/$side" <<'PYEOF'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'startFrom\s+\S+;',  'startFrom       startTime;', s)
s = re.sub(r'startTime\s+\S+;',  'startTime       0;',   s)
s = re.sub(r'endTime\s+\S+;',    'endTime         1;',   s)
s = re.sub(r'deltaT\s+\S+;',     'deltaT          0.4;', s)
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
f = sys.argv[1] + '/system/fvSolution'; s = open(f).read()
open(f, 'w').write(re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S))
PYEOF
    done
    ( cd "$W/of_l" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM on the ratio-2.5 shape"; exit 1; }
    if ( cd "$W/br_l" && "$BRAE" "$W/br_l" > run.log 2>&1 ); then
        ofn=$(grep -c '^Time = ' "$W/of_l/run.log")
        brn=$(grep -c '^Time = ' "$W/br_l/run.log")
        say "ARM 5  legacy driver takes $brn steps, OpenFOAM takes $ofn (start 0, end 1, dt 0.4)" \
            "$([ "$ofn" = "$brn" ] && echo ok || echo FAIL)"
        # C++ std::lround rounds HALF AWAY FROM ZERO; python round() is banker's and would give 2 here,
        # which would make this control silently agree with OpenFOAM and gate nothing.
        lr=$(python3 -c "import math; print(int(math.floor((1-0)/0.4 + 0.5)))")
        say "CONTROL  a rounded quotient would give $lr here, not $ofn" \
            "$([ "$lr" != "$ofn" ] && echo ok || echo FAIL)"
    else
        echo "  ARM 5    skipped: the legacy driver refuses $LEGACY_SRC"
    fi
fi

# The refusal: endTime not beyond startTime.
stage "$W/ref" 0 5 1
( cd "$W/ref" && simpleFoam > seed.log 2>&1 ) || { echo "FAIL: seeding the refusal arm"; exit 1; }
python3 - "$W/ref" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'startTime\s+\S+;', 'startTime       5;', s)
s = re.sub(r'endTime\s+\S+;',   'endTime         5;', s)
open(c, 'w').write(s)
PY
out=$( cd "$W/ref" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/ref" 2>&1 || true )
echo "$out" | grep -q "is not beyond the start time" \
    && say "REFUSAL  endTime not beyond startTime is refused by name" ok \
    || say "REFUSAL  endTime not beyond startTime is refused by name" FAIL

[ $fail -eq 0 ] && echo "PASS: the V2 time loop steps OpenFOAM's own inequality and names OpenFOAM's times"
exit $fail
