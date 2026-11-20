#!/usr/bin/env bash
# THE TIME LOOP OF THE DEFAULT simpleFoam DRIVER, against real OpenFOAM: the default-driver arms of
# tests/timeloop_vs_openfoam.sh, whose own arms run BRAE_SIMPLEFOAM_V2=1 (the opt-in rebuilt driver) on simpleBoxIO,
# a case the default driver refuses (deviceAdjustPhi). Here the committed pitzDaily, which it runs.
#
# OpenFOAM's Time::run() tests `value() < endTime_ - 0.5*deltaT_` (Time.C:785), Time::operator++ advances the value by
# deltaT (Time.C:1067) and Time::timeName formats it (Time.C:721-728). Three of the opt-in gate's (startTime, endTime,
# deltaT) shapes, each run by both codes from the same start with residualControl emptied and a write every step:
#   ARM      the ordered list of time names brae prints is OpenFOAM's (`Time = ` lines): the count, the naming and
#            the per-step print in one assertion. 0 1 0.4 is the shape where a rounded quotient takes one step more.
#   ARM      the last time directory brae writes is OpenFOAM's last.
#   REFUSAL  endTime not beyond startTime is refused by name, not run.
#   CONTROL  at least two shapes have a time sequence that is not 1..N, or a driver counting iterations would pass.
# The restart shape (5 to 8) starts both codes from OpenFOAM's own 5/, written by a seeding run.
#
# LEFT OUT, red today on this driver for a defect, and not loosened: the opt-in gate's shape 0 2.5 0.5. The driver
# reads endTime as an integer (gpuSimpleFoam.cu:351, `controlDict.intOr("endTime", 1000)`), so it stops at 2 where
# OpenFOAM runs to 2.5. MEASURED 2026-11-21: brae [0.5 1 1.5 2], OpenFOAM [0.5 1 1.5 2 2.5]. It belongs here once the
# driver reads endTime as OpenFOAM's scalar time.
#
# Run through tests/default_path_run.sh: brae runs as `brae -case <dir>` with no BRAE_* variable.
set -u
BRAE="${1:?brae binary (the front door)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${2:-$ROOT/validation/pitzDaily}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
BRAE="$(cd "$(dirname "$BRAE")" && pwd)/$(basename "$BRAE")"
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
set +u
set --
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # stage <dir> <startTime> <endTime> <deltaT>
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$1/0"; else cp -r "$SRC/0" "$1/0"; fi
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import re, sys
d, st, et, dt = sys.argv[1:5]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
for key, value in (('startFrom', 'startTime'), ('startTime', st), ('endTime', et), ('deltaT', dt),
                   ('writeControl', 'timeStep'), ('writeInterval', '1'), ('purgeWrite', '0'), ('writePrecision', '15')):
    s, n = re.subn(r'(?m)^%s\s+[^;]*;' % key, '%-15s %s;' % (key, value), s)
    if n == 0:
        s += '\n%-15s %s;\n' % (key, value)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
open(f, 'w').write(re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl\n    {\n    }', s, flags=re.S))
PY
}
retime()   # retime <dir> <startTime> <endTime>
{
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'(?m)^startTime\s+[^;]*;', 'startTime       %s;' % sys.argv[2], s)
s = re.sub(r'(?m)^endTime\s+[^;]*;',   'endTime         %s;' % sys.argv[3], s)
open(c, 'w').write(s)
PY
}
seq_of()  { grep -oP '^Time = \S+' "$1/run.log" | sed 's/Time = //' | tr '\n' ' '; }
lastdir() { ( cd "$1" && ls -d [0-9]* 2> /dev/null | grep -E '^[0-9.e+-]+$' | sort -g | tail -1 ); }

nontrivial=0
for shape in "0 5 1" "0 1 0.4" "5 8 1"
do
    set -- $shape
    st=$1; et=$2; dt=$3
    stage "$W/of" "$st" "$et" "$dt"
    stage "$W/br" "$st" "$et" "$dt"
    if [ "$st" != 0 ]
    then
        # the start directory is OpenFOAM's own, for both codes
        retime "$W/of" 0 "$st"
        ( cd "$W/of" && simpleFoam > seed.log 2>&1 ) < /dev/null || { echo "FAIL: seeding $st"; exit 1; }
        cp -r "$W/of/$st" "$W/br/"
        retime "$W/of" "$st" "$et"
    fi
    ( cd "$W/of" && simpleFoam > run.log 2>&1 ) < /dev/null || { echo "FAIL: OpenFOAM on $shape"; exit 1; }
    ( cd "$W/br" && "$BRAE" -case "$W/br" > run.log 2>&1 ) < /dev/null \
        || { echo "FAIL: brae stopped on $shape"; tail -8 "$W/br/run.log"; exit 1; }
    [ "$st" = 0 ] && [ "$et" = 5 ] && { grep -q "^brae (device-resident)" "$W/br/run.log" \
        && grep -q "colour rule: U, k, epsilon swept in colour order" "$W/br/run.log" \
        && say "the default driver ran, with the colour rule (its banner and colour line)" ok \
        || say "the default driver ran, with the colour rule (its banner and colour line)" FAIL; }
    o="$(seq_of "$W/of")"; b="$(seq_of "$W/br")"
    say "start $st end $et dt $dt  times [$b] == OpenFOAM [$o]" "$([ -n "$o" ] && [ "$o" = "$b" ] && echo ok || echo FAIL)"
    od="$(lastdir "$W/of")"; bd="$(lastdir "$W/br")"
    say "start $st end $et dt $dt  brae wrote ${bd}/ last, OpenFOAM ${od}/" "$([ "$od" = "$bd" ] && echo ok || echo FAIL)"
    n=$(echo "$o" | wc -w)
    [ "$o" != "$(seq 1 "$n" | tr '\n' ' ')" ] && nontrivial=$((nontrivial + 1))
done
say "CONTROL  $nontrivial shapes have a time sequence that is not 1..N" "$([ "$nontrivial" -ge 2 ] && echo ok || echo FAIL)"

# REFUSAL -- endTime not beyond startTime, from OpenFOAM's own 5/
stage "$W/ref" 0 5 1
( cd "$W/ref" && simpleFoam > seed.log 2>&1 ) < /dev/null || { echo "FAIL: seeding the refusal arm"; exit 1; }
retime "$W/ref" 5 5
out=$( cd "$W/ref" && "$BRAE" -case "$W/ref" 2>&1 < /dev/null )
rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q "is not beyond the start time" \
    && say "REFUSAL  endTime not beyond startTime is refused by name" ok \
    || say "REFUSAL  endTime not beyond startTime is refused by name (exit $rc)" FAIL

[ $fail -eq 0 ] && echo "PASS: the default driver steps OpenFOAM's inequality and names OpenFOAM's times"
exit $fail
