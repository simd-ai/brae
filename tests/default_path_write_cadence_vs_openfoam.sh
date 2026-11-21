#!/usr/bin/env bash
# WRITE CADENCE ON THE DEFAULT simpleFoam DRIVER, against real OpenFOAM: the default-driver arms of
# tests/write_cadence_vs_openfoam.sh, which runs BRAE_SIMPLEFOAM_V2=1 (the opt-in rebuilt driver) only.
#
# OpenFOAM writes a time directory when Time::operator++ set writeTime_ (Time.C:1103-1160: under `writeControl
# timeStep`, `!(timeIndex % writeInterval)`), and nowhere else (Time::run() has no end-of-run write). The case is the
# committed pitzDaily (kEpsilon), which the default driver runs (T3A, the opt-in gate's kOmegaSSTLM fixture, is the
# rebuilt driver's), with `writeInterval 7; endTime 20` and residualControl emptied so neither code stops early.
#
#   ARM B   7/ and 14/ each hold the same set of field files OpenFOAM's do.
#   ARM D   OpenFOAM restarts from brae's 14/ (`startFrom latestTime`) and runs one iteration, solving k and
#           epsilon from brae's own files: they are well-formed OpenFOAM fields.
#
# LEFT OUT, red today on this driver for a known defect (audit E-G13: gpuSimpleFoam.cu:1152 writes the final step
# whether or not it is a write time) and not loosened: the opt-in gate's ARM A (OpenFOAM writes {7, 14}, this driver
# {7, 14, 20}), ARM C (purgeWrite 1: OpenFOAM keeps {14}, this driver {20}) and CONTROL (writeInterval 500: OpenFOAM
# writes nothing, this driver 20/), MEASURED 2026-11-21. They belong here once the defect is fixed; the `written` line
# below prints what both codes wrote on every run, so the day they agree is visible.
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
prep() {   # prep <dir> <writeInterval> <purgeWrite>
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$1/0"; else cp -r "$SRC/0" "$1/0"; fi
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
d, wi, pw = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
for key, value in (('startFrom', 'startTime'), ('startTime', '0'), ('endTime', '20'), ('deltaT', '1'),
                   ('writeControl', 'timeStep'), ('writeInterval', wi), ('purgeWrite', pw), ('writePrecision', '15'),
                   ('writeCompression', 'off')):
    s, n = re.subn(r'(?m)^%s\s+[^;]*;' % key, '%-15s %s;' % (key, value), s)
    if n == 0:
        s += '\n%-15s %s;\n' % (key, value)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
open(f, 'w').write(re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl\n    {\n    }', s, flags=re.S))
PY
}
# time directories only, the start excluded
timedirs() { ( cd "$1" && ls -d [0-9]* 2> /dev/null | grep -E '^[0-9]+(\.[0-9]+)?$' | grep -vE '^0$' | sort -n | tr '\n' ' ' ); }
fieldset() { ( cd "$1" && ls -p | grep -v '/$' | sort | tr '\n' ' ' ); }

prep "$W/of7" 7 0
prep "$W/br7" 7 0
( cd "$W/of7" && simpleFoam > log 2>&1 ) < /dev/null || { echo "FAIL: simpleFoam did not run"; tail -8 "$W/of7/log"; exit 1; }
( cd "$W/br7" && "$BRAE" -case "$W/br7" > log 2>&1 ) < /dev/null || { echo "FAIL: brae did not run"; tail -8 "$W/br7/log"; exit 1; }
grep -q "^brae (device-resident)" "$W/br7/log" && grep -q "colour rule: U, k, epsilon swept in colour order" "$W/br7/log" \
    && say "the default driver ran, with the colour rule (its banner and colour line)" ok \
    || say "the default driver ran, with the colour rule (its banner and colour line)" FAIL

o=$(timedirs "$W/of7"); b=$(timedirs "$W/br7")
echo "  written  OpenFOAM: [$o]  brae: [$b]   (ARM A, left out: E-G13)"
[ "$o" = "7 14 " ] || say "premise: OpenFOAM wrote {7, 14} and not 20" FAIL

# ARM B -- the set of field files in each of OpenFOAM's directories
for t in 7 14
do
    fo=$(fieldset "$W/of7/$t"); fb=$( [ -d "$W/br7/$t" ] && fieldset "$W/br7/$t" )
    echo "  ARM B  $t/  OpenFOAM: [$fo]"
    echo "  ARM B  $t/  brae:     [$fb]"
    [ -n "$fo" ] && [ "$fo" = "$fb" ] && say "ARM B  $t/ holds the same field files as OpenFOAM's" ok \
                                     || say "ARM B  $t/ holds the same field files as OpenFOAM's" FAIL
done

# ARM D -- OpenFOAM restarts from brae's 14/
mkdir -p "$W/restart"
cp -r "$W/br7/constant" "$W/br7/system" "$W/br7/14" "$W/restart/" 2> /dev/null
python3 - "$W/restart" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'(?m)^startFrom\s+[^;]*;', 'startFrom       latestTime;', s)
s = re.sub(r'(?m)^endTime\s+[^;]*;',   'endTime         15;', s)
open(c, 'w').write(s)
PY
if ( cd "$W/restart" && simpleFoam > log 2>&1 ) < /dev/null && grep -q "^Time = 15" "$W/restart/log" \
    && grep -q "Solving for epsilon" "$W/restart/log" && grep -q "Solving for k" "$W/restart/log"
then
    say "ARM D  OpenFOAM restarts from brae's 14/ and solves k and epsilon from brae's files" ok
    grep -m1 "Solving for Ux" "$W/restart/log" | sed 's/^/         /'
else
    say "ARM D  OpenFOAM restarts from brae's 14/ and solves k and epsilon from brae's files" FAIL
    tail -6 "$W/restart/log" | sed 's/^/         /'
fi

[ $fail -eq 0 ] && echo "PASS: the default driver writes OpenFOAM's field set, and OpenFOAM restarts from it"
exit $fail
