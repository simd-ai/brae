#!/usr/bin/env bash
# THE START DIRECTORY IS NAMED BY THE START TIME'S VALUE, NOT BY ITS SPELLING.
#
# OpenFOAM reads `startTime` as a scalar (Time.C:155-158) and reads the fields from timeName(value) at the case's
# timeFormat and timePrecision (TimeIO.C:336-338, Time.C:721-728); when no such directory exists it tries every
# precision from 18 down and adopts the lowest that finds one (Time.C:209-250). `startFrom` defaults to latestTime
# (Time.C:149-153). brae returned controlDict's startTime WORD as the directory, so `startTime 0.;` -- as ANSA writes
# the AutoCFD template -- looked for "0." where OpenFOAM reads "0" and `brae check` refused a case OpenFOAM runs; it
# defaulted startFrom to startTime; and its writer named directories %g whatever timeFormat and timePrecision said.
#
#   ARMS     each start shape below is run by real OpenFOAM (the oracle) and by brae on identical copies:
#              - both start, or both stop;
#              - when both start: brae prints OpenFOAM's `Time = ` sequence and writes OpenFOAM's directories;
#              - `brae check` agrees with the run: it names OpenFOAM's start directory, or refuses where both stop.
#            simpleFoam on pitzDaily (the default `brae` path, SIMPLEC) and pimpleFoam on pimFR.
#   PREMISE  in the spelling arms the controlDict's startTime word is NOT the directory OpenFOAM reads, so a code
#            reading the word cannot pass them.
#   CONTROL  run this script with BRAE_BIN set to a binary from before the fix (lc6): the spelling, format,
#            precision and default-latestTime arms fail there.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
PITZ="${1:-$ROOT/validation/pitzDaily}"
PIMFR="${2:-$ROOT/validation/pimFR}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BRAE" ]                         || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$PITZ/constant/polyMesh/points" ] || { echo "SKIP: fixture $PITZ missing"; exit 77; }
[ -f "$PIMFR/system/blockMeshDict" ]    || { echo "SKIP: fixture $PIMFR missing"; exit 77; }
[ -f "$OFBASHRC" ]                     || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-96s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1 || true
}

# pimFR's mesh, once, by OpenFOAM's own blockMesh
cp -r "$PIMFR" "$W/pimfr-mesh"
( cd "$W/pimfr-mesh" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on pimFR"; exit 1; }

# stage <dir> <pitz|pimfr> <controlDict edits: key=value, key=- deletes> -- the case with its 0/ and nothing else
stage()
{
    local d=$1 fx=$2
    shift 2
    rm -rf "$d"
    mkdir -p "$d"
    if [ "$fx" = pitz ]
    then
        cp -r "$PITZ/constant" "$PITZ/system" "$d/"
        cp -r "$PITZ/0" "$d/0"
    else
        cp -r "$W/pimfr-mesh/constant" "$W/pimfr-mesh/system" "$d/"
        cp -r "$PIMFR/0.orig" "$d/0"
    fi
    rm -f "$d"/constant/polyMesh/.brae_*
    python3 - "$d" "$@" << 'PY'
import re, sys
d = sys.argv[1]
c = d + '/system/controlDict'
s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
for kv in ['writeControl=timeStep', 'writeInterval=1', 'purgeWrite=0'] + sys.argv[2:]:
    k, v = kv.split('=', 1)
    pat = r'(?<![A-Za-z])' + k + r'\s+[^;]*;'
    if v == '-':
        s = re.sub(pat, '', s)
    elif re.search(pat, s):
        s = re.sub(pat, '%s %s;' % (k, v), s)
    else:
        s = s.replace('stopAt', '%s %s;\nstopAt' % (k, v), 1)
open(c, 'w').write(s)
f = d + '/system/fvSolution'
s = open(f).read()
open(f, 'w').write(re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S))
PY
}

# seed <dir> <name>: a further time directory holding the start fields, as a run would have left it
seed()
{
    cp -r "$1/0" "$1/$2"
}

times()
{
    grep -oP '^Time = \S+' "$1" | sed 's/Time = //' | tr '\n' ' '
}

dirs()
{
    ( cd "$1" && for t in [0-9]* -[0-9]*
                 do
                     [ -d "$t" ] && echo "$t"
                 done | sort -g | tr '\n' ' ' )
}

# the time directories a run added: after minus before
written()
{
    local t out=""
    for t in $2
    do
        case " $1 " in
            *" $t "*) ;;
            *) out="$out$t " ;;
        esac
    done
    echo "$out"
}

# arm <label> <pitz|pimfr> <expected OpenFOAM start dir, or "-" where OpenFOAM stops> <spelling or -> <seed dirs, comma
#     separated, or -> <keep0|drop0> <controlDict edits...>
# drop0 removes the fixture's 0/ once the seeds are made from it, so the only field directory is the one OpenFOAM reads
# and a code reading any other cannot start.
arm()
{
    local label=$1 fx=$2 expect=$3 spell=$4 seeds=$5 zero=$6
    shift 6
    local of=$W/of-$label br=$W/br-$label
    for d in "$of" "$br"
    do
        stage "$d" "$fx" "$@"
        if [ "$seeds" != "-" ]
        then
            for s in ${seeds//,/ }
            do
                seed "$d" "$s"
            done
        fi
        [ "$zero" = drop0 ] && rm -rf "$d/0"
    done
    local app=simpleFoam
    [ "$fx" = pimfr ] && app=pimpleFoam
    local before
    before="$(dirs "$of")"
    ( cd "$of" && $app > log.of 2>&1 )
    local ofrc=$?
    ( cd "$br" && "$BRAE" check -case "$br" > log.check 2>&1 )
    local ckrc=$?
    ( cd "$br" && "$BRAE" -case "$br" > log.brae 2>&1 )
    local brrc=$?
    if [ "$spell" != "-" ] && [ "$expect" != "-" ]
    then
        say "PREMISE $label: controlDict says startTime $spell, OpenFOAM reads $expect/" \
            "$([ "$spell" != "$expect" ] && echo ok || echo FAIL)"
    fi
    if [ "$expect" = "-" ]
    then
        say "$label: OpenFOAM stops (exit $ofrc); brae stops (exit $brrc)" \
            "$([ $ofrc != 0 ] && [ $brrc != 0 ] && echo ok || echo FAIL)"
        say "$label: brae check refuses it (exit $ckrc: $(grep -m1 -o 'verdict: [A-Z ]*' "$br/log.check"))" \
            "$([ $ckrc -ge 2 ] && echo ok || echo FAIL)"
        return
    fi
    if [ $ofrc != 0 ]
    then
        say "$label: OpenFOAM runs it (exit $ofrc: $(grep -m1 -A3 'FOAM FATAL' "$of/log.of" | tail -n 2 | tr -s ' ' | tr '\n' ' '))" FAIL
        return
    fi
    if [ $brrc != 0 ]
    then
        say "$label: brae runs it (exit $brrc: $(grep -m1 -iE 'error|refus|no start|cannot' "$br/log.brae"))" FAIL
    else
        local ot bt ow bw
        ot="$(times "$of/log.of")"
        bt="$(times "$br/log.brae")"
        ow="$(written "$before" "$(dirs "$of")")"
        bw="$(written "$before" "$(dirs "$br")")"
        say "$label: times [$bt] == OpenFOAM [$ot]" "$([ -n "$ot" ] && [ "$ot" = "$bt" ] && echo ok || echo FAIL)"
        say "$label: brae wrote [$bw] == OpenFOAM [$ow]" "$([ -n "$ow" ] && [ "$ow" = "$bw" ] && echo ok || echo FAIL)"
    fi
    local ckt
    ckt="$(grep -oP 'fields at time \K\S+(?=,)' "$br/log.check")"
    say "$label: brae check reads time ${ckt:-none} (exit $ckrc), OpenFOAM starts from $expect/" \
        "$([ "$ckt" = "$expect" ] && [ $ckrc -lt 2 ] && echo ok || echo FAIL)"
}

# simpleFoam (pitzDaily, deltaT 1)
arm s-plain     pitz 0       -        -       keep0 startFrom=startTime startTime=0 endTime=2
arm s-dot       pitz 0       0.       -       keep0 startFrom=startTime startTime=0. endTime=2
arm s-sci       pitz 0.001   1e-3     0.001   drop0 startFrom=startTime startTime=1e-3 endTime=2.001
arm s-fixed     pitz 0.000   0        0.000   drop0 startFrom=startTime startTime=0 endTime=2 timeFormat=fixed timePrecision=3
arm s-raise     pitz 0.12345 -        0.12345 drop0 startFrom=startTime startTime=0.12345 endTime=2.12345 timePrecision=3
arm s-latest    pitz 3       -        3       keep0 startFrom=latestTime startTime=0 endTime=5
arm s-absent    pitz 3       -        3       keep0 startFrom=- startTime=0 endTime=5
arm s-nodir     pitz -       -        -       keep0 startFrom=startTime startTime=0.5 endTime=2
arm s-badfrom   pitz -       -        -       keep0 startFrom=beginning startTime=0 endTime=2
# pimpleFoam (pimFR, deltaT 1e-4)
arm p-dot       pimfr 0      0.       -       keep0 startFrom=startTime startTime=0. endTime=0.0002
arm p-fixed     pimfr 0.0000 0        0.0000  drop0 startFrom=startTime startTime=0 endTime=0.0002 timeFormat=fixed timePrecision=4
arm p-absent    pimfr 0.0002 -        0.0002  keep0 startFrom=- startTime=0 endTime=0.0004

[ $fail = 0 ] && echo "START TIME PASS" || echo "START TIME FAIL"
exit $fail
