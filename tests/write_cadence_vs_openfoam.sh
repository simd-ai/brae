#!/usr/bin/env bash
# WHEN brae writes a time directory, and WHAT it puts in it -- both against real OpenFOAM.
#
# OpenFOAM writes a time directory when Time::operator++ set writeTime_ (Time.C:1103-1160: under
# `writeControl timeStep` that is `!(timeIndex % writeInterval)`), and on residualControl convergence,
# where simpleControl::loop() calls runTime.writeAndEnd() (simpleControl.C:152). Nowhere else: Time::run()
# has no end-of-run write, so an endTime that is not a multiple of the interval leaves NO directory.
# Every registered AUTO_WRITE field goes into it: U, p, phi (createPhi.H:46), the turbulence fields, and
# for kOmegaSSTLM the two transition scalars (kOmegaSSTLM.C:439, :453).
#
# The V2 driver did neither. It wrote exactly one directory, after the loop, whatever controlDict said --
# a 1000-iteration T3A run with `writeInterval 500` wrote 1000/ and nothing else -- and it wrote five of
# the seven fields it solved: no phi, no ReThetat, no gammaInt. So an LM case could not be restarted from
# brae's own output, by brae or by OpenFOAM, and the two transition fields had no oracle at all.
#
#   ARM A   `writeInterval 7; endTime 20`: both codes write {7, 14} and NOT 20.
#   ARM B   each of those directories holds the same set of field files OpenFOAM's does.
#   ARM C   `purgeWrite 1` on the same run: both keep only 14/.
#   ARM D   OpenFOAM restarts from brae's 14/ (`startFrom latestTime`) and runs one iteration -- so the
#           files are well-formed OpenFOAM fields, phi/ReThetat/gammaInt included, not merely present.
#   CONTROL the fixture's own `writeInterval 500` with `endTime 20`: OpenFOAM writes nothing. brae must
#           write nothing either and SAY why. This is the old behaviour's exact failure -- the previous
#           driver wrote 20/ here -- so the control is the fail-proof of ARM A.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/T3A}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <writeInterval> <purgeWrite>
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$1/"
    cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
d, wi, pw = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^endTime .*',          'endTime         20;', s, flags=re.M)
s = re.sub(r'^writeControl .*',     'writeControl    timeStep;', s, flags=re.M)
s = re.sub(r'^writeInterval .*',    'writeInterval   %s;' % wi, s, flags=re.M)
s = re.sub(r'^purgeWrite .*',       'purgeWrite      %s;' % pw, s, flags=re.M)
s = re.sub(r'^writePrecision .*',   'writePrecision  15;', s, flags=re.M)
s = re.sub(r'^writeCompression .*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PY
}
# Time directories only: `0.orig` also starts with a digit, and 0/ is the start, not output.
timedirs() { ( cd "$1" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9]+(\.[0-9]+)?$' | grep -vE '^0$' | sort -n | tr '\n' ' ' ); }
fieldset() { ( cd "$1" && ls -p | grep -v '/$' | sort | tr '\n' ' ' ); }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
prep "$W/of7"   7   0;  prep "$W/br7"   7   0
prep "$W/ofp"   7   1;  prep "$W/brp"   7   1
prep "$W/of500" 500 0;  prep "$W/br500" 500 0
for d in of7 ofp of500; do ( cd "$W/$d" && simpleFoam > log 2>&1 ) || { echo "FAIL: simpleFoam did not run in $d"; tail -8 "$W/$d/log"; exit 1; }; done
for d in br7 brp br500; do ( cd "$W/$d" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/$d" > log 2>&1 ) || { echo "FAIL: brae crashed in $d"; tail -8 "$W/$d/log"; exit 1; }; done

# ARM A -- the set of time directories.
o=$(timedirs "$W/of7"); b=$(timedirs "$W/br7")
echo "  ARM A  time directories  OpenFOAM: [$o]  brae: [$b]"
[ "$o" = "7 14 " ] || { say "ARM A  OpenFOAM wrote {7,14} and not 20 (premise)" FAIL; }
[ "$b" = "$o" ] && say "ARM A  brae writes exactly the directories OpenFOAM writes" ok \
                || say "ARM A  brae writes exactly the directories OpenFOAM writes" FAIL

# ARM B -- the set of field files in each.
for t in 7 14; do
    fo=$(fieldset "$W/of7/$t"); fb=$(fieldset "$W/br7/$t")
    echo "  ARM B  $t/  OpenFOAM: [$fo]"; echo "  ARM B  $t/  brae:     [$fb]"
    [ "$fo" = "$fb" ] && say "ARM B  $t/ holds the same field files as OpenFOAM's" ok \
                      || say "ARM B  $t/ holds the same field files as OpenFOAM's" FAIL
    for f in phi ReThetat gammaInt; do
        [ -f "$W/br7/$t/$f" ] || say "ARM B  $t/$f is written (it was not, before)" FAIL
    done
done

# ARM C -- purgeWrite.
o=$(timedirs "$W/ofp"); b=$(timedirs "$W/brp")
echo "  ARM C  purgeWrite 1  OpenFOAM: [$o]  brae: [$b]"
[ "$b" = "$o" ] && [ "$o" = "14 " ] && say "ARM C  purgeWrite 1 keeps only 14/, as OpenFOAM does" ok \
                                     || say "ARM C  purgeWrite 1 keeps only 14/, as OpenFOAM does" FAIL

# ARM D -- OpenFOAM restarts from brae's output.
mkdir -p "$W/restart"; cp -r "$W/br7/constant" "$W/br7/system" "$W/br7/14" "$W/restart/"
python3 - "$W/restart" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'^startFrom .*', 'startFrom       latestTime;', s, flags=re.M)
s = re.sub(r'^endTime .*',   'endTime         15;', s, flags=re.M)
open(c, 'w').write(s)
PY
if ( cd "$W/restart" && simpleFoam > log 2>&1 ) && grep -q "^Time = 15" "$W/restart/log" && grep -q "Solving for gammaInt" "$W/restart/log"; then
    say "ARM D  OpenFOAM restarts from brae's 14/ and solves gammaInt from brae's own field" ok
    grep -m1 "Solving for Ux" "$W/restart/log" | sed 's/^/         /'
else
    say "ARM D  OpenFOAM restarts from brae's 14/ and solves gammaInt from brae's own field" FAIL
    tail -6 "$W/restart/log" | sed 's/^/         /'
fi

# CONTROL -- the fixture's own interval: nothing is written, and brae says why.
o=$(timedirs "$W/of500"); b=$(timedirs "$W/br500")
echo "  CONTROL  writeInterval 500, endTime 20  OpenFOAM: [$o]  brae: [$b]"
[ -z "$o" ] || say "CONTROL  OpenFOAM writes nothing at a non-interval endTime (premise)" FAIL
[ -z "$b" ] && say "CONTROL  brae writes nothing either -- the old driver wrote 20/ here" ok \
            || say "CONTROL  brae writes nothing either -- the old driver wrote 20/ here" FAIL
grep -q "is not a write time" "$W/br500/log" && say "CONTROL  ...and its log says why" ok \
                                              || say "CONTROL  ...and its log says why" FAIL

[ $fail -eq 0 ] && echo "PASS: brae writes when OpenFOAM writes, what OpenFOAM writes, and OpenFOAM can restart from it"
exit $fail
