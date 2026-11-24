#!/usr/bin/env bash
# RUN-TO-RUN IDENTITY ON THE DEFAULT PATHS (audit E-G22): the same case run twice through `brae -case`, as a customer
# runs it, must write the same bytes -- every file of every time directory. A comparison against OpenFOAM is only as
# sharp as brae's own run-to-run floor, and a solver that is bit-reproducible must stay so.
#
#   default_path_determinism.sh <brae> <case>      case: rhoBox | damBreak | shockTube
#     rhoBox     rhoSimpleFoam's default (pre-mirror) driver: validation/rhoBox, 40 iterations, one write
#     damBreak   interFoam's device loop (the front door adds -device): laminar/damBreak/damBreak as its Allrun
#                stages it (blockMesh, setFields), to 0.05 s, adjustTimeStep as shipped
#     shockTube  rhoCentralFoam's CUDA arm: the tutorial as shipped (blockMesh, setFields), to 0.002 s
#   CONTROL  one byte changed in a copy of the first run is found by the same comparison.
# tests/determinism_gate.sh is the simpleFoam form of this check (it compares named fields at a time named by the
# iteration count); these cases' time names and field sets differ, so the comparison here is of whole directories.
# pimpleFoam is not here: it is not bit-reproducible run to run (1.4e-8 on pitzDaily, audit E), so its spread is
# measured and reported, not asserted.
#
# Run through tests/default_path_run.sh: brae runs with no BRAE_* variable.
set -u
BRAE="${1:?brae binary (the front door)}"
KIND="${2:?rhoBox, damBreak or shockTube}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
BRAE="$(cd "$(dirname "$BRAE")" && pwd)/$(basename "$BRAE")"
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available (its blockMesh and setFields stage the case)"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set +u
set --
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
edit() {   # edit <controlDict> key value ...
    python3 - "$@" <<'PY'
import re, sys
c = sys.argv[1]; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
kv = sys.argv[2:]
for key, value in zip(kv[::2], kv[1::2]):
    s, n = re.subn(r'(?m)^%s\s+[^;]*;' % key, '%-15s %s;' % (key, value), s)
    if n == 0:
        s += '\n%-15s %s;\n' % (key, value)
open(c, 'w').write(s)
PY
}

case $KIND in
    rhoBox)
        SRC="$ROOT/validation/rhoBox"
        [ -d "$SRC" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
        cp -r "$SRC" "$W/stage"
        rm -rf "$W/stage"/[1-9]* "$W/stage"/log.*
        mkdir -p "$W/stage/0" && cp "$W/stage"/0.orig/* "$W/stage/0/"
        ( cd "$W/stage" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
        edit "$W/stage/system/controlDict" endTime 40 writeControl timeStep writeInterval 40 purgeWrite 0
        # residualControl emptied: both runs take all 40 iterations whatever the stopping point
        python3 - "$W/stage/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
open(p, 'w').write(re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl\n    {\n    }', s, flags=re.S))
PY
        ;;
    damBreak)
        SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
        [ -d "$SRC" ] || { echo "SKIP: no damBreak tutorial at $SRC"; exit 77; }
        cp -r "$SRC" "$W/stage"
        rm -rf "$W/stage"/[1-9]* "$W/stage"/log.* "$W/stage/0"
        cp -r "$W/stage/0.orig" "$W/stage/0"
        ( cd "$W/stage" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) \
            || { echo "FAIL: blockMesh/setFields"; exit 1; }
        edit "$W/stage/system/controlDict" endTime 0.05 writeControl adjustableRunTime writeInterval 0.05
        ;;
    shockTube)
        SRC="$TUT/compressible/rhoCentralFoam/shockTube"
        [ -d "$SRC" ] || { echo "SKIP: no shockTube tutorial at $SRC"; exit 77; }
        cp -r "$SRC" "$W/stage"
        rm -rf "$W/stage"/[1-9]* "$W/stage"/0.[0-9]* "$W/stage"/log.* "$W/stage/0"
        cp -r "$W/stage/0.orig" "$W/stage/0"
        ( cd "$W/stage" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) \
            || { echo "FAIL: blockMesh/setFields"; exit 1; }
        edit "$W/stage/system/controlDict" endTime 0.002 writeInterval 0.001
        ;;
    *) echo "FAIL: unknown case '$KIND'"; exit 2 ;;
esac

for r in A B
do
    cp -r "$W/stage" "$W/$r"
    ( cd "$W/$r" && "$BRAE" -case "$W/$r" > log.brae 2>&1 ) < /dev/null \
        || { echo "FAIL: run $r did not complete"; tail -8 "$W/$r/log.brae"; exit 1; }
done
# what each run wrote: the time directories that were not in the staged case
written() { ( cd "$1" && for d in [0-9]*; do [ -d "$d" ] && [ ! -e "$W/stage/$d" ] && find "$d" -type f; done | sort ); }
written "$W/A" > "$W/filesA"
written "$W/B" > "$W/filesB"
n=$(wc -l < "$W/filesA")
[ "$n" -gt 0 ] || { say "$KIND: run A wrote time directories" FAIL; exit 1; }
cmp -s "$W/filesA" "$W/filesB" && say "$KIND: both runs wrote the same $n files ($(cut -d/ -f1 "$W/filesA" | sort -u | tr '\n' ' '))" ok \
                                || say "$KIND: both runs wrote the same files" FAIL
differ=0
while read -r f
do
    cmp -s "$W/A/$f" "$W/B/$f" || { differ=$((differ + 1)); [ $differ -le 5 ] && echo "     differs: $f"; }
done < "$W/filesA"
[ $differ = 0 ] && say "$KIND: every file byte-identical run to run" ok \
                || say "$KIND: every file byte-identical run to run ($differ differ)" FAIL

# CONTROL: the comparison sees one changed byte
f=$(grep -m1 -E '/(p|T|U|alpha.water)$' "$W/filesA" || head -1 "$W/filesA")
cp "$W/A/$f" "$W/control"
python3 - "$W/control" <<'PY'
import sys
p = sys.argv[1]; b = bytearray(open(p, 'rb').read())
i = len(b) // 2
b[i] = ord('7') if b[i] != ord('7') else ord('3')
open(p, 'wb').write(bytes(b))
PY
cmp -s "$W/A/$f" "$W/control" && say "CONTROL  one changed byte in $f is found" FAIL || say "CONTROL  one changed byte in $f is found" ok

[ $fail -eq 0 ] && echo "PASS: $KIND writes the same bytes run to run on the default path"
exit $fail
