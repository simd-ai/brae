#!/usr/bin/env bash
# forceCoeffs through a run on the legacy simpleFoam driver, under the case's OWN controls (forces_fo.cuh, the functionObject
# lifecycle of brae_time.cuh): every step its writeControl selects, written to postProcessing/forceCoeffs/<startTime>/
# coefficient.dat in OpenFOAM's format. Until brief #17 the driver evaluated forceCoeffs once after the loop and sampled it
# through a run only on request (BRAE_FORCE_INTERVAL, BRAE_FORCE_FROM, both retired); the boxy5 polar compares means over the
# last 300 iterations, which is what a history is for. The values are held to OpenFOAM's on the same fields by
# tests/functionobjects_forces_vs_openfoam.sh; this gate holds the cadence on validation/airFoil2D (a forceCoeffs entry added).
#   EVERY    no writeControl: OpenFOAM's default, a row every step, 1 to 20
#   ARM      writeControl timeStep, writeInterval 5: rows at 5 10 15 20
#   FROM     the same with timeStart 12: rows at 15 20
#   CONTROL  no forceCoeffs entry: no file
#   FINITE   every value of every row finite, and the history not all zero (a selection that matched nothing writes zeros)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/airFoil2D"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-66s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1
    return 0
}
stage()   # stage <dir> <extra entries of the forceCoeffs dictionary, or "none" for no entry>
{
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    sed -i -e 's/^endTime .*/endTime         20;/' -e 's/^writeInterval .*/writeInterval   20;/' "$1/system/controlDict"
    python3 - "$1/system/controlDict" "$2" <<'PY'
import re
import sys
p, extra = sys.argv[1], sys.argv[2]
s = open(p).read()
s = re.sub(r'functions\s*\{.*?\n\}\n', '', s, flags=re.S)
if extra != 'none':
    fc = """functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            (forces);
        patches         (walls);
        rho             rhoInf;
        rhoInf          1;
        magUInf         26;
        lRef            1;
        Aref            1;
        liftDir         (-0.139 0.990 0);
        dragDir         (0.990 0.139 0);
        CofR            (0.25 0 0);
        log             false;
%s
    }
}
""" % extra.replace(';', ';\n')
    s = s.rstrip() + '\n\n' + fc
open(p, 'w').write(s)
PY
}
rows()   # rows <dir>: the times of the rows written
{
    f="$1/postProcessing/forceCoeffs/0/coefficient.dat"
    [ -f "$f" ] && grep -v '^#' "$f" | awk '{printf "%d ", $1}' || echo "none"
}
stage "$W/e" ""
( cd "$W/e" && "$BRAE" -case . > log 2> err ) || { say "EVERY ran" FAIL; tail -3 "$W/e/err"; }
[ "$(rows "$W/e")" = "$(seq -s ' ' 1 20) " ] && say "EVERY a row every step, 1 to 20" ok || say "EVERY rows 1 to 20 (got $(rows "$W/e"))" FAIL
stage "$W/a" "writeControl timeStep; writeInterval 5;"
( cd "$W/a" && "$BRAE" -case . > log 2> err ) || { say "ARM ran" FAIL; tail -3 "$W/a/err"; }
[ "$(rows "$W/a")" = "5 10 15 20 " ] && say "ARM writeInterval 5: rows at 5 10 15 20" ok || say "ARM rows at 5 10 15 20 (got $(rows "$W/a"))" FAIL
stage "$W/f" "writeControl timeStep; writeInterval 5; timeStart 12;"
( cd "$W/f" && "$BRAE" -case . > log 2> err ) || say "FROM ran" FAIL
[ "$(rows "$W/f")" = "15 20 " ] && say "FROM timeStart 12: rows at 15 20" ok || say "FROM rows at 15 20 (got $(rows "$W/f"))" FAIL
stage "$W/c" none
( cd "$W/c" && "$BRAE" -case . > log 2> err ) || say "CONTROL ran" FAIL
[ "$(rows "$W/c")" = "none" ] && say "CONTROL no forceCoeffs entry: no file" ok || say "CONTROL no forceCoeffs entry: no file (got $(rows "$W/c"))" FAIL
python3 - "$W/e/postProcessing/forceCoeffs/0/coefficient.dat" <<'PY' && say "FINITE every value finite, the history not all zero" ok || say "FINITE every value finite, the history not all zero" FAIL
import math
import sys
head = [l[1:].split() for l in open(sys.argv[1]) if l.startswith('# Time')][0]
rows = [[float(x) for x in l.split()] for l in open(sys.argv[1]) if not l.startswith('#') and l.strip()]
ok = rows and all(all(math.isfinite(x) for x in r) for r in rows) and any(x != 0 for r in rows for x in r[1:])
sys.exit(0 if ok else 1)
PY
[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
