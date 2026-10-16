#!/usr/bin/env bash
# forceCoeffs sampled through a run on the legacy simpleFoam driver: BRAE_FORCE_INTERVAL=N writes Cd/Cl/Cm every
# N iterations from iteration BRAE_FORCE_FROM to postProcessing/forceCoeffs/<startTime>/coefficient.dat. The
# driver otherwise evaluates forceCoeffs once, after the loop; the boxy5 polar compares means over the last
# 300 iterations, as the package's OpenFOAM-13 results do.
#   ARM      interval 5 over 20 iterations of validation/airFoil2D (a forceCoeffs entry added): rows at 5, 10,
#            15, 20, and the row at 20 is the end-of-run evaluation -- the same state, the same numbers.
#   FROM     BRAE_FORCE_FROM=12: rows at 15 and 20 only.
#   CONTROL  the variable unset: no file, and the end-of-run line as before.
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
stage()
{
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    sed -i -e 's/^endTime .*/endTime         20;/' -e 's/^writeInterval .*/writeInterval   20;/' "$1/system/controlDict"
    python3 - "$1/system/controlDict" <<'PY'
import re
import sys
p = sys.argv[1]
s = open(p).read()
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
        pitchAxis       (0 0 1);
    }
}
"""
s = re.sub(r'functions\s*\{.*?\n\}\n', '', s, flags=re.S)
s = s.rstrip() + '\n\n' + fc
open(p, 'w').write(s)
PY
}
rows()   # rows <dir>: the iterations sampled
{
    f="$1/postProcessing/forceCoeffs/0/coefficient.dat"
    [ -f "$f" ] && grep -v '^#' "$f" | awk '{printf "%d ", $1}' || echo "none"
}

stage "$W/a"
( cd "$W/a" && BRAE_FORCE_INTERVAL=5 "$BRAE" -case . > log 2> err ) || { say "ARM ran" FAIL; tail -3 "$W/a/err"; }
[ "$(rows "$W/a")" = "5 10 15 20 " ] && say "ARM samples at 5 10 15 20" ok || say "ARM samples at 5 10 15 20 (got $(rows "$W/a"))" FAIL
python3 - "$W/a/log" "$W/a/postProcessing/forceCoeffs/0/coefficient.dat" <<'PY'
import re
import sys
log, dat = sys.argv[1], sys.argv[2]
m = re.search(r'forceCoeffs \(.*\):\s+Cd=(\S+)\s+Cl=(\S+)\s+Cm=(\S+)', open(log).read())
last = [l.split() for l in open(dat) if not l.startswith('#')][-1]
end = [float(x) for x in m.groups()]
row = [float(x) for x in last[1:4]]
ok = all(abs(a - b) <= 1e-6 * max(abs(a), 1e-12) for a, b in zip(end, row))
print('  %-66s %s' % ('ARM row 20 = the end-of-run evaluation (%s vs %s)' % (' '.join('%.6e' % x for x in row),
                                                                           ' '.join('%.6e' % x for x in end)),
                       'ok' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || fail=1
stage "$W/f"
( cd "$W/f" && BRAE_FORCE_INTERVAL=5 BRAE_FORCE_FROM=12 "$BRAE" -case . > log 2> err ) || say "FROM ran" FAIL
[ "$(rows "$W/f")" = "15 20 " ] && say "FROM 12 samples at 15 20" ok || say "FROM 12 samples at 15 20 (got $(rows "$W/f"))" FAIL
stage "$W/c"
( cd "$W/c" && env -u BRAE_FORCE_INTERVAL -u BRAE_FORCE_FROM "$BRAE" -case . > log 2> err ) || say "CONTROL ran" FAIL
[ "$(rows "$W/c")" = "none" ] && grep -q "^forceCoeffs (" "$W/c/log" \
    && say "CONTROL no file, the end-of-run line as before" ok || say "CONTROL no file, the end-of-run line as before" FAIL
[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
