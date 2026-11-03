#!/usr/bin/env bash
# THE V2 simpleFoam DRIVER RESTARTED FROM A CONVERGED 3-D STATE, against OpenFOAM's own restart.
#
# One iteration from OpenFOAM's converged windAroundBuildingsBox 400/ (JUPITER jobs 2667755, 2669843) the V2 driver
# (BRAE_SIMPLEFOAM_V2=1) left OpenFOAM by U 7.6e-05 and p 2.9e-03, where the legacy driver was 2.6e-16 and 1.9e-12.
# The stages named it (tools/dumpSimpleFoam against V2's BRAE_STAGE_DUMP_DIR): U, phi and nuEff as assembled and the
# momentum matrix at 1e-16 of OpenFOAM's, the momentum source 3.0e-01 off in z on the outlet's 200 cells alone. V2
# seeded the first iteration's boundary nut from k's and epsilon's inletOutlet outlet at their inletValue, where
# OpenFOAM's validate() reads the patch values as read. A start from rest cannot see it: the field is the inletValue
# there, which is why every V2 gate from 0/ passed.
#
# THE ORACLE: serial OpenFOAM (simpleFoam v2412) from the same 400/, which foamFormatConvert writes at 17 digits and
# both codes read back to the same bits, every linear solve at 1e-12 relTol 0, N iterations, each written. OpenFOAM's
# one-ulp twin (every internal k value of the start one ulp up) is what its own rounding moves.
#
# CHECKS, relative L2 against OpenFOAM of U p k epsilon nut after 1 and N iterations:
#   PREMISE  the start can show the defect: some outlet face's k differs from k's inletValue
#   PREMISE  OpenFOAM's twin stays under BOUND: the bound is above what OpenFOAM's own rounding moves
#   LEGACY   the legacy driver on the same staging, under BOUND: the staging is right
#   V2       under BOUND
#   CONTROL  BRAE_IO_STORED=0, V2's first boundary nut from the inletValue as before: over BOUND on U and p after 1
#
# Fail-proof: the V2 driver at 89fab3b7 (the 18-patch candidate before this change) fails the V2 check.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/windAroundBuildingsBox}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
FROM=${FROM:-400}     # the converged OpenFOAM time the fixture ships
N=${N:-5}             # iterations from it; 1 and N are judged
BOUND=${BOUND:-1e-10}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC/$FROM" ] || { echo "SKIP: fixture $SRC/$FROM missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); [ -n "${KEEP_W:-}" ] && echo "work dir $W" || trap 'rm -rf "$W"' EXIT
fail=0

stage()   # stage <dir> [twin]: the fixture from FROM/ for N iterations, each written
{
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/log.* "$1"/postProcessing
    # every time but FROM/ out of the way (0/ stays: startFrom latestTime starts at FROM/)
    local t
    for t in "$1"/[1-9]*; do [ -e "$t" ] && [ "$(basename "$t")" != "$FROM" ] && rm -rf "$t"; done
    python3 - "$1" "$FROM" "$N" <<'PY'
import re, sys
d, frm, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in (('startFrom', 'latestTime'), ('endTime', str(frm + n)), ('writeInterval', '1'), ('writeControl', 'timeStep'),
             ('writeFormat', 'ascii'), ('writePrecision', '17'), ('writeCompression', 'off')):
    s, m = re.subn(r'(?m)^%s\s+\S+;' % k, '%s %s;' % (k.ljust(15), v), s)
    if m != 1: sys.exit('controlDict: %s' % k)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
# every linear solve pinned, so the codes differ by what they assemble and not by where a loose solve stopped
s = re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance       1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;',    'relTol          0;', s)
open(f, 'w').write(s)
PY
    [ $? -eq 0 ] || { echo "FAIL: staging $1"; exit 1; }
    ( cd "$1" && foamFormatConvert -noConstant -time "$FROM" > log.foamFormatConvert 2>&1 ) \
        || { echo "FAIL: foamFormatConvert ($1)"; tail -5 "$1/log.foamFormatConvert"; exit 1; }
    [ -n "${2:-}" ] || return 0
    python3 - "$1/$FROM/k" <<'PY'
import math, re, sys
# OpenFOAM's own spread on a restart: every internal k value one ulp up, the boundary values as written
f = sys.argv[1]; s = open(f).read()
m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n?\(', s)
if not m: sys.exit('twin: %s holds no ascii nonuniform internalField' % f)
e = s.index(')', m.end())
v = [float(x) for x in s[m.end():e].split()]
if len(v) != int(m.group(1)): sys.exit('twin: %s lists %d of %s values' % (f, len(v), m.group(1)))
open(f, 'w').write(s[:m.end()] + '\n' + '\n'.join(repr(math.nextafter(x, math.inf)) for x in v) + '\n' + s[e:])
PY
    [ $? -eq 0 ] || { echo "FAIL: the twin's start ($1)"; exit 1; }
}

openfoam()   # openfoam <dir>
{
    ( cd "$1" && simpleFoam > log.simpleFoam 2>&1 ) || { echo "FAIL: OpenFOAM in $1"; tail -5 "$1/log.simpleFoam"; exit 1; }
}

braerun()   # braerun <dir> [VAR=value ...]
{
    local d=$1
    shift
    ( cd "$d" && env "$@" "$BRAE" "$d" > log.brae 2>&1 )
    echo $? > "$d/rc"
}

cat > "$W/rel.py" <<'PY'
import re, sys
import numpy as np
def internal(p):
    s = open(p).read(); t = s[s.index('internalField'):]
    m = re.match(r'internalField\s+uniform\s+([^;]+);', t)
    if m:
        return np.array([float(x) for x in m.group(1).strip('()').split()])
    m = re.match(r'internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\(', t)
    b = t[m.end():]; b = b[:b.index('\n)')]
    a = np.array(b.replace('(', ' ').replace(')', ' ').split(), dtype=float)
    return a.reshape(int(m.group(2)), -1) if m.group(1) == 'vector' else a
# rel.py <run> <oracle> <time> <bound> <fields...>: one line, the worst field first; exit 1 over the bound
run, ref, t, bound = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
out = []
for f in sys.argv[5:]:
    try:
        a, b = internal('%s/%s/%s' % (run, t, f)), internal('%s/%s/%s' % (ref, t, f))
    except (IOError, OSError, ValueError, AttributeError):
        out.append((float('inf'), f, 'missing'))
        continue
    if a.shape != b.shape or not np.all(np.isfinite(a)):
        out.append((float('inf'), f, 'shape or non-finite'))
        continue
    out.append((float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)), f, ''))
worst = max(out)
print('worst %s %.3e | %s' % (worst[1], worst[0], ' '.join('%s %.3e%s' % (f, r, (' ' + w) if w else '') for r, f, w in out)))
sys.exit(0 if worst[0] <= bound else 1)
PY

FIELDS="U p k epsilon nut"
T1=$((FROM + 1))
TN=$((FROM + N))

stage "$W/of"
stage "$W/oft" twin
stage "$W/v2"
stage "$W/legacy"
stage "$W/ctl"

# PREMISE: an outlet face whose stored k is not the inletValue, or the inletValue seed reads what is stored
python3 - "$W/v2/$FROM/k" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
b = s[s.index('boundaryField'):]
blk = re.search(r'\n\s*outlet\s*\{(.*?)\n\s*\}', b, flags=re.S).group(1)
iv = float(re.search(r'inletValue\s+uniform\s+([0-9.eE+-]+);', blk).group(1))
m = re.search(r'value\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\n?\(([^)]*)\)', blk)
v = [float(x) for x in m.group(2).split()] if m else [float(re.search(r'value\s+uniform\s+([0-9.eE+-]+);', blk).group(1))]
d = max(abs(x - iv) for x in v)
print('  outlet k: %d stored values, inletValue %g, largest difference %.3e' % (len(v), iv, d))
sys.exit(0 if d > 0 else 1)
PY
if [ $? -eq 0 ]; then echo "ok   PREMISE the outlet's stored k is not its inletValue"
else echo "FAIL PREMISE the outlet's stored k is its inletValue: the control cannot fail"; fail=1; fi

openfoam "$W/of"
openfoam "$W/oft"
braerun "$W/v2" BRAE_SIMPLEFOAM_V2=1
braerun "$W/legacy"
braerun "$W/ctl" BRAE_SIMPLEFOAM_V2=1 BRAE_IO_STORED=0

grep -q "simpleFoam v2" "$W/v2/log.brae" || { echo "FAIL PREMISE the V2 arm ran the V2 driver"; fail=1; }

for t in $T1 $TN
do
    m=$(python3 "$W/rel.py" "$W/oft" "$W/of" "$t" "$BOUND" $FIELDS)
    [ $? -eq 0 ] && echo "ok   PREMISE OpenFOAM's twin at $t: $m" || { echo "FAIL PREMISE OpenFOAM's twin at $t: $m"; fail=1; }
    for a in legacy v2
    do
        if [ "$(cat "$W/$a/rc")" != 0 ]; then echo "FAIL $a at $t: brae exited $(cat "$W/$a/rc")"; fail=1; continue; fi
        tag=$(echo $a | tr a-z A-Z)
        m=$(python3 "$W/rel.py" "$W/$a" "$W/of" "$t" "$BOUND" $FIELDS)
        [ $? -eq 0 ] && echo "ok   $tag at $t: $m" || { echo "FAIL $tag at $t: $m"; fail=1; }
    done
done

if [ "$(cat "$W/ctl/rc")" != 0 ]; then echo "FAIL CONTROL BRAE_IO_STORED=0: brae exited $(cat "$W/ctl/rc")"; fail=1
else
    m=$(python3 "$W/rel.py" "$W/ctl" "$W/of" "$T1" "$BOUND" U p)
    [ $? -ne 0 ] && echo "ok   CONTROL BRAE_IO_STORED=0 is red at $T1: $m" || { echo "FAIL CONTROL BRAE_IO_STORED=0 stayed green at $T1: $m"; fail=1; }
fi

[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
