#!/usr/bin/env bash
# freestreamVelocity in OpenFOAM-13's form, BRAE_FREESTREAM_VELOCITY_UP=average, against real OpenFOAM-13.
#
# OpenFOAM-13 takes Up = 0.5*(patchInternalField() + *this) (OpenFOAM-13 freestreamVelocityFvPatchVectorField.C)
# where v2412, the version brae ports, takes *this; its freestreamPressure is v2412's. A case written for
# OpenFOAM-13 (the boxy5 UAV) is another problem under v2412's form -- OpenFOAM v2412 diverged on it -- so a run
# that must reproduce OpenFOAM-13 asks for its form by name, and this holds it to OpenFOAM-13 itself.
#
# THE ORACLE is serial OpenFOAM-13 `foamRun -solver incompressibleFluid` on validation/airFoil2D written in
# OpenFOAM-13's dictionaries (physicalProperties, momentumTransport), laminar, every solve pinned: OpenFOAM-13 runs
# 0 -> 3, both codes restart from ITS time 3 and run to 6, U and p compared at 4 and 6.
# THE CONTROL is the same brae run in v2412's form (the variable unset), which must fail.
# SKIPS without OpenFOAM-13 at /opt/openfoam13 (the H100 box has it; the GB10 has no package for its arch).
set -u
SRC="${1:?case dir (validation/airFoil2D)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
OFRC=${OF13RC:-/opt/openfoam13/etc/bashrc}
[ -f "$OFRC" ] || { echo "SKIP: no OpenFOAM-13 at $OFRC"; exit 77; }
set +u
set --
source "$OFRC" > /dev/null 2>&1
set -u
command -v foamRun > /dev/null || { echo "SKIP: foamRun not on PATH"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "$W"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
rc=0
say()
{
    printf '  %-60s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && rc=1
    return 0
}

# stage <dir> <endTime>: the fixture with every solve pinned (tolerance 1e-12, relTol 0), no residualControl,
# no function objects, ascii at 17 digits, a write every iteration
stage()
{
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    cp -r "$SRC/0" "$1/0"
    # LAMINAR: the far field is then the only boundary the two codes treat differently. Turbulent, the
    # Spalding wall rows differ by 1.7e-06 (of the field's scale) in the momentum system and that alone moves
    # U by 2.6e-05 at t4, with or without the stored values -- a gap of its own that would hide this one.
    sed -i -E 's/^(simulationType\s+)[A-Za-z]+;/\1laminar;/' "$1/constant/turbulenceProperties"
    grep -qE '^simulationType\s+laminar;' "$1/constant/turbulenceProperties" || { echo "FAIL: laminar not staged"; exit 1; }
    # ...and OpenFOAM-13's names for the same two files, beside v2412's (each code reads its own)
    NU=$(sed -n 's/^nu[^0-9.-]*\([-0-9.eE+]*\);.*/\1/p' "$1/constant/transportProperties" | head -1)
    [ -n "$NU" ] || { echo "FAIL: nu not found in transportProperties"; exit 1; }
    printf 'FoamFile { format ascii; class dictionary; object physicalProperties; }\nviscosityModel  constant;\nnu              [0 2 -1 0 0 0 0] %s;\n' "$NU" > "$1/constant/physicalProperties"
    printf 'FoamFile { format ascii; class dictionary; object momentumTransport; }\nsimulationType  laminar;\n' > "$1/constant/momentumTransport"
    python3 - "$1/system/controlDict" "$1/system/fvSolution" "$2" <<'PY'
import re
import sys
cd, fs, end = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(cd).read()
for k, v in (('endTime', end), ('writeInterval', '1'), ('writePrecision', '17'), ('writeFormat', 'ascii'),
             ('writeCompression', 'off'), ('startFrom', 'latestTime'), ('solver', 'incompressibleFluid')):
    s, n = re.subn(r'(?m)^%s\s+[^;]*;' % k, '%-16s %s;' % (k, v), s)
    if n == 0:
        s = s.replace('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //',
                      '%-16s %s;\n' % (k, v), 1)
i = s.find('functions')
if i >= 0:
    j = s.index('{', i)
    depth = 0
    for e in range(j, len(s)):
        depth += {'{': 1, '}': -1}.get(s[e], 0)
        if depth == 0:
            break
    s = s[:i] + 'functions {}' + s[e + 1:]
open(cd, 'w').write(s)
f = open(fs).read()
f = re.sub(r'(?m)^(\s*tolerance\s+)[^;]*;', r'\g<1>1e-12;', f)
f = re.sub(r'(?m)^(\s*relTol\s+)[^;]*;', r'\g<1>0;', f)
f = re.sub(r'(?m)^(\s*smoother\s+GaussSeidel;)', r'\1\n        maxIter         5000;', f)
f = re.sub(r'residualControl\s*\{[^}]*\}', '', f)
open(fs, 'w').write(f)
PY
}

# 1. OpenFOAM-13 0 -> 3, the restart state
stage "$W/seed" 3
( cd "$W/seed" && foamRun > log 2>&1 ) || { echo "FAIL: OpenFOAM-13's seed run"; tail -5 "$W/seed/log"; exit 1; }
[ -d "$W/seed/3" ] || { echo "FAIL: the seed run wrote no time 3"; exit 1; }

arm()
{
    stage "$W/$1" 6
    rm -rf "${W:?}/$1/0"
    cp -r "$W/seed/3" "$W/$1/3"
    rm -rf "${W:?}/$1/3/uniform"
}
arm of
( cd "$W/of" && foamRun > log 2>&1 ) || { echo "FAIL: OpenFOAM-13 3 -> 6"; tail -5 "$W/of/log"; exit 1; }
for a in brae ctl
do
    arm "$a"
done
( cd "$W/brae" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_FREESTREAM_VELOCITY_UP=average "$BRAE" -case . > log 2> err ) \
    || { echo "FAIL: brae 3 -> 6"; tail -5 "$W/brae/err"; exit 1; }
( cd "$W/ctl" && env -u BRAE_FREESTREAM_VELOCITY_UP BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2> err ) \
    || { echo "FAIL: brae control 3 -> 6"; tail -5 "$W/ctl/err"; exit 1; }
grep -q "freestreamVelocity Up: OpenFOAM-13's form" "$W/brae/err" && say "ARM announces OpenFOAM-13's form" ok \
    || say "ARM announces OpenFOAM-13's form" FAIL
grep -q "freestreamVelocity Up: OpenFOAM-13's form" "$W/ctl/err" && say "CONTROL runs v2412's form, unannounced" FAIL \
    || say "CONTROL runs v2412's form, unannounced" ok
# 3. L2(brae - of)/L2(of) of the internal fields
diffs()
{
    python3 - "$1" "$2" <<'PY'
import re
import sys

import numpy as np


def read(fn):
    b = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', b)
    n = int(m.group(2))
    nc = 3 if m.group(1) == 'vector' else 1
    body = b[m.end():]
    body = body[:body.index('\n)')]
    return np.array(body.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)


a, b = sys.argv[1], sys.argv[2]
out = []
for f in ('U', 'p'):
    x, y = read(a + '/' + f), read(b + '/' + f)
    out.append('%s=%.3e' % (f, np.linalg.norm(x - y) / np.linalg.norm(y)))
print(' '.join(out))
PY
}
for t in 4 6
do
    for a in of brae ctl
    do
        [ -f "$W/$a/$t/U" ] || { say "time $t written by $a" FAIL; exit 1; }
    done
done
B4=$(diffs "$W/brae/4" "$W/of/4")
C4=$(diffs "$W/ctl/4" "$W/of/4")
B6=$(diffs "$W/brae/6" "$W/of/6")
C6=$(diffs "$W/ctl/6" "$W/of/6")
P4=$(diffs "$W/ctl/4" "$W/brae/4")
echo "  arm     t4: $B4"
echo "  arm     t6: $B6"
echo "  control t4: $C4"
echo "  control t6: $C6"
echo "  control - arm t4: $P4"

# MEASURED on the H100 box (2026-10-16, OpenFOAM-13 from dl.openfoam.org): the arm U 7.3e-13 p 5.2e-12 at t4,
# U 6.5e-13 p 6.2e-12 at t6 -- round-off; the control (v2412's form) U 3.9e-08 p 2.5e-06 at t4, U 3.0e-07
# p 1.4e-05 at t6. The bound sits two orders above the arm and two below the control.
python3 - "$B4" "$B6" "$C4" "$C6" "$P4" <<'PY'
import sys


def parse(s):
    return {k: float(v) for k, v in (t.split('=') for t in s.split())}


b4, b6, c4, c6, p4 = (parse(s) for s in sys.argv[1:6])
BOUND = {'U': 1e-10, 'p': 1e-10}
rc = 0
for f in ('U', 'p'):
    for name, d in (('ARM t4', b4), ('ARM t6', b6)):
        ok = d[f] < BOUND[f]
        print('  %-40s %.3e < %.0e   %s' % (name + ' ' + f, d[f], BOUND[f], 'ok' if ok else 'FAIL'))
        rc |= 0 if ok else 1
for f in ('U', 'p'):
    ok = c4[f] > BOUND[f]
    print('  %-40s %.3e > %.0e   %s' % ('CONTROL t4 ' + f + ' fails the bound', c4[f], BOUND[f], 'ok' if ok else 'FAIL'))
    rc |= 0 if ok else 1
ok = p4['U'] > 0
print('  %-40s %.3e > 0       %s' % ('PREMISE control and arm differ (U t4)', p4['U'], 'ok' if ok else 'FAIL'))
rc |= 0 if ok else 1
sys.exit(rc)
PY
[ $? -eq 0 ] || rc=1
[ $rc -eq 0 ] && echo "PASS" || echo "FAIL"
exit $rc
