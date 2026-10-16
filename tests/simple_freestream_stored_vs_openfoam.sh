#!/usr/bin/env bash
# The FREESTREAM family on the legacy simpleFoam driver: valueFraction from U's STORED patch values, at
# OpenFOAM's own points of the iteration, against real OpenFOAM.
#
# OpenFOAM's freestreamVelocity takes `Up = *this` -- the value the last evaluate left
# (freestreamVelocityFvPatchVectorField.C:105) -- and freestreamPressure looks the same field up
# (freestreamPressureFvPatchScalarField.C:117). U's boundary is evaluated after the momentum solve and again
# after the velocity correction, and updateCoeffs runs at the momentum assembly, at the pressure assembly and
# inside that second evaluate. The driver re-derived Up from the cells once an iteration instead, at the
# momentum assembly, for U and p together: on the boxy5 UAV (4.94 M cells, one laminar iteration from
# potentialFoam's state) that left Ux's initial residual at 0.534657 against OpenFOAM's 0.553727 and the
# far-field cells' relaxed diagonal 1.9e-02 out (of the field's scale); carried, 0.553727 and 3.3e-16.
#
# THE ORACLE is serial OpenFOAM v2412 simpleFoam on validation/airFoil2D (SA, every far-field patch
# freestream), every linear solve pinned tight so neither code's solver stopping point enters: OpenFOAM runs
# 0 -> 3, both codes restart from ITS time 3 -- whose freestream patch values are OpenFOAM's stored ones, not
# a re-evaluation -- and run to 6. U, p and nuTilda are compared at 4 (one iteration: the momentum, pressure
# and correction evaluate points once each) and at 6.
# Both drivers: the legacy one (UbStored_, pbStored_) and the rebuilt one (simpleFoamV2.cu, the step's hooks).
# THE CONTROL is BRAE_CONTROL_FREESTREAM_REEVALUATE=1, the re-derivation this replaced, which must fail.
# PREMISE: the control and the arm differ at 4, so the restart state can witness the stored value.
set -u
SRC="${1:?case dir (validation/airFoil2D)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
OFRC=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -f "$OFRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFRC"; exit 77; }
set +u
set --
source "$OFRC" > /dev/null 2>&1
set -u
command -v simpleFoam > /dev/null || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

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
    python3 - "$1/system/controlDict" "$1/system/fvSolution" "$2" <<'PY'
import re
import sys
cd, fs, end = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(cd).read()
for k, v in (('endTime', end), ('writeInterval', '1'), ('writePrecision', '17'), ('writeFormat', 'ascii'),
             ('writeCompression', 'off'), ('startFrom', 'latestTime')):
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

# 1. OpenFOAM 0 -> 3, the restart state
stage "$W/seed" 3
( cd "$W/seed" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM's seed run"; tail -5 "$W/seed/log"; exit 1; }
[ -d "$W/seed/3" ] || { echo "FAIL: the seed run wrote no time 3"; exit 1; }

# 2. each arm from that time 3 to 6
arm()
{
    stage "$W/$1" 6
    rm -rf "$W/$1/0"
    cp -r "$W/seed/3" "$W/$1/3"
    rm -rf "$W/$1/3/uniform"
}
arm of
( cd "$W/of" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM 3 -> 6"; tail -5 "$W/of/log"; exit 1; }
for a in brae ctl v2 v2ctl
do
    arm "$a"
done
( cd "$W/brae" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2> err ) \
    || { echo "FAIL: brae 3 -> 6"; tail -5 "$W/brae/err"; exit 1; }
( cd "$W/ctl" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_FREESTREAM_REEVALUATE=1 "$BRAE" -case . > log 2> err ) \
    || { echo "FAIL: brae control 3 -> 6"; tail -5 "$W/ctl/err"; exit 1; }
# ...and the rebuilt driver, which keeps the same values through its step's hooks (simpleFoamV2.cu)
( cd "$W/v2" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case . > log 2> err ) \
    || { echo "FAIL: V2 3 -> 6"; tail -5 "$W/v2/err"; exit 1; }
( cd "$W/v2ctl" && BRAE_SIMPLEFOAM_V2=1 BRAE_CONTROL_FREESTREAM_REEVALUATE=1 "$BRAE" -case . > log 2> err ) \
    || { echo "FAIL: V2 control 3 -> 6"; tail -5 "$W/v2ctl/err"; exit 1; }

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
    for a in of brae ctl v2 v2ctl
    do
        [ -f "$W/$a/$t/U" ] || { say "time $t written by $a" FAIL; exit 1; }
    done
done
B4=$(diffs "$W/brae/4" "$W/of/4")
C4=$(diffs "$W/ctl/4" "$W/of/4")
B6=$(diffs "$W/brae/6" "$W/of/6")
C6=$(diffs "$W/ctl/6" "$W/of/6")
P4=$(diffs "$W/ctl/4" "$W/brae/4")
V4=$(diffs "$W/v2/4" "$W/of/4")
V6=$(diffs "$W/v2/6" "$W/of/6")
VC4=$(diffs "$W/v2ctl/4" "$W/of/4")
echo "  V2 arm     t4: $V4"
echo "  V2 arm     t6: $V6"
echo "  V2 control t4: $VC4"
echo "  arm     t4: $B4"
echo "  arm     t6: $B6"
echo "  control t4: $C4"
echo "  control t6: $C6"
echo "  control - arm t4: $P4"

# MEASURED on the GB10 (2026-10-16), L2 relative to OpenFOAM: the arm U 7.3e-13 p 5.2e-12 at t4 and U 6.5e-13
# p 6.2e-12 at t6 -- round-off; the control U 1.0e-07 p 1.1e-06 at t4 and U 1.8e-07 p 5.5e-06 at t6. The rebuilt
# driver (BRAE_SIMPLEFOAM_V2=1): U 2.2e-14 p 3.9e-13 at t4, U 9.2e-14 p 2.3e-12 at t6; its control as the
# legacy one's. The bound sits two orders above the arms and three below the controls.
python3 - "$B4" "$B6" "$C4" "$C6" "$P4" "$V4" "$V6" "$VC4" <<'PY'
import sys


def parse(s):
    return {k: float(v) for k, v in (t.split('=') for t in s.split())}


b4, b6, c4, c6, p4, v4, v6, vc4 = (parse(s) for s in sys.argv[1:9])
BOUND = {'U': 1e-10, 'p': 1e-10}
rc = 0
for f in ('U', 'p'):
    for name, d in (('ARM t4', b4), ('ARM t6', b6), ('V2 ARM t4', v4), ('V2 ARM t6', v6)):
        ok = d[f] < BOUND[f]
        print('  %-40s %.3e < %.0e   %s' % (name + ' ' + f, d[f], BOUND[f], 'ok' if ok else 'FAIL'))
        rc |= 0 if ok else 1
for f in ('U', 'p'):
    ok = c4[f] > BOUND[f]
    print('  %-40s %.3e > %.0e   %s' % ('CONTROL t4 ' + f + ' fails the bound', c4[f], BOUND[f], 'ok' if ok else 'FAIL'))
    rc |= 0 if ok else 1
for f in ('U', 'p'):
    ok = vc4[f] > BOUND[f]
    print('  %-40s %.3e > %.0e   %s' % ('V2 CONTROL t4 ' + f + ' fails the bound', vc4[f], BOUND[f], 'ok' if ok else 'FAIL'))
    rc |= 0 if ok else 1
ok = p4['U'] > 0
print('  %-40s %.3e > 0       %s' % ('PREMISE control and arm differ (U t4)', p4['U'], 'ok' if ok else 'FAIL'))
rc |= 0 if ok else 1
sys.exit(rc)
PY
[ $? -eq 0 ] || rc=1
[ $rc -eq 0 ] && echo "PASS" || echo "FAIL"
exit $rc
