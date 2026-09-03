#!/usr/bin/env bash
# THE p0 TABLE AS OPENFOAM WRITES IT: `p0 table;` + `p0Coeffs { values 2 ( (t v) (t v) ); }` -- read on a
# RESTART from OpenFOAM's own output, both arms, and refused by name where brae cannot evaluate it.
#
# OpenFOAM writes a Function1 table in the coefficients form (Function1::writeData, TableBase::writeEntries),
# never inline, so every restart from its output with a uniformTotalPressure patch carries it. The reader
# expected `table (` and fell over the ';' with a raw tokeniser error (queue item 20); it now reads the
# p0Coeffs entry (Function1New.C reads a table's coefficients from optionalSubDict(name + "Coeffs")),
# accepts `values` with OpenFOAM's list size, and refuses `interpolationScheme` other than linear and
# `outOfBounds` other than clamp (TableBase.C:76's default) by name rather than evaluating the default.
#
#   ARM 1  restart: rhoTP with a constant-valued written table, real rhoSimpleFoam 0 -> 3 writing every
#          step; both mirror arms restart FROM OpenFOAM's 1/ and must match its 3/ (fields at t=2 and 3).
#   ARM 2  a RAMP in the written form (item 14's refusal, now reachable through OpenFOAM's spelling) is
#          refused by name on both arms; a hand-edited `interpolationScheme cubic;` and `outOfBounds
#          repeat;` likewise.
#   CONTROL  OpenFOAM's 1/ and 3/ must differ by >= 10x the bound (the restart is not a no-op).
#
# Measured floors on the restart (2026-09-03, both arms alike): p 2.9e-12, T 9.6e-13, U 1.3e-11, rho 2.6e-12.
# Bounds ~30x. What the first run of this arm found: the totalPressure patch seeded its VALUE from p0, where
# OpenFOAM's constructor takes the written `value` (totalPressureFvPatchScalarField.C:69-73), so a restart
# started the inlet at 100200 instead of the written 100094.86 and both arms read U 1.1e-01 / p 1.8e-04 at
# t=2..3. Value and p0 are separate slots now, on the host class and in the device projection.
# Fail-proofs: (a) the p0Coeffs branch disabled in the reader -> arm 1 does not run (the written form is
# refused by name, `table (no p0Coeffs entry)`) and the field rows FAIL; (b) the value seeding reverted to
# p0 -> both arms restart but read U 1.1e-01 against OpenFOAM at t=2..3, FAIL.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
SRC="$ROOT/validation/rhoTP"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
P_BOUND=${P_BOUND:-1e-10}; T_BOUND=${T_BOUND:-3e-11}; U_BOUND=${U_BOUND:-4e-10}; RHO_BOUND=${RHO_BOUND:-8e-11}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-76s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # $1 dst  $2 p0 table entry  $3 endTime
{
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    P0="$2" ENDT="$3" python3 - "$1" <<'P0STAGE'
import os, re, sys
d = sys.argv[1]
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', os.environ['ENDT']), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0; maxIter 2000;', s)
open(f, 'w').write(s)
p = os.path.join(d, '0/p'); s = open(p).read()
s, n = re.subn(r'inlet\s*\{[^}]*\}', 'inlet { type uniformTotalPressure; p0 %s; value uniform 100200; }' % os.environ['P0'], s, count=1)
assert n == 1
open(p, 'w').write(s)
P0STAGE
}
restart()   # $1 src (OpenFOAM's run)  $2 dst  $3 arm  -> runs from 1/ to 3
{
    rm -rf "$2"; cp -r "$1" "$2"; rm -rf "$2"/2 "$2"/3 "$2"/log.* "$2"/run.log
    python3 - "$2/system/controlDict" <<'P0RESTART'
import re, sys
c = sys.argv[1]; s = open(c).read()
s = re.sub(r'\bstartTime\s+[^;]*;', 'startTime 1;', s); s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 3;', s)
open(c, 'w').write(s)
P0RESTART
    ( cd "$2" && BRAE_RHOSIMPLEFOAM_MIRROR=$3 "$BIN" -case "$2" > run.log 2>&1 )
}

# ---- arm 1: restart from OpenFOAM's written directory ---------------------------------------------
stage "$W/of" "table ((0 100200) (10 100200))" 3
( cd "$W/of" && rhoSimpleFoam > run.log 2>&1 ) || { tail -5 "$W/of/run.log"; echo "FAIL: OpenFOAM did not run"; exit 1; }
[ -d "$W/of/3" ] || { echo "FAIL: OpenFOAM wrote no 3/"; exit 1; }
grep -q "p0Coeffs" "$W/of/1/p" && say "OpenFOAM wrote the table in the coefficients form (p0 table; p0Coeffs)" ok \
    || say "OpenFOAM wrote the table in the coefficients form (p0 table; p0Coeffs)" FAIL
for arm in 1 cuda; do
    restart "$W/of" "$W/re_$arm" $arm || { tail -4 "$W/re_$arm/run.log"; say "arm $arm restarts from OpenFOAM's 1/ with the written table" FAIL; continue; }
    grep -q "^Time = 2" "$W/re_$arm/run.log" && [ -d "$W/re_$arm/3" ] \
        && say "arm $arm restarts from OpenFOAM's 1/ with the written table" ok \
        || { tail -4 "$W/re_$arm/run.log"; say "arm $arm restarts from OpenFOAM's 1/ with the written table" FAIL; }
done
W="$W" P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" U_BOUND="$U_BOUND" RHO_BOUND="$RHO_BOUND" python3 - <<'P0CMP' || fail=1
import os, re, sys
import numpy as np
W = os.environ['W']
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'internalField\s+uniform\s+\(?([^);]+)\)?;', s)
    return np.array([float(x) for x in u.group(1).split()])
def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
bounds = {f: float(os.environ[k]) for f, k in (('p', 'P_BOUND'), ('T', 'T_BOUND'), ('U', 'U_BOUND'), ('rho', 'RHO_BOUND'))}
ok = True
for f, b in bounds.items():
    of3 = read(os.path.join(W, 'of', '3', f))
    # CONTROL: OpenFOAM moved between 1/ and 3/, so a mirror that wrote its start state back would fail.
    r0 = rel(read(os.path.join(W, 'of', '1', f)), of3)
    if r0 < 10 * b:
        print('     %-4s OpenFOAM 1/ vs 3/ %.3e is inside 10x the bound -- vacuous   FAIL' % (f, r0)); ok = False
    for arm in ('1', 'cuda'):
        worst = 0.0
        for t in (2, 3):
            p = os.path.join(W, 're_' + arm, str(t), f)
            if not os.path.exists(p):
                worst = float('inf'); continue
            worst = max(worst, rel(read(p), read(os.path.join(W, 'of', str(t), f))))
        good = worst < b
        print('     arm %-4s %-4s max over t=2..3 vs OpenFOAM %.4e   (bound %.1e)   %s' % (arm, f, worst, b, 'ok' if good else 'FAIL'))
        ok = ok and good
sys.exit(0 if ok else 1)
P0CMP
say "both arms track OpenFOAM after restarting from its written table" "$([ $fail = 0 ] && echo ok || echo FAIL)"

# ---- arm 2: refusals by name through the written form ---------------------------------------------
stage "$W/ramp" "table ((0 100200) (10 100400))" 1
( cd "$W/ramp" && rhoSimpleFoam > run.log 2>&1 ) || { tail -5 "$W/ramp/run.log"; echo "FAIL: OpenFOAM did not run the ramp"; exit 1; }
for arm in 1 cuda; do
    restart "$W/ramp" "$W/ramp_$arm" $arm || true
    grep -q "time-varying p0 table" "$W/ramp_$arm/run.log" && ! grep -q "^Time = 2" "$W/ramp_$arm/run.log" \
        && say "a written RAMP table is refused by name on restart (arm $arm)" ok \
        || { tail -3 "$W/ramp_$arm/run.log"; say "a written RAMP table is refused by name on restart (arm $arm)" FAIL; }
done
for mut in "interpolationScheme cubic" "outOfBounds repeat"; do
    rm -rf "$W/mut"; cp -r "$W/of" "$W/mut"; rm -rf "$W/mut"/2 "$W/mut"/3
    MUT="$mut" python3 - "$W/mut/1/p" <<'P0MUT'
import os, re, sys
p = sys.argv[1]; s = open(p).read()
s, n = re.subn(r'(p0Coeffs\s*\{)', r'\1 %s;' % os.environ['MUT'], s, count=1); assert n == 1
open(p, 'w').write(s)
P0MUT
    for arm in 1 cuda; do
        restart "$W/mut" "$W/mut_$arm" $arm || true
        grep -q "table $mut" "$W/mut_$arm/run.log" && ! grep -q "^Time = 2" "$W/mut_$arm/run.log" \
            && say "a written table with \`$mut\` is refused by name (arm $arm)" ok \
            || { tail -3 "$W/mut_$arm/run.log"; say "a written table with \`$mut\` is refused by name (arm $arm)" FAIL; }
    done
done

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
