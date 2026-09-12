#!/usr/bin/env bash
# A nut WALL FUNCTION must be resolved through OpenFOAM's patch/group/regex rule, not by exact name.
#
# OpenFOAM resolves a boundaryField key against a patch in three passes -- exact name, literal group,
# then regex on the patch name or any group name, last match winning. createFields captured the nut wall
# FAMILY (nutk / nutU / nutLowRe) with `entry.name == patch.name`, so a case keying its walls by regex
# assigned NOTHING and every such patch kept the permissive nutkWallFunction default. The type check
# could not catch it: nutUWallFunction IS recognised, it is the NAME match that fails.
#
# Found on the gasMixing tutorial, whose 0/nut keys `"wall.*"` over walls_pipe_{air,fuel,main} with
# nutUWallFunction. From a still start (0/U is uniform (0 0 0)) magUp is 0, so nutU's fixed point stays
# below yPlusLam and OpenFOAM's wall nut is EXACTLY 0; brae ran nutk against k = 6 and wrote 1.4e-04.
# Worth U 1.716e-03 at iteration 1 -- momentum is solved before turbulence.correct(), so this is
# validate()'s nut reaching the very first momentum assembly -- amplifying to 1.2e-01 by iteration 5.
#
#   ARM 1  nutUWallFunction keyed by explicit NAME, brae vs real OpenFOAM  -> the family is right
#   ARM 2  the SAME case keyed by REGEX must give the IDENTICAL field      -> the resolution is right
#   ARM 3  CONTROL: nutU vs nutk on this fixture must DIFFER               -> ARM 2 is not vacuous
#   ARM 4  the regex-keyed case vs real OpenFOAM                           -> end to end
#
# ARM 3 is the load-bearing control. Without it ARM 2 passes on any fixture where the two families
# happen to agree -- which is exactly what validation/rhoSST does (kOmegaSST does not consume the wall
# family at all there), and picking that fixture would have made this gate green and empty.
#
# FAIL-PROOF, measured in-session before the fix, on gasMixing: wall nut relL2 6.4 / 6.0 / 52 against
# OpenFOAM on the three regex-keyed wall patches, OpenFOAM exactly 0 and brae finite.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-3}
FLOOR=${FLOOR:-1e-10}
IDENT=${IDENT:-1e-14}
DIFFER=${DIFFER:-1e-4}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoKE" ] || { echo "SKIP: fixture rhoKE missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # $1 variant
{
    d="$W/$1"; rm -rf "$d"; mkdir -p "$d"
    cp -r "$ROOT/validation/rhoKE/constant" "$ROOT/validation/rhoKE/system" "$d/"
    cp -r "$ROOT/validation/rhoKE/0.orig" "$d/0"
    N="$N" python3 - "$d" "$1" <<'PYEOF'
import os, re, sys
d, v = sys.argv[1], sys.argv[2]
p = os.path.join(d, '0/nut'); s = open(p).read()
body = {
 'regexU': '    "(hot|cold)Wall" { type nutUWallFunction; value uniform 0; }\n',
 'nameU' : '    hotWall   { type nutUWallFunction; value uniform 0; }\n'
           '    coldWall  { type nutUWallFunction; value uniform 0; }\n',
 'nameK' : '    hotWall   { type nutkWallFunction; value uniform 0; }\n'
           '    coldWall  { type nutkWallFunction; value uniform 0; }\n'}[v]
s2, n = re.subn(r'    hotWall   \{[^\n]*\}\n    coldWall  \{[^\n]*\}\n', body, s)
assert n == 1, 'the 0/nut wall mutation did not apply (fixture reformatted?)'
open(p, 'w').write(s2)
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
for k, val in [('endTime', os.environ['N']), ('writeInterval', os.environ['N']),
               ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'),
               ('deltaT', '1'), ('writeFormat', 'ascii'), ('writePrecision', '15')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, val), s)
s = re.sub(r'\bfunctions\s*\{.*\}\s*$', '', s, flags=re.S)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
open(f, 'w').write(s)
PYEOF
    ( cd "$d" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh ($1)"; exit 1; }
}

for v in nameU regexU nameK; do
    stage "$v"
    ( cd "$W/$v" && BRAE_U_SOLVER=ofOrder BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$W/$v" > log 2>&1 ) \
        || { tail -5 "$W/$v/log"; echo "FAIL: brae did not run ($v)"; exit 1; }
    grep -q "solvers/U solver" "$W/$v/log" && say "$v: no momentum-solver substitution" FAIL \
                                           || say "$v: no momentum-solver substitution" ok
done
for v in nameU regexU; do
    rm -rf "$W/of_$v"; cp -r "$W/$v" "$W/of_$v"; rm -rf "$W/of_$v"/[1-9]* "$W/of_$v"/log
    ( cd "$W/of_$v" && rhoSimpleFoam > log 2>&1 ) \
        || { tail -5 "$W/of_$v/log"; echo "FAIL: OpenFOAM did not run ($v)"; exit 1; }
done

W="$W" N="$N" FLOOR="$FLOOR" IDENT="$IDENT" DIFFER="$DIFFER" python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], os.environ['N']
FLOOR, IDENT, DIFFER = (float(os.environ[k]) for k in ('FLOOR', 'IDENT', 'DIFFER'))

def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if not m:
        u = re.search(r'internalField\s+uniform\s+(\(.*?\)|[-+0-9.eE]+)\s*;', s); v = u.group(1)
        return np.array([float(x) for x in v.strip('()').split()]) if v.startswith('(') else np.array([float(v)])
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])

def rel(a, b, f):
    x, y = read(os.path.join(W, a, N, f)), read(os.path.join(W, b, N, f))
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))

FIELDS = ('U', 'nut', 'k', 'epsilon')
ok = True
def check(label, got, bound, below):
    global ok
    good = (got < bound) if below else (got > bound)
    print('     %-58s %.4e  (%s %.1e)  %s'
          % (label, got, '<' if below else '>', bound, 'ok' if good else 'FAIL'))
    ok = ok and good

for f in FIELDS:
    check('ARM 1  nutU by NAME   vs OpenFOAM      %-8s' % f, rel('nameU', 'of_nameU', f), FLOOR, True)
for f in FIELDS:
    check('ARM 2  regex-keyed == name-keyed       %-8s' % f, rel('regexU', 'nameU', f), IDENT, True)
for f in FIELDS:
    check('ARM 3  CONTROL nutU vs nutk must differ %-8s' % f, rel('nameU', 'nameK', f), DIFFER, False)
for f in FIELDS:
    check('ARM 4  regex-keyed vs OpenFOAM         %-8s' % f, rel('regexU', 'of_regexU', f), FLOOR, True)
sys.exit(0 if ok else 1)
PYEOF

say "a nut wall function resolves through OpenFOAM's patch/group/regex rule" "$([ $fail = 0 ] && echo ok || echo FAIL)"
exit $fail
