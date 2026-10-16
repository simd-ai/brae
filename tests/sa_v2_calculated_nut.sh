#!/usr/bin/env bash
# The rebuilt simpleFoam driver (BRAE_SIMPLEFOAM_V2=1) on Spalart-Allmaras with a `calculated` nut patch.
#
# V2 evaluates kEpsilon's nut = Cmu k^2/epsilon on `calculated` nut patches (keNut), and it excluded the SST
# and realizableKE but not Spalart-Allmaras. An SA case has no epsilon boundary, so the evaluation asked an
# empty boundary for its values and the zero-block launch stopped the run: `brae cuda: bcValue: invalid
# configuration argument`, found on the boxy5 UAV, whose inlet_front nut is `calculated`. airFoil2D's far-field
# nut is `freestream`, so no SA gate reached it. SA's nut is nuTilda*fv1 on every face, which V2 assigns.
#
# THE ORACLE is serial OpenFOAM v2412 on validation/airFoil2D with its inlet nut rewritten to `calculated`, five
# iterations, against the shipped fixture (inlet nut `freestream`): OpenFOAM's two agree to round-off, so V2's
# two must too, and V2 must run the first at all. Fail-proofed against the binary before the fix, which stops
# with `bcValue: invalid argument`.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/airFoil2D"
OFRC=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFRC" ] || { echo "SKIP: no OpenFOAM v2412"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set +u
set --
source "$OFRC" > /dev/null 2>&1
set -u
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-66s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1
    return 0
}
stage()   # stage <dir> <calculated: 1|0>
{
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    sed -i -e 's/^endTime .*/endTime         5;/' -e 's/^writeInterval .*/writeInterval   5;/' \
           -e 's/^writePrecision .*/writePrecision  15;/' "$1/system/controlDict"
    if [ "$2" = 1 ]
    then
        python3 - "$1/0/nut" <<'PY'
import re
import sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(\n    inlet\n    \{)[^}]*\}', r'\1\n        type            calculated;\n        value           $internalField;\n    }', s)
assert n == 1
open(p, 'w').write(s)
PY
        grep -A3 "^    inlet" "$1/0/nut" | grep -q calculated || { echo "FAIL: calculated inlet nut not staged"; exit 1; }
    fi
}
l2()
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
print(' '.join('%s=%.3e' % (f, np.linalg.norm(read(a + '/' + f) - read(b + '/' + f)) / np.linalg.norm(read(b + '/' + f)))
               for f in ('U', 'p', 'nuTilda')))
PY
}
for a in of v2 ofs v2s
do
    case $a in
        of|v2) stage "$W/$a" 1 ;;
        *)     stage "$W/$a" 0 ;;
    esac
done
( cd "$W/of" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM"; exit 1; }
( cd "$W/ofs" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM (shipped)"; exit 1; }
( cd "$W/v2" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case . > log 2> err ); rc=$?
[ $rc = 0 ] && [ -f "$W/v2/5/U" ] && say "V2 runs SA with a calculated nut patch" ok \
    || { say "V2 runs SA with a calculated nut patch" FAIL; grep -m2 ERROR "$W/v2/err"; echo FAIL; exit 1; }
( cd "$W/v2s" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case . > log 2> err ) || { say "CONTROL V2 runs the shipped fixture" FAIL; exit 1; }
A=$(l2 "$W/v2/5" "$W/v2s/5")
O=$(l2 "$W/of/5" "$W/ofs/5")
T=$(l2 "$W/v2/5" "$W/of/5")
echo "  V2, calculated against shipped inlet nut, at 5:       $A"
echo "  OpenFOAM, calculated against shipped inlet nut, at 5: $O"
echo "  (V2 against OpenFOAM at 5, a trajectory read, not gated: $T)"
# MEASURED on the GB10 (2026-10-16): V2 calculated against shipped 0.0 in U, p, nuTilda; OpenFOAM's own
# U 1.1e-11 p 3.2e-10 nuTilda 4.7e-11 -- `calculated` names who fills the patch in, and for SA that is
# nuTilda*fv1 either way. The bound is OpenFOAM's spread, rounded up.
python3 - "$A" "$O" <<'PY'
import sys
a = {k: float(v) for k, v in (t.split('=') for t in sys.argv[1].split())}
o = {k: float(v) for k, v in (t.split('=') for t in sys.argv[2].split())}
rc = 0
for f in ('U', 'p', 'nuTilda'):
    ok = a[f] < 1e-09
    print('  %-66s %s' % ('V2 %s calculated vs shipped %.3e < 1e-09' % (f, a[f]), 'ok' if ok else 'FAIL'))
    rc |= 0 if ok else 1
    ok = o[f] < 1e-09
    print('  %-66s %s' % ('PREMISE OpenFOAM %s calculated vs shipped %.3e < 1e-09' % (f, o[f]), 'ok' if ok else 'FAIL'))
    rc |= 0 if ok else 1
sys.exit(rc)
PY
[ $? -eq 0 ] || fail=1
[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
