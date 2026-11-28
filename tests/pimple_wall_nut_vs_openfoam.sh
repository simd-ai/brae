#!/usr/bin/env bash
# pimpleFoam's nut on a WALL patch that carries no nut wall function, against OpenFOAM v2412.
#
# WHAT IT HOLDS. OpenFOAM's nut on any patch, a wall's included, is what that patch's own nut condition evaluates in correctNut
# (nut_.correctBoundaryConditions()): a calculated wall takes the model's expression of the patch values (kEpsilon: Cmu
# k^2/epsilon), a zeroGradient or slip wall the cell's nut, a fixedValue wall its value. A wall function is a nut condition like
# any other (nutkWallFunction, nutUSpaldingWallFunction, ...). brae gave every `wall`-typed patch a wall function's nut, so a
# slip ground or a low-Re wall ran with the wrong nuEff on its faces.
#
# THE CASE. incompressible/pimpleFoam/RAS/pitzDaily at a fixed deltaT 1e-4, three steps from rest, written every step at
# writePrecision 17, every solve tight (tolerance 1e-12, relTol 0). lowerWall keeps the tutorial's wall functions throughout.
#
# ARMS, every entry of p U and the turbulence fields, and phi, on its own scale (tests/pimple_entry_compare.py) within BOUND:
#   G1  kEpsilon, upperWall a slip ground: U slip, k and epsilon zeroGradient, nut calculated
#   G2  G1 with upperWall's nut zeroGradient
#   G3  SpalartAllmaras, upperWall a low-Re wall (U noSlip, nuTilda and nut fixedValue 0), lowerWall nutUSpaldingWallFunction
#       (wallDist meshWave added to fvSchemes: SpalartAllmaras reads y, the tutorial's kEpsilon does not)
# phi is held entry by entry on its own scale, but for an entry whose scale is round-off of the field's (a slip wall's flux),
# which is held on the field's (worstphi).
# PREMISE: OpenFOAM's G1 differs from OpenFOAM with upperWall's nut a nutkWallFunction (G1W), so the arm can see the condition.
# CONTROL BRAE_CONTROL_WALL_NUT_BY_PATCH_TYPE=1 (every wall's nut from the wall function, as before): G1 and G3 red.
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28, jobs 5867500, 5867561): worst entry over steps 1-3
#   before (develop 43ce161c): G1 1.0 (nut upperWall), G2 4.6e-01 (nut outlet), phi 4.2e-02; G3 refused at start-up ("the SA
#   path writes its wall function on every wall face")
#   after: G1 2.4e-10, G2 2.4e-10, G3 3.5e-10 (p internal); phi 1.4e-11, 1.4e-11, 6.8e-12. So BOUND 1e-8, thirty times the worst,
#   as the other pitzDaily gates. Premise 1.0 (nut upperWall); controls 1.0 (G1), inf (G3, nut upperWall written nonzero where
#   OpenFOAM's is 0).
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_WN_BOUND:-1e-8}
PREMISE=1e-4
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUT="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}/incompressible/pimpleFoam/RAS/pitzDaily"
[ -d "$TUT/system" ] || { echo "SKIP: no pimpleFoam/RAS/pitzDaily tutorial at $TUT"; exit 77; }
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
W=${BRAE_WN_KEEP:-$(mktemp -d)}
[ -n "${BRAE_WN_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-96s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
# stage <dir> <arm>
stage()
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"
    python3 - "$1" "$2" << 'PY'
import re
import sys
d, arm = sys.argv[1:3]
c = open(d + '/system/controlDict').read()
for k, v in (('adjustTimeStep', 'no'), ('deltaT', '0.0001'), ('endTime', '0.0003'), ('writeControl', 'timeStep'),
             ('writeInterval', '1'), ('writePrecision', '17'), ('timePrecision', '18')):
    c, n = re.subn(r'^%s\s+[^;]*;' % k, '%-16s%s;' % (k, v), c, flags=re.M)
    assert n == 1, k
open(d + '/system/controlDict', 'w').write(c)
s = open(d + '/system/fvSolution').read()
i = s.index('solvers')
j = s.index('PIMPLE')
blk = re.sub(r'(\btolerance\s+)[^;]+;', r'\g<1>1e-12;', s[i:j])
blk = re.sub(r'(\brelTol\s+)[^;]+;', r'\g<1>0;', blk)
blk = re.sub(r'(\bsolver\s+[^;]+;)', r'\g<1>\n        maxIter         5000;', blk)
open(d + '/system/fvSolution', 'w').write(s[:i] + blk + s[j:])
def patch(field, name, body):
    p = '%s/0/%s' % (d, field)
    s = open(p).read()
    s2, n = re.subn(r'(\n\s*%s\s*\{)[^}]*\}' % name, r'\1\n        %s\n    }' % body.replace('\\', '\\\\'), s, count=1)
    assert n == 1, (field, name)
    open(p, 'w').write(s2)
if arm in ('G1', 'G2', 'G1W'):
    patch('U', 'upperWall', 'type            slip;')
    patch('k', 'upperWall', 'type            zeroGradient;')
    patch('epsilon', 'upperWall', 'type            zeroGradient;')
    patch('nut', 'upperWall', {'G1': 'type            calculated;\n        value           uniform 0;',
                               'G2': 'type            zeroGradient;',
                               'G1W': 'type            nutkWallFunction;\n        value           uniform 0;'}[arm])
elif arm == 'G3':
    t = open(d + '/constant/turbulenceProperties').read()
    t2, n = re.subn(r'RASModel\s+\w+;', 'RASModel        SpalartAllmaras;', t)
    assert n == 1
    open(d + '/constant/turbulenceProperties', 'w').write(t2)
    nt = open(d + '/0/nuTilda').read()
    nt, n = re.subn(r'internalField\s+uniform\s+[^;]+;', 'internalField   uniform 1e-4;', nt)
    assert n == 1
    open(d + '/0/nuTilda', 'w').write(nt)
    patch('nuTilda', 'inlet', 'type            fixedValue;\n        value           uniform 1e-4;')
    patch('nuTilda', 'outlet', 'type            zeroGradient;')
    patch('nuTilda', 'upperWall', 'type            fixedValue;\n        value           uniform 0;')
    patch('nuTilda', 'lowerWall', 'type            fixedValue;\n        value           uniform 0;')
    patch('nut', 'upperWall', 'type            fixedValue;\n        value           uniform 0;')
    patch('nut', 'lowerWall', 'type            nutUSpaldingWallFunction;\n        value           uniform 0;')
    f = open(d + '/system/fvSchemes').read()
    if 'wallDist' not in f:
        f += '\nwallDist\n{\n    method          meshWave;\n}\n'   # SpalartAllmaras needs y; kEpsilon, as shipped, does not
    if 'div(phi,nuTilda)' not in f:
        f = re.sub(r'(divSchemes\s*\{)', r'\1\n    div(phi,nuTilda) bounded Gauss limitedLinear 1;', f, count=1)
    open(d + '/system/fvSchemes', 'w').write(f)
    s = open(d + '/system/fvSolution').read()
    s = re.sub(r'"\(U\|k\|epsilon\)(Final)?"', lambda m: '"(U|k|epsilon|nuTilda)%s"' % (m.group(1) or ''), s)
    open(d + '/system/fvSolution', 'w').write(s)
PY
}
ofrun() { ( cd "$1" && pimpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM in $1"; tail -5 "$1/log"; exit 1; }; }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ) || { echo "FAIL: brae in $d"; tail -5 "$d/log"; exit 1; }; }
tlist() { (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -vx 0 | sort -g | paste -sd,); }
worst() { python3 "$CMP" "$@" | awk '/^MAX/{print $2, $3, $4, $5}'; }
# phi's worst entry, each entry on its own scale unless that scale is round-off of the field's (below 1e-12 of its largest
# entry at that time), which is held on the field's scale: a slip wall's flux is U_b & Sf with U_b's normal part removed, 6.8e-22
# in OpenFOAM against internal faces of 2.3e-05, and brae's differs there by 7.4e-22 (tests/pimple_entry_compare.py reads that 1.1)
worstphi()
{
    python3 "$CMP" "$@" phi | python3 -c '
import sys
rows = []
for line in sys.stdin:
    f = line.split()
    if len(f) >= 10 and f[0] != "MAX":
        rows.append((f[0], f[2], float(f[5]), float(f[7])))
best = (0.0, "", "")
for t in set(r[0] for r in rows):
    mine = [r for r in rows if r[0] == t]
    top = max(r[2] for r in mine)
    for _, entry, s, d in mine:
        scale = s if s >= 1e-12 * top else top
        rel = d / scale if scale > 0 else (0.0 if d == 0 else float("inf"))
        if rel >= best[0]:
            best = (rel, t, entry)
print("%.3e %s phi %s" % best)'
}
le() { python3 -c "import sys; sys.exit(0 if float('$1') <= float('$2') else 1)"; }
gt() { python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }
fields() { [ "$1" = G3 ] && echo "p U nuTilda nut" || echo "p U k epsilon nut"; }
echo "pimple_wall_nut_vs_openfoam: pitzDaily, nut on a wall patch without a wall function"

for a in G1 G2 G3; do
    stage "$W/$a-of" "$a"
    ofrun "$W/$a-of"
    stage "$W/$a-br" "$a"
    brrun "$W/$a-br"
    T=$(tlist "$W/$a-of")
    w=$(worst "$W/$a-of" "$W/$a-br" "$T" $(fields "$a"))
    wp=$(worstphi "$W/$a-of" "$W/$a-br" "$T")
    if [ "$BOUND" = report ]; then
        say "$a: worst entry at steps 1-3 $w; phi ${wp%% *}" INFO
        continue
    fi
    le "${w%% *}" "$BOUND" && say "$a: every entry of $(fields "$a") at steps 1-3 within $BOUND (worst $w)" PASS \
                           || say "$a: every entry of $(fields "$a") at steps 1-3 within $BOUND (worst $w)" FAIL
    le "${wp%% *}" "$BOUND" && say "$a: phi at steps 1-3 within $BOUND (worst ${wp%% *})" PASS \
                            || say "$a: phi at steps 1-3 within $BOUND (worst ${wp%% *})" FAIL
done

# PREMISE: the wall's nut condition moves OpenFOAM
stage "$W/G1W-of" G1W
ofrun "$W/G1W-of"
w=$(worst "$W/G1-of" "$W/G1W-of" "$(tlist "$W/G1-of")" $(fields G1))
gt "${w%% *}" "$PREMISE" && say "premise: OpenFOAM's upperWall nut calculated differs from nutkWallFunction ($w)" PASS \
                         || say "premise: OpenFOAM's upperWall nut calculated differs from nutkWallFunction ($w)" FAIL

# CONTROL
CB=$BOUND
[ "$CB" = report ] && CB=1e-8
for a in G1 G3; do
    stage "$W/$a-ctl" "$a"
    brrun "$W/$a-ctl" BRAE_CONTROL_WALL_NUT_BY_PATCH_TYPE=1
    w=$(worst "$W/$a-of" "$W/$a-ctl" "$(tlist "$W/$a-of")" $(fields "$a"))
    gt "${w%% *}" "$CB" && say "control BRAE_CONTROL_WALL_NUT_BY_PATCH_TYPE=1: $a goes red ($w)" PASS \
                        || say "control BRAE_CONTROL_WALL_NUT_BY_PATCH_TYPE=1: $a goes red ($w)" FAIL
done

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
