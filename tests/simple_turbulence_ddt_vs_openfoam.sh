#!/usr/bin/env bash
# simpleFoam's turbulence equations under `ddtSchemes default Euler`, against OpenFOAM v2412.
#
# WHAT IT HOLDS. simpleFoam's UEqn has no ddt, but OpenFOAM's turbulence models construct fvm::ddt(alpha, rho, k) and
# fvm::ddt(alpha, rho, epsilon) whatever the solver (kEpsilon.C:254, 275): under Euler each is (psi - psi.oldTime())/deltaT,
# deltaT the solver's own time step, and the old level the field as the iteration found it. brae's steady step dropped
# both terms and said the entry was inert.
#
# THE CASE. incompressible/simpleFoam/squareBend (Euler, kEpsilon, turbulentIntensityKineticEnergyInlet and
# turbulentMixingLengthDissipationRateInlet, inletOutlet outlet, deltaT 1), meshed as Allrun.pre does, three iterations from
# rest written every iteration at writePrecision 17, every solve pinned (tolerance 1e-12, relTol 0), with U's convection
# staged `Gauss linearUpwind grad(U)` in both codes. The tutorial's `Gauss limitedLinearV 1` cannot be judged from rest: the
# cells start uniform, so at iteration 2 every face's limiter sits in NVDVTVDV's 1000-branch, whose sign round-off decides, and
# the codes part there by 1.1e-03 in U (5.1e-10 with linearUpwind, Arrhenius 5872927). From OpenFOAM's own iteration 20, as
# shipped, it holds 3.0e-09 over three iterations (5873191).
#
# ARMS, every entry of p U k epsilon nut on its own scale (tests/pimple_entry_compare.py) within BOUND at 1-3:
#   S1  as shipped
#   S2  the inlet's k and epsilon fixedValue at iteration 1's values (0.375, Cmu^0.75/0.005): the cells next to the inlet
#       start away from it, where the term is largest beside the rest of the diagonal
# PREMISE: OpenFOAM with steadyState differs from its Euler run, so the arms can see the term.
# CONTROL: BRAE_CONTROL_STEADY_TURBULENCE_DDT_IGNORED=1 (the term dropped) red on S1 and S2.
# REFUSALS, each stopping at start-up and naming the scheme: `default backward` on simpleFoam; Euler with a scalarTransport
# function object on simpleFoam (OpenFOAM's tracer takes fvm::ddt(s), scalarTransport.C:341, which the steady step drops);
# on rhoSimpleFoam's default device path (squareBend's own rhoSimpleFoam tutorial restaged) Euler with a RAS model, and
# Euler with a scalarTransport on a laminar case (fvm::ddt(rho, s), :319).
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28, jobs 5872345, 5873140): worst entry at 1-3, on the
# steady step at OpenFOAM's update points (simple_update_points_vs_openfoam)
#   the term dropped (the control): S1 6.7e-02 (nut internal), S2 7.4e-02 (p internal)
#   carried: S1 5.1e-10 (nut internal), S2 2.5e-10 (p internal); phi 1.8e-11, 6.1e-11. So BOUND 1e-8, twenty times the worst.
#   Premise 6.7e-02 (nut internal). Found with OpenFOAM's own kEpsilon with writes added (tools/dumpKEpsilon): its assembled
#   epsilon equation's diagonal 1.0*V above brae's in every cell, the Euler term at deltaT 1 (5871717).
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_TD_BOUND:-1e-8}
PREMISE=1e-4
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUTS="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}"
TUT="$TUTS/incompressible/simpleFoam/squareBend"
RTUT="$TUTS/compressible/rhoSimpleFoam/squareBend"
[ -d "$TUT/system" ] || { echo "SKIP: no simpleFoam/squareBend tutorial at $TUT"; exit 77; }
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }
W=${BRAE_TD_KEEP:-$(mktemp -d)}
[ -n "${BRAE_TD_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-96s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
echo "simple_turbulence_ddt_vs_openfoam: squareBend from rest, Euler in the turbulence equations"

# the mesh, as Allrun.pre makes it
B=$W/base
rm -rf "$B"
mkdir -p "$B"
cp -r "$TUT/0.orig" "$B/0"
cp -r "$TUT/constant" "$TUT/system" "$B/"
( cd "$B" && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 ) || { echo "FAIL: meshing"; tail -5 "$B"/log.*; exit 1; }

# stage <dir> <arm>
stage()
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$B/0" "$B/constant" "$B/system" "$1/"
    python3 - "$1" "$2" << 'PY'
import re
import sys
d, arm = sys.argv[1:3]
p = d + '/system/controlDict'
c = open(p).read()
i = c.find('\nfunctions')
if i >= 0:
    j = c.index('{', i)
    depth, k = 0, j
    while True:
        depth += {'{': 1, '}': -1}.get(c[k], 0)
        k += 1
        if depth == 0:
            break
    c = c[:i] + '\n' + c[k:]
f = open(d + '/system/fvSchemes').read()
f, n = re.subn(r'div\(phi,U\)\s+Gauss limitedLinearV 1;', 'div(phi,U)      Gauss linearUpwind grad(U);', f)
assert n == 1, 'div(phi,U)'
open(d + '/system/fvSchemes', 'w').write(f)
for key, v in (('endTime', '3'), ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
               ('writePrecision', '17'), ('writeCompression', 'off'), ('timePrecision', '6')):
    c, n = re.subn(r'^%s\s+[^;]*;' % key, '%-16s%s;' % (key, v), c, flags=re.M)
    if n == 0:
        c += '\n%-16s%s;\n' % (key, v)
open(p, 'w').write(c)
s = open(d + '/system/fvSolution').read()
i = s.index('solvers')
j = s.index('SIMPLE')
blk = re.sub(r'(\btolerance\s+)[^;]+;', r'\g<1>1e-12;', s[i:j])
blk = re.sub(r'(\brelTol\s+)[^;]+;', r'\g<1>0;', blk)
blk = re.sub(r'(\bsolver\s+[^;]+;)', r'\g<1>\n        maxIter         5000;', blk)
s = s[:i] + blk + s[j:]
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl\n    {\n    }', s)
open(d + '/system/fvSolution', 'w').write(s)
def patch(field, name, body):
    q = '%s/0/%s' % (d, field)
    t = open(q).read()
    k = t.index('boundaryField')
    m = re.compile(r'\n(\s*)%s\s*\{[^{}]*\}' % re.escape(name)).search(t, k)
    assert m, (field, name)
    open(q, 'w').write(t[:m.start()] + '\n%s%s\n%s{\n%s    %s\n%s}' % (m.group(1), name, m.group(1), m.group(1), body, m.group(1)) + t[m.end():])
if arm in ('S2', 'S2steady'):
    patch('k', 'inlet', 'type fixedValue; value uniform 0.375;')
    patch('epsilon', 'inlet', 'type fixedValue; value uniform %.17g;' % (0.09 ** 0.75 / 0.005))
if arm in ('S1steady', 'S2steady'):
    f = open(d + '/system/fvSchemes').read()
    f, n = re.subn(r'(ddtSchemes\s*\{\s*default\s+)Euler;', r'\g<1>steadyState;', f)
    assert n == 1
    open(d + '/system/fvSchemes', 'w').write(f)
if arm == 'Rbackward':
    f = open(d + '/system/fvSchemes').read()
    f, n = re.subn(r'(ddtSchemes\s*\{\s*default\s+)Euler;', r'\g<1>backward;', f)
    assert n == 1
    open(d + '/system/fvSchemes', 'w').write(f)
PY
}
# addtracer <case>: a scalarTransport function object brae accepts (a constant D, its div scheme), and its field
addtracer()
{
    python3 - "$1" << 'PY'
import re
import sys
d = sys.argv[1]
c = open(d + '/system/controlDict').read()
i = c.find('\nfunctions')
if i >= 0:
    j = c.index('{', i)
    depth, k = 0, j
    while True:
        depth += {'{': 1, '}': -1}.get(c[k], 0)
        k += 1
        if depth == 0:
            break
    c = c[:i] + '\n' + c[k:]
c += ('\nfunctions\n{\n    tracer\n    {\n        type            scalarTransport;\n'
      '        libs            (solverFunctionObjects);\n        field           tracer;\n'
      '        D               1e-3;\n    }\n}\n')
open(d + '/system/controlDict', 'w').write(c)
f = open(d + '/system/fvSchemes').read()
f, n = re.subn(r'(divSchemes\s*\{\s*default\s+none;)', r'\g<1>\n    div(phi,tracer) bounded Gauss upwind;', f)
assert n == 1
open(d + '/system/fvSchemes', 'w').write(f)
open(d + '/0/tracer', 'w').write(
    'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n    object      tracer;\n}\n'
    'dimensions      [0 0 0 0 0 0 0];\ninternalField   uniform 0;\nboundaryField\n{\n'
    '    inlet\n    {\n        type            fixedValue;\n        value           uniform 1;\n    }\n'
    '    outlet\n    {\n        type            zeroGradient;\n    }\n'
    '    walls\n    {\n        type            zeroGradient;\n    }\n}\n')
PY
}
ofrun() { ( cd "$1" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM in $1"; tail -5 "$1/log"; exit 1; }; }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ) || { echo "FAIL: brae in $d"; tail -5 "$d/log"; exit 1; }; }
TL=1,2,3
FIELDS="p U k epsilon nut"
worst() { python3 "$CMP" "$@" | awk '/^MAX/{print $2, $3, $4, $5}'; }
# phi's worst entry, each entry on its own scale unless that scale is round-off of the field's (below 1e-12 of its largest
# entry at that time), which is held on the field's scale: a wall's flux is 0 or round-off
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

for a in S1 S2; do
    stage "$W/$a-of" "$a"
    ofrun "$W/$a-of"
    stage "$W/$a-br" "$a"
    brrun "$W/$a-br"
    w=$(worst "$W/$a-of" "$W/$a-br" "$TL" $FIELDS)
    wp=$(worstphi "$W/$a-of" "$W/$a-br" "$TL")
    if [ "$BOUND" = report ]; then
        say "$a: worst entry at 1-3 $w; phi $wp" INFO
        continue
    fi
    le "${w%% *}" "$BOUND" && say "$a: every entry of $FIELDS at 1-3 within $BOUND (worst $w)" PASS \
                           || say "$a: every entry of $FIELDS at 1-3 within $BOUND (worst $w)" FAIL
    le "${wp%% *}" "$BOUND" && say "$a: phi at 1-3 within $BOUND (worst ${wp%% *})" PASS \
                            || say "$a: phi at 1-3 within $BOUND (worst ${wp%% *})" FAIL
done

# PREMISE: the term moves OpenFOAM
stage "$W/S1steady-of" S1steady
ofrun "$W/S1steady-of"
w=$(worst "$W/S1-of" "$W/S1steady-of" "$TL" $FIELDS)
gt "${w%% *}" "$PREMISE" && say "premise: OpenFOAM under steadyState differs from its Euler run ($w)" PASS \
                         || say "premise: OpenFOAM under steadyState differs from its Euler run ($w)" FAIL

# CONTROL
CB=$BOUND
[ "$CB" = report ] && CB=1e-8
for a in S1 S2; do
    stage "$W/$a-ctl" "$a"
    brrun "$W/$a-ctl" BRAE_CONTROL_STEADY_TURBULENCE_DDT_IGNORED=1
    w=$(worst "$W/$a-of" "$W/$a-ctl" "$TL" $FIELDS)
    gt "${w%% *}" "$CB" && say "control BRAE_CONTROL_STEADY_TURBULENCE_DDT_IGNORED=1: $a goes red ($w)" PASS \
                        || say "control BRAE_CONTROL_STEADY_TURBULENCE_DDT_IGNORED=1: $a goes red ($w)" FAIL
done

# REFUSALS, by name at start-up
stage "$W/Rbackward" Rbackward
out=$(cd "$W/Rbackward" && "$BIN" -case . 2>&1)
rc=$?
[ $rc != 0 ] && echo "$out" | grep -q "ddtSchemes default 'backward' on the steady solver simpleFoam" \
    && say "refusal: ddtSchemes default backward on simpleFoam stops at start-up, naming it" PASS \
    || { echo "$out" | tail -3; say "refusal: ddtSchemes default backward on simpleFoam stops at start-up, naming it" FAIL; }
stage "$W/Rtracer" S1
addtracer "$W/Rtracer"
out=$(cd "$W/Rtracer" && "$BIN" -case . 2>&1)
rc=$?
[ $rc != 0 ] && echo "$out" | grep -q "ddtSchemes default Euler with a scalarTransport function object on simpleFoam" \
    && say "refusal: Euler with a scalarTransport function object on simpleFoam stops at start-up" PASS \
    || { echo "$out" | tail -3; say "refusal: Euler with a scalarTransport function object on simpleFoam stops at start-up" FAIL; }
if [ -d "$RTUT/system" ]; then
    R=$W/rho
    rm -rf "$R"
    mkdir -p "$R"
    cp -r "$RTUT/0.orig" "$R/0"
    cp -r "$RTUT/constant" "$RTUT/system" "$R/"
    ( cd "$R" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: rhoSimpleFoam squareBend blockMesh"; exit 1; }
    python3 - "$R/system/fvSchemes" << 'PY'
import re, sys
p = sys.argv[1]
f = open(p).read()
f, n = re.subn(r'(ddtSchemes\s*\{\s*default\s+)\w+;', r'\g<1>Euler;', f)
assert n == 1
open(p, 'w').write(f)
PY
    out=$(cd "$R" && "$BIN" -case . 2>&1)
    rc=$?
    [ $rc != 0 ] && echo "$out" | grep -q "ddtSchemes default 'Euler' with a RAS model on rhoSimpleFoam's default device path" \
        && say "refusal: Euler with a RAS model on rhoSimpleFoam's default device path stops at start-up" PASS \
        || { echo "$out" | tail -3; say "refusal: Euler with a RAS model on rhoSimpleFoam's default device path stops at start-up" FAIL; }
    # laminar, so the RAS refusal cannot be what stops it
    R2=$W/rhotracer
    rm -rf "$R2"
    cp -r "$R" "$R2"
    sed -i 's/^simulationType .*/simulationType  laminar;/' "$R2/constant/turbulenceProperties"
    grep -q '^simulationType  laminar;' "$R2/constant/turbulenceProperties" || { echo "FAIL: staging laminar"; exit 1; }
    addtracer "$R2"
    out=$(cd "$R2" && "$BIN" -case . 2>&1)
    rc=$?
    [ $rc != 0 ] && echo "$out" | grep -q "ddtSchemes default 'Euler' with a scalarTransport function object on rhoSimpleFoam's default device path" \
        && say "refusal: Euler with a scalarTransport (laminar) on rhoSimpleFoam's default device path stops at start-up" PASS \
        || { echo "$out" | tail -3; say "refusal: Euler with a scalarTransport (laminar) on rhoSimpleFoam's default device path stops at start-up" FAIL; }
fi

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
