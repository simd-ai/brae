#!/usr/bin/env bash
# nut on the patches that are not walls, under pimpleFoam on incompressible/pimpleFoam/RAS/TJunction, step by step against
# OpenFOAM v2412.
#
# WHAT IT HOLDS. Every model ends its correct() in correctNut: `nut_ = <expression>; nut_.correctBoundaryConditions();`
# (kEpsilon.C:45-46, kOmegaSSTBase.C:123-124, SpalartAllmarasBase.C:166-167). The assignment hands each patch the
# expression's patch value, and correctBoundaryConditions then evaluates each patch by its own type:
#   calculated      keeps the expression's patch value (Cmu k_b^2/epsilon_b, a1 k_b/max(a1 omega_b, b1 F2_b sqrt(S2_b)),
#                   nuTilda_b fv1(chi_b))
#   zeroGradient    the cell's nut
#   inletOutlet     updateCoeffs sets valueFraction = neg(phi) (inletOutletFvPatchField.C:114-127): an inflow face takes
#                   inletValue, an outflow face the cell's nut
#   freestream      the same, its freestreamValue as inletValue (freestreamFvPatchField.C:171-185)
# The momentum assembly's nuEff = nut + nu takes these patch values, in the laplacian and in div(nuEff dev2(T(grad U))).
#
# THE CASE. The tutorial staged as tests/pimple_tjunction_boundary_vs_openfoam.sh stages it -- fixed deltaT 0.001 to 0.003,
# every step written at writePrecision 17, every linear solve tight (tolerance 1e-12, relTol 0, maxIter 5000) -- with nut's
# non-wall patches given each of the types above. outlet1's faces switch to inflow during step 1 (the SpalartAllmaras gate's
# premise), so an inletOutlet or freestream there has faces of both kinds. The inletValues are apart from the cells' nut.
#   KE  kEpsilon, walls nutkWallFunction:          inlet calculated, outlet1 inletOutlet (inletValue 3e-4), outlet2 zeroGradient
#   KS  kEpsilon, walls nutUSpaldingWallFunction:  the same
#   SS  kOmegaSST, walls nutUSpaldingWallFunction: inlet, outlet1, outlet2 calculated
#   SK  kOmegaSST, walls nutkWallFunction:         inlet calculated, outlet1 inletOutlet (inletValue 3e-4), outlet2 zeroGradient
#   SA  SpalartAllmaras, walls nutUSpaldingWallFunction: inlet calculated, outlet1 freestream (freestreamValue 3e-5),
#       outlet2 zeroGradient (nuTilda as tests/pimple_tjunction_sa_vs_openfoam.sh stages it)
#   SL  SA with walls nutLowReWallFunction (bump2D's SpalartAllmaras walls), the same non-wall patches
#
# CHECKS, each entry (the internal field and every patch's written value) on its own scale (tests/pimple_entry_compare.py):
# every field at steps 1-3 within BOUND; phi on its own line. A zeroGradient patch writes no value; its nut shows through U.
# Control, red on the entry each arm guards: BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1, the non-wall faces as the wall-function
# branch left them -- the file's value under a velocity-based wall function, zero under SpalartAllmaras' nutLowReWallFunction,
# the cell's value on an inletOutlet face under nutkWallFunction.
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-27): worst entry over steps 1-3
#   before (develop's tree 82789eed, job 5845340): KE 9.0e-01 (nut outlet1 at step 1: OpenFOAM's inflow faces take the
#     inletValue 3e-4, brae took the cell), KS 1.0 (the calculated inlet held at the file's 0), SS 1.0 (nut outlet2), SK
#     9.2e-01, SA 2.3e-04 (U outlet2, through the zeroGradient nut); SL the same (its control, job 5845485)
#   after (job 5845485): 5.4e-08 to 1.7e-07 (SS: nut on the walls at step 3), phi 1.0e-07 at most. So BOUND 1e-6, the other
#   TJunction gates' bound, six times the worst. Controls: 9.0e-01, 1.0, 1.0, 9.2e-01, 2.3e-04, 2.3e-04.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="${BRAE_CMP:-$ROOT/tests/pimple_entry_compare.py}"
BOUND=${BRAE_NUT_BOUND:-1e-6}
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUT="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}/incompressible/pimpleFoam/RAS/TJunction"
[ -d "$TUT/system" ] || { echo "SKIP: no pimpleFoam/RAS/TJunction tutorial at $TUT"; exit 77; }
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
W=${BRAE_NUT_KEEP:-$(mktemp -d)}
[ -n "${BRAE_NUT_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-86s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
TIMES=0.001,0.002,0.003

setkey()
{
    sed -i -E "s/^$2[[:space:]][^;]*;/$2 $3;/" "$1/system/controlDict"
    grep -q -E "^$2 $3;" "$1/system/controlDict" || { echo "FAIL: could not set $2 $3"; exit 1; }
}
tight()
{
    python3 - "$1/system/fvSolution" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
i = s.index('solvers')
j = s.index('PIMPLE')
blk = re.sub(r'(\btolerance\s+)[^;]+;', r'\g<1>1e-12;', s[i:j])
blk = re.sub(r'(\brelTol\s+)[^;]+;', r'\g<1>0;', blk)
blk = re.sub(r'(\bsolver\s+[^;]+;)', r'\g<1>\n        maxIter         5000;', blk)
open(p, 'w').write(s[:i] + blk + s[j:])
EOF
}
# arm <case> <model> <wall nut type> <inlet> <outlet1> <outlet2>: the model and its fields, nut's patches as named
# (calculated | zeroGradient | inletOutlet:<v> | freestream:<v>)
arm()
{
    python3 - "$@" <<'EOF'
import os, re, sys
c, model, wall = sys.argv[1], sys.argv[2], sys.argv[3]
kinds = dict(zip(('inlet', 'outlet1', 'outlet2'), sys.argv[4:7]))
def patch(name, body):
    return '    %s\n    {\n%s    }\n' % (name, ''.join('        %-15s %s;\n' % kv for kv in body))
def field(name, cls, dims, internal, patches):
    open(c + '/0/' + name, 'w').write(
        'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n    object      %s;\n}\n\n'
        'dimensions      %s;\n\ninternalField   %s;\n\nboundaryField\n{\n%s}\n' % (cls, name, dims, internal, ''.join(patches)))
nut = []
for p in ('inlet', 'outlet1', 'outlet2'):
    k = kinds[p]
    if k == 'calculated':
        nut.append(patch(p, [('type', 'calculated'), ('value', 'uniform 0')]))
    elif k == 'zeroGradient':
        nut.append(patch(p, [('type', 'zeroGradient')]))
    elif k.startswith('inletOutlet:'):
        nut.append(patch(p, [('type', 'inletOutlet'), ('inletValue', 'uniform ' + k.split(':')[1]), ('value', 'uniform 0')]))
    elif k.startswith('freestream:'):
        nut.append(patch(p, [('type', 'freestream'), ('freestreamValue', 'uniform ' + k.split(':')[1])]))
    else:
        raise SystemExit('unknown nut kind ' + k)
nut.append(patch('defaultFaces', [('type', wall), ('value', 'uniform 0')]))
field('nut', 'volScalarField', '[0 2 -1 0 0 0 0]', 'uniform 0', nut)
tp = c + '/constant/turbulenceProperties'
s = open(tp).read()
fs = c + '/system/fvSchemes'
sc = open(fs).read()
fv = c + '/system/fvSolution'
so = open(fv).read()
if model == 'kOmegaSST':
    om = [patch('inlet', [('type', 'turbulentMixingLengthFrequencyInlet'), ('mixingLength', '0.01'), ('value', '$internalField')])]
    for o in ('outlet1', 'outlet2'):
        om.append(patch(o, [('type', 'inletOutlet'), ('inletValue', '$internalField'), ('value', '$internalField')]))
    om.append(patch('defaultFaces', [('type', 'omegaWallFunction'), ('value', '$internalField')]))
    field('omega', 'volScalarField', '[0 0 -1 0 0 0 0]', 'uniform 11111', om)
    os.remove(c + '/0/epsilon')
    s, n = re.subn(r'RASModel\s+kEpsilon;', 'RASModel        kOmegaSST;', s)
    assert n == 1
    sc, n = re.subn(r'(\n\s*div\(phi,epsilon\)\s+\$turbulence;)', r'\1\n    div(phi,omega)  $turbulence;', sc)
    assert n == 1
    so, n = re.subn(r'"\(U\|k\|epsilon\|s\)(Final)?"', lambda m: '"(U|k|epsilon|omega|s)%s"' % (m.group(1) or ''), so)
    assert n == 2
elif model == 'SpalartAllmaras':
    nt = [patch('inlet', [('type', 'inletOutlet'), ('inletValue', 'uniform 6e-05'), ('value', '$internalField')])]
    for o in ('outlet1', 'outlet2'):
        nt.append(patch(o, [('type', 'inletOutlet'), ('inletValue', 'uniform 2e-05'), ('value', '$internalField')]))
    nt.append(patch('defaultFaces', [('type', 'fixedValue'), ('value', 'uniform 0')]))
    field('nuTilda', 'volScalarField', '[0 2 -1 0 0 0 0]', 'uniform 4e-05', nt)
    for f in ('k', 'epsilon'):
        os.remove(c + '/0/' + f)
    s, n = re.subn(r'RASModel\s+kEpsilon;', 'RASModel        SpalartAllmaras;', s)
    assert n == 1
    assert re.search(r'div\(phi,nuTilda\)\s+\$turbulence;', sc)
    so, n = re.subn(r'"\(U\|k\|epsilon\|s\)(Final)?"', lambda m: '"(U|k|epsilon|nuTilda|s)%s"' % (m.group(1) or ''), so)
    assert n == 2
else:
    assert model == 'kEpsilon'
if model != 'kEpsilon' and 'wallDist' not in sc:
    sc += '\nwallDist\n{\n    method          meshWave;\n}\n'
open(tp, 'w').write(s)
open(fs, 'w').write(sc)
open(fv, 'w').write(so)
EOF
}
mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
grep -q 'type            nutkWallFunction;' "$W/base/0/nut" \
    && grep -q 'RASModel        kEpsilon;' "$W/base/constant/turbulenceProperties" \
    || { echo "FAIL: the tutorial's nut walls or model moved -- the gate's premise moved"; exit 1; }
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
setkey "$W/base" endTime 0.003
setkey "$W/base" adjustTimeStep no
setkey "$W/base" writeControl timeStep
setkey "$W/base" writeInterval 1
setkey "$W/base" writePrecision 17
tight "$W/base"
stage() { mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; }
declare -A FIELDS
for a in KE KS SS SK SA SL; do stage "$W/$a"; done
arm "$W/KE" kEpsilon nutkWallFunction calculated inletOutlet:3e-04 zeroGradient
arm "$W/KS" kEpsilon nutUSpaldingWallFunction calculated inletOutlet:3e-04 zeroGradient
arm "$W/SS" kOmegaSST nutUSpaldingWallFunction calculated calculated calculated
arm "$W/SK" kOmegaSST nutkWallFunction calculated inletOutlet:3e-04 zeroGradient
arm "$W/SA" SpalartAllmaras nutUSpaldingWallFunction calculated freestream:3e-05 zeroGradient
arm "$W/SL" SpalartAllmaras nutLowReWallFunction calculated freestream:3e-05 zeroGradient
FIELDS[KE]="p U k epsilon nut"
FIELDS[KS]="p U k epsilon nut"
FIELDS[SS]="p U k omega nut"
FIELDS[SK]="p U k omega nut"
FIELDS[SA]="p U nuTilda nut"
FIELDS[SL]="p U nuTilda nut"
ofrun() { ( cd "$1" && nice -n 10 pimpleFoam > log 2>&1 ); }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ); }
judge()   # <arm>
{
    local a=$1 m mp
    python3 "$CMP" "$W/$a-of" "$W/$a-br" "$TIMES" ${FIELDS[$a]} > "$W/$a.cmp"
    sed 's/^/      /' "$W/$a.cmp"
    m=$(awk '/^MAX/{print $2, $3, $4, $5}' "$W/$a.cmp")
    python3 "$CMP" "$W/$a-of" "$W/$a-br" "$TIMES" phi > "$W/$a.phi"
    mp=$(awk '/^MAX/{print $2}' "$W/$a.phi")
    if [ "$BOUND" = report ]; then
        say "$a: worst entry of ${FIELDS[$a]} at steps 1-3: $m; phi $mp (report only)" INFO
        return
    fi
    python3 -c "import sys; sys.exit(0 if float('${m%% *}') <= float('$BOUND') else 1)" \
        && say "$a: every entry of ${FIELDS[$a]} at steps 1-3 within $BOUND (worst $m)" PASS \
        || say "$a: every entry of ${FIELDS[$a]} at steps 1-3 within $BOUND (worst $m)" FAIL
    python3 -c "import sys; sys.exit(0 if float('$mp') <= float('$BOUND') else 1)" \
        && say "$a: phi, internal faces and every patch, at steps 1-3 within $BOUND (worst $mp)" PASS \
        || say "$a: phi, internal faces and every patch, at steps 1-3 within $BOUND (worst $mp)" FAIL
}
for a in ${BRAE_NUT_ARMS:-KE KS SS SK SA SL}; do
    cp -r "$W/$a" "$W/$a-of"
    cp -r "$W/$a" "$W/$a-br"
    ofrun "$W/$a-of" || { echo "FAIL: OpenFOAM $a"; tail -5 "$W/$a-of/log"; exit 1; }
    brrun "$W/$a-br" || { say "$a: brae ran ($(grep -a -m1 -i -E 'refus|error|unsupported' "$W/$a-br/log" | cut -c1-120))" FAIL; continue; }
    judge "$a"
done
# Each control must make ITS entry exceed the bound (`report` uses 1e-6 here).
CB=$BOUND
[ "$CB" = report ] && CB=1e-6
control()   # <env> <arm> <time> <field> <entry>
{
    local env=$1 arm=$2 t=$3 f=$4 entry=$5 name=${1%%=*}
    [ -d "$W/$arm-of" ] || return 0
    rm -rf "$W/$arm-$name"
    cp -r "$W/$arm" "$W/$arm-$name"
    brrun "$W/$arm-$name" "$env" || { say "control $env: brae ran on $arm" FAIL; return; }
    local r
    r=$(python3 "$CMP" "$W/$arm-of" "$W/$arm-$name" "$t" "$f" | awk -v e="$entry" '$3 == e {print $10}')
    python3 -c "import sys; sys.exit(0 if '$r' and float('$r') > float('$CB') else 1)" \
        && say "control $env: $arm $f $entry at $t goes red (rel $r)" PASS \
        || say "control $env: $arm $f $entry at $t goes red (rel ${r:-none})" FAIL
}
control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1 KE 0.001 nut outlet1
control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1 KS 0.001 nut inlet
control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1 SS 0.001 nut outlet2
control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1 SK 0.001 nut outlet1
control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1 SA 0.003 U outlet2
control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1 SL 0.003 U outlet2

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
