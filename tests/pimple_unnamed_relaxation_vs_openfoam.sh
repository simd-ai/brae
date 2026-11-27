#!/usr/bin/env bash
# Relaxation of an equation the case names no factor for, under pimpleFoam on incompressible/pimpleFoam/RAS/TJunction, step by
# step against OpenFOAM v2412.
#
# WHAT IT HOLDS. UEqn.relax() and every turbulence equation's relax() are fvMatrix::relax(), which looks the equation up --
# "UFinal", "kFinal", ... on the last outer corrector (psi.select(isFinalIteration())) -- and calls relax(alpha) only when the
# name, a regex covering it or `default` is found (fvMatrix.C:1250-1263, solution.C:379-410). relax(1) is not the identity: it
# adds cmptMax(cmptMag(internalCoeffs)) to the diagonal, takes max(mag(D), sumOff), removes cmptMin(internalCoeffs) and adds
# (D - D0)*psi to the source (fvMatrix.C:1150, 1211, 1238, 1245). A convective face whose patch does not fix the value has
# internalCoeffs phi_b*(1 - valueFraction), negative on inflow, so relax(1) moves its cell. brae relaxed every unnamed equation
# with 1 and did not read `default`.
#
# THE CASE. The tutorial staged as tests/pimple_tjunction_nut_vs_openfoam.sh stages it -- fixed deltaT 0.001 to 0.003, every
# step written at writePrecision 17, every linear solve tight (tolerance 1e-12, relTol 0, maxIter 5000). outlet1's faces turn
# to inflow during step 1. The tutorial names "U.*", "k.*", "epsilon.*" and "s.*" at 1.
#   RK  kEpsilon, relaxationFactors { equations { "U.*" 1; } }: k and epsilon unnamed; their outlets zeroGradient, so the
#       backflow faces carry a negative coefficient (with the shipped inletOutlet, valueFraction 1 on inflow, nothing moves)
#   RS  RK under kOmegaSST: k and omega unnamed, zeroGradient outlets
#   RA  RK under SpalartAllmaras: nuTilda unnamed, zeroGradient outlets (walls nutUSpaldingWallFunction)
#   RN  RK with the tutorial's "k.*" 1 and "epsilon.*" 1 kept: both codes relax with 1
#   RD  as shipped (inletOutlet outlets) with relaxationFactors { equations { default 0.9; } }: OpenFOAM relaxes U, k and
#       epsilon with 0.9 on every corrector, "UFinal" included. Its convection is linearUpwind (U, and the turbulence fields)
#       in place of the tutorial's limitedLinearV 1 / limitedLinear 1: from U = 0 their limiter ratio is ill-conditioned, and
#       with a factor below 1 OpenFOAM moves 1.9e-04 (U) from itself when only its pressure solver changes (GAMG to PCG),
#       which is brae's distance too; with linearUpwind brae is 1e-10 from OpenFOAM (jobs 5860399, 5860422)
# CHECKS, each entry (the internal field and every patch's written value) on its own scale (tests/pimple_entry_compare.py):
# every field at steps 1-3 within BOUND; phi on its own line. Premise: OpenFOAM's RK and RN differ (relax(1) moves the
# solution), so RK can see the defect.
# Controls, red on the entry each arm guards: BRAE_CONTROL_RELAX_UNNAMED_AS_ONE=1 (RK, RS, RA: an unnamed equation relaxed
# with 1, as before), BRAE_CONTROL_RELAX_DEFAULT_IGNORED=1 (RD: `default` not read, as before).
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28), worst entry over steps 1-3:
#   before (develop's tree b0fb1c63, job 5860433): RK 1.6e-02 (epsilon), RS 1.2e-02 (omega), RA 5.0e-03 (nut on outlet1), RD
#     5.3e-01 (`default` not read, so every equation relaxed with 1); RN 6.4e-08 (both codes relax with 1)
#   after: RK 6.4e-08, RS 6.2e-08, RA 7.2e-08, RN 6.4e-08, RD 3.5e-10; phi 5.1e-08 at most. So BOUND 1e-6, the other TJunction
#     gates' bound, fourteen times the worst. Premise: OpenFOAM's RK and RN 1.6e-02 apart. Controls 1.6e-02, 1.2e-02, 2.0e-03
#     (nuTilda), 1.2e-01. About 20 s.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="${BRAE_CMP:-$ROOT/tests/pimple_entry_compare.py}"
BOUND=${BRAE_RLX_BOUND:-1e-6}
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
W=${BRAE_RLX_KEEP:-$(mktemp -d)}
[ -n "${BRAE_RLX_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
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
# factors <case> <entry>...: relaxationFactors' equations hold exactly these entries ("key value" each)
factors()
{
    python3 - "$@" <<'EOF'
import re, sys
c, entries = sys.argv[1], sys.argv[2:]
f = c + '/system/fvSolution'
s = open(f).read()
m = re.search(r'\nrelaxationFactors\s*\{', s)
a = m.end()
d, e = 1, a
while d:
    d += {'{': 1, '}': -1}.get(s[e], 0)
    e += 1
body = '\n    equations\n    {\n' + ''.join('        %s;\n' % x for x in entries) + '    }\n'
open(f, 'w').write(s[:a] + body + s[e - 1:])
EOF
}
# model <case> <kEpsilon|kOmegaSST|SpalartAllmaras>: the model, its fields (as the nut gate stages them) and schemes
model()
{
    python3 - "$1" "$2" <<'EOF'
import os, re, sys
c, model = sys.argv[1:3]
def patch(name, body):
    return '    %s\n    {\n%s    }\n' % (name, ''.join('        %-15s %s;\n' % kv for kv in body))
def field(name, dims, internal, patches):
    open(c + '/0/' + name, 'w').write(
        'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n    object      %s;\n}\n\n'
        'dimensions      %s;\n\ninternalField   %s;\n\nboundaryField\n{\n%s}\n' % (name, dims, internal, ''.join(patches)))
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
    field('omega', '[0 0 -1 0 0 0 0]', 'uniform 11111', om)
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
    field('nuTilda', '[0 2 -1 0 0 0 0]', 'uniform 4e-05', nt)
    nut = [patch(p, [('type', 'calculated'), ('value', 'uniform 0')]) for p in ('inlet', 'outlet1', 'outlet2')]
    nut.append(patch('defaultFaces', [('type', 'nutUSpaldingWallFunction'), ('value', 'uniform 0')]))
    field('nut', '[0 2 -1 0 0 0 0]', 'uniform 0', nut)
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
# zg <case> <field>...: outlet1 and outlet2 zeroGradient for each field
zg()
{
    python3 - "$@" <<'EOF'
import re, sys
c = sys.argv[1]
for fld in sys.argv[2:]:
    f = c + '/0/' + fld
    s = open(f).read()
    i = s.index('boundaryField')
    for patch in ('outlet1', 'outlet2'):
        m = re.search(r'\n\s*%s\s*\n\s*\{' % patch, s[i:])
        a = i + m.end()
        d, e = 1, a
        while d:
            d += {'{': 1, '}': -1}.get(s[e], 0)
            e += 1
        s = s[:a] + '\n        type            zeroGradient;\n    ' + s[e - 1:]
    open(f, 'w').write(s)
EOF
}
mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
grep -q 'RASModel        kEpsilon;' "$W/base/constant/turbulenceProperties" \
    && grep -q -E '"k\.\*" +1;' "$W/base/system/fvSolution" && grep -q -E '"U\.\*" +1;' "$W/base/system/fvSolution" \
    || { echo "FAIL: the tutorial's model or relaxationFactors moved -- the gate's premise moved"; exit 1; }
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
setkey "$W/base" endTime 0.003
setkey "$W/base" adjustTimeStep no
setkey "$W/base" writeControl timeStep
setkey "$W/base" writeInterval 1
setkey "$W/base" writePrecision 17
tight "$W/base"
stage() { rm -rf "$1"; mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; }
declare -A FIELDS
stage "$W/RK"
factors "$W/RK" '"U.*" 1'
zg "$W/RK" k epsilon
stage "$W/RS"
model "$W/RS" kOmegaSST
factors "$W/RS" '"U.*" 1'
zg "$W/RS" k omega
stage "$W/RA"
model "$W/RA" SpalartAllmaras
factors "$W/RA" '"U.*" 1'
zg "$W/RA" nuTilda
stage "$W/RN"
factors "$W/RN" '"U.*" 1' '"k.*" 1' '"epsilon.*" 1'
zg "$W/RN" k epsilon
stage "$W/RD"
factors "$W/RD" 'default 0.9'
python3 - "$W/RD/system/fvSchemes" <<'EOF2'
import re, sys
f = sys.argv[1]
s = open(f).read()
s, a = re.subn(r'div\(phi,U\)\s+Gauss limitedLinearV 1;', 'div(phi,U)      Gauss linearUpwind grad(U);', s)
s, b = re.subn(r'turbulence\s+Gauss limitedLinear 1;', 'turbulence      Gauss linearUpwind default;', s)
assert a == 1 and b == 1, (a, b)
open(f, 'w').write(s)
EOF2
FIELDS[RK]="p U k epsilon nut"
FIELDS[RS]="p U k omega nut"
FIELDS[RA]="p U nuTilda nut"
FIELDS[RN]="p U k epsilon nut"
FIELDS[RD]="p U k epsilon nut"
[ "$(sed -n '/outlet1/,/}/p' "$W/RS/0/omega" | grep -c zeroGradient)" = 1 ] && grep -q -E 'default 0\.9;' "$W/RD/system/fvSolution" \
    && grep -q 'Gauss linearUpwind grad(U);' "$W/RD/system/fvSchemes" \
    && ! grep -q -E '"k\.\*"' "$W/RK/system/fvSolution" && say "staged: RK RS RA unnamed, RN named, RD default 0.9" PASS \
    || say "staged: RK RS RA unnamed, RN named, RD default 0.9" FAIL
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
    python3 -c "import sys; sys.exit(0 if float('${m%% *}') <= float('$BOUND') else 1)" \
        && say "$a: every entry of ${FIELDS[$a]} at steps 1-3 within $BOUND (worst $m)" PASS \
        || say "$a: every entry of ${FIELDS[$a]} at steps 1-3 within $BOUND (worst $m)" FAIL
    python3 -c "import sys; sys.exit(0 if float('$mp') <= float('$BOUND') else 1)" \
        && say "$a: phi, internal faces and every patch, at steps 1-3 within $BOUND (worst $mp)" PASS \
        || say "$a: phi, internal faces and every patch, at steps 1-3 within $BOUND (worst $mp)" FAIL
}
for a in RK RS RA RN RD
do
    cp -r "$W/$a" "$W/$a-of"
    cp -r "$W/$a" "$W/$a-br"
    ofrun "$W/$a-of" || { echo "FAIL: OpenFOAM $a"; tail -5 "$W/$a-of/log"; exit 1; }
    if brrun "$W/$a-br"; then
        judge "$a"
    else
        say "$a: brae ran ($(grep -a -m1 -i -E 'refus|error|unsupported' "$W/$a-br/log" | cut -c1-120))" FAIL
    fi
done
pr=$(python3 "$CMP" "$W/RK-of" "$W/RN-of" "$TIMES" epsilon | awk '/^MAX/{print $2}')
python3 -c "import sys; sys.exit(0 if '$pr' and float('$pr') > 1e3 * float('$BOUND') else 1)" \
    && say "premise: OpenFOAM relaxing k and epsilon with 1 (RN) moves epsilon from not relaxing them (RK): $pr" PASS \
    || say "premise: OpenFOAM relaxing k and epsilon with 1 (RN) moves epsilon from not relaxing them (RK): ${pr:-none}" FAIL
ctl()   # <env> <arm> <field>: brae rerun under a control, red on that field at some step
{
    local env=$1 a=$2 f=$3 name=${1%%=*} r
    rm -rf "$W/$a-$name"
    cp -r "$W/$a" "$W/$a-$name"
    brrun "$W/$a-$name" "$env" || { say "control $env: brae ran on $a" FAIL; return; }
    r=$(python3 "$CMP" "$W/$a-of" "$W/$a-$name" "$TIMES" "$f" | awk '/^MAX/{print $2, $3, $5}')
    python3 -c "import sys; sys.exit(0 if float('${r%% *}') > float('$BOUND') else 1)" \
        && say "control $env: $a $f goes red ($r)" PASS \
        || say "control $env: $a $f goes red (${r:-none})" FAIL
}
ctl BRAE_CONTROL_RELAX_UNNAMED_AS_ONE=1 RK epsilon
ctl BRAE_CONTROL_RELAX_UNNAMED_AS_ONE=1 RS omega
ctl BRAE_CONTROL_RELAX_UNNAMED_AS_ONE=1 RA nuTilda
ctl BRAE_CONTROL_RELAX_DEFAULT_IGNORED=1 RD U
[ "$fail" = 0 ] && echo "  PASS" || echo "  FAIL"
exit "$fail"
