#!/usr/bin/env bash
# pimpleFoam's SpalartAllmaras-family boundary updates on incompressible/pimpleFoam/RAS/TJunction, step by step against
# OpenFOAM v2412.
#
# WHAT IT HOLDS. SpalartAllmarasBase::correct (v2412, SpalartAllmarasBase.C:440-492; SpalartAllmarasDES, DDES and IDDES run
# it as it is, overriding dTilda only):
#   1. fvc::grad(U), dTilda and Stilda from U as the PIMPLE loop left it (:461-463);
#   2. nuTildaEqn's first fvMatrix (fvm::ddt) calls nuTilda_.boundaryFieldRef().updateCoeffs() (fvMatrix.C:396): its
#      inletOutlet faces switch on the flux the last pressure corrector made;
#   3. DnuTildaEff's patch half and the Cb2 term's fvc::grad(nuTilda) (:470-471) read the STORED patch values --
#      updateCoeffs moves a mixed patch's coefficients, not its value -- as do the limitedLinear limiter's gradient;
#   4. solve, bound, correctBoundaryConditions (:485-488), then correctNut (nut = nuTilda fv1, its wall function).
# Nothing earlier in the step touches nuTilda's boundary. brae's pimpleStep switched nuTilda's inletOutlet at the momentum
# assembly, from the flux before the step.
#
# THE CASE. The tutorial staged as tests/pimple_tjunction_boundary_vs_openfoam.sh stages it -- fixed deltaT 0.001 to 0.003,
# every step written at writePrecision 17, every linear solve tight (tolerance 1e-12, relTol 0, maxIter 5000) -- with
# SpalartAllmaras in place of kEpsilon: nuTilda 4e-5 (4 nu) inside, inletOutlet on the inlet (inletValue 6e-5) and both
# outlets (inletValue 2e-5), each face's `value` the internal 4e-5, fixedValue 0 on the walls; nut's walls
# nutUSpaldingWallFunction; k and epsilon dropped (the model reads neither).
#   S1  SpalartAllmaras
#   D1  SpalartAllmarasDDES (LES, cubeRootVol)
#   I1  SpalartAllmarasIDDES (LES, IDDESDelta with hmax maxDeltaxyz)
#   A1  S1 under the tutorial's adjustTimeStep (maxCo 5): OpenFOAM's first three steps
#   R1  S1 stopped at 0.002 and restarted (latestTime) to 0.003, OpenFOAM's restart the oracle
#
# CHECKS, each entry (the internal field and every patch's value) on its own scale (tests/pimple_entry_compare.py):
# p U nuTilda nut at steps 1-3 within BOUND; phi, a surface field, the same way on its own line; R1's restart against
# OpenFOAM's restart within BOUND. A1 is judged at its first two steps and its third is reported: under adjustTimeStep this
# case amplifies a difference about 20 times a step (the SST gate's A1).
# A restart here is NOT exact in OpenFOAM: OpenFOAM restarted from 0.002 is 5.4e-06 from its continuous run at 0.003, in
# nutUSpaldingWallFunction's faces, whose Newton starts from the patch's stored nut and stops at tolerance 0.01
# (nutUSpaldingWallFunctionFvPatchScalarField.C:143-181): the restarted run evaluates the wall function once more before its
# first step. So R1 holds brae's restart to OpenFOAM's restart, not to continuity, and reports both restarts' distance from
# their own continuous runs (equal to every digit, 5.396e-06).
# Controls, each must go red on the entry it guards:
#   BRAE_CONTROL_TURB_SWITCH_AT_MOMENTUM=1     S1 and A1: nuTilda switched at the momentum assembly
#   BRAE_CONTROL_TURB_GRAD_AFTER_UPDATE=1      S1: the gradients read the switched coefficients (the inletValues are apart
#                                              from the faces' stored values so that this can show)
#   BRAE_IO_STORED=0                            R1: the first restarted step's inletOutlet faces at fixedValue(inletValue)
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-27): worst entry over steps 1-3
#   before (develop 1bf7078a, job 5825301): S1, D1, I1 nut outlet1 1.3e+01 at step 1 (nuTilda outlet1 9.9e-01: OpenFOAM switched
#     the face to inflow, 2e-05, brae kept it zeroGradient), nuTilda inlet 3.7e-01, nut internal 2.3e-02; A1 1.3e+01; R1 brae
#     restarted against OpenFOAM restarted 2.1e-02 (nut internal)
#   after (job 5825301): S1, D1, I1 7.2e-08 (nut on the walls at 0.003), phi 4.7e-08; A1 5.0e-08 (step 3 5.9e-07); R1 restarted
#     against OpenFOAM restarted 7.2e-08. So BOUND 1e-6 is a decade above, as in the other TJunction gates. Controls: 9.9e-01,
#     9.9e-01, 2.3e-03, and the restart's 2.0e-03. 15 s.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_TJ_BOUND:-1e-6}
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
W=${BRAE_TJ_KEEP:-$(mktemp -d)}
[ -n "${BRAE_TJ_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-86s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
TIMES=0.001,0.002,0.003
FIELDS="p U nuTilda nut"

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
# sa <case> <model>: nuTilda written with the SpalartAllmaras conditions above, k and epsilon dropped, nut's walls
# nutUSpaldingWallFunction, the model selected (RAS, or LES with its delta)
sa()
{
    python3 - "$@" <<'EOF'
import os, re, sys
c, model = sys.argv[1], sys.argv[2]
def patch(name, body):
    return '    %s\n    {\n%s    }\n' % (name, ''.join('        %-15s %s;\n' % kv for kv in body))
# inletValues apart from the internal value, so that a face's stored value (the file's, then what the last evaluate left)
# differs from what its switched coefficients give: equal values cannot witness which of the two the gradients read
nt = [patch('inlet', [('type', 'inletOutlet'), ('inletValue', 'uniform 6e-05'), ('value', '$internalField')])]
for o in ('outlet1', 'outlet2'):
    nt.append(patch(o, [('type', 'inletOutlet'), ('inletValue', 'uniform 2e-05'), ('value', '$internalField')]))
nt.append(patch('defaultFaces', [('type', 'fixedValue'), ('value', 'uniform 0')]))
open(c + '/0/nuTilda', 'w').write(
    'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n    object      nuTilda;\n}\n\n'
    'dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform 4e-05;\n\nboundaryField\n{\n%s}\n' % ''.join(nt))
for f in ('k', 'epsilon'):
    if os.path.exists(c + '/0/' + f):
        os.remove(c + '/0/' + f)
nut = c + '/0/nut'
s = open(nut).read()
s, n = re.subn(r'type\s+nutkWallFunction;', 'type            nutUSpaldingWallFunction;', s)
assert n == 1, n
open(nut, 'w').write(s)
tp = c + '/constant/turbulenceProperties'
s = open(tp).read()
if model == 'SpalartAllmaras':
    s, n = re.subn(r'RASModel\s+kEpsilon;', 'RASModel        SpalartAllmaras;', s)
    assert n == 1
else:
    delta = ('    delta           cubeRootVol;\n    cubeRootVolCoeffs\n    {\n        deltaCoeff      1;\n    }\n'
             if model == 'SpalartAllmarasDDES' else
             '    delta           IDDESDelta;\n    IDDESDeltaCoeffs\n    {\n        hmax            maxDeltaxyz;\n'
             '        maxDeltaxyzCoeffs\n        {\n        }\n    }\n')
    s, n = re.subn(r'simulationType\s+RAS;', 'simulationType  LES;', s)
    assert n == 1
    s, n = re.subn(r'\nRAS\s*\{.*?\n\}', '\nLES\n{\n    LESModel        %s;\n    turbulence      on;\n    printCoeffs     on;\n%s}'
                   % (model, delta), s, count=1, flags=re.S)
    assert n == 1
open(tp, 'w').write(s)
fs = c + '/system/fvSchemes'
s = open(fs).read()
assert re.search(r'div\(phi,nuTilda\)\s+\$turbulence;', s)
if 'wallDist' not in s:
    s += '\nwallDist\n{\n    method          meshWave;\n}\n'
open(fs, 'w').write(s)
fv = c + '/system/fvSolution'
s = open(fv).read()
s, n = re.subn(r'"\(U\|k\|epsilon\|s\)(Final)?"', lambda m: '"(U|k|epsilon|nuTilda|s)%s"' % (m.group(1) or ''), s)
assert n == 2, n
open(fv, 'w').write(s)
EOF
}
mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
for k in "adjustTimeStep +yes" "deltaT +0.001;" "maxCo +5;"; do
    grep -qE "^$k" "$W/base/system/controlDict" || { echo "FAIL: the tutorial no longer has \`$k\` -- the gate's premise moved"; exit 1; }
done
grep -q 'type            nutkWallFunction;' "$W/base/0/nut" \
    && grep -q 'RASModel        kEpsilon;' "$W/base/constant/turbulenceProperties" \
    && grep -qE 'div\(phi,nuTilda\) +\$turbulence;' "$W/base/system/fvSchemes" \
    || { echo "FAIL: the tutorial's nut walls, model or schemes moved -- the gate's premise moved"; exit 1; }
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
setkey "$W/base" endTime 0.003
setkey "$W/base" writeControl timeStep
setkey "$W/base" writeInterval 1
setkey "$W/base" writePrecision 17
tight "$W/base"
stage() { mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; }
for a in S1 D1 I1 R1 A1; do stage "$W/$a"; done
for a in S1 D1 I1 R1; do setkey "$W/$a" adjustTimeStep no; done
sa "$W/S1" SpalartAllmaras
sa "$W/D1" SpalartAllmarasDDES
sa "$W/I1" SpalartAllmarasIDDES
sa "$W/R1" SpalartAllmaras
sa "$W/A1" SpalartAllmaras
ofrun() { ( cd "$1" && nice -n 10 pimpleFoam > log 2>&1 ); }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ); }
judge()   # <arm> <times> <label>
{
    local a=$1 t=$2 m
    python3 "$CMP" "$W/$a-of" "$W/$a-br" "$t" $FIELDS > "$W/$a.cmp"
    sed 's/^/      /' "$W/$a.cmp"
    m=$(awk '/^MAX/{print $2}' "$W/$a.cmp")
    python3 "$CMP" "$W/$a-of" "$W/$a-br" "$t" phi > "$W/$a.phi"
    local mp
    mp=$(awk '/^MAX/{print $2}' "$W/$a.phi")
    if [ "$BOUND" = report ]; then
        say "$a: worst entry of $3, rel $m; phi $mp (report only)" INFO
        return
    fi
    python3 -c "import sys; sys.exit(0 if float('$m') <= float('$BOUND') else 1)" \
        && say "$a: every entry of p,U,nuTilda,nut at $3 within $BOUND (worst $m)" PASS \
        || say "$a: every entry of p,U,nuTilda,nut at $3 within $BOUND (worst $m)" FAIL
    python3 -c "import sys; sys.exit(0 if float('$mp') <= float('$BOUND') else 1)" \
        && say "$a: phi, internal faces and every patch, at $3 within $BOUND (worst $mp)" PASS \
        || say "$a: phi, internal faces and every patch, at $3 within $BOUND (worst $mp)" FAIL
}

for a in S1 D1 I1; do
    cp -r "$W/$a" "$W/$a-of"
    cp -r "$W/$a" "$W/$a-br"
    ofrun "$W/$a-of" || { echo "FAIL: OpenFOAM $a"; tail -5 "$W/$a-of/log"; exit 1; }
    brrun "$W/$a-br" || { echo "FAIL: brae $a"; tail -5 "$W/$a-br/log"; exit 1; }
    judge "$a" "$TIMES" "steps 1-3"
done

# A1: the tutorial's adjustTimeStep; OpenFOAM's first three written times, which brae's clock must land on too.
cp -r "$W/A1" "$W/A1-of"
cp -r "$W/A1" "$W/A1-br"
# Long enough for three steps from 0.001 growing by at most 1.2 a step (setDeltaT.H); brae then stops at OpenFOAM's third.
setkey "$W/A1-of" endTime 0.006
ofrun "$W/A1-of" || { echo "FAIL: OpenFOAM A1"; tail -5 "$W/A1-of/log"; exit 1; }
AT=$(cd "$W/A1-of" && ls -d 0.* 2> /dev/null | sort -g | head -3 | paste -sd, -)
[ "$(echo "$AT" | tr ',' '\n' | wc -l)" = 3 ] || { echo "FAIL: OpenFOAM A1 wrote fewer than three steps ($AT)"; exit 1; }
LAST=$(echo "$AT" | awk -F, '{print $3}')
setkey "$W/A1" endTime "$LAST"
setkey "$W/A1-br" endTime "$LAST"
brrun "$W/A1-br" || { echo "FAIL: brae A1"; tail -5 "$W/A1-br/log"; exit 1; }
judge A1 "${AT%,*}" "adjustTimeStep steps 1-2 (${AT%,*})"
m3=$(python3 "$CMP" "$W/A1-of" "$W/A1-br" "$LAST" $FIELDS phi | awk '/^MAX/{print $2}')
say "A1: step 3 ($LAST), the case's amplification of the steps before (see the header), rel $m3" INFO

# R1: continuous to 0.003, against stopped at 0.002 and restarted from it.
cp -r "$W/R1" "$W/R1-of"
cp -r "$W/R1" "$W/R1-br"
cp -r "$W/R1" "$W/R1-rs"
cp -r "$W/R1" "$W/R1-ors"
ofrun "$W/R1-of" || { echo "FAIL: OpenFOAM R1"; tail -5 "$W/R1-of/log"; exit 1; }
# OpenFOAM's own restart, the oracle of brae's: nutUSpaldingWallFunction's Newton starts from the patch's stored nut and
# stops at tolerance 0.01 (nutUSpaldingWallFunctionFvPatchScalarField.C:143-181), so a restart is exact only if it starts
# every evaluation where the continuous run did
setkey "$W/R1-ors" endTime 0.002
ofrun "$W/R1-ors" || { echo "FAIL: OpenFOAM R1 to 0.002"; tail -5 "$W/R1-ors/log"; exit 1; }
setkey "$W/R1-ors" endTime 0.003
setkey "$W/R1-ors" startFrom latestTime
ofrun "$W/R1-ors" || { echo "FAIL: OpenFOAM R1 restarted"; tail -5 "$W/R1-ors/log"; exit 1; }
mo=$(python3 "$CMP" "$W/R1-of" "$W/R1-ors" 0.003 $FIELDS phi | awk '/^MAX/{print $2, $4, $5}')
say "R1: OpenFOAM restarted from 0.002 against OpenFOAM continuous at 0.003: $mo" INFO
brrun "$W/R1-br" || { echo "FAIL: brae R1"; tail -5 "$W/R1-br/log"; exit 1; }
setkey "$W/R1-rs" endTime 0.002
brrun "$W/R1-rs" || { echo "FAIL: brae R1 to 0.002"; tail -5 "$W/R1-rs/log"; exit 1; }
cp -r "$W/R1-rs" "$W/R1-ctl"
setkey "$W/R1-rs" endTime 0.003
setkey "$W/R1-rs" startFrom latestTime
brrun "$W/R1-rs" || { echo "FAIL: brae R1 restarted"; tail -5 "$W/R1-rs/log"; exit 1; }
judge R1 "$TIMES" "steps 1-3 (continuous)"
python3 "$CMP" "$W/R1-ors" "$W/R1-rs" 0.003 $FIELDS phi > "$W/R1.ors"
mr=$(awk '/^MAX/{print $2}' "$W/R1.ors")
if [ "$BOUND" = report ]; then
    say "R1: brae restarted against OpenFOAM restarted at 0.003, rel $mr (report only)" INFO
else
    python3 -c "import sys; sys.exit(0 if float('$mr') <= float('$BOUND') else 1)" \
        && say "R1: brae restarted against OpenFOAM restarted at 0.003, every entry of p,U,nuTilda,nut,phi within $BOUND (worst $mr)" PASS \
        || say "R1: brae restarted against OpenFOAM restarted at 0.003, every entry of p,U,nuTilda,nut,phi within $BOUND (worst $mr)" FAIL
fi
mb=$(python3 "$CMP" "$W/R1-br" "$W/R1-rs" 0.003 $FIELDS phi | awk '/^MAX/{print $2, $4, $5}')
say "R1: brae restarted against brae continuous at 0.003 (OpenFOAM's own above): $mb" INFO

# Each control must make ITS entry exceed the bound (`report` uses 1e-6 here).
CB=$BOUND
[ "$CB" = report ] && CB=1e-6
control()   # <env> <arm> <time> <field> <entry>
{
    local env=$1 arm=$2 t=$3 f=$4 entry=$5 name=${1%%=*}
    rm -rf "$W/$arm-$name"
    cp -r "$W/$arm" "$W/$arm-$name"
    brrun "$W/$arm-$name" "$env" || { say "control $env: brae ran" FAIL; return; }
    local r
    r=$(python3 "$CMP" "$W/$arm-of" "$W/$arm-$name" "$t" "$f" | awk -v e="$entry" '$3 == e {print $10}')
    python3 -c "import sys; sys.exit(0 if '$r' and float('$r') > float('$CB') else 1)" \
        && say "control $env: $arm $f $entry at $t goes red (rel $r)" PASS \
        || say "control $env: $arm $f $entry at $t goes red (rel ${r:-none})" FAIL
}
control BRAE_CONTROL_TURB_SWITCH_AT_MOMENTUM=1 S1 0.001 nuTilda outlet1
control BRAE_CONTROL_TURB_SWITCH_AT_MOMENTUM=1 A1 "${AT%%,*}" nuTilda outlet1
control BRAE_CONTROL_TURB_GRAD_AFTER_UPDATE=1 S1 0.002 nuTilda internal
# BRAE_IO_STORED=0 on the restart: against OpenFOAM's restart, on the bound
setkey "$W/R1-ctl" endTime 0.003
setkey "$W/R1-ctl" startFrom latestTime
brrun "$W/R1-ctl" BRAE_IO_STORED=0 || { say "control BRAE_IO_STORED=0: brae ran" FAIL; }
mc=$(python3 "$CMP" "$W/R1-ors" "$W/R1-ctl" 0.003 $FIELDS | awk '/^MAX/{print $2}')
python3 -c "import sys; sys.exit(0 if float('$mc') > float('$CB') else 1)" \
    && say "control BRAE_IO_STORED=0: R1 restarted against OpenFOAM restarted goes red (rel $mc)" PASS \
    || say "control BRAE_IO_STORED=0: R1 restarted against OpenFOAM restarted goes red (rel $mc)" FAIL

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
