#!/usr/bin/env bash
# pimpleFoam's kOmegaSST-family boundary updates on incompressible/pimpleFoam/RAS/TJunction, step by step against OpenFOAM
# v2412.
#
# WHAT IT HOLDS. kOmegaSSTBase::correct (v2412, kOmegaSSTBase.C:500-620; kOmegaSSTDES, DDES and IDDES run it as it is):
#   1. omega_.boundaryFieldRef().updateCoeffs() first (:541): omega's inletOutlet and turbulentMixingLengthFrequencyInlet
#      switch on the flux the last pressure corrector made, the frequency inlet's refValue from k's PATCH value and
#      Cmu = coeffDict().getOrDefault("Cmu", 0.09) (turbulentMixingLengthFrequencyInletFvPatchScalarField.C:137-148), and
#      omegaWallFunction writes the wall cells and its own faces with the wall function's Cmu (omegaWallFunction...C:191,
#      167-177);
#   2. CDkOmega from fvc::grad(k) and fvc::grad(omega) (:557), which read the STORED patch values -- updateCoeffs moves a
#      mixed patch's coefficients, not its value -- as do F1's patch half and both equations' limiters;
#   3. k's inletOutlet and turbulentIntensityKineticEnergyInlet in kEqn's constructor (:600), after omega is solved.
# brae's pimpleStep did the k and omega switches at the momentum assembly from the flux before the step (zero at step 1
# here), gave the frequency inlet and the wall function betaStar for Cmu, and evaluated the gradients' patch values from
# the switched coefficients.
#
# THE CASE. The tutorial staged as tests/pimple_tjunction_boundary_vs_openfoam.sh stages it -- fixed deltaT 0.001 to 0.003,
# every step written at writePrecision 17, every linear solve tight (tolerance 1e-12, relTol 0, maxIter 5000) -- with
# kOmegaSST in place of kEpsilon: omega 11111 (the tutorial's epsilon 200 over 0.09 k), a turbulentMixingLengthFrequencyInlet
# with the epsilon inlet's mixing length 0.01, inletOutlet outlets, omegaWallFunction walls; k and nut as shipped.
#   S1  kOmegaSST
#   S2  S1 with kOmegaSSTCoeffs { Cmu 0.12; }: the frequency inlet's Cmu alone (the model reads no Cmu)
#   S3  S1 with kOmegaSSTCoeffs { betaStar 0.1; }: the model's betaStar; the inlet and the wall function keep 0.09
#   D1  kOmegaSSTDDES (LES, cubeRootVol)
#   D2  D1 with kOmegaSSTDDESCoeffs { Cmu 0.12; }: the DDES coefficients come from kOmegaSSTDDESCoeffs (LESModel.C:72)
#   A1  S1 under the tutorial's adjustTimeStep (maxCo 5): OpenFOAM's first three steps, 0.0012 0.00264 0.004368
#   R1  S1 stopped at 0.002 and restarted (latestTime) to 0.003
#
# CHECKS, each entry (the internal field and every patch's value) on its own scale (tests/pimple_entry_compare.py):
# p U k omega nut at steps 1-3 within BOUND; phi, a surface field, the same way on its own line; R1 restarted against
# continuous within RBOUND (total_bc_roundtrip_vs_openfoam's). A1 is judged at its first two steps and its third is
# reported: under adjustTimeStep this case amplifies any difference about 20 times a step (A1 2.3e-10, 4.5e-08, 9.6e-07),
# as OpenFOAM's own one-ulp twin of the tutorial does (1.8e-13 at step 1 to 1.7e-02 at step 10, Arrhenius 5782463 and 5782481).
# Controls, each must go red on the entry it guards:
#   BRAE_CONTROL_TURB_SWITCH_AT_MOMENTUM=1     S1 and A1: k and omega switched at the momentum assembly
#   BRAE_CONTROL_TURB_GRAD_AFTER_UPDATE=1      S1: the gradients read the switched coefficients
#   BRAE_CONTROL_SST_INLET_CMU_BETASTAR=1      S2: the frequency inlet takes betaStar
#   BRAE_CONTROL_SST_WALL_CMU_BETASTAR=1       S3: omegaWallFunction takes betaStar
#   BRAE_CONTROL_SST_DES_COEFFS_RAS_NAME=1     D2: DDES coefficients read from kOmegaSSTCoeffs
#   BRAE_IO_STORED=0                            R1: the first restarted step's inletOutlet faces at fixedValue(inletValue)
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-26): worst entry over steps 1-3
#   before (develop 524e7dea, job 5805061): S1 k inlet 1.8e+05 at step 1 (zeroGradient 0.2 against OpenFOAM's 1.1e-06),
#     outlet1 nut 9.0, omega 0.91, k 0.34; with every k and omega inlet and outlet fixed 9.1e-08, so the closure itself
#     held; betaStar 0.1 with them fixed: omega on the walls 2.5e-02
#   after (job 5806035): S1 6.6e-08, S2 6.6e-08, S3 5.7e-08, D1 6.6e-08, D2 6.6e-08 (U internal at 0.003); phi 5.3e-08 at
#     most; A1 2.3e-10 and 4.5e-08 (step 3 9.6e-07); R1 restarted = continuous, 0. So BOUND 1e-6 is a decade above, as in
#     the boundary gate. Controls: 9.1e-01, 1.3e-03, 7.5e-02, 4.4e-03, 7.5e-02, 5.6e-03 (R1, on RBOUND). 19 s.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_TJ_BOUND:-1e-6}
RBOUND=${BRAE_TJ_RBOUND:-1e-12}
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
FIELDS="p U k omega nut"

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
# sst <case> <model> [<key>=<value> ...]: omega added, epsilon dropped, the model and its <model>Coeffs written,
# div(phi,omega) as the tutorial's other turbulence entries
sst()
{
    python3 - "$@" <<'EOF'
import os, re, sys
c, model = sys.argv[1], sys.argv[2]
coeffs = [a.split('=') for a in sys.argv[3:]]
def patch(name, body):
    return '    %s\n    {\n%s    }\n' % (name, ''.join('        %-15s %s;\n' % kv for kv in body))
om = [patch('inlet', [('type', 'turbulentMixingLengthFrequencyInlet'), ('mixingLength', '0.01'), ('value', '$internalField')])]
for o in ('outlet1', 'outlet2'):
    om.append(patch(o, [('type', 'inletOutlet'), ('inletValue', '$internalField'), ('value', '$internalField')]))
om.append(patch('defaultFaces', [('type', 'omegaWallFunction'), ('value', '$internalField')]))
open(c + '/0/omega', 'w').write(
    'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n    object      omega;\n}\n\n'
    'dimensions      [0 0 -1 0 0 0 0];\n\ninternalField   uniform 11111;\n\nboundaryField\n{\n%s}\n' % ''.join(om))
for f in ('epsilon', 'nuTilda'):
    if os.path.exists(c + '/0/' + f):
        os.remove(c + '/0/' + f)
tp = c + '/constant/turbulenceProperties'
s = open(tp).read()
cd = ''
if coeffs:
    cd = '    %sCoeffs\n    {\n%s    }\n' % (model, ''.join('        %-15s %s;\n' % (a, b) for a, b in coeffs))
if model == 'kOmegaSST':
    s, n = re.subn(r'RASModel\s+kEpsilon;', 'RASModel        kOmegaSST;\n' + cd.rstrip('\n'), s)
    assert n == 1
else:
    s, n = re.subn(r'simulationType\s+RAS;', 'simulationType  LES;', s)
    assert n == 1
    s, n = re.subn(r'\nRAS\s*\{.*?\n\}', '\nLES\n{\n    LESModel        %s;\n    turbulence      on;\n    printCoeffs     on;\n%s'
                   '    delta           cubeRootVol;\n    cubeRootVolCoeffs\n    {\n        deltaCoeff      1;\n    }\n}' % (model, cd), s,
                   count=1, flags=re.S)
    assert n == 1
open(tp, 'w').write(s)
fs = c + '/system/fvSchemes'
s = open(fs).read()
s, n = re.subn(r'(\n\s*div\(phi,epsilon\)\s+\$turbulence;)', r'\1\n    div(phi,omega)  $turbulence;', s)
assert n == 1
if 'wallDist' not in s:
    s += '\nwallDist\n{\n    method          meshWave;\n}\n'
open(fs, 'w').write(s)
fv = c + '/system/fvSolution'
s = open(fv).read()
s, n = re.subn(r'"\(U\|k\|epsilon\|s\)(Final)?"', lambda m: '"(U|k|epsilon|omega|s)%s"' % (m.group(1) or ''), s)
assert n == 2, n
open(fv, 'w').write(s)
EOF
}
mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
for k in "adjustTimeStep +yes" "deltaT +0.001;" "maxCo +5;"; do
    grep -qE "^$k" "$W/base/system/controlDict" || { echo "FAIL: the tutorial no longer has \`$k\` -- the gate's premise moved"; exit 1; }
done
grep -q 'type            turbulentIntensityKineticEnergyInlet;' "$W/base/0/k" \
    && grep -q 'RASModel        kEpsilon;' "$W/base/constant/turbulenceProperties" \
    && grep -qE 'div\(phi,epsilon\) +\$turbulence;' "$W/base/system/fvSchemes" \
    || { echo "FAIL: the tutorial's k inlet, model or schemes moved -- the gate's premise moved"; exit 1; }
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
setkey "$W/base" endTime 0.003
setkey "$W/base" writeControl timeStep
setkey "$W/base" writeInterval 1
setkey "$W/base" writePrecision 17
tight "$W/base"
stage() { mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; }
for a in S1 S2 S3 D1 D2 R1 A1; do stage "$W/$a"; done
for a in S1 S2 S3 D1 D2 R1; do setkey "$W/$a" adjustTimeStep no; done
sst "$W/S1" kOmegaSST
sst "$W/S2" kOmegaSST Cmu=0.12
sst "$W/S3" kOmegaSST betaStar=0.1
sst "$W/D1" kOmegaSSTDDES
sst "$W/D2" kOmegaSSTDDES Cmu=0.12
sst "$W/R1" kOmegaSST
sst "$W/A1" kOmegaSST
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
        && say "$a: every entry of p,U,k,omega,nut at $3 within $BOUND (worst $m)" PASS \
        || say "$a: every entry of p,U,k,omega,nut at $3 within $BOUND (worst $m)" FAIL
    python3 -c "import sys; sys.exit(0 if float('$mp') <= float('$BOUND') else 1)" \
        && say "$a: phi, internal faces and every patch, at $3 within $BOUND (worst $mp)" PASS \
        || say "$a: phi, internal faces and every patch, at $3 within $BOUND (worst $mp)" FAIL
}

for a in S1 S2 S3 D1 D2; do
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
ofrun "$W/R1-of" || { echo "FAIL: OpenFOAM R1"; tail -5 "$W/R1-of/log"; exit 1; }
brrun "$W/R1-br" || { echo "FAIL: brae R1"; tail -5 "$W/R1-br/log"; exit 1; }
setkey "$W/R1-rs" endTime 0.002
brrun "$W/R1-rs" || { echo "FAIL: brae R1 to 0.002"; tail -5 "$W/R1-rs/log"; exit 1; }
cp -r "$W/R1-rs" "$W/R1-ctl"
setkey "$W/R1-rs" endTime 0.003
setkey "$W/R1-rs" startFrom latestTime
brrun "$W/R1-rs" || { echo "FAIL: brae R1 restarted"; tail -5 "$W/R1-rs/log"; exit 1; }
judge R1 "$TIMES" "steps 1-3 (continuous)"
python3 "$CMP" "$W/R1-br" "$W/R1-rs" 0.003 $FIELDS phi > "$W/R1.rst"
mr=$(awk '/^MAX/{print $2}' "$W/R1.rst")
python3 -c "import sys; sys.exit(0 if float('$mr') <= float('$RBOUND') else 1)" \
    && say "R1: restarted from 0.002 = continuous at 0.003, every entry of p,U,k,omega,nut,phi within $RBOUND (worst $mr)" PASS \
    || say "R1: restarted from 0.002 = continuous at 0.003, every entry of p,U,k,omega,nut,phi within $RBOUND (worst $mr)" FAIL

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
control BRAE_CONTROL_TURB_SWITCH_AT_MOMENTUM=1 S1 0.001 omega outlet1
control BRAE_CONTROL_TURB_SWITCH_AT_MOMENTUM=1 A1 "${AT%%,*}" omega outlet1
control BRAE_CONTROL_TURB_GRAD_AFTER_UPDATE=1 S1 0.002 omega internal
control BRAE_CONTROL_SST_INLET_CMU_BETASTAR=1 S2 0.001 omega inlet
control BRAE_CONTROL_SST_WALL_CMU_BETASTAR=1 S3 0.001 omega internal
control BRAE_CONTROL_SST_DES_COEFFS_RAS_NAME=1 D2 0.001 omega inlet
# BRAE_IO_STORED=0 on the restart: against brae's own continuous run, on RBOUND
setkey "$W/R1-ctl" endTime 0.003
setkey "$W/R1-ctl" startFrom latestTime
brrun "$W/R1-ctl" BRAE_IO_STORED=0 || { say "control BRAE_IO_STORED=0: brae ran" FAIL; }
mc=$(python3 "$CMP" "$W/R1-br" "$W/R1-ctl" 0.003 $FIELDS | awk '/^MAX/{print $2}')
python3 -c "import sys; sys.exit(0 if float('$mc') > float('$RBOUND') else 1)" \
    && say "control BRAE_IO_STORED=0: R1 restarted vs continuous goes red (rel $mc)" PASS \
    || say "control BRAE_IO_STORED=0: R1 restarted vs continuous goes red (rel $mc)" FAIL

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
