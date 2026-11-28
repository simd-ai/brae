#!/usr/bin/env bash
# pimpleFoam's DES and DDES length scales against OpenFOAM v2412, on a case where the limiter acts: a 3-D backward-facing step whose
# separated shear layer leaves OpenFOAM's DDES unshielded.
#
# WHAT IT HOLDS (v2412).
#   kOmegaSSTDES / kOmegaSSTDDES: the k equation's destruction is sqrt(k)/dTilda (epsilonByk, kOmegaSSTDES.C), assembled in
#   kEqn after omega is solved (kOmegaSSTBase.C:600), with lRAS = sqrt(k)/(betaStar omega), lLES = CDES delta and
#   CDES = F1 CDESkom + (1 - F1) CDESkeps (0.82, 0.60; kOmegaSSTDES.C:211-225, .H:104-107).
#     DES   dTilda = min(lLES, lRAS)
#     DDES  dTilda = max(lRAS - fd max(lRAS - lLES, 0), SMALL), fd = 1 - tanh((Cd1 r)^Cd2) (kOmegaSSTDDES.C:42-47, 90-102),
#           r = min(nuEff/(|grad U| (kappa y)^2), 10) (kOmegaSSTDES.C:60-70), Cd1 20, Cd2 3, kappa 0.41
#   SpalartAllmarasDES: dTilda = min(psi CDES delta, y) (SpalartAllmarasDES.C:120-132); SpalartAllmarasDDES:
#   dTilda = y - fd max(y - psi CDES delta, 0), fd = 1 - tanh((Cd1 r)^Cd2) (SpalartAllmarasDDES.C)
#
# THE CASE. A step written here: an inlet channel 0.5 x 0.1 over the step, then 1.5 x 0.2, all 0.2 deep, 8,960 cells, walls
# all round but the inlet and the outlet; nu 1e-4, U 10 at the inlet. Each model starts from OpenFOAM's own steady RANS
# (simpleFoam, 50 iterations: kOmegaSST for the kOmegaSST family, SpalartAllmaras for the SpalartAllmaras family), then
# pimpleFoam, backward, 2 x 2, 5 steps of 1e-4, written every step at 17 digits, every solve tight (p 1e-14, the rest 1e-13,
# relTol 0). delta cubeRootVol.
#   SDD kOmegaSSTDDES   SDE kOmegaSSTDES   ADD SpalartAllmarasDDES   ADE SpalartAllmarasDES
# PREMISE, asserted from OpenFOAM's own runs: each DES model's fields differ from its RAS model's (the limiter acts), and each
# DDES from its DES (the shielding acts).
#
#   SDK SDD with kOmegaSSTDDESCoeffs { CDESkom 0.9; CDESkeps 0.5; Cd1 15; kappa 0.38; } (the omega wall function keeps the wall
#       patch's kappa 0.41: OpenFOAM's takes the patch's own, omegaWallFunctionFvPatchScalarField.C:191-193)
#   SRA kOmegaSST, ARA SpalartAllmaras: the RAS closures on the same starts (ARA's nutUSpaldingWallFunction walls show the wall
#       nut held across the two outer correctors, as OpenFOAM holds nut's patch values until correctNut)
#   SRP SRA with simpleFoam's phi at iteration 50 carried over, which is not flux(U). On backward's first step OpenFOAM applies
#       Euler's form of fvc::ddtCorr to that difference: backwardDdtScheme::deltaT0_(vf) is GREAT while timeIndex() < 2
#       (backwardDdtScheme.C:62-71), so fvcDdtPhiCorr's coefft0 is 1 and coefft00 about 1e-36 (.C:738-776)
# The other arms' starts carry no phi file, so both codes start from flux(U) and that correction is zero there.
#
# CHECKS, every entry on its own scale (tests/pimple_entry_compare.py) at steps 1-5 within BOUND: kOmegaSST family p U k
# omega nut, SpalartAllmaras family p U nuTilda nut; phi on its own line. Controls, each red on the entry it guards:
#   BRAE_CONTROL_DES_F2_SHIELDING=1        SDD: max(lRAS/(CDES delta) (1 - F2), 1) from the omega before its solve
#   BRAE_CONTROL_DES_AS_DDES=1             SDE and ADE: plain DES with DDES's delay function
#   BRAE_CONTROL_SST_DES_OLD_COEFFS=1      SDK: CDES1/CDES2 at 0.78/0.61, the dictionary's kappa on the walls
#   BRAE_CONTROL_NUT_BOUNDARY_ONE_USE=1    ARA: the wall nut evaluated again at the second outer corrector
#   BRAE_CONTROL_BACKWARD_FIRST_NO_DDTCORR=1  SRP: no fvc::ddtCorr on backward's first step
# PREMISE for SRP: OpenFOAM from the start with the phi file differs from OpenFOAM without it (SRA) at step 1.
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-27): worst entry over steps 1-5
#   before (develop 6708c38a, job 5838209): SDD 2.1e-02, SDE 6.5e-02, ADE 3.4e-01 (k, k, nut internal at step 5), SDK 7.7e-02
#     (omega on the walls at step 1), ARA and ADD 2.2e-04 (nut on the walls at step 1), SRA 8.2e-07
#   after (jobs 5830099, 5830744): every arm 2.9e-13 to 5.2e-13 (p), phi 2.4e-15 at most. So BOUND 1e-10: two decades above, for another
#   GPU's arithmetic. Controls: 2.1e-02, 6.4e-02, 3.4e-01, 6.2e-03, 2.2e-04. Premises 6.1e-02, 6.4e-02, 2.6e-01, 3.4e-01; SDK
#   against SDD 8.4e-03. 42 s.
#   SRP (job 5861987): develop 5b2b4144 1.17 (p internal at step 1; phi 4.4e-02), after 2.2e-13 (phi 1.9e-15); the other arms'
#   numbers unchanged to the digit. Control 1.17, premise 5.7e-01 (p at step 1).
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_DES_BOUND:-1e-10}
PREMISE=1e-3
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
W=${BRAE_DES_KEEP:-$(mktemp -d)}
[ -n "${BRAE_DES_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-86s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
TIMES=0.0001,0.0002,0.0003,0.0004,0.0005
LAST=0.0005

cat > "$W/step.py" << 'PY'
import os
import sys

HEAD = 'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n    object      %s;\n}\n\n'


def write(case, name, cls, body):
    path = os.path.join(case, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(HEAD % (cls, os.path.basename(name)) + body)


def field(case, name, cls, dims, value, inlet, outlet, walls):
    write(case, '0/' + name, cls, 'dimensions      %s;\n\ninternalField   uniform %s;\n\nboundaryField\n{\n'
          '    inlet\n    {\n        %s\n    }\n    outlet\n    {\n        %s\n    }\n    walls\n    {\n        %s\n    }\n}\n'
          % (dims, value, inlet, outlet, walls))


def rans(case, family):
    """the step and its steady RANS start: simpleFoam, 50 iterations"""
    write(case, 'system/blockMeshDict', 'dictionary',
          'scale 1;\nvertices\n(\n'
          '    (0 0.1 0) (0.5 0.1 0) (0.5 0.2 0) (0 0.2 0) (0.5 0 0) (2 0 0) (2 0.1 0) (2 0.2 0)\n'
          '    (0 0.1 0.2) (0.5 0.1 0.2) (0.5 0.2 0.2) (0 0.2 0.2) (0.5 0 0.2) (2 0 0.2) (2 0.1 0.2) (2 0.2 0.2)\n);\n'
          'blocks\n(\n'
          '    hex (0 1 2 3 8 9 10 11) (10 8 16) simpleGrading (1 1 1)\n'
          '    hex (4 5 6 1 12 13 14 9) (30 8 16) simpleGrading (1 1 1)\n'
          '    hex (1 6 7 2 9 14 15 10) (30 8 16) simpleGrading (1 1 1)\n);\n'
          'edges ();\nboundary\n(\n'
          '    inlet { type patch; faces ((0 8 11 3)); }\n'
          '    outlet { type patch; faces ((5 6 14 13) (6 7 15 14)); }\n'
          '    walls { type wall; faces ((0 1 9 8) (3 2 10 11) (0 1 2 3) (8 9 10 11) (4 1 9 12) (4 5 13 12) (4 5 6 1)\n'
          '        (12 13 14 9) (2 7 15 10) (1 6 7 2) (9 14 15 10)); }\n);\nmergePatchPairs ();\n')
    write(case, 'system/controlDict', 'dictionary',
          'application     simpleFoam;\nstartFrom       startTime;\nstartTime       0;\nstopAt          endTime;\nendTime         50;\n'
          'deltaT          1;\nwriteControl    timeStep;\nwriteInterval   50;\nwriteFormat     ascii;\nwritePrecision  17;\n'
          'writeCompression off;\ntimeFormat      general;\ntimePrecision   6;\nrunTimeModifiable false;\n')
    second = 'k|omega' if family == 'sst' else 'nuTilda'
    write(case, 'system/fvSchemes', 'dictionary',
          'ddtSchemes\n{\n    default         steadyState;\n}\n'
          'gradSchemes\n{\n    default         Gauss linear;\n}\n'
          'divSchemes\n{\n    default         none;\n    div(phi,U)      bounded Gauss linearUpwind grad(U);\n'
          '    div(phi,k)      bounded Gauss limitedLinear 1;\n    div(phi,omega)  bounded Gauss limitedLinear 1;\n'
          '    div(phi,nuTilda) bounded Gauss limitedLinear 1;\n    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n'
          'laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n'
          'interpolationSchemes\n{\n    default         linear;\n}\n'
          'snGradSchemes\n{\n    default         corrected;\n}\n'
          'wallDist\n{\n    method          meshWave;\n}\n')
    write(case, 'system/fvSolution', 'dictionary',
          'solvers\n{\n    p\n    {\n        solver          GAMG;\n        smoother        GaussSeidel;\n'
          '        tolerance       1e-8;\n        relTol          0.01;\n        maxIter         5000;\n    }\n'
          '    "(U|%s)"\n    {\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n'
          '        tolerance       1e-10;\n        relTol          0.1;\n        maxIter         2000;\n    }\n}\n'
          'SIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n}\n'
          'relaxationFactors\n{\n    fields\n    {\n        p               0.3;\n    }\n'
          '    equations\n    {\n        U               0.7;\n        ".*"            0.7;\n    }\n}\n' % second)
    write(case, 'constant/transportProperties', 'dictionary', 'transportModel  Newtonian;\nnu              1e-4;\n')
    model = 'kOmegaSST' if family == 'sst' else 'SpalartAllmaras'
    write(case, 'constant/turbulenceProperties', 'dictionary',
          'simulationType  RAS;\nRAS\n{\n    RASModel        %s;\n    turbulence      on;\n    printCoeffs     on;\n}\n' % model)
    io = 'type inletOutlet; inletValue uniform %s; value uniform %s;'
    field(case, 'U', 'volVectorField', '[0 1 -1 0 0 0 0]', '(10 0 0)', 'type fixedValue; value uniform (10 0 0);',
          'type inletOutlet; inletValue uniform (0 0 0); value uniform (10 0 0);', 'type noSlip;')
    field(case, 'p', 'volScalarField', '[0 2 -2 0 0 0 0]', '0', 'type zeroGradient;', 'type fixedValue; value uniform 0;',
          'type zeroGradient;')
    if family == 'sst':
        # nut = k/omega near 7e-4 in the core: fd near 1 in the shear layer, lLES = CDES cbrt(V) below lRAS there
        field(case, 'k', 'volScalarField', '[0 2 -2 0 0 0 0]', '0.0375', 'type fixedValue; value uniform 0.0375;',
              io % ('0.0375', '0.0375'), 'type kqRWallFunction; value uniform 0.0375;')
        field(case, 'omega', 'volScalarField', '[0 0 -1 0 0 0 0]', '56.2963', 'type fixedValue; value uniform 56.2963;',
              io % ('56.2963', '56.2963'), 'type omegaWallFunction; value uniform 56.2963;')
        field(case, 'nut', 'volScalarField', '[0 2 -1 0 0 0 0]', '0', 'type calculated; value uniform 0;',
              'type calculated; value uniform 0;', 'type nutkWallFunction; value uniform 0;')
    else:
        # nuTilda 1.4e-3: nut = nuTilda fv1 near the kOmegaSST start's 7e-4
        field(case, 'nuTilda', 'volScalarField', '[0 2 -1 0 0 0 0]', '1.4e-3', 'type fixedValue; value uniform 1.4e-3;',
              io % ('1.4e-3', '1.4e-3'), 'type fixedValue; value uniform 0;')
        field(case, 'nut', 'volScalarField', '[0 2 -1 0 0 0 0]', '0', 'type calculated; value uniform 0;',
              'type calculated; value uniform 0;', 'type nutUSpaldingWallFunction; value uniform 0;')


def transient(case, start, model, coeffs=''):
    """pimpleFoam from the RANS start: backward, 2 x 2, 5 steps of 1e-4, every solve tight; the model, delta and the model's
    <model>Coeffs entries if any"""
    os.makedirs(case + '/0', exist_ok=True)
    # phi is not carried over here (mkcase's KEEP_PHI does it for SRP): simpleFoam's flux at iteration 50 is not flux(U), and
    # on backward's first step OpenFOAM applies Euler's form of fvc::ddtCorr to that difference. Without the file both codes
    # start from flux(U), and the correction is zero.
    for f in os.listdir(start):
        if f in ('U', 'p', 'k', 'omega', 'nut', 'nuTilda'):
            open(case + '/0/' + f, 'w').write(open(os.path.join(start, f)).read())
    write(case, 'system/controlDict', 'dictionary',
          'application     pimpleFoam;\nstartFrom       startTime;\nstartTime       0;\nstopAt          endTime;\nendTime         5e-4;\n'
          'deltaT          1e-4;\nwriteControl    timeStep;\nwriteInterval   1;\nwriteFormat     ascii;\nwritePrecision  17;\n'
          'writeCompression off;\ntimeFormat      general;\ntimePrecision   6;\nrunTimeModifiable false;\nadjustTimeStep  no;\n')
    s = open(case + '/system/fvSchemes').read()
    s = s.replace('default         steadyState;', 'default         backward;')
    open(case + '/system/fvSchemes', 'w').write(s)
    second = 'k|omega' if ('SST' in model or model == 'kOmegaSST') else 'nuTilda'
    tight = ('{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n        tolerance       1e-13;\n'
             '        relTol          0;\n        maxIter         2000;\n    }')
    ptight = ('{\n        solver          GAMG;\n        smoother        GaussSeidel;\n        tolerance       1e-14;\n'
              '        relTol          0;\n        maxIter         5000;\n    }')
    write(case, 'system/fvSolution', 'dictionary',
          'solvers\n{\n    p\n    %s\n    pFinal\n    %s\n    "(U|%s)"\n    %s\n    "(U|%s)Final"\n    %s\n}\n'
          'PIMPLE\n{\n    nOuterCorrectors 2;\n    nCorrectors     2;\n    nNonOrthogonalCorrectors 0;\n}\n'
          % (ptight, ptight, second, tight, second, tight))
    if model in ('kOmegaSST', 'SpalartAllmaras'):
        tp = 'simulationType  RAS;\nRAS\n{\n    RASModel        %s;\n    turbulence      on;\n    printCoeffs     on;\n}\n' % model
    else:
        tp = ('simulationType  LES;\nLES\n{\n    LESModel        %s;\n    turbulence      on;\n    printCoeffs     on;\n'
              '    delta           cubeRootVol;\n    cubeRootVolCoeffs\n    {\n        deltaCoeff      1;\n    }\n%s}\n'
              % (model, ('    %sCoeffs\n    {\n%s    }\n' % (model, ''.join('        %-15s %s;\n' % tuple(kv.split('='))
                                                                       for kv in coeffs.split()))) if coeffs else ''))
    write(case, 'constant/turbulenceProperties', 'dictionary', tp)


if sys.argv[1] == 'rans':
    rans(sys.argv[2], sys.argv[3])
else:
    transient(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else '')
PY

ofrun() { ( cd "$1" && nice -n 10 "$2" > log 2>&1 ); }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ); }
for fam in sst sa; do
    python3 "$W/step.py" rans "$W/rans-$fam" "$fam"
    ( cd "$W/rans-$fam" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
    ofrun "$W/rans-$fam" simpleFoam
    [ -d "$W/rans-$fam/50" ] || { echo "FAIL: OpenFOAM simpleFoam ($fam) did not reach iteration 50"; tail -3 "$W/rans-$fam/log"; exit 1; }
done
# case <arm> <family> <model> [coeffs]: the step's mesh and schemes, the transient case from the family's RANS start
# KEEP_PHI=1: carry simpleFoam's phi at iteration 50 over too (SRP)
mkcase()
{
    rm -rf "$W/$1"
    mkdir -p "$W/$1"
    cp -r "$W/rans-$2/constant" "$W/rans-$2/system" "$W/$1/"
    rm -f "$W/$1"/constant/polyMesh/.brae_*
    python3 "$W/step.py" transient "$W/$1" "$W/rans-$2/50" "$3" "${4:-}"
    if [ -n "${KEEP_PHI:-}" ]; then
        cp "$W/rans-$2/50/phi" "$W/$1/0/phi"
    fi
}
# SDK's coefficients, all of them read by OpenFOAM's kOmegaSSTDDES: the DES blend, the delay function and its kappa (which the
# omega wall function does not take)
SDK="CDESkom=0.9 CDESkeps=0.5 Cd1=15 kappa=0.38"
fields() { [ "$1" = sst ] && echo "p U k omega nut" || echo "p U nuTilda nut"; }

# PREMISE: OpenFOAM's DES differs from its RAS, and its DDES from its DES
for spec in "sst kOmegaSST kOmegaSSTDES kOmegaSSTDDES" "sa SpalartAllmaras SpalartAllmarasDES SpalartAllmarasDDES"; do
    set -- $spec
    fam=$1
    for m in "$2" "$3" "$4"; do
        mkcase "of-$m" "$fam" "$m"
        ofrun "$W/of-$m" pimpleFoam || { echo "FAIL: OpenFOAM $m"; tail -5 "$W/of-$m/log"; exit 1; }
    done
    for pair in "$3 $2" "$4 $3"; do
        set -- $pair
        d=$(python3 "$CMP" "$W/of-$2" "$W/of-$1" "$LAST" $(fields "$fam") | awk '/^MAX/{print $2, $4, $5}')
        r=${d%% *}
        python3 -c "import sys; sys.exit(0 if float('$r') > float('$PREMISE') else 1)" \
            && say "premise: OpenFOAM $1 differs from OpenFOAM $2 at $LAST ($d)" PASS \
            || say "premise: OpenFOAM $1 differs from OpenFOAM $2 at $LAST ($d)" FAIL
    done
done

judge()   # <arm> <family> <model> [coeffs]
{
    local a=$1 fam=$2 m=$3 mx mp
    if [ -n "${4:-}" ] || [ -n "${KEEP_PHI:-}" ]; then
        mkcase "$a-of" "$fam" "$m" "${4:-}"
        ofrun "$W/$a-of" pimpleFoam || { echo "FAIL: OpenFOAM $a"; tail -5 "$W/$a-of/log"; exit 1; }
    else
        cp -r "$W/of-$m" "$W/$a-of"
    fi
    mkcase "$a-br" "$fam" "$m" "${4:-}"
    brrun "$W/$a-br" || { say "$a: brae ran ($(grep -a -m1 -i -E 'error|refus' "$W/$a-br/log" | cut -c1-80))" FAIL; return; }
    python3 "$CMP" "$W/$a-of" "$W/$a-br" "$TIMES" $(fields "$fam") > "$W/$a.cmp"
    mx=$(awk '/^MAX/{print $2, $4, $5, $3}' "$W/$a.cmp")
    mp=$(python3 "$CMP" "$W/$a-of" "$W/$a-br" "$TIMES" phi | awk '/^MAX/{print $2}')
    if [ "$BOUND" = report ]; then
        say "$a ($m): worst entry at steps 1-5, $mx; phi $mp (report only)" INFO
        return
    fi
    python3 -c "import sys; sys.exit(0 if float('${mx%% *}') <= float('$BOUND') else 1)" \
        && say "$a ($m): every entry of $(fields "$fam") at steps 1-5 within $BOUND (worst $mx)" PASS \
        || say "$a ($m): every entry of $(fields "$fam") at steps 1-5 within $BOUND (worst $mx)" FAIL
    python3 -c "import sys; sys.exit(0 if float('$mp') <= float('$BOUND') else 1)" \
        && say "$a ($m): phi at steps 1-5 within $BOUND (worst $mp)" PASS \
        || say "$a ($m): phi at steps 1-5 within $BOUND (worst $mp)" FAIL
}
judge SDD sst kOmegaSSTDDES
judge SDE sst kOmegaSSTDES
judge SDK sst kOmegaSSTDDES "$SDK"
d=$(python3 "$CMP" "$W/SDD-of" "$W/SDK-of" "$LAST" $(fields sst) | awk '/^MAX/{print $2, $4, $5}')
python3 -c "import sys; sys.exit(0 if float('${d%% *}') > float('$PREMISE') else 1)" \
    && say "premise: OpenFOAM kOmegaSSTDDES with SDK's coefficients differs from its defaults at $LAST ($d)" PASS \
    || say "premise: OpenFOAM kOmegaSSTDDES with SDK's coefficients differs from its defaults at $LAST ($d)" FAIL
judge ADD sa SpalartAllmarasDDES
judge ADE sa SpalartAllmarasDES
# RAS on the same starts, the closures the DES models are built on
judge SRA sst kOmegaSST
judge ARA sa SpalartAllmaras
# backward's first step from a start whose phi is not flux(U)
KEEP_PHI=1 judge SRP sst kOmegaSST
d=$(python3 "$CMP" "$W/SRA-of" "$W/SRP-of" 0.0001 $(fields sst) | awk '/^MAX/{print $2, $4, $5}')
python3 -c "import sys; sys.exit(0 if float('${d%% *}') > float('$PREMISE') else 1)" \
    && say "premise: OpenFOAM from the start with simpleFoam's phi differs from without it at 0.0001 ($d)" PASS \
    || say "premise: OpenFOAM from the start with simpleFoam's phi differs from without it at 0.0001 ($d)" FAIL

# Each control must make ITS entry exceed the bound (`report` uses 1e-10 here).
CB=$BOUND
[ "$CB" = report ] && CB=1e-10
control()   # <env> <arm> <family> <model> <time> <field> <entry> [coeffs]
{
    local env=$1 arm=$2 fam=$3 m=$4 t=$5 f=$6 entry=$7 name=${1%%=*} r
    mkcase "$arm-$name" "$fam" "$m" "${8:-}"
    brrun "$W/$arm-$name" "$env" || { say "control $env: brae ran" FAIL; return; }
    r=$(python3 "$CMP" "$W/$arm-of" "$W/$arm-$name" "$t" "$f" | awk -v e="$entry" '$3 == e {print $10}')
    python3 -c "import sys; sys.exit(0 if '$r' and float('$r') > float('$CB') else 1)" \
        && say "control $env: $arm $f $entry at $t goes red (rel $r)" PASS \
        || say "control $env: $arm $f $entry at $t goes red (rel ${r:-none})" FAIL
}
control BRAE_CONTROL_DES_F2_SHIELDING=1 SDD sst kOmegaSSTDDES "$LAST" k internal
control BRAE_CONTROL_DES_AS_DDES=1 SDE sst kOmegaSSTDES "$LAST" k internal
control BRAE_CONTROL_DES_AS_DDES=1 ADE sa SpalartAllmarasDES "$LAST" nut internal
control BRAE_CONTROL_SST_DES_OLD_COEFFS=1 SDK sst kOmegaSSTDDES "$LAST" k internal "$SDK"
control BRAE_CONTROL_NUT_BOUNDARY_ONE_USE=1 ARA sa SpalartAllmaras 0.0001 nut walls
KEEP_PHI=1 control BRAE_CONTROL_BACKWARD_FIRST_NO_DDTCORR=1 SRP sst kOmegaSST 0.0001 p internal

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
