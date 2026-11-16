#!/usr/bin/env bash
# `Gauss LUST <grad>` takes the gradient it NAMES, cellLimited included, against OpenFOAM v2412 pimpleFoam. LUST is a linearUpwind
# (LUST.H: class LUST : public linearUpwind<Type>, built from the same scheme data), so its quarter of linearUpwind's correction uses
# the gradSchemes entry the statement names (linearUpwind.C: mesh.gradScheme(gradSchemeName_)). The parser resolved that name only
# after the word `linearUpwind` (scheme_parse.cuh, luGradLimit), so a LUST case under `grad(U) cellLimited Gauss linear 1` ran its
# correction on the UNLIMITED gradient, nothing said. Found by brae-mgpu's DDES gate on a 3-D step: brae against OpenFOAM serial U
# 3.7e-03 after four steps, 4.1e-02 after forty, 9.4e-14 with the named gradient taken.
#
# CASE validation/pitzDaily laminar from OpenFOAM simpleFoam laminar's own iteration 50 (fields varying everywhere), Euler, STEPS
# steps, PIMPLE 1-2-0, every solve tight, written at 17 digits; div(phi,U) `Gauss LUST <name>` (unbounded, the LES tutorials' form)
# and gradSchemes `grad(U) cellLimited Gauss linear 1` beside an unlimited `unlimitedGrad(U) Gauss linear`.
# ARMS (each brae against OpenFOAM serial on the same case)
#   L   `Gauss LUST grad(U)`: LUST's correction on the cellLimited gradient
#   N   `Gauss LUST unlimitedGrad(U)`: LUST's on the unlimited entry it names while grad(U)'s other readers (the stress term) stay
#       limited: the name is resolved, not grad(U)'s limiter taken whatever the name
#   X   the CONTROL: brae on N's case against OpenFOAM's L, which is what the parser did to L before (LUST unlimited, the stress
#       term limited); asserted red
# CHECKS
#   premise  OpenFOAM's L and N at least 100 x BOUND apart: the case's limiter reaches LUST's correction
#   arms     L and N: U p phi relative L2 per field and patch within BOUND
#   control  X at least 10 x L's distance and above BOUND
# BOUND 7e-12: one decade above the arms' worst on the first run (Arrhenius 5239887: L 2.2e-13, N 6.6e-13, p internal). The control
#   there 8.2e-04 (U internal), the premise (OpenFOAM's two runs) 8.2e-04
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDaily"
CHECK="$ROOT/tests/check_fields.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC/0" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-lust-grad.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
DT=5e-5
STEPS=5
T=$(python3 -c "print('%g' % ($STEPS * $DT))")
BOUND=${BRAE_LUST_GRAD_BOUND:-7e-12}
RATIO=10
fails=0
passes=0
fail()
{
    echo "FAIL $*"
    fails=$((fails + 1))
}
ok()
{
    passes=$((passes + 1))
    echo "ok   $*"
}
of()
{
    bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"
}

# OpenFOAM simpleFoam laminar to iteration 50: the start
D="$W/developed"
rm -rf "$D"
mkdir -p "$D"
cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$D/"
rm -f "$D"/constant/polyMesh/.brae_* "$D"/0/C "$D"/0/Cx "$D"/0/Cy "$D"/0/Cz "$D"/0/V "$D"/0/y
sed -i -E 's/^(endTime\s+)\S+;/\150;/; s/^(writeInterval\s+)\S+;/\150;/; s/^(writePrecision\s+)\S+;/\117;/' "$D/system/controlDict"
sed -i -E 's/^(simulationType\s+)\S+;/\1laminar;/' "$D/constant/turbulenceProperties"
python3 - "$D" << 'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
p = sys.argv[1] + '/system/fvSolution'
s = open(p).read()
s = re.sub(r'residualControl\s*\{.*?\n    \}', 'residualControl\n    {\n    }', s, flags=re.S)
open(p, 'w').write(s)
PY
of "cd '$D' && simpleFoam > log 2>&1" < /dev/null
[ -f "$D/50/U" ] || { fail "OpenFOAM simpleFoam laminar did not write iteration 50"; echo "lust_named_gradient_vs_openfoam: $passes passed, $fails failed"; exit 1; }

# stage <dir> <LUST's gradient name>: the transient case from the developed start
stage()
{
    rm -rf "$1"
    mkdir -p "$1/0"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    cp "$D"/50/* "$1/0/" 2> /dev/null
    rm -rf "$1/0/uniform"
    rm -f "$1"/constant/polyMesh/.brae_*
    sed -i -E 's/^(simulationType\s+)\S+;/\1laminar;/' "$1/constant/turbulenceProperties"
    python3 - "$1" "$DT" "$STEPS" "$2" << 'PY'
import re, sys
d, dt, steps, name = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
c = ('FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n'
     'application pimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime %.10g;\ndeltaT %.10g;\n'
     'writeControl timeStep;\nwriteInterval %d;\npurgeWrite 0;\nwriteFormat ascii;\nwritePrecision 17;\nwriteCompression off;\n'
     'timeFormat general;\ntimePrecision 6;\nrunTimeModifiable false;\nadjustTimeStep no;\n' % (steps * dt, dt, steps))
open(d + '/system/controlDict', 'w').write(c)
p = d + '/system/fvSchemes'
s = open(p).read()
s2 = re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default         Euler;\n}', s, flags=re.S)
assert s2 != s, 'the fixture has no ddtSchemes block'
s3 = re.sub(r'gradSchemes\s*\{[^}]*\}', 'gradSchemes\n{\n    default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n'
            '    unlimitedGrad(U) Gauss linear;\n}', s2, flags=re.S)
assert s3 != s2, 'the fixture has no gradSchemes block'
assert len(re.findall(r'div\(phi,U\)[^;]*;', s3)) == 1, 'expected one div(phi,U)'
s4 = re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      Gauss LUST %s;' % name, s3)
open(p, 'w').write(s4)
tight = '{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-13; relTol 0; maxIter 2000; }'
ptight = '{ solver GAMG; smoother GaussSeidel; tolerance 1e-14; relTol 0; maxIter 5000; }'
open(d + '/system/fvSolution', 'w').write(
    'FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n'
    'solvers\n{\n    p %s\n    pFinal %s\n    "(U|k|epsilon|omega)" %s\n    "(U|k|epsilon|omega)Final" %s\n}\n'
    'PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n    pRefCell 0;\n    pRefValue 0;\n}\n'
    % (ptight, ptight, tight, tight))
PY
}
stage "$W/ofL" "grad(U)"
of "cd '$W/ofL' && pimpleFoam > log 2>&1" < /dev/null || fail "OpenFOAM pimpleFoam (L)"
stage "$W/ofN" "unlimitedGrad(U)"
of "cd '$W/ofN' && pimpleFoam > log 2>&1" < /dev/null || fail "OpenFOAM pimpleFoam (N)"
# brae <dir>
brae()
{
    local d=$1
    (cd "$d" && env BRAE_SIMPLE_CASE_SOLVER=1 timeout 600 "$BRAE" -case . > log 2> err < /dev/null; echo $? > rc)
    [ "$(cat "$d/rc")" = 0 ] || { fail "$(basename "$d") exited $(cat "$d/rc"): $(grep -h 'ERROR\|error\|refused' "$d/log" "$d/err" | head -n 1 | cut -c1-200)"; return 1; }
}
stage "$W/bL" "grad(U)"
brae "$W/bL"
stage "$W/bN" "unlimitedGrad(U)"
brae "$W/bN"
dist()   # dist <run time dir> <reference time dir>: the worst relative L2 of U p phi; the check's output in <run>.<ref name>.cmp
{
    local out=$1.$(basename "$(dirname "$2")").cmp
    python3 "$CHECK" fields "$1" "$2" 1e300 U p phi > "$out" 2>&1
    sed -n 's/^worst \([^ ]*\).*/\1/p' "$out"
}
p=$(dist "$W/ofN/$T" "$W/ofL/$T")
if python3 -c "import sys; sys.exit(0 if float('${p:-0}') >= 100 * $BOUND else 1)"
then
    ok "premise: OpenFOAM's LUST on grad(U) cellLimited against its LUST on the unlimited entry: $p ($(sed -n 's/^worst [^ ]* //p' "$W/ofN/$T.ofL.cmp")): the limiter reaches LUST's correction"
else
    fail "premise: OpenFOAM's two LUST runs ${p:-none} apart, under 100 x $BOUND: the case cannot witness LUST's limiter"
fi
l=$(dist "$W/bL/$T" "$W/ofL/$T")
n=$(dist "$W/bN/$T" "$W/ofN/$T")
x=$(dist "$W/bN/$T" "$W/ofL/$T")
for arm in "L:$l:$W/bL/$T.ofL.cmp:LUST grad(U), grad(U) cellLimited" "N:$n:$W/bN/$T.ofN.cmp:LUST unlimitedGrad(U), grad(U) cellLimited"
do
    IFS=: read -r name v cmpf what <<< "$arm"
    if python3 -c "import sys; sys.exit(0 if float('${v:-1e300}') <= $BOUND else 1)"
    then
        ok "$name ($what) brae against OpenFOAM after $STEPS steps: worst $v ($(sed -n 's/^worst [^ ]* //p' "$cmpf")) within $BOUND"
    else
        fail "$name ($what) brae against OpenFOAM after $STEPS steps: worst ${v:-none} above $BOUND"
        sed 's/^/     /' "$cmpf"
    fi
done
if python3 -c "import sys; v = float('${x:-0}'); s = float('${l:-1e300}'); sys.exit(0 if v > $BOUND and v >= $RATIO * s else 1)"
then
    ok "control (LUST's correction unlimited, as the parser left it) red against OpenFOAM's L: worst $x ($(sed -n 's/^worst [^ ]* //p' "$W/bN/$T.ofL.cmp")), L $l"
else
    fail "control: worst ${x:-none}, not above $BOUND and $RATIO x L's ${l:-none}"
fi
echo "lust_named_gradient_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 4 ]
