#!/usr/bin/env bash
# pimpleFoam's SIMPLEC terms on the second pressure corrector (public brae's PIMPLE step), against OpenFOAM v2412 pimpleFoam.
# pEqn.H (consistent yes) corrects phiHbyA by interpolate(rAtU - rAU)*snGrad(p)*magSf and HbyA by (rAU - rAtU)*grad(p) from the p
# ENTERING the corrector. The legacy driver took grad(p) from the momentum predictor, which is that p on the first corrector
# only: from the second on the previous corrector has solved p. Found by brae-mgpu's oracle gate (brief #18): U 6.3e-02 from
# OpenFOAM after one step, 7.2e-14 with the gradient refreshed. The SIMPLEC block now takes the current p's gradient.
#
# CASE validation/pitzDaily laminar from OpenFOAM simpleFoam laminar's own iteration 50 (fields varying everywhere), Euler, ONE
# step, nOuterCorrectors 1, nCorrectors 2, consistent yes, every solve tight, written at 17 digits. The step is small enough that
# OpenFOAM's Courant number stays below 3: pimpleFoam's SIMPLEC floor, rAtU = 1/max(1/rAU - H1, 0.1/rAU), binds only where a row's
# off-diagonals exceed nine times the rest of its diagonal (a Courant or diffusion number of order ten), and public brae's
# PIMPLE step does not carry it yet, so this gate holds the gradient alone.
# ARMS
#   B   brae (the legacy driver's PIMPLE step) against OpenFOAM serial
#   X   B with BRAE_CONTROL_SIMPLEC_PREDICTOR_GRAD=1 (the predictor's grad(p), as before): the CONTROL, asserted red
# CHECKS
#   premise  OpenFOAM's Courant number max between 0.5 and 3; the control injected
#   arm      U p phi relative L2 per field and patch within BOUND
#   control  X at least 10 x B's distance and above BOUND
# BOUND 3e-12: one decade above B's worst on the first run (JUPITER 2838597: 2.1e-13, p internal; OpenFOAM's Courant number max
#   2.07). The control there: 3.3e-03 (U internal)
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
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-simplec-grad.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
DT=1e-4
T=$(python3 -c "print('%g' % $DT)")
BOUND=${BRAE_SIMPLEC_GRAD_BOUND:-3e-12}
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
[ -f "$D/50/U" ] || { fail "OpenFOAM simpleFoam laminar did not write iteration 50"; echo "pimple_simplec_corrector_grad_vs_openfoam: $passes passed, $fails failed"; exit 1; }

# stage <dir>: the transient case from the developed start, consistent yes
stage()
{
    rm -rf "$1"
    mkdir -p "$1/0"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    cp "$D"/50/* "$1/0/" 2> /dev/null
    rm -rf "$1/0/uniform"
    rm -f "$1"/constant/polyMesh/.brae_*
    sed -i -E 's/^(simulationType\s+)\S+;/\1laminar;/' "$1/constant/turbulenceProperties"
    python3 - "$1" "$DT" << 'PY'
import re, sys
d, dt = sys.argv[1], float(sys.argv[2])
c = ('FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n'
     'application pimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime %.10g;\ndeltaT %.10g;\n'
     'writeControl timeStep;\nwriteInterval 1;\npurgeWrite 0;\nwriteFormat ascii;\nwritePrecision 17;\nwriteCompression off;\n'
     'timeFormat general;\ntimePrecision 6;\nrunTimeModifiable false;\nadjustTimeStep no;\n' % (dt, dt))
open(d + '/system/controlDict', 'w').write(c)
s = open(d + '/system/fvSchemes').read()
s2 = re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default         Euler;\n}', s, flags=re.S)
assert s2 != s, 'the fixture has no ddtSchemes block'
open(d + '/system/fvSchemes', 'w').write(s2)
tight = '{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-13; relTol 0; maxIter 2000; }'
ptight = '{ solver GAMG; smoother GaussSeidel; tolerance 1e-14; relTol 0; maxIter 5000; }'
open(d + '/system/fvSolution', 'w').write(
    'FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n'
    'solvers\n{\n    p %s\n    pFinal %s\n    "(U|k|epsilon|omega)" %s\n    "(U|k|epsilon|omega)Final" %s\n}\n'
    'PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n    consistent yes;\n    pRefCell 0;\n    pRefValue 0;\n}\n'
    % (ptight, ptight, tight, tight))
PY
}
stage "$W/of"
of "cd '$W/of' && pimpleFoam > log 2>&1" < /dev/null || fail "OpenFOAM pimpleFoam"
# brae <dir> [VAR=value ...]
brae()
{
    local d=$1
    shift
    (cd "$d" && env -u BRAE_CONTROL_SIMPLEC_PREDICTOR_GRAD BRAE_SIMPLE_CASE_SOLVER=1 "$@" timeout 600 "$BRAE" -case . > log 2> err < /dev/null; echo $? > rc)
    [ "$(cat "$d/rc")" = 0 ] || { fail "$(basename "$d") exited $(cat "$d/rc"): $(grep -h 'ERROR\|error\|refused' "$d/log" "$d/err" | head -n 1 | cut -c1-200)"; return 1; }
}
stage "$W/b"
brae "$W/b"
stage "$W/x"
brae "$W/x" BRAE_CONTROL_SIMPLEC_PREDICTOR_GRAD=1

co=$(grep -a -m1 '^Courant Number mean' "$W/of/log" | awk '{print $NF}')
if python3 -c "import sys; c = float('${co:-0}'); sys.exit(0 if 0.5 < c < 3 else 1)"
then
    ok "premise: OpenFOAM's Courant number max $co, below the SIMPLEC floor's reach (about 9) and the step not trivial"
else
    fail "premise: OpenFOAM's Courant number max '${co}', not between 0.5 and 3"
fi
if grep -q "CONTROL BRAE_CONTROL_SIMPLEC_PREDICTOR_GRAD INJECTED" "$W/x/err" "$W/x/log"
then
    ok "premise: the control is injected"
else
    fail "premise: the control was not injected"
fi
dist()   # dist <run time dir>: the worst relative L2 of U p phi against OpenFOAM's; the check's output in <dir>.cmp
{
    python3 "$CHECK" fields "$1" "$W/of/$T" 1e300 U p phi > "$1.cmp" 2>&1
    sed -n 's/^worst \([^ ]*\).*/\1/p' "$1.cmp"
}
b=$(dist "$W/b/$T")
x=$(dist "$W/x/$T")
if python3 -c "import sys; sys.exit(0 if float('${b:-1e300}') <= $BOUND else 1)"
then
    ok "B brae against OpenFOAM after one step: worst $b ($(sed -n 's/^worst [^ ]* //p' "$W/b/$T.cmp")) within $BOUND"
else
    fail "B brae against OpenFOAM after one step: worst ${b:-none} above $BOUND"
    sed 's/^/     /' "$W/b/$T.cmp"
fi
if python3 -c "import sys; v = float('${x:-0}'); s = float('${b:-1e300}'); sys.exit(0 if v > $BOUND and v >= $RATIO * s else 1)"
then
    ok "control BRAE_CONTROL_SIMPLEC_PREDICTOR_GRAD red: worst $x ($(sed -n 's/^worst [^ ]* //p' "$W/x/$T.cmp")), B $b"
else
    fail "control BRAE_CONTROL_SIMPLEC_PREDICTOR_GRAD: worst ${x:-none}, not above $BOUND and $RATIO x B's ${b:-none}"
fi
echo "pimple_simplec_corrector_grad_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 4 ]
