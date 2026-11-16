#!/usr/bin/env bash
# omegaWallFunction's G0 reads the STORED wall nut (the legacy single-GPU closures), against OpenFOAM v2412 simpleFoam.
# OpenFOAM's omegaWallFunction and epsilonWallFunction compute the wall cells' production from `turbModel.nut(patchi)`, the nut
# patch values the last correctNut stored (omegaWallFunctionFvPatchScalarField.C:199-200). The legacy closures
# (deviceKOmegaSSTCorrect, deviceKEpsilonCorrect) made that nut again from the current U and k, a Spalding Newton from zero at
# tolerance 0.01 whatever the patch said: on the official DrivAer's boundary conditions k 7.1e-02 from OpenFOAM after 3 iterations,
# 1.1e-03 with the stored value (brae-mgpu brief #18, JUPITER 2836461). They now read it (deviceStoredWallNut).
#
# CASE validation/pitzDailySST with both walls nutUSpaldingWallFunction at the official case's own `tolerance 1e-9` (OpenFOAM's
# correctNut converges to it from the stored nut; the made-again value stopped at 0.01 from zero), kqRWallFunction and
# omegaWallFunction as the fixture has them; every solve tight (relTol 0). Start: OpenFOAM simpleFoam's own iteration 50; ONE
# iteration in both codes, so nothing but this step's operations can differ (the other faces' nut, which the legacy driver keeps
# from the start file under a velocity-based wall function, equals what OpenFOAM's iteration 50 wrote).
# ARMS
#   B   brae (the legacy simpleFoam driver, OpenFOAM's sweep order) at 51 against OpenFOAM serial at 51
#   X   B with BRAE_CONTROL_G0_NUT_RECOMPUTED=1 (the made-again nut, as before): the CONTROL, asserted red
# CHECKS
#   premise  the control is injected
#   arm      U p k omega (internal and every patch) and nut's internal field, relative L2 within BOUND. nut's written NON-WALL
#            patches are measured and printed, not held: the legacy driver writes there a nut that correctNut has not evaluated
#            again (D4, open in public brae; brae-mgpu docs/deploy/OFFICIAL_CASE_AUDIT.md, public change 6), which on the first run
#            put the outlet 1.05e-02 from OpenFOAM's in BOTH arms and hid the control behind it
#   control  X at least 10 x B's distance and above BOUND
# BOUND 2e-11: one decade above B's worst on the first run (JUPITER 2838597: U 5.4e-13, p 1.3e-12, k 3.8e-13, omega 2.0e-12, nut
#   internal 4.2e-13). The control there: k 1.5e-04 internal (7.8e-04 on upperWall), nut internal 1.6e-05
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDailySST"
CHECK="$ROOT/tests/check_fields.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC/0" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-g0-stored-nut.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_G0_STORED_NUT_BOUND:-2e-11}
RATIO=10
START=50
END=51
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

# the case: Spalding on both walls at tolerance 1e-9, tight solves, written every iteration at 17 digits
D="$W/developed"
rm -rf "$D"
mkdir -p "$D"
cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$D/"
rm -f "$D"/constant/polyMesh/.brae_*
python3 - "$D" "$START" << 'PY'
import re, sys
d, start = sys.argv[1], int(sys.argv[2])
p = d + '/0/nut'
s = open(p).read()
for wall in ('upperWall', 'lowerWall'):
    s, n = re.subn(r'(\n    %s\n    \{)[^}]*\}' % wall,
                   r'\1\n        type            nutUSpaldingWallFunction;\n        tolerance       1e-9;\n        value           uniform 0;\n    }',
                   s, count=1)
    assert n == 1, wall
open(p, 'w').write(s)
c = d + '/system/controlDict'
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for key, val in (('startFrom', 'latestTime'), ('endTime', str(start)), ('writeControl', 'timeStep'), ('writeInterval', str(start)),
                 ('writeFormat', 'ascii'), ('writePrecision', '17'), ('writeCompression', 'off')):
    s, n = re.subn(r'(\n%s\s+)[^;]*;' % key, r'\g<1>' + val + ';', s, count=1)
    assert n == 1, key
open(c, 'w').write(s)
f = d + '/system/fvSolution'
s = open(f).read()
i = s.index('solvers')
j = s.index('\nSIMPLE')
s = s[:i] + ('solvers\n{\n    p { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0; maxIter 5000; }\n'
             '    "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; tolerance 1e-13; relTol 0; maxIter 2000; }\n}\n') + s[j:]
open(f, 'w').write(s)
PY
grep -q "nutUSpaldingWallFunction" "$D/0/nut" || { fail "staging: no Spalding wall"; exit 1; }
of "cd '$D' && simpleFoam > log 2>&1" < /dev/null
[ -f "$D/$START/k" ] || { fail "OpenFOAM simpleFoam did not write iteration $START: $(tail -n 3 "$D/log")"; echo "simple_g0_stored_nut_vs_openfoam: $passes passed, $fails failed"; exit 1; }
grep -q "tolerance" "$D/$START/nut" || { fail "premise: OpenFOAM's written nut lost the patch tolerance"; }

# stage <dir>: OpenFOAM's iteration 50 as the start, one iteration to 51
stage()
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$D/constant" "$D/system" "$D/$START" "$1/"
    rm -rf "$1/$START/uniform"
    sed -i -E "s/^(endTime\s+)\S+;/\1$END;/; s/^(writeInterval\s+)\S+;/\11;/" "$1/system/controlDict"
}
stage "$W/of"
of "cd '$W/of' && simpleFoam > log 2>&1" < /dev/null
[ -f "$W/of/$END/k" ] || fail "OpenFOAM did not write $END"
# brae <dir> [VAR=value ...]
brae()
{
    local d=$1
    shift
    (cd "$d" && env -u BRAE_CONTROL_G0_NUT_RECOMPUTED BRAE_SIMPLE_CASE_SOLVER=1 BRAE_TURBULENCE_CASE_SOLVER=1 "$@" \
        timeout 900 "$BRAE" -case . > log 2> err < /dev/null; echo $? > rc)
    [ "$(cat "$d/rc")" = 0 ] || { fail "$(basename "$d") exited $(cat "$d/rc"): $(grep -h 'ERROR\|error\|refused' "$d/log" "$d/err" | head -n 1 | cut -c1-200)"; return 1; }
}
stage "$W/b"
brae "$W/b"
stage "$W/x"
brae "$W/x" BRAE_CONTROL_G0_NUT_RECOMPUTED=1
if grep -q "CONTROL BRAE_CONTROL_G0_NUT_RECOMPUTED INJECTED" "$W/x/err" "$W/x/log"
then
    ok "premise: the control is injected"
else
    fail "premise: the control was not injected"
fi

dist()   # dist <run time dir>: the worst relative L2 of U p k omega (every patch) and of nut's internal field against OpenFOAM's
{        # at END; the checks' output in <dir>.cmp and <dir>.nut
    python3 "$CHECK" fields "$1" "$W/of/$END" 1e300 U p k omega > "$1.cmp" 2>&1
    python3 "$CHECK" fields "$1" "$W/of/$END" 1e300 nut > "$1.nut" 2>&1
    local w n
    w=$(sed -n 's/^worst \([^ ]*\).*/\1/p' "$1.cmp")
    n=$(awk '$2 == "nut" && $3 == "internal" {print $4}' "$1.nut")
    python3 -c "print('%.3e' % max(float('${w:-1e300}'), float('${n:-1e300}')))"
}
b=$(dist "$W/b/$END")
x=$(dist "$W/x/$END")
echo "     measured, not held: nut's written patches, B $(sed -n 's/^worst //p' "$W/b/$END.nut"), X $(sed -n 's/^worst //p' "$W/x/$END.nut") (D4, open in public)"
if python3 -c "import sys; sys.exit(0 if float('${b:-1e300}') <= $BOUND else 1)"
then
    ok "B brae against OpenFOAM at $END: worst $b (U p k omega $(sed -n 's/^worst [^ ]* //p' "$W/b/$END.cmp"); nut internal $(awk '$2 == "nut" && $3 == "internal" {print $4}' "$W/b/$END.nut")) within $BOUND"
else
    fail "B brae against OpenFOAM at $END: worst ${b:-none} above $BOUND"
    sed 's/^/     /' "$W/b/$END.cmp"
fi
if python3 -c "import sys; v = float('${x:-0}'); s = float('${b:-1e300}'); sys.exit(0 if v > $BOUND and v >= $RATIO * s else 1)"
then
    ok "control BRAE_CONTROL_G0_NUT_RECOMPUTED red: worst $x ($(sed -n 's/^worst [^ ]* //p' "$W/x/$END.cmp")), B $b"
else
    fail "control BRAE_CONTROL_G0_NUT_RECOMPUTED: worst ${x:-none}, not above $BOUND and $RATIO x B's ${b:-none}"
fi
echo "simple_g0_stored_nut_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 3 ]
