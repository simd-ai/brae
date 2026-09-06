#!/usr/bin/env bash
# The COMPRESSIBLE kEpsilon closure against real OpenFOAM's own turbulence->correct().
#
# tools/dumpPEqn brackets that call and writes every field it reads and every field it writes, so brae's
# closure can be run on OpenFOAM's exact inputs and its outputs compared. The model is treated as a black
# box because it IS one from the solver's side: k and epsilon are assembled inside OpenFOAM's turbulence
# library and never surface where the solver could instrument them.
#
# ONE ITERATION is enough and more would be worse: the comparison is of a single correct() on a single
# state, so a disagreement is the closure and cannot be a trajectory that drifted.
#
# THE INLET IS NEUTRALISED, as in the other rhoSimpleFoam gates: sbMatched's flowRateInletVelocity
# disagrees with OpenFOAM by ~2.4e-01 (PORT.md) and would dominate a whole-field comparison meant to be
# about the solver.
#
# `consistent yes` and `transonic yes` are left as the fixture ships them, so the path exercised end to
# end is pcEqn.H's transonic branch -- the most involved of the four, and the one carrying both the
# SIMPLEC corrections and the convective pressure term.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_kepsilon_cpp"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-1}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*
[ -d "$W/case/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }
cp -r "$W/case/0.orig" "$W/case/0"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
DUMP="$(command -v dumpPEqn || true)"
[ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpPEqn" ] \
    && DUMP="$FOAM_USER_APPBIN/dumpPEqn"
[ -n "$DUMP" ] || { echo "SKIP: dumpPEqn not built -- (cd tools/dumpPEqn && wmake)"; exit 77; }

python3 - "$W/case/0/U" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'(inlet\s*\{)[^}]*\}',
           r'\1\n        type            fixedValue;\n        value           uniform (1 2 3);\n    }',
           s, count=1)
open(p, 'w').write(s)
PYEOF

grep -q "kEpsilon" "$W/case/constant/turbulenceProperties" \
    || { echo "FAIL: the fixture is no longer kEpsilon; this gate is about the compressible closure"; exit 1; }
for fld in k epsilon nut alphat; do
    [ -f "$W/case/0/$fld" ] || { echo "FAIL: the RAS fixture is missing 0/$fld"; exit 1; }
done

ITERS="$ITERS" python3 - "$W/case" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
n = os.environ['ITERS']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',    s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',       s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         %s;' % n,   s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   %s;' % n,   s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;', s, flags=re.M)
open(c, 'w').write(s)

# residualControl is removed so BOTH codes run exactly ITERS iterations: the comparison is trajectory for
# trajectory, and a solver that stopped early would be compared at a different point.
f = os.path.join(d, 'system/fvSolution')
s = open(f).read()
s = re.sub(r'residualControl\s*\{.*?\n\s*\}', 'residualControl\n    {\n    }', s, flags=re.S)
# The turbulence solvers are gone with the model; leaving their entries is harmless but noisy.
open(f, 'w').write(s)
PYEOF

# The INSTRUMENTED kEpsilon: tools/dumpKEpsilon is OpenFOAM's own model with writes added and its
# equations untouched, selected here so divU, GbyNu, G and both assembled systems come out. Without it the
# comparison can only see the closure's outputs; with it, a disagreement names the term.
if [ -f "${FOAM_USER_LIBBIN:-}/libdumpKEpsilon.so" ]; then
    sed -i 's/RASModel  *kEpsilon;/RASModel            kEpsilonDump;/' "$W/case/constant/turbulenceProperties"
    grep -q "libdumpKEpsilon" "$W/case/system/controlDict" \
        || printf '\nlibs ("libdumpKEpsilon.so");\n' >> "$W/case/system/controlDict"
    grep -q "kEpsilonDump" "$W/case/constant/turbulenceProperties" \
        || { echo "FAIL: could not select the instrumented model"; exit 1; }
else
    echo "  note: libdumpKEpsilon.so not built -- (cd tools/dumpKEpsilon && wmake libso);"
    echo "        the term-by-term comparison will be skipped, outputs still compared"
fi

( cd "$W/case" && BRAE_DUMP_STAGE_ITER="$ITERS" "$DUMP" > dump.log 2>&1 ) \
    || { echo "FAIL: dumpPEqn did not run"; tail -25 "$W/case/dump.log"; exit 1; }
[ -d "$W/case/$ITERS" ] \
    || { echo "FAIL: OpenFOAM wrote no $ITERS/"; tail -25 "$W/case/dump.log"; exit 1; }

for fld in stage_kIn stage_epsIn stage_nutIn stage_alphatIn stage_Uturb stage_phiTurb \
           stage_rhoTurb stage_muTurb stage_kOut stage_epsOut stage_nutOut stage_alphatOut; do
    [ -f "$W/case/$ITERS/$fld" ] \
        || { echo "FAIL: dumpPEqn wrote no $ITERS/$fld"; tail -20 "$W/case/dump.log"; exit 1; }
done

"$BIN" "$W/case" "$ITERS"
