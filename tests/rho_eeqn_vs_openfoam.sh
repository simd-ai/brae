#!/usr/bin/env bash
# rhoSimpleFoam's EEqn.H against REAL OpenFOAM's own assembled ENERGY equation.
#
# THE ORACLE is tools/dumpPEqn: OpenFOAM's rhoSimpleFoam carrying a stage harness that writes, at SIMPLE
# iteration BRAE_DUMP_STAGE_ITER, the momentum equation's observable content --
#
#   stage_Ekp       the kinetic-energy field OpenFOAM's OWN he.name() branch produced
#   stage_he        he as assembled
#   stage_eD        EEqn.D()                              diag + boundary internalCoeffs
#   stage_eSrc      EEqn.source() + sum(boundaryCoeffs)   the full right-hand side
#   stage_alphaEff  turbulence->alphaEff()
#
# Dumping at iteration 1 means the state assembled from is the START-TIME field set, which brae can
# reconstruct exactly through createFields -- so the comparison isolates EEqn.H and carries no accumulated
# trajectory difference.
#
# stage_alphaEff IS INJECTED INTO brae rather than recomputed, for the same reason muEff is in the
# momentum gate: the compressible turbulence closure is a separate manifest component, and a number
# covering both cannot be attributed to either. A failure here means the ENERGY assembly is wrong.
#
# THE CONTROL, which is the reason this gate exists at all: EEqn.H branches on he.name(), with
# Ekp = 0.5|U|^2 + p/rho for `e` and K = 0.5|U|^2 for `h`. The binary builds the OTHER arm on purpose and
# requires it to disagree, both as a field against stage_Ekp and as an assembled source. On this fixture
# p/rho is ~2.9e5 against 0.5|U|^2 of order 10, so the wrong arm is wrong by four orders on the dominant
# term -- and still converges to a smooth, plausible, wrong temperature.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_eeqn_cpp"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

DUMP="$(command -v dumpPEqn || true)"
[ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpPEqn" ] \
    && DUMP="$FOAM_USER_APPBIN/dumpPEqn"
[ -n "$DUMP" ] || { echo "SKIP: dumpPEqn not built -- (cd tools/dumpPEqn && wmake)"; exit 77; }
[ -d "$W/case/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }

cp -r "$W/case/0.orig" "$W/case/0"

# THE INLET IS REPLACED BY A PLAIN fixedValue, and that is deliberate isolation, not convenience.
# sbMatched's inlet is flowRateInletVelocity, whose value OpenFOAM derives from the mass flow rate and
# whichever rho it is handed. brae disagrees with OpenFOAM there by ~2.4e-01 on this case -- a real finding,
# but one belonging to that boundary condition, which is its own manifest component. Left in place it would
# put a BC error inside a number that is supposed to say whether EEqn.H is assembled correctly, and a
# number covering two components cannot be attributed to either.
#
# A uniform non-zero value in all three components is used so the inlet still exercises boundaryCoeffs in
# every component rather than being trivially zero. It need not be physical: what is under test is the
# assembly of a matrix from a given boundary state, and both codes are given the SAME state.
python3 - "$W/case/0/U" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'(inlet\s*\{)[^}]*\}',
           r'\1\n        type            fixedValue;\n        value           uniform (1 2 3);\n    }',
           s, count=1)
open(p, 'w').write(s)
PYEOF
grep -q "fixedValue" "$W/case/0/U" || { echo "FAIL: could not neutralise the inlet BC"; exit 1; }

python3 - "$W/case" <<'PYEOF'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',    s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',       s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         1;',        s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   1;',        s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF

# One SIMPLE iteration, dumping the momentum stages from it.
( cd "$W/case" && BRAE_DUMP_STAGE_ITER=1 "$DUMP" > dump.log 2>&1 ) \
    || { echo "FAIL: dumpPEqn did not run"; tail -20 "$W/case/dump.log"; exit 1; }
for f in stage_Ekp stage_he stage_eD stage_eSrc stage_alphaEff stage_Upred; do
    [ -f "$W/case/1/$f" ] \
        || { echo "FAIL: dumpPEqn wrote no 1/$f"; tail -20 "$W/case/dump.log"; exit 1; }
done

# he's BOUNDARY TYPES are OpenFOAM's ENERGY types -- fixedEnergy, gradientEnergy, mixedEnergy -- which
# basicThermo::heBoundaryTypes() derives from T's. brae has no mapping for them yet (its own manifest
# component), and reading the staged file with T's types instead would put a TEMPERATURE where the
# coefficients want an ENERGY: the inlet's boundaryCoeffs would be -phi_b*1000 rather than -phi_b*4.19e5,
# which is how this gate first read a 19% source error that lived entirely on inlet-adjacent cells.
#
# So the staged file is rewritten into the brae-known equivalents, and ONLY where they are exactly
# equivalent at this state -- asserted, not assumed:
#     fixedEnergy               -> fixedValue     (always equivalent: it IS a fixedValue on he)
#     gradientEnergy  grad 0    -> zeroGradient   (equivalent only while the gradient is zero)
#     mixedEnergy     vf 0      -> zeroGradient   (equivalent only while valueFraction is zero)
# If a future fixture carries a non-zero gradient or valueFraction the greps below fail the gate rather
# than quietly comparing two different problems.
HE="$W/case/1/stage_he"
if grep -q "gradientEnergy" "$HE"; then
    grep -A3 "gradientEnergy" "$HE" | grep -q "gradient *uniform 0;" \
        || { echo "FAIL: gradientEnergy has a NON-ZERO gradient; the zeroGradient substitution is not valid here"; exit 1; }
fi
if grep -q "mixedEnergy" "$HE"; then
    grep -A3 "mixedEnergy" "$HE" | grep -q "valueFraction *uniform 0;" \
        || { echo "FAIL: mixedEnergy has a NON-ZERO valueFraction; the zeroGradient substitution is not valid here"; exit 1; }
fi
sed -i 's/type  *fixedEnergy;/type            fixedValue;/;
        s/type  *gradientEnergy;/type            zeroGradient;/;
        s/type  *mixedEnergy;/type            zeroGradient;/' "$HE"
grep -q "Energy;" "$HE" && { echo "FAIL: an energy BC type survived the rewrite"; exit 1; }

"$BIN" "$W/case" 0 1
