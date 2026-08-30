#!/usr/bin/env bash
# rhoSimpleFoam's createFields.H against REAL OpenFOAM, on a committed compressible fixture.
#
# THE ORACLE. rhoSimpleFoam.C includes postProcess.H, so
#
#     rhoSimpleFoam -postProcess -func "writeObjects(phi,rho)"
#
# constructs createFields.H's entire field set and writes it WITHOUT solving. That is precisely the state
# this component owns, so the comparison is against OpenFOAM's own initial fields rather than against a
# hand-computed expectation, and no instrumented OpenFOAM build is needed.
#
# WHY THIS GATE EARNS ITS KEEP. compressibleCreatePhi.H interpolates the PRODUCT rho*U, while pEqn.H one
# file later interpolates the FACTORS separately for phiHbyA. Both forms are OpenFOAM's own and they
# differ on a non-uniform field, so the binary checks that brae reproduces the first AND that the second
# does not match -- a control, because a bound both forms pass would prove nothing.
#
# A DEVELOPED STATE, NOT THE SHIPPED ONE, and this is the whole reason the script runs OpenFOAM twice.
# sbMatched's 0.orig is uniform p, T and U. A uniform rho makes linearInterpolate(rho*U) algebraically
# identical to interpolate(rho)*interpolate(U), so a gate run from 0.orig passes whichever form is
# implemented. The first run develops the fields; the comparison is taken from a state where rho actually
# varies, and the binary ASSERTS that non-uniformity rather than trusting this comment.
#
# writePrecision is raised to 15 because the oracle is written as ASCII: at the fixture's shipped 6 digits
# the comparison bottoms out around 1e-06, which would hide any real disagreement underneath it.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_create_fields_cpp"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-25}

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

command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
[ -d "$W/case/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }

cp -r "$W/case/0.orig" "$W/case/0"
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
PYEOF

# 1. Develop the fields, so rho is not uniform.
( cd "$W/case" && rhoSimpleFoam > run.log 2>&1 ) || true
[ -d "$W/case/$ITERS" ] \
    || { echo "FAIL: OpenFOAM wrote no $ITERS/ to develop the fields"; tail -15 "$W/case/run.log"; exit 1; }

# 2. The COLD-START input: p, T and U from that state, with rho and phi deliberately absent.
mkdir -p "$W/cold"
for f in p T U; do
    cp "$W/case/$ITERS/$f" "$W/cold/$f" || exit 1
done
# The RAS fields come too. This gate is about rho and phi being COMPUTED rather than read -- that is what
# `cold` means here -- but the case is a kEpsilon case, and OpenFOAM's createFields.H constructs the
# turbulence model, which reads k, epsilon, nut and alphat. A directory without them is not a cold start,
# it is an incomplete case, and brae now says so rather than running a RAS case with no closure.
for f in k epsilon nut alphat; do
    [ -f "$W/case/$ITERS/$f" ] && cp "$W/case/$ITERS/$f" "$W/cold/$f"
done

# 3. The ORACLE: OpenFOAM's own createFields.H run on exactly those p, T and U. Everything except rho and
#    phi is carried over, so the turbulence model postProcess builds has its fields, and rho and phi are
#    the ones createFields.H/compressibleCreatePhi.H compute.
mkdir -p "$W/oracle/system" "$W/oracle/constant" "$W/oracle/$ITERS"
cp -r "$W/case/system/." "$W/oracle/system/"
cp -r "$W/case/constant/." "$W/oracle/constant/"
for f in "$W/case/$ITERS"/*; do
    b=$(basename "$f")
    case "$b" in
        rho|phi) continue ;;
    esac
    cp -r "$f" "$W/oracle/$ITERS/$b"
done
sed -i "s/^startFrom .*/startFrom       startTime;/; s/^startTime .*/startTime       $ITERS;/" \
    "$W/oracle/system/controlDict"
( cd "$W/oracle" && rhoSimpleFoam -postProcess -func "writeObjects(phi,rho)" > pp.log 2>&1 ) \
    || { echo "FAIL: OpenFOAM -postProcess did not run"; tail -15 "$W/oracle/pp.log"; exit 1; }
for f in phi rho; do
    [ -f "$W/oracle/$ITERS/$f" ] \
        || { echo "FAIL: OpenFOAM wrote no $ITERS/$f"; tail -15 "$W/oracle/pp.log"; exit 1; }
done

# 4. The refusal fixture: the same case with an energy variable rhoSimpleFoam does not transport. Only
#    constant/ is needed, since createFields reads thermophysicalProperties from there.
mkdir -p "$W/badenergy/constant"
cp -r "$W/case/constant/." "$W/badenergy/constant/"
sed -i 's/energy  *sensibleInternalEnergy;/energy          absoluteEnthalpy;/' \
    "$W/badenergy/constant/thermophysicalProperties"
grep -q "absoluteEnthalpy" "$W/badenergy/constant/thermophysicalProperties" \
    || { echo "FAIL: could not build the unsupported-energy fixture"; exit 1; }

# 5. The COEFFICIENT fixture: the same case declaring its own kEpsilonCoeffs. createFields runs
#    OpenFOAM's turbulence->validate() equivalent at construction -- correctNut writes nut = Cmu*k^2/eps
#    and alphat = rho*nut/Prt, and those are what the FIRST momentum and energy solves run on. brae used
#    to default-construct the coefficients and hardcode Prt = 1.0 there, discarding a Prt it had already
#    parsed. No shipped fixture declares either, so nothing could see it; this makes one that does.
mkdir -p "$W/coeffs/constant"
cp -r "$W/case/constant/." "$W/coeffs/constant/"
TP="$W/coeffs/constant/turbulenceProperties"
[ -f "$TP" ] || TP="$W/coeffs/constant/momentumTransport"
python3 - "$TP" <<'PYADD'
import re, sys
p = sys.argv[1]
s = open(p).read()
# Add a coeffs dict inside RAS { ... }, and a Prt beside it. Both differ from the defaults (0.09 / 1.0).
s = re.sub(r'(RASModel\s+\w+\s*;)', r'\1\n    kEpsilonCoeffs { Cmu 0.05; Prt 0.5; }', s, count=1)
open(p, 'w').write(s)
PYADD
grep -q "Cmu 0.05" "$TP" || { echo "FAIL: could not build the coefficient fixture"; exit 1; }

"$BIN" "$W/case" "$W/cold" "$W/oracle/$ITERS" "$W/badenergy" "$W/coeffs"
