#!/usr/bin/env bash
# rhoSimpleFoam END TO END: brae's _cpp driver against real OpenFOAM, whole case, same mesh, same
# dictionaries. This is the gate PORT.md requires BEFORE any .cu is written for this solver.
#
# TURBULENT: the fixture's own kEpsilon, as it ships. The closure is gated separately and at machine
# precision by tests/rho_kepsilon_vs_openfoam.sh, against OpenFOAM's own turbulence->correct(); this gate
# is the whole case, both codes running their own outer iteration from the same start.
#
# RUN TO CONVERGENCE, not to a tutorial's endTime, and the difference is the whole reason ITERS is 400.
# At 100 iterations -- residuals U 8.9e-06 -- the two codes disagree by U 7.9e-04, which is trajectory
# drift and not a difference in the equations: at 400, with residuals at ~1e-09, the same comparison reads
# U 1.08e-04. Comparing at an arbitrary iteration count compares two trajectories; comparing at
# convergence compares two answers.
#
# THE FIXTURE'S OWN INLET, as it ships -- flowRateInletVelocity, not a substitute. This gate used to
# replace it with a plain `fixedValue`, which quietly meant every rhoSimpleFoam gate was validating an
# essentially incompressible case: |U| ~ 3.7 m/s against the 523 m/s the case actually asks for. Two real
# defects were invisible at that speed and are gated here now -- the inlet holding its `rhoInlet` seed
# instead of the live rho, and transportAlpha being handed a temperature where it takes a viscosity.
#
# The gate asserts the boundary condition's DEFINING PROPERTY as well as the fields: the inlet delivers
# the prescribed massFlowRate, to the convergence level here and to 2.2e-15 on a single assembly in
# rho_ueqn_vs_openfoam.
#
# `consistent yes` and `transonic yes` are left as the fixture ships them, so the path exercised end to
# end is pcEqn.H's transonic branch -- the most involved of the four, and the one carrying both the
# SIMPLEC corrections and the convective pressure term.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cpp"
# The fixture, overridable so the same OpenFOAM-oracle machinery can be pointed at another compressible
# case. sbMatched is the default and is what the registered gate runs; validation/rhoTP is the only other
# compressible fixture carrying boundary conditions this driver has to recompute per iteration
# (totalPressure + pressureInletOutletVelocity).
SRC="${CASE:-$ROOT/validation/sbMatched}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-400}

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
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }



ITERS="$ITERS" python3 - "$W/case" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
n = os.environ['ITERS']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
# Rewrite each entry up to its OWN semicolon, not to end of line. An OpenFOAM dictionary may put
# several entries on one line -- validation/rhoTP's controlDict is three lines for twelve entries --
# and the line-anchored form silently swallowed everything after the key it matched, leaving a dict
# whose writeInterval and writeFormat had been deleted. OpenFOAM then failed on the FIRST missing
# entry it happened to want, which read as `OpenFOAM did not run` rather than as a broken rewrite.
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', n),
             ('writeInterval', n), ('writeControl', 'timeStep')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)

# residualControl is removed so BOTH codes run exactly ITERS iterations: the comparison is trajectory for
# trajectory, and a solver that stopped early would be compared at a different point.
f = os.path.join(d, 'system/fvSolution')
s = open(f).read()
# `[^{}]*` rather than `.*?\n\s*\}`: residualControl has no nested braces, and the old pattern required
# a NEWLINE before the closing brace, so a fixture that puts the whole block on one line -- rhoBox and
# rhoTP both do -- kept its residualControl and OpenFOAM stopped early on convergence instead of running
# the requested count. The script then failed looking for a time directory OpenFOAM never wrote.
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PYEOF

( cd "$W/case" && rhoSimpleFoam > run.log 2>&1 ) \
    || { echo "FAIL: OpenFOAM's rhoSimpleFoam did not run"; tail -25 "$W/case/run.log"; exit 1; }
[ -d "$W/case/$ITERS" ] \
    || { echo "FAIL: OpenFOAM wrote no $ITERS/"; tail -25 "$W/case/run.log"; exit 1; }
grep -q "^End" "$W/case/run.log" \
    || { echo "FAIL: OpenFOAM did not finish"; tail -25 "$W/case/run.log"; exit 1; }

# An UNPORTED model, so the binary can assert the refusal is real. This was kOmegaSST until kOmegaSST was
# ported for the compressible lineage; the gate caught its own negative control going stale, which is what
# a negative control is for. LaunderSharmaKE is a compressible RAS model brae does not have, and it must be
# refused BY NAME rather than run as kEpsilon or as laminar.
# Both refusal fixtures need a RAS case to mutate. A LAMINAR fixture (validation/rhoTP) has no RASModel
# and no nut file, so they are built only where they mean something and the binary is handed fewer
# arguments; it already treats argv[4] and argv[5] as optional and prints SKIP for each. Guarding on the
# fixture rather than failing keeps the same script usable as an OpenFOAM oracle for a laminar case.
UNPORTED=""
UNPORTEDNUT=""
if grep -q "RASModel" "$SRC/constant/turbulenceProperties" 2>/dev/null; then
mkdir -p "$W/unported/constant"
cp -r "$SRC/constant/." "$W/unported/constant/"
sed -i 's/RASModel *kEpsilon;/RASModel        LaunderSharmaKE;/' "$W/unported/constant/turbulenceProperties"
grep -q "LaunderSharmaKE" "$W/unported/constant/turbulenceProperties" \
    || { echo "FAIL: could not build the unported-model fixture"; exit 1; }
UNPORTED="$W/unported"

# ...and an unported NUT WALL FUNCTION, in a time directory of its own. The closure computes
# nutkWallFunction unconditionally for both the wall nut and the near-wall production, so every other
# member of the family must be refused BY NAME rather than run under nutk's. sbMatched's walls ship
# nutkWallFunction, which is also what makes the negative control in the binary meaningful.
mkdir -p "$W/unportednut"
cp "$W/case/0/"* "$W/unportednut/" 2>/dev/null || true
sed -i 's/nutkWallFunction/nutUSpaldingWallFunction/' "$W/unportednut/nut"
grep -q "nutUSpaldingWallFunction" "$W/unportednut/nut" \
    || { echo "FAIL: could not build the unported-nut fixture"; exit 1; }
UNPORTEDNUT="$W/unportednut"

# ...and the ATM member of the family, in a fixture of its own. atmNutkWallFunction does not start
# with "nut", so a one-prefix guard let it fall through to CalculatedPatchField and run as the smooth
# nutk with z0 unplumbed -- the refusal existed and could not fire. This arm FAILED before the
# two-prefix guard landed, which is its fail-proof.
mkdir -p "$W/unportedatm"
cp "$W/case/0/"* "$W/unportedatm/" 2>/dev/null || true
sed -i 's/nutkWallFunction/atmNutkWallFunction/' "$W/unportedatm/nut"
grep -q "atmNutkWallFunction" "$W/unportedatm/nut" \
    || { echo "FAIL: could not build the unported-atm fixture"; exit 1; }
UNPORTEDATM="$W/unportedatm"
fi

# ...and a LIQUID thermo, which the parser ACCEPTS (the legacy binary carries the NSRDS path) and the
# mirror createFields must therefore refuse itself -- everything downstream of it evaluates
# perfectGas + hConst directly, and nothing else checks the model. Before the guard this ran a gas
# equation of state against a liquid's (unset) coefficients.
LIQUIDTHERMO="$W/liquidthermo"
mkdir -p "$LIQUIDTHERMO/constant" "$LIQUIDTHERMO/system"
cp -r "$SRC/constant/." "$LIQUIDTHERMO/constant/"
cp -r "$SRC/system/." "$LIQUIDTHERMO/system/"
cat > "$LIQUIDTHERMO/constant/thermophysicalProperties" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object thermophysicalProperties; }
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    properties      liquid;
    energy          sensibleInternalEnergy;
}
mixture
{
    H2O;
}
EOF

"$BIN" "$W/case" 0 "$ITERS" $UNPORTED $UNPORTEDNUT $UNPORTEDATM $LIQUIDTHERMO
