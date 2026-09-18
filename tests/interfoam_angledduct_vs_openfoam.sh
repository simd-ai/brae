#!/usr/bin/env bash
# brae's interFoam with an fvOption against REAL OpenFOAM's, on RAS/angledDuct AS SHIPPED, field by field
# and solve by solve.
#
# THE METHOD is the other interFoam gates': exactly N identical FIXED steps at the tutorial's own
# deltaT (1e-3), both codes read at one instant. The mesh is the tutorial's own blockMesh, 28000 cells,
# with the `porosity` cellZone (8000 of them) that blockMesh writes.
#
# WHAT interFoam DOES WITH fvOptions (UEqn.H:9, :14, :31; pEqn.H:65) and what of it is ported:
#   == fvOptions(rho, U)         explicitPorositySource with DarcyForchheimer. The equation is
#                                FORCE-dimensioned, so the model takes the field named `rho` and, finding
#                                no `thermo:mu`, rho*nu with the field named `nu` -- the MIXTURE's laminar
#                                nu (DarcyForchheimer.C:203-218). d (2e8 -1000 -1000), f 0, in a frame
#                                turned 45 degrees about z: negative entries are multiples of the largest.
#   constrain(UEqn), correct(U)  nothing for a source; any option that needs them is refused by name
#
# WHAT ELSE THE CASE EXERCISES that no other gated interFoam case does: kEpsilon on a mesh
# non-orthogonal to 44.5 degrees under `Gauss linear corrected`; a `slip` wall TILTED 45 degrees; a
# flowRateInletVelocity given as a MASS flow rate; turbulentIntensityKineticEnergyInlet and
# turbulentMixingLengthDissipationRateInlet; and a start with no water in the domain at all.
#
# TWO ARMS. `inactive` is the case with the option `active no` in BOTH codes, held at the floor
# (U 4.3e-15); `porous` is the case as shipped (U 5.1e-11). Each is the other's control: the option
# moves OpenFOAM's own U by 23%. The four orders between the two arms are round-off, measured twice --
# tests/test_inter_angledduct_vs_openfoam.cu carries both measurements beside its bounds.
#
# WHAT THE GATE FOUND, and it was not the fvOption. interFoam's loop never refreshed a
# flowRateInletVelocity, which OpenFOAM recomputes at every momentum assembly from the rate and -- for a
# massFlowRate -- the mixture's rho on the patch. This case ships `massFlowRate constant 0.1` beside
# `value uniform (0 0 0)`, so brae's inlet stayed at zero: largest velocity 1.9e-04 m/s against
# OpenFOAM's 0.21, U 100% out, silently, and identically with the option off -- which is what put the
# defect outside the option. Both interFoam loops were affected; the device loop now refuses such an inlet.
#
# A THIRD ARM, STAGED: `porousWater` starts the duct full of water (its control `inactiveWater` too).
# As shipped the duct starts full of air and the front is 4000 steps from the porous zone, so rho there
# is exactly 1 for the whole gated run and the model's rho weighting cannot be seen. U 2.8e-12.
#
# BROKEN ONCE EACH, at t = 0.01 (U, p_rgh, p_rgh iteration counts equal):
#   the inlet never refreshed (what was found)            porous       1.0e+00, 1.0e+00, 12 of 30
#   ...refreshed with the face CELL's rho, not the patch's porous       6.5e+06, 9.3e+07, 13 of 30
#   ...refreshed as a volumetric rate                     porous       2.9e+02, 5.6e+04, 10 of 30
#   the option not applied (the control)                  porous       2.3e-01, 7.9e-01
#   mu as rho*nuEff, not rho*nu                           porous       5.3e-02, 2.1e+01, 11 of 30
#                                                         porousWater  1.8e-01, 4.0e+02, 13 of 30
#   the kinematic form, no rho                            porous       NOTHING, to the last digit
#                                                         porousWater  4.9e-01, 9.9e-01, 14 of 30
# NOT DISCRIMINATED here: the Forchheimer half (f is zero). constant/fvOptions before system/fvOptions
# is held by tests/interfoam_refusals.sh's `fvoptions_constant_wins`.
#
# NOT CLAIMED, each refused by name: every other fvOption type, explicitPorositySource's fixedCoeff
# model, an fvOption under a moving mesh or beside an MRF zone, and the device loop; the scalarTransport
# function object `s`, which brae does not run.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_angledduct_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/angledDuct"
STEPS=${STEPS:-10}
DT=${DT:-1e-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/angledDuct tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the tutorial, apply the profile, fix the step, mesh it as Allrun does, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "type  *explicitPorositySource;" "$C/constant/fvOptions" \
        || { echo "FAIL: the tutorial no longer carries an explicitPorositySource"; return 1; }
    grep -q "type  *DarcyForchheimer;" "$C/constant/fvOptions" \
        || { echo "FAIL: the tutorial's porosity model is no longer DarcyForchheimer"; return 1; }
    if [ "$profile" = inactive ] || [ "$profile" = inactiveWater ]; then
        sed -i 's/type  *explicitPorositySource;/&\n    active          no;/' "$C/constant/fvOptions"
        grep -q "active  *no;" "$C/constant/fvOptions" || { echo "FAIL: the option was not switched off"; return 1; }
    fi
    if [ "$profile" = porousWater ] || [ "$profile" = inactiveWater ]; then
        # THE DUCT STARTS FULL OF WATER. As shipped it starts full of air and the water front is 4000
        # steps from the porous zone, so rho there is exactly 1 for the whole gated run and the model's
        # rho weighting -- mu = rho*nu -- cannot be seen: measured, the kinematic form (no rho at all)
        # changes NO digit of the shipped run. With water in the zone it is rho = 1000.
        sed -i 's/^internalField  *uniform 0;/internalField   uniform 1;/' "$C/0/alpha.water"
        grep -q "^internalField  *uniform 1;" "$C/0/alpha.water" || { echo "FAIL: alpha was not staged"; return 1; }
    fi

    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
# the function objects write nothing the gate reads, and `s` is a scalarTransport brae does not run
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in inactive porous inactiveWater porousWater; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_angledduct_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path
grep -q "Creating finite volume options from \"constant/fvOptions\"" "$W/porous/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log does not create the options from constant/fvOptions"; exit 1; }
grep -q "Porosity region porosity1" "$W/porous/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log does not build the porosity region"; exit 1; }

"$BIN" "$W/inactive" "$W/inactive/0" "$W/inactive/$END" "$STEPS" "$W/inactive/log.interFoam" \
       "$W/porous/$END" inactive || rc=1
"$BIN" "$W/porous" "$W/porous/0" "$W/porous/$END" "$STEPS" "$W/porous/log.interFoam" \
       "$W/inactive/$END" porous || rc=1
"$BIN" "$W/porousWater" "$W/porousWater/0" "$W/porousWater/$END" "$STEPS" "$W/porousWater/log.interFoam" \
       "$W/inactiveWater/$END" porousWater || rc=1

echo "interfoam_angledduct_vs_openfoam: rc $rc"
exit $rc
