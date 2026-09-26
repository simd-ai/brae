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
# THE DEVICE LOOP RUNS ALL THREE ARMS NOW, at the same bounds: inactive alpha 6.1e-16, p_rgh 4.3e-15,
# U 7.6e-15; porous (as shipped) 1.1e-14, 1.3e-11, 1.8e-11; porousWater 5.9e-13, 2.9e-12, 2.6e-12 -- and
# all 30 of its own p_rgh solves take OpenFOAM's iteration counts in every arm. Getting there took one
# module and FOUR defects, each localised by running the HOST closure inside the device loop to say
# whether the loop or the closure owned it, and each fixed by transcribing the host arm's own lines:
#   fvOptions' explicitPorositySource on the device (the module): the full transformed D and F tensors,
#   because this case rotates e1 by 45 degrees, and mu as a FIELD (rho*nu_laminar, three orders across
#   the interface). Dropped: p_rgh 7.9e-01, U 2.3e-01. Handed the kinematic nu instead of rho*nu: the
#   same, since the case's F is zero and its nuLaminar scalar is not the mixture's.
#   the massFlowRate inlet never rebuilt on the device (fvMatrix.C:396 runs updateCoeffs at every
#   assembly; inter_driver_cpp.cu:715-720 is the host's): U 1.0007e+00, a frozen (0 0 0) inlet.
#   the SLIP wall's symmetry refValue never refreshed against this iteration's cells -- the third step of
#   rhoSimpleFoam's own pre-assembly sequence (rhoUEqn.cuh:76-78), missing from interFoam's: U 2.7e-02
#   with the host closure and the porosity off, where the host loop is 4.1e-15.
#   the TURBULENT INLETS frozen at the file's `value`: this inlet carries both
#   turbulentIntensityKineticEnergyInlet and turbulentMixingLengthDissipationRateInlet, which OpenFOAM
#   recomputes from U at every updateCoeffs, and interFoam never handed the device closure their masks
#   (rhoCreateFields.cu:478-500 builds them): U 1.3e+00, k 1.1e+00, epsilon 4.7e+00.
#   the WALL VELOCITY kept from the closure's build: the wall-function production reads
#   (U_wall - U_cell)*deltaCoeffs and the host reads U_wall live (kEpsilon_cpp.cu:401). Exact for every
#   noSlip wall, wrong for a slip one: epsilon 1.5e-02, with its worst cells on `porosityWall`.
# THE LAST TWO ARE NOT interFoam's ALONE -- they are the shared device closure's wiring and the shared
# wall data. Every other gated case has only noSlip walls and un-recomputed inlets, which is why this
# fixture is the one that shows them.
#
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
    # `splitGrad`: grad(k) and grad(epsilon) name DIFFERENT schemes. `fvc::grad(vf)` resolves
    # `grad(<vf>)` by the FIELD`s name, so the two are two different gradients and OpenFOAM computes each;
    # brae`s closures carried ONE pair of flags and the reader REFUSED a mismatch.
    #
    # THIS TUTORIAL IS THE FIXTURE because of its MESH and its LAPLACIAN. The turbulence gradient is read
    # on three paths: the corrected laplacian`s deferred non-orthogonal correction, limitedLinear`s limiter
    # gradient, and (under kOmegaSST) CDkOmega. angledDuct ships `corrected` and its mesh is 44.5 deg
    # non-orthogonal, so the first path is LIVE. RAS/damBreak ships `Gauss linear orthogonal` and `upwind`,
    # so all three paths are dead there and the same profile reads EXACTLY ZERO on every field at every
    # step -- measured, 0 of 2268 cells. Shearing damBreak does not help: `corrected()` is never requested.
    #
    # grad(k) IS WRITTEN EXPLICITLY, equal to the case`s own `default`, so the profile changes exactly one
    # thing. MEASURED, OpenFOAM against OpenFOAM at this gate`s own end time: epsilon 6.795e-03 over
    # 27,870 of 28,000 cells, nut 8.547e-03, k 5.024e-03, U 6.451e-04; and the MIRROR direction (k taking
    # epsilon`s scheme) k 4.485e-02, epsilon 6.081e-02. The control is the `porous` run, which is this case
    # with one `default` for both.
    if [ "$profile" = splitGrad ]; then
        python3 - "$C" <<'GRADEOF' || { echo "FAIL: the splitGrad profile was not staged"; return 1; }
import re, sys
q = sys.argv[1] + "/system/fvSchemes"
t = open(q).read()
m = re.search(r"gradSchemes\s*\{([^}]*)\}", t)
assert m, "no gradSchemes block"
assert "grad(k)" not in m.group(1), "the tutorial names grad(k) already"
t = t[:m.end(1)] + "    grad(k)         Gauss linear;\n    grad(epsilon)   leastSquares;\n" + t[m.end(1):]
open(q, "w").write(t)
GRADEOF
        grep -q "grad(epsilon)   leastSquares;" "$C/system/fvSchemes" \
            && grep -q "grad(k)         Gauss linear;" "$C/system/fvSchemes" \
            && grep -q "default         Gauss linear;" "$C/system/fvSchemes" \
            || { echo "FAIL: the gradient split was not staged, or it changed the default too"; return 1; }
    fi
    # `orthogonal` / `uncorrected` / `limited0`: the case's laplacianSchemes and snGradSchemes `default`.
    # THREE SCHEMES, TWO FACTS. OpenFOAM's correctedSnGrad.H:108-114 and uncorrectedSnGrad.H:113-119 BOTH
    # return mesh().nonOrthDeltaCoeffs(); they differ only in corrected(). Only orthogonalSnGrad.H:113-119
    # returns mesh().deltaCoeffs(). limitedSnGrad.H:165-172 returns nonOrthDeltaCoeffs and :176-178 says
    # corrected() whatever the limiter, and limitedSnGrad.C:48-58's limiter is identically 0 at k = 0 --
    # so `limited 0` IS `uncorrected`, which this gate asserts of OpenFOAM's own output below.
    #
    # brae carried ONE flag, so `uncorrected` ran ORTHOGONAL (behind a mesh refusal) and `limited 0` ran
    # ORTHOGONAL behind nothing at all. The control is therefore the `orthogonal` run, NOT the shipped
    # `corrected` one: orthogonal is precisely what brae was computing under both names.
    #
    # THIS TUTORIAL IS THE FIXTURE: 28,000 cells, max non-orthogonality 44.5185 degrees over 18,575 of
    # them, 1/cos = 1.4025. MEASURED, OpenFOAM against OpenFOAM at this gate's end time, `uncorrected`
    # against `orthogonal` -- what brae ran against what it should have: alpha 1.261e-05, p_rgh 4.552e-05,
    # U 2.142e-03, k 2.998e-02, epsilon 6.430e-02, nut 4.805e-02, all 28,000 of 28,000 cells differing.
    # Against the POROUS bounds this arm is held to, the smallest margin is p_rgh at 2.3e4x.
    if [ "$profile" = orthogonal ] || [ "$profile" = uncorrected ] || [ "$profile" = limited0 ]; then
        lap="Gauss linear orthogonal"; sng="orthogonal"
        [ "$profile" = uncorrected ] && { lap="Gauss linear uncorrected"; sng="uncorrected"; }
        [ "$profile" = limited0 ] && { lap="Gauss linear limited 0"; sng="limited 0"; }
        grep -q "default         Gauss linear corrected;" "$C/system/fvSchemes" \
            && grep -q "default         corrected;" "$C/system/fvSchemes" \
            || { echo "FAIL: the tutorial no longer ships 'corrected' on both blocks"; return 1; }
        python3 - "$C/system/fvSchemes" "$lap" "$sng" <<'SNEOF' || { echo "FAIL: staging $profile"; return 1; }
import re, sys
q, lap, sng = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(q).read()
for block, val in (("laplacianSchemes", lap), ("snGradSchemes", sng)):
    m = re.search(r"(%s\s*\{[^}]*?default\s+)([^;]+)(;)" % block, t, re.S)
    assert m, block
    assert m.group(2).strip().endswith("corrected"), (block, m.group(2))
    t = t[:m.start(2)] + val + t[m.end(2):]
open(q, "w").write(t)
SNEOF
        grep -q "default         $lap;" "$C/system/fvSchemes" \
            && grep -q "default         $sng;" "$C/system/fvSchemes" \
            || { echo "FAIL: $profile's schemes were not staged"; return 1; }
    fi
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
for p in inactive porous inactiveWater porousWater splitGrad orthogonal uncorrected limited0; do
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

# ...and the GRADIENT SPLIT, whose control is the `porous` run: the same case with one `default` for both.
"$BIN" "$W/splitGrad" "$W/splitGrad/0" "$W/splitGrad/$END" "$STEPS" "$W/splitGrad/log.interFoam" \
       "$W/porous/$END" splitGrad || rc=1

# OPENFOAM'S OWN CLAIM, CHECKED: limitedSnGrad at k = 0 is uncorrectedSnGrad, so these two runs must agree
# to the last written digit. If they ever diverge, the `limited0` arm below is measuring something else.
for fld in alpha.water p_rgh U k epsilon nut; do
    cmp -s "$W/uncorrected/$END/$fld" "$W/limited0/$END/$fld" \
        || { echo "FAIL: OpenFOAM's 'limited 0' and 'uncorrected' differ in $fld"; rc=1; }
done
[ $rc = 0 ] && echo "OpenFOAM's 'limited 0' is bitwise 'uncorrected' on all six fields"

# ...and the COEFFICIENT CHOICE, whose control is the `orthogonal` run -- which is what brae ran under
# both of these names. `limited0` is the same solve reached through the limited parse.
"$BIN" "$W/uncorrected" "$W/uncorrected/0" "$W/uncorrected/$END" "$STEPS" "$W/uncorrected/log.interFoam" \
       "$W/orthogonal/$END" uncorrected || rc=1
"$BIN" "$W/limited0" "$W/limited0/0" "$W/limited0/$END" "$STEPS" "$W/limited0/log.interFoam" \
       "$W/orthogonal/$END" limited0 || rc=1

echo "interfoam_angledduct_vs_openfoam: rc $rc"
exit $rc
