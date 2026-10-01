#!/usr/bin/env bash
# brae's interFoam with the MANGROVE fvOptions against REAL OpenFOAM's, on
# laminar/waves/mangroveInteraction, field by field and solve by solve.
#
# THE CASE: a Boussinesq wave paddle against a shallowWaterAbsorption outlet, kEpsilon with k and epsilon
# solved by PBiCG and DILU, and two fvOptions over the cellZone the seaweed surface marks --
# multiphaseMangrovesSource (drag 0.5*Cd*a*N*|U| and added mass 0.25*(Cm + 1)*pi*a^2*N*ddt(U), both
# times rho) and multiphaseMangrovesTurbulenceModel (-Sp(Ckp*Cd*a*N*|U|, k), -Sp(Cep*Cd*a*N*|U|,
# epsilon)). Meshed as Allrun does (blockMesh, setFields, topoSet), the block halved in each direction,
# 450 fixed steps of 0.01 at the case's own tolerances -- the wave reaches the seaweed near t = 4.
#
# MEASURED: alpha 4.6e-14, p_rgh 3.8e-14, U 3.4e-12, k 1.7e-13, epsilon 1.0e-13, nut 3.0e-13; every
# p_rgh, k and epsilon count OpenFOAM's, and k's and epsilon's FINAL residuals with them.
# CONTROLS, OpenFOAM against itself: both options off, U 134%; the turbulence option off, k 52%.
#
# FAIL-PROOFS, each decision broken once in a scratch copy, the gate red every time (U / k):
#   no added mass                                   2.2e-01 / 1.2e-01
#   no drag                                         8.0e-01 / 1.0
#   drag without rho                                7.9e-01 / 1.1
#   the added mass with Cm in place of Cm + 1       1.1e-01 / 6.0e-02
#   Ckp and Cep swapped                             9.9e-01 / 7.5e-01
#   the turbulence source's sign flipped            8.3e-01 / 6.9e-01
#   no turbulence source                            3.4e-01 / 5.2e-01
#   PBiCGStab in place of PBiCG                     9.2e-05 / 1.2e-04, 76 of 450 k counts equal
#   DILU's transpose sweep not transposed           the run fails: k and epsilon go bad and the pressure
#                                                   solve refuses its matrix
# NOT DISCRIMINATED: the added mass reading U where OpenFOAM reads U.oldTime() -- with one outer
# corrector and no momentum predictor the two are the same field when UEqn is assembled.
#
# THE DEVICE LOOP runs the same 450 steps from the same start and is held to OpenFOAM by its OWN bounds
# (D_* in the .cu): alpha 6.4e-14, p_rgh 4.8e-14, U 3.9e-11, k 3.2e-12, epsilon 3.5e-12, nut 1.7e-12; all
# 900 p_rgh counts and all 450 k and 450 epsilon PBiCG counts OpenFOAM's, their final residuals
# OpenFOAM's to the floor a normalised residual has. Two runs print the same digits. THREE MODULES, each
# transcribed from the host reference and taken one at a time against it (twenty steps, U 3.9e-12 with
# the HOST closure inside the device loop, 4.0e-12 with the device's own):
#   the drag and added mass on U      device_fvoptions.cu, in the shared assembler's fvOptions slot
#   PBiCG with DILU                   device_pbicg.cu: the transpose system is a VIEW with upper and
#                                     lower exchanged; tests/test_device_pbicg.cu holds it to the host's
#   the k and epsilon sink            kEpsilon.cu, diag += V*coeff ahead of relax(), kEpsilon.C:258/279
# WHAT THE PORT FOUND: the device closure set its Gauss-Seidel sweep UNCONDITIONALLY, whatever solver
# kFinal named. Only the fvOptions refusal stood between this case and a smoothSolver run under PBiCG's
# entry, with no notice -- 0 of 450 k counts equal and U 1.1e-04, measured below. It names the solver
# now and refuses any it does not run.
# BROKEN ONCE EACH ON THE DEVICE ARM (U / k) -- the first seven are the host arm's numbers to two digits,
# which is what a transcription should give:
#   no added mass                                   2.2e-01 / 1.2e-01
#   no drag                                         8.0e-01 / 1.0
#   drag without rho                                7.9e-01 / 1.1
#   the added mass with Cm in place of Cm + 1       1.1e-01 / 6.0e-02
#   Ckp and Cep swapped                             9.9e-01 / 7.5e-01
#   the turbulence source's sign flipped            8.3e-01 / 6.9e-01
#   no turbulence source                            3.4e-01 / 5.2e-01
#   PBiCGStab (with DILU) in place of PBiCG         9.2e-05 / 1.2e-04, 76 of 450 k counts and 33 of 450
#                                                   epsilon counts equal
#   the Gauss-Seidel sweep in place of PBiCG        1.1e-04 / 1.6e-04, 0 of 450 counts of either
# NOT DISCRIMINATED there either: the added mass on U for U.oldTime() -- every digit the same.
#
# PROFILE twoOptions: a SECOND option of each type over the SAME cellZone as the first (a 0.008, N 300,
# Cm 2, Cd 1.1; Ckp 0.7, Cep 3.5 -> 2.0). OpenFOAM builds one source object per entry and `fvOptions(U)`
# sums the list, so the two overlap and BOTH count -- which is what the device loop refused to do until
# now, and what a single merged coefficient array per type cannot do (it would keep the last entry's
# numbers alone on every shared cell). The device now carries one coefficient set per option and applies
# them in the list's order, so its matrix takes diag += V*coeff_1 then diag += V*coeff_2, term for term
# what the host's loop over the option list leaves rather than V*(coeff_1 + coeff_2).
# THE CONTROL is the `mangrove` run itself -- one option of each instead of two, OpenFOAM against itself:
# the second pair moves its own k by 1.3e-01, against brae's 1.6e-13.
# MEASURED on twoOptions: host alpha 6.0e-14, p_rgh 4.9e-14, U 4.2e-12, k 1.6e-13, epsilon 1.1e-13,
# nut 2.3e-13; device alpha 7.6e-14, p_rgh 5.7e-14, U 3.9e-11, k 3.0e-12, epsilon 3.5e-12, nut 1.8e-12 --
# the one-option run's bounds, unchanged.
# BROKEN ONCE EACH on the device, keeping ONE coefficient array per type as the loop did before:
#   only the last multiphaseMangrovesSource            alpha 3.7e-01, U 7.0e-01, k 6.6e-01  (18 arms red)
#   only the last multiphaseMangrovesTurbulenceModel   alpha 1.5e-01, U 3.3e-01, k 4.5e-01  (16 arms red)
# The host arm is unmoved by either, which is what says the defect is the device's alone.
#
# PROFILE densityVariable: the same two options beside the DENSITY-WEIGHTED k-epsilon (`density
# variable`, with div(rhoPhi,k) and div(rhoPhi,epsilon) to match the flux that lineage transports with).
# `fvOptions(alpha, rho, k_)` with alpha one dispatches to addSup(rho, eqn), which is
# -Sp(rho*kCoeff(U), k) rather than -Sp(kCoeff(U), k) (multiphaseMangrovesTurbulenceModel.C:185-210):
# the product is formed as a FIELD and fvm::Sp then multiplies by V, so the matrix takes V*(rho*coeff)
# and rho belongs in the coefficient, not beside the volume. BOTH loops refused this lineage before --
# the host inside fvOptions_cpp.cu, the device twice over, in its case reader and in its closure.
# THE CONTROLS are OpenFOAM against itself on the same lineage: both options off moves U 1.5e+00, and
# the UNIFORM-density run moves k 1.5e+00.
# MEASURED: host alpha 7.1e-14, p_rgh 6.0e-14, U 5.2e-12, k 1.8e-13, epsilon 1.6e-13, nut 7.8e-14;
# device alpha 7.7e-14, p_rgh 6.0e-14, U 3.7e-11, k 3.7e-12, epsilon 3.8e-12, nut 1.1e-12; all 900
# p_rgh, 450 k and 450 epsilon counts OpenFOAM's on both arms.
# BROKEN ONCE EACH, rho dropped from the coefficient:
#   host     k 3.8e-01, epsilon 4.8e-01, 397 of 450 k counts (16 arms red), the device untouched
#   device   k 9.2e-01, epsilon 2.8e+00, 356 of 450 k counts (16 arms red), the host untouched
# The device REFUSES a null rho outright, so that second defect had to be injected past the guard --
# which is the guard doing its job.
# ONE BOUND IS THIS PROFILE'S OWN: k's final residual on the device, D_KE_FINAL_RHO. Under `density
# uniform` both arms reproduce k's and epsilon's final residuals to every digit OpenFOAM's log prints;
# under `density variable` epsilon still does and k reads 6.975e-06 (host) and 2.621e-05 (device) on a
# residual near 1e-08 -- an absolute 2.6e-13, the size of the k field's own agreement. The arm still
# catches a substituted solver: PBiCGStab on the device reads k 1.1e-04 with 7 of 450 counts.
#
# NOT CLAIMED: the tutorial's 411,600-cell mesh (a bench size), its sampled line sets (stripped), any
# closure but kEpsilon under the turbulence option (refused), a field-name override on either option
# (refused), and either option beside a moving mesh (refused by the reader).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_mangrove_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/waves/mangroveInteraction"
STEPS=${STEPS:-450}
DT=${DT:-0.01}
# the tutorial's (350 28 42) block is 411,600 cells -- a bench size, not a validation one; half of it in
# each direction is 51,450, with the seaweed zone still 14,700 of them
BLOCK=${BLOCK:-"175 14 21"}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: mangroveInteraction tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}

# real OpenFOAM's runs (and meshes) are cached by a hash of the staged case: tests/of_oracle_cache.sh
. "$(dirname "$0")/of_oracle_cache.sh"
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh setFields topoSet interFoam; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

END=$(python3 -c "
t = 0.0
for i in range($STEPS):
    t += float('$DT')
print('%.10g' % t)")

# stage <profile>: copy the tutorial, coarsen and fix the step, mesh it as Allrun does (serial), apply the
# profile to the options, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    grep -q "(350 28 42)" "$C/system/blockMeshDict" || { echo "FAIL: the tutorial's block is no longer (350 28 42)"; return 1; }
    sed -i "s/(350 28 42)/($BLOCK)/" "$C/system/blockMeshDict"
    grep -q "type            multiphaseMangrovesSource;" "$C/system/fvOptions" \
        && grep -q "type            multiphaseMangrovesTurbulenceModel;" "$C/system/fvOptions" \
        || { echo "FAIL: the tutorial no longer names both mangrove options"; return 1; }
    PROFILE="$profile" STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
t = 0.0
for i in range(n):
    t += float(dt)
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
# the line sets write nothing a field comparison reads, and brae runs no function objects
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % t),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
o = os.path.join(d, 'system/fvOptions')
s = open(o).read()
p = os.environ['PROFILE']
if p == 'off':
    s, k = re.subn(r'active\s+yes;', 'active          no;', s)
    assert k == 2, 'off'
elif p == 'turbOff':
    i = s.index('TurbulenciaMangroves')
    head, tail = s[:i], s[i:]
    tail, k = re.subn(r'active\s+yes;', 'active          no;', tail, count=1)
    assert k == 1, 'turbOff'
    s = head + tail
elif p.startswith('densityVariable'):
    # THE DENSITY-WEIGHTED k-epsilon beside the same two options. `fvOptions(alpha, rho, k_)` with
    # alpha one dispatches to addSup(rho, eqn), which is -Sp(rho*kCoeff(U), k) rather than
    # -Sp(kCoeff(U), k) -- a different matrix on every cell of the zone, and the one lineage both
    # arms refused until now. The convection scheme has to follow the flux the lineage transports
    # with, or OpenFOAM stops on a missing div(rhoPhi,k) entry.
    t = os.path.join(d, 'constant/turbulenceProperties')
    ts = open(t).read()
    assert ts.count('simulationType') == 1, 'turbulenceProperties'
    ts = re.sub(r'^simulationType.*$', 'density             variable;\n\nsimulationType      RAS;', ts, flags=re.M)
    open(t, 'w').write(ts)
    f = os.path.join(d, 'system/fvSchemes')
    fs = open(f).read()
    for a, b in [('div(phi,k)', 'div(rhoPhi,k)'), ('div(phi,epsilon)', 'div(rhoPhi,epsilon)')]:
        assert fs.count(a) == 1, a
        fs = fs.replace(a, b)
    open(f, 'w').write(fs)
    if p == 'densityVariableOff':
        # ...and the same lineage with both options off, so the control for `densityVariable` differs
        # from it by the OPTIONS alone and not by the density weighting as well
        s, k = re.subn(r'active\s+yes;', 'active          no;', s)
        assert k == 2, 'densityVariableOff'
elif p == 'twoOptions':
    # A SECOND OPTION OF EACH TYPE, over the SAME cellZone as the first. OpenFOAM constructs one source
    # object per entry and `fvOptions(U)` sums the list, so the two overlap and both count; an
    # implementation that keeps one coefficient array per type would keep only this one's numbers.
    s += '''
Mangroves2
{
    type            multiphaseMangrovesSource;
    active          yes;

    multiphaseMangrovesSourceCoeffs
    {
        regions
        {
            region1
            {
                cellZone        c0;
                a               0.008;
                N               300;
                Cm              2;
                Cd              1.1;
            }
        }
    }
}


TurbulenciaMangroves2
{
    type            multiphaseMangrovesTurbulenceModel;
    active          yes;

    multiphaseMangrovesTurbulenceModelCoeffs
    {
        regions
        {
            region1
            {
                cellZone        c0;
                a               0.008;
                N               300;
                Ckp             0.7;
                Cep             2.0;
                Cd              1.1;
            }
        }
    }
}
'''
open(o, 'w').write(s)
PYEOF
    ( cd "$C" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 \
          && topoSet > log.topoSet 2>&1 ) \
        || { echo "FAIL: meshing [$profile]"; return 1; }
    oracleRun "$C" interfoam_mangrove "$profile" || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in off turbOff mangrove twoOptions densityVariable densityVariableOff; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_mangrove_vs_openfoam: staging failed"; exit 1; }

"$BIN" "$W/mangrove" "$W/mangrove/0" "$W/mangrove/$END" "$STEPS" "$W/mangrove/log.interFoam" "$W/off/$END" "$W/turbOff/$END" || rc=1

# twoOptions: TWO options of each type over the same cellZone, with the ONE-of-each run as the control --
# the second option has to have moved OpenFOAM's own answer, or the profile tests nothing
"$BIN" "$W/twoOptions" "$W/twoOptions/0" "$W/twoOptions/$END" "$STEPS" "$W/twoOptions/log.interFoam" \
       "$W/off/$END" "$W/mangrove/$END" twoOptions || rc=1

# densityVariable: the density-weighted k-epsilon, with the UNIFORM run as the control -- the weighting
# has to have moved OpenFOAM's own k, or the profile tests nothing
grep -q "Selecting turbulence model type RAS" "$W/densityVariable/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log does not select a RAS model on densityVariable"; rc=1; }
"$BIN" "$W/densityVariable" "$W/densityVariable/0" "$W/densityVariable/$END" "$STEPS" \
       "$W/densityVariable/log.interFoam" "$W/densityVariableOff/$END" "$W/mangrove/$END" \
       densityVariable || rc=1

echo "interfoam_mangrove_vs_openfoam: rc $rc"
exit $rc
