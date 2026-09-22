#!/usr/bin/env bash
# brae's interFoam across a CYCLIC BAFFLE, and with porousBafflePressure's jump on it, against REAL
# OpenFOAM's on RAS/damBreakPorousBaffle, field by field and solve by solve.
#
# THE METHOD is the other interFoam gates': exactly N identical FIXED steps at the tutorial's own
# deltaT (1e-3), both codes read at one instant. The mesh is the tutorial's own -- blockMesh, then
# createBaffles, which turns 13 internal faces at x = 0.3042 into the cyclic pair porous_half0/1 -- 2268
# cells. kEpsilon, `MULESCorr yes`, three outer correctors, no momentum predictor, DICPCG on p_rgh and
# symGaussSeidel on everything else, `linearUpwind grad(U)`.
#
# WHAT HAD TO EXIST FIRST: brae's host interFoam had NO coupled patch. The shared factory built a cyclic
# as a zeroGradient placeholder, so the pair would have run as two walls, and the only thing that stopped
# it was CorrectPhi refusing for a reason of its own. Now the driver attaches the coupling to the mesh
# patch (attachCyclicCoupling: neighbour cells, weights, delta, the coupled deltaCoeffs and correction
# vectors, cyclicFvPatch.C and basicFvGeometryScheme.C) and everything on the path branches on it:
#   interpolation   a coupled face from its TWO CELLS, never from a stored patch value: linear, upwind,
#                   vanLeer's limiter from the two cells and their gradients (LimitedScheme.C), and
#                   linearUpwind's explicit correction with the neighbour's gradient (linearUpwind.C)
#   fvm             laplacian: internalCoeffs = boundaryCoeffs = -gamma*magSf*deltaCoeffs, the scheme's
#                   deltaCoeffs; div: phi*w and -phi*(1 - w) -- the second of each an INTERFACE coefficient
#   fvMatrix        A, H (boundaryCoeffs times the cell on the other side), flux (the neighbour's half
#                   through patchNeighbourField), relax
#   linear solvers  PCG/DIC and smoothSolver: the diagonal folded, NO source, Amul and the residual
#                   through the interface, sumA, and Gauss-Seidel's Jacobi update of bPrime every sweep
#   MULES, CMULES   the neighbour cell in the extrema, the per-cell limiter, and syncFaceList's minimum
#                   across the pair -- which is NOT a no-op in serial, as a note in brae said
#   interFoam       phic left alone on a coupled face, the interface normal from the two cells'
#                   gradients, ddtCorr and phig and the reconstruction with the coupled rAUf, pcorr
#
# porousBafflePressure is that cyclic with a JUMP (porousBafflePressureFvPatchField.C:125-189):
#   Un = phi_p/magSf, averaged over the patch under `uniformJump`;
#   jump = -sign(Un)*(D*nu_p + I*0.5*|Un|)*|Un|*length, times rho_p because p_rgh is a pressure.
# Only the OWNER side computes it, at the pressure assembly, from phi as the last corrector left it; the
# other side reads the owner's, negated. It enters the solve through the interface update and ONLY when
# the operand is the field itself (jumpCyclicFvPatchFields.C) -- in PCG, the one Amul that builds the
# initial residual -- and the flux through patchNeighbourField.
#
# FIVE PROFILES, each against its own OpenFOAM run:
#   cyclic             p_rgh a plain cyclic, the coupling alone; 20 steps. CONTROL: OpenFOAM with the
#                      pair as two WALLS, 34% of U
#   porous             the case as createBaffles leaves it; 20 steps. CONTROL: OpenFOAM's `cyclic`, 2.0% of U
#   cyclicWet          the water column staged ACROSS the baffle (setFields box to x 0.4, y 0.13); 60 steps
#   porousWet          ...with the jump. CONTROL: OpenFOAM's cyclicWet, 4.5% of U
#   cyclicWetExplicit  ...with `MULESCorr no`, MULES's own limiter instead of CMULES's
# As shipped, nothing but air reaches the baffle in 20 steps: alpha is zero on both sides of it, and every
# alpha path across the pair is multiplied by nothing. Measured: ten of the breaks below change no digit
# of the two shipped profiles. That is what the wet profiles are for.
#
# THE WET PROFILES' PRESSURE SOLVES ARE PINNED at 1e-13, relTol 0, in both codes. At the case's own 1e-07
# the wet pair's first system is a poorly conditioned Krylov problem: reversing ONE dot product in brae's
# own PCG moves its final residual 4.0e-05 (1e-08 to 1e-10 on the other three systems), and brae against
# OpenFOAM was 1.1e-04 there with every input to the solve exact to 1e-15 and the converged answers
# 6e-15 apart. At that tolerance the comparison measures where two Krylov paths stop. At 1e-13 a PCG
# residual lands on either side of the tolerance, so those profiles hold every count to one and a
# differing pair to a shorter run that stopped within 15% of it: 15 and 11 of 180 do.
#
# MEASURED (alpha absolute, the rest relative):
#   cyclic             alpha 6.9e-15, p_rgh 6.6e-15, U 5.0e-14, k 1.8e-14, epsilon 4.6e-14, nut 1.1e-14;
#                      all 60 p_rgh, 60 alpha, 20 epsilon and 20 k counts OpenFOAM's
#   porous             alpha 7.8e-15, p_rgh 8.0e-15, U 9.9e-13, k 7.1e-13; the jump 1.6e-12 of the one
#                      OpenFOAM writes, the pair's own p_rgh 2.8e-13 on all 26 faces; all counts equal
#   cyclicWet          alpha 2.4e-13, p_rgh 5.1e-13, U 3.2e-12, k 1.4e-12; 180 alpha counts equal
#   porousWet          alpha 6.1e-14, p_rgh 3.0e-13, U 1.1e-12; the jump -- NON-uniform now, rho_p varies
#                      along the baffle -- 5.4e-12
#   cyclicWetExplicit  alpha 3.5e-14, p_rgh 2.4e-13, U 4.0e-13
#
# WHAT THE GATE FOUND. On porousWet, U 1.05e-09 where cyclicWet held 3e-12 -- and brae against itself
# under a round-off perturbation 1e-13, so it was a difference and not sensitivity. The jump agreed to
# 5e-13 after 5 steps and drifted as the interface smeared across the baffle. Refitting OpenFOAM's OWN
# written jump from its own written alpha was exact to 1e-15 on every face but the one whose two cells
# held alpha 0.679 and 0.520: 2.8e-08 out with nu_p = nu(alpha_p), 5e-16 with the two cells' nu
# INTERPOLATED. The cause is v2412's `localConsistency` (on by default, etc/controlDict:225): every
# GeometricField operation ends in correctLocalBoundaryConditions(), which re-evaluates a constraint
# patch of the RESULT, so a derived field's cyclic patch value is the interpolation of its own cells and
# not the operation applied to the operands' patch values. For rho, linear in alpha, the same number;
# for nu, not. brae's mixture now forms its coupled patch values from the cells.
# ALSO FOUND, writing a refusal arm: `Gauss interfaceCompression vanLeer 1` -- interfaceCompression.H's
# limited scheme, not the PhiScheme -- was read as plain vanLeer by a substring match and RAN. Refused.
#
# BROKEN ONCE EACH (U: cyclic, porous, cyclicWet, porousWet; `-` = no digit changed, the profile cannot see it):
#   laplacian without its interface coefficient       4.0e-01  4.0e-01  1.7e+00  1.7e+00
#   upwind div without its interface coefficient      2.8e-03  2.8e-03  9.2e-03  8.2e-03
#   Gauss-Seidel without the interface update         5.6e-03  7.8e-03  9.6e-02  8.9e-02
#   sumA without the interface coefficients           8.8e-06  6.4e-06  counts   counts
#   fvMatrix::flux without the neighbour's half       9.1e-01  NaN      NaN      NaN
#   H() taking the interface coefficient as a source  1.1e+00  1.0e+00  1.1e+00  1.1e+00
#   linearUpwind's coupled correction dropped         2.5e-03  2.5e-03  4.1e-03  3.7e-03
#   ddtCorr's rho*rAU from the patch, not the cells   2.0e-05  2.0e-05  6.3e-05  8.3e-04
#   the reconstruction's rAUf from the face cell      5.2e-05  5.0e-05  4.4e-05  1.4e-03
#   div of the dev2 tensor from its patch value       2.0e-06  2.0e-06  1.5e-05  4.4e-04
#   muEff on the pair from rho_p*nuEff_p              1.3e-07  1.3e-07  6.6e-07  7.1e-06
#   DkEff on the pair from the patch values           1.9e-10  1.9e-10  2.1e-10  4.1e-10
#   phig's rAUf from the face cell                    -        -        1.0e-03  8.5e-03
#   snGrad zero on the pair                           -        -        1.9e-02  6.7e-02
#   phic zeroed on the pair, as on a wall             -        -        3.6e-04  3.6e-04
#   vanLeer's weight on the pair replaced by linear   -        -        1.6e-05  2.9e-04
#   ...its limiter fed the delta reversed             -        -        2.1e-04  1.1e-04
#   the interface normal by the uncoupled formula     -        -        2.1e-06  6.5e-04
#   CMULES without the limiter's sync                 -        -        8.3e-06  5.7e-06
#   CMULES without the neighbour cell in the extrema  -        -        3.2e-11  3.3e-11  (alpha 9.0e-10)
#   the jump left out of the solve's first Amul       -        2.2e-02  -        1.9e-02
#   the jump left out of the flux                     -        2.8e+03  -        4.1e-01
#   the jump not negated on the other side            -        1.2e-01  -        8.7e-02
#   the jump from phiHbyA, not the last phi           -        1.4e-03  -        2.4e-03
#   uniformJump ignored                               -        5.9e-02  -        6.7e-02
#   the jump without rho_p                            -        -        -        4.5e-02
#   nu_p = nu(alpha_p) (what was found)               -        -        -        1.05e-09
# and on cyclicWetExplicit: MULES's bounded donor flux from the patch value, alpha 6.2e-10; its limiter
# without the sync, alpha 6.0e-10.
# NOT DISCRIMINATED, each a transcription with its OpenFOAM lines beside it: fvMatrix::relax's coupled
# branch (the diagonal stays dominant here, so taking the coupled patch as an uncoupled one changes no
# digit); the neighbour cell in the EXPLICIT limiter's extrema; a vector gradient read from the stored
# cyclic value instead of the cells (equal wherever the field was evaluated since it changed, which is
# everywhere on this path); and CorrectPhi across the pair, which every profile starts from phi = 0.
# tests/test_porous_baffle_pressure.cu holds the per-face jump, reversed flow and sign(0), which no
# profile reaches.
#
# NOT CLAIMED, each refused by name (tests/interfoam_refusals.sh): across a cyclic -- GAMG and PBiCGStab,
# a momentum predictor, every div(rhoPhi,U) scheme but upwind and linearUpwind, interfaceCompression,
# leastSquares and cellLimited gradients, a moving mesh, MRF, fvOptions, waves, kOmegaSST, a rotational
# or non-orthogonal pair; cyclicAMI, cyclicACMI and processor patches; on the jump -- `relax`,
# `minJump`, a D or I that is not constant, a mass flux; and the device loop, which refuses any cyclic.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_baffle_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/damBreakPorousBaffle"
STEPS=${STEPS:-20}
STEPS_WET=${STEPS_WET:-60}
PTOL_WET=${PTOL_WET:-1e-13}
DT=${DT:-1e-3}
PROFILES=${PROFILES:-"walls cyclic porous wallsWet cyclicWet porousWet wallsWetExplicit cyclicWetExplicit"}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/damBreakPorousBaffle tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1     || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v createBaffles > /dev/null 2>&1 || { echo "SKIP: createBaffles not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1     || { echo "SKIP: interFoam not on PATH"; exit 77; }

endOf() { python3 -c "print('%.10g' % ($1*float('$DT')))"; }
stepsOf() { case "$1" in *Wet*) echo "$STEPS_WET" ;; *) echo "$STEPS" ;; esac; }

# stage <profile>: copy the tutorial, fix the step, mesh it as Allrun does, apply the profile, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    local N
    N=$(stepsOf "$profile")
    local END
    END=$(endOf "$N")
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "type  *porousBafflePressure;" "$C/system/createBafflesDict" \
        || { echo "FAIL: the tutorial's createBafflesDict no longer carries a porousBafflePressure"; return 1; }

    case "$profile" in
        *Wet*)
            # THE WATER STRADDLES THE BAFFLE. As shipped the column stands at x < 0.1461 and in 20 steps
            # nothing but air reaches the baffle at x = 0.3042, so alpha is identically zero on both
            # sides of it and every alpha path across the pair -- the limited flux, interface
            # compression, the MULES limiter, the interface normal, the density jump in phig -- is
            # multiplied by nothing. Here the column reaches x = 0.4 at half the baffle's height: the
            # free surface crosses the pair and the front falls away beyond it.
            sed -i 's/box  *(0 0 -1) (0.1461 0.292 1);/box (0 0 -1) (0.4 0.13 1);/' "$C/system/setFieldsDict"
            grep -q "box (0 0 -1) (0.4 0.13 1);" "$C/system/setFieldsDict" \
                || { echo "FAIL: the water column was not staged"; return 1; }
            # ...AND THE PRESSURE SOLVES ARE PINNED TIGHT, in both codes. See the header: at the case's own
            # 1e-07 the wet pair's first system moves 4.0e-05 under a reordered dot product in brae's OWN
            # PCG, so at that tolerance the comparison measures where two Krylov paths stop.
            sed -i 's/tolerance  *1e-07;/tolerance       '"$PTOL_WET"';/; s/relTol  *0.05;/relTol          0;/' "$C/system/fvSolution"
            grep -q "tolerance  *$PTOL_WET;" "$C/system/fvSolution" \
                || { echo "FAIL: the p_rgh tolerance was not staged"; return 1; }
            ;;
    esac
    case "$profile" in
        *Explicit)
            # THE EXPLICIT LIMITER. The case ships `MULESCorr yes`, whose limiter is CMULES's; MULES's own
            # (MULESTemplates.C) has its own coupled branches -- upwind's flux kept on a coupled face, the
            # neighbour cell in the extrema, the per-cell limiter and its sync -- and no MULESCorr case
            # reaches them: measured, breaking its bounded donor flux changed no digit of any other profile.
            sed -i 's/MULESCorr  *yes;/MULESCorr       no;/' "$C/system/fvSolution"
            grep -q "MULESCorr  *no;" "$C/system/fvSolution" || { echo "FAIL: MULESCorr was not switched off"; return 1; }
            ;;
    esac

    STEPS="$N" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
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
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && createBaffles -overwrite > log.createBaffles 2>&1 ) || { echo "FAIL: createBaffles [$profile]"; tail -20 "$C/log.createBaffles"; return 1; }

    PROFILE="$profile" python3 - "$C" <<'PYEOF' || { echo "FAIL: applying profile $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
profile = os.environ['PROFILE']
def entries(path, body):
    t = open(path).read()
    t, k = re.subn(r'(porous_half[01]\s*\{)[^}]*\}', lambda mo: mo.group(1) + ' ' + body + ' }', t)
    assert k == 2, (path, k)
    open(path, 'w').write(t)
profile = profile.replace('Explicit', '').replace('Wet', '')
if profile == 'cyclic':
    # the coupling ALONE: p_rgh a plain cyclic on both halves
    entries(os.path.join(d, '0/p_rgh'), 'type cyclic;')
elif profile == 'walls':
    # THE CONTROL: the pair as two WALLS, which is what a zeroGradient placeholder runs
    b = os.path.join(d, 'constant/polyMesh/boundary')
    t = open(b).read()
    t, k = re.subn(r'(porous_half[01]\s*\{[^}]*?type\s+)cyclic;', r'\1wall;', t)
    assert k == 2, k
    t = re.sub(r'\n\s*(neighbourPatch|matchTolerance|transform)\s+[^;]*;', '', t)
    t = re.sub(r'inGroups\s+2\(cyclic cyclicFaces\);', 'inGroups        1(wall);', t)
    open(b, 'w').write(t)
    for name, body in [('p_rgh', 'type fixedFluxPressure; gradient uniform 0; value uniform 0;'),
                       ('U', 'type noSlip;'), ('alpha.water', 'type zeroGradient;'),
                       ('k', 'type kqRWallFunction; value uniform 0.1;'),
                       ('epsilon', 'type epsilonWallFunction; value uniform 0.1;'),
                       ('nut', 'type nutkWallFunction; value uniform 0;'),
                       ('nuTilda', 'type zeroGradient;')]:
        p = os.path.join(d, '0', name)
        t = open(p).read()
        t = re.sub(r'"\(porous_half0\|porous_half1\)"\s*\{[^}]*\}', '', t)
        t = re.sub(r'porous_half[01]\s*\{[^}]*\}', '', t)
        t, k = re.subn(r'(boundaryField\s*\{)', r'\1\n    porous_half0 { %s }\n    porous_half1 { %s }' % (body, body), t, count=1)
        assert k == 1, name
        open(p, 'w').write(t)
PYEOF
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $N steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in $PROFILES; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_baffle_vs_openfoam: staging failed"; exit 1; }

# gate <profile> <its control> <cyclic|porous>
gate()
{
    local N
    N=$(stepsOf "$1")
    local END
    END=$(endOf "$N")
    "$BIN" "$W/$1" "$W/$1/0" "$W/$1/$END" "$N" "$W/$1/log.interFoam" "$W/$2/$END" "$3" || rc=1
}
for p in $PROFILES; do
    case "$p" in
        cyclic)    gate cyclic walls cyclic ;;
        porous)    gate porous cyclic porous ;;
        cyclicWet) gate cyclicWet wallsWet cyclicWet ;;
        porousWet) gate porousWet cyclicWet porousWet ;;
        cyclicWetExplicit) gate cyclicWetExplicit wallsWetExplicit cyclicWetExplicit ;;
    esac
done

echo "interfoam_baffle_vs_openfoam: rc $rc"
exit $rc
