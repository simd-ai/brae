#!/usr/bin/env bash
# brae interFoam on a FULLY PERIODIC mesh against REAL OpenFOAM's, on validation/interFoamCyclic.
#
# THE CASE is brae's own, not a tutorial: 800 cells, x cyclic, y walled, z one cell thick, with the
# water surface staged as a STEP at the periodic boundary (y = 0.3 on one side, 0.15 on the other), so
# alpha, its gradient, nHatf and the pressure all carry a jump through the pair from the first step and
# gravity drives flow across it for the whole run. The baffle gate
# (tests/interfoam_baffle_vs_openfoam.sh) already couples an INTERNAL pair; what is new here is a pair
# on the DOMAIN boundary with no wall behind it, and a case small enough to run the device arm on.
#
# THE ORACLE is OpenFOAM's own fields after ten FIXED steps of 2e-3, written ASCII at 17 digits.
# THE CONTROL is OpenFOAM's own answer with the pair replaced by two WALLS: same block, same cells.
#
# WHAT IT ASSERTS: brae's host loop is OpenFOAM's on this mesh; OpenFOAM moves fluid through the pair,
# so the comparison is not vacuous; walling the pair is a different answer by orders of magnitude; and
# the DEVICE arm agrees with the same OpenFOAM fields, and with brae's own host arm.
#
# MEASURED, ten steps of 2e-3: alpha 1.9e-13, p_rgh 9.8e-12 relative of 1.7e+03, U 7.5e-13 relative of
# 0.73, all 30 p_rgh iteration counts OpenFOAM's. CONTROL: OpenFOAM with the pair two walls, alpha
# 2.1e-01 and U 100% away -- and its step one runs 64/50/1 iterations where the periodic case runs
# 70/32/4, so the iteration-count arm sees the coupling too.
#
# FOUR TURBULENT PROFILES, added when the closures learned the pair. kEpsilon was the ONE closure the
# case reader let across a cyclic; `sst` is kOmegaSST with the wall-function family on the walls,
# `les` is LES kEqn, and `sstCN`/`lesCN` are those two under `CrankNicolson 0.9` with the SAME case
# under Euler as their control. The closure's own fields are compared, not only the three the other
# profiles share: a defect confined to k reaches U through nuEff alone.
# MEASURED, ten steps, host then device:
#   sst    host  alpha 2.5e-13, p_rgh 5.3e-14, U 4.0e-13, k 4.8e-13, omega 8.8e-14, nut 2.4e-12
#          device      7.2e-12,       4.4e-11,   7.5e-11,   2.4e-11,        7.0e-12,     5.0e-10
#   les    host  alpha 3.7e-13, p_rgh 5.4e-12, U 4.9e-13, k 1.8e-13,               nut 1.8e-13
#          device      3.8e-12,       5.9e-11,   8.3e-11,   1.3e-11,                    1.3e-11
#   sstCN  host  alpha 3.5e-13, p_rgh 5.8e-11, U 1.4e-12, k 6.6e-13, omega 1.4e-13, nut 3.6e-12
#   lesCN  host  at the same floor. CONTROL for both: OpenFOAM's own Euler answer, alpha 3.2e-02 and
#          U 39% away. Their DEVICE arm is not here -- the loop refuses CrankNicolson's end-of-step
#          alpha flux across a coupled pair by name -- and lives in interfoam_cn_vs_openfoam instead.
# BROKEN ONCE EACH, on these four:
#   DkEff rebuilt on the pair's faces from the   host k 1.3e-11, omega 4.7e-11, nut 7.7e-11 (which is
#   patch's own nut instead of keeping           why B_TURB is 4x the measurement and not 30x --
#   fvc::interpolate's two CELLS                 a 1e-10 bound passed every one of them)
#   CDkOmega's grad(k)/grad(omega) built         device k 4.8e-05, omega 6.7e-04, nut 3.6e-03,
#   without the interface                        U 7.8e-05
#   the LES closure handed no pair at all        device k 4.3e-01, nut 2.8e-01, U 8.7e-03
# NOT A DEFECT, checked in OpenFOAM's own source: the coupled skip in the effective diffusivity.
# surfaceInterpolationScheme::interpolate takes pLambda*patchInternalField + pY*patchNeighbourField
# wherever the patch field is coupled (:186-191), and reads the patch value only where it is not.
#
# TWO PROFILES, because they are two alpha equations: `MULESCorr` as the fixture ships, and `explicit`
# with MULESCorr off -- no implicit pre-solve, and so no mixture.correct() before the first corrector,
# which is what makes its phir read the nHatf the PREVIOUS TIME STEP left.
#
# MEASURED, ten steps, host then device against OpenFOAM:
#   MULESCorr  host alpha 1.9e-13, p_rgh 9.8e-12, U 7.5e-13; device 4.2e-11, 1.98e-11, 5.7e-11
#   explicit   host alpha 5.9e-14, p_rgh 2.9e-12, U 5.7e-13; device 4.3e-11, 3.1e-11, 3.0e-11
#   jump       host alpha 4.1e-13, p_rgh 1.0e-11, U 4.4e-12; device 6.2e-12, 2.4e-11, 7.5e-11
#   outer      nOuterCorrectors 3: host alpha 3.2e-14, p_rgh 6.7e-13, U 1.3e-13; device 1.9e-09,
#              2.2e-09, 2.2e-07 -- LOOSER because the case's own `tolerance 1e-12` is reached three
#              times per step and its slack compounds, not because the loop differs: with every solve
#              pinned at 1e-16 the same device-vs-host comparison reads 3.8e-13 / 1.8e-11 / 9.2e-13
# and the device against brae's own host arm, 4.2e-11 / 1.02e-11 / 5.8e-11 and 4.3e-11 / 2.9e-11 /
# 3.0e-11. The alpha figure is the device alpha solver's stopping point: pinned at 1e-16 the two arms
# agree to 6.1e-14.
#
# BROKEN ONCE EACH (at t = 0.02):
#   the pair's laplacian deltaCoeffs 0.1% out          host alpha 2.1e-03, p_rgh 3.6e-03, U 7.9e-03,
#                                                      and 2 of 30 iteration counts lost
#   divDevReff handed a null pair (as it was)          device alpha 4.4e-06, p_rgh 6.2e-06, U 5.4e-04
#   the Gauss-Seidel sweep not applying the interface  device alpha 6.6e-01, p_rgh 1.2e+00, U 1.6e+01
#   the jump left out of the device entirely           device alpha 4.9e-02, U 45% -- the control's
#                                                      own distance, i.e. the plain-cyclic answer
#   the device running ONE outer corrector whatever    device alpha 2.2e-02, p_rgh 3.3e-02, U 54% --
#   the case asks (as it did before this)              exactly the `outer` profile's own control
#   the pair's mixture boundary one stage stale        device alpha 6.6e-05, p_rgh 1.1e-04, U 3.8e-03
#                                                      (porousBafflePressure reads nu and rho THERE,
#                                                      and they are the two cells' interpolated)
#   the pair's nHatf in a buffer local to one step     the `explicit` profile ABORTS by name (its
#                                                      first corrector asks for a normal nothing has
#                                                      written yet); `MULESCorr` still passes, which
#                                                      is why that profile alone did not cover it
# NOT DISCRIMINATED, measured: the pair's nHatf SEEDED AT ZERO rather than from calculateK. This
# fixture starts from rest, so phi and therefore phic are zero at step one and phir = phic*nHatf is
# zero whatever the normal is; only a restart with a live flux would read that seed.
# NOT MEASURABLE, because brae refuses it outright: the pair left uncoupled, which every other cyclic
# gate calls its control -- the host driver throws and names the patch rather than running it as two
# walls.
#
# PROFILE sstLsq: a LEAST-SQUARES gradient for grad(k) and grad(omega) ACROSS THE PAIR, both arms.
# leastSquaresVectors.C:131-140 folds a COUPLED face into the dd tensor with the OWNER WEIGHT and the
# patch's own delta() -- the cell-to-cell vector, where an uncoupled face takes the patch-normal
# projection -- and leastSquaresGrad.C:108-119 fits it against the NEIGHBOUR CELL's value. Both loops
# refused the combination outright until this unit: the host in leastSquaresInvDd ("leastSquaresVectors.C
# weights a coupled face by the cell on the other side"), the device at two sites, because
# deviceLeastSquaresGrad carried no interface. It reaches the closure where the cellLimited profile's
# gradient does -- CDkOmega, and the corrected laplacian's non-orthogonal correction.
# THE TRAP: deviceCyclicAddGrad is the GAUSS face term, Sf*value/V. The fit carries its own coupled
# term inside the kernel, so on the least-squares branch that call is a second, differently normalised
# copy of the same face; it is gated to the Gauss branch at all three call sites.
# MEASURED, 10 steps: host alpha 2.5169e-13, p_rgh 5.3333e-14, U 4.0271e-13, k 4.8462e-13,
# nut 2.3931e-12, omega 8.8188e-14, all 30 p_rgh counts OpenFOAM's; device alpha 7.1961e-12,
# p_rgh 4.3689e-11, U 7.4830e-11, k 2.4271e-11, nut 5.0209e-10, omega 7.0367e-12.
# THE CONTROL is sstLsqWalls -- the pair replaced by two walls -- which moves OpenFOAM's own alpha
# 2.1132e-01 and its U 9.9993e-01.
# BROKEN ONCE on the device, the pair's faces dropped from both least-squares kernels (the state this
# unit found): U 1.7464e-08, k 9.3003e-09, nut 6.6245e-07, omega 2.0564e-07 -- five arms red, with the
# HOST arm unmoved, which is what says the defect is the device's own.
# STILL REFUSED: a vector least-squares gradient across a pair (grad(U), at its own site), and a
# ROTATIONAL pair, refused upstream in attachCyclicCoupling -- the neighbour value would have to be
# transformed into this side's frame before the fit reads it.
#
# PROFILE sstLimUpw: THE LIMITER OF THE DEFERRED CORRECTION across the pair. `cellLimited Gauss linear 1`
# on grad(U) with the momentum on `Gauss linearUpwind grad(U)` -- the fixture ships `Gauss upwind`, which
# has no correction to limit, so no profile here could reach this limiter before. OF's cellLimitedGrad
# treats a coupled face as internal: the neighbour CELL joins the cell's min/max and the extrapolation to
# that face is clipped to it. The device arm limited against the internal faces and the non-coupled
# patches only and refused the combination by name; it builds the pair's CellLimitInterface now, with
# the neighbour value through deviceCyclicNbrValue so a rotational pair's transform is applied -- a
# vector component crossing the interface is not the same component on the other side. It is the list
# the dev2 gradient already built (device_komega_sst.cu:838-848), at the two momentum sites that were
# missing it.
# MEASURED, 10 steps: host alpha 3.3196e-14, p_rgh 3.4942e-13, U 3.4660e-13, k 2.8037e-13,
# nut 2.5188e-12, omega 8.3613e-14; device alpha 7.5446e-12, p_rgh 5.5283e-11, U 6.9677e-11,
# k 2.3476e-11, nut 4.5411e-10, omega 8.4689e-12.
# THE CONTROL is sstLimUpwWalls, which moves OpenFOAM's own alpha 2.4717e-01 and its U 9.9995e-01.
# BROKEN ONCE on the device, the pair dropped from the correction's limiter (the state this unit found):
# alpha 3.5622e-02, p_rgh 6.2502e-02, U 3.6401e-01, k 3.0206e-02, nut 5.6639e-01, omega 8.2720e-03 --
# six arms red, and the host arm unmoved.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_cyclic_vs_openfoam"
# ...and the SOLVER itself, which the assembled-system arm at the end runs twice per case (the test
# binary drives its own two arms in one process and cannot dump them into separate directories)
BIN_SOLVER="${BUILD:-$ROOT/build}/brae_interFoam"
HAVE_GPU=0
command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1 && HAVE_GPU=1
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC="$ROOT/validation/interFoamCyclic"
STEPS=${STEPS:-10}
DT=${DT:-0.002}

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: $SRC not found"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v setFields > /dev/null 2>&1 || { echo "SKIP: setFields not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the fixture, apply the profile, mesh it, set the fields, run OpenFOAM
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[0-9]* "$C"/processor* "$C"/log.* "$C"/constant/polyMesh
    cp -r "$SRC/0.orig" "$C/0.orig"
    cp -r "$C/0.orig" "$C/0"
    grep -q "type cyclic; neighbourPatch right;" "$C/system/blockMeshDict" \
        || { echo "FAIL: the fixture's left patch is no longer the cyclic this gate stages"; return 1; }
    if [ "$profile" = explicitMules ] || [ "$profile" = explicitWalls ]; then
        # THE EXPLICIT MULES BRANCH. Without MULESCorr there is no implicit pre-solve and no
        # mixture.correct() before the correctors, so the FIRST corrector's phir reads the nHatf the
        # PREVIOUS TIME STEP left -- on the pair as everywhere else. The device kept that normal in a
        # buffer local to one step, which is empty when the first corrector asks for it, and refused.
        sed -i 's/MULESCorr       yes;/MULESCorr       no;/' "$C/system/fvSolution"
        grep -q "MULESCorr       no;" "$C/system/fvSolution" \
            || { echo "FAIL: the $profile profile did not turn MULESCorr off"; return 1; }
    fi
    if [ "$profile" = outer ] || [ "$profile" = outerControl ]; then
        # THE PIMPLE OUTER LOOP. `outer` runs nOuterCorrectors 3 -- alpha re-solved from the SAME
        # alpha.oldTime() three times with the latest flux, the momentum matrix reassembled, the
        # pressure correctors run again -- and `outerControl` is the same case with 1, which is what
        # the device loop used to run whatever the case asked for.
        n=3
        [ "$profile" = outerControl ] && n=1
        sed -i "s/^    nOuterCorrectors .*/    nOuterCorrectors    $n;/" "$C/system/fvSolution"
        grep -q "nOuterCorrectors    $n;" "$C/system/fvSolution" \
            || { echo "FAIL: the $profile profile did not set nOuterCorrectors $n"; return 1; }
    fi
    if [ "$profile" = jump ]; then
        # A POROUS BAFFLE ON THE PERIODIC BOUNDARY. p_rgh's pair becomes a porousBafflePressure -- a
        # fixedJump cyclic whose jump is rebuilt at every assembly from that assembly's flux and
        # laminar viscosity (porousBafflePressureFvPatchField.C:125-189). The coupling is unchanged;
        # what is added is a pressure drop the matrix has to carry.
        #
        # THE COEFFICIENTS ARE NOT THE TUTORIAL'S, and the control is why. `uniformJump true` averages
        # Un over the patch, and on a gravity-driven periodic boundary that average is ~0: with the
        # tutorial's D 1000, I 500, length 0.15 the jump came out so small that OpenFOAM's own answer
        # with it and without it differed by 1.1e-13 in alpha -- the gate would have passed on a jump
        # that did nothing. Per-face (`uniformJump false`) at the tutorial's coefficients diverges in
        # OPENFOAM ITSELF (alpha 1e+286 by step 10, then a `nan` in its own solver log). I 5 over
        # length 0.05 is the one that is both live and stable, and the control below measures it.
        python3 - "$C" <<'PYEOF' || { echo "FAIL: the jump profile was not staged"; return 1; }
import sys
p = sys.argv[1] + '/0/p_rgh'
s = open(p).read()
old = '    "(left|right)" { type cyclic; }'
assert s.count(old) == 1, 'the fixture no longer writes p_rgh a plain cyclic'
s = s.replace(old,
"""    "(left|right)"
    {
        type            porousBafflePressure;
        patchType       cyclic;
        D               0;
        I               5;
        length          0.05;
        uniformJump     false;
        jump            uniform 0;
        value           uniform 0;
    }""")
open(p, 'w').write(s)
PYEOF
        grep -q "porousBafflePressure" "$C/0/p_rgh" \
            || { echo "FAIL: the jump profile did not reach p_rgh"; return 1; }
    fi
    if [ "$profile" = sstCN ] || [ "$profile" = lesCN ]; then
        # ...AND UNDER CRANKNICOLSON. The closure's equations take fvm::ddt through ddtSchemes
        # (kOmegaSSTBase.C:572 and :602, kEqn.C:172), so `default CrankNicolson 0.9` reaches k, omega
        # and every other ddt on the loop. THE CONTROL for these two is the SAME case under Euler --
        # OpenFOAM's own `sst` and `les` output -- so what the comparison measures is the scheme.
        sed -i 's/^ddtSchemes .*/ddtSchemes      { default CrankNicolson 0.9; }/' "$C/system/fvSchemes"
        grep -q "CrankNicolson 0.9" "$C/system/fvSchemes" \
            || { echo "FAIL: the $profile profile did not set CrankNicolson"; return 1; }
    fi
    # `Gauss limitedLinear 1` for div(phi,k) and div(phi,omega) ACROSS THE PAIR. OpenFOAM's weights
    # are a surfaceScalarField and a coupled patch takes its boundaryField
    # (gaussConvectionScheme.C:105-108); brae refused the combination outright until the limiter was
    # carried onto the pair on both arms, because the interface coefficient was built with upwind's
    # weight whatever the case said.
    export CLOSUREDIV="Gauss upwind"
    [ "${profile#sstLimDiv}" != "$profile" ] && export CLOSUREDIV="Gauss limitedLinear 1"
    if [ "$profile" = sst ] || [ "$profile" = sstWalls ] || [ "$profile" = sstCN ] \
    || [ "${profile#sstLim}" != "$profile" ] || [ "${profile#sstLsq}" != "$profile" ] \
    || [ "$profile" = les ] || [ "$profile" = lesWalls ] || [ "$profile" = lesCN ]; then
        # A TURBULENCE CLOSURE ACROSS THE PAIR. kEpsilon was the one closure carried across a cyclic
        # (the case reader refused every other by name); these two profiles are kOmegaSST and LES kEqn
        # on the same mesh, with the same step and the same control. The walls are a real `wall`
        # patch, so the SST profile runs the wall-function family too -- kqRWallFunction on k,
        # omegaWallFunction on omega, nutkWallFunction on nut -- and the pair has to stay OUT of
        # every one of those face sets.
        MODEL=$profile PROF="$profile" python3 - "$C" <<'PYEOF' || { echo "FAIL: the $profile profile was not staged"; return 1; }
import os, re, sys
d = sys.argv[1]
prof = os.environ['PROF']
les = prof.startswith('les')
def sub(path, pat, rep, n=1):
    t = open(path).read()
    t2, k = re.subn(pat, rep, t, count=n)
    assert k == n, (path, pat)
    open(path, 'w').write(t2)
tp = os.path.join(d, 'constant/turbulenceProperties')
if les:
    sub(tp, r'simulationType\s+laminar;',
        'simulationType  LES;\n\nLES\n{\n    LESModel        kEqn;\n    turbulence      on;\n'
        '    printCoeffs     on;\n    delta           cubeRootVol;\n'
        '    cubeRootVolCoeffs { deltaCoeff 1; }\n}')
else:
    sub(tp, r'simulationType\s+laminar;',
        'simulationType  RAS;\n\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n}')
sc = os.path.join(d, 'system/fvSchemes')
div = os.environ.get('CLOSUREDIV', 'Gauss upwind')
sub(sc, r'(\s+div\(phi,alpha\)[^\n]*\n)',
    r'\1    "div\\(phi,(k|omega)\\)"  ' + div + ';\n')
if not les:
    open(sc, 'a').write('\nwallDist\n{\n    method meshWave;\n}\n')
fs = os.path.join(d, 'system/fvSolution')
sub(fs, r'"\(U\|k\|epsilon\)\.\*"', '"(U|k|epsilon|omega).*"')
# THE FIELDS. The pair is a plain `cyclic` on each, exactly as U and alpha carry it.
hdr = ('FoamFile { version 2.0; format ascii; class volScalarField; object %s; }\n'
       'dimensions      %s;\ninternalField   uniform %s;\nboundaryField\n{\n')
def write(name, dims, value, wallEntry):
    body = hdr % (name, dims, value)
    body += '    walls        { %s }\n' % wallEntry
    body += '    "(left|right)" { type cyclic; }\n'
    body += '    frontAndBack { type empty; }\n}\n'
    open(os.path.join(d, '0.orig', name), 'w').write(body)
    open(os.path.join(d, '0', name), 'w').write(body)
write('k', '[0 2 -2 0 0 0 0]', '0.0001',
      'type zeroGradient;' if les else 'type kqRWallFunction; value uniform 0.0001;')
if not les:
    write('omega', '[0 0 -1 0 0 0 0]', '3',
          'type omegaWallFunction; value uniform 3;')
write('nut', '[0 2 -1 0 0 0 0]', '0',
      'type zeroGradient;' if les else 'type nutkWallFunction; value uniform 0;')
PYEOF
        grep -q "type cyclic" "$C/0/k" \
            || { echo "FAIL: the $profile profile did not give k the pair"; return 1; }
    fi
    if [ "${profile#sstLim}" != "$profile" ] && [ "${profile#sstLimDiv}" = "$profile" ]; then
        # A CELL-LIMITED GRADIENT ACROSS THE PAIR. `cellLimited Gauss linear 1` on grad(k) and
        # grad(omega) reaches the closure at TWO sites, and both were built without the interface:
        #   * CDkOmega takes fvc::grad(k) & fvc::grad(omega) (kOmegaSSTBase.C:555-558), and F1 blends
        #     every coefficient of the omega equation on the result;
        #   * the CORRECTED laplacian's non-orthogonal correction takes the field's own grad scheme
        #     (correctedSnGrad.C:52-55), which is the shared assembler's `fieldGrad`.
        # OF's cellLimitedGrad folds a coupled patch's patchNeighbourField into its range and clips
        # the extrapolation to that face, so a limiter built without the pair treats those cells as
        # boundary cells. The fixture's laplacianSchemes are `Gauss linear corrected`, which is what
        # makes the second site live; every other profile here leaves the gradients unlimited.
        GU=""
        [ "${profile#sstLimU}" != "$profile" ] && GU="grad(U) cellLimited Gauss linear 1; "
        sed -i "s/^gradSchemes .*/gradSchemes     { default Gauss linear; ${GU}grad(k) cellLimited Gauss linear 1; grad(omega) cellLimited Gauss linear 1; }/" \
            "$C/system/fvSchemes"
        grep -q "cellLimited Gauss linear 1" "$C/system/fvSchemes" \
            || { echo "FAIL: the $profile profile did not set cellLimited"; return 1; }
        grep -q "corrected" "$C/system/fvSchemes" \
            || { echo "FAIL: the fixture's laplacian is no longer corrected, so one of the two sites is dead"; return 1; }
        # ...and `sstLimU` adds grad(U) to them. Its DEVICE arm is refused by name: the momentum
        # assembler sums the Gauss half of grad(U) across the pair but its LIMITER walks the internal
        # faces and the non-coupled patches only (UEqn.cu). The host arm is what this profile gates,
        # and the host's own reader refused it until this fixture could hold it.
        if [ "${profile#sstLimU}" != "$profile" ]; then
            grep -q "grad(U) cellLimited" "$C/system/fvSchemes" \
                || { echo "FAIL: the $profile profile did not limit grad(U)"; return 1; }
        fi
        # ...and `sstLimUpw` puts the momentum on `Gauss linearUpwind grad(U)` as well, which is what
        # makes the LIMITER OF THE DEFERRED CORRECTION live. That limiter walked the internal faces and
        # the non-coupled patches only, so the device arm refused the combination by name; the fixture's
        # own scheme is `Gauss upwind`, which has no correction to limit and could not reach it.
        if [ "${profile#sstLimUpw}" != "$profile" ]; then
            sed -i 's|div(rhoPhi,U)  *Gauss upwind;|div(rhoPhi,U)    Gauss linearUpwind grad(U);|' \
                "$C/system/fvSchemes"
            grep -q "div(rhoPhi,U)    Gauss linearUpwind grad(U);" "$C/system/fvSchemes" \
                || { echo "FAIL: the $profile profile did not set linearUpwind on the momentum"; return 1; }
        fi
    fi
    if [ "${profile#sstLsq}" != "$profile" ]; then
        # A LEAST-SQUARES GRADIENT ACROSS THE PAIR. leastSquaresVectors.C:131-140 folds a COUPLED face
        # into the dd tensor with the OWNER WEIGHT and the patch's own delta() -- the cell-to-cell
        # vector, not the patch-normal projection an uncoupled face takes -- and leastSquaresGrad.C:
        # 108-119 fits it against the NEIGHBOUR CELL's value. Both arms refused this outright: the host
        # at fvc.cu's `leastSquaresInvDd`, the device because deviceLeastSquaresGrad carried no
        # interface. It reaches the closure at the same two sites the cellLimited profile names --
        # CDkOmega, and the corrected laplacian's non-orthogonal correction.
        sed -i "s/^gradSchemes .*/gradSchemes     { default Gauss linear; grad(k) leastSquares; grad(omega) leastSquares; }/" \
            "$C/system/fvSchemes"
        grep -q "grad(k) leastSquares" "$C/system/fvSchemes" \
            || { echo "FAIL: the $profile profile did not set leastSquares"; return 1; }
        grep -q "corrected" "$C/system/fvSchemes" \
            || { echo "FAIL: the fixture's laplacian is no longer corrected"; return 1; }
    fi
    if [ "$profile" = walls ] || [ "$profile" = explicitWalls ] \
    || [ "$profile" = sstWalls ] || [ "$profile" = lesWalls ] \
    || [ "$profile" = sstLimWalls ] || [ "$profile" = sstLimUWalls ] || [ "$profile" = sstLimUpwWalls ] \
    || [ "$profile" = sstLimDivWalls ] || [ "$profile" = sstLsqWalls ]; then
        # THE CONTROL: the pair replaced by two walls, in the mesh AND in every field that names it.
        # blockMesh numbers the cells from the block, so the two runs' cells are the same cells.
        sed -i 's/type cyclic; neighbourPatch right;/type wall;/; s/type cyclic; neighbourPatch left; */type wall;/' \
            "$C/system/blockMeshDict"
        grep -q "type cyclic" "$C/system/blockMeshDict" \
            && { echo "FAIL: the control's mesh still names a cyclic"; return 1; }
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type noSlip; }/' "$C/0/U"
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type zeroGradient; }/' "$C/0/alpha.water"
        sed -i 's/"(left|right)" { type cyclic; }/"(left|right)" { type fixedFluxPressure; value uniform 0; }/' \
            "$C/0/p_rgh"
        # ...and the turbulence fields, where the control has them. A walled pair gets the walls'
        # own condition, so the control is the same closure with the periodic faces closed off.
        for fld in k omega nut; do
            [ -f "$C/0/$fld" ] || continue
            wall=$(sed -n 's/^    walls  *{ \(.*\) }$/\1/p' "$C/0/$fld")
            [ -n "$wall" ] || { echo "FAIL: the control cannot read $fld's wall entry"; return 1; }
            python3 - "$C/0/$fld" "$wall" <<'PYEOF' || return 1
import sys
p, w = sys.argv[1], sys.argv[2]
s = open(p).read()
old = '    "(left|right)" { type cyclic; }'
assert s.count(old) == 1, p
open(p, 'w').write(s.replace(old, '    "(left|right)" { %s }' % w))
PYEOF
        done
        grep -q "type cyclic" "$C/0/U" "$C/0/alpha.water" "$C/0/p_rgh" "$C"/0/k "$C"/0/omega "$C"/0/nut 2>/dev/null \
            && { echo "FAIL: the control's fields still name a cyclic"; return 1; }
    fi
    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '17')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
PYEOF
    # THE ORACLE IS CACHED FROM HERE -- see tests/of_oracle_cache.sh. Everything above is staging;
    # everything below is real OpenFOAM, which is what a re-run should not pay for twice.
    local key
    key=$(oracleKey "$C" "interfoam_cyclic" "$profile" "$STEPS" "$DT")
    if oracleRestore "$C" "$key" "$END"; then
        echo "OpenFOAM's $STEPS steps of deltaT $DT to t = $END reused from the oracle cache   [$profile]"
        return 0
    fi
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) \
        || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    oracleStore "$C" "$key"
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

for p in cyclic walls explicitMules explicitWalls jump outer outerControl \
         sst sstWalls les lesWalls sstCN lesCN sstLim sstLimWalls sstLimU sstLimUWalls \
         sstLimDiv sstLimDivWalls sstLsq sstLsqWalls sstLimUpw sstLimUpwWalls; do
    stage "$p" || { echo "interfoam_cyclic_vs_openfoam: staging failed"; exit 1; }
done

# the oracle took the path: OpenFOAM's own mesh carries the pair on one arm and not on the other
grep -q "type  *cyclic" "$W/cyclic/constant/polyMesh/boundary" \
    || { echo "FAIL: OpenFOAM's mesh has no cyclic patch"; exit 1; }
grep -q "type  *cyclic" "$W/walls/constant/polyMesh/boundary" \
    && { echo "FAIL: the control's mesh still has a cyclic patch"; exit 1; }
grep -q "MULESCorr       no;" "$W/explicitMules/system/fvSolution" \
    || { echo "FAIL: the explicit profile ran with MULESCorr still on"; exit 1; }
grep -q "MULESCorr       yes;" "$W/cyclic/system/fvSolution" \
    || { echo "FAIL: the shipped profile no longer sets MULESCorr"; exit 1; }

rc=0
"$BIN" "$W/cyclic" "$W/cyclic/0" "$W/cyclic/$END" "$STEPS" "$W/cyclic/log.interFoam" \
       "$W/walls/$END" MULESCorr || rc=1
# ...and the same case with MULESCorr off, which is a different alpha equation and a different place
# for the pair's nHatf to come from
"$BIN" "$W/explicitMules" "$W/explicitMules/0" "$W/explicitMules/$END" "$STEPS" \
       "$W/explicitMules/log.interFoam" "$W/explicitWalls/$END" explicit || rc=1
# ...and with a JUMP on the pair: a porousBafflePressure, which the matrix, the flux and the gradient
# all have to carry. ITS control is the PLAIN-CYCLIC run -- the pair coupled and nothing else -- so
# what the control measures is the jump itself and not the coupling.
"$BIN" "$W/jump" "$W/jump/0" "$W/jump/$END" "$STEPS" \
       "$W/jump/log.interFoam" "$W/cyclic/$END" jump || rc=1
# ...and with THREE PIMPLE OUTER CORRECTORS. Its control is the SAME case with one, which is a
# different answer and is what the device loop ran before this was wired.
"$BIN" "$W/outer" "$W/outer/0" "$W/outer/$END" "$STEPS" \
       "$W/outer/log.interFoam" "$W/outerControl/$END" outer || rc=1
# ...and A TURBULENCE CLOSURE ACROSS THE PAIR, one profile per closure. kEpsilon was the only one the
# case reader let through; these are kOmegaSST and LES kEqn, each against its own walled control, and
# each comparing k (and omega, and nut) as well as the three shared fields -- the closure is what is
# under test here, and a defect confined to it reaches U only through nuEff.
"$BIN" "$W/sst" "$W/sst/0" "$W/sst/$END" "$STEPS" \
       "$W/sst/log.interFoam" "$W/sstWalls/$END" sst || rc=1
"$BIN" "$W/les" "$W/les/0" "$W/les/$END" "$STEPS" \
       "$W/les/log.interFoam" "$W/lesWalls/$END" les || rc=1
# ...and the same two under CRANKNICOLSON, whose control is the Euler run of the same case: the
# closure's ddt comes through ddtSchemes, so the scheme has to reach k and omega and not only U.
"$BIN" "$W/sstCN" "$W/sstCN/0" "$W/sstCN/$END" "$STEPS" \
       "$W/sstCN/log.interFoam" "$W/sst/$END" sstCN || rc=1
"$BIN" "$W/lesCN" "$W/lesCN/0" "$W/lesCN/$END" "$STEPS" \
       "$W/lesCN/log.interFoam" "$W/les/$END" lesCN || rc=1
# ...and the LIMITED schemes across the pair, which nothing else here exercises: `Gauss limitedLinear
# 1` for k and omega (the limiter's own gradient) and `cellLimited Gauss linear 1` on grad(k) and
# grad(omega) (the limiter's range). Its control is the same case with the pair two walls.
"$BIN" "$W/sstLim" "$W/sstLim/0" "$W/sstLim/$END" "$STEPS" \
       "$W/sstLim/log.interFoam" "$W/sstLimWalls/$END" sstLim || rc=1
# ...and the limited DIV SCHEME across the pair, which is a different thing from the limited gradient
# above: `Gauss limitedLinear 1` on div(phi,k) and div(phi,omega). OpenFOAM computes the scheme's
# weights as a surfaceScalarField and a coupled patch takes its boundaryField
# (gaussConvectionScheme.C:105-108, `const fvsPatchScalarField& pw = weights.boundaryField()[patchi]`),
# with coupledFvPatchField's valueInternalCoeffs(w) = w and valueBoundaryCoeffs(w) = 1 - w. brae had
# neither half: fvm::div threw on a coupled patch the moment it was given a scheme's weights, and the
# device assembler refused the combination by name -- so a case naming the scheme could not run at all
# on a mesh with a pair. Its control is the same case with the pair two walls.
"$BIN" "$W/sstLimDiv" "$W/sstLimDiv/0" "$W/sstLimDiv/$END" "$STEPS" \
       "$W/sstLimDiv/log.interFoam" "$W/sstLimDivWalls/$END" sstLimDiv || rc=1
# ...and the LEAST-SQUARES gradient across the pair, on both arms: `grad(k) leastSquares` and
# `grad(omega) leastSquares`, which each loop refused outright until this unit. Its control is the same
# case with the pair two walls.
"$BIN" "$W/sstLsq" "$W/sstLsq/0" "$W/sstLsq/$END" "$STEPS" \
       "$W/sstLsq/log.interFoam" "$W/sstLsqWalls/$END" sstLsq || rc=1
# ...and the same with grad(U) limited too, HOST ONLY: the device momentum's limiter does not carry
# the pair and refuses by name (armed in tests/interfoam_refusals.sh).
"$BIN" "$W/sstLimU" "$W/sstLimU/0" "$W/sstLimU/$END" "$STEPS" \
       "$W/sstLimU/log.interFoam" "$W/sstLimUWalls/$END" sstLimU || rc=1
# ...and the LIMITER OF THE DEFERRED CORRECTION across the pair: the same limited grad(U) with the
# momentum on `Gauss linearUpwind grad(U)`, so the correction exists and is limited. The device arm
# refused this by name until the limiter was given the pair's neighbour cells.
"$BIN" "$W/sstLimUpw" "$W/sstLimUpw/0" "$W/sstLimUpw/$END" "$STEPS" \
       "$W/sstLimUpw/log.interFoam" "$W/sstLimUpwWalls/$END" sstLimUpw || rc=1
# ...AND THE ASSEMBLED SYSTEM ACROSS THE PAIR, host arm against device arm, at the first closure call.
#
# WHY NOT AGAINST OPENFOAM DIRECTLY, as the waterChannel and damBreak assembly gates are: OpenFOAM's
# instrument writes the internal faces and the folded boundary, and a coupled patch's coefficient is
# neither -- it lives in boundaryCoeffs and is multiplied by the NEIGHBOUR's psi every sweep. The chain
# here is the host arm == OpenFOAM (the sstLimDiv arm above, fields at 2.0e-13) plus the device arm ==
# the host arm (this).
#
# WHY FROM A SPUN-UP FIELD: at t = 0 this fixture's k and omega are uniform, so NVDTVD's `gradf` is a
# cancellation on every face and the limiter is decided by the last bit -- the two arms then differ on
# 4 of 40 pair faces by up to 0.5 and the comparison measures the tie rather than the port. OpenFOAM's
# own ten steps above are the spin-up; its last write becomes the `0` of this one-step case.
#
# WHAT IT COMPARES: the internal-face off-diagonals, D and Src on the rows setValues does not pin,
# those rows' implied value, and THE PAIR'S OWN off-diagonal -- negated, because brae applies it as
# Apsi[own] += ifCoeff*psi[nbr] where OpenFOAM keeps boundaryCoeffs. Its CONTROL is the same comparison
# against the device's UPWIND run of the same fixture, which must miss.
assembly()
{
    local dev="$W/assembly"
    rm -rf "$dev"
    cp -r "$W/sstLimDiv" "$dev" || return 1
    rm -rf "$dev/0"
    mv "$dev/$END" "$dev/0" || { echo "FAIL: the spin-up wrote no $END to promote"; return 1; }
    rm -rf "$dev/0/uniform"
    # OpenFOAM WRITES the wall functions' full dictionaries, including entries brae refuses by name
    # (`blending stepwise` on nut). They are defaults restated, not a change of case.
    sed -i '/blending/d' "$dev/0/nut" "$dev/0/omega" 2>/dev/null
    ASMEND=$(python3 -c "print('%.10g' % float('$DT'))") python3 - "$dev" <<'PYEOF' || { echo "FAIL: the one-step controlDict"; return 1; }
import os, re, sys
c = sys.argv[1] + '/system/controlDict'
s = open(c).read()
s, n = re.subn(r'^endTime\s.*', 'endTime         %s;' % os.environ['ASMEND'], s, flags=re.M)
assert n == 1
open(c, 'w').write(s)
PYEOF
    # ...and the same case with the scheme back to upwind, for the control
    local up="$W/assemblyUpwind"
    rm -rf "$up"
    cp -r "$dev" "$up" || return 1
    sed -i 's/"div\\(phi,(k|omega)\\)".*/"div\\(phi,(k|omega)\\)"  Gauss upwind;/' "$up/system/fvSchemes"
    grep -q "Gauss upwind;" "$up/system/fvSchemes" || { echo "FAIL: the control did not go back to upwind"; return 1; }

    local hd="$dev/dump.host" dd="$dev/dump.device" ud="$up/dump.device"
    rm -rf "$hd" "$dd" "$ud"
    ( cd "$dev" && BRAE_SST_DUMP_DIR="$hd" BRAE_SST_DUMP_ITER=1 "$BIN_SOLVER" -case "$dev" > log.asm.host 2>&1 ) \
        || { echo "FAIL: the host arm did not run [assembly]"; tail -5 "$dev/log.asm.host"; return 1; }
    ( cd "$dev" && BRAE_SST_DUMP_DIR="$dd" BRAE_SST_DUMP_ITER=1 "$BIN_SOLVER" -case "$dev" -device > log.asm.dev 2>&1 ) \
        || { echo "FAIL: the device arm did not run [assembly]"; tail -5 "$dev/log.asm.dev"; return 1; }
    ( cd "$up" && BRAE_SST_DUMP_DIR="$ud" BRAE_SST_DUMP_ITER=1 "$BIN_SOLVER" -case "$up" -device > log.asm.dev 2>&1 ) \
        || { echo "FAIL: the upwind control did not run [assembly]"; tail -5 "$up/log.asm.dev"; return 1; }
    grep -qi "BRAE_CYC_LIMITED_DIAG" "$dev/log.asm.dev" \
        && { echo "FAIL: the device arm ran under the diagnostic bypass, not on its own"; return 1; }

    python3 "$(dirname "$0")/assembly_compare.py" "$hd/host" "$dd/cuda" "$dev/constant/polyMesh" \
        sstHostCuda match "A DEVICE against the HOST arm, limitedLinear across the pair" \
        && echo "  ok:   the device assembles the host's system, pair included" \
        || { echo "  FAIL: the device assembles the host's system, pair included"; return 1; }
    python3 "$(dirname "$0")/assembly_compare.py" "$hd/host" "$ud/cuda" "$dev/constant/polyMesh" \
        sstHostCuda differ "B CONTROL: the host's limitedLinear against the device's UPWIND" \
        && echo "  ok:   CONTROL: the comparison can witness the scheme across the pair" \
        || { echo "  FAIL: CONTROL: the comparison can witness the scheme across the pair"; return 1; }
    return 0
}

if [ "$HAVE_GPU" = 1 ]; then
    assembly || rc=1
else
    echo "  (no GPU: the assembled-system arm is not exercised)"
fi

echo "interfoam_cyclic_vs_openfoam: rc $rc"
exit $rc
