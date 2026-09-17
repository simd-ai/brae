#!/usr/bin/env bash
# brae's interFoam against REAL OpenFOAM's interFoam, on damBreak, field by field.
#
# THE METHOD IS "EXACTLY N IDENTICAL STEPS", not "run both to the end". An adaptive time step makes the
# two solvers take DIFFERENT steps the moment their Courant numbers differ by anything at all, and then
# every field is compared at a different physical time -- which produces a disagreement that looks like
# a discretisation error and is actually a clock. So the case is rewritten with `adjustTimeStep no` and
# a fixed deltaT, both run the same count, and the comparison is at one instant.
#
# The case is damBreak's own, prepared the way the tutorial does (blockMesh, then setFields), because
# it ships 0.orig and no mesh.
#
# IT RUNS AT TWO TIME STEPS, because the two things it measures want opposite fixtures. At dt 1e-4 the
# interface barely moves (3.7e-03) and the fields agree to 3.6e-14, which is where the tight FIELD
# bounds live -- but the alpha pre-solve is then so diagonally dominant (Co ~ 1e-3) that every solver
# lands on the exact solution in one iteration: measured, PBiCGStab's final residuals were 2.6e-03 from
# OpenFOAM's symGaussSeidel's, beside the device's honest 1.5e-03, so a solver-log arm there cannot
# tell solvers apart. At dt 5e-3 the interface moves 0.69, OpenFOAM's smoother takes 0, 5, 2, 2, 2
# sweeps, brae takes the same and leaves its final residuals to 9e-10 -- and the PBiCGStab CONTROL gets
# two counts of five and is 100% out. The `bigstep` profile is where those arms are asserted.
#
# AND A THIRD PROFILE, `inflow`, for a term neither shipped tutorial exercises. rho is built from an
# expression, so its patches are `calculated` and fvc::snGrad(rho) on a patch is
# deltaCoeffs*(rho_b - rho_cell); brae took it from a zeroGradient copy, which is 0 everywhere. On a
# fixedFluxPressure wall that cancels through constrainPressure, and everywhere else on damBreak and
# capillaryRise alpha's patch value equals the cell's -- so it was carried as LATENT. Setting the
# atmosphere's inletValue to 1 makes water enter over air cells: OpenFOAM's boundary snGrad(rho) is then
# 1.57e+05 on 19 of 46 faces, phig there is 1.66e-02 against a phiHbyA of 2.5e-06, max|U| goes 0.19 ->
# 20 m/s in ONE step, and brae was 100% out in alpha, p_rgh and U from the second step on.
# THREE steps only: the fixture is violent enough that round-off grows 700x a step (device against
# host, every solve tightened: 1e-12 after two steps, 7e-10 after three), so a longer run measures the
# conditioning and not the term.
#
# AND A FOURTH, `prevcorr`: `alphaApplyPrevCorr yes`, which caches the compression flux the correctors
# ended on and applies it, limited, as the NEXT step's first guess (alphaEqn.H:133-150, :228-236). Three
# shipped tutorials set it and brae refuses all three for other reasons (RAS, moving meshes, LTS), so
# no tutorial can gate it; OpenFOAM honours the switch on any MULESCorr case, and damBreak is one. It
# runs at the big step, where the interface moves enough for last step's correction to matter, and the
# staging asserts OpenFOAM's log says "Applying the previous iteration compression flux" -- an oracle
# that never took the path would agree with a brae that ignored the switch.
#
# AND `outflow`, for ONE ARGUMENT of that path. MULES::correct's flux argument feeds a single test, the
# boundary outlet test `(phi_b + phiCorr_b) > 0`, and OpenFOAM passes the ALPHA flux alphaPhi10 where
# brae's host passed the volumetric phiCN. It took three fixtures to measure. On damBreak the two agree
# to five digits (the cached correction's boundary half is zero unless water LEAVES through an outflow
# face). With the water column raised to the atmosphere it does leave, and at dt 5e-3 the two tests
# decide differently on up to 13 faces a step -- and the answer is STILL bit-identical, because the cell
# limiter returns 1 there. It needs the limiter biting as well: the same fixture at dt 1e-2, four steps,
# where brae with phiCN is 5.5e-03 of alpha and 2.2% of U from OpenFOAM and with alphaPhi10 4.6e-13.
# The test re-runs the host with BRAE_CONTROL_PREVCORR_PHICN set and requires THAT to fail.
#
# AND THREE PIMPLE CONTROLS damBreak DOES NOT USE: `nouter` (nOuterCorrectors 2), `nonorth`
# (nNonOrthogonalCorrectors 1) and `mompred` (momentumPredictor yes). brae's host RAN all three and
# nothing held any of them against OpenFOAM -- which is the position alphaApplyPrevCorr was in while it
# carried a wrong limiter argument. 5, 4 and 5 of the 44 shipped tutorials set them. Each profile's
# control is the big-step run without the setting: the setting has to move OpenFOAM's own answer, or a
# brae that ignored it would pass. The device loop runs neither outer nor non-orthogonal correctors and
# REFUSES both, so on those two profiles the test asserts the refusal instead of a run.
#
# AND `rhophi`: every flux-conditional condition of the atmosphere -- U's pressureInletOutletVelocity,
# p_rgh's totalPressure, alpha's inletOutlet -- given `phi rhoPhi;`. OpenFOAM's conditions look their
# flux up BY NAME, three shipped tutorials name rhoPhi on a totalPressure top, and brae's reader kept
# no `phi` entry at all (tests/interfoam_waves_vs_openfoam.sh found it, on the two solitary-wave cases
# that write it). Those cases put the name on p_rgh only; this profile puts it on all three, so the
# velocity's and alpha's switches are held against OpenFOAM too. The DEVICE runs U's switch itself and
# reads phi there, so it REFUSES this profile, and the test asserts the refusal.
# IT FAILED THE FIRST TIME IT RAN, on two things a condition naming `phi` can never see, because
# nothing moves phi between the end of one step's pressure correctors and the next step's UEqn -- and
# the ALPHA step moves rhoPhi exactly there:
#   the push.   brae's conditions are TOLD their flux, and were last told at the end of the previous
#               step; OpenFOAM's look it up at every updateCoeffs, so UEqn's U and the first corrector's
#               p_rgh read THIS step's rhoPhi. Naming it on p_rgh alone: alpha 4.7e-10 -> 1.2e-12.
#   the value.  pressureInletOutletVelocity::updateCoeffs ends in directionMixed::evaluate(), which
#               REWRITES THE PATCH VALUE and clears the updated flag -- so UEqn's fvMatrix constructor
#               re-evaluates U's atmosphere from the new flux. brae refreshed the coefficients and kept
#               the value. Naming it on U alone: alpha 2.5e-09 before the push, 8.4e-11 after it,
#               1.2e-12 with the value re-evaluated as well.
# The control: naming rhoPhi moves OpenFOAM's OWN alpha by 1.0e-06 over these five steps.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_dambreak_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
STEPS=${STEPS:-5}
STEPS_INFLOW=${STEPS_INFLOW:-3}
DT=${DT:-1e-4}
DT_BIG=${DT_BIG:-5e-3}
DT_OUT=${DT_OUT:-1e-2}
STEPS_OUT=${STEPS_OUT:-4}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

# run_at <deltaT> <profile> [nSteps]: stage the tutorial at a FIXED step, run real OpenFOAM, run the gate.
run_at()
{
    local dt="$1" profile="$2" STEPS="${3:-$STEPS}"
    local C="$W/$profile"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    if [ "$profile" = inflow ]; then
        # WATER ENTERS OVER AIR CELLS: the one change that makes rho_b differ from rho_cell on a patch
        # where p_rgh fixes a value. setFields rewrites only the internal field, so the patch entry
        # survives it.
        sed -i 's/inletValue *uniform 0;/inletValue      uniform 1;/' "$C/0/alpha.water"
        grep -q "inletValue *uniform 1;" "$C/0/alpha.water" \
            || { echo "FAIL: the inflow fixture's inletValue was not rewritten"; return 1; }
    fi
    if [ "$profile" = rhophi ]; then
        for fld in U p_rgh alpha.water; do
            sed -i '/^ *atmosphere/,/}/ s/^\( *\)type\( .*\)$/\1type\2\n\1phi             rhoPhi;/' "$C/0/$fld"
            grep -q "phi  *rhoPhi;" "$C/0/$fld" || { echo "FAIL: $fld's atmosphere was not given phi rhoPhi"; return 1; }
        done
    fi
    case "$profile" in
        nouter)  sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 2;/' "$C/system/fvSolution"
                 grep -q "nOuterCorrectors 2;" "$C/system/fvSolution" || { echo "FAIL: nOuterCorrectors was not raised"; return 1; } ;;
        nonorth) sed -i 's/nNonOrthogonalCorrectors  *0;/nNonOrthogonalCorrectors 1;/' "$C/system/fvSolution"
                 grep -q "nNonOrthogonalCorrectors 1;" "$C/system/fvSolution" || { echo "FAIL: nNonOrthogonalCorrectors was not raised"; return 1; } ;;
        mompred) sed -i 's/momentumPredictor  *no;/momentumPredictor yes;/' "$C/system/fvSolution"
                 grep -q "momentumPredictor yes;" "$C/system/fvSolution" || { echo "FAIL: momentumPredictor was not switched on"; return 1; }
                 # damBreak names `U` only, and with one outer corrector fvMatrix::solve() selects
                 # `UFinal`: real OpenFOAM stops on this case, "Entry 'UFinal' not found". The key
                 # becomes the regex "U.*", which is how the tutorials that DO run a predictor write it.
                 sed -i 's/^\( *\)U$/\1"U.*"/' "$C/system/fvSolution"
                 grep -q '"U\.\*"' "$C/system/fvSolution" || { echo "FAIL: the U solver entry was not widened to UFinal"; return 1; } ;;
    esac
    if [ "$profile" = outflow ]; then
        # THE WATER COLUMN REACHES THE ATMOSPHERE, whose faces over it turn out to be OUTFLOW (the patch
        # fixes p_rgh, not p, so the column top sees the lower pressure): water leaves through them.
        sed -i 's/box (0 0 -1) (0.1461 0.292 1);/box (0 0 -1) (0.1461 1 1);/' "$C/system/setFieldsDict"
        grep -q "box (0 0 -1) (0.1461 1 1);" "$C/system/setFieldsDict" \
            || { echo "FAIL: the outflow fixture's water column was not raised"; return 1; }
    fi
    if [ "$profile" = prevcorr ] || [ "$profile" = prevcorrsub ] || [ "$profile" = outflow ]; then
        sed -i 's/^\( *\)MULESCorr  *yes;/\1MULESCorr       yes;\n\1alphaApplyPrevCorr yes;/' "$C/system/fvSolution"
        grep -q "alphaApplyPrevCorr yes;" "$C/system/fvSolution" \
            || { echo "FAIL: alphaApplyPrevCorr was not switched on in the staged case"; return 1; }
    fi
    if [ "$profile" = prevcorrsub ]; then
        # ...AND TWO SUB-CYCLES, because talphaPhi1Corr0 outlives the sub-cycle as well as the time step:
        # the second sub-cycle's pre-solve applies the correction the FIRST one ended on. damBreak's own
        # nAlphaSubCycles is 1, which cannot tell a cache that crosses sub-cycles from one that is reset.
        sed -i 's/nAlphaSubCycles  *1;/nAlphaSubCycles 2;/' "$C/system/fvSolution"
        grep -q "nAlphaSubCycles 2;" "$C/system/fvSolution" \
            || { echo "FAIL: nAlphaSubCycles was not raised in the staged case"; return 1; }
    fi

    # A FIXED time step, and write exactly once at step N. writePrecision 15 because the comparison is
    # against brae's fp64 and an ascii round-trip at the default 6 digits would dominate the difference.
    STEPS="$STEPS" DT="$dt" python3 - "$C" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
n  = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'^adjustTimeStep .*', 'adjustTimeStep  no;',        s, flags=re.M)
s = re.sub(r'^deltaT .*',         'deltaT          %s;' % dt,   s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         %.10g;' % (n*float(dt)), s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;',  s, flags=re.M)
# every step is written, so the `inflow` control can read the STANDARD case at its own end time
s = re.sub(r'^writeInterval .*',  'writeInterval   1;',         s, flags=re.M)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',     s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',        s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; return 1; }

    if [ "$profile" = prevcorr ] || [ "$profile" = prevcorrsub ] || [ "$profile" = outflow ]; then
        grep -q "Applying the previous iteration compression flux" "$C/log.interFoam" \
            || { echo "FAIL: OpenFOAM never applied the previous correction, so this oracle cannot gate it"; return 1; }
    fi

    local end
    end=$(python3 -c "print('%.10g' % ($STEPS*float('$dt')))")
    [ -d "$C/$end" ] || { echo "FAIL: OpenFOAM wrote no $end directory"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $dt to t = $end   [$profile]"

    # THE CONTROL CASE for the alpha solve-log arm: identical, except that the alpha entry names
    # PBiCGStab -- the solver brae's host ran in place of the case's symGaussSeidel until it had one.
    # OpenFOAM is NOT re-run on it; the test compares brae-on-PBiCGStab against
    # OpenFOAM-on-symGaussSeidel and, under `bigstep`, requires that comparison to FAIL.
    cp -r "$C" "$C.control"
    python3 - "$C.control" <<'PYEOF'
import os, re, sys
q = os.path.join(sys.argv[1], 'system/fvSolution')
t = open(q).read()
m = re.search(r'("alpha\.water\.\*"\s*\{)([^}]*)\}', t)
assert m, 'no alpha.water.* entry'
body = m.group(2)
body = re.sub(r'solver\s+\w+;', 'solver          PBiCGStab;', body)
body = re.sub(r'smoother\s+\w+;', 'preconditioner  DILU;', body)
t = t[:m.start(2)] + body + t[m.end(2):]
open(q, 'w').write(t)
PYEOF
    grep -q "PBiCGStab" "$C.control/system/fvSolution" || { echo "FAIL: the control case was not rewritten"; return 1; }

    # the `inflow` control reads the STANDARD case's OpenFOAM answer at the same instant
    local std=""
    [ "$profile" = inflow ] && std="$W/small/$end"
    # ...and the `prevcorr` control reads the SAME run without the switch
    [ "$profile" = prevcorr ] && std="$W/bigstep/$end"
    # ...and the sub-cycled one reads the un-sub-cycled one: the sub-cycle count has to be live too
    [ "$profile" = prevcorrsub ] && std="$W/prevcorr/$end"
    # ...and the three PIMPLE profiles read the big-step run without their setting
    case "$profile" in nouter|nonorth|mompred|rhophi) std="$W/bigstep/$end" ;; esac
    "$BIN" "$C" "$C/0" "$C/$end" "$STEPS" "$C/log.interFoam" "$C.control" "$profile" $std
}

rc=0
run_at "$DT" small || rc=1
run_at "$DT_BIG" bigstep || rc=1
run_at "$DT" inflow "$STEPS_INFLOW" || rc=1
run_at "$DT_BIG" prevcorr || rc=1
run_at "$DT_BIG" prevcorrsub || rc=1
run_at "$DT_OUT" outflow "$STEPS_OUT" || rc=1
run_at "$DT_BIG" nouter || rc=1
run_at "$DT_BIG" nonorth || rc=1
run_at "$DT_BIG" mompred || rc=1
run_at "$DT_BIG" rhophi || rc=1
exit $rc
