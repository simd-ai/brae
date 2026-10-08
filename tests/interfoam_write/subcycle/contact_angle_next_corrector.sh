#!/usr/bin/env bash
# The gate: on a contact angle the corrector that follows a curvature pass reads alpha's patch values AS THE PASS
# LEFT THEM, held to OpenFOAM on laminar/capillaryRise restaged with a second alpha corrector, and with MULESCorr.
# mixture.correct() ends, on a contact angle, in `acap.gradient() = ...; acap.evaluate()`
# (interfaceProperties.C:101-102): the wall's gradient is set from the angle and the patch evaluated, AFTER the
# pass's own cell gradient has read the values. The next corrector builds its flux on what that evaluate left
# (alphaEqn.H:164-176; vanLeer's limiter takes fvc::grad(alpha1) on them). The device loop's hooks upload the
# patch values BEFORE the pass -- rightly, the pass's kernels read that state -- and the loop took the host's
# values again only at the top of a sub-step: a second corrector of the same sub-step, and the first corrector
# after the MULESCorr pre-solve's pass, built their flux on the values from before the pass.
# FOUND 2026-10-06 by a reviewer reading the alpha step for another change; no shipped tutorial reaches it
# (capillaryRise, the one case with a contact angle: nAlphaCorr 1, no MULESCorr), so the rows are that tutorial
# restaged. MEASURED at STEPS pinned steps, worst file but the continuity sum:
#   two correctors, explicit      host 2.8e-14   device 2.8e-14   before the fix alpha 3.1e-03, p_rgh 2.7e-02
#   MULESCorr, one corrector      host 3.8e-13   device 2.6e-13   before the fix alpha 3.7e-04, p_rgh 4.0e-03
#   MULESCorr, two correctors     host 4.9e-13   device 3.7e-13   before the fix alpha 3.5e-04, p_rgh 4.0e-03
# The MULESCorr row with ONE corrector is also the one configuration where the pass AFTER the sub-cycle follows a
# corrector that did not relax under MULESCorr (alphaEqn.H:196-204): it took the evaluating hook there until
# subcycle/final_mixture_no_evaluate.sh's fix, and no shipped case or other row runs that.
# The CONTROL puts the old reading back (BRAE_CONTROL_CONTACT_ANGLE_VALUES_BEFORE_PASS=1): each row's device arm
# must leave its bound by orders.
# DOES NOT CLAIM the pass after the sub-cycle (subcycle/final_mixture_no_evaluate.sh), nor a contact angle on a
# mesh that moves or has a coupled pair -- no fixture.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
STEPS=40
# a decade above the measured worst of each row
BOUND_EXPLICIT=3e-13
BOUND_MULESCORR=5e-12
wcase capillaryRise of > "$W/cn_stage.txt" 2>&1
[ -d "$W/w_of_capillaryRise" ] || { say "capillaryRise did not stage" FAIL; finish "the next corrector"; }
grep -q "limit  *gradient;" "$W/w_of_capillaryRise/0/alpha.water" \
    || { say "PREMISE  capillaryRise's contact angle limits its gradient" FAIL; finish "the next corrector"; }
# the MULESCorr row's entries: the solver the pre-solve needs, under a key that matches alpha.waterFinal too
MC='\n\1MULESCorr       yes;\n\1nLimiterIter    5;\n\1solver          smoothSolver;'
MC="$MC"'\n\1smoother        symGaussSeidel;\n\1tolerance       1e-13;\n\1relTol          0;/'
# mc <correctors>: the sed expression of the MULESCorr row with that many
mc()
{
    echo 's/^( *)alpha\.water$/\1"alpha.water.*"/;s/^( *)nAlphaCorr( +)1;/\1nAlphaCorr\2'"$1"';'"$MC"
}
# row <tag> <sed expression for system/fvSolution>: the restaged case run by OpenFOAM, then the host arm, the
# device arm and the device arm with the control, each compared with OpenFOAM's last step. An arm that does
# not run ends the gate with runbrae's own message
row()
{
    local tag="$1" o="$W/cn_of_$1" arm e ot
    rm -rf "${o:?}"
    mkdir -p "$o"
    cp -r "$W/w_of_capillaryRise/0" "$W/w_of_capillaryRise/constant" "$W/w_of_capillaryRise/system" "$o/"
    sed -i -E "$2" "$o/system/fvSolution"
    local dt
    dt=$(sed -n -E 's/^deltaT +([^;]+);/\1/p' "$o/system/controlDict")
    sed -i -E "s/^endTime .*/endTime         $(python3 -c "print('%.12g' % ($STEPS*$dt))");/" \
        "$o/system/controlDict"
    sed -i -E "s/^writeInterval .*/writeInterval   $STEPS;/" "$o/system/controlDict"
    runof "$o"
    ot=$(timedirs "$o")
    [ "$(echo $ot | wc -w)" = 1 ] || { say "PREMISE  OpenFOAM writes the last step [$tag: $ot]" FAIL; return 1; }
    for arm in host device control; do
        e="$W/cn_${tag}_$arm"
        rm -rf "${e:?}"
        mkdir -p "$e"
        cp -r "$o/0" "$o/constant" "$o/system" "$e/"
        case $arm in
            host) runbrae "$e" host ;;
            device) runbrae "$e" device ;;
            control) runbrae "$e" device BRAE_CONTROL_CONTACT_ANGLE_VALUES_BEFORE_PASS=1 ;;
        esac
        python3 "$CMP" "$o" "$e" $ot > "$e/cmp.txt" 2>&1
    done
}
# gap <tag> <arm>: the worst file's relative gap to OpenFOAM, the continuity sum apart; "-" on a structure failure
gap()
{
    python3 - "$W/cn_$1_$2/cmp.txt" <<'PY'
import json, sys
lines = [l for l in open(sys.argv[1]) if l.startswith('RESULT ')]
if not lines:
    print('-')
    sys.exit(0)
r = json.loads(lines[-1][7:])
rel = [v['rel'] for k, v in r['files'].items() if not k.endswith('cumulativeContErr')]
print('-' if r['structure'] != 0 or not rel else '%.1e' % max(rel))
PY
}
# said <tag>: the device arm says it takes the values the pass left and is no control; the control says it is one
said()
{
    grep -q "the next corrector reads alpha's patch values as the curvature pass left them" \
            "$W/cn_$1_device/log.brae" \
        && ! grep -q "CONTROL MODE" "$W/cn_$1_device/log.brae" \
        && grep -q "CONTROL MODE: after a curvature pass the next corrector reads a contact angle's patch" \
            "$W/cn_$1_control/log.brae"
}
row explicit 's/^( *nAlphaCorr +)1;/\12;/' || finish "the next corrector"
grep -q "nAlphaCorr  *2;" "$W/cn_of_explicit/system/fvSolution" \
    && ! grep -q "MULESCorr" "$W/cn_of_explicit/system/fvSolution" \
    || { say "PREMISE  the explicit row has two correctors and no MULESCorr" FAIL; finish "the next corrector"; }
row mulescorr1 "$(mc 1)" || finish "the next corrector"
row mulescorr2 "$(mc 2)" || finish "the next corrector"
grep -q "MULESCorr  *yes;" "$W/cn_of_mulescorr1/system/fvSolution" \
    && grep -q "nAlphaCorr  *1;" "$W/cn_of_mulescorr1/system/fvSolution" \
    && grep -q "MULESCorr  *yes;" "$W/cn_of_mulescorr2/system/fvSolution" \
    && grep -q "nAlphaCorr  *2;" "$W/cn_of_mulescorr2/system/fvSolution" \
    || { say "PREMISE  the MULESCorr rows have MULESCorr, one corrector and two" FAIL; finish "the next corrector"; }
gh=$(gap explicit host)
gd=$(gap explicit device)
what="[host] and [device] capillaryRise, two explicit correctors, $STEPS steps: every file within $gh and $gd of"
what="$what OpenFOAM's"
within "$gh" "$BOUND_EXPLICIT" && within "$gd" "$BOUND_EXPLICIT" && said explicit && say "$what" ok \
    || say "$what" FAIL
m1h=$(gap mulescorr1 host)
m1d=$(gap mulescorr1 device)
m2h=$(gap mulescorr2 host)
m2d=$(gap mulescorr2 device)
what="[host] and [device] the same with MULESCorr: one corrector within $m1h and $m1d, two (the second relaxed)"
what="$what within $m2h and $m2d"
within "$m1h" "$BOUND_MULESCORR" && within "$m1d" "$BOUND_MULESCORR" && said mulescorr1 \
    && within "$m2h" "$BOUND_MULESCORR" && within "$m2d" "$BOUND_MULESCORR" && said mulescorr2 \
    && say "$what" ok || say "$what" FAIL
ce=$(gap explicit control)
c1=$(gap mulescorr1 control)
c2=$(gap mulescorr2 control)
what="CONTROL  the next corrector reading the values from before the pass: $ce, $c1 and $c2 of OpenFOAM's"
above "$ce" 1e-6 && above "$c1" 1e-6 && above "$c2" 1e-6 && say "$what" ok || say "$what" FAIL
finish "on a contact angle the corrector after a curvature pass reads the patch values the pass left, as OpenFOAM's"
