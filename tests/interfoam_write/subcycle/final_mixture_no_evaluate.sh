#!/usr/bin/env bash
# The gate: the device loop's mixture.correct() after the alpha sub-cycle evaluates NO alpha patch, held to
# OpenFOAM on laminar/capillaryRise run for STEPS pinned steps -- long enough for the contact angle's
# `limit gradient` clamp to bite, which the two steps of the tutorial's W row and the five of
# interfoam_capillaryrise_vs_openfoam are not.
# THE DEFECT, found 2026-10-05 by running the case longer for another question. interFoam.C:154 calls
# mixture.correct() after alphaEqnSubCycle.H: calcNu() and calculateK(), which evaluates a contact-angle
# patch inside correctContactAngle and nothing else (interfaceProperties.C:102). The device loop went through
# the hook that evaluates alpha's patches first, on the reading that an evaluate on unchanged cells changes
# nothing. A contact angle with `limit gradient` clamps its gradient against the patch's own value, so a
# second evaluate is not the first again. MEASURED at pinned solves, the host arm 2.0e-15 from OpenFOAM
# throughout: the device arm the same for eleven steps, then alpha 1.3e-03 from it in one, and at the fortieth
# alpha 2.1e-03, U 1.1e-03, p_rgh 1.9e-02; without that evaluate 2.0e-15, 4.8e-15, 2.8e-14.
# ALSO HERE, AS A COUNT: MULES::explicitSolve opens with psi.correctBoundaryConditions() (MULESTemplates.C:168)
# and the device's explicit corrector made that evaluate only where a wave model supplies the values. It is
# made now at every corrector. The second line holds its CALLS on this row -- two a step, one a sub-cycle, from
# the phase table of a timed pair, none with BRAE_CONTROL_MULES_OPENING_EVALUATE_SKIPPED=1 -- and prints, without
# asserting it, what the row shows of its effect: NOTHING. This row cannot tell that evaluate from its absence
# by construction: the values it moves are the contact-angle wall's, a fixed-gradient patch the limiter does
# not read (fixesValue is false, MULESTemplates.C:338), and the closing evaluate lands on the same state either
# way. What that evaluate does to an answer is held by tests/test_device_alpha_opening_evaluate.cu, on a fixture
# built for it.
# The CONTROL puts the evaluate back into the pass after the sub-cycle
# (BRAE_CONTROL_FINAL_MIXTURE_EVALUATES=1): the device arm must leave the bound by orders.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
STEPS=40
# MEASURED 2026-10-05, worst file but the continuity sum: host 2.8e-14, device 2.8e-14 (p_rgh; alpha 2.0e-15).
# The bound is a decade above. (The opening evaluate skipped: 5.2e-14 -- printed below, held to nothing.)
BOUND=3e-13
wcase capillaryRise of > "$W/fm_stage.txt" 2>&1
[ -d "$W/w_of_capillaryRise" ] || { say "capillaryRise did not stage" FAIL; finish "the final mixture pass"; }
o="$W/fm_of"
rm -rf "${o:?}"
mkdir -p "$o"
cp -r "$W/w_of_capillaryRise/0" "$W/w_of_capillaryRise/constant" "$W/w_of_capillaryRise/system" "$o/"
dt=$(sed -n -E 's/^deltaT +([^;]+);/\1/p' "$o/system/controlDict")
sed -i -E "s/^endTime .*/endTime         $(python3 -c "print('%.12g' % ($STEPS*$dt))");/" "$o/system/controlDict"
sed -i -E "s/^writeInterval .*/writeInterval   $STEPS;/" "$o/system/controlDict"
grep -q "limit  *gradient;" "$o/0/alpha.water" \
    || { say "PREMISE  capillaryRise's contact angle limits its gradient" FAIL; finish "the final mixture pass"; }
runof "$o"
ot=$(timedirs "$o")
[ "$(echo $ot | wc -w)" = 1 ] || { say "PREMISE  OpenFOAM writes the last step [$ot]" FAIL; finish "final pass"; }
# arm <name> <host|device> [env...]: one brae run of the row, compared with OpenFOAM's; an arm that does not
# run ends the gate with runbrae's own message
arm()
{
    local e="$W/fm_$1" kind="$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    runbrae "$e" "$kind" "$@"
    python3 "$CMP" "$o" "$e" $ot > "$e/cmp.txt" 2>&1
}
# gap <name>: the worst file's relative gap to OpenFOAM, the continuity sum apart; "-" on a structure failure
gap()
{
    python3 - "$W/fm_$1/cmp.txt" <<'PY'
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
# calls <log>: the opening evaluate's hook row in the phase table, calls a step; 0 where the row is absent
calls()
{
    local c
    c=$(grep -a "hook alpha\.refreshBoundary " "$1" | head -1 | sed -n -E 's/.*\(([0-9.]+) calls\/step\).*/\1/p')
    echo "${c:-0}"
}
arm host host
arm device device
arm timed device BRAE_INTER_PHASE_TIME=1 BRAE_MULES_OPENING_EVALUATE_REPORT=1
arm skipped device BRAE_INTER_PHASE_TIME=1 BRAE_CONTROL_MULES_OPENING_EVALUATE_SKIPPED=1
arm control device BRAE_CONTROL_FINAL_MIXTURE_EVALUATES=1
gh=$(gap host)
gd=$(gap device)
what="[host] and [device] capillaryRise after $STEPS steps: every file within $gh and $gd of OpenFOAM's"
within "$gh" "$BOUND" && within "$gd" "$BOUND" \
    && grep -q "mixture.correct() after the alpha sub-cycle: made (alpha's patch .* is a contact angle" \
            "$W/fm_device/log.brae" \
    && ! grep -q "CONTROL MODE" "$W/fm_device/log.brae" && say "$what" ok || say "$what" FAIL
nsub=$(sed -n -E 's/^ *nAlphaSubCycles +([0-9]+);.*/\1/p' "$o/system/fvSolution" | head -1)
ncorr=$(sed -n -E 's/^ *nAlphaCorr +([0-9]+);.*/\1/p' "$o/system/fvSolution" | head -1)
ct=$(calls "$W/fm_timed/log.brae")
cs=$(calls "$W/fm_skipped/log.brae")
moved=$(grep -a "explicit MULES opening evaluate:" "$W/fm_timed/log.brae" | tail -1 \
        | sed -n -E 's/.*, ([0-9]+) patch values moved in all, at most [0-9]+ in one and by ([0-9.e+-]+).*/\1 by \2/p')
what="[device] the evaluate that opens the limited solve: $ct calls a step ($nsub sub-cycles of $ncorr), $cs"
what="$what skipped; it moves ${moved:-?} at most and the skipped arm is $(gap skipped) of OpenFOAM's (not held)"
python3 -c "import sys; sys.exit(0 if abs($ct - ${nsub:-0}*${ncorr:-0}) < 1e-9 and ${nsub:-0}*${ncorr:-0} > 0 \
        and $cs == 0 else 1)" \
    && grep -q "explicit MULES: alpha's patches are evaluated at the top of the limited solve" \
            "$W/fm_timed/log.brae" \
    && grep -q "CONTROL MODE: explicit MULES does not evaluate alpha's patches at the top" \
            "$W/fm_skipped/log.brae" \
    && say "$what" ok || say "$what" FAIL
gc=$(gap control)
what="CONTROL  the pass after the sub-cycle evaluating alpha's patches first: $gc of OpenFOAM's"
grep -q "CONTROL MODE: the mixture.correct() after the alpha sub-cycle evaluates alpha's patches" \
        "$W/fm_control/log.brae" \
    && above "$gc" 1e-6 && say "$what" ok || say "$what" FAIL
finish "the device loop's mixture.correct() after the sub-cycle evaluates no alpha patch, as OpenFOAM's"
