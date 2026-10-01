#!/usr/bin/env bash
# The write gate on damBreak: arm D on RAS/waterChannel, with the construction-gradient control.
. "$(dirname "$0")/../lib.sh"
# D on RAS/waterChannel (kOmegaSST, omega and nutk wall functions, flowRateInletVelocity), three fixed
# steps of 0.1 -- and the construction-gradient control, which damBreak cannot witness: in the p_rgh
# formulation a zeroGradient-alpha wall has snGrad(rho) = 0 and OpenFOAM's gradient there IS zero. The
# inlet's fixed alpha makes it non-zero.
WCH="$TUT/multiphase/interFoam/RAS/waterChannel"
if [ -d "$WCH" ] && command -v extrudeMesh > /dev/null 2>&1; then
    stage "$WCH" "$W/of_w" 0.3 timeStep 3 0 adjustTimeStep=no deltaT=0.1 || exit 1
    runof "$W/of_w"
    [ "$(timedirs "$W/of_w")" = "0.3 " ] || say "premise: OpenFOAM waterChannel writes {0.3} [$(timedirs "$W/of_w")]" FAIL
    python3 - "$W/of_w/0.3/p_rgh" <<'PY' && say "fixture witnesses: OpenFOAM's waterChannel p_rgh gradient is non-zero" ok \
                                     || say "fixture witnesses: OpenFOAM's waterChannel p_rgh gradient is non-zero" FAIL
import re, sys
grads = re.findall(r'gradient\s+nonuniform List<scalar>\s*\d*\s*\(([^)]*)\)', open(sys.argv[1]).read())
g = [abs(float(x)) for s in grads for x in s.split()]
print('      max|p_rgh gradient| %.3e over %d faces' % (max(g) if g else 0.0, len(g)))
sys.exit(0 if g and max(g) > 1 else 1)
PY
    for arm in $ARMS; do
        stage "$WCH" "$W/br_w_$arm" 0.3 timeStep 3 0 adjustTimeStep=no deltaT=0.1 || exit 1
        runbrae "$W/br_w_$arm" "$arm"
        python3 "$CMP" "$W/of_w" "$W/br_w_$arm" 0.3 > "$W/cmp_w_$arm.txt" 2>&1
        judge "waterChannel $arm" "$W/cmp_w_$arm.txt" "$BOUND_FIELDS" \
            && say "ARM D  [$arm] waterChannel: every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" ok \
            || { say "ARM D  [$arm] waterChannel: every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" FAIL; grep -v RESULT "$W/cmp_w_$arm.txt" | grep -B1 "^      " | head -20; }
    done
    stage "$WCH" "$W/ctlg" 0.3 timeStep 3 0 adjustTimeStep=no deltaT=0.1 || exit 1
    runbrae "$W/ctlg" host BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT=1
    python3 "$CMP" "$W/of_w" "$W/ctlg" 0.3 > "$W/cmp_ctlg.txt" 2>&1
    judge "control gradient" "$W/cmp_ctlg.txt" "$BOUND_FIELDS" | grep -q "over the bound: 0.3/p_rgh" \
        && say "CONTROL  BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT=1 puts p_rgh over the bound" ok \
        || say "CONTROL  BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT=1 puts p_rgh over the bound" FAIL
else
    say "waterChannel arm: tutorial or extrudeMesh missing -- the gradient control cannot run" FAIL
fi
finish "arm D on waterChannel: structure, values and the stored gradient"
