#!/usr/bin/env bash
# One arm of the write gate (tests/interfoam_write/lib.sh has the gate's header and helpers).
. "$(dirname "$0")/../lib.sh"
wcase sloshingTank2D
wcase weirOverflow

# makeRelative at :70). On a solidBody move it is the swept volumes' residue, which both codes compute from
# the same points, so it agrees far below the floor judge() allows: sloshingTank2D 1.2e-03 (host) and
# 1.3e-03 (device) relative, where the relative flux -- this writer's first form -- reads 100% off. Held
# relative, with no floor, one decade above the measured worst. CONTROL: BRAE_CONTROL_CONTINUITY_RELATIVE=1.
contrel()   # contrel <case dir> -- the worst relative cumulativeContErr gap against OpenFOAM
{
    python3 - "$W/w_of_sloshingTank2D" "$1" <<'PY'
import re, sys
worst = 0.0
for t in ('0.01', '0.02'):
    o = float(re.search(r'^value\s+(\S+);', open('%s/%s/uniform/cumulativeContErr' % (sys.argv[1], t)).read(), re.M).group(1))
    b = float(re.search(r'^value\s+(\S+);', open('%s/%s/uniform/cumulativeContErr' % (sys.argv[2], t)).read(), re.M).group(1))
    worst = max(worst, abs(b - o) / abs(o))
print('%.3e' % worst)
PY
}
if [ -d "$W/w_br_sloshingTank2D_host" ]; then
    for arm in $ARMS; do
        r=$(contrel "$W/w_br_sloshingTank2D_$arm")
        python3 -c "import sys; sys.exit(0 if $r < 2e-2 else 1)" \
            && say "ARM M  [$arm] sloshingTank2D's cumulativeContErr, the absolute flux's, within 2e-2 ($r)" ok \
            || say "ARM M  [$arm] sloshingTank2D's cumulativeContErr, the absolute flux's, within 2e-2 ($r)" FAIL
    done
    d="$W/w_ctl_contrel"
    mkdir -p "$d"
    cp -r "$W/w_of_sloshingTank2D/0" "$W/w_of_sloshingTank2D/constant" "$W/w_of_sloshingTank2D/system" "$d/"
    runbrae "$d" host BRAE_CONTROL_CONTINUITY_RELATIVE=1
    r=$(contrel "$d")
    python3 -c "import sys; sys.exit(0 if $r > 0.5 else 1)" \
        && say "CONTROL  BRAE_CONTROL_CONTINUITY_RELATIVE=1 puts it off by more than half ($r)" ok \
        || say "CONTROL  BRAE_CONTROL_CONTINUITY_RELATIVE=1 puts it off by more than half ($r)" FAIL
else
    say "ARM M  sloshingTank2D did not run in arm W" FAIL
fi

if [ -d "$W/w_of_weirOverflow" ]; then
    d="$W/w_ctl_weir"
    mkdir -p "$d"
    cp -r "$W/w_of_weirOverflow/0" "$W/w_of_weirOverflow/constant" "$W/w_of_weirOverflow/system" "$d/"
    runbrae "$d" host BRAE_CONTROL_ALPHA_OLD_START=1
    python3 "$CMP" "$W/w_of_weirOverflow" "$d" $(timedirs "$W/w_of_weirOverflow") > "$W/cmp_w_ctl.txt" 2>&1
    # weirOverflow's host bound in W_CASES
    judge "control start-only" "$W/cmp_w_ctl.txt" 5e-12 | grep -q "over the bound: .*alpha.water_0" \
        && say "CONTROL  BRAE_CONTROL_ALPHA_OLD_START=1 puts weirOverflow's alpha.water_0 over the bound" ok \
        || say "CONTROL  BRAE_CONTROL_ALPHA_OLD_START=1 puts weirOverflow's alpha.water_0 over the bound" FAIL
fi

# P: the continuity error of the mesh update's CorrectPhi (correctPhi.H:11) counts. correctPhi defaults

finish "arm M: a moving mesh's continuity error"
