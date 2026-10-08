#!/usr/bin/env bash
# One arm of the write gate (tests/interfoam_write/lib.sh has the gate's header and helpers).
. "$(dirname "$0")/../lib.sh"
wcase waveMakerPiston

# to mesh.dynamic() (createDyMControls.H:4-7) and the wave makers leave it on; sloshingTank2D, testTubeMixer
# and sloshingCylinder switch it off, so arm M cannot see it. In arm W the pcorr is pinned to 1e-13 and
# its term is 5.6e-21 against a cumulative of 8.2e-14 -- nothing can witness it there. So this arm pins
# every solve but pcorr, which gets a loose tolerance (1e-4, relTol 0) and leaves a real error behind:
# waveMakerPiston's second correctPhi puts 3.2e-11 into OpenFOAM's 3.04e-11 at 0.01. MEASURED
# (2026-09-30), relative to OpenFOAM at 0.01: host 6.1e-05, device 5.2e-04 (the pcorr is host code on
# both arms, the p_rgh solves are not); the bound is a decade above each. At 0.005 the term is 2.1e-13 of
# 3.0e-13 and the device's p_rgh solves move the sum 6.0e-02, so only 0.01 is held.
# CONTROL: BRAE_CONTROL_NO_CORRECTPHI_CONTERR=1 leaves the term out -- -1.9990e-12, 1.07 off. FAIL-PROOF:
# the binary before this fix wrote -1.9989507883302306e-12, the control's value bit for bit.
if [ -d "$W/w_of_waveMakerPiston" ]; then
    stage_pcorr()   # stage_pcorr <dir> -- arm W's waveMakerPiston with a loose pcorr
    {
        mkdir -p "$1"
        cp -r "$W/w_of_waveMakerPiston/0" "$W/w_of_waveMakerPiston/constant" "$W/w_of_waveMakerPiston/system" "$1/"
        python3 - "$1/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'("\(pcorr\|pcorrFinal\)"\s*\{[^}]*?tolerance\s+)[^;]+;', r'\g<1>1e-4;', s)
open(p, 'w').write(s)
sys.exit(0 if n == 1 else 1)
PY
    }
    contp()   # contp <case dir> -- |brae - OpenFOAM| / |OpenFOAM| of cumulativeContErr at 0.01
    {
        python3 - "$W/p_of" "$1" <<'PY'
import re, sys
v = [float(re.search(r'^value\s+(\S+);', open('%s/0.01/uniform/cumulativeContErr' % d).read(), re.M).group(1))
     for d in sys.argv[1:3]]
print('%.3e' % (abs(v[1] - v[0]) / abs(v[0])))
PY
    }
    stage_pcorr "$W/p_of" || say "ARM P  waveMakerPiston's pcorr entry was not found to loosen" FAIL
    runof "$W/p_of"
    python3 - "$W/p_of/log.interFoam" <<'PY' && say "fixture witnesses: OpenFOAM's correctPhi continuity error is most of its cumulative" ok \
                                          || say "fixture witnesses: OpenFOAM's correctPhi continuity error is most of its cumulative" FAIL
import re, sys
log = open(sys.argv[1]).read()
# the continuity line straight after each non-trivial pcorr solve is correctPhi.H:11's
g = [float(x) for x in re.findall(r'Solving for pcorr, Initial residual = (?!0,)[^\n]*\n'
                                   r'time step continuity errors : sum local = \S+, global = (\S+),', log)]
cu = float(re.findall(r'cumulative = (\S+)', log)[-1])
print('      correctPhi globals %s, final cumulative %.3e' % (' '.join('%.3e' % x for x in g), cu))
sys.exit(0 if g and abs(sum(g)) > 0.5 * abs(cu) else 1)
PY
    for arm in $ARMS; do
        bound=7e-4
        [ "$arm" = device ] && bound=6e-3
        stage_pcorr "$W/p_br_$arm"
        runbrae "$W/p_br_$arm" "$arm"
        r=$(contp "$W/p_br_$arm")
        python3 -c "import sys; sys.exit(0 if $r < $bound else 1)" \
            && say "ARM P  [$arm] waveMakerPiston's cumulativeContErr counts correctPhi's, within $bound ($r)" ok \
            || say "ARM P  [$arm] waveMakerPiston's cumulativeContErr counts correctPhi's, within $bound ($r)" FAIL
    done
    stage_pcorr "$W/p_ctl"
    runbrae "$W/p_ctl" host BRAE_CONTROL_NO_CORRECTPHI_CONTERR=1
    r=$(contp "$W/p_ctl")
    python3 -c "import sys; sys.exit(0 if $r > 0.5 else 1)" \
        && say "CONTROL  BRAE_CONTROL_NO_CORRECTPHI_CONTERR=1 puts it off by more than half ($r)" ok \
        || say "CONTROL  BRAE_CONTROL_NO_CORRECTPHI_CONTERR=1 puts it off by more than half ($r)" FAIL
else
    say "ARM P  waveMakerPiston did not run in arm W" FAIL
fi

# Q: a rigid body's uniform/rigidBodyMotionState (rigidBodyMeshMotion.C:392-417), entry by entry. Arm W's

finish "arm P: the mesh update's CorrectPhi continuity error"
