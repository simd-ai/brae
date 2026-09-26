#!/usr/bin/env bash
# brae's interFoam WAVE boundary conditions -- waveAlpha, waveVelocity, and all ten of OpenFOAM's wave
# models behind them -- against REAL OpenFOAM, on the nine laminar/waves tutorials that use them.
#
# THE METHOD is the other interFoam gates': exactly N identical FIXED steps, read at one instant, with
# OpenFOAM run SERIALLY (the tutorial's Allrun decomposes it; a parallel run is a different smoother).
#
# WHAT IS COMPARED. The three fields and every p_rgh iteration count, as elsewhere -- and four things
# only these conditions have an oracle for:
#   the model's derived constants   "Reference water depth" and "Wave length", from OpenFOAM's log
#   the ORDER of its updates        one "Updating <model> wave model for patch <p>" line per update:
#                                   per step, three for the inlet (one per alpha sub-cycle), a fourth
#                                   (U's first updateCoeffs, in UEqn), then the outlet's
#   the patch VALUES                what waveAlpha and waveVelocity last assigned, face by face, from
#                                   the time directory OpenFOAM wrote
#
# NINE TUTORIALS, ONE PER GENERATION MODEL, all over the same solver settings and the same
# shallowWaterAbsorption outlet: stokesI, stokesII, stokesV, cnoidal, streamFunction,
# irregularMultiDirection, solitary (Boussinesq), solitaryGrimshaw, solitaryMcCowan. Each runs on a
# coarsened mesh at full amplitude (the ramped ones with rampTime 0.05; a solitary wave has no ramp) and
# against its own still-water control. What the model ITSELF computes has two direct oracles: every
# number in the block OpenFOAM prints when it creates the model -- the wave length, StokesV's lambda,
# cnoidal's m, a solitary wave's x0 -- and the inlet's velocity face by face, which for StokesV is five
# harmonics in the phase and five in the depth and so shows a wrong coefficient at its own order.
# The three-dimensional ones (Grimshaw, McCowan, irregularMultiDirection) carry slip side walls.
#
# AND SIX PROFILES OF stokesI, which is where the conditions themselves were ported:
#   shipped   the tutorial's own mesh (500 x 75) and waveProperties, ten steps at its own deltaT 0.01.
#             Its rampTime is 3 s, so at t = 0.1 the wave stands at a thirtieth of its height and |U|
#             is 2.7e-03: the weakest signal here, and the one that is the case as it ships. (Ten steps
#             at the tutorial's maxDeltaT 0.05 were tried and dropped: at Co 1.9 OpenFOAM's own STILL
#             tank carries half the velocity of the one with the wave, so that fixture gates noise.)
#   trough    a 100 x 75 mesh with rampTime 0.05, twenty steps of 0.01: full amplitude from step five.
#             The default phase starts the wave on a TROUGH, so the level sits BELOW the reference
#             depth and the inlet flows OUT -- one half of waveModel::setPaddlePropeties.
#   crest     the same with wavePhase pi/2: level ABOVE the reference depth, inflow, the other half.
#   tight     trough with BOTH codes' p_rgh solves at 1e-13 and no relTol: the discretisation with
#             the stopping point taken out.
#   mulescorr trough with `MULESCorr yes`, which no wave tutorial sets either. Under it alpha's first
#             updateCoeffs of a sub-cycle is the PRE-SOLVE's matrix construction, not the explicit
#             solve's correctBoundaryConditions, so the wave update has a second call site on both
#             paths and this is the only profile that reaches it.
#   mompred   trough with `momentumPredictor yes`, which no wave tutorial sets. The tutorial names
#             PBiCG for U; the staging names smoothSolver with symGaussSeidel, which both codes run.
#             It is here for what it found -- see below.
# ...AND A RESTART, `restart`: the `trough` run's own twenty steps, then twenty MORE resumed from the state
# it wrote. The wave model IS an IOdictionary at <startTime>/uniform/waveProperties.<patch>, and OpenFOAM
# reads it back before merging the case's own entries over it -- so the one key it contributes is
# `waterDepthRef`, which OpenFOAM adds to itself precisely so a run can be resumed. Recomputing it instead
# takes the reference depth against a water level the wave has already moved: the stored
# 0.600000000000001 against a recomputed 0.592198129937198 at the inlet, and alpha 2.87e-01 / U 6.43e-01 of
# the answer twenty steps later. THE ORACLE is OpenFOAM's warm restart; THE CONTROL is OpenFOAM's COLD one
# -- not the still tank, because what is under test here is the stored depth and not the wave.
# MEASURED with the changed waveHeight: host alpha 6.0627e-12, p_rgh 2.5892e-12, U 1.2416e-10; device
# 5.4097e-12, 3.2622e-12, 1.4309e-10 -- the ordinary tolerance ball, with all 40 p_rgh counts OpenFOAM's
# and the model's own patch values to 2.8e-16. (Without the height change, 3.3138e-12 / 3.3613e-12 /
# 2.0851e-10.) Control: U 6.3710e-01, alpha 2.7795e-01.
# BROKEN THREE WAYS, each red on BOTH arms, and each a different number:
#   the stored dictionary not read at all (the pre-port behaviour)  U 6.3710e-01, 19 failures
#   `waterDepthRef` alone dropped from what was read                U 6.3710e-01, 18 failures
#   the merge REVERSED, so the stored file wins over the case       U 5.2874e-01, 17 failures
# The first two land on the SAME number, which is the measurement that says `waterDepthRef` is the whole of
# what the file contributes here -- nothing else in it reaches the answer. The third is a different number
# only because the fixture gives the case its own waveHeight; without that it would be silent.
#
# THE PRESSURE SOLVER IS THE TUTORIAL'S OWN, which for eight of the nine is GAMG on the last corrector.
# IT WAS NOT, until brae had OpenFOAM's GAMG on both loops (tests/interfoam_gamg_vs_openfoam.sh): every
# profile here used to be staged with PCG and DIC in its place, because a solver-log arm across two
# different solvers would have been about the solver. With that staging gone the worst profile reads
# alpha 2.9e-11 where it read 1.6e-10, the two 3-D solitary cases lost their 1e-3 residual bound
# (6.4e-09 now), and the field bounds came down.
# WHAT THAT SUBSTITUTION IS WORTH ON THESE CASES WAS MEASURED, and it is not small: OpenFOAM against
# ITSELF on the `trough` fixture, with nothing changed but p_rghFinal's solver -- GAMG or PCG, the same
# DIC, the same tolerance 1e-7 -- differs by 1.8e-02 of alpha and 9.6% of U after twenty steps. brae with
# its substitute WAS 2.3e-02 and 19% from OpenFOAM-with-GAMG. The tank's active absorption feeds the
# water level back into the velocity, and where the last pressure solve stops is part of the answer at
# that tolerance; agreeing with the tutorial AS SHIPPED needed OpenFOAM's GAMG, agglomeration and all.
#
# THE CONTROL is OpenFOAM's own answer for the same tank with NO WAVE -- the generating patch given the
# absorbing model too: on stokesI the wave moves its U by 100% and its alpha by 0.73, against brae's
# 1.8e-08 and 1.4e-10 from it.
#
# WHAT THIS GATE FOUND THAT IS NOT ABOUT WAVES. constrainPressure's gradient is
# (phiHbyA_b - (Sf_b & U_b))/(magSf_b*rAUf_b), and brae subtracted the stored phi_b where OpenFOAM
# subtracts the VELOCITY's flux. On a wall that does not move they are the same number, and every
# fixedFluxPressure patch this solver had met was one. On a patch whose U_b changes, the corrector
# hands back exactly the flux it was given: the inlet's flux starts at zero and stays there. Measured
# here with the wave model's own patch values exact to 4e-16: U 260% out, and OpenFOAM's answer with
# the wave switched OFF (64%) closer to OpenFOAM's than brae was.
#
# A THIRD, from solitaryGrimshaw and solitaryMcCowan, whose totalPressure top says `phi rhoPhi;`. brae's
# field reader kept no `phi` entry and every flux-conditional condition was told phi. rhoPhi is the
# ALPHA step's flux -- one pressure corrector behind phi, and exactly zero at step one of a case at
# rest -- so the top's inflow test decides differently. Both cases were wrong from STEP ONE's p_rgh
# residual with their wave models exact to 7e-17: U 2.4e-05 and 3.0e-05 out after thirty steps,
# 8.0e-10 and 1.4e-08 with the named flux. (Shown by running OpenFOAM itself with `phi phi;`: brae
# agreed with THAT to 9.6e-10.)
#
# AND A SECOND, from `mompred`. The predictor's explicit fvc::snGrad(p_rgh) reads, on a
# fixedFluxPressure patch, the gradient the LAST constrainPressure stored there. brae's driver zeroed
# that gradient before every predictor. On a wall whose alpha is zeroGradient the stored gradient is
# zero anyway -- which was every predictor case gated before this one. At the wave inlet it is not:
# alpha 5.1e-04 and U 5.4% out after twenty steps with the zero, 1.4e-11 and 1.0e-09 with the stored one.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_waves_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
WAVES="$TUT/multiphase/interFoam/laminar/waves"
SRC="$WAVES/stokesI"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: waves/stokesI tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

# KEEP_W=<dir> keeps the staged cases and the oracle runs, which is what makes a fail-proof on one
# arm cheap: stage once, then call $BIN on that arm alone. The other interFoam gates do the same.
W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

# stage <name> <tutorial> <deltaT> <nSteps> <mesh: "nx ny nz"|-> <waves: shipped|trough|crest|still>
#       <solves: case|tight|mompred|mulescorr>
stage()
{
    local name="$1" tutorial="$2" dt="$3" n="$4" nx="$5" waves="$6" solves="$7"
    local C="$W/$name"
    [ -d "$WAVES/$tutorial" ] || { echo "FAIL: no tutorial $WAVES/$tutorial"; return 1; }
    cp -r "$WAVES/$tutorial" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    DT="$dt" N="$n" NX="$nx" WAVES="$waves" SOLVES="$solves" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d = sys.argv[1]
dt, n, nx = os.environ['DT'], int(os.environ['N']), os.environ['NX']
waves, solves = os.environ['WAVES'], os.environ['SOLVES']

c = os.path.join(d, 'system/controlDict')
s = open(c).read()
# the tutorial's function objects sample to disk and are no part of the comparison
s, k = re.subn(r'functions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
assert k == 1, 'controlDict functions block not found'
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '15'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)

if nx != '-':
    b = os.path.join(d, 'system/blockMeshDict')
    t = open(b).read()
    t, k = re.subn(r'\(\s*\d+\s+\d+\s+\d+\s*\)\s*simpleGrading', '(%s) simpleGrading' % nx, t)
    assert k == 1, 'blockMeshDict cell counts not found'
    open(b, 'w').write(t)

q = os.path.join(d, 'system/fvSolution')
t = open(q).read()
m = re.search(r'(p_rghFinal\s*\{)([^}]*)\}', t)
assert m, 'no p_rghFinal entry'
body = m.group(2)
# eight of the nine name GAMG with a DIC smoother; irregularMultiDirection names PCG with DIC. BOTH
# ARE RUN AS THE TUTORIAL NAMES THEM -- this used to rewrite GAMG to PCG, see the header.
if 'GAMG' in body:
    assert re.search(r'smoother\s+DIC;', body), 'p_rghFinal is GAMG without a DIC smoother'
else:
    assert re.search(r'solver\s+PCG;', body) and re.search(r'preconditioner\s+DIC;', body), \
        'p_rghFinal is neither GAMG nor PCG with DIC'
if solves == 'tight':
    t, k = re.subn(r'tolerance\s+1e-0?[67];', 'tolerance       1e-13;', t)
    assert k >= 2, 'p_rgh tolerances not found'
    t = re.sub(r'relTol\s+0\.1;', 'relTol          0;', t)
if solves == 'mulescorr':
    t, k = re.subn(r'(cAlpha\s+[^;]+;)',
                   r'\1\n        MULESCorr       yes;\n        nLimiterIter    3;'
                   r'\n        solver          smoothSolver;\n        smoother        symGaussSeidel;'
                   r'\n        tolerance       1e-8;\n        relTol          0;', t)
    assert k == 1, 'the alpha entry was not found'
if solves == 'mompred':
    t, k = re.subn(r'momentumPredictor\s+no;', 'momentumPredictor yes;', t)
    assert k == 1, 'momentumPredictor not found'
    t, k = re.subn(r'solver\s+PBiCG;', 'solver          smoothSolver;', t)
    assert k == 2, 'the U and UFinal entries are no longer PBiCG'
    t, k = re.subn(r'preconditioner\s+DILU;', 'smoother        symGaussSeidel;', t)
    assert k == 2, 'the U and UFinal entries are no longer DILU-preconditioned'
open(q, 'w').write(t)

w = os.path.join(d, 'constant/waveProperties')
t = open(w).read()
ramped = re.search(r'rampTime\s+[^;]+;', t) is not None
if waves in ('trough', 'crest') and ramped:
    # full amplitude from step five. A solitary wave has no ramp: it is at full height as shipped.
    t, k = re.subn(r'rampTime\s+[^;]+;', 'rampTime        0.05;', t)
    assert k == 1, 'rampTime not found'
if waves == 'crest':
    t, k = re.subn(r'(wavePeriod\s+[^;]+;)', r'\1\n    wavePhase       1.5707963267948966;', t)
    assert k == 1, 'wavePeriod not found'
if waves == 'still':
    # THE SAME TANK WITH NO WAVE: the generating patch is given the absorbing model too, so both ends
    # hold still water. One rule for all nine -- the first tries were a wave height of zero, which a
    # solitary wave divides by (ts = 3.5h/sqrt(H/h)), and then one of 1e-12, on which OpenFOAM's own
    # McCowan Newton-Raphson fails to converge in 10001 passes.
    t, k = re.subn(r'waveModel\s+\w+;', 'waveModel       shallowWaterAbsorption;', t)
    assert k == 2, 'expected a generating and an absorbing waveModel entry'
open(w, 'w').write(t)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$name]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$name]"; tail -30 "$C/log.interFoam"; return 1; }
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    [ -d "$C/$end" ] || { echo "FAIL: OpenFOAM wrote no $end directory [$name]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $n steps of deltaT $dt to t = $end   [$name]"
}

# restartFrom <profile> <end> <mode>: OpenFOAM again, from <profile>'s own <end> directory, with the wave
# model's stored dictionary either kept (`warm` -- THE ORACLE) or removed (`cold` -- THE CONTROL). The
# staged case is a copy, so the first run's output stays intact, and `adjustTimeStep no` is already set by
# stage(), which is what makes the restart's deltaT the same number as OpenFOAM's (Time::setControls reads
# deltaT back out of <start>/uniform/time only under adjustTimeStep, and brae does not read that file).
#
# WHAT THE STORED DICTIONARY IS. waveModel IS an IOdictionary at <startTime>/uniform/waveProperties.<patch>
# (waveModel.C:250-261) and readDict reads it BEFORE merging the case's own sub-dictionary over it
# (:294-302, dictionary::merge at dictionary.C:806-848, where the override replaces what it finds). So the
# only key the file contributes is the one OpenFOAM `add`s to itself to make a resume possible --
# `waterDepthRef` (:342) -- and the `cold` arm removes it. MEASURED on this fixture: the stored
# 0.600000000000001 at the inlet against a recomputed 0.592198129937198, because a recomputed reference
# depth is taken against the water level the wave has already moved. That is worth alpha 2.87e-01 and
# U 6.43e-01 of the answer twenty steps later, over 7489 of 7500 cells.
#
# AND THE MERGE DIRECTION IS WITNESSED, which it would not be by a plain resume: the first run's stored
# file repeats every key the case names, so letting the STORED file win would read the same numbers. The
# restart's own case therefore names a DIFFERENT waveHeight (0.07 where the first run had 0.05), asserted
# against the stored file's old value -- so the case's key must win and the stored `waterDepthRef`, which
# the case does not name, must survive. Both halves of merge(), one fixture.
restartFrom()
{
    local profile="$1" end="$2" mode="$3"
    local B="$W/$profile" C="$W/restart_$mode"
    rm -rf "$C"
    cp -r "$B" "$C" || return 1
    rm -f "$C"/log.interFoam
    python3 - "$C" "$end" "$mode" <<'RSTEOF' || { echo "FAIL: staging restart_$mode"; return 1; }
import os, re, sys
d, end, mode = sys.argv[1], sys.argv[2], sys.argv[3]
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'^startFrom .*', 'startFrom       startTime;', s, flags=re.M)
s = re.sub(r'^startTime .*',  'startTime       %s;' % end,  s, flags=re.M)
s = re.sub(r'^endTime .*',    'endTime         %.10g;' % (2*float(end)), s, flags=re.M)
open(c, 'w').write(s)
u = os.path.join(d, end, 'uniform')
# WHAT OPENFOAM WROTE, asserted rather than assumed: one wave dictionary per wave patch. The cold arm
# removes exactly those two and NOTHING ELSE in uniform/ -- `time` and `cumulativeContErr` are other
# solvers' restart state and taking them out would make the arm about more than the wave model.
got = sorted(os.listdir(u)) if os.path.isdir(u) else []
want = ['waveProperties.inlet', 'waveProperties.outlet']
assert all(w in got for w in want), 'OpenFOAM wrote %s in %s/uniform, without %s' % (got, end, want)

# THE MERGE DIRECTION, made witnessable. The stored file is the first run's merged dictionary, so every
# key the case names it names identically -- and a brae that let the STORED file win would read the same
# numbers and pass. So the restart's own case gets a DIFFERENT waveHeight, which only the case has: the
# override must win on it (dictionary::merge replaces what it finds) while `waterDepthRef`, which only the
# stored file has, must survive. One fixture, both halves of the merge.
w = os.path.join(d, 'constant/waveProperties')
t = open(w).read()
t, k = re.subn(r'waveHeight\s+0\.05;', 'waveHeight      0.07;', t)
assert k == 1, 'the trough profile no longer ships waveHeight 0.05'
open(w, 'w').write(t)
stored = open(os.path.join(u, 'waveProperties.inlet')).read()
assert re.search(r'waveHeight\s+0\.05;', stored), \
    'the stored dictionary does not hold the OLD waveHeight, so the merge direction is not witnessed'
assert re.search(r'waterDepthRef\s+', stored), 'the stored dictionary holds no waterDepthRef'

if mode == 'cold':
    for w in want:
        os.remove(os.path.join(u, w))
RSTEOF
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam [restart_$mode]"; tail -30 "$C/log.interFoam"; return 1; }
    local rend
    rend=$(python3 -c "print('%.10g' % (2*float('$end')))")
    [ -d "$C/$rend" ] || { echo "FAIL: OpenFOAM wrote no $rend directory [restart_$mode]"; ls "$C"; return 1; }
    # THE ARM TOOK THE PATH ITS NAME CLAIMS: the reference depth the model reports is the stored one on
    # `warm` and a recomputed one on `cold`. Read out of OpenFOAM's own log, so it is OpenFOAM saying so.
    local depth
    depth=$(grep -m1 "Reference water depth" "$C/log.interFoam" | awk '{print $NF}')
    if [ "$mode" = warm ]; then
        [ "$depth" = "0.600000000000001" ] \
            || { echo "FAIL: the WARM restart reports a reference depth of $depth, not the stored 0.600000000000001"; return 1; }
    else
        [ "$depth" != "0.600000000000001" ] \
            || { echo "FAIL: the COLD restart reports the stored reference depth $depth -- it cannot have recomputed one"; return 1; }
    fi
    echo "OpenFOAM restarted to t = $rend, reference water depth $depth   [restart_$mode]"
}

# gate <name> <deltaT> <nSteps> <profile> <still-water oracle>
gate()
{
    local name="$1" dt="$2" n="$3" profile="$4" still="$5"
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    "$BIN" "$W/$name" "$W/$name/0" "$W/$name/$end" "$n" "$W/$name/log.interFoam" "$profile" \
           "$W/$still/$end"
}

rc=0
stage shippedStill stokesI 0.01 10 -          still   case    || rc=1
stage shipped      stokesI 0.01 10 -          shipped case    || rc=1
stage smallStill   stokesI 0.01 20 "100 1 75" still   case    || rc=1
stage trough       stokesI 0.01 20 "100 1 75" trough  case    || rc=1
stage crest        stokesI 0.01 20 "100 1 75" crest   case    || rc=1
stage tight        stokesI 0.01 20 "100 1 75" trough  tight   || rc=1
stage mompred      stokesI 0.01 20 "100 1 75" trough  mompred || rc=1
stage mulescorr    stokesI 0.01 20 "100 1 75" trough  mulescorr || rc=1
grep -q "Solving for alpha.water" "$W/mulescorr/log.interFoam" \
    || { echo "FAIL: OpenFOAM's mulescorr run never solved the implicit alpha equation"; exit 1; }

# the other eight models, as "tutorial|mesh|steps". The 3-D ones keep a spanwise direction.
MODELS=(
    "stokesII|100 1 55|20"
    "stokesV|100 1 70|20"
    "cnoidal|100 1 70|20"
    "streamFunction|100 1 80|20"
    "irregularMultiDirection|75 5 60|20"
    "solitary|100 1 150|30"
    "solitaryGrimshaw|70 4 42|30"
    "solitaryMcCowan|70 4 42|30"
)
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tutorial mesh steps <<< "$entry"
    stage "${tutorial}Still" "$tutorial" 0.01 "$steps" "$mesh" still  case || rc=1
    stage "$tutorial"        "$tutorial" 0.01 "$steps" "$mesh" trough case || rc=1
done
[ $rc = 0 ] || { echo "interfoam_waves_vs_openfoam: staging failed"; exit 1; }

# A RESTART from OpenFOAM's own state, off the `trough` run's own twenty steps
for mode in warm cold; do
    restartFrom trough 0.2 "$mode" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_waves_vs_openfoam: restart staging failed"; exit 1; }

# the oracle took the path: a log with no model update in it cannot gate one
grep -q "Updating StokesI wave model for patch inlet" "$W/trough/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log has no StokesI update"; exit 1; }
grep -q "Updating shallowWaterAbsorption wave model for patch outlet" "$W/trough/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log has no shallowWaterAbsorption update"; exit 1; }

gate shipped 0.01 10 shipped shippedStill || rc=1
gate trough  0.01 20 trough  smallStill   || rc=1
gate crest   0.01 20 crest   smallStill   || rc=1
gate tight   0.01 20 tight   smallStill   || rc=1
gate mompred 0.01 20 mompred smallStill   || rc=1
gate mulescorr 0.01 20 mulescorr smallStill || rc=1
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tutorial mesh steps <<< "$entry"
    gate "$tutorial" 0.01 "$steps" "$tutorial" "${tutorial}Still" || rc=1
done

# ...and the RESTART, whose oracle is OpenFOAM's warm restart and whose control is its COLD one -- the
# run that recomputes the reference depth, which is what brae did while this was refused.
"$BIN" "$W/restart_warm" "$W/restart_warm/0.2" "$W/restart_warm/0.4" 20 \
       "$W/restart_warm/log.interFoam" restart "$W/restart_cold/0.4" || rc=1

echo "interfoam_waves_vs_openfoam: rc $rc"
exit $rc
