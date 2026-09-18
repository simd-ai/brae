#!/usr/bin/env bash
# brae's GAMG against REAL OpenFOAM's, where interFoam uses it: the last pressure corrector of the
# laminar/waves tutorials, whose fvSolution says
#
#     p_rghFinal { solver GAMG; smoother DIC; tolerance 1e-7; relTol 0; }
#
# and which the wave gate (interfoam_waves_vs_openfoam.sh) stages with PCG instead, because until this
# solver existed brae ran a substitute there. WHAT THE SUBSTITUTE COST is that gate's measurement:
# OpenFOAM against ITSELF on `trough`, nothing changed but this entry's solver, 1.8e-02 of alpha and
# 9.6% of U after twenty steps -- it is this gate's CONTROL, re-measured on every profile.
#
# THE ORACLE is OpenFOAM run SERIALLY for exactly N fixed steps with
#
#     DebugSwitches { GAMGAgglomeration 1; GAMG 1; }
#
# in its controlDict, which makes it print the hierarchy (GAMGAgglomeration::printLevels: cells, faces
# per cell and the band profile of every level) and the coarsest-level solve of every V-cycle
# ("DICPCG:  Solving for coarsestLevelCorr"). THE SWITCHES ARE INERT, and this script proves it on
# every run rather than saying so: `shipped` is run a second time without them and the three written
# fields must be byte-identical.
#
# PROFILES, all stokesI at full amplitude (the wave gate's `trough`) unless they say otherwise:
#   shipped      100 x 75, the tutorial's own entry. The arm that says brae can run these AS SHIPPED.
#   deep         p_rghFinal at 1e-12: many V-cycles per solve instead of the one or two 1e-7 takes,
#                so the V-cycle's arithmetic is most of the answer rather than its first pass.
#   square       800 x 30, where dx = dz and every face has the same area. faceAreaPair's weights are
#                |Sf/sqrt(magSf) * (1, 1.01, 1.02)|, a perturbation that exists for exactly this mesh;
#                on the others the cell aspect ratio decides the pairing and the perturbation is idle.
#   cubes        solitaryMcCowan on 140 x 11 x 10 CUBES of 0.05: the same, in 3-D, where the y and z
#                factors order two tie-breaks.
#   gaussSeidel, symGaussSeidel, dicGaussSeidel
#                the other three smoothers, on every level.
#   sweeps       every sweep control off its default at once, pre-smoothing included: nPreSweeps 1,
#                preSweepsLevelMultiplier 2, maxPreSweeps 3, nPostSweeps 1, postSweepsLevelMultiplier 2,
#                maxPostSweeps 5, nFinestSweeps 3.
#   coarsest     nCellsInCoarsestLevel 400: a short hierarchy and a coarsest solve that iterates.
#   noscale      scaleCorrection no.
#   both         p_rgh names GAMG as well, with nCellsInCoarsestLevel 50, and p_rghFinal keeps its 10.
#                THE HIERARCHY IS THE MESH'S: the first GAMG solve of the run builds it, from ITS
#                entry, and the other entry's nCellsInCoarsestLevel is never read.
#   tutorial     stokesI's own 500 x 75 mesh, ten steps.
#   pcgGamg      p_rghFinal as the solid-body tutorials write it: `solver PCG; preconditioner {
#                preconditioner GAMG; tolerance 1e-7; relTol 0; nVcycles 2; smoother DICGaussSeidel;
#                nPreSweeps 2; }` -- PCG with a GAMGSolver as its preconditioner, two V-cycles from zero
#                per application, the sub-dictionary its controls. The coarsest-level log is the
#                preconditioner's.
#   pcgGamgTol   the same with the sub-dictionary's tolerance 1e-3, nVcycles 3 and the DIC smoother,
#                against the PCG's 1e-7: the coarsest-level solve takes the PRECONDITIONER's tolerance,
#                and a port that took the PCG's would read other coarsest counts.
#   <the other seven tutorials that name GAMG>
#                stokesII, stokesV, cnoidal, streamFunction, solitary, solitaryGrimshaw and
#                solitaryMcCowan on the wave gate's meshes and step counts, each AS SHIPPED where that
#                gate stages PCG. irregularMultiDirection names PCG itself and is not here.
#
# MEASURED, nineteen profiles: every level of every hierarchy has OpenFOAM's cell count, faces per cell
# and band profile (5 to 12 levels); every p_rgh solve takes OpenFOAM's iteration count (20 to 60 per
# profile) and every coarsest-level solve too (3 to 125); alpha 4.0e-11, p_rgh 1.9e-10 and U 1.9e-09 at
# worst (U 1.4e-08 on `deep`, for a reason that is not the solver's -- the test says which face). THE
# CONTROL, OpenFOAM with PCG against OpenFOAM with GAMG: alpha 1.9e-04 to 8.4e-02, never under 1e5
# times brae's distance. WHAT THIS SOLVER REPLACED, PBiCGStab under a notice: alpha 2.3e-02, U 19% on
# `shipped`, where this reads 8.3e-12 and 5.5e-10.
#
# EVERY PORT DECISION WAS BROKEN ONCE, with switches that are not in the tree, and what the gate read:
#   the pairing direction never alternates       `shipped`: alpha 1.2e-03, no coarsest count equal
#   coarse faces sorted by neighbour             `shipped`: the band profile and the next level's cell
#                                                count move; alpha 1.1e-07
#   the rejected last level kept                 13 levels for 10; no coarsest count equal; THE FIELDS DO
#                                                NOT MOVE (9.8e-12) -- only the log arms hold this
#   the coarsest level solved to 1e-12           no coarsest count equal; the fields do not move at all
#   weights without the (1, 1.01, 1.02) factors  `shipped`: NOTHING, to the last digit -- the cells are
#                                                2 x 0.1 and the aspect ratio decides every pairing.
#                                                `square`: alpha 4.5e-03, 16 of 20 counts. `cubes`: 2.0e-06
#   the y and z factors swapped                  `cubes`: 11 levels for 10, alpha 4.9e-06
#   the level above the coarsest scaled too      `shipped`: the 11th digit of a residual, which is where
#                                                brae and OpenFOAM differ anyway -- a 24-cell level.
#                                                `coarsest` (929 cells there): alpha 5.0e-05, U 1.4e-03
#   prolongation adds instead of overwriting     alpha 2.8e-04 (the fields hold the last cycle's values)
#   post-sweeps without the level multiplier     alpha 9.7e-05
#   the hierarchy built from the entry in use    `both`: 10 levels for 8, alpha 8.5e-09
# `square`, `cubes` and `coarsest` exist because of the two lines that read NOTHING without them.
#
# THE GAMG PRECONDITIONER (`pcgGamg`, `pcgGamgTol`): 40 of 40 p_rgh counts on both, alpha 4.9e-12 and
# 3.4e-12, every coarsest-level count OpenFOAM's (8 and 9). BROKEN ONCE EACH:
#   one V-cycle per application where the entry says two     34 of 40 counts, alpha 1.5e-02
#   no residual recomputed between the two V-cycles          34 of 40, alpha 1.4e-02
#   the application starting from the last one's w, not 0    alpha 7.1e-05, final residuals 98% out
#   the coarsest solve at the PCG's tolerance, not the       0 of 9 coarsest counts; no field moves
#     preconditioner's own (`pcgGamgTol`: 1e-3 against 1e-7)
#
# THE DEVICE LOOP RUNS EVERY DIC PROFILE TOO (sixteen; its GAMG has the DIC smoother and refuses the
# other three, which tests/interfoam_refusals.sh holds). MEASURED: every p_rgh and every coarsest-level
# iteration count OpenFOAM's; alpha 2.9e-11, p_rgh 5.2e-11, U 1.6e-09 at worst (1.5e-08 on `deep`), and
# 4.5e-11 of alpha from the host. WHAT IT REPLACED, Jacobi-BiCGStab under a notice: alpha 2.3e-02 and U
# 45% on `shipped`. BROKEN ONCE EACH on the device: the coarse levels' DIC diagonals left over from the
# first solve 1.8e-03 of alpha; prolongation that adds 7.3e-03; the level above the coarsest scaled,
# on `coarsest`, 5.0e-05; and the coarsest MATRIX left over from the first solve -- the host solves
# that level, from a copy -- which moves NO FIELD and reads 2.3e-02 on the coarsest-solve log arm.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_gamg_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
WAVES="$TUT/multiphase/interFoam/laminar/waves"

[ -x "$BIN" ]            || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$WAVES/stokesI" ]  || { echo "SKIP: waves/stokesI tutorial not found under $WAVES"; exit 77; }
[ -f "$OFBASHRC" ]       || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

# stage <name> <tutorial> <deltaT> <nSteps> <mesh: "nx ny nz"|-> <entry> <debug: debug|quiet>
# <entry> is what becomes of p_rghFinal: shipped, pcg (the control), or a profile named above.
stage()
{
    local name="$1" tutorial="$2" dt="$3" n="$4" nx="$5" entry="$6" debug="$7"
    local C="$W/$name"
    [ -d "$WAVES/$tutorial" ] || { echo "FAIL: no tutorial $WAVES/$tutorial"; return 1; }
    cp -r "$WAVES/$tutorial" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    DT="$dt" N="$n" NX="$nx" ENTRY="$entry" DEBUG="$debug" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d = sys.argv[1]
dt, n, nx = os.environ['DT'], int(os.environ['N']), os.environ['NX']
entry, debug = os.environ['ENTRY'], os.environ['DEBUG']

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
if debug == 'debug':
    # GAMGSolver's switch is its TypeName, `GAMG`
    s += '\nDebugSwitches\n{\n    GAMGAgglomeration 1;\n    GAMG 1;\n}\n'
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
assert re.search(r'solver\s+GAMG;', body) and re.search(r'smoother\s+DIC;', body), \
    'the tutorial no longer names GAMG with a DIC smoother for p_rghFinal'
add = {
    'sweeps': 'nPreSweeps 1; preSweepsLevelMultiplier 2; maxPreSweeps 3; nPostSweeps 1; '
              'postSweepsLevelMultiplier 2; maxPostSweeps 5; nFinestSweeps 3;',
    'coarsest': 'nCellsInCoarsestLevel 400;',
    'noscale': 'scaleCorrection no;',
}
smoother = {'gaussSeidel': 'GaussSeidel', 'symGaussSeidel': 'symGaussSeidel', 'dicGaussSeidel': 'DICGaussSeidel'}
if entry == 'pcg':
    body = re.sub(r'solver\s+GAMG;', 'solver          PCG;', body)
    body = re.sub(r'smoother\s+DIC;', 'preconditioner  DIC;', body)
elif entry in smoother:
    body = re.sub(r'smoother\s+DIC;', 'smoother        %s;' % smoother[entry], body)
elif entry in add:
    body += '    ' + add[entry] + '\n    '
elif entry in ('pcgGamg', 'pcgGamgTol'):
    tol, cycles, smooth, pre = ('1e-7', '2', 'DICGaussSeidel', 'nPreSweeps 2; ') if entry == 'pcgGamg' \
                               else ('1e-3', '3', 'DIC', '')
    body = ('\n        solver          PCG;\n        preconditioner\n        {\n            preconditioner GAMG; '
            'tolerance %s; relTol 0; nVcycles %s; smoother %s; %s\n        }\n        tolerance       1e-7;\n'
            '        relTol          0;\n        maxIter         20;\n    ' % (tol, cycles, smooth, pre))
elif entry == 'deep':
    body, k = re.subn(r'tolerance\s+1e-7;', 'tolerance       1e-12;', body)
    assert k == 1, 'p_rghFinal tolerance not found'
else:
    assert entry in ('shipped', 'both'), entry
t = t[:m.start(2)] + body + t[m.end(2):]
if entry == 'both':
    m = re.search(r'(\n\s*p_rgh\s*\{)([^}]*)\}', t)
    assert m, 'no p_rgh entry'
    body = m.group(2)
    body, k1 = re.subn(r'solver\s+PCG;', 'solver          GAMG;', body)
    body, k2 = re.subn(r'preconditioner\s+DIC;', 'smoother        DIC;\n        nCellsInCoarsestLevel 50;', body)
    assert k1 == 1 and k2 == 1, 'p_rgh is no longer PCG with DIC'
    t = t[:m.start(2)] + body + t[m.end(2):]
open(q, 'w').write(t)

# full amplitude from step five, as the wave gate's `trough`. A solitary wave has no ramp.
w = os.path.join(d, 'constant/waveProperties')
t = open(w).read()
if re.search(r'rampTime\s+[^;]+;', t):
    t, k = re.subn(r'rampTime\s+[^;]+;', 'rampTime        0.05;', t)
    assert k == 1, 'rampTime not found'
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

# gate <name> <deltaT> <nSteps> <profile> <the PCG control's name>
gate()
{
    local name="$1" dt="$2" n="$3" profile="$4" control="$5"
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    grep -q "^GAMGAgglomeration:" "$W/$name/log.interFoam" \
        || { echo "FAIL: OpenFOAM's $name log carries no hierarchy -- the debug switch did not take"; return 1; }
    grep -q "Solving for coarsestLevelCorr" "$W/$name/log.interFoam" \
        || { echo "FAIL: OpenFOAM's $name log carries no coarsest-level solve"; return 1; }
    "$BIN" "$W/$name" "$W/$name/0" "$W/$name/$end" "$n" "$W/$name/log.interFoam" "$profile" \
           "$W/$control/$end"
}

rc=0
SMALL="100 1 75"
stage pcg            stokesI 0.01 20 "$SMALL" pcg            quiet || rc=1
stage shippedQuiet   stokesI 0.01 20 "$SMALL" shipped        quiet || rc=1
for entry in shipped deep gaussSeidel symGaussSeidel dicGaussSeidel sweeps coarsest noscale both pcgGamg pcgGamgTol; do
    stage "$entry"   stokesI 0.01 20 "$SMALL" "$entry"       debug || rc=1
done
stage squarePcg      stokesI 0.01 10 "800 1 30" pcg          quiet || rc=1
stage square         stokesI 0.01 10 "800 1 30" shipped      debug || rc=1
stage cubesPcg       solitaryMcCowan 0.01 10 "140 11 10" pcg     quiet || rc=1
stage cubes          solitaryMcCowan 0.01 10 "140 11 10" shipped debug || rc=1
stage tutorialPcg    stokesI 0.01 10 - pcg                   quiet || rc=1
stage tutorial       stokesI 0.01 10 - shipped               debug || rc=1
# the wave gate's fixtures, as "tutorial|mesh|steps"
MODELS=(
    "stokesII|100 1 55|20"
    "stokesV|100 1 70|20"
    "cnoidal|100 1 70|20"
    "streamFunction|100 1 80|20"
    "solitary|100 1 150|30"
    "solitaryGrimshaw|70 4 42|30"
    "solitaryMcCowan|70 4 42|30"
)
for model in "${MODELS[@]}"; do
    IFS='|' read -r tutorial mesh steps <<< "$model"
    stage "${tutorial}Pcg" "$tutorial" 0.01 "$steps" "$mesh" pcg     quiet || rc=1
    stage "$tutorial"      "$tutorial" 0.01 "$steps" "$mesh" shipped debug || rc=1
done
[ $rc = 0 ] || { echo "interfoam_gamg_vs_openfoam: staging failed"; exit 1; }

# THE SWITCHES ARE INERT: the same case with and without them writes the same bytes
for fld in alpha.water p_rgh U; do
    cmp -s "$W/shipped/0.2/$fld" "$W/shippedQuiet/0.2/$fld" \
        || { echo "FAIL: OpenFOAM's $fld differs with the GAMG debug switches on -- the oracle is not inert"; rc=1; }
done
[ $rc = 0 ] && echo "ok:   OpenFOAM's alpha, p_rgh and U are byte-identical with and without the debug switches"

# `both`: the oracle took the path -- p_rgh's line says GAMG too, and the hierarchy stops at 50
grep -q "^GAMG:  Solving for p_rgh.*No Iterations" "$W/both/log.interFoam" \
    || { echo "FAIL: OpenFOAM's both run has no GAMG p_rgh solve"; rc=1; }
[ "$(grep -c '^DICPCG:  Solving for p_rgh' "$W/both/log.interFoam")" = 0 ] \
    || { echo "FAIL: OpenFOAM's both run still solves p_rgh with PCG"; rc=1; }

for entry in shipped deep gaussSeidel symGaussSeidel dicGaussSeidel sweeps coarsest noscale both pcgGamg pcgGamgTol; do
    gate "$entry" 0.01 20 "$entry" pcg || rc=1
done
gate square   0.01 10 square   squarePcg   || rc=1
gate cubes    0.01 10 cubes    cubesPcg    || rc=1
gate tutorial 0.01 10 tutorial tutorialPcg || rc=1
for model in "${MODELS[@]}"; do
    IFS='|' read -r tutorial mesh steps <<< "$model"
    gate "$tutorial" 0.01 "$steps" "$tutorial" "${tutorial}Pcg" || rc=1
done

echo "interfoam_gamg_vs_openfoam: rc $rc"
exit $rc
