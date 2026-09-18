#!/usr/bin/env bash
# brae's interFoam on a MOVING MESH against REAL OpenFOAM's, and on a CLOSED tank: the two things the
# solid-body tutorials ask for that no gated case had -- a mesh that moves under the fluid, and a
# pressure with no patch to fix its level.
#
# THE ORACLE is OpenFOAM run SERIALLY for exactly N fixed steps, written at writePrecision 18: alpha,
# p_rgh, p, U, the moved polyMesh/points, Uf and meshPhi, and the log's "Solving for p_rgh" lines.
#
# THE FIXTURE is laminar/testTubeMixer, the one solid-body tutorial on an orthogonal mesh: a 1 x 10 x 1
# cm tube of water and air on a turntable (rotatingMotion, 2 pi rad/s) that also tilts about its own
# axis (oscillatingRotatingMotion, 45 degrees at 40 rad/s), every wall a movingWallVelocity, p_rgh a
# fixedFluxPressure everywhere, PIMPLE naming pRefPoint and pRefValue 1e5, and `div(rhoPhi,U) Gauss
# vanLeerV` -- run as shipped now that vanLeerV is ported (it was staged to `Gauss linear` before). ONE
# THING IS STAGED AWAY FROM THE TUTORIAL, a substitution today and not a choice:
#   p_rghFinal      PCG with a GAMG PRECONDITIONER -> `solver GAMG; smoother DIC;` at the same
#                   tolerance 2e-09: the preconditioner form is not ported yet
# It is named in interFoam_dynamicMesh's manifest entry as what this gate does not claim.
#
# PROFILES, ten steps of deltaT 2e-4 (the tutorial's is 1e-4 under maxCo 0.5; at 2e-4 the Courant
# number reaches 0.16 and the walls move a cell width in about thirty steps):
#   mixer          as above, with the tutorial's nAlphaSubCycles 3 -- the sub-cycle reads fvMesh::Vsc
#                  and Vsc0 at its own clock
#   mixerCorr      MULESCorr yes, nAlphaSubCycles 1: the implicit pre-solve's fvm::ddt and CMULES on
#                  the moving mesh
#   mixerOuter     nOuterCorrectors 2 with moveMeshOuterCorrectors yes: mesh.update() TWICE in one
#                  step, where V0 and oldPoints must be taken once (the once-per-time-index rule of
#                  fvMesh::movePoints and polyMesh::movePoints)
#   mixerOuterOnce nOuterCorrectors 2 without it: the second corrector must NOT move the mesh
#   mixerPred      momentumPredictor yes: U solved on the moving mesh (smoothSolver GaussSeidel, the
#                  tutorial's own U entry), with UFinal added because the tutorial names none
#   sloshing2D     laminar/sloshingTank2D as shipped but for p_rghFinal: the SDA roll-sway-heave of a
#                  tank whose chamfered corners make its cells NON-ORTHOGONAL by 44 degrees, with the
#                  tutorial's `Gauss linear corrected` laplacian and `corrected` snGrad -- the first
#                  fixture on which interFoam's non-orthogonal corrections are not zero -- 2-D with
#                  empty front and back, nAlphaSubCycles 3, cAlpha 1.5, ten of its own steps of 0.01
#   cylinder       laminar/sloshingCylinder as shipped but for p_rghFinal: a snappyHexMesh cylinder
#                  (polyhedra, 26 degrees) under an oscillatingLinearMotion and a rotatingMotion,
#                  MULESCorr and nNonOrthogonalCorrectors 1 -- the corrector loop on a corrected
#                  laplacian -- ten of its own steps of 0.001, with the oscillation's phase and
#                  vertical shifts zeroed (the staging says why: as shipped the mesh jumps 6.9 cm at
#                  the first update and OpenFOAM itself blows up at any fixed step)
#   closedDamBreak laminar/damBreak with its atmosphere WALLED OFF (U fixedValue 0, p_rgh
#                  fixedFluxPressure, alpha zeroGradient) and `pRefPoint (0.292 0.292 0.0073);
#                  pRefValue 0;` -- the pressure reference on a mesh that does not move, twenty of the
#                  tutorial's own steps of 0.001
#
# CONTROLS on the oracle: for the motion, OpenFOAM's own tank with `dynamicFvMesh staticFvMesh` --
# still water in a still tube, which is the whole of the answer away; for the reference, OpenFOAM's
# closed dam with `pRefValue 1e5`, which moves its p by 1e5 and nothing else.
#
# MEASURED, the five mixer profiles with the tutorial's own vanLeerV: every p_rgh iteration count
# OpenFOAM's (20 to 40 per profile), initial residuals 3.7e-11 in step one and 1.2e-09 over the run;
# alpha 4.1e-12, p_rgh 1.5e-13, U 4.2e-11 at worst; the wall velocity 7.0e-14 on 1050 faces (|U_wall|
# 1.8); the moved points OpenFOAM's exactly. (With `Gauss linear` staged in vanLeerV's place, before
# that scheme was ported: alpha 5.6e-12, U 1.0e-10.) The closed dam: 60 of 60 counts, alpha 1.2e-14, U 8.1e-14, p 5.7e-15.
#
# THE TWO NON-ORTHOGONAL TANKS: sloshing2D reads 20 of 20 counts, alpha 5.8e-15, p_rgh 3.2e-14, U
# 3.9e-13 (the motion is the whole answer: the still tank is 100% of U away); cylinder 40 of 40,
# alpha 6.9e-11, p_rgh 4.7e-12, U 1.0e-08. sloshing2D READ alpha 2.8e-06 AND 18 OF 20 COUNTS THE FIRST
# TIME, on a still tank as much as a moving one and with every scheme orthogonal: OpenFOAM pinned p_rgh
# in cell 460 and brae in 459 -- pRefPoint (0 0 0.15) lies ON the face between them, findCell takes the
# nearer centre, and the two centres differ in their 16th digit. fv_geometry's face centres were
# (1/3)*(sumAc/sumA) where primitiveMeshTools.C computes (1/3)*sumAc/sumA; with OpenFOAM's order V and
# C are OpenFOAM's to the bit (tests/mesh_motion_vs_openfoam.sh asserts it now) and so is the cell.
#
# THE CORRECTIONS THEMSELVES, BROKEN ONCE EACH (sloshing2D / cylinder):
#   the pressure laplacian's explicit correction dropped from the source   8.7e-02 / 2.1e-01 of alpha
#   its face flux dropped from p_rghEqn.flux()                             9.5e-02 / 2.6e-01
#   the viscous laplacian assembled orthogonal                              6.4e-10 / 2.7e-04
#   the three snGrads assembled orthogonal                                  1.3e-11 / 4.6e-04 -- the
#     tank's interface is horizontal and its alpha and rho gradients sit on faces the chamfers do
#     not touch, so the cylinder is the fixture for this one
#
# EVERY PORT DECISION WAS BROKEN ONCE, with switches that are not in the tree, on `mixer`:
#   phi left ABSOLUTE after the corrector (no makeRelative)     alpha 9.2e-01, U 470%
#   ddtCorr from phi.oldTime() instead of Sf & Uf.oldTime()     alpha 8.1e-02, U 28%
#   Uf left at interpolate(U) (no correctUf)                     alpha 3.3e-02, U 27%
#   gh and ghf not recomputed after the move                    alpha 3.3e-04, U 1.9e-03
#   the GAMG hierarchy kept from step one                       alpha 9.3e-06, 10 of 20 counts
#   the reference cell pinned at pRefValue, not its p_rgh       U 5.3e-07, 18 of 20 counts; on the
#                                                                closed dam U 4.4e-05, 15 of 60
#   the wall velocity as the face-centre displacement alone,    adjustPhi STOPS THE RUN with
#   without the swept volume in its normal component            OpenFOAM's own "continuity error
#                                                                cannot be removed" (adjustPhi.C:106)
#   the mesh moved on EVERY outer corrector (mixerOuterOnce)    alpha 5.1e-07, 29 of 40 counts
#   oldPoints re-taken on the second update (mixerOuter)        alpha 9.9e-02, U 100%
#
# WHAT THIS GATE CANNOT SEE, because the motion is RIGID: every cell keeps its volume to round-off, so
# V0 against V in fvm::ddt and MULES, and Vsc's interpolation inside a sub-cycle, change the answer by
# 1e-15 and no less. Those lines are transcribed from EulerDdtScheme.C, MULESTemplates.C and
# fvMeshGeometry.C, and are gated only when a mesh DEFORMS (the waveMaker tutorials, not yet run).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_moving_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
LAM="$TUT/multiphase/interFoam/laminar"

[ -x "$BIN" ]                 || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$LAM/testTubeMixer" ]   || { echo "SKIP: testTubeMixer tutorial not found under $LAM"; exit 77; }
[ -f "$OFBASHRC" ]            || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

HDR='FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }'

# stage <name> <tutorial> <deltaT> <nSteps> <profile>
stage()
{
    local name="$1" tutorial="$2" dt="$3" n="$4" profile="$5"
    local C="$W/$name"
    cp -r "$LAM/$tutorial" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    if [ -f "$C/system/blockMeshDict.m4" ]; then
        ( cd "$C" && m4 system/blockMeshDict.m4 > system/blockMeshDict ) || { echo "FAIL: m4 [$name]"; return 1; }
    fi
    DT="$dt" N="$n" PROFILE="$profile" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d = sys.argv[1]
dt, n, profile = os.environ['DT'], int(os.environ['N']), os.environ['PROFILE']

c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('adjustTimeStep', 'no'), ('deltaT', dt),
                 ('endTime', '%.10g' % (n*float(dt))), ('writeControl', 'timeStep'),
                 ('writeInterval', str(n)), ('writeFormat', 'ascii'), ('writePrecision', '18'),
                 ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)

q = os.path.join(d, 'system/fvSolution')
t = open(q).read()
if profile.startswith('mixer') or profile.startswith('sloshing2D') or profile.startswith('cylinder'):
    # the staging change the header declares
    m = re.search(r'p_rghFinal\s*\{.*?\n    \}', t, flags=re.S)
    assert m, 'no p_rghFinal entry'
    if 'preconditioner' in m.group(0):
        # the mixer and the sloshing tanks: PCG with a GAMG preconditioner, staged to GAMG with DIC
        assert 'GAMG' in m.group(0), 'p_rghFinal is no longer PCG with a GAMG preconditioner'
        t = t[:m.start()] + ('p_rghFinal\n    {\n        solver          GAMG;\n        smoother        DIC;\n'
                             '        tolerance       2e-09;\n        relTol          0;\n        maxIter         20;\n    }') + t[m.end():]
    else:
        # the cylinder: `$p_rgh; relTol 0; maxIter 20;` on a GAMG/DIC p_rgh, run as shipped
        assert '$p_rgh' in m.group(0), 'p_rghFinal is neither the preconditioner form nor $p_rgh'
    if profile == 'mixerCorr':
        # the pre-solve is solved with the Final entry, and the tutorial's key is the bare name
        t, k = re.subn(r'\n    alpha\.water\n    \{', '\n    "alpha.water.*"\n    {', t)
        assert k == 1, 'the alpha.water entry was not found'
        t, k = re.subn(r'nAlphaSubCycles\s+3;', 'nAlphaSubCycles 1;\n        MULESCorr       yes;\n        nLimiterIter    3;\n'
                       '        solver          smoothSolver;\n        smoother        symGaussSeidel;\n'
                       '        tolerance       1e-8;\n        relTol          0;', t)
        assert k == 1, 'nAlphaSubCycles not found'
    if profile in ('mixerOuter', 'mixerOuterOnce'):
        t, k = re.subn(r'nCorrectors\s+2;', 'nCorrectors     2;\n    nOuterCorrectors 2;'
                       + ('\n    moveMeshOuterCorrectors yes;' if profile == 'mixerOuter' else ''), t)
        assert k == 1, 'nCorrectors not found'
    if profile == 'mixerPred':
        t, k = re.subn(r'momentumPredictor\s+no;', 'momentumPredictor yes;', t)
        assert k == 1, 'momentumPredictor not found'
        t, k = re.subn(r'\n    U\n    \{', '\n    "U.*"\n    {', t)
        assert k == 1, 'the U entry was not found'
elif profile.startswith('closedDamBreak'):
    ref = '0' if profile == 'closedDamBreak' else '1e5'
    t, k = re.subn(r'(nNonOrthogonalCorrectors\s+0;)', r'\1\n    pRefPoint       (0.292 0.292 0.0073);\n    pRefValue       %s;' % ref, t)
    assert k == 1, 'PIMPLE block not found'
open(q, 'w').write(t)

if not profile.startswith('closedDamBreak'):
    # the three tanks name vanLeerV, run as shipped; the `*Static` controls hold the mesh still
    f = os.path.join(d, 'system/fvSchemes')
    t = open(f).read()
    assert re.search(r'div\(rhoPhi,U\)\s+Gauss vanLeerV;', t), 'div(rhoPhi,U) is no longer Gauss vanLeerV'
    p = os.path.join(d, 'constant/dynamicMeshDict')
    t = open(p).read()
    if profile.endswith('Static'):
        t, k = re.subn(r'dynamicFvMesh\s+dynamicMotionSolverFvMesh;', 'dynamicFvMesh   staticFvMesh;', t)
        assert k == 1, 'dynamicFvMesh not found'
    if profile.startswith('cylinder'):
        # the tutorial's oscillation carries `phaseShift 0.01` and `verticalShift (0.05 0 0)`, so its
        # mesh JUMPS 6.9 cm at the first update (the motion is absolute in t and not zero at t = 0):
        # OpenFOAM itself reads a Courant number of 44 after one step of 1e-4 and 79 after ten of
        # 1e-3, and only its adjustable step, collapsing, carries the shipped case. Both shifts are
        # zeroed here so the motion starts from rest; everything else is the tutorial's.
        t, k = re.subn(r'phaseShift\s+[^;]+;', 'phaseShift    0;', t)
        assert k == 1, 'phaseShift not found'
        t, k = re.subn(r'verticalShift\s+\([^)]*\);', 'verticalShift (0 0 0);', t)
        assert k == 1, 'verticalShift not found'
    open(p, 'w').write(t)
else:
    # the atmosphere walled off: no patch fixes p_rgh, so it needs the reference
    for fld, body in [('U', 'type            fixedValue;\n        value           uniform (0 0 0);'),
                      ('p_rgh', 'type            fixedFluxPressure;\n        value           uniform 0;'),
                      ('alpha.water.orig', 'type            zeroGradient;'),
                      ('alpha.water', 'type            zeroGradient;')]:
        p = os.path.join(d, '0', fld)
        if not os.path.exists(p):
            continue
        t = open(p).read()
        t, k = re.subn(r'atmosphere\s*\{[^}]*\}', 'atmosphere\n    {\n        %s\n    }' % body, t, flags=re.S)
        assert k == 1, 'atmosphere entry not found in ' + fld
        open(p, 'w').write(t)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    if [ -f "$C/system/snappyHexMeshDict" ]; then
        mkdir -p "$C/constant/triSurface"
        cp -f "$TUT/resources/geometry/$tutorial.obj.gz" "$C/constant/triSurface/" || { echo "FAIL: no surface for $tutorial"; return 1; }
        ( cd "$C" && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 ) || { echo "FAIL: snappyHexMesh [$name]"; tail -20 "$C/log.snappyHexMesh"; return 1; }
    fi
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$name]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$name]"; tail -30 "$C/log.interFoam"; return 1; }
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    [ -d "$C/$end" ] || { echo "FAIL: OpenFOAM wrote no $end directory [$name]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $n steps of deltaT $dt to t = $end   [$name]"
}

# gate <name> <deltaT> <nSteps> <profile> <control case>
gate()
{
    local name="$1" dt="$2" n="$3" profile="$4" control="$5"
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    "$BIN" "$W/$name" "$W/$name/0" "$W/$name/$end" "$n" "$W/$name/log.interFoam" "$profile" \
           "$W/$control/$end"
}

rc=0
stage mixerStatic    testTubeMixer 2e-4  10 mixerStatic    || rc=1
stage mixer          testTubeMixer 2e-4  10 mixer          || rc=1
stage mixerCorr      testTubeMixer 2e-4  10 mixerCorr      || rc=1
stage mixerOuter     testTubeMixer 2e-4  10 mixerOuter     || rc=1
stage mixerOuterOnce testTubeMixer 2e-4  10 mixerOuterOnce || rc=1
stage mixerPred      testTubeMixer 2e-4  10 mixerPred      || rc=1
stage sloshing2DStatic sloshingTank2D 0.01  10 sloshing2DStatic || rc=1
stage sloshing2D     sloshingTank2D 0.01  10 sloshing2D     || rc=1
stage cylinderStatic sloshingCylinder 0.001 10 cylinderStatic || rc=1
stage cylinder       sloshingCylinder 0.001 10 cylinder      || rc=1
stage closedRef1e5   damBreak/damBreak 0.001 20 closedDamBreakRef || rc=1
stage closedDamBreak damBreak/damBreak 0.001 20 closedDamBreak    || rc=1
[ $rc = 0 ] || { echo "interfoam_moving_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path
grep -q "Constructed SBMF 1 : rotatingBox of type oscillatingRotatingMotion" "$W/mixer/log.interFoam" \
    || { echo "FAIL: OpenFOAM's mixer log does not build the multiMotion"; exit 1; }
grep -q "Selecting dynamicFvMesh staticFvMesh" "$W/mixerStatic/log.interFoam" \
    || { echo "FAIL: OpenFOAM's control did not hold the mesh still"; exit 1; }
grep -q "^Courant Number mean: 0.0[1-9]" "$W/mixer/log.interFoam" \
    || { echo "FAIL: OpenFOAM's mixer never moved its fluid"; exit 1; }

gate mixer          2e-4  10 mixer          mixerStatic  || rc=1
gate mixerCorr      2e-4  10 mixerCorr      mixerStatic  || rc=1
gate mixerOuter     2e-4  10 mixerOuter     mixerStatic  || rc=1
gate mixerOuterOnce 2e-4  10 mixerOuterOnce mixerStatic  || rc=1
gate mixerPred      2e-4  10 mixerPred      mixerStatic  || rc=1
gate sloshing2D     0.01  10 sloshing2D     sloshing2DStatic || rc=1
gate cylinder       0.001 10 cylinder       cylinderStatic   || rc=1
gate closedDamBreak 0.001 20 closedDamBreak closedRef1e5 || rc=1

echo "interfoam_moving_vs_openfoam: rc $rc"
exit $rc
