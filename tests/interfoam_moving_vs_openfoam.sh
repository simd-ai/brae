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
# fixedFluxPressure everywhere, PIMPLE naming pRefPoint and pRefValue 1e5, `div(rhoPhi,U) Gauss
# vanLeerV`, and p_rghFinal as PCG with a GAMG PRECONDITIONER -- ALL AS SHIPPED. The last two were
# staged away (to `Gauss linear` and to `solver GAMG; smoother DIC;`) until each was ported; the
# preconditioner is held on its own in tests/interfoam_gamg_vs_openfoam.sh.
#
# PROFILES, ten steps of deltaT 2e-4 (the tutorial's is 1e-4 under maxCo 0.5; at 2e-4 the Courant
# number reaches 0.16 and the walls move a cell width in about thirty steps):
#   mixer          as above, with the tutorial's nAlphaSubCycles 3 -- the sub-cycle reads fvMesh::Vsc
#                  and Vsc0 at its own clock. ON BOTH ARMS, AS SHIPPED: p_rgh is `solver GAMG;
#                  smoother DIC` and p_rghFinal is `solver PCG; preconditioner { preconditioner GAMG;
#                  ... smoother DICGaussSeidel; nPreSweeps 2; nVcycles 2 }`, and the device runs both
#                  (deviceGamgSolve and devicePcgGamgSolve). It ran neither when the device arm was
#                  first added here, so that arm was a `mixerDevice` profile with the pressure solver
#                  staged to PCG/DIC; the staging is gone and the arm runs the case's own entries.
#                  MEASURED, device against OpenFOAM: 20 of 20 p_rgh counts, alpha 1.3e-12, p_rgh
#                  4.1e-14, U 1.2e-11, Uf 7.2e-12, the walls 7.0e-14 -- the host arm's own distances
#                  being 6.1e-12, 1.0e-13, 3.2e-11 and 3.1e-11. With the solver staged away it read
#                  2.2e-12, 1.2e-13, 1.2e-10 and 8.8e-11
#   mixerCorr      MULESCorr yes, nAlphaSubCycles 1: the implicit pre-solve's fvm::ddt and CMULES on
#                  the moving mesh
#   mixerOuter     nOuterCorrectors 2 with moveMeshOuterCorrectors yes: mesh.update() TWICE in one
#                  step, where V0 and oldPoints must be taken once (the once-per-time-index rule of
#                  fvMesh::movePoints and polyMesh::movePoints)
#   mixerOuterOnce nOuterCorrectors 2 without it: the second corrector must NOT move the mesh
#   mixerPred      momentumPredictor yes: U solved on the moving mesh (smoothSolver GaussSeidel, the
#                  tutorial's own U entry), with UFinal added because the tutorial names none
#   sloshing2D     laminar/sloshingTank2D AS SHIPPED: the SDA roll-sway-heave of a
#                  tank whose chamfered corners make its cells NON-ORTHOGONAL by 44 degrees, with the
#                  tutorial's `Gauss linear corrected` laplacian and `corrected` snGrad -- the first
#                  fixture on which interFoam's non-orthogonal corrections are not zero -- 2-D with
#                  empty front and back, nAlphaSubCycles 3, cAlpha 1.5, ten of its own steps of 0.01
#   sloshing2D3DoF, sloshing3D, sloshing3D3DoF, sloshing3D6DoF
#                  the rest of the sloshing tanks AS SHIPPED, ten steps of 0.01 each against its own
#                  static control: the SDA motion with the 3DoF coefficients in 2-D, the SDA tank in 3-D
#                  (19 x 40 x 34, 25,840 cells) with both coefficient sets, and tabulated6DoFMotion
#                  reading constant/6DoF.dat. MEASURED: U 4.4e-13, 2.7e-12, 6.8e-13, 5.4e-13; alpha to
#                  1.3e-14; the moved points exactly OpenFOAM's; the static control 100% of U on each.
#                  BROKEN ONCE EACH (U, 2D3DoF / 3D / 3D6DoF): SDA without its lamda rescaling 8.2e-01 /
#                  8.2e-01 / green; SDA without its roll 9.9e-01 / 9.9e-01 / green; the 6DoF table's
#                  interpolation flipped green / green / 1.4e-02
#   cylinder       laminar/sloshingCylinder, ON BOTH ARMS: a snappyHexMesh cylinder
#                  (polyhedra, 26 degrees) under an oscillatingLinearMotion and a rotatingMotion,
#                  MULESCorr and nNonOrthogonalCorrectors 1 -- the corrector loop on a corrected
#                  laplacian -- ten of its own steps of 0.001, with the oscillation's phase and
#                  vertical shifts zeroed (the staging says why: as shipped the mesh jumps 6.9 cm at
#                  the first update and OpenFOAM itself blows up at any fixed step).
#                  ITS p_rgh IS `solver GAMG; smoother DIC` AS SHIPPED, which is why the device arm is
#                  here: a moving mesh with the case's own GAMG. MEASURED, device against OpenFOAM:
#                  40 of 40 p_rgh V-cycle counts, alpha 8.8e-11, p_rgh 6.3e-12, U 9.8e-09, Uf 6.2e-09
#                  -- the host arm's own distances being 6.3e-11, 7.6e-12, 1.7e-08 and 1.1e-08.
#   solitaryGamg   waveMakerSolitary with p_rgh and p_rghFinal as `solver GAMG; smoother DIC;
#                  nCellsInCoarsestLevel 200`, ON BOTH ARMS: a DEFORMING mesh with a GAMG pressure
#                  solve, and a case whose MOTION solver is GAMG too (`cellDisplacement`,
#                  nCellsInCoarsestLevel 10). OpenFOAM keeps ONE GAMGAgglomeration per mesh, so the
#                  displacement solve at the first mesh update builds the hierarchy and p_rgh reuses
#                  it -- the 200 is never read. MEASURED: 60 of 60 counts on both arms, device alpha
#                  2.9e-12, p_rgh 3.4e-13, U 2.2e-11, Uf 2.7e-11 (host 3.0e-12, 3.4e-13, 2.3e-11,
#                  2.7e-11).
#                  THE DEVICE GAMG ON A MOVING MESH, BROKEN ONCE EACH:
#                    the hierarchy uploaded once and kept for the run, where OpenFOAM's
#                    GAMGAgglomeration::movePoints sets requireUpdate_ and the next New builds it
#                    again (GAMGAgglomeration.C:311-330, :498-516)
#                                              `cylinder`: 38 of 40 counts, alpha 2.9e-04, U 4.1e-03
#                    p_rgh building a hierarchy of its OWN instead of sharing the run's
#                                          `solitaryGamg`: 48 of 60 counts, alpha 3.3e-04, U 4.0e-01
#                                              (`cylinder` cannot see this one: its pcorr is PCG and
#                                              its motion is solid-body, so p_rgh builds the only
#                                              hierarchy there whether it shares or not)
#   closedDamBreak laminar/damBreak with its atmosphere WALLED OFF (U fixedValue 0, p_rgh
#                  fixedFluxPressure, alpha zeroGradient) and `pRefPoint (0.292 0.292 0.0073);
#                  pRefValue 0;` -- the pressure reference on a mesh that does not move, twenty of the
#                  tutorial's own steps of 0.001
#   closedDamBreakInitU  the same tank STARTED MOVING, U (0.1 0 0) in every cell against walls at rest:
#                  the phi createFields builds is not divergence-free, so initCorrectPhi's pcorr solve
#                  has work to do (79 iterations in both codes); the control is the tank at rest
#   mixerCorrectPhi, sloshing2DCorrectPhi, cylinderCorrectPhi
#                  the three solid-body tanks with `correctPhi yes`: phi rebuilt as Sf & Uf after every
#                  mesh update and CorrectPhi solved against it with the last corrector's rAU -- closed
#                  tanks, so through adjustPhi and pcorr's reference; the tank's non-orthogonal pcorr
#                  laplacian; and the cylinder's non-orthogonal corrector, pcorr then pcorrFinal. The
#                  cylinder's pcorr is staged `maxIter 1000`: the tutorial's 100 stops PCG UNCONVERGED
#                  after every update in both codes, at residuals of 0.1 to 0.3 agreeing to four digits,
#                  and an unconverged Krylov iterate carries every last-bit difference forward --
#                  as shipped 22 of 22 counts and alpha 7.6e-09; converged (150 to 176 iterations, the
#                  same in both) alpha 1.2e-10
#   piston, flap   laminar/waves/waveMakerPiston and waveMakerFlap, thirty steps of 0.01: the paddle
#                  deforming the mesh, `Gauss interfaceCompression` on div(phirb,alpha), correctPhi, an
#                  absorbing outlet -- with p_rgh, p_rghFinal and pcorr CONVERGED to 1e-13 (the staging
#                  says why: as shipped the piston's final corrector takes 135 PCG iterations and pcorr
#                  350 to 480, and last-bit differences ride them out to U 3.9e-07). Converged, solves of
#                  150 to 480 iterations still stop an iteration or three apart -- 194 against 197 at the
#                  piston's last step, both below 1e-13 -- so for these two profiles every solve must end
#                  below its tolerance in both codes and the counts must match on short solves and be
#                  within 2% on long ones. MEASURED: alpha 1.5e-12 and 1.0e-11, U 1.6e-09 and 1.1e-09.
#   multiPiston,   laminar/waves/waveMakerMultiPaddlePiston and waveMakerMultiPaddleFlap AS SHIPPED, on
#   multiFlap      their own 448000-cell 3-D mesh, thirty steps of 0.01: four paddles at 45 degrees,
#                  interfaceCompression, correctPhi, and GAMG (DICGaussSeidel) AS THE SOLVER of pcorr and
#                  p_rgh -- which the case writes as `p_rgh { $pcorr; ... }` against a
#                  `"(pcorr|pcorrFinal)"` key, a keyword reference OpenFOAM resolves through PATTERNS
#                  (dictionary.C:415-443). brae's expander matched literal keys only, read no solver for
#                  p_rgh and ran PBiCGStab under a notice; tests/test_dict_scoped_macro.cu holds the rule.
#                  GAMG's solves are 0 to 21 cycles, so every count is asserted equal, as shipped.
#                  MEASURED (piston, flap): all 90 p_rgh and all 31 pcorr counts OpenFOAM's in both;
#                  alpha 5.7e-13 and 2.9e-12, p_rgh 5.6e-13 and 2.6e-12, U 2.2e-10 and 1.8e-09; the
#                  moved points 1.1e-16 of the extent; the paddles' wall velocity 2.5e-14 on 3.9e-02
#                  and 3.3e-14 on 2.1e-01 m/s. The control, OpenFOAM with its mesh held still: U 100%.
#                  BROKEN ONCE, the pattern lookup taken out of the reader (piston): 48 of 90 p_rgh
#                  counts, 19 of 31 pcorr, alpha 9.7e-04, U 1.0e-01.
#                  Thirty steps and not ten because of the wall check: the paddle is on a 2 s ramp, and
#                  at t = 0.1 the piston's wall moves at 1.5e-02 m/s, where the 2.3e-14 that point
#                  round-off over deltaT leaves is 1.5e-12 of it -- over the 1e-12 bound, which stays.
#   solitary       laminar/waves/waveMakerSolitary AS SHIPPED, thirty steps of 0.01: the first gated
#                  case whose cells CHANGE VOLUME -- displacementLaplacian from a solitary-wave paddle --
#                  under correctPhi (its default), with an absorbing waveVelocity outlet and a
#                  totalPressure atmosphere; an OPEN tank, so no reference; the control holds its mesh
#                  still. BOTH ARMS: the tutorial's p_rgh and pcorr are PCG with DIC, which the device
#                  runs natively, so NOTHING is staged away for it -- the device arm
#                  runs the case's own fvSolution. What it does not cover is the tolerance those solves
#                  are given: 1e-6 with relTol 0, the tutorial's own, which is where two Krylov
#                  implementations stop rather than what they discretise -- though on this case it is
#                  not what separates the arms: tightened to 1e-13/relTol 0 the device moved from
#                  8.4e-10 to 8.4e-10 in alpha, and the three defects below were found instead.
#                  MEASURED, device against OpenFOAM: alpha 2.9e-12, p_rgh 3.3e-13, U 2.2e-11, Uf
#                  2.5e-11, the paddle's wall velocity 1.1e-14 -- the host arm's own distances being
#                  2.9e-12, 3.3e-13, 1.8e-10 and 2.2e-10, which is what the device checks are bounded
#                  by. Both arms end on the same mesh.
#                  BROKEN ONCE EACH, on the device arm -- all three are DEFORMING-MESH defects that the
#                  solid-body mixer cannot see, because there V == V0 and Vsc == V:
#                    mesh.V() in mesh.Vsc()'s and Vsc0()'s place, in MULES' limiter budgets, its
#                    explicit solve, every surfaceIntegrate and the pre-solve's fvm::ddt
#                    (MULESTemplates.C:248 and :397-417, fvcSurfaceIntegrate.C:77, EulerDdtScheme.C:
#                    383-392)                                    alpha 2.6e-03, U 5.1e-02, Uf 5.7e-02
#                    rAU left at the value createFields wrote, where the NEXT mesh update solves
#                    CorrectPhi with fvc::interpolate(rAU) of the LAST corrector (interFoam.C:138)
#                                                                alpha 2.5e-02, U 3.9e-01, Uf 3.9e-01
#                    mixture.correct() on the MOVED mesh not carried to the device, so the alpha
#                    equation convected with the interface normal of the mesh as it stood BEFORE the
#                    move (interFoam.C:141)                       alpha 8.4e-10, U 7.6e-09, Uf 5.6e-09
#                  THE LAST ONE IS WHY THE DEVICE CHECKS ARE BOUNDED BY THE HOST ARM AND NOT BY A FIXED
#                  NUMBER: at alpha 8.4e-10 and U 7.6e-09 it passes every fixed bound on this page.
#
# CORRECTPHI, BROKEN ONCE EACH (the three *CorrectPhi tanks; closedDamBreakInitU where it says so):
#   correctUphiBCs skipped -- phi on the walls left at Sf & Uf       adjustPhi STOPS THE RUN on all three
#                                                                    ("continuity error cannot be removed")
#   phi left as it was instead of rebuilt from Sf & Uf              the same
#   rAUf = 1 instead of interpolate(rAU) of the last corrector       alpha 3.6e-02, 18 of 22 pcorr counts
#   no adjustPhi and no reference for pcorr                          alpha 4.1e-07; InitU 1.3e-08
#   phi left ABSOLUTE after CorrectPhi                               alpha 9.9e-01
#   pcorr's non-orthogonal face flux dropped from its flux()         alpha 4.9e-01 on the cylinder; ZERO on
#                                                                    the others, where pcorr starts at 0 and
#                                                                    one pass leaves the correction zero
#   initCorrectPhi skipped                                           InitU alpha 3.2e-02, U 15%, and the pcorr
#                                                                    count sequence off on every profile
# NOT DISCRIMINATED, and not claimed: the curvature pass mixture.correct() makes after CorrectPhi (it
# moves these tanks' alpha by less than 3e-13), and the corrected flux handed to flux-conditional
# patches (a closed tank has none).
#
# THE DEFORMING MESH, BROKEN ONCE EACH on solitary: fvm::ddt(rho, U)'s source with V for V0, U 1.1e-02;
# the alpha sub-cycle's Vsc and Vsc0 as V, alpha 2.6e-03 and U 5.1e-02; and the outlet's wave model
# updated in UEqn rather than at the mesh update (inter_waves_cpp.cuh), U 2.0e-05 and alpha 1.3e-08.
# MEASURED on solitary: 60 of 60 p_rgh and 31 of 31 pcorr counts, alpha 2.9e-12, p_rgh 3.3e-13, U 1.7e-10;
# the three *CorrectPhi tanks alpha 1.4e-12, 3.0e-13, 1.2e-10 and U 9.9e-12, 8.2e-13, 7.7e-09; the
# start-moving dam alpha 1.2e-14, U 2.3e-12.
#
# CONTROLS on the oracle: for the motion, OpenFOAM's own tank with `dynamicFvMesh staticFvMesh` --
# still water in a still tube, which is the whole of the answer away; for the reference, OpenFOAM's
# closed dam with `pRefValue 1e5`, which moves its p by 1e5 and nothing else.
#
# MEASURED, the five mixer profiles as shipped: every p_rgh iteration count OpenFOAM's (20 to 40 per
# profile), initial residuals 5.8e-11 in step one and 1.6e-09 over the run; alpha 6.1e-12, U 7.0e-11
# at worst; the wall velocity 7.0e-14 on 1050 faces (|U_wall| 1.8); the moved points OpenFOAM's
# exactly. (With p_rghFinal and div(rhoPhi,U) staged, before either was ported: alpha 5.6e-12, U
# 1.0e-10.) The closed dam: 60 of 60 counts, alpha 1.2e-14, U 8.1e-14, p 5.7e-15.
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
# V0 AGAINST V in fvm::ddt and MULES, and Vsc's interpolation inside a sub-cycle, change the rigid tanks'
# answer by 1e-15 and no less -- every cell keeps its volume -- and are gated by `solitary`, whose
# cells change theirs (the deforming-mesh arms above).
#
# PROFILE pistonSST: waves/waveMakerPiston made kOmegaSST -- the paddle a `wall`, so the closure's wall
# distance moves with it, and the top an ordinary patch -- with k and omega solved to 1e-12; its control is
# the laminar piston. kOmegaSST on a moving mesh takes the old volumes in fvm::ddt, the absolute flux in divU,
# and y recomputed after every motion; and under displacementLaplacian y is NOT the distance to the walls:
# the inverseDistance diffusivity registers the `wallDist` mesh object over its own patches first, and
# kOmegaSSTBase's wallDist::New(mesh) finds that one (MeshObject::New looks up the type name alone).
# MEASURED: k 8.6e-12, omega 2.4e-12, nut 3.0e-10, U 1.2e-10; the closure moves OpenFOAM's U by 1.9.
# BROKEN ONCE EACH (nut): y from the wall patches 6.4e-01, V0 dropped 2.0e-03, divU relative 4.8e-03,
# y not recomputed 9.5e-04.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_moving_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
LAM="$TUT/multiphase/interFoam/laminar"

[ -x "$BIN" ]                 || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$LAM/testTubeMixer" ]   || { echo "SKIP: testTubeMixer tutorial not found under $LAM"; exit 77; }
[ -f "$OFBASHRC" ]            || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

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
if profile.startswith('mixer') or profile.startswith('sloshing') or profile.startswith('cylinder'):
    # the staging change the header declares
    m = re.search(r'p_rghFinal\s*\{.*?\n    \}', t, flags=re.S)
    assert m, 'no p_rghFinal entry'
    if 'preconditioner' in m.group(0):
        # the mixer and the sloshing tanks: PCG with a GAMG preconditioner, run as shipped
        assert 'GAMG' in m.group(0), 'p_rghFinal is no longer PCG with a GAMG preconditioner'
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
    if profile.endswith('CorrectPhi'):
        # PIMPLE's `correctPhi`, which every solid-body tutorial writes as `no`
        t, k = re.subn(r'correctPhi\s+no;', 'correctPhi      yes;', t)
        assert k == 1, 'correctPhi no; not found'
    if profile == 'cylinderCorrectPhi':
        # the tutorial caps pcorr at `maxIter 100`, and after every mesh update PCG stops there
        # UNCONVERGED in both codes, at final residuals of 0.1 to 0.3 that agree to four digits. The
        # iterate a Krylov solver leaves unconverged carries every last-bit difference in its input
        # forward, amplified: as shipped the cylinder reads 22 of 22 pcorr counts and alpha 7.6e-09,
        # U 1.1e-07. Allowed to converge (150 to 176 iterations, the same in both), alpha 1.2e-10.
        t, k = re.subn(r'maxIter\s+100;', 'maxIter 1000;', t)
        assert k == 1, 'pcorr maxIter 100 not found'
    if profile == 'mixerPred':
        t, k = re.subn(r'momentumPredictor\s+no;', 'momentumPredictor yes;', t)
        assert k == 1, 'momentumPredictor not found'
        t, k = re.subn(r'\n    U\n    \{', '\n    "U.*"\n    {', t)
        assert k == 1, 'the U entry was not found'
elif profile.startswith('solitary') or profile.startswith('multi'):
    # waves/waveMakerSolitary and the two multi-paddle tanks AS SHIPPED: nothing to stage in fvSolution
    if profile == 'solitaryGamg':
        # ...except p_rgh, which this profile gives the case's OTHER GAMG entry -- the motion
        # solver's is `nCellsInCoarsestLevel 10` -- with a COARSEST LEVEL OF ITS OWN. OpenFOAM keeps
        # one GAMGAgglomeration per mesh (a MeshObject, found by type name), so the displacement
        # solve at the first mesh update builds the hierarchy and p_rgh reuses THAT one: the 200
        # below is never read. An arm that built a second hierarchy from this entry would solve on
        # a different one and read different V-cycle counts.
        for key in ('p_rgh', 'p_rghFinal'):
            m = re.search(r'\n    %s\s*\{[^}]*\}' % re.escape(key), t)
            assert m, 'the %s entry was not found' % key
            t = t.replace(m.group(0),
                          '\n    %s\n    {\n        solver          GAMG;\n'
                          '        smoother        DIC;\n        tolerance       1e-8;\n'
                          '        relTol          0;\n'
                          '        nCellsInCoarsestLevel 200;\n    }' % key)
        assert 'solver          GAMG' in t, 'the GAMG profile did not take' 
elif profile.startswith('piston') or profile.startswith('flap'):
    # the pressure solves converged: p_rgh, p_rghFinal and pcorr at tolerance 1e-13, relTol 0. As shipped
    # (p_rgh relTol 0.05, pcorr 1e-10) the piston's third corrector takes 135 PCG iterations and pcorr
    # 350 to 480, and last-bit differences ride those solves out to U 3.9e-07 and alpha 5.0e-09 after
    # thirty steps; converged, 8.3e-11 and 7.6e-14. The motion solve is left as shipped.
    for key in ('p_rgh', 'p_rghFinal', '"(pcorr|pcorrFinal)"'):
        m = re.search(r'\n    %s\s*\{[^}]*\}' % re.escape(key), t)
        assert m, 'the %s entry was not found' % key
        body = m.group(0)
        body = re.sub(r'tolerance\s+[^;]+;', 'tolerance       1e-13;', body)
        body = re.sub(r'relTol\s+[^;]+;', 'relTol          0;', body)
        if 'tolerance' not in body:
            body = body.replace('}', '    tolerance       1e-13;\n        relTol          0;\n    }')
        t = t.replace(m.group(0), body)
elif profile.startswith('closedDamBreak'):
    ref = '1e5' if profile == 'closedDamBreakRef' else '0'
    t, k = re.subn(r'(nNonOrthogonalCorrectors\s+0;)', r'\1\n    pRefPoint       (0.292 0.292 0.0073);\n    pRefValue       %s;' % ref, t)
    assert k == 1, 'PIMPLE block not found'
open(q, 'w').write(t)

if not profile.startswith('closedDamBreak'):
    # the three tanks name vanLeerV, run as shipped; the `*Static` controls hold the mesh still
    f = os.path.join(d, 'system/fvSchemes')
    t = open(f).read()
    if profile.startswith('piston') or profile.startswith('flap') or profile.startswith('multi'):
        assert re.search(r'div\(phirb,alpha\)\s+Gauss interfaceCompression;', t), \
            'div(phirb,alpha) is no longer Gauss interfaceCompression'
    elif not profile.startswith('solitary'):
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
        if fld == 'U' and profile == 'closedDamBreakInitU':
            # a closed tank that STARTS MOVING: U (0.1 0 0) in every cell against walls at rest, so the
            # phi createFields builds is not divergence-free and initCorrectPhi has work to do
            t, k = re.subn(r'internalField\s+uniform\s*\(0 0 0\);', 'internalField   uniform (0.1 0 0);', t)
            assert k == 1, 'U internalField not found'
        open(p, 'w').write(t)
PYEOF
    if [ "$profile" = pistonSST ]; then
        PROFILE="$profile" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging kOmegaSST into $name"; return 1; }
import os, re, sys
d = sys.argv[1]
def sub(path, pat, rep):
    t = open(path).read()
    t2, k = re.subn(pat, rep, t, count=1)
    assert k == 1, (path, pat)
    open(path, 'w').write(t2)
# THE PADDLE A WALL, so kOmegaSST's wall distance moves with it -- and the top, a `wall` in the tutorial
# with an atmosphere's conditions on it, an ordinary patch
bm = os.path.join(d, 'system/blockMeshDict')
sub(bm, r'(leftwall\s*\{\s*type\s+)patch;', r'\1wall;')
sub(bm, r'(top\s*\{\s*type\s+)wall;', r'\1patch;')
sub(os.path.join(d, 'constant/turbulenceProperties'), r'simulationType\s+laminar;',
    'simulationType  RAS;\n\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n}')
sc = os.path.join(d, 'system/fvSchemes')
sub(sc, r'(div\(rhoPhi,U\)[^\n]*\n)', r'\1    div(phi,k)      Gauss upwind;\n    div(phi,omega)  Gauss upwind;\n')
open(sc, 'a').write('\nwallDist\n{\n    method meshWave;\n}\n')
# k and omega converged, as p_rgh is: at 1e-14 both codes ran k to its 1000-sweep cap
fs = os.path.join(d, 'system/fvSolution')
sub(fs, r'\n    U\n', '\n    "(k|omega).*"\n    {\n        solver          smoothSolver;\n'
    '        smoother        symGaussSeidel;\n        tolerance       1e-12;\n        relTol          0;\n'
    '    }\n\n    U\n')
hdr = ('FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volScalarField;\n    object %s;\n}\n'
       'dimensions %s;\ninternalField uniform %s;\nboundaryField\n{\n%s}\n')
def bf(wall, top):
    s = ''.join('    %s { %s }\n' % (w, wall) for w in ('bottom1', 'bottom2', 'leftwall'))
    s += '    top { %s }\n    rightwall { type zeroGradient; }\n    "(front|back)" { type empty; }\n' % top
    return s
for name, dim, v, wall, top in [
        ('k', '[0 2 -2 0 0 0 0]', '1e-4', 'type kqRWallFunction; value uniform 1e-4;',
         'type inletOutlet; inletValue uniform 1e-4; value uniform 1e-4;'),
        ('omega', '[0 0 -1 0 0 0 0]', '1', 'type omegaWallFunction; value uniform 1;',
         'type inletOutlet; inletValue uniform 1; value uniform 1;'),
        ('nut', '[0 2 -1 0 0 0 0]', '0', 'type nutkWallFunction; value uniform 0;',
         'type calculated; value uniform 0;')]:
    body = bf(wall, top)
    if name == 'nut':
        body = body.replace('rightwall { type zeroGradient; }', 'rightwall { type calculated; value uniform 0; }')
    open(os.path.join(d, '0', name), 'w').write(hdr % (name, dim, v, body))
PYEOF
    fi
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
stage sloshing2D3DoFStatic sloshingTank2D3DoF 0.01 10 sloshing2D3DoFStatic || rc=1
stage sloshing2D3DoF       sloshingTank2D3DoF 0.01 10 sloshing2D3DoF       || rc=1
stage sloshing3DStatic     sloshingTank3D     0.01 10 sloshing3DStatic     || rc=1
stage sloshing3D           sloshingTank3D     0.01 10 sloshing3D           || rc=1
stage sloshing3D3DoFStatic sloshingTank3D3DoF 0.01 10 sloshing3D3DoFStatic || rc=1
stage sloshing3D3DoF       sloshingTank3D3DoF 0.01 10 sloshing3D3DoF       || rc=1
stage sloshing3D6DoFStatic sloshingTank3D6DoF 0.01 10 sloshing3D6DoFStatic || rc=1
stage sloshing3D6DoF       sloshingTank3D6DoF 0.01 10 sloshing3D6DoF       || rc=1
stage cylinderStatic sloshingCylinder 0.001 10 cylinderStatic || rc=1
stage cylinder       sloshingCylinder 0.001 10 cylinder      || rc=1
stage mixerCorrectPhi      testTubeMixer    2e-4  10 mixerCorrectPhi      || rc=1
stage sloshing2DCorrectPhi sloshingTank2D   0.01  10 sloshing2DCorrectPhi || rc=1
stage cylinderCorrectPhi   sloshingCylinder 0.001 10 cylinderCorrectPhi   || rc=1
stage solitaryStatic waves/waveMakerSolitary 0.01 30 solitaryStatic || rc=1
stage solitary       waves/waveMakerSolitary 0.01 30 solitary       || rc=1
stage solitaryGamg   waves/waveMakerSolitary 0.01 30 solitaryGamg   || rc=1
stage pistonStatic   waves/waveMakerPiston   0.01 30 pistonStatic   || rc=1
stage piston         waves/waveMakerPiston   0.01 30 piston         || rc=1
stage pistonSST      waves/waveMakerPiston   0.01 30 pistonSST      || rc=1
stage flapStatic     waves/waveMakerFlap     0.01 30 flapStatic     || rc=1
stage flap           waves/waveMakerFlap     0.01 30 flap           || rc=1
stage multiPistonStatic waves/waveMakerMultiPaddlePiston 0.01 30 multiPistonStatic || rc=1
stage multiPiston       waves/waveMakerMultiPaddlePiston 0.01 30 multiPiston       || rc=1
stage multiFlapStatic   waves/waveMakerMultiPaddleFlap   0.01 30 multiFlapStatic   || rc=1
stage multiFlap         waves/waveMakerMultiPaddleFlap   0.01 30 multiFlap         || rc=1
stage closedRef1e5   damBreak/damBreak 0.001 20 closedDamBreakRef || rc=1
stage closedDamBreak damBreak/damBreak 0.001 20 closedDamBreak    || rc=1
stage closedDamBreakInitU damBreak/damBreak 0.001 20 closedDamBreakInitU || rc=1
[ $rc = 0 ] || { echo "interfoam_moving_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path
grep -q "Constructed SBMF 1 : rotatingBox of type oscillatingRotatingMotion" "$W/mixer/log.interFoam" \
    || { echo "FAIL: OpenFOAM's mixer log does not build the multiMotion"; exit 1; }
grep -q "Selecting dynamicFvMesh staticFvMesh" "$W/mixerStatic/log.interFoam" \
    || { echo "FAIL: OpenFOAM's control did not hold the mesh still"; exit 1; }
grep -q "^Courant Number mean: 0.0[1-9]" "$W/mixer/log.interFoam" \
    || { echo "FAIL: OpenFOAM's mixer never moved its fluid"; exit 1; }

for c in multiPiston multiFlap; do
    grep -q "^GAMG:  Solving for p_rgh" "$W/$c/log.interFoam" \
        || { echo "FAIL: OpenFOAM's $c log does not solve p_rgh with GAMG, the solver \$pcorr brings in"; exit 1; }
done

gate mixer          2e-4  10 mixer          mixerStatic  || rc=1
gate mixerCorr      2e-4  10 mixerCorr      mixerStatic  || rc=1
gate mixerOuter     2e-4  10 mixerOuter     mixerStatic  || rc=1
gate mixerOuterOnce 2e-4  10 mixerOuterOnce mixerStatic  || rc=1
gate mixerPred      2e-4  10 mixerPred      mixerStatic  || rc=1
gate sloshing2D     0.01  10 sloshing2D     sloshing2DStatic || rc=1
gate sloshing2D3DoF 0.01  10 sloshing2D3DoF sloshing2D3DoFStatic || rc=1
gate sloshing3D     0.01  10 sloshing3D     sloshing3DStatic     || rc=1
gate sloshing3D3DoF 0.01  10 sloshing3D3DoF sloshing3D3DoFStatic || rc=1
gate sloshing3D6DoF 0.01  10 sloshing3D6DoF sloshing3D6DoFStatic || rc=1
gate cylinder       0.001 10 cylinder       cylinderStatic   || rc=1
gate mixerCorrectPhi      2e-4  10 mixerCorrectPhi      mixerStatic      || rc=1
gate sloshing2DCorrectPhi 0.01  10 sloshing2DCorrectPhi sloshing2DStatic || rc=1
gate cylinderCorrectPhi   0.001 10 cylinderCorrectPhi   cylinderStatic   || rc=1
gate solitary       0.01  30 solitary       solitaryStatic || rc=1
# the deforming mesh with a GAMG pressure solve, on BOTH arms -- see the solitaryGamg staging
gate solitaryGamg   0.01  30 solitaryGamg   solitaryStatic || rc=1
gate piston         0.01  30 piston         pistonStatic   || rc=1
gate pistonSST      0.01  30 pistonSST      piston         || rc=1
gate flap           0.01  30 flap           flapStatic     || rc=1
gate multiPiston    0.01  30 multiPiston    multiPistonStatic || rc=1
gate multiFlap      0.01  30 multiFlap      multiFlapStatic   || rc=1
gate closedDamBreak 0.001 20 closedDamBreak closedRef1e5 || rc=1
gate closedDamBreakInitU 0.001 20 closedDamBreakInitU closedDamBreak || rc=1

echo "interfoam_moving_vs_openfoam: rc $rc"
exit $rc
