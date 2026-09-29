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
#   mixerPermeable the same tube with permeableAlphaPressureInletOutletVelocity and
#                  prghPermeableAlphaTotalPressure on its walls, its control the SAME MOTION under the
#                  tutorial's own movingWallVelocity. mesh.update() ends in
#                  U.correctBoundaryConditions() and phi is RELATIVE there and ABSOLUTE at pEqn's;
#                  brae read the absolute one at both and left U 1.33e-01 from OpenFOAM after two
#                  steps, 8.66e-11 with it fixed. THE DEVICE ARM RUNS IT (it was refused as unmeasured):
#                  alpha 3.9e-15, p_rgh 3.4e-14, U 4.5e-11 against the host's 2.6e-15 / 5.8e-14 /
#                  3.7e-11. BROKEN ONCE on the device -- prghPermeableAlphaTotalPressure handed the
#                  ABSOLUTE flux at constrainPressure, the refusal's own worry -- it reads U 9.0e-04
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
#   esd            RAS/electrostaticDeposition, TWO STEPS of 1e-3, and it is here for ALPHA'S PATCH
#                  VALUES -- the only thing this harness compares on a boundary rather than in a cell,
#                  and the only interFoam tutorial where OpenFOAM's under-relaxed alpha corrector is
#                  DISTINGUISHABLE from an evaluate. `alpha1 = 0.5*alpha1 + 0.5*alpha10`
#                  (VoF/alphaEqn.H:202) is a whole-field ASSIGNMENT, and its boundary half reaches each
#                  patch through that patch's own VIRTUAL operator= -- nothing on the path consults
#                  assignable(). A `variableHeightFlowRate` patch is mixed and overrides no operator=,
#                  so mixedFvPatchField.H:303-305 leaves its value ALONE; on an OUTFLOW face, where its
#                  valueFraction is 0, an evaluate would write the owner cell there instead.
#                  ITS MESH MOVES, which is why it belongs here: `solidBody tabulated6DoFMotion`, a
#                  RIGID translation at -0.08 m/s, so V == V0 and the volume weights are blind to it.
#                  54,390 cells (blockMesh 9,000 + snappyHexMesh), and the tutorial SHIPS its own
#                  surface (constant/triSurface/metalSheet.stl.gz), so the staging must not copy one
#                  from resources/. Its Allrun.pre also rotates the mesh AFTER setFields.
#                  MEASURED at step two, on side-03..side-06 (600 faces each; side-01 sits at the clamp
#                  and carries nothing): the CELLS agree to 4.6e-16 while OpenFOAM's own written patch
#                  value sits 5.1256e-10 off its own owner cell on 240 of the 2,625 faces -- six orders
#                  apart, so nothing else on the case can be responsible. brae reads 4.4409e-16.
#                  BEFORE THE FIX brae's patch value sat EXACTLY on OpenFOAM's owner cell: the two
#                  differences were the same 5.1256e-10 to five digits, which is what "brae evaluates
#                  where OpenFOAM assigns" looks like measured. It took BOTH sites -- the relaxation in
#                  alpha_eqn_cpp.cu and a second evaluate after the sub-cycle in inter_driver_cpp.cu,
#                  which had been making the first inert.
#                  TWO STEPS AND NOT TEN: this case also carries the OPEN `cellLimited leastSquares`
#                  grad(alpha) item, whose cell gap passes the patch difference from step five (2.2e-09
#                  at five, 2.9e-08 at ten). Those step counts measure the gradient item, not this one.
#                  THE DEVICE ARM RUNS IT (it refused the case's gradients until #28's device_gradLsq)
#                  and reads the host's own numbers, alpha 6.7e-16, p_rgh 1.2411e-07, U 4.8105e-10, and
#                  its patch values 4.4e-16. It took the relaxation's boundary half on that arm too
#                  (DeviceInterAlphaHooks::relaxBoundary) and dropping the evaluate in its
#                  interfaceForces hook: with neither, U 1.17e-07 and the patch on the owner cell
#                  (5.1256e-10); with the first alone the fields matched and the patch did not.
#                  BROKEN ONCE EACH on the device (the site handed Gauss linear): alpha's limiter
#                  gradient alpha 2.3e-09 / U 2.9e-07, nHat's p_rgh 1.8e+00 / U 2.9e-02, grad(U)'s dev2
#                  term p_rgh 9.4e-05 / U 1.5e-04. grad(p_rgh) and grad(rho) are inert here -- the
#                  laplacian is `orthogonal` and momentumPredictor is off.
#                  THE ORACLE IS ASSERTED TO HAVE TAKEN THE PATH -- at least 200 of its own written
#                  faces must be off its own cells -- because an oracle whose patch value equals its
#                  owner cell agrees with a re-evaluating brae by accident.
#   esdNoCorr      THE CONTROL: the same case with `MULESCorr no`, one dictionary entry, so the
#                  relaxation branch never runs. OpenFOAM's patch value goes back ON its owner cell: 0
#                  faces stale, worst 7.1054e-15, brae 1.9451e-16. As a control on the FIELDS it moves
#                  OpenFOAM's own alpha 8.1858e-09 and U 2.9337e-04 relative -- eleven orders above
#                  brae's own distance. THE DEVICE ARM REFUSES this case, on the cellLimited gradient,
#                  and the gate asserts the refusal rather than trusting it.
#   closedDamBreak laminar/damBreak with its atmosphere WALLED OFF (U fixedValue 0, p_rgh
#                  fixedFluxPressure, alpha zeroGradient) and `pRefPoint (0.292 0.292 0.0073);
#                  pRefValue 0;` -- the pressure reference on a mesh that does not move, twenty of the
#                  tutorial's own steps of 0.001
#   closedDamBreakInitU  the same tank STARTED MOVING, U (0.1 0 0) in every cell against walls at rest:
#                  the phi createFields builds is not divergence-free, so initCorrectPhi's pcorr solve
#                  has work to do (79 iterations in both codes); the control is the tank at rest.
#                  BOTH closed tanks run on the DEVICE arm too (U 5.5e-14 and 2.3e-12, the host's own).
#   closedAdjZG    adjustPhi's SCALING half (pEqn.H:21-26): the same dam with its atmosphere left OPEN
#   closedAdjIO    to an outflow adjustPhi can scale -- p_rgh zeroGradient, so it still needs the
#                  reference, and U zeroGradient (adjustable because fixesValue is false) or inletOutlet
#                  (fixesValue true, adjustable because isA<inletOutlet>): the two halves of
#                  adjustPhi.C:59, each the other's control. NO SHIPPED TUTORIAL can witness the scaling:
#                  every interFoam case that needs a reference has walls only, where adjustPhi is a
#                  balance check. MEASURED, host / device against OpenFOAM: U 3.5e-14 / 4.2e-14 and
#                  7.9e-14 / 8.4e-14; OpenFOAM's walled tank against ZG U 2.5e-03, ZG against IO 9.1e-04.
#                  BROKEN ONCE EACH: the device hook a no-op, U 7.0e-06 and 4.6e-05; the shared mask
#                  counting inletOutlet as fixed stops IO in adjustPhi (ZG unchanged), the mask counting
#                  ONLY inletOutlet as adjustable stops ZG (IO unchanged). NOT DISCRIMINATED: adjustPhi's
#                  place before phig -- phig is exactly zero on every boundary face these fixtures sum.
#   mixerTop       testTubeMixer with its top face made an open patch (U inletOutlet, p_rgh and alpha
#                  zeroGradient): the makeRelative/makeAbsolute ROUND TRIP around adjustPhi on a moving
#                  mesh. The top's relative flux is 3.3e-06 each way beside the walls' mesh flux of
#                  1.4e-03. MEASURED U host 2.9e-11, device 1.1e-11; control, the shipped closed tube, U
#                  2.0e-01. BROKEN ONCE EACH on the device: the hook a no-op U 8.4e-02, adjustPhi handed
#                  the ABSOLUTE flux U 2.1e-02.
#   floating       RAS/floatingObject: the first fixture here whose mesh is moved by the FLUID and not
#                  by a prescribed function -- rigidBodyMeshMotion integrating a cuboid on a
#                  `composite (Py Ry)` joint from the pressure and shear on its own patches, with
#                  `nOuterCorrectors 3` and `moveMeshOuterCorrectors yes`, so the body is solved three
#                  times per step from one frozen start state. Its SECOND oracle is the joint state
#                  OpenFOAM writes every step, which the static control cannot produce at all.
#                  STAGED, and each change measured: `ddt Euler` (the tutorial's CrankNicolson on a
#                  DEFORMING mesh is refused on both arms and held separately), `accelerationRelaxation
#                  0.7` (the shipped table is zero until t = 4, so the body would never leave rest),
#                  and converged pressure solves. MEASURED: alpha 1.3e-14, p_rgh 7.9e-14, U 6.0e-14,
#                  the body's q 8.3e-17 -- and the case AMPLIFIES, one ulp on every mesh point's y
#                  reaching U 3.3e-09 and q 2.0e-08 against OpenFOAM itself in the same ten steps.
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
#   piston, flap   laminar/waves/waveMakerPiston and waveMakerFlap, ON BOTH ARMS, thirty steps of
#                  0.01: the paddle
#                  deforming the mesh, `Gauss interfaceCompression` on div(phirb,alpha), correctPhi, an
#                  absorbing outlet -- with p_rgh, p_rghFinal and pcorr CONVERGED to 1e-13 (the staging
#                  says why: as shipped the piston's final corrector takes 135 PCG iterations and pcorr
#                  350 to 480, and last-bit differences ride them out to U 3.9e-07). Converged, solves of
#                  150 to 480 iterations still stop an iteration or three apart -- 194 against 197 at the
#                  piston's last step, both below 1e-13 -- so for these two profiles every solve must end
#                  below its tolerance in both codes and the counts must match on short solves and be
#                  within 2% on long ones. MEASURED: alpha 1.5e-12 and 1.0e-11, U 1.6e-09 and 1.1e-09.
#                  THE DEVICE ARM IS HERE FOR `div(phirb,alpha) Gauss interfaceCompression`, the
#                  PhiScheme four waveMakers name and the last alpha scheme that was host-only.
#                  MEASURED on the piston, device against OpenFOAM: alpha 3.2e-12, p_rgh 5.1e-12, U
#                  1.5e-09, Uf 1.6e-09, the paddle 1.1e-14 -- the host arm's own distances on the same
#                  profile. Its p_rgh counts: 17 of 90 one or two iterations apart, every solve ending
#                  below 1e-13 in both codes, which is why the device arm is allowed ONE iteration
#                  where OpenFOAM took at least twenty (the test says so at countsAgree). The HOST's
#                  rule is untouched: exact below a hundred, 2% above.
#                  THE RESIDUAL CURVES ARE COMPARED TOO, on every p_rgh solve of these profiles: the
#                  staging gives p_rgh `log 2;`, OpenFOAM prints its residual at every iteration
#                  (SolverPerformance.C:70-76), and brae's final residual must be within 15% of
#                  OpenFOAM's AT THE ITERATION BRAE STOPPED ON. MEASURED worst over a run: piston 2.2%
#                  host and 9.1% device, pistonSST 4.3%, flap 3.2% and 6.7%; the medians 0.0% to 1.8%.
#                  THE CONTROL is the same statistic one iteration EARLIER, which must break the bound:
#                  19% to 27% -- the curve falls about 8% an iteration, so it resolves a shift of one.
#                  It is here because a count cannot tell a defect from OpenFOAM's own plateau. On
#                  flap's step 7 OpenFOAM reads 1.0167e-13 at iteration 242, 1.0046e-13 at 243, climbs to
#                  1.0981e-13 and ends at 250; the device arm ended at 242 on 9.993e-14, 1.7% from
#                  OpenFOAM's residual THERE and eight iterations from its count. That arm started doing
#                  so when linearUpwind's grad(U) began reading U's stored patch values (the permeable
#                  wall's unit): its alpha went from 8.8e-13 to 9.9e-13, its U from 1.2e-10 to 5.5e-11 of
#                  OpenFOAM, its worst initial residual from 6.5e-05 to 2.6e-05. A count outside the rule
#                  is therefore accepted ONLY where brae stopped EARLIER and within 5% of OpenFOAM's
#                  residual at that iteration, on at most ONE solve of a run. BROKEN ONCE: the device's
#                  p_rgh tolerance 30% loose -- 61 of 90 counts apart by up to 11, eleven of them on a
#                  "plateau", and the at-most-one check, the count check and the convergence check all
#                  fail; the oracle WITHOUT `log 2` -- the parse check and the control both fail, and
#                  the count rule fails alone, as it did before there was a history.
#                  BROKEN ONCE EACH, on the device arm (piston):
#                    the QUADRATIC compression form, interfaceCompression.H's commented-out line,
#                    for the quartic one it ships        alpha 1.8e-02, U 1.1e-02, Uf 1.1e-02
#                    vanLeer in its place, which is what the mapping took before it named every
#                    scheme                              alpha 1.9e-02, U 1.2e-02, Uf 1.1e-02
#                    the limiter alone, without the (1 - limiter)*pos0(phi) half of
#                    limitedSurfaceInterpolationScheme's blend
#                                                        alpha 6.2e-02, U 3.8e-02, Uf 3.7e-02
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
# THE DEVICE ARM RUNS THIS PROFILE TOO, and it is the only arm here with a RAS closure on a mesh that
# moves. It took three terms, and the third is every DISTANCE the closure holds, recomputed after the
# motion: wallDist::New(mesh).y() for F1/F2, nearWallDist for the wall functions, and DeviceWallData's
# own y, deltaCoeffs and wall velocity (refreshDeviceInterTurbulenceGeometry, called after the host
# block's moveInterTurbulence). MEASURED, device: alpha 3.8e-12, p_rgh 3.7e-12, U 3.4e-10, Uf 3.3e-10,
# all four bounded BY THE HOST ARM and not by a constant.
# BROKEN ONCE, on the device arm: the refresh skipped (the host's kept, so this is the device's own
# stale distance and nothing else) -- alpha 4.2494e-08, p_rgh 4.2660e-08, U 1.7353e-05, 4 failures.
#
# PROFILE pistonLES: the SAME paddle under LES kEqn, on BOTH arms. Its third moving-mesh term is the
# one the RAS closures do not have -- the FILTER WIDTH. cubeRootVolDelta is (deltaCoeff*V)^(1/3) per
# cell; LESModel::correct() calls delta_().correct() first (LESModel.C:251) and
# cubeRootVolDelta::correct() recomputes it whenever the mesh is changing (cubeRootVolDelta.C:128-134).
# Both brae arms took it once, at start-up, and the host LES closure was also missing the other two
# terms (V0 in the ddt source, the absolute flux in divU). k reaches U through nut alone, so this
# profile compares k and nut and not only the three shared fields.
# MEASURED: host alpha 1.4e-12, p_rgh 1.4e-12, U 3.9e-10; device 1.7e-12, 1.7e-12, 1.7e-10;
# k 7.7e-11, nut 3.2e-11, with OpenFOAM's own nut reaching 1.1e-05.
# BROKEN ONCE: the width frozen at start-up -- nut 1.1913e-02, k 9.1652e-04, U 8.8578e-04, alpha
# 1.0770e-07, 8 failures on both arms at once.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_moving_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
LAM="$TUT/multiphase/interFoam/laminar"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]                 || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$LAM/testTubeMixer" ]   || { echo "SKIP: testTubeMixer tutorial not found under $LAM"; exit 77; }
[ -f "$OFBASHRC" ]            || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

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
# THE DEVICE ARMS RUN OPENFOAM'S SUMMATION ORDER (BRAE_DEVICE_OF_REDUCE, reductions.cu): every Krylov dot and
# sum(|x|) as one sequential pass over the cells, as OpenFOAM's serial sumProd/sumMag are. The GPU's own tree
# order is the same sum in another last bit, and on a solve that arrives already converged that bit decides the
# stopping iteration: MEASURED on pistonOuterOnce, tree order 158 of 180 p_rgh counts OpenFOAM's and a 17-vs-18
# on a short solve (initial residual 1.07e-12 against tolerance 1e-13); OpenFOAM's order 177 of 180, all within
# the device rule, step one's initial residuals 8.2e-12 apart. It changes the device's reductions only -- the
# host arm is untouched -- and it is slow, so production runs keep the tree.
export BRAE_DEVICE_OF_REDUCE=1

# MOVING_ONLY="a b c": stage and gate only the named profiles (and the controls they name) -- for measuring
# one change without the whole queue. Unset, everything runs.
selected()
{
    [ -z "${MOVING_ONLY:-}" ] && return 0
    local x
    for x in $MOVING_ONLY; do [ "$x" = "$1" ] && return 0; done
    return 1
}

stage()
{
    local name="$1" tutorial="$2" dt="$3" n="$4" profile="$5"
    selected "$name" || return 0
    local C="$W/$name"
    # CLEAR IT FIRST: `cp -r src dst` copies INTO dst when dst exists, so a re-run under KEEP_W staged
    # the tutorial as a subdirectory and then edited the PREVIOUS run's already-staged files -- every
    # profile-specific assertion then failed on a case that was already correct.
    rm -rf "$C"
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

# ...and the CN profiles' own change: `ddtSchemes default CrankNicolson 0.9` on a MOVING mesh, where
# ddtCorr is a different operator from the static one -- fvcDdtUfCorr, built from Uf.oldTime() and an
# old-old level of it, rather than fvcDdtPhiCorr's phi.oldTime() (CrankNicolsonDdtScheme.C:1201-1257).
if profile.endswith('CN'):
    sc = os.path.join(d, 'system/fvSchemes')
    t = open(sc).read()
    t, k = re.subn(r'(ddtSchemes\s*\{\s*\n\s*default\s+)[^;]+;', r'\1CrankNicolson 0.9;', t)
    assert k == 1, 'no ddtSchemes default to make CrankNicolson'
    open(sc, 'w').write(t)
    # ...and ONE alpha sub-cycle, because OPENFOAM ITSELF refuses the combination: "Sub-cycling is not
    # supported with the CrankNicolson ddt scheme" (VoF/alphaEqn.H:30). The tutorial ships 3, so the
    # profile is the tank with CrankNicolson and no sub-cycling -- not the tutorial with one entry
    # changed -- and the gate's control is the EULER run of this same staging.
    q2 = os.path.join(d, 'system/fvSolution')
    u = open(q2).read()
    u, k = re.subn(r'^(\s*nAlphaSubCycles\s+)\d+;', r'\g<1>1;', u, flags=re.M)
    assert k == 1, 'no nAlphaSubCycles to set to 1'
    if profile == 'solitaryCN':
        # ...and CONVERGED pressure solves, as `piston` and `flap` have. This case amplifies: its own
        # one-ulp control reads U 2.0e-04 over thirty steps as shipped and 1.8e-06 converged, so the
        # stopping point is most of what a loose run would be measuring.
        u, k = re.subn(r'tolerance\s+1e-6;', 'tolerance       1e-13;', u)
        assert k >= 3, 'the pressure solvers were not tightened (%d)' % k
        # ...and `log 2;` on p_rgh, which makes OpenFOAM print its residual at EVERY iteration
        # (SolverPerformance.C:70-76) so the gate can compare the two CURVES rather than a count.
        for key in ('p_rgh', 'p_rghFinal'):
            m = re.search(r'\n    %s\s*\{[^}]*\}' % key, u)
            assert m, 'no %s entry to give `log 2`' % key
            u = u.replace(m.group(0), m.group(0).replace('}', '    log             2;\n    }'))
    open(q2, 'w').write(u)

q = os.path.join(d, 'system/fvSolution')
t = open(q).read()
if profile.startswith('esd'):
    # the tutorial's own alpha settings are what make the relaxation branch reachable at all
    assert 'MULESCorr       yes' in t or re.search(r'MULESCorr\s+yes', t), 'esd no longer sets MULESCorr yes'
    m = re.search(r'nAlphaCorr\s+(\d+)', t)
    assert m and int(m.group(1)) >= 2, 'esd no longer sets nAlphaCorr >= 2'
    # ...and CONVERGED PRESSURE SOLVES, as `solitaryCN`, `piston` and `flap` have. The tutorial ships
    # `p_rgh tolerance 5e-8 relTol 0.01`, and p_rgh's own maximum here is 0.809 while `p` reaches
    # 6.6e+03 -- the field is a near-total cancellation of the hydrostatic head, so its RELATIVE measure
    # is taken against a scale a thousand times smaller than the pressure it came from. MEASURED as
    # shipped: p_rgh 1.5144e-07 relative, all 12 iteration counts OpenFOAM's and the final residuals
    # agreeing to 5.8e-09 -- the same system stopped at the same place, so that number is the KRYLOV
    # STOPPING POINT and not the discretisation. `p` carries the SAME absolute difference (1.2e-07) and
    # reads 1.8539e-11 against its own scale. Pinning the solves is what measures the port instead of
    # the stopping point; the bounds stay the shared ones.
    for old_s, new_s in [(r'tolerance\s+5e-8;', 'tolerance       1e-13;')]:
        t, k = re.subn(old_s, new_s, t)
        assert k >= 1, 'esd: the p_rgh tolerance was not tightened'
    m = re.search(r'\n    p_rgh\s*\{[^}]*\}', t)
    assert m, 'esd: no p_rgh entry'
    t = t.replace(m.group(0), m.group(0).replace('relTol          0.01;', 'relTol          0;'))
    if profile == 'esdNoCorr':
        # THE CONTROL: MULESCorr off, so `alpha1 = 0.5*alpha1 + 0.5*alpha10` never runs and there is no
        # assigned patch value to keep. OpenFOAM's own written value goes back onto its owner cell.
        t, k = re.subn(r'MULESCorr\s+yes;', 'MULESCorr       no;', t)
        assert k == 1, 'the MULESCorr entry was not turned off'
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
    if profile in ('pistonOuter', 'pistonOuterOnce'):
        # TWO OUTER CORRECTORS on a DEFORMING mesh, the device loop's move guard and its V0: `pistonOuter`
        # moves the mesh at both (moveMeshOuterCorrectors yes), `pistonOuterOnce` at the first only
        t, k = re.subn(r'nCorrectors\s+3;', 'nCorrectors     3;\n    nOuterCorrectors 2;'
                       + ('\n    moveMeshOuterCorrectors yes;' if profile == 'pistonOuter' else ''), t)
        assert k == 1, 'the piston nCorrectors 3 was not found'
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
        if key == 'p_rgh':
            # `log 2;` makes OpenFOAM print its residual at EVERY iteration (SolverPerformance.C:70-76),
            # and p_rghFinal takes it through `$p_rgh;`. It changes no arithmetic; the gate reads the
            # history to compare the two residual CURVES where a count alone cannot -- see countsAgree.
            body = body.replace('}', '    log             2;\n    }')
        t = t.replace(m.group(0), body)
elif profile.startswith('floating'):
    # CONVERGED PRESSURE SOLVES, in BOTH codes. The case ships `pcorr` at tolerance 1e-5 and `p_rgh`
    # at relTol 0.01, and pcorr sets phi directly: at those settings the last iteration is where each
    # code stops, not what it computes. MEASURED, at the case's own settings: brae tracks OpenFOAM's
    # joint state to 2e-13 for six steps, then jumps four orders in ONE step -- at exactly the step
    # where the two codes' pcorr took a different iteration count (OpenFOAM 4, brae 5) -- and ends at
    # 8.8e-07. Converged to 1e-13 with relTol 0, the same comparison is 5.3e-13 at step one and
    # 2.5e-12 at step ten, growing smoothly with no jump. The case AMPLIFIES: ONE ULP added to every
    # mesh point's y, run against OpenFOAM itself, reaches U 3.3e-09 and q 2.0e-08 in these same ten
    # steps, so a bound near round-off would be measuring the tank's own chaos.
    # ...and pcorr to 1e-11 and NOT 1e-13. The FIRST pcorr of the run has an identically zero source
    # -- the mesh has not moved yet -- so its initial residual is a cancellation: OpenFOAM enters at
    # 8.0e-12 and brae below 1e-13. At a tolerance of 1e-13 that solve is BELOW it in one code and
    # above it in the other, and OpenFOAM iterates three times where brae iterates none; at 1e-11 both
    # enter converged and take none. MEASURED, end of run: at 1e-13 the body reads 2.5e-12 with 30 of
    # 31 pcorr counts equal, at 1e-11 it reads 8.3e-17 with 31 of 31.
    for old, new in [(r'tolerance\s+1e-0?5;', 'tolerance       1e-11;'),
                     (r'tolerance\s+1e-8;', 'tolerance       1e-13;'),
                     (r'relTol\s+0\.01;', 'relTol          0;'),
                     (r'maxIter\s+(20|100);', 'maxIter         500;')]:
        t, k = re.subn(old, new, t)
        assert k >= 1, 'floating: nothing matched %s' % old
    # `log 2;` makes OpenFOAM print its residual at EVERY iteration, so the gate can compare the two
    # CURVES where a count alone cannot
    m = re.search(r'\n    p_rgh\s*\{[^}]*\}', t)
    assert m, 'no p_rgh entry to give `log 2`'
    t = t.replace(m.group(0), m.group(0).replace('}', '    log             2;\n    }'))
elif profile.startswith('closed'):
    ref = '1e5' if profile == 'closedDamBreakRef' else '0'
    t, k = re.subn(r'(nNonOrthogonalCorrectors\s+0;)', r'\1\n    pRefPoint       (0.292 0.292 0.0073);\n    pRefValue       %s;' % ref, t)
    assert k == 1, 'PIMPLE block not found'
open(q, 'w').write(t)

if not profile.startswith('closed'):
    # the three tanks name vanLeerV, run as shipped; the `*Static` controls hold the mesh still
    f = os.path.join(d, 'system/fvSchemes')
    t = open(f).read()
    if profile.startswith('piston') or profile.startswith('flap') or profile.startswith('multi'):
        assert re.search(r'div\(phirb,alpha\)\s+Gauss interfaceCompression;', t), \
            'div(phirb,alpha) is no longer Gauss interfaceCompression'
    elif profile.startswith('esd'):
        # esd names its own, and two of them are what the profile is for: `Gauss vanLeer` on alpha (so
        # the limiter is live at the outflow patches) and a `cellLimited leastSquares` gradient, which
        # is the OPEN item that takes over this case from step five and the reason it is gated at TWO.
        assert re.search(r'div\(rhoPhi,U\)\s+Gauss upwind;', t), 'esd: div(rhoPhi,U) is no longer Gauss upwind'
        assert re.search(r'div\(phi,alpha\)\s+Gauss vanLeer;', t), 'esd: div(phi,alpha) is no longer Gauss vanLeer'
        assert re.search(r'default\s+cellLimited leastSquares 1;', t), 'esd: the gradient is no longer cellLimited leastSquares 1'
    elif not profile.startswith('solitary'):
        assert re.search(r'div\(rhoPhi,U\)\s+Gauss vanLeerV;', t), 'div(rhoPhi,U) is no longer Gauss vanLeerV'
    if profile.startswith('floating'):
        # THE CASE'S OWN ddt IS `CrankNicolson 0.5`, and a DEFORMING mesh under CrankNicolson is
        # refused on both arms -- localised, 1.06 off OpenFOAM, and held on its own (PORT.md). This
        # profile is the body, not the ddt scheme, so BOTH codes run the same case under Euler.
        assert re.search(r'default\s+CrankNicolson 0\.5;', t), \
            'floatingObject no longer ships CrankNicolson 0.5'
        t = re.sub(r'default\s+CrankNicolson 0\.5;', 'default         Euler;', t)
        # ...and WRITTEN BACK. Everything else this block does to fvSchemes is an assertion, so the
        # file was never reopened for writing and a substitution here would have been thrown away --
        # which it was, and OpenFOAM ran the tutorial's own CrankNicolson while the gate said Euler.
        open(f, 'w').write(t)
    p = os.path.join(d, 'constant/dynamicMeshDict')
    t = open(p).read()
    if profile.startswith('floating'):
        # `accelerationRelaxation` is a table that is ZERO until t = 4, and with aRelax 0 the
        # relaxation returns the PREVIOUS acceleration -- which starts at zero -- so over ten steps of
        # 5e-3 the body would not leave rest and the gate would be measuring a mesh that never moves.
        # The table's own final value, 0.7, applied from t = 0.
        assert re.search(r'accelerationRelaxation\s+table', t), \
            'floatingObject no longer relaxes the acceleration by a table'
        t, k = re.subn(r'accelerationRelaxation\s+table\s*\((?:[^()]|\([^()]*\))*\)\s*;',
                       'accelerationRelaxation 0.7;', t, flags=re.S)
        assert k == 1, 'accelerationRelaxation table not matched'
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
    ATM = [('U', 'type            fixedValue;\n        value           uniform (0 0 0);'),
           ('p_rgh', 'type            fixedFluxPressure;\n        value           uniform 0;'),
           ('alpha.water.orig', 'type            zeroGradient;'),
           ('alpha.water', 'type            zeroGradient;')]
    # ...or left OPEN to an outflow adjustPhi can SCALE: p_rgh zeroGradient still fixes no value, and U
    # zeroGradient (fixesValue false) or inletOutlet (fixesValue true, but isA<inletOutlet>) is the
    # adjustable kind (adjustPhi.C:59). alpha keeps the tutorial's inletOutlet.
    if profile == 'closedAdjZG':
        ATM = [('U', 'type            zeroGradient;'), ('p_rgh', 'type            zeroGradient;')]
    elif profile == 'closedAdjIO':
        ATM = [('U', 'type            inletOutlet;\n        inletValue      uniform (0 0 0);\n'
                     '        value           uniform (0 0 0);'),
               ('p_rgh', 'type            zeroGradient;')]
    for fld, body in ATM:
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

# THE TUBE WITH AN OPEN TOP: its `(3 7 6 2)` face taken out of `walls` into a patch of its own, with U
# inletOutlet, p_rgh zeroGradient -- so the tube still needs a reference -- and alpha zeroGradient. The one
# fixture here that makes pEqn.H:21-26's makeRelative/makeAbsolute around adjustPhi matter: the top's
# relative in- and outflow are 3.3e-06 each while the walls' MESH flux is 1.4e-03 each way, so balancing
# the absolute flux instead would change the top's by order one. alpha inletOutlet there puts a jump in
# rho across the inflow faces and OpenFOAM itself then leaves 1e-7 of continuity in the reference cell.
if profile == 'mixerTop':
    bm = os.path.join(d, 'system/blockMeshDict')
    b = open(bm).read()
    b, k = re.subn(r'\n\s*\(3 7 6 2\)', '', b)
    assert k == 1, 'mixerTop: the top face is not in walls'
    b, k = re.subn(r'(boundary\s*\(\s*walls\s*\{.*?\n    \}\n)',
                   r'\1    top\n    {\n        type patch;\n        faces ( (3 7 6 2) );\n    }\n', b, flags=re.S)
    assert k == 1, 'mixerTop: the boundary list was not found'
    open(bm, 'w').write(b)
    for fld, body in [('U', 'type            inletOutlet;\n        inletValue      uniform (0 0 0);\n'
                            '        value           uniform (0 0 0);'),
                      ('p_rgh', 'type            zeroGradient;'),
                      ('alpha.water', 'type            zeroGradient;')]:
        fp = os.path.join(d, '0', fld)
        u = open(fp).read()
        u, k = re.subn(r'(boundaryField\s*\{)', r'\1\n    top\n    {\n        %s\n    }' % body, u)
        assert k == 1, 'mixerTop: no boundaryField in ' + fld
        open(fp, 'w').write(u)

# THE PERMEABLE-WALL PAIR ON A MOVING MESH: the same tube with its `walls` given
# permeableAlphaPressureInletOutletVelocity and prghPermeableAlphaTotalPressure. Both rebuild their
# valueFraction from the flux AND the phase fraction on their own patch at every updateCoeffs, and
# dynamicMotionSolverFvMesh::update ends in U.correctBoundaryConditions()
# (dynamicMotionSolverFvMesh.C:101-114) -- where phi is the RELATIVE flux the previous step's pEqn
# left, not the ABSOLUTE one that step's own U evaluate read. brae read the absolute flux at both and
# held the extrapolated value on 346 of the 1050 faces where OpenFOAM holds exactly zero: U 1.33e-01
# from OpenFOAM after two steps, 8.66e-11 with it fixed. alphaMin 0.01 is the tutorials' own.
if profile == 'mixerPermeable':
    PERM = {
        '0/U': ('walls\n    {\n'
                '        type            permeableAlphaPressureInletOutletVelocity;\n'
                '        alpha           alpha.water;\n'
                '        alphaMin        0.01;\n'
                '        value           uniform (0 0 0);\n    }'),
        '0/p_rgh': ('walls\n    {\n'
                    '        type            prghPermeableAlphaTotalPressure;\n'
                    '        alpha           alpha.water;\n'
                    '        alphaMin        0.01;\n'
                    '        p               uniform 0;\n'
                    '        value           uniform 0;\n    }'),
    }
    for fname, entry in PERM.items():
        fp = os.path.join(d, fname)
        u = open(fp).read()
        u, k = re.subn(r'walls\s*\{[^}]*\}', lambda _m, e=entry: e, u, count=1)
        assert k == 1, 'no walls entry in ' + fname
        open(fp, 'w').write(u)
PYEOF
    if [ "$profile" = pistonLES ]; then
        # THE SAME PADDLE UNDER LES kEqn. What this profile holds that `pistonSST` cannot is the
        # FILTER WIDTH: cubeRootVolDelta is (deltaCoeff*V)^(1/3) per cell, LESModel::correct() calls
        # delta_().correct() first (LESModel.C:251) and cubeRootVolDelta::correct() recomputes it
        # whenever the mesh is changing (cubeRootVolDelta.C:128-134). Both brae arms took it once, at
        # start-up. Nothing is a wall function here -- kEqn's nut is Ck*sqrt(k)*delta everywhere.
        PROFILE="$profile" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging LES kEqn into $name"; return 1; }
import os, re, sys
d = sys.argv[1]
def sub(path, pat, rep):
    t = open(path).read()
    t2, k = re.subn(pat, rep, t, count=1)
    assert k == 1, (path, pat)
    open(path, 'w').write(t2)
sub(os.path.join(d, 'constant/turbulenceProperties'), r'simulationType\s+laminar;',
    'simulationType  LES;\n\nLES\n{\n    LESModel        kEqn;\n    turbulence      on;\n'
    '    printCoeffs     on;\n    delta           cubeRootVol;\n'
    '    cubeRootVolCoeffs { deltaCoeff 1; }\n}')
sc = os.path.join(d, 'system/fvSchemes')
sub(sc, r'(div\(rhoPhi,U\)[^\n]*\n)', r'\1    div(phi,k)      Gauss upwind;\n')
fs = os.path.join(d, 'system/fvSolution')
sub(fs, r'\n    U\n', '\n    "k.*"\n    {\n        solver          smoothSolver;\n'
    '        smoother        symGaussSeidel;\n        tolerance       1e-12;\n        relTol          0;\n'
    '    }\n\n    U\n')
hdr = ('FoamFile { version 2.0; format ascii; class volScalarField; object %s; }\n'
       'dimensions      %s;\ninternalField   uniform %s;\nboundaryField\n{\n'
       '    "(.*)"       { type zeroGradient; }\n}\n')
for fld, dims, val in [('k', '[0 2 -2 0 0 0 0]', '0.0001'), ('nut', '[0 2 -1 0 0 0 0]', '0')]:
    for t in ('0', '0.orig'):
        p = os.path.join(d, t)
        if os.path.isdir(p):
            open(os.path.join(p, fld), 'w').write(hdr % (fld, dims, val))
PYEOF
    fi
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
    # THE ORACLE IS CACHED FROM HERE. Everything above is the profile's staging -- copies and text
    # edits, cheap -- and everything below is real OpenFOAM, which is this gate's whole wall clock.
    # The key is a hash of every staged byte, so a profile whose staging changes by one character
    # misses. See tests/of_oracle_cache.sh for why a hit is verified rather than trusted.
    local end key
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    key=$(oracleKey "$C" "interfoam_moving" "$name" "$tutorial" "$dt" "$n" "$profile")
    if oracleRestore "$C" "$key" "$end"; then
        echo "OpenFOAM's $n steps of deltaT $dt to t = $end reused from the oracle cache   [$name]"
        return 0
    fi
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    if [ -f "$C/system/snappyHexMeshDict" ]; then
        mkdir -p "$C/constant/triSurface"
        # ...unless the TUTORIAL SHIPS ITS OWN surface, as RAS/electrostaticDeposition does
        # (constant/triSurface/metalSheet.stl.gz). The resources copy would fail on it, and copying a
        # differently-named .obj.gz over it would mesh a different geometry.
        if [ -z "$(ls -A "$C/constant/triSurface" 2>/dev/null)" ]; then
            cp -f "$TUT/resources/geometry/$tutorial.obj.gz" "$C/constant/triSurface/" || { echo "FAIL: no surface for $tutorial"; return 1; }
        fi
        ( cd "$C" && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 ) || { echo "FAIL: snappyHexMesh [$name]"; tail -20 "$C/log.snappyHexMesh"; return 1; }
    fi
    # floatingObject's mesh is blockMesh MINUS the body: topoSet selects the cells outside it and
    # `subsetMesh -patch floatingObject` turns the cut faces into the body's own patch. Without this
    # the case has no floatingObject patch at all and nothing for the body's force to be read from.
    case "$profile" in floating*)
        ( cd "$C" && topoSet > log.topoSet 2>&1 ) || { echo "FAIL: topoSet [$name]"; tail -20 "$C/log.topoSet"; return 1; }
        ( cd "$C" && subsetMesh -overwrite c0 -patch floatingObject > log.subsetMesh 2>&1 ) \
            || { echo "FAIL: subsetMesh [$name]"; tail -20 "$C/log.subsetMesh"; return 1; }
        grep -q "floatingObject" "$C/constant/polyMesh/boundary" || { echo "FAIL: subsetMesh made no floatingObject patch [$name]"; return 1; }
        ;;
    esac
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$name]"; tail -20 "$C/log.setFields"; return 1; }
    # esd's Allrun.pre rotates the meshed case AFTER setFields, so the water column is filled in the
    # unrotated frame and gravity then acts along the rotated axis. Doing it before setFields, or not at
    # all, is a different case.
    case "$profile" in esd*)
        ( cd "$C" && transformPoints -rotate-y -90 > log.transformPoints 2>&1 ) \
            || { echo "FAIL: transformPoints [$name]"; tail -20 "$C/log.transformPoints"; return 1; }
        ;;
    esac
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$name]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$end" ] || { echo "FAIL: OpenFOAM wrote no $end directory [$name]"; ls "$C"; return 1; }
    oracleStore "$C" "$key"
    echo "OpenFOAM ran $n steps of deltaT $dt to t = $end   [$name]"
}

# gate <name> <deltaT> <nSteps> <profile> <control case>
#
# THE 24 OF THESE ARE THE GATE'S CLOCK. With the oracle cached, OpenFOAM's staging is 32% of this
# gate's wall time and brae's own two arms are 68% -- 24 runs of the test binary, each running a HOST
# arm and a DEVICE arm on its own staged case, one after another. They are independent: each reads its
# own case and its own control's written fields, which `stage` finished before any of them started.
# So they are queued here and run through a bounded pool (MOVING_JOBS, default 4).
#
# WHY A POOL AND NOT `&` FOR ALL 24: one GPU. The device arms serialise on it whatever the shell does,
# and 24 processes would also hold 24 cases' fields in memory at once. Four keeps the host arms (which
# are CPU-bound and are most of the time) overlapped while the device arms queue.
#
# THE OUTPUT IS NOT INTERLEAVED: each job writes to its own log and the logs are printed in the order
# the gate declares them, so a failure still reads exactly as it did when this was a serial loop.
# MOVING_JOBS=1 restores that loop, which is what to set if a failure needs to be watched live.
JOBS=${MOVING_JOBS:-4}
QUEUE=()
gate()
{
    selected "$1" || return 0
    QUEUE+=("$1|$2|$3|$4|$5")
}

runOne()
{
    local name="$1" dt="$2" n="$3" profile="$4" control="$5" out="$6"
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    "$BIN" "$W/$name" "$W/$name/0" "$W/$name/$end" "$n" "$W/$name/log.interFoam" "$profile" \
           "$W/$control/$end" > "$out" 2>&1
}

runQueue()
{
    local running=0 outs=() names=()
    for spec in "${QUEUE[@]}"; do
        IFS='|' read -r name dt n profile control <<< "$spec"
        local out="$W/.gate.$name.log"
        outs+=("$out")
        names+=("$name")
        # the STATUS goes to a file beside the log: `wait` cannot be asked twice about one pid, and a
        # status read from the wrong place is how a parallel gate reports green while an arm failed
        ( runOne "$name" "$dt" "$n" "$profile" "$control" "$out"; echo $? > "$out.rc" ) &
        running=$((running + 1))
        if [ "$running" -ge "$JOBS" ]; then
            wait -n 2>/dev/null || wait
            running=$((running - 1))
        fi
    done
    wait
    local k=0
    for name in "${names[@]}"; do
        cat "${outs[$k]}"
        local st
        st=$(cat "${outs[$k]}.rc" 2>/dev/null || echo 1)
        if [ "$st" != 0 ]; then
            echo "  FAIL: the $name arm exited $st"
            rc=1
        fi
        k=$((k + 1))
    done
}

rc=0
stage mixerStatic    testTubeMixer 2e-4  10 mixerStatic    || rc=1
stage mixer          testTubeMixer 2e-4  10 mixer          || rc=1
stage mixerTop       testTubeMixer 2e-4  10 mixerTop       || rc=1
stage mixerCorr      testTubeMixer 2e-4  10 mixerCorr      || rc=1
stage mixerOuter     testTubeMixer 2e-4  10 mixerOuter     || rc=1
stage mixerOuterOnce testTubeMixer 2e-4  10 mixerOuterOnce || rc=1
stage mixerPred      testTubeMixer 2e-4  10 mixerPred      || rc=1
stage mixerPermeable testTubeMixer 2e-4  10 mixerPermeable || rc=1
stage sloshing2DStatic sloshingTank2D 0.01  10 sloshing2DStatic || rc=1
stage sloshing2D     sloshingTank2D 0.01  10 sloshing2D     || rc=1
# ...and the same tank under CRANKNICOLSON, which is THREE branches of the scheme a static case never
# reaches, and this arm holds all three:
#   * fvm::ddt's moving branch -- ddt0 weighted by V0 and V00, the source by V0 (:1029-1065);
#   * ddtCorr is fvcDdtUfCorr (:1201-1257) rather than fvcDdtPhiCorr: built from Uf.oldTime() and an
#     old-old level of Uf, with its own ddt0 surface field;
#   * fvc::meshPhi is the SCHEME's (:1626-1661), so the mesh flux every makeRelative, movingWallVelocity
#     and divU reads is off-centred against the previous move's.
# MEASURED with the third missing and the first two in place -- the state this profile was written in:
# p_rgh 4.4802e-01, alpha 1.1834e-03, U 4.7460e-03, the moving walls 5.0988e-02 out, and 3 of 20 p_rgh
# iteration counts wrong, against 6.1e-14 / 5.6e-15 / 2.7e-13 / 2.7e-13 and 20 of 20 with it. Step ONE
# agrees either way (2.7e-14): the off-centring is born at the second step, which is why a one-step
# comparison cannot hold this scheme.
# Its control is the EULER run of the same case: a static control would be blind to all three.
stage sloshing2DCN   sloshingTank2D 0.01  10 sloshing2DCN   || rc=1
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
# ...and the SAME SCHEME ON A MESH THAT DEFORMS, which is a different question from the tank above and
# found a different defect.
#
# WHY IT EXISTS. sloshing2DCN cannot witness the scheme's moving ddt at all: the tank's motion is SOLID
# BODY (`solidBodyMotionFunction SDA`), so V == V0 == V00 in every cell and ddt0's volume weights are
# arithmetically the static form's. MEASURED -- with V0 and V00 dropped from the device's fvm::ddt,
# sloshing2DCN's device arm still reads U 6.3e-13 (against 3.3e-13 with them): blind. waveMakerSolitary
# DEFORMS (displacementLaplacian, inverseDistance), and it found phi.oldTime()'s LAZY CREATION: on a
# moving mesh ddtCorr is fvcDdtUfCorr and reads Uf.oldTime(), so NOTHING asks for phi.oldTime() until
# alphaEqn's own off-centred blend does -- and the level is then born a copy of the flux beside it,
# leaving the blend inert for that one step. brae blended with the previous step's flux instead:
# phiCN 1.17 RELATIVE off at step two (OpenFOAM's phiCN IS its phi there, to 2.1e-22), alpha 4.7e-05,
# U 1.2e-03 at step two and 1.06e+00 at thirty. Invisible under Euler (ocAlpha is 0) and invisible
# without correctPhi (the flux the blend sees IS last step's there) -- which is why this profile, and
# not the tank, is what holds it. With it modelled: two steps U 1.5e-11, alpha 2.5e-14.
#
# THE CASE AMPLIFIES, so the profile carries its own bounds and its own control. OpenFOAM against
# ITSELF, one ulp of one alpha cell, thirty steps, converged solves: U 1.752e-06, alpha 7.764e-09,
# p_rgh 7.933e-09. brae reads 4.077e-06, 3.960e-08, 2.929e-08 -- a small multiple of OpenFOAM's own
# last bit. The solves are converged here for the same reason `piston` and `flap` are.
stage solitaryCN     waves/waveMakerSolitary 0.01 30 solitaryCN     || rc=1
stage solitaryGamg   waves/waveMakerSolitary 0.01 30 solitaryGamg   || rc=1
stage pistonStatic   waves/waveMakerPiston   0.01 30 pistonStatic   || rc=1
stage piston         waves/waveMakerPiston   0.01 30 piston         || rc=1
stage pistonSST      waves/waveMakerPiston   0.01 30 pistonSST      || rc=1
stage pistonLES      waves/waveMakerPiston   0.01 30 pistonLES      || rc=1
stage pistonOuter    waves/waveMakerPiston   0.01 30 pistonOuter    || rc=1
stage pistonOuterOnce waves/waveMakerPiston  0.01 30 pistonOuterOnce || rc=1
stage flapStatic     waves/waveMakerFlap     0.01 30 flapStatic     || rc=1
stage flap           waves/waveMakerFlap     0.01 30 flap           || rc=1
stage multiPistonStatic waves/waveMakerMultiPaddlePiston 0.01 30 multiPistonStatic || rc=1
stage multiPiston       waves/waveMakerMultiPaddlePiston 0.01 30 multiPiston       || rc=1
stage multiFlapStatic   waves/waveMakerMultiPaddleFlap   0.01 30 multiFlapStatic   || rc=1
stage multiFlap         waves/waveMakerMultiPaddleFlap   0.01 30 multiFlap         || rc=1
stage floatingStatic ../RAS/floatingObject 5e-3 10 floatingStatic || rc=1
stage floating       ../RAS/floatingObject 5e-3 10 floating       || rc=1
stage closedRef1e5   damBreak/damBreak 0.001 20 closedDamBreakRef || rc=1
stage closedDamBreak damBreak/damBreak 0.001 20 closedDamBreak    || rc=1
stage closedDamBreakInitU damBreak/damBreak 0.001 20 closedDamBreakInitU || rc=1
stage closedAdjZG     damBreak/damBreak 0.001 20 closedAdjZG     || rc=1
stage closedAdjIO     damBreak/damBreak 0.001 20 closedAdjIO     || rc=1
# RAS/electrostaticDeposition, TWO STEPS, for one thing no other arm here can see: alpha's PATCH values
# under the under-relaxed corrector. It belongs in this harness because its mesh moves -- `solidBody`
# `tabulated6DoFMotion`, a RIGID translation at -0.08 m/s, so V == V0 and the volume weights are blind --
# and it is the only interFoam tutorial whose alpha carries a `variableHeightFlowRate` AND reaches the
# relaxation branch (`MULESCorr yes`, `nAlphaCorr 2`). DTCHull and DTCHullMoving carry it too and neither
# runs: four features away (localEuler/LTS, outletPhaseMeanVelocity, `linearUpwind limitedGrad` in the
# closure, nutkRoughWallFunction). weirOverflow carries it with `nAlphaCorr 1`, so its own gate's header
# already records that the condition has no control there.
# TWO STEPS AND NOT TEN: this case also carries the OPEN `cellLimited` grad(alpha) item, whose cell gap
# passes the patch difference from step five. The test binary's header carries the per-step table.
stage esdNoCorr ../RAS/electrostaticDeposition 1e-3 2 esdNoCorr || rc=1
stage esd       ../RAS/electrostaticDeposition 1e-3 2 esd       || rc=1
[ $rc = 0 ] || { echo "interfoam_moving_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path (each check only where its profiles were staged: all of them unless MOVING_ONLY)
if selected mixer && selected mixerStatic; then
grep -q "Constructed SBMF 1 : rotatingBox of type oscillatingRotatingMotion" "$W/mixer/log.interFoam" \
    || { echo "FAIL: OpenFOAM's mixer log does not build the multiMotion"; exit 1; }
grep -q "Selecting dynamicFvMesh staticFvMesh" "$W/mixerStatic/log.interFoam" \
    || { echo "FAIL: OpenFOAM's control did not hold the mesh still"; exit 1; }
grep -q "^Courant Number mean: 0.0[1-9]" "$W/mixer/log.interFoam" \
    || { echo "FAIL: OpenFOAM's mixer never moved its fluid"; exit 1; }
fi
if selected floating && selected floatingStatic; then
grep -q "Selecting motion solver: rigidBodyMotion" "$W/floating/log.interFoam" \
    || { echo "FAIL: OpenFOAM's floatingObject did not select rigidBodyMotion"; exit 1; }
grep -q "Selecting dynamicFvMesh staticFvMesh" "$W/floatingStatic/log.interFoam" \
    || { echo "FAIL: OpenFOAM's floating control did not hold the mesh still"; exit 1; }
grep -q "default         Euler;" "$W/floating/system/fvSchemes" \
    || { echo "FAIL: the floating staging did not put the case on Euler"; exit 1; }
# ...AND THE BODY MOVED. `accelerationRelaxation` is zero until t = 4, so a staging that failed to
# replace the table leaves a body that never leaves rest -- and every field of that run is the static
# control's, so the gate would pass while measuring nothing. MEASURED, when the fvSchemes staging was
# silently discarded and the mesh was built without `subsetMesh`: thirty reports of
# `Linear velocity: (0 0 0)` and a written `q 2 { 0 }`.
grep -q "{ 0 }" "$W/floating/0.05/uniform/rigidBodyMotionState" \
    && { echo "FAIL: OpenFOAM's body never moved -- the gate would be vacuous"; exit 1; }
grep -q "floatingObject" "$W/floating/constant/polyMesh/boundary" \
    || { echo "FAIL: the staged mesh has no floatingObject patch for the force to act on"; exit 1; }
fi

for c in multiPiston multiFlap; do
    selected "$c" || continue
    grep -q "^GAMG:  Solving for p_rgh" "$W/$c/log.interFoam" \
        || { echo "FAIL: OpenFOAM's $c log does not solve p_rgh with GAMG, the solver \$pcorr brings in"; exit 1; }
done

gate mixer          2e-4  10 mixer          mixerStatic  || rc=1
gate mixerCorr      2e-4  10 mixerCorr      mixerStatic  || rc=1
gate mixerOuter     2e-4  10 mixerOuter     mixerStatic  || rc=1
gate mixerOuterOnce 2e-4  10 mixerOuterOnce mixerStatic  || rc=1
gate mixerPred      2e-4  10 mixerPred      mixerStatic  || rc=1
gate mixerPermeable 2e-4  10 mixerPermeable mixer        || rc=1
gate sloshing2D     0.01  10 sloshing2D     sloshing2DStatic || rc=1
gate sloshing2DCN   0.01  10 sloshing2DCN   sloshing2D       || rc=1
gate sloshing2D3DoF 0.01  10 sloshing2D3DoF sloshing2D3DoFStatic || rc=1
gate sloshing3D     0.01  10 sloshing3D     sloshing3DStatic     || rc=1
gate sloshing3D3DoF 0.01  10 sloshing3D3DoF sloshing3D3DoFStatic || rc=1
gate sloshing3D6DoF 0.01  10 sloshing3D6DoF sloshing3D6DoFStatic || rc=1
gate cylinder       0.001 10 cylinder       cylinderStatic   || rc=1
gate mixerCorrectPhi      2e-4  10 mixerCorrectPhi      mixerStatic      || rc=1
gate sloshing2DCorrectPhi 0.01  10 sloshing2DCorrectPhi sloshing2DStatic || rc=1
gate cylinderCorrectPhi   0.001 10 cylinderCorrectPhi   cylinderStatic   || rc=1
gate solitary       0.01  30 solitary       solitaryStatic || rc=1
gate solitaryCN     0.01  30 solitaryCN     solitary       || rc=1
# the deforming mesh with a GAMG pressure solve, on BOTH arms -- see the solitaryGamg staging
gate solitaryGamg   0.01  30 solitaryGamg   solitaryStatic || rc=1
gate piston         0.01  30 piston         pistonStatic   || rc=1
gate pistonSST      0.01  30 pistonSST      piston         || rc=1
# ...and the same paddle under LES kEqn, whose FILTER WIDTH moves with the cells. Control: the
# laminar piston, as pistonSST's is.
gate pistonLES      0.01  30 pistonLES      piston         || rc=1
gate pistonOuter    0.01  30 pistonOuter    piston         || rc=1
gate pistonOuterOnce 0.01 30 pistonOuterOnce piston        || rc=1
gate flap           0.01  30 flap           flapStatic     || rc=1
gate multiPiston    0.01  30 multiPiston    multiPistonStatic || rc=1
gate multiFlap      0.01  30 multiFlap      multiFlapStatic   || rc=1
# RAS/floatingObject: a BODY the fluid moves, not a prescribed motion -- rigidBodyMeshMotion driving
# the mesh from the load on its own patches, with the joint state as the second oracle. Control: the
# same tank with the mesh held still.
gate floating       5e-3  10 floating       floatingStatic || rc=1
gate closedDamBreak 0.001 20 closedDamBreak closedRef1e5 || rc=1
gate closedDamBreakInitU 0.001 20 closedDamBreakInitU closedDamBreak || rc=1
# adjustPhi's SCALING half, on both arms: the dam's atmosphere left open to an adjustable outflow under
# a pressure reference. zeroGradient U is adjustable by fixesValue, inletOutlet by isA -- the two halves
# of adjustPhi.C:59 -- so each is the other's control, and the walled tank is the first one's.
gate closedAdjZG    0.001 20 closedAdjZG    closedDamBreak || rc=1
gate closedAdjIO    0.001 20 closedAdjIO    closedAdjZG    || rc=1
# ...and around it, on a MOVING mesh, the relative-flux round trip; its control is the shipped tube
gate mixerTop       2e-4  10 mixerTop       mixer          || rc=1
# ...and its own control is the MULESCorr-off staging, where OpenFOAM's patch value is back on its cell
gate esdNoCorr 1e-3 2 esdNoCorr esd       || rc=1
gate esd       1e-3 2 esd       esdNoCorr || rc=1

runQueue

# THE DEVICE MESH UPDATE'S TWO CONTROLS, each asserted to fail on a device number. The device loop's move block
# used to run at every outer corrector and re-take V0 from the volumes before each update; with two outer
# correctors both are live, and a gate on one corrector cannot see either.
#   BRAE_CONTROL_DEVICE_MOVE_EVERY_OUTER on pistonOuterOnce: the block at the second corrector as well, whose
#     correctPhi copies the host phi the host stage never rewrote there. MEASURED U 3.2e-06 (fixed 2.7e-08).
#   BRAE_CONTROL_DEVICE_V0_FROM_V on pistonOuter: V0 from the volumes as they stand before each update, i.e.
#     the first update's moved volumes at the second. MEASURED U 8.1e-04 (fixed 1.1e-10).
deviceControl()
{
    local name="$1" ctl="$2"
    selected "$name" || return 0
    local out="$W/.control.$name.log"
    env "$ctl" "$BIN" "$W/$name" "$W/$name/0" "$W/$name/0.3" 30 "$W/$name/log.interFoam" "$name" \
        "$W/piston/0.3" > "$out" 2>&1
    local crc=$?
    local pat="FAIL: (the device's alpha|\.\.\.its p_rgh|\.\.\.and its U|\.\.\.and the device's Uf)"
    if [ $crc -ne 0 ] && grep -q "CONTROL MODE" "$out" && grep -qE "$pat" "$out"; then
        echo "  ok:   device control $ctl on $name fails on a number: $(grep -E 'DEVICE:  alpha' "$out" | tr -s ' ')"
    else
        echo "  FAIL: device control $ctl on $name did not fail -- the gate cannot see the defect it guards"
        tail -15 "$out"
        rc=1
    fi
}
deviceControl pistonOuterOnce BRAE_CONTROL_DEVICE_MOVE_EVERY_OUTER=1
deviceControl pistonOuter     BRAE_CONTROL_DEVICE_V0_FROM_V=1

echo "interfoam_moving_vs_openfoam: rc $rc"
exit $rc
