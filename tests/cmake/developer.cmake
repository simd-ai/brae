# Developer executables. Probes, comparison drivers, frozen solver snapshots and helper binaries that no ctest command
# names directly (gate scripts call some of them by path). Built with the tests; not installed.
#
# Included by tests/CMakeLists.txt.

add_executable(test_liqref_inversion ${PROJECT_SOURCE_DIR}/tests/test_liqref_inversion.cu)
target_include_directories(test_liqref_inversion PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_liqref_inversion PRIVATE brae_core)

add_executable(test_leastsquares_grad ${PROJECT_SOURCE_DIR}/tests/test_leastsquares_grad.cu)
target_include_directories(test_leastsquares_grad PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_leastsquares_grad PRIVATE brae_core)

# CUDA rhoUEqn against the _cpp reference at MATRIX granularity -- diag, upper, lower, the three
# sources and every boundary coefficient, not a folded single number. The reference is itself gated
# against OpenFOAM's own dump by tests/rho_ueqn_vs_openfoam.sh, so this closes OpenFOAM -> _cpp -> CUDA.
#
# The fixtures are the incompressible ones because no committed case ships a rho field -- the compressible
# _cpp gates build their state by running real OpenFOAM. The gate SYNTHESIZES a non-uniform rho and
# asserts it varies, which is the point: with a uniform rho, mu_eff is a constant multiple of nu_eff and a
# kernel that dropped the rho weighting from one term would still agree on a rescaled matrix.
# CUDA rhoPEqn against the _cpp reference, stage by stage and BOTH BRANCHES. simple.transonic() selects
# two different equations -- the transonic one adds fvm::div(phid,p) and calls pEqn.relax(), the subsonic
# one runs adjustPhi and neither -- so both are registered. Testing one would leave the other asserted by
# nothing. rho and psi are synthesized non-uniform for the reason the gate's header gives.
# CUDA rhoEEqn against the _cpp reference, stage by stage and BOTH ENERGY VARIABLES. EEqn.H builds the
# kinetic-energy source as Ekp = 0.5|U|^2 + p/rho when he is `e` and as K = 0.5|U|^2 when he is `h` --
# different conservation laws, so both are registered. he, rho and alphaEff are synthesized non-uniform;
# alphaEff varies on the boundary INDEPENDENTLY of the cells so the patch-value rule is exercised.
# CUDA rhoPcEqn (SIMPLEC) against the _cpp reference, BOTH BRANCHES. It needs its own gate rather than
# riding on rho_peqn_cuda: rAtU = 1/(1/rAU - H1) is compared separately from rAU with a control asserting
# they differ, because if H1 came back zero every downstream number would still agree and the equation
# would be SIMPLE wearing SIMPLEC's name. The flux correction and the HbyA correction are asserted to move
# their fields independently -- they are one change, and keeping one without the other is worse than
# neither.
# The CUDA kEpsilon closure against the _cpp reference, stage by stage. The closure folds a production
# term, a wall treatment, two transport operators, a relaxation and two solves into k and epsilon, so one
# number over the solved fields cannot say which of them moved -- eps0/G0/isWallCell are compared on their
# own for exactly that reason. Three controls carry it: the MASS flux must not serve as divU (they differ
# only because rho varies here, which is why rho is synthesized varying rather than uniform), nut's
# boundary must be the PATCH value and not the owner cell's, and the production term must be load-bearing.
# A fourth runs correct() TWICE in one process: DeviceBuffer::resize takes from a pool that does not zero,
# and the first call is always right because the pool has nothing to hand back yet.
# The CUDA rhoSimpleFoam DRIVER against the _cpp driver, over MULTIPLE ITERATIONS -- which is the whole
# point. Every component it composes already has a per-stage gate at 1e-12 or better; what those cannot
# see is what the driver feeds them on the SECOND pass. The incompressible twin's AMG-cache defect was
# EXACT at iteration 1 and 1.3e-01 wrong at iteration 2 with every per-stage gate green, so a
# one-iteration driver test would have certified it. Laminar on purpose: the closure has its own device
# gate, and running it here would fold two more Krylov solves into a comparison about ORDER.
# The device projection of createFields against the host field set it projects. An upload is a copy, so
# the field checks assert EXACT equality -- a tolerance on a memcpy admits a defect rather than measuring
# one. What is actually at risk is the boundary-face ORDER, the padding of the faces DeviceMesh skips,
# which of the two masks answers which question, and whether the wall set is chosen by the BC or by the
# patch type; each has its own check, and the two a fixture can make vacuous say so instead of passing.
# STRUCTURAL measurements on OpenFOAM's angledDuctExplicitFixedCoeff tutorial -- the only case reachable
# from here that can measure two things this port had correct-by-construction and gated by nothing: the
# two boundary masks (it gives porosityWall `type slip` on U, and slip is the only patch class where
# `assignable` and `fixesValue && !inletOutlet` disagree), and the relax -> constrain -> setValues order
# (its fvOptions constrains k and epsilon on the porosity cellZone, whose own wall carries an
# epsilonWallFunction, so the two constraints overlap on every cell of that patch).
#
# NO CONVERGENCE BOUND. The end-to-end gate on this same tutorial is deliberately unregistered because a
# bound would have to accommodate an unexplained 9.43e-04; everything here is structural or a single
# assembly, so none of it inherits that question.
# EMPTY PATCHES MUST BE INERT. In OpenFOAM they cannot contribute to a surface sum because
# emptyFvPatchField is zero-sized; brae keeps the faces, so every summing operator must skip them
# explicitly. This asserts the PROPERTY -- poison an empty patch's values and require bit-identical
# output -- rather than a number, because a number would be satisfied by a mesh that happens to cancel.
# Measured before the fix: a cell's empty-face area vectors sum to 1.6e-16 of their magnitude on rhoBox
# and 2.4e-14 on pitzDaily, so the old behaviour was round-off that depended on the mesh being an exact
# extrusion. The control poisons a NON-empty patch and requires the answer to move.
add_executable(test_empty_patch_inert ${PROJECT_SOURCE_DIR}/tests/test_empty_patch_inert.cu)
target_include_directories(test_empty_patch_inert PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_empty_patch_inert PRIVATE brae_core)
# DELIBERATELY NOT REGISTERED -- and the reason is measured, not assumed. Making empty patches inert is
# correct against OpenFOAM, and the operator-level change it causes is 7.886e-16 relative on pitzDailyFoam
# (the empty contribution to grad(p) is 4.562325e-12 against max|grad(p)| = 5.785680e+03). But a 1e-16
# perturbation is not nothing after 2000 SIMPLE iterations on a separating flow: applying it moved
# tutorial_backwardfacingstep2d's converged U from 3.482e-03 to 1.145e-02 against its 1e-02 bound --
# deterministic and reproducible to every printed digit, in a recirculation whose reattachment point is
# exactly the sensitive quantity. Registering this would therefore mean either shipping a red gate or
# loosening backwardFacingStep2D's bound to accommodate a change, which is the failure mode this port
# exists to avoid. The binary is built so the requirement stays executable and the measurement stays
# reproducible; enforcing it needs a convergence-robust comparison on that tutorial first.
#
# THAT COMPARISON NOW EXISTS (2026-09-03): backwardFacingStep2D is gated on a WINDOW MEAN, because the
# case limit-cycles and a single iterate was never a comparable quantity -- OpenFOAM moves U by 1.66e-02
# between two of its own iterates. A perturbation that shifts the phase moves the mean by ~5e-04, inside
# OpenFOAM's own noise floor. So this blocker is lifted in principle; registering the test still needs its
# own measurement through the windowed gate, which nothing here has done.

add_executable(kepsilon_probe ${PROJECT_SOURCE_DIR}/tests/kepsilon_probe.cu)
target_include_directories(kepsilon_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(kepsilon_probe PRIVATE brae_core)

add_executable(resid_probe ${PROJECT_SOURCE_DIR}/tests/resid_probe.cu)
target_include_directories(resid_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(resid_probe PRIVATE brae_core)

# END-TO-END gate for the rebuilt simpleFoam: one SIMPLE iteration composed of the _cpp components,
# against OpenFOAM's own dumpSimpleStep output. Proves the composition, not just the parts.
# END-TO-END: the CUDA driver against the _cpp driver, one iteration.
# Diagnostic (not a ctest): both drivers, N iterations, side-by-side residuals + divergence trace.
add_executable(diag_simple_loop ${PROJECT_SOURCE_DIR}/tests/diag_simple_loop.cu)
target_include_directories(diag_simple_loop PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(diag_simple_loop PRIVATE brae_core)

# Isolated side-by-side experiment: the target is opt-in and never replaces the production smoother.
add_executable(factory_sym_gauss_seidel_bench EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/factory/benchmark_sym_gauss_seidel.cu)
target_include_directories(factory_sym_gauss_seidel_bench PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(factory_sym_gauss_seidel_bench PRIVATE brae_core)

add_executable(test_simple_sst_cpp ${PROJECT_SOURCE_DIR}/tests/test_simple_sst_cpp.cu)
target_include_directories(test_simple_sst_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_sst_cpp PRIVATE brae_core)

add_executable(test_simple_lm_cpp ${PROJECT_SOURCE_DIR}/tests/test_simple_lm_cpp.cu)
target_include_directories(test_simple_lm_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_lm_cpp PRIVATE brae_core)

add_executable(lm_probe ${PROJECT_SOURCE_DIR}/tests/lm_probe.cu)
target_include_directories(lm_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(lm_probe PRIVATE brae_core)

add_executable(ami_probe ${PROJECT_SOURCE_DIR}/tests/ami_probe.cu)
target_include_directories(ami_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(ami_probe PRIVATE brae_core)

add_executable(ke_localize_probe ${PROJECT_SOURCE_DIR}/tests/ke_localize_probe.cu)
target_include_directories(ke_localize_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(ke_localize_probe PRIVATE brae_core)

add_executable(v2_envelope ${PROJECT_SOURCE_DIR}/tests/v2_envelope.cu)
target_include_directories(v2_envelope PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(v2_envelope PRIVATE brae_core)

add_executable(gradu_probe ${PROJECT_SOURCE_DIR}/tests/gradu_probe.cu)
target_include_directories(gradu_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(gradu_probe PRIVATE brae_core)

add_executable(ueqn_localize ${PROJECT_SOURCE_DIR}/tests/ueqn_localize.cu)
target_include_directories(ueqn_localize PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(ueqn_localize PRIVATE brae_core)

add_executable(rotordisk_probe ${PROJECT_SOURCE_DIR}/tests/rotordisk_probe.cu)
target_include_directories(rotordisk_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(rotordisk_probe PRIVATE brae_core)

add_executable(test_simple_mrf_cpp ${PROJECT_SOURCE_DIR}/tests/test_simple_mrf_cpp.cu)
target_include_directories(test_simple_mrf_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_mrf_cpp PRIVATE brae_core)

add_executable(test_simple_sa_cpp ${PROJECT_SOURCE_DIR}/tests/test_simple_sa_cpp.cu)
target_include_directories(test_simple_sa_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_sa_cpp PRIVATE brae_core)

# brae_rhoSimpleFoam_slice: the vertical-slice port (src/applications/solvers/rhoSimpleFoam/slice).
#
# SLICE 0 is a byte-faithful COPY of the shared driver, so this target exists to prove the seam -- that the
# slice folder builds on its own, links the Tier-A infrastructure it shares rather than re-porting it, and
# reproduces brae_rhoSimpleFoam's output exactly on a real tutorial. See slice/README.md for the tier
# split and why the pimpleFoam port's defects motivate it.
add_executable(brae_rhoSimpleFoam_slice
               ${PROJECT_SOURCE_DIR}/src/applications/solvers/rhoSimpleFoam/slice/solver/rho_simple_foam_slice.cu)
target_include_directories(brae_rhoSimpleFoam_slice PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/src/applications/solvers/rhoSimpleFoam
                           ${PROJECT_SOURCE_DIR}/src/applications/solvers/rhoSimpleFoam/slice/solver)
target_link_libraries(brae_rhoSimpleFoam_slice PRIVATE brae_core)

# brae_rhoSimpleFoam_legacy: frozen snapshot of the GPU-first solver, kept runnable so the CPU-first
# ground-truth rewrite always has a reference to diff against. See rhosimplefoam-ground-truth-port.md.
# brae_rhoSimpleFoam keeps pointing at the current implementation until the new path passes every gate
# below that this one passes; only then does the name move.
add_executable(brae_rhoSimpleFoam_legacy
               ${PROJECT_SOURCE_DIR}/src/applications/solvers/rhoSimpleFoam/legacy/gpuRhoSimpleFoamLegacy.cu)
target_include_directories(brae_rhoSimpleFoam_legacy PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/src/applications/solvers/rhoSimpleFoam)
target_link_libraries(brae_rhoSimpleFoam_legacy PRIVATE brae_core)

# The ENERGY boundary conditions -- fixedEnergy/gradientEnergy/mixedEnergy updateCoeffs -- against
# OpenFOAM's OWN coefficients, dumped by tools/dumpEnergyBC after he.boundaryField().updateCoeffs().
# Coefficients rather than a converged field, because on perfectGas + hConst the live update is
# arithmetically the static construction-time mapping and an end-to-end run cannot discriminate; three
# controls (affine-instead-of-Cpv, T's value instead of its refValue, refGrad unscaled) can.
# brae's cellLimited Gauss linear gradient against OpenFOAM's own (tools/dumpLimitedGrad), cell by cell.
add_executable(test_limited_grad ${PROJECT_SOURCE_DIR}/tests/test_limited_grad.cu)
target_include_directories(test_limited_grad PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_limited_grad PRIVATE brae_core)

# The time-step control against OpenFOAM's own log, a step a row: the program is run by
# tests/interfoam_write/clock/deltat_openfoam_log.sh, which stages the OpenFOAM run and reads its RESULT line.
add_executable(test_time_controls_openfoam_log ${PROJECT_SOURCE_DIR}/tests/test_time_controls_openfoam_log.cu)
target_include_directories(test_time_controls_openfoam_log PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_time_controls_openfoam_log PRIVATE brae_core)

# The whole clock against OpenFOAM's own log -- setDeltaT.H, Time's write cadence and every function object's --
# with the run's controlDict read by the solver's own readers: run by
# tests/interfoam_write/clock/function_object_write_times.sh and function_object_write_entries.sh.
add_executable(test_time_controls_cadence_log ${PROJECT_SOURCE_DIR}/tests/test_time_controls_cadence_log.cu)
target_include_directories(test_time_controls_cadence_log PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_time_controls_cadence_log PRIVATE brae_core)

# brae's three flux corrections against real OpenFOAM's own.
add_executable(test_flux_map_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_flux_map_vs_openfoam.cu)
target_include_directories(test_flux_map_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_flux_map_vs_openfoam PRIVATE brae_core)

# OpenFOAM's faceAreaWeightAMI on the host, against the weights its own cyclicAMI holds on
# RAS/mixerVesselAMI as the rotor turns (run by tests/interfoam_ami_vs_openfoam.sh)
add_executable(test_face_area_weight_ami ${PROJECT_SOURCE_DIR}/tests/test_face_area_weight_ami.cu)
target_include_directories(test_face_area_weight_ami PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_face_area_weight_ami PRIVATE brae_core)

# interFoam across a rotating cyclicAMI against real OpenFOAM on RAS/mixerVesselAMI. The script carries
# the measurements, the controls and the fail-proofs.
# An instrument, not a gate: both interFoam loops on one case, and the device's intermediates of the last
# step against the host's (tools/interTapsDiff/inter_taps_diff.cu)
add_executable(brae_inter_taps_diff ${PROJECT_SOURCE_DIR}/tools/interTapsDiff/inter_taps_diff.cu)
target_include_directories(brae_inter_taps_diff PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae_inter_taps_diff PRIVATE brae_core)

# diag_compare: DIAGNOSTIC, not a test. Assembles the same scalar equation as the OpenFOAM utility
# in tools/dumpScalarMatrix and compares the matrix cell by cell. Needs OF's Ddiag/Dsource in the case,
# so it is built but never registered with ctest.
# D2: boundary coefficients (internalCoeffs/boundaryCoeffs) vs OF for REAL BC types. D1 compared only the
# diagonal and source, and only with fixedValue/zeroGradient.
# Phase 0 stage harness: compare one SIMPLE stage brae-vs-OF in dependency order.
add_executable(stage_compare ${PROJECT_SOURCE_DIR}/tests/stage_compare.cu)
target_include_directories(stage_compare PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(stage_compare PRIVATE brae_core)

add_executable(diag_compare ${PROJECT_SOURCE_DIR}/tests/diag_compare.cu)
target_include_directories(diag_compare PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(diag_compare PRIVATE brae_core)

# ---------------------------------------------------------------------------
# Demo instrumentation (demo/amgpcg): dumps the real brae AMG hierarchy and the
# per-iteration AMG-PCG state as CSV, for the replay animation. Not a test.
# demo/ is gitignored (local-only, exceeds GitHub's file-size limit), so this
# target is built ONLY when the source is present -- CI (no demo/) skips it.
# ---------------------------------------------------------------------------
if(EXISTS "${PROJECT_SOURCE_DIR}/demo/amgpcg/trace_brae.cu")
    add_executable(trace_brae ${PROJECT_SOURCE_DIR}/demo/amgpcg/trace_brae.cu)
    target_include_directories(trace_brae PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(trace_brae PRIVATE brae_core)
endif()
