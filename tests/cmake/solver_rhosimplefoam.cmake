# rhoSimpleFoam. The steady compressible solver (`brae_rhoSimpleFoam`): thermophysics, energy, the host (_cpp) and
# device arms, and its gates against real OpenFOAM v2412.
#
# Included by tests/CMakeLists.txt.

add_executable(test_rho_angledduct_structural ${PROJECT_SOURCE_DIR}/tests/test_rho_angledduct_structural.cu)
target_include_directories(test_rho_angledduct_structural PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_angledduct_structural PRIVATE brae_core)
add_test(NAME rho_angledduct_structural
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_angledduct_structural.sh)
set_tests_properties(rho_angledduct_structural PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77)

add_executable(test_rho_createfields_cuda ${PROJECT_SOURCE_DIR}/tests/test_rho_createfields_cuda.cu)
target_include_directories(test_rho_createfields_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_createfields_cuda PRIVATE brae_core)
# rhoBox is laminar and has neither inletOutlet nor a wall function; sbMatched has both, so between them
# every control below is exercised rather than reported vacuous.
add_test(NAME rho_createfields_cuda_laminar
         COMMAND test_rho_createfields_cuda ${PROJECT_SOURCE_DIR}/validation/rhoBox 0.orig)
add_test(NAME rho_createfields_cuda_turbulent
         COMMAND test_rho_createfields_cuda ${PROJECT_SOURCE_DIR}/validation/sbMatched 0.orig)

add_executable(test_rho_simple_step_cuda ${PROJECT_SOURCE_DIR}/tests/test_rho_simple_step_cuda.cu)
target_include_directories(test_rho_simple_step_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_simple_step_cuda PRIVATE brae_core)
add_test(NAME rho_simple_step_cuda
         COMMAND test_rho_simple_step_cuda ${PROJECT_SOURCE_DIR}/validation/rhoBox 0.orig 8)
add_test(NAME rho_step_cuda_refusals
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_step_cuda_refusals.sh)
set_tests_properties(rho_step_cuda_refusals PROPERTIES SKIP_RETURN_CODE 77)
# The BOUNDARY arm. rhoBox carries only fixedValue, zeroGradient, noSlip and empty, so the whole
# updateCoeffs() block at the top of the step is a no-op there and the arm above cannot see it at all --
# which is exactly why the device driver went without one. sbMatched is the only compressible fixture
# with four inletOutlet patches and a flowRateInletVelocity inlet. ONE iteration and no forced limiters;
# the reasons are in the test's own header. Measured: Ux 4.496e-12, and 1.276e-02 with the driver's
# updateBoundaryCoeffs call removed.
add_test(NAME rho_simple_step_cuda_boundary
         COMMAND test_rho_simple_step_cuda ${PROJECT_SOURCE_DIR}/validation/sbMatched 0.orig --boundary)
# DEVICE-RESIDENT arms: the same driver with thermo.correct(), thermo.rho() and the effective transport
# run on the GPU instead of round-tripping the field set to the _cpp reference and back. Two fixtures,
# because between them they take BOTH branches of every switch in the device thermo:
#   rhoBox     hePsiThermo  + const      + sensibleEnthalpy       -> rho = p*psi, alpha = mu/Pr, CpByCpv 1
#   sbMatched  heRhoThermo  + sutherland + sensibleInternalEnergy -> rho = stored, Eucken kappa/Cp, gamma
# Each carries an oracle control that recomputes the thermo relations on the host from the device's own
# he and p, so the arm cannot pass by two implementations agreeing on nothing.
add_test(NAME rho_simple_step_cuda_device_thermo
         COMMAND test_rho_simple_step_cuda ${PROJECT_SOURCE_DIR}/validation/rhoBox 0.orig 8 --device-thermo)
add_test(NAME rho_simple_step_cuda_device_thermo_rho
         COMMAND test_rho_simple_step_cuda ${PROJECT_SOURCE_DIR}/validation/sbMatched 0.orig --boundary --device-thermo)

add_executable(test_rho_kepsilon_cuda ${PROJECT_SOURCE_DIR}/tests/test_rho_kepsilon_cuda.cu)
target_include_directories(test_rho_kepsilon_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_kepsilon_cuda PRIVATE brae_core)
add_test(NAME rho_kepsilon_cuda
         COMMAND test_rho_kepsilon_cuda ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_rho_pceqn_cuda ${PROJECT_SOURCE_DIR}/tests/test_rho_pceqn_cuda.cu)
target_include_directories(test_rho_pceqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_pceqn_cuda PRIVATE brae_core)
add_test(NAME rho_pceqn_cuda_subsonic
         COMMAND test_rho_pceqn_cuda ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_pceqn_cuda_subsonic_turbulent
         COMMAND test_rho_pceqn_cuda ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
add_test(NAME rho_pceqn_cuda_transonic
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_TRANSONIC=1
                 $<TARGET_FILE:test_rho_pceqn_cuda> ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_pceqn_cuda_transonic_turbulent
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_TRANSONIC=1
                 $<TARGET_FILE:test_rho_pceqn_cuda> ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_rho_eeqn_cuda ${PROJECT_SOURCE_DIR}/tests/test_rho_eeqn_cuda.cu)
target_include_directories(test_rho_eeqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_eeqn_cuda PRIVATE brae_core)
add_test(NAME rho_eeqn_cuda_e
         COMMAND test_rho_eeqn_cuda ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282)
add_test(NAME rho_eeqn_cuda_e_turbulent
         COMMAND test_rho_eeqn_cuda ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
add_test(NAME rho_eeqn_cuda_h
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_ENERGY=h
                 $<TARGET_FILE:test_rho_eeqn_cuda> ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_rho_peqn_cuda ${PROJECT_SOURCE_DIR}/tests/test_rho_peqn_cuda.cu)
target_include_directories(test_rho_peqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_peqn_cuda PRIVATE brae_core)
add_test(NAME rho_peqn_cuda_subsonic
         COMMAND test_rho_peqn_cuda ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_peqn_cuda_subsonic_turbulent
         COMMAND test_rho_peqn_cuda ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
add_test(NAME rho_peqn_cuda_transonic
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_TRANSONIC=1
                 $<TARGET_FILE:test_rho_peqn_cuda> ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_peqn_cuda_transonic_turbulent
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_TRANSONIC=1
                 $<TARGET_FILE:test_rho_peqn_cuda> ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_rho_ueqn_cuda ${PROJECT_SOURCE_DIR}/tests/test_rho_ueqn_cuda.cu)
target_include_directories(test_rho_ueqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_ueqn_cuda PRIVATE brae_core)
add_test(NAME rho_ueqn_cuda
         COMMAND test_rho_ueqn_cuda ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_ueqn_cuda_turbulent
         COMMAND test_rho_ueqn_cuda ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

# One binary, one scheme each. The tolerances differ by scheme on purpose -- the r-ratio limiters divide
# by a face difference that approaches zero, while linearUpwindV and LUST reach 1e-16.
# `grad(U) cellLimited Gauss linear 1` on the 2D pitzDailyTurb, which is the ONLY registered arm that
# makes the EMPTY-patch question visible. emptyFvPatchField is zero-sized in OpenFOAM so those faces do
# not exist there; brae keeps them and must skip them in both the limiter's range and its face loop, and
# in the face loop it is not harmless -- Cf - C points out of the 2D plane, so the extrapolate is
# round-off and the ratio against maxDelta can clamp the limiter below what any real face asks for.
add_test(NAME rho_ueqn_cuda_cell_limited
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_GRADLIMIT=1.0
                 $<TARGET_FILE:test_rho_ueqn_cuda> ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

foreach(_sch upwind limitedLinear limitedLinearV LUST linearUpwindV)
    add_test(NAME rho_ueqn_cuda_scheme_${_sch}
             COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_SCHEME=${_sch}
                     $<TARGET_FILE:test_rho_ueqn_cuda> ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
endforeach()
# grad(U) leastSquares on the device (deviceLeastSquaresGradU / three scalar fits) against the host
# reference, under limitedLinearV so the limiter's gradient is exercised beside divDevRhoReff's and the
# corrected laplacian's. The gradient itself is gated against OpenFOAM's own grad(U) by
# leastsquares_grad_vs_openfoam (its device twin).
add_test(NAME rho_ueqn_cuda_lsq
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_SCHEME=limitedLinearV BRAE_TEST_GRADLSQ=1
                 $<TARGET_FILE:test_rho_ueqn_cuda> ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_test(NAME rho_smoothsolver_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_smoothsolver_vs_openfoam.sh)
set_tests_properties(rho_smoothsolver_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

add_test(NAME rho_reproducible
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_reproducible.sh)
set_tests_properties(rho_reproducible PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)

# rhoSimpleFoam phase 4.3: alphat wall conditions load, Jayatilleke refused by name
add_executable(test_alphat_bc ${PROJECT_SOURCE_DIR}/tests/test_alphat_bc.cu)
target_include_directories(test_alphat_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_alphat_bc PRIVATE brae_core)
add_test(NAME alphat_bc COMMAND test_alphat_bc ${PROJECT_SOURCE_DIR}/validation/rhoBox)

# Host boundary muEff: OF takes nut_b from the patch's own boundaryField, the legacy device path
# extrapolates the adjacent cell. Verified against nut as OpenFOAM itself wrote it -- no instrumentation.
# The test self-reports as vacuous on a case where the two candidates coincide, so the case matters.
add_executable(test_boundary_mu_eff ${PROJECT_SOURCE_DIR}/tests/test_boundary_mu_eff.cu)
target_include_directories(test_boundary_mu_eff PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/src/applications/solvers/rhoSimpleFoam/cpu)
target_link_libraries(test_boundary_mu_eff PRIVATE brae_core)
#
# THE CASE IS PART OF THE TEST. Measured over the checked-in cases, discriminating faces (where the
# patch nut and the adjacent cell nut actually differ, i.e. where the two implementations can be told
# apart) are: backwardFacingStep2D 0/41746, channel 0/20500, duct3d_cf 0/11400, airfoil 78/21812.
# rhoBox has no nut at all. So airfoil is the ONLY checked-in case that can go red here, and even it
# is weak at 0.36%. That measurement IS the explanation for how the 607x boundary-muEff defect
# survived every gate: on a blockMesh duct the extrapolated cell nut and the patch nut coincide.
# A stronger gate needs a case with a `calculated` inlet nut against a turbulent interior.
# scalarTransport gate. Until this existed NOTHING in the suite carried a tracer, and three separate
# defects lived behind that: the field solved but was never written; the convection scheme was hardcoded
# instead of read from div(phi,<field>); and a transient tracer bounded by 1.0 reached 6.32 because the
# functionObject ran before the solver knew deltaT. All three were green on the full suite throughout.
# Carries its own negative control (scheme removed under `default none` must be REFUSED), so the gate
# is proven able to fail rather than merely passing.
# Restart fidelity. A restart must reproduce a continuous run: OF pairs createPhi.H:45 READ_IF_PRESENT
# with AUTO_WRITE so phi RESUMES rather than being rebuilt from fvc::flux(U). Before the fix this case
# measured U 7.85e-06 / p 2.08e-05 against a 9.5e-12 reproducibility floor -- six orders above noise,
# shipping, and covered by nothing. Mutation-checked: reverting the phi read makes this test fail with
# exactly those numbers.
add_test(NAME restart_continuity
         COMMAND bash ${PROJECT_SOURCE_DIR}/validation/restart_continuity.sh
                 $<TARGET_FILE:brae>
                 ${PROJECT_SOURCE_DIR}/validation/channel
                 ${PROJECT_BINARY_DIR}/restart_continuity_work)
set_tests_properties(restart_continuity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME tracer_transport
         COMMAND bash ${PROJECT_SOURCE_DIR}/validation/tracer_transport.sh
                 $<TARGET_FILE:brae_rhoSimpleFoam>
                 ${PROJECT_SOURCE_DIR}/validation/tracerBox
                 ${PROJECT_BINARY_DIR}/tracer_transport_work)
set_tests_properties(tracer_transport PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

add_test(NAME boundary_mu_eff COMMAND test_boundary_mu_eff
         ${PROJECT_SOURCE_DIR}/validation/airfoil 0)
# validation/ is .gitignore'd wholesale, so the airfoil case reaches a fresh checkout only if it was
# force-added. Without this the test's 77 (its "fixture absent" code) is reported as a FAILURE, which
# reads like a solver bug rather than a missing file. Skipping is the fallback, not the intent: airfoil is
# the only checked-in case with faces where the patch nut and the cell nut differ, so a skip here means
# this gate proves nothing -- the fixture belongs in the repo.
set_tests_properties(boundary_mu_eff PROPERTIES SKIP_RETURN_CODE 77)

# rhoSimpleFoam phase 3 gate: brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam on a heated duct.
# Needs an OF install; skipped (77) if absent.
add_test(NAME rho_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/rho_vs_openfoam.sh)
set_tests_properties(rho_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam solver-control gate: SIMPLEC + nNonOrthogonalCorrectors + loose relTol, none of which
# any other compressible case sets. The compressible driver silently dropped 15 fvSolution controls until
# readLinearSolverControls; this is what makes that visible. Tolerance 2e-4 is brae's SIMPLEC floor
# (bisected: nNonOrth alone is exact at 2.6e-6), an open item recorded in the script.
add_test(NAME ctl_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/ctl_vs_openfoam.sh)
set_tests_properties(ctl_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# rhoSimpleFoam kEpsilon gate. Same duct as the kOmegaSST gate, RASModel kEpsilon.
# Tolerance is 2e-2 rather than the SST's 5e-4 for a MEASURED reason: brae's kEpsilon already differs
# from OF by k 5.09e-3 / epsilon 8.11e-3 in the INCOMPRESSIBLE solver on this same geometry
# (brae_simpleFoam vs OF simpleFoam), so that is the floor. This gate certifies the COMPRESSIBLE
# additions land on that floor, not brae's kEpsilon fidelity -- which is a separate open item.
# Dropping the rho weighting gives k 5.0e-2 / eps 2.3e-2, well clear of the floor, so it still bites.
add_test(NAME ke_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/ke_vs_openfoam.sh)
set_tests_properties(ke_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# rhoSimpleFoam BC gate: flowRateInletVelocity. OF sets U_b = avgU*n with
#   avgU = -flowRate/gSum(rho*magSf)   (flowRateInletVelocityFvPatchVectorField::updateValues)
# using the OUTWARD normal, so a positive rate is inflow. This gate runs the massFlowRate form on
# validation/rhoFR; the volumetric form and the incompressible/PIMPLE lineages are covered by
# flowrate_device_solver_vs_openfoam.
#
# This comment used to say the volumetric form "is the easy half, since its sum is pure area and never
# changes". The sum never changes -- and the device never computed it at all, because the flow-rate mask
# was keyed on bcCategory()==9, which is the MASS form only. validation/rhoFRvol sat in the tree
# ungated, running at the wrong flow rate (U 3.27e-02 against real OpenFOAM), for exactly as long as
# that sentence stood. An argument for why something cannot be wrong is not coverage.
#
# It also pins the bug that took the longest to find: buildDeviceVectorBoundary mapped bcCategory 9 to
# device bcType 9, an UNKNOWN type, because only the scalar buildDeviceBoundary had been taught about it.
# U is a vector field, so the inlet ran with an undefined BC and the profile bowed at the wall corners
# (row 0 5.041 vs OF 5.163) while mid-channel looked right. Reverting that one line gives T 7.7e-4.
add_test(NAME fr_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/fr_vs_openfoam.sh)
set_tests_properties(fr_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam BC gate: a totalPressure inlet on a COMPRESSIBLE case. OF's totalPressure branches on
# the dimensions of p -- kinematic gives p0 - 0.5*neg(phi)|U|^2, absolute gives p0 - 0.5*RHO*neg(phi)|U|^2.
# brae implemented only the kinematic form, so every compressible totalPressure inlet was missing the
# density on its dynamic head. No other gate uses totalPressure, so none of them could see it.
add_test(NAME tp_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/tp_vs_openfoam.sh)
set_tests_properties(tp_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam scheme gate: div(phi,h|e) and div(phi,K|Ekp) limitedLinear rather than upwind.
# brae used to HARDCODE upwind for the energy equation and ignore fvSchemes entirely -- a silent
# first-order downgrade. Every other compressible gate specifies upwind, so none of them can see it.
# Covers BOTH halves: the implicit div(phi,he) and the explicit div(phi,K|Ekp), which are separate code
# paths and were separately wrong.
add_test(NAME lu_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/lu_vs_openfoam.sh)
set_tests_properties(lu_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam transport gate: the same heated duct with transport sutherland instead of const.
# The two models compute the laminar thermal diffusivity DIFFERENTLY (const: mu/Pr; sutherland: the
# Eucken kappa/Cp, with no Pr in the dict at all), and sutherland makes mu vary strongly with T -- which
# is what exposes whether the energy laplacian uses the PATCH diffusivity or the adjacent cell's. With
# transport const those are numerically identical, so the const gates cannot see either error.
add_test(NAME suth_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/suth_vs_openfoam.sh)
set_tests_properties(suth_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam energy-form gate: the same heated duct with energy sensibleInternalEnergy (OF "e").
# Every stock OF rhoSimpleFoam tutorial uses e rather than sensibleEnthalpy, and the two forms must give
# the SAME temperature field -- OF's own h and e runs agree to 3e-11. They only do so if Cv, the gamma on
# alphaEff and the p/rho in Ekp are all right, so this gate pins three things the h gate cannot see.
add_test(NAME rhoE_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/rhoE_vs_openfoam.sh)
set_tests_properties(rhoE_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# E1 (dict_audit finding): OF controlDict write cadence on the compressible driver. Found by the audit --
# gpuRhoSimpleFoam read NONE of writeControl/writeInterval/purgeWrite/deltaT and wrote one time directory.
add_test(NAME wc_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/wc_vs_openfoam.sh)
set_tests_properties(wc_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# E5b (dict_audit, found only once the audit ran on REFUSED cases): OF RAS { turbulence off; } freezes the
# model -- k/epsilon|omega/nut keep their initial values and momentum uses that frozen nut.
add_test(NAME turbswitch_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/turbswitch_vs_openfoam.sh)
set_tests_properties(turbswitch_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# mixedEnergy: a  (Robin) T wall -- the third of OF's three energy BC families, and the only one
# that exercises OF's (1-vf) weight on refGrad.
add_test(NAME mx_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/mx_vs_openfoam.sh)
set_tests_properties(mx_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# heRhoThermo vs hePsiThermo rho TIMING at the end of pEqn.H. Compared at iteration 1, because the effect
# IS a one-iteration lag and a low-Mach case cannot distinguish the two.
add_test(NAME rhotiming_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/rhotiming_vs_openfoam.sh)
set_tests_properties(rhotiming_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# B5: heat-flux wall (fixedGradient T). The compressible gates above all use fixedValue T walls, which
# exercise only the fixedEnergy half of basicThermo::heBoundaryTypes(); a fixedGradient T was read and then
# silently discretised as zeroGradient, i.e. an adiabatic wall that converged to a plausible wrong answer.
# Also the only gate that asserts OpenFOAM can PARSE what brae writes.
add_test(NAME ffpi_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/ffpi_vs_openfoam.sh)
set_tests_properties(ffpi_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# The other half of that boundary condition's envelope: fixedFluxPressure TOGETHER WITH MRF is refused,
# because constrainPressure.C:70 makes the flux relative and deviceConstrainPressure does not. Needs a
# fixture with a REAL rotating cellZone -- see the script's header for why pitzDaily cannot host it.
add_test(NAME v2_mrf_fixedfluxpressure_refusal
         COMMAND ${PROJECT_SOURCE_DIR}/tests/v2_mrf_fixedfluxpressure_refusal.sh)
set_tests_properties(v2_mrf_fixedfluxpressure_refusal PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 900)

add_test(NAME ffp_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/ffp_vs_openfoam.sh)
set_tests_properties(ffp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_executable(test_simple_adjustphi_cpp ${PROJECT_SOURCE_DIR}/tests/test_simple_adjustphi_cpp.cu)
target_include_directories(test_simple_adjustphi_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_adjustphi_cpp PRIVATE brae_core)
add_executable(test_abl_profile ${PROJECT_SOURCE_DIR}/tests/test_abl_profile.cu)
target_include_directories(test_abl_profile PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_abl_profile PRIVATE brae_core)
add_test(NAME abl_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/abl_vs_openfoam.sh)
set_tests_properties(abl_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME adjm_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/adjm_vs_openfoam.sh)
set_tests_properties(adjm_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_test(NAME df_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/df_vs_openfoam.sh)
set_tests_properties(df_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_test(NAME sym_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/sym_vs_openfoam.sh)
set_tests_properties(sym_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# rho.relax() on the SIMPLEC branch relaxes against the iteration's starting density (rho.prevIter()),
# both mirror arms vs real OpenFOAM mid-transient -- tests/rho_simplec_rhorelax_vs_openfoam.sh.
add_test(NAME rho_simplec_rhorelax_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_simplec_rhorelax_vs_openfoam.sh)
set_tests_properties(rho_simplec_rhorelax_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# div(phi,K)/div(phi,Ekp) is read from its OWN fvSchemes entry, not copied from the energy entry --
# both mirror arms vs real OpenFOAM in both directions, plus the by-name refusal on K alone.
add_test(NAME rho_ke_scheme_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_ke_scheme_vs_openfoam.sh)
set_tests_properties(rho_ke_scheme_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# kOmegaSST on the host mirror takes its coefficients, Prt and omega relaxation from the case, and the
# wall functions keep their own Cmu -- tests/rho_sst_coeffs_vs_openfoam.sh, at iteration 1.
add_test(NAME rho_sst_coeffs_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_sst_coeffs_vs_openfoam.sh)
set_tests_properties(rho_sst_coeffs_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# The OF-mirror kOmegaSST closure ON THE DEVICE, against real OpenFOAM at the round-off floor, on two
# fixtures, with the LEGACY closure through the same harness as the control that the bound bites --
# tests/rho_sst_device_vs_openfoam.sh.
add_test(NAME rho_sst_device_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_sst_device_vs_openfoam.sh)
set_tests_properties(rho_sst_device_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# flowRateInletVelocity's value must reach the MOMENTUM ASSEMBLY, in both the mass and the volumetric
# form, and `rho none` must take OpenFOAM's volumetric branch --
# tests/rho_flowrate_inlet_vs_openfoam.sh.
add_test(NAME rho_flowrate_inlet_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_flowrate_inlet_vs_openfoam.sh)
set_tests_properties(rho_flowrate_inlet_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# flowRateInletVelocity on the DeviceSimpleSolver lineage -- both forms, and all three of its step()
# variants (incompressible SIMPLE, compressible rhoSimpleStep, PIMPLE), which between them cover five
# drivers -- tests/flowrate_device_solver_vs_openfoam.sh.
add_test(NAME flowrate_device_solver_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/flowrate_device_solver_vs_openfoam.sh)
set_tests_properties(flowrate_device_solver_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# brae's OWN output must be readable by real OpenFOAM and mean the same thing: the writer has to echo
# the INPUT entries OpenFOAM's own write() emits and its reader requires (`intensity`, `mixingLength`,
# `rho`). A field-vs-field gate cannot see this, because it compares a file nothing reads back --
# tests/written_bc_roundtrip_vs_openfoam.sh.
# leastSquares gradient, host and device, against OpenFOAM's OWN grad(T) -- a 2x2 whose off-diagonal is
# the control that the two schemes genuinely differ on this mesh, and whose fixture is 2-D so that
# SymmTensor::safeInv (the singular empty direction) is exercised --
# tests/leastsquares_grad_vs_openfoam.sh.
# A nut wall function must be resolved through OpenFOAM's patch/group/regex rule, not by exact name --
# a case keying its walls `"wall.*"` silently kept the permissive nutk default, which is nutk running
# under another name. The load-bearing arm is the CONTROL that nutU and nutk differ on the fixture --
# tests/nut_wallfn_patchkey_vs_openfoam.sh.
add_test(NAME nut_wallfn_patchkey_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/nut_wallfn_patchkey_vs_openfoam.sh)
set_tests_properties(nut_wallfn_patchkey_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# brae's H2O he->T inversion against OpenFOAM's OWN answer, PATH included: OpenFOAM stops on a
# temperature step measured from the initial guess, so its answer depends on T0 and a converged
# inversion is a different function -- tests/liquid_inversion_vs_openfoam.sh, oracle tools/liqref.
add_test(NAME liquid_inversion_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/liquid_inversion_vs_openfoam.sh)
set_tests_properties(liquid_inversion_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME leastsquares_grad_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/leastsquares_grad_vs_openfoam.sh)
set_tests_properties(leastsquares_grad_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

add_test(NAME written_bc_roundtrip_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/written_bc_roundtrip_vs_openfoam.sh)
set_tests_properties(written_bc_roundtrip_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# The totalPressure family (totalPressure, uniformTotalPressure, fanPressure), totalTemperature and phi on a symmetry
# patch, written by simpleFoam, pimpleFoam, rhoSimpleFoam and rhoCentralFoam and read back by real OpenFOAM, OpenFOAM's
# output read by brae, and brae restarted from its own output against its uninterrupted run; each arm with a control
# that holds the file as brae wrote it before (tests/total_bc_roundtrip_vs_openfoam.sh says what each one found).
add_test(NAME total_bc_roundtrip_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/total_bc_roundtrip_vs_openfoam.sh)
set_tests_properties(total_bc_roundtrip_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800
                     ENVIRONMENT "BRAE_BIN=$<TARGET_FILE:brae>;BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

# turbulence->validate() at construction, both models, both arms where the device runs the model:
# the first iteration must sit on OpenFOAM's -- tests/rho_validate_vs_openfoam.sh.
add_test(NAME rho_validate_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_validate_vs_openfoam.sh)
set_tests_properties(rho_validate_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# The case's gradient limiters and laplacian limiter reach the mirror: grad(e|Ekp) cellLimited on both
# arms, `limited <psi>` on the device (refused by name on the host), `limited 0` refused on both --
# tests/rho_limiters_vs_openfoam.sh.
add_test(NAME rho_limiters_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_limiters_vs_openfoam.sh)
set_tests_properties(rho_limiters_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# The TURBULENCE half of the same scheme, which rho_limiters_vs_openfoam above cannot reach: its fixture
# is laminar by its own header, so no turbulence laplacian is exercised under a cap. Assembly level,
# against OpenFOAM's own fvm::laplacian via tools/dumpKEpsilon; measured ARM 1.71e-11 against a CONTROL of
# 6.65e-06, and it fails at 6.65e-06 with the forwarding line in rhoTurbulenceHook.cu commented out.
# Skips (77) when libdumpKEpsilon.so has not been built -- cd tools/dumpKEpsilon && wmake libso.
add_test(NAME rho_turb_laplacian_limiter_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_turb_laplacian_limiter_vs_openfoam.sh)
set_tests_properties(rho_turb_laplacian_limiter_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# snGradSchemes and laplacianSchemes are two blocks for two operators; brae collapsed them into one
# pair of flags. Both arms, because the SIMPLEC fvc::snGrad(p) was misrouted in each independently.
add_test(NAME rho_sngrad_scheme_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_sngrad_scheme_vs_openfoam.sh)
set_tests_properties(rho_sngrad_scheme_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
add_test(NAME rho_sngrad_scheme_cpp_vs_openfoam
         COMMAND ${CMAKE_COMMAND} -E env ARM=cpu
                 ${PROJECT_SOURCE_DIR}/tests/rho_sngrad_scheme_vs_openfoam.sh)
set_tests_properties(rho_sngrad_scheme_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME tpm_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/tpm_vs_openfoam.sh)
set_tests_properties(tpm_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_test(NAME tf_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/tf_vs_openfoam.sh)
set_tests_properties(tf_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_test(NAME hf_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/hf_vs_openfoam.sh)
set_tests_properties(hf_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam phase 4 gate: brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam, kOmegaSST heated duct.
# Compares seven fields (T p U k omega nut rho) -- the compressible turbulence coupling is the kind that
# converges cleanly while being wrong, so residuals prove nothing here. Skipped (77) without an OF install.
# squareBend (transonic + consistent, Mach ~0.96) to CONVERGENCE vs OF. This is the evidence that
# lifted the `transonic yes` refusal: it is the only gate with an ASYMMETRIC pressure matrix (implicit
# fvm::div(phid,p) -> BiCGStab) and the only one on heRhoThermo's stored-rho timing at high Mach.
add_test(NAME transonic_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/transonic_vs_openfoam.sh)
set_tests_properties(transonic_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     LABELS slow
                     TIMEOUT 1800)

# aerofoilNACA0012 to CONVERGENCE vs OF. Slow (~700 outer iterations on both sides) and labelled so.
# It is the only gate that can catch a defect living in the pressure equation's net mass balance rather
# than in a field norm: the pinned-pressure bug reached endTime with every iteration-2 stage agreeing to
# 1e-3. See the header of the script, and cudafoam/solver-porting-dissection.md for how it was found.
add_test(NAME naca_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/naca_vs_openfoam.sh)
set_tests_properties(naca_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     LABELS slow
                     TIMEOUT 3600)

add_test(NAME sst_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/sst_vs_openfoam.sh)
set_tests_properties(sst_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# linearUpwind on the turbulence scalars -- every other gate uses upwind there, so this is the only
# one that can see which convection scheme k/omega actually run.
add_test(NAME luturb_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/luturb_vs_openfoam.sh)
set_tests_properties(luturb_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1200)

# SpalartAllmaras had NO ctest coverage; this is the first SA gate (airFoil2D, external aero + wall function).
add_test(NAME sa_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/sa_vs_openfoam.sh)
set_tests_properties(sa_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1200)

# inletOutlet on TEMPERATURE -- the only gate that exercises the energy boundary's inletOutlet mask.
add_test(NAME io_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/io_vs_openfoam.sh)
set_tests_properties(io_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# Turbulent inlet BCs on a flowRateInletVelocity inlet -- the only gate where the set-up value and the
# converged value differ, so the per-iteration refresh is actually exercised.
add_test(NAME ti_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/ti_vs_openfoam.sh)
set_tests_properties(ti_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# linearUpwind on the kinetic-energy term + grad(K)'s own boundary values (sensibleInternalEnergy).
add_test(NAME ke2_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/ke2_vs_openfoam.sh)
set_tests_properties(ke2_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# Regex-keyed boundaryField entries (alphatWallFunction Prt) -- OF patch/group/regex resolution.
add_test(NAME rx_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/rx_vs_openfoam.sh)
set_tests_properties(rx_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# pressureControl pMaxFactor/pMinFactor reference: only patches that FIX a value (OF pressureControl.C:75).
add_test(NAME pm_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/pm_vs_openfoam.sh)
set_tests_properties(pm_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# endTime is an ABSOLUTE time (OF Time::run), not a run length, and a restarted run must reproduce a
# continuous one. Both steady drivers read endTime as a length, which is only correct from startTime 0 --
# so every fresh-start gate was blind to it. Uses rhoBox: the identity needs a QUIET state, and
# angledDuct's cold start is chaotic under the pressureControl ceiling.
add_test(NAME roundtrip_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/roundtrip_vs_openfoam.sh)
set_tests_properties(roundtrip_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# flowRateInletVelocity must be fed the SOLVER's rho (the relaxed one OF registers), not thermo.rho().
# Needs a case with BOTH a relaxed rho and a flow-rate inlet; angledDuct is the only one, which is why
# this defect survived every other gate and left the case unable to converge at all.
add_test(NAME flowrate_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/flowrate_vs_openfoam.sh)
set_tests_properties(flowrate_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# RESTART fidelity: OF's createFields.H reads phi and rho with IOobject::READ_IF_PRESENT, so a restart
# resumes the STORED fields. Every other compressible gate starts from 0/, where neither file exists and
# both codes take the fallback -- which is how brae shipped without either read. This gate restarts both
# codes from the same OF-written time directory, and carries its own negative control (the same brae run
# with phi/rho deleted, which must FAIL the tolerance).
add_test(NAME restart_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/restart_vs_openfoam.sh)
set_tests_properties(restart_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# rhoSimpleFoam phase 2: mass flux reduces exactly to the volumetric flux at constant density
add_executable(test_rho_flux ${PROJECT_SOURCE_DIR}/tests/test_rho_flux.cu)
target_include_directories(test_rho_flux PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_flux PRIVATE brae_core)
add_test(NAME rho_flux COMMAND test_rho_flux)

# rhoSimpleFoam phase 1 gate: sensible-enthalpy equation on a frozen velocity field
add_executable(test_energy_frozen ${PROJECT_SOURCE_DIR}/tests/test_energy_frozen.cu)
target_include_directories(test_energy_frozen PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_energy_frozen PRIVATE brae_core)
add_test(NAME energy_frozen COMMAND test_energy_frozen)
# Gate 1 proper: same case, brae vs OpenFOAM scalarTransportFoam. Needs an OF install; skipped if absent.
add_test(NAME energy_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/energy_vs_openfoam.sh)
set_tests_properties(energy_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77)

# INTEGRATION: the real deviceThermoCorrect() liquid path end to end -- inversion wired to property
# evaluation. Catches the wiring errors the standalone tests structurally cannot: wrong energy
# semantics, wrong p, properties evaluated before T is updated, solver rho written instead of rhoThermo.
add_executable(test_liquid_correct ${PROJECT_SOURCE_DIR}/tests/test_liquid_correct.cu)
target_include_directories(test_liquid_correct PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_liquid_correct PRIVATE brae_core)
add_test(NAME liquid_correct COMMAND test_liquid_correct)

# e -> T inversion (sensibleInternalEnergy), the form squareBendLiq transports. Same generic Newton,
# different F and dF/dT. 13 temperatures x 4 PRESSURES, because Es = Hs - p/rho and a single pressure
# cannot distinguish e(T) from e(p,T). Oracle is OF's thermophysicalPropertiesSelector<liquidProperties>.
add_executable(test_etot ${PROJECT_SOURCE_DIR}/tests/test_etot.cu)
target_include_directories(test_etot PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_etot PRIVATE brae_core)
add_test(NAME etot COMMAND test_etot)

# Standalone h -> T Newton inversion for the liquid path. Targets are OF-generated enthalpies and the
# initial guesses are REVERSED (280 K target started from 400 K), so the test exercises inversion rather
# than a good guess. Accepts on the ENTHALPY residual, not |dT| -- a wrong derivative changes the path,
# not the fixed point, so |dT| alone would pass it.
add_executable(test_hetot ${PROJECT_SOURCE_DIR}/tests/test_hetot.cu)
target_include_directories(test_hetot PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_hetot PRIVATE brae_core)
add_test(NAME hetot COMMAND test_hetot)

# Liquid thermo FIELD POPULATION: given a NONUNIFORM T field (280-400 K, one temperature per cell),
# does brae fill Cp/mu/kappa/rhoThermo per cell and per boundary face with OF's values? A uniform field
# cannot tell "evaluated per cell" from "broadcast", which is why the test refuses to use one.
add_executable(test_liquid_fields ${PROJECT_SOURCE_DIR}/tests/test_liquid_fields.cu)
target_include_directories(test_liquid_fields PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_liquid_fields PRIVATE brae_core)
add_test(NAME liquid_fields COMMAND test_liquid_fields)

add_executable(test_liquid_fixed_bc ${PROJECT_SOURCE_DIR}/tests/test_liquid_fixed_bc.cu)
target_include_directories(test_liquid_fixed_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_liquid_fixed_bc PRIVATE brae_core)
add_test(NAME liquid_fixed_bc COMMAND test_liquid_fixed_bc)

add_executable(test_generalized_newtonian ${PROJECT_SOURCE_DIR}/tests/test_generalized_newtonian.cu)
target_include_directories(test_generalized_newtonian PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_generalized_newtonian PRIVATE brae_core)
add_test(NAME generalized_newtonian COMMAND test_generalized_newtonian)

# NSRDS correlation forms + H2O coefficients (squareBendLiq's `properties liquid` path) against a table
# generated by OpenFOAM ITSELF (dissect/liqref), so the test covers form and coefficients together.
# Also asserts dh/dT == Cp(T) -- the consistency a Newton he->T inversion cannot check for itself.
add_executable(test_nsrds ${PROJECT_SOURCE_DIR}/tests/test_nsrds.cu)
target_include_directories(test_nsrds PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_nsrds PRIVATE brae_core)
add_test(NAME nsrds COMMAND test_nsrds)

# rhoSimpleFoam phase 0 gate: thermophysical subsystem (perfectGas + hConst + Sutherland/const)
add_executable(test_thermo_perfect_gas ${PROJECT_SOURCE_DIR}/tests/test_thermo_perfect_gas.cu)
target_include_directories(test_thermo_perfect_gas PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_thermo_perfect_gas PRIVATE brae_core)
add_test(NAME thermo_perfect_gas COMMAND test_thermo_perfect_gas)

# The turbulence blow-up tripwire's DECISION, tested on the two measured sequences it has to tell apart:
# a start-up transient that crosses the ratio bar and recovers (a 5.96M-cell car case both brae and real
# OpenFOAM solve), and a run that genuinely diverges. Carries its own control -- a bar-only detector must
# stop the transient, otherwise the sequences prove nothing.
add_executable(test_turb_blowup ${PROJECT_SOURCE_DIR}/tests/test_turb_blowup.cu)
target_include_directories(test_turb_blowup PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_turb_blowup PRIVATE brae_core)
add_test(NAME turb_blowup COMMAND test_turb_blowup)

add_executable(test_solution_health ${PROJECT_SOURCE_DIR}/tests/test_solution_health.cu)
target_include_directories(test_solution_health PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_solution_health PRIVATE brae_core)
add_test(NAME solution_health COMMAND test_solution_health)

# rhoSimpleFoam's createFields.H against real OpenFOAM. The oracle is OpenFOAM's own
# `rhoSimpleFoam -postProcess -func "writeObjects(phi,rho)"`, which builds createFields.H's field set and
# writes it without solving -- no instrumented OpenFOAM build required. Carries the control that
# distinguishes compressibleCreatePhi.H's interpolate(rho*U) from pEqn.H's interpolate(rho)*flux(U).
add_executable(test_rho_create_fields_cpp ${PROJECT_SOURCE_DIR}/tests/test_rho_create_fields_cpp.cu)
target_include_directories(test_rho_create_fields_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_create_fields_cpp PRIVATE brae_core)
add_test(NAME rho_createfields_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_createfields_vs_openfoam.sh)

add_executable(test_energy_bc ${PROJECT_SOURCE_DIR}/tests/test_energy_bc.cu)
target_include_directories(test_energy_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_energy_bc PRIVATE brae_core)
add_test(NAME energy_bc_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/energy_bc_vs_openfoam.sh)
# The script already exits 77 when tools/dumpEnergyBC is not wmade; without this ctest called that a
# FAILURE, which is what the GH200 reported on 2026-09-15. Every other *_vs_openfoam gate that can
# skip carries this property; this one was missed.
set_tests_properties(energy_bc_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# `properties liquid` END TO END on the host mirror arm (stage H3.4): the NSRDS property correlations,
# OpenFOAM's own he -> T inversion seeded from the cell's previous T, and limitTemperature's he bounds
# as FIELDS over the cell pressure. validation/liqBox, 1200 cells, laminar, against real OpenFOAM.
add_test(NAME liquid_thermo_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/liquid_thermo_vs_openfoam.sh)
set_tests_properties(liquid_thermo_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# A TURBULENT liquid end to end (stage H3.5): kEpsilon AND kOmegaSST on validation/liqBoxRAS with
# squareBendLiq's own wall-function and turbulent-inlet set, at iterations 1 and 200, fields and the
# wall functions' patch values. The liquid enters the closure only through nu = mu(T)/rho(T).
add_test(NAME liquid_turbulence_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/liquid_turbulence_vs_openfoam.sh)
set_tests_properties(liquid_turbulence_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# squareBendLiq, OpenFOAM's own turbulent liquid tutorial, iteration by iteration on kEpsilon AND
# kOmegaSST: the four generic defects stage H3.5 found there (linearUpwind on the turbulence pair, the
# grad(k)/grad(epsilon) slot, linearUpwind's named gradient on U, the wall-function-only patch update).
add_test(NAME rho_squarebendliq_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_squarebendliq_vs_openfoam.sh)
set_tests_properties(rho_squarebendliq_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# generalizedNewtonian (powerLaw) on both mirror arms, stage S1 Half B: OpenFOAM's own
# squareBendLiqNoNewtonian from rest (iteration 1) and restarted from OpenFOAM's iteration 5 with the
# tutorial's coefficients and with the unclamped branch live, against OpenFOAM's written
# generalizedNewtonian:nu as well as U, p, T and rho.
add_test(NAME rho_generalized_newtonian_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_generalized_newtonian_vs_openfoam.sh)
set_tests_properties(rho_generalized_newtonian_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# brae's _cpp limitedLinearV / limitedLinear for div(phi,U) against OpenFOAM's OWN momentum matrix
# off-diagonals (tools/dumpPEqn), on a DEVELOPED field restarted from OpenFOAM's iteration 5.
#
# rho_ueqn_cuda_scheme_limitedLinearV holds the DEVICE against the _cpp reference; _cpp itself had no
# OpenFOAM oracle for this scheme, and when one was put under it the transcription did not hold --
# UUpper 3.31e-03 (limitedLinearV) and 4.63e-02 (limitedLinear) against a `Gauss upwind` control reading
# 1.06e-14 on the same harness. Both causes were found in OpenFOAM's own source and fixed:
#   * the limiter's gradient must be taken from U's patch values BEFORE updateCoeffs --
#     gaussConvectionScheme.C:84 takes tinterpScheme_().weights(vf) and only then builds the fvMatrix
#     at :89 whose constructor calls updateCoeffs (fvMatrix.C:396).   3.31e-03 -> 1.76e-14
#   * `Gauss limitedLinear` on a VECTOR is NVDTVD + limitFuncs::magSqr (LimitedScheme.H:188), so its
#     gradient is fvc::grad(magSqr(U)) resolved under the key `grad(magSqr(U))` -- brae hardcoded Gauss
#     where gasMixing's default is leastSquares.                      4.63e-02 -> 3.22e-13
# The bound was never loosened; it is 1e-12 throughout and the arms came to it.
add_test(NAME rho_limitedlinearv_cpp_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_limitedlinearv_cpp_vs_openfoam.sh)
set_tests_properties(rho_limitedlinearv_cpp_vs_openfoam PROPERTIES
                     SKIP_RETURN_CODE 77 TIMEOUT 3600 LABELS slow)

# `leastSquares` reaching every consumer the rhoSimpleFoam mirror has, on the HOST arm, end to end on
# validation/rhoSST against real OpenFOAM: the closure's CDkOmega and corrected-laplacian gradients, the
# turbulence limiters' own gradient, grad(p) (the velocity correction, SIMPLEC's HbyA, the non-orth
# correction) and grad(U) (divDevRhoReff's dev2 term, the production, validate()'s correctNut). Nine arms,
# each measured: the shipped case as the control, a wrong-scheme control (brae's Gauss against OpenFOAM's
# leastSquares run must DIFFER -- 6.3e-08), the limiter arms restarted from OpenFOAM's own iteration 5,
# and the CUDA arm asserted to REFUSE by name until its modules are ported. The step that unblocks
# gasMixing/injectorPipe.
add_test(NAME rho_leastsquares_closure_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_leastsquares_closure_vs_openfoam.sh)
set_tests_properties(rho_leastsquares_closure_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

# gasMixing/injectorPipe at a DEVELOPED state, host arm, both codes restarted from OpenFOAM's own iteration
# 5 (the protocol and the two measured reasons for it are in the script). The case the leastSquares work was
# for, and the one that found fvPatch::delta()'s projection, limitedLinearV's pre-updateCoeffs boundary, the
# corrected laplacian's own gradient scheme and the closures' fvm::ddt under Euler -- each with its fail-proof
# number in the script header. Worst 2.0e-11 over iterations 6-8; the CUDA arm is asserted to refuse by name.
add_test(NAME rho_gasmixing_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_gasmixing_vs_openfoam.sh)
set_tests_properties(rho_gasmixing_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600 LABELS slow)

# THE SIX rhoSimpleFoam TUTORIALS as OpenFOAM ships them, on both mirror arms, meshed by each tutorial's
# own Allrun.pre -- the standing answer to which of them brae runs. All six RUN on both arms and agree
# with OpenFOAM (a guard drawn too wide turns it red); the refusals by name live in the per-feature gates
# since the last refused tutorial, squareBendLiq's `expression` T walls, was ported. Breadth, not depth:
# three iterations, with the per-component depth left to the dedicated gates. Slow -- it meshes
# snappyHexMesh's injectorPipe -- and labelled so.
add_test(NAME rho_tutorials_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_tutorials_vs_openfoam.sh)
set_tests_properties(rho_tutorials_vs_openfoam PROPERTIES
                     SKIP_RETURN_CODE 77
                     LABELS slow
                     TIMEOUT 3600)

# aerofoilNACA0012, the only rhoSimpleFoam tutorial naming `cellLimited Gauss linear 1` for grad(k) and
# grad(omega) as well as grad(U), on BOTH mirror arms. CDkOmega takes fvc::grad(k) & fvc::grad(omega),
# which resolve their scheme by field name, so this case limits them where every other fixture does not
# -- the device arm took the plain Gauss gradient and no gate ran it (naca_vs_openfoam drives the
# pre-mirror path; limit_temperature runs this tutorial on the host arm only). Three device defects were
# found on it, all of the same class -- a patch value re-evaluated against the cells as they stand where
# OpenFOAM reads the field's STORED value -- in the closure's freestreamVelocity Up, the momentum
# gradients and the pressure equation; the script carries each measurement. Both arms are held to 1e-11
# on every field at every iteration.
add_test(NAME rho_aerofoil_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_aerofoil_vs_openfoam.sh)
set_tests_properties(rho_aerofoil_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

# The `coded` massFlowRate Function1 on both mirror arms: squareBendLiq's own snippet, and a second one
# that varies with the time (5*(1 + 0.05*t)) so a brae that froze the rate cannot pass -- which is what
# found the device k assembly reading k's patch values live where the host reference reads the stored
# ones. Carries the writer round trip (OpenFOAM restarts from brae's written U and recompiles the body)
# and eleven refusals: the four keys OpenFOAM compiles against its own headers, `$`, an empty body, two
# table forms, and the five drivers that take the rate as one number.
add_test(NAME rho_coded_function1_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_coded_function1_vs_openfoam.sh)
set_tests_properties(rho_coded_function1_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# squareBendLiq's T walls -- OpenFOAM's `type expression;` PatchFunction1 -- on both rhoSimpleFoam mirror
# arms against real OpenFOAM's WRITTEN wall values, iteration by iteration, with the cells alongside.
# Carries the constant-wall control that the comparison must reject, the writer round trip OpenFOAM
# restarts from, and the refusals by name (an unsupported function, a functions<> reference, an
# unregistered field, the expression on U and on p, the entry-less form).
add_test(NAME rho_patch_expression_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_patch_expression_vs_openfoam.sh)
set_tests_properties(rho_patch_expression_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

# FP-1 (bench/rhoSimpleFoam/FASTPATH.md): the colour-order smoothSolver on the energy field and the
# turbulence pair, the scalar counterpart of u_colour_gs_vs_openfoam. EXACT where the solves converge,
# DIFFERENT where they do not (the control), ANNOUNCED per field; the opt-out (BRAE_GS_ORDER=ofOrder)
# runs OpenFOAM's own order and passes the same bound.
add_test(NAME scalar_colour_gs_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/scalar_colour_gs_vs_openfoam.sh)
set_tests_properties(scalar_colour_gs_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)

# D-1: the CUDA arm is bit-reproducible run to run on squareBendLiq, injectorPipe and squareBend, and the
# atomic bound() scatter it replaced (BRAE_BOUND_SCATTER=1) is seen to drift. Slow: it meshes injectorPipe.
# FP-10: the captured momentum assembly must compute what the direct one computes. It holds the
# workspaces and the persistent matrix that make the capture's baked-in addresses valid.
add_test(NAME rho_capture_assembly_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_capture_assembly_identity.sh)
set_tests_properties(rho_capture_assembly_identity PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

add_test(NAME rho_run_to_run_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_run_to_run_identity.sh)
set_tests_properties(rho_run_to_run_identity PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 2400)

# T-5: a run whose residuals go non-finite must refuse rather than report a wall time for a run that
# computed nothing. Oracle: OpenFOAM aborts on the same input. Control: BRAE_ALLOW_NONFINITE=1.
add_test(NAME rho_nonfinite_refusal
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_nonfinite_refusal.sh)
set_tests_properties(rho_nonfinite_refusal PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# FP-2 policy: a relaxed `preconditioner DILU` entry on k, omega/epsilon and the energy field takes the
# Neumann series on the CUDA arm, announced, with BRAE_DILU_KE/HE=1 keeping DILU; held against OpenFOAM
# on the aerofoil at the tutorial's own relTol, with the health (bounding) and the speed asserted.
add_test(NAME rho_dilu_entry_policy_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_dilu_entry_policy_vs_openfoam.sh)
# RUN_SERIAL: ARM 3 times the turbulence phase, and a GPU shared with two other gates inflates both
# arms into noise (measured under `ctest -j 3`: 10.3 against 10.4 ms/it, where alone it reads 3.4
# against 1.9).
set_tests_properties(rho_dilu_entry_policy_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200 RUN_SERIAL TRUE)

# rhoSimpleFoam's UEqn.H against OpenFOAM's own assembled momentum matrix (tools/dumpPEqn stage harness:
# rAU, internalCoeffs, boundaryCoeffs). OpenFOAM's muEff is injected so this measures the ASSEMBLY and not
# the unported compressible turbulence closure. Carries the control that distinguishes the compressible
# divDevRhoReff (dynamic mu_eff) from the incompressible divDevReff (kinematic nu_eff).
add_executable(test_rho_ueqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_rho_ueqn_cpp.cu)
target_include_directories(test_rho_ueqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_ueqn_cpp PRIVATE brae_core)
add_test(NAME rho_ueqn_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_ueqn_vs_openfoam.sh)

# rhoSimpleFoam's EEqn.H against OpenFOAM's own assembled energy equation (dumpPEqn stage harness:
# stage_Ekp, stage_he, stage_eD, stage_eSrc, stage_alphaEff). OpenFOAM's alphaEff is injected so this
# measures the ASSEMBLY. Carries the control that distinguishes the two he.name() arms --
# Ekp = 0.5|U|^2 + p/rho for `e` against K = 0.5|U|^2 for `h`.
add_executable(test_rho_eeqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_rho_eeqn_cpp.cu)
target_include_directories(test_rho_eeqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_eeqn_cpp PRIVATE brae_core)
add_test(NAME rho_eeqn_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_eeqn_vs_openfoam.sh)

# rhoSimpleFoam's pEqn.H against OpenFOAM -- BOTH branches, by running OpenFOAM twice on the same fixture
# with `transonic no` and `transonic yes` (and `consistent no`, so the driver reaches pEqn.H rather than
# pcEqn.H). The transonic branch had no stage harness at all until this gate; one was added to
# tools/dumpPEqn. Each run's control is the OTHER branch, which must not reproduce it.
add_executable(test_rho_peqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_rho_peqn_cpp.cu)
target_include_directories(test_rho_peqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_peqn_cpp PRIVATE brae_core)
add_test(NAME rho_peqn_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_peqn_vs_openfoam.sh)

# rhoSimpleFoam's pcEqn.H -- the SIMPLEC pressure equation -- against OpenFOAM, BOTH branches, with
# `consistent yes` so the driver reaches pcEqn.H. Its subsonic branch had no stage harness until this
# gate. Controls: plain SIMPLE must not reproduce the SIMPLEC matrix, and the other transonic branch
# must not either.
add_executable(test_rho_pceqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_rho_pceqn_cpp.cu)
target_include_directories(test_rho_pceqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_pceqn_cpp PRIVATE brae_core)
add_test(NAME rho_pceqn_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_pceqn_vs_openfoam.sh)

# rhoSimpleFoam END TO END: the _cpp driver runs the whole case and is compared against real OpenFOAM.
# This is the gate PORT.md requires BEFORE any .cu is written for this solver. Laminar, because the
# compressible turbulence closure is not ported -- and the binary asserts a turbulent case is REFUSED.
add_executable(test_rho_simple_step_cpp ${PROJECT_SOURCE_DIR}/tests/test_rho_simple_step_cpp.cu)
target_include_directories(test_rho_simple_step_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_simple_step_cpp PRIVATE brae_core)
add_test(NAME rho_simple_end_to_end_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_simple_end_to_end_vs_openfoam.sh)

# The COMPRESSIBLE kEpsilon closure against OpenFOAM's own turbulence->correct(), which tools/dumpPEqn
# brackets so every input it reads and every field it writes can be compared. Carries the control that
# distinguishes the two fluxes: divU is the VOLUMETRIC one, the div operator takes the MASS one.
# The COMPRESSIBLE kOmegaSST closure against OpenFOAM's OWN kOmegaSST, instrumented. tools/dumpKOmegaSST
# is kOmegaSSTBase.C with writes added and its equations untouched, run under tools/dumpPEqn so the
# model's inputs come from the same iteration. Carries the divU flux control: the compressible reading
# takes the VOLUMETRIC flux there and the MASS flux in fvm::div, and substituting one for the other must
# be measurably worse.
add_executable(test_rho_komegasst_cpp ${PROJECT_SOURCE_DIR}/tests/test_rho_komegasst_cpp.cu)
target_include_directories(test_rho_komegasst_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_komegasst_cpp PRIVATE brae_core)
# squareBend: OpenFOAM's OWN tutorial, UNMODIFIED, through the _cpp driver. Every other rhoSimpleFoam
# gate runs validation/sbMatched, which is derived from this case -- so this is the one that measures
# whether the port generalises past the fixture it was developed on. Compared at OpenFOAM's own
# residualControl convergence (156 iterations), not at the shipped endTime.
add_test(NAME rho_squarebend_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_squarebend_vs_openfoam.sh)
set_tests_properties(rho_squarebend_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cpp
                     LABELS slow
                     TIMEOUT 1800)

add_test(NAME rho_komegasst_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_komegasst_vs_openfoam.sh)
set_tests_properties(rho_komegasst_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_komegasst_cpp
                     TIMEOUT 1800)

add_executable(test_rho_kepsilon_cpp ${PROJECT_SOURCE_DIR}/tests/test_rho_kepsilon_cpp.cu)
target_include_directories(test_rho_kepsilon_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_kepsilon_cpp PRIVATE brae_core)
# REGISTERED NOW: the closure agrees with OpenFOAM at machine precision -- epsilon 5.0e-15, k 8.9e-16,
# nut 1.7e-15, wall and interior alike -- and so does every intermediate the gate can see, because
# tools/dumpKEpsilon is OpenFOAM's own kEpsilon with writes added and its equations untouched. gradU,
# GbyNu, G, divU, both diffusivities, the mesh factors, both off-diagonal sets and both assembled systems
# each have an oracle rather than an argument. What that instrument found, in order:
#   - the closure was never given the flux, so U's inletOutlet outlet evaluated on a seeded valueFraction
#     and fvc::grad(U) read the wrong face value there (gradU exact in the interior, 1.5e-01 at the
#     boundary -- the split is what named it);
#   - k and epsilon never had updateCoeffs called on them either, so every flux-conditional patch
#     contributed nothing to its system whatever value it carried;
#   - the turbulent inlets were frozen at the case file's `value`. OpenFOAM RECOMPUTES refValue on every
#     updateCoeffs, epsilon's from k's patch values and k's from U's;
#   - the diffusion terms were assembled with `Gauss linear orthogonal` while the case asked for
#     `corrected` -- a silent scheme substitution, and the last 5e-06.
add_test(NAME rho_kepsilon_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_kepsilon_vs_openfoam.sh)
set_tests_properties(rho_simple_end_to_end_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cpp
                     TIMEOUT 1800)

# div(phi,k)/div(phi,epsilon) `Gauss limitedLinear 1` end to end: sbMatched mutated to the scheme
# validation/squareBend actually ships, BOTH codes run under the mutation, so the oracle converges under
# the scheme being gated. Separate from the e2e gate because it pays its own 400-iteration OpenFOAM run.
# The fail-proof arm reverts only the scheme word and re-runs brae against the same limitedLinear
# oracle: upwind must miss the k/epsilon bound, which is also the non-vacuity statement.
add_test(NAME rho_turb_limitedlinear_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_turb_limitedlinear_vs_openfoam.sh)
set_tests_properties(rho_turb_limitedlinear_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cpp
                     TIMEOUT 2400)

# The non-orthogonal corrector loop (`while (simple.correctNonOrthogonal())`), which the _cpp driver
# used to skip entirely -- one pressure solve whatever the case named. Compared at a matched EARLY
# iteration because the count is a trajectory feature: at convergence every extra pass re-solves an
# unchanged system and no converged gate can see it. The fail-proof replays the old defect (one pass
# against the 2-corrector oracle) and reads p 1.51e-01 where the arm reads 3.97e-04. Also carries the
# DEVICE arm: the CUDA driver's loop vs the OF-gated host at iteration 1 on the same mutation (its
# harness used to hardcode correctedLaplacian=false, under which the loop had nothing to correct);
# forcing the device count to 0 reads Ux 1.85e-01 against the arm's 5.7e-12.
# THE MIRROR AS A SOLVER, not as a step: `brae -case` end to end against real OpenFOAM, on BOTH arms --
# the answer, the OF-format output (real rhoSimpleFoam restarts from brae's written directory), the
# latestTime restart with an ABSOLUTE endTime, the turbulent write payload, the residualControl stop,
# and the binary-writeFormat and unknown-selector refusals. Measured on rhoBox at 200 iterations: the
# host arm p 2.95e-12 / U 2.93e-11 against OpenFOAM, the CUDA arm 3.00e-12 / 4.45e-10 against OpenFOAM
# and 2.83e-12 / 4.62e-10 against the host arm -- the two differ by their linear solvers (AMG-PCG
# against BiCGStab) and by nothing else. Every other rhoSimpleFoam gate drives the step through a
# harness that reads no controlDict and writes nothing, so none of this was gated anywhere.
# `brae -partition` + the warm reload it promises. The AMG cache serialises the agglomeration and
# re-Galerkins the values, but its LOAD path never rebuilt the per-level Galerkin gather lists -- so
# every warm run died in galDiagGatherK ("amul: an illegal memory access"). Nothing was gated on a
# warm reload, on either the compressible or the incompressible path, so nothing caught it. Asserts
# EQUALITY, not a tolerance: a cached hierarchy must reproduce a cold one exactly.
# simpleFoamV2 wrote every field's boundaryField by ECHOING the 0/ template it splices into -- so a V2
# time directory carried the start state on every patch the solve had moved (pitzDailyTurb's wall nut
# among them), and a restart from it restarted from the wrong state. Asserts the written boundary
# DIFFERS from the template (a value list alone proves nothing -- the template has one too) and that
# OpenFOAM still accepts the file.
add_test(NAME v2_written_boundary
         COMMAND ${PROJECT_SOURCE_DIR}/tests/v2_written_boundary.sh)
set_tests_properties(v2_written_boundary PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1200)

add_test(NAME amg_cache_warm_start
         COMMAND ${PROJECT_SOURCE_DIR}/tests/amg_cache_warm_start.sh)
set_tests_properties(amg_cache_warm_start PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1200)

# angledDuctExplicitFixedCoeff, OpenFOAM's own tutorial, at OpenFOAM's own convergence (639 iterations)
# -- porosity + a fixedTemperatureConstraint + a scalarFixedValueConstraint on k and epsilon. It was
# never registered, so nothing ran it; its host comparison reads p 1.9e-06, T 1.6e-06, U 1.7e-05. The
# CUDA arms check that the tutorial's TILTED SLIP wall is refused by name (mesh type `wall`, U BC
# `slip`, plane at 45 degrees) and that with an axis-independent wall the device applies all three
# fvOptions and pins the same cells as the host.
add_test(NAME rho_angledduct_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_angledduct_vs_openfoam.sh)
set_tests_properties(rho_angledduct_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     LABELS slow
                     TIMEOUT 3600)

add_test(NAME rho_mirror_solver_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_mirror_solver_vs_openfoam.sh)
set_tests_properties(rho_mirror_solver_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_relax_default_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_relax_default_vs_openfoam.sh)
set_tests_properties(rho_relax_default_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_uresidual_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_uresidual_vs_openfoam.sh)
set_tests_properties(rho_uresidual_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_tp_transient_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_tp_transient_vs_openfoam.sh)
set_tests_properties(rho_tp_transient_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_turbinlet_cmu_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_turbinlet_cmu_vs_openfoam.sh)
set_tests_properties(rho_turbinlet_cmu_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_coeffdict_fallback_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_coeffdict_fallback_vs_openfoam.sh)
set_tests_properties(rho_coeffdict_fallback_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_p0_written_table_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_p0_written_table_vs_openfoam.sh)
set_tests_properties(rho_p0_written_table_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_closure_gradlimit_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_closure_gradlimit_vs_openfoam.sh)
set_tests_properties(rho_closure_gradlimit_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_p_unlimited_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_p_unlimited_vs_openfoam.sh)
set_tests_properties(rho_p_unlimited_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)
add_test(NAME rho_sbmatched_transient_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_sbmatched_transient_vs_openfoam.sh)
set_tests_properties(rho_sbmatched_transient_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_step_cuda_schemes
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_step_cuda_schemes.sh)
set_tests_properties(rho_step_cuda_schemes PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cuda
                     TIMEOUT 1800)

add_test(NAME rho_step_cuda_turbulent
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_step_cuda_turbulent.sh)
set_tests_properties(rho_step_cuda_turbulent PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cuda
                     TIMEOUT 1800)
add_test(NAME rho_step_cuda_euler
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_step_cuda_euler.sh)
set_tests_properties(rho_step_cuda_euler PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cuda
                     TIMEOUT 1800)
add_test(NAME rho_step_cuda_lsq
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_step_cuda_lsq.sh)
set_tests_properties(rho_step_cuda_lsq PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cuda
                     TIMEOUT 1800)
add_test(NAME rho_naca_restart_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_naca_restart_vs_openfoam.sh)
set_tests_properties(rho_naca_restart_vs_openfoam PROPERTIES
                     ENVIRONMENT "BRAE_BIN=${PROJECT_BINARY_DIR}/brae_rhoSimpleFoam"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     LABELS slow
                     TIMEOUT 2400)
add_test(NAME rho_gradp_lsq_simplec_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_gradp_lsq_simplec_vs_openfoam.sh)
set_tests_properties(rho_gradp_lsq_simplec_vs_openfoam PROPERTIES
                     ENVIRONMENT "BRAE_BIN=${PROJECT_BINARY_DIR}/brae_rhoSimpleFoam"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_nonorth_corrector_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_nonorth_corrector_vs_openfoam.sh)
set_tests_properties(rho_nonorth_corrector_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cpp
                     TIMEOUT 900)
set_tests_properties(rho_kepsilon_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_kepsilon_cpp
                     TIMEOUT 1800)
set_tests_properties(rho_pceqn_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_pceqn_cpp
                     TIMEOUT 900)
set_tests_properties(rho_peqn_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_peqn_cpp
                     TIMEOUT 900)
set_tests_properties(rho_eeqn_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_eeqn_cpp
                     TIMEOUT 900)
set_tests_properties(rho_ueqn_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_ueqn_cpp
                     TIMEOUT 900)
set_tests_properties(rho_createfields_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_create_fields_cpp
                     TIMEOUT 900)

# heRhoThermo/perfectGas must parse identically to hePsiThermo, and heRhoThermo with any OTHER
# equationOfState must be refused. Asserted at the PARSE level because brae is not run-to-run
# bit-reproducible (~3e-9 on T after one iteration), so a solver-level equivalence check would need a
# statistical argument; the claim itself is deterministic.
add_executable(test_thermo_type_equiv ${PROJECT_SOURCE_DIR}/tests/test_thermo_type_equiv.cu)
target_include_directories(test_thermo_type_equiv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_thermo_type_equiv PRIVATE brae_core)
add_test(NAME thermo_type_equiv COMMAND test_thermo_type_equiv)

# D2: boundary coefficients (internalCoeffs/boundaryCoeffs) vs OF across four REAL BC types at once.
add_test(NAME bc_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/validation/bc_vs_openfoam.sh)
set_tests_properties(bc_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS bcoeff_compare
                     TIMEOUT 600)

# E6: the universal gas constant must be RESOLVED like OF (DimensionedConstants NA*k), not compiled in.
add_executable(test_sensible_energy ${PROJECT_SOURCE_DIR}/tests/test_sensible_energy.cu)
target_include_directories(test_sensible_energy PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_sensible_energy PRIVATE brae_core)
add_test(NAME sensible_energy COMMAND test_sensible_energy)

add_executable(bcoeff_compare ${PROJECT_SOURCE_DIR}/tests/bcoeff_compare.cu)
target_include_directories(bcoeff_compare PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(bcoeff_compare PRIVATE brae_core)

# Q1 (Wave 1): rhoSimpleFoam's DEFAULT driver (`brae -case`, no BRAE_RHOSIMPLEFOAM_MIRROR) stops running
# what it computed differently from OpenFOAM, stops on divergence, and writes a restartable state.
# Refusals by name (frozen per-step BCs, `expression`, rotorDisk, fvOptions windows, Maxwell, alphat types,
# a missing alphat, Switch words on both drivers, energy maxIter/minIter, the pressure reference, deltaT and
# endTime, mandatory thermo coefficients), each arm under `brae check --setup`, plus the Switch
# equivalences run three iterations on both drivers. Fixtures with their mesh in the repository; no OpenFOAM.
add_test(NAME rho_default_refusals COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_default_refusals.sh)
set_tests_properties(rho_default_refusals PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)
# The energy equation relaxed only by its own name, a `default` or a matching pattern, as OpenFOAM: rhoBox at
# t=1..5 against OpenFOAM, five relaxationFactors forms, every linear solver pinned.
# Both codes run U, h and e with the solver brae's default driver runs as OpenFOAM does, so the comparison isolates
# the relaxation...
add_test(NAME rho_energy_relax_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_energy_relax_vs_openfoam.sh)
set_tests_properties(rho_energy_relax_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "SOLVERS=matched")
# ...and the same arms as the fixture ships them (DILU asked, a diagonal preconditioner run): the default driver as
# a user meets it, and the preconditioner difference, held to the same bounds.
add_test(NAME rho_default_energy_shipped_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_energy_relax_vs_openfoam.sh)
set_tests_properties(rho_default_energy_shipped_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "SOLVERS=shipped")
# pRefPoint located as OpenFOAM's findCell (rhoBoxP with a reference) and a non-integer endTime, against OpenFOAM.
add_test(NAME rho_default_controls_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_default_controls_vs_openfoam.sh)
set_tests_properties(rho_default_controls_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 600)
# alphat written and read back, writePrecision honoured: brae <-> OpenFOAM restarts on naca0012, the wall-Prt
# control on rhoKE; issue #54's own-restart drift reported beside them.
add_test(NAME rho_default_restart_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_default_restart_vs_openfoam.sh)
set_tests_properties(rho_default_restart_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)
# The divergence stop: squareBend subsonic (OpenFOAM aborts), its BRAE_ALLOW_NONFINITE control, the residual
# and the write-time stops shown on rhoBox with BRAE_TEST_CORRUPT, and the healthy tutorial as fail-proof.
add_test(NAME rho_default_nonfinite_refusal COMMAND ${PROJECT_SOURCE_DIR}/tests/rho_default_nonfinite_refusal.sh)
set_tests_properties(rho_default_nonfinite_refusal PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)
