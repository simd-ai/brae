# interFoam. The two-phase VoF solver (`brae_interFoam`): units, the device step, dynamic meshes (motion, refinement,
# rigid bodies), and the per-file gates under tests/interfoam_*/ that CMake collects by folder.
#
# Included by tests/CMakeLists.txt.

# interFoam step 3: the two-phase mixture. Reads the case's own transportProperties (phase ORDER decides
# which phase alpha1 is) and blends rho/mu/nu, where rho takes the RAW alpha and mu/nu the clamped one --
# a difference invisible on any in-range field, so the gate sweeps alpha outside [0,1] and carries a
# control that fails if rho is clamped.
add_executable(test_two_phase_mixture ${PROJECT_SOURCE_DIR}/tests/test_two_phase_mixture.cu)
target_include_directories(test_two_phase_mixture PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/src/transportModels/twoPhaseMixture)
target_link_libraries(test_two_phase_mixture PRIVATE brae_core)
# interFoam on the DEVICE, first component: the mixture. Held to BIT-FOR-BIT agreement with the host
# reference rather than to a tolerance -- it is per-cell arithmetic with no addressing, reduction or
# interpolation, so there is nothing for a tolerance to absorb and any difference is a defect. The
# sweep leaves [0,1] on purpose: rho takes the raw alpha and mu/nu the clamped one, and a fused kernel
# that clamps once passes every in-range test.
add_executable(test_device_two_phase_mixture ${PROJECT_SOURCE_DIR}/tests/test_device_two_phase_mixture.cu)
target_include_directories(test_device_two_phase_mixture PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_device_two_phase_mixture PRIVATE brae_core)
add_test(NAME device_two_phase_mixture COMMAND test_device_two_phase_mixture)

# ...and the interface normal and curvature. Unlike the mixture this CANNOT be bit-for-bit: K is a
# reduction and the device sums a cell's faces in a different order from the host's serial loop, so a
# difference is guaranteed and only its size is in question. The per-face half has no reduction and is
# held to 4 ULP; K to 1e-12 relative, which is four orders below the curvature it measures.
add_executable(test_device_interface_properties ${PROJECT_SOURCE_DIR}/tests/test_device_interface_properties.cu)
target_include_directories(test_device_interface_properties PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_interface_properties PRIVATE brae_core)
add_test(NAME device_interface_properties COMMAND test_device_interface_properties)

# ...and MULES. Two kinds of arm: the PROPERTY (alpha stays in [0,1]) asserted exactly, because that is
# what MULES is for and needs no host to compare against; and lambda itself within a measured
# tolerance, because each cell's five gathers run in a different ORDER from the host's face loop and
# floating-point addition is not associative. The donor flux has no reduction and IS bit-identical.
add_executable(test_device_mules ${PROJECT_SOURCE_DIR}/tests/test_device_mules.cu)
target_include_directories(test_device_mules PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_mules PRIVATE brae_core)
add_test(NAME device_mules COMMAND test_device_mules)

# ...and the alpha fluxes. The arm worth having is alphaPhiUn: five face operations with two minus
# signs in the middle, where the upwind direction of each interpolation is set by the flux passed in --
# so the negations change WHICH CELL each face reads, not just a sign.
add_executable(test_device_alpha_flux ${PROJECT_SOURCE_DIR}/tests/test_device_alpha_flux.cu)
target_include_directories(test_device_alpha_flux PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_alpha_flux PRIVATE brae_core)
add_test(NAME device_alpha_flux COMMAND test_device_alpha_flux)
set_tests_properties(device_alpha_flux PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_inter_dambreak_alpha ${PROJECT_SOURCE_DIR}/tests/test_device_inter_dambreak_alpha.cu)
target_include_directories(test_device_inter_dambreak_alpha PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_inter_dambreak_alpha PRIVATE brae_core)

add_executable(test_device_inter_step ${PROJECT_SOURCE_DIR}/tests/test_device_inter_step.cu)
target_include_directories(test_device_inter_step PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_inter_step PRIVATE brae_core)
add_test(NAME interfoam_dambreak_device_vs_host
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_dambreak_device_vs_host.sh)
set_tests_properties(interfoam_dambreak_device_vs_host PROPERTIES
                     SKIP_RETURN_CODE 77
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_dependencies(brae test_device_inter_dambreak_alpha)

# What brae's interFoam REFUSES and what it must not: one unported input at a time on damBreak, each beside
# the form of it OpenFOAM treats as nothing. Found by running brae over all 44 tutorials and counting
# the ones that reached End: -- two did, on a dynamicRefineFvMesh brae never opened.
add_test(NAME interfoam_refusals
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_refusals.sh)
set_tests_properties(interfoam_refusals PROPERTIES
                     SKIP_RETURN_CODE 77
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

# The ADAPTIVE CLOCK against real OpenFOAM, on damBreak exactly as it ships -- the one interFoam gate
# that does not pin deltaT, and so the only one that can see setDeltaT or Time::adjustDeltaT.
add_test(NAME interfoam_dambreak_clock_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_dambreak_clock_vs_openfoam.sh)
set_tests_properties(interfoam_dambreak_clock_vs_openfoam PROPERTIES
                     SKIP_RETURN_CODE 77
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

# brae_interFoam's time directories against OpenFOAM's: when it writes, what -- every file's structure
# exactly, every value at a measured bound -- that OpenFOAM restarts from them, and that writing every
# step leaves the run byte-identical.
#
# ONE TEST PER FILE under tests/interfoam_write/<arm>/ and one per tutorial row, not one script: each file
# holds at most three checks, a change is checked by the file it touches, the files run side by side under
# `ctest -j`, and real OpenFOAM's meshes and runs are cached by a hash of the staged case
# (tests/of_oracle_cache.sh), so editing brae re-runs brae alone. As one script this gate took 3100 s on its
# own and 5700 s beside five others (2026-10-01), every time.
#
# TWO TIERS, by label. `fast` is the edit loop: `ctest -L fast -R interfoam_write`. `slow` holds the
# tutorials whose meshes are large (DTCHull and DTCHullMoving 848k cells, mixerVesselAMI 895k, motorBike,
# the two multi-paddle wave makers 448k) and the files that run on them; it runs before a commit. On those
# tutorials the host arm and the device arm are separate tests, and a test that runs a device arm on a
# large mesh takes the GPU alone (RESOURCE_LOCK): five processes sharing one GPU each ran four times slower.
file(GLOB BRAE_WRITE_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_write
     ${PROJECT_SOURCE_DIR}/tests/interfoam_write/*/*.sh)
list(FILTER BRAE_WRITE_TESTS EXCLUDE REGEX "/_common\\.sh$")
set(BRAE_WRITE_SLOW_FILES
    "^ami/|^refine/mapping|^amg_pcg/dtchull|^amg_pcg/coupled_pair|^amg_pcg/pcorr_across_pair|^identity/wave_device\
|^identity/nhat_least_squares")
foreach(rel ${BRAE_WRITE_TESTS})
    string(REPLACE ".sh" "" name ${rel})
    string(REPLACE "/" "_" name ${name})
    set(tierLabel fast)
    if(rel MATCHES "${BRAE_WRITE_SLOW_FILES}")
        set(tierLabel slow)
    endif()
    add_test(NAME interfoam_write_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_write/${rel})
    set_tests_properties(interfoam_write_${name} PROPERTIES
                         SKIP_RETURN_CODE 77
                         LABELS ${tierLabel}
                         TIMEOUT 1800
                         ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
endforeach()
# THE TWO FORTY-STEP GATES take a minute or two by themselves and 1,321 s and 1,312 s in a full suite
# (2026-10-08, -j 6): their device arms share the GPU with every small gate beside them. A limit for a hang,
# clear of that.
set_tests_properties(interfoam_write_subcycle_final_mixture_no_evaluate
                     interfoam_write_subcycle_contact_angle_next_corrector PROPERTIES TIMEOUT 3600)
set(BRAE_WRITE_TUTORIALS_FAST
    capillaryRise weirOverflow angledDuct damBreakLeakage damBreakPorousBaffle damBreakPermeable
    nozzleFlow2D eulerianInjection cnoidal solitary solitaryGrimshaw solitaryMcCowan stokesI stokesII
    stokesV irregularMultiDirection streamFunction mangroveInteraction sloshingTank2D testTubeMixer
    waveMakerSolitary waveMakerPiston waveMakerFlap damBreakWithObstacle oscillatingBox floatingObject
    DTCHullMovingCoarse sloshingTank2D3DoF sloshingTank3D sloshingTank3D3DoF sloshingTank3D6DoF)
set(BRAE_WRITE_TUTORIALS_SLOW
    DTCHull DTCHullMoving mixerVesselAMI motorBike sloshingCylinder electrostaticDeposition
    waveMakerMultiPaddleFlap waveMakerMultiPaddlePiston)
foreach(key ${BRAE_WRITE_TUTORIALS_FAST})
    add_test(NAME interfoam_write_tutorial_${key}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_write/tutorial.sh ${key})
    set_tests_properties(interfoam_write_tutorial_${key} PROPERTIES
                         SKIP_RETURN_CODE 77
                         LABELS fast
                         TIMEOUT 1800
                         ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
endforeach()
foreach(key ${BRAE_WRITE_TUTORIALS_SLOW})
    foreach(arm host device)
        add_test(NAME interfoam_write_tutorial_${key}_${arm}
                 COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_write/tutorial.sh ${key} ${arm})
        set_tests_properties(interfoam_write_tutorial_${key}_${arm} PROPERTIES
                             SKIP_RETURN_CODE 77
                             LABELS slow
                             TIMEOUT 1800
                             ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
    endforeach()
    # ONE TURN FOR EVERY TEST ON THESE MESHES, host or device. A device run on 848k cells loads the CPU and
    # the memory bus as well as the GPU: beside two of them the DTCHullMoving host row took 426 s and the
    # device row 313 s, where each alone takes 289 s and 290 s (2026-10-01).
    set_tests_properties(interfoam_write_tutorial_${key}_device PROPERTIES RESOURCE_LOCK "brae_gpu;brae_bigmesh")
    set_tests_properties(interfoam_write_tutorial_${key}_host PROPERTIES RESOURCE_LOCK brae_bigmesh)
endforeach()
set_tests_properties(interfoam_write_ami_operands_control PROPERTIES RESOURCE_LOCK brae_bigmesh)
set_tests_properties(interfoam_write_ami_pair_sums_ordered PROPERTIES RESOURCE_LOCK "brae_gpu;brae_bigmesh")
set_tests_properties(interfoam_write_ami_bicg_graph_pair PROPERTIES RESOURCE_LOCK "brae_gpu;brae_bigmesh")
add_dependencies(brae brae_interFoam)

add_test(NAME device_inter_step COMMAND test_device_inter_step)
set_tests_properties(device_inter_step PROPERTIES SKIP_RETURN_CODE 77)

# the device laplacian's kernels and fvc::div against the host's, bit for bit, on a sheared mesh, with the
# owner-first sums the device used to form as controls
add_executable(test_device_laplacian_vs_host ${PROJECT_SOURCE_DIR}/tests/test_device_laplacian_vs_host.cu)
target_include_directories(test_device_laplacian_vs_host PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_laplacian_vs_host PRIVATE brae_core)
add_test(NAME device_laplacian_vs_host COMMAND test_device_laplacian_vs_host)
set_tests_properties(device_laplacian_vs_host PROPERTIES SKIP_RETURN_CODE 77)

# the device's CYCLIC-interface laplacian coefficients against the host's fvm::laplacian, the first
# module of interFoam's cyclic path (it depends on nothing else in that loop)
add_executable(test_device_cyclic_laplacian_vs_host ${PROJECT_SOURCE_DIR}/tests/test_device_cyclic_laplacian_vs_host.cu)
target_include_directories(test_device_cyclic_laplacian_vs_host PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_cyclic_laplacian_vs_host PRIVATE brae_core)
add_test(NAME device_cyclic_laplacian_vs_host
         COMMAND test_device_cyclic_laplacian_vs_host
                 ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)
set_tests_properties(device_cyclic_laplacian_vs_host PROPERTIES SKIP_RETURN_CODE 77)
# ...and again on a SKEWED periodic pair, where nonOrthDeltaCoeffs differs from deltaCoeffs, so the
# `corrected` and orthogonal passes are two different matrices. cyclicChannel cannot tell them apart.
add_test(NAME device_cyclic_laplacian_vs_host_skew
         COMMAND test_device_cyclic_laplacian_vs_host
                 ${PROJECT_SOURCE_DIR}/validation/cyclicChannelSkew)
set_tests_properties(device_cyclic_laplacian_vs_host_skew PROPERTIES SKIP_RETURN_CODE 77)

# MULES's limiter across a cyclic pair: the device's lambda against the host's
add_executable(test_device_mules_cyclic_vs_host ${PROJECT_SOURCE_DIR}/tests/test_device_mules_cyclic_vs_host.cu)
target_include_directories(test_device_mules_cyclic_vs_host PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_mules_cyclic_vs_host PRIVATE brae_core)
add_test(NAME device_mules_cyclic_vs_host
         COMMAND test_device_mules_cyclic_vs_host
                 ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)
set_tests_properties(device_mules_cyclic_vs_host PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME device_mules_cyclic_vs_host_skew
         COMMAND test_device_mules_cyclic_vs_host
                 ${PROJECT_SOURCE_DIR}/validation/cyclicChannelSkew)
set_tests_properties(device_mules_cyclic_vs_host_skew PROPERTIES SKIP_RETURN_CODE 77)

# interFoam's device pressure operator across a cyclic pair: the assembled matrix against the host's
add_executable(test_device_inter_peqn_cyclic_vs_host ${PROJECT_SOURCE_DIR}/tests/test_device_inter_peqn_cyclic_vs_host.cu)
target_include_directories(test_device_inter_peqn_cyclic_vs_host PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_inter_peqn_cyclic_vs_host PRIVATE brae_core)
add_test(NAME device_inter_peqn_cyclic_vs_host
         COMMAND test_device_inter_peqn_cyclic_vs_host
                 ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)
set_tests_properties(device_inter_peqn_cyclic_vs_host PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME device_inter_peqn_cyclic_vs_host_skew
         COMMAND test_device_inter_peqn_cyclic_vs_host
                 ${PROJECT_SOURCE_DIR}/validation/cyclicChannelSkew)
set_tests_properties(device_inter_peqn_cyclic_vs_host_skew PROPERTIES SKIP_RETURN_CODE 77)

# the device's nHatf on a cyclic pair against REAL OpenFOAM's own written field
add_executable(test_device_cyclic_nhatf_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_device_cyclic_nhatf_vs_openfoam.cu)
target_include_directories(test_device_cyclic_nhatf_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_cyclic_nhatf_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_cyclic_nhatf_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_cyclic_nhatf_vs_openfoam.sh)
set_tests_properties(interfoam_cyclic_nhatf_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}")

# the device's epsilon at a wall-function-constrained cell that also touches a cyclic pair, against
# REAL OpenFOAM's own epsilon after one step with every solve pinned
add_executable(test_device_wallfn_cyclic_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_device_wallfn_cyclic_vs_openfoam.cu)
target_include_directories(test_device_wallfn_cyclic_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_wallfn_cyclic_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_wallfn_cyclic_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_wallfn_cyclic_vs_openfoam.sh)
set_tests_properties(interfoam_wallfn_cyclic_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}")

# the device Gauss gradient against the host's fvc::gaussGrad, bit for bit, with the two forms the device
# used to have as controls
add_executable(test_device_gauss_grad ${PROJECT_SOURCE_DIR}/tests/test_device_gauss_grad.cu)
target_include_directories(test_device_gauss_grad PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_gauss_grad PRIVATE brae_core)
add_test(NAME device_gauss_grad COMMAND test_device_gauss_grad)
set_tests_properties(device_gauss_grad PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_inter_pressure_step ${PROJECT_SOURCE_DIR}/tests/test_device_inter_pressure_step.cu)
target_include_directories(test_device_inter_pressure_step PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_inter_pressure_step PRIVATE brae_core)
add_test(NAME device_inter_pressure_step COMMAND test_device_inter_pressure_step)
set_tests_properties(device_inter_pressure_step PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_inter_ueqn_assembly ${PROJECT_SOURCE_DIR}/tests/test_device_inter_ueqn_assembly.cu)
target_include_directories(test_device_inter_ueqn_assembly PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_inter_ueqn_assembly PRIVATE brae_core)
add_test(NAME device_inter_ueqn_assembly COMMAND test_device_inter_ueqn_assembly)
set_tests_properties(device_inter_ueqn_assembly PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_alpha_courant ${PROJECT_SOURCE_DIR}/tests/test_device_alpha_courant.cu)
target_include_directories(test_device_alpha_courant PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_alpha_courant PRIVATE brae_core)
add_test(NAME device_alpha_courant COMMAND test_device_alpha_courant)
set_tests_properties(device_alpha_courant PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_inter_alpha_step ${PROJECT_SOURCE_DIR}/tests/test_device_inter_alpha_step.cu)
target_include_directories(test_device_inter_alpha_step PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_inter_alpha_step PRIVATE brae_core)
add_test(NAME device_inter_alpha_step COMMAND test_device_inter_alpha_step)
set_tests_properties(device_inter_alpha_step PROPERTIES SKIP_RETURN_CODE 77)

# The evaluate that opens the limited explicit solve, and the bounded flux's boundary after it: the device's
# alpha step against the host's on an inletOutlet side whose flux reverses every step. The two controls are the
# code's own switches, run by the test itself as children.
add_executable(test_device_alpha_opening_evaluate ${PROJECT_SOURCE_DIR}/tests/test_device_alpha_opening_evaluate.cu)
target_include_directories(test_device_alpha_opening_evaluate PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_alpha_opening_evaluate PRIVATE brae_core)
add_test(NAME device_alpha_opening_evaluate COMMAND test_device_alpha_opening_evaluate)
set_tests_properties(device_alpha_opening_evaluate PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_inter_peqn ${PROJECT_SOURCE_DIR}/tests/test_device_inter_peqn.cu)
target_include_directories(test_device_inter_peqn PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_inter_peqn PRIVATE brae_core)
add_test(NAME device_inter_peqn COMMAND test_device_inter_peqn)
set_tests_properties(device_inter_peqn PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_inter_ueqn ${PROJECT_SOURCE_DIR}/tests/test_device_inter_ueqn.cu)
target_include_directories(test_device_inter_ueqn PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_inter_ueqn PRIVATE brae_core)
add_test(NAME device_inter_ueqn COMMAND test_device_inter_ueqn)
set_tests_properties(device_inter_ueqn PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_alpha_subcycle ${PROJECT_SOURCE_DIR}/tests/test_device_alpha_subcycle.cu)
target_include_directories(test_device_alpha_subcycle PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_alpha_subcycle PRIVATE brae_core)
add_test(NAME device_alpha_subcycle COMMAND test_device_alpha_subcycle)
set_tests_properties(device_alpha_subcycle PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_alpha_presolve ${PROJECT_SOURCE_DIR}/tests/test_device_alpha_presolve.cu)
target_include_directories(test_device_alpha_presolve PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_alpha_presolve PRIVATE brae_core)
add_test(NAME device_alpha_presolve COMMAND test_device_alpha_presolve)
set_tests_properties(device_alpha_presolve PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_mules_corr ${PROJECT_SOURCE_DIR}/tests/test_device_mules_corr.cu)
target_include_directories(test_device_mules_corr PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_mules_corr PRIVATE brae_core)
add_test(NAME device_mules_corr COMMAND test_device_mules_corr)
set_tests_properties(device_mules_corr PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_alpha_step ${PROJECT_SOURCE_DIR}/tests/test_device_alpha_step.cu)
target_include_directories(test_device_alpha_step PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_alpha_step PRIVATE brae_core)
add_test(NAME device_alpha_step COMMAND test_device_alpha_step)
set_tests_properties(device_alpha_step PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(device_mules PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(device_interface_properties PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(device_two_phase_mixture PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME two_phase_mixture
         COMMAND test_two_phase_mixture
                 ${BRAE_OF_TUTORIALS}/multiphase/interFoam/laminar/damBreak/damBreak)

# interFoam step 4: interface curvature and surface tension. The dictionary half is the part that bites --
# damBreak keys its alpha solver as the REGEX "alpha.water.*", cAlpha has NO default in OpenFOAM, and
# deltaN is 1e-8/cbrt(average(V)) rather than a constant.
add_executable(test_interface_properties ${PROJECT_SOURCE_DIR}/tests/test_interface_properties.cu)
target_include_directories(test_interface_properties PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/src/transportModels/interfaceProperties
                           ${PROJECT_SOURCE_DIR}/src/transportModels/twoPhaseMixture)
target_link_libraries(test_interface_properties PRIVATE brae_core)
add_test(NAME interface_properties
         COMMAND test_interface_properties
                 ${BRAE_OF_TUTORIALS}/multiphase/interFoam/laminar/damBreak/damBreak)

# interFoam step 5: the gravity fields. ghRef carries g's SIGN (gh.H:2-7), and NO shipped tutorial sets
# constant/hRef -- so on every stock case ghRef is 0 and the sign is unobservable. The gate supplies its
# own hRef, which is the only way that arm can discriminate.
add_executable(test_inter_create_fields ${PROJECT_SOURCE_DIR}/tests/test_inter_create_fields.cu)
target_include_directories(test_inter_create_fields PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/src/applications/solvers/interFoam)
target_link_libraries(test_inter_create_fields PRIVATE brae_core)
add_test(NAME inter_create_fields
         COMMAND test_inter_create_fields
                 ${BRAE_OF_TUTORIALS}/multiphase/interFoam/laminar/damBreak/damBreak)

# interFoam step 6: fvc::reconstruct, which brae did not have and interFoam needs twice (the momentum
# predictor's source and rebuilding U after the pressure solve). Held against its defining identity --
# reconstruct(V & Sf) == V -- which no fixture is needed for and which both plausible mistakes break.
add_executable(test_fvc_reconstruct ${PROJECT_SOURCE_DIR}/tests/test_fvc_reconstruct.cu)
target_include_directories(test_fvc_reconstruct PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvc_reconstruct PRIVATE brae_core)
add_test(NAME fvc_reconstruct COMMAND test_fvc_reconstruct)

# interFoam step 7: UEqn.H. Assembly from pieces brae already had, plus the two that are new to this
# solver -- fvm::ddt(rho, U) with rho.oldTime() in the SOURCE and rho in the DIAGONAL (they differ by the
# water/air ratio in every cell the interface crossed), and a momentum source that is a reconstructed
# FACE flux rather than a cell gradient. Arms 1 and 3 carry the controls that separate each from its
# plausible substitute; neither difference is visible on a smooth field.
add_executable(test_inter_ueqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_inter_ueqn_cpp.cu)
target_include_directories(test_inter_ueqn_cpp PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_ueqn_cpp PRIVATE brae_core)
add_test(NAME inter_ueqn_cpp COMMAND test_inter_ueqn_cpp)

# interFoam step 8: alphaEqn.H's flux assembly -- the controls, the off-centring, phic, alphaPhiUn and
# rhoPhi. MULES itself is the next stage and is deliberately not here. The arms that matter are the ones
# no field plot shows: phic zeroed on non-coupled boundaries, the two minus signs in the nested
# compressive flux, and vanLeer's limiter reaching 2 rather than being clamped to 1.
add_executable(test_alpha_eqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_alpha_eqn_cpp.cu)
target_include_directories(test_alpha_eqn_cpp PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_alpha_eqn_cpp PRIVATE brae_core)
add_test(NAME alpha_eqn_cpp
         COMMAND test_alpha_eqn_cpp
                 ${BRAE_OF_TUTORIALS}/multiphase/interFoam/laminar/damBreak/damBreak)

# interFoam step 9: MULES. The oracle is BOUNDEDNESS -- an assertion, not a tolerance -- and it needs no
# instrumented OpenFOAM, which matters because a stock run never writes the limiter. A boundedness gate
# alone would pass with lambda == 0 everywhere (i.e. first-order upwind, the case's scheme discarded),
# so the gate is a PAIR: the limited run stays in [0,1] where the unlimited one does not, AND a smooth
# in-bounds field comes back with lambda == 1 on every face.
add_executable(test_mules_cpp ${PROJECT_SOURCE_DIR}/tests/test_mules_cpp.cu)
target_include_directories(test_mules_cpp PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_mules_cpp PRIVATE brae_core)
add_test(NAME mules_cpp
         COMMAND test_mules_cpp
                 ${BRAE_OF_TUTORIALS}/multiphase/interFoam/laminar/damBreak/damBreak)

# interFoam step 10: pEqn.H. Only the four things that are NOT shared with every other pressure
# corrector -- where the pressure gradient went (explicit in UEqn, implicit here), ddtCorr's
# interpolate(rho*rAU) weighting, the /rAUf inside the velocity reconstruction, and the reference that
# moves p AND rebuilds p_rgh. Two of the four are invisible unless rho and rAU actually vary, which is
# what the fixture is built for.
add_executable(test_inter_peqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_inter_peqn_cpp.cu)
target_include_directories(test_inter_peqn_cpp PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_peqn_cpp PRIVATE brae_core)
add_test(NAME inter_peqn_cpp COMMAND test_inter_peqn_cpp)

# interFoam step 11: the SECOND Courant number and the time step that shares it. maxAlphaCo is
# MANDATORY where maxCo defaults to 1 -- the same controlDict is refused by one reader and not the
# other -- and nearInterface() is a 0/1 mask over the CLOSED band [0.01, 0.99], so the two cells at its
# edges are the ones an advancing interface is passing through.
add_executable(test_alpha_courant_cpp ${PROJECT_SOURCE_DIR}/tests/test_alpha_courant_cpp.cu)
target_include_directories(test_alpha_courant_cpp PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_alpha_courant_cpp PRIVATE brae_core)
add_test(NAME alpha_courant_cpp COMMAND test_alpha_courant_cpp)

# interFoam step 12: the time step -- the alpha sub-cycle (33 of the 44 tutorials use it) and the order
# of the PIMPLE loop. The three sub-cycle mistakes all leave a BOUNDED, plausible alpha field, so no
# boundedness gate sees them; and the loop order decides which rho the momentum predictor is built on,
# which is a factor of 1000 at a water/air interface.
add_executable(test_inter_solve_cpp ${PROJECT_SOURCE_DIR}/tests/test_inter_solve_cpp.cu)
target_include_directories(test_inter_solve_cpp PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_solve_cpp PRIVATE brae_core)
add_test(NAME inter_solve_cpp COMMAND test_inter_solve_cpp)

# interFoam step 13: interface CURVATURE, the component the manifest carried as needing an instrumented
# OpenFOAM. It does not: curvature is GEOMETRY. A plane has K = 0 exactly at any resolution, a sphere
# of radius R has K = 2/R as the mesh refines, and the contact angle has an exact postcondition --
# acos(nHat & nf) IS theta. The convergence arm is what makes the sphere's tolerance a statement about
# the scheme rather than about this mesh.
add_executable(test_interface_curvature_cpp ${PROJECT_SOURCE_DIR}/tests/test_interface_curvature_cpp.cu)
target_include_directories(test_interface_curvature_cpp PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_interface_curvature_cpp PRIVATE brae_core)
add_test(NAME interface_curvature_cpp COMMAND test_interface_curvature_cpp)

# interFoam step 14: createFields against damBreak's own case. damBreak ships 0.orig and NO mesh, so
# the fixture is a PROCEDURE -- blockMesh then setFields -- not a directory this repo can check in.
# The shell gate runs OpenFOAM's own two utilities and hands the prepared case to the binary; both
# SKIP rather than fail where OpenFOAM is not installed.
add_executable(test_inter_case_cpp ${PROJECT_SOURCE_DIR}/tests/test_inter_case_cpp.cu)
target_include_directories(test_inter_case_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_inter_case_cpp PRIVATE brae_core)
add_test(NAME interfoam_createfields_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_createfields_vs_openfoam.sh)

# interFoam step 16: brae's interFoam against REAL OpenFOAM's, on damBreak, after exactly N IDENTICAL
# time steps. The gate rewrites the case with `adjustTimeStep no` before running either -- an adaptive
# step makes the two diverge in TIME the moment their Courant numbers differ at all, and then every
# field is compared at a different instant.
add_executable(test_inter_dambreak_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_dambreak_vs_openfoam.cu)
target_include_directories(test_inter_dambreak_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_dambreak_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_dambreak_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_dambreak_vs_openfoam.sh)

# ...and RAS/damBreak: kEpsilon in BOTH of incompressibleInterPhaseTransportModel's lineages, against
# real OpenFOAM's fields and its epsilon and k solver log, with OpenFOAM's laminar answer and its
# other lineage's as the controls.
add_executable(test_inter_ras_dambreak_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_ras_dambreak_vs_openfoam.cu)
target_include_directories(test_inter_ras_dambreak_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_ras_dambreak_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_ras_dambreak_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_ras_dambreak_vs_openfoam.sh)

# ...and kOmegaSST, on RAS/waterChannel as shipped: the ordinary incompressible model handed the
# mixture's nu, the cell wall distance, omega and nutk wall functions, and a convection scheme named
# through a pattern key. The control is OpenFOAM's own laminar run; the device loop must refuse by name.
add_executable(test_inter_waterchannel_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_waterchannel_vs_openfoam.cu)
target_include_directories(test_inter_waterchannel_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_waterchannel_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_waterchannel_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_waterchannel_vs_openfoam.sh)

# ...and the closure's ASSEMBLED SYSTEM against OpenFOAM's own, at the first call, under
# `Gauss limitedLinear 1`. The fields of that profile cannot witness the scheme -- one ulp of the
# initial omega is worth 4.3e-02 there after ten steps -- so the device arm is gated on the matrix it
# assembles, with OpenFOAM's `Gauss upwind` system as the control that must NOT match.
add_test(NAME interfoam_sst_assembly_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_sst_assembly_vs_openfoam.sh)
set_tests_properties(interfoam_sst_assembly_vs_openfoam PROPERTIES LABELS slow)

# ...and its kEpsilon twin, on RAS/damBreak, sharing the comparator (tests/assembly_compare.py). The
# oracle had to reach interFoam's lineage first: tools/dumpKEpsilon registers the VoF-phase table now.
add_test(NAME interfoam_kepsilon_assembly_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_kepsilon_assembly_vs_openfoam.sh)
set_tests_properties(interfoam_kepsilon_assembly_vs_openfoam PROPERTIES LABELS slow)

# ...whose first run found pressureInletOutletVelocity's snGrad() reading the stored value where
# OpenFOAM's directionMixed builds it from the valueFraction and the cell. Held on its own here.
add_executable(test_piov_sngrad ${PROJECT_SOURCE_DIR}/tests/test_piov_sngrad.cu)
target_include_directories(test_piov_sngrad PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_piov_sngrad PRIVATE brae_core)
add_test(NAME piov_sngrad COMMAND test_piov_sngrad)

# ...and MRF, on laminar/mixerVessel2D as shipped: the rotor's frame velocity, rho*(Omega x U), the
# zero-filtered ddtCorr flux and the relative phiHbyA. The control is OpenFOAM with the zone inactive;
# the device loop must refuse by name.
add_executable(test_inter_mrf_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_mrf_vs_openfoam.cu)
target_include_directories(test_inter_mrf_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_mrf_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_mrf_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_mrf_vs_openfoam.sh)

# ...and an fvOption, on RAS/angledDuct as shipped: explicitPorositySource/DarcyForchheimer in a
# force-dimensioned UEqn (rho and rho*nu from the mixture), under kEpsilon on a mesh non-orthogonal to
# 44.5 degrees. The control is OpenFOAM with the option inactive; the device loop must refuse by name.
add_executable(test_inter_angledduct_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_angledduct_vs_openfoam.cu)
target_include_directories(test_inter_angledduct_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_angledduct_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_angledduct_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_angledduct_vs_openfoam.sh)

# ...and the free-level inlet, on RAS/weirOverflow as shipped: variableHeightFlowRateInletVelocity on U
# and variableHeightFlowRate on alpha. The controls are OpenFOAM with the inlet frozen at its file value
# and with the alpha condition an inletOutlet of zero; the device loop must refuse by name.
# ...and localEuler (LTS), on RAS/DTCHull meshed serially by real OpenFOAM: setRDeltaT.H's local time
# step against the rDeltaT OpenFOAM writes at every step, and the fields it advances.
add_executable(test_inter_dtchull_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_dtchull_vs_openfoam.cu)
target_include_directories(test_inter_dtchull_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_dtchull_vs_openfoam PRIVATE brae_core)
# ONE TEST PER FILE under tests/interfoam_ami_device/: the device loop across a static cyclicAMI pair
# against real OpenFOAM, and its controls (run.sh holds the fixture and what is measured)
file(GLOB BRAE_AMI_DEVICE_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_ami_device
     ${PROJECT_SOURCE_DIR}/tests/interfoam_ami_device/*.sh)
list(FILTER BRAE_AMI_DEVICE_TESTS EXCLUDE REGEX "^(run|_common)\\.sh$")
foreach(rel ${BRAE_AMI_DEVICE_TESTS})
    string(REPLACE ".sh" "" name ${rel})
    add_test(NAME interfoam_ami_device_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_ami_device/${rel})
    set_tests_properties(interfoam_ami_device_${name} PROPERTIES
                         SKIP_RETURN_CODE 77
                         TIMEOUT 1800
                         RESOURCE_LOCK brae_gpu
                         ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
endforeach()

# ONE TEST PER FILE under tests/interfoam_dtchull/, each a part of this gate chosen through the gate's own
# selector, so no test is the whole gate; the device arms take the GPU one at a time (see interfoam_moving)
file(GLOB BRAE_DTCHULL_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_dtchull
     ${PROJECT_SOURCE_DIR}/tests/interfoam_dtchull/*.sh)
list(FILTER BRAE_DTCHULL_TESTS EXCLUDE REGEX "^(run|_common)\\.sh$")
foreach(rel ${BRAE_DTCHULL_TESTS})
    string(REPLACE ".sh" "" name ${rel})
    add_test(NAME interfoam_dtchull_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_dtchull/${rel})
    set_tests_properties(interfoam_dtchull_${name} PROPERTIES
                         SKIP_RETURN_CODE 77
                         TIMEOUT 1800
                         RESOURCE_LOCK "brae_gpu;brae_bigmesh"
                         ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
endforeach()

# ...and RAS/DTCHullMoving with its mesh FROZEN: the atmosphere's pressureInletOutletVelocity carries a
# `tangentialVelocity`, which interFoam's host loop claims and hands to the patch field. The control is brae
# with the entry stripped against OpenFOAM with it; OpenFOAM against itself without the entry is the proof the
# window can see it. Meshes the tutorial serially, as the DTCHull gate does.
add_executable(test_inter_dtchullmoving_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_dtchullmoving_vs_openfoam.cu)
target_include_directories(test_inter_dtchullmoving_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_dtchullmoving_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_dtchullmoving_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_dtchullmoving_vs_openfoam.sh)
set_tests_properties(interfoam_dtchullmoving_vs_openfoam PROPERTIES
                     SKIP_RETURN_CODE 77
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

add_executable(test_inter_weiroverflow_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_weiroverflow_vs_openfoam.cu)
target_include_directories(test_inter_weiroverflow_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_weiroverflow_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_weiroverflow_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_weiroverflow_vs_openfoam.sh)

# ...whose alpha condition that case cannot tell from a zeroGradient; its per-face logic, with bounds
# that are not 0 and 1, is held here against OpenFOAM's own text, with an inletOutlet as the control.
add_executable(test_variable_height_flow_rate ${PROJECT_SOURCE_DIR}/tests/test_variable_height_flow_rate.cu)
target_include_directories(test_variable_height_flow_rate PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_variable_height_flow_rate PRIVATE brae_core)
add_test(NAME variable_height_flow_rate COMMAND test_variable_height_flow_rate)

# The two permeable-wall conditions face by face, against OpenFOAM's text: the `alpha none` half and the
# dry-inflow face, which laminar/damBreakPermeable does not reach.
add_executable(test_permeable_wall_conditions ${PROJECT_SOURCE_DIR}/tests/test_permeable_wall_conditions.cu)
target_include_directories(test_permeable_wall_conditions PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_permeable_wall_conditions PRIVATE brae_core)
add_test(NAME permeable_wall_conditions COMMAND test_permeable_wall_conditions)

# ...and the permeable wall, on laminar/damBreakPermeable: a velocity and a pressure condition that each
# switch face by face, on the phase fraction at the patch, between a wall and an open boundary. Two
# profiles -- as shipped (the wall stays dry) and the column staged against the wall (faces go dry as it
# falls); the control is OpenFOAM with the wall closed, and the device loop must refuse by name.
add_executable(test_inter_permeable_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_permeable_vs_openfoam.cu)
target_include_directories(test_inter_permeable_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_permeable_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_permeable_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_permeable_vs_openfoam.sh)

# interFoam across a cyclic baffle, and with porousBafflePressure's jump on it, against real OpenFOAM on
# RAS/damBreakPorousBaffle. The script carries the measurements, the controls and the fail-proofs.
add_executable(test_inter_baffle_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_baffle_vs_openfoam.cu)
target_include_directories(test_inter_baffle_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_baffle_vs_openfoam PRIVATE brae_core)
# ONE TEST PER FILE under tests/interfoam_baffle/, each a part of this gate chosen through the gate's own
# selector, so no test is the whole gate; the device arms take the GPU one at a time (see interfoam_moving)
file(GLOB BRAE_BAFFLE_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_baffle
     ${PROJECT_SOURCE_DIR}/tests/interfoam_baffle/*.sh)
list(FILTER BRAE_BAFFLE_TESTS EXCLUDE REGEX "^(run|_common)\\.sh$")
foreach(rel ${BRAE_BAFFLE_TESTS})
    string(REPLACE ".sh" "" name ${rel})
    add_test(NAME interfoam_baffle_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_baffle/${rel})
    set_tests_properties(interfoam_baffle_${name} PROPERTIES
                         SKIP_RETURN_CODE 77
                         TIMEOUT 1800
                         RESOURCE_LOCK brae_gpu)
endforeach()

# The coupled cyclic patch field and porousBafflePressure's jump face by face, against OpenFOAM's text: the
# per-face jump, reversed flow and sign(0), which RAS/damBreakPorousBaffle does not reach.
add_executable(test_porous_baffle_pressure ${PROJECT_SOURCE_DIR}/tests/test_porous_baffle_pressure.cu)
target_include_directories(test_porous_baffle_pressure PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_porous_baffle_pressure PRIVATE brae_core)
add_test(NAME porous_baffle_pressure COMMAND test_porous_baffle_pressure)

# interFoam with LES kEqn on an axisymmetric wedge against real OpenFOAM on LES/nozzleFlow2D. The script
# carries the measurements, the control and the fail-proofs.
add_executable(test_inter_les_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_les_vs_openfoam.cu)
target_include_directories(test_inter_les_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_les_vs_openfoam PRIVATE brae_core)
# ONE TEST PER FILE under tests/interfoam_les/, each a part of this gate chosen through the gate's own
# selector, so no test is the whole gate; the device arms take the GPU one at a time (see interfoam_moving)
file(GLOB BRAE_LES_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_les
     ${PROJECT_SOURCE_DIR}/tests/interfoam_les/*.sh)
list(FILTER BRAE_LES_TESTS EXCLUDE REGEX "^(run|_common)\\.sh$")
foreach(rel ${BRAE_LES_TESTS})
    string(REPLACE ".sh" "" name ${rel})
    add_test(NAME interfoam_les_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_les/${rel})
    set_tests_properties(interfoam_les_${name} PROPERTIES
                         SKIP_RETURN_CODE 77
                         TIMEOUT 1800
                         RESOURCE_LOCK brae_gpu)
endforeach()

# interFoam on a FULLY PERIODIC mesh against real OpenFOAM, on brae's own validation/interFoamCyclic.
# The script carries the measurements, the control and the device refusal.
add_executable(test_inter_cyclic_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_cyclic_vs_openfoam.cu)
target_include_directories(test_inter_cyclic_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_cyclic_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_cyclic_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_cyclic_vs_openfoam.sh)

# interFoam with limitedLinear on div(rhoPhi,U) against real OpenFOAM on eulerianInjection. The script
# carries the measurements, the controls and the fail-proofs.
add_executable(test_inter_limitedlinear_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_limitedlinear_vs_openfoam.cu)
target_include_directories(test_inter_limitedlinear_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_limitedlinear_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_limitedlinear_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_limitedlinear_vs_openfoam.sh)

# brae's articulated-body dynamics against real OpenFOAM's own rigidBodyMotion::solve.
add_executable(test_rigid_body_dynamics_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_rigid_body_dynamics_vs_openfoam.cu)
target_include_directories(test_rigid_body_dynamics_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_rigid_body_dynamics_vs_openfoam PRIVATE brae_core)
add_test(NAME rigid_body_dynamics_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rigid_body_dynamics_vs_openfoam.sh)
set_tests_properties(rigid_body_dynamics_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# brae's Newmark integrator and acceleration relaxation against real OpenFOAM's own joint state.
add_executable(test_rigid_body_newmark_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_rigid_body_newmark_vs_openfoam.cu)
target_include_directories(test_rigid_body_newmark_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_rigid_body_newmark_vs_openfoam PRIVATE brae_core)
add_test(NAME rigid_body_newmark_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rigid_body_newmark_vs_openfoam.sh)
set_tests_properties(rigid_body_newmark_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# brae's fluid force and moment on a body's patches against real OpenFOAM's functionObjects::forces.
add_executable(test_rigid_body_forces_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_rigid_body_forces_vs_openfoam.cu)
target_include_directories(test_rigid_body_forces_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_rigid_body_forces_vs_openfoam PRIVATE brae_core)
add_test(NAME rigid_body_forces_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rigid_body_forces_vs_openfoam.sh)
set_tests_properties(rigid_body_forces_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# brae's rigid-body point transform against real OpenFOAM's pointDisplacement, with the joint state
# taken from OpenFOAM's own. The script carries the measurements and the controls.
add_executable(test_rigid_body_transform_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_rigid_body_transform_vs_openfoam.cu)
target_include_directories(test_rigid_body_transform_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_rigid_body_transform_vs_openfoam PRIVATE brae_core)
add_test(NAME rigid_body_transform_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rigid_body_transform_vs_openfoam.sh)
set_tests_properties(rigid_body_transform_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# brae's cell mapper against real OpenFOAM's own, through a refinement and an unrefinement step.
add_executable(test_field_map_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_field_map_vs_openfoam.cu)
target_include_directories(test_field_map_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_field_map_vs_openfoam PRIVATE brae_core)
add_test(NAME field_map_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/field_map_vs_openfoam.sh)
set_tests_properties(field_map_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# brae's refinement candidate selection against real OpenFOAM's own dynamicRefineFvMesh members.
add_executable(test_refine_candidates_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_refine_candidates_vs_openfoam.cu)
target_include_directories(test_refine_candidates_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_refine_candidates_vs_openfoam PRIVATE brae_core)
add_test(NAME refine_candidates_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/refine_candidates_vs_openfoam.sh)
set_tests_properties(refine_candidates_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# brae's polyTopoChange FACE ORDERING against OpenFOAM's own mesh numbering -- unit 1 of
# dynamicRefineFvMesh's producing half. The oracle is the mesh itself: a written OpenFOAM mesh is
# already in this order, so it must come back the identity. The script carries the controls.
add_executable(test_poly_topo_change_order ${PROJECT_SOURCE_DIR}/tests/test_poly_topo_change_order.cu)
target_include_directories(test_poly_topo_change_order PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_poly_topo_change_order PRIVATE brae_core)
add_test(NAME poly_topo_change_order_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/poly_topo_change_order_vs_openfoam.sh)
set_tests_properties(poly_topo_change_order_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# UNITS 3 AND 4 of the dynamicRefineFvMesh port: the action surface and changeMesh, against OpenFOAM's
# own mapPolyMesh for a scripted action list both codes read from the same file (tools/dumpTopoChange).
add_executable(test_topo_change_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_topo_change_vs_openfoam.cu)
target_include_directories(test_topo_change_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_topo_change_vs_openfoam PRIVATE brae_core)
add_test(NAME topo_change_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/topo_change_vs_openfoam.sh)
set_tests_properties(topo_change_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# UNIT 5a: faceEdges / edgeFaces / cellEdges -- the primitiveMesh addressing hexRef8::setRefinement reads,
# against OpenFOAM's own (tools/dumpMeshEdges dumps BOTH of OpenFOAM's two forms of each).
add_executable(test_mesh_edges_addressing_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_mesh_edges_addressing_vs_openfoam.cu)
target_include_directories(test_mesh_edges_addressing_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_mesh_edges_addressing_vs_openfoam PRIVATE brae_core)
add_test(NAME mesh_edges_addressing_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/mesh_edges_addressing_vs_openfoam.sh)
set_tests_properties(mesh_edges_addressing_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# UNIT 5b: hexRef8::setRefinement, against OpenFOAM's own (tools/dumpHexRef8 dumps cellAddedCells, the
# levels, the map and the mesh). Stage 5b-1 holds the points, the cells and the levels.
add_executable(test_refine_update_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_refine_update_vs_openfoam.cu)
target_include_directories(test_refine_update_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_refine_update_vs_openfoam PRIVATE brae_core)
add_test(NAME refine_update_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/refine_update_vs_openfoam.sh)
set_tests_properties(refine_update_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_hex_ref8_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_hex_ref8_vs_openfoam.cu)
target_include_directories(test_hex_ref8_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_hex_ref8_vs_openfoam PRIVATE brae_core)
add_test(NAME hex_ref8_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/hex_ref8_vs_openfoam.sh)
set_tests_properties(hex_ref8_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# UNIT 8e: brae's interFoam on an ADAPTIVE mesh against real OpenFOAM's -- the whole chain through the
# solver, on the mesh OpenFOAM itself refined, with a control for each half (the flux rebuild and the
# old-time mapping). The script says what the two steps exercise and what this fixture cannot witness.
add_executable(test_inter_amr_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_amr_vs_openfoam.cu)
target_include_directories(test_inter_amr_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_amr_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_amr_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_amr_vs_openfoam.sh)
set_tests_properties(interfoam_amr_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_amr_vs_openfoam
                     TIMEOUT 900)

# ...and a mesh that is ALREADY REFINED when the run starts, which is what RAS/motorBike needs: brae assigned
# level 0 everywhere and a snappyHexMesh mesh is at levels 1 to 3, so `maxRefinement` meant different things
# in the two codes. Two profiles, one fixture: `levels` reads OpenFOAM's ASCII cellLevel/pointLevel/
# refinementHistory, `levelsBinary` reads the SAME mesh re-written by OpenFOAM in binary with the history
# removed as motorBike's Allrun.pre removes it. Runs OpenFOAM three times (an ASCII seed, a binary seed and
# the oracle) plus a one-ulp twin; all cached.
add_test(NAME interfoam_amr_levels_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_amr_levels_vs_openfoam.sh)
set_tests_properties(interfoam_amr_levels_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_amr_vs_openfoam
                     TIMEOUT 1800)

# ...and RAS/motorBike ITSELF, which is what the levels, instance, RAS and autoMap units were each built for:
# a snappyHexMesh mesh at levels 1 to 3 with BINARY cellLevel/pointLevel and no refinementHistory, 61 patches,
# and kOmegaSST on a mesh that changes under it. The mesh is an INPUT, produced once by real OpenFOAM's own
# SERIAL snappyHexMesh -- the path Allrun.pre ships commented out, 1.53 s for 8655 cells -- because snappy is
# a mesher and brae's subject is interFoam. Nothing in this gate is parallel.
add_test(NAME interfoam_motorbike_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_motorbike_vs_openfoam.sh)
set_tests_properties(interfoam_motorbike_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_amr_vs_openfoam
                     TIMEOUT 1200)

# ...and a mesh that REFINES AND MOVES: laminar/oscillatingBox as shipped at fixed steps (`box`, twenty; and
# `boxUnrefine`, sixty, through its first unrefinement), the motion's points0 carried through every change and
# its points and points0 held to OpenFOAM's bitwise. The device arm refuses the combination by name.
# ONE TEST PER FILE under tests/interfoam_amr_motion/, each one profile of the gate through its
# AMR_MOTION_PART selector. MEASURED alone: box 37 s, unrefine 122 s (the two in one test read 393 s under
# ctest -j 4).
file(GLOB BRAE_AMR_MOTION_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_amr_motion
     ${PROJECT_SOURCE_DIR}/tests/interfoam_amr_motion/*.sh)
foreach(rel ${BRAE_AMR_MOTION_TESTS})
    string(REGEX REPLACE "\\.sh$" "" name ${rel})
    add_test(NAME interfoam_amr_motion_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_amr_motion/${rel})
    set_tests_properties(interfoam_amr_motion_${name} PROPERTIES
                         ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}"
                         SKIP_RETURN_CODE 77
                         DEPENDS test_inter_amr_vs_openfoam
                         RESOURCE_LOCK brae_gpu
                         TIMEOUT 600)
endforeach()

# ...and a GENUINE RESTART, where nothing is moved into constant/polyMesh: the seed leaves an ORIGINAL mesh
# there and a REFINED one in the time directory, and both codes continue from the time directory. That is the
# instance resolution the levels gate above explicitly did not claim -- OpenFOAM SEARCHES for points, faces
# and boundary separately (polyMesh.C:175-295 over Time::findInstance), and brae read constant/polyMesh in
# four places. It also covers `startFrom latestTime`, which braeInterFoam read and then dropped. Runs
# OpenFOAM three times (a seed, the continuation and a one-ulp twin), all cached, and the shipped solver
# twice -- once as written and once under the control that pins every instance to constant/.
add_test(NAME interfoam_restart_instance_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_restart_instance_vs_openfoam.sh)
set_tests_properties(interfoam_restart_instance_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_amr_vs_openfoam
                     TIMEOUT 1800)

# ...and a RAS TURBULENCE MODEL on a refining mesh, which is what RAS/motorBike needs before it can be
# attempted. Its own script for the same reason the MRF one is: it stages RAS/damBreak (already turbulent,
# already held statically by interfoam_ras_dambreak_vs_openfoam) plus one injected dynamicMeshDict, in two
# profiles -- kEpsilon as shipped and the same case made kOmegaSST, which is the only one of the two that
# builds a CELL wall distance.
add_test(NAME interfoam_amr_ras_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_amr_ras_vs_openfoam.sh)
set_tests_properties(interfoam_amr_ras_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_amr_vs_openfoam
                     TIMEOUT 1800)

# ...and MRF on a refining mesh, which needs a DIFFERENT case and therefore its own script: the four
# profiles above stage damBreakWithObstacle, whose fixedFluxPressure walls brae refuses MRF beside. It runs
# the same test binary on laminar/mixerVessel2D with the `mrf` profile, and runs OpenFOAM TWICE -- the
# oracle and a one-ulp twin, which is where its bounds come from. Both are cached.
add_test(NAME interfoam_amr_mrf_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_amr_mrf_vs_openfoam.sh)
set_tests_properties(interfoam_amr_mrf_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_amr_vs_openfoam
                     TIMEOUT 1800)

# brae's dynamicRefineFvMeshCoeffs reader against the three shipped AMR tutorials' own dictionaries --
# flat, flat-beside-a-subdict, and wrapped. The oracle is the dictionary; the script says why all three.
add_executable(test_refine_controls ${PROJECT_SOURCE_DIR}/tests/test_refine_controls.cu)
target_include_directories(test_refine_controls PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_refine_controls PRIVATE brae_core)
add_test(NAME refine_controls_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/refine_controls_vs_openfoam.sh)
set_tests_properties(refine_controls_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# pointConstraints::constrainDisplacement -- the symmetryPlane point patches and the corner constraints the
# rigid-body mesh motion ends every solve with -- against OpenFOAM's own class on the same staged input
# (tools/dumpPointConstraints): RAS/DTCHullMoving's mesh, its fixedValue patches pinned off zero, and a
# parallelepiped whose three tilted symmetry planes meet at a corner.
add_executable(test_point_constraints_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_point_constraints_vs_openfoam.cu)
target_include_directories(test_point_constraints_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_point_constraints_vs_openfoam PRIVATE brae_core)
add_test(NAME point_constraints_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/point_constraints_vs_openfoam.sh)
set_tests_properties(point_constraints_vs_openfoam PROPERTIES
                     SKIP_RETURN_CODE 77
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

# brae's pointPatchDist and the rigidBodyMeshMotion scale against real OpenFOAM, on the mesh
# RAS/floatingObject builds. The script carries the measurements and the controls.
add_executable(test_point_patch_dist_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_point_patch_dist_vs_openfoam.cu)
target_include_directories(test_point_patch_dist_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_point_patch_dist_vs_openfoam PRIVATE brae_core)
add_test(NAME point_patch_dist_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/point_patch_dist_vs_openfoam.sh)
set_tests_properties(point_patch_dist_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)
# ...and on a mesh of two unconnected blocks, where the wave does not reach every point: OpenFOAM leaves those
# at GREAT and rigidBodyMeshMotion does not move them. The control is the old value, 0.
add_test(NAME point_patch_dist_unreached_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/point_patch_dist_unreached_vs_openfoam.sh)
set_tests_properties(point_patch_dist_unreached_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# interFoam across a coded cyclicACMI baffle against real OpenFOAM on damBreakLeakage. The script carries
# the measurements, the controls and the fail-proofs.
add_executable(test_inter_leakage_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_leakage_vs_openfoam.cu)
target_include_directories(test_inter_leakage_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_leakage_vs_openfoam PRIVATE brae_core)
# ONE TEST PER FILE under tests/interfoam_leakage/, each a part of this gate chosen through the gate's own
# selector, so no test is the whole gate; the device arms take the GPU one at a time (see interfoam_moving)
file(GLOB BRAE_LEAKAGE_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_leakage
     ${PROJECT_SOURCE_DIR}/tests/interfoam_leakage/*.sh)
list(FILTER BRAE_LEAKAGE_TESTS EXCLUDE REGEX "^(run|_common)\\.sh$")
foreach(rel ${BRAE_LEAKAGE_TESTS})
    string(REPLACE ".sh" "" name ${rel})
    add_test(NAME interfoam_leakage_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_leakage/${rel})
    set_tests_properties(interfoam_leakage_${name} PROPERTIES
                         SKIP_RETURN_CODE 77
                         TIMEOUT 1800
                         RESOURCE_LOCK brae_gpu)
endforeach()

# interFoam with the mangrove fvOptions against real OpenFOAM on waves/mangroveInteraction. The script
# carries the measurements, the controls and the fail-proofs.
add_executable(test_inter_mangrove_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_mangrove_vs_openfoam.cu)
target_include_directories(test_inter_mangrove_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_mangrove_vs_openfoam PRIVATE brae_core)
# ONE TEST PER FILE under tests/interfoam_mangrove/, each a part of this gate chosen through the gate's own
# selector, so no test is the whole gate; the device arms take the GPU one at a time (see interfoam_moving)
file(GLOB BRAE_MANGROVE_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_mangrove
     ${PROJECT_SOURCE_DIR}/tests/interfoam_mangrove/*.sh)
list(FILTER BRAE_MANGROVE_TESTS EXCLUDE REGEX "^(run|_common)\\.sh$")
foreach(rel ${BRAE_MANGROVE_TESTS})
    string(REPLACE ".sh" "" name ${rel})
    add_test(NAME interfoam_mangrove_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_mangrove/${rel})
    set_tests_properties(interfoam_mangrove_${name} PROPERTIES
                         SKIP_RETURN_CODE 77
                         TIMEOUT 1800
                         RESOURCE_LOCK brae_gpu)
endforeach()

# CrankNicolson ddt on RAS/damBreak, both arms (run by tests/interfoam_cn_vs_openfoam.sh)
add_executable(test_inter_cn_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_cn_vs_openfoam.cu)
target_include_directories(test_inter_cn_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_cn_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_cn_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_cn_vs_openfoam.sh)
set_tests_properties(interfoam_cn_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

add_executable(test_inter_ami_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_ami_vs_openfoam.cu)
target_include_directories(test_inter_ami_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_ami_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_ami_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_ami_vs_openfoam.sh)

# ...and the wave boundary conditions (waveAlpha, waveVelocity) on laminar/waves/stokesI: fields, the
# patch values the wave model assigns, its two derived constants and the ORDER it updates in, all
# against real OpenFOAM; the control is OpenFOAM's own answer with the wave switched off.
add_executable(test_inter_waves_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_waves_vs_openfoam.cu)
target_include_directories(test_inter_waves_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_waves_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_waves_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_waves_vs_openfoam.sh)

# MESH MOTION ON ITS OWN, before any solver reads it: the moved points, meshPhi, V and C of a
# solid-body motion against OpenFOAM's moveDynamicMesh, per time step.
add_executable(test_mesh_motion_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_mesh_motion_vs_openfoam.cu)
target_include_directories(test_mesh_motion_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_mesh_motion_vs_openfoam PRIVATE brae_core)
add_test(NAME mesh_motion_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/mesh_motion_vs_openfoam.sh)

# ...and a DEFORMING one: displacementLaplacian from the waveMaker paddles, the points, the motion
# solver's two fields, meshPhi and every GAMG solve against OpenFOAM's moveDynamicMesh, per time step.
add_executable(test_displacement_laplacian_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_displacement_laplacian_vs_openfoam.cu)
target_include_directories(test_displacement_laplacian_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_displacement_laplacian_vs_openfoam PRIVATE brae_core)
add_test(NAME displacement_laplacian_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/displacement_laplacian_vs_openfoam.sh)

# ...and interFoam ON a mesh in solid-body motion: a closed tank, against OpenFOAM's fields, its moved
# mesh, its Uf and its wall velocity, with the same tank held still as the control.
add_executable(test_inter_moving_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_moving_vs_openfoam.cu)
target_include_directories(test_inter_moving_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_moving_vs_openfoam PRIVATE brae_core)
# ONE TEST PER FILE under tests/interfoam_moving/<family>/, each gating at most three profiles through the
# gate's own selectors (MOVING_ONLY stages a profile and its control, MOVING_GATE gates the profile alone).
# A change to one mesh motion is checked by its file. As ONE test this gate took 2900 s; as ten family
# tests run five abreast the flap family alone took 1112 s where it takes 260 s by itself -- every
# OpenFOAM run was a cache hit, and the time was five processes sharing one GPU. Hence the RESOURCE_LOCK:
# tests that run a device arm take the GPU one at a time, and the CPU-only tests run beside them.
file(GLOB BRAE_MOVING_TESTS RELATIVE ${PROJECT_SOURCE_DIR}/tests/interfoam_moving
     ${PROJECT_SOURCE_DIR}/tests/interfoam_moving/*/*.sh)
foreach(rel ${BRAE_MOVING_TESTS})
    string(REPLACE ".sh" "" name ${rel})
    string(REPLACE "/" "_" name ${name})
    add_test(NAME interfoam_moving_${name}
             COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_moving/${rel})
    # THE TIME LIMIT IS FOR A HANG, and it has to clear the suite's load: piston/outer_once_control is 210 s by
    # itself and ran past 900 s in the full suite at -j 8 (2026-10-08), beside tests of the other solvers that
    # take the GPU without this lock; piston/outer_control was 629 s in the same run. No check changed.
    set_tests_properties(interfoam_moving_${name} PROPERTIES
                         SKIP_RETURN_CODE 77
                         TIMEOUT 1800
                         RESOURCE_LOCK brae_gpu)
endforeach()

# ...and the SOLVER those tutorials name for p_rghFinal, GAMG, as shipped: OpenFOAM's hierarchy level
# by level, the coarsest-level solve of every V-cycle and every p_rgh line, from its log under its own
# debug switches; the control is OpenFOAM's own answer with PCG in GAMG's place.
add_executable(test_inter_gamg_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_gamg_vs_openfoam.cu)
target_include_directories(test_inter_gamg_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_gamg_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_gamg_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_gamg_vs_openfoam.sh)

# ...and capillaryRise, where surface tension IS the answer rather than a correction, and which is the
# ONLY shipped interFoam tutorial with a contact angle. damBreak would still look broadly right with
# sigma set to zero; this one would not move at all.
add_executable(test_inter_capillary_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_capillary_vs_openfoam.cu)
target_include_directories(test_inter_capillary_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS}
                           ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_inter_capillary_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_capillaryrise_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_capillaryrise_vs_openfoam.sh)

# ...and the CURVATURE itself, against OpenFOAM's UNMODIFIED interfaceProperties via
# tools/dumpInterfaceK. Swept over four calculateK pass counts, because the curvature is a FIXED POINT
# in those passes -- it reads the wall gradient its own previous pass wrote -- so agreement at one
# count says much less than agreement at four.
add_executable(test_inter_curvature_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_inter_curvature_vs_openfoam.cu)
target_include_directories(test_inter_curvature_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_inter_curvature_vs_openfoam PRIVATE brae_core)
add_test(NAME interfoam_curvature_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_curvature_vs_openfoam.sh)
set_tests_properties(interfoam_curvature_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_curvature_vs_openfoam
                     TIMEOUT 600)
set_tests_properties(interfoam_capillaryrise_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_capillary_vs_openfoam
                     TIMEOUT 900)
set_tests_properties(interfoam_dambreak_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_dambreak_vs_openfoam
                     TIMEOUT 900)
set_tests_properties(interfoam_createfields_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_inter_case_cpp
                     TIMEOUT 600)

# These five read damBreak's own dictionaries. A box without OpenFOAM must SKIP, not report green: the
# arms silently stopped running the moment $FOAM_TUTORIALS was empty, and nothing said so.
set_tests_properties(two_phase_mixture interface_properties inter_create_fields
                     alpha_eqn_cpp mules_cpp
                     PROPERTIES SKIP_RETURN_CODE 77)

# WAVE 1 / Q3: interFoam's start-up guards, its divergence stop and minIter on the pressure solves (2026-11-21).
# Each script states which loop and which pressure solver each arm runs: the arms that hold the default GPU
# path unset BRAE_PRESSURE_CASE_SOLVER themselves, and the case-solver arms set it, as CI does for every test.
add_test(NAME interfoam_guards
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_guards.sh)
add_test(NAME interfoam_divergence_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_divergence_vs_openfoam.sh)
add_test(NAME interfoam_miniter_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/interfoam_miniter_vs_openfoam.sh)
set_tests_properties(interfoam_guards interfoam_divergence_vs_openfoam interfoam_miniter_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}"
                     SKIP_RETURN_CODE 77
                     RESOURCE_LOCK brae_gpu
                     LABELS slow
                     TIMEOUT 900)
