# Finite-volume units. Mesh topology and geometry, fvc/fvm operators, gradients and interpolation schemes, wall
# distance, coupled interfaces (cyclic, cyclicAMI, cyclicACMI, wedge) and boundary conditions, each held to OpenFOAM
# or to a host reference on its own.
#
# Included by tests/CMakeLists.txt.

add_executable(test_mesh_topology ${PROJECT_SOURCE_DIR}/tests/test_mesh_topology.cu)
target_include_directories(test_mesh_topology PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_mesh_topology PRIVATE brae_core)
add_test(NAME mesh_topology
         COMMAND test_mesh_topology
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)

add_executable(test_mesh_geometry ${PROJECT_SOURCE_DIR}/tests/test_mesh_geometry.cu)
target_include_directories(test_mesh_geometry PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_mesh_geometry PRIVATE brae_core)
add_test(NAME mesh_geometry
         COMMAND test_mesh_geometry ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# Transient ddt (PIMPLE foundation): deviceFvmDdt reproduces OF EulerDdtScheme/backwardDdtScheme exactly.
add_executable(test_gpu_ddt ${PROJECT_SOURCE_DIR}/tests/test_gpu_ddt.cu)
target_include_directories(test_gpu_ddt PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_ddt PRIVATE brae_core)
add_test(NAME gpu_ddt COMMAND test_gpu_ddt)

add_executable(test_gpu_ddt_cn ${PROJECT_SOURCE_DIR}/tests/test_gpu_ddt_cn.cu)
target_include_directories(test_gpu_ddt_cn PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_ddt_cn PRIVATE brae_core)
add_test(NAME gpu_ddt_cn COMMAND test_gpu_ddt_cn)

add_executable(test_gpu_mrf ${PROJECT_SOURCE_DIR}/tests/test_gpu_mrf.cu)
target_include_directories(test_gpu_mrf PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_mrf PRIVATE brae_core)
add_test(NAME gpu_mrf
         COMMAND test_gpu_mrf ${PROJECT_SOURCE_DIR}/validation/mrfBox)

add_executable(test_fvc_grad ${PROJECT_SOURCE_DIR}/tests/test_fvc_grad.cu)
target_include_directories(test_fvc_grad PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvc_grad PRIVATE brae_core)
add_test(NAME fvc_grad
         COMMAND test_fvc_grad ${PROJECT_SOURCE_DIR}/validation/pitzDaily282)

add_executable(test_empty_face_grad ${PROJECT_SOURCE_DIR}/tests/test_empty_face_grad.cu)
target_include_directories(test_empty_face_grad PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_empty_face_grad PRIVATE brae_core)
add_test(NAME empty_face_grad_vs_openfoam
         COMMAND test_empty_face_grad ${PROJECT_SOURCE_DIR}/validation/pitzDaily282 282)

add_executable(test_empty_face_flux ${PROJECT_SOURCE_DIR}/tests/test_empty_face_flux.cu)
target_include_directories(test_empty_face_flux PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_empty_face_flux PRIVATE brae_core)
add_test(NAME empty_face_flux
         COMMAND test_empty_face_flux ${PROJECT_SOURCE_DIR}/validation/pitzDaily282 282)

add_executable(test_fvm_laplacian ${PROJECT_SOURCE_DIR}/tests/test_fvm_laplacian.cu)
target_include_directories(test_fvm_laplacian PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvm_laplacian PRIVATE brae_core)
add_test(NAME fvm_laplacian
         COMMAND test_fvm_laplacian ${PROJECT_SOURCE_DIR}/validation/matrixDump)

add_executable(test_fvm_div ${PROJECT_SOURCE_DIR}/tests/test_fvm_div.cu)
target_include_directories(test_fvm_div PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvm_div PRIVATE brae_core)
add_test(NAME fvm_div
         COMMAND test_fvm_div ${PROJECT_SOURCE_DIR}/validation/matrixDump)

# gradBKernel's type-code branches, one patch per branch, against the host reference. Only two of them
# had a fixture (symmetry, and the flux-conditional ones through waterChannel's oneCorrector profile);
# the rest were held by whole-case gates, where `snGrad = the prescribed gradient` against
# `(value - internal)*deltaCoeffs` can hide behind everything else in the step.
# assignable() on a transform patch: brae returns false where OpenFOAM returns true, as one half of a
# pair with HbyA having no constraint patch fields -- both give a ZERO NORMAL FLUX, which is all the
# boundary of HbyA feeds. This pins the composition, so neither half moves alone (correcting
# assignable() by itself reads U 6.1507e-01 on RAS/damBreakLeakage).
add_executable(test_symmetry_assignable ${PROJECT_SOURCE_DIR}/tests/test_symmetry_assignable.cu)
target_include_directories(test_symmetry_assignable PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_symmetry_assignable PRIVATE brae_core)
add_test(NAME symmetry_assignable
         COMMAND test_symmetry_assignable
                 ${PROJECT_SOURCE_DIR}/validation/windAroundBuildingsBox
                 ${PROJECT_SOURCE_DIR}/validation/halfChannel)
set_tests_properties(symmetry_assignable PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_device_grad_boundary ${PROJECT_SOURCE_DIR}/tests/test_device_grad_boundary.cu)
target_include_directories(test_device_grad_boundary PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_device_grad_boundary PRIVATE brae_core)
add_test(NAME device_grad_boundary
         COMMAND test_device_grad_boundary ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb)
set_tests_properties(device_grad_boundary PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_fvc_ops ${PROJECT_SOURCE_DIR}/tests/test_fvc_ops.cu)
target_include_directories(test_fvc_ops PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvc_ops PRIVATE brae_core)
add_test(NAME fvc_ops
         COMMAND test_fvc_ops ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym)

# OpenFOAM's fusedGauss scheme family (pitzDaily_fused). It is Gauss written without field temporaries,
# so brae reads it AS Gauss and announces that; this gate measures the equivalence from OpenFOAM ITSELF
# (its own two spellings of the same case), asserts brae's rewrite is bit-for-bit a rename, and runs the
# case end to end. Its control forces div(phi,U) to upwind.
add_test(NAME fusedgauss_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/fusedgauss_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(fusedgauss_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# The cyclicAMI CUDA path against the _cpp REFERENCE, stage by stage: interpolation (scalar and rotated
# vector), face value, both matrix assemblies with their diagonals, the matrix action, the interface
# flux, UEqn.H()'s interface term and the interface's share of div(phi). Twelve checks at 1e-12; they
# pass bit-identically. It caught the reference's own assembleLaplacian sign on its first run.
add_test(NAME ami_cpp_vs_device
         COMMAND ${PROJECT_SOURCE_DIR}/tests/ami_cpp_vs_device.sh
                 ${PROJECT_SOURCE_DIR}/validation/pipeCyclicAMI 100
                 ${PROJECT_SOURCE_DIR}/validation/pipeCyclicAMIConformal)
set_tests_properties(ami_cpp_vs_device PROPERTIES SKIP_RETURN_CODE 77)

add_executable(ami_cpp_vs_device ${PROJECT_SOURCE_DIR}/tests/ami_cpp_vs_device.cu)
target_include_directories(ami_cpp_vs_device PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(ami_cpp_vs_device PRIVATE brae_core)

# The least-squares gradient and its cell limiter taken at a subset of the cells (fvc::GradSubset), against the
# whole-mesh functions cell for cell and bit for bit: host only, no case.
add_executable(test_grad_subset ${PROJECT_SOURCE_DIR}/tests/test_grad_subset.cu)
target_include_directories(test_grad_subset PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_grad_subset PRIVATE brae_core)
add_test(NAME grad_subset COMMAND test_grad_subset)

# ...and across a coupled pair (a translational cyclic, and the same pair with a cyclicAMI's addressing): the
# coupled branches of the fit and of the limiter in their subset form, which the solver does not reach (it
# refuses a least-squares nHat on a mesh with a pair) and which nothing else ran.
add_executable(test_grad_subset_coupled ${PROJECT_SOURCE_DIR}/tests/test_grad_subset_coupled.cu)
target_include_directories(test_grad_subset_coupled PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_grad_subset_coupled PRIVATE brae_core)
add_test(NAME grad_subset_coupled COMMAND test_grad_subset_coupled)

# Cyclic increment C1: cyclic interface geometry (serial)
add_executable(test_cyclic_geometry ${PROJECT_SOURCE_DIR}/tests/test_cyclic_geometry.cu)
target_include_directories(test_cyclic_geometry PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_geometry PRIVATE brae_core)
add_test(NAME cyclic_geometry COMMAND test_cyclic_geometry ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C2: cyclic-coupled Laplacian operator (serial)
add_executable(test_cyclic_laplacian ${PROJECT_SOURCE_DIR}/tests/test_cyclic_laplacian.cu)
target_include_directories(test_cyclic_laplacian PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_laplacian PRIVATE brae_core)
add_test(NAME cyclic_laplacian COMMAND test_cyclic_laplacian ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C3a: CyclicFvPatchField + cyclic-aware fvc (serial)
add_executable(test_cyclic_fvc ${PROJECT_SOURCE_DIR}/tests/test_cyclic_fvc.cu)
target_include_directories(test_cyclic_fvc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_fvc PRIVATE brae_core)
add_test(NAME cyclic_fvc COMMAND test_cyclic_fvc ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)

add_executable(test_linear_upwind_const ${PROJECT_SOURCE_DIR}/tests/test_linear_upwind_const.cu)
target_include_directories(test_linear_upwind_const PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_linear_upwind_const PRIVATE brae_core)
add_test(NAME linear_upwind_const
         COMMAND test_linear_upwind_const ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# Cyclic increment C3b: cyclic coupling inside an iterative solver (periodic Poisson)
add_executable(test_cyclic_poisson ${PROJECT_SOURCE_DIR}/tests/test_cyclic_poisson.cu)
target_include_directories(test_cyclic_poisson PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_poisson PRIVATE brae_core)
add_test(NAME cyclic_poisson COMMAND test_cyclic_poisson ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C3c: cyclic in the full SIMPLE loop (periodic channel)
# phi's coupled-patch flux must survive write -> read, or an AMI/cyclic restart silently rebuilds the
# interface flux from U and lands the difference on the momentum diagonal (see the test's header).
# meshPhi / fvc::makeRelative: a fluid moving WITH the mesh must have zero relative flux, on the
# cyclicACMI interface faces as much as anywhere else (see the test's header).
# fvc::ddtCorr: the ddt flux correction pimpleFoam adds to phiHbyA (see the test's header).
# fvc::correctUf: Sf & Uf == phi, and why phi + mesh.phi() cannot stand in for it (see the header).
# A cyclicACMI non-overlap patch is a wall only where it is closed (see the test's header).
add_executable(test_acmi_wall_gate ${PROJECT_SOURCE_DIR}/tests/test_acmi_wall_gate.cu)
target_include_directories(test_acmi_wall_gate PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_acmi_wall_gate PRIVATE brae_core)
add_test(NAME acmi_wall_gate COMMAND test_acmi_wall_gate)
set_tests_properties(acmi_wall_gate PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_cell_limit_interface ${PROJECT_SOURCE_DIR}/tests/test_cell_limit_interface.cu)
target_include_directories(test_cell_limit_interface PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_cell_limit_interface PRIVATE brae_core)
add_test(NAME cell_limit_interface COMMAND test_cell_limit_interface)
set_tests_properties(cell_limit_interface PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_scalar_interface_corr ${PROJECT_SOURCE_DIR}/tests/test_scalar_interface_corr.cu)
target_include_directories(test_scalar_interface_corr PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_scalar_interface_corr PRIVATE brae_core)
add_test(NAME scalar_interface_corr COMMAND test_scalar_interface_corr)
set_tests_properties(scalar_interface_corr PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_wedge_patch ${PROJECT_SOURCE_DIR}/tests/test_wedge_patch.cu)
target_include_directories(test_wedge_patch PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_wedge_patch PRIVATE brae_core)
add_test(NAME wedge_patch COMMAND test_wedge_patch)

add_executable(test_ami_broad_phase ${PROJECT_SOURCE_DIR}/tests/test_ami_broad_phase.cu)
target_include_directories(test_ami_broad_phase PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_ami_broad_phase PRIVATE brae_core)
add_test(NAME ami_broad_phase COMMAND test_ami_broad_phase)

add_executable(test_lust_implicit ${PROJECT_SOURCE_DIR}/tests/test_lust_implicit.cu)
target_include_directories(test_lust_implicit PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_lust_implicit PRIVATE brae_core)
add_test(NAME lust_implicit COMMAND test_lust_implicit)
# DIC on the device: the level-scheduled DILU with lower aliased to upper, held BIT-IDENTICAL to a
# transcription of DICPreconditioner.C, and PCG with it held to the host PCG+DIC's ITERATE at relTol
# 0.05 -- which is where interFoam's p_rgh stops on every corrector but the last.
add_executable(test_device_dic ${PROJECT_SOURCE_DIR}/tests/test_device_dic.cu)
target_include_directories(test_device_dic PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_dic PRIVATE brae_core)
add_test(NAME device_dic COMMAND test_device_dic)
set_tests_properties(device_dic PROPERTIES SKIP_RETURN_CODE 77)

# The fused Gauss gradient (N fields, one launch) is bit-identical to N separate gradKernel launches.
add_executable(test_grad_fused ${PROJECT_SOURCE_DIR}/tests/test_grad_fused.cu)
target_include_directories(test_grad_fused PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_grad_fused PRIVATE brae_core)
add_test(NAME grad_fused COMMAND test_grad_fused)

# FP-3: the same, for the leastSquares fit -- plus the inverted dd tensor cached on the mesh.
add_executable(test_lsq_grad_fused ${PROJECT_SOURCE_DIR}/tests/test_lsq_grad_fused.cu)
target_include_directories(test_lsq_grad_fused PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_lsq_grad_fused PRIVATE brae_core)
add_test(NAME lsq_grad_fused COMMAND test_lsq_grad_fused)

# FP-4: and for the cellLimited limiter, whose 9 momentum launches were 44% of that phase.
add_executable(test_cell_limit_grad_fused ${PROJECT_SOURCE_DIR}/tests/test_cell_limit_grad_fused.cu)
target_include_directories(test_cell_limit_grad_fused PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_cell_limit_grad_fused PRIVATE brae_core)
add_test(NAME cell_limit_grad_fused COMMAND test_cell_limit_grad_fused)

add_executable(test_interface_invariants ${PROJECT_SOURCE_DIR}/tests/test_interface_invariants.cu)
target_include_directories(test_interface_invariants PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_interface_invariants PRIVATE brae_core)
add_test(NAME interface_invariants COMMAND test_interface_invariants)

add_executable(test_periodic_ami ${PROJECT_SOURCE_DIR}/tests/test_periodic_ami.cu)
target_include_directories(test_periodic_ami PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_periodic_ami PRIVATE brae_core)
add_test(NAME periodic_ami COMMAND test_periodic_ami)

add_executable(test_coupled_patch_refusal ${PROJECT_SOURCE_DIR}/tests/test_coupled_patch_refusal.cu)
target_include_directories(test_coupled_patch_refusal PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_coupled_patch_refusal PRIVATE brae_core)
add_test(NAME coupled_patch_refusal COMMAND test_coupled_patch_refusal)

add_executable(test_correct_phi ${PROJECT_SOURCE_DIR}/tests/test_correct_phi.cu)
target_include_directories(test_correct_phi PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_correct_phi PRIVATE brae_core)
add_test(NAME correct_phi COMMAND test_correct_phi)

add_executable(test_wedge_device_coeffs ${PROJECT_SOURCE_DIR}/tests/test_wedge_device_coeffs.cu)
target_include_directories(test_wedge_device_coeffs PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_wedge_device_coeffs PRIVATE brae_core)
add_test(NAME wedge_device_coeffs COMMAND test_wedge_device_coeffs)

add_executable(test_device_grad_symmetry ${PROJECT_SOURCE_DIR}/tests/test_device_grad_symmetry.cu)
target_include_directories(test_device_grad_symmetry PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_grad_symmetry PRIVATE brae_core)
add_test(NAME device_grad_symmetry COMMAND test_device_grad_symmetry ${PROJECT_SOURCE_DIR}/validation)

add_executable(test_vol_point_interp ${PROJECT_SOURCE_DIR}/tests/test_vol_point_interp.cu)
target_include_directories(test_vol_point_interp PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_vol_point_interp PRIVATE brae_core)
add_test(NAME vol_point_interp COMMAND test_vol_point_interp)

add_executable(test_acmi_scale ${PROJECT_SOURCE_DIR}/tests/test_acmi_scale.cu)
target_include_directories(test_acmi_scale PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_acmi_scale PRIVATE brae_core)
add_test(NAME acmi_scale COMMAND test_acmi_scale)
set_tests_properties(acmi_scale PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_limited_linear_v ${PROJECT_SOURCE_DIR}/tests/test_limited_linear_v.cu)
target_include_directories(test_limited_linear_v PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_limited_linear_v PRIVATE brae_core)
add_test(NAME limited_linear_v COMMAND test_limited_linear_v)

add_executable(test_coupled_phi_roundtrip ${PROJECT_SOURCE_DIR}/tests/test_coupled_phi_roundtrip.cu)
target_include_directories(test_coupled_phi_roundtrip PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_coupled_phi_roundtrip PRIVATE brae_core)
add_test(NAME coupled_phi_roundtrip COMMAND test_coupled_phi_roundtrip ${PROJECT_BINARY_DIR}/coupled_phi_roundtrip_work)

add_executable(test_cyclic_channel ${PROJECT_SOURCE_DIR}/tests/test_cyclic_channel.cu)
target_include_directories(test_cyclic_channel PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_channel PRIVATE brae_core)
add_test(NAME cyclic_channel COMMAND test_cyclic_channel ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C-dev: the DEVICE cyclic interface operator vs the validated host (deviceAmul == cyclicAmul + symmetry)
add_executable(test_cyclic_device ${PROJECT_SOURCE_DIR}/tests/test_cyclic_device.cu)
target_include_directories(test_cyclic_device PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_device PRIVATE brae_core)
add_test(NAME cyclic_device COMMAND test_cyclic_device ${PROJECT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C4: rotational transform (geometry forwardT + vector coupling)
add_executable(test_cyclic_rotational ${PROJECT_SOURCE_DIR}/tests/test_cyclic_rotational.cu)
target_include_directories(test_cyclic_rotational PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_rotational PRIVATE brae_core)
add_test(NAME cyclic_rotational COMMAND test_cyclic_rotational ${PROJECT_SOURCE_DIR}/validation/cyclicWedge)

# Cyclic increment C-dev-rot: the DEVICE rotational coupling (deviceCyclicFluxRot == forwardT·U[nbr], conservative)
add_executable(test_cyclic_device_rot ${PROJECT_SOURCE_DIR}/tests/test_cyclic_device_rot.cu)
target_include_directories(test_cyclic_device_rot PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_device_rot PRIVATE brae_core)
add_test(NAME cyclic_device_rot COMMAND test_cyclic_device_rot ${PROJECT_SOURCE_DIR}/validation/cyclicWedge)

add_executable(test_acmi_area_scaling ${PROJECT_SOURCE_DIR}/tests/test_acmi_area_scaling.cu)
target_include_directories(test_acmi_area_scaling PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_acmi_area_scaling PRIVATE brae_core)
add_test(NAME acmi_area_scaling COMMAND test_acmi_area_scaling)
set_tests_properties(acmi_area_scaling PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_acmi_mask ${PROJECT_SOURCE_DIR}/tests/test_acmi_mask.cu)
target_include_directories(test_acmi_mask PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_acmi_mask PRIVATE brae_core)
add_test(NAME acmi_mask COMMAND test_acmi_mask)
set_tests_properties(acmi_mask PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_acmi_patch_read ${PROJECT_SOURCE_DIR}/tests/test_acmi_patch_read.cu)
target_include_directories(test_acmi_patch_read PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_acmi_patch_read PRIVATE brae_core)
add_test(NAME acmi_patch_read COMMAND test_acmi_patch_read)

add_executable(test_ami_curved ${PROJECT_SOURCE_DIR}/tests/test_ami_curved.cu)
target_include_directories(test_ami_curved PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_curved PRIVATE brae_core)
add_test(NAME ami_curved COMMAND test_ami_curved)

add_executable(test_ami_weights ${PROJECT_SOURCE_DIR}/tests/test_ami_weights.cu)
target_include_directories(test_ami_weights PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_weights PRIVATE brae_core)
add_test(NAME ami_weights COMMAND test_ami_weights)

add_executable(test_ami_device ${PROJECT_SOURCE_DIR}/tests/test_ami_device.cu)
target_include_directories(test_ami_device PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_device PRIVATE brae_core)
add_test(NAME ami_device COMMAND test_ami_device)

# The legacy AMI builder's store test and row normalisation against REAL OpenFOAM's own AMI on two constructed
# meshes (a cyclicACMI pair and a plain cyclicAMI pair, each offset by a sliver). The scripts carry the fixtures,
# the premises and the controls.
add_executable(test_ami_store_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_ami_store_vs_openfoam.cu)
target_include_directories(test_ami_store_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_store_vs_openfoam PRIVATE brae_core)
add_test(NAME ami_store_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/ami_store_vs_openfoam.sh)
set_tests_properties(ami_store_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME ami_rows_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/ami_rows_vs_openfoam.sh)
set_tests_properties(ami_rows_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_ami_geometry ${PROJECT_SOURCE_DIR}/tests/test_ami_geometry.cu)
target_include_directories(test_ami_geometry PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_geometry PRIVATE brae_core)
add_test(NAME ami_geometry COMMAND test_ami_geometry ${PROJECT_SOURCE_DIR}/validation/cyclicChannelAMI)

# GPU offload G3: fvc explicit operators on device
add_executable(test_gpu_fvc ${PROJECT_SOURCE_DIR}/tests/test_gpu_fvc.cu)
target_include_directories(test_gpu_fvc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_fvc PRIVATE brae_core)
add_test(NAME gpu_fvc COMMAND test_gpu_fvc ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G4: fvm matrix assembly on device
add_executable(test_gpu_fvm ${PROJECT_SOURCE_DIR}/tests/test_gpu_fvm.cu)
target_include_directories(test_gpu_fvm PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_fvm PRIVATE brae_core)
add_test(NAME gpu_fvm COMMAND test_gpu_fvm ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
