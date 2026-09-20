#!/usr/bin/env python3
"""Generate the OpenFOAM port manifest for one solver.

WHY THIS EXISTS. Porting OpenFOAM has repeatedly failed the same way here: implement what a tutorial
exercises, run it, discover the next missing piece from a wrong answer. Every one of those discoveries
(`relaxationFactors ".*Final"`, MRF silently ignored, the `calculated` nut, LUST's implicit weights) was a
runtime-selected string or a dictionary key that OpenFOAM reads and the port did not. All of them are
statically visible. This emits them as a checklist BEFORE any code is written.

The manifest has two halves and they are kept strictly apart:

  DERIVED   -- produced by querying ofscan's index of the OpenFOAM tree. Never edited by hand. If OpenFOAM
               changes, this half changes on the next run and the diff is the drift.
  CURATED   -- the classification (what kind of component this is) and the brae status (what we intend to
               do about it). This cannot be derived: it is a judgement about OUR code, not about OpenFOAM.

Keeping them apart is the point. A hand-written "OpenFOAM does X" claim rots silently; a derived one cannot.

Usage:
    python3 tools/of_manifest.py simpleFoam > manifest/simpleFoam.yaml
    python3 tools/of_manifest.py simpleFoam --check     # non-zero if DERIVED half has drifted
"""
import argparse
import os
import re
import subprocess
import sys

OFSCAN = os.environ.get("OFSCAN_ROOT", os.path.join(os.path.dirname(__file__),
                                                    "..", "..", "ofscan"))
OF = os.environ.get("FOAM_ROOT", "/usr/lib/openfoam/openfoam2412")

# --------------------------------------------------------------------------------------------------
# CURATED half: classification + status. One entry per component of the solver's closure.
#
# classification (what kind of thing it is):
#   HOST_ONLY            runs once on the host; no GPU form is meaningful
#   CONFIGURATION        dictionary reading / control state
#   DISPATCH             runtime selection of an implementation
#   SHARED_NUMERICAL     finite-volume operator or matrix operation, solver-independent
#   MODEL                turbulence / transport / thermo model
#   BOUNDARY_CONDITION   an fvPatchField implementation
#   LINEAR_SOLVER        lduMatrix solver / preconditioner / smoother
#   GPU_REQUIRED         must be device-resident for the solver to be GPU-native
#   DYNAMIC_OR_UNRESOLVED  ofscan cannot resolve it statically
#
# brae_status (what we do about it):
#   REUSE_EXISTING       lift as-is; it is already solver-independent and validated
#   REVALIDATE_EXISTING  lift, but re-prove against OpenFOAM before trusting it
#   REIMPLEMENT          write again on the new architecture
#   NOT_REQUIRED_ON_GPU  host-side only
#   UNSUPPORTED          explicitly out of scope; the solver must REFUSE, not ignore
# --------------------------------------------------------------------------------------------------
COMPONENTS = {
    "simpleFoam": [
        # ---- orchestration -------------------------------------------------------------------
        dict(name="simpleFoam_main", of_symbol="main",
             of_file="applications/solvers/incompressible/simpleFoam/simpleFoam.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/simpleFoam_cpp.cu",
             brae_cuda="src/applications/solvers/simpleFoam/simpleFoam.cu",
             brae_target="src/applications/solvers/simpleFoam/simpleFoam.cu",
             validation="tests/test_simple_step_cpp.cu -- END-TO-END, one SIMPLE iteration composed of the "
                        "_cpp components vs OpenFOAM dumpSimpleStep (validation/matrixDumpSimple/step.dat): "
                        "p 2.5e-11, U 1.6e-12, phi 1.2e-11, every boundary patch <= 3.5e-13. Gate set at "
                        "1e-9, not the 1e-5 the older step test uses. "
                        "CUDA DRIVER: tests/test_simple_step_cuda.cu runs the device driver and the _cpp "
                        "driver for one iteration from the same fields, laminar and turbulent -- U/p/phi "
                        "agree to 4.1e-09 or better, and the pre-solve p residual to 2.6e-12 ABSOLUTE. "
                        "The gate is 1e-7 because the two paths run DIFFERENT Krylov methods (host "
                        "GAMG/BiCGStab vs device AMG-PCG/BiCGStab); the 1e-16 arithmetic gates are "
                        "test_ueqn_cuda and test_peqn_cuda.",
             note="The _cpp driver owns NO numerics -- 9 calls into shared components, each with its own "
                  "OpenFOAM provenance and test. Replaces a 3578-line file that pimpleFoam, rhoSimpleFoam "
                  "and five common/ headers all included. NOTE ON THE FIXTURE: matrixDumpSimple's "
                  "fvSolution sets `consistent yes`, but step.dat was dumped with plain SIMPLE; the test "
                  "now asserts explicitly that the flag is OFF for that comparison, since the refusal that "
                  "used to enforce it is gone. A SIMPLEC oracle is still needed to test SIMPLEC at "
                  "step granularity; end to end it is covered by ctest stock_pitzdaily_vs_openfoam."),
        dict(name="createFields", of_symbol="createFields.H",
             of_file="applications/solvers/incompressible/simpleFoam/createFields.H",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/createFields_cpp.cu",
             brae_target="src/applications/solvers/simpleFoam/createFields.cu",
             validation="tests/test_simple_step_cpp.cu -- phi READ from disk (not recomputed), "
                        "needReference() false on a case whose outlet fixes p, so no reference cell is "
                        "set and adjustPhi does not run.",
             note="p and U are MUST_READ; phi comes from createPhi.H (READ_IF_PRESENT, else "
                  "fvc::flux(U)) -- the read-if-present half was a past defect. setRefCell REFUSES when a "
                  "reference is needed and neither pRefCell nor pRefPoint is given, rather than quietly "
                  "pinning cell 0; pRefPoint is refused outright (needs mesh.findCell)."),
        dict(name="UEqn", of_symbol="UEqn.H",
             of_file="applications/solvers/incompressible/simpleFoam/UEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/UEqn_cpp.cu",
             brae_cuda="src/applications/solvers/simpleFoam/UEqn.cu",
             brae_target="src/applications/solvers/simpleFoam/UEqn.cu",
             validation="tests/test_ueqn_cpp.cu -- validated by DECOMPOSITION against "
                        "validation/matrixDumpAsym/momentum.dat: the div/laplacian core matches OpenFOAM to "
                        "1e-11 on diag/upper/lower/source; adding divDevReff provably changes only `source`, "
                        "and by exactly the explicit dev2 term; relaxation raises |diag| and leaves the "
                        "off-diagonals alone; MRF and fvOptions both throw. "
                        "CUDA: tests/test_ueqn_cuda.cu compares the device assembly to the reference "
                        "field by field on a laminar AND a turbulent case -- relaxed diag 2.0e-16/2.7e-16, "
                        "upper/lower 0, sources <=1.4e-15, all six boundary-coefficient arrays (25010 "
                        "faces) EXACTLY 0, addPressureGradient <=1.3e-15. MRF/fvOptions refused on the "
                        "device path too.",
             note="24 lines in OpenFOAM. The _cpp reference REFUSES MRF and fvOptions rather than ignoring "
                  "them -- brae has shipped a solver that silently ignored MRFProperties and produced a "
                  "converged wrong answer. The CUDA side (UEqn.cu) mirrors it stage for stage, refusals "
                  "included, and carries `bounded` and the `corrected` laplacian."),
        dict(name="pEqn", of_symbol="pEqn.H",
             of_file="applications/solvers/incompressible/simpleFoam/pEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/pEqn_cpp.cu",
             brae_cuda="src/applications/solvers/simpleFoam/pEqn.cu",
             brae_target="src/applications/solvers/simpleFoam/pEqn.cu",
             validation="tests/test_peqn_cpp.cu -- stage by stage on validation/matrixDumpAsym: rAU and "
                        "HbyA vs OpenFOAM A()/H() (ops.dat) to 1e-11; the pressure Laplacian incl. all "
                        "patch coefficients vs peqn.dat to 1e-11; source == laplacian source + "
                        "div(phiHbyA)*V; setReference asserted to be exactly fvMatrix.C:1011-1023 (it "
                        "DOUBLES the diagonal, it does not overwrite it); correctFlux analytic at p=0; "
                        "relaxField analytic; MRF/fvOptions/consistent all refused. "
                        "CUDA: tests/test_peqn_cuda.cu compares the device stages to the reference on a "
                        "laminar AND a turbulent case -- rAU 1.7e-16, HbyA <=5.2e-16, phiHbyA int/bnd "
                        "2.9e-16/7.4e-17, laplacian upper/lower 0 diag 1.9e-16, source 1.3e-14, patch "
                        "coeffs <=2.8e-16, setReference 1.9e-16, flux correction 2.9e-16, p.relax "
                        "1.8e-16, corrector <=5.2e-16. All three refusals fire on the device path.",
             note="50 lines. Every intermediate is RETURNED rather than kept local, so the first divergent "
                  "stage can be isolated -- a past investigation ended at `phi = phiHbyA - pEqn.flux()`, "
                  "which is stage 7 here. SIMPLEC (`consistent`) is refused: it needs UEqn.H1() and "
                  "fvc::snGrad, neither ported. The CUDA side (pEqn.cu) mirrors it stage for stage; "
                  "assemblePEqn takes p because the `corrected` laplacian needs grad(p), and REFUSES a "
                  "null p rather than treating it as `no correction`."),

        # ---- control -------------------------------------------------------------------------
        dict(name="simpleControl", of_symbol="Foam::simpleControl",
             of_file="src/finiteVolume/cfdTools/general/solutionControl/simpleControl/simpleControl.C",
             classification="CONFIGURATION", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/cfdTools/general/solutionControl/simpleControl/"
                            "simpleControl_cpp.cu",
             brae_target="src/finiteVolume/cfdTools/general/solutionControl/simpleControl/",
             schema_for="solutionControl",
             validation="tests/test_simple_step_cpp.cu -- parses the real matrixDumpSimple SIMPLE block: "
                        "`consistent yes`, nNonOrthogonalCorrectors 0, three residualControl entries "
                        "including the regex key \"(k|epsilon|omega|f|v2)\" which matches 'epsilon' and "
                        "not 'T'; correctNonOrthogonal runs nNonOrth+1 times and resets.",
             note="loop() = setFirstIterFlag; read(); if(initialised && criteriaSatisfied) writeAndEnd(); "
                  "else storePrevIterFields(); return runTime.loop(). Reads its keys from "
                  "solutionDict().subOrEmptyDict('SIMPLE')."),
        dict(name="relaxationFactors", of_symbol="Foam::solution::relaxField/relaxEquation",
             of_file="src/OpenFOAM/matrices/solution/solution.C",
             classification="CONFIGURATION", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/common/solver_controls.cuh",
             brae_target="src/OpenFOAM/matrices/solution/",
             note="Legacy flat form promotes only p*/rho* to FIELD relaxation; select() appends 'Final' on "
                  "the final iteration. Both were past defects -- revalidate, do not assume."),

        # ---- shared numerics -----------------------------------------------------------------
        dict(name="fvm_div", of_symbol="Foam::fvm::div",
             of_file="src/finiteVolume/finiteVolume/divSchemes/divScheme/divScheme.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvm.cu",
             brae_target="src/finiteVolume/finiteVolume/divSchemes/",
             note="Implicit weights come from the SCHEME. limitedSurfaceInterpolationScheme::weights = "
                  "limiter*CDweights + (1-limiter)*pos0(faceFlux); LUST = 0.75*linear + 0.25*upwind. "
                  "Getting the implicit blend wrong is silent -- it was brae's LUST defect."),
        dict(name="fvm_laplacian", of_symbol="Foam::fvm::laplacian",
             of_file="src/finiteVolume/finiteVolume/laplacianSchemes/laplacianScheme/laplacianSchemes.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvm.cu",
             brae_target="src/finiteVolume/finiteVolume/laplacianSchemes/"),
        dict(name="fvc_grad", of_symbol="Foam::fvc::grad",
             of_file="src/finiteVolume/finiteVolume/gradSchemes/gaussGrad/gaussGrad.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvc.cu",
             brae_reference="src/finiteVolume/finiteVolume/fvc.cu",
             brae_target="src/finiteVolume/finiteVolume/gradSchemes/gaussGrad/",
             note="A host std::vector reference ALREADY exists (fvc.cu:6 gaussGrad) -- this is the _cpp "
                  "oracle for grad, already written. Move it, do not rewrite it."),
        dict(name="fvc_div", of_symbol="Foam::fvc::div",
             of_file="src/finiteVolume/finiteVolume/fvc/fvcDiv.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvc.cu",
             brae_reference="src/finiteVolume/finiteVolume/fvc.cu",
             brae_target="src/finiteVolume/finiteVolume/fvc/"),
        dict(name="fvc_flux", of_symbol="Foam::fvc::flux",
             of_file="src/finiteVolume/finiteVolume/fvc/fvcFlux.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvc.cu",
             brae_target="src/finiteVolume/finiteVolume/fvc/"),
        dict(name="fvc_snGrad", of_symbol="Foam::fvc::snGrad",
             of_file="src/finiteVolume/finiteVolume/snGradSchemes/snGradScheme/snGradScheme.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_target="src/finiteVolume/finiteVolume/snGradSchemes/",
             note="Only reached on the `consistent` (SIMPLEC) branch of pEqn.H."),
        dict(name="fvMatrix_A", of_symbol="Foam::fvMatrix<Type>::A",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1314,
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_simple.cu",
             brae_reference="src/finiteVolume/fvMatrices/fv_matrix_ops.cuh",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="Host reference exists at fv_matrix_ops.cuh:138 (matrixA)."),
        dict(name="fvMatrix_H", of_symbol="Foam::fvMatrix<Type>::H",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1333,
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_simple.cu",
             brae_reference="src/finiteVolume/fvMatrices/fv_matrix_ops.cuh",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="Host reference exists at fv_matrix_ops.cuh:155 (matrixH)."),
        dict(name="fvMatrix_H1", of_symbol="Foam::fvMatrix<Type>::H1",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1385,
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="Only used by the `consistent` (SIMPLEC) branch -- AND IT HAS TO REACH THE COUPLED PATCHES, which is where brae's did not. OF H1() is `lduMatrix::H1()` (minus the internal off-diagonal row sum) PLUS boundaryCoeffs summed over every patch with ptf.coupled(), all divided by V. brae's row sum handled the cyclic interface and stopped there, so on a mesh coupled only by a cyclicAMI it was the internal faces alone. THE SECOND HALF, and the larger one: pEqn.H's `phiHbyA += fvc::interpolate(rAtU - rAU)*fvc::snGrad(p)*mesh.magSf()` is a surfaceScalarField whose COUPLED boundary values are evaluated like any other patch's, while brae built it from deviceMatrixFluxInternal (internal faces) and deviceMatrixFluxBoundary (NON-coupled patches) -- a cyclic or AMI face is in neither list, so the interface flux never received the term at all. NEITHER HALF IS RIGHT ALONE: with only the H1 half the converged U got WORSE, 8.253e-02 -> 9.282e-02, because rAtU became more accurate while the flux it must stay consistent with did not; with only the flux half the seeded continuity error stopped at 4.560e-03 against 1.706e-03 for both. UNREACHABLE UNTIL NOW: no case in validation/ combined SIMPLEC with a coupled interface -- every cyclic/AMI case ran plain SIMPLE and every SIMPLEC case was uncoupled -- so validation/implicitAMI was added with the tutorial's mesh. FOUND by scanning the OpenFOAM tutorial tree for a coupled patch plus a div scheme brae assembles as a blend, which turned up implicitAMI; the first scan returned nothing because it looked for constant/polyMesh/boundary in a tree whose meshes are not generated. A REMAINING 5.8e-04 pressure residual at OpenFOAM's converged state is not yet explained.",
             brae_driver='src/applications/solvers/simpleFoam/device_simple_foam.cu (the SIMPLEC rAtU row sum, and the SIMPLEC flux correction on the interface faces)',
             validation="tests/simplec_interface_vs_openfoam.sh on OpenFOAM's own basic/simpleFoam/implicitAMI (validation/implicitAMI), registered as ctest simplec_interface_vs_openfoam. Converged U against real OpenFOAM went 8.253e-02 -> 1.692e-02 and p 2.428e-01 -> 4.113e-02. The SHARP check needs no field comparison: seeded with OpenFOAM's converged U, p and phi -- a state whose true divergence is zero -- brae's own continuity error went 1.284e-01 -> 1.706e-03 and its pressure residual 3.363e-02 -> 5.797e-04, while the momentum residual stayed at 2.752e-07 throughout. Run against a rebuilt pre-fix binary the gate fails check 1 on both fields and check 3 on pressure and continuity, and PASSES the momentum line and the whole SIMPLEC-off branch."),
        dict(name="fvMatrix_relax", of_symbol="Foam::fvMatrix<Type>::relax",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1102,
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/fvMatrices/fv_matrix_ops.cuh",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="ASYMMETRIC: adds cmptMax(cmptMag(iCoeffs)) to the diagonal and subtracts "
                  "cmptMin(iCoeffs) from the source. Guarded by if(relaxEquation(name))."),
        dict(name="fvMatrix_setReference", of_symbol="Foam::fvMatrix<Type>::setReference",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1011,
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/common/solver_controls.cuh",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="Currently lives inside applications/solvers/common -- solver-owned infrastructure, "
                  "exactly the layering defect this rebuild removes."),
        dict(name="fvMatrix_flux", of_symbol="Foam::fvMatrix<Type>::flux",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/simpleFoam/device_simple_foam.cu",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="phi = phiHbyA - pEqn.flux() is the continuity-preserving step; a past investigation "
                  "traced a growing divergence to it."),

        # ---- cfdTools free functions ---------------------------------------------------------
        dict(name="constrainHbyA", of_symbol="Foam::constrainHbyA",
             of_file="src/finiteVolume/cfdTools/general/constrainHbyA/constrainHbyA.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/simpleFoam/simple_foam.cuh",
             brae_target="src/finiteVolume/cfdTools/general/constrainHbyA/",
             note="Solver-owned today; must become shared."),
        dict(name="adjustPhi", of_symbol="Foam::adjustPhi",
             of_file="src/finiteVolume/cfdTools/general/adjustPhi/adjustPhi.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/common/solver_controls.cuh",
             brae_target="src/finiteVolume/cfdTools/general/adjustPhi/"),
        dict(name="constrainPressure", of_symbol="Foam::constrainPressure",
             of_file="src/finiteVolume/cfdTools/general/constrainPressure/constrainPressure.C",
             classification="SHARED_NUMERICAL", status='PORTED',
             brae_existing='src/finiteVolume/fields/fv_patch_field.cuh (FixedFluxPressurePatchField) + the transcribed loop in simpleFoam/pEqn_cpp.cu (pEqn.H:21), rhoSimpleFoam/rhoPEqn_cpp.cu (pEqn.H:12) and rhoPcEqn_cpp.cu (pcEqn.H:16)',
             brae_target="src/finiteVolume/cfdTools/general/constrainPressure/",
             validation="validation/ffp_vs_openfoam.sh on validation/rhoBoxP, twice (SIMPLE and SIMPLEC, the second reaching rhoPcEqn's rhorAtU divisor). The gate compares the solver-set boundary snGrad against the gradient OpenFOAM WRITES into the output p file: 2.5e-06 on max|OF| 0.7347 (SIMPLE), 1.0e-08 on 0.1198 (SIMPLEC), plus p 2.8e-12 / T 1.6e-11 / U 4.1e-09 field norms at the same iteration. Non-vacuity is asserted, not assumed: the fixture pairs fixedFluxPressure with an ASSIGNABLE-U outlet (inletOutlet) because constrainHbyA makes the numerator cancel X-minus-X at every fixed-velocity patch, so the arm refuses any fixture whose converged gradient is ~0. Fail-proof: the old silent zeroGradient mapping restored -> the arm never engages (script guard) AND the field norms fail. The CUDA half is deviceConstrainPressure (snGradMask on DeviceBoundary, refGrad overwritten per assembly), gated by the same script's cuda arm: device-vs-host p ~1e-13/iteration over 60 iterations. What it does NOT claim: MRF inside constrainPressure is refused with MRF itself.",
             note="The BC is fixedGradient plus a solver contract: OF's updateCoeffs FATALS if updateSnGrad was not called this time index (fixedFluxPressureFvPatchScalarField.C:150-163), and FixedFluxPressurePatchField mirrors that refusal at coefficient time -- so a driver that builds the patch and never runs constrainPressure refuses instead of silently assembling the stale gradient, which is what the old factory mapping to zeroGradient did. snGrad = (phiHbyA_b - rho_b*(Sf_b & U_b))/(magSf_b * rhorAU_b), with the divisor rhorAUf (surface) in pEqn.H, rhorAtU (vol, owner-cell boundary) in pcEqn.H, and rAtU alone incompressibly."),
        dict(name="setRefCell", of_symbol="Foam::setRefCell",
             of_file="src/finiteVolume/cfdTools/general/findRefCell/findRefCell.C",
             classification="CONFIGURATION", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/common/solver_controls.cuh",
             brae_target="src/finiteVolume/cfdTools/general/findRefCell/"),
        dict(name="continuityErrs", of_symbol="continuityErrs.H",
             of_file="src/finiteVolume/cfdTools/incompressible/continuityErrs.H",
             classification="HOST_ONLY", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/simpleFoam/gpuSimpleFoam.cu",
             brae_target="src/finiteVolume/cfdTools/incompressible/"),

        # ---- models --------------------------------------------------------------------------
        dict(name="turbulenceModel_New", of_symbol="Foam::incompressible::turbulenceModel::New",
             of_file="applications/solvers/incompressible/simpleFoam/createFields.H", of_line=42,
             classification="DISPATCH", status="REIMPLEMENT",
             selection_base="incompressible::turbulenceModel",
             brae_target="src/TurbulenceModels/",
             note="26 implementations in v2412. brae supports a strict subset -- the manifest must say "
                  "which, and the solver must REFUSE the rest rather than substitute."),
        dict(name="divDevReff", of_symbol="Foam::...::divDevReff",
             of_file="src/TurbulenceModels/turbulenceModels/linearViscousStress/"
                     "linearViscousStress.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_divdevreff.cu",
             brae_reference="src/TurbulenceModels/turbulenceModels/linearViscousStress/"
                            "linearViscousStress_cpp.cu",
             brae_target="src/TurbulenceModels/turbulenceModels/linearViscousStress/",
             validation="tests/test_divdevreff_cpp.cu -- 7.4e-16 vs OpenFOAM dumpDivDevReff over 12225 "
                        "cells (validation/kEpsCorrect), with a wrong-sign check and a wall-nuEff "
                        "negative control that both fire",
             note="FIRST component extracted onto the mirrored architecture; the template for the rest. "
                  "The _cpp reference is host-only and reuses brae existing transpose/dev2/operator* "
                  "rather than restating them. SPEED, FP-10 (2026-09-13): the device form copied each "
                  "velocity-gradient component into the 9*nC tensor with a BLOCKING device-to-device "
                  "cudaMemcpy -- nine per momentum assembly -- where deviceGradU and "
                  "deviceLeastSquaresGradU next to it use the async form on the per-thread stream. Now "
                  "async: same bytes, same order, same bits. It did not move the clock (squareBend, 200 "
                  "iterations: 4.42/4.44/4.52 s against 4.41/4.43), because a blocking copy's cost is "
                  "mostly waiting for work the GPU has to do anyway -- but it was the ONE operation "
                  "that made this assembly impossible to capture into a CUDA graph, since a blocking "
                  "copy is illegal during stream capture. FP-10 THEN CAPTURED THAT ASSEMBLY (2026-09-13) "
                  "and the capture needed three more things, each found by a failure: this function's "
                  "sixteen temporaries had to stop coming from the device pool (they are a per-mesh "
                  "workspace now) and its three std::move handovers had to stop swapping buffer "
                  "pointers (aliased instead); rhoUEqn.cu's eighteen temporary sites likewise; and the "
                  "MomentumMatrix, constructed fresh every iteration and EMPTY on entry, had to become "
                  "persistent -- with a fresh one the memory checker catches axpyKernel reading 0x100 "
                  "on the first replay. All three are bit-identical (squareBend, injectorPipe, "
                  "aerofoilNACA0012: every residual line and all nine written fields against a build "
                  "without them), and tests/rho_capture_assembly_identity.sh holds the captured arm to "
                  "the direct one. The capture is worth 38 launches and 0.17 ms of host API time per "
                  "iteration, not the 3.7 ms first estimated: the assembly is 38 launches where the "
                  "phase is 144, the rest being its solve, and most of an iteration's 1,003 launches "
                  "are solver loops whose trip count depends on a residual the host reads."),
        dict(name="singlePhaseTransportModel", of_symbol="Foam::singlePhaseTransportModel",
             of_file="src/transportModels/incompressible/singlePhaseTransportModel/"
                     "singlePhaseTransportModel.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_target="src/transportModels/incompressible/",
             note="laminarTransport.correct() is called every SIMPLE iteration; a Newtonian model makes it "
                  "a no-op, a non-Newtonian one does not."),
        dict(name="SpalartAllmaras", of_symbol="Foam::RASModels::SpalartAllmaras",
             of_file="src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/"
                     "SpalartAllmaras.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_spalart.cu",
             schema_for="SpalartAllmarasBase",
             brae_target="src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/",
             brae_reference="src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/"
                            "SpalartAllmaras_cpp.cu",
             validation="tests/sa_cpp_vs_openfoam.sh, END TO END on validation/airFoil2D against real "
                        "OpenFOAM's converged 500: U 6.2e-05, p 7.5e-05, nuTilda and nut 1.3e-02 "
                        "L2-relative, with brae's own residuals at OpenFOAM's level (U 4.7e-06 against "
                        "5.4e-06, p 6.6e-05 against 7.5e-05, nuTilda 9.6e-04 against 1.2e-03) and max "
                        "nuTilda 0.3842 against 0.3803. The control is laminar and fails at 1.2e-01 / "
                        "3.5e-01 / 1.0e+00, so the gate measures the closure.",
             note="All 12 coefficients are read via dimensioned<scalar>::getOrAddToDict -- see the derived "
                  "schema below. nut = nuTilda*fv1 on `calculated` patches was a past defect. "
                  "THE DEFECT THAT MATTERED was not in the SA equation at all -- that was within 13% of "
                  "OpenFOAM's residual from the first run. It was nut's WALL value: correctNut writes "
                  "nut_ = nuTilda*fv1 as a FIELD ASSIGNMENT, nuTilda is fixedValue ZERO at a wall, so the "
                  "assignment leaves nut_wall = 0 -- and OpenFOAM's correctBoundaryConditions() then lets "
                  "nutUSpaldingWallFunction overwrite it with Spalding's law (~4.5e-03 here). Taking the "
                  "assignment removes the wall's whole eddy viscosity and with it the wall shear; 25% of "
                  "the momentum residual sat on 78 wall faces, and fixing it moved the end-to-end "
                  "agreement 268x on U, 598x on p and 35x on nuTilda in ONE step. Two boundary defects "
                  "were fixed on the way (the freestream valueFraction was never recomputed from the flow "
                  "angle, and mixed evaluate() took refValue instead of OF's lerp blend), worth 372x -> "
                  "47x on momentum and 3551x -> 136x on pressure between them. RULED OUT along the way, "
                  "each by measurement: the coefficients (OpenFOAM prints them, all match, ft2 inactive); "
                  "Omega = sqrt(2)*mag(skew(gradU)) on a unit shear; the wall distance (meshWave against "
                  "brute force, 1.0e-04 mean over 10720 cells); and the linearUpwind correction's sign "
                  "(upwind 0.162, brae 0.250, flipped 1.72, OpenFOAM 0.380). "
                  "CUDA: PORTED but NOT YET MATCHING the _cpp. Five modules, each measured on airFoil2D "
                  "at OpenFOAM's converged 500 (OF: U 5.36e-06, p 7.51e-05, nuTilda 1.23e-03): (1) the "
                  "model on the k slot -- U 4.56e-03, p 1.00e-01, nuTilda 1.85e-02; (2) linearUpwind on "
                  "div(phi,nuTilda) -- nuTilda 7.67e-03; (3) the freestream valueFraction "
                  "(deviceUpdateMixedFreestream) -- U 2.11e-03, p 1.35e-02; (4) the Spalding wall nut -- "
                  "U 1.34e-04, p 1.88e-03; (5) inletOutlet resolved per iteration. END TO END it reaches "
                  "U 1.2e-02, p 3.3e-02, nuTilda 3.4e-01 against the _cpp's 6.2e-05 / 7.5e-05 / 1.3e-02, "
                  "so it is NOT gated yet. The SA terms themselves are NOT the cause: dumped cell by cell "
                  "(BRAE_DUMP_SA vs BRAE_DUMP_SA_CPP), gradNt2 is identical to 2.4e-11 and every formula "
                  "matches. CUDA IS NOW GATED (tests/sa_cuda_vs_openfoam.sh): END TO END from 0/ it "
                  "reaches U 5.99e-05, p 5.24e-05, nuTilda and nut 1.22e-02 against OpenFOAM's converged "
                  "500 -- matching the _cpp reference (6.20e-05 / 7.54e-05 / 1.28e-02) -- with a laminar "
                  "control that fails at 1.20e-01 / 3.54e-01. "
                  "THE COLD START was the whole of it, and no fixed-point comparison could have found it: "
                  "assembled at OpenFOAM's own fields the two momentum matrices are BIT-IDENTICAL "
                  "(tests/ueqn_localize.cu, 6.904985e-05 on both with the same patch split, at t=0 as well "
                  "as t=500), so the difference was never in the assembly but in the nuEff BOUNDARY handed "
                  "to it. Three defects there: nut = nuTilda*fv1 is a FIELD ASSIGNMENT and the device took "
                  "the adjacent cell instead (2.88e-01 out at the outlet); deviceBoundaryNutSpalding "
                  "rewrites EVERY face and replaced the non-wall ones with the cell value unless handed "
                  "`nutFile`; and deviceBoundaryNut overwrote the boundary buffer each iteration with cell "
                  "values, DESTROYING the previous wall nut that Spalding's Newton warm-starts from. "
                  "OpenFOAM seeds it from nut_ itself; a cold start with 10 iterations and a 1% early-out "
                  "settled 14% low, and that alone took the run from U 1.20e-02 to 5.99e-05 -- from 200x "
                  "worse than the _cpp to matching it. RULED OUT by measurement along the way: the solver "
                  "tolerances (tightening every solve from the case's relTol 0.1 to relTol 0 leaves the "
                  "answer bit-identical), the wall distance (nearWallDist and 1/deltaCoeffs coincide here "
                  "at 9.027330e-02), and the linearUpwind correction (forcing upwind moves the momentum "
                  "residual 1.34e-04 -> 7.99e-03, so it is applied)."),
        dict(name="limitedSnGrad", of_symbol="Foam::fv::limitedSnGrad",
             of_file="src/finiteVolume/finiteVolume/snGradSchemes/limitedSnGrad/limitedSnGrad.C",
             classification="SHARED_NUMERICAL", status="PORTED",
             brae_existing="src/cuda/device_fvm.cu (deviceLaplacianCorrFluxLimited)",
             brae_reference="src/finiteVolume/finiteVolume/fvm.cuh (laplacianCorrFlux, limitCoeff)",
             brae_target="src/finiteVolume/finiteVolume/",
             validation="tests/limitedsngrad_vs_openfoam.sh, on validation/airFoil2D (a genuinely "
                        "non-orthogonal C-grid, so the correction is not a rounding term). Three "
                        "assertions: `limited 1` reproduces `corrected` BIT-FOR-BIT on both paths "
                        "(6.904985e-05), `limited 0.33` changes the answer (7.753060e-05) so the scheme "
                        "is not inert, and host and device agree bit-for-bit at 0.33.",
             note="limiter = min( k*|orth| / ((1 - k)*|corr| + SMALL), 1 ) applied to the non-orthogonal correction, with |orth| the ORTHOGONAL part of the same snGrad (nonOrthDeltaCoeffs*(vf[nei] - vf[own])) and both magnitudes taken BEFORE gamma*magSf. `limited 1` is exactly `corrected` and `limited 0` is `uncorrected`. THE VECTOR CASE is where this is easy to get wrong and where brae's pre-existing device implementation did: OF's limitedSnGrad<Type> takes mag() of the WHOLE snGrad and of the WHOLE correction, so all three velocity components share ONE per-face limiter. The device limited each component independently -- a different scheme, 0.6% out on airFoil2D -- and only writing the host reference and comparing the two exposed it (deviceLaplacianCorrFluxLimitedVec is the fix). Needed by turbineSiting (`Gauss linear limited corrected 0.33`), which now runs end to end -- see actuationDiskSource and atmBoundaryLayer."),

        dict(name="rotorDiskSource", of_symbol="Foam::fv::rotorDiskSource",
             of_file="src/fvOptions/sources/derived/rotorDiskSource/rotorDiskSource.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/cfdTools/fvOptions/rotor_disk.cu",
             brae_reference="src/finiteVolume/cfdTools/general/fvOptions/rotorDiskSource_cpp.cu",
             brae_target="src/finiteVolume/cfdTools/general/fvOptions/",
             validation="tests/rotordisk_vs_openfoam.sh, on validation/rotorDisk -- the tutorial on its "
                        "REAL 71734-cell snappyHexMesh, run to its own convergence at t=224. OpenFOAM's "
                        "rotorDiskSource PRINTS its own answer, so the oracle is unusually direct: "
                        "min/max(AOA) -61.412/-7.44884 against -61.413193/-7.4489333, Effective drag "
                        "-130.396343 against -130.399890, Effective lift 1230.695836 against 1230.716500 "
                        "-- all within 2.7e-05, which is the one-iteration offset between OpenFOAM's "
                        "printed value (start of iteration 224) and the written t=224 field. The gate "
                        "also compares the DEVICE force against the host one (5.6e-16) and asserts the "
                        "lift has OpenFOAM's sign.",
             note="Froude blade-element momentum. Scoped, and refused rather than approximated outside it: geometryMode `specified`, fixedTrim, no coning (Rcone = I), one lookup profile, inletFlowType local|fixed. THE SIGN is the thing worth stating: OF's addSup is `eqn -= force` with force carrying eqn's dimensions PER VOLUME (calculate divides by V), and fvMatrix::operator-= is source() += V*su -- so the extensive source GAINS the raw force. A body force that pushes the wrong way still converges, it just converges to the wrong answer, so nothing about a residual would have caught it. brae's device rotorDisk existed with NO test at all and a header comment describing the opposite (`relaxSrc -= force`); the host port plus this gate established that the device FORCE is right to 5.6e-16 and that the comment refers to the solver's application, not the force. Unblocks the rotorDisk tutorial (its other two blockers, linearUpwind on div(phi,k) and div(phi,omega), were cleared by the turbulence-scheme work). APPLICATION SIGN, corrected 2026-08-20: addSup is `eqn -= force` with force PER VOLUME, so the OPTION matrix's source gains V*force; simpleFoam then writes `UEqn == fvOptions(U)` and the free operator== is `UEqn - fvOptions(U)`, so the MOMENTUM source LOSES it. The V2 driver added it instead. tests/rotordisk_vs_openfoam.sh could not see this -- it evaluates the blade-element force at OpenFOAM's OWN converged velocity, and drag, lift and AOA are identical whichever way the force is then applied. Measured end to end on the rotorDisk tutorial at deep convergence: U L2 1.04e-02 with the minus against 9.76e-01 with the plus, which also fails to converge at all (residual 0.13 after 800 iterations)."),

        dict(name="cellLimitedGrad", of_symbol="Foam::fv::cellLimitedGrad",
             of_file="src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/cellLimitedGrad/"
                     "cellLimitedGrad.C",
             classification="SHARED_NUMERICAL", status="PORTED",
             brae_existing="src/cuda/device_mesh.cu (deviceCellLimitGrad)",
             brae_reference="src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/"
                            "cellLimitedGrad_cpp.cu",
             brae_target="src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/",
             validation="tests/test_celllimited_cpp.cu -- the new host port against the already "
                        "OpenFOAM-validated device one, 3.2e-15 (scalar grad(p)) and 3.7e-16 (vector "
                        "grad(U)) on validation/pitzDailyTurb, with a check that the limiter actually "
                        "bites (5.8e+03) and a TRANSPOSED control that disagrees by 1.005. "
                        "tests/celllimited_vs_openfoam.sh then measures it on the case it unblocks: "
                        "assembling the momentum equation at OpenFOAM's own t=400 on "
                        "validation/windAroundBuildingsBox gives Ux 1.53e-04 (host) and 1.50e-04 "
                        "(device) against OpenFOAM's 1.09e-04 -- 1.40x and 1.37x -- where the SAME "
                        "assembly with the limiter off gives 2.98e-02, i.e. 272x. On the real 185237-cell "
                        "snappyHexMesh the same measurement is 1.33x limited against 6.25x unlimited.",
             note="`cellLimited Gauss linear <k>` scales the base Gauss gradient per cell and PER "
                  "COMPONENT so extrapolating to any of the cell's own face centres cannot overshoot its "
                  "neighbours' range. Three places it is easy to get wrong. (1) limitFaceCmpt RETURNS "
                  "when |extrapolate| <= SMALL -- no constraint at all, which is NOT the same as r = 1. "
                  "(2) The boundary faces contribute the patch VALUE to the range (a coupled patch its "
                  "neighbour field); omitting them lets the gradient overshoot exactly where the field is "
                  "driven from outside. (3) For a vector the limiter is itself a VECTOR and scales COLUMN "
                  "j of grad(U), because OpenFOAM's grad(U)_ij is d(U_j)/d(x_i) -- reading it as a row is "
                  "the transposed control the unit test carries. k < SMALL disables the scheme; k < 1 "
                  "widens the band by (1/k - 1)*(maxDelta - minDelta). WHY IT MATTERS: linearUpwind's "
                  "deferred correction is built from the gradient the scheme NAMES, and that correction "
                  "does not vanish at convergence -- so running the plain Gauss gradient under a limited "
                  "name is a different equation, not a slower one. That is what windAroundBuildings was "
                  "refused for. SPEED, FP-4 (bench/rhoSimpleFoam/FASTPATH.md, 2026-09-12): this limiter "
                  "is a THIRD of the momentum phase on gasMixing/injectorPipe -- 4.1 ms/it with the "
                  "tutorial's schemes, 2.7 with grad(U) unlimited, 2.5 with upwind divergence as well "
                  "(three runs of 100 iterations each), and 12 launches at 1.90 of the iteration's 13.73 "
                  "GPU ms. deviceCellLimitGradFused now limits up to three fields in one launch, "
                  "bit-identical per field (tests/test_cell_limit_grad_fused.cu, ctest "
                  "cell_limit_grad_fused: memcmp for n=1,2,3 at k=1 and k=0.5, sheared and empty-patch "
                  "meshes, one-ulp cross-contamination controls, and a control that the limiter bites at "
                  "all; fail-proof RUN, 10 arms red). IT DID NOT MOVE THE CLOCK, and the measurements say "
                  "why: 12 launches became 8 and 1.90 ms became 1.74, inside run-to-run noise on the "
                  "iteration, because the cost is the SIX face-loop passes over scattered neighbour "
                  "values and fusing N fields removes none of them. Occupancy was excluded (92 registers "
                  "against 58; capping to 80 moved 0.780 ms to 0.771), and so was pass-sharing: a kernel "
                  "fusing the gradient's face passes with the limiter's range pass was written, held "
                  "bit-identical, measured at 1.026 -> 0.999 ms for the two sites it served, and reverted "
                  "-- the second pass was already L2-resident. What remains is the scheme's own gather "
                  "traffic."),

        dict(name="kEpsilon", of_symbol="Foam::RASModels::kEpsilon",
             of_file="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_kepsilon.cu",
             brae_reference="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu "
                            "(standalone host correct() + the KEResiduals oracle); "
                            "src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/k_epsilon.cuh "
                            "(the older host correct(), moved into the mirrored path)",
             brae_target="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/",
             validation="Coefficients checked against the DERIVED schema below -- brae defaults match "
                        "OpenFOAM exactly (Cmu .09, C1 1.44, C2 1.92, C3 0, sigmak 1.0, sigmaEps 1.3). "
                        "tests/test_simple_turbulent_cpp.cu wires it into the _cpp loop on "
                        "validation/pitzDailyTurb from OpenFOAM's converged 1576: the rebuilt loop is "
                        "BIT-IDENTICAL (rel 0) to the pre-existing OpenFOAM-validated host path on U, p, "
                        "phi, k, epsilon and nut, with a laminar control (drift 1.3e-01) proving nut "
                        "really reaches the momentum equation. tests/boundary_nut_vs_openfoam.sh "
                        "assembles k and epsilon at OpenFOAM's OWN converged 1576 fields and compares "
                        "the initial residuals against its log: epsilon 1.900e-07 vs 1.912e-07 (0.64%) "
                        "and k 3.655e-07 vs 3.680e-07 (0.68%), where reverting the patch diffusivity "
                        "gives 1.20e-05 (63x) and 6.44e-06 (17x).",
             note="REUSED, not rewritten: a complete host kEpsilon::correct already existed and is "
                  "validated against OpenFOAM (correct.dat). The new work is the COUPLING -- nuEff = "
                  "nu + nut with boundary values from nut's own boundary field, and the LAGGED ordering "
                  "(turbulence->correct() at the END of the iteration, simpleFoam.C:93-94). That same "
                  "boundary distinction applies a SECOND time, to the k and epsilon equations' own "
                  "diffusivity: OF builds nut by FIELD ASSIGNMENT (nut_ = Cmu*sqr(k_)/epsilon_), which "
                  "writes the boundary from the boundary k and epsilon, and correctBoundaryConditions() "
                  "leaves a `calculated` patch alone -- so DkEff(patchi)/DepsilonEff(patchi) carry "
                  "Cmu*k_b^2/eps_b, not the adjacent cell value. Interpolating the cell nut there put "
                  "90.5% of the whole epsilon residual on pitzDaily's inlet. realizableKE (variable "
                  "rCmu) and kOmegaSST (a1*k/max(a1*omega, b1*F2*sqrt(S2))) have different nut "
                  "expressions and are excluded from that evaluation."),
        dict(name="kOmegaSST", of_symbol="Foam::RASModels::kOmegaSST",
             of_file="src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_komega_sst.cu",
             brae_reference="src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST_cpp.cu",
             brae_target="src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/",
             validation="tests/sst_cpp_vs_openfoam.sh, on validation/pitzDailySST against real "
                        "OpenFOAM (log.simpleFoam, whose 2000/ this repo carries and which a rerun "
                        "reproduces bit-identically). FIXED POINT, one iteration from OpenFOAM's own "
                        "converged fields: omega 3.288e-04 vs 2.836e-04 (1.16x) and k 3.683e-05 vs "
                        "2.537e-05 (1.45x), with an upwind control at 8.3x and 82x that the gate "
                        "requires to breach its bound. END TO END from 0/ (sst_cpp_full): U 1.9e-04, "
                        "p 7.4e-04, k 9.2e-04, omega 9.4e-03, nut 2.4e-03 L2-relative against "
                        "OpenFOAM's converged fields -- at parity with the established CUDA solver on "
                        "the same case (1.7e-04 / 5.1e-04 / 6.8e-04 / 1.3e-02 / 3.9e-03). CUDA: "
                        "tests/sst_cuda_vs_openfoam.sh runs the SAME case through the V2 driver and "
                        "reaches U 1.05x, omega 1.16x and k 1.45x of OpenFOAM's initial residual at its "
                        "converged state -- the _cpp reference's own numbers to within 0.4% (U to seven "
                        "digits) -- and end to end U 2.2e-04, p 7.4e-04, k 9.3e-04, omega 2.2e-02, "
                        "nut 4.8e-03."
                " COMPRESSIBLE fvm::ddt under `Euler` (kOmegaSSTBase.C:572,602), the same term as kEpsilon's: tests/rho_komegasst_vs_openfoam.sh's Euler arm holds omega and k to 1e-13 against the instrumented model and reads omega D() 9.6e-04 / k source 3.0e-02 with the term zeroed. No shipped rhoSimpleFoam tutorial runs kOmegaSST under Euler, so this is gated on sbMatched with the scheme switched. CUDA against the _cpp reference inside the driver, tests/rho_step_cuda_euler.sh (rhoSST switched to Euler, 3 iterations): k 3.5e-13, omega 3.1e-12, nut 1.2e-11, the same floor as the steadyState control run beside it; with the device term withheld k 8.64e-05.",
             note="The _cpp reference could NOT run an SST case until this, and a single-iteration "
                  "probe did not show it: probing from the converged state gave 1.2-1.5x while the "
                  "same code from 0/ reached omega 1e+46 by iteration 200. Three defects, all of them "
                  "end-to-end-only. (1) div(phi,k)/div(phi,omega) were upwind where the tutorials ask "
                  "for `bounded Gauss limitedLinear 1` -- a different matrix, not a looser tolerance. "
                  "(2) A negative omega was FLOORED to SMALL instead of taking Foam::bound's "
                  "neighbour average; the next iteration divides CDkOmega by it, so a floored cell "
                  "contributes ~1e15. OpenFOAM bounds omega 258 times on this case, first at "
                  "min -2445.7, against brae's -2534.0 on the same iteration -- upwind never produces "
                  "the negative cell, which is why the floor survived until (1) was fixed. "
                  "(3) correctNut wrote only wall patches, so a `calculated` inlet shipped as "
                  "`value uniform 0` kept zero eddy viscosity for the whole run. The patch "
                  "diffusivity DkEff(patchi) = alphaK(F1)*nut_b + nu is now taken from nut's own "
                  "boundary as well; on THIS case that is worth 2.7% because the near-inlet cell nut "
                  "is close to the boundary value, unlike kEpsilon's pitzDaily where it is 12x off. "
                  "PORTED TO CUDA one module at a time against that working _cpp, testing after each: "
                  "(1) the div(phi,k)/div(phi,omega) scheme read from fvSchemes instead of hardcoded "
                  "upwind, which alone moved omega 19x -> 6.8x and k 125x -> 23x and also wired the "
                  "`bounded` prefix the kEpsilon path had been dropping; (2) the `calculated` nut "
                  "evaluation; (3) the patch diffusivity. (2) and (3) are inert at the fixed point by "
                  "construction and only the end-to-end mode exercises them. Foam::bound was ALREADY "
                  "correct on the device. Comparing module by module against the _cpp is what exposed "
                  "the wall mask: it keyed on isEpsilonWallFunction while omegaWallFunction mapped to a "
                  "plain zeroGradient field, so every kOmegaSST case on the rebuilt driver ran with NO "
                  "wall faces -- no wall nut, hence wrong wall shear and everything downstream, "
                  "measuring U 48x and omega 6.8x off OpenFOAM. The predicate is now "
                  "isTurbulenceWallFunction and both wall functions answer it."),

        dict(name="bound", of_symbol="Foam::bound",
             of_file="src/finiteVolume/cfdTools/general/bound/bound.C",
             classification="SHARED_NUMERICAL", status="PORTED",
             brae_reference="src/finiteVolume/cfdTools/general/bound/bound_cpp.cu",
             brae_target="src/finiteVolume/cfdTools/general/bound/",
             validation="Exercised by every RAS correct(); the discriminating case is "
                        "tests/sst_cpp_vs_openfoam.sh full mode, which diverges to omega 1e+46 with a "
                        "floor in place of it and converges to 9.4e-03 of OpenFOAM with it.",
             note="NOT a clamp. vsf = max(max(vsf, average(max(vsf, lowerBound))*pos0(-vsf)), "
                  "lowerBound): a cell that solved NEGATIVE takes the area-weighted average of its "
                  "neighbours, and only a merely-small cell is floored. Both lower bounds are SMALL, "
                  "so the floor and OpenFOAM agree everywhere except the negative cell -- which is "
                  "exactly the cell that matters. CUDA (kEpsilon.cu boundField, shared with kOmegaSST): "
                  "the average is ONE GATHER PER CELL in OpenFOAM's face order (fvcSurfaceIntegrate.C: "
                  "owner faces and losort-ordered neighbour faces merged by face index, then the boundary "
                  "faces), not the atomicAdd scatter it was until 2026-09-12 -- that scatter summed a "
                  "clamped cell's faces in hardware order, and was the D-1 run-to-run drift: squareBendLiq "
                  "and injectorPipe wrote different low bits between two runs of one binary (first "
                  "differing stage epsOut at iteration 4 with epsSolveOut identical). "
                  "tests/rho_run_to_run_identity.sh holds three tutorials byte-identical over two "
                  "20-iteration runs, with BRAE_BOUND_SCATTER=1 as the control that still drifts."),

        # ---- MRF / fvOptions -----------------------------------------------------------------
        dict(name="MRFZoneList", of_symbol="Foam::MRFZoneList",
             of_file="src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/cfdTools/MRF/device_mrf.cu",
             schema_for="MRFZone",
             brae_target="src/finiteVolume/cfdTools/general/MRF/",
             brae_reference="src/finiteVolume/cfdTools/general/MRF/MRF_cpp.cu",
             validation="tests/mrf_cpp_vs_openfoam.sh, END TO END on validation/mixerVessel2D (cellZone "
                        "`rotor`, omega 104.72 about z) against real OpenFOAM's converged 500: U 2.0e-03, "
                        "p 8.1e-04, k 1.5e-02, epsilon 1.6e-02 L2-relative, and at OpenFOAM's OWN "
                        "converged state an initial residual of 1.53e-05 against its 1.54e-05 (0.7%). CUDA: "
                        "tests/mrf_cuda_vs_openfoam.sh runs the SAME case through the V2 driver -- "
                        "U 2.5e-03, p 9.2e-04, k 1.5e-02, epsilon 1.6e-02 -- with a control that REMOVES "
                        "constant/MRFProperties, exercising the real dictionary path rather than a debug "
                        "switch. rotatingCylinders (laminar, Omega 100 over 6400 cells) runs on the same "
                        "wiring. Ported one hook at a time: correctBoundaryVelocity alone left the "
                        "momentum residual at 2.53e-02, adding DDt took it to 1.5256e-05, and "
                        "makeRelative left THAT unchanged -- see the note. THE CONTROL IS THE POINT -- "
                        "with the zones dropped the case has no driving force at all, the flow is "
                        "quiescent and every field reads 1.000, so MRF is 100% of this case's physics "
                        "rather than a correction to it. Each hook was checked the same way: dropping "
                        "DDt alone costs 158x on the momentum residual and dropping makeRelative 30x on "
                        "pressure.",
             note="Reached four times in the closure: correctBoundaryVelocity, DDt, makeRelative, update. "
                  "Silently ignoring it produced a converged wrong answer on the compressible path. "
                  "DDt is EXPLICIT: MRFZoneList::DDt builds a volVectorField of Omega x U from the "
                  "current U, which UEqn.H adds to the LHS, so it lands as source -= V*(Omega x U) and "
                  "is lagged like any other deferred term. update() is moving-mesh only and inert on a "
                  "steady static mesh; constrainPressure (pEqn.H:21) does nothing without a "
                  "fixedFluxPressure patch, which is outside this path's envelope anyway. The first working version was 10x (U) and 52x (p) off "
                  "OpenFOAM's initial residual at its own converged state. Localizing it put 93.8% on "
                  "ONE patch -- the rotor wall -- and the cause was not in MRF at all: brae's noSlip "
                  "returned a hardcoded ZERO from valueBoundaryCoeffs and re-zeroed its value on every "
                  "evaluate(), baking in `this wall is stationary`. True of every case that does not "
                  "rotate, false of every case that does. OpenFOAM's noSlip is a fixedValue whose "
                  "coefficients come from the LIVE patch value, and MRFZone::correctBoundaryVelocity "
                  "writes onto it with operator== precisely because that assigns regardless of "
                  "assignable(). Fixing noSlip took U from 1.60e-04 to 1.53e-05 against OpenFOAM's "
                  "1.54e-05, and the rotor's share of the residual from 93.8% to 3.6%. "
                  "MAKERELATIVE LOOKS INERT IN ONE ITERATION AND IS NOT: Omega x r is a solid-body "
                  "rotation and therefore DIVERGENCE-FREE, so removing its flux leaves "
                  "div(phiHbyA) -- the pressure equation's entire source -- unchanged, and p moves "
                  "only in the 7th digit. What it changes is phi itself, the convecting flux for "
                  "the NEXT iteration; end to end it is the difference between U 2.5e-03 and "
                  "U 7.2e-01. A single-iteration gate would have called that hook dead code. (An "
                  "earlier bisect putting makeRelative at 30x on the pressure residual was measured "
                  "before the noSlip fix and does not hold on the fixed code.) NOT TESTED BY THIS "
                  "CASE: OpenFOAM's internalFaces are the faces with EITHER cell in the zone, not "
                  "both, and brae existing (device_mrf.cu) uses BOTH. mixerVessel2D DOES have an "
                  "internal interface -- 96 of its 3024 zone faces -- yet the two readings give "
                  "bit-identical answers, because those 96 faces carry a frame flux of 6.4e-13 "
                  "against 1.5e-04 on the interior ones: the interface is a circle of constant "
                  "radius about the rotation axis, so its normals are radial while Omega x r is "
                  "circumferential and the dot product vanishes by geometry. Gating OR-vs-AND needs "
                  "a zone whose boundary is NOT a surface of revolution about its own axis."),
        dict(name="fvOptions", of_symbol="Foam::fv::options",
             of_file="src/finiteVolume/cfdTools/general/fvOptions/fvOptions.C",
             classification="DISPATCH", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/cfdTools/fvOptions/device_fvoptions.cu",
             selection_base="option",
             brae_target="src/finiteVolume/cfdTools/general/fvOptions/",
             note="Reached three times in UEqn.H/pEqn.H: fvOptions(U), constrain(UEqn), correct(U)."),

        # ---- linear solvers ------------------------------------------------------------------
        dict(name="lduMatrix_solver", of_symbol="Foam::lduMatrix::solver::New",
             of_file="src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixSolver.C",
             classification="LINEAR_SOLVER", status="REUSE_EXISTING",
             selection_base="lduMatrix::solver",
             brae_existing="src/cuda/device_pcg.cu, src/cuda/device_amg_pcg.cu",
             brae_target="src/matrices/lduMatrix/solvers/",
             note="8 keys in v2412. brae implements PCG (+AMG preconditioning) and PBiCGStab. "
                  "PBiCGStab HAS TWO LOOPS AND THEY MUST AGREE: a device conditional-graph loop (4 host "
                  "reads per solve, mailbox) and the host loop it falls back to when checkEvery > 1, "
                  "BRAE_BICG_HOST_LOOP=1 or BRAE_NORMFACTOR_HOST=1 is set, or the graph declines. DEFECT "
                  "FOUND AND FIXED 2026-09-12: the fallback forwarded every argument EXCEPT polyDeg, so a "
                  "solve that took it ran the bare DIAGONAL where the caller asked for a degree-d "
                  "truncated Neumann series -- a silent preconditioner substitution, invisible in a "
                  "residual line because the solve still reaches the case's tolerance and only stops "
                  "somewhere else. It surfaced when the FP-2 policy put the ENERGY solve on the series: "
                  "normfactor_device_identity's rhoBox arm, which holds the two normFactor paths "
                  "identical, then diverged from iteration 2 (49 of 51 lines). "
                  "tests/bicg_polydeg_host_loop.sh holds it directly -- the same case through the graph "
                  "loop, BRAE_BICG_HOST_LOOP=1 and BRAE_NORMFACTOR_HOST=1, every residual line and every "
                  "written field identical, with the case's own DILU as the control that proves the "
                  "preconditioner can move the iterate there. Fail-proof RUN: dropping polyDeg again "
                  "turned both gates red."),
        dict(name="GAMGPreconditioner", of_symbol="Foam::GAMGPreconditioner",
             of_file="src/OpenFOAM/matrices/lduMatrix/preconditioners/GAMGPreconditioner/"
                     "GAMGPreconditioner.C",
             classification="LINEAR_SOLVER", status="REUSE_EXISTING",
             schema_for="GAMGPreconditioner",
             brae_existing="src/cuda/device_amg.cu",
             brae_target="src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/",
             note="brae's AMG is the analogue. Carries the whole-loop conditional-graph PCG."),
        dict(name="DILUPreconditioner", of_symbol="Foam::DILUPreconditioner",
             of_file="src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/"
                     "DILUPreconditioner.C",
             classification="LINEAR_SOLVER", status="REUSE_EXISTING",
             brae_existing="src/cuda/device_dilu.cu",
             brae_target="src/matrices/lduMatrix/preconditioners/DILUPreconditioner/",
             note="Level-scheduled, bit-identical to OpenFOAM. TWO WALKS OF THE SAME LEVELS, same bits "
                  "(tests/dilu_single_block_identity.sh): one kernel launch per level, or ONE thread block "
                  "walking every level with __syncthreads between them. Which one runs is a speed rule on "
                  "the mean level width. FP-2 (bench/rhoSimpleFoam/FASTPATH.md, 2026-09-12) re-measured it "
                  "end to end after the graphs and the mailbox landed: the single block wins at mean 132 "
                  "(aerofoilNACA0012, 121 levels: turbulence 5.8 -> 3.3 ms/it, energy 3.2 -> 2.1), 205 and "
                  "417 (squareBend at 38k and 112k with DILU on every field: 36 -> 22 and 57.5 -> 47.9 ms/it) "
                  "and loses at 817 and 1665 (307k: 121.5 -> 135.4; 896k: 257 -> 396). The rule moved from "
                  "128 to 512, and the widest-level guard went (the block strides a level). On the aerofoil "
                  "the DILU level kernels were 862 of the turbulence phase's 1081 launches per iteration "
                  "and 449 of the energy phase's 539 (nsys, --cuda-graph-trace=node). THE ENTRY POLICY, "
                  "decided 2026-09-12 (rhoSimpleFoamDriver.cu, the CUDA arm): a `preconditioner DILU` "
                  "entry on k, epsilon/omega or the energy field takes the truncated Neumann series "
                  "wherever fvMatrix::relax bounds it -- the derivation the pair already takes on a GAMG "
                  "entry, factored into neumannDegreeIfRelaxed (linear_solver_setup.cuh), degree "
                  "ceil(ln 0.1 / ln alpha) capped at 24 -- announced per field as [approximated]; "
                  "BRAE_DILU_KE=1 and BRAE_DILU_HE=1 keep DILU, and an unrelaxed entry keeps it "
                  "regardless. Measured: aerofoil 3x100 iterations, turbulence 3.2 -> 1.6-1.8 ms/it, "
                  "energy 1.9 -> 1.2-1.4, four phases 10.0-10.2 -> 7.7-8.4 (1.0x 20 cores per "
                  "iteration); sbMatched at ITS pinned 1e-12 relTol 0 (degrees 22 and 11), turbulence "
                  "79 -> 13.5 and energy 30 -> 5.1 ms/it with the iteration-20 residual line identical to "
                  "five digits. Gate tests/rho_dilu_entry_policy_vs_openfoam.sh against real OpenFOAM on "
                  "the aerofoil at its own relTol: the series' k/omega/e residuals within [0.9, 1.2] of "
                  "OpenFOAM's (measured 1.07, 1.04, 1.03) and DILU-kept within [0.9, 1.1] (1.00, 1.00, "
                  "1.01), no bounding on either arm, turbulence under 0.8x (0.59x). The two assembly "
                  "gates that hold sbMatched at 1e-10 with the solvers pinned (rho_sbmatched_transient, "
                  "rho_gradp_lsq_simplec) keep DILU through the hatches, since at 1e-12 the residual "
                  "leaves the iterate free at about 1e-9 and only OpenFOAM's own algorithm lands on "
                  "OpenFOAM's iterate; their bounds are unchanged."),
        dict(name="GAMGSolver", of_symbol="Foam::GAMGSolver",
             of_file="src/OpenFOAM/matrices/lduMatrix/solvers/GAMG/GAMGSolver.C",
             classification="LINEAR_SOLVER", status="UNSUPPORTED",
             schema_for="GAMGSolver",
             brae_target="src/matrices/lduMatrix/solvers/GAMG/",
             note="pitzDaily and motorBike BOTH select `GAMG` for p. brae substitutes AMG-preconditioned "
                  "PCG. That is a different algorithm with a different iteration count -- the solver must "
                  "say so, not silently substitute. SPEED, FP-12 (bench/rhoSimpleFoam/FASTPATH.md, "
                  "2026-09-13): the pressure equation is the largest phase of every rhoSimpleFoam "
                  "tutorial, 37-52% of the four, and the V-cycle is ~95% overhead rather than "
                  "arithmetic. On gasMixing/injectorPipe (74,650 cells, 11 levels, ~15 V-cycles per "
                  "iteration) the phase is 1,814 launches and 6.57 GPU ms -- 121 kernels per cycle, all "
                  "already inside a CUDA graph -- while three fine-level SpMVs move about 6 MB, some 12 "
                  "us of bandwidth against 290 us of measured cycle. The FP32 SpMV costs 4.2 us on "
                  "levels under 1,000 cells where an elementwise kernel on the same grid costs 0.76. "
                  "MEASURED AND REJECTED: a bigger coarsest level (BRAE_AMG_TARGET 32/100/200/500/1k/2k/"
                  "5k gives p solve 4.5/4.2/5.8/8.4/12.7/23.9/34.7 ms/it on injectorPipe and "
                  "6.2/5.6/11.4/32.6/74.8/77.6/56.3 on squareBend -- the dense coarse LU punishes it); a "
                  "block sized to the level (bit-identical, 2.602 -> 2.533 ms SpMV on one case and "
                  "nothing on the other, reverted); the Gauss-Seidel smoother (BRAE_AMG_GS: p solve 4.4 "
                  "-> 13.0); and smoothed aggregation (BRAE_AMG_SA: the solve 4.4 -> 3.3 but the phase "
                  "7.1 -> 12.8, since its RAP setup runs every SIMPLE iteration -- and it scatters with "
                  "atomics, so it is not run-to-run deterministic, and squareBend diverges under it). "
                  "The FP32 V-cycle that is already the default is worth 0.8 ms/it. What the row points "
                  "at next, with the prize measured: fuse the coarse hierarchy into ONE kernel the way "
                  "device_dilu.cu walks its levels in a single block. THAT WAS WRITTEN AND IT LOSES "
                  "(2026-09-13): one block walking every level below a threshold -- down, the coarsest "
                  "LU solve, and back up -- is exactly bit-identical (100 iterations of injectorPipe, "
                  "every residual line) and slower at every threshold, 6.7/6.7/6.9/6.9/7.1/7.0 ms per "
                  "iteration for 64/128/256/512/1024/2048 cells against 6.5 unfused. The reason "
                  "corrects the earlier reading: these kernels are ALREADY graph nodes, so node "
                  "boundaries are cheap (fusing below 64 cells removes a dozen of them on levels where "
                  "one block is ample parallelism and gains nothing). What the coarse levels pay is the "
                  "device-side duration of a scattered indirect gather -- 4.2 us for the SpMV under "
                  "1,000 cells where zeroT on the same grid is 0.76 -- and one multiprocessor makes "
                  "that worse. Reverted; the finding is in device_amg_vcycle.cu. The candidate the "
                  "measurements still support is the coarse operator's LAYOUT, AND THAT ONE WORKS "
                  "(2026-09-13). Every grid now carries its rows contiguously -- entry i of row c is a "
                  "(value, column) pair, the cell's owner faces in ownerStart order then its neighbour "
                  "faces in losort order, which is the exact sequence the face-form kernel sums in, so "
                  "the terms, their order and the bits are unchanged (amulCsrFK against amulFK). The "
                  "values are refilled once per solve from the FP64 face arrays, replacing the two "
                  "casts that grid's upper and lower needed, so the layout costs no launch; the "
                  "structure is built once per hierarchy because the agglomeration is static for the "
                  "life of the mesh. MEASURED on gasMixing/injectorPipe: the SpMV goes 2.602 -> 1.634 "
                  "ms per iteration for the same 451 launches (-37%) and the whole iteration 13.79 -> "
                  "12.77 GPU ms. Pressure solve over three runs of 100 iterations: injectorPipe 4.3 -> "
                  "3.9, aerofoilNACA0012 2.9 -> 2.5, squareBendLiq 5.2 -> 4.8 ms/it; squareBend is flat "
                  "because it is transonic, where the asymmetric matrix puts the V-cycle inside "
                  "BiCGStab and the cycle is a smaller share of the solve. The threshold was swept and "
                  "EVERY grid wins, the finest included (4.3 / 4.3 / 4.2 / 4.0 / 3.9 / 3.8 ms for the "
                  "face form and thresholds 512 / 2,048 / 8,192 / 16,384 / all): the fine grid's owner "
                  "half is already sequential in the face arrays but its neighbour half is read through "
                  "losort. Bit-identical over 100 iterations on injectorPipe, squareBend, "
                  "aerofoilNACA0012 and squareBendLiq; BRAE_AMG_CSR=0 restores the face form and "
                  "BRAE_AMG_CSR_BELOW restricts it by grid size. THE TRANSONIC PATH IS A DIFFERENT "
                  "QUESTION (measured 2026-09-13): an asymmetric pressure matrix runs the FP64 cycle "
                  "with the two-stage Gauss-Seidel smoother, so the row layout -- built for the FP32 "
                  "mirrors -- never applies there; its FP64 SpMV is 1.41 of squareBend's pressure phase "
                  "(4.58 GPU ms per iteration) and would take the same 37%. But that phase is only 38% "
                  "GPU-BUSY under nsys, and so is injectorPipe's: the four phases are 10.92 busy of "
                  "23.41 wall on squareBend and 12.64 of 25.64 on injectorPipe (about 60% and 75% "
                  "unprofiled). The idle time is host launch throughput -- 656 cudaLaunchKernel calls "
                  "across 17 gaps, 38 per gap, because only the SOLVERS are captured into graphs while "
                  "the assemblies, boundary updates and reporting are not -- plus one blocking "
                  "reduction per iteration feeding the flow-rate inlet (463 us). So the transonic tail "
                  "belongs to the launch-floor row, not to the V-cycle. WHAT THE HOST IS DOING, counted on "
                  "squareBend (2026-09-13): 49.6 blocking cudaMemcpy per iteration at 7.04 ms of API time, 5 "
                  "cudaDeviceSynchronize at 1.39, and 1,004 cudaLaunchKernel at 4.45. Two wasteful items "
                  "were removed -- the SIMPLEC row sum built and uploaded a host vector of nCells ones on "
                  "EVERY iteration (a blocking 896 KB copy, 0.53 ms/it, where deviceOnes already existed) "
                  "and the continuity report recomputed sum(V), a mesh property, and took three blocking "
                  "reductions where one cached value and one mailbox read do. Blocking operations fell 54.6 "
                  "-> 50.6 per iteration and their API time 8.43 -> 6.80 ms. THE WALL DID NOT MOVE, and that "
                  "is the lesson: a blocking copy's API duration is mostly the host waiting for work the GPU "
                  "would do anyway, not idle GPU. The idle GPU is the 2.4 ms/it of gaps where the host issues "
                  "~38 kernels apiece, and closing those needs whole-phase graph capture, whose prerequisite "
                  "is hoisting the assemblies' pool-allocated temporaries into a stable workspace."),

        dict(name="dispatch", of_symbol="controlDict application",
             of_file="applications/solvers/incompressible/simpleFoam/simpleFoam.C",
             classification="DISPATCH", status="REIMPLEMENT",
             brae_cuda="src/applications/solvers/simpleFoam/simpleFoamV2.cu",
             brae_target="src/applications/solvers/simpleFoam/simpleFoamV2.cu",
             validation="CONVERGENCE GATE tests/simplefoam_v2_convergence.sh (ctest: "
                        "simplefoam_v2_convergence, ~90 s) -- the rebuilt path runs pitzDailyTurb from 0/ "
                        "to convergence (1713 iterations; OpenFOAM took 1576) and its converged fields are "
                        "compared with OpenFOAM's own 1576/: U 1.257e-01, p 1.922e-01, k 1.187e-02, "
                        "epsilon 2.512e-02, nut 6.686e-03. The bounds are what brae's ESTABLISHED solver "
                        "reaches against the same reference in the passing simple_turbulent_full gate "
                        "(U 1.311e-01, p 1.989e-01, ...) -- the rebuilt path is marginally better on every "
                        "field. The disagreement is localised (6 of 12225 cells above 0.5 m/s, worst at the "
                        "step corner) and is a property of the comparison, not of the rebuild. Carries a "
                        "control requiring the INITIAL field to exceed the bound. "
                        "ALSO tests/simplefoam_v2_dispatch.sh (ctest: simplefoam_v2_dispatch) -- drives the real "
                        "binary on real cases: OFF changes nothing and stays silent; ON+supported runs to "
                        "endTime and writes; ON+unsupported REFUSES with the reason and exits 1 for each of "
                        "MRF, fvOptions, SIMPLEC, a non-upwind div(phi,U), a transient ddtScheme and RAS; "
                        "the GAMG->AMG-PCG substitution is announced; RAS/kEpsilon RUNS and writes k/epsilon/nut "
                        "with a control that nut changed. Negative control: the guard admits the supported "
                        "case, so it is not refusing unconditionally.",
             note="OPT-IN via BRAE_SIMPLEFOAM_V2=1 while the envelope is small, and there is deliberately "
                  "NO try/catch around it: selected-but-unsupported must stop, never fall through to the "
                  "old solver. A user who asked for the new path and silently got the old one cannot tell "
                  "from the output which algorithm produced the answer. ENVELOPE TODAY: steady, laminar, "
                  "upwind or linearUpwind div(phi,U), orthogonal or `corrected` laplacian, SIMPLE or SIMPLEC, "
                  "no MRF/fvOptions, no fixedFluxPressure, no coupled patches -- stock pitzDaily and "
                  "pitzDailyTurb (RAS/kEpsilon) is inside it. The non-orthogonal check reads "
                  "laplacianSchemes ONLY: `laplacianSchemes Gauss linear orthogonal` builds an orthogonal "
                  "laplacian whatever snGradSchemes says, since snGradSchemes governs explicit fvc::snGrad "
                  "which in simpleFoam appears only in the refused SIMPLEC branch. "
                  "NON-ORTHOGONAL CORRECTION: implemented on the _cpp reference exactly as OpenFOAM defines "
                  "it -- BOTH halves, which is the part that is easy to miss: the implicit face "
                  "coefficient switches to nonOrthDeltaCoeffs = 1/max(n.delta, 0.05|delta|) "
                  "(correctedSnGrad.H:108-119, basicFvGeometryScheme.C:266) AND an explicit deferred "
                  "source -V*div(gamma*magSf*(corrVecs & interpolate(grad(vf)))) is added "
                  "(gaussLaplacianScheme.C), with corrVecs zero on boundary faces. VALIDATED against REAL "
                  "OpenFOAM (ctest: nonorth_vs_openfoam, which generates the reference by running "
                  "simpleFoam itself): on validation/shearedChannel -- genuinely non-orthogonal and using "
                  "only implemented schemes -- U 6.92e-04 and p 3.75e-04 vs OpenFOAM WITH the correction, "
                  "against U 8.47e-02 and p 1.37e-01 without it, i.e. 122x and 365x. The control requiring "
                  "the uncorrected path to be >=20x worse is the substance of that gate: on a "
                  "near-orthogonal mesh like pitzDaily every brae path agrees to 4 digits whether or not "
                  "the term is applied, so a gate there would pass with the code deleted. A SIGN ERROR was "
                  "found by this measurement -- divDevReff returns MINUS the laplacian, so its source "
                  "contribution is +V*div(...) while the pressure laplacian keeps OpenFOAM's own sign; "
                  "writing the laplacian sign in both places made U worse with the correction on (1.69e-01) "
                  "while p improved, and that asymmetry is what localised it. "
                  "THE CUDA SIDE IS NOW DONE TOO and carries both halves: deviceLaplacianCoeffs(..., "
                  "nonOrth=true) for the implicit coefficient and deviceLaplacianCorr for the deferred "
                  "source, in UEqn.cu (momentum, sign -1 because divDevReff carries minus the laplacian) "
                  "and pEqn.cu (pressure, sign +1). Matched to the reference at 2.9e-16 on the relaxed "
                  "diag and <=9.2e-16 on every source component, with controls asserting BOTH halves move "
                  "something (implicit 2.0e-03, explicit 1.7e-03 on U; 3.5e-03 and 7.0e-02 on p) so the "
                  "machine-precision agreement is not vacuous. The nonorth_vs_openfoam gate gained a CUDA "
                  "column that reproduces the reference against real OpenFOAM digit for digit: U 6.9193e-04 "
                  "and p 3.7529e-04, versus 8.4713e-02 / 1.3702e-01 uncorrected. TWO ORDERING TRAPS, both "
                  "of which built and ran clean while being wrong: (1) deviceDivDevReff ASSIGNS the "
                  "momentum source (device_divdevreff.cu, `dX[c] = d[0]`) rather than accumulating, so the "
                  "correction must be added AFTER it -- added before, it is silently discarded; (2) in the "
                  "diagnostic the GPU flags were copied from the host struct at the point `gin` is filled "
                  "in, which runs BEFORE those host flags are assigned, so the device got the struct "
                  "defaults and the CUDA column reproduced the uncorrected answer to five digits. Both "
                  "were caught by the gate's control, not by the build. `limited <coeff>` is REFUSED "
                  "separately: limitedSnGrad caps the correction and only `limited 1` equals `corrected`, "
                  "so accepting the whole family would over-correct. STILL OPEN for stock pitzDaily: "
                  "linearUpwind and SIMPLEC (`consistent yes`), which is what that case ships -- both "
                  "remain blockers. KNOWN NARROW GAP: on COUPLED patches OpenFOAM passes the corrected "
                  "deltaCoeffs to gradientInternalCoeffs (gaussLaplacianScheme.C, the pvf.coupled() "
                  "branch) while uncoupled patches use the plain ones; neither brae path does the coupled "
                  "variant, which is unreachable today because coupled patches are refused outright, and "
                  "must be revisited when they are added. "
                  "ENVELOPE WIDENING: `bounded` (-fvm::Sp(fvc::div(phi),U)) is DONE on both paths and "
                  "matched to 2.9e-16 (tests/test_ueqn_cuda.cu), with a control asserting the term "
                  "actually contributes -- it is ~2e-08 of the diagonal on a converged case because it "
                  "vanishes exactly at convergence, which is why agreement alone cannot prove it was "
                  "applied. The kernel sequence is the existing GPU driver's (device_simple_foam.cu:1150) "
                  "minus the cyclic/AMI additions. An earlier report that the CUDA term was a NO-OP was "
                  "wrong: the test's struct is `gi` and the patch set `gmi`, so the flag was never "
                  "enabled. "
                  "linearUpwind IS DONE on both paths. linearUpwind derives from `upwind`, so the MATRIX "
                  "is unchanged and the entire scheme is a deferred source correction "
                  "corr_f = (Cf - C_up) & grad(vf)_up, entering as `fvm += fvc::surfaceIntegrate(phi*corr)` "
                  "-- which is `source -= faceSum(...)`, since fvMatrix::operator+=(DimensionedField) is "
                  "`source() -= V*su` (fvMatrix.C:1855). Boundary faces contribute NOTHING on uncoupled "
                  "patches: linearUpwind::correction fills only the pvf.coupled() branch, so that is the "
                  "scheme, not an omission. OpenFOAM's `vector` SPECIALISATION uses one tensor grad(U) "
                  "rather than three scalar grads; the device does it per component, which is the same "
                  "field under grad(U)_ij = d(U_j)/dx_i, not an approximation. CUDA matches the reference "
                  "at 6.4e-16/8.8e-16 on the source with a control asserting the term moves the source "
                  "(3.5e-03) AND leaves the matrix EXACTLY alone -- a diagonal that moved would mean the "
                  "implicit weights had been changed, which is not what OpenFOAM does. The named gradient "
                  "is GUARDED: `linearUpwind grad(U)` resolves grad(U) through gradSchemes "
                  "(linearUpwind.H gradSchemeName_ -> mesh.gradScheme(name)), and anything but Gauss "
                  "linear -- cellLimited, leastSquares -- is REFUSED, because this correction does not "
                  "vanish at convergence and a different gradient is a different answer. GATE: "
                  "ctest linearupwind_vs_openfoam generates the oracle by running simpleFoam on "
                  "pitzDaily's own discretisation and CONTROLS with a plain-upwind brae run: "
                  "linearUpwind lands 1.9x (U), 3.2x (k), 2.4x (epsilon), 2.0x (nut) closer to OpenFOAM "
                  "than upwind does, measured in L2 -- pitzDaily's MAX-norm error sits in a few cells at "
                  "the step corner and is ~9.3e-02 whatever the scheme is, so the max norm cannot see "
                  "this term at all. Proven to FAIL with the correction disabled (every ratio collapses "
                  "to exactly 1.00x). "

                  "SIMPLEC (`consistent`) IS DONE on both paths: rAtU = 1/(1/rAU - UEqn.H1()), with "
                  "phiHbyA += interpolate(rAtU-rAU)*snGrad(p)*magSf and HbyA -= (rAU-rAtU)*grad(p), placed "
                  "AFTER adjustPhi as pEqn.H places them. matrixH1 (lduMatrixATmul.C) and fvc::snGrad "
                  "(snGradScheme.C + correctedSnGrad.C) were added for it. On the device the flux term is "
                  "written as a LAPLACIAN FLUX plus its non-orth part rather than as an explicit snGrad, "
                  "because fvm::laplacian(gamma,p).flux() IS gamma_f*magSf_f*snGrad(p)_f -- that keeps the "
                  "correction consistent with the equation it corrects by construction. What is still "
                  "refused is constrainPressure, i.e. a fixedFluxPressure patch: brae maps that type to "
                  "zeroGradient, which is the same BC only when the imposed flux is zero. "

                  "A REAL DEFECT WAS FOUND BY THIS WORK, in the non-orthogonal correction rather than in "
                  "SIMPLEC. fvMatrix::flux() adds faceFluxCorrectionPtr_ (fvMatrix.C:1688), which "
                  "gaussLaplacianScheme stores alongside the source correction (gaussLaplacianScheme.C:"
                  "186-199). brae had the correction in the pressure equation's SOURCE and NOT in "
                  "pEqn.flux(), so `phi = phiHbyA - pEqn.flux()` dropped it and phi was not conservative "
                  "on a non-orthogonal mesh -- silently, because the pressure equation still solved and "
                  "every per-stage test still passed at 1e-16. It was exposed by a control that has "
                  "nothing to do with fluxes: SIMPLEC and plain SIMPLE must converge to the SAME fixed "
                  "point, since SIMPLEC changes the iteration and not the discrete system, and brae's "
                  "differed by 5.3e-03 (U, L2). FIXED by porting faceFluxCorrection onto FvMatrix and "
                  "PressureMatrix, with the per-cell source now computed FROM the face flux so the two "
                  "cannot drift apart. Effect: SIMPLEC-vs-SIMPLE 5.3e-03 -> 2.8e-04; the nonorth gate on "
                  "shearedChannel U 6.92e-04 -> 3.12e-05 and p 3.75e-04 -> 2.69e-06 against real "
                  "OpenFOAM, its discriminating ratio 122x -> 2716x, and its bound tightened 5e-03 -> "
                  "1e-04 accordingly. "

                  "STOCK pitzDaily NOW RUNS AND IS GATED (ctest stock_pitzdaily_vs_openfoam): SIMPLEC + "
                  "bounded Gauss linearUpwind grad(U) + Gauss linear corrected + RAS/kEpsilon, "
                  "unmodified apart from tightening residualControl from the stock `p 1e-2` (at which "
                  "both solvers stop early at different states, so the comparison would measure where "
                  "each stopped). vs real OpenFOAM, L2: U 5.18e-03, p 2.25e-02, k 1.85e-02, "
                  "epsilon 3.82e-02, nut 2.75e-02, and brae is settled -- iteration 749 vs 2000 differs "
                  "by 1.8e-08. The gate also asserts every feature is REPORTED as applied, so a silent "
                  "skip fails it. "
                  "PERFORMANCE, measured on GB10 vs OpenFOAM v2412 on 20 cores, 100 SIMPLE iterations on "
                  "scaled pitzDaily (12k/110k/440k/1.22M): the rebuilt path was NOT reading the case's "
                  "fvSolution solver settings at all -- StepInput hardcoded tolerance 1e-10 / relTol 0 "
                  "while pitzDaily asks for relTol 0.1 -- so every inner solve ran to near machine "
                  "precision every outer iteration. nsys: 614 fine-grid SpMV per SIMPLE iteration against "
                  "the existing GPU driver's 62. FIXED by reading solvers/p and solvers/U (subDict "
                  "resolves OpenFOAM regex keys, so `\"(U|k|epsilon|omega|f|v2)\"` is found by \"U\"); "
                  "maxIter is read too, since an `maxIter 10` caps the answer. Effect: 2.3-2.6x faster at "
                  "every mesh size (1.22M: 166s -> 67s) with the converged answer UNCHANGED -- U L2 vs "
                  "OpenFOAM is 5.1790e-03 before and after, which is the point: the inner tolerance "
                  "changes the path, not the fixed point. Side effect on the gate: brae now plateaus like "
                  "OpenFOAM (U ~9e-06, p ~2e-04, flat from iteration ~700) instead of reaching a tight "
                  "residualControl, so stock_pitzdaily_vs_openfoam now ASSERTS settledness by re-running "
                  "to half the endTime rather than trusting a residual. REMAINING GAP to the existing GPU "
                  "driver is ~2.3x and is now dominated by the MOMENTUM solve: the rebuilt path uses "
                  "Jacobi-BiCGStab (~42 iterations per component per outer step, ~90% of all "
                  "linear-algebra time) where the case asks for smoothSolver/symGaussSeidel, which brae "
                  "already implements and the old driver uses (gsColorT). Also unwired on this path: the "
                  "AMG V-cycle CUDA-graph capture (deviceAMGPCG's captureVcycle defaults false; the old "
                  "driver passes useGraph=true), the PCG residual-read cadence (checkEvery), and the "
                  "turbulence hook's nuEff refresh, which round-trips nut to the HOST and does the face "
                  "interpolation there every iteration when deviceInterpolate would keep it resident. "
                  "The rebuilt driver also writes only its final time -- writeInterval is not honoured. "
                  "FOLLOW-UP, all measured on the same sweep (100 iterations, seconds, 12k/110k/440k/1.22M). "
                  "(a) The MOMENTUM solver was routed to smoothSolver/symGaussSeidel, which is what the case "
                  "asks for and what the existing driver uses -- a real fidelity fix (it was a silent solver "
                  "substitution) but PERFORMANCE-NEUTRAL: 6.99->6.49 / 11.95->11.05 / 28.52->30.11 / "
                  "67.05->69.44. An earlier reading of the profile had attributed ~90% of linear-algebra "
                  "time to it; removing BiCGStab from U cut only ~7% of the SpMV, so that attribution was "
                  "wrong and is retracted. (b) The ACTUAL cost was the same defect one level down: the "
                  "turbulence hook passed a hardcoded tol=1e-10 with relTol 0 to deviceKEpsilonCorrect, so "
                  "k and epsilon were solved to near machine precision every outer iteration. Phase timing "
                  "found it -- the hook was 174.5 ms of a ~300 ms outer iteration at 440k, and only 6.1 ms "
                  "of that was the nuEff host round-trip everyone would suspect. Reading solvers/k "
                  "(tolerance, relTol, smoother) cut the hook to 48.7 ms and the run to "
                  "1.87/4.30/17.35/47.94. (c) The nuEff host round-trip is therefore NOT worth fixing on "
                  "cost grounds (6 ms); it stays on the list only as an unnecessary host dependency. "
                  "(d) CUDA GRAPH: deviceAMGPCG on CUDA>=13 already dispatches to a DEVICE-RESIDENT "
                  "conditional-graph PCG (BRAE_PCG_DEVICE, default on) that captures the whole Krylov loop, "
                  "measured A/B 17.9 s on vs 18.7 s off at 440k. The captureVcycle/checkEvery arguments "
                  "this path now passes explicitly are ignored by that dispatch and reach only the "
                  "host-driven fallback -- worth knowing before anyone else measures them and concludes "
                  "graphs do nothing. (e) AMG SMOOTHED AGGREGATION cuts the pressure iteration count "
                  "roughly in half (BRAE_AMG_SA=1 36->27, with BRAE_AMG_GS=1 36->19 AMG-PCG iterations for "
                  "a 10x residual drop at 440k) but does NOT reduce wall time -- each cycle costs more than "
                  "the halved count saves -- so neither is enabled by default. NET: 17.90->2.60 (12k), "
                  "28.05->4.83, 67.58->17.27, 165.84->47.19 (1.22M), i.e. 3.5-6.9x, with the answer "
                  "unchanged (U L2 vs OpenFOAM 5.1790e-03 -> 5.1805e-03). Reference points on the same box: "
                  "OpenFOAM v2412 on 20 cores 1.04/2.52/8.85/26.56 and brae's existing GPU driver "
                  "2.04/4.43/14.79/39.42. STILL OPEN: the GPU is busy 74 ms of a 173 ms outer iteration at "
                  "440k, the rest being blocking scalar convergence reads (188 per outer iteration); "
                  "closing that needs the outer loop itself captured, not the inner solves. Also open as "
                  "FIDELITY, not speed: div(phi,k)/div(phi,epsilon) are run as upwind while pitzDaily asks "
                  "for `bounded Gauss limitedLinear 1`, and the envelope guard does not check them. "
                  "AMG REUSE: the hierarchy is built ONCE per run and reused across every SIMPLE iteration "
                  "(SolverWorkspace::amgBuilt); only the coarse-operator VALUES are re-Galerkined each "
                  "iteration, which is required because the fine coefficients change. Across RUNS it was "
                  "cold every time -- the rebuilt driver called buildAMG, not buildOrLoadAMG -- so the "
                  "existing .brae_amgcache serialization of the agglomeration STRUCTURE was unreachable "
                  "from this path. Now opt-in via BRAE_AMG_CACHE=1, which writes "
                  "constant/polyMesh/.brae_amgcache and reloads it when it is newer than `owner`: startup "
                  "at 1.22M cells 7.27 s cold -> 4.92 s warm (the whole run is ~47 s, so ~5%). "
                  "SCALING to 4.89M cells, 100 iterations: OpenFOAM v2412 on 20 cores 125.4 s, brae's "
                  "existing GPU driver 155.4 s, the rebuilt path 211.2 s. The OF-to-brae ratio narrows "
                  "with mesh size but slowly -- 2.5x at 12k, 2.0x at 440k, 1.8x at 1.22M, 1.7x at 4.89M -- "
                  "so on THIS hardware (GB10, a workstation part with modest FP64) brae does not overtake "
                  "a 20-core CPU by extrapolation. AMG mode at 4.89M: SA+GS 202.8 s and SA 207.4 s beat "
                  "the default 211.2 s by only 2-4%, while GS ALONE is clearly worse at 247.9 s (+17%). "
                  "SA halves the pressure ITERATION count (36->19 at 440k) at every size but only starts "
                  "paying for itself in wall time around 5M, so it stays off by default and GS-without-SA "
                  "should be avoided. "
                  "TUTORIAL SWEEP of all 17 incompressible/simpleFoam tutorials: 4 skipped (no mesh "
                  "without snappyHexMesh/m4/extrude), 13 meshable, of which 2 run -- pitzDaily and "
                  "pitzDailyExptInlet -- and 11 are refused with named reasons. Tally of what blocks them: "
                  "turbulence model 5 (kOmegaSST x3, realizableKE, kOmegaSSTLM), div(phi,U) scheme 5 "
                  "(LUST, linearUpwindV, limitedLinearV x2, fusedGauss), fvOptions 3, MRF 1, cyclicAMI 1, "
                  "transient ddt 1, laplacian `limited` 1, and linearUpwind naming a cellLimited gradient "
                  "1. Nothing ran and gave a wrong answer; every refusal named its component. "
                  "THAT SWEEP MEASURES THE REBUILD, NOT brae. Re-run offering each meshed tutorial to BOTH "
                  "paths: the SHIPPED driver runs 9 of 9 -- backwardFacingStep2D, pipeCyclic, pitzDaily, "
                  "pitzDailyExptInlet, rotatingCylinders, rotorDisk, simpleCar, T3A, and motorBike "
                  "(353,784 cells, snappyHexMesh, kOmegaSST + linearUpwindV, producing Cd/Cl/Cm through "
                  "its forceCoeffs functionObject) -- while the rebuilt path runs 2. The shipped path "
                  "wires 13 turbulence models (kEpsilon, kOmega, kOmegaSST + DES/IDDES/LM, realizableKE, "
                  "RNGkEpsilon, SpalartAllmaras + DDES/IDDES, Smagorinsky, WALE, laminar), MRF, fvOptions, "
                  "cyclic/cyclicAMI/cyclicACMI, porosity. The gap is the REBUILD's by construction: this "
                  "port admits a component only after it is validated against OpenFOAM, and refuses "
                  "otherwise. Closing it is wiring plus gating, not new CUDA -- the device kernels for "
                  "linearUpwindV, limitedLinear(+V), LUST, vanAlbada and the other turbulence models "
                  "already exist and the shipped path drives them. MESHING is not and never was brae's: "
                  "blockMesh/snappyHexMesh are OpenFOAM's, and the `no mesh` rows in the first sweep were "
                  "the sweep harness failing to drive each tutorial's own workflow (geometry copied from "
                  "tutorials/resources/geometry, m4, snappy), not a brae capability. motorBike is the "
                  "proof: meshed with OpenFOAM's own tools, brae reads it and solves it. "
                  "kOmegaSST IS NOW PORTED, and driven from ofscan rather than from tutorial refusals -- "
                  "which was the wrong source of truth and is corrected: `ofscan impls "
                  "incompressible::turbulenceModel` enumerates 26 models, and `ofscan schema "
                  "kOmegaSSTBase` gives 17 dictionary keys with defaults and file:line. That enumeration "
                  "immediately found a gap no tutorial exercises: brae reads 13 of the 17 and not "
                  "decayControl/kInf/omegaInf. THE _cpp REFERENCE CAME FIRST, per the port discipline: "
                  "kOmegaSST_cpp transcribes kOmegaSSTBase::correct() -- S2, GbyNu0, CDkOmega, F1, F2, "
                  "the GbyNu production limiter, both transport equations, the omega wall function "
                  "(blender BINOMIAL n=2 by default: omega = sqrt(omegaVis^2 + omegaLog^2)) and "
                  "correctNut -- and REFUSES F3, decayControl, MRF and fvOptions. Two transcription "
                  "subtleties it has to get right and which a fused port hides: the k equation takes G "
                  "from the RAW GbyNu0 while the omega equation takes the LIMITED one (kOmegaSSTBase.C "
                  "reassigns GbyNu0 only after G is formed), and correctNut(S2) at the end uses the OLD "
                  "S2 but F23() re-read from the NEW post-solve k/omega. "
                  "THE ORACLE IS THE EQUATION RESIDUAL, not a fixed point: at OpenFOAM's converged "
                  "pitzDaily-kOmegaSST fields our assembled equations give omega 3.99e-03 and k 1.53e-02 "
                  "in OpenFOAM's own normalisation, and a (k,omega)x1.5 perturbation raises them to "
                  "7.10e-02 / 7.70e-02 (the control). The tempting test -- run one correct() and check "
                  "nothing moves -- was tried and is INVALID here, because OpenFOAM stops on a residual "
                  "plateau rather than at an exact fixed point, so solving from its state to 1e-12 moves "
                  "omega by 1.5e-01 in max norm with nothing wrong; that measurement is what localised "
                  "the oracle, after first confirming the wall cells were fine (1.1e-03), the wall "
                  "distance right (max 0.0252 m against a 0.0254 m half-height) and the solves converged "
                  "(47 iterations to 3.6e-13). GbyNu0 and S2 agree to four digits on this field, which is "
                  "the expected identity for divergence-free flow and an independent cross-check of two "
                  "separate transcriptions. END TO END the CUDA path then runs the same case and agrees "
                  "with real OpenFOAM at U 9.25e-03, p 1.81e-02, k 2.42e-02, nut 3.50e-02 (L2), "
                  "comparable to the kEpsilon path's U 5.18e-03. STILL OPEN, and NOT started: "
                  "linearUpwindV, limitedLinearV, limitedLinear, LUST, fvOptions, MRF, cyclicAMI, "
                  "realizableKE and kOmegaSSTLM. kEpsilon also still has NO _cpp reference -- it was "
                  "wired straight to CUDA earlier in this port, which is the gap this component avoided. "
                  "LIMITED/BLENDED CONVECTION SCHEMES, _cpp reference: ofscan enumerates 78 "
                  "surfaceInterpolationScheme implementations, of which this step ports the weight side of "
                  "three -- LUST, limitedLinear and limitedLinearV. The classification is the part that "
                  "matters and is easy to get silently wrong: `upwind` is weights only; `linearUpwind` "
                  "derives from upwind so its weights are UNCHANGED and it is a deferred correction only; "
                  "`limitedLinear`(+V) is weights only; `LUST` is BOTH (LUST.H overrides weights() as "
                  "0.75*linear + 0.25*upwind AND correction() as 0.25*linearUpwind's). A scheme ported as "
                  "the wrong one of those three still converges to a plausible answer. Transcribed: "
                  "weights = limiter*CDweights + (1-limiter)*pos0(phi) "
                  "(limitedSurfaceInterpolationScheme.C), limiter = clamp(twoByk*r, 0, 1) with "
                  "twoByk = 2/max(k,SMALL) (limitedLinear.H:82), r = 2*(gradcf/gradf) - 1 with OpenFOAM's "
                  "own 1000x guard (NVDTVD.H), and for the V forms gradf = gradfV & gradfV with "
                  "gradcf = gradfV & (d & gradc) (NVDVTVDV.H) -- ONE limiter shared across the three "
                  "components. fvm::div gained a weights overload so `upwind` is just the pos0 special "
                  "case and every existing call site is untouched. VALIDATED against exact properties of "
                  "OpenFOAM's source rather than a stored field: LUST reproduces its blend identity to "
                  "0.0e+00, limitedLinear collapses to upwind as k->inf (4.0e-10, asymptotic not exact -- "
                  "twoByk*r is ~1e-9 there, and an exact-equality assertion was wrong), reaches the linear "
                  "limit on 22939 of 24170 faces at k->0, and all three stay inside the TVD bound [0,1]. "
                  "THE CONTROL that earns the V variant: limitedLinearV differs from per-component "
                  "limitedLinear by 5.49e-01, so limiting the components independently would have been a "
                  "different scheme, not a rounding difference. NOT YET DONE for these schemes: the "
                  "wiring into UEqn (scheme selection), the envelope acceptance, LUST's 0.25x "
                  "linearUpwind correction at the call site, and the CUDA comparison -- so the envelope "
                  "still REFUSES all of them, which is the correct state until they are gated. "
                  "linearUpwindV is NOT started (deviceLinearUpwindVCorr exists on the device side). "
                  "THE THREE SCHEMES ARE NOW WIRED AND GATED on both paths. fvm::div gained a weights "
                  "overload; UEqn_cpp gained divWithScheme() branching on DivScheme, and a "
                  "correctionFactor() that is a FACTOR rather than a flag because LUST carries 0.25 of "
                  "linearUpwind's deferred correction as well as its own weights. The CUDA side branches "
                  "the same way: deviceDivLimitedCoeffs (on magSqr(U)), deviceDivLimitedVCoeffs, and for "
                  "LUST a 0.75/0.25 blend of the CENTRAL and UPWIND coefficient arrays -- exact, not an "
                  "approximation, because the coefficients are linear in the weights. fvc::gaussGrad "
                  "gained an array form so magSqr(U) does not need a synthetic GeometricField with "
                  "invented boundary types. "
                  "GATE (ctest divschemes_vs_openfoam): each scheme run in OpenFOAM AND in brae on "
                  "pitzDaily, with a plain-upwind brae run against the same oracle as the CONTROL. U "
                  "cannot discriminate -- every scheme including upwind lands within ~5.5e-03 of "
                  "OpenFOAM -- so the gate asserts on k, where the convection scheme actually bites: "
                  "limitedLinear 1.88e-02 vs upwind 5.39e-02 (2.87x), limitedLinearV 1.85e-02 vs 5.53e-02 "
                  "(2.99x), LUST 1.88e-02 vs 5.69e-02 (3.03x). "
                  "CUDA vs the reference with a limiter active is 5.5e-12 on the off-diagonals, NOT the "
                  "1e-16 the upwind path reaches, and that is the scheme's arithmetic rather than a "
                  "defect: r = 2*(gradcf/gradf) - 1 divides by a face difference that approaches zero in "
                  "smooth regions, so the ~1e-16 disagreement between the host and device Gauss gradients "
                  "(different face summation order) is amplified. test_ueqn_cuda therefore carries a "
                  "SCHEME-AWARE tolerance -- 1e-13 for upwind, 5e-11 with a limiter -- with a control "
                  "asserting the limiter moves the off-diagonals by 4.4e-01, so the looser bound is not "
                  "hiding an absent term. Envelope: five div(phi,U) schemes accepted by name, everything "
                  "else refused; the scheme COEFFICIENT is read too, since `limitedLinear 0.2` is a "
                  "materially different scheme from `limitedLinear 1`. "
                  "linearUpwindV IS NOW PORTED TOO, and it is the fourth KIND: weights UNCHANGED (it "
                  "derives from upwind like linearUpwind) but a DIFFERENT correction, not a scaled one, "
                  "so it could not be folded into the LUST correction factor. linearUpwindV.C limits the "
                  "correction against the owner-to-neighbour jump PROJECTED ON THE CORRECTION'S OWN "
                  "DIRECTION: s = magSqr(corr), mx = corr & maxCorr, and then corr is zeroed when mx < 0 "
                  "or scaled by mx/(s + VSMALL) when s > mx -- a test over the VECTOR, so it cannot be "
                  "applied per component, the same trap limitedLinear/limitedLinearV sets. The face "
                  "corrections are exposed (linearUpwindVFaceCorrection) and the per-cell sum computed "
                  "FROM them, because the limiter has an exact postcondition that is only checkable per "
                  "face: after limiting, either corr == 0 or 0 <= magSqr(corr) <= corr & maxCorr. On "
                  "pitzDailyTurb that holds on all 24170 faces with 0 violations, 125 zeroed and 12557 at "
                  "the limit, and the result differs from the UNLIMITED linearUpwind correction by "
                  "8.87e-01 -- the limiter is the scheme, not a detail. Gate: 2.90x closer to OpenFOAM "
                  "than upwind on k (1.87e-02 vs 5.42e-02). "
                  "CUDA-vs-REFERENCE AT MATRIX GRANULARITY for every ported scheme, one ctest each, and "
                  "the numbers correct the earlier blanket tolerance: linearUpwindV 2.9e-16 diag / 0.0 "
                  "upper and LUST 3.0e-16 / 1.1e-16 reach MACHINE PRECISION, while limitedLinearV is "
                  "1.4e-13 and limitedLinear 5.5e-12. Only the r-ratio limiters amplify, because only "
                  "they divide by a face difference that approaches zero; giving LUST and linearUpwindV "
                  "the loose 5e-11 bound would have hidden a real defect in them, so the tolerance is now "
                  "per-scheme. linearUpwindV additionally asserts its matrix is EXACTLY upwind's "
                  "(rel = 0.0) -- for a correction-only scheme, asserting the matrix MOVED would assert "
                  "the opposite of the port. "
                  "realizableKE IS NOW PORTED. ofscan's schema gives its four keys (A0 4.0, C2 1.9, "
                  "sigmak 1.0, sigmaEps 1.2). Three things make it realizableKE rather than k-epsilon "
                  "with different constants, and each is a place a port silently becomes the standard "
                  "model: (1) Cmu IS NOT CONSTANT -- rCmu = 1/(A0 + As*Us*k/eps) with As from the strain "
                  "invariants through W = 2*sqrt(2)*((S&S)&&S)/(magS*S2 + SMALL) and "
                  "phis = acos(clamp(sqrt(6)*W,-1,1))/3; (2) S2 = 2*magSqr(devSymm(gradU)) -- devSymm, "
                  "NOT the symm kOmegaSST uses a few directories away, and they differ by the trace term; "
                  "(3) the EPSILON EQUATION IS A DIFFERENT EQUATION -- C1*magS*eps production with "
                  "C1 = max(eta/(5+eta), 0.43) and C2*eps/(k + sqrt(nuLimited*eps)) destruction, with "
                  "neither kEpsilon's C1*Cmu*k*GbyNu production nor its divU SuSp term. The k equation "
                  "IS kEpsilon's shape. Wall functions keep Cmu = 0.09 regardless, because OpenFOAM's "
                  "nutk/epsilon wall functions read it from their own patch dictionaries. "
                  "_cpp reference validated on the same residual oracle as kOmegaSST: at OpenFOAM's "
                  "converged pitzDaily-realizableKE fields our equations give epsilon 4.82e-03 and "
                  "k 2.26e-05. The perturbation control had to be REDESIGNED: scaling k and epsilon "
                  "TOGETHER barely moves the epsilon residual (4.82e-03 -> 8.48e-03, 1.8x) because the "
                  "destruction term C2*eps/(k + sqrt(nu*eps)) is a ratio in which a common factor largely "
                  "cancels -- so each equation is now probed by perturbing ITS OWN field, giving 26x on "
                  "epsilon and 11x on k. Sharpening the probe was the right fix; loosening the threshold "
                  "would have hidden the cancellation instead of exposing it. Two controls say WHICH "
                  "model was ported: rCmu varies over [0.0239, 0.2500] (a constant 0.09 would be standard "
                  "k-epsilon, which would still work and still converge) and devSymm-S2 differs from "
                  "symm-S2 by 5.84e+05. CUDA needed no new kernel -- KEpsilonCoeffs::realizable already "
                  "selects the variable Cmu, the strain production and the modified destruction inside "
                  "the fused deviceKEpsilonCorrect -- so the wiring is a coefficient change, and the "
                  "envelope now accepts kEpsilon, realizableKE and kOmegaSST. End to end vs real "
                  "OpenFOAM: U 5.41e-03, p 2.80e-02, k 2.66e-02, epsilon 3.09e-02, nut 4.14e-02 (L2), "
                  "in line with kEpsilon's U 5.18e-03. "
                  "fvOptions is NOT started and is deliberately not half-started: ofscan counts 46 "
                  "implementations, and the port splits into a FRAMEWORK (reading system/ or "
                  "constant/fvOptions, the selectionMode cell selection, and the three hooks UEqn.H/pEqn.H "
                  "use -- fvOptions(U) as a source, fvOptions.constrain(UEqn), fvOptions.correct(U)) plus "
                  "individual sources. The three refused tutorials need three different ones: "
                  "explicitPorositySource/DarcyForchheimer (simpleCar), rotorDiskSource (rotorDisk) and "
                  "actuationDiskSource (turbineSiting), so the framework buys nothing on its own. "
                  "THE FRAMEWORK AND explicitPorositySource ARE NOW PORTED. The framework reads system/ or "
                  "constant/fvOptions, resolves the selection through the EXISTING cellSetOption port "
                  "(cell_selection.cuh, written for MRF and reused rather than rewritten), and applies "
                  "UEqn.H's `== fvOptions(U)`; fvOptions.constrain(UEqn) and fvOptions.correct(U) are NOT "
                  "implemented for any option. Any option whose TYPE is not ported is refused BY NAME -- "
                  "reading the framework as `fvOptions is supported` would be precisely the silent "
                  "substitution this guard exists for, and ofscan counts 46 fv::option implementations. "
                  "THE SIGN passes through two negations that cancel: explicitPorositySource does "
                  "`eqn -= porosityEqn` and simpleFoam writes `UEqn == fvOptions(U)`, so the net effect is "
                  "the porosity equation as written -- diag += V*tr(Cd), source -= V*((Cd - I*tr(Cd))&U) "
                  "with Cd = nu*D + |U|*F. Applying one negation and not the other gives a porosity that "
                  "ACCELERATES the flow, so the unit test asserts a positive diagonal explicitly. Other "
                  "transcription points: the 0.5 lives in F (calcTransformModelData's forchCoeff), NOT in "
                  "the resistance; and DarcyForchheimer looks up the field NAMED `nu`, the LAMINAR "
                  "viscosity, not nuEff -- using nuEff would make the Darcy resistance scale with the "
                  "turbulence model. On the device the coordinate system must be IDENTITY: the existing "
                  "deviceFvoPorosity kernels take diagonal d/f, so a rotated coordinateSystem is REFUSED "
                  "rather than having its off-diagonals dropped. "
                  "UNIT (ctest fvoptions_cpp, on simpleCar's mesh): the cellZone resolves to exactly the "
                  "10 cells OpenFOAM's own topoSet reports (12750-12759), D.xx = 5e7 with d.y = -1000 "
                  "keeping its sign, D diagonal under an identity system, and the resistance matches "
                  "DarcyForchheimerTemplates.C to 6.8e-18 on the diagonal and 3.3e-21 on the source, with "
                  "the 12930 cells outside the zone left untouched. "
                  "END-TO-END (ctest fvoptions_vs_openfoam) IS NOT ON simpleCar, and the reason is a "
                  "finding in itself: brae disagrees with OpenFOAM on simpleCar by U 5.05e-01 (L2) with "
                  "the fvOptions file REMOVED FROM BOTH, and the SHIPPED driver disagrees identically "
                  "(5.05e-01) -- so this is a pre-existing, brae-wide disagreement on that case, not "
                  "something this port introduced, and simpleCar can say nothing about the porosity until "
                  "it is understood. Its U boundary conditions (pressureInletOutletVelocity, "
                  "surfaceNormalFixedValue, uniformNormalFixedValue) are all implemented and brae throws "
                  "on genuinely unknown types, so silent BC substitution is ruled out; the cause is "
                  "LOCALISED, and the elimination is worth keeping because most of it was cheap. NOT the porosity (the "
                  "disagreement is identical with fvOptions removed from both). NOT the rebuild (the SHIPPED "
                  "driver disagrees identically, 5.05e-01). NOT the mesh harness (simpleCar's Allrun runs "
                  "createPatch -overwrite, which the first attempt skipped; meshing it properly gives the same "
                  "5.04e-01). NOT the `ramp` on surfaceNormalFixedValue, which brae deliberately ignores -- "
                  "OpenFOAM with and without the ramp agree to 2.2e-04. NOT surfaceNormalFixedValue and NOT "
                  "pressureInletOutletVelocity: replacing airIntake with a plain wall and the outlet with "
                  "zeroGradient IN BOTH leaves 5.05e-01. NOT OpenFOAM being unconverged -- although it was: at "
                  "t=2000 its Ux residual is 2.7e-05 and its own answer still moves 9.4e-02 by t=20000, where "
                  "the residual reaches 1.5e-08; brae is 5.15e-01 from THAT. "
                  "IT IS THE PRESSURE EQUATION. Restarting brae from OpenFOAM's t=20000 converged fields, the "
                  "first-iteration residuals are p 1.2e-01 against U 2.6e-03, and brae moves 29% away within 50 "
                  "iterations -- so OpenFOAM's solution is not a fixed point of brae's pressure equation on "
                  "this mesh. THE MESH is what distinguishes the case: simpleCar is max non-orthogonality "
                  "44.6 degrees (average 11.9) with max skewness 2.30, against pitzDaily's 5.95/1.63 and "
                  "skewness 0.26. The nonorth gate's shearedChannel is a UNIFORM 30.96 degrees with skewness "
                  "1.5 and brae matches OpenFOAM there to 3.1e-05, so plain non-orthogonality is not "
                  "sufficient to reproduce it -- the untested regime is the combination of a high-angle TAIL "
                  "(44.6 max over an 11.9 average) with skewness above 2, and `nNonOrthogonalCorrectors 0` "
                  "means the deferred correction gets a single pass. RULED OUT as the cause: the geometry "
                  "primitives, all four of which match OpenFOAM's formulas exactly -- weights "
                  "SfdNei/(SfdOwn+SfdNei) with the Sf-projected distances (surfaceInterpolation::makeWeights), "
                  "deltaCoeffs 1/|delta|, nonOrthDeltaCoeffs 1/max(n.delta, 0.05|delta|), and correction "
                  "vectors n - delta*nonOrthDeltaCoeffs. STILL OPEN, and it affects the SHIPPED solver, not "
                  "just this port. "
                  "THE _cpp REFERENCE DISAGREES TOO, which redirects the search and which the first pass "
                  "MISSED by chasing simpleCar entirely through the CUDA path: diag_simple_loop from "
                  "OpenFOAM's converged fields gives _cpp U 6.13e-01, old-host 5.97e-01 and CUDA 5.96e-01, "
                  "all three together. So the defect is in the TRANSCRIPTION of OpenFOAM into the _cpp "
                  "reference, not in the CUDA port of it -- which is the order this port is supposed to "
                  "establish first, since nothing downstream can be right if _cpp is not. "
                  "THE MESH-ANGLE HYPOTHESIS IS REFUTED. Sweeping shearedChannel's shear (its top-edge "
                  "offset over a 0.1 height, so the angle is one parameter) against a freshly-run OpenFOAM "
                  "at each level, U on the _cpp path: 30.96 deg/skew 1.5 -> 3.12e-05, 45/2.5 -> 4.85e-05, "
                  "56.3/3.75 -> 5.38e-05, 63.4/4.0 -> 1.75e-04, 68.2/3.45 -> 1.26e-03. The error does grow "
                  "with angle, but even at 68 degrees with skewness 3.4 it is 1.3e-03 -- about 400x "
                  "smaller than simpleCar's 5e-01 on a 44.6 deg / 2.30 mesh. Uniform shear does not "
                  "reproduce it, so `non-orthogonality plus skewness` is NOT the cause. The sweep also has "
                  "_cpp and CUDA tracking to four digits at EVERY level (3.1195e-05 vs 3.1180e-05, and so "
                  "on), independent confirmation that the CUDA port is faithful across the whole range. "
                  "A LAMINAR simpleCar cannot isolate the turbulence either: at nu 1e-5 and 10 m/s the "
                  "case is physically unsteady and OpenFOAM itself does not converge (p residual stuck at "
                  "0.24 after 3000 iterations); brae additionally goes NaN there, a robustness difference "
                  "on an ill-posed case rather than a usable control. Ruled out so far: the porosity, the "
                  "rebuild-vs-shipped split, the mesh harness, the ramp, surfaceNormalFixedValue, "
                  "pressureInletOutletVelocity, OpenFOAM non-convergence, mesh angle and skewness, and the "
                  "four geometry primitives. "
                  "RETRACTION: the claim that the _cpp reference disagrees on simpleCar is WRONG and is "
                  "withdrawn. diag_simple_loop runs LAMINAR by construction (dctl.turbulent = false, "
                  "nuEff = nu everywhere -- it was built for the laminar shearedChannel non-orth work), so "
                  "that measurement compared a laminar brae against a kEpsilon OpenFOAM reference. An "
                  "unequal comparison, the same class of error the diagnostic's own header warns about. "
                  "THE CAUSE IS THE TURBULENCE MODEL, established by removing it from the comparison "
                  "rather than by reasoning about it. tests/resid_probe.cu takes OpenFOAM's converged "
                  "fields INCLUDING ITS OWN nut, sets nuEff = nu + nut, and runs brae's _cpp SIMPLE loop "
                  "with in.turb = nullptr so kepsilon::correct never executes. On simpleCar that converges "
                  "BACK to OpenFOAM: U 2.76e-02 and p 3.14e-02 in the max norm after 300 iterations, "
                  "against 5.9e-01 and 9.8e-01 when brae runs its own turbulence. So the momentum and "
                  "pressure transcription is CORRECT on this mesh, including the non-orthogonal "
                  "correction, and the disagreement is entirely in the k-epsilon model. Consistent with "
                  "the converged fields: k, epsilon and nut are all ~97-99% different while U is 50%. "
                  "A note on what does NOT diagnose this: the pressure equation's residual at a converged "
                  "SIMPLE state is a cancellation and is LARGER on pitzDaily (3.70e-01), where brae agrees "
                  "to 5e-03, than on simpleCar (1.82e-01), where it does not -- so that residual cannot be "
                  "used to rank cases, only the frozen-nut experiment settles it. "
                  "A kEpsilon _cpp REFERENCE NOW EXISTS (kEpsilon_cpp.cu, transcribed from kEpsilon.C) -- "
                  "it was the one turbulence model wired straight to CUDA without one, and it is the one "
                  "that turned out to be wrong. It carries a residuals out-parameter and a diagnostic "
                  "term mask so a disagreement can be bisected. ONE REAL DEFECT FOUND AND FIXED: the "
                  "epsilon/nut wall treatment was keyed on the MESH PATCH TYPE (fvp[pi].type == \"wall\") "
                  "where OpenFOAM keys on the BC TYPE -- createAveragingWeights counts the faces whose "
                  "epsilon field carries an epsilonWallFunction. The two coincide on pitzDaily, whose "
                  "walls carry the wall function explicitly, which is why it survived this long. They do "
                  "NOT coincide on simpleCar: its 0/epsilon sets \"(body|upperWall|lowerWall)\" to "
                  "epsilonWallFunction and then a trailing \".*\" entry overrides it to zeroGradient. "
                  "OpenFOAM resolves that the same way brae does -- last matching regex wins, confirmed by "
                  "OpenFOAM's own written 20000/epsilon showing `type zeroGradient` on body -- so OpenFOAM "
                  "applies NO epsilon wall function there and brae was applying one on three patches. Same "
                  "story for nut, whose nutkWallFunction is overridden by a \".*\" calculated entry. Fixed "
                  "by adding EpsilonWallFunctionPatchField and an isEpsilonWallFunction() discriminator "
                  "(brae had mapped the type to plain zeroGradient, losing the distinction). 27/27 scoped "
                  "tests still pass. Effect on simpleCar: 5.04e-01 -> 4.91e-01, so it is a genuine defect "
                  "but NOT the dominant one. A SECOND GAP found and measured as inert here: the k and "
                  "epsilon equations never had `bounded` wired, though simpleCar asks for `bounded Gauss "
                  "upwind`; adding it changes the residual by nothing (4.2293e-02 either way), as expected "
                  "for a term that vanishes where phi is conservative. THE BAD TERM IS NOT PINNED. A term "
                  "bisect over the epsilon equation -- dropping the production, the divU SuSp, the "
                  "destruction and the diffusion in turn -- makes the residual WORSE in every case on "
                  "simpleCar (6.4e-01, 4.2e-02, 6.3e-01, 2.1e-01 against 4.2e-02), so every term is doing "
                  "useful work and no single one is wrong. CAVEAT on the headline number: the probe runs "
                  "UNRELAXED while OpenFOAM's logged residual is on the relaxed matrix, so the 4300x ratio "
                  "against OpenFOAM's log is inflated; what is like-for-like is pitzDaily 9.3e-06 against "
                  "simpleCar 4.2e-02 under identical probe settings. "
                  "A THIRD DEFECT, found while chasing the second and worth its own line because it "
                  "actively misleads: the field WRITER emits each boundary patch's INPUT specification "
                  "rather than the computed boundary values. brae writes `nut ... value uniform 0` on "
                  "pitzDaily's upperWall, where the nutkWallFunction certainly produced something else, "
                  "and the same on simpleCar's body. That is not a solver error -- the momentum path takes "
                  "its boundary nuEff from deviceBoundaryNut, not from the file -- but every written "
                  "boundary field is unusable for comparison against OpenFOAM, and it cost a full "
                  "investigation branch here: OpenFOAM's `calculated` nut on simpleCar's body carries "
                  "0.6-2.5 while brae's file said 0, which looked exactly like the defect being hunted. "
                  "It is not: for a zeroGradient k and epsilon, OpenFOAM's assignment Cmu*k_b^2/eps_b "
                  "reduces to the adjacent cell's nut, which is what deviceBoundaryNut already supplies "
                  "for a non-wall face (and the kernel even carries calcMask/kBnd/epsBnd parameters for "
                  "the general case). THE BAD TERM REMAINS UNPINNED after three defects found. "
                  "The gate therefore BUILDS its case -- pitzDaily, which brae reproduces to "
                  "~5e-03 on U, with a topoSet porous cellZone cut out of the middle -- so the porosity "
                  "is the only variable. Result: U 5.29e-03 and p 2.72e-02 with the porosity against "
                  "1.62e-02 and 1.66e-01 without it, i.e. 3.1x and 6.1x, which is the control. "
                  "RAS/kEpsilon passes the "
                  "envelope check and the hook IS now wired to deviceKEpsilonCorrect: the hook also owns the "
                  "nuEff refresh (nu + nut, boundary value from deviceBoundaryNut's wall function, never "
                  "the owner cell), which is what makes the lagged coupling work without the driver "
                  "knowing the model. "
                  "CONVERGENCE GAP -- FOUND AND FIXED. The CUDA driver allocated its PressureMatrix and "
                  "folded diagonal fresh every iteration while the AMG hierarchy -- and the V-cycle/PCG "
                  "graph caches keyed on that fine matrix -- persisted across iterations. Iteration 1 was "
                  "exact (dU 8.2e-14 from identical inputs) and iteration 2 was wrong by 1.3e-01, with "
                  "EVERY per-stage test still passing at 1e-16. Fixed by holding the pressure buffers in "
                  "SolverWorkspace, matching how device_simple_foam.cu keeps them as members. All four "
                  "drivers (old host, old GPU, _cpp, CUDA) now agree over 60 iterations "
                  "(iter 59: 2.9136e-02 / 2.914e-02 / 2.91e-02 / 2.91e-02) and the rebuilt solver "
                  "converges end-to-end: 1.0 -> 0.391 -> 0.0615 -> 0.0284 over 60 iterations. "
                  "Regression: tests/test_simple_step_cuda.cu gained a multi-iteration mode "
                  "(ctest simple_step_cuda_loop, 10 iterations, tight solves) -- verified to FAIL "
                  "(U 1.5e-01, p 3.7e-01) when the buffers are made transient again. "
                  "TWO EARLIER DIAGNOSES IN THIS INVESTIGATION WERE WRONG and are retracted: the "
                  "host-vs-GPU 'convergence gap' was a sampling artifact (reading lines 1/5/10/20 of an "
                  "oscillating series), and the 'HbyA wrong by 42%' was a reference rebuilt without the "
                  "momentum predictor. Ruled out on the way: bounded, the non-orthogonal correction, "
                  "solver tolerances, deviceGaussGrad (4e-15), AMG face-weight slicing (no effect -- "
                  "internal faces already come first). Found and fixed as genuine silent-substitution "
                  "holes in the envelope guard: `bounded` and the non-orthogonal laplacian correction."),

        dict(name="cuda_vs_reference", of_symbol="(brae-specific)",
             of_file="-",
             classification="GPU_REQUIRED", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_simple.cu, src/cuda/device_fvm.cu",
             brae_target="src/applications/solvers/simpleFoam/",
             validation="tests/test_gpu_vs_cpp.cu -- CUDA against the _cpp reference at STAGE granularity, "
                        "run on BOTH a laminar case (matrixDumpAsym/282) and a TURBULENT one "
                        "(pitzDailyTurb/1576, nuEff varying per cell and per boundary face). "
                        "PRESSURE: rAU 1.5e-16, laplacian upper/lower 0 diag 1.4e-16, pEqn.flux() 0, "
                        "setReference 0. MOMENTUM: div(phi,U) upper/lower 0 diag 4.6e-18, divDevReff "
                        "source 4.8e-15/5.6e-16/2.4e-16, H(U) 3.8e-16/2.3e-16/2.4e-16, phiHbyA 0, "
                        "corrector 0. TURBULENCE: GbyNu 7.6e-17, nut 0. Three controls fire.",
             note="Closes the chain OpenFOAM -> _cpp -> CUDA. Every other GPU test compares the device "
                  "against CPU code written inline in that same test, which proves consistency but not "
                  "correctness. Running on a turbulent case as well is what makes it load-bearing: with "
                  "constant nuEff a kernel that mishandles a per-face diffusivity, or reads the owner "
                  "cell's value on a wall instead of the patch value, still agrees perfectly. NOT yet "
                  "compared this way: the linear solves themselves, the wall functions (G0/eps0), and the "
                  "k/epsilon transport assembly."),

        # ---- determinism ---------------------------------------------------------------------
        dict(name="deterministic_assembly", of_symbol="(brae-specific)",
             of_file="-",
             classification="GPU_REQUIRED", status="REUSE_EXISTING",
             brae_existing="src/matrices/lduMatrix/lduMatrix/reductions.cu",
             brae_target="src/matrices/lduMatrix/lduMatrix/",
             validation="tests/determinism_gate.sh -- pitzDaily (kEpsilon) and pitzDailyKOmega bit-identical "
                        "over two runs; verified to 200 iterations, plus airfoil and backwardFacingStep2D. "
                        "Carries a 1-ULP negative control.",
             note="DONE for the incompressible simpleFoam path. Was 3.6e-02 after 20 iterations, now 0. "
                  "Three scatter sites, all converted to fixed-order gathers: AMG restriction "
                  "(rc[map[c]] += r[c], hit every level of every V-cycle of every PCG iteration), the "
                  "turbulence wall functions (cells with >1 wall face), and the eps setValues constraint "
                  "(cells with >1 constrained face). The last two are RARE -- bit-identical at 1/5/8/10/15 "
                  "iterations and different at 12 -- so intermittency, not just a systematic offset, is "
                  "what the gate has to catch. STILL OPEN: the opt-in BRAE_AMG_SA path still scatters, and "
                  "the cyclic/AMI (42 sites) and distributed (device_halo) paths are untouched."),
            dict(name='actuationDiskSource',
             of_symbol='Foam::fv::actuationDiskSource',
             of_file='src/fvOptions/sources/derived/actuationDiskSource/actuationDiskSourceTemplates.C',
             classification='MODEL',
             status='PORTED',
             brae_existing='src/finiteVolume/cfdTools/fvOptions/actuation_disk.cu',
             brae_reference='src/finiteVolume/cfdTools/general/fvOptions/actuationDiskSource_cpp.cu',
             brae_target='src/finiteVolume/cfdTools/general/fvOptions/',
             validation='tests/actuationdisk_vs_openfoam.sh and tests/turbinesiting_cuda_vs_openfoam.sh, on validation/turbineSiting -- the tutorial on its REAL 120246-cell snappyHexMesh. OpenFOAM WRITES ITS OWN ANSWER to postProcessing/<name>/<t>/actuationDiskSource.dat (Uref, Cp, Ct, a, T, diskDir) every iteration, so the model gate needs no reimplementation of anything: brae reproduces Uref, a and T BIT-EXACTLY (rel 0.00e+00, both turbines). The exactness comes from respecting the one-iteration offset -- calcFroudeMethod runs while UEqn is assembled at iteration N, so the row at time N was computed from the field WRITTEN at N-1; comparing brae on the N-1 field against the row at N is the same arithmetic on the same input. The end-to-end gate then runs the whole case through the CUDA V2 driver at DEEP convergence: U 1.25e-03, p 1.86e-03, k 1.37e-02, epsilon 2.90e-02, nut 1.22e-02, and the converged thrust within 2.8e-04 on both turbines. Control: the same run with the turbines removed is 23x worse on U.',
             note='Froude variant only. a = 1 - Cp/Ct (sink cancels: OF scales BOTH by sink_), T = 2*rhoRef*diskArea*(Uref & diskDir)^2*a*(1-a) with Uref the mean over the monitor cells and rhoRef 1 on the incompressible path, distributed as (V[c]/Vdisk)*T*diskDir over the disk cells. REFUSED rather than approximated: variant variableScaling (a different thrust law), and Cp/Ct as a non-constant Function1 (OF evaluates them against mag(Uref) every iteration, so a table is a thrust curve -- taking its first knot would run a plausible turbine at the wrong operating point). Both previously caused the source to be dropped SILENTLY, which converges perfectly well to a turbine site with no turbines in it. THE SIGN is the same trap as rotorDiskSource and is documented there: the momentum source LOSES the thrust, via `UEqn == fvOptions(U)`.'),
        dict(name='atmBoundaryLayer',
             of_symbol='Foam::atmBoundaryLayer',
             of_file='src/atmosphericModels/derivedFvPatchFields/atmBoundaryLayer/atmBoundaryLayer.C',
             classification='BOUNDARY_CONDITION',
             status='PORTED',
             brae_existing='src/finiteVolume/fields/fv_patch_field.cuh',
             brae_target='src/finiteVolume/fields/',
             validation="tests/turbinesiting_cuda_vs_openfoam.sh -- turbineSiting is the only case in validation/ with an ABL inlet, and it now converges to OpenFOAM's fixed point.",
             note="PROFILE ORIGIN, fixed 2026-08-20: OF measures the log profile from the PATCH's own lowest point, `groundMin = zDir & boundBox(patch.localPoints()).min()` (atmBoundaryLayer.C:45,218), not from z = 0. brae used the raw height. turbineSiting sits at a real elevation near z = 1000 m, so the inlet was the logarithm of the ALTITUDE rather than a boundary layer -- ~40% fast at the inlet, and the case never converged in either brae path (p stuck at ~3e-3 against OpenFOAM's 8e-5, in a limit cycle). It is the patch POINTS, not the face centres: FvPatch gained ppMin for this. Also: OF v2412 WRITES flowDir/zDir/Uref/Zref/z0/d as Function1/PatchFunction1 (`flowDir constant (1 0 0)`), which brae's reader could not read back and its writer did not emit at all -- so no OpenFOAM-written ABL field could be read, and no brae-written one could be restarted from."),
        dict(name='fusedGauss',
             of_symbol='Foam::fv::fusedGaussConvectionScheme, fusedGaussDivScheme, fusedGaussGrad, fusedGaussLaplacianScheme',
             of_file='src/fused/finiteVolume/fusedGaussConvectionScheme.C',
             classification='SHARED_NUMERICAL',
             status='EQUIVALENT',
             brae_existing='src/applications/solvers/common/scheme_parse.cuh (readFvSchemesText)',
             brae_target='src/applications/solvers/common/',
             validation="tests/fusedgauss_vs_openfoam.sh, on validation/pitzDaily with OpenFOAM's OWN pitzDaily and pitzDaily_fused scheme files. Four assertions: (1) the equivalence measured from OpenFOAM ITSELF -- real OpenFOAM runs the same case under both spellings and lands within 3.4e-05 (U) to 4.5e-04 (nut), the gap being fused summation ORDER, not a different scheme; (2) brae is BIT-IDENTICAL on the two spellings (Linf 0.000e+00 on U, p, k, epsilon, nut), which is the assertion that the rewrite is a rename and nothing else; (3) end to end against OpenFOAM on the fused case -- U 1.02e-04, p 3.55e-04, k 7.78e-04, epsilon 9.57e-04, nut 1.29e-03, TIGHTER than OpenFOAM's own two spellings differ because brae runs identical code for both; (4) control: div(phi,U) forced to upwind is 119x worse.",
             note="fusedGauss is NOT a different discretisation. Every member of the family is the plain Gauss scheme with the field-expression temporaries replaced by fused loops, and OpenFOAM leaves the original lines in as comments directly above the fused calls (`//fvm.lower() = -weights.primitiveField()*faceFlux.primitiveField();` above `multiplySubtract(...)`); fusedGaussLaplacianScheme::fvmLaplacian is line-for-line identical to gaussLaplacianScheme::fvmLaplacian, and fusedGaussGrad::calcGrad computes `area*(lambda*(own - nei) + nei)` summed and divided by V, which is gaussGrad. pitzDaily_fused is stock pitzDaily with `libs (fusedFiniteVolume)` and the scheme words renamed -- diff the two system/ directories. brae rewrites the WHOLE TOKEN `fusedGauss` to `Gauss` once, in readFvSchemesText, so every downstream parser (V2's and the shipped path's) is unchanged and `fusedGauss <scheme brae cannot do>` is still refused by name. It is ANNOUNCED with a new notice kind, `equivalent` -- reporting an exact equivalence as `approximated` would teach the reader to discount the approximated lines that really do differ."),
        dict(name='kOmegaSSTLM',
             of_symbol='Foam::RASModels::kOmegaSSTLM',
             of_file='src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM.C',
             classification='MODEL',
             status='PORTED',
             brae_existing='src/cuda/device_komega_sst.cu (lmReThetatPrepKernel / lmGammaPrepKernel / lmGammaEffKernel / deviceKOmegaSSTLMCorrect)',
             brae_reference='src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM_cpp.cu',
             brae_target='src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/',
             validation="TWO gates. (1) tests/lm_cuda_vs_openfoam.sh -- the CUDA V2 driver end to end on T3A: U 7.89e-05, p 8.88e-05, k 1.41e-03, omega 2.55e-06, nut 1.48e-03, TIGHTER than the _cpp reference it was ported from because the driver reads the case's linear-solver settings. It runs the MODULE comparison first (tests/lm_cuda_probe.cu): Fthetat 1.3e-13, both reaction preps 5.5e-14, gammaIntEff 6.2e-16, the two solved transports 9.8e-08 -- so an end-to-end agreement reached by cancellation cannot pass. Control: plain kOmegaSST through the same driver is 430x worse on U and 431x on k. (2) tests/lm_cpp_vs_openfoam.sh, on validation/T3A -- the ERCOFTAC T3A flat-plate transition tutorial on its own 26820-cell blockMesh, run by OpenFOAM v2412 to the case's own residualControl at t=269. END TO END from 0/, 500 iterations, no CUDA: U 2.04e-04, p 1.26e-03, k 1.39e-03, omega 5.02e-05, nut 1.57e-03, settled at U 9.7e-09. CONTROL: the same case with the transition model removed (plain kOmegaSST, identical solver and schemes) against the same reference is 167x worse on U, 437x on k and 114x on nut -- the transition model is doing the work, not the bounds.",
             note="kOmegaSSTLM IS kOmegaSST plus two transport equations (ReThetat, gammaInt) and three overrides of the base model, so the overrides live in kOmegaSST_cpp as LMHooks rather than as a branch: F1 = max(kOmegaSST::F1, exp(-(Ry/120)^8)) with Ry = y*sqrt(k)/nu, Pk *= gammaIntEff, epsilonByk *= clamp(gammaIntEff, 0.1, 1). That F3 is NOT kOmegaSSTBase's F3 near-wall switch, which brae refuses -- two different functions with the same name in a base and its derived class. THE ORDER IS THE MODEL: correct() runs kOmegaSST::correct() FIRST, so k and omega advance on the PREVIOUS iteration's transition state, and gammaIntEff is constructed ZERO, so the first outer iteration has no turbulent production at all. Within correctReThetatGammaInt, Fthetat is formed ONCE from the OLD ReThetat and OLD gammaInt and reused in gammaSep at the end, while ReThetac/Rev/RT see the NEW ReThetat and gammaSep sees the NEW gammaInt. ReThetat0 is a per-cell fixed-point iteration on lambda with NO iteration cap (OpenFOAM warns past maxLambdaIter and carries on); transcribed as written. DgammaIntEff is nut + nu -- there is no sigmaGamma in this model. REFUSED: MRF and fvOptions on the two transition equations."),
        dict(name='turbulenceSchemesInV2',
             of_symbol='fvSchemes divSchemes/laplacianSchemes, applied to the turbulence equations',
             of_file='src/finiteVolume/finiteVolume/laplacianSchemes/gaussLaplacianScheme/gaussLaplacianScheme.C',
             classification='SHARED_NUMERICAL',
             status='PORTED',
             brae_existing='src/applications/solvers/simpleFoam/simpleFoamV2.cu',
             brae_target='src/applications/solvers/simpleFoam/',
             validation='tests/lm_cuda_vs_openfoam.sh asserts the driver REPORTS reading `div(phi,k): bounded linearUpwind` and `non-orthogonal correction ON`, and then measures the answer end to end; every other V2 gate (stock pitzDaily, SST, SA, MRF, turbineSiting, fusedGauss) was re-run against the change and is unmoved.',
             note="The V2 driver passed linearUpwindK, linearUpwindOmega and the laplacian nonOrth flag as literal FALSE to the turbulence hook, while its own setup line printed what the case had asked for. A case naming `bounded Gauss linearUpwind grad` and `Gauss linear corrected` therefore ran upwind and orthogonal on every turbulence scalar -- printed one thing and did another, which is worse than not reading the entry at all. The same gap existed in the _cpp kOmegaSST reference (no linearUpwind parameter, orthogonal laplacian); on T3A honouring linearUpwind was worth a factor of 12 on the end-to-end error and the corrected laplacian was worth 5.2e-04 on ReThetat, whose mesh reaches 43.8 degrees non-orthogonality. OpenFOAM applies the case's laplacianScheme to EVERY laplacian, the turbulence equations included."),
        dict(name='lmLambdaIteration',
             of_symbol='Foam::RASModels::kOmegaSSTLM::ReThetat0',
             of_file='src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM.C',
             classification='MODEL',
             status='PORTED',
             brae_existing='src/cuda/device_komega_sst.cu (lmReThetatPrepKernel + lmLaunchReThetatPrep)',
             brae_reference='src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM_cpp.cu',
             brae_target='src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/',
             validation='tests/lm_cuda_probe.cu -- the ReThetat reaction (which ReThetat0 feeds) matches the _cpp reference to 5.5e-14 on T3A.',
             note="OpenFOAM's lambda loop is `do { ... } while (lambdaErr > lambdaErr_)` with NO cap at all -- maxLambdaIter is only the threshold past which it WARNS and carries on. The device kernel stopped at maxLambdaIter (10), returning a lambda that had not met the case's own lambdaErr, which is a different ReThetat0 correlation, silently. It now iterates to lambdaErr with a hang guard far above any converging cell, reports the worst count with an atomicMax, warns exactly where OpenFOAM warns, and REFUSES at the guard. Also fixed: the kernel used 1e-37 for deltaU and deltaMin where OpenFOAM uses SMALL (1e-15) -- Us appears squared in a denominator and delta sits under y in y/delta, so the value of the floor is part of the model where the flow stagnates. And LMCoeffs were hardcoded to OpenFOAM's defaults, so a case overriding ca1 or cThetat ran the defaults; they are now read from the RAS dict."),
        dict(name='linearUpwindTurbulence',
             of_symbol='Foam::fv::gaussConvectionScheme with linearUpwind, on k/omega',
             of_file='src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearUpwind/linearUpwind.C',
             classification='SHARED_NUMERICAL',
             status='PORTED',
             brae_reference='src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST_cpp.cu',
             brae_target='src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/',
             validation='tests/lm_cpp_vs_openfoam.sh -- T3A asks for `bounded Gauss linearUpwind grad` on k, omega, gammaInt and ReThetat. Honouring it moved the end-to-end U error from 2.43e-03 to 2.04e-04, a factor of 12.',
             note="The _cpp kOmegaSST had no linearUpwind parameter AT ALL, so a case asking for it silently got UPWIND -- a different matrix under the case's own scheme name, which is the substitution class this port exists to eliminate. SpalartAllmaras already had it (SA::DivScheme::linearUpwind); kOmegaSST did not. linearUpwind's matrix IS upwind's; the scheme is the deferred gradient correction subtracted from the source."),
        dict(name='amgInterfaceAgglomeration',
             of_symbol='Foam::cyclicGAMGInterface / cyclicAMIGAMGInterface',
             of_file='src/finiteVolume/fvMatrices/solvers/GAMGSymSolver/GAMGAgglomerations',
             classification='SHARED_NUMERICAL',
             status='PORTED',
             brae_existing='src/applications/solvers/simpleFoam/device_simple_foam.cu (amgFineCoeffKernel + the extended edge list)',
             brae_target='src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/',
             validation='Measured on the conformal pipeCyclic (1250 cells, one periodic pair): pressure iterations 153/118/8/130/129 before, 129/37/14/37/36 after -- ~3.5x on every iteration past the first. Same for a conforming cyclicAMI (133/36/13/38/40). The converged answer is UNCHANGED to every digit printed (U 6.060e-02, p 2.479e-01 against OpenFOAM before and after), which is what a pure preconditioner change must do. 60 gates re-run green, including every AMG/PCG/SpMV/determinism test.',
             note="brae used to switch the pressure solve OFF AMG entirely on any mesh carrying a cyclic or cyclicAMI patch and fall back to Jacobi-PCG -- a diagonal preconditioner on a Poisson problem. The cost was visible against mesh size: 1250 cells with a periodic pair needed 153 iterations while the interface-free 12225-cell pitzDaily needed 18, the O(sqrt(N)) signature of losing the multigrid. Nothing about a periodic edge actually resists agglomeration: buildAMG takes a plain edge list, and an interface entry IS an edge -- (ownCell, nbrCell) for a conformal pair, (ownCell[i], nbrCell[k]) per stencil entry with the AMI weight folded into the face weight otherwise. THE SIGN is the subtlety: an interface entry is ONE direction (Apsi[own] += ifCoeff*psi[nbr]), so each appended edge carries upper = ifCoeff and lower = ZERO, the reverse direction being the matching entry on the other patch; writing ifCoeff into both would count every periodic connection twice. It still comes out symmetric on the coarse grid because the two edges agglomerate to the same coarse face with opposite orientation and agglomerate()'s faceFlip sends the reversed one's upper into cLower. STILL OPEN: a NON-CONFORMING AMI keeps BiCGStab (the operator is not self-adjoint, so CG is invalid) and so still runs unpreconditioned by AMG; and brae's AMG remains ~7x OpenFOAM's GAMG in iteration count here (37 against 5), which is the general brae-AMG gap rather than an interface one. The hierarchy CACHE is bypassed on interface meshes, since it keys on the mesh alone and must not serve a hierarchy built without the interface edges."),
        dict(name='cyclicAMI',
             of_symbol='Foam::cyclicAMIFvPatchField, Foam::AMIInterpolation',
             of_file='src/finiteVolume/fields/fvPatchFields/constraint/cyclicAMI/cyclicAMIFvPatchField.C',
             classification='BOUNDARY_CONDITION',
             status='CPP_REFERENCE',
             brae_existing='src/cuda/device_ami.cu (the shipped device path)',
             brae_reference='src/finiteVolume/fvMatrices/cyclicAMI/cyclicAMI_cpp.cu',
             brae_target='src/finiteVolume/fvMatrices/cyclicAMI/',
             validation="tests/ami_cpp_vs_device.sh. SIXTEEN stages compared against their device twins at identical inputs, bounded at 1e-12 because these are the same arithmetic in two places: interpolate, interpolateVec (rotated) x3 components, faceValue, assembleLaplacian ifCoeff + diag, assembleMomentum ifCoeff + diag, amul, flux, addH, addDiv, addGrad, and lapCorr on all three components. All pass at 0.000e+00 -- bit-identical. lapCorr is the laplacian's DEFERRED non-orthogonal correction and the one with a trap in it twice over: the neighbour's velocity gradient is a TENSOR, so a rotational interface transforms it as R G R^T on BOTH indices (R G alone leaves the derivative index in the neighbour's frame -- invisible on a translational interface, wrong by the sector angle on a rotational one); and the device packs gUx[l]/gUy[l]/gUz[l] = d(U_l)/d(x,y,z), row = velocity COMPONENT, which is the TRANSPOSE of fvc::gaussGrad's gradU_ij = d(U_j)/d(x_i). The gate hands both sides the same gradient explicitly so neither the rotation nor the packing can hide a difference. Run on validation/pipeCyclicAMI, a NON-CONFORMING rotational AMI (400 source faces against 250, with 50 of them carrying 4-entry stencils) at a developed state -- the first version of this gate shipped a CONFORMAL case and passed all twelve without ever executing the multi-entry interpolation, since a conformal stencil is one entry of weight 1. It now asserts both that some stencil has more than one entry and that the interface flux is nonzero, so it cannot pass on arithmetic it did not run. A SECOND invocation cross-checks the AMI GEOMETRY against the separately-gated cyclic geometry on validation/pipeCyclicAMIConformal, whose cyclic twin is derived by changing one word in constant/polyMesh/boundary: deltaCoeffs and weights agree to 5e-16. The two invocations want opposite meshes on purpose -- the stencil test needs a non-conforming interface, and only a conforming one has a cyclic twin.",
             note="WHY A HOST REFERENCE FOR CODE THAT ALREADY WORKS: brae's device AMI is one fused path, so a disagreement anywhere in it produced a single number for the whole interface. pipeCyclic puts 97% of its momentum residual on interface cells with the interior EXACTLY zero, and four passes of reading the device code did not explain it -- reading is not measuring. Every other defect in this port fell out quickly once a _cpp reference existed to compare stage by stage. WHAT IT CAUGHT IMMEDIATELY: the reference's own assembleLaplacian sign, backwards on first writing because it was copied from assembleMomentum -- momentum carries -laplacian(nuEff,U) and the pressure equation +laplacian(rAUf,p), so the two interface assemblies have OPPOSITE signs (ifCoeff = -lap, diag += lap versus ifCoeff = +c, diag -= c). WHAT IT NOW RULES OUT: all twelve stages compute the right number from the right inputs -- on a non-conforming interface with 4-entry stencils -- AND the geometry those inputs are built from matches the independently-gated cyclic geometry to 5e-16. So pipeCyclic's interface-localised momentum residual is none of: the sixteen stages, the interface geometry, nuEff (correct to 1.3e-05, and that gap is OpenFOAM's own correctNut), the interface flux (a real defect, fixed, and it moved the residual not at all), or gradU. Composition was inspected too and is symmetric -- the flux lifecycle is correctly ordered and interfaceAddLapCorr/interfaceLapCorrP are called for both cyclic and AMI at all three sites, the gradient is built AMI-inclusively, and divDevReff receives the interface. THE REFRAMING THIS FORCES: every one of those gates proves brae's DEVICE equals brae's REFERENCE, and the reference is a transcription of what the device does. Neither has been checked against OPENFOAM's formulation of a cyclicAMI momentum interface beyond reading internalCoeffs/boundaryCoeffs for div and laplacian. Closing this needs an OF-side oracle -- OpenFOAM's own fvMatrix diagonal and source at the interface cells -- not another brae-to-brae comparison. ONE DIRECTION PER ENTRY is the invariant that runs through the whole reference: an interface entry couples own <- nbr only, the reverse being a separate entry on the paired patch -- which is why amul adds to Apsi[own] alone and why the AMG agglomeration gives each appended edge upper = ifCoeff with lower = zero."),
        dict(name='interfaceFluxOnRestart',
             of_symbol='Foam::surfaceScalarField phi, coupled-patch boundary values (createPhi.H READ_IF_PRESENT)',
             of_file='src/finiteVolume/cfdTools/incompressible/createPhi.H',
             classification='SHARED_NUMERICAL',
             status='PORTED',
             brae_existing='src/applications/solvers/simpleFoam/device_simple_foam.cu (the AMI flux init)',
             brae_target='src/applications/solvers/simpleFoam/',
             validation="Measured on pipeCyclic restarted from OpenFOAM's own converged state: brae's rebuilt interface flux differed from the written one by 4.98e-03 relative on side1 (max 1.2e-04 against values up to 1.7e-02). 37 gates re-run green, restart and roundtrip included.",
             note="brae rebuilt the cyclicAMI interface flux from fvc::flux(U) at construction unconditionally. That is RIGHT for a cold start -- the host phi carries a zeroGradient placeholder on a coupled patch, so its boundary value is neither coupled nor rotated nor conservative -- and WRONG on a restart: OpenFOAM writes the interface flux per patch (pipeCyclic's side1/side2 carry 400 and 250 values) and createFields copies those straight into phi.boundary, past the patch-field layer that would otherwise replace them. That flux is the conservative one the previous run ended with; fvc::flux(U) is not, because U alone does not know what the pressure correction did. readPhiIfPresent now reports which branch it took and the driver restores the written interface flux when there is one. HONEST LIMIT: this was found while hunting pipeCyclic's interface-localised momentum residual and it did NOT explain it -- the residual moved from 2.37977e-02 to 2.38361e-02, i.e. not at all. It is a real defect on its own merits (a restart should resume the flux it stopped on) and it also removes a contaminant from the residual ORACLE on interface cases, but the residual itself remains unexplained."),
        dict(name='coupledInterfaceConvection',
             of_symbol='Foam::fvm::div on a coupled patch -- internalCoeffs = phi*w, boundaryCoeffs = -phi*(1-w); LimitedScheme::calcLimiter, coupled() branch',
             of_file='src/finiteVolume/finiteVolume/convectionSchemes/gaussConvectionScheme/gaussConvectionScheme.C',
             classification='SHARED_NUMERICAL',
             status='PORTED',
             brae_driver='src/applications/solvers/simpleFoam/device_simple_foam.cu (momentum), src/cuda/device_scalar_transport.cuh (k/epsilon/omega/nuTilda/ReThetat/gammaInt)',
             brae_reference='src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSchemes_cpp.cu (limitedLinearWeightsCoupled, limitedLinearVWeightsCoupled), src/finiteVolume/fvMatrices/cyclicAMI/cyclicAMI_cpp.cu (limitedLinearWeights, limitedLinearVWeights, interpolateTensor)',
             brae_target='src/cuda/device_ami.cu (amiLimitedVWeightKernel, amiLimitedWeightKernel), src/cuda/device_cyclic.cu (cycLimitedVWeightKernel, cycLimitedWeightKernel)',
             validation="tests/coupledinterfacescheme_vs_openfoam.sh, run on TWO cases and two different schemes -- ctest coupledinterfacescheme_vs_openfoam (validation/pipeCyclic, `bounded Gauss limitedLinearV 1`) and ctest linearinterfacescheme_vs_openfoam (validation/implicitAMI, `bounded Gauss linear`) -- plus tests/ami_cpp_vs_device.sh STAGE 12. The gate takes its bounds from the caller because they are properties of the CASE, and checks the fields the case actually wrote, so a laminar case is not failed for having no k. Its CHECK 0 asserts both codes reached a steady state, counting a run that satisfied the case own residualControl as well as one that ran to endTime with the residual down: implicitAMI ships endTime 100 and at that point OpenFOAM is at residual 1e-04 while brae is at 1e-07, so comparing there measures relative convergence RATE and reports it as a discretisation error -- which is how this investigation first misread that case. Measured on pipeCyclic, brae against OpenFOAM's converged state -- U 5.942e-02 -> 1.127e-03, p 2.406e-01 -> 5.036e-03, epsilon 2.142e-01 -> 1.708e-02, k 2.302e-01 -> 4.317e-02, nut 2.172e-01 -> 4.718e-02. The gate runs FOUR cases: brae and OpenFOAM under both the limited and the upwind scheme, because matching OpenFOAM on one scheme proves nothing unless the case is shown to discriminate the two (it does: OF limited vs OF upwind is 1.85e-01 on U, 6.78e-01 on k). Re-run against the pre-fix binary it fails on all five fields of check 1 and on the symmetry check, and PASSES checks 2, 3 and 4 -- which is why those three alone were never sufficient.",
             note="FOUND BY THE OPENFOAM-SIDE ORACLE (validation/of_apps/dumpUEqnInterface), and findable no other way: brae's device and its _cpp reference agreed perfectly on the SAME WRONG SCHEME through sixteen stage gates. OpenFOAM assembles a coupled patch with the interpolation weights of the div scheme the case NAMED; brae hardcoded UPWIND -- ifCoeff = -lap + min(phi,0), diag += lap + max(phi,0) -- which is the special case w = pos0(phi). Measured L2 6.86e-01 against OpenFOAM's own boundaryCoeffs, the gap equal to phi*(1-w) face by face. WHAT THE PORT ADDS: w = limiter*wCD + (1-limiter)*pos0(phi), with wCD the patch's own interpolation weight, d = fvPatch::delta() (dOwn minus the TRANSFORMED, AMI-interpolated neighbour delta -- now stored on both interfaces rather than re-derived per consumer), and the limiter from the same NVDTVD/NVDVTVDV r the internal faces use (moved to limitedSchemes_cpp.cuh so there is ONE definition, not two). patchNeighbourField is where the transform lives: a scalar is never rotated, a vector is, a gradient TENSOR is rotated on both indices as R G R^T. `Gauss linear` IS NOW WIRED TOO and needed no kernel at all: OF's linear scheme returns mesh.surfaceInterpolation::weights() (linear.H:106), whose boundary field on a coupled patch IS the patch's own interpolation weight -- the number both interface structs already carry and already use for every face value they form -- so the weight is a pointer, not a computation. It is the widest of the blends in absolute terms (w ~ 0.5, the furthest from upwind's 0 or 1). On implicitAMI it took U from 2.894e-01 to 1.780e-02 and p from 2.610e-01 to 5.081e-03, leaving the case tracking OpenFOAM about as well under `Gauss linear` as under upwind (1.692e-02). The upwind run is bit-for-bit unchanged, which is the built-in control. STILL OPEN at a coupled face: LUST alone (w = 0.75*wCD + 0.25*pos0), which has no simpleFoam tutorial to be gated on -- the fourteen tutorials naming it are fireFoam and pimpleFoam LES. linearUpwind needs nothing -- its MATRIX is upwind's and brae applies the deferred correction separately."),
        dict(name='tutorialCoverage',
             of_symbol='tutorials/incompressible/simpleFoam -- the seventeen stock cases',
             of_file='tutorials/incompressible/simpleFoam',
             classification='COVERAGE',
             status='COMPLETE',
             validation='Nine of the seventeen had a dedicated gate built around whatever defect they exposed (pitzDaily, pitzDaily_fused, T3A, turbineSiting, rotorDisk, pipeCyclic, airFoil2D, mixerVessel2D, simpleCar, windAroundBuildings). Three more were swept against real OpenFOAM and now agree: rotatingCylinders U 2.965e-04 / p 4.640e-04 (6400 cells, laminar, MRF, `bounded Gauss linearUpwind grad(U)`), backwardFacingStep2D U 3.482e-03 / p 8.217e-03 / k 9.267e-03 / omega 7.711e-04 / nut 7.211e-03 (20540 cells, kOmegaSST, SIMPLEC, `bounded Gauss LUST grad(U)`), squareBend U 1.773e-03 / p 1.735e-03 / k 4.199e-03 / epsilon 1.032e-02 / nut 1.623e-03 (112000 cells, kEpsilon, SIMPLEC, `Gauss limitedLinearV 1`). The first two are now gated by tests/tutorial_vs_openfoam.sh, which reads the case straight from the OpenFOAM installation so nothing has to be committed.',
             note="backwardFacingStep2D IS THE ONLY simpleFoam tutorial that names LUST, so it is the whole evidence base for brae's internal LUST weights, and it agrees. That also settles the shape of the remaining LUST gap: the INTERNAL faces are validated here, and only the COUPLED-face weight (w = 0.75*wCD + 0.25*pos0) is still assembled upwind -- with no simpleFoam tutorial pairing LUST with a coupled patch to gate it on, since this case has no cyclic or AMI faces. STILL UNGATED, and why: squareBend needs 765 s of OpenFOAM alone at the iteration count where the comparison is fair, too slow for a routine gate, so its agreement is recorded here and not enforced. ALL 17 NOW RUN AND ALL 20 CONFIGURATIONS ARE GATED. The four that had no comparison are done: motorBike (353700 cells from snappyHexMesh, kOmegaSST -- the only tutorial whose OpenFOAM side runs in PARALLEL, 6 processors then reconstructPar) at U 5.001e-02 / p 5.544e-02 / k 1.041e-01 / omega 5.819e-02 / nut 1.056e-02; pitzDailyExptInlet (ships no Allrun at all -- blockMesh plus simpleFoam) at U 2.866e-02 / p 1.630e-01 / nut 1.185e-01; squareBend at U 1.773e-03 / p 1.735e-03 / k 4.199e-03 / epsilon 1.032e-02 / nut 1.623e-03; bump2D:kOmegaSST at U 5.256e-04 / p 5.193e-04 / k 9.445e-03 / omega 1.125e-05 / nut 1.108e-01. THE ONLY CONFIGURATION brae CANNOT RUN is bump2D:kEpsilonPhitF, whose turbulence model is not implemented and which is refused by name -- gated as a refusal. MOTORBIKE IS THE LOOSEST at ~5% on U, and it is LOCALISED TO THE kOmegaSST CLOSURE. Not a stopping-criterion artefact: run both codes 5x longer, to 2500, and the comparison is unchanged (U 5.58e-02 against 5.00e-02) while OpenFOAM residual falls to 6.0e-05 and brae to 1.4e-04. Not the convection scheme either: forcing div(phi,U) to `bounded Gauss upwind` in BOTH codes, with both then converged to ~5e-06, still reads U 4.672e-02. THE SPLIT THAT NAMES IT is running both codes LAMINAR on the same mesh with the same scheme -- U 9.691e-03 and p 5.696e-03, five times closer -- so the base discretisation (momentum, pressure, the non-orthogonal and skew corrections on a mesh whose non-orthogonality reaches 65 degrees with 12 highly skew faces) is good to about 1% here and the turbulence closure carries the rest. SPATIALLY the difference is compact: the top 1% of cells hold 63.8% of the error^2 and sit at x [-0.23, 1.73], y [-0.23, 0.29], z [0.04, 1.26] -- wrapped around the bike body, not the wake or the far field -- where brae |U| is 7.892 against OpenFOAM 8.759. Seeded at OpenFOAM converged state brae momentum residuals split the same way: Ux 7.04e-05 but Uy 1.65e-03 and Uz 1.23e-03, the cross-stream components ~20x worse and not decaying. STILL OPEN: which part of the kOmegaSST closure. The candidates left are the wall distance (the case asks `method meshWave`, which brae implements, but meshWave on a complex snappyHexMesh body is where an implementation would diverge), the nutkWallFunction on the bike patches, and the F1/F2 blending -- all three read y and all three act exactly where the error is. Extracting OpenFOAM own y for a direct comparison was attempted and did not work (`postProcess -func writeObjects(y)` registers nothing), which is the next thing to solve. Every other loose number here IS the stopping criterion -- squareBend reads 2.759e-02 at its shipped endTime and 1.773e-03 at 8000, pitzDailyExptInlet reads p 1.630e-01 at its own p 1e-2 control and 1.540e-02 at 1e-9, bump2D:kOmegaSST reads nut 1.108e-01 at its 1e-5 control and 9.905e-03 at 1e-8. THE GATE GREW THREE CAPABILITIES for these: a fallback for a tutorial with no Allrun, handling for a parallel/reconstructed layout (drop processor*, rebuild 0 from 0.orig), and an optional endTime override plus a per-case steady-state threshold -- the latter used ONLY for motorBike, and only because running longer was shown to move the residual and not the answer. The control also now skips fields it cannot parse (motorBike 0.orig uses #include and $macro) and fails loudly if it could read NONE, rather than reporting unreadable as bounds-too-loose. THE MULTI-SETUP TUTORIALS ARE NOW RUNNABLE: tests/tutorial_vs_openfoam.sh composes `tutorial:setup[:grading]` the way their own Allrun does (common skeleton, then the setup's 0.orig/constant/system copied over it) and expands turbulentFlatPlate's blockMeshDict.template with the named grading, since that case is a MATRIX of (turbulence model x near-wall y+ grading) rather than a list of cases. Of the five setups: turbulentFlatPlate:kOmegaSST:2200 gated at U 1.905e-05 / p 8.685e-03 / k 8.118e-05 / omega 3.543e-05 / nut 3.766e-03 (the tightest agreement of any tutorial here), bump2D:SpalartAllmaras gated at U 3.845e-04 / p 1.928e-04 / nut 2.070e-02, bump2D:kEpsilonPhitF gated as a REFUSAL (brae has no such model and must say so by name rather than run it as kEpsilon), bump2D:kOmegaSST measured but NOT gated (its residualControl of 1e-05 is too loose to compare the two codes -- see nutLowReWallFunction), and turbulentFlatPlate:kEpsilon DIVERGES (see epsilonWallFunction_lowReCorrection). CONVERGENCE MUST BE ASSERTED, NOT ASSUMED, and this cost real time three separate times in one session: comparing two codes at a fixed iteration count compares TRAJECTORIES unless both have arrived. squareBend reads U 2.759e-02 at its tutorial endTime of 500 and 1.773e-03 at 8000 where both codes sit on the same 2.5e-05 residual plateau -- a factor of 16 from nothing but where the comparison was taken. implicitAMI is worse: at its endTime of 100 OpenFOAM is at residual 1e-04 while brae is at 1e-07. Both gates now check it explicitly, counting a run that satisfied its own residualControl as arrived."),
        dict(name='epsilonWallFunction_lowReCorrection',
             of_symbol='Foam::epsilonWallFunctionFvPatchScalarField -- the `lowReCorrection` switch',
             of_file='src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/epsilonWallFunctions/epsilonWallFunction/epsilonWallFunctionFvPatchScalarField.C',
             classification='SHARED_NUMERICAL',
             status='PORTED',
             brae_driver='src/finiteVolume/fields/foam_field_reader.cuh (parses it off the epsilon BC), src/applications/solvers/common/turbulence_setup.cuh (into KEpsilonCoeffs::epsLowRe)',
             brae_reference='src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu (correct, the wall loop)',
             brae_target='src/cuda/device_kepsilon.cu (wallFnKernel)',
             validation="FOUND by the multi-setup tutorial runner: turbulentFlatPlate:kEpsilon at grading 2200 (y+ ~ 1). Real OpenFOAM runs it to 5000 iterations. brae DIVERGED at ITERATION 10 before this port and reaches ITERATION 395 after it -- a 40x improvement and still a failure, so the case is NOT gated. brae's non-finite guard catches the blow-up and refuses to write a field, which is the right behaviour once it has happened.",
             note="THE SWITCH WAS SILENTLY IGNORED, which was the defect: `lowReCorrection` appeared in brae exactly once, in a COMMENT, and the entry sits inside a boundaryField patch dictionary, which the dict audit does not track per key -- so a case asking for it got the high-Re log-law epsilon and nothing warned. Now parsed and honoured on both paths, as two branches transcribed from epsilonWallFunctionFvPatchScalarField.C: a face with y+ < yPlusLam takes epsilon = 2*k*nu/y^2 in place of Cmu^0.75*k^1.5/(kappa*y) (:242, STEPWISE being that BC's default blender per :413), and its wall production is DROPPED ENTIRELY rather than scaled -- OF's guard is `if (!lowReCorrection_ || (yPlus > yPlusLam))` (:338). Both halves or neither: the log epsilon under-predicts dissipation on a resolved mesh while the production keeps feeding k. Carried on KEpsilonCoeffs::epsLowRe rather than threaded through every signature, and yPlusLam was EXPORTED from nut_wall_function.cu (it was file-local) so the threshold has one definition rather than two. PROVABLY INERT ELSEWHERE: the default is false, which reproduces the previous arithmetic exactly, and only two cases in the entire OpenFOAM tutorial tree set the flag -- turbulentFlatPlate:kEpsilon and bump2D:kEpsilonPhitF, and brae refuses the latter's model outright. STILL OPEN: what remains after iteration 395. The k residual settles to ~1e-06 and then runs away, so it is not the initial transient; not localised this pass."),
        dict(name='nutLowReWallFunction',
             of_symbol='Foam::nutLowReWallFunctionFvPatchScalarField::calcNut -- returns Zero, unconditionally',
             of_file='src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutLowReWallFunction/nutLowReWallFunctionFvPatchScalarField.C',
             classification='SHARED_NUMERICAL',
             status='SUBSTITUTED',
             brae_existing='src/applications/solvers/common/turbulence_setup.cuh (setNutWall), src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nut_wall_function.cuh (wallProductionG0)',
             validation="bump2D:kOmegaSST, via the multi-setup runner. brae announces the substitution -- `nut boundary 'bump' uses nutLowReWallFunction; brae applies nutkWallFunction (log law). Identical only where y+<yPlusLam (resolved mesh)` -- and on THIS mesh the claim holds: real OpenFOAM run with the BC swapped to nutkWallFunction writes `value uniform 0` too, and its two runs are BIT-IDENTICAL in U, k, omega and nut. So bump2D cannot discriminate the two wall functions, and no simpleFoam tutorial found so far can.",
             note="A PORT WAS ATTEMPTED THIS PASS AND BACKED OUT. calcNut() is zero unconditionally, so the exact port is trivial, and it was made in two places (the wall nut itself, and wallProductionG0's `(nutw + nu)` production override, where a LowRe case was falling through to the velocity-based BLENDED fallback). It made brae's wall nut match OpenFOAM's `uniform 0` exactly -- verified against OpenFOAM's own written output -- and yet moved every field AWAY from OpenFOAM on bump2D (U 5.261e-04 -> 1.779e-03, nut 1.103e-01 -> 2.429e-01). The reason is a SECOND inconsistency it exposed: OpenFOAM gives bit-identical results under either BC on this mesh, while brae's two paths differ (nut 3.940e-01 apart at iteration 300) even though BOTH write a zero wall nut -- so brae's nutk and LowRe paths disagree somewhere that is not the wall value, and brae's near-wall y+ regime is the prime suspect. Reverted rather than shipped: a change that is correct by source reading but degrades the only case exercising it is exactly the kind this project's discipline exists to stop. THE 24% IS NOT ITSELF A DEFECT -- bump2D's own residualControl of 1e-05 is far too loose to compare the two codes: at 1e-05 the converged nut fields differ by 2.429e-01 and at 1e-08 by 9.905e-03, with U going 1.779e-03 -> 1.495e-05. Any future work here must converge to 1e-08 before believing a number."),
        dict(name='wallFunctionCellSelection',
             of_symbol='epsilonWallFunction / omegaWallFunction -- applied per BOUNDARY CONDITION, not per patch type',
             of_file='src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/epsilonWallFunctions/epsilonWallFunction/epsilonWallFunctionFvPatchScalarField.C',
             classification='SHARED_NUMERICAL',
             status='PORTED',
             brae_target='src/cuda/device_kepsilon.cuh (isTurbWallPatch + buildDeviceWallData), src/applications/solvers/common/turbulence_setup.cuh (the mask), src/applications/solvers/common/solver_controls.cuh (DeviceSimpleControls::turbWallPatch)',
             validation="LOCALISED on turbulentFlatPlate:kEpsilon at grading 2200, which diverges where OpenFOAM runs to 5000. STILL DIVERGES AFTER THIS FIX, and sooner (iteration 171 against 395), because the override was accidentally stabilising. The remaining defect is now LOCALISED, by running both codes to the SAME iteration on the same mesh from the same start (iteration 100) rather than comparing a transient against a converged state -- the mistake that misread this case twice. Median ratio brae/OpenFOAM: at the wall (y<1e-3) k 0.797 and eps 0.833; in the boundary layer (1e-3<y<0.03) k 0.994 and eps 0.862; in the FREESTREAM (y>0.05) k 5.98 and eps 1.77. So the wall closure and the boundary layer are close and the freestream is not: k/eps comes out 3.4x high there, freestream turbulence decays about 3.4x too slowly, and nut ~ k^2/eps inherits ~20x. The blow-up then starts in the freestream at (x 1.61, y 0.125) and reaches 6.98e+08 by iteration 168. RULED OUT BY MEASUREMENT, not argument: spurious production near the slip top boundary (G = nut*gByNu is 1.0e-09 against eps 1.0e-01 there, eight orders down, and G/eps is ~0 in the runaway band itself); k/epsilon bounding (instrumented -- brae bounds on 279 of 336 field-iterations against OpenFOAM bounding on ~4950 of 5000, a comparable rate, and brae implements OpenFOAM neighbour-average recovery rather than a floor clamp); and the topWall slip boundary itself, whose layer is healthy after this fix (k max 1.18e-01). THE DECAY BALANCE WAS CHASED AND IS NOT A WRONG TERM -- IT IS A ROBUSTNESS GAP. With ONLY the turbulence relaxation tightened (k 0.8 -> 0.4, epsilon 0.7 -> 0.4, nothing else touched) brae completes all 5000 iterations and lands at U 7.160e-03, p 5.615e-03, k 7.200e-02, epsilon 5.813e-02, nut 1.742e-01 of OpenFOAM -- so the SAME discretisation converges to about the right answer once it is walked through the transient. Everything else was excluded by running it: forcing div(phi,k|epsilon) to `bounded Gauss upwind` makes it WORSE, not better (blow-up at 124 against 171), which rules out the convection scheme and its limiter -- maximal numerical diffusion should cure a dispersive wiggle and instead accelerates the failure; turning SIMPLEC off with classic SIMPLE relaxation (U 0.7, p 0.3) is worse again (25). The freestream k/eps ratio running 3.4x high at iteration 100 is therefore a SYMPTOM of the developing runaway, not an independent term error. The freestream k profile along x is non-monotonic in brae (8.5e-04, 1.6e-04, 3.2e-03, 1.8e-04, 6.5e-04 at x = -0.30, -0.10, 0.10, 0.50, 0.99) where OpenFOAM decays smoothly (7.9e-04, 2.8e-04, 1.6e-04, 1.2e-04, 3.7e-05), but since upwind does not fix it that oscillation is downstream of the instability rather than its cause. Residual traces give no warning either -- brae k and epsilon sit within 2-4x of OpenFOAM and decay similarly right up to a sudden runaway. ROOT CAUSE FOUND -- THE PRECONDITIONER SUBSTITUTION ON THE TURBULENCE SOLVE, and it is a STABILITY effect, not the cost-only one brae own notice claims. The case asks for PBiCGStab with DILU on (omega|epsilon|k) at relTol 0.1; brae substitutes Jacobi-BiCGStab and says so as an [approximated] notice reading `same linear system and tolerance -- iteration count and cost differ`. Both codes honour relTol 0.1, but they stop in very different places: measured over 60 consecutive k solves, OpenFOAM DILU-preconditioned solve reduces the residual to a MEDIAN 0.0064 of initial -- one iteration overshoots the 0.1 threshold by more than 10x -- while brae Jacobi solve stops right at it, median 0.0726. In the stiff k-epsilon pair that leaves the two fields mutually inconsistent every outer iteration, and on this y+ ~ 1 mesh the inconsistency runs away. PROOF: set relTol to 1e-3 for (omega|epsilon|k) and change NOTHING else -- the case relaxation stays at its own k 0.8 and epsilon 0.7 -- and brae runs all 5000 iterations and lands at U 1.040e-05, p 3.088e-04, k 1.286e-04, epsilon 2.671e-04, nut 6.332e-04 of OpenFOAM, with a final Ux residual of 3.85e-07 against OpenFOAM 3.49e-07. That is the tightest agreement of any tutorial in this port. THE MODEL ITSELF WAS VERIFIED FAITHFUL along the way, term by term against kEpsilon.C: both reaction kernels including the Sp/SuSp splits and the C1*Cmu*k*GbyNu production form that avoids eps/k; the solve order (epsilon, bound, k, bound, correctNut); relax BEFORE the near-wall setValues, using the relaxed diagonal; and bound() implementing the neighbour-average recovery at a comparable firing rate. FIXED: the turbulence BiCGStab now takes the DILU preconditioner the case asks for. brae already had an exact, level-scheduled device DILU (device_dilu.cuh) wired to the MOMENTUM solve only, so this is wiring rather than new numerics -- one DeviceDilu now serves both, since the level schedule is a property of the mesh addressing and diluUpdate recomputes rD from whichever matrix is being solved. Selected by DeviceSimpleControls::diluKE, read from the turbulence solver sub-dictionary AFTER gsK/gsEps (a smoothSolver field has no preconditioner to honour) and resolved with the regex-aware subDict, so the usual (omega|epsilon|k) key needs no special case. Held in a file-scope accessor next to turbStore rather than threaded through deviceKEpsilonCorrect, deviceKOmegaSSTCorrect, deviceSpalartCorrect and deviceEnergyCorrect, which would pass the same pointer four times. RESULT: brae k solve now reduces the residual to a median 0.00680 of initial against OpenFOAM 0.0064 -- it was 0.0726 -- and turbulentFlatPlate:kEpsilon runs all 5000 iterations UNMODIFIED, landing at U 1.812e-05, p 2.470e-04, k 3.296e-04, epsilon 3.566e-03, nut 2.299e-04 of OpenFOAM, with a final Ux residual of 3.59e-07 against 3.49e-07. Gated as ctest tutorial_turbulentflatplate_kepsilon. THE NOTICE TEXT WAS ALSO WRONG IN TWO WAYS and is corrected: it exempted the WORD DILU rather than the fields brae actually wires it on, so a DILU request on k/epsilon was answered with Jacobi and NO notice at all; and it described the substitution as differing in iteration count and cost, when what it really changes is where the solve STOPS -- and that can decide whether the case runs. Measured in the topWall-adjacent layer -- 545 cells, exactly topWall's face count -- brae at iteration 100 against OpenFOAM: epsilon 1.556e-03 vs 3.055e-02 (20x LOW), nut 1.073e-04 vs 1.044e-07 (1028x HIGH), k 1.147e-03 vs 1.551e-04 (7.4x high), |U| 69.40 vs 69.48 (right). brae's epsilon there IS the wall-function formula: Cmu^0.75*k^1.5/(kappa*y) with y = 0.0098, the half-height of that cell layer, gives 1.589e-03 against the measured 1.556e-03, a 2% match. So epsilon is being PINNED by a wall function instead of transported, nut = Cmu*k^2/eps comes out ~1000x high, freestream k never decays (it sits at its 1.08e-03 inlet value while OpenFOAM's decays to 1.55e-04) and runs away.",
             note="brae SELECTS WALL-FUNCTION CELLS BY PATCH TYPE, OpenFOAM BY BC TYPE. buildDeviceWallData takes every patch with `type wall`; OpenFOAM's epsilonWallFunction is a boundary condition OBJECT on the epsilon field, so only patches whose epsilon BC is that type get a cornerWeights_ entry and an epsilon0/G0 override. turbulentFlatPlate's `topWall` is typed `wall` in constant/polyMesh/boundary but carries U `slip`, k `zeroGradient`, epsilon `zeroGradient`, nut `calculated` -- physically a slip far-field. OpenFOAM applies nothing there; brae applies the full epsilon wall function to its 545 faces. FIXED: the wall set is now built from the patches whose epsilon/omega BC IS the wall function (isTurbWallPatch -- a wall-typed patch AND named by that BC), which is the missing half of a check brae already had in the other direction (guardWallFn errors when a wall-function BC sits on a non-wall patch). One predicate, shared by the DeviceWallData faces and the wall-face -> boundary-face map, because those two drifting apart would silently point the map at the wrong faces. An EMPTY mask falls back to the patch type, which is what the SA and LES paths use -- they have no epsilon/omega field to read. MEASURED EFFECT on the topWall layer at iteration 100: epsilon 1.556e-03 -> 9.940e-02 against OpenFOAM 3.055e-02 (from 20x LOW to 3.3x high) and nut 1.073e-04 -> 3.912e-06 against 1.044e-07 (from 1028x high to 37x). THE MASK MUST BE RESOLVED WITH findPatchEntry, NOT a name comparison: a boundaryField key can be an exact name, a GROUP or a REGEX, resolved in that order with the last match winning. The first attempt compared names, so backwardFacingStep2D -- which writes its wall BC as the regex (upperWall|lowerWall) -- matched nothing, the mask came out all zeros and the wall function was removed from the ENTIRE case: U 3.482e-03 -> 1.400e-01 and omega 7.711e-04 -> 9.905e-01 against OpenFOAM. The turbulence suite caught it; it is the reason that suite is the gate for this change. RULED OUT along the way, each by measurement rather than argument: the steady discretisation (seeded at OpenFOAM's own state brae is stable -- Ux residual 2.9e-05, epsilon 3.0e-06, drift 5.9e-05 in U over 5 iterations); k/epsilon bounding (brae's deviceBoundField does implement OpenFOAM's neighbour-average recovery, not a floor clamp, and IS called from deviceSolveScalarTransport -- it simply never prints, so a `bounding` grep of brae's log measures nothing); and the Uz residual of ~1 on this 2D empty case, which is a 0/0 normalisation artefact (|Uz| max 8.4e-17)."),
        dict(name='cellWallDist_meshWave_complexGeometry',
             of_symbol='Foam::patchWave::correct -> cellDistFuncs::correctBoundaryCells (the useCombinedWallPatch branch)',
             of_file='src/meshTools/cellDist/patchWave/patchWave.C',
             classification='SHARED_NUMERICAL',
             status='PORTED',
             brae_target='src/finiteVolume/fvMesh/cell_wall_dist.cuh (cellWallDist, the correctWalls block)',
             validation='Measured directly against OpenFOAM own field on motorBike (353700 cells, 68 wall patches) with validation/of_apps/dumpWallDist against brae BRAE_DUMP_STAGE stage_y. BEFORE: L2 rel 1.688e-04 with 963 cells more than 1% out, brae reading a median 0.894 of OpenFOAM and 0.49 at worst. AFTER: L2 rel 1.506e-15, ZERO cells out, max|dy| 7.1e-15 -- machine precision.',
             note="THE SOURCE HAS TWO BRANCHES AND THE OBVIOUS ONE IS DEAD. patchWave::correct() reads `if (cellDistFuncs::useCombinedWallPatch) correctBoundaryCells(patchIDs.sortedToc(), true, ...) else { correctBoundaryFaceCells(...); correctBoundaryPointCells(...); }` and that switch defaults TRUE (cellDistFuncs.C:42). The pair of functions the class is normally read for is the branch commented `Backwards compatible` and does NOT run. correctBoundaryCells instead builds ONE uindirectPrimitivePatch from every wall patch's faces and does both passes on THAT, so a point's face set spans patch boundaries. Porting the per-patch branch -- which is what a careful reading of cellDistFuncs.C alone produces -- measured WORSE, twice: both passes gave 4680 cells out instead of 963 with brae reading a median 1.295 of OpenFOAM, and the point-cell half alone gave 2847 at 1.239, in both cases too LARGE because a per-patch point sees only a fraction of the faces around it. Those attempts were reverted before the real branch was found. THE INSTRUMENT THAT FOUND IT was validation/of_apps/dumpWallDistDetail, added here: cellDistFuncs' correction routines are public and take the nearestFace Map by reference, so calling them exactly as patchWave does and writing that Map out gives OpenFOAM's own per-cell face choice. It showed OpenFOAM final y never exceeding its own correctBoundaryFaceCells/PointCells output (max ratio exactly 1.000 over 5325 differing cells, median 0.819) -- a field that is supposed to be an unconditional overwrite cannot be uniformly smaller than itself, which is what pointed at a different routine doing the work. THE PORTED RULE: combined patch, faces in patch-then-face order; meshPoints in first-appearance order walking those faces; pass 1 gives each wall face's owner smallestDist over getPointNeighbours on the combined patch, written unconditionally so a cell owning several wall faces keeps the LAST; pass 2 lets the first combined meshPoint to reach an unclaimed cell set it from that point's combined faces and LOCKS it. EFFECT ON THE CASE IT WAS FOUND ON: motorBike U against OpenFOAM went 5.580e-02 -> 4.021e-02 at 2500 iterations and p 7.066e-02 -> 4.651e-02, about 28% of the U gap. NOT the whole gap, which matches what the correlation said before the fix -- the y-error cells and the velocity-error cells shared a region but only overlapped 3.2x above baseline, so y was one contributor and not the only one. What remains of motorBike's difference is still open."),
    ],

    # ==================================================================================================
    # rhoSimpleFoam -- the GROUND-UP port. Nothing here is copied from brae's existing rhoSimpleFoam or
    # from simpleFoam: every entry is transcribed from the OpenFOAM file named in `of_file`, host `_cpp`
    # first, and only then moved to CUDA. The existing gpuRhoSimpleFoam.cu stays untouched and keeps its
    # gates until this path outruns it.
    #
    # WHY GROUND-UP RATHER THAN KEEPING WHAT WORKS. The existing solver was built the way this manifest
    # exists to prevent: implement what a tutorial exercises, run it, find the next missing piece from a
    # wrong answer. That produces a solver which passes the cases it was debugged against and makes no
    # statement about anything else. OpenFOAM's whole rhoSimpleFoam driver is 446 lines across 7 files, so
    # the systematic version is small; the closure BELOW it (thermo, the compressible turbulence set) is
    # where the real surface is, and that is exactly what case-by-case porting leaves unmeasured.
    # interFoam: NOTHING IS PORTED YET. Every entry below is the SCOPE, read from OpenFOAM v2412's own
    # source, not a claim about brae. It exists so the work can be argued about before any code is
    # written -- which is the whole point of emitting the checklist first (see the header).
    #
    # What makes interFoam different from the three solvers already ported: it is a PIMPLE solver, and
    # brae_pimpleFoam already runs that loop. The new work is not the time stepping -- it is the VoF
    # half: a bounded explicit alpha solve (MULES) that is not a matrix assembly at all, interface
    # curvature, and a pressure equation in p_rgh rather than p.
    "interFoam": [
        dict(name="interFoam_main", of_symbol="main",
             of_file="applications/solvers/multiphase/interFoam/interFoam.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_solve_cpp.cuh",
             brae_target="src/applications/solvers/interFoam/braeInterFoam.cu",
             validation="A WHOLE TIME STEP'S ALPHA HALF NOW RUNS ON THE DEVICE "
                        "(device_inter_alpha_step.cu: the sub-cycle, the correctors, the MULESCorr "
                        "pre-solve, and mixture.correct() at all three placements OpenFOAM runs it). "
                        "Against the host solver over six steps of three sub-cycles and two correctors: "
                        "the explicit path 8.9e-16 in alpha and 9.1e-13 in rho, the MULESCorr path "
                        "6.9e-11 and 6.8e-08 -- larger because two different linear solvers run the "
                        "pre-solve -- and the two paths differ from each other by 1.4e-02, so neither "
                        "arm is a copy of the other. Leaving out the mixture.correct() AFTER the "
                        "sub-cycle (alphaEqnSubCycle.H:36-38) leaves rho 1.1e+02 of 1000 wrong while "
                        "alpha is 0.000e+00 unchanged: UEqn would be built on the wrong density and no "
                        "interface gate could see it. "
                        "ON damBreak'S OWN MESH AND CASE, the device alpha half tracks the host to "
                        "4.2e-11 in alpha and 1.2e-10 relative in rhoPhi over five steps from a "
                        "developed state (tests/interfoam_dambreak_device_vs_host.sh). That case brings "
                        "what no box fixture here did: 4536 of its 4744 boundary faces are EMPTY, its "
                        "atmosphere is inletOutlet, and its fvSolution asks for MULESCorr with "
                        "nLimiterIter 5 and nAlphaCorr 2. It also has to be WARMED UP -- damBreak starts "
                        "from rest, so phi is zero and the alpha equation advects nothing until the "
                        "pressure corrector has made a flux; run from 0/ the gate reported three green "
                        "numbers on two fields that had not moved. THE WHOLE STEP ON damBreak PRODUCED "
                        "NaN IN ALL 2268 CELLS, and the cause was fvc::reconstruct on a 2-D mesh: "
                        "surfaceSum skips empty patches, so sum(SfHat (x) Sf) has NO zz contribution "
                        "and a plain cofactor inverse divides by zero. OpenFOAM's inv(Field<tensor>) "
                        "is Tensor::safeInv (tensorField.C:55, TensorI.H:608-661), NOT Tensor::inv: it "
                        "adds 1 to any diagonal below SMALL*(magSqr of the three), inverts, subtracts 1 "
                        "back, and returns the ZERO tensor below ROOTVSMALL. Ported to both sides. "
                        "brae's own correctVelocity was ALSO not skipping empty patches, which is what "
                        "had been hiding the singularity on the host. End to end after the fix: alpha "
                        "3.4346e-09 unchanged, U 3.308e-06 unchanged, p_rgh 2.29e-06 -> 2.364e-06 (both "
                        "far inside the two linear solves' residuals), capillaryRise 0.7% unchanged. "
                        "TWO TRAPS WORTH CARRYING. std::fmax(a, NaN) returns a -- it IGNORES the NaN -- "
                        "so a worst-difference loop built on fmax, which is how nearly every arm in "
                        "this tree accumulates, reports 0.000e+00 for a field that has gone entirely "
                        "non-finite; a finiteness check has to precede any fmax accumulator. And "
                        "BRAE_INTER_STEP_CHECK on deviceInterStep names the first stage whose output is "
                        "not finite, which is what turned this from 'damBreak gives NaN' into a line. "
                        "WITH THAT FIXED the whole step runs on damBreak: finite, bounded to 2.0e-11, "
                        "advancing, and alpha tracks the host to 1.85e-03 of a field whose range is 1. "
                        "U is 1.60e-01 of 2.65e-01 -- SIXTY PER CENT, a defect and not a tolerance -- "
                        "so the whole-step arm stays behind BRAE_INTER_WHOLE_STEP rather than red in "
                        "the suite. damBreak sets `momentumPredictor no`, so U comes only from "
                        "HbyA + rAU*reconstruct((phig - flux)/rAUf): the error is in HbyA or in the "
                        "momentum matrix under this case's real conditions. Two more things that arm "
                        "paid for: the staged case must be forced to `adjustTimeStep no`, because the "
                        "host reads damBreak's own controlDict and grows dt while the device loop takes "
                        "the fixed one -- the two sat at different physical times and alpha read "
                        "9.57e-01 out; and deviceInterStep was not computing fvc::ddtCorr at all, which "
                        "cost alpha 2.28e-03 -> 1.85e-03 and p_rgh 1.22e+02 -> 7.57e+01; and the step "
                        "ran ONE pressure corrector where damBreak's PIMPLE block asks for THREE -- "
                        "interFoam.C wraps the WHOLE of pEqn.H in while(pimple.correct()), so rAU, "
                        "HbyA, phiHbyA and phig are all rebuilt each pass. That last one is a real "
                        "omission and NOT the cause of the 60%: measured, three correctors instead of "
                        "one moved U from 1.6039e-01 to 1.5848e-01, about one per cent of the gap. "
                        "The step also hardcoded solutionD as all-valid where damBreak is 2-D and "
                        "polyMesh::solutionD() gives (1 1 -1), which fvMatrix::H()'s validComponents "
                        "block honours -- fixed, and also NOT the cause. "
                        "THE TAPS (DeviceInterStepTaps) LOCALISED IT, against the host's own "
                        "assembleUEqn/matrixA/matrixH from the SAME post-alpha state: UEqn.diag "
                        "3.6e-15 of 2.4e+01 and rAU 1.4e-20 of 3.8e-05, both exact, so the implicit "
                        "half -- div coefficients, laplacian, ddt -- is right; UEqn.source.x 1.6e-03 of "
                        "2.1e-01, 0.76%, so the fault is an EXPLICIT term; and HbyA.x 5.0e-03 of "
                        "2.0e-01 follows from it. Two candidates, both visible: gpu::assembleUEqn "
                        "passes UbStored = nullptr so the device RE-DERIVES U's boundary instead of "
                        "taking the stored value, and damBreak's atmosphere is "
                        "pressureInletOutletVelocity where those differ; and deviceInterStep builds the "
                        "boundary mixture from the FACE CELL's rho where InterFields builds it from "
                        "alpha's PATCH values -- the distinction worth 12.8% on capillaryRise. BOTH "
                        "ARE NOW FIXED (MomentumInput gained UbStored, and the boundary mixture is "
                        "built from alpha1Bnd through the same deviceMixtureCorrect the cells use) and "
                        "NEITHER moved the gap: 1.5628e-03 to five digits before and after. The viscous "
                        "term looked ruled out too -- assembling both sides with nuEff ZEROED left the "
                        "source gap IDENTICAL -- BUT THAT READING WAS WRONG, and only because the "
                        "linearUpwind error below was two orders larger and masked it: with that fixed, "
                        "the SAME bisect collapses the gap to 1.99e-11. A bisect only speaks about the "
                        "DOMINANT term. At the time it correctly said the source was dominated by "
                        "the ddt (rDeltaT*rhoOld*V*UOld, about 0.2 on this mesh). UOld is the same "
                        "array on both sides, V is the mesh's, and "
                        "rhoOld is mixtureRho of the same alpha1Old; the diagonal matches exactly so "
                        "relaxedDiag does too; the relax source was bisected out as well (relax OFF on "
                        "both sides leaves the gap IDENTICAL), and rho.oldTime() matched EXACTLY. "
                        "FOUND: deviceInterStep HARDCODED div(rhoPhi,U) as upwind, and damBreak's "
                        "fvSchemes says `Gauss linearUpwind grad(U)` -- the ONE scheme whose MATRIX is "
                        "pure upwind while the whole of it lives in a DEFERRED SOURCE CORRECTION, so it "
                        "gives an exact diagonal with a wrong source, which is exactly the signature "
                        "that took six candidates to reach. 24 of the 44 shipped tutorials name it. "
                        "Taking the scheme from the case: source 1.5628e-03 -> 2.2390e-06 of 2.0650e-01 "
                        "(700x) and HbyA.x 5.0146e-03 -> 8.2621e-05 of 1.9643e-01 (60x). The momentum "
                        "equation is now right to 1.1e-05 relative. WHAT REMAINS IS THE PRESSURE HALF, "
                        "and it is wrong from the FIRST step, not accumulating: after one step alpha is "
                        "1.39e-11 and phiHbyA 1.66e-07 of 6.63e-04 (2.5e-04 relative), but the SOLVED "
                        "p_rgh is 6.35e+01 of 2.85e+03 (2.2%) and U 5.86e-02 of 2.55e-01 (23%). That is "
                        "consistent rather than contradictory: div(phiHbyA) is a near-cancellation of "
                        "four face fluxes, so 2.5e-04 in the flux is tens of per cent in the divergence "
                        "the pressure equation is built on -- which means the momentum source's "
                        "remaining 1.1e-05 is not yet small enough, and HbyA amplifies it 40x. damBreak "
                        "names `gradSchemes default Gauss linear`, so linearUpwind's correction takes an "
                        "UNLIMITED gradient and gradULimitK 0 is right. RE-RUNNING THE BISECTS AFTER "
                        "THE FIX NAMES THE REMAINDER: zeroing nuEff now drops the source gap from "
                        "2.2390e-06 to 1.9933e-11, while upwind-on-both and relax-off leave it "
                        "unchanged -- so the whole of the residual 1.1e-05 is the EXPLICIT "
                        "dev2(T(grad U)) term of divDevRhoReff. OF forms (rho*nuEff)*dev2(T(grad U)) "
                        "per CELL and lets fvc::div interpolate the product; interpolating mu and the "
                        "tensor separately is a different field -- but the device already forms the "
                        "product per cell (sigmaKernel) and interpolates THAT, so it is not the fault. "
                        "SPLITTING THE dev2 CONTRIBUTION INTERIOR FROM PATCH-ADJACENT NAMES IT: exact "
                        "in all 2066 interior cells (1.39e-17 of 1.03e-03) and wrong only in the 202 "
                        "cells touching a patch (2.24e-06 of 9.06e-05, 2.5% of the term there). Exact "
                        "inside and wrong at the boundary is a boundary face value, not a scheme -- the "
                        "same split that named the stale inletOutlet on rhoSimpleFoam. It is in "
                        "divDevRhoReff's BOUNDARY sigma. SPLIT BY PATCH it was the ATMOSPHERE -- "
                        "pressureInletOutletVelocity, 46 faces -- at 2.24e-06 of 2.24e-06, ONE HUNDRED "
                        "PER CENT of the term there, against 0.13%, 0.8% and exact on the three noSlip "
                        "walls. CAUSE: this port filled UbStored with deviceBCValue, which RE-DERIVES "
                        "the patch value on the device -- precisely what UbStored exists to avoid. "
                        "fvc::grad(U) inside divDevRhoReff reads the STORED value, and at a "
                        "flux-conditional patch the two are different fields. Taking it from the host's "
                        "evaluate instead: every patch drops to round-off (atmosphere 5.08e-21) and the "
                        "momentum source goes 2.2390e-06 -> 1.9933e-11, five orders. "
                        "WHAT IS STILL OPEN, and it is now ONE named quantity: HbyA is 4.95e-05 of "
                        "1.96e-01 (2.5e-04) with the source and diagonal both exact, and comparing the "
                        "rest of H()'s inputs says upper, lower and iC are BIT-IDENTICAL while bC.x is "
                        "3.3406e-06 of 3.3406e-06 -- 100% wrong. The two paths agree exactly on the "
                        "implicit half of every boundary condition and disagree entirely on the explicit "
                        "one, on that same atmosphere patch. THE DEVICE SIDE IS AT bC = 0 AND THE HOST "
                        "AT 3.3406e-06, and the device's reason is one line in "
                        "buildDeviceVectorBoundary: bcCategory 6 (pressureInletOutletVelocity) is "
                        "mapped to device bcType 0, PLAIN zeroGradient, whose valueInternalCoeffs is 1 "
                        "and valueBoundaryCoeffs is 0 -- so the device cannot produce a non-zero bC "
                        "there at all, while brae's HOST class overrides both as transformFvPatchField "
                        "does (iC = 1 - d_k, bC = value_k - (1 - d_k)*pif_k). WHICH SIDE IS RIGHT IS "
                        "NOT SETTLED: d_k = sqrt(|valueFraction_kk|) is ZERO on an outflow face, which "
                        "is consistent with iC agreeing bit-for-bit, and then bC = value - pif turns on "
                        "whether the stored patch value equals the extrapolation at that instant. "
                        "Deciding that by reasoning between brae's two implementations is what the "
                        "of-instrument skill exists to prevent, so tools/dumpInterFoam's UEqn.H now "
                        "dumps OpenFOAM's own assembled UEqn boundaryCoeffs per patch, after relax(). "
                        "OPENFOAM SAYS |bC| = 0 ON THE ATMOSPHERE -- and on all three noSlip walls. "
                        "THE DEVICE IS RIGHT AND brae's HOST IS WRONG: device 0, OpenFOAM 0, host "
                        "3.3406e-06. So the fix is to the HOST's PressureInletOutletVelocityPatchField, "
                        "not to buildDeviceVectorBoundary's category-6 mapping, which had looked like "
                        "the defect for a whole turn. (One caveat recorded in the tool: the assembled "
                        "matrix's internalCoeffs/boundaryCoeffs are the SUM of the div and laplacian "
                        "contributions, each already scaled by its own face coefficient -- they are not "
                        "the per-BC valueInternalCoeffs/valueBoundaryCoeffs that transformFvPatchField's "
                        "identity relates, so that identity must not be checked against them.) "
                        "THE FIX IS updateFromPatchVelocity, which rhoSimpleFoam has called since its "
                        "own pcEqn gate and interFoam never did: OpenFOAM's directionMixed evaluate() "
                        "leaves the patch value at patchInternalField on an OUTFLOW face, and its "
                        "valueBoundaryCoeffs = value - (1 - d)*pif is then identically zero BECAUSE "
                        "value == pif there; leaving the written seed makes it value - pif. Added to "
                        "pressureCorrector and to both sides of the device gate. WITH IT THE WHOLE "
                        "MOMENTUM-TO-PRESSURE HAND-OFF IS EXACT: source 2.8e-17, diag 3.6e-15, rAU "
                        "1.4e-20, upper/lower/iC/bC all 0.000e+00, HbyA 2.1e-17 and phiHbyA 2.2e-19 -- "
                        "from 4.9e-05 and 1.7e-07. damBreak end to end is UNCHANGED (alpha 3.4346e-09, "
                        "p_rgh 2.364e-06, U 3.308e-06), so the solver path was already close enough "
                        "there for it not to show. AND THE ONE-STEP U AND p_rgh DID NOT MOVE EITHER "
                        "(5.86e-02 of 2.55e-01, 6.35e+01 of 2.85e+03): with the pressure equation's "
                        "entire input now exact to 2e-19, what is left is WHOLLY INSIDE THE PRESSURE "
                        "SOLVE -- OR IN THE COMPARISON ITSELF, and it turned out to be the latter. "
                        "Taking the pressure half apart the same way: the p_rgh error is NOT a constant "
                        "offset (mean 1.09, spread 62.4, so not the reference) and NOT localised to a "
                        "patch (interior 6.0e+01, lowerWall 6.4e+01, atmosphere 0.03); phiHbyA's "
                        "BOUNDARY is exact too (2.1e-22, and fvc::div sums it, so the source is exact); "
                        "and the p_rgh MATRIX is exact -- diag 4.2e-22, upper 0, source 3.3e-19, iC 0, "
                        "bC 0. THEN SOLVING THE IDENTICAL SYSTEM ON THE HOST LANDS ON THE DEVICE'S "
                        "ANSWER TO 4.3e-07 of 2.9e+03 (1.5e-10 relative). So the device's whole pressure "
                        "pass is correct and the 6.4e+01 gap is against runInterFoam, which is solving a "
                        "DIFFERENT system: the test's reconstruction of the host's sequence is what is "
                        "off, not the device. Two fixture faults already found on the way there -- the "
                        "case must be forced to `adjustTimeStep no`, and brae's host driver HARDCODES "
                        "the p_rgh solve to 1e-9 (BRAE_PTOL) instead of reading the case's "
                        "solvers/p_rgh entry, which damBreak sets to tolerance 1e-07 relTol 0.05. "
                        "THE CAUSE WAS IN THE GATE: its hooks called mixtureNu(alpha1, alpha2, ...) "
                        "where the second argument is mu -- nu = mu/(clamped blend), not a complement "
                        "(two_phase_mixture_cpp.cuh:130). Both sides of the in-test comparison used the "
                        "same nonsense, so every coefficient agreed to round-off while runInterFoam, "
                        "which calls it correctly, did not; that alone was 58% of U and it was in the "
                        "measurement, not in the code under test. WITH IT FIXED THE WHOLE DEVICE TIME "
                        "STEP TRACKS THE HOST ON damBreak'S REAL CASE: over five steps alpha 7.3e-11, "
                        "U 6.8e-10 of 2.7e-01 (2.6e-09 relative) and p_rgh 2.0e-07 of 2.8e+03 "
                        "(6.9e-11), from U 1.53e-01 (58%). The arm is now a GATE rather than a "
                        "diagnostic, at bounds two orders above what is measured. "
                        "THE HOST NOW READS solvers/p_rgh instead of hardcoding psc.tolP = 1e-9: "
                        "OpenFOAM reads it, and a case asking for a LOOSER solve (damBreak says "
                        "tolerance 1e-07 relTol 0.05) was being solved far tighter than OpenFOAM solves "
                        "it. relTol comes from the Final entry, because PIMPLE selects p_rghFinal on "
                        "the last corrector and the tutorials set relTol 0 there so the step ends on a "
                        "converged pressure. THE END-TO-END NUMBERS MOVED AND ARE RECORDED AS MEASURED: "
                        "damBreak alpha 3.4346e-09 -> 1.2402e-08, p_rgh 2.364e-06 -> 3.351e-06, U "
                        "3.308e-06 -> 9.756e-06, water volume still identical to ten digits, "
                        "capillaryRise still 0.7%. All remain two orders inside the gate's 1e-4 bound, "
                        "which that gate already attributes to the two linear solves' own residuals -- "
                        "solving TIGHTER than OpenFOAM happened to land closer to it, and that is not a "
                        "reason to keep doing it, because OpenFOAM's answer IS the one solved at the "
                        "case's tolerance. BRAE_PTOL still overrides, because a device-vs-host gate has "
                        "to pin both sides to one stopping point. "
                        "THE fmax TRAP IS NOW GUARDED EVERYWHERE: tests/device_gate_finite.cuh's "
                        "nonFinite() sits ahead of 124 accumulators across all 16 device gates and "
                        "adds straight into each gate's failure count, so a field that has gone "
                        "non-finite can no longer read as 0.000e+00. It carries its own fail-proof in "
                        "test_device_alpha_flux.cu: handed {1, NaN, 3} it reports the NaN, while fmax "
                        "over the same differences returns 2.0 with the NaN silently dropped. "
                        "brae_interFoam NOW TAKES -device, running runInterFoamDevice: the SAME case "
                        "translation and the same components the gates call, with the hooks living in "
                        "the driver rather than in a test -- a device path whose boundary evaluation is "
                        "written inside a gate is the defect braeInterFoam.cu's own header names. On "
                        "damBreak at 5 fixed steps the two paths print the same summary line: max|U| "
                        "0.2702 m/s both, alpha in [0, 1] against [-5.1e-34, 1], worst |div(phi)| "
                        "8.5e-05 host against 5.0e-05 device. It USED TO REFUSE adjustTimeStep by "
                        "name; the device loop now computes both Courant numbers off the flux the last "
                        "step left behind and runs them through the same setDeltaTVoF the host calls, "
                        "and the gate has an arm for it -- twelve steps on damBreak's own clock, Co "
                        "1.5130e-03 agreeing to 6.2e-12, alphaCo 1.4556e-03 to 2.3e-10, the same dt to "
                        "3.7e-10 and the same physical time to 2.2e-10. THE FIRST DRAFT OF THAT ARM WAS "
                        "VACUOUS: setDeltaT takes min(maxCo/Co, 1 + 0.1*maxCo/Co, 1.2), so below "
                        "maxCo/Co = 2 the answer is the constant 1.2 and dt rides the cap whatever Co "
                        "says -- damBreak at its own maxCo 1 does that for its first TWENTY-THREE "
                        "steps, and both runs grew by 1.2^5 and agreed to 0.000e+00 with the ported "
                        "Courant number never consulted. The arm now runs at maxCo 0.0015 and refuses "
                        "the cap trajectory by name. Twelve steps rather than five for a second reason: "
                        "damBreak's setFields leaves a SHARP interface, so nearInterface()'s [0.01, "
                        "0.99] band is EMPTY and alphaCoNum is identically zero until step 11. (The "
                        "report's maxU and worstDivPhi had to be filled "
                        "too: left at their defaults the device path printed `max|U| 0 m/s, worst "
                        "|div(phi)| 0.000e+00` on a run whose U reaches 0.27.) THE GATE NOW CALLS "
                        "BOTH SHIPPED DRIVERS rather than a hand-rolled copy of the hooks, so what is "
                        "measured is what runs: from damBreak's 0/ over five fixed steps, alpha "
                        "1.6e-10, U 3.7e-08 relative, p_rgh 2.1e-10, excursion 8.9e-16, and the "
                        "interface moved 5.6e-03 -- which a start from rest only does through the "
                        "pressure corrector, so that number is what says the whole loop ran and not "
                        "just its alpha half. The 900 lines of taps and bisects that found the six "
                        "defects are gone with the hunt; DeviceInterStepTaps stays in the production "
                        "header for the next one. "
                        "AND A WHOLE TIME STEP NOW RUNS ON THE DEVICE (device_inter_step.cu): the alpha "
                        "sub-cycle, mixture.correct(), the momentum matrix and the pressure corrector, "
                        "with only the boundary conditions on the host. Five steps at Co ~ 0.16 leave "
                        "alpha in [0,1] EXACTLY and p == p_rgh + rho*gh exactly. Two arms pin the LOOP "
                        "ORDER, which is the thing nothing underneath can check: the mixture UEqn is "
                        "built on differs from the one the step STARTED with by 1.3e+02 of 1000 in 43 of "
                        "256 cells -- the interface band, and the single-phase order would use the "
                        "second; and rhoPhi is the MULES-limited flux the alpha equation left, 9.1e-01 "
                        "of 2.4e+00 away from interpolate(rho)*phi rebuilt afterwards. "
                        "SHIPPED and gated end to end. brae_interFoam runs a case from the command line and "
                        "`brae -case` routes to it (tests/solver_dispatch.sh). Against real OpenFOAM: "
                        "damBreak alpha 3.4e-09, p_rgh 2.3e-06 and U 3.3e-06 relative "
                        "(tests/interfoam_dambreak_vs_openfoam.sh); capillaryRise -- where surface "
                        "tension IS the answer and the only tutorial with a contact angle -- 0.7% "
                        "(tests/interfoam_capillaryrise_vs_openfoam.sh); the curvature itself to "
                        "2.5e-14 at four calculateK pass counts "
                        "(tests/interfoam_curvature_vs_openfoam.sh). THE SUB-CYCLE IS ON THE DEVICE "
                        "(device_alpha_subcycle.cu) and matches the host's loop to 1.1e-16 in alpha and "
                        "2.2e-16 in the weighted rhoPhi over three sub-steps. Its gate is four "
                        "deliberately-wrong sub-cycles run beside the right one, because each failure "
                        "leaves a bounded plausible field: the full deltaT per sub-step 3.2e-01, "
                        "restarting each from the old alpha 9.8e-02, the last rhoPhi instead of the "
                        "weighted sum 4.1e-02 of 8.4e-01 -- and that last one is 0.000e+00 in ALPHA, so "
                        "no boundedness or interface gate could ever see it.",
             note="The time loop, and the sub-cycle inside it. THE SUB-CYCLE is used by 33 of the 44 "
                  "shipped tutorials (23 at nAlphaSubCycles 3), and all three ways to get it wrong "
                  "leave a BOUNDED, plausible alpha field that no boundedness gate sees: running the "
                  "sub-steps at the full deltaT (Time::subCycle divides it, Time.C:1006-1009), "
                  "restarting each from the time step's old alpha instead of the previous sub-step's "
                  "result, and taking the last rhoPhi instead of the time-weighted sum "
                  "sum_k (dt_k/dt_total)*rhoPhi_k. alpha.oldTime() is also RESTORED afterwards "
                  "(subCycleField's destructor), or the PIMPLE outer loop's second pass advances from "
                  "the first's result. THE LOOP ORDER is not numerics but which field each equation "
                  "sees: alpha advances BEFORE the momentum predictor and mixture.correct runs between "
                  "them, so UEqn is built on the new rho. The single-phase order -- UEqn first -- "
                  "converges fine, on last step's density, which at a water/air interface is wrong by "
                  "1000. frozenFlow's `continue` skips the turbulence corrector as well as UEqn and "
                  "pEqn. adjustTimeStep/maxAlphaCo is interFoam_alphaCourantNo and is landed. "
                  "WHAT interFoam.C HONOURS AND brae DID NOT EVEN READ. The next port on the list was "
                  "`vanLeerV` (8 tutorials), and a census was taken first to rank it: every one of the 44 "
                  "shipped tutorials was meshed and handed to brae_interFoam for one step, and brae's OWN "
                  "message recorded. That changed the ranking -- all eight vanLeerV cases also carry a "
                  "moving mesh, so the scheme alone unlocks none; the single-blocker counts are waves 9, "
                  "turbulence 7, moving mesh 2, MRF 1, limitedLinear 1 -- and it found something worse than "
                  "a missing scheme. TWO TUTORIALS REACHED `End:` THAT SHOULD NOT HAVE: "
                  "laminar/damBreakWithObstacle and laminar/oscillatingBox both ask for "
                  "`dynamicRefineFvMesh`, adaptive refinement driven by alpha, and brae ran them on the "
                  "mesh as written without a word. brae's interFoam never opened constant/dynamicMeshDict. "
                  "Seventeen more tutorials carry a moving mesh and were stopped only because they hit some "
                  "OTHER refusal first. A one-input-at-a-time sweep on damBreak then found the rest: "
                  "MRFProperties and fvOptions (both locations) ran silently although braeInterFoam.cu's "
                  "own header listed both as refused; a DICTIONARY-form sigma -- a surfaceTensionModel -- "
                  "and a MISSING sigma both became ZERO surface tension, because it was read with "
                  "scalarOr(\"sigma\", 0); a setTimeStep function object, which overrides deltaT from inside "
                  "Time::adjustDeltaT, was ignored; and the device loop ran nOuterCorrectors as 1 and "
                  "nNonOrthogonalCorrectors as 0 whatever the case said (5 and 4 tutorials ask for more), "
                  "where the host honours both. All are refused by name now, each by OpenFOAM's OWN rule "
                  "read from the source: dynamicFvMesh is mandatory once the dictionary exists and only "
                  "staticFvMesh is static (dynamicFvMeshNew.C:65-128); an MRF zone and an fvOption are "
                  "active unless they say otherwise (MRFZone.C:553, fvOption.C:72); constant/fvOptions is "
                  "looked up BEFORE system/fvOptions and OpenFOAM stops at the first (fvOptions.C:46-84); a "
                  "dictionary named sigma selects a model and a scalar one has no default "
                  "(surfaceTensionModelNew.C:36-66). tests/interfoam_refusals.sh holds 25 arms, and EVERY "
                  "REFUSAL HAS ITS OPPOSITE beside it -- staticFvMesh, `active no` on a zone and on an "
                  "option, an inactive constant/fvOptions hiding an active system/ one, sigma 0, a harmless "
                  "function object -- which must still RUN, because a refusal that fires on a case OpenFOAM "
                  "would run unchanged is a defect too. After it the census shows exactly two tutorials "
                  "reaching `End:`: damBreak and capillaryRise, the two that are gated. That note ended by "
                  "saying what it did NOT claim -- that nOuterCorrectors 2, nNonOrthogonalCorrectors 1 "
                  "and momentumPredictor yes were RIGHT on the host, only that they ran. They were "
                  "checked next, and two of the three were not. "
                  "THREE PIMPLE CONTROLS RAN AND WERE NEVER HELD TO ANYTHING, and two of them were wrong. "
                  "damBreak uses none of nOuterCorrectors 2, nNonOrthogonalCorrectors 1 or "
                  "momentumPredictor yes (5, 4 and 5 of the 44 shipped tutorials set them), brae's host "
                  "executed all three, and no gate compared any of them with OpenFOAM -- the position "
                  "alphaApplyPrevCorr was in while it carried a wrong limiter argument. Each is now a "
                  "profile of tests/interfoam_dambreak_vs_openfoam.sh at the big step, with the control on "
                  "the ORACLE: the setting moves OpenFOAM's own alpha by 1.6e-01, 5.4e-04 and 7.5e-02 "
                  "respectively, so a brae that ignored it could not pass. (1) nOuterCorrectors 2 WAS "
                  "WRONG: alpha 4.9e-06, p_rgh 4.5e-06, U 1.0e-03, one alpha solve taking 3 sweeps for "
                  "OpenFOAM's 2. The solver log localised it in one read: every FIRST-pass alpha solve "
                  "exact to 1e-13, every SECOND-pass one 84% to 170% out in its initial residual, and p_rgh "
                  "exact through step two's first pass. alphaEqnStep reset alpha1 to its old time at the "
                  "top of every call, and alphaEqn.H has NO such assignment anywhere: the old time enters "
                  "through fvmDdt's source and MULES' psi.oldTime(), while the CURRENT alpha1 is the pre- "
                  "solve's initial guess, the field alphaPhiUn is built from, and what the limiter takes "
                  "its extrema from (MULESTemplates.C:208 against :244). With one outer corrector the two "
                  "are the same field, so the reset was a no-op for as long as the function had only ever "
                  "been run that way. After: alpha 2.8e-12, U 8.6e-12, 10 of 10 alpha counts. brae's MULES "
                  "already took psi and psiOld separately, as OpenFOAM's does; only the caller forced them "
                  "equal. (2) nNonOrthogonalCorrectors 1 WAS RIGHT: alpha 1.2e-12, 30 of 30 p_rgh counts. "
                  "(3) momentumPredictor yes FOUND THREE THINGS. OpenFOAM itself REFUSES damBreak with the "
                  "predictor on -- `Entry 'UFinal' not found` -- because fvMatrix::solve() selects the "
                  "Final entry on the last outer corrector and with one corrector that is the only one; "
                  "brae ran the case, reading NEITHER entry: a struct default of 1e-7 on PBiCGStab on the "
                  "host and a hardcoded 1e-12 on Jacobi-BiCGStab on the device, where the case says "
                  "smoothSolver symGaussSeidel at 1e-06. That is the third time the same substitution has "
                  "turned up (p_rgh, alpha, U). Both paths now read U and UFinal, require each exactly "
                  "where OpenFOAM does, run the case's smoother (solveVector took a VectorLinearSolver; the "
                  "device calls deviceSymGaussSeidel), skip the component a 2-D mesh does not solve as "
                  "fvMatrixSolve.C:162-164 does, and log every solve. THEN THE HOST WAS STILL THREE ORDERS "
                  "WORSE THAN THE DEVICE on the same case with the same solver and the same sweep counts -- "
                  "alpha 9.1e-10 against 1.4e-12, U 9.2e-08 against 8.7e-12 -- which is a comparison that "
                  "only exists because both paths are gated against the same oracle. fvMatrix::solve() ends "
                  "with correctBoundaryConditions(); the host's predictor ended with evaluateBoundary() "
                  "alone, and brae's pressureInletOutletVelocity keeps its STORED value through that on "
                  "purpose, so the atmosphere's U_b was one solve stale going into the first pressure "
                  "corrector, where totalPressure reads 0.5*rho*|U_b|^2. The device refreshed it and the "
                  "host did not. One shared updateVelocityPatchesFromCells now serves the corrector, the "
                  "predictor and the device driver: host alpha 1.4e-12, p_rgh 1.4e-12, U 8.4e-12, every Ux, "
                  "Uy, p_rgh and alpha count OpenFOAM's on both paths. The device REFUSES the first two "
                  "controls (its loop runs one outer corrector and no non-orthogonal pass), and on those "
                  "profiles the gate asserts the refusal. ALONG THE WAY the residual comparison got the "
                  "floor a normalised residual has: it had failed on two agreements it should have praised, "
                  "a 5e-8 initial residual the two codes held 6e-18 apart ('1.3e-10 relative') and a "
                  "1.935e-14 final residual matching to four digits ('2.2e-05 relative'); below 1e-16 "
                  "absolute the difference is now zero, and a different SOLVER is still nine orders above "
                  "that."),
        dict(name="interFoam_alphaEqn", of_symbol="alphaEqn",
             of_file="applications/solvers/multiphase/VoF/alphaEqn.H",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/alpha_eqn_cpp.cuh",
             brae_target="src/applications/solvers/interFoam/device_alpha_step.cu",
             validation="tests/test_alpha_eqn_cpp.cu covers the FLUX ASSEMBLY. Boundedness is the gate for "
                        "the MULES half and is an ASSERTION, not a tolerance: 0 <= alpha <= 1 exactly, "
                        "every cell, every sub-cycle. A VoF gate that only checks agreement can pass "
                        "while the field goes unbounded and is then clipped. ON THE DEVICE, "
                        "tests/test_device_alpha_step.cu runs the WHOLE explicit corrector on a rotating "
                        "blob -- discretely divergence-free, so boundedness means something -- against "
                        "the host: alpha in [0,1] exactly, 1.1e-16 after one step and 9.99e-16 after "
                        "forty of two correctors each.",
             note="minIter ON THE PRE-SOLVE is honoured on the host (AlphaStepInput::minIterAlpha): gated as laminar/damBreak `alphaminiter` on the sweep count, which is all damBreak can witness (the forced sweep acts on an exact solution); ignored it logs 0 where OpenFOAM logs 1. The device refuses it. "
                  "ALPHA'S BOUNDARY IS NOT EVALUATED BEFORE ITS FLUXES (alphaEqn.H has no such call): the explicit path reads the patch values the last MULES solve left, at the first step the file's own `value`; brae evaluated at the top of alphaEqnStep. MEASURED on RAS/mixerVesselAMI, whose outlet writes `value uniform 0` under water: OpenFOAM's 62 outlet cells rise to 1.104 in the first sub-cycle, brae held them at 1, alpha 9.4e-02 after one step (tests/interfoam_ami_vs_openfoam.sh; restored, 1.9e-06 of U after 100 steps). "
                  "SPLIT IN TWO. The flux assembly is landed: alphaControls, the off-centring, phic, "
                  "phiCN, alphaPhiUn and rhoPhi. MULES is interFoam_MULES and is not. Su/Sp/divU are "
                  "identically zero for THIS solver (interFoam/alphaSuSp.H is three zeroFields); they "
                  "are live in interPhaseChangeFoam, which has its own. Three things the gate pins that "
                  "no field shows: phic is zeroed on every NON-COUPLED boundary (alphaEqn.H:79-89, "
                  "compression at an inlet sharpens an interface the boundary does not have); "
                  "alphaPhiUn's compressive term is a nested flux with TWO minus signs, which cancel "
                  "exactly under `Gauss linear` (31 tutorials, so the mistake is invisible there) and "
                  "do not under `Gauss vanLeer` (7); and rhoPhi's two branches multiply rho2f by phiCN "
                  "on the Euler path and phi on the other. Only Euler, localEuler and CrankNicolson ddt "
                  "are accepted (alphaEqn.H:49-51), and CrankNicolson is refused when sub-cycling. "
                  "`Gauss interfaceCompression` on div(phirb,alpha) (4 tutorials) is interFoam_interfaceCompression. "
                  "THE WHOLE MULESCorr PATH NOW RUNS ON THE DEVICE and is gated end to end against the "
                  "host over 40 steps at nAlphaCorr 2 / nLimiterIter 5, damBreak's own settings: "
                  "1.07e-10, and it differs from the explicit path by 3.3e-02 so the arm is not a "
                  "second copy of it. Two things that gate found. CMULES' outlet test reads "
                  "alphaPhiUn, not phiCN -- OpenFOAM passes talphaPhi1Un() -- and passing the "
                  "volumetric flux there cost 3.99e-09 against the host. And the path is NOT exactly "
                  "bounded at nLimiterIter 5: 1.307e-06, the host's number to four digits, which is "
                  "the fixed-point iteration and nothing else -- the pre-solve alone is 0.000e+00 at "
                  "solver tolerance 1e-12 and 1.31e-06 at 1e-6, and the final field is 0.000e+00 at "
                  "nLimiterIter 40. "
                  "THE IMPLICIT PRE-SOLVE IS ON THE DEVICE (device_alpha_presolve.cu): the upwind LDU, "
                  "the Euler ddt, the BiCGStab solve and alpha1Eqn.flux(), with only the boundary "
                  "coefficients built on the host. Its gate pins the flux by the closure "
                  "alpha == alphaOld - (dt/V)*sum_f alphaPhi10, which tracks the SOLVER RESIDUAL "
                  "(1.3e-12 at tolerance 1e-12, 3.0e-06 at 1e-6) where a flux rebuilt from the "
                  "pre-solve field leaves a fixed 5.1e-03 and a central one 2.3e-02. "
                  "THE DEVICE SPLIT IS ONE CORRECTOR PER CALL, not the nAlphaCorr loop: OpenFOAM "
                  "re-evaluates alpha's boundary at the end of every MULES solve and runs "
                  "mixture.correct() after that, and the next corrector's flux, gradient and limiter all "
                  "read them. Looping inside the call put nHatf 2.885e-04 out on a field whose largest "
                  "value is 1.736e-03, and 3.2e-09 into alpha by the end of a single step. "
                  "THE PRE-SOLVE'S LINEAR SOLVER WAS A SUBSTITUTION HIDDEN BY A TOLERANCE. Every MULESCorr "
                  "tutorial names `solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8` for alpha. "
                  "The device ran Jacobi-BiCGStab to a HARDCODED 1e-12 and the host DILU-PBiCGStab to a "
                  "struct default of 1e-8; neither read the case. It matters because an implicit upwind "
                  "matrix is nearly triangular in flow order, so a Gauss-Seidel sweep is nearly an exact "
                  "solve -- OpenFOAM's log on damBreak reads `No Iterations 2, Final residual 9.3e-14` -- "
                  "while Jacobi-BiCGStab stops AT its tolerance. Measured against real OpenFOAM, five steps "
                  "of damBreak, device alpha pre-solve tolerance swept: 1e-16 -> alpha 5.5e-11; 1e-12 (the "
                  "hardcoded value) -> 1.6e-10; 1e-8 (THE CASE'S OWN) -> alpha 3.3e-06 and U 1.2e-03. The "
                  "1.6e-10 had been in the device/host gate all along, under a bound of 1e-8 and a comment "
                  "attributing it to 'the pre-solve's residual' -- which was true and was not followed. The "
                  "device had OpenFOAM's smoother already (deviceSymGaussSeidel: smoothSolver's stopping "
                  "rule around symGaussSeidelSmoother's own sweep, level-scheduled and exact, gated by "
                  "gs_ladder). With the case's entry read (InterFields::aSolve, lduMatrix's defaults, "
                  "refused when MULESCorr names no solver) and the device on it at the case's 1e-8: device "
                  "against OpenFOAM alpha 1.6e-10 -> 3.6e-14, p_rgh 2.1e-10 -> 8.6e-12, U 3.8e-08 -> "
                  "2.3e-11 -- the host's numbers to three digits -- and device against host through the "
                  "shipped drivers alpha 1.6e-10 -> 1.1e-15, U 3.7e-08 -> 2.6e-14 relative, p_rgh 2.5e-10 "
                  "-> 2.5e-14, the adaptive clock 3e-10 -> 2e-12. Every bound in "
                  "tests/test_device_inter_dambreak_alpha.cu followed, from 1e-8/1e-7/1e-6 to "
                  "1e-13/1e-12/1e-12. THE HOST SUBSTITUTED TOO (DILU-PBiCGStab, under a notice) until it had a "
                  "smoothSolver of its own -- see interFoam_smoothSolver. On damBreak that was measured "
                  "harmless, 3.6e-14, because DILU on a near-triangular matrix is as nearly exact as the "
                  "sweep; measured on one case, not proven, which is why it was closed rather than left. "
                  "Any alpha solver that is not smoothSolver with a Gauss-Seidel smoother still runs "
                  "BiCGStab on both paths, under a notice that carries the 3.3e-06. "
                  "`alphaApplyPrevCorr` WAS HALF PORTED AND THE OTHER HALF WAS SILENT. alphaEqn.H:133-150 "
                  "and :228-236: the compression flux the correctors ended on is cached in talphaPhi1Corr0 "
                  "and the NEXT pre-solve applies it, limited through MULES::correct, before its own "
                  "correctors run. Three shipped tutorials set it (floatingObject, DTCHull, DTCHullMoving) "
                  "and brae refuses all three for other reasons -- RAS, moving meshes, LTS -- so no "
                  "tutorial can gate it; OpenFOAM honours it on any MULESCorr case and damBreak is one. THE "
                  "HOST had an implementation nothing had ever run. THE DEVICE had nothing at all -- no "
                  "code and NO REFUSAL -- so `brae_interFoam -device` on such a case ignored the switch and "
                  "said nothing. Measured on damBreak at dt 5e-3 with the switch on, before any change: "
                  "host alpha 1.2e-12 from OpenFOAM, device 1.0690e-02 -- to five digits OpenFOAM's own "
                  "distance from itself with the switch OFF -- and U 15% out. Now implemented on the device "
                  "with the two CMULES primitives it already had (deviceMulesLimitCorr, "
                  "deviceMulesCorrect); the cache is owned by the CALLER because it outlives the call, and "
                  "the switch without a cache, without MULESCorr, or without the curvature-free "
                  "refreshBoundary hook is refused by name. Device after: alpha 1.2e-12, p_rgh 1.7e-12, U "
                  "9.2e-12, every p_rgh and alpha iteration count OpenFOAM's. Gated as two more profiles of "
                  "tests/interfoam_dambreak_vs_openfoam.sh. `prevcorr` asserts that OpenFOAM's log says "
                  "`Applying the previous iteration compression flux` -- an oracle that never took the path "
                  "would agree with a brae that ignored it -- and its control is that the switch moves "
                  "OpenFOAM's OWN alpha by 1.07e-02, nine orders more than brae's distance from it. "
                  "`prevcorrsub` adds nAlphaSubCycles 2, because the cache outlives the SUB-CYCLE as well "
                  "as the time step and damBreak's own count of 1 cannot tell a cache that crosses from one "
                  "that is reset: ten alpha solves (0, 0, 3, 3, 2, 2, 2, 2, 2, 2), every count equal on "
                  "both paths, alpha 1.6e-12, and the sub-cycle count moves OpenFOAM's own alpha by "
                  "8.2e-02. "
                  "ONE FIX HERE WAS MADE FROM THE SOURCE FIRST AND MEASURED AFTERWARDS, and the measuring "
                  "is the part worth keeping. The host passed phiCN, the VOLUMETRIC flux, as "
                  "MULES::correct's flux argument where OpenFOAM passes alphaPhi10 (alphaEqn.H:140). "
                  "limiterCorr reads it in one place, the boundary outlet test `(phi_b + phiCorr_b) > "
                  "SMALL*SMALL` (CMULESTemplates.C:546) -- the same slip the corrector loop had, fixed "
                  "there at 3.99e-09 -> 1.07e-10. IT TOOK THREE FIXTURES. (1) damBreak: the host's numbers "
                  "are the same to five digits either way, because the boundary half of the cached "
                  "correction is zero unless alpha's patch value moves between the pre-solve and the "
                  "correctors, which needs water LEAVING through an outflow face. (2) The water column "
                  "raised to the atmosphere, dt 5e-3: water does leave, and a probe counting the faces on "
                  "which the two tests DECIDE DIFFERENTLY found up to 13 a step, on corrections worth 15% "
                  "of the face flux -- and the answer was still bit-identical, alpha 5.8831e-13 both ways. "
                  "Running the limiter both ways on the same state showed why: the limited correction came "
                  "out identical, because on those faces the cell limiter returned 1 and made the decision "
                  "moot. A difference in a DECISION is not a difference in an ANSWER. (3) The same fixture "
                  "at dt 1e-2, found by sweeping brae alone for a step where the two limited corrections "
                  "actually differ: one application there moves one cell's alpha by 0.10. Against real "
                  "OpenFOAM, four steps: with phiCN alpha 5.5e-03, p_rgh 5.7e-03, U 2.2e-02; with "
                  "alphaPhi10 4.6e-13, 2.7e-13, 5.5e-13 -- ten orders from one argument. The device, "
                  "written with the right argument from the start, reads 4.7e-13, 2.9e-13 and 4.5e-11 on "
                  "it. Gated as the `outflow` profile, whose control is on BRAE rather than the oracle: the "
                  "host runs again with BRAE_CONTROL_PREVCORR_PHICN set (read in the driver as BRAE_PTOL "
                  "is, announced on every step it is on, and carried by "
                  "AlphaStepInput::controlPrevCorrOutletOnPhiCN, which no solver sets) and must come out "
                  "more than 1e-6 from OpenFOAM. It reads 5.5193e-03. The sweep also bounds the fixture: at "
                  "dt 2e-2 alpha leaves [-0.34, 1.12] and the inflow variants go to NaN, so dt 1e-2 for "
                  "four steps is what this case can carry."),
        dict(name="interFoam_MULES", of_symbol="MULES::limiter",
             of_file="src/finiteVolume/fvMatrices/solvers/MULES/MULES.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fvMatrices/solvers/MULES/mules_cpp.cuh",
             brae_target="src/finiteVolume/fvMatrices/solvers/MULES/device_mules.cu",
             validation="tests/test_mules_cpp.cu. The oracle is BOUNDEDNESS, not a stored field, which is "
                        "what makes the gate possible at all -- a stock run never writes the limiter. It "
                        "is a PAIR of arms, because either alone is trivially satisfiable: the limited run "
                        "stays in [0,1] exactly where the unlimited one does not, AND a monotone in-bounds "
                        "field comes back with lambda == 1 on every face away from the domain ends. "
                        "lambda == 0 everywhere passes the first and fails the second, and it is "
                        "first-order upwind -- the case's scheme silently discarded.",
             note="THE LARGEST SINGLE PIECE, and the least brae-like. An EXPLICIT, ITERATIVE, "
                  "bound-preserving flux limiter -- not a matrix assembly -- so almost none of the fvm:: "
                  "machinery the other solvers share applies. Landed: limiter, limit, explicitSolve. "
                  "Four things the gate pins that no converged field shows: psiMaxn/psiMinn are "
                  "initialised SWAPPED (:280-281) so the neighbour scan builds from the far end; the "
                  "neighbourhood extrema EXCLUDE the cell's own value, which is what lets an interface "
                  "advance; phiBD is overwritten with phiPsi on every non-coupled boundary so the "
                  "prescribed flux is never limited (:600-609); and a wedge patch gets lambda = 0 "
                  "outright (:530-533). Two things measured and NOT claimed: the running "
                  "min(lambda,...) is defensive rather than load-bearing (removing it gives a "
                  "bit-identical field, because both accumulators are built from lambda), and in one "
                  "dimension the first pass is already the fixed point -- nLimiterIter only bites where "
                  "a cell's faces are limited by different neighbours at once. AND ACROSS A COUPLED PATCH "
                  "the limiter is synced by syncTools::syncFaceList (:566), which exchanges across "
                  "processor and cyclicPolyPatch ONLY: an AMI pair keeps each side's own limiter, so the "
                  "limited flux across a cyclicACMI is not equal and opposite. brae synced it; measured "
                  "on damBreakLeakage's opening step, the receiving side's alpha flux 75% of the giving "
                  "side's in OpenFOAM (FvPatch::ami, interFoam_cyclicACMI)."),
        dict(name="interFoam_CMULES", of_symbol="MULES::correct",
             of_file="src/finiteVolume/fvMatrices/solvers/MULES/CMULESTemplates.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fvMatrices/solvers/MULES/mules_cpp.cuh",
             brae_target="src/finiteVolume/fvMatrices/solvers/MULES/device_mules.cu",
             validation="tests/test_mules_cpp.cu arm 7. Shares the boundedness oracle; the four structural "
                        "differences from the explicit path are asserted one by one, each with its own "
                        "fail-proof. ON THE DEVICE, tests/test_device_mules_corr.cu repeats A, B, C and "
                        "the empty-patch skip against the host limiter: psi stays in [0,1] exactly where "
                        "the unlimited control reaches 4.0e-01, and lambda came out bit-identical on all "
                        "264 faces, 32 of them strictly inside (0,1) -- not guaranteed, since the budgets "
                        "are gathers there and a face loop here, so the gate keeps a tolerance.",
             note="LANDED AND RUNNING on damBreak's own mesh. The semi-implicit variant, `MULESCorr yes` -- 13 of the 44 shipped tutorials, damBreak "
                  "included. With it the upwind part of the alpha equation goes through an implicit "
                  "MATRIX, so by the time CMULES runs psi is ALREADY advanced and only the antidiffusive "
                  "correction is left. Four consequences, all of them places a port that reuses the "
                  "explicit code is quietly wrong. (A) correct() uses the CURRENT rho and psi, not their "
                  "oldTime values: with a zero correction it is the identity, where the explicit form "
                  "would hand back psi.oldTime() and undo the implicit solve -- bounded, so a "
                  "boundedness gate never notices, the interface just moves at the wrong speed. (B) the "
                  "budget transform carries NO sumPhiBD, because there is no donor flux. (C) uncoupled "
                  "boundary faces ARE limited, but outlets only, and `outlet` means phi + phiCorr > "
                  "SMALL*SMALL -- the TOTAL, so an inflow face whose correction reverses it is limited "
                  "like any other. (D) nLimiterIter is get<label> with NO DEFAULT here where the "
                  "explicit limiter defaults it to 3, so the same dictionary is accepted by one reader "
                  "and refused by the other. MEASURED on damBreak: 20 alpha steps on a projected "
                  "divergence-free flux leave the field within 2.1e-11 of [0,1] against the explicit "
                  "path's 9.5e-14. Three sweeps characterise that gap: it is present at step 0 and "
                  "barely accumulates; it is INDEPENDENT of the linear-solver tolerance (1e-8 to "
                  "1e-14 moves it not at all) while the MASS DRIFT tracks that tolerance exactly; and "
                  "it scales as about dt^4. So conservation is the solve's accuracy and boundedness "
                  "is a conditioning effect in the budget that grows with the correction. The exact "
                  "mechanism is not attributed; the gate asserts the characterisation -- the bound AND "
                  "the scaling -- so a change in behaviour fails rather than being absorbed. "
                  "NOT CMULES, AS IT TURNED OUT: on damBreak over ten adaptive steps brae's alpha "
                  "reached 1 + 2.306e-06 where real OpenFOAM reached 1 + 1.1361e-06, 1.98x, and it "
                  "was first recorded here as an open CMULES gap. Measuring it put it elsewhere. "
                  "interFoam's alphaSuSp.H has no divU, so the MULESCorr pre-solve on a full cell "
                  "gives a = 1/(1 + dt*div(phi)) and the excursion is dt times the continuity error "
                  "the PRESSURE solve left. Tightening only the alpha solve changed nothing to the "
                  "last digit; tightening only p_rgh took both codes from 1.16e-06 to 4.9e-13. The "
                  "per-step ratio ran 0.12x to 3.18x, not a steady 2x. Closed under interFoam_pEqn, "
                  "and test_inter_case_cpp now holds it at OpenFOAM's own value to 2% (0.9931x). "
                  "OpenFOAM prints Max(alpha.water) TWICE a step -- after the pre-solve "
                  "(alphaEqn.H:124) and after the correctors (alphaEqn.H:263) -- and the gate's "
                  "constant had been taken from the first while it reads the second."),
        dict(name="interFoam_interfaceProperties", of_symbol="interfaceProperties",
             of_file="src/transportModels/interfaceProperties/interfaceProperties.C",
             classification="MODEL", status="REIMPLEMENT",
             brae_reference="src/transportModels/interfaceProperties/interface_properties_cpp.cuh",
             brae_target="src/transportModels/interfaceProperties/device_interface_properties.cu",
             validation="tests/test_interface_properties.cu (the dictionaries) and "
                        "tests/test_interface_curvature_cpp.cu (K itself). NO INSTRUMENTED OpenFOAM "
                        "WAS NEEDED: curvature is GEOMETRY. A plane's spurious K is 5+ orders below a "
                        "real one and tracks deltaN; a sphere of radius R gives 2/R to 0.1% at 16^3 "
                        "and the error FALLS on refinement, which is what makes the bound the "
                        "scheme's rather than the mesh's; a bubble gives the exact negative; and the "
                        "contact angle has the exact postcondition acos(nHat & nf) == theta.",
             note="K_ = -div(nHatf) with nHatf = (gradAlphaf/(|gradAlphaf| + deltaN)) & Sf "
                  "(interfaceProperties.C:141-153), plus the alphaContactAngle correction on wall "
                  "patches. surfaceTensionForce() = interpolate(sigma*K)*snGrad(alpha1). THE MINUS in "
                  "K is the sign convention for the whole solver -- drop it and every surface-tension "
                  "force points the wrong way with the right magnitude, which a flat-interface test "
                  "cannot see because -0 is 0. fvc::grad(alpha1, \"nHat\") resolves a NAMED "
                  "gradScheme. The contact-angle correction also writes alpha's own WALL GRADIENT "
                  "(acap.gradient() = (nf & nHat)*mag(gradAlphaf)), not only the normal used for "
                  "curvature; only laminar/capillaryRise sets one. nAlphaSmoothCurvature is set by NO "
                  "shipped tutorial anywhere in OpenFOAM; smoothing is fvc::average, which is "
                  "AREA-WEIGHTED and indistinguishable from a plain mean on a cube, so its gate runs "
                  "on anisotropic cells. Combining it with a leastSquares gradient is refused by "
                  "name -- there is no case to validate that against. INSTRUMENTED 2026-09-16: "
                  "tools/dumpInterfaceK reads K, nHatf, the surface-tension force and alpha's "
                  "post-correct boundary from OpenFOAM's UNMODIFIED class -- nHatf() and sigmaK() are "
                  "public, so nothing is copied or patched. It found that calculateK is a FIXED POINT: "
                  "it reads the wall gradient correctContactAngle wrote at the end of its own previous "
                  "pass, so on capillaryRise that gradient runs 7070.5 -> 8659.4 -> 9353.1 -> 9681.2 "
                  "over four passes and the curvature depends on WHERE the solver calls it, not only "
                  "on the formula. brae's formula is exact -- 2.5e-14 relative against OpenFOAM at "
                  "every one of four pass counts (tests/interfoam_curvature_vs_openfoam.sh) -- and its "
                  "call sites were wrong: at the top of each alpha corrector instead of the bottom, "
                  "absent from createFields, and absent from the mixture.correct() between the "
                  "sub-cycle and UEqn. Fixing those took capillaryRise from 24.9% to 12.8%."),
        dict(name="interFoam_twoPhaseMixture", of_symbol="twoPhaseMixture",
             of_file="src/transportModels/twoPhaseMixture/twoPhaseMixture/twoPhaseMixture.C",
             classification="MODEL", status="REIMPLEMENT",
             brae_reference="src/transportModels/twoPhaseMixture/two_phase_mixture_cpp.cuh",
             brae_target="src/transportModels/twoPhaseMixture/device_two_phase_mixture.cu",
             validation="rho and mu as alpha-weighted blends, against OpenFOAM's own fields at iteration 1.",
             note="Field plumbing rather than new numerics: rho = alpha1*rho1 + (1-alpha1)*rho2, same for mu."),
        dict(name="interFoam_immiscibleMixture", of_symbol="immiscibleIncompressibleTwoPhaseMixture",
             of_file="src/transportModels/immiscibleIncompressibleTwoPhaseMixture/immiscibleIncompressibleTwoPhaseMixture.C",
             classification="MODEL", status="REIMPLEMENT",
             brae_reference="src/transportModels/twoPhaseMixture/two_phase_mixture_cpp.cuh",
             brae_target="src/transportModels/twoPhaseMixture/device_two_phase_mixture.cu",
             validation="Shares the twoPhaseMixture gate.",
             note="Joins twoPhaseMixture and interfaceProperties into the one object interFoam.C holds -- "
                  "AND THE ORDER IT JOINS THEM IN IS PART OF THE ANSWER. correct() is calcNu() THEN "
                  "interfaceProperties::correct() (immiscibleIncompressibleTwoPhaseMixture.H:78-82), and at "
                  "a contact-angle wall the second call rewrites alpha's patch gradient and so its patch "
                  "value. Three things follow, all measured on capillaryRise against tools/dumpInterFoam "
                  "and together the whole of the 0.7% that case carried: (1) the boundary viscosity is "
                  "blended BEFORE the curvature pass; brae did it after, and UEqn.A() was exact in every "
                  "cell off the wall and 1.26% high in the air cells at it -- 0.2% of U after one step, "
                  "before the contact line had moved a face. (2) `alpha2 = 1.0 - alpha1` (alphaEqn.H:151 "
                  "and :223) is a whole-field assignment one line ABOVE mixture.correct(), so alpha2's "
                  "PATCH values are one contact-angle pass older than alpha1's, and `rho == alpha1*rho1 + "
                  "alpha2*rho2` blends the two: 7.4e-05 of rho*nu at the wall with 1 - alpha1, 1e-17 with "
                  "the stale value carried (InterFields::alpha2Bnd on the host, deviceBoundaryRho on the "
                  "device). (3) because the curvature pass has that side effect, the NUMBER of passes is "
                  "part of the answer -- the gradient runs 7070.5, 8659.4, 9353.1, 9681.2 -- and the "
                  "device's alpha-boundary hook ran the host's calculateK on every call, including the sub- "
                  "step's reset of alpha1, which is not a mixture.correct(): five passes a step where "
                  "OpenFOAM takes three, K 0.6% out at the contact line and U 1.9% out after ONE step. The "
                  "hook is now two (DeviceInterAlphaHooks::refreshBoundary). damBreak has no contact angle, "
                  "where calculateK has no side effect, and could see none of the three."),
        dict(name="interFoam_UEqn", of_symbol="UEqn",
             of_file="applications/solvers/multiphase/interFoam/UEqn.H",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_ueqn_cpp.cuh",
             brae_target="src/applications/solvers/interFoam/device_inter_ueqn.cu",
             validation="tests/test_inter_ueqn_cpp.cu, and end to end at 3.3e-06 relative in U on damBreak "
                        "against real OpenFOAM. ON THE DEVICE, tests/test_device_inter_ueqn.cu: "
                        "fvc::reconstruct pinned by its own identity -- reconstruct(V & Sf) == V to "
                        "0.000e+00 on 2:1:0.5 cells, where the div-sign convention gives 2.5e+00 -- and "
                        "bit-identical to the host on a non-uniform flux; the ddt bit-identical, with "
                        "rho.oldTime() replaced by rho measured at exactly 1000.0x, the density ratio; "
                        "and the p_rgh body force on a hydrostatic start 0.000e+00 in the bulk against "
                        "1.96e+04 at the interface, which is what the formulation is FOR. mu = rho*nuEff "
                        "is bit-identical too, and its FACE value is the product interpolated ONCE -- "
                        "interpolating the two factors separately is 3.4e-03 where the right answer is "
                        "1.0e-03. The device ASSEMBLY reuses gpu::assembleUEqn with rhoPhi and mu; that "
                        "gained an optional fvm::ddt(rho,U) applied BEFORE relax(), and a relaxEquation "
                        "flag so relax(1) still runs the diagonal-dominance clamp -- damBreak's "
                        "fvSolution says `equations { \".*\" 1; }` and skipping it there would differ "
                        "from OpenFOAM on every shipped interFoam tutorial. "
                        "THE ASSEMBLED MATRIX IS GATED WHOLE (tests/test_device_inter_ueqn_assembly.cu) "
                        "against the host's assembleUEqn on a VoF state: off-diagonals and boundary "
                        "coefficients bit-identical, the relaxed diagonal 1.1e-13 of 1.0e+06 and the "
                        "source 5.8e-11 of 4.5e+05. Two things it measures that no solved field could: "
                        "relax(1)'s diagonal-dominance clamp MOVES the diagonal by 1.2e+03, so skipping "
                        "relax at alpha == 1 is a different matrix; and passing rho for rho.oldTime() "
                        "puts the ddt source 1000.0x out in exactly 64 of 512 cells -- one layer, the "
                        "one the interface crossed. THE WHOLE PREDICTOR is gated there too: the "
                        "reconstructed face force lands in the source to 1.2e-16 relative, negating it "
                        "on the device moves the source by 2.00x the force exactly (a sign flip's "
                        "signature), and the force is 15.4% of the final source -- so assembling and "
                        "relaxing BEFORE adding it, which is what OpenFOAM does, is not cosmetic.",
             note="fvm::ddt(rho,U) + fvm::div(rhoPhi,U) + MRF.DDt(rho,U) + turbulence->divDevRhoReff(rho,U), "
                  "with the momentum predictor's source reconstructed from surfaceTensionForce() - "
                  "ghf*snGrad(rho) - snGrad(p_rgh). Assembly from pieces brae already had, EXCEPT two. "
                  "(1) fvm::ddt(rho,U) puts rho on the diagonal and rho.oldTime() in the SOURCE "
                  "(EulerDdtScheme.C:455-467); here those differ by the water/air ratio in every cell the "
                  "interface crossed, so carrying one rho field is a 1000x error exactly at the interface. "
                  "(2) the momentum source is a reconstructed FACE flux, not a cell gradient, and "
                  "`solve(UEqn == R)` adds it with a PLUS where rhoSimpleFoam's twin carries the minus "
                  "inside R. MRF, fvOptions and localEuler/CrankNicolson ddt are refused by name. `Gauss "
                  "limitedLinear` on div(rhoPhi,U) was too, and runs now (interFoam_limitedLinear)."),
        dict(name="interFoam_pEqn", of_symbol="pEqn",
             of_file="applications/solvers/multiphase/interFoam/pEqn.H",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_peqn_cpp.cuh",
             brae_target="src/applications/solvers/interFoam/device_inter_peqn.cu",
             validation="tests/test_inter_peqn_cpp.cu covers the four parts that are interFoam's own; the "
                        "laplacian, the solve and the non-orthogonal loop are shared machinery gated "
                        "elsewhere. ON THE DEVICE, tests/test_device_inter_peqn.cu: all four bit-identical "
                        "to the host, each beside its own wrong version on a fixture built to make that "
                        "version wrong -- rAU = dt/rho so the interface carries a 1000x jump. Carrying "
                        "snGrad(p_rgh) into phig moves it by 1.4e+02 of 1.3e+02; "
                        "interpolate(rho)*interpolate(rAU) is 2.5e-01 where the true product is dt = "
                        "1.0e-03 exactly, i.e. 250x; reconstruct without the rAUf division is 8.7e-01 out "
                        "in U; and a constant ddtCorr coefficient instead of the limiter is 1.5e+02 of "
                        "4.0e+02, while the real one is 0.000e+00 on every value-fixing patch. THE "
                        "p_rgh MATRIX IS ASSEMBLED ON THE DEVICE TOO and gated as a MATRIX -- "
                        "off-diagonals bit-identical, diagonal 1.7e-18 of 1.1e-02 and source 4.4e-16 of "
                        "2.2e+00 -- with two arms a solved field could not show: fvMatrix::operator== is "
                        "a PLUS (taking it as a minus moves the source by twice its own size, and still "
                        "converges), and setReference DOUBLES the diagonal entry rather than replacing "
                        "the row, touching no other cell. phiHbyA's two interFoam-only terms and "
                        "pEqn.flux() are bit-identical to the host on both sides -- and dropping phig's "
                        "BOUNDARY half moves fvc::div(phiHbyA) by 9.4e+01 of 1.9e+02, half the source, "
                        "because `phiHbyA += phig` is a whole-surfaceScalarField operation and the "
                        "divergence sums the boundary faces. On capillaryRise, where momentumPredictor "
                        "is off, that is the only route surface tension has into the solution. "
                        "ONE WHOLE PASS runs on the device (device_inter_pressure_step.cu, stages 1-4 "
                        "reusing simpleFoam's pressurePredictor) and is gated against the host's "
                        "pressureCorrector: p_rgh 4.0e-11 relative, U 1.6e-12, p rebuilt exactly. Two "
                        "arms measure placements: dropping phig's boundary half moves the SOLVED p_rgh "
                        "by 6.1e+05 of 1.2e+06 -- HALF the field, not just the flux -- and p is 0.000e+00 "
                        "from p_rgh_solved + rho*gh against 1.2e+06 from the p_rgh the pass started "
                        "with, which is the difference between rebuilding p and carrying it.",
             note="rAU = 1/UEqn.A(), phiHbyA, the p_rgh laplacian, then U and phi rebuilt. Four things "
                  "are not shared with any other pressure corrector. (1) phig carries NO snGrad(p_rgh) "
                  "where UEqn's source does -- the pressure gradient is explicit there and IMPLICIT "
                  "here, and carrying it twice still converges, to the wrong balance at the interface. "
                  "(2) ddtCorr is weighted by interpolate(rho*rAU), the PRODUCT interpolated once, not "
                  "by interpolate(rho)*rAUf: rAU is of order dt/rho, so the two factors jump opposite "
                  "ways across the interface and the rival form is 250x out there while agreeing "
                  "exactly on any single-phase fixture. (3) the velocity correction divides by rAUf "
                  "INSIDE reconstruct and multiplies by rAU outside -- identical for a uniform rAU, "
                  "which is every easy fixture. (4) when p_rgh has no value-fixing patch, p is shifted "
                  "AND p_rgh is rebuilt from the shifted p; damBreak's totalPressure atmosphere hides "
                  "this, 8 shipped tutorials do not. "
                  "THE SOLVE ITSELF WAS A SUBSTITUTION IN TWO WAYS, and on a VoF case it shows. brae ran "
                  "PBiCGStab where every damBreak-family tutorial names `solver PCG; preconditioner DIC;`, "
                  "and took relTol from p_rghFinal for all three correctors where pEqn.H:50 solves with "
                  "p_rgh.select(pimple.finalInnerIter()) -- `p_rgh` (relTol 0.05) for the first two, "
                  "`p_rghFinal` for the last (pimpleControlI.H:98-111). Both are fixed: brae::pcg is "
                  "lduMatrix PCG + DICPreconditioner (gated in tests/test_pcg.cu), InterFields reads both "
                  "entries with lduMatrix::solver's own defaults (tolerance 1e-6, maxIter 1000 -- brae had "
                  "2000), a missing p_rghFinal is refused by name, and any other solver still runs "
                  "PBiCGStab under a noticeApproximated that says it moves the stopping point, not just the "
                  "cost. MEASURED on damBreak against real OpenFOAM, five fixed steps on the case's own "
                  "solves: alpha 1.24e-08 -> 2.2337e-12, p_rgh 3.35e-06 -> 7.194e-10 relative, U 9.76e-06 "
                  "-> 1.966e-08 relative, and the gate bounds tightened to 5e-11, 1e-8 and 4e-7 with them. "
                  "Over ten adaptive steps alpha's over-1 excursion -- which is dt times this solve's "
                  "continuity error -- went from 0.12-3.18x OpenFOAM's to 0.99-1.00x. Each half alone is "
                  "not enough: PCG with the old relTol gave 0.73-4.90x, PBiCGStab with the per-corrector "
                  "selection 1.98-18.41x. This is the lesson turbulentFlatPlate:kEpsilon taught about DILU: "
                  "a substituted solver at the same tolerance does not only cost differently, it STOPS "
                  "somewhere else. capillaryRise does not move (0.7%), so that gap is not this. (The device "
                  "ran its own solver on p_rgh at this point; it takes the case's too now -- see the end "
                  "of this note.) "
                  "WHAT capillaryRise's 0.7% WAS, since it was not the solver: three defects in how the "
                  "mixture meets a contact-angle wall (interFoam_immiscibleMixture) and one here. "
                  "OpenFOAM's flux-conditional patches LOOK PHI UP when they update -- "
                  "pressureInletOutletVelocity, inletOutlet, totalPressure all call "
                  "lookupPatchField(phiName_) inside updateCoeffs -- and brae's are TOLD, through "
                  "updateFromFlux, which interFoam never called. Every such face was therefore an OUTFLOW "
                  "face for the whole run. capillaryRise's bottom inlet draws water IN, so its "
                  "pressureInletOutletVelocity lost the two fixed tangential components and with them 2/3 "
                  "muEff magSf deltaCoeffs of UEqn.A() in every cell of the bottom row: 5.33e+05 by that "
                  "formula, 5.339e+05 measured against OpenFOAM's own UEqn.A() dump. It appears at step two "
                  "because step one starts from rest. Fixed with pushFluxToPatches (at start-up, inside "
                  "pressureCorrector between `phi = phiHbyA - p_rghEqn.flux()` and U's boundary evaluation, "
                  "which is pEqn.H's order, and after the corrector loop for alpha's inletOutlet); the "
                  "device had the same omission plus its own -- it never called "
                  "deviceUpdatePressureInletOutletVelocity, so its inflow faces stayed zeroGradient -- and "
                  "agreed with the host only while the host was wrong the same way. RESULT, "
                  "tests/interfoam_capillaryrise_vs_openfoam.sh, five steps: U 2.8e-03 -> 2.1e-08 (0.7% -> "
                  "5.0e-08 of |U|), alpha 4.6e-05 -> 4.6e-09, p_rgh 1.3e-04 -> 8.1e-08 relative, wall "
                  "gradient 4.5e-05 -> 5.2e-09 of its peak, UEqn.A() to 4.3e-09 and the wall's rho*nu to "
                  "1e-17; with every solve tightened the host is 5.5e-14 of |U| after one step. The gate's "
                  "bound went 0.02 -> 1e-6 and it now checks alpha, p_rgh and the pressure corrector's "
                  "terms wall-against-rest. damBreak moved too, the right way: alpha 2.23e-12 -> 1.30e-12, "
                  "p_rgh 7.2e-10 -> 6.5e-10, U 2.0e-08 -> 1.5e-08. THE DEVICE now has a gate on this case, "
                  "against OpenFOAM directly: 3.5e-08 with every solve tightened on both codes (2.2e-12 "
                  "after one step), and 9.2e-04 at the case's own tolerances, which WAS its p_rgh solver "
                  "stopping somewhere the case's PCG+DIC does not -- closed since, 4.8e-08. "
                  "snGrad(rho) ON A PATCH WAS CARRIED HERE AS `LATENT` AND IT IS NOT SMALL. createFields.H "
                  "builds rho from `alpha1*rho1 + alpha2*rho2`, so its patches are `calculated` and their "
                  "snGrad() is the base class's, deltaCoeffs*(rho_b - rho_cell) (fvPatchField.C:220-223); "
                  "brae took fvc::snGrad(rho) from a zeroGradient copy, which is 0 on every patch. On a "
                  "fixedFluxPressure wall the term cancels through constrainPressure, and everywhere else "
                  "on damBreak and capillaryRise alpha's patch value equals the cell's, so no shipped- "
                  "tutorial gate could see it and this note said so. THE FIXTURE THAT SEES IT took two "
                  "attempts, and the first is worth keeping: raising damBreak's water column to the "
                  "atmosphere did NOT exercise the term -- brae already agreed to 3.6e-13 -- because the "
                  "faces over the water turned out to be OUTFLOW (the patch fixes p_rgh, not p, so the "
                  "column top sees the lower pressure) and OpenFOAM's own dumped boundary snGrad(rho) was "
                  "`uniform 0`. What does it is the atmosphere's inletValue set to 1, so water enters over "
                  "air cells: OpenFOAM's boundary snGrad(rho) is then 1.57e+05 on 19 of 46 faces, phig "
                  "there is 1.66e-02 against a phiHbyA of 2.5e-06 -- four orders above everything else in "
                  "the pressure source -- max|U| goes 0.19 -> 20 m/s in ONE step, and brae was 100% out in "
                  "alpha, p_rgh and U from the second step on, with step two's alpha solve still exact "
                  "(4.4e-13) and its first p_rgh solve 100% different. Fixed on both paths with rho's real "
                  "patch values (rhoWithPatchValues; the device hands its own rhoBnd, which carries "
                  "alpha2's one-pass-older patch values, to the interfaceForces hook, and a missing patch "
                  "value is refused by name rather than defaulted to the cell's). Gated as the `inflow` "
                  "profile of tests/interfoam_dambreak_vs_openfoam.sh, three steps: host alpha 1.3e-14, "
                  "p_rgh 4.3e-11, U 6.1e-13; device 2.3e-14, 6.9e-10, 5.8e-10; every p_rgh and alpha "
                  "iteration count OpenFOAM's; the water let in equal to 1e-12. Its control is on the "
                  "ORACLE: OpenFOAM's own answer is 67x the standard case's whole velocity away from the "
                  "standard case's, so agreeing with it cannot be had without the term. THREE STEPS ONLY, "
                  "and the device's 6.9e-10 is why: it does not move when every solve is tightened on both "
                  "codes, so it is not the tolerance ball, and device against host under tight solves is "
                  "1e-12 after two steps and 7e-10 after three -- round-off growing 700x a step through a "
                  "violent transient, where a formula difference would have appeared at full size in step "
                  "two. At dt 1e-3 OpenFOAM's own alpha reaches 8 on this fixture; it is a probe for one "
                  "term, not a flow. "
                  "THE DEVICE NOW RUNS THE CASE'S PCG WITH DIC TOO, and DIC did not have to be written. "
                  "DICPreconditioner.C and DILUPreconditioner.C are the same two functions with `lower` "
                  "replaced by `upper` -- same face order, same multiplication order, same divisions -- so "
                  "the level-scheduled exact DILU the device already had IS DIC on a symmetric matrix. "
                  "deviceDICPCG aliases lower to upper (a symmetric lduMatrix has no lower; "
                  "lduMatrix::lower() returns upper) and tests/test_device_dic.cu holds it BIT-IDENTICAL, "
                  "by memcmp, to a line-for-line transcription of DIC's loops, with the `lower` buffer "
                  "deliberately filled with garbage. The solve arm is posed on the ITERATE at relTol 0.05, "
                  "not the converged answer, on a fixture with interFoam's own 1000x coefficient jump: host "
                  "and device both take 8 iterations and land 5.3e-16 apart, where Jacobi in the SAME CG "
                  "loop takes 21 and lands 3.4e-03 away. (Its first fixture, a uniform laplacian on 315 "
                  "cells, reached relTol 0.05 in ONE iteration, so 'the same iteration count' compared 1 "
                  "with 1; the gate now refuses a fixture that takes fewer than five.) THE SOLVER'S OWN LOG "
                  "IS NOW AN ORACLE: RunReport carries every p_rgh solve's initial residual, final residual "
                  "and iteration count, and both OpenFOAM gates read OpenFOAM's `DICPCG: Solving for p_rgh` "
                  "lines beside them. On capillaryRise and on damBreak, host and device, ALL FIFTEEN solves "
                  "take OpenFOAM's iteration count (capillaryRise 1/22/165, 1/65/173, 1/43/177, 1/19/173, "
                  "1/4/126), with step one's initial residuals to 1.3e-13. capillaryRise's device arm at "
                  "the case's own tolerances went 9.2e-04 -> 4.8e-08 and its bound 2e-3 -> 1e-6, which is "
                  "the host's; what is left on both is the last corrector's `tolerance 1e-07` ball, inside "
                  "which two correct CG runs whose reductions sum in a different order land in different "
                  "places. THAT LOG THEN FOUND A DEFECT NO FIELD GATE COULD SEE. On damBreak the third "
                  "solve of step one was 1.7e-06 out in its initial residual with the first two exact (0, "
                  "9.3e-14). totalPressure's updateCoeffs for a dimPressure field is p0 - "
                  "0.5*rho_b*neg(phi_b)*|U_b|^2 (totalPressureFvPatchScalarField.C:118-127), run by the "
                  "fvMatrix constructor inside fvm::laplacian with rho, phi and U looked up as PATCH "
                  "fields. brae's class implements it and interFoam never called it, so damBreak's "
                  "atmosphere sat at p0 for the whole run with no dynamic pressure on the faces drawing air "
                  "in. Fixed on both paths (updatePressurePatchesFromVelocity, refusing by name a "
                  "totalPressure patch with no rho patch values rather than defaulting rho to 1, which is "
                  "the incompressible form). damBreak against OpenFOAM: alpha 1.30e-12 -> 3.6e-14, p_rgh "
                  "6.5e-10 -> 8.6e-12, U 1.5e-08 -> 2.3e-11, step one's residuals 1.7e-06 -> 1.6e-13; "
                  "bounds tightened to 1e-12, 2e-10 and 5e-10. That gate also has a DEVICE arm now, at the "
                  "case's own tolerances -- what `brae_interFoam -device` runs -- held to the host's "
                  "bounds."),
        dict(name="interFoam_smoothSolver", of_symbol="smoothSolver",
             of_file="src/OpenFOAM/matrices/lduMatrix/solvers/smoothSolver/smoothSolver.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/OpenFOAM/matrices/smooth_solver_cpp.cuh",
             brae_target="src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_gauss_seidel.cu",
             validation="THREE LINKS, EACH MEASURED. (1) THE SWEEP against OpenFOAM's own numbers: "
                        "tests/gs_ladder.cu used to carry its own inline transcription of "
                        "symGaussSeidelSmoother.C, which proved that THAT copy was OpenFOAM's; its LEG 1 and LEG "
                        "3 now call brae::gaussSeidelSmoothFolded, the function the host solver runs, and hold it "
                        "to OpenFOAM's residual after exactly n sweeps for n = 1..10 on T3A's real momentum "
                        "system -- symGaussSeidel 8.0e-13, GaussSeidel 1.7e-13. (2) THE LOOP, "
                        "tests/test_smooth_solver_cpp.cu: nIterations counts SWEEPS and the residual is evaluated "
                        "once per nSweeps, so `nSweeps 2; maxIter 5` runs SIX; a negative nSweeps is a fixed "
                        "count that reports NO residual; minIter forces a converged system through the loop; a "
                        "mesh out of OpenFOAM's upper-triangular face order is refused, not smoothed. Its oracle "
                        "is the device's deviceSymGaussSeidel in LEVEL-SCHEDULED mode: same sweep count on four "
                        "solves (13, 94, 28, 187) and iterates 0.000e+00 apart -- bit-identical after 187 sweeps "
                        "from two implementations that share no loop. THE FIRST DRAFT OF THAT ARM COMPARED A HOST "
                        "SWEEP WITH A HOST SWEEP: deviceSymGaussSeidel's DEFAULT is to run the sweep on the CPU "
                        "(faster on these meshes), so 0.000e+00 was true and said nothing. The gate now selects "
                        "the device loop before the first solve and ASSERTS it got it, through "
                        "deviceGaussSeidelUsesHostSmoother(). Its control: at relTol 0.05 PBiCGStab's iterate is "
                        "3.3 away from this solver's from the same initial residual. (3) END TO END, "
                        "tests/interfoam_dambreak_vs_openfoam.sh, against OpenFOAM's `smoothSolver: Solving for "
                        "alpha.water` log lines, host AND device: every sweep count equal, and at dt 5e-3 "
                        "(OpenFOAM's counts 0, 5, 2, 2, 2) the FINAL residuals to 9.1e-10 on the host and 5.0e-09 "
                        "on the device, with a PBiCGStab control that gets two counts of five and is 100% out. "
                        "WHAT IT DOES NOT CLAIM: coupled interfaces. The smoothers' "
                        "initMatrixInterfaces/updateMatrixInterfaces calls have nothing to do in a serial run "
                        "with no coupled patch, and the host transcription omits them; a processor or cyclic "
                        "patch is not covered.",
             note="Ported because interFoam's MULESCorr pre-solve names `solver smoothSolver; smoother "
                  "symGaussSeidel;` in every tutorial that sets MULESCorr and the host ran DILU-PBiCGStab "
                  "in its place, under a notice. The device already had OpenFOAM's (deviceSymGaussSeidel); "
                  "the host did not. THREE THINGS TO GET RIGHT, none visible in a converged field: the "
                  "reverse half of symGaussSeidel does NOT distribute -- it gathers off the bPrime the "
                  "forward half LEFT (symGaussSeidelSmoother.C:192); nIterations counts sweeps, "
                  "`(nIterations += nSweeps) < maxIter`; and the loop's residual is lduMatrix::residual "
                  "(lduMatrixATmul.C:268), which starts from source - diag*psi and subtracts the off- "
                  "diagonals face by face -- NOT b - A.psi. That last one is invisible until the solve ends "
                  "at round-off: on damBreak the smoother stops at 1e-14, where the rounding IS the "
                  "residual, and with OpenFOAM's operation order transcribed the host reproduces its final "
                  "residuals to four digits there (6.835e-15, 1.202e-14, 1.851e-14, 2.067e-14). THE END-TO- "
                  "END ARM TAUGHT ITS OWN LESSON ABOUT FIXTURES. At the gate's dt 1e-4 the alpha system is "
                  "so diagonally dominant (Co ~ 1e-3) that every solver lands on the exact solution in ONE "
                  "iteration: the PBiCGStab control read 2.6e-03 from OpenFOAM's final residuals, right "
                  "beside the device's honest 1.5e-03, so the arm could not tell solvers apart and would "
                  "have certified either. The gate now runs a second profile at dt 5e-3, where the "
                  "interface moves 0.69 and the solve is real, and asserts the solver-log arms only there. "
                  "The fields hold on that profile too: alpha 1.2e-12 and p_rgh 1.3e-12 relative on both "
                  "paths, U 4.3e-12 on the host and 8.5e-12 on the device."),
        dict(name="interFoam_turbulence", of_symbol="incompressibleInterPhaseTransportModel",
             of_file="src/phaseSystemModels/twoPhaseInter/incompressibleInterPhaseTransportModel/"
                     "incompressibleInterPhaseTransportModel.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_turbulence_cpp.cuh",
             brae_target="src/applications/solvers/interFoam/device_inter_turbulence.cu",
             validation="tests/interfoam_ras_dambreak_vs_openfoam.sh, real OpenFOAM on RAS/damBreak, five fixed steps of "
                        "1e-3, THREE PROFILES. `variable` is the tutorial as shipped; `uniform` is the same case without "
                        "`density variable` (and with div(phi,k) for div(rhoPhi,k), which OpenFOAM then requires); "
                        "`custom` is uniform with its own kEpsilonCoeffs, equation relaxation 0.7, and a solver tolerance "
                        "of 0.5 so that `minIter 1` is what forces each sweep. MEASURED, worst of the three: alpha "
                        "4.9e-13, p_rgh 7.1e-13, U 1.3e-11, k 2.8e-13, epsilon 2.4e-13, nut 4.8e-13, and EVERY p_rgh, "
                        "alpha, epsilon and k solve takes OpenFOAM's iteration count (15+5+5+5 per profile) with initial "
                        "residuals to 1.8e-13. CONTROLS ON THE ORACLE: OpenFOAM run laminar is 133% of U away, its other "
                        "lineage 29%, the custom settings 3.7%. WHAT EACH PROFILE CAN SEE was measured by breaking one "
                        "decision at a time, once, with temporary switches that are not in the tree. On `variable`: "
                        "validate() called in the wrong lineage 29% of U; rho.oldTime() taken as the current rho 5.6%; "
                        "divU from the mass flux 53%; inletOutlet handed rhoPhi where it looks up phi 4.9e-04; PBiCGStab "
                        "for the case's symGaussSeidel 1 of 5 iteration counts and 2.7e-06. On `variable`, ignoring "
                        "minIter and ignoring the `\".*\" 1` relaxation change NOTHING, to the last digit -- which is why "
                        "`custom` exists: there, ignoring minIter is 14% of U and 0 of 5 epsilon counts, ignoring the "
                        "relaxation 2.2%, the coefficients 3.5% (sigmak/sigmaEps alone 0.3%, C3 alone 1.2e-08 of epsilon "
                        "because divU is nearly zero). The staging asserts OpenFOAM's custom log holds a solve whose "
                        "initial residual is under 0.5 and still reads one iteration. tests/interfoam_refusals.sh holds "
                        "16 turbulence arms on RAS/damBreak itself. THE DEVICE LOOP RUNS EVERY PROFILE TWICE. With the "
                        "device closure -- what `-device` runs -- against OpenFOAM at the HOST's bounds: worst of the "
                        "three alpha 4.9e-13, p_rgh 7.1e-13, U 3.3e-11, k 2.8e-13, epsilon 2.5e-13, nut 4.8e-13, every "
                        "iteration count OpenFOAM's. And with the HOST closure in its place inside the same device loop "
                        "(BRAE_INTER_HOST_CLOSURE, announced every run), held to the first: U 3.3e-14, k 1.1e-15, epsilon "
                        "1.6e-15, nut 1.9e-15, the same sweep counts solve for solve. Against OpenFOAM a disagreement "
                        "could be the loop's or the closure's; between those two it can only be the closure's. Each arm "
                        "asserts WHICH closure it ran (RunReport::turbulenceOnDevice). THE DEVICE-SIDE DECISIONS were "
                        "each broken once, as the host's were: the patches handed rhoPhi for phi 4.9e-04 of U; the wall "
                        "nut recomputed instead of read from the stored patch 6.1e-06; rho.oldTime() taken as rho 5.6%; "
                        "divU from the mass flux 53%; BiCGStab for symGaussSeidel 1 of 5 epsilon counts; on `custom`, "
                        "minIter ignored 14% and the wall laplacian coefficient left out of relax() 1.1e-04 of every "
                        "epsilon residual with the fields unmoved. NOT VISIBLE on these fixtures, and so not claimed: the "
                        "closure's storedIoSeed (the case starts at rest with inletValue equal to the internal field; "
                        "switching it off changes no digit). WHAT IT DOES NOT CLAIM: any nut wall function but nutk (nutU "
                        "and nutLowRe are accepted because the shared closure gates them under rhoSimpleFoam, not here); "
                        "kMin/epsilonMin, which bound() never reaches on this case; a non-orthogonal mesh; more than one "
                        "outer corrector with the closure on.",
             note="nut = Cmu*sqr(k)/epsilon DOES NOT REACH a fixedValue or mixed patch (their operator= is empty); correctBoundaryConditions then evaluates them. brae wrote Cmu*k^2/epsilon there: on RAS/mixerVesselAMI's fixedValue gasInlet 1.29e-03 where OpenFOAM keeps 0, U 2.7e-03 after one step. AN inletOutlet nut, under kEpsilon and kOmegaSST alike, is EVALUATED after the assignment against the flux (Compressible::nutPhi), inflow faces taking the inletValue: brae wrote Cmu*k_b^2/epsilon_b there (kEpsilon) or left the patch stale (kOmegaSST, which already skipped fixedValue) -- measured U 2.7e-03 on RAS/damBreak `nutAtmosphere` and nut 8.5e-04 on waterChannel `nutPatches`; a caller with no flux is refused by name, and so is the device closure. kOmegaSST ALSO SKIPPED every other non-calculated patch (zeroGradient, symmetry, slip, coupled): they kept the value they were built with; they are now evaluated as correctBoundaryConditions does -- measured U 8.7e-05 with waterChannel's inlet nut zeroGradient, 3.8e-12 after (`nutInletZeroGrad`). kEpsilon ON A MOVING MESH: V0 in the ddt source and divU from the absolute flux (kEpsilon.C:232-235), not discriminated by the rigid rotation that gates it. "
                  "incompressibleInterPhaseTransportModel IS TWO MODELS BEHIND ONE KEYWORD, and brae's own refusal "
                  "message had it wrong. It said interFoam's turbulence `selects a MIXTURE model and hands it the "
                  "blended rho -- not the single-phase model brae already has`; that was written from memory. The "
                  "source (incompressibleInterPhaseTransportModel.C:46-110): with `density` absent or `uniform` -- "
                  "15 of the 17 turbulent tutorials -- it builds incompressible::turbulenceModel::New(U, phi, "
                  "mixture), the ORDINARY model, and calls validate(); with `density variable` (RAS/damBreak, "
                  "laminar/damBreakPermeable) it builds phaseIncompressibleTurbulenceModel::New(rho, U, rhoPhi, "
                  "phi, mixture) and does NOT call validate(). So the variable lineage's first UEqn runs on the "
                  "case file's nut (uniform 0) where the other has 9e-03 from k = epsilon = 0.1, and the fvSchemes "
                  "KEY differs too: div(rhoPhi,k) against div(phi,k). brae already had the closure for both -- "
                  "kEpsilonRef::correct's `Compressible` struct is OpenFOAM's one templated kEpsilon.C read with "
                  "rho, a field nu, the mass flux for div and the volumetric one for divU -- so the port is what it "
                  "is HANDED, in inter_turbulence_cpp.cu. Both lineages take the MIXTURE's nu, a field. UEqn takes "
                  "rho*nuEff with nuEff = nut + nu on cells and on every patch. turbulence->correct() runs after "
                  "the last pressure corrector of the final outer corrector (interFoam.C:169-172, "
                  "pimpleControl.C:51-52), where fvMatrix::solve() AND fvMatrix::relax() both select the Final "
                  "name, so kFinal/epsilonFinal are the entries read and `turbOnFinalIterOnly no` with more than "
                  "one outer corrector is refused. THREE THINGS THE CLOSURE NEEDED. (1) The case's linear solver: "
                  "it hardwired PBiCGStab, and every kEpsilon tutorial names smoothSolver/symGaussSeidel -- the "
                  "fourth time in this port a substituted solver at the same tolerance stopped somewhere else. (2) "
                  "Compressible::bcPhi: inletOutlet looks up the registry's `phi`, which in rhoSimpleFoam is the "
                  "equation's own mass flux and in this lineage is NOT (the equation convects with rhoPhi). A sign "
                  "argument -- rho_b > 0 -- says it cannot matter, and that argument went into the header before it "
                  "was measured. It is wrong: rhoPhi is the ALPHA step's flux, built from the phi the time step "
                  "STARTED on, and the pressure correctors move phi afterwards. Counted at step one of "
                  "RAS/damBreak, which starts at rest: rhoPhi is exactly 0 on all 46 atmosphere faces while phi is "
                  "not, so every one of them reads as outflow. 4.9e-04 of U after five steps, host and device "
                  "alike. (3) A DEFECT IN THE SHARED CLOSURE, found by the `custom` profile: epsilon agreed with "
                  "OpenFOAM to 1e-13 while every initial and final residual of its solve sat 1.1e-04 away, only "
                  "under relaxation, only for epsilon. OpenFOAM's epsilonWallFunction is a fixedValue patch, so at "
                  "relax() its faces carry the laplacian's coefficient ic and the diagonal becomes max(|D0 + ic|, "
                  "sumOff)/alpha - ic; brae maps the patch to zeroGradient and got D0/alpha. setValues pins exactly "
                  "those rows next, so the solution cannot see it, but the diagonal survives into normFactor and a "
                  "residual is what the stopping rule reads. Fixed in the host reference: 1.1e-04 -> 4.6e-14, and "
                  "then in the device twin kEpsilon.cu, measured the same way from the device loop: 1.1e-04 -> "
                  "5.3e-14. STILL OPEN, and not measured: neither kOmegaSST closure carries the term -- "
                  "omegaWallFunction has the same fixedValue base class "
                  "(omegaWallFunctionFvPatchScalarField.H:88-90), so the same term is expected there, which is a "
                  "lead and not a measurement. It moves a residual by 1e-04, so it can matter only where a solve "
                  "stops within 1e-04 of its tolerance; rhoSimpleFoam's kEpsilon gates, which relax at 0.7, pass "
                  "with and without it. TWO LATENT SUBSTITUTIONS THE TURBULENCE REFUSAL HAD BEEN HIDING, found by "
                  "asking what else would run once it lifted. `ddtSchemes default` was parsed into f.ddtU and "
                  "handed to nobody, so CrankNicolson (floatingObject) and localEuler (DTCHull) would have run as "
                  "Euler: now refused in buildInterFields. And laplacianSchemes/snGradSchemes were never opened: "
                  "brae assembles the pressure laplacian and its three snGrads orthogonal while 28 of 44 tutorials "
                  "say `corrected` or `limited`. On a mesh of rectangles that is not a substitution (the correction "
                  "vector is zero; it is why damBreak agrees to 1e-12 under `Gauss linear corrected`); on any other "
                  "it is, so a corrected scheme on a mesh more than 1e-10 off orthogonal is refused, and the "
                  "refusals gate shears damBreak six degrees to prove both sides. `minIter` is honoured for k and "
                  "epsilon and REFUSED for the alpha pre-solve and the momentum predictor, which drop it (three "
                  "tutorials name it for alpha). THE CENSUS CAUGHT ITS OWN DEFECT: run with blockMesh alone it "
                  "showed damBreakLeakage and damBreakPorousBaffle reaching End, on meshes without the baffles "
                  "their Allrun creates. Prepared by their own Allrun utilities both are refused "
                  "(porousBafflePressure by name; the leakage case in the mesh reader, on a cyclicACMI `coded` "
                  "block -- a refusal, but not a named one). 33 tutorials prepare cleanly: 3 reach End and all 3 "
                  "are gated; waves 9, moving meshes 12. ON THE DEVICE IN TWO STEPS, one module at a time. First "
                  "the device loop with the HOST closure -- nuEff into the device UEqn, and U, phi, rhoPhi, rho, "
                  "rho.oldTime() and nu handed to the closure at the right instants -- which matched OpenFOAM at "
                  "the host's numbers before any device closure ran. Then the closure's existing device twin "
                  "(gpu::kEpsilonRAS::correct, gated under rhoSimpleFoam) in its place, fed by "
                  "device_inter_turbulence.cu, which owns no arithmetic: the equation's flux is rhoPhi or phi by "
                  "lineage, divU's and the patches' flux is phi always, rho is the mixture's or a vector of ones, "
                  "and rho's patch values are the device step's own blend. nut is added to the mixture's nu ON the "
                  "device (DeviceInterStepControls::nutCell), so it never crosses to the host. The twin needed "
                  "three things the host reference had been given: the patches' flux (bcPhiBnd), whole-solve "
                  "reporting for the log arm, and the wall coefficient in relax(). A nut wall function on a patch "
                  "that is not a `wall` is refused on both paths, as OpenFOAM's nutWallFunction::checkType refuses "
                  "it."),
        dict(name="interFoam_kOmegaSST", of_symbol="kOmegaSSTBase",
             of_file="src/TurbulenceModels/turbulenceModels/Base/kOmegaSST/kOmegaSSTBase.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST_cpp.cuh",
             validation="tests/interfoam_waterchannel_vs_openfoam.sh, real OpenFOAM on RAS/waterChannel AS SHIPPED "
                        "(its own 28000-cell blockMesh + two extrudeMesh passes, non-orthogonal to 13.7 degrees), "
                        "ten fixed steps of the tutorial's deltaT 0.1. MEASURED: all 20 p_rgh, 10 alpha, 10 omega and "
                        "10 k iteration counts OpenFOAM's, initial residuals within 9.3e-13; alpha 7.9e-13, p_rgh "
                        "7.6e-13, U 4.3e-12, k 2.4e-12, omega 3.9e-12, nut 3.5e-12, the wall's nut 9.2e-13 of its "
                        "largest value; bounds at about 30x. THE CONTROL: OpenFOAM's own laminar run, 51% of U. THE "
                        "PATH is asserted: the model, the uniform lineage, a wall distance in every cell, OpenFOAM's "
                        "log selecting kOmegaSST and solving omega. BROKEN ONCE EACH (U, omega at t = 1): validate() "
                        "skipped 5.1e-01, 1.7e-01; fvm::ddt dropped 3.8e-01, 2.2e+01 and 0 of 10 omega counts; the "
                        "wall distance doubled 4.1e-04, 5.1e-01; PBiCGStab for the case's symGaussSeidel 1.7e-04, "
                        "3.7e-02; water's nu as a scalar for the mixture's field 3.5e-05, 8.5e-03; the laplacian's "
                        "non-orthogonal correction dropped 3.7e-07, 7.2e-04. NOT DISCRIMINATED, measured: the inline "
                        "(value - cell)*deltaCoeffs for the patch's snGrad() on k, omega and the wall U (identical to "
                        "the last digit), and relax() not called at the case's factor of 1. "
                        "tests/interfoam_refusals.sh holds ten kOmegaSST arms on RAS/damBreak made kOmegaSST, on a "
                        "base that runs. NOT CLAIMED, each refused by name: `density variable` with kOmegaSST (no "
                        "tutorial pairs them), F3, decayControl, a wall-function blending other than binomial n = 2 "
                        "or coefficients other than the defaults, any nut wall function but nutk, a convection "
                        "scheme other than Gauss upwind; on a moving mesh, a wallDist "
                        "updateInterval other than 1 (a missing `wallDist { method }`, which OpenFOAM stops on, and a "
                        "method other than meshWave or correctWalls false are refused everywhere). ON A MOVING MESH "
                        "(2026-09-19), tests/interfoam_moving_vs_openfoam.sh `pistonSST`: waves/waveMakerPiston made "
                        "kOmegaSST, its paddle a wall and its top a patch, 30 steps of 0.01 against OpenFOAM, k and "
                        "omega solved to 1e-12. MEASURED: k 8.6e-12, omega 2.4e-12, nut 3.0e-10, U 1.2e-10, alpha "
                        "1.4e-12, every p_rgh, pcorr, k and omega count OpenFOAM's or one apart at an edge stop; "
                        "bounds at about 30x. THE CONTROL: OpenFOAM's laminar piston, 1.9 of U. BROKEN ONCE EACH "
                        "(nut, U): y from the wall patches instead of the shared wallDist 6.4e-01, 2.3e-02; fvm::ddt's "
                        "old volumes dropped 2.0e-03, 3.6e-05; divU from the relative flux 4.8e-03, 4.5e-05; y not "
                        "recomputed after the motion 9.5e-04, 1.7e-05. THE SHARED wallDist: MeshObject::New finds an "
                        "object by its type name alone, and the motion solver's inverseDistance diffusivity "
                        "registers `wallDist` over ITS patches before the turbulence model is built, so under "
                        "displacementLaplacian kOmegaSST's y is the distance to the diffusivity's patches -- on the "
                        "piston 0.01, 0.03, 0.05 away from the paddle and 1.01 at x = 1, over a bottom wall 0.0025 "
                        "below; OpenFOAM's own static run, which builds no such object, gives brae's wall-patch "
                        "answer to every digit. brae mirrors it (InterTurbulence::wallDistPatchIDs). ON THE DEVICE "
                        "(2026-09-20): device_inter_turbulence.cu's SST branch hands the closure rhoSimpleFoam gates "
                        "(kOmegaSST.cuh) what the host's SST branch hands its reference -- the volumetric phi as the "
                        "equation's flux and divU's, rho = 1 (`density variable` with kOmegaSST is refused), the "
                        "mixture's nu, 1/deltaT, phi for nut's flux-conditional patches, the cell wall distance, the "
                        "F1-one and nut-calculated masks, and the case's Final solver entry and relaxation. GATED as "
                        "tests/interfoam_ras_dambreak_vs_openfoam.sh `sst`, RAS/damBreak made kOmegaSST (its nut "
                        "atmosphere is `calculated`, which is what the device closure can run): host against OpenFOAM "
                        "k 4.2e-13, omega 2.4e-13, nut 3.7e-12, every omega (45/9/8/7/7) and k (142/31/22/16/13) "
                        "iteration count OpenFOAM's; THE DEVICE CLOSURE AGAINST THE HOST CLOSURE IN THE SAME DEVICE "
                        "LOOP -- the module's own claim -- U 8.6e-13, k 1.2e-13, omega 6.6e-14, nut 9.5e-13, and with "
                        "every solve tightened 1.4e-13 / 5.3e-15 / 7.8e-16 / 3.7e-14. Device against OpenFOAM is "
                        "looser (k 4.4e-12, omega 3.1e-11, nut 1.7e-10, U 5.1e-11) and TIGHTENING DOES NOT MOVE IT: "
                        "the device LOOP's U is about 5e-11 from OpenFOAM on the kEpsilon profiles too and nut carries "
                        "it through k/omega and F2, so the `sst` bounds are that, at about 30x. BROKEN ONCE EACH "
                        "(device closure against host closure): the wall distance replaced by 1 -- nut 4.3e-02; nut's "
                        "`calculated` mask dropped 1.2e-01; the stored wall nut dropped 8.8e-05 of k. NOT "
                        "DISCRIMINATED: the F1-one mask -- dropping it changes no digit on this fixture, where F1 "
                        "evaluates to 1 on those faces anyway. AN INLETOUTLET nut RUNS ON THE DEVICE TOO (2026-09-20): "
                        "nut carries its own DeviceBoundary and deviceCorrectInterTurbulence evaluates the "
                        "flux-conditional faces after the closure, where the host closure does -- "
                        "deviceUpdateInletOutlet's valueFraction = neg(phi), then deviceBCValue, copied onto the "
                        "ioMask faces alone so a wall function's or a `calculated` patch's value is untouched. Gated as "
                        "tests/interfoam_ras_dambreak_vs_openfoam.sh `nutAtmosphere` on BOTH closures: device against "
                        "OpenFOAM alpha 5.3e-15, U 2.0e-13, k 2.1e-15, epsilon 2.9e-15, nut 2.7e-15, and the device "
                        "closure against the host closure 1.5e-15 of nut; without the evaluation nut is 4.7e-02 out and "
                        "U 2.7e-03. THE DEVICE GAMG NOW CARRIES OpenFOAM's OTHER SMOOTHERS (2026-09-20): GaussSeidel, "
                        "DICGaussSeidel and symGaussSeidel, each one smoothLevel() running nSweeps of "
                        "deviceSymGaussSeidelSweepExact (the forward-then-reverse walk gated bit-for-bit against the "
                        "host) after DIC's half where the name asks for it, at all three V-cycle sites. Gated as "
                        "tests/interfoam_gamg_vs_openfoam.sh: the three Gauss-Seidel profiles' device arms take "
                        "OpenFOAM's OWN iteration counts, 40 of 40 p_rgh solves and every coarsest count, initial "
                        "residuals within 2.3e-10, alpha 2.1e-12 / 2.6e-12 / 3.2e-12. BROKEN ONCE EACH: the two "
                        "Gauss-Seidel variants swapped 1.1e-02 and 4.1e-03; DIC's half dropped from DICGaussSeidel "
                        "2.5e-02 (and correctly no digit on the plain `gaussSeidel` profile). A `DILU` smoother on a "
                        "symmetric matrix is still refused by name (interfoam_refusals `device_gamg_smootherDILU`). "
                        "WHAT LIFTING THAT REFUSAL EXPOSED: RAS/waterChannel's U outlet is a plain inletOutlet, and "
                        "OpenFOAM's updateCoeffs sets its valueFraction from the flux sign at EVERY momentum assembly "
                        "while the device loop assembles the switch U's boundary was built with -- measured U 8.9e-02, "
                        "nut 8.7e-01 against OpenFOAM where the host loop on the same case is 4.3e-12, with alpha exact "
                        "to 5.3e-15 at step 1, every solve tightened on both codes, the HOST closure in the device loop "
                        "still 8.5e-01 of nut, and 1.1e-02 of U with the schemes forced orthogonal -- so neither the "
                        "closure, the stopping point nor the non-orthogonal correction. PORTED (2026-09-20), and it took "
                        "TWO fixes, each localised with the host closure run inside the device loop. (1) The io switch: "
                        "deviceUpdateInletOutlet(dbU, phiBnd) now runs at all three sites device_inter_step.cu rebuilds "
                        "dbU, before the piov switch, in the order rhoSimpleFoam's device arm takes them "
                        "(rhoUEqn.cuh:72-79), and once on the state the run starts from. (2) nut's boundary: the device "
                        "closure evaluated its flux-conditional faces ONLY, where the host closure evaluates every patch "
                        "that is not a `wall`, not `empty` and not `calculated` (kOmegaSST_cpp.cu:946-984) -- so a "
                        "zeroGradient nut inlet kept the value it was built with, U 1.9e-05 and nut 5.5e-06 from "
                        "OpenFOAM where the host closure in the same loop is 2.3e-12. A coupled nut patch is refused "
                        "there by name, because buildDeviceBoundary drops it and the per-face evaluate would read past "
                        "the end. GATED: tests/interfoam_waterchannel_vs_openfoam.sh runs all three profiles on the "
                        "device at the HOST's own bounds -- worst of the three, alpha 2.8e-12, p_rgh 2.6e-12, U 5.9e-12, "
                        "k 3.3e-12, omega 4.7e-12, nut 3.5e-12, and all 20 p_rgh iteration counts OpenFOAM's, which is "
                        "the device GAMG GaussSeidel smoother taking OpenFOAM's counts on a real case. Four device "
                        "modules meet on this fixture: kOmegaSST, nut's boundary, U's inletOutlet and the smoother.",
             note="WIRING, not a new closure: kOmegaSST_cpp.cu is the reference rhoSimpleFoam and simpleFoam already "
                  "gate, handed what incompressible::turbulenceModel::New(U, phi, mixture) hands OpenFOAM's -- the "
                  "mixture's nu as a field, the volumetric phi, 1/deltaT for fvm::ddt, wallDist's cell y -- by "
                  "inter_turbulence_cpp.cu. The reference gained the case's linear solver (it ran PBiCGStab "
                  "whatever the case named) and the two solves' full records. THREE SHARED DEFECTS THE CASE FOUND. "
                  "(1) pressureInletOutletVelocity::snGrad() returned (stored value - cell)*deltaCoeffs where "
                  "OpenFOAM's directionMixed builds it from the valueFraction and the cell and never reads the "
                  "stored value: at construction waterChannel's atmosphere holds the file's (0 0 0) over cells at "
                  "(1 0 0), correctNut read a shear of 1/d there and wrote nut 3.7e-05 for OpenFOAM's k/omega = "
                  "3.33e-02 -- the patch 100% out, the run's FIRST p_rgh residual 7.1e-03, and at t = 1 U 3.5e-03, "
                  "omega 4.0e-02, nut 2.3e-01. Localised by running the same case laminar (already 6e-13) and then "
                  "comparing the constructed nut, patch by patch, with `interFoam -postProcess -func "
                  "writeObjects(nut)`; tests/test_piov_sngrad.cu holds it. (2) The per-field div scheme parser "
                  "searched fvSchemes for the literal key, and the case names its scheme through a pattern, "
                  "`\"div\\(phi,(k|omega)\\)\" Gauss upwind;` -- refused as having no div(phi,k); now the literal "
                  "key, else the last matching pattern, as the dictionary lookup does (tests/test_scheme_blocks.cu). "
                  "(3) kOmegaSST_cpp took three boundary gradients inline instead of from the patch; switched to "
                  "snGrad(), which changes nothing on any gated case. ALSO: kOmegaSST_cpp.cuh says decayControl is "
                  "refused and nothing read the key; interFoam's reader does now. The closure keys both wall "
                  "functions on the MESH patch type, so the reader holds every `wall` patch to nutkWallFunction "
                  "and omegaWallFunction and every other patch to neither. HOST ONLY SO FAR: the device loop "
                  "refuses by name, first among its refusals."),
        dict(name="interFoam_MRF", of_symbol="MRFZoneList",
             of_file="src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/cfdTools/general/MRF/MRF_cpp.cuh",
             validation="tests/interfoam_mrf_vs_openfoam.sh, real OpenFOAM on laminar/mixerVessel2D AS SHIPPED (its "
                        "own m4 blockMesh, topoSet and setsToZones; 3072 cells, the `rotor` zone 1536 of them with "
                        "3024 internal faces and 192 rotor faces that move with the frame), twenty fixed steps of the "
                        "tutorial's deltaT 1e-3. MEASURED: all 60 p_rgh iteration counts OpenFOAM's, initial residuals "
                        "within 1.2e-11; alpha 8.2e-15, p_rgh 4.9e-14, U 1.9e-15; bounds at about 30x. THE CONTROL: "
                        "OpenFOAM with the zone `active no`, where nothing moves (100% of U). THE PATH is asserted: "
                        "one active zone, a proper part of the mesh, with included faces, turning at the case's "
                        "6.2831853 rad/s; OpenFOAM's log building the zone list. BROKEN ONCE EACH (U, p_rgh, p_rgh "
                        "counts equal): makeRelative(phiHbyA) dropped 1.1e+00, 8.7e-01, 24 of 60; the ddtCorr flux "
                        "not zero-filtered 1.7e-01, 2.2e-01, 30 of 60; correctBoundaryVelocity skipped 1.6e-01, "
                        "6.1e-01, 29 of 60; MRF.DDt dropped 2.7e-02, 2.9e-01, 42 of 60; MRF.DDt(U) for MRF.DDt(rho, "
                        "U) the same to two digits (rho is 1000 in the water). tests/interfoam_refusals.sh holds nine "
                        "MRF arms. THE DEVICE LOOP RUNS IT TOO (2026-09-20), at the same "
                        "bounds: alpha 7.3e-15, p_rgh 3.0e-14, U 2.1e-15, and all 60 of ITS OWN p_rgh solves take "
                        "OpenFOAM's iteration counts (initial residuals within 1.5e-11). All four calls were "
                        "transcribed from the host arm's lines: correctBoundaryVelocity in the U-boundary hook, on "
                        "the HOST field the snapshot is taken from (UEqn.H:1); DDt(rho, U) rho-weighted in the shared "
                        "assembler before relax, formed as rhoSimpleFoam's device arm forms it (rhoUEqn.cu:769-777, "
                        "acc then hadamard by rho); zeroFilter on the ddtCorr term (new deviceMrfZeroFilter, whose "
                        "face set is MRFZone::zero's, not makeRelative's); and makeRelative BETWEEN that term and "
                        "phig, which is why the device pEqn now splits the two adds it used to fuse. BROKEN ONCE EACH "
                        "on the device (alpha, p_rgh, U): correctBoundaryVelocity skipped 8.5e-02, 6.1e-01, 1.6e-01; "
                        "MRF.DDt dropped 1.7e-02, 2.9e-01, 2.7e-02; not rho-weighted the same to two digits; "
                        "zeroFilter dropped 1.7e-01, 2.2e-01, 1.7e-01; makeRelative dropped 9.9e-01, 8.7e-01, "
                        "1.1e+00. NOT CLAIMED, each refused by name: MRF under a moving mesh, under RAS, beside a "
                        "fixedFluxPressure patch (constrainPressure's MRF.relative(Sf & U_b)), and an omega that "
                        "varies in time. The rotor's frame velocity has no oracle of its own -- "
                        "OpenFOAM writes a noSlip patch as its type alone -- and is held through U.",
             note="interFoam reaches MRF in four places that do arithmetic (UEqn.H:1, :6; pEqn.H:17, :19), and the "
                  "host reference simpleFoam and rhoSimpleFoam already gate supplies three of them: "
                  "correctBoundaryVelocity, addCoriolis (weighted by rho here, MRFZoneList::DDt(rho, U) = "
                  "rho*DDt(U)) and makeRelative. THE FOURTH IS NEW: MRF.zeroFilter on the ddtCorr flux -- "
                  "MRFZone::zero sets it to Zero on the zone's internal, included and excluded faces "
                  "(MRFZoneTemplates.C:213-247), because inside the zone phi.oldTime() is relative to the frame "
                  "and the flux of U.oldTime() is not, so their difference there is the frame flux and not a "
                  "correction. ONE SHARED DEFECT: `omega` is a Function1 (MRFZone.C) and the tutorial writes "
                  "`omega constant 6.2831853;`; the reader took scalarOr, which reads the LAST token -- right for "
                  "that spelling by accident, a crash on a table's `)` and a silent ZERO for the dictionary form. "
                  "It reads a constant in its three spellings now, refuses any other type by name, and refuses a "
                  "zone with no omega, which OpenFOAM's mandatory Function1::New stops on and brae ran at zero. "
                  "The refusal that used to stand here fired on any active zone; brae's interFoam had once read "
                  "MRFProperties, ignored it and converged. ON THE DEVICE TOO NOW -- and the fixture it needed dragged "
                  "the PRESSURE REFERENCE onto the device with it, since mixerVessel2D is closed: setReference pins "
                  "the cell at its CURRENT p_rgh (pEqn.H:47, getRefCellValue) and not at pRefValue, and pEqn.H:74-83 "
                  "then shifts p's level and REBUILDS p_rgh from the shifted p, so both fields move. Three controls "
                  "in the device driver had never been set, because the device refused every case that needs a "
                  "reference: the step pinned nothing, and p_rgh came out a constant -2.17e+01 from OpenFOAM with a "
                  "spread of only 1.5e-02 about it. What this fixture CANNOT discriminate is the value setReference "
                  "pins at -- g is (0 0 0) there, so p == p_rgh and the cell's own p_rgh IS pRefValue to the last "
                  "digit of every field and every iteration count; test_device_inter_peqn.cu asserts that one "
                  "bit-level with the old behaviour as its control. STILL REFUSED on the device: a case that needs a "
                  "reference and has a boundary patch adjustPhi would weigh -- anything but a wall fixing its flux. "
                  "OpenFOAM either scales the adjustable outflow or aborts when the fixed fluxes do not balance "
                  "(adjustPhi.C:106), and the device step carries neither; damBreak with its atmosphere turned into "
                  "a fixedFluxPressure ran to a worst |div(phi)| of 5.2e-02 before that refusal was narrowed to it."),
        dict(name="interFoam_fvOptions", of_symbol="fv::options",
             of_file="src/finiteVolume/cfdTools/general/fvOptions/fvOptions.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/cfdTools/general/fvOptions/fvOptions_cpp.cuh",
             validation="tests/interfoam_angledduct_vs_openfoam.sh, real OpenFOAM on RAS/angledDuct (its own blockMesh, "
                        "28000 cells non-orthogonal to 44.5 degrees, the `porosity` zone 8000 of them), ten fixed steps "
                        "of the tutorial's deltaT 1e-3, THREE ARMS. `inactive` has the option `active no` in both codes "
                        "and is held at the floor: alpha 5.7e-16, p_rgh 3.8e-15, U 4.3e-15, k 7.6e-15, epsilon 5.8e-15, "
                        "nut 5.2e-15. `porous` is the case as shipped: alpha 1.6e-14, p_rgh 6.3e-11, U 5.1e-11, k "
                        "4.6e-12, epsilon 5.3e-12, nut 5.7e-12. `porousWater` starts the duct full of water: p_rgh "
                        "4.6e-12, U 2.8e-12. Every p_rgh, alpha, epsilon and k iteration count OpenFOAM's in all three. "
                        "THE FOUR ORDERS BETWEEN inactive AND porous ARE ROUND-OFF, MEASURED TWICE: one entry of the "
                        "resistance tensor multiplied by (1 + 2.2e-16) moves brae's p_rgh agreement from 6.3e-11 to "
                        "1.7e-11; and the residual that is 1.5e-10 out is the second corrector's, itself 2.2e-05, "
                        "where the first solve of the run agrees to 1.1e-16 and the same line with the option off -- "
                        "2.5e-02 there -- agrees to 1e-13. Bounds at about 30x, per arm. EACH ARM IS ANOTHER'S "
                        "CONTROL: the option moves OpenFOAM's own U by 23% (73% in water). BROKEN ONCE EACH (U): the "
                        "inlet never refreshed 100%; refreshed with the face cell's rho 6.5e+06; as a volumetric rate "
                        "2.9e+02; mu as rho*nuEff 5.3e-02 (1.8e-01 in water); the kinematic form, no rho, NOTHING on "
                        "`porous` -- rho is exactly 1 in the zone for the whole shipped run -- and 4.9e-01 on "
                        "`porousWater`, which is why that arm exists. tests/interfoam_refusals.sh holds six fvOptions "
                        "arms. NOT CLAIMED, each refused by name: every other option type, explicitPorositySource's "
                        "fixedCoeff model, an option under a moving mesh or beside an MRF zone, the device loop; the "
                        "Forchheimer half (f is zero in the tutorial).",
             note="interFoam reaches fvOptions in four places (UEqn.H:9 `== fvOptions(rho, U)`, :14 constrain, :31 and "
                  "pEqn.H:65 correct); a source reaches the first alone. The arithmetic is the host reference "
                  "simpleFoam and rhoSimpleFoam already gate; what is interFoam's is what it is handed. The "
                  "equation is FORCE-dimensioned, so DarcyForchheimer::correct takes the field named `rho` and, "
                  "finding no `thermo:mu`, rho*nu with the field named `nu` (DarcyForchheimer.C:203-218) -- the "
                  "mixture's rho and its LAMINAR nu. WHAT THE GATE FOUND WAS NOT THE OPTION. interFoam's loops never "
                  "refreshed a flowRateInletVelocity, which OpenFOAM recomputes at every momentum assembly "
                  "(fvMatrix.C:396 -> updateCoeffs) from the rate at that time and, for a massFlowRate, the field "
                  "named `rho` on the patch. angledDuct ships `massFlowRate constant 0.1` beside `value uniform (0 "
                  "0 0)`: brae's inlet stayed at zero, its largest velocity 1.9e-04 m/s against OpenFOAM's 0.21, "
                  "silently -- the frozen inlet this project's rules name. It was localised by the `inactive` arm: "
                  "the same 100% with the option off. The host loop refreshes it now; the device loop, which "
                  "uploads U's patches once, refuses a mass rate or a volumetric one beside a `value`. "
                  "waterChannel's volumetric inlet carries no `value`, so its constructor had built the right one. "
                  "ALSO: the shared fvOptions reader looked in system/ before constant/, where "
                  "fv::options::createIOobject (fvOptions.C:46-84) looks in constant/ first. HOST ONLY SO FAR."),
        dict(name="interFoam_mangroves", of_symbol="multiphaseMangrovesSource",
             of_file="src/waveModels/fvOptions/multiphaseMangrovesSource/multiphaseMangrovesSource.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/cfdTools/general/fvOptions/fvOptions_cpp.cu",
             validation="tests/interfoam_mangrove_vs_openfoam.sh, real OpenFOAM on laminar/waves/mangroveInteraction "
                        "as Allrun meshes it (blockMesh, setFields, topoSet) with the block halved in each "
                        "direction, 51,450 cells, the seaweed zone 14,700 of them; a Boussinesq paddle, "
                        "shallowWaterAbsorption, kEpsilon with PBiCG/DILU; 450 fixed steps of 0.01 at the case's "
                        "own tolerances. MEASURED: alpha 4.6e-14, p_rgh 3.8e-14, U 3.4e-12, k 1.7e-13, epsilon "
                        "1.0e-13, nut 3.0e-13, every p_rgh, k and epsilon count and k's and epsilon's final "
                        "residuals OpenFOAM's. CONTROLS: both options off, U 134%; the turbulence option off, k "
                        "52%. BROKEN ONCE EACH (U / k): no added mass 2.2e-01; no drag 8.0e-01; drag without rho "
                        "7.9e-01; Cm for Cm + 1 1.1e-01; Ckp and Cep swapped 9.9e-01; the turbulence sign flipped "
                        "8.3e-01; no turbulence source 3.4e-01 / 5.2e-01. NOT DISCRIMINATED: the added mass on U "
                        "rather than U.oldTime() (the same field at assembly with one outer corrector and no "
                        "predictor). tests/test_fvoptions_cpp.cu holds the reading and the refusals. NOT CLAIMED, "
                        "each refused: the density-weighted k-epsilon, any other closure under the turbulence "
                        "option, a field-name override, and the device loop.",
             note="TWO OPTIONS over each region's cellZone. multiphaseMangrovesSource: addSup(rho, eqn) adds "
                  "-Sp(rho*0.5*Cd*a*N*|U|, U) - rho*0.25*(Cm + 1)*pi*a^2*N*ddt(U), which UEqn == options "
                  "turns into diag += V*rho*drag + (rDeltaT*V)*rho*inertia and source += "
                  "((rDeltaT*U0)*V)*rho*inertia, in OpenFOAM's operation order. "
                  "multiphaseMangrovesTurbulenceModel: addSup(eqn) -- interFoam's k-epsilon is the "
                  "incompressible lineage, alpha = rho = 1 -- adds -Sp(Ckp*Cd*a*N*|U|, k) and "
                  "-Sp(Cep*Cd*a*N*|U|, epsilon), U looked up by name. Every region coefficient is readEntry "
                  "in OpenFOAM and required here. firstUnsupported() still reports both types to every "
                  "driver but interFoam's host loop, which checks its options itself: the drivers that ask "
                  "treat an implemented option they do not recognise as a porosity. HOST ONLY SO FAR."),
        dict(name="interFoam_PBiCG", of_symbol="PBiCG",
             of_file="src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/OpenFOAM/matrices/pbicg.cu",
             validation="tests/interfoam_mangrove_vs_openfoam.sh: all 450 k and 450 epsilon solves take "
                        "OpenFOAM's iteration count and end on its final residual. BROKEN ONCE EACH: "
                        "PBiCGStab in its place, 76 of 450 k counts equal and U 9.2e-05; DILU's transpose sweep "
                        "not transposed, the run fails. tests/interfoam_refusals.sh: PBiCGStab named for k is "
                        "refused.",
             note="PBiCG with the DILU preconditioner, transcribed from PBiCG.C and DILUPreconditioner.C: "
                  "the direct and TRANSPOSE systems side by side (Amul/Tmul, precondition/preconditionT), "
                  "the singularity break before the count moves. Not PBiCGStab, which brae already had and "
                  "which stops at different iterates at the same tolerance. The sweeps run in face order: "
                  "for any cell the subtractions arrive in the same face order as losortAddr's, so the bits "
                  "are the same. Wired for interFoam's kEpsilon (LinearSolverChoice::pbicgDILU); the LES and "
                  "kOmegaSST closures still refuse it. HOST ONLY SO FAR."),
        dict(name="interFoam_variableHeightFlowRate", of_symbol="variableHeightFlowRateInletVelocityFvPatchVectorField",
             of_file="src/finiteVolume/fields/fvPatchFields/derived/variableHeightFlowRateInletVelocity/"
                     "variableHeightFlowRateInletVelocityFvPatchVectorField.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fields/fv_patch_field.cuh",
             validation="tests/interfoam_weiroverflow_vs_openfoam.sh, real OpenFOAM on RAS/weirOverflow AS SHIPPED (its "
                        "own blockMesh, 5080 cells non-orthogonal to 26.4 degrees, setFields), ten fixed steps of the "
                        "tutorial's deltaT 1e-3. MEASURED: all 30 p_rgh, 10 epsilon and 10 k iteration counts "
                        "OpenFOAM's; alpha 4.0e-13, p_rgh 8.0e-13, U 1.8e-11, k 5.1e-14, epsilon 6.2e-14, nut 8.8e-14; "
                        "the inlet's velocity 2.7e-13 of its largest value, face by face against the `value` OpenFOAM "
                        "writes; bounds at about 30x. One p_rgh initial residual is 7.2e-08 out -- step one's third "
                        "corrector, itself 2.8e-05 after a PCG solve stopped at relTol 0.05, where p_rgh agrees to "
                        "8e-13 -- and that bound is 2e-06 for that stated reason. THE CONTROL: OpenFOAM with the inlet "
                        "U a fixedValue at the file's (0 0 0), 100% of U. BROKEN ONCE EACH (U): U_p without the "
                        "alpha_p weight 7.6e-01; avgU over the whole inlet area 6.0e-01; alpha_p from the face cells "
                        "4.7e-03; ddtCorr's boundary half dropped 3.6e-05 and 22 of 30 p_rgh counts; the alpha "
                        "condition never updated -- the inlet stays dry and the velocity condition refuses by name. "
                        "THE ALPHA CONDITION HAS NO CONTROL ON THE CASE: inside [0, 1] its value is the face cell's "
                        "on inflow and on outflow, and an inletOutlet of 0 in its place sends OpenFOAM itself to NaN "
                        "(the velocity condition divides by a wet area of zero). Its per-face logic, with bounds "
                        "that are not 0 and 1, is held by tests/test_variable_height_flow_rate.cu against "
                        "OpenFOAM's text, with an inletOutlet as the control. NOT CLAIMED: a flowRate that varies in "
                        "time (refused), and the device loop (refused by name).",
             note="TWO CONDITIONS. variableHeightFlowRateInletVelocity is a fixedValue whose updateCoeffs rebuilds "
                  "n*avgU*alpha_p, avgU = -flowRate/gSum(magSf*alpha_p), from the phase field's STORED patch values "
                  "clipped to [0, 1]; the driver hands them over at the momentum assembly, where the flow-rate inlet "
                  "is refreshed too. variableHeightFlowRate is a mixed condition (refGrad 0, assignable false) whose "
                  "updateCoeffs sets, where phi < -SMALL, valueFraction 1 and refValue the face cell's value "
                  "clipped at lowerBound and upperBound, and elsewhere zeroGradient; the flux reaches it through "
                  "updateFromFlux and the cells through evaluate(). WHAT THE GATE FOUND WAS IN THE PRESSURE "
                  "EQUATION. brae added fvc::ddtCorr to phiHbyA on the internal faces only; OpenFOAM adds a whole "
                  "surface field and zeroes the coupling coefficient only where U FIXES A VALUE (ddtScheme.C, "
                  "fvcDdtPhiCoeff). weirOverflow's outlet is `U zeroGradient`, the first gated open patch that "
                  "fixes nothing: phi 3.2e-04 out at the outlet after the third step -- when its flux turned "
                  "outward and the limiter's coefficient left zero -- and U 3.6e-05 after ten. Localised by where "
                  "and when (cells, stored patch values and phi all agreed to 3e-10 after two steps), and "
                  "converging the pressure solves to 1e-13 changed nothing, which ruled the stopping point out. "
                  "The boundary term reads U.oldTime()'s PATCH value, not the face cell's (fvc::dotInterpolate): "
                  "on angledDuct's tilted slip wall the cell's velocity made a correction OpenFOAM does not have, "
                  "and that gate's floor arm caught it (4e-15 to 1.4e-11). The device pressure equation has no "
                  "boundary half and its header said OpenFOAM has none; corrected, and the device loop refuses an "
                  "open patch whose U fixes no value. HOST ONLY SO FAR."),
        dict(name="interFoam_LESkEqn", of_symbol="kEqn",
             of_file="src/TurbulenceModels/turbulenceModels/LES/kEqn/kEqn.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/TurbulenceModels/turbulenceModels/LES/kEqn/les_kEqn_cpp.cu",
             validation="tests/interfoam_les_vs_openfoam.sh, real OpenFOAM on LES/nozzleFlow2D (Allrun's mesh: "
                        "blockMesh and two topoSet/refineMesh passes, 20603 cells, an axisymmetric WEDGE, "
                        "non-orthogonal to 40 degrees), 100 fixed steps of 1e-9. MEASURED: alpha 6.6e-12, p_rgh "
                        "8.2e-11, U 4.5e-12, k 2.1e-12, nut 1.1e-12; all 400 p_rgh and 100 k iteration counts and "
                        "every k final residual OpenFOAM's; the filter width 1.1e-15 before and after smoothing "
                        "against the delta OpenFOAM writes, and 8.2e-16 on the same mesh made 3-D. THE CONTROL: "
                        "OpenFOAM run laminar, 19% of U. tools/dumpKEqn is OpenFOAM's kEqn with its stages "
                        "written; at step one G, divU, the convection and diffusion coefficients, the matrix "
                        "and the solved k agree with it to 1e-12. BROKEN ONCE EACH, fourteen: every term of the "
                        "equation, the delta's smoothing, its 2-D branch, the wave's tolerance, validate(), the "
                        "limitedLinear scheme -- all red (the script tabulates U, k and the p_rgh counts). NOT "
                        "DISCRIMINATED: Ce and kMin from the LES dictionary rather than kEqnCoeffs (the case "
                        "leaves both at their defaults). NOT CLAIMED, each refused by name: every other LESModel "
                        "and LESdelta, `density variable`, k schemes other than upwind and limitedLinear, "
                        "gradients other than Gauss linear, nut wall functions under LES, a coupled patch under "
                        "the smooth delta, and the device loop.",
             note="kEqn in the uniform lineage: ddt(k) + div(phi,k) - laplacian(nut + nu, k) == nut*(grad(U) && "
                  "devTwoSymm(grad(U))) - SuSp(2/3 divU, k) - Sp(Ce*sqrt(k)/delta, k); bound; nut = "
                  "Ck*sqrt(k)*delta. The filter width is les_delta_cpp.cu: cubeRootVol (sqrt(V/thickness) on a "
                  "2-D mesh, and a wedge is 2-D to it -- calcDirections knocks the wedge normal out of "
                  "geometricD) and smooth, a FaceCellWave transcribed in OpenFOAM's visit order, since a "
                  "value is taken only when it is 1% larger and so the order is part of the answer. WHAT THE "
                  "GATE FOUND was outside the model, by running the case LAMINAR in both codes first: the host "
                  "wedge had no vector matrix coefficients (see interFoam_wedgeCoefficients); grad(U)'s wedge "
                  "value was not rotated; and fvc::gaussGrad interpolated as w*P + (1 - w)*N where OpenFOAM "
                  "writes lambda*(P - N) + N -- on the uniform k of step one, limitedLinear's limiter is "
                  "decided by the SIGN of a gradient of a uniform field, which is that ulp: k 2.3e-03 out "
                  "after one step, 3.9e-14 with OpenFOAM's arithmetic. HOST ONLY SO FAR."),
        dict(name="interFoam_wedgeCoefficients", of_symbol="wedgeFvPatchField",
             of_file="src/finiteVolume/fields/fvPatchFields/constraint/wedge/wedgeFvPatchField.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fields/fv_patch_field.cuh",
             validation="tests/interfoam_les_vs_openfoam.sh (LES/nozzleFlow2D, every cell on the two wedge "
                        "planes). BROKEN ONCE EACH (U after 100 steps): the host coefficients as the base "
                        "class's 1.1e-04; grad(U)'s wedge value unrotated 2.9e-04. tests/test_wedge_patch.cu "
                        "holds the geometry. NOT DISCRIMINATED: grad(alpha)'s wedge value rotated for the "
                        "interface normal -- provably invisible, the normal correction removes the one component "
                        "nHatf keeps. symmetryPlane and symmetry give the gradient field THEIR constraint type "
                        "the same way: fvc::gradUBoundary mirror-averages the cell gradient there, gated by "
                        "tests/interfoam_leakage_vs_openfoam.sh (HbyA 3% out beside its symmetry patches "
                        "without it; U 7.8e-02 with it broken).",
             note="The host WedgePatchField overrode evaluate() alone: a vector wedge assembled with the "
                  "zeroGradient coefficients, where transformFvPatchField (.C:95-136) has valueInternalCoeffs 1 - "
                  "d, gradientInternalCoeffs -deltaCoeffs*d, the boundary coefficients from snGrad and the patch "
                  "internal field, d = 0.5*(1 - cellT_kk), and wedgeFvPatchField's snGrad (cellT & pif - pif)*"
                  "0.5*deltaCoeffs. The device built the right coefficients from wedgeCellT() in its mixed slot, "
                  "so only the host was wrong, and no host gate had a wedge until nozzleFlow2D: UEqn.A() 2e-05 "
                  "low in every cell, 1.7e-04 in the axis corner, against OpenFOAM's dumped A(). SECOND: "
                  "fvPatchField::New gives a constraint patch its own type in a DERIVED field too, so grad(U) "
                  "has a wedge patch whose value is faceT & G & faceT^T; gaussGrad::correctBoundaryConditions "
                  "starts from that, and fvc::gradUBoundary started from the cell gradient."),
        dict(name="interFoam_cyclic", of_symbol="cyclicFvPatchField",
             of_file="src/finiteVolume/fields/fvPatchFields/constraint/cyclic/cyclicFvPatchField.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fvMesh/fv_patch.cuh",
             validation="tests/interfoam_baffle_vs_openfoam.sh, real OpenFOAM on RAS/damBreakPorousBaffle (blockMesh "
                        "then createBaffles: 13 internal faces become the cyclic pair porous_half0/1; 2268 cells; "
                        "kEpsilon, MULESCorr, three outer correctors, DICPCG and symGaussSeidel, linearUpwind "
                        "grad(U)), fixed steps of 1e-3. Profiles `cyclic` (p_rgh a plain cyclic, 20 steps), "
                        "`cyclicWet` (the water column staged ACROSS the baffle, 60 steps) and `cyclicWetExplicit` "
                        "(MULESCorr off). MEASURED: cyclic alpha 6.9e-15, p_rgh 6.6e-15, U 5.0e-14, k 1.8e-14, "
                        "epsilon 4.6e-14, nut 1.1e-14, every p_rgh, alpha, epsilon and k iteration count "
                        "OpenFOAM's; cyclicWet alpha 2.4e-13, p_rgh 5.1e-13, U 3.2e-12; cyclicWetExplicit alpha "
                        "3.5e-14, U 4.0e-13. THE CONTROL: OpenFOAM with the pair as two WALLS -- what the shared "
                        "factory's zeroGradient placeholder would have run -- 34% of U. THE WET PROFILES' p_rgh "
                        "solves are pinned at 1e-13 in both codes: at the case's 1e-07 the wet pair's first system "
                        "moves 4.0e-05 under ONE reordered dot product in brae's own PCG (1e-08 to 1e-10 on the "
                        "other systems), so that tolerance measures two Krylov paths' stopping points; counts are "
                        "then held to one apart at an edge stop (15 of 180). BROKEN ONCE EACH, 20 breaks of the "
                        "coupling, every one red on a profile (the script tabulates U per profile): from the "
                        "laplacian's interface coefficient dropped, 4.0e-01, down to DkEff on the pair taken from "
                        "patch values, 1.9e-10; ten of them change NO digit of the two shipped profiles, where "
                        "only air reaches the baffle, which is why the wet ones exist. NOT DISCRIMINATED: "
                        "fvMatrix::relax's coupled branch, the neighbour cell in the EXPLICIT limiter's extrema, "
                        "a vector gradient read from the stored cyclic value, CorrectPhi across the pair (phi "
                        "starts at 0). NOT CLAIMED, each refused by name (tests/interfoam_refusals.sh): across a "
                        "cyclic, GAMG and PBiCGStab, a momentum predictor, div(rhoPhi,U) schemes other than "
                        "upwind and linearUpwind, interfaceCompression, leastSquares and cellLimited gradients, a "
                        "moving mesh, MRF, fvOptions, waves, kOmegaSST, a rotational or non-orthogonal pair; "
                        "cyclicAMI, cyclicACMI, processor; the device loop, which refuses any cyclic.",
             note="brae's host interFoam had NO coupled patch: the shared factory builds a cyclic as a "
                  "zeroGradient placeholder (its coupling belongs to the legacy drivers' CyclicInterface), so the "
                  "pair would have run as two walls, stopped only by CorrectPhi refusing for its own reason. The "
                  "coupling is now OPT-IN: attachCyclicCoupling() fills FvPatch::coupled, nbrFaceCells, weights, "
                  "delta, the coupled deltaCoeffs (1/|delta|), nonOrthDeltaCoeffs and correction vectors "
                  "(cyclicFvPatch.C, basicFvGeometryScheme.C); only interFoam's host path calls it, so no other "
                  "driver's operators change. WHAT BRANCHES ON IT: fvc (a coupled face is interpolated from its "
                  "two CELLS, never a stored patch value -- HbyA's cyclic patch holds interpolate(rAU)*"
                  "interpolate(H) and fvc::flux does not read it; snGrad with the scheme's deltaCoeffs and the "
                  "correction); fvm (laplacian internalCoeffs = boundaryCoeffs = -gamma*magSf*dc, upwind div "
                  "phi*w and -phi*(1-w), linearUpwind's coupled correction with the NEIGHBOUR's gradient); "
                  "fvMatrix A, H, flux, relax; PCG/DIC and smoothSolver (diagonal folded, no source, Amul and "
                  "residual through the interface, sumA, Gauss-Seidel's Jacobi update of bPrime per sweep); MULES "
                  "and CMULES (neighbour cell in the extrema, per-cell limiter, and syncFaceList's minimum "
                  "across the pair -- NOT a no-op in serial, as a note in mules_cpp.cu said); interFoam (phic "
                  "left alone on a coupled face, nHatf from the two cells' gradients, ddtCorr, phig and the "
                  "reconstruction with the coupled rAUf, pcorr a plain cyclic). ALSO FOUND: `Gauss "
                  "interfaceCompression vanLeer 1`, interfaceCompression.H's limited scheme, was read as plain "
                  "vanLeer by a substring match and ran; refused. HOST ONLY SO FAR."),
        dict(name="interFoam_cyclicACMI", of_symbol="cyclicACMIPolyPatch",
             of_file="src/meshTools/AMIInterpolation/patches/cyclicACMI/cyclicACMIPolyPatch/cyclicACMIPolyPatch.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fvMesh/fvPatches/constraint/cyclicACMI/cyclic_acmi_cpp.cu",
             validation="tests/interfoam_leakage_vs_openfoam.sh, real OpenFOAM on RAS/damBreakLeakage as its "
                        "Allrun meshes it (2268 cells; a createBaffles cyclicACMI pair of 13 faces whose "
                        "non-overlap patches are symmetry planes; a coded per-face scale that opens two faces at "
                        "t > 0.5), every solve pinned at 1e-13. MEASURED, 520 steps from t = 0 with the baffle "
                        "opening at step 500: alpha 3.4e-13, p_rgh 1.5e-13, U 4.9e-12, k 9.2e-13, epsilon "
                        "2.2e-13, nut 3.5e-13, the flux through the four baffle patches 3.2e-14 of its largest; "
                        "and 30 steps restarted from OpenFOAM's own state at t = 0.49, U 5.2e-12. CONTROLS: the "
                        "baffle never opened, U 100%; opened on every face, 179%. BROKEN ONCE EACH (U, leak / "
                        "restart): MULES synced across the pair 1.7e-01; phic formed after the rescale 1.7e-09; "
                        "no symmetry reflection of grad(U) 7.8e-02; the clock at 0 green / 1.0; the neighbour "
                        "unscaled 2.7; the non-overlap patch at full area 7.0e-01; never rescaled 1.0; the "
                        "patches' |Sf| not reset 1.8. NOT DISCRIMINATED: recomputing the face cells' volumes "
                        "after the rescale (bitwise the same here). NOT CLAIMED, each refused: a pair not "
                        "coincident face for face, a moving mesh, the explicit MULES path, icAlpha, scAlpha and "
                        "alpha sub-cycling under a moving scale, and the device loop.",
             note="A COINCIDENT pair (createBaffles) gives each face one AMI partner, its twin, with weight 1 "
                  "(measured: all 13 faces `1(i)`, weight `1(1)`), so the coupling is the cyclic's "
                  "(coupleTranslationalPair) on areas the mask splits: coupled raw*max(tol, mask), "
                  "non-overlap raw*(1 - min(max(mask, tol), 1 - tol)), mask = min(1 - tol, max(tol, "
                  "scale*1)), tol 1e-10, the neighbour's scale a clone of the owner's evaluated on its own "
                  "faces. The rescale runs once per time index, lazily in OpenFOAM, and WHERE is part of "
                  "the answer: after alphaEqn.H forms phic on the old areas and before the pre-solve "
                  "(alphaEqnStep's geometryUpdate). Found on the way: MULES does not sync its limiter across "
                  "an AMI pair (interFoam_MULES); grad(U) on a symmetry patch is mirror-averaged "
                  "(interFoam_wedgeCoefficients); the host loop's clock started at 0 on every restart. "
                  "buildPatches' ACMI refusal stays for every other driver; the interFoam host opts out "
                  "with mirrorACMI. HOST ONLY SO FAR."),
        dict(name="interFoam_cyclicAMI", of_symbol="cyclicAMIFvPatch",
             of_file="src/finiteVolume/fvMesh/fvPatches/constraint/cyclicAMI/cyclicAMIFvPatch.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fvMesh/fvPatches/constraint/cyclicAMI/cyclic_ami_cpp.cu",
             validation="tests/interfoam_ami_vs_openfoam.sh, real OpenFOAM on RAS/mixerVesselAMI as Allrun.pre "
                        "meshes it with the background block (50 50 100) coarsened to (22 22 44): 82,510 cells, "
                        "8,872 faces a side, the rotor's cellZone turned by solidBodyMotionSolver at omega -5; "
                        "kEpsilon, explicit MULES in two sub-cycles, momentum predictor, two outer correctors, "
                        "correctPhi, grad(U) cellLimited, limited corrected 0.33, a rotatingWallVelocity shaft. "
                        "STAGED IN BOTH CODES: p_rgh and pcorr PCG/DIC for GAMG (not ported across an AMI), every "
                        "solve pinned at 1e-13, fixed steps of 2e-4. MEASURED, 100 steps (0.1 rad): alpha 4.4e-13, "
                        "p_rgh 5.4e-14, U 3.5e-12, k 9.5e-14, epsilon 1.2e-13, nut 2.1e-12, the flux across the "
                        "pair 1.3e-11 of its largest, the moved points bitwise; all 200 p_rgh, 600 U, 100 k, 100 "
                        "epsilon and 101 pcorr counts OpenFOAM's. At the case's own tolerances U 3.9e-07 and 14 of "
                        "200 p_rgh counts apart -- where two Krylov solvers stopped. CONTROL: the rotor held still, "
                        "U 1.1e-01. BROKEN ONCE EACH (U after 100 steps): the AMI not "
                        "recomputed after the move, refused by name; delta without the neighbour's interpolated half "
                        "9.7e-03; no non-orthogonal correction on coupled faces 4.0e-02; ddtCorr live on the "
                        "cyclicAMI 8.7e-03; MULES synced across it 1.4e-04; cellLimited's coupled range from the "
                        "stored value 1.3e-03; grad(U) unlimited 2.7e-01; rotatingWallVelocity's omega flipped "
                        "1.2e-03. NOT DISCRIMINATED: the segregated solve's coupled source added and taken back "
                        "(round-off), kEpsilon's V0 and absolute divU under a rigid rotation. NOT CLAIMED, refused by name: "
                        "GAMG across the pair, a transform, a periodic AMI, lowWeightCorrection, requireMatch "
                        "false, any AMI keyword not read, a momentum predictor or a cellLimited grad(U) across a "
                        "cyclic or cyclicACMI (ungated there), and the device loop.",
             note="THE COUPLING IS A STENCIL: FvPatch carries amiOffsets/amiNbrFaces/amiNbrCells/amiWeights, and "
                  "every operator reads the neighbour through patchNeighbourValue -- the cell across a cyclic, the "
                  "AMI's weighted sum (weightedSum, multiplyWeightedOp: zero, then += w*value slot by slot) across "
                  "a cyclicAMI; nbrFaceCells is empty there. The owner side reads the AMI's src addressing, the "
                  "other its tgt addressing (interpolateUntransformed). Geometry, cyclicAMIFvPatch.C: w = "
                  "|dn|/(|d| + |dn|) with dn the AMI interpolate of the neighbour's nf & (Cf - Cn), delta = (Cf - "
                  "Cn) - interpolate(neighbour's Cf - Cn), and basicFvGeometryScheme's coupled deltaCoeffs, "
                  "nonOrthDeltaCoeffs and correction vectors from it. After every mesh move the AMI is recomputed "
                  "and both patches coupled again (initMovePoints marks it stale; the next AMI() resets it). "
                  "PORTED WITH IT, each found by the gate: the laplacian's non-orthogonal correction on coupled "
                  "faces (fvm::laplacianCorrFluxCoupled, into the source and the pressure flux); "
                  "fvMatrix::solveSegregated's coupled source (the vector bc*pnf added, each component's "
                  "interface taken back); cellLimitedGrad's coupled range from patchNeighbourField; ddtCorr's "
                  "coefficient zeroed on cyclicAMI; MULES left unsynced across it (syncTools does not reach an "
                  "AMI). HOST ONLY SO FAR."),
        dict(name="interFoam_faceAreaWeightAMI", of_symbol="faceAreaWeightAMI",
             of_file="src/meshTools/AMIInterpolation/AMIInterpolation/faceAreaWeightAMI/faceAreaWeightAMI.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/meshTools/AMIInterpolation/faceAreaWeightAMI/face_area_weight_ami_cpp.cu",
             validation="tests/test_face_area_weight_ami.cu, run by tests/interfoam_ami_vs_openfoam.sh against the "
                        "srcAddress/srcWeights/srcWeightsSum and tgt* OpenFOAM's own cyclicAMI holds after each of "
                        "three steps of 2e-4 (a coded function object prints them): every partner set of all 8,872 "
                        "faces a side OpenFOAM's, weights and sums 2.5e-14 and 4.3e-14, from OpenFOAM's points and "
                        "from brae's own motion alike. CONTROL: OpenFOAM leaves 43 to 47 source faces a step less "
                        "than 99% covered, which the test asserts -- an all-pairs search covers them in full, and "
                        "BROKEN THAT WAY 47 source and 49 target partner sets differ, weight sums 3.8e-02 apart "
                        "(and in the interFoam gate U 6.0e-05 after 100 steps). Also red: the weights normalised "
                        "by the face area rather than their sum, 1.0e-01, and the interFoam run non-finite by "
                        "step 6.",
             note="THE WALK IS THE ANSWER, not an implementation detail: OpenFOAM reaches a target face only "
                  "through the neighbours of a face it has already kept, so a partner can be missed -- one step "
                  "in, face 2354 gets two partners covering 96.4% of it and its weights are normalised by that. "
                  "Transcribed: the walk from source 0 against target 0, PrimitivePatch's faceFaces order, the "
                  "89-degree neighbour filter, LIFO processing, the seeds setNextFaces leaves (first overlap "
                  "chosen, last overlap stored), the octree's nearest face found by brute force over "
                  "face::nearestPoint's distance, restartUncoveredSourceFace below 0.95; triangles by face::split, "
                  "areas by faceAreaIntersect (tri sliced by each target edge plane, target vertices 2, 1, 0), "
                  "area normalisation `project`. HOST_ONLY: recomputed once per mesh move from the points, a "
                  "geometric preprocessing step like the mesh motion itself (0.1 s at 8,872 faces a side)."),
        dict(name="interFoam_codedPatchFunction1", of_symbol="CodedField",
             of_file="src/meshTools/PatchFunction1/CodedField/CodedField.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/meshTools/PatchFunction1/CodedField/codedPatchFunction1.cu",
             validation="tests/interfoam_leakage_vs_openfoam.sh (the scale that opens damBreakLeakage's baffle: "
                        "the masks at the end are open on exactly the faces the code names, on both sides, and "
                        "the neighbour's clone left unevaluated puts U 2.7 out); tests/interfoam_refusals.sh "
                        "(codeInclude refused, a name the shim lacks -- this->time().timeIndex() -- refused "
                        "with the compiler's output).",
             note="`code` pasted as the body of value(x) in a class that gives this->patch() and "
                  "this->time(), as codedPatchFunction1Template does, compiled with g++ against a SHIM "
                  "of OpenFOAM's scope: the coded Function1's scalar one plus Vector, Field, tmp, Zero, "
                  "forAll, the field-vector inner product, a polyPatch with name/size/faceCentres and a "
                  "Time with value/timeOutputValue. Not timeIndex(): OpenFOAM continues it across a "
                  "restart from <start>/uniform/time, which brae does not read. The compile-and-load half "
                  "is shared with the coded Function1 (coded_library). HOST_ONLY: evaluated once per step "
                  "on the host, where the mask is formed."),
        dict(name="interFoam_porousBafflePressure", of_symbol="porousBafflePressureFvPatchField",
             of_file="src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/porousBafflePressure/"
                     "porousBafflePressureFvPatchField.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fields/fv_patch_field.cuh",
             validation="tests/interfoam_baffle_vs_openfoam.sh, profiles `porous` (RAS/damBreakPorousBaffle as "
                        "createBaffles leaves it, 20 steps) and `porousWet` (the free surface staged across the "
                        "baffle, 60 steps, p_rgh solves pinned at 1e-13). MEASURED: porous alpha 7.8e-15, p_rgh "
                        "8.0e-15, U 9.9e-13, the jump 1.6e-12 of the one OpenFOAM writes, the pair's own p_rgh "
                        "2.8e-13 on all 26 faces, every iteration count OpenFOAM's; porousWet alpha 6.1e-14, "
                        "p_rgh 3.0e-13, U 1.1e-12, the jump -- NON-uniform there, rho_p varies along the baffle "
                        "-- 5.4e-12. THE CONTROL: OpenFOAM's plain-cyclic answer, 2.0% and 4.5% of U. BROKEN ONCE "
                        "EACH (U porous, porousWet): the jump left out of the solve's first Amul 2.2e-02, "
                        "1.9e-02; out of the flux 2.8e+03, 4.1e-01; not negated on the other side 1.2e-01, "
                        "8.7e-02; built from phiHbyA 1.4e-03, 2.4e-03; uniformJump ignored 5.9e-02, 6.7e-02; "
                        "without rho_p NOTHING, 4.5e-02 (air alone reaches the shipped baffle, rho_p = 1); nu_p "
                        "= nu(alpha_p) NOTHING, 1.05e-09. tests/test_porous_baffle_pressure.cu holds the "
                        "per-face jump, reversed flow and sign(0) against OpenFOAM's text. NOT CLAIMED, refused "
                        "by name: `relax`, `minJump`, a D or I that is not `constant`, a mass flux `phi`, a "
                        "`jump` that is neither uniform nor a list; the device loop.",
             note="A fixedJump cyclic. Un = phi_p/magSf (gAverage under `uniformJump`); jump = -sign(Un)*(D*nu_p "
                  "+ I*0.5*|Un|)*|Un|*length, times rho_p because p_rgh has the dimensions of pressure. Only the "
                  "OWNER side computes it (fixedJump::setJump is a no-op on the other, whose jump() returns the "
                  "owner's), at the pressure ASSEMBLY -- fvMatrix's constructor runs updateCoeffs -- from phi as "
                  "the last corrector left it. It enters the solve through the interface update and ONLY when "
                  "the operand is the field itself (jumpCyclicFvPatchFields.C, `only apply jump to original "
                  "field`): in PCG the one Amul that builds the initial residual and the normalisation; and the "
                  "flux through patchNeighbourField. WHAT THE GATE FOUND: porousWet at U 1.05e-09 where "
                  "cyclicWet held 3e-12, and brae against itself under a round-off perturbation at 1e-13, so a "
                  "difference and not sensitivity. The jump agreed to 5e-13 after 5 steps and drifted as the "
                  "interface smeared over the baffle's faces. Refitting OpenFOAM's OWN written jump from its "
                  "own written alpha was exact to 1e-15 except on the face whose two cells held alpha 0.679 "
                  "and 0.520: 2.8e-08 out with nu_p = nu(alpha_p), 5e-16 with the two cells' nu interpolated. "
                  "CAUSE: v2412's `localConsistency` (default on, etc/controlDict:225) -- every GeometricField "
                  "operation ends in correctLocalBoundaryConditions() (GeometricFieldFunctionsM.C), which "
                  "re-evaluates a constraint patch of the RESULT: coupledFvPatchField::evaluateLocal is "
                  "evaluate(). A derived field's cyclic patch value is the interpolation of its own CELLS, not "
                  "the operation applied to the operands' patch values. Same number for rho, linear in alpha; "
                  "not for nu. The mixture's coupled patch values now come from the cells. HOST ONLY SO FAR."),
        dict(name="interFoam_permeableWall", of_symbol="prghPermeableAlphaTotalPressureFvPatchScalarField",
             of_file="src/finiteVolume/fields/fvPatchFields/derived/prghPermeableAlphaTotalPressure/"
                     "prghPermeableAlphaTotalPressureFvPatchScalarField.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/fields/fv_patch_field.cuh",
             validation="tests/interfoam_permeable_vs_openfoam.sh, real OpenFOAM on laminar/damBreakPermeable (kEpsilon "
                        "despite the directory; 2268 cells), fixed steps of 1e-3, two profiles. `shipped`, 20 steps, "
                        "the case as it is: the wall is dry throughout. `wetWall`, 140 steps, the setFields box "
                        "mirrored against the wall: 27 of its 50 faces wet at the start, 25 at the end. MEASURED: "
                        "shipped alpha 8.5e-15, p_rgh 8.0e-15, U 2.7e-15, k 2.6e-15, epsilon 6.6e-15, nut 3.7e-15; "
                        "wetWall alpha 4.0e-13, p_rgh 5.0e-14, U 1.4e-14, all 420 p_rgh counts OpenFOAM's. The "
                        "wall's U and p_rgh face by face against the values OpenFOAM writes (shipped: 40 moving "
                        "faces, 5.3e-16; wetWall: OpenFOAM writes uniform 0 and brae is held to exactly 0). THE "
                        "CONTROL: OpenFOAM with the wall a noSlip and fixedFluxPressure, 99% of U on wetWall. BROKEN "
                        "ONCE EACH (U shipped, U wetWall): hydrostatic term out 1.0, 1.0; pressure open on every "
                        "face nothing, 17; velocity ignoring alpha nothing, 1.3e-01; threshold 0.5 for alphaMin "
                        "0.01 nothing, 2.2e-02; U's patches updated in every corrector 8.6e-08, 3.4e-06; the "
                        "dynamic term out 8.6e-08, 3.4e-06; the lag applied to pressureInletOutletVelocity too "
                        "nothing, 3.9e-02. shipped cannot see four of the seven, which is why wetWall exists. "
                        "tests/test_permeable_wall_conditions.cu holds the per-face logic against OpenFOAM's text, "
                        "the `alpha none` half and strict pos included, with an inletOutlet as the control. NOT "
                        "ASSERTED: epsilon's solver RESIDUALS (counts are) -- 1.8e-06 per step apart with the "
                        "field at 7e-15, and 0.0 apart with the wall closed: brae's epsilonWallFunction is a "
                        "zeroGradient whose rows setValues writes, OpenFOAM's derives from fixedValue, and on a "
                        "wall that carries flux the two normalise differently. NOT DISCRIMINATED: whether the "
                        "phase fraction reaches the patches before or after alpha's own boundary evaluation. NOT "
                        "CLAIMED: wetWall beyond 140 steps; a mass flux `phi`, a `p` that is not uniform, an "
                        "`alpha` naming another field (each refused by name, tests/interfoam_refusals.sh); the "
                        "device loop (refused by name).",
             note="TWO CONDITIONS on one wall. permeableAlphaPressureInletOutletVelocity "
                  "(pressurePermeableAlphaInletOutletVelocityFvPatchVectorField.C:127-178) is a mixed condition: "
                  "refValue (phi/magSf)*n, valueFraction neg(phi), and with an alpha valueFraction = "
                  "max(pos(alpha_p - alphaMin), valueFraction) with refValue 0 where that is 1 -- closed where "
                  "wet, closed to inflow, zeroGradient where dry and leaving. prghPermeableAlphaTotalPressure "
                  "(.C:151-243) is driven by constrainPressure like fixedFluxPressure: refValue p0 - "
                  "0.5*rho*neg(phi)*|U_b|^2 - rho*gh, refGrad snGradp, valueFraction 1 - pos(alpha_p - alphaMin). "
                  "pos is STRICT in v2412. WHAT THE GATE FOUND: fvMatrix's constructor runs "
                  "U.boundaryFieldRef().updateCoeffs() at the momentum assembly, which sets each patch's updated_ "
                  "flag; with momentumPredictor off nothing evaluates U until the FIRST corrector's "
                  "U.correctBoundaryConditions(), where mixed::evaluate sees the flag and skips updateCoeffs -- so "
                  "that one evaluation uses the ASSEMBLY-time valueFraction, not the new flux's. brae pushed the "
                  "new flux to every U patch in every corrector: U 8.6e-08 on shipped, 3.4e-06 on wetWall. Its "
                  "whole effect runs through the dynamic term -- COUNTED: the term is non-zero 51 times in shipped "
                  "and 23 in wetWall, and never with the lag removed, since an updated patch holds U_b = 0 "
                  "wherever phi < 0. PER CLASS: a condition whose updateCoeffs ends in evaluate() clears the flag "
                  "itself (pressureInletOutletVelocity; also fixedNormalInletOutletVelocity and "
                  "fluxCorrectedVelocity, not ported) and is exempt -- applying the lag to it broke wetWall at "
                  "3.9e-02. WHY 140 STEPS: at step 142 a wet face goes dry, where phi was 0 or one ulp of "
                  "round-off; neg(phi) then reads the SIGN of that round-off (brae -3.4e-21), and UEqn.A() in cell "
                  "1792 is 2.018795e+04 in OpenFOAM against 2.019174e+04. The condition is ill-conditioned there "
                  "in OpenFOAM itself. operator=(pvf) is not ported: interFoam's one assignment to U is followed "
                  "at once by correctBoundaryConditions (pEqn.H:58-59). HOST ONLY SO FAR."),
        dict(name="interFoam_waveModel", of_symbol="waveModel",
             of_file="src/waveModels/waveModel/waveModel.C",
             classification="BOUNDARY_CONDITION", status="REIMPLEMENT",
             brae_reference="src/waveModels/waveModel/wave_model_cpp.cuh",
             brae_target="",
             validation="tests/interfoam_waves_vs_openfoam.sh, real OpenFOAM run SERIALLY, fixed steps, on ALL NINE "
                        "laminar/waves tutorials that carry a wave inlet -- one per generation model -- each on a "
                        "coarsened mesh at full amplitude and against its own no-wave control (the generating patch given "
                        "the absorbing model). MEASURED, worst of the eight beyond stokesI: alpha 1.6e-10, p_rgh 1.6e-10, "
                        "U 1.4e-08 (McCowan, the one with active absorption at a solitary inlet; the other seven are at "
                        "or under 4.9e-11, 5.6e-11 and 3.8e-09), every p_rgh iteration count OpenFOAM's (40 or 60 of them "
                        "per case). WHAT THE MODEL ITSELF COMPUTES has two direct oracles. Every constant in the block "
                        "OpenFOAM prints when it creates the model is compared label for label -- wave length, StokesV's "
                        "Lambda, cnoidal's m, a solitary wave's x0 -- and agrees to the log's 15 digits (worst 2.3e-15). "
                        "And the inlet's velocity is compared face by face: StokesV 1.2e-13, cnoidal 4.5e-14, "
                        "irregularMultiDirectional 1.1e-15, Grimshaw 6.9e-17. A 1 PER CENT ERROR IN StokesV's SMALLEST "
                        "COEFFICIENT, the fifth-order A55, moves that arm to 3.9e-07 and the fields to 2.2e-06; in B55, "
                        "alpha 3.2e-05. Three of the nine are 3-D with slip side walls (Grimshaw, McCowan, "
                        "irregularMultiDirection), which is the first time this solver's host path has been held against "
                        "OpenFOAM on a 3-D mesh. ON stokesI, where the conditions themselves were ported, FIVE PROFILES: "
                        "`shipped` (the tutorial's own 500 x 75 mesh and waveProperties, ten steps of its deltaT 0.01), "
                        "`trough` and `crest` (100 x 75, rampTime 0.05 so the wave is at full height from step five; the "
                        "default phase puts the level BELOW the reference depth with the inlet flowing OUT, wavePhase "
                        "pi/2 puts it ABOVE with inflow -- the two halves of waveModel::setPaddlePropeties), `tight` "
                        "(trough with both codes' p_rgh solves at 1e-13) and `mompred` (trough with momentumPredictor "
                        "yes). BESIDE THE FIELDS it compares four things only these conditions have an oracle for: the "
                        "model's derived constants against the log's `Reference water depth` and `Wave length` "
                        "(0.600000000000001 and 6.95175828555869, both patches, to the log's 15 digits); the ORDER of the "
                        "updates against the log's `Updating <model> wave model for patch <p>` lines, update for update "
                        "(100 of 100: per step three for the inlet, a fourth in UEqn, then the outlet); and the patch "
                        "VALUES waveAlpha and waveVelocity last assigned, face by face, from the time directory (alpha "
                        "4.4e-16). MEASURED at the case's own tolerances: alpha 1.4e-10, p_rgh 6.2e-10, U 1.8e-08 at full "
                        "amplitude and 1.2e-07 on `shipped` (|U| 4.7e-03 there), every p_rgh iteration count OpenFOAM's; "
                        "under `tight` alpha 7.7e-12, p_rgh 7.7e-12, U 6.3e-10 -- p_rgh's is alpha's times rho*g*h. "
                        "CONTROL ON THE ORACLE: OpenFOAM's own tank with waveHeight 0 is 100% of U and 0.73 of alpha "
                        "away. BROKEN ONCE EACH, with switches that are not in the tree: every sub-cycle evaluated at the "
                        "step's time 1.0e-04 of alpha; no re-update in UEqn 9.5e-03 and the order arm fails; the active "
                        "absorption left out 2.1e-01. THE NUMBERS ABOVE WERE MEASURED WITH ONE STAGING CHANGE THAT IS NOT "
                        "ABOUT WAVES, since removed (interFoam_GAMG): the tutorial solves p_rghFinal with GAMG, which "
                        "brae's interFoam then substituted under a notice, so the staging named PCG with DIC. AS "
                        "SHIPPED, on both loops, the fourteen profiles now read alpha 2.9e-11, p_rgh 5.2e-11 and U "
                        "9.3e-09 at worst. WHAT THAT SUBSTITUTION WAS WORTH WAS MEASURED: OpenFOAM against "
                        "ITSELF on `trough`, with nothing changed but p_rghFinal's solver (GAMG or PCG, the same DIC, the "
                        "same 1e-7), differs by 1.8e-02 of alpha and 9.6% of U after twenty steps; brae with its "
                        "substitute is 2.3e-02 and 19% from OpenFOAM-with-GAMG. The tank's active absorption feeds the "
                        "water level back into the velocity, so where the last pressure solve stops is part of the answer "
                        "at that tolerance, and agreeing with these tutorials AS SHIPPED needs OpenFOAM's GAMG itself. "
                        "tests/interfoam_refusals.sh holds fifteen wave and named-flux arms. WHAT IT DOES NOT CLAIM: "
                        "nPaddle above 1 and a non-zero waveAngle (every tutorial names one paddle and angle 0, the 3-D "
                        "ones included; only irregularMultiDirectional's per-component directions put a wave at an "
                        "angle), a patch that is not axis-aligned (the local frame's tensor is diagonal here, so "
                        "OpenFOAM's pairing of Rgl and Rlg cannot be told from its transpose), a restart, and two timings "
                        "no fixture could see -- below. THE DEVICE LOOP RUNS EVERY PROFILE TOO, at the host's bounds: "
                        "worst alpha 1.3e-10, p_rgh 6.3e-10, U 1.8e-08, every iteration count and every model update in "
                        "OpenFOAM's order, the two solitary cases with their `phi rhoPhi;` top included. A SIXTH stokesI "
                        "PROFILE, `mulescorr`, exists for the update's second call site: under MULESCorr alpha's first "
                        "updateCoeffs of a sub-cycle is the pre-solve's matrix construction, and updating after the pre- "
                        "solve instead reads alpha 8.9e-07 from OpenFOAM against 1.1e-10 -- so THAT position is gated, on "
                        "both paths, with all 60 alpha sweep counts OpenFOAM's.",
             note="waveAlpha AND waveVelocity ARE fixedValue PATCHES WHOSE VALUE A MODEL SUPPLIES, one model per "
                  "patch shared by both (waveModel::lookupOrCreate), updated ONCE PER TIME INDEX (waveModel.C:354). "
                  "The model reads the solver back -- alpha's patchInternalField for the active absorption's water "
                  "level, and in shallowWaterAbsorption U's -- so WHEN it updates is part of the answer, and the "
                  "time index is not the time step: Time::subCycle sets it to (n-1)*nSubCycles and ++ moves it, so "
                  "every alpha sub-cycle is a new index at the SUB-CYCLE's time, and after endSubCycle the index is "
                  "n again, which is a different number, so U's first updateCoeffs of the step (UEqn's fvMatrix "
                  "constructor) updates the model a fourth time from the alpha the sub-cycles left. OpenFOAM's log "
                  "prints a line per update, which made the order an oracle rather than an argument. The models are "
                  "transcribed in src/waveModels (StokesI, whose wave length is 100 fixed-point passes of the "
                  "dispersion relation with no tolerance, and shallowWaterAbsorption, which zeroes the GLOBAL x and "
                  "y of U's patchInternalField and then rotates the result as though it were local). Nine tutorials "
                  "name waveAlpha; each names a DIFFERENT generation model (StokesI, StokesII, StokesV, cnoidal, "
                  "Boussinesq, Grimshaw, McCowan, streamFunction, irregularMultiDirectional) over the same solver "
                  "settings. ALL TEN MODELS ARE PORTED, one file each under src/waveModels as OpenFOAM lays them "
                  "out, on OpenFOAM's own class hierarchy (regularWaveModel IS an irregularWaveModel there, which "
                  "is where the ramp lives). FOUR THINGS IN THEM LOOK LIKE SLIPS AND ARE TRANSCRIBED, because the "
                  "oracle is OpenFOAM: StokesV solves the fifth-order dispersion relation for the wave number AND "
                  "lambda, keeps lambda and DROPS the wave number, so its wave length stays StokesI's linear one; a "
                  "solitary wave's x0 is the patch's minimum GLOBAL x, taken in the constructor while the wave "
                  "angle is still 0, and subtracted from the paddle's LOCAL x; cnoidal finds its m by a scan in "
                  "steps of 1e-4 (so m is a four-digit number) and re-integrates the wave's mean square, 1000 "
                  "terms, for every face at every update; McCowan's two Newton solves stop at an ABSOLUTE 1e-5. "
                  "StokesV's twenty-two coefficient functions -- rational functions of sinh and cosh with integer "
                  "constants up to 324000 -- were produced from OpenFOAM's source text by a rewrite of its function "
                  "calls rather than retyped, since the only thing that can go wrong in them is a digit. THE SHARED "
                  "PATCH-FIELD FACTORY STILL REFUSES BOTH TYPE NAMES, on purpose: buildInterFields rewrites them to "
                  "fixedValue on its own copy of the file data, so no other solver can build a wave inlet frozen at "
                  "the file's `value`. THE FIXTURES FOUND THREE DEFECTS THAT ARE NOT ABOUT WAVES. The first two are "
                  "invisible on a wall that does not move, which is every fixedFluxPressure patch interFoam had "
                  "met. (1) constrainPressure's gradient is (phiHbyA_b - (Sf_b & U_b))/(magSf_b*rAUf_b) "
                  "(constrainPressure.C:62-72) and brae subtracted the stored phi_b. With phi_b the corrector hands "
                  "back exactly the flux it was given, so an inlet whose flux starts at zero never opens: with the "
                  "wave model's own patch values exact to 4e-16, U was 260% out and OpenFOAM's answer with the wave "
                  "switched OFF was closer to OpenFOAM's (64%) than brae was. Fixed on the host and in the device "
                  "hook: 260% -> 1.8e-08. (2) The momentum predictor's explicit fvc::snGrad(p_rgh) reads, on a "
                  "fixedFluxPressure patch, the gradient the LAST constrainPressure stored; brae's drivers zeroed "
                  "it before every predictor. Zero is OpenFOAM's construction value and right before the first "
                  "pressure equation only. Measured on `mompred`: alpha 5.1e-04 and U 5.4% -> 1.4e-11 and 1.0e-09; "
                  "Ux and Uz take OpenFOAM's sweep counts, 20 of 20 each. (3) A FLUX-CONDITIONAL CONDITION LOOKS "
                  "ITS FLUX UP BY NAME -- the `phi` entry, default phi -- and brae's field reader kept no such "
                  "entry: every condition was told phi. solitaryGrimshaw, solitaryMcCowan and mangroveInteraction "
                  "write `phi rhoPhi;` on their totalPressure top. rhoPhi is the ALPHA step's flux, built from the "
                  "phi the time step started on, so it is one pressure corrector behind and exactly zero at step "
                  "one of a case at rest, and the top's inflow test decides differently. Both solitary cases were "
                  "wrong from STEP ONE's p_rgh residual with their wave models exact to 7e-17: U 2.4e-05 and "
                  "3.0e-05 out after thirty steps, 8.0e-10 and 1.4e-08 with the named flux. It was confirmed before "
                  "it was fixed, by running OpenFOAM ITSELF with `phi phi;`: brae agreed with that run to 9.6e-10. "
                  "The reader now keeps the entry, every patch field carries its flux name, the host hands phi or "
                  "rhoPhi accordingly and refuses any other name; the turbulence closure, which hands over phi "
                  "only, refuses a k or epsilon patch that names anything else. Other solvers still ignore the "
                  "entry -- they carry one flux -- and that is recorded here rather than claimed. TWO TIMINGS ARE "
                  "TRANSCRIBED AND NOT GATED, and the header says so. The update opens MULES::explicitSolve "
                  "(MULESTemplates.C:168), so it sits between the high-order flux, built on the value the last "
                  "update left, and the limiter; moving it to AFTER the solve changed no digit on any fixture up to "
                  "the tutorial's own mesh at Co 1.9, because the explicit limiter never reduces lambda on an "
                  "uncoupled boundary face (MULESTemplates.C:533-564 treats wedge and coupled patches only) and the "
                  "new value reaches only the adjacent cell's extrema. And the outlet model is CREATED at U's first "
                  "updateCoeffs, so a case naming no reference depth takes it from alpha AFTER the first step's "
                  "advance; a case that starts from rest has the same alpha there on both sides of that step. A "
                  "FIXTURE THAT WAS DROPPED: ten steps at the tutorial's maxDeltaT 0.05 on its own mesh agreed "
                  "(alpha 1.2e-09), but at Co 1.9 OpenFOAM's own STILL tank carries half the velocity of the one "
                  "with the wave, so it gated noise. THE MODELS STAY ON THE HOST ON BOTH PATHS, as every other "
                  "boundary condition of this solver does: the device loop reaches them through per-patch hooks. "
                  "What the device port added is WHERE it calls them: the alpha corrector takes a hook between its "
                  "high-order flux and its limiter (DeviceAlphaBoundary::updateModelled), rebuilds the bounded "
                  "flux's boundary on the new patch values as upwind's phi_b*psi_b, and lets the limiter see a "
                  "boundary correction that is no longer zero -- the device's explicit limiter already kept lambda "
                  "at 1 on uncoupled faces, as OpenFOAM's does; the alpha step counts sub-cycles for the model's "
                  "clock; and the velocity hook updates the model at the step's own clock before UEqn. ONE CASE "
                  "READS VERY DIFFERENTLY ON THE TWO PATHS AND IS NOT A DEFECT: on solitaryMcCowan the host is "
                  "1.6e-10 of alpha from OpenFOAM and the device 2.3e-12; with every p_rgh solve tightened on all "
                  "three codes they read 1.9e-12 and 2.0e-12, so the host's figure is where that case's relTol 0.1 "
                  "left it. THE NAMED FLUX HAD TWO MORE DEFECTS BEHIND IT, found by giving the damBreak gate a "
                  "`rhophi` profile that names rhoPhi on U, p_rgh AND alpha rather than letting that combination "
                  "merely run. Both are invisible to a condition naming phi, because nothing moves phi between the "
                  "end of one step's correctors and the next step's UEqn, and the alpha step moves rhoPhi exactly "
                  "there. (a) brae's conditions were last TOLD their flux at the end of the previous step, where "
                  "OpenFOAM's look it up at every updateCoeffs: p_rgh alone 4.7e-10 of alpha -> 1.2e-12 with a push "
                  "after the alpha step. (b) pressureInletOutletVelocity::updateCoeffs ends in "
                  "directionMixed::evaluate(), which REWRITES THE PATCH VALUE and clears the updated flag, so "
                  "UEqn's fvMatrix constructor re-evaluates the atmosphere's velocity from the new flux; brae "
                  "refreshed the coefficients and kept the value: U alone 2.5e-09 -> 8.4e-11 with the push -> "
                  "1.2e-12 with the value re-evaluated. On the device a p_rgh or alpha condition naming rhoPhi is "
                  "handed it (the hooks push after the alpha step already); a VELOCITY condition naming it is "
                  "refused, because that one switch runs on the device and reads phi."),
        dict(name="interFoam_GAMG", of_symbol="GAMGSolver",
             of_file="src/OpenFOAM/matrices/lduMatrix/solvers/GAMG/GAMGSolver.C",
             classification="LINEAR_SOLVER", status="REIMPLEMENT",
             brae_reference="src/matrices/lduMatrix/solvers/GAMG/gamg_solver_cpp.cuh",
             brae_target="src/matrices/lduMatrix/solvers/GAMG/device_gamg_solver.cu",
             validation="tests/interfoam_gamg_vs_openfoam.sh, real OpenFOAM run SERIALLY for fixed steps with "
                        "`DebugSwitches { GAMGAgglomeration 1; GAMG 1; }`, which the script proves inert on every "
                        "run (alpha, p_rgh and U byte-identical with and without). The switches make OpenFOAM print "
                        "its hierarchy level by level -- cells, faces per cell, and lduAddressing::band()'s profile, "
                        "which moves when the NUMBERING does -- and the coarsest-level solve of every V-cycle. "
                        "NINETEEN PROFILES on the laminar/waves tutorials, whose p_rghFinal names GAMG with a DIC "
                        "smoother: stokesI as shipped on three meshes; `deep` (1e-12, 125 V-cycles); the other three "
                        "smoothers; every sweep control off its default with pre-smoothing on; nCellsInCoarsestLevel "
                        "400; scaleCorrection no; both entries GAMG with different nCellsInCoarsestLevel; `square` "
                        "(dx = dz) and `cubes` (3-D, 0.05 cubes), where every face has one area; and the seven other "
                        "tutorials that name GAMG, as shipped. MEASURED: every level of every hierarchy is OpenFOAM's "
                        "(5 to 12 levels); every p_rgh iteration count (20 to 60 per profile) and every coarsest-level "
                        "count (3 to 125) is OpenFOAM's; alpha 4.0e-11, p_rgh 1.9e-10, U 1.9e-09 at worst. WHAT IT "
                        "REPLACED, PBiCGStab under a notice: alpha 2.3e-02 and U 19% on the same fixture. CONTROL ON "
                        "THE ORACLE: OpenFOAM with PCG against OpenFOAM with GAMG, alpha 1.9e-04 to 8.4e-02, never "
                        "under 1e5 times brae's distance. EVERY PORT DECISION BROKEN ONCE with switches not in the "
                        "tree: no alternation of the pairing direction 1.2e-03 of alpha; coarse faces sorted by "
                        "neighbour 1.1e-07 and the band profile moves; the rejected last level kept, or the coarsest "
                        "level solved to 1e-12, and NO FIELD MOVES -- only the coarsest-solve log arm holds those "
                        "two; prolongation that adds 2.8e-04; post-sweeps without the level multiplier 9.7e-05; the "
                        "hierarchy built from the entry in use rather than the first 8.5e-09 and 10 levels for 8. "
                        "THE GAMG PRECONDITIONER (`solver PCG; preconditioner { preconditioner GAMG; ... }`, six of "
                        "the seven solid-body tutorials' p_rghFinal) has two profiles of its own, the tutorials' form "
                        "(DICGaussSeidel, nPreSweeps 2, nVcycles 2) and one whose sub-dictionary tolerance differs "
                        "from the PCG's: 40 of 40 p_rgh counts, alpha 4.9e-12 and 3.4e-12, every coarsest count "
                        "OpenFOAM's; broken once each -- one V-cycle for two 1.5e-02, no residual between cycles "
                        "1.4e-02, the application not started from zero 7.1e-05, the coarsest at the PCG's "
                        "tolerance 0 of 9 coarsest counts. And the seven tanks of interfoam_moving_vs_openfoam.sh "
                        "run with it as shipped. "
                        "TWO DECISIONS READ NOTHING ON THE SHIPPED FIXTURE and have fixtures of their own: the "
                        "(1, 1.01, 1.02) weight perturbation is idle on 2 x 0.1 cells (identical to the last digit) "
                        "and worth 4.5e-03 on `square`, 2.0e-06 on `cubes`, 4.9e-06 with y and z swapped; scaling the "
                        "level above the coarsest moves the 11th digit with a 24-cell level there and 5.0e-05 of alpha "
                        "on `coarsest`. tests/interfoam_refusals.sh holds fifteen GAMG arms and three for the device. WHAT IT "
                        "DOES NOT CLAIM: an "
                        "asymmetric matrix, any refused control, and a parallel run (the smoothers and the hierarchy "
                        "are per-processor there). THE DEVICE LOOP RUNS THE SIXTEEN DIC PROFILES TOO, against "
                        "OpenFOAM directly: every p_rgh and every coarsest-level iteration count OpenFOAM's; alpha "
                        "2.9e-11, p_rgh 5.2e-11, U 1.6e-09 at worst (1.5e-08 on `deep`), 4.5e-11 of alpha from the "
                        "host. WHAT IT REPLACED there, Jacobi-BiCGStab under a notice: alpha 2.3e-02, U 45%. BROKEN "
                        "ONCE EACH ON THE DEVICE: coarse-level DIC diagonals left from the first solve 1.8e-03; "
                        "prolongation that adds 7.3e-03; the level above the coarsest scaled 5.0e-05 on `coarsest`; "
                        "the coarsest MATRIX left from the first solve moves no field and reads 2.3e-02 on the "
                        "coarsest-solve arm. The device's GAMG has the DIC smoother only and the driver REFUSES "
                        "the other three by name. WITH BOTH LOOPS ON THE TUTORIALS' OWN SOLVER the wave gate "
                        "(interfoam_waves_vs_openfoam.sh) dropped its PCG staging: fourteen profiles as shipped, "
                        "alpha 2.9e-11 where it read 1.6e-10, and its bounds came down with it.",
             note="GAMG IS OpenFOAM'S, NOT A MULTIGRID: the hierarchy's numbering, the smoother's face order and "
                  "the stopping rule are all part of where a loose solve stops, and on the wave tanks that is part "
                  "of the answer -- the active absorption feeds the water level back into the inlet velocity, and "
                  "OpenFOAM against ITSELF with PCG in GAMG's place differs by 1.8e-02 of alpha after twenty steps. "
                  "THE HIERARCHY (pair_gamg_agglomeration_cpp.cuh): faceAreaPair weighs faces by "
                  "|Sf/sqrt(magSf) * (1, 1.01, 1.02)|, a perturbation against jagged pairings on axis-aligned "
                  "meshes; each pass pairs a cell with its heaviest unpaired neighbour, else joins the heaviest "
                  "neighbour's cluster; the sweep direction flips after EVERY pass through a STATIC flag, "
                  "including the pass continueAgglomerating() rejects, whose level is discarded whole; coarse "
                  "faces are numbered owner-major and within an owner by FIRST APPEARANCE in the fine face walk, "
                  "not by neighbour. It is a MeshObject: the first GAMG solve of the run builds it from ITS "
                  "entry's nCellsInCoarsestLevel and no later entry's is read. THE V-CYCLE (gamg_solver_cpp.cuh): "
                  "no pre-smoothing by default; restriction is summation; the coarsest level is solved by PCG "
                  "with DIC from zero to the GAMG's OWN tolerance and relTol, not directly; prolongation is "
                  "injection that OVERWRITES; the correction is scaled by (x.b)/(x.Ax) with a Jacobi step on the "
                  "remainder, for a symmetric matrix and NOT on the level above the coarsest; post-sweeps are "
                  "min(nPostSweeps + postSweepsLevelMultiplier*level, maxPostSweeps); the finest correction is "
                  "scaled against the residual and psi is smoothed nFinestSweeps times against the SOURCE. DIC "
                  "as a smoother is an iteration on the true residual. A FINDING THAT IS NOT THE SOLVER'S, from "
                  "`deep`: ten consecutive solves read 1.24e-05 on their initial residuals with alpha 6e-12 from "
                  "OpenFOAM, because one top-patch face's flux sits at 1e-14 and pressureInletOutletVelocity "
                  "branches on its sign; the branch changes U's diagonal under that face, hence rAU and the "
                  "p_rgh matrix's row sum on a patch whose value is the solution there, hence normFactor and "
                  "every residual divided by it -- without moving the solution. brae::gamg "
                  "(src/OpenFOAM/matrices/gamg.cu), which simpleFoam_cpp uses, is an earlier multigrid with a "
                  "sorted coarse numbering and a direct coarsest solve; it is not this and was left alone. "
                  "THE PRECONDITIONER (GAMGPreconditioner.C) is a GAMGSolver built on the PCG's own matrix with "
                  "the `preconditioner` SUB-DICTIONARY as its controls (lduMatrixPreconditioner.C, New) -- its "
                  "tolerance and relTol are the coarsest solve's, not the PCG's -- and each application runs "
                  "nVcycles (default 2) of the same V-cycle from w = 0, recomputing r - A*w between cycles. PCG "
                  "builds it only once the first residual is above tolerance. UNDER IT THE COARSEST LEVEL "
                  "USUALLY DOES NOTHING: its source is the PCG residual restricted over aggregates, which "
                  "cancels to ~1e-26, and a solve from zero normalised by sum|source| + 1e-20 reads an initial "
                  "residual below tolerance and takes no iteration -- six of eight on the gated fixture, in "
                  "OpenFOAM and in brae alike. The gate asserts the coarsest counts there and holds the "
                  "residuals, which are that cancellation read to its last digits, to 1e-3. "
                  "PCG.C's checkSingularity break is transcribed in the coarsest-level PCG and is ABSENT from "
                  "brae::pcg (it fires only on wApA/normFactor under 1e-300). ON THE DEVICE "
                  "(device_gamg_solver.cuh): the hierarchy is the host's own code, uploaded once; the coarse "
                  "matrices and the restriction are FIXED-ORDER GATHERS in the host's accumulation order (a "
                  "scatter with atomic adds is neither ordered nor reproducible); the smoother is DIC through "
                  "the level-scheduled DILU with lower aliased to upper, one schedule per level, its "
                  "diagonal recomputed at every solve; the scaling's two dot products and the residual norms "
                  "are device reductions; and THE COARSEST LEVEL IS SOLVED ON THE HOST by the host's own "
                  "PCG -- ten cells by default, a pair of short copies against dozens of launches and a "
                  "read-back per iteration. That is a finished state, not a deferred one."),
        dict(name="interFoam_solidBodyMotion", of_symbol="dynamicMotionSolverFvMesh",
             of_file="src/dynamicFvMesh/dynamicMotionSolverFvMesh/dynamicMotionSolverFvMesh.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/dynamicFvMesh/dynamicMotionSolverFvMesh/dynamic_motion_solver_fv_mesh_cpp.cuh",
             brae_target="",
             validation="tests/mesh_motion_vs_openfoam.sh: the moved polyMesh/points and the mesh flux meshPhi "
                        "against OpenFOAM's moveDynamicMesh, written per step at writePrecision 18, and V and C of the "
                        "moved mesh against postProcess's writeCellVolumes and writeCellCentres on those points. "
                        "SEVEN PROFILES: testTubeMixer (multiMotion of a rotatingMotion and an "
                        "oscillatingRotatingMotion), sloshingTank2D (SDA, m4 mesh), sloshingTank3D6DoF "
                        "(tabulated6DoFMotion, spline, across two knots), sloshingCylinder (multiMotion of an "
                        "oscillatingLinearMotion and a rotatingMotion on a snappyHexMesh mesh of polyhedra), and "
                        "linearMotion, axisRotationMotion and tabulated6DoFMotion with linear interpolation on the "
                        "tutorials' meshes. MEASURED: points and meshPhi differ from OpenFOAM's by EXACTLY ZERO on "
                        "every profile, and the gate asserts zero (BRAE_MESH_MOTION_ROUNDOFF=1 holds 1e-15 and 1e-12 "
                        "instead, for a compiler that orders floating point differently) -- and V and C of the moved "
                        "mesh are exact too, since fv_geometry's face centres took primitiveMeshTools.C's operation "
                        "order (interFoam_nonOrthCorrection); they read 1.7e-14 and 8.0e-16 before. CONTROL on brae's side (the oracle has "
                        "no wrong answer to offer): the two multiMotion entries in the other order, 1.5e-01 of the "
                        "extent, which the script asserts FAILS. BROKEN ONCE EACH -- and the reason the comparison "
                        "is exact, since all but the last sit under any round-off bound: meshPhi as sweptVol/deltaT "
                        "rather than sweptVol*(1.0/deltaT) 2.0e-16; the swept volume fanned about the vertex mean "
                        "rather than face::centre 6.7e-15; points rotated through the quaternion rather than the "
                        "tensor q.R() 4.4e-16 and 1.8e-14; septernion*septernion without normalising the product "
                        "rotation 6.1e-16 and 2.1e-14; the time as index*deltaT rather than accumulated 1.8e-16 and "
                        "8.4e-15; oldPoints never refreshed 4.1e+00. END TO END in interFoam "
                        "(tests/interfoam_moving_vs_openfoam.sh), beside sloshingTank2D and the cylinder, the other "
                        "four tanks as shipped, ten steps of 0.01 against their static controls: sloshingTank2D3DoF "
                        "U 4.4e-13, sloshingTank3D 2.7e-12, sloshingTank3D3DoF 6.8e-13, sloshingTank3D6DoF 5.4e-13, "
                        "the moved points exact; SDA without its lamda rescaling 8.2e-01, without its roll 9.9e-01, "
                        "the 6DoF table's interpolation flipped 1.4e-02. A CELLZONE's motion (zoneMotion.C: the points of "
                        "every face of every zone cell, ascending; the rest of the mesh held where it stands) is "
                        "gated on RAS/mixerVesselAMI (tests/interfoam_ami_vs_openfoam.sh, test_face_area_weight_ami): "
                        "the moved points bitwise OpenFOAM's at every step checked; the whole mesh moved 1.3e-03, the "
                        "zone's points transformed from the CURRENT points 9.6e-04, the boundary faces left out of "
                        "the marking 9.6e-04; the owner side of the faces alone NOT DISCRIMINATED (it marks the same "
                        "points on that mesh). WHAT IT DOES NOT CLAIM: a cellSet, a cellZone named by a regular "
                        "expression or a zone group, a zone under displacementLaplacian, points0 read from a file, a restart of a moved mesh, any "
                        "motionSolver but solidBody, drivenLinearMotion, a non-constant Function1 coefficient, "
                        "mergeLevels, and the device -- refused by name, fifteen arms in tests/interfoam_refusals.sh.",
             note="EVERY MOTION IS AN ABSOLUTE FUNCTION OF TIME ON THE ORIGINAL POINTS (points0MotionSolver), a "
                  "septernion built and multiplied as OpenFOAM's quaternion and septernion multiply (src/OpenFOAM/"
                  "primitives/septernion/septernion_cpp.cuh), and applied by transformPoints: the translation "
                  "subtracted when it exceeds VSMALL, the rotation by the TENSOR q.R() when mag(R - I) exceeds "
                  "SMALL. multiMotion is the product of its entries in dictionary order. fvMesh::movePoints "
                  "takes V into V0 and the points into oldPoints ONCE PER TIME INDEX (a second update in the "
                  "same step, moveMeshOuterCorrectors, keeps both), sets meshPhi = face::sweptVol*(1/deltaT) "
                  "on every face but an empty patch's, and recomputes the geometry. face::sweptVol fans each "
                  "face about face::centre at both times, and face::centre is NOT primitiveMesh's Cf: a third "
                  "arrangement of the same sums, differing in the last digits on any face that is not a "
                  "triangle. src/OpenFOAM/motion/solid_body_motion.cuh and swept_volume.cuh, which pimpleFoam "
                  "still uses, are earlier transcriptions with Rodrigues' formula and a different centre; they "
                  "are the same motion and not the same digits, and were left alone. fvMesh::Vsc and Vsc0 "
                  "interpolate V0 and V to a sub-cycle's clock (fvMeshGeometry.C), and fvc::surfaceIntegrate "
                  "divides by Vsc, not V. HOST_ONLY is a finished state: the motion is mesh topology and "
                  "geometry, computed once per step; what the DEVICE loop needs is the moved geometry uploaded, "
                  "which is interFoam_dynamicMesh's target."),
        dict(name="interFoam_displacementLaplacian", of_symbol="displacementLaplacianFvMotionSolver",
             of_file="src/fvMotionSolver/fvMotionSolvers/displacement/laplacian/displacementLaplacianFvMotionSolver.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/fvMotionSolver/fvMotionSolvers/displacement/laplacian/"
                            "displacement_laplacian_fv_motion_solver_cpp.cuh",
             brae_target="",
             validation="tests/displacement_laplacian_vs_openfoam.sh: the moved polyMesh/points, the motion solver's "
                        "cellDisplacement and pointDisplacement, meshPhi and every GAMG solve of the displacement "
                        "equation against OpenFOAM's moveDynamicMesh run SERIALLY with a fixed deltaT, written per "
                        "step at writePrecision 18, and V and C of the moved mesh from postProcess. SEVEN PROFILES, "
                        "each the tutorial's own mesh and dictionaries: waveMakerPiston, waveMakerFlap (the wall "
                        "tilts) and waveMakerSolitary (0.26 m of stroke, two blocks), 20 steps of 0.05 to t = 1, "
                        "waveMakerMultiPaddleFlap and waveMakerMultiPaddlePiston (four paddles at 45 degrees, 3-D, "
                        "448000 cells), 4 steps of 0.1, and the piston and the flap again with `secondOrder yes` "
                        "staged into the paddle. MEASURED, worst of the seven: all 204 GAMG solves take "
                        "OpenFOAM's iteration count; points 2.9e-16 of the extent; cellDisplacement and "
                        "pointDisplacement 1.3e-14 of the largest displacement; meshPhi 2.4e-13 of the run's largest "
                        "|meshPhi|; V 1.1e-13 relative, C 5.9e-16 of the extent. The bounds are those numbers and say why: see the note. "
                        "THE CONTROL: the wall distance taken from rightwall instead of leftwall, 9.9e-01 of the "
                        "displacement, which the script asserts FAILS. BROKEN ONCE EACH, as a fraction of the largest "
                        "displacement: the paddle's values not written back over the interpolated ones 3.7e-02; y on "
                        "the named patch the next cell's instead of the wave's SMALL 4.0e-01 and 20 of 40 counts; the "
                        "wall distance without correctWalls 2.3e-05 (the flap only, whose wall tilts); the 2-D "
                        "correction skipped 7.4e-02 (the solitary only); volPointInterpolation's weights made once "
                        "6.2e-04; the diffusivity computed once 9.1e-02; the GAMG hierarchy kept from step one "
                        "1.9e-01; the non-orthogonal correction dropped 7.9e-02. TEN REFUSALS asserted by name: a "
                        "diffusivity other than inverseDistance, a cellDisplacement solver other than GAMG, a "
                        "laplacian other than Gauss linear corrected, a point condition other than fixedValue, "
                        "zeroGradient, empty and waveMaker, an unknown motionType, a waveMaker without `value`, and "
                        "frozenPointsZone; and three more on fvSchemes' `patchDist`, which the diffusivity's "
                        "wallDist reads (its patch type name is \"patch\"): a method other than meshWave, "
                        "correctWalls false, an updateInterval other than 1 -- with the defaults written out as the "
                        "arm that must run. NOT CLAIMED, because these meshes cannot tell: pointCells' order (the "
                        "pointFaces walk OpenFOAM takes here and the ascending order give the same lists on a block "
                        "mesh) and face::average against a vertex mean for cellMotion (equal on flat parallel faces); "
                        "the Final solver entry, which moveDynamicMesh never selects; and the device. UNDER "
                        "interFoam: all five are gated end to end (interFoam_correctPhi, "
                        "interFoam_interfaceCompression).",
             note="ONE newPoints(), IN OpenFOAM'S ORDER, all on the mesh before the move (motionSolver.C:200-204, "
                  "displacementLaplacianFvMotionSolver.C:199-318): the inverseDistance diffusivity 1/interpolate(y), "
                  "y the meshWave distance to the named patches (patchWave: FaceCellWave<wallPoint> seeded with the "
                  "patch face centres, visiting cells() in OpenFOAM's order and taking a new origin only when nearer "
                  "by 1%, then correctBoundaryCells' exact distance for every cell touching the patches), whose "
                  "boundary value is the wave's own on the named patches -- sqrt(0) + SMALL, so the diffusivity there "
                  "is 1e15 -- and the next cell's elsewhere; the point conditions' updateCoeffs, each writing its "
                  "values straight into the point field in patch order; cellMotion on every patch whose point "
                  "condition fixes a value, the face::average of the POINT field; laplacian(diffusivity, "
                  "cellDisplacement) Gauss linear corrected, solved component by component with GAMG from the last "
                  "step's solution, the empty direction skipped; then volPointInterpolation -- cells to points by "
                  "1/|p - C|, boundary faces of real patches to patch points by 1/|p - Cf|, the weights remade on "
                  "every move -- the fixing conditions evaluated again over it, points0 + pointDisplacement, and "
                  "twoDPointCorrector, whose plane normal is taken once and whose mid-plane is the current bounds'. "
                  "THE GAMG SOLVE HAS TO BE OPENFOAM'S EXACTLY: the paddle's 1e15 diffusivity dominates the residual "
                  "normalisation, the equation is declared converged after one V-cycle, and the mesh is that "
                  "V-cycle's. The hierarchy is the mesh's MeshObject, shared with p_rgh's GAMG and rebuilt after "
                  "each move; DynamicMotionSolverFvMesh::update takes the run's cache and invalidates it. "
                  "AN OPEN FINDING THAT BELONGS TO THE SHARED OPERATORS: brae's fvm::laplacian forms its face "
                  "coefficient as (deltaCoeffs*gamma)*magSf where gaussLaplacianScheme forms gammaMagSf = "
                  "gamma*magSf first, and brae's linear interpolation is w*P + (1 - w)*N where "
                  "surfaceInterpolationScheme::dotInterpolate is lambda*(P - N) + N. Switched to OpenFOAM's order, "
                  "every profile of this gate is EXACT for its first four to seven steps; they are what the gate's "
                  "bounds are. Left as they are here because every solver's laplacian and interpolation read them. "
                  "face::centre, face::average and face::nearestPoint are src/OpenFOAM/meshes/meshShapes/face/"
                  "face_cpp.cuh; the wave src/meshTools/cellDist/patchWave/; the interpolation "
                  "src/finiteVolume/interpolation/volPointInterpolation/vol_point_interpolation_cpp.cuh (the older "
                  "vol_point_interpolation.cuh, which velocityComponentLaplacian uses, was left alone). "
                  "moveDynamicMesh does not link libwaveModels, which interFoam does: the gate's controlDict loads it."),
        dict(name="interFoam_waveMaker", of_symbol="waveMakerPointPatchVectorField",
             of_file="src/waveModels/derivedPointPatchFields/waveMaker/waveMakerPointPatchVectorField.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/waveModels/derivedPointPatchFields/waveMaker/"
                            "wave_maker_point_patch_vector_field_cpp.cuh",
             brae_target="",
             validation="tests/displacement_laplacian_vs_openfoam.sh (interFoam_displacementLaplacian): the paddle is "
                        "every profile's only moving boundary, so its displacement is the whole of the motion -- "
                        "piston and flap, one paddle and four at 45 degrees, and the solitary stroke with its Newton "
                        "iteration -- and the piston and the flap with `secondOrder yes` staged in, which no tutorial "
                        "turns on. The paddle points' pointDisplacement is OpenFOAM's to the gate's 2e-14 of the "
                        "largest displacement. NOT CLAIMED: a waveMaker without `value`, which OpenFOAM evaluates "
                        "at construction (refused).",
             note="updateCoeffs (waveMakerPointPatchVectorField.C:278-436): the wave length by 100 fixed-point "
                  "iterations of L = L0 tanh(2 pi h/L), the paddle's phase its centre's position alone (wavePhase is "
                  "read and never used), and the ramp clamp(t/rampTime, 0, 1) on the piston and the flap but not the "
                  "solitary stroke; the flap's stroke scaled by the point's height above the patch's lowest point. "
                  "The paddle centres and every point's paddle are taken from the patch's points at construction "
                  "(initialiseGeometry) and never again; the reference depth on the first updateCoeffs. HOST_ONLY is "
                  "a finished state: the condition is a function of time on the patch's points, and what the "
                  "device needs is the moved mesh."),
        dict(name="interFoam_dynamicMesh", of_symbol="mesh.update",
             of_file="applications/solvers/multiphase/interFoam/interFoam.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_driver_cpp.cu",
             brae_target="",
             validation="tests/interfoam_moving_vs_openfoam.sh, real OpenFOAM run SERIALLY, ten fixed steps of 2e-4 on "
                        "laminar/testTubeMixer -- a closed tube of water and air on a turntable that also tilts, every "
                        "wall a movingWallVelocity -- against its fields, its moved points, its Uf and its wall "
                        "velocity, with the same tube held still (`staticFvMesh`) as the control. FIVE PROFILES: as "
                        "staged (the tutorial's nAlphaSubCycles 3), MULESCorr, nOuterCorrectors 2 with "
                        "moveMeshOuterCorrectors (the mesh moved twice in one step), without it (the second "
                        "corrector must NOT move it), and momentumPredictor yes. MEASURED, worst of the five: every "
                        "p_rgh iteration count OpenFOAM's (20 to 40 per profile), initial residuals 3.7e-11 in step "
                        "one and 4.8e-09 over the run, alpha 5.6e-12, p_rgh 1.5e-13, U 1.0e-10, Uf 6.8e-11, the wall "
                        "velocity 7.0e-14 on 1050 faces, the moved points OpenFOAM's exactly. THE CONTROL: the still "
                        "tube is 100% of U and 0.8 of alpha away. THE TUTORIAL RUNS AS SHIPPED: vanLeerV (interFoam_vanLeerV) and "
                        "p_rghFinal's PCG with a GAMG preconditioner (interFoam_GAMG) were staged away until each "
                        "was ported. With them, alpha 6.1e-12 and U 7.0e-11 at worst over the five profiles; the "
                        "two non-orthogonal tanks run as shipped too (interFoam_nonOrthCorrection). BROKEN ONCE EACH: phi left absolute after the corrector alpha 9.2e-01 and U "
                        "470%; ddtCorr from phi.oldTime() instead of Sf & Uf.oldTime() 8.1e-02 and 28%; Uf left at "
                        "interpolate(U) 3.3e-02 and 27%; gh and ghf not recomputed 3.3e-04; the GAMG hierarchy kept "
                        "from step one 9.3e-06 and 10 of 20 counts; the mesh moved on every outer corrector 5.1e-07 "
                        "and 29 of 40; oldPoints re-taken on the second update 9.9e-02 and 100%; the wall velocity "
                        "as the face-centre displacement alone, without the swept volume in its normal component, "
                        "and adjustPhi STOPS THE RUN with OpenFOAM's own `continuity error cannot be removed`. WHAT "
                        "THIS GATE CANNOT SEE, because the motion is rigid: every cell keeps its volume to round-off, "
                        "so V0 against V in fvm::ddt and MULES and Vsc's interpolation inside a sub-cycle move the "
                        "answer by 1e-15 -- transcribed, and gated only by a mesh that deforms. NOT CLAIMED: "
                        "a turbulent case on a moving mesh (refused) and the device (refused). correctPhi is "
                        "interFoam_correctPhi, and a wave condition on a moving mesh runs: see there.",
             note="WHAT interFoam DOES ON A MOVING MESH, in the order of interFoam.C:118-148 and pEqn.H. At the top "
                  "of the first outer corrector: mesh.update(), whose last line is U.correctBoundaryConditions() -- "
                  "a movingWallVelocity takes Uwall = Up + n*(Un - n&Up) with Up the face::centre displacement "
                  "over deltaT and Un = meshPhi/(magSf + VSMALL): the normal component is the SWEPT VOLUME's, so "
                  "the wall's flux is the mesh flux and a closed tank's adjustPhi closes on it; then gh and ghf "
                  "from the new centres; the correctPhi block is skipped when the case says `correctPhi no`, "
                  "which every solid-body tutorial does. The alpha sub-cycle convects with the RELATIVE phi the "
                  "last corrector left and divides by Vsc; fvm::ddt(rho, U) carries V0 in its source; ddtCorr(U, "
                  "phi, Uf) is ddtCorr(U, Uf) on a dynamic mesh: Sf & Uf.oldTime() in phi.oldTime()'s place, with "
                  "the NEW Sf and weights; adjustPhi sees phiHbyA made relative and is made absolute again; after "
                  "`phi = phiHbyA - p_rghEqn.flux()` and U's correction, fvc::correctUf replaces Uf's normal "
                  "component by the ABSOLUTE flux's and fvc::makeRelative subtracts meshPhi, so the phi that "
                  "leaves the corrector is relative. GAMGAgglomeration::movePoints marks the hierarchy for "
                  "rebuilding at every step (updateInterval 1), from wherever the static pairing direction was "
                  "left. A TRANSCRIPTION SLIP FOUND ON THE WAY: ddtCorr's limiter divided by |phi| + 1e-37, "
                  "the FLOAT build's VSMALL (floatScalar.H:64), where OpenFOAM's double build adds SMALL = 1e-15 "
                  "(doubleScalar.H:62); the two differ only where |phi| is at or below 1e-15 and no gated number "
                  "moved when it was corrected, on the host or on the device (device_inter_peqn.cu had it too and "
                  "tests/test_device_inter_peqn.cu holds the two bit-identical); time_controls.cuh's setDeltaT "
                  "still divides by CoNum + 1e-37 where setDeltaT.H adds SMALL (open finding, in that "
                  "component). THE DRIVER TAKES ITS MESH BY CONST REFERENCE and a moving "
                  "case needs the same objects mutable -- the points, the geometry, and the patches every patch "
                  "field references -- so runInterFoam takes a MutableMesh naming them, checked by address, and "
                  "refuses a moving case without one."),
        dict(name="interFoam_correctPhi", of_symbol="CorrectPhi",
             of_file="src/finiteVolume/cfdTools/general/CorrectPhi/CorrectPhi.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_correct_phi_cpp.cuh",
             brae_target="",
             validation="tests/interfoam_moving_vs_openfoam.sh, real OpenFOAM run serially for fixed steps. FIVE "
                        "PROFILES: the three solid-body tanks with `correctPhi yes` (testTubeMixer; sloshingTank2D's "
                        "non-orthogonal pcorr laplacian; sloshingCylinder's non-orthogonal corrector, pcorr then "
                        "pcorrFinal, staged `maxIter 1000` because the shipped 100 leaves PCG unconverged at 0.1 to "
                        "0.3 in both codes and then reads alpha 7.6e-09 on 22 of 22 counts); laminar/waves/"
                        "waveMakerSolitary AS SHIPPED, thirty steps -- a deforming mesh, correctPhi by default, an "
                        "absorbing outlet, an open top; and closed damBreak STARTED MOVING, U (0.1 0 0) against its "
                        "walls, where initCorrectPhi does 79 iterations of work. Every other profile of the gate "
                        "compares initCorrectPhi's start-up solve too. MEASURED: every pcorr and p_rgh iteration count "
                        "OpenFOAM's; waveMakerSolitary alpha 2.9e-12, p_rgh 3.3e-13, U 1.7e-10; the tanks alpha "
                        "1.4e-12, 3.0e-13, 1.2e-10; the moving dam alpha 1.2e-14, U 2.3e-12. BROKEN ONCE EACH: "
                        "correctUphiBCs skipped, and phi not rebuilt from Sf & Uf, both stop the run in adjustPhi as "
                        "OpenFOAM would; rAUf = 1 alpha 3.6e-02; no pcorr reference 4.1e-07; phi left absolute "
                        "9.9e-01; pcorr's non-orthogonal flux dropped 4.9e-01 (the cylinder -- identically zero with "
                        "no corrector, since pcorr starts at 0); initCorrectPhi skipped 3.2e-02 on the moving dam; "
                        "fvm::ddt's V0 as V 1.1e-02 and the sub-cycle's Vsc as V 2.6e-03 on waveMakerSolitary, the "
                        "first gated case whose cells change volume; the outlet's wave model updated in UEqn "
                        "rather than at the mesh update U 2.0e-05. NOT CLAIMED: the curvature pass after "
                        "CorrectPhi (under 3e-13 on these tanks), the corrected flux handed to flux-conditional "
                        "patches (a closed tank has none), a divU, and the device (it runs initCorrectPhi on the "
                        "host and refuses a moving mesh).",
             note="ONE CorrectPhi (CorrectPhi.C:36-117): correctUphiBCs when the mesh is changing -- every velocity "
                  "patch that FIXES A VALUE evaluated again, a pressureInletOutletVelocity against the Sf & Uf "
                  "flux, and phi there set to U_b & Sf; pcorr zero, fixedValue where p_rgh fixes a value and "
                  "zeroGradient elsewhere; on a closed domain phi made relative, adjustPhi, made absolute; "
                  "laplacian(rAUf, pcorr) == div(phi), pinned at cell 0 to 0 when it needs a reference, solved "
                  "nNonOrthogonalCorrectors + 1 times with pcorrFinal last; phi -= flux() on the last pass, the "
                  "face correction included. interFoam.C:136-146 then makes phi relative and calls "
                  "mixture.correct(), which recomputes nHatf on the moved mesh. initCorrectPhi.H runs it at the "
                  "start of EVERY case with rAUf exactly 1 (OpenFOAM's interpolation of a uniform 1 is exact); on "
                  "a case at rest it is one solve of zero iterations, and no existing gate moved a digit when it "
                  "went in. rAU is kept across steps under correctPhi (pEqn.H:4). THREE THINGS CHANGED ON THE WAY: "
                  "pressureInletOutletVelocity now FIXES A VALUE, as directionMixed does in OpenFOAM (it said "
                  "false; switching it moved nothing on damBreak, waves or capillaryRise, host or device); PCG "
                  "with a GAMG preconditioner builds the mesh's hierarchy only when its initial residual has not "
                  "converged, as PCG.C constructs its preconditioner, so a pcorr at rest neither builds it nor "
                  "flips the pairing direction; and on a moving mesh the wave velocity model updates at the mesh "
                  "update, where OpenFOAM's log has it. A wave condition on a moving mesh is no longer refused: "
                  "OpenFOAM's model keeps its construction geometry and reads the patch's current magSf, and "
                  "brae's does the same. CLOSED, found on the way: brae's dictionary expansion resolved `$name` "
                  "only against literal keys, where OpenFOAM's keyword substitution matches patterns too "
                  "(REGEX_RECURSIVE, dictionary.C:415-443) -- the two multi-paddle waveMakers write "
                  "`p_rgh { $pcorr; ... }` against a `\"(pcorr|pcorrFinal)\"` entry, and brae read no solver "
                  "there and ran PBiCGStab under a notice. The expander now searches as dictionary::csearch "
                  "does (tests/test_dict_scoped_macro.cu, scored against foamDictionary -expand over the 9,500 "
                  "tutorial dictionaries: 2,958 moved entries agree with OpenFOAM, 17 agreed only before), and "
                  "both cases are profiles of the moving gate."),
        dict(name="interFoam_interfaceCompression", of_symbol="interfaceCompressionLimiter",
             of_file="src/transportModels/interfaceProperties/interfaceCompression/interfaceCompression.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSchemes_cpp.cuh",
             brae_target="",
             validation="tests/interfoam_dambreak_vs_openfoam.sh's `compression`: damBreak at dt 5e-3 with "
                        "`div(phirb,alpha) Gauss interfaceCompression`, on a mesh that does not move. MEASURED: "
                        "every p_rgh and alpha iteration count OpenFOAM's, alpha 6.6e-14, p_rgh 3.1e-14, U 2.2e-13. "
                        "THE CONTROL: the scheme moves OpenFOAM's own alpha by 1.03e-01 against `Gauss linear`. "
                        "BROKEN ONCE EACH: the quadratic form OpenFOAM leaves commented out 1.0e-02 of alpha; min "
                        "for max 8.1e-02. NOT DISCRIMINATED: pos against pos0 in the blend (no face where the "
                        "limiter is below 1 carries exactly zero flux). END TO END, tests/interfoam_moving_vs_openfoam."
                        "sh's `piston` and `flap`: laminar/waves/waveMakerPiston and waveMakerFlap, thirty steps of "
                        "0.01, the mesh deformed by the paddle, correctPhi, the absorbing outlet -- with p_rgh, "
                        "p_rghFinal and pcorr converged to 1e-13, because as shipped their 135- to 480-iteration "
                        "solves carry last-bit differences out to U 3.9e-07 on the piston (converged: 1.6e-09); "
                        "alpha 1.5e-12 and 1.0e-11, U 1.6e-09 and 1.1e-09, every solve below its tolerance in both "
                        "codes, counts equal on short solves and within 2% on long ones (3 of 194 at worst). NOT "
                        "CLAIMED: the device, which refuses the scheme by name -- its alpha mapping took it as "
                        "linear until the host ported it. THE TWO MULTI-PADDLE waveMakers, `multiPiston` and "
                        "`multiFlap` of the same script: AS SHIPPED on their own 448000-cell 3-D mesh, four paddles "
                        "at 45 degrees, thirty steps of 0.01, GAMG as the solver of pcorr and p_rgh through "
                        "`p_rgh { $pcorr; }`. MEASURED (piston, flap): all 90 p_rgh and all 31 pcorr iteration "
                        "counts OpenFOAM's, alpha 5.7e-13 and 2.9e-12, p_rgh 5.6e-13 and 2.6e-12, U 2.2e-10 and "
                        "1.8e-09, the moved points 1.1e-16 of the extent; the control, OpenFOAM with its mesh held "
                        "still, U 100%. BROKEN ONCE, the pattern lookup taken out of the dictionary reader: 48 of "
                        "90 p_rgh counts, alpha 9.7e-04, U 1.0e-01.",
             note="A PhiScheme (interfaceCompression.C:33-41), so the limiter reads the two cell values of the field "
                  "and nothing else: clamp(1 - max(sqr(1 - 4 phiP (1 - phiP)), sqr(1 - 4 phiN (1 - phiN))), 0, 1), "
                  "the quartic form -- 1 where both cells are half full, 0 where either is empty or full -- blended "
                  "as limitedSurfaceInterpolationScheme::weights does, limiter*CDweight + (1 - limiter)*"
                  "pos0(faceFlux), and 1 on an uncoupled patch (PhiScheme.C). alphaEqn.H applies it twice through "
                  "fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme), alpha1, alpharScheme): on alpha2 upwinded by "
                  "-phir, then on alpha1 by the negated result."),
        dict(name="interFoam_pressureReference", of_symbol="setRefCell",
             of_file="src/finiteVolume/cfdTools/general/findRefCell/findRefCell.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_case_cpp.cu",
             brae_target="",
             validation="tests/interfoam_moving_vs_openfoam.sh's `closedDamBreak`: laminar/damBreak with its atmosphere "
                        "WALLED OFF (U fixedValue 0, p_rgh fixedFluxPressure, alpha zeroGradient) and PIMPLE naming "
                        "`pRefPoint (0.292 0.292 0.0073); pRefValue 0;`, twenty of the tutorial's own steps against real "
                        "OpenFOAM. MEASURED: 60 of 60 p_rgh iteration counts, initial residuals 2.0e-11 over the run, "
                        "alpha 1.2e-14, p_rgh 9.5e-15, p 5.7e-15, U 8.1e-14. THE CONTROL: OpenFOAM's own closed dam with "
                        "`pRefValue 1e5`, whose p is 89 times its magnitude away and whose U is 3.0e-11 away -- the "
                        "reference moves the level and nothing else. BROKEN ONCE: the reference cell pinned at "
                        "pRefValue rather than its current p_rgh (what this code did on a path no gated case had "
                        "taken) reads 15 of 60 counts and U 4.4e-05 on the closed dam, 18 of 20 and 5.3e-07 on the "
                        "moving tube. Every solid-body tutorial is a closed tank and takes this path -- and "
                        "sloshingTank2D's pRefPoint (0 0 0.15) LIES ON A FACE: findCell takes the nearer of two "
                        "centres 5e-16 apart, brae took the other one while its face centres differed from "
                        "OpenFOAM's in the last bit, and read alpha 2.8e-06 and 18 of 20 counts for it. NOT CLAIMED: "
                        "polyMesh::findCell's octree fallback when the face-plane test finds no cell (refused by "
                        "name), and the device (refused: its reference is the pRefValue one).",
             note="p_rgh.needReference() is `no patch fixes a value` (GeometricField.C:1068-1085). setRefCell "
                  "(findRefCell.C:36-110) then reads pRefCell, or pRefPoint located by findCell(point, FACE_PLANES): "
                  "the nearest cell centre if the point is inside that cell by every face plane -- (Sf & (p - Cf)) "
                  "<= 0 with Sf flipped for a neighbour -- else the FIRST cell in index order that is; and "
                  "pRefValue, mandatory. createFields.H:115-124 shifts p by pRefValue - p[refCell] and rebuilds "
                  "p_rgh from it once at the start; pEqn.H:47 pins the reference cell at getRefCellValue(p_rgh, "
                  "pRefCell), ITS CURRENT p_rgh -- not at pRefValue, which is p's level, applied again at "
                  "pEqn.H:74-83 after the solve with p_rgh rebuilt from the shifted p. adjustPhi (adjustPhi.C) "
                  "runs on such a case only: the outflow of every patch whose U does not fix a value (an "
                  "inletOutlet counts as adjustable) is scaled to balance the inflow, and an imbalance no "
                  "adjustable outflow can remove stops the run -- which is what a wall velocity with the wrong "
                  "normal component does on a closed tank. brae's driver set needReference = false with a note "
                  "that damBreak's atmosphere is a totalPressure; every case gated before this one had one."),
        dict(name="interFoam_vanLeerV", of_symbol="vanLeerV",
             of_file="src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/vanLeer/vanLeer.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSchemes_cpp.cuh",
             brae_target="src/cuda/device_fvm.cu",
             validation="tests/interfoam_dambreak_vs_openfoam.sh's `vanleerv` profile: laminar/damBreak at the big step "
                        "with `div(rhoPhi,U) Gauss vanLeerV` in place of its linearUpwind, five steps against real "
                        "OpenFOAM, host and device, with the big-step run as the control. MEASURED: alpha 1.2e-13, "
                        "U 3.1e-13 on the host and alpha 3.8e-14, U 1.2e-11 on the device, all 15 p_rgh counts "
                        "OpenFOAM's; the scheme moves OpenFOAM's own alpha by 2.0e-01. AND ON THE FIXTURE IT EXISTS "
                        "FOR: tests/interfoam_moving_vs_openfoam.sh's testTubeMixer, whose own fvSchemes name it, now "
                        "run as shipped there. BROKEN ONCE: "
                        "vanLeer's limiter replaced by limitedLinear's clamp of 2r on the same r, host and device "
                        "alike, alpha 2.1e-01 (45% of the change the scheme makes). A SECOND PROFILE, `linear`, holds "
                        "`Gauss linear` on div(rhoPhi,U) on both paths (alpha 9.8e-14 host, 1.2e-13 device), for the "
                        "reason in the note.",
             note="vanLeerV IS LimitedScheme<vector, vanLeerLimiter<NVDVTVDV>, null> (vanLeer.C:37): the r of the V "
                  "schemes -- one per face, from the vector difference across it dotted with the upwind cell's "
                  "gradient projected on d (NVDVTVDV.H) -- with vanLeer's (r + |r|)/(1 + |r|), which is not "
                  "clamped and reaches 2. The gradient is fvc::grad(U) through the case's grad(U) entry "
                  "(LimitedScheme::calcLimiter). brae already had both halves: rVector for limitedLinearV and "
                  "vanLeerLimiter for div(phi,alpha); vanLeerVWeights is their product. On the device the V kernel "
                  "selected its limiter by clamping alone, so the limiter selection was hoisted out of the scalar "
                  "kernel into limiterOfR and both kernels take it, vanLeer chosen by the kVanLeerTwoByk sentinel. "
                  "THE `linear` PROFILE EXISTS BECAUSE OF WHAT WAS FOUND WIRING THIS: interFoam's own scheme enum "
                  "has `linear` (three shipped tutorials name it) and its device mapping had no case for it, so "
                  "it fell through the switch's `default` to upwind -- a -device run would have convected upwind "
                  "under the name `linear`. The shared cpu::DivScheme now has `linear` (deviceDivCentralCoeffs; "
                  "the mesh's own weights on the host) and the mapping names every scheme with no default."),
        dict(name="interFoam_limitedLinear", of_symbol="limitedLinear",
             of_file="src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedLinear/limitedLinear.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_ueqn_cpp.cu",
             validation="tests/interfoam_limitedlinear_vs_openfoam.sh, real OpenFOAM on "
                        "laminar/vofToLagrangian/eulerianInjection AS ITS Allrun MESHES IT (blockMesh, topoSet, "
                        "subsetMesh, the patch and collector sets, setFields) with the block at 45^3 in place of "
                        "75^3, 90627 cells; its own `div(rhoPhi,U) Gauss limitedLinear 0.2`; 60 fixed steps of "
                        "2e-5. MEASURED: alpha 9.2e-14, p_rgh 8.8e-12, U 2.7e-12, all 180 p_rgh counts OpenFOAM's; "
                        "bounds at about 30x. THE CONTROLS, OpenFOAM against itself: `Gauss upwind` 8.0e-01, "
                        "`Gauss limitedLinear 1` 7.8e-01. BROKEN ONCE EACH (U): the V form's weights 8.5e-01; "
                        "magSqr's patch values taken from the face cells 4.3e-01; the coefficient ignored 7.8e-01; "
                        "upwind in its place 8.0e-01; the limiter on mag(U) 8.2e-01. WHY 60 STEPS: the gap opens at "
                        "step 68 in one near-air cell at the jet edge, where the density ratio of 1000 turns "
                        "alpha's 4e-13 into rho's 3e-10, and reads U 1.2e-08 at 100; brae against itself with only "
                        "the face interpolation's arithmetic reordered reads 2.9e-09 there, so that growth is the "
                        "case's conditioning. NOT CLAIMED: the tutorial's 420k-cell mesh (a bench size), its "
                        "Lagrangian function object (stripped), limitedLinear across a coupled patch (refused), "
                        "and the device loop (refused by name).",
             note="limitedLinear on a VECTOR is LimitedScheme<vector, limitedLinearLimiter<NVDTVD>, "
                  "limitFuncs::magSqr> (LimitedScheme.H:185-189): ONE scalar limiter per face, computed on "
                  "magSqr(U) with grad(magSqr(U)), carrying all three components -- not per component and not "
                  "the V form. The host had the scalar weights already (limitedLinearWeights, k and epsilon); "
                  "interFoam's momentum refused the scheme by name until now. The coefficient is read with no "
                  "default and must lie in [0, 1] or OpenFOAM stops (limitedLinear.H:67-76); brae's reader "
                  "used to default a missing one to 1, and now refuses both. THE DEVICE momentum has a "
                  "limitedLinear branch that reads U 5.1e-01 against OpenFOAM on a DIC-smoother twin of this "
                  "case at 50 steps, where the host reads 3.9e-12 and the device under upwind 1.1e-13; the "
                  "device loop refuses the scheme and that branch is an open finding. HOST ONLY SO FAR."),
        dict(name="interFoam_nonOrthCorrection", of_symbol="correctedSnGrad",
             of_file="src/finiteVolume/finiteVolume/snGradSchemes/correctedSnGrad/correctedSnGrad.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_peqn_cpp.cu",
             brae_target="",
             validation="tests/interfoam_moving_vs_openfoam.sh's two non-orthogonal tanks, against real OpenFOAM: "
                        "laminar/sloshingTank2D as shipped but for p_rghFinal -- SDA motion, 44-degree chamfered "
                        "cells, `Gauss linear corrected` and `corrected`, nAlphaSubCycles 3, cAlpha 1.5, 2-D -- ten of "
                        "its own steps: 20 of 20 p_rgh counts, alpha 5.8e-15, p_rgh 3.2e-14, U 3.9e-13, the still "
                        "tank 100% of U away; and laminar/sloshingCylinder -- a snappyHexMesh cylinder of polyhedra "
                        "at 26 degrees, MULESCorr, nNonOrthogonalCorrectors 1, its oscillation's phase and vertical "
                        "shifts zeroed because as shipped the mesh jumps 6.9 cm at the first update and OpenFOAM "
                        "itself blows up at any fixed step -- ten steps of 0.001: 40 of 40 counts, alpha 6.9e-11, U "
                        "1.0e-08. BROKEN ONCE EACH (tank / cylinder): the pressure laplacian's explicit correction "
                        "dropped from the source 8.7e-02 / 2.1e-01; its face flux dropped from p_rghEqn.flux() 9.5e-02 "
                        "/ 2.6e-01; the viscous laplacian orthogonal 6.4e-10 / 2.7e-04; the three snGrads orthogonal "
                        "1.3e-11 / 4.6e-04 (the tank's interface is horizontal; the cylinder is the fixture for that "
                        "one). tests/interfoam_refusals.sh: `corrected` on a sheared damBreak RUNS on the host and is "
                        "refused on the device; `uncorrected` on it is refused by name; and three gradSchemes arms. "
                        "WHAT IT DOES NOT CLAIM: `uncorrected` on a mesh that is not orthogonal, any gradSchemes "
                        "entry but `Gauss linear`. ON THE DEVICE (2026-09-19), one module at a time, each transcribed "
                        "from the host: the p_rgh laplacian's non-orthogonal LOOP (device_inter_pressure_step.cu, from "
                        "pressureCorrector) -- nNonOrthogonalCorrectors + 1 passes, nonOrthDeltaCoeffs, the explicit "
                        "correction from grad(p_rgh) on its stored patch values, the face-flux correction in "
                        "p_rghEqn.flux(), the Final entry on the last pass only. Its kernels are the host's bit for bit "
                        "(tests/test_device_laplacian_vs_host.cu, tests/test_device_gauss_grad.cu); the pass against the "
                        "host's on a sheared box, corrected, one non-orthogonal corrector: p_rgh 3.4e-11 relative, U "
                        "9.3e-13, the device assembled orthogonal 3.9e-02 (tests/test_device_inter_pressure_step.cu); "
                        "BROKEN ONCE EACH, p_rgh / U.x of 1.2e+06 / 1.2e+02: the correction dropped from the source "
                        "8.3e+04 / 1.3e+01, its face flux dropped 0 / 2.0e+01, one pass 8.3e+04 / 1.7e+01, the boundary "
                        "not re-evaluated between passes 5.0e+05 / 1.2e+02. End to end, laminar/damBreak `nonorth` on "
                        "the device against OpenFOAM: alpha 6.8e-14, p_rgh 7.1e-14, U 8.9e-12, all 30 p_rgh counts "
                        "OpenFOAM's. THEN THE OTHER SITES, each transcribed from the host driver's own call: the three "
                        "snGrads in the interface-force hook -- snGrad(alpha) in the surface tension, snGrad(rho) in "
                        "phig, snGrad(p_rgh) in the predictor's source, each under snGradSchemes with that field's own "
                        "gradSchemes entry, where the hook had `false` -- and the viscous laplacian, the shared "
                        "assembler handed the case's flags (device_inter_step.cu; its own gate is "
                        "tests/test_device_inter_ueqn_assembly.cu on a sheared box, corrected and `limited 0.5`, the "
                        "device assembled orthogonal 3.5e-05 of the source, and the fixture needed a 1e5x viscosity "
                        "before the correction was visible in a source dominated by rho*U/dt). CorrectPhi's pcorr is "
                        "host operators under the same flag; the closure's k and epsilon carry theirs already. WITH ALL "
                        "OF IT the device runs a mesh that is not orthogonal: laminar/damBreak `sheared` (six degrees, "
                        "momentum predictor on, nNonOrthogonalCorrectors 1, a moving start, phirb vanLeer) against "
                        "OpenFOAM -- alpha 1.8e-14, p_rgh 1.6e-14, U 4.9e-12 relative, every p_rgh and alpha count "
                        "OpenFOAM's. BROKEN ONCE EACH, that arm's alpha / U: snGrad(rho) orthogonal 4.3e-01 / 4.3e+00, "
                        "snGrad(p_rgh) 7.2e-03 / 3.8e+00, snGrad(alpha) 5.5e-04 / 3.3e-03. STILL REFUSED on the device: "
                        "`uncorrected` on a mesh that is not orthogonal (tests/interfoam_refusals.sh "
                        "device_sheared_uncorrected), and every gradSchemes entry but `Gauss linear`.",
             note="THE OPERATORS, from gaussLaplacianSchemes.C and correctedSnGrad.C: a corrected laplacian's matrix "
                  "takes nonOrthDeltaCoeffs on the internal faces and the patch's own coefficients on the boundary; "
                  "its source takes -V*div(gammaMagSf*snGradCorrection(vf)), the correction being "
                  "nonOrthCorrectionVectors dotted with the linear interpolate of grad(vf) THROUGH vf's OWN "
                  "gradSchemes entry, a vector field's by component and so through `default`; and because "
                  "p_rgh is fluxRequired the same face field is kept as the matrix's faceFluxCorrection, which "
                  "p_rghEqn.flux() adds back so `phi = phiHbyA - flux` stays conservative. A corrected snGrad is "
                  "nonOrthDeltaCoeffs*(vfN - vfP) plus that correction. `limited <c>` caps the correction per "
                  "face against the orthogonal part. brae had all of it (fvm.cuh, fvc.cu, addDivDevReff) from "
                  "rhoSimpleFoam; interFoam refused every corrected scheme on a mesh that was not orthogonal and "
                  "now passes the case's flags through the pressure equation, the viscous term and the three "
                  "snGrads -- p_rgh in the predictor's force, rho in phig, alpha in the surface-tension force. "
                  "`uncorrected` is NOT brae's orthogonal assembly: uncorrectedSnGrad takes nonOrthDeltaCoeffs, "
                  "orthogonalSnGrad takes deltaCoeffs (uncorrectedSnGrad.H:91, orthogonalSnGrad.H:91), so it is "
                  "refused on a mesh where they differ. gradSchemes WERE NEVER READ by this solver: 40 of the 44 "
                  "tutorials write `default Gauss linear;` alone, and the four that limit a gradient "
                  "(mixerVesselAMI, DTCHull, DTCHullMoving, electrostaticDeposition) would have run unlimited; "
                  "every entry but `Gauss linear` is refused now. TWO FINDINGS ON THE WAY, both about a mesh "
                  "that is not a mesh of rectangles: fv_geometry's face centres were (1/3)*(sumAc/sumA) where "
                  "primitiveMeshTools.C computes (1/3)*sumAc/sumA, and a triangle's (a+b+c)/3 where OpenFOAM "
                  "writes (1/3)*(a+b+c) -- with OpenFOAM's order V and C are OpenFOAM's to the bit on every mesh "
                  "the motion gate runs (hex, chamfered, snappy), where they read 1e-14 before; and that last "
                  "digit decided the pressure reference of sloshingTank2D, whose pRefPoint lies on a face "
                  "(interFoam_pressureReference)."),
        dict(name="interFoam_vanLeer", of_symbol="vanLeer",
             of_file="src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/vanLeer/vanLeer.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_reference="src/cuda/device_fvm.cu",
             brae_target="src/cuda/device_fvm.cu",
             validation="tests/test_scheme_blocks.cu -- the selector, with a fail-proof; and the limiter "
                        "against vanLeer.H:70 directly. It asymptotes to 2, not 1: it is a Sweby TVD "
                        "limiter, not a [0,1] blend, and writing the clamp limitedLinear needs would "
                        "make it a different scheme.",
             note="EVERY interFoam tutorial uses `div(phi,alpha) Gauss vanLeer`. brae has limitedLinear/V, "
                  "vanAlbada, LUST, linearUpwind/V -- not vanLeer. Small, and blocking."),
        dict(name="interFoam_alphaCourantNo", of_symbol="alphaCourantNo",
             of_file="applications/solvers/multiphase/VoF/alphaCourantNo.H",
             classification="CONFIGURATION", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/cfdTools/general/time_controls.cuh",
             brae_target="src/finiteVolume/cfdTools/general/device_alpha_courant.cu",
             validation="tests/test_alpha_courant_cpp.cu. Four fail-proofs: defaulting maxAlphaCo, "
                        "pos for pos0, masking the mean's denominator, and dropping the alpha limit "
                        "from setDeltaT. ON THE DEVICE, tests/test_device_alpha_courant.cu repeats "
                        "three of them plus the boundary half of surfaceSum (0.0035 against 0.0043), "
                        "with the fixture landing 40 cells EXACTLY on the band's edges so pos and pos0 "
                        "differ. Agreement is 1e-14 relative and NOT bit-exact, and that is itself the "
                        "measurement: a max over per-cell sums would be bit-identical if the sums were, "
                        "so the one ULP proves sumPhi is a gather here and a scatter on the host.",
             note="A SECOND Courant number, over the interface cells only, because the ordinary one is a "
                  "global max set by whatever corner runs fastest -- usually nowhere near the "
                  "interface -- and MULES does not stop the interface being advected more than a cell "
                  "per step. Landed alongside the existing time_controls, which already gives every "
                  "transient solver CourantNo/setInitialDeltaT/setDeltaT (brae_pimpleFoam stopped "
                  "refusing adjustTimeStep when that landed). Four things: maxAlphaCo is get<scalar> "
                  "with NO default where maxCo defaults to 1, so the same controlDict is refused by "
                  "one reader and not the other; nearInterface() is a 0/1 MASK over the CLOSED band "
                  "[0.01, 0.99] (pos0 is 1 at exactly zero), so the two cells at its edges -- the ones "
                  "an advancing interface is passing through -- are counted; meanAlphaCoNum divides by "
                  "gSum(V) over the WHOLE mesh, not the interface volume; and with no interface "
                  "alphaCoNum is exactly 0 so maxAlphaCo/(0+SMALL) lets maxCo alone set the step, "
                  "which a differently-guarded division would turn into a freeze. FEEDING IT BACK "
                  "INTO deltaT found two more things, both silent. Time::run() is `value() < endTime - "
                  "0.5*deltaT`, tested BEFORE setDeltaT with the previous step -- nothing in brae "
                  "implemented it, brae_interFoam bounded its loop by a step COUNT computed from the "
                  "initial deltaT, and damBreak at `endTime 0.004` ran out to t = 0.054, thirteen times "
                  "past the end of the case, under a comment claiming the end time stopped it. And "
                  "Time::setDeltaT takes `adjust = true` by default (Time.C:1178) and calls "
                  "Time::adjustDeltaT, which under `writeControl adjustable` SHORTENS the step to land "
                  "on the next write time: damBreak writes every 0.05, so OpenFOAM's first step is "
                  "0.05/417 = 0.000119904 where brae's unclamped setDeltaTVoF gave 0.00012. Both are "
                  "ported (WriteCadence + adjustDeltaT in time_controls.cuh, and note that "
                  "Time::writeControlNames tabulates wcAdjustableRunTime TWICE, as `adjustable` and as "
                  "`adjustableRunTime` -- damBreak uses the short spelling and matching only the long "
                  "one made the whole clamp a no-op on the very case it was measured against). "
                  "tests/interfoam_dambreak_clock_vs_openfoam.sh now holds the clock against real "
                  "OpenFOAM on the tutorial AS IT SHIPS: thirteen steps over one write interval, deltaT "
                  "BIT-IDENTICAL, 0.000e+00 on every step at 17 digits, both landing exactly on 0.05 "
                  "-- and `brae_interFoam -device`, which refused adjustTimeStep until this landed, is "
                  "bit-identical too. Over those thirteen steps the Courant number never binds at "
                  "maxCo 1, so this gate is the cap, adjustDeltaT and the loop bound; the Courant "
                  "feedback is the device gate's adaptive arm at maxCo 0.0015. GETTING TO A TRUE "
                  "NUMBER TOOK THREE PRINTFS: at OpenFOAM's default six figures it read 3.2e-06 (half "
                  "an ULP of 0.00119047619 logged as 0.00119048); at writePrecision/timePrecision 14 "
                  "it read 2.389e-09 and stayed there, identical on host and device, while the fields "
                  "moved four orders closer -- which was brae's own `%.9g`; at 17 digits on both "
                  "sides it is zero. A comment in the gate had explained the 2.389e-09 as field "
                  "disagreement arriving through the Courant number. It was a printf."),
        dict(name="interFoam_createFields", of_symbol="createFields",
             of_file="applications/solvers/multiphase/interFoam/createFields.H",
             classification="CONFIGURATION", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/interFoam/inter_case_cpp.cuh",
             brae_target="src/applications/solvers/interFoam/device_inter_case.cu",
             validation="tests/interfoam_createfields_vs_openfoam.sh -- damBreak's OWN case, prepared "
                        "the way the tutorial does. damBreak ships 0.orig and NO mesh, so the fixture "
                        "is a PROCEDURE (blockMesh then setFields), not a directory this repo can "
                        "check in; the gate runs OpenFOAM's own utilities and SKIPs where OpenFOAM is "
                        "absent. 2268 cells, 5 patches, 324 water cells.",
             note="ONE translation from case to fields, shared by the gate and by the solver -- a "
                  "private copy in the driver is the defect rhoSimpleFoam's mirror wrote down: the "
                  "gate proves the step, the driver feeds it something else, nothing compares them. "
                  "Three things: the alpha field's NAME comes from `phases (water air)`, so a "
                  "hard-coded alpha1 finds no file on any shipped case; phi is READ when present "
                  "(createAlphaFluxes.H) and only computed from U when it is not, so a restart "
                  "continues from the written flux rather than discarding the continuity error the "
                  "last pressure corrector drove down; and fvSchemes KEYS contain parentheses and "
                  "commas, so FoamDict cannot look them up -- they are read as text through "
                  "scheme_parse's fvSchemesBlock, the same route every other brae solver uses. "
                  "simulationType RAS is read here and handed to interFoam_turbulence, which holds what "
                  "is run (kEpsilon, both lineages) and what is refused. THIS NOTE USED TO SAY a "
                  "non-laminar simulationType is refused because interFoam's turbulence is `a MIXTURE "
                  "model handed the blended rho`; that was written from memory and is wrong for 15 of "
                  "the 17 turbulent tutorials (incompressibleInterPhaseTransportModel.C:99-106 builds "
                  "the ordinary single-phase model by default). solvers/p_rgh and p_rghFinal are read "
                  "as the case names them: PCG with DIC, or GAMG with its controls (interFoam_GAMG); "
                  "anything else still substitutes PBiCGStab under a notice. constant/dynamicMeshDict is read here "
                  "too (interFoam_solidBodyMotion), and p_rgh's need for a reference (interFoam_pressureReference). "
                  "GRADIENT SCHEMES (2026-09-19): each fvc::grad is resolved by the name its call site asks for "
                  "-- grad(U), grad(alpha.<phase1>), grad(alpha.<phase2>), grad(p_rgh), grad(pcorr), grad(rho), "
                  "and interfaceProperties' `nHat` -- then `default`, and carried to that site as a GradChoice "
                  "(gradSchemes/grad_choice.cuh): the vanLeer limiters, snGrad's and the laplacians' corrections, "
                  "CorrectPhi and the curvature. The host takes `Gauss linear`, `leastSquares` and `cellLimited` "
                  "over either; anything else is refused by name, and so is a non-Gauss grad(magSqr(U)) under "
                  "limitedLinear (its limiter is ported Gauss only). GATED in tests/interfoam_dambreak_vs_openfoam.sh "
                  "`gradLsqLimited` on a sheared, predictor-on, moving-start damBreak (every site live there, "
                  "alpha 1.9e-14), each site broken once and red, and `nHatLimited` at the small step (alpha "
                  "5.6e-13). A LIMITED nHat is not held at the big step, and not for a defect: brae's limited K "
                  "from OpenFOAM's own 17-digit post-MULES alpha agrees to 3e-16 relative, but the limiter turns "
                  "MULES' bulk round-off (alpha 1 +- 1e-7) into 1.1e-06 of p_rgh a step later. NOT CLAIMED: any "
                  "non-Gauss entry across a cyclic or cyclicACMI, or leastSquares across any coupled patch "
                  "(refused; grad(U) cellLimited alone is gated across a cyclicAMI); the device, whose operators "
                  "are Gauss linear and which refuses every other entry."),
    ],

    "rhoSimpleFoam": [
        dict(name="rhoSimpleFoam_main", of_symbol="main",
             of_file="applications/solvers/compressible/rhoSimpleFoam/rhoSimpleFoam.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam.cu",
             validation="tests/rho_simple_end_to_end_vs_openfoam.sh -- THE WHOLE CASE in _cpp against real OpenFOAM. 112k-cell sbMatched TURBULENT, with the fixture own kEpsilon and its own flowRateInletVelocity, `consistent yes` + `transonic yes` so the path exercised is pcEqn.H transonic. 400 iterations from the same start. EIGHT FIELDS ARE GATED, not one: U 2.364262e-05, p 1.230477e-05, T 4.544945e-06, rho 1.341389e-05, k 7.648476e-05, epsilon 1.373127e-04, nut 4.302340e-05, alphat 4.866781e-05. The same binary runs tests/rho_squarebend_vs_openfoam.sh, which is the wider of the two on every field and is where the bounds come from: p 4.850e-04, T 1.096e-04, U 4.410791e-04, rho 4.341e-04, k 1.358e-03, epsilon 1.652e-03, nut 1.166e-03, alphat 9.666e-04. Bounds p 1.0e-3, T 3.0e-4, rho 1.0e-3, turbulence 3.0e-3, U 5e-4. EACH BOUND IS ASSERTED ONLY WHERE THAT FIELD MOVES, decided per field and per fixture by gateIfItMoves(): the bound is applied when the START state misses OpenFOAM answer by at least 10x it, and otherwise the field is printed with the reason it is not held. Measured start-state ratios on squareBend: p 437x, T 101x, rho 453x, and 83x on each of k, epsilon, nut and alphat -- so every bound above separates a converged solver from one that did nothing. This REPLACES an earlier arrangement in which U alone was gated and everything else printed `(reported, see the control)`; the stated reason was that on this fixture p and T barely move, and the measurement contradicts it -- p starts 2.1698e-01 away from OpenFOAM answer, 434x its bound. THE START-STATE CONTROL ON U CANNOT FAIL, and that is a property of the fixture rather than of the solver: sbMatched/0.orig/U is `uniform (0 0 0)`, so relL2 against OpenFOAM is exactly 1.0 for any code state. The other seven ratios are real measurements and are not fixed by construction. THE WIDEST FIELDS ARE squareBend k and epsilon at ~1.6e-03, and that case asks for `Gauss limitedLinear 1` on div(phi,k) and div(phi,epsilon) while the closure discretises both upwind regardless -- fvm::div plain overload, where the weighted one exists and the incompressible path uses it. That difference is what ~1.6e-03 is worth at convergence, and it is an open finding rather than a tolerance. RUN TO CONVERGENCE, NOT TO A TUTORIAL endTime: comparing at an endTime compares trajectories, and 400 is where both codes have stopped moving. AN UNPORTED RAS MODEL IS REFUSED BY NAME, and so is an unported nut wall function -- both asserted by the gate, each with a negative control proving createFields still accepts the case own. WHAT THIS FIXTURE NEVER REACHES: limitTemperature, any fvOptions (sbMatched ships no fvOptions and no MRFProperties), MRF, the kOmegaSST branch, the laminar branch, the SUBSONIC pEqn.H branch and the rho.relax() it guards. Each is a live branch of the driver that this gate does not exercise. tests/rho_angledduct_vs_openfoam.sh would reach the fvOptions and porosity paths and is deliberately NOT registered (PORT.md), so those remain ungated end-to-end.",
             note="The driver branches on simple.consistent(): pcEqn.H for SIMPLEC, pEqn.H otherwise. "
                  "Both must be ported -- the existing solver reached only one of them by case. Order is "
                  "UEqn, EEqn, then p; turbulence->correct() LAST, after the pressure solve. "
                  "LINEAR-SOLVER POLICY ON THE CUDA ARM (bench/rhoSimpleFoam/FASTPATH.md FP-1, 2026-09-12): "
                  "wherever the case names `smoothSolver` with a GaussSeidel-family smoother -- on U since "
                  "2026-09-08, on e/h, k, epsilon and omega since FP-1 -- the sweep runs in COLOUR order "
                  "through the fused multicolour engine (deviceColourGaussSeidelFused) under the entry's own "
                  "stop rule, announced per field as `approximated` (a different iterate after n sweeps, the "
                  "same converged solution); BRAE_U_SOLVER=ofOrder / BRAE_GS_ORDER=ofOrder run OpenFOAM's "
                  "index order instead, which is what the gates that hold an exact iterate at a loose relTol "
                  "pin. Measured on squareBendLiq at 112k: the e solve 6.8 -> 1.3 ms per iteration and the "
                  "turbulence block 14.7 -> 4.0, gated by tests/scalar_colour_gs_vs_openfoam.sh (EXACT "
                  "1.3e-12 at 1e-14, the relTol-0.1 control 1.3e-02) and test_colour_gs_fused arm (m)."),
        dict(name="createFields", of_symbol="createFields.H",
             of_file="applications/solvers/compressible/rhoSimpleFoam/createFields.H",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoCreateFields.cu",
             validation="tests/rho_createfields_vs_openfoam.sh -- against OpenFOAM's OWN createFields.H, obtained by running rhoSimpleFoam -postProcess with writeObjects(phi,rho); postProcess.H builds the field set without solving. On a developed 112k-cell sbMatched state: rho 6.1e-16, phi 2.1e-15, initialMass 3.9e-16, psi bounded at 1e-14 -- but against 1.0/(R*T), the SAME expression the implementation evaluates, so it verifies brae against its own formula and reads no OpenFOAM psi at all; psiBnd, which the transonic pressure branch is built on, is unchecked. initialMass likewise multiplies OpenFOAM rho by brae OWN g.V(), so a volume error cancels on both sides and OpenFOAM domainIntegrate is never read. rho and phi are gated on INTERNAL fields only: rho patch values and phi patch fluxes, the inlet mass flux among them, are compared to nothing, and a restart reads both fields byte-exact. THE CONTROL is the point: interpolate(rho)*flux(U), which is pEqn.H form and brae fvc::rhoFlux, reads 3.7e-04 against the same oracle -- 11 orders worse -- so the gate discriminates the two flux forms instead of passing either. It also asserts the fixture rho is non-uniform, because a uniform rho makes the two forms algebraically identical and the shipped 0.orig is uniform.",
             note="rho is READ_IF_PRESENT and falls back to thermo.rho() -- a restart reads it, a cold "
                  "start computes it, and getting that backwards was a past defect. p is a REFERENCE into "
                  "the thermo (thermo.p()), not an independent field. thermo.validate(.., h, e) refuses a "
                  "thermo whose energy is neither. pressureControl carries pMin/pMax/pRefCell/pRefValue "
                  "from the SIMPLE dict. phi comes from compressibleCreatePhi.H, so it is rho*flux(U)."),
        dict(name="UEqn", of_symbol="UEqn.H",
             of_file="applications/solvers/compressible/rhoSimpleFoam/UEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu",
             validation="tests/rho_ueqn_vs_openfoam.sh -- against OpenFOAM's OWN assembled momentum matrix, via the tools/dumpPEqn stage harness (stage_rAU = 1/UEqn.A(), stage_UIC, stage_UBC, stage_muEff, stage_Uass) at SIMPLE iteration 1 on 112k-cell sbMatched: rAU 6.13e-15, internalCoeffs 7.1e-15, boundaryCoeffs 4.89e-16 -- but ALL THREE ARE GATED AT 1e-10, five orders looser than the figures suggest, so the bound would pass a far worse assembly than the one measured. The per-patch inlet/outlet/walls figures are printf, not bounded; on this fixture the outlet iC/bC and the walls bC are identically zero in both codes, where relL2 degenerates to an absolute norm. OpenFOAM's own muEff is INJECTED so the number measures the assembly rather than the closure -- and so is OpenFOAM's VELOCITY, internal field and every written patch value, which the text did not say. The injection is a near no-op here (brae's own muEff sits at 1.9e-15 against OpenFOAM's) and the closure is no longer unported: compressible kEpsilon is ported and separately gated, so the live reason for injecting is isolation, not absence. THE CONTROL: assembling with the kinematic nu_eff -- the incompressible divDevReff -- reads 6.2e-01, fourteen orders worse, and forcing that form into the implementation fails the gate at 6.2e-01, so it discriminates the one thing that distinguishes this solver's momentum equation from simpleFoam's."
                " _cpp AGAINST OPENFOAM'S OWN MOMENTUM MATRIX ON A DEVELOPED FIELD (tests/rho_limitedlinearv_cpp_vs_openfoam.sh, gasMixing restarted at OpenFOAM's iteration 5, off-diagonals stage_UUpper/ULower, `Gauss upwind` as the control at 1.06e-14): limitedLinearV 3.31e-03 -> 1.76e-14 and limitedLinear 4.63e-02 -> 3.22e-13. Two causes read out of the source: the limiter's gradient is built from U's patch values BEFORE updateCoeffs (gaussConvectionScheme.C:84 takes the weights, :89 builds the fvMatrix whose constructor calls updateCoeffs, fvMatrix.C:396) where brae refreshed first; and `Gauss limitedLinear` on a VECTOR is NVDTVD + limitFuncs::magSqr (LimitedScheme.H:188), so its gradient is fvc::grad(magSqr(U)) under the key grad(magSqr(U)) -- default, leastSquares there -- where brae hardcoded Gauss. Before this the device had been validated against a _cpp reference that was itself unvalidated for both schemes."
                   " CUDA: the device momentum limiters took the refreshed boundary (RhoMomentumInput::UxPreUpdateBnd "
                   "now carries the step's pre-updateCoeffs snapshot, required under limitedLinear/V) and a plain Gauss "
                   "grad(magSqr(U)); measured on gasMixing/injectorPipe restarted at OpenFOAM's iteration 5 the "
                   "momentum off-diagonals were 3.3e-03 off the host and read 1.2e-15 after, with test_rho_ueqn_cuda "
                   "(one shared boundary array) at 3.2e-14 either way -- the twin cannot see the snapshot, the "
                   "restarted end-to-end gate can.",
             note="fvm::div(phi,U) + MRF.DDt(rho,U) + turbulence->divDevRhoReff(U) == fvOptions(rho,U), "
                  "solved against -fvc::grad(p). divDevRhoReff is the COMPRESSIBLE form (rho-weighted, "
                  "dev2 transpose term) and is NOT the incompressible divDevReff that simpleFoam uses. "
                  "TARGET IS rhoUEqn.cu, NOT UEqn.cu: brae puts every source directory on ONE include "
                  "path, so a companion UEqn.cuh here would resolve to simpleFoam/UEqn.cuh or the "
                  "reverse, whichever the compiler saw first. simpleFoam already owns UEqn.cuh and "
                  "pEqn.cuh. The host references are already rhoUEqn_cpp/rhoPEqn_cpp/rhoEEqn_cpp for "
                  "the same reason; the device targets follow them."),
        dict(name="EEqn", of_symbol="EEqn.H",
             of_file="applications/solvers/compressible/rhoSimpleFoam/EEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu",
             validation="tests/rho_eeqn_vs_openfoam.sh -- against OpenFOAM's OWN assembled energy equation via the tools/dumpPEqn harness (stage_Ekp, stage_he, stage_eD, stage_eSrc, stage_alphaEff, stage_Upred) at SIMPLE iteration 1 on 112k-cell sbMatched: Ekp 6.4e-16 (and 6.1e-16 on every patch), EEqn.D() 3.8e-15, source+boundaryCoeffs 3.9e-15, interior and boundary cells both at machine precision. THE INLET IS NEUTRALISED: the script replaces sbMatched flowRateInletVelocity with a plain `fixedValue uniform (1 2 3)`, ~3.7 m/s against the case own ~523 m/s, so this gate says nothing about the inlet and the energy it carries. NO OFF-DIAGONAL HAS AN OPENFOAM ORACLE: dumpPEqn writes none for this equation and the test compares none, where the sibling momentum gate does. Bounds are 1e-12 on Ekp and 1e-10 on D and source; the per-patch and interior/boundary figures are printf, and the interior/boundary one is an ABSOLUTE L2 with no denominator, so machine precision is not the quantity it reports. alphaEff, he AND OpenFOAM solved U are INJECTED so the number measures the assembly, not the unported compressible turbulence closure or the unported energy boundary types. THE CONTROL: the `h` arm (K = 0.5|U|^2) reads 1.0 against stage_Ekp and builds a convection term differing from the `e` arm by 100% of its own magnitude; forcing the h arm into the implementation fails the gate. NOTE the control is taken on the UNBOUNDED convection term on purpose -- `div(phi,Ekp)` is bounded, and at iteration 1 the bounded subtraction removes the near-uniform p/rho that IS the difference between the arms (|KE div| 1.3e-04 bounded against 1.4e+01 unbounded), so the assembled source at this state cannot discriminate them."
                " THE CORRECTED LAPLACIAN'S OWN GRADIENT: correctedSnGrad::fullGradCorrection resolves gradScheme::New(mesh.gradScheme('grad(' + name + ')')), i.e. grad(e)'s own gradSchemes entry, and brae took a hardcoded Gauss gradient there. On gasMixing/injectorPipe (default leastSquares, snappyHexMesh) the energy source was 2.8e-07 and the solved he 2.2e-06 off OpenFOAM's own at a developed restart; with the entry honoured 6.3e-11 and 4.7e-16, and T 8.6e-07 -> 9.6e-13 end to end (tests/rho_gasmixing_vs_openfoam.sh). NOT gated on sbMatched: its mesh is orthogonal to 0.78 degrees, the correction is ~0 there, and a leastSquares arm passed with the defect present and absent alike -- measured, then removed as vacuous."
                   " test_rho_eeqn_cuda, the device twin, now assembles `limitedLinear` on BOTH convection terms "
                   "against the host (diag 5.7e-13, upper 2.4e-12, lower 1.4e-12, source 1.2e-12 on matrixDumpAsym "
                   "and pitzDailyTurb, e and h) where it used to assert a REFUSAL the port had since made stale -- "
                   "three registrations red for that reason. WHAT THE ARM FOUND: with the twin's synthesized "
                   "he = 2e5 + 6e4*frac(x) the two arms read upper 5.8e-02 apart on 1228 of 24170 faces, every "
                   "one with he[N] == he[P] -- NVDTVD's 0/0 branch, where r = 2000*sign(gradcf)*sign(0) - 1 is "
                   "decided by the round-off sign of gradcf (the gradients agree to 4e-15); the same phenomenon "
                   "gasMixing/injectorPipe shows from rest. The field now carries an incommensurate sine and the "
                   "twin asserts zero plateau faces; LUST stands in as the unported-scheme refusal control.",
             note="The kinetic-energy source DIFFERS BY ENERGY VARIABLE: he==e uses Ekp = 0.5|U|^2 + p/rho, "
                  "he==h uses K = 0.5|U|^2. Picking one is a wrong equation for the other thermo. MRF adds "
                  "fvc::div(MRF.phi(), p). thermo.correct() runs at the END and is what updates T, psi, mu "
                  "and alpha for every consumer downstream."),
        dict(name="pEqn", of_symbol="pEqn.H",
             of_file="applications/solvers/compressible/rhoSimpleFoam/pEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoPEqn_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu",
             validation="tests/rho_peqn_vs_openfoam.sh -- BOTH branches, by running OpenFOAM twice on 112k-cell sbMatched with transonic no and transonic yes (and consistent no, so the driver reaches pEqn.H rather than pcEqn.H). Subsonic: rAU 6.1e-15, rhorAUf 4.8e-15, HbyA 2.2e-15, phiHbyA 6.4e-12, after adjustPhi 6.4e-12 -- BUT adjustPhi NEVER RUNS on this fixture, so that figure re-measures the same phiHbyA rather than testing the step: rhoPEqn_cpp.cu:73-76 returns immediately when any p patch fixes a value, and the outlet is fixedValue. The fatal `continuity error cannot be removed` path is unreached on every arm, and the `adjustable` mask the step consumes is inert here, pEqn.D() 3.3e-15, source 6.5e-14. Transonic: the same, plus phid 6.4e-12 and the psi*p subtraction 1.3e-15 (normalised by the pre-subtraction flux, because psi*p is rho for a perfect gas so the subtraction is a near-total cancellation -- residual 1.8e-10 of an inflow of 1.6e-04). The transonic branch had NO stage harness before this gate; one was added to tools/dumpPEqn. OpenFOAM own muEff, Uass, Upred, psi and rho are injected so the number is about pEqn.H and not about the turbulence closure or thermo.correct(). CONTROLS, each verified to fail: the other branch does not reproduce this one; dropping fvm::div(phid,p) fails transonic D at 3.1e-07; omitting pEqn.relax() on the transonic branch fails D at 1.1e-07 and source at 4.4e-06.",
             note="TWO BRANCHES, both required: simple.transonic() builds fvm::div(phid,p) with phid = "
                  "(psi/rho)*phiHbyA and subtracts fvc::interpolate(psi*p)*phiHbyA/interpolate(rho); the "
                  "subsonic branch runs adjustPhi and has no div term. closedVolume comes from adjustPhi "
                  "and drives the psi-weighted mass correction after the solve. rho.relax() is SKIPPED "
                  "when transonic. pressureControl.limit(p) may clip p and then requires "
                  "correctBoundaryConditions()."),
        dict(name="pcEqn", of_symbol="pcEqn.H",
             of_file="applications/solvers/compressible/rhoSimpleFoam/pcEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoPcEqn_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoPcEqn.cu",
             validation="tests/rho_pceqn_vs_openfoam.sh -- BOTH branches, with consistent yes so the driver reaches pcEqn.H. The comparison is taken from a RESTART at iteration 20, not from iteration 1: sbMatched starts from a uniform p, where fvc::grad(p) is analytically zero and BOTH SIMPLEC corrections are no-ops (OpenFOAM own HbyA correction moves it by 9e-10 there, pure round-off), so an iteration-1 gate would report machine precision while testing none of SIMPLEC. On the developed state the correction moves HbyA by 2.4e-01 and rAtU is 6.9x rAU. Results: rAU 5.1e-15, rAtU 5.2e-15, rhorAtU 5.5e-15, HbyA 2.1e-15, constrainHbyA boundary 6.7e-16, phiHbyA 1.4e-15, phid 1.4e-15, pcEqn.D() 4.2e-15; the two fvc::grad(p)-derived quantities sit at 1.4e-10 and 1.2e-09 and are bounded at 1e-8 with the alternatives ruled out in the test comments. CONTROLS, AS REGISTERED -- and this list was previously overstated. What the gate actually asserts is that SIMPLEC is live (rAtU differs from rAU by more than 1e-3, test:235) and that the plain-SIMPLE BUNDLE disagrees on the assembled system: D and source, each bounded > 1e-8 (test:389-390). That bundle swaps rAU for rAtU AND rhorAUf for rhorAtU AND drops the flux correction together, so it does not isolate any one of them. The figures 8.8e-01 and 2.4e-01 are printf, not bounds. THERE IS NO ARM THAT MUTATES phiHbyA: the previous text claimed a control `dropping the SIMPLEC flux correction fails phiHbyA at 2.6e-01`, and 2.6e-01 appears nowhere in the tree outside that sentence -- no test produces it. Isolating the two halves of SIMPLEC, which the module header argues are one change, is not gated. FOUND BY THIS GATE: inletOutlet assignable() was false in brae because it derives from mixed, but OpenFOAM OVERRIDES it back to true (inletOutletFvPatchField.H:164) -- constrainHbyA was taking U at every inletOutlet patch, worth 1.3e-03 on HbyA boundary and 3.5e-03 on phiHbyA. Shared code; outletInlet genuinely does inherit false and was correct.",
             note="The SIMPLEC pressure equation, selected by `consistent yes`. Carries its own rAtU = "
                  "1/(1/rAU - UEqn.H1()) and the (rAtU - rAU) flux correction -- the same shape that cost "
                  "real time in simpleFoam, so it is transcribed here rather than re-derived by guess."),
        dict(name="kEpsilon_compressible", of_symbol="kEpsilon<BasicMomentumTransportModel>::correct",
             of_file="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu",
             brae_target="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu",
             validation="tests/rho_kepsilon_vs_openfoam.sh -- against OpenFOAM's OWN kEpsilon, instrumented. tools/dumpKEpsilon is that model with writes added and its equations untouched, registered as kEpsilonDump through makeRASModel, so gradU, divU, GbyNu, G, both diffusivities, the mesh factors the laplacian coefficient is a product of, both off-diagonal sets and both assembled systems (before and after relax/boundaryManipulate) each have an oracle. On 112k-cell sbMatched at SIMPLE iteration 1, given OpenFOAM's own inputs: epsilon 5.0e-15, k 8.9e-16, nut 1.7e-15, alphat 2.3e-15, wall and interior alike, with every intermediate at the same order. THE CONTROL: substituting the MASS flux for the volumetric one in divU -- the one difference between the compressible and incompressible readings of this same templated model -- must be worse by at least 1e3x, measured 5.0e-06 against 5.0e-15. A term sweep drops each of the eight terms in turn and every one of them moves the answer. A short source array for a turbulent inlet is REFUSED by name, with a full-length array as the negative control."
                " FVM::DDT UNDER `Euler` (kEpsilon.C:254,275): tests/rho_kepsilon_vs_openfoam.sh's Euler arm holds the assembled k and epsilon systems to 1e-14 against the instrumented model with `ddtSchemes default Euler`, and reads epsilon D() 8.0e-04 / k source 2.8e-02 with the term zeroed (the fail-proof). Found on gasMixing/injectorPipe, the one rhoSimpleFoam tutorial shipping Euler: with every input to the closure exact, its epsilon diagonal was 5.66e-04 and k source 6.14e-03 off OpenFOAM's own assembly, and OpenFOAM minus brae equalled rho*V/deltaT to 4.8e-09. rho.oldTime() in the source is the closure-time rho on a process's first step (GeometricField::oldTime() copies at first use) and the start-of-step rho afterwards -- measured 1.8e-09 and 5.6e-12 respectively, the other choice 1.3e-03 / 2.9e-04. End to end: tests/rho_gasmixing_vs_openfoam.sh, k 2.2e-12 at the first restarted iteration.",
             note="OpenFOAM has ONE templated kEpsilon.C; the compressible instantiation supplies alpha=1, "
                  "rho as a field, alphaRhoPhi as the MASS flux and a nu that varies with T. TWO FLUXES, "
                  "not one: fvm::div takes the mass flux while divU takes the VOLUMETRIC one, because "
                  "compressibleTurbulenceModel::phi() divides by interpolate(rho). Four defects were found "
                  "by reading the instrumented model's numbers rather than by reasoning -- the closure "
                  "never being given the flux (so fvc::grad(U) read a stale outlet face value), k and "
                  "epsilon never having updateCoeffs called (so every flux-conditional patch contributed "
                  "nothing), the turbulent inlets being frozen at the case file's `value` instead of "
                  "recomputed from U and k, and the diffusion terms being assembled orthogonally while "
                  "the case asked for `corrected`. See PORT.md. FP-2 FUSIONS (2026-09-12): the shared conv-diff assembly (turbulence_transport.cu) subtracts the five laplacian coefficient arrays in one launch, and the device kOmegaSST closure fuses S2+production+G, CD+F1+F2, the gamma/beta blends with the GbyNu limit, and the compressible DEff*rho+nu*rho chain (device_komega_sst.cu, FP-2 block), each kernel text-identical in its expressions to the ones it replaces and __dmul_rn pinning the products that used to cross a kernel boundary -- byte-identical fields and residual lines on aerofoilNACA0012 and squareBend over 5 iterations and on all 18 SST stage dumps. Worth 25 launches per iteration on the aerofoil, inside the run-to-run noise; the closure's cost there is the DILU-preconditioned k/omega solves the case names, not its kernels."),
        dict(name="createFieldRefs", of_symbol="createFieldRefs.H",
             of_file="applications/solvers/compressible/rhoSimpleFoam/createFieldRefs.H",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoCreateFields.cu",
             validation="tests/rho_createfields_vs_openfoam.sh -- psi bounded at 1e-14 against 1.0/(R*T), from the same T that rho came from. NOT AN OPENFOAM COMPARISON, and the previous wording (`exactly`) implied one: the right-hand side is the SAME expression perfectGasPsi evaluates, so this asserts brae agrees with its own formula. No OpenFOAM psi file is read anywhere in the gate. Cells only -- psiBnd, which the transonic pressure branch interpolates to faces, is compared to nothing.",
             note="One line: psi is a const reference to thermo.psi(). It matters because pEqn uses psi "
                  "AFTER thermo.correct() has moved it."),
        dict(name="leastSquaresGrad", of_symbol="Foam::fv::leastSquaresGrad",
             of_file="src/finiteVolume/finiteVolume/gradSchemes/leastSquaresGrad/leastSquaresGrad.C",
             classification="SHARED_NUMERICAL", status="PORTED",
             brae_existing="src/cuda/device_mesh.cu (deviceLeastSquaresGrad)",
             brae_reference="src/finiteVolume/finiteVolume/fvc.cu (fvc::leastSquaresGrad)",
             brae_target="src/finiteVolume/finiteVolume/gradSchemes/",
             validation="tests/leastsquares_grad_vs_openfoam.sh -- the GRADIENT itself, SCALAR and VECTOR "
                        "arms on an orthogonal mesh AND a pitzDaily arm whose boundary faces are SKEWED "
                        "(fvPatch::delta() is the patch-normal projection of Cf - Cn, not Cf - Cn: brae "
                        "used the raw vector and an orthogonal mesh cannot tell the two apart -- 1.18e-01 "
                        "vs 1.84e-12 on pitzDaily, 4.6e-15 either way on squareBend). Scalar and vector "
                        "arms, against OpenFOAM's own grad(T) and grad(U) on validation/rhoSST, each a 2x2 "
                        "with the other scheme as the control: brae leastSquares vs OF leastSquares "
                        "2.5e-13 (scalar) and 2.3e-13 (vector), against OF Gauss linear 2.5e-01, and OF "
                        "leastSquares against brae Gauss 3.3e-01 (device scalar 2.5e-13, device-vs-host "
                        "1.6e-16; no device twin for the vector form yet). "
                        "tests/rho_leastsquares_closure_vs_openfoam.sh -- the CONSUMERS, nine arms on the "
                        "host arm end to end against real OpenFOAM: the closure's CDkOmega and "
                        "corrected-laplacian gradients, the turbulence limiters' own gradient, grad(p) and "
                        "grad(U), each restarted from OpenFOAM's own iteration 5 and each measured "
                        "separately (before -> after: k 3.1e-06 -> 8.7e-12, 1.7e-05 -> 2.9e-12, "
                        "3.6e-06 -> 2.9e-12), with the shipped case as the control, a wrong-scheme control "
                        "(brae's Gauss against OpenFOAM's leastSquares run must differ, 6.3e-08). Two "
                        "fail-proofs, each watched: divDevRhoReff's flag removed, and the limiter's gradient "
                        "back to Gauss -- different arms go red. CUDA, the CLOSURES' grad(k)/grad(epsilon|omega): "
                        "test_rho_kepsilon_cuda's leastSquares arm against the host closure on pitzDaily "
                        "(`corrected` laplacians, so the correction path is live) epsilon 7.9e-16, k 8.4e-16, "
                        "nut 2.7e-15, with the Gauss device answer 3.36e-03 off the leastSquares reference "
                        "(the control); rho_leastsquares_closure_vs_openfoam.sh's lsq arm now runs the CUDA "
                        "arm against OpenFOAM, worst p 2.9e-12 (the same floor as the host and as the shipped "
                        "control). CUDA, grad(p) at its five consumers (RhoStepInput::gradPLeastSq): the "
                        "closure gate's lsqp arm (grad(p) leastSquares explicit on rhoSST) host 2.900e-12, CUDA "
                        "2.901e-12 vs OpenFOAM, the shipped Gauss run 1.42e-05 off the leastSquares oracle; "
                        "tests/rho_step_cuda_lsq.sh (rhoKE and sbMatched, in-gate Gauss control 1.4e-05 / "
                        "4.7e-03); and tests/rho_gradp_lsq_simplec_vs_openfoam.sh, the SIMPLEC+transonic case "
                        "nothing else covers: both arms 2e-11 of OpenFOAM at iterations 1-2 with the shipped "
                        "control at the same floor and OpenFOAM's own two schemes 1.6e-02 apart. THAT GATE "
                        "FOUND A HOST DEFECT: the device arm was 5.7e-12 of OpenFOAM and the host 1.98e-09, "
                        "because fvc::snGrad's non-orthogonal correction (SIMPLEC's phiHbyA term, pcEqn.H:27) "
                        "took a hardcoded Gauss gradient where correctedSnGrad.C:52-55 resolves grad(p)'s own "
                        "entry -- invisible on rhoSST (not SIMPLEC) and at iteration 1 of sbMatched (p uniform). "
                        "Fixed in fvc.cu (snGrad takes the field's scheme); 2.02e-11 after. CUDA, the VECTOR form "
                        "(deviceLeastSquaresGradU: three scalar fits packed as grad(U), leastSquaresGrad.C's "
                        "lsGrad += ownLs*deltaVsf): leastsquares_grad_vs_openfoam's device twin 9.4e-15 of "
                        "OpenFOAM's own grad(U) on sbMatched and 4.2e-13 on pitzDaily, device-host 1.5e-16; its "
                        "consumers -- divDevRhoReff's dev2 term, the corrected laplacian, the limitedLinearV limiter "
                        "and the linearUpwind-named gradient (rho_ueqn_cuda_lsq, sources 8e-16 of the host), both "
                        "closures' production (test_rho_kepsilon_cuda's grad(U) arm: GbyNu 5.5e-16, the Gauss "
                        "GbyNu 5.0e-01 off as the control) -- and end to end rho_leastsquares_closure_vs_openfoam's "
                        "restart arms now on BOTH mirror arms: lsq_none/CUDA p 2.85e-12, lsq_all/CUDA (`default "
                        "leastSquares` with every limiter) omega 2.46e-11 against the host's 2.45e-11.",
             note="OpenFOAM's inverse-distance least-squares fit (leastSquaresVectors.C), not a Gauss sum; "
                  "a case naming it gets it or is refused. fvc::grad(vf) resolves `grad(<name>)` through "
                  "gradSchemes (fvcGrad.C:149), so `default leastSquares` reaches every consumer: the "
                  "limitedLinear limiters' gradient in the energy equation (reached by default) and in "
                  "both turbulence closures' divWithScheme (LimitedScheme.C:51-55), kOmegaSST's CDkOmega "
                  "and the k/omega|epsilon corrected-laplacian corrections (gradKLeastSq), grad(p) (the "
                  "velocity correction pEqn.H:86 / pcEqn.H:99, SIMPLEC's HbyA pcEqn.H:30,65, the non-orth "
                  "corrections) and grad(U) (divDevRhoReff's dev2 term, the closures' production, "
                  "validate()'s correctNut) -- the VECTOR form, lsGrad_ij = ownLs_i*deltaVsf_j, sharing "
                  "one leastSquaresInvDd with the scalar one. Four of those consumers computed the Gauss "
                  "gradient under the case's own scheme name until this. The DEVICE computes the scalar "
                  "form (deviceLeastSquaresGrad): the energy limiters' gradient, both closures' limiter "
                  "gradient, the closures' corrected-laplacian correction (turbulence_transport.cu, "
                  "TransportScheme::gradFieldLeastSq) and kOmegaSST's CDkOmega (kOmegaSST.cu) take it under "
                  "the case's scheme, and so do grad(p)'s five consumers (the momentum source, U = HbyA - "
                  "rAtU*grad(p), SIMPLEC's HbyA correction, both pressure branches' non-orth corrections), the "
                  "energy equation's non-orth correction (rhoEEqn.cu, grad(he)'s base and cellLimited "
                  "coefficient) and grad(U)'s VECTOR form at divDevRhoReff, the corrected laplacian, the momentum "
                  "limiters and both closures' production (deviceLeastSquaresGradU) -- the CUDA mirror arm "
                  "refuses nothing of leastSquares any more; the linearUpwind-NAMED gradient under leastSquares "
                  "is refused by the shared parse on both arms, and so is any limiter brae does not apply "
                  "(cellMDLimited, faceLimited, faceMDLimited) on grad(U), grad(p), grad(magSqr(U)), grad(he) and "
                  "grad(Ekp|K) -- the parse recorded it and the shared reader only warned, so the unlimited gradient "
                  "ran under the case's scheme name (tests/eeqn_limitedlinear_vs_openfoam.sh's cellMDLimited arm). "
                  "A cellLimited grad(p) (parseFieldGradScheme's "
                  "cellLimitK for p, applied nowhere until this) now limits every grad(p) consumer on both arms, "
                  "fvc::snGrad's correction included -- tests/rho_gradp_lsq_simplec_vs_openfoam.sh's lim arm, "
                  "sbMatched with `grad(p) cellLimited Gauss linear 1`: host 2.02e-11, CUDA 2.48e-11 of OpenFOAM "
                  "where OpenFOAM's own Gauss and cellLimited runs are 5.4e-02 apart (the limiter bites; the "
                  "unlimited arms equal the Gauss run, so that number is the fail-proof). OPEN: the "
                  "incompressible simpleFoam pEqn_cpp.cu "
                  "calls fvc::snGrad without the scheme argument and so still builds SIMPLEC's correction from "
                  "Gauss under a leastSquares grad(p). The non-orth corrections and the gradient linearUpwind NAMES are ported "
                  "but not exercised by rhoSST (orthogonal laplacians, upwind divs). "
                  "SPEED, FP-3 (bench/rhoSimpleFoam/FASTPATH.md, 2026-09-12), all three levers "
                  "bit-identical end to end -- a reference binary with the FP-3 files at HEAD against the "
                  "new one wrote byte-identical fields and residual lines on injectorPipe and squareBend: "
                  "(1) THE INVERTED dd TENSOR IS THE MESH'S, not the gradient call's. OpenFOAM builds "
                  "leastSquaresVectors once per mesh as a MeshObject and invalidates it on a move; brae "
                  "rebuilt it inside every call -- on gasMixing/injectorPipe at 74,650 cells that was 10 "
                  "lsqInvDdKernel launches and 1.99 of the iteration's 15.79 GPU ms. It now lives on the "
                  "DeviceMesh (lsqInvDdFor, dropped in refreshDeviceMeshGeometry where OF invalidates its "
                  "MeshObject), 0.01 ms/it; BRAE_LSQ_INVDD=recompute restores the rebuild. (2) THE FUSED "
                  "FIT, deviceLeastSquaresGradFused, up to three fields in one launch -- the least-squares "
                  "twin of deviceGaussGradFused, held to memcmp per field by tests/test_lsq_grad_fused.cu "
                  "(ctest lsq_grad_fused; n=1,2,3, sheared and empty-patch meshes, one-ulp "
                  "cross-contamination controls, the cache's per-mesh control; fail-proof RUN, 10 arms "
                  "red). Its raw form writes into caller-owned slices, so deviceLeastSquaresGradU fills "
                  "the 9*nC grad(U) tensor in ONE launch where it took three fits, nine device-to-device "
                  "copies and three cudaStreamSynchronize. (3) ONE GRADIENT PER (base scheme, cellLimited "
                  "coefficient) PAIR an assembly asks for: the limitedLinear limiter, linearUpwind's "
                  "correction and the corrected laplacian all read the case's grad(<field>) entry, so the "
                  "energy assembly fitted grad(he) twice and each closure its own field twice. Net on "
                  "injectorPipe: 15.79 -> 13.73 GPU ms/it, four phases 19.1 -> 17.3 ms/it wall, EEqn 3.4 "
                  "-> 2.8 and turbulence 4.5 -> 3.6. The row's UEqn target (40 ns/cell) is NOT met at 56: "
                  "that case names grad(U) cellLimited Gauss linear 0.99, so the momentum phase runs no "
                  "least-squares fit at all and its cost is cellLimitGradKernel (FP-4)."),

        dict(name="fvm_ddt_closure", of_symbol="Foam::fv::EulerDdtScheme<Type>::fvmDdt",
             of_file="src/finiteVolume/finiteVolume/ddtSchemes/EulerDdtScheme/EulerDdtScheme.C",
             classification="SHARED_NUMERICAL", status="PORTED",
             brae_files="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu, "
                        "src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST_cpp.cu "
                        "(the term, per equation); src/applications/solvers/common/scheme_parse.cuh "
                        "(parseDdtScheme); src/applications/solvers/rhoSimpleFoam/rhoSimpleFoamDriver_cpp.cu "
                        "(StepInput::ddtEuler, rDeltaT, firstIteration). CUDA: "
                        "src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu (inside the "
                        "reaction kernels, at the reference's position), "
                        "src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST.cu (ddtKernel after "
                        "the shared reaction), src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam.cu "
                        "(RhoSolverFields::rhoOld written before the closure hook), rhoTurbulenceHook.cu "
                        "(TurbulenceHookOptions::rDeltaT)",
             validation="tests/rho_kepsilon_vs_openfoam.sh and tests/rho_komegasst_vs_openfoam.sh, each with an "
                        "arm that switches sbMatched to `ddtSchemes default Euler` and holds the assembled k and "
                        "epsilon|omega systems to 1e-13 against the instrumented models (fail-proofs: the term "
                        "zeroed reads 8.0e-04 / 2.8e-02 and 9.6e-04 / 3.0e-02). End to end on the tutorial that "
                        "ships Euler, gasMixing/injectorPipe: tests/rho_gasmixing_vs_openfoam.sh, worst 2.0e-11 "
                        "over iterations 6-8 with both codes restarted from OpenFOAM's own iteration 5. "
                        "CUDA against the _cpp reference: test_rho_kepsilon_cuda's Euler arm (rDeltaT 2, "
                        "rho.oldTime() synthesized distinct from rho) epsilon 3.2e-16, k 8.4e-16, nut 2.0e-15, "
                        "with the term withheld on the device k 4.85e-05 and with rho.oldTime() := rho "
                        "1.79e-05; inside the driver, tests/rho_step_cuda_euler.sh (rhoKE and rhoSST switched "
                        "to Euler, 3 iterations, both firstIteration rules exercised) k 4.4e-13, epsilon 3.1e-12, "
                        "nut 1.8e-11 on kEpsilon, and the in-gate fail-proof with the device term withheld reads "
                        "k 7.87e-05.",
             notes=("rhoSimpleFoam's own equations carry no fvm::ddt; the closures do, because the model is "
                    "shared with the transient solvers, and `steadyState` (five of the six tutorials) makes the "
                    "term an empty matrix (steadyStateDdtScheme::fvmDdt) while `Euler` makes it rho*V/deltaT on "
                    "the diagonal and rho.oldTime()*psi.oldTime()*V/deltaT in the source. brae neither parsed nor "
                    "refused ddtSchemes before this. deltaT is controlDict's LAST entry (the tutorial has two; "
                    "OpenFOAM's dictionary takes the last, and so does brae's). backward, CrankNicolson, "
                    "localEuler and bounded are refused by name (device_ddt.cuh carries their coefficients for the "
                    "transient solvers; this port takes only Euler's). OPEN: the incompressible simpleFoam driver "
                    "does not yet resolve ddtSchemes for its closures. The CUDA rhoSimpleFoam arm computes the "
                    "term in both closures (RhoStepInput::firstIteration mirrors the host rule). A RESTART is not "
                    "a continuation under Euler: OpenFOAM "
                    "itself, restarted from its own written iteration, differs from its continuous run at the "
                    "first restarted iteration by k 3.8e-07 (oldTime() created at first use), so gates restart "
                    "BOTH codes.")),
        dict(name="generalizedNewtonian_compressible",
             of_symbol="laminarModels::generalizedNewtonian<BasicMomentumTransportModel>::correct",
             of_file="src/TurbulenceModels/turbulenceModels/laminar/generalizedNewtonian/generalizedNewtonian.C",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/TurbulenceModels/turbulenceModels/laminar/generalizedNewtonian/generalizedNewtonian_cpp.cu",
             brae_target="src/TurbulenceModels/turbulenceModels/laminar/generalizedNewtonian/generalizedNewtonian.cu",
             validation="tests/rho_generalized_newtonian_vs_openfoam.sh -- OpenFOAM's own squareBendLiqNoNewtonian, "
                        "both arms, against OpenFOAM's WRITTEN generalizedNewtonian:nu as well as U, p, T and rho. "
                        "A: from rest, iteration 1 only (the constructor's nu_, nuMax over the quiescent field). "
                        "B: restart from OpenFOAM's iteration 5 with the tutorial's coefficients, iterations 6-8 "
                        "(nu_ at nuMin in every cell). C: the same restart with nuMin 1e-9, the coefficients in "
                        "generalizedNewtonianCoeffs { powerLawCoeffs { } } beside the flat ones, grad(U) "
                        "cellLimited -- the unclamped branch in every cell and on every patch face. Bounds 1e-10 "
                        "(T, rho 1e-11) except C's interior nu_ at 5e-9, which is OpenFOAM's own floor there "
                        "(OpenFOAM against itself, p solver swapped: 2.2e-10..1.1e-09). NOT CLAIMED: the tutorial "
                        "from rest past iteration 1 -- its first correct() reads the round-off of a plug (strain "
                        "rate ~1e-9 1/s in 21120 cells) and OpenFOAM against itself differs by U 1.04e-03 at "
                        "iteration 2; a converged comparison (nu_ = nuMin everywhere, a constant viscosity) is not "
                        "run; viscosity models other than powerLaw are refused by name.",
             note="The model REPLACES the molecular viscosity: nuEff() is nu_ and linearViscousStress assembles "
                  "rho_*nu_, cells and patch faces, with nu0 = mu/rho_ (the SOLVER's rho). nu_ is STORED: "
                  "built by the constructor from the initial U, rebuilt only in correct() at the end of the "
                  "iteration. Its patch values are the formula on the patch values of nu0 and strainRate, and "
                  "strainRate's come from gaussGrad's boundary correction -- the wall shear. The gate found a "
                  "generic restart defect before it could see the model: brae's inletOutlet/outletInlet/"
                  "freestream took their construction value from inletValue where OpenFOAM keeps the file's "
                  "`value`. See PORT.md, Stage S1 Half B. THE DEVICE ARM CARRIED THE SAME DEFECT until the "
                  "gasMixing/injectorPipe restart gate reached it (tests/rho_gasmixing_vs_openfoam.sh): "
                  "buildDeviceBoundary seeded an inletOutlet face as fixedValue(inletValue), so every evaluate "
                  "before the first flux switch -- the closure's stored-patch-value reconstruction at its first "
                  "step -- returned inletValue; measured as the epsilon equation's off-diagonals 1.6e-04 off the "
                  "host with 100% of it on outlet-adjacent faces, and 8.8e-04 on the two turbulent inlets when "
                  "seeded zeroGradient instead. DeviceBoundary::ioStored now carries the file's `value` (the "
                  "host patch field's, extrapolated where absent) and ioFresh marks it until "
                  "deviceUpdateInletOutlet first runs (inletOutletFvPatchField.C's dictionary constructor: "
                  "valueFraction 0, readValueEntry or extrapolateInternal). Both boundary builders. After it the "
                  "CUDA arm sits at 1.2e-11 of OpenFOAM's own restart over iterations 6-8, the host at 2.0e-11. "
                  "The seed is taken by the CLOSURE boundaries (buildDeviceBoundary's storedIoSeed, dbK and dbEps in "
                  "rhoCreateFields.cu) and is INERT on the solver fields: nothing evaluates U/p/he/T's inletOutlet "
                  "faces before the step's first flux switch, and with BRAE_IO_STORED_SOLVER=1 neither mirror arm "
                  "changes a digit. MEASURED CLOSED by tests/rho_naca_restart_vs_openfoam.sh (aerofoilNACA0012: "
                  "kOmegaSST, inletOutlet T/k/omega, freestream U/p, restarted from OpenFOAM's iteration 100 under "
                  "the gasMixing protocol): host and CUDA both 1.8e-09 of OpenFOAM's own restart (p at the first "
                  "restarted iteration, <= 2.2e-10 elsewhere), CUDA vs host <= 5.2e-13; at the case's own "
                  "tolerances the arms read 2.5e-05 apart, solver stopping points. The 1.16e-03 that "
                  "validation/restart_vs_openfoam.sh once read with a seed was the PRE-MIRROR driver (which that "
                  "gate runs, BRAE_RHOSIMPLEFOAM_MIRROR unset) under the outletInlet/freestreamPressure seed, no "
                  "longer taken for any field. BRAE_IO_STORED=0 disables the seed everywhere (the control)."),
        dict(name="PatchFunction1Expression",
             of_symbol="PatchFunction1Types::PatchExprField<Type>::value",
             of_file="src/finiteVolume/expressions/PatchFunction1/PatchFunction1Expression.C",
             classification="HOST_ONLY", status="PORTED",
             brae_reference="src/finiteVolume/expressions/PatchFunction1/patchExprFunction1.cu",
             brae_target="src/finiteVolume/expressions/PatchFunction1/patchExprFunction1.cu",
             validation="tests/rho_patch_expression_vs_openfoam.sh -- squareBendLiq's own T walls, `type "
                        "expression` with its variables and functions<scalar> table verbatim, both mirror arms "
                        "against OpenFOAM's WRITTEN wall values iteration by iteration and the cells alongside "
                        "(solvers tightened to 1e-14 relTol 0 on both codes, since the expression reads |U_c| "
                        "after the momentum solve). Walls at iteration 1: host 1.21e-13, CUDA 1.21e-13 (22400 "
                        "distinct values around 500.003); iterations 2-3 exactly 0 (both 500); cells over 1-3 "
                        "1.86e-12 on both arms. Bounds 1e-12 walls, 1e-11 cells. THE CONTROL is a `constant 500` "
                        "wall on the host arm, which the same comparison rejects at 3.63e-07 -- six orders above "
                        "the bound. Also asserted: the oracle is non-uniform and above 500 at iteration 1; brae's "
                        "written T carries the dictionary and OpenFOAM restarts from it; six refusals by name "
                        "(sqrt, a functions<scalar> entry referenced, an unregistered field, the expression on U, "
                        "on p, and the entry-less form). tests/test_patch_expr.cu covers the grammar without "
                        "OpenFOAM: precedence and associativity as operator-precedence.m4 declares them, lookups "
                        "in OpenFOAM's order, Foam::max's NaN side, mag() of a vector, 17 refusals by name. "
                        "tests/rho_tutorials_vs_openfoam.sh now RUNS squareBendLiq on both arms (1.9e-12).",
             note="THE SUBSET, and nothing else: numbers, + - * /, unary minus, parentheses, identifiers that "
                  "are `variables` entries or registered fields (the PATCH value, getField), internalField(x), "
                  "snGrad(x) (the patch class's own virtual on the host; the fixedValue formula on the CUDA arm "
                  "for the patch's own field only), mag(vector), max, min, time(), deltaT(), arg(), pi(). "
                  "OpenFOAM's grammar has ~60 more rules; every other token -- another function, a comparison, "
                  "?:, .x component access, vector arithmetic, a functions<> entry referenced, name{where} "
                  "remote variables -- is refused BY NAME at parse or first evaluation. WHERE IT IS EVALUATED "
                  "is the port: uniformFixedValue::updateCoeffs runs inside fixedEnergy::updateCoeffs -> "
                  "Tw.evaluate(), once per iteration (the first fvm:: term's fvMatrix constructor; later ones "
                  "find updated()), over U AFTER the momentum solve and T's cells and T's own patch value from "
                  "the previous iteration -- so both steps evaluate it immediately before energy_boundary's "
                  "updateEnergyBoundaryCoeffs. On T ONLY: an expression on any other field is refused by name "
                  "(that field's own updateCoeffs is where OpenFOAM would evaluate it, and no brae step "
                  "reproduces that); the entry-less form (no `value`) is refused because OpenFOAM evaluates it "
                  "at construction over fields brae has not built. The CUDA arm evaluates the same host code "
                  "on the patch's cells downloaded at the assembly and pushes the result into dbT's refValue "
                  "(operator==), which deviceBCValue then exposes -- the flowRate Function1's arrangement. "
                  "Foam::max is (b < a) ? a : b, so max(NaN, 0) is 0 -- transcribed, not std::max; the walls "
                  "reach exactly 500 from iteration 2 because max((500 - 500.003)/500, 0) is 0 and "
                  "par1*T_c*0 is 0. heBoundaryTypes dispatches on the CLASS: uniformFixedValue is a "
                  "fixedValueFvPatchField, so he gets fixedEnergy seeded from T's `value`; createFields_cpp "
                  "now maps it so. THE WRITER echoes the dictionary (PatchExprField::writeData echoes dict_): "
                  "the functions<> tokens are re-emitted with the tokenizer's single-character delimiters "
                  "bare -- quoting them made OpenFOAM's reader fail at the functionObjectTrigger's `{`. "
                  "SPEED, FP-7 (2026-09-13): those downloads were WHOLE FIELDS. Every accessor of the "
                  "expression context called .host() on a 112,000-cell or 22,400-face array and sliced the "
                  "patch out of it, once per field the expression names, once per patch, once per "
                  "iteration -- measured on squareBendLiq as ONE gap of 3.15 ms per iteration inside an "
                  "energy phase whose wall is 4.9 ms and whose GPU work is 1.9. The boundary slice is now a "
                  "copy of the patch's own bytes (it is contiguous), the internal values are gathered on the "
                  "DEVICE (deviceGatherIndexed) into a patch-sized buffer, and each field is fetched at most "
                  "ONCE per evaluation. Sizing alone halved the bytes (142.3 -> 70.4 MB per 20 iterations) "
                  "and changed nothing, because it turned five big blocking copies into 74 small ones and "
                  "the cost is the ROUND TRIP. The cache took the energy phase 5.8 -> 4.7 ms/it, and BATCHING "
                  "took it to 3.9: the first request fills the whole cache, every registered field gathered "
                  "(scattered internal values) or sliced (contiguous boundary values) into ONE device buffer "
                  "read back in a single copy. Blocking copies in that phase went 8.0 -> 6.0 -> 2.0 per "
                  "iteration and their API time 2.75 -> 3.03 -> 0.79 ms, with the phase 39% -> 65% GPU-busy; "
                  "the four phases went 21.0 -> 19.0 ms/it, which puts squareBendLiq at 1.21x OpenFOAM on 20 "
                  "cores per iteration where it was 1.08x. rho_patch_expression_vs_openfoam holds its 1e-12 "
                  "walls bound throughout. Two round trips remain -- the batch coming back and the result "
                  "going out -- and the second needs a pinned staging buffer to go async."),
    ],
}

SELECTION_NOTE = {
    "incompressible::turbulenceModel": "createFields.H:42",
    "lduMatrix::solver": "fvSolution solvers/<field>/solver",
    "option": "constant/fvOptions or system/fvOptions",
}


# ofscan is a SIBLING REPO, not a build dependency: it is checked out next to brae and indexes the
# OpenFOAM tree. Only the DERIVED half of the manifest needs it, and only these two --check tests read
# it -- the porting workflow itself (the of-port / of-gate / of-instrument / of-measure skills) never
# does. A box without it therefore cannot answer the drift question, which is NOT the same as the
# manifest having drifted. Exit 77 so ctest records a SKIP; the GH200 reported a red here purely for
# lacking the sibling checkout, and a red that means "not asked" is how a suite stops being read.
SKIP_EXIT = 77


def db():
    sys.path.insert(0, os.path.abspath(OFSCAN))
    try:
        from ofscan.graph.database import Db
    except ImportError:
        sys.stderr.write("SKIP: ofscan not available at %s (set OFSCAN_ROOT) -- the manifest's DERIVED\n"
                         "      half cannot be regenerated here, so drift cannot be checked.\n"
                         % os.path.abspath(OFSCAN))
        sys.exit(SKIP_EXIT)
    return Db(os.path.join(os.path.abspath(OFSCAN), "ofscan.db"))


def impls_for(d, base):
    rows = d.q("SELECT DISTINCT selection_key k FROM runtime_types WHERE base=? AND selection_key IS NOT NULL"
               " ORDER BY k", (base,))
    return [r["k"] for r in rows]


def schema_for(d, cls):
    sys.path.insert(0, os.path.abspath(OFSCAN))
    from ofscan.foam import dictionary as dictmod
    out = []
    for r in dictmod.schema_for(d, cls):
        out.append(dict(key=r["key"], required=bool(r["r"]), default=r["d"],
                        op=r["op"], at="%s:%s" % (os.path.basename(r["p"] or "?"), r["l"])))
    return out


def case_selections(d, case):
    sys.path.insert(0, os.path.abspath(OFSCAN))
    from ofscan.foam import case as casemod
    out = []
    for cat, base, key, where in casemod.selections(case):
        r = casemod.resolve(d, base, key)
        out.append(dict(category=cat, keyword=key, where=where,
                        resolved=(r[0] if r else None), how=(r[3] if r else "UNRESOLVED")))
    return out


def y(s):
    """Minimal YAML scalar quoting."""
    if s is None:
        return "null"
    s = str(s)
    if s == "":
        return '""'
    # A leading '-' or '?' is a YAML INDICATOR, not text: an unquoted `file: -` parses as the start of a
    # sequence and makes the whole document unloadable. Quote on indicators anywhere it matters.
    if (re.search(r'[:#\[\]{},&*?|<>=!%@`"\']|^\s|\s$', s)
            or re.match(r'^[-?]', s)
            or s.lower() in ("yes", "no", "true", "false", "null", "~")):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def verify_provenance(solver):
    """Every OpenFOAM path the manifest claims must EXIST in the OpenFOAM tree.

    Provenance that is not checked is just a comment. This caught a real error on the first run: the
    curated table cited `src/MomentumTransportModels/...` for the turbulence models, which is the path in a
    DIFFERENT OpenFOAM lineage -- v2412 puts them under `src/TurbulenceModels/`. A manifest whose whole
    purpose is mechanical drift detection cannot ship paths that were never there to drift from.
    """
    bad = []
    for c in COMPONENTS[solver]:
        f = c["of_file"]
        if f in ("-", "", None):
            continue
        if not os.path.exists(os.path.join(OF, f)):
            bad.append((c["name"], f))
    return bad


def emit(solver, d, cases):
    L = []
    add = L.append
    add("# GENERATED by tools/of_manifest.py -- do not edit the `derived:` blocks by hand.")
    add("# The `classification:`/`brae_status:`/`note:` fields are CURATED and live in that script.")
    add("solver: %s" % y(solver))
    add("openfoam_root: %s" % y(OF))
    add("")
    add("components:")
    for c in COMPONENTS[solver]:
        add("  %s:" % c["name"])
        add("    openfoam:")
        add("      symbol: %s" % y(c["of_symbol"]))
        add("      file: %s" % y(c["of_file"]))
        if c.get("of_line"):
            add("      line: %d" % c["of_line"])
        add("    classification: %s" % c["classification"])
        add("    brae_status: %s" % c["status"])
        for k in ("brae_driver", "brae_existing", "brae_reference", "brae_cuda", "brae_target", "validation"):
            if c.get(k):
                add("    %s: %s" % (k, y(c[k])))
        if c.get("note"):
            add("    note: %s" % y(c["note"]))
        if c.get("selection_base"):
            ks = impls_for(d, c["selection_base"])
            add("    derived_runtime_selection:")
            add("      base: %s" % y(c["selection_base"]))
            add("      selected_at: %s" % y(SELECTION_NOTE.get(c["selection_base"], "-")))
            add("      openfoam_implementations: %d" % len(ks))
            add("      keys: [%s]" % ", ".join(y(k) for k in ks))
        if c.get("schema_for"):
            rows = schema_for(d, c["schema_for"])
            add("    derived_dictionary_schema:")
            add("      class: %s" % y(c["schema_for"]))
            if not rows:
                add("      keys: []")
            else:
                add("      keys:")
                for r in rows:
                    add("        - key: %s" % y(r["key"]))
                    add("          required: %s" % ("true" if r["required"] else "false"))
                    add("          default: %s" % y(r["default"]))
                    add("          read_by: %s" % y(r["op"]))
                    add("          at: %s" % y(r["at"]))
        add("")

    add("validation_cases:")
    for path in cases:
        if not os.path.isdir(path):
            continue
        sels = case_selections(d, path)
        unres = [s for s in sels if s["how"] == "UNRESOLVED"]
        add("  %s:" % y(os.path.basename(path)))
        add("    path: %s" % y(path))
        add("    derived_selection_sites: %d" % len(sels))
        add("    derived_unresolved: %d" % len(unres))
        seen = set()
        add("    derived_required_implementations:")
        for s in sorted(sels, key=lambda x: (x["category"], x["keyword"])):
            tag = (s["category"], s["keyword"])
            if tag in seen:
                continue
            seen.add(tag)
            add("      - {category: %s, keyword: %s, resolves_to: %s}"
                % (s["category"], y(s["keyword"]), y(s["resolved"])))
        add("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solver")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--check", metavar="FILE",
                    help="compare against an existing manifest; non-zero exit if it has drifted")
    a = ap.parse_args()
    if a.solver not in COMPONENTS:
        sys.exit("no curated component table for '%s' (add one to tools/of_manifest.py)" % a.solver)
    bad = verify_provenance(a.solver)
    if bad:
        sys.stderr.write("PROVENANCE ERROR -- these OpenFOAM paths do not exist under %s:\n" % OF)
        for name, f in bad:
            sys.stderr.write("  %-28s %s\n" % (name, f))
        return 2
    cases = a.cases or [
        os.path.join(OF, "tutorials/incompressible/simpleFoam/pitzDaily"),
        os.path.join(OF, "tutorials/incompressible/simpleFoam/airFoil2D"),
        os.path.join(OF, "tutorials/incompressible/simpleFoam/motorBike"),
    ]
    text = emit(a.solver, db(), cases)
    if a.check:
        old = open(a.check, encoding="utf-8").read() if os.path.exists(a.check) else ""
        if old != text:
            sys.stderr.write("manifest DRIFTED: %s is not what tools/of_manifest.py now produces.\n"
                             "Re-generate it and read the diff -- OpenFOAM or ofscan has changed.\n" % a.check)
            return 1
        sys.stderr.write("manifest up to date: %s\n" % a.check)
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
