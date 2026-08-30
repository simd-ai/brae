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
             note="Only used by the `consistent` (SIMPLEC) branch."),
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
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/fields/fv_patch_field.cuh",
             brae_target="src/finiteVolume/cfdTools/general/constrainPressure/"),
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
                  "rather than restating them."),
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
             note="limiter = min( k*|orth| / ((1 - k)*|corr| + SMALL), 1 ) applied to the non-orthogonal "
                  "correction, with |orth| the ORTHOGONAL part of the same snGrad "
                  "(nonOrthDeltaCoeffs*(vf[nei] - vf[own])) and both magnitudes taken BEFORE gamma*magSf. "
                  "`limited 1` is exactly `corrected` and `limited 0` is `uncorrected`. THE VECTOR CASE is "
                  "where this is easy to get wrong and where brae's pre-existing device implementation "
                  "did: OF's limitedSnGrad<Type> takes mag() of the WHOLE snGrad and of the WHOLE "
                  "correction, so all three velocity components share ONE per-face limiter. The device "
                  "limited each component independently -- a different scheme, 0.6% out on airFoil2D -- "
                  "and only writing the host reference and comparing the two exposed it "
                  "(deviceLaplacianCorrFluxLimitedVec is the fix). Needed by turbineSiting "
                  "(`Gauss linear limited corrected 0.33`), which remains blocked on "
                  "actuationDiskSource alone."),

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
             note="Froude blade-element momentum. Scoped, and refused rather than approximated outside "
                  "it: geometryMode `specified`, fixedTrim, no coning (Rcone = I), one lookup profile, "
                  "inletFlowType local|fixed. THE SIGN is the thing worth stating: OF's addSup is "
                  "`eqn -= force` with force carrying eqn's dimensions PER VOLUME (calculate divides by "
                  "V), and fvMatrix::operator-= is source() += V*su -- so the extensive source GAINS the "
                  "raw force. A body force that pushes the wrong way still converges, it just converges "
                  "to the wrong answer, so nothing about a residual would have caught it. brae's device "
                  "rotorDisk existed with NO test at all and a header comment describing the opposite "
                  "(`relaxSrc -= force`); the host port plus this gate established that the device FORCE "
                  "is right to 5.6e-16 and that the comment refers to the solver's application, not the "
                  "force. Unblocks the rotorDisk tutorial (its other two blockers, linearUpwind on "
                  "div(phi,k) and div(phi,omega), were cleared by the turbulence-scheme work)."),

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
                  "refused for."),

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
                        "nut 4.8e-03.",
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
                  "exactly the cell that matters."),

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
             note="8 keys in v2412. brae implements PCG (+AMG preconditioning) and PBiCGStab."),
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
             note="Level-scheduled, bit-identical to OpenFOAM."),
        dict(name="GAMGSolver", of_symbol="Foam::GAMGSolver",
             of_file="src/OpenFOAM/matrices/lduMatrix/solvers/GAMG/GAMGSolver.C",
             classification="LINEAR_SOLVER", status="UNSUPPORTED",
             schema_for="GAMGSolver",
             brae_target="src/matrices/lduMatrix/solvers/GAMG/",
             note="pitzDaily and motorBike BOTH select `GAMG` for p. brae substitutes AMG-preconditioned "
                  "PCG. That is a different algorithm with a different iteration count -- the solver must "
                  "say so, not silently substitute."),

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
    "rhoSimpleFoam": [
        dict(name="rhoSimpleFoam_main", of_symbol="main",
             of_file="applications/solvers/compressible/rhoSimpleFoam/rhoSimpleFoam.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam.cu",
             validation="tests/rho_simple_end_to_end_vs_openfoam.sh -- THE WHOLE CASE in _cpp against real OpenFOAM. 112k-cell sbMatched TURBULENT, with the fixture's own kEpsilon and its own flowRateInletVelocity, `consistent yes` + `transonic yes` so the path exercised is pcEqn.H transonic. 400 iterations from the same start. ONE FIELD IS GATED: U, at 5e-4, measuring 2.364262e-05 -- and the bound sits ~21x above the measurement, so the gate would pass a solver twenty times worse than that number. The per-patch confinement is asserted for every patch at the same 5e-4 (worst: walls 3.4954e-05), which is what would fire if a defect moved the error somewhere new. EVERYTHING ELSE IS REPORTED, NOT GATED, and the run labels it so: p 1.230477e-05, T 4.544945e-06, rho 1.341389e-05, k 7.648476e-05 all print `(reported, see the control)`. They are not bounded because on this fixture p and T barely move from their initial values, so no bound could both pass a correct solver and fail one that did nothing -- the control proves that rather than assuming it. The residual history (U 1.0 -> 1.047e-08, k -> 6.607e-08, epsilon -> 1.577e-08) is printed too; note that the U residual is not inert, because the inlet mass-flow bound is 10x it. THE START-STATE CONTROL CANNOT FAIL, and that is a property of the fixture rather than of the solver: sbMatched/0.orig/U is `uniform (0 0 0)`, so relL2 against OpenFOAM is exactly 1.0 for any code state. It establishes that 5e-4 lies below the do-nothing error and nothing that could ever come out differently. RUN TO CONVERGENCE, NOT TO A TUTORIAL endTime: comparing at an endTime compares trajectories, and 400 is where both codes have stopped moving. AN UNPORTED RAS MODEL IS REFUSED BY NAME, and so is an unported nut wall function -- both asserted by the gate, each with a negative control proving createFields still accepts the case's own. WHAT THIS FIXTURE NEVER REACHES: limitTemperature, any fvOptions (sbMatched ships no fvOptions and no MRFProperties), MRF, the kOmegaSST branch, the laminar branch, the SUBSONIC pEqn.H branch and the rho.relax() it guards. Each is a live branch of the driver that this gate does not exercise. tests/rho_angledduct_vs_openfoam.sh would reach the fvOptions and porosity paths and is deliberately NOT registered (PORT.md), so those remain ungated end-to-end.",
             note="The driver branches on simple.consistent(): pcEqn.H for SIMPLEC, pEqn.H otherwise. "
                  "Both must be ported -- the existing solver reached only one of them by case. Order is "
                  "UEqn, EEqn, then p; turbulence->correct() LAST, after the pressure solve."),
        dict(name="createFields", of_symbol="createFields.H",
             of_file="applications/solvers/compressible/rhoSimpleFoam/createFields.H",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoCreateFields.cu",
             validation="tests/rho_createfields_vs_openfoam.sh -- against OpenFOAM's OWN createFields.H, obtained by running rhoSimpleFoam -postProcess with writeObjects(phi,rho); postProcess.H builds the field set without solving. On a developed 112k-cell sbMatched state: rho 6.1e-16, phi 2.1e-15, initialMass 3.9e-16, psi exact, and a restart reads both fields byte-exact. THE CONTROL is the point: interpolate(rho)*flux(U), which is pEqn.H form and brae fvc::rhoFlux, reads 3.7e-04 against the same oracle -- 11 orders worse -- so the gate discriminates the two flux forms instead of passing either. It also asserts the fixture rho is non-uniform, because a uniform rho makes the two forms algebraically identical and the shipped 0.orig is uniform.",
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
             validation="tests/rho_ueqn_vs_openfoam.sh -- against OpenFOAM's OWN assembled momentum matrix, via the tools/dumpPEqn stage harness (stage_rAU = 1/UEqn.A(), stage_UIC, stage_UBC, stage_muEff, stage_Uass) at SIMPLE iteration 1 on 112k-cell sbMatched: rAU 6.1e-15, internalCoeffs 7.1e-15, boundaryCoeffs 4.1e-15, per patch inlet/outlet/walls all at machine precision. OpenFOAM's own muEff is INJECTED so the number measures the assembly and not the unported compressible turbulence closure. THE CONTROL: assembling with the kinematic nu_eff -- the incompressible divDevReff -- reads 6.2e-01, fourteen orders worse, and forcing that form into the implementation fails the gate at 6.2e-01, so it discriminates the one thing that distinguishes this solver's momentum equation from simpleFoam's.",
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
             validation="tests/rho_eeqn_vs_openfoam.sh -- against OpenFOAM's OWN assembled energy equation via the tools/dumpPEqn harness (stage_Ekp, stage_he, stage_eD, stage_eSrc, stage_alphaEff, stage_Upred) at SIMPLE iteration 1 on 112k-cell sbMatched: Ekp 6.4e-16 (and 6.1e-16 on every patch), EEqn.D() 3.8e-15, source+boundaryCoeffs 3.9e-15, interior and boundary cells both at machine precision. alphaEff and he are INJECTED so the number measures the assembly, not the unported compressible turbulence closure or the unported energy boundary types. THE CONTROL: the `h` arm (K = 0.5|U|^2) reads 1.0 against stage_Ekp and builds a convection term differing from the `e` arm by 100% of its own magnitude; forcing the h arm into the implementation fails the gate. NOTE the control is taken on the UNBOUNDED convection term on purpose -- `div(phi,Ekp)` is bounded, and at iteration 1 the bounded subtraction removes the near-uniform p/rho that IS the difference between the arms (|KE div| 1.3e-04 bounded against 1.4e+01 unbounded), so the assembled source at this state cannot discriminate them.",
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
             validation="tests/rho_kepsilon_vs_openfoam.sh -- against OpenFOAM's OWN kEpsilon, instrumented. tools/dumpKEpsilon is that model with writes added and its equations untouched, registered as kEpsilonDump through makeRASModel, so gradU, divU, GbyNu, G, both diffusivities, the mesh factors the laplacian coefficient is a product of, both off-diagonal sets and both assembled systems (before and after relax/boundaryManipulate) each have an oracle. On 112k-cell sbMatched at SIMPLE iteration 1, given OpenFOAM's own inputs: epsilon 5.0e-15, k 8.9e-16, nut 1.7e-15, alphat 2.3e-15, wall and interior alike, with every intermediate at the same order. THE CONTROL: substituting the MASS flux for the volumetric one in divU -- the one difference between the compressible and incompressible readings of this same templated model -- must be worse by at least 1e3x, measured 5.0e-06 against 5.0e-15. A term sweep drops each of the eight terms in turn and every one of them moves the answer. A short source array for a turbulent inlet is REFUSED by name, with a full-length array as the negative control.",
             note="OpenFOAM has ONE templated kEpsilon.C; the compressible instantiation supplies alpha=1, "
                  "rho as a field, alphaRhoPhi as the MASS flux and a nu that varies with T. TWO FLUXES, "
                  "not one: fvm::div takes the mass flux while divU takes the VOLUMETRIC one, because "
                  "compressibleTurbulenceModel::phi() divides by interpolate(rho). Four defects were found "
                  "by reading the instrumented model's numbers rather than by reasoning -- the closure "
                  "never being given the flux (so fvc::grad(U) read a stale outlet face value), k and "
                  "epsilon never having updateCoeffs called (so every flux-conditional patch contributed "
                  "nothing), the turbulent inlets being frozen at the case file's `value` instead of "
                  "recomputed from U and k, and the diffusion terms being assembled orthogonally while "
                  "the case asked for `corrected`. See PORT.md."),
        dict(name="createFieldRefs", of_symbol="createFieldRefs.H",
             of_file="applications/solvers/compressible/rhoSimpleFoam/createFieldRefs.H",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu",
             brae_target="src/applications/solvers/rhoSimpleFoam/rhoCreateFields.cu",
             validation="tests/rho_createfields_vs_openfoam.sh -- psi == 1/(R T) exactly, from the same T that rho came from.",
             note="One line: psi is a const reference to thermo.psi(). It matters because pEqn uses psi "
                  "AFTER thermo.correct() has moved it."),
    ],
}

SELECTION_NOTE = {
    "incompressible::turbulenceModel": "createFields.H:42",
    "lduMatrix::solver": "fvSolution solvers/<field>/solver",
    "option": "constant/fvOptions or system/fvOptions",
}


def db():
    sys.path.insert(0, os.path.abspath(OFSCAN))
    from ofscan.graph.database import Db
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
        for k in ("brae_existing", "brae_reference", "brae_cuda", "brae_target", "validation"):
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
