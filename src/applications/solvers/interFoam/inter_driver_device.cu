// runInterFoamDevice -- the same interFoam time loop, on the GPU.
//
// See inter_driver_cpp.cuh for why the hooks live here rather than in the gate. Everything this calls
// is separately landed and separately gated; what this file owns is the wiring, and that wiring is
// what tests/test_device_inter_dambreak_alpha.cu measures against the host driver on damBreak.
#include "inter_driver_cpp.cuh"
#include "inter_correct_phi_cpp.cuh"
#include "inter_case_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "inter_ueqn_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include "two_phase_mixture_cpp.cuh"
#include "solution_directions.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "brae_notice.cuh"
#include "device_inter_step.cuh"
#include "device_inter_turbulence.cuh"
#include "device_blas.cuh"
#include "device_alpha_courant.cuh"
#include "time_controls.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

// EVERY ONE OF THESE FEEDS A DEVICE BOUNDARY ARRAY, and the device mesh keeps a COUPLED patch out of
// its boundary gather entirely (device_mesh.cuh:41-44): its Sf layout is [internal | non-cyclic
// boundary] and its bndCell list matches. Flattening every patch here makes each array longer than the
// device's boundary-face count, which is how a periodic mesh first failed -- the alpha pre-solve
// refused its coefficients by size, and the momentum assembly refused rho and nuEff the same way.
std::vector<scalar> flattenPatches(const std::vector<std::vector<scalar>>& b,
                                   const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> v;
    for (std::size_t pi = 0; pi < b.size(); ++pi)
    {
        if (pi < fvp.size() && isCoupledInterfaceType(fvp[pi].type)) continue;
        v.insert(v.end(), b[pi].begin(), b[pi].end());
    }
    return v;
}

std::vector<scalar> patchValues(const GeometricField<scalar>& f, const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> v;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;
        const std::vector<scalar>& b = f.boundary[pi]->value();
        v.insert(v.end(), b.begin(), b.end());
    }
    return v;
}

std::vector<scalar> fullFace(const SurfaceScalarField& f, const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> v(f.internal);
    for (std::size_t pi = 0; pi < fvp.size() && pi < f.boundary.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;   // the device's Sf layout, see above
        v.insert(v.end(), f.boundary[pi].begin(), f.boundary[pi].end());
    }
    return v;
}

// ...and the halves fullFace drops: a surface field's values ON THE PERIODIC PAIR, in the order
// buildDeviceCyclic lays the pair out -- the cyclics in the order buildCyclicInterfaces returns them,
// each patch's faces in patch order. That is the order every DeviceCyclic array is in, so an array
// built here indexes face for face with cyc.magSf and cyc.Sf.
std::vector<scalar> coupledFace(
    const SurfaceScalarField& f,
    const std::vector<CyclicInterface>& cyclics)
{
    std::vector<scalar> v;
    for (const CyclicInterface& c : cyclics)
    {
        const std::size_t pi = static_cast<std::size_t>(c.patch);
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            v.push_back(pi < f.boundary.size() && i < f.boundary[pi].size()
                        ? f.boundary[pi][i] : scalar(0));
        }
    }
    return v;
}

// OpenFOAM's directionMixed evaluate for the flux-conditional velocity patches. evaluateBoundary()
// alone does not resolve them, and their matrix coefficients are built from the value it leaves --
// see the note in pressureCorrector, and tools/dumpInterFoam for OpenFOAM's own numbers.
void updateVelocityPatches(GeometricField<vector>& U, const std::vector<FvPatch>& fvp)
{
    updateVelocityPatchesFromCells(U, fvp);
}

}   // namespace


RunReport runInterFoamDevice(
    const std::string& caseDir,
    const std::string& startDir,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    label nSteps,
    bool verbose,
    InterFields* fieldsOut,
    scalar endTime,
    DeviceInterStepTaps* tapsOut,
    const MutableMesh* mutableMesh)
{
    InterFields f = buildInterFields(caseDir, startDir, m, g, fvp);
    // THE PAIR, built here and not at the device-mesh stage, because the hooks below fill its share of
    // the surface fields and they are defined before the DeviceCyclic is.
    // ...INCLUDING a cyclicACMI the caller has coupled as a coincident pair (cpu::cyclicACMI::setup):
    // on the mask-scaled areas it IS a cyclic, with one difference this loop has to honour -- MULES
    // does not sync its limiter across it (CyclicInterface::ami).
    const std::vector<CyclicInterface> cyclics =
        buildCyclicInterfaces(m, g, fvp, /*includeCoupledACMI=*/true);
    // A COUPLED PATCH THAT IS NOT IN THAT LIST WOULD BE IN NOTHING: the device mesh and every boundary
    // flatten skip the coupled types (device_mesh.cuh:41-44), so its faces would be neither boundary
    // nor interface, silently. A cyclicAMI coupled by a harness is the live case; refused by name.
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (!fvp[pi].coupled) continue;
        bool carried = false;
        for (const CyclicInterface& c : cyclics)
        {
            carried = carried || static_cast<std::size_t>(c.patch) == pi;
        }
        if (!carried)
        {
            throw std::runtime_error(
                "brae interFoam -device: patch `" + fvp[pi].name + "` of type `" + fvp[pi].type +
                "` is coupled and this loop carries no interface for it -- it couples a translational "
                "`cyclic` and a coincident `cyclicACMI`. Its faces would be in no operator at all. "
                "Run without -device.");
        }
    }
    // stf and snGrad(rho) on the pair, refilled by the interfaceForces hook every step; ghf is the
    // mesh's and is built once, below.
    DeviceBuffer<scalar> dStfIf, dSnRhoIf, dGhfIf;
    // The pair's flux, for the HOST fields the hooks read. cyc.phi is the device's copy and the only
    // current one; a condition that looks phi up on a coupled patch -- porousBafflePressure computes
    // its jump from it -- reads f.phi.boundary there, which the unflatten deliberately does not touch
    // (that array has no coupled patch in it). Set once the DeviceCyclic exists.
    const DeviceBuffer<scalar>* cycPhiForHost = nullptr;
    // ...and nHatf on the pair, which outlives the step: the first corrector's phir reads the normal
    // the LAST mixture.correct() left, and without MULESCorr that is the previous TIME STEP's.
    DeviceBuffer<scalar> dNHIf;
    // FIRST among the device refusals: the model is what a reader of the message has to change, and
    // every refusal below is about the loop around it
    // (kOmegaSST runs on this loop now: device_inter_turbulence.cu's SST branch hands the closure
    // rhoSimpleFoam gates what the host's SST branch hands its reference. Gated against OpenFOAM on
    // RAS/waterChannel.)
    // LES kEqn runs on this loop (device_les_keqn.cu, a transcription of les_kEqn_cpp.cu), with the
    // filter width taken once from the host's LESdelta::compute. It was refused twice: first because
    // the closure was host-only, then -- the equation ported -- because LES/nozzleFlow2D read U
    // 1.7e-01 from OpenFOAM after ONE step with alpha exact. That was not the closure: the case is a
    // WEDGE, and this loop never refreshed the wedge's refValue (device_inter_step.cu says what that
    // cost and why a field at rest hides it).
    //
    // A MOVING mesh is still refused: LESdelta is a MeshObject in OpenFOAM and moves with the mesh,
    // and the device closure takes the width once.
    if (f.turbulence.on && f.turbulence.model == cpu::interFoam::InterRasModel::KEqnLES
        && f.dynamicMesh)
        throw std::runtime_error(
            "brae interFoam (device): the case is LES kEqn AND moves its mesh. The device closure takes "
            "the filter width once, from the host's LESdelta::compute on the mesh as it starts; a mesh "
            "that moves changes it at every update. Run without -device.");

    // fvOptions: the device UEqn applies explicitPorositySource/DarcyForchheimer, and nothing else. Each
    // other type is refused BY ITS OWN NAME rather than by a blanket notice -- the mangroves pair, the
    // constraints and the rest are host-only and gated there.
    for (const fvOptions::Option& o : f.fvOptions.options)
    {
        if (!o.active) continue;
        const bool darcy = o.unsupported.empty()
                        && (o.type == "explicitPorositySource")
                        && !o.fixedCoeff;
        if (darcy) continue;
        throw std::runtime_error(
            "brae interFoam (device): fvOptions has an active option `" + o.name + "` (" + o.type
            + "). The device loop's UEqn applies explicitPorositySource/DarcyForchheimer only; the host "
            "loop carries this one. Refused rather than run the case without it.");
    }
    // THE PAIR RUNS ON THE DEVICE. validation/interFoamCyclic, ten steps of 2e-3, against real
    // OpenFOAM: alpha 4.2e-11, p_rgh 1.98e-11 relative of 1.6e+03, U 5.7e-11 relative -- and the same
    // three against brae's own host loop. FIVE THINGS HAD TO CARRY THE PAIR, and every one of them is
    // identically zero at the first step of a case at rest, which is why only a multi-step fixture
    // could see them: the host<->device boundary-array layout (ten flatten sites), the momentum
    // matrix's OWN interface coefficient in fvMatrix::H(), the Gauss-Seidel smoother applying the
    // interface to its right-hand side every sweep, fvc::ddtCorr, and divDevReff's grad(U) and its
    // stress flux.

    // A JUMP ON THE PAIR runs here. It reaches the device everywhere OpenFOAM applies one -- the
    // neighbour value is psi[nbr] - jump in fvMatrix::flux(), in fvc::grad and in the matrix product,
    // the last ONLY when the operand is the solution field ("only apply jump to original field",
    // jumpCyclicFvPatchField.C:169-177, so never a Krylov search direction) -- it is rebuilt at every
    // assembly by the pressureCoeffs hook from that assembly's flux, and the pair's flux is pushed
    // back to the host field the jump is computed from. MEASURED on validation/interFoamCyclic's
    // `jump` profile, ten steps against brae's own host loop: alpha 1.1e-12, p_rgh 3.8e-08 of
    // 1.7e+03, U 4.7e-11. With the jump absent the device lands on the PLAIN-CYCLIC answer, alpha
    // 4.9e-02 and U 45%, which is what that profile's control measures.

    // THE ISOTROPIC AND SHEAR COMPRESSION TERMS. alphaEqn.H:60-75 blends phic with
    // cAlpha*icAlpha*interpolate(mag(U)) and then ADDS scAlpha*mag(delta() & interpolate(symm(grad(U)))).
    // The device alpha step carries neither -- DeviceAlphaStepInput has cAlpha and nothing else -- so a
    // case that sets either would run with its compression quietly reduced to the standard term. The
    // host loop carries both (alpha_eqn_cpp.cu:140-147).
    if (f.alphaCtl.icAlpha != scalar(0) || f.alphaCtl.scAlpha != scalar(0))
    {
        throw std::runtime_error(
            "brae interFoam (device): the case sets icAlpha or scAlpha. The device alpha step carries "
            "the standard interface compression only; the host loop (no -device) carries the isotropic "
            "and shear terms. Refused rather than run a different compression.");
    }

    // A variableHeightFlowRateInletVelocity is rebuilt by the U-boundary hook now, from the phase
    // fraction on its patch, as the host driver rebuilds it (inter_driver_cpp.cu:722-735).
    // THE PERMEABLE-WALL PAIR runs here too: both halves are host patches, told the flux and the phase
    // field's patch values by pushFlux at every hook, the pressure half rebuilt inside the pressure
    // hook's constrainPressure, and the velocity half kept off the new flux in the one corrector
    // OpenFOAM keeps it off (DeviceInterStepHooks::updateUBoundary). What is still refused is the pair
    // on a MOVING mesh: prghPermeableAlphaTotalPressure reads phi as it stands at constrainPressure,
    // and on a moving mesh that is a flux this loop makes relative at a different point from the host
    // loop. No tutorial ships the combination and neither arm has measured it.
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const bool permeable = f.U.boundary[pi]->needsAlphaPatchValues()
                            || f.p_rgh.boundary[pi]->needsAlphaPatchValues()
                            || f.p_rgh.boundary[pi]->isPrghPermeableAlphaTotalPressure();
        if (permeable && f.dynamicMesh)
            throw std::runtime_error(
                "brae interFoam (device): patch `" + fvp[pi].name + "` carries a permeable-wall condition "
                "(permeableAlphaPressureInletOutletVelocity or prghPermeableAlphaTotalPressure) and the "
                "mesh moves. The pressure half reads the flux as it stands when constrainPressure runs, "
                "and the device loop has not been measured against OpenFOAM on a moving mesh with that "
                "condition. Refused rather than run an unmeasured flux; a static mesh runs (gated on "
                "laminar/damBreakPermeable).");
    }
    // `grad(U) cellLimited`: the host momentum equation limits the gradient; the device's does not read
    // the coefficient
    if (f.gradULimitK > 0)
        throw std::runtime_error(
            "brae interFoam (device): fvSchemes limits grad(U) (cellLimited, k = "
            + std::to_string((double)f.gradULimitK) + "). The host loop carries it into linearUpwind and the "
            "viscous term; the device loop's momentum equation does not. Refused rather than run it unlimited.");
    // ddtCorr's BOUNDARY HALF is on the device now: deviceDdtCorr already computed it (and already
    // zeroed it wherever U fixes a value, as fvcDdtPhiCoeff does), and the pressure step adds it to
    // phiHbyA with interpolate(rho*rAU)'s patch value, as the host does (inter_peqn_cpp.cu:531-573).
    // Gated on RAS/weirOverflow, whose outlet is a zeroGradient U.
    // A flowRateInletVelocity is REBUILT at every momentum assembly, and the U-boundary hook below does
    // that now, from the mixture's boundary rho, as the host driver does. What is still refused is a rate
    // that is a Function1 of time, which this driver takes as one number for the whole run -- the patch
    // itself throws on that (flowRateValue) -- and the variableHeight form, which reads the phase field.

    // NOT ON THE DEVICE YET, refused rather than run on the mesh as it started or on a singular
    // pressure system: a mesh that moves (the host loop has it, inter_driver_cpp.cu), and a closed
    // case -- one whose p_rgh fixes its value on no patch -- whose pressure reference the device
    // step pins at pRefValue where OpenFOAM pins it at the cell's current p_rgh, with neither the
    // level shift of p nor adjustPhi.
    // A MESH THAT MOVES. The motion solve itself is a host operation on both arms -- OpenFOAM's
    // motion solver is a Laplacian on the point field, and brae has one host implementation of it --
    // so the device loop moves the mesh exactly as the host loop does (inter_driver_cpp.cu's
    // Stage::meshUpdate) and then refreshes the buffers it had uploaded from the geometry.
    DynamicMotionSolverFvMesh* dyn = f.dynamicMesh.get();
    // A cyclicACMI PAIR WHOSE `scale` MOVES WITH TIME is rescaled at every step, in place, on the
    // caller's mutable objects -- the host loop's guards, transcribed (inter_driver_cpp.cu:256-302),
    // because the rescale point this loop carries is the one that loop gates and no other.
    cpu::cyclicACMI::Interfaces* acmi = mutableMesh ? mutableMesh->acmi : nullptr;
    {
        bool hasACMI = false;
        for (const FvPatch& q : fvp)
        {
            hasACMI = hasACMI || q.type == "cyclicACMI";
        }
        if (hasACMI && (!acmi || acmi->empty()))
        {
            throw std::runtime_error(
                "brae interFoam -device: the mesh has a cyclicACMI pair and the caller handed the driver "
                "no ACMI state. Couple it with cpu::cyclicACMI::setup and pass the result through "
                "MutableMesh.");
        }
        if (acmi && acmi->scaled())
        {
            if (mutableMesh->m != &m || mutableMesh->g != &g || mutableMesh->patches != &fvp)
            {
                throw std::runtime_error(
                    "brae interFoam -device: MutableMesh names different objects from the mesh, geometry "
                    "and patches the fields were built against; the cyclicACMI rescale would move a copy.");
            }
            if (dyn)
            {
                throw std::runtime_error(
                    "brae interFoam -device: the case scales a cyclicACMI interface AND moves its mesh. "
                    "OpenFOAM then re-runs the AMI and rescales the mesh flux in "
                    "cyclicACMIFvPatch::movePoints; that is not ported.");
            }
            // WHERE IN THE STEP the rescale lands is part of the answer (cyclic_acmi_cpp.cuh): after
            // alphaEqn.H forms phic and before the pre-solve. That point is gated for the pre-solving
            // path with no isotropic or shear compression and one alpha sub-cycle; each of the others
            // moves OpenFOAM's first interpolation across the pair, and none is gated on either arm.
            if (!f.alphaCtl.MULESCorr || f.alphaCtl.icAlpha != scalar(0) || f.alphaCtl.scAlpha != scalar(0)
             || f.alphaCtl.nAlphaSubCycles != 1)
            {
                throw std::runtime_error(
                    "brae interFoam -device: the case scales a cyclicACMI interface with time, and the "
                    "step's rescale point is ported for `MULESCorr yes`, icAlpha 0, scAlpha 0 and "
                    "nAlphaSubCycles 1 only; this case sets MULESCorr "
                    + std::string(f.alphaCtl.MULESCorr ? "yes" : "no") + ", icAlpha "
                    + std::to_string((double)f.alphaCtl.icAlpha) + ", scAlpha "
                    + std::to_string((double)f.alphaCtl.scAlpha) + ", nAlphaSubCycles "
                    + std::to_string((long)f.alphaCtl.nAlphaSubCycles) + ".");
            }
        }
    }
    if (dyn)
    {
        // THIS LOOP MOVES A MESH, and it agrees with the host arm when both run the SAME linear
        // solver: sloshingTank2D, one step, every solve pinned at 1e-14 --
        //
        //     same solver (PCG+DIC on both):  p_rgh 1.2107e-08 of 5.2780e+06, phi 3.4e-12, |U| 3.2e-12
        //     end to end, shipped binary:     host max|U| 10.32, div(phi) 4.814e-11
        //                                     device       10.32,            4.278e-11
        //
        // The case's own GAMG SOLVER runs here: the hierarchy is the mesh's, shared with the motion
        // solve and with pcorr, and it is rebuilt on every move -- OpenFOAM's GAMGAgglomeration is a
        // MeshObject whose movePoints sets requireUpdate_ and whose next New builds it again
        // (GAMGAgglomeration.C:311-330, :498-516). The smoother is checked below, where a static-mesh
        // case's is.
        //
        // The GAMG PRECONDITIONER runs here as well (devicePcgGamgSolve). It was refused on a moving
        // mesh while this loop substituted Jacobi-BiCGStab for it, because the substitute did not
        // reach the tolerance it was given: measured on sloshingTank2D, p_rgh 2.5406e+05 of
        // 5.2780e+06 against the host, worst |div(phi)| 3.274e-01 against 9.173e-07. With the
        // preconditioner itself there is nothing left to substitute.

        // A COUPLED PAIR ON A MOVING MESH is not ported: buildDeviceCyclic lays the interface out from
        // the geometry, and every hook in this loop holds a pointer into that layout, so rebuilding it
        // mid-run would leave them addressing the old one. Refused by name rather than run on a pair
        // whose geometry has moved out from under it.
        for (const FvPatch& q : fvp)
        {
            if (q.coupled)
            {
                throw std::runtime_error(
                    "brae interFoam -device: the case moves its mesh AND carries the coupled patch `"
                    + q.name + "`. The pair's device interface is built from the geometry once and the "
                    "step's hooks hold pointers into it; this loop does not rebuild it after a move. "
                    "Run without -device.");
            }
        }
        if (!mutableMesh || !mutableMesh->m || !mutableMesh->g || !mutableMesh->patches)
        {
            throw std::runtime_error(
                "brae interFoam -device: the case moves its mesh (" + dyn->motionType() + ") and the "
                "caller handed the driver a mesh it may not move. Pass the same mesh, geometry and "
                "patches through MutableMesh; the fields hold references to those patches and must "
                "see every move.");
        }
        if (mutableMesh->m != &m || mutableMesh->g != &g || mutableMesh->patches != &fvp)
        {
            throw std::runtime_error(
                "brae interFoam -device: MutableMesh names different objects from the mesh, geometry "
                "and patches the fields were built against. Moving a copy would leave every field on "
                "the old mesh.");
        }
        // WHAT IS NOT PORTED YET, refused by name rather than run on a stale interface. A moving AMI
        // has to re-run the overlap and re-upload it (the legacy simpleFoam driver does both in
        // DeviceSimpleSolver::moveMesh); this loop refreshes the mesh geometry and the pair, not an
        // AMI's weights.
        for (const FvPatch& q : fvp)
        {
            if (q.type == "cyclicAMI" || q.type == "cyclicACMI")
            {
                throw std::runtime_error(
                    "brae interFoam -device: the case moves its mesh AND carries the patch `" + q.name
                    + "` of type `" + q.type + "`. A moving AMI's weights are a function of the moved "
                      "geometry and this loop does not recompute them; it would interpolate across "
                      "faces that have moved away. Run without -device.");
            }
        }
        dyn->attach(*mutableMesh->m, *mutableMesh->g, *mutableMesh->patches);
    }
    // THE NON-ORTHOGONAL CORRECTION IS ON THE DEVICE NOW, module by module and each transcribed from the
    // host: the pressure laplacian's loop and its face-flux correction (device_inter_pressure_step.cu),
    // the viscous laplacian (the shared assembler, handed the case's flags in device_inter_step.cu), the
    // three snGrads above, CorrectPhi's pcorr (host operators, cpc.correctedLaplacian) and the closure's
    // k and epsilon (DeviceInterTurbulence). Gated end to end on laminar/damBreak `sheared`.
    // A CASE THAT NEEDS A PRESSURE REFERENCE RUNS ON THE DEVICE NOW: setReference pins the cell at its
    // CURRENT p_rgh (pEqn.H:47) and the level shift of p with p_rgh rebuilt from it (pEqn.H:74-83) are
    // both in the device pressure step, transcribed from the host's own lines. What is NOT there is
    // adjustPhi (pEqn.H:21-26), which scales the ADJUSTABLE outflow to balance continuity. On a case
    // whose every boundary face has its flux fixed by U there is nothing for it to scale -- OpenFOAM's
    // adjustableMassOut is then zero, its guard fails and massCorr stays 1 (adjustPhi.C:96-106), so the
    // device matches by doing nothing. Anywhere else it is refused by name rather than skipped.
    if (f.pRef.needReference)
    {
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (fvp[pi].type == "empty" || isCoupledInterfaceType(fvp[pi].type)) continue;
            // adjustPhi's own predicate, as the host arm has it (inter_peqn_cpp.cu:371) and as OpenFOAM
            // writes it: a fixed outflow is not adjustable, an inletOutlet's is. mixedFvPatchField and
            // directionMixedFvPatchField BOTH return fixesValue() = true (mixedFvPatchField.H:197,
            // directionMixedFvPatchField.H:130), so a pressureInletOutletVelocity is a fixed outflow to
            // adjustPhi -- brae's piov reports the same, and that is not a defect.
            const bool fixed = f.U.boundary[pi]->fixesValue() && !f.U.boundary[pi]->isInletOutlet();
            // ...but a patch the PRESSURE drives still leaves adjustPhi something to weigh: with nothing
            // adjustable, OpenFOAM tests whether the FIXED fluxes balance and aborts when they do not
            // (adjustPhi.C:106), and the device step carries neither the scaling nor that test. MEASURED
            // on damBreak with its atmosphere turned into a fixedFluxPressure (interfoam_refusals
            // `device_closed`): the device ran it to a worst |div(phi)| of 5.2e-02.
            const bool pressureDriven = f.U.boundary[pi]->bcCategory() == 6;
            if (fixed && !pressureDriven) continue;
            throw std::runtime_error(
                "brae interFoam -device: p_rgh needs a reference cell and U patch `" + fvp[pi].name
                + "` is not a wall that fixes its flux, so pEqn.H:21-26 has adjustPhi weigh it -- "
                "scaling the adjustable outflow, or aborting if the fixed fluxes do not balance. The "
                "device pressure step carries neither. The host loop does. Run without -device.");
        }
    }
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const label nFaces = static_cast<label>(g.magSf().size());
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    // nOuterCorrectors IS the loop below now, transcribed from the host's runTimeStep
    // (inter_solve_cpp.cu:111-131). frozenFlow is not: pimple.frozenFlow() makes OpenFOAM `continue`
    // past the momentum, the pressure AND the turbulence corrector, and this loop has no such branch.
    if (f.pimple.frozenFlow)
    {
        throw std::runtime_error(
            "brae interFoam (device): `solveFlow no` skips the momentum, the pressure and the "
            "turbulence corrector for the whole outer iteration (interFoam.C:163-166), and the device "
            "loop has no branch for it. The host path (no -device) does.");
    }

    // ...AND A VELOCITY CONDITION THAT NAMES A FLUX OTHER THAN phi. p_rgh's and alpha's conditions are
    // evaluated on the host, which hands each the flux its `phi` entry names (namedPatchFlux, through
    // pushFlux below). U's pressureInletOutletVelocity switch runs ON THE DEVICE and reads phi.
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::string& name = f.U.boundary[pi]->fluxName();
        if (name == "phi") continue;
        throw std::runtime_error(
            "brae interFoam (device): U's patch `" + fvp[pi].name + "` names the flux `" + name
            + "` in its `phi` entry, and the device loop's velocity switch reads phi. The host path "
            "(no -device) hands each patch the flux it names.");
    }

    // THE MESH'S GAMG HIERARCHY, one for the whole run, as OpenFOAM keeps one GAMGAgglomeration per
    // mesh and every GAMG solve shares it -- the motion solver's, pcorr's and p_rgh's. It is built by
    // the FIRST of them, from ITS entry's nCellsInCoarsestLevel, and a mesh move sends it un-built
    // (dynamic_motion_solver_fv_mesh_cpp.cu, GAMGAgglomeration::movePoints).
    GamgAgglomerationCache meshAgglomeration;

    // initCorrectPhi.H, on the host and before anything is uploaded -- see runInterFoam. A GAMG pcorr
    // SOLVER builds the hierarchy even at rest (the GAMGSolver constructor builds it before the first
    // residual), and OpenFOAM's p_rgh GAMG then reuses that one.
    std::vector<LinearSolveRecord> initPcorrSolves;
    {
        CorrectPhiControls cpc;
        cpc.pcorr = &f.pcorrSolve;
        cpc.pcorrFinal = &f.pcorrSolveFinal;
        cpc.gamgCache = &meshAgglomeration;
        cpc.correctedLaplacian = f.laplacianScheme.corrected;
        cpc.snGradLimitCoeff = f.laplacianScheme.limitCoeff;
        cpc.nNonOrthogonalCorrectors = f.nNonOrthogonalCorrectors;
        const SurfaceScalarField one = unitFaceField(m, fvp);
        CorrectPhiInput cin;
        cin.rAUf = &one;
        cin.rhoPhi = &f.rhoPhi;
        cin.solveLog = &initPcorrSolves;
        correctPhi(f.U, f.phi, f.p_rgh, cin, cpc, m, g, fvp);
        pushFluxToPatches(f, fvp);
        // (a pcorr GAMG with a different nCellsInCoarsestLevel from p_rgh's was refused here while
        // the device built a hierarchy of its own. It shares the run's now, so the first solve's
        // entry decides it and the second entry's number is never read -- which is what OpenFOAM
        // does, and what the `both` profile of tests/interfoam_gamg_vs_openfoam.sh holds.)
    }

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // the masks the device needs that the mesh does not carry
    std::vector<int> aFixes, aFlag, takeU, uFixes;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        // EVERY DEVICE BOUNDARY ARRAY IS [non-coupled patches, in patch order] (device_mesh.cuh:41-44).
        // A loop over all of fvp here makes the mask longer than the device's boundary-face count and
        // shifts every patch after the pair onto the wrong faces.
        if (isCoupledInterfaceType(fvp[pi].type)) continue;
        const int fx = f.alpha1.boundary[pi]->fixesValue() ? 1 : 0;
        const int fl = (fvp[pi].type == "empty") ? 1 : ((fvp[pi].type == "wedge") ? 2 : 0);
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            aFixes.push_back(fx);
            aFlag.push_back(fl);
            takeU.push_back(f.U.boundary[pi]->assignable() ? 0 : 1);
            uFixes.push_back(f.U.boundary[pi]->fixesValue() ? 1 : 0);
        }
    }
    DeviceBuffer<int> dAFixes(aFixes), dAFlag(aFlag), dTakeU(takeU), dUFixes(uFixes);
    // ...and one of them is NOT a property of the patch but of the face and the instant. alpha's
    // variableHeightFlowRate is a MIXED condition whose rebuild() sets valueFraction to 1 on an INFLOW
    // face and 0 on the rest (fv_patch_field.cuh: `if (!(phi < -SMALL)) continue`), so a mask taken once
    // from fixesValue() answers per patch a question OpenFOAM asks per face per update, and MULES limits
    // with it. Refreshed wherever alpha's boundary is re-evaluated, from the patch's OWN valueFraction,
    // and left alone for every other condition.
    bool hasPerFaceAlphaFixes = false;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (f.alpha1.boundary[pi]->isVariableHeightFlowRate()) hasPerFaceAlphaFixes = true;
    }
    auto refreshAlphaFixes = [&]()
    {
        if (!hasPerFaceAlphaFixes) return;
        std::vector<int> fx;
        fx.reserve(aFixes.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            const std::vector<scalar>* vf = f.alpha1.boundary[pi]->isVariableHeightFlowRate()
                                          ? f.alpha1.boundary[pi]->valueFractionPtr() : nullptr;
            const int patchFx = f.alpha1.boundary[pi]->fixesValue() ? 1 : 0;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                fx.push_back(vf && static_cast<std::size_t>(i) < vf->size()
                             ? ((*vf)[static_cast<std::size_t>(i)] > scalar(0.5) ? 1 : 0)
                             : patchFx);
            }
        }
        dAFixes.copyFrom(fx);
    };

    // THE BOUNDARY FLUX, declared above the hooks because they read it. OpenFOAM's flux-conditional
    // patches look phi up whenever they update; brae's are told, and the device path holds the only
    // current copy. See pushFluxToPatches for what not telling them cost on capillaryRise.
    DeviceBuffer<scalar> dPhiB(flattenPatches(f.phi.boundary, fvp));
    // rhoPhi, which the alpha step writes. Declared HERE because pushFlux reads its boundary: a
    // condition may name `phi rhoPhi;` (three tutorials' totalPressure top does), and the device holds
    // the only current copy. Empty until the first alpha step, when the host's -- built at rest by
    // buildInterFields -- is the field.
    DeviceBuffer<scalar> dRpI, dRpB;
    bool namesRhoPhi = false;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (f.p_rgh.boundary[pi]->fluxName() == "rhoPhi" || f.alpha1.boundary[pi]->fluxName() == "rhoPhi")
        {
            namesRhoPhi = true;
        }
    }
    auto unflatten = [&](const DeviceBuffer<scalar>& d, std::vector<std::vector<scalar>>& out)
    {
        std::vector<scalar> flat;
        d.copyTo(flat);
        // ...and a coupled patch is NOT in the array, so its entry keeps whatever the host holds and
        // the offset does not advance over it. Walking every patch here reads the faces of the next
        // patch and runs off the end at the last.
        out.resize(fvp.size());
        std::size_t off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
            out[pi].assign(flat.begin() + off, flat.begin() + off + n);
            off += n;
        }
    };
    auto pushFlux = [&](bool uPatchesStillUpdated = false)
    {
        unflatten(dPhiB, f.phi.boundary);
        // ...and the PAIR's own faces, which that array does not carry. Without this a
        // porousBafflePressure computes its jump from a flux that is still zero: measured on the
        // gate's `jump` profile, max|jump| 0 at every assembly and the device running the
        // plain-cyclic answer (alpha 4.9e-02 from the host, exactly the control's distance).
        if (cycPhiForHost && !cyclics.empty())
        {
            std::vector<scalar> pif;
            cycPhiForHost->copyTo(pif);
            std::size_t off = 0;
            for (const CyclicInterface& c : cyclics)
            {
                const std::size_t pi = static_cast<std::size_t>(c.patch);
                const std::size_t n = c.faceCells.size();
                if (pi < f.phi.boundary.size() && off + n <= pif.size())
                {
                    f.phi.boundary[pi].assign(pif.begin() + off, pif.begin() + off + n);
                }
                off += n;
            }
        }
        if (namesRhoPhi && dRpB.size() == dPhiB.size())
        {
            unflatten(dRpB, f.rhoPhi.boundary);
        }
        pushFluxToPatches(f, fvp, uPatchesStillUpdated);
    };

    // THE WAVE CONDITIONS' CLOCK: OpenFOAM's time and time index for the step being taken, which the
    // hooks need and the loop below sets before each step. See inter_waves_cpp.cuh.
    scalar stepTime = 0;
    scalar stepDeltaT = 0;
    label stepIndex = 0;

    // TURBULENCE. The closure is the device's (device_inter_turbulence.cuh). BRAE_INTER_HOST_CLOSURE=1
    // puts the HOST reference in its place inside the same device loop -- the mixed build the closure
    // was wired against, kept because it is the one comparison that says whether a disagreement is the
    // closure's or the loop's. It is a gate's instrument, not a mode, and it says so every run.
    const bool hostClosure = f.turbulence.on && std::getenv("BRAE_INTER_HOST_CLOSURE") != nullptr;
    const bool deviceClosure = f.turbulence.on && !hostClosure;
    if (hostClosure)
    {
        std::printf("  *** BRAE_INTER_HOST_CLOSURE: the device loop is running the HOST kEpsilon. ***\n");
    }
    // (an inletOutlet nut runs on this loop now: deviceCorrectInterTurbulence evaluates it against the
    // flux after the closure, where the host closure does. Gated on RAS/damBreak `nutAtmosphere`.)
    DeviceInterTurbulence dTurb = deviceClosure
        ? buildDeviceInterTurbulence(f.turbulence, f.U, m, g, fvp)
        : DeviceInterTurbulence();
    // what the closure reads after the step, kept by the interfaceForces hook: rho's patch values as
    // the device step blended them, and the mixture's nu on cells and patches
    std::vector<std::vector<scalar>> stepRhoBnd;
    DeviceBuffer<scalar> dStepRhoBnd, dStepNu, dStepNuBnd;

    // the hooks: every one is per-patch host work, and nothing else
    DeviceInterStepHooks H;
    H.alpha.updateBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd, DeviceBuffer<scalar>& nBnd)
    {
        pushFlux();
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        aBnd.copyFrom(patchValues(f.alpha1, fvp));
        // The boundary viscosity from alpha's patch values as they stand HERE, before the curvature
        // pass below rewrites the contact-angle gradient: mixture.correct() is calcNu() and THEN
        // interfaceProperties::correct(). The last call of a step is the one UEqn reads. See the host
        // driver's mixtureCorrect stage for the measurement.
        updateMixtureBoundary(f, fvp);
        refreshAlphaFixes();
        SurfaceScalarField nHb;
        std::vector<scalar> Kb;
        interfaceProps::calculateK(f.alpha1, f.interface, m, g, fvp, false, nHb, Kb);
        nBnd.copyFrom(flattenPatches(nHb.boundary, fvp));
    };
    H.alpha.refreshBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd)
    {
        pushFlux();
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        aBnd.copyFrom(patchValues(f.alpha1, fvp));
    };
    if (f.waves.any)
    {
        H.alpha.updateModelledBoundary =
            [&](int subCycle, const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd)
        {
            // waveAlpha's updateCoeffs at the SUB-CYCLE's clock. The model reads alpha's and U's cell
            // values: alpha's are the device's, and f.U's are the last updateUBoundary's.
            a.copyTo(f.alpha1.internal);
            const SubCycleClock clock = subCycleClock(stepTime, stepDeltaT, stepIndex,
                                                      f.alphaCtl.nAlphaSubCycles, subCycle);
            updateWaveAlpha(f.waves, f.alpha1, f.U, clock.t, clock.timeIndex, m, g, fvp);
            f.alpha1.evaluateBoundary();
            aBnd.copyFrom(patchValues(f.alpha1, fvp));
        };
    }
    if (dyn)
    {
        // fvMesh::Vsc() and Vsc0() at this sub-cycle's clock, from the SAME call the host loop makes
        // (inter_driver_cpp.cu:514-532). On a mesh whose cells change volume these are not V: the
        // deforming-mesh tutorial reads alpha 1.8e-02 and U 5.8e-02 against the host arm with V in
        // their place, after five steps.
        H.alpha.subCycleVolumes =
            [&](int subCycle, DeviceBuffer<scalar>& Vsc, DeviceBuffer<scalar>& Vsc0)
        {
            SubCycleTimeState ts;
            ts.subCycling = f.alphaCtl.nAlphaSubCycles > 1;
            const SubCycleClock clock = subCycleClock(stepTime, stepDeltaT, stepIndex,
                                                      f.alphaCtl.nAlphaSubCycles, subCycle);
            ts.value   = clock.t;
            ts.deltaT  = stepDeltaT / static_cast<scalar>(f.alphaCtl.nAlphaSubCycles);
            ts.value0  = stepTime;
            ts.deltaT0 = stepDeltaT;
            Vsc.copyFrom(dyn->Vsc(ts));
            Vsc0.copyFrom(dyn->Vsc0(ts));
        };
    }
    H.alpha.divCoeffs =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        FvScalarMatrix M = fvm::div<scalar>(f.phi.internal, f.phi.boundary, f.alpha1, m, fvp);
        std::vector<scalar> i2, b2;
        // The device's boundary arrays hold the UNCOUPLED patches only, as its mesh does
        // (device_mesh.cuh:41-44): a coupled patch's coefficients are the interface's, and the alpha
        // pre-solve adds those itself from the pair. Flattening every patch here made the array longer
        // than the device's boundary-face count and the pre-solve refused it by size.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(M.internalCoeffs[pi][i]); b2.push_back(M.boundaryCoeffs[pi][i]); }
        }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };
    H.updateUBoundary =
        [&](const DeviceBuffer<scalar>& ux, const DeviceBuffer<scalar>& uy,
            const DeviceBuffer<scalar>& uz, DeviceVectorBoundary& db, DeviceBuffer<scalar>* ubOut,
            DeviceUBoundaryCall call)
    {
        // p_rgh's and alpha's patches take the new flux at every call; U's do not at the one call
        // where OpenFOAM's are still updated() -- see DeviceUBoundaryCall
        pushFlux(call == DeviceUBoundaryCall::evaluateStillUpdated);
        std::vector<scalar> x, y, z;
        ux.copyTo(x); uy.copyTo(y); uz.copyTo(z);
        for (label c = 0; c < nC; ++c) f.U.internal[c] = vector{x[c], y[c], z[c]};
        // waveVelocity's updateCoeffs, at the STEP's clock: the first call of a step is UEqn's, where
        // the model -- last updated inside the alpha sub-cycles -- updates again from the alpha they
        // left (f.alpha1 is the alpha hooks' last copy) and the U the step started on. Every later
        // call of the step is the same time index and re-assigns the same values.
        updateWaveVelocity(f.waves, f.alpha1, f.U, stepTime, stepIndex, m, g, fvp);
        // THE ASSEMBLY IS NOT AN EVALUATE. At the momentum assembly OpenFOAM's patches update their
        // coefficients and keep their STORED values, which is what the host loop does at the same
        // point (inter_driver_cpp.cu:602-611: the wave model, then the classes whose updateCoeffs
        // ends in evaluate(), and nothing else). Evaluating here is the same value bit for bit while
        // nothing a patch reads has moved since the last corrector's evaluate -- and the phase
        // fraction HAS, on a permeable wall: MEASURED on damBreakPermeable's staged wet wall, at the
        // step where the first face goes dry (81 of 140), U 1.3e-03 and p_rgh 1.5e-01 from the host
        // loop in that one step, from 2e-13 the step before.
        if (call != DeviceUBoundaryCall::assembly)
        {
            f.U.evaluateBoundary();
        }
        updateVelocityPatches(f.U, fvp);
        // U.boundaryFieldRef().updateCoeffs() for a flowRateInletVelocity, which the fvMatrix constructor
        // runs at every momentum assembly (fvMatrix.C:396): the rate at THIS time and, for a
        // massFlowRate, the field named `rho` on the patch, which in interFoam is the mixture's
        // (flowRateInletVelocity...C:201-237). The host driver does exactly this
        // (inter_driver_cpp.cu:715-720) and f.rhoBnd is the same blended boundary rho the device step
        // wrote; without it the inlet keeps the file's `value`, which on RAS/angledDuct is (0 0 0).
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (f.U.boundary[pi]->isFlowRateInlet())
            {
                f.U.boundary[pi]->updateFromDensity(f.rhoBnd[pi], stepTime);
            }
            // ...and a variableHeightFlowRateInletVelocity rebuilds itself there too, from the STORED
            // values of the phase field it names on this patch
            // (variableHeightFlowRateInletVelocityFvPatchVectorField.C:103-139), which is what the host
            // driver does at the same point (inter_driver_cpp.cu:722-735). f.alpha1's patch values are
            // the alpha hooks' last, which is the step's own alpha.
            if (f.U.boundary[pi]->isVariableHeightFlowRateInlet())
            {
                if (f.U.boundary[pi]->alphaFieldName() != f.alphaName)
                {
                    throw std::runtime_error(
                        "brae interFoam: U patch `" + fvp[pi].name + "` is a "
                        "variableHeightFlowRateInletVelocity naming `alpha "
                        + f.U.boundary[pi]->alphaFieldName() + "`, and this case's phase field is `"
                        + f.alphaName + "`. OpenFOAM looks the named field up and stops without it.");
                }
                f.U.boundary[pi]->updateFromAlphaPatch(f.alpha1.boundary[pi]->value(), stepTime);
            }
        }
        // MRF.correctBoundaryVelocity(U), UEqn.H:1, where the host driver has it
        // (inter_driver_cpp.cu:741-745): it overwrites U's values on the INCLUDED patches with the frame
        // velocity Omega x (Cf - origin) (MRFZone.C:499-526), so it has to run on the HOST field the
        // snapshot below is taken from -- once dbU exists the values are already copied. This is the
        // same precondition rhoSimpleFoam's device arm states (rhoUEqn.cuh:72-75).
        if (!f.mrfZones.empty())
        {
            MRF::correctBoundaryVelocity(f.U, f.mrfZones, fvp);
        }
        db = buildDeviceVectorBoundary(f.U, fvp, g);
        if (!ubOut) return;
        std::vector<scalar> bx, by, bz;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            for (const vector& u : f.U.boundary[pi]->value())
            { bx.push_back(u.x); by.push_back(u.y); bz.push_back(u.z); }
        }
        ubOut[0].copyFrom(bx);
        ubOut[1].copyFrom(by);
        ubOut[2].copyFrom(bz);
    };
    H.interfaceForces =
        [&](const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& Kd,
            const DeviceBuffer<scalar>& rd, const DeviceBuffer<scalar>& rhoBd,
            DeviceBuffer<scalar>& stf, DeviceBuffer<scalar>& snRho,
            DeviceBuffer<scalar>& nuC, DeviceBuffer<scalar>& nuB, DeviceBuffer<scalar>& snP)
    {
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        Kd.copyTo(f.K);
        rd.copyTo(f.rho);

        // surfaceTensionForce() = interpolate(sigma*K)*snGrad(alpha1), boundary INCLUDED -- at a
        // contact-angle wall that boundary term is the contact angle's only route into the equations.
        std::vector<scalar> sK;
        interfaceProps::sigmaK(f.K, f.interface.sigma, sK);
        const SurfaceScalarField sKf = fvc::interpolate(sK, m, g, fvp);
        // ...under the case's snGradSchemes, each correction through that field's own gradSchemes entry,
        // as the host driver takes them (inter_driver_cpp.cu, the UEqn stage). This read `false` -- the
        // orthogonal form -- whatever the case said; the mesh refusal below kept that from being a
        // silent substitution, and the three calls here are what lifts it.
        const bool snCorr = f.snGradScheme.corrected;
        const scalar snLim = f.snGradScheme.limitCoeff;
        const SurfaceScalarField snA = fvc::snGrad(f.alpha1, m, g, fvp, snCorr,
                                                   f.gradAlpha1.leastSquares, f.gradAlpha1.cellLimitK, snLim);
        SurfaceScalarField t;
        t.internal.resize(static_cast<std::size_t>(nIf));
        for (label i = 0; i < nIf; ++i) t.internal[i] = sKf.internal[i]*snA.internal[i];
        t.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                t.boundary[pi].push_back(sKf.boundary[pi][i]*snA.boundary[pi][i]);
        stf.copyFrom(fullFace(t, fvp));
        // ...and its half ON THE PAIR, which fullFace drops because the device's boundary arrays
        // exclude coupled patches. fvc::interpolate and fvc::snGrad both fill a coupled patch above,
        // so this is the same number the host reference reads there.
        if (!cyclics.empty()) dStfIf.copyFrom(coupledFace(t, cyclics));

        // rho's CALCULATED patch values, from the device's own blend (which carries alpha2's
        // one-pass-older patch values) -- not a zeroGradient copy. See rhoWithPatchValues.
        std::vector<scalar> rbFlat;
        rhoBd.copyTo(rbFlat);
        std::vector<std::vector<scalar>> rb(fvp.size());
        {
            // THE DEVICE'S BOUNDARY ARRAY HAS NO COUPLED PATCHES IN IT (device_mesh.cuh:41-44), so
            // walking every patch here reads the wrong faces for every patch after the first coupled
            // one and past the end of the array at the last. MEASURED on validation/interFoamCyclic,
            // where it put rho's wall values on the pair: p_rgh's shape 2.4e+02 of 1.6e+03 away from
            // the host at the FIRST step, with alpha still identical; damBreak, which has no pair,
            // read 5.3e-12 at the same point. A coupled patch's rho is its own two cells interpolated,
            // which is what its patch field returns and what rhoWithPatchValues builds there.
            std::size_t off = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                const FvPatch& q = fvp[pi];
                const std::size_t n = static_cast<std::size_t>(q.size);
                if (isCoupledInterfaceType(q.type))
                {
                    rb[pi].resize(n);
                    for (label i = 0; i < q.size; ++i)
                    {
                        rb[pi][static_cast<std::size_t>(i)] = coupledLinear(q, i, f.rho);
                    }
                    continue;
                }
                rb[pi].assign(rbFlat.begin() + off, rbFlat.begin() + off + n);
                off += n;
            }
        }
        // ...kept: the closure reads rho's patch values after the step, and these are the device's
        stepRhoBnd = rb;
        if (deviceClosure)
        {
            deviceCopy(dStepRhoBnd, rhoBd);
        }
        const GeometricField<scalar> rhoF = rhoWithPatchValues(f.rho, rb, fvp);
        const SurfaceScalarField snRhoF = fvc::snGrad(rhoF, m, g, fvp, snCorr, f.gradRho.leastSquares,
                                                      f.gradRho.cellLimitK, snLim);
        snRho.copyFrom(fullFace(snRhoF, fvp));
        if (!cyclics.empty()) dSnRhoIf.copyFrom(coupledFace(snRhoF, cyclics));

        // the mixture's own nu. NOTE mixtureNu's second argument is mu, not alpha2.
        cpu::twoPhase::mixtureMu(f.alpha1.internal, f.mixture.phases, f.mu);
        cpu::twoPhase::mixtureNu(f.alpha1.internal, f.mu, f.mixture.phases, f.nu);
        // ...and the COUPLED patches' share of the mixture boundary, which is the two CELLS'
        // interpolated (updateMixtureBoundary's coupled branch, inter_case_cpp.cu:141-163) and so is
        // only as current as f.rho, f.mu and f.nu -- which this hook has just rebuilt. The alpha hook
        // ran updateMixtureBoundary one stage earlier, on the PREVIOUS step's cell fields, and its
        // ordering there is deliberate: a contact-angle wall must be blended before the curvature
        // pass moves alpha's patch value. So only the pair is refreshed here, where the host's single
        // call already has current cells (inter_driver_cpp.cu:552).
        //
        // MEASURED on validation/interFoamCyclic's `jump` profile, where porousBafflePressure reads
        // exactly these patch values: without this the device's nu and rho on the pair stayed at their
        // step-one values (nu 7.90e-06 against the host's 1.08e-06 at step two), the jump came out 1%
        // wrong on its largest face, and U was 3.8e-03 from the host by step ten.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (!fvp[pi].coupled) continue;
            const std::size_t n = f.rhoBnd.size() > pi ? f.rhoBnd[pi].size() : 0;
            for (std::size_t i = 0; i < n; ++i)
            {
                const label k = static_cast<label>(i);
                f.rhoBnd[pi][i] = coupledLinear(fvp[pi], k, f.rho);
                f.muBnd[pi][i]  = coupledLinear(fvp[pi], k, f.mu);
                f.nuBnd[pi][i]  = coupledLinear(fvp[pi], k, f.nu);
            }
        }
        // f.nuBnd is the alpha hook's, taken BEFORE its curvature pass. Rebuilding it here would read
        // alpha's patch values one contact-angle pass too late.
        // nuEff = nut + nu: the mixture's nu alone on a laminar case. nut is the closure's, as the
        // LAST step's correct() left it -- or validate()'s, or the case file's, at the first.
        if (deviceClosure)
        {
            // the mixture's nu alone: the step adds the device's nut (DeviceInterStepControls::nutCell)
            nuC.copyFrom(f.nu);
            nuB.copyFrom(flattenPatches(f.nuBnd, fvp));
            deviceCopy(dStepNu, nuC);
            deviceCopy(dStepNuBnd, nuB);
        }
        else
        {
            std::vector<scalar> nuEff;
            std::vector<std::vector<scalar>> nuEffB;
            interNuEff(f.turbulence, f.nu, f.nuBnd, nuEff, nuEffB);
            nuC.copyFrom(nuEff);
            nuB.copyFrom(flattenPatches(nuEffB, fvp));
        }

        // snGrad(p_rgh) is read ONLY when the case runs a momentum predictor; it is explicit there and
        // implicit in the pressure equation, and carrying it into both would count it twice.
        if (f.momentumPredictorOn)
        {
            // the gradient the last constrainPressure left, and zero only before the first -- see the
            // host driver's UEqn stage for what zeroing it every step cost
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (!f.p_rgh.boundary[pi]->updateableSnGrad()) continue;
                if (f.p_rgh.boundary[pi]->snGradEverSet()) continue;
                f.p_rgh.boundary[pi]->updateSnGrad(
                    std::vector<scalar>(static_cast<std::size_t>(fvp[pi].size), scalar(0)));
            }
            snP.copyFrom(fullFace(fvc::snGrad(f.p_rgh, m, g, fvp, f.snGradScheme.corrected,
                                              f.gradPrgh.leastSquares, f.gradPrgh.cellLimitK,
                                              f.snGradScheme.limitCoeff), fvp));
        }
        else snP.resize(0);
    };
    H.pressure.pressureCoeffs =
        [&](const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>& phiHB,
            const DeviceBuffer<scalar>& rAUfAll, const DeviceBuffer<scalar>& rAUCell,
            DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC, DeviceBuffer<scalar>& cycJump)
    {
        // THE PAIR'S FLUX, as it stands at THIS assembly. porousBafflePressure's jump is built from it
        // below and the host pEqn takes it from the phi the LAST CORRECTOR wrote, so it is refreshed
        // here rather than left to the step's own pushFlux, which runs once per corrector and not once
        // per non-orthogonal pass.
        if (cycPhiForHost && !cyclics.empty())
        {
            std::vector<scalar> pif;
            cycPhiForHost->copyTo(pif);
            std::size_t off = 0;
            for (const CyclicInterface& c : cyclics)
            {
                const std::size_t pj = static_cast<std::size_t>(c.patch);
                const std::size_t n = c.faceCells.size();
                if (pj < f.phi.boundary.size() && off + n <= pif.size())
                {
                    f.phi.boundary[pj].assign(pif.begin() + off, pif.begin() + off + n);
                }
                off += n;
            }
        }
        std::vector<scalar> hB, rA, rAUc;
        phiHB.copyTo(hB);
        rAUfAll.copyTo(rA);
        rAUCell.copyTo(rAUc);
        // constrainPressure: a fixedFluxPressure gradient is PRESCRIBED from phiHbyA, and brae refuses
        // to assemble one that has not been set. rAUf is taken PER FACE from the full array.
        label off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const FvPatch& q = fvp[pi];
            // the device's phiHbyA and rAUf arrays have no coupled patch in them, and a cyclic never
            // prescribes a gradient anyway
            if (isCoupledInterfaceType(q.type)) continue;
            if (f.p_rgh.boundary[pi]->updateableSnGrad())
            {
                // (phiHbyA_b - (Sf_b & U_b))/(magSf_b*rAUf_b): the VELOCITY's flux, not the stored
                // phi_b -- see pressureCorrector, and tests/interfoam_waves_vs_openfoam.sh for what
                // the difference is worth on a patch whose U_b changes. f.U's patch values are current:
                // updateUBoundary ran after the last corrector.
                const std::vector<vector>& ub = f.U.boundary[pi]->value();
                std::vector<scalar> sn(static_cast<std::size_t>(q.size));
                for (label i = 0; i < q.size; ++i)
                {
                    const scalar SfU = dot(g.Sf()[q.start + i], ub[i]);
                    sn[i] = (hB[off + i] - SfU) / (q.magSf[i] * rA[nIf + off + i]);
                }
                // prghPermeableAlphaTotalPressure rebuilds its refValue and valueFraction INSIDE
                // updateSnGrad, from rho, phi and U on the patch and gh at the face centres
                // (...FvPatchScalarField.C:151-212), which the host pEqn does at this same point
                // (inter_peqn_cpp.cu:761-771). The phi is the field as it stands -- the last
                // corrector's, which the step's last pushFlux unflattened into f.phi -- not phiHbyA.
                if (f.p_rgh.boundary[pi]->isPrghPermeableAlphaTotalPressure())
                {
                    if (f.rhoBnd.size() <= pi || f.ghfBoundary.size() <= pi || f.phi.boundary.size() <= pi)
                    {
                        throw std::runtime_error(
                            "brae interFoam (device): p_rgh patch `" + q.name + "` is a "
                            "prghPermeableAlphaTotalPressure, which needs rho's patch values, gh at the "
                            "patch's face centres and phi on the patch, and the driver has none for it.");
                    }
                    f.p_rgh.boundary[pi]->updatePermeableTotalPressure(f.rhoBnd[pi], f.phi.boundary[pi], ub,
                                                                       f.ghfBoundary[pi]);
                }
                f.p_rgh.boundary[pi]->updateSnGrad(sn);
            }
            off += q.size;
        }
        SurfaceScalarField rf;
        rf.internal.assign(rA.begin(), rA.begin() + nIf);
        rf.boundary.resize(fvp.size());
        { label o = nIf;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          {
              const FvPatch& q = fvp[pi];
              // a coupled patch's rAUf is not in the device's array: it is fvc::interpolate(rAU)
              // there, which is the two cells' rAU on the pair's own weights. fvm::laplacian reads
              // gammaf.boundary on that patch, so it cannot be left empty.
              if (isCoupledInterfaceType(q.type))
              {
                  for (label i = 0; i < q.size; ++i)
                  {
                      rf.boundary[pi].push_back(coupledLinear(q, i, rAUc));
                  }
                  continue;
              }
              for (label i = 0; i < q.size; ++i) rf.boundary[pi].push_back(rA[o++]);
          } }
        // totalPressure's updateCoeffs, where the fvMatrix constructor runs it. f.U's patch values and
        // f.phi's are current (updateUBoundary ran after the last corrector and pushed the flux); a
        // totalPressure patch is never a contact-angle wall, so f.rhoBnd is exact on it.
        updatePressurePatchesFromVelocity(f.p_rgh, f.U, &f.rhoBnd, fvp);
        // porousBafflePressure::updateCoeffs, which the fvMatrix constructor runs at THIS assembly:
        // the OWNER's jump from phi as it stands -- the last corrector's, not phiHbyA -- and from the
        // stored patch values of the laminar nu and of rho; the other side takes the owner's. The
        // host pEqn does this inside its own corrector loop (inter_peqn_cpp.cu:723-740); on the
        // device the assembly is split across the hook and the step, and this is the hook's half.
        // Without it the jump stays at the file's value and the device ran the PLAIN-CYCLIC answer --
        // measured on the gate's `jump` profile, alpha 4.9e-02 and U 45% from the host, which is
        // exactly the control's distance.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (!fvp[pi].owner || !f.p_rgh.boundary[pi]->isPorousBafflePressure()) continue;
            if (f.rhoBnd.size() <= pi || f.nuBnd.size() <= pi)
            {
                throw std::runtime_error(
                    "brae interFoam (device): p_rgh patch `" + fvp[pi].name + "` is a "
                    "porousBafflePressure, which needs the patch values of rho and of the mixture's "
                    "laminar nu, and the driver has none for it.");
            }
            const std::vector<scalar> jump = f.p_rgh.boundary[pi]->porousBaffleJump(
                namedPatchFlux(f.p_rgh.boundary[pi]->fluxName(), pi, fvp[pi].name, f.phi, &f.rhoPhi),
                f.nuBnd[pi], f.rhoBnd[pi]);
            f.p_rgh.boundary[pi]->setOwnerJump(jump);
            f.p_rgh.boundary[static_cast<std::size_t>(fvp[pi].nbrPatch)]->setOwnerJump(jump);
        }
        FvScalarMatrix pe = fvm::laplacian<scalar>(rf, f.p_rgh, m, g, fvp, false);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;   // the pair's are the interface's
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(pe.internalCoeffs[pi][i]); b2.push_back(pe.boundaryCoeffs[pi][i]); }
        }
        iC.copyFrom(i2);
        bC.copyFrom(b2);

        // p_rgh's JUMP on the pair, as the updates above have just left it. porousBafflePressure
        // recomputes it in updateCoeffs from THIS assembly's flux and viscosity, so it is taken here
        // and not once at the start. The value a patch returns is already signed -- the owner's on
        // the owner side and its negative on the other (fixedJumpFvPatchField::jump, and
        // fv_patch_field.cuh's setOwnerJump) -- which is the sign the matrix wants.
        std::vector<scalar> jf;
        bool anyJump = false;
        for (const CyclicInterface& c : cyclics)
        {
            const std::size_t pj = static_cast<std::size_t>(c.patch);
            const std::vector<scalar>* j = f.p_rgh.boundary[pj]->coupledJump();
            if (j && !j->empty())
            {
                anyJump = true;
            }
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                jf.push_back((j && i < j->size()) ? (*j)[i] : scalar(0));
            }
        }
        if (anyJump)
        {
            cycJump.copyFrom(jf);
        }
        else
        {
            cycJump.resize(0);        // no jump on this pair: the kernels take the plain neighbour
        }
    };
    // p_rgh's stored patch values, for the corrected laplacian's grad(p_rgh): what the host's
    // gradOf(p_rgh) reads, after pressureCoeffs has run the patches' updates
    H.pressure.boundaryValues = [&](DeviceBuffer<scalar>& bval)
    {
        std::vector<scalar> flat;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            const std::vector<scalar>& v = f.p_rgh.boundary[pi]->value();
            flat.insert(flat.end(), v.begin(), v.end());
        }
        bval.copyFrom(flat);
    };
    H.pressure.updateBoundary = [&](const DeviceBuffer<scalar>& pr)
    {
        pr.copyTo(f.p_rgh.internal);
        f.p_rgh.evaluateBoundary();
    };

    // THE MESH'S PERIODIC PAIR on the device. The caller attaches the coupling before it hands the
    // patches over (attachCyclicCoupling, braeInterFoam.cu:166), so a coupled patch here is one whose
    // neighbour cells and weights are already filled. Its VOLUMETRIC FLUX is state of the same kind as
    // phi: seeded from the field the case started with, rewritten by every pressure corrector.
    DeviceCyclic dCyc = buildDeviceCyclic(cyclics, g, fvp);
    DeviceBuffer<scalar> dAlphaPhiIf, dRhoPhiIf;
    if (dCyc.n > 0)
    {
        std::vector<scalar> seed;
        for (const CyclicInterface& c : cyclics)
        {
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                seed.push_back(f.phi.boundary[static_cast<std::size_t>(c.patch)][i]);
            }
        }
        dCyc.phi.copyFrom(seed);
        dAlphaPhiIf.copyFrom(std::vector<scalar>(static_cast<std::size_t>(dCyc.n), scalar(0)));
        dRhoPhiIf.copyFrom(std::vector<scalar>(static_cast<std::size_t>(dCyc.n), scalar(0)));
    }

    // THE CASE'S MRF ZONES on the device, built from the host's validated cpu::MRF::Zone rather than from
    // a second face classification (device_MRF.cuh). The geometry is static and Omega constant, so the
    // per-face frame flux is precomputed once here. Three of interFoam's four MRF calls are the step's
    // (DDt, zeroFilter, makeRelative); the fourth is in the U-boundary hook above.
    // fvOptions' porosity on the device, built from the HOST OptionList's own transformed tensors so
    // there is no second reading of the dictionary and no second coordinate transform. RAS/angledDuct
    // rotates e1 by 45 degrees, so D and F have off-diagonal entries and the diagonal d/f form the other
    // device solvers use (and refuse a rotated system for) cannot carry them.
    DevicePorosity dPorosity;
    for (const fvOptions::Option& o : f.fvOptions.options)
    {
        if (!o.active || !o.unsupported.empty() || o.fixedCoeff) continue;
        if (o.type != "explicitPorositySource") continue;
        if (dPorosity.active)
        {
            throw std::runtime_error(
                "brae interFoam (device): more than one active explicitPorositySource. The device step "
                "carries one zone. The host loop carries them all.");
        }
        std::vector<label> cells = o.cells;
        if (o.allCells)
        {
            cells.resize(static_cast<std::size_t>(m.nCells()));
            for (label c = 0; c < m.nCells(); ++c) cells[static_cast<std::size_t>(c)] = c;
        }
        dPorosity.active = true;
        dPorosity.tensorForm = true;
        dPorosity.cells.copyFrom(cells);
        const scalar* D = &o.D.xx;
        const scalar* F = &o.F.xx;
        for (int k = 0; k < 9; ++k)
        {
            dPorosity.dT[k] = D[k];
            dPorosity.fT[k] = F[k];
        }
    }

    std::vector<DeviceMRFZone> dMrf;
    for (const cpu::MRF::Zone& z : f.mrfZones)
    {
        dMrf.push_back(buildDeviceMRFZone(z, m, g, fvp));
    }

    // ---- controls --------------------------------------------------------------------------------
    DeviceInterStepControls C;
    C.alpha.nAlphaSubCycles = static_cast<int>(f.alphaCtl.nAlphaSubCycles);
    C.alpha.nAlphaCorr      = static_cast<int>(f.alphaCtl.nAlphaCorr);
    C.alpha.MULESCorr       = f.alphaCtl.MULESCorr;
    // THE CASE'S OWN alpha SOLVE: its smoother where it names a Gauss-Seidel one (the device has
    // OpenFOAM's, level-scheduled and exact), and its tolerances either way. See deviceAlphaPreSolve.
    // `alphaApplyPrevCorr`: the cache outlives every step, so it lives here. See
    // DeviceInterAlphaControls -- the device ignored this switch until it was measured.
    DeviceBuffer<scalar> dPrevCorrI, dPrevCorrB;
    C.alpha.alphaApplyPrevCorr = f.alphaCtl.alphaApplyPrevCorr;
    C.alpha.prevCorrInt = &dPrevCorrI;
    C.alpha.prevCorrBnd = &dPrevCorrB;
    // gradSchemes: the device operators take Gauss linear and unlimited, apart from grad(U)'s cellLimited
    // (refused above); the host takes leastSquares and cellLimited on every gradient
    for (const GradChoice* gc : {&f.gradAlpha1, &f.gradAlpha2, &f.gradPrgh, &f.gradPcorr, &f.gradRho,
                                 &f.interface.nHatGrad})
    {
        if (!gc->gaussLinear() || f.gradULeastSq)
            throw std::runtime_error(
                "brae interFoam (device): fvSchemes gradSchemes names a leastSquares or cellLimited "
                "gradient for alpha, p_rgh, pcorr, rho, U or nHat. The host loop takes each by its own entry "
                "(gated on laminar/damBreak `gradLsqLimited`); the device operators are Gauss linear. "
                "Refused rather than run another gradient.");
    }
    if (f.alphaCtl.MULESCorr && f.aSolve.minIter > 0)
        throw std::runtime_error(
            "brae interFoam (device): `solvers/" + f.alphaName + "` names `minIter "
            + std::to_string(f.aSolve.minIter) + "`, which the device's alpha pre-solve does not honour. "
            "The host loop does (gated on laminar/damBreak `alphaminiter`). Refused rather than stop a "
            "sweep earlier than OpenFOAM does.");
    C.alpha.preSolve.tol = f.aSolve.tol;
    C.alpha.preSolve.relTol = f.aSolve.relTol;
    C.alpha.preSolve.maxIter = f.aSolve.maxIter;
    C.alpha.preSolve.smoothSolver = f.aSolve.gaussSeidel();
    C.alpha.preSolve.symmetric = (f.aSolve.smoother == "symGaussSeidel");
    C.alpha.preSolve.nSweeps = f.aSolve.nSweeps;
    C.mules = DeviceMulesControls{f.mulesCtl.nLimiterIter, f.mulesCtl.smoothLimiter,
                                  f.mulesCtl.extremaCoeff, f.mulesCtl.boundaryExtremaCoeff};
    C.alphaInput.cAlpha = f.interface.cAlpha;
    C.alphaInput.deltaN = interfaceProps::deltaN(g.V());
    // The alpha fluxes' schemes, EVERY ONE NAMED and neither mapping ending in a fall-through: both
    // used to, and `Gauss interfaceCompression` would have been taken as vanLeer on one flux and as
    // linear on the other without a word.
    auto alphaScheme = [](AlphaFluxScheme s, const char* which) -> DeviceAlphaScheme
    {
        switch (s)
        {
            case AlphaFluxScheme::linear:  return DeviceAlphaScheme::linear;
            case AlphaFluxScheme::upwind:  return DeviceAlphaScheme::upwind;
            case AlphaFluxScheme::vanLeer: return DeviceAlphaScheme::vanLeer;
            case AlphaFluxScheme::interfaceCompression:
                return DeviceAlphaScheme::interfaceCompression;
        }
        throw std::runtime_error(
            std::string("brae interFoam -device: the case's ") + which + " scheme is not one this "
            "loop implements (linear, upwind, vanLeer, interfaceCompression).");
    };
    C.alphaInput.alphaScheme  = alphaScheme(f.divPhiAlpha, "div(phi,alpha)");
    C.alphaInput.alpharScheme = alphaScheme(f.divPhirbAlpha, "div(phirb,alpha)");
    // EVERY SCHEME NAMED, NO default: this switch used to end in `default: upwind`, and interFoam's
    // `linear` -- which its own enum carries and three shipped tutorials name -- fell through it, so
    // a -device run of such a case would have convected upwind under the name `linear`.
    switch (f.divRhoPhiU)
    {
        case DivScheme::linearUpwind:   C.divScheme = brae::cpu::DivScheme::linearUpwind;   break;
        case DivScheme::linearUpwindV:  C.divScheme = brae::cpu::DivScheme::linearUpwindV;  break;
        case DivScheme::limitedLinear:  C.divScheme = brae::cpu::DivScheme::limitedLinear;  break;
        case DivScheme::limitedLinearV: C.divScheme = brae::cpu::DivScheme::limitedLinearV; break;
        case DivScheme::LUST:           C.divScheme = brae::cpu::DivScheme::LUST;           break;
        case DivScheme::vanLeerV:       C.divScheme = brae::cpu::DivScheme::vanLeerV;       break;
        case DivScheme::linear:         C.divScheme = brae::cpu::DivScheme::linear;         break;
        case DivScheme::upwind:         C.divScheme = brae::cpu::DivScheme::upwind;         break;
    }
    C.divSchemeCoeff = f.divRhoPhiUCoeff;
    C.nCorrectors = static_cast<int>(f.pimple.nCorrectors);
    C.mrf = f.mrfZones.empty() ? nullptr : &dMrf;
    C.cyc        = (dCyc.n > 0) ? &dCyc : nullptr;
    C.alphaPhiIf = (dCyc.n > 0) ? &dAlphaPhiIf : nullptr;
    C.rhoPhiIf   = (dCyc.n > 0) ? &dRhoPhiIf : nullptr;
    // ...and phig's three fields on the pair. dGhfIf is filled below, before the first step.
    C.stfIf       = (dCyc.n > 0) ? &dStfIf : nullptr;
    C.ghfIf       = (dCyc.n > 0) ? &dGhfIf : nullptr;
    C.snGradRhoIf = (dCyc.n > 0) ? &dSnRhoIf : nullptr;
    C.nHatfIf     = (dCyc.n > 0) ? &dNHIf : nullptr;
    cycPhiForHost = (dCyc.n > 0) ? &dCyc.phi : nullptr;
    C.porosity = dPorosity.active ? &dPorosity : nullptr;
    // p_rgh's reference, where the host driver sets it (inter_driver_cpp.cu:779-781). These three were
    // dead for as long as the device refused a case that needs one, and a step that never pins leaves the
    // singular system's level to the solver: MEASURED on laminar/mixerVessel2D before they were set, a
    // constant -2.17e+01 in p_rgh with a spread of only 1.5e-02 about it.
    C.needReference = f.pRef.needReference;
    C.pRefCell      = static_cast<int>(f.pRef.pRefCell);
    C.pRefValue     = f.pRef.pRefValue;
    C.nNonOrthogonalCorrectors = static_cast<int>(f.nNonOrthogonalCorrectors);
    C.correctedLaplacian = f.laplacianScheme.corrected;
    C.snGradLimitCoeff = f.laplacianScheme.limitCoeff;
    C.momentumPredictor = f.momentumPredictorOn;
    C.relaxU = f.relaxU;
    C.relaxEquationU = f.relaxEquationU;
    // Both entries, selected per corrector inside the step -- and the case's own PCG with DIC where it
    // names one, which every shipped interFoam tutorial does. The DIC is the level-scheduled DILU with
    // lower aliased to upper, bit-identical to DICPreconditioner.C (tests/test_device_dic.cu). Any other
    // solver still runs the device BiCGStab, under the notice buildInterFields already printed.
    //
    // ...AND THE CASE'S OWN GAMG, where it names that: OpenFOAM's V-cycle on the device
    // (device_gamg_solver.cuh) on the host-built faceAreaPair hierarchy. The device's has the DIC
    // smoother, which is what every shipped interFoam tutorial that names GAMG asks for; the other
    // three the host runs are refused here rather than run as DIC.
    for (const InterFields::PressureLinearSolve* entry : {&f.pSolve, &f.pSolveFinal})
    {
        // ...and the GAMG PRECONDITIONER, which this loop runs too (devicePcgGamgSolve): the same
        // PCG, with that sub-dictionary's V-cycles in place of DIC's sweeps. Its smoother is checked
        // like the solver's, on the SUB-DICTIONARY's entry, because that is where it is written.
        const std::string& smoother = entry->pcgGamg() ? entry->gamgPrecond.gamg.smoother
                                                       : entry->gamg.smoother;
        if ((entry->gamgSolver() || entry->pcgGamg()) && !deviceGamgSmootherPorted(smoother))
        {
            throw std::runtime_error(
                "brae interFoam -device: fvSolution's GAMG entry for p_rgh asks for `smoother "
                + smoother + "`, which the device's GAMG does not run (device_gamg_solver.cuh has "
                "DIC, DICGaussSeidel, GaussSeidel and symGaussSeidel). Refused rather than smoothed "
                "with something the case did not name.");
        }
    }
    DeviceDilu dic = buildDeviceDilu(m.owner(), m.neighbour(), nC);
    C.dic = &dic;
    std::vector<DeviceSolverPerf> pLog, aLog, uLog[3];
    C.momentumSolveLog = uLog;
    C.pressureSolveLog = &pLog;
    C.alpha.preSolveLog = &aLog;
    C.pressurePcgDIC = f.pSolve.pcgDIC();
    C.pressureFinalPcgDIC = f.pSolveFinal.pcgDIC();
    DeviceGamgCache gamgCache;
    gamgCache.mesh = &m;
    gamgCache.geometry = &g;
    gamgCache.host = &meshAgglomeration;
    GamgSolveLog gamgLog;
    C.pressureGamg = f.pSolve.gamgSolver() ? &f.pSolve.gamg : nullptr;
    C.pressureFinalGamg = f.pSolveFinal.gamgSolver() ? &f.pSolveFinal.gamg : nullptr;
    C.pressurePcgGamg = f.pSolve.pcgGamg() ? &f.pSolve.gamgPrecond : nullptr;
    C.pressureFinalPcgGamg = f.pSolveFinal.pcgGamg() ? &f.pSolveFinal.gamgPrecond : nullptr;
    C.gamgCache = &gamgCache;
    C.gamgLog = &gamgLog;
    C.pressure.tol = f.pSolve.tol;
    C.pressure.relTol = f.pSolve.relTol;
    C.pressure.maxIter = f.pSolve.maxIter;
    C.pressureFinal.tol = f.pSolveFinal.tol;
    C.pressureFinal.relTol = f.pSolveFinal.relTol;
    C.pressureFinal.maxIter = f.pSolveFinal.maxIter;
    // THE CASE'S OWN SOLVE FOR U, where a hardcoded 1e-12 on Jacobi-BiCGStab used to stand. Which
    // entry fvMatrix::solve() selects is the mesh's finalIteration flag: `UFinal` on the LAST outer
    // corrector and `U` on the others (inter_driver_cpp.cu:697-698, and the note on
    // InterFields::uSolve). Set per outer corrector in the loop below.
    auto setMomentumSolve = [&](bool finalOuter)
    {
        if (!f.momentumPredictorOn) return;
        const InterFields::AlphaLinearSolve& us = finalOuter ? f.uSolveFinal : f.uSolve;
        C.momentum.tol = us.tol;
        C.momentum.relTol = us.relTol;
        C.momentum.maxIter = us.maxIter;
        C.momentum.smoothSolver = us.gaussSeidel();
        C.momentum.symmetric = (us.smoother == "symGaussSeidel");
        C.momentum.nSweeps = us.nSweeps;
    };
    setMomentumSolve(true);
    C.takeUAtBoundary = &dTakeU;
    if (deviceClosure)
    {
        C.nutCell = &dTurb.nut;
        C.nutBnd = &dTurb.nutBnd;
    }
    { const SolutionDirections sd = solutionDirections(fvp);
      for (int k = 0; k < 3; ++k) C.solutionD[k] = sd.d[k]; }

    DevicePhaseProperties props{f.mixture.phases.rho1, f.mixture.phases.nu1,
                                f.mixture.phases.rho2, f.mixture.phases.nu2};

    // ---- the state on the device -----------------------------------------------------------------
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    { ux[c] = f.U.internal[c].x; uy[c] = f.U.internal[c].y; uz[c] = f.U.internal[c].z; }

    DeviceBuffer<scalar> dA(f.alpha1.internal), dAOld(f.alpha1.internal);
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dUox(ux), dUoy(uy), dUoz(uz);
    // U.oldTime()'s PATCH values, snapshotted with the cells below: ddtCorr's boundary half
    // interpolates the STORED patch value on an uncoupled patch, not the face cell
    DeviceBuffer<scalar> dUobx, dUoby, dUobz;
    DeviceBuffer<scalar> dPhiI(f.phi.internal);
    DeviceBuffer<scalar> dPrgh(f.p_rgh.internal), dP;
    DeviceBuffer<scalar> dNH(f.nHatf.internal), dNHB(flattenPatches(f.nHatf.boundary, fvp));
    // ...and ON THE PAIR, seeded from the same field buildInterFields built: at the first step of a
    // run that normal is calculateK's own.
    if (dCyc.n > 0)
    {
        dNHIf.copyFrom(coupledFace(f.nHatf, cyclics));
    }
    DeviceBuffer<scalar> dABnd(patchValues(f.alpha1, fvp)), dK(f.K);
    // |Sf| over the full face array IN THE DEVICE'S ORDER -- internal faces, then the boundary patches
    // with the coupled ones LEFT OUT, as every array the pressure step indexes it beside is laid out
    // (fullFace). It was uploaded in MESH order, which is the same array only while no coupled patch
    // precedes another patch: on damBreakLeakage the two symmetry blocks follow the pair and read each
    // other's neighbours' areas. NOT DISCRIMINATED by any gate -- phig is zero on every patch that
    // could see it there (snGrad(rho) and stf both vanish on a symmetry or a wall) -- so this is a
    // correction by construction and is claimed as nothing more.
    auto magSfAll = [&]()
    {
        SurfaceScalarField a;
        const label nIfA = m.nInternalFaces();
        a.internal.assign(g.magSf().begin(), g.magSf().begin() + nIfA);
        a.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            a.boundary[pi].assign(g.magSf().begin() + fvp[pi].start,
                                  g.magSf().begin() + fvp[pi].start + fvp[pi].size);
        }
        return fullFace(a, fvp);
    };
    DeviceBuffer<scalar> dGh(f.gh), dGhf, dMagSf(magSfAll());
    { SurfaceScalarField gf; gf.internal = f.ghfInternal; gf.boundary = f.ghfBoundary;
      dGhf.copyFrom(fullFace(gf, fvp));
      // ghf on the pair: gravity dotted with the face centre, which does not change with the solution
      if (!cyclics.empty()) dGhfIf.copyFrom(coupledFace(gf, cyclics)); }
    DeviceBuffer<scalar> dRho, dMu, dNu;
    DeviceVectorBoundary dbU = buildDeviceVectorBoundary(f.U, fvp, g);
    // the io switch on the state the run starts from, as rhoSimpleFoam's device arm does right after its
    // own build (rhoSimpleFoam.cu:391): anything that reads dbU before the first momentum assembly --
    // correctPhi's constrainHbyA, the alpha step's gradients -- would otherwise see an inletOutlet as a
    // fixedValue wall at its inletValue.
    deviceUpdateInletOutlet(dbU, dPhiB);

    // THE MESH UPDATE'S OWN OBJECTS, as the host driver keeps them (inter_driver_cpp.cu): the mesh's
    // GAMG hierarchy, which the motion solve builds and the run keeps, and the case's CorrectPhi
    // controls, which interMeshUpdate uses when `correctPhi` is on.
    CorrectPhiControls meshCpc;
    meshCpc.pcorr = &f.pcorrSolve;
    meshCpc.pcorrFinal = &f.pcorrSolveFinal;
    meshCpc.gamgCache = &meshAgglomeration;
    meshCpc.correctedLaplacian = f.laplacianScheme.corrected;
    meshCpc.snGradLimitCoeff = f.laplacianScheme.limitCoeff;
    meshCpc.gradPcorr = f.gradPcorr;
    meshCpc.nNonOrthogonalCorrectors = f.nNonOrthogonalCorrectors;
    // the volumes the mesh had before this step's move; empty on a static mesh, where the ddt's
    // other branch runs
    DeviceBuffer<scalar> dV0;
    // ...and the mesh flux the move produced, which the pressure corrector makes phi relative to
    DeviceBuffer<scalar> dMeshPhi;
    // Uf.oldTime() and the face flux ddtCorr reads off it, (Sf & Uf.oldTime()) per internal face.
    // The host driver keeps the same pair (UfOld, dc.UfOld); on a moving mesh OpenFOAM's ddtCorr
    // takes this in phi.oldTime()'s place (EulerDdtScheme's fvcDdtUfCorr).
    SurfaceVectorField UfOld = f.Uf;
    DeviceBuffer<scalar> dPhiUfOld;
    // ...and the ABSOLUTE flux the pressure step leaves for fvc::correctUf below, which reads phi
    // one line before makeRelative turns it relative (pEqn.H:66 and :69).
    DeviceBuffer<scalar> dPhiAbsI, dPhiAbsB;
    SurfaceScalarField phiAbs;
    // ...and rAU, which the NEXT step's mesh update solves CorrectPhi with (interFoam.C:138,
    // correctPhi.H) on a case that asks for it
    DeviceBuffer<scalar> dRAU;

    // THE cyclicACMI RESCALE, through the alpha step's geometryUpdate -- after phic, before the
    // pre-solve, ONCE PER TIME STEP -- with the same host call the host loop makes and at the same
    // clock: `stepTime` is rep.time + rep.deltaT, the ACCUMULATED sum the host hands rescale(), and
    // the baffle opens on the step where that sum first passes the scale's threshold (500 additions of
    // 1e-3 are 0.50000000000000033, not 0.5). What follows it re-uploads what the device holds of the
    // geometry that moved, IN PLACE:
    //   the mesh's areas, volumes and centres        refreshDeviceMeshGeometry, from the host's g --
    //                                                which keeps its cached weights and deltaCoeffs
    //                                                across a rescale, as OpenFOAM's static mesh does
    //   the pair's own areas                         refreshDeviceCyclicAreas, and ONLY the areas
    //   |Sf| for phig                                dMagSf
    // dbU is rebuilt by the momentum's updateUBoundary before anything reads it, and the grad(U) memo
    // carries the boundary areas in its fingerprint for exactly this step (device_kepsilon.cu).
    bool acmiRescaledThisStep = false;
    if (acmi && acmi->scaled())
    {
        // the closure's boundary arrays are built once and keep the OLD areas on the non-overlap
        // patches. That is inert while k, epsilon and nut are zero-gradient there -- a coefficient of
        // zero times an area -- which is what a symmetry patch gives them; anything else is refused
        if (deviceClosure)
        {
            for (const cpu::cyclicACMI::Side& side : acmi->sides())
            {
                const std::size_t no = static_cast<std::size_t>(side.nonOverlap);
                const bool flat = f.turbulence.k.boundary[no]->bcCategory() == 0
                               && f.turbulence.nut.boundary[no]->bcCategory() == 0;
                if (!flat)
                {
                    throw std::runtime_error(
                        "brae interFoam -device: the cyclicACMI's non-overlap patch `" + fvp[no].name +
                        "` carries a turbulence condition that is not zero-gradient, and the device "
                        "closure's boundary areas are built once -- they would keep the area the patch "
                        "had before the interface moved. Run without -device.");
                }
            }
        }
        H.alpha.geometryUpdate = [&]()
        {
            if (acmiRescaledThisStep)
            {
                return;
            }
            acmi->rescale(stepTime, m, *mutableMesh->g, *mutableMesh->patches);
            acmiRescaledThisStep = true;
            refreshDeviceMeshGeometry(dm, m, g, fvp);
            refreshDeviceCyclicAreas(dCyc, cyclics, g, fvp);
            dMagSf.copyFrom(magSfAll());
        };
    }

    RunReport rep;
    rep.pcorrSolves = initPcorrSolves;
    rep.deltaT = f.deltaT;
    rep.turbulenceOnDevice = deviceClosure;
    // THE CLOCK STARTS WHERE THE START DIRECTORY SAYS, as the host loop's does (startTimeOf). This one
    // started at 0 whatever the directory was named, which a run from 0 cannot see and a RESTART
    // cannot survive: every time-dependent input -- a cyclicACMI's scale, a wave, a table, the mesh
    // motion -- was evaluated `startTime` early. adjustDeltaT and the write cadence measure from the
    // start (Time.C:1150), so they take rep.time - startTime, exactly rep.time when the start is 0.
    const scalar startTime = startTimeOf(startDir);
    rep.time = startTime;
    for (label s = 0; s < nSteps; ++s)
    {
        if (!(rep.time < endTime - scalar(0.5)*rep.deltaT)) break;   // Time::run(), Time.C:1000

        // interFoam.C:98-100 -- CourantNo.H, alphaCourantNo.H, setDeltaT.H, and only THEN ++runTime.
        // Both numbers come off the flux the LAST step left behind and the step size still in force;
        // computing them after the step, or after deltaT moves, throttles on the wrong pair.
        //
        // OpenFOAM's two #includes each build their own surfaceSum(mag(phi)), and so do these two
        // calls. The host driver caches one and shares it, which is the single place it deliberately
        // differs from OpenFOAM; the device does not need to, so it does not.
        rep.CoNum = deviceAlphaCourantNo(dm, dPhiI, dPhiB, nullptr, rep.deltaT).CoNum;
        rep.alphaCoNum = deviceAlphaCourantNo(dm, dPhiI, dPhiB, &dA, rep.deltaT).CoNum;
        rep.deltaT = setDeltaTVoF(rep.deltaT, rep.CoNum, rep.alphaCoNum, f.timeCtl,
                                  rep.time - startTime, &f.writeCadence);

        // Uf.oldTime(), snapshotted where the host driver snapshots it (inter_driver_cpp.cu:897,
        // alongside UOld and phiOld). The FLUX off it is built after this step's move, below: the
        // field is the step's, the geometry it is dotted with is the mesh's as it stands then.
        if (dyn)
        {
            UfOld = f.Uf;
            // Uf is built for a dynamic mesh only (inter_case_cpp.cu, createUfIfPresent.H), and the
            // loop below reads it face by face -- so its size is checked rather than assumed.
            // Reading past it is what a missing Uf would do QUIETLY, and phiHbyA is six orders
            // larger than the field it feeds when that happens.
            if (UfOld.internal.size() != static_cast<std::size_t>(nIf))
            {
                throw std::runtime_error(
                    "brae interFoam -device: the case moves its mesh but Uf has " +
                    std::to_string(UfOld.internal.size()) + " internal faces, not " +
                    std::to_string(nIf) + ". ddtCorr reads (Sf & Uf.oldTime()) off it on a moving "
                    "mesh (EulerDdtScheme's fvcDdtUfCorr); refusing rather than reading past it.");
            }
        }

        std::vector<scalar> ca, cx, cy, cz;
        dA.copyTo(ca);   dAOld.copyFrom(ca);
        dUx.copyTo(cx);  dUy.copyTo(cy);  dUz.copyTo(cz);
        dUox.copyFrom(cx); dUoy.copyFrom(cy); dUoz.copyFrom(cz);
        {
            std::vector<scalar> bx, by, bz;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (const vector& v : f.U.boundary[pi]->value())
                {
                    bx.push_back(v.x); by.push_back(v.y); bz.push_back(v.z);
                }
            }
            dUobx.copyFrom(bx); dUoby.copyFrom(by); dUobz.copyFrom(bz);
        }
        std::vector<scalar> poi, pob;
        dPhiI.copyTo(poi); dPhiB.copyTo(pob);
        DeviceBuffer<scalar> dPhiOI(poi), dPhiOB(pob);
        // ...and the PAIR's flux at the same instant. fvc::ddtCorr compares phi.oldTime() with the
        // flux of U.oldTime() on every face a coupled patch included, and the pressure corrector
        // rewrites cyc.phi, so the snapshot has to be taken here with the other two.
        DeviceBuffer<scalar> dPhiOIf;
        if (dCyc.n > 0)
        {
            std::vector<scalar> poif;
            dCyc.phi.copyTo(poif);
            dPhiOIf.copyFrom(poif);
            C.phiOldIf = &dPhiOIf;
        }

        // OpenFOAM's clock for the step about to be taken: ++runTime comes before the alpha step
        stepDeltaT = rep.deltaT;
        stepTime = rep.time + rep.deltaT;
        acmiRescaledThisStep = false;   // once per TIME STEP, not per outer corrector
        stepIndex = s + 1;

        // rho.oldTime() for the closure's ddt: f.rho still holds what the LAST step's hook left, and
        // dRho what the last step wrote -- nothing yet at the first, where the host's is the field
        const std::vector<scalar> rhoOldHost = hostClosure ? f.rho : std::vector<scalar>();
        DeviceBuffer<scalar> dRhoOld;
        if (deviceClosure)
        {
            if (dRho.size() == static_cast<std::size_t>(nC))
            {
                deviceCopy(dRhoOld, dRho);
            }
            else
            {
                dRhoOld.copyFrom(f.rho);
            }
        }

        // THE PIMPLE OUTER LOOP, interFoam.C:107-176 and the host's runTimeStep
        // (inter_solve_cpp.cu:111-131). Every outer corrector re-solves alpha from the SAME
        // alpha.oldTime() with the latest flux, reassembles the momentum matrix and runs the pressure
        // correctors again; the old-time fields above are the TIME STEP's and are snapshotted once,
        // OUTSIDE this loop, which is what makes that true.
        for (label outer = 0; outer < f.pimple.nOuterCorrectors; ++outer)
        {
            const bool finalOuter = (outer == f.pimple.nOuterCorrectors - 1);
            setMomentumSolve(finalOuter);

            // interFoam.C:112-149, THE MESH UPDATE, through the same host stage the host loop calls
            // (interMeshUpdate). The motion solve, CorrectPhi and mixture.correct() are host
            // operations on either arm; what this loop owes afterwards is the geometry it had
            // uploaded. A moving mesh with an AMI or a coupled pair is refused above, so the
            // interfaces below it do not move.
            if (dyn)
            {
                // storeOldVol BEFORE the geometry is recomputed -- OF fvMesh::movePoints:944. The
                // ddt's old-time term belongs to the volume the old-time field was stored in.
                dV0.copyFrom(dm.V.host());
                C.V0 = &dV0;
                interMeshUpdate(dyn, f, m, g, fvp, mutableMesh, /*amiPairs=*/nullptr,
                                meshAgglomeration, meshCpc, rep, stepTime, stepIndex, outer,
                                f.pimple.nOuterCorrectors);
                // ...and now every buffer this loop uploaded from the geometry. clearGeom +
                // clearOut on the device side: the addressing is untouched, as the move keeps the
                // topology fixed (device_mesh.cuh).
                refreshDeviceMeshGeometry(dm, m, g, fvp);
                dMagSf.copyFrom(magSfAll());
                dGh.copyFrom(f.gh);
                {
                    SurfaceScalarField gf;
                    gf.internal = f.ghfInternal;
                    gf.boundary = f.ghfBoundary;
                    dGhf.copyFrom(fullFace(gf, fvp));
                }
                // the patch geometry moved with the cells, and dbU carries the patch deltas and
                // normals every boundary evaluation reads
                dbU = buildDeviceVectorBoundary(f.U, fvp, g);
                // CorrectPhi and makeRelative rewrote phi on the host -- but ONLY under `correctPhi`
                // (interFoam.C:130, and interMeshUpdate does nothing to phi without it). Taking the
                // host's phi unconditionally overwrote the flux THIS loop had just computed with a
                // stale copy at every outer corrector after the first, which is why carrying meshPhi
                // into the pressure step changed no digit until this guard went in.
                if (f.correctPhi)
                {
                    dPhiI.copyFrom(f.phi.internal);
                    dPhiB.copyFrom(flattenPatches(f.phi.boundary, fvp));
                    // ...and mixture.correct() ON THE MOVED MESH, which interFoam.C:141 runs inside
                    // this same `if (correctPhi)` and interMeshUpdate has just done on the host. The
                    // alpha equation below reads nHatf at the TOP of its first corrector, before any
                    // mixture.correct() of its own, so the device would otherwise convect with the
                    // interface normal of the mesh as it stood BEFORE the move. K is left to the
                    // alpha step, whose own mixture.correct() rewrites it before the pressure
                    // corrector reads it.
                    dNH.copyFrom(f.nHatf.internal);
                    dNHB.copyFrom(flattenPatches(f.nHatf.boundary, fvp));
                }
                // ...and THE MESH FLUX the move produced, which the pressure corrector makes phi
                // relative to (fvc::makeRelative, pEqn.H:73). Over the full face array, as phi is.
                dMeshPhi.copyFrom(fullFace(dyn->meshPhi(), fvp));
                C.meshPhiAll = &dMeshPhi;
                C.phiAbsIntOut = &dPhiAbsI;
                C.phiAbsBndOut = &dPhiAbsB;
                C.rAUOut       = &dRAU;
                // ...and (Sf & Uf.oldTime()), the flux ddtCorr takes in phi.oldTime()'s place, on
                // the mesh AS IT STANDS NOW. fvcDdtUfCorr dots the STORED old Uf with mesh().Sf()
                // (EulerDdtScheme.C:527-531), which the move above has just changed, so this cannot
                // be built before it. Built before it, the error is INVISIBLE in step one -- Uf
                // starts at zero and so does the flux, whatever the geometry -- and by step two it
                // is 13% of |U| on testTubeMixer (measured: |U| 4.8e-01 of 3.6e+00 against the host
                // arm, alpha still 6e-13; step one 1.6e-11).
                {
                    std::vector<scalar> pu(static_cast<std::size_t>(nIf));
                    for (label fc = 0; fc < nIf; ++fc)
                    {
                        pu[static_cast<std::size_t>(fc)] = dot(g.Sf()[fc], UfOld.internal[fc]);
                    }
                    dPhiUfOld.copyFrom(pu);
                    C.phiUfOldInt = &dPhiUfOld;
                }
            }

            deviceInterStep(dm, rep.deltaT, C, props, H, dGh, dGhf, dMagSf,
                            dA, dAOld, dUx, dUy, dUz, dUox, dUoy, dUoz,
                            dUobx, dUoby, dUobz,
                            dPhiI, dPhiB, dPhiOI, dPhiOB, dUFixes, dPrgh, dP,
                            dNH, dNHB, dABnd, dK, dAFixes, dAFlag, dbU,
                            dRho, dMu, dNu, dRpI, dRpB, tapsOut);

            // turbulence->correct(), interFoam.C:169-172 -- after this outer corrector's last
            // pressure corrector. dbU is current: the step's last updateUBoundary rebuilt it and the
            // pressureInletOutletVelocity switch ran on it after that.
            //
            // pimple.turbCorr(): with turbOnFinalIterOnly -- OpenFOAM's default -- the closure
            // advances ONCE per time step, on the final outer corrector. Running it on every one
            // would advance k and epsilon nOuterCorrectors times per physical step.
            if (!f.pimple.turbOnFinalIterOnly || finalOuter)
            {
            if (deviceClosure)
            {
                DeviceInterTurbulenceStepInput ti;
                ti.Ux = &dUx;
                ti.Uy = &dUy;
                ti.Uz = &dUz;
                ti.phiInt = &dPhiI;
                ti.phiBnd = &dPhiB;
                ti.rhoPhiInt = &dRpI;
                ti.rhoPhiBnd = &dRpB;
                ti.rho = &dRho;
                ti.rhoBnd = &dStepRhoBnd;
                ti.cyc       = (dCyc.n > 0) ? &dCyc : nullptr;
                ti.cycPhi    = (dCyc.n > 0) ? &dCyc.phi : nullptr;
                ti.cycRhoPhi = (dCyc.n > 0) ? &dRhoPhiIf : nullptr;
                ti.rhoOld = &dRhoOld;
                ti.nu = &dStepNu;
                ti.nuBnd = &dStepNuBnd;
                ti.deltaT = rep.deltaT;
                // the second field's log is omega's under kOmegaSST, as the host driver keeps it
                ti.epsilonLog = (f.turbulence.model == cpu::interFoam::InterRasModel::KOmegaSST)
                              ? &rep.omegaSolves : &rep.epsilonSolves;
                ti.kLog = &rep.kSolves;
                deviceCorrectInterTurbulence(dTurb, f.turbulence, ti, dm, dbU);
            }
            // ...or THE HOST CLOSURE in the same loop. f.U is current (the step's last updateUBoundary
            // wrote it, patches included) and so are f.rho, f.nu and f.nuBnd (the hooks'); phi's interior
            // and rhoPhi are the device's only.
            if (hostClosure)
            {
                dPhiI.copyTo(f.phi.internal);
                pushFlux();
                dRpI.copyTo(f.rhoPhi.internal);
                {
                    std::vector<scalar> rb;
                    dRpB.copyTo(rb);
                    f.rhoPhi.boundary.resize(fvp.size());
                    std::size_t off = 0;
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        if (isCoupledInterfaceType(fvp[pi].type)) continue;
                        const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
                        f.rhoPhi.boundary[pi].assign(rb.begin() + off, rb.begin() + off + n);
                        off += n;
                    }
                    // ...and the PAIR's own faces, which that array does not carry. The host
                    // reference fills rhoPhi on every patch, coupled included
                    // (alpha_eqn_cpp.cu:389-394), so leaving them here is a value from before the
                    // step -- and this block is a GATE'S INSTRUMENT, whose whole job is to say
                    // whether a disagreement is the closure's or the loop's.
                    // NOT DISCRIMINATED by any gate in the tree: the closure reads rhoPhi only under
                    // `density variable` (inter_turbulence_cpp.cu:531) and no coupled fixture sets
                    // it, so the baffle gate's five profiles read identically with and without this
                    // block, to every digit. It is here because the reference's contract says so.
                    if (dCyc.n > 0 && !cyclics.empty())
                    {
                        std::vector<scalar> rif;
                        dRhoPhiIf.copyTo(rif);
                        std::size_t o = 0;
                        for (const CyclicInterface& c : cyclics)
                        {
                            const std::size_t pi = static_cast<std::size_t>(c.patch);
                            const std::size_t n = c.faceCells.size();
                            if (pi < f.rhoPhi.boundary.size() && o + n <= rif.size())
                            {
                                f.rhoPhi.boundary[pi].assign(rif.begin() + o, rif.begin() + o + n);
                            }
                            o += n;
                        }
                    }
                }
                InterTurbulenceStepInput ti;
                ti.U = &f.U;
                ti.phi = &f.phi;
                ti.rhoPhi = &f.rhoPhi;
                ti.rho = &f.rho;
                ti.rhoBnd = &stepRhoBnd;
                ti.rhoOld = &rhoOldHost;
                ti.nu = &f.nu;
                ti.nuBnd = &f.nuBnd;
                ti.deltaT = rep.deltaT;
                // the host closure keeps the two second fields in separate logs and fills the model's own
                ti.omegaLog = &rep.omegaSolves;
                ti.epsilonLog = &rep.epsilonSolves;
                ti.kLog = &rep.kSolves;
                correctInterTurbulence(f.turbulence, ti, m, g, fvp);
            }
            }   // pimple.turbCorr()

            // fvc::correctUf(Uf, U, phi), pEqn.H:70-72 -- the END of the pressure corrector on a
            // moving mesh. Uf is a HOST field (next step's ddtCorr reads (Sf & Uf.oldTime()) off it),
            // so the device's U and phi come back for it. One copy of the arithmetic, shared with the
            // host loop (correctUf, inter_peqn_cpp.cu).
            if (dyn)
            {
                { std::vector<scalar> ux, uy, uz;
                  dUx.copyTo(ux); dUy.copyTo(uy); dUz.copyTo(uz);
                  for (label c = 0; c < nC; ++c)
                      f.U.internal[c] = vector{ux[c], uy[c], uz[c]};
                  f.U.evaluateBoundary(); }
                dPhiI.copyTo(f.phi.internal);
                pushFlux();
                // ...with the flux as it stood BEFORE makeRelative, which is the one pEqn.H:66 hands
                // fvc::correctUf. Handing it the relative flux instead puts the MESH's normal
                // velocity into Uf, and nothing in the step that wrote it can see the error: Uf is
                // read only by the NEXT step's ddtCorr. MEASURED on testTubeMixer against the host
                // arm, one step: |Uf| 1.5004e+00 of 3.3328e+00 while U was 1.6e-11 and alpha 5.6e-15;
                // by step two that Uf was |U| 4.8e-01 of 3.6e+00.
                phiAbs.boundary = f.phi.boundary;
                dPhiAbsI.copyTo(phiAbs.internal);
                unflatten(dPhiAbsB, phiAbs.boundary);
                correctUf(f.Uf, f.U, phiAbs, m, g, fvp);
                // ...and rAU, for the same reason and with the same blindness: the NEXT mesh update
                // interpolates it for CorrectPhi's laplacian, and nothing THIS step does reads it.
                // Left at the value buildInterFields wrote, step one is exact (both arms start there)
                // and step two is 2.3e-02 of |U| on waveMakerSolitary.
                if (dRAU.size()) dRAU.copyTo(f.rAU);
            }
        }   // the outer corrector loop

        rep.steps = s + 1;
        rep.time += rep.deltaT;
        f.writeCadence.advance(rep.time - startTime, rep.deltaT);   // Time::operator++, Time.C:1046-1074

        if (verbose)
        {
            std::vector<scalar> av;
            dA.copyTo(av);
            scalar lo = av.empty() ? 0 : av[0], hi = lo;
            for (scalar v : av) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
            std::printf("   t = %.17g  dt = %.17g  Co %.3f  alphaCo %.3f  [device]  "
                        "alpha [%.3e, %.15g]\n",
                        (double)rep.time, (double)rep.deltaT, (double)rep.CoNum,
                        (double)rep.alphaCoNum, (double)lo, (double)hi);
        }
    }

    if (meshAgglomeration.built)
    {
        const GamgAgglomeration& a = meshAgglomeration.agglomeration;
        for (label leveli = 0; leveli <= a.size(); ++leveli)
        {
            const GamgLduAddressing& addr = a.meshLevel(leveli);
            RunReport::GamgLevel lv;
            lv.nCells = addr.nCells;
            lv.nFaces = static_cast<label>(addr.upperAddr.size());
            lv.profile = gamgLduBand(addr).second;
            rep.gamgLevels.push_back(lv);
        }
        for (const SolverPerformance& sp : gamgLog.coarsest)
        {
            rep.gamgCoarsestSolves.push_back(LinearSolveRecord{sp.initialResidual, sp.finalResidual, sp.nIterations});
        }
    }
    for (const DeviceSolverPerf& sp : pLog)
    {
        rep.pSolves.push_back(PressureSolveRecord{sp.initialResidual, sp.finalResidual, sp.nIterations});
    }
    for (const DeviceSolverPerf& sp : aLog)
    {
        rep.alphaSolves.push_back(LinearSolveRecord{sp.initialResidual, sp.finalResidual, sp.nIterations});
    }
    for (int k = 0; k < 3; ++k)
    {
        for (const DeviceSolverPerf& sp : uLog[k])
        {
            rep.uSolves[k].push_back(LinearSolveRecord{sp.initialResidual, sp.finalResidual, sp.nIterations});
        }
    }

    // hand the device's answer back through the host fields, so a caller compares the same objects
    dA.copyTo(f.alpha1.internal);
    f.alpha1.evaluateBoundary();
    { std::vector<scalar> x, y, z;
      dUx.copyTo(x); dUy.copyTo(y); dUz.copyTo(z);
      for (label c = 0; c < nC; ++c) f.U.internal[c] = vector{x[c], y[c], z[c]};
      f.U.evaluateBoundary(); }
    dPrgh.copyTo(f.p_rgh.internal);
    f.p_rgh.evaluateBoundary();
    dP.copyTo(f.p);
    dRho.copyTo(f.rho);
    if (deviceClosure)
    {
        downloadDeviceInterTurbulence(dTurb, f.turbulence, fvp);
    }
    dPhiI.copyTo(f.phi.internal);

    // ...and the REPORT's own numbers. Leaving maxU and worstDivPhi at their defaults printed
    // "max|U| 0 m/s, worst |div(phi)| 0.000e+00" on a run whose U reaches 0.27 -- a false statement in
    // the solver's own output, and the kind a reader trusts because the rest of the line is right.
    rep.maxU = 0;
    for (label c = 0; c < nC; ++c)
        rep.maxU = std::fmax(rep.maxU, std::sqrt(f.U.internal[c].x*f.U.internal[c].x
                                               + f.U.internal[c].y*f.U.internal[c].y
                                               + f.U.internal[c].z*f.U.internal[c].z));
    {
        // dPhiB carries ONLY the non-coupled patches (device_mesh.cuh:41-44), and the pair's own faces
        // come from the interface array. Walking fvp whole here shifted every patch after the first
        // coupled one: MEASURED on damBreakPorousBaffle, this report read worst |div(phi)| 2.875e-01
        // against the host's 7.932e-05 on a run whose max|U| agreed to every digit.
        pushFlux();
        SurfaceScalarField phiOut;
        phiOut.internal = f.phi.internal;
        phiOut.boundary = f.phi.boundary;
        const std::vector<scalar> d = fvc::div(phiOut, m, g, fvp);
        rep.worstDivPhi = 0;
        for (label c = 0; c < nC; ++c) rep.worstDivPhi = std::fmax(rep.worstDivPhi, std::fabs(d[c]));
    }

    std::vector<scalar> av;
    dA.copyTo(av);
    rep.alphaMin = av.empty() ? 0 : av[0];
    rep.alphaMax = rep.alphaMin;
    rep.alphaMass = 0;
    for (label c = 0; c < nC; ++c)
    {
        rep.alphaMin = std::fmin(rep.alphaMin, av[c]);
        rep.alphaMax = std::fmax(rep.alphaMax, av[c]);
        rep.alphaMass += av[c]*g.V()[c];
    }
    (void)nFaces;
    (void)nBf;
    if (fieldsOut) *fieldsOut = std::move(f);
    return rep;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
