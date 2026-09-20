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
    DeviceInterStepTaps* tapsOut)
{
    InterFields f = buildInterFields(caseDir, startDir, m, g, fvp);
    // THE PAIR, built here and not at the device-mesh stage, because the hooks below fill its share of
    // the surface fields and they are defined before the DeviceCyclic is.
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
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
    if (f.turbulence.on && f.turbulence.model == cpu::interFoam::InterRasModel::KEqnLES)
        throw std::runtime_error(
            "brae interFoam (device): the case is LES kEqn. The device loop runs kEpsilon's device twin "
            "and nothing else; kEqn and its filter width are ported on the host (les_kEqn_cpp.cu, "
            "les_delta_cpp.cu) and gated there against OpenFOAM on LES/nozzleFlow2D.");

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
    // fraction on its patch, as the host driver rebuilds it (inter_driver_cpp.cu:722-735). What is still
    // refused here is the permeable-wall pair, which reads the phase field too but at another point.
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (f.U.boundary[pi]->needsAlphaPatchValues() || f.p_rgh.boundary[pi]->needsAlphaPatchValues()
         || f.p_rgh.boundary[pi]->isPrghPermeableAlphaTotalPressure())
            throw std::runtime_error(
                "brae interFoam (device): patch `" + fvp[pi].name + "` carries a permeable-wall condition "
                "(permeableAlphaPressureInletOutletVelocity or prghPermeableAlphaTotalPressure). Each "
                "switches face by face between a wall and an open boundary on the phase fraction at the "
                "patch, at every update; the host loop carries that (gated on laminar/damBreakPermeable) "
                "and the device loop uploads the blend once. Refused rather than run a wall that never opens.");
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
    if (f.dynamicMesh)
    {
        throw std::runtime_error(
            "brae interFoam -device: the case moves its mesh (" + f.dynamicMesh->motionType() + "). The "
            "device loop does not move one; the host loop does. Run without -device.");
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

    // initCorrectPhi.H, on the host and before anything is uploaded -- see runInterFoam. A GAMG pcorr
    // SOLVER builds the mesh's hierarchy even at rest (the GAMGSolver constructor builds it before the
    // first residual), and OpenFOAM's p_rgh GAMG then reuses that one; the device builds its own from
    // p_rgh's entry, which is the same hierarchy only when the two ask for the same coarsest level.
    std::vector<LinearSolveRecord> initPcorrSolves;
    {
        GamgAgglomerationCache initCache;
        CorrectPhiControls cpc;
        cpc.pcorr = &f.pcorrSolve;
        cpc.pcorrFinal = &f.pcorrSolveFinal;
        cpc.gamgCache = &initCache;
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
        for (const InterFields::PressureLinearSolve* ps : {&f.pSolve, &f.pSolveFinal})
        {
            if (initCache.built && ps->gamgSolver()
                && ps->gamg.nCellsInCoarsestLevel != f.pcorrSolveFinal.gamg.nCellsInCoarsestLevel)
            {
                throw std::runtime_error(
                    "brae interFoam -device: pcorrFinal is GAMG with nCellsInCoarsestLevel " +
                    std::to_string(f.pcorrSolveFinal.gamg.nCellsInCoarsestLevel) + " and p_rgh with " +
                    std::to_string(ps->gamg.nCellsInCoarsestLevel) + ". OpenFOAM's p_rgh reuses the "
                    "hierarchy pcorr built at the start; the device builds its own from p_rgh's entry. "
                    "Run without -device.");
            }
        }
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
    auto pushFlux = [&]()
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
        pushFluxToPatches(f, fvp);
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
            const DeviceBuffer<scalar>& uz, DeviceVectorBoundary& db, DeviceBuffer<scalar>* ubOut)
    {
        pushFlux();
        std::vector<scalar> x, y, z;
        ux.copyTo(x); uy.copyTo(y); uz.copyTo(z);
        for (label c = 0; c < nC; ++c) f.U.internal[c] = vector{x[c], y[c], z[c]};
        // waveVelocity's updateCoeffs, at the STEP's clock: the first call of a step is UEqn's, where
        // the model -- last updated inside the alpha sub-cycles -- updates again from the alpha they
        // left (f.alpha1 is the alpha hooks' last copy) and the U the step started on. Every later
        // call of the step is the same time index and re-assigns the same values.
        updateWaveVelocity(f.waves, f.alpha1, f.U, stepTime, stepIndex, m, g, fvp);
        f.U.evaluateBoundary();
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
    // `Gauss interfaceCompression` runs on the host (limitedSchemes_cpp.cuh). The device has no such
    // scheme, and the two mappings below would take it as vanLeer and as linear without a word.
    if (f.divPhiAlpha == AlphaFluxScheme::interfaceCompression
        || f.divPhirbAlpha == AlphaFluxScheme::interfaceCompression)
    {
        throw std::runtime_error(
            "brae interFoam -device: the case names `Gauss interfaceCompression` for an alpha flux. The "
            "device alpha step has linear, upwind and vanLeer; the host loop has interfaceCompression. "
            "Run without -device.");
    }
    C.alphaInput.alphaScheme  = (f.divPhiAlpha == AlphaFluxScheme::linear) ? DeviceAlphaScheme::linear
                              : (f.divPhiAlpha == AlphaFluxScheme::upwind) ? DeviceAlphaScheme::upwind
                              : DeviceAlphaScheme::vanLeer;
    C.alphaInput.alpharScheme = (f.divPhirbAlpha == AlphaFluxScheme::vanLeer) ? DeviceAlphaScheme::vanLeer
                              : (f.divPhirbAlpha == AlphaFluxScheme::upwind) ? DeviceAlphaScheme::upwind
                              : DeviceAlphaScheme::linear;
    // EVERY SCHEME NAMED, NO default: this switch used to end in `default: upwind`, and interFoam's
    // `linear` -- which its own enum carries and three shipped tutorials name -- fell through it, so
    // a -device run of such a case would have convected upwind under the name `linear`.
    switch (f.divRhoPhiU)
    {
        case DivScheme::linearUpwind:   C.divScheme = brae::cpu::DivScheme::linearUpwind;   break;
        case DivScheme::linearUpwindV:  C.divScheme = brae::cpu::DivScheme::linearUpwindV;  break;
        case DivScheme::limitedLinear:
            // REFUSED, with the measurement. The device momentum's limitedLinear branch (shared with
            // simpleFoam, UEqn.cu) runs laminar/vofToLagrangian/eulerianInjection -- with its GAMG
            // smoother switched to DIC so the device accepts it -- to a U 5.1e-01 away from OpenFOAM
            // after 50 steps, where the same case under `Gauss upwind` agrees to 1.1e-13 and the host's
            // limitedLinear to 3.9e-12. The defect is in that branch; it is not ported here yet.
            throw std::runtime_error(
                "brae interFoam -device: `div(rhoPhi,U) Gauss limitedLinear` runs the device momentum's "
                "limitedLinear branch, which is 51% off OpenFOAM's U on eulerianInjection where the host "
                "loop agrees to 4e-12. Run without -device.");
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
        // the GAMG PRECONDITIONER is the host's only: the reader no longer declares it, so the
        // device does, and says what it runs instead
        if (entry->pcgGamg())
        {
            noticeApproximated("interFoam p_rgh solve on the device",
                "the case asks for `solver PCG; preconditioner { preconditioner GAMG; ... }` and the "
                "device loop runs Jacobi-BiCGStab at the same tolerance; the GAMG preconditioner is "
                "ported on the host loop only (pcgGamgSolve). The difference is where the solve stops.");
        }
        if (entry->gamgSolver() && !deviceGamgSmootherPorted(entry->gamg.smoother))
        {
            throw std::runtime_error(
                "brae interFoam -device: fvSolution's GAMG entry for p_rgh asks for `smoother "
                + entry->gamg.smoother + "`. The device's GAMG has the DIC smoother only "
                "(device_gamg_solver.cuh); the host loop runs DIC, DICGaussSeidel, GaussSeidel and "
                "symGaussSeidel. Refused rather than smoothed with something the case did not name.");
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
    GamgSolveLog gamgLog;
    C.pressureGamg = f.pSolve.gamgSolver() ? &f.pSolve.gamg : nullptr;
    C.pressureFinalGamg = f.pSolveFinal.gamgSolver() ? &f.pSolveFinal.gamg : nullptr;
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
    DeviceBuffer<scalar> dGh(f.gh), dGhf, dMagSf(g.magSf());
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

    RunReport rep;
    rep.pcorrSolves = initPcorrSolves;
    rep.deltaT = f.deltaT;
    rep.turbulenceOnDevice = deviceClosure;
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
                                  rep.time, &f.writeCadence);

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
        }   // the outer corrector loop

        rep.steps = s + 1;
        rep.time += rep.deltaT;
        f.writeCadence.advance(rep.time, rep.deltaT);   // Time::operator++, Time.C:1046-1074

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

    if (gamgCache.host.built)
    {
        const GamgAgglomeration& a = gamgCache.host.agglomeration;
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
