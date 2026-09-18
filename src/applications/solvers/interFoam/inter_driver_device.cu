// runInterFoamDevice -- the same interFoam time loop, on the GPU.
//
// See inter_driver_cpp.cuh for why the hooks live here rather than in the gate. Everything this calls
// is separately landed and separately gated; what this file owns is the wiring, and that wiring is
// what tests/test_device_inter_dambreak_alpha.cu measures against the host driver on damBreak.
#include "inter_driver_cpp.cuh"
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

std::vector<scalar> flattenPatches(const std::vector<std::vector<scalar>>& b)
{
    std::vector<scalar> v;
    for (const auto& p : b) v.insert(v.end(), p.begin(), p.end());
    return v;
}

std::vector<scalar> patchValues(const GeometricField<scalar>& f, const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> v;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& b = f.boundary[pi]->value();
        v.insert(v.end(), b.begin(), b.end());
    }
    return v;
}

std::vector<scalar> fullFace(const SurfaceScalarField& f, const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> v(f.internal);
    for (std::size_t pi = 0; pi < fvp.size() && pi < f.boundary.size(); ++pi)
        v.insert(v.end(), f.boundary[pi].begin(), f.boundary[pi].end());
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
    scalar endTime)
{
    InterFields f = buildInterFields(caseDir, startDir, m, g, fvp);

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
    if ((f.laplacianScheme.corrected || f.snGradScheme.corrected) && maxNonOrthogonality(m, g) >= scalar(1e-10))
    {
        throw std::runtime_error(
            "brae interFoam -device: fvSchemes asks for a non-orthogonal correction (laplacianSchemes `"
            + f.laplacianScheme.raw + "`, snGradSchemes `" + f.snGradScheme.raw + "`) and the mesh is not "
            "orthogonal. The device loop assembles the pressure laplacian and its snGrads orthogonal; the "
            "host loop takes the case's scheme. Run without -device.");
    }
    if (f.pRef.needReference)
    {
        throw std::runtime_error(
            "brae interFoam -device: p_rgh fixes its value on no patch and needs a reference cell. The "
            "device pressure step's reference is not OpenFOAM's (pEqn.H:47 pins the cell at its current "
            "p_rgh and :74-83 shifts p's level); the host loop's is. Run without -device.");
    }
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const label nFaces = static_cast<label>(g.magSf().size());
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    // TWO PIMPLE CONTROLS THE HOST HONOURS AND THIS LOOP DOES NOT, refused rather than run as 1 and 0.
    // deviceInterStep is one outer corrector with no non-orthogonal pass; 5 shipped tutorials ask for
    // nOuterCorrectors 2 or 3 and 4 for nNonOrthogonalCorrectors 1, and until this was here `-device`
    // would have taken every one of them at the smaller number without a word.
    if (f.pimple.nOuterCorrectors > 1)
    {
        throw std::runtime_error(
            "brae interFoam (device): `nOuterCorrectors " + std::to_string(f.pimple.nOuterCorrectors)
            + "` is not wired into the device loop, which runs one. The host path (no -device) does.");
    }
    if (f.nNonOrthogonalCorrectors > 0)
    {
        throw std::runtime_error(
            "brae interFoam (device): `nNonOrthogonalCorrectors "
            + std::to_string(f.nNonOrthogonalCorrectors) + "` is not wired into the device pressure "
              "step, which runs none. The host path (no -device) does.");
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

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // the masks the device needs that the mesh does not carry
    std::vector<int> aFixes, aFlag, takeU, uFixes;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
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

    // THE BOUNDARY FLUX, declared above the hooks because they read it. OpenFOAM's flux-conditional
    // patches look phi up whenever they update; brae's are told, and the device path holds the only
    // current copy. See pushFluxToPatches for what not telling them cost on capillaryRise.
    DeviceBuffer<scalar> dPhiB(flattenPatches(f.phi.boundary));
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
        out.assign(fvp.size(), std::vector<scalar>());
        std::size_t off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
            out[pi].assign(flat.begin() + off, flat.begin() + off + n);
            off += n;
        }
    };
    auto pushFlux = [&]()
    {
        unflatten(dPhiB, f.phi.boundary);
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
        SurfaceScalarField nHb;
        std::vector<scalar> Kb;
        interfaceProps::calculateK(f.alpha1, f.interface, m, g, fvp, false, nHb, Kb);
        nBnd.copyFrom(flattenPatches(nHb.boundary));
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
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(M.internalCoeffs[pi][i]); b2.push_back(M.boundaryCoeffs[pi][i]); }
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
        db = buildDeviceVectorBoundary(f.U, fvp, g);
        if (!ubOut) return;
        std::vector<scalar> bx, by, bz;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (const vector& u : f.U.boundary[pi]->value())
            { bx.push_back(u.x); by.push_back(u.y); bz.push_back(u.z); }
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
        const SurfaceScalarField snA = fvc::snGrad(f.alpha1, m, g, fvp, false);
        SurfaceScalarField t;
        t.internal.resize(static_cast<std::size_t>(nIf));
        for (label i = 0; i < nIf; ++i) t.internal[i] = sKf.internal[i]*snA.internal[i];
        t.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                t.boundary[pi].push_back(sKf.boundary[pi][i]*snA.boundary[pi][i]);
        stf.copyFrom(fullFace(t, fvp));

        // rho's CALCULATED patch values, from the device's own blend (which carries alpha2's
        // one-pass-older patch values) -- not a zeroGradient copy. See rhoWithPatchValues.
        std::vector<scalar> rbFlat;
        rhoBd.copyTo(rbFlat);
        std::vector<std::vector<scalar>> rb(fvp.size());
        {
            std::size_t off = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
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
        snRho.copyFrom(fullFace(fvc::snGrad(rhoF, m, g, fvp, false), fvp));

        // the mixture's own nu. NOTE mixtureNu's second argument is mu, not alpha2.
        cpu::twoPhase::mixtureMu(f.alpha1.internal, f.mixture.phases, f.mu);
        cpu::twoPhase::mixtureNu(f.alpha1.internal, f.mu, f.mixture.phases, f.nu);
        // f.nuBnd is the alpha hook's, taken BEFORE its curvature pass. Rebuilding it here would read
        // alpha's patch values one contact-angle pass too late.
        // nuEff = nut + nu: the mixture's nu alone on a laminar case. nut is the closure's, as the
        // LAST step's correct() left it -- or validate()'s, or the case file's, at the first.
        if (deviceClosure)
        {
            // the mixture's nu alone: the step adds the device's nut (DeviceInterStepControls::nutCell)
            nuC.copyFrom(f.nu);
            nuB.copyFrom(flattenPatches(f.nuBnd));
            deviceCopy(dStepNu, nuC);
            deviceCopy(dStepNuBnd, nuB);
        }
        else
        {
            std::vector<scalar> nuEff;
            std::vector<std::vector<scalar>> nuEffB;
            interNuEff(f.turbulence, f.nu, f.nuBnd, nuEff, nuEffB);
            nuC.copyFrom(nuEff);
            nuB.copyFrom(flattenPatches(nuEffB));
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
            snP.copyFrom(fullFace(fvc::snGrad(f.p_rgh, m, g, fvp, false), fvp));
        }
        else snP.resize(0);
    };
    H.pressure.pressureCoeffs =
        [&](const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>& phiHB,
            const DeviceBuffer<scalar>& rAUfAll, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        std::vector<scalar> hB, rA;
        phiHB.copyTo(hB);
        rAUfAll.copyTo(rA);
        // constrainPressure: a fixedFluxPressure gradient is PRESCRIBED from phiHbyA, and brae refuses
        // to assemble one that has not been set. rAUf is taken PER FACE from the full array.
        label off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const FvPatch& q = fvp[pi];
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
            for (label i = 0; i < fvp[pi].size; ++i) rf.boundary[pi].push_back(rA[o++]); }
        // totalPressure's updateCoeffs, where the fvMatrix constructor runs it. f.U's patch values and
        // f.phi's are current (updateUBoundary ran after the last corrector and pushed the flux); a
        // totalPressure patch is never a contact-angle wall, so f.rhoBnd is exact on it.
        updatePressurePatchesFromVelocity(f.p_rgh, f.U, &f.rhoBnd, fvp);
        FvScalarMatrix pe = fvm::laplacian<scalar>(rf, f.p_rgh, m, g, fvp, false);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(pe.internalCoeffs[pi][i]); b2.push_back(pe.boundaryCoeffs[pi][i]); }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };
    H.pressure.updateBoundary = [&](const DeviceBuffer<scalar>& pr)
    {
        pr.copyTo(f.p_rgh.internal);
        f.p_rgh.evaluateBoundary();
    };

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
        case DivScheme::limitedLinear:  C.divScheme = brae::cpu::DivScheme::limitedLinear;  break;
        case DivScheme::limitedLinearV: C.divScheme = brae::cpu::DivScheme::limitedLinearV; break;
        case DivScheme::LUST:           C.divScheme = brae::cpu::DivScheme::LUST;           break;
        case DivScheme::vanLeerV:       C.divScheme = brae::cpu::DivScheme::vanLeerV;       break;
        case DivScheme::linear:         C.divScheme = brae::cpu::DivScheme::linear;         break;
        case DivScheme::upwind:         C.divScheme = brae::cpu::DivScheme::upwind;         break;
    }
    C.divSchemeCoeff = f.divRhoPhiUCoeff;
    C.nCorrectors = static_cast<int>(f.pimple.nCorrectors);
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
    // THE CASE'S OWN SOLVE FOR U, where a hardcoded 1e-12 on Jacobi-BiCGStab used to stand. The device
    // loop runs ONE outer corrector (more are refused above), and that one is the final one, so the
    // entry fvMatrix::solve() selects is UFinal.
    if (f.momentumPredictorOn)
    {
        C.momentum.tol = f.uSolveFinal.tol;
        C.momentum.relTol = f.uSolveFinal.relTol;
        C.momentum.maxIter = f.uSolveFinal.maxIter;
        C.momentum.smoothSolver = f.uSolveFinal.gaussSeidel();
        C.momentum.symmetric = (f.uSolveFinal.smoother == "symGaussSeidel");
        C.momentum.nSweeps = f.uSolveFinal.nSweeps;
    }
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
    DeviceBuffer<scalar> dPhiI(f.phi.internal);
    DeviceBuffer<scalar> dPrgh(f.p_rgh.internal), dP;
    DeviceBuffer<scalar> dNH(f.nHatf.internal), dNHB(flattenPatches(f.nHatf.boundary));
    DeviceBuffer<scalar> dABnd(patchValues(f.alpha1, fvp)), dK(f.K);
    DeviceBuffer<scalar> dGh(f.gh), dGhf, dMagSf(g.magSf());
    { SurfaceScalarField gf; gf.internal = f.ghfInternal; gf.boundary = f.ghfBoundary;
      dGhf.copyFrom(fullFace(gf, fvp)); }
    DeviceBuffer<scalar> dRho, dMu, dNu;
    DeviceVectorBoundary dbU = buildDeviceVectorBoundary(f.U, fvp, g);

    RunReport rep;
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
        std::vector<scalar> poi, pob;
        dPhiI.copyTo(poi); dPhiB.copyTo(pob);
        DeviceBuffer<scalar> dPhiOI(poi), dPhiOB(pob);

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

        deviceInterStep(dm, rep.deltaT, C, props, H, dGh, dGhf, dMagSf,
                        dA, dAOld, dUx, dUy, dUz, dUox, dUoy, dUoz,
                        dPhiI, dPhiB, dPhiOI, dPhiOB, dUFixes, dPrgh, dP,
                        dNH, dNHB, dABnd, dK, dAFixes, dAFlag, dbU,
                        dRho, dMu, dNu, dRpI, dRpB);

        // turbulence->correct(), interFoam.C:169-172 -- after the last pressure corrector of the one
        // outer corrector this loop runs. dbU is current: the step's last updateUBoundary rebuilt it
        // and the pressureInletOutletVelocity switch ran on it after that.
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
            ti.rhoOld = &dRhoOld;
            ti.nu = &dStepNu;
            ti.nuBnd = &dStepNuBnd;
            ti.deltaT = rep.deltaT;
            ti.epsilonLog = &rep.epsilonSolves;
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
                f.rhoPhi.boundary.assign(fvp.size(), std::vector<scalar>());
                std::size_t off = 0;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
                    f.rhoPhi.boundary[pi].assign(rb.begin() + off, rb.begin() + off + n);
                    off += n;
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
            ti.epsilonLog = &rep.epsilonSolves;
            ti.kLog = &rep.kSolves;
            correctInterTurbulence(f.turbulence, ti, m, g, fvp);
        }
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
        std::vector<scalar> pb;
        dPhiB.copyTo(pb);
        SurfaceScalarField phiOut;
        phiOut.internal = f.phi.internal;
        phiOut.boundary.resize(fvp.size());
        { label o = 0;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          { for (label i = 0; i < fvp[pi].size; ++i) phiOut.boundary[pi].push_back(pb[o + i]);
            o += fvp[pi].size; } }
        f.phi.boundary = phiOut.boundary;
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
