// brae's interFoam time loop -- see inter_driver_cpp.cuh for why the driver owns no numerics and for
// the four old-time fields that are its actual content.
#include <filesystem>
#include "inter_driver_cpp.cuh"
#include "inter_correct_phi_cpp.cuh"
#include "inter_solve_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include "time_controls.cuh"
#include "foam_field_reader.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

// movingWallVelocityFvPatchVectorField::updateCoeffs when the mesh moves: Uwall() on every patch of
// that type, assigned as the patch's fixed value.
//
//     oldFc = face::centre(oldPoints)                 -- face::centre, not the mesh's Cf
//     Up    = (faceCentres - oldFc)/deltaT            -- the new face::centre
//     Un    = meshPhi_p/(magSf + VSMALL)
//     Uwall = Up + n*(Un - (n & Up))
//
// The normal component is REPLACED by the swept volume's, so the wall's flux is the mesh flux
// exactly and a closed tank's boundary balance (adjustPhi) closes on it.
void updateMovingWallVelocity(
    InterFields& f,
    const DynamicMotionSolverFvMesh& dyn,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar deltaT)
{
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!f.movingWallVelocityPatch[pi]) continue;
        const FvPatch& q = patches[pi];
        std::vector<vector> uwall(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i)
        {
            const label facei = q.start + i;
            const vector oldFc = faceCentreOfPoints(m, facei, dyn.oldPoints());
            const vector newFc = faceCentreOfPoints(m, facei, m.points());
            const vector Up = (newFc - oldFc)/deltaT;
            const vector n = g.Sf()[facei]/q.magSf[i];
            // VSMALL
            const scalar Un = dyn.meshPhi().boundary[pi][i]/(q.magSf[i] + scalar(1e-300));
            uwall[i] = Up + n*(Un - dot(n, Up));
        }
        f.U.boundary[pi]->setStoredValues(std::move(uwall));
    }
}

}   // namespace


namespace {

// The time the start directory names -- OpenFOAM's time directories ARE their times. A directory whose
// name is not one cannot tell the loop when it starts, and is refused rather than read as 0.
scalar startTimeOf(const std::string& startDir)
{
    std::filesystem::path p(startDir);
    if (p.filename().empty())
    {
        p = p.parent_path();
    }
    const std::string name = p.filename().string();
    char* end = nullptr;
    const double v = std::strtod(name.c_str(), &end);
    if (name.empty() || end == name.c_str() || *end != '\0' || !std::isfinite(v))
    {
        throw std::runtime_error(
            "brae interFoam: the start directory `" + startDir + "` is not named by a time, so the loop "
            "cannot tell when the case starts. OpenFOAM reads the fields from the directory of the start "
            "time; pass that one.");
    }
    return static_cast<scalar>(v);
}

} // namespace


RunReport runInterFoam(
    const std::string& caseDir,
    const std::string& startDir,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    label nSteps,
    bool verbose,
    InterFields* fieldsOut,
    scalar endTime,
    PressureTaps* pressureTaps,
    const MutableMesh* mutableMesh)
{
    InterFields f = buildInterFields(caseDir, startDir, m, g, patches);
    const label nC = m.nCells();

    // A MESH THAT MOVES is attached to the caller's mutable objects -- the ones the fields were just
    // built against, checked by address -- and moved in place at every mesh update below.
    DynamicMotionSolverFvMesh* dyn = f.dynamicMesh.get();
    if (dyn)
    {
        if (!mutableMesh || !mutableMesh->m || !mutableMesh->g || !mutableMesh->patches)
        {
            throw std::runtime_error(
                "brae interFoam: the case moves its mesh (" + dyn->motionType() + ") and the caller handed "
                "the driver a mesh it may not move. Pass the same mesh, geometry and patches through "
                "MutableMesh; the fields hold references to those patches and must see every move.");
        }
        if (mutableMesh->m != &m || mutableMesh->g != &g || mutableMesh->patches != &patches)
        {
            throw std::runtime_error(
                "brae interFoam: MutableMesh names different objects from the mesh, geometry and patches "
                "the fields were built against. Moving a copy would leave every field on the old mesh.");
        }
        dyn->attach(*mutableMesh->m, *mutableMesh->g, *mutableMesh->patches);
    }
    // A cyclicACMI pair whose `scale` moves with time is rescaled at every step, in place, on the
    // caller's mutable objects -- the same check as for a moving mesh
    cpu::cyclicACMI::Interfaces* acmi = mutableMesh ? mutableMesh->acmi : nullptr;
    {
        bool hasACMI = false;
        for (const FvPatch& q : patches)
        {
            hasACMI = hasACMI || q.type == "cyclicACMI";
        }
        if (hasACMI && (!acmi || acmi->empty()))
        {
            throw std::runtime_error(
                "brae interFoam: the mesh has a cyclicACMI pair and the caller handed the driver no ACMI "
                "state. Couple it with cpu::cyclicACMI::setup and pass the result through MutableMesh.");
        }
        if (acmi && acmi->scaled())
        {
            if (mutableMesh->m != &m || mutableMesh->g != &g || mutableMesh->patches != &patches)
            {
                throw std::runtime_error(
                    "brae interFoam: MutableMesh names different objects from the mesh, geometry and "
                    "patches the fields were built against; the cyclicACMI rescale would move a copy.");
            }
            if (dyn)
            {
                throw std::runtime_error(
                    "brae interFoam: the case scales a cyclicACMI interface AND moves its mesh. OpenFOAM "
                    "then re-runs the AMI and rescales the mesh flux in cyclicACMIFvPatch::movePoints; "
                    "that is not ported.");
            }
            // WHERE IN THE STEP the rescale lands is part of the answer (cyclic_acmi_cpp.cuh): after
            // alphaEqn.H forms phic and before the pre-solve. That point is gated for the pre-solving
            // (MULESCorr) path with no isotropic or shear compression and one alpha sub-cycle. Each of
            // the three moves OpenFOAM's first interpolation across the pair -- the explicit path to the
            // high-order flux after phir, icAlpha/scAlpha to an interpolate before phic, a sub-cycle to
            // a rescale at every sub-cycle's own time -- and none is gated, so each is refused.
            if (!f.alphaCtl.MULESCorr || f.alphaCtl.icAlpha != scalar(0) || f.alphaCtl.scAlpha != scalar(0)
             || f.alphaCtl.nAlphaSubCycles != 1)
            {
                throw std::runtime_error(
                    "brae interFoam: the case scales a cyclicACMI interface with time, and the step's "
                    "rescale point is ported for `MULESCorr yes`, icAlpha 0, scAlpha 0 and nAlphaSubCycles 1 "
                    "only; this case sets MULESCorr " + std::string(f.alphaCtl.MULESCorr ? "yes" : "no") +
                    ", icAlpha " + std::to_string((double)f.alphaCtl.icAlpha) + ", scAlpha " +
                    std::to_string((double)f.alphaCtl.scAlpha) + ", nAlphaSubCycles " +
                    std::to_string((long)f.alphaCtl.nAlphaSubCycles) + ".");
            }
        }
    }
    // A cyclicAMI pair is coupled by the caller; after every mesh move its AMI is recomputed and both
    // patches coupled again, on the caller's mutable objects
    cpu::cyclicAMIFvPatch::Interfaces* amiPairs = mutableMesh ? mutableMesh->ami : nullptr;
    {
        bool hasAMI = false;
        for (const FvPatch& q : patches)
        {
            hasAMI = hasAMI || q.type == "cyclicAMI";
        }
        if (hasAMI && (!amiPairs || amiPairs->empty()))
        {
            throw std::runtime_error(
                "brae interFoam: the mesh has a cyclicAMI pair and the caller handed the driver no AMI "
                "state. Couple it with cpu::cyclicAMIFvPatch::setup and pass the result through MutableMesh.");
        }
    }
    RunReport rep;
    // THE CLOCK STARTS AT THE START TIME: the instant the fields are read from, which is where
    // OpenFOAM's Time begins (Time::setControls). It started at 0 whatever the case said, so a restart
    // at 0.49 read every time-dependent input -- a wave, a table, a coded ACMI scale -- half a second
    // early, and endTime was measured from the wrong origin.
    const scalar startTime = startTimeOf(startDir);
    rep.time = startTime;
    // the mesh's GAMG hierarchy, built by the first GAMG solve -- pcorr's, p_rgh's or the mesh
    // motion's -- and kept for the run
    GamgAgglomerationCache gamgCache;

    // THE CASE'S OWN CorrectPhi CONTROLS: pcorr and pcorrFinal, the laplacian's default and PIMPLE's
    // non-orthogonal correctors
    CorrectPhiControls cpc;
    cpc.pcorr = &f.pcorrSolve;
    cpc.pcorrFinal = &f.pcorrSolveFinal;
    cpc.gamgCache = &gamgCache;
    cpc.correctedLaplacian = f.laplacianScheme.corrected;
    cpc.snGradLimitCoeff = f.laplacianScheme.limitCoeff;
    cpc.nNonOrthogonalCorrectors = f.nNonOrthogonalCorrectors;

    // initCorrectPhi.H, which runs for EVERY case, moving or not and with correctPhi or without: CorrectPhi
    // on the phi createFields built, with rAUf exactly 1. At rest it is exact and costs one solve of
    // zero iterations; on a case that starts moving it makes the first alpha step convect a flux that
    // is divergence-free. It runs BEFORE the old-time copies below, so phi.oldTime() is the corrected
    // flux, and before the first Courant number reads it.
    {
        const SurfaceScalarField one = unitFaceField(m, patches);
        CorrectPhiInput cin;
        cin.rAUf = &one;
        cin.meshChanging = false;
        cin.rhoPhi = &f.rhoPhi;
        cin.solveLog = &rep.pcorrSolves;
        correctPhi(f.U, f.phi, f.p_rgh, cin, cpc, m, g, patches);
        pushFluxToPatches(f, patches);
    }

    // which outer corrector of the step this is, for the mesh update interFoam.C:118 makes on the
    // first one only; reset with every time step
    label outerOfStep = -1;

    // The four old-time fields. See the header: each is read by something different, and each is
    // correct in isolation, which is why losing one is invisible to any single equation's gate.
    std::vector<scalar> alphaOld = f.alpha1.internal;
    std::vector<vector> UOld     = f.U.internal;
    // U.oldTime()'s PATCH values, which ddtCorr's boundary half reads -- see DdtCorrInput::UOldBnd
    auto patchValuesOf = [&](const GeometricField<vector>& fld)
    {
        std::vector<std::vector<vector>> b(fld.boundary.size());
        for (std::size_t pi = 0; pi < fld.boundary.size(); ++pi)
        {
            b[pi] = fld.boundary[pi]->value();
        }
        return b;
    };
    std::vector<std::vector<vector>> UOldBnd = patchValuesOf(f.U);
    std::vector<scalar> rhoOld   = f.rho;
    SurfaceScalarField  phiOld   = f.phi;
    // ...and a fifth on a moving mesh: Uf.oldTime(), which ddtCorr reads in phi.oldTime()'s place
    SurfaceVectorField UfOld = f.Uf;

    SurfaceScalarField prevCorr;                 // alphaApplyPrevCorr's cache
    // cyclicACMIPolyPatch::updateAreas runs once per time index (prevTimeIndex_)
    bool acmiRescaledThisStep = false;
    GamgSolveLog gamgLog;
    rep.deltaT = f.deltaT;

    for (label step = 0; step < nSteps; ++step)
    {
        // Time::run() (Time.C:1000), and it sits HERE -- above CourantNo.H and setDeltaT.H -- so the
        // deltaT it tests is the one the previous iteration ended with, not the one this iteration is
        // about to choose. The half-step slack is OpenFOAM's: a run overshoots endTime by up to half a
        // step rather than landing on it, unless writeControl is adjustableRunTime and Time::
        // adjustDeltaT() trims the last few.
        if (!(rep.time < endTime - scalar(0.5)*rep.deltaT)) break;

        // The stages, in interFoam.C's order. runTimeStep owns the order; this lambda owns the work.
        // THE CASE'S OWN PIMPLE CONTROLS, read in buildInterFields. These were hardcoded here until
        // damBreak's fvSolution was actually read: it says nOuterCorrectors 1, nCorrectors 3 AND
        // `momentumPredictor no`, and the last of those changes which algorithm runs.
        const LoopControls lc = f.pimple;
        // which outer corrector this is: alphaControls opens each one. fvMatrix::solve() selects the
        // `Final` solver entry on the last, and U's is the one place this driver has to know.
        label outerIndex = -1;
        const SolutionDirections solD = solutionDirections(patches);

        SolverHooks hooks;
        hooks.run = [&](Stage s)
        {
            switch (s)
            {
                case Stage::courantNo:
                {
                    const std::vector<scalar> sumPhi = surfaceSumMagPhi(
                        m.owner(), m.neighbour(), f.phi.internal,
                        [&]{ std::vector<scalar> b;
                             for (const auto& p : f.phi.boundary) b.insert(b.end(), p.begin(), p.end());
                             return b; }(),
                        nC, m.nInternalFaces());
                    rep.CoNum = courantNo(sumPhi, g.V(), rep.deltaT).CoNum;
                    // alphaCourantNo needs the same sumPhi, so it is cached on the report rather than
                    // recomputed -- OpenFOAM's two #includes build it twice, which is the one place
                    // this driver deliberately differs and it changes no number.
                    rep.alphaCoNum = alphaCourantNo(sumPhi, f.alpha1.internal, g.V(), rep.deltaT).CoNum;
                    break;
                }
                case Stage::alphaCourantNo:  break;    // computed above, from the same sumPhi
                case Stage::setDeltaT:
                    // Time::adjustDeltaT measures from the start, value() - startTime_ (Time.C:1150)
                    rep.deltaT = setDeltaTVoF(rep.deltaT, rep.CoNum, rep.alphaCoNum, f.timeCtl,
                                              rep.time - startTime, &f.writeCadence);
                    break;
                case Stage::advanceTime:
                    rep.time += rep.deltaT;
                    ++rep.steps;
                    acmiRescaledThisStep = false;
                    outerOfStep = -1;
                    // Time::operator++ moves writeTimeIndex_ AFTER the time, with the step that took
                    // it -- which is what the next adjustDeltaT measures the distance to.
                    f.writeCadence.advance(rep.time - startTime, rep.deltaT);
                    break;

                case Stage::meshUpdate:
                {
                    ++outerOfStep;
                    if (!dyn) break;
                    // interFoam.C:118: on the first outer corrector, or on every one under
                    // moveMeshOuterCorrectors
                    if (outerOfStep != 0 && !f.moveMeshOuterCorrectors) break;
                    // pimpleControl sets "finalIteration" on the mesh before the last outer
                    // corrector's body runs, so a displacement solve there takes the Final entry.
                    // The GAMG hierarchy is the mesh's: a displacement solve builds it or reuses the
                    // one p_rgh left, and the move marks it for rebuilding.
                    const bool finalIteration = (outerOfStep >= lc.nOuterCorrectors - 1);
                    dyn->update(rep.time, rep.deltaT, rep.steps, finalIteration, &gamgCache);
                    // cyclicAMIPolyPatch::initMovePoints marks the AMI out of date, and the next AMI()
                    // recomputes it on the moved points: before anything below interpolates across it
                    if (amiPairs && !amiPairs->empty())
                    {
                        amiPairs->update(m, *mutableMesh->g, *mutableMesh->patches);
                    }
                    // the move rebuilt every patch uncoupled; a cyclicAMI left that way would be read
                    // through an empty stencil by every coupled field
                    for (const FvPatch& q : *mutableMesh->patches)
                    {
                        if (q.type == "cyclicAMI" && (!q.coupled || q.amiOffsets.size() != static_cast<std::size_t>(q.size) + 1))
                        {
                            throw std::runtime_error(
                                "brae interFoam: the cyclicAMI patch `" + q.name + "` was not coupled again after "
                                "the mesh moved.");
                        }
                    }

                    // dynamicMotionSolverFvMesh::update ends in U.correctBoundaryConditions(): a
                    // movingWallVelocity patch takes the wall's velocity from the motion of THIS
                    // step (movingWallVelocityFvPatchVectorField.C, Uwall), every other patch is
                    // evaluated as it stands
                    updateMovingWallVelocity(f, *dyn, m, g, patches, rep.deltaT);
                    // ...and a wave condition's model, whose FIRST update of the step is this one on a
                    // moving mesh: OpenFOAM's log prints "Updating ... wave model" right after the
                    // motion solve, ahead of pcorr and the alpha sub-cycles, so the absorber reads the
                    // water level the last step left. The UEqn stage's update is then a no-op for this
                    // time index, as OpenFOAM's is. Measured on waveMakerSolitary with pcorr converged:
                    // updated in UEqn instead, from the alpha the sub-cycles left, U went from 7.1e-12 of
                    // OpenFOAM after step one to 1.1e-07 after step two, the gap at the outlet.
                    updateWaveVelocity(f.waves, f.alpha1, f.U, rep.time, rep.steps, m, g, patches);
                    f.U.evaluateBoundary();

                    // interFoam.C:130-131: gh and ghf follow the cell and face centres
                    ghField(f.g, f.ghRefValue, g.C(), f.gh);
                    {
                        std::vector<vector> Cf(g.Cf().begin(), g.Cf().begin() + m.nInternalFaces());
                        ghField(f.g, f.ghRefValue, Cf, f.ghfInternal);
                        for (std::size_t pi = 0; pi < patches.size(); ++pi)
                        {
                            const FvPatch& q = patches[pi];
                            std::vector<vector> bCf(g.Cf().begin() + q.start, g.Cf().begin() + q.start + q.size);
                            ghField(f.g, f.ghRefValue, bCf, f.ghfBoundary[pi]);
                        }
                    }
                    // interFoam.C:136-146 under `correctPhi`: the ABSOLUTE flux rebuilt from the face
                    // velocity on the moved mesh, CorrectPhi against it with the last corrector's rAU,
                    // the result made relative, and the mixture corrected on the new geometry --
                    // which moves nHatf, and nHatf is what the alpha step compresses along
                    if (f.correctPhi)
                    {
                        // phi = mesh.Sf() & Uf()
                        for (label face = 0; face < m.nInternalFaces(); ++face)
                        {
                            f.phi.internal[face] = dot(g.Sf()[face], f.Uf.internal[face]);
                        }
                        for (std::size_t pi = 0; pi < patches.size(); ++pi)
                        {
                            const FvPatch& q = patches[pi];
                            if (q.type == "empty") continue;
                            for (label i = 0; i < q.size; ++i)
                            {
                                f.phi.boundary[pi][i] = dot(g.Sf()[q.start + i], f.Uf.boundary[pi][i]);
                            }
                        }
                        // correctPhi.H: CorrectPhi(U, phi, p_rgh, interpolate(rAU), 0, pimple)
                        const SurfaceScalarField rAUf = fvc::interpolate(f.rAU, m, g, patches);
                        CorrectPhiInput cin;
                        cin.rAUf = &rAUf;
                        cin.meshChanging = true;
                        cin.meshPhi = &dyn->meshPhi();
                        cin.rhoPhi = &f.rhoPhi;
                        cin.solveLog = &rep.pcorrSolves;
                        correctPhi(f.U, f.phi, f.p_rgh, cin, cpc, m, g, patches);
                        // fvc::makeRelative(phi, U)
                        makeRelativeFlux(f.phi, dyn->meshPhi());
                        pushFluxToPatches(f, patches);
                        // mixture.correct(): calcNu, whose values alpha has not moved, then
                        // interfaceProperties::correct() on the moved mesh
                        cpu::twoPhase::mixtureMu(f.alpha1.internal, f.mixture.phases, f.mu);
                        cpu::twoPhase::mixtureNu(f.alpha1.internal, f.mu, f.mixture.phases, f.nu);
                        updateMixtureBoundary(f, patches);
                        interfaceProps::calculateK(f.alpha1, f.interface, m, g, patches, false, f.nHatf, f.K);
                    }
                    break;
                }

                case Stage::alphaControls:             // read once, in buildInterFields
                    ++outerIndex;
                    break;

                case Stage::alphaEqnSubCycle:
                {
                    AlphaStepInput ai;
                    ai.phi = &f.phi; ai.phiCN = &f.phi;   // Euler: ocCoeff 0, so phiCN IS phi
                    ai.cAlpha = f.interface.cAlpha;
                    ai.nAlphaCorr = f.alphaCtl.nAlphaCorr;
                    ai.icAlpha = f.alphaCtl.icAlpha;
                    ai.scAlpha = f.alphaCtl.scAlpha;
                    ai.rho1 = f.mixture.phases.rho1; ai.rho2 = f.mixture.phases.rho2;
                    ai.alphaScheme  = f.divPhiAlpha;
                    ai.alpharScheme = f.divPhirbAlpha;
                    ai.MULESCorr = f.alphaCtl.MULESCorr;
                    ai.alphaApplyPrevCorr = f.alphaCtl.alphaApplyPrevCorr;
                    ai.alpha2BndOut = &f.alpha2Bnd;
                    // the case's own tolerances, where a struct default of 1e-8 used to stand
                    ai.tolAlpha = f.aSolve.tol;
                    ai.relTolAlpha = f.aSolve.relTol;
                    ai.maxIterAlpha = f.aSolve.maxIter;
                    ai.minIterAlpha = f.aSolve.minIter;
                    // ...and the case's own smoother, which the host can now run
                    ai.smoothSolver = f.aSolve.gaussSeidel();
                    ai.symmetric = (f.aSolve.smoother == "symGaussSeidel");
                    ai.nSweeps = f.aSolve.nSweeps;
                    ai.solveLog = &rep.alphaSolves;
                    // A GATE'S CONTROL, read here as BRAE_PTOL is and announced every step it is on:
                    // see AlphaStepInput::controlPrevCorrOutletOnPhiCN. It makes the answer WRONG.
                    if (std::getenv("BRAE_CONTROL_PREVCORR_PHICN"))
                    {
                        ai.controlPrevCorrOutletOnPhiCN = true;
                        std::printf("  *** CONTROL MODE: the previous-correction limiter is reading "
                                    "phiCN, not alphaPhi10. This run is deliberately wrong. ***\n");
                    }

                    // which sub-cycle this is, 1-based: the wave conditions are evaluated at the
                    // SUB-CYCLE's time and time index (inter_waves_cpp.cuh)
                    label subCycle = 0;
                    auto step1 = [&](const std::vector<scalar>& aOld, scalar dtSub,
                                     std::vector<scalar>& aNew, SurfaceScalarField& rPhi)
                    {
                        ++subCycle;
                        AlphaStepInput sub = ai;
                        // cyclicACMIPolyPatch::updateAreas: the step's first interpolation across the
                        // pair, which alphaEqnStep places after phic (see its geometryUpdate)
                        if (acmi && acmi->scaled())
                        {
                            sub.geometryUpdate = [&]()
                            {
                                if (acmiRescaledThisStep)
                                {
                                    return;
                                }
                                acmi->rescale(rep.time, m, *mutableMesh->g, *mutableMesh->patches);
                                acmiRescaledThisStep = true;
                            };
                        }
                        sub.deltaT = dtSub;
                        // a moving mesh's volumes at this sub-cycle's clock: fvMesh::Vsc and Vsc0
                        // interpolate between V0 and V by the sub-cycle's position in the step
                        std::vector<scalar> VscK;
                        std::vector<scalar> Vsc0K;
                        if (dyn)
                        {
                            SubCycleTimeState ts;
                            ts.subCycling = f.alphaCtl.nAlphaSubCycles > 1;
                            const SubCycleClock clock = subCycleClock(
                                rep.time, rep.deltaT, rep.steps, f.alphaCtl.nAlphaSubCycles, subCycle);
                            ts.value = clock.t;
                            ts.deltaT = dtSub;
                            ts.value0 = rep.time;
                            ts.deltaT0 = rep.deltaT;
                            VscK = dyn->Vsc(ts);
                            Vsc0K = dyn->Vsc0(ts);
                            sub.Vsc = &VscK;
                            sub.Vsc0 = &Vsc0K;
                        }
                        if (f.waves.any)
                        {
                            const SubCycleClock clock = subCycleClock(
                                rep.time, rep.deltaT, rep.steps, f.alphaCtl.nAlphaSubCycles, subCycle);
                            sub.updateModelledBoundary = [&f, &m, &g, &patches, clock]()
                            {
                                updateWaveAlpha(f.waves, f.alpha1, f.U, clock.t, clock.timeIndex,
                                                m, g, patches);
                            };
                        }
                        SurfaceScalarField aPhi;
                        alphaEqnStep(f.alpha1, aOld, sub, f.interface, f.mulesCtl,
                                     m, g, patches, aPhi, rPhi, f.nHatf, f.K, &prevCorr);
                        aNew = f.alpha1.internal;
                    };
                    alphaEqnSubCycle(f.alphaCtl.nAlphaSubCycles, rep.deltaT,
                                     f.alpha1.internal, alphaOld, f.rhoPhi, step1);
                    // rhoPhi HAS JUST CHANGED, and a condition may name it (`phi rhoPhi;`). OpenFOAM's
                    // conditions look their flux up at every updateCoeffs, so UEqn's U and the first
                    // pressure corrector's p_rgh read THIS step's rhoPhi; brae's are told, and were
                    // last told at the end of the previous step. With phi the two moments hold the
                    // same field -- the alpha step does not touch phi -- which is why this push was
                    // never missed. Measured on damBreak with the atmosphere naming rhoPhi on U:
                    // alpha 2.5e-09 and U 2.2e-05 from OpenFOAM, from step two; on p_rgh alone 4.7e-10.
                    pushFluxToPatches(f, patches);
                    // ...and the boundary with it. Dropping this was tried together with the reset in
                    // alphaEqnStep: damBreak's alpha went thirty times further from OpenFOAM and
                    // capillaryRise did not move, so the extra evaluations are load-bearing rather
                    // than spurious.
                    f.alpha1.evaluateBoundary();
                    // ...and the conditions that look alpha up read the values THIS evaluate left
                    pushAlphaToPatches(f, patches);
                    break;
                }

                case Stage::mixtureCorrect:
                {
                    // rho == alpha1*rho1 + alpha2*rho2 (alphaEqnSubCycle.H:36), and the viscosities
                    // with it. rho.oldTime() is NOT touched: fvm::ddt's source needs the value from
                    // the start of the step, and this is where it would be lost.
                    for (label c = 0; c < nC; ++c) f.alpha2[c] = scalar(1) - f.alpha1.internal[c];
                    cpu::twoPhase::mixtureRho(f.alpha1.internal, f.alpha2, f.mixture.phases, f.rho);
                    cpu::twoPhase::mixtureMu (f.alpha1.internal, f.mixture.phases, f.mu);
                    cpu::twoPhase::mixtureNu (f.alpha1.internal, f.mu, f.mixture.phases, f.nu);
                    // THE BOUNDARY BLENDS FIRST, from alpha's patch values as they stand NOW. `rho ==`
                    // (alphaEqnSubCycle.H:36) and calcNu() both read them before the curvature below
                    // moves them: immiscibleIncompressibleTwoPhaseMixture::correct() is calcNu() THEN
                    // interfaceProperties::correct() (immiscibleIncompressibleTwoPhaseMixture.H:78-82),
                    // and at a contact-angle wall the second call rewrites alpha's gradient and so its
                    // patch value. Blending after it gave the laplacian a wall viscosity one contact-
                    // angle pass ahead: measured on capillaryRise against OpenFOAM's own UEqn.A() after
                    // one step, exact in every cell off the wall and 1.26% high in the air cells at it,
                    // which was the whole of the 0.2% that step leaves in U.
                    updateMixtureBoundary(f, patches);
                    // ...THEN THE CURVATURE. interFoam.C:154 calls mixture.correct() here, after the
                    // sub-cycle and before UEqn, and interfaceProperties::correct() IS calculateK.
                    // Rebuilding only rho/mu/nu leaves UEqn's surface-tension force one pass behind.
                    interfaceProps::calculateK(f.alpha1, f.interface, m, g, patches, false, f.nHatf, f.K);
                    break;
                }

                case Stage::UEqn:
                case Stage::pEqn:
                {
                    // The face force and the pressure corrector share every field they are built from,
                    // so they are assembled together here and pEqn runs on the matrix UEqn produced.
                    // Splitting them across two hook calls would mean rebuilding the curvature.
                    if (s == Stage::pEqn) break;       // done in the UEqn pass, see above

                    // U's boundaryField().updateCoeffs(), which UEqn's fvMatrix constructor runs: the
                    // step's own time and time index, so a wave model the alpha sub-cycles updated
                    // updates AGAIN, from the alpha they left. Ahead of everything that reads U_b.
                    updateWaveVelocity(f.waves, f.alpha1, f.U, rep.time, rep.steps, m, g, patches);
                    // ...AND pressureInletOutletVelocity's, WHICH RE-EVALUATES THE PATCH VALUE, not
                    // only its coefficients: its updateCoeffs() ends in directionMixed::evaluate()
                    // (pressureInletOutletVelocityFvPatchVectorField.C:118-131), and that evaluate
                    // clears the updated flag, so EVERY updateCoeffs looks the flux up again and
                    // rewrites the value. With `phi` nothing has moved since the last corrector's
                    // evaluate and this is the same value bit for bit. With `phi rhoPhi;` the alpha
                    // step has just moved the flux: measured on damBreak, alpha 8.4e-11 from OpenFOAM
                    // and the p_rgh residuals 5e-07 from step two without it, 1.2e-12 with.
                    updateVelocityPatchesFromCells(f.U, patches);

                    // UEqn.H uses mixture.surfaceTensionForce(), which reads the K the LAST
                    // mixture.correct() left -- it does not recompute one. An extra pass here would
                    // put UEqn's force one iteration ahead of the alpha equation's.
                    std::vector<scalar> sK;
                    interfaceProps::sigmaK(f.K, f.interface.sigma, sK);
                    const SurfaceScalarField sKf = fvc::interpolate(sK, m, g, patches);

                    // rho's CALCULATED patch values, not a zeroGradient copy -- see rhoWithPatchValues
                    const GeometricField<scalar> rhoF = rhoWithPatchValues(f.rho, f.rhoBnd, patches);
                    // snGradSchemes' default: the corrected (or limited) form on a mesh that is
                    // not orthogonal, through the fields' own Gauss linear gradients
                    const bool snCorr = f.snGradScheme.corrected;
                    const scalar snLim = f.snGradScheme.limitCoeff;
                    const SurfaceScalarField snRho = fvc::snGrad(rhoF, m, g, patches, snCorr, false, 0, snLim);
                    const SurfaceScalarField snA = fvc::snGrad(f.alpha1, m, g, patches, snCorr, false, 0, snLim);

                    SurfaceScalarField stf;
                    stf.internal.resize(static_cast<std::size_t>(m.nInternalFaces()));
                    for (label fi = 0; fi < m.nInternalFaces(); ++fi)
                        stf.internal[fi] = sKf.internal[fi]*snA.internal[fi];
                    // THE BOUNDARY IS NOT ZERO. surfaceTensionForce() is a surfaceScalarField and at a
                    // contact-angle wall snGrad(alpha1) is the gradient correctContactAngle just wrote
                    // -- 9681 on capillaryRise -- so this is precisely where the contact angle enters
                    // the pressure equation. Zeroing it cost 6% of the velocity at step 1, all of it
                    // at the wall.
                    stf.boundary.assign(patches.size(), std::vector<scalar>{});
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        const FvPatch& q = patches[pi];
                        stf.boundary[pi].resize(static_cast<std::size_t>(q.size));
                        for (label i = 0; i < q.size; ++i)
                            stf.boundary[pi][i] = sKf.boundary[pi][i] * snA.boundary[pi][i];
                    }

                    // fvc::snGrad(p_rgh) ON A fixedFluxPressure PATCH IS THE GRADIENT THE LAST
                    // constrainPressure LEFT THERE -- the previous step's last corrector's -- and zero
                    // only before the first one, which is OpenFOAM's construction value
                    // (fixedFluxPressureFvPatchScalarField.C, `gradient() = 0.0`). This zeroed it EVERY
                    // step. The term is read by the momentum predictor alone, and on a wall whose
                    // alpha is zeroGradient the stored gradient IS zero (phig_b vanishes with
                    // snGrad(rho)), which is every predictor case gated before this one. On
                    // waves/stokesI with `momentumPredictor yes` the inlet's is not: measured against
                    // real OpenFOAM after twenty steps, alpha 5.1e-04 and U 5.4% with the zero.
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        if (!f.p_rgh.boundary[pi]->updateableSnGrad()) continue;
                        if (f.p_rgh.boundary[pi]->snGradEverSet()) continue;
                        f.p_rgh.boundary[pi]->updateSnGrad(
                            std::vector<scalar>(static_cast<std::size_t>(patches[pi].size), scalar(0)));
                    }
                    const SurfaceScalarField snP = fvc::snGrad(f.p_rgh, m, g, patches, snCorr, false, 0, snLim);

                    SurfaceScalarField force;
                    {
                        std::vector<scalar> out;
                        momentumSourceFlux(stf.internal, f.ghfInternal, snRho.internal,
                                           snP.internal, g.magSf(), out);
                        force.internal = out;
                        force.boundary.assign(patches.size(), std::vector<scalar>{});
                        for (std::size_t pi = 0; pi < patches.size(); ++pi)
                        {
                            const FvPatch& q = patches[pi];
                            momentumSourceFlux(stf.boundary[pi], f.ghfBoundary[pi],
                                               snRho.boundary[pi], snP.boundary[pi],
                                               q.magSf, force.boundary[pi]);
                        }
                    }

                    std::vector<std::vector<scalar>> rhoB(patches.size()), nuB(patches.size()),
                                                     phB(patches.size());
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        const FvPatch& q = patches[pi];
                        rhoB[pi].resize(static_cast<std::size_t>(q.size));
                        nuB[pi].resize(static_cast<std::size_t>(q.size));
                        phB[pi] = f.rhoPhi.boundary.size() > pi
                                ? f.rhoPhi.boundary[pi]
                                : std::vector<scalar>(static_cast<std::size_t>(q.size), scalar(0));
                        // alpha's PATCH values, not the face cell's -- see InterFields::rhoBnd.
                        rhoB[pi] = f.rhoBnd[pi];
                        nuB[pi]  = f.nuBnd[pi];
                    }

                    InterMomentumInput mi;
                    mi.rhoPhi = &f.rhoPhi.internal; mi.rhoPhiBnd = &phB;
                    mi.rho = &f.rho; mi.rhoOld = &rhoOld; mi.rhoBnd = &rhoB;
                    mi.UOld = &UOld;
                    mi.V0 = dyn ? &dyn->V0() : nullptr;
                    // nuEff = nut + nu: the mixture's nu alone when the case is laminar
                    std::vector<scalar> nuEff;
                    std::vector<std::vector<scalar>> nuEffB;
                    interNuEff(f.turbulence, f.nu, nuB, nuEff, nuEffB);
                    mi.nuEff = &nuEff;
                    mi.nuEffBnd = &nuEffB;
                    mi.deltaT = rep.deltaT;
                    mi.scheme = f.divRhoPhiU;
                    mi.schemeCoeff = f.divRhoPhiUCoeff;
                    mi.relaxEquationU = f.relaxEquationU; mi.relaxU = f.relaxU;
                    // laplacianSchemes' default, for the viscous term's fvm::laplacian(rho*nuEff, U)
                    mi.correctedLaplacian = f.laplacianScheme.corrected;
                    mi.snGradLimitCoeff = f.laplacianScheme.limitCoeff;
                    // gradSchemes' grad(U): cellLimited or not, for linearUpwind and the viscous term
                    mi.gradULimitK = f.gradULimitK;

                    // THE CASE'S OWN SOLVE FOR U: UFinal on the last outer corrector, U on the others,
                    // its smoother where it names a Gauss-Seidel one, and only the components the
                    // mesh solves. This was a struct default of 1e-7 on PBiCGStab, reading nothing.
                    MomentumSolveControls msc;
                    if (f.momentumPredictorOn)
                    {
                        const bool finalOuter = (outerIndex >= lc.nOuterCorrectors - 1);
                        const InterFields::AlphaLinearSolve& us = finalOuter ? f.uSolveFinal : f.uSolve;
                        msc.tolU = us.tol;
                        msc.relTolU = us.relTol;
                        msc.maxIterU = us.maxIter;
                        msc.which.smoothSolver = us.gaussSeidel();
                        msc.which.symmetric = (us.smoother == "symGaussSeidel");
                        msc.which.nSweeps = us.nSweeps;
                        msc.solutionD = &solD;
                        msc.solveLog = rep.uSolves;
                    }
                    // U.boundaryFieldRef().updateCoeffs(), which the fvMatrix constructor runs when UEqn
                    // is assembled (fvMatrix.C:396). A flowRateInletVelocity recomputes its value there
                    // from the rate at this time and -- for a massFlowRate -- the field named `rho` on
                    // the patch, which in interFoam is the MIXTURE's (flowRateInletVelocity...C:201-237).
                    // This loop never called it: RAS/angledDuct ships `massFlowRate constant 0.1` beside
                    // `value uniform (0 0 0)`, the inlet stayed at the file's zero, and after ten steps
                    // brae's largest velocity was 1.9e-04 m/s against OpenFOAM's 0.21 -- U 100% out, with
                    // no notice. waterChannel's volumetric inlet carries no `value`, so its constructor
                    // had already built the right one.
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        if (f.U.boundary[pi]->isFlowRateInlet())
                        {
                            f.U.boundary[pi]->updateFromDensity(f.rhoBnd[pi], rep.time);
                        }
                        // ...and a variableHeightFlowRateInletVelocity rebuilds itself there too, from
                        // the STORED values of the phase field it names on this patch
                        // (variableHeightFlowRateInletVelocityFvPatchVectorField.C:103-139)
                        if (f.U.boundary[pi]->isVariableHeightFlowRateInlet())
                        {
                            if (f.U.boundary[pi]->alphaFieldName() != f.alphaName)
                                throw std::runtime_error(
                                    "brae interFoam: U patch `" + patches[pi].name + "` is a "
                                    "variableHeightFlowRateInletVelocity naming `alpha "
                                    + f.U.boundary[pi]->alphaFieldName() + "`, and this case's phase field is `"
                                    + f.alphaName + "`. OpenFOAM looks the named field up and stops "
                                    "without it.");
                            f.U.boundary[pi]->updateFromAlphaPatch(f.alpha1.boundary[pi]->value(), rep.time);
                        }
                    }
                    if (!f.fvOptions.empty())
                    {
                        mi.fvOptions = &f.fvOptions;
                        mi.nuLaminar = &f.nu;
                    }
                    // MRF.correctBoundaryVelocity(U), UEqn.H:1 -- before the matrix reads U's patches
                    if (!f.mrfZones.empty())
                    {
                        MRF::correctBoundaryVelocity(f.U, f.mrfZones, patches);
                        mi.mrf = &f.mrfZones;
                    }
                    FvVectorMatrix UEqn;
                    momentumPredictor(f.U, mi, force, msc, m, g, patches,
                                      f.momentumPredictorOn, UEqn);

                    DdtCorrInput dc;
                    dc.phiOld = &phiOld; dc.UOld = &UOld; dc.deltaT = rep.deltaT;
                    dc.UOldBnd = &UOldBnd;
                    // ddtCorr(U, phi, Uf) is ddtCorr(U, Uf) when the mesh is dynamic
                    dc.UfOld = dyn ? &UfOld : nullptr;

                    PressureStepInput pin;
                    pin.UEqn = &UEqn; pin.rho = &f.rho; pin.gh = &f.gh; pin.ghf = &f.ghfInternal;
                    pin.ghfBnd = &f.ghfBoundary;
                    pin.stf = &stf; pin.snGradRho = &snRho; pin.ddt = &dc;
                    pin.rhoBnd = &f.rhoBnd;
                    pin.nuBnd = &f.nuBnd;
                    pin.rhoPhi = &f.rhoPhi;
                    pin.taps = pressureTaps;
                    pin.solveLog = &rep.pSolves;
                    pin.mrf = f.mrfZones.empty() ? nullptr : &f.mrfZones;
                    pin.meshPhi = dyn ? &dyn->meshPhi() : nullptr;
                    pin.Uf = dyn ? &f.Uf : nullptr;
                    // pEqn.H:4, rAU.ref() = 1/UEqn.A(): kept for the next mesh update's CorrectPhi
                    pin.rAUOut = &f.rAU;

                    PressureSolveControls psc;
                    psc.nCorrectors = lc.nCorrectors;
                    psc.nNonOrthogonalCorrectors = f.nNonOrthogonalCorrectors;
                    psc.correctedLaplacian = f.laplacianScheme.corrected;
                    psc.snGradLimitCoeff = f.laplacianScheme.limitCoeff;
                    // p_rgh.needReference() and setRefCell, read with the case -- see InterFields::pRef
                    psc.needReference = f.pRef.needReference;
                    psc.pRefCell = f.pRef.pRefCell;
                    psc.pRefValue = f.pRef.pRefValue;
                    // the CASE's own solve, not a hardcoded 1e-9. BRAE_PTOL still overrides, because
                    // a device-vs-host gate has to pin both sides to one stopping point.
                    const char* ptol = std::getenv("BRAE_PTOL");
                    psc.tolP = ptol ? std::atof(ptol) : f.pSolve.tol;
                    psc.relTolP = ptol ? scalar(0) : f.pSolve.relTol;
                    psc.maxIterP = f.pSolve.maxIter;
                    psc.pcgDIC = f.pSolve.pcgDIC();
                    psc.tolPFinal = ptol ? std::atof(ptol) : f.pSolveFinal.tol;
                    psc.relTolPFinal = ptol ? scalar(0) : f.pSolveFinal.relTol;
                    psc.maxIterPFinal = f.pSolveFinal.maxIter;
                    psc.pcgDICFinal = f.pSolveFinal.pcgDIC();
                    // GAMG where the entry names it, at the same stopping point as the lines above
                    GamgControls gamgP = f.pSolve.gamg;
                    gamgP.tolerance = psc.tolP;
                    gamgP.relTol = psc.relTolP;
                    GamgControls gamgPFinal = f.pSolveFinal.gamg;
                    gamgPFinal.tolerance = psc.tolPFinal;
                    gamgPFinal.relTol = psc.relTolPFinal;
                    psc.gamg = f.pSolve.gamgSolver() ? &gamgP : nullptr;
                    psc.gamgFinal = f.pSolveFinal.gamgSolver() ? &gamgPFinal : nullptr;
                    // the preconditioner's own controls are the sub-dictionary's, BRAE_PTOL or not
                    psc.pcgGamg = f.pSolve.pcgGamg() ? &f.pSolve.gamgPrecond : nullptr;
                    psc.pcgGamgFinal = f.pSolveFinal.pcgGamg() ? &f.pSolveFinal.gamgPrecond : nullptr;
                    psc.gamgCache = &gamgCache;
                    pin.gamgLog = &gamgLog;
                    for (label c = 0; c < lc.nCorrectors; ++c)
                    {
                        psc.finalCorrector = (c == lc.nCorrectors - 1);
                        // a predictor's solve ends in U.correctBoundaryConditions(), which clears the flag
                        pin.uPatchesUpdatedAtEntry = (c == 0) && !f.momentumPredictorOn;
                        pressureCorrector(f.p_rgh, f.U, f.phi, f.p, pin, psc, m, g, patches);
                    }
                    // alpha1's inletOutlet reads the flux the step ended on, at the next MULES pass.
                    pushFluxToPatches(f, patches);
                    break;
                }

                case Stage::turbulenceCorrect:
                {
                    // interFoam.C:169-172, after the last pressure corrector: the closure sees the
                    // U and phi this outer corrector ended on, and the rho, rhoPhi and nu the alpha
                    // step left. Does nothing on a laminar case.
                    InterTurbulenceStepInput ti;
                    ti.U = &f.U;
                    ti.phi = &f.phi;
                    ti.rhoPhi = &f.rhoPhi;
                    ti.rho = &f.rho;
                    ti.rhoBnd = &f.rhoBnd;
                    ti.rhoOld = &rhoOld;
                    ti.nu = &f.nu;
                    ti.nuBnd = &f.nuBnd;
                    ti.deltaT = rep.deltaT;
                    // a moving mesh's old volumes and mesh flux, for the closure's ddt and divU
                    if (dyn && dyn->moving())
                    {
                        ti.V0 = &dyn->V0();
                        ti.meshPhi = &dyn->meshPhi();
                    }
                    ti.fvOptions = f.fvOptions.empty() ? nullptr : &f.fvOptions;
                    ti.epsilonLog = &rep.epsilonSolves;
                    ti.omegaLog = &rep.omegaSolves;
                    ti.kLog = &rep.kSolves;
                    correctInterTurbulence(f.turbulence, ti, m, g, patches);
                    break;
                }
                case Stage::write:            break;
            }
        };

        runTimeStep(lc, hooks);

        // ...and the old-time set moves forward, all four together -- five on a moving mesh.
        alphaOld = f.alpha1.internal;
        UOld     = f.U.internal;
        UOldBnd  = patchValuesOf(f.U);
        rhoOld   = f.rho;
        phiOld   = f.phi;
        UfOld = f.Uf;

        if (verbose)
        {
            scalar aMin = f.alpha1.internal[0], aMax = f.alpha1.internal[0], mass = 0, maxU = 0;
            for (label c = 0; c < nC; ++c)
            {
                aMin = std::fmin(aMin, f.alpha1.internal[c]);
                aMax = std::fmax(aMax, f.alpha1.internal[c]);
                mass += f.alpha1.internal[c]*g.V()[c];
                const vector& v = f.U.internal[c];
                maxU = std::fmax(maxU, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
            }
            // %.17g on t and dt -- round-trip precision -- because these two are what a clock gate
            // reads back. At %.9g the gate bottomed out at 2.389e-09 on host AND device and stayed
            // there when the fields moved four orders closer to OpenFOAM's: it was measuring this
            // printf. See tests/interfoam_dambreak_clock_vs_openfoam.sh.
            std::printf("  t = %.17g  dt = %.17g  Co %.3f  alphaCo %.3f  "
                        "alpha [%.3e, %.15g]  max|U| %.4f\n",
                        (double)rep.time, (double)rep.deltaT, (double)rep.CoNum,
                        (double)rep.alphaCoNum, (double)aMin, (double)aMax, (double)maxU);
        }
    }

    // the final state
    rep.alphaMin = f.alpha1.internal[0];
    rep.alphaMax = f.alpha1.internal[0];
    rep.alphaMass = 0;
    rep.maxU = 0;
    for (label c = 0; c < nC; ++c)
    {
        rep.alphaMin = std::fmin(rep.alphaMin, f.alpha1.internal[c]);
        rep.alphaMax = std::fmax(rep.alphaMax, f.alpha1.internal[c]);
        rep.alphaMass += f.alpha1.internal[c]*g.V()[c];
        const vector& v = f.U.internal[c];
        rep.maxU = std::fmax(rep.maxU, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
    }
    {
        const std::vector<scalar> d = fvc::div(f.phi, m, g, patches);
        for (scalar v : d) rep.worstDivPhi = std::fmax(rep.worstDivPhi, std::fabs(v));
    }
    if (gamgCache.built)
    {
        const GamgAgglomeration& a = gamgCache.agglomeration;
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
    if (fieldsOut) *fieldsOut = std::move(f);
    return rep;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
