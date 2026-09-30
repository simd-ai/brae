// brae's interFoam time loop -- see inter_driver_cpp.cuh for why the driver owns no numerics and for
// the four old-time fields that are its actual content.
#include <filesystem>
#include "inter_driver_cpp.cuh"
#include "inter_amr_cpp.cuh"
#include "inter_correct_phi_cpp.cuh"
#include "inter_solve_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include "time_controls.cuh"
#include "foam_field_reader.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

// movingWallVelocityFvPatchVectorField::updateCoeffs when the mesh moves: Uwall() on every patch of
// that type, assigned as the patch's fixed value.
//
//     oldFc = face::centre(oldPoints)                 -- face::centre, not the mesh's Cf
//     Up    = (pp.faceCentres() - oldFc)/deltaT       -- and the CURRENT one is the MESH's Cf, which is
//                                                        a DIFFERENT algorithm from face::centre
//
// THOSE TWO CENTRES DO NOT CANCEL, not even for a rigid translation, and OpenFOAM's Up is not the
// prescribed wall velocity because of it. `face::centre` (face.C) seeds its centroid at Zero and adds
// every point, takes its triangle area from `(thisPoint - centre) ^ (nextPoint - centre)`, and ends in
// `sumAc/(3*sumA)`; `primitiveMeshTools::makeFaceCentresAndAreas` seeds at `p[f[0]]`, takes the area
// from `(nextPoint - thisPoint) ^ (fCentre - thisPoint)`, and ends in `(1/3)*sumAc/sumA`. Computing the
// current centre with face::centre instead makes the difference cancel EXACTLY, which is the one thing
// OpenFOAM does not do: MEASURED on RAS/electrostaticDeposition's metalSheet, brae's wall read
// (0, 0, -8.00119971999802e-02) against OpenFOAM's (-8.7e-16, 0, -8.00119972000634e-02) -- 8.3271e-14
// apart, TANGENTIAL to the face normal (so not the flux term), identical at step one and step two and
// bit-identical across three stagings whose pressure fields differ.
//     Un    = meshPhi_p/(magSf + VSMALL)
//     Uwall = Up + n*(Un - (n & Up))
//
// The normal component is REPLACED by the swept volume's, so the wall's flux is the mesh flux
// exactly and a closed tank's boundary balance (adjustPhi) closes on it.
void updateMovingWallVelocity(
    InterFields& f,
    const DynamicMotionSolverFvMesh& dyn,
    // fvc::meshPhi(U) -- the SCHEME's mesh flux. Uwall's normal component IS this number, so under
    // CrankNicolson a moving wall carries the off-centred flux and not mesh().phi()
    const SurfaceScalarField& meshPhiU,
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
            // THE CURRENT CENTRE SHOULD BE g.Cf()[facei] -- see the header above -- and is not, yet.
            // MEASURED with `g.Cf()[facei]` in its place: RAS/electrostaticDeposition's metalSheet goes
            // from 8.3271e-14 to EXACTLY 0 on all 3792 faces at both steps, and of the 27 moving-gate
            // arms 20 are unchanged or better and the wall is 0 or ~1e-16 on every one. It is HELD BACK
            // because laminar/sloshingCylinder then reads alpha 1.5769e-09 where it read 2.7674e-10,
            // against that case's OWN one-ulp control of 2.0913e-10 (OpenFOAM against itself, ten steps)
            // -- 1.3x its noise floor before, 7.5x after. g.Cf() is NOT stale (probed: bit-identical to
            // a freshly recomputed mesh-algorithm centre on every face), so the cylinder is a SECOND
            // defect this one was compensating, and it has to be found before the faithful form lands.
            const vector newFc = faceCentreOfPoints(m, facei, m.points());
            const vector Up = (newFc - oldFc)/deltaT;
            const vector n = g.Sf()[facei]/q.magSf[i];
            // VSMALL
            const scalar Un = meshPhiU.boundary[pi][i]/(q.magSf[i] + scalar(1e-300));
            uwall[i] = Up + n*(Un - dot(n, Up));
        }
        f.U.boundary[pi]->setStoredValues(std::move(uwall));
    }
}

}   // namespace


const SurfaceScalarField& fvcMeshPhi(
    const DynamicMotionSolverFvMesh& dyn,
    const InterFields& f)
{
    // fvc::meshPhi(U) asks the ddt scheme named for `ddt(U)`: EulerDdtScheme::meshPhi returns
    // mesh().phi() itself (EulerDdtScheme.C:759-765), CrankNicolson's is off-centred
    if (f.ddtU != DdtScheme::CrankNicolson) return dyn.meshPhi();
    if (f.meshPhiCN.internal.size() != dyn.meshPhi().internal.size())
        throw std::runtime_error(
            "brae interFoam: fvc::meshPhi under CrankNicolson was read before the mesh moved. The "
            "scheme's mesh flux is built at every move (interMeshUpdate); mesh().phi() is NOT it from "
            "the second step on.");
    return f.meshPhiCN;
}


// The time the start directory names -- see inter_driver_cpp.cuh.
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


// interFoam.C:112-149 -- THE MESH UPDATE both loops make at the top of an outer corrector, in
// OpenFOAM's own order: mesh.update(), and then, if the mesh changed, gh and ghf off the moved
// centres, the MRF, and under `correctPhi` the absolute flux rebuilt from the face velocity,
// CorrectPhi, fvc::makeRelative(phi, U) and mixture.correct().
//
// ONE COPY, called by both drivers. The device loop moves the mesh through this same host code --
// the motion solve is a host operation on either arm -- and then refreshes the buffers it uploaded
// from the geometry. A second copy in the device driver would drift from this one, and this one is
// the arm the tutorial gates hold.
void interMeshUpdate(
    DynamicMotionSolverFvMesh*             dyn,
    InterFields&                           f,
    const PrimitiveMesh&                   m,
    const FvGeometry&                      g,
    const std::vector<FvPatch>&            patches,
    const MutableMesh*                     mutableMesh,
    cpu::cyclicAMIFvPatch::Interfaces*     amiPairs,
    GamgAgglomerationCache&                gamgCache,
    const CorrectPhiControls&              cpc,
    RunReport&                             rep,
    // THE CLOCK OF THE STEP BEING TAKEN, passed rather than read off `rep`: the host driver advances
    // rep.time in its advanceTime stage before this one (interFoam.C does ++runTime before the PIMPLE
    // loop), while the device loop keeps rep.time at the step it is leaving and carries the new
    // instant separately. Reading rep.time here evaluated the motion one step late on the device --
    // measured on sloshingTank2D, max|U| 0.014 against the host's 10.38 after one step.
    scalar                                 time,
    label                                  timeIndex,
    label                                  outerOfStep,
    label                                  nOuterCorrectors,
    const fv::CrankNicolsonClock*          cn,
    std::vector<scalar>*                   correctPhiDivOut)
{
    if (!dyn) return;
    // interFoam.C:118: on the first outer corrector, or on every one under
    // moveMeshOuterCorrectors
    if (outerOfStep != 0 && !f.moveMeshOuterCorrectors) return;
    // pimpleControl sets "finalIteration" on the mesh before the last outer
    // corrector's body runs, so a displacement solve there takes the Final entry.
    // The GAMG hierarchy is the mesh's: a displacement solve builds it or reuses the
    // one p_rgh left, and the move marks it for rebuilding.
    const bool finalIteration = (outerOfStep >= nOuterCorrectors - 1);
    // fvMesh::movePoints grabs mesh().phi().oldTime() when the time index has advanced, BEFORE the
    // new flux overwrites it (fvMesh.C:971-978) -- so a second move in the same step under
    // moveMeshOuterCorrectors leaves the old level where it is, as OpenFOAM does.
    if (cn && dyn->moving() && f.meshPhiPrevIndex != timeIndex)
    {
        f.meshPhiPrev = dyn->meshPhi();
        f.meshPhiPrevIndex = timeIndex;
    }
    // THE FLUID LOAD ON THE BODY, for a rigidBodyMotion mesh alone. OpenFOAM builds a `forces`
    // function object inside rigidBodyMeshMotion::solve (rigidBodyMeshMotion.C:299-307) and it looks
    // the fields up in the registry, which is to say it takes them AS THEY STAND at the moment the
    // mesh is moved -- the previous outer corrector's, or the previous step's on the first. They are
    // gathered here, where the solver has them, rather than given to the mesh to own.
    //
    // `p` is the TOTAL pressure, p_rgh + rho*gh, and its boundary is p_rgh's plus the live mixture
    // density times gh at the boundary FACE CENTRES (`p == p_rgh + rho*gh` in pEqn.H forces the patch
    // values; `mesh.C()`'s boundary field is Cf). p_rgh's own boundary would leave out exactly the
    // buoyancy the body floats on.
    std::vector<std::vector<scalar>> bodyP, bodyNuEff;
    std::vector<scalar> bodyNuEffCells;
    BodyLoad load;
    const bool needLoad = (dyn->rigidBody() != nullptr);
    if (needLoad)
    {
        bodyP.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& prghB = f.p_rgh.boundary[pi]->value();
            const std::vector<scalar>& rb = f.rhoBnd[pi];
            const std::vector<scalar>& gb = f.ghfBoundary[pi];
            bodyP[pi].assign(prghB.size(), scalar(0));
            for (std::size_t k = 0; k < prghB.size() && k < rb.size() && k < gb.size(); ++k)
            {
                bodyP[pi][k] = prghB[k] + rb[k]*gb[k];
            }
        }
        // eddyViscosity::nuEff = nut + nu, which is what forces::devRhoReff takes the shear from
        interNuEff(f.turbulence, f.nu, f.nuBnd, bodyNuEffCells, bodyNuEff);
        load.U = &f.U;
        load.p = &bodyP;
        load.rho = &f.rhoBnd;
        load.nuEff = &bodyNuEff;
    }
    dyn->update(time, rep.deltaT, timeIndex, finalIteration, &gamgCache,
                needLoad ? &load : nullptr);
    // cyclicACMIFvPatch::movePoints, HERE and not later: fvMesh::movePoints runs the boundary's own
    // movePoints as part of the move, so it lands before fvc::meshPhi is read -- and under
    // CrankNicolson the off-centred mesh flux is built from it, so a scaling applied after that blend
    // would be blended out of the flux the step actually uses.
    //
    // It is two things (cyclicACMIFvPatch.C):
    //   1. updateAreas() -- the AMI re-run on the moved points, the mask re-applied, the face areas
    //      rescaled. cpu::cyclicACMI::setup IS that: it also rebuilds `g` from the scaled areas and
    //      `patches` from `g`, in place on the caller's objects. attachCyclicCoupling follows, as at
    //      the start, because rebuilding the patch list leaves a plain cyclic uncoupled and
    //      damBreakLeakage carries one beside the ACMI pair.
    //   2. THE MESH FLUX SCALED to the areas that just changed:
    //          the coupled face   phip *= magSf/geomArea, which scalePatchFaceAreas makes
    //                             max(tolerance, mask)
    //          the non-overlap    phip *= 1 - mask
    //      The swept volume dyn->update() just computed is the face's FULL geometric flux, so without
    //      this the mesh flux and the face area disagree by the mask and the discrete space
    //      conservation law breaks. MEASURED on damBreakLeakage shaken from t = 0.49: worst
    //      |div(phi)| 2.630e+01 and alpha reaching 1.89 without it, against OpenFOAM's
    //      1.00000000000037, and 1.5e-10 / 1.0000000000005 with it.
    //      OpenFOAM zeroes a coupled face whose AMI found no partner; this loop couples a COINCIDENT
    //      pair only (cpu::cyclicACMI::setup refuses any other), where every face has its twin, so
    //      that branch cannot be reached here.
    if (mutableMesh && mutableMesh->acmi && !mutableMesh->acmi->empty())
    {
        *mutableMesh->acmi = cpu::cyclicACMI::setup(*mutableMesh->m, *mutableMesh->g,
                                                   *mutableMesh->patches, time);
        attachCyclicCoupling(*mutableMesh->patches, *mutableMesh->m, *mutableMesh->g);
        SurfaceScalarField& mphi = dyn->meshPhiRef();
        for (const cpu::cyclicACMI::Side& s : mutableMesh->acmi->sides())
        {
            std::vector<scalar>& cpl = mphi.boundary[static_cast<std::size_t>(s.patch)];
            std::vector<scalar>& nov = mphi.boundary[static_cast<std::size_t>(s.nonOverlap)];
            for (std::size_t k = 0; k < s.scaledMask.size(); ++k)
            {
                cpl[k] *= std::max(cpu::cyclicACMI::tolerance, s.scaledMask[k]);
                nov[k] *= scalar(1) - s.scaledMask[k];
            }
        }
    }
    // ...and fvc::meshPhi(U) follows the move: the scheme's off-centred flux, which everything below
    // and the pressure corrector, the closure and CorrectPhi read through fvcMeshPhi()
    if (cn)
    {
        fv::meshPhi(*cn, f.cnMeshPhi0, dyn->meshPhi(), f.meshPhiPrev, f.meshPhiCN);
    }
    const SurfaceScalarField& meshPhiU = fvcMeshPhi(*dyn, f);
    // ...and fvMesh::movePoints moves the mesh objects with it: kOmegaSST's wall distance
    moveInterTurbulence(f.turbulence, m, g, patches, timeIndex);
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
    updateMovingWallVelocity(f, *dyn, meshPhiU, m, g, patches, rep.deltaT);
    // ...and a wave condition's model, whose FIRST update of the step is this one on a
    // moving mesh: OpenFOAM's log prints "Updating ... wave model" right after the
    // motion solve, ahead of pcorr and the alpha sub-cycles, so the absorber reads the
    // water level the last step left. The UEqn stage's update is then a no-op for this
    // time index, as OpenFOAM's is. Measured on waveMakerSolitary with pcorr converged:
    // updated in UEqn instead, from the alpha the sub-cycles left, U went from 7.1e-12 of
    // OpenFOAM after step one to 1.1e-07 after step two, the gap at the outlet.
    updateWaveVelocity(f.waves, f.alpha1, f.U, time, timeIndex, m, g, patches);
    // ...AND THE FLUX AND PHASE FRACTION THAT EVALUATE READS, as they stand HERE.
    // dynamicMotionSolverFvMesh::update ends in U.correctBoundaryConditions()
    // (dynamicMotionSolverFvMesh.C:101-114) and OpenFOAM's updateCoeffs LOOKS phi UP: at this instant
    // phi is the RELATIVE flux the previous step's pEqn left (fvc::makeRelative, pEqn.H:71), not the
    // ABSOLUTE one that step's last U evaluate read. brae's patches are told rather than looking up,
    // and this site used to evaluate them with the flux they were last handed -- the absolute one.
    // The two have OPPOSITE SIGNS on a wall that moves, and
    // pressurePermeableAlphaInletOutletVelocity's valueFraction is neg(phi). MEASURED on
    // laminar/testTubeMixer at step two: 346 of the 1050 wall faces held the extrapolated value where
    // OpenFOAM holds exactly zero, with refValue and valueFraction agreeing on every face -- the
    // stored value alone, from an evaluate made against the wrong flux. It reaches the momentum
    // through grad(U)'s boundary values as a 43 per cent error in divDevRhoReff's explicit term,
    // which is the only term of UEqn's source that is out (ddt agrees to 5.3e-14, div(rhoPhi,U)
    // contributes no source, the relaxation contribution to 1.3e-10).
    {
        const SurfaceScalarField* rhoPhi =
            (f.rhoPhi.boundary.size() == patches.size()) ? &f.rhoPhi : nullptr;
        for (std::size_t pi = 0; pi < patches.size() && pi < f.phi.boundary.size(); ++pi)
        {
            const std::string& fluxName = f.U.boundary[pi]->fluxName();
            if (fluxName != "rhoPhi" || rhoPhi)
            {
                f.U.boundary[pi]->updateFromFlux(
                    namedPatchFlux(fluxName, pi, patches[pi].name, f.phi, rhoPhi));
            }
            if (f.U.boundary[pi]->needsAlphaPatchValues())
            {
                f.U.boundary[pi]->updateFromAlphaValues(f.alpha1.boundary[pi]->value());
            }
        }
    }
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
    // which moves nHatf, and nHatf is what the alpha step compresses along.
    // THE AMR GATE'S "mapped flux left alone" CONTROL reaches here on a mesh that refines AND moves,
    // because on that mesh this block is the only one after a change (interAfterMeshChange's own is
    // skipped, see its motionFollows): no Sf & Uf rebuild, no pcorr and no makeRelative, the mapped flux
    // kept. The mixture and the curvature below still run -- the adapter carries neither through a change.
    const bool amrNoCorrectPhi = f.amr && f.amr->active
                              && std::getenv("BRAE_CONTROL_AMR_NO_CORRECTPHI") != nullptr;
    if (f.correctPhi)
    {
        if (!amrNoCorrectPhi)
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
            cin.meshPhi = &meshPhiU;
            cin.rhoPhi = &f.rhoPhi;
            cin.solveLog = &rep.pcorrSolves;
            correctPhi(f.U, f.phi, f.p_rgh, cin, cpc, m, g, patches);
            // correctPhi.H:11, #include "continuityErrs.H" on the absolute flux.
            // BRAE_CONTROL_NO_CORRECTPHI_CONTERR=1 leaves it uncounted, the writer's form before F2 -- the
            // write gate's arm P asserts that this moves the cumulative continuity error off OpenFOAM's.
            if (correctPhiDivOut && std::getenv("BRAE_CONTROL_NO_CORRECTPHI_CONTERR") == nullptr)
            {
                *correctPhiDivOut = fvc::div(f.phi, m, g, patches);
            }
            // fvc::makeRelative(phi, U)
            makeRelativeFlux(f.phi, meshPhiU);
            pushFluxToPatches(f, patches);
        }
        // mixture.correct(): calcNu, whose values alpha has not moved, then
        // interfaceProperties::correct() on the moved mesh
        cpu::twoPhase::mixtureMu(f.alpha1.internal, f.mixture.phases, f.mu);
        cpu::twoPhase::mixtureNu(f.alpha1.internal, f.mu, f.mixture.phases, f.nu);
        updateMixtureBoundary(f, patches);
        interfaceProps::calculateK(f.alpha1, f.interface, m, g, patches, false, f.nHatf, f.K);
    }
}


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
    const MutableMesh* mutableMesh,
    AlphaTaps* alphaTaps,
    InterWriter* writer)
{
    InterFields f = buildInterFields(caseDir, startDir, m, g, patches);
    if (writer)
    {
        registerUnwritten(*writer, f);
    }
    // THE CELL COUNT IS READ LIVE, NOT CAPTURED. `m` is the caller's mutable mesh, which an adaptive step
    // REPLACES in place -- so a cached count makes every per-cell reduction after a refinement run over
    // the pre-refinement cells only. MEASURED from OpenFOAM's own written phi on this case: the Courant
    // number over the whole 42,266-cell mesh is 0.08797028, which is OpenFOAM's logged 0.0879703, and over
    // the first 32,256 cells alone it is 0.04125969 -- which is exactly what brae reported. The refined
    // children are the smallest cells, where Co is largest, so a stale count hides them. With
    // `adjustTimeStep yes` that reading would go into setDeltaTVoF and pick a step twice too large.
    //
    // `nCAtStart` is kept for the two PRE-LOOP calls below only, which seed the CrankNicolson levels
    // against the mesh as read.
    const label nCAtStart = m.nCells();

    // A MESH THAT MOVES is attached to the caller's mutable objects -- the ones the fields were just
    // built against, checked by address -- and moved in place at every mesh update below.
    // AN ADAPTIVE MESH, built against the same objects the fields were: its own copy of the mesh is what
    // refineUpdate advances, and the caller's mutable mesh is what every field and patch reads.
    // BUILT BY buildInterFields, on both arms. A driver that built its own would be the only one that
    // had it -- which is exactly what happened to the device arm.
    if (!f.amr)
    {
        throw std::runtime_error(
            "brae interFoam: the case build handed this driver no InterAmr. buildInterFields makes one "
            "for every case (inactive on a static mesh), and a caller that skipped it would run an "
            "adaptive case as a static one.");
    }
    if (verbose)
    {
        std::printf("  adaptive mesh: %s\n", f.amr->active ? "yes" : "no");
    }
    if (f.amr->active && !mutableMesh)
    {
        throw std::runtime_error(
            "brae interFoam: the case asks for `dynamicFvMesh dynamicRefineFvMesh` and the driver was "
            "handed no mutable mesh. A refinement replaces the mesh, its geometry and its patches, and "
            "every field and patch field reads those -- running it on a copy they cannot see would solve a "
            "different problem at every step.");
    }

    DynamicMotionSolverFvMesh* dyn = f.dynamicMesh.get();
    if (dyn)
    {
        // THE PERMEABLE-WALL PAIR ON A MOVING MESH -- SOLVED, and the refusal that stood here is
        // gone. What it accused was never the pressure half: it was the ORDER of one evaluate.
        // dynamicMotionSolverFvMesh::update ends in U.correctBoundaryConditions()
        // (dynamicMotionSolverFvMesh.C:101-114), whose updateCoeffs LOOKS phi UP, and at that instant
        // phi is the RELATIVE flux the previous step's pEqn left. This loop evaluated the same patches
        // there with the flux they were last TOLD, which is the ABSOLUTE one its own
        // U.correctBoundaryConditions read inside pEqn -- and on a wall that moves the two have
        // opposite signs, while pressurePermeableAlphaInletOutletVelocity's valueFraction is neg(phi).
        // interMeshUpdate now hands U's patches the flux and the phase fraction as they stand before
        // that evaluate; see the note there for the measurement.
        //
        // HOW IT WAS FOUND, because six earlier readings pointed elsewhere. The chain was walked from
        // the outside in, each step measured against tools/dumpInterFoam at a MATCHED corrector (the
        // taps used to compare corrector 2 against corrector 1 and read HbyA 4.66e-01, which is what
        // sent three units after terms that were exact):
        //     rAU 3.1e-10, U as H() multiplies it 1.0e-13, UEqn upper/lower/diag 1.5e-09 -- so the
        //     matrix and psi were right and the SOURCE was not (3.8e-04)
        //     the source bisected by term: ddt 5.3e-14, div(rhoPhi,U) no source at all, the relaxation
        //     contribution 1.3e-10, divDevRhoReff 4.3e-01 -- one term left
        //     its explicit half bisected: muEff 1.1e-15, and grad(U) differing on 312 of 1250 cells
        //     U's PATCH VALUES at the assembly: 346 of 1050 faces holding the extrapolation where
        //     OpenFOAM holds exactly zero, with refValue AND valueFraction agreeing on every face --
        //     so the coefficients were right and the stored value was from the wrong evaluate
        //     U dumped at three points of the step: equal at the loop entry, different immediately
        //     after mesh.update()
        // MEASURED after the fix, laminar/testTubeMixer with the permeable pair, two steps:
        //     U at the end of the step   1.33e-01 -> 8.66e-11
        //     UEqn source                3.82e-04 -> 1.76e-10
        //     HbyA                       1.39e-03 -> 1.68e-10
        //     phiHbyA                    1.87e-03 -> 7.68e-11
        // The DEVICE loop still refuses it at its own site: it evaluates U's boundary in its own
        // kernels and nothing here has tested that half.
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
            // A cyclicACMI PAIR ON A MOVING MESH RUNS. The refusal that stood here read "OpenFOAM
            // then re-runs the AMI and rescales the mesh flux in cyclicACMIFvPatch::movePoints; that
            // is not ported" -- and naming both halves is what made the port short. interMeshUpdate
            // does them, in the order fvMesh::movePoints does (see the note there).
            //
            // WHAT HELD IT, three things, each measured rather than guessed:
            //   * `patches = buildPatches(...)` inside cpu::cyclicACMI::setup MOVE-ASSIGNED, stealing
            //     the returned vector's buffer and freeing the old one even at equal size. Every
            //     fvPatchField holds `const FvPatch&` into that vector, so the first
            //     U.evaluateBoundary() after the move read freed memory. gdb: SIGSEGV in
            //     PressureInletOutletVelocityPatchField<vector>::evaluate, the vector's data moving
            //     0xaaaaab7c9150 -> 0xaaaaab8d3590 at size 9 both sides. setup assigns element by
            //     element now.
            //   * the mesh move rebuilt the patch list with buildPatches(m, g) and NO mirrorACMI, so
            //     it threw the cyclicACMI refusal the start had legitimately passed -- with the mesh
            //     already moved. It carries the exemption now, keyed on the list already holding such
            //     a patch (dynamic_motion_solver_fv_mesh_cpp.cu).
            //   * THE MESH FLUX was never scaled to the areas the rescale had just changed, so the
            //     discrete space conservation law broke by exactly the mask: worst |div(phi)|
            //     2.630e+01 and alpha reaching 1.89 against OpenFOAM's 1.00000000000037.
            //
            // AND THE FIXTURE, which is half of why this took three units. The tutorial shaken from
            // t = 0 reads max|U| 7.3e-06 at ten steps and 2.7e-04 at a hundred -- nothing a velocity
            // comparison can discriminate. damBreakLeakage started at t = 0.49, where the baffle is
            // about to open, with the tank shaking, reaches max|U| 1.18. MEASURED there, 40 steps:
            // alpha 1.00000000000035 against OpenFOAM's 1.00000000000037 and worst |div(phi)|
            // 9.378e-11. Gated as the `moving` arm of tests/interfoam_leakage_vs_openfoam.sh.
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

    // THE CASE'S OWN CorrectPhi CONTROLS: pcorr and pcorrFinal, the laplacian's default, grad(pcorr)'s
    // entry and PIMPLE's non-orthogonal correctors
    const CorrectPhiControls cpc = correctPhiControlsOf(f, gamgCache);

    // initCorrectPhi.H, which runs for EVERY case, moving or not and with correctPhi or without: CorrectPhi
    // on the phi createFields built. It runs BEFORE the old-time copies below, so phi.oldTime() is the
    // corrected flux, and before the first Courant number reads it.
    //
    // ITS rAUf IS NOT ALWAYS 1, and this site used to pass 1 unconditionally. initCorrectPhi.H's two
    // branches are identical except for that argument: under `correctPhi` it is
    // `fvc::interpolate(rAU())` (correctPhi.H:6) where rAU is the READ_IF_PRESENT field, and only the
    // `else` branch passes a literal 1 (initCorrectPhi.H:28). On a COLD start rAU defaults to 1 and the
    // two coincide, which is why every gate agreed; on a RESTART rAU is the file's -- MEASURED on a
    // static damBreak with `correctPhi yes`, 9.9e-07 .. 1.0e-03 -- and passing 1 instead put brae's
    // p_rgh initial residuals 2.5e-04 from OpenFOAM's for the whole run, with every iteration count
    // still matching. Found by the rAU restart arm of tests/interfoam_dambreak_vs_openfoam.sh.
    {
        const SurfaceScalarField one = unitFaceField(m, patches);
        const SurfaceScalarField rAUfStart = f.correctPhi ? fvc::interpolate(f.rAU, m, g, patches) : one;
        CorrectPhiInput cin;
        cin.rAUf = &rAUfStart;
        cin.meshChanging = false;
        cin.rhoPhi = &f.rhoPhi;
        cin.solveLog = &rep.pcorrSolves;
        correctPhi(f.U, f.phi, f.p_rgh, cin, cpc, m, g, patches);
        pushFluxToPatches(f, patches);
        // ...and continuityErrs.H after it, in both of initCorrectPhi.H's branches (correctPhi.H:11 and
        // initCorrectPhi.H:31): the cumulative error OpenFOAM writes starts with this term, at the
        // deltaT runTime holds before setInitialDeltaT. Without it RAS/waterChannel's 0.3 wrote
        // -0.0023482777840429 where OpenFOAM writes -0.0023482774955524, 2.885e-10 apart -- exactly
        // this line's `global` in OpenFOAM's log.
        if (writer)
        {
            writer->addContinuityError(f.deltaT, fvc::div(f.phi, m, g, patches), g.V());
        }
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
    // CRANKNICOLSON's state. The scheme reads the OLD-OLD level of every operand -- U, rho and phi,
    // U's patch values too -- which GeometricField::oldTime().oldTime() creates as a copy of the old
    // level on the first step (nothing reads it then) and storeOldTimes rotates from the second. And
    // it keeps a ddt0 field per operand set (crank_nicolson_ddt_scheme_cpp.cuh): the momentum's, and
    // ddtCorr's two. Its clock is Time's: the step index, this step's deltaT and the previous step's.
    const bool cnDdt = (f.ddtU == DdtScheme::CrankNicolson);
    std::vector<vector> UOO = UOld;
    std::vector<std::vector<vector>> UOOBnd = UOldBnd;
    std::vector<scalar> rhoOO = rhoOld;
    // phi's old-old level is DIFFERENT. U's and rho's are asked for on the first step, unconditionally,
    // at the top of fvmDdt (`vf.oldTime().oldTime(); rho.oldTime().oldTime();`), so they exist as copies
    // of U^0 and rho^0 and rotate from the second step. phi.oldTime().oldTime() is asked for ONLY inside
    // fvcDdtPhiCorr's evaluate branch, which first runs on the second step -- and GeometricField::
    // oldTime() CREATES a level that does not exist as a copy of the one it hangs off, phi.oldTime(),
    // which by then is phi^1. So dphidt0's first estimate is rDtCoef0*(phi^1 - phi^1) = 0, and the
    // level rotates only from the third step. MEASURED with phi^0 in its place: U 2.3e-01 from
    // OpenFOAM after two steps, alpha 5e-15 -- the alpha step, which reads no old-old level, exact.
    SurfaceScalarField phiOO;
    bool phiOOExists = false;
    // ...and whether ANYTHING has asked for phi.oldTime() yet, which is what decides whether the level
    // exists (GeometricField::oldTime() creates it on the first request). On a static mesh ddtCorr
    // asks in step one's pEqn; on a moving mesh ddtCorr reads Uf.oldTime() and never asks, so the
    // alpha step's off-centred flux is the first request -- see the note at that line.
    bool phiOldRequested = false;
    scalar deltaTPrev = f.deltaT;   // Time::deltaT0_ starts equal to deltaT_ (Time.C, the constructor)
    fv::CrankNicolsonClock cnClock;
    cnClock.ocCoeff = f.ddtOcCoeff;
    fv::CrankNicolsonDdt0<vector> cnDdt0RhoU;
    cnDdt0RhoU.name = "ddt0(rho,U)";
    fv::CrankNicolsonDdt0<vector> cnDdtCorrU;
    cnDdtCorrU.name = "ddtCorrDdt0(U)";
    fv::CrankNicolsonDdt0<scalar> cnDdtCorrPhi;
    cnDdtCorrPhi.name = "ddtCorrDdt0(phi)";
    // alphaEqn.H:236-262 under a non-Euler ddt(rho,U): the MULES flux is un-blended at the end of the
    // step with alphaPhi10.oldTime(), and alphaPhi10 is a field whose old time is CREATED, as a copy
    // of the current value, by that very first call (GeometricField::oldTime() on a field that has
    // never been asked for one) -- on the first step the blend is on, the un-blend divides a flux by
    // itself; storeOldTimes rotates it from the next step on. `alphaPhi10End` is the flux the last
    // pass left, which is what the first access of a new step stores.
    SurfaceScalarField alphaPhi10End;
    SurfaceScalarField alphaPhi10Old;
    bool alphaPhi10OldExists = false;
    label alphaPhi10OldIndex = -1;
    // ...and a fifth on a moving mesh: Uf.oldTime(), which ddtCorr reads in phi.oldTime()'s place
    SurfaceVectorField UfOld = f.Uf;
    // ...and under CrankNicolson a SIXTH: Uf.oldTime().oldTime(), which fvcDdtUfCorr's second ddt0
    // field is built from (CrankNicolsonDdtScheme.C:1230-1235). Created as a copy of Uf.oldTime() the
    // first time it is asked for, exactly as phiOO is -- GeometricField::oldTime() on a level that has
    // never been stored returns the current one.
    SurfaceVectorField UfOO = f.Uf;
    bool UfOOExists = false;
    fv::CrankNicolsonDdt0<vector> cnDdtCorrUf;
    cnDdtCorrUf.name = "ddtCorrDdt0(Uf)";
    // A RESTART from a directory OpenFOAM wrote under CrankNicolson: each ddt0 field OpenFOAM would
    // find there, with startTimeIndex -2 so the scheme is WARM on the first step rather than Euler for
    // one step and Euler-estimated for the next. inter_cn_restart.cuh carries the two facts; the
    // objects are the driver's, so the seeding is. `ddtCorrDdt0(Uf)` is not among them -- a moving mesh
    // is refused at the case reader for want of a fixture that could witness a seed.
    seedCnDdt0(cnDdt0RhoU, f.cnRestart, static_cast<std::size_t>(nCAtStart), patches);
    seedCnDdt0(cnDdtCorrU, f.cnRestart, static_cast<std::size_t>(nCAtStart), patches);
    seedCnDdt0(cnDdtCorrPhi, f.cnRestart, static_cast<std::size_t>(m.nInternalFaces()), patches);
    // ...AND THE OLD-OLD LEVELS, which the scheme reads and which OpenFOAM reads back off disk as
    // <field>_0 (inter_cn_restart.cuh): the level the restart directory holds is the one that rotates
    // into oldTime().oldTime() on the first step, while oldTime() becomes the field at the start time --
    // the value already in UOld/phiOld here. Without them a restart's first ddt0 estimate is
    // rDtCoef0*(x - x) = 0 where OpenFOAM's is a real difference. rho_0 and alpha.water_0 are never
    // written, so those two levels stay the copies OpenFOAM starts them as.
    readCnOldOld(f.cnRestart, "U_0", static_cast<std::size_t>(nCAtStart), patches, UOO, UOOBnd);
    phiOOExists = readCnOldOldSurface(f.cnRestart, "phi_0", m.nInternalFaces(), patches, phiOO);
    // ...and phi.oldTime() then EXISTS from the first step, so alphaEqn's blend reads the level rather
    // than the flux beside it. NUMERICALLY A NO-OP HERE, measured: with this line removed the restart
    // gate reads the same digits on both arms, because the first step's storeOldTimes rotates the level
    // to phi at the start time before the blend and that is the flux beside it. Kept because it is the
    // state OpenFOAM is in, and because a case that wrote phi between the two would part here.
    if (phiOOExists) phiOldRequested = true;

    SurfaceScalarField prevCorr;                 // alphaApplyPrevCorr's cache
    // cyclicACMIPolyPatch::updateAreas runs once per time index (prevTimeIndex_)
    bool acmiRescaledThisStep = false;
    GamgSolveLog gamgLog;
    rep.deltaT = f.deltaT;

    // A GATE'S CONTROL, never set by a solver: the named localEuler consumers -- `alpha` (the pre-solve's ddt
    // and MULES), `ueqn` (fvm::ddt(rho, U)), `ddtcorr`, `turbulence` (kOmegaSST's fvm::ddt(omega) and
    // fvm::ddt(k)) -- read the global 1/deltaT in the local rDeltaT's place, which is what a port that left
    // them on the Euler form runs. WRONG; tests/interfoam_dtchull_vs_openfoam.sh asserts that each one
    // fails. See readLtsScalarControl.
    const std::set<std::string> ltsScalarControl = readLtsScalarControl();
    std::vector<scalar> rDeltaTGlobal;
    if (!ltsScalarControl.empty())
    {
        rDeltaTGlobal.assign(static_cast<std::size_t>(m.nCells()), scalar(1)/f.deltaT);
    }
    // GATE CONTROLS for outletPhaseMeanVelocity, never set by a solver: FROZEN keeps the condition as read
    // (no updateCoeffs at all), IGNORE_LAG updates it in the first corrector too. Both make the answer WRONG.
    const bool opmvFrozen = std::getenv("BRAE_CONTROL_OPMV_FROZEN") != nullptr;
    const bool opmvIgnoreLag = std::getenv("BRAE_CONTROL_OPMV_NOLAG") != nullptr;
    if (opmvFrozen)
        std::printf("  *** CONTROL MODE: outletPhaseMeanVelocity is never updated. This run is deliberately "
                    "wrong. ***\n");
    if (opmvIgnoreLag)
        std::printf("  *** CONTROL MODE: outletPhaseMeanVelocity is updated in the lagged first corrector. "
                    "This run is deliberately wrong. ***\n");
    // GATE CONTROL for fvSolution's `cache { grad(U); }`, never set by a solver: every grad(U) site forms
    // its own from U as it stands, which is OpenFOAM's UNCACHED answer. WRONG where the case caches it.
    const bool gradUUncached = std::getenv("BRAE_CONTROL_GRADU_UNCACHED") != nullptr;
    if (gradUUncached)
        std::printf("  *** CONTROL MODE: fvSolution's cached grad(U) is formed afresh at every site. This run is "
                    "deliberately wrong. ***\n");
    // gradScheme::grad bypasses the registry while mesh().changing() and deletes what it holds
    // (gradScheme.C:132-142). A MOTION-SOLVER mesh is moving() from its first update on (polyMesh's flag is
    // never reset), so from step one every request is formed afresh: the per-step bypass below.
    // RAS/electrostaticDeposition ships the cache on a solid-body motion (interfoam_moving_vs_openfoam
    // `esd`). A REFINING mesh changes on some steps only; the step after one that did not is what the stale
    // refusal at the assembly would meet, but the refiner's own reset of the flag is not modelled here.
    if (f.gradUCache.on && f.amr && f.amr->active)
        throw std::runtime_error(
            "brae interFoam: fvSolution caches grad(U) on a refining mesh. OpenFOAM bypasses and deletes the "
            "cached field on the steps the mesh changes (gradScheme.C:132-142) and reuses it on the others; "
            "that is not ported.");
    auto rDeltaTFor = [&](const char* consumer) -> const std::vector<scalar>*
    {
        if (!f.lts) return nullptr;
        return ltsScalarControl.count(consumer) ? &rDeltaTGlobal : &f.rDeltaT;
    };

    // Time::writeTime_ for the step being taken (Time.C:1103-1130), decided when the time advances, and the
    // alpha flux the step's LAST alpha solve leaves, kept only on a write step: OpenFOAM writes
    // alphaPhi0.<phase1> from alphaPhi10 (createAlphaFluxes.H), which that solve overwrites
    bool writeNow = false;
    SurfaceScalarField alphaPhi10Write;
    // alpha.<phase1>_0 of a sub-cycled alpha: alpha as the step found it, taken only on a write step
    std::vector<scalar> alphaOldWrite;
    std::vector<std::vector<scalar>> alphaOldBndWrite;
    // ...and its patch values as the step's last sub-cycle began (InterWriteState::alpha1SubCycleBoundary)
    std::vector<std::vector<scalar>> alphaSubBndWrite;

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
                        m.nCells(), m.nInternalFaces());
                    rep.CoNum = courantNo(sumPhi, g.V(), rep.deltaT).CoNum;
                    // alphaCourantNo needs the same sumPhi, so it is cached on the report rather than
                    // recomputed -- OpenFOAM's two #includes build it twice, which is the one place
                    // this driver deliberately differs and it changes no number.
                    rep.alphaCoNum = alphaCourantNo(sumPhi, f.alpha1.internal, g.V(), rep.deltaT).CoNum;
                    break;
                }
                case Stage::alphaCourantNo:  break;    // computed above, from the same sumPhi
                case Stage::setRDeltaT:
                {
                    // setRDeltaT.H, before ++runTime: rhoPhi is the last alpha step's (or createFields'
                    // at the first step), phi the last pressure corrector's, rho the last mixture's.
                    // Damping from the third step of this run: timeIndex > startTimeIndex + 1 tested
                    // before the time advances, and rep.steps counts the steps this run has completed.
                    SetRDeltaTInput ri;
                    ri.rhoPhi = &f.rhoPhi;
                    ri.phi = &f.phi;
                    ri.alpha1 = &f.alpha1;
                    ri.rho = &f.rho;
                    ri.damp = rep.steps > 1;
                    LocalEulerControls lec = f.ltsCtl;
                    applySetRDeltaTControls(lec, ri.damp);
                    const SetRDeltaTReport lr = setRDeltaT(f.rDeltaT, lec, ri, m, g, patches);
                    rep.ltsLog.push_back(lr);
                    rep.rDeltaTPerStep.push_back(f.rDeltaT);
                    if (verbose)
                    {
                        std::printf("  Flow time scale min/max = %.17g, %.17g\n",
                                    (double)lr.flowMin, (double)lr.flowMax);
                        std::printf("  Smoothed flow time scale min/max = %.17g, %.17g\n",
                                    (double)lr.smoothedMin, (double)lr.smoothedMax);
                        if (lr.damped)
                        {
                            std::printf("  Damped flow time scale min/max = %.17g, %.17g\n",
                                        (double)lr.dampedMin, (double)lr.dampedMax);
                        }
                    }
                    break;
                }
                case Stage::setDeltaT:
                    // Time::adjustDeltaT measures from the start, value() - startTime_ (Time.C:1150)
                    rep.deltaT = setDeltaTVoF(rep.deltaT, rep.CoNum, rep.alphaCoNum, f.timeCtl,
                                              rep.time - startTime, &f.writeCadence);
                    break;
                case Stage::advanceTime:
                    rep.time += rep.deltaT;
                    ++rep.steps;
                    // Time::operator++: deltaT0_ = deltaTSave_; deltaTSave_ = deltaT_ -- the step
                    // before this one, and this one, as the CrankNicolson coefficients read them
                    cnClock.timeIndex = rep.steps;
                    cnClock.deltaT = rep.deltaT;
                    cnClock.deltaT0 = deltaTPrev;
                    acmiRescaledThisStep = false;
                    outerOfStep = -1;
                    // Time::operator++ moves writeTimeIndex_ AFTER the time, with the step that took
                    // it -- which is what the next adjustDeltaT measures the distance to -- and whether it
                    // moved is the write time under runTime/adjustableRunTime
                    {
                        const bool indexMoved = f.writeCadence.advance(rep.time - startTime, rep.deltaT);
                        writeNow = false;
                        if (writer)
                        {
                            writer->stepTaken(rep.deltaT);
                            writeNow = writer->isWriteTime(writer->startTimeIndex() + rep.steps, indexMoved);
                        }
                        if (writeNow && writer->writesAlphaOld())
                        {
                            alphaOldWrite = f.alpha1.internal;
                            alphaOldBndWrite.assign(f.alpha1.boundary.size(), std::vector<scalar>());
                            for (std::size_t pi = 0; pi < f.alpha1.boundary.size(); ++pi)
                            {
                                alphaOldBndWrite[pi] = f.alpha1.boundary[pi]->value();
                            }
                        }
                    }
                    break;

                case Stage::meshUpdate:
                {
                    ++outerOfStep;
                    // A MESH THAT REFINES AND MOVES (laminar/oscillatingBox): dynamicRefineFvMesh::update
                    // changes the topology FIRST and moves the points AFTER (dynamicRefineFvMesh.C:1468-1474),
                    // and interFoam.C:112-148 then runs one block on the moved mesh. So the move waits for
                    // the change here; a mesh that only moves, or only refines, is unchanged.
                    const bool refineAndMove = dyn && f.amr && f.amr->active;
                    // interFoam.C:112-149, one copy shared with the device loop: see interMeshUpdate.
                    if (!refineAndMove)
                    {
                        std::vector<scalar> cpDiv;
                        interMeshUpdate(dyn, f, m, g, patches, mutableMesh, amiPairs, gamgCache, cpc,
                                        rep, rep.time, rep.steps, outerOfStep, lc.nOuterCorrectors,
                                        cnDdt ? &cnClock : nullptr, writer ? &cpDiv : nullptr);
                        if (writer && !cpDiv.empty())
                        {
                            writer->addContinuityError(rep.deltaT, cpDiv, g.V());
                        }
                    }
                    // ...and the ADAPTIVE mesh, which is the same line of interFoam.C for a different
                    // dynamicFvMesh: mesh.update() selects, refines, unrefines and maps, and everything
                    // the solver rebuilds afterwards is what `changed` gates. Only on the FIRST outer
                    // corrector unless the case asks otherwise, exactly as the motion branch is.
                    // outerOfStep is ZERO on the first corrector of a step: it starts at -1 and is reset
                    // to -1 at advanceTime, and this stage increments it. Comparing it against 1 ran the
                    // adaptive branch on no corrector at all -- the case ran to completion, printed
                    // nothing, and refined nothing.
                    if (f.amr && f.amr->active
                     && (outerOfStep == 0 || f.moveMeshOuterCorrectors))
                    {
                        if (refineAndMove)
                        {
                            // hexRef8 lays its new points out on the LIVE mesh (hexRef8.C:3362, :3462,
                            // :3650); the adapter's copy was last written at the previous change, so it is
                            // given the points the motion has moved it to since. Without this the change
                            // would hand the mesh back at its old position.
                            f.amr->state.m.movePoints(mutableMesh->m->points());
                            // ...and the motion's points0, which each change maps (RefineUpdateState::points0)
                            f.amr->state.points0 = &dyn->points0Ref();
                        }
                        // the old-time levels the ddt terms read, which OpenFOAM maps as registered
                        // fields and brae keeps as locals of this loop -- see InterAmrOldTime
                        InterAmrOldTime oldT;
                        oldT.alphaOld = &alphaOld;
                        oldT.UOld = &UOld;
                        oldT.UOldBnd = &UOldBnd;
                        oldT.rhoOld = &rhoOld;
                        oldT.UOO = &UOO;
                        oldT.UOOBnd = &UOOBnd;
                        oldT.rhoOO = &rhoOO;
                        oldT.phiOld = &phiOld;
                        oldT.UfOld = &UfOld;
                        // ...and the CrankNicolson levels, which are FIELDS in OpenFOAM's registry and
                        // are autoMapped with everything else. Only the ones that exist are carried; on
                        // an Euler case every `exists` is false and nothing rides.
                        InterAmrCn cnState;
                        if (cnDdt)
                        {
                            cnState.ddt0RhoU = &cnDdt0RhoU;
                            cnState.ddtCorrU = &cnDdtCorrU;
                            cnState.ddtCorrUf = &cnDdtCorrUf;
                            cnState.ddtCorrPhi = &cnDdtCorrPhi;
                            cnState.UfOO = &UfOO;
                            cnState.phiOO = &phiOO;
                            cnState.alphaPhiEnd = &alphaPhi10End;
                            cnState.alphaPhiOld = &alphaPhi10Old;
                        }
                        if (cnDdt && std::getenv("BRAE_AMR_TRACE"))
                        {
                            const auto amx = [](const std::vector<scalar>& v)
                            { scalar m = 0; for (scalar x : v) m = std::fmax(m, std::fabs(x)); return m; };
                            const auto amxv = [](const std::vector<vector>& v)
                            { scalar m = 0; for (const vector& x : v) m = std::fmax(m, mag(x)); return m; };
                            std::printf("  TRACE host   step %ld: ddt0RhoU %.9g/%d/%ld ddtCorrU %.9g "
                                        "ddtCorrUf %.9g alphaOO %.9g UOO %.9g phiOO %.9g UfOO %.9g "
                                        "aPhiEnd %.9g aPhiOld %.9g aOld %.9g UOld %.9g phiOld %.9g "
                                        "UfOld %.9g\n",
                                        (long)rep.steps, (double)amxv(cnDdt0RhoU.internal),
                                        (int)cnDdt0RhoU.exists, (long)cnDdt0RhoU.startTimeIndex,
                                        (double)amxv(cnDdtCorrU.internal),
                                        (double)amxv(cnDdtCorrUf.internal), (double)0.0,
                                        (double)amxv(UOO), (double)amx(phiOO.internal),
                                        (double)amxv(UfOO.internal), (double)amx(alphaPhi10End.internal),
                                        (double)amx(alphaPhi10Old.internal), (double)amx(alphaOld),
                                        (double)amxv(UOld), (double)amx(phiOld.internal),
                                        (double)amxv(UfOld.internal));
                        }
                        const bool changed =
                            interAmrUpdate(*f.amr, f, *mutableMesh, rep.steps, oldT, cnState);
                        if (changed)
                        {
                            // interFoam.C:118-123, FIRST of everything the change triggers: the previous
                            // step's MULES correction flux is DROPPED when the topology changed --
                            // talphaPhi1Corr0.clear(). It is a flux on faces that no longer exist.
                            // Empty is how alphaEqnStep is told there is none (alpha_eqn_cpp.cu:552).
                            prevCorr.internal.clear();
                            prevCorr.boundary.clear();
                            // fvMesh::updateMesh as the motion sees it: the change's V0, the mesh flux
                            // recreated on the new faces (see DynamicMotionSolverFvMesh::topoChanged)
                            if (refineAndMove)
                            {
                                dyn->topoChanged(f.amr->state.V0, rep.steps);
                            }
                            interAfterMeshChange(f, *mutableMesh, gamgCache, cpc, rep, rep.steps,
                                                 refineAndMove);
                        }
                        // OpenFOAM prints "Refined from N to M cells." at every change; this is the same
                        // line, and a run that silently refines nothing is what it exists to show.
                        if (verbose)
                        {
                            std::printf("    mesh: %d cells refined, %d split points unrefined, "
                                        "now %d cells\n", (int)f.amr->nRefined, (int)f.amr->nUnrefined,
                                        (int)mutableMesh->m->nCells());
                        }
                    }
                    // ...and THEN the move, with the block interFoam.C:112-148 runs after mesh.update():
                    // the moving walls, gh/ghf, `phi = Sf & Uf`, CorrectPhi with the mesh flux,
                    // makeRelative and the mixture -- once, on the moved mesh
                    if (refineAndMove)
                    {
                        std::vector<scalar> cpDiv;
                        interMeshUpdate(dyn, f, m, g, patches, mutableMesh, amiPairs, gamgCache, cpc,
                                        rep, rep.time, rep.steps, outerOfStep, lc.nOuterCorrectors,
                                        cnDdt ? &cnClock : nullptr, writer ? &cpDiv : nullptr);
                        if (writer && !cpDiv.empty())
                        {
                            writer->addContinuityError(rep.deltaT, cpDiv, g.V());
                        }
                    }
                    break;
                }

                case Stage::alphaControls:             // read once, in buildInterFields
                    ++outerIndex;
                    break;

                case Stage::alphaEqnSubCycle:
                {
                    // subCycle.H:78: the run's first alpha1.oldTime() creates the old level here, and it
                    // keeps the valueFractions and contact-angle gradients alpha has now (idempotent)
                    if (writer && writer->writesAlphaOld())
                    {
                        writer->noteAlphaOldCreation(f.alpha1);
                    }
                    AlphaStepInput ai;
                    // alphaEqn.H:18-56: the off-centring coefficient the scheme constructed for
                    // ddt(alpha) gives -- 0 for Euler, and 0 on the first step of a cold start under
                    // CrankNicolson (timeIndex > startTimeIndex + 1 is false), the scheme's own after
                    // -- and the blended flux phiCN = cnCoeff*phi + (1 - cnCoeff)*phi.oldTime(), which
                    // IS phi when ocCoeff is 0
                    // ...and alphaRestart ORed into the warm-up test (alphaEqn.H:36-45): a start
                    // directory that holds alphaPhi0 off-centres from the FIRST step. Only the file's
                    // presence reaches the answer -- see fact (2) in inter_cn_restart.cuh.
                    const scalar ocAlpha = offCentringCoeff(f.ddtAlpha, f.alphaCtl.nAlphaSubCycles,
                                                            f.ddtAlphaOcCoeff,
                                                            f.cnAlphaRestart || rep.steps > 1);
                    const scalar cnAlpha = blendingCoeff(ocAlpha);
                    SurfaceScalarField phiCN;
                    // WHICH phi.oldTime() -- and whether there IS one yet. GeometricField::oldTime()
                    // CREATES the level on the first request, as a copy of the field as it stands
                    // then (GeometricField.C:940-980), and who requests it first depends on the mesh:
                    //   * a STATIC mesh: ddtCorr is fvcDdtPhiCorr and asks for phi.oldTime() in step
                    //     one's pEqn, so by this line in step two the level is step one's flux;
                    //   * a MOVING mesh: ddtCorr is fvcDdtUfCorr and reads Uf.oldTime() INSTEAD
                    //     (fvcDdt.C, the three-argument ddtCorr picks by mesh.dynamic()), so nothing
                    //     has asked -- and THIS line is the first request. The level is then a copy
                    //     of the flux beside it and the blend is inert for that one step.
                    // Blending with the previous step's flux there is a different equation: measured
                    // on waves/waveMakerSolitary under `CrankNicolson 0.9` with correctPhi on, brae's
                    // phiCN was 1.17 RELATIVE from OpenFOAM's at step two (whose phiCN is its phi to
                    // 2.1e-22), and that carried into alpha 4.7e-05, U 1.2e-03 at step two and 1.06 at
                    // thirty. It is invisible without correctPhi, where the flux this line sees IS
                    // last step's, and invisible under Euler, where ocAlpha is 0.
                    const SurfaceScalarField& phiForBlend = phiOldRequested ? phiOld : f.phi;
                    offCentredFlux(f.phi, phiForBlend, cnAlpha, ocAlpha, phiCN);
                    if (ocAlpha > scalar(0)) phiOldRequested = true;
                    ai.phi = &f.phi; ai.phiCN = &phiCN;
                    ai.cAlpha = f.interface.cAlpha;
                    ai.nAlphaCorr = f.alphaCtl.nAlphaCorr;
                    ai.icAlpha = f.alphaCtl.icAlpha;
                    ai.scAlpha = f.alphaCtl.scAlpha;
                    ai.rho1 = f.mixture.phases.rho1; ai.rho2 = f.mixture.phases.rho2;
                    ai.alphaScheme  = f.divPhiAlpha;
                    ai.alpharScheme = f.divPhirbAlpha;
                    ai.MULESCorr = f.alphaCtl.MULESCorr;
                    // localEuler: the pre-solve's ddt and both MULES::correct calls take the local rDeltaT
                    ai.rDeltaT = rDeltaTFor("alpha");
                    ai.alphaApplyPrevCorr = f.alphaCtl.alphaApplyPrevCorr;
                    ai.alpha2BndOut = &f.alpha2Bnd;
                    // the case's own tolerances, where a struct default of 1e-8 used to stand
                    ai.tolAlpha = f.aSolve.tol;
                    ai.relTolAlpha = f.aSolve.relTol;
                    ai.maxIterAlpha = f.aSolve.maxIter;
                    ai.minIterAlpha = f.aSolve.minIter;
                    ai.gradAlpha1 = f.gradAlpha1;
                    ai.gradAlpha2 = f.gradAlpha2;
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
                        // the last sub-cycle's storeOldTime (GeometricField.C:932) copies alpha's patches
                        // into the old level as they stand now, before this sub-cycle updates any
                        if (writeNow && writer && writer->writesAlphaOld()
                            && subCycle == f.alphaCtl.nAlphaSubCycles)
                        {
                            alphaSubBndWrite.assign(f.alpha1.boundary.size(), std::vector<scalar>());
                            for (std::size_t pi = 0; pi < f.alpha1.boundary.size(); ++pi)
                            {
                                alphaSubBndWrite[pi] = f.alpha1.boundary[pi]->value();
                            }
                        }
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
                        sub.taps = alphaTaps;
                        alphaEqnStep(f.alpha1, aOld, sub, f.interface, f.mulesCtl,
                                     m, g, patches, aPhi, rPhi, f.nHatf, f.K, &prevCorr);
                        // alphaEqn.H:236-262: with ddt(rho,U) neither Euler nor localEuler the
                        // end-of-step alpha flux is un-blended -- alphaPhi10 = (alphaPhi10 - (1 -
                        // cnCoeff)*alphaPhi10.oldTime())/cnCoeff when ocCoeff > 0 -- and rhoPhi takes
                        // phi, not phiCN, beside rho2. alphaEqnStep formed the Euler branch's rhoPhi;
                        // this is the other branch, on top of it.
                        if (cnDdt)
                        {
                            if (ocAlpha > scalar(0))
                            {
                                if (alphaPhi10OldIndex != rep.steps)
                                {
                                    // the step's first oldTime(): storeOldTimes on a field that has
                                    // one, or the creating copy on a field that has none
                                    alphaPhi10Old = alphaPhi10OldExists ? alphaPhi10End : aPhi;
                                    alphaPhi10OldExists = true;
                                    alphaPhi10OldIndex = rep.steps;
                                }
                                const scalar oneMinusCn = scalar(1) - cnAlpha;
                                for (std::size_t q = 0; q < aPhi.internal.size(); ++q)
                                {
                                    aPhi.internal[q] = (aPhi.internal[q] - oneMinusCn*alphaPhi10Old.internal[q])/cnAlpha;
                                }
                                for (std::size_t pi = 0; pi < aPhi.boundary.size(); ++pi)
                                {
                                    for (std::size_t q = 0; q < aPhi.boundary[pi].size(); ++q)
                                    {
                                        aPhi.boundary[pi][q] = (aPhi.boundary[pi][q]
                                                              - oneMinusCn*alphaPhi10Old.boundary[pi][q])/cnAlpha;
                                    }
                                }
                            }
                            massFlux(aPhi, f.phi, f.mixture.phases.rho1, f.mixture.phases.rho2, rPhi);
                            alphaPhi10End = aPhi;
                        }
                        // a copy, and only on a write step -- the last call of the step wins, as the last
                        // assignment to alphaPhi10 does in OpenFOAM
                        if (writeNow)
                        {
                            alphaPhi10Write = aPhi;
                        }
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
                    // ...and NOT the boundary. OpenFOAM's alphaEqnSubCycle.H ends at
                    // `rho == alpha1*rho1 + alpha2*rho2` and evaluates alpha nowhere: the patch values
                    // it leaves are whichever of the two things alphaEqn.H last did to them -- MULES's
                    // own trailing correctBoundaryConditions (CMULESTemplates.C, mirrored at
                    // mules_cpp.cu:750) when the relaxation did not run, and the relaxation's
                    // ASSIGNMENT when it did. Evaluating here overwrote the second case, which made
                    // the faithful assignment in alpha_eqn_cpp.cu inert: MEASURED on
                    // RAS/electrostaticDeposition at step two, brae's patch value sat exactly on
                    // OpenFOAM's own owner cell, 5.1256e-10 from the value OpenFOAM wrote, while the
                    // CELLS agreed to 4.6e-16.
                    // Dropping this was tried before and rejected, but BUNDLED with the reset in
                    // alphaEqnStep and before the relaxation assigned: that experiment does not bear on
                    // this line alone.
                    // ...and the conditions that look alpha up read the values THIS evaluate left
                    pushAlphaToPatches(f, patches);
                    break;
                }

                case Stage::mixtureCorrect:
                {
                    // rho == alpha1*rho1 + alpha2*rho2 (alphaEqnSubCycle.H:36), and the viscosities
                    // with it. rho.oldTime() is NOT touched: fvm::ddt's source needs the value from
                    // the start of the step, and this is where it would be lost.
                    f.alpha2.resize(static_cast<std::size_t>(m.nCells()));
                    for (label c = 0; c < m.nCells(); ++c)
                    {
                        f.alpha2[c] = scalar(1) - f.alpha1.internal[c];
                    }
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
                    // Instrument: BRAE_STAGE_DUMP_DIR=<dir> (+ BRAE_STAGE_DUMP_ITER=n, default 1) writes
                    // what the alpha step leaves for the momentum, at the nth step, under the names
                    // tools/dumpInterFoam writes OpenFOAM's -- nHatfA, sigmaKA (as K), alphaPostMULES.
                    // The pEqn's surface-tension force is built from THIS curvature and not the one the
                    // mesh-change block left, so the two have to be compared separately.
                    if (const char* dd = std::getenv("BRAE_STAGE_DUMP_DIR"))
                    {
                        const char* it = std::getenv("BRAE_STAGE_DUMP_ITER");
                        if (rep.steps == (it && *it ? std::atoi(it) : 1))
                        {
                            std::error_code ec;
                            std::filesystem::create_directories(dd, ec);
                            const std::string dir(dd);
                            const auto wS = [&dir](const char* n, const std::vector<scalar>& v)
                            {
                                std::ofstream o(dir + "/" + n);
                                o.precision(17);
                                for (const scalar x : v) o << x << "\n";
                            };
                            wS("nHatfA", f.nHatf.internal);
                            wS("KA", f.K);
                            wS("alphaPostMULES", f.alpha1.internal);
                            wS("rhoAfterAlpha", f.rho);
                        }
                    }
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
                    // ...and WHICH delta coefficients, which `uncorrected` and `limited 0` do not
                    // share with `orthogonal` (uncorrectedSnGrad.H:113-119)
                    const bool snNonOrth = f.snGradScheme.nonOrthCoeffs;
                    // each correction takes grad(<field>)'s own entry (correctedSnGrad.C:52-55)
                    const SurfaceScalarField snRho = fvc::snGrad(rhoF, m, g, patches, snCorr, f.gradRho.leastSquares,
                                                                 f.gradRho.cellLimitK, snLim, snNonOrth);
                    const SurfaceScalarField snA = fvc::snGrad(f.alpha1, m, g, patches, snCorr,
                                                               f.gradAlpha1.leastSquares, f.gradAlpha1.cellLimitK, snLim,
                                                               snNonOrth);

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
                    const SurfaceScalarField snP = fvc::snGrad(f.p_rgh, m, g, patches, snCorr, f.gradPrgh.leastSquares,
                                                               f.gradPrgh.cellLimitK, snLim, snNonOrth);

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
                    // the case's ddt scheme -- read into f.ddtU and, until CrankNicolson was ported,
                    // never handed on (the input's default is Euler)
                    mi.ddtScheme = f.ddtU;
                    if (cnDdt)
                    {
                        mi.cn = &cnClock;
                        mi.cnDdt0 = &cnDdt0RhoU;
                        mi.rhoOO = &rhoOO;
                        mi.UOO = &UOO;
                    }
                    mi.V0 = dyn ? &dyn->V0() : nullptr;
                    // ...and V00 only under CrankNicolson, which is the only scheme that reads it:
                    // asking for it creates the level (fvMesh::V00() is lazy), and a mesh that never
                    // needs it should not have one.
                    mi.V00 = (dyn && cnDdt) ? &dyn->V00() : nullptr;
                    // nuEff = nut + nu: the mixture's nu alone when the case is laminar
                    std::vector<scalar> nuEff;
                    std::vector<std::vector<scalar>> nuEffB;
                    interNuEff(f.turbulence, f.nu, nuB, nuEff, nuEffB);
                    mi.nuEff = &nuEff;
                    mi.nuEffBnd = &nuEffB;
                    mi.deltaT = rep.deltaT;
                    mi.rDeltaT = rDeltaTFor("ueqn");
                    mi.scheme = f.divRhoPhiU;
                    mi.schemeCoeff = f.divRhoPhiUCoeff;
                    mi.relaxEquationU = f.relaxEquationU; mi.relaxU = f.relaxU;
                    // laplacianSchemes' default, for the viscous term's fvm::laplacian(rho*nuEff, U)
                    mi.correctedLaplacian = f.laplacianScheme.corrected;
                    mi.nonOrthCoeffs = f.laplacianScheme.nonOrthCoeffs;
                    mi.snGradLimitCoeff = f.laplacianScheme.limitCoeff;
                    // gradSchemes' grad(U): cellLimited or not, for linearUpwind and the viscous term
                    mi.gradULimitK = f.gradULimitK;
                    mi.gradULeastSq = f.gradULeastSq;

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
                        msc.minIterU = us.minIter;
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
                        // ...and an outletPhaseMeanVelocity, from the phase field's stored patch values and
                        // U's face cells as they stand at the assembly (the step's starting U)
                        if (f.U.boundary[pi]->isOutletPhaseMeanVelocity() && !opmvFrozen)
                        {
                            if (f.U.boundary[pi]->alphaFieldName() != f.alphaName)
                                throw std::runtime_error(
                                    "brae interFoam: U patch `" + patches[pi].name + "` is an "
                                    "outletPhaseMeanVelocity naming `alpha " + f.U.boundary[pi]->alphaFieldName()
                                    + "`, and this case's phase field is `" + f.alphaName + "`. OpenFOAM looks "
                                    "the named field up and stops without it.");
                            f.U.boundary[pi]->updatePhaseMean(f.alpha1.boundary[pi]->value(), f.U.internal,
                                                              g.Sf(), g.magSf());
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
                        // ...through U.boundaryFieldRef(), which moves U's eventNo (GeometricField.C:
                        // boundaryFieldRef -> setUpToDate), so a cached grad(U) is out of date from here
                        f.gradUCache.valid = false;
                    }
                    // fvSolution's cached grad(U): the one the last closure call formed, while U has not
                    // changed since. When it HAS, OpenFOAM's first grad(U) request of this assembly forms
                    // and stores it and the others reuse that -- which of UEqn.H's operands asks first is
                    // the compiler's operand order, and brae does not model it. That is a laminar or
                    // kEpsilon case (no validate gradient), MRF, and any corrector U changed since.
                    // mesh().changing(): the registry is bypassed and its field deleted
                    const bool meshChanging = dyn && dyn->moving();
                    if (meshChanging)
                    {
                        f.gradUCache.valid = false;
                    }
                    if (f.gradUCache.on && !gradUUncached && !meshChanging)
                    {
                        if (!f.gradUCache.valid)
                            throw std::runtime_error(
                                "brae interFoam: fvSolution caches grad(U), and at this UEqn assembly U has "
                                "changed since grad(U) was last formed (or nothing formed it: only kOmegaSST's "
                                "validate and correct do). OpenFOAM's first request in the assembly would form "
                                "and store it for the others, in an operand order brae does not model.");
                        mi.gradUCached = &f.gradUCache.cells;
                        mi.gradUBndCached = &f.gradUCache.bnd;
                    }
                    FvVectorMatrix UEqn;
                    momentumPredictor(f.U, mi, force, msc, m, g, patches,
                                      f.momentumPredictorOn, UEqn);
                    // the predictor's solve moves U's eventNo
                    if (f.momentumPredictorOn)
                    {
                        f.gradUCache.valid = false;
                    }

                    DdtCorrInput dc;
                    dc.phiOld = &phiOld; dc.UOld = &UOld; dc.deltaT = rep.deltaT;
                    dc.rDeltaT = rDeltaTFor("ddtcorr");
                    dc.UOldBnd = &UOldBnd;
                    if (cnDdt)
                    {
                        // the step's first ddtCorr will evaluate dphidt0 and ask for
                        // phi.oldTime().oldTime(): created here, as a copy of phi.oldTime(), the
                        // first time it is asked for -- see phiOOExists
                        if (!phiOOExists && cnDdtCorrPhi.exists && cnDdtCorrPhi.timeIndex != rep.steps)
                        {
                            phiOO = phiOld;
                            phiOOExists = true;
                        }
                        if (!phiOOExists)
                        {
                            phiOO = phiOld;   // read by nothing on the step the field is created
                        }
                        dc.cn = &cnClock;
                        dc.cnDdt0U = &cnDdtCorrU;
                        dc.cnDdt0Phi = &cnDdtCorrPhi;
                        dc.UOO = &UOO;
                        dc.UOOBnd = &UOOBnd;
                        dc.phiOO = &phiOO;
                        // ...and on a DYNAMIC mesh the Uf pair, which fvcDdtUfCorr takes in phi's
                        // place. Same lazy creation as phiOO above.
                        if (f.meshIsDynamic)
                        {
                            if (!UfOOExists && cnDdtCorrUf.exists && cnDdtCorrUf.timeIndex != rep.steps)
                            {
                                UfOO = UfOld;
                                UfOOExists = true;
                            }
                            if (!UfOOExists)
                            {
                                UfOO = UfOld;   // read by nothing on the step the field is created
                            }
                            dc.cnDdt0Uf = &cnDdtCorrUf;
                            dc.UfOO = &UfOO;
                        }
                    }
                    // ddtCorr(U, phi, Uf) is ddtCorr(U, Uf) when the mesh is DYNAMIC -- fvcDdt.C:219
                    // asks mesh.dynamic(), which is moving OR topo-changing, so a REFINING mesh takes
                    // the Uf branch as a moving one does. Asking dyn instead sent an adaptive case down
                    // the phi branch, where OpenFOAM's own run of it writes a Uf at every step.
                    dc.UfOld = f.meshIsDynamic ? &UfOld : nullptr;
                    // ...and the static form READS phi.oldTime(), which creates the level: from here
                    // on the alpha step's blend has one to use (see phiOldRequested)
                    if (!f.meshIsDynamic) phiOldRequested = true;

                    PressureStepInput pin;
                    // outletPhaseMeanVelocity's updateCoeffs inside U.correctBoundaryConditions(): the
                    // corrected cells and the phase field's stored patch values
                    pin.controlIgnoreUpdatedLag = opmvIgnoreLag;
                    pin.uUpdateCoeffsFromCells = [&]()
                    {
                        if (opmvFrozen) return;
                        for (std::size_t pi = 0; pi < patches.size(); ++pi)
                        {
                            if (!f.U.boundary[pi]->isOutletPhaseMeanVelocity()) continue;
                            f.U.boundary[pi]->updatePhaseMean(f.alpha1.boundary[pi]->value(), f.U.internal,
                                                              g.Sf(), g.magSf());
                        }
                    };
                    pin.UEqn = &UEqn; pin.rho = &f.rho; pin.gh = &f.gh; pin.ghf = &f.ghfInternal;
                    pin.ghfBnd = &f.ghfBoundary;
                    pin.stf = &stf; pin.snGradRho = &snRho; pin.ddt = &dc;
                    pin.rhoBnd = &f.rhoBnd;
                    pin.nuBnd = &f.nuBnd;
                    pin.rhoPhi = &f.rhoPhi;
                    pin.taps = pressureTaps;
                    pin.solveLog = &rep.pSolves;
                    pin.mrf = f.mrfZones.empty() ? nullptr : &f.mrfZones;
                    pin.meshPhi = dyn ? &fvcMeshPhi(*dyn, f) : nullptr;
                    // fvc::correctUf runs on a DYNAMIC mesh (fvcMeshPhi.C:224), which a refining one is
                    pin.Uf = f.meshIsDynamic ? &f.Uf : nullptr;
                    // pEqn.H:4, rAU.ref() = 1/UEqn.A(): kept for the next mesh update's CorrectPhi
                    pin.rAUOut = &f.rAU;

                    PressureSolveControls psc;
                    psc.nCorrectors = lc.nCorrectors;
                    psc.nNonOrthogonalCorrectors = f.nNonOrthogonalCorrectors;
                    psc.correctedLaplacian = f.laplacianScheme.corrected;
                    psc.nonOrthCoeffs = f.laplacianScheme.nonOrthCoeffs;
                    psc.snGradLimitCoeff = f.laplacianScheme.limitCoeff;
                    psc.gradPrgh = f.gradPrgh;
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
                        pin.correctorIndex = c;
                        // continuityErrs.H, included by pEqn.H after every corrector, on the flux the
                        // corrector holds there (PressureStepInput::continuityDivOut): only the written
                        // cumulativeContErr reads it, so it is formed only when there is a writer
                        std::vector<scalar> continuityDiv;
                        pin.continuityDivOut = writer ? &continuityDiv : nullptr;
                        pressureCorrector(f.p_rgh, f.U, f.phi, f.p, pin, psc, m, g, patches);
                        if (writer)
                        {
                            // BRAE_CONTROL_CONTINUITY_RELATIVE=1 takes the flux the corrector LEAVES --
                            // relative on a moving mesh, this writer's first form, 100% off OpenFOAM's
                            // cumulativeContErr on sloshingTank2D. The control
                            // tests/interfoam_write_vs_openfoam.sh must go red on; never the default.
                            const char* rel = std::getenv("BRAE_CONTROL_CONTINUITY_RELATIVE");
                            if (rel && std::string(rel) == "1")
                            {
                                continuityDiv = fvc::div(f.phi, m, g, patches);
                            }
                            writer->addContinuityError(rep.deltaT, continuityDiv, g.V());
                        }
                    }
                    // `U = HbyA + ...; U.correctBoundaryConditions()` (pEqn.H) moved U's eventNo
                    f.gradUCache.valid = false;
                    // alpha1's inletOutlet reads the flux the step ended on, at the next MULES pass;
                    // U's patches keep their coefficients until the next momentum assembly (the alpha
                    // stage's push, above), as OpenFOAM's do -- the turbulence correct below reads
                    // their snGrad() with the switch the assembly set. pushFluxToPatches' declaration
                    // has the measurement.
                    pushFluxToPatches(f, patches, /*uCoefficientsKept=*/true);
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
                    if (cnDdt)
                    {
                        ti.cn = &cnClock;
                        ti.rhoOO = &rhoOO;
                    }
                    ti.nu = &f.nu;
                    ti.nuBnd = &f.nuBnd;
                    ti.deltaT = rep.deltaT;
                    ti.rDeltaT = rDeltaTFor("turbulence");
                    // THE STEP'S INDEX -- the same value Stage::advanceTime gives cnClock.timeIndex, so the
                    // closure's old-time snapshot and CrankNicolson's clock agree on what a step is.
                    ti.timeIndex = rep.steps;
                    // ...and WHICH OUTER CORRECTOR this is. With `turbOnFinalIterOnly no` the closure runs
                    // on every one and fvMatrix::solve() selects `<field>Final` only on the last
                    // (fvMatrix.C:1536-1542); the same flag the momentum solve already picks by at :1084.
                    ti.finalIter = (outerIndex >= lc.nOuterCorrectors - 1) ? 1 : 0;
                    // a moving mesh's old volumes and mesh flux, for the closure's ddt and divU
                    if (dyn && dyn->moving())
                    {
                        ti.V0 = &dyn->V0();
                        ti.meshPhi = &fvcMeshPhi(*dyn, f);
                    }
                    ti.fvOptions = f.fvOptions.empty() ? nullptr : &f.fvOptions;
                    ti.epsilonLog = &rep.epsilonSolves;
                    ti.omegaLog = &rep.omegaSolves;
                    ti.kLog = &rep.kSolves;
                    // ...and "Updating grad(U)": kOmegaSST's correct forms it, and a caching case keeps it
                    ti.gradUCache = &f.gradUCache;
                    correctInterTurbulence(f.turbulence, ti, m, g, patches);
                    // ...unless the mesh is changing, where the closure's gradient is formed and not kept
                    if (dyn && dyn->moving())
                    {
                        f.gradUCache.valid = false;
                    }
                    break;
                }
                case Stage::write:
                    // runTime.write() (interFoam.C:175): the stored state, nothing re-evaluated
                    if (writer && writeNow)
                    {
                        const std::vector<std::vector<scalar>> pB =
                            staticPressureBoundary(f.p_rgh, f.rhoBnd, f.ghfBoundary);
                        InterWriteState ws;
                        ws.time = rep.time;
                        ws.timeIndex = writer->startTimeIndex() + rep.steps;
                        ws.deltaT = rep.deltaT;
                        ws.alpha1 = &f.alpha1;
                        ws.U = &f.U;
                        ws.p_rgh = &f.p_rgh;
                        ws.p = &f.p;
                        ws.pBoundary = &pB;
                        ws.phi = &f.phi;
                        ws.alphaPhi0 = &alphaPhi10Write;
                        ws.turbulence = &f.turbulence;
                        // this loop's cells ARE the fields' own
                        ws.alpha1Cells = &f.alpha1.internal;
                        ws.UCells = &f.U.internal;
                        ws.p_rghCells = &f.p_rgh.internal;
                        ws.alpha1OldCells = &alphaOldWrite;
                        ws.alpha1OldBoundary = &alphaOldBndWrite;
                        ws.alpha1SubCycleBoundary = &alphaSubBndWrite;
                        ws.rDeltaT = &f.rDeltaT;
                        ws.Uf = &f.Uf;
                        ws.meshPhi = f.dynamicMesh ? &f.dynamicMesh->meshPhi() : nullptr;
                        ws.points = &m.points();
                        ws.displacement = f.dynamicMesh ? f.dynamicMesh->displacement() : nullptr;
                        ws.rigidBody = f.dynamicMesh ? f.dynamicMesh->rigidBody() : nullptr;
                        ws.rAU = &f.rAU;
                        writer->write(ws);
                    }
                    break;
            }
        };

        runTimeStep(lc, hooks);

        // ...and the old-time set moves forward, all four together -- five on a moving mesh -- with
        // the old-old level CrankNicolson reads taking the old one first
        UOO      = UOld;
        UOOBnd   = UOldBnd;
        rhoOO    = rhoOld;
        if (phiOOExists)
        {
            phiOO = phiOld;
        }
        deltaTPrev = rep.deltaT;
        alphaOld = f.alpha1.internal;
        UOld     = f.U.internal;
        UOldBnd  = patchValuesOf(f.U);
        rhoOld   = f.rho;
        phiOld   = f.phi;
        if (UfOOExists)
        {
            UfOO = UfOld;
        }
        UfOld = f.Uf;

        if (verbose)
        {
            scalar aMin = f.alpha1.internal[0], aMax = f.alpha1.internal[0], mass = 0, maxU = 0;
            for (label c = 0; c < m.nCells(); ++c)
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
    for (label c = 0; c < m.nCells(); ++c)
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
