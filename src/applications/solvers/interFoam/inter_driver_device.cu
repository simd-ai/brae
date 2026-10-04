// runInterFoamDevice -- the same interFoam time loop, on the GPU.
//
// See inter_driver_cpp.cuh for why the hooks live here rather than in the gate. Everything this calls
// is separately landed and separately gated; what this file owns is the wiring, and that wiring is
// what tests/test_device_inter_dambreak_alpha.cu measures against the host driver on damBreak.
#include "inter_phase_time.cuh"
#include "inter_driver_cpp.cuh"
#include "inter_set_rdeltat_cpp.cuh"
#include "device_fvc_smooth.cuh"
#include "device_inter_pcorr_solve.cuh"
#include "device_displacement_laplacian_assembly.cuh"
#include "device_patch_wave.cuh"
#include <cstring>
#include <optional>
#include <set>
#include "inter_amr_cpp.cuh"
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
#include <algorithm>
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

// A SURFACE VECTOR FIELD onto the device, component by component: the internal faces into `into[k]`
// and the boundary faces into `intoB[k]`, in the device's layout (a coupled patch's faces are in
// neither array -- see flattenPatches). Uf's two old levels are the only surface vector fields this
// loop uploads, and fvcDdtUfCorr reads both.
void uploadSurfaceVector(
    const SurfaceVectorField& f,
    const std::vector<FvPatch>& fvp,
    DeviceBuffer<scalar>* into,
    DeviceBuffer<scalar>* intoB)
{
    std::vector<scalar> c[3], b[3];
    for (const vector& v : f.internal)
    {
        c[0].push_back(v.x); c[1].push_back(v.y); c[2].push_back(v.z);
    }
    for (std::size_t pi = 0; pi < fvp.size() && pi < f.boundary.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;
        for (const vector& v : f.boundary[pi])
        {
            b[0].push_back(v.x); b[1].push_back(v.y); b[2].push_back(v.z);
        }
    }
    for (int k = 0; k < 3; ++k)
    {
        into[k].copyFrom(c[k]);
        intoB[k].copyFrom(b[k]);
    }
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

// AN `empty` PATCH'S ENTRIES ARE ITS CELLS' VALUES (EmptyPatchField::evaluate): written here for every such
// boundary face, three components at once, into arrays in the device mesh's boundary numbering. The device's
// own form of what the momentum hook built on the host and uploaded at every call.
__global__
void mirrorEmptyFacesKernel(
    int nB,
    const label* __restrict__ bndIsEmpty,
    const label* __restrict__ bndCell,
    const scalar* __restrict__ ux,
    const scalar* __restrict__ uy,
    const scalar* __restrict__ uz,
    scalar* __restrict__ ox,
    scalar* __restrict__ oy,
    scalar* __restrict__ oz)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= nB) return;
    if (!bndIsEmpty[b]) return;
    const label c = bndCell[b];
    ox[b] = ux[c];
    oy[b] = uy[c];
    oz[b] = uz[c];
}

// calculateNHatBoundary's first half on the device: each boundary cell's sum over its internal faces of
// Sf*interpolate(alpha), added on the owner side and subtracted on the neighbour's, in ascending face order --
// the host loop's terms and order (interface_properties_cpp.cu), so the host's second half
// (finishNHatBoundary) continues from it. MEASURED on RAS/DTCHull (68,113 boundary cells): 3.2 ms a call on
// the host, four calls a step.
__global__ void nHatStencilInternalKernel(
    int nStencil,
    const label* cells,
    const label* start,
    const label* faces,
    const label* own,
    const label* nei,
    const scalar* w,
    const scalar* Sfx,
    const scalar* Sfy,
    const scalar* Sfz,
    const scalar* alpha,
    bool ownerSideOnly,
    scalar* accx,
    scalar* accy,
    scalar* accz)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nStencil) return;
    const label cell = cells[k];
    vector acc{0, 0, 0};
    for (label j = start[k]; j < start[k + 1]; ++j)
    {
        const label f = faces[j];
        const label o = own[f];
        const scalar P = alpha[o];
        const scalar N = alpha[nei[f]];
        const scalar pf = w[f]*(P - N) + N;
        const vector Sfssf = vector{Sfx[f], Sfy[f], Sfz[f]}*pf;
        if (o == cell)
        {
            acc += Sfssf;
        }
        else if (!ownerSideOnly)
        {
            acc = acc - Sfssf;
        }
    }
    accx[k] = acc.x;
    accy[k] = acc.y;
    accz[k] = acc.z;
}

// fvc::snGrad's INTERNAL faces on the device, the host's arithmetic term for term (fvc.cu): the orthogonal
// part dc*(N - P), and under `corrected` the correction vector dotted with grad interpolated w*P + (1 - w)*N,
// capped by `limited <psi>`. The patch faces are the host's (snGradBoundary), which needs no gradient.
__global__ void snGradInternalKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ dcf,
    const scalar* __restrict__ v,
    const scalar* __restrict__ w,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    int corrected,
    scalar limitCoeff,
    scalar* __restrict__ out)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const label o = own[f];
    const label n = nei[f];
    scalar s = dcf[f]*(v[n] - v[o]);
    if (corrected)
    {
        const scalar wf = w[f];
        const scalar gfx = wf*gx[o] + (scalar(1) - wf)*gx[n];
        const scalar gfy = wf*gy[o] + (scalar(1) - wf)*gy[n];
        const scalar gfz = wf*gz[o] + (scalar(1) - wf)*gz[n];
        scalar corr = cvx[f]*gfx + cvy[f]*gfy + cvz[f]*gfz;
        if (limitCoeff > scalar(0) && limitCoeff < scalar(1))
        {
            corr *= fmin(limitCoeff*fabs(s)/((scalar(1) - limitCoeff)*fabs(corr) + scalar(1e-15)), scalar(1));
        }
        s += corr;
    }
    out[f] = s;
}

// surfaceTensionForce()'s internal faces: interpolate(sigma*K) -- fvc::interpolate(std::vector), w*P + (1 - w)*N
// -- times snGrad(alpha1), which `stf` holds on entry
__global__ void surfaceTensionInternalKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ K,
    scalar sigma,
    scalar* __restrict__ stf)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar sKo = sigma*K[own[f]];
    const scalar sKn = sigma*K[nei[f]];
    const scalar sKf = w[f]*sKo + (scalar(1) - w[f])*sKn;
    stf[f] = sKf*stf[f];
}

// fvc::snGrad's PATCH faces for an uncoupled patch, from the patch field's gradient coefficients and its face
// cells -- the host's branch, alone (fvc.cu); flattened in the device's boundary order (no coupled patch)
std::vector<scalar> snGradBoundary(
    const GeometricField<scalar>& vf,
    const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> out;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& fp = fvp[pi];
        const std::vector<scalar> gIC = vf.boundary[pi]->gradientInternalCoeffs();
        const std::vector<scalar> gBC = vf.boundary[pi]->gradientBoundaryCoeffs();
        for (label i = 0; i < fp.size; ++i)
        {
            out.push_back(gIC[static_cast<std::size_t>(i)]*vf.internal[static_cast<std::size_t>(fp.faceCells[i])]
                        + gBC[static_cast<std::size_t>(i)]);
        }
    }
    return out;
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
    const MutableMesh* mutableMesh,
    InterWriter* writer)
{
    InterFields f = buildInterFields(caseDir, startDir, m, g, fvp);
    if (writer)
    {
        registerUnwritten(*writer, f);
        // CrankNicolson's state on a moving mesh is written from the HOST loop's own objects (the ddt0
        // fields' patch values, the old levels' stored patch values); this loop keeps the cells on the
        // device and no patch half, so it names the files rather than reach its first write without them
        // ...KEPT NOW: the patch half of each ddt0 field is advanced on the device beside its cells
        // (DeviceCnDdt0PatchOperands) and comes down at a write time. BRAE_CONTROL_DEVICE_CN_WRITE_REFUSED=1
        // puts the refusal back.
        if (writer->writesCrankNicolson() && std::getenv("BRAE_CONTROL_DEVICE_CN_WRITE_REFUSED") != nullptr)
        {
            writer->refuseAtFirstWrite(
                "ddt0(rho,U), ddtCorrDdt0(U), ddtCorrDdt0(Uf), meshPhiCN_0, U_0, V0",
                "CrankNicolson's state on a moving mesh, which the device loop does not keep in its written form");
        }
    }
    // LOCAL TIME STEPPING. setRDeltaT.H runs on the HOST (inter_set_rdeltat_cpp: the smoothing is a
    // FaceCellWave) from this loop's fields, and each consumer -- the alpha pre-solve and CMULES,
    // fvm::ddt(rho, U), ddtCorr, kOmegaSST's two fvm::ddts -- reads the uploaded field. The gate's control
    // switches consumers off by name (readLtsScalarControl), on this loop as on the host's. kEpsilon and LES
    // under localEuler are refused where the case is read, for both loops.
    const std::set<std::string> ltsScalar = readLtsScalarControl();
    // GATE CONTROLS for outletPhaseMeanVelocity, never set by a solver, the host loop's two: FROZEN never
    // updates it, NOLAG updates it in the still-updated first corrector too. Both make the answer WRONG.
    const bool opmvFrozen = std::getenv("BRAE_CONTROL_OPMV_FROZEN") != nullptr;
    const bool opmvIgnoreLag = std::getenv("BRAE_CONTROL_OPMV_NOLAG") != nullptr;
    if (opmvFrozen)
        std::printf("  *** CONTROL MODE: outletPhaseMeanVelocity is never updated. This run is deliberately "
                    "wrong. ***\n");
    if (opmvIgnoreLag)
        std::printf("  *** CONTROL MODE: outletPhaseMeanVelocity is updated in the lagged first corrector. "
                    "This run is deliberately wrong. ***\n");
    // fvSolution's `cache { grad(U); }`: this loop reuses the closure's grad(U) at the next UEqn as the host
    // loop does (DeviceGradUCache, device_inter_step.cuh) -- uploaded from the host's validate() at the start,
    // re-formed after each kOmegaSST correct, bypassed while the mesh changes. On a motion-solver mesh, changing()
    // from its first update on, gradScheme bypasses the registry at every step (RAS/electrostaticDeposition,
    // interfoam_moving_vs_openfoam `esd`). REFUSED: a refining mesh (the host loop's own refusal), and on a STATIC
    // mesh, where the cache is live, a limited or least-squares grad(U) or a coupled pair -- the cached field is
    // formed unlimited Gauss on an uncoupled mesh here, and the assembly's sites skip what it was formed without.
    if (f.gradUCache.on && f.amr && f.amr->active)
        throw std::runtime_error(
            "brae interFoam: fvSolution caches grad(U) on a refining mesh. OpenFOAM bypasses and deletes the "
            "cached field on the steps the mesh changes (gradScheme.C:132-142) and reuses it on the others; "
            "that is not ported.");
    if (f.gradUCache.on && !f.dynamicMesh && (f.gradULimitK > 0 || f.gradULeastSq))
        throw std::runtime_error(
            "brae interFoam (device): fvSolution caches grad(U) and gradSchemes' grad(U) is limited or least-squares. "
            "The device loop caches an unlimited Gauss grad(U) only; the host loop carries the rest. Run without "
            "-device.");
    // GATE CONTROL, the host loop's, never set by a solver: every grad(U) formed afresh -- OpenFOAM's UNCACHED
    // answer. WRONG where the case caches it.
    const bool gradUUncached = std::getenv("BRAE_CONTROL_GRADU_UNCACHED") != nullptr;
    if (gradUUncached)
        std::printf("  *** CONTROL MODE: fvSolution's cached grad(U) is formed afresh at every site. This run is "
                    "deliberately wrong. ***\n");
    // ...and two that isolate the DEVICE-formed half, which only steps 2 onward read: STALE never re-forms it after
    // the closure (every step reuses validate()'s), BND_LIVE takes the cached cells but rebuilds the dev2 term's
    // boundary at the assembly from the patches as they stand then. Both WRONG.
    const bool gradUStale = std::getenv("BRAE_CONTROL_GRADU_STALE") != nullptr;
    if (gradUStale)
        std::printf("  *** CONTROL MODE: the cached grad(U) is never re-formed after the closure. This run is "
                    "deliberately wrong. ***\n");
    // ...and two for the mesh update on a moving mesh, each half of the defect the guard below removed:
    // MOVE_EVERY_OUTER runs this loop's move block at every outer corrector whatever moveMeshOuterCorrectors says
    // (the host stage still returns early there, so what runs is the refresh -- the host phi it did not rewrite
    // copied over the device's under correctPhi); V0_FROM_V takes V0 from the volumes as they stand before each
    // update instead of the mesh's once-per-step store. Both WRONG, together the old behaviour.
    const bool moveEveryOuter = std::getenv("BRAE_CONTROL_DEVICE_MOVE_EVERY_OUTER") != nullptr;
    if (moveEveryOuter)
        std::printf("  *** CONTROL MODE: the mesh update runs at every outer corrector. This run is "
                    "deliberately wrong. ***\n");
    const bool v0FromV = std::getenv("BRAE_CONTROL_DEVICE_V0_FROM_V") != nullptr;
    if (v0FromV)
        std::printf("  *** CONTROL MODE: V0 is re-taken from the volumes before every mesh update. This run is "
                    "deliberately wrong. ***\n");
    const bool gradUBndLive = std::getenv("BRAE_CONTROL_GRADU_BND_LIVE") != nullptr;
    if (gradUBndLive)
        std::printf("  *** CONTROL MODE: the cached grad(U)'s boundary is rebuilt at the assembly. This run is "
                    "deliberately wrong. ***\n");
    // THE PAIR, built here and not at the device-mesh stage, because the hooks below fill its share of
    // the surface fields and they are defined before the DeviceCyclic is.
    // ...INCLUDING a cyclicACMI the caller has coupled as a coincident pair (cpu::cyclicACMI::setup):
    // on the mask-scaled areas it IS a cyclic, with one difference this loop has to honour -- MULES
    // does not sync its limiter across it (CyclicInterface::ami).
    // ...AND a cyclicAMI the caller has coupled (cpu::cyclicAMIFvPatch::setup): its faces meet their
    // neighbours through the AMI's weighted stencil, which the DeviceCyclic carries beside the pair
    // (CyclicNbr) so every kernel reads the neighbour the host's patchNeighbourValue does.
    // NOT const: a moving mesh re-couples its cyclicAMI pairs and this list is rebuilt from them, in place,
    // so the hooks that hold a reference to it read the moved pair.
    std::vector<CyclicInterface> cyclics =
        buildCyclicInterfaces(m, g, fvp, /*includeCoupledACMI=*/true, /*includeCoupledAMI=*/true);
    // ...ANY coupled patch -- a cyclic, a cyclicACMI or a cyclicAMI pair alike: the cached field is formed without
    // the pair's contribution and the dev2 term's given boundary has no pair half
    for (std::size_t pi = 0; pi < fvp.size() && f.gradUCache.on && !f.dynamicMesh; ++pi)
    {
        if (fvp[pi].coupled)
            throw std::runtime_error(
                "brae interFoam (device): fvSolution caches grad(U) on a mesh with the coupled patch `" + fvp[pi].name
                + "`. The device loop caches grad(U) on an uncoupled mesh only; the host loop carries the pair. Run "
                "without -device.");
    }
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
    // ...and snGrad(p_rgh) on the pair, for the momentum predictor's force (the same hook, the same call)
    DeviceBuffer<scalar> dSnPrghIf;
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
    // ...AND A MOVING MESH IS CARRIED NOW. LESModel::correct() calls delta_().correct() first
    // (LESModel.C:251) and cubeRootVolDelta::correct() recomputes the width whenever the mesh is
    // changing (cubeRootVolDelta.C:128-134); the move branch of this loop re-runs the host's
    // LESdelta::compute and re-uploads what it produced, beside the wall distances.
    // EVERY CLOSURE ON THIS LOOP CROSSES A PAIR NOW. kEpsilon carried one; kOmegaSST and LES kEqn
    // refused, because k's equation is fvm::div - fvm::laplacian like the momentum's and across a
    // pair it needs the interface's off-diagonal in the matrix AND in the solve, the pair's own flux
    // in both divergences, its cells' grad(U), and the diffusivity there as the two CELLS
    // interpolated. All five are transcribed from the kEpsilon closure into KOmegaSSTInput::cyc and
    // LESkEqn::Input::cyc, and gated on validation/interFoamCyclic (`sst` and `les`), each against
    // its own walled control.
    // ...and so is a RAS closure on a mesh that moves. The host closure takes the moved mesh's old
    // volumes for fvm::ddt and the mesh flux for the convection's relative phi
    // (InterTurbulenceStepInput::V0, meshPhi; gated on waves/waveMakerPiston `pistonSST`, host arm);
    // the device closure's input has neither, and the instrument that runs the host closure inside
    // this loop is not handed them either -- both would run the ddt on the current volumes and
    // convect with the absolute flux, with nothing saying so. Found by an audit of hand-built control
    // structs (fields set at one construction site and not another), not by a case: no runnable
    // tutorial moves its mesh under a RAS closure on the device without an AMI, which is refused
    // separately. Refused until V0 and meshPhi are ported into the device closure and gated.
    // THE THIRD TERM WAS THE WALL DISTANCE, and it is ported now: the loop's move branch re-runs the
    // host block's moveInterTurbulence and re-uploads what moved with it -- wallDist::New(mesh).y()
    // for F1 and F2, nearWallDist for the wall functions, and DeviceWallData's own y/deltaCoeffs/wall
    // velocity (refreshDeviceInterTurbulenceGeometry). MEASURED on waves/waveMakerPiston `pistonSST`,
    // thirty steps of 0.01: the device arm went from alpha 4.2494e-08, p_rgh 4.2660e-08, U 1.7353e-05
    // to the host's own floor. Each of the three terms is load-bearing on its own -- V for V0 reads
    // 1.8155e-05, a relative divU 4.4704e-05, and the stale wall distance is the fail-proof the gate
    // carries. What is still refused is a moving mesh under LES kEqn (the filter width moves with the
    // volumes and the LES closure does not refresh it) and one carrying a coupled pair or an AMI,
    // both by name below.


    // fvOptions: the device loop applies explicitPorositySource/DarcyForchheimer on U and THE MANGROVE
    // PAIR -- multiphaseMangrovesSource on U, multiphaseMangrovesTurbulenceModel on k and epsilon under
    // kEpsilon (the case reader has already refused it under any other closure). Each other type is
    // refused BY ITS OWN NAME rather than by a blanket notice: the constraints and the rest are
    // host-only and gated there.
    for (const fvOptions::Option& o : f.fvOptions.options)
    {
        if (!o.active) continue;
        const bool darcy = o.unsupported.empty()
                        && (o.type == "explicitPorositySource")
                        && !o.fixedCoeff;
        if (darcy) continue;
        if (o.unsupported.empty() && o.mangroves == fvOptions::Option::Mangroves::source) continue;
        // ...and the turbulence option on EITHER lineage: the uniform one takes addSup(eqn),
        // -Sp(coeff), and the density-weighted one addSup(rho, eqn), -Sp(rho*coeff)
        // (fvOptionListTemplates.C:283-289 picks between them). Both arms carry both now, gated by the
        // `mangrove` and `densityVariable` profiles of tests/interfoam_mangrove_vs_openfoam.sh.
        if (o.unsupported.empty() && o.mangroves == fvOptions::Option::Mangroves::turbulence) continue;
        throw std::runtime_error(
            "brae interFoam (device): fvOptions has an active option `" + o.name + "` (" + o.type
            + "). The device loop applies explicitPorositySource/DarcyForchheimer and the mangrove pair "
            "only; the host loop carries this one. Refused rather than run the case without it.");
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
    // OpenFOAM keeps it off (DeviceInterStepHooks::updateUBoundary) -- AND ON A MOVING MESH, where it was
    // refused as unmeasured: the pressure half reads phi as it stands at constrainPressure, and on a
    // moving mesh that must be the RELATIVE flux (pEqn.H:70 made it so, and the post-move CorrectPhi block
    // ends in makeRelative), which is what pushFlux hands it here. MEASURED on
    // tests/interfoam_moving_vs_openfoam.sh `mixerPermeable` (testTubeMixer with the pair on its walls):
    // device alpha 3.9e-15, p_rgh 3.4e-14, U 4.5e-11 from OpenFOAM, the host's own level; handed the
    // ABSOLUTE flux instead, U 9.0e-04.
    // `grad(U) cellLimited` RUNS on this loop, and what the refusal here hid was not the momentum:
    // the shared assembler limits divDevReff's dev2 gradient through deviceCellLimitGradU, WITH the
    // pair (device_komega_sst.cu:838-846). What was actually wrong sat in the TURBULENCE closure --
    // KOmegaSSTInput carries the grad(U) limiter twice and this loop filled only the coeffs half, so
    // the production ran unlimited (nut 4.1315e-01). Gated on validation/interFoamCyclic `sstLimU`,
    // both arms, with grad(U), grad(k) and grad(omega) all cellLimited on a corrected laplacian.
    // What is still refused, at its own site, is the V-scheme reconstruction's limiter, which takes
    // no interface list (UEqn.cu).
    // ddtCorr's BOUNDARY HALF is on the device now: deviceDdtCorr already computed it (and already
    // zeroed it wherever U fixes a value, as fvcDdtPhiCoeff does), and the pressure step adds it to
    // phiHbyA with interpolate(rho*rAU)'s patch value, as the host does (inter_peqn_cpp.cu:531-573).
    // Gated on RAS/weirOverflow, whose outlet is a zeroGradient U.
    // A flowRateInletVelocity is REBUILT at every momentum assembly, and the U-boundary hook below does
    // that now, from the mixture's boundary rho, as the host driver does. What is still refused is a rate
    // that is a Function1 of time, which this driver takes as one number for the whole run -- the patch
    // itself throws on that (flowRateValue) -- and the variableHeight form, which reads the phase field.

    // A MESH THAT MOVES. The motion solve itself is a host operation on both arms -- OpenFOAM's
    // motion solver is a Laplacian on the point field, and brae has one host implementation of it --
    // so the device loop moves the mesh exactly as the host loop does (inter_driver_cpp.cu's
    // Stage::meshUpdate) and then refreshes the buffers it had uploaded from the geometry.
    // AN ADAPTIVE MESH IS A HOST CAPABILITY AND THIS ARM REFUSES IT BY NAME. buildInterFields is shared
    // with the host loop, and the moment it stopped handing an adaptive case to the motion factory (so the
    // host could carry a topology change) THIS loop started running such a case as if it were static.
    // MEASURED: tests/interfoam_refusals.sh's `device_mesh_dynamic` arm caught it -- "expected refused
    // naming `dynamicRefineFvMesh`, got runs". A capability claimed where it is not implemented is the
    // defect this project keeps finding; the refusal lives here, beside the loop that lacks it.
    // THIS LOOP NOW RE-UPLOADS THE MESH. The refusal that stood here said "this loop uploads the mesh
    // once and has no path that re-uploads it", which was true until the branch below `if (dyn)` was
    // written; the refusals gate's `device_refine_runs` arm holds it (the bare dictionary that
    // `device_mesh_dynamic` staged is one OpenFOAM itself stops on, now `mesh_dynamicRefine_device`), and every refusal an
    // adaptive case still has -- turbulence, a motion solver, MRF, fvOptions, CrankNicolson, a pressure
    // reference, `correctPhi no`, a coupled patch -- is raised by the SHARED case build and the shared
    // adapter (inter_case_cpp.cu, inter_amr_cpp.cu), so it fires on this arm without a copy here. The
    // list has shrunk twice since: fvOptions are RE-SELECTED through a change and MRF's face lists are
    // REBUILT, both on this arm too (buildPorosity and buildMrf are re-run inside the branch below), so
    // what an adaptive case still has refused is turbulence, a motion solver, `correctPhi no`, a CN
    // restart directory and a coupled patch.

    DynamicMotionSolverFvMesh* dyn = f.dynamicMesh.get();
    // THE MOTION SOLVER'S WALL DISTANCE ON THE GPU: displacementLaplacian's inverseDistance diffusivity asks for a
    // patch distance at every step, and its FaceCellWave runs on the device in the host's order
    // (devicePatchWave), bit for bit. MEASURED on waveMakerPiston refined to 896,000 cells: 804 ms a step with
    // the host wave. BRAE_CONTROL_PATCH_WAVE_HOST=1 keeps the host wave -- the identity gate's other arm.
    DevicePatchWave motionWave;
    CellFaces motionWaveCells;
    bool motionWaveCellsBuilt = false;
    if (dyn && dyn->hasDisplacementSolver() && std::getenv("BRAE_CONTROL_PATCH_WAVE_HOST") == nullptr)
    {
        dyn->setPatchWaveRunner(
            [&motionWave, &motionWaveCells, &motionWaveCellsBuilt](
                const PrimitiveMesh& mesh,
                const FvGeometry& geo,
                const std::vector<label>& seedFaces,
                std::vector<scalar>& cellDistSqr,
                std::vector<scalar>& boundaryDistSqr)
            {
                // the displacement solver's topology is fixed once it is attached
                if (!motionWaveCellsBuilt)
                {
                    motionWaveCells = cellFaces(mesh);
                    motionWaveCellsBuilt = true;
                    std::printf("  wall distance: the motion solver's wave runs on the GPU in the host's order; "
                                "BRAE_CONTROL_PATCH_WAVE_HOST=1 runs the host wave\n");
                }
                devicePatchWave(mesh, geo, motionWaveCells, seedFaces, motionWave, cellDistSqr, boundaryDistSqr);
            });
    }
    // ...and taken back when this loop ends: the runner holds this function's buffers, and the fields (with the
    // mesh motion in them) can be handed on to the caller
    struct RunnerReset
    {
        DynamicMotionSolverFvMesh* dyn;
        ~RunnerReset()
        {
            if (dyn)
            {
                dyn->setPatchWaveRunner(PatchWaveRunner());
                dyn->setDisplacementAssemblyRunner(DisplacementAssemblyRunner());
            }
        }
    } runnerReset{dyn};
    // A MESH THAT REFINES AND MOVES (laminar/oscillatingBox) runs on the HOST arm: the change first, then
    // the move, points0 carried through the change (inter_driver_cpp.cu's meshUpdate stage). This loop's
    // two branches -- the topology re-upload and the motion refresh -- are an if/else-if, so it would take
    // the motion and never refine. Refused by name until they are composed in OpenFOAM's order.
    // ...COMPOSED NOW, in that order (the outer corrector's refineAndMove). BRAE_CONTROL_DEVICE_REFINE_MOVE_REFUSED=1
    // puts the refusal back.
    if (dyn && f.amr && f.amr->active && std::getenv("BRAE_CONTROL_DEVICE_REFINE_MOVE_REFUSED") != nullptr)
        throw std::runtime_error(
            "brae interFoam (device): the mesh refines AND a motion solver moves it. The host arm runs it "
            "(the change, then the move); this loop's topology and motion branches are not yet composed in "
            "that order. Run without -device.");
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
        // ...A cyclicAMI PAIR IS CARRIED NOW: the layout is the pair's faces and owner cells, which a
        // move keeps, and refreshDeviceCyclicAfterMove rewrites the geometry and the stencil into the
        // same DeviceBuffer objects. A plain cyclic or a cyclicACMI under a move is still refused: its
        // rebuild goes through a different branch of buildCyclicInterfaces and nothing has gated it.
        for (const FvPatch& q : fvp)
        {
            if (q.coupled && q.type != "cyclicAMI")
            {
                throw std::runtime_error(
                    "brae interFoam -device: the case moves its mesh AND carries the coupled patch `"
                    + q.name + "`. The pair's device interface is built from the geometry once and the "
                    "step's hooks hold pointers into it; this loop does not rebuild it after a move. "
                    "Run without -device.");
            }
        }
        // A BODY THE FLUID MOVES (rigidBodyMeshMotion). The motion is a host stage on either arm, and it
        // takes the pressure and the shear on the body's patches from InterFields' own arrays AS THEY
        // STAND when the mesh is moved (the `forces` object looks the fields up, rigidBodyMeshMotion.C:
        // 299-307). This loop's hooks keep p_rgh (cells and patches, after every pressure solve), rho's
        // patch values and the mixture's nu on the host; what they do NOT keep is the closure's nut,
        // which lives in device buffers, and U's cells after the last corrector. Both are brought down
        // right before the mesh update -- see the stage.
        // kEpsilon UNDER CrankNicolson ON A MOVING MESH: the scheme's moving branch weights ddt0 by V0 and
        // V00 (CrankNicolsonDdtScheme.C:862-893). The host closure carries it (RAS/floatingObject); the
        // device kEpsilon carries the Euler form alone and would stop at its first call, so the case is
        // named here, before the first step.
        // ...CARRIED NOW: deviceCnFvmDdt takes V0 and V00 and the closure hands them on
        // (KEpsilonInput::V00). BRAE_CONTROL_DEVICE_CN_CLOSURE_REFUSED=1 puts the refusal back.
        if (f.turbulence.on && f.turbulence.model == cpu::interFoam::InterRasModel::KEpsilon
            && f.ddtU == DdtScheme::CrankNicolson
            && std::getenv("BRAE_CONTROL_DEVICE_CN_CLOSURE_REFUSED") != nullptr)
        {
            throw std::runtime_error(
                "brae interFoam -device: the case moves its mesh under CrankNicolson with a kEpsilon closure. "
                "CrankNicolson's fvm::ddt on a moving mesh is the scheme's moving branch (V0 and V00 weights), "
                "which the device kEpsilon does not carry. Run without -device.");
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
            if (q.type == "cyclicACMI")
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
    // A CASE THAT NEEDS A PRESSURE REFERENCE RUNS ON THE DEVICE: setReference pins the cell at its
    // CURRENT p_rgh (pEqn.H:47), the level shift of p rebuilds p_rgh from it (pEqn.H:74-83), and
    // adjustPhi (pEqn.H:21-26) is the pressure step's adjustPhi hook -- the host's own function, below.
    // It was refused here wherever a U patch could leave adjustPhi something to weigh -- damBreak with a
    // fixedFluxPressure atmosphere among them, which OpenFOAM itself aborts at step one's third corrector
    // (tests/interfoam_refusals.sh `closed_abort_device`) -- and silently skipped on every other closed
    // case, the balance test OpenFOAM stops on (adjustPhi.C:108) included.
    // NOT CONST: a topology change moves all four, and the branch that takes one re-reads them from the
    // mesh. Ten sites below index device arrays with them, and a cached count is how the host arm's own
    // Courant number came out at half OpenFOAM's (inter_driver_cpp.cu's nCAtStart note).
    label nC = m.nCells(), nIf = m.nInternalFaces();
    label nFaces = static_cast<label>(g.magSf().size());
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    // nOuterCorrectors IS the loop below now, transcribed from the host's runTimeStep
    // (inter_solve_cpp.cu:111-131). frozenFlow is not: pimple.frozenFlow() makes OpenFOAM `continue`
    // past the momentum, the pressure AND the turbulence corrector, and this loop has no such branch.
    // frozenFlow RUNS on this arm now: DeviceInterStepControls::frozenFlow returns out of
    // deviceInterStep after the alpha equation and the interface properties, and the turbulence
    // corrector below is guarded by the same flag -- the three things OpenFOAM's `continue` skips.

    // ...AND A VELOCITY CONDITION THAT NAMES A FLUX OTHER THAN phi. p_rgh's and alpha's conditions are
    // evaluated on the host, which hands each the flux its `phi` entry names (namedPatchFlux, through
    // pushFlux below) -- U's included, since U's patches are evaluated on the host by the updateUBoundary
    // hook. U's inletOutlet and pressureInletOutletVelocity switches that run ON THE DEVICE read the flux
    // each patch names too: rhoPhi where it says `phi rhoPhi;`, per face (DeviceInterStepControls::
    // uFluxIsRhoPhi). This case was REFUSED at U 2.6e-07 from the host, and the cause was not the switch:
    // namesRhoPhi (below) asked p_rgh and alpha only, so with U ALONE naming rhoPhi the host's rhoPhi was
    // never refreshed from the device and the host-evaluated U read the run's starting mass flux.
    // MEASURED on tests/interfoam_dambreak_vs_openfoam.sh `rhophiU`: device U 8.5e-12 from OpenFOAM, and
    // 4.8e-05 with U left out of namesRhoPhi. Any other name is refused by name.
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::string& name = f.U.boundary[pi]->fluxName();
        if (name == "phi" || name == "rhoPhi") continue;
        throw std::runtime_error(
            "brae interFoam (device): U's patch `" + fvp[pi].name + "` names the flux `" + name
            + "` in its `phi` entry; the device loop's velocity switches read phi or rhoPhi.");
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
    // THE PRESSURE RULE for pcorr: it is solved by the AMG-PCG on the GPU (DevicePcorrSolver), at the start and
    // at every mesh update, unless BRAE_PRESSURE_CASE_SOLVER asks for the case's own -- as every test does
    DevicePcorrSolver pcorrSolver;
    pcorrSolver.caseDir = caseDir;
    // a mesh that moves and does not refine keeps its topology, so pcorr's hierarchy is built once: the case
    // for its smoothed-aggregation hierarchy (DevicePcorrSolver::fixedTopology)
    pcorrSolver.fixedTopology = dyn != nullptr && !(f.amr && f.amr->active);
    auto withDevicePcorr = [&](CorrectPhiControls c)
    {
        if (std::getenv("BRAE_PRESSURE_CASE_SOLVER") == nullptr)
        {
            c.amgPcgSolve = [&pcorrSolver](
                const std::string& asked,
                const FvScalarMatrix& M,
                std::vector<scalar>& psi,
                const PrimitiveMesh& mesh,
                const FvGeometry& geo,
                const std::vector<FvPatch>& patches,
                scalar tol,
                scalar relTol,
                int maxIter,
                int minIter,
                SolverPerformance& perf)
            {
                return pcorrSolver.solve(asked, M, psi, mesh, geo, patches, tol, relTol, maxIter, minIter, perf);
            };
        }
        return c;
    };
    {
        // the same controls the host driver and the mesh update take, grad(pcorr)'s entry included
        const CorrectPhiControls cpc = withDevicePcorr(correctPhiControlsOf(f, meshAgglomeration));
        const SurfaceScalarField one = unitFaceField(m, fvp);
        // initCorrectPhi.H's rAUf: `fvc::interpolate(rAU())` under `correctPhi` (correctPhi.H:6) and a
        // literal 1 only in the `else` branch (initCorrectPhi.H:28). See the host loop for the
        // measurement -- a restart's rAU is the file's, and passing 1 there is 2.5e-04 of p_rgh.
        const SurfaceScalarField rAUfStart = f.correctPhi ? fvc::interpolate(f.rAU, m, g, fvp) : one;
        CorrectPhiInput cin;
        cin.rAUf = &rAUfStart;
        cin.rhoPhi = &f.rhoPhi;
        cin.solveLog = &initPcorrSolves;
        correctPhi(f.U, f.phi, f.p_rgh, cin, cpc, m, g, fvp);
        pushFluxToPatches(f, fvp);
        // ...and continuityErrs.H after it, in both of initCorrectPhi.H's branches (correctPhi.H:11 and
        // initCorrectPhi.H:31): the cumulative error OpenFOAM writes starts with this term, at the
        // deltaT runTime holds before setInitialDeltaT. Without it RAS/waterChannel's 0.3 wrote
        // -0.0023482777840429 where OpenFOAM writes -0.0023482774955524, 2.885e-10 apart -- exactly
        // this line's `global` in OpenFOAM's log.
        if (writer)
        {
            writer->addContinuityError(f.deltaT, fvc::div(f.phi, m, g, fvp), g.V());
        }
        // (a pcorr GAMG with a different nCellsInCoarsestLevel from p_rgh's was refused here while
        // the device built a hierarchy of its own. It shares the run's now, so the first solve's
        // entry decides it and the second entry's number is never read -- which is what OpenFOAM
        // does, and what the `both` profile of tests/interfoam_gamg_vs_openfoam.sh holds.)
    }

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    // how many times dm's GEOMETRY has been made: a build or a refresh after a move. What a cache of
    // geometry-derived boundary arrays keys on where it leaves the empty patches' faces out of its own key
    // (the momentum hook's device boundary).
    unsigned long long meshGeometryEpoch = 1;
    // THE MOTION SOLVER'S EQUATION ON THE GPU: the interior of displacementLaplacian's laplacian -- the face
    // coefficients and the diagonal, grad(cellDisplacement), the non-orthogonal correction and its per-cell sum
    // -- is assembled on the device mesh, each cell folding its faces in the host's order and every fused
    // product the host's (deviceDisplacementAssembly), bit for bit. The geometry it reads is dm's, which this
    // loop refreshes from the host's after every move, so it is the mesh the host assembly would read. The
    // patches' coefficients and the solve stay with the motion solver: cellDisplacement keeps the case's own
    // solver. MEASURED on waveMakerPiston refined to 896,000 cells: the host's matrix, gradient and correction
    // are 94 ms a step, the device's 5.3. BRAE_CONTROL_MOTION_ASSEMBLY_HOST=1 assembles on the host -- the
    // identity gate's other arm. Taken back with the wave runner when this loop ends.
    DeviceDisplacementAssembly motionAssembly;
    bool motionAssemblySaid = false;
    if (dyn && dyn->hasDisplacementSolver() && std::getenv("BRAE_CONTROL_MOTION_ASSEMBLY_HOST") == nullptr)
    {
        dyn->setDisplacementAssemblyRunner(
            [&dm, &motionAssembly, &motionAssemblySaid](
                const std::vector<FvPatch>& patches,
                const std::vector<scalar>& gamma,
                const std::vector<vector>& D,
                const std::vector<std::vector<vector>>& boundary,
                std::vector<scalar>& upper,
                std::vector<scalar>& diag,
                std::vector<vector>& corr)
            {
                if (!motionAssemblySaid)
                {
                    motionAssemblySaid = true;
                    std::printf("  mesh motion: the displacement equation's interior is assembled on the GPU in "
                                "the host's order; BRAE_CONTROL_MOTION_ASSEMBLY_HOST=1 assembles it on the host\n");
                }
                deviceDisplacementAssembly(dm, patches, gamma, D, boundary, motionAssembly, upper, diag, corr);
            });
    }

    // the masks the device needs that the mesh does not carry. IN A LAMBDA because a topology change
    // re-runs it: they are per boundary FACE, and hexRef8 splits boundary faces within their patch.
    std::vector<int> aFixes, aFlag, takeU, uFixes, uNamesRhoPhi;
    const auto buildBoundaryMasks = [&]()
    {
        aFixes.clear(); aFlag.clear(); takeU.clear(); uFixes.clear(); uNamesRhoPhi.clear();
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
                uNamesRhoPhi.push_back(f.U.boundary[pi]->fluxName() == "rhoPhi" ? 1 : 0);
            }
        }
    };
    buildBoundaryMasks();
    DeviceBuffer<int> dAFixes(aFixes), dAFlag(aFlag), dTakeU(takeU), dUFixes(uFixes), dUNamesRhoPhi(uNamesRhoPhi);
    // ...and whether any U patch names rhoPhi at all, which is what selects the named switch flux
    const bool uNamesRhoPhiAny =
        std::find(uNamesRhoPhi.begin(), uNamesRhoPhi.end(), 1) != uNamesRhoPhi.end();
    // the flux U's switches read at the two driver-level sites, from the HOST fields: the named one per
    // patch, as the step builds it on the device (DeviceInterStepControls::uFluxIsRhoPhi). phi's own
    // device array is passed through untouched when no patch names rhoPhi.
    const auto uSwitchFlux = [&](const DeviceBuffer<scalar>& phiB) -> DeviceBuffer<scalar>
    {
        std::vector<scalar> v;
        if (!uNamesRhoPhiAny || f.rhoPhi.boundary.size() != fvp.size())
        {
            phiB.copyTo(v);
            return DeviceBuffer<scalar>(v);
        }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            const std::vector<scalar>& src = (f.U.boundary[pi]->fluxName() == "rhoPhi")
                                           ? f.rhoPhi.boundary[pi] : f.phi.boundary[pi];
            v.insert(v.end(), src.begin(), src.end());
        }
        return DeviceBuffer<scalar>(v);
    };
    // ...and one of them is NOT a property of the patch but of the face and the instant. alpha's
    // variableHeightFlowRate is a MIXED condition whose rebuild() sets valueFraction to 1 on an INFLOW
    // face and 0 on the rest (fv_patch_field.cuh: `if (!(phi < -SMALL)) continue`), so a mask taken once
    // from fixesValue() answers per patch a question OpenFOAM asks per face per update, and MULES limits
    // with it. Refreshed wherever alpha's boundary is re-evaluated, from the patch's OWN valueFraction,
    // and left alone for every other condition.
    // ...BUT THAT IS NOT THE QUESTION MULES ASKS. MULESTemplates.C:337 tests `psiPf.fixesValue()`, a property of
    // the PATCH, and mixedFvPatchField::fixesValue() is `return true` (mixedFvPatchField.H:197) whatever the
    // valueFraction -- so every face of a variableHeightFlowRate patch joins the extrema, the outflow ones
    // included, as the host's mules_cpp.cu has them. The per-face mask dropped the outflow faces. MEASURED on
    // RAS/electrostaticDeposition, two steps, device against OpenFOAM: U 1.7e-05 with the per-face mask, where
    // OpenFOAM against itself at one more digit of tolerance moves 2.8e-06 and the host arm reads 1.8e-07.
    // BRAE_CONTROL_DEVICE_ALPHA_FIXES_PER_FACE=1 puts the per-face mask back -- the gate's control.
    bool hasPerFaceAlphaFixes = false;
    if (std::getenv("BRAE_CONTROL_DEVICE_ALPHA_FIXES_PER_FACE") != nullptr)
    {
        std::printf("  *** CONTROL MODE: alpha's fixesValue mask on a variableHeightFlowRate patch is taken per face. "
                    "This run is deliberately wrong. ***\n");
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (f.alpha1.boundary[pi]->isVariableHeightFlowRate()) hasPerFaceAlphaFixes = true;
        }
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
    // localEuler's per-cell rDeltaT, uploaded each step from the host setRDeltaT, and ddtCorr's face field
    // interpolate(rDeltaT) beside it; empty otherwise
    DeviceBuffer<scalar> dRDeltaT, dRDeltaTfI, dRDeltaTfB;
    // fvSolution's cached grad(U) on this loop -- see DeviceGradUCache
    DeviceGradUCache dGradUCache;
    // ...U's among them. U's patches are EVALUATED on the host too (the updateUBoundary hook), told their
    // flux by pushFlux -- and with U the only one naming rhoPhi, this used to leave the host's rhoPhi at the
    // value it started the run with: MEASURED on laminar/damBreak with U alone naming it, device U 2.6e-07
    // from the host, which the refusal of this case recorded and attributed to the device's switch.
    bool namesRhoPhi = false;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (f.p_rgh.boundary[pi]->fluxName() == "rhoPhi" || f.alpha1.boundary[pi]->fluxName() == "rhoPhi"
         || f.U.boundary[pi]->fluxName() == "rhoPhi")
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
    auto pushFlux = [&](bool uCoefficientsKept = false)
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
        pushFluxToPatches(f, fvp, uCoefficientsKept);
    };

    // U's FLUX-CONDITIONAL PATCHES ON A MOVING MESH, told the ABSOLUTE flux the pressure step handed back
    // before it made phi relative (C.phiAbsBndOut). No-op on a mesh that does not move, where the two are
    // one flux, and for a condition that names rhoPhi. BRAE_CONTROL_DEVICE_U_PATCH_RELATIVE=1 leaves them on
    // the relative flux -- the gate's control.
    DeviceBuffer<scalar>* phiAbsBndForU = nullptr;
    const bool uPatchRelativeControl = std::getenv("BRAE_CONTROL_DEVICE_U_PATCH_RELATIVE") != nullptr;
    if (uPatchRelativeControl)
    {
        std::printf("  *** CONTROL MODE: U's patches read the RELATIVE flux after the pressure corrector on a moving "
                    "mesh. This run is deliberately wrong. ***\n");
    }
    auto tellUAbsoluteFlux = [&](bool uCoefficientsKept)
    {
        if (!phiAbsBndForU || uPatchRelativeControl) return;
        std::vector<scalar> flat;
        phiAbsBndForU->copyTo(flat);
        std::size_t off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
            if (off + n > flat.size()) return;
            // the same patches pushFluxToPatches tells: a still-updated() patch keeps the flux its assembly
            // read unless its updateCoeffs evaluates. MEASURED on the permeable tube mixer (no predictor),
            // telling every patch here: device U 2.8e-03 from OpenFOAM, 4.1e-11 with this test.
            const bool tellU = !uCoefficientsKept || f.U.boundary[pi]->updateCoeffsEvaluates();
            if (tellU && f.U.boundary[pi]->fluxName() != "rhoPhi")
            {
                f.U.boundary[pi]->updateFromFlux(std::vector<scalar>(flat.begin() + off, flat.begin() + off + n));
            }
            off += n;
        }
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

    // THE BOUNDARY NORMAL the alpha hooks hand the device: calculateK's boundary half alone where that
    // reproduces it (interfaceProps::calculateNHatBoundary), the whole calculateK otherwise. MEASURED on
    // RAS/DTCHull (845,536 cells): calculateK 37 ms a call and four calls a step, for a boundary normal; with
    // this the alpha step is 112 ms a step against 310, every written file byte-identical. NOT ALWAYS TO THE
    // BIT: on capillaryRise (contact angle, 8000 cells) the boundary cells' gradient differs from the full
    // pass's by one ulp in a few cells -- the compiler fuses the multiply-adds of the two loops differently on
    // this aarch64 build -- and the written fields by 2.3e-16 at most.
    // The stencil is the mesh's addressing's, rebuilt when buildDeviceMesh stamps a new addressingId.
    // BRAE_CONTROL_NHAT_FULL=1 takes the whole calculateK every time -- the identity check's other arm.
    interfaceProps::NHatBoundaryStencil nHatStencil;
    unsigned long long nHatStencilId = 0;
    bool nHatStencilBuilt = false;
    const bool nHatFull = std::getenv("BRAE_CONTROL_NHAT_FULL") != nullptr;
    // ...and its first half, each boundary cell's internal-face sum, on the GPU from the hook's own device alpha
    // (nHatStencilInternalKernel), the host taking it from there. BRAE_CONTROL_NHAT_HOST_GRADIENT=1 forms it on
    // the host -- the identity check's other arm.
    const bool nHatHostGradient = std::getenv("BRAE_CONTROL_NHAT_HOST_GRADIENT") != nullptr;
    // BRAE_CONTROL_NHAT_DEVICE_OWNER_ONLY=1 drops the neighbour side's terms in the kernel -- the identity
    // check's control, which has to show the comparison sees the GPU half
    const bool nHatOwnerOnly = std::getenv("BRAE_CONTROL_NHAT_DEVICE_OWNER_ONLY") != nullptr;
    // AN `empty` PATCH IS LEFT OUT of the stencil and takes zeros (NHatBoundaryStencil::skipEmpty): OpenFOAM has
    // no faces there, and what reads the boundary normal on the device skips them or multiplies them by a zero
    // flux. On a 2-D mesh the stencil was every cell. BRAE_CONTROL_NHAT_EMPTY_CELLS=1 keeps them in, as before
    // -- the identity gate's other arm.
    const bool nHatEmptyCells = std::getenv("BRAE_CONTROL_NHAT_EMPTY_CELLS") != nullptr;
    DeviceBuffer<label> dStCells;
    DeviceBuffer<label> dStStart;
    DeviceBuffer<label> dStFaces;
    // the three components in one buffer, so they come down in ONE copy: three blocking copies were 1.5 ms a
    // call, most of it each copy's wait (RAS/DTCHull)
    DeviceBuffer<scalar> dStAcc;
    auto boundaryNHat = [&](SurfaceScalarField& nHb, const DeviceBuffer<scalar>& aDev)
    {
        if (!nHatFull && (!nHatStencilBuilt || nHatStencilId != dm.addressingId))
        {
            nHatStencil = interfaceProps::nHatBoundaryStencil(m, fvp, !nHatEmptyCells);
            nHatStencilId = dm.addressingId;
            nHatStencilBuilt = true;
            dStCells.copyFrom(nHatStencil.cells);
            dStStart.copyFrom(nHatStencil.start);
            dStFaces.copyFrom(nHatStencil.faces);
        }
        if (!nHatFull && interfaceProps::nHatBoundaryOnlyApplies(f.interface, false, nHatStencil))
        {
            static bool announced = false;
            if (!announced)
            {
                announced = true;
                std::printf("  nHat: the alpha hooks take the boundary normal from the boundary's own cells%s "
                            "(%zu of %ld); BRAE_CONTROL_NHAT_FULL=1 takes calculateK whole\n",
                            nHatEmptyCells ? "" : ", an empty patch's left out",
                            nHatStencil.cells.size(), (long)m.nCells());
            }
            if (nHatHostGradient)
            {
                interfaceProps::calculateNHatBoundary(f.alpha1, f.interface, m, g, fvp, nHatStencil, nHb);
                return;
            }
            const int nSt = static_cast<int>(nHatStencil.cells.size());
            dStAcc.resize(3*static_cast<std::size_t>(nSt));
            if (nSt > 0)
            {
                nHatStencilInternalKernel<<<(nSt + 255)/256, 256>>>(
                    nSt,
                    dStCells.data(),
                    dStStart.data(),
                    dStFaces.data(),
                    dm.owner.data(),
                    dm.nei.data(),
                    dm.w.data(),
                    dm.Sfx.data(),
                    dm.Sfy.data(),
                    dm.Sfz.data(),
                    aDev.data(),
                    nHatOwnerOnly,
                    dStAcc.data(),
                    dStAcc.data() + nSt,
                    dStAcc.data() + 2*static_cast<std::size_t>(nSt));
                cudaCheck(cudaGetLastError(), "nHatStencilInternalKernel");
            }
            std::vector<scalar> a3;
            dStAcc.copyTo(a3);
            std::vector<vector> acc(static_cast<std::size_t>(nSt));
            for (std::size_t k = 0; k < acc.size(); ++k)
            {
                acc[k] = vector{a3[k], a3[k + acc.size()], a3[k + 2*acc.size()]};
            }
            interfaceProps::finishNHatBoundary(f.alpha1, f.interface, g, fvp, nHatStencil, acc, nHb);
            return;
        }
        std::vector<scalar> Kb;
        interfaceProps::calculateK(f.alpha1, f.interface, m, g, fvp, false, nHb, Kb);
    };

    // the hooks: every one is per-patch host work, and nothing else
    DeviceInterStepHooks H;
    H.alpha.updateBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd, DeviceBuffer<scalar>& nBnd)
    {
        interPhase::Nested timed("hook alpha.updateBoundary");
        std::optional<interPhase::Nested> part;
        part.emplace("alpha hook: the flux to the patches");
        pushFlux();
        part.emplace("alpha hook: alpha down, the patches evaluated, their values up");
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        aBnd.copyFrom(patchValues(f.alpha1, fvp));
        part.emplace("alpha hook: the mixture's boundary and alpha's fixes");
        // The boundary viscosity from alpha's patch values as they stand HERE, before the curvature
        // pass below rewrites the contact-angle gradient: mixture.correct() is calcNu() and THEN
        // interfaceProperties::correct(). The last call of a step is the one UEqn reads. See the host
        // driver's mixtureCorrect stage for the measurement.
        updateMixtureBoundary(f, fvp);
        refreshAlphaFixes();
        part.emplace("alpha hook: the boundary normal");
        SurfaceScalarField nHb;
        boundaryNHat(nHb, a);
        part.emplace("alpha hook: the normal up");
        nBnd.copyFrom(flattenPatches(nHb.boundary, fvp));
        part.reset();
    };
    H.alpha.refreshBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd)
    {
        interPhase::Nested timed("hook alpha.refreshBoundary");
        pushFlux();
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        aBnd.copyFrom(patchValues(f.alpha1, fvp));
    };
    // ...and the top of the alpha step, which evaluates nothing: the host field's patch values as the
    // last evaluate left them (every evaluate of this loop goes through f.alpha1, so they are current)
    if (std::getenv("BRAE_CONTROL_DEVICE_ALPHA_TOP_EVALUATE") == nullptr)
    {
        H.alpha.storedBoundary = [&](DeviceBuffer<scalar>& aBnd)
        {
            aBnd.copyFrom(patchValues(f.alpha1, fvp));
        };
    }
    else
    {
        std::printf("  *** CONTROL MODE: alpha's boundary is evaluated at the top of the alpha step. This run "
                    "is deliberately wrong. ***\n");
    }
    // A RELAXED CORRECTOR'S BOUNDARY, as the host corrector takes it (alpha_eqn_cpp.cu): MULES's own
    // evaluate on the cells it left, then the relaxation's assignment through each patch's operator=.
    // f.alpha1's patch values on entry are alpha10's -- the last mixture.correct() left them, contact
    // angle included, and `volScalarField alpha10("alpha10", alpha1)` (VoF/alphaEqn.H:181) copies
    // exactly that.
    H.alpha.relaxBoundary =
        [&](const DeviceBuffer<scalar>& postMules, const DeviceBuffer<scalar>& relaxed,
            DeviceBuffer<scalar>& aBnd)
    {
        interPhase::Nested timed("hook alpha.relaxBoundary");
        pushFlux();
        std::vector<std::vector<scalar>> alpha10B(f.alpha1.boundary.size());
        for (std::size_t pi = 0; pi < f.alpha1.boundary.size(); ++pi)
        {
            alpha10B[pi] = f.alpha1.boundary[pi]->value();
        }
        postMules.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        relaxed.copyTo(f.alpha1.internal);
        cpu::interFoam::relaxAlphaBoundary(f.alpha1, alpha10B);
        aBnd.copyFrom(patchValues(f.alpha1, fvp));
    };
    // ...and updateBoundary's mixture.correct() WITHOUT its evaluate, which would overwrite that
    // assignment with the relaxed cells' evaluate
    H.alpha.mixtureCorrect =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd, DeviceBuffer<scalar>& nBnd)
    {
        interPhase::Nested timed("hook alpha.mixtureCorrect");
        pushFlux();
        a.copyTo(f.alpha1.internal);
        aBnd.copyFrom(patchValues(f.alpha1, fvp));
        updateMixtureBoundary(f, fvp);
        refreshAlphaFixes();
        SurfaceScalarField nHb;
        boundaryNHat(nHb, a);
        nBnd.copyFrom(flattenPatches(nHb.boundary, fvp));
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
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC,
            const DeviceBuffer<scalar>* phiCNBnd)
    {
        interPhase::Nested timed("hook alpha.divCoeffs");
        a.copyTo(f.alpha1.internal);
        f.alpha1.evaluateBoundary();
        // the flux the coefficients are built from: phiCN's patch values under CrankNicolson (a
        // coupled patch keeps the host's -- the device array holds none, and the pre-solve adds the
        // pair's coefficients itself), phi's otherwise
        std::vector<std::vector<scalar>> fluxBnd = f.phi.boundary;
        if (phiCNBnd)
        {
            unflatten(*phiCNBnd, fluxBnd);
        }
        // fvm::div's PATCH coefficients alone (fvm.cuh, the uncoupled branch): the patch flux times the field's
        // valueInternal/BoundaryCoeffs. Nothing here reads the internal faces, so the whole-mesh assembly this
        // was is not built -- MEASURED on RAS/DTCHull: 9.9 ms a call with it.
        // The device's boundary arrays hold the UNCOUPLED patches only, as its mesh does
        // (device_mesh.cuh:41-44): a coupled patch's coefficients are the interface's, and the alpha
        // pre-solve adds those itself from the pair. Flattening every patch here made the array longer
        // than the device's boundary-face count and the pre-solve refused it by size.
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            if (pi >= fluxBnd.size() || fluxBnd[pi].size() != static_cast<std::size_t>(fvp[pi].size))
            {
                throw std::runtime_error("brae: the alpha divergence hook has no flux on patch '" + fvp[pi].name
                                         + "' of its size; fvm::div's patch coefficients are that flux's");
            }
            const std::vector<scalar> vIC = f.alpha1.boundary[pi]->valueInternalCoeffs();
            const std::vector<scalar> vBC = f.alpha1.boundary[pi]->valueBoundaryCoeffs();
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const scalar pf = fluxBnd[pi][static_cast<std::size_t>(i)];
                i2.push_back(pf * vIC[static_cast<std::size_t>(i)]);
                b2.push_back((-pf) * vBC[static_cast<std::size_t>(i)]);
            }
        }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };
    // which device boundary object updateUBoundary last built whole, and on which addressing
    const DeviceVectorBoundary* uBoundaryBuiltIn = nullptr;
    DeviceVectorBoundaryShape uBoundaryBuiltShape;
    unsigned long long uBoundaryBuiltEpoch = 0;
    unsigned long long uBoundaryBuiltAddressing = 0;
    // AN `empty` PATCH'S ENTRIES STAY ON THE GPU. brae's empty patch mirrors its cells, the kernels read its
    // entries, and on a 2-D mesh the two empty patches hold two faces a cell: the hook built their key, their
    // state and their values on the host and uploaded them at every call. Now the key leaves them out (they
    // change only with the mesh: meshGeometryEpoch and the addressing say when), the state refresh and the
    // values go up for the other patches only, run by run, and the empty faces are mirrored from the device's
    // own U by a kernel at the calls where the host patch evaluates -- every call but the assembly's, where
    // OpenFOAM's patches keep their stored values and so do these. MEASURED on waveMakerPiston refined to
    // 896,000 cells, ms a step: the key 29, the state refresh 58 and the values 22 before.
    // BRAE_CONTROL_U_EMPTY_FROM_HOST=1 builds and uploads them on the host as before -- the identity gate's
    // other arm; BRAE_CONTROL_U_EMPTY_NO_MIRROR=1 skips the kernel -- its control.
    const bool uEmptyFromHost = std::getenv("BRAE_CONTROL_U_EMPTY_FROM_HOST") != nullptr;
    const bool uEmptyNoMirror = std::getenv("BRAE_CONTROL_U_EMPTY_NO_MIRROR") != nullptr;
    // BRAE_CONTROL_U_EMPTY_CHECK=1: at every call the host builds every face's state and values as it used to,
    // and one bit's difference from what the device now holds stops the run and names the entry -- the gate's
    // oracle. The WRITTEN FILES cannot be it: the kernels multiply an empty face's entries by zero, so a run
    // with the mirror skipped writes the same bytes (measured on weirOverflow, waveMakerPiston, capillaryRise).
    const bool uEmptyCheck = std::getenv("BRAE_CONTROL_U_EMPTY_CHECK") != nullptr;
    DeviceBuffer<scalar> uValuesStage;
    auto sameOnDevice = [](
        const char* what,
        int k,
        const DeviceBuffer<scalar>& d,
        const std::vector<scalar>& h)
    {
        std::vector<scalar> got;
        d.copyTo(got);
        if (got.size() == h.size() && (h.empty() || std::memcmp(got.data(), h.data(), h.size()*sizeof(scalar)) == 0))
        {
            return;
        }
        std::size_t at = 0;
        while (at < got.size() && at < h.size() && std::memcmp(&got[at], &h[at], sizeof(scalar)) == 0)
        {
            ++at;
        }
        char line[200];
        std::snprintf(line, sizeof(line), "%s, component %d, boundary face %zu of %zu (host %zu): the device holds "
                      "%.17g, the host builds %.17g", what, k, at, got.size(), h.size(),
                      at < got.size() ? got[at] : 0.0, at < h.size() ? h[at] : 0.0);
        throw std::runtime_error(std::string("brae interFoam: U's boundary kept on the GPU is not the host's: ")
                                 + line);
    };
    H.updateUBoundary =
        [&](const DeviceBuffer<scalar>& ux, const DeviceBuffer<scalar>& uy,
            const DeviceBuffer<scalar>& uz, DeviceVectorBoundary& db, DeviceBuffer<scalar>* ubOut,
            DeviceUBoundaryCall call)
    {
        interPhase::Nested timed("hook updateUBoundary");
        std::optional<interPhase::Nested> part;
        part.emplace("U hook: the flux to the patches");
        // p_rgh's and alpha's patches take the new flux at every call; U's do not at the one call
        // where OpenFOAM's are still updated() -- see DeviceUBoundaryCall
        pushFlux(call == DeviceUBoundaryCall::evaluateStillUpdated);
        // ...and on a MOVING mesh U's patches read the ABSOLUTE flux at the corrector's
        // correctBoundaryConditions (pEqn.H:61, ahead of makeRelative at :69) -- see tellUAbsoluteFlux
        // -- and where the patches are still updated(), only the classes pushFlux itself tells there
        if (call == DeviceUBoundaryCall::evaluate || call == DeviceUBoundaryCall::evaluateStillUpdated)
        {
            tellUAbsoluteFlux(call == DeviceUBoundaryCall::evaluateStillUpdated);
        }
        part.emplace("U hook: U down to the host field");
        std::vector<scalar> x, y, z;
        ux.copyTo(x); uy.copyTo(y); uz.copyTo(z);
        for (label c = 0; c < nC; ++c) f.U.internal[c] = vector{x[c], y[c], z[c]};
        part.emplace("U hook: the patches' updates and evaluate");
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
        // outletPhaseMeanVelocity's updateCoeffs, from the phase field's stored patch values and U's face
        // cells as they stand (outletPhaseMeanVelocityFvPatchVectorField.C:130-163), at the host loop's
        // instants: the assembly, and each pressure corrector's correctBoundaryConditions BEFORE its
        // evaluate -- but not where the patch is still updated(): the first corrector of a pass with no
        // predictor (PressureStepInput::uPatchesUpdatedAtEntry) and the predictor solve's own evaluate.
        const bool opmvNow = call == DeviceUBoundaryCall::assembly || call == DeviceUBoundaryCall::evaluate
                          || (call == DeviceUBoundaryCall::evaluateStillUpdated && opmvIgnoreLag);
        if (opmvNow && !opmvFrozen)
        {
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (!f.U.boundary[pi]->isOutletPhaseMeanVelocity()) continue;
                if (f.U.boundary[pi]->alphaFieldName() != f.alphaName)
                    throw std::runtime_error(
                        "brae interFoam: U patch `" + fvp[pi].name + "` is an outletPhaseMeanVelocity naming "
                        "`alpha " + f.U.boundary[pi]->alphaFieldName() + "`, and this case's phase field is `"
                        + f.alphaName + "`. OpenFOAM looks the named field up and stops without it.");
                f.U.boundary[pi]->updatePhaseMean(f.alpha1.boundary[pi]->value(), f.U.internal, g.Sf(), g.magSf());
            }
        }
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
        // The boundary the last FULL build left in this same object keeps its geometry and masks, and only the
        // patches' state is built and uploaded (refreshDeviceVectorBoundaryState) -- where everything the other
        // arrays are built from is unchanged (deviceVectorBoundaryShape: a mesh that moves, or a cyclicACMI whose
        // open fraction moves, changes it and is rebuilt). MEASURED on RAS/DTCHull: the full build 16 ms a call,
        // three calls a step; the hook 76 ms a step before, 32 after. BRAE_CONTROL_U_BOUNDARY_FULL=1 builds it
        // whole every call -- the identity check's other arm; BRAE_CONTROL_U_BOUNDARY_STALE=1 skips the refresh --
        // its control.
        static const bool uBoundaryFull = std::getenv("BRAE_CONTROL_U_BOUNDARY_FULL") != nullptr;
        static const bool uBoundaryStale = std::getenv("BRAE_CONTROL_U_BOUNDARY_STALE") != nullptr;
        part.emplace("U hook: the device boundary's key");
        // where the patches that are not empty sit in the boundary's numbering; the empty faces stay on the
        // device when there are any and the boundary is in the device mesh's numbering
        std::size_t nBndAll = 0;
        const std::vector<DeviceBoundaryRange> ranges = deviceNonEmptyBoundaryRanges(fvp, nBndAll);
        std::size_t nHostFaces = 0;
        for (const DeviceBoundaryRange& r : ranges)
        {
            nHostFaces += r.n;
        }
        const bool emptyOnDevice = !uEmptyFromHost && nHostFaces < nBndAll
                                && nBndAll == static_cast<std::size_t>(dm.nBndFaces);
        if (emptyOnDevice)
        {
            static bool said = false;
            if (!said)
            {
                said = true;
                std::printf("  U boundary: an empty patch's entries stay on the GPU, mirrored from its cells "
                            "(%zu of %zu boundary faces are the host's); BRAE_CONTROL_U_EMPTY_FROM_HOST=1 builds "
                            "and uploads them all\n", nHostFaces, nBndAll);
            }
        }
        const bool evaluated = call != DeviceUBoundaryCall::assembly;
        auto mirrorEmpty = [&](
            DeviceBuffer<scalar>& ox,
            DeviceBuffer<scalar>& oy,
            DeviceBuffer<scalar>& oz)
        {
            if (uEmptyNoMirror) return;
            const int nB = dm.nBndFaces;
            mirrorEmptyFacesKernel<<<(nB + 255)/256, 256, 0, cudaStreamPerThread>>>(
                nB,
                dm.bndIsEmpty.data(),
                dm.bndCell.data(),
                ux.data(),
                uy.data(),
                uz.data(),
                ox.data(),
                oy.data(),
                oz.data());
            cudaCheck(cudaGetLastError(), "mirrorEmptyFacesKernel");
        };
        {
            // the key first, which is cheap: equal keys give equal non-state arrays (deviceVectorBoundaryShape),
            // so only the state is built; otherwise everything is, and checked against the last full build
            DeviceVectorBoundaryShape shape = deviceVectorBoundaryShape(f.U, fvp, g, emptyOnDevice);
            const bool sameShape = !uBoundaryFull && uBoundaryBuiltIn == &db && shape == uBoundaryBuiltShape
                                && uBoundaryBuiltEpoch == meshGeometryEpoch
                                && uBoundaryBuiltAddressing == dm.addressingId;
            if (sameShape)
            {
                part.emplace("U hook: the device boundary's state refreshed");
                if (!uBoundaryStale && emptyOnDevice)
                {
                    refreshDeviceVectorBoundaryState(
                        db,
                        deviceVectorBoundaryArrays(f.U, fvp, g, false, true, true),
                        &ranges);
                    if (evaluated)
                    {
                        mirrorEmpty(db.comp[0].refValue, db.comp[1].refValue, db.comp[2].refValue);
                    }
                    if (uEmptyCheck)
                    {
                        const DeviceVectorBoundaryHost all = deviceVectorBoundaryArrays(f.U, fvp, g, false, true);
                        for (int k = 0; k < 3; ++k)
                        {
                            sameOnDevice("refValue", k, db.comp[k].refValue, all.ref[k]);
                            sameOnDevice("valueFraction", k, db.comp[k].valueFraction, all.vf[k]);
                            sameOnDevice("refGrad", k, db.comp[k].refGrad, all.rg[k]);
                            sameOnDevice("the stored inletOutlet value", k, db.comp[k].ioStored, all.iost[k]);
                        }
                    }
                }
                else if (!uBoundaryStale)
                {
                    refreshDeviceVectorBoundaryState(db, deviceVectorBoundaryArrays(f.U, fvp, g, false, true));
                }
            }
            else
            {
                part.emplace("U hook: the device boundary built whole");
                db = uploadDeviceVectorBoundary(deviceVectorBoundaryArrays(f.U, fvp, g, false));
                uBoundaryBuiltIn = &db;
                uBoundaryBuiltShape = std::move(shape);
                uBoundaryBuiltEpoch = meshGeometryEpoch;
                uBoundaryBuiltAddressing = dm.addressingId;
            }
        }
        part.reset();
        if (!ubOut) return;
        part.emplace("U hook: the patch values out");
        // AN `empty` PATCH'S ENTRIES ARE READ ON THE DEVICE, here and in the state arrays above: MEASURED with NaN
        // written over them, RAS/weirOverflow stops in its second step. brae's empty patch mirrors its cells
        // and the kernels' terms there vanish or cancel, so they have to stay current -- on the device, from the
        // device's own U (mirrorEmptyFacesKernel), where the buffers are already this step's.
        if (emptyOnDevice && evaluated && ubOut[0].size() == nBndAll && ubOut[1].size() == nBndAll
         && ubOut[2].size() == nBndAll)
        {
            std::vector<scalar> pack(3*nHostFaces);
            std::size_t at = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type) || fvp[pi].type == "empty") continue;
                for (const vector& u : f.U.boundary[pi]->value())
                {
                    pack[at] = u.x;
                    pack[nHostFaces + at] = u.y;
                    pack[2*nHostFaces + at] = u.z;
                    ++at;
                }
            }
            if (at != nHostFaces)
            {
                throw std::runtime_error("brae interFoam: U's patches do not hold a value a face.");
            }
            if (nHostFaces > 0)
            {
                uValuesStage.copyFrom(pack);
            }
            for (int k = 0; k < 3; ++k)
            {
                std::size_t from = static_cast<std::size_t>(k)*nHostFaces;
                for (const DeviceBoundaryRange& r : ranges)
                {
                    cudaCheck(cudaMemcpyAsync(ubOut[k].data() + r.at, uValuesStage.data() + from,
                                              r.n*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread),
                              "U patch values out");
                    from += r.n;
                }
            }
            mirrorEmpty(ubOut[0], ubOut[1], ubOut[2]);
            if (uEmptyCheck)
            {
                std::vector<scalar> all[3];
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    if (isCoupledInterfaceType(fvp[pi].type)) continue;
                    for (const vector& u : f.U.boundary[pi]->value())
                    {
                        all[0].push_back(u.x);
                        all[1].push_back(u.y);
                        all[2].push_back(u.z);
                    }
                }
                for (int k = 0; k < 3; ++k)
                {
                    sameOnDevice("the patch values", k, ubOut[k], all[k]);
                }
            }
            part.reset();
            return;
        }
        std::vector<scalar> bx, by, bz;
        std::size_t nOut = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (!isCoupledInterfaceType(fvp[pi].type)) nOut += static_cast<std::size_t>(fvp[pi].size);
        }
        bx.reserve(nOut);
        by.reserve(nOut);
        bz.reserve(nOut);
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            for (const vector& u : f.U.boundary[pi]->value())
            {
                bx.push_back(u.x);
                by.push_back(u.y);
                bz.push_back(u.z);
            }
        }
        ubOut[0].copyFrom(bx);
        ubOut[1].copyFrom(by);
        ubOut[2].copyFrom(bz);
        part.reset();
    };
    H.interfaceForces =
        [&](const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& Kd,
            const DeviceBuffer<scalar>& rd, const DeviceBuffer<scalar>& rhoBd,
            DeviceBuffer<scalar>& stf, DeviceBuffer<scalar>& snRho,
            DeviceBuffer<scalar>& nuC, DeviceBuffer<scalar>& nuB, DeviceBuffer<scalar>& snP)
    {
        interPhase::Nested timed("hook interfaceForces");
        // ...and NOT alpha's boundary: alphaEqnSubCycle.H evaluates alpha nowhere after the sub-cycle,
        // and the alpha step's last hook left f.alpha1's patch values as OpenFOAM's stand -- after a
        // relaxed corrector that is an ASSIGNMENT, which an evaluate here overwrote (the host driver
        // dropped the same line, inter_driver_cpp.cu's alphaSubCycle stage). MEASURED on
        // RAS/electrostaticDeposition at step two: 240 variableHeightFlowRate faces 5.1256e-10 from the
        // value OpenFOAM wrote, exactly on its owner cell.
        a.copyTo(f.alpha1.internal);
        Kd.copyTo(f.K);
        rd.copyTo(f.rho);
        // THE FACE LOOPS ON THE DEVICE where no pair is in them: the internal faces of the surface-tension
        // flux and of snGrad(rho) by the kernels above, the patch faces by the host's own branch. MEASURED on
        // RAS/DTCHull (845,536 cells): this hook 57 ms a step on the host, of which snGrad(rho) and its upload
        // 25, snGrad(alpha) 12 and the flux's product and upload 11. BRAE_CONTROL_FORCES_HOST=1 keeps the host's
        // whole -- the identity check's other arm.
        const bool forcesOnDevice = cyclics.empty() && std::getenv("BRAE_CONTROL_FORCES_HOST") == nullptr;
        // BRAE_CONTROL_FORCES_NO_CORRECTION=1 drops the non-orthogonal correction from the device's face loop --
        // the identity gate's control, which has to show the comparison can fail
        static const bool forcesNoCorrection = std::getenv("BRAE_CONTROL_FORCES_NO_CORRECTION") != nullptr;
        const bool snCorrD = f.snGradScheme.corrected;
        const scalar snLimD = f.snGradScheme.limitCoeff;
        const bool snNonOrthD = f.snGradScheme.nonOrthCoeffs;
        const int blocksIf = (nIf + 255)/256;
        auto deviceSnGradInternal = [&](
            const DeviceBuffer<scalar>& v,
            const DeviceBuffer<scalar>& bval,
            const GradChoice& gradChoice,
            DeviceBuffer<scalar>& out)
        {
            out.resize(static_cast<std::size_t>(nIf));
            DeviceBuffer<scalar> gx, gy, gz;
            if (snCorrD)
            {
                deviceGradOf(dm, v, bval, gradChoice.leastSquares, gradChoice.cellLimitK, gx, gy, gz);
            }
            const DeviceBuffer<scalar>& dcf = (snCorrD || snNonOrthD) ? dm.nonOrthDc : dm.dc;
            snGradInternalKernel<<<blocksIf, 256>>>(
                nIf, dm.owner.data(), dm.nei.data(), dcf.data(), v.data(), dm.w.data(),
                snCorrD ? gx.data() : nullptr, snCorrD ? gy.data() : nullptr, snCorrD ? gz.data() : nullptr,
                dm.corrVecX.data(), dm.corrVecY.data(), dm.corrVecZ.data(),
                (snCorrD && !forcesNoCorrection) ? 1 : 0, snLimD, out.data());
            cudaCheck(cudaGetLastError(), "snGradInternal");
        };
        auto appendBoundary = [&](DeviceBuffer<scalar>& full, const DeviceBuffer<scalar>& internal,
                                  const std::vector<scalar>& bnd)
        {
            full.resize(static_cast<std::size_t>(nIf) + bnd.size());
            cudaCheck(cudaMemcpyAsync(full.data(), internal.data(), static_cast<std::size_t>(nIf)*sizeof(scalar),
                                      cudaMemcpyDeviceToDevice, cudaStreamPerThread), "forces internal copy");
            if (!bnd.empty())
            {
                cudaCheck(cudaMemcpyAsync(full.data() + nIf, bnd.data(), bnd.size()*sizeof(scalar),
                                          cudaMemcpyHostToDevice, cudaStreamPerThread), "forces boundary copy");
                cudaCheck(cudaStreamSynchronize(cudaStreamPerThread), "forces boundary sync");
            }
        };
        if (forcesOnDevice)
        {
            // surfaceTensionForce() = interpolate(sigma*K)*snGrad(alpha1)
            const DeviceBuffer<scalar> alphaBval(patchValues(f.alpha1, fvp));
            DeviceBuffer<scalar> stfInternal;
            deviceSnGradInternal(a, alphaBval, f.gradAlpha1, stfInternal);
            surfaceTensionInternalKernel<<<blocksIf, 256>>>(
                nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), Kd.data(), f.interface.sigma, stfInternal.data());
            cudaCheck(cudaGetLastError(), "surfaceTensionInternal");
            const std::vector<scalar> snAB = snGradBoundary(f.alpha1, fvp);
            std::vector<scalar> stfB(snAB.size());
            {
                std::size_t k = 0;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    for (label i = 0; i < fvp[pi].size; ++i, ++k)
                    {
                        const scalar sKb = f.interface.sigma*f.K[static_cast<std::size_t>(fvp[pi].faceCells[i])];
                        stfB[k] = sKb*snAB[k];
                    }
                }
            }
            appendBoundary(stf, stfInternal, stfB);

            // rho's CALCULATED patch values, from the device's own blend -- see the host branch below
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
            stepRhoBnd = rb;
            if (deviceClosure)
            {
                deviceCopy(dStepRhoBnd, rhoBd);
            }
            const GeometricField<scalar> rhoF = rhoWithPatchValues(f.rho, rb, fvp);
            DeviceBuffer<scalar> snRhoInternal;
            deviceSnGradInternal(rd, rhoBd, f.gradRho, snRhoInternal);
            appendBoundary(snRho, snRhoInternal, snGradBoundary(rhoF, fvp));
        }
        else
        {
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
            // ...and WHICH delta coefficients: `uncorrected` and `limited 0` take nonOrthDeltaCoeffs
            // without the correction (uncorrectedSnGrad.H:113-119), which `orthogonal` does not
            const bool snNonOrth = f.snGradScheme.nonOrthCoeffs;
            const SurfaceScalarField snA = fvc::snGrad(f.alpha1, m, g, fvp, snCorr,
                                                       f.gradAlpha1.leastSquares, f.gradAlpha1.cellLimitK, snLim,
                                                       snNonOrth);
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
                                                          f.gradRho.cellLimitK, snLim, snNonOrth);
            snRho.copyFrom(fullFace(snRhoF, fvp));
            if (!cyclics.empty()) dSnRhoIf.copyFrom(coupledFace(snRhoF, cyclics));
        }

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
            const SurfaceScalarField snPF = fvc::snGrad(f.p_rgh, m, g, fvp, f.snGradScheme.corrected,
                                                        f.gradPrgh.leastSquares, f.gradPrgh.cellLimitK,
                                                        f.snGradScheme.limitCoeff, f.snGradScheme.nonOrthCoeffs);
            snP.copyFrom(fullFace(snPF, fvp));
            if (!cyclics.empty()) dSnPrghIf.copyFrom(coupledFace(snPF, cyclics));
        }
        else snP.resize(0);
    };
    H.pressure.pressureCoeffs =
        [&](const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>& phiHB,
            const DeviceBuffer<scalar>& rAUfAll, const DeviceBuffer<scalar>& rAUCell,
            DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC, DeviceBuffer<scalar>& cycJump)
    {
        interPhase::Nested timed("hook pressure.pressureCoeffs");
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
        // ONLY THE BOUNDARY'S rAUf comes down -- the hook reads no internal face -- and rAU's cells only where a
        // coupled patch interpolates them. MEASURED on RAS/DTCHull (845,536 cells, 69,887 boundary faces): the
        // whole of both, and the whole-mesh fvm::laplacian below it, made this hook 34 ms a step.
        std::vector<scalar> hB, rA, rAUc;
        phiHB.copyTo(hB);
        {
            const std::size_t nBnd = rAUfAll.size() - static_cast<std::size_t>(nIf);
            rA.resize(nBnd);
            cudaCheck(cudaMemcpy(rA.data(), rAUfAll.data() + nIf, nBnd*sizeof(scalar), cudaMemcpyDeviceToHost),
                      "pressureCoeffs rAUf boundary");
        }
        if (!cyclics.empty())
        {
            rAUCell.copyTo(rAUc);
        }
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
                    sn[i] = (hB[off + i] - SfU) / (q.magSf[i] * rA[off + i]);
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
        // rAUf on the patches: the device's array on an uncoupled one, fvc::interpolate(rAU) -- the two cells'
        // rAU on the pair's own weights -- on a coupled one, which the device's array does not carry
        std::vector<std::vector<scalar>> rfB(fvp.size());
        {
            label o = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                const FvPatch& q = fvp[pi];
                if (isCoupledInterfaceType(q.type))
                {
                    for (label i = 0; i < q.size; ++i)
                    {
                        rfB[pi].push_back(coupledLinear(q, i, rAUc));
                    }
                    continue;
                }
                for (label i = 0; i < q.size; ++i)
                {
                    rfB[pi].push_back(rA[static_cast<std::size_t>(o++)]);
                }
            }
        }
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
        // fvm::laplacian's PATCH coefficients alone (fvm.cuh, the uncoupled branch): gamma*magSf times the
        // patch field's gradient coefficients. The pair's are the interface's, and nothing here reads the
        // internal faces, so the whole-mesh assembly this was is not built.
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;   // the pair's are the interface's
            const std::vector<scalar> gIC = f.p_rgh.boundary[pi]->gradientInternalCoeffs();
            const std::vector<scalar> gBC = f.p_rgh.boundary[pi]->gradientBoundaryCoeffs();
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const scalar pGamma = rfB[pi][static_cast<std::size_t>(i)]
                                    * g.magSf()[static_cast<std::size_t>(fvp[pi].start + i)];
                i2.push_back(pGamma * gIC[static_cast<std::size_t>(i)]);
                b2.push_back((-pGamma) * gBC[static_cast<std::size_t>(i)]);
            }
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
        interPhase::Nested timed("hook pressure.boundaryValues");
        std::vector<scalar> flat;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            const std::vector<scalar>& v = f.p_rgh.boundary[pi]->value();
            flat.insert(flat.end(), v.begin(), v.end());
        }
        bval.copyFrom(flat);
    };
    // adjustPhi(phiHbyA, U, p_rgh) through the host's own function, on the boundary flux the device
    // step hands over: U's patch TYPES decide which faces are adjustable (adjustPhi reads no U values),
    // and those live on the host. Coupled patches are not in the device array and stay empty here, which
    // adjustPhi.C:57 skips anyway.
    H.pressure.adjustPhi = [&](const DeviceBuffer<scalar>& phiHI, DeviceBuffer<scalar>& phiHB)
    {
        SurfaceScalarField ph;
        phiHI.copyTo(ph.internal);
        unflatten(phiHB, ph.boundary);
        cpu::interFoam::adjustPhi(ph, f.U, true, fvp);
        phiHB.copyFrom(flattenPatches(ph.boundary, fvp));
    };
    H.pressure.updateBoundary = [&](const DeviceBuffer<scalar>& pr)
    {
        interPhase::Nested timed("hook pressure.updateBoundary");
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
    // IN A LAMBDA because a topology change re-runs it: OpenFOAM RE-SELECTS an option's cells at every
    // change (cellSetOption.C:383-396) and the host adapter does the same, so this list is a different
    // list afterwards -- 26,711 cells where it was 2,736 on the gate's porosity profile, the zone having
    // gained the seven children of every cell of it that was split. Uploaded once, the device applied the
    // resistance to an EIGHTH of the zone OpenFOAM applies it to: alpha 1.4218e-02 and U 9.5948e-02 from
    // the host arm, which is the same number as the host's own keep-the-labels control.
    const auto buildPorosity = [&]()
    {
        bool seen = false;
        for (const fvOptions::Option& o : f.fvOptions.options)
        {
            if (!o.active || !o.unsupported.empty() || o.fixedCoeff) continue;
            if (o.type != "explicitPorositySource") continue;
            if (seen)
            {
                throw std::runtime_error(
                    "brae interFoam (device): more than one active explicitPorositySource. The device step "
                    "carries one zone. The host loop carries them all.");
            }
            seen = true;
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
    };
    buildPorosity();

    // THE MANGROVE PAIR, from the HOST OptionList's own regions: each coefficient without its |U|, per
    // cell, in the host reference's multiplication order -- zero everywhere, then each region's cells
    // ASSIGNED in order, so a later region overwrites an earlier one on a shared cell as OpenFOAM's
    // loops do (fvOptions_cpp.cu, multiphaseMangrovesSource.C:36-101).
    DeviceMangroves dMangroves;
    {
        const std::size_t nCz = static_cast<std::size_t>(m.nCells());
        const scalar pi = 3.14159265358979323846;   // constant::mathematical::pi, M_PI
        for (const fvOptions::Option& o : f.fvOptions.options)
        {
            if (!o.active || !o.unsupported.empty()) continue;
            // ONE ENTRY PER OPTION. The list is a sum in OpenFOAM, so two options over the same
            // cellZone each contribute; one merged array would keep the last option's number alone.
            if (o.mangroves == fvOptions::Option::Mangroves::source)
            {
                std::vector<scalar> dragFac(nCz, scalar(0));
                std::vector<scalar> inertia(nCz, scalar(0));
                for (const fvOptions::Option::MangroveRegion& r : o.mangroveRegions)
                {
                    for (const label c : r.cells)
                    {
                        dragFac[static_cast<std::size_t>(c)] = 0.5*r.Cd*r.a*r.N;
                        inertia[static_cast<std::size_t>(c)] = 0.25*(r.Cm + 1)*pi*r.a*r.a*r.N;
                    }
                }
                DeviceMangroves::Source src;
                src.dragFac.copyFrom(dragFac);
                src.inertia.copyFrom(inertia);
                dMangroves.sources.push_back(std::move(src));
            }
            if (o.mangroves == fvOptions::Option::Mangroves::turbulence)
            {
                std::vector<scalar> kFac(nCz, scalar(0));
                std::vector<scalar> epsFac(nCz, scalar(0));
                for (const fvOptions::Option::MangroveRegion& r : o.mangroveRegions)
                {
                    for (const label c : r.cells)
                    {
                        kFac[static_cast<std::size_t>(c)] = r.Ckp*r.Cd*r.a*r.N;
                        epsFac[static_cast<std::size_t>(c)] = r.Cep*r.Cd*r.a*r.N;
                    }
                }
                DeviceMangroves::Turbulence t;
                t.kFac.copyFrom(kFac);
                t.epsFac.copyFrom(epsFac);
                dMangroves.turbulences.push_back(std::move(t));
            }
        }
    }

    // IN A LAMBDA for the same reason buildPorosity is: a topology change rebuilds every host zone's face
    // lists (MRF::update, mirroring MRFZone::update -> setMRFFaces) and every buffer here is derived from
    // them AND from the new geometry -- frameFluxInt and frameFluxBnd are Omega x (Cf - origin) dotted
    // with Sf, face by face, and a refined face has a new centre and a quarter of the area. The vector
    // itself is refilled rather than replaced, because C.mrf holds its address for the whole run.
    std::vector<DeviceMRFZone> dMrf;
    const auto buildMrf = [&]()
    {
        dMrf.clear();
        for (const cpu::MRF::Zone& z : f.mrfZones)
        {
            dMrf.push_back(buildDeviceMRFZone(z, m, g, fvp));
        }
    };
    buildMrf();

    // ---- controls --------------------------------------------------------------------------------
    // CRANKNICOLSON's state, the host driver's set on the device (inter_driver_cpp.cu, `cnDdt`): the
    // old-old level of U (cells and patches) and of alpha1 -- rho.oldTime().oldTime() is the mixture
    // at it -- rotated with the old ones; phi's, created when OpenFOAM creates it; the three ddt0
    // fields; and alphaPhi10's levels for alphaEqn.H's un-blend. The closure keeps its own.
    const bool cnDdt = (f.ddtU == DdtScheme::CrankNicolson);
    DeviceBuffer<scalar> dAOO(f.alpha1.internal);
    DeviceBuffer<scalar> dUoox(f.U.internal.size()), dUooy(f.U.internal.size()), dUooz(f.U.internal.size());
    DeviceBuffer<scalar> dUoobx, dUooby, dUoobz;
    // ...and rho's patch values at the same two levels, with the operands of ddt0(rho,U)'s patch half:
    // read only by the state's writer (the host loop's cnPatchRhoU)
    DeviceBuffer<scalar> dRhoOB, dRhoOOB;
    DeviceCnDdt0PatchOperands cnPatchRhoU;
    const bool keepCnPatches = writer && writer->writesCrankNicolson();
    // a cold start's U.oldTime(), whose fixed-value patches take what the FIRST assembly's updateCoeffs
    // left (the host loop's UOldCreationPending)
    bool uOldCreationPending = keepCnPatches;
    DeviceBuffer<scalar> dPhiOOI, dPhiOOB;
    std::vector<scalar> phiOldPrevI, phiOldPrevB;
    bool phiOOExists = false;
    // ...and whether anything has asked for phi.oldTime() yet -- the static ddtCorr does, the moving
    // one does not (it reads Uf.oldTime()), and the alpha blend asks last. See the host driver.
    bool phiOldRequested = false;
    // ...and the PAIR's, kept beside them for the same reason its old level is
    DeviceBuffer<scalar> dPhiOOIf;
    std::vector<scalar> phiOldPrevIf;
    bool phiOOIfExists = false;
    scalar deltaTPrev = f.deltaT;
    cpu::fv::CrankNicolsonClock cnClock;
    cnClock.ocCoeff = f.ddtOcCoeff;
    DeviceInterCrankNicolson dCn;
    dCn.clock = &cnClock;
    dCn.ddt0RhoU.name = "ddt0(rho,U)";
    dCn.ddtCorrU.name = "ddtCorrDdt0(U)";
    dCn.ddtCorrPhi.name = "ddtCorrDdt0(phi)";
    dCn.ddtCorrUf.name = "ddtCorrDdt0(Uf)";
    // A RESTART from a directory OpenFOAM wrote under CrankNicolson, the host driver's seed on the
    // device (inter_cn_restart.cuh): each ddt0 field with startTimeIndex -2, so the scheme is WARM on the
    // first step. deviceCnFvmDdt keeps no boundary for the momentum's field and deviceCnDdtCorr keeps one
    // over the non-coupled faces for its two, which is what the sizes below say.
    seedCnDdt0(dCn.ddt0RhoU, f.cnRestart, 3, static_cast<std::size_t>(nC), 0, fvp);
    seedCnDdt0(dCn.ddtCorrU, f.cnRestart, 3, static_cast<std::size_t>(nC),
               static_cast<std::size_t>(dm.nBndFaces), fvp);
    seedCnDdt0(dCn.ddtCorrPhi, f.cnRestart, 1, static_cast<std::size_t>(nIf),
               static_cast<std::size_t>(dm.nBndFaces), fvp);
    // ...AND phi's OLD-TIME LEVEL, out of phi_0. The loop keeps it as the host vectors it rotates from,
    // so the seed fills those and says the level exists: the first step then takes phi.oldTime().oldTime()
    // from the file rather than creating it as a copy of the flux beside it. `phiOldRequested` follows,
    // because on a restart the level is there before anything asks for it.
    {
        SurfaceScalarField phiOldFile;
        if (readCnOldOldSurface(f.cnRestart, "phi_0", nIf, fvp, phiOldFile))
        {
            phiOldPrevI = phiOldFile.internal;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                phiOldPrevB.insert(phiOldPrevB.end(),
                                   phiOldFile.boundary[pi].begin(), phiOldFile.boundary[pi].end());
            }
            phiOOExists = true;
            phiOldRequested = true;
        }
    }
    DeviceBuffer<scalar> dAlphaPhiEndI, dAlphaPhiEndB, dAlphaPhiOldI, dAlphaPhiOldB, dAlphaPhiOutI, dAlphaPhiOutB;
    bool alphaPhiOldExists = false;
    label alphaPhiOldIndex = -1;
    // ...and the PAIR's three, kept beside them: alphaPhi10 on the coupled faces lives in its own
    // array, so its old level and its end-of-step copy do too.
    DeviceBuffer<scalar> dAlphaPhiEndIf, dAlphaPhiOldIf, dAlphaPhiOutIf;
    label alphaPhiOldIfIndex = -1;
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
    // gradSchemes: every scalar gradient on this arm takes the case's own entry. grad(p_rgh) goes through
    // deviceGradOf at the corrected laplacian's explicit correction; alpha1's and alpha2's limiter
    // gradients through the alpha step (DeviceAlphaStepInput::gradAlpha1LeastSquares and its siblings);
    // nHat through deviceInterfaceCorrect (DeviceAlphaStepInput::nHatGradLeastSquares); grad(rho)'s one
    // consumer is snGrad(rho), which the interfaceForces hook takes on the HOST with f.gradRho. NOT
    // grad(pcorr): CorrectPhi runs on the host on both arms (correctPhi, inter_correct_phi_cpp.cu) and
    // takes the entry through correctPhiControlsOf -- gated on LES/nozzleFlow2D `pcorrGrad`, grad(pcorr)
    // leastSquares on a mesh non-orthogonal to 40 degrees with one non-orthogonal pass, where the entry
    // moves OpenFOAM's own U by 1.7e-05.
    //
    // RAS/electrostaticDeposition is the case that needs them: `gradSchemes { default cellLimited
    // leastSquares 1; }`, host arm RUNS. It is the only device refusal in this port that a stock
    // tutorial reaches, which is why the sites were lifted at all.
    C.alphaInput.nHatGradLeastSquares = f.interface.nHatGrad.leastSquares;
    C.alphaInput.nHatGradCellLimitK   = f.interface.nHatGrad.cellLimitK;
    // deviceInterfaceCorrect takes grad(alpha1) unsmoothed; the host's calculateK smooths a copy first
    // (interfaceProperties.C:119-131), and the boundary hook above runs THAT one, so without this the
    // two halves of one nHatf would come from two different fields
    if (f.interface.nAlphaSmoothCurvature > 0)
        throw std::runtime_error(
            "brae interFoam (device): nAlphaSmoothCurvature is not ported to the device curvature "
            "(deviceInterfaceCorrect). The host arm runs it. Refused rather than take the unsmoothed "
            "gradient.");
    // grad(U) under leastSquares: the momentum's dev2 term takes it (gpu::MomentumInput::
    // gradUSchemeLeastSq, which refuses it beside any site still Gauss), and so do the kEpsilon and
    // kOmegaSST closures (co.gradULeastSq, inter_turbulence_cpp.cu readGradU). The LES kEqn closure
    // does not: device_les_keqn.cu builds its production from deviceGradU alone.
    if (f.gradULeastSq && f.turbulence.model == cpu::interFoam::InterRasModel::KEqnLES)
        throw std::runtime_error(
            "brae interFoam (device): fvSchemes grad(U) resolves to leastSquares under LES kEqn, whose "
            "device closure takes grad(U) Gauss (device_les_keqn.cu). The host arm runs it. Refused "
            "rather than run another gradient.");
    C.alpha.preSolve.minIter = f.aSolve.minIter;   // gated on laminar/damBreak `alphaminiter`
    C.alpha.preSolve.tol = f.aSolve.tol;
    C.alpha.preSolve.relTol = f.aSolve.relTol;
    C.alpha.preSolve.maxIter = f.aSolve.maxIter;
    C.alpha.preSolve.smoothSolver = f.aSolve.gaussSeidel();
    C.alpha.preSolve.symmetric = (f.aSolve.smoother == "symGaussSeidel");
    C.alpha.preSolve.nSweeps = f.aSolve.nSweeps;
    C.mules = DeviceMulesControls{f.mulesCtl.nLimiterIter, f.mulesCtl.smoothLimiter,
                                  f.mulesCtl.extremaCoeff, f.mulesCtl.boundaryExtremaCoeff};
    C.alphaInput.cAlpha = f.interface.cAlpha;
    // the CONSTRUCTOR's deltaN, which buildInterFields took off the mesh before any motion -- the
    // same number the host arm's calculateK uses, and not this mesh's (InterfaceCoeffs::deltaN)
    C.alphaInput.deltaN = f.interface.deltaN;
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
    // localEuler: the pre-solve's fvm::ddt and CMULES read the local step (unless the gate's control
    // switches the consumer off, which is what the host loop's control does too)
    C.alphaInput.rDeltaT = (f.lts && !ltsScalar.count("alpha")) ? &dRDeltaT : nullptr;
    C.rDeltaTUEqn = (f.lts && !ltsScalar.count("ueqn")) ? &dRDeltaT : nullptr;
    C.rDeltaTfInt = (f.lts && !ltsScalar.count("ddtcorr")) ? &dRDeltaTfI : nullptr;
    C.rDeltaTfBnd = (f.lts && !ltsScalar.count("ddtcorr")) ? &dRDeltaTfB : nullptr;
    // fvSolution's cached grad(U): the host's field packed for the device -- validate() ran on the host, so the
    // registry the first assembly reuses is that one, and so is what the host-closure instrument leaves
    auto uploadHostGradU = [&]()
    {
        const std::size_t n = static_cast<std::size_t>(nC);
        std::vector<scalar> t9(9*n);
        for (std::size_t c = 0; c < n; ++c)
        {
            const scalar* q = &f.gradUCache.cells[c].xx;
            for (int k = 0; k < 9; ++k) t9[static_cast<std::size_t>(k)*n + c] = q[k];
        }
        std::size_t nB = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (!isCoupledInterfaceType(fvp[pi].type)) nB += static_cast<std::size_t>(fvp[pi].size);
        std::vector<scalar> b9(9*nB);
        std::size_t b = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            for (label i = 0; i < fvp[pi].size; ++i, ++b)
            {
                const scalar* q = &f.gradUCache.bnd[pi][static_cast<std::size_t>(i)].xx;
                for (int k = 0; k < 9; ++k) b9[static_cast<std::size_t>(k)*nB + b] = q[k];
            }
        }
        deviceSetGradU(dGradUCache, t9, b9, nC);
    };
    dGradUCache.on = f.gradUCache.on;
    C.gradUCache = dGradUCache.on ? &dGradUCache : nullptr;
    C.gradUUncachedControl = gradUUncached;
    C.gradUBndLiveControl = gradUBndLive;
    if (dGradUCache.on && f.gradUCache.valid)
    {
        uploadHostGradU();
    }
    C.alphaInput.alphaScheme  = alphaScheme(f.divPhiAlpha, "div(phi,alpha)");
    C.alphaInput.alpharScheme = alphaScheme(f.divPhirbAlpha, "div(phirb,alpha)");
    // ...and the gradient each limiter reads, by the field's OWN gradSchemes entry, as the host hands
    // fluxWithScheme gradAlpha1 for alpha1 and gradAlpha2 for alpha2 (alpha_eqn_cpp.cu:324-356)
    C.alphaInput.gradAlpha1LeastSquares = f.gradAlpha1.leastSquares;
    C.alphaInput.gradAlpha1CellLimitK   = f.gradAlpha1.cellLimitK;
    C.alphaInput.gradAlpha2LeastSquares = f.gradAlpha2.leastSquares;
    C.alphaInput.gradAlpha2CellLimitK   = f.gradAlpha2.cellLimitK;
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
    C.snGradPrghIf = (dCyc.n > 0) ? &dSnPrghIf : nullptr;
    C.nHatfIf     = (dCyc.n > 0) ? &dNHIf : nullptr;
    cycPhiForHost = (dCyc.n > 0) ? &dCyc.phi : nullptr;
    C.porosity = dPorosity.active ? &dPorosity : nullptr;
    C.mangroves = dMangroves.source() ? &dMangroves : nullptr;
    C.cn = cnDdt ? &dCn : nullptr;
    // p_rgh's reference, where the host driver sets it (inter_driver_cpp.cu:779-781). These three were
    // dead for as long as the device refused a case that needs one, and a step that never pins leaves the
    // singular system's level to the solver: MEASURED on laminar/mixerVessel2D before they were set, a
    // constant -2.17e+01 in p_rgh with a spread of only 1.5e-02 about it.
    C.needReference = f.pRef.needReference;
    C.pRefCell      = static_cast<int>(f.pRef.pRefCell);
    C.pRefValue     = f.pRef.pRefValue;
    C.nNonOrthogonalCorrectors = static_cast<int>(f.nNonOrthogonalCorrectors);
    // gradSchemes' `grad(U)`, which the momentum takes at TWO sites -- linearUpwind's reconstruction
    // and divDevReff's dev2 term -- and which this loop set at NEITHER. The host hands the same one
    // entry to both (inter_driver_cpp.cu:809, `mi.gradULimitK = f.gradULimitK`), and the case struct
    // carries one field because interFoam's tutorials name it once. Left at 0 the device ran the
    // viscous term on an unlimited gradient while the host limited it, which the refusal in front of
    // it made unreachable: the refusal-in-front-of-a-substitution shape again.
    C.gradULimitK       = f.gradULimitK;
    C.gradUSchemeLimitK = f.gradULimitK;
    C.gradUSchemeLeastSq = f.gradULeastSq;
    C.frozenFlow = f.pimple.frozenFlow;
    C.correctedLaplacian = f.laplacianScheme.corrected;
    // ...and grad(p_rgh)'s own gradSchemes entry, which the corrected laplacian's explicit correction is
    // built from. The host takes it through gradOf; the device now takes the same two numbers.
    C.prghGradLeastSquares = f.gradPrgh.leastSquares;
    C.prghGradCellLimitK   = f.gradPrgh.cellLimitK;
    C.nonOrthCoeffs = f.laplacianScheme.nonOrthCoeffs;
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
    DeviceAmgPcgCache amgPcgCache;
    amgPcgCache.mesh = &m;
    amgPcgCache.geometry = &g;
    amgPcgCache.caseDir = caseDir;
    C.amgPcg = &amgPcgCache;
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
        C.momentum.minIter = us.minIter;
        // minIter reaches BOTH device branches: deviceSymGaussSeidel takes it, and
        // deviceJacobiBiCGStab has taken it on both overloads all along (device_pcg.cuh:128-138) --
        // device_inter_step.cu simply never passed it. The refusal that stood here claimed that branch
        // had no iteration floor; it was wrong about brae's own code, and the fix was to pass the
        // argument rather than to refuse the case.
        C.momentum.smoothSolver = us.gaussSeidel();
        C.momentum.symmetric = (us.smoother == "symGaussSeidel");
        C.momentum.nSweeps = us.nSweeps;
    };
    setMomentumSolve(true);
    C.takeUAtBoundary = &dTakeU;
    C.uFluxIsRhoPhi = uNamesRhoPhiAny ? &dUNamesRhoPhi : nullptr;
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
    // A RESTART: the OLD-TIME level OpenFOAM reads back out of U_0, cells and patch values (see
    // inter_cn_restart.cuh). It goes into the OLD level and not the old-old one BECAUSE the rotation at
    // the top of this loop's first step moves it there -- U_0_0 = U_0 = the file, then U_0 = U at the
    // start time -- which is exactly what OpenFOAM's storeOldTime does on that step. Seeding the old-old
    // level directly would be overwritten by that rotation.
    if (cnDdt)
    {
        std::vector<vector> uOO;
        std::vector<std::vector<vector>> uOOBnd;
        if (readCnOldOld(f.cnRestart, "U_0", static_cast<std::size_t>(nC), fvp, uOO, uOOBnd))
        {
            std::vector<scalar> cx(nC), cy(nC), cz(nC), bx, by, bz;
            for (label c = 0; c < nC; ++c)
            { cx[c] = uOO[c].x; cy[c] = uOO[c].y; cz[c] = uOO[c].z; }
            dUox.copyFrom(cx); dUoy.copyFrom(cy); dUoz.copyFrom(cz);
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (const vector& v : uOOBnd[pi])
                {
                    bx.push_back(v.x); by.push_back(v.y); bz.push_back(v.z);
                }
            }
            dUobx.copyFrom(bx); dUoby.copyFrom(by); dUobz.copyFrom(bz);
        }
    }
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
    if (uNamesRhoPhiAny) deviceUpdateInletOutlet(dbU, uSwitchFlux(dPhiB));
    else                 deviceUpdateInletOutlet(dbU, dPhiB);

    // THE MESH UPDATE'S OWN OBJECTS, as the host driver keeps them (inter_driver_cpp.cu): the mesh's
    // GAMG hierarchy, which the motion solve builds and the run keeps, and the case's CorrectPhi
    // controls, which interMeshUpdate uses when `correctPhi` is on.
    const CorrectPhiControls meshCpc = withDevicePcorr(correctPhiControlsOf(f, meshAgglomeration));
    // the volumes the mesh had before this step's move; empty on a static mesh, where the ddt's
    // other branch runs
    DeviceBuffer<scalar> dV0;
    // ...and the volumes two steps back (fvMesh::V00), which only CrankNicolson's moving branch reads
    DeviceBuffer<scalar> dV00;
    // ...and the mesh flux the move produced, which the pressure corrector makes phi relative to
    DeviceBuffer<scalar> dMeshPhi;
    // Uf.oldTime() and the face flux ddtCorr reads off it, (Sf & Uf.oldTime()) per internal face.
    // The host driver keeps the same pair (UfOld, dc.UfOld); on a moving mesh OpenFOAM's ddtCorr
    // takes this in phi.oldTime()'s place (EulerDdtScheme's fvcDdtUfCorr).
    SurfaceVectorField UfOld = f.Uf;
    DeviceBuffer<scalar> dPhiUfOld;
    // ...and UNDER CRANKNICOLSON its old-old level and the four component arrays each of them needs on
    // the device: fvcDdtUfCorr is built from Uf.oldTime() and Uf.oldTime().oldTime() as VECTORS, on
    // the internal faces and the boundary faces. `UfOOExists` is the host driver's lazy creation --
    // GeometricField::oldTime() on a level that has never been stored returns the current one.
    SurfaceVectorField UfOO = f.Uf;
    bool UfOOExists = false;
    DeviceBuffer<scalar> dUfOld[3], dUfOldB[3], dUfOO[3], dUfOOB[3];
    // ...and the ABSOLUTE flux the pressure step leaves for fvc::correctUf below, which reads phi
    // one line before makeRelative turns it relative (pEqn.H:66 and :69).
    DeviceBuffer<scalar> dPhiAbsI, dPhiAbsB;
    // ...and on the pair's faces, with the mesh flux there (DeviceInterPressureInput::meshPhiIf)
    DeviceBuffer<scalar> dPhiAbsIf, dMeshPhiIf;
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
            ++meshGeometryEpoch;
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
    // Time::writeTime_ for the step being taken, decided at its start (as ++runTime is), and the alpha flux
    // its last alpha solve leaves, copied out on the device only on a write step
    bool writeNow = false;
    DeviceBuffer<scalar> dAlphaPhiWriteI, dAlphaPhiWriteB, dAlphaPhiWriteIf;
    // alpha.<phase1>_0 of a sub-cycled alpha: alpha as the step found it, downloaded only on a write step
    std::vector<scalar> alphaOldWrite;
    std::vector<std::vector<scalar>> alphaOldBndWrite;
    // ...and alpha1Bnd as the step's last sub-cycle began, copied on the device on a write step
    DeviceBuffer<scalar> dAlphaSubBndWrite;
    std::vector<std::vector<scalar>> alphaSubBndWrite;
    // continuityErrs.H after every corrector, for uniform/cumulativeContErr -- one divergence and one
    // read-back per corrector, and none at all without a writer. A periodic pair's faces are not in
    // deviceDiv's sum; they cancel in the volume-weighted total up to rounding.
    if (writer)
    {
        H.correctorDone = [&](
            const DeviceBuffer<scalar>& phiInt,
            const DeviceBuffer<scalar>& phiBnd)
        {
            interPhase::Nested timed("hook correctorDone");
            // continuityErrs.H takes the ABSOLUTE flux (pEqn.H:64, before makeRelative at :70), which a
            // moving mesh's step hands back in dPhiAbs* before making phi relative
            const bool absolute = C.phiAbsIntOut && C.phiAbsBndOut;
            DeviceBuffer<scalar> divPhi;
            deviceDiv(
                dm,
                absolute ? *C.phiAbsIntOut : phiInt,
                absolute ? *C.phiAbsBndOut : phiBnd,
                divPhi);
            // ...AND THE PAIR's faces, which fvc::div sums into their own cells. Across a plain cyclic
            // the two sides cancel in the volume-weighted total; across a cyclicAMI they do not (the
            // weighted stencil is not conservative face by face) and on a moving mesh the flux here is
            // the absolute one. MEASURED on RAS/mixerVesselAMI, one step, without it: cumulativeContErr
            // 2.9e+09 of OpenFOAM's in relative terms, with every field at 7e-13.
            if (dCyc.n > 0 && std::getenv("BRAE_CONTROL_DEVICE_CONTERR_NO_PAIR") == nullptr)
            {
                const bool absIf = C.phiAbsIfOut
                                && C.phiAbsIfOut->size() == static_cast<std::size_t>(dCyc.n);
                deviceCyclicAddDivFlux(dCyc, absIf ? *C.phiAbsIfOut : dCyc.phi, dm.V, divPhi);
            }
            std::vector<scalar> dv;
            divPhi.copyTo(dv);
            writer->addContinuityError(rep.deltaT, dv, g.V());
        };
    }

    CellFaces ltsCells;
    unsigned long long ltsCellsId = 0;
    bool ltsCellsBuilt = false;
    // setRDeltaT's wave on the device, in the host's order (device_fvc_smooth.cuh). MEASURED on RAS/DTCHull:
    // the host wave is 430 ms a step. BRAE_CONTROL_WAVE_HOST=1 runs the host wave, the identity gate's other arm.
    static const bool waveHost = std::getenv("BRAE_CONTROL_WAVE_HOST") != nullptr;
    DeviceSmoothWave ltsWave;
    interPhase::start();
    for (label s = 0; s < nSteps; ++s)
    {
        interPhase::mark("0 between steps (clock, write, downloads)");
        if (!(rep.time < endTime - scalar(0.5)*rep.deltaT)) break;   // Time::run(), Time.C:1000
        // whether Uf.oldTime() was STORED with its old-old level in existence, which is when OpenFOAM
        // starts writing Uf_0: the state the last step ended in (the host loop's UfOldStoredWithOO)
        const bool ufOldStoredWithOO = UfOOExists;

        // interFoam.C:98-100 -- CourantNo.H, alphaCourantNo.H, setDeltaT.H, and only THEN ++runTime.
        // Both numbers come off the flux the LAST step left behind and the step size still in force;
        // computing them after the step, or after deltaT moves, throttles on the wrong pair.
        //
        // OpenFOAM's two #includes each build their own surfaceSum(mag(phi)), and so do these two
        // calls. The host driver caches one and shares it, which is the single place it deliberately
        // differs from OpenFOAM; the device does not need to, so it does not.
        if (f.lts)
        {
            // setRDeltaT.H in the three includes' place (interFoam.C:94-103), before ++runTime, on the HOST
            // from this loop's fields: rhoPhi is the last alpha step's -- the host's createFields one at the
            // first step, before this loop has written any -- phi the last corrector's, rho the last
            // mixture's, alpha1 the cells and its STORED patch values (the alpha hook keeps f.alpha1's).
            // deltaT stays controlDict's: under localEuler it is 1 and no ddt reads it.
            SurfaceScalarField phiH = f.phi;
            dPhiI.copyTo(phiH.internal);
            unflatten(dPhiB, phiH.boundary);
            SurfaceScalarField rhoPhiH = f.rhoPhi;
            if (dRpI.size() > 0)
            {
                dRpI.copyTo(rhoPhiH.internal);
                unflatten(dRpB, rhoPhiH.boundary);
            }
            dA.copyTo(f.alpha1.internal);
            std::vector<scalar> rhoH = f.rho;
            if (dRho.size() == static_cast<std::size_t>(nC))
            {
                dRho.copyTo(rhoH);
            }
            SetRDeltaTInput ri;
            ri.rhoPhi = &rhoPhiH;
            ri.phi = &phiH;
            ri.alpha1 = &f.alpha1;
            ri.rho = &rhoH;
            // timeIndex > startTimeIndex + 1, before ++runTime: `s` steps of this run are done
            ri.damp = s > 1;
            // ...and the mesh's cell-to-face list, kept until the addressing changes (buildDeviceMesh stamps a
            // new addressingId at every topology change). BRAE_CONTROL_LTS_CELLS_REBUILT=1 rebuilds it every
            // step, as the call did before -- the identity check's other arm.
            if (!ltsCellsBuilt || ltsCellsId != dm.addressingId)
            {
                ltsCells = cellFaces(m);
                ltsCellsId = dm.addressingId;
                ltsCellsBuilt = true;
                ltsWave.built = false;
            }
            ri.cells = std::getenv("BRAE_CONTROL_LTS_CELLS_REBUILT") ? nullptr : &ltsCells;
            if (!waveHost)
            {
                ri.smooth = [&](std::vector<scalar>& field, scalar coeff)
                {
                    deviceSmooth(field, coeff, m, fvp, ltsCells, ltsWave);
                };
            }
            LocalEulerControls lec = f.ltsCtl;
            applySetRDeltaTControls(lec, ri.damp);
            interPhase::mark("0 before setRDeltaT (downloads)");
            const SetRDeltaTReport lr = setRDeltaT(f.rDeltaT, lec, ri, m, g, fvp);
            interPhase::mark("0 setRDeltaT (host)");
            rep.ltsLog.push_back(lr);
            rep.rDeltaTPerStep.push_back(f.rDeltaT);
            dRDeltaT.copyFrom(f.rDeltaT);
            // ddtCorr's face field, by the host's own interpolation (patch faces take the face cell's)
            const SurfaceScalarField rDeltaTf = interpolateRDeltaT(f.rDeltaT, m, g, fvp);
            dRDeltaTfI.copyFrom(rDeltaTf.internal);
            dRDeltaTfB.copyFrom(flattenPatches(rDeltaTf.boundary, fvp));
            if (verbose)
            {
                std::printf("  Flow time scale min/max = %.17g, %.17g\n", (double)lr.flowMin, (double)lr.flowMax);
                std::printf("  Smoothed flow time scale min/max = %.17g, %.17g\n",
                            (double)lr.smoothedMin, (double)lr.smoothedMax);
                if (lr.damped)
                {
                    std::printf("  Damped flow time scale min/max = %.17g, %.17g\n",
                                (double)lr.dampedMin, (double)lr.dampedMax);
                }
            }
        }
        else
        {
            rep.CoNum = deviceAlphaCourantNo(dm, dPhiI, dPhiB, nullptr, rep.deltaT).CoNum;
            rep.alphaCoNum = deviceAlphaCourantNo(dm, dPhiI, dPhiB, &dA, rep.deltaT).CoNum;
            rep.deltaT = setDeltaTVoF(rep.deltaT, rep.CoNum, rep.alphaCoNum, f.timeCtl,
                                      rep.time - startTime, &f.writeCadence);
        }

        // Uf.oldTime(), snapshotted where the host driver snapshots it (inter_driver_cpp.cu:897,
        // alongside UOld and phiOld). The FLUX off it is built after this step's move, below: the
        // field is the step's, the geometry it is dotted with is the mesh's as it stands then.
        // ON A DYNAMIC MESH, which is moving OR topo-changing: Uf exists whenever mesh.dynamic() does
        // (createUfIfPresent.H) and ddtCorr reads its old level on the same condition (fvcDdt.C:219).
        // Asking `dyn` here left an ADAPTIVE case's UfOld at the field Uf started the run as, for the
        // whole run -- alpha was exact (2.2e-15) and p_rgh 1.5e-02, U 2.2e-01 and phi 3.6e-01 out,
        // measured against the host arm on damBreakWithObstacle. The rotation is the whole fix.
        if (f.meshIsDynamic)
        {
            // storeOldTime rotates the level that EXISTS, before the old one is overwritten -- the
            // host driver does it at the end of the step, which is the same place: between the last
            // read of UfOld and its reassignment
            if (UfOOExists) UfOO = UfOld;
            UfOld = f.Uf;
            // Uf is built for a dynamic mesh only (inter_case_cpp.cu, createUfIfPresent.H), and the
            // loop below reads it face by face -- so its size is checked rather than assumed.
            // Reading past it is what a missing Uf would do QUIETLY, and phiHbyA is six orders
            // larger than the field it feeds when that happens.
            if (UfOld.internal.size() != static_cast<std::size_t>(nIf))
            {
                throw std::runtime_error(
                    "brae interFoam -device: the case has a dynamic mesh but Uf has " +
                    std::to_string(UfOld.internal.size()) + " internal faces, not " +
                    std::to_string(nIf) + ". ddtCorr reads (Sf & Uf.oldTime()) off it on a dynamic "
                    "mesh (EulerDdtScheme's fvcDdtUfCorr); refusing rather than reading past it.");
            }
        }

        // the old-old levels CrankNicolson reads take the old ones first (GeometricField::storeOldTime
        // recurses into the old level before overwriting it)
        if (cnDdt)
        {
            deviceCopy(dAOO, dAOld);
            deviceCopy(dUoox, dUox);
            deviceCopy(dUooy, dUoy);
            deviceCopy(dUooz, dUoz);
            if (dUobx.size())
            {
                deviceCopy(dUoobx, dUobx);
                deviceCopy(dUooby, dUoby);
                deviceCopy(dUoobz, dUobz);
            }
            // rho's two patch levels, rotated with them: the old one is what the last step left
            if (keepCnPatches)
            {
                DeviceBuffer<scalar> now(flattenPatches(stepRhoBnd.empty() ? f.rhoBnd : stepRhoBnd, fvp));
                deviceCopy(dRhoOOB, dRhoOB.size() ? dRhoOB : now);
                deviceCopy(dRhoOB, now);
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
            // ...and at the first step the old-old patch values, which are the old ones (a copy)
            if (cnDdt && dUoobx.size() == 0)
            {
                dUoobx.copyFrom(bx); dUooby.copyFrom(by); dUoobz.copyFrom(bz);
            }
        }
        std::vector<scalar> poi, pob;
        dPhiI.copyTo(poi); dPhiB.copyTo(pob);
        DeviceBuffer<scalar> dPhiOI(poi), dPhiOB(pob);
        // ...and the PAIR's flux at the same instant. fvc::ddtCorr compares phi.oldTime() with the
        // flux of U.oldTime() on every face a coupled patch included, and the pressure corrector
        // rewrites cyc.phi, so the snapshot has to be taken here with the other two.
        std::vector<scalar> poif;
        DeviceBuffer<scalar> dPhiOIf;
        if (dCyc.n > 0)
        {
            dCyc.phi.copyTo(poif);
            dPhiOIf.copyFrom(poif);
            C.phiOldIf = &dPhiOIf;
        }
        if (cnDdt)
        {
            // phi.oldTime().oldTime(): rotated once it exists; CREATED, as a copy of phi.oldTime(), on
            // the step whose first ddtCorr evaluates dphidt0 and so asks for it -- the host driver's
            // phiOOExists, with the measurement
            const label thisIndex = s + 1;
            if (phiOOExists)
            {
                dPhiOOI.copyFrom(phiOldPrevI);
                dPhiOOB.copyFrom(phiOldPrevB);
            }
            else if (dCn.ddtCorrPhi.exists && dCn.ddtCorrPhi.timeIndex != thisIndex)
            {
                dPhiOOI.copyFrom(poi);
                dPhiOOB.copyFrom(pob);
                phiOOExists = true;
            }
            else
            {
                dPhiOOI.copyFrom(poi);   // read by nothing on the step the field is created
                dPhiOOB.copyFrom(pob);
            }
            // ...and the PAIR's old-old level, rotated by the SAME test: one surfaceScalarField's
            // levels, split across two arrays because the device keeps the coupled faces apart.
            if (dCyc.n > 0)
            {
                if (phiOOIfExists)
                {
                    dPhiOOIf.copyFrom(phiOldPrevIf);
                }
                else
                {
                    dPhiOOIf.copyFrom(poif);
                    if (dCn.ddtCorrPhi.exists && dCn.ddtCorrPhi.timeIndex != thisIndex)
                    {
                        phiOOIfExists = true;
                    }
                }
                phiOldPrevIf = poif;
                dCn.phiOOIf = &dPhiOOIf;
            }
            phiOldPrevI = poi;
            phiOldPrevB = pob;
            dCn.phiOOInt = &dPhiOOI;
            dCn.phiOOBnd = &dPhiOOB;
            // ...and ON A DYNAMIC MESH the Uf pair fvcDdtUfCorr takes in phi's place, with the same
            // lazy creation (the host driver's UfOOExists): the level is CREATED, as a copy of
            // Uf.oldTime(), on the step whose first ddtCorr evaluates dUfdt0 and so asks for it.
            //
            // DYNAMIC, not moving: fvcDdt.C:219 routes ddtCorr(U, phi, Uf) on mesh.dynamic(), and a
            // REFINING mesh is dynamic. Asking `dyn` here left an adaptive case's CN ddtCorr on the PHI
            // branch, which CREATES ddtCorrDdt0(phi) -- a level OpenFOAM never makes on such a case, and
            // the level the host arm does not have either. The adaptive branch's own check caught it.
            if (f.meshIsDynamic)
            {
                if (!UfOOExists && dCn.ddtCorrUf.exists && dCn.ddtCorrUf.timeIndex != thisIndex)
                {
                    UfOO = UfOld;
                    UfOOExists = true;
                }
                if (!UfOOExists)
                {
                    UfOO = UfOld;   // read by nothing on the step the field is created
                }
                uploadSurfaceVector(UfOld, fvp, dUfOld, dUfOldB);
                uploadSurfaceVector(UfOO, fvp, dUfOO, dUfOOB);
                for (int k = 0; k < 3; ++k)
                {
                    dCn.UfOld[k] = &dUfOld[k];
                    dCn.UfOldBnd[k] = &dUfOldB[k];
                    dCn.UfOO[k] = &dUfOO[k];
                    dCn.UfOOBnd[k] = &dUfOOB[k];
                }
            }
            dCn.alpha1OO = &dAOO;
            dCn.UOO[0] = &dUoox; dCn.UOO[1] = &dUooy; dCn.UOO[2] = &dUooz;
            dCn.UOOBnd[0] = &dUoobx; dCn.UOOBnd[1] = &dUooby; dCn.UOOBnd[2] = &dUoobz;
            if (keepCnPatches)
            {
                cnPatchRhoU.rhoOld = &dRhoOB;
                cnPatchRhoU.rhoOO = &dRhoOOB;
                cnPatchRhoU.vfOld[0] = &dUobx; cnPatchRhoU.vfOld[1] = &dUoby; cnPatchRhoU.vfOld[2] = &dUobz;
                cnPatchRhoU.vfOO[0] = &dUoobx; cnPatchRhoU.vfOO[1] = &dUooby; cnPatchRhoU.vfOO[2] = &dUoobz;
                dCn.ddt0RhoUPatch = &cnPatchRhoU;
            }
            // Time::operator++: deltaT0_ = deltaTSave_; deltaTSave_ = deltaT_
            cnClock.timeIndex = thisIndex;
            cnClock.deltaT = rep.deltaT;
            cnClock.deltaT0 = deltaTPrev;
            deltaTPrev = rep.deltaT;
            // alphaEqn.H:18-56: the off-centring the scheme constructed for ddt(alpha) gives on this
            // step -- 0 before the scheme is warm -- and alphaPhi10.oldTime(), created by the first
            // un-blend as a copy of the current flux, then the flux the previous step ended on
            // ...with alphaRestart ORed into the warm-up test (alphaEqn.H:36-45), as the host loop has
            // it: a start directory that holds alphaPhi0 off-centres from the FIRST step
            dCn.ocAlpha = offCentringCoeff(f.ddtAlpha, f.alphaCtl.nAlphaSubCycles, f.ddtAlphaOcCoeff,
                                           f.cnAlphaRestart || thisIndex > 1);
            dCn.cnAlpha = blendingCoeff(dCn.ocAlpha);
            // ...and whether phi HAS an old-time level for that blend to use. On a MOVING mesh
            // ddtCorr is fvcDdtUfCorr and reads Uf.oldTime(), so nothing requests phi.oldTime()
            // before the alpha step -- which then creates it as a copy of the flux beside it, making
            // the blend inert for that one step (see the host driver's note at offCentredFlux).
            dCn.phiOldExists = phiOldRequested;
            if (dCn.ocAlpha > scalar(0)) phiOldRequested = true;
            // ...and on a mesh that is NOT DYNAMIC the pressure corrector's own ddtCorr (fvcDdtPhiCorr)
            // asks for phi.oldTime() later in this same step, so the level exists from the next one.
            //
            // DYNAMIC, NOT MOVING, and this is the EIGHTH site of that rule in this port. An ADAPTIVE mesh
            // has no motion solver, so `dyn` is null and this said "nothing else will ask" -- where the
            // host asks f.meshIsDynamic (inter_driver_cpp.cu) and does not. The consequence is one step of
            // the ALPHA equation: at step 2, the first step whose ocAlpha is non-zero, the device convected
            // with (1-cn)*phi.oldTime() + cn*phi where OpenFOAM's level is BORN there, as a copy of the
            // flux beside it, so the blend is inert for that one step. MEASURED on damBreakWithObstacle
            // under CrankNicolson with a change at step 2: max(alpha) 1.00008757432554 against the host's
            // 1.00000006354563, and alpha 6.9e-04 from OpenFOAM by the end. It hides everywhere else --
            // with no change that step, `phi` at that line already IS phi.oldTime() and the blend is inert
            // either way; under Euler ocAlpha is 0 and it never runs. The value is unchanged for a static
            // mesh (both predicates true) and for a moving one (both false).
            if (!f.meshIsDynamic) phiOldRequested = true;
            dCn.alphaPhiOldInt = nullptr;
            dCn.alphaPhiOldBnd = nullptr;
            if (dCn.ocAlpha > scalar(0))
            {
                if (alphaPhiOldExists)
                {
                    if (alphaPhiOldIndex != thisIndex)
                    {
                        deviceCopy(dAlphaPhiOldI, dAlphaPhiEndI);
                        deviceCopy(dAlphaPhiOldB, dAlphaPhiEndB);
                        alphaPhiOldIndex = thisIndex;
                    }
                    dCn.alphaPhiOldInt = &dAlphaPhiOldI;
                    dCn.alphaPhiOldBnd = &dAlphaPhiOldB;
                }
                // ...else null: the step's first un-blend creates the level from the current flux
            }
            dCn.alphaPhiOutInt = &dAlphaPhiOutI;
            dCn.alphaPhiOutBnd = &dAlphaPhiOutB;
            dCn.alphaPhiCreatedInt = &dAlphaPhiOldI;
            dCn.alphaPhiCreatedBnd = &dAlphaPhiOldB;
            // ...and the PAIR's own level of the same field, rotated at the same instant. Its faces
            // are in neither of the two arrays above, and the un-blend is face-local.
            if (dCyc.n > 0)
            {
                dCn.alphaPhiOldIf = nullptr;
                if (dCn.ocAlpha > scalar(0) && alphaPhiOldExists)
                {
                    if (alphaPhiOldIfIndex != thisIndex)
                    {
                        deviceCopy(dAlphaPhiOldIf, dAlphaPhiEndIf);
                        alphaPhiOldIfIndex = thisIndex;
                    }
                    dCn.alphaPhiOldIf = &dAlphaPhiOldIf;
                }
                dCn.alphaPhiOutIf = &dAlphaPhiOutIf;
                dCn.alphaPhiCreatedIf = &dAlphaPhiOldIf;
            }
        }

        // OpenFOAM's clock for the step about to be taken: ++runTime comes before the alpha step
        stepDeltaT = rep.deltaT;
        stepTime = rep.time + rep.deltaT;
        acmiRescaledThisStep = false;   // once per TIME STEP, not per outer corrector
        stepIndex = s + 1;
        // Time::operator++ moves writeTimeIndex_ with the step's time and deltaT, BEFORE the step -- what
        // the next adjustDeltaT measures from, and under runTime/adjustableRunTime the write flag
        {
            const bool indexMoved = f.writeCadence.advance(stepTime - startTime, rep.deltaT);
            writeNow = false;
            if (writer)
            {
                writer->stepTaken(rep.deltaT);
                writeNow = writer->isWriteTime(writer->startTimeIndex() + stepIndex, indexMoved);
            }
            if (writeNow && writer->writesAlphaOld())
            {
                dA.copyTo(alphaOldWrite);
                alphaOldBndWrite.assign(f.alpha1.boundary.size(), std::vector<scalar>());
                for (std::size_t pi = 0; pi < f.alpha1.boundary.size(); ++pi)
                {
                    alphaOldBndWrite[pi] = f.alpha1.boundary[pi]->value();
                }
            }
            C.alphaPhiWriteInt = writeNow ? &dAlphaPhiWriteI : nullptr;
            C.alphaPhiWriteBnd = writeNow ? &dAlphaPhiWriteB : nullptr;
            C.alphaPhiWriteIf  = writeNow ? &dAlphaPhiWriteIf : nullptr;
            C.alphaSubCycleBndWrite = (writeNow && writer->writesAlphaOld()) ? &dAlphaSubBndWrite : nullptr;
        }

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
        // ...and rho.oldTime().oldTime() for its CrankNicolson ddt: the mixture at alpha1's old-old
        // level, rebuilt as the step rebuilds rho.oldTime() from alpha1.oldTime()
        DeviceBuffer<scalar> dRhoOO;
        if (cnDdt && deviceClosure && f.turbulence.variableDensity)
        {
            dRhoOO.resize(static_cast<std::size_t>(nC));
            deviceMixtureCorrect(dAOO.data(), nC, props, nullptr, dRhoOO.data(), nullptr, nullptr);
        }
        if (cnDdt && hostClosure)
            throw std::runtime_error(
                "brae interFoam (device): BRAE_INTER_HOST_CLOSURE under CrankNicolson is not carried "
                "(the instrument keeps no rho.oldTime().oldTime() for the host closure).");

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
            // interFoam.C:112: `if (pimple.firstIter() || moveMeshOuterCorrectors)` -- the move, CorrectPhi and
            // the geometry refresh below run on the first outer corrector only unless the case asks for more,
            // as the host stage returns early (inter_driver_cpp.cu, interMeshUpdate) and the adaptive branch
            // below is guarded. Unguarded, this block ran at every corrector: the host phi it copies under
            // correctPhi is the one the host skipped rewriting (stale), and V0 was re-taken from the MOVED
            // volumes -- neither seen by a gate while every device-arm moving profile ran one corrector.
            // A MESH THAT REFINES AND MOVES (laminar/oscillatingBox): dynamicRefineFvMesh::update changes the
            // topology FIRST and moves the points AFTER (dynamicRefineFvMesh.C:1468-1474), and interFoam.C:
            // 112-148 then runs one block on the moved mesh. So the motion below is a block this corrector
            // runs either on its own or AFTER the adaptive branch -- the host loop's meshUpdate stage.
            const bool refineAndMove = dyn && f.amr && f.amr->active
                                    && std::getenv("BRAE_CONTROL_DEVICE_REFINE_MOVE_REFUSED") == nullptr;
            const bool moveThisCorrector = dyn && (outer == 0 || f.moveMeshOuterCorrectors || moveEveryOuter);
            auto moveMeshNow = [&]()
            {
                if (v0FromV)
                {
                    dV0.copyFrom(dm.V.host());
                }
                // THE BODY'S LOAD reads U (its cells beside the body's patches, for the shear) and nuEff
                // = nut + nu on those patches, as they stand now. The closure's nut is the device's on
                // this arm, and the host copy is whatever the build uploaded; U's cells are the device's
                // after the last corrector. Stale, either would move the body by the wrong load with
                // nothing saying so. BRAE_CONTROL_DEVICE_BODY_STALE=1 skips both -- the gate's control.
                if (dyn->rigidBody() && std::getenv("BRAE_CONTROL_DEVICE_BODY_STALE") != nullptr)
                {
                    std::printf("  *** CONTROL MODE: the body's load is taken from the host copies as they "
                                "stand, stale. This run is deliberately wrong. ***\n");
                }
                if (dyn->rigidBody() && std::getenv("BRAE_CONTROL_DEVICE_BODY_STALE") == nullptr)
                {
                    if (deviceClosure && dTurb.k.size() == static_cast<std::size_t>(nC))
                    {
                        downloadDeviceInterTurbulence(dTurb, f.turbulence, fvp);
                    }
                    std::vector<scalar> cx;
                    std::vector<scalar> cy;
                    std::vector<scalar> cz;
                    dUx.copyTo(cx);
                    dUy.copyTo(cy);
                    dUz.copyTo(cz);
                    for (label c = 0; c < nC; ++c)
                    {
                        const std::size_t cs = static_cast<std::size_t>(c);
                        f.U.internal[cs] = vector{cx[cs], cy[cs], cz[cs]};
                    }
                }
                std::vector<scalar> cpDiv;
                {
                    interPhase::Nested timedHost("mesh: the host update, whole (motion, CorrectPhi, mixture)");
                    interMeshUpdate(dyn, f, m, g, fvp, mutableMesh, mutableMesh ? mutableMesh->ami : nullptr,
                                    meshAgglomeration, meshCpc, rep, stepTime, stepIndex, outer,
                                    f.pimple.nOuterCorrectors,
                                    f.ddtU == DdtScheme::CrankNicolson ? &cnClock : nullptr,
                                    writer ? &cpDiv : nullptr);
                }
                interPhase::Nested timedRefresh("mesh: the device refresh after it (geometry, fluxes, boundary)");
                // correctPhi.H:11 -- the mesh update's CorrectPhi is host code on this arm too
                if (writer && !cpDiv.empty())
                {
                    writer->addContinuityError(rep.deltaT, cpDiv, g.V());
                }
                // storeOldVol -- OF fvMesh::movePoints:944, the volumes the cells had when the TIME STEP
                // began, stored ONCE per time index inside update() (dynamic_motion_solver_fv_mesh_cpp.cu).
                // Not dm.V before the call: under moveMeshOuterCorrectors the second update's dm.V is the
                // first update's moved volume.
                if (!v0FromV)
                {
                    dV0.copyFrom(dyn->V0());
                }
                C.V0 = &dV0;
                // ...and now every buffer this loop uploaded from the geometry. clearGeom +
                // clearOut on the device side: the addressing is untouched, as the move keeps the
                // topology fixed (device_mesh.cuh).
                // ...and mesh().V00(), which CrankNicolson's moving branch weights the old-old level
                // by. Asked for AFTER the move, as the host arm asks at its momentum assembly: the
                // mesh rotates V00 <- V0 inside update() and creates it on first use, so the two arms
                // get the same array whichever asks first.
                if (cnDdt)
                {
                    dV00.copyFrom(dyn->V00());
                    C.V00 = &dV00;
                }
                refreshDeviceMeshGeometry(dm, m, g, fvp);
                ++meshGeometryEpoch;
                // ...and THE PAIR: the host stage above has recomputed every cyclicAMI's weights on the
                // moved points and coupled its patches again (cyclicAMIFvPatch::Interfaces::update)
                if (dCyc.n > 0)
                {
                    cyclics = buildCyclicInterfaces(m, g, fvp, /*includeCoupledACMI=*/true,
                                                    /*includeCoupledAMI=*/true);
                    if (std::getenv("BRAE_CONTROL_DEVICE_PAIR_STALE") == nullptr)
                    {
                        refreshDeviceCyclicAfterMove(dCyc, cyclics, g, fvp);
                    }
                    else
                    {
                        std::printf("  *** CONTROL MODE: the device keeps the pair it was built with after the "
                                    "mesh moves. This run is deliberately wrong. ***\n");
                    }
                }
                dMagSf.copyFrom(magSfAll());
                dGh.copyFrom(f.gh);
                {
                    SurfaceScalarField gf;
                    gf.internal = f.ghfInternal;
                    gf.boundary = f.ghfBoundary;
                    dGhf.copyFrom(fullFace(gf, fvp));
                    if (dCyc.n > 0) dGhfIf.copyFrom(coupledFace(gf, cyclics));
                }
                // the patch geometry moved with the cells, and dbU carries the patch deltas and
                // normals every boundary evaluation reads
                dbU = buildDeviceVectorBoundary(f.U, fvp, g);
                // EVERY DISTANCE THE CLOSURE HOLDS WAS MEASURED ON THE OLD MESH. OpenFOAM's wallDist
                // and nearWallDist are MeshObjects that fvMesh::movePoints updates; this loop ran
                // neither, on either closure path, so kOmegaSST's F1/F2 blended on the distance the
                // cells had at time zero and every wall function read a y the wall had moved away
                // from. The host block goes first -- moveInterTurbulence re-runs wallDist's own
                // method on the moved points -- and the device arrays follow it.
                if (f.turbulence.on)
                {
                    moveInterTurbulence(f.turbulence, m, g, fvp, stepIndex);
                    if (deviceClosure)
                    {
                        refreshDeviceInterTurbulenceGeometry(dTurb, f.turbulence, f.U, m, g, fvp);
                    }
                }
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
                    // ...and both on the pair's faces: CorrectPhi's flux, already relative, and nHatf
                    if (dCyc.n > 0)
                    {
                        dCyc.phi.copyFrom(coupledFace(f.phi, cyclics));
                        dNHIf.copyFrom(coupledFace(f.nHatf, cyclics));
                    }
                }
                // ...and THE MESH FLUX the move produced, which the pressure corrector makes phi
                // relative to (fvc::makeRelative, pEqn.H:73). Over the full face array, as phi is.
                dMeshPhi.copyFrom(fullFace(fvcMeshPhi(*dyn, f), fvp));
                C.meshPhiAll = &dMeshPhi;
                if (dCyc.n > 0)
                {
                    dMeshPhiIf.copyFrom(coupledFace(fvcMeshPhi(*dyn, f), cyclics));
                    C.meshPhiIf = &dMeshPhiIf;
                }
            };
            if (moveThisCorrector && !refineAndMove)
            {
                moveMeshNow();
            }
            // ...AND THE ADAPTIVE MESH, which is the same line of interFoam.C for a different
            // dynamicFvMesh: mesh.update() selects, refines, unrefines and MAPS EVERY FIELD, and
            // everything the solver rebuilds afterwards is what `changed` gates. `dyn` is null here --
            // buildInterFields hands an adaptive case to no motion factory, and a case that asks for
            // both is refused by name in readInterAmr -- so the two branches are exclusive.
            //
            // THE CHANGE IS HOST WORK ON EITHER ARM, exactly as the motion solve and CorrectPhi are:
            // topology surgery, six integer maps and one autoMap per patch field, none of it per-cell
            // arithmetic (manifest interFoam_dynamicRefine, HOST_ONLY). What this loop owes is the
            // round trip -- the fields the mapper carries come down, the change happens, and every
            // mesh-sized buffer goes back up. The moving branch above owes only the GEOMETRY, because
            // a move keeps the addressing; a topology change keeps nothing, so this rebuilds the
            // DeviceMesh itself and with it every schedule cache that keys on its addressingId.
            if (f.amr && f.amr->active && (outer == 0 || f.moveMeshOuterCorrectors))
            {
                // ---- DOWN: the state the mapper carries, as the device holds it now.
                //
                // A BUFFER THIS LOOP HAS NOT WRITTEN YET IS EMPTY, and then the HOST's copy is the one
                // both arms hold -- buildInterFields wrote it and nothing has touched it. rhoPhi's is
                // the case: the alpha step writes it, and at the first change no alpha step has run.
                // Reading it anyway walked off the end of the array (SIGSEGV in unflatten, frame one a
                // memcpy). Any size that is neither empty nor the mesh's is a defect and says so.
                const auto ready = [&](const char* what, std::size_t have, label want) -> bool
                {
                    if (have == 0) return false;
                    if (have != static_cast<std::size_t>(want))
                        throw std::runtime_error(
                            std::string("brae interFoam (device, adaptive mesh): ") + what + " is "
                            + std::to_string(have) + " long where the mesh has " + std::to_string(want)
                            + ". A buffer at another mesh's size cannot be mapped.");
                    return true;
                };
                // THE TURBULENCE STATE FIRST, whole. On this arm the closure lives in device buffers and
                // the HOST copy is whatever the build uploaded -- never updated, because nothing on this
                // loop calls the host correct(). The mapper maps the HOST copy, so without this download a
                // change would map the initial fields and hand them back as the step's. It brings k, the
                // second scalar, nut, their patch values and the four old-time arrays.
                if (f.turbulence.on && ready("turbulence k", dTurb.k.size(), nC))
                {
                    downloadDeviceInterTurbulence(dTurb, f.turbulence, fvp);
                }
                if (f.turbulence.on && std::getenv("BRAE_AMR_TRACE"))
                {
                    const auto mnmx = [](const std::vector<scalar>& v)
                    {
                        scalar lo = v.empty() ? scalar(0) : v[0], hi = lo;
                        for (const scalar x : v) { lo = std::fmin(lo, x); hi = std::fmax(hi, x); }
                        return std::pair<scalar, scalar>{lo, hi};
                    };
                    const auto kk = mnmx(f.turbulence.k.internal);
                    const auto nn = mnmx(f.turbulence.nut.internal);
                    std::printf("  [trace] DOWNLOADED before the change: k %zu [%.6g, %.6g]  nut %zu "
                                "[%.6g, %.6g]  dTurb.k %zu  kOldStep %zu\n",
                                f.turbulence.k.internal.size(), (double)kk.first, (double)kk.second,
                                f.turbulence.nut.internal.size(), (double)nn.first, (double)nn.second,
                                (std::size_t)dTurb.k.size(), f.turbulence.kOldStep.size());
                }
                if (ready("alpha", dA.size(), nC))          dA.copyTo(f.alpha1.internal);
                if (ready("p_rgh", dPrgh.size(), nC))       dPrgh.copyTo(f.p_rgh.internal);
                if (ready("nHatf", dNH.size(), nIf))        dNH.copyTo(f.nHatf.internal);
                if (ready("K", dK.size(), nC))              dK.copyTo(f.K);
                if (ready("rhoPhi", dRpI.size(), nIf))      dRpI.copyTo(f.rhoPhi.internal);
                if (ready("phi", dPhiI.size(), nIf))        dPhiI.copyTo(f.phi.internal);
                if (ready("phi's patches", dPhiB.size(), nBf))     unflatten(dPhiB, f.phi.boundary);
                if (ready("nHatf's patches", dNHB.size(), nBf))    unflatten(dNHB, f.nHatf.boundary);
                if (ready("rhoPhi's patches", dRpB.size(), nBf))   unflatten(dRpB, f.rhoPhi.boundary);
                {
                    std::vector<scalar> cx, cy, cz;
                    dUx.copyTo(cx); dUy.copyTo(cy); dUz.copyTo(cz);
                    for (label c = 0; c < nC; ++c)
                    {
                        f.U.internal[static_cast<std::size_t>(c)] =
                            vector{cx[static_cast<std::size_t>(c)], cy[static_cast<std::size_t>(c)],
                                   cz[static_cast<std::size_t>(c)]};
                    }
                    f.U.evaluateBoundary();
                }
                // alpha's PATCH VALUES are the device's too, and they are what alpha's own patch field
                // maps -- setValue writes value_ alone, which is the only part that was ever uploaded.
                if (ready("alpha's patches", dABnd.size(), nBf))
                {
                    std::vector<std::vector<scalar>> ab;
                    unflatten(dABnd, ab);
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        if (isCoupledInterfaceType(fvp[pi].type)) continue;
                        f.alpha1.boundary[pi]->setValue(ab[pi]);
                    }
                }
                // ...and THE OLD-TIME LEVELS, which this loop keeps in device buffers where the host
                // loop keeps them as locals. They go through the mapper as host vectors and come back
                // mapped. At this point in the step they are bit-copies of their own fields -- the
                // snapshot above the outer loop has just stored them -- which is why the host gate's
                // re-capture control reads 0.0e+00; they are mapped anyway, because relying on that
                // equality is how the next change (moveMeshOuterCorrectors) would go wrong in silence.
                std::vector<scalar> aOldH = f.alpha1.internal;
                if (ready("alpha.oldTime", dAOld.size(), nC)) dAOld.copyTo(aOldH);
                std::vector<vector> uOldH = f.U.internal;
                if (ready("U.oldTime", dUox.size(), nC))
                {
                    std::vector<scalar> ox, oy, oz;
                    dUox.copyTo(ox); dUoy.copyTo(oy); dUoz.copyTo(oz);
                    for (label c = 0; c < nC; ++c)
                    {
                        uOldH[static_cast<std::size_t>(c)] =
                            vector{ox[static_cast<std::size_t>(c)], oy[static_cast<std::size_t>(c)],
                                   oz[static_cast<std::size_t>(c)]};
                    }
                }
                std::vector<std::vector<vector>> uOldBndH(fvp.size());
                for (std::size_t pi = 0; pi < fvp.size(); ++pi) uOldBndH[pi] = f.U.boundary[pi]->value();
                if (ready("U.oldTime's patches", dUobx.size(), nBf))
                {
                    std::vector<scalar> bx, by, bz;
                    dUobx.copyTo(bx); dUoby.copyTo(by); dUobz.copyTo(bz);
                    std::size_t off = 0;
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        if (isCoupledInterfaceType(fvp[pi].type)) continue;
                        const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
                        if (off + n > bx.size()) break;
                        uOldBndH[pi].resize(n);
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            uOldBndH[pi][i] = vector{bx[off + i], by[off + i], bz[off + i]};
                        }
                        off += n;
                    }
                }
                SurfaceScalarField phiOldH = f.phi;
                if (ready("phi.oldTime", dPhiOI.size(), nIf)) dPhiOI.copyTo(phiOldH.internal);
                if (ready("phi.oldTime's patches", dPhiOB.size(), nBf))
                    unflatten(dPhiOB, phiOldH.boundary);

                // rho.oldTime() FOR THE CLOSURE, and it is the defect this unit's gate found. dRhoOld is
                // captured at the TOP of the step from the previous step's dRho (above), i.e. BEFORE the
                // change -- so after one it is still at the old cell count while every other closure input
                // is at the new one. Nothing else reads it: the laminar path has no ddt(rho, k), so the
                // buffer stayed at 2268 against a mesh of 2814 for as long as an adaptive turbulent case
                // was refused. MEASURED, before this was carried: EVERY ONE of the 546 cells the change
                // added (78 refined x 7 children) came out of the closure's first solve with k collapsed
                // to 1.2e-12 where OpenFOAM's minimum is 0.0922, and nut followed it to 3.8e-25 against
                // 9.06e-05 -- because the kernel read past the end of rhoOld for exactly those cells.
                // It is CARRIED rather than recomputed from the mapped alpha.oldTime(), because OpenFOAM's
                // rho.oldTime() is an autoMapped old-time level of a registered field and that is what the
                // host arm hands the mapper too.
                std::vector<scalar> rhoOldH;
                if (deviceClosure && ready("rho.oldTime", dRhoOld.size(), nC))
                {
                    dRhoOld.copyTo(rhoOldH);
                }

                InterAmrOldTime oldT;
                oldT.alphaOld = &aOldH;
                oldT.UOld = &uOldH;
                oldT.UOldBnd = &uOldBndH;
                oldT.phiOld = &phiOldH;
                oldT.UfOld = &UfOld;
                oldT.rhoOld = rhoOldH.empty() ? nullptr : &rhoOldH;

                // ---- AND THE CrankNicolson STATE, which on THIS arm lives in device buffers where the
                // host loop keeps it in locals. Every level is a registered field in OpenFOAM and is
                // autoMapped with the rest (DDt0Field), so the same InterAmrCn carries them: they come
                // down component by component, go through the mapper as host objects, and go back up.
                //
                // `exists`, `startTimeIndex` and `timeIndex` travel WITH each level and are not re-seeded:
                // they are the field's own state in OpenFOAM too, and re-seeding them restarts the scheme
                // as Euler at each change -- which is exactly what the gate's control does on purpose.
                fv::CrankNicolsonDdt0<vector> hRhoU, hCorrU, hCorrUf;
                std::vector<scalar> alphaOOH;
                std::vector<vector> uOOH;
                std::vector<std::vector<vector>> uOOBndH(fvp.size());
                SurfaceScalarField phiOOH, alphaPhiEndH, alphaPhiOldH;
                InterAmrCn cnState;
                const auto downComponents = [&](const DeviceBuffer<scalar>* comp[3],
                                                std::vector<vector>& out,
                                                label n)
                {
                    std::vector<scalar> c[3];
                    for (int k = 0; k < 3; ++k) comp[k]->copyTo(c[k]);
                    out.assign(static_cast<std::size_t>(n), vector{0, 0, 0});
                    for (label i = 0; i < n; ++i)
                    {
                        const std::size_t ii = static_cast<std::size_t>(i);
                        out[ii] = vector{c[0][ii], c[1][ii], c[2][ii]};
                    }
                };
                const auto splitPatches = [&](const std::vector<vector>& flat,
                                              std::vector<std::vector<vector>>& out)
                {
                    out.assign(fvp.size(), std::vector<vector>());
                    std::size_t off = 0;
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        if (isCoupledInterfaceType(fvp[pi].type)) continue;
                        const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
                        if (off + n > flat.size()) break;
                        out[pi].assign(flat.begin() + off, flat.begin() + off + n);
                        off += n;
                    }
                };
                const auto downDdt0 = [&](const DeviceCnDdt0& d,
                                          fv::CrankNicolsonDdt0<vector>& h,
                                          label nInternal)
                {
                    h.name = d.name;
                    h.exists = d.exists;
                    h.startTimeIndex = d.startTimeIndex;
                    h.timeIndex = d.timeIndex;
                    h.internal.clear();
                    h.boundary.clear();
                    if (!d.exists) return;
                    const DeviceBuffer<scalar>* ci[3] = {&d.internal[0], &d.internal[1], &d.internal[2]};
                    if (ready((h.name + "'s cells").c_str(), d.internal[0].size(), nInternal))
                    {
                        downComponents(ci, h.internal, nInternal);
                    }
                    // a level created by the fvmDdt path carries NO boundary at all, which is brae's own
                    // shape and not a shortcut (crank_nicolson_ddt_scheme_cpp.cu:91 against :257 and :406)
                    if (d.boundary[0].size() == 0) return;
                    if (!ready((h.name + "'s patches").c_str(), d.boundary[0].size(), nBf)) return;
                    const DeviceBuffer<scalar>* cb[3] = {&d.boundary[0], &d.boundary[1], &d.boundary[2]};
                    std::vector<vector> flat;
                    downComponents(cb, flat, nBf);
                    splitPatches(flat, h.boundary);
                };
                if (cnDdt)
                {
                    downDdt0(dCn.ddt0RhoU, hRhoU, nC);
                    downDdt0(dCn.ddtCorrU, hCorrU, nC);
                    downDdt0(dCn.ddtCorrUf, hCorrUf, nIf);
                    if (ready("alpha.oldTime().oldTime()", dAOO.size(), nC)) dAOO.copyTo(alphaOOH);
                    {
                        const DeviceBuffer<scalar>* ci[3] = {&dUoox, &dUooy, &dUooz};
                        if (ready("U.oldTime().oldTime()", dUoox.size(), nC))
                        {
                            downComponents(ci, uOOH, nC);
                        }
                        const DeviceBuffer<scalar>* cb[3] = {&dUoobx, &dUooby, &dUoobz};
                        if (dUoobx.size() && ready("U.oldTime().oldTime()'s patches", dUoobx.size(), nBf))
                        {
                            std::vector<vector> flat;
                            downComponents(cb, flat, nBf);
                            splitPatches(flat, uOOBndH);
                        }
                    }
                    // phi's old-old LEVEL is the device buffer, NOT the host vector beside it.
                    // `phiOldPrevI` is this loop's bookkeeping for the NEXT step's level and has ALREADY
                    // been advanced to this step's phiOld by the time this branch runs, so reading it here
                    // carried phi(end of k-1) where the level holds phi(end of k-2). MEASURED by the trace:
                    // at step 3 it gave phiOO 7.40256964e-05 against the host arm's 0.000325582339, which
                    // is phiOld and not the old-old level at all.
                    if (ready("phi.oldTime().oldTime()", dPhiOOI.size(), nIf))
                    {
                        dPhiOOI.copyTo(phiOOH.internal);
                        phiOOH.boundary.assign(fvp.size(), std::vector<scalar>());
                        if (dPhiOOB.size()) unflatten(dPhiOOB, phiOOH.boundary);
                    }
                    const auto downFlux = [&](const DeviceBuffer<scalar>& di,
                                              const DeviceBuffer<scalar>& db,
                                              SurfaceScalarField& out,
                                              const char* what)
                    {
                        if (!ready(what, di.size(), nIf)) return;
                        di.copyTo(out.internal);
                        out.boundary.assign(fvp.size(), std::vector<scalar>());
                        if (db.size()) unflatten(db, out.boundary);
                    };
                    downFlux(dAlphaPhiEndI, dAlphaPhiEndB, alphaPhiEndH, "alphaPhi10 at the step's end");
                    downFlux(dAlphaPhiOldI, dAlphaPhiOldB, alphaPhiOldH, "alphaPhi10.oldTime()");

                    cnState.ddt0RhoU = &hRhoU;
                    cnState.ddtCorrU = &hCorrU;
                    cnState.ddtCorrUf = &hCorrUf;
                    cnState.ddtCorrPhi = nullptr;   // never created on a dynamic mesh; dCn's is checked below
                    cnState.UfOO = &UfOO;
                    cnState.phiOO = phiOOH.internal.empty() ? nullptr : &phiOOH;
                    cnState.alphaPhiEnd = alphaPhiEndH.internal.empty() ? nullptr : &alphaPhiEndH;
                    cnState.alphaPhiOld = alphaPhiOldH.internal.empty() ? nullptr : &alphaPhiOldH;
                    oldT.alphaOO = alphaOOH.empty() ? nullptr : &alphaOOH;
                    oldT.UOO = uOOH.empty() ? nullptr : &uOOH;
                    oldT.UOOBnd = uOOBndH[0].empty() && uOOBndH.size() ? nullptr : &uOOBndH;
                    if (dCn.ddtCorrPhi.exists)
                        throw std::runtime_error(
                            "brae interFoam (device, adaptive mesh): ddtCorrDdt0(phi) exists. ddtCorr takes "
                            "the Uf branch on a dynamic mesh (fvcDdt.C:219), so this level should never have "
                            "been created -- and nothing here maps it.");
                }
                if (cnDdt && std::getenv("BRAE_AMR_TRACE"))
                {
                    const auto amx = [](const std::vector<scalar>& v)
                    { scalar m = 0; for (scalar x : v) m = std::fmax(m, std::fabs(x)); return m; };
                    const auto amxv = [](const std::vector<vector>& v)
                    { scalar m = 0; for (const vector& x : v) m = std::fmax(m, mag(x)); return m; };
                    std::printf("  TRACE device step %ld: ddt0RhoU %.9g/%d/%ld ddtCorrU %.9g ddtCorrUf %.9g "
                                "alphaOO %.9g UOO %.9g phiOO %.9g UfOO %.9g aPhiEnd %.9g aPhiOld %.9g "
                                "aOld %.9g UOld %.9g phiOld %.9g UfOld %.9g\n",
                                (long)stepIndex, (double)amxv(hRhoU.internal), (int)hRhoU.exists,
                                (long)hRhoU.startTimeIndex, (double)amxv(hCorrU.internal),
                                (double)amxv(hCorrUf.internal), (double)amx(alphaOOH), (double)amxv(uOOH),
                                (double)amx(phiOOH.internal), (double)amxv(UfOO.internal),
                                (double)amx(alphaPhiEndH.internal), (double)amx(alphaPhiOldH.internal),
                                (double)amx(aOldH), (double)amxv(uOldH), (double)amx(phiOldH.internal),
                                (double)amxv(UfOld.internal));
                }
                if (refineAndMove)
                {
                    // hexRef8 lays its new points out on the LIVE mesh (hexRef8.C:3362, :3462, :3650); the
                    // adapter's copy was last written at the previous change, so it is given the points the
                    // motion has moved it to since -- and the motion's points0, which each change maps
                    // (the host loop's two lines, inter_driver_cpp.cu)
                    // BRAE_CONTROL_DEVICE_REFINE_STALE_POINTS=1 leaves the adapter on the points of its last
                    // change -- the write gate's control: the change then hands the mesh back where it was
                    if (std::getenv("BRAE_CONTROL_DEVICE_REFINE_STALE_POINTS") == nullptr)
                    {
                        f.amr->state.m.movePoints(mutableMesh->m->points());
                    }
                    else
                    {
                        std::printf("  *** CONTROL MODE: the refinement works on the points of its last change, not "
                                    "the moved ones. This run is deliberately wrong. ***\n");
                    }
                    f.amr->state.points0 = &dyn->points0Ref();
                }
                const bool changed =
                    interAmrUpdate(*f.amr, f, *mutableMesh, stepIndex, oldT, cnState);
                // fvMesh::updateMesh as the motion sees it: the change's V0, the mesh flux recreated on the
                // new faces (DynamicMotionSolverFvMesh::topoChanged)
                if (changed && refineAndMove)
                {
                    dyn->topoChanged(f.amr->state.V0, stepIndex);
                }
                // the sub-cycled old level on a refining mesh is alpha as this update left it (see the host
                // loop): the host copy the change mapped. Without a change the start-of-step copy already is.
                if (changed && writeNow && writer->writesAlphaOld())
                {
                    alphaOldWrite = f.alpha1.internal;
                    alphaOldBndWrite.assign(f.alpha1.boundary.size(), std::vector<scalar>());
                    for (std::size_t pi = 0; pi < f.alpha1.boundary.size(); ++pi)
                    {
                        alphaOldBndWrite[pi] = f.alpha1.boundary[pi]->value();
                    }
                }
                if (changed)
                {
                    // the solver's own rebuild, interFoam.C:118-142: gh and ghf, the flux from Sf & Uf
                    // and its pcorr solve, the mixture and the curvature. One copy, shared with the
                    // host loop, and the GAMG hierarchy un-built inside it.
                    // the GLOBAL time index, which the wall-distance schedule tests (wallDist.C:198)
                    interAfterMeshChange(f, *mutableMesh, meshAgglomeration, meshCpc, rep,
                                         f.amr->startTimeIndex + stepIndex, /*motionFollows=*/refineAndMove);

                    // ---- the counts every array below is sized by
                    nC = m.nCells();
                    nIf = m.nInternalFaces();
                    nFaces = static_cast<label>(g.magSf().size());
                    nBf = 0;
                    for (const FvPatch& q : fvp) nBf += q.size;

                    // ---- A FRESH DeviceMesh, not a geometry refresh. refreshDeviceMeshGeometry
                    // THROWS on a changed count on purpose (device_mesh.cuh:237), and it would be the
                    // wrong call anyway: buildDeviceMesh stamps a new addressingId, and every schedule
                    // cache in the tree -- the Gauss-Seidel colourings, the AMG hierarchies, the PCG
                    // and V-cycle workspaces, the coarse-level ids -- keys its validity on that id
                    // (tools/cache_key_audit.py). Rebuilding the mesh is what invalidates all of them.
                    dm = buildDeviceMesh(m, g, fvp);
                    ++meshGeometryEpoch;
                    dic = buildDeviceDilu(m.owner(), m.neighbour(), nC);
                    C.dic = &dic;
                    // ...and the GAMG upload, whose key is the HOST hierarchy's build count. That count
                    // is monotone across the change (inter_amr_cpp.cu un-builds the cache without
                    // resetting it), so this would be correct without the line; it is set because an
                    // upload from the old mesh must not survive on the strength of an argument.
                    gamgCache.uploaded = false;
                    gamgCache.uploadedBuild = -1;

                    // ---- the masks and the boundary geometry, per boundary FACE
                    // ...the porosity's cell list, which interAfterMeshChange has just RE-SELECTED
                    buildPorosity();
                    // ...and the MRF zones, whose host face lists it has just REBUILT
                    buildMrf();
                    // ...and the TURBULENCE closure, which is a FULL rebuild rather than the moving-mesh
                    // refresh: refreshDeviceInterTurbulenceGeometry refuses a changed boundary-face count
                    // by name ("A move keeps the topology fixed; this is a topology change"), and every
                    // buffer here -- the fields, the wall-function masks, the per-face coefficients, the
                    // two wall distances, the DILU level schedule -- is sized by the mesh. What a fresh
                    // build does NOT carry is the closure's own old-time state, so it is restored from the
                    // host arrays the mapper has just mapped, and oldStepTimeIndex with it: a fresh build
                    // leaves that at -1, which would make the next step's advance take the COLD-START
                    // branch and copy k into the old-old level mid-run.
                    if (f.turbulence.on)
                    {
                        dTurb = buildDeviceInterTurbulence(f.turbulence, f.U, m, g, fvp);
                        // THE OLD-TIME LEVELS ARE SET UNCONDITIONALLY, empty included, and that is not
                        // tidiness. buildDeviceInterTurbulence does ONCE-PER-RUN RESTART WORK -- it calls
                        // seedCnDdt0 twice and readClosureCnOldTime, which seed kOldStep and epsOldStep
                        // from the START directory's `<field>_0` files (device_inter_turbulence.cu:355-365)
                        // -- and it is being called MID-RUN here. Written as `if (!host.empty()) copyFrom`,
                        // the result would depend on the ORDER of those two writes: correct only because
                        // the restore happens to come second, and wrong at the FIRST change of a restarted
                        // case, where the host array is empty and the seed is not. A start directory
                        // holding `k_0` means a CrankNicolson restart, which is refused beside a change
                        // twice over -- so this is unreachable today, and an invariant that rests on the
                        // order of two lines is the kind this port keeps finding broken.
                        dTurb.kOldStep.copyFrom(f.turbulence.kOldStep);
                        dTurb.epsOldStep.copyFrom(f.turbulence.epsOldStep);
                        dTurb.cnKOO.copyFrom(f.turbulence.cn.kOO);
                        dTurb.cnEpsOO.copyFrom(f.turbulence.cn.epsOO);
                        dTurb.oldStepTimeIndex = f.turbulence.oldStepTimeIndex;
                        // ...and rho.oldTime(), which the mapper has just mapped through rhoOldH
                        if (!rhoOldH.empty()) dRhoOld.copyFrom(rhoOldH);
                        if (std::getenv("BRAE_AMR_TRACE"))
                        {
                            const auto mnmx = [](const std::vector<scalar>& v)
                            {
                                scalar lo = v.empty() ? scalar(0) : v[0], hi = lo;
                                for (const scalar x : v) { lo = std::fmin(lo, x); hi = std::fmax(hi, x); }
                                return std::pair<scalar, scalar>{lo, hi};
                            };
                            const auto kk = mnmx(f.turbulence.k.internal);
                            const auto nn = mnmx(f.turbulence.nut.internal);
                            std::vector<scalar> dk, dn;
                            dTurb.k.copyTo(dk);
                            dTurb.nut.copyTo(dn);
                            const auto dkk = mnmx(dk);
                            const auto dnn = mnmx(dn);
                            std::printf("  [trace] ...and the DEVICE buffers after the rebuild: k %zu "
                                        "[%.6g, %.6g]  nut %zu [%.6g, %.6g]\n",
                                        dk.size(), (double)dkk.first, (double)dkk.second,
                                        dn.size(), (double)dnn.first, (double)dnn.second);
                            std::printf("  [trace] REBUILT after the change: host k %zu [%.6g, %.6g]  nut "
                                        "%zu [%.6g, %.6g]  dTurb.k %zu nut %zu yCell %zu\n",
                                        f.turbulence.k.internal.size(), (double)kk.first, (double)kk.second,
                                        f.turbulence.nut.internal.size(), (double)nn.first, (double)nn.second,
                                        (std::size_t)dTurb.k.size(), (std::size_t)dTurb.nut.size(),
                                        f.turbulence.yCell.size());
                        }
                    }
                    buildBoundaryMasks();
                    dAFixes.copyFrom(aFixes);
                    dAFlag.copyFrom(aFlag);
                    dTakeU.copyFrom(takeU);
                    dUFixes.copyFrom(uFixes);
                    dUNamesRhoPhi.copyFrom(uNamesRhoPhi);
                    dbU = buildDeviceVectorBoundary(f.U, fvp, g);
                    if (uNamesRhoPhiAny) deviceUpdateInletOutlet(dbU, uSwitchFlux(dPhiB));
                    else                 deviceUpdateInletOutlet(dbU, dPhiB);

                    // ---- UP: every mesh-sized buffer the step READS, from the mapped host fields
                    dA.copyFrom(f.alpha1.internal);
                    dAOld.copyFrom(aOldH);
                    {
                        std::vector<scalar> cx(static_cast<std::size_t>(nC)),
                                           cy(static_cast<std::size_t>(nC)),
                                           cz(static_cast<std::size_t>(nC));
                        for (label c = 0; c < nC; ++c)
                        {
                            const vector& u = f.U.internal[static_cast<std::size_t>(c)];
                            cx[static_cast<std::size_t>(c)] = u.x;
                            cy[static_cast<std::size_t>(c)] = u.y;
                            cz[static_cast<std::size_t>(c)] = u.z;
                        }
                        dUx.copyFrom(cx); dUy.copyFrom(cy); dUz.copyFrom(cz);
                        for (label c = 0; c < nC; ++c)
                        {
                            const vector& u = uOldH[static_cast<std::size_t>(c)];
                            cx[static_cast<std::size_t>(c)] = u.x;
                            cy[static_cast<std::size_t>(c)] = u.y;
                            cz[static_cast<std::size_t>(c)] = u.z;
                        }
                        dUox.copyFrom(cx); dUoy.copyFrom(cy); dUoz.copyFrom(cz);
                    }
                    {
                        std::vector<scalar> bx, by, bz;
                        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                        {
                            if (isCoupledInterfaceType(fvp[pi].type)) continue;
                            for (const vector& v : uOldBndH[pi])
                            {
                                bx.push_back(v.x); by.push_back(v.y); bz.push_back(v.z);
                            }
                        }
                            if (dUobx.size()) { dUobx.copyFrom(bx); dUoby.copyFrom(by); dUobz.copyFrom(bz); }
                    }
                    dPhiI.copyFrom(f.phi.internal);
                    dPhiB.copyFrom(flattenPatches(f.phi.boundary, fvp));
                    if (dPhiOI.size()) dPhiOI.copyFrom(phiOldH.internal);
                    if (dPhiOB.size()) dPhiOB.copyFrom(flattenPatches(phiOldH.boundary, fvp));
                    dPrgh.copyFrom(f.p_rgh.internal);
                    dNH.copyFrom(f.nHatf.internal);
                    dNHB.copyFrom(flattenPatches(f.nHatf.boundary, fvp));
                    dABnd.copyFrom(patchValues(f.alpha1, fvp));
                    dK.copyFrom(f.K);
                    if (dRpI.size()) dRpI.copyFrom(f.rhoPhi.internal);
                    if (dRpB.size()) dRpB.copyFrom(flattenPatches(f.rhoPhi.boundary, fvp));
                    dGh.copyFrom(f.gh);
                    {
                        SurfaceScalarField gf;
                        gf.internal = f.ghfInternal;
                        gf.boundary = f.ghfBoundary;
                        dGhf.copyFrom(fullFace(gf, fvp));
                    }
                    dMagSf.copyFrom(magSfAll());
                    dRAU.copyFrom(f.rAU);
                    // rho, mu, nu and p are NOT uploaded: the step resizes and rewrites all four from
                    // the mapped alpha before anything reads them (device_inter_alpha_step.cu:112-117),
                    // and interAfterMeshChange has just recomputed the host's for its own pcorr solve.

                    // ---- and the MULES correction flux is DROPPED, interFoam.C:118-123: it is a flux
                    // on faces that no longer exist. Empty is how the alpha step is told there is none
                    // (device_inter_alpha_step.cu:224 tests its size against nIf).
                    dPrevCorrI.copyFrom(std::vector<scalar>());
                    dPrevCorrB.copyFrom(std::vector<scalar>());

                    // ---- AND THE CrankNicolson STATE BACK UP, each level into the buffers it came from,
                    // at the NEW counts. The scalars ride with it untouched.
                    if (cnDdt)
                    {
                        const auto upComponents = [&](const std::vector<vector>& src,
                                                      DeviceBuffer<scalar>* comp[3])
                        {
                            std::vector<scalar> c[3];
                            for (int k = 0; k < 3; ++k) c[k].resize(src.size());
                            for (std::size_t i = 0; i < src.size(); ++i)
                            {
                                c[0][i] = src[i].x; c[1][i] = src[i].y; c[2][i] = src[i].z;
                            }
                            for (int k = 0; k < 3; ++k) comp[k]->copyFrom(c[k]);
                        };
                        const auto flatPatches = [&](const std::vector<std::vector<vector>>& b)
                        {
                            std::vector<vector> flat;
                            for (std::size_t pi = 0; pi < fvp.size() && pi < b.size(); ++pi)
                            {
                                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                                flat.insert(flat.end(), b[pi].begin(), b[pi].end());
                            }
                            return flat;
                        };
                        const auto upDdt0 = [&](const fv::CrankNicolsonDdt0<vector>& h, DeviceCnDdt0& d)
                        {
                            if (!h.exists) return;
                            DeviceBuffer<scalar>* ci[3] = {&d.internal[0], &d.internal[1], &d.internal[2]};
                            upComponents(h.internal, ci);
                            if (h.boundary.empty()) return;
                            DeviceBuffer<scalar>* cb[3] = {&d.boundary[0], &d.boundary[1], &d.boundary[2]};
                            upComponents(flatPatches(h.boundary), cb);
                        };
                        upDdt0(hRhoU, dCn.ddt0RhoU);
                        upDdt0(hCorrU, dCn.ddtCorrU);
                        upDdt0(hCorrUf, dCn.ddtCorrUf);
                        // ...and the Uf pair, which is uploaded at the TOP of a step from the host copies
                        // the mapper has just rewritten: without this the rest of THIS step reads the old
                        // face count, exactly as phi's old-old level would.
                        uploadSurfaceVector(UfOld, fvp, dUfOld, dUfOldB);
                        uploadSurfaceVector(UfOO, fvp, dUfOO, dUfOOB);
                        if (!alphaOOH.empty()) dAOO.copyFrom(alphaOOH);
                        if (!uOOH.empty())
                        {
                            DeviceBuffer<scalar>* ci[3] = {&dUoox, &dUooy, &dUooz};
                            upComponents(uOOH, ci);
                        }
                        if (dUoobx.size())
                        {
                            DeviceBuffer<scalar>* cb[3] = {&dUoobx, &dUooby, &dUoobz};
                            upComponents(flatPatches(uOOBndH), cb);
                        }
                        if (!phiOOH.internal.empty())
                        {
                            // the LEVEL back into its own buffer, which the rest of THIS step reads: it is
                            // uploaded at the TOP of a step, so after a change in the middle of one it
                            // would still be the old face count -- what the ddtCorr size check caught.
                            if (dPhiOOI.size()) dPhiOOI.copyFrom(phiOOH.internal);
                            if (dPhiOOB.size()) dPhiOOB.copyFrom(flattenPatches(phiOOH.boundary, fvp));
                        }
                        // ...and the loop's bookkeeping for the NEXT step's level, which holds THIS step's
                        // phiOld and is mapped with it.
                        if (!phiOldPrevI.empty())
                        {
                            phiOldPrevI = phiOldH.internal;
                            phiOldPrevB = flattenPatches(phiOldH.boundary, fvp);
                        }
                        if (!alphaPhiEndH.internal.empty())
                        {
                            dAlphaPhiEndI.copyFrom(alphaPhiEndH.internal);
                            if (dAlphaPhiEndB.size())
                                dAlphaPhiEndB.copyFrom(flattenPatches(alphaPhiEndH.boundary, fvp));
                        }
                        if (!alphaPhiOldH.internal.empty())
                        {
                            dAlphaPhiOldI.copyFrom(alphaPhiOldH.internal);
                            if (dAlphaPhiOldB.size())
                                dAlphaPhiOldB.copyFrom(flattenPatches(alphaPhiOldH.boundary, fvp));
                        }
                    }
                }
                if (changed && cnDdt && std::getenv("BRAE_AMR_TRACE"))
                {
                    std::printf("  TRACE sizes after the change: nC %ld nIf %ld nBf %ld | ddt0RhoU %zu/%zu "
                                "ddtCorrU %zu/%zu ddtCorrUf %zu/%zu | dAOO %zu dUoox %zu dUoobx %zu "
                                "dPhiOO %zu/%zu aPhiEnd %zu/%zu aPhiOld %zu/%zu dUfOld %zu/%zu "
                                "dUfOO %zu/%zu\n",
                                (long)nC, (long)nIf, (long)nBf,
                                dCn.ddt0RhoU.internal[0].size(), dCn.ddt0RhoU.boundary[0].size(),
                                dCn.ddtCorrU.internal[0].size(), dCn.ddtCorrU.boundary[0].size(),
                                dCn.ddtCorrUf.internal[0].size(), dCn.ddtCorrUf.boundary[0].size(),
                                dAOO.size(), dUoox.size(), dUoobx.size(),
                                dPhiOOI.size(), dPhiOOB.size(),
                                dAlphaPhiEndI.size(), dAlphaPhiEndB.size(),
                                dAlphaPhiOldI.size(), dAlphaPhiOldB.size(),
                                dUfOld[0].size(), dUfOldB[0].size(),
                                dUfOO[0].size(), dUfOOB[0].size());
                }
                if (verbose)
                {
                    std::printf("    mesh: %ld cells refined, %ld split points unrefined, now %ld cells\n",
                                (long)f.amr->nRefined, (long)f.amr->nUnrefined, (long)m.nCells());
                }
            }
            // ...and THEN the move, with the block interFoam.C:112-148 runs after mesh.update(): the moving
            // walls, gh/ghf, `phi = Sf & Uf`, CorrectPhi with the mesh flux, makeRelative and the mixture --
            // once, on the moved mesh, with the geometry refreshed on the DeviceMesh the change just rebuilt
            if (moveThisCorrector && refineAndMove)
            {
                moveMeshNow();
            }

            // (Sf & Uf.oldTime()), the flux ddtCorr takes in phi.oldTime()'s place ON A DYNAMIC MESH --
            // fvcDdt.C:219 asks mesh.dynamic(), which is moving OR topo-changing, so a REFINING mesh
            // takes this branch as a moving one does. Built HERE, after either mesh update: fvcDdtUfCorr
            // dots the STORED old Uf with mesh().Sf() (EulerDdtScheme.C:527-531), which a move or a
            // change has just rewritten. Built before it, the error is INVISIBLE in step one -- Uf starts
            // at zero and so does the flux, whatever the geometry -- and by step two it is 13% of |U| on
            // testTubeMixer (measured: |U| 4.8e-01 of 3.6e+00 against the host arm, alpha still 6e-13;
            // step one 1.6e-11).
            // correctPhi on a mesh that does NOT move: rAU is still AUTO_WRITE and 1/UEqn.A()
            // (initCorrectPhi.H:3-17, pEqn.H:4), and nothing in this loop reads it, so it is taken back
            // on a write step only. Without it the device wrote the start value (1 on a cold start).
            if (!f.meshIsDynamic)
            {
                C.rAUOut = (writer && writeNow && writer->writesRAU()) ? &dRAU : nullptr;
            }
            if (f.meshIsDynamic)
            {
                // the ABSOLUTE flux fvc::correctUf reads at the end of the corrector, and rAU for the
                // NEXT mesh update's CorrectPhi. Both are wanted on a refining mesh exactly as on a
                // moving one, and both were set inside the moving branch until this block existed.
                C.phiAbsIntOut = &dPhiAbsI;
                C.phiAbsBndOut = &dPhiAbsB;
                // ...on a mesh that MOVES only: a refining mesh's flux is absolute throughout
                phiAbsBndForU = dyn ? &dPhiAbsB : nullptr;
                C.phiAbsIfOut  = (dCyc.n > 0) ? &dPhiAbsIf : nullptr;
                C.rAUOut       = &dRAU;
                std::vector<scalar> pu(static_cast<std::size_t>(nIf));
                for (label fc = 0; fc < nIf; ++fc)
                {
                    pu[static_cast<std::size_t>(fc)] = dot(g.Sf()[fc], UfOld.internal[fc]);
                }
                dPhiUfOld.copyFrom(pu);
                C.phiUfOldInt = &dPhiUfOld;
            }

            // mesh().changing() this step: the registry is bypassed and deleted (gradScheme.C:132-142)
            C.gradUMeshChanging = dyn && dyn->moving();
            // subCycle.H:78: the run's first alpha1.oldTime() creates the old level ahead of this step's
            // alpha solve, keeping the valueFractions and contact-angle gradients alpha has now (idempotent)
            if (writer && writer->writesAlphaOld())
            {
                writer->noteAlphaOldCreation(f.alpha1);
            }
            // phi.oldTime() IS CREATED BY THE ALPHA BLEND of the first corrector that off-centres on a
            // dynamic mesh (alphaEqn.H:91-97; GeometricField::oldTime() copies the field as it stands), and
            // THE COPY STAYS for the rest of the time index: the next outer corrector blends with it, not
            // with the flux beside it and not with the previous step's. The level is made here, from the
            // flux this corrector's alpha step is about to read, so every corrector takes one path. The
            // host driver's twin has the measurement (RAS/floatingObject, U 3.1e-08 at step two).
            // BRAE_CONTROL_CN_PHIOLD_PREV=1 leaves the previous step's flux in the level -- the control.
            if (cnDdt && dCn.ocAlpha > scalar(0) && !dCn.phiOldExists)
            {
                if (std::getenv("BRAE_CONTROL_CN_PHIOLD_PREV") != nullptr)
                {
                    std::printf("  *** CONTROL MODE: phi.oldTime() keeps the previous step's flux where "
                                "OpenFOAM creates it from this corrector's. This run is deliberately wrong. ***\n");
                }
                else
                {
                    deviceCopy(dPhiOI, dPhiI);
                    deviceCopy(dPhiOB, dPhiB);
                    if (dCyc.n > 0)
                    {
                        deviceCopy(dPhiOIf, dCyc.phi);
                    }
                }
                dCn.phiOldExists = true;
            }
            // a cold start's U.oldTime() is CREATED inside the first assembly's fvm::ddt, after
            // updateCoeffs (GeometricField::oldTime() copies the field as it stands): on every patch
            // where U fixes its value the old level -- and the old-old one, a copy of it -- holds what
            // the patch holds then, not what the start file wrote. The solve never reads it; U_0 and
            // ddt0's patches are written with it. BRAE_CONTROL_CN_OLD_AT_ENTRY=1 leaves it as it began.
            // TAKEN HERE, ahead of the step: the mesh update above has run U.correctBoundaryConditions()
            // (a moving wall holds this step's velocity), and the patches whose updateCoeffs the assembly
            // itself would move -- a time-varying inlet -- are not on any case that reaches this.
            if (uOldCreationPending)
            {
                if (cnDdt && std::getenv("BRAE_CONTROL_CN_OLD_AT_ENTRY") == nullptr)
                {
                    std::vector<scalar> bx, by, bz;
                    dUobx.copyTo(bx); dUoby.copyTo(by); dUobz.copyTo(bz);
                    std::size_t off = 0;
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        if (isCoupledInterfaceType(fvp[pi].type)) continue;
                        const std::vector<vector>& v = f.U.boundary[pi]->value();
                        if (f.U.boundary[pi]->fixesValue())
                        {
                            for (std::size_t i = 0; i < v.size() && off + i < bx.size(); ++i)
                            {
                                bx[off + i] = v[i].x; by[off + i] = v[i].y; bz[off + i] = v[i].z;
                            }
                        }
                        off += v.size();
                    }
                    dUobx.copyFrom(bx); dUoby.copyFrom(by); dUobz.copyFrom(bz);
                    dUoobx.copyFrom(bx); dUooby.copyFrom(by); dUoobz.copyFrom(bz);
                }
                uOldCreationPending = false;
            }
            interPhase::mark("0 step preparation");
            deviceInterStep(dm, rep.deltaT, C, props, H, dGh, dGhf, dMagSf,
                            dA, dAOld, dUx, dUy, dUz, dUox, dUoy, dUoz,
                            dUobx, dUoby, dUobz,
                            dPhiI, dPhiB, dPhiOI, dPhiOB, dUFixes, dPrgh, dP,
                            dNH, dNHB, dABnd, dK, dAFixes, dAFlag, dbU,
                            dRho, dMu, dNu, dRpI, dRpB, tapsOut);
            interPhase::mark("5 pressure correctors");
            if (cnDdt && dCn.ocAlpha > scalar(0))
            {
                // the flux this pass ended on is alphaPhi10 as it stands: the next step's oldTime()
                // stores it
                deviceCopy(dAlphaPhiEndI, dAlphaPhiOutI);
                deviceCopy(dAlphaPhiEndB, dAlphaPhiOutB);
                if (dCyc.n > 0) deviceCopy(dAlphaPhiEndIf, dAlphaPhiOutIf);
                if (!alphaPhiOldExists)
                {
                    // GeometricField::oldTime() on the first blended step: the level was created by this
                    // pass as a copy of the flux BEFORE the un-blend (alphaPhiCreated*, written into the
                    // old level's buffer), and the step's other passes read that same copy --
                    // storeOldTimes stores nothing within a time index
                    alphaPhiOldExists = true;
                    alphaPhiOldIndex = s + 1;
                    dCn.alphaPhiOldInt = &dAlphaPhiOldI;
                    dCn.alphaPhiOldBnd = &dAlphaPhiOldB;
                    if (dCyc.n > 0)
                    {
                        alphaPhiOldIfIndex = s + 1;
                        dCn.alphaPhiOldIf = &dAlphaPhiOldIf;
                    }
                }
            }

            // turbulence->correct(), interFoam.C:169-172 -- after this outer corrector's last
            // pressure corrector. dbU is current: the step's last updateUBoundary rebuilt it and the
            // pressureInletOutletVelocity switch ran on it after that.
            //
            // pimple.turbCorr(): with turbOnFinalIterOnly -- OpenFOAM's default -- the closure
            // advances ONCE per time step, on the final outer corrector. Running it on every one
            // would advance k and epsilon nOuterCorrectors times per physical step.
            // ...and frozenFlow skips the closure too: the `continue` at interFoam.C:158 jumps past
            // turbulence->correct() at :171, not only past the momentum and the pressure. A port that
            // guarded UEqn and pEqn alone would keep advancing k and epsilon on a frozen velocity.
            if (!f.pimple.frozenFlow && (!f.pimple.turbOnFinalIterOnly || finalOuter))
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
                // localEuler: the closure's two fvm::ddts take the local step (unless the gate's control
                // switches the `turbulence` consumer off)
                ti.rDeltaT = (f.lts && !ltsScalar.count("turbulence")) ? &dRDeltaT : nullptr;
                ti.deltaT = rep.deltaT;
                // the step's index and the corrector, as the host loop supplies them -- `s + 1` is the
                // expression the CrankNicolson block already uses for this step's index
                ti.timeIndex = s + 1;
                ti.finalIter = finalOuter ? 1 : 0;
                // A MOVING MESH's two terms, the pair the HOST closure has always taken
                // (InterTurbulenceStepInput::V0/meshPhi at inter_driver_cpp.cu): the volumes the cells
                // had BEFORE this step's move -- dV0 above, the mesh's own once-per-step store, which is OF
                // fvMesh::movePoints:944's storeOldVol -- and the
                // mesh flux, split back into the internal and boundary arrays the closure's deviceDiv
                // takes (dMeshPhi is the FULL face array, as phi is).
                DeviceBuffer<scalar> dMeshPhiI, dMeshPhiB;
                if (dyn && dMeshPhi.size() == static_cast<std::size_t>(nFaces))
                {
                    const std::vector<scalar> mp = dMeshPhi.host();
                    dMeshPhiI.copyFrom(std::vector<scalar>(mp.begin(), mp.begin() + nIf));
                    dMeshPhiB.copyFrom(std::vector<scalar>(mp.begin() + nIf, mp.end()));
                    ti.V0         = &dV0;
                    ti.V00        = (cnDdt && dV00.size() == static_cast<std::size_t>(nC)) ? &dV00 : nullptr;
                    ti.meshPhiInt = &dMeshPhiI;
                    ti.meshPhiBnd = &dMeshPhiB;
                    ti.meshPhiIf  = (dCyc.n > 0) ? &dMeshPhiIf : nullptr;
                }
                // the second field's log is omega's under kOmegaSST, as the host driver keeps it
                ti.epsilonLog = (f.turbulence.model == cpu::interFoam::InterRasModel::KOmegaSST)
                              ? &rep.omegaSolves : &rep.epsilonSolves;
                ti.kLog = &rep.kSolves;
                ti.mangroves = dMangroves.turbulence() ? &dMangroves : nullptr;
                if (cnDdt)
                {
                    ti.cn = &cnClock;
                    ti.rhoOO = &dRhoOO;
                }
                if (std::getenv("BRAE_AMR_TRACE"))
                {
                    // EVERY MESH-SIZED INPUT THE CLOSURE IS HANDED, BY SIZE, and this line is what named
                    // the rho.oldTime() defect above: one entry read 2268 where every other read 2814, and
                    // 546 of 2814 cells -- every cell the change had added -- came out of the solve with k
                    // collapsed. A buffer left at the old count looks like nothing else from inside a
                    // kernel, and it is cheaper to print the sizes than to bisect the arithmetic.
                    const auto sz = [](const DeviceBuffer<scalar>* b) { return b ? (long)b->size() : -1L; };
                    std::printf("  [trace] closure inputs at step %ld (nC %ld nIf %ld nBf %ld): Ux %ld "
                                "phiInt %ld phiBnd %ld rhoPhiInt %ld rho %ld rhoBnd %ld rhoOld %ld nu %ld "
                                "nuBnd %ld k %ld eps %ld nut %ld nutBnd %ld\n",
                                (long)stepIndex, (long)nC, (long)nIf, (long)nBf,
                                sz(ti.Ux), sz(ti.phiInt), sz(ti.phiBnd), sz(ti.rhoPhiInt), sz(ti.rho),
                                sz(ti.rhoBnd), sz(ti.rhoOld), sz(ti.nu), sz(ti.nuBnd),
                                (long)dTurb.k.size(), (long)dTurb.epsilon.size(),
                                (long)dTurb.nut.size(), (long)dTurb.nutBnd.size());
                }
                interPhase::mark("6 after the step, before the closure");
                deviceCorrectInterTurbulence(dTurb, f.turbulence, ti, dm, dbU);
                interPhase::mark("7 turbulence closure");
                // ...and "Updating grad(U)": kOmegaSST's correct forms it, and a caching case keeps it -- the host
                // loop's re-formation after the closure (inter_turbulence_cpp.cu), from U as it stands and its
                // STORED patch values (f.U's, which the step's last U hook wrote). Not on a changing mesh, where
                // OpenFOAM forms and does not keep it.
                if (dGradUCache.on && f.turbulence.model == cpu::interFoam::InterRasModel::KOmegaSST
                 && !f.turbulence.frozen && !(dyn && dyn->moving()) && gradUStale)
                {
                    dGradUCache.valid = true;   // the control: validate()'s field, kept
                }
                else if (dGradUCache.on && f.turbulence.model == cpu::interFoam::InterRasModel::KOmegaSST
                 && !f.turbulence.frozen && !(dyn && dyn->moving()))
                {
                    std::vector<scalar> bx, by, bz;
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        if (isCoupledInterfaceType(fvp[pi].type)) continue;
                        for (const vector& u : f.U.boundary[pi]->value())
                        {
                            bx.push_back(u.x);
                            by.push_back(u.y);
                            bz.push_back(u.z);
                        }
                    }
                    DeviceBuffer<scalar> sbx(bx), sby(by), sbz(bz);
                    const DeviceBuffer<scalar>* sb[3] = {&sbx, &sby, &sbz};
                    deviceStoreGradU(dGradUCache, dm, dbU, dUx, dUy, dUz, sb);
                }
                interPhase::mark("8 grad(U) kept after the closure");
                if (std::getenv("BRAE_AMR_TRACE"))
                {
                    std::vector<scalar> tk, tn;
                    dTurb.k.copyTo(tk);
                    dTurb.nut.copyTo(tn);
                    scalar klo = tk.empty() ? 0 : tk[0], khi = klo, nlo = tn.empty() ? 0 : tn[0], nhi = nlo;
                    for (const scalar x : tk) { klo = std::fmin(klo, x); khi = std::fmax(khi, x); }
                    for (const scalar x : tn) { nlo = std::fmin(nlo, x); nhi = std::fmax(nhi, x); }
                    std::printf("  [trace] after correct() at step %ld: k [%.6g, %.6g]  nut [%.6g, %.6g]\n",
                                (long)stepIndex, (double)klo, (double)khi, (double)nlo, (double)nhi);
                }
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
                // localEuler: the host closure takes the host's field, as the host loop hands it
                ti.rDeltaT = (f.lts && !ltsScalar.count("turbulence")) ? &f.rDeltaT : nullptr;
                // the step's index and the corrector, as the host loop supplies them -- `s + 1` is the
                // expression the CrankNicolson block already uses for this step's index
                ti.timeIndex = s + 1;
                ti.finalIter = finalOuter ? 1 : 0;
                // fvOptions(k) and fvOptions(epsilon), as the host loop hands them (inter_driver_cpp.cu):
                // without this the instrument ran the mangroves' case with no turbulence source at all
                ti.fvOptions = f.fvOptions.empty() ? nullptr : &f.fvOptions;
                // the host closure keeps the two second fields in separate logs and fills the model's own
                ti.omegaLog = &rep.omegaSolves;
                ti.epsilonLog = &rep.epsilonSolves;
                ti.kLog = &rep.kSolves;
                // the host closure fills the host registry; the device assembly reads its own copy of it
                ti.gradUCache = &f.gradUCache;
                correctInterTurbulence(f.turbulence, ti, m, g, fvp);
                // only when THIS call formed it: a frozen closure returns before its store, and the host flag
                // still says valid from validate() -- this loop never clears it (review finding)
                if (dGradUCache.on && f.turbulence.model == cpu::interFoam::InterRasModel::KOmegaSST
                 && !f.turbulence.frozen && !(dyn && dyn->moving()))
                {
                    uploadHostGradU();
                }
            }
            }   // pimple.turbCorr()

            // fvc::correctUf(Uf, U, phi), pEqn.H:70-72 -- the END of the pressure corrector on a
            // DYNAMIC mesh, which fvcMeshPhi.C:224 gates on mesh.dynamic() and a refining mesh is. Uf is
            // a HOST field (next step's ddtCorr reads (Sf & Uf.oldTime()) off it), so the device's U and
            // phi come back for it. One copy of the arithmetic, shared with the host loop (correctUf,
            // inter_peqn_cpp.cu). rAU comes back with it, for the next change's CorrectPhi.
            if (f.meshIsDynamic)
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
                // ...and the pair's, which the unflatten does not touch (it has no coupled patch in it):
                // the copy above left the RELATIVE flux there, and the next move's `phi = Sf & Uf` reads
                // the pair's Uf like any other face's
                if (dCyc.n > 0 && dPhiAbsIf.size() == static_cast<std::size_t>(dCyc.n))
                {
                    const std::vector<scalar> pa = dPhiAbsIf.host();
                    std::size_t off = 0;
                    for (const CyclicInterface& c : cyclics)
                    {
                        const std::size_t pi = static_cast<std::size_t>(c.patch);
                        const std::size_t n = c.faceCells.size();
                        phiAbs.boundary[pi].assign(pa.begin() + off, pa.begin() + off + n);
                        off += n;
                    }
                }
                correctUf(f.Uf, f.U, phiAbs, m, g, fvp);
                // ...and rAU, for the same reason and with the same blindness: the NEXT mesh update
                // interpolates it for CorrectPhi's laplacian, and nothing THIS step does reads it.
                // Left at the value buildInterFields wrote, step one is exact (both arms start there)
                // and step two is 2.3e-02 of |U| on waveMakerSolitary.
                if (dRAU.size()) dRAU.copyTo(f.rAU);
            }
            if (!f.meshIsDynamic && C.rAUOut && dRAU.size())
            {
                dRAU.copyTo(f.rAU);
            }
        }   // the outer corrector loop
        interPhase::mark("9 end of the outer corrector (Uf, rAU)");

        rep.steps = s + 1;
        rep.time += rep.deltaT;

        // runTime.write() (interFoam.C:175): the cells from the device into writer-owned copies, the patch
        // values as the hooks left them on the host. The closure is the exception: its download writes
        // f.turbulence, the host copy this loop never reads -- arm F of tests/interfoam_write_vs_openfoam.sh
        // holds the run byte-identical whether it writes every step or once.
        if (writer && writeNow)
        {
            std::vector<scalar> aW, prghW, pW, ux, uy, uz;
            dA.copyTo(aW);
            dPrgh.copyTo(prghW);
            dP.copyTo(pW);
            dUx.copyTo(ux);
            dUy.copyTo(uy);
            dUz.copyTo(uz);
            std::vector<vector> uW(static_cast<std::size_t>(nC));
            for (label c = 0; c < nC; ++c)
            {
                uW[c] = vector{ux[c], uy[c], uz[c]};
            }
            SurfaceScalarField phiW;
            dPhiI.copyTo(phiW.internal);
            phiW.boundary = f.phi.boundary;
            SurfaceScalarField aPhiW;
            dAlphaPhiWriteI.copyTo(aPhiW.internal);
            aPhiW.boundary.assign(fvp.size(), std::vector<scalar>());
            unflatten(dAlphaPhiWriteB, aPhiW.boundary);
            if (!cyclics.empty() && dAlphaPhiWriteIf.size())
            {
                std::vector<scalar> pif;
                dAlphaPhiWriteIf.copyTo(pif);
                std::size_t off = 0;
                for (const CyclicInterface& c : cyclics)
                {
                    const std::size_t pi = static_cast<std::size_t>(c.patch);
                    const std::size_t n = c.faceCells.size();
                    if (pi < aPhiW.boundary.size() && off + n <= pif.size())
                    {
                        aPhiW.boundary[pi].assign(pif.begin() + off, pif.begin() + off + n);
                    }
                    off += n;
                }
            }
            if (deviceClosure)
            {
                downloadDeviceInterTurbulence(dTurb, f.turbulence, fvp);
                // ...whose evaluateBoundary blends a flux-conditional patch with the HOST object's
                // valueFraction, which this loop never switches: the device keeps the switch in
                // dbK/dbEps (deviceUpdateInletOutlet). RAS/damBreak's atmosphere wrote the cells, epsilon
                // 1.7 off OpenFOAM's `uniform` inletValue at 0.0012, and RAS/waterChannel's outlet omega
                // 3.0e-01 off, where the host arm wrote OpenFOAM's values. Those faces take the device's
                // own values -- what its last correctBoundaryConditions left, as OpenFOAM's are.
                auto deviceFluxConditional = [&](
                    const DeviceBoundary& db,
                    const DeviceBuffer<scalar>& cells,
                    GeometricField<scalar>& fld)
                {
                    DeviceBuffer<scalar> dv;
                    deviceBCValue(db, cells, dv);
                    std::vector<std::vector<scalar>> bv;
                    unflatten(dv, bv);
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        const int cat = fld.boundary[pi]->bcCategory();
                        if (cat == 3 || cat == 4)
                        {
                            fld.boundary[pi]->setValue(bv[pi]);
                        }
                    }
                };
                deviceFluxConditional(dTurb.dbK, dTurb.k, f.turbulence.k);
                if (f.turbulence.model != cpu::interFoam::InterRasModel::KEqnLES)
                {
                    const bool sst = f.turbulence.model == cpu::interFoam::InterRasModel::KOmegaSST;
                    GeometricField<scalar>& second = sst ? f.turbulence.omega : f.turbulence.epsilon;
                    deviceFluxConditional(dTurb.dbEps, dTurb.epsilon, second);
                }
            }
            const std::vector<std::vector<scalar>> pB =
                staticPressureBoundary(f.p_rgh, stepRhoBnd, f.ghfBoundary);
            InterWriteState ws;
            ws.time = rep.time;
            ws.timeIndex = writer->startTimeIndex() + rep.steps;
            ws.deltaT = rep.deltaT;
            ws.alpha1 = &f.alpha1;
            ws.U = &f.U;
            ws.p_rgh = &f.p_rgh;
            ws.p = &pW;
            ws.pBoundary = &pB;
            ws.phi = &phiW;
            ws.alphaPhi0 = &aPhiW;
            ws.turbulence = &f.turbulence;
            ws.alpha1Cells = &aW;
            ws.UCells = &uW;
            ws.p_rghCells = &prghW;
            ws.alpha1OldCells = &alphaOldWrite;
            ws.alpha1OldBoundary = &alphaOldBndWrite;
            // coupled patches keep the host's values; the writer takes the step's start there anyway
            alphaSubBndWrite.assign(f.alpha1.boundary.size(), std::vector<scalar>());
            for (std::size_t pi = 0; pi < f.alpha1.boundary.size(); ++pi)
            {
                alphaSubBndWrite[pi] = f.alpha1.boundary[pi]->value();
            }
            if (writer->writesAlphaOld() && dAlphaSubBndWrite.size())
            {
                unflatten(dAlphaSubBndWrite, alphaSubBndWrite);
            }
            ws.alpha1SubCycleBoundary = &alphaSubBndWrite;
            ws.rDeltaT = &f.rDeltaT;
            ws.Uf = &f.Uf;
            ws.meshPhi = f.dynamicMesh ? &f.dynamicMesh->meshPhi() : nullptr;
            ws.points = &m.points();
            ws.displacement = f.dynamicMesh ? f.dynamicMesh->displacement() : nullptr;
            ws.rigidBody = f.dynamicMesh ? f.dynamicMesh->rigidBody() : nullptr;
            ws.rAU = &f.rAU;
            // CrankNicolson on a moving mesh: the scheme's registry state as the step left it, brought
            // down in the host loop's own form (InterWriteCrankNicolson)
            InterWriteCrankNicolson wcn;
            fv::CrankNicolsonDdt0<vector> wDdt0RhoU, wDdtCorrU, wDdtCorrUf;
            std::vector<vector> wUOld;
            std::vector<std::vector<vector>> wUOldBnd;
            if (writer->writesCrankNicolson() && cnDdt && dyn)
            {
                auto splitVec = [&](const DeviceBuffer<scalar>* b[3], std::vector<std::vector<vector>>& out)
                {
                    std::vector<scalar> x, y, z;
                    b[0]->copyTo(x); b[1]->copyTo(y); b[2]->copyTo(z);
                    out.assign(fvp.size(), {});
                    std::size_t off = 0;
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        if (isCoupledInterfaceType(fvp[pi].type)) continue;
                        const std::size_t n = static_cast<std::size_t>(fvp[pi].size);
                        if (off + n > x.size()) break;
                        out[pi].resize(n);
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            out[pi][i] = vector{x[off + i], y[off + i], z[off + i]};
                        }
                        off += n;
                    }
                };
                auto down = [&](const DeviceCnDdt0& src, const char* name, fv::CrankNicolsonDdt0<vector>& dst)
                {
                    std::vector<scalar> x, y, z;
                    src.internal[0].copyTo(x); src.internal[1].copyTo(y); src.internal[2].copyTo(z);
                    dst.name = name;
                    dst.internal.resize(x.size());
                    for (std::size_t i = 0; i < x.size(); ++i)
                    {
                        dst.internal[i] = vector{x[i], y[i], z[i]};
                    }
                    const DeviceBuffer<scalar>* bb[3] = {&src.boundary[0], &src.boundary[1], &src.boundary[2]};
                    splitVec(bb, dst.boundary);
                    dst.startTimeIndex = src.startTimeIndex;
                    dst.timeIndex = src.timeIndex;
                    dst.exists = src.exists;
                };
                down(dCn.ddt0RhoU, "ddt0(rho,U)", wDdt0RhoU);
                down(dCn.ddtCorrU, "ddtCorrDdt0(U)", wDdtCorrU);
                down(dCn.ddtCorrUf, "ddtCorrDdt0(Uf)", wDdtCorrUf);
                {
                    std::vector<scalar> x, y, z;
                    dUox.copyTo(x); dUoy.copyTo(y); dUoz.copyTo(z);
                    wUOld.resize(x.size());
                    for (std::size_t i = 0; i < x.size(); ++i)
                    {
                        wUOld[i] = vector{x[i], y[i], z[i]};
                    }
                    const DeviceBuffer<scalar>* bb[3] = {&dUobx, &dUoby, &dUobz};
                    splitVec(bb, wUOldBnd);
                }
                wcn.ddt0RhoU = &wDdt0RhoU;
                wcn.ddtCorrU = &wDdtCorrU;
                wcn.ddtCorrUf = &wDdtCorrUf;
                wcn.meshPhi0 = &f.cnMeshPhi0;
                wcn.UOld = &wUOld;
                wcn.UOldBnd = &wUOldBnd;
                wcn.V0 = &dyn->V0();
                wcn.UfOld = ufOldStoredWithOO ? &UfOld : nullptr;
                ws.cn = &wcn;
            }
            if (f.amr && f.amr->active)
            {
                // the LIVE history compacted, as refinementHistory's operator<< does (see the host loop)
                cpu::hexRef8::compactHistory(f.amr->state.history);
                ws.mesh = &m;
                ws.amr = &*f.amr;
                ws.points0 = f.dynamicMesh ? &f.dynamicMesh->points0() : nullptr;
            }
            writer->write(ws);
        }

        interPhase::mark("9 write");
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

    // hand the device's answer back through the host fields, so a caller compares the same objects.
    // alpha's patch values are the ones the last alpha step's hooks left, not re-evaluated: see the
    // interfaceForces hook.
    dA.copyTo(f.alpha1.internal);
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
    rep.gradUCacheConsumed = dGradUCache.consumed;
    interPhase::mark("0 between steps (clock, write, downloads)");
    interPhase::report(static_cast<long>(rep.steps));
    return rep;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
