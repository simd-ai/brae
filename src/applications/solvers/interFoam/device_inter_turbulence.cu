// interFoam's turbulence on the device. See device_inter_turbulence.cuh for what each input is.
#include "device_inter_turbulence.cuh"
#include "device_les_keqn.cuh"
#include "device_blas.cuh"
#include "near_wall_dist.cuh"
#include "nut_wall_function.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

namespace brae {

namespace {

std::vector<scalar> patchValuesOf(
    const GeometricField<scalar>& f,
    const std::vector<FvPatch>& patches)
{
    std::vector<scalar> v;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // THE DEVICE'S BOUNDARY LAYOUT holds the NON-COUPLED patches only (device_mesh.cuh:41-44),
        // and so does buildDeviceBoundary. A flatten that walks every patch is longer than the arrays
        // it is indexed with and shifts every patch after the first coupled one.
        if (isCoupledInterfaceType(patches[pi].type)) continue;
        const std::vector<scalar>& b = f.boundary[pi]->value();
        v.insert(v.end(), b.begin(), b.end());
    }
    return v;
}

// GeometricField::storeOldTimes for the closure's fields under CrankNicolson: the old-old level is
// rotated ONCE per time index, and at the first step oldTime().oldTime() is a copy of oldTime() --
// OpenFOAM creates the missing level lazily. `second` is false under LES kEqn, which solves k alone.
// One function for all three closures so they cannot drift on when the rotation happens.
// storeOldTimes for the device closure's transported scalars: ONCE per TIME INDEX, whatever the scheme and
// however many outer correctors run. The host twin is advanceTurbulenceOldTime.
void advanceDeviceTurbulenceOldTime(
    DeviceInterTurbulence& d,
    label timeIndex,
    bool second)
{
    if (d.oldStepTimeIndex == timeIndex) return;
    if (d.kOldStep.size() == 0)
    {
        deviceCopy(d.cnKOO, d.k);
        if (second) deviceCopy(d.cnEpsOO, d.epsilon);
    }
    else
    {
        deviceCopy(d.cnKOO, d.kOldStep);
        if (second) deviceCopy(d.cnEpsOO, d.epsOldStep);
    }
    deviceCopy(d.kOldStep, d.k);
    if (second) deviceCopy(d.epsOldStep, d.epsilon);
    d.oldStepTimeIndex = timeIndex;
}

} // namespace


DeviceInterTurbulence buildDeviceInterTurbulence(
    const cpu::interFoam::InterTurbulence& t,
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    DeviceInterTurbulence d;
    if (!t.on) return d;
    // PBiCG's preconditioner needs the mesh's level schedule, once (see DeviceInterTurbulence::dilu)
    // ...for EITHER closure. Keyed on kEpsilon alone, the SST branch below would have found no
    // schedule and refused a case it can run.
    // ANY of the four entries may name PBiCG now that the two equations take their own. Built from
    // kFinal alone, a case where only the SECOND equation names `solver PBiCG; preconditioner DILU;`
    // would reach the solve with no schedule and throw -- a hole the per-equation lift would have opened.
    if (t.kSolveFinal.pbicgDILU() || t.epsSolveFinal.pbicgDILU() || t.omegaSolveFinal.pbicgDILU()
     || t.kSolve.pbicgDILU() || t.epsSolve.pbicgDILU() || t.omegaSolve.pbicgDILU())
    {
        d.dilu = buildDeviceDilu(m.owner(), m.neighbour(), m.nCells());
    }

    // The device closure builds its wall set from patches that are BOTH a `wall` and carry the
    // wall function (isTurbWallPatch); the host reference asks the boundary condition alone. They
    // agree on every shipped tutorial, and where they would not the device must say so.
    // kOmegaSST keeps its second field in ::omega; every slot below is that field's under SST
    const bool sst = (t.model == cpu::interFoam::InterRasModel::KOmegaSST);
    // LES kEqn HAS NO SECOND FIELD: the host closure reads k and nut and nothing else, so t.epsilon
    // here is a default-constructed field with no patches at all. Walking it built a wall set from
    // null boundary pointers.
    const bool les = (t.model == cpu::interFoam::InterRasModel::KEqnLES);
    const GeometricField<scalar>& second = sst ? t.omega : t.epsilon;
    std::vector<char> wfPatch(patches.size(), 0);
    for (std::size_t pi = 0; pi < patches.size() && !les; ++pi)
    {
        const bool wf = second.boundary[pi]->isTurbulenceWallFunction();
        wfPatch[pi] = wf ? 1 : 0;
        if (wf && patches[pi].type != "wall")
            throw std::runtime_error(
                "brae interFoam (device): patch `" + patches[pi].name + "` carries an "
                "epsilon or omega wall function and is a `" + patches[pi].type + "`, not a `wall`. The device "
                "closure constrains wall patches only. The host path (no -device) runs it.");
    }

    d.k.copyFrom(t.k.internal);
    if (!les) d.epsilon.copyFrom(second.internal);
    d.nut.copyFrom(t.nut.internal);
    d.nutBnd.copyFrom(patchValuesOf(t.nut, patches));
    // storedIoSeed: the closure reconstructs an inletOutlet patch's STORED value at its first step
    d.dbK = buildDeviceBoundary(t.k, patches, g, /*storedIoSeed=*/true);
    if (!les) d.dbEps = buildDeviceBoundary(second, patches, g, /*storedIoSeed=*/true);
    // nut's own: the closure writes every other kind of face, and this decides the inletOutlet ones
    d.dbNut = buildDeviceBoundary(t.nut, patches, g, /*storedIoSeed=*/false);
    // correctBoundaryConditions on nut, the host closure's own loop (kOmegaSST_cpp.cu:946-984): every
    // patch that is not a `wall` (the wall functions write those), not `empty` (OpenFOAM's
    // emptyFvPatchField has size 0, so there is nothing there to evaluate) and not `calculated` (nut's
    // field assignment fills those) is EVALUATED against the cell nut just written -- a zeroGradient or
    // symmetry face takes the cell, a fixedValue face keeps what the case pinned, an inletOutlet face
    // takes the flux switch first. Only the inletOutlet class was evaluated here, which left a
    // zeroGradient nut patch at the value it was built with for the whole run: MEASURED on
    // RAS/waterChannel with the inlet's nut zeroGradient, nut 5.5e-06 and U 1.9e-05 from OpenFOAM after
    // ten steps, where the HOST closure in this same device loop is 2.3e-12.
    std::vector<label> nutEval;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // A COUPLED PATCH IS NOT IN THE DEVICE'S BOUNDARY ARRAYS, so it is not in this mask either.
        // It needs no evaluate: OpenFOAM ends every field assignment with
        // correctLocalBoundaryConditions(), which gives a constraint patch the RESULT's own two cells
        // interpolated (`localConsistency`, GeometricFieldFunctionsM.C) -- a value nothing on the
        // device reads, because the device's nut boundary does not hold that face.
        if (isCoupledInterfaceType(patches[pi].type)) continue;
        // "not a `wall`" stands for "not written by a wall function", which is true of every RAS
        // case this loop runs and FALSE under LES kEqn: there nothing is a wall function (the host
        // reader refuses a `nut*` patch type under that model), and a wall's nut is an ordinary
        // condition that correctBoundaryConditions evaluates like any other. LES/nozzleFlow2D's
        // `walls` is zeroGradient. Skipped, it kept the 0 directory's value for the whole run --
        // exact for one step, since both arms start there, and U 4.8e-06 from OpenFOAM at step two.
        const bool wallFunctionWrites = !les && patches[pi].type == "wall";
        const bool eval = !wallFunctionWrites
                       && patches[pi].type != "empty"
                       && t.nut.boundary[pi]->bcCategory() != 2;
        if (eval) d.nutEvalFaces += static_cast<int>(patches[pi].size);
        if (t.nut.boundary[pi]->isInletOutlet()) d.nutIoFaces += static_cast<int>(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            nutEval.push_back(eval ? 1 : 0);
        }
    }
    d.nutEvalMask.copyFrom(nutEval);
    // Every array above is now in the device's own boundary-face order, so this is a size check and
    // no longer a refusal: it catches a builder that starts walking every patch again.
    if (d.nutEvalFaces > 0 && static_cast<std::size_t>(d.dbNut.n) != nutEval.size())
    {
        throw std::runtime_error(
            "brae interFoam (device): nut's evaluate mask is " + std::to_string(nutEval.size())
            + " faces and its device boundary is " + std::to_string(d.dbNut.n)
            + ". They index each other, so one of the two is walking the coupled patches and the "
              "other is not.");
    }

    // A COUPLED PATCH RUNS HERE NOW, and it took five sites, each of which is a number that is
    // nearly right rather than a missing one:
    //   1. the transport matrix -- k and epsilon are fvm::div - fvm::laplacian, the momentum's shape,
    //      so across a pair they take the momentum's interface coefficient, with the diffusivity as
    //      CELLS (a coupled face takes fvc::interpolate's value) and the pair's OWN convecting flux;
    //   2. the solve, whose LDU view has to carry that off-diagonal or it runs a different operator
    //      from the matrix;
    //   3. fvc::grad(U) for the production term;
    //   4. divU and divPhi, EACH with its own flux -- the volumetric one and the equation's, which
    //      are one field only in the incompressible lineage;
    //   5. correctNut's boundary: correctNut() ends with nut_.correctBoundaryConditions()
    //      (kEpsilon.C:45-46), so a constraint patch takes lerp(pnf, pif, w) -- the two CELLS of the
    //      nut just written -- and not Cmu*k_b^2/eps_b, which is a non-linear function of k's and
    //      epsilon's own patch values.
    // WHAT IS STILL REFUSED is a leastSquares grad(U) on a pair (deviceLeastSquaresGradU carries no
    // interface) and a non-upwind div scheme for k or epsilon, both by name, at the sites themselves.

    d.wall = buildDeviceWallData(m, g, patches, U, wfPatch);

    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, patches);
    std::vector<label> mask;
    std::vector<label> kind;
    std::vector<label> wallFaceOfBnd;
    std::vector<scalar> yBnd;
    std::vector<scalar> nCmu25;
    std::vector<scalar> nKappa;
    std::vector<scalar> nE;
    std::vector<scalar> nYpl;
    std::vector<scalar> eCmu25;
    std::vector<scalar> eCmu75;
    std::vector<scalar> eKappa;
    std::vector<scalar> eE;
    std::vector<scalar> eYpl;
    label bndIdx = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // ...and neither is it in any of the per-face arrays below, which are indexed by the device's
        // boundary-face number. A coupled patch is never a wall and never a turbulence inlet.
        if (isCoupledInterfaceType(patches[pi].type)) continue;
        const bool isWF = isTurbWallPatch(patches, pi, wfPatch);
        // each wall function's OWN coefficients: the nut patch's for nutkWallFunction, the epsilon
        // patch's for epsilonWallFunction
        const WallFunctionCoeffs& nc = t.nut.boundary[pi]->wallCoeffs();
        // ...and the second field's, which a kEqn case does not have (and whose wall functions the
        // host reader refuses under that model, so there is nothing here to read)
        static const WallFunctionCoeffs noWf{};
        const WallFunctionCoeffs& ec = les ? noWf : second.boundary[pi]->wallCoeffs();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            // this face's index in boundary-face order; taken here because the loop has an early exit
            const label thisBnd = bndIdx;
            ++bndIdx;
            mask.push_back(isWF ? 1 : 0);
            yBnd.push_back(isWF ? yW[pi][i] : scalar(0));
            nCmu25.push_back(std::pow(nc.Cmu, 0.25));
            nKappa.push_back(nc.kappa);
            nE.push_back(nc.E);
            nYpl.push_back(nc.yPlusLam());
            // ...and nut's wall-function kind, which the host reader fills on the RAS path ONLY: a
            // kEqn case has no such patch (it refuses a `nut*` type under that model), so the list is
            // empty there and indexing it walked off the end.
            const int nwk = (pi < t.nutWallKind.size()) ? t.nutWallKind[pi] : -1;
            kind.push_back(nwk >= 0 ? nwk : static_cast<int>(NutWall::Nutk));
            if (!isWF) continue;
            // A PATCH beta1 IS NOT PORTED HERE. omegaWallFunction reads its own `beta1` from the patch
            // dictionary (omegaWallFunctionFvPatchScalarField.C:404-409) and the HOST closure uses it
            // (kOmegaSST_cpp.cu's omega wall loop); this closure's kernel takes beta1 as one scalar for
            // the whole field (device_komega_sst.cu:371), so a patch's own would be silently replaced by
            // the model's. Refused rather than run the wrong beta1 at the wall.
            if (!les && ec.beta1 != t.sstCoeffs.beta1)
                throw std::runtime_error(
                    "brae interFoam (device): omega patch `" + patches[pi].name + "` names `beta1 "
                    + std::to_string(ec.beta1) + "`, its own omegaWallFunction coefficient. The device "
                    "closure carries one beta1 for the whole field; the host honours the patch's. "
                    "Refused rather than run the model's value at the wall.");
            eCmu25.push_back(std::pow(ec.Cmu, 0.25));
            eCmu75.push_back(std::pow(ec.Cmu, 0.75));
            eKappa.push_back(ec.kappa);
            eE.push_back(ec.E);
            eYpl.push_back(ec.yPlusLam());
            wallFaceOfBnd.push_back(thisBnd);
        }
    }
    if (static_cast<int>(eCmu25.size()) != d.wall.nWF)
        throw std::runtime_error(
            "brae interFoam (device): the epsilon wall-function coefficient list ("
            + std::to_string(eCmu25.size()) + " faces) does not match the wall-face set ("
            + std::to_string(d.wall.nWF) + "); the two walks disagree on what a wall is.");
    d.wfBndMask.copyFrom(mask);
    d.wallYBndFace.copyFrom(yBnd);
    d.nutWfKindBnd.copyFrom(kind);
    d.nutWfCmu25Bnd.copyFrom(nCmu25);
    d.nutWfKappaBnd.copyFrom(nKappa);
    d.nutWfEBnd.copyFrom(nE);
    d.nutWfYplLamBnd.copyFrom(nYpl);
    d.wall.wfCmu25.copyFrom(eCmu25);
    d.wall.wfCmu75.copyFrom(eCmu75);
    d.wall.wfKappa.copyFrom(eKappa);
    d.wall.wfE.copyFrom(eE);
    d.wall.wfYplLam.copyFrom(eYpl);
    d.nWallFaces = static_cast<int>(wallFaceOfBnd.size());
    if (d.nWallFaces > 0)
    {
        d.wallFaceOfBnd.copyFrom(wallFaceOfBnd);
    }

    // The turbulent inlets, built exactly as rhoSimpleFoam's device arm builds them
    // (rhoCreateFields.cu:478-500): the patch itself says which kind it is and what its coefficient is,
    // so "no such patch" is told from "a coefficient that happens to be zero". Without these the device
    // closure freezes both inlets at the case file's `value` where OpenFOAM recomputes them from U --
    // MEASURED on RAS/angledDuct, whose inlet carries both: k 3.7e-01, epsilon 1.2e+00, U 9.0e-01
    // against OpenFOAM, where the HOST closure in the same device loop is at round-off.
    {
        std::vector<label>  km(static_cast<std::size_t>(bndIdx), 0), em(static_cast<std::size_t>(bndIdx), 0);
        std::vector<scalar> ki(static_cast<std::size_t>(bndIdx), scalar(0)), el(static_cast<std::size_t>(bndIdx), scalar(0));
        label bi = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (isCoupledInterfaceType(patches[pi].type)) continue;   // the device's boundary layout
            const int  kk = t.k.boundary[pi]->turbulentInletKind();
            const int  ek = les ? -1 : second.boundary[pi]->turbulentInletKind();
            const scalar kc = t.k.boundary[pi]->turbulentInletCoefficient();
            const scalar ec = les ? scalar(0) : second.boundary[pi]->turbulentInletCoefficient();
            for (label i = 0; i < patches[pi].size; ++i, ++bi)
            {
                if (bi >= bndIdx) break;
                if (kk == 0) { km[static_cast<std::size_t>(bi)] = 1; ki[static_cast<std::size_t>(bi)] = kc; d.hasTurbulentInlet = true; }
                if (ek == 1 || ek == 2) { em[static_cast<std::size_t>(bi)] = ek; el[static_cast<std::size_t>(bi)] = ec; d.hasTurbulentInlet = true; }
            }
        }
        if (d.hasTurbulentInlet)
        {
            d.turbInletKMask.copyFrom(km);
            d.turbInletEpsMask.copyFrom(em);
            d.turbInletKInt.copyFrom(ki);
            d.turbInletEpsLen.copyFrom(el);
        }
    }

    if (!t.variableDensity)
    {
        d.onesCell.copyFrom(std::vector<scalar>(static_cast<std::size_t>(m.nCells()), scalar(1)));
        d.onesBnd.copyFrom(std::vector<scalar>(static_cast<std::size_t>(bndIdx), scalar(1)));
    }

    // the LES filter width, from the host's own LESdelta::compute -- once, on a mesh that does not
    // move (the driver refuses a kEqn case whose mesh does)
    if (t.model == cpu::interFoam::InterRasModel::KEqnLES)
    {
        if (t.delta.size() != static_cast<std::size_t>(m.nCells()))
        {
            throw std::runtime_error(
                "brae interFoam (device): the LES filter width was not computed for this mesh. It is "
                "LESdelta::compute's, taken once by the host closure's own build.");
        }
        d.lesDelta.copyFrom(t.delta);
    }

    if (sst)
    {
        // wallDist::New(mesh).y(), as the host block resolved it (InterTurbulence::yCell -- the
        // diffusivity's patch set when a motion solver registered one)
        d.yCell.copyFrom(t.yCell);
        // ...and the two per-face masks the closure takes, as rhoCreateFields builds them: F1 is 1 by
        // construction on a wall or empty patch, and nut's field assignment reaches a `calculated`
        // patch only (fixedValueFvPatchField::operator= is a no-op)
        std::vector<label> f1One, nutCalc;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const bool f1IsOne = (patches[pi].type == "wall" || patches[pi].type == "empty");
            const bool calc = (t.nut.boundary[pi]->bcCategory() == 2);
            for (label i = 0; i < patches[pi].size; ++i)
            {
                f1One.push_back(f1IsOne ? 1 : 0);
                nutCalc.push_back(calc ? 1 : 0);
            }
        }
        d.f1OneMask.copyFrom(f1One);
        d.nutCalcMask.copyFrom(nutCalc);
    }
    return d;
}


void refreshDeviceInterTurbulenceGeometry(
    DeviceInterTurbulence& d,
    const cpu::interFoam::InterTurbulence& t,
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!t.on) return;
    const bool sst = (t.model == cpu::interFoam::InterRasModel::KOmegaSST);
    const bool les = (t.model == cpu::interFoam::InterRasModel::KEqnLES);
    const GeometricField<scalar>& second = sst ? t.omega : t.epsilon;

    // The SAME predicate the build walked, recomputed rather than stored: a stored copy is one more
    // thing that can go stale, and the two walks have to agree face for face or the arrays below
    // index each other wrongly.
    std::vector<char> wfPatch(patches.size(), 0);
    for (std::size_t pi = 0; pi < patches.size() && !les; ++pi)
    {
        wfPatch[pi] = second.boundary[pi]->isTurbulenceWallFunction() ? 1 : 0;
    }

    // The wall faces' y, deltaCoeffs and wall velocity. The wall velocity comes from the host U as it
    // stands, which is what the build took and what this driver's dbU is rebuilt from after a move.
    d.wall = buildDeviceWallData(m, g, patches, U, wfPatch);

    // ...and the per-boundary-face near-wall distance the nut and epsilon/omega wall functions read.
    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, patches);
    std::vector<scalar> yBnd;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (isCoupledInterfaceType(patches[pi].type)) continue;
        const bool isWF = isTurbWallPatch(patches, pi, wfPatch);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            yBnd.push_back(isWF ? yW[pi][i] : scalar(0));
        }
    }
    if (yBnd.size() != d.wallYBndFace.size())
        throw std::runtime_error(
            "brae interFoam (device): the mesh move changed the boundary-face count of the turbulence "
            "closure (" + std::to_string(d.wallYBndFace.size()) + " -> " + std::to_string(yBnd.size())
            + "). A move keeps the topology fixed; this is a topology change.");
    d.wallYBndFace.copyFrom(yBnd);

    // kOmegaSST's CELL wall distance, which F1 and F2 blend on. The host block's moveInterTurbulence
    // has already re-run wallDist's method on the moved points; this uploads what it produced.
    // ...and the LES filter width, which is the same MeshObject story: the host block's
    // moveInterTurbulence has just re-run LESdelta::compute on the moved cells.
    if (les)
    {
        if (t.delta.size() != d.lesDelta.size())
            throw std::runtime_error(
                "brae interFoam (device): the host LES filter width is " + std::to_string(t.delta.size())
                + " cells and the device holds " + std::to_string(d.lesDelta.size())
                + "; moveInterTurbulence did not run on this mesh.");
        d.lesDelta.copyFrom(t.delta);
    }

    if (sst)
    {
        if (t.yCell.size() != d.yCell.size())
            throw std::runtime_error(
                "brae interFoam (device): the host wall distance is " + std::to_string(t.yCell.size())
                + " cells and the device holds " + std::to_string(d.yCell.size())
                + "; moveInterTurbulence did not run on this mesh.");
        d.yCell.copyFrom(t.yCell);
    }
}


namespace {

// nutBnd[f] = evaluated[f] on the inletOutlet faces alone: every other face carries what the closure
// wrote (a wall function's value, or the field assignment's on a `calculated` patch).
__global__ void nutEvalCopyKernel(
    int nBf,
    const label* __restrict__ evalMask,
    const scalar* __restrict__ evaluated,
    scalar* __restrict__ nutBnd)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nBf && evalMask[f]) nutBnd[f] = evaluated[f];
}

// nut.correctBoundaryConditions() after the closure wrote the cells, as the host closure runs it
// (kOmegaSST_cpp.cu:946-984, kEpsilon_cpp.cu:899-910): the flux switch first on the flux-conditional
// faces -- OpenFOAM's valueFraction = neg(phi), so an inflow face takes the inletValue -- then the
// evaluate itself on every face the mask carries.
void evaluateNutBoundary(
    DeviceInterTurbulence& d,
    const DeviceBuffer<scalar>& phiBnd)
{
    if (d.nutEvalFaces <= 0) return;
    if (d.nutIoFaces > 0) deviceUpdateInletOutlet(d.dbNut, phiBnd);
    DeviceBuffer<scalar> evaluated;
    deviceBCValue(d.dbNut, d.nut, evaluated);
    const int nBf = static_cast<int>(d.nutBnd.size());
    if (nBf <= 0) return;
    constexpr int TPB = 256;
    nutEvalCopyKernel<<<(nBf + TPB - 1) / TPB, TPB>>>(nBf, d.nutEvalMask.data(), evaluated.data(),
                                                      d.nutBnd.data());
    cudaCheck(cudaGetLastError(), "nut correctBoundaryConditions");
}

}   // namespace


void deviceCorrectInterTurbulence(
    DeviceInterTurbulence& d,
    const cpu::interFoam::InterTurbulence& t,
    const DeviceInterTurbulenceStepInput& in,
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU)
{
    if (!t.on) return;
    // `turbulence off` -- the same early return the host closure takes, and OpenFOAM's own
    // (kEpsilon.C:216-219, kOmegaSSTBase.C:502-505, kEqn.C:141-144). The device closure is still BUILT,
    // because the model is still constructed and validated; it simply has nothing to advance, and the
    // nut it built from validate() is the nut nuEff keeps for the whole run.
    if (t.frozen) return;
    // the sentinels, refused rather than defaulted -- see the host twin
    if (in.timeIndex < 0 || in.finalIter < 0)
        throw std::runtime_error(
            "brae interFoam (device): deviceCorrectInterTurbulence needs the step's timeIndex and "
            "finalIter; the caller supplied " + std::to_string(in.timeIndex) + " and "
            + std::to_string(in.finalIter) + ".");
    advanceDeviceTurbulenceOldTime(d, in.timeIndex, /*second=*/t.model != cpu::interFoam::InterRasModel::KEqnLES);
    if (!in.Ux || !in.Uy || !in.Uz || !in.phiInt || !in.phiBnd || !in.rhoPhiInt || !in.rhoPhiBnd
     || !in.rho || !in.rhoBnd || !in.rhoOld || !in.nu || !in.nuBnd)
        throw std::runtime_error("brae interFoam (device): deviceCorrectInterTurbulence needs every input.");
    if (!(in.deltaT > 0))
        throw std::runtime_error("brae interFoam (device): deviceCorrectInterTurbulence needs a positive deltaT.");

    // nu on the WALL faces, and the ENTERING wall nut: OpenFOAM's G0 reads the stored nut patch value,
    // and nutBnd is also what correct() overwrites, so it is snapshotted rather than aliased
    deviceCopy(d.nutBndIn, d.nutBnd);
    if (d.nWallFaces > 0)
    {
        d.nuWall.resize(static_cast<std::size_t>(d.nWallFaces));
        d.nutWallIn.resize(static_cast<std::size_t>(d.nWallFaces));
        deviceGatherWallNu(d.wallFaceOfBnd, *in.nuBnd, d.nuWall);
        deviceGatherWallNu(d.wallFaceOfBnd, d.nutBndIn, d.nutWallIn);
        // THE WALL VELOCITY, refreshed here rather than kept from the build. The wall-function
        // production reads (U_wall - U_cell)*deltaCoeffs, and the host closure takes U_wall from the
        // patch's own values at every call (kEpsilon_cpp.cu:401). buildDeviceWallData snapshots it once,
        // which is exact only while the wall's velocity never moves -- true of every noSlip wall, and
        // false of a SLIP one. MEASURED on RAS/angledDuct, whose `porosityWall` is a 45-degree slip
        // wall: epsilon 2.1e-02 against OpenFOAM with its worst cells on that patch, where the host
        // closure in the same device loop is at round-off.
        for (int k = 0; k < 3; ++k)
        {
            const DeviceBuffer<scalar>* Uk = (k == 0) ? in.Ux : (k == 1) ? in.Uy : in.Uz;
            DeviceBuffer<scalar>& out = (k == 0) ? d.wall.wfUwx : (k == 1) ? d.wall.wfUwy : d.wall.wfUwz;
            DeviceBuffer<scalar> ub;
            deviceBCValue(dbU.comp[k], *Uk, ub);
            deviceGatherWallNu(d.wallFaceOfBnd, ub, out);
        }
    }

    if (t.model == cpu::interFoam::InterRasModel::KEqnLES)
    {
        // A TRANSCRIPTION of the host's kEqn branch (inter_turbulence_cpp.cu, correctInterTurbulence):
        // alpha = rho = 1, the VOLUMETRIC phi as the equation's flux and as divU's, the mixture's nu,
        // and the Final entries for the solve -- one outer corrector, so that is the entry in force.
        // Nothing here is a wall function: nut is Ck*sqrt(k)*delta everywhere.
        gpu::LESkEqn::Input lin;
        lin.Ux = in.Ux;
        lin.Uy = in.Uy;
        lin.Uz = in.Uz;
        lin.phiInt = in.phiInt;
        lin.phiBnd = in.phiBnd;
        lin.nu = in.nu;
        lin.nuBnd = in.nuBnd;
        lin.nutBnd = &d.nutBndIn;      // the nut patch values the LAST correctNut left, as DkEff reads
        lin.delta = &d.lesDelta;
        lin.rDeltaT = scalar(1) / in.deltaT;
        lin.co = t.lesCoeffs;
        // THE PAIR, as the two RAS branches carry it. One flux: this lineage is the uniform one, and
        // k convects with the volumetric phi like every other equation on it.
        lin.cyc    = in.cyc;
        lin.cycPhi = in.cycPhi;
        // ...and a MOVING mesh's two terms, as the RAS branches take them
        lin.V0         = in.V0;
        lin.meshPhiInt = in.meshPhiInt;
        lin.meshPhiBnd = in.meshPhiBnd;
        // ...and fvm::ddt(k) under CrankNicolson, with k's old-old level rotated once per time index.
        // alpha = rho = 1 in this lineage, so there is no rho old-old to carry.
        if (in.cn)
        {
            d.cnDdt0K.name = "ddt0(k)";
            lin.cn      = in.cn;
            lin.cnDdt0K = &d.cnDdt0K;
            lin.kOO     = &d.cnKOO;
        }
        // WHICH DICTIONARY THIS CORRECTOR USES, the same rule the host loop applies: `<field>Final` on
        // the final outer corrector and `<field>` on the others (fvMatrix.C:1536-1542 for solve(),
        // :1249-1263 for relax()). With `turbOnFinalIterOnly no` the closure runs on EVERY corrector,
        // so this is not always the Final entry.
        const bool fin = (in.finalIter != 0);
        const cpu::interFoam::SmoothLinearSolve& ks = fin ? t.kSolveFinal : t.kSolve;
        // the case's own solver for k, as the two RAS branches read it
        if (ks.pbicgDILU())
        {
            if (!d.dilu.valid)
                throw std::runtime_error(
                    "brae interFoam (device): kFinal names PBiCG with DILU and the closure was built "
                    "with no DILU schedule for this mesh.");
            lin.pbicg = true;
            lin.precon = &d.dilu;
        }
        lin.symmetric = (ks.smoother == "symGaussSeidel");
        lin.nSweeps = ks.nSweeps;
        lin.tol = ks.tol;
        lin.relTol = ks.relTol;
        lin.maxIter = ks.maxIter;
        lin.minIter = ks.minIter;
        lin.relaxOn = fin ? t.kRelaxFinal.on : t.kRelax.on;
        lin.relax = fin ? t.kRelaxFinal.factor : t.kRelax.factor;
        // k.oldTime(): the PREVIOUS STEP's field, advanced once per time index above -- not the field as
        // it enters this call, which is the previous CORRECTOR's once the closure runs on every one.
        lin.kOld = &d.kOldStep;
        const DeviceSolverPerf p = gpu::LESkEqn::correct(dm, d.dbK, dbU, d.k, d.nut, lin);
        if (in.kLog)
        {
            in.kLog->push_back({p.initialResidual, p.finalResidual, p.nIterations});
        }
        // nut.correctBoundaryConditions(), through the same call the RAS branches make -- the flux
        // switch on the flux-conditional faces, then the evaluate on every face the mask carries.
        // Nothing on a kEqn case is a wall function (the host reader refuses a `nut*` patch type
        // under this model), so what it leaves is each patch's own evaluate.
        evaluateNutBoundary(d, *in.phiBnd);
        return;
    }

    if (t.model == cpu::interFoam::InterRasModel::KOmegaSST)
    {
        // A TRANSCRIPTION of the host's SST branch (inter_turbulence_cpp.cu, correctInterTurbulence):
        // the ordinary incompressible kOmegaSST -- alpha = rho = 1, the VOLUMETRIC phi as the equation's
        // flux and as divU's, the mixture's nu, 1/deltaT for fvm::ddt, and phi for nut's
        // flux-conditional patches. `density variable` with kOmegaSST is refused on the host, so there
        // is no mass-flux lineage here. The host passes bounded, limitedLinear and linearUpwind off and
        // the limiter coefficient 1; the closure is the one rhoSimpleFoam gates (kOmegaSST.cuh).
        gpu::kOmegaSSTRAS::KOmegaSSTInput sin;
        sin.phiInt = in.phiInt;
        sin.phiBnd = in.phiBnd;
        sin.phiByRhoInt = in.phiInt;
        sin.phiByRhoBnd = in.phiBnd;
        // THE PAIR, exactly as the kEpsilon branch below hands it: the off-diagonal for the transport
        // matrix and for the solve, and the flux each equation convects with on those faces. This
        // closure refused a pair outright until the five sites the kEpsilon closure carries were
        // transcribed into it -- grad(U), the two divergences, each equation's gammaCell, and the
        // solve's interface -- so `hasCoupledPatches` is no longer set here. The uniform lineage has
        // one flux, so the volumetric and the equation's are the same array.
        sin.cyc         = in.cyc;
        sin.cycPhi      = in.cycPhi;
        sin.cycPhiByRho = in.cycPhi;
        // ...and fvm::ddt under CrankNicolson on BOTH equations, which is how kOmegaSSTBase takes it
        // (ddtSchemes at :572 and :602). The second field is omega here and lives in the kEpsilon
        // slots -- d.epsilon, d.cnEpsOO, d.cnDdt0Eps -- as every other slot on this branch does.
        if (in.cn)
        {
            d.cnDdt0K.name   = "ddt0(k)";
            d.cnDdt0Eps.name = "ddt0(omega)";
            sin.cn          = in.cn;
            sin.cnDdt0K     = &d.cnDdt0K;
            sin.cnDdt0Omega = &d.cnDdt0Eps;
            sin.kOO         = &d.cnKOO;
            sin.omegaOO     = &d.cnEpsOO;
            // the uniform lineage: rho is 1 at every level, and the variable one is refused below
            sin.rhoOOCell   = &d.onesCell;
        }
        sin.rhoCell = &d.onesCell;
        sin.rhoBndFace = &d.onesBnd;
        sin.rhoOldCell = &d.onesCell;
        // ...and its turbulent inlets, which kOmegaSST takes under the same names (kOmegaSST.cuh);
        // the mask's 2 selects the frequency form, omega = sqrt(k)/(Cmu^0.25*L)
        if (d.hasTurbulentInlet)
        {
            sin.turbInletKMask   = &d.turbInletKMask;
            sin.turbInletKInt    = &d.turbInletKInt;
            sin.turbInletOmegaMask = &d.turbInletEpsMask;
            sin.turbInletOmegaLen  = &d.turbInletEpsLen;
        }
        sin.rDeltaT = scalar(1) / in.deltaT;
        // psi.oldTime(), per STEP -- see DeviceInterTurbulence::kOldStep
        sin.kOldIn = &d.kOldStep;
        sin.omegaOldIn = &d.epsOldStep;
        // the moved mesh's old volumes and its flux -- see DeviceInterTurbulenceStepInput::V0
        sin.V0         = in.V0;
        sin.meshPhiInt = in.meshPhiInt;
        sin.meshPhiBnd = in.meshPhiBnd;
        sin.nuCell = in.nu;
        sin.nuBndFace = in.nuBnd;
        sin.nuWallFace = &d.nuWall;
        sin.nutBndFace = &d.nutBndIn;
        sin.nutWallFace = (d.nWallFaces > 0) ? &d.nutWallIn : nullptr;
        sin.wfBndMask = &d.wfBndMask;
        sin.wallYBndFace = &d.wallYBndFace;
        sin.f1OneMask = &d.f1OneMask;
        sin.nutCalcMask = &d.nutCalcMask;
        sin.nutWfKindBnd = &d.nutWfKindBnd;
        sin.nutWfCmu25Bnd = &d.nutWfCmu25Bnd;
        sin.nutWfKappaBnd = &d.nutWfKappaBnd;
        sin.nutWfEBnd = &d.nutWfEBnd;
        sin.nutWfYplLamBnd = &d.nutWfYplLamBnd;
        sin.Ux = in.Ux;
        sin.Uy = in.Uy;
        sin.Uz = in.Uz;
        sin.yCell = &d.yCell;
        sin.co = t.sstCoeffs;
        // THE CASE'S CONVECTION SCHEME for the pair, and the LIMITER's own gradient entry. Every one
        // of these is a field this site never filled, so the device closure convected with upwind
        // where the case said `Gauss limitedLinear <k>` and the host limited it -- the substitution
        // this project keeps finding, and the third time this struct has had a coefficient carried
        // twice. `limGradK`/`limGradLeastSq` are the gradient the LIMITER takes (the host hands
        // divWithScheme co.gradKLimitK/gradKLeastSq); they are the same fvSchemes entry as
        // co.gradKLimitK below, and the closure refuses the two disagreeing.
        // ONE SCHEME FOR BOTH EQUATIONS is what this closure's kernels carry, so a case whose two
        // div entries differ is refused rather than run under k's -- the host honours it per equation
        // (tests/interfoam_ras_dambreak_vs_openfoam.sh `splitDiv`).
        if (t.kDiv.limitedLinear != t.secondDiv.limitedLinear
         || (t.kDiv.limitedLinear && t.kDiv.limiterCoeff != t.secondDiv.limiterCoeff))
            throw std::runtime_error(
                "brae interFoam (device): fvSchemes gives div(phi,k) and div(phi,omega) different "
                "convection schemes; this closure carries one scheme for both equations.");
        // ...and the GRADIENT of each equation, which this closure also carries as ONE pair: `sin.co` is
        // t.sstCoeffs, whose gradKLeastSq/gradKLimitK are k's. The host honours grad(k) and grad(omega)
        // separately (measured on RAS/waterChannel: giving grad(omega) its own scheme moves OpenFOAM's
        // own omega 9.4e-02 over all 28,000 cells), so a mismatch is refused here rather than run under
        // k's gradient.
        if (t.kGrad.leastSquares != t.secondGrad.leastSquares
         || t.kGrad.cellLimitK != t.secondGrad.cellLimitK)
            throw std::runtime_error(
                "brae interFoam (device): fvSchemes gives grad(k) and grad(omega) different schemes; "
                "this closure carries one gradient for both equations.");
        sin.limitedLinear   = t.kDiv.limitedLinear;
        sin.limiterCoeff    = t.kDiv.limiterCoeff;
        // ...and the CONVECTION scheme, which this arm now RUNS. It was refused by name, and the
        // refusal was for want of a gate rather than for a measured defect: the only case that names
        // the scheme -- RAS/waterChannel `limitedLinear` -- cannot witness it on fields, because it
        // amplifies the last bit of the turbulence fields by fourteen orders over ten steps. ONE ULP
        // on the initial omega field, run on the HOST, reads k 4.5e-02, omega 4.3e-02, nut 1.7e-02
        // there; this arm reads 1.4724e-05 / 1.7822e-04 / 1.6747e-05, two orders BELOW that, and the
        // same numbers it read while it was silently convecting with upwind. No field bound on that
        // profile can separate the scheme from the last bit.
        //
        // WHAT LIFTED IT: tests/interfoam_sst_assembly_vs_openfoam.sh, which holds this arm's
        // ASSEMBLED SYSTEM against OpenFOAM's own (tools/dumpKOmegaSST) at the first closure call --
        // omega and k, both off-diagonals, and D and Src on the 21,690 rows setValues does not
        // eliminate: 3.4e-14, 3.9e-14 and 5.4e-13, as close as the host's own 3.2e-14. Its CONTROL is
        // OpenFOAM's `Gauss upwind` system, which the same comparison misses by 5.0e-01, and its
        // second control is this arm's own upwind against OpenFOAM's upwind (3.6e-14), so the gap in
        // the first is the LIMITER and not a broken upwind path.
        //
        // The kEpsilon closure below runs it too, gated the same way on RAS/damBreak
        // (tests/interfoam_kepsilon_assembly_vs_openfoam.sh).
        //
        // NOTHING IS SET HERE FOR grad(U)'s LIMITER any more. This site used to have to fill
        // `sin.gradULimitK` as well as `sin.co.gradULimitK` -- one fvSchemes entry in two fields of
        // one struct -- and filling only the second ran the production strain on an UNLIMITED
        // gradient while the host limited it (nut 4.1315e-01 on validation/interFoamCyclic
        // `sstLimU`). The second field is gone; `sin.co = t.sstCoeffs` above carries the entry once.

        sin.correctedLaplacian = t.coeffs.correctedLaplacian;
        sin.nonOrthCoeffs = t.coeffs.nonOrthCoeffs;
        sin.snGradLimitCoeff = t.coeffs.snGradLimitCoeff;
        // the Final entries, as the kEpsilon branch below takes them: one outer corrector, so that one
        // is the final one
        // the corrector's own entries -- see the LES branch above for the rule
        const bool fin = (in.finalIter != 0);
        sin.relaxEquationOmega = fin ? t.omegaRelaxFinal.on : t.omegaRelax.on;
        sin.relaxOmega = fin ? t.omegaRelaxFinal.factor : t.omegaRelax.factor;
        sin.relaxEquationK = fin ? t.kRelaxFinal.on : t.kRelax.on;
        sin.relaxK = fin ? t.kRelaxFinal.factor : t.kRelax.factor;
        const cpu::interFoam::SmoothLinearSolve& ks = fin ? t.kSolveFinal : t.kSolve;
        const cpu::interFoam::SmoothLinearSolve& os = fin ? t.omegaSolveFinal : t.omegaSolve;
        // omegaFinal NEED NOT MATCH kFinal: fvMatrix::solve() looks the dictionary up by FIELD name
        // (fvMatrix.C:1536-1542) and kOmegaSSTBase.C:593 solves omega with its own entry, :618 k with
        // kFinal's. This branch compared all eight fields and refused any difference; the host has
        // honoured the pair since the per-equation solver unit, and the closure now takes omega's own
        // through KOmegaSSTInput::omegaSolve.
        gpu::turbulence::SolveControls svOmega;
    // NON-DICTIONARY FIELDS COME FROM k's, not from a default. The closure's own builder fills
    // colouring/gsColour/polyDeg/precon from `sin` (kEpsilon.cu's finishAndSolve, kOmegaSST.cu's solveOf),
    // and only the eight the SOLVER ENTRY decides differ per equation. Leaving them default here ran the
    // second equation with a different colouring and polynomial degree from k's -- which the defaults
    // audit flagged as three fields set at one of the sites and not the other.
    svOmega.gsColour  = sin.gsColour;
    svOmega.colouring = sin.colouring;
    svOmega.polyDeg   = sin.polyDeg;
        svOmega.tol         = os.tol;
        svOmega.relTol      = os.relTol;
        svOmega.maxIter     = os.maxIter;
        svOmega.minIter     = os.minIter;
        svOmega.nSweeps     = os.nSweeps;
        svOmega.gsSymmetric = (os.smoother == "symGaussSeidel");
        svOmega.pbicg       = os.pbicgDILU();
        // THE FAMILY IS STILL CHECKED, per equation. Without this a second equation naming PBiCGStab or
        // GAMG would fall through to BiCGStab silently -- the substitution the kFinal message below
        // exists to prevent, which the lift would otherwise have reopened for omega alone.
        if (!os.gaussSeidel() && !os.pbicgDILU())
            throw std::runtime_error(
                "brae interFoam (device): `solvers/omegaFinal` names `solver " + os.solver
                + "; smoother " + os.smoother + "; preconditioner " + os.preconditioner + ";`, which the "
                  "device kOmegaSST does not run: a Gauss-Seidel smoothSolver, or PBiCG with DILU, and "
                  "nothing else. A substituted solver at the same tolerance stops somewhere else.");
        if (os.pbicgDILU())
        {
            if (!d.dilu.valid)
                throw std::runtime_error(
                    "brae interFoam (device): omegaFinal names PBiCG with DILU and the SST closure was "
                    "built with no DILU schedule for this mesh.");
            svOmega.precon = &d.dilu;
        }
        sin.omegaSolve = &svOmega;
        // THE SOLVER THE CASE NAMES, and only that -- the kEpsilon branch below has read it since
        // waves/mangroveInteraction, and this one did not: it set gsK and gsOmega unconditionally, so
        // `solver PBiCG; preconditioner DILU;` on kFinal and omegaFinal ran symGaussSeidel sweeps
        // under PBiCG's tolerance and said nothing.
        if (ks.pbicgDILU())
        {
            if (!d.dilu.valid)
                throw std::runtime_error(
                    "brae interFoam (device): kFinal names PBiCG with DILU and the SST closure was "
                    "built with no DILU schedule for this mesh.");
            sin.pbicgKE = true;
            sin.precon = &d.dilu;
        }
        else if (ks.gaussSeidel())
        {
            sin.gsK = true;
            sin.gsOmega = true;
        }
        else
        {
            throw std::runtime_error(
                "brae interFoam (device): kFinal names `solver " + ks.solver + "`, which the device "
                "kOmegaSST does not run: a Gauss-Seidel smoothSolver, or PBiCG with DILU, and nothing "
                "else.");
        }
        sin.gsSymmetric = (ks.smoother == "symGaussSeidel");
        sin.nSweepsKE = ks.nSweeps;
        sin.tol = ks.tol;
        sin.relTol = ks.relTol;
        sin.maxIter = ks.maxIter;
        sin.minIter = ks.minIter;

        gpu::kOmegaSSTRAS::KOmegaSSTResiduals sres;
        gpu::kOmegaSSTRAS::correct(d.k, d.epsilon, d.nut, d.nutBnd, /*alphat=*/nullptr,
                                   /*alphatBnd=*/nullptr, sres, dm, dbU, d.dbK, d.dbEps, d.wall, sin);
        if (in.epsilonLog)
        {
            in.epsilonLog->push_back({sres.omegaPerf.initialResidual, sres.omegaPerf.finalResidual,
                                      sres.omegaPerf.nIterations});
        }
        if (in.kLog)
        {
            in.kLog->push_back({sres.kPerf.initialResidual, sres.kPerf.finalResidual,
                                sres.kPerf.nIterations});
        }
        evaluateNutBoundary(d, *in.phiBnd);
        return;
    }

    gpu::kEpsilonRAS::KEpsilonInput kin;
    // divU and the flux-conditional patches read the volumetric phi in BOTH lineages
    kin.phiByRhoInt = in.phiInt;
    kin.phiByRhoBnd = in.phiBnd;
    // THE PAIR, with each flux going where its internal-face twin goes. Without this the closure
    // solves k and epsilon as if the pair were two walls -- MEASURED on RAS/damBreakPorousBaffle,
    // twenty steps against OpenFOAM: k 4.7e-01, epsilon 4.6e-01, nut 1.8e-01.
    kin.cyc         = in.cyc;
    kin.cycPhiByRho = in.cycPhi;
    kin.bcPhiBnd = in.phiBnd;
    // ...and the same for kEpsilon's device closure.
    // as the SST branch above: one scheme for both equations in the kernels, so a mismatch is refused
    if (t.kDiv.limitedLinear != t.secondDiv.limitedLinear
     || (t.kDiv.limitedLinear && t.kDiv.limiterCoeff != t.secondDiv.limiterCoeff))
        throw std::runtime_error(
            "brae interFoam (device): fvSchemes gives div(phi,k) and div(phi,epsilon) different "
            "convection schemes; this closure carries one scheme for both equations.");
    // ...and the gradients, as the SST branch above: one pair in the kernels, so a mismatch is refused
    if (t.kGrad.leastSquares != t.secondGrad.leastSquares
     || t.kGrad.cellLimitK != t.secondGrad.cellLimitK)
        throw std::runtime_error(
            "brae interFoam (device): fvSchemes gives grad(k) and grad(epsilon) different schemes; "
            "this closure carries one gradient for both equations.");
    kin.limitedLinear  = t.kDiv.limitedLinear;
    kin.limiterCoeff   = t.kDiv.limiterCoeff;
    // ...and the same for kEpsilon's device closure, which RUNS it now too. It was refused as
    // "ungated", and that is what it was: the kOmegaSST half was lifted on its assembled system and
    // this half had no oracle at all, because tools/dumpKEpsilon registered only the compressible
    // table and interFoam's RAS lineage is PhaseIncompressibleTurbulenceModel<transportModel>.
    //
    // WHAT LIFTED IT: tests/interfoam_kepsilon_assembly_vs_openfoam.sh, on RAS/damBreak with
    // `Gauss limitedLinear 1` for div(rhoPhi,(k|epsilon)) -- this arm's assembled epsilon and k
    // systems against OpenFOAM's own at the first closure call, 1.8e-15 on the off-diagonals and
    // 1.3e-14 on the diagonals of the rows setValues leaves alive, as close as the host's own
    // 1.8e-15 / 1.7e-14. Its controls are OpenFOAM's `Gauss upwind` system (missed by far more than
    // the floor, both ways round) and this arm's own upwind against OpenFOAM's upwind, so the gap in
    // the first is the LIMITER and not a broken upwind path.

    if (t.variableDensity)
    {
        kin.phiInt = in.rhoPhiInt;
        kin.phiBnd = in.rhoPhiBnd;
        kin.cycPhi = in.cycRhoPhi;
        kin.rhoCell = in.rho;
        kin.rhoBndFace = in.rhoBnd;
        kin.rhoOldCell = in.rhoOld;
    }
    else
    {
        kin.phiInt = in.phiInt;
        kin.phiBnd = in.phiBnd;
        kin.cycPhi = in.cycPhi;
        kin.rhoCell = &d.onesCell;
        kin.rhoBndFace = &d.onesBnd;
        kin.rhoOldCell = &d.onesCell;
    }
    kin.rDeltaT = scalar(1) / in.deltaT;
    // psi.oldTime(), per STEP -- see DeviceInterTurbulence::kOldStep
    kin.kOldIn = &d.kOldStep;
    kin.epsOldIn = &d.epsOldStep;
    kin.nuCell = in.nu;
    kin.nuBndFace = in.nuBnd;
    kin.nuWallFace = &d.nuWall;
    kin.nutBndFace = &d.nutBndIn;
    kin.nutWallFace = (d.nWallFaces > 0) ? &d.nutWallIn : nullptr;
    kin.wfBndMask = &d.wfBndMask;
    kin.wallYBndFace = &d.wallYBndFace;
    kin.nutWfKindBnd = &d.nutWfKindBnd;
    kin.nutWfCmu25Bnd = &d.nutWfCmu25Bnd;
    kin.nutWfKappaBnd = &d.nutWfKappaBnd;
    kin.nutWfEBnd = &d.nutWfEBnd;
    kin.nutWfYplLamBnd = &d.nutWfYplLamBnd;
    kin.Ux = in.Ux;
    kin.Uy = in.Uy;
    kin.Uz = in.Uz;
    kin.co = t.coeffs;
    kin.correctedLaplacian = t.coeffs.correctedLaplacian;
    kin.nonOrthCoeffs = t.coeffs.nonOrthCoeffs;
    kin.snGradLimitCoeff = t.coeffs.snGradLimitCoeff;
    // the Final entries: the device loop runs one outer corrector, and that one is the final one
    // The turbulent inlets, as rhoSimpleFoam's hook passes them (rhoTurbulenceHook.cu:128-136):
    // passing null is not "no turbulent inlet", it is silently no turbulent inlet at all on a case whose
    // 0/k asks for one.
    if (d.hasTurbulentInlet)
    {
        kin.turbInletKMask   = &d.turbInletKMask;
        kin.turbInletKInt    = &d.turbInletKInt;
        kin.turbInletEpsMask = &d.turbInletEpsMask;
        kin.turbInletEpsLen  = &d.turbInletEpsLen;
    }
    // the corrector's own entries -- see the LES branch above for the rule
    const bool fin = (in.finalIter != 0);
    kin.relaxEquationEps = fin ? t.epsRelaxFinal.on : t.epsRelax.on;
    kin.relaxEps = fin ? t.epsRelaxFinal.factor : t.epsRelax.factor;
    kin.relaxEquationK = fin ? t.kRelaxFinal.on : t.kRelax.on;
    kin.relaxK = fin ? t.kRelaxFinal.factor : t.kRelax.factor;
    const cpu::interFoam::SmoothLinearSolve& ks = fin ? t.kSolveFinal : t.kSolve;
    const cpu::interFoam::SmoothLinearSolve& es = fin ? t.epsSolveFinal : t.epsSolve;
    // epsilonFinal NEED NOT MATCH kFinal -- see the SST branch above. kEpsilon.C:268 solves epsilon with
    // its own entry and :288 solves k with kFinal's.
    gpu::turbulence::SolveControls svEps;
    // NON-DICTIONARY FIELDS COME FROM k's, not from a default. The closure's own builder fills
    // colouring/gsColour/polyDeg/precon from `kin` (kEpsilon.cu's finishAndSolve, kOmegaSST.cu's solveOf),
    // and only the eight the SOLVER ENTRY decides differ per equation. Leaving them default here ran the
    // second equation with a different colouring and polynomial degree from k's -- which the defaults
    // audit flagged as three fields set at one of the sites and not the other.
    svEps.gsColour  = kin.gsColour;
    svEps.colouring = kin.colouring;
    svEps.polyDeg   = kin.polyDeg;
    svEps.tol         = es.tol;
    svEps.relTol      = es.relTol;
    svEps.maxIter     = es.maxIter;
    svEps.minIter     = es.minIter;
    svEps.nSweeps     = es.nSweeps;
    svEps.gsSymmetric = (es.smoother == "symGaussSeidel");
    svEps.pbicg       = es.pbicgDILU();
    if (!es.gaussSeidel() && !es.pbicgDILU())
        throw std::runtime_error(
            "brae interFoam (device): `solvers/epsilonFinal` names `solver " + es.solver + "; smoother "
            + es.smoother + "; preconditioner " + es.preconditioner + ";`, which the device kEpsilon does "
              "not run: a Gauss-Seidel smoothSolver, or PBiCG with DILU, and nothing else.");
    if (es.pbicgDILU())
    {
        if (!d.dilu.valid)
            throw std::runtime_error(
                "brae interFoam (device): epsilonFinal names PBiCG with DILU and the closure was built "
                "with no DILU schedule for this mesh.");
        svEps.precon = &d.dilu;
    }
    kin.epsSolve = &svEps;
    // THE SOLVER THE CASE NAMES, and only that. The host reader admits two for kEpsilon: a
    // Gauss-Seidel smoothSolver and PBiCG with DILU. This branch set the first unconditionally, so a
    // case naming PBiCG -- waves/mangroveInteraction -- would have run symGaussSeidel sweeps under
    // PBiCG's tolerance and said nothing; it was only ever refused for its fvOptions.
    if (ks.pbicgDILU())
    {
        if (!d.dilu.valid)
            throw std::runtime_error(
                "brae interFoam (device): kFinal names PBiCG with DILU and the closure was built with no "
                "DILU schedule for this mesh.");
        kin.pbicgKE = true;
        kin.precon = &d.dilu;
    }
    else if (ks.gaussSeidel())
    {
        kin.gsK = true;
        kin.gsEps = true;
    }
    else
    {
        throw std::runtime_error(
            "brae interFoam (device): kFinal names `solver " + ks.solver + "`, which the device kEpsilon "
            "does not run: a Gauss-Seidel smoothSolver, or PBiCG with DILU, and nothing else.");
    }
    // + fvOptions(epsilon) and + fvOptions(k): the mangroves' turbulence source at the U this closure
    // was handed, which is the one OpenFOAM's lookupObject finds when kEpsilon::correct builds them
    if (in.mangroves && in.mangroves->turbulence())
    {
        // -Sp(rho*coeff) on the DENSITY-WEIGHTED lineage, -Sp(coeff) on the uniform one: `fvOptions
        // (alpha, rho, k_)` with alpha one dispatches to the rho overload (fvOptionListTemplates.C:
        // 283-289). rho multiplies the coefficient, not the volume, as the host reference has it.
        const DeviceBuffer<scalar>* mgRho = t.variableDensity ? in.rho : nullptr;
        if (t.variableDensity && !mgRho)
            throw std::runtime_error(
                "brae interFoam (device): multiphaseMangrovesTurbulenceModel under the `density variable` "
                "k-epsilon needs the mixture's rho, and the caller supplied none.");
        // ONE COEFFICIENT FIELD PER OPTION -- each option's Ckp*Cd*a*N*|U| stands on its own, because
        // the k equation takes -Sp from each of them in turn (kEpsilon.C:279 sums the list)
        const std::size_t nOpt = in.mangroves->turbulences.size();
        if (d.mangroveK.size() != nOpt)
        {
            d.mangroveK.clear();
            d.mangroveEps.clear();
            d.mangroveK.resize(nOpt);
            d.mangroveEps.resize(nOpt);
        }
        kin.fvoSpK.clear();
        kin.fvoSpEps.clear();
        for (std::size_t i = 0; i < nOpt; ++i)
        {
            deviceMangrovesCoeff(in.mangroves->turbulences[i].kFac, *in.Ux, *in.Uy, *in.Uz, mgRho, d.mangroveK[i]);
            deviceMangrovesCoeff(in.mangroves->turbulences[i].epsFac, *in.Ux, *in.Uy, *in.Uz, mgRho, d.mangroveEps[i]);
            kin.fvoSpK.push_back(&d.mangroveK[i]);
            kin.fvoSpEps.push_back(&d.mangroveEps[i]);
        }
    }
    // fvm::ddt under CrankNicolson (the host wrapper's block, inter_turbulence_cpp.cu): rotate the
    // old-old levels once per time index -- at the first step oldTime().oldTime() is a copy of oldTime()
    // -- and hand the closure its two ddt0 fields
    if (in.cn)
    {
        d.cnDdt0K.name = t.variableDensity ? "ddt0(rho,k)" : "ddt0(k)";
        d.cnDdt0Eps.name = t.variableDensity ? "ddt0(rho,epsilon)" : "ddt0(epsilon)";
        kin.cn = in.cn;
        kin.cnDdt0K = &d.cnDdt0K;
        kin.cnDdt0Eps = &d.cnDdt0Eps;
        kin.kOO = &d.cnKOO;
        kin.epsOO = &d.cnEpsOO;
        if (t.variableDensity)
        {
            if (!in.rhoOO)
                throw std::runtime_error(
                    "brae interFoam (device): CrankNicolson under `density variable` needs rho.oldTime().oldTime().");
            kin.rhoOOCell = in.rhoOO;
        }
        else
        {
            kin.rhoOOCell = &d.onesCell;
        }
    }
    kin.gsSymmetric = (ks.smoother == "symGaussSeidel");
    kin.nSweepsKE = ks.nSweeps;
    kin.tol = ks.tol;
    kin.relTol = ks.relTol;
    kin.maxIter = ks.maxIter;
    kin.minIter = ks.minIter;

    gpu::kEpsilonRAS::correct(d.k, d.epsilon, d.nut, d.nutBnd, /*alphat=*/nullptr,
                              /*alphatBnd=*/nullptr, d.stages, dm, dbU, d.dbK, d.dbEps, d.wall, kin);
    if (in.epsilonLog)
    {
        in.epsilonLog->push_back({d.stages.epsPerf.initialResidual, d.stages.epsPerf.finalResidual,
                                  d.stages.epsPerf.nIterations});
    }
    if (in.kLog)
    {
        in.kLog->push_back({d.stages.kPerf.initialResidual, d.stages.kPerf.finalResidual,
                            d.stages.kPerf.nIterations});
    }
    evaluateNutBoundary(d, *in.phiBnd);
}


void downloadDeviceInterTurbulence(
    const DeviceInterTurbulence& d,
    cpu::interFoam::InterTurbulence& t,
    const std::vector<FvPatch>& patches)
{
    if (!t.on) return;
    // the second field is ::omega under kOmegaSST, ::epsilon otherwise -- the same slot the upload took
    GeometricField<scalar>& second = (t.model == cpu::interFoam::InterRasModel::KOmegaSST) ? t.omega
                                                                                          : t.epsilon;
    d.k.copyTo(t.k.internal);
    // ...and the second field, which a kEqn case does not have
    if (t.model != cpu::interFoam::InterRasModel::KEqnLES) d.epsilon.copyTo(second.internal);
    d.nut.copyTo(t.nut.internal);
    t.k.evaluateBoundary();
    if (t.model != cpu::interFoam::InterRasModel::KEqnLES) second.evaluateBoundary();
    std::vector<scalar> nb;
    d.nutBnd.copyTo(nb);
    std::size_t off = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // A COUPLED PATCH IS NOT IN d.nutBnd (the device's boundary layout), so walking every patch
        // here reads the next patch's faces and runs off the end at the last -- the same shape the
        // driver's flattens had. What it takes instead is what OpenFOAM gives it: correctNut() ends
        // with nut_.correctBoundaryConditions() (kEpsilon.C:45-46), and on a constraint patch that is
        // coupledFvPatchField::evaluate, lerp(patchNeighbourField, patchInternalField, weights) --
        // the two CELLS of the nut just written, and not Cmu*k_b^2/eps_b.
        if (isCoupledInterfaceType(patches[pi].type))
        {
            t.nut.boundary[pi]->evaluate(t.nut.internal);
            continue;
        }
        const std::size_t n = static_cast<std::size_t>(patches[pi].size);
        t.nut.boundary[pi]->setValue(std::vector<scalar>(nb.begin() + off, nb.begin() + off + n));
        off += n;
    }
}

} // namespace brae
