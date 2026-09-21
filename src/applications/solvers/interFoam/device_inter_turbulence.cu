// interFoam's turbulence on the device. See device_inter_turbulence.cuh for what each input is.
#include "device_inter_turbulence.cuh"
#include "device_les_keqn.cuh"
#include "device_blas.cuh"
#include "near_wall_dist.cuh"
#include "nut_wall_function.cuh"
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
    if (t.model == cpu::interFoam::InterRasModel::KEpsilon && t.kSolveFinal.pbicgDILU())
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
        const cpu::interFoam::SmoothLinearSolve& ks = t.kSolveFinal;
        lin.symmetric = (ks.smoother == "symGaussSeidel");
        lin.nSweeps = ks.nSweeps;
        lin.tol = ks.tol;
        lin.relTol = ks.relTol;
        lin.maxIter = ks.maxIter;
        lin.minIter = ks.minIter;
        lin.relaxOn = t.kRelaxFinal.on;
        lin.relax = t.kRelaxFinal.factor;
        // k.oldTime(): the field as it enters this call, which is what fvm::ddt reads
        DeviceBuffer<scalar> kOld;
        deviceCopy(kOld, d.k);
        lin.kOld = &kOld;
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
        sin.correctedLaplacian = t.coeffs.correctedLaplacian;
        sin.snGradLimitCoeff = t.coeffs.snGradLimitCoeff;
        // the Final entries, as the kEpsilon branch below takes them: one outer corrector, so that one
        // is the final one
        sin.relaxEquationOmega = t.omegaRelaxFinal.on;
        sin.relaxOmega = t.omegaRelaxFinal.factor;
        sin.relaxEquationK = t.kRelaxFinal.on;
        sin.relaxK = t.kRelaxFinal.factor;
        const cpu::interFoam::SmoothLinearSolve& ks = t.kSolveFinal;
        const cpu::interFoam::SmoothLinearSolve& os = t.omegaSolveFinal;
        if (ks.smoother != os.smoother || ks.tol != os.tol || ks.relTol != os.relTol
         || ks.maxIter != os.maxIter || ks.minIter != os.minIter || ks.nSweeps != os.nSweeps)
            throw std::runtime_error(
                "brae interFoam (device): fvSolution gives kFinal and omegaFinal different solver "
                "settings; the closure takes one set for both equations.");
        sin.gsK = true;
        sin.gsOmega = true;
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
    kin.relaxEquationEps = t.epsRelaxFinal.on;
    kin.relaxEps = t.epsRelaxFinal.factor;
    kin.relaxEquationK = t.kRelaxFinal.on;
    kin.relaxK = t.kRelaxFinal.factor;
    const cpu::interFoam::SmoothLinearSolve& ks = t.kSolveFinal;
    const cpu::interFoam::SmoothLinearSolve& es = t.epsSolveFinal;
    if (ks.solver != es.solver || ks.preconditioner != es.preconditioner
     || ks.smoother != es.smoother || ks.tol != es.tol || ks.relTol != es.relTol
     || ks.maxIter != es.maxIter || ks.minIter != es.minIter || ks.nSweeps != es.nSweeps)
        throw std::runtime_error(
            "brae interFoam (device): fvSolution gives kFinal and epsilonFinal different solver "
            "settings; the closure takes one set for both equations.");
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
    if (in.mangroves && in.mangroves->turbulence)
    {
        if (t.variableDensity)
            throw std::runtime_error(
                "brae interFoam (device): multiphaseMangrovesTurbulenceModel under the `density variable` "
                "k-epsilon is -Sp(rho*coeff) in OpenFOAM, and no gate holds that form.");
        deviceMangrovesCoeff(in.mangroves->kFac, *in.Ux, *in.Uy, *in.Uz, d.mangroveK);
        deviceMangrovesCoeff(in.mangroves->epsFac, *in.Ux, *in.Uy, *in.Uz, d.mangroveEps);
        kin.fvoSpK = &d.mangroveK;
        kin.fvoSpEps = &d.mangroveEps;
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
