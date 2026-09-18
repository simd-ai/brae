// interFoam's turbulence on the device. See device_inter_turbulence.cuh for what each input is.
#include "device_inter_turbulence.cuh"
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

    // The device closure builds its wall set from patches that are BOTH a `wall` and carry the
    // wall function (isTurbWallPatch); the host reference asks the boundary condition alone. They
    // agree on every shipped tutorial, and where they would not the device must say so.
    std::vector<char> wfPatch(patches.size(), 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const bool wf = t.epsilon.boundary[pi]->isTurbulenceWallFunction();
        wfPatch[pi] = wf ? 1 : 0;
        if (wf && patches[pi].type != "wall")
            throw std::runtime_error(
                "brae interFoam (device): patch `" + patches[pi].name + "` carries an "
                "epsilonWallFunction and is a `" + patches[pi].type + "`, not a `wall`. The device "
                "closure constrains wall patches only. The host path (no -device) runs it.");
    }

    d.k.copyFrom(t.k.internal);
    d.epsilon.copyFrom(t.epsilon.internal);
    d.nut.copyFrom(t.nut.internal);
    d.nutBnd.copyFrom(patchValuesOf(t.nut, patches));
    // storedIoSeed: the closure reconstructs an inletOutlet patch's STORED value at its first step
    d.dbK = buildDeviceBoundary(t.k, patches, g, /*storedIoSeed=*/true);
    d.dbEps = buildDeviceBoundary(t.epsilon, patches, g, /*storedIoSeed=*/true);
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
        const bool isWF = isTurbWallPatch(patches, pi, wfPatch);
        // each wall function's OWN coefficients: the nut patch's for nutkWallFunction, the epsilon
        // patch's for epsilonWallFunction
        const WallFunctionCoeffs& nc = t.nut.boundary[pi]->wallCoeffs();
        const WallFunctionCoeffs& ec = t.epsilon.boundary[pi]->wallCoeffs();
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
            kind.push_back(t.nutWallKind[pi] >= 0 ? t.nutWallKind[pi] : static_cast<int>(NutWall::Nutk));
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

    if (!t.variableDensity)
    {
        d.onesCell.copyFrom(std::vector<scalar>(static_cast<std::size_t>(m.nCells()), scalar(1)));
        d.onesBnd.copyFrom(std::vector<scalar>(static_cast<std::size_t>(bndIdx), scalar(1)));
    }
    return d;
}


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
    }

    gpu::kEpsilonRAS::KEpsilonInput kin;
    // divU and the flux-conditional patches read the volumetric phi in BOTH lineages
    kin.phiByRhoInt = in.phiInt;
    kin.phiByRhoBnd = in.phiBnd;
    kin.bcPhiBnd = in.phiBnd;
    if (t.variableDensity)
    {
        kin.phiInt = in.rhoPhiInt;
        kin.phiBnd = in.rhoPhiBnd;
        kin.rhoCell = in.rho;
        kin.rhoBndFace = in.rhoBnd;
        kin.rhoOldCell = in.rhoOld;
    }
    else
    {
        kin.phiInt = in.phiInt;
        kin.phiBnd = in.phiBnd;
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
    kin.relaxEquationEps = t.epsRelaxFinal.on;
    kin.relaxEps = t.epsRelaxFinal.factor;
    kin.relaxEquationK = t.kRelaxFinal.on;
    kin.relaxK = t.kRelaxFinal.factor;
    const cpu::interFoam::SmoothLinearSolve& ks = t.kSolveFinal;
    const cpu::interFoam::SmoothLinearSolve& es = t.epsSolveFinal;
    if (ks.smoother != es.smoother || ks.tol != es.tol || ks.relTol != es.relTol
     || ks.maxIter != es.maxIter || ks.minIter != es.minIter || ks.nSweeps != es.nSweeps)
        throw std::runtime_error(
            "brae interFoam (device): fvSolution gives kFinal and epsilonFinal different solver "
            "settings; the closure takes one set for both equations.");
    kin.gsK = true;
    kin.gsEps = true;
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
}


void downloadDeviceInterTurbulence(
    const DeviceInterTurbulence& d,
    cpu::interFoam::InterTurbulence& t,
    const std::vector<FvPatch>& patches)
{
    if (!t.on) return;
    d.k.copyTo(t.k.internal);
    d.epsilon.copyTo(t.epsilon.internal);
    d.nut.copyTo(t.nut.internal);
    t.k.evaluateBoundary();
    t.epsilon.evaluateBoundary();
    std::vector<scalar> nb;
    d.nutBnd.copyTo(nb);
    std::size_t off = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::size_t n = static_cast<std::size_t>(patches[pi].size);
        t.nut.boundary[pi]->setValue(std::vector<scalar>(nb.begin() + off, nb.begin() + off + n));
        off += n;
    }
}

} // namespace brae
