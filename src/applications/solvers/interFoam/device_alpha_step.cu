// The device alpha step -- see device_alpha_step.cuh for the provenance and for what is NOT here.
#include "device_alpha_step.cuh"
#include "device_blas.cuh"       // deviceCopy
#include "device_alpha_flux.cuh"
#include "device_mules.cuh"
#include "inter_phase_time.cuh"
#include <cstdlib>
#include <optional>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

void ckS(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae interFoam device alpha step: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// alpha2 = 1 - alpha1, rebuilt from the CURRENT alpha1 every corrector -- the compressive term reads
// it, and a stale alpha2 would compress towards where the interface was one corrector ago.
__global__ void complementKernel(const scalar* __restrict__ in, int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = scalar(1) - in[i];
}

// zeroGradient: the patch value IS the face cell's. This is alpha2's boundary condition in the host's
// alphaEqnStep, and it is NOT 1 - alpha1's patch value -- on damBreak's atmosphere patch alpha1 is
// inletOutlet, so the two differ wherever that patch is taking its inlet branch.
__global__ void zeroGradientKernel(const label* __restrict__ bndCell, const scalar* __restrict__ vol,
                                   int nB, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nB) out[i] = vol[bndCell[i]];
}

// pos0(phi) -- upwind's face weight. 1 takes the owner, 0 the neighbour, and phi == 0 takes the owner
// (pos0, not pos), which is the tie-break every OpenFOAM upwind scheme makes.
__global__ void upwindWeightKernel(const scalar* __restrict__ phi, int n, scalar* __restrict__ w)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = (phi[i] >= scalar(0)) ? scalar(1) : scalar(0);
}

// alphaPhi10 += w*corr (alphaEqn.H:203). w is 1 on the first corrector and 0.5 after.
__global__ void axpyKernel(scalar w, const scalar* __restrict__ corr, int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += w*corr[i];
}

// alpha1 = 0.5*alpha1 + 0.5*alpha10 (alphaEqn.H:197-201). BOTH halves of the state are relaxed -- the
// field here and the flux in axpyKernel above. Relaxing one and not the other leaves alpha and
// alphaPhi10 describing different states, and rhoPhi is built from the flux while UEqn is built on the
// field.
__global__ void relaxKernel(const scalar* __restrict__ alpha10, int nC, scalar* __restrict__ alpha1)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) alpha1[c] = scalar(0.5)*alpha1[c] + scalar(0.5)*alpha10[c];
}

__global__ void addKernel(const scalar* __restrict__ a, const scalar* __restrict__ b,
                          int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

void copyFaces(int n, const DeviceBuffer<scalar>& in, DeviceBuffer<scalar>& out)
{
    if (n <= 0) return;
    out.resize(static_cast<std::size_t>(n));
    ckS(cudaMemcpy(out.data(), in.data(), sizeof(scalar)*n, cudaMemcpyDeviceToDevice), "copy faces");
}

void addFaces(int n, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b,
              DeviceBuffer<scalar>& out)
{
    if (n <= 0) return;
    out.resize(static_cast<std::size_t>(n));
    addKernel<<<nBlocks(n), TPB>>>(a.data(), b.data(), n, out.data());
    ckS(cudaGetLastError(), "add faces");
}

// The internal-face weights for one scheme, computed from THAT flux. vanLeer takes the cell gradient
// of the field through the case's gradSchemes -- 43 of the 44 shipped interFoam tutorials say
// `default Gauss linear`, and this takes that, as the host's fluxWithScheme does.
void schemeWeights(const DeviceMesh&           dm,
                   DeviceAlphaScheme           scheme,
                   const DeviceBuffer<scalar>& psiInt,
                   const DeviceBuffer<scalar>& field,
                   const DeviceBuffer<scalar>& fieldBnd,
                   DeviceBuffer<scalar>&       w,
                   const DeviceCyclic*         cyc,
                   bool                        gradLeastSquares,
                   scalar                      gradCellLimitK)
{
    const int nIf = dm.nInternalFaces;
    switch (scheme)
    {
        case DeviceAlphaScheme::linear:
        {
            // the mesh's own central weights, copied rather than aliased so the caller owns one buffer
            // whatever the scheme and cannot free the mesh's out from under it. ON THE DEVICE: they went down
            // to a fresh host vector and up again, twice a corrector for `div(phirb,alpha) Gauss linear`
            // (31 of the 42 tutorials name it) -- six round trips of the internal faces a step on a case with
            // three sub-cycles (laminar/waves/streamFunction, 317,920 faces, 2026-10-05).
            deviceCopyNotViaHost(w, dm.w, "the linear scheme's weights");
            break;
        }
        case DeviceAlphaScheme::upwind:
        {
            w.resize(static_cast<std::size_t>(nIf));
            upwindWeightKernel<<<nBlocks(nIf), TPB>>>(psiInt.data(), nIf, w.data());
            ckS(cudaGetLastError(), "upwind weights");
            break;
        }
        case DeviceAlphaScheme::interfaceCompression:
        {
            // a PhiScheme: the limiter is the two cells' own values, so no gradient is built here and
            // a periodic pair needs none either (the pair's own faces are weighted the same way, in
            // deviceAlphaCyclicFluxWith).
            deviceInterfaceCompressionWeights(dm, psiInt, field, w);
            break;
        }
        case DeviceAlphaScheme::vanLeer:
        default:
        {
            DeviceBuffer<scalar> gx, gy, gz;
            deviceGradOf(dm, field, fieldBnd, gradLeastSquares, gradCellLimitK, gx, gy, gz,
                         (gradLeastSquares && cyc && cyc->n > 0) ? cyc : nullptr);
            // ...WITH THE PAIR IN IT. These are the INTERNAL faces' weights, but the limiter reads the
            // cell gradient, and a cell on a periodic boundary has one of its faces in the pair. A
            // gradient built without it is wrong in every such cell, so the limiter on that cell's
            // ordinary faces is wrong too. MEASURED on validation/interFoamCyclic: alpha 7.8e-02 from
            // the host at step two, all of it in the pair's own cells, with every gradient the pair's
            // OWN flux uses already carrying it. A leastSquares fit took the pair inside it above, so
            // adding the pair's Gauss faces again would count them twice.
            if (!gradLeastSquares && cyc && cyc->n > 0)
            {
                deviceCyclicAddGrad(*cyc, field, dm.V, gx, gy, gz);
            }
            deviceLimitedFaceWeights(dm, psiInt, field, gx, gy, gz, kVanLeerTwoByk, w);
            break;
        }
    }
}

// fvc::flux(psi, vf, scheme) on both sides of the mesh. At an UNCOUPLED patch surfaceInterpolation
// returns the patch value itself -- there is no second cell to weight against -- so the boundary face
// flux is psi_b*vf_b whatever the scheme, which is why no weights appear below the internal call.
void fluxWithScheme(const DeviceMesh&           dm,
                    DeviceAlphaScheme           scheme,
                    const DeviceBuffer<scalar>& psiInt,
                    const DeviceBuffer<scalar>& psiBnd,
                    const DeviceBuffer<scalar>& field,
                    const DeviceBuffer<scalar>& fieldBnd,
                    DeviceBuffer<scalar>&       outInt,
                    DeviceBuffer<scalar>&       outBnd,
                    const DeviceCyclic*         cyc,
                    bool                        gradLeastSquares,
                    scalar                      gradCellLimitK)
{
    DeviceBuffer<scalar> w;
    schemeWeights(dm, scheme, psiInt, field, fieldBnd, w, cyc, gradLeastSquares, gradCellLimitK);
    deviceAlphaFaceFlux(dm, dm.nInternalFaces, psiInt, w, field, outInt);
    deviceMultiplyFaces(dm.nBndFaces, psiBnd, fieldBnd, outBnd);
}

}   // namespace


void deviceAlphaCorrector(
    const DeviceMesh&              dm,
    DeviceBuffer<scalar>&          alpha1,
    const DeviceBuffer<scalar>&    alpha1Old,
    const DeviceAlphaStepInput&    in,
    const DeviceAlphaBoundary&     bnd,
    const DeviceMulesControls&     mulesCtl,
    const DeviceBuffer<scalar>&    nHatfInt,
    DeviceBuffer<scalar>&          alphaPhi10Int,
    DeviceBuffer<scalar>&          alphaPhi10Bnd)
{
    if (!in.phiInt || !in.phiBnd || !in.phiCNInt || !in.phiCNBnd)
        throw std::runtime_error("brae interFoam device alphaEqn: phi and phiCN are both required.");
    if (in.cyc && in.cyc->n > 0 && (!in.phiCNIf || static_cast<label>(in.phiCNIf->size()) != in.cyc->n))
    {
        throw std::runtime_error(
            "brae interFoam device alphaEqn: the mesh has a coupled pair and the caller handed no phiCN for "
            "its faces. The bounded flux there is upwind's of phiCN, as on every face (MULESTemplates.C:596); "
            "the pair's own flux is not it under CrankNicolson.");
    }
    if (!bnd.alpha1 || !bnd.nHatfBnd || !bnd.fixesValue || !bnd.flag)
        throw std::runtime_error(
            "brae interFoam device alphaEqn: alpha1's boundary values, nHatf's boundary values, the "
            "fixesValue mask and the patch-type flags are all required. They are evaluated on the "
            "host -- see device_alpha_step.cuh for why -- and passing null would silently run the "
            "step with a boundary of zeros.");
    // A LIMITED gradient across a periodic pair: deviceCellLimitGrad takes the pair's faces only as a
    // CellLimitInterface list, which no caller here builds, and a Gauss gradient has the pair added
    // AFTER deviceGradOf has already limited it -- so both orders are wrong in every cell on the pair.
    if (in.cyc && in.cyc->n > 0
     && (in.gradAlpha1CellLimitK > scalar(0) || in.gradAlpha2CellLimitK > scalar(0)))
        throw std::runtime_error(
            "brae interFoam device alphaEqn: fvSchemes names a cellLimited gradient for alpha1 or alpha2 "
            "on a mesh with a periodic pair. The limiter would not see the pair's faces. Refused rather "
            "than limit a gradient without them.");
    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nBf = dm.nBndFaces;
    const scalar rDeltaT = scalar(1) / in.deltaT;
    if (static_cast<int>(alpha1.size()) != nC)
        throw std::runtime_error(
            "brae interFoam device alphaEqn: alpha1 must already hold this corrector's starting field. "
            "The corrector does NOT reset it to alpha1Old -- alphaEqn.H resets alpha1 once per "
            "sub-cycle and each corrector continues from the last one's answer.");

    DeviceBuffer<scalar> phicInt, phicBnd, phirInt, phirBnd;
    DeviceBuffer<scalar> alpha2(static_cast<std::size_t>(nC)), alpha2Bnd(static_cast<std::size_t>(nBf));
    DeviceBuffer<scalar> advInt, advBnd, negPhirInt, negPhirBnd;
    DeviceBuffer<scalar> innerInt, innerBnd, negInnerInt, negInnerBnd, compInt, compBnd;
    DeviceBuffer<scalar> phiBDInt, phiBDBnd, corrInt, corrBnd, lamInt, lamBnd;
    DeviceBuffer<scalar> unInt, unBnd, alpha10;
    DeviceMulesFields mf;                       // all null: rho == 1, Sp == Su == 0, bounds [0,1]
    mf.Vsc  = in.Vsc;                           // ...except the volumes of a mesh that moves
    mf.Vsc0 = in.Vsc0;
    mf.rDeltaT = in.rDeltaT ? in.rDeltaT->data() : nullptr;   // ...and a localEuler case's local step

    std::optional<interPhase::Nested> part;
    part.emplace("alpha corrector: the compression flux and the scheme fluxes");
    // phic = cAlpha*|phi/magSf|, zeroed on every non-coupled boundary face.
    // THE COMPRESSION ON A PERIODIC PAIR. OpenFOAM zeroes phic on every UNCOUPLED patch and leaves a
    // coupled one alone (alphaEqn.H:79-89): the interface does pass through it, and phic there is
    // cAlpha*|phi_b/magSf|, what it is on an internal face (alpha_eqn_cpp.cu:154-172). icAlpha and
    // scAlpha across a coupled face are refused -- by the host arm and, below, here.
    if (in.phicIntPre && in.phicBndPre)
    {
        deviceCopy(phicInt, *in.phicIntPre);
        deviceCopy(phicBnd, *in.phicBndPre);
    }
    else
    {
        deviceCompressionFlux(dm, nIf, nBf, *in.phiInt, in.cAlpha, phicInt, phicBnd);
    }

    // phir = phic*nHatf, from the nHatf the PREVIOUS mixture.correct() left.
    deviceMultiplyFaces(nIf, phicInt, nHatfInt, phirInt);
    deviceMultiplyFaces(nBf, phicBnd, *bnd.nHatfBnd, phirBnd);
    // ...and on the pair, whose nHatf the same mixture.correct() produced (gated against OpenFOAM's
    // own written field in tests/interfoam_cyclic_nhatf_vs_openfoam.sh)
    DeviceBuffer<scalar> phicIf, phirIf;
    if (in.cyc && in.cyc->n > 0)
    {
        if (!in.nHatfIf || static_cast<int>(in.nHatfIf->size()) != in.cyc->n)
        {
            throw std::runtime_error(
                "brae interFoam device alphaEqn: the mesh has a periodic pair and no nHatf was given "
                "there. phir is phic*nHatf, and a coupled face is the one kind alphaEqn.H:79-89 leaves "
                "compressed.");
        }
        if (in.phicIfPre)
        {
            deviceCopy(phicIf, *in.phicIfPre);
        }
        else
        {
            deviceAlphaCyclicCompressionFlux(*in.cyc, in.cAlpha, phicIf);
        }
        deviceMultiplyFaces(in.cyc->n, phicIf, *in.nHatfIf, phirIf);
    }

    complementKernel<<<nBlocks(nC), TPB>>>(alpha1.data(), nC, alpha2.data());
    ckS(cudaGetLastError(), "alpha2");
    if (nBf > 0)
    {
        zeroGradientKernel<<<nBlocks(nBf), TPB>>>(dm.bndCell.data(), alpha2.data(), nBf,
                                                  alpha2Bnd.data());
        ckS(cudaGetLastError(), "alpha2 boundary");
    }

    // alphaPhiUn, alphaEqn.H:164-176. Term 1 is plain advection; term 2 is TWO MINUS SIGNS deep,
    //     fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme), alpha1, alpharScheme)
    // and each negation changes which cell the interpolation reads, not just the result's sign.
    fluxWithScheme(dm, in.alphaScheme, *in.phiInt, *in.phiBnd, alpha1, *bnd.alpha1, advInt, advBnd,
                   in.cyc, in.gradAlpha1LeastSquares, in.gradAlpha1CellLimitK);
    // ...and on the pair, where the scheme's own weight applies as on an internal face. The limiter
    // reads fvc::grad(alpha), which must carry the pair itself -- see device_alpha_flux.cuh.
    DeviceBuffer<scalar> advIf;
    if (in.cyc && in.cyc->n > 0)
    {
        DeviceBuffer<scalar> gx, gy, gz;
        // the pair's own limiter gradient, under the SAME entry as the internal one above: a leastSquares
        // fit takes the pair inside it, a Gauss one has the pair's faces added after (deviceCyclicAddGrad)
        deviceGradOf(dm, alpha1, *bnd.alpha1, in.gradAlpha1LeastSquares, in.gradAlpha1CellLimitK,
                     gx, gy, gz, in.gradAlpha1LeastSquares ? in.cyc : nullptr);
        if (!in.gradAlpha1LeastSquares) deviceCyclicAddGrad(*in.cyc, alpha1, dm.V, gx, gy, gz);
        deviceAlphaCyclicFlux(*in.cyc, static_cast<int>(in.alphaScheme), alpha1, gx, gy, gz, advIf);
    }
    deviceNegateFaces(nIf, phirInt, negPhirInt);
    deviceNegateFaces(nBf, phirBnd, negPhirBnd);
    fluxWithScheme(dm, in.alpharScheme, negPhirInt, negPhirBnd, alpha2, alpha2Bnd,
                   innerInt, innerBnd, in.cyc, in.gradAlpha2LeastSquares, in.gradAlpha2CellLimitK);
    deviceNegateFaces(nIf, innerInt, negInnerInt);              // the OUTER minus
    deviceNegateFaces(nBf, innerBnd, negInnerBnd);
    fluxWithScheme(dm, in.alpharScheme, negInnerInt, negInnerBnd, alpha1, *bnd.alpha1,
                   compInt, compBnd, in.cyc, in.gradAlpha1LeastSquares, in.gradAlpha1CellLimitK);
    addFaces(nIf, advInt, compInt, unInt);
    addFaces(nBf, advBnd, compBnd, unBnd);
    // ...and the pair's own compressive half, the same two nested negations: each changes which cell
    // the interpolation reads, not just the result's sign.
    DeviceBuffer<scalar> unIf;
    if (in.cyc && in.cyc->n > 0)
    {
        DeviceBuffer<scalar> a2gx, a2gy, a2gz, negPhirIf, innerIf, negInnerIf, compIf;
        deviceGradOf(dm, alpha2, alpha2Bnd, in.gradAlpha2LeastSquares, in.gradAlpha2CellLimitK,
                     a2gx, a2gy, a2gz, in.gradAlpha2LeastSquares ? in.cyc : nullptr);
        if (!in.gradAlpha2LeastSquares) deviceCyclicAddGrad(*in.cyc, alpha2, dm.V, a2gx, a2gy, a2gz);
        deviceNegateFaces(in.cyc->n, phirIf, negPhirIf);
        deviceAlphaCyclicFluxWith(*in.cyc, negPhirIf, static_cast<int>(in.alpharScheme), alpha2,
                                  a2gx, a2gy, a2gz, innerIf);
        deviceNegateFaces(in.cyc->n, innerIf, negInnerIf);
        DeviceBuffer<scalar> a1gx, a1gy, a1gz;
        deviceGradOf(dm, alpha1, *bnd.alpha1, in.gradAlpha1LeastSquares, in.gradAlpha1CellLimitK,
                     a1gx, a1gy, a1gz, in.gradAlpha1LeastSquares ? in.cyc : nullptr);
        if (!in.gradAlpha1LeastSquares) deviceCyclicAddGrad(*in.cyc, alpha1, dm.V, a1gx, a1gy, a1gz);
        deviceAlphaCyclicFluxWith(*in.cyc, negInnerIf, static_cast<int>(in.alpharScheme), alpha1,
                                  a1gx, a1gy, a1gz, compIf);
        addFaces(in.cyc->n, advIf, compIf, unIf);
    }

    if (in.MULESCorr)
    {
        part.emplace("alpha corrector: the correction formed (MULESCorr)");
        // alphaEqn.H:178-205. The high-order flux is not limited as a whole here: what CMULES limits
        // is what it ADDS to the flux the implicit pre-solve already applied, because that half has
        // already advanced alpha a full time step.
        if (static_cast<int>(alphaPhi10Int.size()) != nIf
         || static_cast<int>(alphaPhi10Bnd.size()) != nBf)
            throw std::runtime_error(
                "brae interFoam device alphaEqn: on the MULESCorr path alphaPhi10 comes IN as the flux "
                "the pre-solve or the previous corrector left. An empty one would make the correction "
                "the whole high-order flux, which is the explicit path wearing CMULES' limiter.");

        deviceSubtractFaces(nIf, unInt, alphaPhi10Int, corrInt);
        deviceSubtractFaces(nBf, unBnd, alphaPhi10Bnd, corrBnd);
        // ...and the pair's own correction, against the flux the pre-solve left there
        DeviceBuffer<scalar> corrIf;
        if (in.cyc && in.cyc->n > 0)
        {
            if (!in.alphaPhiIf || static_cast<int>(in.alphaPhiIf->size()) != in.cyc->n)
            {
                throw std::runtime_error(
                    "brae interFoam device alphaEqn: on the MULESCorr path the pair's alphaPhi10 comes "
                    "IN as the flux the pre-solve left there; an empty one would make the correction "
                    "the whole high-order flux on those faces.");
            }
            deviceSubtractFaces(in.cyc->n, unIf, *in.alphaPhiIf, corrIf);
        }

        // saved BEFORE the correction, for the relaxation below
        alpha10.resize(static_cast<std::size_t>(nC));
        ckS(cudaMemcpy(alpha10.data(), alpha1.data(), sizeof(scalar)*nC, cudaMemcpyDeviceToDevice),
            "alpha10 = alpha1");

        // MULES::correctLimited: limit the correction in place, then apply it to the CURRENT alpha.
        //
        // THE FLUX THE OUTLET TEST READS IS alphaPhiUn, NOT phiCN. OpenFOAM's call is
        //     MULES::correct(geometricOneField(), alpha1, talphaPhi1Un(), talphaPhi1Corr.ref(), ...)
        // so the "total flux leaves the domain" test of device_mules.cuh (C) is on
        // alphaPhiUn + phiCorr. Passing phiCN here was this file's first wiring and it is a different
        // quantity -- phiCN is the volumetric flux, alphaPhiUn is the alpha flux, and on a boundary
        // face holding alpha they differ by a factor of alpha.
        const bool havePair = (in.cyc && in.cyc->n > 0);
        part.emplace("alpha corrector: the correction limited and applied (MULESCorr)");
        deviceMulesLimitCorr(dm, nIf, nBf, rDeltaT, alpha1, *bnd.alpha1, *bnd.fixesValue, *bnd.flag,
                             unBnd, corrInt, corrBnd, mf, mulesCtl, nullptr, nullptr,
                             havePair ? in.cyc : nullptr, havePair ? &corrIf : nullptr);
        deviceMulesCorrect(dm, rDeltaT, corrInt, corrBnd, mf, alpha1,
                           havePair ? in.cyc : nullptr, havePair ? &corrIf : nullptr);
        part.emplace("alpha corrector: the relaxation and alphaPhi10 (MULESCorr)");
        // UNDER-RELAXED FOR EVERY CORRECTOR BUT THE FIRST, both halves (alphaEqn.H:195-205).
        const scalar w = (in.aCorr == 0) ? scalar(1) : scalar(0.5);
        if (in.aCorr != 0)
        {
            if (bnd.alphaPostMules)
            {
                deviceCopy(*bnd.alphaPostMules, alpha1);
            }
            relaxKernel<<<nBlocks(nC), TPB>>>(alpha10.data(), nC, alpha1.data());
            ckS(cudaGetLastError(), "corrector relaxation");
        }
        if (nIf > 0)
        {
            axpyKernel<<<nBlocks(nIf), TPB>>>(w, corrInt.data(), nIf, alphaPhi10Int.data());
            ckS(cudaGetLastError(), "alphaPhi10 += w*corr");
        }
        if (nBf > 0)
        {
            axpyKernel<<<nBlocks(nBf), TPB>>>(w, corrBnd.data(), nBf, alphaPhi10Bnd.data());
            ckS(cudaGetLastError(), "alphaPhi10 += w*corr, boundary");
        }
        // ...and the pair, with the SAME weight: the relaxation is of the flux, not of the faces the
        // device happens to hold in one array rather than another.
        if (havePair)
        {
            axpyKernel<<<nBlocks(in.cyc->n), TPB>>>(w, corrIf.data(), in.cyc->n,
                                                    in.alphaPhiIf->data());
            ckS(cudaGetLastError(), "alphaPhi10 += w*corr, interface");
        }
        return;
    }

    // MULES::explicitSolve(geometricOneField(), alpha1, phiCN, alphaPhi10, 0, 0, 1, 0) --
    // alphaEqn.H:208-220. alphaPhi10 IS alphaPhiUn on the explicit path, limited IN PLACE, and the
    // limiter runs against phiCN rather than phi: on a Crank-Nicolson case those differ, and
    // bounding the correction against the wrong flux would bound the wrong equation.
    part.emplace("alpha corrector: the donor flux (explicit)");
    copyFaces(nIf, unInt, alphaPhi10Int);
    copyFaces(nBf, unBnd, alphaPhi10Bnd);
    deviceMulesDonorFlux(dm, nIf, nBf, *in.phiCNInt, alpha1, alphaPhi10Bnd, phiBDInt, phiBDBnd);
    if (bnd.updateModelled)
    {
        // psi.correctBoundaryConditions(), the statement that opens MULES::explicitSolve
        // (MULESTemplates.C:168). THE BOUNDED FLUX'S BOUNDARY IS NOT REBUILT ON THE NEW VALUES: limit()
        // overwrites it with the high-order flux's on every patch that is not coupled (`if
        // (!phiBDPf.coupled()) phiBDPf = phiPsiBf[patchi]`, :599-609), which is what deviceMulesDonorFlux
        // has just done -- so the correction is zero there and the new values reach the limiter through
        // the fixes-value extrema alone (:338-347). This rebuilt it as phiCN_b*alpha_b on the new values
        // until 2026-10-05, on the reading that upwind's own boundary flux stands: the face's flux came out
        // the same (lambda is 1 on an uncoupled face), but its cell's sums did not -- a part of the
        // boundary flux was counted as correction, which moves lambda in that cell where its bound is
        // active. brae's host MULES has the overwrite (mules_cpp.cu, boundedDonorFlux).
        // NO SHIPPED CASE TELLS THE TWO FORMS APART. MEASURED 2026-10-06, device arm at pinned solves:
        // laminar/waves/stokesI after 40 steps and RAS/weirOverflow after 60 write the same bytes with
        // either (0 of 7 and 0 of 10 files differ), the small static AMI fixture is 7.7e-13 of OpenFOAM
        // with the old form. It takes a patch value the evaluate moved, on a face that carries a flux, in
        // a cell whose bound is active -- which tests/test_device_alpha_opening_evaluate.cu builds: the
        // device step is the host's to 4.4e-16 with the overwrite and 2.4e-02 from it with the rebuild.
        //   BRAE_CONTROL_MULES_BOUNDARY_DONOR_NEW_VALUES=1: rebuilt on the new values, as before
        static const bool donorOnNewValues = std::getenv("BRAE_CONTROL_MULES_BOUNDARY_DONOR_NEW_VALUES") != nullptr;
        bnd.updateModelled(alpha1);
        if (donorOnNewValues)
        {
            static bool said = false;
            if (!said)
            {
                said = true;
                std::printf("  *** CONTROL MODE: explicit MULES rebuilds the bounded flux's boundary on the "
                            "re-evaluated patch values, as before 2026-10-05. Not OpenFOAM's form "
                            "(MULESTemplates.C:599-609). ***\n");
            }
            deviceMultiplyFaces(nBf, *in.phiCNBnd, *bnd.alpha1, phiBDBnd);
        }
    }
    deviceSubtractFaces(nIf, alphaPhi10Int, phiBDInt, corrInt);
    deviceSubtractFaces(nBf, alphaPhi10Bnd, phiBDBnd, corrBnd);
    // THE PAIR takes the same three steps: its donor flux is upwind's own (a coupled face is not
    // overwritten by phiPsi the way an uncoupled one is, MULESTemplates.C:605), its correction is the
    // high-order flux less that, and both go into the limiter and the blend.
    const bool havePairEx = (in.cyc && in.cyc->n > 0);
    DeviceBuffer<scalar> phiBDIf, corrIfEx, lamIf, blendIf;
    if (havePairEx)
    {
        // THE BOUNDED FLUX ON THE PAIR IS UPWIND'S OF phiCN, as the interior's above: limit() builds it as
        // upwind(mesh, phi).flux(psi) with the phi explicitSolve was handed (MULESTemplates.C:596), which
        // alphaEqn.H:210-220 makes phiCN. It was built from the pair's own flux cyc.phi. The two are one
        // under Euler and to one rounding on the first outer corrector of a step (phi.oldTime() is then phi
        // itself); from the second outer corrector on under CrankNicolson they differ by (1 - cnCoeff) of
        // the flux change since the step began. FOUND 2026-10-06 by a reviewer reading; held by
        // tests/test_device_mules_cyclic_vs_host.cu on a flux that differs from the pair's own.
        //   BRAE_CONTROL_DEVICE_PAIR_DONOR_RAW_PHI=1: a gate's CONTROL, deliberately wrong -- cyc.phi
        static const bool rawPhi = std::getenv("BRAE_CONTROL_DEVICE_PAIR_DONOR_RAW_PHI") != nullptr;
        static bool saidRaw = false;
        if (rawPhi && !saidRaw)
        {
            saidRaw = true;
            std::printf("  *** CONTROL MODE: the pair's bounded flux is built from its raw flux, not phiCN. This "
                        "run is deliberately wrong. ***\n");
        }
        deviceMulesDonorFluxCyclic(*in.cyc, rawPhi ? in.cyc->phi : *in.phiCNIf, alpha1, phiBDIf);
        deviceSubtractFaces(in.cyc->n, unIf, phiBDIf, corrIfEx);
    }
    part.emplace("alpha corrector: the limiter (explicit)");
    deviceMulesLimiter(dm, nIf, nBf, rDeltaT, alpha1, alpha1Old, *bnd.alpha1,
                       *bnd.fixesValue, *bnd.flag, phiBDInt, phiBDBnd, corrInt, corrBnd,
                       mf, mulesCtl, lamInt, lamBnd,
                       havePairEx ? in.cyc : nullptr,
                       havePairEx ? &phiBDIf : nullptr,
                       havePairEx ? &corrIfEx : nullptr,
                       havePairEx ? &lamIf : nullptr);
    part.emplace("alpha corrector: the blend and the explicit solve");
    deviceMulesBlend(nIf, nBf, phiBDInt, phiBDBnd, lamInt, lamBnd, corrInt, corrBnd,
                     alphaPhi10Int, alphaPhi10Bnd);
    if (havePairEx)
    {
        // phiPsi = phiBD + lambda*phiCorr on the pair too (MULESTemplates.C:634)
        blendIf.resize(static_cast<std::size_t>(in.cyc->n));
        ckS(cudaMemcpy(blendIf.data(), phiBDIf.data(), sizeof(scalar)*in.cyc->n,
                       cudaMemcpyDeviceToDevice), "phiPsi = phiBD, interface");
        DeviceBuffer<scalar> scaled;
        scaled.resize(static_cast<std::size_t>(in.cyc->n));
        ckS(cudaMemcpy(scaled.data(), corrIfEx.data(), sizeof(scalar)*in.cyc->n,
                       cudaMemcpyDeviceToDevice), "phiCorr copy, interface");
        deviceMultiplyFaces(in.cyc->n, scaled, lamIf, scaled);
        DeviceBuffer<scalar> sum;
        addFaces(in.cyc->n, blendIf, scaled, sum);
        in.alphaPhiIf->resize(static_cast<std::size_t>(in.cyc->n));
        ckS(cudaMemcpy(in.alphaPhiIf->data(), sum.data(), sizeof(scalar)*in.cyc->n,
                       cudaMemcpyDeviceToDevice), "alphaPhi10 = phiPsi, interface");
    }
    deviceMulesExplicitSolve(dm, rDeltaT, alpha1Old, alphaPhi10Int, alphaPhi10Bnd, mf, alpha1,
                             havePairEx ? in.cyc : nullptr,
                             havePairEx ? in.alphaPhiIf : nullptr);
}

} // namespace brae
