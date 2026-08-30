// CUDA implementation -- see rhoEEqn.cuh for the provenance and the contract. Every term is transcribed
// from rhoEEqn_cpp.cu in the order that file produces it.
#include "rhoEEqn.cuh"
#include "device_blas.cuh"
#include "device_simple.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// Ekp = 0.5|U|^2 + p/rho  (he == e), or K = 0.5|U|^2  (he == h). One kernel, one branch, because the two
// forms are the same expression with a term switched off -- and because a second kernel would be a second
// place for the branch to be wrong.
__global__
void kineticEnergyKernel(
    int                        n,
    const scalar* __restrict__ ux,
    const scalar* __restrict__ uy,
    const scalar* __restrict__ uz,
    const scalar* __restrict__ pf,
    const scalar* __restrict__ rf,
    int                        isE,
    scalar*       __restrict__ ke)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar half = scalar(0.5) * (ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i]);
    ke[i] = isE ? half + pf[i] / rf[i] : half;
}

// fvc::div(phi, vf) with UPWIND face values, returned EXTENSIVE (the flux sum, not divided by V) --
// which is what an fvMatrix consumes. Gathered per cell over its owner faces, its losort (neighbour)
// faces and its boundary faces, exactly as device_fvc.cu's divKernel does: a face-to-cell scatter would
// need atomics, and the owner-sorted addressing exists precisely to avoid them.
__global__
void explicitUpwindDivKernel(
    int                        nC,
    const label*  __restrict__ owner,
    const label*  __restrict__ nei,
    const label*  __restrict__ ownerStart,
    const label*  __restrict__ losort,
    const label*  __restrict__ losortStart,
    const scalar* __restrict__ phiInt,
    const label*  __restrict__ bndCellStart,
    const label*  __restrict__ bndPerm,
    const scalar* __restrict__ phiBnd,
    const scalar* __restrict__ vf,
    const scalar* __restrict__ vfBnd,
    scalar*       __restrict__ d)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
    {
        const scalar pf = phiInt[f];
        s += pf * ((pf >= 0.0) ? vf[owner[f]] : vf[nei[f]]);          // +owner
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        const scalar pf = phiInt[f];
        s -= pf * ((pf >= 0.0) ? vf[owner[f]] : vf[nei[f]]);          // -neighbour
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int b = bndPerm[k];
        s += phiBnd[b] * vfBnd[b];
    }
    d[c] = s;
}

// `bounded Gauss <scheme>` on an EXPLICIT convection term: d -= div(phi)*vf, extensive. It vanishes at
// convergence, which is exactly why it needs its own path rather than being inferred from agreement.
__global__
void boundedExplicitKernel(
    int                        nC,
    const label*  __restrict__ ownerStart,
    const label*  __restrict__ losort,
    const label*  __restrict__ losortStart,
    const scalar* __restrict__ phiInt,
    const label*  __restrict__ bndCellStart,
    const label*  __restrict__ bndPerm,
    const scalar* __restrict__ phiBnd,
    const scalar* __restrict__ vf,
    scalar*       __restrict__ d)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) s += phiInt[f];
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) s -= phiInt[losort[k]];
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k) s += phiBnd[bndPerm[k]];
    d[c] -= s * vf[c];
}

// fvm::laplacian on the HOST zero-initialises its source; deviceLaplacianCoeffs fills only
// diag/upper/lower. This equation ACCUMULATES into the source, so it has to start at zero -- the same
// trap rhoPEqn.cu hit, where an unzeroed source surfaced as an invalid __global__ read.
void zeroed(DeviceBuffer<scalar>& b, int n)
{
    b.resize(n);
    if (n) cudaCheck(cudaMemsetAsync(b.data(), 0, sizeof(scalar) * n, cudaStreamPerThread),
                     "rhoEEqn zero");
}

// Only `Gauss upwind` and `Gauss linearUpwind <grad>` are ported, for EITHER convection term. The check
// covers both because EEqn.H has two of them -- div(phi,he) and div(phi,Ekp|K) -- and they are separate
// entries in fvSchemes, so a case can name a ported scheme for one and an unported one for the other.
//
// It used to cover only `he`. div(phi,Ekp|K) fell through to upwind with no throw, which is the silent
// substitution this project exists to catch: a case saying `div(phi,K) Gauss limitedLinear` got upwind
// and converged to a different kinetic energy, hence a different temperature. The host reference has
// always refused both in one guard (rhoEEqn_cpp.cu:41-56, `if (!okKE || !okHe)`), so the DEVICE was the
// more permissive of the two -- the wrong direction for a twin whose oracle is that reference.
//
// LATENT AS SHIPPED, and worth stating exactly. Reaching the hole needed a case naming a PORTED scheme
// for he and an UNPORTED one for Ekp|K, because the he check ran first. No checked-in case separates
// them: validation/rhoLU is the only one naming an unported energy scheme at all and it names
// `bounded Gauss limitedLinear 1` for all four of div(phi,h), div(phi,e), div(phi,K) and div(phi,Ekp),
// so it was already refused on he -- and no registered test drives it. What was wrong was the
// PERMISSIVENESS, not an answer any fixture produced.
bool schemePorted(cpu::rhoSimple::DivScheme s)
{
    return s == cpu::rhoSimple::DivScheme::upwind
        || s == cpu::rhoSimple::DivScheme::linearUpwind;
}

void refuseUnsupported(const RhoEnergyInput& in)
{
    if (!schemePorted(in.schemeHe) || !schemePorted(in.schemeKE))
    {
        throw std::runtime_error(
            "rhoSimpleFoam EEqn(cuda): only `Gauss upwind` and `Gauss linearUpwind <grad>` are ported for "
            "the energy convection terms, and that applies to div(phi,Ekp|K) exactly as it does to "
            "div(phi,he) -- they are separate fvSchemes entries. Refusing rather than substituting upwind "
            "for the scheme the case named: a silently different convection term converges to a different "
            "kinetic energy and therefore a different temperature.");
    }
    if (in.hasMRF)
    {
        throw std::runtime_error(
            "rhoSimpleFoam EEqn(cuda): the case declares MRF, and EEqn.H adds fvc::div(MRF.phi(), p) when "
            "it is active. Refusing rather than solving an energy equation missing that term.");
    }
    if (in.hasFvOptions)
    {
        throw std::runtime_error(
            "rhoSimpleFoam EEqn(cuda): the case declares an fvOption this path does not implement"
            + (in.fvOptionUnsupported.empty() ? std::string()
                                              : std::string(" (") + in.fvOptionUnsupported + ")")
            + ". EEqn.H applies fvOptions(rho, he).");
    }
    if (in.hasCoupledPatches)
    {
        throw std::runtime_error(
            "rhoSimpleFoam EEqn(cuda): the mesh has cyclic/AMI/processor patches. buildDeviceMesh keeps "
            "those faces out of the LDU, so they would contribute nothing to the convection, the "
            "diffusion or the kinetic-energy divergence -- silently.");
    }
    if (!in.phiInt || !in.phiBnd || !in.alphaEffCell || !in.alphaEffBndFace)
    {
        throw std::runtime_error(
            "rhoSimpleFoam EEqn(cuda): phi and alphaEff are required on faces AND boundary faces. The "
            "boundary alphaEff is not the owner cell's -- on a wall with an alphat wall function they "
            "differ by the whole of alphat.");
    }
    if (!in.Ux || !in.Uy || !in.Uz || !in.pCell || !in.rhoCell
        || !in.UxBnd || !in.UyBnd || !in.UzBnd || !in.pBnd || !in.rhoBnd)
    {
        throw std::runtime_error(
            "rhoSimpleFoam EEqn(cuda): U, p and rho are required on cells and boundary faces to build the "
            "kinetic-energy term. Both are demanded even on the `h` branch, which does not read p or rho, "
            "so a caller cannot half-supply the inputs of an equation it selected.");
    }
}

} // namespace


void kineticEnergy(
    DeviceBuffer<scalar>& keCell,
    DeviceBuffer<scalar>& keBnd,
    const DeviceMesh&     dm,
    const RhoEnergyInput& in)
{
    refuseUnsupported(in);
    keCell.resize(dm.nCells);
    if (dm.nCells > 0)
    {
        kineticEnergyKernel<<<nBlocks(dm.nCells), TPB>>>(
            dm.nCells, in.Ux->data(), in.Uy->data(), in.Uz->data(),
            in.pCell->data(), in.rhoCell->data(), in.isE ? 1 : 0, keCell.data());
    }
    keBnd.resize(dm.nBndFaces);
    if (dm.nBndFaces > 0)
    {
        kineticEnergyKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
            dm.nBndFaces, in.UxBnd->data(), in.UyBnd->data(), in.UzBnd->data(),
            in.pBnd->data(), in.rhoBnd->data(), in.isE ? 1 : 0, keBnd.data());
    }
    cudaCheck(cudaGetLastError(), "rhoEEqn kineticEnergy");
}


void kineticEnergyDivergence(
    DeviceBuffer<scalar>& out,
    const DeviceMesh&     dm,
    const RhoEnergyInput& in)
{
    // This entry point APPLIES the KE scheme, so it carries the same refusal. It had none at all: a
    // caller reaching it directly -- as the gate does -- bypassed every check in the module.
    refuseUnsupported(in);
    DeviceBuffer<scalar> ke, keB;
    kineticEnergy(ke, keB, dm, in);

    out.resize(dm.nCells);
    if (dm.nCells > 0)
    {
        explicitUpwindDivKernel<<<nBlocks(dm.nCells), TPB>>>(
            dm.nCells, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(),
            dm.losort.data(), dm.losortStart.data(), in.phiInt->data(),
            dm.bndCellStart.data(), dm.bndPerm.data(), in.phiBnd->data(),
            ke.data(), keB.data(), out.data());
    }

    // linearUpwind on the KE term is a deferred face correction on top of the upwind value. It does NOT
    // vanish at convergence, unlike `bounded`.
    if (in.schemeKE == cpu::rhoSimple::DivScheme::linearUpwind)
    {
        DeviceBuffer<scalar> gx, gy, gz, corr;
        deviceGaussGrad(dm, ke, keB, gx, gy, gz);
        if (in.gradKELimitK > 0.0) deviceCellLimitGrad(dm, ke, keB, gx, gy, gz, in.gradKELimitK);
        deviceLinearUpwindCorr(dm, *in.phiInt, gx, gy, gz, corr);
        deviceAxpy(1.0, corr, out);
    }

    if (in.boundedKE && dm.nCells > 0)
    {
        boundedExplicitKernel<<<nBlocks(dm.nCells), TPB>>>(
            dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
            in.phiInt->data(), dm.bndCellStart.data(), dm.bndPerm.data(), in.phiBnd->data(),
            ke.data(), out.data());
    }
    cudaCheck(cudaGetLastError(), "rhoEEqn kineticEnergyDivergence");
}


void assembleEEqn(
    PressureMatrix&             E,
    const DeviceMesh&           dm,
    const DeviceBoundary&       dbHe,
    const DeviceBuffer<scalar>& he,
    const RhoEnergyInput&       in)
{
    refuseUnsupported(in);
    const int nC = dm.nCells;

    // ---- fvm::div(phi, he) ------------------------------------------------------------------
    // The scheme refusal for BOTH convection terms is in refuseUnsupported above, where the host
    // reference keeps it too (rhoEEqn_cpp.cu:41-56, one guard on okKE && okHe).
    deviceDivUpwindCoeffs(dm, *in.phiInt, E.diag, E.upper, E.lower);
    deviceBCDivCoeffs(dbHe, *in.phiBnd, E.iC, E.bC);
    zeroed(E.source, nC);

    // linearUpwind's deferred correction on the IMPLICIT term: the matrix stays pure upwind and the whole
    // scheme lives in the source, which is what OpenFOAM does.
    if (in.schemeHe == cpu::rhoSimple::DivScheme::linearUpwind)
    {
        DeviceBuffer<scalar> hb, gx, gy, gz, corr;
        deviceBCValue(dbHe, he, hb);
        deviceGaussGrad(dm, he, hb, gx, gy, gz);
        if (in.gradHeLimitK > 0.0) deviceCellLimitGrad(dm, he, hb, gx, gy, gz, in.gradHeLimitK);
        deviceLinearUpwindCorr(dm, *in.phiInt, gx, gy, gz, corr);
        deviceAxpy(-1.0, corr, E.source);
    }

    // `bounded` on the IMPLICIT term: diag -= V*div(phi). Vanishes at convergence.
    if (in.boundedHe)
    {
        DeviceBuffer<scalar> divPhi, t;
        deviceDiv(dm, *in.phiInt, *in.phiBnd, divPhi);
        deviceHadamard(t, divPhi, dm.V);
        deviceAxpy(-1.0, t, E.diag);
    }

    // ---- + fvc::div(phi, Ekp|K), the explicit kinetic-energy term ---------------------------
    // Already extensive, so it is subtracted from the source with no volume factor of its own.
    {
        DeviceBuffer<scalar> keDiv;
        kineticEnergyDivergence(keDiv, dm, in);
        deviceAxpy(-1.0, keDiv, E.source);
    }

    // ---- - fvm::laplacian(alphaEff, he) -----------------------------------------------------
    // BOTH halves of `corrected`. Implementing only the implicit one left the host reference's source
    // 2.14e-05 out on angledDuct while its diagonal stayed exact at 1.63e-15 -- see rhoEEqn.cuh.
    {
        DeviceBuffer<scalar> gammaf, lD, lU, lL, lIC, lBC;
        deviceInterpolate(dm, *in.alphaEffCell, gammaf);
        deviceLaplacianCoeffs(dm, gammaf, lD, lU, lL, in.correctedLaplacian);
        // The FACE variant: the boundary diffusivity is the PATCH alphaEff, not the owner cell's.
        deviceBCLaplacianCoeffsFace(dbHe, *in.alphaEffBndFace, lIC, lBC);

        deviceAxpy(-1.0, lD, E.diag);
        deviceAxpy(-1.0, lU, E.upper);
        deviceAxpy(-1.0, lL, E.lower);
        deviceAxpy(-1.0, lIC, E.iC);
        deviceAxpy(-1.0, lBC, E.bC);

        if (in.correctedLaplacian)
        {
            DeviceBuffer<scalar> hb, gx, gy, gz, lc;
            deviceBCValue(dbHe, he, hb);
            deviceGaussGrad(dm, he, hb, gx, gy, gz);
            if (in.snGradLimitCoeff > 0.0)
            {
                DeviceBuffer<scalar> ffcL;
                deviceLaplacianCorrFluxLimited(dm, gammaf, he, gx, gy, gz, in.snGradLimitCoeff, ffcL);
                deviceFaceDivSource(dm, ffcL, lc);
            }
            else
            {
                deviceLaplacianCorr(dm, gammaf, gx, gy, gz, lc);
            }
            // The laplacian enters with -1, so its explicit source does too.
            deviceAxpy(-1.0, lc, E.source);
        }
    }

    // ---- EEqn.relax() -----------------------------------------------------------------------
    // The guard is "the case NAMES a factor", not "the factor is below 1": relax(1.0) still runs the
    // dominance clamp and adds (D - D0)*psi to the source. See the sentinel note in rhoEEqn.cuh.
    if (in.relaxEquationHe && in.relaxHe > 0.0)
    {
        DeviceBuffer<scalar> relaxedDiag, delta, t;
        deviceRelaxDiag(E.view(dm), dm, E.iC, in.relaxHe, relaxedDiag, delta);
        deviceCopy(E.diag, relaxedDiag);
        deviceHadamard(t, delta, he);
        deviceAxpy(1.0, t, E.source);
    }
    E.faceFluxCorr.resize(0);
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
