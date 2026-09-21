// kEqn on the device -- see device_les_keqn.cuh for the provenance and for what is shared with the
// RAS closures rather than written again here.
#include "device_les_keqn.cuh"
#include "device_blas.cuh"
#include "pEqn.cuh"                 // PressureMatrix
#include "turbulence_transport.cuh"
#include <stdexcept>

namespace brae {
namespace gpu {
namespace LESkEqn {

namespace {

constexpr int TPB = 256;
inline int nBlk(int n) { return (n + TPB - 1) / TPB; }

// kEqn.C:46-47, nut = Ck*sqrt(k)*delta
__global__
void nutKernel(
    int nC,
    const scalar* __restrict__ k,
    const scalar* __restrict__ delta,
    scalar Ck,
    scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    nut[c] = Ck*sqrt(k[c])*delta[c];
}

// Everything kEqn adds to the transport matrix, in the host reference's order and its expressions
// (les_kEqn_cpp.cu:168-184): fvm::ddt, == G, - fvm::SuSp((2/3)*divU, k), - fvm::Sp(Ce*sqrt(k)/delta, k).
// `k` is the field as it stands BEFORE the solve, which is what OpenFOAM's Sp and SuSp read.
__global__
void reactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ kOld,
    const scalar* __restrict__ k,
    const scalar* __restrict__ G,
    const scalar* __restrict__ divU,
    const scalar* __restrict__ delta,
    scalar rDeltaT,
    scalar Ce,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar Vc = V[c];
    scalar d = diag[c];
    scalar s = source[c];
    // fvm::ddt(k), Euler
    d += rDeltaT*Vc;
    s += rDeltaT*kOld[c]*Vc;
    // == G
    s += G[c]*Vc;
    // - fvm::SuSp((2/3)*divU, k)
    const scalar sp = (scalar(2)/scalar(3))*divU[c];
    d += Vc*fmax(sp, scalar(0));
    s -= Vc*fmin(sp, scalar(0))*k[c];
    // - fvm::Sp(Ce*sqrt(k)/delta, k), with k as it stands before the solve
    d += Vc*(Ce*sqrt(k[c])/delta[c]);
    diag[c] = d;
    source[c] = s;
}

// DkEff = nut + nu, with NO sigma (kEqn.H:152-157) -- which is the one thing this equation's
// diffusivity does not share with kEpsilon's.
__global__
void dkEffKernel(
    int n,
    const scalar* __restrict__ nut,
    const scalar* __restrict__ nu,
    scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = nut[i] + nu[i];
}

}   // namespace


void correctNut(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& delta,
    const cpu::LESkEqn::Coeffs& co,
    DeviceBuffer<scalar>&       nut)
{
    const int nC = static_cast<int>(k.size());
    if (static_cast<int>(delta.size()) != nC)
    {
        throw std::runtime_error("brae LES kEqn (device): delta and k differ in length.");
    }
    nut.resize(static_cast<std::size_t>(nC));
    nutKernel<<<nBlk(nC), TPB>>>(nC, k.data(), delta.data(), co.Ck, nut.data());
    cudaCheck(cudaGetLastError(), "kEqn correctNut");
}


DeviceSolverPerf correct(
    const DeviceMesh&           dm,
    DeviceBoundary&             dbK,
    const DeviceVectorBoundary& dbU,
    DeviceBuffer<scalar>&       k,
    DeviceBuffer<scalar>&       nut,
    const Input&                in)
{
    if (!in.Ux || !in.Uy || !in.Uz || !in.phiInt || !in.phiBnd || !in.nu || !in.nuBnd
     || !in.nutBnd || !in.delta || !in.kOld)
    {
        throw std::runtime_error("brae LES kEqn (device): correct needs every input field.");
    }
    if (!(in.rDeltaT > scalar(0)))
    {
        throw std::runtime_error("brae LES kEqn (device): correct needs a positive deltaT.");
    }
    const int nC = dm.nCells;
    const int nB = dbK.n;

    // divU = fvc::div(fvc::absolute(phi, U)); a static mesh, so phi itself
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, *in.phiInt, *in.phiBnd, divU);

    // G = nut*(gradU && devTwoSymm(gradU)), with the nut the previous correctNut left
    // `Gauss linear` for grad(U): the only gradScheme the caller accepts on a kEqn case, and what
    // LES/nozzleFlow2D names. deviceGradU packs the tensor as OpenFOAM's column convention, which is
    // what deviceGByNuFromGradU reads.
    DeviceBuffer<scalar> gradU, gByNu, G;
    deviceGradU(dm, dbU, *in.Ux, *in.Uy, *in.Uz, gradU, /*ami=*/nullptr, /*cyc=*/nullptr);
    deviceGByNuFromGradU(gradU, nC, gByNu);
    deviceHadamard(G, nut, gByNu);

    // DkEff = nut + nu on the cells, interpolated linearly; its patch values are nut_b + nu_b
    DeviceBuffer<scalar> Dcell, gammaFace, gammaBnd;
    Dcell.resize(static_cast<std::size_t>(nC));
    dkEffKernel<<<nBlk(nC), TPB>>>(nC, nut.data(), in.nu->data(), Dcell.data());
    cudaCheck(cudaGetLastError(), "kEqn DkEff cells");
    deviceInterpolate(dm, Dcell, gammaFace);
    gammaBnd.resize(static_cast<std::size_t>(nB));
    if (nB > 0)
    {
        dkEffKernel<<<nBlk(nB), TPB>>>(nB, in.nutBnd->data(), in.nuBnd->data(), gammaBnd.data());
        cudaCheck(cudaGetLastError(), "kEqn DkEff boundary");
    }

    // k's patches learn the flux (inletOutlet) where the fvMatrix constructor runs updateCoeffs
    deviceUpdateInletOutlet(dbK, *in.phiBnd);

    // fvm::div(phi, k) - fvm::laplacian(DkEff, k), through the call kEpsilon's k equation makes
    PressureMatrix M;
    turbulence::TransportScheme sc;
    sc.phiInt             = in.phiInt;
    sc.phiBnd             = in.phiBnd;
    sc.limitedLinear      = in.co.limitedLinear;
    sc.limiterCoeff       = in.co.limitedLinearCoeff;
    sc.correctedLaplacian = in.co.correctedLaplacian;
    sc.snGradLimitCoeff   = in.co.snGradLimitCoeff;
    turbulence::assembleScalarTransport(M, dm, dbK, k, gammaFace, gammaBnd, sc);

    reactionKernel<<<nBlk(nC), TPB>>>(nC, dm.V.data(), in.kOld->data(), k.data(), G.data(),
                                      divU.data(), in.delta->data(), in.rDeltaT, in.co.Ce,
                                      M.diag.data(), M.source.data());
    cudaCheck(cudaGetLastError(), "kEqn reaction");

    turbulence::SolveControls sv;
    sv.tol = in.tol;
    sv.relTol = in.relTol;
    sv.maxIter = in.maxIter;
    sv.minIter = in.minIter;
    sv.nSweeps = in.nSweeps;
    sv.gsSymmetric = in.symmetric;
    DeviceSolverPerf perf;
    scalar residual = 0;
    turbulence::solveScalarEqn(M, k, dm, in.relaxOn, in.relax, nullptr, nullptr, nullptr, nullptr,
                               sv, residual, std::string(), /*gs=*/true, &perf);

    // bound(k, kMin), then correctNut from the bounded k -- kEqn.C:186-188 in that order
    deviceBoundField(dm, k, in.co.kMin, "k", &dbK);
    correctNut(k, *in.delta, in.co, nut);
    return perf;
}

} // namespace LESkEqn
} // namespace gpu
} // namespace brae
