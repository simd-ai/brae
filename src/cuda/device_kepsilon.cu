// cf GPU offload: k-epsilon production + eddy viscosity. gradU is the OF-convention tensor (column i =
// gaussGrad(U_i), as in divDevReff); GbyNu = sum_ij g_ij*(g_ij + g_ji - (2/3)tr d_ij).
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"
#include "device_scalar_transport.cuh"  // generic scalar-transport scaffold (deviceSolveScalarTransport)
#include "spalart_coeffs.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include "device_ami.cuh"        // cyclicAMI scalar-transport interface coupling
#include "device_cyclic.cuh"     // cyclic scalar-transport interface coupling
#include "device_interface.cuh"  // interface<Op>() overloads dispatching to the cyclic/AMI backends
#include "device_amg.cuh"        // deviceSymGaussSeidel (scalar smoothSolver, for stiff low-Re k/omega)
#include "nut_wall_function.cuh" // nutkWallFunctionValue / yPlusWall (shared wall-nut physics, BRAE_HD)
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace brae {

// OF-style turbulence residual report (see device_kepsilon.cuh). Single-threaded per solve; the SIMPLE driver
// clears it before turbulence->correct() and reads it after, to print the "Solving for k/omega/..." lines.
void clearTurbulenceReport() { turbStore().clear(); }
const std::vector<ScalarSolveEntry>& turbulenceReport() { return turbStore(); }

namespace {
inline scalar yPlusLamHost(scalar kappa, scalar E) { scalar y = 11.0; for (int i = 0; i < 10; ++i) y = std::log(std::fmax(E * y, 1.0)) / kappa; return y; }


__device__
void gByNuKernel(int nC, const scalar* __restrict__ gradU, scalar* __restrict__ gByNu)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];

    const scalar t23 = (2.0 / 3.0) * (t[0] + t[4] + t[8]);
    scalar gg = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar dts = t[i*3+j] + t[j*3+i] - ((i == j) ? t23 : 0.0);   // devTwoSymm
            gg += t[i*3+j] * dts;                                              // doubleDot
        }
    gByNu[c] = gg;
}


__device__
void nutKernel(int nC, const scalar* __restrict__ k, const scalar* __restrict__ eps, scalar Cmu, scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) nut[c] = Cmu * k[c] * k[c] / eps[c];
}


// realizableKE rCmu (variable Cmu) + magS from the gradU tensor. S=devSymm(gradU); S2=2*magSqr(S); magS=sqrt(S2);
// W=2sqrt2*tr(S^3)/(magS*S2); phis=acos(clamp(sqrt6*W,[-1,1]))/3; As=sqrt6*cos(phis); Us=sqrt(S2/2+magSqr(skew));
// rCmu=1/(A0 + As*Us*k/eps).  (OF realizableKE::rCmu, gradU_ij = t[i*3+j] = dU_j/dx_i.)
__device__
void rkeStrainKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ k,
    const scalar* __restrict__ eps,
    scalar A0,
    scalar* __restrict__ rCmu,
    scalar* __restrict__ magS)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];

    const scalar third = (t[0] + t[4] + t[8]) / 3.0;
    scalar S[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S[i*3+j] = 0.5*(t[i*3+j] + t[j*3+i]) - ((i==j) ? third : 0.0);

    scalar magSqrS = 0.0, skSq = 0.0, tr3 = 0.0;
    for (int q = 0; q < 9; ++q)
        magSqrS += S[q]*S[q];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar sk = 0.5*(t[i*3+j]-t[j*3+i]);
            skSq += sk*sk;
        }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int m = 0; m < 3; ++m)
                tr3 += S[i*3+j]*S[j*3+m]*S[m*3+i];

    const scalar S2 = 2.0*magSqrS, mS = sqrt(S2);
    scalar arg = sqrt(6.0) * (2.0*sqrt(2.0)*tr3 / (mS*S2 + 1e-37));
    arg = fmin(fmax(arg, -1.0), 1.0);
    const scalar As = sqrt(6.0)*cos((1.0/3.0)*acos(arg));
    const scalar Us = sqrt(0.5*S2 + skSq);
    rCmu[c] = 1.0/(A0 + As*Us*k[c]/eps[c]);
    magS[c] = mS;
}


__device__
void rkeNutKernel(
    int nC,
    const scalar* __restrict__ rCmu,
    const scalar* __restrict__ k,
    const scalar* __restrict__ eps,
    scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) nut[c] = rCmu[c] * k[c] * k[c] / eps[c];
}


// realizableKE eps reaction: production C1*magS*eps (explicit), destruction fvm::Sp(C2*eps/(k+sqrt(nu*eps)),eps).
// C1 = max(eta/(5+eta), 0.43), eta = magS*k/eps. No divU term (unlike standard kEpsilon).
__device__
void rkeEpsReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ eps,
    const scalar* __restrict__ k,
    const scalar* __restrict__ magS,
    scalar nu,
    scalar C2,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar e = eps[c], kc = k[c], mS = magS[c];
    const scalar eta = mS * kc / e;
    const scalar C1 = fmax(eta/(5.0 + eta), 0.43);
    diag[c]   += V[c] * (C2 * e / (kc + sqrt(fmax(nu*e, 0.0))));   // destruction Sp
    source[c] += V[c] * (C1 * mS * e);                            // production
}


__device__
void wallFnKernel(
    int nWF,
    const label* __restrict__ wfCell,
    const scalar* __restrict__ wfY,
    const scalar* __restrict__ wfDc,
    const scalar* __restrict__ wux,
    const scalar* __restrict__ wuy,
    const scalar* __restrict__ wuz,
    const scalar* __restrict__ invNw,
    const scalar* __restrict__ k,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar nu,
    scalar yplLam,
    scalar Cmu25,
    scalar Cmu75,
    scalar kappa,
    scalar E,
    scalar atmZ0,
    bool   atmBoundNut,
    int nutWall,
    scalar* __restrict__ eps0,
    scalar* __restrict__ G0,
    const scalar* __restrict__ nuFace)   // compressible: per-wall-face nu, null -> the scalar nu
{
    const int wf = blockIdx.x * blockDim.x + threadIdx.x;
    if (wf >= nWF) return;

    const int c = wfCell[wf];
    const scalar y = wfY[wf], dc = wfDc[wf], kc = k[c];
    // OF epsilonWallFunction reads turbulenceModel::nu(patchi) = mu_b/rho_b, a per-FACE field. The scalar
    // fallback is only right for constant-property incompressible flow.
    const scalar nuw = nuFace ? nuFace[wf] : nu;
    wallProductionG0(c, wf, y, dc, kc, invNw[c], wux, wuy, wuz, Ux, Uy, Uz, nuw, yplLam, Cmu25, kappa, E, atmZ0, atmBoundNut, nutWall, G0);
    atomicAdd(&eps0[c], invNw[c] * Cmu75 * pow(kc, 1.5) / (kappa * y));   // kEpsilon: distinct eps wall value
}


// boundary nut per face: wall -> nutkWallFunction(k[cell], y, nu); else -> nut[cell] (calculated/extrapolated).
__device__
void boundaryNutKernel(
    int n,
    const label* __restrict__ fc,
    const label* __restrict__ isWall,
    const scalar* __restrict__ y,
    const scalar* __restrict__ k,
    const scalar* __restrict__ nut,
    scalar nu,
    scalar yplLam,
    scalar Cmu25,
    scalar kappa,
    scalar E,
    scalar atmZ0,        // >0 -> atmNutkWallFunction (rough); 0 -> nutkWallFunction (smooth)
    bool   atmBoundNut,
    scalar* __restrict__ nutBnd,
    const scalar* __restrict__ nuFace,   // compressible: per-face nu = mu_b/rho_b, null -> the scalar nu
    const label*  __restrict__ calcMask, // 1 where the nut BC is 'calculated' -> evaluate, not extrapolate
    const scalar* __restrict__ kBnd,     // boundary k and epsilon for that evaluation
    const scalar* __restrict__ epsBnd,
    scalar Cmu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    const scalar nuw = nuFace ? nuFace[i] : nu;
    if (isWall[i])
    {
        nutBnd[i] = kBasedWallNut(yPlusWall(Cmu25, y[i], k[c], nuw), y[i], atmZ0, atmBoundNut, nuw, yplLam, kappa, E);
    }
    else if (calcMask && calcMask[i] && kBnd && epsBnd)
    {
        // OF sets nut by FIELD ASSIGNMENT (nut_ = Cmu*sqr(k_)/epsilon_), which fills the boundary from the
        // BOUNDARY k and epsilon; correctBoundaryConditions() then leaves a 'calculated' patch alone. So a
        // calculated patch carries Cmu*k_b^2/eps_b, NOT the adjacent cell value. At a fixed-k/eps inlet the
        // two differ by more than 12x, and extrapolating there changes nuEff on the inlet faces -- which
        // moves the momentum and pressure and shows up as a few 1e-3 in the converged k/epsilon.
        // A zeroGradient nut patch DOES take the cell value, which is why this is masked per patch.
        const scalar eb = epsBnd[i];
        nutBnd[i] = (eb > scalar(0)) ? Cmu * kBnd[i] * kBnd[i] / eb : nut[c];
    }
    else
    {
        nutBnd[i] = nut[c];
    }
}


__device__
void epsReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ eps,
    const scalar* __restrict__ k,
    const scalar* __restrict__ gByNu,
    const scalar* __restrict__ divU,
    scalar C1,
    scalar C2,
    scalar C3,
    scalar Cmu,
    int    rng,           // RNGkEpsilon: production coefficient (C1 - R) instead of C1
    scalar eta0,
    scalar beta,
    scalar* __restrict__ diag,
    scalar* __restrict__ source,
    const scalar* __restrict__ rho)   // compressible: OF weights EVERY RHS term by alpha*rho
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // OF kEpsilon.C epsilon equation:
    //   == C1*alpha*rho*GbyNu*Cmu*k - fvm::SuSp(((2/3)C1 - C3)*alpha*rho*divU, eps)
    //      - fvm::Sp(C2*alpha*rho*eps/k, eps)
    const scalar rw = rho ? rho[c] : scalar(1);
    const scalar sp = ((2.0/3.0) * C1 - C3) * divU[c];   // OF SuSp(((2/3)C1 - C3)*divU, eps)
    // RNGkEpsilon.C: the G production alone carries (C1 - R). gByNu IS OF's S2 (= dev(twoSymm(gradU)) &&
    // gradU, the same contraction G/nut is built from), so eta needs nothing the standard path does not
    // already compute. The SuSp term above keeps the plain C1, exactly as OF writes it.
    scalar C1p = C1;
    if (rng)
    {
        const scalar eta = sqrt(fabs(gByNu[c])) * k[c] / eps[c];
        const scalar R   = (eta * (scalar(1) - eta / eta0)) / (beta * eta * eta * eta + scalar(1));
        C1p = C1 - R;
    }
    diag[c]   += rw * V[c] * (C2 * eps[c] / k[c] + fmax(sp, 0.0));
    source[c] += rw * (V[c] * (C1p * Cmu * k[c] * gByNu[c]) - V[c] * fmin(sp, 0.0) * eps[c]);
}


__device__
void kReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ k,
    const scalar* __restrict__ eps,
    const scalar* __restrict__ G,
    const scalar* __restrict__ divU,
    scalar* __restrict__ diag,
    scalar* __restrict__ source,
    const scalar* __restrict__ rho)   // compressible: alpha*rho on every RHS term
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // OF kEpsilon.C k equation:
    //   == alpha*rho*G - fvm::SuSp((2/3)*alpha*rho*divU, k) - fvm::Sp(alpha*rho*eps/k, k)
    const scalar rw = rho ? rho[c] : scalar(1);
    const scalar sp = (2.0/3.0) * divU[c];
    diag[c]   += rw * V[c] * (eps[c] / k[c] + fmax(sp, 0.0));
    source[c] += rw * (V[c] * G[c] - V[c] * fmin(sp, 0.0) * k[c]);
}




// OF Foam::bound(): negative cells -> fvc::average(max(field,floor)) (local face-neighbour avg), positive cells ->
// max(field,floor). Prevents nut=Cmu k^2/eps blowing up when limitedLinear overshoots eps<0 (vs clamping to floor).
__device__
void boundClampKernel(int nC, scalar floor, const scalar* __restrict__ x, scalar* __restrict__ cl)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) cl[c] = fmax(x[c], floor);
}


__device__
void boundAvgGatherKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const scalar* __restrict__ faceInterp,
    const scalar* __restrict__ cl,
    scalar* __restrict__ avg)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar s = 0.0;
    int nf = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) { s += faceInterp[f]; ++nf; }             // c is owner
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) { s += faceInterp[losort[k]]; ++nf; }    // c is neighbour
    const int nb = bndCellStart[c + 1] - bndCellStart[c];                                              // boundary faces (zeroGradient = cell value)
    s += cl[c] * nb;
    nf += nb;
    avg[c] = (nf > 0) ? s / nf : cl[c];
}


__device__
void boundApplyKernel(int nC, scalar floor, const scalar* __restrict__ avg, scalar* __restrict__ x)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    x[c] = (x[c] <= 0.0) ? avg[c] : fmax(x[c], floor);   // OF: max(max(vsf, avg*pos0(-vsf)), floor)
}


} // namespace


// Build the OF-convention gradU tensor (9*nC, column i = gaussGrad(U_i)). Shared by GbyNu (k-eps) and S2 (SST).
void deviceGradU(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& gradU,
    DeviceAMI* ami,
    DeviceCyclic* cyc)
{
    const int nC = dm.nCells;
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };
    gradU.resize(static_cast<std::size_t>(9) * nC);

    // interface (cyclic/cyclicAMI) face contribution to grad(U): without it the gradient at interface cells is
    // ONE-SIDED -> wrong turbulence production G=nut*(gradU&&devTwoSymm(gradU)) there (the divDevReff x-invariance
    // bug, in the production term). For a ROTATIONAL interface the neighbour vector rotates (forwardT).
    DeviceBuffer<scalar> amiURot[3];
    if (ami && ami->n && ami->rotational) deviceAmiInterpolateVec(*ami, Ux, Uy, Uz, amiURot[0], amiURot[1], amiURot[2]);
    for (int i = 0; i < 3; ++i)
    {
        DeviceBuffer<scalar> bval;
        deviceBCValue(dbU.comp[i], *Uc[i], bval);
        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, *Uc[i], bval, gx, gy, gz);
        if (ami && ami->n) { if (ami->rotational) deviceAmiAddGradRot(*ami, *Uc[i], amiURot[i], dm.V, gx, gy, gz);
                             else interfaceAddGrad(*ami, *Uc[i], dm.V, gx, gy, gz); }
        if (cyc && cyc->n) { if (cyc->rotational) deviceCyclicAddGradRot(*cyc, Ux, Uy, Uz, i, dm.V, gx, gy, gz);
                             else interfaceAddGrad(*cyc, *Uc[i], dm.V, gx, gy, gz); }
        // async D2D on the per-thread stream (ordered before the consumer kernel): a plain cudaMemcpy here would
        // drain the GPU pipeline every turbulence iteration.
        cudaCheck(cudaMemcpyAsync(gradU.data() + (0*3+i)*nC, gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "gradU g");
        cudaCheck(cudaMemcpyAsync(gradU.data() + (1*3+i)*nC, gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "gradU g");
        cudaCheck(cudaMemcpyAsync(gradU.data() + (2*3+i)*nC, gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "gradU g");
    }
}


void deviceGByNuFromGradU(const DeviceBuffer<scalar>& gradU, int nC, DeviceBuffer<scalar>& gByNu)
{
    gByNu.resize(nC);
    const scalar* gradUd = gradU.data(); scalar* gByNud = gByNu.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { gByNuKernel(nC, gradUd, gByNud); });
    cudaCheck(cudaGetLastError(), "gByNu");
}


void deviceGbyNu(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& gByNu,
    DeviceAMI* ami,
    DeviceCyclic* cyc)
{
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);
    deviceGByNuFromGradU(gradU, dm.nCells, gByNu);
}


void deviceNut(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& eps,
    DeviceBuffer<scalar>& nut,
    const KEpsilonCoeffs& co)
{
    const int nC = static_cast<int>(k.size());
    nut.resize(nC);
    const scalar* kd = k.data(); const scalar* epsd = eps.data(); const scalar Cmu = co.Cmu; scalar* nutd = nut.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { nutKernel(nC, kd, epsd, Cmu, nutd); });
    cudaCheck(cudaGetLastError(), "nut");
}


// realizableKE: rCmu + magS from gradU, nut = rCmu*k^2/eps, eps reaction (strain production + k+sqrt(nu*eps) destruction).
void deviceRealizableStrain(
    const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& eps,
    scalar A0,
    int nC,
    DeviceBuffer<scalar>& rCmu,
    DeviceBuffer<scalar>& magS)
{
    rCmu.resize(nC);
    magS.resize(nC);
    const scalar *gradUd = gradU.data(), *kd = k.data(), *epsd = eps.data();
    scalar *rCmud = rCmu.data(), *magSd = magS.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { rkeStrainKernel(nC, gradUd, kd, epsd, A0, rCmud, magSd); });
    cudaCheck(cudaGetLastError(), "rkeStrain");
}


void deviceRealizableNut(
    const DeviceBuffer<scalar>& rCmu,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& eps,
    DeviceBuffer<scalar>& nut)
{
    const int nC = static_cast<int>(k.size());
    nut.resize(nC);
    const scalar *rCmud = rCmu.data(), *kd = k.data(), *epsd = eps.data(); scalar* nutd = nut.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { rkeNutKernel(nC, rCmud, kd, epsd, nutd); });
    cudaCheck(cudaGetLastError(), "rkeNut");
}


void deviceEpsReactionRealizable(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& eps,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& magS,
    scalar nu,
    scalar C2,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source)
{
    {
        const int nC = dm.nCells;
        const scalar *Vd = dm.V.data(), *epsd = eps.data(), *kd = k.data(), *magSd = magS.data();
        scalar *diagd = diag.data(), *sourced = source.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            rkeEpsReactionKernel(nC, Vd, epsd, kd, magSd, nu, C2, diagd, sourced);
        });
    }
    cudaCheck(cudaGetLastError(), "rkeEpsReaction");
}


// OF Foam::bound(field, floor): negative cells -> fvc::average(max(field,floor)); positive -> max(field,floor).
void deviceBoundField(const DeviceMesh& dm, DeviceBuffer<scalar>& x, scalar floor)
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> cl(static_cast<std::size_t>(nC));
    scalar* xd = x.data(); scalar* cld = cl.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { boundClampKernel(nC, floor, xd, cld); });
    DeviceBuffer<scalar> fi;
    deviceInterpolate(dm, cl, fi);   // linearInterpolate(max(field,floor))
    DeviceBuffer<scalar> avg(static_cast<std::size_t>(nC));
    {
        const label *ownerStart = dm.ownerStart.data(), *losort = dm.losort.data(), *losortStart = dm.losortStart.data();
        const label* bndCellStart = dm.bndCellStart.data(); const scalar* fid = fi.data(); scalar* avgd = avg.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            boundAvgGatherKernel(nC, ownerStart, losort, losortStart, bndCellStart, fid, cld, avgd);
        });
    }
    {
        const scalar* avgd = avg.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { boundApplyKernel(nC, floor, avgd, xd); });
    }
    cudaCheck(cudaGetLastError(), "boundField");
}


void deviceWallEpsG0(
    const DeviceWallData& w,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    scalar nu,
    DeviceBuffer<scalar>& eps0,
    DeviceBuffer<scalar>& G0,
    const KEpsilonCoeffs& co,
    int nutWall,
    scalar atmZ0,
    bool   atmBoundNut,
    const DeviceBuffer<scalar>* nuFace,
    const DeviceBuffer<scalar>* nutFile)   // compressible: nu = mu_b/rho_b per WALL face (OF nu(patchi))
{
    const int nC = static_cast<int>(k.size());
    eps0.resize(nC);
    G0.resize(nC);   // zero on-device (memsetAsync) instead of a host-vector alloc + H2D every iter
    cudaCheck(cudaMemsetAsync(eps0.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "eps0 zero");
    cudaCheck(cudaMemsetAsync(G0.data(),   0, nC*sizeof(scalar), cudaStreamPerThread), "G0 zero");
    const scalar Cmu25 = std::pow(co.Cmu, 0.25), Cmu75 = std::pow(co.Cmu, 0.75), yplLam = yPlusLamHost(co.kappa, co.E);
    if (w.nWF > 0)
    {
        const int nWF = w.nWF;
        const label* wfCell = w.wfCell.data();
        const scalar *wfY = w.wfY.data(), *wfDc = w.wfDc.data();
        const scalar *wux = w.wfUwx.data(), *wuy = w.wfUwy.data(), *wuz = w.wfUwz.data(), *invNw = w.invNw.data();
        const scalar *kd = k.data(), *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data();
        const scalar kappa = co.kappa, E = co.E; scalar *eps0d = eps0.data(), *G0d = G0.data();
        const scalar* nuFaced = (nuFace && nuFace->size()) ? nuFace->data() : nullptr;
        pcudaParallelFor(nBlocks(nWF), TPB, [=] __device__ () {
            wallFnKernel(nWF, wfCell, wfY, wfDc, wux, wuy, wuz, invNw, kd, Uxd, Uyd, Uzd, nu, yplLam, Cmu25, Cmu75,
                        kappa, E, atmZ0, atmBoundNut, nutWall, eps0d, G0d, nuFaced);
        });
    }
    cudaCheck(cudaGetLastError(), "wallFn");
}


void deviceEpsReaction(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& eps,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& gByNu,
    const DeviceBuffer<scalar>& divU,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source,
    const KEpsilonCoeffs& co,
    const DeviceBuffer<scalar>* rho)
{
    {
        const int nC = dm.nCells;
        const scalar *Vd = dm.V.data(), *epsd = eps.data(), *kd = k.data(), *gByNud = gByNu.data(), *divUd = divU.data();
        const scalar C1 = co.C1, C2 = co.C2, C3 = co.C3, Cmu = co.Cmu, eta0 = co.eta0, beta = co.beta;
        const int rng = co.rng ? 1 : 0;
        scalar *diagd = diag.data(), *sourced = source.data(); const scalar* rhod = rho ? rho->data() : nullptr;
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            epsReactionKernel(nC, Vd, epsd, kd, gByNud, divUd, C1, C2, C3, Cmu, rng, eta0, beta, diagd, sourced, rhod);
        });
    }
    cudaCheck(cudaGetLastError(), "epsReaction");
}


void deviceKReaction(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& eps,
    const DeviceBuffer<scalar>& G,
    const DeviceBuffer<scalar>& divU,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source,
    const DeviceBuffer<scalar>* rho)
{
    {
        const int nC = dm.nCells;
        const scalar *Vd = dm.V.data(), *kd = k.data(), *epsd = eps.data(), *Gd = G.data(), *divUd = divU.data();
        scalar *diagd = diag.data(), *sourced = source.data(); const scalar* rhod = rho ? rho->data() : nullptr;
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            kReactionKernel(nC, Vd, kd, epsd, Gd, divUd, diagd, sourced, rhod);
        });
    }
    cudaCheck(cudaGetLastError(), "kReaction");
}


void deviceBoundaryNut(
    const DeviceBoundary& db,
    const DeviceBuffer<label>& isWall,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& nut,
    scalar nu,
    DeviceBuffer<scalar>& nutBnd,
    const KEpsilonCoeffs& co,
    scalar atmZ0,
    bool   atmBoundNut,
    const DeviceBuffer<scalar>* nuFace,
    const DeviceBuffer<label>*  calcMask,
    const DeviceBuffer<scalar>* kBnd,
    const DeviceBuffer<scalar>* epsBnd)
{
    // OF turbulenceModel::nu(patchi) = mu(patchi)/rho.boundaryField()[patchi] -- a per-FACE kinematic
    // viscosity. Constant-property incompressible flow makes that a single number, which is why the scalar
    // argument exists; compressible flow does not, so nuFace overrides it when supplied.
    nutBnd.resize(db.n);
    const scalar Cmu25 = std::pow(co.Cmu, 0.25), yplLam = yPlusLamHost(co.kappa, co.E);
    {
        const int n = db.n;
        const label* fc = db.faceCell.data(); const label* isWalld = isWall.data();
        const scalar *yd = y.data(), *kd = k.data(), *nutd = nut.data();
        const scalar kappa = co.kappa, E = co.E, Cmu = co.Cmu;
        scalar* nutBndd = nutBnd.data();
        const scalar* nuFaced = nuFace ? nuFace->data() : nullptr;
        const label* calcMaskd = calcMask ? calcMask->data() : nullptr;
        const scalar* kBndd = kBnd ? kBnd->data() : nullptr;
        const scalar* epsBndd = epsBnd ? epsBnd->data() : nullptr;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            boundaryNutKernel(n, fc, isWalld, yd, kd, nutd, nu, yplLam, Cmu25, kappa, E, atmZ0, atmBoundNut,
                              nutBndd, nuFaced, calcMaskd, kBndd, epsBndd, Cmu);
        });
    }
    cudaCheck(cudaGetLastError(), "boundaryNut");
}




// out = a/b, face by face. Turns the compressible MASS flux back into the volumetric one the turbulence
// dilatation term needs (OF compressibleTurbulenceModel::phi()).
__device__
void divideFaceK(
    int n,
    const scalar* __restrict__ a,
    const scalar* __restrict__ b,
    scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] / b[i];
}

// D <- rho*D + mu. OF's laplacian is alpha*rho*DEff with DEff kinematic, so the compressible coefficient
// is rho*(nut/sigma) + mu. D comes in built with nu = 0 so only the turbulent part is scaled.
static void scaleDEffCompressibleKE(
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& muLam,
    DeviceBuffer<scalar>& D)
{
    DeviceBuffer<scalar> t;
    deviceHadamard(t, rho, D);
    deviceCopy(D, t);
    deviceAxpy(1.0, muLam, D);
}

void deviceKEpsilonCorrect(
    const DeviceMesh& dm,
    const DeviceWallData& wall,
    const DeviceBoundary& dbEps,
    const DeviceBoundary& dbK,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& k,
    DeviceBuffer<scalar>& eps,
    DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    scalar nu,
    scalar relaxEps,
    scalar relaxK,
    scalar tol,
    bool bounded,
    bool boundedEps,   // div(phi,epsilon|omega) `bounded`; `bounded` above is div(phi,k)'s
    bool limitedK,
    bool limitedEps,
    scalar twoBykK,
    scalar twoBykEps,
    const KEpsilonCoeffs& co,
    scalar relTolKE,
    int keCheckEvery,
    bool linearUpwindK,
    bool linearUpwindEps,
    bool nonOrth,
    bool gsK,
    bool gsEps,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    int nutWall,
    scalar atmZ0,
    bool atmBoundNut,
    const ScalarDdt& kDdt,
    const ScalarDdt& eDdt,
    const DeviceBuffer<scalar>* rho,          // compressible: alpha*rho on every RHS term + the diffusivity
    const DeviceBuffer<scalar>* muLam,        // compressible: laminar DYNAMIC viscosity mu [Pa s]
    const DeviceBuffer<scalar>* rhoBnd,       // compressible: rho at boundary faces (volumetric flux for divU)
    const DeviceBuffer<scalar>* nuWallFace,   // compressible: nu = mu_b/rho_b per WALL face (OF nu(patchi))
    const DeviceBuffer<scalar>* nutBnd,       // nut at boundary FACES -> DkEff/DepsEff(patchi), as OF's laplacian uses
    const DeviceBuffer<scalar>* muBnd,
    scalar gradScalarLimitK,
    scalar gradULimitK,
                            const DeviceBuffer<label>*  fvoKMask,
                            const DeviceBuffer<scalar>* fvoKVal,
                            const DeviceBuffer<label>*  fvoEMask,
                            const DeviceBuffer<scalar>* fvoEVal)        // compressible: mu at boundary faces (the +mu of rho*D+mu)
{
    const int nC = dm.nCells;
    // production + divU, wall functions + near-wall override.
    DeviceBuffer<scalar> gByNu, rCmu, magS;
    // THE NAMED grad(U) SCHEME. OF's kEpsilon::correct() opens with `tmp<volTensorField> tgradU =
    // fvc::grad(U)`, which resolves gradSchemes `grad(U)` -- `cellLimited Gauss linear 1` on any case
    // that says so. kOmegaSST's brae path has honoured that for a long time; kEpsilon's never did, and
    // an UNLIMITED production gradient is larger exactly where the limiter would have bitten. Measured
    // on pimpleFoam/RAS/oscillatingInletACMI2D, step 1: GbyNu peaked at 1.9e+04 against OpenFOAM's
    // 3.6e+03 in the inlet channel, a factor of 5 on the production term itself.
    if (co.realizable || gradULimitK > scalar(0))
    {
        DeviceBuffer<scalar> gradU;
        deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);
        if (gradULimitK > scalar(0)) deviceCellLimitGradU(dm, dbU, Ux, Uy, Uz, gradU, gradULimitK, cyc, ami);
        deviceGByNuFromGradU(gradU, nC, gByNu);
        if (co.realizable) deviceRealizableStrain(gradU, k, eps, co.A0, nC, rCmu, magS);
    }
    else deviceGbyNu(dm, dbU, Ux, Uy, Uz, gByNu, ami, cyc);   // interface-aware grad(U) for production
    DeviceBuffer<scalar> G;
    deviceHadamard(G, nut, gByNu);
    // BRAE_DUMP_TERMS: GbyNu and its factors, before the wall override. Production is C1*Cmu*k*GbyNu,
    // and the term dump showed it collapsing two orders in one iteration while k held -- so GbyNu is the
    // suspect and it has to be looked at directly rather than inferred from the product.
    if (const char* td = std::getenv("BRAE_DUMP_TERMS"))
    {
        static int gCall = 0;
        const int gi = gCall++;
        std::error_code gec; std::filesystem::create_directories(td, gec);
        char gfn[512]; std::snprintf(gfn, sizeof gfn, "%s/gbynu_%04d", td, gi);
        std::ofstream go(gfn); go.precision(10);
        const std::vector<scalar> hG = gByNu.host(), hK = k.host(), hE = eps.host(), hN = nut.host();
        // The full gradU tensor too: GbyNu is a contraction, so a low value can mean either a small
        // gradient or a gradient that is nearly antisymmetric (pure rotation contracts to ~0 legitimately).
        // Only the components separate those two.
        DeviceBuffer<scalar> gradUd;
        deviceGradU(dm, dbU, Ux, Uy, Uz, gradUd, ami, cyc);
        if (gradULimitK > scalar(0)) deviceCellLimitGradU(dm, dbU, Ux, Uy, Uz, gradUd, gradULimitK, cyc, ami);
        const std::vector<scalar> hT = gradUd.host();
        go << "# cell gByNu k eps nut g0..g8 (OF-convention gradU, column i = grad(U_i))\n";
        for (int c = 0; c < nC; ++c)
        {
            go << c << ' ' << hG[c] << ' ' << hK[c] << ' ' << hE[c] << ' ' << hN[c];
            for (int q = 0; q < 9; ++q) go << ' ' << hT[q*nC + c];   // 9*nC, component-major
            go << '\n';
        }
    }
    // TWO divergences, as in the SST. "bounded" subtracts div of the flux the CONVECTION carries (the
    // MASS flux when compressible); the reactions' dilatation term is div of the VOLUMETRIC flux, which
    // OF builds from compressibleTurbulenceModel::phi() = phi/fvc::interpolate(rho). Equal at constant
    // rho, so incompressible is bit-identical.
    DeviceBuffer<scalar> divPhi;
    deviceDiv(dm, phiInt, phiBnd, divPhi);
    if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divPhi);   // interface flux into div(phi) (bounded term)
    if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divPhi);
    DeviceBuffer<scalar> divU;
    if (rho && rhoBnd && rhoBnd->size())
    {
        DeviceBuffer<scalar> rhoF;
        deviceInterpolate(dm, *rho, rhoF);
        const int nIf = dm.nInternalFaces;
        DeviceBuffer<scalar> pvI(static_cast<std::size_t>(nIf));
        {
            const scalar *phiIntd = phiInt.data(), *rhoFd = rhoF.data(); scalar* pvId = pvI.data();
            pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () { divideFaceK(nIf, phiIntd, rhoFd, pvId); });
        }
        const int nB = static_cast<int>(phiBnd.size());
        DeviceBuffer<scalar> pvB(static_cast<std::size_t>(nB));
        {
            const scalar *phiBndd = phiBnd.data(), *rhoBndd = rhoBnd->data(); scalar* pvBd = pvB.data();
            pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () { divideFaceK(nB, phiBndd, rhoBndd, pvBd); });
        }
        cudaCheck(cudaGetLastError(), "keVolFlux");
        deviceDiv(dm, pvI, pvB, divU);
        if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divU);
        if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divU);
    }
    else deviceCopy(divU, divPhi);
    DeviceBuffer<scalar> eps0, G0;
    deviceWallEpsG0(wall, k, Ux, Uy, Uz, nu, eps0, G0, co, nutWall, atmZ0, atmBoundNut, nuWallFace);
    // BRAE_DUMP_TERMS: the wall-function inputs, BEFORE the override rewrites eps/eps0. eps0 is the raw
    // near-wall value invNw*Cmu^.75*k^1.5/(kappa*y), so a disagreement with OF here is one of invNw
    // (the corner weighting, 1/nWallFaces), y, or k -- and the three are separable only side by side.
    if (const char* wd = std::getenv("BRAE_DUMP_TERMS"))
    {
        static int wCall = 0;
        const int wi = wCall++;
        std::error_code wec; std::filesystem::create_directories(wd, wec);
        char wfn[512]; std::snprintf(wfn, sizeof wfn, "%s/wall_%04d", wd, wi);
        std::ofstream wo(wfn); wo.precision(10);
        const std::vector<scalar> hE0 = eps0.host(), hG0 = G0.host(), hW = wall.wallW.size() ? wall.wallW.host() : std::vector<scalar>(nC, 1.0);
        const std::vector<scalar> hInv = wall.invNw.host(), hE = eps.host(), hK = k.host();
        const std::vector<label>  hIs = wall.isWallCell.host();
        wo << "# cell isWall wallW invNw eps0 G0 epsIn k\n";
        for (int c = 0; c < nC; ++c)
            wo << c << ' ' << (int)hIs[c] << ' ' << hW[c] << ' ' << hInv[c] << ' '
               << hE0[c] << ' ' << hG0[c] << ' ' << hE[c] << ' ' << hK[c] << '\n';
    }
    {
        const label* isWd = wall.isWallCell.data();
        const scalar* G0d = G0.data(); scalar* eps0d = eps0.data(); scalar* Gd = G.data(); scalar* epsd = eps.data();
        const scalar* wallWd = wall.wallW.size() ? wall.wallW.data() : nullptr;
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { overrideKernel(nC, isWd, G0d, eps0d, Gd, epsd, wallWd); });
    }

    // epsilon equation (loose solve) with the near-wall setValues constraint
    // DepsilonEff = nut/sigmaEps + nu. OF's laplacian is alpha*rho*DepsilonEff, so compressible wants
    // rho*(nut/sigmaEps) + mu -- build it kinematic with nu=0 then scale, exactly as the SST does.
    DeviceBuffer<scalar> Deps(static_cast<std::size_t>(nC));
    {
        const scalar* nutd = nut.data(); const scalar sigmaEps = co.sigmaEps; const scalar nuArg = rho ? scalar(0) : nu;
        scalar* Depsd = Deps.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { depsKernel(nC, nutd, sigmaEps, nuArg, Depsd); });
    }
    if (rho && muLam) scaleDEffCompressibleKE(*rho, *muLam, Deps);
    // OF's laplacian(DepsilonEff, epsilon) uses the PATCH diffusivity, DepsilonEff(patchi) = nut_b/sigmaEps
    // + nu_b -- not the adjacent cell's. Identical wherever nut_b happens to equal nut_cell, which is why
    // this only shows up once nut_b is evaluated correctly (see the boundaryNut 'calculated' branch).
    DeviceBuffer<scalar> DepsB, DkB;
    if (nutBnd && nutBnd->size())
    {
        const int nB = static_cast<int>(nutBnd->size());
        DepsB.resize(nB); DkB.resize(nB);
        {
            const scalar* nutBndd = nutBnd->data(); const scalar sigmaEps = co.sigmaEps, sigmaK = co.sigmaK;
            const scalar nuArg = rho ? scalar(0) : nu; scalar *DepsBd = DepsB.data(), *DkBd = DkB.data();
            pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () { depsKernel(nB, nutBndd, sigmaEps, nuArg, DepsBd); });
            pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () { depsKernel(nB, nutBndd, sigmaK, nuArg, DkBd); });
        }
        cudaCheck(cudaGetLastError(), "DEffBnd");
        if (rho && rhoBnd && muBnd && muBnd->size())
        {
            scaleDEffCompressibleKE(*rhoBnd, *muBnd, DepsB);
            scaleDEffCompressibleKE(*rhoBnd, *muBnd, DkB);
        }
    }
    deviceSolveScalarTransport(dm, dbEps, eps, "epsilon", Deps, phiInt, phiBnd, divPhi, boundedEps, limitedEps, linearUpwindEps, nonOrth, twoBykEps,
                               relaxEps, tol, relTolKE, keCheckEvery, gsEps,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){
                                   if (co.realizable) deviceEpsReactionRealizable(dm, eps, k, magS, nu, co.C2, diag, src);
                                   else               deviceEpsReaction(dm, eps, k, gByNu, divU, diag, src, co, rho); },
                               &wall, &eps0, ami, cyc, eDdt, DepsB.size() ? &DepsB : nullptr, gradScalarLimitK,
                               true, fvoEMask, fvoEVal);

    // k equation (loose solve)
    DeviceBuffer<scalar> Dk(static_cast<std::size_t>(nC));
    {
        const scalar* nutd = nut.data(); const scalar sigmaK = co.sigmaK; const scalar nuArg = rho ? scalar(0) : nu;
        scalar* Dkd = Dk.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { depsKernel(nC, nutd, sigmaK, nuArg, Dkd); });
    }
    if (rho && muLam) scaleDEffCompressibleKE(*rho, *muLam, Dk);
    deviceSolveScalarTransport(dm, dbK, k, "k", Dk, phiInt, phiBnd, divPhi, bounded, limitedK, linearUpwindK, nonOrth, twoBykK,
                               relaxK, tol, relTolKE, keCheckEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ deviceKReaction(dm, k, eps, G, divU, diag, src, rho); },
                               nullptr, nullptr, ami, cyc, kDdt, DkB.size() ? &DkB : nullptr, gradScalarLimitK,
                               true, fvoKMask, fvoKVal);

    // correctNut (cell): nut = Cmu k^2 / eps (realizableKE: rCmu k^2 / eps with the variable Cmu).
    if (co.realizable) deviceRealizableNut(rCmu, k, eps, nut);
    else               deviceNut(k, eps, nut, co);
}


// OF kOmegaSSTBase::correct/correctNut use tgradU = fvc::grad(U) = the named grad(U) scheme for BOTH the strain


namespace {  // wall-function nut kernels (spaldingNut/blendedNut), used by deviceBoundaryNut* below
// nutUSpaldingWallFunction: wall faces -> Newton uTau from Spalding's law, nut = max(0, uTau^2/magGradU - nu).
// magGradU = |snGrad U| = |U_cell|*deltaCoeffs (noSlip U_wall=0); magUp = |U_cell|; y = near-wall distance.
// Warm-started from the previous wall nut (nutBnd in/out), 10 Newton iters with the OF tol=0.01 early-out.
__device__
void spaldingNutKernel(
    int n,
    const label* __restrict__ fc,
    const label* __restrict__ isWall,
    const scalar* __restrict__ y,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ nutCell,
    scalar nu,
    scalar kappa,
    scalar E,
    scalar* __restrict__ nutBnd,
    const scalar* __restrict__ nuFace,   // compressible: per-face nu = mu_b/rho_b, null -> the scalar nu
    const scalar* __restrict__ nutFile)  // the patch's OWN nut boundaryField, null -> extrapolate
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    // OF evaluates nut on a non-wall patch from that patch's OWN boundaryField; extrapolating the
    // adjacent cell is 607x wrong at a `calculated` inlet (see rhosimplefoam-ground-truth-port.md).
    if (!isWall[i]) { nutBnd[i] = nutFile ? nutFile[i] : nutCell[c]; return; }
    const scalar magUp = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    const scalar magGradU = magUp * dc[i];
    const scalar nuw = nuFace ? nuFace[i] : nu;
    nutBnd[i] = spaldingNutValue(magUp, magGradU, y[i], nuw, kappa, E, nutBnd[i]);   // shared with G0 (nut_wall_function.cuh)
}


// nutUBlendedWallFunction (OpenFOAM v2412): wall faces -> uTau from the binomial blend of the viscous and
// log velocity scales, nut = max(0, uTau^2/magGradU - nu). uTau = (uTauVis^n + uTauLog^n)^(1/n), n=4:
//   yPlus = y*uTau/nu ; uTauVis = magUp/yPlus ; uTauLog = kappa*magUp/log(max(E*yPlus, 1+1e-4)).
// 10 iters, tol 1e-3, under-relaxed uTau update (ut = 0.5*(ut+utNew)), warm-started like the Spalding kernel.
// magUp/magGradU use the same convention as spaldingNutKernel (|U_cell|, |snGrad U|=|U_cell|*deltaCoeffs).
__device__
void blendedNutKernel(
    int n,
    const label* __restrict__ fc,
    const label* __restrict__ isWall,
    const scalar* __restrict__ y,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ nutCell,
    scalar nu,
    scalar kappa,
    scalar E,
    scalar* __restrict__ nutBnd,
    const scalar* __restrict__ nuFace,   // compressible: per-face nu = mu_b/rho_b, null -> the scalar nu
    const scalar* __restrict__ nutFile)  // the patch's OWN nut boundaryField, null -> extrapolate
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    // OF evaluates nut on a non-wall patch from that patch's OWN boundaryField; extrapolating the
    // adjacent cell is 607x wrong at a `calculated` inlet (see rhosimplefoam-ground-truth-port.md).
    if (!isWall[i]) { nutBnd[i] = nutFile ? nutFile[i] : nutCell[c]; return; }
    const scalar magUp = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    const scalar magGradU = magUp * dc[i];
    const scalar nuw = nuFace ? nuFace[i] : nu;
    nutBnd[i] = blendedNutValue(magUp, magGradU, y[i], nuw, kappa, E, nutBnd[i]);   // shared with G0 (nut_wall_function.cuh)
}
} // namespace


void deviceBoundaryNutSpalding(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<label>& isWall,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& nutCell,
    scalar nu,
    const SpalartAllmarasCoeffs& co,
    DeviceBuffer<scalar>& nutBnd,
    const DeviceBuffer<scalar>* nuFace,
    const DeviceBuffer<scalar>* nutFile)
{
    // OF turbulenceModel::nu(patchi) = mu(patchi)/rho.boundaryField()[patchi] -- a per-FACE kinematic
    // viscosity. Constant-property incompressible flow makes that a single number, which is why the scalar
    // argument exists; compressible flow does not, so nuFace overrides it when supplied.

    const int n = dbU.comp[0].n;
    if (static_cast<int>(nutBnd.size()) != n)   // first call: zero seed (warm-starts thereafter)
    {
        nutBnd.resize(n);
        cudaCheck(cudaMemsetAsync(nutBnd.data(), 0, n*sizeof(scalar), cudaStreamPerThread), "spalding init");
    }
    {
        const label* fc = dbU.comp[0].faceCell.data(); const label* isWalld = isWall.data();
        const scalar *yd = y.data(), *dcd = dbU.comp[0].deltaCoeffs.data();
        const scalar *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data(), *nutCelld = nutCell.data();
        const scalar kappa = co.kappa, E = co.E; scalar* nutBndd = nutBnd.data();
        const scalar* nuFaced = nuFace ? nuFace->data() : nullptr;
        const scalar* nutFiled = nutFile ? nutFile->data() : nullptr;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            spaldingNutKernel(n, fc, isWalld, yd, dcd, Uxd, Uyd, Uzd, nutCelld, nu, kappa, E, nutBndd, nuFaced, nutFiled);
        });
    }
    cudaCheck(cudaGetLastError(), "spaldingNut");
}


// nutUBlendedWallFunction wall nut (velocity-based binomial blend). kappa/E passed explicitly so it works
// on ANY RAS model (kEpsilon/kOmegaSST/SA), honouring the 0/nut BC type rather than the model.
void deviceBoundaryNutBlended(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<label>& isWall,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& nutCell,
    scalar nu,
    scalar kappa,
    scalar E,
    DeviceBuffer<scalar>& nutBnd,
    const DeviceBuffer<scalar>* nuFace,
    const DeviceBuffer<scalar>* nutFile)
{
    // OF turbulenceModel::nu(patchi) = mu(patchi)/rho.boundaryField()[patchi] -- a per-FACE kinematic
    // viscosity. Constant-property incompressible flow makes that a single number, which is why the scalar
    // argument exists; compressible flow does not, so nuFace overrides it when supplied.

    const int n = dbU.comp[0].n;
    if (static_cast<int>(nutBnd.size()) != n)   // first call: zero seed (warm-starts thereafter)
    {
        nutBnd.resize(n);
        cudaCheck(cudaMemsetAsync(nutBnd.data(), 0, n*sizeof(scalar), cudaStreamPerThread), "blended init");
    }
    {
        const label* fc = dbU.comp[0].faceCell.data(); const label* isWalld = isWall.data();
        const scalar *yd = y.data(), *dcd = dbU.comp[0].deltaCoeffs.data();
        const scalar *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data(), *nutCelld = nutCell.data();
        scalar* nutBndd = nutBnd.data();
        const scalar* nuFaced = nuFace ? nuFace->data() : nullptr;
        const scalar* nutFiled = nutFile ? nutFile->data() : nullptr;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            blendedNutKernel(n, fc, isWalld, yd, dcd, Uxd, Uyd, Uzd, nutCelld, nu, kappa, E, nutBndd, nuFaced, nutFiled);
        });
    }
    cudaCheck(cudaGetLastError(), "blendedNut");
}


} // namespace brae

namespace brae {
namespace {
// nutUWallFunction boundary nut: log-law yPlus, STEPWISE blend (OF's default). Shares the wall-face
// geometry and the |U_cell - U_wall| convention with the Spalding/Blended kernels above.
__device__
void nutUWallKernel(
    int n, const label* __restrict__ fc, const label* __restrict__ isWall,
    const scalar* __restrict__ y, const scalar* __restrict__ dc,
    const scalar* __restrict__ Ux, const scalar* __restrict__ Uy, const scalar* __restrict__ Uz,
    const scalar* __restrict__ nutCell, scalar nu, scalar kappa, scalar E, scalar yPlusLam,
    scalar* __restrict__ nutBnd, const scalar* __restrict__ nuFace,
    const scalar* __restrict__ nutFile)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = fc[i];
    // OF evaluates nut on a non-wall patch from that patch's OWN boundaryField; extrapolating the
    // adjacent cell is 607x wrong at a `calculated` inlet (see rhosimplefoam-ground-truth-port.md).
    if (!isWall[i]) { nutBnd[i] = nutFile ? nutFile[i] : nutCell[c]; return; }
    const scalar magUp = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);   // wall is stationary
    const scalar nuw = nuFace ? nuFace[i] : nu;
    nutBnd[i] = nutUWallValue(magUp, y[i], nuw, kappa, E, yPlusLam);
    (void)dc;
}
}   // namespace

void deviceBoundaryNutU(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<label>& isWall,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& nutCell,
    scalar nu, scalar kappa, scalar E,
    DeviceBuffer<scalar>& nutBnd,
    const DeviceBuffer<scalar>* nuFace,
    const DeviceBuffer<scalar>* nutFile)
{
    const int n = dbU.comp[0].n;
    if (!n) return;
    if (static_cast<int>(nutBnd.size()) != n) nutBnd.resize(n);
    {
        const label* fc = dbU.comp[0].faceCell.data(); const label* isWalld = isWall.data();
        const scalar *yd = y.data(), *dcd = dbU.comp[0].deltaCoeffs.data();
        const scalar *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data(), *nutCelld = nutCell.data();
        const scalar yplLam = yPlusLamHost(kappa, E); scalar* nutBndd = nutBnd.data();
        const scalar* nuFaced = nuFace ? nuFace->data() : nullptr;
        const scalar* nutFiled = nutFile ? nutFile->data() : nullptr;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            nutUWallKernel(n, fc, isWalld, yd, dcd, Uxd, Uyd, Uzd, nutCelld, nu, kappa, E, yplLam, nutBndd, nuFaced, nutFiled);
        });
    }
    cudaCheck(cudaGetLastError(), "nutUWall");
}

}   // namespace brae
