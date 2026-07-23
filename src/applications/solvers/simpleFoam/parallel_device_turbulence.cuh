#pragma once
// brae distributed DEVICE turbulence (Phase 4b). Built on the 4a machinery -- NO new communication mechanism,
// just more fields through the same processor interface (the plan's promise). The reusable core is
// deviceParallelScalarTransport: one distributed k/epsilon/omega/nuTilda transport solve. It composes ONLY
// primitives already teeth-tested in 4a:
//   deviceMomentumInterface  -- the processor convection+diffusion coeff (scalar-valued; w*phi + coeff)
//   deviceInterfaceOffDiagSum + deviceRelaxDiag -- relax with the interface in the diagonal-dominance sum
//   deviceParallelJacobiBiCGStab -- the interface-coupled non-symmetric solve
// mirroring the host solveScalarEqn (parallel_kepsilon.cuh): momentumDistributed -> relaxScalar -> setValues
// -> parallelPBiCGStab.
//
// The wall functions and the nut update are ENTIRELY LOCAL (a cell owns its own wall face; a cut cannot
// separate them -- confirmed by the host oracle's "WALL FUNCTIONS are unchanged and entirely local"), so they
// reuse the single-GPU kernels unchanged. The one genuinely new device piece is the near-wall setValues
// constraint (fixed epsilon at wall-adjacent cells), which also zeroes those cells' interface coeffs.
#include "parallel_device_interface.cuh"   // deviceMomentumInterface / deviceInterfaceOffDiagSum
#include "device_simple.cuh"
#include "device_pcg.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"
#include <vector>

namespace brae {

namespace detail {

// setValues (eps near-wall constraint) for the DISTRIBUTED matrix -- the exact analogue of the single-GPU
// svFace/svBnd/svCell (device_kepsilon.cu), plus the processor interface. A wall cell's eps is FIXED = eps0, so
// its entire row must become diag*eps = diag*eps0: zero every off-diagonal (internal, boundary AND interface)
// and move the known eps0 to each neighbour's RHS. My first version zeroed only the interface, which left the
// wall cell's INTERNAL off-diagonals live -- so at np=1 it coupled to its x+1 neighbour (internal) while at
// np>1 that same coupling was the zeroed interface: a ~2% cut-face mismatch. All four pieces are required.
__global__
void svFaceKernel(   // internal faces: zero wall-cell off-diagonals, move eps0 to the neighbour RHS
    int nIf, const label* __restrict__ own, const label* __restrict__ nei, const label* __restrict__ isW,
    const scalar* __restrict__ eps0, scalar* __restrict__ upper, scalar* __restrict__ lower, scalar* __restrict__ source)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const int o = own[f], n = nei[f];
    const bool ow = isW[o], nw = isW[n];
    if (ow) atomicAdd(&source[n], -lower[f] * eps0[o]);
    if (nw) atomicAdd(&source[o], -upper[f] * eps0[n]);
    if (ow || nw) { upper[f] = 0.0; lower[f] = 0.0; }
}

__global__
void svBndKernel(   // boundary faces: zero wall-cell boundary coeffs
    int nB, const label* __restrict__ faceCell, const label* __restrict__ isW,
    scalar* __restrict__ iC, scalar* __restrict__ bC)
{
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi < nB && isW[faceCell[bi]]) { iC[bi] = 0.0; bC[bi] = 0.0; }
}

__global__
void svCellKernel(   // wall cells: source = relaxedDiag * eps0 (with off-diagonals zeroed -> eps = eps0)
    int nC, const label* __restrict__ isW, const scalar* __restrict__ relaxedDiag,
    const scalar* __restrict__ eps0, scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC && isW[c]) source[c] = relaxedDiag[c] * eps0[c];
}

__global__
void svInterfaceKernel(   // processor faces: zero the interface off-diagonal of a wall-cell owner
    const label* __restrict__ faceCells, const label* __restrict__ isW, scalar* __restrict__ ifCoeff, int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n && isW[faceCells[f]]) ifCoeff[f] = 0.0;
}


// eps equation local source assembly (mirrors host parallelCorrect exactly). sp = ((2/3)C1 - C3)*divU.
//   Sp[c] = ( C2*eps/k + max(sp,0) - bounded*divU ) * V          (implicit, added to the diagonal)
//   Su[c] = ( C1*Cmu*k*gByNu - min(sp,0)*eps )     * V          (explicit source)
__global__
void epsSourceKernel(
    const scalar* __restrict__ k, const scalar* __restrict__ eps, const scalar* __restrict__ gByNu,
    const scalar* __restrict__ divU, const scalar* __restrict__ V,
    scalar C1, scalar C2, scalar C3, scalar Cmu, int bounded,
    scalar* __restrict__ Sp, scalar* __restrict__ Su, int nC)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar sp = ((2.0/3.0)*C1 - C3) * divU[c];
    Sp[c] = ( C2 * eps[c] / k[c] + fmax(sp, 0.0) - (bounded ? divU[c] : 0.0) ) * V[c];
    Su[c] = ( C1 * Cmu * k[c] * gByNu[c] - fmin(sp, 0.0) * eps[c] ) * V[c];
}

// k equation local source assembly. sp = (2/3)*divU. G = the production (wall-overridden at wall cells).
//   Sp[c] = ( eps/k + max(sp,0) - bounded*divU ) * V ; Su[c] = ( G - min(sp,0)*k ) * V
__global__
void kSourceKernel(
    const scalar* __restrict__ k, const scalar* __restrict__ eps, const scalar* __restrict__ G,
    const scalar* __restrict__ divU, const scalar* __restrict__ V, int bounded,
    scalar* __restrict__ Sp, scalar* __restrict__ Su, int nC)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar sp = (2.0/3.0) * divU[c];
    Sp[c] = ( eps[c] / k[c] + fmax(sp, 0.0) - (bounded ? divU[c] : 0.0) ) * V[c];
    Su[c] = ( G[c] - fmin(sp, 0.0) * k[c] ) * V[c];
}

// production G = nut * gByNu, then OVERRIDE G at wall cells with G0 and set eps := eps0 (host wall treatment).
__global__
void prodAndWallKernel(
    const scalar* __restrict__ nut, const scalar* __restrict__ gByNu,
    const label*  __restrict__ isWall, const scalar* __restrict__ eps0, const scalar* __restrict__ G0,
    scalar* __restrict__ G, scalar* __restrict__ eps, int nC)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    if (isWall[c]) { G[c] = G0[c]; eps[c] = eps0[c]; }
    else           { G[c] = nut[c] * gByNu[c]; }
}

// lower-bound a field in place (OF bound(psi, small)).
__global__
void boundFieldKernel(scalar* __restrict__ psi, scalar lo, int nC)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    psi[c] = fmax(psi[c], lo);
}

} // namespace detail

// Solve one distributed scalar transport equation on the device, matching host solveScalarEqn:
//     div(phi, psi) - laplacian(gamma, psi) + Sp*psi == Su      (+ processor interface, + wall setValues)
// psi is solved IN PLACE. Returns the initial residual (OF residualControl measure). All the *_interface
// helpers are the 4a ones. `phiF[i]`/`gammaCoeffGeo[i]` are the processor-face flux and gammaF*magSf*procDelta
// of interface i (same order as the halo), `Sp`/`Su` the local implicit/explicit source (already assembled by
// the caller, as the host adds them to Mloc before solveScalarEqn). conCells/conVals: near-wall constraint.
inline scalar deviceParallelScalarTransport(
    const DeviceMesh& dm,
    DeviceHalo& halo,
    const DeviceBoundary& db,
    const std::vector<DeviceBuffer<label>>& faceCellsD,
    const std::vector<label>& procStart,
    const std::vector<DeviceBuffer<scalar>>& weights,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const DeviceBuffer<scalar>& gammaFint,
    const DeviceBuffer<scalar>& gammaBnd,
    const std::vector<DeviceBuffer<scalar>>& gammaCoeffGeo,
    const DeviceBuffer<scalar>& Sp,
    const DeviceBuffer<scalar>& Su,
    DeviceBuffer<scalar>& psi,
    scalar relax,
    scalar tol,
    int maxIter,
    label globalNCells,
    const DeviceBuffer<scalar>& ones,
    const DeviceBuffer<label>&               isWallCell,  // per-cell wall mask (empty -> no constraint, e.g. k)
    const DeviceBuffer<scalar>&              wallVal,     // per-cell eps0 (read only where isWallCell)
    // INEXACT SIMPLE, same as the momentum/pressure solves: OF stops each turbulence transport at a RELATIVE
    // reduction per outer iteration. Trailing + defaulted 0.0 so existing callers keep the fully-converged
    // behaviour the np=1-vs-np>1 correctness tests rely on.
    scalar relTol = 0.0)
{
    const label lnC = dm.nCells;

    // local matrix: div(phi,psi) - laplacian(gamma,psi)
    DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
    deviceDivUpwindCoeffs(dm, phiInt, mDiag, mUp, mLo);
    deviceLaplacianCoeffs(dm, gammaFint, lD, lU, lL, false);
    deviceAxpy(-1.0, lD, mDiag);
    deviceAxpy(-1.0, lU, mUp);
    deviceAxpy(-1.0, lL, mLo);
    deviceAxpy(1.0, Sp, mDiag);                              // + Sp on the diagonal (local)

    // processor interface: convection + diffusion coeff (scalar), folds into mDiag, yields ifCoeff
    std::vector<DeviceBuffer<scalar>> ifCoeff;
    deviceMomentumInterface(halo, faceCellsD, phiF, gammaCoeffGeo, mDiag, ifCoeff);

    // boundary coeffs of div - laplacian
    DeviceBuffer<scalar> iC, bC, lIC, lBC;
    deviceBCDivCoeffs(db, phiBnd, iC, bC);
    deviceBCLaplacianCoeffsFace(db, gammaBnd, lIC, lBC);
    deviceAxpy(-1.0, lIC, iC);
    deviceAxpy(-1.0, lBC, bC);

    // relax (interface counts toward diagonal dominance), source += delta*psi_old
    DeviceBuffer<scalar> sumOff(std::vector<scalar>(lnC, 0.0));
    deviceInterfaceOffDiagSum(halo, faceCellsD, ifCoeff, sumOff);
    DeviceBuffer<scalar> mDiagR, delta;
    deviceRelaxDiag(deviceLduView(dm, mDiag, mUp, mLo), dm, iC, relax, mDiagR, delta, sumOff.data());
    DeviceBuffer<scalar> src;
    deviceHadamard(src, delta, psi);
    deviceAxpy(1.0, Su, src);                                // + explicit source (local)

    // near-wall setValues (eps): pin psi = eps0 at wall cells by zeroing their ENTIRE row -- internal
    // off-diagonals (svFace, moving eps0 to neighbour RHS), boundary coeffs (svBnd), the processor interface
    // (svInterface), and setting source = relaxedDiag*eps0 (svCell). Applied BEFORE the fold, exactly as the
    // single-GPU deviceSolveScalarTransport. Empty mask -> no constraint (the k equation).
    if (isWallCell.size() > 0)
    {
        constexpr int TPB = 128;
        detail::svFaceKernel<<<(dm.nInternalFaces + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
            dm.nInternalFaces, dm.owner.data(), dm.nei.data(), isWallCell.data(), wallVal.data(),
            mUp.data(), mLo.data(), src.data());
        detail::svBndKernel<<<(dm.nBndFaces + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
            dm.nBndFaces, dm.bndCell.data(), isWallCell.data(), iC.data(), bC.data());
        for (int i = 0; i < halo.nInterfaces(); ++i)
        {
            const int n = static_cast<int>(halo.size(i));
            if (n <= 0) continue;
            detail::svInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
                faceCellsD[i].data(), isWallCell.data(), ifCoeff[i].data(), n);
        }
        detail::svCellKernel<<<(lnC + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
            lnC, isWallCell.data(), mDiagR.data(), wallVal.data(), src.data());
    }

    // fold boundary iC/bC into diag/source
    DeviceBuffer<scalar> diagC, b;
    deviceFold(dm, mDiagR, src, iC, bC, diagC, b);

    const DeviceLduView mv = deviceLduView(dm, diagC, mUp, mLo);
    const scalar nf = deviceParallelNormFactor(mv, halo, ifCoeff, psi, b, ones, globalNCells);
    const DeviceSolverPerf perf =
        deviceParallelJacobiBiCGStab(mv, halo, ifCoeff, b, psi, nf, tol, relTol, maxIter);
    return perf.initialResidual;
}


// Distributed device kEpsilon::correct() -- the composition, mirroring host kepsilon::parallelCorrect exactly.
// k/eps/nut updated in place. Reuses the LOCAL single-GPU kernels unchanged (wall functions, GbyNu, nut) and
// the distributed scalar-transport core (deviceParallelScalarTransport) for the two transport solves. The only
// processor-coupled pieces are (a) gradU: U's cut-face value from the halo before gaussGrad, and (b) Deps/Dk at
// cut faces: the halo-interpolated diffusivity. Everything else -- production, wall functions, the Sp/Su source
// terms, nut -- is entirely local (a cell owns its own wall face; the host oracle confirms wall functions are
// local). `geomD[i]` is magSf*procDelta of interface i (static). `nutBndOut` receives the wall-function nut.
inline void parallelDeviceKEpsilonCorrect(
    const DeviceMesh& dm,
    DeviceHalo& halo,
    const DeviceWallData& wall,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary& dbK,
    const DeviceBoundary& dbEps,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& k, DeviceBuffer<scalar>& eps, DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const std::vector<DeviceBuffer<label>>& faceCellsD,
    const std::vector<label>& procStart,
    const std::vector<DeviceBuffer<scalar>>& weightsD,
    const std::vector<DeviceBuffer<scalar>>& geomD,      // magSf*procDelta per interface
    const DeviceBuffer<label>& isWallCell,
    scalar nu, scalar relaxEps, scalar relaxK, scalar tol, int maxIter, label globalNCells,
    const DeviceBuffer<scalar>& ones, bool bounded, const KEpsilonCoeffs& co,
    scalar* resEps = nullptr, scalar* resK = nullptr, scalar relTolKE = 0.0)
{
    constexpr int TPB = 128;
    const int nC = dm.nCells;
    auto nb = [&](int n){ return (n + TPB - 1) / TPB; };

    // ---- halo-consistent gradU (9*nC, column i = gaussGrad(U_i)); processor cut value from the halo ----
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };
    DeviceBuffer<scalar> gradU(static_cast<std::size_t>(9) * nC);
    for (int i = 0; i < 3; ++i)
    {
        DeviceBuffer<scalar> ub;
        deviceBCValue(dbU.comp[i], *Uc[i], ub);
        halo.exchange(Uc[i]->data());
        halo.scatterBoundaryValues(Uc[i]->data(), weightsD, procStart, ub.data());
        halo.waitExchange();
        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, *Uc[i], ub, gx, gy, gz);
        cudaCheck(cudaMemcpyAsync(gradU.data() + (0*3+i)*static_cast<std::size_t>(nC), gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "keGradU x");
        cudaCheck(cudaMemcpyAsync(gradU.data() + (1*3+i)*static_cast<std::size_t>(nC), gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "keGradU y");
        cudaCheck(cudaMemcpyAsync(gradU.data() + (2*3+i)*static_cast<std::size_t>(nC), gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "keGradU z");
    }
    DeviceBuffer<scalar> gByNu;
    deviceGByNuFromGradU(gradU, nC, gByNu);
    // realizableKE: variable Cmu (rCmu) + strain magnitude (magS) from the SAME halo-consistent gradU tensor
    // (cell-local, no extra halo). Feeds the strain-based eps reaction + nut = rCmu*k^2/eps below.
    DeviceBuffer<scalar> rCmu, magS;
    if (co.realizable) deviceRealizableStrain(gradU, k, eps, co.A0, nC, rCmu, magS);

    // ---- divU (processor-consistent via the maintained phi) ----
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);

    // ---- wall functions (LOCAL) -> eps0, G0 at wall cells ----
    DeviceBuffer<scalar> eps0, G0;
    deviceWallEpsG0(wall, k, Ux, Uy, Uz, nu, eps0, G0, co);

    // production G = nut*gByNu, overridden to G0 at wall cells; eps := eps0 at wall cells
    DeviceBuffer<scalar> G(nC);
    detail::prodAndWallKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(
        nut.data(), gByNu.data(), isWallCell.data(), eps0.data(), G0.data(), G.data(), eps.data(), nC);

    // ---- helper: Deps/Dk = gammaCell/sigma + nu, at internal + boundary + processor faces ----
    auto faceGammaGeo = [&](const DeviceBuffer<scalar>& gammaCell,
                            DeviceBuffer<scalar>& gInt, DeviceBuffer<scalar>& gBnd,
                            std::vector<DeviceBuffer<scalar>>& gCoeffGeo)
    {
        deviceInterpolate(dm, gammaCell, gInt);                 // internal faces
        deviceBCValue(dbEps, gammaCell, gBnd);                  // boundary faces (zeroGradient shape -> adj cell)
        // processor faces: halo-interpolated gammaCell
        DeviceBuffer<scalar> gBndProc;
        deviceBCValue(dbEps, gammaCell, gBndProc);
        halo.exchange(gammaCell.data());
        halo.scatterBoundaryValues(gammaCell.data(), weightsD, procStart, gBndProc.data());
        halo.waitExchange();
        const std::vector<scalar> gh = gBndProc.host();
        gCoeffGeo.clear(); gCoeffGeo.resize(procStart.size());
        for (std::size_t i = 0; i < procStart.size(); ++i)
        {
            const int n = static_cast<int>(halo.size(static_cast<int>(i)));
            if (n <= 0) continue;
            const std::vector<scalar> gd = geomD[i].host();
            std::vector<scalar> cg(n);
            for (int f = 0; f < n; ++f) cg[f] = gh[procStart[i] + f] * gd[f];   // gammaFace * magSf*procDelta
            gCoeffGeo[i].copyFrom(cg);
        }
    };

    // ---- epsilon equation ----
    DeviceBuffer<scalar> DepsCell(nC), Deps_i, Deps_b;
    { std::vector<scalar> d(nC); const std::vector<scalar> nh = nut.host(); for (int c=0;c<nC;++c) d[c]=nh[c]/co.sigmaEps+nu; DepsCell.copyFrom(d); }
    std::vector<DeviceBuffer<scalar>> DepsGeo;
    faceGammaGeo(DepsCell, Deps_i, Deps_b, DepsGeo);
    DeviceBuffer<scalar> SpE(nC), SuE(nC);
    if (co.realizable)
    {
        // realizableKE eps reaction: production C1(eta)*magS*eps + destruction C2*eps^2/(k+sqrt(nu*eps)). The kernel
        // ACCUMULATES (+=) so zero first; then fold in the bounded -div(phi) term (Sp += -divU*V) that the standard
        // epsSourceKernel applies -- distributed transport does not handle `bounded`, the source does.
        cudaCheck(cudaMemsetAsync(SpE.data(), 0, static_cast<std::size_t>(nC)*sizeof(scalar), cudaStreamPerThread), "rke SpE 0");
        cudaCheck(cudaMemsetAsync(SuE.data(), 0, static_cast<std::size_t>(nC)*sizeof(scalar), cudaStreamPerThread), "rke SuE 0");
        deviceEpsReactionRealizable(dm, eps, k, magS, nu, co.C2, SpE, SuE);
        if (bounded)
        {
            DeviceBuffer<scalar> vDiv;
            deviceHadamard(vDiv, divU, dm.V);
            deviceAxpy(-1.0, vDiv, SpE);
        }
    }
    else
        detail::epsSourceKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(
            k.data(), eps.data(), gByNu.data(), divU.data(), dm.V.data(),
            co.C1, co.C2, co.C3, co.Cmu, bounded ? 1 : 0, SpE.data(), SuE.data(), nC);
    const scalar rE = deviceParallelScalarTransport(dm, halo, dbEps, faceCellsD, procStart, weightsD,
        phiInt, phiBnd, phiF, Deps_i, Deps_b, DepsGeo, SpE, SuE, eps, relaxEps, tol, maxIter, globalNCells, ones,
        isWallCell, eps0, relTolKE);
    detail::boundFieldKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(eps.data(), 1e-15, nC);
    if (resEps) *resEps = rE;

    // ---- k equation ----
    DeviceBuffer<scalar> DkCell(nC), Dk_i, Dk_b;
    { std::vector<scalar> d(nC); const std::vector<scalar> nh = nut.host(); for (int c=0;c<nC;++c) d[c]=nh[c]/co.sigmaK+nu; DkCell.copyFrom(d); }
    std::vector<DeviceBuffer<scalar>> DkGeo;
    faceGammaGeo(DkCell, Dk_i, Dk_b, DkGeo);
    DeviceBuffer<scalar> SpK(nC), SuK(nC);
    detail::kSourceKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(
        k.data(), eps.data(), G.data(), divU.data(), dm.V.data(), bounded ? 1 : 0, SpK.data(), SuK.data(), nC);
    DeviceBuffer<label> noWall; DeviceBuffer<scalar> noVal;
    const scalar rK = deviceParallelScalarTransport(dm, halo, dbK, faceCellsD, procStart, weightsD,
        phiInt, phiBnd, phiF, Dk_i, Dk_b, DkGeo, SpK, SuK, k, relaxK, tol, maxIter, globalNCells, ones,
        noWall, noVal, relTolKE);
    detail::boundFieldKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(k.data(), 1e-15, nC);
    if (resK) *resK = rK;

    // ---- correctNut (LOCAL: nut = Cmu*k^2/eps; realizableKE: rCmu*k^2/eps with the variable Cmu) ----
    if (co.realizable) deviceRealizableNut(rCmu, k, eps, nut);
    else               deviceNut(k, eps, nut, co);
}


// Distributed device kOmegaSST::correct() -- mirrors the single-GPU deviceKOmegaSSTCorrect (device_kepsilon.cu)
// exactly, reusing its exposed LOCAL kernels (S2, F1, F2, CDkOmega, blends, reactions, nutSST, wall omega) and
// the distributed scalar-transport core for the two solves. Processor-coupled inputs: gradU, grad(k), grad(omega)
// (each field's cut value from the halo before gaussGrad), and the F1-blended diffusivities DkEff/DomegaEff at
// cut faces (halo-interpolated). k/omega/nut updated in place. Same wall/setValues/geomD scaffolding as k-eps.
inline void parallelDeviceKOmegaSSTCorrect(
    const DeviceMesh& dm, DeviceHalo& halo, const DeviceWallData& wall,
    const DeviceVectorBoundary& dbU, const DeviceBoundary& dbK, const DeviceBoundary& dbOmega,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& k, DeviceBuffer<scalar>& omega, DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y,                       // cell wall distance
    const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const std::vector<DeviceBuffer<label>>& faceCellsD, const std::vector<label>& procStart,
    const std::vector<DeviceBuffer<scalar>>& weightsD, const std::vector<DeviceBuffer<scalar>>& geomD,
    const DeviceBuffer<label>& isWallCell,
    scalar nu, scalar relaxOmega, scalar relaxK, scalar tol, int maxIter, label globalNCells,
    const DeviceBuffer<scalar>& ones, bool bounded, const KOmegaSSTCoeffs& co,
    bool lm = false,                                     // kOmegaSSTLM: F1 override + k-production intermittency
    const DeviceBuffer<scalar>* gammaIntEff = nullptr,   // effective intermittency (nullptr -> plain kOmegaSST)
    scalar relTolKE = 0.0)
{
    constexpr int TPB = 128;
    const int nC = dm.nCells;
    auto nb = [&](int n){ return (n + TPB - 1) / TPB; };

    // halo-consistent scalar gradient: inject the field's cut value from the halo before gaussGrad
    auto haloGrad = [&](const DeviceBoundary& db, const DeviceBuffer<scalar>& fld,
                        DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz)
    {
        DeviceBuffer<scalar> bv;
        deviceBCValue(db, fld, bv);
        halo.exchange(fld.data());
        halo.scatterBoundaryValues(fld.data(), weightsD, procStart, bv.data());
        halo.waitExchange();
        deviceGaussGrad(dm, fld, bv, gx, gy, gz);
    };
    // F1-blended diffusivity DEff at internal/boundary/cut faces (like k-eps faceGammaGeo)
    auto faceGammaGeo = [&](const DeviceBuffer<scalar>& gammaCell, const DeviceBoundary& db,
                            DeviceBuffer<scalar>& gInt, DeviceBuffer<scalar>& gBnd,
                            std::vector<DeviceBuffer<scalar>>& gCoeffGeo)
    {
        deviceInterpolate(dm, gammaCell, gInt);
        deviceBCValue(db, gammaCell, gBnd);
        DeviceBuffer<scalar> gBndProc;
        deviceBCValue(db, gammaCell, gBndProc);
        halo.exchange(gammaCell.data());
        halo.scatterBoundaryValues(gammaCell.data(), weightsD, procStart, gBndProc.data());
        halo.waitExchange();
        const std::vector<scalar> gh = gBndProc.host();
        gCoeffGeo.clear(); gCoeffGeo.resize(procStart.size());
        for (std::size_t i = 0; i < procStart.size(); ++i)
        {
            const int n = static_cast<int>(halo.size(static_cast<int>(i)));
            if (n <= 0) continue;
            const std::vector<scalar> gd = geomD[i].host();
            std::vector<scalar> cg(n);
            for (int f = 0; f < n; ++f) cg[f] = gh[procStart[i] + f] * gd[f];
            gCoeffGeo[i].copyFrom(cg);
        }
    };

    // gradU (halo) -> GbyNu0, S2 ; production G = nut*GbyNu0 ; divU
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };
    DeviceBuffer<scalar> gradU(static_cast<std::size_t>(9) * nC);
    for (int i = 0; i < 3; ++i)
    {
        DeviceBuffer<scalar> gx, gy, gz;
        haloGrad(dbU.comp[i], *Uc[i], gx, gy, gz);
        cudaCheck(cudaMemcpyAsync(gradU.data()+(0*3+i)*(std::size_t)nC, gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "sstGU");
        cudaCheck(cudaMemcpyAsync(gradU.data()+(1*3+i)*(std::size_t)nC, gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "sstGU");
        cudaCheck(cudaMemcpyAsync(gradU.data()+(2*3+i)*(std::size_t)nC, gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "sstGU");
    }
    DeviceBuffer<scalar> GbyNu0; deviceGByNuFromGradU(gradU, nC, GbyNu0);
    DeviceBuffer<scalar> G;      deviceHadamard(G, nut, GbyNu0);
    DeviceBuffer<scalar> divU;   deviceDiv(dm, phiInt, phiBnd, divU);
    DeviceBuffer<scalar> S2;     deviceS2(gradU, nC, S2);

    // omega wall FIRST: omega0/G0, override omega & G at wall cells
    DeviceBuffer<scalar> omega0, G0;
    deviceWallOmegaG0(wall, k, Ux, Uy, Uz, nu, omega0, G0, co);
    detail::prodAndWallKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(
        nut.data(), GbyNu0.data(), isWallCell.data(), omega0.data(), G0.data(), G.data(), omega.data(), nC);
    // NB prodAndWallKernel sets G=nut*GbyNu0 for non-wall; SST needs G=nut*GbyNu0 too (GbyNu0 passed as gByNu). OK.

    // CDkOmega from grad(k), grad(omega) (halo-consistent); F1, F2
    DeviceBuffer<scalar> kgx, kgy, kgz, ogx, ogy, ogz;
    haloGrad(dbK, k, kgx, kgy, kgz);
    haloGrad(dbOmega, omega, ogx, ogy, ogz);
    DeviceBuffer<scalar> CD; deviceCDkOmega(kgx, kgy, kgz, ogx, ogy, ogz, omega, co.alphaOmega2, CD);
    DeviceBuffer<scalar> F1; deviceF1(k, omega, y, CD, nu, co, F1, lm);   // lm: F1 = max(F1, F3) near-wall override
    DeviceBuffer<scalar> F2; deviceF2(k, omega, y, nu, co, F2);

    // blends, limited production, diffusivities
    DeviceBuffer<scalar> gamma; deviceBlend(F1, co.gamma1, co.gamma2, gamma);
    DeviceBuffer<scalar> beta;  deviceBlend(F1, co.beta1,  co.beta2,  beta);
    DeviceBuffer<scalar> GbyNu0lim; deviceGbyNuLimit(GbyNu0, omega, F2, S2, co, GbyNu0lim);
    DeviceBuffer<scalar> DomegaEff; deviceDEff(F1, nut, co.alphaOmega1, co.alphaOmega2, nu, DomegaEff);
    DeviceBuffer<scalar> DkEff;     deviceDEff(F1, nut, co.alphaK1,     co.alphaK2,     nu, DkEff);

    // ---- omega equation (with the omega near-wall setValues constraint) ----
    {
        DeviceBuffer<scalar> gInt, gBnd; std::vector<DeviceBuffer<scalar>> gGeo;
        faceGammaGeo(DomegaEff, dbOmega, gInt, gBnd, gGeo);
        DeviceBuffer<scalar> Sp(std::vector<scalar>(nC,0.0)), Su(std::vector<scalar>(nC,0.0));
        deviceOmegaReaction(dm.V, gamma, beta, GbyNu0lim, F1, CD, omega, divU, Sp, Su);
        deviceParallelScalarTransport(dm, halo, dbOmega, faceCellsD, procStart, weightsD, phiInt, phiBnd, phiF,
            gInt, gBnd, gGeo, Sp, Su, omega, relaxOmega, tol, maxIter, globalNCells, ones, isWallCell, omega0, relTolKE);
        detail::boundFieldKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(omega.data(), 1e-15, nC);
    }
    // ---- k equation ----
    {
        DeviceBuffer<scalar> gInt, gBnd; std::vector<DeviceBuffer<scalar>> gGeo;
        faceGammaGeo(DkEff, dbK, gInt, gBnd, gGeo);
        DeviceBuffer<scalar> Sp(std::vector<scalar>(nC,0.0)), Su(std::vector<scalar>(nC,0.0));
        deviceKReactionSST(dm.V, k, omega, G, divU, co, Sp, Su, gammaIntEff ? gammaIntEff->data() : nullptr);   // lm: Pk*=gammaIntEff
        DeviceBuffer<label> noWall; DeviceBuffer<scalar> noVal;
        deviceParallelScalarTransport(dm, halo, dbK, faceCellsD, procStart, weightsD, phiInt, phiBnd, phiF,
            gInt, gBnd, gGeo, Sp, Su, k, relaxK, tol, maxIter, globalNCells, ones, noWall, noVal, relTolKE);
        detail::boundFieldKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(k.data(), 1e-15, nC);
    }
    // ---- correctNut (Bradshaw limiter) ----
    deviceNutSST(k, omega, F2, S2, co, nut);
}

// Distributed kOmegaSSTLM::correct() -- the two Langtry-Menter transition transport equations (ReThetat, gammaInt)
// solved via the distributed scalar-transport core with the exposed cell-local LM source kernels (gradU is the only
// halo-coupled input), then gammaIntEff, then parallelDeviceKOmegaSSTCorrect with lm=true + gammaIntEff. Mirrors the
// single-GPU deviceKOmegaSSTLMCorrect. ReThetat/gammaInt are extra device-resident fields owned by the solver. As in
// realizableKE, the bounded -Sp(div(phi)) term is folded into Sp here (the distributed transport does not apply it).
inline void parallelDeviceKOmegaSSTLMCorrect(
    const DeviceMesh& dm, DeviceHalo& halo, const DeviceWallData& wall,
    const DeviceVectorBoundary& dbU, const DeviceBoundary& dbK, const DeviceBoundary& dbOmega,
    const DeviceBoundary& dbReThetat, const DeviceBoundary& dbGammaInt,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& k, DeviceBuffer<scalar>& omega, DeviceBuffer<scalar>& nut,
    DeviceBuffer<scalar>& ReThetat, DeviceBuffer<scalar>& gammaInt,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const std::vector<DeviceBuffer<label>>& faceCellsD, const std::vector<label>& procStart,
    const std::vector<DeviceBuffer<scalar>>& weightsD, const std::vector<DeviceBuffer<scalar>>& geomD,
    const DeviceBuffer<label>& isWallCell,
    scalar nu, scalar relaxOmega, scalar relaxK, scalar tol, int maxIter, label globalNCells,
    const DeviceBuffer<scalar>& ones, bool bounded, const KOmegaSSTCoeffs& co, scalar relTolKE = 0.0)
{
    constexpr int TPB = 128;
    const int nC = dm.nCells;
    auto nb = [&](int n){ return (n + TPB - 1) / TPB; };

    // halo-consistent gradU tensor (9*nC, column i = gaussGrad(U_i)); processor cut value from the halo
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };
    DeviceBuffer<scalar> gradU(static_cast<std::size_t>(9) * nC);
    for (int i = 0; i < 3; ++i)
    {
        DeviceBuffer<scalar> ub;
        deviceBCValue(dbU.comp[i], *Uc[i], ub);
        halo.exchange(Uc[i]->data());
        halo.scatterBoundaryValues(Uc[i]->data(), weightsD, procStart, ub.data());
        halo.waitExchange();
        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, *Uc[i], ub, gx, gy, gz);
        cudaCheck(cudaMemcpyAsync(gradU.data()+(0*3+i)*static_cast<std::size_t>(nC), gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "lmGradU x");
        cudaCheck(cudaMemcpyAsync(gradU.data()+(1*3+i)*static_cast<std::size_t>(nC), gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "lmGradU y");
        cudaCheck(cudaMemcpyAsync(gradU.data()+(2*3+i)*static_cast<std::size_t>(nC), gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "lmGradU z");
    }
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);

    // gammaCell -> face diffusivity at internal/boundary/processor faces (same helper as k-eps/SST faceGammaGeo)
    auto faceGammaGeo = [&](const DeviceBuffer<scalar>& gammaCell, const DeviceBoundary& db,
                            DeviceBuffer<scalar>& gInt, DeviceBuffer<scalar>& gBnd, std::vector<DeviceBuffer<scalar>>& gGeo)
    {
        deviceInterpolate(dm, gammaCell, gInt);
        deviceBCValue(db, gammaCell, gBnd);
        DeviceBuffer<scalar> gBndProc;
        deviceBCValue(db, gammaCell, gBndProc);
        halo.exchange(gammaCell.data());
        halo.scatterBoundaryValues(gammaCell.data(), weightsD, procStart, gBndProc.data());
        halo.waitExchange();
        const std::vector<scalar> gh = gBndProc.host();
        gGeo.clear(); gGeo.resize(procStart.size());
        for (std::size_t i = 0; i < procStart.size(); ++i)
        {
            const int n = static_cast<int>(halo.size(static_cast<int>(i)));
            if (n <= 0) continue;
            const std::vector<scalar> gd = geomD[i].host();
            std::vector<scalar> cg(n);
            for (int f = 0; f < n; ++f) cg[f] = gh[procStart[i] + f] * gd[f];
            gGeo[i].copyFrom(cg);
        }
    };
    DeviceBuffer<label> noWall; DeviceBuffer<scalar> noVal;
    // fold in the bounded -Sp(div(phi)) term (the distributed transport does not; the LM prep kernels do not either)
    auto addBounded = [&](DeviceBuffer<scalar>& Sp){ if (bounded){ DeviceBuffer<scalar> vDiv; deviceHadamard(vDiv, divU, dm.V); deviceAxpy(-1.0, vDiv, Sp); } };

    // ---- ReThetat equation:  DReThetatEff = sigmaThetat*(nut+nu); reaction = Pthetat*ReThetat0 - Sp(Pthetat) ----
    DeviceBuffer<scalar> Fth, spR, suR;
    deviceLMReThetatPrep(dm, gradU, Ux, Uy, Uz, k, omega, y, ReThetat, gammaInt, nu, Fth, spR, suR);
    DeviceBuffer<scalar> DRe;
    deviceLMReDiff(nut, nu, DRe);
    {
        DeviceBuffer<scalar> gI, gB; std::vector<DeviceBuffer<scalar>> gG;
        faceGammaGeo(DRe, dbReThetat, gI, gB, gG);
        DeviceBuffer<scalar> Sp(std::vector<scalar>(nC, 0.0)), Su(std::vector<scalar>(nC, 0.0));
        deviceLMAddReaction(dm, spR, suR, Sp, Su);
        addBounded(Sp);
        deviceParallelScalarTransport(dm, halo, dbReThetat, faceCellsD, procStart, weightsD, phiInt, phiBnd, phiF,
            gI, gB, gG, Sp, Su, ReThetat, relaxK, tol, maxIter, globalNCells, ones, noWall, noVal, relTolKE);
        detail::boundFieldKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(ReThetat.data(), 0.0, nC);
    }

    // ---- gammaInt equation:  DgammaIntEff = nut+nu; reaction = Pgamma+Egamma - Sp(ce1*Pgamma+ce2*Egamma) ----
    DeviceBuffer<scalar> spG, suG;
    deviceLMGammaPrep(dm, gradU, Ux, Uy, Uz, k, omega, y, ReThetat, gammaInt, nu, spG, suG);
    DeviceBuffer<scalar> DgI(nC);
    { std::vector<scalar> d(nC); const std::vector<scalar> nh = nut.host(); for (int c=0;c<nC;++c) d[c]=nh[c]+nu; DgI.copyFrom(d); }   // nut/1 + nu
    {
        DeviceBuffer<scalar> gI, gB; std::vector<DeviceBuffer<scalar>> gG;
        faceGammaGeo(DgI, dbGammaInt, gI, gB, gG);
        DeviceBuffer<scalar> Sp(std::vector<scalar>(nC, 0.0)), Su(std::vector<scalar>(nC, 0.0));
        deviceLMAddReaction(dm, spG, suG, Sp, Su);
        addBounded(Sp);
        deviceParallelScalarTransport(dm, halo, dbGammaInt, faceCellsD, procStart, weightsD, phiInt, phiBnd, phiF,
            gI, gB, gG, Sp, Su, gammaInt, relaxK, tol, maxIter, globalNCells, ones, noWall, noVal, relTolKE);
        detail::boundFieldKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(gammaInt.data(), 0.0, nC);
    }

    // ---- effective intermittency, then the SST correct with the LM F1 override + Pk modulation ----
    DeviceBuffer<scalar> gammaIntEff;
    deviceLMGammaEff(dm, gradU, Ux, Uy, Uz, k, omega, y, ReThetat, gammaInt, Fth, nu, gammaIntEff);
    parallelDeviceKOmegaSSTCorrect(dm, halo, wall, dbU, dbK, dbOmega, Ux, Uy, Uz, k, omega, nut, y,
        phiInt, phiBnd, phiF, faceCellsD, procStart, weightsD, geomD, isWallCell,
        nu, relaxOmega, relaxK, tol, maxIter, globalNCells, ones, bounded, co, /*lm*/ true, &gammaIntEff);
}

// Distributed SpalartAllmaras::correct() -- the single nuTilda transport equation via the distributed scalar-
// transport core + the exposed cell-local SA source kernels (gradU and grad(nuTilda) are the halo-coupled inputs).
// nut = nuTilda*fv1. Mirrors the single-GPU deviceSpalartAllmarasCorrect. The bounded -Sp(div phi) term is folded
// into Sp here (the distributed transport does not apply it), same as k-eps/SST. One turbulence field, no wall
// setValues constraint (the nuTilda wall BC is fixedValue 0 via dbNuTilda).
inline void parallelDeviceSpalartAllmarasCorrect(
    const DeviceMesh& dm, DeviceHalo& halo,
    const DeviceVectorBoundary& dbU, const DeviceBoundary& dbNuTilda,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& nuTilda, DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const std::vector<DeviceBuffer<label>>& faceCellsD, const std::vector<label>& procStart,
    const std::vector<DeviceBuffer<scalar>>& weightsD, const std::vector<DeviceBuffer<scalar>>& geomD,
    scalar nu, scalar relax, scalar tol, int maxIter, label globalNCells,
    const DeviceBuffer<scalar>& ones, bool bounded, const SpalartAllmarasCoeffs& co, scalar relTolKE = 0.0)
{
    constexpr int TPB = 128;
    const int nC = dm.nCells;
    auto nb = [&](int n){ return (n + TPB - 1) / TPB; };

    // halo-consistent gradU tensor (9*nC) for vorticity/production
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };
    DeviceBuffer<scalar> gradU(static_cast<std::size_t>(9) * nC);
    for (int i = 0; i < 3; ++i)
    {
        DeviceBuffer<scalar> ub;
        deviceBCValue(dbU.comp[i], *Uc[i], ub);
        halo.exchange(Uc[i]->data());
        halo.scatterBoundaryValues(Uc[i]->data(), weightsD, procStart, ub.data());
        halo.waitExchange();
        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, *Uc[i], ub, gx, gy, gz);
        cudaCheck(cudaMemcpyAsync(gradU.data()+(0*3+i)*static_cast<std::size_t>(nC), gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "saGradU x");
        cudaCheck(cudaMemcpyAsync(gradU.data()+(1*3+i)*static_cast<std::size_t>(nC), gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "saGradU y");
        cudaCheck(cudaMemcpyAsync(gradU.data()+(2*3+i)*static_cast<std::size_t>(nC), gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "saGradU z");
    }
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);

    DeviceBuffer<scalar> Stilda; deviceSAStilda(dm, gradU, nuTilda, y, nu, co, Stilda);
    DeviceBuffer<scalar> fw;     deviceSAFw(dm, nuTilda, Stilda, y, co, fw);

    // grad(nuTilda) halo-consistent -> |grad nuTilda|^2 (the Cb2 term)
    DeviceBuffer<scalar> nbv;
    deviceBCValue(dbNuTilda, nuTilda, nbv);
    halo.exchange(nuTilda.data());
    halo.scatterBoundaryValues(nuTilda.data(), weightsD, procStart, nbv.data());
    halo.waitExchange();
    DeviceBuffer<scalar> gnx, gny, gnz;
    deviceGaussGrad(dm, nuTilda, nbv, gnx, gny, gnz);
    DeviceBuffer<scalar> gradNt2; deviceSAMagSqr(dm, gnx, gny, gnz, gradNt2);

    DeviceBuffer<scalar> D; deviceSADEff(dm, nuTilda, nu, co.sigmaNut, D);

    // gammaCell -> face diffusivity at internal/boundary/processor faces (same helper as k-eps/SST)
    auto faceGammaGeo = [&](const DeviceBuffer<scalar>& gammaCell, const DeviceBoundary& db,
                            DeviceBuffer<scalar>& gInt, DeviceBuffer<scalar>& gBnd, std::vector<DeviceBuffer<scalar>>& gGeo)
    {
        deviceInterpolate(dm, gammaCell, gInt);
        deviceBCValue(db, gammaCell, gBnd);
        DeviceBuffer<scalar> gBndProc;
        deviceBCValue(db, gammaCell, gBndProc);
        halo.exchange(gammaCell.data());
        halo.scatterBoundaryValues(gammaCell.data(), weightsD, procStart, gBndProc.data());
        halo.waitExchange();
        const std::vector<scalar> gh = gBndProc.host();
        gGeo.clear(); gGeo.resize(procStart.size());
        for (std::size_t i = 0; i < procStart.size(); ++i)
        {
            const int n = static_cast<int>(halo.size(static_cast<int>(i)));
            if (n <= 0) continue;
            const std::vector<scalar> gd = geomD[i].host();
            std::vector<scalar> cg(n);
            for (int f = 0; f < n; ++f) cg[f] = gh[procStart[i] + f] * gd[f];
            gGeo[i].copyFrom(cg);
        }
    };

    DeviceBuffer<scalar> gI, gB; std::vector<DeviceBuffer<scalar>> gG;
    faceGammaGeo(D, dbNuTilda, gI, gB, gG);
    DeviceBuffer<scalar> Sp(std::vector<scalar>(nC, 0.0)), Su(std::vector<scalar>(nC, 0.0));
    deviceSAReaction(dm, nuTilda, Stilda, fw, y, gradNt2, co, Sp, Su);   // += ; Sp/Su pre-zeroed above
    if (bounded) { DeviceBuffer<scalar> vDiv; deviceHadamard(vDiv, divU, dm.V); deviceAxpy(-1.0, vDiv, Sp); }
    DeviceBuffer<label> noWall; DeviceBuffer<scalar> noVal;
    deviceParallelScalarTransport(dm, halo, dbNuTilda, faceCellsD, procStart, weightsD, phiInt, phiBnd, phiF,
        gI, gB, gG, Sp, Su, nuTilda, relax, tol, maxIter, globalNCells, ones, noWall, noVal, relTolKE);
    detail::boundFieldKernel<<<nb(nC), TPB, 0, cudaStreamPerThread>>>(nuTilda.data(), 1e-15, nC);

    deviceNutSA(nuTilda, nu, co.Cv1, nut);   // nut = nuTilda*fv1
}

} // namespace brae
