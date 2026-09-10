// cf GPU offload, cyclicAMI weighted-stencil coupling kernels. See device_ami.cuh.
#include "device_ami.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {
namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__device__
void amiInterpKernel(
    int n,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ psi,
    scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    scalar s = 0;
    for (label k = off[i]; k < off[i+1]; ++k)
        s += w[k] * psi[nbr[k]];   // sum_k weight*psi[nbrCell]
    out[i] = s;
}


__device__
void amiInterpVecKernel(
    int n,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ fT,
    int rotational,
    scalar* __restrict__ oX,
    scalar* __restrict__ oY,
    scalar* __restrict__ oZ)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    scalar sx = 0, sy = 0, sz = 0;
    // sum_k w * (forwardT . U[nbrCell]) ; transform BEFORE the weighted sum
    for (label k = off[i]; k < off[i+1]; ++k)
    {
        const int c = nbr[k];
        const scalar wk = w[k];
        scalar vx = Ux[c], vy = Uy[c], vz = Uz[c];
        if (rotational)   // forwardT for THIS source face i (per-face packed)
        {
            const scalar rx = fT[(0*3+0)*n+i]*vx + fT[(0*3+1)*n+i]*vy + fT[(0*3+2)*n+i]*vz;
            const scalar ry = fT[(1*3+0)*n+i]*vx + fT[(1*3+1)*n+i]*vy + fT[(1*3+2)*n+i]*vz;
            const scalar rz = fT[(2*3+0)*n+i]*vx + fT[(2*3+1)*n+i]*vy + fT[(2*3+2)*n+i]*vz;
            vx = rx; vy = ry; vz = rz;
        }
        sx += wk*vx; sy += wk*vy; sz += wk*vz;
    }
    oX[i] = sx; oY[i] = sy; oZ[i] = sz;
}


__device__
void amiAmulKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ psi,
    scalar* __restrict__ Apsi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    scalar s = 0;
    for (label k = off[i]; k < off[i+1]; ++k)
        s += w[k] * psi[nbr[k]];   // interpolate-to-source
    atomicAdd(&Apsi[own[i]], ifc[i] * s);   // scale by the interface coefficient
}
} // namespace


void deviceAmiAmul(const DeviceAMI& ami, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi)
{
    if (ami.n == 0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const label* off = ami.off.data();
        const label* nbr = ami.nbrCell.data(); const scalar* w = ami.weight.data(); const scalar* ifc = ami.ifCoeff.data();
        const scalar* psid = psi.data(); scalar* Apsid = Apsi.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiAmulKernel(n, own, off, nbr, w, ifc, psid, Apsid); });
    }
    cudaCheck(cudaGetLastError(), "amiAmul");
}

// Like deviceAmiAmul but with an EXPLICIT off-diagonal coefficient: the ROTATIONAL momentum matvec must use the
// per-component diag-scaled ifCoeffC[kk] (= ifCoeff*forwardT[kk][kk]), NOT the base ifCoeff, to be consistent with
// the deferred-rotation source. Translational / scalar (pressure): pass ami.ifCoeff (== deviceAmiAmul).
void deviceAmiAmulCoeff(const DeviceAMI& ami, const DeviceBuffer<scalar>& coeff,
                        const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi)
{
    if (ami.n == 0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const label* off = ami.off.data();
        const label* nbr = ami.nbrCell.data(); const scalar* w = ami.weight.data(); const scalar* coeffD = coeff.data();
        const scalar* psid = psi.data(); scalar* Apsid = Apsi.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiAmulKernel(n, own, off, nbr, w, coeffD, psid, Apsid); });
    }
    cudaCheck(cudaGetLastError(), "amiAmulCoeff");
}


namespace {
__device__
void amiMomKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ nu,
    const scalar* __restrict__ nuN,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ phi,
    scalar* __restrict__ ifc,
    scalar* __restrict__ diag)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar lap = (w[i]*nu[o] + (1.0-w[i])*nuN[i]) * dc[i] * magSf[i];
    const scalar p = phi[i];
    ifc[i] = -lap + (p<0.0?p:0.0);
    atomicAdd(&diag[o], lap + (p>0.0?p:0.0));
}


__device__
void amiLaplKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ ga,
    const scalar* __restrict__ gaN,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    scalar* __restrict__ ifc,
    scalar* __restrict__ diag,
    int addToDiag)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar c = (w[i]*ga[o] + (1.0-w[i])*gaN[i]) * dc[i] * magSf[i];
    ifc[i] = c;
    if (addToDiag) atomicAdd(&diag[o], -c);
}


__device__
void amiOffSumKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ wsum,
    scalar* __restrict__ sumOff)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    atomicAdd(&sumOff[own[i]], fabs(ifc[i])*wsum[i]);
}


__device__
void amiAddHKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ Un,
    const scalar* __restrict__ V,
    scalar* __restrict__ H)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    atomicAdd(&H[o], -ifc[i]*Un[i]/V[o]);
}


__device__
void amiFluxKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Hx,
    const scalar* __restrict__ Hy,
    const scalar* __restrict__ Hz,
    const scalar* __restrict__ Hnx,
    const scalar* __restrict__ Hny,
    const scalar* __restrict__ Hnz,
    const scalar* __restrict__ sfx,
    const scalar* __restrict__ sfy,
    const scalar* __restrict__ sfz,
    scalar* __restrict__ phi)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar wj=w[i], wn=1.0-wj;
    phi[i] = (wj*Hx[o]+wn*Hnx[i])*sfx[i] + (wj*Hy[o]+wn*Hny[i])*sfy[i] + (wj*Hz[o]+wn*Hnz[i])*sfz[i];
}


__device__
void amiDivAddKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ V,
    scalar* __restrict__ div)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    atomicAdd(&div[own[i]], phi[i]/V[own[i]]);
}


__device__
void amiGradAddKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ w,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ pN,
    const scalar* __restrict__ sfx,
    const scalar* __restrict__ sfy,
    const scalar* __restrict__ sfz,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar fv = (w[i]*psi[o] + (1.0-w[i])*pN[i]) / V[o];
    atomicAdd(&gx[o], sfx[i]*fv);
    atomicAdd(&gy[o], sfy[i]*fv);
    atomicAdd(&gz[o], sfz[i]*fv);
}


__device__
void amiFluxCorrKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ pN,
    const scalar* __restrict__ p,
    scalar* __restrict__ phi)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    phi[i] -= ifc[i]*(pN[i] - p[own[i]]);
}
} // namespace


void deviceAmiAssembleMomentum(DeviceAMI& ami, const DeviceBuffer<scalar>& nuEffCell, DeviceBuffer<scalar>& diag)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> nuN;
    deviceAmiInterpolate(ami, nuEffCell, nuN);
    {
        const int n = ami.n; const label* own = ami.ownCell.data();
        const scalar* nu = nuEffCell.data(); const scalar* nuNd = nuN.data(); const scalar* dc = ami.deltaCoeffs.data();
        const scalar* w = ami.weights.data(); const scalar* magSf = ami.magSf.data(); const scalar* phi = ami.phi.data();
        scalar* ifc = ami.ifCoeff.data(); scalar* diagd = diag.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiMomKernel(n, own, nu, nuNd, dc, w, magSf, phi, ifc, diagd); });
    }
    cudaCheck(cudaGetLastError(),"amiMom");
}


void deviceAmiAssembleLaplacian(DeviceAMI& ami, const DeviceBuffer<scalar>& gammaCell, DeviceBuffer<scalar>& diag, bool addToDiag)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> gN;
    deviceAmiInterpolate(ami, gammaCell, gN);
    {
        const int n = ami.n; const label* own = ami.ownCell.data();
        const scalar* ga = gammaCell.data(); const scalar* gaN = gN.data(); const scalar* dc = ami.deltaCoeffs.data();
        const scalar* w = ami.weights.data(); const scalar* magSf = ami.magSf.data();
        scalar* ifc = ami.ifCoeff.data(); scalar* diagd = diag.data(); const int addD = addToDiag ? 1 : 0;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiLaplKernel(n, own, ga, gaN, dc, w, magSf, ifc, diagd, addD); });
    }
    cudaCheck(cudaGetLastError(),"amiLapl");
}


void deviceAmiOffDiagSum(const DeviceAMI& ami, DeviceBuffer<scalar>& sumOff)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* ifc = ami.ifCoeff.data();
        const scalar* wsum = ami.weightsSum.data(); scalar* sumOffd = sumOff.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiOffSumKernel(n, own, ifc, wsum, sumOffd); });
    }
    cudaCheck(cudaGetLastError(),"amiOffSum");
}


void deviceAmiAddH(const DeviceAMI& ami, const DeviceBuffer<scalar>& UkNbr, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& H)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* ifc = ami.ifCoeff.data();
        const scalar* UkNbrD = UkNbr.data(); const scalar* Vd = V.data(); scalar* Hd = H.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiAddHKernel(n, own, ifc, UkNbrD, Vd, Hd); });
    }
    cudaCheck(cudaGetLastError(),"amiAddH");
}


void deviceAmiFlux(DeviceAMI& ami, const DeviceBuffer<scalar>& Hx, const DeviceBuffer<scalar>& Hy, const DeviceBuffer<scalar>& Hz)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> Hnx,Hny,Hnz;
    deviceAmiInterpolateVec(ami, Hx,Hy,Hz, Hnx,Hny,Hnz);
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* w = ami.weights.data();
        const scalar* Hxd = Hx.data(); const scalar* Hyd = Hy.data(); const scalar* Hzd = Hz.data();
        const scalar* Hnxd = Hnx.data(); const scalar* Hnyd = Hny.data(); const scalar* Hnzd = Hnz.data();
        const scalar* sfx = ami.Sfx.data(); const scalar* sfy = ami.Sfy.data(); const scalar* sfz = ami.Sfz.data();
        scalar* phid = ami.phi.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiFluxKernel(n, own, w, Hxd, Hyd, Hzd, Hnxd, Hnyd, Hnzd, sfx, sfy, sfz, phid); });
    }
    cudaCheck(cudaGetLastError(),"amiFlux");
}


void deviceAmiAddDiv(const DeviceAMI& ami, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* phi = ami.phi.data();
        const scalar* Vd = V.data(); scalar* divd = div.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiDivAddKernel(n, own, phi, Vd, divd); });
    }
    cudaCheck(cudaGetLastError(),"amiDiv");
}


namespace {
__device__
void amiZeroWallKernel(int n, const label* __restrict__ own, const label* __restrict__ isW, scalar* __restrict__ ifc)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n && isW[own[i]]) ifc[i]=0.0;
}
} // namespace


// epsilon setValues: a wall cell's eps is fixed (eps0), so the interface off-diagonal must not perturb it (the
// internal-face setValues already zeros internal off-diagonals; do the same for the AMI interface coupling).
void deviceAmiZeroWallIfCoeff(DeviceAMI& ami, const DeviceBuffer<label>& isWallCell)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const label* isW = isWallCell.data();
        scalar* ifc = ami.ifCoeff.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiZeroWallKernel(n, own, isW, ifc); });
    }
    cudaCheck(cudaGetLastError(),"amiZeroWall");
}


namespace {
// face value of a CELL field on an interface face: fvc::interpolate on a coupled patch, i.e.
//   w*patchInternalField + (1-w)*patchNeighbourField
__device__
void amiFaceValueKernel(int n, const label* __restrict__ own, const scalar* __restrict__ w,
                        const scalar* __restrict__ cell, const scalar* __restrict__ nbrInterp,
                        scalar* __restrict__ out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = w[i]*cell[own[i]] + (scalar(1) - w[i])*nbrInterp[i];
}
}   // namespace

void deviceAmiFaceValue(const DeviceAMI& ami, const DeviceBuffer<scalar>& cell, DeviceBuffer<scalar>& out)
{
    out.resize(ami.n);
    if (ami.n == 0) return;
    DeviceBuffer<scalar> nbr;
    deviceAmiInterpolate(ami, cell, nbr);
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* w = ami.weights.data();
        const scalar* celld = cell.data(); const scalar* nbrd = nbr.data(); scalar* outd = out.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiFaceValueKernel(n, own, w, celld, nbrd, outd); });
    }
    cudaCheck(cudaGetLastError(), "amiFaceValue");
}


void deviceAmiAddGrad(
    const DeviceAMI& ami,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> pN;
    deviceAmiInterpolate(ami, psi, pN);
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* w = ami.weights.data();
        const scalar* psid = psi.data(); const scalar* pNd = pN.data();
        const scalar* sfx = ami.Sfx.data(); const scalar* sfy = ami.Sfy.data(); const scalar* sfz = ami.Sfz.data();
        const scalar* Vd = V.data(); scalar* gxd = gx.data(); scalar* gyd = gy.data(); scalar* gzd = gz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiGradAddKernel(n, own, w, psid, pNd, sfx, sfy, sfz, Vd, gxd, gyd, gzd); });
    }
    cudaCheck(cudaGetLastError(),"amiGrad");
}


namespace {
__device__
void amiTensorDivKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ wf,
    const scalar* __restrict__ sfx,
    const scalar* __restrict__ sfy,
    const scalar* __restrict__ sfz,
    const scalar* __restrict__ sigmaC,
    int nC,
    const scalar* __restrict__ fT,
    int rotational,
    scalar* __restrict__ dX,
    scalar* __restrict__ dY,
    scalar* __restrict__ dZ)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    scalar sn[9];
    for (int q=0;q<9;++q)
        sn[q]=0.0;
    scalar R[9];
    if (rotational)
        for (int q=0;q<9;++q)
            R[q]=fT[q*n+i];
    // interp neighbour sigma (rotated for rotational)
    for (label k=off[i]; k<off[i+1]; ++k)
    {
        const int nb=nbr[k];
        const scalar wk=w[k];
        scalar s[9];
        for (int q=0;q<9;++q)
            s[q]=sigmaC[q*nC+nb];
        if (rotational)   // sigma' = R*sigma*R^T
        {
            scalar M[9];
            for (int a=0;a<3;++a)
                for (int l=0;l<3;++l)
                {
                    scalar t=0;
                    for (int c=0;c<3;++c)
                        t+=R[3*a+c]*s[3*c+l];
                    M[3*a+l]=t;
                }
            for (int a=0;a<3;++a)
                for (int b=0;b<3;++b)
                {
                    scalar t=0;
                    for (int l=0;l<3;++l)
                        t+=M[3*a+l]*R[3*b+l];
                    s[3*a+b]=t;
                }
        }
        for (int q=0;q<9;++q)
            sn[q] += wk*s[q];
    }
    const scalar w0=wf[i], w1=1.0-w0;
    scalar d[3];
    for (int jc=0;jc<3;++jc)
    {
        const scalar s0=w0*sigmaC[(0*3+jc)*nC+o]+w1*sn[0*3+jc];
        const scalar s1=w0*sigmaC[(1*3+jc)*nC+o]+w1*sn[1*3+jc];
        const scalar s2=w0*sigmaC[(2*3+jc)*nC+o]+w1*sn[2*3+jc];
        d[jc]=sfx[i]*s0 + sfy[i]*s1 + sfz[i]*s2;
    }
    atomicAdd(&dX[o],d[0]);
    atomicAdd(&dY[o],d[1]);
    atomicAdd(&dZ[o],d[2]);
}
} // namespace
namespace {
__device__
void amiScaleImplicitKernel(
    int n,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ fT,
    scalar* __restrict__ ic0,
    scalar* __restrict__ ic1,
    scalar* __restrict__ ic2)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    ic0[i]=ifc[i]*fT[0*n+i];   // *forwardT[kk][kk]
    ic1[i]=ifc[i]*fT[4*n+i];
    ic2[i]=ifc[i]*fT[8*n+i];
}


__device__
void amiDeferredRotKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ fT,
    int comp,
    const scalar* __restrict__ uiX,
    const scalar* __restrict__ uiY,
    const scalar* __restrict__ uiZ,
    scalar* __restrict__ src)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const scalar ux=uiX[i], uy=uiY[i], uz=uiZ[i];
    const scalar rComp = fT[(3*comp+0)*n+i]*ux + fT[(3*comp+1)*n+i]*uy + fT[(3*comp+2)*n+i]*uz;  // (forwardT.U_interp)[comp]
    const scalar diag  = fT[(3*comp+comp)*n+i];
    const scalar uComp = (comp==0)?ux:(comp==1)?uy:uz;
    atomicAdd(&src[own[i]], -ifc[i]*(rComp - diag*uComp));
}


__device__
void amiGradRotKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Uown,
    const scalar* __restrict__ UNbr,
    const scalar* __restrict__ sfx,
    const scalar* __restrict__ sfy,
    const scalar* __restrict__ sfz,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar fv = (w[i]*Uown[o] + (1.0-w[i])*UNbr[i]) / V[o];
    atomicAdd(&gx[o], sfx[i]*fv);
    atomicAdd(&gy[o], sfy[i]*fv);
    atomicAdd(&gz[o], sfz[i]*fv);
}
} // namespace


void deviceAmiScaleImplicit(DeviceAMI& ami)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const scalar* ifc = ami.ifCoeff.data(); const scalar* fT = ami.fT.data();
        scalar* ic0 = ami.ifCoeffC[0].data(); scalar* ic1 = ami.ifCoeffC[1].data(); scalar* ic2 = ami.ifCoeffC[2].data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiScaleImplicitKernel(n, ifc, fT, ic0, ic1, ic2); });
    }
    cudaCheck(cudaGetLastError(),"amiScaleImplicit");
}


void deviceAmiAddDeferredRot(
    const DeviceAMI& ami,
    const DeviceBuffer<scalar>& uiX,
    const DeviceBuffer<scalar>& uiY,
    const DeviceBuffer<scalar>& uiZ,
    int comp,
    DeviceBuffer<scalar>& src)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* ifc = ami.ifCoeff.data();
        const scalar* fT = ami.fT.data(); const int compD = comp;
        const scalar* uiXd = uiX.data(); const scalar* uiYd = uiY.data(); const scalar* uiZd = uiZ.data();
        scalar* srcd = src.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiDeferredRotKernel(n, own, ifc, fT, compD, uiXd, uiYd, uiZd, srcd); });
    }
    cudaCheck(cudaGetLastError(),"amiDeferredRot");
}


void deviceAmiAddGradRot(
    const DeviceAMI& ami,
    const DeviceBuffer<scalar>& Uown,
    const DeviceBuffer<scalar>& UNbr,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* w = ami.weights.data();
        const scalar* Uownd = Uown.data(); const scalar* UNbrd = UNbr.data();
        const scalar* sfx = ami.Sfx.data(); const scalar* sfy = ami.Sfy.data(); const scalar* sfz = ami.Sfz.data();
        const scalar* Vd = V.data(); scalar* gxd = gx.data(); scalar* gyd = gy.data(); scalar* gzd = gz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiGradRotKernel(n, own, w, Uownd, UNbrd, sfx, sfy, sfz, Vd, gxd, gyd, gzd); });
    }
    cudaCheck(cudaGetLastError(),"amiGradRot");
}


void deviceAmiAddTensorDiv(
    const DeviceAMI& ami,
    const DeviceBuffer<scalar>& sigmaC,
    int nC,
    DeviceBuffer<scalar>& srcX,
    DeviceBuffer<scalar>& srcY,
    DeviceBuffer<scalar>& srcZ)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const label* off = ami.off.data();
        const label* nbr = ami.nbrCell.data(); const scalar* w = ami.weight.data(); const scalar* wf = ami.weights.data();
        const scalar* sfx = ami.Sfx.data(); const scalar* sfy = ami.Sfy.data(); const scalar* sfz = ami.Sfz.data();
        const scalar* sigmaCd = sigmaC.data(); const int nCd = nC;
        const scalar* fT = ami.rotational ? ami.fT.data() : nullptr; const int rot = ami.rotational ? 1 : 0;
        scalar* dXd = srcX.data(); scalar* dYd = srcY.data(); scalar* dZd = srcZ.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiTensorDivKernel(n, own, off, nbr, w, wf, sfx, sfy, sfz, sigmaCd, nCd, fT, rot, dXd, dYd, dZd); });
    }
    cudaCheck(cudaGetLastError(),"amiTensorDiv");
}


void deviceAmiCorrectFlux(DeviceAMI& ami, const DeviceBuffer<scalar>& p)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> pN;
    deviceAmiInterpolate(ami, p, pN);
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* ifc = ami.ifCoeff.data();
        const scalar* pNd = pN.data(); const scalar* pd = p.data(); scalar* phid = ami.phi.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiFluxCorrKernel(n, own, ifc, pNd, pd, phid); });
    }
    cudaCheck(cudaGetLastError(),"amiFluxCorr");
}


void deviceAmiInterpolate(const DeviceAMI& ami, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& out)
{
    if (ami.n == 0) return;
    out.resize(ami.n);
    {
        const int n = ami.n; const label* off = ami.off.data(); const label* nbr = ami.nbrCell.data();
        const scalar* w = ami.weight.data(); const scalar* psid = psi.data(); scalar* outd = out.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiInterpKernel(n, off, nbr, w, psid, outd); });
    }
    cudaCheck(cudaGetLastError(), "amiInterp");
}


void deviceAmiInterpolateVec(
    const DeviceAMI& ami,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& oX,
    DeviceBuffer<scalar>& oY,
    DeviceBuffer<scalar>& oZ)
{
    if (ami.n == 0) return;
    oX.resize(ami.n); oY.resize(ami.n); oZ.resize(ami.n);
    {
        const int n = ami.n; const label* off = ami.off.data(); const label* nbr = ami.nbrCell.data();
        const scalar* w = ami.weight.data();
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        const scalar* fT = ami.rotational ? ami.fT.data() : nullptr; const int rot = ami.rotational ? 1 : 0;
        scalar* oXd = oX.data(); scalar* oYd = oY.data(); scalar* oZd = oZ.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiInterpVecKernel(n, off, nbr, w, Uxd, Uyd, Uzd, fT, rot, oXd, oYd, oZd); });
    }
    cudaCheck(cudaGetLastError(), "amiInterpVec");
}


namespace {
__device__
void amiLinUpwindKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ gx0,
    const scalar* __restrict__ gy0,
    const scalar* __restrict__ gz0,
    const scalar* __restrict__ gx1,
    const scalar* __restrict__ gy1,
    const scalar* __restrict__ gz1,
    const scalar* __restrict__ gx2,
    const scalar* __restrict__ gy2,
    const scalar* __restrict__ gz2,
    const scalar* __restrict__ dox,
    const scalar* __restrict__ doy,
    const scalar* __restrict__ doz,
    const scalar* __restrict__ dnx,
    const scalar* __restrict__ dny,
    const scalar* __restrict__ dnz,
    const scalar* __restrict__ fT,
    int rotational,
    int comp,
    scalar* __restrict__ corr)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar pf=phi[i];
    scalar c;
    if (pf >= 0)   // own upwind: grad(U_comp)[own] . dOwn
    {
        const scalar* gx=(comp==0)?gx0:(comp==1)?gx1:gx2;
        const scalar* gy=(comp==0)?gy0:(comp==1)?gy1:gy2;
        const scalar* gz=(comp==0)?gz0:(comp==1)?gz1:gz2;
        c = pf*(gx[o]*dox[i] + gy[o]*doy[i] + gz[o]*doz[i]);
    }
    else   // nbr upwind: forwardT . (Sum_k w_k (gradU[nbr_k] . dNbr_k))
    {
        scalar rl0=0, rl1=0, rl2=0;
        for (label k=off[i]; k<off[i+1]; ++k)
        {
            const int nb=nbr[k];
            const scalar wk=w[k], dx=dnx[k],dy=dny[k],dz=dnz[k];
            rl0 += wk*(gx0[nb]*dx+gy0[nb]*dy+gz0[nb]*dz);
            rl1 += wk*(gx1[nb]*dx+gy1[nb]*dy+gz1[nb]*dz);
            rl2 += wk*(gx2[nb]*dx+gy2[nb]*dy+gz2[nb]*dz);
        }
        c = rotational ? pf*(fT[(3*comp+0)*n+i]*rl0+fT[(3*comp+1)*n+i]*rl1+fT[(3*comp+2)*n+i]*rl2)
                       : pf*((comp==0)?rl0:(comp==1)?rl1:rl2);
    }
    atomicAdd(&corr[o], c);
}


__device__
void amiLapCorrKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ wf,
    const scalar* __restrict__ nu,
    const scalar* __restrict__ nuN,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ gx0,
    const scalar* __restrict__ gy0,
    const scalar* __restrict__ gz0,
    const scalar* __restrict__ gx1,
    const scalar* __restrict__ gy1,
    const scalar* __restrict__ gz1,
    const scalar* __restrict__ gx2,
    const scalar* __restrict__ gy2,
    const scalar* __restrict__ gz2,
    const scalar* __restrict__ fT,
    int rotational,
    int comp,
    scalar* __restrict__ src)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar w0=wf[i], w1=1.0-w0;
    const scalar* gxc=(comp==0)?gx0:(comp==1)?gx1:gx2;        // own grad(U_comp): row `comp`, components x/y/z
    const scalar* gyc=(comp==0)?gy0:(comp==1)?gy1:gy2;
    const scalar* gzc=(comp==0)?gz0:(comp==1)?gz1:gz2;
    scalar gn[3]={0,0,0};
    scalar R[9];
    if (rotational)
        for (int q=0;q<9;++q)
            R[q]=fT[q*n+i];
    for (label k=off[i]; k<off[i+1]; ++k)
    {
        const int nb=nbr[k];
        const scalar wk=w[k];
        const scalar G[9]={gx0[nb],gy0[nb],gz0[nb], gx1[nb],gy1[nb],gz1[nb], gx2[nb],gy2[nb],gz2[nb]};
        if (rotational)
        {
            for (int j=0;j<3;++j)
            {
                scalar s=0;
                for (int a=0;a<3;++a)
                    for (int b=0;b<3;++b)
                        s+=R[3*comp+a]*G[3*a+b]*R[3*j+b];
                gn[j]+=wk*s;
            }
        }
        else
        {
            gn[0]+=wk*G[3*comp+0];
            gn[1]+=wk*G[3*comp+1];
            gn[2]+=wk*G[3*comp+2];
        }
    }
    const scalar gfx=w0*gxc[o]+w1*gn[0], gfy=w0*gyc[o]+w1*gn[1], gfz=w0*gzc[o]+w1*gn[2];
    const scalar gammaf=w0*nu[o]+w1*nuN[i];
    atomicAdd(&src[o], -(gammaf*magSf[i]*(cvx[i]*gfx + cvy[i]*gfy + cvz[i]*gfz)));
}


__device__
void amiLapCorrPKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ wf,
    const scalar* __restrict__ ga,
    const scalar* __restrict__ gaN,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ gxN,
    const scalar* __restrict__ gyN,
    const scalar* __restrict__ gzN,
    scalar* __restrict__ bp,
    scalar* __restrict__ ffcOut)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar w0=wf[i], w1=1.0-w0;
    const scalar gfx=w0*gx[o]+w1*gxN[i], gfy=w0*gy[o]+w1*gyN[i], gfz=w0*gz[o]+w1*gzN[i];   // grad(p)_face (scalar field)
    const scalar gammaf=w0*ga[o]+w1*gaN[i];
    const scalar ffc=gammaf*magSf[i]*(cvx[i]*gfx + cvy[i]*gfy + cvz[i]*gfz);
    ffcOut[i]=ffc;
    atomicAdd(&bp[o], -ffc);
}
} // namespace


void deviceAmiAddLinUpwindCorr(
    const DeviceAMI& ami,
    int comp,
    const DeviceBuffer<scalar>* gUx,
    const DeviceBuffer<scalar>* gUy,
    const DeviceBuffer<scalar>* gUz,
    DeviceBuffer<scalar>& corr)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const label* off = ami.off.data();
        const label* nbr = ami.nbrCell.data(); const scalar* w = ami.weight.data(); const scalar* phi = ami.phi.data();
        const scalar* gx0 = gUx[0].data(); const scalar* gy0 = gUy[0].data(); const scalar* gz0 = gUz[0].data();
        const scalar* gx1 = gUx[1].data(); const scalar* gy1 = gUy[1].data(); const scalar* gz1 = gUz[1].data();
        const scalar* gx2 = gUx[2].data(); const scalar* gy2 = gUy[2].data(); const scalar* gz2 = gUz[2].data();
        const scalar* dox = ami.dOwnX.data(); const scalar* doy = ami.dOwnY.data(); const scalar* doz = ami.dOwnZ.data();
        const scalar* dnx = ami.dNbrX.data(); const scalar* dny = ami.dNbrY.data(); const scalar* dnz = ami.dNbrZ.data();
        const scalar* fT = ami.rotational ? ami.fT.data() : nullptr; const int rot = ami.rotational ? 1 : 0;
        const int compD = comp; scalar* corrd = corr.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiLinUpwindKernel(n, own, off, nbr, w, phi, gx0, gy0, gz0, gx1, gy1, gz1, gx2, gy2, gz2,
                                dox, doy, doz, dnx, dny, dnz, fT, rot, compD, corrd); });
    }
    cudaCheck(cudaGetLastError(),"amiLinUpwind");
}


// SCALAR overloads. k/epsilon/omega/nuTilda/he have one gradient, not three, and are never transformed
// across a rotational interface -- so the same kernels run with comp = 0, rotational off, and the single
// gradient handed to all three component slots (the other two are read and discarded).
void deviceAmiAddLinUpwindCorr(
    const DeviceAMI& ami,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& corr)
{
    if (ami.n==0) return;
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const label* off = ami.off.data();
        const label* nbr = ami.nbrCell.data(); const scalar* w = ami.weight.data(); const scalar* phi = ami.phi.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        const scalar* dox = ami.dOwnX.data(); const scalar* doy = ami.dOwnY.data(); const scalar* doz = ami.dOwnZ.data();
        const scalar* dnx = ami.dNbrX.data(); const scalar* dny = ami.dNbrY.data(); const scalar* dnz = ami.dNbrZ.data();
        scalar* corrd = corr.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiLinUpwindKernel(n, own, off, nbr, w, phi, gxd, gyd, gzd, gxd, gyd, gzd, gxd, gyd, gzd,
                                dox, doy, doz, dnx, dny, dnz, nullptr, 0, 0, corrd); });
    }
    cudaCheck(cudaGetLastError(),"amiLinUpwindScalar");
}


void deviceAmiAddLapCorr(
    const DeviceAMI& ami,
    int comp,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>* gUx,
    const DeviceBuffer<scalar>* gUy,
    const DeviceBuffer<scalar>* gUz,
    DeviceBuffer<scalar>& corr)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> nuN;
    deviceAmiInterpolate(ami, gammaCell, nuN);
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const label* off = ami.off.data();
        const label* nbr = ami.nbrCell.data(); const scalar* w = ami.weight.data(); const scalar* wf = ami.weights.data();
        const scalar* ga = gammaCell.data(); const scalar* gaN = nuN.data(); const scalar* magSf = ami.magSf.data();
        const scalar* cvx = ami.corrVecX.data(); const scalar* cvy = ami.corrVecY.data(); const scalar* cvz = ami.corrVecZ.data();
        const scalar* gx0 = gUx[0].data(); const scalar* gy0 = gUy[0].data(); const scalar* gz0 = gUz[0].data();
        const scalar* gx1 = gUx[1].data(); const scalar* gy1 = gUy[1].data(); const scalar* gz1 = gUz[1].data();
        const scalar* gx2 = gUx[2].data(); const scalar* gy2 = gUy[2].data(); const scalar* gz2 = gUz[2].data();
        const scalar* fT = ami.rotational ? ami.fT.data() : nullptr; const int rot = ami.rotational ? 1 : 0;
        const int compD = comp; scalar* srcd = corr.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiLapCorrKernel(n, own, off, nbr, w, wf, ga, gaN, magSf, cvx, cvy, cvz,
                              gx0, gy0, gz0, gx1, gy1, gz1, gx2, gy2, gz2, fT, rot, compD, srcd); });
    }
    cudaCheck(cudaGetLastError(),"amiLapCorr");
}


void deviceAmiAddLapCorr(
    const DeviceAMI& ami,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& corr)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> nuN;
    deviceAmiInterpolate(ami, gammaCell, nuN);
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const label* off = ami.off.data();
        const label* nbr = ami.nbrCell.data(); const scalar* w = ami.weight.data(); const scalar* wf = ami.weights.data();
        const scalar* ga = gammaCell.data(); const scalar* gaN = nuN.data(); const scalar* magSf = ami.magSf.data();
        const scalar* cvx = ami.corrVecX.data(); const scalar* cvy = ami.corrVecY.data(); const scalar* cvz = ami.corrVecZ.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        scalar* srcd = corr.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiLapCorrKernel(n, own, off, nbr, w, wf, ga, gaN, magSf, cvx, cvy, cvz,
                              gxd, gyd, gzd, gxd, gyd, gzd, gxd, gyd, gzd, nullptr, 0, 0, srcd); });
    }
    cudaCheck(cudaGetLastError(),"amiLapCorrScalar");
}


void deviceAmiLapCorrP(
    const DeviceAMI& ami,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& bp,
    DeviceBuffer<scalar>& ffcOut)
{
    if (ami.n==0) return;
    ffcOut.resize(ami.n);
    DeviceBuffer<scalar> gaN, gxN, gyN, gzN;
    deviceAmiInterpolate(ami, gammaCell, gaN); deviceAmiInterpolate(ami, gx, gxN);
    deviceAmiInterpolate(ami, gy, gyN); deviceAmiInterpolate(ami, gz, gzN);
    {
        const int n = ami.n; const label* own = ami.ownCell.data(); const scalar* wf = ami.weights.data();
        const scalar* ga = gammaCell.data(); const scalar* gaNd = gaN.data(); const scalar* magSf = ami.magSf.data();
        const scalar* cvx = ami.corrVecX.data(); const scalar* cvy = ami.corrVecY.data(); const scalar* cvz = ami.corrVecZ.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        const scalar* gxNd = gxN.data(); const scalar* gyNd = gyN.data(); const scalar* gzNd = gzN.data();
        scalar* bpd = bp.data(); scalar* ffcOutd = ffcOut.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            amiLapCorrPKernel(n, own, wf, ga, gaNd, magSf, cvx, cvy, cvz, gxd, gyd, gzd, gxNd, gyNd, gzNd, bpd, ffcOutd); });
    }
    cudaCheck(cudaGetLastError(),"amiLapCorrP");
}

} // namespace brae
