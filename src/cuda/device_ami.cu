// cf GPU offload, cyclicAMI weighted-stencil coupling kernels. See device_ami.cuh.
#include "device_ami.cuh"
#include <cuda_runtime.h>

namespace brae {
namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__global__
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


__global__
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


__global__
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
    amiAmulKernel<<<nBlocks(ami.n), TPB>>>(ami.n, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(),
                                           ami.weight.data(), ami.ifCoeff.data(), psi.data(), Apsi.data());
    cudaCheck(cudaGetLastError(), "amiAmul");
}

// Like deviceAmiAmul but with an EXPLICIT off-diagonal coefficient: the ROTATIONAL momentum matvec must use the
// per-component diag-scaled ifCoeffC[kk] (= ifCoeff*forwardT[kk][kk]), NOT the base ifCoeff, to be consistent with
// the deferred-rotation source. Translational / scalar (pressure): pass ami.ifCoeff (== deviceAmiAmul).
void deviceAmiAmulCoeff(const DeviceAMI& ami, const DeviceBuffer<scalar>& coeff,
                        const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi)
{
    if (ami.n == 0) return;
    amiAmulKernel<<<nBlocks(ami.n), TPB>>>(ami.n, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(),
                                           ami.weight.data(), coeff.data(), psi.data(), Apsi.data());
    cudaCheck(cudaGetLastError(), "amiAmulCoeff");
}


namespace {
__global__
void amiMomKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ nu,
    const scalar* __restrict__ nuN,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ wsch,   // div-scheme face weight; null = upwind (pos0(phi))
    scalar* __restrict__ ifc,
    scalar* __restrict__ diag)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;

    const int o=own[i];
    const scalar lap = (w[i]*nu[o] + (1.0-w[i])*nuN[i]) * dc[i] * magSf[i];
    const scalar p = phi[i];
    // THE CONVECTIVE SPLIT IS THE SCHEME'S, not always upwind. OpenFOAM assembles a coupled patch with
    // the interpolation weights of the div scheme the case NAMED:
    //     internalCoeffs = phi*w      boundaryCoeffs = -phi*(1-w)
    // and the solver applies -boundaryCoeffs against the neighbour, so the off-diagonal is phi*(1-w).
    // Upwind is the special case w = pos0(phi), which recovers max(phi,0)/min(phi,0) exactly -- which is
    // why a null wsch here is bit-identical to the hardcoded form it replaces.
    const scalar ws = wsch ? wsch[i] : (p > 0.0 ? scalar(1) : scalar(0));
    ifc[i] = -lap + p*(1.0 - ws);
    atomicAdd(&diag[o], lap + p*ws);
}


__global__
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


__global__
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


__global__
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


__global__
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


__global__
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


__global__
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


__global__
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


void deviceAmiAssembleMomentum(DeviceAMI& ami, const DeviceBuffer<scalar>& nuEffCell, DeviceBuffer<scalar>& diag,
                               const DeviceBuffer<scalar>* wsch)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> nuN;
    deviceAmiInterpolate(ami, nuEffCell, nuN);
    amiMomKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), nuEffCell.data(), nuN.data(), ami.deltaCoeffs.data(),
        ami.weights.data(), ami.magSf.data(), ami.phi.data(),
        (wsch && (label)wsch->size() == ami.n) ? wsch->data() : nullptr,
        ami.ifCoeff.data(), diag.data());
    cudaCheck(cudaGetLastError(),"amiMom");
}


void deviceAmiAssembleLaplacian(DeviceAMI& ami, const DeviceBuffer<scalar>& gammaCell, DeviceBuffer<scalar>& diag, bool addToDiag)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> gN;
    deviceAmiInterpolate(ami, gammaCell, gN);
    amiLaplKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), gammaCell.data(), gN.data(), ami.deltaCoeffs.data(),
        ami.weights.data(), ami.magSf.data(), ami.ifCoeff.data(), diag.data(), addToDiag?1:0);
    cudaCheck(cudaGetLastError(),"amiLapl");
}


void deviceAmiOffDiagSum(const DeviceAMI& ami, DeviceBuffer<scalar>& sumOff)
{
    if (ami.n==0) return;
    amiOffSumKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.ifCoeff.data(), ami.weightsSum.data(), sumOff.data());
    cudaCheck(cudaGetLastError(),"amiOffSum");
}


void deviceAmiAddH(const DeviceAMI& ami, const DeviceBuffer<scalar>& UkNbr, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& H)
{
    if (ami.n==0) return;
    amiAddHKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.ifCoeff.data(), UkNbr.data(), V.data(), H.data());
    cudaCheck(cudaGetLastError(),"amiAddH");
}


void deviceAmiFlux(DeviceAMI& ami, const DeviceBuffer<scalar>& Hx, const DeviceBuffer<scalar>& Hy, const DeviceBuffer<scalar>& Hz)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> Hnx,Hny,Hnz;
    deviceAmiInterpolateVec(ami, Hx,Hy,Hz, Hnx,Hny,Hnz);
    amiFluxKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.weights.data(), Hx.data(),Hy.data(),Hz.data(),
        Hnx.data(),Hny.data(),Hnz.data(), ami.Sfx.data(),ami.Sfy.data(),ami.Sfz.data(), ami.phi.data());
    cudaCheck(cudaGetLastError(),"amiFlux");
}


void deviceAmiAddDiv(const DeviceAMI& ami, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div)
{
    if (ami.n==0) return;
    amiDivAddKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.phi.data(), V.data(), div.data());
    cudaCheck(cudaGetLastError(),"amiDiv");
}


namespace {
__global__
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
    amiZeroWallKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), isWallCell.data(), ami.ifCoeff.data());
    cudaCheck(cudaGetLastError(),"amiZeroWall");
}


namespace {
// face value of a CELL field on an interface face: fvc::interpolate on a coupled patch, i.e.
//   w*patchInternalField + (1-w)*patchNeighbourField
__global__
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
    amiFaceValueKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.weights.data(),
                                               cell.data(), nbr.data(), out.data());
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
    amiGradAddKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.weights.data(), psi.data(), pN.data(),
        ami.Sfx.data(),ami.Sfy.data(),ami.Sfz.data(), V.data(), gx.data(),gy.data(),gz.data());
    cudaCheck(cudaGetLastError(),"amiGrad");
}


namespace {
__global__
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
__global__
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


__global__
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


__global__
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
    amiScaleImplicitKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ifCoeff.data(), ami.fT.data(),
        ami.ifCoeffC[0].data(), ami.ifCoeffC[1].data(), ami.ifCoeffC[2].data());
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
    amiDeferredRotKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.ifCoeff.data(), ami.fT.data(), comp,
        uiX.data(), uiY.data(), uiZ.data(), src.data());
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
    amiGradRotKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.weights.data(), Uown.data(), UNbr.data(),
        ami.Sfx.data(), ami.Sfy.data(), ami.Sfz.data(), V.data(), gx.data(), gy.data(), gz.data());
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
    amiTensorDivKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(),
        ami.weight.data(), ami.weights.data(), ami.Sfx.data(), ami.Sfy.data(), ami.Sfz.data(),
        sigmaC.data(), nC, ami.rotational?ami.fT.data():nullptr, ami.rotational?1:0, srcX.data(), srcY.data(), srcZ.data());
    cudaCheck(cudaGetLastError(),"amiTensorDiv");
}


void deviceAmiCorrectFlux(DeviceAMI& ami, const DeviceBuffer<scalar>& p)
{
    if (ami.n==0) return;
    DeviceBuffer<scalar> pN;
    deviceAmiInterpolate(ami, p, pN);
    amiFluxCorrKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.ifCoeff.data(), pN.data(), p.data(), ami.phi.data());
    cudaCheck(cudaGetLastError(),"amiFluxCorr");
}


void deviceAmiInterpolate(const DeviceAMI& ami, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& out)
{
    if (ami.n == 0) return;
    out.resize(ami.n);
    amiInterpKernel<<<nBlocks(ami.n), TPB>>>(ami.n, ami.off.data(), ami.nbrCell.data(), ami.weight.data(),
                                             psi.data(), out.data());
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
    amiInterpVecKernel<<<nBlocks(ami.n), TPB>>>(ami.n, ami.off.data(), ami.nbrCell.data(), ami.weight.data(),
        Ux.data(), Uy.data(), Uz.data(), ami.rotational ? ami.fT.data() : nullptr, ami.rotational ? 1 : 0,
        oX.data(), oY.data(), oZ.data());
    cudaCheck(cudaGetLastError(), "amiInterpVec");
}


namespace {
__global__
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


__global__
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


__global__
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
    amiLinUpwindKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(),
        ami.weight.data(), ami.phi.data(), gUx[0].data(),gUy[0].data(),gUz[0].data(), gUx[1].data(),gUy[1].data(),gUz[1].data(),
        gUx[2].data(),gUy[2].data(),gUz[2].data(), ami.dOwnX.data(),ami.dOwnY.data(),ami.dOwnZ.data(),
        ami.dNbrX.data(),ami.dNbrY.data(),ami.dNbrZ.data(), ami.rotational?ami.fT.data():nullptr, ami.rotational?1:0, comp, corr.data());
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
    amiLinUpwindKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(),
        ami.weight.data(), ami.phi.data(), gx.data(),gy.data(),gz.data(), gx.data(),gy.data(),gz.data(),
        gx.data(),gy.data(),gz.data(), ami.dOwnX.data(),ami.dOwnY.data(),ami.dOwnZ.data(),
        ami.dNbrX.data(),ami.dNbrY.data(),ami.dNbrZ.data(), nullptr, 0, 0, corr.data());
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
    amiLapCorrKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(), ami.weight.data(),
        ami.weights.data(), gammaCell.data(), nuN.data(), ami.magSf.data(), ami.corrVecX.data(),ami.corrVecY.data(),ami.corrVecZ.data(),
        gUx[0].data(),gUy[0].data(),gUz[0].data(), gUx[1].data(),gUy[1].data(),gUz[1].data(), gUx[2].data(),gUy[2].data(),gUz[2].data(),
        ami.rotational?ami.fT.data():nullptr, ami.rotational?1:0, comp, corr.data());
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
    amiLapCorrKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(), ami.weight.data(),
        ami.weights.data(), gammaCell.data(), nuN.data(), ami.magSf.data(), ami.corrVecX.data(),ami.corrVecY.data(),ami.corrVecZ.data(),
        gx.data(),gy.data(),gz.data(), gx.data(),gy.data(),gz.data(), gx.data(),gy.data(),gz.data(),
        nullptr, 0, 0, corr.data());
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
    amiLapCorrPKernel<<<nBlocks(ami.n),TPB>>>(ami.n, ami.ownCell.data(), ami.weights.data(), gammaCell.data(), gaN.data(),
        ami.magSf.data(), ami.corrVecX.data(),ami.corrVecY.data(),ami.corrVecZ.data(), gx.data(),gy.data(),gz.data(),
        gxN.data(),gyN.data(),gzN.data(), bp.data(), ffcOut.data());
    cudaCheck(cudaGetLastError(),"amiLapCorrP");
}

} // namespace brae

namespace brae {
namespace {

// The TVD limiter at an AMI face. OF LimitedScheme::calcLimiter's coupled branch, with the four
// patch-side substitutions spelled out in limitedSchemes_cpp.cuh:
//   CDweights -> ami.weights,  C[nei]-C[own] -> ami.delta,  lPhi[nei] -> patchNeighbourField,
//   gradc[nei] -> the gradient's patchNeighbourField (R G R^T on a rotational interface).
//
// This is the SAME arithmetic divLimitedVFaceKernel does on an internal face -- deliberately, because a
// coupled face is not a special scheme, it is the same scheme reading its neighbour through the AMI.
__global__
void amiLimitedVWeightKernel(
    int n,
    int nC,
    const label*  __restrict__ own,
    const label*  __restrict__ off,
    const label*  __restrict__ nbr,
    const scalar* __restrict__ wq,        // AMI stencil weights
    const scalar* __restrict__ cd,        // the patch's interpolation weights
    const scalar* __restrict__ dX,
    const scalar* __restrict__ dY,
    const scalar* __restrict__ dZ,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ U0,
    const scalar* __restrict__ U1,
    const scalar* __restrict__ U2,
    const scalar* __restrict__ g,         // packed grad(U): g[q*nC + c], q = 3i+j = d(U_j)/d(x_i)
    const scalar* __restrict__ fT,
    int                        rotational,
    scalar                     twoByk,
    scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int o = own[i];
    const scalar p = phi[i];

    scalar R[9];
    if (rotational)
        for (int q = 0; q < 9; ++q) R[q] = fT[q*n + i];

    // patchNeighbourField of U and of grad(U), both accumulated over the AMI stencil.
    scalar uN[3] = {0, 0, 0};
    scalar gN[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (label k = off[i]; k < off[i+1]; ++k)
    {
        const int c = nbr[k];
        const scalar wk = wq[k];
        scalar v[3] = { U0[c], U1[c], U2[c] };
        scalar G[9];
        for (int q = 0; q < 9; ++q) G[q] = g[q*nC + c];
        if (rotational)
        {
            const scalar rv[3] = { R[0]*v[0] + R[1]*v[1] + R[2]*v[2],
                                   R[3]*v[0] + R[4]*v[1] + R[5]*v[2],
                                   R[6]*v[0] + R[7]*v[1] + R[8]*v[2] };
            v[0] = rv[0]; v[1] = rv[1]; v[2] = rv[2];
            // A gradient is a rank-2 tensor: R G R^T, both indices. R G alone would leave the derivative
            // index in the neighbour's frame, which is invisible until the interface actually rotates.
            scalar t[9];
            for (int r = 0; r < 3; ++r)
                for (int c2 = 0; c2 < 3; ++c2)
                {
                    scalar sum = 0;
                    for (int x = 0; x < 3; ++x)
                        for (int y = 0; y < 3; ++y)
                            sum += R[3*r + x] * G[3*x + y] * R[3*c2 + y];
                    t[3*r + c2] = sum;
                }
            for (int q = 0; q < 9; ++q) G[q] = t[q];
        }
        uN[0] += wk*v[0]; uN[1] += wk*v[1]; uN[2] += wk*v[2];
        for (int q = 0; q < 9; ++q) gN[q] += wk*G[q];
    }

    const scalar dx = dX[i], dy = dY[i], dz = dZ[i];
    const scalar gf0 = uN[0] - U0[o], gf1 = uN[1] - U1[o], gf2 = uN[2] - U2[o];   // gradfV
    const scalar gradf = gf0*gf0 + gf1*gf1 + gf2*gf2;

    scalar G[9];                                             // the UPWIND cell's gradient
    if (p > 0.0) for (int q = 0; q < 9; ++q) G[q] = g[q*nC + o];
    else         for (int q = 0; q < 9; ++q) G[q] = gN[q];

    // (d & gradc)_j = d_i * gradc_ij -- a COLUMN dot in this packing.
    const scalar dg0 = dx*G[0] + dy*G[3] + dz*G[6];
    const scalar dg1 = dx*G[1] + dy*G[4] + dz*G[7];
    const scalar dg2 = dx*G[2] + dy*G[5] + dz*G[8];
    const scalar gradcf = gf0*dg0 + gf1*dg1 + gf2*dg2;

    scalar r;
    if (fabs(gradcf) >= 1000.0 * fabs(gradf))
        r = 2.0 * 1000.0 * ((gradcf >= 0.0) ? 1.0 : -1.0) * ((gradf >= 0.0) ? 1.0 : -1.0) - 1.0;
    else
        r = 2.0 * (gradcf / gradf) - 1.0;
    scalar limiter = twoByk * r;
    limiter = (limiter < 0.0) ? 0.0 : (limiter > 1.0 ? 1.0 : limiter);
    out[i] = limiter * cd[i] + (1.0 - limiter) * ((p >= 0.0) ? 1.0 : 0.0);
}


// The scalar form. A scalar is never rotated across the interface; its gradient is, as a vector.
__global__
void amiLimitedWeightKernel(
    int n,
    const label*  __restrict__ own,
    const label*  __restrict__ off,
    const label*  __restrict__ nbr,
    const scalar* __restrict__ wq,
    const scalar* __restrict__ cd,
    const scalar* __restrict__ dX,
    const scalar* __restrict__ dY,
    const scalar* __restrict__ dZ,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ f,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ fT,
    int                        rotational,
    scalar                     twoByk,
    scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int o = own[i];
    const scalar p = phi[i];

    scalar R[9];
    if (rotational)
        for (int q = 0; q < 9; ++q) R[q] = fT[q*n + i];

    scalar fN = 0, gNx = 0, gNy = 0, gNz = 0;
    for (label k = off[i]; k < off[i+1]; ++k)
    {
        const int c = nbr[k];
        const scalar wk = wq[k];
        fN += wk * f[c];                       // NOT transformed: a scalar has no orientation
        scalar vx = gx[c], vy = gy[c], vz = gz[c];
        if (rotational)
        {
            const scalar rx = R[0]*vx + R[1]*vy + R[2]*vz;
            const scalar ry = R[3]*vx + R[4]*vy + R[5]*vz;
            const scalar rz = R[6]*vx + R[7]*vy + R[8]*vz;
            vx = rx; vy = ry; vz = rz;
        }
        gNx += wk*vx; gNy += wk*vy; gNz += wk*vz;
    }

    const scalar dx = dX[i], dy = dY[i], dz = dZ[i];
    const scalar gradf = fN - f[o];
    const scalar gcx = (p > 0.0) ? gx[o] : gNx;
    const scalar gcy = (p > 0.0) ? gy[o] : gNy;
    const scalar gcz = (p > 0.0) ? gz[o] : gNz;
    const scalar gradcf = dx*gcx + dy*gcy + dz*gcz;

    scalar r;
    if (fabs(gradcf) >= 1000.0 * fabs(gradf))
        r = 2.0 * 1000.0 * ((gradcf >= 0.0) ? 1.0 : -1.0) * ((gradf >= 0.0) ? 1.0 : -1.0) - 1.0;
    else
        r = 2.0 * (gradcf / gradf) - 1.0;
    scalar limiter;
    if (twoByk > 0.0)
    {
        limiter = twoByk * r;
        limiter = (limiter < 0.0) ? 0.0 : (limiter > 1.0 ? 1.0 : limiter);
    }
    else
    {
        limiter = r * (r + 1.0) / (r*r + 1.0);    // vanAlbada, as divLimitedFaceKernel does
    }
    out[i] = limiter * cd[i] + (1.0 - limiter) * ((p >= 0.0) ? 1.0 : 0.0);
}

} // namespace


void deviceAmiLimitedVWeights(
    const DeviceAMI&            ami,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& gradU,
    const int                   nC,
    const scalar                twoByk,
    DeviceBuffer<scalar>&       out)
{
    if (!ami.n) return;
    out.resize(ami.n);
    amiLimitedVWeightKernel<<<nBlocks(ami.n), TPB>>>(
        ami.n, nC, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(), ami.weight.data(),
        ami.weights.data(), ami.dX.data(), ami.dY.data(), ami.dZ.data(), ami.phi.data(),
        Ux.data(), Uy.data(), Uz.data(), gradU.data(),
        ami.rotational ? ami.fT.data() : nullptr, ami.rotational ? 1 : 0, twoByk, out.data());
    cudaCheck(cudaGetLastError(), "amiLimitedVWeight");
}


void deviceAmiLimitedWeights(
    const DeviceAMI&            ami,
    const DeviceBuffer<scalar>& f,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    const scalar                twoByk,
    DeviceBuffer<scalar>&       out)
{
    if (!ami.n) return;
    out.resize(ami.n);
    amiLimitedWeightKernel<<<nBlocks(ami.n), TPB>>>(
        ami.n, ami.ownCell.data(), ami.off.data(), ami.nbrCell.data(), ami.weight.data(),
        ami.weights.data(), ami.dX.data(), ami.dY.data(), ami.dZ.data(), ami.phi.data(),
        f.data(), gx.data(), gy.data(), gz.data(),
        ami.rotational ? ami.fT.data() : nullptr, ami.rotational ? 1 : 0, twoByk, out.data());
    cudaCheck(cudaGetLastError(), "amiLimitedWeight");
}

} // namespace brae
