#pragma once
// cf GPU offload, cyclicAMI (Arbitrary Mesh Interface) coupling. The cyclic 1:1 (own,nbr) pair becomes a
// weighted STENCIL: each source face couples to a CSR list of neighbour cells with area weights (the AMI).
//   pnf[srcFace]      = sum_{k in [off[i],off[i+1])} w[k] * (forwardT . psi[nbrCell[k]])      (interpolateToSource)
//   Apsi[ownCell[i]] += ifCoeff[i] * pnf[srcFace]                                             (updateInterfaceMatrix)
// Mirrors OF AMIInterpolation::interpolateToSource + cyclicAMIFvPatchField::updateInterfaceMatrix. Both src+tgt
// sides are flattened (symmetric), exactly like DeviceCyclic. Host weights: AMIInterface (faceAreaWeightAMI).
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "interface/ami_interface.cuh"
#include <vector>

namespace brae {

struct DeviceAMI
{
    int n = 0;                                  // total SOURCE faces (both sides of the pair flattened)
    int nnz = 0;                                // total stencil entries
    DeviceBuffer<label>  ownCell;               // owner cell of each source face            (n)
    DeviceBuffer<label>  off;                   // CSR offsets into (nbrCell,weight)          (n+1)
    DeviceBuffer<label>  nbrCell;               // target-face owner cell per stencil entry   (nnz)
    DeviceBuffer<scalar> weight;                // normalised AMI weight per stencil entry    (nnz)
    DeviceBuffer<scalar> weightsSum;            // per src face coverage (lowWeight gate)      (n)
    DeviceBuffer<scalar> magSf, deltaCoeffs, weights;   // per src face geometry               (n)
    DeviceBuffer<scalar> Sfx, Sfy, Sfz;         // per src face area vector (out of ownCell)   (n)
    DeviceBuffer<scalar> corrVecX, corrVecY, corrVecZ;  // per src face non-orth correction vec (n)
    DeviceBuffer<scalar> dOwnX, dOwnY, dOwnZ;   // per src face Cf - C[own]                    (n)
    DeviceBuffer<scalar> dNbrX, dNbrY, dNbrZ;   // per STENCIL ENTRY Cf_tgt - C[nbr]           (nnz)
    DeviceBuffer<scalar> dX, dY, dZ;            // per src face fvPatch::delta()               (n)
    DeviceBuffer<scalar> ifCoeff;               // assembled per-field off-diagonal           (n)
    DeviceBuffer<scalar> phi;                   // assembled interface flux                   (n)
    bool rotational = false;
    DeviceBuffer<scalar> fT;                    // 9*n packed forwardT[(3i+j)*n+face] (per source face's interface)
    DeviceBuffer<scalar> ifCoeffC[3];           // ROTATIONAL: per-component diag-scaled implicit (ifCoeff*forwardT[kk][kk])
    bool empty() const { return n == 0; }
};

inline DeviceAMI buildDeviceAMI(const std::vector<AMIInterface>& amis)
{
    std::vector<label> oc, offv, nc;
    std::vector<scalar> w, ws, msf, dcf, wt, sfx, sfy, sfz;
    std::vector<scalar> cvx,cvy,cvz, dox,doy,doz, dnx,dny,dnz, dlx,dly,dlz;
    bool rot = false;
    for (const auto& a : amis)
        if (!a.translational) rot = true;
    offv.push_back(0);
    for (const auto& a : amis)
    {
        for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        {
            oc.push_back(a.ownCell[i]);
            ws.push_back(a.weightsSum[i]);
            msf.push_back(a.magSf[i]);
            dcf.push_back(a.deltaCoeffs[i]);
            wt.push_back(a.weights[i]);
            sfx.push_back(a.Sf[i].x);
            sfy.push_back(a.Sf[i].y);
            sfz.push_back(a.Sf[i].z);
            // corrVec/dOwn/dNbr feed only the linearUpwind/non-orth interface corrections; a minimal AMIInterface
            // (e.g. the device unit test) may leave them empty -> default to zero (those corrections then no-op).
            const vector cv = i < a.corrVec.size() ? a.corrVec[i] : vector{0,0,0};
            const vector dO = i < a.dOwn.size()    ? a.dOwn[i]    : vector{0,0,0};
            const vector dL = i < a.delta.size()   ? a.delta[i]   : vector{0,0,0};
            cvx.push_back(cv.x);
            cvy.push_back(cv.y);
            cvz.push_back(cv.z);
            dox.push_back(dO.x);
            doy.push_back(dO.y);
            doz.push_back(dO.z);
            dlx.push_back(dL.x);
            dly.push_back(dL.y);
            dlz.push_back(dL.z);
            const label b = a.srcOffset[i], e = a.srcOffset[i+1];
            for (label k = b; k < e; ++k)
            {
                nc.push_back(a.nbrCell[k]);
                w.push_back(a.weight[k]);
                const vector dN = (std::size_t)k < a.dNbr.size() ? a.dNbr[k] : vector{0,0,0};
                dnx.push_back(dN.x);
                dny.push_back(dN.y);
                dnz.push_back(dN.z);
            }
            offv.push_back((label)nc.size());
        }
    }
    DeviceAMI d;
    d.n = (int)oc.size();
    d.nnz = (int)nc.size();
    d.rotational = rot;
    d.ownCell.copyFrom(oc);
    d.off.copyFrom(offv);
    d.nbrCell.copyFrom(nc);
    d.weight.copyFrom(w);
    d.weightsSum.copyFrom(ws);
    d.magSf.copyFrom(msf);
    d.deltaCoeffs.copyFrom(dcf);
    d.weights.copyFrom(wt);
    d.Sfx.copyFrom(sfx);
    d.Sfy.copyFrom(sfy);
    d.Sfz.copyFrom(sfz);
    d.corrVecX.copyFrom(cvx);
    d.corrVecY.copyFrom(cvy);
    d.corrVecZ.copyFrom(cvz);
    d.dOwnX.copyFrom(dox);
    d.dOwnY.copyFrom(doy);
    d.dOwnZ.copyFrom(doz);
    d.dNbrX.copyFrom(dnx);
    d.dNbrY.copyFrom(dny);
    d.dNbrZ.copyFrom(dnz);
    d.dX.copyFrom(dlx);
    d.dY.copyFrom(dly);
    d.dZ.copyFrom(dlz);
    d.ifCoeff.resize(d.n);
    d.phi.resize(d.n);
    if (rot)
    {
        std::vector<scalar> ft((std::size_t)9 * d.n);
        std::size_t f = 0;
        for (const auto& a : amis)
        {
            const tensor& T = a.forwardT;
            const scalar tc[9] = { T.xx,T.xy,T.xz, T.yx,T.yy,T.yz, T.zx,T.zy,T.zz };
            for (std::size_t i = 0; i < a.ownCell.size(); ++i, ++f)
                for (int k = 0; k < 9; ++k)
                    ft[(std::size_t)k * d.n + f] = tc[k];
        }
        d.fT.copyFrom(ft);
        for (int k = 0; k < 3; ++k)
            d.ifCoeffC[k].resize(d.n);
    }
    return d;
}

// AMI interpolate-to-source of a SCALAR cell field: out[i] = sum_k weight[k]*psi[nbrCell[k]]  (no transform).
void deviceAmiInterpolate(const DeviceAMI& ami, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& out);
// AMI interpolate-to-source of a VECTOR field, rotated: outX/Y/Z[i] = sum_k w[k]*(forwardT.U[nbrCell[k]]).
void deviceAmiInterpolateVec(const DeviceAMI& ami, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                             const DeviceBuffer<scalar>& Uz, DeviceBuffer<scalar>& oX, DeviceBuffer<scalar>& oY,
                             DeviceBuffer<scalar>& oZ);
// implicit matrix coupling (OF cyclicAMIFvPatchField::updateInterfaceMatrix): the SCALAR off-diagonal SpMV pass ,
//   Apsi[ownCell[i]] += ifCoeff[i] * sum_k weight[k]*psi[nbrCell[k]]
// (interpolate-to-source THEN scale by the assembled interface coefficient). Scalar field -> no rotation. Call
// alongside the internal-face amul; pass cyc.ifCoeff assembled by the laplacian/momentum AMI assembly.
void deviceAmiAmul(const DeviceAMI& ami, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi);
// deviceAmiAmul with an explicit off-diagonal coefficient (rotational momentum: ifCoeffC[kk]; else ami.ifCoeff).
void deviceAmiAmulCoeff(const DeviceAMI& ami, const DeviceBuffer<scalar>& coeff,
                        const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi);

// assembly (mirrors device_cyclic, with the AMI weighted stencil; explicit ops use the interpolated nbr)
// MOMENTUM M = div(phi,U) - laplacian(nuEff,U): ifCoeff[i] = -(nuFace*dc*magSf) + min(phi,0); diag[own] +=
// (nuFace*dc*magSf) + max(phi,0). nuFace = w*nuEff[own] + (1-w)*interp(nuEff). ami.phi must hold the current flux.
// `wsch`, when given, is the div scheme's face interpolation weight per source face; null means upwind
// (pos0(phi)), which is what this assembly used to hardcode for every case.
void deviceAmiAssembleMomentum(DeviceAMI& ami, const DeviceBuffer<scalar>& nuEffCell, DeviceBuffer<scalar>& diag,
                               const DeviceBuffer<scalar>* wsch = nullptr);
// laplacian(gamma): ifCoeff[i] = gammaFace*dc*magSf; diag[own] -= ifCoeff (when addToDiag).
void deviceAmiAssembleLaplacian(DeviceAMI& ami, const DeviceBuffer<scalar>& gammaCell, DeviceBuffer<scalar>& diag, bool addToDiag = true);
// per-owner sum |ifCoeff|*weightsSum (relax diagonal-dominance term).
void deviceAmiOffDiagSum(const DeviceAMI& ami, DeviceBuffer<scalar>& sumOff);
// H[own] -= ifCoeff * UkNbr[i] / V  (UkNbr = precomputed interp of the component, rotated if rotational).
void deviceAmiAddH(const DeviceAMI& ami, const DeviceBuffer<scalar>& UkNbr, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& H);
// interface flux: phi[i] = (w*H[own] + (1-w)*interp(forwardT.H[nbr])) . Sf.
void deviceAmiFlux(DeviceAMI& ami, const DeviceBuffer<scalar>& Hx, const DeviceBuffer<scalar>& Hy, const DeviceBuffer<scalar>& Hz);
// fvc::interpolate of a CELL field onto the interface faces: w*psi[own] + (1-w)*interp(psi[nbr]).
// Needed wherever a face diffusivity is wanted on the interface -- fvc::ddtCorr's interpolate(rAU).
void deviceAmiFaceValue(const DeviceAMI& ami, const DeviceBuffer<scalar>& cell, DeviceBuffer<scalar>& out);
// continuity: div[own] += phi[i]/V.
void deviceAmiAddDiv(const DeviceAMI& ami, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div);
// epsilon setValues: zero the interface off-diagonal for wall-cell owners (their eps is fixed = eps0).
void deviceAmiZeroWallIfCoeff(DeviceAMI& ami, const DeviceBuffer<label>& isWallCell);
// gaussGrad (scalar psi): grad[own] += Sf*(w*psi[own]+(1-w)*interp(psi))/V.
void deviceAmiAddGrad(const DeviceAMI& ami, const DeviceBuffer<scalar>& psi, const DeviceBuffer<scalar>& V,
                      DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz);
// pressure-correction flux: phi[i] -= ifCoeff*(interp(p) - p[own]).
void deviceAmiCorrectFlux(DeviceAMI& ami, const DeviceBuffer<scalar>& p);
// divDevReff stress flux at the AMI interface (raw V*fvc::div, like the cyclic): srcX/Y/Z[own] += (Sf & sigma_face),
// sigma_face = w*sigma[own] + (1-w)*interp(sigma[nbr]); a rank-2 tensor rotates as sigma'=R*sigma*R^T (rotational).
void deviceAmiAddTensorDiv(const DeviceAMI& ami, const DeviceBuffer<scalar>& sigmaC, int nC,
                           DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ);

// ROTATIONAL (the predictor needs the diag-implicit + deferred split, like cyclic Bug #3)
// fill ifCoeffC[kk] = ifCoeff * forwardT[kk][kk] (call after AssembleMomentum; the predictor uses ifCoeffC).
void deviceAmiScaleImplicit(DeviceAMI& ami);
// DEFERRED rotation mixing into the predictor source (component comp): src[own] -= ifCoeff*[(forwardT.U_interp)[comp]
//   - forwardT[comp][comp]*U_interp[comp]]. U_interp{X,Y,Z} = the UN-rotated AMI-interpolated neighbour per src face
// (= deviceAmiInterpolate of each component, precomputed from the U_old snapshot). Pairs with ifCoeffC in the matrix
// and the FULL-rotation H (deviceAmiAddH on the rotated interpVec of the post-solve U) -> consistent full rotation.
void deviceAmiAddDeferredRot(const DeviceAMI& ami, const DeviceBuffer<scalar>& uiX, const DeviceBuffer<scalar>& uiY,
                             const DeviceBuffer<scalar>& uiZ, int comp, DeviceBuffer<scalar>& src);
// rotational gaussGrad of a component for divDevReff: grad[own] += Sf*(w*Uown[own] + (1-w)*UNbr[i])/V, UNbr = the
// ROTATED AMI-interpolated neighbour component (= deviceAmiInterpolateVec output), precomputed.
void deviceAmiAddGradRot(const DeviceAMI& ami, const DeviceBuffer<scalar>& Uown, const DeviceBuffer<scalar>& UNbr,
                         const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz);

// linearUpwind / non-orth deferred corrections at the AMI interface (Bug #5/#6 analogs; rotated)
// linearUpwind deferred correction (component comp): corr[own] += phi*(gradU_upwind.d_upwind)[comp]; outflow uses
// grad(U_comp)[own].dOwn (no rotation), inflow uses (forwardT.(Sum_k w_k.(gradU[nbr_k].dNbr_k)))[comp]. gU{x,y,z}[l] =
// grad(U_l). Adds to the SAME corr buffer the internal linearUpwind wrote; caller does relaxSrc[comp] -= corr.
void deviceAmiAddLinUpwindCorr(const DeviceAMI& ami, int comp, const DeviceBuffer<scalar>* gUx,
                               const DeviceBuffer<scalar>* gUy, const DeviceBuffer<scalar>* gUz, DeviceBuffer<scalar>& corr);
// non-orth laplacian correction (momentum, component comp): src[own] -= gammaFace*magSf*(corrVec.grad(U_comp)_face);
// neighbour grad rotated as a TENSOR R.gradU.R^T then AMI-interpolated. gammaCell = nuEff per cell.
// SCALAR overloads (k/epsilon/omega/nuTilda/he): one gradient, never rotated across the interface.
void deviceAmiAddLinUpwindCorr(const DeviceAMI& ami, const DeviceBuffer<scalar>& gx,
                               const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                               DeviceBuffer<scalar>& corr);
void deviceAmiAddLapCorr(const DeviceAMI& ami, const DeviceBuffer<scalar>& gammaCell,
                         const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                         const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corr);
void deviceAmiAddLapCorr(const DeviceAMI& ami, int comp, const DeviceBuffer<scalar>& gammaCell,
                         const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy, const DeviceBuffer<scalar>* gUz,
                         DeviceBuffer<scalar>& corr);
// PRESSURE non-orth correction (scalar p): ffc = rAtU_face*magSf*(corrVec.grad(p)_face) (no rotation); -ffc into
// bp[own]; ffcOut[i] = ffc for the post-solve flux correction (ami.phi -= ffcOut).
// THE DIV SCHEME'S FACE WEIGHT AT THE INTERFACE, for `Gauss limitedLinear[V] k`. Feeds
// deviceAmiAssembleMomentum's `wsch`; without it the interface is assembled upwind whatever the case
// asked for -- see amiMomKernel. OF LimitedScheme::calcLimiter, coupled branch.
//
// gradU is the PACKED cell gradient deviceGradU produces, gradU[q*nC + c] with q = 3i + j = d(U_j)/d(x_i),
// which is already the packing NVDVTVDV::r wants -- no transpose on this path. ami.phi must hold the
// current interface flux (the same field the assembly upwinds on).
void deviceAmiLimitedVWeights(const DeviceAMI& ami, const DeviceBuffer<scalar>& Ux,
                              const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                              const DeviceBuffer<scalar>& gradU, int nC, scalar twoByk,
                              DeviceBuffer<scalar>& out);
// The SCALAR form (k, epsilon, omega, nuTilda): the value is not rotated across the interface, its
// gradient is. twoByk <= 0 selects vanAlbada, exactly as divLimitedFaceKernel does on internal faces.
void deviceAmiLimitedWeights(const DeviceAMI& ami, const DeviceBuffer<scalar>& f,
                             const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                             const DeviceBuffer<scalar>& gz, scalar twoByk, DeviceBuffer<scalar>& out);

void deviceAmiLapCorrP(const DeviceAMI& ami, const DeviceBuffer<scalar>& gammaCell, const DeviceBuffer<scalar>& gx,
                       const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& bp, DeviceBuffer<scalar>& ffcOut);

} // namespace brae
