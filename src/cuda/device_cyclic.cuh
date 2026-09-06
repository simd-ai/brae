#pragma once
// cf GPU offload, cyclic (periodic) coupling as a SEPARATE lduInterface, exactly like OpenFOAM's
// cyclicFvPatchField. The periodic faces are NOT merged into the owner-sorted internal-face LDU (that breaks
// the amulKernel's owner-contiguous scatter); instead this carries the interface coupling and applies it:
//   - matrix:  diag[own] -= ifCoeff           (folded into the assembled diagonal)
//              Apsi[own] += ifCoeff * psi[nbr] (the off-diagonal, applied inside deviceAmul -> updateInterfaceMatrix)
//   - flux/continuity: phi_cyc = interp(HbyA).Sf ; div[own] += phi_cyc ; phi_cyc -= ifCoeff*(p[nbr]-p[own])
//   - gradient: grad[own] += Sf * (w*psi[own] + (1-w)*psi[nbr]) / V[own]
// Both sides of each periodic pair are stored (buildCyclicInterfaces returns both), so the coupling is symmetric.
// Translational only (R = I): scalars and vector components couple with identity transform. Rotational = Phase 1.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "interface/cyclic_interface.cuh"
#include <vector>

namespace brae {

struct DeviceCyclic
{
    int n = 0;                                  // total cyclic faces (BOTH sides of every pair)
    DeviceBuffer<label>  ownCell, nbrCell;      // this-side cell, periodic-neighbour cell
    DeviceBuffer<scalar> deltaCoeffs, weights, magSf;   // face geometry (own weight w; |Sf|; 1/|delta|)
    DeviceBuffer<scalar> Sfx, Sfy, Sfz;         // face area vector, oriented OUT of ownCell
    DeviceBuffer<scalar> dOwnX, dOwnY, dOwnZ;   // Cf - C[own]          (linearUpwind face delta, own side)
    DeviceBuffer<scalar> dNbrX, dNbrY, dNbrZ;   // Cf_nbr - C[nbr]      (linearUpwind face delta, nbr side, UN-rotated)
    DeviceBuffer<scalar> corrVecX, corrVecY, corrVecZ;   // non-orth correction vector (laplacian "corrected")
    DeviceBuffer<scalar> dX, dY, dZ;            // fvPatch::delta() = dOwn - transform(forwardT, dNbr)
    DeviceBuffer<scalar> ifCoeff;               // assembled per-field off-diagonal: gammaFace*dc*magSf (+ convection)
    DeviceBuffer<scalar> phi;                   // assembled cyclic-face flux (for continuity / corrector)
    // ROTATIONAL (Phase 1): a vector neighbour value is rotated by forwardT (nbr->own) before coupling. Scalars are
    // never rotated. fT packed (3*i+j)*n + face = forwardT[i][j] (identity for translational). ifCoeffC[kk] = the
    // per-component implicit off-diagonal (= ifCoeff * forwardT[kk][kk]; OF transformCoupleField diag-only).
    bool rotational = false;
    DeviceBuffer<scalar> fT;                    // 9*n packed: fT[(3*i+j)*n + face]
    DeviceBuffer<scalar> ifCoeffC[3];
    bool empty() const { return n == 0; }
};

// Flatten every cyclic interface (both patches of each pair) into device arrays.
inline DeviceCyclic buildDeviceCyclic(
    const std::vector<CyclicInterface>& cyclics,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    std::vector<label> oc, nc;
    std::vector<scalar> dc, w, ms, sfx, sfy, sfz;
    std::vector<scalar> dox, doy, doz, dnx, dny, dnz, cvx, cvy, cvz, dlx, dly, dlz;
    bool rot = false;
    for (const auto& c : cyclics)
    {
        if (!c.translational) rot = true;
    }
    for (const auto& c : cyclics)
    {
        const FvPatch& P = fvp[c.patch];
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            const label gf = P.start + (label)i;
            oc.push_back(c.faceCells[i]);
            nc.push_back(c.nbrFaceCells[i]);
            dc.push_back(c.deltaCoeffs[i]);
            w.push_back(c.weights[i]);
            ms.push_back(g.magSf()[gf]);
            sfx.push_back(g.Sf()[gf].x);
            sfy.push_back(g.Sf()[gf].y);
            sfz.push_back(g.Sf()[gf].z);
            dox.push_back(c.dOwn[i].x);
            doy.push_back(c.dOwn[i].y);
            doz.push_back(c.dOwn[i].z);
            dnx.push_back(c.dNbr[i].x);
            dny.push_back(c.dNbr[i].y);
            dnz.push_back(c.dNbr[i].z);
            cvx.push_back(c.corrVec[i].x);
            cvy.push_back(c.corrVec[i].y);
            cvz.push_back(c.corrVec[i].z);
            dlx.push_back(c.delta[i].x);
            dly.push_back(c.delta[i].y);
            dlz.push_back(c.delta[i].z);
        }
    }
    DeviceCyclic d;
    d.n = (int)oc.size();
    d.rotational = rot;
    d.ownCell.copyFrom(oc);
    d.nbrCell.copyFrom(nc);
    d.deltaCoeffs.copyFrom(dc);
    d.weights.copyFrom(w);
    d.magSf.copyFrom(ms);
    d.Sfx.copyFrom(sfx);
    d.Sfy.copyFrom(sfy);
    d.Sfz.copyFrom(sfz);
    d.dOwnX.copyFrom(dox);
    d.dOwnY.copyFrom(doy);
    d.dOwnZ.copyFrom(doz);
    d.dNbrX.copyFrom(dnx);
    d.dNbrY.copyFrom(dny);
    d.dNbrZ.copyFrom(dnz);
    d.corrVecX.copyFrom(cvx);
    d.corrVecY.copyFrom(cvy);
    d.corrVecZ.copyFrom(cvz);
    d.dX.copyFrom(dlx);
    d.dY.copyFrom(dly);
    d.dZ.copyFrom(dlz);
    d.ifCoeff.resize(d.n);
    d.phi.resize(d.n);
    if (rot)   // pack forwardT (9*n, component-major) + the per-component implicit scratch
    {
        std::vector<scalar> ft((std::size_t)9 * d.n);
        std::size_t f = 0;
        for (const auto& c : cyclics)
        {
            const tensor& T = c.forwardT;
            const scalar tc[9] = { T.xx, T.xy, T.xz, T.yx, T.yy, T.yz, T.zx, T.zy, T.zz };
            for (std::size_t i = 0; i < c.faceCells.size(); ++i, ++f)
                for (int k = 0; k < 9; ++k)
                    ft[(std::size_t)k * d.n + f] = tc[k];
        }
        d.fT.copyFrom(ft);
        for (int k = 0; k < 3; ++k)
            d.ifCoeffC[k].resize(d.n);
    }
    return d;
}

// ifCoeff[j] = gammaFace_j * deltaCoeffs_j * magSf_j  with gammaFace = w*gamma[own] + (1-w)*gamma[nbr]
// (the implicit Laplacian off-diagonal); ALSO folds the diagonal contribution diag[own] -= ifCoeff[j].
// gammaCell is a per-cell diffusivity (rAU for pressure, nuEff for momentum). Pass addToDiag=false to skip
// the diag fold (e.g. when reusing ifCoeff only for the flux corrector).
void deviceCyclicAssembleLaplacian(DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                                   DeviceBuffer<scalar>& diag, bool addToDiag = true);

// upwind convection on the interface: diag[own] += max(phi,0), ifCoeff[j] += min(phi,0). Adds to the existing
// ifCoeff (call AFTER deviceCyclicAssembleLaplacian) so Apsi[own] += ifCoeff*psi[nbr] carries div-laplacian.
void deviceCyclicAddConvection(DeviceCyclic& cyc, DeviceBuffer<scalar>& diag,
                               const DeviceBuffer<scalar>* wsch = nullptr);

// MOMENTUM matrix interface coupling M = div(phi,U) - laplacian(nuEff,U) (cyc.phi must hold the current flux):
//   ifCoeff[j] = -(nuFace*dc*magSf) + min(phi,0)            (off-diagonal, used in deviceAmul)
//   diag[own] += (nuFace*dc*magSf) + max(phi,0)             (folded into the momentum diagonal)
// with nuFace = w*nuEff[own]+(1-w)*nuEff[nbr]. Mirrors the interior (deviceDivUpwindCoeffs - deviceLaplacianCoeffs).
// `wsch`: the div scheme's face interpolation weight; null = upwind (pos0(phi)), the previous behaviour.
void deviceCyclicAssembleMomentum(DeviceCyclic& cyc, const DeviceBuffer<scalar>& nuEffCell, DeviceBuffer<scalar>& diag,
                                  const DeviceBuffer<scalar>* wsch = nullptr);

// add the cyclic off-diagonal to H (OF fvMatrix::H): H[own] -= ifCoeff[j]*psi[nbr]/V[own]. Call AFTER deviceMatrixH.
void deviceCyclicAddH(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& psi, const DeviceBuffer<scalar>& V,
                      DeviceBuffer<scalar>& H);
// per-owner sum of |ifCoeff| (the cyclic off-diagonal magnitude) for the relaxation's diagonal-dominance term.
void deviceCyclicOffDiagSum(const DeviceCyclic& cyc, DeviceBuffer<scalar>& sumOff);

// interface off-diagonal pass for the SpMV: Apsi[own] += ifCoeff[j]*psi[nbr]. Called inside deviceAmul when the
// DeviceLduView carries cyclic data (declared in device_ldu.cuh / applied in device_spmv.cu).

// cyclic-face flux of an interpolated vector field: phi[j] = (w*Hx[own]+(1-w)*Hx[nbr])*Sfx + (y,z).
void deviceCyclicFlux(DeviceCyclic& cyc, const DeviceBuffer<scalar>& Hx, const DeviceBuffer<scalar>& Hy,
                      const DeviceBuffer<scalar>& Hz);

// ROTATIONAL (Phase 1), only the VECTOR (U/HbyA) couplings differ; scalars use the functions above
// fill the per-component implicit off-diagonal ifCoeffC[kk] = ifCoeff * forwardT[kk][kk] (call after AssembleMomentum).
void deviceCyclicScaleImplicit(DeviceCyclic& cyc);
// rotational HbyA flux: the neighbour vector is rotated, phi = (w*H[own] + (1-w)*forwardT.H[nbr]) . Sf.
void deviceCyclicFluxRot(DeviceCyclic& cyc, const DeviceBuffer<scalar>& Hx, const DeviceBuffer<scalar>& Hy,
                         const DeviceBuffer<scalar>& Hz);
// rotational H (full rotation, mixes components): H[kk][own] -= ifCoeff * (forwardT.U[nbr])[kk] / V[own], all 3 kk.
void deviceCyclicAddHRot(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                         const DeviceBuffer<scalar>& Uz, const DeviceBuffer<scalar>& V,
                         DeviceBuffer<scalar>& Hx, DeviceBuffer<scalar>& Hy, DeviceBuffer<scalar>& Hz);
// rotational gaussGrad of vector component `comp`: grad[own] += Sf * (w*U[comp][own] + (1-w)*(forwardT.U[nbr])[comp]) / V.
void deviceCyclicAddGradRot(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                            const DeviceBuffer<scalar>& Uz, int comp, const DeviceBuffer<scalar>& V,
                            DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz);
// DEFERRED rotation correction (component comp) into the momentum source: src[own] -= ifCoeff*[(forwardT.U[nbr])[comp]
//   - forwardT[comp][comp]*U[comp][nbr]] (the off-diagonal component-MIXING the diag-only implicit drops). With this in
// the source AND the diagonal H (deviceCyclicAddHDiag), the predictor + H both carry the FULL rotation -> consistent ->
// converges. Mirrors the standard deferred-correction split of a non-implementable implicit term. NOT /V (matrixH does).
void deviceCyclicAddDeferredRot(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                                const DeviceBuffer<scalar>& Uz, int comp, DeviceBuffer<scalar>& src);
// diagonal cyclic H for component comp: H[own] -= ifCoeffC[comp]*U[comp][nbr]/V (pairs with the deferred correction).
void deviceCyclicAddHDiag(const DeviceCyclic& cyc, int comp, const DeviceBuffer<scalar>& psi,
                          const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& H);
// continuity: div[own] += phi[j]/V[own]  (deviceDiv returns the volume-normalized divergence Sum phi / V).
void deviceCyclicAddDiv(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div);
// epsilon setValues: zero the cyclic interface off-diagonal for wall-cell owners (their eps is fixed = eps0).
void deviceCyclicZeroWallIfCoeff(DeviceCyclic& cyc, const DeviceBuffer<label>& isWallCell);
// pressure-correction flux: phi[j] -= ifCoeff[j]*(p[nbr]-p[own])  (the snGrad(p) flux across the periodic face).
void deviceCyclicCorrectFlux(DeviceCyclic& cyc, const DeviceBuffer<scalar>& p);
// gaussGrad contribution: grad[own] += Sf_j * (w*psi[own]+(1-w)*psi[nbr]) / V[own].
// fvc::interpolate of a CELL field onto the cyclic faces: w*psi[own] + (1-w)*psi[nbr].
void deviceCyclicFaceValue(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& cell, DeviceBuffer<scalar>& out);
// OF patchNeighbourField() for a cyclic patch: the raw neighbour cell value per face (rotated by forwardT
// on a rotational cyclic, which is why it takes all three components). For cellLimitedGrad's range.
void deviceCyclicNbrValue(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& cell,
                          const DeviceBuffer<scalar>& c0, const DeviceBuffer<scalar>& c1,
                          const DeviceBuffer<scalar>& c2, int comp, DeviceBuffer<scalar>& out);

void deviceCyclicAddGrad(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& psi, const DeviceBuffer<scalar>& V,
                         DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz);
// linearUpwind deferred correction at the cyclic interface (component comp). Mirrors deviceLinearUpwindCorr but
// the neighbour reconstruction is ROTATED: per face, corr[own] += phi * (gradU_upwind . d_upwind)[comp] with
//   phi>=0 (own upwind):  (grad(U_comp)[own] . dOwn)
//   phi< 0 (nbr upwind):  (forwardT . (gradU[nbr] . dNbr))[comp]  = sum_l forwardT[comp][l] * (grad(U_l)[nbr].dNbr)
// gU{x,y,z}[l] = grad(U_l) (all 3 components, cyclic-inclusive). Caller does relaxSrc[comp] -= corr (as for internal).
void deviceCyclicAddLinUpwindCorr(const DeviceCyclic& cyc, int comp,
                                  const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy,
                                  const DeviceBuffer<scalar>* gUz, DeviceBuffer<scalar>& corr);
// non-orth laplacian "corrected" correction at the cyclic interface (component comp). Per face the explicit face-flux
// correction is ffc = gammaFace*magSf*(corrVec . grad(U_comp)_face), gathered as src[own] -= ffc (owner side), and the
// caller does relaxSrc -= src (mirrors the internal deviceLaplacianCorr). The neighbour gradient ROTATES as a tensor:
// grad((forwardT.U)_comp) = (forwardT . gradU[nbr] . forwardT^T)[comp][:], so it needs ALL 3 component gradients.
// gammaCell = nuEff per cell (interpolated to the face by w). corr is the SAME buffer the internal corr was written to.
// SCALAR overloads (k/epsilon/omega/nuTilda/he): one gradient, never rotated across the interface.
void deviceCyclicAddLinUpwindCorr(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gx,
                                  const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                                  DeviceBuffer<scalar>& corr);
void deviceCyclicAddLapCorr(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                            const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                            const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corr);
void deviceCyclicAddLapCorr(const DeviceCyclic& cyc, int comp, const DeviceBuffer<scalar>& gammaCell,
                            const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy,
                            const DeviceBuffer<scalar>* gUz, DeviceBuffer<scalar>& corr);
// PRESSURE non-orth correction at the cyclic interface (SCALAR field p, no rotation). Per face ffc =
// rAtU_face*magSf*(corrVec . grad(p)_face); adds -ffc to bp[own] (the -V*div pressure source) and returns ffc per
// face in ffcOut so the caller can correct the continuity flux post-solve (cyc.phi -= ffcOut).
void deviceCyclicLapCorrP(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                          const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                          DeviceBuffer<scalar>& bp, DeviceBuffer<scalar>& ffcOut);
// cyclic contribution to the divDevReff tensor divergence (the RAW sum V*fvc::div(sigma), no /V, matches
// tensorDivKernel): srcX/Y/Z[own] += (Sf_j & interp(sigmaC))_{x/y/z}, sigmaC packed component-major (i*3+j)*nC+c.
void deviceCyclicAddTensorDiv(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& sigmaC, int nC,
                              DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ);

// The DIV SCHEME's face weight at the cyclic faces -- the cyclic twin of deviceAmiLimitedVWeights, with
// the 1:1 neighbour in place of the AMI stencil. See device_ami.cuh for the packing and the provenance.
void deviceCyclicLimitedVWeights(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& Ux,
                                 const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                                 const DeviceBuffer<scalar>& gradU, int nC, scalar twoByk,
                                 DeviceBuffer<scalar>& out);
void deviceCyclicLimitedWeights(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& f,
                                const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                                const DeviceBuffer<scalar>& gz, scalar twoByk, DeviceBuffer<scalar>& out);

} // namespace brae
