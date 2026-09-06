#pragma once
// cf GPU offload (G3): DeviceMesh, the finite-volume geometry + addressing uploaded once. Internal-face
// gather (ownerStart / losort / losortStart, as in G1) turns the grad/div scatter into a per-cell gather;
// a parallel boundary-face gather (bndCellStart / bndPerm) adds the boundary contributions race-free.
// Face area vectors are stored SoA (Sfx/Sfy/Sfz). Boundary face values are passed per fvc call (flattened
// across patches in patch order).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "device_buffer.cuh"
#include "interface/cyclic_interface.cuh"
#include "foam_dict.cuh"   // isCoupledInterfaceType
#include <vector>

namespace brae {

struct DeviceMesh
{
    int nCells = 0, nInternalFaces = 0, nBndFaces = 0;
    DeviceBuffer<label>  owner, nei;                    // internal faces
    DeviceBuffer<scalar> w, V, Sfx, Sfy, Sfz;           // weights(nIf), V(nC), Sf components(all faces)
    DeviceBuffer<scalar> dc, magSf;                     // deltaCoeffs(nIf), |Sf|(all faces), for fvm assembly
    DeviceBuffer<label>  ownerStart, losort, losortStart;
    DeviceBuffer<label>  bndCell, bndGFace, bndPerm, bndCellStart;  // boundary-face gather
    DeviceBuffer<label>  bndIsEmpty;                                // 1 if the boundary face is on an empty patch
    DeviceBuffer<scalar> dOwnX,dOwnY,dOwnZ, dNeiX,dNeiY,dNeiZ;      // internal face: Cf - C(owner) / Cf - C(neighbour) (linearUpwind)
    DeviceBuffer<scalar> dBndX,dBndY,dBndZ;                         // boundary face: Cf - C(faceCell) (cellLimited grad extrapolation)
    // non-orthogonal correction (OF "corrected" laplacian/snGrad): nonOrthDc = 1/max(n.delta, 0.05|delta|)
    // is the implicit deltaCoeffs; corrVec = n - delta*nonOrthDc the explicit deferred-correction vector.
    DeviceBuffer<scalar> nonOrthDc, corrVecX, corrVecY, corrVecZ;   // per internal face
};

// cyclic (periodic) patches are SKIPPED in the boundary gather: they are not boundary faces. The periodic
// coupling is applied as a SEPARATE interface (DeviceCyclic, device_cyclic.cuh) exactly like OpenFOAM's
// cyclicFvPatchField::updateInterfaceMatrix, it is NOT merged into this owner-sorted internal-face LDU (doing
// so breaks the amulKernel's owner-contiguous scatter, which assumes faces sorted by owner). The `cyclics`
// argument is kept for API symmetry but the LDU itself carries only the mesh internal faces.
// Refresh ONLY the geometric buffers after a mesh move -- OF fvMesh::movePoints.
//
// OF invalidates exactly two groups and nothing else:
//   primitiveMesh::clearGeom        cellCentres, cellVolumes, faceCentres, faceAreas
//   surfaceInterpolation::clearOut  weights, deltaCoeffs, nonOrthDeltaCoeffs, nonOrthCorrectionVectors
//
// TOPOLOGY SURVIVES. owner/neighbour and the derived addressing (ownerStart, losort, losortStart, the
// boundary gather) are functions of the FACES, not the points, and polyMesh::movePoints keeps the
// topology fixed. Rebuilding them would be wasted work; worse, the LDU addressing is what every
// assembled matrix and the AMG hierarchy are indexed by, so silently reordering it mid-run would
// invalidate them. This function therefore moves only the geometry across and leaves addressing alone.
//
// Implemented by rebuilding through buildDeviceMesh and adopting the geometric buffers, rather than
// duplicating the upload logic: a second copy of it would drift from the builder, and the two would
// disagree on some later change to how a quantity is computed. The cost is that the addressing is
// rebuilt and discarded -- correct first; if it shows up in a profile, split the builder in two.
inline void refreshDeviceMeshGeometry(
    DeviceMesh& dm,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::vector<CyclicInterface>& cyclics = {});

inline DeviceMesh buildDeviceMesh(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::vector<CyclicInterface>& cyclics = {})
{
    (void)cyclics;
    const int nC = m.nCells(), nIf0 = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<vector>& Sf = g.Sf();
    const std::vector<scalar>& magSfAll = g.magSf();
    const std::vector<vector>& Cf = g.Cf();
    const std::vector<vector>& C = g.C();
    const std::vector<scalar>& wts = g.weights();
    const std::vector<scalar>& dcAll = g.deltaCoeffs();
    const std::vector<scalar>& nodcAll = g.nonOrthDeltaCoeffs();
    const std::vector<vector>& cvAll = g.nonOrthCorrectionVectors();

    const int nIf = nIf0;                                  // internal faces = mesh internal faces only (cyclic is a separate interface)

    std::vector<label>  ownI(nIf), neiI(nIf);
    std::vector<scalar> wI(nIf), dcI(nIf), nodcI(nIf);
    std::vector<scalar> dOwnX(nIf),dOwnY(nIf),dOwnZ(nIf), dNeiX(nIf),dNeiY(nIf),dNeiZ(nIf), corrX(nIf),corrY(nIf),corrZ(nIf);
    std::vector<scalar> SfxN, SfyN, SfzN, magSfN;          // Sf layout: [internal | non-cyclic boundary]
    SfxN.reserve(magSfAll.size());
    SfyN.reserve(magSfAll.size());
    SfzN.reserve(magSfAll.size());
    magSfN.reserve(magSfAll.size());
    for (int f = 0; f < nIf0; ++f)                         // original internal faces
    {
        ownI[f]=own[f];
        neiI[f]=nei[f];
        wI[f]=wts[f];
        dcI[f]=dcAll[f];
        nodcI[f]=nodcAll[f];
        const vector dO=Cf[f]-C[own[f]], dN=Cf[f]-C[nei[f]];
        dOwnX[f]=dO.x;
        dOwnY[f]=dO.y;
        dOwnZ[f]=dO.z;
        dNeiX[f]=dN.x;
        dNeiY[f]=dN.y;
        dNeiZ[f]=dN.z;
        corrX[f]=cvAll[f].x;
        corrY[f]=cvAll[f].y;
        corrZ[f]=cvAll[f].z;
        SfxN.push_back(Sf[f].x);
        SfyN.push_back(Sf[f].y);
        SfzN.push_back(Sf[f].z);
        magSfN.push_back(magSfAll[f]);
    }
    std::vector<label> bndCell, bndGFace, bndIsEmpty;      // non-cyclic boundary faces (cyclic patches skipped)
    std::vector<scalar> dBndX, dBndY, dBndZ;               // Cf - C(faceCell) per boundary face (cellLimited grad)
    for (const FvPatch& p : fvp)
    {
        if (isCoupledInterfaceType(p.type)) continue;
        const label emp = (p.type == "empty") ? 1 : 0;
        for (label i = 0; i < p.size; ++i)
        {
            bndCell.push_back(p.faceCells[i]);
            bndGFace.push_back((label)SfxN.size());
            bndIsEmpty.push_back(emp);
            SfxN.push_back(Sf[p.start+i].x);
            SfyN.push_back(Sf[p.start+i].y);
            SfzN.push_back(Sf[p.start+i].z);
            magSfN.push_back(magSfAll[p.start+i]);
            const vector dB = Cf[p.start+i] - C[p.faceCells[i]];
            dBndX.push_back(dB.x);
            dBndY.push_back(dB.y);
            dBndZ.push_back(dB.z);
        }
    }
    const int nB = static_cast<int>(bndCell.size());

    std::vector<label> ownerStart(nC+1,0), losortStart(nC+1,0), losort(nIf);   // gather addressing over the extended faces
    for (int f = 0; f < nIf; ++f)
    {
        ownerStart[ownI[f]+1]++;
        losortStart[neiI[f]+1]++;
    }
    for (int c = 0; c < nC; ++c)
    {
        ownerStart[c+1]+=ownerStart[c];
        losortStart[c+1]+=losortStart[c];
    }
    {
        std::vector<label> pos(losortStart.begin(), losortStart.end());
        for (int f = 0; f < nIf; ++f)
            losort[pos[neiI[f]]++] = f;
    }
    std::vector<label> bndCellStart(nC+1,0), bndPerm(nB);
    for (int k = 0; k < nB; ++k)
        bndCellStart[bndCell[k]+1]++;
    for (int c = 0; c < nC; ++c)
        bndCellStart[c+1]+=bndCellStart[c];
    {
        std::vector<label> pos(bndCellStart.begin(), bndCellStart.end());
        for (int k = 0; k < nB; ++k)
            bndPerm[pos[bndCell[k]]++] = k;
    }

    DeviceMesh dm;
    dm.nCells = nC;
    dm.nInternalFaces = nIf;
    dm.nBndFaces = nB;
    dm.owner.copyFrom(ownI);
    dm.nei.copyFrom(neiI);
    dm.w.copyFrom(wI);
    dm.V.copyFrom(g.V());
    dm.Sfx.copyFrom(SfxN);
    dm.Sfy.copyFrom(SfyN);
    dm.Sfz.copyFrom(SfzN);
    dm.dc.copyFrom(dcI);
    dm.magSf.copyFrom(magSfN);
    dm.ownerStart.copyFrom(ownerStart);
    dm.losort.copyFrom(losort);
    dm.losortStart.copyFrom(losortStart);
    dm.bndCell.copyFrom(bndCell);
    dm.bndGFace.copyFrom(bndGFace);
    dm.bndPerm.copyFrom(bndPerm);
    dm.bndCellStart.copyFrom(bndCellStart);
    dm.bndIsEmpty.copyFrom(bndIsEmpty);
    dm.dOwnX.copyFrom(dOwnX);
    dm.dOwnY.copyFrom(dOwnY);
    dm.dOwnZ.copyFrom(dOwnZ);
    dm.dNeiX.copyFrom(dNeiX);
    dm.dNeiY.copyFrom(dNeiY);
    dm.dNeiZ.copyFrom(dNeiZ);
    dm.dBndX.copyFrom(dBndX);
    dm.dBndY.copyFrom(dBndY);
    dm.dBndZ.copyFrom(dBndZ);
    dm.nonOrthDc.copyFrom(nodcI);
    dm.corrVecX.copyFrom(corrX);
    dm.corrVecY.copyFrom(corrY);
    dm.corrVecZ.copyFrom(corrZ);
    return dm;
}

inline void refreshDeviceMeshGeometry(
    DeviceMesh& dm,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::vector<CyclicInterface>& cyclics)
{
    DeviceMesh fresh = buildDeviceMesh(m, g, fvp, cyclics);
    if (fresh.nCells != dm.nCells || fresh.nInternalFaces != dm.nInternalFaces
     || fresh.nBndFaces != dm.nBndFaces)
        throw std::runtime_error("brae: refreshDeviceMeshGeometry saw a different mesh size -- a move "
                                 "must not change topology (OF polyMesh::movePoints keeps it fixed).");

    // primitiveMesh::clearGeom equivalents
    dm.V     = std::move(fresh.V);
    dm.Sfx   = std::move(fresh.Sfx);
    dm.Sfy   = std::move(fresh.Sfy);
    dm.Sfz   = std::move(fresh.Sfz);
    dm.magSf = std::move(fresh.magSf);
    // surfaceInterpolation::clearOut equivalents
    dm.w         = std::move(fresh.w);
    dm.dc        = std::move(fresh.dc);
    dm.nonOrthDc = std::move(fresh.nonOrthDc);
    dm.corrVecX  = std::move(fresh.corrVecX);
    dm.corrVecY  = std::move(fresh.corrVecY);
    dm.corrVecZ  = std::move(fresh.corrVecZ);
    // face-to-cell offsets: functions of faceCentres and cellCentres, both cleared by clearGeom
    dm.dOwnX = std::move(fresh.dOwnX); dm.dOwnY = std::move(fresh.dOwnY); dm.dOwnZ = std::move(fresh.dOwnZ);
    dm.dNeiX = std::move(fresh.dNeiX); dm.dNeiY = std::move(fresh.dNeiY); dm.dNeiZ = std::move(fresh.dNeiZ);
    dm.dBndX = std::move(fresh.dBndX); dm.dBndY = std::move(fresh.dBndY); dm.dBndZ = std::move(fresh.dBndZ);
    // owner/nei, ownerStart/losort/losortStart, bnd* deliberately NOT touched: topology, unchanged.
}


// Flatten a boundary field (per patch -> per face) into one device array in patch order (matches bndCell order).
inline std::vector<scalar> flattenBoundary(const std::vector<std::vector<scalar>>& bnd)
{
    std::vector<scalar> out;
    for (const auto& pb : bnd)
        out.insert(out.end(), pb.begin(), pb.end());
    return out;
}

void deviceInterpolate(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol, DeviceBuffer<scalar>& sfInt);
void deviceDiv(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& bval, DeviceBuffer<scalar>& d);
void deviceGaussGrad(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol, const DeviceBuffer<scalar>& bval,
                     DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz,
                     const int* skipIf = nullptr);

// The COUPLED-PATCH half of cellLimitedGrad, which brae's addressing cannot see on its own.
//
// A cyclic/cyclicAMI/cyclicACMI face is a boundary face to brae (DeviceBoundary skips it, so it is in
// neither dm.bndCellStart nor Ubnd) but an INTERNAL one to OF's limiter: cellLimitedGrad::calcGrad folds
// psf.patchNeighbourField() into maxVsf/minVsf for every coupled patch, and its second loop limits the
// extrapolation to EVERY boundary face, coupled ones included. Without this an interface cell is limited
// as though the interface were not there.
//
// nbrVal is this component's neighbour value per interface face, already interpolated (AMI) and rotated
// (rotational cyclic) by the caller -- the rotation machinery lives there and is not duplicated here.
struct CellLimitInterface
{
    int n = 0;
    const label*  ownCell = nullptr;
    const scalar* nbrVal  = nullptr;
    const scalar* dOwnX   = nullptr;   // Cf - C[own], the extrapolation vector for this face
    const scalar* dOwnY   = nullptr;
    const scalar* dOwnZ   = nullptr;
};

// cellLimited Gauss linear <k> (OF cellLimitedGrad<minmod>) applied to one component's gradient (gx,gy,gz in/out).
// With no interfaces this is the single-kernel path it has always been, bit for bit.
void deviceCellLimitGrad(const DeviceMesh& dm, const DeviceBuffer<scalar>& U, const DeviceBuffer<scalar>& Ubnd,
                         DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz, scalar k,
                         const CellLimitInterface* ifs = nullptr, int nIfs = 0);

// G4: fvm matrix assembly on device, produces the raw lduMatrix coefficients (diag/upper/lower), the
// boundary internalCoeffs/boundaryCoeffs are folded by the solver (G5). gammafInt / phiInt are nIf-sized.
// nonOrth=false: orthogonal laplacian (deltaCoeffs = 1/|delta|, the OF "orthogonal" scheme).
// nonOrth=true : the implicit part of the OF "corrected" scheme (nonOrthDeltaCoeffs = 1/max(n.delta, 0.05|delta|));
//                the matching explicit deferred correction is deviceLaplacianCorr (added to the eqn source).
void deviceLaplacianCoeffs(const DeviceMesh& dm, const DeviceBuffer<scalar>& gammafInt,
                           DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& upper, DeviceBuffer<scalar>& lower,
                           bool nonOrth = false);
// Explicit non-orth correction of fvm::laplacian(gamma, vf): returns lapCorrSource = -V*fvc::div(faceFluxCorr),
// faceFluxCorr_f = gamma_f*|Sf|_f * (corrVec_f . grad(vf)_f) with grad linearly interpolated to the face.
// This equals fvm::laplacian(gamma,vf).source()'s correction term (OF gaussLaplacianScheme). gx/gy/gz = cell grad(vf).
void deviceLaplacianCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& gammafInt,
                         const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                         DeviceBuffer<scalar>& lapCorrSource);
// Split helpers: the per-face flux correction ffc_f (nIf), and the integrated source -V*fvc::div(ffc) (nC).
// The pressure path needs BOTH, add the source to b AND subtract ffc from the reconstructed flux (continuity).
void deviceLaplacianCorrFlux(const DeviceMesh& dm, const DeviceBuffer<scalar>& gammafInt,
                             const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                             DeviceBuffer<scalar>& ffc);
// limitedSnGrad variant: per-face OF limiter psi (>=1 -> unlimited, bit-identical). phi = field being corrected.
void deviceLaplacianCorrFluxLimited(const DeviceMesh& dm, const DeviceBuffer<scalar>& gammafInt, const DeviceBuffer<scalar>& phi,
                                    const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                                    scalar psi, DeviceBuffer<scalar>& ffc);
// The VECTOR form of the above: ONE limiter per face from mag(snGrad) and mag(corr) over all three
// components, as OF's limitedSnGrad<Type> forms it. Limiting each component on its own is a different
// scheme -- see the kernel comment. phi/gxc/gyc/gzc/ffc are arrays of 3.
void deviceLaplacianCorrFluxLimitedVec(const DeviceMesh& dm, const DeviceBuffer<scalar>& gammafInt,
                                       const DeviceBuffer<scalar>& p0, const DeviceBuffer<scalar>& p1,
                                       const DeviceBuffer<scalar>& p2,
                                       const DeviceBuffer<scalar>* gxc, const DeviceBuffer<scalar>* gyc,
                                       const DeviceBuffer<scalar>* gzc,
                                       scalar psi, DeviceBuffer<scalar>* ffc);
void deviceFaceDivSource(const DeviceMesh& dm, const DeviceBuffer<scalar>& ffc, DeviceBuffer<scalar>& src);
void deviceDivUpwindCoeffs(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt,
                           DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& upper, DeviceBuffer<scalar>& lower);
// limitedLinear convection (OF "Gauss limitedLinear k_"): implicit limited face weight W_f = limiter*CDweight +
// (1-limiter)*pos0(phi), limiter = clamp(twoByk*r, 0, 1), r = NVDTVD ratio from grad(field). twoByk = 2/max(k_,SMALL)
// (k_=1 -> twoByk=2). gx/gy/gz = cell grad(field) (Gauss). Reduces to deviceDivUpwindCoeffs at limiter=0.
void deviceDivCentralCoeffs(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt,
                            DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& upper, DeviceBuffer<scalar>& lower);
void deviceDivLimitedCoeffs(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& field,
                            const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                            scalar twoByk, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& upper, DeviceBuffer<scalar>& lower);
// limitedLinearV (OF "Gauss limitedLinearV k_"): the NVDVTVDV vector limiter -- ONE limiter per face from the vector
// U (gradfV=U[N]-U[P], gradcf=gradfV.(d&gradU[upwind]), r=2*gradcf/gradf-1), applied to all 3 components. U/gUx/gUy/gUz
// are 3-element arrays (per component). Reduces to deviceDivUpwindCoeffs at limiter=0. Built implicitly like magSqr.
void deviceDivLimitedVCoeffs(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>* U,
                             const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy, const DeviceBuffer<scalar>* gUz,
                             scalar twoByk, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& upper, DeviceBuffer<scalar>& lower);
// ... taking deviceGradU's PACKED gradient (gradU[q*nC + c], q = 3i + j) rather than nine buffers.
void deviceDivLimitedVCoeffs(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>* U,
                             const DeviceBuffer<scalar>& gradU,
                             scalar twoByk, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& upper, DeviceBuffer<scalar>& lower);
// G5: fold the boundary + pEqn source for the pressure solve -> diagC (= rawDiag + internalCoeffs),
// b (= V*divPhi + boundaryCoeffs). iC/bC are flattened pEqn boundary coeffs (patch order = bndCell order).
void deviceFoldPressure(const DeviceMesh& dm, const DeviceBuffer<scalar>& rawDiag, const DeviceBuffer<scalar>& divPhi,
                        const DeviceBuffer<scalar>& iC, const DeviceBuffer<scalar>& bC,
                        DeviceBuffer<scalar>& diagC, DeviceBuffer<scalar>& b);
// Generic boundary fold (source given directly): diagC = rawDiag + internalCoeffs; b = source + boundaryCoeffs.
void deviceFold(const DeviceMesh& dm, const DeviceBuffer<scalar>& rawDiag, const DeviceBuffer<scalar>& source,
                const DeviceBuffer<scalar>& iC, const DeviceBuffer<scalar>& bC,
                DeviceBuffer<scalar>& diagC, DeviceBuffer<scalar>& b);

} // namespace brae
