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
#include "foam_dict.cuh"   // isCyclicFamilyType
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace brae {

// A process-wide serial handed to every LDU addressing uploaded to the device (a DeviceMesh, a
// DeviceLduMatrix), carried by the views over it. The device pool recycles blocks of equal size, so
// the ADDRESS of an owner array is not the identity of its CONTENT: the GAMG hierarchy of a moving
// mesh is rebuilt every step and its coarse levels come back at the same pointers with the same cell
// counts (224000 112000 56000 ... on waveMakerMultiPaddlePiston, 448k cells) but a different pairing,
// and the pointer-keyed Gauss-Seidel level cache replayed the previous step's schedule inside the
// DICGaussSeidel smoother: device U 3.0e-02 vs OpenFOAM at 30 steps where the PCG twin read 6.4e-11,
// and a second run in one process not reproducing the first (1.9e-04). The caches compare this id.
inline unsigned long long nextDeviceAddressingId()
{
    static unsigned long long n = 0;
    return ++n;
}

struct DeviceMesh
{
    int nCells = 0, nInternalFaces = 0, nBndFaces = 0;
    // THE MULTI-GPU TAIL (InterFoamStep.md section 7). A mesh with `processor` patches keeps their faces as
    // the LAST boundary faces of the gather below (bcType 8) and adds, after the nCells owned cells, nHalo
    // slots holding the neighbour ranks' cells: a cell array a processor operator reads has nCellsExt =
    // nCells + nHalo entries, the neighbour of processor face i sits at index nCells + slot (DeviceProcInterface,
    // device_processor.cuh). nBndFacesPhys = the boundary faces BEFORE them (every physical patch), the length
    // of a per-physical-face array. With no processor faces: nHalo = 0, nCellsExt = nCells, nBndFacesPhys =
    // nBndFaces, and nothing else in this struct differs from the single-GPU mesh.
    int nHalo = 0, nCellsExt = 0, nBndFacesPhys = 0;
    unsigned long long addressingId = 0;                // see nextDeviceAddressingId
    DeviceBuffer<label>  owner, nei;                    // internal faces
    DeviceBuffer<scalar> w, V, Sfx, Sfy, Sfz;           // weights(nIf), V(nC), Sf components(all faces)
    DeviceBuffer<scalar> dc, magSf;                     // deltaCoeffs(nIf), |Sf|(all faces), for fvm assembly
    // the laplacian's coefficients for the transported scalars, solver_controls.cuh's `nonOrthCoeffs`: `uncorrected`
    // takes nonOrthDeltaCoeffs with no correction flux, and the scaffold's `nonOrth` says only whether to add that flux
    bool scalarNonOrthCoeffs = false;
    DeviceBuffer<label>  ownerStart, losort, losortStart;
    DeviceBuffer<label>  bndCell, bndGFace, bndPerm, bndCellStart;  // boundary-face gather
    DeviceBuffer<label>  bndIsEmpty;                                // 1 if the boundary face is on an empty patch
    // internal face centres and cell centres, which the face-to-cell offsets are formed from (FaceOffsets)
    DeviceBuffer<scalar> Cfx, Cfy, Cfz;                             // internal face centres (nIf)
    DeviceBuffer<scalar> Cx, Cy, Cz;                                // cell centres (nC)
    DeviceBuffer<scalar> dBndX,dBndY,dBndZ;                         // boundary face: Cf - C(faceCell) (cellLimited grad extrapolation)
    // leastSquaresVectors' inverted dd tensor, 6*nCells component-major, GEOMETRY ONLY (d vectors,
    // weights, |Sf|, the empty-patch skip) -- OpenFOAM builds it once per mesh as a MeshObject and so
    // does brae since FP-3: filled by the first least-squares gradient (lsqInvDdFor, device_fvc.cu).
    // Mutable because every gradient call takes the mesh const, and the tensor is a property of it.
    // Measured before, gasMixing/injectorPipe at 74,650 cells: the tensor was rebuilt at every one of
    // the 10 least-squares gradients of an iteration, 2.0 of the iteration's 15.8 GPU ms.
    mutable DeviceBuffer<scalar> lsqInvDd;
    // ...and WHAT IT WAS BUILT WITH. The tensor folds a periodic pair's faces in (leastSquaresVectors.C:
    // 131-140), so one built without the pair is a different tensor on the pair's cells and the cache
    // cannot be keyed on the cell count alone. -1 = nothing built yet.
    mutable int lsqInvDdCycN = -1;
    // non-orthogonal correction (OF "corrected" laplacian/snGrad): nonOrthDc = 1/max(n.delta, 0.05|delta|)
    // is the implicit deltaCoeffs; corrVec = n - delta*nonOrthDc the explicit deferred-correction vector.
    DeviceBuffer<scalar> nonOrthDc, corrVecX, corrVecY, corrVecZ;   // per internal face
};

// THE CELL-ARRAY LENGTH POLICY of the multi-GPU storage (docs/multigpu-processor-interface.md, "halo-extended
// storage"). A cell array that a neighbour-reading kernel (the interface kernels, neighbour index nCells + slot)
// reads holds nCellsExt = nCells + nHalo entries, the owned cells first and the halo tail after them. Its
// DeviceBuffer keeps size() == nCells -- every launch, check, copyTo and size comparison sees the owned cells only --
// and gets capacity() == nCellsExt. With no processor faces nHalo = 0 and every helper below returns what a plain
// `DeviceBuffer(n)` allocates: same size, same pool class, same pointers.
inline std::size_t cellArrayLength(const DeviceMesh& dm)
{
    return static_cast<std::size_t>(dm.nCellsExt > dm.nCells ? dm.nCellsExt : dm.nCells);
}
// a cell buffer from host values of the owned cells
template <typename T>
inline DeviceBuffer<T> makeCellBuffer(const DeviceMesh& dm, const std::vector<T>& h)
{
    return DeviceBuffer<T>(h, cellArrayLength(dm));
}
// an uninitialised cell buffer (owned cells + halo tail)
template <typename T>
inline DeviceBuffer<T> makeCellBuffer(const DeviceMesh& dm)
{
    return DeviceBuffer<T>(static_cast<std::size_t>(dm.nCells), cellArrayLength(dm));
}
// resize an existing buffer to the owned cells with the halo capacity
template <typename T>
inline void resizeCellBuffer(DeviceBuffer<T>& b, const DeviceMesh& dm)
{
    b.resize(static_cast<std::size_t>(dm.nCells), cellArrayLength(dm));
}

// The face-to-cell offsets of an internal face, Cf - C(owner) and Cf - C(neighbour), formed where a kernel
// reads them (linearUpwind, the cell limiters, least squares) instead of stored per face. It is the one
// subtraction of the same two doubles that buildDeviceMesh and the moving mesh's geometry kernel make, so
// the same bits; and it holds three face and three cell scalars where the stored form held six face
// scalars: 0.5 GB of boxy5's 17.5 GB at 4.94 M cells (BRAE_POOL_STATS census, 2026-10-17).
struct FaceOffsets
{
    const scalar* Cfx;
    const scalar* Cfy;
    const scalar* Cfz;
    const scalar* Cx;
    const scalar* Cy;
    const scalar* Cz;

    // Cf[f] - C[cell], cell being f's owner or its neighbour
    BRAE_HD vector of(
        int f,
        int cell) const
    {
        return vector{Cfx[f] - Cx[cell], Cfy[f] - Cy[cell], Cfz[f] - Cz[cell]};
    }
};

inline FaceOffsets faceOffsets(const DeviceMesh& dm)
{
    return FaceOffsets{dm.Cfx.data(), dm.Cfy.data(), dm.Cfz.data(), dm.Cx.data(), dm.Cy.data(), dm.Cz.data()};
}

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
    std::vector<scalar> CfX(nIf), CfY(nIf), CfZ(nIf), corrX(nIf),corrY(nIf),corrZ(nIf);
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
        CfX[f]=Cf[f].x;
        CfY[f]=Cf[f].y;
        CfZ[f]=Cf[f].z;
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
    int nPhysBnd = -1;                                     // boundary faces before the first processor face
    label haloEnd = nC;                                    // one past the last halo slot (nC + nHalo)
    for (const FvPatch& p : fvp)
    {
        if (isProcessorType(p.type))
        {
            if (p.size > 0 && nPhysBnd < 0) nPhysBnd = static_cast<int>(bndCell.size());
            if (p.coupled) haloEnd = std::max(haloEnd, p.haloBase + p.nHaloCells);
        }
        else if (nPhysBnd >= 0 && p.size > 0 && !isCyclicFamilyType(p.type))
        {
            throw std::runtime_error("brae device mesh: patch '" + p.name + "' has faces after a processor patch; "
                                     "processor patches must be the last boundary patches.");
        }
        if (isCyclicFamilyType(p.type)) continue;
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
    dm.nBndFacesPhys = nPhysBnd < 0 ? nB : nPhysBnd;
    dm.nHalo = static_cast<int>(haloEnd - nC);
    dm.nCellsExt = nC + dm.nHalo;
    dm.addressingId = nextDeviceAddressingId();
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
    dm.Cfx.copyFrom(CfX);
    dm.Cfy.copyFrom(CfY);
    dm.Cfz.copyFrom(CfZ);
    {
        std::vector<scalar> CX(nC), CY(nC), CZ(nC);
        for (int c = 0; c < nC; ++c)
        {
            CX[c] = C[c].x;
            CY[c] = C[c].y;
            CZ[c] = C[c].z;
        }
        dm.Cx.copyFrom(CX);
        dm.Cy.copyFrom(CY);
        dm.Cz.copyFrom(CZ);
    }
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
    // faceCentres and cellCentres, both cleared by clearGeom, and the boundary offsets made from them
    dm.Cfx = std::move(fresh.Cfx); dm.Cfy = std::move(fresh.Cfy); dm.Cfz = std::move(fresh.Cfz);
    dm.Cx = std::move(fresh.Cx); dm.Cy = std::move(fresh.Cy); dm.Cz = std::move(fresh.Cz);
    dm.dBndX = std::move(fresh.dBndX); dm.dBndY = std::move(fresh.dBndY); dm.dBndZ = std::move(fresh.dBndZ);
    // leastSquaresVectors is a MeshObject with MoveableMeshObject semantics: a move invalidates it
    // (leastSquaresVectors.C movePoints -> calcLeastSquaresVectors). Every input of the cached tensor
    // -- w, magSf, Sf, Cf, C, dBnd -- has just been replaced above, so dropping it here is what
    // keeps the cache from serving a pre-move fit on a moving mesh (FP-3).
    dm.lsqInvDd.resize(0);
    // owner/nei, ownerStart/losort/losortStart, bnd* and addressingId deliberately NOT touched:
    // topology, unchanged, so the caches keyed on the id keep serving this mesh.
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
// fvc::surfaceIntegrate divides by mesh.Vsc(), which is mesh.V() unless the mesh MOVES and the caller
// is inside a sub-cycle (fvcSurfaceIntegrate.C:77, fvMeshGeometry.C). Pass that volume as `V` there;
// null takes dm.V, which is every static-mesh caller.
void deviceDiv(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& bval, DeviceBuffer<scalar>& d,
               const DeviceBuffer<scalar>* V = nullptr);
void deviceGaussGrad(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol, const DeviceBuffer<scalar>& bval,
                     DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz,
                     const int* skipIf = nullptr);
// The same gradient for n = 1..3 fields in ONE launch, reading the mesh addressing and geometry once
// instead of n times. gx/gy/gz are arrays of n buffers, resized here; vol/bval are arrays of n pointers.
// BIT-IDENTICAL, per field, to n separate deviceGaussGrad calls -- same faces, same order, same
// expressions, one register set per field (tests/test_grad_fused.cu asserts it with memcmp). skipIf
// behaves exactly as it does above. Measured motive: gradKernel was 4.5 ms of a 30 ms iteration at 306k
// cells, 16 launches of which nine were velocity components in threes.
// leastSquares gradient -- the same signature as deviceGaussGrad, so a caller switches on the case's
// gradScheme and nothing else. See the kernels in device_fvc.cu for why no volume division appears.
// `cyc` non-null carries the PERIODIC PAIR: the neighbour cell's value across the interface with the
// coupled fit vector (leastSquaresGrad.C:108-119). SCALARS only -- a scalar is never rotated across the
// interface, so no transform appears; grad(U) across a pair is refused at its own site.
void deviceLeastSquaresGrad(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol,
                            const DeviceBuffer<scalar>& bval,
                            DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz,
                            const struct DeviceCyclic* cyc = nullptr);

// fvc::grad UNDER THE CASE'S OWN gradSchemes ENTRY, which is two steps and not one: the scheme itself
// (leastSquares or Gauss linear) and then cellLimited when its k is above zero. It is the device twin of
// the host's gradOf (grad_choice.cuh:34-48) and takes the same pair of numbers a GradChoice holds, so a
// call site switches on the case's entry and nothing else -- which is what deviceLeastSquaresGrad's own
// note above anticipated.
//
// SPLIT INTO TWO NUMBERS rather than taking GradChoice, because that type lives in the host tree beside
// fvc and this header is included by device translation units that must not pull it in.
//
// RAS/electrostaticDeposition is the shipped case that needs it: `gradSchemes { default cellLimited
// leastSquares 1; }`, so every gradient of that case is both. Its host arm RUNS and its device arm
// refused, which is the only device refusal in brae's interFoam port that a stock tutorial reaches.
void deviceGradOf(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol, const DeviceBuffer<scalar>& bval,
                  bool leastSquares, scalar cellLimitK,
                  DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz,
                  const struct DeviceCyclic* cyc = nullptr);

void deviceGaussGradFused(const DeviceMesh& dm, int n,
                          const DeviceBuffer<scalar>* const* vol, const DeviceBuffer<scalar>* const* bval,
                          DeviceBuffer<scalar>* gx, DeviceBuffer<scalar>* gy, DeviceBuffer<scalar>* gz,
                          const int* skipIf = nullptr);
// The least-squares twin of deviceGaussGradFused: up to three fields fitted in ONE launch that reads
// the addressing, the d vectors and the cell's inverted dd tensor once. Bit-identical per field to
// deviceLeastSquaresGrad (tests/test_lsq_grad_fused.cu, memcmp). The raw form writes into slices the
// caller owns (a 9*nC grad(U) tensor), so the vector gradient needs no copies and no stream sync.
void deviceLeastSquaresGradFused(const DeviceMesh& dm, int n,
                                 const DeviceBuffer<scalar>* const* vol, const DeviceBuffer<scalar>* const* bval,
                                 DeviceBuffer<scalar>* gx, DeviceBuffer<scalar>* gy, DeviceBuffer<scalar>* gz);
void deviceLeastSquaresGradFusedRaw(const DeviceMesh& dm, int n,
                                    const scalar* const* vol, const scalar* const* bval,
                                    scalar* const* gx, scalar* const* gy, scalar* const* gz);
// The mesh's inverted dd tensor, built on the first request and cached on the mesh (see DeviceMesh).
// BRAE_LSQ_INVDD=recompute rebuilds it at every request: the control that restores the old path.
// `cyc` non-null folds the periodic pair's faces into the tensor, as leastSquaresVectors.C:131-140
// does for any coupled patch; the cache remembers which it was built with.
const scalar* lsqInvDdFor(const DeviceMesh& dm, const struct DeviceCyclic* cyc = nullptr);

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
// The N-field form (FP-4): up to three fields limited in ONE launch that reads the addressing and the
// face offsets once, bit-identical per field to n separate deviceCellLimitGrad calls
// (tests/test_cell_limit_grad_fused.cu, memcmp). Coupled interfaces are not handled here -- a caller
// with them keeps the per-field call, which still runs the five-phase scatter.
void deviceCellLimitGradFused(const DeviceMesh& dm, int n,
                              const DeviceBuffer<scalar>* const* U, const DeviceBuffer<scalar>* const* Ubnd,
                              DeviceBuffer<scalar>* gx, DeviceBuffer<scalar>* gy, DeviceBuffer<scalar>* gz,
                              scalar k);

// G4: fvm matrix assembly on device, produces the raw lduMatrix coefficients (diag/upper/lower), the
// boundary internalCoeffs/boundaryCoeffs are folded by the solver (G5). gammafInt / phiInt are nIf-sized.
// nonOrth=false: orthogonal laplacian (deltaCoeffs = 1/|delta|, the OF "orthogonal" scheme).
// nonOrth=true : the implicit part of the OF "corrected" scheme (nonOrthDeltaCoeffs = 1/max(n.delta, 0.05|delta|));
//                the matching explicit deferred correction is deviceLaplacianCorr (added to the eqn source).
void deviceLaplacianCoeffs(const DeviceMesh& dm, const DeviceBuffer<scalar>& gammafInt,
                           DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& upper, DeviceBuffer<scalar>& lower,
                           bool nonOrth = false);
// The same matrix holding its off-diagonal once: a laplacian's lower is its upper bit for bit, so a caller
// whose matrix nothing makes asymmetric afterwards hands `upper` to every reader as the lower too. One
// face scalar saved: 208 MB on boxy5's 4.94 M cells (BRAE_POOL_STATS census, 2026-10-17).
void deviceLaplacianCoeffsSymmetric(const DeviceMesh& dm, const DeviceBuffer<scalar>& gammafInt,
                                    DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& upper,
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
// THE LIMITER SELECTOR carried in `twoByk` everywhere below. It is a real coefficient for limitedLinear
// (2/max(k,SMALL), always > 0); 0 already means vanAlbada; and kVanLeerTwoByk selects vanLeer. A sentinel
// rather than a sign convention, because device_ami.cu reads `twoByk <= 0` as vanAlbada and a negative
// RANGE would have changed the AMI path's meaning without a word. vanLeer is what every interFoam
// tutorial names for div(phi,alpha) -- OF vanLeer.H:70, limiter = (r + |r|)/(1 + |r|).
constexpr scalar kVanLeerTwoByk = -1.0;

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
// The limitedLinear FACE WEIGHTS alone, for an EXPLICIT divergence -- rhoSimpleFoam's fvc::div(phi,Ekp),
// where the scheme changes the face value rather than any matrix coefficient. Same limiter as
// deviceDivLimitedCoeffs, through the same device function, and the same currency: twoByk = 2/max(k,SMALL),
// NOT the raw k the case writes. Face value is then w*field[own] + (1-w)*field[nei].
// `Gauss interfaceCompression`, the PhiScheme four of the waveMaker tutorials name for the alpha
// fluxes: its limiter is the quartic function of the TWO CELL VALUES alone, so it takes no gradient.
// Face value is then w*field[own] + (1-w)*field[nei], as with the limited weights below.
void deviceInterfaceCompressionWeights(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt,
                                       const DeviceBuffer<scalar>& field, DeviceBuffer<scalar>& w);
void deviceLimitedFaceWeights(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& field,
                              const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                              scalar twoByk, DeviceBuffer<scalar>& w);
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
