#include "device_fv_geometry.cuh"
#include "inter_phase_time.cuh"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

const char* const WHO = "brae deviceFvGeometry: ";

// fv_geometry.cu's own
constexpr scalar geoRootVSmall = 1.0e-150;
constexpr scalar geoVSmall = 1.0e-300;

constexpr int geoTPB = 256;

int geoBlocks(label n)
{
    return static_cast<int>((n + geoTPB - 1)/geoTPB);
}

struct GeoVec
{
    scalar x;
    scalar y;
    scalar z;
};

__device__
inline GeoVec geoAt(
    const scalar* p,
    label i)
{
    const scalar* q = p + 3*static_cast<std::size_t>(i);
    return GeoVec{q[0], q[1], q[2]};
}

__device__
inline void geoPut(
    scalar* p,
    label i,
    const GeoVec& v)
{
    scalar* q = p + 3*static_cast<std::size_t>(i);
    q[0] = v.x;
    q[1] = v.y;
    q[2] = v.z;
}

__device__
inline GeoVec geoSub(
    const GeoVec& a,
    const GeoVec& b)
{
    return GeoVec{a.x - b.x, a.y - b.y, a.z - b.z};
}

// a ^ b as the host compiles it: each component's first product fused, its second rounded
__device__
inline GeoVec geoCross(
    const GeoVec& a,
    const GeoVec& b)
{
    return GeoVec
    {
        fma(a.y, b.z, -__dmul_rn(a.z, b.y)),
        fma(a.z, b.x, -__dmul_rn(a.x, b.z)),
        fma(a.x, b.y, -__dmul_rn(a.y, b.x))
    };
}

// a & b as the host compiles it: the y product rounded, x fused onto it, z fused last
__device__
inline scalar geoDot(
    const GeoVec& a,
    const GeoVec& b)
{
    return fma(a.z, b.z, fma(a.x, b.x, __dmul_rn(a.y, b.y)));
}

// primitiveMeshTools::updateFaceCentresAndAreas, a face a thread, and |Sf|
__global__
void geoFaceKernel(
    label nF,
    const label* faceOffsets,
    const label* faceVerts,
    const scalar* points,
    scalar* Cf,
    scalar* Sf,
    scalar* magSf)
{
    const label f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF)
    {
        return;
    }
    const label* verts = faceVerts + faceOffsets[f];
    const label k = faceOffsets[f + 1] - faceOffsets[f];
    const scalar third = 1.0/3.0;
    GeoVec centre;
    GeoVec area;
    if (k == 3)
    {
        const GeoVec a = geoAt(points, verts[0]);
        const GeoVec b = geoAt(points, verts[1]);
        const GeoVec c = geoAt(points, verts[2]);
        // triangle::centre and areaNormal: (1/3)*(sum), not sum/3
        centre = GeoVec
        {
            __dmul_rn((a.x + b.x) + c.x, third),
            __dmul_rn((a.y + b.y) + c.y, third),
            __dmul_rn((a.z + b.z) + c.z, third)
        };
        const GeoVec n = geoCross(geoSub(b, a), geoSub(c, a));
        area = GeoVec{__dmul_rn(n.x, 0.5), __dmul_rn(n.y, 0.5), __dmul_rn(n.z, 0.5)};
    }
    else
    {
        GeoVec fCentre = geoAt(points, verts[0]);
        for (label pi = 1; pi < k; ++pi)
        {
            const GeoVec q = geoAt(points, verts[pi]);
            fCentre.x += q.x;
            fCentre.y += q.y;
            fCentre.z += q.z;
        }
        const scalar count = static_cast<scalar>(k);
        fCentre.x = fCentre.x/count;
        fCentre.y = fCentre.y/count;
        fCentre.z = fCentre.z/count;
        GeoVec sumN{0, 0, 0};
        scalar sumA = 0;
        GeoVec sumAc{0, 0, 0};
        for (label pi = 0; pi < k; ++pi)
        {
            const GeoVec thisP = geoAt(points, verts[pi]);
            const GeoVec nextP = geoAt(points, verts[pi == k - 1 ? 0 : pi + 1]);
            const GeoVec c
            {
                (thisP.x + nextP.x) + fCentre.x,
                (thisP.y + nextP.y) + fCentre.y,
                (thisP.z + nextP.z) + fCentre.z
            };
            const GeoVec n = geoCross(geoSub(nextP, thisP), geoSub(fCentre, thisP));
            const scalar a = sqrt(fma(n.z, n.z, fma(n.x, n.x, __dmul_rn(n.y, n.y))));
            sumN.x += n.x;
            sumN.y += n.y;
            sumN.z += n.z;
            sumA += a;
            sumAc.x = fma(a, c.x, sumAc.x);
            sumAc.y = fma(a, c.y, sumAc.y);
            sumAc.z = fma(a, c.z, sumAc.z);
        }
        if (sumA < geoRootVSmall)
        {
            centre = fCentre;
            area = GeoVec{0, 0, 0};
        }
        else
        {
            // (1.0/3.0)*sumAc/sumA, the scaling BEFORE the division
            centre = GeoVec
            {
                __dmul_rn(sumAc.x, third)/sumA,
                __dmul_rn(sumAc.y, third)/sumA,
                __dmul_rn(sumAc.z, third)/sumA
            };
            area = GeoVec{__dmul_rn(sumN.x, 0.5), __dmul_rn(sumN.y, 0.5), __dmul_rn(sumN.z, 0.5)};
        }
    }
    geoPut(Cf, f, centre);
    geoPut(Sf, f, area);
    magSf[f] = sqrt(fma(area.z, area.z, fma(area.x, area.x, __dmul_rn(area.y, area.y))));
}

// primitiveMeshTools::updateCellCentresAndVols, a cell a thread: the estimated centre from its faces' centres,
// then the pyramids, each over the cell's faces in the host's order
__global__
void geoCellKernel(
    label nC,
    const label* cellStart,
    const label* cellFaces,
    const label* own,
    const scalar* Cf,
    const scalar* Sf,
    bool unfused,
    scalar* C,
    scalar* V)
{
    const label c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC)
    {
        return;
    }
    const label s = cellStart[c];
    const label e = cellStart[c + 1];
    GeoVec est{0, 0, 0};
    for (label k = s; k < e; ++k)
    {
        const GeoVec fc = geoAt(Cf, cellFaces[k]);
        est.x += fc.x;
        est.y += fc.y;
        est.z += fc.z;
    }
    const scalar count = static_cast<scalar>(e - s);
    est.x = est.x/count;
    est.y = est.y/count;
    est.z = est.z/count;
    GeoVec centre{0, 0, 0};
    scalar vol = 0;
    for (label k = s; k < e; ++k)
    {
        const label f = cellFaces[k];
        const GeoVec fc = geoAt(Cf, f);
        const GeoVec sf = geoAt(Sf, f);
        // the owner's pyramid points out of the cell, the neighbour's into it
        const GeoVec d = (own[f] == c) ? geoSub(fc, est) : geoSub(est, fc);
        // the control rounds both fused products first: in a 2-D mesh one of the three terms is always zero,
        // and rounding only the last (z) one changed nothing there
        const scalar xy = unfused
            ? __dadd_rn(__dmul_rn(sf.x, d.x), __dmul_rn(sf.y, d.y))
            : fma(sf.x, d.x, __dmul_rn(sf.y, d.y));
        const scalar pyr3Vol = unfused ? __dadd_rn(__dmul_rn(sf.z, d.z), xy) : fma(sf.z, d.z, xy);
        const GeoVec pc
        {
            fma(fc.x, 0.75, __dmul_rn(est.x, 0.25)),
            fma(fc.y, 0.75, __dmul_rn(est.y, 0.25)),
            fma(fc.z, 0.75, __dmul_rn(est.z, 0.25))
        };
        centre.x = fma(pc.x, pyr3Vol, centre.x);
        centre.y = fma(pc.y, pyr3Vol, centre.y);
        centre.z = fma(pc.z, pyr3Vol, centre.z);
        vol += pyr3Vol;
    }
    if (fabs(vol) > geoVSmall)
    {
        centre.x = centre.x/vol;
        centre.y = centre.y/vol;
        centre.z = centre.z/vol;
    }
    else
    {
        centre = est;
    }
    geoPut(C, c, centre);
    // the separate final pass, as in OpenFOAM
    V[c] = __dmul_rn(vol, 1.0/3.0);
}

// basicFvGeometryScheme on an internal face: weights, deltaCoeffs, nonOrthDeltaCoeffs, nonOrthCorrectionVectors
__global__
void geoInterpolationKernel(
    label nIf,
    const label* own,
    const label* nei,
    const scalar* Cf,
    const scalar* Sf,
    const scalar* magSf,
    const scalar* C,
    scalar* weights,
    scalar* deltaCoeffs,
    scalar* nonOrthDeltaCoeffs,
    scalar* nonOrthCorr)
{
    const label f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf)
    {
        return;
    }
    const GeoVec fc = geoAt(Cf, f);
    const GeoVec sf = geoAt(Sf, f);
    const GeoVec co = geoAt(C, own[f]);
    const GeoVec cn = geoAt(C, nei[f]);
    const scalar SfdOwn = fabs(geoDot(sf, geoSub(fc, co)));
    const scalar SfdNei = fabs(geoDot(sf, geoSub(cn, fc)));
    const scalar both = SfdOwn + SfdNei;
    weights[f] = (both > geoRootVSmall) ? SfdNei/both : 0.5;
    const GeoVec delta = geoSub(cn, co);
    const scalar magDelta = sqrt(fma(delta.z, delta.z, fma(delta.x, delta.x, __dmul_rn(delta.y, delta.y))));
    deltaCoeffs[f] = 1.0/magDelta;
    const scalar area = magSf[f];
    const GeoVec unitArea{sf.x/area, sf.y/area, sf.z/area};
    const scalar nodc = 1.0/fmax(geoDot(unitArea, delta), __dmul_rn(magDelta, 0.05));
    nonOrthDeltaCoeffs[f] = nodc;
    geoPut(nonOrthCorr, f, GeoVec
    {
        fma(-nodc, delta.x, unitArea.x),
        fma(-nodc, delta.y, unitArea.y),
        fma(-nodc, delta.z, unitArea.z)
    });
}

template <typename T>
void geoDown(
    const DeviceBuffer<scalar>& d,
    std::vector<T>& h,
    std::size_t n)
{
    h.resize(n);
    if (n > 0)
    {
        cudaCheck(cudaMemcpy(h.data(), d.data(), n*sizeof(T), cudaMemcpyDeviceToHost), WHO);
    }
}

} // namespace


void deviceFvGeometry(
    const PrimitiveMesh& m,
    unsigned long long topology,
    DeviceFvGeometry& w,
    FvGeometry& g)
{
    static_assert(sizeof(vector) == 3*sizeof(scalar), "a vector is read here as three scalars");
    const label nF = m.nFaces();
    const label nIf = m.nInternalFaces();
    const label nC = m.nCells();
    const label nP = m.nPoints();
    interPhase::Nested timed("geometry: faces, cells and interpolation (device)");
    static const bool unfused = std::getenv("BRAE_CONTROL_GEOMETRY_UNFUSED") != nullptr;
    const std::size_t sF = static_cast<std::size_t>(nF);
    const std::size_t sIf = static_cast<std::size_t>(nIf);
    const std::size_t sC = static_cast<std::size_t>(nC);
    if (!w.built || w.topology != topology || w.nF != nF || w.nIf != nIf || w.nC != nC || w.nP != nP)
    {
        const std::vector<label>& own = m.owner();
        const std::vector<label>& nei = m.neighbour();
        if (own.size() != sF || nei.size() < sIf)
        {
            throw std::runtime_error(std::string(WHO) + "the mesh's owner and neighbour lists are not its faces'.");
        }
        // primitiveMesh::cells(): per cell the faces it owns, ascending, then those it neighbours, ascending
        std::vector<label> start(sC + 1, 0);
        for (label f = 0; f < nF; ++f)
        {
            ++start[static_cast<std::size_t>(own[static_cast<std::size_t>(f)]) + 1];
        }
        for (label f = 0; f < nIf; ++f)
        {
            ++start[static_cast<std::size_t>(nei[static_cast<std::size_t>(f)]) + 1];
        }
        for (std::size_t c = 0; c < sC; ++c)
        {
            start[c + 1] += start[c];
        }
        std::vector<label> faces(static_cast<std::size_t>(start[sC]));
        std::vector<label> next(start.begin(), start.end() - 1);
        for (label f = 0; f < nF; ++f)
        {
            faces[static_cast<std::size_t>(next[static_cast<std::size_t>(own[static_cast<std::size_t>(f)])]++)] = f;
        }
        for (label f = 0; f < nIf; ++f)
        {
            faces[static_cast<std::size_t>(next[static_cast<std::size_t>(nei[static_cast<std::size_t>(f)])]++)] = f;
        }
        w.faceOffsets.copyFrom(m.faceOffsets());
        w.faceVerts.copyFrom(m.faceVerts());
        w.own.copyFrom(own);
        w.nei.copyFrom(std::vector<label>(nei.begin(), nei.begin() + nIf));
        w.cellStart.copyFrom(start);
        w.cellFaces.copyFrom(faces);
        w.points.resize(3*static_cast<std::size_t>(nP));
        w.Cf.resize(3*sF);
        w.Sf.resize(3*sF);
        w.magSf.resize(sF);
        w.C.resize(3*sC);
        w.V.resize(sC);
        w.weights.resize(sIf);
        w.deltaCoeffs.resize(sIf);
        w.nonOrthDeltaCoeffs.resize(sIf);
        w.nonOrthCorr.resize(3*sIf);
        w.nF = nF;
        w.nIf = nIf;
        w.nC = nC;
        w.nP = nP;
        w.topology = topology;
        w.built = true;
    }
    cudaCheck(cudaMemcpy(w.points.data(), m.points().data(), 3*static_cast<std::size_t>(nP)*sizeof(scalar),
                         cudaMemcpyHostToDevice), WHO);
    geoFaceKernel<<<geoBlocks(nF), geoTPB, 0, cudaStreamPerThread>>>(
        nF,
        w.faceOffsets.data(),
        w.faceVerts.data(),
        w.points.data(),
        w.Cf.data(),
        w.Sf.data(),
        w.magSf.data());
    geoCellKernel<<<geoBlocks(nC), geoTPB, 0, cudaStreamPerThread>>>(
        nC,
        w.cellStart.data(),
        w.cellFaces.data(),
        w.own.data(),
        w.Cf.data(),
        w.Sf.data(),
        unfused,
        w.C.data(),
        w.V.data());
    if (nIf > 0)
    {
        geoInterpolationKernel<<<geoBlocks(nIf), geoTPB, 0, cudaStreamPerThread>>>(
            nIf,
            w.own.data(),
            w.nei.data(),
            w.Cf.data(),
            w.Sf.data(),
            w.magSf.data(),
            w.C.data(),
            w.weights.data(),
            w.deltaCoeffs.data(),
            w.nonOrthDeltaCoeffs.data(),
            w.nonOrthCorr.data());
    }
    cudaCheck(cudaGetLastError(), WHO);
    ++w.count;
    // down into the arrays the last adopt handed back, and g takes them
    FvGeometry::Built& h = w.host;
    geoDown(w.Cf, h.Cf, sF);
    geoDown(w.Sf, h.Sf, sF);
    geoDown(w.magSf, h.magSf, sF);
    geoDown(w.C, h.C, sC);
    geoDown(w.V, h.V, sC);
    geoDown(w.weights, h.weights, sIf);
    geoDown(w.deltaCoeffs, h.deltaCoeffs, sIf);
    geoDown(w.nonOrthDeltaCoeffs, h.nonOrthDeltaCoeffs, sIf);
    geoDown(w.nonOrthCorr, h.nonOrthCorr, sIf);
    g.adopt(h, m);
    w.hostGeneration = g.generation();
}

namespace {

// the internal faces' share of the device mesh: the face-to-cell offsets and the correction vectors by component
__global__
void geoMeshInternalKernel(
    label nIf,
    const label* own,
    const label* nei,
    const scalar* Cf,
    const scalar* C,
    const scalar* nonOrthCorr,
    scalar* dOwnX,
    scalar* dOwnY,
    scalar* dOwnZ,
    scalar* dNeiX,
    scalar* dNeiY,
    scalar* dNeiZ,
    scalar* corrX,
    scalar* corrY,
    scalar* corrZ)
{
    const label f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf)
    {
        return;
    }
    const GeoVec fc = geoAt(Cf, f);
    const GeoVec dO = geoSub(fc, geoAt(C, own[f]));
    const GeoVec dN = geoSub(fc, geoAt(C, nei[f]));
    const GeoVec cv = geoAt(nonOrthCorr, f);
    dOwnX[f] = dO.x;
    dOwnY[f] = dO.y;
    dOwnZ[f] = dO.z;
    dNeiX[f] = dN.x;
    dNeiY[f] = dN.y;
    dNeiZ[f] = dN.z;
    corrX[f] = cv.x;
    corrY[f] = cv.y;
    corrZ[f] = cv.z;
}

// Sf and |Sf| in the device mesh's layout: the internal faces, then its boundary slots
__global__
void geoMeshAreaKernel(
    label nAll,
    label nIf,
    const label* bndFace,
    const scalar* Sf,
    const scalar* magSf,
    scalar* Sfx,
    scalar* Sfy,
    scalar* Sfz,
    scalar* magSfOut)
{
    const label j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= nAll)
    {
        return;
    }
    const label f = (j < nIf) ? j : bndFace[j - nIf];
    const GeoVec a = geoAt(Sf, f);
    Sfx[j] = a.x;
    Sfy[j] = a.y;
    Sfz[j] = a.z;
    magSfOut[j] = magSf[f];
}

// a boundary slot's offset from its cell: Cf - C(faceCell)
__global__
void geoMeshBoundaryKernel(
    label nB,
    const label* bndFace,
    const label* bndCell,
    const scalar* Cf,
    const scalar* C,
    scalar* dBndX,
    scalar* dBndY,
    scalar* dBndZ)
{
    const label k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB)
    {
        return;
    }
    const GeoVec d = geoSub(geoAt(Cf, bndFace[k]), geoAt(C, bndCell[k]));
    dBndX[k] = d.x;
    dBndY[k] = d.y;
    dBndZ[k] = d.z;
}

} // namespace

bool refreshDeviceMeshFromDeviceGeometry(
    DeviceMesh& dm,
    DeviceFvGeometry& w,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    if (!w.built || w.hostGeneration != g.generation() || w.nC != dm.nCells || w.nIf != dm.nInternalFaces)
    {
        return false;
    }
    const label nIf = dm.nInternalFaces;
    const label nB = dm.nBndFaces;
    const std::size_t sIf = static_cast<std::size_t>(nIf);
    const std::size_t sB = static_cast<std::size_t>(nB);
    const std::size_t sC = static_cast<std::size_t>(dm.nCells);
    if (w.bndFaceAddressing != dm.addressingId || w.bndFace.size() != sB)
    {
        // buildDeviceMesh's boundary numbering: the patches that are not coupled interfaces, in order
        std::vector<label> face;
        face.reserve(sB);
        for (const FvPatch& p : fvp)
        {
            if (isCoupledInterfaceType(p.type)) continue;
            for (label i = 0; i < p.size; ++i)
            {
                face.push_back(p.start + i);
            }
        }
        if (face.size() != sB)
        {
            return false;
        }
        w.bndFace.copyFrom(face);
        w.bndFaceAddressing = dm.addressingId;
    }
    if (dm.w.size() != sIf || dm.dc.size() != sIf || dm.nonOrthDc.size() != sIf || dm.V.size() != sC
     || dm.Sfx.size() != sIf + sB || dm.magSf.size() != sIf + sB || dm.dOwnX.size() != sIf
     || dm.dBndX.size() != sB || dm.corrVecX.size() != sIf)
    {
        return false;
    }
    auto copy = [](
        DeviceBuffer<scalar>& to,
        const DeviceBuffer<scalar>& from,
        std::size_t n)
    {
        if (n == 0) return;
        cudaCheck(cudaMemcpyAsync(to.data(), from.data(), n*sizeof(scalar), cudaMemcpyDeviceToDevice,
                                  cudaStreamPerThread), WHO);
    };
    // primitiveMesh::clearGeom's and surfaceInterpolation::clearOut's, as refreshDeviceMeshGeometry lists them
    copy(dm.V, w.V, sC);
    copy(dm.w, w.weights, sIf);
    copy(dm.dc, w.deltaCoeffs, sIf);
    copy(dm.nonOrthDc, w.nonOrthDeltaCoeffs, sIf);
    // BRAE_CONTROL_DEVICE_MESH_STALE_OFFSETS=1 leaves the internal faces' offsets and correction vectors as the
    // last mesh's -- the identity gate's control
    static const bool staleOffsets = std::getenv("BRAE_CONTROL_DEVICE_MESH_STALE_OFFSETS") != nullptr;
    if (nIf > 0 && !staleOffsets)
    {
        geoMeshInternalKernel<<<geoBlocks(nIf), geoTPB, 0, cudaStreamPerThread>>>(
            nIf,
            dm.owner.data(),
            dm.nei.data(),
            w.Cf.data(),
            w.C.data(),
            w.nonOrthCorr.data(),
            dm.dOwnX.data(),
            dm.dOwnY.data(),
            dm.dOwnZ.data(),
            dm.dNeiX.data(),
            dm.dNeiY.data(),
            dm.dNeiZ.data(),
            dm.corrVecX.data(),
            dm.corrVecY.data(),
            dm.corrVecZ.data());
    }
    geoMeshAreaKernel<<<geoBlocks(nIf + nB), geoTPB, 0, cudaStreamPerThread>>>(
        nIf + nB,
        nIf,
        w.bndFace.data(),
        w.Sf.data(),
        w.magSf.data(),
        dm.Sfx.data(),
        dm.Sfy.data(),
        dm.Sfz.data(),
        dm.magSf.data());
    if (nB > 0)
    {
        geoMeshBoundaryKernel<<<geoBlocks(nB), geoTPB, 0, cudaStreamPerThread>>>(
            nB,
            w.bndFace.data(),
            dm.bndCell.data(),
            w.Cf.data(),
            w.C.data(),
            dm.dBndX.data(),
            dm.dBndY.data(),
            dm.dBndZ.data());
    }
    cudaCheck(cudaGetLastError(), WHO);
    // leastSquaresVectors: a move invalidates the cached tensor, as the host's refresh has it
    dm.lsqInvDd.resize(0);
    return true;
}

} // namespace brae
