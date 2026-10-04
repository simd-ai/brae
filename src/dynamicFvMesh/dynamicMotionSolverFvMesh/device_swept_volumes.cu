#include "device_swept_volumes.cuh"
#include "inter_phase_time.cuh"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

const char* const WHO = "brae deviceSweptVolumes: ";

// face_cpp.cu's vSmall
constexpr scalar svVSmall = 1.0e-300;

constexpr int svTPB = 256;

int svBlocks(label n)
{
    return static_cast<int>((n + svTPB - 1)/svTPB);
}

struct Vec3
{
    scalar x;
    scalar y;
    scalar z;
};

__device__
inline Vec3 svPoint(
    const scalar* p,
    label i)
{
    const scalar* q = p + 3*static_cast<std::size_t>(i);
    return Vec3{q[0], q[1], q[2]};
}

__device__
inline Vec3 svSub(
    const Vec3& a,
    const Vec3& b)
{
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

// a ^ b as the host compiles it: each component's first product fused, its second rounded
__device__
inline Vec3 svCross(
    const Vec3& a,
    const Vec3& b)
{
    return Vec3
    {
        fma(a.y, b.z, -__dmul_rn(a.z, b.y)),
        fma(a.z, b.x, -__dmul_rn(a.x, b.z)),
        fma(a.x, b.y, -__dmul_rn(a.y, b.x))
    };
}

// u & (v ^ w): the y product rounded, x fused onto it, z fused last. `unfused` is the gate's control.
__device__
inline scalar svTriple(
    const Vec3& u,
    const Vec3& v,
    const Vec3& w,
    bool unfused)
{
    const Vec3 c = svCross(v, w);
    const scalar xy = fma(u.x, c.x, __dmul_rn(u.y, c.y));
    if (unfused)
    {
        return __dadd_rn(__dmul_rn(u.z, c.z), xy);
    }
    return fma(u.z, c.z, xy);
}

// face::centre(points) -- faceCentreOfPoints
__device__
inline Vec3 svFaceCentre(
    const label* verts,
    label n,
    const scalar* p)
{
    if (n == 3)
    {
        const Vec3 p0 = svPoint(p, verts[0]);
        const Vec3 p1 = svPoint(p, verts[1]);
        const Vec3 p2 = svPoint(p, verts[2]);
        const scalar third = 1.0/3.0;
        return Vec3
        {
            __dmul_rn((p0.x + p1.x) + p2.x, third),
            __dmul_rn((p0.y + p1.y) + p2.y, third),
            __dmul_rn((p0.z + p1.z) + p2.z, third)
        };
    }
    Vec3 centre{0, 0, 0};
    for (label i = 0; i < n; ++i)
    {
        const Vec3 q = svPoint(p, verts[i]);
        centre.x += q.x;
        centre.y += q.y;
        centre.z += q.z;
    }
    const scalar count = static_cast<scalar>(n);
    centre.x = centre.x/count;
    centre.y = centre.y/count;
    centre.z = centre.z/count;
    scalar sumA = 0;
    Vec3 sumAc{0, 0, 0};
    for (label i = 0; i < n; ++i)
    {
        const Vec3 thisPoint = svPoint(p, verts[i]);
        const Vec3 nextPoint = svPoint(p, verts[(i + 1)%n]);
        // 3*triangle centre
        const Vec3 ttc
        {
            (thisPoint.x + nextPoint.x) + centre.x,
            (thisPoint.y + nextPoint.y) + centre.y,
            (thisPoint.z + nextPoint.z) + centre.z
        };
        // 2*triangle area
        const Vec3 c = svCross(svSub(thisPoint, centre), svSub(nextPoint, centre));
        const scalar ta = sqrt(fma(c.z, c.z, fma(c.x, c.x, __dmul_rn(c.y, c.y))));
        sumA += ta;
        sumAc.x = fma(ta, ttc.x, sumAc.x);
        sumAc.y = fma(ta, ttc.y, sumAc.y);
        sumAc.z = fma(ta, ttc.z, sumAc.z);
    }
    if (sumA > svVSmall)
    {
        const scalar den = __dmul_rn(sumA, 3.0);
        return Vec3{sumAc.x/den, sumAc.y/den, sumAc.z/den};
    }
    return centre;
}

// triangle::sweptVol: the triangle (a, b, c) swept to (ta, tb, tc)
__device__
inline scalar svTriangle(
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const Vec3& ta,
    const Vec3& tb,
    const Vec3& tc,
    bool unfused)
{
    const scalar t1 = svTriple(svSub(ta, a), svSub(b, a), svSub(c, a), unfused);
    const scalar t2 = svTriple(svSub(tb, b), svSub(c, b), svSub(ta, b), unfused);
    const scalar t3 = svTriple(svSub(c, tc), svSub(tb, tc), svSub(ta, tc), unfused);
    const scalar t5 = svTriple(svSub(b, tb), svSub(ta, tb), svSub(tc, tb), unfused);
    const scalar t6 = svTriple(svSub(c, tc), svSub(b, tc), svSub(ta, tc), unfused);
    // the fourth term is the first again
    const scalar sum = ((((t1 + t2) + t3) + t1) + t5) + t6;
    return __dmul_rn(sum, 1.0/12.0);
}

// face::sweptVol over the faces [first, first + count), times 1/deltaT
__global__
void sweptVolumeKernel(
    label first,
    label count,
    const label* faceOffsets,
    const label* faceVerts,
    const scalar* oldPoints,
    const scalar* newPoints,
    scalar rdt,
    bool unfused,
    scalar* out)
{
    const label i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    const label f = first + i;
    const label* verts = faceVerts + faceOffsets[f];
    const label n = faceOffsets[f + 1] - faceOffsets[f];
    // a central decomposition, the centre point first in every triangle
    const Vec3 centreOld = svFaceCentre(verts, n, oldPoints);
    const Vec3 centreNew = svFaceCentre(verts, n, newPoints);
    scalar sv = 0;
    for (label k = 0; k < n; ++k)
    {
        const label p = verts[k];
        const label q = verts[(k + 1)%n];
        sv += svTriangle(
            centreOld,
            svPoint(oldPoints, p),
            svPoint(oldPoints, q),
            centreNew,
            svPoint(newPoints, p),
            svPoint(newPoints, q),
            unfused);
    }
    out[f] = __dmul_rn(sv, rdt);
}

} // namespace


void deviceSweptVolumes(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const std::vector<vector>& oldPoints,
    const std::vector<vector>& newPoints,
    scalar rdt,
    unsigned long long topology,
    DeviceSweptVolumes& w,
    SurfaceScalarField& meshPhi)
{
    static_assert(sizeof(vector) == 3*sizeof(scalar), "a point is read here as three scalars");
    const label nF = m.nFaces();
    const label nIf = m.nInternalFaces();
    const label nP = m.nPoints();
    if (oldPoints.size() != static_cast<std::size_t>(nP) || newPoints.size() != static_cast<std::size_t>(nP)
     || meshPhi.internal.size() != static_cast<std::size_t>(nIf) || meshPhi.boundary.size() != patches.size())
    {
        throw std::runtime_error(std::string(WHO) + "the points or the flux handed in are not this mesh's.");
    }
    interPhase::Nested timed("geometry: the swept volumes (device)");
    static const bool unfused = std::getenv("BRAE_CONTROL_SWEPT_VOLUME_UNFUSED") != nullptr;
    if (!w.built || w.topology != topology || w.nF != nF || w.nP != nP)
    {
        w.faceOffsets.copyFrom(m.faceOffsets());
        w.faceVerts.copyFrom(m.faceVerts());
        w.oldPoints.resize(3*static_cast<std::size_t>(nP));
        w.newPoints.resize(3*static_cast<std::size_t>(nP));
        w.out.resize(static_cast<std::size_t>(nF));
        w.nF = nF;
        w.nP = nP;
        w.topology = topology;
        w.built = true;
    }
    const std::size_t pointBytes = 3*static_cast<std::size_t>(nP)*sizeof(scalar);
    cudaCheck(cudaMemcpy(w.oldPoints.data(), oldPoints.data(), pointBytes, cudaMemcpyHostToDevice), WHO);
    cudaCheck(cudaMemcpy(w.newPoints.data(), newPoints.data(), pointBytes, cudaMemcpyHostToDevice), WHO);
    auto run = [&](label first, label count)
    {
        if (count <= 0) return;
        sweptVolumeKernel<<<svBlocks(count), svTPB, 0, cudaStreamPerThread>>>(
            first,
            count,
            w.faceOffsets.data(),
            w.faceVerts.data(),
            w.oldPoints.data(),
            w.newPoints.data(),
            rdt,
            unfused,
            w.out.data());
        cudaCheck(cudaGetLastError(), WHO);
    };
    run(0, nIf);
    for (const FvPatch& p : patches)
    {
        // Empty patches
        if (p.type == "empty") continue;
        run(p.start, p.size);
    }
    if (nIf > 0)
    {
        cudaCheck(cudaMemcpy(meshPhi.internal.data(), w.out.data(), static_cast<std::size_t>(nIf)*sizeof(scalar),
                             cudaMemcpyDeviceToHost), WHO);
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        if (p.type == "empty" || p.size <= 0) continue;
        if (meshPhi.boundary[pi].size() != static_cast<std::size_t>(p.size))
        {
            throw std::runtime_error(std::string(WHO) + "patch " + p.name + "'s flux is not the patch's size.");
        }
        cudaCheck(cudaMemcpy(meshPhi.boundary[pi].data(), w.out.data() + p.start,
                             static_cast<std::size_t>(p.size)*sizeof(scalar), cudaMemcpyDeviceToHost), WHO);
    }
}

} // namespace brae
