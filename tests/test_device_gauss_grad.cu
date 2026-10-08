// The device Gauss gradient against the host's fvc::gaussGrad, BIT FOR BIT.
//
// The device kernels (gradKernel, gradFusedKernel in src/cuda/device_fvc.cu) are transcriptions of the
// host operator, which is OpenFOAM's gaussGrad::gradf: each face's value in OpenFOAM's arithmetic,
// lambda*(P - N) + N, and each cell's internal faces summed in face order, then its boundary faces in
// patch order, then the division by V. Neither is cosmetic. On a uniform field the gradient IS the
// rounding of those sums, and its sign decides a TVD limiter at every face where the field does not
// change (see fvc.cu). MEASURED before the transcription, on this fixture: 912 of 960 cells differed in
// some bit on the smooth field and 934 on the uniform one, where the difference (3.9e-27) was as large
// as the gradient.
//
// THE CONTROLS are the two things the device used to do, computed on the host: the face value as
// w*P + (1 - w)*N, and the owner faces summed before the neighbour faces. Each must differ from the
// host's gradient on this fixture, or the bit-identity arm could not tell the transcription from them.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "device_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

// cells whose gradient differs from the host's in any bit of any component
long bitDiffs(
    const std::vector<vector>& host,
    const std::vector<scalar>& x,
    const std::vector<scalar>& y,
    const std::vector<scalar>& z)
{
    long n = 0;
    for (std::size_t c = 0; c < host.size(); ++c)
    {
        if (std::memcmp(&x[c], &host[c].x, sizeof(scalar)) != 0
         || std::memcmp(&y[c], &host[c].y, sizeof(scalar)) != 0
         || std::memcmp(&z[c], &host[c].z, sizeof(scalar)) != 0)
        {
            ++n;
        }
    }
    return n;
}

long bitDiffs(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    long n = 0;
    for (std::size_t c = 0; c < a.size(); ++c)
    {
        if (std::memcmp(&a[c], &b[c], sizeof(vector)) != 0) ++n;
    }
    return n;
}

// the host's gaussGrad with ONE thing changed, for the controls
std::vector<vector> variantGrad(
    const std::vector<scalar>& v,
    const std::vector<std::vector<scalar>>& b,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    bool weightedForm,
    bool ownerFirst)
{
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w = g.weights();
    const std::vector<vector>& Sf = g.Sf();
    auto faceValue = [&](label f)
    {
        return weightedForm ? w[f] * v[own[f]] + (1.0 - w[f]) * v[nei[f]]
                            : w[f] * (v[own[f]] - v[nei[f]]) + v[nei[f]];
    };
    std::vector<vector> grad(static_cast<std::size_t>(nC), vector{0, 0, 0});
    if (ownerFirst)
    {
        for (label f = 0; f < nIf; ++f) grad[own[f]] += Sf[f] * faceValue(f);
        for (label f = 0; f < nIf; ++f) grad[nei[f]] = grad[nei[f]] - Sf[f] * faceValue(f);
    }
    else
    {
        for (label f = 0; f < nIf; ++f)
        {
            const vector s = Sf[f] * faceValue(f);
            grad[own[f]] += s;
            grad[nei[f]] = grad[nei[f]] - s;
        }
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "empty") continue;
        for (label i = 0; i < patches[pi].size; ++i)
            grad[patches[pi].faceCells[i]] += Sf[patches[pi].start + i] * b[pi][i];
    }
    for (label c = 0; c < nC; ++c) grad[c] = grad[c] / g.V()[c];
    return grad;
}
}   // namespace

int main()
{
    std::printf("== the device Gauss gradient against the host's, bit for bit\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    // sheared, with three different cell sizes, so no face sum cancels by symmetry
    const PrimitiveMesh m = boxtest::boxMesh(12, 10, 8, scalar(0.35), scalar(0.7), scalar(1.3), scalar(0.9));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells();

    for (int kind = 0; kind < 2; ++kind)
    {
        const char* name = (kind == 0) ? "smooth" : "uniform";
        std::vector<scalar> v(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            const vector& C = g.C()[c];
            v[c] = (kind == 0) ? std::sin(C.x) * std::cos(scalar(0.7) * C.y) + scalar(0.3) * C.z : scalar(1e-11);
        }
        std::vector<std::vector<scalar>> b(fvp.size());
        std::vector<scalar> bflat;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                // a boundary value that is not the face cell's, so the boundary sum is not a copy
                const scalar bv = v[fvp[pi].faceCells[i]] * ((kind == 0) ? scalar(1.1) : scalar(1));
                b[pi].push_back(bv);
                bflat.push_back(bv);
            }
        }
        const std::vector<vector> host = fvc::gaussGrad(v, b, m, g, fvp);

        DeviceBuffer<scalar> dv(v), db(bflat), gx, gy, gz;
        deviceGaussGrad(dm, dv, db, gx, gy, gz);
        std::vector<scalar> x, y, z;
        gx.copyTo(x);
        gy.copyTo(y);
        gz.copyTo(z);
        const long d1 = bitDiffs(host, x, y, z);
        std::printf("  %s: deviceGaussGrad differs from the host in %ld of %d cells\n", name, d1, (int)nC);
        check("deviceGaussGrad is the host's gaussGrad, bit for bit", d1 == 0);

        // the fused kernel, with the field in every slot
        for (int n = 1; n <= 3; ++n)
        {
            const DeviceBuffer<scalar>* vol[3] = {&dv, &dv, &dv};
            const DeviceBuffer<scalar>* bv[3] = {&db, &db, &db};
            DeviceBuffer<scalar> fx[3], fy[3], fz[3];
            deviceGaussGradFused(dm, n, vol, bv, fx, fy, fz);
            long dn = 0;
            for (int i = 0; i < n; ++i)
            {
                std::vector<scalar> a, bb, cc;
                fx[i].copyTo(a);
                fy[i].copyTo(bb);
                fz[i].copyTo(cc);
                dn += bitDiffs(host, a, bb, cc);
            }
            std::printf("  %s: deviceGaussGradFused(%d) differs in %ld cell-fields\n", name, n, dn);
            check("...and so is the fused kernel", dn == 0);
        }

        // THE CONTROLS: the two forms the device used to have
        const std::vector<vector> weighted = variantGrad(v, b, m, g, fvp, true, false);
        const std::vector<vector> ownerFirst = variantGrad(v, b, m, g, fvp, false, true);
        const std::vector<vector> same = variantGrad(v, b, m, g, fvp, false, false);
        const long cW = bitDiffs(weighted, host);
        const long cO = bitDiffs(ownerFirst, host);
        const long cS = bitDiffs(same, host);
        std::printf("  %s: CONTROLS against the host -- w*P + (1 - w)*N %ld cells, owner faces first %ld, "
                    "the host's own form rewritten %ld\n", name, cW, cO, cS);
        check("the control's rewrite of the host form is the host's (so the controls change one thing)", cS == 0);
        check("the w*P + (1 - w)*N face value differs on this fixture", cW > 0);
        check("summing the owner faces first differs on this fixture", cO > 0);
    }

    std::printf("test_device_gauss_grad: %d failures\n", failures);
    return failures ? 1 : 0;
}
