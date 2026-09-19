// The device laplacian's kernels against the host's fvm.cuh, BIT FOR BIT, on a mesh that is not orthogonal.
//
// Each device kernel in src/cuda/device_fvm.cu is a transcription of a host operator:
//   deviceLaplacianCoeffs            <- fvm::laplacian's internal half (orthogonal and `corrected`)
//   deviceLaplacianCorrFlux          <- fvm::laplacianCorrFlux, unlimited
//   deviceLaplacianCorrFluxLimited   <- fvm::laplacianCorrFlux, `limited <k> corrected`
//   deviceLaplacianCorrFluxLimitedVec<- the same for a vector field (one limiter per face)
//   deviceFaceDivSource              <- fvm::laplacianNonOrthSource, with the sign flipped
//   deviceDiv                        <- fvc::div
// Every kernel is handed the HOST's gradient, so what is compared is the kernel and nothing upstream.
//
// MEASURED before the transcription: the diagonal and the correction's per-cell sum were accumulated
// owner faces first, then neighbour faces, where the host sums in face order; and the limited fluxes
// multiplied (gamma*magSf*limiter)*corr where the host forms (gamma*magSf)*(limiter*corr). THE CONTROLS
// recompute the owner-first sums on the host and must differ from the host's own on this fixture.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "fvm.cuh"
#include "device_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

long bitDiffs(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    if (a.size() != b.size()) return -1;
    long n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::memcmp(&a[i], &b[i], sizeof(scalar)) != 0) ++n;
    }
    return n;
}

std::vector<scalar> fromDevice(const DeviceBuffer<scalar>& d)
{
    std::vector<scalar> h;
    d.copyTo(h);
    return h;
}
}   // namespace

int main()
{
    std::printf("== the device laplacian's kernels against the host's, bit for bit\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const PrimitiveMesh m = boxtest::boxMesh(12, 10, 8, scalar(0.35), scalar(0.7), scalar(1.3), scalar(0.9));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    // a field, its host gradient, and a non-uniform face diffusivity
    auto scalarField = [&](int k)
    {
        auto f = std::make_unique<GeometricField<scalar>>();
        f->internal.resize(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            const vector& C = g.C()[c];
            f->internal[c] = std::sin(C.x + scalar(k)) * std::cos(scalar(0.7) * C.y) + scalar(0.3 + 0.1*k) * C.z;
        }
        for (const FvPatch& q : fvp) f->boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        f->evaluateBoundary();
        return f;
    };
    const auto vf = scalarField(0);
    SurfaceScalarField gammaf;
    gammaf.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f) gammaf.internal[f] = scalar(1) + scalar(0.5) * std::sin(scalar(0.37) * scalar(f));
    gammaf.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) gammaf.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(1));
    const std::vector<vector> grad = fvc::gaussGrad(*vf, m, g, fvp);
    std::vector<scalar> gxh(nC), gyh(nC), gzh(nC);
    for (label c = 0; c < nC; ++c) { gxh[c] = grad[c].x; gyh[c] = grad[c].y; gzh[c] = grad[c].z; }
    DeviceBuffer<scalar> dGamma(gammaf.internal), dVf(vf->internal), dGx(gxh), dGy(gyh), dGz(gzh);

    // ---- 1. the coefficients, orthogonal and corrected ---------------------------------------------
    for (int corrected = 0; corrected < 2; ++corrected)
    {
        const FvScalarMatrix M = fvm::laplacian<scalar>(gammaf, *vf, m, g, fvp, corrected != 0);
        DeviceBuffer<scalar> dd, du, dl;
        deviceLaplacianCoeffs(dm, dGamma, dd, du, dl, corrected != 0);
        const long nd = bitDiffs(fromDevice(dd), M.diag);
        const long nu = bitDiffs(fromDevice(du), M.upper);
        const long nl = bitDiffs(fromDevice(dl), M.lower);
        std::printf("  %s coefficients: diag %ld, upper %ld, lower %ld differ\n",
                    corrected ? "corrected" : "orthogonal", nd, nu, nl);
        check("the laplacian's internal coefficients are the host's, bit for bit", nd == 0 && nu == 0 && nl == 0);
    }

    // ---- 2. the correction flux, unlimited and limited ---------------------------------------------
    const std::vector<scalar> ffcHost = fvm::laplacianCorrFlux<scalar, vector>(gammaf, grad, m, g, 0.0, vf.get());
    {
        DeviceBuffer<scalar> ffc;
        deviceLaplacianCorrFlux(dm, dGamma, dGx, dGy, dGz, ffc);
        const long n = bitDiffs(fromDevice(ffc), ffcHost);
        std::printf("  unlimited correction flux: %ld of %d faces differ\n", n, (int)nIf);
        check("the correction flux is the host's, bit for bit", n == 0);
    }
    {
        const scalar k = scalar(0.5);
        const std::vector<scalar> host = fvm::laplacianCorrFlux<scalar, vector>(gammaf, grad, m, g, k, vf.get());
        DeviceBuffer<scalar> ffc;
        deviceLaplacianCorrFluxLimited(dm, dGamma, dVf, dGx, dGy, dGz, k, ffc);
        const long n = bitDiffs(fromDevice(ffc), host);
        long limitedFaces = 0;
        for (label f = 0; f < nIf; ++f) limitedFaces += (host[f] != ffcHost[f]) ? 1 : 0;
        std::printf("  limited 0.5 correction flux: %ld of %d faces differ (%ld faces limited)\n", n, (int)nIf, limitedFaces);
        check("the limited correction flux is the host's, bit for bit", n == 0);
        check("...on a fixture where the limiter acts", limitedFaces > 0);
    }

    // ---- 3. the vector form ------------------------------------------------------------------------
    {
        std::unique_ptr<GeometricField<scalar>> comp[3] = {scalarField(0), scalarField(1), scalarField(2)};
        GeometricField<vector> U;
        U.internal.resize(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) U.internal[c] = vector{comp[0]->internal[c], comp[1]->internal[c], comp[2]->internal[c]};
        for (const FvPatch& q : fvp) U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
        U.evaluateBoundary();
        std::vector<vector> gc[3];
        for (int i = 0; i < 3; ++i) gc[i] = fvc::gaussGrad(*comp[i], m, g, fvp);
        std::vector<tensor> gradU(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            // grad(U)_ij = d(U_j)/d(x_i)
            gradU[c] = tensor{gc[0][c].x, gc[1][c].x, gc[2][c].x,
                              gc[0][c].y, gc[1][c].y, gc[2][c].y,
                              gc[0][c].z, gc[1][c].z, gc[2][c].z};
        }
        const scalar k = scalar(0.5);
        const std::vector<vector> host = fvm::laplacianCorrFlux<vector, tensor>(gammaf, gradU, m, g, k, &U);
        DeviceBuffer<scalar> dp[3], dgx[3], dgy[3], dgz[3], ffc[3];
        for (int i = 0; i < 3; ++i)
        {
            dp[i].copyFrom(comp[i]->internal);
            std::vector<scalar> x(nC), y(nC), z(nC);
            for (label c = 0; c < nC; ++c) { x[c] = gc[i][c].x; y[c] = gc[i][c].y; z[c] = gc[i][c].z; }
            dgx[i].copyFrom(x);
            dgy[i].copyFrom(y);
            dgz[i].copyFrom(z);
        }
        deviceLaplacianCorrFluxLimitedVec(dm, dGamma, dp[0], dp[1], dp[2], dgx, dgy, dgz, k, ffc);
        long n = 0;
        for (int i = 0; i < 3; ++i)
        {
            std::vector<scalar> h(static_cast<std::size_t>(nIf));
            for (label f = 0; f < nIf; ++f) h[f] = (i == 0) ? host[f].x : (i == 1) ? host[f].y : host[f].z;
            n += bitDiffs(fromDevice(ffc[i]), h);
        }
        std::printf("  vector limited correction flux: %ld face-components differ\n", n);
        check("the vector correction flux is the host's, bit for bit", n == 0);
    }

    // ---- 4. the per-cell source --------------------------------------------------------------------
    {
        const std::vector<scalar> host = fvm::laplacianNonOrthSource<scalar, vector>(gammaf, *vf, grad, m, g, fvp, 0.0);
        DeviceBuffer<scalar> dFfc(ffcHost), src;
        deviceFaceDivSource(dm, dFfc, src);
        std::vector<scalar> neg = fromDevice(src);
        for (scalar& x : neg) x = -x;
        const long n = bitDiffs(neg, host);
        std::printf("  correction source (sign flipped): %ld of %d cells differ\n", n, (int)nC);
        check("the correction's per-cell source is the host's, bit for bit", n == 0);

        // THE CONTROLS: the owner-first sums the device used to form
        std::vector<scalar> ownerFirst(static_cast<std::size_t>(nC), scalar(0));
        for (label f = 0; f < nIf; ++f) ownerFirst[own[f]] += ffcHost[f];
        for (label f = 0; f < nIf; ++f) ownerFirst[nei[f]] -= ffcHost[f];
        const FvScalarMatrix M = fvm::laplacian<scalar>(gammaf, *vf, m, g, fvp, true);
        std::vector<scalar> diagOwnerFirst(static_cast<std::size_t>(nC), scalar(0));
        for (label f = 0; f < nIf; ++f) diagOwnerFirst[own[f]] -= M.lower[f];
        for (label f = 0; f < nIf; ++f) diagOwnerFirst[nei[f]] -= M.upper[f];
        const long cS = bitDiffs(ownerFirst, host);
        const long cD = bitDiffs(diagOwnerFirst, M.diag);
        std::printf("  CONTROLS: owner faces first -- source %ld cells, diagonal %ld cells\n", cS, cD);
        check("summing the owner faces first differs on this fixture (source)", cS > 0);
        check("...and the diagonal", cD > 0);
    }

    // ---- 5. fvc::div ---------------------------------------------------------------------------------
    // the same face-order sum, on a flux that is not a difference of cell values
    {
        SurfaceScalarField phi;
        phi.internal.resize(static_cast<std::size_t>(nIf));
        for (label f = 0; f < nIf; ++f) phi.internal[f] = std::sin(scalar(0.91) * scalar(f)) * g.magSf()[f];
        phi.boundary.resize(fvp.size());
        std::vector<scalar> bflat;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const scalar v = std::cos(scalar(0.37) * scalar(fvp[pi].start + i));
                phi.boundary[pi].push_back(v);
                bflat.push_back(v);
            }
        }
        const std::vector<scalar> host = fvc::div(phi, m, g, fvp);
        DeviceBuffer<scalar> dPhi(phi.internal), dB(bflat), d;
        deviceDiv(dm, dPhi, dB, d);
        const long n = bitDiffs(fromDevice(d), host);
        // THE CONTROL: owner faces first, as the device used to sum
        std::vector<scalar> ownerFirst(static_cast<std::size_t>(nC), scalar(0));
        for (label f = 0; f < nIf; ++f) ownerFirst[own[f]] += phi.internal[f];
        for (label f = 0; f < nIf; ++f) ownerFirst[nei[f]] -= phi.internal[f];
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (fvp[pi].type == "empty") continue;
            for (label i = 0; i < fvp[pi].size; ++i) ownerFirst[fvp[pi].faceCells[i]] += phi.boundary[pi][i];
        }
        for (label c = 0; c < nC; ++c) ownerFirst[c] /= g.V()[c];
        const long cD = bitDiffs(ownerFirst, host);
        std::printf("  fvc::div: %ld of %d cells differ; CONTROL owner faces first %ld cells\n", n, (int)nC, cD);
        check("deviceDiv is the host's fvc::div, bit for bit", n == 0);
        check("summing the owner faces first differs on this fixture (div)", cD > 0);
    }

    std::printf("test_device_laplacian_vs_host: %d failures\n", failures);
    return failures ? 1 : 0;
}
