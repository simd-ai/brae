// interFoam's alpha fluxes on the device, against the host -- including the nested one.
//
// THE ARM WORTH HAVING IS alphaPhiUn. It is five face operations with two minus signs in the middle,
// and the upwind direction of each interpolation is set by the flux passed in -- so the negations
// change WHICH CELL each face reads, not just a sign. The host gate
// (tests/test_alpha_eqn_cpp.cu) established that those signs cancel exactly under `Gauss linear` and
// do not under `Gauss vanLeer`; this file composes the device version the same way and compares.
//
// phic, the mass flux and the plain face flux are per-face arithmetic and are held tightly; the only
// slack anywhere here is the fused multiply-add the two compilers choose differently.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "alpha_eqn_cpp.cuh"
#include "device_alpha_flux.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
scalar worstOf(const std::vector<scalar>& a, const std::vector<scalar>& b, scalar& ref)
{
    scalar w = 0; ref = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        w   = std::fmax(w,   std::fabs(a[i] - b[i]));
        ref = std::fmax(ref, std::fabs(b[i]));
    }
    return w;
}
}   // namespace

int main()
{
    std::printf("== interFoam alpha fluxes: device vs host ==\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 12;
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    // a diagonal flux and an interface that is not aligned with anything
    const vector U{scalar(0.8), scalar(0.6), scalar(0)};
    SurfaceScalarField phi, phir;
    phi.internal.resize(static_cast<std::size_t>(nIf));
    phir.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const vector& S = g.Sf()[f];
        phi.internal[f]  = U.x*S.x + U.y*S.y + U.z*S.z;
        phir.internal[f] = scalar(0.37) * phi.internal[f];
    }
    phi.boundary.resize(fvp.size());
    phir.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phi.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        phir.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
    }

    std::vector<scalar> a1v(static_cast<std::size_t>(nC)), a2v(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::hypot(C.x - scalar(N)/2, C.y - scalar(N)/2);
        a1v[c] = scalar(0.5)*(scalar(1) + std::tanh(scalar(N)/5 - r));
        a2v[c] = scalar(1) - a1v[c];
    }
    auto mkField = [&](const std::vector<scalar>& v)
    {
        GeometricField<scalar> f;
        f.internal = v;
        for (const FvPatch& q : fvp)
            f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        f.evaluateBoundary();
        return f;
    };
    const GeometricField<scalar> alpha1 = mkField(a1v);
    const GeometricField<scalar> alpha2 = mkField(a2v);

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceBuffer<scalar> dPhi(phi.internal), dPhir(phir.internal), dA1(a1v), dA2(a2v);

    // ---- 1. phic, and the boundary that must be zero ---------------------------------------------
    {
        SurfaceScalarField hPhic;
        compressionFlux(scalar(1), phi, g.magSf(), fvp, scalar(0), {}, scalar(0), {}, hPhic);

        DeviceBuffer<scalar> dPhicInt, dPhicBnd;
        deviceCompressionFlux(dm, (int)nIf, (int)nBf, dPhi, scalar(1), dPhicInt, dPhicBnd);
        cudaDeviceSynchronize();
        std::vector<scalar> pi_, pb;
        dPhicInt.copyTo(pi_);
        dPhicBnd.copyTo(pb);

        scalar ref = 0;
        const scalar w = worstOf(pi_, hPhic.internal, ref);
        std::printf("  phic: worst %.3e of %.3e\n", (double)w, (double)ref);
        check("phic matches the host", w <= scalar(4) * std::numeric_limits<scalar>::epsilon() * ref);
        check("...and is zero on every boundary face",
              *std::max_element(pb.begin(), pb.end()) == scalar(0)
           && *std::min_element(pb.begin(), pb.end()) == scalar(0));
    }

    // ---- 2. alphaPhiUn, composed the same way on both sides ---------------------------------------
    {
        // The host, through its own gated composition.
        SurfaceScalarField hUn;
        alphaPhiUn(phi, phir, alpha1, alpha2, AlphaFluxScheme::vanLeer, AlphaFluxScheme::linear,
                   m, g, fvp, hUn);

        // The device. vanLeer needs grad(alpha1) and the mesh's own CD weights serve `linear`.
        std::vector<scalar> ab1, ab2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& v1 = alpha1.boundary[pi]->value();
            const std::vector<scalar>& v2 = alpha2.boundary[pi]->value();
            ab1.insert(ab1.end(), v1.begin(), v1.end());
            ab2.insert(ab2.end(), v2.begin(), v2.end());
        }
        DeviceBuffer<scalar> dAb1(ab1), dAb2(ab2), gx(nC), gy(nC), gz(nC);
        deviceGaussGrad(dm, dA1, dAb1, gx, gy, gz);

        DeviceBuffer<scalar> wVL, wLin(dm.w.size());
        {   // `linear` weights are the mesh's own, copied so the flux kernel takes one uniform argument
            std::vector<scalar> hw;
            dm.w.copyTo(hw);
            wLin.copyFrom(hw);
        }
        deviceLimitedFaceWeights(dm, dPhi, dA1, gx, gy, gz, scalar(-1.0) /*kVanLeerTwoByk*/, wVL);

        DeviceBuffer<scalar> adv, negPhir, inner, negInner, comp;
        deviceAlphaFaceFlux(dm, (int)nIf, dPhi, wVL, dA1, adv);          // flux(phi, alpha1, vanLeer)
        deviceNegateFaces((int)nIf, dPhir, negPhir);                     // -phir
        deviceAlphaFaceFlux(dm, (int)nIf, negPhir, wLin, dA2, inner);    // flux(-phir, alpha2, linear)
        deviceNegateFaces((int)nIf, inner, negInner);                    // the OUTER minus
        deviceAlphaFaceFlux(dm, (int)nIf, negInner, wLin, dA1, comp);    // flux(that, alpha1, linear)
        cudaDeviceSynchronize();

        std::vector<scalar> hAdv, hComp;
        adv.copyTo(hAdv);
        comp.copyTo(hComp);
        std::vector<scalar> dUn(static_cast<std::size_t>(nIf));
        for (label f = 0; f < nIf; ++f) dUn[f] = hAdv[f] + hComp[f];

        scalar ref = 0;
        const scalar w = worstOf(dUn, hUn.internal, ref);
        std::printf("  alphaPhiUn: worst %.3e of %.3e (relative %.3e)\n",
                    (double)w, (double)ref, (double)(w/ref));
        check("alphaPhiUn matches the host, nested minus signs and all", w < scalar(1e-14) * ref);

        // CONTROL: the compressive term must actually be doing something, or the agreement above is
        // agreement about the advective term alone.
        scalar maxComp = 0;
        for (scalar v : hComp) maxComp = std::fmax(maxComp, std::fabs(v));
        std::printf("  ...the compressive term alone reaches %.3e of that\n", (double)(maxComp/ref));
        check("the compressive term is a real part of the answer", maxComp > scalar(0.01)*ref);
    }

    // ---- 3. rhoPhi -------------------------------------------------------------------------------
    {
        SurfaceScalarField hAlphaPhi;
        alphaPhiUn(phi, phir, alpha1, alpha2, AlphaFluxScheme::vanLeer, AlphaFluxScheme::linear,
                   m, g, fvp, hAlphaPhi);
        SurfaceScalarField hRhoPhi;
        massFlux(hAlphaPhi, phi, scalar(1000), scalar(1), hRhoPhi);

        DeviceBuffer<scalar> dAlphaPhi(hAlphaPhi.internal), dRhoPhi;
        deviceMassFlux((int)nIf, dAlphaPhi, dPhi, scalar(1000), scalar(1), dRhoPhi);
        cudaDeviceSynchronize();
        std::vector<scalar> r;
        dRhoPhi.copyTo(r);

        scalar ref = 0;
        const scalar w = worstOf(r, hRhoPhi.internal, ref);
        std::printf("  rhoPhi: worst %.3e of %.3e\n", (double)w, (double)ref);
        check("rhoPhi matches the host", w < scalar(1e-14) * ref);
    }

    std::printf("test_device_alpha_flux: %d failures\n", failures);
    return failures ? 1 : 0;
}
