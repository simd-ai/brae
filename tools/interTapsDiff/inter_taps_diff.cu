// inter_taps_diff <caseDir> <startDir> <nSteps>: brae's HOST loop and its DEVICE loop on one case for nSteps,
// and how far the device's intermediates of the LAST step are from the host's -- the momentum matrix, rAU,
// HbyA, the p_rgh system and its source's parts, then the fields the step leaves. An instrument, not a gate:
// it asserts nothing. BRAE_TAP_CORRECTOR picks the pressure corrector both arms tap (0, the first, unless set).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "inter_case_cpp.cuh"
#include "inter_driver_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "alpha_eqn_cpp.cuh"
#include "device_inter_step.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
// worst |a - b| over the worst |a|, and where
void report(
    const char* what,
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    if (a.empty() || a.size() != b.size())
    {
        std::printf("  %-28s not comparable (host %zu, device %zu)\n", what, a.size(), b.size());
        return;
    }
    scalar worst = 0;
    scalar ref = 0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const scalar d = std::fabs(a[i] - b[i]);
        if (d > worst)
        {
            worst = d;
            at = i;
        }
        ref = std::fmax(ref, std::fabs(a[i]));
    }
    // ...and how many entries are further apart than 1e-13 of that scale: one cell or the field
    std::size_t nOver = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::fabs(a[i] - b[i]) > scalar(1e-13)*ref) ++nOver;
    }
    std::printf("  %-28s rel %.3e   (abs %.3e of %.3e, at %zu of %zu; %zu over 1e-13)\n",
                what, (double)(ref > 0 ? worst/ref : worst), (double)worst, (double)ref, at, a.size(), nOver);
    // ...and whether the difference is one SHIFT of the whole field or noise: its mean beside its rms
    scalar sum = 0;
    scalar sumSq = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const scalar d = b[i] - a[i];
        sum += d;
        sumSq += d*d;
    }
    const scalar n = static_cast<scalar>(a.size());
    std::printf("  %-28s     device - host: mean %+.3e, rms %.3e\n", "", (double)(sum/n), (double)std::sqrt(sumSq/n));
}

std::vector<scalar> component(
    const std::vector<vector>& v,
    int d)
{
    std::vector<scalar> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        out[i] = (d == 0) ? v[i].x : (d == 1) ? v[i].y : v[i].z;
    }
    return out;
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <startDir> <nSteps>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const label nSteps = std::atoi(argv[3]);

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);

    InterFields hostF;
    InterFields devF;
    PressureTaps ht;
    AlphaTaps hat;
    DeviceInterStepTaps dt;
    const RunReport rh = runInterFoam(caseDir, startDir, m, g, fvp, nSteps, /*verbose=*/false, &hostF,
                                      scalar(1.0e300), &ht, /*mutableMesh=*/nullptr, &hat);
    const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, fvp, nSteps, /*verbose=*/false, &devF,
                                            scalar(1.0e300), &dt);
    std::printf("== device against host, step %ld (host %ld steps, device %ld), corrector tapped: host %ld, "
                "device %d ==\n",
                (long)nSteps, (long)rh.steps, (long)rd.steps, (long)ht.tapCorrector, dt.tapCorrector);

    std::printf(" the alpha step\n");
    report("alpha after the pre-solve", hat.preSolveAlpha, dt.preSolveAlpha.host());
    report("alpha after the step", hostF.alpha1.internal, dt.alphaAfterAlphaStep.host());
    std::printf(" the momentum matrix\n");
    report("UEqn diag", ht.uEqnDiag, dt.UEqnDiag.host());
    report("UEqn upper", ht.uEqnUpper, dt.UEqnUpper.host());
    report("UEqn lower", ht.uEqnLower, dt.UEqnLower.host());
    report("UEqn source x", ht.uEqnSourceX, dt.UEqnSourceX.host());
    report("rAU", ht.rAU, dt.rAU.host());
    for (int d = 0; d < 3; ++d)
    {
        const std::string name = std::string("HbyA ") + "xyz"[d];
        report(name.c_str(), component(ht.HbyA, d), dt.HbyA[d].host());
    }
    std::printf(" the p_rgh system\n");
    report("phiHbyA before phig", ht.phiHbyA, dt.phiHbyAIntPrePhig.host());
    report("phig", ht.phig, dt.phigIntTap.host());
    report("p_rgh diag", ht.pDiag, dt.pDiag.host());
    report("p_rgh upper", ht.pUpper, dt.pUpper.host());
    report("p_rgh lower", ht.pLower, dt.pLower.host());
    report("div(phiHbyA)", ht.pDivPhiHbyA, dt.divPhiHbyA.host());
    report("non-orthogonal source", ht.pNonOrthSource, dt.nonOrthSource.host());
    report("p_rgh source", ht.pSource, dt.pSource.host());
    std::printf(" the fields the step leaves\n");
    report("alpha", hostF.alpha1.internal, devF.alpha1.internal);
    report("p_rgh", hostF.p_rgh.internal, devF.p_rgh.internal);
    for (int d = 0; d < 3; ++d)
    {
        const std::string name = std::string("U ") + "xyz"[d];
        report(name.c_str(), component(hostF.U.internal, d), component(devF.U.internal, d));
    }
    report("phi", hostF.phi.internal, devF.phi.internal);
    report("rho", hostF.rho, devF.rho);
    // phiHbyA on the boundary at the tapped corrector, before phig: what U's patch values put into the
    // pressure equation's source
    std::printf(" phiHbyA on the boundary at the tapped corrector, per patch (sum host, net difference, worst face)\n");
    {
        const std::vector<scalar> dev = dt.phiHbyABndPrePhig.host();
        std::size_t off = 0;
        for (std::size_t pi = 0; pi < fvp.size() && pi < ht.phiHbyABndPrePhig.size(); ++pi)
        {
            const std::vector<scalar>& a = ht.phiHbyABndPrePhig[pi];
            if (off + a.size() > dev.size()) break;
            scalar sa = 0;
            scalar sd = 0;
            scalar worst = 0;
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                sa += a[i];
                sd += dev[off + i] - a[i];
                worst = std::fmax(worst, std::fabs(dev[off + i] - a[i]));
            }
            off += a.size();
            std::printf("  %-22s sum %+.6e   net difference %+.3e   worst face %.3e\n",
                        fvp[pi].name.c_str(), (double)sa, (double)sd, (double)worst);
        }
    }
    // THE BOUNDARY, patch by patch: a net flux difference there is what a pressure equation answers with one
    // smooth shift of the whole field, which no interior tap shows
    std::printf(" the boundary flux the step leaves, per patch (sum host, sum device - sum host, worst face)\n");
    for (std::size_t pi = 0; pi < fvp.size() && pi < hostF.phi.boundary.size(); ++pi)
    {
        const std::vector<scalar>& a = hostF.phi.boundary[pi];
        const std::vector<scalar>& b = devF.phi.boundary[pi];
        if (a.empty() || a.size() != b.size()) continue;
        scalar sa = 0;
        scalar sd = 0;
        scalar worst = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            sa += a[i];
            sd += b[i] - a[i];
            worst = std::fmax(worst, std::fabs(b[i] - a[i]));
        }
        std::printf("  %-22s sum %+.6e   net difference %+.3e   worst face %.3e\n",
                    fvp[pi].name.c_str(), (double)sa, (double)sd,
                    (double)worst);
    }
    if (hostF.turbulence.on)
    {
        report("k", hostF.turbulence.k.internal, devF.turbulence.k.internal);
        report("nut", hostF.turbulence.nut.internal, devF.turbulence.nut.internal);
    }
    return 0;
}
