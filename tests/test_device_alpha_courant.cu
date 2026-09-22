// The VoF Courant number on the device -- alphaCourantNo.H.
//
// It is one number, and a wrong one does not break a run: it makes the time step wrong, which shows up
// as a case that is slower than it needs to be or, worse, one that advects the interface further than
// MULES can bound. So every arm here is a control against a specific wrong version, and none of them
// could be caught by looking at a field.
//
//   THE MASK IS 0/1 OVER A CLOSED BAND, not a weight and not an open one. Arm 2 puts cells at EXACTLY
//   0.01 and EXACTLY 0.99 in the fixture, because pos0 is 1 at zero and pos is not -- the two differ
//   only on those cells and nowhere else.
//
//   surfaceSum TOUCHES THE BOUNDARY. Arm 3 runs the same fixture with the boundary half dropped; the
//   gap is the flux through the domain's own faces, which is exactly where the limiting cell usually
//   is.
//
//   THE DENOMINATOR IS NOT MASKED. Arm 4 masks it too -- the reading that makes meanAlphaCoNum an
//   interface-local average instead of a domain average of an interface quantity.
//
//   WITH NO INTERFACE THE ANSWER IS ZERO. Arm 5 says so, and says the ORDINARY Courant number on the
//   same fixture is not, so the zero is the mask's doing and not an empty fixture's.
//
// NONE OF THE COMPARISONS WITH THE HOST IS BIT-EXACT, and arm 1 explains why: sumPhi is a per-cell
// gather here and a per-cell scatter on the host, so each cell's |phi| terms are summed in a different
// order. A max over those sums would be bit-identical if the sums were -- so the one ULP that shows up
// in CoNum is itself the measurement that they are not.
#include "box_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "time_controls.cuh"
#include "device_alpha_courant.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
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
}   // namespace

int main()
{
    std::printf("== the VoF Courant number: device vs host\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    // anisotropic cells, so V varies and the max-of-ratios is not the max-of-sums
    const label N = 10;
    PrimitiveMesh m = boxtest::boxMesh(N, N, 2, scalar(0), scalar(2), scalar(1), scalar(0.5));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    const vector U{scalar(1.3), scalar(-0.7), scalar(0.4)};
    std::vector<scalar> phiInt(static_cast<std::size_t>(nIf)), phiBnd;
    for (label f = 0; f < nIf; ++f)
    {
        const vector& S = g.Sf()[f];
        phiInt[f] = U.x*S.x + U.y*S.y + U.z*S.z;
    }
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const vector& S = g.Sf()[fvp[pi].start + i];
            phiBnd.push_back(U.x*S.x + U.y*S.y + U.z*S.z);
        }

    // an alpha field that lands cells EXACTLY on both ends of the band -- see arm 2
    std::vector<scalar> alpha(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const scalar t = scalar(c % N) / scalar(N - 1);        // 0 .. 1
        alpha[c] = t;
        if (c % N == 1) alpha[c] = scalar(0.01);               // exactly the lower edge
        if (c % N == 2) alpha[c] = scalar(0.99);               // exactly the upper edge
        if (c % N == 3) alpha[c] = scalar(0.005);              // just outside
        if (c % N == 4) alpha[c] = scalar(0.995);              // just outside
    }

    // the host's own surfaceSum(mag(phi)), built here so the oracle is the host FORMULA and not a
    // second copy of the device's gather
    std::vector<scalar> sumPhi(static_cast<std::size_t>(nC), scalar(0));
    for (label f = 0; f < nIf; ++f)
    {
        sumPhi[m.owner()[f]]     += std::fabs(phiInt[f]);
        sumPhi[m.neighbour()[f]] += std::fabs(phiInt[f]);
    }
    {
        label off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (fvp[pi].type != "empty")
                for (label i = 0; i < fvp[pi].size; ++i)
                    sumPhi[fvp[pi].faceCells[i]] += std::fabs(phiBnd[off + i]);
            off += fvp[pi].size;
        }
    }

    const scalar dt = scalar(2e-3);
    DeviceBuffer<scalar> dPhiInt(phiInt), dPhiBnd(phiBnd), dAlpha(alpha);

    // ---- 1. against the host --------------------------------------------------------------------
    const CourantNumbers host = alphaCourantNo(sumPhi, alpha, g.V(), dt);
    const DeviceCourantNumbers dev = deviceAlphaCourantNo(dm, dPhiInt, dPhiBnd, &dAlpha, dt);
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }
    std::printf("  alphaCoNum: device %.17g, host %.17g;  mean: %.17g / %.17g\n",
                (double)dev.CoNum, (double)host.CoNum, (double)dev.meanCoNum, (double)host.meanCoNum);
    // NOT bit for bit, and the reason is worth stating because it is the opposite of what it looks
    // like. A max IS order-independent -- so if the per-cell surfaceSum agreed exactly, this max would
    // too. It does not, which is the measurement: sumPhi is a GATHER here and a SCATTER on the host,
    // each cell's |phi| terms are added in a different order, and floating-point addition is not
    // associative. The gap is one ULP of the ratio. This file asserted equality first and that claim
    // was wrong.
    check("the max agrees with the host to 1e-14 relative -- a max over per-cell SUMS, and the sums "
          "are a gather here and a scatter there",
          std::fabs(dev.CoNum - host.CoNum) <= scalar(1e-14)*std::fabs(host.CoNum));
    check("...and so does the mean, which is a second reduction on top of that",
          std::fabs(dev.meanCoNum - host.meanCoNum) <= scalar(1e-14)*std::fabs(host.meanCoNum));
    check("...and it is not zero, so the fixture has an interface in it", dev.CoNum > scalar(1e-3));

    // ---- 2. pos0, NOT pos: the band is CLOSED ----------------------------------------------------
    {
        std::vector<scalar> open(static_cast<std::size_t>(nC), scalar(0));
        for (label c = 0; c < nC; ++c)
        {
            const scalar lo = alpha[c] - scalar(0.01), hi = scalar(0.99) - alpha[c];
            const scalar mask = ((lo > scalar(0)) ? scalar(1) : scalar(0))       // pos, not pos0
                              * ((hi > scalar(0)) ? scalar(1) : scalar(0));
            open[c] = mask*sumPhi[c];
        }
        const CourantNumbers openCo = courantNo(open, g.V(), dt);
        int onEdge = 0;
        for (label c = 0; c < nC; ++c)
            if (alpha[c] == scalar(0.01) || alpha[c] == scalar(0.99)) ++onEdge;
        std::printf("  %d cells sit EXACTLY on the band's edges; pos instead of pos0 gives mean "
                    "%.17g against %.17g\n", onEdge, (double)openCo.meanCoNum, (double)host.meanCoNum);
        check("the fixture actually lands cells on the edges, or arm 2 is untestable", onEdge > 0);
        check("pos0 includes them and pos does not -- the band is CLOSED",
              openCo.meanCoNum != host.meanCoNum);
    }

    // ---- 3. surfaceSum includes the BOUNDARY faces -----------------------------------------------
    {
        std::vector<scalar> noBnd(static_cast<std::size_t>(nC), scalar(0));
        for (label f = 0; f < nIf; ++f)
        {
            noBnd[m.owner()[f]]     += std::fabs(phiInt[f]);
            noBnd[m.neighbour()[f]] += std::fabs(phiInt[f]);
        }
        const CourantNumbers without = alphaCourantNo(noBnd, alpha, g.V(), dt);
        std::printf("  dropping the boundary half of surfaceSum: %.6f against %.6f\n",
                    (double)without.CoNum, (double)host.CoNum);
        check("the boundary faces are in the sum, and leaving them out understates Co",
              without.CoNum < host.CoNum*scalar(0.99));
    }

    // ---- 4. the DENOMINATOR is not masked --------------------------------------------------------
    {
        const std::vector<scalar> mask = nearInterface(alpha);
        std::vector<scalar> maskedPhi(static_cast<std::size_t>(nC)), maskedV(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            maskedPhi[c] = mask[c]*sumPhi[c];
            maskedV[c]   = mask[c]*g.V()[c];        // the wrong reading
        }
        scalar sPhi = 0, sV = 0;
        for (label c = 0; c < nC; ++c) { sPhi += maskedPhi[c]; sV += maskedV[c]; }
        const scalar wrongMean = (sV > 0) ? scalar(0.5)*(sPhi/sV)*dt : scalar(0);
        std::printf("  masking the volume too: mean %.6e against %.6e\n",
                    (double)wrongMean, (double)host.meanCoNum);
        // MEASURED at 1.67x on this fixture. The factor is the ratio of the whole mesh volume to the
        // band's, so it is the fixture's geometry and not a universal number -- the bound is therefore
        // "materially different", not "twice".
        std::printf("  ...a factor of %.3f, which is the whole mesh volume over the band's\n",
                    (double)(wrongMean/host.meanCoNum));
        check("meanAlphaCoNum divides by the WHOLE mesh volume -- masking it is a different diagnostic",
              std::fabs(wrongMean - host.meanCoNum) > scalar(0.1)*host.meanCoNum);
    }

    // ---- 5. no interface -> zero, and the ORDINARY Courant number on the same flux is not ---------
    {
        std::vector<scalar> dry(static_cast<std::size_t>(nC), scalar(0));   // all air, no band
        DeviceBuffer<scalar> dDry(dry);
        const DeviceCourantNumbers none = deviceAlphaCourantNo(dm, dPhiInt, dPhiBnd, &dDry, dt);
        const DeviceCourantNumbers ord  = deviceAlphaCourantNo(dm, dPhiInt, dPhiBnd, nullptr, dt);
        const CourantNumbers hostOrd = courantNo(sumPhi, g.V(), dt);
        std::printf("  with no interface: alphaCo %.3e; the ordinary Co on the same flux %.6f "
                    "(host %.6f)\n", (double)none.CoNum, (double)ord.CoNum, (double)hostOrd.CoNum);
        check("no interface means an alpha Courant number of exactly zero", none.CoNum == scalar(0));
        check("...while the ordinary Courant number is not, so that zero is the mask's doing",
              ord.CoNum > scalar(1e-3));
        check("...and the unmasked path matches the host's courantNo to 1e-14 relative, the same "
              "gather-versus-scatter difference as arm 1",
              std::fabs(ord.CoNum - hostOrd.CoNum) <= scalar(1e-14)*std::fabs(hostOrd.CoNum));

        // and what setDeltaTVoF does with it: maxAlphaCo/(0 + SMALL) is astronomical, so the min picks
        // the ordinary branch and a case with no interface yet is NOT throttled.
        VoFTimeControls tc;
        tc.base.adjustTimeStep = true;
        tc.base.maxCo = scalar(0.5);
        tc.base.maxDeltaT = scalar(1);
        tc.maxAlphaCo = scalar(0.2);
        const scalar grown = setDeltaTVoF(dt, ord.CoNum, none.CoNum, tc);
        std::printf("  setDeltaTVoF with no interface: dt %.3e -> %.3e\n", (double)dt, (double)grown);
        check("a case with no interface yet is limited by maxCo alone, not throttled to zero",
              grown > dt);
    }

    std::printf("test_device_alpha_courant: %d failures\n", failures);
    return failures ? 1 : 0;
}
