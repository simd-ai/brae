// interFoam's MOMENTUM MATRIX, assembled on the device, against the host's own assembleUEqn.
//
// The matrix is the thing every later number depends on: rAU is 1/A(), HbyA is H()/A(), and the
// pressure equation is built on both. A matrix that is wrong in the last digit is a solver that
// converges to a different answer, so this gate compares the ASSEMBLY -- diagonal, both off-diagonals,
// all three source components and the boundary coefficients -- rather than a solved field, where a
// linear solver's tolerance would absorb the difference.
//
// THE DEVICE PATH IS simpleFoam's assembleUEqn, REUSED, and that is the point: interFoam's momentum
// equation differs from it in exactly three ways and none of them is a new operator.
//
//   the flux is rhoPhi, the MULES-limited MASS flux, not the volumetric phi
//   the viscosity is mu = rho*nuEff, the rho-weighted overload of linearViscousStress
//   there is a ddt, and it carries rho on the diagonal and rho.oldTime() in the source
//
// The first two are arguments. The third was added to the shared assembler for this, along with a
// relaxEquation flag -- because simpleFoam's device path SKIPS relax when the factor is 1, and
// interFoam must not: damBreak's fvSolution says `equations { ".*" 1; }`, relaxEquation() finds it, and
// relax(1) still runs the diagonal-dominance clamp. Arm 3 measures what that clamp is worth.
//
// THE FIXTURE IS A VoF STATE, not a smooth one: rho jumps 1000x across a plane and rho.oldTime() is the
// same plane moved one cell, so the cells the interface crossed have rho_old = 1 against rho = 1000.
// On a converged smooth field the two rho fields agree and the ddt arm would be vacuous.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "inter_ueqn_cpp.cuh"
#include "device_inter_ueqn.cuh"
#include "device_boundary.cuh"
#include "device_mesh.cuh"
#include "UEqn.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
namespace ifm = brae::cpu::interFoam;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
// see test_device_inter_ueqn.cu: fabs(a - b*c) contracts into an FMA on ARM64 g++ and reports a
// fraction of an ULP between identical values, so a relative comparison is the honest form here.
scalar relWorst(const std::vector<scalar>& a, const std::vector<scalar>& b, scalar& scaleOut)
{
    scalar w = 0, s = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        w = std::fmax(w, std::fabs(a[i] - b[i]));
        s = std::fmax(s, std::fabs(b[i]));
    }
    scaleOut = s;
    return w;
}
}   // namespace

int main()
{
    std::printf("== interFoam's momentum matrix: device vs host\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 8;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N, scalar(0), scalar(2), scalar(1), scalar(0.5));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // ---- the state -------------------------------------------------------------------------------
    GeometricField<vector> U;
    U.internal.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        U.internal[c] = vector{scalar(0.5)*std::sin(C.x), scalar(0.3)*C.y - scalar(0.2),
                               scalar(0.1)*std::cos(C.z)};
    }
    for (const FvPatch& q : fvp)
    {
        if (q.type == "wall")
            U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                q, true, vector{0,0,0}, std::vector<vector>{}));
        else
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();

    const scalar dt = scalar(1e-3);
    std::vector<scalar> rho(static_cast<std::size_t>(nC)), rhoOld(static_cast<std::size_t>(nC)),
                        nuEff(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const scalar y = g.C()[c].y;
        rho[c]    = (y < scalar(4))        ? scalar(1000) : scalar(1);
        rhoOld[c] = (y < scalar(4) - scalar(1)) ? scalar(1000) : scalar(1);   // the interface moved
        nuEff[c]  = (y < scalar(4)) ? scalar(1e-6) : scalar(1.48e-5);
    }
    std::vector<vector> UOld(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
        UOld[c] = vector{U.internal[c].x*scalar(0.9), U.internal[c].y*scalar(1.1), U.internal[c].z};

    // rhoPhi: a MASS flux, which is what alphaEqn leaves behind
    const vector Ubulk{scalar(1.2), scalar(-0.6), scalar(0.3)};
    std::vector<scalar> rhoPhi(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const vector& S = g.Sf()[f];
        const scalar w = g.weights()[f];
        const scalar rhof = w*rho[m.owner()[f]] + (scalar(1) - w)*rho[m.neighbour()[f]];
        rhoPhi[f] = rhof * (Ubulk.x*S.x + Ubulk.y*S.y + Ubulk.z*S.z);
    }
    std::vector<std::vector<scalar>> rhoPhiBndH(fvp.size());
    std::vector<scalar> rhoPhiBnd, rhoBnd, nuBnd;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label gf = fvp[pi].start + i, cc = fvp[pi].faceCells[i];
            const vector& S = g.Sf()[gf];
            const scalar v = rho[cc] * (Ubulk.x*S.x + Ubulk.y*S.y + Ubulk.z*S.z);
            rhoPhiBndH[pi].push_back(v);
            rhoPhiBnd.push_back(v);
            rhoBnd.push_back(rho[cc]);
            nuBnd.push_back(nuEff[cc]);
        }
    std::vector<std::vector<scalar>> rhoBndH(fvp.size()), nuBndH(fvp.size());
    {
        label off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                rhoBndH[pi].push_back(rhoBnd[off + i]);
                nuBndH[pi].push_back(nuBnd[off + i]);
            }
            off += fvp[pi].size;
        }
    }

    // ---- the HOST matrix -------------------------------------------------------------------------
    ifm::InterMomentumInput hin;
    hin.rhoPhi    = &rhoPhi;
    hin.rhoPhiBnd = &rhoPhiBndH;
    hin.rho       = &rho;
    hin.rhoOld    = &rhoOld;
    hin.rhoBnd    = &rhoBndH;
    hin.UOld      = &UOld;
    hin.nuEff     = &nuEff;
    hin.nuEffBnd  = &nuBndH;
    hin.deltaT    = dt;
    hin.scheme    = ifm::DivScheme::upwind;
    hin.relaxEquationU = true;      // damBreak's `equations { ".*" 1; }`
    hin.relaxU    = scalar(1);
    const FvVectorMatrix H = ifm::assembleUEqn(U, hin, m, g, fvp);

    // ---- the DEVICE matrix -----------------------------------------------------------------------
    std::vector<scalar> ux(nC), uy(nC), uz(nC), uox(nC), uoy(nC), uoz(nC);
    for (label c = 0; c < nC; ++c)
    {
        ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z;
        uox[c] = UOld[c].x; uoy[c] = UOld[c].y; uoz[c] = UOld[c].z;
    }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dUox(uox), dUoy(uoy), dUoz(uoz);
    DeviceBuffer<scalar> dRho(rho), dRhoOld(rhoOld), dNu(nuEff), dRhoB(rhoBnd), dNuB(nuBnd);
    DeviceBuffer<scalar> dRhoPhi(rhoPhi), dRhoPhiB(rhoPhiBnd);

    DeviceBuffer<scalar> muCell, muFace, muBnd;
    deviceInterMuEff(dm, dRho, dNu, dRhoB, dNuB, muCell, muFace, muBnd);

    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);

    gpu::MomentumInput din;
    din.phiInt       = &dRhoPhi;          // the MASS flux
    din.phiBnd       = &dRhoPhiB;
    din.nuEffCell    = &muCell;           // mu, not nu -- the rho-weighted overload
    din.nuEffFace    = &muFace;
    din.nuEffBndFace = &muBnd;
    din.scheme       = brae::cpu::DivScheme::upwind;
    din.relaxU       = scalar(1);
    din.relaxEquation = true;
    din.ddtRho       = &dRho;
    din.ddtRhoOld    = &dRhoOld;
    din.ddtUOld[0]   = &dUox;
    din.ddtUOld[1]   = &dUoy;
    din.ddtUOld[2]   = &dUoz;
    din.ddtDeltaT    = dt;

    gpu::MomentumMatrix M;
    gpu::assembleUEqn(M, dm, dbU, dUx, dUy, dUz, din);
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }

    // ---- 1. the off-diagonals ---------------------------------------------------------------------
    {
        std::vector<scalar> up, lo;
        M.upper.copyTo(up);
        M.lower.copyTo(lo);
        scalar su = 0, sl = 0;
        const scalar wu = relWorst(up, H.upper, su);
        const scalar wl = relWorst(lo, H.lower, sl);
        std::printf("  upper: worst %.3e of %.3e;  lower: worst %.3e of %.3e\n",
                    (double)wu, (double)su, (double)wl, (double)sl);
        check("the off-diagonals match the host", wu <= scalar(1e-14)*su && wl <= scalar(1e-14)*sl);
        check("...and they are not all zero, so the convection term is in the matrix",
              su > scalar(1e-6));
    }

    // ---- 2. the diagonal, before and after relax --------------------------------------------------
    {
        std::vector<scalar> raw, rel;
        M.diag.copyTo(raw);
        check("relax() ran, because the case NAMES a factor even though it is 1", M.relaxed);
        M.relaxedDiag.copyTo(rel);
        scalar sr = 0;
        const scalar wr = relWorst(rel, H.diag, sr);
        std::printf("  diagonal after relax: worst %.3e of %.3e\n", (double)wr, (double)sr);
        check("the relaxed diagonal matches the host's", wr <= scalar(1e-13)*sr);

        // THE CLAMP IS NOT A NO-OP AT alpha == 1. D = max(|D|, sumOff)/alpha still runs, and on this
        // fixture it moves the diagonal -- which is why interFoam cannot use simpleFoam's
        // "skip relax when the factor is 1" condition.
        scalar moved = 0;
        for (label c = 0; c < nC; ++c) moved = std::fmax(moved, std::fabs(rel[c] - raw[c]));
        std::printf("  ...and the diagonal-dominance clamp moved it by %.3e of %.3e\n",
                    (double)moved, (double)sr);
        check("relax(1) is not a no-op -- the clamp runs, which is the whole reason for relaxEquation",
              moved > scalar(1e-12)*sr);
    }

    // ---- 3. the source, all three components ------------------------------------------------------
    {
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> src, want(static_cast<std::size_t>(nC));
            M.source[k].copyTo(src);
            for (label c = 0; c < nC; ++c)
                want[c] = (k == 0) ? H.source[c].x : (k == 1) ? H.source[c].y : H.source[c].z;
            scalar sc = 0;
            const scalar w = relWorst(src, want, sc);
            std::printf("  source[%d]: worst %.3e of %.3e\n", k, (double)w, (double)sc);
            check("the source matches the host", w <= scalar(1e-12)*sc);
        }
    }

    // ---- 4. the boundary coefficients -------------------------------------------------------------
    {
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> iC, bC, wantI, wantB;
            M.iC[k].copyTo(iC);
            M.bC[k].copyTo(bC);
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    const vector& a = H.internalCoeffs[pi][i];
                    const vector& b = H.boundaryCoeffs[pi][i];
                    wantI.push_back((k == 0) ? a.x : (k == 1) ? a.y : a.z);
                    wantB.push_back((k == 0) ? b.x : (k == 1) ? b.y : b.z);
                }
            scalar si = 0, sb = 0;
            const scalar wi = relWorst(iC, wantI, si);
            const scalar wb = relWorst(bC, wantB, sb);
            if (k == 0)
                std::printf("  boundary coeffs: internal worst %.3e of %.3e, boundary %.3e of %.3e\n",
                            (double)wi, (double)si, (double)wb, (double)sb);
            check("the boundary coefficients match the host",
                  wi <= scalar(1e-13)*std::fmax(si, scalar(1e-300)));
        }
    }

    // ---- 5. THE ddt CONTROL: rho.oldTime() is load-bearing ---------------------------------------
    // Assemble again with rho passed twice -- the port a careless reading of fvm::ddt(rho, U) writes --
    // and measure the source. It is bounded, it is smooth, and it is wrong by the density ratio in
    // exactly the cells the interface crossed.
    {
        gpu::MomentumInput wrong = din;
        wrong.ddtRhoOld = &dRho;
        gpu::MomentumMatrix W;
        gpu::assembleUEqn(W, dm, dbU, dUx, dUy, dUz, wrong);
        std::vector<scalar> good, bad;
        M.source[0].copyTo(good);
        W.source[0].copyTo(bad);
        scalar ratio = 0;
        int nMoved = 0;
        for (label c = 0; c < nC; ++c)
        {
            if (std::fabs(good[c]) > scalar(1e-30))
            {
                const scalar r = std::fabs(bad[c]/good[c]);
                ratio = std::fmax(ratio, r);
                if (r > scalar(2) || r < scalar(0.5)) ++nMoved;
            }
        }
        std::printf("  passing rho twice instead of rho.oldTime(): source up to %.1fx out, in %d of "
                    "%ld cells\n", (double)ratio, nMoved, (long)nC);
        check("rho.oldTime() in the ddt source is load-bearing at the interface", ratio > scalar(100));
        check("...and only in the cells the interface crossed, not everywhere",
              nMoved > 0 && nMoved < nC/2);
    }

    std::printf("test_device_inter_ueqn_assembly: %d failures\n", failures);
    return failures ? 1 : 0;
}
