// brae::smoothSolver, the host reference -- the LOOP around the Gauss-Seidel sweeps.
//
// THE SWEEP ITSELF is not gated here. tests/gs_ladder.cu holds brae::gaussSeidelSmoothFolded against
// OpenFOAM's own residual after exactly n sweeps, n = 1..10, for both smoothers (8.0e-13 and 1.7e-13).
// What that cannot see is smoothSolver::solve: where the residual is evaluated, what nIterations
// counts, when the loop is entered and when it stops. Those are this file's.
//
// THE ORACLE for the loop is the device's deviceSymGaussSeidel in its LEVEL-SCHEDULED mode, which is
// OpenFOAM's smoothSolver stopping rule around the same sweep, re-ordered into dependency levels and
// run on the GPU, and itself held to OpenFOAM's ladder. Two implementations of one .C file that share
// no loop: where they agree on a sweep count the count is OpenFOAM's. The arms with no device
// counterpart (the negative-sweep branch) are checked against the sweep directly.
//
// THE MODE IS PINNED, and the first run of this gate is why. deviceSymGaussSeidel's DEFAULT is to run
// the sweep on the CPU -- it is faster on these meshes -- so the first draft compared a host sweep
// with a host sweep and reported the iterates 0.000e+00 apart. That number was true and said nothing.
// main() selects the device loop before the first solve and asserts it got it.
//
// AND THE CONTROL THE WHOLE PORT EXISTS FOR: at a loose tolerance the iterate this solver hands back
// is not the iterate PBiCGStab hands back. If it were, the substitution the host ran until now would
// have been free, and on the device it was worth 3.3e-06 of alpha on damBreak.
#include "box_mesh.cuh"
#include "device_amg.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include "device_gate_finite.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "device_pcg.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include "pbicgstab.cuh"
#include "primitive_mesh.cuh"
#include "smooth_solver_cpp.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(
    bool ok,
    const char* what)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

scalar worstAbs(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    scalar w = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        w = std::fmax(w, std::fabs(a[i] - b[i]));
    }
    return w;
}
} // namespace

int main()
{
    std::printf("== brae::smoothSolver: OpenFOAM's loop around its Gauss-Seidel smoothers ==\n");
    // before ANY solve: the mode is fixed at first use
    setenv("BRAE_GS_HOST_SMOOTHER", "0", 1);

    const PrimitiveMesh m = boxtest::boxMesh(12, 10, 6);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();

    // AN IMPLICIT CONVECTION-DIFFUSION MATRIX, built by hand so nothing in it depends on brae's own
    // assembly: a flux that changes sign across the box (so the upwind side alternates and the matrix
    // is not triangular), a diffusion that keeps both off-diagonals non-zero, and a ddt diagonal weak
    // enough that one sweep is NOT enough. The alpha pre-solve this solver is for is much easier than
    // this -- OpenFOAM takes 2 sweeps there -- and an easy fixture would compare 1 with 1.
    FvScalarMatrix M;
    M.diag.assign(static_cast<std::size_t>(nC), 0.0);
    M.source.assign(static_cast<std::size_t>(nC), 0.0);
    M.upper.assign(static_cast<std::size_t>(nIf), 0.0);
    M.lower.assign(static_cast<std::size_t>(nIf), 0.0);
    for (label f = 0; f < nIf; ++f)
    {
        const scalar phi = 0.8*std::sin(0.37*f);
        const scalar d = 0.35 + 0.1*std::cos(0.11*f);
        const std::size_t sf = static_cast<std::size_t>(f);
        M.upper[sf] = -d + (phi < 0 ? phi : 0.0);
        M.lower[sf] = -d - (phi > 0 ? phi : 0.0);
        M.diag[static_cast<std::size_t>(m.owner()[sf])] -= M.lower[sf];
        M.diag[static_cast<std::size_t>(m.neighbour()[sf])] -= M.upper[sf];
    }
    for (label c = 0; c < nC; ++c)
    {
        M.diag[static_cast<std::size_t>(c)] += 0.05;
        M.source[static_cast<std::size_t>(c)] = 1.0 + std::sin(0.013*c);
    }
    M.internalCoeffs.resize(fvp.size());
    M.boundaryCoeffs.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        // a fixedValue-like patch: it adds to the face cell's diagonal and source, so the fold is live
        M.internalCoeffs[pi].assign(static_cast<std::size_t>(fvp[pi].size), 0.2);
        M.boundaryCoeffs[pi].assign(static_cast<std::size_t>(fvp[pi].size), 0.1);
    }
    const std::vector<scalar> zeros(static_cast<std::size_t>(nC), 0.0);

    // The refusal, first: it needs no device.
    {
        std::vector<label> own(m.owner().begin(), m.owner().begin() + nIf);
        std::vector<label> nei = m.neighbour();
        std::swap(own[3], own[nIf - 4]);
        std::swap(nei[3], nei[nIf - 4]);
        bool threw = false;
        try
        {
            lduOwnerStart(own, nei, nC);
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        check(threw, "a mesh out of OpenFOAM's upper-triangular face order is REFUSED, not smoothed");
    }

    // nIterations counts SWEEPS, evaluated once per nSweeps.
    {
        std::vector<scalar> x = zeros;
        const SolverPerformance sp = smoothSolver(M, x, m, fvp, true, 0.0, 0.0, 5, 0, 2);
        std::printf("  nSweeps 2, maxIter 5, tolerance 0: %d sweeps\n", sp.nIterations);
        check(sp.nIterations == 6,
              "`(nIterations += nSweeps) < maxIter` runs SIX sweeps at nSweeps 2, maxIter 5");
    }

    // A negative nSweeps is a fixed count with no residual.
    {
        std::vector<scalar> x = zeros;
        const SolverPerformance sp = smoothSolver(M, x, m, fvp, true, 1e-30, 0.0, 1000, 0, -3);

        std::vector<scalar> diagC = M.diag, b = M.source;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const std::size_t c = static_cast<std::size_t>(fvp[pi].faceCells[i]);
                diagC[c] += M.internalCoeffs[pi][static_cast<std::size_t>(i)];
                b[c] += M.boundaryCoeffs[pi][static_cast<std::size_t>(i)];
            }
        }
        const std::vector<label> own(m.owner().begin(), m.owner().begin() + nIf);
        std::vector<scalar> ref = zeros;
        gaussSeidelSmoothFolded(lduOwnerStart(own, m.neighbour(), nC), m.neighbour(),
                                diagC, M.upper, M.lower, b, ref, 3, true);
        check(sp.nIterations == 3 && sp.initialResidual == 0 && sp.finalResidual == 0,
              "nSweeps -3 reports 3 iterations and NO residual");
        check(worstAbs(x, ref) == 0, "...and is exactly three sweeps of the folded system");
    }

    // minIter forces the loop on a solve that is already converged.
    {
        std::vector<scalar> x = zeros;
        smoothSolver(M, x, m, fvp, true, 1e-13, 0.0, 1000, 0, 1);
        const SolverPerformance again = smoothSolver(M, x, m, fvp, true, 1e-6, 0.0, 1000, 0, 1);
        const SolverPerformance forced = smoothSolver(M, x, m, fvp, true, 1e-6, 0.0, 1000, 4, 1);
        check(again.nIterations == 0, "a converged system takes no sweeps");
        check(forced.nIterations == 4, "...and exactly minIter of them when minIter asks");
    }

    // The two smoothers are different solvers.
    std::vector<scalar> xSym = zeros, xFwd = zeros;
    const SolverPerformance spSym = smoothSolver(M, xSym, m, fvp, true, 1e-30, 0.05, 1000);
    const SolverPerformance spFwd = smoothSolver(M, xFwd, m, fvp, false, 1e-30, 0.05, 1000);
    std::printf("  relTol 0.05: symGaussSeidel %d sweeps, GaussSeidel %d\n",
                spSym.nIterations, spFwd.nIterations);
    check(spSym.nIterations >= 5, "the fixture is hard enough that relTol 0.05 takes five sweeps or more");
    check(spSym.nIterations != spFwd.nIterations && worstAbs(xSym, xFwd) > 1e-3,
          "GaussSeidel and symGaussSeidel stop in different places");

    // THE CONTROL: it is not PBiCGStab.
    {
        std::vector<scalar> xB = zeros;
        const SolverPerformance spB = pbicgstab(M, xB, m, fvp, 1e-30, 0.05, 1000);
        const scalar away = worstAbs(xB, xSym);
        std::printf("  relTol 0.05: PBiCGStab %d iterations, iterate %.3e from symGaussSeidel's\n",
                    spB.nIterations, (double)away);
        check(away > 1e-3,
              "at relTol 0.05 PBiCGStab's iterate is NOT this solver's -- the substitution is visible");
        check(std::fabs(spB.initialResidual - spSym.initialResidual) < 1e-12*spSym.initialResidual,
              "...from the same normFactor-scaled initial residual, so only the solver differs");
    }

    // Against the device's OpenFOAM-held solver.
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device arms are skipped)\n");
    }
    else
    {
        check(!deviceGaussSeidelUsesHostSmoother(),
              "the device arm runs the LEVEL-SCHEDULED GPU sweep, not the CPU one it defaults to");
        DeviceMesh dm = buildDeviceMesh(m, g, fvp);
        std::vector<scalar> diagC = M.diag, b = M.source;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const std::size_t c = static_cast<std::size_t>(fvp[pi].faceCells[i]);
                diagC[c] += M.internalCoeffs[pi][static_cast<std::size_t>(i)];
                b[c] += M.boundaryCoeffs[pi][static_cast<std::size_t>(i)];
            }
        }
        DeviceBuffer<scalar> dD, dU, dL, dB;
        dD.copyFrom(diagC);
        dU.copyFrom(M.upper);
        dL.copyFrom(M.lower);
        dB.copyFrom(b);
        const DeviceLduView A = deviceLduView(dm, dD, dU, dL);

        struct Case
        {
            bool symmetric;
            scalar tol;
            scalar relTol;
            const char* name;
        };
        const Case cases[] = {
            {true, 1e-30, 0.05, "symGaussSeidel, relTol 0.05"},
            {true, 1e-10, 0.0, "symGaussSeidel, tolerance 1e-10"},
            {false, 1e-30, 0.05, "GaussSeidel, relTol 0.05"},
            {false, 1e-10, 0.0, "GaussSeidel, tolerance 1e-10"},
        };
        for (const Case& k : cases)
        {
            std::vector<scalar> xh = zeros;
            const SolverPerformance sh = smoothSolver(M, xh, m, fvp, k.symmetric, k.tol, k.relTol, 5000);

            DeviceBuffer<scalar> dx;
            dx.copyFrom(zeros);
            const scalar nf = deviceNormFactor(A, dx, dB, deviceOnes(nC));
            DeviceSolverPerf sd;
            deviceSymGaussSeidel(A, dB, dx, nf, k.tol, k.relTol, 5000, &sd, 0, 1, k.symmetric);
            std::vector<scalar> xd;
            dx.copyTo(xd);
            failures += brae::gatecheck::nonFinite("host iterate", xh);
            failures += brae::gatecheck::nonFinite("device iterate", xd);

            const scalar e = worstAbs(xh, xd);
            std::printf("  %-34s host %4d sweeps (final %.6e), device %4d (final %.6e); iterate %.3e apart\n",
                        k.name, sh.nIterations, (double)sh.finalResidual,
                        sd.nIterations, (double)sd.finalResidual, (double)e);
            check(sh.nIterations == sd.nIterations && sh.nIterations > 1,
                  "...the host takes the device's sweep count");
            check(std::fabs(sh.initialResidual - sd.initialResidual) < 1e-12*sd.initialResidual,
                  "...from the same initial residual");
            check(e < 1e-11, "...and hands back the same iterate, to 1e-11");
        }
    }

    std::printf("test_smooth_solver_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
