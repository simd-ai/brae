// PBiCG with DILU on the device must be the host reference's PBiCG, which is OpenFOAM's.
//
// WHY THIS GATE EXISTS. waves/mangroveInteraction names `solver PBiCG; preconditioner DILU;` for k and
// epsilon, and PBiCG is NOT PBiCGStab: it carries a TRANSPOSE system beside the direct one, and at the
// same tolerance the two stop at different iterates. The device had PBiCGStab and a Gauss-Seidel sweep,
// and would have run one of them under the other's name.
//
// THE REFERENCE is brae::pbicgDILU (src/OpenFOAM/matrices/pbicg.cu), transcribed from PBiCG.C and
// DILUPreconditioner.C and held to OpenFOAM's own log -- every k and epsilon iteration count and final
// residual of 450 steps -- by tests/interfoam_mangrove_vs_openfoam.sh. Two links, each measured.
//
// THE DEVICE'S TRANSPOSE IS A VIEW: deviceAmul and diluApply on the same matrix with upper and lower
// exchanged. Legs 1 and 2 turn that argument into a measurement, against Tmul and preconditionT
// written out from OpenFOAM's text, and each has a control that says the exchange MATTERS on this
// matrix -- on a symmetric one it would be the identity and both legs would pass with it left out.
//
// MEASURED (5760 cells, upwind convection minus a diffusion that jumps by 200, a barely dominant
// diagonal): Tmul 1.6e-15 from lduMatrix::Tmul where Amul is 4.2e-01 from it; preconditionT bit for bit
// where precondition is 1.0 from it; at relTol 0.01 five iterations in both and the iterate 1.1e-14
// apart, with PBiCGStab taking three and landing 2.6e-03 away; converged, 25 iterations in both and the
// solution 7.6e-16 apart.
// BROKEN ONCE EACH, in device_pbicg.cu: preconditionT on the direct view -- 1000 iterations, the
// iterate 9.1e+06 away; Tmul on the direct view -- 661 iterations for 5; beta inverted -- 1000
// iterations. Eight checks fail each time.
#include "box_mesh.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include "device_dilu.cuh"
#include "device_gate_finite.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "device_pbicg.cuh"
#include "device_pcg.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "geometric_field.cuh"
#include "pbicg.cuh"
#include "primitive_mesh.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
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

bool sameBits(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    return a.size() == b.size()
        && std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0;
}

scalar worstRel(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    scalar w = 0;
    scalar s = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        w = std::fmax(w, std::fabs(a[i] - b[i]));
        s = std::fmax(s, std::fabs(b[i]));
    }
    return w/std::fmax(s, scalar(1e-300));
}
} // namespace

int main()
{
    std::printf("== PBiCG with DILU on the device is the host reference's, which is OpenFOAM's ==\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  SKIP: no CUDA device\n");
        return 77;
    }

    // A k-epsilon-shaped system: upwind convection by a flux that changes sign across the box, minus
    // a diffusion whose coefficient varies by face, plus a positive diagonal standing for ddt and the
    // sink. Upwind makes upper != lower on every face the flux crosses, which is what gives the
    // transpose system something to be different about.
    const PrimitiveMesh m = boxtest::boxMesh(24, 20, 12);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();

    GeometricField<scalar> fld;
    fld.internal.assign((std::size_t)nC, 0.0);
    for (const FvPatch& p : fvp)
    {
        if (p.type == "empty")
        {
            fld.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        }
        else
        {
            fld.boundary.push_back(
                std::make_unique<FixedValuePatchField<scalar>>(p, true, 0.0, std::vector<scalar>{}));
        }
    }
    fld.evaluateBoundary();

    scalar yTop = 0;
    for (label c = 0; c < nC; ++c)
    {
        yTop = std::fmax(yTop, g.C()[(std::size_t)c].y);
    }
    SurfaceScalarField phi;
    SurfaceScalarField gammaf;
    phi.internal.resize((std::size_t)nIf);
    gammaf.internal.resize((std::size_t)nIf);
    for (label f = 0; f < nIf; ++f)
    {
        const vector& x = g.Cf()[(std::size_t)f];
        const vector u{std::sin(scalar(2.1)*x.y) + scalar(0.3), std::cos(scalar(1.7)*x.x), scalar(0.4)*std::sin(scalar(3.0)*x.z)};
        phi.internal[(std::size_t)f] = dot(g.Sf()[(std::size_t)f], u);
        // ...a diffusion that JUMPS by 200 across the box's mid-height, as nut does across an interface
        const scalar hi = x.y > scalar(0.5)*yTop ? scalar(200) : scalar(1);
        gammaf.internal[(std::size_t)f] = scalar(0.02)*hi*(scalar(1) + scalar(0.5)*std::sin(scalar(0.9)*f));
    }
    phi.boundary.resize(fvp.size());
    gammaf.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phi.boundary[pi].assign((std::size_t)fvp[pi].size, scalar(0));
        gammaf.boundary[pi].assign((std::size_t)fvp[pi].size, scalar(0.02));
    }
    FvScalarMatrix M = fvm::div<scalar>(phi.internal, phi.boundary, fld, m, fvp);
    const FvScalarMatrix L = fvm::laplacian<scalar>(gammaf, fld, m, g, fvp, /*corrected=*/false);
    addEqual(M, L, scalar(-1));
    for (label c = 0; c < nC; ++c)
    {
        const vector& x = g.C()[(std::size_t)c];
        // the flux above is not divergence-free (|div u| <= 1.2), so the convection's own diagonal
        // can be negative: the added term keeps the row dominant, and only JUST -- at 40 the first
        // draft of this fixture converged in one iteration at relTol 0.01 and four at 1e-12, and "the
        // device takes the host's count" compared 1 with 1
        M.diag[(std::size_t)c] += g.V()[(std::size_t)c]*scalar(1.5);
        M.source[(std::size_t)c] = g.V()[(std::size_t)c]*(scalar(1) + std::sin(scalar(1.3)*x.x + scalar(0.7)*x.y));
    }

    std::vector<scalar> diagC = M.diag;
    std::vector<scalar> b = M.source;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label c = fvp[pi].faceCells[i];
            diagC[(std::size_t)c] += M.internalCoeffs[pi][i];
            b[(std::size_t)c] += M.boundaryCoeffs[pi][i];
        }
    }
    {
        scalar asym = 0;
        scalar scale = 0;
        for (label f = 0; f < nIf; ++f)
        {
            asym = std::fmax(asym, std::fabs(M.upper[(std::size_t)f] - M.lower[(std::size_t)f]));
            scale = std::fmax(scale, std::fabs(M.upper[(std::size_t)f]));
        }
        std::printf("        (%d cells, %d internal faces; largest |upper - lower| %.3e of %.3e)\n", (int)nC,
                    (int)nIf, (double)asym, (double)scale);
        check(asym > scalar(0.1)*scale, "the fixture is ASYMMETRIC, so a transpose has something to exchange");
    }

    DeviceBuffer<scalar> dDiag, dUp, dLo, db;
    dDiag.copyFrom(diagC);
    dUp.copyFrom(M.upper);
    dLo.copyFrom(M.lower);
    db.copyFrom(b);
    const DeviceLduView A = deviceLduView(dm, dDiag, dUp, dLo);
    DeviceLduView T = A;
    T.upper = A.lower;
    T.lower = A.upper;
    DeviceDilu dilu = buildDeviceDilu(m.owner(), m.neighbour(), nC);
    diluUpdate(A, dilu);

    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<scalar> x0((std::size_t)nC);
    for (label c = 0; c < nC; ++c)
    {
        x0[(std::size_t)c] = std::cos(scalar(0.3)*c) + scalar(1.1);
    }
    DeviceBuffer<scalar> dx0;
    dx0.copyFrom(x0);

    // LEG 1: lduMatrix::Tmul (lduMatrixATmul.C), written out, against deviceAmul on the exchanged view
    {
        std::vector<scalar> refT((std::size_t)nC), refA((std::size_t)nC);
        for (label c = 0; c < nC; ++c)
        {
            refT[(std::size_t)c] = diagC[(std::size_t)c]*x0[(std::size_t)c];
            refA[(std::size_t)c] = diagC[(std::size_t)c]*x0[(std::size_t)c];
        }
        for (label f = 0; f < nIf; ++f)
        {
            refT[(std::size_t)nei[f]] += M.upper[(std::size_t)f]*x0[(std::size_t)own[f]];
            refT[(std::size_t)own[f]] += M.lower[(std::size_t)f]*x0[(std::size_t)nei[f]];
            refA[(std::size_t)nei[f]] += M.lower[(std::size_t)f]*x0[(std::size_t)own[f]];
            refA[(std::size_t)own[f]] += M.upper[(std::size_t)f]*x0[(std::size_t)nei[f]];
        }
        DeviceBuffer<scalar> dT;
        deviceAmul(T, dx0, dT);
        std::vector<scalar> gotT;
        dT.copyTo(gotT);
        failures += brae::gatecheck::nonFinite("device Tmul", gotT);
        const scalar e = worstRel(gotT, refT);
        const scalar control = worstRel(refA, refT);
        std::printf("        Tmul: device against lduMatrix::Tmul %.3e; Amul against Tmul %.3e\n", (double)e,
                    (double)control);
        // a cell's terms arrive in another order on the device, so this is round-off and not bits
        check(e < scalar(1e-14), "deviceAmul on the exchanged view is lduMatrix::Tmul");
        check(control > scalar(1e-3), "...and Tmul is NOT Amul on this matrix, so the exchange is witnessed");
    }

    // LEG 2: DILUPreconditioner::preconditionT (DILUPreconditioner.C:126-170), written out
    {
        std::vector<scalar> rD = diagC;
        for (label f = 0; f < nIf; ++f)
        {
            rD[(std::size_t)nei[f]] -= M.upper[(std::size_t)f]*M.lower[(std::size_t)f]/rD[(std::size_t)own[f]];
        }
        for (label c = 0; c < nC; ++c)
        {
            rD[(std::size_t)c] = scalar(1)/rD[(std::size_t)c];
        }
        std::vector<scalar> refWT((std::size_t)nC), refWA((std::size_t)nC);
        for (label c = 0; c < nC; ++c)
        {
            refWT[(std::size_t)c] = rD[(std::size_t)c]*x0[(std::size_t)c];
            refWA[(std::size_t)c] = rD[(std::size_t)c]*x0[(std::size_t)c];
        }
        for (label f = 0; f < nIf; ++f)
        {
            refWT[(std::size_t)nei[f]] -= rD[(std::size_t)nei[f]]*M.upper[(std::size_t)f]*refWT[(std::size_t)own[f]];
            refWA[(std::size_t)nei[f]] -= rD[(std::size_t)nei[f]]*M.lower[(std::size_t)f]*refWA[(std::size_t)own[f]];
        }
        for (label f = nIf - 1; f >= 0; --f)
        {
            refWT[(std::size_t)own[f]] -= rD[(std::size_t)own[f]]*M.lower[(std::size_t)f]*refWT[(std::size_t)nei[f]];
            refWA[(std::size_t)own[f]] -= rD[(std::size_t)own[f]]*M.upper[(std::size_t)f]*refWA[(std::size_t)nei[f]];
        }
        DeviceBuffer<scalar> dWT;
        diluApply(T, dilu, dx0, dWT);
        std::vector<scalar> gotWT, gotRD;
        dWT.copyTo(gotWT);
        dilu.rD.copyTo(gotRD);
        failures += brae::gatecheck::nonFinite("device preconditionT", gotWT);
        check(sameBits(gotRD, rD), "rD is BIT-IDENTICAL to DILUPreconditioner::calcReciprocalD");
        check(sameBits(gotWT, refWT), "diluApply on the exchanged view is BIT-IDENTICAL to preconditionT");
        const scalar control = worstRel(refWA, refWT);
        std::printf("        preconditionT against precondition on the same vector: %.3e\n", (double)control);
        check(control > scalar(1e-3), "...and preconditionT is NOT precondition on this matrix");
    }

    // THE SOLVE, against the host reference
    const std::vector<scalar> zeros((std::size_t)nC, 0.0);
    auto hostSolve = [&](
        scalar tol,
        scalar relTol,
        int minIter,
        std::vector<scalar>& x)
    {
        x = zeros;
        return pbicgDILU(M, x, m, fvp, tol, relTol, 1000, minIter);
    };
    auto deviceSolve = [&](
        scalar tol,
        scalar relTol,
        int minIter,
        bool stab,
        std::vector<scalar>& x)
    {
        DeviceBuffer<scalar> dx;
        dx.copyFrom(zeros);
        const scalar nf = deviceNormFactor(A, dx, db, deviceOnes(nC));
        DeviceSolverPerf perf;
        if (stab)
        {
            diluUpdate(A, dilu);
            perf = deviceJacobiBiCGStab(A, db, dx, nf, tol, relTol, 1000, /*checkEvery=*/1, minIter, &dilu);
        }
        else
        {
            perf = devicePBiCGDilu(A, db, dx, nf, tol, relTol, 1000, minIter, dilu);
        }
        dx.copyTo(x);
        return perf;
    };

    // Arm A: THE ITERATE AT A LOOSE relTol, which is what a closure's solve hands back
    {
        std::vector<scalar> xh, xd, xs;
        const SolverPerformance sh = hostSolve(1e-30, 0.01, 0, xh);
        const DeviceSolverPerf sd = deviceSolve(1e-30, 0.01, 0, false, xd);
        const DeviceSolverPerf ss = deviceSolve(1e-30, 0.01, 0, true, xs);
        failures += brae::gatecheck::nonFinite("device iterate", xd);
        std::printf("        relTol 0.01: host %d iterations (final %.6e), device %d (final %.6e); PBiCGStab %d\n",
                    (int)sh.nIterations, (double)sh.finalResidual, (int)sd.nIterations, (double)sd.finalResidual,
                    (int)ss.nIterations);
        const scalar e = worstRel(xd, xh);
        const scalar control = worstRel(xs, xh);
        std::printf("        the iterate: device PBiCG %.3e from the host's, device PBiCGStab %.3e\n", (double)e,
                    (double)control);
        check(sh.nIterations >= 3, "the loose solve takes several iterations, so the count says something");
        check(sd.nIterations == sh.nIterations, "the device takes the host's iteration count at relTol 0.01");
        check(std::fabs(sd.initialResidual - sh.initialResidual) <= scalar(1e-12)*sh.initialResidual,
              "...from the host's initial residual");
        check(std::fabs(sd.finalResidual - sh.finalResidual) <= scalar(1e-9)*sh.finalResidual,
              "...to the host's final residual");
        check(e < scalar(1e-11), "...and hands back the host's ITERATE");
        check(control > scalar(1000)*std::fmax(e, scalar(1e-15)),
              "PBiCGStab in its place hands back a DIFFERENT iterate, so the gate tells the two apart");
    }

    // Arm B: converged
    {
        std::vector<scalar> xh, xd;
        const SolverPerformance sh = hostSolve(1e-12, 0, 0, xh);
        const DeviceSolverPerf sd = deviceSolve(1e-12, 0, 0, false, xd);
        failures += brae::gatecheck::nonFinite("device solution", xd);
        std::printf("        tolerance 1e-12: host %d iterations, device %d; solution %.3e apart\n",
                    (int)sh.nIterations, (int)sd.nIterations, (double)worstRel(xd, xh));
        check(sh.nIterations >= 8, "the converged solve takes many iterations");
        check(sd.nIterations == sh.nIterations, "the device takes the host's iteration count at tolerance 1e-12");
        check(worstRel(xd, xh) < scalar(1e-10), "...to the host's solution");
        check(sd.finalResidual < scalar(1e-12) && sh.finalResidual < scalar(1e-12), "...both below the tolerance");
    }

    // Arm C: minIter, which PBiCG.C honours even when the first residual already passes
    {
        std::vector<scalar> xh, xd;
        const SolverPerformance sh = hostSolve(1e30, 0, 4, xh);
        const DeviceSolverPerf sd = deviceSolve(1e30, 0, 4, false, xd);
        std::printf("        tolerance 1e30 with minIter 4: host %d iterations, device %d\n", (int)sh.nIterations,
                    (int)sd.nIterations);
        check(sh.nIterations == 4 && sd.nIterations == 4, "both run exactly minIter iterations");
        check(worstRel(xd, xh) < scalar(1e-12), "...to the same iterate");
    }

    std::printf("test_device_pbicg: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
