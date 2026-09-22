// DIC on the device must be OpenFOAM's DIC, and PCG with it must stop where OpenFOAM's PCG stops.
//
// WHY THIS GATE EXISTS. interFoam's p_rgh is `solver PCG; preconditioner DIC;` at relTol 0.05 on every
// corrector but the last, so the iterate the solver hands back is NOT the converged solution -- it is
// wherever that preconditioner left it. With the device on a different solver, capillaryRise sat
// 9.2e-04 of |U| from OpenFOAM at the case's own tolerances and 3.5e-08 with every solve tightened: the
// discretisation was right and the stopping point was not. "Converges to the same answer" is therefore
// not the bar here. The bar is the ITERATE at a loose relTol.
//
// THE REFERENCE IS A TRANSCRIPTION, not a re-derivation. `ofDic` below is DICPreconditioner.C written
// out line for line: it reads ONLY `upper`, in natural face order, with OpenFOAM's multiplication
// order. The device does not have a DIC of its own -- it runs the level-scheduled DILU with `lower`
// aliased to `upper`, on the argument that the two .C files are the same functions with that one
// substitution. Legs 1 and 2 are what turn that argument into a measurement.
//
// THE SOLVE ARMS compare against the HOST brae::pcg, which tests/test_pcg.cu holds against OpenFOAM's
// own DICPCG (solution, iteration count and residuals). Two links, each measured.
#include "box_mesh.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include "device_dilu.cuh"
#include "device_gate_finite.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "device_pcg.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "geometric_field.cuh"
#include "pcg.cuh"
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

// Bit equality by memcmp, not by ==, because "bit-identical" is the claim and == is a weaker one:
// -0.0 == +0.0 is true with different bits. (A NaN fails == against itself, which would at least fail
// loudly; the nonFinite guards ahead of these checks name it instead.)
bool sameBits(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    return a.size() == b.size()
        && std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0;
}

// OpenFOAM DICPreconditioner.C, transcribed. lPtr = lowerAddr = owner, uPtr = upperAddr = neighbour.
// There is no `lower` argument because DIC never reads one.
void ofDic(
    const std::vector<label>& owner,
    const std::vector<label>& nei,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& rA,
    std::vector<scalar>& rD,
    std::vector<scalar>& wA)
{
    const std::size_t nCells = diag.size();
    const std::size_t nFaces = nei.size();
    rD = diag;
    for (std::size_t f = 0; f < nFaces; ++f)
    {
        rD[(std::size_t)nei[f]] -= upper[f]*upper[f]/rD[(std::size_t)owner[f]];
    }
    for (std::size_t c = 0; c < nCells; ++c)
    {
        rD[c] = scalar(1)/rD[c];
    }

    wA.assign(nCells, 0);
    for (std::size_t c = 0; c < nCells; ++c)
    {
        wA[c] = rD[c]*rA[c];
    }
    for (std::size_t f = 0; f < nFaces; ++f)
    {
        wA[(std::size_t)nei[f]] -= rD[(std::size_t)nei[f]]*upper[f]*wA[(std::size_t)owner[f]];
    }
    for (std::size_t i = nFaces; i-- > 0; )
    {
        wA[(std::size_t)owner[i]] -= rD[(std::size_t)owner[i]]*upper[i]*wA[(std::size_t)nei[i]];
    }
}

scalar worstRel(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    scalar w = 0, s = 0;
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
    std::printf("== DIC on the device is OpenFOAM's, and PCG with it stops where OpenFOAM's does ==\n");
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

    // A 3-D box, so the DAG has real depth and a cell has several lower- and upper-neighbours.
    const PrimitiveMesh m = boxtest::boxMesh(9, 7, 5);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells();
    const int nIf = (int)m.nInternalFaces();

    // THE PRECONDITIONER ALONE, on a symmetric matrix with varying coefficients.
    // Varying, because on a uniform laplacian every face carries the same number and a sweep that
    // read the wrong face's coefficient would still be exact.
    std::vector<scalar> hDiag((std::size_t)nC), hUp((std::size_t)nIf), hR((std::size_t)nC);
    for (label c = 0; c < nC; ++c)
    {
        hDiag[(std::size_t)c] = 12.0 + 0.5*std::sin(0.7*c);
        hR[(std::size_t)c] = std::cos(0.3*c) + 1.1;
    }
    for (int f = 0; f < nIf; ++f)
    {
        hUp[(std::size_t)f] = -(1.0 + 0.25*std::sin(0.9*f));
    }
    // THE `lower` BUFFER IS DELIBERATELY GARBAGE. OpenFOAM's DIC never reads a lower, and deviceDICPCG
    // aliases it to upper for that reason; if anything on the DIC path read this buffer, Legs 1-2 and
    // every solve arm below would fail by orders of magnitude rather than by a rounding.
    std::vector<scalar> hGarbage((std::size_t)nIf);
    for (int f = 0; f < nIf; ++f)
    {
        hGarbage[(std::size_t)f] = 37.0 + f;
    }

    DeviceBuffer<scalar> D, U, L, r;
    D.copyFrom(hDiag);
    U.copyFrom(hUp);
    L.copyFrom(hGarbage);
    r.copyFrom(hR);
    const DeviceLduView Araw = deviceLduView(dm, D, U, L);
    DeviceLduView A = Araw;
    A.lower = Araw.upper;

    DeviceDilu dic = buildDeviceDilu(m.owner(), m.neighbour(), nC);
    diluUpdate(A, dic);
    DeviceBuffer<scalar> w;
    diluApply(A, dic, r, w);
    std::vector<scalar> gotRD, gotW;
    dic.rD.copyTo(gotRD);
    w.copyTo(gotW);

    std::vector<scalar> refRD, refW;
    ofDic(m.owner(), m.neighbour(), hDiag, hUp, hR, refRD, refW);

    failures += brae::gatecheck::nonFinite("device rD", gotRD);
    failures += brae::gatecheck::nonFinite("device M^-1 r", gotW);
    check(sameBits(gotRD, refRD), "rD is BIT-IDENTICAL to DICPreconditioner::calcReciprocalD");
    check(sameBits(gotW, refW), "M^-1 r is BIT-IDENTICAL to DICPreconditioner::precondition");
    std::printf("        (%d cells, %d internal faces, %d levels)\n", (int)nC, nIf, dic.levels());

    // ...and it is not Jacobi. Without this a fallback to w = r/diag would pass on any matrix whose
    // off-diagonals happened to be small.
    {
        DeviceBuffer<scalar> j;
        deviceJacobi(j, r, A.diag);
        std::vector<scalar> hj;
        j.copyTo(hj);
        const scalar rel = worstRel(hj, gotW);
        std::printf("        (largest relative difference from Jacobi: %.3f)\n", (double)rel);
        check(rel > 0.05, "...and differs materially from Jacobi, so it is a real factorisation");
    }

    // THE SOLVE, against the host PCG+DIC that test_pcg holds against OpenFOAM's DICPCG.
    // ON ITS OWN, HARDER MESH, and the first draft of this gate is why. On the 315-cell box above with a
    // uniform laplacian, DIC reaches relTol 0.05 in ONE iteration -- so "the device takes the host's
    // iteration count" compared 1 with 1, after a single application of the preconditioner. The
    // fixture below is interFoam's own difficulty instead: the laplacian's coefficient is rAUf, which
    // is dt/rho and so JUMPS BY 1000 across the water/air interface, under a smooth source that puts
    // the error in the long wavelengths CG is slowest on.
    const PrimitiveMesh m2 = boxtest::boxMesh(24, 20, 12);
    FvGeometry g2;
    g2.build(m2);
    const std::vector<FvPatch> fvp2 = buildPatches(m2, g2);
    DeviceMesh dm2 = buildDeviceMesh(m2, g2, fvp2);
    const label nC2 = m2.nCells();
    const label nIf2 = m2.nInternalFaces();
    DeviceDilu dic2 = buildDeviceDilu(m2.owner(), m2.neighbour(), nC2);

    scalar yMid = 0;
    for (label c = 0; c < nC2; ++c)
    {
        yMid += g2.C()[(std::size_t)c].y;
    }
    yMid /= nC2;

    GeometricField<scalar> fld;
    fld.internal.assign((std::size_t)nC2, 0.0);
    for (const FvPatch& p : fvp2)
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

    // "water" below yMid, "air" above: rAUf = dt/rho, 1e-3 and 1
    SurfaceScalarField gammaf;
    gammaf.internal.resize((std::size_t)nIf2);
    for (label f = 0; f < nIf2; ++f)
    {
        const scalar yo = g2.C()[(std::size_t)m2.owner()[(std::size_t)f]].y;
        const scalar yn = g2.C()[(std::size_t)m2.neighbour()[(std::size_t)f]].y;
        const scalar go = yo < yMid ? scalar(1e-3) : scalar(1);
        const scalar gn = yn < yMid ? scalar(1e-3) : scalar(1);
        gammaf.internal[(std::size_t)f] = scalar(0.5)*(go + gn);
    }
    gammaf.boundary.resize(fvp2.size());
    for (std::size_t pi = 0; pi < fvp2.size(); ++pi)
    {
        for (label i = 0; i < fvp2[pi].size; ++i)
        {
            const scalar yc = g2.C()[(std::size_t)fvp2[pi].faceCells[i]].y;
            gammaf.boundary[pi].push_back(yc < yMid ? scalar(1e-3) : scalar(1));
        }
    }
    FvScalarMatrix M = fvm::laplacian<scalar>(gammaf, fld, m2, g2, fvp2, /*corrected=*/false);
    for (label c = 0; c < nC2; ++c)
    {
        const vector& x = g2.C()[(std::size_t)c];
        M.source[(std::size_t)c] = g2.V()[(std::size_t)c]*(scalar(1) + std::sin(scalar(1.3)*x.x + scalar(0.7)*x.y));
    }

    std::vector<scalar> diagC = M.diag, b = M.source;
    for (std::size_t pi = 0; pi < fvp2.size(); ++pi)
    {
        for (label i = 0; i < fvp2[pi].size; ++i)
        {
            const label c = fvp2[pi].faceCells[i];
            diagC[(std::size_t)c] += M.internalCoeffs[pi][i];
            b[(std::size_t)c] += M.boundaryCoeffs[pi][i];
        }
    }
    std::vector<scalar> hGarbage2((std::size_t)nIf2);
    for (label f = 0; f < nIf2; ++f)
    {
        hGarbage2[(std::size_t)f] = 37.0 + f;
    }
    DeviceBuffer<scalar> dDiag, dUp, dLo, db;
    dDiag.copyFrom(diagC);
    dUp.copyFrom(M.upper);
    dLo.copyFrom(hGarbage2);
    db.copyFrom(b);
    const DeviceLduView P = deviceLduView(dm2, dDiag, dUp, dLo);
    const std::vector<scalar> zeros((std::size_t)nC2, 0.0);

    auto hostSolve = [&](
        scalar tol,
        scalar relTol,
        std::vector<scalar>& x)
    {
        x = zeros;
        return pcg(M, x, m2, fvp2, tol, relTol, 1000, 0);
    };
    auto deviceSolve = [&](
        scalar tol,
        scalar relTol,
        bool useDic,
        std::vector<scalar>& x)
    {
        DeviceBuffer<scalar> dx;
        dx.copyFrom(zeros);
        const scalar nf = deviceNormFactor(P, dx, db, deviceOnes(nC2));
        DeviceSolverPerf perf;
        if (useDic)
        {
            perf = deviceDICPCG(P, db, dx, nf, tol, relTol, 1000, 0, dic2);
        }
        else
        {
            DeviceLduView S = P;
            S.lower = P.upper;
            perf = deviceJacobiPCG(S, db, dx, nf, tol, relTol, 1000);
        }
        dx.copyTo(x);
        return perf;
    };

    // Arm A: THE ITERATE AT A LOOSE relTol. This is the arm the gate exists for.
    {
        std::vector<scalar> xh, xd, xj;
        const SolverPerformance sh = hostSolve(1e-30, 0.05, xh);
        const DeviceSolverPerf sd = deviceSolve(1e-30, 0.05, true, xd);
        const DeviceSolverPerf sj = deviceSolve(1e-30, 0.05, false, xj);
        failures += brae::gatecheck::nonFinite("device iterate", xd);
        const scalar eDic = worstRel(xd, xh);
        const scalar eJac = worstRel(xj, xh);
        std::printf("  relTol 0.05: host PCG+DIC %d iterations (initial %.6e, final %.6e)\n",
                    sh.nIterations, (double)sh.initialResidual, (double)sh.finalResidual);
        std::printf("               device PCG+DIC %d iterations (initial %.6e, final %.6e); "
                    "iterate %.3e from the host's\n",
                    sd.nIterations, (double)sd.initialResidual, (double)sd.finalResidual, (double)eDic);
        std::printf("               device PCG+Jacobi %d iterations; iterate %.3e from the host's\n",
                    sj.nIterations, (double)eJac);
        // more than a handful, or the count compares too little to mean anything -- see the fixture
        check(sh.nIterations >= 5, "the fixture is hard enough that relTol 0.05 takes five iterations or more");
        check(sd.nIterations == sh.nIterations, "at relTol 0.05 the device takes the HOST's iteration count");
        check(std::fabs(sd.initialResidual - sh.initialResidual) < 1e-12*sh.initialResidual,
              "...from the same normFactor-scaled initial residual");
        check(eDic < 1e-10, "...and hands back the host's ITERATE, to 1e-10");
        // THE CONTROL. Same matrix, same relTol, same CG loop, Jacobi in place of DIC: what the device
        // pressure solve effectively was. If this arm read 1e-10 too, the arm above would be measuring
        // nothing about the preconditioner.
        check(eJac > 1e-3, "...which the SAME loop with Jacobi does not: its iterate is >1e-3 away");
    }

    // Arm B: to convergence. Every correct solver passes this; it is here so that Arm A's agreement
    // cannot be two wrong iterates that happen to coincide.
    {
        std::vector<scalar> xh, xd;
        const SolverPerformance sh = hostSolve(1e-12, 0, xh);
        const DeviceSolverPerf sd = deviceSolve(1e-12, 0, true, xd);
        const scalar e = worstRel(xd, xh);
        std::printf("  tolerance 1e-12: host %d iterations, device %d; solution %.3e apart\n",
                    sh.nIterations, sd.nIterations, (double)e);
        check(sd.nIterations == sh.nIterations, "to 1e-12 the iteration counts agree too");
        check(e < 1e-10 && sd.finalResidual < 1e-12, "...and so does the converged solution");
    }

    // Arm C: a schedule built for another mesh is refused, not run. diluUpdate returns silently on an
    // invalid schedule, which would leave rD at whatever it held and the solve would still converge.
    {
        DeviceDilu empty;
        DeviceBuffer<scalar> dx;
        dx.copyFrom(zeros);
        bool threw = false;
        try
        {
            deviceDICPCG(P, db, dx, scalar(1), 1e-12, 0, 10, 0, empty);
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        check(threw, "an unbuilt level schedule is REFUSED rather than solved with a stale rD");
    }

    std::printf("test_device_dic: %d failures\n", failures);
    return failures ? 1 : 0;
}
