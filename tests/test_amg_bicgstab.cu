// AMG as the PRECONDITIONER of the device BiCGStab -- the solver the ASYMMETRIC pressure matrix takes.
//
// OpenFOAM registers GAMGPreconditioner in the asymmetric constructor table as well as the symmetric one
// (GAMGPreconditioner.C:38-42), so `p { solver PBiCGStab; preconditioner GAMG; }` is a legal setting and the
// like-for-like reference for what this seam does. The hierarchy needs nothing new: buildAMG agglomerates on
// the face weights alone and amgGalerkin already takes OpenFOAM's asymmetric branch. What a nonsymmetric
// operator invalidates is the COARSEST SOLVE -- deviceCoarsePCG is a conjugate gradient, whose step length
// alpha = (r.z)/(p.Ap) is only a step length when p.Ap is an A-norm -- and CG does not announce that: its
// alpha/beta guards make it return finite garbage.
//
// Two legs are the controls for that defect, because it has two halves. Leg C is the KERNEL: it holds the
// new single-block BiCGStab to a host direct solve and shows the CG on the SAME system missing it by
// orders of magnitude. Leg F is the DISPATCH: the same hierarchy and the same V-cycle run with
// asymmetric = true and with false, so a coarsest branch that went back to calling the CG fails there even
// though both kernels still exist.
//
// Every system here is DIAGONALLY DOMINANT by construction (diag = (1+delta)*sum|off-diagonals|), so the
// host Gaussian elimination in Leg C is an oracle and not another iterative guess.
#include "box_mesh.cuh"
#include "device_amg.cuh"
#include "device_amg_coarse.cuh"
#include "device_amg_detail.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "device_pcg.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// A diagonally dominant ASYMMETRIC system on the box addressing: a diffusion operator plus a CONVECTION
// in +x, upper = -(1+c), lower = -(1-c) on the same face with c = asym*(Sf.x/|Sf|).
//
// The asymmetry has to be a COHERENT field, not noise indexed by face number. An earlier version of this
// used c = asym*sin(0.7*f), which is asymmetric face by face and yet produced coarse operators that were
// symmetric to plotting accuracy: the agglomerated coefficient is a SUM of fine coefficients, and a term
// that alternates sign with the face index cancels in that sum. Leg F measured exactly 1.8256e-01 for both
// branches on that matrix -- a test that could not have seen the defect. A convection aligned with the
// mesh survives agglomeration, which is also what a real flux does.
// `delta` is how far the diagonal exceeds the row sum, i.e. how far this is from a singular Laplacian.
void buildSystem(
    const PrimitiveMesh& m,
    const FvGeometry& geo,
    scalar asym,
    scalar delta,
    std::vector<scalar>& diag,
    std::vector<scalar>& upper,
    std::vector<scalar>& lower,
    std::vector<scalar>& src)
{
    const label nC = m.nCells();
    const int nIf = static_cast<int>(m.nInternalFaces());
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    upper.assign(static_cast<std::size_t>(nIf), 0.0);
    lower.assign(static_cast<std::size_t>(nIf), 0.0);
    diag.assign(static_cast<std::size_t>(nC), 0.0);
    for (int f = 0; f < nIf; ++f)
    {
        const std::size_t fi = static_cast<std::size_t>(f);
        const scalar c = asym * geo.Sf()[fi].x / geo.magSf()[fi];   // +x convection, zero on the y/z faces
        upper[fi] = -(1.0 + c);
        lower[fi] = -(1.0 - c);
        diag[static_cast<std::size_t>(own[fi])] += std::fabs(upper[fi]);
        diag[static_cast<std::size_t>(nei[fi])] += std::fabs(lower[fi]);
    }
    for (label c = 0; c < nC; ++c) diag[static_cast<std::size_t>(c)] *= (1.0 + delta);
    src.assign(static_cast<std::size_t>(nC), 0.0);
    for (label c = 0; c < nC; ++c)
        src[static_cast<std::size_t>(c)] = std::sin(0.11 * c) + 0.5 * std::cos(0.037 * c);
}

// Face weights for buildAMG: |Sf| over the INTERNAL faces only (magSf spans every face, owner/nei do not).
std::vector<scalar> internalFaceWeights(const PrimitiveMesh& m, const FvGeometry& g)
{
    const std::size_t nIf = static_cast<std::size_t>(m.nInternalFaces());
    return std::vector<scalar>(g.magSf().begin(), g.magSf().begin() + nIf);
}

std::vector<label> internalOwner(const PrimitiveMesh& m)
{
    const std::size_t nIf = static_cast<std::size_t>(m.nInternalFaces());
    return std::vector<label>(m.owner().begin(), m.owner().begin() + nIf);
}

// Dense A*x - b, max norm, on the host: the residual an oracle can be held to.
scalar hostResidual(
    const std::vector<std::vector<scalar>>& A,
    const std::vector<scalar>& x,
    const std::vector<scalar>& b)
{
    scalar worst = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i)
    {
        scalar s = 0.0;
        for (std::size_t j = 0; j < A.size(); ++j) s += A[i][j] * x[j];
        worst = std::fmax(worst, std::fabs(s - b[i]));
    }
    return worst;
}

// Gaussian elimination with PARTIAL PIVOTING: the direct solve Leg C measures the coarsest solver against.
// Written here rather than called, so the oracle owes nothing to the code under test.
std::vector<scalar> denseSolve(
    std::vector<std::vector<scalar>> A,
    std::vector<scalar> b)
{
    const std::size_t n = b.size();
    for (std::size_t k = 0; k < n; ++k)
    {
        std::size_t piv = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::fabs(A[i][k]) > std::fabs(A[piv][k])) piv = i;
        std::swap(A[k], A[piv]);
        std::swap(b[k], b[piv]);
        for (std::size_t i = k + 1; i < n; ++i)
        {
            const scalar f = A[i][k] / A[k][k];
            if (f == 0.0) continue;
            for (std::size_t j = k; j < n; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    std::vector<scalar> x(n, 0.0);
    for (std::size_t i = n; i-- > 0; )
    {
        scalar s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

std::vector<std::vector<scalar>> denseFrom(
    const PrimitiveMesh& m,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower)
{
    const std::size_t nC = static_cast<std::size_t>(m.nCells());
    const int nIf = static_cast<int>(m.nInternalFaces());
    std::vector<std::vector<scalar>> A(nC, std::vector<scalar>(nC, 0.0));
    for (std::size_t c = 0; c < nC; ++c) A[c][c] = diag[c];
    for (int f = 0; f < nIf; ++f)
    {
        const std::size_t o = static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(f)]);
        const std::size_t n = static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(f)]);
        A[o][n] = upper[static_cast<std::size_t>(f)];
        A[n][o] = lower[static_cast<std::size_t>(f)];
    }
    return A;
}

// The Chebyshev refusal is the one flag that is read from the environment ONCE per process (useChebyshev's
// function-local static), so it cannot be turned on part-way through a run. This mode is what the parent
// re-executes with BRAE_CHEBYSHEV=1 set, so the refusal is exercised through vcycleAt itself and not only
// through the guard it calls.
int chebyshevChild()
{
    const PrimitiveMesh m = boxtest::boxMesh(8, 6, 5);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<scalar> hDiag, hUp, hLo, hB;
    buildSystem(m, g, 0.4, 0.05, hDiag, hUp, hLo, hB);
    DeviceBuffer<scalar> D(hDiag), U(hUp), L(hLo), b(hB), x(static_cast<std::size_t>(m.nCells()));
    const DeviceLduView A = deviceLduView(dm, D, U, L);
    AMGData amg = buildAMG(internalOwner(m), m.neighbour(), internalFaceWeights(m, g), m.nCells());
    amgGalerkin(amg, D, U, L);
    try
    {
        vcycleAt(0, amg, A, b, x, true);
        cudaDeviceSynchronize();
    }
    catch (const std::exception& e)
    {
        std::printf("%s\n", e.what());
        return 3;
    }
    std::printf("NO REFUSAL\n");
    return 0;
}
}   // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--refuse-chebyshev") == 0) return chebyshevChild();

    std::printf("== AMG-preconditioned BiCGStab on an ASYMMETRIC matrix ==\n");

    // ---- the fine system: 24x20x16 = 7680 cells, weakly dominant (delta 0.02) so the diagonal
    // preconditioner has real work to do, strongly asymmetric (upper/lower differ by up to 2*0.4).
    const PrimitiveMesh m = boxtest::boxMesh(24, 20, 16);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const std::size_t nC = static_cast<std::size_t>(m.nCells());

    std::vector<scalar> hDiag, hUp, hLo, hB;
    buildSystem(m, g, 0.4, 0.02, hDiag, hUp, hLo, hB);
    DeviceBuffer<scalar> D(hDiag), U(hUp), L(hLo), b(hB);
    const DeviceLduView A = deviceLduView(dm, D, U, L);
    {
        scalar worst = 0.0;
        for (int f = 0; f < static_cast<int>(m.nInternalFaces()); ++f)
            worst = std::fmax(worst, std::fabs(hUp[static_cast<std::size_t>(f)] - hLo[static_cast<std::size_t>(f)]));
        check(worst > 0.1, "the fine matrix really is asymmetric (max |upper - lower| over the faces)");
        std::printf("        (max |upper - lower| = %.3f over %d internal faces, %zu cells)\n",
                    static_cast<double>(worst), static_cast<int>(m.nInternalFaces()), nC);
    }

    AMGData amg = buildAMG(internalOwner(m), m.neighbour(), internalFaceWeights(m, g), m.nCells());
    amgGalerkin(amg, D, U, L);

    DeviceBuffer<scalar> ones(nC), zero(nC);
    {
        std::vector<scalar> h1(nC, 1.0), h0(nC, 0.0);
        ones.copyFrom(h1);
        zero.copyFrom(h0);
    }
    const scalar normFactor = deviceNormFactor(A, zero, b, ones);

    const scalar tol = 1e-13;
    const int maxIter = 20000;

    // ---- Leg A: the ANSWER does not depend on the preconditioner ------------------------------------
    DeviceBuffer<scalar> xAmg(nC), xDiag(nC);
    xAmg.copyFrom(std::vector<scalar>(nC, 0.0));
    xDiag.copyFrom(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf pAmg =
        deviceJacobiBiCGStab(A, b, xAmg, normFactor, tol, 0.0, maxIter, 1, 0, nullptr, &amg);
    const DeviceSolverPerf pDiag =
        deviceJacobiBiCGStab(A, b, xDiag, normFactor, tol, 0.0, maxIter, 1, 0, nullptr, nullptr);
    std::vector<scalar> hAmg, hDia;
    xAmg.copyTo(hAmg);
    xDiag.copyTo(hDia);
    {
        scalar worst = 0.0, mag = 0.0;
        for (std::size_t c = 0; c < nC; ++c)
        {
            worst = std::fmax(worst, std::fabs(hAmg[c] - hDia[c]));
            mag   = std::fmax(mag, std::fabs(hDia[c]));
        }
        const scalar rel = worst / mag;
        check(pAmg.finalResidual < tol && pDiag.finalResidual < tol, "both arms converged to the requested tolerance");
        check(rel < 1e-10, "the AMG-preconditioned solution matches the diagonal-preconditioned one to 1e-10 relative");
        std::printf("        (max relative difference %.3e; AMG final %.3e, diagonal final %.3e)\n",
                    static_cast<double>(rel), static_cast<double>(pAmg.finalResidual),
                    static_cast<double>(pDiag.finalResidual));
    }

    // ---- Leg B: it is doing something -- strictly fewer iterations ----------------------------------
    // Without this a preconditioner that quietly fell back to the diagonal would pass Leg A perfectly.
    check(pAmg.nIterations < pDiag.nIterations, "the AMG preconditioner takes strictly FEWER iterations than the diagonal");
    std::printf("        (AMG %d iterations, diagonal %d)\n", pAmg.nIterations, pDiag.nIterations);

    // ---- Leg C: the coarsest solve, against a host direct solve -------------------------------------
    // A small STRONGLY asymmetric system (asym 0.9) that fits the single-block coarsest solvers, solved by
    // the new BiCGStab kernel and by the CG it replaces, both held to Gaussian elimination.
    {
        const PrimitiveMesh cm = boxtest::boxMesh(6, 5, 4);
        FvGeometry cg;
        cg.build(cm);
        const std::vector<FvPatch> cfvp = buildPatches(cm, cg);
        DeviceMesh cdm = buildDeviceMesh(cm, cg, cfvp);
        const std::size_t cn = static_cast<std::size_t>(cm.nCells());
        std::vector<scalar> cDiag, cUp, cLo, cB;
        buildSystem(cm, cg, 0.9, 0.05, cDiag, cUp, cLo, cB);
        DeviceBuffer<scalar> cD(cDiag), cU(cUp), cL(cLo), cbuf(cB);
        DeviceBuffer<scalar> xB(cn), xC(cn);
        const DeviceLduView cA = deviceLduView(cdm, cD, cU, cL);
        check(static_cast<int>(cn) <= SB_CG_MAX, "the coarse test system is inside the single-block cap (SB_CG_MAX)");

        const int nIt = 200;
        deviceCoarseBiCGStab(cA, cbuf, xB, nIt);
        deviceCoarsePCG(cA, cbuf, xC, nIt);
        cudaCheck(cudaStreamSynchronize(cudaStreamPerThread), "coarse solves");
        std::vector<scalar> hB2, hC2;
        xB.copyTo(hB2);
        xC.copyTo(hC2);

        const std::vector<std::vector<scalar>> dense = denseFrom(cm, cDiag, cUp, cLo);
        const std::vector<scalar> xRef = denseSolve(dense, cB);
        scalar worst = 0.0, mag = 0.0;
        for (std::size_t c = 0; c < cn; ++c)
        {
            worst = std::fmax(worst, std::fabs(hB2[c] - xRef[c]));
            mag   = std::fmax(mag, std::fabs(xRef[c]));
        }
        const scalar rel = worst / mag;
        const scalar resB = hostResidual(dense, hB2, cB);
        const scalar resC = hostResidual(dense, hC2, cB);
        const scalar resRef = hostResidual(dense, xRef, cB);
        check(rel < 1e-10, "the single-block BiCGStab coarsest solve matches the host direct solve to 1e-10 relative");
        // The fail-proof for the defect this replaces: the CG on the SAME asymmetric system does not get there.
        check(resC > 100.0 * resB, "the coarsest CG on the same asymmetric system is at least 100x worse (the defect)");
        std::printf("        (%zu cells; BiCGStab |Ax-b|inf %.3e, CG %.3e, direct %.3e; BiCGStab relErr %.3e)\n",
                    cn, static_cast<double>(resB), static_cast<double>(resC),
                    static_cast<double>(resRef), static_cast<double>(rel));
    }

    // ---- Leg F: the coarsest DISPATCH is live -------------------------------------------------------
    // Leg C holds the two coarsest KERNELS apart. This holds the two BRANCHES apart -- the same hierarchy
    // and the same V-cycle, run once with asymmetric = true and once with false, which is what this code
    // did before -- so a dispatch that went back to the CG fails here even though both kernels still exist.
    // The V-cycle is driven as a stationary iteration, x += M^-1 (b - A x), because ONE cycle is not
    // enough to see it: a coarsest error that survives is amplified cycle over cycle.
    //
    // The mesh is small ON PURPOSE. 120 cells over one level makes the coarsest grid 60 of them, so the
    // coarsest solve is most of the preconditioner. On the 7680-cell hierarchy above, the coarsest is 60
    // cells of 7680, and one cycle's fine residual is set by the smoother: both branches read 7.2265e-02
    // and 7.2264e-02 there -- a measurement that could not have seen the defect.
    {
        const PrimitiveMesh sm = boxtest::boxMesh(6, 5, 4);
        FvGeometry sg;
        sg.build(sm);
        const std::vector<FvPatch> sfvp = buildPatches(sm, sg);
        DeviceMesh sdm = buildDeviceMesh(sm, sg, sfvp);
        const std::size_t sn = static_cast<std::size_t>(sm.nCells());
        std::vector<scalar> aDiag, aUp, aLo, aB;
        buildSystem(sm, sg, 0.9, 0.02, aDiag, aUp, aLo, aB);
        DeviceBuffer<scalar> aD(aDiag), aU(aUp), aL(aLo), ab(aB);
        const DeviceLduView aA = deviceLduView(sdm, aD, aU, aL);
        AMGData amg2 = buildAMG(internalOwner(sm), sm.neighbour(), internalFaceWeights(sm, sg), sm.nCells());
        amgGalerkin(amg2, aD, aU, aL);

        // The premise of the whole leg: the operator the coarsest solver is handed is itself asymmetric.
        // It is Galerkin-built, so this is not implied by the fine matrix being asymmetric -- an asymmetry
        // that alternates sign face by face cancels in the agglomerated sum.
        std::vector<scalar> cUp, cLo;
        amg2.level.back().cUpper.copyTo(cUp);
        amg2.level.back().cLower.copyTo(cLo);
        scalar gap = 0.0, mag = 0.0;
        for (std::size_t f = 0; f < cUp.size(); ++f)
        {
            gap = std::fmax(gap, std::fabs(cUp[f] - cLo[f]));
            mag = std::fmax(mag, std::fabs(cUp[f]));
        }
        check(gap > 0.1 * mag, "the Galerkin COARSEST operator is itself asymmetric (the premise of this leg)");
        std::printf("        (%d levels, coarsest %d cells: max |upper - lower| %.3e vs max |upper| %.3e)\n",
                    amg2.nLevels(), amg2.level.back().nCoarse,
                    static_cast<double>(gap), static_cast<double>(mag));

        auto stationary = [&](bool asymmetric, int nCycles)
        {
            DeviceBuffer<scalar> x(sn), z(sn), Ax(sn), r(sn);
            x.copyFrom(std::vector<scalar>(sn, 0.0));
            deviceCopy(r, ab);
            for (int k = 0; k < nCycles; ++k)
            {
                vcycleAt(0, amg2, aA, r, z, asymmetric);
                deviceAxpy(1.0, z, x);
                deviceAmul(aA, x, Ax);
                deviceCopy(r, ab);
                deviceAxpy(-1.0, Ax, r);
            }
            return deviceSumMag(r) / deviceSumMag(ab);
        };
        const scalar asymLeft = stationary(true, 10);
        const scalar symLeft  = stationary(false, 10);
        check(asymLeft < 1e-3, "10 asymmetric V-cycles cut the residual by more than 1e3");
        check(symLeft > 100.0 * asymLeft, "the same V-cycle with the CG coarsest solve is at least 100x worse");
        std::printf("        (|b-Ax|/|b| after 10 stationary V-cycles: asymmetric coarsest %.4e, CG coarsest %.4e)\n",
                    static_cast<double>(asymLeft), static_cast<double>(symLeft));
    }

    // ---- Leg D: the refusals ------------------------------------------------------------------------
    {
        DeviceBuffer<scalar> z(nC);
        auto refuses = [&](const char* needle, const char* what)
        {
            bool threw = false, named = false;
            try
            {
                vcycleAt(0, amg, A, b, z, true);
                cudaDeviceSynchronize();
            }
            catch (const std::exception& e)
            {
                threw = true;
                named = std::string(e.what()).find(needle) != std::string::npos;
            }
            check(threw && named, what);
        };
        amg.corrScaling = true;
        refuses("correction scaling", "correction scaling under asymmetric = true throws, naming itself");
        amg.corrScaling = false;
        amg.saSmooth = true;
        refuses("smoothed aggregation", "smoothed aggregation under asymmetric = true throws, naming itself");
        amg.saSmooth = false;

        // Both preconditioners at once is a caller error, not a preference to be resolved silently.
        bool threwBoth = false;
        DeviceDilu dilu = buildDeviceDilu(internalOwner(m), m.neighbour(), m.nCells());
        try
        {
            DeviceBuffer<scalar> xx(nC);
            xx.copyFrom(std::vector<scalar>(nC, 0.0));
            deviceJacobiBiCGStab(A, b, xx, normFactor, tol, 0.0, 10, 1, 0, &dilu, &amg);
        }
        catch (const std::exception&) { threwBoth = true; }
        check(threwBoth, "passing BOTH a DILU factorisation and an AMG hierarchy throws");

        // Chebyshev: read from the environment once per process, so re-run this binary with it set.
        const std::string cmd = "BRAE_CHEBYSHEV=1 '" + std::string(argv[0]) + "' --refuse-chebyshev 2>&1";
        std::string out;
        FILE* pipe = popen(cmd.c_str(), "r");
        int status = -1;
        if (pipe)
        {
            char buf[512];
            while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
            status = pclose(pipe);
        }
        const bool named = out.find("Chebyshev") != std::string::npos
                        && out.find("ASYMMETRIC") != std::string::npos;
        check(pipe != nullptr && status != 0 && named,
              "the Chebyshev smoother under asymmetric = true throws, naming itself (child run, BRAE_CHEBYSHEV=1)");
        if (!named) std::printf("        (child output: %s)\n", out.c_str());
    }

    // ---- Leg E: determinism -------------------------------------------------------------------------
    // The second solve replays the cached conditional graph; a solve whose report moved between the two
    // would mean the V-cycle inside the captured body is reading something that is not held fixed.
    {
        DeviceBuffer<scalar> x1(nC), x2(nC);
        x1.copyFrom(std::vector<scalar>(nC, 0.0));
        x2.copyFrom(std::vector<scalar>(nC, 0.0));
        const DeviceSolverPerf r1 = deviceJacobiBiCGStab(A, b, x1, normFactor, tol, 0.0, maxIter, 1, 0, nullptr, &amg);
        x1.copyFrom(std::vector<scalar>(nC, 0.0));
        const DeviceSolverPerf r2 = deviceJacobiBiCGStab(A, b, x1, normFactor, tol, 0.0, maxIter, 1, 0, nullptr, &amg);
        const bool same = r1.initialResidual == r2.initialResidual
                       && r1.finalResidual == r2.finalResidual
                       && r1.nIterations == r2.nIterations;
        check(same, "two identical AMG-preconditioned solves report the same residuals and iteration count, bit for bit");
        std::printf("        (run 1: init %.17e final %.17e iters %d)\n",
                    static_cast<double>(r1.initialResidual), static_cast<double>(r1.finalResidual), r1.nIterations);
        std::printf("        (run 2: init %.17e final %.17e iters %d)\n",
                    static_cast<double>(r2.initialResidual), static_cast<double>(r2.finalResidual), r2.nIterations);
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
