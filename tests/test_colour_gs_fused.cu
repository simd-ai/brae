// The fused colour-ordered Gauss-Seidel engine runs OpenFOAM's smoothSolver loop, per component,
// around a colour sweep -- and nothing else.
//
// THE REFERENCE IS A TRANSCRIPTION of smoothSolver::solve (smoothSolver.C:82-221) and of
// SolverPerformance::checkConvergence (SolverPerformance.C:62-88), run on the host around the SAME
// colour-ordered sweep the device performs. Cells within a colour share no face, so the order within
// a colour is immaterial and the device's per-launch parallelism is the host's sequential walk of the
// same colour. What the device is held to: the same nIterations per component (the stop rule is the
// whole point -- a loose relTol stops the solve where the rule says, and a rule that is off by one
// sweep is a different outer iteration), initial and final residuals within 1e-13 relative, psi within
// 1e-13 of the field's magnitude. The device may contract a*b-c into FMA where the host does not, so
// the numbers are not required to be bit-identical; 1e-13 is where they must agree (the printed psi
// difference has been 0.0 on this box before and after the colour-major layout). A residual that sits
// at the rounding floor of its own terms (the near-exact arms) cannot be compared relatively at all;
// those use the floor, DBL_EPSILON times the sum of the terms' magnitudes over the normFactor, and
// say so.
//
// Every stop-rule branch gets an arm, because each is a place a rewrite can quietly diverge from
// OpenFOAM: the nSweeps < 0 fixed count, the overshoot past maxIter, minIter overriding a passing
// initial residual, a component that passes before the first sweep and must be left untouched, and
// a component compacted out of the launches while the others keep sweeping (the three diagonals are
// scaled 1, 2, 4 so their sweep counts under relTol 0.1 cannot all be equal).
//
// CONTROLS. (g) OpenFOAM's own natural-order GaussSeidelSmoother, transcribed, on the same system
// under the same rule: it must leave a DIFFERENT iterate -- that is the order dependence the colour
// path announces, and a test that could not see it would be blind to the substitution. (h) the
// colouring the sweep is handed is the one it runs: a grouping that is not a colouring is refused at
// the build, a colouring whose arrays are missing is refused at the solve, and a VALID colouring
// with the colours in the opposite order must give an iterate that matches the host reference on
// that order and differs from the reference on the original order -- a deterministic difference,
// where a corrupted colouring only produces a data race whose outcome depends on warp scheduling.
// (i) the colour-major layout itself: the permutation is a bijection with each colour's cells
// ascending, every row's entries are the losort-then-owned order of the device's own addressing,
// and A x through the layout equals deviceAmul on the natural layout to the rows' rounding floor
// (the two sum a row in different orders). (j) every refusal the engine documents, held to its
// wording: nSweeps 0, a cyclic interface, a colouring of another size, a component with its own
// upper, nComp 4, and a second mesh of the SAME size whose addressing lives elsewhere -- the one the
// size check is blind to.
//
// The symmetric arm (b) is also the proof that the engine's skipped launches are no-ops: the host
// reference performs every colour launch of every pass, the device skips the colour a pass starts
// on when the previous pass ended on it, and the two iterates still agree.
//
// (k) THE RESIDUAL FROM THE SWEEP. On two colours the engine writes each pass's residual from the
// sweep launches themselves (the block's last launch from its rows' new values, the next block's
// first launch -- issued speculatively -- from its rows' old ones) and rolls a stopping component's
// speculated rows back; every colouring else keeps the explicit residual pass. The oracles are the
// engine's own paths that run no speculation: (k1) the vector those two launches write is BIT FOR
// BIT what the explicit pass computes over the same state (a solve seeded with the final psi that
// sweeps nothing leaves its initial, explicit, pass in the colouring's rP), with a control showing
// OpenFOAM's own summation order gives other bits so memcmp is not vacuous; (k2) a component that
// stopped after a speculation has exactly the psi a fixed-count solve (nSweeps -n: no residual, no
// speculation) of its n counted sweeps leaves, with the fail-proof that sweep n+1 would move it;
// (k3) the 2-colour box with one class cut in two halves is a 3-class colouring on the IDENTICAL
// permuted layout, so the explicit path solves the same problem and must give the same counts,
// psi and residual vectors to the bit and the reported residuals to the rounding of their sum (see
// (l3) for why not to the bit); (k4) the symmetric schedule, whose block ends on colour 0 and whose
// speculation is therefore colour 1, under (k2) and (k3). Each arm prints which path the engine
// took. The (k) arms run with the colouring's writeResidualVector on, since they read the residual
// rows; every arm before them runs the production path, which stores none.
//
// (l) THE REDUCTION IN THE LAUNCH. The launches that produce residual rows reduce |r| in their own
// thread blocks into per-block partials, and one final kernel sums the partials, all in fixed
// orders; no residual vector is written on the production path. (l1) each reported residual is the
// host's exact (quad precision) sum of |r| over the rows of the bit-identical vector, over nf, to
// 4*DBL_EPSILON of that sum -- a block left out of the partition, counted twice, or read stale
// from the previous pass (the residual falls about 10x per sweep here) is off by far more -- for
// the fused launches' pass and for the explicit kernel's; and two identical solves report the same
// bits (no atomics anywhere). (l2) the vector write changes no arithmetic: the same solve with it
// off reports the same counts and residuals and leaves the same psi to the bit, and a sentinel in
// rP survives it. (l3) the 3-class split's explicit kernel against the 2-colour fused launches:
// the same per-row numbers in both, both fixed-order sums, but NOT the same order -- the block
// partition follows the colour bounds, and the split cuts colour 1's rows into two blocks where
// the 2-colour layout has one, so the partials pair different rows and the sums can differ in the
// last bits; they are held to the rounding of the sum with the actual difference printed.
#include "box_mesh.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include "device_colour_gauss_seidel.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

namespace
{

constexpr int NCOMP = 3;
constexpr scalar REL = 1e-13;

int failures = 0;

void check(
    bool ok,
    const std::string& what)
{
    if (!ok)
    {
        std::printf("  FAIL: %s\n", what.c_str());
        ++failures;
    }
    else
    {
        std::printf("  ok:   %s\n", what.c_str());
    }
}

// The shared LDU on the host, with the addressing read back from the DeviceMesh so the reference
// indexes exactly the arrays the device kernels index.
struct HostLdu
{
    int nC = 0;
    int nIf = 0;
    std::vector<label> owner, nei, ownerStart, losort, losortStart;
    std::vector<scalar> upper, lower;
};

struct HostColouring
{
    int nColours = 0;
    std::vector<label> cells, start;
};

HostColouring hostColouring(const DeviceCellColouring& col)
{
    HostColouring h;
    h.nColours = col.nColours;
    col.cells.copyTo(h.cells);
    h.start = col.startH;
    return h;
}

struct System
{
    std::vector<scalar> diag, b, psi0;
};

// lduMatrix::Amul's terms per cell (the cell-based loop, lduMatrixATmul.C:92-120), as a gather.
scalar rowProduct(
    const HostLdu& A,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& psi,
    label c)
{
    scalar s = diag[(std::size_t)c]*psi[(std::size_t)c];
    for (label f = A.ownerStart[(std::size_t)c]; f < A.ownerStart[(std::size_t)c + 1]; ++f)
        s += A.upper[(std::size_t)f]*psi[(std::size_t)A.nei[(std::size_t)f]];
    for (label j = A.losortStart[(std::size_t)c]; j < A.losortStart[(std::size_t)c + 1]; ++j)
    {
        const label f = A.losort[(std::size_t)j];
        s += A.lower[(std::size_t)f]*psi[(std::size_t)A.owner[(std::size_t)f]];
    }
    return s;
}

// gSumMag(residual) with residual = source - A psi (lduMatrix::residual, lduMatrixATmul.C:268-340).
scalar residualSumMag(
    const HostLdu& A,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& b,
    const std::vector<scalar>& psi)
{
    scalar s = 0;
    for (label c = 0; c < A.nC; ++c)
        s += std::fabs(b[(std::size_t)c] - rowProduct(A, diag, psi, c));
    return s;
}

// The magnitudes one row of A psi is a sum of: the rounding floor of that row under any summation
// order.
scalar rowTermSum(
    const HostLdu& A,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& psi,
    label c)
{
    scalar s = std::fabs(diag[(std::size_t)c]*psi[(std::size_t)c]);
    for (label f = A.ownerStart[(std::size_t)c]; f < A.ownerStart[(std::size_t)c + 1]; ++f)
        s += std::fabs(A.upper[(std::size_t)f]*psi[(std::size_t)A.nei[(std::size_t)f]]);
    for (label j = A.losortStart[(std::size_t)c]; j < A.losortStart[(std::size_t)c + 1]; ++j)
    {
        const label f = A.losort[(std::size_t)j];
        s += std::fabs(A.lower[(std::size_t)f]*psi[(std::size_t)A.owner[(std::size_t)f]]);
    }
    return s;
}

// The magnitudes the residual is a difference of: the rounding floor of sum|b - A psi|.
scalar termSum(
    const HostLdu& A,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& b,
    const std::vector<scalar>& psi)
{
    scalar s = 0;
    for (label c = 0; c < A.nC; ++c)
        s += std::fabs(b[(std::size_t)c]) + rowTermSum(A, diag, psi, c);
    return s;
}

// lduMatrix::solver::normFactor (lduMatrixSolver.C:235-274): sumA (lduMatrixATmul.C:219-265) times
// the average of psi, then sum(|Apsi - tmp| + |source - tmp|) + small_ (1e-20).
scalar ofNormFactor(
    const HostLdu& A,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& b,
    const std::vector<scalar>& psi)
{
    std::vector<scalar> sumA(diag);
    for (int f = 0; f < A.nIf; ++f)
    {
        sumA[(std::size_t)A.nei[(std::size_t)f]] += A.lower[(std::size_t)f];
        sumA[(std::size_t)A.owner[(std::size_t)f]] += A.upper[(std::size_t)f];
    }
    scalar avg = 0;
    for (label c = 0; c < A.nC; ++c)
        avg += psi[(std::size_t)c];
    avg /= scalar(A.nC);
    scalar nf = 0;
    for (label c = 0; c < A.nC; ++c)
    {
        const scalar tmp = sumA[(std::size_t)c]*avg;
        nf += std::fabs(rowProduct(A, diag, psi, c) - tmp) + std::fabs(b[(std::size_t)c] - tmp);
    }
    return nf + 1e-20;
}

// One cell's update, the terms in the device kernel's order: the source, the lower terms in losort
// order, the upper terms in face order (GaussSeidelSmoother.C:157-160), then the unguarded division
// (:163).
void cellUpdate(
    const HostLdu& A,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    label c)
{
    scalar acc = b[(std::size_t)c];
    for (label j = A.losortStart[(std::size_t)c]; j < A.losortStart[(std::size_t)c + 1]; ++j)
    {
        const label f = A.losort[(std::size_t)j];
        acc -= A.lower[(std::size_t)f]*psi[(std::size_t)A.owner[(std::size_t)f]];
    }
    for (label f = A.ownerStart[(std::size_t)c]; f < A.ownerStart[(std::size_t)c + 1]; ++f)
        acc -= A.upper[(std::size_t)f]*psi[(std::size_t)A.nei[(std::size_t)f]];
    psi[(std::size_t)c] = acc/diag[(std::size_t)c];
}

// The colour sweep: colours ascending, then (symmetric) descending. Sequential within a colour,
// which is the same as parallel because same-colour cells share no face.
void colourSweep(
    const HostLdu& A,
    const HostColouring& col,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    bool symmetric)
{
    auto colour = [&](int k)
    {
        for (label i = col.start[(std::size_t)k]; i < col.start[(std::size_t)k + 1]; ++i)
            cellUpdate(A, diag, b, psi, col.cells[(std::size_t)i]);
    };
    for (int k = 0; k < col.nColours; ++k)
        colour(k);
    if (!symmetric) return;
    for (int k = col.nColours - 1; k >= 0; --k)
        colour(k);
}

// OpenFOAM's GaussSeidelSmoother::smooth, one sweep, transcribed (GaussSeidelSmoother.C:116-173):
// natural cell order with the bPrime scatter. This is the ORDER the colour sweep does not have.
void naturalSweep(
    const HostLdu& A,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi)
{
    std::vector<scalar> bPrime(b);
    label fEnd = A.ownerStart[0];
    for (label celli = 0; celli < A.nC; ++celli)
    {
        const label fStart = fEnd;
        fEnd = A.ownerStart[(std::size_t)celli + 1];
        scalar psii = bPrime[(std::size_t)celli];
        for (label facei = fStart; facei < fEnd; ++facei)
            psii -= A.upper[(std::size_t)facei]*psi[(std::size_t)A.nei[(std::size_t)facei]];
        psii /= diag[(std::size_t)celli];
        for (label facei = fStart; facei < fEnd; ++facei)
            bPrime[(std::size_t)A.nei[(std::size_t)facei]] -= A.lower[(std::size_t)facei]*psii;
        psi[(std::size_t)celli] = psii;
    }
}

struct HostRun
{
    scalar init = 0;
    scalar fin = 0;
    int nIter = 0;
};

// SolverPerformance::checkConvergence, SolverPerformance.C:62-88.
bool ofConverged(
    scalar tol,
    scalar relTol,
    scalar finalRes,
    scalar initRes)
{
    return finalRes < tol || (relTol > 1e-20 && finalRes < relTol*initRes);
}

// smoothSolver::solve, smoothSolver.C:82-221, around any one-sweep callable.
template <class Sweep>
HostRun ofSmoothSolve(
    Sweep sweep,
    const HostLdu& A,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    scalar nf,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    int nSweeps)
{
    HostRun p;
    if (nSweeps < 0)
    {
        for (int s = 0; s < -nSweeps; ++s)
            sweep(psi);
        p.nIter = -nSweeps;
        return p;
    }
    p.init = residualSumMag(A, diag, b, psi)/nf;
    p.fin = p.init;
    if (minIter > 0 || !ofConverged(tol, relTol, p.fin, p.init))
    {
        do
        {
            for (int s = 0; s < nSweeps; ++s)
                sweep(psi);
            p.fin = residualSumMag(A, diag, b, psi)/nf;
        }
        while (((p.nIter += nSweeps) < maxIter && !ofConverged(tol, relTol, p.fin, p.init)) || p.nIter < minIter);
    }
    return p;
}

struct Params
{
    scalar tol = 1e-12;
    scalar relTol = 0;
    int maxIter = 1000;
    int minIter = 0;
    int nSweeps = 1;
    bool symmetric = false;
    bool nfOnDevice = false;
};

struct DeviceRun
{
    DeviceSolverPerf perf[NCOMP];
    std::vector<scalar> psi[NCOMP];
};

// The three components on the device, with comps[k] pointing into A[k], B[k], P[k]. Filled in place
// (not returned) because comps holds the addresses of its own members.
struct DeviceSystem
{
    DeviceBuffer<scalar> D[NCOMP], B[NCOMP], P[NCOMP], dNf;
    DeviceLduView A[NCOMP];
    GSFusedComponent comps[NCOMP];
};

void fillDeviceSystem(
    DeviceSystem& s,
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& U,
    const DeviceBuffer<scalar>& L,
    const System* sys,
    const scalar* nf,
    bool nfOnDevice)
{
    if (nfOnDevice) s.dNf.copyFrom(std::vector<scalar>(nf, nf + NCOMP));
    for (int k = 0; k < NCOMP; ++k)
    {
        s.D[k].copyFrom(sys[k].diag);
        s.B[k].copyFrom(sys[k].b);
        s.P[k].copyFrom(sys[k].psi0);
        s.A[k] = deviceLduView(dm, s.D[k], U, L);
        s.comps[k].A = &s.A[k];
        s.comps[k].b = &s.B[k];
        s.comps[k].psi = &s.P[k];
        // -1 poisons the host value when the device one must win: a solver reading the wrong one
        // reports negative residuals and fails the comparison.
        s.comps[k].normFactor = nfOnDevice ? scalar(-1) : nf[k];
        s.comps[k].dNormFactor = nfOnDevice ? s.dNf.data() + k : nullptr;
    }
}

DeviceRun runDevice(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& U,
    const DeviceBuffer<scalar>& L,
    const System* sys,
    const scalar* nf,
    const DeviceCellColouring& col,
    const Params& p)
{
    DeviceSystem s;
    fillDeviceSystem(s, dm, U, L, sys, nf, p.nfOnDevice);
    DeviceRun out;
    deviceColourGaussSeidelFused(NCOMP, s.comps, col, p.tol, p.relTol, p.maxIter, p.minIter, p.nSweeps, p.symmetric, out.perf);
    for (int k = 0; k < NCOMP; ++k)
        s.P[k].copyTo(out.psi[k]);
    return out;
}

// What a call threw, or "" when it ran: the refusal arms hold the message's wording, since a
// refusal that names the wrong thing sends the reader to the wrong place.
template <class Call>
std::string refusalOf(Call call)
{
    try
    {
        call();
    }
    catch (const std::runtime_error& e)
    {
        return e.what();
    }
    return "";
}

// max_c |a - b| over max_c |b|: relative to the field's magnitude, so a cell where psi crosses zero
// does not turn a rounding difference into a relative one.
scalar fieldRelDiff(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    scalar diff = 0;
    scalar mag = 0;
    for (std::size_t c = 0; c < a.size(); ++c)
    {
        diff = std::fmax(diff, std::fabs(a[c] - b[c]));
        mag = std::fmax(mag, std::fabs(b[c]));
    }
    return diff/std::fmax(mag, scalar(1e-300));
}

bool residualsAgree(
    scalar got,
    scalar ref,
    scalar floor)
{
    const scalar d = std::fabs(got - ref);
    return d <= REL*std::fmax(std::fabs(got), std::fabs(ref)) || d <= floor;
}

// Run the device and the host reference on the same systems under the same rule, and hold them
// together. Returns the reference runs (with the reference iterates in refPsi) for the controls.
struct ArmResult
{
    HostRun ref[NCOMP];
    std::vector<scalar> refPsi[NCOMP];
    DeviceRun dev;
};

ArmResult compareArm(
    const char* name,
    const HostLdu& A,
    const HostColouring& hcol,
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& U,
    const DeviceBuffer<scalar>& L,
    const System* sys,
    const scalar* nf,
    const DeviceCellColouring& col,
    const Params& p)
{
    ArmResult r;
    r.dev = runDevice(dm, U, L, sys, nf, col, p);
    bool ok = true;
    std::string detail;
    for (int k = 0; k < NCOMP; ++k)
    {
        r.refPsi[k] = sys[k].psi0;
        r.ref[k] = ofSmoothSolve([&](std::vector<scalar>& psi) { colourSweep(A, hcol, sys[k].diag, sys[k].b, psi, p.symmetric); },
                                 A, sys[k].diag, sys[k].b, r.refPsi[k], nf[k], p.tol, p.relTol, p.maxIter, p.minIter, p.nSweeps);
        const scalar floor = 4*DBL_EPSILON*termSum(A, sys[k].diag, sys[k].b, r.refPsi[k])/nf[k];
        const DeviceSolverPerf& g = r.dev.perf[k];
        const bool iterOk = g.nIterations == r.ref[k].nIter;
        const bool initOk = residualsAgree(g.initialResidual, r.ref[k].init, floor);
        const bool finOk = residualsAgree(g.finalResidual, r.ref[k].fin, floor);
        const scalar psiDiff = fieldRelDiff(r.dev.psi[k], r.refPsi[k]);
        const bool psiOk = psiDiff <= REL;
        ok = ok && iterOk && initOk && finOk && psiOk;
        char buf[256];
        std::snprintf(buf, sizeof buf, " [k%d: nIter %d/%d init %.3e/%.3e final %.3e/%.3e psi %.1e]",
                      k, g.nIterations, r.ref[k].nIter, (double)g.initialResidual, (double)r.ref[k].init,
                      (double)g.finalResidual, (double)r.ref[k].fin, (double)psiDiff);
        detail += buf;
    }
    // Which residual path the engine took on this colouring under these parameters.
    const char* path = p.nSweeps < 0 ? "no residual (fixed count)"
                     : deviceColourGaussSeidelFusesResidual(col) ? "fused residual" : "explicit residual";
    detail += std::string(" [") + path + "]";
    check(ok, std::string(name) + detail);
    return r;
}

bool bitsEqual(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0;
}

// sum|r| of a residual vector on the host in long double (binary128 on this aarch64 box; 64-bit
// mantissa on x86): 315 doubles within a few decades of one another add exactly there, so the
// result is the exact sum rounded once -- the oracle for a device sum in any fixed order.
scalar hostSumMag(const std::vector<scalar>& r)
{
    long double s = 0;
    for (scalar v : r)
        s += std::fabs((long double)v);
    return (scalar)s;
}

// Two reported residuals (sum|r|/nf) that are fixed-order sums of the SAME per-row numbers in
// different orders agree to the rounding of the sum: 4*DBL_EPSILON of sum|r| over nf.
// sumMagOverNf is that magnitude: the host's sum of the stored rows over nf where the rows are at
// hand (a solve's last pass), else the reported residual itself, which is that sum to its own
// rounding (the initial pass's rows are overwritten by the passes after it).
bool sumsAgree(
    scalar a,
    scalar b,
    scalar sumMagOverNf)
{
    return std::fabs(a - b) <= 4*DBL_EPSILON*sumMagOverNf;
}

// The classes of a colouring reordered: colour k of the result is colour order[k] of the input.
void reorderClasses(
    const HostColouring& in,
    const std::vector<int>& order,
    std::vector<label>& cells,
    std::vector<label>& start)
{
    cells.clear();
    start.assign(1, 0);
    for (int k : order)
    {
        for (label i = in.start[(std::size_t)k]; i < in.start[(std::size_t)k + 1]; ++i)
            cells.push_back(in.cells[(std::size_t)i]);
        start.push_back((label)cells.size());
    }
}

}   // namespace

int main()
{
    std::printf("== colour Gauss-Seidel, fused: OpenFOAM's smoothSolver loop per component around the colour sweep ==\n");

    // A 3-D box, so cells have up to six neighbours and the colouring has real structure. An
    // ASYMMETRIC matrix (upper != lower): the sweep reads upper on owned faces and lower on
    // neighboured ones, and a symmetric matrix would hide a swap of the two.
    const PrimitiveMesh m = boxtest::boxMesh(9, 7, 5);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    HostLdu A;
    A.nC = (int)m.nCells();
    A.nIf = (int)m.nInternalFaces();
    A.owner.assign(m.owner().begin(), m.owner().begin() + A.nIf);
    A.nei = m.neighbour();
    dm.ownerStart.copyTo(A.ownerStart);
    dm.losort.copyTo(A.losort);
    dm.losortStart.copyTo(A.losortStart);
    A.upper.resize((std::size_t)A.nIf);
    A.lower.resize((std::size_t)A.nIf);
    for (int f = 0; f < A.nIf; ++f)
    {
        A.upper[(std::size_t)f] = -(1.0 + 0.25*std::sin(0.9*f));
        A.lower[(std::size_t)f] = -(1.0 + 0.25*std::cos(0.4*f));
    }
    DeviceBuffer<scalar> U, L;
    U.copyFrom(A.upper);
    L.copyFrom(A.lower);

    // Three components with their own diagonal (diag_k = base*2^k, standing in for the per-component
    // fold of fvMatrixSolve.C:169 and scaled so the three converge at different rates), source and
    // initial guess, all different, so a component read through another's operands is visible.
    System std3[NCOMP];
    for (int k = 0; k < NCOMP; ++k)
    {
        std3[k].diag.resize((std::size_t)A.nC);
        std3[k].b.resize((std::size_t)A.nC);
        std3[k].psi0.resize((std::size_t)A.nC);
        for (label c = 0; c < A.nC; ++c)
        {
            std3[k].diag[(std::size_t)c] = (12.0 + 0.5*std::sin(0.7*c))*scalar(1 << k);
            std3[k].b[(std::size_t)c] = std::cos(0.3*c + 0.7*k) + 1.1 + 0.2*k;
            std3[k].psi0[(std::size_t)c] = 0.1*std::sin(0.5*c + 2.0*k);
        }
    }
    // The near-exact systems: b = A psi*, initial guess psi* plus a 1e-6 perturbation, so the initial
    // residual is a real number well below a tol of 1e-4 -- not a rounding-level one that no two
    // summation orders would agree on.
    System near3[NCOMP];
    for (int k = 0; k < NCOMP; ++k)
    {
        near3[k].diag = std3[k].diag;
        std::vector<scalar> star((std::size_t)A.nC);
        for (label c = 0; c < A.nC; ++c)
            star[(std::size_t)c] = 0.3*std::cos(0.2*c + k) + 0.5;
        near3[k].b.resize((std::size_t)A.nC);
        near3[k].psi0.resize((std::size_t)A.nC);
        for (label c = 0; c < A.nC; ++c)
        {
            near3[k].b[(std::size_t)c] = rowProduct(A, near3[k].diag, star, c);
            near3[k].psi0[(std::size_t)c] = star[(std::size_t)c] + 1e-6*std::sin(0.9*c + k);
        }
    }
    auto normFactors = [&](const System* sys, scalar* nf)
    {
        for (int k = 0; k < NCOMP; ++k)
            nf[k] = ofNormFactor(A, sys[k].diag, sys[k].b, sys[k].psi0);
    };
    scalar nfStd[NCOMP], nfNear[NCOMP];
    normFactors(std3, nfStd);
    normFactors(near3, nfNear);

    const DeviceCellColouring col = buildDeviceCellColouring(A.owner, A.nei, A.nC);
    const HostColouring hcol = hostColouring(col);
    std::printf("  (%d cells, %d internal faces, %d colours)\n", A.nC, A.nIf, col.nColours);

    // (a) GaussSeidel, tol 1e-12 / relTol 0.1, nSweeps 1: the common momentum setting. The three
    // diagonals converge at different rates, so at least one component is compacted out of the
    // launches while another keeps sweeping -- the mid-loop freeze -- and the iterate must still be
    // the reference's for every one of them.
    Params pa;
    pa.relTol = 0.1;
    const ArmResult ra = compareArm("(a) GaussSeidel relTol 0.1", A, hcol, dm, U, L, std3, nfStd, col, pa);
    check(ra.ref[0].nIter > 0 && ra.ref[0].fin > pa.tol, "(a) the stop was the relative test, not the absolute one");
    {
        const int n0 = ra.dev.perf[0].nIterations;
        const int n1 = ra.dev.perf[1].nIterations;
        const int n2 = ra.dev.perf[2].nIterations;
        char buf[160];
        std::snprintf(buf, sizeof buf, "(a) sweep counts %d/%d/%d are not all equal: a component froze mid-loop while the others swept on", n0, n1, n2);
        check(!(n0 == n1 && n1 == n2), buf);
    }

    // (b) symGaussSeidel, the normFactor on the device (dNormFactor must win over the poisoned host value).
    Params pb = pa;
    pb.symmetric = true;
    pb.nfOnDevice = true;
    const ArmResult rb = compareArm("(b) symGaussSeidel relTol 0.1, normFactor on the device", A, hcol, dm, U, L, std3, nfStd, col, pb);

    // (c) nSweeps 2 with maxIter 5: nIterations is incremented BEFORE the maxIter test
    // (smoothSolver.C:205), so the solve takes 2, 4, 6 and reports 6.
    Params pc;
    pc.nSweeps = 2;
    pc.maxIter = 5;
    const ArmResult rc = compareArm("(c) nSweeps 2, maxIter 5", A, hcol, dm, U, L, std3, nfStd, col, pc);
    check(rc.dev.perf[0].nIterations == 6 && rc.dev.perf[1].nIterations == 6 && rc.dev.perf[2].nIterations == 6,
          "(c) overshoots maxIter 5 to nIterations 6 as OpenFOAM does");
    check(rc.ref[0].fin > pc.tol, "(c) the cap, not the tolerance, ended the solve");

    // (d) minIter 3 on an initial guess that already passes: smoothSolver.C:159-165 enters the loop
    // on minIter alone and :208 keeps it sweeping until nIterations >= minIter.
    Params pd;
    pd.tol = 1e-4;
    pd.minIter = 3;
    const ArmResult rd = compareArm("(d) minIter 3 over a passing initial residual", A, hcol, dm, U, L, near3, nfNear, col, pd);
    check(rd.ref[0].init < pd.tol && rd.ref[1].init < pd.tol && rd.ref[2].init < pd.tol, "(d) the initial residual really passed");
    check(rd.dev.perf[0].nIterations == 3 && rd.dev.perf[1].nIterations == 3 && rd.dev.perf[2].nIterations == 3,
          "(d) still swept exactly 3");

    // (e) nSweeps -2: a fixed count, no residual (smoothSolver.C:95-119), nIterations 2.
    Params pe;
    pe.nSweeps = -2;
    const ArmResult re = compareArm("(e) nSweeps -2", A, hcol, dm, U, L, std3, nfStd, col, pe);
    check(re.dev.perf[0].initialResidual == 0 && re.dev.perf[0].finalResidual == 0 && re.dev.perf[0].nIterations == 2
          && re.dev.perf[2].initialResidual == 0 && re.dev.perf[2].finalResidual == 0 && re.dev.perf[2].nIterations == 2,
          "(e) reports 0, 0, 2");

    // (f) one component (k = 2) whose initial residual passes with minIter 0: 0 sweeps and psi
    // untouched, while the other two solve to tol beside it.
    System mix3[NCOMP] = {std3[0], std3[1], near3[2]};
    scalar nfMix[NCOMP] = {nfStd[0], nfStd[1], nfNear[2]};
    Params pf;
    pf.tol = 1e-4;
    const ArmResult rf = compareArm("(f) component 2 passes before the first sweep", A, hcol, dm, U, L, mix3, nfMix, col, pf);
    check(rf.dev.perf[2].nIterations == 0 && rf.dev.psi[2] == mix3[2].psi0, "(f) component 2: 0 sweeps, psi bit-identical to its initial guess");
    check(rf.dev.perf[0].nIterations > 0 && rf.dev.perf[1].nIterations > 0 && rf.dev.perf[0].finalResidual < pf.tol,
          "(f) components 0 and 1 swept to tol beside it");

    // (g) CONTROL: OpenFOAM's natural-order GaussSeidelSmoother under (a)'s rule leaves a different
    // iterate. If it did not, the colour order would be a harmless reordering and the announce that
    // names it a substitution would be wrong; it is not harmless, and this measures by how much.
    {
        scalar worst = 0;
        std::string detail;
        for (int k = 0; k < NCOMP; ++k)
        {
            std::vector<scalar> psi = std3[k].psi0;
            const HostRun nat = ofSmoothSolve([&](std::vector<scalar>& x) { naturalSweep(A, std3[k].diag, std3[k].b, x); },
                                              A, std3[k].diag, std3[k].b, psi, nfStd[k], pa.tol, pa.relTol, pa.maxIter, pa.minIter, pa.nSweeps);
            const scalar d = fieldRelDiff(psi, ra.refPsi[k]);
            worst = std::fmax(worst, d);
            char buf[128];
            std::snprintf(buf, sizeof buf, " [k%d: natural %d sweeps, colour %d, iterates differ %.2e]", k, nat.nIter, ra.ref[k].nIter, (double)d);
            detail += buf;
        }
        check(worst > 1e-6, "(g) control: OpenFOAM's natural order leaves a different iterate" + detail);
    }

    // (h) CONTROL: the sweep runs the colouring it is handed. Deterministic checks, in place of a
    // corrupted colouring: a cell moved into a face-neighbour's colour makes its launch a data race
    // whose outcome depends on warp scheduling -- it reproduces the sound iterate whenever the moved
    // cell's thread happens to run in the right place -- so no placement of it makes a control that
    // fails only when the sweep is broken.
    {
        const label c = 4 + 9*(3 + 7*2);
        const label n = c + 1;
        bool adjacent = false;
        for (int f = 0; f < A.nIf; ++f)
            if (A.owner[(std::size_t)f] == c && A.nei[(std::size_t)f] == n) adjacent = true;
        check(adjacent, "(h) the chosen cells share a face");
        std::vector<label> colourOf((std::size_t)A.nC, -1);
        for (int k = 0; k < hcol.nColours; ++k)
            for (label i = hcol.start[(std::size_t)k]; i < hcol.start[(std::size_t)k + 1]; ++i)
                colourOf[(std::size_t)hcol.cells[(std::size_t)i]] = k;
        const int kc = colourOf[(std::size_t)c];
        const int kn = colourOf[(std::size_t)n];
        check(kc != kn, "(h) the sound colouring keeps the two apart");

        // (h1) the grouping with c moved into n's class is not a colouring: the build must refuse
        // it, naming the face, rather than lay it out.
        {
            std::vector<label> cells, start(1, 0);
            for (int k = 0; k < hcol.nColours; ++k)
            {
                if (k == kn) cells.push_back(c);
                for (label i = hcol.start[(std::size_t)k]; i < hcol.start[(std::size_t)k + 1]; ++i)
                {
                    const label cell = hcol.cells[(std::size_t)i];
                    if (cell != c) cells.push_back(cell);
                }
                start.push_back((label)cells.size());
            }
            std::string what;
            try
            {
                buildDeviceCellColouringFromClasses(A.owner, A.nei, A.nC, cells, start);
            }
            catch (const std::runtime_error& e)
            {
                what = e.what();
            }
            check(what.find("of one colour") != std::string::npos,
                  "(h1) a grouping that joins two cells of one class across a face is refused at the build (" + what + ")");
        }

        // (h2) a grouping that lists a cell twice (and so misses one) is refused as well.
        {
            std::vector<label> cells(hcol.cells);
            cells[(std::size_t)hcol.start[(std::size_t)kn]] = c;
            std::string what;
            try
            {
                buildDeviceCellColouringFromClasses(A.owner, A.nei, A.nC, cells, hcol.start);
            }
            catch (const std::runtime_error& e)
            {
                what = e.what();
            }
            check(what.find("listed twice") != std::string::npos,
                  "(h2) a grouping that lists a cell twice is refused at the build (" + what + ")");
        }

        // (h3) a colouring whose layout arrays were never built is refused at the solve, not swept.
        {
            DeviceCellColouring half;
            half.nColours = col.nColours;
            half.nCells = A.nC;
            half.nInternalFaces = A.nIf;
            half.startH = col.startH;
            half.valid = true;
            std::string what;
            try
            {
                runDevice(dm, U, L, std3, nfStd, half, pa);
            }
            catch (const std::runtime_error& e)
            {
                what = e.what();
            }
            check(what.find("inconsistent") != std::string::npos,
                  "(h3) a colouring without its layout arrays is refused at the solve (" + what + ")");
        }

        // (h4) the same colouring with its colours in the OPPOSITE order is valid and is a different
        // Gauss-Seidel order: three fixed sweeps on the device must match the host reference on the
        // reversed order and differ from the reference on the original order. A sweep that ignored
        // the colour ranges it was handed could not do both.
        {
            std::vector<int> order;
            for (int k = hcol.nColours - 1; k >= 0; --k)
                order.push_back(k);
            std::vector<label> cells, start;
            reorderClasses(hcol, order, cells, start);
            const DeviceCellColouring rev = buildDeviceCellColouringFromClasses(A.owner, A.nei, A.nC, cells, start);
            const HostColouring hrev = hostColouring(rev);
            Params ph;
            ph.nSweeps = -3;
            const ArmResult rh = compareArm("(h4) the colours in the opposite order, 3 fixed sweeps", A, hrev, dm, U, L, std3, nfStd, rev, ph);
            scalar least = 1e300;
            std::string detail;
            for (int k = 0; k < NCOMP; ++k)
            {
                std::vector<scalar> psi = std3[k].psi0;
                for (int s = 0; s < 3; ++s)
                    colourSweep(A, hcol, std3[k].diag, std3[k].b, psi, false);
                const scalar d = fieldRelDiff(rh.dev.psi[k], psi);
                least = std::fmin(least, d);
                char buf[64];
                std::snprintf(buf, sizeof buf, " [k%d: %.2e]", k, (double)d);
                detail += buf;
            }
            check(least > 1e-6, "(h4) control: the opposite colour order leaves a different iterate" + detail);
        }
    }

    // (i) THE LAYOUT. The permutation is a bijection with each colour's cells ascending; every row's
    // entries are, in order, the losort faces of the cell (coefficient lower, neighbour the face's
    // owner) then its owned faces (coefficient upper, neighbour the face's neighbour), read off the
    // device's OWN losort/ownerStart so the summation-order claim is checked rather than asserted;
    // and A x through the layout equals deviceAmul on the natural layout for a random x.
    {
        std::vector<label> cells, rowStart, nbr, face;
        std::vector<unsigned char> isUpper;
        const std::vector<label>& newIndex = col.newIndex;
        col.cells.copyTo(cells);
        col.rowStart.copyTo(rowStart);
        col.nbr.copyTo(nbr);
        col.face.copyTo(face);
        col.isUpper.copyTo(isUpper);
        bool bijection = (int)cells.size() == A.nC && (int)newIndex.size() == A.nC
                      && col.startH.front() == 0 && col.startH.back() == A.nC;
        std::vector<int> seen((std::size_t)A.nC, 0);
        for (label i = 0; i < A.nC && bijection; ++i)
        {
            const label c = cells[(std::size_t)i];
            bijection = c >= 0 && c < A.nC && !seen[(std::size_t)c] && newIndex[(std::size_t)c] == i;
            if (bijection) seen[(std::size_t)c] = 1;
        }
        bool ascending = true;
        for (int k = 0; k < col.nColours; ++k)
            for (label i = col.startH[(std::size_t)k] + 1; i < col.startH[(std::size_t)k + 1]; ++i)
                if (cells[(std::size_t)i] <= cells[(std::size_t)i - 1]) ascending = false;
        check(bijection, "(i) cells[] and newIndex[] are mutual inverses over every cell");
        check(ascending, "(i) each colour's cells are in ascending original index");

        bool rows = (int)rowStart.size() == A.nC + 1 && rowStart[0] == 0 && (int)rowStart.back() == 2*A.nIf
                 && nbr.size() == (std::size_t)2*A.nIf && face.size() == nbr.size() && isUpper.size() == nbr.size();
        for (label i = 0; i < A.nC && rows; ++i)
        {
            const label c = cells[(std::size_t)i];
            std::size_t e = (std::size_t)rowStart[(std::size_t)i];
            for (label j = A.losortStart[(std::size_t)c]; j < A.losortStart[(std::size_t)c + 1] && rows; ++j, ++e)
            {
                const label f = A.losort[(std::size_t)j];
                rows = e < (std::size_t)rowStart[(std::size_t)i + 1] && face[e] == f && isUpper[e] == 0
                    && nbr[e] == newIndex[(std::size_t)A.owner[(std::size_t)f]];
            }
            for (label f = A.ownerStart[(std::size_t)c]; f < A.ownerStart[(std::size_t)c + 1] && rows; ++f, ++e)
            {
                rows = e < (std::size_t)rowStart[(std::size_t)i + 1] && face[e] == f && isUpper[e] == 1
                    && nbr[e] == newIndex[(std::size_t)A.nei[(std::size_t)f]];
            }
            if (rows) rows = e == (std::size_t)rowStart[(std::size_t)i + 1];
        }
        check(rows, "(i) every row's entries are the device's losort faces then its owned faces, in that order, with the right side of each");

        std::mt19937 rng(20260907u);
        std::uniform_real_distribution<scalar> uni(-1.0, 1.0);
        std::vector<scalar> xh((std::size_t)A.nC);
        for (scalar& v : xh)
            v = uni(rng);
        DeviceBuffer<scalar> D, x, Ax, AxLayout;
        D.copyFrom(std3[0].diag);
        x.copyFrom(xh);
        const DeviceLduView Av = deviceLduView(dm, D, U, L);
        deviceAmul(Av, x, Ax);
        deviceColourLayoutAmul(Av, col, x, AxLayout);
        std::vector<scalar> a, b;
        Ax.copyTo(a);
        AxLayout.copyTo(b);
        // Two summation orders of one row: amulKernel adds the owned faces before the neighboured
        // ones, the layout the neighboured before the owned, so the two agree to the rounding of
        // the row's terms and no tighter. The bound is the residual arms' floor per cell, DBL_EPSILON
        // times the magnitudes the row is a sum of (with the same 4x margin); reported as the worst
        // cell's difference over its floor.
        scalar worst = 0;
        for (label c = 0; c < A.nC && (int)b.size() == A.nC; ++c)
        {
            const scalar floor = 4*DBL_EPSILON*rowTermSum(A, std3[0].diag, xh, c);
            worst = std::fmax(worst, std::fabs(b[(std::size_t)c] - a[(std::size_t)c])/floor);
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "(i) A x through the layout equals deviceAmul on the natural layout to the rows' rounding floor (worst cell %.2f of its floor)", (double)worst);
        check((int)b.size() == A.nC && worst <= 1, buf);
    }

    // (j) THE REFUSALS. One arm per way the engine must throw rather than run, each holding the
    // wording, since a refusal that names the wrong thing sends the reader to the wrong place. A
    // run that went ahead leaves "" and fails the substring check.
    {
        // (j1) nSweeps 0: smoothSolver.C:178-209 would loop forever, never advancing nIterations.
        {
            Params p0 = pa;
            p0.nSweeps = 0;
            const std::string what = refusalOf([&]() { runDevice(dm, U, L, std3, nfStd, col, p0); });
            check(what.find("nSweeps 0") != std::string::npos, "(j1) nSweeps 0 is refused (" + what + ")");
        }
        // (j2) a view carrying a cyclic interface: the sweep applies no interfaces.
        {
            DeviceSystem s;
            fillDeviceSystem(s, dm, U, L, std3, nfStd, false);
            s.A[0].nCyc = 1;
            DeviceSolverPerf perf[NCOMP];
            const std::string what = refusalOf([&]()
            {
                deviceColourGaussSeidelFused(NCOMP, s.comps, col, pa.tol, pa.relTol, pa.maxIter, pa.minIter, pa.nSweeps, pa.symmetric, perf);
            });
            check(what.find("cyclic interface") != std::string::npos, "(j2) a cyclic interface is refused (" + what + ")");
        }
        // (j3) a colouring built for a smaller box handed this box's matrix: refused on the sizes.
        {
            const PrimitiveMesh ms = boxtest::boxMesh(9, 7, 4);
            const std::vector<label> ownerS(ms.owner().begin(), ms.owner().begin() + (std::ptrdiff_t)ms.nInternalFaces());
            const DeviceCellColouring colS = buildDeviceCellColouring(ownerS, ms.neighbour(), (int)ms.nCells());
            const std::string what = refusalOf([&]() { runDevice(dm, U, L, std3, nfStd, colS, pa); });
            check(what.find("cells, matrix has") != std::string::npos, "(j3) a colouring of another size is refused (" + what + ")");
        }
        // (j4) comps[1] with an upper of its own (equal values, another buffer): the fused sweep
        // reads one matrix topology, so it must refuse rather than sweep component 1 with 0's.
        {
            DeviceSystem s;
            fillDeviceSystem(s, dm, U, L, std3, nfStd, false);
            DeviceBuffer<scalar> U1;
            U1.copyFrom(A.upper);
            s.A[1] = deviceLduView(dm, s.D[1], U1, L);
            DeviceSolverPerf perf[NCOMP];
            const std::string what = refusalOf([&]()
            {
                deviceColourGaussSeidelFused(NCOMP, s.comps, col, pa.tol, pa.relTol, pa.maxIter, pa.minIter, pa.nSweeps, pa.symmetric, perf);
            });
            check(what.find("does not share") != std::string::npos, "(j4) a component with its own upper is refused (" + what + ")");
        }
        // (j5) nComp 4: the scratch is sized for three.
        {
            DeviceSystem s;
            fillDeviceSystem(s, dm, U, L, std3, nfStd, false);
            GSFusedComponent four[4] = {s.comps[0], s.comps[1], s.comps[2], s.comps[2]};
            DeviceSolverPerf perf[4];
            const std::string what = refusalOf([&]()
            {
                deviceColourGaussSeidelFused(4, four, col, pa.tol, pa.relTol, pa.maxIter, pa.minIter, pa.nSweeps, pa.symmetric, perf);
            });
            check(what.find("nComp 4") != std::string::npos, "(j5) nComp 4 is refused (" + what + ")");
        }
        // (j6) a second mesh with the SAME cell and internal-face counts but other faces: the 5x7x9
        // box against the 9x7x5 one. The size check cannot see it; the colouring must refuse the
        // matrix on its owner/neighbour addressing, which it recorded on its first sweep (arm (a)).
        {
            const PrimitiveMesh m2 = boxtest::boxMesh(5, 7, 9);
            FvGeometry g2;
            g2.build(m2);
            const std::vector<FvPatch> fvp2 = buildPatches(m2, g2);
            const DeviceMesh dm2 = buildDeviceMesh(m2, g2, fvp2);
            check(dm2.nCells == A.nC && dm2.nInternalFaces == A.nIf, "(j6) the second mesh has the same cell and internal-face counts, so only its addressing tells it apart");
            DeviceSystem s;
            fillDeviceSystem(s, dm2, U, L, std3, nfStd, false);
            DeviceSolverPerf perf[NCOMP];
            const std::string what = refusalOf([&]()
            {
                deviceColourGaussSeidelFused(NCOMP, s.comps, col, pa.tol, pa.relTol, pa.maxIter, pa.minIter, pa.nSweeps, pa.symmetric, perf);
            });
            check(what.find("owner/neighbour addressing") != std::string::npos, "(j6) another mesh of the same size is refused on its addressing (" + what + ")");
            // And the colouring still sweeps its own mesh afterwards: the refusal recorded nothing.
            const std::string again = refusalOf([&]() { runDevice(dm, U, L, std3, nfStd, col, pa); });
            check(again.empty(), "(j6) the colouring's own mesh still runs after the refusal (" + again + ")");
        }
    }

    // (k) THE RESIDUAL FROM THE SWEEP (see the top of the file). These arms read the residual rows,
    // which the production path (every arm above) never stores: the colouring is asked to write
    // them from here on, and (l2) holds that the request changes nothing else.
    {
        col.writeResidualVector = true;
        check(deviceColourGaussSeidelFusesResidual(col), "(k) the 2-colour box takes the fused residual path");
        // Colour 1 of the greedy colouring cut in two halves by ascending original index. Each class
        // is laid out ascending and the classes in order, so cells[] is the 2-colour one and the
        // rows over it are byte for byte the same; only the colour ranges differ. Cells of one class
        // share no face, so updating the halves one after the other is the update of the class at
        // once, and this is the same Gauss-Seidel on the same layout with three classes -- the
        // explicit residual path on the fused path's own problem.
        std::vector<label> start3;
        start3.push_back(0);
        start3.push_back(hcol.start[1]);
        start3.push_back((hcol.start[1] + hcol.start[2])/2);
        start3.push_back(hcol.start[2]);
        const DeviceCellColouring col3 = buildDeviceCellColouringFromClasses(A.owner, A.nei, A.nC, hcol.cells, start3);
        col3.writeResidualVector = true;
        const HostColouring hcol3 = hostColouring(col3);
        {
            std::vector<label> c2, c3;
            col.cells.copyTo(c2);
            col3.cells.copyTo(c3);
            check(col3.nColours == 3 && !deviceColourGaussSeidelFusesResidual(col3), "(k3) the 3-class split takes the explicit residual path");
            check(c2 == c3 && start3[2] > start3[1] && start3[3] > start3[2],
                  "(k3) the split keeps the 2-colour permutation: the same cells[], colour 1 cut into two nonempty halves");
        }
        // A fixed-count solve of n sweeps on col: no residual, no speculation, the same launch
        // sequence from the same skip state. The oracle for what n counted sweeps leave in psi.
        auto fixedCount = [&](
            const System* sys,
            const scalar* nfv,
            int n,
            bool symmetric)
        {
            Params q;
            q.nSweeps = -n;
            q.symmetric = symmetric;
            return runDevice(dm, U, L, sys, nfv, col, q);
        };
        // A stopped component's psi against that oracle, and the fail-proof: one more sweep moves
        // the field by more than the bound the comparison is made at, so a speculated launch left
        // in place (half of that sweep, and the half whose change the other half's rides on) could
        // not hide.
        auto holdRollback = [&](
            const char* arm,
            const DeviceRun& run,
            const System* sys,
            const scalar* nfv,
            int k,
            bool symmetric)
        {
            const int n = run.perf[k].nIterations;
            const DeviceRun fixed = fixedCount(sys, nfv, n, symmetric);
            const DeviceRun more = fixedCount(sys, nfv, n + 1, symmetric);
            const scalar moved = fieldRelDiff(more.psi[k], fixed.psi[k]);
            char buf[240];
            std::snprintf(buf, sizeof buf, "%s component %d stopped at %d sweeps with the psi of %d fixed sweeps to the bit (fail-proof: sweep %d moves it %.1e)",
                          arm, k, n, n, n + 1, (double)moved);
            check(n > 0 && bitsEqual(run.psi[k], fixed.psi[k]) && moved > REL, buf);
        };
        std::vector<scalar> rFused[NCOMP];

        // (k1) The fused launches' r against the explicit pass over the SAME state. Arm (a)'s solve
        // runs again so col.rP holds its last pass's vectors (colour 1's rows from the RES_NEW
        // launch, colour 0's from the speculative one, the stopped components' rows rolled back in
        // psi but not in r); then a solve seeded with its final psi under a tol nothing fails sweeps
        // nothing and leaves col.rP as its initial pass -- the explicit one on every colouring --
        // wrote it over that same state.
        {
            const DeviceRun da = runDevice(dm, U, L, std3, nfStd, col, pa);
            for (int k = 0; k < NCOMP; ++k)
                col.rP[k].copyTo(rFused[k]);
            System seeded[NCOMP];
            for (int k = 0; k < NCOMP; ++k)
            {
                seeded[k] = std3[k];
                seeded[k].psi0 = da.psi[k];
            }
            Params pk = pa;
            pk.tol = 1e300;
            const DeviceRun de = runDevice(dm, U, L, seeded, nfStd, col, pk);
            check(de.perf[0].nIterations == 0 && de.perf[1].nIterations == 0 && de.perf[2].nIterations == 0,
                  "(k1) the seeded solve swept nothing: col.rP is its initial, explicit pass over arm (a)'s final state");
            bool same = true;
            bool sumsSame = true;
            scalar worst = 0;
            for (int k = 0; k < NCOMP; ++k)
            {
                std::vector<scalar> rExplicit;
                col.rP[k].copyTo(rExplicit);
                same = same && (int)rExplicit.size() == A.nC && bitsEqual(rExplicit, rFused[k]);
                for (std::size_t i = 0; i < rExplicit.size() && i < rFused[k].size(); ++i)
                    worst = std::fmax(worst, std::fabs(rExplicit[i] - rFused[k][i]));
                sumsSame = sumsSame && de.perf[k].initialResidual == da.perf[k].finalResidual;
            }
            char buf[240];
            std::snprintf(buf, sizeof buf, "(k1) the fused launches' r is the explicit pass's r bit for bit, every component (max |diff| %.1e)", (double)worst);
            check(same, buf);
            check(sumsSame, "(k1) and so are the reduced sums: the seeded solve's initial residual is arm (a)'s final residual to the bit");
            // CONTROL: memcmp can see a summation order. The residual in OpenFOAM's own order
            // (diag*psi first, then the faces; lduMatrixATmul.C:315-327, as rowProduct sums it)
            // over the same state differs from the device's in the last bits of some rows; were it
            // not so, an identity check between two orders would be vacuous.
            int rowsDiffer = 0;
            scalar worstOrder = 0;
            for (int k = 0; k < NCOMP; ++k)
            {
                for (label i = 0; i < A.nC; ++i)
                {
                    const label c = hcol.cells[(std::size_t)i];
                    const scalar rOF = std3[k].b[(std::size_t)c] - rowProduct(A, std3[k].diag, da.psi[k], c);
                    const scalar d = std::fabs(rOF - rFused[k][(std::size_t)i]);
                    if (d != 0) ++rowsDiffer;
                    worstOrder = std::fmax(worstOrder, d);
                }
            }
            std::snprintf(buf, sizeof buf, "(k1) control: OpenFOAM's summation order gives other bits in %d of %d rows (max |diff| %.1e), so the identity is not vacuous",
                          rowsDiffer, NCOMP*A.nC, (double)worstOrder);
            check(rowsDiffer > 0, buf);
        }

        // (k2a) The three components solved to tol 1e-4: with their diagonals scaled 1, 2, 4 they
        // stop at three different counts (arm (a)'s relTol 0.1 stops them at 2/1/1, which rolls two
        // back together), so here two are rolled back while another keeps sweeping and the last is
        // rolled back alone; also the arm against the host reference.
        {
            Params pk;
            pk.tol = 1e-4;
            const ArmResult rk = compareArm("(k2a) GaussSeidel tol 1e-4, three components stopping at three counts", A, hcol, dm, U, L, std3, nfStd, col, pk);
            const int n0 = rk.dev.perf[0].nIterations;
            const int n1 = rk.dev.perf[1].nIterations;
            const int n2 = rk.dev.perf[2].nIterations;
            char buf[200];
            std::snprintf(buf, sizeof buf, "(k2a) the counts %d/%d/%d are pairwise different: two rollbacks happen while another component sweeps on", n0, n1, n2);
            check(n0 != n1 && n1 != n2 && n0 != n2, buf);
            for (int k = 0; k < NCOMP; ++k)
                holdRollback("(k2a)", rk.dev, std3, nfStd, k, false);
        }
        // (k2b) arm (f): component 2 never swept (its psi is its initial guess to the bit, checked
        // there); components 0 and 1 solved to tol and were each rolled back at their own count.
        {
            for (int k = 0; k < 2; ++k)
                holdRollback("(k2b)", rf.dev, mix3, nfMix, k, false);
        }

        // (k3) The explicit path on the split against arm (a): counts, psi and the last pass's
        // residual vectors to the bit (same layout, same per-row arithmetic); the reported
        // residuals to the rounding of their sum, since the split's block partition is not the
        // 2-colour one (arm (l3) has the reason and prints the difference).
        {
            const ArmResult r3 = compareArm("(k3) GaussSeidel relTol 0.1 on the 3-class split", A, hcol3, dm, U, L, std3, nfStd, col3, pa);
            bool same = true;
            bool rSame = true;
            bool sums = true;
            for (int k = 0; k < NCOMP; ++k)
            {
                same = same && r3.dev.perf[k].nIterations == ra.dev.perf[k].nIterations
                    && bitsEqual(r3.dev.psi[k], ra.dev.psi[k]);
                std::vector<scalar> r;
                col3.rP[k].copyTo(r);
                rSame = rSame && bitsEqual(r, rFused[k]);
                sums = sums && sumsAgree(r3.dev.perf[k].initialResidual, ra.dev.perf[k].initialResidual, ra.dev.perf[k].initialResidual)
                    && sumsAgree(r3.dev.perf[k].finalResidual, ra.dev.perf[k].finalResidual, hostSumMag(rFused[k])/nfStd[k]);
            }
            check(same, "(k3) the explicit path on the split gives arm (a)'s counts and psi to the bit");
            check(rSame, "(k3) and its last pass's residual vectors are the fused launches' bit for bit");
            check(sums, "(k3) and its reported residuals are arm (a)'s to the rounding of the sum");
        }

        // (k4) The symmetric schedule: arm (b)'s components against the fixed-count oracle, and
        // the split (whose descending pass relaunches the first half of colour 1 on unchanged
        // operands, a no-op to the bit) against arm (b).
        {
            for (int k = 0; k < NCOMP; ++k)
                holdRollback("(k4)", rb.dev, std3, nfStd, k, true);
            const ArmResult r3 = compareArm("(k4) symGaussSeidel relTol 0.1 on the 3-class split", A, hcol3, dm, U, L, std3, nfStd, col3, pb);
            bool same = true;
            bool sums = true;
            for (int k = 0; k < NCOMP; ++k)
            {
                same = same && r3.dev.perf[k].nIterations == rb.dev.perf[k].nIterations
                    && bitsEqual(r3.dev.psi[k], rb.dev.psi[k]);
                // The split's last-pass rows give the magnitude of the final sum the bound is set
                // from (as in (k3); arm (b) itself ran on the production path and stored none).
                std::vector<scalar> r;
                col3.rP[k].copyTo(r);
                sums = sums && sumsAgree(r3.dev.perf[k].initialResidual, rb.dev.perf[k].initialResidual, rb.dev.perf[k].initialResidual)
                    && sumsAgree(r3.dev.perf[k].finalResidual, rb.dev.perf[k].finalResidual, hostSumMag(r)/nfStd[k]);
            }
            check(same, "(k4) the explicit path on the split gives arm (b)'s counts and psi to the bit");
            check(sums, "(k4) and its reported residuals are arm (b)'s to the rounding of the sum");
        }

        // (l) THE REDUCTION IN THE LAUNCH (see the top of the file).
        //
        // (l1) The reported residuals against the host's exact sum of the stored rows: the fused
        // launches' last pass (arm (a)'s parameters; col.rP holds each component's rows of its own
        // last pass, the pass that produced its final residual) and the explicit kernel's initial
        // pass (the seeded no-sweep solve of (k1), whose initial residual is that pass's sum). And
        // the determinism control: the same solve twice, the same bits.
        {
            const DeviceRun d1 = runDevice(dm, U, L, std3, nfStd, col, pa);
            const DeviceRun d2 = runDevice(dm, U, L, std3, nfStd, col, pa);
            bool det = true;
            bool within = true;
            std::string detail;
            System seeded[NCOMP];
            for (int k = 0; k < NCOMP; ++k)
            {
                det = det && d1.perf[k].initialResidual == d2.perf[k].initialResidual
                    && d1.perf[k].finalResidual == d2.perf[k].finalResidual
                    && d1.perf[k].nIterations == d2.perf[k].nIterations;
                std::vector<scalar> r;
                col.rP[k].copyTo(r);
                const scalar sumMag = hostSumMag(r);
                const scalar bound = 4*DBL_EPSILON*sumMag/nfStd[k];
                const scalar d = std::fabs(d2.perf[k].finalResidual - sumMag/nfStd[k]);
                within = within && (int)r.size() == A.nC && d <= bound;
                char buf[96];
                std::snprintf(buf, sizeof buf, " [k%d fused: %.1e, %.2f of the bound]", k, (double)d, (double)(d/bound));
                detail += buf;
                seeded[k] = std3[k];
                seeded[k].psi0 = d2.psi[k];
            }
            Params pk = pa;
            pk.tol = 1e300;
            const DeviceRun de = runDevice(dm, U, L, seeded, nfStd, col, pk);
            bool explicitWithin = de.perf[0].nIterations == 0 && de.perf[1].nIterations == 0 && de.perf[2].nIterations == 0;
            for (int k = 0; k < NCOMP; ++k)
            {
                std::vector<scalar> r;
                col.rP[k].copyTo(r);
                const scalar sumMag = hostSumMag(r);
                const scalar bound = 4*DBL_EPSILON*sumMag/nfStd[k];
                const scalar d = std::fabs(de.perf[k].initialResidual - sumMag/nfStd[k]);
                explicitWithin = explicitWithin && (int)r.size() == A.nC && d <= bound;
                char buf[96];
                std::snprintf(buf, sizeof buf, " [k%d explicit: %.1e, %.2f of the bound]", k, (double)d, (double)(d/bound));
                detail += buf;
            }
            check(within, "(l1) the fused launches' reported residual is the host's exact sum of the stored |r| rows over nf, to 4*DBL_EPSILON of the sum" + detail);
            check(explicitWithin, "(l1) and so is the explicit kernel's, on the seeded no-sweep solve's initial pass");
            check(det, "(l1) determinism control: two identical solves report the same residuals and counts bit for bit");
        }

        // (l2) The vector write changes no arithmetic. rP is filled with a sentinel, the solve runs
        // with the write off (the production path), and the sentinel must survive it; the solve
        // with the write on must then report the same counts and residuals and leave the same psi,
        // bit for bit. The residual is reduced from the same register either way; the flag only
        // adds the store.
        {
            const std::vector<scalar> sentinel((std::size_t)A.nC, scalar(-12345.0));
            col.writeResidualVector = false;
            for (int k = 0; k < NCOMP; ++k)
                col.rP[k].copyFrom(sentinel);
            const DeviceRun off = runDevice(dm, U, L, std3, nfStd, col, pa);
            bool untouched = true;
            for (int k = 0; k < NCOMP; ++k)
            {
                std::vector<scalar> r;
                col.rP[k].copyTo(r);
                untouched = untouched && r == sentinel;
            }
            col.writeResidualVector = true;
            const DeviceRun on = runDevice(dm, U, L, std3, nfStd, col, pa);
            bool same = true;
            bool stored = true;
            for (int k = 0; k < NCOMP; ++k)
            {
                same = same && off.perf[k].initialResidual == on.perf[k].initialResidual
                    && off.perf[k].finalResidual == on.perf[k].finalResidual
                    && off.perf[k].nIterations == on.perf[k].nIterations
                    && bitsEqual(off.psi[k], on.psi[k]);
                std::vector<scalar> r;
                col.rP[k].copyTo(r);
                stored = stored && r != sentinel;
            }
            check(untouched, "(l2) with the vector write off the solve stored no residual row: the sentinel in rP survived");
            check(stored, "(l2) with it on the rows were stored (the sentinel was overwritten), so the off solve's silence is not the buffer's");
            check(same, "(l2) the two solves report the same counts and residuals and leave the same psi, bit for bit");
        }

        // (l3) The explicit kernel's sums on the 3-class split against the fused launches' on the
        // 2-colour layout, same problem (arm (a)'s): the same per-row numbers -- (k3) holds the
        // residual vectors together bit for bit -- through two fixed-order sums whose block
        // partitions differ (the colouring's nBlocks says how: colour 1's 157 rows are one block on
        // the 2-colour layout and two on the split), so the last bits may differ; held to the
        // rounding of the sum, with the difference printed so a reader can see whether they did.
        {
            check(!deviceColourGaussSeidelFusesResidual(col3), "(l3) the 3-class split takes the explicit kernel");
            const DeviceRun d2 = runDevice(dm, U, L, std3, nfStd, col, pa);
            std::vector<scalar> r2[NCOMP];
            for (int k = 0; k < NCOMP; ++k)
                col.rP[k].copyTo(r2[k]);
            const DeviceRun d3 = runDevice(dm, U, L, std3, nfStd, col3, pa);
            bool rows = true;
            bool sums = true;
            int bitIdentical = 0;
            std::string detail;
            for (int k = 0; k < NCOMP; ++k)
            {
                std::vector<scalar> r3;
                col3.rP[k].copyTo(r3);
                rows = rows && bitsEqual(r3, r2[k]);
                // The final pass's rows are stored; the initial pass's were overwritten, so its
                // magnitude is the reported value (see sumsAgree).
                const scalar boundInit = 4*DBL_EPSILON*d2.perf[k].initialResidual;
                const scalar boundFin = 4*DBL_EPSILON*hostSumMag(r2[k])/nfStd[k];
                const scalar dInit = std::fabs(d3.perf[k].initialResidual - d2.perf[k].initialResidual);
                const scalar dFin = std::fabs(d3.perf[k].finalResidual - d2.perf[k].finalResidual);
                sums = sums && d3.perf[k].nIterations == d2.perf[k].nIterations && dInit <= boundInit && dFin <= boundFin;
                if (dInit == 0 && dFin == 0) ++bitIdentical;
                char buf[128];
                std::snprintf(buf, sizeof buf, " [k%d: init %.1e (%.2f of the bound) final %.1e (%.2f)]", k, (double)dInit, (double)(dInit/boundInit), (double)dFin, (double)(dFin/boundFin));
                detail += buf;
            }
            char buf[200];
            std::snprintf(buf, sizeof buf, "(l3) the split's sums are the 2-colour sums to the rounding of the sum (%d of %d components bit-identical; %d blocks against %d)",
                          bitIdentical, NCOMP, col3.nBlocks, col.nBlocks);
            check(rows, "(l3) the split's residual rows are the fused launches' bit for bit: the two sums are of the same numbers");
            check(sums, buf + detail);
            check(col3.nBlocks != col.nBlocks, "(l3) and the two block partitions differ, which is why the sums are held to the rounding and not to the bit");
        }
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
