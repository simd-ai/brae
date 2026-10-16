// Gate: the AMG coarsest grid's precomputed inverse (BRAE_AMG_COARSE_SOLVE=inverse) against the LU it stands in for.
//
//   test_amg_coarse_inverse <pitzDaily case>          (a child run: test_amg_coarse_inverse child <case>)
//
// ORACLE    a long double LU with the device's partial pivoting and the same pivot floor (DENSE_LU_EPS); and the
//           LU kind itself, which the inverse must reproduce
// PART A    coarse levels of 16, 32, 64, 96 and 128 cells: singular (all-Neumann), near-singular (one cell shifted by
//           1e-11), one cell held, anisotropic (1, 1e-3, 1e-6), stretched (six decades), negative definite (OpenFOAM's
//           sign), condition 1e12 and 1e16, convection-diffusion at Peclet 10 and 1000 (not symmetric). Each level:
//             where the check sends its solves to the LU (a floored pivot, an entry not finite, the probes): every bit
//             the LU's -- the null-space behaviour kept; the singular levels must go that way
//             where the inverse stands: |x - x_lu| / |x_lu|, the normwise backward error, the forward error against
//             the oracle (no worse than the LU's own, by a factor), the backward error on a consistent b = A w (red
//             on 13 levels before the check's consistent probe: cond 1e12 and 1e16, near-singular; 1e-7 to 3.9e-3)
//             and, for a symmetric level, |u.M v - v.M u|
//           and a stale-factorisation check: a second level through the same state takes its own inverse
// PART B    the whole preconditioner on pitzDaily's Laplacian: one V-cycle M (amgVCycleApply) with each coarsest
//           solve -- its symmetry, and its definiteness with A's own sign on random vectors; the AMG-PCG's iterations
//           and answer, inverse against LU; and the same closed (all-zeroGradient, singular) -- a floored coarsest
//           grid, whose solves must take the LU and give the LU's iterations and bits
// CONTROLS  (BRAE_CONTROL_COARSE_INVERSE, each a child run that must SAY it was injected) perturb (one entry moved
//           after the check): red; transpose: red on the asymmetric levels; stale: red; nofallback: red on the
//           singular levels; half: red; probe_blind (one entry moved BEFORE the check): the check must send every
//           level the inverse took to the LU, and nothing goes red
// REFUSALS  (a child each) BRAE_AMG_COARSE_SOLVE=invrse with no direct level, BRAE_AMG_DENSE_MAX=512 and =12x stop
//           by name; BRAE_AMG_DENSE_MAX=96 runs and says so
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"
#include "device_amg_coarse.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace brae;

namespace
{

// BOUNDS, each a decade over the worst measured on the GB10 (2026-10-15; the numbers in the comments)
constexpr double VS_LU_BOUND = 3e-13;          // |x - x_lu| / |x_lu| where the inverse stands (worst 2.82e-14)
constexpr double BACKWARD_BOUND = 3e-14;       // normwise backward error (worst 2.21e-15; the LU's own 5.4e-16)
constexpr double FORWARD_FACTOR = 10.0;        // the inverse's forward error against the LU's own, plus FORWARD_FLOOR
constexpr double FORWARD_FLOOR = 1e-13;
constexpr double SYMMETRY_FACTOR = 10.0;       // the operator's |u.Mv - v.Mu| against the LU's, plus SYMMETRY_FLOOR
constexpr double SYMMETRY_FLOOR = 1e-15;

int fails = 0;
void fail(const std::string& what)
{
    std::printf("FAIL %s\n", what.c_str());
    ++fails;
}

struct Dense
{
    std::string name;
    int n = 0;
    bool symmetric = false;
    bool singular = false;
    std::vector<double> a;
};

void dims(int n, int& nx, int& ny, int& nz)
{
    if (n == 16) { nx = 4; ny = 2; nz = 2; }
    else if (n == 32) { nx = 4; ny = 4; nz = 2; }
    else if (n == 64) { nx = 4; ny = 4; nz = 4; }
    else if (n == 96) { nx = 6; ny = 4; nz = 4; }
    else { nx = 8; ny = 4; nz = 4; }
}

Dense grid(const std::string& family, int n)
{
    int nx, ny, nz;
    dims(n, nx, ny, nz);
    Dense d;
    d.name = family;
    d.n = n;
    d.a.assign(static_cast<std::size_t>(n)*n, 0.0);
    auto id = [&](int i, int j, int k) { return i + nx*(j + ny*k); };
    auto add = [&](int p, int q, double c)
    {
        d.a[p*n + p] += c;
        d.a[q*n + q] += c;
        d.a[p*n + q] -= c;
        d.a[q*n + p] -= c;
    };
    auto hz = [&](int k) { return std::pow(10.0, static_cast<double>(k)); };
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const int p = id(i, j, k);
                double kx = 1.0, ky = 1.0, kz = 1.0;
                if (family == "aniso") { ky = 1e-3; kz = 1e-6; }
                if (family == "stretched") kz = 2.0/(hz(k)*(hz(k) + (k + 1 < nz ? hz(k + 1) : hz(k))));
                if (i + 1 < nx) add(p, id(i + 1, j, k), kx);
                if (j + 1 < ny) add(p, id(i, j + 1, k), ky);
                if (k + 1 < nz) add(p, id(i, j, k + 1), kz);
            }
    const bool asym = family == "asym10" || family == "asym1000";
    if (asym)
    {
        const double u = family == "asym10" ? 10.0 : 1000.0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    const int p = id(i, j, k);
                    d.a[p*n + p] += u;
                    if (i > 0) d.a[p*n + id(i - 1, j, k)] -= u;
                }
    }
    if (family == "neumann") d.singular = true;
    else if (family == "nearsing") d.a[0] += 1e-11*d.a[0];
    else if (family == "pinned" || family == "negdef") d.a[0] += 2.0;
    else for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) d.a[id(0, j, k)*n + id(0, j, k)] += 2.0;
    if (family == "negdef") for (double& v : d.a) v = -v;
    d.symmetric = !asym;
    return d;
}

Dense illConditioned(int n, double cond, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> q(static_cast<std::size_t>(n)*n);
    for (auto& v : q) v = g(rng);
    for (int j = 0; j < n; ++j)
    {
        for (int p = 0; p < j; ++p)
        {
            double dot = 0.0;
            for (int i = 0; i < n; ++i) dot += q[i*n + j]*q[i*n + p];
            for (int i = 0; i < n; ++i) q[i*n + j] -= dot*q[i*n + p];
        }
        double norm = 0.0;
        for (int i = 0; i < n; ++i) norm += q[i*n + j]*q[i*n + j];
        norm = std::sqrt(norm);
        for (int i = 0; i < n; ++i) q[i*n + j] /= norm;
    }
    Dense d;
    d.name = cond > 1e14 ? "cond1e16" : "cond1e12";
    d.n = n;
    d.symmetric = true;
    d.a.assign(static_cast<std::size_t>(n)*n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
        {
            double s = 0.0;
            for (int p = 0; p < n; ++p) s += q[i*n + p]*std::pow(cond, -static_cast<double>(p)/(n - 1))*q[j*n + p];
            d.a[i*n + j] = s;
            d.a[j*n + i] = s;
        }
    return d;
}

DeviceLduMatrix toLdu(const Dense& d)
{
    const int n = d.n;
    std::vector<scalar> diag(n), upper, lower;
    std::vector<label> owner, nei;
    for (int i = 0; i < n; ++i)
    {
        diag[i] = d.a[i*n + i];
        for (int j = i + 1; j < n; ++j)
            if (d.a[i*n + j] != 0.0 || d.a[j*n + i] != 0.0)
            {
                owner.push_back(i);
                nei.push_back(j);
                upper.push_back(d.a[i*n + j]);
                lower.push_back(d.a[j*n + i]);
            }
    }
    return buildDeviceLdu(diag, upper, lower, owner, nei, n);
}

std::vector<long double> referenceSolve(const Dense& d, const std::vector<double>& b)
{
    const int n = d.n;
    std::vector<long double> A(d.a.begin(), d.a.end());
    std::vector<int> piv(n);
    long double scale = 0.0L;
    for (int c = 0; c < n; ++c) scale = std::max(scale, std::fabs(A[c*n + c]));
    if (!(scale > 0.0L)) scale = 1.0L;
    for (int k = 0; k < n; ++k)
    {
        int r = k;
        for (int i = k + 1; i < n; ++i) if (std::fabs(A[i*n + k]) > std::fabs(A[r*n + k])) r = i;
        piv[k] = r;
        if (r != k) for (int j = 0; j < n; ++j) std::swap(A[k*n + j], A[r*n + j]);
        long double p = A[k*n + k];
        const long double floor_ = 1e-14L*scale;
        if (!(std::fabs(p) > floor_)) p = (p < 0.0L) ? -floor_ : floor_;
        A[k*n + k] = p;
        for (int i = k + 1; i < n; ++i)
        {
            A[i*n + k] /= p;
            for (int j = k + 1; j < n; ++j) A[i*n + j] -= A[i*n + k]*A[k*n + j];
        }
    }
    std::vector<long double> y(b.begin(), b.end());
    for (int k = 0; k < n; ++k) if (piv[k] != k) std::swap(y[k], y[piv[k]]);
    for (int k = 0; k < n; ++k) for (int i = k + 1; i < n; ++i) y[i] -= A[i*n + k]*y[k];
    for (int k = n - 1; k >= 0; --k)
    {
        y[k] /= A[k*n + k];
        for (int i = 0; i < k; ++i) y[i] -= A[i*n + k]*y[k];
    }
    return y;
}

double backwardError(const Dense& d, const std::vector<double>& x, const std::vector<double>& b)
{
    const int n = d.n;
    long double r2 = 0.0L, x2 = 0.0L, b2 = 0.0L, a2 = 0.0L;
    for (int i = 0; i < n; ++i)
    {
        long double s = b[i];
        for (int j = 0; j < n; ++j)
        {
            s -= static_cast<long double>(d.a[i*n + j])*x[j];
            a2 += static_cast<long double>(d.a[i*n + j])*d.a[i*n + j];
        }
        r2 += s*s;
        x2 += static_cast<long double>(x[i])*x[i];
        b2 += static_cast<long double>(b[i])*b[i];
    }
    return static_cast<double>(std::sqrt(r2)/(std::sqrt(a2)*std::sqrt(x2) + std::sqrt(b2)));
}

struct Level
{
    DeviceBuffer<scalar> lu;
    DeviceBuffer<int> piv;
    CoarseDirect cd;
};

void factor(const DeviceLduMatrix& m, CoarseSolveKind kind, Level& L)
{
    const int n = m.nCells;
    L.lu.resize(static_cast<std::size_t>(n)*n);
    L.piv.resize(n);
    deviceCoarseLUFactor(m.view(), L.lu, L.piv);
    deviceCoarseDirectPrepare(kind, m.view(), nullptr, L.lu, L.piv, L.cd);
}

std::vector<double> solve(int n, Level& L, const std::vector<double>& b)
{
    DeviceBuffer<scalar> rb, rx;
    rb.copyFrom(b);
    rx.resize(n);
    deviceCoarseDirectSolve(n, L.lu, L.piv, L.cd, rb, rx);
    cudaCheck(cudaDeviceSynchronize(), "solve");
    return rx.host();
}

int unsafeOf(const Level& L)
{
    if (L.cd.kind != CoarseSolveKind::Inverse) return -1;
    int u = 0;
    cudaCheck(cudaMemcpy(&u, L.cd.unsafe.data(), sizeof(int), cudaMemcpyDeviceToHost), "unsafe");
    return u;
}

double relDiff(const std::vector<double>& x, const std::vector<double>& y)
{
    double d = 0.0, s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        d = std::max(d, std::fabs(x[i] - y[i]));
        s = std::max(s, std::fabs(y[i]));
    }
    return s > 0.0 ? d/s : d;
}

double symmetryOf(int n, Level& L, const std::vector<double>& u, const std::vector<double>& v)
{
    const std::vector<double> mv = solve(n, L, v);
    const std::vector<double> mu = solve(n, L, u);
    long double uMv = 0.0L, vMu = 0.0L, un = 0.0L, mvn = 0.0L;
    for (int i = 0; i < n; ++i)
    {
        uMv += static_cast<long double>(u[i])*mv[i];
        vMu += static_cast<long double>(v[i])*mu[i];
        un += static_cast<long double>(u[i])*u[i];
        mvn += static_cast<long double>(mv[i])*mv[i];
    }
    return static_cast<double>(std::fabs(uMv - vMu)/(std::sqrt(un)*std::sqrt(mvn)));
}

// PART A; returns the number of levels whose solves the check sent to the LU for the probes (reason 3)
int partA()
{
    std::vector<Dense> mats;
    for (int n : {16, 32, 64, 96, 128})
    {
        for (const char* f : {"neumann", "nearsing", "pinned", "aniso", "stretched", "negdef", "asym10", "asym1000"}) mats.push_back(grid(f, n));
        mats.push_back(illConditioned(n, 1e12, 1000u + n));
        mats.push_back(illConditioned(n, 1e16, 2000u + n));
    }
    double worstVs = 0.0, worstBwd = 0.0, worstBwdC = 0.0, worstFwdRatio = 0.0, worstSym = 0.0;
    int byProbe = 0, taken = 0, toLu = 0;
    for (const Dense& d : mats)
    {
        const int n = d.n;
        const DeviceLduMatrix m = toLdu(d);
        std::mt19937_64 rng(77u + n);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        std::vector<double> b(n), u(n), v(n);
        for (auto& x : b) x = uni(rng);
        for (auto& x : u) x = uni(rng);
        for (auto& x : v) x = uni(rng);
        Level lu, inv;
        factor(m, CoarseSolveKind::LU, lu);
        factor(m, CoarseSolveKind::Inverse, inv);
        const std::vector<double> xlu = solve(n, lu, b);
        const std::vector<double> xin = solve(n, inv, b);
        const int unsafe = unsafeOf(inv);
        const std::string tag = d.name + " n=" + std::to_string(n);
        bool finite = true;
        for (double x : xin) finite = finite && std::isfinite(x);
        if (!finite) fail(tag + ": the inverse kind's answer is not finite");
        if (d.singular && unsafe != 1) fail(tag + ": a singular level must take the LU for its floored pivot (check said " + std::to_string(unsafe) + ")");
        if (unsafe != 0)
        {
            ++toLu;
            if (unsafe == 3) ++byProbe;
            bool same = true;
            for (int i = 0; i < n; ++i) same = same && (xin[i] == xlu[i] || (std::isnan(xin[i]) && std::isnan(xlu[i])));
            if (!same) fail(tag + ": the check sent the solves to the LU (" + std::to_string(unsafe) + ") and the answer is not the LU's to the bit");
            continue;
        }
        ++taken;
        const std::vector<long double> xref = referenceSolve(d, b);
        long double ni = 0.0L, nl = 0.0L, nr = 0.0L;
        for (int i = 0; i < n; ++i)
        {
            ni += (xin[i] - xref[i])*(xin[i] - xref[i]);
            nl += (xlu[i] - xref[i])*(xlu[i] - xref[i]);
            nr += xref[i]*xref[i];
        }
        const double fwdIn = static_cast<double>(std::sqrt(ni/nr));
        const double fwdLu = static_cast<double>(std::sqrt(nl/nr));
        const double vs = relDiff(xin, xlu);
        const double bwd = backwardError(d, xin, b);
        worstVs = std::max(worstVs, vs);
        worstBwd = std::max(worstBwd, bwd);
        worstFwdRatio = std::max(worstFwdRatio, fwdIn/std::max(fwdLu, 1e-300));
        if (!(vs <= VS_LU_BOUND)) fail(tag + ": |x - x_lu| / |x_lu| " + std::to_string(vs));
        if (!(bwd <= BACKWARD_BOUND)) fail(tag + ": backward error " + std::to_string(bwd));
        if (!(fwdIn <= FORWARD_FACTOR*fwdLu + FORWARD_FLOOR)) fail(tag + ": forward error " + std::to_string(fwdIn) + " against the LU's " + std::to_string(fwdLu));
        // a consistent right-hand side, b = A w: its answer is the size of w, where a random b's answer on a
        // near-singular level is the near-null mode's 1/lambda and hides an error in every other component (what a
        // V-cycle's restricted residual, nearly orthogonal to that mode, meets)
        std::vector<double> w(n), bc(n);
        for (auto& x : w) x = uni(rng);
        for (int i = 0; i < n; ++i)
        {
            long double s = 0.0L;
            for (int j = 0; j < n; ++j) s += static_cast<long double>(d.a[i*n + j])*w[j];
            bc[i] = static_cast<double>(s);
        }
        const double bwdC = backwardError(d, solve(n, inv, bc), bc);
        worstBwdC = std::max(worstBwdC, bwdC);
        if (!(bwdC <= BACKWARD_BOUND)) fail(tag + ": backward error on b = A w " + std::to_string(bwdC) + " (the LU's " + std::to_string(backwardError(d, solve(n, lu, bc), bc)) + ")");
        if (d.symmetric)
        {
            const double si = symmetryOf(n, inv, u, v);
            const double sl = symmetryOf(n, lu, u, v);
            worstSym = std::max(worstSym, si);
            if (!(si <= SYMMETRY_FACTOR*sl + SYMMETRY_FLOOR)) fail(tag + ": operator symmetry " + std::to_string(si) + " against the LU's " + std::to_string(sl));
        }
    }
    // a second level through the same state takes its own inverse
    {
        const Dense d1 = grid("pinned", 64), d2 = grid("aniso", 64);
        const DeviceLduMatrix m1 = toLdu(d1), m2 = toLdu(d2);
        Level lu2, inv;
        factor(m1, CoarseSolveKind::Inverse, inv);
        std::vector<double> b(64);
        for (int i = 0; i < 64; ++i) b[i] = std::cos(0.1*i);
        (void)solve(64, inv, b);
        factor(m2, CoarseSolveKind::Inverse, inv);
        factor(m2, CoarseSolveKind::LU, lu2);
        const double vs = relDiff(solve(64, inv, b), solve(64, lu2, b));
        if (!(vs <= VS_LU_BOUND)) fail("stale: the second level's answer is " + std::to_string(vs) + " from its own LU's");
    }
    std::printf("PART A: %d levels, the inverse taken on %d and the LU on %d (%d by the probes); worst |x - x_lu|/|x_lu| %.2e, backward %.2e, "
                "backward on b = A w %.2e, forward/LU's %.2f, symmetry %.2e\n", static_cast<int>(mats.size()), taken, toLu, byProbe, worstVs, worstBwd,
                worstBwdC, worstFwdRatio, worstSym);
    return byProbe;
}

// PART B: pitzDaily's Laplacian, as test_gpu_amg builds it (closed: every patch zeroGradient)
void partB(const std::string& caseDir, bool closed)
{
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    GeometricField<scalar> f;
    f.internal.assign(nC, 0.0);
    for (const FvPatch& p : fvp)
    {
        if (p.type == "empty") f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        else if (p.type == "wall" || closed) f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        else f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(p, true, 0.0, std::vector<scalar>{}));
    }
    f.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(f, 1.0, m, g, fvp);
    std::vector<scalar> diagC = M.diag, b(nC);
    for (label c = 0; c < nC; ++c) b[c] = g.V()[c]*std::sin(0.01*c + 0.4);
    if (closed)
    {
        // a consistent right-hand side for the singular system: no net source
        double s = 0.0, v = 0.0;
        for (label c = 0; c < nC; ++c) { s += b[c]; v += g.V()[c]; }
        for (label c = 0; c < nC; ++c) b[c] -= g.V()[c]*s/v;
    }
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label c = fvp[pi].faceCells[i];
            diagC[c] += M.internalCoeffs[pi][i];
            b[c] += M.boundaryCoeffs[pi][i];
        }
    scalar normFactor = 1e-20;
    for (label c = 0; c < nC; ++c) normFactor += std::fabs(b[c]);
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.lower, ownerInt, m.neighbour(), nC);
    DeviceBuffer<scalar> db(b);
    const std::vector<scalar> magSfInt(g.magSf().begin(), g.magSf().begin() + nIf);
    const std::string tag = closed ? "pitzDaily closed" : "pitzDaily";

    std::mt19937_64 rng(5u);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<std::vector<double>> w(8, std::vector<double>(nC));
    for (auto& x : w) for (auto& y : x) y = uni(rng);

    struct Arm { int its = 0; double res = 0.0; std::vector<double> x; double sym = 0.0; int unsafe = -1; int nc = 0; bool definite = true; };
    Arm arms[2];
    for (int k = 0; k < 2; ++k)
    {
        const CoarseSolveKind kind = k == 0 ? CoarseSolveKind::LU : CoarseSolveKind::Inverse;
        AMGData amg = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
        amgGalerkin(amg, dM.diag, dM.upper, dM.lower);
        Arm& a = arms[k];
        a.nc = amg.level.back().nCoarse;
        if (amg.coarseLUn != a.nc) fail(tag + ": the coarsest grid (" + std::to_string(a.nc) + " cells) has no direct solve");
        deviceCoarseDirectPrepare(kind, amg.level.back().coarseView(), nullptr, amg.coarseLU, amg.coarsePiv, amg.coarseDirect);
        a.unsafe = -1;
        if (kind == CoarseSolveKind::Inverse) cudaCheck(cudaMemcpy(&a.unsafe, amg.coarseDirect.unsafe.data(), sizeof(int), cudaMemcpyDeviceToHost), "unsafe");
        // the effective preconditioner M (one V-cycle): symmetry, and definiteness with A's sign
        std::vector<std::vector<double>> mw(8);
        for (int i = 0; i < 8; ++i)
        {
            DeviceBuffer<scalar> r(w[i]), z(std::vector<scalar>(nC, 0.0));
            amgVCycleApply(amg, dM.view(), r, z);
            cudaCheck(cudaDeviceSynchronize(), "vcycle");
            mw[i] = z.host();
        }
        long double uMv = 0.0L, vMu = 0.0L, un = 0.0L, mvn = 0.0L;
        for (label c = 0; c < nC; ++c)
        {
            uMv += static_cast<long double>(w[0][c])*mw[1][c];
            vMu += static_cast<long double>(w[1][c])*mw[0][c];
            un += static_cast<long double>(w[0][c])*w[0][c];
            mvn += static_cast<long double>(mw[1][c])*mw[1][c];
        }
        a.sym = static_cast<double>(std::fabs(uMv - vMu)/(std::sqrt(un)*std::sqrt(mvn)));
        // A's sign on the same vectors (the Laplacian here has OpenFOAM's sign: negative definite)
        int signM = 0, signA = 0;
        for (int i = 0; i < 8; ++i)
        {
            long double wMw = 0.0L, wAw = 0.0L;
            const std::vector<scalar> dg = dM.diag.host(), up = dM.upper.host();
            for (label c = 0; c < nC; ++c)
            {
                wMw += static_cast<long double>(w[i][c])*mw[i][c];
                wAw += static_cast<long double>(w[i][c])*dg[c]*w[i][c];
            }
            for (label fi = 0; fi < nIf; ++fi) wAw += 2.0L*w[i][ownerInt[fi]]*up[fi]*w[i][m.neighbour()[fi]];
            const int sm = wMw > 0.0L ? 1 : (wMw < 0.0L ? -1 : 0);
            const int sa = wAw > 0.0L ? 1 : (wAw < 0.0L ? -1 : 0);
            if (i == 0) { signM = sm; signA = sa; }
            if (sm != signM || sa != signA || sm == 0) a.definite = false;
        }
        a.definite = a.definite && signM == signA && signM != 0;
        // closed: the existing LU's floored pivot gives the singular coarsest grid a huge mode of either sign, so neither
        // kind's V-cycle is A's sign there -- held below as the LU's behaviour kept, not asserted
        if (!closed && !a.definite) fail(tag + " " + (k ? "inverse" : "lu") + ": the preconditioner is not definite with A's sign on the random vectors");
        // the AMG-PCG
        DeviceBuffer<scalar> x(std::vector<scalar>(nC, 0.0));
        const DeviceSolverPerf s = deviceAMGPCG(dM.view(), amg, db, x, normFactor, 1e-10, 0.0, 30000);
        a.its = s.nIterations;
        a.res = s.finalResidual;
        a.x = x.host();
        if (!closed && !(a.res < 1e-10)) fail(tag + " " + (k ? "inverse" : "lu") + ": the AMG-PCG did not converge (" + std::to_string(a.res) + ")");
    }
    if (closed)
    {
        if (arms[1].unsafe != 1) fail(tag + ": the singular coarsest grid's solves must take the LU (check said " + std::to_string(arms[1].unsafe) + ")");
        bool same = arms[0].its == arms[1].its;
        for (label c = 0; c < nC && same; ++c) same = arms[0].x[c] == arms[1].x[c];
        if (!same) fail(tag + ": the inverse kind's AMG-PCG is not the LU's to the bit");
    }
    else
    {
        if (arms[1].unsafe != 0) fail(tag + ": the inverse was not taken (check said " + std::to_string(arms[1].unsafe) + ")");
        if (std::abs(arms[0].its - arms[1].its) > 1) fail(tag + ": iterations " + std::to_string(arms[1].its) + " against the LU's " + std::to_string(arms[0].its));
        if (!(relDiff(arms[1].x, arms[0].x) <= 1e-8)) fail(tag + ": the answers differ by " + std::to_string(relDiff(arms[1].x, arms[0].x)));
    }
    if (!(arms[1].sym <= SYMMETRY_FACTOR*arms[0].sym + 1e-14)) fail(tag + ": the V-cycle's symmetry " + std::to_string(arms[1].sym) + " against the LU's " + std::to_string(arms[0].sym));
    std::printf("PART B %s: coarsest %d cells, check %d; AMG-PCG %d / %d iterations (LU / inverse), residual %.2e / %.2e, |x - x_lu|/|x_lu| %.2e; "
                "V-cycle symmetry %.2e / %.2e, definite with A's sign %d / %d\n", tag.c_str(), arms[0].nc, arms[1].unsafe, arms[0].its, arms[1].its,
                arms[0].res, arms[1].res, relDiff(arms[1].x, arms[0].x), arms[0].sym, arms[1].sym, arms[0].definite ? 1 : 0, arms[1].definite ? 1 : 0);
    if (closed && arms[0].res > 1e-10)
    {
        std::printf("note: the closed (singular, no reference cell) system is not solved by the LU kind either (residual %.2e after %d iterations): "
                    "the existing floored pivot's behaviour, which the inverse kind keeps to the bit\n", arms[0].res, arms[0].its);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 2 && std::string(argv[1]) == "child")
    {
        // a control's run: Part A only, then one line for the parent
        const int byProbe = partA();
        std::printf("CHILD failures %d probes %d\n", fails, byProbe);
        return 0;
    }
    if (argc > 2 && std::string(argv[1]) == "refusal")
    {
        // a refusal's run: pitzDaily's hierarchy and its Galerkin product (where the coarsest solve is factorised)
        // under the BRAE_AMG_* the parent set, then one line for the parent
        try
        {
            partB(argv[2], false);
            std::printf("CHILD ran\n");
        }
        catch (const std::exception& e)
        {
            std::printf("CHILD refused: %s\n", e.what());
        }
        return 0;
    }
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    if (std::getenv("BRAE_CONTROL_COARSE_INVERSE")) { std::printf("FAIL run the gate without a control set\n"); return 1; }
    const int byProbe = partA();
    if (byProbe != 0) std::printf("note: %d level(s) went to the LU on the probes\n", byProbe);
    partB(caseDir, false);
    partB(caseDir, true);
    // the controls, a child each
    struct Want { const char* control; bool red; };
    const Want wants[] = {{"perturb", true}, {"transpose", true}, {"stale", true}, {"nofallback", true}, {"half", true}, {"probe_blind", false}};
    for (const Want& c : wants)
    {
        const std::string cmd = std::string("BRAE_CONTROL_COARSE_INVERSE=") + c.control + " " + argv[0] + " child " + caseDir + " 2>&1";
        FILE* p = popen(cmd.c_str(), "r");
        if (!p) { fail(std::string("control ") + c.control + ": could not run the child"); continue; }
        char line[1024];
        bool injected = false;
        int childFails = -1, childProbes = -1;
        while (std::fgets(line, sizeof(line), p))
        {
            if (std::strstr(line, "INJECTED")) injected = true;
            std::sscanf(line, "CHILD failures %d probes %d", &childFails, &childProbes);
        }
        pclose(p);
        if (!injected) { fail(std::string("control ") + c.control + ": the child did not say it was injected"); continue; }
        if (c.red && childFails > 0) std::printf("RED  control %-12s %d failure(s)\n", c.control, childFails);
        else if (!c.red && childFails == 0 && childProbes > 0) std::printf("HELD control %-12s the check sent %d level(s) to the LU, nothing red\n", c.control, childProbes);
        else fail(std::string("control ") + c.control + ": failures " + std::to_string(childFails) + ", probes " + std::to_string(childProbes));
    }
    // the refusals, a child each: a kind that is not one, or a size that is not a whole number from 1 to 128, stops the
    // run by name -- the kind read when the hierarchy is built, whether or not a direct level reaches it (here none:
    // BRAE_AMG_COARSE_LU=0) -- and a size in range runs and is said (the arm that shows a child can run)
    struct Refusal { const char* env; const char* says; };
    const Refusal refusals[] = {{"BRAE_AMG_COARSE_SOLVE=invrse BRAE_AMG_COARSE_LU=0", "BRAE_AMG_COARSE_SOLVE='invrse'"},
                                {"BRAE_AMG_DENSE_MAX=512", "BRAE_AMG_DENSE_MAX='512'"},
                                {"BRAE_AMG_DENSE_MAX=12x", "BRAE_AMG_DENSE_MAX='12x'"},
                                {"BRAE_AMG_DENSE_MAX=96", nullptr}};
    for (const Refusal& r : refusals)
    {
        const std::string cmd = std::string(r.env) + " " + argv[0] + " refusal " + caseDir + " 2>&1";
        FILE* p = popen(cmd.c_str(), "r");
        if (!p) { fail(std::string(r.env) + ": could not run the child"); continue; }
        char line[2048];
        bool refused = false, ran = false, said = false;
        while (std::fgets(line, sizeof(line), p))
        {
            if (std::strstr(line, "CHILD refused:") && r.says && std::strstr(line, r.says)) refused = true;
            if (std::strstr(line, "CHILD ran")) ran = true;
            if (std::strstr(line, "solved directly up to 96 cells (BRAE_AMG_DENSE_MAX")) said = true;
        }
        pclose(p);
        if (r.says && refused && !ran) std::printf("REFUSED %s\n", r.env);
        else if (!r.says && ran && said) std::printf("RAN     %s, and said so\n", r.env);
        else fail(std::string(r.env) + ": refused " + (refused ? "yes" : "no") + ", ran " + (ran ? "yes" : "no") + ", said " + (said ? "yes" : "no"));
    }
    std::printf("%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
