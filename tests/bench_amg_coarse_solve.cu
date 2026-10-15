// The AMG coarsest grid's direct solve, measured kind by kind (BRAE_AMG_COARSE_SOLVE; branch perf/amg-coarse-solve):
// lu (coarseLUSolveKernel, the default), tri (the same factors by one warp), inverse (A^-1 formed once, applied as a
// product), cholesky (L L^T where the level is SPD, the LU otherwise).
//
//   bench_amg_coarse_solve [matrix files written by BRAE_AMG_COARSE_DUMP ...]
//
// For each matrix (synthetic families at 16, 32, 64, 96 and 128 cells, then any dumped CFD coarsest grids) and kind:
//   setup      the LU factorisation alone, and the LU with the kind's preparation, a call (median of 30)
//   apply      a solve, launched back to back (200) and inside one captured CUDA graph (100 solves a graph, 20 graphs)
//   accuracy   the normwise backward error ||b - A x|| / (||A||_F ||x|| + ||b||) and the forward error against a long
//              double LU with the same partial pivoting and the same pivot floor (DENSE_LU_EPS) on the host
//   vs lu      the largest difference from the default kind's x, relative to its largest entry, and whether every bit agrees
//   symmetry   |u.M v - v.M u| / (|u| |M v|) for the operator a kind applies (symmetric families)
// The families: neumann (a 3-D Laplacian, all-Neumann: singular, the floored pivot), pinned (the same with one cell
// held), aniso (coefficients 1, 1e-3, 1e-6 and one held face), stretched (spacings growing geometrically, coefficients
// over six decades), illcond (dense SPD, eigenvalues 1e-12 .. 1), asym (upwind convection-diffusion, not symmetric).
#include "device_amg.cuh"
#include "device_amg_coarse.cuh"
#include "device_ldu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace brae;

namespace
{

struct Dense
{
    std::string name;
    int n = 0;
    bool symmetric = false;
    std::vector<double> a;    // row-major
};

void dims(int n, int& nx, int& ny, int& nz)
{
    if (n == 16) { nx = 4; ny = 2; nz = 2; }
    else if (n == 32) { nx = 4; ny = 4; nz = 2; }
    else if (n == 64) { nx = 4; ny = 4; nz = 4; }
    else if (n == 96) { nx = 6; ny = 4; nz = 4; }
    else { nx = 8; ny = 4; nz = 4; }
}

// a 7-point operator on an nx*ny*nz grid: kx, ky, kz per face (by the face's lower cell), held cells (+held on the
// diagonal), and an upwind velocity in x (asym)
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
    // stretched: spacing in z grows by 10 a cell, so face coefficients ~ 1/h^2 cover six decades
    auto hz = [&](int k) { return std::pow(10.0, static_cast<double>(k)); };
    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
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
        }
    }
    if (family == "asym")
    {
        // first-order upwind of a velocity +u in x: row p gains u on its diagonal and -u from its upwind neighbour
        const double u = 10.0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    const int p = id(i, j, k);
                    d.a[p*n + p] += u;
                    if (i > 0) d.a[p*n + id(i - 1, j, k)] -= u;
                }
    }
    if (family != "neumann")
    {
        // one face held (x = 0): a Dirichlet half-cell coefficient on its cells' diagonals
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                d.a[id(0, j, k)*n + id(0, j, k)] += (family == "pinned") ? (j == 0 && k == 0 ? 2.0 : 0.0) : 2.0;
    }
    d.symmetric = family != "asym";
    return d;
}

// dense SPD with eigenvalues 1e-12 .. 1 on a random orthogonal basis, made exactly symmetric
Dense illConditioned(int n, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> q(static_cast<std::size_t>(n)*n);
    for (auto& v : q) v = g(rng);
    // modified Gram-Schmidt on the columns
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
    d.name = "illcond";
    d.n = n;
    d.symmetric = true;
    d.a.assign(static_cast<std::size_t>(n)*n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
        {
            double s = 0.0;
            for (int p = 0; p < n; ++p)
            {
                const double lambda = std::pow(10.0, -12.0*p/(n - 1));
                s += q[i*n + p]*lambda*q[j*n + p];
            }
            d.a[i*n + j] = s;
            d.a[j*n + i] = s;
        }
    return d;
}

bool readDump(const std::string& path, Dense& d)
{
    std::ifstream in(path);
    int n = 0;
    if (!(in >> n) || n < 1) return false;
    d.n = n;
    d.a.resize(static_cast<std::size_t>(n)*n);
    for (auto& v : d.a) if (!(in >> v)) return false;
    double asym = 0.0, scale = 0.0;
    for (int i = 0; i < n; ++i)
    {
        scale = std::max(scale, std::fabs(d.a[i*n + i]));
        for (int j = i + 1; j < n; ++j) asym = std::max(asym, std::fabs(d.a[i*n + j] - d.a[j*n + i]));
    }
    d.symmetric = asym <= 1e-12*scale;
    const std::size_t slash = path.find_last_of('/');
    d.name = "cfd:" + (slash == std::string::npos ? path : path.substr(slash + 1));
    return true;
}

// the dense matrix as the LDU form the factorisation reads: a face for every pair with an entry either side
DeviceLduMatrix toLdu(const Dense& d)
{
    const int n = d.n;
    std::vector<scalar> diag(n), upper, lower;
    std::vector<label> owner, nei;
    for (int i = 0; i < n; ++i)
    {
        diag[i] = d.a[i*n + i];
        for (int j = i + 1; j < n; ++j)
        {
            if (d.a[i*n + j] != 0.0 || d.a[j*n + i] != 0.0)
            {
                owner.push_back(i);
                nei.push_back(j);
                upper.push_back(d.a[i*n + j]);
                lower.push_back(d.a[j*n + i]);
            }
        }
    }
    return buildDeviceLdu(diag, upper, lower, owner, nei, n);
}

// the host reference: LU with the device's partial pivoting and pivot floor, in long double
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

const char* kindName(CoarseSolveKind k)
{
    switch (k)
    {
    case CoarseSolveKind::LU: return "lu";
    case CoarseSolveKind::Tri: return "tri";
    case CoarseSolveKind::Inverse: return "inverse";
    case CoarseSolveKind::Cholesky: return "cholesky";
    }
    return "?";
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

struct Prepared
{
    DeviceBuffer<scalar> lu;
    DeviceBuffer<int> piv;
    CoarseDirect cd;
};

void prepare(const DeviceLduMatrix& m, CoarseSolveKind kind, Prepared& p)
{
    const int n = m.nCells;
    p.lu.resize(static_cast<std::size_t>(n)*n);
    p.piv.resize(n);
    deviceCoarseLUFactor(m.view(), p.lu, p.piv);
    deviceCoarseDirectPrepare(kind, m.view(), nullptr, p.lu, p.piv, p.cd);
}

std::vector<double> solve(int n, Prepared& p, const std::vector<double>& b)
{
    DeviceBuffer<scalar> rb, rx;
    rb.copyFrom(b);
    rx.resize(n);
    deviceCoarseDirectSolve(n, p.lu, p.piv, p.cd, rb, rx);
    cudaCheck(cudaStreamSynchronize(cudaStreamPerThread), "solve sync");
    std::vector<double> x(n);
    cudaCheck(cudaMemcpy(x.data(), rx.data(), n*sizeof(scalar), cudaMemcpyDeviceToHost), "x down");
    return x;
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<Dense> mats;
    for (int n : {16, 32, 64, 96, 128})
    {
        for (const char* f : {"neumann", "pinned", "aniso", "stretched", "asym"}) mats.push_back(grid(f, n));
        mats.push_back(illConditioned(n, 1234u + static_cast<unsigned>(n)));
    }
    for (int a = 1; a < argc; ++a)
    {
        Dense d;
        if (readDump(argv[a], d) && d.n <= 128) mats.push_back(d);
        else std::fprintf(stderr, "skipped %s (unreadable, or more than 128 cells)\n", argv[a]);
    }
    const CoarseSolveKind kinds[] = {CoarseSolveKind::LU, CoarseSolveKind::Tri, CoarseSolveKind::Inverse, CoarseSolveKind::Cholesky};
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::printf("matrix,n,kind,taken,setup_lu_us,setup_us,apply_launch_us,apply_graph_us,backward_err,forward_err,vs_lu,bitwise_lu,symmetry\n");
    for (const Dense& d : mats)
    {
        const int n = d.n;
        const DeviceLduMatrix m = toLdu(d);
        std::mt19937_64 rng(77u + static_cast<unsigned>(n));
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        std::vector<double> b(n), u(n), v(n);
        for (auto& x : b) x = uni(rng);
        for (auto& x : u) x = uni(rng);
        for (auto& x : v) x = uni(rng);
        const std::vector<long double> xref = referenceSolve(d, b);
        std::vector<double> xlu;
        // the LU's own setup, a call
        double setupLu = 0.0;
        {
            Prepared p;
            prepare(m, CoarseSolveKind::LU, p);
            std::vector<double> t;
            for (int r = 0; r < 30; ++r)
            {
                cudaEventRecord(e0, cudaStreamPerThread);
                deviceCoarseLUFactor(m.view(), p.lu, p.piv);
                cudaEventRecord(e1, cudaStreamPerThread);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                t.push_back(1e3*ms);
            }
            setupLu = median(t);
        }
        for (CoarseSolveKind kind : kinds)
        {
            Prepared p;
            prepare(m, kind, p);
            std::vector<double> ts;
            for (int r = 0; r < 30; ++r)
            {
                cudaEventRecord(e0, cudaStreamPerThread);
                deviceCoarseLUFactor(m.view(), p.lu, p.piv);
                deviceCoarseDirectPrepare(kind, m.view(), nullptr, p.lu, p.piv, p.cd);
                cudaEventRecord(e1, cudaStreamPerThread);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                ts.push_back(1e3*ms);
            }
            const std::vector<double> x = solve(n, p, b);
            if (kind == CoarseSolveKind::LU) xlu = x;
            // which path ran (cholesky falls back on the device)
            std::string taken = kindName(p.cd.kind);
            if (p.cd.kind == CoarseSolveKind::Cholesky)
            {
                int flag = 0;
                cudaCheck(cudaMemcpy(&flag, p.cd.chol.data(), sizeof(int), cudaMemcpyDeviceToHost), "chol flag");
                taken = flag == 1 ? "cholesky" : "cholesky->lu";
            }
            // apply: back to back, and inside one captured graph
            DeviceBuffer<scalar> rb, rx;
            rb.copyFrom(b);
            rx.resize(n);
            for (int w = 0; w < 20; ++w) deviceCoarseDirectSolve(n, p.lu, p.piv, p.cd, rb, rx);
            cudaEventRecord(e0, cudaStreamPerThread);
            for (int r = 0; r < 200; ++r) deviceCoarseDirectSolve(n, p.lu, p.piv, p.cd, rb, rx);
            cudaEventRecord(e1, cudaStreamPerThread);
            cudaEventSynchronize(e1);
            float msLaunch = 0.0f;
            cudaEventElapsedTime(&msLaunch, e0, e1);
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "capture");
            for (int r = 0; r < 100; ++r) deviceCoarseDirectSolve(n, p.lu, p.piv, p.cd, rb, rx);
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &graph), "end capture");
            cudaCheck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
            cudaGraphLaunch(exec, cudaStreamPerThread);
            cudaStreamSynchronize(cudaStreamPerThread);
            cudaEventRecord(e0, cudaStreamPerThread);
            for (int r = 0; r < 20; ++r) cudaGraphLaunch(exec, cudaStreamPerThread);
            cudaEventRecord(e1, cudaStreamPerThread);
            cudaEventSynchronize(e1);
            float msGraph = 0.0f;
            cudaEventElapsedTime(&msGraph, e0, e1);
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
            // accuracy
            const double bwd = backwardError(d, x, b);
            long double num = 0.0L, den = 0.0L;
            for (int i = 0; i < n; ++i)
            {
                num += (x[i] - xref[i])*(x[i] - xref[i]);
                den += xref[i]*xref[i];
            }
            const double fwd = static_cast<double>(std::sqrt(num/den));
            double diff = 0.0, big = 0.0;
            bool same = true;
            for (int i = 0; i < n; ++i)
            {
                diff = std::max(diff, std::fabs(x[i] - xlu[i]));
                big = std::max(big, std::fabs(xlu[i]));
                if (x[i] != xlu[i]) same = false;
            }
            double sym = -1.0;
            if (d.symmetric)
            {
                const std::vector<double> mv = solve(n, p, v);
                const std::vector<double> mu = solve(n, p, u);
                long double uMv = 0.0L, vMu = 0.0L, un = 0.0L, mvn = 0.0L;
                for (int i = 0; i < n; ++i)
                {
                    uMv += static_cast<long double>(u[i])*mv[i];
                    vMu += static_cast<long double>(v[i])*mu[i];
                    un += static_cast<long double>(u[i])*u[i];
                    mvn += static_cast<long double>(mv[i])*mv[i];
                }
                sym = static_cast<double>(std::fabs(uMv - vMu)/(std::sqrt(un)*std::sqrt(mvn)));
            }
            std::printf("%s,%d,%s,%s,%.2f,%.2f,%.3f,%.3f,%.3e,%.3e,%.3e,%d,%.3e\n", d.name.c_str(), n, kindName(kind), taken.c_str(),
                        setupLu, median(ts), 1e3*msLaunch/200.0, 1e3*msGraph/2000.0, bwd, fwd, big > 0.0 ? diff/big : diff,
                        same ? 1 : 0, sym);
            std::fflush(stdout);
        }
    }
    return 0;
}
