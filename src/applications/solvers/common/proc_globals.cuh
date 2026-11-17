#pragma once
// proc_globals.cuh -- the host-side globals a rank-local program needs to behave as ONE OpenFOAM run: gSum, gMin,
// gMax, gAverage of a rank's numbers, "does any rank say yes", rank-ordered gathers and the collective error.
//
// WHY A WRAPPER. Everything that is not the numerical step (setup constants, end-of-run statistics, the
// continuity error, function objects, the writer) asks a global question of rank-local data: OpenFOAM answers it
// with gSum/returnReduce over Pstream. Here ProcComm answers it. `pg::comm()` is null when there is no
// multi-rank communicator (no ProcComm installed, or size 1): every function below then returns its argument on
// its first line, touches nothing, and the single-GPU program runs the same instructions it always ran.
//
// ORDER. A reduction folds the ranks' values in rank order 0..np-1 (ProcComm's contract), the same bits on every
// rank. OpenFOAM folds along its communication tree (np <= 4: linear), so a sum of np ranks is bit-equal to
// OpenFOAM's only where the two orders coincide; min/max/integer results are exact.
//
// COLLECTIVE DISCIPLINE. Every function here that communicates is collective: every rank calls it, in the same
// order. A condition that only some ranks can raise is raised with pg::fail (ProcComm::collectiveAbort), which
// makes the other ranks throw the same ProcCommAbort at their next collective; a condition computed from a
// reduced value is the same on all ranks and is thrown plainly.
#include "proc_comm.cuh"
#include "cf_types.cuh"
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brae {
namespace pg {

// the communicator of this thread/process when it has more than one rank, else null
inline ProcComm* comm()
{
    ProcComm* c = ProcComm::current();
    return (c && c->size() > 1) ? c : nullptr;
}
inline bool parallel() { return comm() != nullptr; }
inline int  rank()     { ProcComm* c = comm(); return c ? c->rank() : 0; }
inline int  nProcs()   { ProcComm* c = comm(); return c ? c->size() : 1; }
inline bool isMaster() { return rank() == 0; }

inline double sum(double v)
{
    ProcComm* c = comm();
    return c ? c->allReduce(v, ReduceOp::Sum) : v;
}
inline double min(double v)
{
    ProcComm* c = comm();
    return c ? c->allReduce(v, ReduceOp::Min) : v;
}
inline double max(double v)
{
    ProcComm* c = comm();
    return c ? c->allReduce(v, ReduceOp::Max) : v;
}
// in place, n doubles at once (one collective)
inline void sum(double* v, int n)
{
    if (ProcComm* c = comm()) c->allReduceHost(v, n, ReduceOp::Sum);
}
inline void min(double* v, int n)
{
    if (ProcComm* c = comm()) c->allReduceHost(v, n, ReduceOp::Min);
}
inline void max(double* v, int n)
{
    if (ProcComm* c = comm()) c->allReduceHost(v, n, ReduceOp::Max);
}
// integer counts travel as doubles: exact below 2^53
inline long long sumCount(long long v)
{
    ProcComm* c = comm();
    return c ? static_cast<long long>(c->allReduce(static_cast<double>(v), ReduceOp::Sum)) : v;
}
// returnReduceAnd / returnReduceOr
inline bool andAll(bool v)
{
    ProcComm* c = comm();
    return c ? c->allReduce(v ? 1.0 : 0.0, ReduceOp::Min) != 0.0 : v;
}
inline bool orAll(bool v)
{
    ProcComm* c = comm();
    return c ? c->allReduce(v ? 1.0 : 0.0, ReduceOp::Max) != 0.0 : v;
}
inline double bcast(double v, int root)
{
    ProcComm* c = comm();
    return c ? c->bcastScalar(v, root) : v;
}
inline void barrier()
{
    if (ProcComm* c = comm()) c->barrier();
}

// The collective error: np == 1 is a plain std::runtime_error.
[[noreturn]] inline void fail(const std::string& msg)
{
    if (ProcComm* c = comm()) c->collectiveAbort(msg);
    throw std::runtime_error(msg);
}

// Every rank's list, on every rank (OpenFOAM's Pstream::gatherList + scatterList of a list of lists), in rank
// order. Doubles travel bit-exactly: the gather is a Max over a buffer that is -inf where a rank does not write.
// NaN payloads are not preserved.
inline std::vector<std::vector<double>> gatherAll(const std::vector<double>& local)
{
    ProcComm* c = comm();
    if (!c) return std::vector<std::vector<double>>(1, local);
    const int np = c->size();
    std::vector<double> counts(static_cast<std::size_t>(np), 0.0);
    counts[static_cast<std::size_t>(c->rank())] = static_cast<double>(local.size());
    c->allReduceHost(counts.data(), np, ReduceOp::Sum);
    std::vector<std::size_t> start(static_cast<std::size_t>(np) + 1, 0);
    for (int r = 0; r < np; ++r) start[static_cast<std::size_t>(r) + 1] = start[static_cast<std::size_t>(r)]
                                                                      + static_cast<std::size_t>(counts[static_cast<std::size_t>(r)]);
    std::vector<double> buf(start[static_cast<std::size_t>(np)], -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < local.size(); ++i) buf[start[static_cast<std::size_t>(c->rank())] + i] = local[i];
    if (!buf.empty()) c->allReduceHost(buf.data(), static_cast<int>(buf.size()), ReduceOp::Max);
    std::vector<std::vector<double>> out(static_cast<std::size_t>(np));
    for (int r = 0; r < np; ++r)
    {
        out[static_cast<std::size_t>(r)].assign(buf.begin() + static_cast<std::ptrdiff_t>(start[static_cast<std::size_t>(r)]),
                                                buf.begin() + static_cast<std::ptrdiff_t>(start[static_cast<std::size_t>(r) + 1]));
    }
    return out;
}

// gAverage of a field of cell values: (sum over ranks of the local sums) / (sum of the sizes)
// (Field.C:647-671: local sum(f), then sumReduce(result, count)). A serial program uses its own loop.
inline double globalAverage(const std::vector<double>& v)
{
    double s = 0;
    for (double x : v) s += x;
    double sc[2] = {s, static_cast<double>(v.size())};
    sum(sc, 2);
    return sc[1] > 0 ? sc[0]/sc[1] : 0.0;
}

// The report's end-of-run statistics are global: max|U|, worst |div(phi)|, alpha min/max, alpha mass.
// (OpenFOAM prints none of them; they are brae's own report, and a rank's own extremum would be a false
// statement about the run.) No-op without a communicator.
inline void reduceEndStats(double& maxU, double& worstDivPhi, double& alphaMin, double& alphaMax, double& alphaMass)
{
    if (!comm()) return;
    double mx[2] = {maxU, worstDivPhi};
    max(mx, 2);
    maxU = mx[0];
    worstDivPhi = mx[1];
    double mn[1] = {alphaMin};
    min(mn, 1);
    alphaMin = mn[0];
    double am[1] = {alphaMax};
    max(am, 1);
    alphaMax = am[0];
    alphaMass = sum(alphaMass);
}

}   // namespace pg
}   // namespace brae
