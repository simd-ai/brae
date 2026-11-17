#include "device_proc_interface.cuh"

#include <stdexcept>

namespace brae {

DeviceProcInterface::DeviceProcInterface(ProcComm& comm, const HaloPlan& plan)
:
    comm_(&comm),
    plan_(&plan)
{
    nGlobalCells_ = plan.nCells();
    if (comm.size() == 1) return;
    nGlobalCells_ = static_cast<long long>(comm.allReduce(static_cast<double>(plan.nCells()), ReduceOp::Sum));
    const int n = plan.nProcFaces();
    coeff_.resize(static_cast<std::size_t>(n));
    if (n > 0)
    {
        const std::vector<label> own(plan.desc().faceCells.begin(), plan.desc().faceCells.end());
        const std::vector<label> rank = pairOwnerRanks(own, nRanks_);
        // Stable grouping by pass. Within a pass the original face order is retained, so a cell's additions stay
        // bitwise identical to the old full-face/filter launches; only the grid extents change.
        std::vector<label> count(nRanks_, 0), start(nRanks_ + 1, 0), perm(n), fc(n), ni(n);
        for (const label p : rank) ++count[p];
        for (int p = 0; p < nRanks_; ++p) start[p + 1] = start[p] + count[p];
        std::vector<label> next = start;
        for (int i = 0; i < n; ++i)
        {
            const int j = next[rank[i]]++;
            perm[j] = i;
            fc[j] = plan.desc().faceCells[i];
            ni[j] = plan.desc().nCells + plan.desc().faceToSlot[i];
        }
        passStart_.copyFrom(start);
        faceCell_.copyFrom(fc);
        nbrIdx_.copyFrom(ni);
        // Keep the ranks in the SAME grouped order as the face arrays: the full-face fallback (A/B control and
        // older callers) then reproduces the original filtered launch exactly, with no mixed indexing.
        std::vector<label> groupedRank(n);
        for (int j = 0; j < n; ++j) groupedRank[j] = rank[perm[j]];
        rank_.copyFrom(groupedRank);
        perm_.copyFrom(perm);
    }
}

// Coefficients are uploaded in the original patch-face order. Group them with the same stable permutation as the
// face addressing, so launchProcFaces can run contiguous ranges rather than filter a full-face grid per pass.

void DeviceProcInterface::setCoeff(const std::vector<scalar>& hostCoeff)
{
    if (comm_->size() == 1) return;
    if (hostCoeff.size() != static_cast<std::size_t>(plan_->nProcFaces()))
    {
        throw std::invalid_argument("DeviceProcInterface::setCoeff: expected one coefficient per processor face");
    }
    if (perm_.size() == hostCoeff.size() && !hostCoeff.empty())
    {
        const std::vector<label> perm = perm_.host();
        std::vector<scalar> grouped(hostCoeff.size());
        for (std::size_t i = 0; i < hostCoeff.size(); ++i) grouped[i] = hostCoeff[perm[i]];
        coeff_.copyFrom(grouped);
    }
    else coeff_.copyFrom(hostCoeff);
}

DeviceProcView DeviceProcInterface::view() const
{
    DeviceProcView v;
    if (comm_->size() == 1) return v;
    v.comm = comm_;
    v.plan = plan_;
    v.nFaces = plan_->nProcFaces();
    v.faceCell = faceCell_.data();
    v.nbrIdx = nbrIdx_.data();
    v.coeff = coeff_.data();
    v.rank = rank_.data();
    v.nRanks = nRanks_;
    v.passStart = passStart_.data();
    const std::vector<label> starts = passStart_.host();
    for (int p = 0; p <= nRanks_ && p < 65; ++p) v.passFirst[p] = starts[p];
    v.nGlobalCells = nGlobalCells_;
    return v;
}

} // namespace brae
