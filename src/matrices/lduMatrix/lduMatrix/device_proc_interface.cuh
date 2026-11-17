#pragma once
// Owner of the device data behind a DeviceProcView (device_ldu.cuh): the processor-interface coefficients of one rank's
// matrix and the per-cell face order. One per (rank, matrix addressing); the coefficients are refilled every time the
// matrix is reassembled (setCoeff / coeff()), the plan-derived index arrays are fixed.
//
//   DeviceProcInterface ifc(comm, plan);              // collective (the global cell count is one reduction)
//   ifc.setCoeff(hostCoeffPerProcessorFace);          // = -interfaceBouCoeffs, patch face order, the plan's face order
//   DeviceLduView A = ...; A.proc = ifc.view();
//
// np == 1 (comm.size() == 1): view() is inactive and nothing is allocated, so a single-GPU run that goes through
// this code is the single-GPU run.
#include "device_ldu.cuh"
#include "proc_comm.cuh"
#include <vector>

namespace brae {

class DeviceProcInterface
{
public:
    DeviceProcInterface(ProcComm& comm, const HaloPlan& plan);

    DeviceBuffer<scalar>& coeff() { return coeff_; }
    void setCoeff(const std::vector<scalar>& hostCoeff);   // length plan.nProcFaces()
    DeviceProcView view() const;

private:
    ProcComm* comm_;
    const HaloPlan* plan_;
    DeviceBuffer<scalar> coeff_;
    DeviceBuffer<label> rank_;
    DeviceBuffer<label> passStart_, faceCell_, nbrIdx_, perm_;
    int nRanks_ = 1;
    long long nGlobalCells_ = 0;
};

} // namespace brae
