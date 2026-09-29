#pragma once
// GradUMemo: grad(U) per component and the boundary values it used -- the memo deviceGradUShared keeps
// (device_kepsilon.cuh has what it is keyed on and why) and the form interFoam's cached grad(U) is handed to the
// momentum assembly in (DeviceGradUCache, device_inter_step.cuh). In its own header so a holder need not include
// device_kepsilon.cuh, whose device_ddt.cuh DdtScheme collides with interFoam's own for a caller using both
// namespaces.
#include "cf_types.cuh"
#include "device_buffer.cuh"

namespace brae {

struct GradUMemo
{
    int nC = 0;
    bool valid = false;
    unsigned long long fp = 0;
    DeviceBuffer<scalar> gx[3], gy[3], gz[3];     // gaussGrad(U_k), unlimited, interior + boundary faces
    DeviceBuffer<scalar> ub[3];                   // the boundary values it used (deviceBCValue per component)
    DeviceBuffer<unsigned long long> dev;         // device state: acc, stored fingerprint, valid, hit, nHit, nMiss
    unsigned long long computed = 0, reused = 0;  // read only under BRAE_GRADU_MEMO_STATS
};

} // namespace brae
