#pragma once
// The device arm: one rhoCentralFoam step on the device, in rhoCentralFoam.C's order; the twin of
// rhoCentralStep_cpp.cu. The arm reads nothing back by itself: the Courant numbers and the status of up to
// kReportWindow steps stay on the device until flush(). At a fixed deltaT the driver flushes once a window; under
// adjustTimeStep it flushes every step, because setDeltaT.H reads the previous step's Courant number.
#include "rhoCentralFoam.cuh"

#include <memory>

namespace brae {
namespace gpu {
namespace rhoCentral {

std::unique_ptr<brae::rhoCentral::Arm> makeDeviceArm(const brae::rhoCentral::Case& cs,
                                                     const brae::rhoCentral::State& s0,
                                                     brae::rhoCentral::PatchFields& pf, int control);

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
