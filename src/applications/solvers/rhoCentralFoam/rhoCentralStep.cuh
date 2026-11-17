#pragma once
// The device arm: one rhoCentralFoam step on the device, in rhoCentralFoam.C's order; the twin of
// rhoCentralStep_cpp.cu. Nothing is read back per step: the Courant numbers and the status of up to
// kReportWindow steps stay on the device until flush().
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
