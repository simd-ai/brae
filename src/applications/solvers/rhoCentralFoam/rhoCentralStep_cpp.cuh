#pragma once
// The host reference arm: one rhoCentralFoam step on the host, in rhoCentralFoam.C's order.
#include "rhoCentralFoam.cuh"

#include <memory>

namespace brae {
namespace cpu {
namespace rhoCentral {

std::unique_ptr<brae::rhoCentral::Arm> makeHostArm(const brae::rhoCentral::Case& cs, brae::rhoCentral::State& s,
                                                   brae::rhoCentral::PatchFields& pf);

} // namespace rhoCentral
} // namespace cpu
} // namespace brae
