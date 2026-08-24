// ACPP-only stand-in for device_coded_bc.cu: coded BCs JIT-compile user C++ via NVRTC, no PCUDA equivalent.
// These stubs let brae_core link; a case using a coded BC fails loudly at runtime instead.
#include "device_coded_bc.cuh"
#include <stdexcept>

namespace brae {

namespace {
[[noreturn]] void notSupported(const char* what)
{
    throw std::runtime_error(std::string("brae: coded boundary conditions (") + what +
                              ") are not supported in the ACPP build -- no NVRTC/driver-API JIT path under "
                              "AdaptiveCpp/PCUDA. Use the CUDA build (BRAE_BACKEND=CUDA) for coded BCs.");
}
} // namespace

CodedBcKernel compileCodedVectorBc(const std::string&, const std::string&) { notSupported("compileCodedVectorBc"); }
CodedBcKernel compileCodedScalarBc(const std::string&, const std::string&) { notSupported("compileCodedScalarBc"); }
CodedBcKernel compileCodedMixedVectorBc(const std::string&, const std::string&) { notSupported("compileCodedMixedVectorBc"); }
CodedBcKernel compileCodedMixedScalarBc(const std::string&, const std::string&) { notSupported("compileCodedMixedScalarBc"); }

void launchCodedVectorBc(
    const CodedBcKernel&, int, int, scalar,
    const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&,
    const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&,
    const DeviceBuffer<label>&,
    DeviceBuffer<scalar>&, DeviceBuffer<scalar>&, DeviceBuffer<scalar>&)
{
    notSupported("launchCodedVectorBc");
}

void launchCodedScalarBc(
    const CodedBcKernel&, int, int, scalar,
    const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&,
    const DeviceBuffer<scalar>&, const DeviceBuffer<label>&,
    DeviceBuffer<scalar>&)
{
    notSupported("launchCodedScalarBc");
}

void launchCodedMixedVectorBc(
    const CodedBcKernel&, int, int, scalar,
    const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&,
    const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&,
    const DeviceBuffer<label>&,
    DeviceBuffer<scalar>&, DeviceBuffer<scalar>&, DeviceBuffer<scalar>&,
    DeviceBuffer<scalar>&, DeviceBuffer<scalar>&, DeviceBuffer<scalar>&)
{
    notSupported("launchCodedMixedVectorBc");
}

void launchCodedMixedScalarBc(
    const CodedBcKernel&, int, int, scalar,
    const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&,
    const DeviceBuffer<scalar>&, const DeviceBuffer<label>&,
    DeviceBuffer<scalar>&, DeviceBuffer<scalar>&)
{
    notSupported("launchCodedMixedScalarBc");
}

} // namespace brae
