#pragma once
// rhoSimpleFoamDriver.cuh -- the OF-mirror rhoSimpleFoam on the DEVICE, as a runnable solver.
//
// The host twin (rhoSimpleFoamDriver_cpp) made the mirror runnable; this is the same solver with the
// device modules doing the arithmetic. Everything OUTSIDE the iteration is shared with the host driver
// on purpose -- the controlDict/Time/residualControl/write machinery is not solver arithmetic, and two
// copies of it is how the write cadence and the endTime rule drift apart between two paths that are
// supposed to be the same solver.
//
// WHAT IS SHARED, and why each:
//   cpu::rhoSimple::buildStepInput   the case -> StepInput parse. The device input struct is filled FROM
//                                    it (buildDeviceStepInput below), so a scheme, a relaxation factor
//                                    or a refusal the case names cannot reach one arm and not the other.
//   gpu::rhoSimple::correctTurbulence the closure hook, device-resident (rhoTurbulenceHook.cuh).
//   the host field set               createDeviceFields projects the device state FROM the host's, so
//                                    the host createFields -- with every refusal it carries -- runs
//                                    first here exactly as it does in the harness.
#include "cf_types.cuh"
#include "rhoCreateFields.cuh"
#include "rhoSimpleFoam.cuh"
#include "rhoCaseRefusals.cuh"
#include "rhoSimpleFoam_cpp.cuh"
#include <string>

namespace brae {
namespace gpu {
namespace rhoSimple {

// The host StepInput -> the device one, plus the device pointers createDeviceFields owns. Shared with
// the CUDA harness so the gate and the solver drive the same input struct.
//
// `porosity` is CALLER-OWNED and must outlive the returned struct: RhoStepInput::porosity points into
// it, and it is left inactive on a case with no porous zone.
RhoStepInput buildDeviceStepInput(
    const cpu::rhoSimple::StepInput&        hin,
    const cpu::rhoSimple::RhoSimpleFields&  hf,
    const cpu::rhoSimple::CaseRefusals&     refusals,
    const RhoDeviceFields&                  dev,
    const std::vector<FvPatch>&             patches,
    DevicePorosity&                         porosity);

// One whole run on the device: `brae -case <dir>` with BRAE_RHOSIMPLEFOAM_MIRROR=cuda.
int runMirrorCuda(const std::string& caseDir);

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
