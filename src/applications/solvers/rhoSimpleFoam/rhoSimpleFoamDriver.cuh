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
//   gpu::rhoSimple::correctTurbulence the closure hook, device-resident (rhoTurbulenceHook.cuh), and
//                                    its OPTIONS (buildTurbulenceHookOptions below) -- the CUDA harness
//                                    used to fill those by hand and dropped the fvOptions k/epsilon
//                                    constraints and minIter on the way.
//   the host field set               createDeviceFields projects the device state FROM the host's, so
//                                    the host createFields -- with every refusal it carries -- runs
//                                    first here exactly as it does in the harness.
#include "cf_types.cuh"
#include "rhoCreateFields.cuh"
#include "rhoSimpleFoam.cuh"
#include "rhoCaseRefusals.cuh"
#include "rhoSimpleFoam_cpp.cuh"
#include "rhoTurbulenceHook.cuh"
#include <string>

namespace brae {
namespace gpu {
namespace rhoSimple {

// The host StepInput -> the device one, plus the device pointers createDeviceFields owns. Shared with
// the CUDA harness so the gate and the solver drive the same input struct.
//
// `porosity` is CALLER-OWNED and must outlive the returned struct: RhoStepInput::porosity points into
// it, and it is left inactive on a case with no porous zone.
// The fvOptions CONSTRAINT buffers a case declares, per constrained field. Caller-owned, because
// RhoStepInput and the turbulence hook only hold pointers into them.
struct DeviceConstraints
{
    DeviceBuffer<label>  heMask,  kMask,  epsMask;
    DeviceBuffer<scalar> heVal,   kVal,   epsVal;
    bool hasHe = false, hasK = false, hasEps = false;
};

RhoStepInput buildDeviceStepInput(
    const cpu::rhoSimple::StepInput&        hin,
    const cpu::rhoSimple::RhoSimpleFields&  hf,
    const cpu::rhoSimple::CaseRefusals&     refusals,
    const RhoDeviceFields&                  dev,
    const std::vector<FvPatch>&             patches,
    DevicePorosity&                         porosity,
    DeviceConstraints&                      constraints,
    label                                   nCells);

// The closure hook's options FROM the shared StepInput: coefficients, Prt, the div/laplacian scheme
// flags, relaxation, the k/epsilon solver controls and the fvOptions constraint masks. Shared with the
// CUDA harness so the closure the gate drives is configured exactly as the one `brae -case` runs.
// `constraints` is caller-owned: the returned options point into its k/epsilon buffers.
TurbulenceHookOptions buildTurbulenceHookOptions(
    const cpu::rhoSimple::StepInput&        hin,
    const cpu::rhoSimple::RhoSimpleFields&  hf,
    const DeviceConstraints&                constraints);

// One whole run on the device: `brae -case <dir>` with BRAE_RHOSIMPLEFOAM_MIRROR=cuda.
int runMirrorCuda(const std::string& caseDir);

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
