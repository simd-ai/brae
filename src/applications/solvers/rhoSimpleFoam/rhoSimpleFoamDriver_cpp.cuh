#pragma once
// rhoSimpleFoamDriver_cpp.cuh -- the OF-mirror rhoSimpleFoam as a RUNNABLE SOLVER, not a harness.
//
// Everything under this directory was, until now, reachable only from tests/test_rho_simple_step_cpp.cu:
// the step and its createFields are gated end to end against real OpenFOAM, and no user could run them.
// `brae -case <dir>` on a `application rhoSimpleFoam` case reached the PRE-MIRROR driver
// (gpuRhoSimpleFoam.cu) instead, which is a different implementation of the same solver.
//
// Two functions, and the split between them is the point:
//
//   buildStepInput  the case's dictionaries -> StepInput. SHARED with the harness, so the translation
//                   the gates measure is the translation the solver runs. A private copy in the driver
//                   would be the defect this project keeps finding one level up: the gate proves the
//                   step, the driver feeds it something else, and nothing compares the two.
//
//   runMirror       controlDict, the SIMPLE loop, residualControl, OF-format time directories. The
//                   loop shape is OF's own (Time::loop / writeAndEnd), reusing the same shared classes
//                   the incompressible and legacy compressible drivers use -- brae::Time (startFrom,
//                   write cadence, functionObjects), ResidualControl (OF's `achieved && checked` rule)
//                   and foam_field_writer.
//
// WHAT IT DOES NOT DO. Every refusal the step and createFields carry is in force here: an unported
// thermo, RAS model, boundary condition, fvOption, MRF zone, coupled patch or turbulence convection
// scheme throws by name rather than running a substitute. That is the contract that makes the mirror
// worth shipping at all -- a case brae cannot mirror exactly does not run.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "rhoCaseRefusals.cuh"
#include "rhoCreateFields_cpp.cuh"
#include "rhoSimpleFoam_cpp.cuh"
#include <string>

namespace brae {
namespace cpu {
namespace rhoSimple {

// The case's dictionaries -> StepInput: schemes, relaxation, refusals, turbulence convection.
//
// `refusals` is CALLER-OWNED and must outlive the returned StepInput: StepInput::fvOpts points into
// its `opts` vector (the fvOptions the case declares and brae implements).
StepInput buildStepInput(
    const std::string&    caseDir,
    const RhoSimpleFields& f,
    const FoamDict&       fvSolution,
    const PrimitiveMesh&  m,
    CaseRefusals&         refusals,
    bool                  verbose = true);

// One whole run: `brae -case <dir>`. Returns a process exit code (0 = ran to endTime or converged).
int runMirror(const std::string& caseDir);

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
