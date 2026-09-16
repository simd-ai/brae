#pragma once
// brae's interFoam as a RUNNABLE SOLVER -- the time loop with every component wired to it.
//
// provenance:
//   openfoam: applications/solvers/multiphase/interFoam/interFoam.C:90-176
//   brae:     the stage ORDER is inter_solve_cpp.cuh's runTimeStep, which this fills in; the case-to-
//             fields translation is inter_case_cpp.cuh's buildInterFields, which the gate also calls.
//
// WHY THE DRIVER OWNS NO NUMERICS. Every stage below is one call into a component that has its own
// gate: alphaEqnStep, momentumPredictor, pressureCorrector, alphaCourantNo, setDeltaTVoF. What this
// file adds is the STATE -- which fields are carried between steps and which are rebuilt -- and that
// is the part interFoam.C itself is mostly about.
//
// THE OLD-TIME SET IS THE CONTENT OF THIS FILE. Four fields are carried from one step to the next and
// each is read by something different:
//
//   alpha1.oldTime()  the sub-cycle's starting value, and RESTORED after it so the PIMPLE outer loop's
//                     second pass starts from the same place (inter_solve_cpp.cuh, note d)
//   U.oldTime()       fvm::ddt(rho, U)'s source, and ddtCorr's interpolated flux
//   rho.oldTime()     fvm::ddt's SOURCE while the diagonal takes the new rho -- they differ by the
//                     density ratio in every cell the interface crossed (inter_ueqn_cpp.cuh, note 1)
//   phi.oldTime()     ddtCorr's stored flux, the half that has no cell-centred representation
//
// Losing any one of them leaves a solver that runs and is wrong in a way no single equation's gate
// would catch, because each is correct in isolation.
#include "cf_types.cuh"
#include "inter_case_cpp.cuh"
#include <string>

namespace brae {
namespace cpu {
namespace interFoam {

struct RunReport
{
    label  steps        = 0;
    scalar time         = 0;
    scalar deltaT       = 0;
    scalar CoNum        = 0;
    scalar alphaCoNum   = 0;
    scalar alphaMin     = 0;
    scalar alphaMax     = 0;
    scalar maxU         = 0;
    scalar worstDivPhi  = 0;
    scalar alphaMass    = 0;      // sum(alpha*V), which a closed domain must conserve
};

// Run `nSteps` of interFoam on a prepared case. Returns the state at the end; `verbose` prints the
// per-step line OpenFOAM's own solver prints.
RunReport runInterFoam(const std::string&          caseDir,
                       const std::string&          startDir,
                       const PrimitiveMesh&        m,
                       const FvGeometry&           g,
                       const std::vector<FvPatch>& patches,
                       label                       nSteps,
                       bool                        verbose = true,
                       // The FIELDS at the end, for a gate that has to compare them against
                       // OpenFOAM's own. A solver's real output is files; this exists so the
                       // comparison does not have to write and re-read them.
                       InterFields*                fieldsOut = nullptr);

} // namespace interFoam
} // namespace cpu
} // namespace brae
