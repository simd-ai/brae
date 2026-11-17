#pragma once
// rhoCentralFoam, the CUDA arm: the same step as stepCpp (rhoCentralFoam_cpp.cu), device-resident.
//
// Per step, eight launches and one small read-back:
//   aux (cells, patches)   rPsi = 1/psi, c = sqrt(Cp/Cv*rPsi)                       rhoCentralFoam.C:107, 135
//   gradients x3           Gauss linear grad of rho, rhoU (3), rPsi, e, c        deviceGaussGradFused (brae, reused)
//   faces, patch faces     vanLeer/vanLeerV directed reconstruction + the flux   :102-210
//   cells                  diagonal solves, U, e, thermo.correct, p             :222-287 (cell half)
//   patch faces            U/e/T/p patches, rhoU_b, rhoE_b, rho_b                :232-233, 265-271, 285-288
//   report                 central Courant number, status                       centralCourantNo.H
// The face arithmetic is rhoCentralFoam_cpp.cu's, expression for expression; the device gathers in OpenFOAM's
// scatter order (owner and neighbour faces merged by face index, then the patch faces), as brae's deviceDiv does.
#include "rhoCentralFoam_cpp.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"

#include <memory>
#include <vector>

namespace brae {
namespace rhoCentral {

// NEGATIVE CONTROLS, for the gates only: deliberately wrong numerics a gate must see. Selected by
// BRAE_CONTROL_RCF (gpuRhoCentralFoam.cu), device arm only, announced on stderr when set.
//   upwind           every directed reconstruction with limiter 0: first-order upwind instead of vanLeer/vanLeerV
//   noPressureJump   phiEp without its aSf*p_pos - aSf*p_neg term (rhoCentralFoam.C:208)
enum Control : int
{
    controlNone = 0,
    controlUpwind = 1,
    controlNoPressureJump = 2
};

class DeviceSolver
{
public:
    void setControl(int c) { control_ = c; }
    DeviceSolver(const Case& cs, const State& s0);
    ~DeviceSolver();

    // One step, queued on the device: nothing is read back. The step's Courant numbers and status are kept on the
    // device (up to kReportWindow steps) until flush() reads them, in step order.
    void step();
    std::vector<StepReport> flush();
    static constexpr int kReportWindow = 256;
    int pending() const { return pending_; }
    void download(State& s) const;

    // the running time integral of the boundary fluxes since the start, deltaT * sum over steps
    void boundaryIntegrals(scalar& mass, vector& mom, scalar& energy) const;

    std::size_t deviceBytes() const { return bytes_; }

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
    const Case& cs_;
    std::size_t bytes_ = 0;
    int control_ = controlNone;
    int pending_ = 0;
};

} // namespace rhoCentral
} // namespace brae
