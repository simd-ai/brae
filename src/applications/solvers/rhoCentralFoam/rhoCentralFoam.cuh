#pragma once
// rhoCentralFoam's driver: rhoCentralFoam.C's main() -- start-up, the time loop, the writes -- held once for both
// arms. An arm is the step itself: the host reference (rhoCentralStep_cpp.cu) or the device (rhoCentralStep.cu).
#include "rhoCentralControls.cuh"
#include "rhoCentralState.cuh"

#include <cstddef>
#include <string>
#include <vector>

namespace brae {
namespace rhoCentral {

// The step of rhoCentralFoam.C:102-288 (static mesh, !LTS, runTime already advanced), on one arm.
class Arm
{
public:
    virtual ~Arm() = default;

    // advance the state one step of `deltaT` (runTime.deltaTValue() after setDeltaT.H and ++runTime: the Euler
    // ddt's 1/deltaT and the Courant numbers' scale); its report is kept until flush()
    virtual void step(scalar deltaT) = 0;
    // the reports of the steps since the last flush, in step order
    virtual std::vector<StepReport> flush() = 0;
    virtual int pending() const = 0;
    // how many reports the arm keeps before it must be flushed
    virtual int window() const = 0;
    // the state into `s` (the device arm copies it back; the host arm's is already there)
    virtual void download(State& s) = 0;
    // deltaT * the sum over steps of what left through the boundary: mass, momentum, energy (diagnostics)
    virtual void boundaryOutflow(scalar& mass, vector& mom, scalar& energy) const = 0;
    // over the steps and the faces where U, p or T is inletOutlet: how many took the inflow branch (phi < 0), of all
    virtual void inletOutletInflow(long long& inflow, long long& faceSteps) const = 0;
    virtual std::size_t deviceBytes() const { return 0; }
};

// Integral diagnostics of a state: totals of V*rho, V*rhoU, V*rhoE, and the extrema the positivity check reads.
struct Integrals
{
    long double mass = 0, energy = 0;
    long double mom[3] = {0, 0, 0};
    scalar minRho = 0, minP = 0, minT = 0, minE = 0;
    scalar maxRho = 0, maxP = 0, maxT = 0, maxMagU = 0;
    // sum over the cells of ulp(|q|)*V: one unit in the last place of each cell's total, the scale of one rounding
    // of the conservative update (diag roundoff); for momentum, of |rhoU| in every component
    long double ulpMass = 0, ulpEnergy = 0;
    long double ulpMom[3] = {0, 0, 0};
};
Integrals integrals(const Case& cs, const State& s);

// Writes U, p, T, rho to <caseDir>/<timeName>, as OpenFOAM's AUTO_WRITE fields, at the case's writePrecision.
void writeTime(const Case& cs, const State& s, const std::string& timeName);

// The whole run. onDevice selects the arm; control is a gate's negative control (rhoCentralState.cuh says which arm).
int runRhoCentralFoam(const std::string& caseDir, bool onDevice, int control);

} // namespace rhoCentral
} // namespace brae
