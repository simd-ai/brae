#pragma once
// rhoCentralFoam's central-upwind flux assembly on the host: rhoCentralFoam.C:102-210 (directed interpolation of
// rho, rhoU, rPsi, e and c through `reconstruct(...)`, then the face flux), centralCourantNo.H, and the three
// fvc::div of the explicit solves (surfaceIntegrate). The host reference the device arm (rhoCentralFlux.cu) is held
// to, and both are held to OpenFOAM.
#include "rhoCentralControls.cuh"
#include "rhoCentralState.cuh"

#include <vector>

namespace brae {
namespace cpu {
namespace rhoCentral {

using brae::rhoCentral::Case;
using brae::rhoCentral::State;

struct HostFluxes
{
    std::vector<scalar> divPhi, divPhiEp;   // fvc::div(phi), fvc::div(phiEp), per cell
    std::vector<vector> divPhiUp;           // fvc::div(phiUp)
    std::vector<scalar> phiB;               // phi on every boundary face, flat in patch order (0 on empty ones): the
                                            // flux inletOutlet switches on
    scalar CoNum = 0, meanCoNum = 0;        // centralCourantNo.H
    // what left through the non-empty boundary faces this step (diagnostics)
    scalar bndMass = 0, bndEnergy = 0;
    vector bndMom{0, 0, 0};
};

// deltaT: this step's, which the Courant numbers are scaled by (centralCourantNo.H reads runTime.deltaTValue())
HostFluxes centralFluxes(const Case& cs, const State& s, scalar deltaT);

} // namespace rhoCentral
} // namespace cpu
} // namespace brae
