#pragma once
// The central-upwind flux of one face, rhoCentralFoam.C:114-210, for both arms (BRAE_HD).
//
// Inputs are the face's two directed reconstructions ([0] = _pos, [1] = _neg: rho, rhoU, rPsi, e and c), its Sf
// and |Sf|. Outputs are the mass flux phi, the momentum flux phiUp (convective plus a_pos*p_pos + a_neg*p_neg on
// Sf), the energy flux phiEp and amaxSf (the larger one-sided flux, which centralCourantNo.H sums). Written once so
// the host reference and the device kernels evaluate the same expressions; on an uncoupled patch face both
// directions carry the patch value.
#include "cf_types.cuh"
#include "rhoCentralState.cuh"

#include <cmath>

namespace brae {
namespace rhoCentral {

struct FaceStates
{
    scalar rho[2], rPsi[2], e[2], c[2];
    vector rhoU[2];
};

struct FaceFlux
{
    scalar phi, phiUx, phiUy, phiUz, phiEp, amaxSf;
};

BRAE_HD inline scalar maxOF(scalar a, scalar b) { return (a > b) ? a : b; }   // Foam::max
BRAE_HD inline scalar minOF(scalar a, scalar b) { return (a < b) ? a : b; }   // Foam::min

// ctrl: controlNoPressureJump drops aSf*p_pos - aSf*p_neg from phiEp (a gate control, rhoCentralState.cuh)
BRAE_HD inline FaceFlux centralFaceFlux(const FaceStates& in, scalar Sx, scalar Sy, scalar Sz, scalar magSf,
                                        bool tadmor, int ctrl)
{
    const vector U_pos{in.rhoU[0].x/in.rho[0], in.rhoU[0].y/in.rho[0], in.rhoU[0].z/in.rho[0]};   // :114
    const vector U_neg{in.rhoU[1].x/in.rho[1], in.rhoU[1].y/in.rho[1], in.rhoU[1].z/in.rho[1]};   // :115
    const scalar p_pos = in.rho[0]*in.rPsi[0];                                                    // :117
    const scalar p_neg = in.rho[1]*in.rPsi[1];                                                    // :118
    scalar phiv_pos = U_pos.x*Sx + U_pos.y*Sy + U_pos.z*Sz;                                       // :120
    scalar phiv_neg = U_neg.x*Sx + U_neg.y*Sy + U_neg.z*Sz;                                       // :123
    const scalar cSf_pos = in.c[0]*magSf;                                                         // :136-140
    const scalar cSf_neg = in.c[1]*magSf;                                                         // :142-146
    const scalar ap = maxOF(maxOF(phiv_pos + cSf_pos, phiv_neg + cSf_neg), 0.0);                  // :148-152
    const scalar am = minOF(minOF(phiv_pos - cSf_pos, phiv_neg - cSf_neg), 0.0);                  // :154-158
    scalar a_pos = ap/(ap - am);                                                                  // :160
    scalar amaxSf = maxOF(fabs(am), fabs(ap));                                                    // :162
    scalar aSf = am*a_pos;                                                                        // :164
    if (tadmor)                                                                                   // :166-170
    {
        aSf = -0.5*amaxSf;
        a_pos = 0.5;
    }
    const scalar a_neg = 1.0 - a_pos;                                                             // :172
    phiv_pos *= a_pos;                                                                            // :174
    phiv_neg *= a_neg;                                                                            // :175
    const scalar aphiv_pos = phiv_pos - aSf;                                                      // :177
    const scalar aphiv_neg = phiv_neg + aSf;                                                      // :178
    amaxSf = maxOF(fabs(aphiv_pos), fabs(aphiv_neg));                                             // :182

    FaceFlux o;
    o.amaxSf = amaxSf;
    o.phi = aphiv_pos*in.rho[0] + aphiv_neg*in.rho[1];                                            // :195
    const scalar pf = a_pos*p_pos + a_neg*p_neg;                                                  // :202
    o.phiUx = (aphiv_pos*in.rhoU[0].x + aphiv_neg*in.rhoU[1].x) + pf*Sx;                          // :197-202
    o.phiUy = (aphiv_pos*in.rhoU[0].y + aphiv_neg*in.rhoU[1].y) + pf*Sy;
    o.phiUz = (aphiv_pos*in.rhoU[0].z + aphiv_neg*in.rhoU[1].z) + pf*Sz;
    const scalar K_pos = U_pos.x*U_pos.x + U_pos.y*U_pos.y + U_pos.z*U_pos.z;
    const scalar K_neg = U_neg.x*U_neg.x + U_neg.y*U_neg.y + U_neg.z*U_neg.z;
    o.phiEp = aphiv_pos*(in.rho[0]*(in.e[0] + 0.5*K_pos) + p_pos)                                  // :204-210
            + aphiv_neg*(in.rho[1]*(in.e[1] + 0.5*K_neg) + p_neg);
    if (ctrl != controlNoPressureJump) o.phiEp = o.phiEp + aSf*p_pos - aSf*p_neg;
    return o;
}

} // namespace rhoCentral
} // namespace brae
