#pragma once
// k-omega SST model coefficients, shared by the device (device_komega_sst) and CPU paths. Defaults = OpenFOAM
// v2412 kOmegaSST defaults (kOmegaSSTBase.C constructor, getOrAddToDict values). Read from turbulenceProperties
// RAS.kOmegaSSTCoeffs (absent keys keep OF defaults). kappa/E are the wall-function coeffs (the omega/nut wall
// functions; OF reads them from the BC dicts, default 0.41/9.8); the wall-function Cmu == betaStar. A
// default-constructed struct reproduces the OF defaults exactly, so callers that don't read a dict stay faithful.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <string>

namespace brae {

struct KOmegaSSTCoeffs
{
    // gradSchemes/grad(k) and grad(omega). CDkOmega is (2*alphaOmega2/omega)*(grad(k) & grad(omega)) and
    // F1 is built from it, so the blend between the k-omega and k-epsilon branches is only as right as
    // these gradients. OpenFOAM resolves each against its own gradSchemes entry -- aerofoilNACA0012 names
    // `cellLimited Gauss linear 1` for both while squareBend leaves them at the unlimited default, so a
    // fixture with only the latter cannot tell the two apart. 0 = unlimited.
    scalar gradKLimitK = 0.0;

    scalar alphaK1 = 0.85,     alphaK2 = 1.0;          // k diffusivity blend  (kOmegaSSTBase.C:263,272)
    scalar alphaOmega1 = 0.5,  alphaOmega2 = 0.856;    // omega diffusivity blend         (:281,290)
    scalar gamma1 = 5.0/9.0,   gamma2 = 0.44;          // production coeff blend          (:299,308)
    scalar beta1 = 0.075,      beta2 = 0.0828;         // omega destruction blend         (:317,326)
    scalar betaStar = 0.09;                            // k destruction / Cmu             (:335)
    scalar a1 = 0.31, b1 = 1.0, c1 = 10.0;             // nut limiter (a1,b1) + Pk cap c1 (:344,353,362)
    bool   F3 = false;                                 // F3 near-wall correction switch  (:371)
    scalar kappa = 0.41, E = 9.8;                      // wall-function coeffs (Cmu_wf == betaStar)
    scalar CDES1 = 0.78, CDES2 = 0.61;                 // kOmegaSST-DES/DDES C_DES blend (OF kOmegaSSTDES defaults)
    // kOmegaSST-IDDES (Gritskevich/Garbaruk/Schuetze/Menter 2012) blending constants. Exponents fixed per the reference
    // (f_dt cube, f_l ^10, f_t cube); only the multipliers are carried here (shared values with SA-IDDES).
    scalar Cdt1 = 20.0, Cl = 5.0, Ct = 1.87, Cw = 0.15;
};

// Read RAS.kOmegaSSTCoeffs into c (absent keys keep OF defaults). `ras` may be null (-> all defaults).
inline void readKOmegaSSTCoeffs(const FoamDict* ras, KOmegaSSTCoeffs& c)
{
    const FoamDict* kc = ras ? ras->subDict("kOmegaSSTCoeffs") : nullptr;
    if (!kc) return;
    c.alphaK1 = kc->scalarOr("alphaK1", c.alphaK1);
    c.alphaK2 = kc->scalarOr("alphaK2", c.alphaK2);
    c.alphaOmega1 = kc->scalarOr("alphaOmega1", c.alphaOmega1);
    c.alphaOmega2 = kc->scalarOr("alphaOmega2", c.alphaOmega2);
    c.gamma1 = kc->scalarOr("gamma1", c.gamma1);
    c.gamma2 = kc->scalarOr("gamma2", c.gamma2);
    c.beta1  = kc->scalarOr("beta1",  c.beta1);
    c.beta2  = kc->scalarOr("beta2",  c.beta2);
    c.betaStar = kc->scalarOr("betaStar", c.betaStar);
    c.a1 = kc->scalarOr("a1", c.a1);
    c.b1 = kc->scalarOr("b1", c.b1);
    c.c1 = kc->scalarOr("c1", c.c1);
    const std::string f3 = kc->wordOr("F3", c.F3 ? "true" : "false");   // Switch: true/on/yes/1
    c.F3 = (f3 == "true" || f3 == "on" || f3 == "yes" || f3 == "1");
    c.kappa = kc->scalarOr("kappa", c.kappa);
    c.E = kc->scalarOr("E", c.E);
}

} // namespace brae
