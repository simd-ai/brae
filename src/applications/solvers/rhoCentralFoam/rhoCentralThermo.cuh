#pragma once
// rhoCentralFoam's thermo arithmetic: hePsiThermo + pureMixture + hConst + perfectGas + sensibleInternalEnergy,
// written as OpenFOAM v2412 evaluates it, term by term, so the host reference and the device kernels run the
// same expressions.
//
// provenance:
//   openfoam: src/thermophysicalModels/specie/thermo/hConst/hConstThermoI.H:94-121 (Cp, Hs)
//             src/thermophysicalModels/specie/thermo/thermo/HtoEthermo.H (Cv = Cp - CpMCv, Es = Hs - p/rho)
//             src/thermophysicalModels/specie/equationOfState/perfectGas/perfectGasI.H:76-135 (rho, H, Cp, psi, CpMCv)
//             src/thermophysicalModels/specie/thermo/thermo/thermoI.H:44-88 (the Newton loop T(f, p, T0)),
//             :397-413 (TEs), thermo.C:33-36 (tol_ 1e-4, maxIter_ 100)
//
// Why the Newton loop and not the closed form. hConst's Es is affine in T, so (e - Hsref + Cp*Tref)/Cv is the
// fixed point; but OpenFOAM does not compute that. It runs Tnew = Test - (Es(p, Test) - e)/Cv(p, Test) from the
// field's own previous T, stopping when |Tnew - Test| <= T0*1e-4, and Es carries p/(p/(R*T)), which is R*T only to
// rounding. The two answers part in the last bits, and an explicit solver run for hundreds of steps through a
// TVD limiter is where last bits are not guaranteed to stay last bits. Same expressions, same seed, same stop.
#include "cf_types.cuh"

namespace brae {
namespace rhoCentral {

struct GasCoeffs
{
    scalar R     = 0;   // specie::R() = RR/molWeight (specieI.H:87-90)
    scalar Cp    = 0;   // hConst Cp_
    scalar Tref  = 0;   // hConst Tref_ (default Tstd)
    scalar Hsref = 0;   // hConst Hsref_ (read as `Href`, hConstThermo.C:40)
    scalar mu    = 0;   // constTransport mu_ (milestone G: > 0 runs the viscous correctors)
    scalar rPr   = 0;   // constTransport rPr_ = 1.0/Pr (constTransport.C:38)
    int    sutherland = 0;   // transport sutherland (milestone H): mu(T) from As, Ts
    scalar As    = 0;
    scalar Ts    = 0;
};

// perfectGas::Cp = 0, so hConst Cp(p, T) = Cp_ + 0
BRAE_HD inline scalar gasCp(const GasCoeffs& g)
{
    return g.Cp + scalar(0);
}

// HtoEthermo.H: Cv = Cp(p, T) - EquationOfState::CpMCv(p, T), perfectGas CpMCv = R
BRAE_HD inline scalar gasCv(const GasCoeffs& g)
{
    return gasCp(g) - g.R;
}

// thermoI.H:125-134, species::thermo::gamma: Cp/(Cp - CpMCv), the sensibleInternalEnergy CpByCpv
// (sensibleInternalEnergy.H:97-108)
BRAE_HD inline scalar gasGamma(const GasCoeffs& g)
{
    const scalar Cp = gasCp(g);
    return Cp/(Cp - g.R);
}
// The laminar thermal diffusivity rhoCentralFoam's e equation reads (rhoCentralFoam.C:275,
// turbulence->alphaEff()): ThermalDiffusivity::alphaEff() = thermo.alphahe() = CpByCpv*alpha_ (heThermo.C:809-813),
// alpha_ = constTransport::alphah = mu*rPr (constTransportI.H:101-108, set in hePsiThermo::calculate). With const
// transport and hConst both factors are the same in every cell and on every face.
BRAE_HD inline scalar gasAlphaEff(const GasCoeffs& g)
{
    return gasGamma(g)*(g.mu*g.rPr);
}
// sutherlandTransportI.H:114-143 (milestone H): mu = As*sqrt(T)/(1 + Ts/T); kappa = mu*Cv*(1.32 + 1.77*R/Cv) (the
// modified Eucken correlation; Pr is not read); alphah = kappa/Cp. Cv = Cp - R (hConst on a perfectGas).
BRAE_HD inline scalar gasSutherlandMu(scalar T, const GasCoeffs& g)
{
    return g.As*sqrt(T)/(1.0 + g.Ts/T);
}
// thermo.alphahe() = CpByCpv*alphah at a cell or face of temperature T (sutherland), heThermo.C:809-813
BRAE_HD inline scalar gasSutherlandAlphahe(scalar T, const GasCoeffs& g)
{
    const scalar Cv = gasCv(g);
    const scalar kappa = gasSutherlandMu(T, g)*Cv*(1.32 + 1.77*g.R/Cv);
    return gasGamma(g)*(kappa/gasCp(g));
}
// hConstThermoI.H:120: Cp_*(T - Tref_) + Hsref_ + EquationOfState::H(p, T), perfectGas H = 0
BRAE_HD inline scalar gasHs(scalar /*p*/, scalar T, const GasCoeffs& g)
{
    return g.Cp*(T - g.Tref) + g.Hsref + scalar(0);
}

// perfectGasI.H:78
BRAE_HD inline scalar gasRho(scalar p, scalar T, const GasCoeffs& g)
{
    return p/(g.R*T);
}

// HtoEthermo.H: Es = Hs(p, T) - p/EquationOfState::rho(p, T). sensibleInternalEnergy::HE is Es.
BRAE_HD inline scalar gasEs(scalar p, scalar T, const GasCoeffs& g)
{
    return gasHs(p, T, g) - p/gasRho(p, T, g);
}

// perfectGasI.H:120
BRAE_HD inline scalar gasPsi(scalar /*p*/, scalar T, const GasCoeffs& g)
{
    return 1.0/(g.R*T);
}

struct GasTResult
{
    scalar T;
    int    status;   // 0 converged, 1 negative seed (OpenFOAM: FatalError), 2 maxIter exceeded (FatalError)
};

// thermoI.H:44-88 with F = Es, dFdT = Cv, limit = hConst's identity. The tolerance comes from the SEED.
BRAE_HD inline GasTResult gasTEs(scalar e, scalar p, scalar T0, const GasCoeffs& g)
{
    GasTResult r{T0, 0};
    if (T0 < 0)
    {
        r.status = 1;
        return r;
    }
    scalar Test = T0;
    scalar Tnew = T0;
    const scalar Ttol = T0*1.0e-4;
    int iter = 0;
    do
    {
        Test = Tnew;
        Tnew = Test - (gasEs(p, Test, g) - e)/gasCv(g);
        if (iter++ > 100)
        {
            r.T = Tnew;
            r.status = 2;
            return r;
        }
    } while (fabs(Tnew - Test) > Ttol);
    r.T = Tnew;
    return r;
}

} // namespace rhoCentral
} // namespace brae
