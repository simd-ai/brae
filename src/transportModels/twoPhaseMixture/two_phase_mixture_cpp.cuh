#pragma once
// The two-phase mixture's blended properties -- the host reference.
//
// provenance:
//   openfoam:  src/transportModels/twoPhaseMixture/twoPhaseMixture/twoPhaseMixture.C  (alpha1, alpha2)
//              src/transportModels/incompressible/incompressibleTwoPhaseMixture/
//                  incompressibleTwoPhaseMixture.C:43-56 (calcNu), :126-141 (mu)
//              applications/solvers/multiphase/interFoam/createFields.H:44-54 (rho)
//   cuda:      src/transportModels/twoPhaseMixture/device_two_phase_mixture.cu   (not written yet)
//
// RHO AND MU DO NOT CLAMP THE SAME WAY, and that asymmetry is the whole reason this file is separate
// from a one-line blend:
//
//     rho = alpha1*rho1 + alpha2*rho2                       createFields.H:54   -- RAW alpha
//     mu  = limitedAlpha1*rho1*nu1 + (1-limitedAlpha1)*rho2*nu2                 -- CLAMPED alpha
//     nu  = mu/(limitedAlpha1*rho1 + (1-limitedAlpha1)*rho2)                    -- CLAMPED alpha
//
// with limitedAlpha1 = clamp(alpha1, 0, 1). MULES keeps alpha1 in [0,1] to round-off, so on a converged
// field the two agree -- which is exactly why clamping rho as well would pass every smooth test and then
// differ on the overshoot MULES is allowed to leave. OpenFOAM does not clamp rho; neither does this.
//
// alpha2 is a FIELD, not `1 - alpha1` computed at the point of use (twoPhaseMixture.H holds both), so a
// port that substitutes 1-alpha1 into rho is making an assumption OpenFOAM does not: the two agree only
// while alpha2 is maintained, and interFoam's alphaEqn is what maintains it.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace twoPhase {

// The mixture's two constant phases, read from constant/transportProperties.
struct PhaseProperties
{
    scalar rho1 = 0;    // phase 1 density
    scalar rho2 = 0;    // phase 2 density
    scalar nu1  = 0;    // phase 1 kinematic viscosity (Newtonian; a viscosityModel comes later)
    scalar nu2  = 0;
};

// constant/transportProperties, as interFoam's mixture reads it:
//     phases (water air);
//     water { transportModel Newtonian; nu 1e-06; rho 1000; }
//     air   { transportModel Newtonian; nu 1.48e-05; rho 1; }
//     sigma 0.07;
// PHASE ORDER IS THE `phases` ENTRY'S ORDER, not alphabetical and not the order the sub-dicts appear:
// phase1 is alpha1's phase, and getting the two the wrong way round inverts the whole field while still
// running. damBreak reads (water air), so alpha1 is WATER.
//
// Only `Newtonian` is accepted. The others (BirdCarreau, CrossPowerLaw, ...) are real viscosityModels
// with their own coefficients; running one as Newtonian would silently use a single nu for a
// shear-thinning fluid, so it is refused by name.
struct MixtureSpec
{
    std::string   phase1Name, phase2Name;
    PhaseProperties phases;
    scalar        sigma = 0;      // surface tension, read by interfaceProperties
};

inline MixtureSpec readTransportProperties(const std::string& casePath)
{
    const FoamDict d = readDict(casePath + "/constant/transportProperties");
    const std::vector<std::string> names = d.wordListOr("phases", {});
    if (names.size() != 2)
        throw std::runtime_error(
            "brae interFoam: constant/transportProperties needs `phases (<phase1> <phase2>);` with "
            "exactly two names; alpha1 belongs to the FIRST.");
    MixtureSpec m;
    m.phase1Name = names[0];
    m.phase2Name = names[1];
    for (int i = 0; i < 2; ++i)
    {
        const std::string& n = (i == 0) ? m.phase1Name : m.phase2Name;
        const FoamDict* sd = d.subDict(n);
        if (!sd)
            throw std::runtime_error("brae interFoam: constant/transportProperties names phase '" + n +
                                     "' in `phases` but has no `" + n + " { ... }` sub-dictionary.");
        const std::string model = sd->wordOr("transportModel", "");
        if (model != "Newtonian")
            throw std::runtime_error(
                "brae interFoam: phase '" + n + "' asks for transportModel '" + model + "'; brae has "
                "`Newtonian` only. A non-Newtonian model carries its own coefficients and running it "
                "as Newtonian would use one nu for a shear-dependent fluid.");
        const scalar rho = sd->scalarOr("rho", -1), nu = sd->scalarOr("nu", -1);
        if (rho <= 0 || nu <= 0)
            throw std::runtime_error("brae interFoam: phase '" + n + "' needs positive `rho` and `nu`.");
        if (i == 0) { m.phases.rho1 = rho; m.phases.nu1 = nu; }
        else        { m.phases.rho2 = rho; m.phases.nu2 = nu; }
    }
    // surfaceTensionModel::New (surfaceTensionModelNew.C:36-66): a DICTIONARY named sigma selects a
    // model by its `type`; otherwise sigma is a dimensionedScalar read with no default
    // (constantSurfaceTension.C:53). This read it with scalarOr("sigma", 0), so a temperature-dependent
    // model AND a missing entry both became ZERO SURFACE TENSION, silently -- on capillaryRise that is
    // the whole answer.
    if (const FoamDict* sd = d.subDict("sigma"))
    {
        throw std::runtime_error(
            "brae interFoam: `sigma` is a dictionary selecting surfaceTensionModel `"
            + sd->wordOr("type", "?") + "`; brae has the constant model only.");
    }
    const scalar kAbsent = scalar(-1.0e300);
    m.sigma = d.scalarOr("sigma", kAbsent);
    if (m.sigma == kAbsent)
    {
        throw std::runtime_error(
            "brae interFoam: constant/transportProperties has no `sigma`. OpenFOAM reads it with no "
            "default and stops; zero is a value a case has to ask for.");
    }
    return m;
}

inline scalar limitedAlpha(scalar a)
{
    return std::min(std::max(a, scalar(0)), scalar(1));   // OF clamp(alpha1, zero_one{})
}

// rho = alpha1*rho1 + alpha2*rho2, on the RAW fields (createFields.H:54).
inline void mixtureRho(const std::vector<scalar>& alpha1,
                       const std::vector<scalar>& alpha2,
                       const PhaseProperties& p,
                       std::vector<scalar>& rho)
{
    rho.resize(alpha1.size());
    for (std::size_t i = 0; i < alpha1.size(); ++i)
        rho[i] = alpha1[i] * p.rho1 + alpha2[i] * p.rho2;
}

// mu = limitedAlpha1*rho1*nu1 + (1 - limitedAlpha1)*rho2*nu2 (incompressibleTwoPhaseMixture.C:137-138).
inline void mixtureMu(const std::vector<scalar>& alpha1,
                      const PhaseProperties& p,
                      std::vector<scalar>& mu)
{
    mu.resize(alpha1.size());
    for (std::size_t i = 0; i < alpha1.size(); ++i)
    {
        const scalar a = limitedAlpha(alpha1[i]);
        mu[i] = a * p.rho1 * p.nu1 + (scalar(1) - a) * p.rho2 * p.nu2;
    }
}

// nu = mu / (limitedAlpha1*rho1 + (1 - limitedAlpha1)*rho2) (incompressibleTwoPhaseMixture.C:55).
// NOTE the denominator is the CLAMPED blend, not the rho field above -- they differ wherever alpha1
// leaves [0,1], and OpenFOAM uses the clamped one here and the raw one in the momentum ddt.
inline void mixtureNu(const std::vector<scalar>& alpha1,
                      const std::vector<scalar>& mu,
                      const PhaseProperties& p,
                      std::vector<scalar>& nu)
{
    nu.resize(alpha1.size());
    for (std::size_t i = 0; i < alpha1.size(); ++i)
    {
        const scalar a = limitedAlpha(alpha1[i]);
        nu[i] = mu[i] / (a * p.rho1 + (scalar(1) - a) * p.rho2);
    }
}

}   // namespace twoPhase
}   // namespace cpu
}   // namespace brae
