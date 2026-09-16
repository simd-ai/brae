#pragma once
// Interface curvature and surface tension -- the host reference.
//
// provenance:
//   openfoam:  src/transportModels/interfaceProperties/interfaceProperties.C
//                :108-140  calculateK()
//                :170-196  the constructor, where deltaN_ and the two coefficients come from
//                :~165     surfaceTensionForce()
//   cuda:      src/transportModels/interfaceProperties/device_interface_properties.cu  (not written yet)
//
// calculateK(), step by step, because every line of it is a place a VoF port goes quietly wrong:
//
//   1.  gradAlpha  = fvc::grad(alpha1, "nHat")     <- a NAMED gradScheme, not the default
//   2.  gradAlphaf = fvc::interpolate(gradAlpha)   <- cell gradient interpolated to faces
//   3.  nHatfv     = gradAlphaf/(mag(gradAlphaf) + deltaN)
//   4.  correctContactAngle(nHatfv.boundary, gradAlphaf.boundary)
//   5.  nHatf      = nHatfv & Sf
//   6.  K          = -fvc::div(nHatf)
//
// THREE THINGS WORTH NAMING:
//
// * `fvc::grad(alpha1, "nHat")` looks up gradSchemes entry **nHat**, not `grad(alpha1)` and not
//   `default`. A case that sets one and not the other gets a different curvature, and nothing says so.
//
// * deltaN = 1e-8/cbrt(average(mesh.V())) (interfaceProperties.C:194). It is a MESH-DEPENDENT
//   stabiliser, not a constant: it keeps nHat finite where gradAlpha vanishes, which is everywhere away
//   from the interface -- i.e. most of the domain. Hard-coding 1e-8 gives the wrong normal on any mesh
//   whose cells are not of order unit volume.
//
// * cAlpha has NO DEFAULT in OpenFOAM: `solverDict(alpha1.name()).get<scalar>("cAlpha")` throws if the
//   case omits it (:184-187). nAlphaSmoothCurvature DOES default, to 0 (:179-182). Giving cAlpha a
//   default here would run a case OpenFOAM refuses, with an interface compression the user never chose.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interfaceProps {

// fvSolution's solvers/alpha1 sub-dictionary, plus sigma from constant/transportProperties.
struct InterfaceCoeffs
{
    scalar cAlpha = 0;                 // interface compression; MANDATORY, no default
    int    nAlphaSmoothCurvature = 0;  // curvature smoothing passes; defaults to 0
    scalar sigma  = 0;                 // surface tension
};

// solverDict(alpha1.name()) is the fvSolution `solvers` entry for the alpha field -- damBreak names it
// `alpha.water`, so the phase name decides the key and a hard-coded "alpha1" would miss it.
inline InterfaceCoeffs readInterfaceCoeffs(const FoamDict& fvSolution,
                                           const std::string& alphaFieldName,
                                           scalar sigma)
{
    const FoamDict* solvers = fvSolution.subDict("solvers");
    const FoamDict* sd = solvers ? solvers->subDict(alphaFieldName) : nullptr;
    if (!sd)
        throw std::runtime_error(
            "brae interFoam: fvSolution has no `solvers/" + alphaFieldName + "` entry. OpenFOAM reads "
            "cAlpha from there (interfaceProperties.C:184), so the interface compression is undefined.");
    InterfaceCoeffs c;
    c.cAlpha = sd->scalarOr("cAlpha", scalar(-1));
    if (c.cAlpha < 0)
        throw std::runtime_error(
            "brae interFoam: `cAlpha` is missing from solvers/" + alphaFieldName + ". OpenFOAM has NO "
            "default for it (get<scalar>, interfaceProperties.C:184-187) and FatalErrors; defaulting it "
            "here would run the case with an interface compression nobody asked for.");
    c.nAlphaSmoothCurvature = static_cast<int>(sd->scalarOr("nAlphaSmoothCurvature", scalar(0)));
    c.sigma = sigma;
    return c;
}

// deltaN = 1e-8/cbrt(average(V)) -- interfaceProperties.C:194. Mesh-dependent on purpose.
inline scalar deltaN(const std::vector<scalar>& cellVolumes)
{
    if (cellVolumes.empty()) return scalar(0);
    scalar sum = 0;
    for (scalar v : cellVolumes) sum += v;
    const scalar avg = sum / static_cast<scalar>(cellVolumes.size());
    return scalar(1e-8) / std::cbrt(avg);
}

// nHatfv = gradAlphaf/(mag(gradAlphaf) + deltaN), per face (interfaceProperties.C:141).
// Returned rather than applied so the dot with Sf, which is the next line in OpenFOAM, stays visible.
inline void faceUnitNormal(const std::vector<vector>& gradAlphaf,
                           scalar dN,
                           std::vector<vector>& nHatfv)
{
    nHatfv.resize(gradAlphaf.size());
    for (std::size_t f = 0; f < gradAlphaf.size(); ++f)
    {
        const vector& g = gradAlphaf[f];
        const scalar m = std::sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
        const scalar s = scalar(1) / (m + dN);
        nHatfv[f] = vector{g.x * s, g.y * s, g.z * s};
    }
}

// nHatf = nHatfv & Sf (interfaceProperties.C:150).
inline void faceNormalFlux(const std::vector<vector>& nHatfv,
                           const std::vector<vector>& Sf,
                           std::vector<scalar>& nHatf)
{
    nHatf.resize(nHatfv.size());
    for (std::size_t f = 0; f < nHatfv.size(); ++f)
        nHatf[f] = nHatfv[f].x*Sf[f].x + nHatfv[f].y*Sf[f].y + nHatfv[f].z*Sf[f].z;
}

// sigmaK = sigma*K. surfaceTensionForce() is interpolate(sigmaK)*snGrad(alpha1) -- note the
// interpolation is of the PRODUCT, which matters the moment sigma stops being constant.
inline void sigmaK(const std::vector<scalar>& K, scalar sigma, std::vector<scalar>& out)
{
    out.resize(K.size());
    for (std::size_t i = 0; i < K.size(); ++i) out[i] = sigma * K[i];
}

}   // namespace interfaceProps
}   // namespace cpu
}   // namespace brae
