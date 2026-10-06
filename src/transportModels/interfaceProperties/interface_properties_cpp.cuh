#pragma once
// Interface curvature and surface tension -- the host reference.
//
// provenance:
//   openfoam:  src/transportModels/interfaceProperties/interfaceProperties.C
//                :108-140  calculateK()
//                :170-196  the constructor, where deltaN_ and the two coefficients come from
//                :~165     surfaceTensionForce()
//   cuda:      src/transportModels/interfaceProperties/device_interface_properties.cu  (not written yet)
//   tests:    tests/test_interface_properties.cu, tests/test_interface_curvature_cpp.cu and
//             tests/interfoam_curvature_vs_openfoam.sh (nHat, curvature and the surface-tension
//             force against OpenFOAM); tests/test_device_interface_properties.cu holds the device twin.
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
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "grad_choice.cuh"
#include <cmath>
#include <cstdlib>
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
    // theta0 per patch, in DEGREES, or < 0 where the patch is not an alphaContactAngle. Only one of
    // the 44 shipped tutorials sets one (laminar/capillaryRise), and that is the case where surface
    // tension IS the answer -- so it is the one a missing contact angle cannot hide in.
    std::vector<scalar> contactAngleDeg;
    // gradSchemes' `nHat` entry (interfaceProperties.C:96, fvc::grad(alpha1_, "nHat")), then `default`
    GradChoice nHatGrad;
    // deltaN = 1e-8/cbrt(average(mesh.V())), THE CONSTRUCTOR'S VALUE and never recomputed
    // (interfaceProperties.C:190-195: `deltaN_` is a dimensionedScalar member initialised in the
    // member-initialiser list). On a mesh that does not move, or one that moves rigidly, recomputing
    // it per call gives the same number. ON A MESH THAT DEFORMS IT DOES NOT, and brae recomputed it:
    // measured on waves/waveMakerSolitary, nHatf and K were 8.5e-09 from OpenFOAM's after the FIRST
    // mesh update, which a curvature fixed point then carries forward. 0 means "never set", which
    // calculateK refuses rather than running without the stabiliser.
    scalar deltaN = 0;
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

// deltaN = 1e-8/cbrt(average(V)) -- interfaceProperties.C:194. CALLED ONCE, where OpenFOAM's
// constructor calls it, and the result kept in InterfaceCoeffs::deltaN; see the note there for what
// recomputing it costs on a mesh that deforms.
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

// --------------------------------------------------------------------------------------------------
// calculateK and its parts. See interface_properties_cpp.cu for the provenance and for how curvature
// is gated without an instrumented OpenFOAM: it is a GEOMETRIC quantity, so a flat interface has K = 0
// exactly at any resolution and a sphere of radius R has K -> 2/R as the mesh refines.

// alpha1L = fvc::average(fvc::interpolate(alpha1L)), nPasses times. fvc::average is AREA-WEIGHTED.
void smoothAlpha(std::vector<scalar>&        alpha,
                 int                         nPasses,
                 const PrimitiveMesh&        m,
                 const FvGeometry&           g,
                 const std::vector<FvPatch>& patches);

// The contact-angle rotation. POSTCONDITION: acos(nHatp & nf) == theta.
void correctContactAngle(std::vector<vector>&       nHatp,
                         const std::vector<vector>& nf,
                         const std::vector<scalar>& theta,       // RADIANS
                         scalar                     dN);

// acap.gradient() = (nf & nHatp)*mag(gradAlphaf) -- the correction writes back into alpha's own wall
// gradient, not only into the normal used for curvature.
std::vector<scalar> contactAngleGradient(const std::vector<vector>& nHatp,
                                         const std::vector<vector>& nf,
                                         const std::vector<vector>& gradAlphaf);

// fvc::gaussGrad written against raw values, for the smoothed field which has no patch objects of its
// own. Gated to reproduce fvc::gaussGrad exactly on an unsmoothed field, so the two cannot drift.
std::vector<vector> gaussGradFromValues(const std::vector<scalar>&              cells,
                                        const std::vector<std::vector<scalar>>& boundary,
                                        const PrimitiveMesh&                    m,
                                        const FvGeometry&                       g,
                                        const std::vector<FvPatch>&             patches);

// K = -fvc::div(nHatf). The MINUS is the sign convention for the whole solver.
void curvature(const SurfaceScalarField&   nHatf,
               const PrimitiveMesh&        m,
               const FvGeometry&           g,
               const std::vector<FvPatch>& patches,
               std::vector<scalar>&        K);

// interfaceProperties.C:107-165, end to end.
void calculateK(const GeometricField<scalar>& alpha1,
                const InterfaceCoeffs&        c,
                const PrimitiveMesh&          m,
                const FvGeometry&             g,
                const std::vector<FvPatch>&   patches,
                bool                          gradLeastSquares,
                SurfaceScalarField&           nHatf,
                std::vector<scalar>&          K);

// THE BOUNDARY HALF OF calculateK ALONE, for a caller that needs nHatf on the patches (and the contact
// angle's write-back of alpha's wall gradient) but not the curvature: the cell gradient is taken at the
// patches' face cells only. MEASURED on RAS/DTCHull (845,536 cells): calculateK 37 ms a call, called four
// times a step by the device loop's alpha hooks for the boundary normal alone. The same arithmetic in the same
// order as calculateK's boundary where it applies -- a plain Gauss linear nHat gradient of the unsmoothed field
// on a mesh with no coupled patch -- and refused by name elsewhere; the compiler may still fuse a multiply-add
// differently in the two loops (one ulp in a few cells on capillaryRise).
struct NHatBoundaryStencil
{
    bool usable = false;
    std::vector<label> cells;     // the patches' face cells, ascending
    std::vector<label> slot;      // per mesh cell: its place in `cells`, -1 when it is not one
    std::vector<label> start;     // per entry of `cells`: its internal faces, ascending
    std::vector<label> faces;
    std::vector<vector> gradAlpha;   // full size; only `cells` are ever written or read
    // `cells` and every internal face that touches one, for a least-squares `nHat` taken at them alone
    // (calculateNHatBoundaryOfSubsetGradient, which finds it at its first call); its arrays are kept between
    // calls
    fvc::GradSubset subset;
    // AN `empty` PATCH LEFT OUT: its cells are not in `cells` unless another patch puts them there, and
    // finishNHatBoundary writes ZERO on its faces. emptyFvPatch::size() is 0 in OpenFOAM, so there is no normal
    // there to form; brae keeps the faces in its addressing and every reader skips them (deviceDiv, by
    // bndIsEmpty) or multiplies them by a flux that is zero there. On a 2-D mesh EVERY cell touches the empty
    // patches, and the stencil was the whole mesh: MEASURED on waveMakerPiston refined to 896,000 cells, the
    // alpha hooks' boundary normal 71 ms a step with them in.
    bool skipEmpty = false;
};
NHatBoundaryStencil nHatBoundaryStencil(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    bool skipEmpty = false);
bool nHatBoundaryOnlyApplies(
    const InterfaceCoeffs& c,
    bool gradLeastSquares,
    const NHatBoundaryStencil& st);
// calculateNHatBoundary's second half, for a caller that forms the first itself: `acc` holds each stencil cell's
// sum over its internal faces -- Sf*interpolate(alpha), added on the owner side and subtracted on the
// neighbour's, in ascending face order -- and this adds the patches' terms in patch order, divides by the
// volume and runs nHatBoundary. The device loop's alpha hooks form `acc` on the GPU.
void finishNHatBoundary(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    NHatBoundaryStencil& st,
    std::vector<vector>& acc,
    SurfaceScalarField& nHatf);
// THE BOUNDARY NORMAL WHERE THE BOUNDARY-ONLY FORM DOES NOT APPLY (a leastSquares or cellLimited `nHat`, or a
// stencil a coupled pair leaves unusable): calculateK's own cell gradient over the whole mesh and its boundary
// half, and nothing else of it -- no gradient at the faces, no internal faces' normal, no curvature, which is
// what a caller that wants the patches' normal alone throws away. Bit for bit calculateK's nHatf.boundary, the
// contact angle's write-back included; nHatf.internal comes back empty.
void calculateNHatBoundaryOfWholeGradient(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    SurfaceScalarField& nHatf);
// ...AND FROM A LEAST-SQUARES GRADIENT TAKEN AT THE STENCIL'S CELLS ALONE, with its cell limiter where the case
// names one. A least-squares fit and its limiter are local to a cell (fvc::GradSubset), so the patches' face
// cells need no other cell's gradient: fvc::leastSquaresGradAt and cpu::cellLimitGradAt run the whole-mesh
// functions' own loops over the stencil's faces and cells, and nHatBoundary reads the result there. Bit for bit
// calculateK's nHatf.boundary on every patch the stencil holds (an `empty` one it leaves out takes zeros, as
// in the Gauss form). nHatBoundaryOfSubsetApplies says when: an unsmoothed leastSquares `nHat` on a usable
// stencil.
bool nHatBoundaryOfSubsetApplies(
    const InterfaceCoeffs& c,
    const NHatBoundaryStencil& st);
void calculateNHatBoundaryOfSubsetGradient(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    NHatBoundaryStencil& st,
    SurfaceScalarField& nHatf);
void calculateNHatBoundary(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    NHatBoundaryStencil& st,
    SurfaceScalarField& nHatf);

}   // namespace interfaceProps
}   // namespace cpu
}   // namespace brae
