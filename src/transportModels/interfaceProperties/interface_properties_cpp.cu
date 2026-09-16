// interfaceProperties::calculateK and correctContactAngle -- the host reference.
//
// provenance:
//   openfoam: src/transportModels/interfaceProperties/interfaceProperties.C
//               :107-165  calculateK()
//               :40-99    correctContactAngle()
//             src/finiteVolume/finiteVolume/fvc/fvcAverage.C  (average = surfaceSum(magSf*ssf)/surfaceSum(magSf))
//   tests:    tests/test_interface_curvature_cpp.cu
//
// THE ORACLE PROBLEM, AND HOW IT IS ANSWERED HERE. A stock OpenFOAM run never writes K_, so there is
// no stored field to compare against without instrumenting OpenFOAM's own class. But curvature has
// something better than a stored field: it is a GEOMETRIC quantity with known values. A flat interface
// has K = 0 exactly, at any resolution; a sphere of radius R has K = 2/R, approached as the mesh
// refines. Both are checkable here, and together they pin the magnitude, the sign and the absence of
// spurious curvature, which is what the instrumented comparison would have been for.
//
// The contact-angle correction has an exact postcondition of its own: after it,
// acos(nHat & nf) IS theta. That is what the 2x2 solve is for, and it is an identity, not a tolerance.
#include "interface_properties_cpp.cuh"
#include "fvc.cuh"
#include <algorithm>

namespace brae {
namespace cpu {
namespace interfaceProps {

void smoothAlpha(std::vector<scalar>&        alpha,
                 int                         nPasses,
                 const PrimitiveMesh&        m,
                 const FvGeometry&           g,
                 const std::vector<FvPatch>& patches)
{
    // interfaceProperties.C:117-127: alpha1L = fvc::average(fvc::interpolate(alpha1L)), n times.
    //
    // fvc::average is AREA-WEIGHTED -- surfaceSum(magSf*ssf)/surfaceSum(magSf) (fvcAverage.C) -- not a
    // plain mean of the face values. On a uniform mesh the two coincide, which is why the weighting is
    // easy to drop; on a graded or anisotropic mesh they do not, and the difference lands in the
    // interface NORMAL, so it shows up as curvature rather than as an obviously wrong alpha.
    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<scalar>& magSf = g.magSf();

    for (int pass = 0; pass < nPasses; ++pass)
    {
        const SurfaceScalarField f = fvc::interpolate(alpha, m, g, patches);
        std::vector<scalar> num(alpha.size(), scalar(0)), den(alpha.size(), scalar(0));
        for (label fi = 0; fi < nIf; ++fi)
        {
            const scalar a = magSf[fi];
            // surfaceSum adds to BOTH sides with the same sign
            num[own[fi]] += a * f.internal[fi];  den[own[fi]] += a;
            num[nei[fi]] += a * f.internal[fi];  den[nei[fi]] += a;
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            for (label i = 0; i < q.size; ++i)
            {
                const label ci = q.faceCells[i];
                const scalar a = magSf[q.start + i];
                num[ci] += a * f.boundary[pi][i];
                den[ci] += a;
            }
        }
        for (std::size_t c = 0; c < alpha.size(); ++c)
            if (den[c] > scalar(0)) alpha[c] = num[c] / den[c];
    }
}


void correctContactAngle(std::vector<vector>&       nHatp,
                         const std::vector<vector>& nf,
                         const std::vector<scalar>& theta,
                         scalar                     dN)
{
    if (nf.size() != nHatp.size() || theta.size() != nHatp.size())
        throw std::runtime_error(
            "brae interfaceProperties: the contact-angle patch fields differ in length.");

    // interfaceProperties.C:76-96. The interface normal is ROTATED INSIDE THE (nf, nHat) PLANE until
    // its angle to the wall normal is theta. The 2x2 system is the Gram matrix of that plane:
    //
    //     nHat' = a*nf + b*nHat,   with   nHat'&nf = cos(theta)          = b1
    //                                     nHat'&nHat = cos(acos(a12) - theta) = b2
    //
    // whose solution is a = (b1 - a12*b2)/det, b = (b2 - a12*b1)/det with det = 1 - a12^2. THE
    // POSTCONDITION IS THE POINT: after this, acos(nHat' & nf) is theta, and that is what the gate
    // asserts rather than the algebra.
    //
    // det vanishes when the interface normal is already parallel to the wall normal (a12 = +-1), where
    // the plane is undefined; OpenFOAM divides anyway and relies on the field never reaching exactly
    // that. This carries the same arithmetic rather than a guard of its own, because a guard here
    // would silently produce a different normal than OpenFOAM on the faces that hit it.
    for (std::size_t i = 0; i < nHatp.size(); ++i)
    {
        const vector& n = nf[i];
        const scalar a12 = nHatp[i].x*n.x + nHatp[i].y*n.y + nHatp[i].z*n.z;
        const scalar b1  = std::cos(theta[i]);
        const scalar b2  = std::cos(std::acos(a12) - theta[i]);
        const scalar det = scalar(1) - a12*a12;
        const scalar a   = (b1 - a12*b2) / det;
        const scalar b   = (b2 - a12*b1) / det;

        vector v{a*n.x + b*nHatp[i].x, a*n.y + b*nHatp[i].y, a*n.z + b*nHatp[i].z};
        const scalar mv = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        // ...and the SAME deltaN stabiliser as everywhere else, not a bare normalisation.
        const scalar s = scalar(1) / (mv + dN);
        nHatp[i] = vector{v.x*s, v.y*s, v.z*s};
    }
}


std::vector<scalar> contactAngleGradient(const std::vector<vector>& nHatp,
                                         const std::vector<vector>& nf,
                                         const std::vector<vector>& gradAlphaf)
{
    // interfaceProperties.C:97: acap.gradient() = (nf & nHatp)*mag(gradAlphaf[patchi]).
    //
    // THE CORRECTION WRITES BACK INTO alpha1's BOUNDARY. alphaContactAngle is a zeroGradient-family
    // patch whose gradient is SET here, so the contact angle does not only bend the normal used for
    // curvature -- it changes alpha's own wall gradient, and therefore the next gradient of alpha.
    // A port that corrected nHat and stopped would leave the wall wetting itself the same way
    // regardless of theta.
    std::vector<scalar> grad(nHatp.size());
    for (std::size_t i = 0; i < nHatp.size(); ++i)
    {
        const scalar dot = nf[i].x*nHatp[i].x + nf[i].y*nHatp[i].y + nf[i].z*nHatp[i].z;
        const vector& ga = gradAlphaf[i];
        grad[i] = dot * std::sqrt(ga.x*ga.x + ga.y*ga.y + ga.z*ga.z);
    }
    return grad;
}


std::vector<vector> gaussGradFromValues(const std::vector<scalar>&              cells,
                                        const std::vector<std::vector<scalar>>& boundary,
                                        const PrimitiveMesh&                    m,
                                        const FvGeometry&                       g,
                                        const std::vector<FvPatch>&             patches)
{
    // fvc::gaussGrad, written against raw values rather than a GeometricField, because the smoothed
    // field above has no patch objects of its own. The gate asserts this reproduces fvc::gaussGrad
    // EXACTLY on an unsmoothed field, so the two cannot drift: if they ever disagree, the arm fails
    // rather than the smoothing path quietly diverging from the ordinary one.
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<scalar>& w   = g.weights();
    const std::vector<vector>& Sf  = g.Sf();
    const std::vector<scalar>& V   = g.V();

    std::vector<vector> grad(static_cast<std::size_t>(nC), vector{0, 0, 0});
    for (label f = 0; f < nIf; ++f)
    {
        const scalar vf = w[f]*cells[own[f]] + (scalar(1) - w[f])*cells[nei[f]];
        const vector& S = Sf[f];
        grad[own[f]].x += vf*S.x; grad[own[f]].y += vf*S.y; grad[own[f]].z += vf*S.z;
        grad[nei[f]].x -= vf*S.x; grad[nei[f]].y -= vf*S.y; grad[nei[f]].z -= vf*S.z;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        for (label i = 0; i < q.size; ++i)
        {
            const label ci = q.faceCells[i];
            const vector& S = Sf[q.start + i];
            const scalar vf = boundary[pi][i];
            grad[ci].x += vf*S.x; grad[ci].y += vf*S.y; grad[ci].z += vf*S.z;
        }
    }
    for (label c = 0; c < nC; ++c)
    {
        grad[c].x /= V[c]; grad[c].y /= V[c]; grad[c].z /= V[c];
    }
    return grad;
}


void curvature(const SurfaceScalarField&   nHatf,
               const PrimitiveMesh&        m,
               const FvGeometry&           g,
               const std::vector<FvPatch>& patches,
               std::vector<scalar>&        K)
{
    // interfaceProperties.C:152: K_ = -fvc::div(nHatf_). The MINUS is the whole sign convention:
    // gradAlpha points towards the phase-1 side, so nHat points into phase 1, and -div of it is
    // positive for a phase-1 drop and negative for a phase-1 bubble. Dropping the minus inverts every
    // surface-tension force in the solver while leaving its magnitude right.
    K = fvc::div(nHatf, m, g, patches);
    for (scalar& v : K) v = -v;
}


void calculateK(const GeometricField<scalar>& alpha1,
                const InterfaceCoeffs&        c,
                const PrimitiveMesh&          m,
                const FvGeometry&             g,
                const std::vector<FvPatch>&   patches,
                bool                          gradLeastSquares,
                SurfaceScalarField&           nHatf,
                std::vector<scalar>&          K)
{
    const scalar dN = deltaN(g.V());

    // 1. the cell gradient, optionally smoothed first. `fvc::grad(alpha1, "nHat")` looks up the
    //    gradSchemes entry NAMED nHat -- not grad(alpha.water) and not default. 43 of the 44 shipped
    //    tutorials say `default Gauss linear` and name no nHat entry, so they fall to that; the
    //    caller resolves which, and passing the wrong one changes the interface normal.
    std::vector<vector> gradAlpha;
    if (c.nAlphaSmoothCurvature < 1)
    {
        gradAlpha = gradLeastSquares ? fvc::leastSquaresGrad(alpha1, m, g, patches)
                                     : fvc::gaussGrad(alpha1, m, g, patches);
    }
    else
    {
        // OpenFOAM smooths a COPY of alpha1 -- boundary conditions and all -- and takes the gradient
        // of THAT (interfaceProperties.C:117-130). The gradient must therefore be built from the
        // smoothed values, which is what this branch exists to do; taking grad(alpha1) after smoothing
        // a local copy would make the smoothing a no-op that still costs its passes.
        if (gradLeastSquares)
            throw std::runtime_error(
                "brae interfaceProperties: nAlphaSmoothCurvature with a leastSquares gradient is not "
                "ported. The smoothed field needs its own least-squares stencil evaluation, and NO "
                "shipped OpenFOAM tutorial sets nAlphaSmoothCurvature at all, so there is no case to "
                "validate the combination against.");

        std::vector<scalar> a = alpha1.internal;
        smoothAlpha(a, c.nAlphaSmoothCurvature, m, g, patches);

        // fvc::average calls correctBoundaryConditions, so the smoothed field's boundary is
        // re-evaluated from its own internal field. zeroGradient patches therefore follow the
        // smoothed cell value; a patch that FIXES a value keeps it.
        std::vector<std::vector<scalar>> ab(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            ab[pi].resize(static_cast<std::size_t>(q.size));
            const bool fixes = alpha1.boundary[pi]->fixesValue();
            const std::vector<scalar>& fixed = alpha1.boundary[pi]->value();
            for (label i = 0; i < q.size; ++i)
                ab[pi][i] = fixes ? fixed[i] : a[q.faceCells[i]];
        }
        gradAlpha = gaussGradFromValues(a, ab, m, g, patches);
    }

    // 2. interpolate the cell gradient to faces, component by component.
    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<scalar>& w   = g.weights();

    std::vector<vector> gradAlphaf(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const vector& go = gradAlpha[own[f]];
        const vector& gn = gradAlpha[nei[f]];
        gradAlphaf[f] = vector{w[f]*go.x + (scalar(1) - w[f])*gn.x,
                               w[f]*go.y + (scalar(1) - w[f])*gn.y,
                               w[f]*go.z + (scalar(1) - w[f])*gn.z};
    }

    // 3. nHatfv = gradAlphaf/(mag + deltaN), on the internal faces.
    std::vector<vector> nHatfv;
    faceUnitNormal(gradAlphaf, dN, nHatfv);

    // 4. nHatf = nHatfv & Sf.
    const std::vector<vector>& Sf = g.Sf();
    nHatf.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
        nHatf.internal[f] = nHatfv[f].x*Sf[f].x + nHatfv[f].y*Sf[f].y + nHatfv[f].z*Sf[f].z;

    // ...and the boundary, where the contact-angle correction lives. Patches that are not
    // alphaContactAngle take the face cell's gradient unchanged (fvc::interpolate at an uncoupled
    // patch is the patch value, and the gradient's patch value is the cell's).
    nHatf.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        std::vector<vector> gb(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i) gb[i] = gradAlpha[q.faceCells[i]];

        std::vector<vector> nb;
        faceUnitNormal(gb, dN, nb);
        if (!c.contactAngleDeg.empty() && pi < c.contactAngleDeg.size()
            && c.contactAngleDeg[pi] >= scalar(0))
        {
            const std::vector<scalar> th(static_cast<std::size_t>(q.size),
                                         c.contactAngleDeg[pi] * scalar(M_PI) / scalar(180));
            correctContactAngle(nb, q.nf, th, dN);
        }
        nHatf.boundary[pi].resize(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i)
        {
            const vector& S = Sf[q.start + i];
            nHatf.boundary[pi][i] = nb[i].x*S.x + nb[i].y*S.y + nb[i].z*S.z;
        }
    }

    // 5. K = -div(nHatf)
    curvature(nHatf, m, g, patches, K);
}

}   // namespace interfaceProps
}   // namespace cpu
}   // namespace brae
