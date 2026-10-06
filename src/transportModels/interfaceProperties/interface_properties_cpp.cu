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
#include <cstdlib>

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


// calculateK's BOUNDARY HALF: nHatf on every patch, and the contact angle's write-back of alpha's wall
// gradient, from the cell gradient at the patches' face cells (and, on a coupled patch, at the cells on its
// other side). Shared by calculateK and calculateNHatBoundary, so the two cannot drift apart.
void nHatBoundary(
    const GeometricField<scalar>& alpha1,
    scalar dN,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const std::vector<vector>& gradAlpha,
    SurfaceScalarField& nHatf,
    bool skipEmpty = false)
{
    const std::vector<vector>& Sf = g.Sf();
    // Patches that are not alphaContactAngle take the face cell's gradient unchanged (fvc::interpolate at
    // an uncoupled patch is the patch value, and the gradient's patch value is the cell's).
    nHatf.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        // an empty patch has no faces in OpenFOAM; a caller whose readers skip them takes zeros and no work
        // (NHatBoundaryStencil::skipEmpty)
        if (skipEmpty && q.type == "empty")
        {
            nHatf.boundary[pi].assign(static_cast<std::size_t>(q.size), scalar(0));
            continue;
        }

        // gradAlphaf's BOUNDARY VALUE is not the owner cell's gradient. fvc::grad runs
        // gaussGrad::correctBoundaryConditions, which replaces the WALL-NORMAL COMPONENT with the
        // patch's own snGrad (gaussGrad.C):
        //
        //     gGrad_b += n*(vsf_b.snGrad() - (n & gGrad_b))
        //
        // and fvc::interpolate at an uncoupled patch then returns that value unchanged. This mattered
        // more than anywhere else it could have: on a contact-angle patch alpha's snGrad is exactly
        // what the contact angle SETS, so taking the raw cell gradient here discards the boundary
        // condition's own contribution to the normal the curvature is built from. Measured on
        // capillaryRise, where the contact angle carries the whole case.
        std::vector<vector> gb(static_cast<std::size_t>(q.size));
        const std::vector<scalar> snA = alpha1.boundary[pi]->snGrad(alpha1.internal);
        for (label i = 0; i < q.size; ++i)
        {
            if (q.coupled)
            {
                // ...and at a COUPLED patch it returns the two cells' gradients interpolated, as on an
                // internal face; gaussGrad::correctBoundaryConditions leaves a coupled patch alone
                gb[i] = coupledLinear(q, i, gradAlpha);
                continue;
            }
            const vector& n  = q.nf[i];
            // on a wedge grad(alpha)'s own patch value is the cell gradient ROTATED onto the plane,
            // faceT & grad -- see fvc::gradUBoundary
            const tensor* faceT = alpha1.boundary[pi]->wedgeFaceT();
            const vector& gcell = gradAlpha[q.faceCells[i]];
            const vector  gc = faceT ? vector{faceT->xx*gcell.x + faceT->xy*gcell.y + faceT->xz*gcell.z,
                                              faceT->yx*gcell.x + faceT->yy*gcell.y + faceT->yz*gcell.z,
                                              faceT->zx*gcell.x + faceT->zy*gcell.y + faceT->zz*gcell.z}
                                     : gcell;
            const scalar  nn = n.x*gc.x + n.y*gc.y + n.z*gc.z;
            const scalar  d  = snA[i] - nn;
            gb[i] = vector{gc.x + n.x*d, gc.y + n.y*d, gc.z + n.z*d};
        }

        std::vector<vector> nb;
        faceUnitNormal(gb, dN, nb);

        // correctContactAngle, on every patch that IS one. The patch itself carries theta0 -- brae's
        // ConstantAlphaContactAnglePatchField returns it from contactAngleTheta0() -- so the dispatch
        // is the same question OpenFOAM asks (isA<alphaContactAngleTwoPhaseFvPatchScalarField>) and
        // there is no parallel list to keep in step with the boundary conditions.
        // BRAE_NO_CONTACT_ANGLE is the gate's CONTROL, not a user switch: capillaryRise's whole
        // motion comes from this correction, and tests/test_inter_capillary_vs_openfoam.cu turns it
        // off to show that -- without it brae's velocity is 200x too small, not 10% off.
        const scalar theta0 = std::getenv("BRAE_NO_CONTACT_ANGLE")
                            ? scalar(-1) : alpha1.boundary[pi]->contactAngleTheta0();
        if (theta0 >= scalar(0))
        {
            const std::vector<scalar> th(static_cast<std::size_t>(q.size),
                                         theta0 * scalar(M_PI) / scalar(180));
            correctContactAngle(nb, q.nf, th, dN);

            // ...AND THE PATCH'S OWN GRADIENT IS WRITTEN BACK (interfaceProperties.C:97):
            //     acap.gradient() = (nf & nHatp)*mag(gradAlphaf[patchi]);  acap.evaluate();
            // The contact angle does not only bend the normal the curvature is built from -- it sets
            // alpha's WALL GRADIENT, and therefore the next gradient of alpha, and therefore where the
            // interface meets the wall at all. A port that corrected nHat and stopped would wet the
            // wall identically whatever theta0 said.
            const std::vector<scalar> gr = contactAngleGradient(nb, q.nf, gb);
            auto* fg = dynamic_cast<FixedGradientPatchField<scalar>*>(
                const_cast<fvPatchField<scalar>*>(alpha1.boundary[pi].get()));
            if (!fg)
                throw std::runtime_error(
                    "brae interfaceProperties: patch '" + q.name + "' reports a contact angle but is "
                    "not a fixedGradient patch, so its wall gradient cannot be set. OpenFOAM's "
                    "alphaContactAngle derives from fixedGradient for exactly this reason.");
            fg->setGradient(gr);
            const_cast<GeometricField<scalar>&>(alpha1).boundary[pi]->evaluate(alpha1.internal);
        }
        nHatf.boundary[pi].resize(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i)
        {
            const vector& S = Sf[q.start + i];
            nHatf.boundary[pi][i] = nb[i].x*S.x + nb[i].y*S.y + nb[i].z*S.z;
        }
    }

}

namespace {

// calculateK's step 1, the cell gradient the normal is formed from. ONE function for calculateK and for
// calculateNHatBoundaryOfWholeGradient, so the boundary normal the second hands back cannot be another
// gradient's than the first's.
std::vector<vector> nHatCellGradient(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    bool gradLeastSquares)
{
    // `fvc::grad(alpha1, "nHat")` looks up the gradSchemes entry NAMED nHat -- not grad(alpha.water) and
    // not default. 43 of the 44 shipped tutorials say `default Gauss linear` and name no nHat entry, so
    // they fall to that; the caller resolves which, and passing the wrong one changes the interface normal.
    // the caller's flag, or the case's own `nHat` entry (InterfaceCoeffs::nHatGrad)
    GradChoice nHatGrad = c.nHatGrad;
    nHatGrad.leastSquares = nHatGrad.leastSquares || gradLeastSquares;
    std::vector<vector> gradAlpha;
    if (c.nAlphaSmoothCurvature < 1)
    {
        gradAlpha = gradOf(alpha1, nHatGrad, m, g, patches);
    }
    else
    {
        // OpenFOAM smooths a COPY of alpha1 -- boundary conditions and all -- and takes the gradient
        // of THAT (interfaceProperties.C:117-130). The gradient must therefore be built from the
        // smoothed values, which is what this branch exists to do; taking grad(alpha1) after smoothing
        // a local copy would make the smoothing a no-op that still costs its passes.
        if (!nHatGrad.gaussLinear())
            throw std::runtime_error(
                "brae interfaceProperties: nAlphaSmoothCurvature with a leastSquares or limited gradient is not "
                "ported. The smoothed field needs its own least-squares stencil evaluation, and NO "
                "shipped OpenFOAM tutorial sets nAlphaSmoothCurvature at all, so there is no case to "
                "validate the combination against.");

        for (const FvPatch& q : patches)
        {
            if (q.coupled)
                throw std::runtime_error(
                    "brae interfaceProperties: nAlphaSmoothCurvature across the coupled patch '" + q.name
                    + "' is not ported.");
        }
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
    return gradAlpha;
}

} // namespace

void calculateK(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    bool gradLeastSquares,
    SurfaceScalarField& nHatf,
    std::vector<scalar>& K)
{
    // THE CONSTRUCTOR'S deltaN, not this mesh's: OpenFOAM's is a member set once and a deforming
    // mesh does not move it (interfaceProperties.C:190-195, and the note on InterfaceCoeffs::deltaN)
    if (!(c.deltaN > scalar(0)))
    {
        throw std::runtime_error(
            "brae interfaceProperties::calculateK: InterfaceCoeffs::deltaN is unset. It is "
            "1e-8/cbrt(average(mesh.V())) AT CONSTRUCTION -- the caller has to take it once, from the "
            "mesh as it stands then, because on a mesh that deforms recomputing it is a different "
            "number from OpenFOAM's.");
    }
    const scalar dN = c.deltaN;

    // 1. the cell gradient, optionally smoothed first (nHatCellGradient)
    const std::vector<vector> gradAlpha = nHatCellGradient(alpha1, c, m, g, patches, gradLeastSquares);

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

    // ...and the boundary, where the contact-angle correction lives (nHatBoundary)
    nHatBoundary(alpha1, dN, g, patches, gradAlpha, nHatf);

    // 5. K = -div(nHatf)
    curvature(nHatf, m, g, patches, K);
}

NHatBoundaryStencil nHatBoundaryStencil(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    bool skipEmpty)
{
    NHatBoundaryStencil st;
    st.skipEmpty = skipEmpty;
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<char> wanted(static_cast<std::size_t>(nC), 0);
    for (const FvPatch& q : patches)
    {
        if (skipEmpty && q.type == "empty") continue;
        for (label i = 0; i < q.size; ++i) wanted[static_cast<std::size_t>(q.faceCells[i])] = 1;
    }
    // A COUPLED PATCH (cyclic, cyclicAMI) is in the stencil like any other. Its face's normal reads the gradient
    // on BOTH sides (coupledLinear), and the cells on the other side are the partner patch's face cells -- in
    // the stencil already; that is confirmed here rather than assumed, and a pair whose neighbour cells are
    // anything else leaves the stencil unusable, as every coupled mesh did until 2026-10-04.
    // MEASURED then on RAS/mixerVesselAMI (894,950 cells): the alpha hooks took calculateK whole on the host,
    // six times a step, 152.5 ms.
    for (const FvPatch& q : patches)
    {
        if (!q.coupled) continue;
        for (const label c : q.nbrFaceCells)
        {
            if (!wanted[static_cast<std::size_t>(c)]) return st;
        }
        for (const label c : q.amiNbrCells)
        {
            if (!wanted[static_cast<std::size_t>(c)]) return st;
        }
    }
    st.slot.assign(static_cast<std::size_t>(nC), label(-1));
    for (label c = 0; c < nC; ++c)
    {
        if (!wanted[static_cast<std::size_t>(c)]) continue;
        st.slot[static_cast<std::size_t>(c)] = static_cast<label>(st.cells.size());
        st.cells.push_back(c);
    }
    // each cell's internal faces in ascending face order -- the order gaussGrad's face loop reaches it
    std::vector<std::vector<label>> faces(st.cells.size());
    for (label f = 0; f < nIf; ++f)
    {
        const label so = st.slot[static_cast<std::size_t>(own[f])];
        const label sn = st.slot[static_cast<std::size_t>(nei[f])];
        if (so >= 0) faces[static_cast<std::size_t>(so)].push_back(f);
        if (sn >= 0) faces[static_cast<std::size_t>(sn)].push_back(f);
    }
    st.start.push_back(0);
    for (const std::vector<label>& fl : faces)
    {
        st.faces.insert(st.faces.end(), fl.begin(), fl.end());
        st.start.push_back(static_cast<label>(st.faces.size()));
    }
    st.gradAlpha.assign(static_cast<std::size_t>(nC), vector{0, 0, 0});
    st.usable = true;
    return st;
}

bool nHatBoundaryOnlyApplies(
    const InterfaceCoeffs& c,
    bool gradLeastSquares,
    const NHatBoundaryStencil& st)
{
    GradChoice nHatGrad = c.nHatGrad;
    nHatGrad.leastSquares = nHatGrad.leastSquares || gradLeastSquares;
    return st.usable && c.nAlphaSmoothCurvature < 1 && nHatGrad.gaussLinear();
}

void calculateNHatBoundary(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    NHatBoundaryStencil& st,
    SurfaceScalarField& nHatf)
{
    if (!nHatBoundaryOnlyApplies(c, false, st))
    {
        throw std::runtime_error(
            "brae interfaceProperties::calculateNHatBoundary: the case's nHat is not a plain Gauss linear "
            "gradient of the unsmoothed field, on a stencil that holds every cell it reads, which is all this "
            "boundary-only form reproduces; calculateK is the call for it.");
    }
    // fvc::gaussGrad (fvc.cu) at the patches' face cells only: every term in the order the full face loop
    // adds it to that cell -- its internal faces ascending, then its boundary faces patch by patch -- and
    // the same operators, so each cell's gradient is the full one's to the bit
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w = g.weights();
    const std::vector<vector>& Sf = g.Sf();
    std::vector<vector> acc(st.cells.size(), vector{0, 0, 0});
    for (std::size_t k = 0; k < st.cells.size(); ++k)
    {
        const label cell = st.cells[k];
        for (label j = st.start[k]; j < st.start[k + 1]; ++j)
        {
            const label f = st.faces[static_cast<std::size_t>(j)];
            const label o = own[static_cast<std::size_t>(f)];
            const label n = nei[static_cast<std::size_t>(f)];
            const scalar P = alpha1.internal[static_cast<std::size_t>(o)];
            const scalar N = alpha1.internal[static_cast<std::size_t>(n)];
            const scalar pf = w[static_cast<std::size_t>(f)] * (P - N) + N;
            const vector Sfssf = Sf[static_cast<std::size_t>(f)] * pf;
            if (o == cell)
            {
                acc[k] += Sfssf;
            }
            else
            {
                acc[k] = acc[k] - Sfssf;
            }
        }
    }
    finishNHatBoundary(alpha1, c, g, patches, st, acc, nHatf);
}

void calculateNHatBoundaryOfWholeGradient(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    SurfaceScalarField& nHatf)
{
    if (!(c.deltaN > scalar(0)))
    {
        throw std::runtime_error("brae interfaceProperties::calculateNHatBoundaryOfWholeGradient: "
                                 "InterfaceCoeffs::deltaN is unset.");
    }
    // calculateK's step 1 and its boundary half, with nothing of what lies between them or after: the
    // gradient at the faces, the internal faces' normal and the curvature feed no patch face's normal
    const std::vector<vector> gradAlpha = nHatCellGradient(alpha1, c, m, g, patches, false);
    nHatf.internal.clear();
    nHatBoundary(alpha1, c.deltaN, g, patches, gradAlpha, nHatf);
}

bool nHatBoundaryOfSubsetApplies(
    const InterfaceCoeffs& c,
    const NHatBoundaryStencil& st)
{
    return st.usable && c.nAlphaSmoothCurvature < 1 && c.nHatGrad.leastSquares;
}

void calculateNHatBoundaryOfSubsetGradient(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    NHatBoundaryStencil& st,
    SurfaceScalarField& nHatf)
{
    if (!nHatBoundaryOfSubsetApplies(c, st))
    {
        throw std::runtime_error(
            "brae interfaceProperties::calculateNHatBoundaryOfSubsetGradient: the case's nHat is not an "
            "unsmoothed leastSquares gradient on a stencil that holds every cell it reads; "
            "calculateNHatBoundaryOfWholeGradient is the call for it.");
    }
    if (!(c.deltaN > scalar(0)))
    {
        throw std::runtime_error("brae interfaceProperties::calculateNHatBoundaryOfSubsetGradient: "
                                 "InterfaceCoeffs::deltaN is unset.");
    }
    // the patch values, as gradOf's two field forms take them (fvc::leastSquaresGrad, cpu::cellLimitGrad)
    std::vector<std::vector<scalar>> bnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (c.nHatGrad.cellLimitK > scalar(0) && alpha1.boundary[pi]->coupledJump())
        {
            throw std::runtime_error(
                "brae: a cellLimited gradient of a field with a jump across the coupled patch '"
                + patches[pi].name + "' is not ported.");
        }
        bnd[pi] = alpha1.boundary[pi]->value();
    }
    // the subset is found at the first call that wants it: a Gauss-linear case builds the stencil at every
    // change of topology and never reads this
    if (st.subset.cells.size() != st.cells.size())
    {
        st.subset = fvc::gradSubset(m, st.cells);
    }
    fvc::leastSquaresGradAt(alpha1.internal, bnd, m, g, patches, st.subset);
    if (c.nHatGrad.cellLimitK > scalar(0))
    {
        cellLimitGradAt(st.subset, alpha1.internal, bnd, c.nHatGrad.cellLimitK, m, g, patches);
    }
    nHatf.internal.clear();
    nHatBoundary(alpha1, c.deltaN, g, patches, st.subset.grad, nHatf, st.skipEmpty);
}

void finishNHatBoundary(
    const GeometricField<scalar>& alpha1,
    const InterfaceCoeffs& c,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    NHatBoundaryStencil& st,
    std::vector<vector>& acc,
    SurfaceScalarField& nHatf)
{
    if (!(c.deltaN > scalar(0)))
    {
        throw std::runtime_error("brae interfaceProperties::finishNHatBoundary: InterfaceCoeffs::deltaN is unset.");
    }
    if (acc.size() != st.cells.size())
    {
        throw std::runtime_error("brae interfaceProperties::finishNHatBoundary: the internal sums are not the "
                                 "stencil's cells.");
    }
    const std::vector<vector>& Sf = g.Sf();
    // BRAE_CONTROL_NHAT_NO_PATCH_TERMS=1 leaves the patch faces' terms out of the gradient -- the identity
    // gate's control, which has to show the comparison can fail
    static const bool noPatchTerms = std::getenv("BRAE_CONTROL_NHAT_NO_PATCH_TERMS") != nullptr;
    for (std::size_t pi = 0; pi < patches.size() && !noPatchTerms; ++pi)
    {
        const FvPatch& fp = patches[pi];
        if (fp.type == "empty") continue;
        if (fp.coupled)
        {
            // a coupled face is interpolated from its two cells, whatever the patch array holds: fvc::gaussGrad's
            // own term (fvc.cu)
            for (label i = 0; i < fp.size; ++i)
            {
                const std::size_t k = static_cast<std::size_t>(st.slot[static_cast<std::size_t>(fp.faceCells[i])]);
                acc[k] += Sf[static_cast<std::size_t>(fp.start + i)] * coupledLinear(fp, i, alpha1.internal);
            }
            continue;
        }
        const std::vector<scalar>& pv = alpha1.boundary[pi]->value();
        for (label i = 0; i < fp.size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(st.slot[static_cast<std::size_t>(fp.faceCells[i])]);
            acc[k] += Sf[static_cast<std::size_t>(fp.start + i)] * pv[static_cast<std::size_t>(i)];
        }
    }
    for (std::size_t k = 0; k < st.cells.size(); ++k)
    {
        const label cell = st.cells[k];
        st.gradAlpha[static_cast<std::size_t>(cell)] = acc[k] / g.V()[static_cast<std::size_t>(cell)];
    }
    nHatBoundary(alpha1, c.deltaN, g, patches, st.gradAlpha, nHatf, st.skipEmpty);
}

}   // namespace interfaceProps
}   // namespace cpu
}   // namespace brae
