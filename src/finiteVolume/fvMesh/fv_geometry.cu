#include "fv_geometry.cuh"
#include <stdexcept>
#include <cmath>

namespace brae {
namespace {
    constexpr scalar ROOTVSMALL = 1.0e-150;
    constexpr scalar VSMALL     = 1.0e-300;
}

// primitiveMeshTools::updateFaceCentresAndAreas (triangle special-case + triangle fan).
void FvGeometry::makeFaceCentresAndAreas(const PrimitiveMesh& m)
{
    const label nF = m.nFaces();
    const std::vector<vector>& P = m.points();
    Cf_.resize(nF);
    Sf_.resize(nF);

    for (label f = 0; f < nF; ++f)
    {
        const label k = m.faceSize(f);

        if (k == 3)
        {
            const vector& a = P[m.faceVert(f, 0)];
            const vector& b = P[m.faceVert(f, 1)];
            const vector& c = P[m.faceVert(f, 2)];
            // triangle::centre and areaNormal (triangleI.H:150-190): (1/3)*(sum), not sum/3
            Cf_[f] = (1.0 / 3.0) * (a + b + c);
            Sf_[f] = 0.5 * cross(b - a, c - a);
        }
        else
        {
            vector fCentre = P[m.faceVert(f, 0)];
            for (label pi = 1; pi < k; ++pi)
                fCentre += P[m.faceVert(f, pi)];
            fCentre = fCentre / static_cast<scalar>(k);

            vector sumN  = {0, 0, 0};
            scalar sumA  = 0.0;
            vector sumAc = {0, 0, 0};
            for (label pi = 0; pi < k; ++pi)
            {
                const vector& thisP = P[m.faceVert(f, pi)];
                const vector& nextP = P[m.faceVert(f, pi == k - 1 ? 0 : pi + 1)];
                const vector c = thisP + nextP + fCentre;
                const vector n = cross(nextP - thisP, fCentre - thisP);
                const scalar a = mag(n);
                sumN  += n;
                sumA  += a;
                sumAc += a * c;
            }

            if (sumA < ROOTVSMALL)
            {
                Cf_[f] = fCentre;
                Sf_[f] = {0, 0, 0};
            }
            else
            {
                // primitiveMeshTools.C: (1.0/3.0)*sumAc/sumA, the scaling BEFORE the division. This
                // was (1/3)*(sumAc/sumA), which differs in the last bit -- enough to move a cell centre
                // by 1e-16 and, on sloshingTank2D, to pin p_rgh's reference in the cell OpenFOAM did
                // not: pRefPoint (0 0 0.15) lies on a face, and findCell takes the nearer centre.
                Cf_[f] = ((1.0 / 3.0) * sumAc) / sumA;
                Sf_[f] = 0.5 * sumN;
            }
        }
    }

    magSf_.resize(nF);
    for (label f = 0; f < nF; ++f)
        magSf_[f] = mag(Sf_[f]);
}

// primitiveMeshTools::makeCellCentresAndVols (estimate centre from face centres, then
// pyramid decomposition), accumulated face-by-face over owner/neighbour.
void FvGeometry::makeCellCentresAndVols(const PrimitiveMesh& m)
{
    const label nC  = m.nCells();
    const label nF  = m.nFaces();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    // Estimate cell centres = mean of adjacent face centres. OF accumulates in TWO separate
    // passes (all-faces owner, then internal-faces neighbour), replicated exactly so the
    // summation order matches OF (matters on ill-conditioned, large-coordinate meshes).
    std::vector<vector> cEst(nC, vector{0, 0, 0});
    std::vector<label>  nCellFaces(nC, 0);
    for (label f = 0; f < nF;  ++f)
    {
        cEst[own[f]] += Cf_[f];
        ++nCellFaces[own[f]];
    }
    for (label f = 0; f < nIf; ++f)
    {
        cEst[nei[f]] += Cf_[f];
        ++nCellFaces[nei[f]];
    }
    for (label c = 0; c < nC; ++c)
        cEst[c] = cEst[c] / static_cast<scalar>(nCellFaces[c]);

    C_.assign(nC, vector{0, 0, 0});
    V_.assign(nC, 0.0);
    // Owner pyramid pass (all faces).
    for (label f = 0; f < nF; ++f)
    {
        const label o = own[f];
        const scalar pyr3Vol = dot(Sf_[f], Cf_[f] - cEst[o]);
        const vector pc      = 0.75 * Cf_[f] + 0.25 * cEst[o];
        C_[o] += pyr3Vol * pc;
        V_[o] += pyr3Vol;
    }
    // Neighbour pyramid pass (internal faces).
    for (label f = 0; f < nIf; ++f)
    {
        const label n = nei[f];
        const scalar pyr3Vol = dot(Sf_[f], cEst[n] - Cf_[f]);
        const vector pc      = 0.75 * Cf_[f] + 0.25 * cEst[n];
        C_[n] += pyr3Vol * pc;
        V_[n] += pyr3Vol;
    }
    for (label c = 0; c < nC; ++c)
        C_[c] = (std::fabs(V_[c]) > VSMALL) ? C_[c] / V_[c] : cEst[c];
    for (label c = 0; c < nC; ++c)
        V_[c] *= (1.0 / 3.0);   // separate final pass, as in OF
}

// basicFvGeometryScheme: weights, deltaCoeffs, nonOrthDeltaCoeffs, nonOrthCorrectionVectors.
void FvGeometry::makeInterpolation(const PrimitiveMesh& m)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    weights_.resize(nIf);
    deltaCoeffs_.resize(nIf);
    nonOrthDeltaCoeffs_.resize(nIf);
    nonOrthCorr_.resize(nIf);

    for (label f = 0; f < nIf; ++f)
    {
        const label o = own[f], n = nei[f];

        const scalar SfdOwn = std::fabs(dot(Sf_[f], Cf_[f] - C_[o]));
        const scalar SfdNei = std::fabs(dot(Sf_[f], C_[n] - Cf_[f]));
        weights_[f] = (std::fabs(SfdOwn + SfdNei) > ROOTVSMALL)
                          ? SfdNei / (SfdOwn + SfdNei)
                          : 0.5;

        const vector delta    = C_[n] - C_[o];
        deltaCoeffs_[f]       = 1.0 / mag(delta);

        const vector unitArea = Sf_[f] / magSf_[f];
        nonOrthDeltaCoeffs_[f] = 1.0 / std::fmax(dot(unitArea, delta), 0.05 * mag(delta));
        nonOrthCorr_[f]        = unitArea - delta * nonOrthDeltaCoeffs_[f];
    }
}

void FvGeometry::buildFaceGeometry(const PrimitiveMesh& m)
{
    makeFaceCentresAndAreas(m);
    areaScaled_ = false;          // areas are raw again: a fresh scale is now legal
    rawArea_.clear();
}

void FvGeometry::buildCellGeometry(const PrimitiveMesh& m)
{
    makeCellCentresAndVols(m);
    makeInterpolation(m);
}

void FvGeometry::applyAreaScaling(const std::vector<std::pair<label, scalar>>& faceScale)
{
    if (areaScaled_)
        throw std::runtime_error(
            "brae: FvGeometry::applyAreaScaling called twice without an intervening buildFaceGeometry. "
            "cyclicACMI area scaling is defined against the RAW face areas; applying it to already-scaled "
            "areas would compound the mask every step and shrink the interface away.");
    for (const auto& fs : faceScale)
    {
        rawArea_.emplace(fs.first, magSf_[fs.first]);   // remember the raw area before touching it
        Sf_[fs.first]    = Sf_[fs.first]*fs.second;
        magSf_[fs.first] = magSf_[fs.first]*fs.second;
    }
    areaScaled_ = true;
}

void FvGeometry::setFaceArea(label f, const vector& Sf)
{
    Sf_[f] = Sf;
    magSf_[f] = mag(Sf);
}

void FvGeometry::updateCellCentresAndVols(const PrimitiveMesh& m)
{
    makeCellCentresAndVols(m);
}

void FvGeometry::build(const PrimitiveMesh& m)
{
    buildFaceGeometry(m);
    buildCellGeometry(m);
}

} // namespace brae
