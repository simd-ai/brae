#include "inter_phase_time.cuh"
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
    interPhase::Nested timed("FvGeometry: face centres and areas");
    ++generation_;
    makeFaceCentresAndAreas(m);
    areaScaled_ = false;          // areas are raw again: a fresh scale is now legal
    rawArea_.clear();
}

void FvGeometry::buildCellGeometry(const PrimitiveMesh& m)
{
    ++generation_;
    {
        interPhase::Nested timed("FvGeometry: cell centres and volumes");
        makeCellCentresAndVols(m);
    }
    interPhase::Nested timed("FvGeometry: weights, deltaCoeffs and correction vectors");
    makeInterpolation(m);
}

void FvGeometry::applyAreaScaling(const std::vector<std::pair<label, scalar>>& faceScale)
{
    if (areaScaled_)
        throw std::runtime_error(
            "brae: FvGeometry::applyAreaScaling called twice without an intervening buildFaceGeometry. "
            "cyclicACMI area scaling is defined against the RAW face areas; applying it to already-scaled "
            "areas would compound the mask every step and shrink the interface away.");
    ++generation_;
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
    ++generation_;
    Sf_[f] = Sf;
    magSf_[f] = mag(Sf);
}

void FvGeometry::updateCellCentresAndVols(const PrimitiveMesh& m)
{
    ++generation_;
    makeCellCentresAndVols(m);
}

void FvGeometry::adopt(
    Built& b,
    const PrimitiveMesh& m)
{
    const std::size_t nF = static_cast<std::size_t>(m.nFaces());
    const std::size_t nIf = static_cast<std::size_t>(m.nInternalFaces());
    const std::size_t nC = static_cast<std::size_t>(m.nCells());
    if (b.Cf.size() != nF || b.Sf.size() != nF || b.magSf.size() != nF || b.C.size() != nC || b.V.size() != nC
     || b.weights.size() != nIf || b.deltaCoeffs.size() != nIf || b.nonOrthDeltaCoeffs.size() != nIf
     || b.nonOrthCorr.size() != nIf)
    {
        throw std::runtime_error("brae: FvGeometry::adopt was handed a geometry of another mesh's sizes.");
    }
    Cf_.swap(b.Cf);
    Sf_.swap(b.Sf);
    magSf_.swap(b.magSf);
    C_.swap(b.C);
    V_.swap(b.V);
    weights_.swap(b.weights);
    deltaCoeffs_.swap(b.deltaCoeffs);
    nonOrthDeltaCoeffs_.swap(b.nonOrthDeltaCoeffs);
    nonOrthCorr_.swap(b.nonOrthCorr);
    // as buildFaceGeometry: the areas are raw again
    areaScaled_ = false;
    rawArea_.clear();
    ++generation_;
}

void FvGeometry::copyFrom(
    const FvGeometry& o,
    const PrimitiveMesh& m)
{
    const std::size_t nF = static_cast<std::size_t>(m.nFaces());
    const std::size_t nIf = static_cast<std::size_t>(m.nInternalFaces());
    const std::size_t nC = static_cast<std::size_t>(m.nCells());
    if (o.Cf_.size() != nF || o.Sf_.size() != nF || o.magSf_.size() != nF || o.C_.size() != nC || o.V_.size() != nC
     || o.weights_.size() != nIf || o.deltaCoeffs_.size() != nIf || o.nonOrthDeltaCoeffs_.size() != nIf
     || o.nonOrthCorr_.size() != nIf)
    {
        throw std::runtime_error("brae: FvGeometry::copyFrom was handed a geometry of another mesh's sizes.");
    }
    if (o.areaScaled_ || !o.rawArea_.empty())
    {
        throw std::runtime_error(
            "brae: FvGeometry::copyFrom was handed a geometry whose face areas are scaled; build() leaves them "
            "raw, and a copy standing in for a build must be of the raw ones.");
    }
    Cf_ = o.Cf_;
    Sf_ = o.Sf_;
    magSf_ = o.magSf_;
    C_ = o.C_;
    V_ = o.V_;
    weights_ = o.weights_;
    deltaCoeffs_ = o.deltaCoeffs_;
    nonOrthDeltaCoeffs_ = o.nonOrthDeltaCoeffs_;
    nonOrthCorr_ = o.nonOrthCorr_;
    // as buildFaceGeometry: the areas are raw again
    areaScaled_ = false;
    rawArea_.clear();
    ++generation_;
}

void FvGeometry::build(const PrimitiveMesh& m)
{
    buildFaceGeometry(m);
    buildCellGeometry(m);
}

} // namespace brae
