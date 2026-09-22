#include "vol_point_interpolation_cpp.cuh"
#include "foam_dict.cuh"
#include <stdexcept>
#include <string>

namespace brae {

void VolPointInterpolation::makeWeights(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const std::vector<std::vector<label>>& pointCells)
{
    pointCells_ = &pointCells;
    nInternalFaces_ = m.nInternalFaces();
    patchStart_.clear();
    patchSize_.clear();
    for (const FvPatch& p : patches)
    {
        if (isCoupledInterfaceType(p.type))
        {
            throw std::runtime_error(
                "brae volPointInterpolation: patch `" + p.name + "` is " + p.type + ". A coupled patch "
                "sums its points' weights across the coupling (syncUntransformedData) and a separated one "
                "adds a normalisation; neither is ported.");
        }
        patchStart_.push_back(p.start);
        patchSize_.push_back(p.size);
    }

    // calcBoundaryAddressing
    boundary_ = primitivePatch(m, faceRange(m.nInternalFaces(), m.nFaces() - m.nInternalFaces()));
    boundaryIsPatchFace_.assign(boundary_.faces.size(), 0);
    isPatchPoint_.assign(static_cast<std::size_t>(m.nPoints()), 0);
    for (const FvPatch& pp : patches)
    {
        if (pp.type == "empty") continue;
        label bFacei = pp.start - m.nInternalFaces();
        for (label i = 0; i < pp.size; ++i)
        {
            boundaryIsPatchFace_[static_cast<std::size_t>(bFacei)] = 1;
            const label f = pp.start + i;
            for (label fp = 0; fp < m.faceSize(f); ++fp)
            {
                isPatchPoint_[static_cast<std::size_t>(m.faceVert(f, fp))] = 1;
            }
            bFacei++;
        }
    }

    // Running sum of weights
    std::vector<scalar> sumWeights(static_cast<std::size_t>(m.nPoints()), 0.0);

    // makeInternalWeights
    const std::vector<vector>& points = m.points();
    const std::vector<vector>& cellCentres = g.C();
    pointWeights_.assign(points.size(), std::vector<scalar>());
    for (std::size_t pointi = 0; pointi < points.size(); ++pointi)
    {
        if (!isPatchPoint_[pointi])
        {
            const std::vector<label>& pcp = pointCells[pointi];
            std::vector<scalar>& pw = pointWeights_[pointi];
            pw.resize(pcp.size());
            for (std::size_t pointCelli = 0; pointCelli < pcp.size(); ++pointCelli)
            {
                pw[pointCelli] = 1.0/mag(points[pointi] - cellCentres[static_cast<std::size_t>(pcp[pointCelli])]);
                sumWeights[pointi] += pw[pointCelli];
            }
        }
    }

    // makeBoundaryWeights
    const std::vector<vector>& faceCentres = g.Cf();
    boundaryPointWeights_.assign(boundary_.meshPoints.size(), std::vector<scalar>());
    for (std::size_t i = 0; i < boundary_.meshPoints.size(); ++i)
    {
        const label pointi = boundary_.meshPoints[i];
        if (isPatchPoint_[static_cast<std::size_t>(pointi)])
        {
            const std::vector<label>& pFaces = boundary_.pointFaces[i];
            std::vector<scalar>& pw = boundaryPointWeights_[i];
            pw.resize(pFaces.size());
            sumWeights[static_cast<std::size_t>(pointi)] = 0.0;
            for (std::size_t j = 0; j < pFaces.size(); ++j)
            {
                if (boundaryIsPatchFace_[static_cast<std::size_t>(pFaces[j])])
                {
                    const label facei = m.nInternalFaces() + pFaces[j];
                    pw[j] = 1.0/mag(points[static_cast<std::size_t>(pointi)] - faceCentres[static_cast<std::size_t>(facei)]);
                    sumWeights[static_cast<std::size_t>(pointi)] += pw[j];
                }
                else
                {
                    pw[j] = 0.0;
                }
            }
        }
    }

    // Normalise internal weights
    for (std::size_t pointi = 0; pointi < pointWeights_.size(); ++pointi)
    {
        for (scalar& w : pointWeights_[pointi])
        {
            w /= sumWeights[pointi];
        }
    }
    // Normalise boundary weights
    for (std::size_t i = 0; i < boundary_.meshPoints.size(); ++i)
    {
        const label pointi = boundary_.meshPoints[i];
        for (scalar& w : boundaryPointWeights_[i])
        {
            w /= sumWeights[static_cast<std::size_t>(pointi)];
        }
    }
}

void VolPointInterpolation::interpolateInternalField(
    const std::vector<vector>& vf,
    std::vector<vector>& pf) const
{
    const std::vector<std::vector<label>>& pointCells = *pointCells_;
    for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
    {
        if (!isPatchPoint_[pointi])
        {
            const std::vector<scalar>& pw = pointWeights_[pointi];
            const std::vector<label>& ppc = pointCells[pointi];
            pf[pointi] = vector{0, 0, 0};
            for (std::size_t pointCelli = 0; pointCelli < ppc.size(); ++pointCelli)
            {
                pf[pointi] += pw[pointCelli]*vf[static_cast<std::size_t>(ppc[pointCelli])];
            }
        }
    }
}

void VolPointInterpolation::interpolateBoundaryField(
    const std::vector<std::vector<vector>>& boundaryField,
    std::vector<vector>& pf) const
{
    // flatBoundaryField: every boundary face's value, zero on empty patches
    std::vector<vector> boundaryVals(boundary_.faces.size(), vector{0, 0, 0});
    for (std::size_t patchi = 0; patchi < patchStart_.size(); ++patchi)
    {
        const std::vector<vector>& pfld = boundaryField[patchi];
        const std::size_t offset = static_cast<std::size_t>(patchStart_[patchi] - nInternalFaces_);
        // restrict transcribing to actual size of the patch field - handles "empty" patch type etc.
        for (std::size_t i = 0; i < pfld.size(); ++i)
        {
            if (boundaryIsPatchFace_[offset + i])
            {
                boundaryVals[offset + i] = pfld[i];
            }
        }
    }

    // Do points on 'normal' patches from the surrounding patch faces
    for (std::size_t i = 0; i < boundary_.meshPoints.size(); ++i)
    {
        const label pointi = boundary_.meshPoints[i];
        if (isPatchPoint_[static_cast<std::size_t>(pointi)])
        {
            const std::vector<label>& pFaces = boundary_.pointFaces[i];
            const std::vector<scalar>& pWeights = boundaryPointWeights_[i];
            vector& val = pf[static_cast<std::size_t>(pointi)];
            val = vector{0, 0, 0};
            for (std::size_t j = 0; j < pFaces.size(); ++j)
            {
                if (boundaryIsPatchFace_[static_cast<std::size_t>(pFaces[j])])
                {
                    val += pWeights[j]*boundaryVals[static_cast<std::size_t>(pFaces[j])];
                }
            }
        }
    }
}

} // namespace brae
