#pragma once
// OpenFOAM's waveMaker point boundary condition: the displacement of a wave paddle's points -- a piston,
// a flap hinged at the bed, or a solitary-wave piston stroke. The host reference.
//
// provenance:
//   openfoam: src/waveModels/derivedPointPatchFields/waveMaker/waveMakerPointPatchVectorField.C:41-56
//                 (motionTypeNames), :60-102 (g, waveLength, timeCoeff), :104-148
//                 (initialiseGeometry), :174-235 (the dictionary constructor), :278-436 (updateCoeffs)
//             src/waveModels/derivedPointPatchFields/waveMaker/waveMakerPointPatchVectorField.H:201
//                 (firstTime)
//
// THE PADDLE GEOMETRY IS THE PATCH'S AT CONSTRUCTION: its bounding box, the paddle centres and every
// point's paddle are taken from the points the mesh had then and never again. updateCoeffs reads the
// CURRENT points only for the flap's depth scaling, and a flap moves its points along n, so that height
// is the one they started with.
//
// wavePhase is read and never used: updateCoeffs' phase is the paddle centre's position alone. The
// solitary branch overwrites wavePeriod with its own, which changes nothing it computes -- the wave
// length that reads it is used by the other two branches only.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <string>
#include <vector>

namespace brae {

class WaveMakerPointPatchVectorField
{
public:
    // the dictionary constructor. localPoints are the patch's points in meshPoints order; g is
    // constant/g's value; startTime is the run's, the default of the `startTime` entry.
    WaveMakerPointPatchVectorField(
        const FoamDict& dict,
        const std::string& patchName,
        const std::vector<vector>& localPoints,
        const vector& g,
        scalar startTime);

    // updateCoeffs at time `time`: the displacement of every local point
    std::vector<vector> updateCoeffs(
        scalar time,
        const std::vector<vector>& localPoints);

    // whether the dictionary carried `value`: without one, the constructor calls updateCoeffs
    bool hadValue() const
    {
        return hadValue_;
    }

private:
    enum class MotionType
    {
        piston,
        flap,
        solitary
    };

    scalar waveLength(
        scalar h,
        scalar T) const;
    scalar timeCoeff(scalar t) const;

    std::string patchName_;
    MotionType motionType_ = MotionType::piston;
    vector n_{0, 0, 0};
    vector g_{0, 0, 0};
    scalar initialDepth_ = 0;
    scalar wavePeriod_ = 0;
    scalar waveHeight_ = 0;
    scalar waveAngle_ = 0;
    scalar startTime_ = 0;
    scalar rampTime_ = 1;
    bool secondOrder_ = false;
    label nPaddle_ = 1;
    bool hadValue_ = false;
    // initialiseGeometry
    scalar zMinGb_ = 0;
    std::vector<scalar> xPaddle_;
    std::vector<scalar> yPaddle_;
    std::vector<label> pointToPaddle_;
    // updateCoeffs' first call
    std::vector<scalar> waterDepthRef_;
    bool firstTime_ = true;
};

} // namespace brae
