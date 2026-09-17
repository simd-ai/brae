// shallowWaterAbsorption -- the outlet every wave tutorial uses: still water, and an active correction.
//
// provenance:
//   openfoam:  src/waveModels/waveAbsorptionModels/base/waveAbsorptionModel/waveAbsorptionModel.C:47-90
//              src/waveModels/waveAbsorptionModels/derived/shallowWaterAbsorption/
//                  shallowWaterAbsorption.C:46-84
//   tests:     tests/interfoam_waves_vs_openfoam.sh, every profile: it is each case's outlet
//
// The level it asks for is the reference depth, with no ramp ("No time ramping applied for
// absorption") and activeAbsorption ALWAYS on, whatever the dictionary says. So its velocity is
// waveModel::correct's active correction -- (reference depth - the water actually standing against the
// patch)*sqrt(g/depth), along the inward normal -- on top of a copy of the cell's vertical velocity.
#include "wave_generation_bases_cpp.cuh"

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

class ShallowWaterAbsorption : public WaveModel
{
public:
    ShallowWaterAbsorption(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : WaveModel(patch, m, g, gravity)
    {
        type_ = "shallowWaterAbsorption";
    }

protected:
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override
    {
        WaveModel::readDict(d, alphaInternal);
        // waveAbsorptionModel::readDict: "always set to true"
        activeAbsorption_ = true;
    }

    scalar timeCoeff(scalar) const override { return scalar(1); }

    void setLevel(
        scalar,
        scalar,
        std::vector<scalar>& level) const override
    {
        for (scalar& l : level)
        {
            l = waterDepthRef_;
        }
    }

    // U's patchInternalField with x and y zeroed -- "zero-gradient condition to z-component of
    // velocity only". The components zeroed are the GLOBAL ones, and correct() then rotates the
    // result as though it were local.
    void setVelocity(
        scalar,
        scalar,
        const std::vector<scalar>&) override
    {
        for (label facei = 0; facei < patch_.size; ++facei)
        {
            const vector& uc = (*UNow_)[patch_.faceCells[facei]];
            U_[facei] = vector{scalar(0), scalar(0), uc.z};
        }
    }

    // alpha's patchInternalField
    void setAlpha(const std::vector<scalar>&) override
    {
        for (label facei = 0; facei < patch_.size; ++facei)
        {
            alpha_[facei] = (*alphaNow_)[patch_.faceCells[facei]];
        }
    }
};

} // namespace


std::unique_ptr<WaveModel> makeShallowWaterAbsorption(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<ShallowWaterAbsorption>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
