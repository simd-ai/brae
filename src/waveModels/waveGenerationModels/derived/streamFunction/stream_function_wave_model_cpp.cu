// streamFunction -- Fenton's Fourier approximation, with the coefficients SUPPLIED BY THE CASE.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/streamFunction/streamFunctionWaveModel.C:
//                  51-105, :108-175, :207-227 (readDict)
//   tests:     tests/interfoam_waves_vs_openfoam.sh (streamFunction)
//
// THE MODEL SOLVES NOTHING. uMean, the wave length and the two coefficient lists Bjs and Ejs all come
// from waveProperties -- someone ran a stream-function solver and pasted its output -- and all four
// are mandatory. The wave length in particular is the dictionary's, NOT the dispersion relation's.
#include "wave_generation_bases_cpp.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

class StreamFunction : public RegularWaveModel
{
public:
    StreamFunction(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : RegularWaveModel(patch, m, g, gravity)
    {
        type_ = "streamFunction";
    }

    std::vector<std::pair<std::string, scalar>> info() const override
    {
        std::vector<std::pair<std::string, scalar>> out = WaveModel::info();
        regularInfo(out);
        out.push_back({"uMean", uMean_});
        out.push_back({"Stream function wavelength", waveLength_});
        return out;
    }

protected:
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override
    {
        RegularWaveModel::readDict(d, alphaInternal);
        uMean_ = requiredWaveScalar(d, "uMean", patchName_);
        waveLength_ = requiredWaveScalar(d, "waveLength", patchName_);
        Bjs_ = requiredList(d, "Bjs");
        Ejs_ = requiredList(d, "Ejs");
    }

    std::vector<scalar> requiredList(
        const FoamDict& d,
        const std::string& key) const
    {
        if (!d.found(key))
            throw std::runtime_error(
                "brae waveModel: waveProperties entry for patch `" + patchName_ + "` has no `" + key
                + "`. OpenFOAM reads it with readEntry and stops without it.");
        return d.scalarListOr(key, {});
    }

    scalar eta(
        scalar h,
        scalar kx,
        scalar ky,
        scalar T,
        scalar x,
        scalar y,
        scalar omega,
        scalar t,
        scalar phase) const
    {
        (void)h;
        (void)T;
        const scalar k = std::sqrt(kx*kx + ky*ky);
        scalar strfnAux = 0.0;
        for (std::size_t j = 0; j < Ejs_.size(); ++j)
        {
            strfnAux += Ejs_[j]*std::cos(static_cast<scalar>(j + 1)*(kx*x + ky*y - omega*t + phase));
        }
        return (1/k)*strfnAux;
    }

    vector Uf(
        scalar h,
        scalar kx,
        scalar ky,
        scalar T,
        scalar x,
        scalar y,
        scalar omega,
        scalar t,
        scalar phase,
        scalar z) const
    {
        const scalar k = std::sqrt(kx*kx + ky*ky);
        const scalar phaseTot = kx*x + ky*y - omega*t + phase;
        scalar u = 0.0;
        scalar w = 0.0;
        for (std::size_t j = 0; j < Bjs_.size(); ++j)
        {
            const scalar n = static_cast<scalar>(j + 1);
            u += n*Bjs_[j]*std::cosh(n*k*z)/std::cosh(n*k*h)*std::cos(n*phaseTot);
            w += n*Bjs_[j]*std::sinh(n*k*z)/std::cosh(n*k*h)*std::sin(n*phaseTot);
        }
        u = waveLength_/T - uMean_ + std::sqrt(mag(g_)/k)*u;
        w = std::sqrt(mag(g_)/k)*w;
        const scalar v = u*std::sin(waveAngle_);
        u *= std::cos(waveAngle_);
        return vector{u, v, w};
    }

    void setLevel(
        scalar t,
        scalar tCoeff,
        std::vector<scalar>& level) const override
    {
        const scalar waveOmega = scalar(2)*wavePi/wavePeriod_;
        const scalar waveK = scalar(2)*wavePi/waveLength_;
        const scalar waveKx = waveK*std::cos(waveAngle_);
        const scalar waveKy = waveK*std::sin(waveAngle_);
        for (std::size_t p = 0; p < level.size(); ++p)
        {
            const scalar e = eta(waterDepthRef_, waveKx, waveKy, wavePeriod_, xPaddle_[p], yPaddle_[p],
                                 waveOmega, t, wavePhase_);
            level[p] = waterDepthRef_ + tCoeff*e;
        }
    }

    void setVelocity(
        scalar t,
        scalar tCoeff,
        const std::vector<scalar>& level) override
    {
        const scalar waveOmega = scalar(2)*wavePi/wavePeriod_;
        const scalar waveK = scalar(2)*wavePi/waveLength_;
        const scalar waveKx = waveK*std::cos(waveAngle_);
        const scalar waveKy = waveK*std::sin(waveAngle_);
        for (label facei = 0; facei < patch_.size; ++facei)
        {
            scalar fraction = 1;
            scalar z = 0;
            setPaddlePropeties(level, facei, fraction, z);
            if (!(fraction > 0)) continue;
            const label p = faceToPaddle_[facei];
            const vector U = Uf(waterDepthRef_, waveKx, waveKy, wavePeriod_, xPaddle_[p], yPaddle_[p],
                                waveOmega, t, wavePhase_, z);
            U_[facei] = (fraction*U)*tCoeff;
        }
    }

private:
    scalar uMean_ = 0;
    std::vector<scalar> Bjs_;
    std::vector<scalar> Ejs_;
};

} // namespace


std::unique_ptr<WaveModel> makeStreamFunction(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<StreamFunction>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
