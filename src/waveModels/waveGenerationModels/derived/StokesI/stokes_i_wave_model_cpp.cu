// StokesI -- linear (Airy) waves.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/StokesI/StokesIWaveModel.C:55-177
//   tests:     tests/interfoam_waves_vs_openfoam.sh (stokesI: five profiles)
#include "wave_generation_bases_cpp.cuh"
#include <cmath>

namespace brae {
namespace cpu {
namespace waveModels {

StokesI::StokesI(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
    : RegularWaveModel(patch, m, g, gravity)
{
    type_ = "StokesI";
}


std::vector<std::pair<std::string, scalar>> StokesI::info() const
{
    std::vector<std::pair<std::string, scalar>> out = WaveModel::info();
    regularInfo(out);
    return out;
}


void StokesI::readDict(
    const FoamDict& d,
    const std::vector<scalar>& alphaInternal)
{
    RegularWaveModel::readDict(d, alphaInternal);
    waveLength_ = waveLength(waterDepthRef_, wavePeriod_);
}


scalar StokesI::waveLength(
    scalar h,
    scalar T) const
{
    const scalar L0 = mag(g_)*T*T/(scalar(2)*wavePi);
    scalar L = L0;
    for (int i = 1; i <= 100; ++i)
    {
        L = L0*std::tanh(scalar(2)*wavePi*h/L);
    }
    return L;
}


namespace {

scalar etaStokesI(
    scalar H,
    scalar Kx,
    scalar x,
    scalar Ky,
    scalar y,
    scalar omega,
    scalar t,
    scalar phase)
{
    const scalar phaseTot = Kx*x + Ky*y - omega*t + phase;
    return H*scalar(0.5)*std::cos(phaseTot);
}

} // namespace


vector StokesI::UfBase(
    scalar H,
    scalar h,
    scalar Kx,
    scalar x,
    scalar Ky,
    scalar y,
    scalar omega,
    scalar t,
    scalar phase,
    scalar z) const
{
    const scalar k = std::sqrt(Kx*Kx + Ky*Ky);
    const scalar phaseTot = Kx*x + Ky*y - omega*t + phase;
    scalar u = H*scalar(0.5)*omega*std::cos(phaseTot)*std::cosh(k*z)/std::sinh(k*h);
    const scalar w = H*scalar(0.5)*omega*std::sin(phaseTot)*std::sinh(k*z)/std::sinh(k*h);
    const scalar v = u*std::sin(waveAngle_);
    u *= std::cos(waveAngle_);
    return vector{u, v, w};
}


void StokesI::setLevel(
    scalar t,
    scalar tCoeff,
    std::vector<scalar>& level) const
{
    const scalar waveOmega = scalar(2)*wavePi/wavePeriod_;
    const scalar waveK = scalar(2)*wavePi/waveLength_;
    const scalar waveKx = waveK*std::cos(waveAngle_);
    const scalar waveKy = waveK*std::sin(waveAngle_);
    for (std::size_t p = 0; p < level.size(); ++p)
    {
        const scalar e = etaStokesI(waveHeight_, waveKx, xPaddle_[p], waveKy, yPaddle_[p], waveOmega,
                                    t, wavePhase_);
        level[p] = waterDepthRef_ + tCoeff*e;
    }
}


void StokesI::setVelocity(
    scalar t,
    scalar tCoeff,
    const std::vector<scalar>& level)
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
        // virtual: StokesII supplies its own
        const vector Uf = UfBase(waveHeight_, waterDepthRef_, waveKx, xPaddle_[p], waveKy,
                                 yPaddle_[p], waveOmega, t, wavePhase_, z);
        U_[facei] = (fraction*Uf)*tCoeff;
    }
}


std::unique_ptr<WaveModel> makeStokesI(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<StokesI>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
