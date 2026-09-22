// StokesII -- StokesI plus the second harmonic, in the elevation and in both velocity components.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/StokesII/StokesIIWaveModel.C:53-137
//   tests:     tests/interfoam_waves_vs_openfoam.sh (stokesII)
//
// It IS a StokesI: the wave length is still the LINEAR dispersion relation's, and setVelocity is
// StokesI's own, reaching this class's velocity through the virtual UfBase.
#include "wave_generation_bases_cpp.cuh"
#include <cmath>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

class StokesII : public StokesI
{
public:
    StokesII(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : StokesI(patch, m, g, gravity)
    {
        type_ = "StokesII";
    }

protected:
    scalar eta(
        scalar H,
        scalar h,
        scalar Kx,
        scalar x,
        scalar Ky,
        scalar y,
        scalar omega,
        scalar t,
        scalar phase) const
    {
        const scalar k = std::sqrt(Kx*Kx + Ky*Ky);
        const scalar sigma = std::tanh(k*h);
        const scalar phaseTot = Kx*x + Ky*y - omega*t + phase;
        return H*scalar(0.5)*std::cos(phaseTot)
             + k*H*H/scalar(4)*(scalar(3) - sigma*sigma)/(scalar(4)*wavePow3(sigma))
              *std::cos(scalar(2)*phaseTot);
    }

    vector UfBase(
        scalar H,
        scalar h,
        scalar Kx,
        scalar x,
        scalar Ky,
        scalar y,
        scalar omega,
        scalar t,
        scalar phase,
        scalar z) const override
    {
        const scalar k = std::sqrt(Kx*Kx + Ky*Ky);
        const scalar phaseTot = Kx*x + Ky*y - omega*t + phase;
        const scalar sh = std::sinh(k*h);
        const scalar sh4 = wavePow4(sh);
        scalar u = H*scalar(0.5)*omega*std::cos(phaseTot)*std::cosh(k*z)/sh
                 + scalar(3)/scalar(4)*H*H/scalar(4)*omega*k*std::cosh(scalar(2)*k*z)/sh4
                  *std::cos(scalar(2)*phaseTot);
        const scalar w = H*scalar(0.5)*omega*std::sin(phaseTot)*std::sinh(k*z)/sh
                       + scalar(3)/scalar(4)*H*H/scalar(4)*omega*k*std::sinh(scalar(2)*k*z)/sh4
                        *std::sin(scalar(2)*phaseTot);
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
            const scalar e = eta(waveHeight_, waterDepthRef_, waveKx, xPaddle_[p], waveKy, yPaddle_[p],
                                 waveOmega, t, wavePhase_);
            level[p] = waterDepthRef_ + tCoeff*e;
        }
    }
};

} // namespace


std::unique_ptr<WaveModel> makeStokesII(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<StokesII>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
