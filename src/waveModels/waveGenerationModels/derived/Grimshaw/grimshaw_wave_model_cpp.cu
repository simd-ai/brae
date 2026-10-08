// Grimshaw -- the third-order solitary wave.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/Grimshaw/GrimshawWaveModel.C:50-176,
//                  :210-245 (setVelocity)
//   tests:     tests/interfoam_waves_vs_openfoam.sh (solitaryGrimshaw)
#include "wave_generation_bases_cpp.cuh"
#include <cmath>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

class Grimshaw : public SolitaryWaveModel
{
public:
    Grimshaw(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : SolitaryWaveModel(patch, m, g, gravity)
    {
        type_ = "Grimshaw";
    }

    std::vector<std::pair<std::string, scalar>> info() const override
    {
        std::vector<std::pair<std::string, scalar>> out = WaveModel::info();
        solitaryInfo(out);
        return out;
    }

protected:
    scalar alfa(
        scalar H,
        scalar h) const
    {
        const scalar eps = H/h;
        return std::sqrt(0.75*eps)*(1.0 - 0.625*eps + (71.0/128.0)*eps*eps);
    }

    scalar eta(
        scalar H,
        scalar h,
        scalar x,
        scalar y,
        scalar theta,
        scalar t,
        scalar X0) const
    {
        const scalar eps = H/h;
        const scalar eps2 = eps*eps;
        const scalar eps3 = eps*eps2;
        const scalar C = std::sqrt(mag(g_)*h)*std::sqrt(1.0 + eps - 0.05*eps2 - (3.0/70.0)*eps3);
        const scalar ts = 3.5*h/std::sqrt(H/h);
        const scalar xa = -C*t + ts - X0 + x*std::cos(theta) + y*std::sin(theta);
        const scalar al = alfa(H, h);
        const scalar s = (1.0)/(std::cosh(al*(xa/h)));
        const scalar s2 = s*s;
        const scalar q = std::tanh(al*(xa/h));
        const scalar q2 = q*q;
        return h
              *(
                   eps*s2
                 - 0.75*eps2*s2*q2
                 + eps3*(0.625*s2*q2 - 1.2625*s2*s2*q2)
               );
    }

    vector Uf(
        scalar H,
        scalar h,
        scalar x,
        scalar y,
        scalar theta,
        scalar t,
        scalar X0,
        scalar z) const
    {
        const scalar eps = H/h;
        const scalar eps2 = eps*eps;
        const scalar eps3 = eps*eps2;
        const scalar C = std::sqrt(mag(g_)*h)*std::sqrt(1.0 + eps - 0.05*eps2 - (3.0/70.0)*eps3);
        const scalar ts = 3.5*h/std::sqrt(eps);
        const scalar xa = -C*t + ts - X0 + x*std::cos(theta) + y*std::sin(theta);
        const scalar al = alfa(H, h);
        const scalar s = (1.0)/(std::cosh(al*(xa/h)));
        const scalar s2 = s*s;
        const scalar s4 = s2*s2;
        const scalar s6 = s2*s4;
        const scalar zbyh = z/h;
        const scalar zbyh2 = zbyh*zbyh;
        const scalar zbyh4 = zbyh2*zbyh2;
        scalar outa = eps*s2 - eps2*(-0.25*s2 + s4 + zbyh2*(1.5*s2 - 2.25*s4));
        scalar outb = 0.475*s2 + 0.2*s4 - 1.2*s6;
        scalar outc = zbyh2*(-1.5*s2 - 3.75*s4 + 7.5*s6);
        scalar outd = zbyh4*(-0.375*s2 + (45.0/16.0)*s4 - (45.0/16.0)*s6);
        scalar u = std::sqrt(mag(g_)*h)*(outa - eps3*(outb + outc + outd));
        outa = eps*s2 - eps2*(0.375*s2 + 2*s4 + zbyh2*(0.5*s2 - 1.5*s4));
        outb = (49.0/640.0)*s2 - 0.85*s4 - 3.6*s6;
        outc = zbyh2*((-13.0/16.0)*s2 -(25.0/16.0)*s4 + 7.5*s6);
        outd = zbyh4*(-0.075*s2 -1.125*s4 - (27.0/16.0)*s6);
        const scalar w = std::sqrt(mag(g_)*h)*(outa - eps3*(outb + outc + outd));
        const scalar v = u*std::sin(waveAngle_);
        u *= std::cos(waveAngle_);
        return vector{u, v, w};
    }

    void setLevel(
        scalar t,
        scalar tCoeff,
        std::vector<scalar>& level) const override
    {
        for (std::size_t p = 0; p < level.size(); ++p)
        {
            const scalar e = eta(waveHeight_, waterDepthRef_, xPaddle_[p], yPaddle_[p], waveAngle_, t, x0_);
            level[p] = waterDepthRef_ + tCoeff*e;
        }
    }

    void setVelocity(
        scalar t,
        scalar tCoeff,
        const std::vector<scalar>& level) override
    {
        for (label facei = 0; facei < patch_.size; ++facei)
        {
            scalar fraction = 1;
            scalar z = 0;
            setPaddlePropeties(level, facei, fraction, z);
            if (!(fraction > 0)) continue;
            const label p = faceToPaddle_[facei];
            const vector U = Uf(waveHeight_, waterDepthRef_, xPaddle_[p], yPaddle_[p], waveAngle_, t,
                                x0_, z);
            U_[facei] = (fraction*U)*tCoeff;
        }
    }
};

} // namespace


std::unique_ptr<WaveModel> makeGrimshaw(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<Grimshaw>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
