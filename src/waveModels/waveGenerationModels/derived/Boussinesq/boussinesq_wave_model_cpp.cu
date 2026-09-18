// Boussinesq -- the first-order solitary wave.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/Boussinesq/BoussinesqWaveModel.C:52-190
//   tests:     tests/interfoam_waves_vs_openfoam.sh (solitary)
//
// The crest starts ts = 3.5*h/sqrt(H/h) UPSTREAM of the patch, so the wave enters gradually; there is
// no ramp (timeCoeff is 1) and OpenFOAM's tutorial carries a `wavePeriod 0.0` this model never reads.
#include "wave_generation_bases_cpp.cuh"
#include <cmath>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

class Boussinesq : public SolitaryWaveModel
{
public:
    Boussinesq(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : SolitaryWaveModel(patch, m, g, gravity)
    {
        type_ = "Boussinesq";
    }

    std::vector<std::pair<std::string, scalar>> info() const override
    {
        std::vector<std::pair<std::string, scalar>> out = WaveModel::info();
        solitaryInfo(out);
        return out;
    }

protected:
    scalar eta(
        scalar H,
        scalar h,
        scalar x,
        scalar y,
        scalar theta,
        scalar t,
        scalar X0) const
    {
        const scalar C = std::sqrt(mag(g_)*(H + h));
        const scalar ts = 3.5*h/std::sqrt(H/h);
        const scalar aux = std::sqrt(3.0*H/(4.0*h))/h;
        const scalar Xa = -C*t + ts - X0 + x*std::cos(theta) + y*std::sin(theta);
        return H*1.0/waveSqr(std::cosh(aux*Xa));
    }

    // the first three x-derivatives of eta, in closed form
    vector Deta(
        scalar H,
        scalar h,
        scalar x,
        scalar y,
        scalar theta,
        scalar t,
        scalar X0) const
    {
        const scalar C = std::sqrt(mag(g_)*(H + h));
        const scalar ts = 3.5*h/std::sqrt(H/h);
        const scalar a = std::sqrt(3*H/(4*h))/h;
        const scalar Xa = -C*t + ts - X0 + x*std::cos(theta) + y*std::sin(theta);
        const scalar expTerm = std::exp(2*a*Xa);
        const scalar b = 8*a*h*expTerm;
        vector deta{0, 0, 0};
        deta.x = b*(1 - expTerm)/wavePow3(1 + expTerm);
        deta.y = 2*a*b*(std::exp(4*a*Xa) - 4*expTerm + 1)/wavePow4(1 + expTerm);
        deta.z = -4*waveSqr(a)*b*(std::exp(6*a*Xa) - 11*std::exp(4*a*Xa) + 11*expTerm - 1)
                /wavePow5(1 + expTerm);
        return deta;
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
        const scalar C = std::sqrt(mag(g_)*(H + h));
        const scalar e = eta(H, h, x, y, theta, t, X0);
        const vector De = Deta(H, h, x, y, theta, t, X0);
        scalar u = C*e/h
                  *(
                       1.0
                     - e/(4.0*h)
                     + waveSqr(h)/(3.0*e)*(1.0 - 3.0/2.0*waveSqr(z/h))*De.y
                   );
        const scalar w = -C*z/h
                        *(
                             (1.0 - e/(2.0*h))*De.x
                           + waveSqr(h)/3.0*(1.0 - 1.0/2.0*waveSqr(z/h))*De.z
                         );
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


std::unique_ptr<WaveModel> makeBoussinesq(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<Boussinesq>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
