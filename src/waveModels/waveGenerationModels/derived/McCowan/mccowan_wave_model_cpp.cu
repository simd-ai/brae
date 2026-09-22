// McCowan -- the solitary wave whose surface is the root of an implicit equation.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/McCowan/McCowanWaveModel.C:50-246
//   tests:     tests/interfoam_waves_vs_openfoam.sh (solitaryMcCowan)
//
// TWO NEWTON-RAPHSON SOLVES, both stopping at a residual of 1e-5 -- an ABSOLUTE one, five digits, so
// the elevation this model hands the boundary condition is not converged to round-off in OpenFOAM
// either. What it returns depends on the starting point and the pass count, and both are transcribed:
// m starts from 1, the elevation from H/2, and every call starts again from there.
#include "wave_generation_bases_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

class McCowan : public SolitaryWaveModel
{
public:
    McCowan(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : SolitaryWaveModel(patch, m, g, gravity)
    {
        type_ = "McCowan";
    }

    std::vector<std::pair<std::string, scalar>> info() const override
    {
        std::vector<std::pair<std::string, scalar>> out = WaveModel::info();
        solitaryInfo(out);
        return out;
    }

protected:
    scalar newtonRapsonF1(
        scalar x0,
        scalar H,
        scalar h) const
    {
        const label N = 10000;
        const scalar eps = 1.e-5;
        const scalar maxval = 10000.0;
        label iter = 1;
        scalar x = x0;
        scalar residual = 0;
        while (iter <= N)
        {
            const scalar a = x + 1.0 + 2.0*H/(3.0*h);
            const scalar b = 0.5*x*(1.0 + H/h);
            const scalar c = 0.5*x*(1.0 + h/H);
            const scalar c1 = std::sin(a);
            const scalar fx = (2.0/3.0)*waveSqr(c1) - x*H/(h*std::tan(b));
            residual = std::fabs(fx);
            if (residual < eps) return x;
            if ((iter > 1) && (residual > maxval))
                throw std::runtime_error(
                    "brae waveModel: McCowan's Newton-Raphson for m is diverging (residual "
                    + std::to_string(residual) + "). OpenFOAM stops on the same test.");
            const scalar c2 = 1.0/std::tan(c);
            const scalar c3 = 1.0/std::sin(b);
            const scalar fprime = (4.0/3.0)*c1*std::cos(a) - c2*h/H - b*waveSqr(c3);
            x -= fx/fprime;
            iter++;
        }
        // OpenFOAM warns and carries on with the last iterate
        std::printf("  waveModel McCowan: m did not converge in %ld passes, residual %.3e\n",
                    (long)iter, (double)residual);
        return x;
    }

    scalar newtonRapsonF2(
        scalar x0,
        scalar H,
        scalar h,
        scalar xa,
        scalar m,
        scalar n) const
    {
        (void)H;
        const label N = 10000;
        const scalar eps = 1.e-5;
        const scalar maxval = 10000;
        label iter = 1;
        scalar x = x0;
        scalar residual = 0;
        while (iter <= N)
        {
            const scalar a = m*(1.0 + x/h);
            const scalar c1 = std::cos(a);
            const scalar c2 = std::sin(a);
            const scalar fx = x - (h*n/m*(c2/(c1 + std::cosh(m*xa/h))));
            residual = std::fabs(fx);
            if (residual < eps) return x;
            if ((iter > 1) && (residual > maxval))
                throw std::runtime_error(
                    "brae waveModel: McCowan's Newton-Raphson for the elevation is diverging (residual "
                    + std::to_string(residual) + "). OpenFOAM stops on the same test.");
            const scalar c3 = std::cosh(xa*m/h) + c1;
            const scalar fprime = 1 - n/c3*(c1 - waveSqr(c2)/c3);
            x -= fx/fprime;
            iter++;
        }
        std::printf("  waveModel McCowan: the elevation did not converge in %ld passes, residual %.3e\n",
                    (long)iter, (double)residual);
        return x;
    }

    // (m, n) of the McCowan solution
    void mn(
        scalar H,
        scalar h,
        scalar& mOut,
        scalar& nOut) const
    {
        const scalar xin = 1;
        mOut = newtonRapsonF1(xin, H, h);
        const scalar c1 = std::sin(mOut + (1.0 + (2.0*H/(3.0*h))));
        nOut = (2.0/3.0)*waveSqr(c1);
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
        scalar mm = 0;
        scalar nn = 0;
        mn(H, h, mm, nn);
        const scalar C = std::sqrt(((mag(g_)*h)/mm)*std::tan(mm));
        const scalar ts = 3.5*h/std::sqrt(H/h);
        const scalar Xa = -C*t + ts - X0 + x*std::cos(theta) + y*std::sin(theta);
        const scalar xin = 0.5*H;
        return newtonRapsonF2(xin, H, h, Xa, mm, nn);
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
        scalar mm = 0;
        scalar nn = 0;
        mn(H, h, mm, nn);
        const scalar C = std::sqrt((mag(g_)*h)/mm*std::tan(mm));
        const scalar ts = 3.5*h/std::sqrt(H/h);
        const scalar Xa = -C*t + ts - X0 + x*std::cos(theta) + y*std::sin(theta);
        scalar outa = C*nn*(1.0 + std::cos(mm*z/h)*std::cosh(mm*Xa/h));
        const scalar outb = waveSqr(std::cos(mm*z/h) + std::cosh(mm*Xa/h));
        scalar u = outa/outb;
        outa = C*nn*std::sin(mm*z/h)*std::sinh(mm*Xa/h);
        const scalar w = outa/outb;
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


std::unique_ptr<WaveModel> makeMcCowan(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<McCowan>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
