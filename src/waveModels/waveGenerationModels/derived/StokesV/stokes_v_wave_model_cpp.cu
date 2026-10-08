// StokesV -- Skjelbreia and Hendrickson's fifth-order Stokes wave, as OpenFOAM writes it.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/StokesV/StokesVWaveModel.C:53-560 (the
//                  twenty-four coefficient functions), :563-643 (initialise), :646-745 (eta, Uf),
//                  :859-885 (readDict)
//   tests:     tests/interfoam_waves_vs_openfoam.sh (stokesV): the log's `Lambda`, and the inlet's
//              velocity face by face from OpenFOAM's time directory -- five harmonics in the phase,
//              five in the depth, so a wrong coefficient shows at its own order
//
// THE COEFFICIENTS ARE RATIONAL FUNCTIONS OF sinh(kh) AND cosh(kh) with integer constants up to
// 324000, and the only thing that can go wrong in porting them is a digit. They were therefore
// produced FROM OpenFOAM's source text by a rewrite of its function calls (sinh -> std::sinh, pow4 ->
// wavePow4, ...) rather than retyped, and the term order is OpenFOAM's. C3 and C4 are not here: OpenFOAM
// defines both and calls neither.
//
// ONE THING OpenFOAM DOES THAT LOOKS LIKE A SLIP AND IS TRANSCRIBED. readDict solves the fifth-order
// dispersion relation for the wave number AND lambda, keeps lambda, and DROPS the wave number: it is
// written to a local and never stored (StokesVWaveModel.C:863-875). So the wave length every later
// call uses is still StokesI's -- the LINEAR dispersion relation's -- while the amplitude parameter is
// the fifth-order one. brae does the same, because the oracle is OpenFOAM.
#include "wave_generation_bases_cpp.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

scalar stokesVA11(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    return 1.0/s;
}

scalar stokesVA13(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return -waveSqr(c)*(5*waveSqr(c) + 1)/(8*wavePow5(s));
}

scalar stokesVA15(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
       -(
            1184*std::pow(c, 10)
          - 1440*std::pow(c, 8)
          - 1992*wavePow6(c)
          + 2641*wavePow4(c)
          - 249*waveSqr(c) + 18
        )
       /(1536*std::pow(s, 11));
}

scalar stokesVA22(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    return 3/(8*wavePow4(s));
}

scalar stokesVA24(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
        (192*std::pow(c, 8) - 424*std::pow(c, 6) - 312*wavePow4(c) + 480*waveSqr(c) - 17)
       /(768*std::pow(s, 10));
}

scalar stokesVA33(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return (13 - 4*waveSqr(c))/(64*std::pow(s, 7));
}

scalar stokesVA35(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
        (
            512*std::pow(c, 12)
          + 4224*std::pow(c, 10)
          - 6800*std::pow(c, 8)
          - 12808*std::pow(c, 6)
          + 16704.0*wavePow4(c)
          - 3154*waveSqr(c)
          + 107
        )
       /(4096*std::pow(s, 13)*(6*waveSqr(c) - 1));
}

scalar stokesVA44(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
        (80*std::pow(c, 6) - 816*wavePow4(c) + 1338*waveSqr(c) - 197)
       /(1536*std::pow(s, 10)*(6*waveSqr(c) - 1));
}

scalar stokesVA55(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
       -(
            2880*std::pow(c, 10)
          - 72480*std::pow(c, 8)
          + 324000*std::pow(c, 6)
          - 432000*wavePow4(c)
          + 163470*waveSqr(c)
          - 16245
        )
       /(61440*std::pow(s, 11)*(6*waveSqr(c) - 1)*(8*wavePow4(c) - 11*waveSqr(c) + 3));
}

scalar stokesVB22(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return (2*waveSqr(c) + 1)*c/(4*wavePow3(s));
}

scalar stokesVB24(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
        (272*std::pow(c, 8) - 504*std::pow(c, 6) - 192*wavePow4(c) + 322*waveSqr(c) + 21)*c
       /(384*std::pow(s, 9));
}

scalar stokesVB33(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return (8*wavePow6(c) + 1)*3/(64*wavePow6(s));
}

scalar stokesVB33k(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);

    const scalar sk = h*s;
    const scalar ck = h*c;
    return 9.*wavePow5(c)*ck/(4*wavePow6(s)) - (9*(8*wavePow6(c) + 1))/(32*std::pow(s, 7))*sk;
}

scalar stokesVB35(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
        (
            88128*std::pow(c, 14)
          - 208224*std::pow(c, 12)
          + 70848*std::pow(c, 10)
          + 54000*std::pow(c, 8)
          - 21816*wavePow6(c)
          + 6264*wavePow4(c)
          - 54*waveSqr(c)
          - 81
        )
       /(12288*std::pow(s, 12)*(6*waveSqr(c) - 1));
}

scalar stokesVB35k(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);

    const scalar sk = h*s;
    const scalar ck = h*c;
    return
        (
            14*88128*std::pow(c, 13)*ck
          - 12*208224*std::pow(c, 11)*ck
          + 10*70848*std::pow(c, 9)*ck
          + 8*54000.0*std::pow(c, 7)*ck
          - 6*21816*wavePow5(c)*ck
          + 4*6264*wavePow3(c)*ck
          - 2*54*c*ck
        )
       /(12288*std::pow(s, 12)*(6*waveSqr(c) - 1))
      - (
            88128*std::pow(c, 14)
          - 208224*std::pow(c, 12)
          + 70848*std::pow(c, 10)
          + 54000*std::pow(c, 8)
          - 21816*wavePow6(c)
          + 6264*wavePow4(c)
          - 54*waveSqr(c)
          - 81
        )*12
       /(12288*std::pow(s, 13)*(6*waveSqr(c) - 1))*sk
      - (
            88128*std::pow(c, 14)
          - 208224*std::pow(c, 12)
          + 70848*std::pow(c, 10)
          + 54000*std::pow(c, 8)
          - 21816*wavePow6(c)
          + 6264*wavePow4(c)
          - 54*waveSqr(c)
          - 81
        )*12*c*ck
       /(12288*std::pow(s, 12)*waveSqr(6*waveSqr(c) - 1));
}

scalar stokesVB44(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
        (
            768*std::pow(c, 10)
          - 448*std::pow(c, 8)
          - 48*wavePow6(c)
          + 48*wavePow4(c)
          + 106*waveSqr(c)
          - 21
        )*c
       /(384*std::pow(s, 9)*(6*waveSqr(c) - 1));
}

scalar stokesVB55(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
        (
            192000*std::pow(c, 16)
          - 262720*std::pow(c, 14)
          + 83680*std::pow(c, 12)
          + 20160*std::pow(c, 10)
          - 7280*std::pow(c, 8)
          + 7160*std::pow(c, 6)
          - 1800*std::pow(c, 4)
          - 1050*waveSqr(c)
          + 225
        )
       /(12288*std::pow(s, 10)*(6*waveSqr(c) - 1)*(8*wavePow4(c) - 11*waveSqr(c) + 3));
}

scalar stokesVB55k(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);

    const scalar sk = h*s;
    const scalar ck = h*c;
    return
        (
            16*192000*std::pow(c, 15)*ck
          - 14*262720*std::pow(c, 13)*ck
          + 12*83680*std::pow(c, 11)*ck
          + 10*20160*std::pow(c, 9)*ck
          - 8*7280*std::pow(c, 7)*ck
          + 6*7160*std::pow(c, 5)*ck
          - 4*1800*std::pow(c, 3)*ck
          - 2*1050*std::pow(c, 1)*ck
        )
       /(12288*std::pow(s, 10)*(6*waveSqr(c) - 1)*(8*std::pow(c, 4) - 11*waveSqr(c) + 3))
      - (
            192000*std::pow(c, 16)
          - 262720*std::pow(c, 14)
          + 83680*std::pow(c, 12)
          + 20160*std::pow(c, 10)
          - 7280*std::pow(c, 8)
          + 7160*std::pow(c, 6)
          - 1800*std::pow(c, 4)
          - 1050*std::pow(c, 2)
          + 225
        )*10.0
       /(12288*std::pow(s, 11)*(6*waveSqr(c) - 1)*(8*wavePow4(c) - 11*waveSqr(c) + 3))*sk
      - (
            192000*std::pow(c, 16)
          - 262720*std::pow(c, 14)
          + 83680*std::pow(c, 12)
          + 20160*std::pow(c, 10)
          - 7280*std::pow(c, 8)
          + 7160*std::pow(c, 6)
          - 1800*std::pow(c, 4)
          - 1050*std::pow(c, 2)
          + 225
        )*12*c*ck
        /(12288*std::pow(s, 10)*waveSqr(6*waveSqr(c) - 1)*(8*wavePow4(c) - 11*waveSqr(c) + 3))
      - (
            192000*std::pow(c, 16)
          - 262720*std::pow(c, 14)
          + 83680*std::pow(c, 12)
          + 20160*std::pow(c, 10)
          - 7280*std::pow(c, 8)
          + 7160*std::pow(c, 6)
          - 1800*std::pow(c, 4)
          - 1050*std::pow(c, 2)
          + 225
        )*(32*wavePow3(c) - 22*c)*ck
        /(12288*std::pow(s, 10)*(6*waveSqr(c) - 1)*waveSqr(8*wavePow4(c) - 11*waveSqr(c) + 3));
}

scalar stokesVC1(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return (8*wavePow4(c) - 8*waveSqr(c) + 9)/(8*wavePow4(s));
}

scalar stokesVC1k(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);

    const scalar sk = h*s;
    const scalar ck = h*c;
    return
        (4*8*wavePow3(c)*ck - 2*8*c*ck)/(8*wavePow4(s))
      - (8*wavePow4(c) - 8*waveSqr(c) + 9)*4*sk/(8*wavePow5(s));
}

scalar stokesVC2(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);
    return
        (
            3840*std::pow(c, 12)
          - 4096*std::pow(c, 10)
          + 2592*std::pow(c, 8)
          - 1008*std::pow(c, 6)
          + 5944*std::pow(c, 4)
          - 1830*std::pow(c, 2)
          + 147
        )
       /(512*std::pow(s, 10)*(6*waveSqr(c) - 1));
}

scalar stokesVC2k(
    scalar h,
    scalar k)
{
    const scalar s = std::sinh(k*h);
    const scalar c = std::cosh(k*h);

    const scalar sk = h*s;
    const scalar ck = h*c;
    return
        (
            12*3840*std::pow(c, 11)*ck
          - 10*4096*std::pow(c, 9)*ck
          + 8*2592*std::pow(c, 7)*ck
          - 6*1008*std::pow(c, 5)*ck
          + 4*5944*std::pow(c, 3)*ck
          - 2*1830*c*ck
        )
       /(512*std::pow(s, 10)*(6*waveSqr(c) - 1))
      - (
            3840*std::pow(c, 12)
          - 4096*std::pow(c, 10)
          + 2592*std::pow(c, 8)
          - 1008*std::pow(c, 6)
          + 5944*std::pow(c, 4)
          - 1830*std::pow(c, 2)
          + 147
        )*10*sk
       /(512*std::pow(s, 11)*(6*waveSqr(c) - 1))
      - (
            3840*std::pow(c, 12)
          - 4096*std::pow(c, 10)
          + 2592*std::pow(c, 8)
          - 1008*std::pow(c, 6)
          + 5944*std::pow(c, 4)
          - 1830*std::pow(c, 2)
          + 147
        )*12*c*ck
       /(512*std::pow(s, 10)*waveSqr(6*waveSqr(c) - 1));
}

// The Newton solve for (k, lambda) from the wave height and period. Tolerance 1e-12 on both
// residuals, at most 10000 passes; the caller refuses a result whose residuals exceed 1e-3.
void stokesVInitialise(
    scalar magG,
    scalar H,
    scalar d,
    scalar T,
    scalar& kOut,
    scalar& lambdaOut,
    scalar& f1Out,
    scalar& f2Out)
{
    scalar f1 = 1;
    scalar f2 = 1;
    const scalar pi = wavePi;
    scalar k = 2.0*pi/(std::sqrt(magG*d)*T);
    scalar lambda = H/2.0*k;
    label n = 0;
    const scalar tolerance = 1e-12;
    const label iterMax = 10000;
    while ((std::fabs(f1) > tolerance || std::fabs(f2) > tolerance) && (n < iterMax))
    {
        const scalar b33 = stokesVB33(d, k);
        const scalar b35 = stokesVB35(d, k);
        const scalar b55 = stokesVB55(d, k);
        const scalar c1 = stokesVC1(d, k);
        const scalar c2 = stokesVC2(d, k);
        const scalar b33k = stokesVB33k(d, k);
        const scalar b35k = stokesVB35k(d, k);
        const scalar b55k = stokesVB55k(d, k);
        const scalar c1k = stokesVC1k(d, k);
        const scalar c2k = stokesVC2k(d, k);
        const scalar l2 = waveSqr(lambda);
        const scalar l3 = l2*lambda;
        const scalar l4 = l3*lambda;
        const scalar l5 = l4*lambda;
        const scalar Bmat11 =
            2*pi/(waveSqr(k)*d)*(lambda + l3*b33 + l5*(b35 + b55))
          - 2*pi/(k*d)*(l3*b33k + l5*(b35k + b55k));
        const scalar Bmat12 =
          - 2*pi/(k*d)*(1 + 3*l2*b33 + 5*l4*(b35 + b55));
        const scalar Bmat21 =
          - d/(2*pi)*std::tanh(k*d)*(1 + l2*c1 + l4*c2)
          - k*d/(2*pi)*(1 - waveSqr(std::tanh(k*d)))*d*(1 + l2*c1 + l4*c2)
          - k*d/(2*pi)*std::tanh(k*d)*(l2*c1k + l4*c2k);
        const scalar Bmat22 = - k*d/(2.0*pi)*std::tanh(k*d)*(2*lambda*c1 + 4*l3*c2);
        f1 = pi*H/d - 2*pi/(k*d)*(lambda + l3*b33 + l5*(b35 + b55));
        f2 = (2*pi*d)/(magG*waveSqr(T)) - k*d/(2*pi)*std::tanh(k*d)*(1 + l2*c1 + l4*c2);
        const scalar lambdaPr = (f1*Bmat21 - f2*Bmat11)/(Bmat11*Bmat22 - Bmat12*Bmat21);
        const scalar kPr = (f2*Bmat12 - f1*Bmat22)/(Bmat11*Bmat22 - Bmat12*Bmat21);
        lambda += lambdaPr;
        k += kPr;
        n++;
    }
    kOut = k;
    lambdaOut = lambda;
    f1Out = std::fabs(f1);
    f2Out = std::fabs(f2);
}


class StokesV : public StokesI
{
public:
    StokesV(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : StokesI(patch, m, g, gravity)
    {
        type_ = "StokesV";
    }

    std::vector<std::pair<std::string, scalar>> info() const override
    {
        std::vector<std::pair<std::string, scalar>> out = StokesI::info();
        out.push_back({"Lambda", lambda_});
        return out;
    }

protected:
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override
    {
        StokesI::readDict(d, alphaInternal);
        scalar f1 = 0;
        scalar f2 = 0;
        // solved for, and dropped: see the header
        scalar waveK = 0;
        stokesVInitialise(mag(g_), waveHeight_, waterDepthRef_, wavePeriod_, waveK, lambda_, f1, f2);
        if (f1 > 0.001 || f2 > 0.001)
            throw std::runtime_error(
                "brae waveModel: patch `" + patchName_ + "`: no convergence for Stokes V wave theory "
                "(f1 " + std::to_string(f1) + ", f2 " + std::to_string(f2) + "). OpenFOAM stops on "
                "the same test.");
    }

    scalar eta(
        scalar h,
        scalar kx,
        scalar ky,
        scalar lambda,
        scalar T,
        scalar x,
        scalar y,
        scalar t,
        scalar phase) const
    {
        const scalar k = std::sqrt(kx*kx + ky*ky);
        const scalar b22 = stokesVB22(h, k);
        const scalar b24 = stokesVB24(h, k);
        const scalar b33 = stokesVB33(h, k);
        const scalar b35 = stokesVB35(h, k);
        const scalar b44 = stokesVB44(h, k);
        const scalar b55 = stokesVB55(h, k);
        const scalar l2 = waveSqr(lambda);
        const scalar l3 = l2*lambda;
        const scalar l4 = l3*lambda;
        const scalar l5 = l4*lambda;
        const scalar amp1 = lambda/k;
        const scalar amp2 = (b22*l2 + b24*l4)/k;
        const scalar amp3 = (b33*l3 + b35*l5)/k;
        const scalar amp4 = b44*l4/k;
        const scalar amp5 = b55*l5/k;
        const scalar theta = kx*x + ky*y - 2.0*wavePi/T*t + phase;
        return amp1*std::cos(theta)
             + amp2*std::cos(2*theta)
             + amp3*std::cos(3*theta)
             + amp4*std::cos(4*theta)
             + amp5*std::cos(5*theta);
    }

    vector Uf(
        scalar d,
        scalar kx,
        scalar ky,
        scalar lambda,
        scalar T,
        scalar x,
        scalar y,
        scalar t,
        scalar phase,
        scalar z) const
    {
        const scalar k = std::sqrt(kx*kx + ky*ky);
        const scalar a11 = stokesVA11(d, k);
        const scalar a13 = stokesVA13(d, k);
        const scalar a15 = stokesVA15(d, k);
        const scalar a22 = stokesVA22(d, k);
        const scalar a24 = stokesVA24(d, k);
        const scalar a33 = stokesVA33(d, k);
        const scalar a35 = stokesVA35(d, k);
        const scalar a44 = stokesVA44(d, k);
        const scalar a55 = stokesVA55(d, k);
        const scalar pi = wavePi;
        const scalar l2 = waveSqr(lambda);
        const scalar l3 = l2*lambda;
        const scalar l4 = l3*lambda;
        const scalar l5 = l4*lambda;
        const scalar a1u = 2*pi/T/k*(lambda*a11 + l3*a13 + l5*a15);
        const scalar a2u = 2*2*pi/T/k*(l2*a22 + l4*a24);
        const scalar a3u = 3*2*pi/T/k*(l3*a33 + l5*a35);
        const scalar a4u = 4*2*pi/T/k*(l4*a44);
        const scalar a5u = 5*2*pi/T/k*(l5*a55);
        const scalar theta = kx*x + ky*y - 2*pi/T*t + phase;
        scalar u = a1u*std::cosh(k*z)*std::cos(theta)
                 + a2u*std::cosh(2*k*z)*std::cos(2*theta)
                 + a3u*std::cosh(3*k*z)*std::cos(3*theta)
                 + a4u*std::cosh(4*k*z)*std::cos(4*theta)
                 + a5u*std::cosh(5*k*z)*std::cos(5*theta);
        const scalar w = a1u*std::sinh(k*z)*std::sin(theta)
                       + a2u*std::sinh(2*k*z)*std::sin(2*theta)
                       + a3u*std::sinh(3*k*z)*std::sin(3*theta)
                       + a4u*std::sinh(4*k*z)*std::sin(4*theta)
                       + a5u*std::sinh(5*k*z)*std::sin(5*theta);
        const scalar v = u*std::sin(waveAngle_);
        u *= std::cos(waveAngle_);
        return vector{u, v, w};
    }

    void setLevel(
        scalar t,
        scalar tCoeff,
        std::vector<scalar>& level) const override
    {
        const scalar waveK = scalar(2)*wavePi/waveLength_;
        const scalar waveKx = waveK*std::cos(waveAngle_);
        const scalar waveKy = waveK*std::sin(waveAngle_);
        for (std::size_t p = 0; p < level.size(); ++p)
        {
            const scalar e = eta(waterDepthRef_, waveKx, waveKy, lambda_, wavePeriod_, xPaddle_[p],
                                 yPaddle_[p], t, wavePhase_);
            level[p] = waterDepthRef_ + tCoeff*e;
        }
    }

    void setVelocity(
        scalar t,
        scalar tCoeff,
        const std::vector<scalar>& level) override
    {
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
            const vector U = Uf(waterDepthRef_, waveKx, waveKy, lambda_, wavePeriod_, xPaddle_[p],
                                yPaddle_[p], t, wavePhase_, z);
            U_[facei] = (fraction*U)*tCoeff;
        }
    }

private:
    scalar lambda_ = 0;
};

} // namespace


std::unique_ptr<WaveModel> makeStokesV(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<StokesV>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
