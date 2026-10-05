// cnoidal -- the shallow-water periodic wave, in Jacobi elliptic functions.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/cnoidal/cnoidalWaveModel.C:52-198,
//                  :252-340
//   tests:     tests/interfoam_waves_vs_openfoam.sh (cnoidal): the log's `Cnoidal m parameter` and
//              `Wave length`, both of which come out of the scan below
//
// THE PARAMETER m IS FOUND BY A SCAN, not a solve: from 0.5 upward in steps of 1e-4 to 1, keeping the m
// whose period is closest to the case's. So m is known to four digits, the wave length with it, and
// the scan's `<=` keeps the LAST of equal errors. Transcribed: a root-finder would return a different
// wave.
//
// THE VELOCITY ASKS FOR THE WAVE'S MEAN SQUARE ELEVATION AT EVERY FACE. Uf calls etaMeanSq, which sums 1000
// elevations over a period, and OpenFOAM does that per face, per update. It is a constant of the wave -- a
// function of H, m and T alone -- so it is summed ONCE, in OpenFOAM's order, and the number kept with the three
// it was summed for: the same call gives the same bits whether it is made again or remembered. (What would
// change the last digit is summing the 1000 terms in another order, and nothing here does.)
// MEASURED on laminar/waves/cnoidal (52,500 cells), 2026-10-05: about 29 of the step's 44.5 ms.
//   BRAE_CONTROL_CNOIDAL_MEAN_RECOMPUTED=1   summed at every call, as before
//   BRAE_CONTROL_CNOIDAL_MEAN_CHECK=1        summed at every call as well, and the kept number held to it
//   BRAE_CONTROL_CNOIDAL_MEAN_SHORT=1        a gate's CONTROL, deliberately wrong: the kept number is the mean
//                                            of 999 elevations
#include "wave_generation_bases_cpp.cuh"
#include "wave_elliptic_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

class Cnoidal : public RegularWaveModel
{
public:
    Cnoidal(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : RegularWaveModel(patch, m, g, gravity)
    {
        type_ = "cnoidal";
    }

    std::vector<std::pair<std::string, scalar>> info() const override
    {
        std::vector<std::pair<std::string, scalar>> out = WaveModel::info();
        regularInfo(out);
        out.push_back({"Cnoidal m parameter", m_});
        return out;
    }

protected:
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override
    {
        RegularWaveModel::readDict(d, alphaInternal);
        initialise(waveHeight_, waterDepthRef_, wavePeriod_, m_, waveLength_);
    }

    void initialise(
        scalar H,
        scalar d,
        scalar T,
        scalar& mOut,
        scalar& LOut) const
    {
        const scalar mTolerance = 0.0001;
        scalar mElliptic = 0.5;
        scalar LElliptic = 0;
        scalar phaseSpeed = 0;
        scalar mError = 0.0;
        // GREAT
        scalar mMinError = 1.0e+15;
        while (mElliptic < 1.0)
        {
            scalar KElliptic = 0;
            scalar EElliptic = 0;
            elliptic::ellipticIntegralsKE(mElliptic, KElliptic, EElliptic);
            LElliptic = KElliptic*std::sqrt(16.0*wavePow3(d)*mElliptic/(3.0*H));
            phaseSpeed = std::sqrt(mag(g_)*d)
                        *(1.0 - H/d/2.0 + H/d/mElliptic*(1.0 - 3.0/2.0*EElliptic/KElliptic));
            mError = std::fabs(T - LElliptic/phaseSpeed);
            if (mError <= mMinError)
            {
                mOut = mElliptic;
                LOut = LElliptic;
                mMinError = mError;
            }
            mElliptic += mTolerance;
        }
    }

    scalar eta(
        scalar H,
        scalar m,
        scalar kx,
        scalar ky,
        scalar T,
        scalar x,
        scalar y,
        scalar t) const
    {
        scalar K = 0;
        scalar E = 0;
        elliptic::ellipticIntegralsKE(m, K, E);
        const scalar uCnoidal = K/wavePi*(kx*x + ky*y - 2.0*wavePi*t/T);
        scalar sn = 0;
        scalar cn = 0;
        scalar dn = 0;
        elliptic::JacobiSnCnDn(uCnoidal, m, sn, cn, dn);
        return H*((1.0 - E/K)/m - 1.0 + waveSqr(cn));
    }

    scalar eta1D(
        scalar H,
        scalar m,
        scalar t,
        scalar T) const
    {
        scalar K = 0;
        scalar E = 0;
        elliptic::ellipticIntegralsKE(m, K, E);
        const scalar uCnoidal = -2.0*K*(t/T);
        scalar sn = 0;
        scalar cn = 0;
        scalar dn = 0;
        elliptic::JacobiSnCnDn(uCnoidal, m, sn, cn, dn);
        return H*((1.0 - E/K)/m - 1.0 + waveSqr(cn));
    }

    // cnoidalWaveModel.C's etaMeanSq, term for term: the mean of n squared elevations over a period (n is 1000;
    // the gate's control asks for 999)
    scalar etaMeanSqSummed(
        scalar H,
        scalar m,
        scalar T,
        int n) const
    {
        scalar e = 0;
        scalar etaSumSq = 0;
        for (int i = 0; i < n; i++)
        {
            e = eta1D(H, m, i*T/(1000.0), T);
            etaSumSq += e*e;
        }
        etaSumSq /= 1000.0;
        return etaSumSq;
    }

    scalar etaMeanSq(
        scalar H,
        scalar m,
        scalar T) const
    {
        static const bool recomputed = std::getenv("BRAE_CONTROL_CNOIDAL_MEAN_RECOMPUTED") != nullptr;
        static const bool check = std::getenv("BRAE_CONTROL_CNOIDAL_MEAN_CHECK") != nullptr;
        static const bool shortSum = std::getenv("BRAE_CONTROL_CNOIDAL_MEAN_SHORT") != nullptr;
        if (recomputed) return etaMeanSqSummed(H, m, T, 1000);
        const scalar key[3] = {H, m, T};
        if (!meanHeld_ || std::memcmp(key, meanKey_, sizeof(key)) != 0)
        {
            // said at EVERY sum, not once a run: a wave whose H, m and T do not change is summed once a patch,
            // so the count of this line in a log is the count of sums (the gate holds it to one)
            std::printf("  cnoidal: the wave's mean square elevation is summed once and kept; "
                        "BRAE_CONTROL_CNOIDAL_MEAN_RECOMPUTED=1 sums it at every face\n");
            if (shortSum)
            {
                std::printf("  *** CONTROL MODE: the kept mean is of 999 elevations, not 1000. This run is "
                            "deliberately wrong. ***\n");
            }
            meanValue_ = etaMeanSqSummed(H, m, T, shortSum ? 999 : 1000);
            std::memcpy(meanKey_, key, sizeof(key));
            meanHeld_ = true;
        }
        if (check)
        {
            const scalar want = etaMeanSqSummed(H, m, T, 1000);
            if (std::memcmp(&want, &meanValue_, sizeof(scalar)) != 0)
            {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                              "brae cnoidal: BRAE_CONTROL_CNOIDAL_MEAN_CHECK: the kept mean square elevation is "
                              "%.17g and summing it gives %.17g.", meanValue_, want);
                throw std::runtime_error(buf);
            }
        }
        return meanValue_;
    }

    vector dEtaDx(
        scalar H,
        scalar m,
        scalar uCnoidal,
        scalar L,
        scalar K,
        scalar E) const
    {
        (void)E;
        const scalar dudx = 2.0*K/L;
        const scalar dudxx = 2.0*K/L*dudx;
        const scalar dudxxx = 2.0*K/L*dudxx;
        scalar sn = 0;
        scalar cn = 0;
        scalar dn = 0;
        elliptic::JacobiSnCnDn(uCnoidal, m, sn, cn, dn);
        const scalar d1 = -2.0*H*cn*dn*sn*dudx;
        const scalar d2 = 2.0*H*(dn*dn*sn*sn - cn*cn*dn*dn + m*cn*cn*sn*sn)*dudxx;
        const scalar d3 = 8.0*H
                         *(
                              cn*sn*dn*dn*dn*(-4.0 - 2.0*m)
                            + 4.0*m*cn*sn*sn*sn*dn
                            - 2.0*m*cn*cn*cn*sn*dn
                          )
                         *dudxxx;
        return vector{d1, d2, d3};
    }

    vector Uf(
        scalar H,
        scalar h,
        scalar m,
        scalar kx,
        scalar ky,
        scalar T,
        scalar x,
        scalar y,
        scalar t,
        scalar z) const
    {
        scalar K = 0;
        scalar E = 0;
        elliptic::ellipticIntegralsKE(m, K, E);
        const scalar uCnoidal = K/wavePi*(kx*x + ky*y - 2.0*wavePi*t/T);
        const scalar k = std::sqrt(kx*kx + ky*ky);
        const scalar L = 2.0*wavePi/k;
        const scalar c = L/T;
        const scalar etaCN = eta(H, m, kx, ky, T, x, y, t);
        const vector etaX = dEtaDx(H, m, uCnoidal, L, K, E);
        const scalar etaMS = etaMeanSq(H, m, T);
        scalar u = c*etaCN/h
                 - c*(etaCN*etaCN/h/h + etaMS*etaMS/h/h)
                 + 1.0/2.0*c*h*(1.0/3.0 - z*z/h/h)*etaX.y;
        const scalar w = -c*z*(etaX.x/h*(1.0 - 2.0*etaCN/h) + 1.0/6.0*h*(1.0 - z*z/h/h)*etaX.z);
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
            const scalar e = eta(waveHeight_, m_, waveKx, waveKy, wavePeriod_, xPaddle_[p], yPaddle_[p], t);
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
            const vector U = Uf(waveHeight_, waterDepthRef_, m_, waveKx, waveKy, wavePeriod_,
                                xPaddle_[p], yPaddle_[p], t, z);
            U_[facei] = (fraction*U)*tCoeff;
        }
    }

private:
    scalar m_ = 0;
    // etaMeanSq's number and the (H, m, T) it was summed for: a memo, so mutable
    mutable scalar meanKey_[3] = {0, 0, 0};
    mutable scalar meanValue_ = 0;
    mutable bool meanHeld_ = false;
};

} // namespace


std::unique_ptr<WaveModel> makeCnoidal(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<Cnoidal>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
