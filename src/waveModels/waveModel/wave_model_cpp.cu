// OpenFOAM's waveModel, the host reference. See wave_model_cpp.cuh.
#include "wave_model_cpp.cuh"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

const char* const WHO = "brae waveModel: ";
const scalar PI = 3.14159265358979323846;
// OpenFOAM's SMALL and ROOTVSMALL in double precision
const scalar OF_SMALL = 1.0e-15;
const scalar OF_ROOTVSMALL = 1.0e-150;

// T & v, row by row
vector dotTV(
    const tensor& T,
    const vector& v)
{
    return vector{T.xx*v.x + T.xy*v.y + T.xz*v.z,
                  T.yx*v.x + T.yy*v.y + T.yz*v.z,
                  T.zx*v.x + T.zy*v.y + T.zz*v.z};
}

scalar requiredScalar(
    const FoamDict& d,
    const std::string& key,
    const std::string& patchName)
{
    if (!d.found(key))
        throw std::runtime_error(
            std::string(WHO) + "waveProperties entry for patch `" + patchName + "` has no `" + key
            + "`. OpenFOAM reads it with readEntry and stops without it.");
    return d.scalarOr(key, scalar(0));
}

// OpenFOAM's Switch: the words it accepts as true or false, and a FatalIOError on anything else
bool requiredSwitch(
    const FoamDict& d,
    const std::string& key,
    const std::string& patchName)
{
    if (!d.found(key))
        throw std::runtime_error(
            std::string(WHO) + "waveProperties entry for patch `" + patchName + "` has no `" + key
            + "`. OpenFOAM reads it with readEntry and stops without it.");
    const std::string w = d.wordOr(key, "");
    if (w == "yes" || w == "true" || w == "on" || w == "y" || w == "t" || w == "1") return true;
    if (w == "no" || w == "false" || w == "off" || w == "n" || w == "f" || w == "0") return false;
    throw std::runtime_error(
        std::string(WHO) + "waveProperties entry for patch `" + patchName + "` has `" + key + " " + w
        + "`, which is not a Switch.");
}


// waveGenerationModel -> irregularWaveModel -> regularWaveModel -> StokesI
class StokesI : public WaveModel
{
public:
    StokesI(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : WaveModel(patch, m, g, gravity)
    {
        type_ = "StokesI";
    }

    scalar waveLength() const override { return waveLength_; }

    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override
    {
        WaveModel::readDict(d, alphaInternal);
        // waveGenerationModel::readDict
        activeAbsorption_ = requiredSwitch(d, "activeAbsorption", patchName_);
        // irregularWaveModel::readDict
        rampTime_ = requiredScalar(d, "rampTime", patchName_);
        // regularWaveModel::readDict
        waveHeight_ = requiredScalar(d, "waveHeight", patchName_);
        if (waveHeight_ < 0)
            throw std::runtime_error(std::string(WHO) + "waveHeight must not be negative.");
        // degToRad
        waveAngle_ = requiredScalar(d, "waveAngle", patchName_)*PI/scalar(180);
        wavePeriod_ = requiredScalar(d, "wavePeriod", patchName_);
        if (wavePeriod_ < 0)
            throw std::runtime_error(std::string(WHO) + "wavePeriod must not be negative.");
        wavePhase_ = d.scalarOr("wavePhase", scalar(1.5)*PI);
        // StokesI::readDict -- the dispersion relation by 100 fixed-point passes, no tolerance
        const scalar L0 = mag(g_)*wavePeriod_*wavePeriod_/(scalar(2)*PI);
        scalar L = L0;
        for (int i = 1; i <= 100; ++i)
        {
            L = L0*std::tanh(scalar(2)*PI*waterDepthRef_/L);
        }
        waveLength_ = L;
    }

protected:
    scalar timeCoeff(scalar t) const override
    {
        // clamp(t/rampTime_, zero_one{})
        const scalar c = t/rampTime_;
        return c < scalar(0) ? scalar(0) : (c > scalar(1) ? scalar(1) : c);
    }

    scalar eta(
        scalar H,
        scalar Kx,
        scalar x,
        scalar Ky,
        scalar y,
        scalar omega,
        scalar t,
        scalar phase) const
    {
        const scalar phaseTot = Kx*x + Ky*y - omega*t + phase;
        return H*scalar(0.5)*std::cos(phaseTot);
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

    void setLevel(
        scalar t,
        scalar tCoeff,
        std::vector<scalar>& level) const override
    {
        const scalar waveOmega = scalar(2)*PI/wavePeriod_;
        const scalar waveK = scalar(2)*PI/waveLength_;
        const scalar waveKx = waveK*std::cos(waveAngle_);
        const scalar waveKy = waveK*std::sin(waveAngle_);
        for (std::size_t p = 0; p < level.size(); ++p)
        {
            const scalar e = eta(waveHeight_, waveKx, xPaddle_[p], waveKy, yPaddle_[p], waveOmega, t,
                                 wavePhase_);
            level[p] = waterDepthRef_ + tCoeff*e;
        }
    }

    void setVelocity(
        scalar t,
        scalar tCoeff,
        const std::vector<scalar>& level) override
    {
        const scalar waveOmega = scalar(2)*PI/wavePeriod_;
        const scalar waveK = scalar(2)*PI/waveLength_;
        const scalar waveKx = waveK*std::cos(waveAngle_);
        const scalar waveKy = waveK*std::sin(waveAngle_);
        for (label facei = 0; facei < patch_.size; ++facei)
        {
            scalar fraction = 1;
            scalar z = 0;
            setPaddlePropeties(level, facei, fraction, z);
            if (!(fraction > 0)) continue;
            const label p = faceToPaddle_[facei];
            const vector Uf = UfBase(waveHeight_, waterDepthRef_, waveKx, xPaddle_[p], waveKy,
                                     yPaddle_[p], waveOmega, t, wavePhase_, z);
            U_[facei] = (fraction*Uf)*tCoeff;
        }
    }

private:
    scalar rampTime_ = 0;
    scalar waveHeight_ = 0;
    scalar waveAngle_ = 0;
    scalar wavePeriod_ = 0;
    scalar waveLength_ = 0;
    scalar wavePhase_ = 0;
};


// waveAbsorptionModel -> shallowWaterAbsorption
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

    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override
    {
        WaveModel::readDict(d, alphaInternal);
        // waveAbsorptionModel::readDict: "always set to true"
        activeAbsorption_ = true;
    }

protected:
    // "No time ramping applied for absorption"
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


WaveModel::WaveModel(
    const FvPatch& patch,
    const PrimitiveMesh&,
    const FvGeometry&,
    const vector& gravity)
    : patch_(patch), patchName_(patch.name), g_(gravity), Rgl_{1, 0, 0, 0, 1, 0, 0, 0, 1},
      Rlg_{1, 0, 0, 0, 1, 0, 0, 0, 1}
{
    U_.assign(static_cast<std::size_t>(patch.size), vector{0, 0, 0});
    alpha_.assign(static_cast<std::size_t>(patch.size), scalar(0));
}


void WaveModel::initialiseGeometry(
    const PrimitiveMesh& m,
    const FvGeometry& g)
{
    const label n = patch_.size;
    // x: streamwise, the INWARD patch normal; z: up; y = z ^ x
    vector sumSf{0, 0, 0};
    for (label i = 0; i < n; ++i)
    {
        sumSf = sumSf + g.Sf()[patch_.start + i];
    }
    const vector avg = sumSf/static_cast<scalar>(n);
    const vector negAvg{-avg.x, -avg.y, -avg.z};
    const scalar s = mag(negAvg);
    const vector x = (s < OF_ROOTVSMALL) ? vector{0, 0, 0} : negAvg/s;
    const scalar mg = mag(g_);
    const vector z{-g_.x/mg, -g_.y/mg, -g_.z/mg};
    const vector y = cross(z, x);
    // tensor(x, y, z): the three vectors are its ROWS
    Rlg_ = tensor{x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z};
    Rgl_ = transpose(Rlg_);

    // the patch's points in the local frame, and their bounding box
    const scalar big = std::numeric_limits<scalar>::max();
    vector lo{big, big, big};
    vector hi{-big, -big, -big};
    zMin_.assign(static_cast<std::size_t>(n), scalar(0));
    zMax_.assign(static_cast<std::size_t>(n), scalar(0));
    for (label i = 0; i < n; ++i)
    {
        const label f = patch_.start + i;
        for (label k = 0; k < m.faceSize(f); ++k)
        {
            const vector pl = dotTV(Rgl_, m.points()[m.faceVert(f, k)]);
            lo = vector{std::fmin(lo.x, pl.x), std::fmin(lo.y, pl.y), std::fmin(lo.z, pl.z)};
            hi = vector{std::fmax(hi.x, pl.x), std::fmax(hi.y, pl.y), std::fmax(hi.z, pl.z)};
            if (k == 0)
            {
                zMin_[i] = pl.z;
                zMax_[i] = pl.z;
            }
            else
            {
                zMin_[i] = std::fmin(zMin_[i], pl.z);
                zMax_[i] = std::fmax(zMax_[i], pl.z);
            }
        }
    }
    zSpan_ = hi.z - lo.z;

    const scalar xMid = lo.x + scalar(0.5)*(hi.x - lo.x);
    const scalar paddleDy = (hi.y - lo.y)/static_cast<scalar>(nPaddle_);
    xPaddle_.assign(static_cast<std::size_t>(nPaddle_), scalar(0));
    yPaddle_.assign(static_cast<std::size_t>(nPaddle_), scalar(0));
    for (label p = 0; p < nPaddle_; ++p)
    {
        xPaddle_[p] = xMid;
        yPaddle_[p] = static_cast<scalar>(p)*paddleDy + lo.y + scalar(0.5)*paddleDy;
    }

    z_.assign(static_cast<std::size_t>(n), scalar(0));
    faceToPaddle_.assign(static_cast<std::size_t>(n), label(-1));
    zMin0_ = big;
    for (label i = 0; i < n; ++i)
    {
        const vector cl = dotTV(Rgl_, g.Cf()[patch_.start + i]);
        z_[i] = cl.z;
        zMin0_ = std::fmin(zMin0_, zMin_[i]);
        faceToPaddle_[i] = static_cast<label>(std::floor((cl.y - lo.y)/paddleDy));
    }
}


std::vector<scalar> WaveModel::waterLevel(const std::vector<scalar>& alphaInternal) const
{
    std::vector<scalar> level(static_cast<std::size_t>(nPaddle_), initialDepth_);
    std::vector<scalar> paddleMagSf(static_cast<std::size_t>(nPaddle_), scalar(0));
    std::vector<scalar> paddleWettedMagSf(static_cast<std::size_t>(nPaddle_), scalar(0));
    for (label i = 0; i < patch_.size; ++i)
    {
        const label p = faceToPaddle_[i];
        paddleMagSf[p] += patch_.magSf[i];
        paddleWettedMagSf[p] += patch_.magSf[i]*alphaInternal[patch_.faceCells[i]];
    }
    for (label p = 0; p < nPaddle_; ++p)
    {
        level[p] += paddleWettedMagSf[p]*zSpan_/(paddleMagSf[p] + OF_ROOTVSMALL);
    }
    return level;
}


void WaveModel::setAlpha(const std::vector<scalar>& level)
{
    for (label facei = 0; facei < patch_.size; ++facei)
    {
        const scalar paddleCalc = level[faceToPaddle_[facei]];
        const scalar zMin0 = zMin_[facei] - zMin0_;
        const scalar zMax0 = zMax_[facei] - zMin0_;
        if (zMax0 < paddleCalc)
        {
            alpha_[facei] = 1.0;
        }
        else if (zMin0 > paddleCalc)
        {
            alpha_[facei] = 0.0;
        }
        else
        {
            const scalar dz = paddleCalc - zMin0;
            alpha_[facei] = dz/(zMax0 - zMin0);
        }
    }
}


void WaveModel::setPaddlePropeties(
    const std::vector<scalar>& level,
    label facei,
    scalar& fraction,
    scalar& z) const
{
    const scalar paddleCalc = level[faceToPaddle_[facei]];
    const scalar paddleHeight = std::fmin(paddleCalc, waterDepthRef_);
    const scalar zMin = zMin_[facei] - zMin0_;
    const scalar zMax = zMax_[facei] - zMin0_;

    fraction = 1;
    z = 0;
    if (zMax < paddleHeight)
    {
        z = z_[facei] - zMin0_;
    }
    else if (zMin > paddleCalc)
    {
        fraction = -1;
    }
    else
    {
        if (paddleCalc < waterDepthRef_)
        {
            if ((zMax > paddleCalc) && (zMin < paddleCalc))
            {
                const scalar dz = paddleCalc - zMin;
                fraction = dz/(zMax - zMin);
                z = z_[facei] - zMin0_;
            }
        }
        else
        {
            if (zMax < paddleCalc)
            {
                z = waterDepthRef_;
            }
            else if ((zMax > paddleCalc) && (zMin < paddleCalc))
            {
                const scalar dz = paddleCalc - zMin;
                fraction = dz/(zMax - zMin);
                z = waterDepthRef_;
            }
        }
    }
}


void WaveModel::readDict(
    const FoamDict& d,
    const std::vector<scalar>& alphaInternal)
{
    // `U` names the velocity field the model looks up; brae's interFoam has one, and it is U
    const std::string uName = d.wordOr("U", "U");
    if (uName != "U")
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName_ + "` names the velocity field `" + uName
            + "`; the solver's is `U`.");
    nPaddle_ = static_cast<label>(requiredScalar(d, "nPaddle", patchName_));
    if (nPaddle_ < 1)
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName_ + "`: nPaddle must be greater than zero.");
    initialDepth_ = d.scalarOr("initialDepth", scalar(0));
    // "Need to initialise the geometry before calling waterLevel()" -- and after nPaddle is known
    initialiseGeometry(*mesh_, *geometry_);

    if (d.found("waterDepthRef"))
    {
        waterDepthRef_ = d.scalarOr("waterDepthRef", scalar(0));
        return;
    }
    if (d.found("waterDepth"))
    {
        waterDepthRef_ = d.scalarOr("waterDepth", scalar(0));
    }
    else
    {
        // from the water standing against the patch NOW
        waterDepthRef_ = waterLevel(alphaInternal).front();
    }
    // "Avoid potential zero..."
    waterDepthRef_ += OF_SMALL;
}


bool WaveModel::correct(
    scalar t,
    label timeIndex,
    const std::vector<scalar>& alphaInternal,
    const std::vector<vector>& UInternal)
{
    if (timeIndex == currTimeIndex_) return false;
    alphaNow_ = &alphaInternal;
    UNow_ = &UInternal;

    const scalar tCoeff = timeCoeff(t);
    U_.assign(static_cast<std::size_t>(patch_.size), vector{0, 0, 0});
    alpha_.assign(static_cast<std::size_t>(patch_.size), scalar(0));
    std::vector<scalar> calculatedLevel(static_cast<std::size_t>(nPaddle_), scalar(0));
    if (patch_.size > 0)
    {
        setLevel(t, tCoeff, calculatedLevel);
        setVelocity(t, tCoeff, calculatedLevel);
        setAlpha(calculatedLevel);
    }

    if (activeAbsorption_)
    {
        const std::vector<scalar> activeLevel = waterLevel(alphaInternal);
        for (label facei = 0; facei < patch_.size; ++facei)
        {
            const label p = faceToPaddle_[facei];
            if (zMin_[facei] - zMin0_ < activeLevel[p])
            {
                const scalar UCorr = (calculatedLevel[p] - activeLevel[p])
                                   *std::sqrt(mag(g_)/activeLevel[p]);
                U_[facei].x += UCorr;
            }
            else
            {
                U_[facei].x = 0;
            }
        }
    }

    // "Transform velocity into global coordinate system"
    for (vector& u : U_)
    {
        u = dotTV(Rlg_, u);
    }
    currTimeIndex_ = timeIndex;
    alphaNow_ = nullptr;
    UNow_ = nullptr;
    return true;
}


void requireAlphaName(
    const FoamDict& patchDict,
    const std::string& patchName,
    const std::string& alphaName)
{
    // waveModel's alphaName_ defaults to "alpha" and the model looks the field up by it
    const std::string named = patchDict.wordOr("alpha", "alpha");
    if (named == alphaName) return;
    throw std::runtime_error(
        std::string(WHO) + "waveProperties entry for patch `" + patchName + "` names the phase "
        "fraction `" + named + "` and the solver's is `" + alphaName + "`. OpenFOAM looks the field "
        "up by that name and stops when there is none.");
}


std::unique_ptr<WaveModel> WaveModel::New(
    const FoamDict& waveProperties,
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity,
    const std::string& alphaName,
    const std::vector<scalar>& alphaInternal)
{
    const FoamDict* pd = waveProperties.subDict(patch.name);
    if (!pd)
        throw std::runtime_error(
            std::string(WHO) + "waveProperties has no entry for patch `" + patch.name
            + "`. OpenFOAM stops on it: \"Dictionary entry for patch ... not found\".");
    requireAlphaName(*pd, patch.name, alphaName);
    const std::string model = pd->wordOr("waveModel", "");
    std::unique_ptr<WaveModel> w;
    if (model == "StokesI")
    {
        w = std::make_unique<StokesI>(patch, m, g, gravity);
    }
    else if (model == "shallowWaterAbsorption")
    {
        w = std::make_unique<ShallowWaterAbsorption>(patch, m, g, gravity);
    }
    else
    {
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patch.name + "` asks for waveModel `" + model + "`. brae "
            "has StokesI and shallowWaterAbsorption; StokesII, StokesV, cnoidal, Boussinesq, Grimshaw, "
            "McCowan, streamFunction, irregularMultiDirectional and the rest are different wave "
            "theories and are not substituted.");
    }
    w->mesh_ = &m;
    w->geometry_ = &g;
    w->readDict(*pd, alphaInternal);
    return w;
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
