// OpenFOAM's waveModel, the host reference. See wave_model_cpp.cuh.
#include "wave_model_cpp.cuh"
#include "wave_generation_bases_cpp.cuh"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

const char* const WHO = "brae waveModel: ";
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
    nPaddle_ = static_cast<label>(requiredWaveScalar(d, "nPaddle", patchName_));
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


std::vector<std::pair<std::string, scalar>> WaveModel::info() const
{
    return {{"Reference water depth", waterDepthRef_}};
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
    // the run-time selection table: OpenFOAM's ten
    const std::pair<const char*, WaveModelMaker*> table[] =
    {
        {"StokesI", &makeStokesI},
        {"StokesII", &makeStokesII},
        {"StokesV", &makeStokesV},
        {"cnoidal", &makeCnoidal},
        {"streamFunction", &makeStreamFunction},
        {"irregularMultiDirectional", &makeIrregularMultiDirectional},
        {"Boussinesq", &makeBoussinesq},
        {"Grimshaw", &makeGrimshaw},
        {"McCowan", &makeMcCowan},
        {"shallowWaterAbsorption", &makeShallowWaterAbsorption},
    };
    std::unique_ptr<WaveModel> w;
    for (const auto& entry : table)
    {
        if (model == entry.first)
        {
            w = entry.second(patch, m, g, gravity);
        }
    }
    if (!w)
    {
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patch.name + "` asks for waveModel `" + model + "`, which "
            "is none of OpenFOAM's ten (StokesI, StokesII, StokesV, cnoidal, streamFunction, "
            "irregularMultiDirectional, Boussinesq, Grimshaw, McCowan, shallowWaterAbsorption).");
    }
    w->mesh_ = &m;
    w->geometry_ = &g;
    w->readDict(*pd, alphaInternal);
    return w;
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
