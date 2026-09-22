// The wave generation models' base classes. See wave_generation_bases_cpp.cuh for the hierarchy.
#include "wave_generation_bases_cpp.cuh"
#include <stdexcept>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {
const char* const WHO = "brae waveModel: ";
} // namespace

const scalar wavePi = 3.14159265358979323846;


scalar requiredWaveScalar(
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
bool requiredWaveSwitch(
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


void WaveGenerationModel::readDict(
    const FoamDict& d,
    const std::vector<scalar>& alphaInternal)
{
    WaveModel::readDict(d, alphaInternal);
    activeAbsorption_ = requiredWaveSwitch(d, "activeAbsorption", patchName_);
}


scalar WaveGenerationModel::readWaveHeight(const FoamDict& d) const
{
    const scalar h = requiredWaveScalar(d, "waveHeight", patchName_);
    if (h < 0)
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName_ + "`: waveHeight must not be negative.");
    return h;
}


scalar WaveGenerationModel::readWaveAngle(const FoamDict& d) const
{
    // degToRad
    return requiredWaveScalar(d, "waveAngle", patchName_)*wavePi/scalar(180);
}


void IrregularWaveModel::readDict(
    const FoamDict& d,
    const std::vector<scalar>& alphaInternal)
{
    WaveGenerationModel::readDict(d, alphaInternal);
    rampTime_ = requiredWaveScalar(d, "rampTime", patchName_);
}


scalar IrregularWaveModel::timeCoeff(scalar t) const
{
    const scalar c = t/rampTime_;
    return c < scalar(0) ? scalar(0) : (c > scalar(1) ? scalar(1) : c);
}


void RegularWaveModel::readDict(
    const FoamDict& d,
    const std::vector<scalar>& alphaInternal)
{
    IrregularWaveModel::readDict(d, alphaInternal);
    waveHeight_ = readWaveHeight(d);
    waveAngle_ = readWaveAngle(d);
    wavePeriod_ = requiredWaveScalar(d, "wavePeriod", patchName_);
    if (wavePeriod_ < 0)
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName_ + "`: wavePeriod must not be negative.");
    wavePhase_ = d.scalarOr("wavePhase", scalar(1.5)*wavePi);
}


void RegularWaveModel::regularInfo(std::vector<std::pair<std::string, scalar>>& out) const
{
    out.push_back({"Ramp time", rampTime_});
    out.push_back({"Wave height", waveHeight_});
    // radToDeg, as the log prints it
    out.push_back({"Wave angle", waveAngle_*scalar(180)/wavePi});
    out.push_back({"Wave period", wavePeriod_});
    out.push_back({"Wave length", waveLength_});
    out.push_back({"Wave phase", wavePhase_});
}


SolitaryWaveModel::SolitaryWaveModel(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
    : WaveGenerationModel(patch, m, g, gravity)
{
    // x_ = Cf.x*cos(waveAngle_) + Cf.y*sin(waveAngle_) with waveAngle_ still 0: the global x
    bool first = true;
    for (const vector& c : patch.Cf)
    {
        x0_ = first ? c.x : std::fmin(x0_, c.x);
        first = false;
    }
}


void SolitaryWaveModel::readDict(
    const FoamDict& d,
    const std::vector<scalar>& alphaInternal)
{
    WaveGenerationModel::readDict(d, alphaInternal);
    waveHeight_ = readWaveHeight(d);
    waveAngle_ = readWaveAngle(d);
}


void SolitaryWaveModel::solitaryInfo(std::vector<std::pair<std::string, scalar>>& out) const
{
    out.push_back({"Wave height", waveHeight_});
    out.push_back({"Wave angle", waveAngle_*scalar(180)/wavePi});
    out.push_back({"x0", x0_});
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
