#pragma once
// The wave GENERATION models' base classes, and the factory function each derived model exposes.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/base/waveGenerationModel/waveGenerationModel.C
//              src/waveModels/waveGenerationModels/base/irregularWaveModel/irregularWaveModel.C
//              src/waveModels/waveGenerationModels/base/regularWaveModel/regularWaveModel.C
//              src/waveModels/waveGenerationModels/base/solitaryWaveModel/solitaryWaveModel.C
//              src/waveModels/waveGenerationModels/derived/StokesI/StokesIWaveModel.{H,C}
//   tests:     tests/interfoam_waves_vs_openfoam.sh
//
// THE HIERARCHY IS OpenFOAM's, because each level reads its own dictionary entries and a model gets
// exactly the entries of the levels above it:
//
//   waveModel                     U, alpha, nPaddle, initialDepth, waterDepthRef | waterDepth
//     waveGenerationModel         activeAbsorption                        (mandatory)
//       irregularWaveModel        rampTime                                (mandatory); ramps 0 -> 1
//         regularWaveModel        waveHeight, waveAngle, wavePeriod       (mandatory), wavePhase
//           StokesI               -- the wave length, from the linear dispersion relation
//             StokesII, StokesV
//           cnoidal, streamFunction
//         irregularMultiDirectional
//       solitaryWaveModel         waveHeight, waveAngle; NO ramp, timeCoeff is 1
//         Boussinesq, Grimshaw, McCowan
//
// regularWaveModel IS AN irregularWaveModel in OpenFOAM, not the other way round; that is where the
// ramp lives, so it is kept.
//
// StokesI is declared here rather than in its own file because StokesII and StokesV derive from it.
#include "wave_model_cpp.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace waveModels {

// the reads every level makes with readEntry / get<>: mandatory, and OpenFOAM stops without them
scalar requiredWaveScalar(
    const FoamDict& d,
    const std::string& key,
    const std::string& patchName);
bool requiredWaveSwitch(
    const FoamDict& d,
    const std::string& key,
    const std::string& patchName);
// constant::mathematical::pi
extern const scalar wavePi;

// sqr and pow3..pow6 COMPOSED AS OpenFOAM COMPOSES THEM (Scalar.H:338-365): pow4 is sqr(sqr(s)) and
// pow6 is pow3(sqr(s)), not four and six multiplications in a row, and the products round differently.
inline scalar waveSqr(scalar s) { return s*s; }
inline scalar wavePow3(scalar s) { return s*waveSqr(s); }
inline scalar wavePow4(scalar s) { return waveSqr(waveSqr(s)); }
inline scalar wavePow5(scalar s) { return s*wavePow4(s); }
inline scalar wavePow6(scalar s) { return wavePow3(waveSqr(s)); }

class WaveGenerationModel : public WaveModel
{
protected:
    using WaveModel::WaveModel;
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override;
    // readWaveHeight: refuses a negative one. readWaveAngle: degrees in the file, radians here.
    scalar readWaveHeight(const FoamDict& d) const;
    scalar readWaveAngle(const FoamDict& d) const;
};

class IrregularWaveModel : public WaveGenerationModel
{
protected:
    using WaveGenerationModel::WaveGenerationModel;
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override;
    // clamp(t/rampTime_, zero_one{})
    scalar timeCoeff(scalar t) const override;
    scalar rampTime_ = 0;
};

class RegularWaveModel : public IrregularWaveModel
{
protected:
    using IrregularWaveModel::IrregularWaveModel;
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override;
    void regularInfo(std::vector<std::pair<std::string, scalar>>& out) const;
    scalar waveHeight_ = 0;
    scalar waveAngle_ = 0;
    scalar wavePeriod_ = 0;
    // "to be set in derived classes"
    scalar waveLength_ = 0;
    scalar wavePhase_ = 0;
};

class SolitaryWaveModel : public WaveGenerationModel
{
protected:
    SolitaryWaveModel(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity);
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override;
    // "Ramping not applicable to solitary waves"
    scalar timeCoeff(scalar) const override { return scalar(1); }
    void solitaryInfo(std::vector<std::pair<std::string, scalar>>& out) const;
    scalar waveHeight_ = 0;
    scalar waveAngle_ = 0;
    // gMin of the patch's face-centre x -- the GLOBAL x, taken in the constructor while waveAngle_
    // is still 0 and never again (solitaryWaveModel.C:67-72). The models subtract it from the
    // paddle's LOCAL x. Transcribed as written.
    scalar x0_ = 0;
};

class StokesI : public RegularWaveModel
{
public:
    StokesI(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity);
    std::vector<std::pair<std::string, scalar>> info() const override;

protected:
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override;
    // the linear dispersion relation by 100 fixed-point passes, no tolerance
    virtual scalar waveLength(
        scalar h,
        scalar T) const;
    virtual vector UfBase(
        scalar H,
        scalar h,
        scalar Kx,
        scalar x,
        scalar Ky,
        scalar y,
        scalar omega,
        scalar t,
        scalar phase,
        scalar z) const;
    void setLevel(
        scalar t,
        scalar tCoeff,
        std::vector<scalar>& level) const override;
    void setVelocity(
        scalar t,
        scalar tCoeff,
        const std::vector<scalar>& level) override;
};

// one factory per model, each defined beside the model
using WaveModelMaker = std::unique_ptr<WaveModel>(
    const FvPatch&,
    const PrimitiveMesh&,
    const FvGeometry&,
    const vector&);

WaveModelMaker makeStokesI;
WaveModelMaker makeStokesII;
WaveModelMaker makeStokesV;
WaveModelMaker makeCnoidal;
WaveModelMaker makeStreamFunction;
WaveModelMaker makeIrregularMultiDirectional;
WaveModelMaker makeBoussinesq;
WaveModelMaker makeGrimshaw;
WaveModelMaker makeMcCowan;
WaveModelMaker makeShallowWaterAbsorption;

} // namespace waveModels
} // namespace cpu
} // namespace brae
