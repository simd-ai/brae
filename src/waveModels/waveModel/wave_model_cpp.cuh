#pragma once
// brae::cpu::waveModels -- OpenFOAM's waveModel, what the waveAlpha and waveVelocity boundary conditions
// evaluate. The host reference.
//
// provenance:
//   openfoam:  src/waveModels/waveModel/waveModel.C:55-128 (initialiseGeometry), :131-166 (waterLevel),
//                  :169-193 (setAlpha), :196-248 (setPaddlePropeties), :300-349 (readDict),
//                  :352-406 (correct)
//              src/waveModels/waveModel/waveModelNew.C:33-100 (New, lookupOrCreate)
//              src/waveModels/waveGenerationModels/base/{waveGenerationModel,irregularWaveModel,
//                  regularWaveModel}/*.C
//              src/waveModels/waveGenerationModels/derived/StokesI/StokesIWaveModel.C
//              src/waveModels/waveAbsorptionModels/derived/shallowWaterAbsorption/shallowWaterAbsorption.C
//              src/waveModels/derivedFvPatchFields/{waveAlpha,waveVelocity}/*.C
//   tests:     tests/interfoam_waves_vs_openfoam.sh, against real OpenFOAM's fields, its log's
//              "Reference water depth" and "Wave length", and its ORDER of "Updating ... wave model" lines
//
// ONE MODEL PER PATCH, SHARED BY BOTH BOUNDARY CONDITIONS. waveAlpha and waveVelocity each call
// lookupOrCreate and then model.correct(t), and correct() does its work ONCE PER TIME INDEX
// (waveModel.C:354). Whichever condition reaches it first in a time index decides WHAT STATE THE MODEL
// READS, and the model reads two things from the solver: alpha's patchInternalField, for the active
// absorption's water level, and -- in shallowWaterAbsorption -- U's. So the answer depends on when
// correct() fires, which is the caller's to get right and is why the time index is an argument here:
//
//   the time index is NOT the time step. Inside an alpha sub-cycle Time::subCycle (Time.C:1006) sets it
//   to (n-1)*nSubCycles and ++ moves it, so every sub-cycle is a new index and the model updates in
//   each, at the SUB-CYCLE's time. After endSubCycle the index is n again -- a different number from the
//   last sub-cycle's -- so U's first updateCoeffs of the step updates it once more, at the step's time
//   and from the alpha the sub-cycles left. OpenFOAM's log says so: on stokesI, per step, three
//   "Updating StokesI wave model for patch inlet" each ahead of its "MULES: Solving", then a fourth.
//
// THE LOCAL FRAME IS TRANSCRIBED, NOT DERIVED. x is the inward patch normal, z is up, y = z ^ x;
// Rlg_ = tensor(x, y, z) puts them in ROWS, Rgl_ is its transpose, points go to the local frame through
// Rgl_ and velocities come back through Rlg_. For an axis-aligned patch the tensor is diagonal and the
// two are the same matrix; whether OpenFOAM's pairing is the one its names promise on a patch that is
// not is not something this port decides.
//
// THE MODELS are OpenFOAM's nine generation models and its one absorption model, one file each under
// waveGenerationModels/ and waveAbsorptionModels/ as OpenFOAM lays them out; this file is the base
// class and the selector. REFUSED, by name: a model outside those ten, and a restart -- OpenFOAM
// re-reads <startTime>/uniform/waveProperties.<patch> for the reference depth it stored, and brae
// does not.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace brae {
namespace cpu {
namespace waveModels {

class WaveModel
{
public:
    virtual ~WaveModel() = default;

    // waveModel::New for one patch: the patch's sub-dictionary of constant/<waveDict>, the model it
    // names, and readDict -- which needs alpha, because a case that names no reference depth takes
    // it from the water standing against the patch AT THAT MOMENT.
    static std::unique_ptr<WaveModel> New(
        const FoamDict& waveProperties,
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity,
        const std::string& alphaName,
        const std::vector<scalar>& alphaInternal);

    // waveModel::correct(t). Does nothing when `timeIndex` is the one it last ran at. Returns
    // whether it ran, which is what OpenFOAM's "Updating ..." log line reports.
    bool correct(
        scalar t,
        label timeIndex,
        const std::vector<scalar>& alphaInternal,
        const std::vector<vector>& UInternal);

    const std::vector<vector>& U() const { return U_; }
    const std::vector<scalar>& alpha() const { return alpha_; }
    const std::string& type() const { return type_; }
    const std::string& patchName() const { return patchName_; }
    scalar waterDepthRef() const { return waterDepthRef_; }
    bool activeAbsorption() const { return activeAbsorption_; }
    const tensor& Rlg() const { return Rlg_; }
    // waveModel::info, as far as it prints NUMBERS: each (label, value) pair is one line of the block
    // OpenFOAM writes to its log when it creates the model, label for label, so a gate can hold every
    // derived constant -- reference depth, wave length, StokesV's lambda, cnoidal's m, a solitary
    // wave's x0 -- against OpenFOAM's own without knowing which model it is looking at.
    virtual std::vector<std::pair<std::string, scalar>> info() const;

protected:
    WaveModel(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity);

    // waveModel::readDict, and each derived class's on top of it
    virtual void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal);

    virtual scalar timeCoeff(scalar t) const = 0;
    virtual void setLevel(
        scalar t,
        scalar tCoeff,
        std::vector<scalar>& level) const = 0;
    virtual void setVelocity(
        scalar t,
        scalar tCoeff,
        const std::vector<scalar>& level) = 0;
    virtual void setAlpha(const std::vector<scalar>& level);

    void initialiseGeometry(
        const PrimitiveMesh& m,
        const FvGeometry& g);
    std::vector<scalar> waterLevel(const std::vector<scalar>& alphaInternal) const;
    void setPaddlePropeties(
        const std::vector<scalar>& level,
        label facei,
        scalar& fraction,
        scalar& z) const;

    const FvPatch& patch_;
    // held for readDict, which initialises the geometry only once nPaddle is known
    const PrimitiveMesh* mesh_ = nullptr;
    const FvGeometry* geometry_ = nullptr;
    std::string patchName_;
    std::string type_;
    vector g_;
    tensor Rgl_;
    tensor Rlg_;
    label nPaddle_ = 1;
    std::vector<scalar> xPaddle_;
    std::vector<scalar> yPaddle_;
    // local face-centre heights, and each face's local z extent
    std::vector<scalar> z_;
    scalar zSpan_ = 0;
    std::vector<scalar> zMin_;
    std::vector<scalar> zMax_;
    scalar zMin0_ = 0;
    std::vector<label> faceToPaddle_;
    scalar waterDepthRef_ = 0;
    scalar initialDepth_ = 0;
    label currTimeIndex_ = -1;
    bool activeAbsorption_ = false;
    std::vector<vector> U_;
    std::vector<scalar> alpha_;
    // the solver's fields as they stand while correct() runs -- shallowWaterAbsorption reads both
    const std::vector<scalar>* alphaNow_ = nullptr;
    const std::vector<vector>* UNow_ = nullptr;
};

// waveProperties' patch entry names `alpha`; it has to be the solver's own field, because OpenFOAM
// looks the field up by that name and stops when there is none.
void requireAlphaName(
    const FoamDict& patchDict,
    const std::string& patchName,
    const std::string& alphaName);

} // namespace waveModels
} // namespace cpu
} // namespace brae
