// irregularMultiDirectional -- a sea state: a sum of linear waves, each with its own period, height,
// phase and DIRECTION.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/irregularMultiDirectional/
//                  irregularMultiDirectionalWaveModel.C:52-66 (eta), :71-83 (waveLength),
//                  :86-130 (Uf), :133-166 (setLevel), :169-190 (uMultiDirec), :247-281 (readDict)
//   tests:     tests/interfoam_waves_vs_openfoam.sh (irregularMultiDirection)
//
// THE FOUR INPUTS ARE LISTS OF LISTS -- frequency band by direction -- and must be the same shape:
// OpenFOAM indexes all of them with one pair of loop counters taken from waveHeights. The tutorial
// carries 57 bands of 26 directions. Each component's wave length is the LINEAR dispersion relation's,
// by the same 100 fixed-point passes StokesI uses.
//
// It is an irregularWaveModel and NOT a regularWaveModel: it has a ramp and no waveHeight, waveAngle
// or wavePeriod of its own.
#include "wave_generation_bases_cpp.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace waveModels {

namespace {

using ScalarListList = std::vector<std::vector<scalar>>;

// A List<List<scalar>> from a dictionary entry's raw tokens: `N ( (a b ...) (c d ...) ... )`. brae's
// tokenizer may leave a parenthesis attached to a number or standing alone, so the entry is walked
// character by character; a number met at depth 1 is the list's own size prefix and is dropped.
ScalarListList readScalarListList(
    const FoamDict& d,
    const std::string& key,
    const std::string& patchName)
{
    const std::vector<std::string>* tokens = d.find(key);
    if (!tokens)
        throw std::runtime_error(
            "brae waveModel: waveProperties entry for patch `" + patchName + "` has no `" + key
            + "`. OpenFOAM reads it with readEntry and stops without it.");
    std::string text;
    for (const std::string& t : *tokens)
    {
        text += " " + t;
    }
    ScalarListList out;
    int depth = 0;
    std::string number;
    auto flush = [&]()
    {
        if (number.empty()) return;
        if (depth == 2)
        {
            out.back().push_back(std::stod(number));
        }
        number.clear();
    };
    for (char ch : text)
    {
        if (ch == '(')
        {
            flush();
            ++depth;
            if (depth == 2)
            {
                out.emplace_back();
            }
            continue;
        }
        if (ch == ')')
        {
            flush();
            --depth;
            continue;
        }
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
        {
            flush();
            continue;
        }
        number += ch;
    }
    if (depth != 0 || out.empty())
        throw std::runtime_error(
            "brae waveModel: `" + key + "` for patch `" + patchName + "` is not a list of lists.");
    return out;
}


class IrregularMultiDirectional : public IrregularWaveModel
{
public:
    IrregularMultiDirectional(
        const FvPatch& patch,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const vector& gravity)
        : IrregularWaveModel(patch, m, g, gravity)
    {
        type_ = "irregularMultiDirectional";
    }

    std::vector<std::pair<std::string, scalar>> info() const override
    {
        std::vector<std::pair<std::string, scalar>> out = WaveModel::info();
        out.push_back({"Ramp time", rampTime_});
        out.push_back({"Wave periods", static_cast<scalar>(periods_.size())});
        out.push_back({"Wave heights", static_cast<scalar>(heights_.size())});
        out.push_back({"Wave phases", static_cast<scalar>(phases_.size())});
        out.push_back({"Wave lengths", static_cast<scalar>(lengths_.size())});
        out.push_back({"Wave directions", static_cast<scalar>(dirs_.size())});
        return out;
    }

protected:
    void readDict(
        const FoamDict& d,
        const std::vector<scalar>& alphaInternal) override
    {
        IrregularWaveModel::readDict(d, alphaInternal);
        periods_ = readScalarListList(d, "wavePeriods", patchName_);
        heights_ = readScalarListList(d, "waveHeights", patchName_);
        phases_ = readScalarListList(d, "wavePhases", patchName_);
        dirs_ = readScalarListList(d, "waveDirs", patchName_);
        // OpenFOAM indexes all four by waveHeights' shape and reads past the end of a shorter one
        for (const ScalarListList* q : {&periods_, &phases_, &dirs_})
        {
            bool same = q->size() == heights_.size();
            for (std::size_t i = 0; same && i < heights_.size(); ++i)
            {
                same = (*q)[i].size() == heights_[i].size();
            }
            if (!same)
                throw std::runtime_error(
                    "brae waveModel: patch `" + patchName_ + "`: wavePeriods, waveHeights, wavePhases "
                    "and waveDirs are not the same shape. OpenFOAM indexes all four with waveHeights' "
                    "counters and would read outside the shorter ones.");
        }
        lengths_ = heights_;
        for (std::size_t i = 0; i < heights_.size(); ++i)
        {
            for (std::size_t j = 0; j < heights_[i].size(); ++j)
            {
                lengths_[i][j] = waveLength(waterDepthRef_, periods_[i][j]);
                // degToRad
                dirs_[i][j] = dirs_[i][j]*wavePi/scalar(180);
            }
        }
    }

    scalar waveLength(
        scalar h,
        scalar T) const
    {
        const scalar L0 = mag(g_)*T*T/(scalar(2)*wavePi);
        scalar L = L0;
        for (int i = 1; i <= 100; ++i)
        {
            L = L0*std::tanh(scalar(2)*wavePi*h/L);
        }
        return L;
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

    vector uMultiDirec(
        scalar irregH,
        scalar irregWaveOmega,
        scalar pha,
        scalar irregWaveKs,
        scalar zz,
        scalar hh,
        scalar irregDir) const
    {
        const scalar ksh = irregWaveKs*hh;
        const scalar ksz = irregWaveKs*zz;
        const scalar u = irregH*scalar(0.5)*irregWaveOmega*std::cos(pha)*(std::cosh(ksz)/std::sinh(ksh))
                        *std::cos(irregDir);
        const scalar v = irregH*scalar(0.5)*irregWaveOmega*std::cos(pha)*(std::cosh(ksz)/std::sinh(ksh))
                        *std::sin(irregDir);
        const scalar w = irregH*scalar(0.5)*irregWaveOmega*std::sin(pha)*(std::sinh(ksz)/std::sinh(ksh));
        return vector{u, v, w};
    }

    vector Uf(
        scalar h,
        scalar x,
        scalar y,
        scalar t,
        scalar z) const
    {
        scalar u = 0.0;
        scalar v = 0.0;
        scalar w = 0.0;
        for (std::size_t i = 0; i < heights_.size(); ++i)
        {
            for (std::size_t j = 0; j < heights_[i].size(); ++j)
            {
                const scalar waveKs = scalar(2)*wavePi/lengths_[i][j];
                const scalar waveOmegas = scalar(2)*wavePi/periods_[i][j];
                const scalar phaseTot = waveKs*x*std::cos(dirs_[i][j])
                                      + waveKs*y*std::sin(dirs_[i][j])
                                      - waveOmegas*t
                                      + phases_[i][j];
                const vector U = uMultiDirec(heights_[i][j], waveOmegas, phaseTot, waveKs, z, h,
                                             dirs_[i][j]);
                u += U.x;
                v += U.y;
                w += U.z;
            }
        }
        return vector{u, v, w};
    }

    void setLevel(
        scalar t,
        scalar tCoeff,
        std::vector<scalar>& level) const override
    {
        for (std::size_t p = 0; p < level.size(); ++p)
        {
            scalar e = 0;
            for (std::size_t i = 0; i < heights_.size(); ++i)
            {
                for (std::size_t j = 0; j < heights_[i].size(); ++j)
                {
                    const scalar waveKs = scalar(2)*wavePi/lengths_[i][j];
                    const scalar waveOmegas = scalar(2)*wavePi/periods_[i][j];
                    e += eta(heights_[i][j], waveKs*std::cos(dirs_[i][j]), xPaddle_[p],
                             waveKs*std::sin(dirs_[i][j]), yPaddle_[p], waveOmegas, t, phases_[i][j]);
                }
            }
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
            const vector U = Uf(waterDepthRef_, xPaddle_[p], yPaddle_[p], t, z);
            U_[facei] = (fraction*U)*tCoeff;
        }
    }

private:
    ScalarListList periods_;
    ScalarListList heights_;
    ScalarListList phases_;
    ScalarListList lengths_;
    ScalarListList dirs_;
};

} // namespace


std::unique_ptr<WaveModel> makeIrregularMultiDirectional(
    const FvPatch& patch,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const vector& gravity)
{
    return std::make_unique<IrregularMultiDirectional>(patch, m, g, gravity);
}

} // namespace waveModels
} // namespace cpu
} // namespace brae
