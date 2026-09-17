// interFoam's wave boundary conditions. See inter_waves_cpp.cuh.
#include "inter_waves_cpp.cuh"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

const char* const WHO = "brae interFoam: ";

// The boundary condition reads `waveDict` (default waveProperties) and brae's field reader does not
// keep the key, so the file is searched for it. OpenFOAM WRITES the name back as `waveDictName`, which
// it never reads; only the first spelling selects a dictionary.
void refuseOtherWaveDict(const std::string& fieldPath)
{
    std::ifstream in(fieldPath);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    for (std::size_t p = text.find("waveDict"); p != std::string::npos; p = text.find("waveDict", p + 1))
    {
        const std::size_t e = p + 8;
        if (e < text.size() && (std::isalnum(static_cast<unsigned char>(text[e])) || text[e] == '_')) continue;
        std::size_t b = text.find_first_not_of(" \t", e);
        const std::size_t semi = text.find(';', e);
        if (b == std::string::npos || semi == std::string::npos) continue;
        const std::string name = text.substr(b, semi - b);
        if (name == "waveProperties") continue;
        throw std::runtime_error(
            std::string(WHO) + fieldPath + " names `waveDict " + name + "`. brae reads "
            "constant/waveProperties only.");
    }
}

template <class T>
void claimWavePatches(
    FieldData<T>& fd,
    const std::string& typeName,
    const std::vector<FvPatch>& patches,
    std::vector<char>& mark)
{
    for (auto& b : fd.boundary)
    {
        if (b.type != typeName) continue;
        bool found = false;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].name != b.name) continue;
            mark[pi] = 1;
            found = true;
        }
        if (!found)
            throw std::runtime_error(
                std::string(WHO) + "the `" + typeName + "` entry `" + b.name + "` names no patch. "
                "A wave condition is per patch -- its model is looked up by the patch's own name -- "
                "so a group or regex key is not resolved here.");
        if (!b.hasValue)
            throw std::runtime_error(
                std::string(WHO) + "the `" + typeName + "` entry `" + b.name + "` has no `value`. "
                "It is a fixedValue patch and OpenFOAM reads one.");
        // the factory builds the patch; this file is the only thing that ever assigns to it
        b.type = "fixedValue";
    }
}

std::shared_ptr<waveModels::WaveModel> lookupOrCreate(
    InterWaves& w,
    std::size_t pi,
    const std::vector<scalar>& alphaInternal,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!w.model[pi])
    {
        w.model[pi] = waveModels::WaveModel::New(w.waveProperties, patches[pi], m, g, w.gravity,
                                                 w.alphaName, alphaInternal);
    }
    return w.model[pi];
}

} // namespace


InterWaves readInterWaves(
    const std::string& caseDir,
    const std::string& startDir,
    FieldData<scalar>& alphaData,
    FieldData<vector>& UData,
    const std::vector<FvPatch>& patches,
    const vector& gravity,
    const std::string& alphaName)
{
    InterWaves w;
    w.alphaPatch.assign(patches.size(), 0);
    w.UPatch.assign(patches.size(), 0);
    w.model.resize(patches.size());
    w.gravity = gravity;
    w.alphaName = alphaName;
    claimWavePatches(alphaData, "waveAlpha", patches, w.alphaPatch);
    claimWavePatches(UData, "waveVelocity", patches, w.UPatch);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (w.alphaPatch[pi] || w.UPatch[pi])
        {
            w.any = true;
        }
    }
    if (!w.any) return w;

    refuseOtherWaveDict(startDir + "/" + alphaName);
    refuseOtherWaveDict(startDir + "/U");
    const std::string path = caseDir + "/constant/waveProperties";
    if (!std::filesystem::exists(path))
        throw std::runtime_error(
            std::string(WHO) + "the case has wave boundary conditions and no constant/waveProperties. "
            "OpenFOAM reads it MUST_READ when the first of them updates.");
    w.waveProperties = readDict(path);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!w.alphaPatch[pi] && !w.UPatch[pi]) continue;
        // waveModel is an IOdictionary at <startTime>/uniform/waveProperties.<patch>, READ_IF_PRESENT:
        // a restart takes the reference depth the first run stored there
        const std::string stored = startDir + "/uniform/waveProperties." + patches[pi].name;
        if (std::filesystem::exists(stored))
            throw std::runtime_error(
                std::string(WHO) + stored + " exists: this is a restart, and OpenFOAM re-reads the "
                "wave model's stored reference depth from it. brae does not.");
        if (!w.waveProperties.subDict(patches[pi].name))
            throw std::runtime_error(
                std::string(WHO) + "constant/waveProperties has no entry for patch `"
                + patches[pi].name + "`, which carries a wave boundary condition.");
    }
    return w;
}


void updateWaveAlpha(
    InterWaves& w,
    GeometricField<scalar>& alpha1,
    const GeometricField<vector>& U,
    scalar t,
    label timeIndex,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!w.any) return;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!w.alphaPatch[pi]) continue;
        const std::shared_ptr<waveModels::WaveModel> model =
            lookupOrCreate(w, pi, alpha1.internal, m, g, patches);
        if (model->correct(t, timeIndex, alpha1.internal, U.internal))
        {
            w.updateLog.push_back(patches[pi].name + "@" + std::to_string(timeIndex));
        }
        // operator==(model.alpha())
        alpha1.boundary[pi]->setStoredValues(model->alpha());
    }
}


void updateWaveVelocity(
    InterWaves& w,
    const GeometricField<scalar>& alpha1,
    GeometricField<vector>& U,
    scalar t,
    label timeIndex,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!w.any) return;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!w.UPatch[pi]) continue;
        const std::shared_ptr<waveModels::WaveModel> model =
            lookupOrCreate(w, pi, alpha1.internal, m, g, patches);
        if (model->correct(t, timeIndex, alpha1.internal, U.internal))
        {
            w.updateLog.push_back(patches[pi].name + "@" + std::to_string(timeIndex));
        }
        // operator==(model.U())
        U.boundary[pi]->setStoredValues(model->U());
    }
}


SubCycleClock subCycleClock(
    scalar tNew,
    scalar deltaT,
    label stepIndex,
    label nSubCycles,
    label k)
{
    SubCycleClock c;
    if (nSubCycles <= 1)
    {
        c.t = tNew;
        c.timeIndex = stepIndex;
        return c;
    }
    // Time::subCycle: setTime(*this - deltaT(), (timeIndex() - 1)*nSubCycles); deltaT_ /= nSubCycles
    scalar t = tNew - deltaT;
    const scalar sub = deltaT/static_cast<scalar>(nSubCycles);
    for (label i = 0; i < k; ++i)
    {
        t = t + sub;
    }
    c.t = t;
    c.timeIndex = (stepIndex - 1)*nSubCycles + k;
    return c;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
