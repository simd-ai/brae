#pragma once
// fanPressure: read the fan curve and the patch's operating parameters out of the case.
//
// OF's fanPressureFvPatchScalarField::updateCoeffs computes the patch's volumetric flow rate, looks up the
// fan's pressure rise at that rate, and hands totalPressure a shifted p0:
//     dir         = -1 for `direction in`, +1 for `out`   (2*ffd - 1, ffdIn = 0)
//     volFlowRate = dir*sum(phi_patch)
//     p0_eff      = p0 - dir*fanCurve(max(volFlowRate, 0))
// The FACE treatment is then plain totalPressure, which brae already has, so all that is needed here is
// the curve and the sign.
//
// The curve arrives as a Function1 `table`, and the tutorial writes it as an external file
// (`fanCurve { type table; file "<constant>/FluxVsdP.dat"; }`). brae's Function1 builds inline tables
// only, so the file form is read here instead -- this runs in the DRIVER, which knows the case directory
// and can expand `<constant>`; the field parser does not and must not need to.
#include "foam_dict.cuh"
#include "fv_patch.cuh"
#include "device_simple_foam.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>

namespace brae {

// Parse "((x0 y0) (x1 y1) ...)" -- the body of an OF table, from a file or an inline entry. Anything that
// is not a number or a bracket is ignored, which is how OF's own token reader treats the comments these
// files usually carry.
inline std::vector<std::pair<scalar, scalar>> parseFanTable(const std::string& text)
{
    std::vector<scalar> nums;
    std::string t = text;
    for (char& c : t) if (c == '(' || c == ')' || c == ',' || c == ';') c = ' ';
    // strip // comments
    std::istringstream ls(t);
    std::string line, cleaned;
    while (std::getline(ls, line))
    {
        const std::size_t c = line.find("//");
        cleaned += (c == std::string::npos ? line : line.substr(0, c));
        cleaned += '\n';
    }
    std::istringstream is(cleaned);
    scalar v;
    while (is >> v) nums.push_back(v);
    std::vector<std::pair<scalar, scalar>> out;
    for (std::size_t i = 0; i + 1 < nums.size(); i += 2) out.push_back({nums[i], nums[i+1]});
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}

// One entry per patch whose p field says `type fanPressure`. timeDir is the directory the p field was
// read from, so a restart picks up the same boundary specification the run is actually using.
inline std::vector<DeviceSimpleSolver::FanPressure> collectFanPressure(
    const std::string& caseDir,
    const std::string& timeDir,
    const std::vector<FvPatch>& patches)
{
    std::vector<DeviceSimpleSolver::FanPressure> out;
    const std::string pPath = timeDir + "/p";
    { std::ifstream f(pPath); if (!f.good()) { std::ifstream g(pPath + ".gz"); if (!g.good()) return out; } }
    const FoamDict pd = readDict(pPath);
    const FoamDict* bf = pd.subDict("boundaryField");
    if (!bf) return out;

    label start = 0;
    for (const FvPatch& fp : patches)
    {
        if (isCyclicFamilyType(fp.type)) continue;   // DeviceBoundary order skips these
        const FoamDict* b = bf->subDict(fp.name);
        if (b && b->wordOr("type", "") == "fanPressure")
        {
            DeviceSimpleSolver::FanPressure e;
            e.start = start;
            e.count = fp.size;
            e.p0    = b->scalarOr("p0", 0.0);
            const std::string dirw = b->wordOr("direction", "in");
            if (dirw != "in" && dirw != "out")
                throw std::runtime_error("brae: fanPressure on patch " + fp.name + " has direction '" + dirw
                                         + "'; OpenFOAM allows only `in` or `out`.");
            e.dir = (dirw == "out") ? scalar(1) : scalar(-1);   // OF: 2*ffd - 1, ffdIn = 0
            if (b->wordOr("nonDimensional", "false") == "true")
                throw std::runtime_error(
                    "brae: fanPressure on patch " + fp.name + " is nonDimensional, which rescales the curve "
                    "by rpm and mean diameter (fanPressureFvPatchScalarField.C). Not implemented; the "
                    "dimensional form is.");
            const FoamDict* fc = b->subDict("fanCurve");
            if (!fc)
                throw std::runtime_error("brae: fanPressure on patch " + fp.name + " has no `fanCurve` entry.");
            const std::string ftype = fc->wordOr("type", "table");
            if (ftype != "table" && ftype != "tableFile")
                throw std::runtime_error("brae: fanPressure on patch " + fp.name + " has fanCurve type '"
                                         + ftype + "'; only `table` is implemented.");
            std::string file = fc->wordOr("file", "");
            if (!file.empty())
            {
                for (char& c : file) if (c == '"') c = ' ';
                const std::size_t a = file.find_first_not_of(" \t");
                const std::size_t z = file.find_last_not_of(" \t");
                file = (a == std::string::npos) ? "" : file.substr(a, z - a + 1);
                const std::string tag = "<constant>";
                const std::size_t k = file.find(tag);
                if (k != std::string::npos) file.replace(k, tag.size(), caseDir + "/constant");
                else if (!file.empty() && file[0] != '/') file = caseDir + "/" + file;
                std::ifstream tf(file);
                if (!tf) throw std::runtime_error("brae: fanPressure on patch " + fp.name
                                                  + " -- cannot read fanCurve file " + file);
                const std::string txt((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
                e.curve = parseFanTable(txt);
            }
            else
            {
                // inline `values ((0 20) (0.0023 10) ...)`: scalarListOr already returns every numeric
                // token in order, which is the (flowRate, deltaP) sequence.
                const std::vector<scalar> n = fc->scalarListOr("values", {});
                for (std::size_t i = 0; i + 1 < n.size(); i += 2) e.curve.push_back({n[i], n[i+1]});
                std::sort(e.curve.begin(), e.curve.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            if (e.curve.size() < 2)
                throw std::runtime_error("brae: fanPressure on patch " + fp.name
                                         + " -- the fan curve has fewer than two points, so there is nothing to "
                                           "interpolate. Check the `fanCurve` entry.");
            out.push_back(std::move(e));
        }
        start += fp.size;
    }
    return out;
}

// OF fixedMeanFvPatchField::updateCoeffs, as a pure function over ONE patch, so the solver and its test
// share a single copy of the formula:
//     mean = sum(magSf*psi)/sum(magSf)
//     |target| > SMALL and |mean| > 0.5*|target| : psi *= |target|/|mean|   (SCALE -- matches the
//                                                                           MAGNITUDE, keeps psi's sign)
//     otherwise                                  : psi += (target - mean)   (SHIFT)
// psi is the patchInternalField (the adjacent cell values); out may alias nothing.
inline void applyFixedMean(scalar target, const scalar* magSf, const scalar* psi, label n, scalar* out)
{
    scalar sA = 0, sV = 0;
    for (label i = 0; i < n; ++i) { sA += magSf[i]; sV += magSf[i]*psi[i]; }
    if (!(sA > scalar(0))) return;
    const scalar mean = sV / sA;
    const bool scaleIt = std::fabs(target) > scalar(1e-15)
                      && std::fabs(mean)   > scalar(0.5)*std::fabs(target);
    for (label i = 0; i < n; ++i)
        out[i] = scaleIt ? psi[i]*std::fabs(target)/std::fabs(mean) : psi[i] + (target - mean);
}

// fixedMean, same family: a boundary condition whose per-step value the DRIVER has to read out of the
// case, because the field parser has no case directory and no way to know which field it belongs to.
//
// brae recomputes it for the PRESSURE field only, which is where OF's tutorials use it. On any other
// field the patch would be built as a plain fixedValue and never updated -- a frozen boundary where the
// case asked for a maintained mean -- so that is refused rather than run.
inline std::vector<DeviceSimpleSolver::FixedMean> collectFixedMean(
    const std::string& timeDir,
    const std::vector<FvPatch>& patches,
    const std::vector<std::string>& otherFields)
{
    std::vector<DeviceSimpleSolver::FixedMean> out;
    for (const std::string& f : otherFields)
    {
        const std::string path = timeDir + "/" + f;
        { std::ifstream a(path); if (!a.good()) { std::ifstream g(path + ".gz"); if (!g.good()) continue; } }
        const FoamDict d = readDict(path);
        const FoamDict* bf = d.subDict("boundaryField");
        if (!bf) continue;
        for (const FvPatch& fp : patches)
        {
            const FoamDict* b = bf->subDict(fp.name);
            if (b && b->wordOr("type", "") == "fixedMean")
                throw std::runtime_error(
                    "brae: patch " + fp.name + " of field " + f + " is `fixedMean`, which brae maintains "
                    "for the pressure only. On any other field it would be built as a plain fixedValue "
                    "and frozen at its initial value, so the prescribed mean would silently not hold.");
        }
    }
    const std::string pPath = timeDir + "/p";
    { std::ifstream a(pPath); if (!a.good()) { std::ifstream g(pPath + ".gz"); if (!g.good()) return out; } }
    const FoamDict pd = readDict(pPath);
    const FoamDict* bf = pd.subDict("boundaryField");
    if (!bf) return out;
    label start = 0;
    for (const FvPatch& fp : patches)
    {
        if (isCyclicFamilyType(fp.type)) continue;   // DeviceBoundary order skips these
        const FoamDict* b = bf->subDict(fp.name);
        if (b && b->wordOr("type", "") == "fixedMean")
        {
            DeviceSimpleSolver::FixedMean e;
            e.start = start;
            e.count = fp.size;
            e.meanValue = b->scalarOr("meanValue", 0.0);
            out.push_back(e);
        }
        start += fp.size;
    }
    return out;
}

} // namespace brae
