#pragma once
// deriveCaseRefusals -- the CASE-derived refusal flags, factored out of the host harness so the CUDA
// harness derives the SAME ones. Before this existed, the device-twin guards (hasMRF / hasFvOptions /
// hasCoupledPatches on RhoStepInput, checked in rhoUEqn.cu, rhoEEqn.cu, rhoPEqn.cu, rhoPcEqn.cu) were
// set only by fail-proof arms: a case declaring MRFProperties or an fvOption ran the CUDA path with the
// term silently dropped, while the host arm refused the very same case.
//
// hasMRF is EXISTENCE of constant/MRFProperties -- OF reads it MUST_READ whenever the header is valid
// (IOMRFZoneList.C:37-63), so existence over-refuses only a present-but-empty file, the safe direction.
// The fvOptions walk mirrors OpenFOAM's: limitTemperature over all cells is an implemented CORRECTION
// (in.limitT), anything else refuses by name, and cpu::fvOptions::read's own `unsupported` marker
// covers the types the dict walk admits but the option reader does not implement.
#include "foam_dict.cuh"
#include "fvOptions_cpp.cuh"
#include "primitive_mesh.cuh"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace rhoSimple {

struct CaseRefusals
{
    bool   hasMRF = false;
    bool   hasFvOptions = false;          // an fvOption the case declares and brae does not implement
    std::string fvOptionUnsupported;      // its type name, for the refusal message
    bool   limitT = false;                // limitTemperature over all cells (implemented)
    scalar limitTmin = 0, limitTmax = 0;
    fvOptions::OptionList opts;           // the IMPLEMENTED options, BY VALUE -- the caller owns the
                                          // lifetime (a dangling in.fvOpts was the old static's risk)
};

inline CaseRefusals deriveCaseRefusals(const std::string& caseDir, const PrimitiveMesh& m)
{
    CaseRefusals cr;
    auto has = [&](const char* rel)
    {
        std::ifstream f((caseDir + "/" + rel).c_str());
        return f.good();
    };
    const std::string fvoPath = has("system/fvOptions") ? caseDir + "/system/fvOptions"
                              : (has("constant/fvOptions") ? caseDir + "/constant/fvOptions" : "");
    if (!fvoPath.empty())
    {
        const FoamDict fvo = readDict(fvoPath);
        bool anyOther = false;
        for (const auto& entry : fvo.subs)
        {
            const FoamDict* o = &entry.second;
            const std::string ty = o->wordOr("type", "");
            if (ty == "limitTemperature")
            {
                const std::string sel = o->wordOr("selectionMode", "all");
                if (sel != "all")
                    throw std::runtime_error(
                        "rhoSimpleFoam: limitTemperature with selectionMode '" + sel
                        + "'. brae applies it over all cells; a cell subset is a different option. "
                          "Refusing rather than limiting the wrong cells.");
                cr.limitT    = true;
                cr.limitTmin = o->scalarOr("min", 0.0);
                cr.limitTmax = o->scalarOr("max", 0.0);
                std::printf("  fvOption limitTemperature [%g, %g]\n",
                            (double)cr.limitTmin, (double)cr.limitTmax);
            }
            else if (!ty.empty())
            {
                anyOther = true;
            }
        }
        cr.hasFvOptions = anyOther;
    }
    {
        cr.opts = fvOptions::read(caseDir, m);
        const std::string bad = cr.opts.firstUnsupported();
        if (!bad.empty())
        {
            cr.hasFvOptions = true;
            cr.fvOptionUnsupported = bad;
            std::printf("  fvOptions: '%s' is not implemented -- the case will be refused\n",
                        bad.c_str());
            std::fflush(stdout);   // the refusal aborts; an unflushed buffer loses this line
        }
        else if (!cr.opts.empty())
        {
            // The dict walk above set hasFvOptions on ANY non-limitTemperature type; the option
            // reader implementing every one of them OVERRIDES that back to false -- dropping this
            // reset in the factoring refused rhoBoxDF's implemented DarcyForchheimer, caught by the
            // df gate on the first regression sweep.
            cr.hasFvOptions = false;
            std::printf("  fvOptions: %zu option(s), all implemented\n", cr.opts.options.size());
        }
    }
    cr.hasMRF = has("constant/MRFProperties");
    if (cr.hasFvOptions) std::printf("  the case declares fvOptions\n");
    if (cr.hasMRF)       std::printf("  the case declares MRFProperties\n");
    return cr;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
