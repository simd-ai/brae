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
#include <vector>

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
    // The fvOptions DICT KEY, not the type: OpenFOAM's report line prints name_ (limitTemperature.C:203
    // `type() << "=" << name_`), so a case calling its option `limitT` prints `limitTemperature=limitT`
    // and one calling it `clamp` prints `limitTemperature=clamp`. Reporting the type twice would be a
    // line that never matches the oracle.
    std::string limitTname;
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
            // `active false` is not a comment-out: OpenFOAM still CONSTRUCTS the option and still reads
            // its dictionary, but fvOptionList's correct()/addSup() skip it (fvOption.C:72
            // active_(dict_.getOrDefault("active", true)); fvOptionListTemplates.C:386 gates
            // source.correct(field) on source.isActive()). This walk did not read the key, so a case
            // declaring `limitTemperature { active false; }` had its clamp APPLIED by brae and skipped
            // by OpenFOAM -- and, the other way, an inactive unimplemented option refused a case
            // OpenFOAM runs. fvOptions::read already reads it the same way (fvOptions_cpp.cu:125-126).
            const std::string act = o->wordOr("active", "true");
            if (act == "false" || act == "no" || act == "off" || act == "0") continue;
            if (ty == "limitTemperature")
            {
                const std::string sel = o->wordOr("selectionMode", "all");
                if (sel != "all")
                    throw std::runtime_error(
                        "rhoSimpleFoam: limitTemperature with selectionMode '" + sel
                        + "'. brae applies it over all cells; a cell subset is a different option. "
                          "Refusing rather than limiting the wrong cells.");
                cr.limitT     = true;
                cr.limitTname = entry.first;
                cr.limitTmin  = o->scalarOr("min", 0.0);
                cr.limitTmax  = o->scalarOr("max", 0.0);
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
        // limitTemperature is IMPLEMENTED by this driver, on both arms -- the host clamps he between
        // he(p,Tmin) and he(p,Tmax) after the energy solve (rhoSimpleFoam_cpp.cu, EEqn.H:28) and the
        // CUDA arm runs limitEnergyKernel (device_fvoptions.cu) for the same thing. It is resolved out
        // of the option list by the dict walk above, so fvOptions::read never builds an Option for it
        // and its catch-all (fvOptions_cpp.cu:202-206) marks the type unsupported. That mark was then
        // promoted straight back into hasFvOptions below, overriding the branch above and refusing
        // aerofoilNACA0012 -- the only rhoSimpleFoam tutorial whose sole blocker was a capability brae
        // already had. Measured: with system/fvOptions deleted the host arm runs all 10 iterations and
        // tracks OpenFOAM to ~1% on U, e, k and omega.
        //
        // The exemption is CONDITIONAL on the walk having accepted this case's option: an unsupported
        // selectionMode already threw above, so reaching here with cr.limitT set means brae will really
        // apply it. Every other driver passes nothing and still refuses limitTemperature by name.
        const std::string bad = cr.opts.firstUnsupported(
            cr.limitT ? std::vector<std::string>{"limitTemperature"} : std::vector<std::string>{});
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
