#pragma once
// write_control.cuh -- OF's controlDict write cadence (writeControl / writeInterval / purgeWrite /
// deltaT / startTime / stopAt), ported from Foam::Time and shared by every driver.
//
// Found by dict_audit, not by reading OF's source: gpuRhoSimpleFoam never looked at ANY of these keys.
// The incompressible driver had the whole mechanism (12 references); the compressible one had zero and
// wrote exactly one time directory, at the end. So a compressible case asking to write every 100
// iterations got nothing until convergence -- no intermediate fields to inspect, no way to watch a run
// develop, and purgeWrite silently ignored -- while the run itself was perfectly correct. That is the
// house failure mode: the input was read off disk by nobody and the output still looked plausible.
//
// Only the POLICY lives here (when to write, what to call it, what to delete). The PAYLOAD -- which
// fields a given solver writes -- stays with the driver, because that is the part that genuinely differs.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "brae_notice.cuh"
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <string>

namespace brae {

class WriteControl
{
public:
    explicit WriteControl(const FoamDict& controlDict)
      : control_(controlDict.wordOr("writeControl", "timeStep")),
        // OF's default is GREAT, i.e. "never" -- only the final state gets written. Keeping that default
        // means a case with no writeInterval behaves exactly as before this class existed.
        interval_(controlDict.scalarOr("writeInterval", 1e30)),
        purge_(std::max(0, controlDict.intOr("purgeWrite", 0))),
        deltaT_(controlDict.scalarOr("deltaT", 1.0)),
        startTime_(controlDict.scalarOr("startTime", 0.0))
    {
        // OF stopAt: endTime | writeNow | noWriteNow | nextWrite. The last three are runtime-modifiable
        // stop requests (Foam::Time::stopAt), which brae has no mechanism for -- it does not re-read
        // controlDict mid-run. Say so rather than let the entry sit there looking honoured.
        const std::string stopAt = controlDict.wordOr("stopAt", "endTime");
        if (stopAt != "endTime")
            noticeIgnored("controlDict stopAt",
                          "'" + stopAt + "' -- brae does not re-read controlDict during a run, so only "
                          "'endTime' is meaningful; the run will go to endTime or residualControl");

        // OpenFOAM's writeFormat is a WRITE option ONLY: TimeIO.C:370-372 sets writeStreamOption_ and
        // nothing reads it back. A file's format is taken from its OWN FoamFile header
        // (IOobjectReadHeader.C:51, applied at :111), so an ascii time directory written under
        // `writeFormat binary` is read by real OpenFOAM without complaint -- measured on squareBend at
        // 112k, OF restarted from brae's ascii output with binary still set and ran to End. brae's
        // writer rewrites a binary template header to `format ascii` (foam_field_writer.cuh:241-242),
        // so the file is honestly labelled rather than lying about its contents.
        //
        // What IS lost is precision and size, and that is why this is a NOTICE and not silence: brae
        // writes 12 significant digits where binary is exact, and the file is several times larger on
        // the big meshes that ask for binary in the first place. It is not a REFUSAL because nothing
        // numerical is at stake and the output is readable by every OpenFOAM tool -- and because only
        // the rhoSimpleFoam mirror ever refused it, while simpleFoamV2, gpuSimpleFoam, gpuPimpleFoam
        // and the legacy rho drivers all wrote ascii under the same setting saying nothing at all.
        // One notice here covers every driver, since every one of them builds a WriteControl.
        if (controlDict.wordOr("writeFormat", "ascii") == "binary")
            noticeApproximated("controlDict writeFormat",
                               "'binary' -- brae's field writer emits ascii only. OpenFOAM reads the "
                               "format from each file's own header, not from controlDict, so the output "
                               "is readable; it is larger and carries 12 significant digits rather than "
                               "binary's exact bits");
    }

    scalar deltaT() const { return deltaT_; }
    scalar startTime() const { return startTime_; }

    // `startFrom latestTime` can resolve to a directory that is NOT controlDict's startTime, and every
    // time value this class produces is measured from the start. Without this, a run restarted from 10
    // with `startTime 0` still in the dict names its output 1, 2, 3... -- overwriting the case's early
    // history and losing the restart's place in the timeline. Call it with the RESOLVED start.
    void setStartTime(scalar t) { startTime_ = t; }
    scalar timeValue(int iter) const { return startTime_ + static_cast<scalar>(iter) * deltaT_; }

    // Integer name for whole times (deltaT = 1, the steady case), else %g -- OF's timeName formatting.
    static std::string timeName(scalar t)
    {
        if (t == std::floor(t) && std::fabs(static_cast<double>(t)) < 1e15)
            return std::to_string(static_cast<long long>(std::llround(static_cast<double>(t))));
        char b[64];
        std::snprintf(b, sizeof b, "%g", static_cast<double>(t));
        return std::string(b);
    }

    // Foam::Time::operator++ write switch. Not const: the runTime branch latches the interval index.
    bool isWriteTime(int iter, scalar tval)
    {
        // OF's absent-entry default is GREAT; casting that to long is undefined, and it means "never"
        // (Time.C: !(timeIndex % writeInterval) with an interval no index reaches).
        if (control_ == "timeStep")
            return interval_ >= 1 && interval_ < 1e18 && (iter % static_cast<long>(interval_)) == 0;
        if (control_ == "runTime" || control_ == "adjustable" || control_ == "adjustableRunTime")
        {
            const long wi = static_cast<long>(((tval - startTime_) + 0.5 * deltaT_) / interval_);
            if (wi > writeTimeIndex_) { writeTimeIndex_ = wi; return true; }
            return false;
        }
        return false;   // none / clockTime / cpuTime: only the final state, written after the loop
    }

    // Foam::Time TimeIO purgeWrite: keep only the last N time directories. Call after each write.
    void recordWritten(const std::string& caseDir, const std::string& tname)
    {
        if (purge_ <= 0) return;
        if (written_.empty() || written_.back() != tname) written_.push_back(tname);
        while (static_cast<int>(written_.size()) > purge_)
        {
            std::error_code ec;
            std::filesystem::remove_all(caseDir + "/" + written_.front(), ec);
            written_.pop_front();
        }
    }

private:
    std::string control_;
    scalar interval_;
    int purge_;
    scalar deltaT_;
    scalar startTime_;
    long writeTimeIndex_ = 0;
    std::deque<std::string> written_;
};

}   // namespace brae
