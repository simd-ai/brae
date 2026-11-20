#pragma once
// purge_write.cuh -- controlDict's purgeWrite: keep only the last N time directories a run writes.
//
// Foam::Time::writeObject (TimeIO.C:557-582), after the objects of a write time are written:
//   if (writeTime_ && purgeWrite_)
//   {
//       if (previousWriteTimes_.empty() || previousWriteTimes_.top() != timeName())
//           previousWriteTimes_.push(timeName());
//       while (previousWriteTimes_.size() > purgeWrite_)
//           fileHandler().rmDir(fileHandler().filePath(objectRegistry::path(previousWriteTimes_.pop()), false));
//   }
// previousWriteTimes_ is a FIFOStack<word> (FIFOStack.H: top() the newest, pop() the oldest). It starts empty in every
// run, so the start time's directory -- 0, or the time a restart reads -- is never pushed and never removed, and a
// restart leaves the earlier run's directories alone. TimeIO.C:322-333 reads the entry; a negative value is warned
// about and taken as 0.
//
// Shared by every driver: WriteControl::recordWritten (write_control.cuh) and rhoCentralFoam's driver call it after
// each write.
#include <cstdio>
#include <deque>
#include <filesystem>
#include <string>
#include <system_error>

namespace brae {

class PurgeWrite
{
public:
    // n: controlDict's purgeWrite as read; a negative one is OpenFOAM's warning and 0 (TimeIO.C:324-332), said once
    explicit PurgeWrite(int n, bool warnNegative = false) : n_(n < 0 ? 0 : n)
    {
        if (n < 0 && warnNegative)
            std::fprintf(stderr, "brae NOTICE: invalid value for purgeWrite %d, should be >= 0, setting to 0 "
                                 "(as OpenFOAM, TimeIO.C:324-332)\n", n);
    }
    int keep() const { return n_; }

    // after a write of <caseDir>/<timeName>: remember it, and remove the oldest written directories beyond N
    void written(const std::string& caseDir, const std::string& timeName)
    {
        if (n_ <= 0) return;
        if (written_.empty() || written_.back() != timeName) written_.push_back(timeName);
        while (static_cast<int>(written_.size()) > n_)
        {
            std::error_code ec;
            std::filesystem::remove_all(caseDir + "/" + written_.front(), ec);
            written_.pop_front();
        }
    }

private:
    int n_;
    std::deque<std::string> written_;
};

}   // namespace brae
