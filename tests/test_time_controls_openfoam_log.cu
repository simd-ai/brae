// THE TIME-STEP CONTROL AGAINST OPENFOAM'S OWN LOG, one step a row.
//
// interFoam prints, at the top of every step, the two Courant numbers it has just computed and then the deltaT
// setDeltaT.H chose from them (CourantNo.H, alphaCourantNo.H, VoF/setDeltaT.H). At writePrecision 17 each of
// those is the double OpenFOAM held. So a row of that log -- the deltaT before, the two numbers, the deltaT
// after -- is the formula's input and its output, with nothing of the flow in between: brae's setDeltaTVoF is
// handed the first three and must return the fourth TO THE BIT. No flux, no solver tolerance, no rounding of a
// Courant number enters, which is what lets a constant that moves deltaT by a few ulp be held.
//
// The rows come from tests/interfoam_write/clock/deltat_openfoam_log.sh, which stages the run and reads this
// program's RESULT line; the control (BRAE_CONTROL_DELTAT_SMALL_FLOAT=1) is run through it too.
//
// usage: test_time_controls_openfoam_log <rows file> <maxCo> <maxAlphaCo> <maxDeltaT>
//        a row: <deltaT before> <Courant number> <interface Courant number> <deltaT after>
#include "time_controls.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

using namespace brae;

namespace {

struct Row
{
    scalar before;
    scalar co;
    scalar alphaCo;
    scalar after;
};

bool sameBits(
    scalar a,
    scalar b)
{
    return std::memcmp(&a, &b, sizeof(scalar)) == 0;
}

} // namespace

int main(
    int argc,
    char** argv)
{
    if (argc != 5)
    {
        std::printf("usage: %s <rows file> <maxCo> <maxAlphaCo> <maxDeltaT>\n", argv[0]);
        return 2;
    }
    std::vector<Row> rows;
    {
        std::ifstream in(argv[1]);
        Row r;
        std::string a;
        std::string b;
        std::string c;
        std::string d;
        while (in >> a >> b >> c >> d)
        {
            r.before = std::strtod(a.c_str(), nullptr);
            r.co = std::strtod(b.c_str(), nullptr);
            r.alphaCo = std::strtod(c.c_str(), nullptr);
            r.after = std::strtod(d.c_str(), nullptr);
            rows.push_back(r);
        }
    }
    VoFTimeControls tc;
    tc.base.adjustTimeStep = true;
    tc.base.maxCo = std::strtod(argv[2], nullptr);
    tc.maxAlphaCo = std::strtod(argv[3], nullptr);
    tc.base.maxDeltaT = std::strtod(argv[4], nullptr);

    // which term of setDeltaT.H decides a row -- from OpenFOAM's own expression, written out here so that the
    // sorting does not lean on the function under test:
    //   maxDeltaTFact = min(maxCo/(CoNum + SMALL), maxAlphaCo/(alphaCoNum + SMALL))
    //   deltaTFact = min(min(maxDeltaTFact, 1 + 0.1*maxDeltaTFact), 1.2)
    // (the damped term as this machine's OpenFOAM computes it: a fused multiply-add -- see time_controls.cuh)
    const scalar ofSmall = 1.0e-15;
    long capped = 0;
    long damped = 0;
    long byCo = 0;
    long byAlphaCo = 0;
    long byMaxDeltaT = 0;
    long vofWrong = 0;
    long vofWrongOfLimited = 0;
    scalar worst = 0;
    long worstRow = -1;
    long generalRows = 0;
    long generalWrong = 0;
    for (std::size_t k = 0; k < rows.size(); ++k)
    {
        const Row& r = rows[k];
        const scalar fCo = tc.base.maxCo/(r.co + ofSmall);
        const scalar fAlpha = tc.maxAlphaCo/(r.alphaCo + ofSmall);
        const scalar f = std::min(fCo, fAlpha);
        const scalar dampedTerm = std::fma(scalar(0.1), f, scalar(1));
        const bool limited = f < dampedTerm;
        if (std::min(std::min(f, dampedTerm), scalar(1.2))*r.before > tc.base.maxDeltaT)
        {
            ++byMaxDeltaT;
        }
        else if (limited)
        {
            ++(fCo <= fAlpha ? byCo : byAlphaCo);
        }
        else if (dampedTerm < scalar(1.2))
        {
            ++damped;
        }
        else
        {
            ++capped;
        }
        const scalar got = setDeltaTVoF(r.before, r.co, r.alphaCo, tc);
        if (!sameBits(got, r.after))
        {
            ++vofWrong;
            vofWrongOfLimited += limited ? 1 : 0;
            const scalar gap = std::fabs(got - r.after)/std::fabs(r.after);
            if (gap > worst)
            {
                worst = gap;
                worstRow = static_cast<long>(k);
            }
        }
        // setDeltaT.H of cfdTools is the same expression without the interface's term: on a row where maxCo's
        // branch is the smaller of the two, OpenFOAM's result is that function's too
        if (fCo <= fAlpha)
        {
            ++generalRows;
            generalWrong += sameBits(setDeltaT(r.before, r.co, tc.base), r.after) ? 0 : 1;
        }
    }
    // setInitialDeltaT.H, from its source and not from a run:
    //   if (timeIndex == 0 && CoNum > SMALL) deltaT = min(maxCo*deltaT/CoNum, min(deltaT, maxDeltaT))
    // Four cases worked by hand in numbers a double holds exactly, one for each term: a Courant number between
    // the two constants this file once confused leaves deltaT alone (even above maxDeltaT); one above maxCo
    // cuts it, 0.5*0.25/2; one below maxCo leaves it; and maxDeltaT below deltaT caps it.
    TimeControls first;
    first.adjustTimeStep = true;
    first.maxCo = 0.5;
    first.maxDeltaT = 1.0;
    const bool initialKept = sameBits(setInitialDeltaT(scalar(2), scalar(1.0e-20), first), scalar(2));
    bool initialSet = sameBits(setInitialDeltaT(scalar(0.25), scalar(2), first), scalar(0.0625));
    initialSet = initialSet && sameBits(setInitialDeltaT(scalar(0.25), scalar(0.25), first), scalar(0.25));
    first.maxDeltaT = 0.125;
    initialSet = initialSet && sameBits(setInitialDeltaT(scalar(0.25), scalar(0.25), first), scalar(0.125));
    const bool greatDefault = sameBits(TimeControls().maxDeltaT, scalar(1.0e+15));

    std::printf("  %zu steps of OpenFOAM's log: %ld limited by the Courant number, %ld by the interface's, %ld "
                "damped (1 + 0.1 f), %ld at the 1.2 cap, %ld at maxDeltaT\n", rows.size(), byCo, byAlphaCo,
                damped, capped, byMaxDeltaT);
    std::printf("  setDeltaTVoF: %ld of them not OpenFOAM's bits (%ld of the limited ones), the worst %.3e "
                "relative at step %ld\n", vofWrong, vofWrongOfLimited, (double)worst, worstRow + 1);
    std::printf("  setDeltaT: %ld of the %ld steps maxCo's branch decides not OpenFOAM's bits\n", generalWrong,
                generalRows);
    std::printf("  setInitialDeltaT: a Courant number of 1e-20 leaves deltaT %s; its three worked cases %s; "
                "maxDeltaT defaults to GREAT = 1e15: %s\n", initialKept ? "alone" : "CHANGED",
                initialSet ? "hold" : "FAIL", greatDefault ? "yes" : "NO");
    std::printf("RESULT rows=%zu byCo=%ld byAlphaCo=%ld damped=%ld capped=%ld byMaxDeltaT=%ld vofWrong=%ld "
                "vofWrongLimited=%ld worst=%.3e generalRows=%ld generalWrong=%ld initialKept=%s initialSet=%s "
                "great=%s\n", rows.size(), byCo, byAlphaCo, damped, capped, byMaxDeltaT, vofWrong,
                vofWrongOfLimited, (double)worst, generalRows, generalWrong, initialKept ? "ok" : "no",
                initialSet ? "ok" : "no", greatDefault ? "ok" : "no");
    return 0;
}
