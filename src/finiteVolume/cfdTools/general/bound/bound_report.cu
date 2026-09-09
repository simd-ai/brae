#include "bound_report.cuh"
#include <cstdio>

namespace brae {

// IOstream::defaultPrecision() is a static on OpenFOAM's side too, set once from controlDict and read
// by every Info line thereafter; this mirrors it rather than threading a precision through six drivers.
namespace {
int& reportPrecision()
{
    static int p = 6;   // IOstream's own default (IOstream.H, precision_ = 6)
    return p;
}
}


void setBoundReportPrecision(int writePrecision)
{
    // OpenFOAM applies the entry as given; a nonsensical one is the case's problem, not this line's.
    // Guarded only against a value printf cannot use.
    if (writePrecision > 0 && writePrecision <= 40) reportPrecision() = writePrecision;
}


void printBounding(
    const char* fieldName,
    scalar      minValue,
    scalar      maxValue,
    scalar      average)
{
    // bound.C:41-46 -- Info<< "bounding " << name << ", min: " << min << " max: " << max
    //                       << " average: " << avg << endl;
    // A comma after the name and nothing after the numbers, which is easy to get wrong from memory;
    // it is transcribed from a real OpenFOAM log, not from the source's operator<< chain.
    const int prec = reportPrecision();
    std::printf("bounding %s, min: %.*g max: %.*g average: %.*g\n",
                fieldName,
                prec, static_cast<double>(minValue),
                prec, static_cast<double>(maxValue),
                prec, static_cast<double>(average));
}

} // namespace brae
