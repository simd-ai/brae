#include "bound_report.cuh"
#include "io_precision.cuh"
#include <cstdio>

namespace brae {

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
    const int prec = ioPrecision();   // the case's writePrecision -- see io_precision.cuh
    std::printf("bounding %s, min: %.*g max: %.*g average: %.*g\n",
                fieldName,
                prec, static_cast<double>(minValue),
                prec, static_cast<double>(maxValue),
                prec, static_cast<double>(average));
}

} // namespace brae
